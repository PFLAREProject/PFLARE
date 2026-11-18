// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>

// The definition of the device copy of the cf markers on a given level 
// is stored in Device_Datak.kokkos.cxx and imported as extern from 
// kokkos_helper.hpp

//------------------------------------------------------------------------------------------------------------------------

// Rows of the local (or nonlocal) block with more entries than this go to ddc_ratio_long_rows
#define DDC_LONG_ROW 64
// Entries of a long row loaded into scratch at a time, and the threads per team that load them
#define DDC_LONG_CHUNK 1024
#define DDC_LONG_TEAM 128
// With at least this many fine rows the long rows are listed (which needs a host sync for the
// count) and only they get a team; with fewer every fine row gets a team and the short ones
// return at once, which costs less than the sync
#define DDC_LIST_MIN_ROWS 65536

// One team per fine row with more than DDC_LONG_ROW entries in the block (i_d, j_d, vals_d), the
// rest are left to the one thread per row kernels. With use_list the rows are long_rows_d[0, n)
// (indices into is_fine_local_d, listed by the one thread per row kernels), otherwise the teams
// are the n fine rows and the short ones return. Adds the |val| of the block's fine
// off-diagonal entries to diag_dom_ratio_d in the row's order, starting from 0 for the local
// block (which also writes diag_entry_d) and from the local sum for the nonlocal block
// The team loads a chunk of the row into scratch at a time, |val| for an entry that counts and 0
// for the rest, and one thread adds the chunk in order; adding 0 to the non-negative sum leaves
// it unchanged, so the sum is bitwise that of the one thread per row loop
static void ddc_ratio_long_rows(const bool use_list, const PetscIntKokkosView &long_rows_d, const PetscInt n, const PetscIntKokkosView &is_fine_local_d, \
                  const PetscInt *i_d, const PetscInt *j_d, const PetscScalar *vals_d, const intKokkosView &cf_d, \
                  const bool local, const PetscScalarKokkosView &diag_entry_d, const PetscScalarKokkosView &diag_dom_ratio_d)
{
   if (n == 0) return;
   auto exec = PetscGetKokkosExecutionSpace();
   const size_t scratch = ScratchScalarView::shmem_size(DDC_LONG_CHUNK) + ScratchScalarView::shmem_size(1);
   Kokkos::TeamPolicy<> probe(exec, 1, 1);
   probe.set_scratch_size(0, Kokkos::PerTeam(scratch));
   auto body = KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {
      const PetscInt i_idx_is_row = use_list ? long_rows_d(t.league_rank()) : t.league_rank();
      const PetscInt i = is_fine_local_d(i_idx_is_row);
      const PetscInt row_start = i_d[i], row_end = i_d[i + 1];
      if (row_end - row_start <= DDC_LONG_ROW) return;

      ScratchScalarView vals(t.team_scratch(0), DDC_LONG_CHUNK);
      // The diagonal, found by whichever thread loads it
      ScratchScalarView diag(t.team_scratch(0), 1);
      Kokkos::single(Kokkos::PerTeam(t), [&]() { diag(0) = 0.0; });
      PetscScalar sum_val = local ? 0.0 : diag_dom_ratio_d(i_idx_is_row);

      for (PetscInt chunk = row_start; chunk < row_end; chunk += DDC_LONG_CHUNK)
      {
         const PetscInt chunk_len = row_end - chunk < DDC_LONG_CHUNK ? row_end - chunk : DDC_LONG_CHUNK;
         t.team_barrier();
         Kokkos::parallel_for(Kokkos::TeamThreadRange(t, chunk_len), [&](const PetscInt c) {
            const PetscInt k = chunk + c;
            const PetscInt target_col = j_d[k];
            PetscScalar val = 0.0;
            // Is this column fine? F_POINT == -1
            if (cf_d(target_col) == -1)
            {
               if (local && target_col == i) diag(0) = Kokkos::abs(vals_d[k]);
               else val = Kokkos::abs(vals_d[k]);
            }
            vals(c) = val;
         });
         t.team_barrier();
         Kokkos::single(Kokkos::PerTeam(t), [&](PetscScalar &s) {
            for (PetscInt c = 0; c < chunk_len; c++) s += vals(c);
         }, sum_val);
      }
      t.team_barrier();
      Kokkos::single(Kokkos::PerTeam(t), [&]() {
         if (local) diag_entry_d(i_idx_is_row) = diag(0);
         diag_dom_ratio_d(i_idx_is_row) = sum_val;
      });
   };
   const int team_size = std::min(DDC_LONG_TEAM, probe.team_size_max(body, Kokkos::ParallelForTag()));
   Kokkos::TeamPolicy<> policy(exec, static_cast<int>(n), team_size);
   policy.set_scratch_size(0, Kokkos::PerTeam(scratch));
   Kokkos::parallel_for(policy, body);
}

//------------------------------------------------------------------------------------------------------------------------

// Computes the diagonal dominance ratio of the input matrix over fine points in global variable cf_markers_local_d
// This code is very similar to MatCreateSubMatrix_kokkos
PETSC_INTERN void MatDiagDomRatio_kokkos(Mat *input_mat, PetscReal *max_dd_ratio_achieved, PetscInt *local_rows_aff)
{
   // The matrix's blocks on the device (a Kokkos AIJ matrix or a raw shell)
   AIJDeviceAccess acc;
   aij_device_access_get(*input_mat, acc);
   const bool mpi = acc.mpi;
   MPI_Comm MPI_COMM_MATRIX = acc.comm;
   // The matrix's own mult SF (the borrowed mvctx) drives the halo broadcast
   PetscSF sf = acc.sf;
   (void)MPI_COMM_MATRIX;

   PetscInt cols_ao = acc.cols_ao;

   // Can't use the global directly within the parallel 
   // regions on the device
   intKokkosView cf_markers_d = cf_markers_local_d;   
   intKokkosView cf_markers_nonlocal_d;
   PetscIntKokkosView is_fine_local_d;
   auto exec = PetscGetKokkosExecutionSpace();

   // ~~~~~~~~~~~~
   // Get the F point local indices from cf_markers_local_d
   // ~~~~~~~~~~~~
   const int match_cf = -1; // F_POINT == -1
   create_cf_is_device_kokkos(input_mat, match_cf, is_fine_local_d);
   PetscInt local_rows_row = is_fine_local_d.extent(0);
   *local_rows_aff = local_rows_row;

   // Create device memory for the diag_dom_ratio
   diag_dom_ratio_local_d = PetscScalarKokkosView("diag_dom_ratio_local_d", local_rows_row);
   PetscScalarKokkosView diag_dom_ratio_d = diag_dom_ratio_local_d;

   // ~~~~~~~~~~~~~~~
   // Can now go and compute the diagonal dominance sums
   // ~~~~~~~~~~~~~~~

   // The off-diagonal component requires some comms which we can start now
   if (mpi)
   {
      cf_markers_nonlocal_d = intKokkosView("cf_markers_nonlocal_d", cols_ao);


      // Start the broadcast of the cf splitting, then overlap with local-only work below.
      // The local work only reads the root cf_markers_d, which is allowed while the
      // broadcast is in flight. Nothing may touch the leaf until End.
      // Be careful these aren't petscints
      PetscCallVoid(sf_sync_before_begin_kokkos());
      PetscCallVoid(PetscSFBcastWithMemTypeBegin(sf, MPI_INT,
                  PETSC_MEMTYPE_KOKKOS, cf_markers_d.data(),
                  PETSC_MEMTYPE_KOKKOS, cf_markers_nonlocal_d.data(),
                  MPI_REPLACE));
   }

   // ~~~~~~~~~~~~~~~
   // Do the local component so work/comms are overlapped
   // ~~~~~~~~~~~~~~~

   // ~~~~~~~~~~~~
   // Get pointers to the local i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = acc.local_i, *device_local_j = acc.local_j;
   const PetscScalar *device_local_vals = acc.local_vals.data();

   // Have to store the diagonal entry, every entry is written by the local kernel
   PetscScalarKokkosView diag_entry_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "diag_entry_d"), local_rows_row);

   // Get the diagonal entry of each fine row, while also summing up the
   // local non-diagonal fine entries into diag_dom_ratio_d
   // Each row is summed in the row's order into a single accumulator (continued by the
   // nonlocal kernels below), which is exactly the order of MatDiagDomRatio_cpu, so the
   // device ratios are bitwise the same as the CPU ones
   // The DDC compares these ratios against bin edges and many sit exactly on one,
   // so a different summation order would change the CF splitting
   // Rows with at most DDC_LONG_ROW entries: one thread per row, which on big matrices lists the
   // longer ones (in any order, each is summed on its own)
   const bool use_list = local_rows_row >= DDC_LIST_MIN_ROWS;
   PetscIntKokkosView long_rows_d;
   Kokkos::View<PetscInt, Kokkos::DefaultExecutionSpace::memory_space> n_long_d;
   PetscInt n_long = local_rows_row;
   if (use_list)
   {
      long_rows_d = PetscIntKokkosView(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "ddc_long_rows_d"), local_rows_row);
      n_long_d = Kokkos::View<PetscInt, Kokkos::DefaultExecutionSpace::memory_space>(
         Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "ddc_n_long_d"));
      Kokkos::deep_copy(exec, n_long_d, 0);
   }
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, local_rows_row), KOKKOS_LAMBDA(const PetscInt i_idx_is_row) {

         const PetscInt i = is_fine_local_d(i_idx_is_row);
         if (device_local_i[i + 1] - device_local_i[i] > DDC_LONG_ROW)
         {
            if (use_list) long_rows_d(Kokkos::atomic_fetch_add(&n_long_d(), PetscInt(1))) = i_idx_is_row;
            return;
         }
         PetscScalar sum_val = 0.0, diag_val = 0.0;
         for (PetscInt k = device_local_i[i]; k < device_local_i[i + 1]; k++)
         {
            const PetscInt target_col = device_local_j[k];
            // Is this column fine? F_POINT == -1
            if (cf_markers_d(target_col) == -1)
            {
               const PetscScalar val = Kokkos::abs(device_local_vals[k]);
               if (target_col == i) diag_val = val;
               else sum_val += val;
            }
         }
         diag_entry_d(i_idx_is_row) = diag_val;
         diag_dom_ratio_d(i_idx_is_row) = sum_val;
   });
   // Longer rows: one team per row. The team loads a chunk of the row into scratch, |val| for
   // each fine off-diagonal entry and 0 for the rest, then one thread adds the chunk in order
   // Adding 0 to the (non-negative) sum leaves it unchanged, so this is the same sum as above
   if (use_list)
   {
      Kokkos::deep_copy(exec, n_long, n_long_d);
      exec.fence();
   }
   ddc_ratio_long_rows(use_list, long_rows_d, n_long, is_fine_local_d, device_local_i, device_local_j, device_local_vals,
                       cf_markers_d, true, diag_entry_d, diag_dom_ratio_d);

   // Finish the in-flight broadcast and only then read from the leaf
   if (mpi)
   {
      PetscCallVoid(PetscSFBcastEnd(sf, MPI_INT, cf_markers_d.data(), cf_markers_nonlocal_d.data(), MPI_REPLACE));
      PetscCallVoid(sf_sync_after_end_kokkos());
   }

   // ~~~~~~~~~~~~~~~
   // Finish the comms and add the non-local entries to diag_dom_ratio_d
   // before we divide by the diagonal entry
   // ~~~~~~~~~~~~~~~

   // The off-diagonal component requires some comms
   // Basically a copy of MatCreateSubMatrix_MPIAIJ_SameRowColDist
   if (mpi)
   {
      // ~~~~~~~~~~~~
      // Get pointers to the nonlocal i,j,vals on the device
      // ~~~~~~~~~~~~
      const PetscInt *device_nonlocal_i = acc.nonlocal_i, *device_nonlocal_j = acc.nonlocal_j;
      const PetscScalar *device_nonlocal_vals = acc.nonlocal_vals.data();

      // Sum up the nonlocal matching entries into diag_dom_ratio_d
      // Nothing to do if this rank has no off-process columns
      if (cols_ao > 0)
      {
         if (use_list) Kokkos::deep_copy(exec, n_long_d, 0);
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(exec, 0, local_rows_row), KOKKOS_LAMBDA(const PetscInt i_idx_is_row) {

               const PetscInt i = is_fine_local_d(i_idx_is_row);
               // Most rows have no nonlocal entries
               const PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
               if (ncols_nonlocal == 0) return;
               if (ncols_nonlocal > DDC_LONG_ROW)
               {
                  if (use_list) long_rows_d(Kokkos::atomic_fetch_add(&n_long_d(), PetscInt(1))) = i_idx_is_row;
                  return;
               }

               // Carry on from the local sum, as the CPU does
               PetscScalar sum_val = diag_dom_ratio_d(i_idx_is_row);
               for (PetscInt k = device_nonlocal_i[i]; k < device_nonlocal_i[i + 1]; k++)
               {
                  // Is this nonlocal column fine? F_POINT == -1
                  if (cf_markers_nonlocal_d(device_nonlocal_j[k]) == -1) sum_val += Kokkos::abs(device_nonlocal_vals[k]);
               }
               diag_dom_ratio_d(i_idx_is_row) = sum_val;
         });
         if (use_list)
         {
            Kokkos::deep_copy(exec, n_long, n_long_d);
            exec.fence();
         }
         ddc_ratio_long_rows(use_list, long_rows_d, n_long, is_fine_local_d, device_nonlocal_i, device_nonlocal_j, device_nonlocal_vals,
                             cf_markers_nonlocal_d, false, diag_entry_d, diag_dom_ratio_d);
      }
   }

   // ~~~~~~~~~~~~~
   // Compute the diag dominance ratio
   // ~~~~~~~~~~~~~
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, local_rows_row), KOKKOS_LAMBDA(PetscInt i) {

      // If diag_val is zero we didn't find a diagonal
      if (diag_entry_d(i) != 0.0){
         // Compute the diagonal dominance ratio
         diag_dom_ratio_d(i) = diag_dom_ratio_d(i) / diag_entry_d(i);
      }
      else{
         diag_dom_ratio_d(i) = 0.0;
      }
   });
   // Ensure we're done before we exit
   Kokkos::fence();

   // Kokkos::Max over an empty range returns the reducer identity (lowest representable value),
   // so only reduce if we have local F points (matches the CPU which uses zero if there are none)
   PetscReal max_dd_ratio_local = 0.0;
   if (local_rows_row > 0) {
      Kokkos::parallel_reduce("max_dd_ratio", Kokkos::RangePolicy<>(exec, 0, local_rows_row),
         KOKKOS_LAMBDA(const PetscInt i, PetscReal& thread_max) {
            PetscReal dd_ratio = diag_dom_ratio_d(i);
            thread_max = (dd_ratio > thread_max) ? dd_ratio : thread_max;
         },
         Kokkos::Max<PetscReal>(max_dd_ratio_local)
      );
   }

   PetscCallMPIAbort(MPI_COMM_MATRIX, MPI_Allreduce(&max_dd_ratio_local, max_dd_ratio_achieved, 1, MPIU_REAL, MPI_MAX, MPI_COMM_MATRIX));

   exec.fence();
   aij_device_access_restore(acc);
   return;
}

//------------------------------------------------------------------------------------------------------------------------