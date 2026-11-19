// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>

// The definition of the device copy of the cf markers on a given level
// is stored in Device_Datak.kokkos.cxx and imported as extern from
// kokkos_helper.hpp

//------------------------------------------------------------------------------------------------------------------------

// The strength test of remove_small_from_sparse_kokkos, which keeps an entry if abs(val) >= tol
// Every test below goes through this so a NaN is treated the same everywhere
KOKKOS_INLINE_FUNCTION bool pmisr_is_strong(const PetscScalar val, const PetscScalar tol)
{
   return Kokkos::abs(val) >= tol;
}

// Runs row_body(t, i) once for every row i in [0, nrows), one thread per row with vector lanes
// for the row's entries (the team-vector pattern of DropSmallRowsFunctor): the vector length is the
// largest power of 2 no bigger than avg_row_nnz (at most 32), and each team takes team_size rows
// One thread per row on its own is slow on long rows, one team per row on short ones
template <class RowBody>
static void pmisr_rows_team_vector(const PetscInt nrows, const double avg_row_nnz, const RowBody &row_body)
{
   if (nrows == 0) return;
   auto exec = PetscGetKokkosExecutionSpace();
   int vector_length = 1;
   while (2 * vector_length <= avg_row_nnz && vector_length < 32) vector_length *= 2;
   vector_length = std::min(vector_length, Kokkos::TeamPolicy<>::vector_length_max());
   auto body = KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {
      const PetscInt row_start = static_cast<PetscInt>(t.league_rank()) * t.team_size();
      const PetscInt row_end = row_start + t.team_size() < nrows ? row_start + t.team_size() : nrows;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, row_start, row_end), [&](const PetscInt i) { row_body(t, i); });
   };
   Kokkos::TeamPolicy<> probe(exec, 1, 1, vector_length);
   int team_size = std::max(1, 256 / vector_length);
   team_size = std::min(team_size, probe.team_size_max(body, Kokkos::ParallelForTag()));
   const int league_size = static_cast<int>((nrows + team_size - 1) / team_size);
   Kokkos::parallel_for(Kokkos::TeamPolicy<>(exec, league_size, team_size, vector_length), body);
}

// The Luby loop of the device PMISR, straight from the blocks of a square matrix. The edges are
// the off-diagonal entries of the matrix that pass the edge test: strong compared to row_tol_d
// (as remove_small_from_sparse_kokkos), or every stored entry if row_tol_d is empty
// Undirected, the graph is the edges in both directions (each edge is handled from whichever row
// stores it, so no transpose is needed). Directed, a node only looks at the entries of its own
// row, as a PMISR on that matrix does: it is veto'd only by the neighbours in its row, and when
// it enters the set only those neighbours become C
// Only the nodes with a zero cf_markers_d take part; any other marker only keeps its sign (all
// markers are made -1 or 1 at the end) and its edges are ignored. Undecided nodes with
// abs(measure) < 1 get isolated_marker, then each step puts the undecided nodes that have the
// smallest measure among their undecided neighbours (a tie is broken on the global index,
// larger loses) in the set as F, and their undecided neighbours become C. With a positive
// max_luby_steps the nodes still undecided after that many steps become C
// colmap_d is the device copy of colmap (only read in parallel)
static void pmisr_luby_kokkos(const AIJDeviceAccess &acc, const PetscIntKokkosView &colmap_d, const PetscScalarKokkosView &row_tol_d, \
                  const bool directed, const int isolated_marker, \
                  const int max_luby_steps, const PetscScalarKokkosView &measure_local_d, const intKokkosView &cf_markers_d)
{
   const bool mpi = acc.mpi;
   PetscSF sf = acc.sf;
   MPI_Comm MPI_COMM_MATRIX = acc.comm;
   const PetscInt local_rows = acc.local_rows;
   const PetscInt cols_ao = acc.cols_ao;
   const PetscInt global_row_start = acc.row_start;
   const PetscInt *device_local_i = acc.local_i, *device_local_j = acc.local_j;
   const PetscInt *device_nonlocal_i = acc.nonlocal_i, *device_nonlocal_j = acc.nonlocal_j;
   Kokkos::View<const PetscScalar *> device_local_vals = acc.local_vals, device_nonlocal_vals = acc.nonlocal_vals;
   const double avg_nnz_local = local_rows > 0 ? static_cast<double>(acc.nnz_local) / local_rows : 0.0;
   const double avg_nnz_nonlocal = local_rows > 0 && mpi ? static_cast<double>(acc.nnz_nonlocal) / local_rows : 0.0;
   // Is every stored entry an edge
   const bool all_entries = row_tol_d.extent(0) == 0;

   auto exec = PetscGetKokkosExecutionSpace();

   intKokkosView veto_d("veto_d", local_rows);
   intKokkosView cf_markers_nonlocal_d, veto_nonlocal_d;
   PetscScalarKokkosView measure_nonlocal_d;
   if (mpi) {
      measure_nonlocal_d = PetscScalarKokkosView("measure_nonlocal_d", cols_ao);
      cf_markers_nonlocal_d = intKokkosView("cf_markers_nonlocal_d", cols_ao);
      veto_nonlocal_d = intKokkosView("veto_nonlocal_d", cols_ao);
   }

   // The measure is only read by everything below, so this overlaps with the initial count
   if (mpi)
   {
      PetscCallVoid(sf_sync_before_begin_kokkos());
      PetscCallVoid(PetscSFBcastWithMemTypeBegin(sf, MPIU_SCALAR,
                  PETSC_MEMTYPE_KOKKOS, measure_local_d.data(),
                  PETSC_MEMTYPE_KOKKOS, measure_nonlocal_d.data(),
                  MPI_REPLACE));
   }

   // Undecided nodes with abs(measure) < 1 (in the first pass, those with no strong neighbours)
   // are assigned straight away
   PetscInt counter_in_set_start = 0;
   Kokkos::parallel_reduce("Reduction", Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i, PetscInt &update) {
      if (cf_markers_d(i) != 0)
      {
         update++;
      }
      else if (Kokkos::abs(measure_local_d[i]) < 1)
      {
         cf_markers_d(i) = isolated_marker;
         update++;
      }
   }, counter_in_set_start);

   if (mpi)
   {
      PetscCallVoid(PetscSFBcastEnd(sf, MPIU_SCALAR, measure_local_d.data(), measure_nonlocal_d.data(), MPI_REPLACE));
      PetscCallVoid(sf_sync_after_end_kokkos());
   }

   PetscInt counter_undecided, counter_parallel;
   if (max_luby_steps < 0) {
      counter_undecided = local_rows - counter_in_set_start;
      PetscCallMPIAbort(MPI_COMM_MATRIX, MPI_Allreduce(&counter_undecided, &counter_parallel, 1, MPIU_INT, MPI_SUM, MPI_COMM_MATRIX));
      counter_undecided = counter_parallel;
   }
   else {
      // With fixed steps we still enter the loop for the collective scatters on empty ranks
      counter_undecided = 1;
   }

   // ~~~~~~~~~~~~
   // The outer Luby loop. Each edge of the graph is seen from the row that stores it, which
   // vetoes whichever of its two end points loses. Larger measure loses, with a tie broken on
   // the global index
   // ~~~~~~~~~~~~
   int loops_through = -1;
   do
   {
      if (counter_undecided == 0) break;
      if (max_luby_steps > 0 && max_luby_steps + 1 == -loops_through) break;

      // Reading cf_markers_d during the broadcast is fine, it isn't written before the End
      if (mpi) {
         PetscCallVoid(sf_sync_before_begin_kokkos());
         PetscCallVoid(PetscSFBcastWithMemTypeBegin(sf, MPI_INT,
                     PETSC_MEMTYPE_KOKKOS, cf_markers_d.data(),
                     PETSC_MEMTYPE_KOKKOS, cf_markers_nonlocal_d.data(),
                     MPI_REPLACE));
      }

      // Anything assigned already can't be in the set
      Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
         veto_d(i) = (cf_markers_d(i) != 0);
      });

      // Local edges (only stores of 1, so the order doesn't matter)
      pmisr_rows_team_vector(local_rows, avg_nnz_local, KOKKOS_LAMBDA(const KokkosTeamMemberType &t, const PetscInt i) {
         if (cf_markers_d(i) != 0) return;
         Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, device_local_i[i], device_local_i[i + 1]), [&](const PetscInt k) {
            const PetscInt col = device_local_j[k];
            if (col == i || !(all_entries || pmisr_is_strong(device_local_vals(k), row_tol_d(i))) || cf_markers_d(col) != 0) return;
            // Each end is only veto'd if it loses, so a NaN measure vetoes neither, as in the CPU PMISR
            if (measure_local_d(i) > measure_local_d(col) || (measure_local_d(i) == measure_local_d(col) && i > col)) {
               Kokkos::atomic_store(&veto_d(i), 1);
            } else if (!directed && (measure_local_d(col) > measure_local_d(i) || (measure_local_d(i) == measure_local_d(col) && col > i))) {
               Kokkos::atomic_store(&veto_d(col), 1);
            }
         });
      });

      if (mpi) {
         PetscCallVoid(PetscSFBcastEnd(sf, MPI_INT, cf_markers_d.data(), cf_markers_nonlocal_d.data(), MPI_REPLACE));
         PetscCallVoid(sf_sync_after_end_kokkos());

         // Cross-rank edges: the row side knows its own loser, and vetoes a ghost through the reduce
         if (!directed) Kokkos::deep_copy(exec, veto_nonlocal_d, 0);
         pmisr_rows_team_vector(local_rows, avg_nnz_nonlocal, KOKKOS_LAMBDA(const KokkosTeamMemberType &t, const PetscInt i) {
            if (cf_markers_d(i) != 0) return;
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, device_nonlocal_i[i], device_nonlocal_i[i + 1]), [&](const PetscInt k) {
               const PetscInt col = device_nonlocal_j[k];
               if (!(all_entries || pmisr_is_strong(device_nonlocal_vals(k), row_tol_d(i))) || cf_markers_nonlocal_d(col) != 0) return;
               if (measure_local_d(i) > measure_nonlocal_d(col) || (measure_local_d(i) == measure_nonlocal_d(col) && global_row_start + i > colmap_d(col))) {
                  Kokkos::atomic_store(&veto_d(i), 1);
               } else if (!directed && (measure_nonlocal_d(col) > measure_local_d(i) || (measure_local_d(i) == measure_nonlocal_d(col) && colmap_d(col) > global_row_start + i))) {
                  Kokkos::atomic_store(&veto_nonlocal_d(col), 1);
               }
            });
         });

         // Directed, a ghost is only veto'd by its own row, on the rank that owns it
         if (!directed) {
            PetscCallVoid(sf_sync_before_begin_kokkos());
            PetscCallVoid(PetscSFReduceWithMemTypeBegin(sf, MPI_INT,
                        PETSC_MEMTYPE_KOKKOS, veto_nonlocal_d.data(),
                        PETSC_MEMTYPE_KOKKOS, veto_d.data(),
                        MPI_LOR));
            PetscCallVoid(PetscSFReduceEnd(sf, MPI_INT, veto_nonlocal_d.data(), veto_d.data(), MPI_LOR));
            PetscCallVoid(sf_sync_after_end_kokkos());
         }
      }

      // Anything not veto'd is in the set
      Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
         if (!veto_d(i)) cf_markers_d(i) = loops_through;
      });

      // ~~~~~~~~
      // Undecided neighbours of nodes that just entered the set become C. Each edge is handled
      // from the row that stores it, again in both directions
      // ~~~~~~~~
      // Only 0 -> 1 changes, and the set (loops_through) is fixed during the kernel, so the order
      // doesn't matter
      pmisr_rows_team_vector(local_rows, avg_nnz_local, KOKKOS_LAMBDA(const KokkosTeamMemberType &t, const PetscInt i) {
         // Loaded once and broadcast, so all the lanes of the row take the same branch even if
         // another row changes cf_markers_d(i) meanwhile
         int cf_i = 0;
         Kokkos::single(Kokkos::PerThread(t), [&](int &v) { v = Kokkos::atomic_load(&cf_markers_d(i)); }, cf_i);
         if (cf_i != loops_through && (directed || cf_i != 0)) return;
         Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, device_local_i[i], device_local_i[i + 1]), [&](const PetscInt k) {
            const PetscInt col = device_local_j[k];
            if (col == i || !(all_entries || pmisr_is_strong(device_local_vals(k), row_tol_d(i)))) return;
            if (cf_i == loops_through) {
               Kokkos::atomic_compare_exchange(&cf_markers_d(col), 0, 1);
            } else if (!directed && Kokkos::atomic_load(&cf_markers_d(col)) == loops_through) {
               Kokkos::atomic_compare_exchange(&cf_markers_d(i), 0, 1);
            }
         });
      });

      if (mpi) {
         // The ghosts need to know who just entered the set
         Kokkos::deep_copy(exec, veto_nonlocal_d, 0);
         PetscCallVoid(sf_sync_before_begin_kokkos());
         PetscCallVoid(PetscSFBcastWithMemTypeBegin(sf, MPI_INT,
                     PETSC_MEMTYPE_KOKKOS, cf_markers_d.data(),
                     PETSC_MEMTYPE_KOKKOS, cf_markers_nonlocal_d.data(),
                     MPI_REPLACE));

         // Nodes in the set mark their ghosts as C. This only reads the root
         // cf_markers_d so it overlaps with the broadcast
         pmisr_rows_team_vector(local_rows, avg_nnz_nonlocal, KOKKOS_LAMBDA(const KokkosTeamMemberType &t, const PetscInt i) {
            if (cf_markers_d(i) != loops_through) return;
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, device_nonlocal_i[i], device_nonlocal_i[i + 1]), [&](const PetscInt k) {
               if (all_entries || pmisr_is_strong(device_nonlocal_vals(k), row_tol_d(i))) Kokkos::atomic_store(&veto_nonlocal_d(device_nonlocal_j[k]), 1);
            });
         });

         PetscCallVoid(PetscSFBcastEnd(sf, MPI_INT, cf_markers_d.data(), cf_markers_nonlocal_d.data(), MPI_REPLACE));
         PetscCallVoid(sf_sync_after_end_kokkos());

         Kokkos::deep_copy(exec, veto_d, 0);
         PetscCallVoid(sf_sync_before_begin_kokkos());
         PetscCallVoid(PetscSFReduceWithMemTypeBegin(sf, MPI_INT,
                     PETSC_MEMTYPE_KOKKOS, veto_nonlocal_d.data(),
                     PETSC_MEMTYPE_KOKKOS, veto_d.data(),
                     MPI_LOR));

         // Undecided nodes with a ghost in the set become C (undirected only). This doesn't touch
         // the data in the reduction so it overlaps with it
         if (!directed) pmisr_rows_team_vector(local_rows, avg_nnz_nonlocal, KOKKOS_LAMBDA(const KokkosTeamMemberType &t, const PetscInt i) {
            if (cf_markers_d(i) != 0) return;
            int any_in_set = 0;
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, device_nonlocal_i[i], device_nonlocal_i[i + 1]), [&](const PetscInt k, int &found) {
               if ((all_entries || pmisr_is_strong(device_nonlocal_vals(k), row_tol_d(i))) && cf_markers_nonlocal_d(device_nonlocal_j[k]) == loops_through) found = 1;
            }, Kokkos::Max<int>(any_in_set));
            if (any_in_set > 0) Kokkos::single(Kokkos::PerThread(t), [&]() { cf_markers_d(i) = 1; });
         });

         PetscCallVoid(PetscSFReduceEnd(sf, MPI_INT, veto_nonlocal_d.data(), veto_d.data(), MPI_LOR));
         PetscCallVoid(sf_sync_after_end_kokkos());

         Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
            if (veto_d(i) && cf_markers_d(i) == 0) cf_markers_d(i) = 1;
         });
      }

      loops_through = loops_through - 1;

      if (max_luby_steps < 0) {
         counter_undecided = 0;
         Kokkos::parallel_reduce("ReductionCounter_undecided", Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i, PetscInt &update) {
            if (cf_markers_d(i) == 0) update++;
         }, counter_undecided);
         PetscCallMPIAbort(MPI_COMM_MATRIX, MPI_Allreduce(&counter_undecided, &counter_parallel, 1, MPIU_INT, MPI_SUM, MPI_COMM_MATRIX));
         counter_undecided = counter_parallel;
      }
   }
   while (counter_undecided != 0);

   Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
      if (cf_markers_d(i) < 0) cf_markers_d(i) = -1;
      else cf_markers_d(i) = 1;
   });
   Kokkos::fence();
}

//------------------------------------------------------------------------------------------------------------------------

// The device PMISR (or PMIS), straight from the blocks of a square matrix (a Kokkos AIJ matrix or a
// raw shell), without building S, S^T or S+S^T as PETSc matrices
// With strength_int the input is the level matrix A and the strength is evaluated on the fly with
// the same test as remove_small_from_sparse_kokkos (as generate_sabs), otherwise the input is a
// strength matrix and every stored entry is an edge
// With symmetrize_int the graph is S+S^T: every edge is handled from whichever row stores it,
// writing in both directions, so no local transpose is needed, and the strong entries that cross
// ranks are gathered to the rows they point at with a PetscSF, to get the exact degree of the
// mutual pairs. Otherwise each node only looks at its own row of S (see pmisr_luby_kokkos)
// The measure is the random numbers in measure_local plus the node's degree in the graph (the
// row size of S+S^T or S), negated for a PMIS. The cf markers are the same as the CPU pmisr_cpu on
// the strength matrix generate_sabs builds (or on the given one), on any number of ranks
// Column indices in each row of both blocks are assumed sorted, as elsewhere in PFLARE
PETSC_INTERN void pmisr_from_matrix_kokkos(Mat *input_mat, const PetscReal strong_threshold, const int strength_int, const int symmetrize_int, \
                  const int pmis_int, const int zero_measure_c_point_int, const int max_luby_steps, PetscReal *measure_local)
{
   const bool strength = strength_int == 1, symmetrize = symmetrize_int == 1, pmis = pmis_int == 1;
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols;
   PetscInt global_row_start_temp, global_col_start_temp;
   PetscInt cols_ao = 0;
   // The matrix's blocks on the device (a Kokkos AIJ matrix or a raw shell)
   AIJDeviceAccess acc;
   aij_device_access_get(*input_mat, acc);
   // Are we in parallel?
   const bool mpi = acc.mpi;
   PetscSF sf = acc.sf;
   const PetscInt *colmap = acc.colmap;
   cols_ao = acc.cols_ao;

   MPI_COMM_MATRIX = acc.comm;
   local_rows = acc.local_rows;
   local_cols = acc.local_cols;
   global_row_start_temp = acc.row_start;
   global_col_start_temp = acc.col_start;
   // This path only sees square level matrices, where row and column ownership match
   // and hence the local index of a column is also the local index of its row
   if (global_row_start_temp != global_col_start_temp || local_rows != local_cols) {
      SETERRABORT(MPI_COMM_MATRIX, PETSC_ERR_ARG_SIZ, "pmisr_from_matrix_kokkos needs a square matrix with matching row and column layouts");
   }
   // The symmetrized degree is only worked out from the strength test
   if (symmetrize && !strength) {
      SETERRABORT(MPI_COMM_MATRIX, PETSC_ERR_SUP, "pmisr_from_matrix_kokkos only symmetrizes a strength graph it evaluates itself");
   }
   const PetscInt global_row_start = global_row_start_temp;

   const PetscInt *device_local_i = acc.local_i, *device_local_j = acc.local_j;
   const PetscInt *device_nonlocal_i = acc.nonlocal_i, *device_nonlocal_j = acc.nonlocal_j;
   Kokkos::View<const PetscScalar *> device_local_vals = acc.local_vals, device_nonlocal_vals = acc.nonlocal_vals;

   auto exec = PetscGetKokkosExecutionSpace();

   PetscIntKokkosView colmap_d;
   if (mpi)
   {
      colmap_d = PetscIntKokkosView("colmap_d", cols_ao);
      Kokkos::deep_copy(exec, colmap_d, PetscIntConstKokkosViewHost(colmap, cols_ao));
      PetscCallVoid(PetscLogCpuToGpu(cols_ao * sizeof(PetscInt)));
   }

   // ~~~~~~~~~~~~
   // The relative row tolerance, identical to the -1 relative_max_row_tolerance_int branch
   // of remove_small_from_sparse_kokkos: the max is over the off-diagonal entries in both blocks
   // ~~~~~~~~~~~~
   // Empty if every stored entry is an edge
   PetscScalarKokkosView row_tol_d;
   // The average row lengths, for the vector length of the row kernels below
   const double avg_nnz_local = local_rows > 0 ? static_cast<double>(acc.nnz_local) / local_rows : 0.0;
   const double avg_nnz_nonlocal = local_rows > 0 && mpi ? static_cast<double>(acc.nnz_nonlocal) / local_rows : 0.0;
   if (strength) row_tol_d = PetscScalarKokkosView("row_tol_d", local_rows);
   if (strength) pmisr_rows_team_vector(local_rows, avg_nnz_local + avg_nnz_nonlocal, KOKKOS_LAMBDA(const KokkosTeamMemberType &t, const PetscInt i) {
      // A max is the same in any order; it starts from -1 (as the CPU path does), which any
      // entry beats, and a NaN never wins a comparison either way
      PetscReal max_l = -1.0, max_nl = -1.0;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, device_local_i[i], device_local_i[i + 1]), [&](const PetscInt k, PetscReal &m) {
         if (device_local_j[k] == i) return;
         const PetscReal val = Kokkos::abs(device_local_vals(k));
         if (val > m) m = val;
      }, Kokkos::Max<PetscReal>(max_l));
      if (mpi) {
         Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, device_nonlocal_i[i], device_nonlocal_i[i + 1]), [&](const PetscInt k, PetscReal &m) {
            const PetscReal val = Kokkos::abs(device_nonlocal_vals(k));
            if (val > m) m = val;
         }, Kokkos::Max<PetscReal>(max_nl));
      }
      PetscReal max_val = -1.0;
      if (max_l > max_val) max_val = max_l;
      if (max_nl > max_val) max_val = max_nl;
      Kokkos::single(Kokkos::PerThread(t), [&]() { row_tol_d(i) = strong_threshold * max_val; });
   });

   // ~~~~~~~~~~~~
   // The degree of each node in the graph. degree_local_d and degree_nonlocal_d are the row sizes
   // of the local and non-local blocks of S+S^T (or S). They are kept separate so the measure is
   // summed in the same order as the CPU pmisr
   // ~~~~~~~~~~~~
   intKokkosView degree_local_d("degree_local_d", local_rows);
   intKokkosView degree_nonlocal_d("degree_nonlocal_d", local_rows);

   // The row sizes of S: every stored entry of a strength matrix (which generate_sabs builds
   // without diagonals), or the strong off-diagonal entries of A
   if (!symmetrize)
   {
      pmisr_rows_team_vector(local_rows, avg_nnz_local + avg_nnz_nonlocal, KOKKOS_LAMBDA(const KokkosTeamMemberType &t, const PetscInt i) {
         int count_l = 0, count_nl = 0;
         if (strength) {
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, device_local_i[i], device_local_i[i + 1]), [&](const PetscInt k, int &count) {
               if (device_local_j[k] != i && pmisr_is_strong(device_local_vals(k), row_tol_d(i))) count++;
            }, count_l);
            if (mpi) {
               Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, device_nonlocal_i[i], device_nonlocal_i[i + 1]), [&](const PetscInt k, int &count) {
                  if (pmisr_is_strong(device_nonlocal_vals(k), row_tol_d(i))) count++;
               }, count_nl);
            }
         }
         else {
            count_l = static_cast<int>(device_local_i[i + 1] - device_local_i[i]);
            if (mpi) count_nl = static_cast<int>(device_nonlocal_i[i + 1] - device_nonlocal_i[i]);
         }
         Kokkos::single(Kokkos::PerThread(t), [&]() {
            degree_local_d(i) = count_l;
            degree_nonlocal_d(i) = count_nl;
         });
      });
   }
   // The row sizes of S+S^T, with mutual pairs counted once
   else
   {
      PetscIntKokkosView strong_nonlocal_d;
      if (mpi) strong_nonlocal_d = PetscIntKokkosView("strong_nonlocal_d", local_rows);

      // Integer counts and atomic increments, so the result doesn't depend on the order
      pmisr_rows_team_vector(local_rows, avg_nnz_local + avg_nnz_nonlocal, KOKKOS_LAMBDA(const KokkosTeamMemberType &t, const PetscInt i) {
         int out_minus_mutual = 0;
         Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, device_local_i[i], device_local_i[i + 1]), [&](const PetscInt k, int &count) {
            const PetscInt col = device_local_j[k];
            if (col == i || !pmisr_is_strong(device_local_vals(k), row_tol_d(i))) return;
            count++;
            Kokkos::atomic_inc(&degree_local_d(col));
            // Is the reverse entry strong as well? Then this pair was counted twice
            const PetscInt col_start = device_local_i[col];
            const PetscInt pos = binary_search_sorted([&](const PetscInt m) { return device_local_j[col_start + m]; },
                                                      device_local_i[col + 1] - col_start, i);
            if (pos >= 0 && pmisr_is_strong(device_local_vals(col_start + pos), row_tol_d(col))) count--;
         }, out_minus_mutual);
         Kokkos::single(Kokkos::PerThread(t), [&]() { Kokkos::atomic_add(&degree_local_d(i), out_minus_mutual); });

         if (mpi) {
            PetscInt strong_count = 0;
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, device_nonlocal_i[i], device_nonlocal_i[i + 1]), [&](const PetscInt k, PetscInt &count) {
               if (pmisr_is_strong(device_nonlocal_vals(k), row_tol_d(i))) count++;
            }, strong_count);
            Kokkos::single(Kokkos::PerThread(t), [&]() {
               strong_nonlocal_d(i) = strong_count;
               degree_nonlocal_d(i) = static_cast<int>(strong_count);
            });
         }
      });

      // The strong entries in the transpose that cross ranks and their mutual pairs need comms.
      // Each strong non-local entry (i, ghost j) is a leaf of an SF whose root is row j (by its
      // boundary-row number, see below) on the rank that owns it, taken from the matrix's own
      // mult SF. The root degree is then the number of strong entries in other ranks' rows that
      // point at that row, and gathering the global row index of each leaf to its root lets the
      // owner find the mutual pairs. The messages only hold the strong entries that cross ranks;
      // the in-degree and two cheap host loops are over the boundary rows (those other ranks hold
      // as ghosts), so no parallel matrix is built or transposed on the host
      if (mpi)
      {
         PetscInt strong_nonlocal_nnz = 0;
         Kokkos::parallel_reduce(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i, PetscInt &update) {
            update += strong_nonlocal_d(i);
         }, strong_nonlocal_nnz);
         // Only whether any rank has one matters, and a flag can't overflow like a global count
         int any_strong_nonlocal = strong_nonlocal_nnz > 0, any_strong_nonlocal_global = 0;
         PetscCallMPIAbort(MPI_COMM_MATRIX, MPI_Allreduce(&any_strong_nonlocal, &any_strong_nonlocal_global, 1, MPI_INT, MPI_MAX, MPI_COMM_MATRIX));

         if (any_strong_nonlocal_global)
         {
            // The ghost column and global row index of each strong non-local entry
            PetscIntKokkosView i_strong_d("i_strong_d", local_rows + 1);
            PetscIntKokkosView ghost_strong_d("ghost_strong_d", strong_nonlocal_nnz);
            PetscIntKokkosView row_strong_d("row_strong_d", strong_nonlocal_nnz);
            Kokkos::parallel_scan(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i, PetscInt &update, const bool final) {
               update += strong_nonlocal_d(i);
               if (final) i_strong_d(i + 1) = update;
            });
            Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
               PetscInt pos = i_strong_d(i);
               for (PetscInt k = device_nonlocal_i[i]; k < device_nonlocal_i[i + 1]; k++) {
                  if (pmisr_is_strong(device_nonlocal_vals(k), row_tol_d(i))) {
                     ghost_strong_d(pos) = device_nonlocal_j[k];
                     row_strong_d(pos) = global_row_start + i;
                     pos++;
                  }
               }
            });
            auto ghost_strong_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ghost_strong_d);
            auto row_strong_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), row_strong_d);
            PetscCallVoid(PetscLogGpuToCpu(2 * strong_nonlocal_nnz * sizeof(PetscInt)));

            // Where each ghost column lives, from the leaves of the mult SF
            PetscInt sf_nroots, sf_nleaves;
            const PetscInt *sf_ilocal = NULL;
            const PetscSFNode *sf_iremote = NULL;
            PetscCallVoid(PetscSFGetGraph(sf, &sf_nroots, &sf_nleaves, &sf_ilocal, &sf_iremote));
            // Every ghost column must have exactly one leaf, or ghost_remote below would be incomplete
            if (sf_nleaves != cols_ao) {
               SETERRABORT(MPI_COMM_MATRIX, PETSC_ERR_PLIB, "pmisr_from_matrix_kokkos needs one mult SF leaf per ghost column");
            }
            // Only the local rows other ranks hold as ghost columns (the roots of the mult SF that have
            // leaves) can be pointed at by a strong entry from another rank. They are numbered
            // compactly, in increasing row order, and the edge SF's roots are those numbers, so the
            // degree and gather below (and the host loops over their roots) scale with the boundary
            // rows rather than all the local rows
            // Mark them with a reduce from every ghost leaf of the mult SF (any SF type), and number
            // them with a scan
            PetscIntKokkosView ones_ghost_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "ones_ghost_d"), cols_ao);
            Kokkos::deep_copy(exec, ones_ghost_d, 1);
            // Zero initialised on exec
            PetscIntKokkosView bnd_flag_d(Kokkos::view_alloc(exec, "bnd_flag_d"), local_rows);
            PetscCallVoid(sf_sync_before_begin_kokkos());
            PetscCallVoid(PetscSFReduceWithMemTypeBegin(sf, MPIU_INT, PETSC_MEMTYPE_KOKKOS, ones_ghost_d.data(), \
                        PETSC_MEMTYPE_KOKKOS, bnd_flag_d.data(), MPI_MAX));
            PetscCallVoid(PetscSFReduceEnd(sf, MPIU_INT, ones_ghost_d.data(), bnd_flag_d.data(), MPI_MAX));
            PetscCallVoid(sf_sync_after_end_kokkos());
            // Each boundary row's compact number (the rest of comp_d is never read)
            PetscIntKokkosView comp_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "comp_d"), local_rows);
            PetscInt n_bnd = 0;
            Kokkos::parallel_scan(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i, PetscInt &update, const bool final) {
               if (final) comp_d(i) = update;
               if (bnd_flag_d(i)) update++;
            }, n_bnd);
            PetscIntKokkosView bnd_rows_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "bnd_rows_d"), n_bnd);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
               if (bnd_flag_d(i)) bnd_rows_d(comp_d(i)) = i;
            });
            auto bnd_rows = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bnd_rows_d);
            PetscCallVoid(PetscLogGpuToCpu(n_bnd * sizeof(PetscInt)));

            // Each boundary row's compact number, sent to the ranks that have it as a ghost column
            PetscIntKokkosView comp_ghost_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "comp_ghost_d"), cols_ao);
            PetscCallVoid(sf_sync_before_begin_kokkos());
            PetscCallVoid(PetscSFBcastWithMemTypeBegin(sf, MPIU_INT, PETSC_MEMTYPE_KOKKOS, comp_d.data(), \
                        PETSC_MEMTYPE_KOKKOS, comp_ghost_d.data(), MPI_REPLACE));
            PetscCallVoid(PetscSFBcastEnd(sf, MPIU_INT, comp_d.data(), comp_ghost_d.data(), MPI_REPLACE));
            PetscCallVoid(sf_sync_after_end_kokkos());
            auto comp_ghost_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), comp_ghost_d);
            PetscCallVoid(PetscLogGpuToCpu(cols_ao * sizeof(PetscInt)));

            PetscSFNode *ghost_remote = NULL, *edge_remote = NULL;
            PetscCallVoid(PetscMalloc1(cols_ao, &ghost_remote));
            for (PetscInt l = 0; l < sf_nleaves; l++) ghost_remote[sf_ilocal ? sf_ilocal[l] : l] = sf_iremote[l];
            PetscCallVoid(PetscMalloc1(strong_nonlocal_nnz, &edge_remote));
            for (PetscInt k = 0; k < strong_nonlocal_nnz; k++) {
               edge_remote[k].rank = ghost_remote[ghost_strong_h(k)].rank;
               edge_remote[k].index = comp_ghost_h(ghost_strong_h(k));
            }
            PetscCallVoid(PetscFree(ghost_remote));

            PetscSF edge_sf;
            PetscCallVoid(PetscSFCreate(MPI_COMM_MATRIX, &edge_sf));
            PetscCallVoid(PetscSFSetGraph(edge_sf, n_bnd, strong_nonlocal_nnz, NULL, PETSC_COPY_VALUES, edge_remote, PETSC_OWN_POINTER));
            // The order of the gathered entries at each root doesn't matter
            PetscCallVoid(PetscSFSetRankOrder(edge_sf, PETSC_FALSE));
            PetscCallVoid(PetscSFSetUp(edge_sf));

            const PetscInt *in_degree = NULL;
            PetscCallVoid(PetscSFComputeDegreeBegin(edge_sf, &in_degree));
            PetscCallVoid(PetscSFComputeDegreeEnd(edge_sf, &in_degree));

            // Only the rows that something points at need any work (in increasing row order, as the
            // boundary rows are numbered in that order)
            PetscInt in_rows = 0, in_nnz = 0;
            for (PetscInt b = 0; b < n_bnd; b++) {
               if (in_degree[b] > 0) in_rows++;
               in_nnz += in_degree[b];
            }
            PetscIntKokkosViewHost in_row_h("in_row_h", in_rows), in_offset_h("in_offset_h", in_rows + 1);
            in_offset_h(0) = 0;
            for (PetscInt b = 0, n = 0; b < n_bnd; b++) {
               if (in_degree[b] == 0) continue;
               in_row_h(n) = bnd_rows(b);
               in_offset_h(n + 1) = in_offset_h(n) + in_degree[b];
               n++;
            }

            // The global row index of every leaf pointing at each root, contiguous per root
            PetscIntKokkosViewHost in_global_h("in_global_h", in_nnz);
            PetscCallVoid(PetscSFGatherBegin(edge_sf, MPIU_INT, row_strong_h.data(), in_global_h.data()));
            PetscCallVoid(PetscSFGatherEnd(edge_sf, MPIU_INT, row_strong_h.data(), in_global_h.data()));
            PetscCallVoid(PetscSFDestroy(&edge_sf));

            auto in_row_d = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), in_row_h);
            auto in_offset_d = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), in_offset_h);
            auto in_global_d = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), in_global_h);
            PetscCallVoid(PetscLogCpuToGpu((2 * in_rows + 1 + in_nnz) * sizeof(PetscInt)));

            // A mutual pair has the other row in both the strong entries of row i and the
            // entries that point at row i, and must only be counted once
            // The strong entries of row i are in column order and colmap is increasing, so their
            // global columns are sorted
            Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, in_rows), KOKKOS_LAMBDA(const PetscInt n) {
               const PetscInt i = in_row_d(n);
               const PetscInt strong_start = i_strong_d(i);
               const PetscInt strong_size = i_strong_d(i + 1) - strong_start;
               int in_minus_mutual = static_cast<int>(in_offset_d(n + 1) - in_offset_d(n));
               for (PetscInt k = in_offset_d(n); k < in_offset_d(n + 1); k++) {
                  if (binary_search_sorted([&](const PetscInt m) { return colmap_d(ghost_strong_d(strong_start + m)); },
                                           strong_size, in_global_d(k)) >= 0) in_minus_mutual--;
               }
               degree_nonlocal_d(i) += in_minus_mutual;
            });
            Kokkos::fence();
         }
      }
   }

   // Device memory for the global variable cf_markers_local_d - be careful these aren't petsc ints
   cf_markers_local_d = intKokkosView("cf_markers_local_d", local_rows);
   intKokkosView cf_markers_d = cf_markers_local_d;

   // The measure is random + row sizes of the graph, summed in the same order as the CPU pmisr,
   // and negated for a PMIS
   PetscScalarKokkosViewHost measure_local_h(measure_local, local_rows);
   PetscScalarKokkosView measure_local_d("measure_local_d", local_rows);
   Kokkos::deep_copy(exec, measure_local_d, measure_local_h);
   PetscCallVoid(PetscLogCpuToGpu(local_rows * sizeof(PetscReal)));

   Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
      measure_local_d(i) += degree_local_d(i);
      if (mpi) measure_local_d(i) += degree_nonlocal_d(i);
      if (pmis) measure_local_d(i) *= -1;
   });

   // Nodes with no neighbours (abs(measure) < 1) become F, or C if zero_measure_c_point_int is set
   // A PMIS swaps C and F at the end, so it starts them the other way round
   int isolated_marker = zero_measure_c_point_int == 1 ? 1 : -1;
   if (pmis) isolated_marker *= -1;

   pmisr_luby_kokkos(acc, colmap_d, row_tol_d, !symmetrize, isolated_marker, max_luby_steps, measure_local_d, cf_markers_d);

   // A PMIS puts the set in C rather than F
   if (pmis) {
      Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
         cf_markers_d(i) *= -1;
      });
      Kokkos::fence();
   }

   aij_device_access_restore(acc);
}

//------------------------------------------------------------------------------------------------------------------------

// PMISR with an existing measure and cf markers on the device, straight from a square matrix (a
// Kokkos AIJ matrix or a raw shell) with every stored off-diagonal entry an edge, so the graph is
// the pattern of A + A^T without forming it: see pmisr_luby_kokkos
PETSC_INTERN void pmisr_existing_measure_from_matrix_kokkos(Mat *input_mat, const int max_luby_steps, PetscScalarKokkosView &measure_local_d, intKokkosView &cf_markers_d)
{
   AIJDeviceAccess acc;
   aij_device_access_get(*input_mat, acc);
   if (acc.row_start != acc.col_start || acc.local_rows != acc.local_cols) {
      SETERRABORT(acc.comm, PETSC_ERR_ARG_SIZ, "pmisr_existing_measure_from_matrix_kokkos needs a square matrix with matching row and column layouts");
   }
   auto exec = PetscGetKokkosExecutionSpace();
   PetscIntKokkosView colmap_d;
   if (acc.mpi)
   {
      colmap_d = PetscIntKokkosView("colmap_d", acc.cols_ao);
      Kokkos::deep_copy(exec, colmap_d, PetscIntConstKokkosViewHost(acc.colmap, acc.cols_ao));
      PetscCallVoid(PetscLogCpuToGpu(acc.cols_ao * sizeof(PetscInt)));
   }
   pmisr_luby_kokkos(acc, colmap_d, PetscScalarKokkosView(), false, -1, max_luby_steps, measure_local_d, cf_markers_d);
   aij_device_access_restore(acc);
}

//------------------------------------------------------------------------------------------------------------------------
