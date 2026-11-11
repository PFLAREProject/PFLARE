// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>

// The definition of the device copy of the cf markers on a given level
// is stored in Device_Datak.kokkos.cxx and imported as extern from
// kokkos_helper.hpp

//------------------------------------------------------------------------------------------------------------------------

// PMISR implementation that takes an existing measure and cf_markers on the device
// and then does the Luby algorithm to assign the rest of the CF markers
// This mirrors the CPU version pmisr_existing_measure_cf_markers in PMISR_Module.F90
PETSC_INTERN void pmisr_existing_measure_cf_markers_kokkos(Mat *strength_mat, const int max_luby_steps, const int pmis_int, PetscScalarKokkosView &measure_local_d, intKokkosView &cf_markers_d, const int zero_measure_c_point_int)
{

   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt rows_ao, cols_ao;
   MatType mat_type;

   PetscCallVoid(MatGetType(*strength_mat, &mat_type));
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat mat_local = NULL, mat_nonlocal = NULL;
   // The matrix's own mult SF (the borrowed mvctx) drives the halo comms
   PetscSF sf = NULL;
   // colmap (garray) maps off-diagonal local columns to global indices; needed on
   // the device for the global-index tie break in the non-local comparison
   const PetscInt *colmap = NULL;

   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*strength_mat, &mat_local, &mat_nonlocal, &colmap));
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao));
      PetscCallVoid(MatGetMultPetscSF(*strength_mat, &sf));
   }
   else
   {
      mat_local = *strength_mat;
   }

   // Get the comm
   PetscCallVoid(PetscObjectGetComm((PetscObject)*strength_mat, &MPI_COMM_MATRIX));
   PetscCallVoid(MatGetLocalSize(*strength_mat, &local_rows, &local_cols));
   PetscCallVoid(MatGetSize(*strength_mat, &global_rows, &global_cols));
   // This returns the global index of the local portion of the matrix
   PetscCallVoid(MatGetOwnershipRange(*strength_mat, &global_row_start, &global_row_end_plus_one));

   // ~~~~~~~~~~~~
   // Get pointers to the i,j on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, NULL, &mtype));

   intKokkosView cf_markers_nonlocal_d;
   // Scratch buffer used for local update bookkeeping during overlap with the reverse reduction
   intKokkosView cf_markers_temp_d;
   // Root of the reverse reduction, which only accumulates the neighbour marks
   intKokkosView cf_markers_reduced_d;

   PetscScalarKokkosView measure_nonlocal_d;
   // Device copy of colmap for the non-local tie break
   PetscIntKokkosView colmap_d;

   if (mpi) {
      measure_nonlocal_d = PetscScalarKokkosView("measure_nonlocal_d", cols_ao);
      cf_markers_nonlocal_d = intKokkosView("cf_markers_nonlocal_d", cols_ao);
      cf_markers_temp_d = intKokkosView("cf_markers_temp_d", local_rows);
      cf_markers_reduced_d = intKokkosView("cf_markers_reduced_d", local_rows);
      colmap_d = PetscIntKokkosView("colmap_d", cols_ao);
      Kokkos::deep_copy(colmap_d, PetscIntConstKokkosViewHost(colmap, cols_ao));
   }

   // Device memory for the mark
   boolKokkosView mark_d("mark_d", local_rows);
   auto exec = PetscGetKokkosExecutionSpace();

   // Start the broadcast of the measure, overlapped with the initial count below
   // which only reads the root measure_local_d
   if (mpi)
   {
      PetscCallVoid(sf_sync_before_begin_kokkos());
      PetscCallVoid(PetscSFBcastWithMemTypeBegin(sf, MPIU_SCALAR,
                  PETSC_MEMTYPE_KOKKOS, measure_local_d.data(),
                  PETSC_MEMTYPE_KOKKOS, measure_nonlocal_d.data(),
                  MPI_REPLACE));
   }

   // Initialise the set
   PetscInt counter_in_set_start = 0;
   // Count how many in the set to begin with and set their CF markers
   Kokkos::parallel_reduce ("Reduction", Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA (const PetscInt i, PetscInt& update) {
      // If already assigned by the input
      if (cf_markers_d(i) != 0)
      {
         update++;
      }
      else if (Kokkos::abs(measure_local_d[i]) < 1)
      {
         if (zero_measure_c_point_int == 1) {
            if (pmis_int == 1) {
               // Set as F here but reversed below to become C
               cf_markers_d(i) = -1;
            }
            else {
               // Becomes C
               cf_markers_d(i) = 1;
            }
         }
         else {
            if (pmis_int == 1) {
               // Set as C here but reversed below to become F
               // Otherwise dirichlet conditions persist down onto the coarsest grid
               cf_markers_d(i) = 1;
            }
            else {
               // Becomes F
               cf_markers_d(i) = -1;
            }
         }
         // Count
         update++;
      }
   }, counter_in_set_start);

   // Finish the measure broadcast
   if (mpi)
   {
      PetscCallVoid(PetscSFBcastEnd(sf, MPIU_SCALAR, measure_local_d.data(), measure_nonlocal_d.data(), MPI_REPLACE));
      PetscCallVoid(sf_sync_after_end_kokkos());
   }

   // Check the total number of undecided in parallel
   PetscInt counter_undecided, counter_parallel;
   if (max_luby_steps < 0) {
      counter_undecided = local_rows - counter_in_set_start;
      // Parallel reduction!
      PetscCallMPIAbort(MPI_COMM_MATRIX, MPI_Allreduce(&counter_undecided, &counter_parallel, 1, MPIU_INT, MPI_SUM, MPI_COMM_MATRIX));
      counter_undecided = counter_parallel;

   // If we're doing a fixed number of steps, then we don't care
   // how many undecided nodes we have - have to take care here not to use
   // local_rows for counter_undecided, as we may have zero DOFs on some procs
   // but we have to enter the loop below for the collective scatters
   }
   else {
      counter_undecided = 1;
   }

   // ~~~~~~~~~~~~
   // Now go through the outer Luby loop
   // ~~~~~~~~~~~~

   // Let's keep track of how many times we go through the loops
   int loops_through = -1;
   do
   {
      // Match the fortran version and include a pre-test on the do-while
      if (counter_undecided == 0) break;

      // If max_luby_steps is positive, then we only take that many times through this top loop
      // We typically find 2-3 iterations decides >99% of the nodes
      // and a fixed number of outer loops means we don't have to do any parallel reductions
      // We will do redundant nearest neighbour comms in the case we have already
      // finished deciding all the nodes, but who cares
      // Any undecided nodes just get turned into C points
      // We can do this as we know we won't ruin Aff by doing so, unlike in a normal multigrid
      if (max_luby_steps > 0 && max_luby_steps+1 == -loops_through) break;

      // ~~~~~~~~~
      // Start the broadcast of the nonlocal cf_markers
      // The local component below only reads the root cf_markers_d, which is allowed
      // while the broadcast is in flight
      // ~~~~~~~~~
      if (mpi) {
         PetscCallVoid(sf_sync_before_begin_kokkos());
         PetscCallVoid(PetscSFBcastWithMemTypeBegin(sf, MPI_INT,
                     PETSC_MEMTYPE_KOKKOS, cf_markers_d.data(),
                     PETSC_MEMTYPE_KOKKOS, cf_markers_nonlocal_d.data(),
                     MPI_REPLACE));
      }


      // mark_d keeps track of which of the candidate nodes can become in the set
      // Only need this because we want to do async comms so we need a way to trigger
      // a node not being in the set due to either strong local neighbours *or* strong offproc neighbours

      // ~~~~~~~~
      // Go and do the local component
      // ~~~~~~~~
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            const PetscInt i = t.league_rank();
            PetscInt strong_neighbours = 0;

            // Check this row isn't already marked
            if (cf_markers_d(i) == 0)
            {
               const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];

               // Reduce over local columns to get the number of strong neighbours
               Kokkos::parallel_reduce(
                  Kokkos::TeamThreadRange(t, ncols_local),
                  [&](const PetscInt j, PetscInt& strong_count) {

                  // Have to only check active strong neighbours. In single precision
                  // may need a tie break and we use the global index (larger index loses)
                  const PetscInt col = device_local_j[device_local_i[i] + j];
                  if ((measure_local_d(i) > measure_local_d(col) || \
                           (measure_local_d(i) == measure_local_d(col) && i > col)) && \
                           cf_markers_d(col) == 0)
                  {
                     strong_count++;
                  }

               }, strong_neighbours
               );

               // Only want one thread in the team to write the result
               Kokkos::single(Kokkos::PerTeam(t), [&]() {
                  // If we have any strong neighbours
                  if (strong_neighbours > 0)
                  {
                     mark_d(i) = false;
                  }
                  else
                  {
                     mark_d(i) = true;
                  }
               });
            }
            // Any that aren't zero cf marker are already assigned so set to to false
            else
            {
               // Only want one thread in the team to write the result
               Kokkos::single(Kokkos::PerTeam(t), [&]() {
                  mark_d(i) = false;
               });
            }
      });

      // ~~~~~~~~
      // Now go through and do the non-local part of the matrix
      // ~~~~~~~~
      if (mpi) {

         // Complete the in-flight broadcast before reading the leaf
         PetscCallVoid(PetscSFBcastEnd(sf, MPI_INT, cf_markers_d.data(), cf_markers_nonlocal_d.data(), MPI_REPLACE));
         PetscCallVoid(sf_sync_after_end_kokkos());

         Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

               // Row
               const PetscInt i = t.league_rank();
               PetscInt strong_neighbours = 0;

               // Check this row isn't already marked
               if (mark_d(i))
               {
                  PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];

                  // Reduce over nonlocal columns to get the number of strong neighbours
                  Kokkos::parallel_reduce(
                     Kokkos::TeamThreadRange(t, ncols_nonlocal),
                     [&](const PetscInt j, PetscInt& strong_count) {

                     // Same deterministic global-index tie break as the local loop;
                     // the non-local neighbour's global index is colmap_d(col)
                     const PetscInt col = device_nonlocal_j[device_nonlocal_i[i] + j];
                     if ((measure_local_d(i) > measure_nonlocal_d(col) || \
                              (measure_local_d(i) == measure_nonlocal_d(col) && global_row_start + i > colmap_d(col))) && \
                              cf_markers_nonlocal_d(col) == 0)
                     {
                        strong_count++;
                     }

                  }, strong_neighbours
                  );

                  // Only want one thread in the team to write the result
                  Kokkos::single(Kokkos::PerTeam(t), [&]() {
                     // If we don't have any strong neighbours
                     if (strong_neighbours == 0) cf_markers_d(i) = loops_through;
                  });
               }
         });
      }
      // This cf_markers_d(i) = loops_through happens above in the case of mpi, saves a kernel launch
      else
      {
         // The nodes that have mark equal to true have no strong active neighbours in the IS
         // hence they can be in the IS
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {

               if (mark_d(i)) cf_markers_d(i) = loops_through;
         });
      }

      if (mpi)
      {
         // We're going to do an add reverse scatter, so set them to zero
         Kokkos::deep_copy(exec, cf_markers_nonlocal_d, 0);

         Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

               // Row
               const PetscInt i = t.league_rank();

               // Reuse the send buffer as a temporary local-update buffer.
               // Value 2 marks rows assigned in this top loop and already stored
               // in cf_markers_d as loops_through. Value 1 marks locally discovered
               // neighbours that must be merged into cf_markers_d after the reduction.
               cf_markers_temp_d(i) = 0;

               // Check if this node has been assigned during this top loop
               if (cf_markers_d(i) == loops_through)
               {
                  cf_markers_temp_d(i) = 2;
                  PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];

                  // For over nonlocal columns
                  Kokkos::parallel_for(
                     Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {

                        // Needs to be atomic as may being set by many threads
                        Kokkos::atomic_store(&cf_markers_nonlocal_d(device_nonlocal_j[device_nonlocal_i[i] + j]), 1);
                  });
               }
         });

         // We've updated the values in cf_markers_nonlocal
         // A reverse reduction with a sum will then update the values of cf_markers_local
         // The reduction carries only the 0/1 neighbour marks, not the cf marker values
         // themselves, so the sum is just a logical or of the marks (matching the
         // MPI_LOR PetscSFReduce into assigned_local in the CPU version) and we merge
         // it back below only into nodes that are still unassigned
         // This keeps the same result if the strength matrix is not structurally
         // symmetric, where an already assigned node (either in the set this loop or
         // assigned in an earlier loop) can be a neighbour of an in-set node on
         // another rank and hence receive a mark
         // The root only accumulates the neighbour marks, so start at zero
         Kokkos::deep_copy(exec, cf_markers_reduced_d, 0);
         PetscCallVoid(sf_sync_before_begin_kokkos());
         PetscCallVoid(PetscSFReduceWithMemTypeBegin(sf, MPI_INT,
                     PETSC_MEMTYPE_KOKKOS, cf_markers_nonlocal_d.data(),
                     PETSC_MEMTYPE_KOKKOS, cf_markers_reduced_d.data(),
                     MPI_SUM));

         // While the reverse reduction is in flight, do local-only updates in cf_markers_temp_d.
         // This must not touch cf_markers_reduced_d or cf_markers_nonlocal_d.
         Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

               // Row
               const PetscInt i = t.league_rank();

               // Check if this node was assigned during this top loop.
               // We read the temporary buffer here so we do not race with the
               // reduction into cf_markers_d.
               // Atomic as other teams in this kernel do atomic_max on cf_markers_temp_d
               if (Kokkos::atomic_load(&cf_markers_temp_d(i)) == 2)
               {
                  const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];

                  // For over local columns
                  Kokkos::parallel_for(
                     Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

                        // Needs to be atomic as may be set by many threads.
                        // Use atomic_max so that assigned-row markers stay at 2.
                        Kokkos::atomic_max(&cf_markers_temp_d(device_local_j[device_local_i[i] + j]), 1);
                  });
               }
         });

         // Complete the reverse reduction before reading the reduced root
         PetscCallVoid(PetscSFReduceEnd(sf, MPI_INT, cf_markers_nonlocal_d.data(), cf_markers_reduced_d.data(), MPI_SUM));
         PetscCallVoid(sf_sync_after_end_kokkos());

         // Merge the reduced neighbour marks back into cf_markers_d after End.
         // Any node that received a mark from a remote in-set node is now assigned,
         // but only if it hasn't already been assigned - we must never overwrite the
         // loops_through value of a node in the set this loop, or the marker of a node
         // assigned in an earlier loop
         // The CPU version leaves such nodes with a zero cf marker until the final
         // "unassigned becomes C" pass, so assigning them as C here is equivalent
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
               if (cf_markers_reduced_d(i) > 0 && cf_markers_d(i) == 0) cf_markers_d(i) = 1;
         });

         // Merge the local updates after the reduction has completed.
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {

            // Only the locally discovered neighbours are merged here.
            // Entries left at 2 were already selected into the set earlier.
            if (cf_markers_temp_d(i) == 1 && cf_markers_d(i) == 0) cf_markers_d(i) = 1;
         });
      }
      // In serial we can just update cf_markers_d directly without needing the send buffer
      else
      {
         Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

               // Row
               const PetscInt i = t.league_rank();

               // In serial there is no reduction, so we can
               // update cf_markers_d directly as before.
               // Atomic as other teams in this kernel compare and exchange on cf_markers_d
               if (Kokkos::atomic_load(&cf_markers_d(i)) == loops_through)
               {
                  const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];

                  Kokkos::parallel_for(
                     Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

                        // Needs to be atomic as may be set by many threads
                        // Only assign neighbours that are still unassigned - if the
                        // strength matrix is not structurally symmetric two neighbouring
                        // nodes can be selected into the set in the same loop, and we
                        // must not overwrite the loops_through value of an in-set node
                        // (or the marker of a node assigned in an earlier loop)
                        // This matches the CPU version, which tracks assignment in a
                        // separate boolean and never touches an assigned node's marker
                        // It also means the loops_through values are stable while this
                        // kernel runs, so the guard above cannot race
                        Kokkos::atomic_compare_exchange(&cf_markers_d(device_local_j[device_local_i[i] + j]), 0, 1);
                  });
               }
         });
      }

      // We've done another top level loop
      loops_through = loops_through - 1;

      // ~~~~~~~~~~~~
      // Check the total number of undecided in parallel before we loop again
      // ~~~~~~~~~~~~
      if (max_luby_steps < 0) {

         counter_undecided = 0;
         Kokkos::parallel_reduce ("ReductionCounter_undecided", Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA (const PetscInt i, PetscInt& update) {
            if (cf_markers_d(i) == 0) update++;
         }, counter_undecided);

         // Parallel reduction!
         PetscCallMPIAbort(MPI_COMM_MATRIX, MPI_Allreduce(&counter_undecided, &counter_parallel, 1, MPIU_INT, MPI_SUM, MPI_COMM_MATRIX));
         counter_undecided = counter_parallel;
      }

   }
   while (counter_undecided != 0);

   // ~~~~~~~~~
   // Now assign our final cf markers
   // ~~~~~~~~~

   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {

      if (cf_markers_d(i) == 0)
      {
         cf_markers_d(i) = 1;
      }
      else if (cf_markers_d(i) < 0)
      {
         cf_markers_d(i) = -1;
      }
      else
      {
         cf_markers_d(i) = 1;
      }
   });
   // Ensure we're done before we exit
   Kokkos::fence();

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// PMISR implementation that takes an existing measure and cf_markers on the device
// and then does the Luby algorithm to assign the rest of the CF markers
// This version takes S (not S+S^T) as the strength matrix and handles the transpose
// implicitly - it never forms the full parallel S+S^T
// See the full comments in the CPU version pmisr_existing_measure_implicit_transpose
PETSC_INTERN void pmisr_existing_measure_implicit_transpose_kokkos(Mat *strength_mat, const int max_luby_steps, const int pmis_int, PetscScalarKokkosView &measure_local_d, intKokkosView &cf_markers_d, const int zero_measure_c_point_int)
{

   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt rows_ao, cols_ao;
   MatType mat_type;

   PetscCallVoid(MatGetType(*strength_mat, &mat_type));
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat mat_local = NULL, mat_nonlocal = NULL, mat_local_spst = NULL, mat_nonlocal_transpose = NULL;
   // The matrix's own mult SF (the borrowed mvctx) drives the halo comms
   PetscSF sf = NULL;

   // ~~~~~~~~~~~~~~~~~~~~~
   // PMISR needs to work with S+S^T to keep out large entries from Aff
   // but we never want to form S+S^T explicitly as it is expensive
   // So instead we do several comms steps in our Luby loop to get/send the data we need
   // We do compute local copies of the transpose of S (which happen on the device)
   // but we never have the full parallel S+S^T
   // On this rank we have the number of:
   // local strong dependencies (from the local S)
   // local strong influences (from the local S^T)
   // non-local strong dependencies (from the non-local part of S)
   // But we don't have the number of non-local strong influences (from the non-local part of S^T)
   // Now we have to be careful as the local part of S and S^T may have entries in the same
   // row/column position, so we have to be sure not to count them twice (the same can't happen
   // for the non-local components)
   // ~~~~~~~~~~~~~~~~~~~~~

   bool destroy_nonlocal_transpose = false;
   bool destroy_spst = false;
   // colmap (garray) maps off-diagonal local columns to global indices; needed on
   // the device for the global-index tie break in the non-local comparisons
   const PetscInt *colmap = NULL;

   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*strength_mat, &mat_local, &mat_nonlocal, &colmap));
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao));
      PetscCallVoid(MatGetMultPetscSF(*strength_mat, &sf));
      // Read the nnz from the host i pointer
      PetscInt n;
      PetscBool symmetric = PETSC_FALSE, inodecompressed = PETSC_FALSE, done;
      const PetscInt *ao_ia;
      PetscCallVoid(MatGetRowIJ(mat_nonlocal, 0, symmetric, inodecompressed, &n, &ao_ia, NULL, &done));
      const PetscInt nonlocal_nnz = ao_ia[n];
      PetscCallVoid(MatRestoreRowIJ(mat_nonlocal, 0, symmetric, inodecompressed, &n, &ao_ia, NULL, &done));
      // The transpose can crash if mat_nonlocal is empty
      if (nonlocal_nnz > 0)
      {
         PetscCallVoid(MatTranspose(mat_nonlocal, MAT_INITIAL_MATRIX, &mat_nonlocal_transpose));
         destroy_nonlocal_transpose = true;
      }
   }
   else
   {
      mat_local = *strength_mat;
   }

   // Get the comm
   PetscCallVoid(PetscObjectGetComm((PetscObject)*strength_mat, &MPI_COMM_MATRIX));
   PetscCallVoid(MatGetLocalSize(*strength_mat, &local_rows, &local_cols));
   PetscCallVoid(MatGetSize(*strength_mat, &global_rows, &global_cols));
   // This returns the global index of the local portion of the matrix
   PetscCallVoid(MatGetOwnershipRange(*strength_mat, &global_row_start, &global_row_end_plus_one));

   // Read the nnz from the host i pointer of the local block
   PetscInt local_nnz;
   {
      PetscInt n;
      PetscBool symmetric = PETSC_FALSE, inodecompressed = PETSC_FALSE, done;
      const PetscInt *ad_ia;
      PetscCallVoid(MatGetRowIJ(mat_local, 0, symmetric, inodecompressed, &n, &ad_ia, NULL, &done));
      local_nnz = ad_ia[n];
      PetscCallVoid(MatRestoreRowIJ(mat_local, 0, symmetric, inodecompressed, &n, &ad_ia, NULL, &done));
   }

   // ~~~~~~~~~~~~
   // Form the local S+S^T and get CSR pointers
   // We explicitly compute the local part of S+S^T so we don't have to
   // match the row/column indices - could do this as a symbolic as we don't need the values
   // ~~~~~~~~~~~~
   PetscScalar one = 1.0;
   if (local_nnz > 0)
   {
      PetscCallVoid(MatTranspose(mat_local, MAT_INITIAL_MATRIX, &mat_local_spst));
      PetscCallVoid(MatAXPY(mat_local_spst, one, mat_local, DIFFERENT_NONZERO_PATTERN));
      destroy_spst = true;
   }
   else
   {
      mat_local_spst = mat_local;
   }

   // ~~~~~~~~~~~~
   // Get pointers to the i,j on the device for all the matrices we need
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i_spst = nullptr, *device_local_j_spst = nullptr;
   const PetscInt *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   const PetscInt *device_nonlocal_i_transpose = nullptr, *device_nonlocal_j_transpose = nullptr;
   PetscMemType mtype;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_spst, &device_local_i_spst, &device_local_j_spst, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, NULL, &mtype));
   if (mpi && mat_nonlocal_transpose != NULL) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_transpose, &device_nonlocal_i_transpose, &device_nonlocal_j_transpose, NULL, &mtype));

   intKokkosView cf_markers_nonlocal_d;
   PetscScalarKokkosView measure_nonlocal_d;
   // Device copy of colmap for the non-local tie break
   PetscIntKokkosView colmap_d;

   // ~~~~~~~~~~~~~~~
   // veto stores whether a node has been veto'd as a candidate
   // .NOT. veto(i) means the node can be in the set
   // veto(i) means the node cannot be in the set
   // These are ints rather than bools so they can be reduced with MPI_LOR by PetscSF
   // on the device, which has no device kernels for MPI_C_BOOL
   // ~~~~~~~~~~~~~~~
   intKokkosView veto_local_d("veto_local_d", local_rows);
   intKokkosView veto_nonlocal_d;

   if (mpi) {
      measure_nonlocal_d = PetscScalarKokkosView("measure_nonlocal_d", cols_ao);
      cf_markers_nonlocal_d = intKokkosView("cf_markers_nonlocal_d", cols_ao);
      veto_nonlocal_d = intKokkosView("veto_nonlocal_d", cols_ao);
      colmap_d = PetscIntKokkosView("colmap_d", cols_ao);
      Kokkos::deep_copy(colmap_d, PetscIntConstKokkosViewHost(colmap, cols_ao));
   }

   auto exec = PetscGetKokkosExecutionSpace();

   // Start the broadcast of the measure, overlapped with the initial count below
   // which only reads the root measure_local_d
   if (mpi)
   {
      PetscCallVoid(sf_sync_before_begin_kokkos());
      PetscCallVoid(PetscSFBcastWithMemTypeBegin(sf, MPIU_SCALAR,
                  PETSC_MEMTYPE_KOKKOS, measure_local_d.data(),
                  PETSC_MEMTYPE_KOKKOS, measure_nonlocal_d.data(),
                  MPI_REPLACE));
   }

   // ~~~~~~~~~~~~
   // Initialise the set
   // ~~~~~~~~~~~~
   PetscInt counter_in_set_start = 0;
   // Count how many in the set to begin with and set their CF markers
   Kokkos::parallel_reduce ("Reduction", Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA (const PetscInt i, PetscInt& update) {
      // If already assigned by the input
      if (cf_markers_d(i) != 0)
      {
         update++;
      }
      else if (Kokkos::abs(measure_local_d[i]) < 1)
      {
         if (zero_measure_c_point_int == 1) {
            if (pmis_int == 1) {
               // Set as F here but reversed below to become C
               cf_markers_d(i) = -1;
            }
            else {
               // Becomes C
               cf_markers_d(i) = 1;
            }
         }
         else {
            if (pmis_int == 1) {
               // Set as C here but reversed below to become F
               // Otherwise dirichlet conditions persist down onto the coarsest grid
               cf_markers_d(i) = 1;
            }
            else {
               // Becomes F
               cf_markers_d(i) = -1;
            }
         }
         // Count
         update++;
      }
   }, counter_in_set_start);

   // Finish the measure broadcast
   if (mpi)
   {
      PetscCallVoid(PetscSFBcastEnd(sf, MPIU_SCALAR, measure_local_d.data(), measure_nonlocal_d.data(), MPI_REPLACE));
      PetscCallVoid(sf_sync_after_end_kokkos());
   }

   // Check the total number of undecided in parallel
   PetscInt counter_undecided, counter_parallel;
   if (max_luby_steps < 0) {
      counter_undecided = local_rows - counter_in_set_start;
      // Parallel reduction!
      PetscCallMPIAbort(MPI_COMM_MATRIX, MPI_Allreduce(&counter_undecided, &counter_parallel, 1, MPIU_INT, MPI_SUM, MPI_COMM_MATRIX));
      counter_undecided = counter_parallel;

   // If we're doing a fixed number of steps, then we don't care
   // how many undecided nodes we have - have to take care here not to use
   // local_rows for counter_undecided, as we may have zero DOFs on some procs
   // but we have to enter the loop below for the collective scatters
   }
   else {
      counter_undecided = 1;
   }

   // ~~~~~~~~~~~~
   // Now go through the outer Luby loop
   // ~~~~~~~~~~~~

   // Let's keep track of how many times we go through the loops
   int loops_through = -1;
   do
   {
      // Match the fortran version and include a pre-test on the do-while
      if (counter_undecided == 0) break;

      // If max_luby_steps is positive, then we only take that many times through this top loop
      // We typically find 2-3 iterations decides >99% of the nodes
      // and a fixed number of outer loops means we don't have to do any parallel reductions
      // We will do redundant nearest neighbour comms in the case we have already
      // finished deciding all the nodes, but who cares
      // Any undecided nodes just get turned into C points
      // We can do this as we know we won't ruin Aff by doing so, unlike in a normal multigrid
      if (max_luby_steps > 0 && max_luby_steps+1 == -loops_through) break;

      // ~~~~~~~~~
      // Start the broadcast of the nonlocal cf_markers
      // The local veto below only reads the root cf_markers_d, which is allowed
      // while the broadcast is in flight
      // ~~~~~~~~~
      if (mpi) {
         PetscCallVoid(sf_sync_before_begin_kokkos());
         PetscCallVoid(PetscSFBcastWithMemTypeBegin(sf, MPI_INT,
                     PETSC_MEMTYPE_KOKKOS, cf_markers_d.data(),
                     PETSC_MEMTYPE_KOKKOS, cf_markers_nonlocal_d.data(),
                     MPI_REPLACE));
      }

      // ~~~~~~~~
      // Now we use veto to keep track of which candidates can be in the set
      // Locally we know which ones cannot be in the set due to local strong dependencies (mat_local),
      // strong influences (mat_local_transpose), and non-local dependencies (mat_nonlocal)
      // but not the non-local influences as they are stored on many other ranks (ie in S^T)
      // ~~~~~~~~

      // Let's start by veto'ing any candidates that have strong local dependencies or influences
      // using mat_local_spst which is the local S+S^T
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            const PetscInt i = t.league_rank();
            PetscInt strong_neighbours = 0;

            // Check this row is unassigned
            if (cf_markers_d(i) == 0)
            {
               PetscInt ncols_local = device_local_i_spst[i + 1] - device_local_i_spst[i];

               // Reduce over local columns in S+S^T to get the number of strong unassigned influences
               Kokkos::parallel_reduce(
                  Kokkos::TeamThreadRange(t, ncols_local),
                  [&](const PetscInt j, PetscInt& strong_count) {

                  const PetscInt col = device_local_j_spst[device_local_i_spst[i] + j];

                  // Skip the diagonal
                  // Have to only check active strong influences. In single precision
                  // may need a tie break and we use the global index (larger index loses)
                  if ((measure_local_d(i) > measure_local_d(col) || \
                        (measure_local_d(i) == measure_local_d(col) && i > col)) && \
                        cf_markers_d(col) == 0 && col != i)
                  {
                     strong_count++;
                  }

               }, strong_neighbours
               );

               // Only want one thread in the team to write the result
               Kokkos::single(Kokkos::PerTeam(t), [&]() {
                  // If we have any strong neighbours
                  if (strong_neighbours > 0)
                  {
                     veto_local_d(i) = true;
                  }
                  else
                  {
                     veto_local_d(i) = false;
                  }
               });
            }
            // Any that aren't zero cf marker are already assigned so set to true
            else
            {
               // Only want one thread in the team to write the result
               Kokkos::single(Kokkos::PerTeam(t), [&]() {
                  veto_local_d(i) = true;
               });
            }
      });

      // ~~~~~~~~
      // Now let's go through and veto candidates which have strong influences on this rank
      // ie non-local nodes that influence local nodes through S^T
      // ~~~~~~~~
      if (mpi) {

         // Complete the in-flight broadcast before reading the leaf
         PetscCallVoid(PetscSFBcastEnd(sf, MPI_INT, cf_markers_d.data(), cf_markers_nonlocal_d.data(), MPI_REPLACE));
         PetscCallVoid(sf_sync_after_end_kokkos());

         // Initialise to false
         Kokkos::deep_copy(exec, veto_nonlocal_d, false);

         // Let's go and mark any non-local entries that have strong influences and comm to other ranks
         // We iterate over the transpose of the non-local part of S
         // Row k of Ao^T tells us which local rows connect to nonlocal column k
         if (mat_nonlocal_transpose != NULL)
         {
            Kokkos::parallel_for(
               Kokkos::TeamPolicy<>(exec, cols_ao, Kokkos::AUTO()),
               KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

                  // Row
                  const PetscInt i = t.league_rank();
                  PetscInt strong_influences = 0;

                  // Check this row is unassigned
                  if (cf_markers_nonlocal_d(i) == 0)
                  {
                     PetscInt ncols_nonlocal = device_nonlocal_i_transpose[i + 1] - device_nonlocal_i_transpose[i];

                     // Reduce over nonlocal columns in the transpose to get the number of strong unassigned influences
                     Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(t, ncols_nonlocal),
                        [&](const PetscInt j, PetscInt& strong_count) {

                        const PetscInt col = device_nonlocal_j_transpose[device_nonlocal_i_transpose[i] + j];

                        // Have to only check active strong influences. In single
                        // precision may need a tie break using the global index (larger
                        // index loses); the nonlocal node's global index is colmap_d(i)
                        if ((measure_nonlocal_d(i) > measure_local_d(col) || \
                              (measure_nonlocal_d(i) == measure_local_d(col) && colmap_d(i) > global_row_start + col)) && \
                              cf_markers_d(col) == 0)
                        {
                           strong_count++;
                        }

                     }, strong_influences
                     );
                  }

                  // Only want one thread in the team to write the result
                  Kokkos::single(Kokkos::PerTeam(t), [&]() {
                     // If this non-local node has strong influences on this rank it may veto it
                     if (strong_influences > 0) veto_nonlocal_d(i) = true;
                  });
            });
         }

         // Reduce the vetos into veto_local_d with a lor
         PetscCallVoid(sf_sync_before_begin_kokkos());
         PetscCallVoid(PetscSFReduceWithMemTypeBegin(sf, MPI_INT,
                     PETSC_MEMTYPE_KOKKOS, veto_nonlocal_d.data(),
                     PETSC_MEMTYPE_KOKKOS, veto_local_d.data(),
                     MPI_LOR));
         PetscCallVoid(PetscSFReduceEnd(sf, MPI_INT, veto_nonlocal_d.data(), veto_local_d.data(), MPI_LOR));
         PetscCallVoid(sf_sync_after_end_kokkos());

         // Now the comms have finished, we know exactly which local nodes on this rank have no
         // local strong dependencies, influences, non-local influences but not yet non-local dependencies
         // Let's do the non-local dependencies and then now that the comms are done on veto_local_d
         // the combination of both of those gives us all our vetos, so we can assign anything without
         // a veto into the set
         Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

               // Row
               const PetscInt i = t.league_rank();
               PetscInt strong_neighbours = 0;

               // Check this row isn't already marked
               if (!veto_local_d(i))
               {
                  PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];

                  // Reduce over nonlocal columns to get the number of non-local strong unassigned dependencies
                  Kokkos::parallel_reduce(
                     Kokkos::TeamThreadRange(t, ncols_nonlocal),
                     [&](const PetscInt j, PetscInt& strong_count) {

                     // Same deterministic global-index tie break as the local loop;
                     // the non-local neighbour's global index is colmap_d(col)
                     const PetscInt col = device_nonlocal_j[device_nonlocal_i[i] + j];
                     if ((measure_local_d(i) > measure_nonlocal_d(col) || \
                              (measure_local_d(i) == measure_nonlocal_d(col) && global_row_start + i > colmap_d(col))) && \
                              cf_markers_nonlocal_d(col) == 0)
                     {
                        strong_count++;
                     }

                  }, strong_neighbours
                  );

                  // Only want one thread in the team to write the result
                  Kokkos::single(Kokkos::PerTeam(t), [&]() {
                     // If we don't have any non-local strong dependencies and the rest of our vetos are false
                     // we know we are in the set
                     if (strong_neighbours == 0 && !veto_local_d(i)) cf_markers_d(i) = loops_through;
                  });
               }
         });
      }
      // This cf_markers_d(i) = loops_through happens above in the case of mpi, saves a kernel launch
      else
      {
         // The nodes that have .NOT. veto(i) have no strong active neighbours in the IS
         // hence they can be in the IS
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {

               if (!veto_local_d(i)) cf_markers_d(i) = loops_through;
         });
      }

      // ~~~~~~~~~~~~~
      // At this point all the local cf_markers that have been included in the set in this loop are correct
      // We need to set all the strong neighbours of these as not in the set
      // We can do all the local strong dependencies and influences without comms, but we need to do
      // comms to set the non-local strong dependencies and influences
      // ~~~~~~~~~~~~~

      // Now we need to set any non-local dependencies or influences of local nodes added to the set in this loop
      // to be not in the set
      if (mpi)
      {
         // Go and do local strong dependencies and influences via S+S^T
         Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

               // Row
               const PetscInt i = t.league_rank();

               // Check if this node has been assigned during this top loop
               // Atomic as other teams in this kernel store to cf_markers_d
               if (Kokkos::atomic_load(&cf_markers_d(i)) == loops_through)
               {
                  // Do the strong dependencies and influences
                  PetscInt ncols_local = device_local_i_spst[i + 1] - device_local_i_spst[i];

                  Kokkos::parallel_for(
                     Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

                        const PetscInt col = device_local_j_spst[device_local_i_spst[i] + j];

                        // Skip the diagonal - we don't want to mark ourselves as a neighbor
                        // Needs to be atomic as may being set by many threads
                        // Only unassigned neighbours become C, like the CPU version which
                        // never changes the marker of an assigned node
                        if (col != i) Kokkos::atomic_compare_exchange(&cf_markers_d(col), 0, 1);
                  });
               }
         });

         // Now for the influences, we need to broadcast the cf_markers so that
         // on other ranks we know which nodes have cf_markers_nonlocal_d(i) == loops_through
         // The non-local dependencies below only read the root cf_markers_d, so they
         // overlap with the broadcast
         PetscCallVoid(sf_sync_before_begin_kokkos());
         PetscCallVoid(PetscSFBcastWithMemTypeBegin(sf, MPI_INT,
                     PETSC_MEMTYPE_KOKKOS, cf_markers_d.data(),
                     PETSC_MEMTYPE_KOKKOS, cf_markers_nonlocal_d.data(),
                     MPI_REPLACE));

         // We use the veto arrays here to do this comms
         Kokkos::deep_copy(exec, veto_nonlocal_d, 0);
         Kokkos::deep_copy(exec, veto_local_d, 0);

         // Set non-local strong dependencies
         Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

               // Row
               const PetscInt i = t.league_rank();

               // Check if this node has been assigned during this top loop
               if (cf_markers_d(i) == loops_through)
               {
                  PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];

                  // For over nonlocal columns
                  Kokkos::parallel_for(
                     Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {

                        // Needs to be atomic as may being set by many threads
                        // If false set it with true, if not do nothing
                        Kokkos::atomic_compare_exchange(&veto_nonlocal_d(device_nonlocal_j[device_nonlocal_i[i] + j]), 0, 1);
                  });
               }
         });

         // Complete the in-flight broadcast; the leaf is read by the influences below
         PetscCallVoid(PetscSFBcastEnd(sf, MPI_INT, cf_markers_d.data(), cf_markers_nonlocal_d.data(), MPI_REPLACE));
         PetscCallVoid(sf_sync_after_end_kokkos());

         // Reduce the veto_nonlocal_d into veto_local_d with a lor
         // Any local node with veto set to true is not in the set
         PetscCallVoid(sf_sync_before_begin_kokkos());
         PetscCallVoid(PetscSFReduceWithMemTypeBegin(sf, MPI_INT,
                     PETSC_MEMTYPE_KOKKOS, veto_nonlocal_d.data(),
                     PETSC_MEMTYPE_KOKKOS, veto_local_d.data(),
                     MPI_LOR));
         PetscCallVoid(PetscSFReduceEnd(sf, MPI_INT, veto_nonlocal_d.data(), veto_local_d.data(), MPI_LOR));
         PetscCallVoid(sf_sync_after_end_kokkos());

         // Let's finish the non-local dependencies
         // If this node has been veto'd, then set it to not in the set
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
               // Only unassigned nodes become C, like the CPU version
               if (veto_local_d(i) && cf_markers_d(i) == 0) {
                  cf_markers_d(i) = 1;
               }
         });

         // And now we have the information we need to set any of the non-local influences
         if (mat_nonlocal_transpose != NULL)
         {
            Kokkos::parallel_for(
               Kokkos::TeamPolicy<>(exec, cols_ao, Kokkos::AUTO()),
               KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

                  // Row
                  const PetscInt i = t.league_rank();

                  // Check if this node has been assigned during this top loop
                  if (cf_markers_nonlocal_d(i) == loops_through)
                  {
                     PetscInt ncols_nonlocal = device_nonlocal_i_transpose[i + 1] - device_nonlocal_i_transpose[i];

                     // For over nonlocal columns
                     Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {

                           // Needs to be atomic as may being set by many threads
                           // Only unassigned nodes become C, like the CPU version
                           Kokkos::atomic_compare_exchange(&cf_markers_d(device_nonlocal_j_transpose[device_nonlocal_i_transpose[i] + j]), 0, 1);
                     });
                  }
            });
         }
      }
      else
      {
         // In serial we can just update cf_markers_d directly without needing the send buffer
         // Go and do local strong dependencies and influences via S+S^T
         Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

               // Row
               const PetscInt i = t.league_rank();

               // Check if this node has been assigned during this top loop
               // Atomic as other teams in this kernel store to cf_markers_d
               if (Kokkos::atomic_load(&cf_markers_d(i)) == loops_through)
               {
                  // Do the strong dependencies and influences
                  PetscInt ncols_local = device_local_i_spst[i + 1] - device_local_i_spst[i];

                  Kokkos::parallel_for(
                     Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

                        const PetscInt col = device_local_j_spst[device_local_i_spst[i] + j];

                        // Skip the diagonal - we don't want to mark ourselves as a neighbor
                        // Needs to be atomic as may being set by many threads
                        // Only unassigned neighbours become C, like the CPU version which
                        // never changes the marker of an assigned node
                        if (col != i) Kokkos::atomic_compare_exchange(&cf_markers_d(col), 0, 1);
                  });
               }
         });
      }

      // We've done another top level loop
      loops_through = loops_through - 1;

      // ~~~~~~~~~~~~
      // Check the total number of undecided in parallel before we loop again
      // ~~~~~~~~~~~~
      if (max_luby_steps < 0) {

         counter_undecided = 0;
         Kokkos::parallel_reduce ("ReductionCounter_undecided", Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA (const PetscInt i, PetscInt& update) {
            if (cf_markers_d(i) == 0) update++;
         }, counter_undecided);

         // Parallel reduction!
         PetscCallMPIAbort(MPI_COMM_MATRIX, MPI_Allreduce(&counter_undecided, &counter_parallel, 1, MPIU_INT, MPI_SUM, MPI_COMM_MATRIX));
         counter_undecided = counter_parallel;
      }

   }
   while (counter_undecided != 0);

   // Cleanup the local transposes
   if (destroy_spst) PetscCallVoid(MatDestroy(&mat_local_spst));
   if (destroy_nonlocal_transpose) PetscCallVoid(MatDestroy(&mat_nonlocal_transpose));

   // ~~~~~~~~~
   // Now assign our final cf markers
   // ~~~~~~~~~

   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {

      if (cf_markers_d(i) == 0)
      {
         cf_markers_d(i) = 1;
      }
      else if (cf_markers_d(i) < 0)
      {
         cf_markers_d(i) = -1;
      }
      else
      {
         cf_markers_d(i) = 1;
      }
   });
   // Ensure we're done before we exit
   Kokkos::fence();

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// First pass PMISR straight from the level matrix A, without building S, S^T or S+S^T as
// PETSc matrices. Produces the same cf markers as generate_sabs + pmisr_kokkos on any number
// of ranks. Strength is evaluated on the fly with the same test as remove_small_from_sparse_kokkos
// and every edge of the symmetrized strong graph is handled from whichever row stores it, writing
// in both directions, so no local transpose is needed. The strong entries that cross ranks are
// gathered to the rows they point at with a PetscSF, to get the exact degree of the mutual pairs
// Column indices in each row of both blocks are assumed sorted, as elsewhere in PFLARE

// The strength test of remove_small_from_sparse_kokkos, which keeps an entry if abs(val) >= tol
// Every test below goes through this so a NaN is treated the same everywhere
KOKKOS_INLINE_FUNCTION bool pmisr_is_strong(const PetscScalar val, const PetscScalar tol)
{
   return Kokkos::abs(val) >= tol;
}

PETSC_INTERN void pmisr_from_matrix_kokkos(Mat *input_mat, const PetscReal strong_threshold, const int max_luby_steps, PetscReal *measure_local)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start_temp, global_row_end_plus_one_temp, global_col_start_temp, global_col_end_plus_one_temp;
   PetscInt rows_ao = 0, cols_ao = 0;
   MatType mat_type;

   PetscCallVoid(MatGetType(*input_mat, &mat_type));
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat mat_local = NULL, mat_nonlocal = NULL;
   PetscSF sf = NULL;
   const PetscInt *colmap = NULL;

   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*input_mat, &mat_local, &mat_nonlocal, &colmap));
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao));
      PetscCallVoid(MatGetMultPetscSF(*input_mat, &sf));
   }
   else
   {
      mat_local = *input_mat;
   }

   PetscCallVoid(PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX));
   PetscCallVoid(MatGetLocalSize(*input_mat, &local_rows, &local_cols));
   PetscCallVoid(MatGetSize(*input_mat, &global_rows, &global_cols));
   PetscCallVoid(MatGetOwnershipRange(*input_mat, &global_row_start_temp, &global_row_end_plus_one_temp));
   PetscCallVoid(MatGetOwnershipRangeColumn(*input_mat, &global_col_start_temp, &global_col_end_plus_one_temp));
   // This path only sees square level matrices, where row and column ownership match
   // and hence the local index of a column is also the local index of its row
   if (global_row_start_temp != global_col_start_temp || local_rows != local_cols) {
      SETERRABORT(MPI_COMM_MATRIX, PETSC_ERR_ARG_SIZ, "pmisr_from_matrix_kokkos needs a square matrix with matching row and column layouts");
   }
   const PetscInt global_row_start = global_row_start_temp;

   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, NULL, &mtype));
   Kokkos::View<const PetscScalar *> device_local_vals, device_nonlocal_vals;
   PetscCallVoid(MatSeqAIJGetKokkosView(mat_local, &device_local_vals));
   if (mpi) PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal, &device_nonlocal_vals));

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
   PetscScalarKokkosView row_tol_d("row_tol_d", local_rows);
   Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
      PetscScalar max_val = -1.0;
      for (PetscInt k = device_local_i[i]; k < device_local_i[i + 1]; k++) {
         if (device_local_j[k] == i) continue;
         const PetscScalar val = Kokkos::abs(device_local_vals(k));
         if (val > max_val) max_val = val;
      }
      if (mpi) {
         for (PetscInt k = device_nonlocal_i[i]; k < device_nonlocal_i[i + 1]; k++) {
            const PetscScalar val = Kokkos::abs(device_nonlocal_vals(k));
            if (val > max_val) max_val = val;
         }
      }
      row_tol_d(i) = strong_threshold * max_val;
   });

   // ~~~~~~~~~~~~
   // The degree of each node in the symmetrized strong graph, with mutual pairs counted once
   // degree_local_d and degree_nonlocal_d are the row sizes of the local and non-local blocks
   // of S+S^T. They are kept separate so the measure is summed in the same order as pmisr_kokkos
   // ~~~~~~~~~~~~
   intKokkosView degree_local_d("degree_local_d", local_rows);
   intKokkosView degree_nonlocal_d("degree_nonlocal_d", local_rows);
   PetscIntKokkosView strong_nonlocal_d;
   if (mpi) strong_nonlocal_d = PetscIntKokkosView("strong_nonlocal_d", local_rows);

   Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
      int out_minus_mutual = 0;
      for (PetscInt k = device_local_i[i]; k < device_local_i[i + 1]; k++) {
         const PetscInt col = device_local_j[k];
         if (col == i || !pmisr_is_strong(device_local_vals(k), row_tol_d(i))) continue;
         out_minus_mutual++;
         Kokkos::atomic_inc(&degree_local_d(col));
         // Is the reverse entry strong as well? Then this pair was counted twice
         const PetscInt col_start = device_local_i[col];
         const PetscInt pos = binary_search_sorted([&](const PetscInt m) { return device_local_j[col_start + m]; },
                                                   device_local_i[col + 1] - col_start, i);
         if (pos >= 0 && pmisr_is_strong(device_local_vals(col_start + pos), row_tol_d(col))) out_minus_mutual--;
      }
      Kokkos::atomic_add(&degree_local_d(i), out_minus_mutual);

      if (mpi) {
         PetscInt strong_count = 0;
         for (PetscInt k = device_nonlocal_i[i]; k < device_nonlocal_i[i + 1]; k++) {
            if (pmisr_is_strong(device_nonlocal_vals(k), row_tol_d(i))) strong_count++;
         }
         strong_nonlocal_d(i) = strong_count;
         degree_nonlocal_d(i) = static_cast<int>(strong_count);
      }
   });

   // The strong entries in the transpose that cross ranks and their mutual pairs need comms.
   // Each strong non-local entry (i, ghost j) is a leaf of an SF whose root is row j on the rank
   // that owns it, taken from the matrix's own mult SF. The root degree is then the number of
   // strong entries in other ranks' rows that point at that row, and gathering the global row
   // index of each leaf to its root lets the owner find the mutual pairs. The messages only hold
   // the strong entries that cross ranks; the in-degree and two cheap host loops are over the
   // local rows. This replaces building and transposing a parallel matrix on the host
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
         PetscSFNode *ghost_remote = NULL, *edge_remote = NULL;
         PetscCallVoid(PetscMalloc1(cols_ao, &ghost_remote));
         for (PetscInt l = 0; l < sf_nleaves; l++) ghost_remote[sf_ilocal ? sf_ilocal[l] : l] = sf_iremote[l];
         PetscCallVoid(PetscMalloc1(strong_nonlocal_nnz, &edge_remote));
         for (PetscInt k = 0; k < strong_nonlocal_nnz; k++) edge_remote[k] = ghost_remote[ghost_strong_h(k)];
         PetscCallVoid(PetscFree(ghost_remote));

         PetscSF edge_sf;
         PetscCallVoid(PetscSFCreate(MPI_COMM_MATRIX, &edge_sf));
         PetscCallVoid(PetscSFSetGraph(edge_sf, local_rows, strong_nonlocal_nnz, NULL, PETSC_COPY_VALUES, edge_remote, PETSC_OWN_POINTER));
         // The order of the gathered entries at each root doesn't matter
         PetscCallVoid(PetscSFSetRankOrder(edge_sf, PETSC_FALSE));
         PetscCallVoid(PetscSFSetUp(edge_sf));

         const PetscInt *in_degree = NULL;
         PetscCallVoid(PetscSFComputeDegreeBegin(edge_sf, &in_degree));
         PetscCallVoid(PetscSFComputeDegreeEnd(edge_sf, &in_degree));

         // Only the rows that something points at need any work
         PetscInt in_rows = 0, in_nnz = 0;
         for (PetscInt i = 0; i < local_rows; i++) {
            if (in_degree[i] > 0) in_rows++;
            in_nnz += in_degree[i];
         }
         PetscIntKokkosViewHost in_row_h("in_row_h", in_rows), in_offset_h("in_offset_h", in_rows + 1);
         in_offset_h(0) = 0;
         for (PetscInt i = 0, n = 0; i < local_rows; i++) {
            if (in_degree[i] == 0) continue;
            in_row_h(n) = i;
            in_offset_h(n + 1) = in_offset_h(n) + in_degree[i];
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

   // Device memory for the global variable cf_markers_local_d - be careful these aren't petsc ints
   cf_markers_local_d = intKokkosView("cf_markers_local_d", local_rows);
   intKokkosView cf_markers_d = cf_markers_local_d;

   // The measure is random + row sizes of S+S^T, summed in the same order as pmisr_kokkos
   PetscScalarKokkosViewHost measure_local_h(measure_local, local_rows);
   PetscScalarKokkosView measure_local_d("measure_local_d", local_rows);
   Kokkos::deep_copy(exec, measure_local_d, measure_local_h);
   PetscCallVoid(PetscLogCpuToGpu(local_rows * sizeof(PetscReal)));

   Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
      measure_local_d(i) += degree_local_d(i);
      if (mpi) measure_local_d(i) += degree_nonlocal_d(i);
   });

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

   // Nodes with no strong neighbours are F points
   PetscInt counter_in_set_start = 0;
   Kokkos::parallel_reduce("Reduction", Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i, PetscInt &update) {
      if (Kokkos::abs(measure_local_d[i]) < 1)
      {
         cf_markers_d(i) = -1;
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
   // The outer Luby loop. Each edge of the symmetrized strong graph is seen from the row that
   // stores it, which vetoes whichever of its two end points loses. Larger measure loses, with
   // a tie broken on the global index
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

      // Local edges
      Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
         if (cf_markers_d(i) != 0) return;
         for (PetscInt k = device_local_i[i]; k < device_local_i[i + 1]; k++) {
            const PetscInt col = device_local_j[k];
            if (col == i || !pmisr_is_strong(device_local_vals(k), row_tol_d(i)) || cf_markers_d(col) != 0) continue;
            const bool i_loses = measure_local_d(i) > measure_local_d(col) || (measure_local_d(i) == measure_local_d(col) && i > col);
            Kokkos::atomic_store(&veto_d(i_loses ? i : col), 1);
         }
      });

      if (mpi) {
         PetscCallVoid(PetscSFBcastEnd(sf, MPI_INT, cf_markers_d.data(), cf_markers_nonlocal_d.data(), MPI_REPLACE));
         PetscCallVoid(sf_sync_after_end_kokkos());

         // Cross-rank edges: the row side knows its own loser, and vetoes a ghost through the reduce
         Kokkos::deep_copy(exec, veto_nonlocal_d, 0);
         Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
            if (cf_markers_d(i) != 0) return;
            for (PetscInt k = device_nonlocal_i[i]; k < device_nonlocal_i[i + 1]; k++) {
               const PetscInt col = device_nonlocal_j[k];
               if (!pmisr_is_strong(device_nonlocal_vals(k), row_tol_d(i)) || cf_markers_nonlocal_d(col) != 0) continue;
               if (measure_local_d(i) > measure_nonlocal_d(col) || (measure_local_d(i) == measure_nonlocal_d(col) && global_row_start + i > colmap_d(col))) {
                  Kokkos::atomic_store(&veto_d(i), 1);
               } else {
                  Kokkos::atomic_store(&veto_nonlocal_d(col), 1);
               }
            }
         });

         PetscCallVoid(sf_sync_before_begin_kokkos());
         PetscCallVoid(PetscSFReduceWithMemTypeBegin(sf, MPI_INT,
                     PETSC_MEMTYPE_KOKKOS, veto_nonlocal_d.data(),
                     PETSC_MEMTYPE_KOKKOS, veto_d.data(),
                     MPI_LOR));
         PetscCallVoid(PetscSFReduceEnd(sf, MPI_INT, veto_nonlocal_d.data(), veto_d.data(), MPI_LOR));
         PetscCallVoid(sf_sync_after_end_kokkos());
      }

      // Anything not veto'd is in the set
      Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
         if (!veto_d(i)) cf_markers_d(i) = loops_through;
      });

      // ~~~~~~~~
      // Neighbours of nodes that just entered the set become C. Each edge is handled from the
      // row that stores it, again in both directions
      // ~~~~~~~~
      Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
         const int cf_i = Kokkos::atomic_load(&cf_markers_d(i));
         if (cf_i != loops_through && cf_i != 0) return;
         for (PetscInt k = device_local_i[i]; k < device_local_i[i + 1]; k++) {
            const PetscInt col = device_local_j[k];
            if (col == i || !pmisr_is_strong(device_local_vals(k), row_tol_d(i))) continue;
            if (cf_i == loops_through) {
               Kokkos::atomic_compare_exchange(&cf_markers_d(col), 0, 1);
            } else if (Kokkos::atomic_load(&cf_markers_d(col)) == loops_through) {
               Kokkos::atomic_compare_exchange(&cf_markers_d(i), 0, 1);
            }
         }
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
         Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
            if (cf_markers_d(i) != loops_through) return;
            for (PetscInt k = device_nonlocal_i[i]; k < device_nonlocal_i[i + 1]; k++) {
               if (pmisr_is_strong(device_nonlocal_vals(k), row_tol_d(i))) Kokkos::atomic_store(&veto_nonlocal_d(device_nonlocal_j[k]), 1);
            }
         });

         PetscCallVoid(PetscSFBcastEnd(sf, MPI_INT, cf_markers_d.data(), cf_markers_nonlocal_d.data(), MPI_REPLACE));
         PetscCallVoid(sf_sync_after_end_kokkos());

         Kokkos::deep_copy(exec, veto_d, 0);
         PetscCallVoid(sf_sync_before_begin_kokkos());
         PetscCallVoid(PetscSFReduceWithMemTypeBegin(sf, MPI_INT,
                     PETSC_MEMTYPE_KOKKOS, veto_nonlocal_d.data(),
                     PETSC_MEMTYPE_KOKKOS, veto_d.data(),
                     MPI_LOR));

         // Undecided nodes with a ghost in the set become C. This doesn't touch
         // the data in the reduction so it overlaps with it
         Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
            if (cf_markers_d(i) != 0) return;
            for (PetscInt k = device_nonlocal_i[i]; k < device_nonlocal_i[i + 1]; k++) {
               if (pmisr_is_strong(device_nonlocal_vals(k), row_tol_d(i)) && cf_markers_nonlocal_d(device_nonlocal_j[k]) == loops_through) {
                  cf_markers_d(i) = 1;
                  return;
               }
            }
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

   PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local, &device_local_vals));
   if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal, &device_nonlocal_vals));
}

//------------------------------------------------------------------------------------------------------------------------

// PMISR cf splitting but on the device
// This no longer copies back to the host pointer cf_markers_local at the end
// You have to explicitly call copy_cf_markers_d2h(cf_markers_local) to do this
PETSC_INTERN void pmisr_kokkos(Mat *strength_mat, const int max_luby_steps, const int pmis_int, PetscReal *measure_local, const int zero_measure_c_point_int)
{

   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   MatType mat_type;

   PetscCallVoid(MatGetType(*strength_mat, &mat_type));
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat mat_local = NULL, mat_nonlocal = NULL;

   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*strength_mat, &mat_local, &mat_nonlocal, NULL));
   }
   else
   {
      mat_local = *strength_mat;
   }

   // Get the comm
   PetscCallVoid(PetscObjectGetComm((PetscObject)*strength_mat, &MPI_COMM_MATRIX));
   PetscCallVoid(MatGetLocalSize(*strength_mat, &local_rows, &local_cols));
   PetscCallVoid(MatGetSize(*strength_mat, &global_rows, &global_cols));

   // ~~~~~~~~~~~~
   // Get pointers to the i,j on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, NULL, &mtype));

   // Device memory for the global variable cf_markers_local_d - be careful these aren't petsc ints
   cf_markers_local_d = intKokkosView("cf_markers_local_d", local_rows);
   // Can't use the global directly within the parallel
   // regions on the device so just take a shallow copy
   intKokkosView cf_markers_d = cf_markers_local_d;

   // Host and device memory for the measure
   PetscScalarKokkosViewHost measure_local_h(measure_local, local_rows);
   PetscScalarKokkosView measure_local_d("measure_local_d", local_rows);

   auto exec = PetscGetKokkosExecutionSpace();

   // If you want to generate the randoms on the device
   //Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
   // Copy the input measure from host to device
   Kokkos::deep_copy(exec, measure_local_d, measure_local_h);
   // Log copy with petsc
   size_t bytes = measure_local_h.extent(0) * sizeof(PetscReal);
   PetscCallVoid(PetscLogCpuToGpu(bytes));

   // Compute the measure
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {

      // Randoms on the device
      // auto generator = random_pool.get_state();
      // measure_local_d(i) = generator.drand(0., 1.);
      // random_pool.free_state(generator);

      const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];
      measure_local_d(i) += ncols_local;

      if (mpi)
      {
         PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
         measure_local_d(i) += ncols_nonlocal;
      }
      // Flip the sign if pmis
      if (pmis_int == 1) measure_local_d(i) *= -1;
   });

   // The callee obtains the halo SF from strength_mat via MatGetMultPetscSF.
   pmisr_existing_measure_cf_markers_kokkos(strength_mat, max_luby_steps, pmis_int, measure_local_d, cf_markers_d, zero_measure_c_point_int);

   // If PMIS then we swap the CF markers from PMISR
   if (pmis_int) {
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
            cf_markers_d(i) *= -1;
      });
      // Ensure we're done before we exit
      Kokkos::fence();
   }

   return;
}

//------------------------------------------------------------------------------------------------------------------------
