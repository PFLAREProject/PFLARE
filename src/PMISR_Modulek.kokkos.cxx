// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

// The definition of the device copy of the cf markers on a given level
// is stored in Device_Datak.kokkos.cxx and imported as extern from
// kokkos_helper.hpp

//------------------------------------------------------------------------------------------------------------------------

// PMISR implementation that takes an existing measure and cf_markers on the device
// and then does the Luby algorithm to assign the rest of the CF markers
// This mirrors the CPU version pmisr_existing_measure_cf_markers in PMISR_Module.F90
PETSC_INTERN void pmisr_existing_measure_cf_markers_kokkos(Mat *strength_mat, const int max_luby_steps, const int pmis_int, PetscScalarKokkosView &measure_local_d, intKokkosView &cf_markers_d, const int zero_measure_c_point_int)
{
   PflareKokkosTrace _trace("pmisr_existing_measure_cf_markers_kokkos");

   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt rows_ao, cols_ao;
   MatType mat_type;

   PetscCallVoid(MatGetType(*strength_mat, &mat_type));
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat_MPIAIJ *mat_mpi = nullptr;
   Mat mat_local = NULL, mat_nonlocal = NULL;

   if (mpi)
   {
      mat_mpi = (Mat_MPIAIJ *)(*strength_mat)->data;
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*strength_mat, &mat_local, &mat_nonlocal, NULL));
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao));
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
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   Kokkos::fence();
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr, *device_nonlocal_vals = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype));

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   // Validation checks (run once before main algorithm loops)
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   {
      int rank_check;
      MPI_Comm_rank(MPI_COMM_MATRIX, &rank_check);
      bool found_error = false;

      // -- Host-side view extent checks --
      if ((PetscInt)measure_local_d.extent(0) < local_rows) {
         fprintf(stderr, "[PFLARE pmisr check rank=%d] INVALID: measure_local_d extent %zu < local_rows %d\n",
                 rank_check, measure_local_d.extent(0), (int)local_rows);
         found_error = true;
      }
      if ((PetscInt)cf_markers_d.extent(0) < local_rows) {
         fprintf(stderr, "[PFLARE pmisr check rank=%d] INVALID: cf_markers_d extent %zu < local_rows %d\n",
                 rank_check, cf_markers_d.extent(0), (int)local_rows);
         found_error = true;
      }

      if (mpi) {
         // lvec size must match cols_ao
         PetscInt lvec_size;
         PetscCallVoid(VecGetLocalSize(mat_mpi->lvec, &lvec_size));
         if (lvec_size != cols_ao) {
            fprintf(stderr, "[PFLARE pmisr check rank=%d] INVALID: lvec_size %d != cols_ao %d\n",
                    rank_check, (int)lvec_size, (int)cols_ao);
            found_error = true;
         }

         // garray: each entry must be in [0, global_cols) and outside local ownership
         const PetscInt *garray;
         PetscCallVoid(MatMPIAIJGetSeqAIJ(*strength_mat, NULL, NULL, &garray));
         for (PetscInt k = 0; k < cols_ao; k++) {
            if (garray[k] < 0 || garray[k] >= global_cols) {
               fprintf(stderr, "[PFLARE pmisr check rank=%d] INVALID: garray[%d]=%d out of [0, %d)\n",
                       rank_check, (int)k, (int)garray[k], (int)global_cols);
               found_error = true;
            } else if (garray[k] >= global_row_start && garray[k] < global_row_end_plus_one) {
               fprintf(stderr, "[PFLARE pmisr check rank=%d] INVALID: garray[%d]=%d is in local ownership [%d, %d)\n",
                       rank_check, (int)k, (int)garray[k], (int)global_row_start, (int)global_row_end_plus_one);
               found_error = true;
            }
         }
      }

      // -- Device-side CSR checks via Kokkos parallel_reduce --
      auto exec_check = PetscGetKokkosExecutionSpace();

      // Read device_local_i[local_rows] (nnz) to host via a 1-element reduce
      PetscInt nnz_local_check = 0;
      if (local_rows > 0) {
         Kokkos::parallel_reduce(Kokkos::RangePolicy<>(exec_check, 0, 1),
            KOKKOS_LAMBDA(PetscInt, PetscInt& v) { v = device_local_i[local_rows]; },
            Kokkos::Max<PetscInt>(nnz_local_check));
      }

      // Check device_local_i[0] == 0
      if (local_rows > 0) {
         PetscInt local_i_zero;
         Kokkos::parallel_reduce(Kokkos::RangePolicy<>(exec_check, 0, 1),
            KOKKOS_LAMBDA(PetscInt, PetscInt& v) { v = device_local_i[0]; },
            Kokkos::Max<PetscInt>(local_i_zero));
         if (local_i_zero != 0) {
            fprintf(stderr, "[PFLARE pmisr check rank=%d] INVALID: device_local_i[0]=%d != 0\n",
                    rank_check, (int)local_i_zero);
            found_error = true;
         }
      }

      // Check device_local_i is non-decreasing
      if (local_rows > 0) {
         PetscInt mono_err_local = 0;
         Kokkos::parallel_reduce(Kokkos::RangePolicy<>(exec_check, 0, local_rows),
            KOKKOS_LAMBDA(PetscInt i, PetscInt& err) {
               if (device_local_i[i + 1] < device_local_i[i]) err++;
            }, mono_err_local);
         if (mono_err_local > 0) {
            fprintf(stderr, "[PFLARE pmisr check rank=%d] INVALID: device_local_i is non-monotone (%d violations)\n",
                    rank_check, (int)mono_err_local);
            found_error = true;
         }
      }

      // Check device_local_j values are in [0, local_cols)
      if (nnz_local_check > 0) {
         PetscInt j_min_local = local_cols;
         PetscInt j_max_local = -1;
         Kokkos::parallel_reduce(Kokkos::RangePolicy<>(exec_check, 0, nnz_local_check),
            KOKKOS_LAMBDA(PetscInt k, PetscInt& lo) {
               if (device_local_j[k] < lo) lo = device_local_j[k];
            }, Kokkos::Min<PetscInt>(j_min_local));
         Kokkos::parallel_reduce(Kokkos::RangePolicy<>(exec_check, 0, nnz_local_check),
            KOKKOS_LAMBDA(PetscInt k, PetscInt& hi) {
               if (device_local_j[k] > hi) hi = device_local_j[k];
            }, Kokkos::Max<PetscInt>(j_max_local));
         if (j_min_local < 0 || j_max_local >= local_cols) {
            fprintf(stderr, "[PFLARE pmisr check rank=%d] INVALID: device_local_j range [%d, %d] not in [0, %d)\n",
                    rank_check, (int)j_min_local, (int)j_max_local, (int)local_cols);
            found_error = true;
         }
      }

      if (mpi) {
         // Read device_nonlocal_i[local_rows] (nnz) to host
         PetscInt nnz_nonlocal_check = 0;
         if (local_rows > 0) {
            Kokkos::parallel_reduce(Kokkos::RangePolicy<>(exec_check, 0, 1),
               KOKKOS_LAMBDA(PetscInt, PetscInt& v) { v = device_nonlocal_i[local_rows]; },
               Kokkos::Max<PetscInt>(nnz_nonlocal_check));
         }

         // Check device_nonlocal_i[0] == 0
         if (local_rows > 0) {
            PetscInt nonlocal_i_zero;
            Kokkos::parallel_reduce(Kokkos::RangePolicy<>(exec_check, 0, 1),
               KOKKOS_LAMBDA(PetscInt, PetscInt& v) { v = device_nonlocal_i[0]; },
               Kokkos::Max<PetscInt>(nonlocal_i_zero));
            if (nonlocal_i_zero != 0) {
               fprintf(stderr, "[PFLARE pmisr check rank=%d] INVALID: device_nonlocal_i[0]=%d != 0\n",
                       rank_check, (int)nonlocal_i_zero);
               found_error = true;
            }
         }

         // Check device_nonlocal_i is non-decreasing
         if (local_rows > 0) {
            PetscInt mono_err_nonlocal = 0;
            Kokkos::parallel_reduce(Kokkos::RangePolicy<>(exec_check, 0, local_rows),
               KOKKOS_LAMBDA(PetscInt i, PetscInt& err) {
                  if (device_nonlocal_i[i + 1] < device_nonlocal_i[i]) err++;
               }, mono_err_nonlocal);
            if (mono_err_nonlocal > 0) {
               fprintf(stderr, "[PFLARE pmisr check rank=%d] INVALID: device_nonlocal_i is non-monotone (%d violations)\n",
                       rank_check, (int)mono_err_nonlocal);
               found_error = true;
            }
         }

         // Check device_nonlocal_j values are in [0, cols_ao)
         if (nnz_nonlocal_check > 0) {
            PetscInt j_min_nonlocal = cols_ao;
            PetscInt j_max_nonlocal = -1;
            Kokkos::parallel_reduce(Kokkos::RangePolicy<>(exec_check, 0, nnz_nonlocal_check),
               KOKKOS_LAMBDA(PetscInt k, PetscInt& lo) {
                  if (device_nonlocal_j[k] < lo) lo = device_nonlocal_j[k];
               }, Kokkos::Min<PetscInt>(j_min_nonlocal));
            Kokkos::parallel_reduce(Kokkos::RangePolicy<>(exec_check, 0, nnz_nonlocal_check),
               KOKKOS_LAMBDA(PetscInt k, PetscInt& hi) {
                  if (device_nonlocal_j[k] > hi) hi = device_nonlocal_j[k];
               }, Kokkos::Max<PetscInt>(j_max_nonlocal));
            if (j_min_nonlocal < 0 || j_max_nonlocal >= cols_ao) {
               fprintf(stderr, "[PFLARE pmisr check rank=%d] INVALID: device_nonlocal_j range [%d, %d] not in [0, %d)\n",
                       rank_check, (int)j_min_nonlocal, (int)j_max_nonlocal, (int)cols_ao);
               found_error = true;
            }
         }
      }

      fflush(stderr);
      if (found_error) PETSCABORT(MPI_COMM_MATRIX, PETSC_ERR_ARG_WRONG);
   }
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   // Checkpoint helper: fence then print location to stderr so the last printed
   // line before a GPU fault tells us exactly which kernel caused it.
   int rank_cp; MPI_Comm_rank(MPI_COMM_MATRIX, &rank_cp);
   // loops_through is declared here (not inside the do-while) so PMISR_CP can use it pre-loop.
   int loops_through = -1;
#define PMISR_CP(label) do { Kokkos::fence(); \
      fprintf(stderr, "[PFLARE pmisr cp rank=%d iter=%d] " label "\n", rank_cp, loops_through); \
      fflush(stderr); } while(0)

   intKokkosView cf_markers_nonlocal_d;
   // Scratch buffer used for local update bookkeeping during overlap with reverse scatter.
   intKokkosView cf_markers_temp_d;

   PetscScalarKokkosView measure_nonlocal_d;

   if (mpi) {
      measure_nonlocal_d = PetscScalarKokkosView("measure_nonlocal_d", cols_ao);
      cf_markers_nonlocal_d = intKokkosView("cf_markers_nonlocal_d", cols_ao);
      cf_markers_temp_d = intKokkosView("cf_markers_temp_d", local_rows);
   }

   // Device memory for the mark
   boolKokkosView mark_d("mark_d", local_rows);
   auto exec = PetscGetKokkosExecutionSpace();

   // Scatter the measure using VecScatter (matching PETSc's own buffer management)
   if (mpi)
   {
      Vec measure_root_vec;
      PetscCallVoid(MatCreateVecs(*strength_mat, &measure_root_vec, NULL));
      {
         PetscScalarKokkosView root_scalar_d;
         PetscCallVoid(VecGetKokkosViewWrite(measure_root_vec, &root_scalar_d));
         Kokkos::deep_copy(exec, root_scalar_d, measure_local_d);
         PetscCallVoid(VecRestoreKokkosViewWrite(measure_root_vec, &root_scalar_d));
      }
      // Ensure send/receive buffers are stable before Begin.
      Kokkos::fence();      
      PetscCallVoid(VecScatterBegin(mat_mpi->Mvctx, measure_root_vec, mat_mpi->lvec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCallVoid(VecScatterEnd(mat_mpi->Mvctx, measure_root_vec, mat_mpi->lvec, INSERT_VALUES, SCATTER_FORWARD));
      {
         ConstPetscScalarKokkosView lvec_scalar_d;
         PetscCallVoid(VecGetKokkosView(mat_mpi->lvec, &lvec_scalar_d));
         Kokkos::deep_copy(exec, measure_nonlocal_d, lvec_scalar_d);
         PetscCallVoid(VecRestoreKokkosView(mat_mpi->lvec, &lvec_scalar_d));
      }
      PetscCallVoid(VecDestroy(&measure_root_vec));
   }

   // Initialise the set
   PetscInt counter_in_set_start = 0;
   PMISR_CP("A: before initial parallel_reduce");
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
   PMISR_CP("B: after initial parallel_reduce");

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

   // Create reusable Vecs for VecScatter inside the loop (cf_markers int → PetscScalar)
   Vec scatter_root_vec = NULL;
   if (mpi) {
      PetscCallVoid(MatCreateVecs(*strength_mat, &scatter_root_vec, NULL));
   }

   // Let's keep track of how many times we go through the loops
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
      // Start the scatter of the nonlocal cf_markers
      // ~~~~~~~~~
      if (mpi) {
         // Convert int → PetscScalar for VecScatter.
         // We write directly from cf_markers_d; no extra send staging is needed.
         {
            PetscScalarKokkosView root_scalar_d;
      PMISR_CP("C: before fwd scatter kernel");
            PetscCallVoid(VecGetKokkosViewWrite(scatter_root_vec, &root_scalar_d));
            Kokkos::parallel_for(
               Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
                  root_scalar_d(i) = (PetscScalar)cf_markers_d(i);
            });
            PetscCallVoid(VecRestoreKokkosViewWrite(scatter_root_vec, &root_scalar_d));
         }
         // Ensure the root buffer is no longer being written before Begin.
         Kokkos::fence();
         PetscCallVoid(VecScatterBegin(mat_mpi->Mvctx, scatter_root_vec, mat_mpi->lvec, INSERT_VALUES, SCATTER_FORWARD));
         // Complete the in-flight forward scatter before reading the receive buffer.
         PetscCallVoid(VecScatterEnd(mat_mpi->Mvctx, scatter_root_vec, mat_mpi->lvec, INSERT_VALUES, SCATTER_FORWARD));
      }


      // mark_d keeps track of which of the candidate nodes can become in the set
      // Only need this because we want to do async comms so we need a way to trigger
      // a node not being in the set due to either strong local neighbours *or* strong offproc neighbours

      // ~~~~~~~~
      // Go and do the local component
      // ~~~~~~~~
      PMISR_CP("D: before local TeamPolicy kernel");
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

                  // Have to only check active strong neighbours
                  if (measure_local_d(i) >= measure_local_d(device_local_j[device_local_i[i] + j]) && \
                           cf_markers_d(device_local_j[device_local_i[i] + j]) == 0)
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

         // Convert PetscScalar → int after End, when the receive buffer is complete.
         {
            ConstPetscScalarKokkosView leaf_scalar_d;
         PMISR_CP("E: before nonlocal convert kernel");
            PetscCallVoid(VecGetKokkosView(mat_mpi->lvec, &leaf_scalar_d));
            Kokkos::parallel_for(
               Kokkos::RangePolicy<>(exec, 0, cols_ao), KOKKOS_LAMBDA(PetscInt i) {
                  cf_markers_nonlocal_d(i) = (int)leaf_scalar_d(i);
            });
            PetscCallVoid(VecRestoreKokkosView(mat_mpi->lvec, &leaf_scalar_d));
         }

         PMISR_CP("F: before nonlocal TeamPolicy kernel");
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

                     if (measure_local_d(i) >= measure_nonlocal_d(device_nonlocal_j[device_nonlocal_i[i] + j])  && \
                              cf_markers_nonlocal_d(device_nonlocal_j[device_nonlocal_i[i] + j]) == 0)
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
         PMISR_CP("G: before reverse scatter setup");
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
         // Calling a reverse scatter add will then update the values of cf_markers_local
         PMISR_CP("H: before rev scatter convert kernels");
         // Reduce with a sum via VecScatter with ADD_VALUES, SCATTER_REVERSE
         // Convert int → PetscScalar for the leaf (nonlocal) data
         {
            PetscScalarKokkosView leaf_scalar_d;
            PetscCallVoid(VecGetKokkosViewWrite(mat_mpi->lvec, &leaf_scalar_d));
            Kokkos::parallel_for(
               Kokkos::RangePolicy<>(exec, 0, cols_ao), KOKKOS_LAMBDA(PetscInt i) {
                  leaf_scalar_d(i) = (PetscScalar)cf_markers_nonlocal_d(i);
            });
            PetscCallVoid(VecRestoreKokkosViewWrite(mat_mpi->lvec, &leaf_scalar_d));
         }
         // Convert int → PetscScalar for the root (local) data
         {
            PetscScalarKokkosView root_scalar_d;
            PetscCallVoid(VecGetKokkosViewWrite(scatter_root_vec, &root_scalar_d));
            Kokkos::parallel_for(
               Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
                  root_scalar_d(i) = (PetscScalar)cf_markers_d(i);
            });
            PetscCallVoid(VecRestoreKokkosViewWrite(scatter_root_vec, &root_scalar_d));
         }
         // Ensure send/receive buffers are stable before Begin.
         Kokkos::fence();
         fprintf(stderr, "[PFLARE pmisr cp rank=%d iter=%d] I: before rev VecScatterBegin\n", rank_cp, loops_through); fflush(stderr);
         PetscCallVoid(VecScatterBegin(mat_mpi->Mvctx, mat_mpi->lvec, scatter_root_vec, ADD_VALUES, SCATTER_REVERSE));
         // Complete reverse scatter before reading reduced root buffer.
         PetscCallVoid(VecScatterEnd(mat_mpi->Mvctx, mat_mpi->lvec, scatter_root_vec, ADD_VALUES, SCATTER_REVERSE));

         // While reverse scatter is in-flight, do local-only updates in cf_markers_temp_d.
         PMISR_CP("J: before overlap TeamPolicy kernel");
         // This must not touch scatter_root_vec/mat_mpi->lvec.
         Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

               // Row
               const PetscInt i = t.league_rank();

               // Check if this node was assigned during this top loop.
               // We read the temporary buffer here so we do not race with the
               // reduction into cf_markers_d.
               if (cf_markers_temp_d(i) == 2)
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

         // Convert PetscScalar → int back to cf_markers_d after End.
         PMISR_CP("K: before root->cf_markers_d convert");
         {
            ConstPetscScalarKokkosView root_scalar_d;
            PetscCallVoid(VecGetKokkosView(scatter_root_vec, &root_scalar_d));
            Kokkos::parallel_for(
               Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
                  cf_markers_d(i) = (int)root_scalar_d(i);
            });
            PetscCallVoid(VecRestoreKokkosView(scatter_root_vec, &root_scalar_d));
         }

         // Merge the local updates after the VecScatter reduction has completed.
         PMISR_CP("L: before merge kernel");
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
               if (cf_markers_d(i) == loops_through)
               {
                  const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];

                  Kokkos::parallel_for(
                     Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

                        Kokkos::atomic_store(&cf_markers_d(device_local_j[device_local_i[i] + j]), 1);
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
      } else {
         // If we're doing a fixed number of steps, then we need an extra fence
         // as we don't hit the parallel reduce above (which implicitly fences)
         Kokkos::fence();
      }

   }
   while (counter_undecided != 0);

   // Cleanup loop Vecs
   PetscCallVoid(VecDestroy(&scatter_root_vec));

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
   PflareKokkosTrace _trace("pmisr_existing_measure_implicit_transpose_kokkos");

   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt rows_ao, cols_ao;
   MatType mat_type;

   PetscCallVoid(MatGetType(*strength_mat, &mat_type));
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat_MPIAIJ *mat_mpi = nullptr;
   Mat mat_local = NULL, mat_nonlocal = NULL, mat_local_spst = NULL, mat_nonlocal_transpose = NULL;

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

   Mat_SeqAIJKokkos *mat_nonlocal_kok, *mat_local_kok;
   PetscInt zero = 0;
   bool destroy_nonlocal_transpose = false;
   bool destroy_spst = false;

   if (mpi)
   {
      mat_mpi = (Mat_MPIAIJ *)(*strength_mat)->data;
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*strength_mat, &mat_local, &mat_nonlocal, NULL));
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao));
      mat_nonlocal_kok = static_cast<Mat_SeqAIJKokkos *>(mat_nonlocal->spptr);
      // The transpose can crash if mat_nonlocal is empty
      if (mat_nonlocal_kok->csrmat.nnz() > zero)
      {
         PetscCallVoid(MatTranspose(mat_nonlocal, MAT_INITIAL_MATRIX, &mat_nonlocal_transpose));
         destroy_nonlocal_transpose = true;
      }
   }
   else
   {
      mat_local = *strength_mat;
   }
   mat_local_kok = static_cast<Mat_SeqAIJKokkos *>(mat_local->spptr);

   // Get the comm
   PetscCallVoid(PetscObjectGetComm((PetscObject)*strength_mat, &MPI_COMM_MATRIX));
   PetscCallVoid(MatGetLocalSize(*strength_mat, &local_rows, &local_cols));
   PetscCallVoid(MatGetSize(*strength_mat, &global_rows, &global_cols));
   // This returns the global index of the local portion of the matrix
   PetscCallVoid(MatGetOwnershipRange(*strength_mat, &global_row_start, &global_row_end_plus_one));

   // ~~~~~~~~~~~~
   // Form the local S+S^T and get CSR pointers
   // We explicitly compute the local part of S+S^T so we don't have to
   // match the row/column indices - could do this as a symbolic as we don't need the values
   // ~~~~~~~~~~~~
   PetscScalar one = 1.0;
   if (mat_local_kok->csrmat.nnz() > zero)
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
   Kokkos::fence();
   const PetscInt *device_local_i_spst = nullptr, *device_local_j_spst = nullptr;
   const PetscInt *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   const PetscInt *device_nonlocal_i_transpose = nullptr, *device_nonlocal_j_transpose = nullptr;
   PetscMemType mtype;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_spst, &device_local_i_spst, &device_local_j_spst, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, NULL, &mtype));
   if (mpi && mat_nonlocal_transpose != NULL) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_transpose, &device_nonlocal_i_transpose, &device_nonlocal_j_transpose, NULL, &mtype));

   intKokkosView cf_markers_nonlocal_d;
   PetscScalarKokkosView measure_nonlocal_d;

   // ~~~~~~~~~~~~~~~
   // veto stores whether a node has been veto'd as a candidate
   // .NOT. veto(i) means the node can be in the set
   // veto(i) means the node cannot be in the set
   // ~~~~~~~~~~~~~~~
   boolKokkosView veto_local_d("veto_local_d", local_rows);
   boolKokkosView veto_nonlocal_d;

   if (mpi) {
      measure_nonlocal_d = PetscScalarKokkosView("measure_nonlocal_d", cols_ao);
      cf_markers_nonlocal_d = intKokkosView("cf_markers_nonlocal_d", cols_ao);
      veto_nonlocal_d = boolKokkosView("veto_nonlocal_d", cols_ao);
   }

   auto exec = PetscGetKokkosExecutionSpace();

   // Use VecScatter instead of PetscSFBcastWithMemTypeBegin — passing Kokkos GPU view pointers
   // directly to PetscSF causes intermittent failures in parallel GPU runs (exact cause unknown).
   if (mpi)
   {
      Vec measure_root_vec;
      PetscCallVoid(MatCreateVecs(*strength_mat, &measure_root_vec, NULL));
      {
         PetscScalarKokkosView root_scalar_d;
         PetscCallVoid(VecGetKokkosViewWrite(measure_root_vec, &root_scalar_d));
         Kokkos::deep_copy(exec, root_scalar_d, measure_local_d);
         PetscCallVoid(VecRestoreKokkosViewWrite(measure_root_vec, &root_scalar_d));
      }
      // Ensure send/receive buffers are stable before Begin.
      Kokkos::fence();      
      PetscCallVoid(VecScatterBegin(mat_mpi->Mvctx, measure_root_vec, mat_mpi->lvec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCallVoid(VecScatterEnd(mat_mpi->Mvctx, measure_root_vec, mat_mpi->lvec, INSERT_VALUES, SCATTER_FORWARD));
      {
         ConstPetscScalarKokkosView lvec_scalar_d;
         PetscCallVoid(VecGetKokkosView(mat_mpi->lvec, &lvec_scalar_d));
         Kokkos::deep_copy(exec, measure_nonlocal_d, lvec_scalar_d);
         PetscCallVoid(VecRestoreKokkosView(mat_mpi->lvec, &lvec_scalar_d));
      }
      PetscCallVoid(VecDestroy(&measure_root_vec));
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

   // Create reusable Vecs for VecScatter inside the loop
   Vec scatter_root_vec = NULL;
   if (mpi) {
      PetscCallVoid(MatCreateVecs(*strength_mat, &scatter_root_vec, NULL));
   }

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
      // Scatter the nonlocal cf_markers via VecScatter
      // ~~~~~~~~~
      if (mpi) {
         // Convert int → PetscScalar for VecScatter
         {
            PetscScalarKokkosView root_scalar_d;
            PetscCallVoid(VecGetKokkosViewWrite(scatter_root_vec, &root_scalar_d));
            Kokkos::parallel_for(
               Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
                  root_scalar_d(i) = (PetscScalar)cf_markers_d(i);
            });
            PetscCallVoid(VecRestoreKokkosViewWrite(scatter_root_vec, &root_scalar_d));
         }
         // Ensure send/receive buffers are stable before Begin.
         Kokkos::fence();         
         PetscCallVoid(VecScatterBegin(mat_mpi->Mvctx, scatter_root_vec, mat_mpi->lvec, INSERT_VALUES, SCATTER_FORWARD));
         PetscCallVoid(VecScatterEnd(mat_mpi->Mvctx, scatter_root_vec, mat_mpi->lvec, INSERT_VALUES, SCATTER_FORWARD));
         // Convert PetscScalar → int
         {
            ConstPetscScalarKokkosView leaf_scalar_d;
            PetscCallVoid(VecGetKokkosView(mat_mpi->lvec, &leaf_scalar_d));
            Kokkos::parallel_for(
               Kokkos::RangePolicy<>(exec, 0, cols_ao), KOKKOS_LAMBDA(PetscInt i) {
                  cf_markers_nonlocal_d(i) = (int)leaf_scalar_d(i);
            });
            PetscCallVoid(VecRestoreKokkosView(mat_mpi->lvec, &leaf_scalar_d));
         }
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
                  // Have to only check active strong influences
                  if (measure_local_d(i) >= measure_local_d(col) && cf_markers_d(col) == 0 && col != i)
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

                        // Have to only check active strong influences
                        if (measure_nonlocal_d(i) >= measure_local_d(col) && cf_markers_d(col) == 0)
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

         // Reduce the vetos with a lor via VecScatter ADD_VALUES SCATTER_REVERSE
         // (LOR is equivalent to sum when values are 0/1 bools)
         {
            PetscScalarKokkosView leaf_scalar_d;
            PetscCallVoid(VecGetKokkosViewWrite(mat_mpi->lvec, &leaf_scalar_d));
            Kokkos::parallel_for(
               Kokkos::RangePolicy<>(exec, 0, cols_ao), KOKKOS_LAMBDA(PetscInt i) {
                  leaf_scalar_d(i) = veto_nonlocal_d(i) ? 1.0 : 0.0;
            });
            PetscCallVoid(VecRestoreKokkosViewWrite(mat_mpi->lvec, &leaf_scalar_d));
         }
         {
            PetscScalarKokkosView root_scalar_d;
            PetscCallVoid(VecGetKokkosViewWrite(scatter_root_vec, &root_scalar_d));
            Kokkos::parallel_for(
               Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
                  root_scalar_d(i) = veto_local_d(i) ? 1.0 : 0.0;
            });
            PetscCallVoid(VecRestoreKokkosViewWrite(scatter_root_vec, &root_scalar_d));
         }
         // Ensure send/receive buffers are stable before Begin.
         Kokkos::fence();
         PetscCallVoid(VecScatterBegin(mat_mpi->Mvctx, mat_mpi->lvec, scatter_root_vec, ADD_VALUES, SCATTER_REVERSE));
         PetscCallVoid(VecScatterEnd(mat_mpi->Mvctx, mat_mpi->lvec, scatter_root_vec, ADD_VALUES, SCATTER_REVERSE));
         {
            ConstPetscScalarKokkosView root_scalar_d;
            PetscCallVoid(VecGetKokkosView(scatter_root_vec, &root_scalar_d));
            Kokkos::parallel_for(
               Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
                  veto_local_d(i) = root_scalar_d(i) > 0.0;
            });
            PetscCallVoid(VecRestoreKokkosView(scatter_root_vec, &root_scalar_d));
         }

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

                     if (measure_local_d(i) >= measure_nonlocal_d(device_nonlocal_j[device_nonlocal_i[i] + j])  && \
                              cf_markers_nonlocal_d(device_nonlocal_j[device_nonlocal_i[i] + j]) == 0)
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
               if (cf_markers_d(i) == loops_through)
               {
                  // Do the strong dependencies and influences
                  PetscInt ncols_local = device_local_i_spst[i + 1] - device_local_i_spst[i];

                  Kokkos::parallel_for(
                     Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

                        const PetscInt col = device_local_j_spst[device_local_i_spst[i] + j];

                        // Skip the diagonal - we don't want to mark ourselves as a neighbor
                        // Needs to be atomic as may being set by many threads
                        if (cf_markers_d(col) != 1 && col != i)
                        {
                           Kokkos::atomic_store(&cf_markers_d(col), 1);
                        }
                  });
               }
         });

         // Now for the influences, we need to broadcast the cf_markers so that
         // on other ranks we know which nodes have cf_markers_nonlocal_d(i) == loops_through
         {
            PetscScalarKokkosView root_scalar_d;
            PetscCallVoid(VecGetKokkosViewWrite(scatter_root_vec, &root_scalar_d));
            Kokkos::parallel_for(
               Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
                  root_scalar_d(i) = (PetscScalar)cf_markers_d(i);
            });
            PetscCallVoid(VecRestoreKokkosViewWrite(scatter_root_vec, &root_scalar_d));
         }
         // Ensure send/receive buffers are stable before Begin.
         Kokkos::fence();         
         PetscCallVoid(VecScatterBegin(mat_mpi->Mvctx, scatter_root_vec, mat_mpi->lvec, INSERT_VALUES, SCATTER_FORWARD));
         PetscCallVoid(VecScatterEnd(mat_mpi->Mvctx, scatter_root_vec, mat_mpi->lvec, INSERT_VALUES, SCATTER_FORWARD));
         {
            ConstPetscScalarKokkosView leaf_scalar_d;
            PetscCallVoid(VecGetKokkosView(mat_mpi->lvec, &leaf_scalar_d));
            Kokkos::parallel_for(
               Kokkos::RangePolicy<>(exec, 0, cols_ao), KOKKOS_LAMBDA(PetscInt i) {
                  cf_markers_nonlocal_d(i) = (int)leaf_scalar_d(i);
            });
            PetscCallVoid(VecRestoreKokkosView(mat_mpi->lvec, &leaf_scalar_d));
         }

         // We use the veto arrays here to do this comms
         Kokkos::deep_copy(exec, veto_nonlocal_d, false);
         Kokkos::deep_copy(exec, veto_local_d, false);

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
                        Kokkos::atomic_compare_exchange(&veto_nonlocal_d(device_nonlocal_j[device_nonlocal_i[i] + j]), false, true);
                  });
               }
         });

         // Reduce the veto_nonlocal_d with a lor via VecScatter ADD_VALUES SCATTER_REVERSE
         // Any local node with veto set to true is not in the set
         {
            PetscScalarKokkosView leaf_scalar_d;
            PetscCallVoid(VecGetKokkosViewWrite(mat_mpi->lvec, &leaf_scalar_d));
            Kokkos::parallel_for(
               Kokkos::RangePolicy<>(exec, 0, cols_ao), KOKKOS_LAMBDA(PetscInt i) {
                  leaf_scalar_d(i) = veto_nonlocal_d(i) ? 1.0 : 0.0;
            });
            PetscCallVoid(VecRestoreKokkosViewWrite(mat_mpi->lvec, &leaf_scalar_d));
         }
         {
            PetscScalarKokkosView root_scalar_d;
            PetscCallVoid(VecGetKokkosViewWrite(scatter_root_vec, &root_scalar_d));
            Kokkos::parallel_for(
               Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
                  root_scalar_d(i) = veto_local_d(i) ? 1.0 : 0.0;
            });
            PetscCallVoid(VecRestoreKokkosViewWrite(scatter_root_vec, &root_scalar_d));
         }
         // Ensure send/receive buffers are stable before Begin.
         Kokkos::fence();
         PetscCallVoid(VecScatterBegin(mat_mpi->Mvctx, mat_mpi->lvec, scatter_root_vec, ADD_VALUES, SCATTER_REVERSE));
         PetscCallVoid(VecScatterEnd(mat_mpi->Mvctx, mat_mpi->lvec, scatter_root_vec, ADD_VALUES, SCATTER_REVERSE));
         {
            ConstPetscScalarKokkosView root_scalar_d;
            PetscCallVoid(VecGetKokkosView(scatter_root_vec, &root_scalar_d));
            Kokkos::parallel_for(
               Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
                  veto_local_d(i) = root_scalar_d(i) > 0.0;
            });
            PetscCallVoid(VecRestoreKokkosView(scatter_root_vec, &root_scalar_d));
         }

         // Let's finish the non-local dependencies
         // If this node has been veto'd, then set it to not in the set
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
               if (veto_local_d(i)) {
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
                           if (cf_markers_d(device_nonlocal_j_transpose[device_nonlocal_i_transpose[i] + j]) != 1)
                           {
                              Kokkos::atomic_store(&cf_markers_d(device_nonlocal_j_transpose[device_nonlocal_i_transpose[i] + j]), 1);
                           }
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
               if (cf_markers_d(i) == loops_through)
               {
                  // Do the strong dependencies and influences
                  PetscInt ncols_local = device_local_i_spst[i + 1] - device_local_i_spst[i];

                  Kokkos::parallel_for(
                     Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

                        const PetscInt col = device_local_j_spst[device_local_i_spst[i] + j];

                        // Skip the diagonal - we don't want to mark ourselves as a neighbor
                        // Needs to be atomic as may being set by many threads
                        if (cf_markers_d(col) != 1 && col != i)
                        {
                           Kokkos::atomic_store(&cf_markers_d(col), 1);
                        }
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
      } else {
         // If we're doing a fixed number of steps, then we need an extra fence
         // as we don't hit the parallel reduce above (which implicitly fences)
         Kokkos::fence();
      }

   }
   while (counter_undecided != 0);

   // Cleanup loop Vecs
   PetscCallVoid(VecDestroy(&scatter_root_vec));

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

// PMISR cf splitting but on the device
// This no longer copies back to the host pointer cf_markers_local at the end
// You have to explicitly call copy_cf_markers_d2h(cf_markers_local) to do this
PETSC_INTERN void pmisr_kokkos(Mat *strength_mat, const int max_luby_steps, const int pmis_int, PetscReal *measure_local, const int zero_measure_c_point_int)
{
   PflareKokkosTrace _trace("pmisr_kokkos");

   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt rows_ao, cols_ao;
   MatType mat_type;

   PetscCallVoid(MatGetType(*strength_mat, &mat_type));
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat mat_local = NULL, mat_nonlocal = NULL;

   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*strength_mat, &mat_local, &mat_nonlocal, NULL));
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao));
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
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   Kokkos::fence();
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr, *device_nonlocal_vals = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype));

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
   Kokkos::fence();

   // Call the existing measure cf markers function
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
