/*
  C port of PMISR_Module.F90 (module pmisr_module)

  The optional arguments of the Fortran are pointers that are NULL when the
  argument is absent, so present(x) is x != NULL. The CF markers are int
  arrays (the Fortran integer) with one entry per local row; cf_markers_local
  is an int ** where the Fortran allocated it (PetscMalloc1, freed with PetscFree).
  The loops over the local rows and the row entries are 0-based, so the
  "+ 1"s the Fortran added to the 0-based CSR indices are gone
*/

#include <petscmat.h>
#include <petscsf.h>
#include "pflareimpl.h"
#include "PETSc_Helper.h"
#include "PMISR_Modulek.h"
#include "Device_Datak.h"
#include "Pflare_Random.h"
#include "PMISR_Module.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode pmisr(Mat strength_mat, int max_luby_steps, PetscBool pmis, int **cf_markers_local, const PetscBool *zero_measure_c_point)
{
   // Wrapper

   // ~~~~~~
#if defined(PETSC_HAVE_KOKKOS)
   MPI_Comm MPI_COMM_MATRIX;
   Mat A_array;
   MatType mat_type;
   int pmis_int, zero_measure_c_point_int;
   PetscMPIInt comm_rank;
   PetscReal *measure_local = NULL;
   PetscInt local_rows, local_cols, ifree;
   PetscReal *measure_local_ptr;
   int *cf_markers_local_ptr;
   int *cf_markers_local_two = NULL;
   PetscBool is_kokkos;
#endif
   // ~~~~~~~~~~

   PetscFunctionBegin;

   PetscCheck(max_luby_steps != 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "max_luby_steps must be positive or negative (0 is invalid)");

#if defined(PETSC_HAVE_KOKKOS)

   PetscCall(MatGetType(strength_mat, &mat_type));
   PetscCall(PetscStrcmpAny(mat_type, &is_kokkos, MATMPIAIJKOKKOS, MATSEQAIJKOKKOS, MATAIJKOKKOS, ""));
   if (is_kokkos) {

      PetscCall(PetscObjectGetComm((PetscObject)strength_mat, &MPI_COMM_MATRIX));
      PetscCallMPI(MPI_Comm_rank(MPI_COMM_MATRIX, &comm_rank));

      A_array = strength_mat;
      pmis_int = 0;
      if (pmis) pmis_int = 1;
      zero_measure_c_point_int = 0;
      if (zero_measure_c_point) {
         if (*zero_measure_c_point) zero_measure_c_point_int = 1;
      }

      // Let's generate the random values on the host for now so they match
      // for comparisons with pmisr_cpu
      PetscCall(MatGetLocalSize(strength_mat, &local_rows, &local_cols));
      // At least one entry, so that like the Fortran's (possibly zero-size) allocation
      // the array the Fortran pflare_random_fill is handed is never NULL, which
      // PetscMalloc1 gives for zero entries
      PetscCall(PetscMalloc1(PetscMax(local_rows, 1), &measure_local));
      // Fill the measure with random numbers
      pflare_random_fill(comm_rank, local_rows, measure_local);

      measure_local_ptr = measure_local;

      PetscCall(PetscMalloc1(local_rows, cf_markers_local));
      cf_markers_local_ptr = *cf_markers_local;

      // Creates a cf_markers on the device
      pmisr_kokkos(&A_array, max_luby_steps, pmis_int, measure_local_ptr, zero_measure_c_point_int);

      // If debugging do a comparison between CPU and Kokkos results
      if (kokkos_debug()) {

         // Kokkos PMISR by default now doesn't copy back to the host, as any following ddc calls
         // use the device data
         copy_cf_markers_d2h(cf_markers_local_ptr);
         PetscCall(pmisr_cpu(strength_mat, max_luby_steps, pmis,
                        &cf_markers_local_two, zero_measure_c_point));

         for (ifree = 0; ifree < local_rows; ifree++) {
            PetscCheck((*cf_markers_local)[ifree] == cf_markers_local_two[ifree], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Kokkos and CPU versions of pmisr do not match");
         }
         PetscCall(PetscFree(cf_markers_local_two));
      }

      // The Fortran deallocates measure_local on return
      PetscCall(PetscFree(measure_local));

   } else {
      PetscCall(pmisr_cpu(strength_mat, max_luby_steps, pmis,
                     cf_markers_local, zero_measure_c_point));
   }
#else
   PetscCall(pmisr_cpu(strength_mat, max_luby_steps, pmis,
                  cf_markers_local, zero_measure_c_point));
#endif

   // ~~~~~~

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode pmisr_cpu(Mat strength_mat, int max_luby_steps, PetscBool pmis, int **cf_markers_local, const PetscBool *zero_measure_c_point)
{
   // Let's do our own independent set with a Luby algorithm
   // If PMIS is true, this is a traditional PMIS algorithm
   // If PMIS is false, this is a PMISR
   // PMISR swaps the C-F definition compared to a PMIS and
   // also checks the measure from smallest, rather than the largest
   // PMISR should give an Aff with no off-diagonal strong connections
   // If you set positive max_luby_steps, it will avoid all parallel reductions
   // by taking a fixed number of times in the Luby top loop

   // ~~~~~~

   // Local
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one, ifree, ncols;
   PetscInt rows_ao, cols_ao, n_ad, n_ao;
   PetscMPIInt comm_size;
   PetscMPIInt comm_rank;
   MPI_Comm MPI_COMM_MATRIX;
   PetscReal *measure_local;
   Mat Ad, Ao = NULL;
   const PetscInt *colmap;
   const PetscInt *ad_ia, *ad_ja, *ao_ia = NULL, *ao_ja = NULL;
   PetscInt shift = 0;
   PetscBool symmetric = PETSC_FALSE, inodecompressed = PETSC_FALSE, done;
   PetscBool zero_measure_c;

   // ~~~~~~

   PetscFunctionBegin;

   // Set here rather than with an initialiser in the declaration, which would give it
   // an implicit save and keep the value from a previous call
   zero_measure_c = PETSC_FALSE;
   if (zero_measure_c_point) zero_measure_c = *zero_measure_c_point;
   // zero_measure_c is set but never used here, as in the Fortran (the optional
   // argument is passed on below)
   (void)zero_measure_c;

   // Get the comm size
   PetscCall(PetscObjectGetComm((PetscObject)strength_mat, &MPI_COMM_MATRIX));
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));
   // Get the comm rank
   PetscCallMPI(MPI_Comm_rank(MPI_COMM_MATRIX, &comm_rank));

   // Get the local sizes
   PetscCall(MatGetLocalSize(strength_mat, &local_rows, &local_cols));
   PetscCall(MatGetSize(strength_mat, &global_rows, &global_cols));
   PetscCall(MatGetOwnershipRange(strength_mat, &global_row_start, &global_row_end_plus_one));

   if (comm_size != 1) {
      PetscCall(MatMPIAIJGetSeqAIJ(strength_mat, &Ad, &Ao, &colmap));
      // We know the col size of Ao is the size of colmap, the number of non-zero offprocessor columns
      PetscCall(MatGetSize(Ao, &rows_ao, &cols_ao));
   } else {
      Ad = strength_mat;
   }

   // ~~~~~~~~
   // Get pointers to the sequential diagonal and off diagonal aij structures
   // ~~~~~~~~
   PetscCall(MatGetRowIJ(Ad, shift, symmetric, inodecompressed, &n_ad, &ad_ia, &ad_ja, &done));
   PetscCheck(done, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Pointers not set in call to MatGetRowIJ");
   if (comm_size != 1) {
      PetscCall(MatGetRowIJ(Ao, shift, symmetric, inodecompressed, &n_ao, &ao_ia, &ao_ja, &done));
      PetscCheck(done, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Pointers not set in call to MatGetRowIJ");
   }
   // ~~~~~~~~~~

   // Get the number of connections in S
   // At least one entry, so that like the Fortran's (possibly zero-size) allocation
   // the array the Fortran pflare_random_fill is handed is never NULL, which
   // PetscMalloc1 gives for zero entries
   PetscCall(PetscMalloc1(PetscMax(local_rows, 1), &measure_local));
   // Allocated and set to zero
   PetscCall(PetscCalloc1(local_rows, cf_markers_local));

   // ~~~~~~~~~~~~
   // Seed the measure_local between 0 and 1
   // ~~~~~~~~~~~~

   // To get the same results regardless of number of processors, you can
   // force the random number on each node to match across all processors
   // This is tricky to do, given the numbering of rows is different in parallel
   // I did code up a version that used the unique spatial node positions to seed the random
   // number generator and test that and it works the same regardless of num of procs
   // so I'm fairly confident things are correct

   // Fill the measure with random numbers
   pflare_random_fill(comm_rank, local_rows, measure_local);

   // ~~~~~~~~~~

   // ~~~~~~~~~~~~
   // Add the number of connections in S to the randomly seeded measure_local
   // The number of connections is just equal to a matvec with a vec of all ones and the strength_mat
   // We don't have to bother with a matvec though as we know the strenth_mat has entries of one
   // ~~~~~~~~~~~~
   for (ifree = 0; ifree < local_rows; ifree++) {

      // Do local component
      ncols = ad_ia[ifree + 1] - ad_ia[ifree];
      measure_local[ifree] = measure_local[ifree] + ncols;

      // Do non local component
      if (comm_size != 1) {
         ncols = ao_ia[ifree + 1] - ao_ia[ifree];
         measure_local[ifree] = measure_local[ifree] + ncols;
      }
   }

   // Restore the sequantial pointers once we're done
   PetscCall(MatRestoreRowIJ(Ad, shift, symmetric, inodecompressed, &n_ad, &ad_ia, &ad_ja, &done));
   if (comm_size != 1) {
      PetscCall(MatRestoreRowIJ(Ao, shift, symmetric, inodecompressed, &n_ao, &ao_ia, &ao_ja, &done));
   }

   // If PMIS then we want to search the measure based on the largest entry
   // PMISR searches the measure based on the smallest entry
   // We just let the measure be negative rather than change the .ge. comparison
   // in our Luby below
   if (pmis) {
      for (ifree = 0; ifree < local_rows; ifree++) measure_local[ifree] = measure_local[ifree] * (-1);
   }

   PetscCall(pmisr_existing_measure_cf_markers(strength_mat,
            max_luby_steps, pmis,
            measure_local, *cf_markers_local, zero_measure_c_point));

   PetscCall(PetscFree(measure_local));
   // If PMIS then we swap the CF markers from PMISR
   if (pmis) {
      for (ifree = 0; ifree < local_rows; ifree++) (*cf_markers_local)[ifree] = (*cf_markers_local)[ifree] * (-1);
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode pmisr_existing_measure_cf_markers(Mat strength_mat, int max_luby_steps, PetscBool pmis,
   PetscReal *measure_local, int *cf_markers_local, const PetscBool *zero_measure_c_point)
{
   // PMISR implementation that takes an existing measure_local and cf_markers_local
   // and then does the Luby algorithm to assign the rest of the CF markers
   //
   // The halo scatters use strength_mat's own mult PetscSF (via
   // MatGetMultPetscSF inside the C scatter routines); we build a matching
   // leaf Vec locally for the measure scatter and destroy it at the end.

   // ~~~~~~

   // Local
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one, ifree;
   PetscInt jfree;
   PetscInt rows_ao, cols_ao = 0, n_ad, n_ao;
   PetscInt counter_undecided, counter_in_set_start, counter_parallel;
   PetscMPIInt comm_size;
   int loops_through;
   PetscMPIInt comm_rank;
   MPI_Comm MPI_COMM_MATRIX;
   PetscBool *in_set_this_loop;
   PetscBool *assigned_local, *assigned_nonlocal;
   // The raw Vec array (PetscScalar) returned by VecGetArrayRead
   const PetscScalar *measure_nonlocal = NULL;
   Mat Ad, Ao = NULL;
   Vec measure_vec = NULL, leaf_vec = NULL;
   PetscSF sf = NULL;
   const PetscInt *colmap = NULL;
   const PetscInt *ad_ia, *ad_ja, *ao_ia = NULL, *ao_ja = NULL;
   PetscInt shift = 0;
   PetscBool symmetric = PETSC_FALSE, inodecompressed = PETSC_FALSE, done;
   PetscBool zero_measure_c;
   PetscInt one = 1;

   // ~~~~~~

   PetscFunctionBegin;

   // Set here rather than with an initialiser in the declaration, which would give it
   // an implicit save and keep the value from a previous call
   zero_measure_c = PETSC_FALSE;
   if (zero_measure_c_point) zero_measure_c = *zero_measure_c_point;

   // Get the comm size
   PetscCall(PetscObjectGetComm((PetscObject)strength_mat, &MPI_COMM_MATRIX));
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));
   // Get the comm rank
   PetscCallMPI(MPI_Comm_rank(MPI_COMM_MATRIX, &comm_rank));

   // Get the local sizes
   PetscCall(MatGetLocalSize(strength_mat, &local_rows, &local_cols));
   PetscCall(MatGetSize(strength_mat, &global_rows, &global_cols));
   PetscCall(MatGetOwnershipRange(strength_mat, &global_row_start, &global_row_end_plus_one));

   if (comm_size != 1) {
      PetscCall(MatMPIAIJGetSeqAIJ(strength_mat, &Ad, &Ao, &colmap));
      // We know the col size of Ao is the size of colmap, the number of non-zero offprocessor columns
      PetscCall(MatGetSize(Ao, &rows_ao, &cols_ao));
   } else {
      Ad = strength_mat;
   }

   // ~~~~~~~~
   // Get pointers to the sequential diagonal and off diagonal aij structures
   // ~~~~~~~~
   PetscCall(MatGetRowIJ(Ad, shift, symmetric, inodecompressed, &n_ad, &ad_ia, &ad_ja, &done));
   PetscCheck(done, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Pointers not set in call to MatGetRowIJ");
   if (comm_size != 1) {
      PetscCall(MatGetRowIJ(Ao, shift, symmetric, inodecompressed, &n_ao, &ao_ia, &ao_ja, &done));
      PetscCheck(done, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Pointers not set in call to MatGetRowIJ");
   }
   // ~~~~~~~~~~

   // Get the number of connections in S
   PetscCall(PetscMalloc1(local_rows, &in_set_this_loop));
   PetscCall(PetscMalloc1(local_rows, &assigned_local));

   // ~~~~~~~~~~~~
   // Create parallel vec and scatter the measure using strength_mat's mult SF
   // and a locally-built leaf Vec (matches the off-diagonal block layout)
   // ~~~~~~~~~~~~
   if (comm_size != 1) {

      // This is fine being mpi type specifically as strength_mat is always a mataij
      PetscCall(VecCreateMPIWithArray(MPI_COMM_MATRIX, one,
         local_rows, global_rows, measure_local, &measure_vec));

      // Leaf Vec sized/typed to match the off-diagonal block
      PetscCall(MatCreateVecs(Ao, &leaf_vec, NULL));

      // Use strength_mat's mult SF for the measure scatter and (below) the
      // boolean halo exchanges. Keep it checked out until cleanup.
      PetscCall(MatGetMultPetscSF(strength_mat, &sf));
      PetscCall(VecScatterBegin(sf, measure_vec, leaf_vec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(sf, measure_vec, leaf_vec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecGetArrayRead(leaf_vec, &measure_nonlocal));

      PetscCall(PetscMalloc1(cols_ao, &assigned_nonlocal));
   } else {
      // Need to avoid uninitialised warning
      PetscCall(PetscMalloc1(0, &assigned_nonlocal));
   }

   // ~~~~~~~~~~~~
   // Initialise the set
   // ~~~~~~~~~~~~
   counter_in_set_start = 0;
   for (ifree = 0; ifree < local_rows; ifree++) assigned_local[ifree] = PETSC_FALSE;
   for (ifree = 0; ifree < cols_ao; ifree++) assigned_nonlocal[ifree] = PETSC_FALSE;

   // If already assigned by the input
   for (ifree = 0; ifree < local_rows; ifree++) {
      if (cf_markers_local[ifree] != 0) assigned_local[ifree] = PETSC_TRUE;
   }

   for (ifree = 0; ifree < local_rows; ifree++) {

      // Skip if already assigned
      if (assigned_local[ifree]) {
         counter_in_set_start = counter_in_set_start + 1;
         continue;
      }

      // If there are no strong neighbours (not measure_local == 0 as we have added a random number to it)
      // then we treat it special
      // Absolute value here given measure_local could be negative (pmis) or positive (pmisr)
      if (PetscAbsReal(measure_local[ifree]) < 1) {

         // Assign this node
         assigned_local[ifree] = PETSC_TRUE;

         // This is typically enabled in a second pass of PMIS just on C points
         // (ie aggressive coarsening based on MIS(MIS(1))), we want to keep
         // C-points with no other strong C connections as C points
         if (zero_measure_c) {
            if (pmis) {
               // Set as F here but reversed below to become C
               cf_markers_local[ifree] = PFLARE_F_POINT;
            } else {
               // Becomes C
               cf_markers_local[ifree] = PFLARE_C_POINT;
            }
         } else {
            if (pmis) {
               // Set as C here but reversed below to become F
               // Otherwise dirichlet conditions persist down onto the coarsest grid
               cf_markers_local[ifree] = PFLARE_C_POINT;
            } else {
               // Becomes F
               cf_markers_local[ifree] = PFLARE_F_POINT;
            }
         }
         counter_in_set_start = counter_in_set_start + 1;
      }
   }

   // Check the total number of undecided in parallel
   if (max_luby_steps < 0) {
      counter_undecided = local_rows - counter_in_set_start;
      // Parallel reduction!
      // This is just an allreduce sum, but we can't use MPIU_INTEGER, as if we call the pmisr
      // cf splitting from C it is not defined - also have to pass the matrix so we can get the comm
      // given they're different in C and fortran
      PetscCallMPI(MPI_Allreduce(&counter_undecided, &counter_parallel, 1, MPIU_INT, MPI_SUM, MPI_COMM_MATRIX));
      counter_undecided = counter_parallel;

   // If we're doing a fixed number of steps, then we don't care
   // how many undecided nodes we have - have to take care here not to use
   // local_rows for counter_undecided, as we may have zero DOFs on some procs
   // but we have to enter the loop below for the collective scatters
   } else {
      counter_undecided = 1;
   }

   // ~~~~~~~~~~~~
   // Now go through the outer Luby loop
   // ~~~~~~~~~~~~

   // Let's keep track of how many times we go through the loops
   loops_through = -1;

   while (counter_undecided != 0) {

      // If max_luby_steps is positive, then we only take that many times through this top loop
      // We typically find 2-3 iterations decides >99% of the nodes
      // and a fixed number of outer loops means we don't have to do any parallel reductions
      // We will do redundant nearest neighbour comms in the case we have already
      // finished deciding all the nodes, but who cares
      // Any undecided nodes just get turned into C points
      // We can do this as we know we won't ruin Aff by doing so, unlike in a normal multigrid
      if (max_luby_steps > 0 && max_luby_steps + 1 == -loops_through) break;

      // ~~~~~~~~~
      // Start the async broadcast of assigned_local to assigned_nonlocal
      // ~~~~~~~~~
      if (comm_size != 1) {
         PetscCall(PetscSFBcastBegin(sf, MPI_C_BOOL, assigned_local, assigned_nonlocal, MPI_REPLACE));
      }

      // Reset in_set_this_loop, which keeps track of which nodes are added to the set this loop
      for (ifree = 0; ifree < local_rows; ifree++) {
         // If they're already assigned they can't be added
         if (assigned_local[ifree]) {
            in_set_this_loop[ifree] = PETSC_FALSE;
         // We assume any unassigned are added to the set this loop and then rule them out below
         } else {
            in_set_this_loop[ifree] = PETSC_TRUE;
         }
      }

      // ~~~~~~~~
      // The Luby algorithm has measure_local(v) > measure_local(u) for all u in active neighbours
      // and then you have to loop from the nodes with biggest measure_local down
      // That is the definition of PMIS
      // PMISR swaps the CF definitions from a traditional PMIS
      // PMISR starts from the smallest measure_local and ensure
      // measure_local(v) < measure_local(u) for all u in active neighbours
      // measure_local is negative for PMIS and positive for PMISR
      // that way we dont have to change the .ge. in the comparison code below
      // ~~~~~~~~

      // ~~~~~~~~
      // Go and do the local component
      // ~~~~~~~~
      for (ifree = 0; ifree < local_rows; ifree++) {

         // Check if this node is already in A
         if (assigned_local[ifree]) continue;
         // Loop over all the active strong neighbours on the local processors
         for (jfree = ad_ia[ifree]; jfree < ad_ia[ifree + 1]; jfree++) {

            // Have to only check unassigned strong neighbours
            if (assigned_local[ad_ja[jfree]]) continue;

            // Check the measure_local. In single precision may need a tie break
            // and we use the global index (larger index loses)
            if (measure_local[ifree] > measure_local[ad_ja[jfree]] ||
                  (measure_local[ifree] == measure_local[ad_ja[jfree]] &&
                   ifree > ad_ja[jfree])) {
               in_set_this_loop[ifree] = PETSC_FALSE;
               // cycle node_loop_local
               break;
            }
         }
      }

      // ~~~~~~~~
      // Finish the async broadcast, assigned_nonlocal is now correct
      // ~~~~~~~~
      if (comm_size != 1) {
         PetscCall(PetscSFBcastEnd(sf, MPI_C_BOOL, assigned_local, assigned_nonlocal, MPI_REPLACE));
      }

      // ~~~~~~~~
      // Now go through and do the non-local part of the matrix
      // ~~~~~~~~
      if (comm_size != 1) {

         for (ifree = 0; ifree < local_rows; ifree++) {

            // Check if already ruled out by local loop or already assigned
            if (assigned_local[ifree] || !in_set_this_loop[ifree]) continue;

            // Loop over all the active strong neighbours on the non-local processors
            for (jfree = ao_ia[ifree]; jfree < ao_ia[ifree + 1]; jfree++) {

               // Have to only check unassigned strong neighbours
               if (assigned_nonlocal[ao_ja[jfree]]) continue;

               // Check the measure_local. Same deterministic global-index tie
               // break as the local loop above; the non-local neighbour's global
               // index is colmap(ao_ja(jfree) + 1)
               if (measure_local[ifree] > PetscRealPart(measure_nonlocal[ao_ja[jfree]]) ||
                     (measure_local[ifree] == PetscRealPart(measure_nonlocal[ao_ja[jfree]]) &&
                      global_row_start + ifree > colmap[ao_ja[jfree]])) {
                  in_set_this_loop[ifree] = PETSC_FALSE;
                  // cycle node_loop
                  break;
               }
            }

         }
      }

      // We now know all nodes which were added to the set this loop, so let's record them
      for (ifree = 0; ifree < local_rows; ifree++) {
         if (in_set_this_loop[ifree]) {
            assigned_local[ifree] = PETSC_TRUE;
            cf_markers_local[ifree] = PFLARE_F_POINT;
         }
      }

      // ~~~~~~~~~~~~~~
      // All the work below here is now to ensure assigned_local is correct for the next iteration
      // Update the nonlocal values first then comm them
      // ~~~~~~~~~~~~~~
      if (comm_size != 1) {

         // We're going to do an LOR reduce so start all as false
         for (ifree = 0; ifree < cols_ao; ifree++) assigned_nonlocal[ifree] = PETSC_FALSE;

         for (ifree = 0; ifree < local_rows; ifree++) {

            // Only need to update neighbours of nodes assigned this top loop
            if (!in_set_this_loop[ifree]) continue;

            // We know all neighbours of points assigned this loop are C points
            // We don't actually need to record that they're C points, just that they're assigned
            for (jfree = ao_ia[ifree]; jfree < ao_ia[ifree + 1]; jfree++) {
               assigned_nonlocal[ao_ja[jfree]] = PETSC_TRUE;
            }
         }

         // ~~~~~~~~~~~
         // We need to start the async reduce LOR of the assigned_nonlocal into assigned_local
         // After this comms finishes any local node in another processors halo
         // that has been assigned on another process will be correctly marked in assigned_local
         // ~~~~~~~~~~~
         // MPIUNI's Fortran headers do not provide MPI_LOR; this branch only
         // runs when comm_size /= 1, i.e. never in a serial MPIUNI build
         PetscCall(PetscSFReduceBegin(sf, MPI_C_BOOL, assigned_nonlocal, assigned_local, MPI_LOR));
      }

      // ~~~~~~~~~~~~~~
      // Now go and update the local values
      // ~~~~~~~~~~~~~~

      for (ifree = 0; ifree < local_rows; ifree++) {

         // Only need to update neighbours of nodes assigned this top loop
         if (!in_set_this_loop[ifree]) continue;

         // Don't need a guard here to check if they're already assigned, as we
         // can guarantee they won't be
         for (jfree = ad_ia[ifree]; jfree < ad_ia[ifree + 1]; jfree++) {
            assigned_local[ad_ja[jfree]] = PETSC_TRUE;
         }
      }

      // ~~~~~~~~~
      // In parallel we have to finish our asyn comms
      // ~~~~~~~~~
      if (comm_size != 1) {
         // Finishes the reduce LOR, assigned_local will now be correct
         PetscCall(PetscSFReduceEnd(sf, MPI_C_BOOL, assigned_nonlocal, assigned_local, MPI_LOR));
      }

      // ~~~~~~~~~~~~
      // We've now done another top level loop
      // ~~~~~~~~~~~~
      loops_through = loops_through - 1;

      // ~~~~~~~~~~~~
      // Check the total number of undecided in parallel before we loop again
      // ~~~~~~~~~~~~
      if (max_luby_steps < 0) {
         // Count how many are undecided
         counter_undecided = local_rows;
         for (ifree = 0; ifree < local_rows; ifree++) {
            if (assigned_local[ifree]) counter_undecided = counter_undecided - 1;
         }
         // Parallel reduction!
         PetscCallMPI(MPI_Allreduce(&counter_undecided, &counter_parallel, 1, MPIU_INT, MPI_SUM, MPI_COMM_MATRIX));
         counter_undecided = counter_parallel;
      }
   }

   // Any unassigned become C points
   for (ifree = 0; ifree < local_rows; ifree++) {
      if (cf_markers_local[ifree] == 0) cf_markers_local[ifree] = PFLARE_C_POINT;
   }

   // ~~~~~~~~~~~~
   // We're finished our IS now
   // ~~~~~~~~~~~~

   // Restore the sequantial pointers once we're done
   PetscCall(MatRestoreRowIJ(Ad, shift, symmetric, inodecompressed, &n_ad, &ad_ia, &ad_ja, &done));
   if (comm_size != 1) {
      PetscCall(MatRestoreRowIJ(Ao, shift, symmetric, inodecompressed, &n_ao, &ao_ia, &ao_ja, &done));
   }

   // ~~~~~~~~~
   // Cleanup
   // ~~~~~~~~~
   PetscCall(PetscFree(in_set_this_loop));
   PetscCall(PetscFree(assigned_local));
   if (comm_size != 1) {
      PetscCall(VecRestoreArrayRead(leaf_vec, &measure_nonlocal));
      PetscCall(VecDestroy(&measure_vec));
      PetscCall(VecDestroy(&leaf_vec));
   }
   PetscCall(PetscFree(assigned_nonlocal));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode pmisr_existing_measure_implicit_transpose(Mat strength_mat, int max_luby_steps, PetscBool pmis,
   PetscReal *measure_local, int *cf_markers_local, const PetscBool *zero_measure_c_point)
{
   // ~~~~~~~~~~~~~~~~~~~~~
   // PMISR implementation that takes an existing measure_local and cf_markers_local
   // and then does the Luby algorithm to assign the rest of the CF markers
   //
   // Unlike pmisr_existing_measure_cf_markers, this routine takes the strength matrix S
   // (not S+S^T) and handles the transpose implicitly. This avoids the expensive explicit
   // formation of S+S^T, particularly in parallel.
   // This only works because you the measure is generated outside
   // of this routine and so can be based on S+S^T, but the Luby loop only needs to know
   // the strong dependencies (from S) and strong influences (from S^T) of each node,
   // not the full S+S^T
   // Unlike pmisr_existing_measure_cf_markers you don't have to have removed the diagonal
   // of S
   //
   // PMISR needs to work with S+S^T to keep out large entries from Aff
   // So instead we do several comms steps in our Luby loop to get/send the data we need
   // We do compute local copies of the transpose of S (which is cheap and local)
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

   // ~~~~~~

   // Local
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one, ifree;
   PetscInt jfree, kfree;
   PetscInt rows_ao, cols_ao = 0, n_ad, n_ao, n_spst, n_aot;
   PetscInt counter_undecided, counter_in_set_start, counter_parallel;
   PetscMPIInt comm_size;
   int loops_through;
   PetscMPIInt comm_rank;
   MPI_Comm MPI_COMM_MATRIX;
   PetscBool *in_set_this_loop;
   PetscBool *assigned_local, *assigned_nonlocal;
   PetscBool *veto_local, *veto_nonlocal;
   // The raw Vec array (PetscScalar) returned by VecGetArrayRead
   const PetscScalar *measure_nonlocal = NULL;
   Mat Ad, Ao = NULL, Ad_spst, Ao_transpose = NULL;
   Vec measure_vec = NULL, leaf_vec = NULL;
   PetscSF sf = NULL;
   const PetscInt *colmap = NULL;
   const PetscInt *ad_ia, *ad_ja, *ao_ia = NULL, *ao_ja = NULL;
   const PetscInt *spst_ia, *spst_ja, *aot_ia = NULL, *aot_ja = NULL;
   PetscInt shift = 0;
   PetscBool symmetric = PETSC_FALSE, inodecompressed = PETSC_FALSE, done;
   PetscBool zero_measure_c;
   PetscBool destroy_spst, destroy_aot;
   PetscInt one = 1;
   PetscReal petsc_one = 1.0;

   // ~~~~~~

   PetscFunctionBegin;

   // Set here rather than with an initialiser in the declaration, which would give it
   // an implicit save and keep the value from a previous call
   zero_measure_c = PETSC_FALSE;
   if (zero_measure_c_point) zero_measure_c = *zero_measure_c_point;

   // Get the comm size
   PetscCall(PetscObjectGetComm((PetscObject)strength_mat, &MPI_COMM_MATRIX));
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));
   // Get the comm rank
   PetscCallMPI(MPI_Comm_rank(MPI_COMM_MATRIX, &comm_rank));

   // Get the local sizes
   PetscCall(MatGetLocalSize(strength_mat, &local_rows, &local_cols));
   PetscCall(MatGetSize(strength_mat, &global_rows, &global_cols));
   PetscCall(MatGetOwnershipRange(strength_mat, &global_row_start, &global_row_end_plus_one));

   if (comm_size != 1) {
      PetscCall(MatMPIAIJGetSeqAIJ(strength_mat, &Ad, &Ao, &colmap));
      // We know the col size of Ao is the size of colmap, the number of non-zero offprocessor columns
      PetscCall(MatGetSize(Ao, &rows_ao, &cols_ao));
   } else {
      Ad = strength_mat;
   }

   // ~~~~~~~~
   // Get pointers to the sequential diagonal and off diagonal aij structures
   // ~~~~~~~~
   PetscCall(MatGetRowIJ(Ad, shift, symmetric, inodecompressed, &n_ad, &ad_ia, &ad_ja, &done));
   PetscCheck(done, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Pointers not set in call to MatGetRowIJ");
   if (comm_size != 1) {
      PetscCall(MatGetRowIJ(Ao, shift, symmetric, inodecompressed, &n_ao, &ao_ia, &ao_ja, &done));
      PetscCheck(done, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Pointers not set in call to MatGetRowIJ");
   }
   // ~~~~~~~~~~

   // ~~~~~~~~
   // Compute local Ad + Ad^T
   // We explicitly compute the local part of S+S^T so we don't have to
   // match the row/column indices in the Luby loop
   // This is cheap as it is purely local (no communication)
   // ~~~~~~~~
   destroy_spst = PETSC_FALSE;
   if (ad_ia[n_ad] > 0) {
      PetscCall(MatTranspose(Ad, MAT_INITIAL_MATRIX, &Ad_spst));
      PetscCall(MatAXPY(Ad_spst, petsc_one, Ad, DIFFERENT_NONZERO_PATTERN));
      destroy_spst = PETSC_TRUE;
   } else {
      Ad_spst = Ad;
   }

   // Get CSR pointers for Ad+Ad^T
   PetscCall(MatGetRowIJ(Ad_spst, shift, symmetric, inodecompressed, &n_spst, &spst_ia, &spst_ja, &done));
   PetscCheck(done, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Pointers not set in call to MatGetRowIJ for Ad_spst");

   // ~~~~~~~~
   // Compute local Ao^T (MPI only)
   // Ao has dimensions [local_rows x cols_ao]
   // Ao^T has dimensions [cols_ao x local_rows]
   // Row k of Ao^T tells us which local rows have connections to nonlocal column k
   // This lets us handle the non-local strong influences without forming the
   // full parallel S+S^T
   // ~~~~~~~~
   destroy_aot = PETSC_FALSE;
   if (comm_size != 1) {
      if (ao_ia[n_ao] > 0) {
         PetscCall(MatTranspose(Ao, MAT_INITIAL_MATRIX, &Ao_transpose));
         destroy_aot = PETSC_TRUE;
         PetscCall(MatGetRowIJ(Ao_transpose, shift, symmetric, inodecompressed, &n_aot, &aot_ia, &aot_ja, &done));
         PetscCheck(done, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Pointers not set in call to MatGetRowIJ for Ao_transpose");
      }
   }

   // ~~~~~~~~~~

   PetscCall(PetscMalloc1(local_rows, &in_set_this_loop));
   PetscCall(PetscMalloc1(local_rows, &assigned_local));
   PetscCall(PetscMalloc1(local_rows, &veto_local));

   // ~~~~~~~~~~~~
   // Create parallel vec and scatter the measure using strength_mat's mult SF
   // and a locally-built leaf Vec (matches the off-diagonal block layout)
   // ~~~~~~~~~~~~
   if (comm_size != 1) {

      // This is fine being mpi type specifically as strength_mat is always a mataij
      PetscCall(VecCreateMPIWithArray(MPI_COMM_MATRIX, one,
         local_rows, global_rows, measure_local, &measure_vec));

      // Leaf Vec sized/typed to match the off-diagonal block
      PetscCall(MatCreateVecs(Ao, &leaf_vec, NULL));

      // Use strength_mat's mult SF for the measure scatter and (below) the
      // boolean halo exchanges. Keep it checked out until cleanup.
      PetscCall(MatGetMultPetscSF(strength_mat, &sf));
      PetscCall(VecScatterBegin(sf, measure_vec, leaf_vec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(sf, measure_vec, leaf_vec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecGetArrayRead(leaf_vec, &measure_nonlocal));

      PetscCall(PetscMalloc1(cols_ao, &assigned_nonlocal));
      PetscCall(PetscMalloc1(cols_ao, &veto_nonlocal));
   } else {
      // Need to avoid uninitialised warning
      PetscCall(PetscMalloc1(0, &assigned_nonlocal));
      PetscCall(PetscMalloc1(0, &veto_nonlocal));
   }

   // ~~~~~~~~~~~~
   // Initialise the set
   // ~~~~~~~~~~~~
   counter_in_set_start = 0;
   for (ifree = 0; ifree < local_rows; ifree++) assigned_local[ifree] = PETSC_FALSE;
   for (ifree = 0; ifree < cols_ao; ifree++) assigned_nonlocal[ifree] = PETSC_FALSE;

   // If already assigned by the input
   for (ifree = 0; ifree < local_rows; ifree++) {
      if (cf_markers_local[ifree] != 0) assigned_local[ifree] = PETSC_TRUE;
   }

   for (ifree = 0; ifree < local_rows; ifree++) {

      // Skip if already assigned
      if (assigned_local[ifree]) {
         counter_in_set_start = counter_in_set_start + 1;
         continue;
      }

      // If there are no strong neighbours (not measure_local == 0 as we have added a random number to it)
      // then we treat it special
      // Absolute value here given measure_local could be negative (pmis) or positive (pmisr)
      if (PetscAbsReal(measure_local[ifree]) < 1) {

         // Assign this node
         assigned_local[ifree] = PETSC_TRUE;

         // This is typically enabled in a second pass of PMIS just on C points
         // (ie aggressive coarsening based on MIS(MIS(1))), we want to keep
         // C-points with no other strong C connections as C points
         if (zero_measure_c) {
            if (pmis) {
               // Set as F here but reversed below to become C
               cf_markers_local[ifree] = PFLARE_F_POINT;
            } else {
               // Becomes C
               cf_markers_local[ifree] = PFLARE_C_POINT;
            }
         } else {
            if (pmis) {
               // Set as C here but reversed below to become F
               // Otherwise dirichlet conditions persist down onto the coarsest grid
               cf_markers_local[ifree] = PFLARE_C_POINT;
            } else {
               // Becomes F
               cf_markers_local[ifree] = PFLARE_F_POINT;
            }
         }
         counter_in_set_start = counter_in_set_start + 1;
      }
   }

   // Check the total number of undecided in parallel
   if (max_luby_steps < 0) {
      counter_undecided = local_rows - counter_in_set_start;
      // Parallel reduction!
      // This is just an allreduce sum, but we can't use MPIU_INTEGER, as if we call the pmisr
      // cf splitting from C it is not defined - also have to pass the matrix so we can get the comm
      // given they're different in C and fortran
      PetscCallMPI(MPI_Allreduce(&counter_undecided, &counter_parallel, 1, MPIU_INT, MPI_SUM, MPI_COMM_MATRIX));
      counter_undecided = counter_parallel;

   // If we're doing a fixed number of steps, then we don't care
   // how many undecided nodes we have - have to take care here not to use
   // local_rows for counter_undecided, as we may have zero DOFs on some procs
   // but we have to enter the loop below for the collective scatters
   } else {
      counter_undecided = 1;
   }

   // ~~~~~~~~~~~~
   // Now go through the outer Luby loop
   // The key difference from pmisr_existing_measure_cf_markers is that we use a
   // veto pattern to handle the implicit transpose. Instead of iterating over
   // neighbours in S+S^T directly, we:
   // 1. Check local dependencies and influences via Ad+Ad^T (computed locally)
   // 2. Check non-local influences via Ao^T + reverse scatter
   // 3. Check non-local dependencies via Ao (same as before)
   // This avoids forming the full parallel S+S^T
   // ~~~~~~~~~~~~

   // Let's keep track of how many times we go through the loops
   loops_through = -1;

   while (counter_undecided != 0) {

      // If max_luby_steps is positive, then we only take that many times through this top loop
      // We typically find 2-3 iterations decides >99% of the nodes
      // and a fixed number of outer loops means we don't have to do any parallel reductions
      // We will do redundant nearest neighbour comms in the case we have already
      // finished deciding all the nodes, but who cares
      // Any undecided nodes just get turned into C points
      // We can do this as we know we won't ruin Aff by doing so, unlike in a normal multigrid
      if (max_luby_steps > 0 && max_luby_steps + 1 == -loops_through) break;

      // ~~~~~~~~~
      // Start the async broadcast of assigned_local to assigned_nonlocal
      // We need assigned_nonlocal for both the non-local dependency check
      // and the non-local influence veto
      // ~~~~~~~~~
      if (comm_size != 1) {
         PetscCall(PetscSFBcastBegin(sf, MPI_C_BOOL, assigned_local, assigned_nonlocal, MPI_REPLACE));
      }

      // ~~~~~~~~
      // Now we use veto to keep track of which candidates can be in the set
      // Locally we know which ones cannot be in the set due to local strong
      // dependencies (from Ad) and strong influences (from Ad^T), combined in Ad+Ad^T
      // but not the non-local influences as they are stored on many other ranks (ie in S^T)
      // ~~~~~~~~

      // ~~~~~~~~
      // Local veto: check neighbours in Ad+Ad^T (covers both local dependencies and influences)
      // ~~~~~~~~
      for (ifree = 0; ifree < local_rows; ifree++) {

         // Already assigned nodes are always vetoed
         if (assigned_local[ifree]) {
            veto_local[ifree] = PETSC_TRUE;
            continue;
         }

         // Assume not vetoed, then check local Ad+Ad^T neighbours
         veto_local[ifree] = PETSC_FALSE;

         for (jfree = spst_ia[ifree]; jfree < spst_ia[ifree + 1]; jfree++) {

            // Skip the diagonal - Ad+Ad^T includes diagonal entries from the original
            // matrix but we only care about off-diagonal strong connections
            if (spst_ja[jfree] == ifree) continue;

            // Have to only check unassigned strong neighbours
            if (assigned_local[spst_ja[jfree]]) continue;

            // Check the measure_local. In single precision may need a tie break
            // and we use the global index (larger index loses)
            if (measure_local[ifree] > measure_local[spst_ja[jfree]] ||
                  (measure_local[ifree] == measure_local[spst_ja[jfree]] &&
                   ifree > spst_ja[jfree])) {
               veto_local[ifree] = PETSC_TRUE;
               // cycle node_loop_local
               break;
            }
         }
      }

      // ~~~~~~~~
      // Finish the async broadcast, assigned_nonlocal is now correct
      // ~~~~~~~~
      if (comm_size != 1) {
         PetscCall(PetscSFBcastEnd(sf, MPI_C_BOOL, assigned_local, assigned_nonlocal, MPI_REPLACE));
      }

      // ~~~~~~~~
      // Non-local influence veto using Ao^T
      // For each nonlocal column k, Ao^T row k tells us which local rows
      // have connections TO nonlocal node k. If nonlocal node k is unassigned
      // and has measure >= any of those local rows' measures, then nonlocal node k
      // has a local influence that vetoes it.
      // We set veto_nonlocal(k) = TRUE which will be reverse scattered back to
      // the owning processor to veto their local node.
      // ~~~~~~~~
      if (comm_size != 1) {

         for (kfree = 0; kfree < cols_ao; kfree++) veto_nonlocal[kfree] = PETSC_FALSE;

         if (destroy_aot) {
            for (kfree = 0; kfree < cols_ao; kfree++) {

               // Only check unassigned nonlocal nodes
               if (assigned_nonlocal[kfree]) continue;

               for (jfree = aot_ia[kfree]; jfree < aot_ia[kfree + 1]; jfree++) {

                  // The column index in Ao^T is a local row index
                  // Have to only check unassigned local rows
                  if (assigned_local[aot_ja[jfree]]) continue;

                  // If the nonlocal node's measure >= local row's measure,
                  // the nonlocal node is vetoed by this local influence. In single
                  // precision may need a tie break using the global index (larger
                  // index loses); the nonlocal node's global index is colmap(kfree)
                  if (PetscRealPart(measure_nonlocal[kfree]) > measure_local[aot_ja[jfree]] ||
                        (PetscRealPart(measure_nonlocal[kfree]) == measure_local[aot_ja[jfree]] &&
                         colmap[kfree] > global_row_start + aot_ja[jfree])) {
                     veto_nonlocal[kfree] = PETSC_TRUE;
                     break;
                  }
               }
            }
         }

         // ~~~~~~~~
         // Reverse scatter the veto: veto_nonlocal → veto_local with LOR
         // After this, veto_local(i) is TRUE if any non-local transpose neighbour
         // vetoes local node i
         // ~~~~~~~~
         PetscCall(PetscSFReduceBegin(sf, MPI_C_BOOL, veto_nonlocal, veto_local, MPI_LOR));
         // Not sure we have any chance to overlap this with anything else
         PetscCall(PetscSFReduceEnd(sf, MPI_C_BOOL, veto_nonlocal, veto_local, MPI_LOR));

         // ~~~~~~~~
         // Now the comms have finished, we know exactly which local nodes on this rank have no
         // local strong dependencies, influences, non-local influences but not yet non-local
         // dependencies
         // Let's do the non-local dependencies and then now that the comms are done on veto_local
         // the combination of both of those gives us all our vetos, so we can assign anything
         // without a veto into the set
         // ~~~~~~~~
         for (ifree = 0; ifree < local_rows; ifree++) {

            // Check if already vetoed (by local check or reverse scatter) or already assigned
            if (veto_local[ifree]) continue;

            // Loop over all the active strong neighbours on the non-local processors
            for (jfree = ao_ia[ifree]; jfree < ao_ia[ifree + 1]; jfree++) {

               // Have to only check unassigned strong neighbours
               if (assigned_nonlocal[ao_ja[jfree]]) continue;

               // Check the measure_local. Same deterministic global-index tie
               // break as above; the non-local neighbour's global index is
               // colmap(ao_ja(jfree) + 1)
               if (measure_local[ifree] > PetscRealPart(measure_nonlocal[ao_ja[jfree]]) ||
                     (measure_local[ifree] == PetscRealPart(measure_nonlocal[ao_ja[jfree]]) &&
                      global_row_start + ifree > colmap[ao_ja[jfree]])) {
                  veto_local[ifree] = PETSC_TRUE;
                  // cycle node_loop
                  break;
               }
            }

         }
      }

      // ~~~~~~~~
      // We now know all nodes which were added to the set this loop
      // Nodes with veto_local = FALSE are in the set
      // Record them in cf_markers and assigned_local, and track which were
      // just assigned for the neighbour marking phase
      // ~~~~~~~~
      for (ifree = 0; ifree < local_rows; ifree++) {
         if (!veto_local[ifree]) {
            in_set_this_loop[ifree] = PETSC_TRUE;
            assigned_local[ifree] = PETSC_TRUE;
            cf_markers_local[ifree] = PFLARE_F_POINT;
         } else {
            in_set_this_loop[ifree] = PETSC_FALSE;
         }
      }

      // ~~~~~~~~~~~~~~
      // All the work below here is now to ensure assigned_local is correct for the next iteration
      // We need to mark all neighbours of just-assigned nodes as assigned (C points)
      // This has four components:
      // 1. Local dependencies and influences via Ad+Ad^T
      // 2. Non-local dependencies via Ao + reverse scatter (existing pattern)
      // 3. Non-local influences via forward scatter of in_set + Ao^T
      // ~~~~~~~~~~~~~~

      // ~~~~~~~~~~~~~~
      // 1. Local: mark Ad+Ad^T neighbours of just-assigned nodes
      // This covers both local strong dependencies and local strong influences
      // ~~~~~~~~~~~~~~
      for (ifree = 0; ifree < local_rows; ifree++) {

         // Only need to update neighbours of nodes assigned this top loop
         if (!in_set_this_loop[ifree]) continue;

         // Don't need a guard here to check if they're already assigned, as we
         // can guarantee they won't be
         for (jfree = spst_ia[ifree]; jfree < spst_ia[ifree + 1]; jfree++) {
            // Skip the diagonal
            if (spst_ja[jfree] == ifree) continue;
            assigned_local[spst_ja[jfree]] = PETSC_TRUE;
         }
      }

      if (comm_size != 1) {

         // ~~~~~~~~~~~~~~
         // 2. Non-local dependencies: for each just-assigned local node, mark its
         // Ao neighbours as assigned on the owning processor via reverse scatter
         // This tells remote processors: "your local row is now assigned because
         // it is a forward neighbour of one of my just-assigned nodes"
         // ~~~~~~~~~~~~~~

         // We reuse veto_nonlocal for the reverse scatter
         for (kfree = 0; kfree < cols_ao; kfree++) veto_nonlocal[kfree] = PETSC_FALSE;

         for (ifree = 0; ifree < local_rows; ifree++) {

            // Only need to update neighbours of nodes assigned this top loop
            if (!in_set_this_loop[ifree]) continue;

            // We know all neighbours of points assigned this loop are C points
            // We don't actually need to record that they're C points, just that they're assigned
            for (jfree = ao_ia[ifree]; jfree < ao_ia[ifree + 1]; jfree++) {
               veto_nonlocal[ao_ja[jfree]] = PETSC_TRUE;
            }
         }

         // ~~~~~~~~~~~
         // Reduce LOR of veto_nonlocal into assigned_local
         // After this comms finishes any local node in another processors halo
         // that has been assigned on another process will be correctly marked in assigned_local
         // ~~~~~~~~~~~
         PetscCall(PetscSFReduceBegin(sf, MPI_C_BOOL, veto_nonlocal, assigned_local, MPI_LOR));
         PetscCall(PetscSFReduceEnd(sf, MPI_C_BOOL, veto_nonlocal, assigned_local, MPI_LOR));

         // ~~~~~~~~~~~~~~
         // 3. Non-local influences: we need to know which nonlocal nodes were just
         // assigned this loop so we can mark their local transpose neighbours
         // Forward scatter in_set_this_loop to learn which nonlocal columns were just assigned
         // Then iterate over Ao^T to mark local rows that are influenced by those nodes
         // ~~~~~~~~~~~~~~

         // Reuse veto_nonlocal to receive the forward scatter result
         for (kfree = 0; kfree < cols_ao; kfree++) veto_nonlocal[kfree] = PETSC_FALSE;
         PetscCall(PetscSFBcastBegin(sf, MPI_C_BOOL, in_set_this_loop, veto_nonlocal, MPI_REPLACE));
         PetscCall(PetscSFBcastEnd(sf, MPI_C_BOOL, in_set_this_loop, veto_nonlocal, MPI_REPLACE));

         // Now veto_nonlocal(k) = TRUE means the remote node at nonlocal column k
         // was just assigned this loop on its owning processor
         // All local rows that connect to that nonlocal node via Ao (i.e. Ao^T row k)
         // should be marked as assigned
         if (destroy_aot) {
            for (kfree = 0; kfree < cols_ao; kfree++) {
               if (!veto_nonlocal[kfree]) continue;

               for (jfree = aot_ia[kfree]; jfree < aot_ia[kfree + 1]; jfree++) {
                  assigned_local[aot_ja[jfree]] = PETSC_TRUE;
               }
            }
         }

      }

      // ~~~~~~~~~~~~
      // We've now done another top level loop
      // ~~~~~~~~~~~~
      loops_through = loops_through - 1;

      // ~~~~~~~~~~~~
      // Check the total number of undecided in parallel before we loop again
      // ~~~~~~~~~~~~
      if (max_luby_steps < 0) {
         // Count how many are undecided
         counter_undecided = local_rows;
         for (ifree = 0; ifree < local_rows; ifree++) {
            if (assigned_local[ifree]) counter_undecided = counter_undecided - 1;
         }
         // Parallel reduction!
         PetscCallMPI(MPI_Allreduce(&counter_undecided, &counter_parallel, 1, MPIU_INT, MPI_SUM, MPI_COMM_MATRIX));
         counter_undecided = counter_parallel;
      }
   }

   // Any unassigned become C points
   for (ifree = 0; ifree < local_rows; ifree++) {
      if (cf_markers_local[ifree] == 0) cf_markers_local[ifree] = PFLARE_C_POINT;
   }

   // ~~~~~~~~~~~~
   // We're finished our IS now
   // ~~~~~~~~~~~~

   // Restore the sequential pointers once we're done
   PetscCall(MatRestoreRowIJ(Ad_spst, shift, symmetric, inodecompressed, &n_spst, &spst_ia, &spst_ja, &done));
   PetscCall(MatRestoreRowIJ(Ad, shift, symmetric, inodecompressed, &n_ad, &ad_ia, &ad_ja, &done));
   if (comm_size != 1) {
      PetscCall(MatRestoreRowIJ(Ao, shift, symmetric, inodecompressed, &n_ao, &ao_ia, &ao_ja, &done));
      if (destroy_aot) {
         PetscCall(MatRestoreRowIJ(Ao_transpose, shift, symmetric, inodecompressed, &n_aot, &aot_ia, &aot_ja, &done));
      }
   }

   // ~~~~~~~~~
   // Cleanup
   // ~~~~~~~~~
   if (destroy_spst) PetscCall(MatDestroy(&Ad_spst));
   if (destroy_aot) PetscCall(MatDestroy(&Ao_transpose));

   PetscCall(PetscFree(in_set_this_loop));
   PetscCall(PetscFree(assigned_local));
   PetscCall(PetscFree(veto_local));
   if (comm_size != 1) {
      PetscCall(VecRestoreArrayRead(leaf_vec, &measure_nonlocal));
      PetscCall(VecDestroy(&measure_vec));
      PetscCall(VecDestroy(&leaf_vec));
   }
   PetscCall(PetscFree(assigned_nonlocal));
   PetscCall(PetscFree(veto_nonlocal));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------
