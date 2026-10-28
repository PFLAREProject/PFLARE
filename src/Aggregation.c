/*
  C port of Aggregation.F90 (module aggregation)

  The Fortran allocatable arrays are pointers: *cf_markers == NULL is the
  Fortran .not. allocated(cf_markers). Both arrays are allocated with
  PetscMalloc1 and freed with PetscFree. The loops are 0-based; the aggregate
  numbers stored in aggregates start at 1 as in the Fortran
*/

#include <petscmat.h>
#include "pflareimpl.h"
#include "Aggregation.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode generate_serial_aggregation(Mat strength_mat, int **cf_markers, PetscInt **aggregates)
{
   // Do an aggregation algorithm - this is currently an exact copy of the aggregation
   // algorithm in PyAMG
   // Only in serial

   // Local
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt a_global_row_start, a_global_row_end_plus_one, ifree, ncols, kfree;
   PetscInt aggregate, max_neighbour_index, max_neighbour_value, jfree;
   PetscMPIInt comm_size;
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt *indices;
   const PetscInt *cols = NULL;
   PetscBool mark_neigh;

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)strength_mat, &MPI_COMM_MATRIX));
   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));
   // Serial only!
   PetscCheck(comm_size == 1, MPI_COMM_MATRIX, PETSC_ERR_SUP, "Fix me - aggregation algorithm is serial only");

   // Get the local sizes
   PetscCall(MatGetLocalSize(strength_mat, &local_rows, &local_cols));
   PetscCall(MatGetSize(strength_mat, &global_rows, &global_cols));
   PetscCall(MatGetOwnershipRange(strength_mat, &a_global_row_start, &a_global_row_end_plus_one));

   // Get the number of connections in S
   PetscCall(PetscMalloc1(local_rows, &indices));
   PetscCall(PetscCalloc1(local_rows, aggregates));

   // If we've passed in a partially completed cf splitting
   if (*cf_markers) {
      // we assign them a negative aggregate, which stops any new
      // points being added to those "aggregates" in step 2
      for (ifree = 0; ifree < local_rows; ifree++) {
         if ((*cf_markers)[ifree] != 0) {
            (*aggregates)[ifree] = -1;
         }
      }
   } else {
      PetscCall(PetscCalloc1(local_rows, cf_markers));
   }

   // Serial ordering to match pyamg in serial
   for (ifree = 0; ifree < local_rows; ifree++) {
      indices[ifree] = ifree;
   }

   aggregate = 1;

   // Loop over all the rows - Step 1 - initial covering
   for (ifree = 0; ifree < local_rows; ifree++) {

      // Get S_i - distance 1 neighbours
      PetscCall(MatGetRow(strength_mat, a_global_row_start + indices[ifree], &ncols, &cols, NULL));

      if (ncols == 0) {

         // No neighbours it stays fine, unless it came in pre-assigned
         // (e.g., a C point from the boundary PMIS in PMIS_AGG whose strong
         // neighbours are all off-process), in which case we leave it alone
         if ((*cf_markers)[indices[ifree]] == 0) (*cf_markers)[indices[ifree]] = -1;

      } else {

         // Check if this node or any of the strongly connected neighbours are already in an aggregate
         mark_neigh = PETSC_FALSE;
         for (jfree = 0; jfree < ncols; jfree++) {
            if ((*cf_markers)[cols[jfree]] != 0 || (*cf_markers)[indices[ifree]] != 0) mark_neigh = PETSC_TRUE;
         }

         // Skip if this node or any strong neighbours are already in an aggregate
         if (!mark_neigh) {

            // This is the root node
            (*cf_markers)[indices[ifree]] = 1;
            (*aggregates)[indices[ifree]] = aggregate;
            // It's neighbours are fine
            for (jfree = 0; jfree < ncols; jfree++) {
               (*cf_markers)[cols[jfree]] = -1;
               (*aggregates)[cols[jfree]] = aggregate;
            }
            // Advance to a new aggregate
            aggregate = aggregate + 1;
         }

      }

      PetscCall(MatRestoreRow(strength_mat, a_global_row_start + indices[ifree], &ncols, &cols, NULL));

   }

   // Step 2 - enlarging the sets
   for (ifree = 0; ifree < local_rows; ifree++) {

      // Check if this node has been assigned
      if ((*cf_markers)[indices[ifree]] != 0) continue;

      // Get S_i - distance 1 neighbours
      PetscCall(MatGetRow(strength_mat, a_global_row_start + indices[ifree], &ncols, &cols, NULL));

      max_neighbour_index = -1;
      max_neighbour_value = 0;
      // max_neighbour_value is set but never read, as in the Fortran
      (void)max_neighbour_value;

      // Loop over strongly connected neighbours
      // j_loop
      for (jfree = 0; jfree < ncols; jfree++) {

         // Have we hit a strongly connected neighbour in an aggregate
         // that wasn't added as part of stage 2
         if ((*aggregates)[cols[jfree]] <= 0) continue;
         // The pyamg version doesn't test for the actual strength of this connection
         // It just finds the first strongly connected aggregate neighbour and appends to that
         // k_loop
         for (kfree = 0; kfree < ncols; kfree++) {
            if (cols[kfree] == cols[jfree]) {
               max_neighbour_index = jfree;
               break;
            }
         }
         // exit j_loop once the k_loop has found a match
         if (max_neighbour_index != -1) break;
      }

      // If we didn't have a strongly connected neighbour, cycle
      if (max_neighbour_index != -1) {

         // Otherwise we have identified the strongest connection this unassigned node has
         // And so this node becomes part of that aggregate (ie an F point)
         (*cf_markers)[indices[ifree]] = -1;
         // Make this negative to record that it was added as part of stage 2
         (*aggregates)[indices[ifree]] = -(*aggregates)[cols[max_neighbour_index]];

         // Advance to a new aggregate
         aggregate = aggregate + 1;
      }

      PetscCall(MatRestoreRow(strength_mat, a_global_row_start + indices[ifree], &ncols, &cols, NULL));
   }

   // Swap the negative ones back to positive
   for (ifree = 0; ifree < local_rows; ifree++) {
      if ((*aggregates)[ifree] < 0) (*aggregates)[ifree] = (*aggregates)[ifree] * (-1);
   }

   // Step 3 - any remnants
   for (ifree = 0; ifree < local_rows; ifree++) {

      // Check if this node has been assigned
      if ((*cf_markers)[indices[ifree]] != 0) continue;

      // Get S_i - distance 1 neighbours
      PetscCall(MatGetRow(strength_mat, a_global_row_start + indices[ifree], &ncols, &cols, NULL));

      // This is the root node
      (*cf_markers)[indices[ifree]] = 1;
      (*aggregates)[indices[ifree]] = aggregate;

      // Any unassigned strong neighbours are fine and in this aggregate
      for (jfree = 0; jfree < ncols; jfree++) {
         if ((*cf_markers)[cols[jfree]] != 0) continue;
         (*cf_markers)[cols[jfree]] = -1;
         (*aggregates)[cols[jfree]] = aggregate;
      }
      // Advance to a new aggregate
      aggregate = aggregate + 1;

      PetscCall(MatRestoreRow(strength_mat, a_global_row_start + indices[ifree], &ncols, &cols, NULL));
   }

   PetscCall(PetscFree(indices));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------
