/*
  C port of SAbs.F90 (module sabs)

  The optional arguments of the Fortran are pointers that are NULL when the
  argument is absent, so present(x) is x != NULL. The matrix the Fortran
  returned in an intent(inout) handle is a Mat *, and the loop over the local
  rows is 0-based
*/

#include <petscmat.h>
#include "pflareimpl.h"
#include "PETSc_Helper.h"
#include "SAbs.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode generate_sabs(Mat input_mat, PetscReal strong_threshold, PetscBool symmetrize, PetscBool square, Mat *output_mat,
   const PetscBool *allow_drop_diagonal, const PetscBool *allow_diag_strength)
{
   // Generate strength of connection matrix with absolute value
   // Output has no diagonal entries

   // ~~~~~~~~~~
   PetscInt ifree;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt global_col_start, global_col_end_plus_one, counter;
   PetscMPIInt comm_size;
   MPI_Comm MPI_COMM_MATRIX;
   Mat transpose_mat;
   IS zero_diags;
   const PetscInt *zero_diags_pointer;
   // The size of zero_diags_pointer, which the Fortran pointer carries
   PetscInt zero_diags_size;
   PetscBool drop_diag, diag_strength;
   // The Fortran passes these literals to the optional arguments of
   // remove_small_from_sparse, which in C take their address
   int relative_max_row_tol_int = -1, drop_diagonal_int = -1, diag_strength_int = 1;
   int drop_diagonal_int_square = 1;
   PetscBool lump = PETSC_FALSE;

   // ~~~~~~~~~~

   PetscFunctionBegin;

   drop_diag = PETSC_TRUE;
   diag_strength = PETSC_FALSE;
   if (allow_drop_diagonal) drop_diag = *allow_drop_diagonal;
   if (allow_diag_strength) diag_strength = *allow_diag_strength;
   // drop_diag is set but never used, as in the Fortran
   (void)drop_diag;

   PetscCall(PetscObjectGetComm((PetscObject)input_mat, &MPI_COMM_MATRIX));
   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));

   // Get the local sizes
   PetscCall(MatGetLocalSize(input_mat, &local_rows, &local_cols));
   PetscCall(MatGetSize(input_mat, &global_rows, &global_cols));
   // This returns the global index of the local portion of the matrix
   PetscCall(MatGetOwnershipRange(input_mat, &global_row_start, &global_row_end_plus_one));
   PetscCall(MatGetOwnershipRangeColumn(input_mat, &global_col_start, &global_col_end_plus_one));

   // Drop entries smaller than the strong_threshold, with a relative tolerance measured
   // against the biggest abs non-diagonal entry, don't lump and always drop the diagonal
   if (!diag_strength) {
      PetscCall(remove_small_from_sparse(input_mat, strong_threshold, output_mat,
            &relative_max_row_tol_int, &lump, &drop_diagonal_int, NULL));
   } else {
      // Measure the strength of connection relative to the diagonal entry,
      // not the max row value excluding the diagonal
      PetscCall(remove_small_from_sparse(input_mat, strong_threshold, output_mat,
            &relative_max_row_tol_int, &lump, &drop_diagonal_int,
            &diag_strength_int));
   }

   // Now symmetrize if desired
   if (symmetrize) {

      // We could just do a symbolic transpose and add the two sets of indices together,
      // but its so much simpler to just add the two together - and the symbolic will be the expensive part
      // anyway
      PetscCall(MatTranspose(*output_mat, MAT_INITIAL_MATRIX, &transpose_mat));
      // Kokkos + MPI doesn't have a gpu mataxpy yet, so we have a wrapper around our own version
      PetscCall(MatAXPYWrapper(*output_mat, PFLARE_ONE, transpose_mat));

      // Don't forget to destroy the explicit transpose
      PetscCall(MatDestroy(&transpose_mat));

   }

   // Square the strength matrix to aggressively coarsen (gives a distance 2 MIS)
   if (square) {

      // Set the entries to 1 before squaring, otherwise the signed values
      // can cancel (e.g., skew-symmetric off-diagonals give S + S^T == 0 and
      // products of mixed sign entries can sum to zero in the square),
      // and those cancelled connections would be dropped below
      PetscCall(MatSetAllValues(*output_mat, PFLARE_ONE));

      if (symmetrize) {
         PetscCall(MatMatMult(*output_mat, *output_mat,
                     MAT_INITIAL_MATRIX, PFLARE_ONE, &transpose_mat));
      } else {
         PetscCall(MatTransposeMatMult(*output_mat, *output_mat,
                     MAT_INITIAL_MATRIX, PFLARE_ONE, &transpose_mat));
      }

      // Also have to add in the original distance 1 connections to the square
      // as the dist 1 strength matrix has had the diagonals removed, so the square won't
      // have the dist 1 connetions in it
      PetscCall(MatAXPYWrapper(transpose_mat, PFLARE_ONE, *output_mat));
      PetscCall(MatDestroy(output_mat));

      // Can end up with diagonal entries we have to remove
      // Let's get the diagonals that are zero or unassigned
      PetscCall(MatFindZeroDiagonals(transpose_mat, &zero_diags));
      PetscCall(ISGetIndices(zero_diags, &zero_diags_pointer));
      PetscCall(ISGetLocalSize(zero_diags, &zero_diags_size));
      // Then let's just set every other row to have a zero diagonal
      // as we know they're already preallocated
      counter = 0;
      for (ifree = 0; ifree < local_rows; ifree++) {

         if (counter < zero_diags_size) {
            // Skip over any rows that don't have diagonals or are already zero
            if (zero_diags_pointer[counter] - global_row_start == ifree) {
               counter = counter + 1;
               continue;
            }
         }

         // Set the diagonal to 0
         PetscCall(MatSetValue(transpose_mat, ifree + global_row_start, ifree + global_row_start,
                  PFLARE_ZERO, INSERT_VALUES));
      }

      PetscCall(ISRestoreIndices(zero_diags, &zero_diags_pointer));
      PetscCall(ISDestroy(&zero_diags));

      PetscCall(MatAssemblyBegin(transpose_mat, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(transpose_mat, MAT_FINAL_ASSEMBLY));

      // Could call MatEliminateZeros in later versions of petsc, but for here
      // given we know the entries are ==1, we will just create a copy with "small" stuff removed
      // ie the zero diagonal
      PetscCall(remove_small_from_sparse(transpose_mat, PFLARE_SENTINEL_DROP, output_mat, NULL, NULL, &drop_diagonal_int_square, NULL));
      PetscCall(MatDestroy(&transpose_mat));

   }

   // Reset the entries in the strength matrix back to 1
   if (symmetrize || square) PetscCall(MatSetAllValues(*output_mat, PFLARE_ONE));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------
