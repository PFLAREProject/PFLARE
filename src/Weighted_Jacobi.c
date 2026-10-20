/*
  C port of Weighted_Jacobi.F90 (module weighted_jacobi)
*/

#include <petscmat.h>
#include "pflareimpl.h"
#include "Weighted_Jacobi.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode calculate_and_build_weighted_jacobi_inverse(Mat matrix, PetscBool weighted, Mat *inv_matrix)
{
   // Builds an assembled weighted jacobi approximate inverse

   // Local variables
   Mat temp_mat;
   Vec diag_vec;
   PetscReal norm_inf, weight;
   PetscBool reuse_triggered;

   PetscFunctionBegin;

   // Let's create a matrix to represent the inverse diagonal
   reuse_triggered = (*inv_matrix != NULL) ? PETSC_TRUE : PETSC_FALSE;

   if (!reuse_triggered) {
      PetscCall(MatCreateVecs(matrix, NULL, &diag_vec));
   } else {
      PetscCall(MatDiagonalGetDiagonal(*inv_matrix, &diag_vec));
   }
   PetscCall(MatGetDiagonal(matrix, diag_vec));

   // If weighting the Jacobi
   if (weighted) {

      PetscCall(MatDuplicate(matrix, MAT_COPY_VALUES, &temp_mat));

      // D^(1/2)
      // Currently happens on the host
      PetscCall(VecSqrtAbs(diag_vec));
      // D^(-1/2)
      PetscCall(VecReciprocal(diag_vec));
      // D^(-1/2) * A * D^(-1/2)
      PetscCall(MatDiagonalScale(temp_mat, diag_vec, diag_vec));
      // || D^(-1/2) * A * D^(-1/2) ||_inf
      // Currently happens on the host
      PetscCall(MatNorm(temp_mat, NORM_INFINITY, &norm_inf));
      PetscCall(MatDestroy(&temp_mat));

      PetscCall(MatGetDiagonal(matrix, diag_vec));

      // This is the weight that hypre uses, even in assymetric problems
      // 3 / ( 4 * || D^(-1/2) * A * D^(-1/2) ||_inf )
      // (PetscReal literals so a single precision build divides in single, as the Fortran does)
      weight = (PetscReal)3.0 / ((PetscReal)4.0 * norm_inf);

   // Unweighted
   } else {

      weight = 1.0;

   }

   PetscCall(VecReciprocal(diag_vec));
   PetscCall(VecScale(diag_vec, weight));

   // We may be reusing with the same sparsity
   if (!reuse_triggered) {
      // The matrix takes ownership of diag_vec
      PetscCall(MatCreateDiagonal(diag_vec, inv_matrix));
      PetscCall(VecDestroy(&diag_vec));
   } else {
      PetscCall(MatDiagonalRestoreDiagonal(*inv_matrix, &diag_vec));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------
