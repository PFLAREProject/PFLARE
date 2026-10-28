/*
  C port of Weighted_Jacobi.F90 (module weighted_jacobi)
*/

#include <petscmat.h>
#include "pflareimpl.h"
#include "Weighted_Jacobi.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode calculate_and_build_weighted_jacobi_inverse(Mat matrix, int inverse_type, Mat *inv_matrix)
{
   // Builds an assembled Jacobi approximate inverse, either weighted, unweighted or l1
   // inverse_type is one of PFLAREINV_WJACOBI, PFLAREINV_JACOBI or PFLAREINV_L1JACOBI

   // Local variables
   Mat temp_mat;
   Vec diag_vec;
   PetscReal norm_inf, weight;
   PetscBool reuse_triggered;

   PetscFunctionBegin;

   // Unweighted by default
   weight = 1.0;

   // Let's create a matrix to represent the inverse diagonal
   reuse_triggered = (*inv_matrix != NULL) ? PETSC_TRUE : PETSC_FALSE;

   if (!reuse_triggered) {
      PetscCall(MatCreateVecs(matrix, NULL, &diag_vec));
   } else {
      PetscCall(MatDiagonalGetDiagonal(*inv_matrix, &diag_vec));
   }
   PetscCall(MatGetDiagonal(matrix, diag_vec));

   // l1 Jacobi has its own weight on each row
   if (inverse_type == PFLAREINV_L1JACOBI) {

      PetscCall(calculate_l1_jacobi_inverse_diagonal(matrix, diag_vec));

   // If weighting the Jacobi
   } else if (inverse_type == PFLAREINV_WJACOBI) {

      PetscCall(MatDuplicate(matrix, MAT_COPY_VALUES, &temp_mat));

      // D^(1/2)
      PetscCall(VecSqrtAbs(diag_vec));
      // D^(-1/2)
      PetscCall(VecReciprocal(diag_vec));
      // D^(-1/2) * A * D^(-1/2)
      PetscCall(MatDiagonalScale(temp_mat, diag_vec, diag_vec));
      // || D^(-1/2) * A * D^(-1/2) ||_inf
      PetscCall(MatNorm(temp_mat, NORM_INFINITY, &norm_inf));
      PetscCall(MatDestroy(&temp_mat));

      PetscCall(MatGetDiagonal(matrix, diag_vec));

      // This is the weight that hypre's BoomerAMG computes with relax_weight = 0
      // (see par_amg_setup.c), even in asymmetric problems
      // 4 / ( 3 * || D^(-1/2) * A * D^(-1/2) ||_inf )
      // Note the hypre docs for HYPRE_BoomerAMGSetRelaxWt quote 3 / (4 * ||.||)
      // but the code uses 4 / (3 * ||.||)
      // (PetscReal literals so a single precision build divides in single, as the Fortran does)
      weight = (PetscReal)4.0 / ((PetscReal)3.0 * norm_inf);

   }

   if (inverse_type != PFLAREINV_L1JACOBI) {
      PetscCall(VecReciprocal(diag_vec));
      PetscCall(VecScale(diag_vec, weight));
   }

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

PETSC_INTERN PetscErrorCode calculate_l1_jacobi_inverse_diagonal(Mat matrix, Vec diag_vec)
{
   // On entry diag_vec holds the diagonal of matrix, on exit it is overwritten
   // with the l1 Jacobi inverse diagonal
   // sign(a_ii) / sum_j |a_ij|, where the sum is over the whole (parallel) row
   // This is hypre's hypre_ParCSRComputeL1Norms with option 1, which also
   // flips the sign of the l1 norm on rows with a negative diagonal
   // Rows with a zero diagonal are set to zero (ie they are never relaxed)

   // Local variables
   Vec abs_diag_vec, row_sum_vec;
   MatType mat_type;
   PetscBool is_diagonal;

   PetscFunctionBegin;

   // MATDIAGONAL has no MatGetRowSumAbs, but there sum_j |a_ij| = |a_ii|
   // so the l1 Jacobi inverse is just the inverse of the diagonal
   PetscCall(MatGetType(matrix, &mat_type));
   PetscCall(PetscStrcmp(mat_type, MATDIAGONAL, &is_diagonal));
   if (is_diagonal) {
      PetscCall(VecReciprocal(diag_vec));
      PetscFunctionReturn(PETSC_SUCCESS);
   }
   PetscCall(VecDuplicate(diag_vec, &abs_diag_vec));
   PetscCall(VecDuplicate(diag_vec, &row_sum_vec));

   // sign(a_ii) = a_ii / |a_ii|
   // VecReciprocal leaves zeros as zero, so a zero diagonal gives a zero sign
   PetscCall(VecCopy(diag_vec, abs_diag_vec));
   PetscCall(VecAbs(abs_diag_vec));
   PetscCall(VecReciprocal(abs_diag_vec));
   PetscCall(VecPointwiseMult(diag_vec, diag_vec, abs_diag_vec));

   // 1 / sum_j |a_ij|
   PetscCall(MatGetRowSumAbs(matrix, row_sum_vec));
   PetscCall(VecReciprocal(row_sum_vec));
   PetscCall(VecPointwiseMult(diag_vec, diag_vec, row_sum_vec));

   PetscCall(VecDestroy(&abs_diag_vec));
   PetscCall(VecDestroy(&row_sum_vec));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------
