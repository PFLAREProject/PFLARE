/*
  C port of Grid_Transfer_Improve.F90 (module grid_transfer_improve)

  W and Z are refilled in place (their handles do not change), the two reuse
  matrices are Mat * as they may be created here. The timers and
  remove_from_sparse_match (remove_from_sparse_match_c in C, see PETSc_Helper.c)
  are called directly
*/

#include <petscmat.h>
#include "pflareimpl.h"
#include "Timers.h"
#include "PETSc_Helper.h"
#include "Grid_Transfer_Improve.h"

// -------------------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------------------------------------
// Grid transfer improvement routines
// -------------------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode improve_w(Mat W, Mat A_ff, Mat A_fc, Mat A_ff_inv, Mat *reuse_mat, Mat *reuse_mat_two,
   int its, PetscBool reuse_sparsity)
{
   // Does a richardson iteration to improve the ideal W

   // ~~~~~~~~~~
   int i;
   // PetscReal :: residual
   Mat temp_mat_residual_copy = NULL, temp_mat, temp_mat_axpy = NULL, temp_mat_sparsity = NULL;
   Vec left_vec_aff = NULL, left_vec_inv_aff = NULL;
   MatType mat_type_aff, mat_type_inv_aff;
   PetscBool diag_aff_inv, trigger_delete, aff_is_diagonal;
   // The alpha passed to remove_from_sparse_match
   PetscReal alpha = PFLARE_MINUS_ONE;
   // ~~~~~~~~~~

   PetscFunctionBegin;

   // Return if nothing to do
   if (its == 0) PetscFunctionReturn(PETSC_SUCCESS);

   // Check if Aff is diagonal
   PetscCall(MatGetType(A_ff, &mat_type_aff));
   PetscCall(PetscStrcmp(mat_type_aff, MATDIAGONAL, &aff_is_diagonal));
   // Pull out the diagonals if Aff is diagonal
   if (aff_is_diagonal) {
      // Get a reference to Aff's diagonal (restored below), avoiding a copy
      PetscCall(MatDiagonalGetDiagonal(A_ff, &left_vec_aff));
   }

   // Check if inv Aff is diagonal
   PetscCall(MatGetType(A_ff_inv, &mat_type_inv_aff));
   // Pull out the diagonals
   diag_aff_inv = PETSC_FALSE;
   // We now just always pull out the diagonal of Aff^-1, it works very well
   // and saves a matmatmult if Aff^-1 is not diagonal
   //if (mat_type_inv_aff == MATDIAGONAL) then
      PetscCall(MatCreateVecs(W, NULL, &left_vec_inv_aff));
      PetscCall(MatGetDiagonal(A_ff_inv, left_vec_inv_aff));
      diag_aff_inv = PETSC_TRUE;
   //end if

   trigger_delete = PETSC_FALSE;

   // Do the number of iterations requested
   // Can reuse the same sparsity if doing multiple iterations
   for (i = 1; i <= its; i++) {

      // Compute the residual - Aff W^n + Afc
      if (aff_is_diagonal) {

         temp_mat = *reuse_mat;
         if (temp_mat) {

            // The residual will have a different sparsity pattern to W
            // after Aff W^n + Afc occurs in the first iteration
            // (even if Aff is diagonal due to the A_fc)
            PetscCall(MatCopy(W,
                     *reuse_mat,
                     DIFFERENT_NONZERO_PATTERN));
         } else {
            PetscCall(MatDuplicate(W,
                     MAT_COPY_VALUES,
                     reuse_mat));
         }

         // Left multiply
         PetscCall(MatDiagonalScale(*reuse_mat,
                  left_vec_aff, NULL));

         temp_mat_axpy = *reuse_mat;

      // If not matdiagonal
      } else {
         temp_mat = *reuse_mat;

         // Now because kokkos is picky about having the exact
         // same matrix in the MatMatMult when reusing
         // We cannot just use reuse_mat everywhere
         // as MatAXPY rebuilds a new matrix internally and replaces it
         // Therefore have to keep around an extra copy just
         // for reuse in kokkos
         // But if we are only doing a max of 1 iteration and not doing external reuse
         if (!temp_mat) {
            PetscCall(MatMatMult(A_ff, W, MAT_INITIAL_MATRIX, PFLARE_ONE,
                  reuse_mat));
            temp_mat_axpy = *reuse_mat;

            // Only need to duplicate if we are doing more than 1 iteration
            if (its > 1 || reuse_sparsity) {
               PetscCall(MatDuplicate(*reuse_mat,
                     MAT_COPY_VALUES,
                     &temp_mat_residual_copy));

               trigger_delete = PETSC_TRUE;
               temp_mat_axpy = temp_mat_residual_copy;
            }

         } else {
            PetscCall(MatMatMult(A_ff, W, MAT_REUSE_MATRIX, PFLARE_ONE,
                  reuse_mat));
            if (i == 1) {
               PetscCall(MatDuplicate(*reuse_mat,
                     MAT_COPY_VALUES,
                     &temp_mat_residual_copy));
            } else {
               PetscCall(MatCopy(*reuse_mat,
                     temp_mat_residual_copy,
                     DIFFERENT_NONZERO_PATTERN));
            }

            trigger_delete = PETSC_TRUE;
            temp_mat_axpy = temp_mat_residual_copy;
         }
      }

      // Afc should have a subset of the sparsity of W if Aff
      // has a diagonal, but just to be safe lets
      // say its different
      PetscCall(MatAXPYWrapper(temp_mat_axpy, PFLARE_ONE, A_fc));

      // If you want to print the residual
      // call MatNorm(temp_mat_axpy, NORM_FROBENIUS, residual, ierr)
      // print *, i, "residual = ", residual

      // Multiply on the left by the preconditioner
      // Aff_inv (Aff W^n - Afc)

      // Special case if Aff inv is matdiagonal
      if (diag_aff_inv) {

         // Left multiply
         PetscCall(MatDiagonalScale(temp_mat_axpy,
                  left_vec_inv_aff, NULL));

         temp_mat_sparsity = temp_mat_axpy;

      // If not matdiagonal
      } else {

         temp_mat = *reuse_mat_two;
         if (!temp_mat) {
            PetscCall(MatMatMult(A_ff_inv, temp_mat_axpy, MAT_INITIAL_MATRIX, PFLARE_ONE,
                  reuse_mat_two));
         } else {
            PetscCall(MatMatMult(A_ff_inv, temp_mat_axpy, MAT_REUSE_MATRIX, PFLARE_ONE,
                  reuse_mat_two));
         }
         temp_mat_sparsity = *reuse_mat_two;
      }

      PetscCall(timer_start(PFLARE_TIMER_ID_AIR_DROP));
      // Compute
      // W^n+1 = W^n - Aff_inv * (Aff W^n + Afc)
      // and drop any non-zeros outside of the sparsity pattern of W
      PetscCall(remove_from_sparse_match_c(temp_mat_sparsity, W, NULL, &alpha));
      PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_DROP));

   }

   // Destroy the temporaries
   if (trigger_delete) PetscCall(MatDestroy(&temp_mat_residual_copy));
   // left_vec_aff is a reference to Aff's diagonal when Aff is MATDIAGONAL, so restore it
   if (aff_is_diagonal) {
      PetscCall(MatDiagonalRestoreDiagonal(A_ff, &left_vec_aff));
   } else {
      PetscCall(VecDestroy(&left_vec_aff));
   }
   PetscCall(VecDestroy(&left_vec_inv_aff));

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode improve_z(Mat Z, Mat A_ff, Mat A_cf, Mat A_ff_inv, Mat *reuse_mat, Mat *reuse_mat_two,
   int its, PetscBool reuse_sparsity)
{
   // Does a richardson iteration to improve the ideal Z

   // ~~~~~~~~~~
   int i;
   // PetscReal :: residual
   Mat temp_mat_residual_copy = NULL, temp_mat, temp_mat_axpy = NULL, temp_mat_sparsity = NULL;
   Vec right_vec_aff = NULL, right_vec_inv_aff = NULL;
   MatType mat_type_aff, mat_type_inv_aff;
   PetscBool diag_aff_inv, trigger_delete, aff_is_diagonal;
   // The alpha passed to remove_from_sparse_match
   PetscReal alpha = PFLARE_MINUS_ONE;
   // ~~~~~~~~~~

   PetscFunctionBegin;

   // Return if nothing to do
   if (its == 0) PetscFunctionReturn(PETSC_SUCCESS);

   // Check if Aff is diagonal
   PetscCall(MatGetType(A_ff, &mat_type_aff));
   PetscCall(PetscStrcmp(mat_type_aff, MATDIAGONAL, &aff_is_diagonal));
   // Pull out the diagonals if Aff is diagonal
   if (aff_is_diagonal) {
      // Get a reference to Aff's diagonal (restored below), avoiding a copy
      PetscCall(MatDiagonalGetDiagonal(A_ff, &right_vec_aff));
   }

   // Check if inv Aff is diagonal
   PetscCall(MatGetType(A_ff_inv, &mat_type_inv_aff));
   // Pull out the diagonals
   diag_aff_inv = PETSC_FALSE;
   // We now just always pull out the diagonal of Aff^-1, it works very well
   // and saves a matmatmult if Aff^-1 is not diagonal
   //if (mat_type_inv_aff == MATDIAGONAL) then
      PetscCall(MatCreateVecs(Z, &right_vec_inv_aff, NULL));
      PetscCall(MatGetDiagonal(A_ff_inv, right_vec_inv_aff));
      diag_aff_inv = PETSC_TRUE;
   //end if

   trigger_delete = PETSC_FALSE;

   // Do the number of iterations requested
   // Can reuse the same sparsity if doing multiple iterations
   for (i = 1; i <= its; i++) {

      // Compute the residual - Z^n Aff + Acf
      if (aff_is_diagonal) {

         temp_mat = *reuse_mat;
         if (temp_mat) {

            // The residual will have a different sparsity pattern to Z
            // after Z * Aff + A_cf occurs in the first iteration
            // (even if Aff is diagonal due to the A_cf)
            PetscCall(MatCopy(Z,
                     *reuse_mat,
                     DIFFERENT_NONZERO_PATTERN));
         } else {
            PetscCall(MatDuplicate(Z,
                     MAT_COPY_VALUES,
                     reuse_mat));
         }

         // Right multiply
         PetscCall(MatDiagonalScale(*reuse_mat,
                  NULL, right_vec_aff));

         temp_mat_axpy = *reuse_mat;

      // If not matdiagonal
      } else {
         temp_mat = *reuse_mat;

         // Now because kokkos is picky about having the exact
         // same matrix in the MatMatMult when reusing
         // We cannot just use reuse_mat everywhere
         // as MatAXPY rebuilds a new matrix internally and replaces it
         // Therefore have to keep around an extra copy just
         // for reuse in kokkos
         // But if we are only doing a max of 1 iteration and not doing external reuse
         if (!temp_mat) {
            PetscCall(MatMatMult(Z, A_ff, MAT_INITIAL_MATRIX, PFLARE_ONE,
                  reuse_mat));
            temp_mat_axpy = *reuse_mat;

            // Only need to duplicate if we are doing more than 1 iteration
            if (its > 1 || reuse_sparsity) {
               PetscCall(MatDuplicate(*reuse_mat,
                     MAT_COPY_VALUES,
                     &temp_mat_residual_copy));

               trigger_delete = PETSC_TRUE;
               temp_mat_axpy = temp_mat_residual_copy;
            }

         } else {
            PetscCall(MatMatMult(Z, A_ff, MAT_REUSE_MATRIX, PFLARE_ONE,
                  reuse_mat));
            if (i == 1) {
               PetscCall(MatDuplicate(*reuse_mat,
                     MAT_COPY_VALUES,
                     &temp_mat_residual_copy));
            } else {
               PetscCall(MatCopy(*reuse_mat,
                     temp_mat_residual_copy,
                     DIFFERENT_NONZERO_PATTERN));
            }

            trigger_delete = PETSC_TRUE;
            temp_mat_axpy = temp_mat_residual_copy;
         }
      }

      // Acf should have a subset of the sparsity of Z if Aff
      // has a diagonal, but just to be safe lets
      // say its different
      PetscCall(MatAXPYWrapper(temp_mat_axpy, PFLARE_ONE, A_cf));

      // If you want to print the residual
      // call MatNorm(temp_mat_axpy, NORM_FROBENIUS, residual, ierr)
      // print *, i, "residual = ", residual

      // Multiply on the right by the preconditioner
      // (Z^n Aff - Acf) * Aff_inv

      // Special case if Aff inv is matdiagonal
      if (diag_aff_inv) {

         // Right multiply
         PetscCall(MatDiagonalScale(temp_mat_axpy,
                  NULL, right_vec_inv_aff));

         temp_mat_sparsity = temp_mat_axpy;

      // If not matdiagonal
      } else {

         temp_mat = *reuse_mat_two;
         if (!temp_mat) {
            PetscCall(MatMatMult(temp_mat_axpy, A_ff_inv, MAT_INITIAL_MATRIX, PFLARE_ONE,
                  reuse_mat_two));
         } else {
            PetscCall(MatMatMult(temp_mat_axpy, A_ff_inv, MAT_REUSE_MATRIX, PFLARE_ONE,
                  reuse_mat_two));
         }
         temp_mat_sparsity = *reuse_mat_two;
      }

      PetscCall(timer_start(PFLARE_TIMER_ID_AIR_DROP));
      // Compute
      // Z^n+1 = Z^n - (Z^n Aff + Acf) * Aff_inv
      // and drop any non-zeros outside of the sparsity pattern of Z
      PetscCall(remove_from_sparse_match_c(temp_mat_sparsity, Z, NULL, &alpha));
      PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_DROP));

   }

   // Destroy the temporaries
   if (trigger_delete) PetscCall(MatDestroy(&temp_mat_residual_copy));
   // right_vec_aff is a reference to Aff's diagonal when Aff is MATDIAGONAL, so restore it
   if (aff_is_diagonal) {
      PetscCall(MatDiagonalRestoreDiagonal(A_ff, &right_vec_aff));
   } else {
      PetscCall(VecDestroy(&right_vec_aff));
   }
   PetscCall(VecDestroy(&right_vec_inv_aff));

   PetscFunctionReturn(PETSC_SUCCESS);
}

//-------------------------------------------------------------------------------------------------------------------------------
