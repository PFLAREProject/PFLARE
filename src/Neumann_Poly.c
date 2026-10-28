/*
  C port of Neumann_Poly.F90 (module neumann_poly)
*/

#include <petscmat.h>
#include "pflare.h"
#include "pflareimpl.h"
#include "Gmres_Poly.h"
#include "Gmres_Poly_Apply.h"
#include "Matshell_Data_Type.h"
#include "TSQR.h"
#include "Neumann_Poly.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode calculate_and_build_neumann_polynomial_inverse(Mat matrix, int poly_order,
   tsqr_buffers *buffers, PetscReal coefficients[], PetscInt coefficients_size, int poly_sparsity_order, PetscBool matrix_free,
   Mat *reuse_mat, Mat **reuse_submatrices, Mat *inv_matrix)
{
   // Builds an assembled neumann polynomial approximate inverse
   // If poly_sparsity_order < poly_order it will build a fixed sparsity approximation
   // Scales by the diagonal and applies the scaled diagonal to the rhs of the output
   // so it can be used
   // The coefficients passed in must all be one (the caller sets this) - the matshell
   // points at this storage in the matrix-free case, exactly like the gmres builders,
   // so the ownership rules in calculate_and_build_approximate_inverse hold for all
   // the polynomial types

   // ~~~~~~
   PetscMPIInt comm_size;
   MPI_Comm MPI_COMM_MATRIX;
   VecType vtype;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   Mat temp_mat;
   Vec rhs_copy, diag_inverse_vec;
   mat_ctxtype *mat_ctx = NULL, *mat_ctx_scaled = NULL;
   // ~~~~~~

   PetscFunctionBegin;

   // Have to allocate heap memory if matrix-free as the context in inv_matrix
   // just points at the coefficients
   if (matrix_free) {

      // We might want to call the gmres poly creation on a sub communicator
      // so let's get the comm attached to the matrix and make sure to use that
      PetscCall(PetscObjectGetComm((PetscObject)matrix, &MPI_COMM_MATRIX));
      // Get the comm size
      PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));

      // Get the local sizes
      PetscCall(MatGetLocalSize(matrix, &local_rows, &local_cols));
      PetscCall(MatGetSize(matrix, &global_rows, &global_cols));

      // If not re-using
      if (!*inv_matrix) {

         // Have to dynamically allocate this
         PetscCall(create_mat_ctx(&mat_ctx));
         // The inner matshell we build below applies I - D^-1 A rather than D^-1 A
         // This is what tells the block (multiple rhs) apply which arithmetic to use
         mat_ctx->neumann_inner = PETSC_TRUE;

         // Create the matshell
         PetscCall(MatCreateShell(MPI_COMM_MATRIX, local_rows, local_cols, global_rows, global_cols,
                  mat_ctx, inv_matrix));
         // The subroutine petsc_matvec_right_scale_poly_mf applies
         // q(mat) D^-1
         PetscCall(MatShellSetOperation(*inv_matrix,
                  MATOP_MULT, (PetscErrorCodeFn *)petsc_matvec_right_scale_poly_mf));
         // The subroutine petsc_matvec_poly_transpose_mf applies the transpose of
         // the above - it builds a transposed twin of this matshell on demand,
         // see ensure_transpose_mat
         PetscCall(MatShellSetOperation(*inv_matrix,
                  MATOP_MULT_TRANSPOSE, (PetscErrorCodeFn *)petsc_matvec_poly_transpose_mf));

         PetscCall(MatAssemblyBegin(*inv_matrix, MAT_FINAL_ASSEMBLY));
         PetscCall(MatAssemblyEnd(*inv_matrix, MAT_FINAL_ASSEMBLY));
         // Have to make sure to set the type of vectors the shell creates
         PetscCall(MatGetVecType(matrix, &vtype));
         PetscCall(MatShellSetVecType(*inv_matrix, vtype));

         // Create temporary vector we use during horner
         // Make sure to use matrix here to get the right type (as the shell doesn't know about gpus)
         PetscCall(MatCreateVecs(matrix, &mat_ctx->mf_temp_vec[MF_VEC_TEMP], NULL));

         // ~~~~~~~~~~~~~
         // Now we allocate a new matshell that applies a diagonally scaled version of
         // the matrix minus I
         // ~~~~~~~~~~~~~

         // Have to dynamically allocate this
         PetscCall(create_mat_ctx(&mat_ctx_scaled));

         // Create the matshell
         PetscCall(MatCreateShell(MPI_COMM_MATRIX, local_rows, local_cols, global_rows, global_cols,
                  mat_ctx_scaled, &mat_ctx->mat_scaled));
         // The subroutine petsc_matvec_ida_neumann_poly_mf applies I - D^-1 A
         PetscCall(MatShellSetOperation(mat_ctx->mat_scaled,
                  MATOP_MULT, (PetscErrorCodeFn *)petsc_matvec_ida_neumann_poly_mf));

         PetscCall(MatAssemblyBegin(mat_ctx->mat_scaled, MAT_FINAL_ASSEMBLY));
         PetscCall(MatAssemblyEnd(mat_ctx->mat_scaled, MAT_FINAL_ASSEMBLY));
         // Have to make sure to set the type of vectors the shell creates
         PetscCall(MatGetVecType(matrix, &vtype));
         PetscCall(MatShellSetVecType(mat_ctx->mat_scaled, vtype));

         // Create temporary vector we use during horner
         // Make sure to use matrix here to get the right type (as the shell doesn't know about gpus)
         PetscCall(MatCreateVecs(matrix, &mat_ctx->mf_temp_vec[MF_VEC_RHS], &mat_ctx->mf_temp_vec[MF_VEC_DIAG]));

      // Reusing
      } else {
         PetscCall(MatShellGetContext(*inv_matrix, &mat_ctx));
         PetscCall(MatShellGetContext(mat_ctx->mat_scaled, &mat_ctx_scaled));
      }

      // Free old owned coefficients before reassigning, exactly as the gmres
      // builders do - avoids a memory leak when the same matshell is reused across
      // PCSetUp calls (SAME_NONZERO_PATTERN) but fresh coefficients are passed in
      if (mat_ctx->own_coefficients && mat_ctx->coefficients) {
         PetscCall(PetscFree(mat_ctx->coefficients));
      }

      // The matshell points at the caller's coefficient storage (all ones)
      // Ownership is the caller's decision, exactly as with the gmres builders:
      // calculate_and_build_approximate_inverse sets own_coefficients after we
      // return, and for PCAIR it stays .FALSE. as PCAIR manages its own storage
      mat_ctx->coefficients = coefficients;
      mat_ctx->coefficients_size = coefficients_size;
      mat_ctx_scaled->coefficients = coefficients;
      mat_ctx_scaled->coefficients_size = coefficients_size;

      // This is the matrix whose inverse we are applying (just copying the pointer here)
      mat_ctx->mat = matrix;
      mat_ctx_scaled->mat = matrix;

      // Get the diagonal
      PetscCall(MatGetDiagonal(matrix, mat_ctx->mf_temp_vec[MF_VEC_DIAG]));
      mat_ctx_scaled->mf_temp_vec[MF_VEC_DIAG] = mat_ctx->mf_temp_vec[MF_VEC_DIAG];

   // If not matrix free
   } else {

      // The coefficients we've been passed are already all one

      // Need to build an assembled I - D^-1 A
      PetscCall(MatDuplicate(matrix, MAT_COPY_VALUES, &temp_mat));
      PetscCall(MatCreateVecs(matrix, &rhs_copy, &diag_inverse_vec));
      PetscCall(MatGetDiagonal(matrix, diag_inverse_vec));
      PetscCall(VecReciprocal(diag_inverse_vec));
      PetscCall(MatDiagonalScale(temp_mat, diag_inverse_vec, NULL));

      // Computes: I - D^-1 A
      PetscCall(MatScale(temp_mat, PFLARE_MINUS_ONE));
      PetscCall(MatShift(temp_mat, PFLARE_ONE));

      // If we feed in coefficients=1 and leave buffers->R_buffer_receive unallocated as it will just skip
      // the "gmres" coefficient calculation and just calculate the sum of (potentailly fixed sparsity) matrix powers
      // We set diag_scale_polys to false as we have already scaled the matrix by the diagonal
      // and we scale the columns ourself next
      PetscCall(build_gmres_polynomial_inverse(temp_mat, poly_order, buffers, coefficients, coefficients_size,
               poly_sparsity_order, PETSC_FALSE, PETSC_FALSE, reuse_mat, reuse_submatrices, inv_matrix));

      // Now this computes (I - D^-1 A)^-1 D^-1
      // For the F-point smoothing and grid-transfer operators in our air multigrid,
      // this is equivalent to using (I - Dff^-1 Aff)^-1 Dff^-1 everywhere we normally use
      // Aff^-1
      PetscCall(MatDiagonalScale(*inv_matrix, NULL, diag_inverse_vec));

      // Cleanup
      PetscCall(VecDestroy(&rhs_copy));
      PetscCall(VecDestroy(&diag_inverse_vec));
      PetscCall(MatDestroy(&temp_mat));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}
