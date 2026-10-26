/*
  C port of Approx_Inverse_Setup.F90 (module approx_inverse_setup)

  The coefficients are column-major with coefficients_rows rows and
  coefficients_cols columns, and are allocated with PetscMalloc and freed with
  PetscFree (the Fortran allocate/deallocate). The memory contract of the
  coefficients PCPFLAREINV hands to users (a malloc'd copy the user frees with
  free) is kept by PCPFLAREINV.c, which used to get that copy from the C binding
  in C_Fortran_Bindings.F90
*/

#include <petscmat.h>
#include "pflare.h"
#include "pflareimpl.h"
#include "TSQR.h"
#include "Gmres_Poly.h"
#include "Gmres_Poly_Apply.h"
#include "Gmres_Poly_Newton.h"
#include "Neumann_Poly.h"
#include "Weighted_Jacobi.h"
#include "SAI_Z.h"
#include "Repartition.h"
#include "PETSc_Helper.h"
#include "Matshell_Data_Type.h"
#include "Approx_Inverse_Setup.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode calculate_and_build_approximate_inverse(Mat matrix, int inverse_type,
   int poly_order, int inverse_sparsity_order,
   PetscBool matrix_free, PetscBool diag_scale_polys, PetscBool subcomm,
   Mat *inv_matrix, PetscReal **coefficients, PetscInt *coefficients_rows, PetscInt *coefficients_cols)
{
   // Builds an approximate inverse
   // inverse_type:
   // PFLAREINV_POWER - GMRES polynomial with the power basis
   // PFLAREINV_ARNOLDI - GMRES polynomial with the arnoldi basis
   // PFLAREINV_NEWTON - GMRES polynomial with the newton basis with extra roots for stability - can only be used matrix-free atm
   // PFLAREINV_NEWTON_NO_EXTRA - GMRES polynomial with the newton basis with no extra roots - can only be used matrix-free atm
   // PFLAREINV_NEUMANN - Neumann polynomial
   // PFLAREINV_SAI - SAI
   // PFLAREINV_ISAI - Incomplete SAI (ie a restricted additive schwartz)
   // PFLAREINV_WJACOBI - Weighted Jacobi with weight 4 / ( 3 * || D^(-1/2) * A * D^(-1/2) ||_inf )
   // PFLAREINV_JACOBI - Unweighted Jacobi
   // PFLAREINV_L1JACOBI - l1 Jacobi with inverse sign(a_ii) / sum_j |a_ij|
   // This is just a wrapper around the start and finish routines below
   // No opportunity to put work between the async comms
   // If you want to do that, you will have to call the individual routines
   //
   // Optional coefficients argument (NULL when absent):
   //   - If present and associated on entry: the existing coefficients are reused and
   //     start_approximate_inverse is skipped entirely. buffers is set up manually
   //     (subcomm=.FALSE., buffers%matrix = matrix with ref incremented) so that
   //     finish_approximate_inverse can clean up normally. For a matrix-free inverse
   //     the matshell does NOT take ownership (the caller owns them).
   //   - If present but not associated on entry: fresh coefficients are computed, a
   //     heap allocation is made, and on return the pointer is associated with that
   //     allocation. For matrix-free the matshell also does NOT take ownership (the
   //     caller takes ownership via the returned pointer).
   //   - If absent: existing behaviour. The matshell takes ownership of heap-allocated
   //     coefficients (matrix-free case) or stack storage is used (assembled case).
   //   - Non-polynomial inverse types (SAI, ISAI, WJACOBI, JACOBI, L1JACOBI) have no coefficients:
   //     supplied coefficients are ignored and none are allocated or returned, so
   //     coefficients is left unassociated if it was not associated on entry.
   // (As in the Fortran, in the matrix-free fresh case with coefficients present the
   // matshell is marked as owning them below, so the caller has to take a copy before
   // the matshell is reset - PCPFLAREINV.c does)

   // ~~~~~~
   tsqr_buffers *buffers;
   PetscReal *work_coefficients;
   PetscInt work_coefficients_rows, work_coefficients_cols;
   // The Fortran used stack storage (poly_order + 1 rows) here
   PetscReal *coefficients_stack;
   mat_ctxtype *mat_ctx;
   Mat reuse_mat = NULL, inv_matrix_temp = NULL;
   Mat *reuse_submatrices = NULL;
   PetscBool heap_allocated, coefficients_supplied, poly_type, matrix_free_shell;
   int poly_order_eff, inverse_sparsity_order_eff;
   PetscInt global_rows, global_cols;
   // ~~~~~~

   PetscFunctionBegin;

   // Only the polynomial types have coefficients. The other types (SAI, ISAI, WJACOBI, JACOBI, L1JACOBI)
   // don't allocate, return or take ownership of any
   poly_type = inverse_type_is_polynomial(inverse_type);
   // Only the polynomial types build a matshell when applied matrix-free
   // The other types (e.g., Jacobi) ignore matrix_free and are always assembled
   // so we must not touch a matshell context for them
   matrix_free_shell = (matrix_free && poly_type) ? PETSC_TRUE : PETSC_FALSE;

   // For matrices with size smaller than the subspace size (ie polynomial order + 1)
   // the gmres polynomials can only go up to the matrix size (at which point they are
   // (close to) an exact solver), so clamp the order like setup_gmres_poly_data does for PCAIR
   // Also make sure we aren't trying to take sparsity bigger than the clamped order
   poly_order_eff = poly_order;
   inverse_sparsity_order_eff = inverse_sparsity_order;
   // The Neumann polynomial doesn't build a Krylov subspace so doesn't need clamping
   if (poly_type && inverse_type != PFLAREINV_NEUMANN) {
      PetscCall(MatGetSize(matrix, &global_rows, &global_cols));
      if (poly_order_eff + 1 > global_rows) {
         poly_order_eff = (int)(global_rows - 1);
      }
      if (inverse_sparsity_order_eff > poly_order_eff) {
         inverse_sparsity_order_eff = poly_order_eff;
      }
   }

   // The Fortran had to work around the way petsc 3.22 tests a Fortran matrix for null
   // (a null matrix from C is not the null the Fortran mat creation routines expect),
   // by building into inv_matrix_temp and copying it back into inv_matrix at the end
   // That isn't needed in C, but the copy is kept so the flow is the same
   inv_matrix_temp = *inv_matrix;

   PetscCall(PetscMalloc1(poly_order + 1, &coefficients_stack));

   // The Fortran tsqr_buffers was a local with its default initialisation, freed on return
   PetscCall(create_tsqr_buffers(&buffers));

   // The inverse is always returned on the same comm as the input matrix
   // but some methods benefit from having their intermediate calculations
   // done on a subcomm
   buffers->subcomm = subcomm;
   // Don't do any overlapping of comms, if you want that call the start/finish yourself

   coefficients_supplied = PETSC_FALSE;
   if (coefficients) {
      if (*coefficients) coefficients_supplied = PETSC_TRUE;
   }

   if (coefficients_supplied) {

      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      // Reuse path: valid coefficients supplied by the caller;
      // skip start_approximate_inverse entirely to avoid calling
      // PetscObjectReference on a potential null matrix.
      // Replicate the non-subcomm setup that start_approximate_inverse
      // would have done so that finish_approximate_inverse can clean up normally.
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      work_coefficients = *coefficients;
      work_coefficients_rows = *coefficients_rows;
      work_coefficients_cols = *coefficients_cols;
      buffers->on_subcomm = PETSC_FALSE;
      buffers->matrix = matrix;
      PetscCall(PetscObjectReference((PetscObject)matrix));
      PetscCall(finish_approximate_inverse(matrix, inverse_type,
               poly_order_eff, inverse_sparsity_order_eff,
               buffers, work_coefficients, work_coefficients_rows, work_coefficients_cols,
               matrix_free, diag_scale_polys,
               &reuse_mat, &reuse_submatrices, &inv_matrix_temp));
      // Caller owns the coefficient memory; matshell must not free it
      if (matrix_free_shell) {
         PetscCall(MatShellGetContext(inv_matrix_temp, &mat_ctx));
         mat_ctx->own_coefficients = PETSC_FALSE;
      }

   } else {

      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      // Fresh computation path
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      // Heap-allocate when: matrix-free (matshell will hold the pointer),
      // coefficients are being returned to the caller, or Newton basis (2 columns).
      // Otherwise, stack storage suffices (unless the order has been clamped,
      // as the stack storage is sized with the unclamped order). Non-polynomial types
      // never touch the coefficients, so they just get the (unused) stack storage.
      heap_allocated = (poly_type && (matrix_free || poly_order_eff != poly_order ||
                       (coefficients && !coefficients_supplied) ||
                       inverse_type == PFLAREINV_NEWTON || inverse_type == PFLAREINV_NEWTON_NO_EXTRA)) ? PETSC_TRUE : PETSC_FALSE;
      if (heap_allocated) {
         if (inverse_type == PFLAREINV_NEWTON || inverse_type == PFLAREINV_NEWTON_NO_EXTRA) {
            // Newton basis needs storage for real and imaginary roots
            work_coefficients_cols = 2;
         } else {
            work_coefficients_cols = 1;
         }
         work_coefficients_rows = poly_order_eff + 1;
         PetscCall(PetscMalloc1(work_coefficients_rows * work_coefficients_cols, &work_coefficients));
      } else {
         work_coefficients = coefficients_stack;
         work_coefficients_rows = poly_order + 1;
         work_coefficients_cols = 1;
      }

      // Start the calculation
      PetscCall(start_approximate_inverse(matrix, inverse_type,
               poly_order_eff, diag_scale_polys,
               buffers, &work_coefficients, &work_coefficients_rows, &work_coefficients_cols));
      // Finish it
      PetscCall(finish_approximate_inverse(matrix, inverse_type,
               poly_order_eff, inverse_sparsity_order_eff,
               buffers, work_coefficients, work_coefficients_rows, work_coefficients_cols,
               matrix_free, diag_scale_polys,
               &reuse_mat, &reuse_submatrices, &inv_matrix_temp));

      if (!poly_type) {
         // No coefficients to return; leave coefficients unassociated
      } else if (matrix_free_shell) {
         PetscCall(MatShellGetContext(inv_matrix_temp, &mat_ctx));
         // The matshell always owns its coefficients and frees them in
         // reset_inverse_mat. When coefficients are present, the caller is
         // responsible for making its own copy before this pointer is freed.
         mat_ctx->own_coefficients = PETSC_TRUE;
         if (coefficients) {
            *coefficients = work_coefficients;
            *coefficients_rows = work_coefficients_rows;
            *coefficients_cols = work_coefficients_cols;
         }
      } else if (coefficients) {
         // Return heap-allocated coefficients to caller
         *coefficients = work_coefficients;
         *coefficients_rows = work_coefficients_rows;
         *coefficients_cols = work_coefficients_cols;
      } else if (heap_allocated) {
         // Heap-allocated (Newton assembled path) but not needed further: free
         PetscCall(PetscFree(work_coefficients));
      }
   }

   if (reuse_mat) {
      PetscCall(destroy_matrix_reuse(&reuse_mat, &reuse_submatrices));
   }

   // Copy the pointer - see comment above about petsc 3.22
   *inv_matrix = inv_matrix_temp;

   PetscCall(PetscFree(coefficients_stack));
   PetscCall(destroy_tsqr_buffers(&buffers));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscBool inverse_type_is_polynomial(int inverse_type)
{
   // Returns true if the inverse type is a polynomial, ie one that has coefficients
   // and builds a matshell when applied matrix-free. The other types (SAI, ISAI,
   // WJACOBI, JACOBI, L1JACOBI) have no coefficients and are always assembled
   // Also called from PCPFLAREINV.c

   return (inverse_type == PFLAREINV_POWER ||
           inverse_type == PFLAREINV_ARNOLDI ||
           inverse_type == PFLAREINV_NEWTON ||
           inverse_type == PFLAREINV_NEWTON_NO_EXTRA ||
           inverse_type == PFLAREINV_NEUMANN) ? PETSC_TRUE : PETSC_FALSE;
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode start_approximate_inverse(Mat matrix, int inverse_type, int poly_order, PetscBool diag_scale_polys,
   tsqr_buffers *buffers, PetscReal **coefficients, PetscInt *coefficients_rows, PetscInt *coefficients_cols)
{
   // Starts the assembly of an approximate inverse
   // We have different types of inverses we can use
   // Have to call finish_approximate_inverse before it can be used
   // The Newton bases can reallocate the coefficients

   // ~~~~~~
   MPI_Comm MPI_COMM_MATRIX, MPI_COMM_BUFFERS_MATRIX;
   VecType vtype;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscMPIInt count;
   PetscInt n_roots;
   mat_ctxtype *mat_ctx_da;
   // ~~~~~~

   PetscFunctionBegin;

   PetscCheck(!(buffers->subcomm && inverse_type == PFLAREINV_POWER), PetscObjectComm((PetscObject)matrix), PETSC_ERR_ARG_INCOMP,
      "There is no reason to use a subcomm with the power basis, turn off subcomm");

   if (buffers->subcomm &&
         (inverse_type == PFLAREINV_ARNOLDI ||
         inverse_type == PFLAREINV_NEWTON ||
         inverse_type == PFLAREINV_NEWTON_NO_EXTRA)) {

      // Create a version of matrix on a subcomm and store it in buffers->matrix
      // If all ranks have entries, then this just returns a pointer to the original
      // matrix (ie buffers->matrix = matrix) with the reference counter incremented
      // If not, then a copy is taken on a subcomm with ranks that have entries
      // That is the matrix we use to start our polynomial coefficient calculation
      // But *importantly* the inverse matrix we create is always on MPI_COMM_WORLD
      // I did try building a mg hierarchy that progressively moved onto smaller
      // subcomms with processor agglomeration, but it ended up expensive
      // This way, only the reductions done in the coefficient calculation happen on
      // the subcomm
      // On any rank that isn't part of this subcomm, we then need to comm the coefficients
      // to those ranks, as petsc in debug mode checks that matscale (and mataxpy) all use
      // the same coefficient, even if the matrix doesn't have any rows and hence wouldn't use them
      PetscCall(MatMPICreateNonemptySubcomm(matrix, &buffers->on_subcomm, &buffers->matrix));
   } else {

      buffers->matrix = matrix;
      // Increase the reference counter
      PetscCall(PetscObjectReference((PetscObject)matrix));
   }

   // ~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~

   // If we're on the subcomm, we don't need to do anything
   if (buffers->matrix) {

      // Have to dynamically allocate this
      PetscCall(create_mat_ctx(&mat_ctx_da));
      mat_ctx_da->mat_scaled = buffers->matrix;

      // If we want to diagonally scale (ie apply Jacobi preconditioned gmres polynomial)
      if ((inverse_type == PFLAREINV_POWER ||
           inverse_type == PFLAREINV_ARNOLDI ||
           inverse_type == PFLAREINV_NEWTON ||
           inverse_type == PFLAREINV_NEWTON_NO_EXTRA) &&
           diag_scale_polys) {

         // ~~~~~~~~~~~~~
         // Now we allocate a new matshell that applies a diagonally scaled version of
         // the matrix to compute the coefficients
         // ~~~~~~~~~~~~~
         // Get the comm on the buffers matrix (which may be a subcomm)
         PetscCall(PetscObjectGetComm((PetscObject)buffers->matrix, &MPI_COMM_BUFFERS_MATRIX));

         // Get the sizes
         PetscCall(MatGetLocalSize(buffers->matrix, &local_rows, &local_cols));
         PetscCall(MatGetSize(buffers->matrix, &global_rows, &global_cols));

         // Create the matshell
         PetscCall(MatCreateShell(MPI_COMM_BUFFERS_MATRIX, local_rows, local_cols, global_rows, global_cols,
                  mat_ctx_da, &mat_ctx_da->mat_scaled));
         // The subroutine petsc_matvec_da_poly_mf applies D^-1 A
         PetscCall(MatShellSetOperation(mat_ctx_da->mat_scaled,
                  MATOP_MULT, (PetscErrorCodeFn *)petsc_matvec_da_poly_mf));

         PetscCall(MatAssemblyBegin(mat_ctx_da->mat_scaled, MAT_FINAL_ASSEMBLY));
         PetscCall(MatAssemblyEnd(mat_ctx_da->mat_scaled, MAT_FINAL_ASSEMBLY));
         // Have to make sure to set the type of vectors the shell creates
         PetscCall(MatGetVecType(buffers->matrix, &vtype));
         PetscCall(MatShellSetVecType(mat_ctx_da->mat_scaled, vtype));

         // Create temporary vector we use during horner
         // Make sure to use matrix here to get the right type (as the shell doesn't know about gpus)
         PetscCall(MatCreateVecs(buffers->matrix, NULL, &mat_ctx_da->mf_temp_vec[MF_VEC_DIAG]));
         // Get the diagonal
         PetscCall(MatGetDiagonal(buffers->matrix, mat_ctx_da->mf_temp_vec[MF_VEC_DIAG]));
         // This is the matrix whose inverse we are applying (just copying the pointer here)
         mat_ctx_da->mat = buffers->matrix;
      }

      // ~~~~~~~~~
      // mat_ctx_da->mat_scaled now contains a shell that just does D^-1 A
      // We compute the coefficients with that and then finish_approximate_inverse builds
      // whatever matrix it needs to compute the appropriate inverse with those coeffs
      // ~~~~~~~~~

      // Gmres poylnomial with power basis
      if (inverse_type == PFLAREINV_POWER) {

         // Only does one reduction in parallel
         // Want to start the coefficient calculation asap, as we have a non-blocking
         // all reduce in it
         PetscCall(start_gmres_polynomial_coefficients_power(mat_ctx_da->mat_scaled, poly_order,
                  buffers, *coefficients));

      // Gmres polynomial with arnoldi basis
      } else if (inverse_type == PFLAREINV_ARNOLDI) {

         // Does lots of reductions in parallel
         PetscCall(calculate_gmres_polynomial_coefficients_arnoldi(mat_ctx_da->mat_scaled, poly_order, *coefficients, NULL));

      // Gmres polynomial with Newton basis with extra added roots for stability - Can only use matrix-free
      } else if (inverse_type == PFLAREINV_NEWTON) {

         // Does lots of reductions in parallel
         PetscCall(calculate_gmres_polynomial_roots_newton(mat_ctx_da->mat_scaled, poly_order, PETSC_TRUE, coefficients, coefficients_rows, coefficients_cols));

      // Gmres polynomial with Newton basis without extra added roots - Can only use matrix-free
      } else if (inverse_type == PFLAREINV_NEWTON_NO_EXTRA) {

         // Does lots of reductions in parallel
         PetscCall(calculate_gmres_polynomial_roots_newton(mat_ctx_da->mat_scaled, poly_order, PETSC_FALSE, coefficients, coefficients_rows, coefficients_cols));
      }

      // Destroy the shell matrix
      if ((inverse_type == PFLAREINV_POWER ||
           inverse_type == PFLAREINV_ARNOLDI ||
           inverse_type == PFLAREINV_NEWTON ||
           inverse_type == PFLAREINV_NEWTON_NO_EXTRA) &&
           diag_scale_polys) {

         PetscCall(VecDestroy(&mat_ctx_da->mf_temp_vec[MF_VEC_DIAG]));
         PetscCall(MatDestroy(&mat_ctx_da->mat_scaled));
      }
      PetscCall(PetscFree(mat_ctx_da));
   }

   // If we ended up on a subcomm, we need to comm the coefficients back to
   // MPI_COMM_MATRIX
   // Both the arnoldi and newton basis have already finished their comms, so we can start an
   // all reduce here and conclude it in finish_approximate_inverse
   // The power basis however does not yet have its coefficients finished, so we have to do all the
   // comms in finish (although it doesn't really make sense to move onto a subcomm for the power basis
   // given it only does one reduction on MPI_COMM_MATRIX anyway)
   // Gmres polynomial with arnoldi basis
   if (buffers->on_subcomm) {

      // Get the comm
      PetscCall(PetscObjectGetComm((PetscObject)matrix, &MPI_COMM_MATRIX));
      buffers->request = MPI_REQUEST_NULL;

      if (inverse_type == PFLAREINV_ARNOLDI) {

         // We know rank 0 will always have the coefficients, just broadcast them to everyone on MPI_COMM_MATRIX
         // Some of the ranks on MPI_COMM_MATRIX will already have the coefficients, but I'm not going
         // to bother creating an intercommunicator to send the coefficients from any rank on the comm of buffers->matrix
         // to the comm of ranks which aren't in MPI_COMM_MATRIX but are in buffers->matrix
         PetscCall(PetscMPIIntCast(*coefficients_rows, &count));
         PetscCallMPI(MPI_Ibcast(*coefficients, count, MPIU_REAL, 0,
                  MPI_COMM_MATRIX, &buffers->request));

      // Gmres polynomial with Newton basis - Can only use matrix-free
      } else if (inverse_type == PFLAREINV_NEWTON || inverse_type == PFLAREINV_NEWTON_NO_EXTRA) {

         // The ranks on the subcomm may have resized coefficients to hold extra roots
         // so the ranks not on the subcomm need to match that size before the broadcast
         n_roots = *coefficients_rows;
         PetscCallMPI(MPI_Bcast(&n_roots, 1, MPIU_INT, 0, MPI_COMM_MATRIX));
         if (n_roots != *coefficients_rows) {
            PetscCall(PetscFree(*coefficients));
            *coefficients_rows = n_roots;
            *coefficients_cols = 2;
            PetscCall(PetscMalloc1(n_roots * 2, coefficients));
         }

         // Have to broadcast the 2D real and imaginary roots
         PetscCall(PetscMPIIntCast(*coefficients_rows * *coefficients_cols, &count));
         PetscCallMPI(MPI_Ibcast(*coefficients, count, MPIU_REAL, 0,
                  MPI_COMM_MATRIX, &buffers->request));
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode finish_approximate_inverse(Mat matrix, int inverse_type,
   int poly_order, int inverse_sparsity_order,
   tsqr_buffers *buffers, PetscReal coefficients[], PetscInt coefficients_rows, PetscInt coefficients_cols,
   PetscBool matrix_free, PetscBool diag_scale_polys, Mat *reuse_mat, Mat **reuse_submatrices, Mat *inv_matrix)
{
   // Finish the assembly of an approximate inverse

   // ~~~~~~
   PetscBool incomplete;
   PetscInt i_loc;
   // ~~~~~~

   PetscFunctionBegin;

   // ~~~~~~~~~~~~
   // For any calculations started in start_approximate_inverse that happen on
   // a subcomm, we need to finish the broadcasts
   // ~~~~~~~~~~~~
   if (buffers->on_subcomm && buffers->request != MPI_REQUEST_NULL) {

      // Finish the non-blocking comms
      PetscCallMPI(MPI_Wait(&buffers->request, MPI_STATUS_IGNORE));

      buffers->request = MPI_REQUEST_NULL;
   }

   // ~~~~~~~~~~~~

   // Gmres poylnomial with power or arnoldi basis
   if (inverse_type == PFLAREINV_POWER || inverse_type == PFLAREINV_ARNOLDI) {

      PetscCall(build_gmres_polynomial_inverse(matrix, poly_order,
               buffers, coefficients, coefficients_rows,
               inverse_sparsity_order, matrix_free, diag_scale_polys,
               reuse_mat, reuse_submatrices, inv_matrix));

   // Gmres polynomial with newton basis
   } else if (inverse_type == PFLAREINV_NEWTON || inverse_type == PFLAREINV_NEWTON_NO_EXTRA) {

      PetscCall(build_gmres_polynomial_newton_inverse(matrix, poly_order,
               coefficients, coefficients_rows,
               inverse_sparsity_order, matrix_free, diag_scale_polys, reuse_mat, reuse_submatrices,
               inv_matrix));

   // Neumann polynomial
   } else if (inverse_type == PFLAREINV_NEUMANN) {

      for (i_loc = 0; i_loc < coefficients_rows * coefficients_cols; i_loc++) coefficients[i_loc] = 1.0;
      PetscCall(calculate_and_build_neumann_polynomial_inverse(matrix, poly_order,
               buffers, coefficients, coefficients_rows, inverse_sparsity_order, matrix_free,
               reuse_mat, reuse_submatrices, inv_matrix));

   // Sparse approximate inverse
   } else if (inverse_type == PFLAREINV_SAI || inverse_type == PFLAREINV_ISAI) {

      // ~~~~~~~~~~~
      // ~~~~~~~~~~~

      if (inverse_type == PFLAREINV_SAI) {
         incomplete = PETSC_FALSE;

      // Incomplete sparse approximate inverse
      // This is equivalent to a one-level restricted additive schwartz
      // where each subdomain is a single unknown with the overlap
      // given by neighbouring unknowns
      } else {
         incomplete = PETSC_TRUE;
      }

      PetscCall(calculate_and_build_sai(matrix, inverse_sparsity_order, incomplete,
               reuse_mat, reuse_submatrices, inv_matrix));

   // Weighted jacobi
   } else if (inverse_type == PFLAREINV_WJACOBI) {

      PetscCall(calculate_and_build_weighted_jacobi_inverse(matrix, PFLAREINV_WJACOBI, inv_matrix));

   // Unweighted jacobi
   } else if (inverse_type == PFLAREINV_JACOBI) {

      PetscCall(calculate_and_build_weighted_jacobi_inverse(matrix, PFLAREINV_JACOBI, inv_matrix));

   // l1 jacobi
   } else if (inverse_type == PFLAREINV_L1JACOBI) {

      PetscCall(calculate_and_build_weighted_jacobi_inverse(matrix, PFLAREINV_L1JACOBI, inv_matrix));
   }

   // If we were on a subcomm we created a copy of our matrix, if not we incremented
   // the reference counter
   PetscCall(MatDestroy(&buffers->matrix));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode reset_inverse_mat(Mat *matrix)
{
   // Resets the matrix

   // ~~~~~~
   PetscInt i_loc;
   MatType mat_type;
   PetscBool is_shell;
   mat_ctxtype *mat_ctx, *mat_ctx_scaled;
   Mat temp_mat_local;
   // ~~~~~~

   PetscFunctionBegin;

   if (*matrix) {
      PetscCall(MatGetType(*matrix, &mat_type));
      PetscCall(PetscStrcmp(mat_type, MATSHELL, &is_shell));
      // If its a matshell, make sure to delete its ctx
      if (is_shell) {
         PetscCall(MatShellGetContext(*matrix, &mat_ctx));
         if (mat_ctx->own_coefficients) {
            PetscCall(PetscFree(mat_ctx->coefficients));
         }
         PetscCall(VecDestroy(&mat_ctx->mf_temp_vec[MF_VEC_TEMP]));

         // Both newton and neumann polynomials use some extra temporary vectors
         if (mat_ctx->mat_scaled || mat_ctx->real_roots) {

            PetscCall(VecDestroy(&mat_ctx->mf_temp_vec[MF_VEC_RHS]));
            PetscCall(VecDestroy(&mat_ctx->mf_temp_vec[MF_VEC_DIAG]));
            PetscCall(VecDestroy(&mat_ctx->mf_temp_vec[MF_VEC_TEMP_TWO]));
            PetscCall(VecDestroy(&mat_ctx->mf_temp_vec[MF_VEC_TEMP_THREE]));
         }

         // Any dense scratch blocks built by a multiple rhs (block) apply
         // These are only ever created by the block applies, so this is a no-op
         // for the jacobi/PCAIR matshells
         // MatDestroy nulls the slots in the context, so no dangling handles are
         // left in the context
         for (i_loc = 0; i_loc < 4; i_loc++) {
            if (mat_ctx->mf_temp_mat[i_loc]) {
               // The block kernels attach products that hold references between
               // the scratch mats - the horner ping-pong pair reference each
               // other, so plain destroys would leave a reference cycle alive -
               // clear the products first to break it
               // The parallel aij x dense numeric attaches its own inner product
               // to the local dense block, which recreates the same cycle between
               // the local blocks - clear that one too (for seq dense the local
               // matrix is the mat itself, so this is just a repeat clear)
               PetscCall(MatProductClear(mat_ctx->mf_temp_mat[i_loc]));
               PetscCall(MatDenseGetLocalMatrix(mat_ctx->mf_temp_mat[i_loc], &temp_mat_local));
               PetscCall(MatProductClear(temp_mat_local));
               PetscCall(MatDestroy(&mat_ctx->mf_temp_mat[i_loc]));
            }
         }
         PetscCall(VecDestroy(&mat_ctx->mf_vec_diag_recip));

         // Any transposed twin built by a transposed apply (MATOP_MULT_TRANSPOSE)
         // This is a no-op unless someone has actually done a PCApplyTranspose
         PetscCall(destroy_transpose_mat(mat_ctx));

         // Neumann polynomial has extra context that needs deleting
         if (mat_ctx->mat_scaled) {
            PetscCall(MatShellGetContext(mat_ctx->mat_scaled, &mat_ctx_scaled));
            PetscCall(PetscFree(mat_ctx_scaled));
            PetscCall(MatDestroy(&mat_ctx->mat_scaled));
         }
         PetscCall(PetscFree(mat_ctx));
      }
      PetscCall(MatDestroy(matrix));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}
