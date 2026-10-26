/*
  C port of Gmres_Poly_Apply.F90 (module gmres_poly_apply)

  The apply side of the matrix-free gmres polynomial inverses - everything
  that runs when a polynomial matshell built by gmres_poly/gmres_poly_newton
  is applied: the scalar matvec callbacks the matshells register with
  MatShellSetOperation, the horner/newton kernels they call, their multiple
  rhs (block) twins, and the block dispatch shell_poly_block_apply
  The construction of the polynomials stays in gmres_poly/gmres_poly_newton

  The context of every matshell is the mat_ctxtype of pflareimpl.h. The roots
  and coefficients are passed with their sizes and the loops over them are
  0-based; highest_nonzero_coeff keeps the Fortran meaning of its result (the
  number of coefficients up to and including the highest nonzero one)
*/

#include <petscmat.h>
#include "pflareimpl.h"
#include "Matshell_Data_Type.h"
#include "Gmres_Poly_Apply.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscInt highest_nonzero_coeff(const PetscReal coefficients[], PetscInt coefficients_size)
{
   // Returns the index of the highest order nonzero monomial coefficient, ie
   // the leading (highest order) zero coefficients are the only ones that
   // can be skipped when applying the polynomial
   // Returns zero if every coefficient is zero
   // The zero check is exactly 0d0, normally can only happen in an Arnoldi that
   // terminates early (or if the user has set the coefficients)
   // The index is 1-based as in the Fortran, ie it is the number of coefficients
   // up to and including the highest nonzero one

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PetscInt top_order;
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   for (top_order = coefficients_size; top_order >= 1; top_order--) {
      if (coefficients[top_order - 1] != 0.0) return top_order;
   }
   return 0;
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode petsc_matvec_da_poly_mf(Mat mat, Vec x, Vec y)
{
   // Applies D^-1 A as a shell
   // y = A x

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   mat_ctxtype *mat_ctx;
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(MatShellGetContext(mat, &mat_ctx));

   // ~~~~~~~~~~~~
   // We want to apply (D^-1 A) x
   // ~~~~~~~~~~~~

   // Multiply by A
   PetscCall(MatMult(mat_ctx->mat, x, y));

   // Doing D^-1 on the result
   PetscCall(VecPointwiseDivide(y, y, mat_ctx->mf_temp_vec[MF_VEC_DIAG]));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode petsc_matvec_ida_neumann_poly_mf(Mat mat, Vec x, Vec y)
{
   // Applies I-D^-1 A matrix-free
   // y = A x

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   mat_ctxtype *mat_ctx_scaled;
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(MatShellGetContext(mat, &mat_ctx_scaled));

   // ~~~~~~~~~~~~
   // We want to apply (I-D^-1 A) x
   // ~~~~~~~~~~~~

   // Multiply by A
   PetscCall(MatMult(mat_ctx_scaled->mat, x, y));

   // Doing D^-1 on the result
   PetscCall(VecPointwiseDivide(y, y, mat_ctx_scaled->mf_temp_vec[MF_VEC_DIAG]));

   // Now do x - D^-1 A x
   PetscCall(VecAXPBY(y, PFLARE_ONE, PFLARE_MINUS_ONE, x));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode petsc_matvec_poly_mf(Mat mat, Vec x, Vec y)
{
   // Applies a matrix polynomial matrix-free as an inverse
   // Just uses a Horner iteration to apply the mat_ctx->coefficients
   // to mat_ctx->mat in the input matshell
   // y = A x

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   mat_ctxtype *mat_ctx;
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(MatShellGetContext(mat, &mat_ctx));
   PetscCheck(mat_ctx->coefficients, PetscObjectComm((PetscObject)mat), PETSC_ERR_ARG_WRONGSTATE, "Polynomial coefficients in context aren't found");

   // Call the Horner iteration
   PetscCall(petsc_horner(mat_ctx->mat, mat_ctx->coefficients, mat_ctx->coefficients_size, mat_ctx->mf_temp_vec[MF_VEC_TEMP], x, y));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode petsc_matvec_right_scale_poly_mf(Mat mat, Vec x, Vec y)
{
   // Applies a polynomial matrix-free with a right diagonal scaling added
   // q(mat_ctx->mat_scaled) D^-1, as an inverse
   // y = A x

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   mat_ctxtype *mat_ctx;
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(MatShellGetContext(mat, &mat_ctx));
   PetscCheck(mat_ctx->coefficients, PetscObjectComm((PetscObject)mat), PETSC_ERR_ARG_WRONGSTATE, "Polynomial coefficients in context aren't found");

   // ~~~~~~~~~~~~
   // We want to apply q(mat_ctx->mat_scaled) D^-1
   // ~~~~~~~~~~~~

   // Doing rhs_copy = D^-1 x
   PetscCall(VecPointwiseDivide(mat_ctx->mf_temp_vec[MF_VEC_RHS], x, mat_ctx->mf_temp_vec[MF_VEC_DIAG]));

   // and now we call the horner method to apply our polynomial
   // q(mat_ctx->mat_scaled) to rhs_copy (D^-1 x)
   PetscCall(petsc_horner(mat_ctx->mat_scaled, mat_ctx->coefficients, mat_ctx->coefficients_size, mat_ctx->mf_temp_vec[MF_VEC_TEMP],
            mat_ctx->mf_temp_vec[MF_VEC_RHS], y));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode petsc_horner(Mat mat, const PetscReal coefficients[], PetscInt coefficients_size, Vec temp_vec, Vec x, Vec y)
{
   // Uses a horner iteration to apply
   // y = (coeff(1) + coeff(2) * A + coeff(3) * A^2 + ...) x

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PetscInt order, n_products, top_order;
   Vec cur_vec, other_vec, swap_vec;
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   PetscFunctionBegin;

   // ~~~~~~~
   // This applies the polynomial of order n as:
   // xn - x0 = q_n-1(A) r_0
   // where
   // q_n-1(z) = alpha_0 + alpha_1 * z + alpha_2 * z^2 + ...
   // So the q_n-1(A) r_0 term is just the repeated application of
   // y = alpha_n-1 r_0
   // do i = 1, n-1
   //   y = A * y + alpha_n-i-1 r_0
   // where the output y is xn - x0.
   // Practically, if we choose ksprichardson and then use this as a preconditioner B,
   // we are doing (where these n are the iteration count, not the order of the polynomial we're
   // applying above)
   // x^n+1 = x^n + B * r^n
   // so the x passed in should be the residual r^n, and we don't need to add x^n to
   // the solution, as the richardson is doing that for us. We have to ensure the richardson scale is one though.
   //
   // The iteration ping-pongs between temp_vec and y (like petsc_horner_block does
   // with its two dense temporaries), so each order is just a matvec and an axpy
   // with no copy. Each matvec writes into the other vector, so we count the
   // matvecs up front and choose which vector to start in so the final result
   // lands in y without a copy at the end either
   // ~~~~~~~

   // Only the leading (highest order) zero coefficients can be skipped, e.g., from an
   // Arnoldi that terminates early - once we've hit a nonzero coefficient every
   // lower order still needs its multiply by A, even if its own coefficient is zero
   // This has to match compute_mf_gmres_poly_num_matvecs
   top_order = highest_nonzero_coeff(coefficients, coefficients_size);

   // Count the matvecs we'll do - one per order below the highest nonzero order
   n_products = PetscMax(top_order - 1, 0);

   // An odd number of matvecs starting in temp_vec ends in y, an even number starting in y ends in y
   if (n_products % 2 == 1) {
      cur_vec = temp_vec;
      other_vec = y;
   } else {
      cur_vec = y;
      other_vec = temp_vec;
   }

   // Let's do the first cur = alpha_n-1 r_0 (ie the highest nonzero order term first)
   // If every coefficient is zero this just zeros y
   PetscCall(VecAXPBY(cur_vec, coefficients[PetscMax(top_order, 1) - 1], PFLARE_ZERO, x));

   // Loop down from the second highest nonzero order term down to the constant, one matvec
   // per order - a zeroth order polynomial has only the constant so never enters the loop
   for (order = top_order - 1; order >= 1; order--) {

      // other = A * cur
      PetscCall(MatMult(mat, cur_vec, other_vec));

      // Compute other = A * cur + alpha_n-i-1 r_0
      // Only the axpy can be skipped for a zero coefficient
      if (coefficients[order - 1] != 0.0) {
         PetscCall(VecAXPY(other_vec, coefficients[order - 1], x));
      }

      // The result of this order is the input of the next
      swap_vec = cur_vec;
      cur_vec = other_vec;
      other_vec = swap_vec;
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode ensure_transpose_mat(mat_ctxtype *mat_ctx)
{
   // Makes sure mat_ctx has a transposed twin, ie a matshell that applies the
   // transpose of whatever polynomial mat_ctx applies, and that the twin is
   // pointing at the current coefficients and diagonal
   //
   // All our polynomials have real coefficients, so
   //    q(A)^T = q(A^T)
   // and for the right diagonally scaled variants, using
   //    D^-1 (A^T D^-1)^k = (D^-1 A^T)^k D^-1
   // we get
   //    (q(D^-1 A) D^-1)^T = D^-1 q(A^T D^-1) = q(D^-1 A^T) D^-1
   // (and the same for the Neumann form with A replaced by I - D^-1 A, as
   // diag(A^T) = diag(A) so D is unchanged)
   // In other words the transpose has the identical shell structure with the
   // matrix replaced by its transpose, so the twin just runs the very same
   // forward callbacks against a MATTRANSPOSEVIRTUAL of mat_ctx->mat - none of
   // the horner/newton kernels need to know anything about transposes
   //
   // Note the inner operator the scaled twin needs is D^-1 A^T, which is NOT
   // mat_scaled^T = A^T D^-1, so we cannot get there by putting a
   // MATOP_MULT_TRANSPOSE on the existing inner matshell - the twin builds its
   // own inner matshell around the transposed matrix instead
   //
   // This is built lazily, so the very common case of never doing a transposed
   // apply costs nothing. Every rank reaches here through MatMultTranspose, so
   // the collective creation below is safe, in the same way the lazy
   // VecDuplicate in shell_poly_block_apply is

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PetscBool scaled;
   PetscInt local_rows, local_cols, global_rows, global_cols, i_loc;
   MPI_Comm MPI_COMM_MATRIX;
   VecType vtype;
   mat_ctxtype *twin, *twin_inner;
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   PetscFunctionBegin;

   // Are we applying q(D^-1 A) D^-1 (or the Neumann equivalent) rather than q(A)?
   scaled = mat_ctx->mat_scaled ? PETSC_TRUE : PETSC_FALSE;

   // If the matrix has been swapped out from under us the twin is stale - the
   // MATTRANSPOSEVIRTUAL holds a reference to the old matrix, so it would
   // silently keep applying it
   if (mat_ctx->transpose_mat) {
      if (mat_ctx->mf_transpose_source != mat_ctx->mat) {
         PetscCall(destroy_transpose_mat(mat_ctx));
      }
   }

   // ~~~~~~~~~~~~
   // Build the twin if we don't have one
   // ~~~~~~~~~~~~
   if (!mat_ctx->transpose_mat) {

      PetscCall(PetscObjectGetComm((PetscObject)mat_ctx->mat, &MPI_COMM_MATRIX));
      PetscCall(MatGetLocalSize(mat_ctx->mat, &local_rows, &local_cols));
      PetscCall(MatGetSize(mat_ctx->mat, &global_rows, &global_cols));
      PetscCall(MatGetVecType(mat_ctx->mat, &vtype));

      // Have to dynamically allocate this
      PetscCall(create_mat_ctx(&twin));

      // The transposed matrix - this is virtual, MatMult on it just calls
      // MatMultTranspose on mat_ctx->mat, so it costs no memory and no assembly
      // It takes a reference on mat_ctx->mat, dropped in destroy_transpose_mat
      PetscCall(MatCreateTranspose(mat_ctx->mat, &twin->mat));

      // The twin shell itself - the row/col sizes are written swapped to say
      // what we mean, they are equal as our inverses are always square
      PetscCall(MatCreateShell(MPI_COMM_MATRIX, local_cols, local_rows, global_cols, global_rows,
               twin, &mat_ctx->transpose_mat));

      // Give the twin the same forward callback the original shell has - that is
      // what makes it apply the same polynomial, just to the transposed matrix
      if (mat_ctx->real_roots) {
         if (scaled) {
            PetscCall(MatShellSetOperation(mat_ctx->transpose_mat,
                     MATOP_MULT, (PetscErrorCodeFn *)petsc_matvec_right_scale_poly_newton_mf));
         } else {
            PetscCall(MatShellSetOperation(mat_ctx->transpose_mat,
                     MATOP_MULT, (PetscErrorCodeFn *)petsc_matvec_poly_newton_mf));
         }
      } else {
         if (scaled) {
            PetscCall(MatShellSetOperation(mat_ctx->transpose_mat,
                     MATOP_MULT, (PetscErrorCodeFn *)petsc_matvec_right_scale_poly_mf));
         } else {
            PetscCall(MatShellSetOperation(mat_ctx->transpose_mat,
                     MATOP_MULT, (PetscErrorCodeFn *)petsc_matvec_poly_mf));
         }
      }

      PetscCall(MatAssemblyBegin(mat_ctx->transpose_mat, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(mat_ctx->transpose_mat, MAT_FINAL_ASSEMBLY));
      // Have to make sure to set the type of vectors the shell creates
      PetscCall(MatShellSetVecType(mat_ctx->transpose_mat, vtype));

      // The inner matshell, applying D^-1 A^T (or I - D^-1 A^T for the Neumann
      // polynomial) - the same callbacks the forward inner matshell uses, just
      // with the transposed matrix in the context
      if (scaled) {

         // Have to dynamically allocate this
         PetscCall(create_mat_ctx(&twin_inner));

         PetscCall(MatCreateShell(MPI_COMM_MATRIX, local_cols, local_rows, global_cols, global_rows,
                  twin_inner, &twin->mat_scaled));
         if (mat_ctx->neumann_inner) {
            PetscCall(MatShellSetOperation(twin->mat_scaled,
                     MATOP_MULT, (PetscErrorCodeFn *)petsc_matvec_ida_neumann_poly_mf));
         } else {
            PetscCall(MatShellSetOperation(twin->mat_scaled,
                     MATOP_MULT, (PetscErrorCodeFn *)petsc_matvec_da_poly_mf));
         }

         PetscCall(MatAssemblyBegin(twin->mat_scaled, MAT_FINAL_ASSEMBLY));
         PetscCall(MatAssemblyEnd(twin->mat_scaled, MAT_FINAL_ASSEMBLY));
         PetscCall(MatShellSetVecType(twin->mat_scaled, vtype));

         // The inner shell borrows the transposed matrix, the twin owns it
         twin_inner->mat = twin->mat;
      }

      // Remember what we built the twin from, so we can spot it going stale
      mat_ctx->mf_transpose_source = mat_ctx->mat;
   }

   // ~~~~~~~~~~~~
   // Refresh everything the twin borrows, every single apply
   // This is not cosmetic - the builders deallocate and reallocate the
   // coefficients whenever the matshell is reused with the same nonzero pattern
   // but fresh coefficients are computed, and the diagonal in MF_VEC_DIAG is
   // updated in place, with no hook that would let us know either has happened
   // Everything here is borrowed from mat_ctx and must never be freed by the twin
   // ~~~~~~~~~~~~
   PetscCall(MatShellGetContext(mat_ctx->transpose_mat, &twin));

   twin->coefficients = mat_ctx->coefficients;
   twin->coefficients_size = mat_ctx->coefficients_size;
   twin->real_roots = mat_ctx->real_roots;
   twin->imag_roots = mat_ctx->imag_roots;
   twin->roots_size = mat_ctx->roots_size;
   // The twin never owns the coefficients, mat_ctx does
   twin->own_coefficients = PETSC_FALSE;
   twin->neumann_inner = mat_ctx->neumann_inner;
   // A forward and a transposed apply never overlap, so the twin can share the
   // scratch vectors rather than duplicating them. They are the right layout as
   // our inverses are always square
   for (i_loc = 0; i_loc < 5; i_loc++) twin->mf_temp_vec[i_loc] = mat_ctx->mf_temp_vec[i_loc];

   if (scaled) {
      PetscCall(MatShellGetContext(twin->mat_scaled, &twin_inner));
      twin_inner->coefficients = mat_ctx->coefficients;
      twin_inner->coefficients_size = mat_ctx->coefficients_size;
      twin_inner->own_coefficients = PETSC_FALSE;
      twin_inner->neumann_inner = mat_ctx->neumann_inner;
      // diag(A^T) = diag(A), so the forward diagonal is exactly what the
      // transposed inner operator needs
      twin_inner->mf_temp_vec[MF_VEC_DIAG] = mat_ctx->mf_temp_vec[MF_VEC_DIAG];
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode destroy_transpose_mat(mat_ctxtype *mat_ctx)
{
   // Destroys the transposed twin built by ensure_transpose_mat, if there is one
   // The twin borrows the coefficients, the roots, the diagonal and all the
   // scratch vectors from mat_ctx, so all we own here are the matrices

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   mat_ctxtype *twin, *twin_inner;
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   PetscFunctionBegin;

   if (!mat_ctx->transpose_mat) PetscFunctionReturn(PETSC_SUCCESS);

   PetscCall(MatShellGetContext(mat_ctx->transpose_mat, &twin));

   // The inner matshell, if the twin has one - it borrows everything in its
   // context, so there is nothing but the shell to get rid of
   if (twin->mat_scaled) {
      PetscCall(MatShellGetContext(twin->mat_scaled, &twin_inner));
      PetscCall(PetscFree(twin_inner));
      PetscCall(MatDestroy(&twin->mat_scaled));
   }

   // The MATTRANSPOSEVIRTUAL, which drops the reference it holds on mat_ctx->mat
   PetscCall(MatDestroy(&twin->mat));

   PetscCall(PetscFree(twin));

   PetscCall(MatDestroy(&mat_ctx->transpose_mat));
   mat_ctx->mf_transpose_source = NULL;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode petsc_matvec_poly_transpose_mf(Mat mat, Vec x, Vec y)
{
   // Applies the transpose of a matrix-free polynomial inverse
   // This is the MATOP_MULT_TRANSPOSE of every polynomial matshell we build -
   // one routine covers all of them, because the twin built by
   // ensure_transpose_mat already carries the right forward arithmetic
   // y = A^T x

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   mat_ctxtype *mat_ctx;
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(MatShellGetContext(mat, &mat_ctx));

   // Build the transposed twin the first time round
   PetscCall(ensure_transpose_mat(mat_ctx));

   // The twin applies the same polynomial to the transposed matrix, which is
   // exactly the transpose of what mat applies
   PetscCall(MatMult(mat_ctx->transpose_mat, x, y));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode petsc_horner_block(mat_ctxtype *mat_ctx, Mat x_mat, Mat y_mat, Vec recip_diag, PetscBool neumann_inner, PetscBool *block_applied)
{
   // Uses a horner iteration to apply
   // y_mat = (coeff(1) + coeff(2) * B + coeff(3) * B^2 + ...) x_mat
   // for a block of right hand sides, x_mat, ie the multiple rhs version of petsc_horner
   // The matvecs of petsc_horner become sparse matrix-dense matrix products (SpMM)

   // The iteration ping-pongs between the two dense temporaries in the context,
   // so the products always target our own scratch (never the caller's y_mat)
   // and can stay attached between applies - only the numeric phase runs each
   // order, and the result is copied into y_mat once at the end

   // There are three different inner operators, B, we can apply, all of them built
   // out of real products with the *unscaled* mat in the context. We do this rather
   // than running the products on the matshells the scalar versions use, as every
   // product on a shell degrades to a column by column matvec
   // 1) recip_diag null                : B = A, the plain polynomial q(A)
   // 2) recip_diag non-null            : B = D^-1 A, the diagonally scaled polynomial
   //                                     q(D^-1 A), with recip_diag = D^-1
   // 3) neumann_inner (implies scaled) : B = I - D^-1 A, the Neumann polynomial
   // In the scaled cases the caller must have already scaled the block of rhs,
   // ie x_mat = D^-1 X

   // block_applied comes back false (with y_mat untouched) if the block of rhs we've
   // been given has no product with mat, so the caller can fall back to a column by
   // column apply

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PetscInt order, top_order;
   PetscBool scaled;
   Mat cur_mat, other_mat, swap_mat;
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   PetscFunctionBegin;

   *block_applied = PETSC_FALSE;
   scaled = recip_diag ? PETSC_TRUE : PETSC_FALSE;

   // Only the leading (highest order) zero coefficients can be skipped - once we've
   // hit a nonzero coefficient every lower order still needs its product
   top_order = highest_nonzero_coeff(mat_ctx->coefficients, mat_ctx->coefficients_size);

   // A zeroth order polynomial has no products, just the scaled copy
   // (if every coefficient is zero this just zeros y_mat)
   if (top_order <= 1) {
      PetscCall(MatCopy(x_mat, y_mat, SAME_NONZERO_PATTERN));
      PetscCall(MatScale(y_mat, mat_ctx->coefficients[0]));
      *block_applied = PETSC_TRUE;
      PetscFunctionReturn(PETSC_SUCCESS);
   }

   // For a first order polynomial or above we do an extra product per order,
   // alternating between the two temporaries - make sure both products are
   // attached (they stay attached between applies)
   PetscCall(ensure_block_kernel_products(mat_ctx, MF_MAT_TEMP, block_applied));
   if (!*block_applied) PetscFunctionReturn(PETSC_SUCCESS);
   *block_applied = PETSC_FALSE;

   // Let's do the first cur = alpha_n-1 r_0 (ie the highest nonzero order term first)
   cur_mat = mat_ctx->mf_temp_mat[MF_MAT_TEMP];
   other_mat = mat_ctx->mf_temp_mat[MF_MAT_TEMP_TWO];
   PetscCall(MatCopy(x_mat, cur_mat, SAME_NONZERO_PATTERN));
   PetscCall(MatScale(cur_mat, mat_ctx->coefficients[top_order - 1]));

   // Loop down from the second highest nonzero order term down to the constant
   for (order = top_order - 1; order >= 1; order--) {

      // other = A * cur - each temporary's attached product reads the other
      // temporary, so which product runs is decided by which one is cur
      PetscCall(MatProductNumeric(other_mat));

      // This is the arithmetic of the D^-1 A matshell, done blockwise
      if (scaled) PetscCall(MatDiagonalScale(other_mat, recip_diag, NULL));

      // The inner operator is I - D^-1 A rather than D^-1 A, so finish
      // other = cur - D^-1 A cur
      // cur still holds the value we did the product with
      // This is the arithmetic of the I - D^-1 A matshell, done blockwise
      if (neumann_inner) {
         PetscCall(MatScale(other_mat, PFLARE_MINUS_ONE));
         PetscCall(MatAXPY(other_mat, PFLARE_ONE, cur_mat, SAME_NONZERO_PATTERN));
      }

      // Compute other = B * cur + alpha_n-i-1 r_0
      // Only the axpy can be skipped for a zero coefficient
      if (mat_ctx->coefficients[order - 1] != 0.0) {
         PetscCall(MatAXPY(other_mat, mat_ctx->coefficients[order - 1], x_mat, SAME_NONZERO_PATTERN));
      }

      // The result of this order is the input of the next
      swap_mat = cur_mat;
      cur_mat = other_mat;
      other_mat = swap_mat;
   }

   PetscCall(MatCopy(cur_mat, y_mat, SAME_NONZERO_PATTERN));

   *block_applied = PETSC_TRUE;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode petsc_matvec_poly_newton_mf(Mat mat, Vec x, Vec y)
{
   // Applies a matrix polynomial matrix-free as an inverse
   // y = A x

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   mat_ctxtype *mat_ctx;
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(MatShellGetContext(mat, &mat_ctx));
   PetscCheck(mat_ctx->real_roots, PetscObjectComm((PetscObject)mat), PETSC_ERR_ARG_WRONGSTATE, "Polynomial coefficients in context aren't found");

   // Apply polynomial
   PetscCall(petsc_newton(mat_ctx->mat,
            mat_ctx->real_roots, mat_ctx->imag_roots, mat_ctx->roots_size,
            mat_ctx->mf_temp_vec[MF_VEC_TEMP],
            mat_ctx->mf_temp_vec[MF_VEC_TEMP_TWO],
            mat_ctx->mf_temp_vec[MF_VEC_TEMP_THREE],
            x, y));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode petsc_matvec_right_scale_poly_newton_mf(Mat mat, Vec x, Vec y)
{
   // Applies a Newton polynomial matrix-free with a right diagonal scaling added
   // q(mat_ctx->mat_scaled) D^-1, as an inverse
   // y = A x

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   mat_ctxtype *mat_ctx;
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(MatShellGetContext(mat, &mat_ctx));
   PetscCheck(mat_ctx->real_roots, PetscObjectComm((PetscObject)mat), PETSC_ERR_ARG_WRONGSTATE, "Polynomial coefficients in context aren't found");

   // ~~~~~~~~~~~~
   // We want to apply q(mat_ctx->mat_scaled) D^-1
   // ~~~~~~~~~~~~

   // Doing MF_VEC_RHS = D^-1 x
   PetscCall(VecPointwiseDivide(mat_ctx->mf_temp_vec[MF_VEC_RHS], x, mat_ctx->mf_temp_vec[MF_VEC_DIAG]));

   // and now we apply our polynomial
   // q(mat_ctx->mat_scaled) to MF_VEC_RHS (D^-1 x)
   PetscCall(petsc_newton(mat_ctx->mat_scaled,
            mat_ctx->real_roots, mat_ctx->imag_roots, mat_ctx->roots_size,
            mat_ctx->mf_temp_vec[MF_VEC_TEMP],
            mat_ctx->mf_temp_vec[MF_VEC_TEMP_TWO],
            mat_ctx->mf_temp_vec[MF_VEC_TEMP_THREE],
            mat_ctx->mf_temp_vec[MF_VEC_RHS], y));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode petsc_matvec_poly_newton_residual_mf(Mat mat, Vec x, Vec y)
{
   // Applies a matrix polynomial matrix-free as an inverse and computes the residual
   // y = A x

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   mat_ctxtype *mat_ctx;
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(MatShellGetContext(mat, &mat_ctx));
   PetscCheck(mat_ctx->real_roots, PetscObjectComm((PetscObject)mat), PETSC_ERR_ARG_WRONGSTATE, "Polynomial coefficients in context aren't found");

   // Apply polynomial
   PetscCall(petsc_newton_residual(mat_ctx->mat,
            mat_ctx->real_roots, mat_ctx->imag_roots, mat_ctx->roots_size,
            mat_ctx->mf_temp_vec[MF_VEC_TEMP],
            mat_ctx->mf_temp_vec[MF_VEC_TEMP_TWO],
            mat_ctx->mf_temp_vec[MF_VEC_TEMP_THREE],
            x, y));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode petsc_matvec_right_scale_poly_newton_residual_mf(Mat mat, Vec x, Vec y)
{
   // Applies a Newton polynomial matrix-free with a right diagonal scaling added
   // q(mat_ctx->mat_scaled) D^-1, as an inverse and computes the residual
   // y = A x

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   mat_ctxtype *mat_ctx;
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(MatShellGetContext(mat, &mat_ctx));
   PetscCheck(mat_ctx->real_roots, PetscObjectComm((PetscObject)mat), PETSC_ERR_ARG_WRONGSTATE, "Polynomial coefficients in context aren't found");

   // ~~~~~~~~~~~~
   // We want to apply q(mat_ctx->mat_scaled) D^-1 and then compute a residual
   // ~~~~~~~~~~~~

   // Doing MF_VEC_RHS = D^-1 x
   PetscCall(VecPointwiseDivide(mat_ctx->mf_temp_vec[MF_VEC_RHS], x, mat_ctx->mf_temp_vec[MF_VEC_DIAG]));

   // and now we apply our polynomial
   // q(mat_ctx->mat_scaled) to MF_VEC_RHS (D^-1 x)
   PetscCall(petsc_newton_residual(mat_ctx->mat_scaled,
            mat_ctx->real_roots, mat_ctx->imag_roots, mat_ctx->roots_size,
            mat_ctx->mf_temp_vec[MF_VEC_TEMP],
            mat_ctx->mf_temp_vec[MF_VEC_TEMP_TWO],
            mat_ctx->mf_temp_vec[MF_VEC_TEMP_THREE],
            mat_ctx->mf_temp_vec[MF_VEC_RHS], y));

   PetscCall(VecPointwiseMult(y, y, mat_ctx->mf_temp_vec[MF_VEC_DIAG]));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode petsc_newton(Mat mat, const PetscReal real_roots[], const PetscReal imag_roots[], PetscInt roots_size,
   Vec temp_vec, Vec temp_vec_two, Vec temp_vec_three, Vec x, Vec y)
{
   // Applies a gmres polynomial in the newton basis matrix-free as an inverse
   // The roots are stored in real_roots, imag_roots in the input matshell
   // Based on Loe 2021 Toward efficient polynomial preconditioning for GMRES
   // This is Algorithm 3 in Loe
   // y = A x

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PetscInt i;
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   PetscFunctionBegin;

   // temp_vec = x
   PetscCall(VecCopy(x, temp_vec));
   // y = 0
   PetscCall(VecSet(y, PFLARE_ZERO));

   // ~~~~~~~~~~~~
   // Iterate over the i
   // ~~~~~~~~~~~~
   i = 0;
   while (i < roots_size - 1) {

      // If real this is easy
      if (imag_roots[i] == 0.0) {

         // Skips zero eigenvalues - calculate_gmres_polynomial_roots_newton
         // sets numerically zero roots to exactly zero (relative to ||H||)
         if (real_roots[i] == 0.0) {
            i = i + 1;
            continue;
         }

         // y = y + theta_i * temp_vec
         PetscCall(VecAXPY(y, PFLARE_ONE / real_roots[i], temp_vec));

         // temp_vec_two = A * temp_vec
         PetscCall(MatMult(mat, temp_vec, temp_vec_two));
         // temp_vec = temp_vec - theta_i * temp_vec_two
         PetscCall(VecAXPY(temp_vec, PFLARE_MINUS_ONE / real_roots[i], temp_vec_two));

         i = i + 1;

      // If imaginary, then have to combine the e'val and its
      // complex conjugate to keep the arithmetic real
      // Relies on the complex conjugate being next to each other
      } else {

         // A complex pair is never zero as its imaginary part is nonzero

         // temp_vec_two = A * temp_vec
         PetscCall(MatMult(mat, temp_vec, temp_vec_two));
         // temp_vec_two = 2 * Re(theta_i) * temp_vec - temp_vec_two
         PetscCall(VecAXPBY(temp_vec_two, 2 * real_roots[i], PFLARE_MINUS_ONE, temp_vec));

         // y = y + 1/(Re(theta_i)^2 + Imag(theta_i)^2) * temp_vec_two
         PetscCall(VecAXPY(y, PFLARE_ONE / (real_roots[i] * real_roots[i] + imag_roots[i] * imag_roots[i]), temp_vec_two));

         if (i < roots_size - 2) {
            // temp_vec_three = A * temp_vec_two
            PetscCall(MatMult(mat, temp_vec_two, temp_vec_three));

            // temp_vec = temp_vec - 1/(Re(theta_i)^2 + Imag(theta_i)^2) * temp_vec_three
            PetscCall(VecAXPY(temp_vec, PFLARE_MINUS_ONE / (real_roots[i] * real_roots[i] + imag_roots[i] * imag_roots[i]), temp_vec_three));
         }

         // Skip two evals
         i = i + 2;
      }
   }

   // Final step if last root is real
   if (imag_roots[roots_size - 1] == 0.0) {

      // Skips zero eigenvalues
      if (real_roots[roots_size - 1] != 0.0) {

         // y = y + theta_i * temp_vec
         PetscCall(VecAXPBY(y, PFLARE_ONE / real_roots[roots_size - 1], PFLARE_ONE, temp_vec));
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode petsc_newton_residual(Mat mat, const PetscReal real_roots[], const PetscReal imag_roots[], PetscInt roots_size,
   Vec temp_vec, Vec temp_vec_two, Vec temp_vec_three, Vec x, Vec y)
{
   // Applies a gmres residual polynomial in the newton basis matrix-free as an inverse
   // This is different than petsc_newton which applies p(A)v,
   // whereas this routine applies pi(A)v
   // This is (a slightly modified) Algorithm 1 in Loe and saves some flops when we don't need the solution
   // just the residual
   // y = A x

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PetscInt order;
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   PetscFunctionBegin;

   (void)temp_vec_three;

   // y = x
   PetscCall(VecCopy(x, y));

   // ~~~~~~~~~~~~
   // Iterate over the order
   // ~~~~~~~~~~~~
   order = 0;
   // Does every e'val in this loop unlike when we apply p(A)v
   while (order < roots_size) {

      // If real this is easy
      if (imag_roots[order] == 0.0) {

         // Skips zero eigenvalues - calculate_gmres_polynomial_roots_newton
         // sets numerically zero roots to exactly zero (relative to ||H||)
         if (real_roots[order] == 0.0) {
            order = order + 1;
            continue;
         }

         // temp_vec_two = A * y
         PetscCall(MatMult(mat, y, temp_vec_two));

         // y = y - theta_i * temp_vec_two
         PetscCall(VecAXPY(y, PFLARE_MINUS_ONE / real_roots[order], temp_vec_two));

         order = order + 1;

      // If imaginary, then have to combine the e'val and its
      // complex conjugate to keep the arithmetic real
      // Relies on the complex conjugate being next to each other
      } else {

         // A complex pair is never zero as its imaginary part is nonzero

         // temp_vec_two = A * y
         PetscCall(MatMult(mat, y, temp_vec_two));

         // temp_vec = A * temp_vec_two
         PetscCall(MatMult(mat, temp_vec_two, temp_vec));

         // temp_vec = temp_vec - 2 * Re(theta_i) * temp_vec_two
         PetscCall(VecAXPY(temp_vec, -2 * real_roots[order], temp_vec_two));

         // y = y + 1/(Re(theta_i)^2 + Imag(theta_i)^2) * temp_vec
         PetscCall(VecAXPY(y, PFLARE_ONE / (real_roots[order] * real_roots[order] + imag_roots[order] * imag_roots[order]), temp_vec));

         // Skip two evals
         order = order + 2;
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode petsc_newton_block(mat_ctxtype *mat_ctx, Mat x_mat, Mat y_mat, Vec recip_diag, PetscBool *block_applied)
{
   // Applies a gmres polynomial in the newton basis matrix-free as an inverse
   // for a block of right hand sides, x_mat, ie the multiple rhs version of petsc_newton
   // The roots are stored in real_roots, imag_roots in the input matshell
   // Based on Loe 2021 Toward efficient polynomial preconditioning for GMRES
   // This is Algorithm 3 in Loe
   // The matvecs of petsc_newton become sparse matrix-dense matrix products (SpMM)
   // y_mat = A x_mat

   // If recip_diag is not null we are applying the diagonally scaled polynomial
   // q(D^-1 A), with the mat passed in the *unscaled* A and recip_diag = D^-1.
   // We do this rather than running the products on the D^-1 A matshell, as every
   // product on a shell degrades to a column by column matvec. The caller must have
   // already scaled the block of rhs, ie x_mat = D^-1 X
   // Every product below is therefore scaled by D^-1 immediately after it is computed
   // and before the result is used for anything else - that is exactly the arithmetic
   // of the inner D^-1 A matshell, done blockwise

   // block_applied comes back false (with y_mat untouched) if the block of rhs we've
   // been given has no product with mat, so the caller can fall back to a column by
   // column apply

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   const PetscReal *real_roots, *imag_roots;
   Mat temp_mat, temp_mat_two, temp_mat_three;
   PetscInt i, nroots;
   PetscBool scaled;
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   PetscFunctionBegin;

   *block_applied = PETSC_FALSE;
   scaled = recip_diag ? PETSC_TRUE : PETSC_FALSE;
   real_roots = mat_ctx->real_roots;
   imag_roots = mat_ctx->imag_roots;
   nroots = mat_ctx->roots_size;

   temp_mat       = mat_ctx->mf_temp_mat[MF_MAT_TEMP];
   temp_mat_two   = mat_ctx->mf_temp_mat[MF_MAT_TEMP_TWO];
   temp_mat_three = mat_ctx->mf_temp_mat[MF_MAT_TEMP_THREE];

   // If we have more than one root we have to do products as we iterate over
   // the roots - make sure they are attached (they stay attached between
   // applies, only the numeric phase runs below)
   // This has to happen before we write any values into the temporaries, as the
   // symbolic product may set them up
   if (nroots > 1) {
      PetscCall(ensure_block_kernel_products(mat_ctx, MF_MAT_TEMP_THREE, block_applied));
      if (!*block_applied) PetscFunctionReturn(PETSC_SUCCESS);
      *block_applied = PETSC_FALSE;
   }

   // temp_mat = x_mat
   PetscCall(MatCopy(x_mat, temp_mat, SAME_NONZERO_PATTERN));
   // y_mat = 0
   PetscCall(MatZeroEntries(y_mat));

   // ~~~~~~~~~~~~
   // Iterate over the i
   // ~~~~~~~~~~~~
   i = 0;
   while (i < nroots - 1) {

      // If real this is easy
      if (imag_roots[i] == 0.0) {

         // Skips zero eigenvalues - calculate_gmres_polynomial_roots_newton
         // sets numerically zero roots to exactly zero (relative to ||H||)
         if (real_roots[i] == 0.0) {
            i = i + 1;
            continue;
         }

         // y_mat = y_mat + theta_i * temp_mat
         PetscCall(MatAXPY(y_mat, PFLARE_ONE / real_roots[i], temp_mat, SAME_NONZERO_PATTERN));

         // temp_mat_two = A * temp_mat
         PetscCall(MatProductNumeric(temp_mat_two));
         if (scaled) PetscCall(MatDiagonalScale(temp_mat_two, recip_diag, NULL));

         // temp_mat = temp_mat - theta_i * temp_mat_two
         PetscCall(MatAXPY(temp_mat, PFLARE_MINUS_ONE / real_roots[i], temp_mat_two, SAME_NONZERO_PATTERN));

         i = i + 1;

      // If imaginary, then have to combine the e'val and its
      // complex conjugate to keep the arithmetic real
      // Relies on the complex conjugate being next to each other
      } else {

         // A complex pair is never zero as its imaginary part is nonzero

         // temp_mat_two = A * temp_mat
         PetscCall(MatProductNumeric(temp_mat_two));
         if (scaled) PetscCall(MatDiagonalScale(temp_mat_two, recip_diag, NULL));

         // temp_mat_two = 2 * Re(theta_i) * temp_mat - temp_mat_two
         PetscCall(MatScale(temp_mat_two, PFLARE_MINUS_ONE));
         PetscCall(MatAXPY(temp_mat_two, PFLARE_TWO * real_roots[i], temp_mat, SAME_NONZERO_PATTERN));

         // y_mat = y_mat + 1/(Re(theta_i)^2 + Imag(theta_i)^2) * temp_mat_two
         PetscCall(MatAXPY(y_mat, PFLARE_ONE / (real_roots[i] * real_roots[i] + imag_roots[i] * imag_roots[i]), temp_mat_two, SAME_NONZERO_PATTERN));

         if (i < nroots - 2) {
            // temp_mat_three = A * temp_mat_two
            PetscCall(MatProductNumeric(temp_mat_three));
            if (scaled) PetscCall(MatDiagonalScale(temp_mat_three, recip_diag, NULL));

            // temp_mat = temp_mat - 1/(Re(theta_i)^2 + Imag(theta_i)^2) * temp_mat_three
            PetscCall(MatAXPY(temp_mat, PFLARE_MINUS_ONE / (real_roots[i] * real_roots[i] + imag_roots[i] * imag_roots[i]), temp_mat_three, SAME_NONZERO_PATTERN));
         }

         // Skip two evals
         i = i + 2;
      }
   }

   // Final step if last root is real
   if (imag_roots[nroots - 1] == 0.0) {

      // Skips zero eigenvalues
      if (real_roots[nroots - 1] != 0.0) {

         // y_mat = y_mat + theta_i * temp_mat
         PetscCall(MatAXPY(y_mat, PFLARE_ONE / real_roots[nroots - 1], temp_mat, SAME_NONZERO_PATTERN));
      }
   }

   *block_applied = PETSC_TRUE;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode shell_poly_block_apply(Mat shell_mat, Mat x_mat, Mat y_mat, PetscBool *block_applied)
{
   // Applies one of the matrix-free gmres polynomial inverses to a block of
   // right hand sides, x_mat, ie the multiple rhs version of the matvecs
   // petsc_matvec_poly_mf / petsc_matvec_right_scale_poly_mf and their newton equivalents
   // The shell_mat passed in is the matshell built by build_gmres_polynomial_inverse
   // (or its newton equivalent) and we dispatch on what its context contains

   // block_applied comes back false (with y_mat untouched) whenever we can't do a
   // block apply, and the caller must then apply column by column instead

   // The polynomials with a right diagonal scaling store an inner matshell in
   // mat_ctx->mat_scaled and the block kernels below deliberately bypass it - they do
   // real products with the unscaled mat_ctx->mat and then apply the rest of the inner
   // shell's arithmetic themselves (a product on a shell would degrade to a column by
   // column matvec). That means the kernels have to be told which inner operator they
   // are emulating - mat_ctx->neumann_inner picks I - D^-1 A (the Neumann polynomial,
   // see Neumann_Poly.c) rather than D^-1 A (the diagonally scaled gmres polynomials)
   // Any new inner operator variant must add to that flag rather than reuse it,
   // otherwise the bypass will silently give the wrong answer

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PetscInt n_temps;
   PetscBool scaled;
   mat_ctxtype *mat_ctx;
   // The block of rhs and the D^-1 we hand to the kernels
   // kernel_recip is left null in the unscaled case
   Mat kernel_x;
   Vec kernel_recip = NULL;
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   PetscFunctionBegin;

   *block_applied = PETSC_FALSE;
   PetscCall(MatShellGetContext(shell_mat, &mat_ctx));

   // Are we applying q(D^-1 A) D^-1 rather than just q(A)?
   scaled = mat_ctx->mat_scaled ? PETSC_TRUE : PETSC_FALSE;

   kernel_x = x_mat;

   if (scaled) {

      // Lazily build somewhere to store D^-1
      // The new vec goes straight into the context, so we don't create
      // (and leak) a new vec on every apply
      if (!mat_ctx->mf_vec_diag_recip) {
         PetscCall(VecDuplicate(mat_ctx->mf_temp_vec[MF_VEC_DIAG], &mat_ctx->mf_vec_diag_recip));
      }
      // We have to refresh the values every apply - the diagonal in MF_VEC_DIAG is
      // updated in place whenever the matshell is reused with the same nonzero
      // pattern, and there is no hook that would let us know that has happened
      PetscCall(VecCopy(mat_ctx->mf_temp_vec[MF_VEC_DIAG], mat_ctx->mf_vec_diag_recip));
      PetscCall(VecReciprocal(mat_ctx->mf_vec_diag_recip));
      kernel_recip = mat_ctx->mf_vec_diag_recip;
   }

   // ~~~~~~~~~~~~
   // Dispatch on what sort of polynomial is in the context
   // The newton basis needs three dense temporaries, the power/arnoldi/neumann
   // basis two (the horner iteration ping-pongs between them)
   // The Neumann polynomial has to be checked first, as it also has coefficients
   // ~~~~~~~~~~~~
   if (mat_ctx->neumann_inner) {
      // The Neumann polynomial is q(I - D^-1 A) D^-1 with coefficients of one and is
      // always built with its inner matshell, so we should always have a scaled apply
      // here - don't rely on that though, we can't emulate the inner operator without
      // the diagonal
      if (!scaled) PetscFunctionReturn(PETSC_SUCCESS);
      if (!mat_ctx->coefficients) PetscFunctionReturn(PETSC_SUCCESS);
      n_temps = 2;
   } else if (mat_ctx->real_roots) {
      n_temps = 3;
   } else if (mat_ctx->coefficients) {
      n_temps = 2;
   } else {
      // Nothing we know how to apply blockwise
      PetscFunctionReturn(PETSC_SUCCESS);
   }

   // Make the dense temporaries, plus somewhere to put D^-1 X if we need it
   PetscCall(ensure_block_temp_mats(mat_ctx, x_mat, n_temps, &scaled));

   // Do the right diagonal scaling on the whole block, ie MF_MAT_RHS = D^-1 X
   if (scaled) {
      PetscCall(MatCopy(x_mat, mat_ctx->mf_temp_mat[MF_MAT_RHS], SAME_NONZERO_PATTERN));
      PetscCall(MatDiagonalScale(mat_ctx->mf_temp_mat[MF_MAT_RHS], kernel_recip, NULL));
      kernel_x = mat_ctx->mf_temp_mat[MF_MAT_RHS];
   }

   if (mat_ctx->neumann_inner) {

      // Neumann polynomial - horner with an inner operator of I - D^-1 A
      PetscCall(petsc_horner_block(mat_ctx, kernel_x, y_mat, kernel_recip, PETSC_TRUE, block_applied));

   } else if (mat_ctx->real_roots) {

      // Newton basis
      PetscCall(petsc_newton_block(mat_ctx, kernel_x, y_mat, kernel_recip, block_applied));

   } else {

      // Power/arnoldi basis
      PetscCall(petsc_horner_block(mat_ctx, kernel_x, y_mat, kernel_recip, PETSC_FALSE, block_applied));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}
