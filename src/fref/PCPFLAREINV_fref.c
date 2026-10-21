/*
  The reference PC type pflareinv_fref, registered only in PFLARE_PORT_DEBUG builds
  This is src/PCPFLAREINV.c as it was before the approximate inverses were ported to
  C (stage 5 of docs/dev/fortran_removal_plan.md), calling the Fortran reference
  implementations in src/fref through the _fref bindings in C_Fortran_Bindings.F90,
  so tests/port_compare_pc.c can compare the pflareinv PC against it. The public
  PCPFLAREINV API in src/PCPFLAREINV.c works on it too, through the composed methods
  Only the names have changed: the type name, the statics and the bindings have an
  _fref suffix, and the public API functions and their documentation are left out
*/

// Include the petsc header files
#include <petsc/private/pcimpl.h>
#include "pflare.h"

/* SUBMANSEC = PC */

// Defined in C_Fortran_Bindings.F90
PETSC_EXTERN void reset_inverse_mat_c_fref(Mat *mat);
// Whether an inverse type is a polynomial, ie has coefficients and builds a matshell when
// applied matrix-free - the non-polynomial types (SAI, ISAI, WJacobi, Jacobi, L1Jacobi) have no
// coefficients and are always assembled. *is_polynomial comes back as 1 or 0
PETSC_EXTERN void inverse_type_is_polynomial_c_fref(int inverse_type, int *is_polynomial);
// coeffs_ptr/row_size/col_size are in/out:
//   *coeffs_ptr == NULL on entry  -> fresh: Fortran allocates, writes c_loc to *coeffs_ptr on return
//   *coeffs_ptr != NULL on entry  -> reuse: existing coefficients used, polynomial step skipped
// The by-value arguments are integer(c_int) on the Fortran side, so must be int here
// (not PetscInt, which is 64-bit with --with-64-bit-indices)
PETSC_EXTERN void calculate_and_build_approximate_inverse_c_fref(Mat *input_mat, int inverse_type, int order, \
                     int sparsity_order, int matrix_free_int, int diag_scale_polys_int, \
                     int subcomm_int, \
                     PetscReal **coeffs_ptr, PetscInt *row_size, PetscInt *col_size, \
                     Mat *inv_matrix);
// Block (multiple rhs) apply of a matrix-free polynomial matshell, Y = q(A) X.
// *applied comes back 0 if no block apply was possible and the caller has to apply
// column by column instead - that includes being handed a matshell whose context
// isn't one of the polynomials shell_poly_block_apply (Gmres_Poly_Newton.F90) knows.
PETSC_EXTERN void pflareinv_shell_block_matapply_c_fref(Mat *mat, Mat *X, Mat *Y, int *applied);

// The types available as approximate inverses are (see include/pflare.h):
//
// PFLAREINV_POWER      - GMRES polynomial with the power basis 
// PFLAREINV_ARNOLDI    - GMRES polynomial with the arnoldi basis 
// PFLAREINV_NEWTON     - GMRES polynomial with the newton basis with extra roots for stability 
// PFLAREINV_NEWTON_NO_EXTRA     - GMRES polynomial with the newton basis without extra roots    
// PFLAREINV_NEUMANN    - Neumann polynomial
// PFLAREINV_SAI        - SAI - cannot be used matrix-free atm
// PFLAREINV_ISAI       - Incomplete SAI - cannot be used matrix-free atm
//                           ie a one-level restricted additive schwartz where each unknown is its own 
//                           subdomain with overlap given by the connectivity of that unknown in matrix powers
// PFLAREINV_WJACOBI    - Weighted Jacobi - The Jacobi types really only exist for use in PCAIR, if you want to use 
//                           a Jacobi PC just use the existing one in petsc
// PFLAREINV_JACOBI     - Unweighted Jacobi - 
// PFLAREINV_L1JACOBI   - l1 Jacobi - 

 /*
    Private context (data structure) for the PFLAREINV preconditioner.
 */
typedef struct {
   // Stores the mat which is our inverse
   Mat mat_inverse;

   // What type of inverse to apply
   int inverse_type;
   // The polynomial order
   int poly_order;
   // The power of the sparsity we use (if assembled)
   int inverse_sparsity_order;
   // Whether or not the mat_inverse is just a matshell and applied
   // matrix free
   PetscBool matrix_free;
   int subcomm_int;

   // Stored polynomial coefficients (column-major).
   // Populated after every PCSetUp call. Memory is either Fortran-allocated (from
   // calculate_and_build_approximate_inverse_c_fref) or C-malloc'd (from SetPolyCoeffs);
   // in both cases a plain C free() is correct because Fortran allocate uses the C heap.
   // Populated/updated after every PCSetUp call
   PetscReal *poly_coeffs;
   PetscInt   poly_coeffs_rows;
   PetscInt   poly_coeffs_cols;
   // If PETSC_TRUE and SAME_NONZERO_PATTERN, skip coefficient recomputation
   PetscBool  reuse_poly_coeffs;

   // Dense work matrix for the assembled PCMatApply, with the product
   // block_work = mat_inverse * X attached so repeat applies only run the numeric
   // phase - the symbolic phase of the mpi product builds the dense scatter every
   // time it runs. Lazily built on the first PCMatApply and torn down by PCReset.
   // The product keeps the last X it was bound to (and the mat_inverse it was set
   // up with) alive until the next apply or reset.
   Mat        block_work;
   // The mat_inverse (and its nonzero state - the mpi scatter is built from its
   // sparsity) and the lda of X the product on block_work was set up with. The
   // product holds a reference to that mat_inverse so the handle comparison is safe
   Mat              block_work_inverse;
   PetscObjectState block_work_inverse_nz_state;
   PetscInt         block_work_lda;

} PC_PFLAREINV_fref;

// ~~~~~~~~~~

// Drops the cached block work matrix (and the product attached to it, which is
// cleared by MatDestroy) so the next PCMatApply sets the product up again
static PetscErrorCode PCPFLAREINVResetBlockWork_Private_fref(PC_PFLAREINV_fref *inv_data)
{
   PetscFunctionBegin;
   PetscCall(MatDestroy(&inv_data->block_work));
   inv_data->block_work_inverse          = NULL;
   inv_data->block_work_inverse_nz_state = 0;
   inv_data->block_work_lda              = -1;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

static PetscErrorCode PCReset_PFLAREINV_fref(PC pc)
{
   PC_PFLAREINV_fref *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV_fref *)pc->data;    
   // The block work matrix's product references mat_inverse, so drop it first
   PetscCall(PCPFLAREINVResetBlockWork_Private_fref(inv_data));
   reset_inverse_mat_c_fref(&(inv_data->mat_inverse));
   // Free stored polynomial coefficients (Fortran-allocated or C-malloc'd, both freed with free())
   free(inv_data->poly_coeffs);
   inv_data->poly_coeffs      = NULL;
   inv_data->poly_coeffs_rows = 0;
   inv_data->poly_coeffs_cols = 0;
   // Note: reuse_poly_coeffs is intentionally NOT reset here (persists across resets)
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~
// Now all the get/set routines for options
// For explanation see the comments above the set routines
// ~~~~~~~~~~~~~~~~~~~~~

// Get routines


static PetscErrorCode PCPFLAREINVGetPolyOrder_PFLAREINV_fref(PC pc, PetscInt *poly_order)
{
   PC_PFLAREINV_fref *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV_fref *)pc->data;    
   *poly_order = inv_data->poly_order;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~


static PetscErrorCode PCPFLAREINVGetSparsityOrder_PFLAREINV_fref(PC pc, PetscInt *inverse_sparsity_order)
{
   PC_PFLAREINV_fref *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV_fref *)pc->data;    
   *inverse_sparsity_order = inv_data->inverse_sparsity_order;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~


static PetscErrorCode PCPFLAREINVGetType_PFLAREINV_fref(PC pc, PCPFLAREINVType *inverse_type)
{
   PC_PFLAREINV_fref *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV_fref *)pc->data;
   *inverse_type = (PCPFLAREINVType)inv_data->inverse_type;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~


static PetscErrorCode PCPFLAREINVGetMatrixFree_PFLAREINV_fref(PC pc, PetscBool *flg)
{
   PC_PFLAREINV_fref *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV_fref *)pc->data;   
   *flg = inv_data->matrix_free;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~


static PetscErrorCode PCPFLAREINVGetInverseMat_PFLAREINV_fref(PC pc, Mat *mat)
{
   PC_PFLAREINV_fref *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV_fref *)pc->data;
   *mat = inv_data->mat_inverse;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// Set routines


 static PetscErrorCode PCPFLAREINVSetPolyOrder_PFLAREINV_fref(PC pc, PetscInt poly_order)
{
   PC_PFLAREINV_fref *inv_data;
   PetscInt old_order;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV_fref *)pc->data;
   PetscCall(PCPFLAREINVGetPolyOrder(pc, &old_order));
   if (old_order == poly_order) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_PFLAREINV_fref(pc));
   inv_data->poly_order = (int)poly_order;
   pc->setupcalled = PETSC_FALSE;  
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~


 static PetscErrorCode PCPFLAREINVSetSparsityOrder_PFLAREINV_fref(PC pc, PetscInt inverse_sparsity_order)
{
   PC_PFLAREINV_fref *inv_data;
   PetscInt old_order;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV_fref *)pc->data;
   PetscCall(PCPFLAREINVGetSparsityOrder(pc, &old_order));
   if (old_order == inverse_sparsity_order) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_PFLAREINV_fref(pc));
   inv_data->inverse_sparsity_order = (int)inverse_sparsity_order;
   pc->setupcalled = PETSC_FALSE;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~


 static PetscErrorCode PCPFLAREINVSetType_PFLAREINV_fref(PC pc, PCPFLAREINVType inverse_type)
{
   PC_PFLAREINV_fref *inv_data;
   PCPFLAREINVType old_type;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV_fref *)pc->data;
   PetscCall(PCPFLAREINVGetType(pc, &old_type));
   if (old_type == inverse_type) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_PFLAREINV_fref(pc));
   inv_data->inverse_type = inverse_type;
   pc->setupcalled = PETSC_FALSE;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~


static PetscErrorCode PCPFLAREINVSetMatrixFree_PFLAREINV_fref(PC pc, PetscBool flg)
{
   PC_PFLAREINV_fref *inv_data;
   PetscBool old_flag;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV_fref *)pc->data;
   PetscCall(PCPFLAREINVGetMatrixFree(pc, &old_flag));
   if (old_flag == flg) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_PFLAREINV_fref(pc));   
   inv_data->matrix_free = flg;
   pc->setupcalled = PETSC_FALSE;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~


static PetscErrorCode PCPFLAREINVGetPolyCoeffs_PFLAREINV_fref(PC pc, PetscReal **coeffs, PetscInt *rows, PetscInt *cols)
{
   PC_PFLAREINV_fref *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV_fref *)pc->data;
   *coeffs = inv_data->poly_coeffs;
   *rows   = inv_data->poly_coeffs_rows;
   *cols   = inv_data->poly_coeffs_cols;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~


static PetscErrorCode PCPFLAREINVSetPolyCoeffs_PFLAREINV_fref(PC pc, PetscReal *coeffs, PetscInt rows, PetscInt cols)
{
   PC_PFLAREINV_fref *inv_data;
   PetscReal    *new_coeffs;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV_fref *)pc->data;
   // Copy into a new buffer before freeing the old one, as coeffs may be the
   // pointer returned by PCPFLAREINVGetPolyCoeffs (ie it may alias poly_coeffs)
   new_coeffs = (PetscReal *)malloc((size_t)rows * (size_t)cols * sizeof(PetscReal));
   PetscCheck(new_coeffs, PETSC_COMM_SELF, PETSC_ERR_MEM, "malloc failed in PCPFLAREINVSetPolyCoeffs");
   memcpy(new_coeffs, coeffs, (size_t)rows * (size_t)cols * sizeof(PetscReal));
   free(inv_data->poly_coeffs);
   inv_data->poly_coeffs      = new_coeffs;
   inv_data->poly_coeffs_rows = rows;
   inv_data->poly_coeffs_cols = cols;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~


static PetscErrorCode PCPFLAREINVGetReusePolyCoeffs_PFLAREINV_fref(PC pc, PetscBool *flg)
{
   PC_PFLAREINV_fref *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV_fref *)pc->data;
   *flg = inv_data->reuse_poly_coeffs;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~


static PetscErrorCode PCPFLAREINVSetReusePolyCoeffs_PFLAREINV_fref(PC pc, PetscBool flg)
{
   PC_PFLAREINV_fref *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV_fref *)pc->data;
   inv_data->reuse_poly_coeffs = flg;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static PetscErrorCode PCApply_PFLAREINV_fref(PC pc, Vec x, Vec y)
{
   PC_PFLAREINV_fref *inv_data;
   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV_fref *)pc->data;
   // Just call a matmult
   PetscCall(MatMult(inv_data->mat_inverse, x, y));
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

// Transposed apply, y = mat_inverse^T x - this is the exact transpose of PCApply,
// not a separately computed approximate inverse of A^T (which would be a different
// polynomial and hence not an adjoint of what PCApply does).
// The assembled inverses transpose natively (aij, and matdiagonal for the jacobi
// types and for poly_order 0). The matrix-free polynomials register a
// MATOP_MULT_TRANSPOSE that applies the same polynomial to a virtual transpose of
// the operator, which works because our coefficients are real, so q(A)^T = q(A^T).
static PetscErrorCode PCApplyTranspose_PFLAREINV_fref(PC pc, Vec x, Vec y)
{
   PC_PFLAREINV_fref *inv_data;
   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV_fref *)pc->data;

   // A matrix-free inverse applies the operator itself, so the operator has to
   // know how to do a transposed matvec. That is always true of the assembled
   // types PFLARE builds, but the user is free to hand us their own MatShell as
   // the operator (which is why we never turn on the diagonal scaling below), and
   // theirs may only have MATOP_MULT. Catch that here rather than letting it fail
   // several layers down with an error that mentions neither PFLARE nor which of
   // the shells in play is the problem.
   if (inv_data->matrix_free) {
      PetscBool has_transpose;
      PetscCall(MatHasOperation(pc->pmat, MATOP_MULT_TRANSPOSE, &has_transpose));
      PetscCheck(has_transpose, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, \
            "PCPFLAREINV cannot apply its transpose matrix-free as the operator has no MatMultTranspose - " \
            "register MATOP_MULT_TRANSPOSE on your MatShell, or don't use -pc_pflareinv_matrix_free");
   }

   // Just call a transposed matmult
   PetscCall(MatMultTranspose(inv_data->mat_inverse, x, y));
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

// Multi-RHS apply: Y = mat_inverse * X for dense X, Y.
// Lets KSPMatSolve / PCMatApply hit a real SpMM (cuSPARSE/hipSPARSE/Kokkos/CPU
// AIJxDense) instead of PETSc's default column-by-column PCApply fallback.
static PetscErrorCode PCMatApply_PFLAREINV_fref(PC pc, Mat X, Mat Y)
{
   PC_PFLAREINV_fref *inv_data;
   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV_fref *)pc->data;

   if (inv_data->matrix_free) {
      // mat_inverse is a MatShell with only MATOP_MULT and MATOP_MULT_TRANSPOSE
      // registered, so MatMatMult on it
      // would fail. The polynomial matshells can however be applied blockwise by
      // doing the products with the underlying matrix - the Fortran routine below does
      // that and reports whether it managed it.
      // The gate here is just the list of families that have a block kernel - it isn't
      // load bearing for correctness, shell_poly_block_apply works out which arithmetic
      // to use from the matshell's context and reports back if it can't do the apply
      // (the Jacobi types are never matrix-free anyway).
      int applied = 0;
      if (inv_data->inverse_type == PFLAREINV_POWER || inv_data->inverse_type == PFLAREINV_ARNOLDI || \
          inv_data->inverse_type == PFLAREINV_NEWTON || inv_data->inverse_type == PFLAREINV_NEWTON_NO_EXTRA || \
          inv_data->inverse_type == PFLAREINV_NEUMANN)
      {
         pflareinv_shell_block_matapply_c_fref(&(inv_data->mat_inverse), &X, &Y, &applied);
      }

      if (applied) {
         PetscCall(PetscInfo(pc, "matrix-free PCMatApply used the block polynomial kernel\n"));
      } else {
         // Anything we can't do blockwise is applied column-by-column.
         PetscInt n_cols, j;
         Vec cx, cy;
         PetscCall(PetscInfo(pc, "matrix-free PCMatApply solving column by column\n"));
         PetscCall(MatGetSize(X, NULL, &n_cols));
         for (j = 0; j < n_cols; j++) {
            PetscCall(MatDenseGetColumnVecRead(X, j, &cx));
            PetscCall(MatDenseGetColumnVecWrite(Y, j, &cy));
            PetscCall(MatMult(inv_data->mat_inverse, cx, cy));
            PetscCall(MatDenseRestoreColumnVecWrite(Y, j, &cy));
            PetscCall(MatDenseRestoreColumnVecRead(X, j, &cx));
         }
      }
   } else {
      // Assembled inverse: real SpMM via MatProduct.
      // The product is attached to a dense work matrix the pc owns and stays attached
      // between applies, so only its numeric phase runs on repeat calls - the same
      // as the matrix-free block kernels and KSPRICHARDSON's block solve do. The
      // symbolic phase of the mpi product builds the dense scatter, so we only want
      // it to run when something about the operands changes.
      // The product is bound to the caller's X so the input isn't copied, but the
      // bookkeeping can't live on the caller's Y (petsc errors if a mat that already
      // has a product is used as the output of another one), so the result goes
      // through the work matrix and is copied into Y.
      PetscBool        reuse = PETSC_FALSE;
      PetscInt         lda;
      PetscObjectState nz_state;

      PetscCall(MatDenseGetLDA(X, &lda));
      PetscCall(MatGetNonzeroState(inv_data->mat_inverse, &nz_state));
      if (inv_data->block_work) {
         // We can reuse the product if the inverse is the one it was set up with and
         // its sparsity hasn't changed (a setup with the same nonzero pattern only
         // changes its values), Y has the same type and layout as the work matrix
         // (which fixes the number of right-hand sides) and X has the lda the mpi
         // scatter was built with
         // Any change in X's type is handled by MatProductReplaceMats below
         PetscBool same_type;
         PetscInt  work_m, work_n, work_M, work_N, y_m, y_n, y_M, y_N;
         PetscCall(PetscObjectTypeCompare((PetscObject)inv_data->block_work, ((PetscObject)Y)->type_name, &same_type));
         PetscCall(MatGetLocalSize(inv_data->block_work, &work_m, &work_n));
         PetscCall(MatGetLocalSize(Y, &y_m, &y_n));
         PetscCall(MatGetSize(inv_data->block_work, &work_M, &work_N));
         PetscCall(MatGetSize(Y, &y_M, &y_N));
         reuse = (PetscBool)(same_type && inv_data->block_work_inverse == inv_data->mat_inverse && \
                             inv_data->block_work_inverse_nz_state == nz_state && \
                             inv_data->block_work_lda == lda && \
                             work_m == y_m && work_n == y_n && work_M == y_M && work_N == y_N);
         if (!reuse) PetscCall(PCPFLAREINVResetBlockWork_Private_fref(inv_data));
      }

      if (reuse) {
         PetscCall(PetscInfo(pc, "assembled PCMatApply reusing the cached symbolic phase of the product\n"));
         // Bind the product to this call's X - the symbolic phase only reruns if X's type changed
         PetscCall(MatProductReplaceMats(NULL, X, NULL, inv_data->block_work));
      } else {
         PetscCall(PetscInfo(pc, "assembled PCMatApply setting up the product\n"));
         PetscCall(MatDuplicate(Y, MAT_DO_NOT_COPY_VALUES, &inv_data->block_work));
         PetscCall(MatProductCreateWithMat(inv_data->mat_inverse, X, NULL, inv_data->block_work));
         PetscCall(MatProductSetType(inv_data->block_work, MATPRODUCT_AB));
         PetscCall(MatProductSetFromOptions(inv_data->block_work));
         PetscCall(MatProductSymbolic(inv_data->block_work));
         inv_data->block_work_inverse          = inv_data->mat_inverse;
         inv_data->block_work_inverse_nz_state = nz_state;
         inv_data->block_work_lda              = lda;
      }
      PetscCall(MatProductNumeric(inv_data->block_work));
      PetscCall(MatCopy(inv_data->block_work, Y, SAME_NONZERO_PATTERN));
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

static PetscErrorCode PCDestroy_PFLAREINV_fref(PC pc)
{
   PC_PFLAREINV_fref *inv_data;

   PetscFunctionBegin;

   inv_data = (PC_PFLAREINV_fref *)pc->data;

   // Reset the mat
   PetscCall(PCReset_PFLAREINV_fref(pc));

   // Then destroy the heap pointer
   PetscCall(PetscFree(inv_data));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetType_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetType_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetMatrixFree_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetMatrixFree_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetPolyOrder_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetPolyOrder_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetSparsityOrder_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetSparsityOrder_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetPolyCoeffs_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetPolyCoeffs_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetReusePolyCoeffs_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetReusePolyCoeffs_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetInverseMat_C", NULL));
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

static PetscErrorCode PCSetFromOptions_PFLAREINV_fref(PC pc, PetscOptionItems PetscOptionsObject)
{
   PetscBool    flg, matrix_free;
   PCPFLAREINVType deflt, type;
   PetscInt poly_order, inverse_sparsity_order;
   PC_PFLAREINV_fref *inv_data;

   PetscFunctionBegin;

   inv_data = (PC_PFLAREINV_fref *)pc->data;

   PetscCall(PCPFLAREINVGetType(pc, &deflt));
   PetscOptionsHeadBegin(PetscOptionsObject, "PCPFLAREINV options");
   const char *const PCPFLAREINVTypes[] = {"POWER", "ARNOLDI", "NEWTON", "NEWTON_NO_EXTRA", "NEUMANN", "SAI", "ISAI", "WJACOBI", "JACOBI", "L1JACOBI", "PCPFLAREINVType", "PFLAREINV_", NULL};
   PetscCall(PetscOptionsEnum("-pc_pflareinv_type", "Inverse type", "PCPFLAREINVSetType", PCPFLAREINVTypes, (PetscEnum)deflt, (PetscEnum *)&type, &flg));
   if (flg) PetscCall(PCPFLAREINVSetType(pc, type));
   // Go through the setter so a changed value resets the PC
   PetscCall(PetscOptionsBool("-pc_pflareinv_matrix_free", "Apply matrix free", "PCPFLAREINVSetMatrixFree", inv_data->matrix_free, &matrix_free, &flg));
   if (flg) PetscCall(PCPFLAREINVSetMatrixFree(pc, matrix_free));
   PetscCall(PetscOptionsBool("-pc_pflareinv_reuse_poly_coeffs", "Reuses gmres polynomial coefficients during setup", "PCPFLAREINVSetReusePolyCoeffs", inv_data->reuse_poly_coeffs, &inv_data->reuse_poly_coeffs, NULL));
   PetscCall(PetscOptionsInt("-pc_pflareinv_poly_order", "Order of polynomial", "PCPFLAREINVSetPolyOrder", inv_data->poly_order, &poly_order, &flg));
   if (flg) PetscCall(PCPFLAREINVSetPolyOrder(pc, poly_order));
   PetscCall(PetscOptionsInt("-pc_pflareinv_sparsity_order", "Sparsity order of assembled inverse", "PCPFLAREINVSetSparsityOrder", inv_data->inverse_sparsity_order, &inverse_sparsity_order, &flg));
   if (flg) PetscCall(PCPFLAREINVSetSparsityOrder(pc, inverse_sparsity_order));
   PetscOptionsHeadEnd();
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

static PetscErrorCode PCSetUp_PFLAREINV_fref(PC pc)
{
   PCPFLAREINVType type;
   MPI_Comm   comm; 
   PC_PFLAREINV_fref *inv_data;
   PetscReal *new_coeffs;
   PetscInt   new_rows, new_cols;

   PetscFunctionBegin;

   inv_data = (PC_PFLAREINV_fref *)pc->data;
   comm = PetscObjectComm((PetscObject)pc);

   // ~~~~~~~
   // Check options
   // ~~~~~~~
   PetscCall(PCPFLAREINVGetType(pc, &type));

   // SAI/ISAI can't be matrix free
   if (type == PFLAREINV_SAI || type == PFLAREINV_ISAI)
   {
      PetscCheck(!inv_data->matrix_free, comm, PETSC_ERR_ARG_WRONGSTATE, "PFLARE SAI/ISAI inverses cannot be applied matrix-free - Use PCASM");
   }     

   // We have to pass in an int
   int matrix_free_int = inv_data->matrix_free == PETSC_TRUE;   
   // We don't allow polynomail diagonal scaling through PCPFLAREINV
   // as the user could pass in a matshell as the matrix
   // It is the user's responsibility to have done the scaling beforehand
   // if they want
   int diag_scale_polys_int = 0;
   // Similarly for the subcomm
   int subcomm_int = 0;

   // If we haven't setup yet
   if (pc->setupcalled == 0)
   {
      // Fresh setup: pass NULL to let Fortran allocate and return the coefficient pointer
      new_coeffs = NULL; new_rows = 0; new_cols = 0;
      calculate_and_build_approximate_inverse_c_fref(&(pc->pmat), \
            type, \
            inv_data->poly_order, inv_data->inverse_sparsity_order, \
            matrix_free_int, diag_scale_polys_int, subcomm_int, \
            &new_coeffs, &new_rows, &new_cols, \
            &(inv_data->mat_inverse));
      free(inv_data->poly_coeffs);
      inv_data->poly_coeffs      = new_coeffs;
      inv_data->poly_coeffs_rows = new_rows;
      inv_data->poly_coeffs_cols = new_cols;
   }
   else
   {
      // If we've got a different non-zero pattern we've got to 
      // start again       
      if (pc->flag == DIFFERENT_NONZERO_PATTERN)
      {
         // PCReset also frees poly_coeffs
         PetscCall(PCReset_PFLAREINV_fref(pc));
         // Fresh: pass NULL to let Fortran allocate and return the coefficient pointer
         new_coeffs = NULL; new_rows = 0; new_cols = 0;
         calculate_and_build_approximate_inverse_c_fref(&(pc->pmat), \
               type, \
               inv_data->poly_order, inv_data->inverse_sparsity_order, \
               matrix_free_int, diag_scale_polys_int, subcomm_int, \
               &new_coeffs, &new_rows, &new_cols, \
               &(inv_data->mat_inverse));
         inv_data->poly_coeffs      = new_coeffs;
         inv_data->poly_coeffs_rows = new_rows;
         inv_data->poly_coeffs_cols = new_cols;
      }
      else if (pc->flag == SAME_NONZERO_PATTERN)
      {
         // We don't call reset on the pc here so it reuses the sparsity of mat_inverse
         // Optionally reuse stored polynomial coefficients:
         //   poly_coeffs != NULL and reuse flag set  ->  pass the existing pointer (reuse path)
         //   otherwise                               ->  pass NULL so Fortran computes fresh ones
         // Non-polynomial types (SAI, ISAI, WJacobi, Jacobi, L1Jacobi) have no coefficients, so any
         // stored ones are dropped and Fortran returns NULL
         int is_polynomial;
         inverse_type_is_polynomial_c_fref((int)type, &is_polynomial);
         if (!(is_polynomial && inv_data->reuse_poly_coeffs == PETSC_TRUE && inv_data->poly_coeffs != NULL)) {
            // Fresh: free old coefficients so poly_coeffs is NULL going into the call
            free(inv_data->poly_coeffs);
            inv_data->poly_coeffs      = NULL;
            inv_data->poly_coeffs_rows = 0;
            inv_data->poly_coeffs_cols = 0;
         }
         // Pass poly_coeffs by address: NULL -> fresh allocation on return; non-NULL -> reuse
         calculate_and_build_approximate_inverse_c_fref(&(pc->pmat), \
               type, \
               inv_data->poly_order, inv_data->inverse_sparsity_order, \
               matrix_free_int, diag_scale_polys_int, subcomm_int, \
               &inv_data->poly_coeffs, &inv_data->poly_coeffs_rows, &inv_data->poly_coeffs_cols, \
               &(inv_data->mat_inverse));
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

static PetscErrorCode PCView_PFLAREINV_fref(PC pc, PetscViewer viewer)
{

   PC_PFLAREINV_fref *inv_data;
   PetscBool  iascii;
   PCPFLAREINVType type;

   PetscFunctionBegin;

   inv_data = (PC_PFLAREINV_fref *)pc->data;

   // Print out details about our PFLAREINV
   PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));

   if (iascii) {
      PetscCall(PCPFLAREINVGetType(pc, &type));

      // What type of inverse
      if (type == PFLAREINV_POWER)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  GMRES polynomial, power basis, order %i \n", inv_data->poly_order));
      }
      else if (type == PFLAREINV_ARNOLDI)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  GMRES polynomial, arnoldi basis, order %i \n", inv_data->poly_order));
      }
      else if (type == PFLAREINV_NEWTON)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  GMRES polynomial, newton basis with extra roots, order %i \n", inv_data->poly_order));
      }
      else if (type == PFLAREINV_NEWTON_NO_EXTRA)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  GMRES polynomial, newton basis without extra roots, order %i \n", inv_data->poly_order));
      }
      else if (type == PFLAREINV_SAI)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  SAI \n"));
      }
      else if (type == PFLAREINV_ISAI)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  ISAI \n"));
      }
      else if (type == PFLAREINV_NEUMANN)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  Neumann polynomial, order %i \n", inv_data->poly_order));
      }
      else if (type == PFLAREINV_WJACOBI)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  Weighted Jacobi \n"));
      }
      else if (type == PFLAREINV_JACOBI)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  Unweighted Jacobi \n"));
      }
      else if (type == PFLAREINV_L1JACOBI)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  l1 Jacobi \n"));
      }
      if (type != PFLAREINV_WJACOBI && type != PFLAREINV_JACOBI && type != PFLAREINV_L1JACOBI)
      {
         // If matrix-free or not
         if (inv_data->matrix_free)
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "  matrix-free inverse \n"));
         }
         else
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "  assembled inverse, sparsity order %i\n", inv_data->inverse_sparsity_order));
         }
         if (inv_data->reuse_poly_coeffs) PetscCall(PetscViewerASCIIPrintf(viewer, "  Reusing gmres polynomial coefficients during setup \n"));
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~


// Creates the structure we need for this PC
static PetscErrorCode PCCreate_PFLAREINV_fref(PC pc)
{
   // Create our data structure on the heap
   PC_PFLAREINV_fref *inv_data;

   PetscFunctionBegin;

   // Now we call petsc fortran routines from this PC
   // so we have to have made sure this is called
   PetscCall(PetscInitializeFortran());

   PetscCall(PetscNew(&inv_data));
   pc->data = (void *)inv_data;   

   // ~~~~~~~~~~~~
   // Default options for the PCPFLAREINV
   // ~~~~~~~~~~~~

   // Have to be very careful in the Fortran routines 
   // with any matrix that has been set to null in C
   // in petsc > 3.22
   // See calculate_and_build_approximate_inverse in 
   // Approx_Inverse_Setup.F90
   inv_data->mat_inverse = NULL;   
   // What type of inverse to apply
   // Default to gmres polynomial with the Arnoldi basis
   inv_data->inverse_type = PFLAREINV_ARNOLDI;
   // The polynomial order
   inv_data->poly_order = 6;
   // The power of the sparsity we use (if assembled)
   inv_data->inverse_sparsity_order = 1;
   // Whether or not the mat_inverse is just a matshell and applied
   // matrix free
   inv_data->matrix_free = PETSC_FALSE;
   // Polynomial coefficient storage (initially empty)
   inv_data->poly_coeffs       = NULL;
   inv_data->poly_coeffs_rows  = 0;
   inv_data->poly_coeffs_cols  = 0;
   inv_data->reuse_poly_coeffs = PETSC_FALSE;
   inv_data->block_work                  = NULL;
   inv_data->block_work_inverse          = NULL;
   inv_data->block_work_inverse_nz_state = 0;
   inv_data->block_work_lda              = -1;

   // Set the method functions
   pc->ops->apply               = PCApply_PFLAREINV_fref;
   pc->ops->matapply            = PCMatApply_PFLAREINV_fref;
   // We deliberately don't set matapplytranspose - PCMatApplyTranspose falls back to
   // applying PCApplyTranspose column by column when it is NULL, which is correct
   pc->ops->applytranspose      = PCApplyTranspose_PFLAREINV_fref;
   pc->ops->setup               = PCSetUp_PFLAREINV_fref;
   pc->ops->destroy             = PCDestroy_PFLAREINV_fref;
   pc->ops->view                = PCView_PFLAREINV_fref;  
   pc->ops->reset               = PCReset_PFLAREINV_fref;
   pc->ops->setfromoptions      = PCSetFromOptions_PFLAREINV_fref;

   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetType_C", PCPFLAREINVSetType_PFLAREINV_fref));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetType_C", PCPFLAREINVGetType_PFLAREINV_fref));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetMatrixFree_C", PCPFLAREINVSetMatrixFree_PFLAREINV_fref));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetMatrixFree_C", PCPFLAREINVGetMatrixFree_PFLAREINV_fref));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetPolyOrder_C", PCPFLAREINVSetPolyOrder_PFLAREINV_fref));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetPolyOrder_C", PCPFLAREINVGetPolyOrder_PFLAREINV_fref));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetSparsityOrder_C", PCPFLAREINVSetSparsityOrder_PFLAREINV_fref));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetSparsityOrder_C", PCPFLAREINVGetSparsityOrder_PFLAREINV_fref));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetPolyCoeffs_C", PCPFLAREINVGetPolyCoeffs_PFLAREINV_fref));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetPolyCoeffs_C", PCPFLAREINVSetPolyCoeffs_PFLAREINV_fref));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetReusePolyCoeffs_C", PCPFLAREINVGetReusePolyCoeffs_PFLAREINV_fref));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetReusePolyCoeffs_C", PCPFLAREINVSetReusePolyCoeffs_PFLAREINV_fref));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetInverseMat_C", PCPFLAREINVGetInverseMat_PFLAREINV_fref));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

// Registers the reference PC type (called by PCRegister_PFLAREINV in PFLARE_PORT_DEBUG builds)
PETSC_INTERN void PCRegister_PFLAREINV_fref(void)
{
   PetscCallVoid(PCRegister("pflareinv_fref", PCCreate_PFLAREINV_fref));
}