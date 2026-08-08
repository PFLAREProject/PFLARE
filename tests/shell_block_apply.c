static char help[] = "Tests the multiple rhs (block) apply of the matrix-free GMRES polynomial matshells.\n\n";

/*
  This drives the block apply of the matrix-free polynomial matshells directly,
  without a PC in the loop, and checks it against a column-by-column MatMult of
  the same matshell.

  The reason this exists rather than just going through PCPFLAREINV is the
  diagonally scaled variant of the polynomials, ie q(D^-1 A) D^-1. PCPFLAREINV
  deliberately never turns that on (the user may pass in a matshell as the
  operator, so scaling is left to them), but PCAIR builds its polynomial
  smoothers that way, so the block kernels have to support it. Here we call
  calculate_and_build_approximate_inverse_c directly, which does expose the
  diagonal scaling flag, and test both modes.

    ./shell_block_apply -n 1000 -nrhs 5
    mpiexec -n 2 ./shell_block_apply -n 1000 -nrhs 5

  The operator is the same 1D upwind advection as adv_1d_multi_rhs.c.
*/
#include <petscksp.h>
#include "pflare.h"

// Tolerance for the comparison against the column-by-column reference apply
#if defined(PETSC_USE_REAL_SINGLE)
  #define DEFAULT_CHECK_TOL 1e-5
#else
  #define DEFAULT_CHECK_TOL 1e-10
#endif

// Defined in C_Fortran_Bindings.F90 - these are the same prototypes PCPFLAREINV.c uses
PETSC_EXTERN void reset_inverse_mat_c(Mat *mat);
// coeffs_ptr/row_size/col_size are in/out:
//   *coeffs_ptr == NULL on entry  -> fresh: Fortran allocates, writes c_loc to *coeffs_ptr on return
//   *coeffs_ptr != NULL on entry  -> reuse: existing coefficients used, polynomial step skipped
PETSC_EXTERN void calculate_and_build_approximate_inverse_c(Mat *input_mat, PetscInt inverse_type, PetscInt order, \
                     PetscInt sparsity_order, PetscInt matrix_free_int, PetscInt diag_scale_polys_int, \
                     PetscInt subcomm_int, \
                     PetscReal **coeffs_ptr, PetscInt *row_size, PetscInt *col_size, \
                     Mat *inv_matrix);
PETSC_EXTERN void pflareinv_shell_block_matapply_c(Mat *mat, Mat *X, Mat *Y, int *applied);

/*
   Builds the 1D upwind advection operator with a Dirichlet condition on the left
   boundary, assembled through the COO interface so this works on the device too.
*/
static PetscErrorCode BuildAdvectionOperator(PetscInt n, Mat *A_out)
{
  Vec          x;
  Mat          A;
  PetscInt     i, global_row_start, global_row_end_plus_one, local_size, start_assign;
  PetscCount   counter;
  PetscInt    *oor, *ooc;
  PetscScalar *v;

  PetscFunctionBeginUser;

  /* A vector purely to get the parallel layout */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecGetLocalSize(x, &local_size));
  PetscCall(VecGetOwnershipRange(x, &global_row_start, &global_row_end_plus_one));
  PetscCall(VecDestroy(&x));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, local_size, local_size, n, n));
  PetscCall(MatSetFromOptions(A));

  PetscCall(PetscMalloc2(2 * local_size, &oor, 2 * local_size, &ooc));
  PetscCall(PetscMalloc1(2 * local_size, &v));

  counter = 0;
  // Dirichlet condition on left boundary
  if (global_row_start == 0) {
    start_assign = 1;
    oor[counter] = 0;
    ooc[counter] = 0;
    v[counter]   = 1;
    counter      = counter + 1;
  } else {
    start_assign = global_row_start;
  }

  for (i = start_assign; i < global_row_end_plus_one; i++) {
    // Upwinded dimensionless uniform grid finite difference operator
    oor[counter] = i;
    ooc[counter] = i - 1;
    v[counter]   = -1.0;

    oor[counter + 1] = i;
    ooc[counter + 1] = i;
    v[counter + 1]   = 1.0;

    counter = counter + 2;
  }

  PetscCall(MatSetPreallocationCOO(A, counter, oor, ooc));
  PetscCall(PetscFree2(oor, ooc));
  PetscCall(MatSetValuesCOO(A, v, INSERT_VALUES));
  PetscCall(PetscFree(v));

  *A_out = A;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Relative Frobenius comparison of a block apply against the reference block
*/
static PetscErrorCode CheckBlock(Mat X, Mat Xref, PetscReal check_tol, const char *label)
{
  Mat       diff;
  PetscReal diff_norm, x_norm;

  PetscFunctionBeginUser;
  PetscCall(MatDuplicate(Xref, MAT_COPY_VALUES, &diff));
  PetscCall(MatNorm(X, NORM_FROBENIUS, &x_norm));
  PetscCall(MatAXPY(diff, -1.0, X, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(diff, NORM_FROBENIUS, &diff_norm));
  PetscCheck(diff_norm <= check_tol * x_norm, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "%s: block apply differs from the column-by-column apply: ||X - Xref||_F = %g, ||X||_F = %g, tolerance %g",
             label, (double)diff_norm, (double)x_norm, (double)check_tol);
  PetscCall(MatDestroy(&diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Builds a matrix-free polynomial inverse of A - either the plain q(A) or the
   right diagonally scaled q(D^-1 A) D^-1 - and checks its block apply
*/
static PetscErrorCode RunMode(Mat A, PetscInt inverse_type, PetscInt poly_order, PetscInt nrhs,
                              PetscInt diag_scale, PetscReal check_tol, const char *label)
{
  Mat        inv_shell = NULL;
  Mat        B, X, Xref;
  VecType    vtype;
  PetscReal *coeffs = NULL;
  PetscInt   coeff_rows = 0, coeff_cols = 0;
  PetscInt   j, n, local_size;
  int        applied = 0;

  PetscFunctionBeginUser;

  PetscCall(MatGetSize(A, &n, NULL));
  PetscCall(MatGetLocalSize(A, &local_size, NULL));

  // Build the matshell. matrix_free = 1, sparsity order is unused matrix-free,
  // no subcomm. inv_shell must be NULL on entry so a fresh one is built.
  calculate_and_build_approximate_inverse_c(&A, inverse_type, poly_order, 1, 1, diag_scale, 0, \
                                            &coeffs, &coeff_rows, &coeff_cols, &inv_shell);

  // Dense blocks of rhs and solutions, taking the vector type from A so a device
  // matrix gives device dense blocks
  PetscCall(MatGetVecType(A, &vtype));
  PetscCall(MatCreateDenseFromVecType(PETSC_COMM_WORLD, vtype, local_size, PETSC_DECIDE, n, nrhs, PETSC_DECIDE, NULL, &B));
  PetscCall(MatCreateDenseFromVecType(PETSC_COMM_WORLD, vtype, local_size, PETSC_DECIDE, n, nrhs, PETSC_DECIDE, NULL, &X));
  PetscCall(MatDuplicate(X, MAT_DO_NOT_COPY_VALUES, &Xref));

  // Give each column of B a different constant
  for (j = 0; j < nrhs; j++) {
    Vec cb;
    PetscCall(MatDenseGetColumnVecWrite(B, j, &cb));
    PetscCall(VecSet(cb, 1.0 + (PetscScalar)j));
    PetscCall(MatDenseRestoreColumnVecWrite(B, j, &cb));
  }

  // The block apply we're testing
  PetscCall(MatZeroEntries(X));
  pflareinv_shell_block_matapply_c(&inv_shell, &B, &X, &applied);
  PetscCheck(applied == 1, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "%s: the block apply reported it couldn't be done", label);

  // The reference - the same matshell applied one column at a time
  for (j = 0; j < nrhs; j++) {
    Vec cb, cx;
    PetscCall(MatDenseGetColumnVecRead(B, j, &cb));
    PetscCall(MatDenseGetColumnVecWrite(Xref, j, &cx));
    PetscCall(MatMult(inv_shell, cb, cx));
    PetscCall(MatDenseRestoreColumnVecWrite(Xref, j, &cx));
    PetscCall(MatDenseRestoreColumnVecRead(B, j, &cb));
  }

  PetscCall(CheckBlock(X, Xref, check_tol, label));

  // Do it a second time - this exercises the cached dense temporaries in the
  // matshell context and the refresh of the reciprocal of the diagonal
  applied = 0;
  PetscCall(MatZeroEntries(X));
  pflareinv_shell_block_matapply_c(&inv_shell, &B, &X, &applied);
  PetscCheck(applied == 1, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "%s: the second block apply reported it couldn't be done", label);
  PetscCall(CheckBlock(X, Xref, check_tol, label));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s: block apply matches the column-by-column apply\n", label));

  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&Xref));
  // Destroys the matshell and everything hanging off its context
  reset_inverse_mat_c(&inv_shell);
  // The coefficients come back in a C-malloc'd buffer we own - PCPFLAREINV frees
  // its copy the same way
  free(coeffs);

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat       A;
  PetscInt  n = 100, nrhs = 5, poly_order = 6;
  PetscInt  inverse_type = PFLAREINV_ARNOLDI;
  PetscReal check_tol = DEFAULT_CHECK_TOL;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  // We call PFLARE fortran routines directly below, so this has to have been called
  PetscCall(PetscInitializeFortran());

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nrhs", &nrhs, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-poly_order", &poly_order, NULL));
  // The integer value of the PCPFLAREINVType we want, ie 0 power, 1 arnoldi,
  // 2 newton, 3 newton_no_extra
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-inverse_type", &inverse_type, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-check_tol", &check_tol, NULL));

  // Register the pflare types
  PCRegister_PFLARE();

  PetscCall(BuildAdvectionOperator(n, &A));

  // The plain polynomial, q(A) - this is what PCPFLAREINV builds
  PetscCall(RunMode(A, inverse_type, poly_order, nrhs, 0, check_tol, "Plain polynomial"));
  // The right diagonally scaled polynomial, q(D^-1 A) D^-1 - what PCAIR builds
  PetscCall(RunMode(A, inverse_type, poly_order, nrhs, 1, check_tol, "Diagonally scaled polynomial"));

  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}
