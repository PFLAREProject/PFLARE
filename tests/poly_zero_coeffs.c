static char help[] = "Tests monomial polynomial coefficients that are exactly zero in PCPFLAREINV.\n\n";

/*
  Sets monomial (power or Arnoldi basis) polynomial coefficients that contain exact
  zeros into PCPFLAREINV with PCPFLAREINVSetPolyCoeffs and reuse on, and checks both
  PCApply and PCMatApply against an explicit apply of the polynomial
     y = c0 x + c1 A x + c2 A^2 x + ...
  built out of repeated MatMults.

  Only the leading (highest order) zero coefficients can be skipped when applying
  the polynomial, a zero coefficient below a nonzero one still needs its power of A.
  The coefficient sets below have zeros in the leading, interior and both positions.

  This compares against the exact polynomial, so the assembled inverse must not
  have its sparsity constrained, ie the sparsity order defaults to the poly order.

    ./poly_zero_coeffs
    ./poly_zero_coeffs -pc_pflareinv_matrix_free
    mpiexec -n 2 ./poly_zero_coeffs -pc_pflareinv_type arnoldi
*/
#include <petscksp.h>
#include "pflare.h"

#if defined(PETSC_USE_REAL_SINGLE)
  #define DEFAULT_CHECK_TOL 1e-5
#else
  #define DEFAULT_CHECK_TOL 1e-10
#endif

#define POLY_ORDER 3
#define N_COEFF_SETS 7
#define N_RHS 3

/*
   Writes a 1D upwind advection-diffusion stencil into A, scaled so the powers
   of A stay O(1)
*/
static PetscErrorCode SetOperatorValues(Mat A, PetscInt n)
{
  PetscInt    i, global_row_start, global_row_end_plus_one, cols[3], n_cols;
  PetscScalar vals[3];

  PetscFunctionBeginUser;
  PetscCall(MatGetOwnershipRange(A, &global_row_start, &global_row_end_plus_one));

  for (i = global_row_start; i < global_row_end_plus_one; i++) {
    n_cols = 0;
    if (i > 0) {
      cols[n_cols] = i - 1;
      vals[n_cols] = -0.3;
      n_cols++;
    }
    cols[n_cols] = i;
    vals[n_cols] = 1.0;
    n_cols++;
    if (i < n - 1) {
      cols[n_cols] = i + 1;
      vals[n_cols] = -0.1;
      n_cols++;
    }
    PetscCall(MatSetValues(A, 1, &i, n_cols, cols, vals, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   y = c0 x + c1 A x + c2 A^2 x + ... with every power computed explicitly
*/
static PetscErrorCode ExplicitPolyApply(Mat A, const PetscReal *coeffs, PetscInt n_coeffs, Vec x, Vec y)
{
  Vec      power, temp;
  PetscInt i;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(x, &power));
  PetscCall(VecDuplicate(x, &temp));
  PetscCall(VecCopy(x, power));
  PetscCall(VecSet(y, 0.0));
  for (i = 0; i < n_coeffs; i++) {
    if (i > 0) {
      PetscCall(MatMult(A, power, temp));
      PetscCall(VecCopy(temp, power));
    }
    PetscCall(VecAXPY(y, coeffs[i], power));
  }
  PetscCall(VecDestroy(&power));
  PetscCall(VecDestroy(&temp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckDiff(Vec y, Vec ref, PetscReal tol, const char *label, PetscInt set)
{
  PetscReal diff_norm, ref_norm;
  Vec       diff;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(y, &diff));
  PetscCall(VecWAXPY(diff, -1.0, ref, y));
  PetscCall(VecNorm(diff, NORM_2, &diff_norm));
  PetscCall(VecNorm(ref, NORM_2, &ref_norm));
  if (ref_norm < 1.0) ref_norm = 1.0;
  PetscCheck(diff_norm / ref_norm <= tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, \
             "%s: coefficient set %" PetscInt_FMT " does not match the explicit polynomial, relative difference %g > %g", \
             label, set, (double)(diff_norm / ref_norm), (double)tol);
  PetscCall(VecDestroy(&diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat             A, X, Y;
  Vec             x, y, ref, col_x, col_y;
  PC              pc;
  PetscRandom     rand;
  PCPFLAREINVType type;
  PetscBool       matrix_free;
  PetscInt        n = 100, set, j;
  PetscReal       tol = DEFAULT_CHECK_TOL;
  // Monomial coefficients, lowest order first, with exact zeros in the leading
  // (highest order) and/or interior positions
  PetscReal coeff_sets[N_COEFF_SETS][POLY_ORDER + 1] = {
    {1.0, 0.0, 1.0, 0.5},   // interior zero at first order
    {1.0, 0.5, 0.0, 0.25},  // interior zero at second order
    {1.0, 0.0, 0.0, 0.5},   // consecutive interior zeros
    {1.0, 0.5, 0.25, 0.0},  // leading zero, which can be skipped
    {0.5, 0.0, 0.25, 0.0},  // leading and interior zeros
    {0.0, 0.0, 1.0, 0.0},   // just A^2
    {0.0, 0.0, 0.0, 0.0},   // all zero
  };

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PCRegister_PFLARE();
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-tol", &tol, NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, 3, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 3, NULL, 2, NULL));
  PetscCall(MatSetUp(A));
  PetscCall(SetOperatorValues(A, n));

  PetscCall(PCCreate(PETSC_COMM_WORLD, &pc));
  PetscCall(PCSetType(pc, PCPFLAREINV));
  PetscCall(PCPFLAREINVSetType(pc, PFLAREINV_POWER));
  PetscCall(PCPFLAREINVSetPolyOrder(pc, POLY_ORDER));
  // Don't constrain the sparsity, so the assembled inverse is the exact polynomial
  PetscCall(PCPFLAREINVSetSparsityOrder(pc, POLY_ORDER));
  PetscCall(PCSetOperators(pc, A, A));
  PetscCall(PCSetFromOptions(pc));
  PetscCall(PCPFLAREINVGetType(pc, &type));
  PetscCheck(type == PFLAREINV_POWER || type == PFLAREINV_ARNOLDI, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, \
             "This test needs monomial coefficients, ie the power or arnoldi types");
  PetscCall(PCPFLAREINVGetMatrixFree(pc, &matrix_free));
  // First setup computes the coefficients (and the sparsity pattern of the assembled inverse)
  PetscCall(PCSetUp(pc));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(VecDuplicate(x, &ref));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, N_RHS, NULL, &X));
  PetscCall(MatDuplicate(X, MAT_DO_NOT_COPY_VALUES, &Y));

  for (set = 0; set < N_COEFF_SETS; set++) {
    // Set our coefficients and have the next setup reuse them
    PetscCall(PCPFLAREINVSetPolyCoeffs(pc, coeff_sets[set], POLY_ORDER + 1, 1));
    PetscCall(PCPFLAREINVSetReusePolyCoeffs(pc, PETSC_TRUE));
    // Same values and nonzero pattern, but a new state so the setup is redone
    PetscCall(SetOperatorValues(A, n));
    PetscCall(PCSetOperators(pc, A, A));
    PetscCall(PCSetUp(pc));

    // Single vector apply
    PetscCall(VecSetRandom(x, rand));
    PetscCall(PCApply(pc, x, y));
    PetscCall(ExplicitPolyApply(A, coeff_sets[set], POLY_ORDER + 1, x, ref));
    PetscCall(CheckDiff(y, ref, tol, "PCApply", set));

    // Block apply
    PetscCall(MatSetRandom(X, rand));
    PetscCall(PCMatApply(pc, X, Y));
    for (j = 0; j < N_RHS; j++) {
      PetscCall(MatDenseGetColumnVecRead(X, j, &col_x));
      PetscCall(MatDenseGetColumnVecRead(Y, j, &col_y));
      PetscCall(ExplicitPolyApply(A, coeff_sets[set], POLY_ORDER + 1, col_x, ref));
      PetscCall(CheckDiff(col_y, ref, tol, "PCMatApply", set));
      PetscCall(MatDenseRestoreColumnVecRead(Y, j, &col_y));
      PetscCall(MatDenseRestoreColumnVecRead(X, j, &col_x));
    }
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  %s: %d coefficient sets with zeros match the explicit polynomial\n", \
                        matrix_free ? "matrix-free" : "assembled", N_COEFF_SETS));

  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&Y));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&ref));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PCDestroy(&pc));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}
