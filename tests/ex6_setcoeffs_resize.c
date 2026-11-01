static char help[] = "Tests that PCAIRSetPolyCoeffs can change the number of roots of a Newton polynomial.\n\
\n\
For the Newton inverse types the coefficients are the roots of the polynomial, and the\n\
number of roots can differ between setups. This sets up PCAIR on a 2D upwind\n\
advection-diffusion problem, then for each inverse tested:\n\
  - grows the stored n x 2 roots to (n + 2) x 2 by appending two exact-zero roots, which\n\
    are skipped when applying, so the polynomial is the same\n\
  - rebuilds the smoothers from the stored roots (-pc_air_reuse_poly_coeffs) by solving\n\
    again with the Pmat values unchanged; use -pc_air_reuse_sparsity or\n\
    -pc_air_diag_update to choose which rebuild path is taken\n\
  - shrinks them back to the original n x 2 and rebuilds again\n\
After each resize the stored size is checked, and PCApply with a fixed vector and the\n\
iteration count must match those of the original setup. With -pc_air_matrix_free_polys\n\
PCApply is also checked straight after each resize, before any setup, as the matrix-free\n\
polynomials must use the new roots immediately.\n\
Any size change that isn't allowed (one column, fewer rows than the polynomial order + 1,\n\
or any change for the non-Newton types) must error and leave the stored roots alone.\n\
\n\
  -test_acc    : also resize COEFFS_INV_ACC (needs C point smoothing, e.g. -pc_air_smooth_type fc)\n\
  -test_coarse : also resize COEFFS_INV_COARSE (needs -pc_air_coarsest_inverse_type newton or newton_no_extra)\n\
  -n           : grid size (default 30)\n\n";

#include <petscksp.h>
#include "pflare.h"

#define MAX_INVERSES 3

/* Builds a 2D upwind advection-diffusion operator on an n x n grid, with the velocity
   along (1, 1) and a little diffusion so A isn't triangular
   The diagonal varies, otherwise the top grid A_ff can be a constant diagonal, which the
   polynomial inverts exactly, so adding roots to it would never change the result */
static PetscErrorCode BuildMatrix(PetscInt n, Mat *A)
{
  PetscInt    Istart, Iend, row, i, j;
  PetscReal   alpha = 0.1;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(PETSC_COMM_WORLD, A));
  PetscCall(MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, n * n, n * n));
  PetscCall(MatSetFromOptions(*A));
  PetscCall(MatSeqAIJSetPreallocation(*A, 5, NULL));
  PetscCall(MatMPIAIJSetPreallocation(*A, 5, NULL, 5, NULL));
  PetscCall(MatGetOwnershipRange(*A, &Istart, &Iend));
  for (row = Istart; row < Iend; row++) {
    i = row % n;
    j = row / n;
    PetscCall(MatSetValue(*A, row, row, 2.0 + 4.0 * alpha + 0.5 * PetscSinReal((PetscReal)row), INSERT_VALUES));
    if (i > 0) PetscCall(MatSetValue(*A, row, row - 1, -1.0 - alpha, INSERT_VALUES));
    if (j > 0) PetscCall(MatSetValue(*A, row, row - n, -1.0 - alpha, INSERT_VALUES));
    if (i < n - 1) PetscCall(MatSetValue(*A, row, row + 1, -alpha, INSERT_VALUES));
    if (j < n - 1) PetscCall(MatSetValue(*A, row, row + n, -alpha, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Takes a copy of the stored roots, as PCAIRGetPolyCoeffs returns a pointer into PCAIR */
static PetscErrorCode CopyCoeffs(PC pc, PetscInt petsc_level, int which_inverse, PetscReal **coeffs, PetscInt *rows, PetscInt *cols)
{
  PetscReal *pc_coeffs;

  PetscFunctionBeginUser;
  PetscCall(PCAIRGetPolyCoeffs(pc, petsc_level, which_inverse, &pc_coeffs, rows, cols));
  PetscCall(PetscMalloc1((*rows) * (*cols), coeffs));
  PetscCall(PetscArraycpy(*coeffs, pc_coeffs, (*rows) * (*cols)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Checks the stored roots are rows x cols and equal to coeffs */
static PetscErrorCode CheckStored(PC pc, PetscInt petsc_level, int which_inverse, const PetscReal *coeffs, PetscInt rows, PetscInt cols)
{
  PetscReal *pc_coeffs;
  PetscInt   pc_rows, pc_cols;
  PetscBool  same;

  PetscFunctionBeginUser;
  PetscCall(PCAIRGetPolyCoeffs(pc, petsc_level, which_inverse, &pc_coeffs, &pc_rows, &pc_cols));
  PetscCheck(pc_rows == rows && pc_cols == cols, PETSC_COMM_SELF, PETSC_ERR_PLIB, "which_inverse %d stores %" PetscInt_FMT " x %" PetscInt_FMT " roots, expected %" PetscInt_FMT " x %" PetscInt_FMT, which_inverse, pc_rows, pc_cols, rows, cols);
  PetscCall(PetscArraycmp(pc_coeffs, coeffs, rows * cols, &same));
  PetscCheck(same, PETSC_COMM_SELF, PETSC_ERR_PLIB, "which_inverse %d does not store the roots that were set", which_inverse);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Checks PCApply(b) matches y_ref to rounding */
static PetscErrorCode CheckApply(PC pc, Vec b, Vec y_ref, Vec y, const char *stage)
{
  PetscReal norm_ref, norm_diff;

  PetscFunctionBeginUser;
  PetscCall(PCApply(pc, b, y));
  PetscCall(VecAXPY(y, -1.0, y_ref));
  PetscCall(VecNorm(y, NORM_2, &norm_diff));
  PetscCall(VecNorm(y_ref, NORM_2, &norm_ref));
  PetscCheck(norm_diff <= 1e3 * PETSC_MACHINE_EPSILON * norm_ref, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PCApply %s differs from the original setup, relative difference %g", stage, (double)(norm_diff / norm_ref));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Rebuilds the smoothers from the stored roots by solving again with the Pmat values
   unchanged, then checks PCApply and the iteration count match the original setup */
static PetscErrorCode RebuildAndCheck(KSP ksp, Mat A, Vec b_solve, Vec x, Vec b, Vec y_ref, Vec y, PetscInt its_ref, const char *stage)
{
  PC       pc;
  PetscInt its;

  PetscFunctionBeginUser;
  PetscCall(KSPGetPC(ksp, &pc));
  /* Tell PCAIR the Pmat has changed so the next solve sets it up again */
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
  PetscCall(VecSet(x, 0.0));
  PetscCall(KSPSolve(ksp, b_solve, x));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCheck(its == its_ref, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Solve %s took %" PetscInt_FMT " iterations, the original setup took %" PetscInt_FMT, stage, its, its_ref);
  PetscCall(CheckApply(pc, b, y_ref, y, stage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat              A;
  Vec              x, b_solve, b, y_ref, y;
  KSP              ksp;
  PC               pc;
  PetscInt         n = 30, num_levels, its_ref, k, i, n_inverses = 0;
  PetscBool        test_acc = PETSC_FALSE, test_coarse = PETSC_FALSE, matrix_free, newton, reuse_sparsity, diag_update;
  PCPFLAREINVType  inverse_type;
  PetscRandom      rctx;
  PetscErrorCode   ierr[3];
  int              which[MAX_INVERSES];
  PetscInt         level[MAX_INVERSES], rows[MAX_INVERSES], cols[MAX_INVERSES];
  PetscReal       *orig[MAX_INVERSES], *grown[MAX_INVERSES];

  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PCRegister_PFLARE();
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_acc", &test_acc, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_coarse", &test_coarse, NULL));

  PetscCall(BuildMatrix(n, &A));
  PetscCall(MatCreateVecs(A, &x, &b_solve));
  PetscCall(VecDuplicate(b_solve, &b));
  PetscCall(VecDuplicate(b_solve, &y_ref));
  PetscCall(VecDuplicate(b_solve, &y));
  PetscCall(VecSet(b_solve, 1.0));
  /* A fixed random vector to apply the PC to */
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(PetscRandomSetSeed(rctx, 42));
  PetscCall(PetscRandomSeed(rctx));
  PetscCall(VecSetRandom(b, rctx));
  PetscCall(PetscRandomDestroy(&rctx));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPGMRES));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCAIR));
  PetscCall(PCAIRSetInverseType(pc, PFLAREINV_NEWTON));
  PetscCall(KSPSetFromOptions(ksp));

  /* The original setup */
  PetscCall(KSPSolve(ksp, b_solve, x));
  PetscCall(KSPGetIterationNumber(ksp, &its_ref));
  PetscCall(PCApply(pc, b, y_ref));

  PetscCall(PCAIRGetNumLevels(pc, &num_levels));
  PetscCall(PCAIRGetInverseType(pc, &inverse_type));
  PetscCall(PCAIRGetMatrixFreePolys(pc, &matrix_free));
  PetscCall(PCAIRGetReuseSparsity(pc, &reuse_sparsity));
  PetscCall(PCAIRGetDiagUpdate(pc, &diag_update));
  newton = (PetscBool)(inverse_type == PFLAREINV_NEWTON || inverse_type == PFLAREINV_NEWTON_NO_EXTRA);
  PetscCheck(num_levels > 1, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Need more than one level");
  /* Without either of these the next setup starts from scratch and computes new roots */
  PetscCheck(!newton || reuse_sparsity || diag_update, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Use -pc_air_reuse_sparsity or -pc_air_diag_update to rebuild from the stored roots");

  which[n_inverses] = COEFFS_INV_AFF;    level[n_inverses++] = num_levels - 1;
  if (test_acc) { which[n_inverses] = COEFFS_INV_ACC; level[n_inverses++] = num_levels - 1; }
  if (test_coarse) { which[n_inverses] = COEFFS_INV_COARSE; level[n_inverses++] = 0; }

  for (k = 0; k < n_inverses; k++) {
    PetscCall(CopyCoeffs(pc, level[k], which[k], &orig[k], &rows[k], &cols[k]));
    /* The same roots followed by two exact-zero roots, which are skipped when applying */
    PetscCall(PetscCalloc1((rows[k] + 2) * cols[k], &grown[k]));
    for (i = 0; i < cols[k]; i++) PetscCall(PetscArraycpy(&grown[k][i * (rows[k] + 2)], &orig[k][i * rows[k]], rows[k]));
  }

  /* Size changes that aren't allowed must error and leave the stored roots alone
     The errors are returned (not aborted on) while PetscReturnErrorHandler is pushed */
  for (k = 0; k < n_inverses; k++) {
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
    if (newton) {
      /* One column */
      ierr[0] = PCAIRSetPolyCoeffs(pc, level[k], which[k], grown[k], rows[k] + 2, 1);
      /* Fewer rows than the polynomial order + 1; the stored roots never have fewer */
      ierr[1] = PCAIRSetPolyCoeffs(pc, level[k], which[k], grown[k], 1, cols[k]);
      ierr[2] = PCAIRSetPolyCoeffs(pc, level[k], which[k], grown[k], 0, cols[k]);
    } else {
      /* Any change for the non-Newton types */
      ierr[0] = PCAIRSetPolyCoeffs(pc, level[k], which[k], grown[k], rows[k] + 2, cols[k]);
      ierr[1] = PCAIRSetPolyCoeffs(pc, level[k], which[k], grown[k], rows[k] - 1, cols[k]);
      ierr[2] = PCAIRSetPolyCoeffs(pc, level[k], which[k], grown[k], rows[k], cols[k] + 1);
    }
    PetscCall(PetscPopErrorHandler());
    for (i = 0; i < 3; i++) PetscCheck(ierr[i] != PETSC_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Disallowed PCAIRSetPolyCoeffs %" PetscInt_FMT " on which_inverse %d did not error", i, which[k]);
    PetscCall(CheckStored(pc, level[k], which[k], orig[k], rows[k], cols[k]));
  }

  /* A matrix-free polynomial must apply new roots straight away, so growing to different
     roots must change PCApply before any setup. If the matshell were left pointing at the
     old (freed) roots it would still apply the original polynomial
     All the roots are scaled as well as two being added, as when the polynomial is an exact
     inverse (e.g. a diagonal A_ff with diagonal scaling) one factor of the Newton product is
     zero and adding roots alone would not change the result */
  if (newton && matrix_free) {
    for (k = 0; k < n_inverses; k++) {
      PetscReal *changed;
      PetscCall(PetscMalloc1((rows[k] + 2) * cols[k], &changed));
      for (i = 0; i < (rows[k] + 2) * cols[k]; i++) changed[i] = 1.5 * grown[k][i];
      /* Two extra real roots, repeating the first root's real part */
      changed[rows[k]] = changed[rows[k] + 1] = orig[k][0];
      PetscCall(PCAIRSetPolyCoeffs(pc, level[k], which[k], changed, rows[k] + 2, cols[k]));
      PetscCall(PetscFree(changed));
    }
    {
      PetscReal norm_ref, norm_diff;
      PetscCall(PCApply(pc, b, y));
      PetscCall(VecAXPY(y, -1.0, y_ref));
      PetscCall(VecNorm(y, NORM_2, &norm_diff));
      PetscCall(VecNorm(y_ref, NORM_2, &norm_ref));
      PetscCheck(norm_diff > 1e3 * PETSC_MACHINE_EPSILON * norm_ref, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PCApply did not change after setting different roots; the matrix-free polynomial is not using them");
    }
    /* Back to the original roots, without a setup */
    for (k = 0; k < n_inverses; k++) PetscCall(PCAIRSetPolyCoeffs(pc, level[k], which[k], orig[k], rows[k], cols[k]));
    PetscCall(CheckApply(pc, b, y_ref, y, "after restoring the roots, before a setup"));
  }

  if (newton) {
    /* Grow */
    for (k = 0; k < n_inverses; k++) {
      PetscCall(PCAIRSetPolyCoeffs(pc, level[k], which[k], grown[k], rows[k] + 2, cols[k]));
      PetscCall(CheckStored(pc, level[k], which[k], grown[k], rows[k] + 2, cols[k]));
    }
    /* A matrix-free polynomial must use the new roots straight away; an assembled inverse
       keeps using the old ones until it is rebuilt, which is the same polynomial here */
    PetscCall(CheckApply(pc, b, y_ref, y, "after growing the roots, before a setup"));

    PetscCall(PCAIRSetReusePolyCoeffs(pc, PETSC_TRUE));
    PetscCall(RebuildAndCheck(ksp, A, b_solve, x, b, y_ref, y, its_ref, "after growing the roots"));
    /* The rebuild must have used, not recomputed, the stored roots */
    for (k = 0; k < n_inverses; k++) PetscCall(CheckStored(pc, level[k], which[k], grown[k], rows[k] + 2, cols[k]));

    /* Shrink back */
    for (k = 0; k < n_inverses; k++) {
      PetscCall(PCAIRSetPolyCoeffs(pc, level[k], which[k], orig[k], rows[k], cols[k]));
      PetscCall(CheckStored(pc, level[k], which[k], orig[k], rows[k], cols[k]));
    }
    PetscCall(CheckApply(pc, b, y_ref, y, "after shrinking the roots, before a setup"));
    PetscCall(RebuildAndCheck(ksp, A, b_solve, x, b, y_ref, y, its_ref, "after shrinking the roots"));
    for (k = 0; k < n_inverses; k++) PetscCall(CheckStored(pc, level[k], which[k], orig[k], rows[k], cols[k]));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "OK: %" PetscInt_FMT " inverse(s), %s, %s, rebuild with %s, %" PetscInt_FMT " iterations\n", n_inverses, newton ? "resized" : "resizing rejected",
                        matrix_free ? "matrix-free" : "assembled", reuse_sparsity ? "reuse sparsity" : (diag_update ? "diagonal update" : "none"), its_ref));

  for (k = 0; k < n_inverses; k++) {
    PetscCall(PetscFree(orig[k]));
    PetscCall(PetscFree(grown[k]));
  }
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b_solve));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&y_ref));
  PetscCall(VecDestroy(&y));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}
