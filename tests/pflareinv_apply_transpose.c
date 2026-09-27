static char help[] = "Tests PCApplyTranspose for PCPFLAREINV.\n\n";

/*
  Checks that PCApplyTranspose on PCPFLAREINV really is the transpose of PCApply,
  for every inverse type, both assembled and matrix-free.

  There are two checks, both comparing the preconditioner against itself, so
  neither needs a reference implementation - which matters for the matrix-free
  polynomials, where the assembled and matrix-free applies are not bit-identical
  operators:

  - the bilinear identity  y . (M x) == x . (M^T y)  over random vectors, which
    is cheap enough to run at any size
  - building M and M^T out column by column and comparing every entry, which is
    2n applies so it is only run on a small operator, but is far sharper. The
    bilinear identity contracts a whole matrix down to one number and so is only
    mildly sensitive; the explicit check catches a transposed apply that is
    subtly rather than grossly wrong.

  The operator has to be strongly nonsymmetric or this test proves nothing at
  all - a symmetric one would pass no matter what the transposed apply did. We
  use a 1D upwind advection-diffusion stencil with a shift that makes it
  strictly diagonally dominant, so it is cheap, nonsingular, and the Neumann
  polynomial converges on it.

    ./pflareinv_apply_transpose
    ./pflareinv_apply_transpose -pc_pflareinv_type neumann -pc_pflareinv_matrix_free
    mpiexec -n 2 ./pflareinv_apply_transpose -pc_pflareinv_type newton
    ./pflareinv_apply_transpose -transpose_solve

  The explicit check is skipped above -explicit_max rows, so -n 5000 is a
  bilinear-identity-only run at a realistic size.

  -rebuild re-does the setup with new values in the same nonzero pattern and
  re-checks, which is what covers the transposed twin being refreshed rather
  than left pointing at stale coefficients.

  -toggle_matrix_free (with -rebuild) also flips -pc_pflareinv_matrix_free
  through the options database and PCSetFromOptions before the re-setup. That
  has to reset the PC and rebuild the inverse in the new form, rather than
  reusing an assembled inverse as a matshell (or vice versa).
*/
#include <petscksp.h>
#include "pflare.h"

// Tolerance for the transpose identity
#if defined(PETSC_USE_REAL_SINGLE)
  #define DEFAULT_CHECK_TOL 1e-5
#else
  #define DEFAULT_CHECK_TOL 1e-10
#endif

/*
   Writes the 1D upwind advection-diffusion stencil into A. The advection makes
   the sub and super diagonals differ, ie A != A^T, and the shift keeps it
   strictly diagonally dominant. Called again with a different advection to test
   a re-setup with the same nonzero pattern.
*/
static PetscErrorCode SetOperatorValues(Mat A, PetscInt n, PetscReal advection, PetscReal shift)
{
  PetscInt    i, global_row_start, global_row_end_plus_one, cols[3], n_cols;
  PetscScalar vals[3];

  PetscFunctionBeginUser;
  PetscCall(MatGetOwnershipRange(A, &global_row_start, &global_row_end_plus_one));

  for (i = global_row_start; i < global_row_end_plus_one; i++) {
    n_cols = 0;
    if (i > 0) {
      cols[n_cols] = i - 1;
      vals[n_cols] = -1.0 - advection;
      n_cols++;
    }
    cols[n_cols] = i;
    vals[n_cols] = 2.0 + advection + shift;
    n_cols++;
    if (i < n - 1) {
      cols[n_cols] = i + 1;
      vals[n_cols] = -1.0;
      n_cols++;
    }
    PetscCall(MatSetValues(A, 1, &i, n_cols, cols, vals, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BuildOperator(PetscInt n, PetscReal advection, PetscReal shift, Mat *A_out)
{
  Mat A;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  // Three per row on the diagonal block, at most one either side off it
  PetscCall(MatSeqAIJSetPreallocation(A, 3, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 3, NULL, 2, NULL));
  PetscCall(MatSetUp(A));

  PetscCall(SetOperatorValues(A, n, advection, shift));

  *A_out = A;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   The transpose identity itself. Uses VecTDot rather than VecDot so this is the
   bilinear form in both real and complex builds - PCApplyTranspose is the true
   transpose, not the Hermitian transpose.
*/
static PetscErrorCode CheckTransposeIdentity(PC pc, Mat A, PetscRandom rand, PetscReal tol, \
                                             PetscInt n_pairs, const char *label)
{
  Vec         x, y, mx, mty;
  PetscScalar lhs, rhs;
  PetscReal   diff, denom;
  PetscInt    i;

  PetscFunctionBeginUser;
  PetscCall(MatCreateVecs(A, &x, &mx));
  PetscCall(MatCreateVecs(A, &y, &mty));

  for (i = 0; i < n_pairs; i++) {
    // Random vectors - constant ones would hide a row/column mixup
    PetscCall(VecSetRandom(x, rand));
    PetscCall(VecSetRandom(y, rand));

    PetscCall(PCApply(pc, x, mx));
    PetscCall(PCApplyTranspose(pc, y, mty));

    // y . (M x) has to equal x . (M^T y)
    PetscCall(VecTDot(mx, y, &lhs));
    PetscCall(VecTDot(x, mty, &rhs));

    diff  = PetscAbsScalar(lhs - rhs);
    denom = PetscMax(PetscAbsScalar(lhs), PetscAbsScalar(rhs));
    if (denom < 1.0) denom = 1.0;

    PetscCheck(diff / denom <= tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, \
               "%s: PCApplyTranspose is not the transpose of PCApply on pair %" PetscInt_FMT \
               " - y.(Mx) = %g, x.(M^T y) = %g, relative difference %g > %g", \
               label, i, (double)PetscRealPart(lhs), (double)PetscRealPart(rhs), \
               (double)(diff / denom), (double)tol);
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  %s: transpose identity holds over %" PetscInt_FMT " vector pairs\n", \
                        label, n_pairs));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&mx));
  PetscCall(VecDestroy(&mty));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Builds the preconditioner out explicitly, one column at a time, either as M or
   as M^T.
*/
static PetscErrorCode BuildExplicit(PC pc, Mat A, PetscInt n, PetscBool transpose, Mat *out)
{
  Mat      dense;
  Vec      e, col;
  VecType  vtype;
  PetscInt j, local_rows;

  PetscFunctionBeginUser;
  PetscCall(MatGetLocalSize(A, &local_rows, NULL));
  // From the operator's vector type, so the columns we hand to PCApply are the
  // same type as the vectors the preconditioner works with on the device
  PetscCall(MatGetVecType(A, &vtype));
  PetscCall(MatCreateDenseFromVecType(PetscObjectComm((PetscObject)A), vtype, local_rows, PETSC_DECIDE, \
                                      n, n, PETSC_DECIDE, NULL, &dense));
  PetscCall(MatCreateVecs(A, NULL, &e));

  for (j = 0; j < n; j++) {
    PetscCall(VecZeroEntries(e));
    PetscCall(VecSetValue(e, j, 1.0, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(e));
    PetscCall(VecAssemblyEnd(e));

    PetscCall(MatDenseGetColumnVecWrite(dense, j, &col));
    if (transpose) {
      PetscCall(PCApplyTranspose(pc, e, col));
    } else {
      PetscCall(PCApply(pc, e, col));
    }
    PetscCall(MatDenseRestoreColumnVecWrite(dense, j, &col));
  }

  PetscCall(MatAssemblyBegin(dense, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(dense, MAT_FINAL_ASSEMBLY));
  PetscCall(VecDestroy(&e));

  *out = dense;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   The sharp version of the check - build M and M^T out column by column and
   compare every single entry, rather than the single number the bilinear
   identity gives us. Only worth doing on a small operator, but it is what
   catches a transposed apply that is subtly rather than grossly wrong, for
   instance one that applied the diagonal scaling on the wrong side.
*/
static PetscErrorCode CheckTransposeExplicitly(PC pc, Mat A, PetscInt n, PetscReal tol, const char *label)
{
  Mat       m, mt, mt_transposed;
  PetscReal diff, scale;

  PetscFunctionBeginUser;
  PetscCall(BuildExplicit(pc, A, n, PETSC_FALSE, &m));
  PetscCall(BuildExplicit(pc, A, n, PETSC_TRUE, &mt));

  // (M^T)^T has to be M, entry for entry
  PetscCall(MatTranspose(mt, MAT_INITIAL_MATRIX, &mt_transposed));
  PetscCall(MatNorm(m, NORM_FROBENIUS, &scale));
  PetscCall(MatAXPY(mt_transposed, -1.0, m, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(mt_transposed, NORM_FROBENIUS, &diff));

  if (scale < 1.0) scale = 1.0;
  PetscCheck(diff / scale <= tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, \
             "%s: the explicit PCApplyTranspose matrix is not the transpose of the explicit PCApply matrix" \
             " - relative Frobenius difference %g > %g", label, (double)(diff / scale), (double)tol);

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, \
                        "  %s: explicit %" PetscInt_FMT "x%" PetscInt_FMT \
                        " transpose matches entry for entry, relative difference %g\n", \
                        label, n, n, (double)(diff / scale)));

  PetscCall(MatDestroy(&m));
  PetscCall(MatDestroy(&mt));
  PetscCall(MatDestroy(&mt_transposed));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Runs one solve and reports the iteration count, checking both that the KSP
   converged and that the answer really does solve the system it claimed to.
*/
static PetscErrorCode RunSolve(Mat A, PCSide side, PetscBool transpose, PetscInt *its_out)
{
  KSP                ksp;
  PC                 pc;
  Vec                b, x, r;
  PetscReal          rnorm, bnorm;
  KSPConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecDuplicate(b, &r));
  PetscCall(VecSet(b, 1.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCPFLAREINV));
  PetscCall(KSPSetFromOptions(ksp));
  // After KSPSetFromOptions so this wins over -ksp_pc_side, we are deliberately
  // comparing the two sides here
  PetscCall(KSPSetPCSide(ksp, side));

  if (transpose) {
    PetscCall(KSPSolveTranspose(ksp, b, x));
  } else {
    PetscCall(KSPSolve(ksp, b, x));
  }

  PetscCall(KSPGetConvergedReason(ksp, &reason));
  PetscCheck(reason > 0, PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, \
             "%s did not converge, reason %s", transpose ? "KSPSolveTranspose" : "KSPSolve", \
             KSPConvergedReasons[reason]);

  // The true residual of the system we actually asked for
  if (transpose) {
    PetscCall(MatMultTranspose(A, x, r));
  } else {
    PetscCall(MatMult(A, x, r));
  }
  PetscCall(VecAXPY(r, -1.0, b));
  PetscCall(VecNorm(r, NORM_2, &rnorm));
  PetscCall(VecNorm(b, NORM_2, &bnorm));

  PetscCheck(rnorm / bnorm <= 1e-4, PETSC_COMM_WORLD, PETSC_ERR_PLIB, \
             "%s converged but the answer does not solve the system, relative residual %g", \
             transpose ? "KSPSolveTranspose" : "KSPSolve", (double)(rnorm / bnorm));

  PetscCall(KSPGetIterationNumber(ksp, its_out));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&r));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   End to end check - does the transposed apply actually precondition as well as
   the forward one does?

   With M the approximate inverse, left preconditioned GMRES on the transposed
   system iterates on M^T A^T = (A M)^T, which has the spectrum of A M, ie of the
   RIGHT preconditioned forward problem rather than the left preconditioned M A.
   So the forward count to compare against in general is the right preconditioned
   one, and that is what this asserts against. Both forward counts are reported,
   and on an operator as well behaved as this one they come out the same anyway,
   so this does not actually distinguish the two - it is the theoretically
   correct reference, not a demonstrated distinction.

   A transposed apply that was subtly wrong would still often converge, just more
   slowly, so this catches things the algebraic checks above would not.
*/
static PetscErrorCode CheckTransposeSolve(Mat A, PetscInt its_slack)
{
  PetscInt its_left, its_right, its_transpose, gap;

  PetscFunctionBeginUser;
  PetscCall(RunSolve(A, PC_LEFT, PETSC_FALSE, &its_left));
  PetscCall(RunSolve(A, PC_RIGHT, PETSC_FALSE, &its_right));
  PetscCall(RunSolve(A, PC_LEFT, PETSC_TRUE, &its_transpose));

  gap = PetscAbsInt(its_transpose - its_right);
  PetscCheck(gap <= its_slack, PETSC_COMM_WORLD, PETSC_ERR_PLIB, \
             "the transposed solve does not converge like the forward one - transposed took %" PetscInt_FMT \
             " iterations against %" PetscInt_FMT " for the equivalent forward right preconditioned solve," \
             " a gap of %" PetscInt_FMT " > %" PetscInt_FMT, its_transpose, its_right, gap, its_slack);

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, \
                        "  solves converged in %" PetscInt_FMT " its transposed, against %" PetscInt_FMT \
                        " forward right preconditioned and %" PetscInt_FMT \
                        " forward left preconditioned\n", its_transpose, its_right, its_left));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat         A;
  PC          pc;
  PetscRandom rand;
  PetscInt    n = 200, n_pairs = 5, explicit_max = 400, its_slack = 2;
  PetscReal   advection = 1.0, shift = 0.1, tol = DEFAULT_CHECK_TOL;
  PetscBool   transpose_solve = PETSC_FALSE, rebuild = PETSC_FALSE, toggle_matrix_free = PETSC_FALSE;
  PetscBool   matrix_free, is_shell;
  Mat         mat_inverse;

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_pairs", &n_pairs, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-explicit_max", &explicit_max, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-advection", &advection, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-check_tol", &tol, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-transpose_solve", &transpose_solve, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-its_slack", &its_slack, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-rebuild", &rebuild, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-toggle_matrix_free", &toggle_matrix_free, NULL));

  // Register the PFLARE types
  PCRegister_PFLARE();

  PetscCall(BuildOperator(n, advection, shift, &A));

  // Seeded so a failure is reproducible
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(PetscRandomSetSeed(rand, 314159));
  PetscCall(PetscRandomSeed(rand));

  PetscCall(PCCreate(PETSC_COMM_WORLD, &pc));
  PetscCall(PCSetType(pc, PCPFLAREINV));
  PetscCall(PCSetOperators(pc, A, A));
  PetscCall(PCSetFromOptions(pc));
  PetscCall(PCSetUp(pc));

  // The second and later pairs also cover the transposed twin being cached
  // rather than rebuilt on every apply
  PetscCall(CheckTransposeIdentity(pc, A, rand, tol, n_pairs, "first setup"));
  // 2n applies, so only on an operator small enough to make that cheap
  if (n <= explicit_max) PetscCall(CheckTransposeExplicitly(pc, A, n, tol, "first setup"));

  if (rebuild) {
    // New values in the same nonzero pattern. This is what exercises the twin
    // picking up refreshed coefficients and a refreshed diagonal, rather than
    // holding on to the ones it was built with
    PetscCall(SetOperatorValues(A, n, 2.0 * advection, shift));
    if (toggle_matrix_free) {
      // Flip matrix-free through the options database, the PC has to notice
      // the change and rebuild rather than reuse the old inverse
      PetscCall(PCPFLAREINVGetMatrixFree(pc, &matrix_free));
      PetscCall(PetscOptionsSetValue(NULL, "-pc_pflareinv_matrix_free", matrix_free ? "false" : "true"));
      PetscCall(PCSetFromOptions(pc));
    }
    PetscCall(PCSetOperators(pc, A, A));
    PetscCall(PCSetUp(pc));
    if (toggle_matrix_free) {
      // The inverse has to be in the form we asked for
      PetscCall(PCPFLAREINVGetMatrixFree(pc, &matrix_free));
      PetscCall(PCPFLAREINVGetInverseMat(pc, &mat_inverse));
      PetscCall(PetscObjectTypeCompare((PetscObject)mat_inverse, MATSHELL, &is_shell));
      PetscCheck(is_shell == matrix_free, PETSC_COMM_WORLD, PETSC_ERR_PLIB, \
                 "after toggling -pc_pflareinv_matrix_free the inverse is %s but matrix_free is %s", \
                 is_shell ? "a matshell" : "assembled", matrix_free ? "true" : "false");
    }
    PetscCall(CheckTransposeIdentity(pc, A, rand, tol, n_pairs, "after rebuild"));
    if (n <= explicit_max) PetscCall(CheckTransposeExplicitly(pc, A, n, tol, "after rebuild"));
  }

  PetscCall(PCDestroy(&pc));

  if (transpose_solve) PetscCall(CheckTransposeSolve(A, its_slack));

  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}
