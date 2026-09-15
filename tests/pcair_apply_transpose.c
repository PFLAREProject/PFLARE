static char help[] = "Tests PCApplyTranspose for PCAIR.\n\n";

/*
  Checks that PCApplyTranspose on PCAIR really is the transpose of PCApply, for
  every configuration of the reduction multigrid - the F-C point smoothing
  Kaskade cycle, the full up and down smoothing V-cycle, and the single level
  case, assembled and matrix-free.

  As in tests/pflareinv_apply_transpose.c every check compares the
  preconditioner against itself, so none of them need a reference
  implementation - which matters here because the multigrid hierarchy is built
  by a randomised CF splitting and there is nothing else to compare against:

  - the bilinear identity  y . (M x) == x . (M^T y)  over random vectors, which
    is cheap enough to run at any size
  - building M and M^T out column by column and comparing every entry, which is
    2n applies so it is only run on a small operator, but is far sharper. The
    bilinear identity contracts a whole matrix down to one number and so is only
    mildly sensitive; the explicit check catches a transposed apply that is
    subtly rather than grossly wrong, for instance one that restricted with the
    wrong operator on one level of the hierarchy.
  - PCMatApplyTranspose on a block of vectors against PCApplyTranspose column by
    column. PCAIR does not define matapplytranspose, so petsc falls back to
    exactly that loop and this is currently a near-tautology; it is here so that
    a future fused block transposed apply cannot be added without being checked,
    and it does at least exercise the transposed apply on the columns of a dense
    Mat rather than on standalone vectors.

  The operator has to be strongly nonsymmetric or this test proves nothing at
  all - a symmetric one would pass no matter what the transposed apply did. We
  use the same 1D upwind advection-diffusion stencil as the PCPFLAREINV test,
  with a shift that makes it strictly diagonally dominant, so it is cheap,
  nonsingular, and the Neumann polynomial converges on it. With -dim 2 we build
  the 5-point upwind advection-diffusion stencil on an n x n grid with a
  velocity that is not aligned with either axis, which gives a genuinely
  two dimensional CF splitting and a non-trivial lAIR restrictor.

    ./pcair_apply_transpose
    ./pcair_apply_transpose -pc_air_smooth_type fcf
    ./pcair_apply_transpose -dim 2 -n 20 -pc_air_z_type lair
    mpiexec -n 2 ./pcair_apply_transpose -pc_air_full_smoothing_up_and_down
    ./pcair_apply_transpose -transpose_solve

  The explicit check is skipped above -explicit_max rows, so -n 5000 is a
  bilinear-identity-only run at a realistic size.

  -rebuild re-does the setup with new values in the same nonzero pattern and
  re-checks, which is what covers a rebuilt hierarchy being picked up rather
  than stale operators being left in place.

  -rebuild_pattern changes the values and the nonzero pattern, so the hierarchy
  is rebuilt from scratch with a different CF splitting and different level
  sizes, which is what covers the transposed cycle picking up the per-level work
  vectors petsc's PCMG recreates at the new sizes rather than stale ones.
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
   a re-setup with the same nonzero pattern, and with extra_coupling to test one
   with a different nonzero pattern.
*/
static PetscErrorCode SetOperatorValues1D(Mat A, PetscInt n, PetscReal advection, PetscReal shift, \
                                          PetscBool extra_coupling)
{
  PetscInt    i, global_row_start, global_row_end_plus_one, cols[4], n_cols;
  PetscScalar vals[4];

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
    // A second super-diagonal, only there to give a different nonzero pattern.
    // Small enough that the row is still strictly diagonally dominant
    if (extra_coupling && i < n - 2) {
      cols[n_cols] = i + 2;
      vals[n_cols] = -0.05;
      n_cols++;
    }
    PetscCall(MatSetValues(A, 1, &i, n_cols, cols, vals, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   The 2D version - a 5-point upwind advection-diffusion stencil on an n x n
   grid, row index j*n + i. The velocity is (advection, 0.5 * advection), ie not
   aligned with either grid axis, so the strength of connection is anisotropic
   in a direction the grid does not follow and the CF splitting and the AIR
   restrictor both end up non-trivial. Upwinding puts the extra weight on the
   west and south neighbours only, so again A != A^T, and the shift keeps every
   row strictly diagonally dominant.
*/
static PetscErrorCode SetOperatorValues2D(Mat A, PetscInt n, PetscReal advection, PetscReal shift, \
                                          PetscBool extra_coupling)
{
  PetscInt    row, i, j, global_row_start, global_row_end_plus_one, cols[6], n_cols;
  PetscScalar vals[6];
  PetscReal   adv_x = advection, adv_y = 0.5 * advection;

  PetscFunctionBeginUser;
  PetscCall(MatGetOwnershipRange(A, &global_row_start, &global_row_end_plus_one));

  for (row = global_row_start; row < global_row_end_plus_one; row++) {
    i      = row % n;
    j      = row / n;
    n_cols = 0;

    // South
    if (j > 0) {
      cols[n_cols] = row - n;
      vals[n_cols] = -1.0 - adv_y;
      n_cols++;
    }
    // West
    if (i > 0) {
      cols[n_cols] = row - 1;
      vals[n_cols] = -1.0 - adv_x;
      n_cols++;
    }
    cols[n_cols] = row;
    vals[n_cols] = 4.0 + adv_x + adv_y + shift;
    n_cols++;
    // East
    if (i < n - 1) {
      cols[n_cols] = row + 1;
      vals[n_cols] = -1.0;
      n_cols++;
    }
    // North
    if (j < n - 1) {
      cols[n_cols] = row + n;
      vals[n_cols] = -1.0;
      n_cols++;
    }
    // Only there to give a different nonzero pattern, see the 1D version
    if (extra_coupling && i < n - 2) {
      cols[n_cols] = row + 2;
      vals[n_cols] = -0.05;
      n_cols++;
    }
    PetscCall(MatSetValues(A, 1, &row, n_cols, cols, vals, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetOperatorValues(Mat A, PetscInt dim, PetscInt n, PetscReal advection, \
                                        PetscReal shift, PetscBool extra_coupling)
{
  PetscFunctionBeginUser;
  if (dim == 2) {
    PetscCall(SetOperatorValues2D(A, n, advection, shift, extra_coupling));
  } else {
    PetscCall(SetOperatorValues1D(A, n, advection, shift, extra_coupling));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Builds the operator. In 2D the grid is n x n so the matrix is n^2 x n^2 -
   global_size is what the rest of the test works in.
*/
static PetscErrorCode BuildOperator(PetscInt dim, PetscInt n, PetscReal advection, PetscReal shift, \
                                    PetscInt *global_size, Mat *A_out)
{
  Mat      A;
  PetscInt size = (dim == 2) ? n * n : n;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, size, size));
  PetscCall(MatSetFromOptions(A));
  // Preallocated for the stencil as it is here - the extra coupling
  // -rebuild_pattern adds is a new nonzero on an already assembled matrix and
  // has to malloc whatever we preallocate now, which is why that path turns off
  // MAT_NEW_NONZERO_ALLOCATION_ERR
  if (dim == 2) {
    PetscCall(MatSeqAIJSetPreallocation(A, 5, NULL));
    PetscCall(MatMPIAIJSetPreallocation(A, 5, NULL, 4, NULL));
  } else {
    PetscCall(MatSeqAIJSetPreallocation(A, 3, NULL));
    PetscCall(MatMPIAIJSetPreallocation(A, 3, NULL, 2, NULL));
  }
  PetscCall(MatSetUp(A));

  PetscCall(SetOperatorValues(A, dim, n, advection, shift, PETSC_FALSE));

  *global_size = size;
  *A_out       = A;
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
   instance one that dropped the coarse grid correction on a single level.
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
   The block transposed apply has to agree with the single vector one. PCAIR
   does not define matapplytranspose, so petsc applies column by column and this
   can only fail if that ever changes - which is the point, a fused block
   transposed apply must not be able to appear without being checked. It also
   runs PCApplyTranspose on the columns of a dense Mat rather than on standalone
   vectors, which is a different vector layout to everything above.
*/
static PetscErrorCode CheckMatApplyTranspose(PC pc, Mat A, PetscRandom rand, PetscInt n, PetscInt n_cols, \
                                             PetscReal tol, const char *label)
{
  Mat       X, Y, Y_col;
  Vec       cx, cy;
  VecType   vtype;
  PetscInt  j, local_rows;
  PetscReal diff, scale;

  PetscFunctionBeginUser;
  PetscCall(MatGetLocalSize(A, &local_rows, NULL));
  PetscCall(MatGetVecType(A, &vtype));
  PetscCall(MatCreateDenseFromVecType(PetscObjectComm((PetscObject)A), vtype, local_rows, PETSC_DECIDE, \
                                      n, n_cols, PETSC_DECIDE, NULL, &X));
  PetscCall(MatDuplicate(X, MAT_DO_NOT_COPY_VALUES, &Y));
  PetscCall(MatDuplicate(X, MAT_DO_NOT_COPY_VALUES, &Y_col));

  PetscCall(MatSetRandom(X, rand));

  PetscCall(PCMatApplyTranspose(pc, X, Y));

  // The same block, one column at a time
  for (j = 0; j < n_cols; j++) {
    PetscCall(MatDenseGetColumnVecRead(X, j, &cx));
    PetscCall(MatDenseGetColumnVecWrite(Y_col, j, &cy));
    PetscCall(PCApplyTranspose(pc, cx, cy));
    PetscCall(MatDenseRestoreColumnVecWrite(Y_col, j, &cy));
    PetscCall(MatDenseRestoreColumnVecRead(X, j, &cx));
  }

  PetscCall(MatNorm(Y, NORM_FROBENIUS, &scale));
  PetscCall(MatAXPY(Y_col, -1.0, Y, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(Y_col, NORM_FROBENIUS, &diff));

  if (scale < 1.0) scale = 1.0;
  PetscCheck(diff / scale <= tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, \
             "%s: PCMatApplyTranspose does not match PCApplyTranspose column by column" \
             " - relative Frobenius difference %g > %g", label, (double)(diff / scale), (double)tol);

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, \
                        "  %s: PCMatApplyTranspose on %" PetscInt_FMT " columns matches the column by column" \
                        " transposed apply\n", label, n_cols));

  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&Y));
  PetscCall(MatDestroy(&Y_col));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   All three algebraic checks in one, so the rebuild paths run exactly what the
   first setup ran.
*/
static PetscErrorCode CheckAll(PC pc, Mat A, PetscRandom rand, PetscInt n, PetscInt n_pairs, \
                               PetscInt explicit_max, PetscInt matapply_cols, PetscReal tol, \
                               const char *label)
{
  PetscFunctionBeginUser;
  PetscCall(CheckTransposeIdentity(pc, A, rand, tol, n_pairs, label));
  // 2n applies, so only on an operator small enough to make that cheap
  if (n <= explicit_max) PetscCall(CheckTransposeExplicitly(pc, A, n, tol, label));
  if (matapply_cols > 0) PetscCall(CheckMatApplyTranspose(pc, A, rand, n, matapply_cols, tol, label));
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
  PetscLogStage      stage;

  PetscFunctionBeginUser;
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecDuplicate(b, &r));
  PetscCall(VecSet(b, 1.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCAIR));
  PetscCall(KSPSetFromOptions(ksp));
  // After KSPSetFromOptions so this wins over -ksp_pc_side, we are deliberately
  // comparing the two sides here
  PetscCall(KSPSetPCSide(ksp, side));

  // Set up outside the log stage so -log_view shows the solve on its own, in
  // particular the host/device copy counts of the transposed apply without the
  // hierarchy setup mixed in
  PetscCall(KSPSetUp(ksp));
  PetscCall(PetscLogStageRegister(transpose ? "KSPSolveTranspose" : (side == PC_LEFT ? "KSPSolve left" : "KSPSolve right"), &stage));
  PetscCall(PetscLogStagePush(stage));
  if (transpose) {
    PetscCall(KSPSolveTranspose(ksp, b, x));
  } else {
    PetscCall(KSPSolve(ksp, b, x));
  }
  PetscCall(PetscLogStagePop());

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

   With M the multigrid cycle, left preconditioned GMRES on the transposed
   system iterates on M^T A^T = (A M)^T, which has the spectrum of A M, ie of the
   RIGHT preconditioned forward problem rather than the left preconditioned M A.
   So the forward count to compare against in general is the right preconditioned
   one, and that is what this asserts against. Both forward counts are reported,
   and on an operator as well behaved as this one they come out the same anyway,
   so this does not actually distinguish the two - it is the theoretically
   correct reference, not a demonstrated distinction.

   A transposed cycle that was subtly wrong - a smoother applied in the forward
   order, say - would still often converge, just more slowly, so this catches
   things the algebraic checks above would not.

   The forward solves are run first so that this reports a forward failure as a
   forward failure rather than as a transposed one.
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
  PetscInt    dim = 1, n = 200, size, n_pairs = 5, explicit_max = 400, matapply_cols = 3, its_slack = 2;
  PetscReal   advection = 1.0, shift = 0.1, tol = DEFAULT_CHECK_TOL;
  PetscBool   transpose_solve = PETSC_FALSE, rebuild = PETSC_FALSE, rebuild_pattern = PETSC_FALSE;

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_pairs", &n_pairs, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-explicit_max", &explicit_max, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-matapply_cols", &matapply_cols, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-advection", &advection, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-check_tol", &tol, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-transpose_solve", &transpose_solve, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-its_slack", &its_slack, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-rebuild", &rebuild, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-rebuild_pattern", &rebuild_pattern, NULL));

  PetscCheck(dim == 1 || dim == 2, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, \
             "-dim must be 1 or 2, not %" PetscInt_FMT, dim);

  // Register the PFLARE types
  PCRegister_PFLARE();

  PetscCall(BuildOperator(dim, n, advection, shift, &size, &A));

  // Seeded so a failure is reproducible
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(PetscRandomSetSeed(rand, 314159));
  PetscCall(PetscRandomSeed(rand));

  // Everything about the hierarchy comes from the command line, so one driver
  // covers every PCAIR configuration
  PetscCall(PCCreate(PETSC_COMM_WORLD, &pc));
  PetscCall(PCSetType(pc, PCAIR));
  PetscCall(PCSetOperators(pc, A, A));
  PetscCall(PCSetFromOptions(pc));
  PetscCall(PCSetUp(pc));

  // The second and later pairs also cover any state the transposed apply caches
  // rather than rebuilding on every apply
  PetscCall(CheckAll(pc, A, rand, size, n_pairs, explicit_max, matapply_cols, tol, "first setup"));

  if (rebuild) {
    // New values in the same nonzero pattern. The hierarchy keeps its sizes and
    // its CF splitting, but every operator on it has changed, so this is what
    // catches the transposed cycle holding on to anything from the first setup
    PetscCall(SetOperatorValues(A, dim, n, 2.0 * advection, shift, PETSC_FALSE));
    PetscCall(PCSetOperators(pc, A, A));
    PetscCall(PCSetUp(pc));
    PetscCall(CheckAll(pc, A, rand, size, n_pairs, explicit_max, matapply_cols, tol, "after rebuild"));
  }

  if (rebuild_pattern) {
    // New values AND an extra coupling, so petsc reports DIFFERENT_NONZERO_PATTERN
    // and the whole hierarchy is thrown away and rebuilt - different number of
    // levels, different CF splitting, different level sizes. That is what
    // exercises the transposed cycle running on the per-level work vectors the
    // inner PCMG recreates at the new sizes rather than on anything stale
    PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    PetscCall(SetOperatorValues(A, dim, n, 2.0 * advection, shift, PETSC_TRUE));
    PetscCall(PCSetOperators(pc, A, A));
    PetscCall(PCSetUp(pc));
    PetscCall(CheckAll(pc, A, rand, size, n_pairs, explicit_max, matapply_cols, tol, "after rebuild_pattern"));
  }

  PetscCall(PCDestroy(&pc));

  if (transpose_solve) PetscCall(CheckTransposeSolve(A, its_slack));

  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}
