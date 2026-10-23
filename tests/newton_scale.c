static char help[] = "Tests the Newton GMRES polynomial does not depend on the scale of the operator.\n\n";

/*
  In exact arithmetic the GMRES polynomial is scale invariant. If A is replaced
  by s*A, the Newton roots scale by s and the residual polynomial 1 - A p(A) is
  unchanged, so ||x - A M x|| / ||x|| for M = p(A) does not depend on s.
  The tolerances used to compute, cluster, order and skip the roots must therefore
  all be relative to the size of the operator.

  Here we build Newton PCPFLAREINVs at scale 1 and at a small and a large scale
  and check that:
  - the roots computed at scale s, divided by s, match those at scale 1, in the
    same (Leja) order, and with the same number of extra roots added
  - ||x - A M x|| / ||x|| matches

  This is done for
  - a nonsymmetric 1D stencil (which has complex roots) at low order with the
    matrix-free apply and the assembled inverse with unconstrained and
    constrained sparsity
  - the same stencil at high order (>= 75), where LAPACK overwrites the Hessenberg
    matrix with values that don't scale with the matrix
  - a block diagonal operator with 2x2 blocks [d y_k; -y_k d], with the y_k
    log-spaced, which has roots with a large product of factors, so extra
    (duplicate) roots are added and have to be Leja ordered

  In single precision the order 40 polynomial on the block operator is unstable at
  every scale (as in newton_imag_roots) so is skipped, and at order 80 the Leja
  ordering of the roots is decided by rounding, so there we only compare
  ||x - A M x|| / ||x||, with a looser tolerance

    ./newton_scale
    mpiexec -n 2 ./newton_scale
*/
#include <petscksp.h>
#include "pflare.h"

#if defined(PETSC_USE_REAL_SINGLE)
  #define DEFAULT_CHECK_TOL 1e-3
  #define DEFAULT_ROOT_TOL  1e-2
  #define HIGH_ORDER_TOL    1e-2
  #define SMALL_SCALE       1e-6
  #define LARGE_SCALE       1e6
  #define IS_SINGLE         PETSC_TRUE
#else
  #define DEFAULT_CHECK_TOL 1e-8
  #define DEFAULT_ROOT_TOL  1e-8
  #define HIGH_ORDER_TOL    1e-8
  #define SMALL_SCALE       1e-9
  #define LARGE_SCALE       1e9
  #define IS_SINGLE         PETSC_FALSE
#endif

#define N_CASES 7

typedef enum {
  OP_STENCIL,
  OP_BLOCKS
} OperatorType;

typedef struct {
  OperatorType op;
  PetscInt     n, poly_order, sparsity_order;
  PetscBool    matrix_free, extra_roots;
  // In single precision, whether to skip the case or only compare the residual
  PetscBool    skip_single, residual_only_single;
  const char  *name;
} TestCase;

/*
   Writes the scaled operator into A
*/
static PetscErrorCode SetOperatorValues(Mat A, OperatorType op, PetscInt n, PetscReal scale)
{
  PetscInt    i, global_row_start, global_row_end_plus_one, cols[3], n_cols;
  PetscScalar vals[3];

  PetscFunctionBeginUser;
  PetscCall(MatGetOwnershipRange(A, &global_row_start, &global_row_end_plus_one));

  for (i = global_row_start; i < global_row_end_plus_one; i++) {
    n_cols = 0;
    if (op == OP_STENCIL) {
      if (i > 0) {
        cols[n_cols] = i - 1;
        vals[n_cols] = -1.0 * scale;
        n_cols++;
      }
      cols[n_cols] = i;
      vals[n_cols] = 1.2 * scale;
      n_cols++;
      if (i < n - 1) {
        cols[n_cols] = i + 1;
        vals[n_cols] = -0.1 * scale;
        n_cols++;
      }
    } else {
      // 2x2 blocks [d y_k; -y_k d], with y_k log-spaced over 3 orders of magnitude
      cols[n_cols] = i;
      vals[n_cols] = 0.05 * scale;
      n_cols++;
      cols[n_cols] = i % 2 == 0 ? i + 1 : i - 1;
      vals[n_cols] = scale * PetscPowReal(10.0, 3.0 * (PetscReal)(i / 2) / (PetscReal)(n / 2 - 1));
      if (i % 2 == 1) vals[n_cols] = -vals[n_cols];
      n_cols++;
    }
    PetscCall(MatSetValues(A, 1, &i, n_cols, cols, vals, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Builds a Newton PCPFLAREINV on the scaled operator, returns ||x - A M x|| / ||x||
   and a copy of the roots divided by the scale (column-major, real then imaginary)
*/
static PetscErrorCode ComputeResidual(const TestCase *test, PetscReal scale, PetscReal *rel_res, PetscReal **roots, PetscInt *n_roots)
{
  Mat         A;
  PC          pc;
  Vec         x, y, r;
  PetscRandom rand;
  PetscReal   r_norm, x_norm, *coeffs;
  PetscInt    i, cols;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, test->n, test->n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, 3, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 3, NULL, 2, NULL));
  PetscCall(MatSetUp(A));
  PetscCall(SetOperatorValues(A, test->op, test->n, scale));

  PetscCall(PCCreate(PETSC_COMM_WORLD, &pc));
  PetscCall(PCSetType(pc, PCPFLAREINV));
  PetscCall(PCPFLAREINVSetType(pc, PFLAREINV_NEWTON));
  PetscCall(PCPFLAREINVSetPolyOrder(pc, test->poly_order));
  PetscCall(PCPFLAREINVSetSparsityOrder(pc, test->sparsity_order));
  PetscCall(PCPFLAREINVSetMatrixFree(pc, test->matrix_free));
  PetscCall(PCSetOperators(pc, A, A));
  PetscCall(PCSetFromOptions(pc));
  PetscCall(PCSetUp(pc));

  PetscCall(PCPFLAREINVGetPolyCoeffs(pc, &coeffs, n_roots, &cols));
  PetscCall(PetscMalloc1(2 * *n_roots, roots));
  for (i = 0; i < 2 * *n_roots; i++) (*roots)[i] = coeffs[i] / scale;

  // The same random x for every scale
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(VecDuplicate(x, &r));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(VecSetRandom(x, rand));

  // r = x - A M x
  PetscCall(PCApply(pc, x, y));
  PetscCall(MatMult(A, y, r));
  PetscCall(VecAYPX(r, -1.0, x));
  PetscCall(VecNorm(r, NORM_2, &r_norm));
  PetscCall(VecNorm(x, NORM_2, &x_norm));
  *rel_res = r_norm / x_norm;

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&r));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PCDestroy(&pc));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  PetscInt  test, s, i, n_roots_one, n_roots_scaled;
  PetscReal tol = DEFAULT_CHECK_TOL, root_tol = DEFAULT_ROOT_TOL, res_one, res_scaled, res_diff;
  PetscReal root_diff, root_max, *roots_one, *roots_scaled;
  PetscReal scales[2] = {SMALL_SCALE, LARGE_SCALE};
  // -1 sparsity order means unconstrained
  const TestCase tests[N_CASES] = {
    {OP_STENCIL, 50,  8,  -1, PETSC_TRUE,  PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, "stencil order 8 matrix-free"},
    {OP_STENCIL, 50,  8,  -1, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, "stencil order 8 assembled"},
    {OP_STENCIL, 50,  8,  1,  PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, "stencil order 8 assembled fixed sparsity"},
    {OP_STENCIL, 100, 80, -1, PETSC_TRUE,  PETSC_FALSE, PETSC_FALSE, PETSC_TRUE,  "stencil order 80 matrix-free"},
    {OP_BLOCKS,  200, 4,  -1, PETSC_TRUE,  PETSC_TRUE,  PETSC_FALSE, PETSC_FALSE, "blocks order 4 matrix-free"},
    {OP_BLOCKS,  100, 40, -1, PETSC_TRUE,  PETSC_TRUE,  PETSC_TRUE,  PETSC_FALSE, "blocks order 40 matrix-free"},
    {OP_BLOCKS,  100, 10, 1,  PETSC_FALSE, PETSC_TRUE,  PETSC_FALSE, PETSC_FALSE, "blocks order 10 assembled fixed sparsity"},
  };
  TestCase  this_test;
  PetscBool check_roots;
  PetscReal case_tol;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PCRegister_PFLARE();
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-tol", &tol, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-root_tol", &root_tol, NULL));

  for (test = 0; test < N_CASES; test++) {
    this_test = tests[test];
    if (IS_SINGLE && this_test.skip_single) continue;
    if (this_test.sparsity_order < 0) this_test.sparsity_order = this_test.poly_order;
    check_roots = (PetscBool)!(IS_SINGLE && this_test.residual_only_single);
    case_tol    = IS_SINGLE && this_test.residual_only_single ? HIGH_ORDER_TOL : tol;

    PetscCall(ComputeResidual(&this_test, 1.0, &res_one, &roots_one, &n_roots_one));
    // Make sure we are testing the extra roots if we want to be
    PetscCheck(!this_test.extra_roots || n_roots_one > this_test.poly_order + 1, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "%s: no extra roots added, test proves nothing", this_test.name);

    for (s = 0; s < 2; s++) {
      PetscCall(ComputeResidual(&this_test, scales[s], &res_scaled, &roots_scaled, &n_roots_scaled));

      PetscCheck(n_roots_one == n_roots_scaled, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "%s: %" PetscInt_FMT " roots at scale 1 but %" PetscInt_FMT " at scale %g", this_test.name, n_roots_one, n_roots_scaled, (double)scales[s]);

      // The roots/scale should match those at scale 1, in the same order
      // Which root of a complex conjugate pair comes first can be decided by rounding,
      // and doesn't matter, so we just compare the size of the imaginary parts
      root_diff = 0.0;
      root_max  = 0.0;
      for (i = 0; i < n_roots_one; i++) {
        root_diff = PetscMax(root_diff, PetscSqrtReal(PetscSqr(roots_one[i] - roots_scaled[i]) + PetscSqr(PetscAbsReal(roots_one[n_roots_one + i]) - PetscAbsReal(roots_scaled[n_roots_one + i]))));
        root_max  = PetscMax(root_max, PetscSqrtReal(PetscSqr(roots_one[i]) + PetscSqr(roots_one[n_roots_one + i])));
      }

      res_diff = PetscAbsReal(res_one - res_scaled) / res_one;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  %s, scale %g: %" PetscInt_FMT " roots, ||x - A M x||/||x|| = %g (%g at scale 1)\n", this_test.name, (double)scales[s], n_roots_scaled, (double)res_scaled, (double)res_one));
      PetscCheck(!check_roots || root_diff <= root_tol * root_max, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "%s: the roots depend on the scale of the operator, relative difference %g > %g at scale %g", this_test.name, (double)(root_diff / root_max), (double)root_tol, (double)scales[s]);
      PetscCheck(res_diff <= case_tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "%s: the Newton polynomial depends on the scale of the operator, relative difference %g > %g at scale %g", this_test.name, (double)res_diff, (double)case_tol, (double)scales[s]);
      PetscCall(PetscFree(roots_scaled));
    }
    PetscCall(PetscFree(roots_one));
  }

  PetscCall(PetscFinalize());
  return 0;
}
