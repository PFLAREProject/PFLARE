static char help[] = "Tests the Newton GMRES polynomial does not depend on the scale of the operator.\n\n";

/*
  In exact arithmetic the GMRES polynomial is scale invariant. If A is replaced
  by s*A, the Newton roots scale by s and the residual polynomial 1 - A p(A) is
  unchanged, so ||x - A M x|| / ||x|| for M = p(A) does not depend on s.
  The tolerances used to compute, cluster and skip the roots must therefore all
  be relative to the size of the operator.

  Here we build a Newton PCPFLAREINV for a nonsymmetric 1D stencil (which has
  complex roots) at scale 1 and at a small scale, and check
  ||x - A M x|| / ||x|| matches for:
  - the matrix-free apply
  - the assembled inverse with unconstrained sparsity
  - the assembled inverse with constrained sparsity

    ./newton_scale
    mpiexec -n 2 ./newton_scale
*/
#include <petscksp.h>
#include "pflare.h"

#if defined(PETSC_USE_REAL_SINGLE)
  #define DEFAULT_CHECK_TOL 1e-3
  #define DEFAULT_SCALE     1e-6
#else
  #define DEFAULT_CHECK_TOL 1e-8
  #define DEFAULT_SCALE     1e-9
#endif

/*
   Writes a scaled 1D nonsymmetric stencil into A
*/
static PetscErrorCode SetOperatorValues(Mat A, PetscInt n, PetscReal scale)
{
  PetscInt    i, global_row_start, global_row_end_plus_one, cols[3], n_cols;
  PetscScalar vals[3];

  PetscFunctionBeginUser;
  PetscCall(MatGetOwnershipRange(A, &global_row_start, &global_row_end_plus_one));

  for (i = global_row_start; i < global_row_end_plus_one; i++) {
    n_cols = 0;
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
    PetscCall(MatSetValues(A, 1, &i, n_cols, cols, vals, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Returns ||x - A M x|| / ||x|| for a Newton PCPFLAREINV built on A = scale * stencil
*/
static PetscErrorCode ComputeResidual(PetscInt n, PetscReal scale, PetscInt poly_order, PetscInt sparsity_order, PetscBool matrix_free, PetscReal *rel_res)
{
  Mat         A;
  PC          pc;
  Vec         x, y, r;
  PetscRandom rand;
  PetscReal   r_norm, x_norm;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, 3, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 3, NULL, 2, NULL));
  PetscCall(MatSetUp(A));
  PetscCall(SetOperatorValues(A, n, scale));

  PetscCall(PCCreate(PETSC_COMM_WORLD, &pc));
  PetscCall(PCSetType(pc, PCPFLAREINV));
  PetscCall(PCPFLAREINVSetType(pc, PFLAREINV_NEWTON));
  PetscCall(PCPFLAREINVSetPolyOrder(pc, poly_order));
  PetscCall(PCPFLAREINVSetSparsityOrder(pc, sparsity_order));
  PetscCall(PCPFLAREINVSetMatrixFree(pc, matrix_free));
  PetscCall(PCSetOperators(pc, A, A));
  PetscCall(PCSetFromOptions(pc));
  PetscCall(PCSetUp(pc));

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
  PetscInt  n = 50, poly_order = 8, test;
  PetscReal tol = DEFAULT_CHECK_TOL, scale = DEFAULT_SCALE, res_one, res_scaled, rel_diff;
  // Matrix-free, assembled with unconstrained sparsity, assembled with constrained sparsity
  const PetscBool   test_mf[3]          = {PETSC_TRUE, PETSC_FALSE, PETSC_FALSE};
  const PetscInt    test_sparsity[3]    = {-1, -1, 1};
  const char *const test_names[3]       = {"matrix-free", "assembled", "assembled fixed sparsity"};
  PetscInt          sparsity_order;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PCRegister_PFLARE();
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-poly_order", &poly_order, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-scale", &scale, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-tol", &tol, NULL));

  for (test = 0; test < 3; test++) {
    // -1 means unconstrained sparsity
    sparsity_order = test_sparsity[test] < 0 ? poly_order : test_sparsity[test];

    PetscCall(ComputeResidual(n, 1.0, poly_order, sparsity_order, test_mf[test], &res_one));
    PetscCall(ComputeResidual(n, scale, poly_order, sparsity_order, test_mf[test], &res_scaled));

    rel_diff = PetscAbsReal(res_one - res_scaled) / res_one;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  %s: ||x - A M x||/||x|| = %g at scale 1, %g at scale %g\n", test_names[test], (double)res_one, (double)res_scaled, (double)scale));
    PetscCheck(rel_diff <= tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "%s: the Newton polynomial depends on the scale of the operator, relative difference %g > %g", test_names[test], (double)rel_diff, (double)tol);
  }

  PetscCall(PetscFinalize());
  return 0;
}
