static char help[] = "Tests changing PCAIR options between setups on the same pc, \n\
with the values of the pmat changing (same nonzero pattern) between the solves.\n\
\n\
  -toggle_reuse_sparsity : flip -pc_air_reuse_sparsity before every solve after the first\n\
  -complete_graph        : use a dense matrix whose hierarchy loses one point per level\n\
  -min_levels            : fail if the hierarchy has fewer levels than this\n\
  -m                     : the matrix has m * m rows (default 20)\n\
  -nsteps                : number of solves (default 4)\n\n";

#include <petscksp.h>
#include "pflare.h"

// Fills (or refills) a nonsymmetric 2D advection-diffusion like five point stencil
// The nonzero pattern never changes, the values depend on step
// With complete_graph every row is instead coupled equally to every other row, so
// the strength of connection graph is complete and an independent set of F points
// can only ever have one point in it, ie the coarsening removes one point per level
static PetscErrorCode FillMat(Mat A, PetscInt m, PetscInt step, PetscBool complete_graph)
{
  PetscInt Istart, Iend, II, i, j, J;

  PetscFunctionBegin;
  PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
  PetscCall(MatZeroEntries(A));
  for (II = Istart; II < Iend; II++) {
    if (complete_graph) {
      for (J = 0; J < m * m; J++) {
        PetscScalar v = (J == II) ? (PetscScalar)(m * m + 1 + step) : -1.0;
        PetscCall(MatSetValues(A, 1, &II, 1, &J, &v, INSERT_VALUES));
      }
      continue;
    }
    // A deterministic row dependent perturbation that changes with the step
    PetscReal   r = (PetscReal)((II * 7919 + step * 104729) % 1000) / 1000.0;
    PetscReal   a = 1.0 + 0.5 * r, eps = 0.01, d = 0.0;
    PetscScalar v;

    i = II / m;
    j = II - i * m;
    if (i > 0)     { J = II - m; v = -(eps + a);       d += eps + a;       PetscCall(MatSetValues(A, 1, &II, 1, &J, &v, INSERT_VALUES)); }
    if (i < m - 1) { J = II + m; v = -eps;             d += eps;           PetscCall(MatSetValues(A, 1, &II, 1, &J, &v, INSERT_VALUES)); }
    if (j > 0)     { J = II - 1; v = -(eps + 0.5 * a); d += eps + 0.5 * a; PetscCall(MatSetValues(A, 1, &II, 1, &J, &v, INSERT_VALUES)); }
    if (j < m - 1) { J = II + 1; v = -eps;             d += eps;           PetscCall(MatSetValues(A, 1, &II, 1, &J, &v, INSERT_VALUES)); }
    v = d + 0.1 + 0.5 * r;
    PetscCall(MatSetValues(A, 1, &II, 1, &II, &v, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat                A;
  Vec                x, b;
  KSP                ksp;
  PC                 pc;
  PetscInt           m = 20, nsteps = 4, step, its, min_levels = -1, num_levels;
  PetscBool          toggle_reuse_sparsity = PETSC_FALSE, reuse, complete_graph = PETSC_FALSE;
  KSPConvergedReason reason;

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  // Register the pflare types
  PCRegister_PFLARE();

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nsteps", &nsteps, NULL));
  // Flip -pc_air_reuse_sparsity before every solve after the first
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-toggle_reuse_sparsity", &toggle_reuse_sparsity, NULL));
  // Use a matrix whose hierarchy loses one point per level
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-complete_graph", &complete_graph, NULL));
  // Fail if the hierarchy has fewer levels than this
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-min_levels", &min_levels, NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m * m, m * m));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, complete_graph ? m * m : 5, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, complete_graph ? m * m : 5, NULL, complete_graph ? m * m : 4, NULL));
  PetscCall(FillMat(A, m, 0, complete_graph));

  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecSet(b, 1.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCAIR));
  PetscCall(KSPSetFromOptions(ksp));

  for (step = 0; step < nsteps; step++) {

    if (step > 0) {
      // Change the options on a pc that has already been set up
      if (toggle_reuse_sparsity) {
        PetscCall(PCAIRGetReuseSparsity(pc, &reuse));
        PetscCall(PCAIRSetReuseSparsity(pc, (PetscBool)!reuse));
      }

      // Change the values of the pmat, with the same nonzero pattern
      PetscCall(FillMat(A, m, step, complete_graph));
    }

    PetscCall(VecSet(x, 0.0));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(KSPGetIterationNumber(ksp, &its));
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    PetscCheck(reason > 0, PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, "Solve %" PetscInt_FMT " did not converge, reason %d", step, (int)reason);
    PetscCall(PCAIRGetNumLevels(pc, &num_levels));
    PetscCheck(num_levels >= min_levels, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Hierarchy has %" PetscInt_FMT " levels, expected at least %" PetscInt_FMT, num_levels, min_levels);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Solve %" PetscInt_FMT " iterations %" PetscInt_FMT " levels %" PetscInt_FMT "\n", step, its, num_levels));
  }

  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(PetscFinalize());
  return 0;
}
