static char help[] = "Tests KSPSetReusePreconditioner with PCAIR freezes the hierarchy \n\
when the values of the same pmat object change between solves.\n\n";

#include <petscksp.h>
#include "pflare.h"

int main(int argc, char **args)
{
  Mat       A;
  Vec       x, b;
  KSP       ksp;
  PC        pc;
  PetscInt  n = 1000, Istart, Iend, i;
  PetscInt  its_first, its_frozen, its_rebuilt;
  PetscReal shift = 10.0;
  PetscReal gc_first, cc_first, gc_frozen, cc_frozen, gc_rebuilt, cc_rebuilt;

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  // Register the pflare types
  PCRegister_PFLARE();

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-shift", &shift, NULL));

  // Nonsymmetric 1D advection-diffusion tridiagonal matrix
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, 3, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 3, NULL, 2, NULL));
  PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
  for (i = Istart; i < Iend; i++) {
    PetscScalar v[3]    = {-1.5, 3.0, -0.5};
    PetscInt    cols[3] = {i - 1, i, i + 1};
    if (i == 0) PetscCall(MatSetValues(A, 1, &i, 2, &cols[1], &v[1], INSERT_VALUES));
    else if (i == n - 1) PetscCall(MatSetValues(A, 1, &i, 2, cols, v, INSERT_VALUES));
    else PetscCall(MatSetValues(A, 1, &i, 3, cols, v, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecSet(b, 1.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  // A is both the amat and the pmat
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCAIR));
  PetscCall(KSPSetFromOptions(ksp));

  // ~~~~~~~~~~~~~~~~~~
  // First solve - builds the hierarchy
  // ~~~~~~~~~~~~~~~~~~
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(KSPGetIterationNumber(ksp, &its_first));
  PetscCall(PCAIRGetGridComplexity(pc, &gc_first));
  PetscCall(PCAIRGetCycleComplexity(pc, &cc_first));

  // ~~~~~~~~~~~~~~~~~~
  // Freeze the preconditioner, then change the values of the
  // same pmat object (sparsity pattern unchanged) and solve again
  // No new hierarchy should be built
  // ~~~~~~~~~~~~~~~~~~
  PetscCall(KSPSetReusePreconditioner(ksp, PETSC_TRUE));
  PetscCall(MatShift(A, shift));
  PetscCall(VecSet(x, 0.0));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(KSPGetIterationNumber(ksp, &its_frozen));
  PetscCall(PCAIRGetGridComplexity(pc, &gc_frozen));
  PetscCall(PCAIRGetCycleComplexity(pc, &cc_frozen));

  // The frozen hierarchy must be untouched, so the complexities are
  // bit-identical to the first solve
  PetscCheck(gc_frozen == gc_first && cc_frozen == cc_first, PETSC_COMM_WORLD, PETSC_ERR_PLIB, \
     "Reused preconditioner was rebuilt - complexities changed (grid %f -> %f, cycle %f -> %f)", \
     (double)gc_first, (double)gc_frozen, (double)cc_first, (double)cc_frozen);

  // ~~~~~~~~~~~~~~~~~~
  // Unfreeze and solve the same shifted system again - this must
  // trigger a rebuild on the shifted matrix
  // ~~~~~~~~~~~~~~~~~~
  PetscCall(KSPSetReusePreconditioner(ksp, PETSC_FALSE));
  PetscCall(VecSet(x, 0.0));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(KSPGetIterationNumber(ksp, &its_rebuilt));
  PetscCall(PCAIRGetGridComplexity(pc, &gc_rebuilt));
  PetscCall(PCAIRGetCycleComplexity(pc, &cc_rebuilt));

  // The rebuilt hierarchy on the (strongly diagonally dominant) shifted
  // matrix differs from the original one - this also proves the frozen
  // complexity check above is not vacuous
  PetscCheck(gc_rebuilt != gc_first || cc_rebuilt != cc_first, PETSC_COMM_WORLD, PETSC_ERR_PLIB, \
     "Rebuild after unfreezing did not change the hierarchy - test cannot detect reuse");

  // The frozen preconditioner was built for the unshifted matrix so it
  // must take more iterations on the shifted system than the rebuilt one
  PetscCheck(its_frozen > its_rebuilt, PETSC_COMM_WORLD, PETSC_ERR_PLIB, \
     "Frozen preconditioner iteration count (%" PetscInt_FMT ") does not exceed rebuilt (%" PetscInt_FMT ")", \
     its_frozen, its_rebuilt);

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Iterations: first %" PetscInt_FMT ", frozen %" PetscInt_FMT ", rebuilt %" PetscInt_FMT "\n", \
     its_first, its_frozen, its_rebuilt));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(PetscFinalize());
  return 0;
}
