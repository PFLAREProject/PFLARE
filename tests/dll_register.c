static char help[] = "Checks the dynamic library registration routine registers every PFLARE PC type.\n\n";

/*
  When PETSc loads libpetscpflare as a shared library (e.g., with --download-pflare)
  it calls PetscDLLibraryRegister_petscpflare, and the user code never calls
  PCRegister_PFLARE. Here we call the registration routine directly instead of
  PCRegister_PFLARE, then check each PFLARE PC type can be created and used in a solve.
  With -register_twice we also call PCRegister_PFLARE afterwards, to check registering
  the types twice is harmless.
*/

#include <petscksp.h>
#include "pflare.h"

// Not in pflare.h, as petsc calls it when loading the shared library
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscpflare(void);

int main(int argc, char **args)
{
  Mat                A;
  Vec                x, b;
  KSP                ksp;
  PC                 pc;
  PetscInt           i, n = 100, row_start, row_end, col[3], k;
  PetscScalar        v[3];
  PetscBool          register_twice = PETSC_FALSE, match;
  const char        *pc_types[]     = {PCAIR, PCPFLAREINV};
  KSPConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-register_twice", &register_twice, NULL));

  // Register the PFLARE PC types the same way petsc does when loading the shared library
  PetscCall(PetscDLLibraryRegister_petscpflare());
  if (register_twice) PCRegister_PFLARE();

  // Diagonally dominant non-symmetric tridiagonal operator
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A, &row_start, &row_end));
  for (i = row_start; i < row_end; i++) {
    k = 0;
    if (i > 0) {
      col[k] = i - 1;
      v[k++] = -1.0;
    }
    col[k] = i;
    v[k++] = 3.0;
    if (i < n - 1) {
      col[k] = i + 1;
      v[k++] = -0.5;
    }
    PetscCall(MatSetValues(A, 1, &i, k, col, v, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecSet(b, 1.0));

  for (i = 0; i < 2; i++) {
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetType(ksp, KSPGMRES));
    PetscCall(KSPSetTolerances(ksp, 1e-8, PETSC_CURRENT, PETSC_CURRENT, 100));
    PetscCall(KSPGetPC(ksp, &pc));
    // Errors with an unknown type if the registration routine missed this type
    PetscCall(PCSetType(pc, pc_types[i]));
    PetscCall(PetscObjectTypeCompare((PetscObject)pc, pc_types[i], &match));
    PetscCheck(match, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PC type is not %s", pc_types[i]);
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(VecSet(x, 0.0));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    PetscCheck(reason > 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Solve with PC type %s did not converge", pc_types[i]);
    PetscCall(KSPDestroy(&ksp));
  }

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}
