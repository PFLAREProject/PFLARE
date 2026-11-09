static char help[] = "Tests PCPFLAREINVSetPolyCoeffs on a PCPFLAREINV that is already set up.\n\
\n\
Setting coefficients doesn't trigger a rebuild, so the inverse that is already built\n\
(assembled or matrix-free) must keep applying the coefficients it was built with until\n\
the next PCSetUp, which with PCPFLAREINVSetReusePolyCoeffs must then use the ones set.\n\
\n\
  -n : number of unknowns (default 60)\n\n";

#include <petscksp.h>
#include "pflare.h"

// Fills (or refills) a nonsymmetric tridiagonal matrix
// The nonzero pattern never changes, the values depend on step
static PetscErrorCode FillMat(Mat A, PetscInt n, PetscInt step)
{
  PetscInt Istart, Iend, i;

  PetscFunctionBegin;
  PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
  for (i = Istart; i < Iend; i++) {
    PetscReal   r       = (PetscReal)((i * 7919 + step * 104729) % 1000) / 1000.0;
    PetscScalar v[3]    = {-1.5 - 0.3 * r, 3.0 + r, -0.5};
    PetscInt    cols[3] = {i - 1, i, i + 1};
    if (i == 0) PetscCall(MatSetValues(A, 1, &i, 2, &cols[1], &v[1], INSERT_VALUES));
    else if (i == n - 1) PetscCall(MatSetValues(A, 1, &i, 2, cols, v, INSERT_VALUES));
    else PetscCall(MatSetValues(A, 1, &i, 3, cols, v, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat        A;
  Vec        x, b, y_before, y_after;
  KSP        ksp;
  PC         pc;
  PetscInt   n = 60, rows, cols, rows_new, cols_new, i;
  PetscReal *coeffs, *saved, *scratch, diff, expected;
  PCPFLAREINVType type;

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  // Register the pflare types
  PCRegister_PFLARE();

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, 3, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 3, NULL, 2, NULL));
  PetscCall(FillMat(A, n, 0));

  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecDuplicate(x, &y_before));
  PetscCall(VecDuplicate(x, &y_after));
  PetscCall(VecSet(b, 1.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCPFLAREINV));
  PetscCall(KSPSetFromOptions(ksp));

  // ~~~~~~~~~~~~~~~~~~
  // First setup computes the coefficients, the second (same nonzero pattern,
  // new values) rebuilds the inverse from the stored ones
  // ~~~~~~~~~~~~~~~~~~
  PetscCall(PCPFLAREINVSetReusePolyCoeffs(pc, PETSC_TRUE));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(FillMat(A, n, 1));
  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(PCPFLAREINVGetPolyCoeffs(pc, &coeffs, &rows, &cols));
  PetscCheck(coeffs && rows > 0 && cols > 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "No coefficients stored - this test needs a polynomial inverse type");
  PetscCall(PetscMalloc1(rows * cols, &saved));
  for (i = 0; i < rows * cols; i++) saved[i] = 0.5 * coeffs[i];

  // ~~~~~~~~~~~~~~~~~~
  // Set different coefficients - the inverse that is already built must be untouched
  // The allocations in between are there to reuse the memory of the old stored
  // coefficients, in case the inverse is still (wrongly) pointing at it
  // ~~~~~~~~~~~~~~~~~~
  PetscCall(PCApply(pc, b, y_before));
  PetscCall(PCPFLAREINVSetPolyCoeffs(pc, saved, rows, cols));
  for (i = 0; i < 8; i++) {
    scratch = (PetscReal *)malloc((size_t)(rows * cols) * sizeof(PetscReal));
    PetscCheck(scratch, PETSC_COMM_SELF, PETSC_ERR_MEM, "malloc failed");
    memset(scratch, 0x7f, (size_t)(rows * cols) * sizeof(PetscReal));
    free(scratch);
  }
  PetscCall(PCApply(pc, b, y_after));
  PetscCall(VecAXPY(y_after, -1.0, y_before));
  PetscCall(VecNorm(y_after, NORM_INFINITY, &diff));
  PetscCheck(diff == 0.0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PCApply changed after PCPFLAREINVSetPolyCoeffs without a setup, difference %g", (double)diff);

  // And the same for the transposed apply
  PetscCall(PCApplyTranspose(pc, b, y_before));
  PetscCall(PCPFLAREINVSetPolyCoeffs(pc, saved, rows, cols));
  PetscCall(PCApplyTranspose(pc, b, y_after));
  PetscCall(VecAXPY(y_after, -1.0, y_before));
  PetscCall(VecNorm(y_after, NORM_INFINITY, &diff));
  PetscCheck(diff == 0.0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PCApplyTranspose changed after PCPFLAREINVSetPolyCoeffs without a setup, difference %g", (double)diff);

  // ~~~~~~~~~~~~~~~~~~
  // The next setup must keep the coefficients we set
  // ~~~~~~~~~~~~~~~~~~
  PetscCall(FillMat(A, n, 2));
  PetscCall(PCSetUp(pc));
  PetscCall(PCPFLAREINVGetPolyCoeffs(pc, &coeffs, &rows_new, &cols_new));
  PetscCheck(rows_new == rows && cols_new == cols, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Stored coefficients changed size in a setup with reuse");
  PetscCall(PCPFLAREINVGetType(pc, &type));
  for (i = 0; i < rows * cols; i++) {
    // The coefficients of a Neumann polynomial are always one
    expected = type == PFLAREINV_NEUMANN ? 1.0 : saved[i];
    PetscCheck(coeffs[i] == expected, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Coefficient %" PetscInt_FMT " was not reused, %g vs %g", i, (double)coeffs[i], (double)expected);
  }
  PetscCall(PCApply(pc, b, y_after));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "OK - %" PetscInt_FMT " x %" PetscInt_FMT " coefficients\n", rows, cols));

  PetscCall(PetscFree(saved));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&y_before));
  PetscCall(VecDestroy(&y_after));
  PetscCall(PetscFinalize());
  return 0;
}
