static char help[] = "Tests extra roots are added to purely imaginary Newton polynomial roots.\n\n";

/*
  The GMRES polynomial in the Newton basis adds extra copies of roots with a
  large product of factors, prod_j |1 - theta_j/theta_i|, for stability
  (Loe & Morgan). A root must only be treated as zero (and skipped) when both its
  real and imaginary parts are zero, which is what the polynomial apply does.

  Here the operator is skew-symmetric and block diagonal, with 2x2 blocks
  [0 y_k; -y_k 0], so its eigenvalues are +-i y_k. The harmonic Ritz values (ie
  the Newton roots) then have a numerically zero real part but a nonzero
  imaginary part. The y_k are log-spaced over -spread orders of magnitude, so at
  high order the largest roots have a large product of factors and extra roots
  must be added. Without them the polynomial is (1 - A p(A)) ~ 1e8 on the largest
  eigenvalues and right-preconditioned GMRES does not converge in 100 iterations.

  We check that:
  - some of the roots are purely imaginary (otherwise this test proves nothing)
  - extra roots have been added
  - the solve converges

    ./newton_imag_roots
    mpiexec -n 2 ./newton_imag_roots
*/
#include <petscksp.h>
#include "pflare.h"

int main(int argc, char **argv)
{
  Mat                A;
  Vec                x, b;
  KSP                ksp;
  PC                 pc;
  PetscInt           n = 100, poly_order = 40, Istart, Iend, i, rows, cols, n_imag = 0;
  PetscReal         *coeffs, tol_zero = 1e-12, spread = 3.0;
  PCPFLAREINVType    type;
  KSPConvergedReason reason;
  PetscInt           its;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PCRegister_PFLARE();

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-poly_order", &poly_order, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-spread", &spread, NULL));
  PetscCheck(n % 2 == 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "n must be even so the skew operator is nonsingular");

  // Skew-symmetric block diagonal operator, with 2x2 blocks [0 y_k; -y_k 0]
  // The eigenvalues are +-i y_k, with y_k log-spaced over -spread orders of magnitude
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, 1, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 1, NULL, 1, NULL));
  PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
  for (i = Istart; i < Iend; i++) {
    PetscInt    k = i / 2, col;
    PetscScalar val;
    val = PetscPowReal(10.0, spread * (PetscReal)k / (PetscReal)(n / 2 - 1));
    if (i % 2 == 0) {
      col = i + 1;
    } else {
      col = i - 1;
      val = -val;
    }
    PetscCall(MatSetValues(A, 1, &i, 1, &col, &val, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecSet(x, 1.0));
  PetscCall(MatMult(A, x, b));
  PetscCall(VecSet(x, 0.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPGMRES));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCPFLAREINV));
  PetscCall(PCPFLAREINVSetType(pc, PFLAREINV_NEWTON));
  PetscCall(PCPFLAREINVSetPolyOrder(pc, poly_order));
  PetscCall(PCPFLAREINVSetMatrixFree(pc, PETSC_TRUE));
  // Right preconditioning so the iteration count reflects the true residual, the
  // preconditioned residual is misleading when the polynomial is poorly behaved
  PetscCall(KSPSetPCSide(ksp, PC_RIGHT));
  PetscCall(KSPGMRESSetRestart(ksp, 100));
  PetscCall(KSPSetTolerances(ksp, 1e-8, PETSC_CURRENT, PETSC_CURRENT, 100));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));

  PetscCall(PCPFLAREINVGetType(pc, &type));
  PetscCheck(type == PFLAREINV_NEWTON, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Test requires -pc_pflareinv_type newton");
  PetscCall(PCPFLAREINVGetPolyOrder(pc, &poly_order));
  PetscCall(PCPFLAREINVGetPolyCoeffs(pc, &coeffs, &rows, &cols));
  PetscCheck(cols == 2, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Expected a Newton basis polynomial");

  // Count the purely imaginary roots - column-major, real parts then imaginary
  for (i = 0; i < rows; i++) {
    if (PetscAbsReal(coeffs[i]) < tol_zero && PetscAbsReal(coeffs[rows + i]) > tol_zero) n_imag++;
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Poly order %" PetscInt_FMT ", roots %" PetscInt_FMT ", purely imaginary roots %" PetscInt_FMT "\n", poly_order, rows, n_imag));

  PetscCheck(n_imag > 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Expected purely imaginary roots, test proves nothing");
  PetscCheck(rows > poly_order + 1, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "No extra roots added for the purely imaginary roots");

  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Iterations %" PetscInt_FMT "\n", its));
  PetscCheck(reason > 0, PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, "Solve did not converge");

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
  return 0;
}
