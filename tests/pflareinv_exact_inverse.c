static char help[] = "Tests SAI/ISAI in PCPFLAREINV recover the exact inverse of a small dense matrix.\n\n";

/*
  When the sparsity pattern of the (I)SAI is full, each row of the (I)SAI is
  the solution of a square dense system with A(J,J)^T, ie the (I)SAI is the
  exact inverse of A. We build a small dense nonsymmetric matrix whose diagonal
  is deliberately small compared to its off-diagonals, so the dense LU in the
  exact (non-iterative) solve has to pivot, and check that the preconditioner
  applied to A x gives back x.

  The matrix is small enough (n <= 40) that the exact dense solve is used
  rather than the iterative one.

    ./pflareinv_exact_inverse -pc_pflareinv_type isai
    ./pflareinv_exact_inverse -pc_pflareinv_type sai
    mpiexec -n 2 ./pflareinv_exact_inverse -pc_pflareinv_type isai
*/
#include <petscksp.h>
#include "pflare.h"

#if defined(PETSC_USE_REAL_SINGLE)
  #define DEFAULT_CHECK_TOL 1e-3
#else
  #define DEFAULT_CHECK_TOL 1e-8
#endif

int main(int argc, char **args)
{
  Mat         A;
  PC          pc;
  Vec         x, b, y;
  PetscInt    n = 12, i, j, global_row_start, global_row_end_plus_one;
  PetscInt    *cols;
  PetscScalar *vals;
  PetscReal   tol = DEFAULT_CHECK_TOL, err, xnorm;
  PetscRandom rand;

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-check_tol", &tol, NULL));

  // Register the PFLARE types
  PCRegister_PFLARE();

  // Dense nonsymmetric matrix stored as aij
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, n, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, n, NULL, n, NULL));
  PetscCall(MatSetUp(A));

  PetscCall(PetscMalloc2(n, &cols, n, &vals));
  PetscCall(MatGetOwnershipRange(A, &global_row_start, &global_row_end_plus_one));
  for (i = global_row_start; i < global_row_end_plus_one; i++) {
    for (j = 0; j < n; j++) {
      cols[j] = j;
      // Deterministic pseudo-random off-diagonals of order one and a small
      // diagonal, so partial pivoting has to swap rows
      vals[j] = PetscSinReal((PetscReal)(1 + i * n + j));
      if (i == j) vals[j] = 0.05 * vals[j];
    }
    PetscCall(MatSetValues(A, 1, &i, n, cols, vals, INSERT_VALUES));
  }
  PetscCall(PetscFree2(cols, vals));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  // Full sparsity pattern (the pattern of A), so the (I)SAI is the exact inverse
  PetscCall(PCCreate(PETSC_COMM_WORLD, &pc));
  PetscCall(PCSetOperators(pc, A, A));
  PetscCall(PCSetType(pc, PCPFLAREINV));
  PetscCall(PCPFLAREINVSetType(pc, PFLAREINV_ISAI));
  PetscCall(PCPFLAREINVSetSparsityOrder(pc, 1));
  PetscCall(PCSetFromOptions(pc));
  PetscCall(PCSetUp(pc));

  // Check M A x == x
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecDuplicate(x, &y));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetSeed(rand, 42));
  PetscCall(PetscRandomSeed(rand));
  PetscCall(VecSetRandom(x, rand));
  PetscCall(MatMult(A, x, b));
  PetscCall(PCApply(pc, b, y));
  PetscCall(VecAXPY(y, -1.0, x));
  PetscCall(VecNorm(y, NORM_2, &err));
  PetscCall(VecNorm(x, NORM_2, &xnorm));

  PetscCheck(err / xnorm <= tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, \
             "(I)SAI with a full sparsity pattern is not the exact inverse - relative error %g > %g", \
             (double)(err / xnorm), (double)tol);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  (I)SAI with a full sparsity pattern is the exact inverse\n"));

  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&y));
  PetscCall(PCDestroy(&pc));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}
