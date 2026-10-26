static char help[] = "Tests SAI in PCPFLAREINV gives the minimum norm solution when the local least-squares systems are rank deficient.\n\n";

/*
  Each row j of the SAI M solves min ||M(j, J) A(J, I) - eye(j, I)||_2. If A is
  singular the rows A(J, :) may be linearly dependent, so this least-squares
  problem is rank deficient and has to be solved for the minimum norm solution.

  We check two small singular matrices (in serial):
   - [2 -1 0; 0 0 0; 0 -1 2] with the zero row stored, so A(J, I) has a zero row
   - [0 1 1; 1 0 0; 1 0 0] with no diagonal and two identical rows, so for row 0
     there are more unknowns than equations (|J| = 2 > |I| = 1). With no diagonal
     in the sparsity, the rhs eye(j, J) is zero, so the minimum norm solution is
     zero, but the solve has to be done without reading past the end of the rhs
  The minimum norm solutions for these are easy to compute by hand.

    ./pflareinv_sai_rank_deficient
*/
#include <petscksp.h>
#include "pflare.h"

#if defined(PETSC_USE_REAL_SINGLE)
  #define DEFAULT_CHECK_TOL 1e-5
#else
  #define DEFAULT_CHECK_TOL 1e-12
#endif

#define N 3

// Builds the SAI of A (with the sparsity of A) and returns the biggest difference to M_exact
static PetscErrorCode check_sai(Mat A, PetscScalar M_exact[N][N], PetscReal *err)
{
  Mat                M;
  PC                 pc;
  PetscInt           i, j, ncols;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBeginUser;
  PetscCall(PCCreate(PETSC_COMM_WORLD, &pc));
  PetscCall(PCSetOperators(pc, A, A));
  PetscCall(PCSetType(pc, PCPFLAREINV));
  PetscCall(PCPFLAREINVSetType(pc, PFLAREINV_SAI));
  PetscCall(PCPFLAREINVSetSparsityOrder(pc, 1));
  PetscCall(PCSetFromOptions(pc));
  PetscCall(PCSetUp(pc));
  PetscCall(PCPFLAREINVGetInverseMat(pc, &M));

  *err = 0.0;
  for (i = 0; i < N; i++) {
    PetscCall(MatGetRow(M, i, &ncols, &cols, &vals));
    for (j = 0; j < ncols; j++) {
      PetscReal diff = PetscAbsScalar(vals[j] - M_exact[i][cols[j]]);
      // Keep any NaN so the check fails
      if (PetscIsNanReal(diff) || diff > *err) *err = diff;
    }
    PetscCall(MatRestoreRow(M, i, &ncols, &cols, &vals));
  }
  if (!(*err < DEFAULT_CHECK_TOL)) PetscCall(MatView(M, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PCDestroy(&pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat         A;
  PetscInt    j;
  PetscReal   err;
  PetscMPIInt size;
  PetscBool   pass = PETSC_TRUE;
  // Dense copies of the expected SAIs
  PetscScalar M_zero_row[N][N] = {{0.0}}, M_no_diag[N][N] = {{0.0}};

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This test is serial only");

  // Register the PFLARE types
  PCRegister_PFLARE();

  // ~~~~~~~~~~~~
  // [2 -1 0; 0 0 0; 0 -1 2] with the zero row stored explicitly
  // ~~~~~~~~~~~~
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, N, N, N, N));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, N, NULL));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetValue(A, 0, 0, 2.0, INSERT_VALUES));
  PetscCall(MatSetValue(A, 0, 1, -1.0, INSERT_VALUES));
  for (j = 0; j < N; j++) PetscCall(MatSetValue(A, 1, j, 0.0, INSERT_VALUES));
  PetscCall(MatSetValue(A, 2, 1, -1.0, INSERT_VALUES));
  PetscCall(MatSetValue(A, 2, 2, 2.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  // Minimum norm solutions, the entry in the zero column is always zero
  // Row 0: min ||m0 (2, -1, 0) - (1, 0, 0)||
  M_zero_row[0][0] = 2.0 / 5.0;
  // Row 1: min ||m0 (2, -1, 0) + m2 (0, -1, 2) - (0, 1, 0)||
  M_zero_row[1][0] = -1.0 / 6.0;
  M_zero_row[1][2] = -1.0 / 6.0;
  // Row 2: min ||m2 (0, -1, 2) - (0, 0, 1)||
  M_zero_row[2][2] = 2.0 / 5.0;

  PetscCall(check_sai(A, M_zero_row, &err));
  PetscCall(MatDestroy(&A));
  if (!(err < DEFAULT_CHECK_TOL)) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "FAIL: SAI with a zero row differs from the minimum norm solution by %g\n", (double)err));
    // Stop now, a reference LAPACK given an illegal argument below would
    // stop the program with a zero exit code
    PetscCall(PetscFinalize());
    return 1;
  }

  // ~~~~~~~~~~~~
  // [0 1 1; 1 0 0; 1 0 0] with no diagonal, the SAI is zero
  // ~~~~~~~~~~~~
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, N, N, N, N));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, N, NULL));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetValue(A, 0, 1, 1.0, INSERT_VALUES));
  PetscCall(MatSetValue(A, 0, 2, 1.0, INSERT_VALUES));
  PetscCall(MatSetValue(A, 1, 0, 1.0, INSERT_VALUES));
  PetscCall(MatSetValue(A, 2, 0, 1.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(check_sai(A, M_no_diag, &err));
  PetscCall(MatDestroy(&A));
  if (!(err < DEFAULT_CHECK_TOL)) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "FAIL: SAI with no diagonal differs from the minimum norm solution by %g\n", (double)err));
    pass = PETSC_FALSE;
  }

  if (pass) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "OK: SAI matches the minimum norm solutions\n"));

  PetscCall(PetscFinalize());
  return pass ? 0 : 1;
}
