static char help[] = "Tests SAI in PCPFLAREINV gives the minimum norm solution when the local least-squares systems are rank deficient.\n\n";

/*
  Each row j of the SAI M solves min ||M(j, J) A(J, I) - eye(j, I)||_2. If A is
  singular the rows A(J, :) may be linearly dependent, so this least-squares
  problem is rank deficient and has to be solved for the minimum norm solution.

  We check small singular 3x3 matrices (in serial), with the SAI having the
  sparsity of A:
   - [2 -1 0; 0 0 0; 0 -1 2] with the zero row stored, so A(J, I) has a zero row
     and is exactly rank deficient
   - [1 3 0; 2 6 0; 0 0 1], where the first two rows are linearly dependent, so
     A(J, I) is only rank deficient up to roundoff in the QR
   - [0 1 1; 1 0 0; 1 0 0] with no diagonal and two identical rows, so for row 0
     there are more unknowns than equations (|J| = 2 > |I| = 1). With no diagonal
     in the sparsity, the rhs eye(j, J) is zero, so the minimum norm solution is
     zero, but the solve has to be done without reading past the end of the rhs
   - [1 0 0; 0 0 1; 0 0 0] with an empty last row, so for row 1 the rows A(J, :)
     have no entries at all (|I| = 0) and the minimum norm solution is zero
  The minimum norm solutions for these are easy to compute by hand.

  We stop at the first failure, as a reference LAPACK given an illegal argument
  in a later case would stop the program with a zero exit code.

    ./pflareinv_sai_rank_deficient
*/
#include <petscksp.h>
#include "pflare.h"

#if defined(PETSC_USE_REAL_SINGLE)
  #define DEFAULT_CHECK_TOL 1e-5
#else
  #define DEFAULT_CHECK_TOL 1e-12
#endif

#define N         3
#define MAX_NNZ   N *N
#define N_CASES   4

typedef struct {
  const char *name;
  // Stored entries of A, including any explicit zeros
  PetscInt    nnz;
  PetscInt    rows[MAX_NNZ], cols[MAX_NNZ];
  PetscScalar vals[MAX_NNZ];
  // Dense copy of the expected SAI
  PetscScalar M_exact[N][N];
} SAICase;

// Builds the SAI of A (with the sparsity of A) and returns the biggest difference to M_exact
static PetscErrorCode check_sai(const SAICase *sai_case, PetscReal *err)
{
  Mat                A, M;
  PC                 pc;
  PetscInt           i, j, ncols;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, N, N, N, N));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, N, NULL));
  PetscCall(MatSetUp(A));
  for (i = 0; i < sai_case->nnz; i++) PetscCall(MatSetValue(A, sai_case->rows[i], sai_case->cols[i], sai_case->vals[i], INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

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
      PetscReal diff = PetscAbsScalar(vals[j] - sai_case->M_exact[i][cols[j]]);
      // Keep any NaN so the check fails
      if (PetscIsNanReal(diff) || diff > *err) *err = diff;
    }
    PetscCall(MatRestoreRow(M, i, &ncols, &cols, &vals));
  }
  if (!(*err < DEFAULT_CHECK_TOL)) PetscCall(MatView(M, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PCDestroy(&pc));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  PetscInt    c;
  PetscReal   err;
  PetscMPIInt size;
  // Entries not listed in M_exact are zero
  const SAICase cases[N_CASES] = {
    {"a zero row",
     7, {0, 0, 1, 1, 1, 2, 2}, {0, 1, 0, 1, 2, 1, 2}, {2.0, -1.0, 0.0, 0.0, 0.0, -1.0, 2.0},
     // Row 0: min ||m0 (2, -1, 0) - (1, 0, 0)||, the entry in the zero column is zero
     // Row 1: min ||m0 (2, -1, 0) + m2 (0, -1, 2) - (0, 1, 0)||
     // Row 2: min ||m2 (0, -1, 2) - (0, 0, 1)||
     {{2.0 / 5.0, 0.0, 0.0}, {-1.0 / 6.0, 0.0, -1.0 / 6.0}, {0.0, 0.0, 2.0 / 5.0}}},
    {"two linearly dependent rows",
     5, {0, 0, 1, 1, 2}, {0, 1, 0, 1, 2}, {1.0, 3.0, 2.0, 6.0, 1.0},
     // Rows 0 and 1: (m0 + 2 m1) (1, 3) ~= e_j, so m0 + 2 m1 = (1, 3) . e_j / 10
     // and the minimum norm solution is (m0, m1) = (m0 + 2 m1) (1, 2) / 5
     {{0.02, 0.04, 0.0}, {0.06, 0.12, 0.0}, {0.0, 0.0, 1.0}}},
    {"no diagonal",
     4, {0, 0, 1, 2}, {1, 2, 0, 0}, {1.0, 1.0, 1.0, 1.0},
     // The rhs is zero for every row
     {{0.0}}},
    {"an empty row",
     2, {0, 1}, {0, 2}, {1.0, 1.0},
     // Row 0: m0 = 1, row 1 has |I| = 0 so is zero, row 2 has |J| = 0 so is empty
     {{1.0, 0.0, 0.0}, {0.0}, {0.0}}},
  };

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This test is serial only");

  // Register the PFLARE types
  PCRegister_PFLARE();

  for (c = 0; c < N_CASES; c++) {
    PetscCall(check_sai(&cases[c], &err));
    if (!(err < DEFAULT_CHECK_TOL)) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "FAIL: SAI with %s differs from the minimum norm solution by %g\n", cases[c].name, (double)err));
      PetscCall(PetscFinalize());
      return 1;
    }
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "OK: SAI matches the minimum norm solutions\n"));

  PetscCall(PetscFinalize());
  return 0;
}
