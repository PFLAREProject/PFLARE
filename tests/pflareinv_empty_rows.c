static char help[] = "Tests the assembled fixed sparsity polynomial inverses in PCPFLAREINV on a matrix with empty rows.\n\n";

/*
  The matrix is block diagonal with 3x3 blocks

     [  2 -1  1 ]
     [ -1  2  1 ]
     [  .  .  . ]

  where the last row of each block has no stored entries at all, but its column is
  used by the other two rows. The fixed sparsity matrix powers therefore have to
  look up a row with nothing in it.

  The powers of this matrix all have the sparsity of the matrix itself, so the
  assembled inverse with a sparsity order of one is exactly the polynomial q(A),
  rather than an approximation to it. For the power and Arnoldi basis we check
  PCApply against a Horner evaluation of q(A) x with the stored coefficients.
  For the Newton basis we only check that the inverse can be built and applied.

    ./pflareinv_empty_rows -pc_pflareinv_type arnoldi
    ./pflareinv_empty_rows -pc_pflareinv_type newton
    mpiexec -n 2 ./pflareinv_empty_rows -pc_pflareinv_type power
*/
#include <petscksp.h>
#include "pflare.h"

#if defined(PETSC_USE_REAL_SINGLE)
  #define DEFAULT_CHECK_TOL 1e-4
#else
  #define DEFAULT_CHECK_TOL 1e-10
#endif

int main(int argc, char **args)
{
  Mat         A;
  PC          pc;
  Vec         x, y, y_ref, temp;
  PetscInt    n_blocks = 10, n, i, global_row_start, global_row_end_plus_one, block_start;
  PetscInt    rows, cols, k;
  PetscReal  *coeffs, tol = DEFAULT_CHECK_TOL, err, norm;
  PetscRandom rand;

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_blocks", &n_blocks, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-check_tol", &tol, NULL));
  n = 3 * n_blocks;

  // Register the PFLARE types
  PCRegister_PFLARE();

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, 3, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 3, NULL, 2, NULL));
  PetscCall(MatGetOwnershipRange(A, &global_row_start, &global_row_end_plus_one));
  for (i = global_row_start; i < global_row_end_plus_one; i++) {
    PetscInt    block_cols[3];
    PetscScalar first_row[3] = {2.0, -1.0, 1.0}, second_row[3] = {-1.0, 2.0, 1.0};

    block_start   = 3 * (i / 3);
    block_cols[0] = block_start;
    block_cols[1] = block_start + 1;
    block_cols[2] = block_start + 2;
    if (i == block_start) PetscCall(MatSetValues(A, 1, &i, 3, block_cols, first_row, INSERT_VALUES));
    else if (i == block_start + 1) PetscCall(MatSetValues(A, 1, &i, 3, block_cols, second_row, INSERT_VALUES));
    // The third row of each block is left empty
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  // Assembled inverse with the sparsity of A
  PetscCall(PCCreate(PETSC_COMM_WORLD, &pc));
  PetscCall(PCSetOperators(pc, A, A));
  PetscCall(PCSetType(pc, PCPFLAREINV));
  PetscCall(PCPFLAREINVSetType(pc, PFLAREINV_ARNOLDI));
  PetscCall(PCPFLAREINVSetPolyOrder(pc, 4));
  PetscCall(PCPFLAREINVSetSparsityOrder(pc, 1));
  PetscCall(PCSetFromOptions(pc));
  PetscCall(PCSetUp(pc));

  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(VecDuplicate(x, &y_ref));
  PetscCall(VecDuplicate(x, &temp));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetSeed(rand, 42));
  PetscCall(PetscRandomSeed(rand));
  PetscCall(VecSetRandom(x, rand));

  PetscCall(PCApply(pc, x, y));
  PetscCall(VecNorm(y, NORM_2, &norm));
  PetscCheck(!PetscIsInfOrNanReal(norm), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PCApply gave a non-finite result");

  PetscCall(PCPFLAREINVGetPolyCoeffs(pc, &coeffs, &rows, &cols));
  if (cols == 1) {
    // Horner, y_ref = (c_0 + c_1 A + c_2 A^2 + ...) x
    PetscCall(VecCopy(x, y_ref));
    PetscCall(VecScale(y_ref, coeffs[rows - 1]));
    for (k = rows - 2; k >= 0; k--) {
      PetscCall(MatMult(A, y_ref, temp));
      PetscCall(VecWAXPY(y_ref, coeffs[k], x, temp));
    }
    PetscCall(VecAXPY(y_ref, -1.0, y));
    PetscCall(VecNorm(y_ref, NORM_2, &err));
    PetscCheck(err <= tol * norm, PETSC_COMM_WORLD, PETSC_ERR_PLIB, \
               "Assembled inverse is not the polynomial q(A) - relative error %g > %g", \
               (double)(err / norm), (double)tol);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Assembled inverse matches the polynomial on a matrix with empty rows\n"));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Assembled inverse built and applied on a matrix with empty rows\n"));
  }

  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&y_ref));
  PetscCall(VecDestroy(&temp));
  PetscCall(PCDestroy(&pc));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}
