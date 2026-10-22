static char help[] = "Tests assembled Newton polynomial inverses skip numerically zero roots like the matrix-free apply.\n\n";

/*
  The matrix-free apply of a GMRES polynomial in the Newton basis skips any root
  that is numerically zero (|theta| < tol for a real root, |theta|^2 < tol for a
  complex pair). Here we set Newton roots containing zero and tiny roots into
  PCPFLAREINV with PCPFLAREINVSetPolyCoeffs and reuse on, and check the assembled
  inverse (with unconstrained sparsity) gives the same result as the
  matrix-free apply. This covers the 0th order, 1st order and higher order
  assembled builds.

    ./newton_zero_roots
    mpiexec -n 2 ./newton_zero_roots
*/
#include <petscksp.h>
#include "pflare.h"

#if defined(PETSC_USE_REAL_SINGLE)
  #define DEFAULT_CHECK_TOL 1e-5
  #define TINY              1e-7
#else
  #define DEFAULT_CHECK_TOL 1e-10
  #define TINY              1e-14
#endif

#define MAX_ROOTS 4
#define N_ROOT_SETS 9

/*
   Writes a 1D upwind advection-diffusion stencil into A
*/
static PetscErrorCode SetOperatorValues(Mat A, PetscInt n)
{
  PetscInt    i, global_row_start, global_row_end_plus_one, cols[3], n_cols;
  PetscScalar vals[3];

  PetscFunctionBeginUser;
  PetscCall(MatGetOwnershipRange(A, &global_row_start, &global_row_end_plus_one));

  for (i = global_row_start; i < global_row_end_plus_one; i++) {
    n_cols = 0;
    if (i > 0) {
      cols[n_cols] = i - 1;
      vals[n_cols] = -0.3;
      n_cols++;
    }
    cols[n_cols] = i;
    vals[n_cols] = 1.0;
    n_cols++;
    if (i < n - 1) {
      cols[n_cols] = i + 1;
      vals[n_cols] = -0.1;
      n_cols++;
    }
    PetscCall(MatSetValues(A, 1, &i, n_cols, cols, vals, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Builds a Newton PCPFLAREINV of the given order and sparsity order, then
   rebuilds it reusing the given roots (column-major, real parts then imaginary parts)
*/
static PetscErrorCode CreatePC(Mat A, PetscInt n, PetscInt n_roots, PetscInt sparsity_order, PetscBool matrix_free, PetscReal *roots, PC *pc)
{
  PetscFunctionBeginUser;
  PetscCall(PCCreate(PETSC_COMM_WORLD, pc));
  PetscCall(PCSetType(*pc, PCPFLAREINV));
  PetscCall(PCPFLAREINVSetType(*pc, PFLAREINV_NEWTON_NO_EXTRA));
  PetscCall(PCPFLAREINVSetPolyOrder(*pc, n_roots - 1));
  PetscCall(PCPFLAREINVSetSparsityOrder(*pc, sparsity_order));
  PetscCall(PCPFLAREINVSetMatrixFree(*pc, matrix_free));
  PetscCall(PCSetOperators(*pc, A, A));
  PetscCall(PCSetFromOptions(*pc));
  // First setup computes the roots (and the sparsity pattern of the assembled inverse)
  PetscCall(PCSetUp(*pc));

  // Set our roots and have the next setup reuse them
  PetscCall(PCPFLAREINVSetPolyCoeffs(*pc, roots, n_roots, 2));
  PetscCall(PCPFLAREINVSetReusePolyCoeffs(*pc, PETSC_TRUE));
  // Same values and nonzero pattern, but a new state so the setup is redone
  PetscCall(SetOperatorValues(A, n));
  PetscCall(PCSetOperators(*pc, A, A));
  PetscCall(PCSetUp(*pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat         A;
  Vec         x, y, ref, diff;
  PC          pc, pc_mf;
  PetscRandom rand;
  PetscInt    n = 20, set, i;
  PetscReal   tol = DEFAULT_CHECK_TOL, diff_norm, ref_norm;
  PetscReal   roots[2 * MAX_ROOTS];
  // Number of roots, then the real and imaginary parts of each root
  // The complex conjugate pairs are next to each other
  const PetscInt  set_n_roots[N_ROOT_SETS] = {1, 1, 2, 2, 2, 2, 2, 4, 4};
  const PetscReal set_roots[N_ROOT_SETS][MAX_ROOTS][2] = {
    {{0.0, 0.0}},                                                  // 0th order, zero root
    {{TINY, 0.0}},                                                 // 0th order, tiny root
    {{0.0, 0.0}, {0.5, 0.0}},                                      // 1st order, zero first root
    {{TINY, 0.0}, {0.5, 0.0}},                                     // 1st order, tiny first root
    {{0.8, 0.0}, {0.0, 0.0}},                                      // 1st order, zero last root
    {{0.0, 0.0}, {0.0, 0.0}},                                      // 1st order, all zero
    {{TINY, TINY}, {TINY, -TINY}},                                 // 1st order, tiny complex pair
    {{0.0, 0.0}, {0.5, 0.0}, {0.7, 0.0}, {0.9, 0.0}},              // 3rd order, zero first root
    {{TINY, TINY}, {TINY, -TINY}, {0.5, 0.0}, {0.9, 0.0}},         // 3rd order, tiny complex first pair
  };

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PCRegister_PFLARE();
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-tol", &tol, NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, 3, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 3, NULL, 2, NULL));
  PetscCall(MatSetUp(A));
  PetscCall(SetOperatorValues(A, n));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(VecDuplicate(x, &ref));
  PetscCall(VecDuplicate(x, &diff));

  for (set = 0; set < N_ROOT_SETS; set++) {
    for (i = 0; i < set_n_roots[set]; i++) {
      roots[i]                    = set_roots[set][i][0];
      roots[set_n_roots[set] + i] = set_roots[set][i][1];
    }

    // Assembled with unconstrained sparsity and matrix-free
    PetscCall(CreatePC(A, n, set_n_roots[set], set_n_roots[set] - 1, PETSC_FALSE, roots, &pc));
    PetscCall(CreatePC(A, n, set_n_roots[set], set_n_roots[set] - 1, PETSC_TRUE, roots, &pc_mf));

    PetscCall(VecSetRandom(x, rand));
    PetscCall(PCApply(pc, x, y));
    PetscCall(PCApply(pc_mf, x, ref));

    PetscCall(VecWAXPY(diff, -1.0, ref, y));
    PetscCall(VecNorm(diff, NORM_2, &diff_norm));
    PetscCall(VecNorm(ref, NORM_2, &ref_norm));
    if (ref_norm < 1.0) ref_norm = 1.0;
    PetscCheck(diff_norm / ref_norm <= tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, \
               "Root set %" PetscInt_FMT ": the assembled inverse does not match the matrix-free inverse, relative difference %g > %g", \
               set, (double)(diff_norm / ref_norm), (double)tol);

    PetscCall(PCDestroy(&pc));
    PetscCall(PCDestroy(&pc_mf));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  %d root sets with zero roots match between assembled and matrix-free\n", N_ROOT_SETS));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&ref));
  PetscCall(VecDestroy(&diff));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}
