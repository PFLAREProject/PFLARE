static char help[] = "Tests saving and restoring GMRES polynomial coefficients via the C API.\n\
Mirrors tests/ex6f_getcoeffs.F90 using the C interface.\n\
Works with both PCAIR (multi-level) and PCPFLAREINV (single-level); the PC\n\
type is chosen via -pc_type on the command line.\n\
\n\
Three solves are performed:\n\
  Solve 1 - original system  (poly coefficients saved afterwards)\n\
  Solve 2 - perturbed system (forces a fresh setup)\n\
  Solve 3 - original system again, with saved coefficients restored\n\
\n\
The test passes when the residual norm of solve 3 equals that of solve 1\n\
to within a relative tolerance of 1e-8.\n\n";

#include <petscksp.h>
#include <string.h>
#include <stdlib.h>
#include "pflare.h"

int main(int argc, char **args)
{
  Vec                x, b, u;
  Mat                A;
  KSP                ksp;
  PC                 pc;
  PetscInt           i, j, II, JJ, m, n;
  PetscInt           Istart, Iend;
  PetscInt           nsteps, count;
  PetscBool          flg;
  PetscScalar        v;
  PetscReal          norm_first = 0.0, norm_third = 0.0;
  PCType             pctype;
  PetscBool          is_air, is_pflareinv;
  PetscMPIInt        rank;
  KSPConvergedReason reason;

  /* Storage for saved coefficients (PCAIR) */
  PetscInt    num_levels = 0;
  PetscReal **saved_coeffs = NULL;
  PetscInt   *saved_rows   = NULL;
  PetscInt   *saved_cols   = NULL;

  /* Storage for saved coefficients (PCPFLAREINV) */
  PetscReal  *saved_pflareinv_coeffs = NULL;
  PetscInt    saved_pflareinv_rows   = 0;
  PetscInt    saved_pflareinv_cols   = 0;

  /* -no_power: skip power-basis PCPFLAREINV (for Intel MPI CI) */
  PetscBool   no_power = PETSC_FALSE;
  PetscBool   skip     = PETSC_FALSE;

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  m      = 5;
  n      = 5;
  nsteps = 3;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  /* -----------------------------------------------------------------------
   * Build 2-D five-point Laplacian (same matrix as ex6f_getcoeffs.F90)
   * --------------------------------------------------------------------- */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m * n, m * n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
  for (II = Istart; II < Iend; II++) {
    v = -1.0;
    i = II / n;
    j = II - i * n;
    if (i > 0) {
      JJ = II - n;
      PetscCall(MatSetValues(A, 1, &II, 1, &JJ, &v, ADD_VALUES));
    }
    if (i < m - 1) {
      JJ = II + n;
      PetscCall(MatSetValues(A, 1, &II, 1, &JJ, &v, ADD_VALUES));
    }
    if (j > 0) {
      JJ = II - 1;
      PetscCall(MatSetValues(A, 1, &II, 1, &JJ, &v, ADD_VALUES));
    }
    if (j < n - 1) {
      JJ = II + 1;
      PetscCall(MatSetValues(A, 1, &II, 1, &JJ, &v, ADD_VALUES));
    }
    v = 4.0;
    PetscCall(MatSetValues(A, 1, &II, 1, &II, &v, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /* -----------------------------------------------------------------------
   * Vectors
   * --------------------------------------------------------------------- */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
  PetscCall(VecSetSizes(u, PETSC_DECIDE, m * n));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecDuplicate(u, &b));
  PetscCall(VecDuplicate(u, &x));

  /* Register PFLARE PC types */
  PCRegister_PFLARE();

  /* Read -no_power flag (disables power basis for Intel MPI CI) */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-no_power", &no_power, &flg));

  /* -----------------------------------------------------------------------
   * KSP / PC  (PC type and options come from the command line)
   * --------------------------------------------------------------------- */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetFromOptions(ksp));

  /* Determine PC type */
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCGetType(pc, &pctype));
  is_air       = (PetscBool)(strcmp(pctype, PCAIR) == 0);
  is_pflareinv = (PetscBool)(strcmp(pctype, PCPFLAREINV) == 0);

  if (!is_air && !is_pflareinv) {
    if (rank == 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Unexpected PC type '%s'; expected 'air' or 'pflareinv'\n", pctype));
    PetscCall(PetscFinalize());
    return 1;
  }

  /* If -no_power is set and user configured power-basis PCPFLAREINV, skip */
  if (no_power && is_pflareinv) {
    PCPFLAREINVType pflare_type;
    PetscCall(PCPFLAREINVGetType(pc, &pflare_type));
    if (pflare_type == PFLAREINV_POWER) skip = PETSC_TRUE;
  }

  if (!skip) {
    for (count = 1; count <= nsteps; count++) {

      /* Modify the operator between solves so that solve 3 reproduces solve 1 */
      PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
      if (count == 1) {
        v = 2.0;
      } else if (count == 2) {
        v = 0.1;
      } else {
        v = -0.1;
      }
      for (II = Istart; II < Iend; II++) {
        PetscCall(MatSetValues(A, 1, &II, 1, &II, &v, ADD_VALUES));
      }
      PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

      PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_FALSE));
      PetscCall(KSPSetReusePreconditioner(ksp, PETSC_FALSE));
      PetscCall(KSPSetOperators(ksp, A, A));
      PetscCall(VecSet(u, 1.0));
      PetscCall(MatMult(A, u, b));

      /* On first solve: tell PCAIR to reuse CF splitting / sparsity pattern */
      if (count == 1 && is_air) {
        PetscCall(PCAIRSetReuseSparsity(pc, PETSC_TRUE));
      }

      /* On solve 3: restore saved polynomial coefficients */
      if (count == 3) {
        if (is_air) {
          PetscInt petsc_level;
          for (petsc_level = num_levels - 1; petsc_level >= 1; petsc_level--) {
            PetscCall(PCAIRSetPolyCoeffs(pc, petsc_level, COEFFS_INV_AFF,
                                          saved_coeffs[petsc_level],
                                          saved_rows[petsc_level],
                                          saved_cols[petsc_level]));
          }
          PetscCall(PCAIRSetPolyCoeffs(pc, 0, COEFFS_INV_COARSE,
                                        saved_coeffs[0],
                                        saved_rows[0],
                                        saved_cols[0]));
          PetscCall(PCAIRSetReusePolyCoeffs(pc, PETSC_TRUE));
        } else if (is_pflareinv) {
          PetscCall(PCPFLAREINVSetPolyCoeffs(pc, saved_pflareinv_coeffs,
                                              saved_pflareinv_rows,
                                              saved_pflareinv_cols));
          PetscCall(PCPFLAREINVSetReusePolyCoeffs(pc, PETSC_TRUE));
        }
      }

      PetscCall(KSPSolve(ksp, b, x));

      PetscCall(KSPGetConvergedReason(ksp, &reason));
      if (reason <= 0) {
        if (rank == 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, "KSP did not converge on solve %d (reason %d)\n", (int)count, (int)reason));
        PetscCall(PetscFinalize());
        return 1;
      }

      /* Compute residual norm: u = b - A*x */
      PetscCall(MatResidual(A, b, x, u));
      {
        PetscReal norm;
        PetscCall(VecNorm(u, NORM_2, &norm));
        if (count == 1) norm_first = norm;
        if (count == 3) norm_third = norm;
      }

      /* After solve 1: save polynomial coefficients (copy internal data) */
      if (count == 1) {
        if (is_air) {
          PetscReal *ptr;
          PetscInt   rows, cols, petsc_level;

          PetscCall(PCAIRGetNumLevels(pc, &num_levels));

          saved_coeffs = (PetscReal **)calloc((size_t)num_levels, sizeof(PetscReal *));
          saved_rows   = (PetscInt *)calloc((size_t)num_levels, sizeof(PetscInt));
          saved_cols   = (PetscInt *)calloc((size_t)num_levels, sizeof(PetscInt));

          for (petsc_level = num_levels - 1; petsc_level >= 1; petsc_level--) {
            PetscCall(PCAIRGetPolyCoeffs(pc, petsc_level, COEFFS_INV_AFF, &ptr, &rows, &cols));
            saved_coeffs[petsc_level] = (PetscReal *)malloc((size_t)(rows * cols) * sizeof(PetscReal));
            memcpy(saved_coeffs[petsc_level], ptr, (size_t)(rows * cols) * sizeof(PetscReal));
            saved_rows[petsc_level] = rows;
            saved_cols[petsc_level] = cols;
          }
          /* Coarsest PETSc level (petsc_level == 0) uses COEFFS_INV_COARSE */
          PetscCall(PCAIRGetPolyCoeffs(pc, 0, COEFFS_INV_COARSE, &ptr, &rows, &cols));
          saved_coeffs[0] = (PetscReal *)malloc((size_t)(rows * cols) * sizeof(PetscReal));
          memcpy(saved_coeffs[0], ptr, (size_t)(rows * cols) * sizeof(PetscReal));
          saved_rows[0] = rows;
          saved_cols[0] = cols;

        } else if (is_pflareinv) {
          PetscReal *ptr;
          PetscInt   rows, cols;

          PetscCall(PCPFLAREINVGetPolyCoeffs(pc, &ptr, &rows, &cols));
          saved_pflareinv_coeffs = (PetscReal *)malloc((size_t)(rows * cols) * sizeof(PetscReal));
          memcpy(saved_pflareinv_coeffs, ptr, (size_t)(rows * cols) * sizeof(PetscReal));
          saved_pflareinv_rows = rows;
          saved_pflareinv_cols = cols;
        }
      }
    } /* end for count */

    /* -----------------------------------------------------------------------
     * Check that solve 3 reproduced solve 1
     * --------------------------------------------------------------------- */
    if (rank == 0) {
      PetscReal rel_diff = PetscAbsReal(norm_first - norm_third) / norm_first;
      if (rel_diff > 1e-8) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,
          "FAIL: residual norms differ by %g (solve1=%g, solve3=%g)\n",
          (double)rel_diff, (double)norm_first, (double)norm_third));
        PetscCall(PetscFinalize());
        return 1;
      }
    }
  } /* end if !skip */

  /* Free saved PCAIR coefficients */
  if (saved_coeffs) {
    PetscInt petsc_level;
    for (petsc_level = 0; petsc_level < num_levels; petsc_level++) {
      free(saved_coeffs[petsc_level]);
    }
    free(saved_coeffs);
    free(saved_rows);
    free(saved_cols);
  }
  free(saved_pflareinv_coeffs);

  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(PetscFinalize());
  return 0;
}
