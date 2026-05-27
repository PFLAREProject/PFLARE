static char help[] = "Reads a PETSc matrix and computes an ILU(0) factorisation via a\n\
matrix-form (block-Jacobi-like) Chow ParILU. The L and U factors are stored as\n\
separate PETSc matrices, then:\n\
  - L y = b is solved with Richardson + PCAIR (unpreconditioned norm) to rtol 1e-6\n\
  - U x = b is solved with Richardson + PCAIR (unpreconditioned norm) to rtol 1e-6\n\
  - A x = b is solved with GMRES(30) + shell PC applying U^-1 L^-1 (one PCAIR\n\
    apply per factor) to rtol 1e-6\n\
Iteration counts are reported for each. Requires -mat_type aijkokkos -vec_type kokkos.\n\
Input arguments are:\n\
  -f <input_file>           : binary matrix file to load\n\
  -parilu_tol <real>        : ParILU stencil-residual tolerance (default 1e-4)\n\
  -parilu_max_sweeps <int>  : max ParILU sweeps (default 100)\n\
  -L_*, -U_*, -A_*          : KSP/PC options for the three solves\n\n";

#include <petscksp.h>
#include <string.h>
#include "pflare.h"

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/*  Helpers                                                                    */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

static PetscErrorCode CreateMatLike(MPI_Comm comm, MatType mtype,
                                    PetscInt m, PetscInt n, PetscInt M, PetscInt N,
                                    const PetscInt *dnnz, const PetscInt *onnz,
                                    Mat *out)
{
  PetscFunctionBeginUser;
  PetscCall(MatCreate(comm, out));
  PetscCall(MatSetSizes(*out, m, n, M, N));
  PetscCall(MatSetType(*out, mtype));
  PetscCall(MatXAIJSetPreallocation(*out, 1, dnnz, onnz, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Apply the python-reference DEFAULT_AIR_OPTS to a PCAIR. */
static PetscErrorCode ApplyPythonAIRDefaults(PC pc)
{
  PetscFunctionBeginUser;
  PetscCall(PCAIRSetDiagScalePolys(pc, PETSC_TRUE));
  PetscCall(PCAIRSetADrop(pc, 1e-6));
  PetscCall(PCAIRSetRDrop(pc, 1e-4));
  PetscCall(PCAIRSetCoarsestDiagScalePolys(pc, PETSC_TRUE));
  PetscCall(PCAIRSetCFSplittingType(pc, CF_DIAG_DOM));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Build an inner preonly+PCAIR KSP for a triangular factor, matching the
   python _LUAirShellPC with LU_PC_INNER max_it=1 (single PCAIR application,
   no residual norm work). */
static PetscErrorCode CreateInnerAIRKSP(MPI_Comm comm, Mat factor, const char *prefix, KSP *ksp)
{
  PC pc;
  PetscFunctionBeginUser;
  PetscCall(KSPCreate(comm, ksp));
  PetscCall(KSPSetType(*ksp, KSPPREONLY));
  PetscCall(KSPSetNormType(*ksp, KSP_NORM_NONE));
  PetscCall(KSPSetOperators(*ksp, factor, factor));
  PetscCall(KSPSetInitialGuessNonzero(*ksp, PETSC_FALSE));
  PetscCall(KSPGetPC(*ksp, &pc));
  PetscCall(PCSetType(pc, PCAIR));
  PetscCall(ApplyPythonAIRDefaults(pc));
  if (prefix) PetscCall(KSPSetOptionsPrefix(*ksp, prefix));
  PetscCall(KSPSetFromOptions(*ksp));
  PetscCall(KSPSetUp(*ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PCShell context: applies y = U^-1 L^-1 x via inner KSPs wrapping PCAIR.
   Mirrors _LUAirShellPC from test_ilu.py with the default LU_PC_INNER. */
typedef struct {
  KSP ksp_L;
  KSP ksp_U;
  Vec tmp;
} LUShellCtx;

static PetscErrorCode LUShellApply(PC pc, Vec x, Vec y)
{
  LUShellCtx *ctx;
  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(KSPSolve(ctx->ksp_L, x, ctx->tmp));
  PetscCall(KSPSolve(ctx->ksp_U, ctx->tmp, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LUShellDestroy(PC pc)
{
  LUShellCtx *ctx;
  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(KSPDestroy(&ctx->ksp_L));
  PetscCall(KSPDestroy(&ctx->ksp_U));
  PetscCall(VecDestroy(&ctx->tmp));
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/*  Main                                                                       */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

int main(int argc, char **args)
{
  Mat A, A_diff_type;
  Mat L = NULL, U = NULL, A_L_strict = NULL, A_U = NULL, R_L = NULL, R_U = NULL, M = NULL;
  Vec inv_dU, b_rand, y, x_sol;
  PetscRandom rnd;
  PetscViewer fd;
  char file[PETSC_MAX_PATH_LEN];
  PetscBool flg;
  PetscInt m, n, M_size, N_size, one = 1;
  MatType mtype, mtype_input;
  int npe;
  PetscInt max_sweeps = 100, sweep;
  PetscReal parilu_tol = 1e-4;
  PetscBool converged = PETSC_FALSE;

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  /* Register the pflare PC types so PCAIR is available. */
  PCRegister_PFLARE();

  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", file, sizeof(file), &flg));
  if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Must indicate binary file with the -f option");
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatLoad(A, fd));
  PetscCall(PetscViewerDestroy(&fd));

  /* Partition the loaded matrix when in parallel */
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &npe));
  if (npe != 1) {
    MatPartitioning part;
    IS is, isrows;
    Mat A_partitioned;
    PetscCall(MatPartitioningCreate(PETSC_COMM_WORLD, &part));
    PetscCall(MatPartitioningSetAdjacency(part, A));
    PetscCall(MatPartitioningSetNParts(part, npe));
    PetscCall(MatPartitioningSetFromOptions(part));
    PetscCall(MatPartitioningApply(part, &is));
    PetscCall(ISBuildTwoSided(is, NULL, &isrows));
    PetscCall(MatCreateSubMatrix(A, isrows, isrows, MAT_INITIAL_MATRIX, &A_partitioned));
    PetscCall(MatDestroy(&A));
    PetscCall(MatPartitioningDestroy(&part));
    PetscCall(ISDestroy(&is));
    PetscCall(ISDestroy(&isrows));
    A = A_partitioned;
  }

  /* Convert to the user-requested matrix type (e.g. aijkokkos) */
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatGetSize(A, &M_size, &N_size));
  PetscCall(MatCreateFromOptions(PETSC_COMM_WORLD, NULL, one, m, n, M_size, N_size, &A_diff_type));
  PetscCall(MatAssemblyBegin(A_diff_type, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A_diff_type, MAT_FINAL_ASSEMBLY));

  PetscCall(MatGetType(A, &mtype));
  PetscCall(MatGetType(A_diff_type, &mtype_input));
  if (strcmp(mtype, mtype_input) != 0) {
    PetscCall(MatCopy(A, A_diff_type, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatDestroy(&A));
    A = A_diff_type;
  } else {
    PetscCall(MatDestroy(&A_diff_type));
  }

  /* Require kokkos matrix type since the ParILU pattern restriction routine
     we call only handles kokkos AIJ. The serial backend of Kokkos covers CPU. */
  PetscCall(MatGetType(A, &mtype));
  {
    PetscBool ok_seq = PETSC_FALSE, ok_mpi = PETSC_FALSE, ok_any = PETSC_FALSE;
    PetscCall(PetscStrcmp(mtype, MATSEQAIJKOKKOS, &ok_seq));
    PetscCall(PetscStrcmp(mtype, MATMPIAIJKOKKOS, &ok_mpi));
    PetscCall(PetscStrcmp(mtype, MATAIJKOKKOS,    &ok_any));
    PetscCheck(ok_seq || ok_mpi || ok_any, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "ilu_factors requires a kokkos AIJ matrix; invoke with -mat_type aijkokkos -vec_type kokkos (got %s)", mtype);
  }

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-parilu_max_sweeps", &max_sweeps, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-parilu_tol", &parilu_tol, NULL));

  PetscInt rstart, rend, cstart, cend;
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  PetscCall(MatGetOwnershipRangeColumn(A, &cstart, &cend));
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatGetSize(A, &M_size, &N_size));

  /* Pass 1: count strict-lower (without diag), upper (with diag) entries per row,
     split into diagonal-block (d) and off-diagonal-block (o) for MPI preallocation. */
  PetscInt *L_d_nnz, *L_o_nnz, *Ls_d_nnz, *Ls_o_nnz, *U_d_nnz, *U_o_nnz;
  PetscCall(PetscMalloc6(m, &L_d_nnz, m, &L_o_nnz, m, &Ls_d_nnz, m, &Ls_o_nnz, m, &U_d_nnz, m, &U_o_nnz));

  for (PetscInt i = 0; i < m; i++) {
    PetscInt gi = rstart + i;
    PetscInt ncols;
    const PetscInt *cols;
    PetscCall(MatGetRow(A, gi, &ncols, &cols, NULL));
    PetscInt ls_d = 0, ls_o = 0, u_d = 0, u_o = 0;
    for (PetscInt j = 0; j < ncols; j++) {
      PetscInt c = cols[j];
      if (c < gi) {
        if (c >= cstart && c < cend) ls_d++;
        else ls_o++;
      } else {
        if (c >= cstart && c < cend) u_d++;
        else u_o++;
      }
    }
    PetscCall(MatRestoreRow(A, gi, &ncols, &cols, NULL));
    Ls_d_nnz[i] = ls_d;
    Ls_o_nnz[i] = ls_o;
    U_d_nnz[i]  = u_d;
    U_o_nnz[i]  = u_o;
    L_d_nnz[i]  = ls_d + 1; /* +1 for the unit diagonal we add */
    L_o_nnz[i]  = ls_o;
  }

  PetscCall(CreateMatLike(PETSC_COMM_WORLD, mtype, m, n, M_size, N_size, L_d_nnz,  L_o_nnz,  &L));
  PetscCall(CreateMatLike(PETSC_COMM_WORLD, mtype, m, n, M_size, N_size, U_d_nnz,  U_o_nnz,  &U));
  PetscCall(CreateMatLike(PETSC_COMM_WORLD, mtype, m, n, M_size, N_size, Ls_d_nnz, Ls_o_nnz, &A_L_strict));
  PetscCall(CreateMatLike(PETSC_COMM_WORLD, mtype, m, n, M_size, N_size, U_d_nnz,  U_o_nnz,  &A_U));
  PetscCall(CreateMatLike(PETSC_COMM_WORLD, mtype, m, n, M_size, N_size, Ls_d_nnz, Ls_o_nnz, &R_L));
  PetscCall(CreateMatLike(PETSC_COMM_WORLD, mtype, m, n, M_size, N_size, U_d_nnz,  U_o_nnz,  &R_U));

  PetscCall(PetscFree6(L_d_nnz, L_o_nnz, Ls_d_nnz, Ls_o_nnz, U_d_nnz, U_o_nnz));

  /* Pass 2: insert values. L gets 1.0 on diagonal and 0.0 on the strict-lower
     pattern. U gets A's values on the upper pattern (incl. diagonal). A_L_strict
     and A_U cache A's restricted values for the residual update each sweep.
     R_L and R_U are initialised to zero on their patterns; they are overwritten
     each sweep via MatCopy. */
  for (PetscInt i = 0; i < m; i++) {
    PetscInt gi = rstart + i;
    PetscInt ncols;
    const PetscInt *cols;
    const PetscScalar *vals;
    PetscCall(MatGetRow(A, gi, &ncols, &cols, &vals));
    PetscScalar one_val = 1.0, zero_val = 0.0;
    PetscCall(MatSetValues(L, 1, &gi, 1, &gi, &one_val, INSERT_VALUES));
    for (PetscInt j = 0; j < ncols; j++) {
      PetscInt c = cols[j];
      PetscScalar v = vals[j];
      if (c < gi) {
        PetscCall(MatSetValues(L,          1, &gi, 1, &c, &zero_val, INSERT_VALUES));
        PetscCall(MatSetValues(A_L_strict, 1, &gi, 1, &c, &v,        INSERT_VALUES));
        PetscCall(MatSetValues(R_L,        1, &gi, 1, &c, &zero_val, INSERT_VALUES));
      } else {
        PetscCall(MatSetValues(U,   1, &gi, 1, &c, &v,        INSERT_VALUES));
        PetscCall(MatSetValues(A_U, 1, &gi, 1, &c, &v,        INSERT_VALUES));
        PetscCall(MatSetValues(R_U, 1, &gi, 1, &c, &zero_val, INSERT_VALUES));
      }
    }
    PetscCall(MatRestoreRow(A, gi, &ncols, &cols, &vals));
  }

  {
    Mat all[] = {L, U, A_L_strict, A_U, R_L, R_U};
    for (int k = 0; k < 6; k++) PetscCall(MatAssemblyBegin(all[k], MAT_FINAL_ASSEMBLY));
    for (int k = 0; k < 6; k++) PetscCall(MatAssemblyEnd  (all[k], MAT_FINAL_ASSEMBLY));
  }

  PetscReal A_norm;
  PetscCall(MatNorm(A, NORM_FROBENIUS, &A_norm));
  PetscReal threshold = parilu_tol * A_norm;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "||A||_F = %.6e, ParILU stencil-residual threshold = %.6e\n",
                        (double)A_norm, (double)threshold));

  PetscCall(MatCreateVecs(U, &inv_dU, NULL));

  /* ParILU sweep:
       M     = L * U
       R_L   = A_L_strict - M  restricted to pat(L_strict)
       R_U   = A_U        - M  restricted to pat(U)
       res   = sqrt(||R_L||_F^2 + ||R_U||_F^2)
       if res < threshold: done
       R_L   *= diag(U)^-1   (right scaling)
       L     += R_L (subset pattern)
       U     += R_U (same pattern)
  */
  for (sweep = 0; sweep < max_sweeps; sweep++) {
    if (sweep == 0) PetscCall(MatMatMult(L, U, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &M));
    else            PetscCall(MatMatMult(L, U, MAT_REUSE_MATRIX,   PETSC_DEFAULT, &M));

    PetscCall(MatCopy(A_L_strict, R_L, SAME_NONZERO_PATTERN));
    remove_from_sparse_match_kokkos(&M, &R_L, 0, 1, -1.0);

    PetscCall(MatCopy(A_U, R_U, SAME_NONZERO_PATTERN));
    remove_from_sparse_match_kokkos(&M, &R_U, 0, 1, -1.0);

    PetscReal rl_norm, ru_norm;
    PetscCall(MatNorm(R_L, NORM_FROBENIUS, &rl_norm));
    PetscCall(MatNorm(R_U, NORM_FROBENIUS, &ru_norm));
    PetscReal res = PetscSqrtReal(rl_norm * rl_norm + ru_norm * ru_norm);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "  ParILU sweep %3" PetscInt_FMT "  stencil residual = %.6e\n",
                          sweep, (double)res));
    if (res < threshold) { converged = PETSC_TRUE; sweep++; break; }

    PetscCall(MatGetDiagonal(U, inv_dU));
    PetscCall(VecReciprocal(inv_dU));
    PetscCall(MatDiagonalScale(R_L, NULL, inv_dU));

    PetscCall(MatAXPY(L, 1.0, R_L, SUBSET_NONZERO_PATTERN));
    PetscCall(MatAXPY(U, 1.0, R_U, SAME_NONZERO_PATTERN));
  }
  if (converged) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "ParILU converged in %" PetscInt_FMT " sweeps\n", sweep));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "ParILU did NOT converge within %" PetscInt_FMT " sweeps\n", max_sweeps));
  }

  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  /*  Solves                                                                  */
  /*    1. L y = b   with Richardson + PCAIR (unpreconditioned norm)          */
  /*    2. U x = b   with Richardson + PCAIR (unpreconditioned norm)          */
  /*    3. A x = b   with GMRES(30)  + shell PC = U^-1 L^-1 via PCAIR         */
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rnd));
  PetscCall(PetscRandomSetFromOptions(rnd));

  PetscCall(MatCreateVecs(L, &b_rand, &y));
  PetscCall(MatCreateVecs(U, NULL, &x_sol));

  PetscInt           its_L, its_U, its_A;
  KSPConvergedReason reason_L, reason_U, reason_A;
  KSP                ksp_L, ksp_U, ksp_A;
  PC                 pc_L_solve, pc_U_solve;

  /* 1. Standalone L y = b ---------------------------------------------------- */
  PetscCall(VecSetRandom(b_rand, rnd));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_L));
  PetscCall(KSPSetType(ksp_L, KSPRICHARDSON));
  PetscCall(KSPSetNormType(ksp_L, KSP_NORM_UNPRECONDITIONED));
  PetscCall(KSPSetOperators(ksp_L, L, L));
  PetscCall(KSPSetTolerances(ksp_L, 1e-6, 1e-50, PETSC_DEFAULT, 2000));
  PetscCall(KSPGetPC(ksp_L, &pc_L_solve));
  PetscCall(PCSetType(pc_L_solve, PCAIR));
  PetscCall(PCAIRSetDiagScalePolys(pc_L_solve, PETSC_TRUE));
  PetscCall(PCAIRSetADrop(pc_L_solve, 1e-6));
  PetscCall(PCAIRSetRDrop(pc_L_solve, 1e-4));
  PetscCall(PCAIRSetCoarsestDiagScalePolys(pc_L_solve, PETSC_TRUE));
  PetscCall(PCAIRSetCFSplittingType(pc_L_solve, CF_DIAG_DOM));
  PetscCall(KSPSetOptionsPrefix(ksp_L, "L_"));
  PetscCall(KSPSetFromOptions(ksp_L));
  PetscCall(KSPSolve(ksp_L, b_rand, y));
  PetscCall(KSPGetIterationNumber(ksp_L, &its_L));
  PetscCall(KSPGetConvergedReason(ksp_L, &reason_L));
  PetscCall(KSPDestroy(&ksp_L));

  /* 2. Standalone U x = b ---------------------------------------------------- */
  PetscCall(VecSetRandom(b_rand, rnd));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_U));
  PetscCall(KSPSetType(ksp_U, KSPRICHARDSON));
  PetscCall(KSPSetNormType(ksp_U, KSP_NORM_UNPRECONDITIONED));
  PetscCall(KSPSetOperators(ksp_U, U, U));
  PetscCall(KSPSetTolerances(ksp_U, 1e-6, 1e-50, PETSC_DEFAULT, 2000));
  PetscCall(KSPGetPC(ksp_U, &pc_U_solve));
  PetscCall(PCSetType(pc_U_solve, PCAIR));
  PetscCall(PCAIRSetDiagScalePolys(pc_U_solve, PETSC_TRUE));
  PetscCall(PCAIRSetADrop(pc_U_solve, 1e-6));
  PetscCall(PCAIRSetRDrop(pc_U_solve, 1e-4));
  PetscCall(PCAIRSetCoarsestDiagScalePolys(pc_U_solve, PETSC_TRUE));
  PetscCall(PCAIRSetCFSplittingType(pc_U_solve, CF_DIAG_DOM));
  PetscCall(KSPSetOptionsPrefix(ksp_U, "U_"));
  PetscCall(KSPSetFromOptions(ksp_U));
  PetscCall(KSPSolve(ksp_U, b_rand, x_sol));
  PetscCall(KSPGetIterationNumber(ksp_U, &its_U));
  PetscCall(KSPGetConvergedReason(ksp_U, &reason_U));
  PetscCall(KSPDestroy(&ksp_U));

  /* 3. A x = b with GMRES(30) and a shell PC applying U^-1 L^-1 ------------- */
  LUShellCtx *shell_ctx;
  PetscCall(PetscNew(&shell_ctx));
  PetscCall(CreateInnerAIRKSP(PETSC_COMM_WORLD, L, "A_pc_L_", &shell_ctx->ksp_L));
  PetscCall(CreateInnerAIRKSP(PETSC_COMM_WORLD, U, "A_pc_U_", &shell_ctx->ksp_U));
  PetscCall(MatCreateVecs(L, &shell_ctx->tmp, NULL));

  Vec b_A, x_A;
  PetscCall(MatCreateVecs(A, &b_A, &x_A));
  PetscCall(VecSetRandom(b_A, rnd));
  /* x_A defaults to 0 from MatCreateVecs */

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_A));
  PetscCall(KSPSetType(ksp_A, KSPGMRES));
  PetscCall(KSPGMRESSetRestart(ksp_A, 30));
  PetscCall(KSPSetNormType(ksp_A, KSP_NORM_UNPRECONDITIONED));
  PetscCall(KSPSetOperators(ksp_A, A, A));
  PetscCall(KSPSetTolerances(ksp_A, 1e-6, 1e-50, PETSC_DEFAULT, 2000));

  {
    PC pc_A;
    PetscCall(KSPGetPC(ksp_A, &pc_A));
    PetscCall(PCSetType(pc_A, PCSHELL));
    PetscCall(PCShellSetContext(pc_A, shell_ctx));
    PetscCall(PCShellSetApply(pc_A, LUShellApply));
    PetscCall(PCShellSetDestroy(pc_A, LUShellDestroy));
    PetscCall(PCShellSetName(pc_A, "LU_AIR_shell"));
  }

  PetscCall(KSPSetOptionsPrefix(ksp_A, "A_"));
  PetscCall(KSPSetFromOptions(ksp_A));
  PetscCall(KSPSolve(ksp_A, b_A, x_A));
  PetscCall(KSPGetIterationNumber(ksp_A, &its_A));
  PetscCall(KSPGetConvergedReason(ksp_A, &reason_A));

  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L solve (richardson + PCAIR): %" PetscInt_FMT " iterations (reason %d)\n", its_L, (int)reason_L));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "U solve (richardson + PCAIR): %" PetscInt_FMT " iterations (reason %d)\n", its_U, (int)reason_U));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "A x = b solve (gmres(30) + LU shell PC): %" PetscInt_FMT " iterations (reason %d)\n", its_A, (int)reason_A));

  int exit_code = 0;
  if (!converged || reason_L < 0 || reason_U < 0 || reason_A < 0) exit_code = 1;

  /* Cleanup */
  PetscCall(KSPDestroy(&ksp_A));   /* destroys pc_A which destroys shell_ctx */
  PetscCall(VecDestroy(&b_A));
  PetscCall(VecDestroy(&x_A));
  PetscCall(VecDestroy(&b_rand));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&x_sol));
  PetscCall(VecDestroy(&inv_dU));
  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(MatDestroy(&M));
  PetscCall(MatDestroy(&L));
  PetscCall(MatDestroy(&U));
  PetscCall(MatDestroy(&A_L_strict));
  PetscCall(MatDestroy(&A_U));
  PetscCall(MatDestroy(&R_L));
  PetscCall(MatDestroy(&R_U));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return exit_code;
}
