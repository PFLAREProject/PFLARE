static char help[] = "Reads a PETSc matrix and computes an ILU(0) factorisation via a\n\
matrix-form (block-Jacobi-like) Chow ParILU. The L and U factors are stored as\n\
separate PETSc matrices, then four iterative solves are run, all using GMRES(30)\n\
or Richardson with the unpreconditioned norm to rtol 1e-6:\n\
  - L y = b   with Richardson + PCAIR\n\
  - U x = b   with Richardson + PCAIR\n\
  - A x = b   with GMRES(30)  + shell PC applying U^-1 L^-1 via one PCAIR apply per factor\n\
  - A x = b   with GMRES(30)  + PETSc's built-in PCBJACOBI/ILU (a fair comparison\n\
                                to our ParILU(0))\n\
Iteration counts (and the final ||b-Op*x||/||b|| if we hit max iterations) are\n\
reported for each. Works on either CPU AIJ or Kokkos AIJ matrices.\n\
Input arguments are:\n\
  -f <input_file>           : binary matrix file to load\n\
  -parilu_tol <real>        : ParILU stencil-residual tolerance (default 1e-4)\n\
  -parilu_max_sweeps <int>  : max ParILU sweeps (default 100)\n\
  -L_*, -U_*, -A_*, -Apc_*  : KSP/PC options for the four solves\n\n";

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

/* Report the outcome of a KSP solve. If the solver hit max iterations
   (KSP_DIVERGED_ITS) print the actual relative residual ||b - Op*x||_2/||b||_2
   so we can see how close it got. */
static PetscErrorCode ReportSolve(const char *label, KSP ksp, Mat op, Vec b, Vec x)
{
  PetscInt           its;
  KSPConvergedReason reason;
  PetscFunctionBeginUser;
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  if (reason == KSP_DIVERGED_ITS) {
    Vec       r;
    PetscReal rn, bn;
    PetscCall(VecDuplicate(b, &r));
    PetscCall(MatMult(op, x, r));
    PetscCall(VecAYPX(r, -1.0, b));         /* r = b - Op*x */
    PetscCall(VecNorm(r, NORM_2, &rn));
    PetscCall(VecNorm(b, NORM_2, &bn));
    PetscCall(VecDestroy(&r));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "%s: %" PetscInt_FMT " iterations (reason %d, hit max iterations); "
                          "final ||b-Op*x||/||b|| = %.6e\n",
                          label, its, (int)reason,
                          (double)(bn > 0.0 ? rn / bn : rn)));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "%s: %" PetscInt_FMT " iterations (reason %d)\n",
                          label, its, (int)reason));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/*  Main                                                                       */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

int main(int argc, char **args)
{
  Mat         A, A_diff_type;
  Mat         L = NULL, U = NULL, A_L_strict = NULL, A_U = NULL, R_L = NULL, R_U = NULL, M = NULL;
  Vec         inv_dU, b_rand, x_sol;
#if defined(PETSC_USE_LOG)
  PetscLogStage stage_parilu, stage_L_solve, stage_U_solve, stage_A_shell, stage_A_pcilu;
#endif
  PetscRandom rnd;
  PetscViewer fd;
  char        file[PETSC_MAX_PATH_LEN];
  PetscBool   flg;
  PetscInt    m, n, M_size, N_size, one = 1;
  MatType     mtype, mtype_input;
  int         npe;
  PetscInt    max_sweeps = 100, sweep;
  PetscReal   parilu_tol = 1e-4;
  PetscBool   converged = PETSC_FALSE;

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  /* Register the pflare PC types so PCAIR is available. */
  PCRegister_PFLARE();

  /* One log stage per phase that has user-meaningful timing — visible in
     -log_view as separate sections. The KSP solves below each call an
     explicit KSPSetUp before KSPSolve so the per-stage breakdown also splits
     setup vs solve via the standard KSPSetUp / KSPSolve events. */
  PetscCall(PetscLogStageRegister("ParILU sweeps",                  &stage_parilu));
  PetscCall(PetscLogStageRegister("L solve (Richardson+PCAIR)",     &stage_L_solve));
  PetscCall(PetscLogStageRegister("U solve (Richardson+PCAIR)",     &stage_U_solve));
  PetscCall(PetscLogStageRegister("A solve (GMRES+LU shell PC)",    &stage_A_shell));
  PetscCall(PetscLogStageRegister("A solve (GMRES+PCBJACOBI/ILU)",  &stage_A_pcilu));

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
    IS              is, isrows;
    Mat             A_partitioned;
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

  PetscCall(MatGetType(A, &mtype));
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
    PetscInt        gi = rstart + i;
    PetscInt        ncols;
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
    PetscInt           gi = rstart + i;
    PetscInt           ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscCall(MatGetRow(A, gi, &ncols, &cols, &vals));
    PetscScalar one_val = 1.0, zero_val = 0.0;
    PetscCall(MatSetValues(L, 1, &gi, 1, &gi, &one_val, INSERT_VALUES));
    for (PetscInt j = 0; j < ncols; j++) {
      PetscInt    c = cols[j];
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
    This won't necessarily converge as well as a true asynchronous (ie Gauss-Seidel) ParILU
    but the benefit of this is it works with MPI parallel and in Kokkos automatically
       M     = L * U
       R_L   = A_L_strict - M  restricted to pat(L_strict)
       R_U   = A_U        - M  restricted to pat(U)
       res   = sqrt(||R_L||_F^2 + ||R_U||_F^2)
       if res < threshold: done
       R_L   *= diag(U)^-1   (right scaling)
       L     += R_L (subset pattern)
       U     += R_U (same pattern)
  */
  PetscCall(PetscLogStagePush(stage_parilu));
  for (sweep = 0; sweep < max_sweeps; sweep++) {
    if (sweep == 0) PetscCall(MatMatMult(L, U, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &M));
    else            PetscCall(MatMatMult(L, U, MAT_REUSE_MATRIX,   PETSC_DEFAULT, &M));

    PetscCall(MatCopy(A_L_strict, R_L, SAME_NONZERO_PATTERN));
    // PFLARE function that drops entries outside of a given matrix sparsity
    remove_from_sparse_match(M, R_L, 0, 1, -1.0);

    PetscCall(MatCopy(A_U, R_U, SAME_NONZERO_PATTERN));
    remove_from_sparse_match(M, R_U, 0, 1, -1.0);

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
  PetscCall(PetscLogStagePop());
  if (converged) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "ParILU converged in %" PetscInt_FMT " sweeps\n", sweep));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "ParILU did NOT converge within %" PetscInt_FMT " sweeps\n", max_sweeps));
  }

  /* ParILU scratch state isn't needed past this point — release it now. */
  PetscCall(MatDestroy(&M));
  PetscCall(MatDestroy(&R_L));
  PetscCall(MatDestroy(&R_U));
  PetscCall(MatDestroy(&A_L_strict));
  PetscCall(MatDestroy(&A_U));
  PetscCall(VecDestroy(&inv_dU));

  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  /*  Solves                                                                  */
  /*    1. L y = b      Richardson + PCAIR                                    */
  /*    2. U x = b      Richardson + PCAIR                                    */
  /*    3. A x = b      GMRES(30)  + shell PC = U^-1 L^-1 via PCAIR           */
  /*    4. A x = b      GMRES(30)  + PETSc's built-in PCBJACOBI/ILU           */
  /*  All four solves use the unpreconditioned residual norm and rtol = 1e-6. */
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rnd));
  PetscCall(PetscRandomSetFromOptions(rnd));
  PetscCall(MatCreateVecs(A, &b_rand, &x_sol));

  /* 1. Standalone L y = b ---------------------------------------------------- */
  PetscCall(PetscLogStagePush(stage_L_solve));
  {
    KSP ksp_L;
    PC  pc_L;
    PetscCall(VecSetRandom(b_rand, rnd));
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_L));
    PetscCall(KSPSetType(ksp_L, KSPRICHARDSON));
    PetscCall(KSPSetNormType(ksp_L, KSP_NORM_UNPRECONDITIONED));
    PetscCall(KSPSetOperators(ksp_L, L, L));
    PetscCall(KSPSetTolerances(ksp_L, 1e-6, 1e-50, PETSC_DEFAULT, 2000));
    PetscCall(KSPGetPC(ksp_L, &pc_L));
    PetscCall(PCSetType(pc_L, PCAIR));
    PetscCall(ApplyPythonAIRDefaults(pc_L));
    PetscCall(KSPSetOptionsPrefix(ksp_L, "L_"));
    PetscCall(KSPSetFromOptions(ksp_L));
    /* Explicit setup so -log_view times KSPSetUp separately from KSPSolve. */
    PetscCall(KSPSetUp(ksp_L));
    PetscCall(KSPSolve(ksp_L, b_rand, x_sol));
    PetscCall(ReportSolve("L solve (richardson + PCAIR)", ksp_L, L, b_rand, x_sol));
    PetscCall(KSPDestroy(&ksp_L));
  }
  PetscCall(PetscLogStagePop());

  /* 2. Standalone U x = b ---------------------------------------------------- */
  PetscCall(PetscLogStagePush(stage_U_solve));
  {
    KSP ksp_U;
    PC  pc_U;
    PetscCall(VecSetRandom(b_rand, rnd));
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_U));
    PetscCall(KSPSetType(ksp_U, KSPRICHARDSON));
    PetscCall(KSPSetNormType(ksp_U, KSP_NORM_UNPRECONDITIONED));
    PetscCall(KSPSetOperators(ksp_U, U, U));
    PetscCall(KSPSetTolerances(ksp_U, 1e-6, 1e-50, PETSC_DEFAULT, 2000));
    PetscCall(KSPGetPC(ksp_U, &pc_U));
    PetscCall(PCSetType(pc_U, PCAIR));
    PetscCall(ApplyPythonAIRDefaults(pc_U));
    PetscCall(KSPSetOptionsPrefix(ksp_U, "U_"));
    PetscCall(KSPSetFromOptions(ksp_U));
    PetscCall(KSPSetUp(ksp_U));
    PetscCall(KSPSolve(ksp_U, b_rand, x_sol));
    PetscCall(ReportSolve("U solve (richardson + PCAIR)", ksp_U, U, b_rand, x_sol));
    PetscCall(KSPDestroy(&ksp_U));
  }
  PetscCall(PetscLogStagePop());

  /* 3. A x = b with GMRES(30) and a shell PC applying U^-1 L^-1 ------------- */
  PetscCall(PetscLogStagePush(stage_A_shell));
  {
    KSP         ksp_A;
    PC          pc_A;
    LUShellCtx *shell_ctx;
    /* Build the inner KSPs inside this stage so their PCAIR setup (the bulk
       of the shell PC's "setup" cost) is timed here too. */
    PetscCall(PetscNew(&shell_ctx));
    PetscCall(CreateInnerAIRKSP(PETSC_COMM_WORLD, L, "A_pc_L_", &shell_ctx->ksp_L));
    PetscCall(CreateInnerAIRKSP(PETSC_COMM_WORLD, U, "A_pc_U_", &shell_ctx->ksp_U));
    PetscCall(MatCreateVecs(L, &shell_ctx->tmp, NULL));

    PetscCall(VecSetRandom(b_rand, rnd));
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_A));
    PetscCall(KSPSetType(ksp_A, KSPGMRES));
    PetscCall(KSPGMRESSetRestart(ksp_A, 30));
    PetscCall(KSPSetNormType(ksp_A, KSP_NORM_UNPRECONDITIONED));
    PetscCall(KSPSetOperators(ksp_A, A, A));
    PetscCall(KSPSetTolerances(ksp_A, 1e-6, 1e-50, PETSC_DEFAULT, 2000));
    PetscCall(KSPGetPC(ksp_A, &pc_A));
    PetscCall(PCSetType(pc_A, PCSHELL));
    PetscCall(PCShellSetContext(pc_A, shell_ctx));
    PetscCall(PCShellSetApply(pc_A, LUShellApply));
    PetscCall(PCShellSetDestroy(pc_A, LUShellDestroy));
    PetscCall(PCShellSetName(pc_A, "LU_AIR_shell"));
    PetscCall(KSPSetOptionsPrefix(ksp_A, "A_"));
    PetscCall(KSPSetFromOptions(ksp_A));
    PetscCall(KSPSetUp(ksp_A));
    PetscCall(KSPSolve(ksp_A, b_rand, x_sol));
    PetscCall(ReportSolve("A x = b solve (gmres(30) + LU shell PC)", ksp_A, A, b_rand, x_sol));
    /* KSPDestroy triggers the PCSHELL destroy callback which tears down
       shell_ctx (its inner KSPs and tmp vec). */
    PetscCall(KSPDestroy(&ksp_A));
  }
  PetscCall(PetscLogStagePop());

  /* L and U are only used by the three LU-based solves above; release before
     the PCILU comparison to free memory. */
  PetscCall(MatDestroy(&L));
  PetscCall(MatDestroy(&U));

  /* 4. A x = b with GMRES(30) and PETSc's built-in PCBJACOBI/ILU ------------ */
  PetscCall(PetscLogStagePush(stage_A_pcilu));
  {
    KSP ksp_Apc;
    PC  pc_Apc;
    PetscCall(VecSetRandom(b_rand, rnd));
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_Apc));
    PetscCall(KSPSetType(ksp_Apc, KSPGMRES));
    PetscCall(KSPGMRESSetRestart(ksp_Apc, 30));
    PetscCall(KSPSetNormType(ksp_Apc, KSP_NORM_UNPRECONDITIONED));
    PetscCall(KSPSetOperators(ksp_Apc, A, A));
    PetscCall(KSPSetTolerances(ksp_Apc, 1e-6, 1e-50, PETSC_DEFAULT, 2000));
    PetscCall(KSPGetPC(ksp_Apc, &pc_Apc));
    /* PCBJACOBI's default sub-PC for AIJ blocks is PCILU(0). In serial that's
       exactly PETSc's PCILU on the whole matrix; in MPI it's a per-rank ILU
       inside block-Jacobi — the standard "parallel PCILU" pattern. */
    PetscCall(PCSetType(pc_Apc, PCBJACOBI));
    PetscCall(KSPSetOptionsPrefix(ksp_Apc, "Apc_"));
    PetscCall(KSPSetFromOptions(ksp_Apc));
    PetscCall(KSPSetUp(ksp_Apc));
    PetscCall(KSPSolve(ksp_Apc, b_rand, x_sol));
    PetscCall(ReportSolve("A x = b solve (gmres(30) + PCBJACOBI/ILU)", ksp_Apc, A, b_rand, x_sol));
    PetscCall(KSPDestroy(&ksp_Apc));
  }
  PetscCall(PetscLogStagePop());

  int exit_code = converged ? 0 : 1;

  /* Final cleanup */
  PetscCall(VecDestroy(&b_rand));
  PetscCall(VecDestroy(&x_sol));
  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return exit_code;
}
