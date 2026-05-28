static char help[] = "Reads a PETSc matrix and computes an ILU(0) factorisation via a\n\
matrix-form (block-Jacobi-like) Chow ParILU. The L and U factors are stored as\n\
separate PETSc matrices, then a suite of iterative solves is run, all using\n\
GMRES(30) or Richardson with the unpreconditioned norm to rtol 1e-6.\n\
\n\
For each inner-PC kind in {AIRG, GMRES poly (matrix-free), Neumann poly\n\
(matrix-free), ISAI, Jacobi} the test runs:\n\
  - L y = b   with Richardson + inner PC\n\
  - U x = b   with Richardson + inner PC\n\
\n\
It also runs A x = b twice via the LU shell PC applying U^-1 L^-1:\n\
  - GMRES(30) + shell PC with PCAIR inner solves (one apply per factor)\n\
  - GMRES(30) + shell PC with PCJACOBI inner solves (one Jacobi sweep per factor)\n\
\n\
And a baseline:\n\
  - A x = b   with GMRES(30) + PETSc's built-in PCBJACOBI/ILU\n\
\n\
Iteration counts (and the final ||b-Op*x||/||b|| if we hit max iterations) are\n\
reported for each. Works on either CPU AIJ or Kokkos AIJ matrices.\n\
Input arguments are:\n\
  -f <input_file>           : binary matrix file to load\n\
  -parilu_tol <real>        : ParILU stencil-residual tolerance (default 1e-4)\n\
  -parilu_max_sweeps <int>  : max ParILU sweeps (default 100)\n\
  -L_*,   -U_*              : AIRG  factor solves\n\
  -L_gmres_*,   -U_gmres_*  : GMRES-poly factor solves\n\
  -L_neumann_*, -U_neumann_*: Neumann-poly factor solves\n\
  -L_isai_*,    -U_isai_*   : ISAI factor solves\n\
  -L_jac_*,     -U_jac_*    : Jacobi factor solves\n\
  -A_*                      : Ax=b GMRES + LU shell PC (PCAIR inner)\n\
  -A_jac_*                  : Ax=b GMRES + LU shell PC (PCJACOBI inner)\n\
  -Apc_*                    : Ax=b GMRES + PCBJACOBI/ILU baseline\n\n";

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

/* Report the outcome of a KSP solve. If the solver hit max iterations
   (KSP_DIVERGED_ITS) print the actual relative residual ||b - Op*x||_2/||b||_2
   so we can see how close it got. */
static PetscErrorCode ReportSolve(const char *label, KSP ksp, Mat op, Vec b, Vec x,
                                  PetscBool *all_converged)
{
  PetscInt           its;
  KSPConvergedReason reason;
  PetscFunctionBeginUser;
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  if (all_converged && reason <= 0) *all_converged = PETSC_FALSE;
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

/* Inner-PC options used both for the standalone L/U solves and for the inner
   factor solves inside the Ax=b LU shell PC. Mirrors the python OPTION_SETS
   at test_ilu.py:1404-1411 (AIRG / GMRES poly / Neumann poly / ISAI / Jacobi). */
typedef enum {
  INNER_PC_AIR,
  INNER_PC_GMRES_POLY,
  INNER_PC_NEUMANN_POLY,
  INNER_PC_ISAI,
  INNER_PC_JACOBI,
} InnerPCKind;

/* Set the PC type and any kind-specific configuration. Matches the python
   option dicts DEFAULT_AIR_OPTS / OPTS_GMRES_POLY / OPTS_NEUMANN / OPTS_ISAI /
   OPTS_JAC at test_ilu.py:241-273. */
static PetscErrorCode ConfigureInnerPC(PC pc, InnerPCKind kind)
{
  PetscFunctionBeginUser;
  switch (kind) {
  case INNER_PC_AIR:
    PetscCall(PCSetType(pc, PCAIR));
    PetscCall(ApplyPythonAIRDefaults(pc));
    break;
  case INNER_PC_GMRES_POLY:
    PetscCall(PCSetType(pc, PCPFLAREINV));
    PetscCall(PCPFLAREINVSetType(pc, PFLAREINV_NEWTON));
    PetscCall(PCPFLAREINVSetMatrixFree(pc, PETSC_TRUE));
    break;
  case INNER_PC_NEUMANN_POLY:
    PetscCall(PCSetType(pc, PCPFLAREINV));
    PetscCall(PCPFLAREINVSetType(pc, PFLAREINV_NEUMANN));
    PetscCall(PCPFLAREINVSetMatrixFree(pc, PETSC_TRUE));
    break;
  case INNER_PC_ISAI:
    PetscCall(PCSetType(pc, PCPFLAREINV));
    PetscCall(PCPFLAREINVSetType(pc, PFLAREINV_ISAI));
    break;
  case INNER_PC_JACOBI:
    PetscCall(PCSetType(pc, PCJACOBI));
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Build an inner preonly+<kind> KSP for a triangular factor, matching the
   python _LUAirShellPC with LU_PC_INNER max_it=1 (single PC application,
   no residual norm work). */
static PetscErrorCode CreateInnerKSP(MPI_Comm comm, Mat factor, const char *prefix,
                                     InnerPCKind kind, KSP *ksp)
{
  PC pc;
  PetscFunctionBeginUser;
  PetscCall(KSPCreate(comm, ksp));
  PetscCall(KSPSetType(*ksp, KSPPREONLY));
  PetscCall(KSPSetNormType(*ksp, KSP_NORM_NONE));
  PetscCall(KSPSetOperators(*ksp, factor, factor));
  PetscCall(KSPSetInitialGuessNonzero(*ksp, PETSC_FALSE));
  PetscCall(KSPGetPC(*ksp, &pc));
  PetscCall(ConfigureInnerPC(pc, kind));
  if (prefix) PetscCall(KSPSetOptionsPrefix(*ksp, prefix));
  PetscCall(KSPSetFromOptions(*ksp));
  PetscCall(KSPSetUp(*ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Standalone Richardson + <kind> solve on `factor`. rtol=1e-6, max_it=2000,
   unpreconditioned residual norm, explicit setup so -log_view sees setup vs
   solve separately. Reports iteration count via ReportSolve. */
static PetscErrorCode RunFactorSolve(MPI_Comm comm, Mat factor, Vec b, Vec x,
                                     InnerPCKind kind, const char *label,
                                     const char *opts_prefix, PetscBool *all_converged)
{
  KSP ksp;
  PC  pc;
  PetscFunctionBeginUser;
  PetscCall(KSPCreate(comm, &ksp));
  PetscCall(KSPSetType(ksp, KSPRICHARDSON));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCall(KSPSetOperators(ksp, factor, factor));
  PetscCall(KSPSetTolerances(ksp, 1e-6, 1e-50, PETSC_DEFAULT, 2000));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(ConfigureInnerPC(pc, kind));
  if (opts_prefix) PetscCall(KSPSetOptionsPrefix(ksp, opts_prefix));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(ReportSolve(label, ksp, factor, b, x, all_converged));
  PetscCall(KSPDestroy(&ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PCShell context: applies y = U_raw^-1 L^-1 x via inner KSPs wrapping PCAIR.
   If inv_diag_U_raw is provided, the intermediate vector is element-wise multiplied
   by it between the L and U solves to recover U_raw^-1 from U_scaled^-1
   (where U_scaled = diag(1/diag(U_raw)) * U_raw is what the inner KSP sees). */
typedef struct {
  KSP ksp_L;
  KSP ksp_U;
  Vec tmp;
  Vec inv_diag_U_raw;   /* may be NULL */
} LUShellCtx;

static PetscErrorCode LUShellApply(PC pc, Vec x, Vec y)
{
  LUShellCtx *ctx;
  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(KSPSolve(ctx->ksp_L, x, ctx->tmp));
  if (ctx->inv_diag_U_raw) PetscCall(VecPointwiseMult(ctx->tmp, ctx->inv_diag_U_raw, ctx->tmp));
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
  PetscCall(VecDestroy(&ctx->inv_diag_U_raw));
  PetscCall(PetscFree(ctx));
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
  PetscLogStage stage_L_gmres, stage_U_gmres, stage_L_neumann, stage_U_neumann;
  PetscLogStage stage_L_isai,  stage_U_isai,  stage_L_jac,     stage_U_jac;
  PetscLogStage stage_A_shell_jac;
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
  PetscBool   solves_converged = PETSC_TRUE;

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  /* Register the pflare PC types so PCAIR is available. */
  PCRegister_PFLARE();

  /* One log stage per phase that has user-meaningful timing — visible in
     -log_view as separate sections. The KSP solves below each call an
     explicit KSPSetUp before KSPSolve so the per-stage breakdown also splits
     setup vs solve via the standard KSPSetUp / KSPSolve events. */
  PetscCall(PetscLogStageRegister("ParILU sweeps",                       &stage_parilu));
  PetscCall(PetscLogStageRegister("L solve (Richardson+PCAIR)",          &stage_L_solve));
  PetscCall(PetscLogStageRegister("U solve (Richardson+PCAIR)",          &stage_U_solve));
  PetscCall(PetscLogStageRegister("A solve (GMRES+LU shell PC)",         &stage_A_shell));
  PetscCall(PetscLogStageRegister("A solve (GMRES+PCBJACOBI/ILU)",       &stage_A_pcilu));
  PetscCall(PetscLogStageRegister("L solve (Richardson+GMRES poly)",     &stage_L_gmres));
  PetscCall(PetscLogStageRegister("U solve (Richardson+GMRES poly)",     &stage_U_gmres));
  PetscCall(PetscLogStageRegister("L solve (Richardson+Neumann poly)",   &stage_L_neumann));
  PetscCall(PetscLogStageRegister("U solve (Richardson+Neumann poly)",   &stage_U_neumann));
  PetscCall(PetscLogStageRegister("L solve (Richardson+ISAI)",           &stage_L_isai));
  PetscCall(PetscLogStageRegister("U solve (Richardson+ISAI)",           &stage_U_isai));
  PetscCall(PetscLogStageRegister("L solve (Richardson+PCJACOBI)",       &stage_L_jac));
  PetscCall(PetscLogStageRegister("U solve (Richardson+PCJACOBI)",       &stage_U_jac));
  PetscCall(PetscLogStageRegister("A solve (GMRES+LU shell, Jacobi inner)", &stage_A_shell_jac));

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

  /* Left-scale A by 1/diag(A) before ParILU (mirrors test_ilu.py scale_mode=1).
     Reduces non-normality of the input and produces a unit-diagonal A which is
     the ParILU starting point seen by all downstream solves. */
  {
    Vec inv_dA;
    PetscCall(MatCreateVecs(A, &inv_dA, NULL));
    PetscCall(MatGetDiagonal(A, inv_dA));
    PetscCall(VecReciprocal(inv_dA));
    PetscCall(MatDiagonalScale(A, inv_dA, NULL));
    PetscCall(VecDestroy(&inv_dA));
  }

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

  /* Capture 1/diag(U_raw) and left-scale U by it (mirrors test_ilu.py).
     U becomes unit-diagonal, which reduces non-normality and helps the inner
     AIR solves on the U factor. L is already unit-diagonal so left-scaling by
     its inverse diagonal is a no-op and is skipped. The shell PC for the
     Ax=b solve later multiplies by inv_diag_U_raw between the L and U inner solves
     to recover U_raw^-1 = U_scaled^-1 * diag(1/diag(U_raw)). */
  Vec inv_diag_U_raw;
  PetscCall(MatCreateVecs(U, &inv_diag_U_raw, NULL));
  PetscCall(MatGetDiagonal(U, inv_diag_U_raw));
  PetscCall(VecReciprocal(inv_diag_U_raw));
  PetscCall(MatDiagonalScale(U, inv_diag_U_raw, NULL));

  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  /*  Solves                                                                  */
  /*    L y = b      Richardson + {AIRG, GMRES poly, Neumann poly, ISAI,      */
  /*                               Jacobi} — one solve per inner PC           */
  /*    U x = b      same five inner PCs                                      */
  /*    A x = b      GMRES(30)  + shell PC = U^-1 L^-1 via PCAIR  inner       */
  /*    A x = b      GMRES(30)  + shell PC = U^-1 L^-1 via PCJACOBI inner     */
  /*    A x = b      GMRES(30)  + PETSc's built-in PCBJACOBI/ILU              */
  /*  All factor solves use the unpreconditioned residual norm and rtol=1e-6. */
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rnd));
  PetscCall(PetscRandomSetFromOptions(rnd));
  PetscCall(MatCreateVecs(A, &b_rand, &x_sol));
  /* Single random rhs reused across all solves. */
  PetscCall(VecSetRandom(b_rand, rnd));

  /* The PCAIR Ax=b shell-PC solve below consumes inv_diag_U_raw (its destroy
     callback frees it). The PCJACOBI-inner Ax=b shell solve added later also
     needs the raw-diagonal compensation between L and U solves, so duplicate
     the vec up-front and hand the copy to that shell. */
  Vec inv_diag_U_raw_for_jac;
  PetscCall(VecDuplicate(inv_diag_U_raw, &inv_diag_U_raw_for_jac));
  PetscCall(VecCopy(inv_diag_U_raw, inv_diag_U_raw_for_jac));

  /* L and U standalone solves: Richardson + each inner PC. */
  PetscCall(PetscLogStagePush(stage_L_solve));
  PetscCall(RunFactorSolve(PETSC_COMM_WORLD, L, b_rand, x_sol, INNER_PC_AIR,
                           "L solve (richardson + PCAIR)", "L_", &solves_converged));
  PetscCall(PetscLogStagePop());

  PetscCall(PetscLogStagePush(stage_U_solve));
  PetscCall(RunFactorSolve(PETSC_COMM_WORLD, U, b_rand, x_sol, INNER_PC_AIR,
                           "U solve (richardson + PCAIR)", "U_", &solves_converged));
  PetscCall(PetscLogStagePop());

  PetscCall(PetscLogStagePush(stage_L_gmres));
  PetscCall(RunFactorSolve(PETSC_COMM_WORLD, L, b_rand, x_sol, INNER_PC_GMRES_POLY,
                           "L solve (richardson + GMRES poly)", "L_gmres_", &solves_converged));
  PetscCall(PetscLogStagePop());

  PetscCall(PetscLogStagePush(stage_U_gmres));
  PetscCall(RunFactorSolve(PETSC_COMM_WORLD, U, b_rand, x_sol, INNER_PC_GMRES_POLY,
                           "U solve (richardson + GMRES poly)", "U_gmres_", &solves_converged));
  PetscCall(PetscLogStagePop());

  PetscCall(PetscLogStagePush(stage_L_neumann));
  PetscCall(RunFactorSolve(PETSC_COMM_WORLD, L, b_rand, x_sol, INNER_PC_NEUMANN_POLY,
                           "L solve (richardson + Neumann poly)", "L_neumann_", &solves_converged));
  PetscCall(PetscLogStagePop());

  PetscCall(PetscLogStagePush(stage_U_neumann));
  PetscCall(RunFactorSolve(PETSC_COMM_WORLD, U, b_rand, x_sol, INNER_PC_NEUMANN_POLY,
                           "U solve (richardson + Neumann poly)", "U_neumann_", &solves_converged));
  PetscCall(PetscLogStagePop());

  PetscCall(PetscLogStagePush(stage_L_isai));
  PetscCall(RunFactorSolve(PETSC_COMM_WORLD, L, b_rand, x_sol, INNER_PC_ISAI,
                           "L solve (richardson + ISAI)", "L_isai_", &solves_converged));
  PetscCall(PetscLogStagePop());

  PetscCall(PetscLogStagePush(stage_U_isai));
  PetscCall(RunFactorSolve(PETSC_COMM_WORLD, U, b_rand, x_sol, INNER_PC_ISAI,
                           "U solve (richardson + ISAI)", "U_isai_", &solves_converged));
  PetscCall(PetscLogStagePop());

  PetscCall(PetscLogStagePush(stage_L_jac));
  PetscCall(RunFactorSolve(PETSC_COMM_WORLD, L, b_rand, x_sol, INNER_PC_JACOBI,
                           "L solve (richardson + PCJACOBI)", "L_jac_", &solves_converged));
  PetscCall(PetscLogStagePop());

  PetscCall(PetscLogStagePush(stage_U_jac));
  PetscCall(RunFactorSolve(PETSC_COMM_WORLD, U, b_rand, x_sol, INNER_PC_JACOBI,
                           "U solve (richardson + PCJACOBI)", "U_jac_", &solves_converged));
  PetscCall(PetscLogStagePop());

  /* A x = b with GMRES(30) and a shell PC applying U^-1 L^-1 via PCAIR inner. */
  PetscCall(PetscLogStagePush(stage_A_shell));
  {
    KSP         ksp_A;
    PC          pc_A;
    LUShellCtx *shell_ctx;
    /* Build the inner KSPs inside this stage so their PCAIR setup (the bulk
       of the shell PC's "setup" cost) is timed here too. */
    PetscCall(PetscNew(&shell_ctx));
    PetscCall(CreateInnerKSP(PETSC_COMM_WORLD, L, "A_pc_L_", INNER_PC_AIR, &shell_ctx->ksp_L));
    PetscCall(CreateInnerKSP(PETSC_COMM_WORLD, U, "A_pc_U_", INNER_PC_AIR, &shell_ctx->ksp_U));
    PetscCall(MatCreateVecs(L, &shell_ctx->tmp, NULL));
    /* Hand inv_diag_U_raw to the shell — LUShellDestroy will free it. */
    shell_ctx->inv_diag_U_raw = inv_diag_U_raw;
    inv_diag_U_raw            = NULL;

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
    PetscCall(ReportSolve("A x = b solve (gmres(30) + LU shell PC)", ksp_A, A, b_rand, x_sol, &solves_converged));
    /* KSPDestroy triggers the PCSHELL destroy callback which tears down
       shell_ctx (its inner KSPs and tmp vec). */
    PetscCall(KSPDestroy(&ksp_A));
  }
  PetscCall(PetscLogStagePop());

  /* A x = b with GMRES(30) and a shell PC applying U^-1 L^-1 via one PCJACOBI
     sweep per factor (preonly + PCJACOBI inner). */
  PetscCall(PetscLogStagePush(stage_A_shell_jac));
  {
    KSP         ksp_Ajac;
    PC          pc_Ajac;
    LUShellCtx *shell_ctx;
    PetscCall(PetscNew(&shell_ctx));
    PetscCall(CreateInnerKSP(PETSC_COMM_WORLD, L, "A_jac_pc_L_", INNER_PC_JACOBI, &shell_ctx->ksp_L));
    PetscCall(CreateInnerKSP(PETSC_COMM_WORLD, U, "A_jac_pc_U_", INNER_PC_JACOBI, &shell_ctx->ksp_U));
    PetscCall(MatCreateVecs(L, &shell_ctx->tmp, NULL));
    shell_ctx->inv_diag_U_raw = inv_diag_U_raw_for_jac;
    inv_diag_U_raw_for_jac    = NULL;

    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_Ajac));
    PetscCall(KSPSetType(ksp_Ajac, KSPGMRES));
    PetscCall(KSPGMRESSetRestart(ksp_Ajac, 30));
    PetscCall(KSPSetNormType(ksp_Ajac, KSP_NORM_UNPRECONDITIONED));
    PetscCall(KSPSetOperators(ksp_Ajac, A, A));
    PetscCall(KSPSetTolerances(ksp_Ajac, 1e-6, 1e-50, PETSC_DEFAULT, 2000));
    PetscCall(KSPGetPC(ksp_Ajac, &pc_Ajac));
    PetscCall(PCSetType(pc_Ajac, PCSHELL));
    PetscCall(PCShellSetContext(pc_Ajac, shell_ctx));
    PetscCall(PCShellSetApply(pc_Ajac, LUShellApply));
    PetscCall(PCShellSetDestroy(pc_Ajac, LUShellDestroy));
    PetscCall(PCShellSetName(pc_Ajac, "LU_Jacobi_shell"));
    PetscCall(KSPSetOptionsPrefix(ksp_Ajac, "A_jac_"));
    PetscCall(KSPSetFromOptions(ksp_Ajac));
    PetscCall(KSPSetUp(ksp_Ajac));
    PetscCall(KSPSolve(ksp_Ajac, b_rand, x_sol));
    PetscCall(ReportSolve("A x = b solve (gmres(30) + LU shell PC, Jacobi inner)",
                          ksp_Ajac, A, b_rand, x_sol, &solves_converged));
    PetscCall(KSPDestroy(&ksp_Ajac));
  }
  PetscCall(PetscLogStagePop());

  /* L and U are no longer needed past the shell-PC solves; release before the
     PCILU comparison to free memory. */
  PetscCall(MatDestroy(&L));
  PetscCall(MatDestroy(&U));

  /* A x = b with GMRES(30) and PETSc's built-in PCBJACOBI/ILU baseline. */
  PetscCall(PetscLogStagePush(stage_A_pcilu));
  {
    KSP ksp_Apc;
    PC  pc_Apc;
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
    PetscCall(ReportSolve("A x = b solve (gmres(30) + PCBJACOBI/ILU)", ksp_Apc, A, b_rand, x_sol, &solves_converged));
    PetscCall(KSPDestroy(&ksp_Apc));
  }
  PetscCall(PetscLogStagePop());

  int exit_code = (converged && solves_converged) ? 0 : 1;

  /* Final cleanup */
  PetscCall(VecDestroy(&b_rand));
  PetscCall(VecDestroy(&x_sol));
  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return exit_code;
}
