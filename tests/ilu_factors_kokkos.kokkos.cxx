static char help[] = "Reads a PETSc matrix and computes an ILU(0) factorisation using\n\
Kokkos Kernels' spiluk (level-0 incomplete LU). The L and U factors are stored as\n\
separate PETSc Kokkos AIJ matrices, then a suite of iterative solves is run, all using\n\
GMRES(30) or Richardson with the unpreconditioned norm to rtol 1e-6.\n\
\n\
This is a Kokkos C++ version of ilu_factors.c. The factorisation differs by rank count:\n\
  Single rank (SeqAIJKokkos): the L and U factors come directly from a Kokkos Kernels\n\
    incomplete-LU built from the algorithm's device views. The algorithm is selected with\n\
    -ilu_algorithm: 'spiluk' (level-based ILU(k), the default) or 'parilut' (threshold-\n\
    based parallel ILUT).\n\
  Parallel (MPIAIJKokkos): the matrix-form (block-Jacobi-like) Chow ParILU sweep loop\n\
    from ilu_factors.c is used over the whole distributed L/U, but the local diagonal\n\
    block of L/U is *seeded* with an exact Kokkos Kernels spiluk(0) factorisation of that\n\
    block. The outer Chow-Patel sweeps then converge the remaining non-local (off-process)\n\
    coupling. In parallel only -ilu_algorithm spiluk with -fill_level 0 is supported;\n\
    parilut and higher fill levels error out.\n\
\n\
For each inner-PC kind in {AIRG, GMRES poly (matrix-free), Neumann poly\n\
(matrix-free), ISAI, Jacobi} the test runs:\n\
  - L y = b   with Richardson + inner PC\n\
  - U x = b   with Richardson + inner PC\n\
\n\
It also runs A x = b twice via the LU shell PC applying U^-1 L^-1:\n\
  - GMRES(30) + shell PC with PCAIR inner solves (one apply per factor)\n\
  - GMRES(30) + shell PC with PCJACOBI inner solves (sweeps = ceil(max AIR cycle complexity of L,U))\n\
\n\
And a baseline:\n\
  - A x = b   with GMRES(30) + PETSc's built-in PCBJACOBI/ILU\n\
\n\
Iteration counts (and the final ||b-Op*x||/||b|| if we hit max iterations) are\n\
reported for each. Runs on a Kokkos AIJ matrix on one or more MPI ranks.\n\
Input arguments are:\n\
  -f <input_file>           : binary matrix file to load\n\
  -ilu_algorithm <name>     : 'spiluk' (default) or 'parilut' (parilut single-rank only)\n\
  -fill_level <int>         : spiluk ILU fill level k (default 0; parallel requires 0)\n\
  -parilut_max_iter <int>   : parilut max iterations (default 20)\n\
  -parilut_fill_in_limit <r>: parilut nnz(L+U)/nnz(A) cap (default 0.75)\n\
  -parilut_residual_delta <r>: parilut residual-change stop (default 1e-2)\n\
  -parilu_tol <real>        : (parallel) ParILU stencil-residual tolerance (default 1e-4)\n\
  -parilu_max_sweeps <int>  : (parallel) max ParILU sweeps (default 100)\n\
  -L_*,   -U_*              : AIRG  factor solves\n\
  -L_gmres_*,   -U_gmres_*  : GMRES-poly factor solves\n\
  -L_neumann_*, -U_neumann_*: Neumann-poly factor solves\n\
  -L_isai_*,    -U_isai_*   : ISAI factor solves\n\
  -L_jac_*,     -U_jac_*    : Jacobi factor solves\n\
  -A_*                      : Ax=b GMRES + LU shell PC (PCAIR inner)\n\
  -A_jac_*                  : Ax=b GMRES + LU shell PC (PCJACOBI inner)\n\
  -Apc_*                    : Ax=b GMRES + PCBJACOBI/ILU baseline\n\n";

/* Our petsc kokkos definitions - has to go first (pulls in petscvec_kokkos.hpp,
   petscmat_kokkos.hpp, the SeqAIJKokkos internals and DefaultExecutionSpace/Space). */
#include "kokkos_helper.hpp"
#include <petscksp.h>
#include <string.h>
#include "pflare.h"
#include <KokkosKernels_Handle.hpp>
#include <KokkosSparse_spiluk.hpp>
#include <KokkosSparse_par_ilut.hpp>
#include <KokkosSparse_SortCrs.hpp>

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/*  Helpers                                                                    */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

/* Create an AIJ-like matrix of the requested type with per-row d/o nonzero
   counts. Mirrors the helper of the same name in ilu_factors.c — used to
   preallocate the distributed L/U and the ParILU scratch matrices. */
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

/* Build an inner KSP for a triangular factor.
   max_it == 1: preonly (single PC application, no residual norm work), matching
                python _LUAirShellPC with LU_PC_INNER max_it=1.
   max_it  > 1: Richardson with unpreconditioned norm, rtol=1e-6, atol=1e-50,
                matching python LU_PC_INNER with max_it>1. */
static PetscErrorCode CreateInnerKSP(MPI_Comm comm, Mat factor, const char *prefix,
                                     InnerPCKind kind, PetscInt max_it, KSP *ksp)
{
  PC pc;
  PetscFunctionBeginUser;
  PetscCall(KSPCreate(comm, ksp));
  if (max_it == 1) {
    PetscCall(KSPSetType(*ksp, KSPPREONLY));
    PetscCall(KSPSetNormType(*ksp, KSP_NORM_NONE));
  } else {
    PetscCall(KSPSetType(*ksp, KSPRICHARDSON));
    PetscCall(KSPSetNormType(*ksp, KSP_NORM_UNPRECONDITIONED));
    PetscCall(KSPSetTolerances(*ksp, 1e-6, 1e-50, PETSC_DEFAULT, max_it));
  }
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

/* Run Kokkos Kernels spiluk (level-based ILU(k)) on the SeqAIJKokkos matrix
   A_seq and return the L (unit lower-triangular) and U (upper-triangular with
   pivots on the diagonal) factors as SeqAIJKokkos matrices built directly from
   the algorithm's device views. A_seq must be unit-diagonal and sorted (the
   ParILU convention). Used both for the single-rank whole-matrix factorisation
   and to seed the local diagonal block in the parallel path. */
static PetscErrorCode ComputeSpilukFactors(MPI_Comm comm, Mat A_seq, PetscInt fill_lev,
                                           Mat *L, Mat *U)
{
  PetscInt m, n;
  PetscFunctionBeginUser;
  PetscCall(MatGetLocalSize(A_seq, &m, &n));

  /* A_seq's synced device values — MatSeqAIJGetKokkosView guarantees the device
     side is up to date. */
  Kokkos::View<const PetscScalar *> A_values;
  PetscCall(MatSeqAIJGetKokkosView(A_seq, &A_values));

  /* A_seq's CSR structure straight from the Kokkos CSR graph — already
     PetscInt-typed device views (64-bit safe). */
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos *>(A_seq->spptr);
  auto     A_rowmap  = aijkok->csrmat.graph.row_map;   /* const PetscInt view, size m+1 */
  auto     A_entries = aijkok->csrmat.graph.entries;   /* const PetscInt view, size nnz */
  PetscInt annz      = (PetscInt)A_values.extent(0);

  using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
      PetscInt, PetscInt, PetscScalar,
      DefaultExecutionSpace, DefaultMemorySpace, DefaultMemorySpace>;
  KernelHandle kh;

  Kokkos::View<PetscInt *>    L_row_map("L_row_map", m + 1);
  Kokkos::View<PetscInt *>    U_row_map("U_row_map", m + 1);
  Kokkos::View<PetscInt *>    L_entries, U_entries;
  Kokkos::View<PetscScalar *> L_values,  U_values;

  /* Initial nnz estimate for the L/U entries views (symbolic computes the exact
     count and we resize below). annz + m is the exact upper bound for ILU(0);
     for higher fill levels use a generous over-estimate, as in the Kokkos
     Kernels spiluk perf test (EXPAND_FACT * nnz * (fill_lev + 1)). */
  const PetscInt EXPAND_FACT = 6;
  PetscInt       nnz_est     = (fill_lev == 0) ? (annz + m) : EXPAND_FACT * annz * (fill_lev + 1);
  kh.create_spiluk_handle(KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_TP1,
                          m, nnz_est, nnz_est);
  auto h = kh.get_spiluk_handle();

  L_entries = Kokkos::View<PetscInt *>("L_entries", h->get_nnzL());
  U_entries = Kokkos::View<PetscInt *>("U_entries", h->get_nnzU());

  KokkosSparse::spiluk_symbolic(&kh, fill_lev, A_rowmap, A_entries,
                                L_row_map, L_entries, U_row_map, U_entries);
  Kokkos::fence();

  Kokkos::resize(L_entries, h->get_nnzL());
  Kokkos::resize(U_entries, h->get_nnzU());
  L_values = Kokkos::View<PetscScalar *>("L_values", h->get_nnzL());
  U_values = Kokkos::View<PetscScalar *>("U_values", h->get_nnzU());

  KokkosSparse::spiluk_numeric(&kh, fill_lev, A_rowmap, A_entries, A_values,
                               L_row_map, L_entries, L_values,
                               U_row_map, U_entries, U_values);
  Kokkos::fence();
  kh.destroy_spiluk_handle();
  /* spiluk_symbolic already sorts the L and U graphs internally, so the PETSc
     AIJ sorted-column invariant holds — wrap the device views directly. */

  PetscCall(MatSeqAIJRestoreKokkosView(A_seq, &A_values));

  PetscCall(MatCreateSeqAIJKokkosWithKokkosViews(comm, m, n, L_row_map, L_entries, L_values, L));
  PetscCall(MatCreateSeqAIJKokkosWithKokkosViews(comm, m, n, U_row_map, U_entries, U_values, U));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/*  Main                                                                       */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

int main(int argc, char **args)
{
  Mat         A, A_diff_type;
  Mat         L = NULL, U = NULL;
  Vec         b_rand, x_sol;
#if defined(PETSC_USE_LOG)
  PetscLogStage stage_ilu, stage_L_solve, stage_U_solve, stage_A_shell, stage_A_pcilu;
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
  PetscBool   converged = PETSC_FALSE;
  PetscBool   solves_converged = PETSC_TRUE;
  PetscInt    jac_max_it = 1; /* Jacobi inner iterations, derived from AIR cycle complexity */

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  /* Abort with a stack trace at the point of any PETSc error (the API
     equivalent of -on_error_abort) so every solve below fails loudly. */
  PetscCall(PetscPushErrorHandler(PetscAbortErrorHandler, NULL));

  /* Register the pflare PC types so PCAIR is available. */
  PCRegister_PFLARE();

  /* One log stage per phase that has user-meaningful timing — visible in
     -log_view as separate sections. The KSP solves below each call an
     explicit KSPSetUp before KSPSolve so the per-stage breakdown also splits
     setup vs solve via the standard KSPSetUp / KSPSolve events. */
  PetscCall(PetscLogStageRegister("ILU factorisation (spiluk / seeded ParILU)", &stage_ilu));
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
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Input matrix file (-f): %s\n", file));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatLoad(A, fd));
  PetscCall(PetscViewerDestroy(&fd));

  /* Partition the loaded matrix when in parallel (same as ilu_factors.c). */
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

  /* Reorder A for better ILU quality; default is natural (no-op).
     Override with -mat_ordering_type rcm|nd|qmd|... on the command line. */
  {
    char ordering[64] = MATORDERINGNATURAL;
    IS   row_perm, col_perm;
    Mat  A_reordered;
    PetscCall(PetscOptionsGetString(NULL, NULL, "-mat_ordering_type", ordering, sizeof(ordering), NULL));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Reordering requested (-mat_ordering_type): %s\n", ordering));
    PetscCall(MatGetOrdering(A, ordering, &row_perm, &col_perm));
    PetscCall(MatPermute(A, row_perm, col_perm, &A_reordered));
    PetscCall(MatDestroy(&A));
    A = A_reordered;
    PetscCall(ISDestroy(&row_perm));
    PetscCall(ISDestroy(&col_perm));
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

  /* This path needs a Kokkos AIJ matrix: SeqAIJKokkos (single rank) or
     MPIAIJKokkos (parallel). Reject any non-Kokkos type. */
  PetscCall(MatGetType(A, &mtype));
  if (strcmp(mtype, MATSEQAIJKOKKOS) != 0 && strcmp(mtype, MATMPIAIJKOKKOS) != 0) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "ilu_factors_kokkos requires a Kokkos AIJ matrix; rerun with "
                          "-mat_type aijkokkos -vec_type kokkos.\n"));
    PetscCall(MatDestroy(&A));
    PetscCall(PetscFinalize());
    return 1;
  }

  /* Read the factorisation controls up front. The parallel path only supports an
     exact spiluk(0) seed of the local diagonal block; parilut and higher fill
     levels are single-rank only. */
  char     algorithm[32] = "spiluk";
  PetscInt fill_lev      = 0;
  PetscCall(PetscOptionsGetString(NULL, NULL, "-ilu_algorithm", algorithm, sizeof(algorithm), NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-fill_level", &fill_lev, NULL));
  if (npe != 1 && (strcmp(algorithm, "spiluk") != 0 || fill_lev != 0)) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "ilu_factors_kokkos in parallel only supports -ilu_algorithm spiluk with "
                          "-fill_level 0 (the local diagonal block is seeded with an exact spiluk(0) "
                          "factorisation); rerun single-rank for parilut / higher fill.\n"));
    PetscCall(MatDestroy(&A));
    PetscCall(PetscFinalize());
    return 1;
  }

  /* Left-scale A by 1/diag(A) before the factorisation (mirrors test_ilu.py scale_mode=1).
     Reduces non-normality of the input and produces a unit-diagonal A which is
     the ILU starting point seen by all downstream solves. */
  {
    Vec inv_dA;
    PetscCall(MatCreateVecs(A, &inv_dA, NULL));
    PetscCall(MatGetDiagonal(A, inv_dA));
    PetscCall(VecReciprocal(inv_dA));
    PetscCall(MatDiagonalScale(A, inv_dA, NULL));
    PetscCall(VecDestroy(&inv_dA));
  }

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rnd));
  PetscCall(PetscRandomSetFromOptions(rnd));
  PetscCall(MatCreateVecs(A, &b_rand, &x_sol));
  /* Single random rhs reused across all solves. */
  PetscCall(VecSetRandom(b_rand, rnd));

  /* A x = b with GMRES(30) and PETSc's built-in PCBJACOBI/ILU baseline, run
     before the factorisation to establish whether the problem is solvable
     independently of factor quality. */
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

  /* Incomplete LU factorisation. With A unit-diagonal and sorted both paths
     produce
       L = unit lower-triangular (1.0 stored on the diagonal)
       U = upper-triangular with the pivots on the diagonal
     which is the convention the post-scaling step below expects.

     Single rank: the factors come directly from a Kokkos Kernels incomplete-LU
       (spiluk or parilut) built from the algorithm's device views.
     Parallel:    the matrix-form (block-Jacobi-like) Chow ParILU sweep loop runs
       over the whole distributed L/U, with the local diagonal block seeded by an
       exact spiluk(0) factorisation so the sweeps only need to converge the
       non-local (off-process) coupling. */
  PetscCall(PetscLogStagePush(stage_ilu));
  if (npe == 1) {
    PetscInt nnzL = 0, nnzU = 0;

    if (strcmp(algorithm, "parilut") == 0) {
      PetscCall(MatGetLocalSize(A, &m, &n));

      /* A's synced device values + CSR structure for the Kokkos kernel. */
      Kokkos::View<const PetscScalar *> A_values;
      PetscCall(MatSeqAIJGetKokkosView(A, &A_values));
      Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
      auto A_rowmap  = aijkok->csrmat.graph.row_map;
      auto A_entries = aijkok->csrmat.graph.entries;

      Kokkos::View<PetscInt *>    L_row_map("L_row_map", m + 1);
      Kokkos::View<PetscInt *>    U_row_map("U_row_map", m + 1);
      Kokkos::View<PetscInt *>    L_entries, U_entries;
      Kokkos::View<PetscScalar *> L_values,  U_values;

      using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
          PetscInt, PetscInt, PetscScalar,
          DefaultExecutionSpace, DefaultMemorySpace, DefaultMemorySpace>;
      KernelHandle kh;

      /* Threshold-based parallel ILUT. Tunable via the command line; defaults
         match KokkosKernelsHandle::create_par_ilut_handle. */
      PetscInt  par_max_iter = 100;
      PetscReal par_fill_in  = 1.0; /* nnz(L+U) capped at fill_in_limit x nnz(A) */
      PetscReal par_res_delta = 1e-4; /* stop when the residual change drops below this */
      PetscCall(PetscOptionsGetInt(NULL, NULL, "-parilut_max_iter", &par_max_iter, NULL));
      PetscCall(PetscOptionsGetReal(NULL, NULL, "-parilut_fill_in_limit", &par_fill_in, NULL));
      PetscCall(PetscOptionsGetReal(NULL, NULL, "-parilut_residual_delta", &par_res_delta, NULL));
      kh.create_par_ilut_handle((PetscInt)par_max_iter, par_res_delta, par_fill_in, false, false);
      auto h = kh.get_par_ilut_handle();

      /* Symbolic sets the initial nnz(L)/nnz(U) on the handle. */
      KokkosSparse::Experimental::par_ilut_symbolic(&kh, A_rowmap, A_entries, L_row_map, U_row_map);
      Kokkos::fence();
      L_entries = Kokkos::View<PetscInt *>("L_entries", h->get_nnzL());
      L_values  = Kokkos::View<PetscScalar *>("L_values", h->get_nnzL());
      U_entries = Kokkos::View<PetscInt *>("U_entries", h->get_nnzU());
      U_values  = Kokkos::View<PetscScalar *>("U_values", h->get_nnzU());

      /* Numeric iterates; its internal threshold filtering reallocates the L/U
         entries/values views (passed by reference) to the final filtered nnz and
         keeps them consistent with the row maps. Use those final views directly —
         the handle's get_nnzL()/get_nnzU() still report the symbolic estimate, so
         take the true nnz from the views' extents. */
      KokkosSparse::Experimental::par_ilut_numeric(&kh, A_rowmap, A_entries, A_values,
                                                   L_row_map, L_entries, L_values,
                                                   U_row_map, U_entries, U_values);
      Kokkos::fence();
      kh.destroy_par_ilut_handle();
      nnzL = (PetscInt)L_entries.extent(0);
      nnzU = (PetscInt)U_entries.extent(0);

      /* Sort each row ascending for the PETSc AIJ invariant (par_ilut sorts
         internally, this is belt-and-braces). */
      KokkosSparse::sort_crs_matrix(L_row_map, L_entries, L_values);
      KokkosSparse::sort_crs_matrix(U_row_map, U_entries, U_values);
      Kokkos::fence();
      PetscCall(MatSeqAIJRestoreKokkosView(A, &A_values));

      PetscCall(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, m, n, L_row_map, L_entries, L_values, &L));
      PetscCall(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, m, n, U_row_map, U_entries, U_values, &U));
    } else if (strcmp(algorithm, "spiluk") == 0) {
      /* Level-based ILU(k); fill level k from -fill_level (default 0). */
      MatInfo infoL, infoU;
      PetscCall(ComputeSpilukFactors(PETSC_COMM_SELF, A, fill_lev, &L, &U));
      PetscCall(MatGetInfo(L, MAT_LOCAL, &infoL));
      PetscCall(MatGetInfo(U, MAT_LOCAL, &infoU));
      nnzL = (PetscInt)infoL.nz_used;
      nnzU = (PetscInt)infoU.nz_used;
    } else {
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT,
              "Unknown -ilu_algorithm '%s'; use 'spiluk' or 'parilut'", algorithm);
    }

    converged = PETSC_TRUE;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "Computed ILU via Kokkos Kernels %s: nnz(L) = %" PetscInt_FMT
                          ", nnz(U) = %" PetscInt_FMT "\n", algorithm, nnzL, nnzU));
  } else {
    /* ~~~ Parallel: spiluk-seeded block-Jacobi Chow ParILU ~~~ */
    Mat       Ad, L_dd, U_dd, A_L_strict, A_U, R_L, R_U, Mprod = NULL;
    Vec       inv_dU;
    PetscInt  rstart, rend, cstart, cend;
    PetscInt  max_sweeps = 100, sweep;
    PetscReal parilu_tol = 1e-4;
    PetscBool parilu_diverged = PETSC_FALSE;

    PetscCall(PetscOptionsGetInt(NULL, NULL, "-parilu_max_sweeps", &max_sweeps, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-parilu_tol", &parilu_tol, NULL));

    /* Exact spiluk(0) of the local diagonal block (owned by A — read only). */
    PetscCall(MatGetDiagonalBlock(A, &Ad));
    PetscCall(ComputeSpilukFactors(PETSC_COMM_SELF, Ad, 0, &L_dd, &U_dd));

    PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
    PetscCall(MatGetOwnershipRangeColumn(A, &cstart, &cend));
    PetscCall(MatGetLocalSize(A, &m, &n));
    PetscCall(MatGetSize(A, &M_size, &N_size));

    /* Pass 1: count strict-lower (no diag) and upper (with diag) entries per row,
       split into diagonal-block (d) and off-diagonal-block (o) for MPI prealloc. */
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
          if (c >= cstart && c < cend) ls_d++; else ls_o++;
        } else {
          if (c >= cstart && c < cend) u_d++; else u_o++;
        }
      }
      PetscCall(MatRestoreRow(A, gi, &ncols, &cols, NULL));
      Ls_d_nnz[i] = ls_d;  Ls_o_nnz[i] = ls_o;
      U_d_nnz[i]  = u_d;   U_o_nnz[i]  = u_o;
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

    /* Pass 2: L gets 1.0 on the diagonal and 0.0 on the strict-lower pattern; U
       gets A's upper values. A_L_strict / A_U cache A's restricted values (the
       residual target); R_L / R_U are zero-initialised on their patterns. */
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

    /* Seed: overwrite the local diagonal-block entries of L/U with the exact
       spiluk(0) factors of A's diagonal block. L_dd/U_dd are local (m x m) with
       block-local column indices; the global column is cstart + jloc. The unit
       diagonal of L is left untouched. Off-diagonal-block entries keep A's
       values (a good starting guess); the sweeps below converge them. */
    for (PetscInt i = 0; i < m; i++) {
      PetscInt           gi = rstart + i, ncols;
      const PetscInt    *cols;
      const PetscScalar *vals;
      PetscCall(MatGetRow(L_dd, i, &ncols, &cols, &vals));
      for (PetscInt j = 0; j < ncols; j++) {
        if (cols[j] >= i) continue;            /* strict-lower only (skip unit diag) */
        PetscInt gc = cstart + cols[j];
        PetscCall(MatSetValues(L, 1, &gi, 1, &gc, &vals[j], INSERT_VALUES));
      }
      PetscCall(MatRestoreRow(L_dd, i, &ncols, &cols, &vals));
      PetscCall(MatGetRow(U_dd, i, &ncols, &cols, &vals));
      for (PetscInt j = 0; j < ncols; j++) {
        if (cols[j] < i) continue;             /* upper incl. diagonal pivot */
        PetscInt gc = cstart + cols[j];
        PetscCall(MatSetValues(U, 1, &gi, 1, &gc, &vals[j], INSERT_VALUES));
      }
      PetscCall(MatRestoreRow(U_dd, i, &ncols, &cols, &vals));
    }
    PetscCall(MatAssemblyBegin(L, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(L, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyBegin(U, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(U, MAT_FINAL_ASSEMBLY));
    PetscCall(MatDestroy(&L_dd));
    PetscCall(MatDestroy(&U_dd));

    PetscReal A_norm;
    PetscCall(MatNorm(A, NORM_FROBENIUS, &A_norm));
    PetscReal threshold = parilu_tol * A_norm;
    PetscCall(MatCreateVecs(U, &inv_dU, NULL));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "||A||_F = %.6e, ParILU stencil-residual threshold = %.6e\n",
                          (double)A_norm, (double)threshold));

    /* ParILU sweep (block-Jacobi style, MPI/Kokkos friendly):
         M     = L * U
         R_L   = A_L_strict - M  restricted to pat(L_strict)
         R_U   = A_U        - M  restricted to pat(U)
         res   = sqrt(||R_L||_F^2 + ||R_U||_F^2); stop if < threshold
         R_L  *= diag(U)^-1  (right scaling); L += R_L; U += R_U
       Seeded with the exact spiluk(0) of the diagonal block, the local-block
       residual starts near zero so the sweeps converge the off-process coupling. */
    PetscReal res_initial = -1.0;
    for (sweep = 0; sweep < max_sweeps; sweep++) {
      if (sweep == 0) PetscCall(MatMatMult(L, U, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Mprod));
      else            PetscCall(MatMatMult(L, U, MAT_REUSE_MATRIX,   PETSC_DEFAULT, &Mprod));

      PetscCall(MatCopy(A_L_strict, R_L, SAME_NONZERO_PATTERN));
      remove_from_sparse_match(Mprod, R_L, 0, 1, -1.0);
      PetscCall(MatCopy(A_U, R_U, SAME_NONZERO_PATTERN));
      remove_from_sparse_match(Mprod, R_U, 0, 1, -1.0);

      PetscReal rl_norm, ru_norm;
      PetscCall(MatNorm(R_L, NORM_FROBENIUS, &rl_norm));
      PetscCall(MatNorm(R_U, NORM_FROBENIUS, &ru_norm));
      PetscReal res = PetscSqrtReal(rl_norm * rl_norm + ru_norm * ru_norm);
      if (sweep == 0) res_initial = res;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                            "  ParILU sweep %3" PetscInt_FMT "  stencil residual = %.6e\n",
                            sweep, (double)res));
      if (res < threshold) { converged = PETSC_TRUE; sweep++; break; }
      if (PetscIsInfOrNanReal(res) || (res_initial > 0.0 && res > 1000.0 * res_initial)) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                              "  ParILU diverged at sweep %" PetscInt_FMT
                              ": residual %.6e exceeds 1000x initial %.6e or is non-finite\n",
                              sweep, (double)res, (double)res_initial));
        parilu_diverged = PETSC_TRUE;
        break;
      }

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

    PetscCall(MatDestroy(&Mprod));
    PetscCall(MatDestroy(&R_L));
    PetscCall(MatDestroy(&R_U));
    PetscCall(MatDestroy(&A_L_strict));
    PetscCall(MatDestroy(&A_U));
    PetscCall(VecDestroy(&inv_dU));

    if (parilu_diverged) {
      PetscCall(PetscLogStagePop());
      PetscCall(MatDestroy(&L));
      PetscCall(MatDestroy(&U));
      PetscCall(VecDestroy(&b_rand));
      PetscCall(VecDestroy(&x_sol));
      PetscCall(PetscRandomDestroy(&rnd));
      PetscCall(MatDestroy(&A));
      PetscCall(PetscFinalize());
      return 1;
    }
  }
  PetscCall(PetscLogStagePop());

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
  /*  All factor solves use the unpreconditioned residual norm and rtol=1e-6. */
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

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
    PC          pc_A, pcL_inner, pcU_inner;
    LUShellCtx *shell_ctx;
    PetscReal   cL = -1.0, cU = -1.0, sL = -1.0, sU = -1.0, gL = -1.0, gU = -1.0;
    /* Build the inner KSPs inside this stage so their PCAIR setup (the bulk
       of the shell PC's "setup" cost) is timed here too. */
    PetscCall(PetscNew(&shell_ctx));
    PetscCall(CreateInnerKSP(PETSC_COMM_WORLD, L, "A_pc_L_", INNER_PC_AIR, 1, &shell_ctx->ksp_L));
    PetscCall(CreateInnerKSP(PETSC_COMM_WORLD, U, "A_pc_U_", INNER_PC_AIR, 1, &shell_ctx->ksp_U));
    /* Fetch cycle complexity from the two AIR hierarchies (PCSetUp was called
       inside CreateInnerKSP above) to derive the Jacobi inner iteration count
       used in the subsequent stage_A_shell_jac solve, mirroring
       _jac_inner_from_cycle in test_ilu.py. */
    PetscCall(KSPGetPC(shell_ctx->ksp_L, &pcL_inner));
    PetscCall(KSPGetPC(shell_ctx->ksp_U, &pcU_inner));
    PetscCall(PCAIRGetCycleComplexity(pcL_inner, &cL));
    PetscCall(PCAIRGetCycleComplexity(pcU_inner, &cU));
    PetscCall(PCAIRGetStorageComplexity(pcL_inner, &sL));
    PetscCall(PCAIRGetStorageComplexity(pcU_inner, &sU));
    PetscCall(PCAIRGetGridComplexity(pcL_inner, &gL));
    PetscCall(PCAIRGetGridComplexity(pcU_inner, &gU));
    {
      PetscReal cmax = (cL > 0.0 && cU > 0.0) ? PetscMax(cL, cU) :
                       (cL > 0.0) ? cL : (cU > 0.0) ? cU : 1.0;
      jac_max_it = PetscMax((PetscInt)1, (PetscInt)PetscCeilReal(cmax));
    }
    // Allow user to override the inner jacobi iterations
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-jac_max_it", &jac_max_it, NULL));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "AIR complexities: cycle L=%.3f U=%.3f, storage L=%.3f U=%.3f, grid L=%.3f U=%.3f\n",
                          (double)cL, (double)cU, (double)sL, (double)sU, (double)gL, (double)gU));
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
    PetscCall(ReportSolve("A x = b solve (gmres(30) + LU shell PC, AIRG inner)", ksp_A, A, b_rand, x_sol, &solves_converged));
    /* KSPDestroy triggers the PCSHELL destroy callback which tears down
       shell_ctx (its inner KSPs and tmp vec). */
    PetscCall(KSPDestroy(&ksp_A));
  }
  PetscCall(PetscLogStagePop());

  /* A x = b with GMRES(30) and a shell PC applying U^-1 L^-1 via PCJACOBI inner.
     The number of Jacobi sweeps per factor apply equals jac_max_it, derived from
     the AIR cycle complexities of the L and U PCAIR hierarchies above */
  PetscCall(PetscLogStagePush(stage_A_shell_jac));
  {
    KSP         ksp_Ajac;
    PC          pc_Ajac;
    LUShellCtx *shell_ctx;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Jacobi inner max_it=%" PetscInt_FMT "\n", jac_max_it));
    PetscCall(PetscNew(&shell_ctx));
    PetscCall(CreateInnerKSP(PETSC_COMM_WORLD, L, "A_jac_pc_L_", INNER_PC_JACOBI, jac_max_it, &shell_ctx->ksp_L));
    PetscCall(CreateInnerKSP(PETSC_COMM_WORLD, U, "A_jac_pc_U_", INNER_PC_JACOBI, jac_max_it, &shell_ctx->ksp_U));
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

  PetscCall(MatDestroy(&L));
  PetscCall(MatDestroy(&U));

  int exit_code = (converged && solves_converged) ? 0 : 1;

  /* Final cleanup */
  PetscCall(VecDestroy(&b_rand));
  PetscCall(VecDestroy(&x_sol));
  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return exit_code;
}
