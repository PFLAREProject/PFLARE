/*   DMDA/KSP solving a system of linear equations.
     Steady advection-diffusion equation in 2D or 3D with finite difference, advection is upwinded
     Default is 2D. Use -dim 3 for 3D.

     ./adv_diff_fd
             : pure advection with theta = pi/4, dimensionless
               BCs left and bottom dirichlet (u=0), top and right outflow
               Same equation as advection_2d in PyAMG, except we don't eliminate the dirichlet dofs
     ./adv_diff_fd -adv_nondim 0
             : pure advection with theta = pi/4, scaled by Hx * Hy
               BCs left and bottom dirichlet (u=0), top and right outflow
     ./adv_diff_fd -u 0 -v 0 -alpha 1.0
             : pure diffusion scaled by Hx * Hy
               BCs dirichlet on all sides
     ./adv_diff_fd -alpha 1.0
             : advection-diffusion scaled by Hx * Hy with theta=pi/4
               BCs dirichlet on all sides
     ./adv_diff_fd -bottom_only_inflow_one
             : pure advection with theta = pi/4, dimensionless
               BC bottom face dirichlet inflow u=1, left face dirichlet u=0, top and right outflow
     ./adv_diff_fd -dim 3
             : pure advection in 3D with default velocity (1,1,1) normalised
               BCs west (x=0), south (y=0), bottom (z=0) dirichlet u=0; other faces outflow
     ./adv_diff_fd -dim 3 -alpha 1.0
             : advection-diffusion in 3D scaled by Hx*Hy*Hz
               BCs dirichlet on all sides

     Can control dimension with -dim (2 default)
     Can control grid size with -da_grid_x, -da_grid_y, -da_grid_z (default 11)
     Can control the direction of advection with -theta (pi/4 default, 2D only),
       or by giving the -u and -v and -w directly
     In 3D the default velocity is (1,1,1) normalised; -theta only affects the 2D x-y plane
     Can optionally left scale the matrix by the inverse diagonal before solving (-diag_scale)
     Can specify inflow of 1 on the bottom face with -bottom_only_inflow_one (default false),
       in 2D: bottom (j=0) face; in 3D: bottom (k=0, z=0) face
       all other inflow faces are set to u=0
     Can change default velocity from straight line to curved with -curved_velocity (default false)
       2D curved field: u(x,y) = y, v(x,y) = 1-x, w = 0 (rotation about (1,0),
       3D curved field: u = z, v = z, w = 2-x-y 
     Can normalise velocity with -unit_velocity (default true) so that we have a unit velocity.
       If any of u,v,w are set explicitly then unit velocity will be disabled
     Can control domain size in each direction with -L_x, -L_y, -L_z (default 1.0)
     Can write the solution to a VTK structured grid file with -vec_view vtk:solution.vts
       (note: DMDA is a structured mesh so PETSc produces .vts, not .vtu)

*/

static char help[] = "Solves steady advection-diffusion with finite-difference on a structured grid in 2D or 3D.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscvec.h>
#include <string.h>
#include <math.h>

#include "pflare.h"

#define MINE_TRANSPORT_N_REF 450

// Helper function to compute velocity at a point.
// dim: spatial dimension (2 or 3)
// u_const,v_const,w_const: constant velocity components (used when curved_velocity is false)
// x[3]: node coordinates (x, y, z)
// curved_velocity: if true, use spatially-varying field u(x,y)=y, v(x,y)=1-x, w=0
// unit_velocity: if true, normalise the velocity to unit magnitude
static inline void GetVelocity(PetscInt dim, PetscReal u_const, PetscReal v_const, PetscReal w_const,
                               const PetscReal x[], PetscBool curved_velocity, PetscBool unit_velocity,
                               PetscReal vel[])
{
  if (curved_velocity) {
    if (dim == 2) {
      // 2D rotating field around centre (1, 0), always >= 0 on [0,1]^2
      // Enters through bottom and left, exits through right and top
      vel[0] = x[1];         // u(x,y) = y
      vel[1] = 1.0 - x[0];   // v(x,y) = 1-x
      vel[2] = 0.0;
    } else {
      // 3D curved field, symmetric under x<->y:
      //   u = z
      //   v = z
      //   w = (2-x-y)
      vel[0] = x[2];                               // u
      vel[1] = x[2];                               // v
      vel[2] = (2.0 - x[0] - x[1]); // w
    }
  } else {
    // Constant velocity field
    vel[0] = u_const;
    vel[1] = v_const;
    vel[2] = w_const;
  }

  // Normalise velocity magnitude if unit_velocity flag is set
  if (unit_velocity) {
    PetscReal mag = 0.0;
    for (int d = 0; d < dim; ++d) mag += vel[d]*vel[d];
    mag = PetscSqrtReal(mag);
    if (mag > 1e-12) { for (int d = 0; d < dim; ++d) vel[d] /= mag; }
  }
}

extern PetscErrorCode ComputeMat(DM,Mat,PetscInt,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscBool,PetscBool,PetscBool);

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/*  ILU helpers (used only when -ilu_test is active)                          */
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
    PetscCall(VecAYPX(r, -1.0, b));
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

typedef enum {
  INNER_PC_AIR,
  INNER_PC_GMRES_POLY,
  INNER_PC_NEUMANN_POLY,
  INNER_PC_ISAI,
  INNER_PC_JACOBI,
} InnerPCKind;

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

static PetscErrorCode RunFactorSolve(MPI_Comm comm, Mat factor, Vec b, Vec x,
                                     InnerPCKind kind, const char *label,
                                     const char *opts_prefix, PetscBool *all_converged,
                                     KSP *ksp_out)
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
  if (ksp_out) *ksp_out = ksp;
  else PetscCall(KSPDestroy(&ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  KSP ksp_L;
  KSP ksp_U;
  Vec tmp;
  Vec inv_diag_U_raw;
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

int main(int argc,char **argv)
{
  KSP            ksp;
  PC             pc;
  DM             da;
  PetscInt       dim;
  PetscScalar    theta, alpha, u, v, w, u_test, v_test, w_test;
  PetscScalar    L_x, L_y, L_z, L_x_test, L_y_test, L_z_test;
  PetscBool      option_found_u, option_found_v, option_found_w;
  PetscBool      adv_nondim, check_nondim, diag_scale, bottom_only_inflow_one, curved_velocity, unit_velocity;
  Vec x, b, diag_vec;
  Mat A, A_temp;
  KSPConvergedReason reason;
  PetscLogStage setup, gpu_copy;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //       Let's use PFLARE
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  

  // Register the pflare types
  PCRegister_PFLARE();

  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  /*  Mine-transport ILU test  (-ilu_test)                                       */
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  {
    PetscBool ilu_test = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-ilu_test", &ilu_test, NULL));

    if (ilu_test) {
      Mat         A_ilu = NULL, L = NULL, U = NULL;
      Mat         A_L_strict = NULL, A_U = NULL, R_L = NULL, R_U = NULL, M = NULL;
      Vec         inv_dU, inv_diag_U_raw, inv_diag_U_raw_for_jac, b_rand, x_sol;
      PetscRandom rnd;
      MatType     mtype;
      PetscInt    mine_n = 100, max_sweeps = 100, sweep;
      PetscReal   mine_beta = 1500.0, mine_alpha = 1.0, parilu_tol = 1e-4;
      PetscBool   converged = PETSC_FALSE, parilu_diverged = PETSC_FALSE;
      PetscBool   solves_converged = PETSC_TRUE;
      PetscInt    jac_max_it = 1;
      PetscInt    rstart, rend, m, Ntot, M_size;
      int         npe;
      KSP         ksp_L_air, ksp_U_air;
#if defined(PETSC_USE_LOG)
      PetscLogStage stage_parilu, stage_L_air, stage_U_air, stage_A_shell, stage_A_shell_jac;
      PetscLogStage stage_A_pcilu, stage_A_air;
      PetscLogStage stage_L_gmres, stage_U_gmres, stage_L_neumann, stage_U_neumann;
      PetscLogStage stage_L_isai, stage_U_isai, stage_L_jac, stage_U_jac;
#endif

      PetscCall(PetscOptionsGetInt (NULL, NULL, "-mine_n",        &mine_n,      NULL));
      PetscCall(PetscOptionsGetReal(NULL, NULL, "-mine_beta",     &mine_beta,   NULL));
      PetscCall(PetscOptionsGetReal(NULL, NULL, "-alpha",         &mine_alpha,  NULL));
      PetscCall(PetscOptionsGetInt (NULL, NULL, "-parilu_max_sweeps", &max_sweeps, NULL));
      PetscCall(PetscOptionsGetReal(NULL, NULL, "-parilu_tol",    &parilu_tol,  NULL));

      PetscCall(PetscLogStageRegister("ILU A solve (PCBJACOBI/ILU)",    &stage_A_pcilu));
      PetscCall(PetscLogStageRegister("ILU A solve (Richardson+PCAIR)", &stage_A_air));
      PetscCall(PetscLogStageRegister("ILU ParILU sweeps",              &stage_parilu));
      PetscCall(PetscLogStageRegister("ILU L solve (PCAIR)",            &stage_L_air));
      PetscCall(PetscLogStageRegister("ILU U solve (PCAIR)",            &stage_U_air));
      PetscCall(PetscLogStageRegister("ILU A solve (AIRG shell)",       &stage_A_shell));
      PetscCall(PetscLogStageRegister("ILU A solve (Jacobi shell)",     &stage_A_shell_jac));
      PetscCall(PetscLogStageRegister("ILU L solve (GMRES poly)",       &stage_L_gmres));
      PetscCall(PetscLogStageRegister("ILU U solve (GMRES poly)",       &stage_U_gmres));
      PetscCall(PetscLogStageRegister("ILU L solve (Neumann poly)",     &stage_L_neumann));
      PetscCall(PetscLogStageRegister("ILU U solve (Neumann poly)",     &stage_U_neumann));
      PetscCall(PetscLogStageRegister("ILU L solve (ISAI)",             &stage_L_isai));
      PetscCall(PetscLogStageRegister("ILU U solve (ISAI)",             &stage_U_isai));
      PetscCall(PetscLogStageRegister("ILU L solve (PCJACOBI)",         &stage_L_jac));
      PetscCall(PetscLogStageRegister("ILU U solve (PCJACOBI)",         &stage_U_jac));

      /* ── Build mine transport matrix (N²×N², interior unknowns only) ──────── */
      Ntot   = mine_n * mine_n;
      M_size = Ntot;

      PetscCall(MatCreate(PETSC_COMM_WORLD, &A_ilu));
      PetscCall(MatSetSizes(A_ilu, PETSC_DECIDE, PETSC_DECIDE, Ntot, Ntot));
      PetscCall(MatSetFromOptions(A_ilu));
      /* MatGetOwnershipRange triggers MatSetUp internally, establishing the
         parallel row distribution before we compute preallocation counts. */
      PetscCall(MatGetOwnershipRange(A_ilu, &rstart, &rend));
      m = rend - rstart;

      {
        PetscInt *d_nnz_A, *o_nnz_A;
        PetscCall(PetscMalloc2(m, &d_nnz_A, m, &o_nnz_A));
        for (PetscInt k = rstart; k < rend; k++) {
          PetscInt li = k - rstart;
          PetscInt ci = k % mine_n, cj = k / mine_n;
          d_nnz_A[li] = 1; o_nnz_A[li] = 0;
          PetscInt nc[4]     = {k - 1, k + 1, k - mine_n, k + mine_n};
          PetscBool ok[4]    = {ci > 0, ci < mine_n - 1, cj > 0, cj < mine_n - 1};
          for (int q = 0; q < 4; q++) {
            if (!ok[q]) continue;
            if (nc[q] >= rstart && nc[q] < rend) d_nnz_A[li]++;
            else                                  o_nnz_A[li]++;
          }
        }
        PetscCall(MatXAIJSetPreallocation(A_ilu, 1, d_nnz_A, o_nnz_A, NULL, NULL));
        PetscCall(PetscFree2(d_nnz_A, o_nnz_A));
      }

      /* Fill mine transport stencil.
         Row k = j*N + i (0-indexed interior), coordinates x_i=(i+1)*h, y_j=(j+1)*h.
         Central difference of -alpha*Laplacian + beta_eff*(d/dx(e^{xy}u) + d/dy(e^{-xy}u)). */
      {
        PetscReal h      = 1.0 / (mine_n + 1);
        PetscReal h_ref  = 1.0 / (MINE_TRANSPORT_N_REF + 1);
        PetscReal beff   = mine_beta * (h_ref / h);
        PetscReal a_coef = beff / (2.0 * h);
        PetscReal inv_h2 = mine_alpha / (h * h);
        for (PetscInt k = rstart; k < rend; k++) {
          PetscInt    ci = k % mine_n, cj = k / mine_n;
          PetscReal   xi = (ci + 1) * h, yj = (cj + 1) * h;
          PetscScalar diag = (PetscScalar)(4.0 * inv_h2);
          PetscCall(MatSetValues(A_ilu, 1, &k, 1, &k, &diag, INSERT_VALUES));
          if (ci < mine_n - 1) { /* East k+1 */
            PetscInt c = k + 1;
            PetscScalar v = (PetscScalar)(-inv_h2 + a_coef * exp((ci + 2) * h * yj));
            PetscCall(MatSetValues(A_ilu, 1, &k, 1, &c, &v, INSERT_VALUES));
          }
          if (ci > 0) { /* West k-1 */
            PetscInt c = k - 1;
            PetscScalar v = (PetscScalar)(-inv_h2 - a_coef * exp(ci * h * yj));
            PetscCall(MatSetValues(A_ilu, 1, &k, 1, &c, &v, INSERT_VALUES));
          }
          if (cj < mine_n - 1) { /* North k+N */
            PetscInt c = k + mine_n;
            PetscScalar v = (PetscScalar)(-inv_h2 + a_coef * exp(-xi * (cj + 2) * h));
            PetscCall(MatSetValues(A_ilu, 1, &k, 1, &c, &v, INSERT_VALUES));
          }
          if (cj > 0) { /* South k-N */
            PetscInt c = k - mine_n;
            PetscScalar v = (PetscScalar)(-inv_h2 - a_coef * exp(-xi * cj * h));
            PetscCall(MatSetValues(A_ilu, 1, &k, 1, &c, &v, INSERT_VALUES));
          }
        }
        PetscCall(MatAssemblyBegin(A_ilu, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd  (A_ilu, MAT_FINAL_ASSEMBLY));
      }

      /* ── Reorder A (default: natural = no-op; override with -mat_ordering_type) ── */
      {
        char ordering[64] = MATORDERINGNATURAL;
        IS   row_perm, col_perm;
        Mat  A_reordered, A_aij;
        PetscCall(PetscOptionsGetString(NULL, NULL, "-mat_ordering_type", ordering, sizeof(ordering), NULL));
        /* MatGetOrdering only special-cases MATMPIAIJ for the parallel-aware path
           that builds the permutation IS on the matrix's communicator. For
           MATMPIAIJKOKKOS that branch is skipped and the IS is created on
           PETSC_COMM_SELF, which then mismatches A_ilu's communicator inside
           MatPermute. Compute the ordering from a plain-AIJ copy to stay on the
           supported path; the permutation still applies to the kokkos matrix. */
        PetscCall(MatConvert(A_ilu, MATAIJ, MAT_INITIAL_MATRIX, &A_aij));
        PetscCall(MatGetOrdering(A_aij, ordering, &row_perm, &col_perm));
        PetscCall(MatDestroy(&A_aij));
        PetscCall(MatPermute(A_ilu, row_perm, col_perm, &A_reordered));
        PetscCall(MatDestroy(&A_ilu));
        A_ilu = A_reordered;
        PetscCall(ISDestroy(&row_perm));
        PetscCall(ISDestroy(&col_perm));
      }

      /* ── Diagonal left-scale: A ← diag(A)^{-1} A  (mirrors test_ilu.py scale_mode=1) ── */
      {
        Vec inv_dA;
        PetscCall(MatCreateVecs(A_ilu, &inv_dA, NULL));
        PetscCall(MatGetDiagonal(A_ilu, inv_dA));
        PetscCall(VecReciprocal(inv_dA));
        PetscCall(MatDiagonalScale(A_ilu, inv_dA, NULL));
        PetscCall(VecDestroy(&inv_dA));
      }

      PetscCall(MatGetOwnershipRange(A_ilu, &rstart, &rend));
      PetscCall(MatGetLocalSize(A_ilu, &m, NULL));
      PetscCall(MatGetSize(A_ilu, &M_size, NULL));
      PetscCall(MatGetType(A_ilu, &mtype));

      /* ── Preallocation pass for L, U, A_L_strict, A_U, R_L, R_U ─────────── */
      PetscInt *L_d_nnz, *L_o_nnz, *Ls_d_nnz, *Ls_o_nnz, *U_d_nnz, *U_o_nnz;
      PetscCall(PetscMalloc6(m, &L_d_nnz, m, &L_o_nnz,
                              m, &Ls_d_nnz, m, &Ls_o_nnz,
                              m, &U_d_nnz,  m, &U_o_nnz));

      for (PetscInt i = 0; i < m; i++) {
        PetscInt        gi = rstart + i;
        PetscInt        ncols;
        const PetscInt *cols;
        PetscCall(MatGetRow(A_ilu, gi, &ncols, &cols, NULL));
        PetscInt cstart, cend;
        PetscCall(MatGetOwnershipRangeColumn(A_ilu, &cstart, &cend));
        PetscInt ls_d = 0, ls_o = 0, u_d = 0, u_o = 0;
        for (PetscInt j = 0; j < ncols; j++) {
          PetscInt c = cols[j];
          if (c < gi) {
            if (c >= cstart && c < cend) ls_d++;
            else                         ls_o++;
          } else {
            if (c >= cstart && c < cend) u_d++;
            else                         u_o++;
          }
        }
        PetscCall(MatRestoreRow(A_ilu, gi, &ncols, &cols, NULL));
        Ls_d_nnz[i] = ls_d;   Ls_o_nnz[i] = ls_o;
        U_d_nnz[i]  = u_d;    U_o_nnz[i]  = u_o;
        L_d_nnz[i]  = ls_d + 1;  /* +1 for unit diagonal */
        L_o_nnz[i]  = ls_o;
      }

      PetscCall(CreateMatLike(PETSC_COMM_WORLD, mtype, m, m, M_size, M_size, L_d_nnz,  L_o_nnz,  &L));
      PetscCall(CreateMatLike(PETSC_COMM_WORLD, mtype, m, m, M_size, M_size, U_d_nnz,  U_o_nnz,  &U));
      PetscCall(CreateMatLike(PETSC_COMM_WORLD, mtype, m, m, M_size, M_size, Ls_d_nnz, Ls_o_nnz, &A_L_strict));
      PetscCall(CreateMatLike(PETSC_COMM_WORLD, mtype, m, m, M_size, M_size, U_d_nnz,  U_o_nnz,  &A_U));
      PetscCall(CreateMatLike(PETSC_COMM_WORLD, mtype, m, m, M_size, M_size, Ls_d_nnz, Ls_o_nnz, &R_L));
      PetscCall(CreateMatLike(PETSC_COMM_WORLD, mtype, m, m, M_size, M_size, U_d_nnz,  U_o_nnz,  &R_U));
      PetscCall(PetscFree6(L_d_nnz, L_o_nnz, Ls_d_nnz, Ls_o_nnz, U_d_nnz, U_o_nnz));

      /* ── Initialise L (unit diag + 0 on strict-lower) and U (A on upper) ─── */
      for (PetscInt i = 0; i < m; i++) {
        PetscInt           gi = rstart + i;
        PetscInt           ncols;
        const PetscInt    *cols;
        const PetscScalar *vals;
        PetscCall(MatGetRow(A_ilu, gi, &ncols, &cols, &vals));
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
        PetscCall(MatRestoreRow(A_ilu, gi, &ncols, &cols, &vals));
      }
      {
        Mat all[] = {L, U, A_L_strict, A_U, R_L, R_U};
        for (int k = 0; k < 6; k++) PetscCall(MatAssemblyBegin(all[k], MAT_FINAL_ASSEMBLY));
        for (int k = 0; k < 6; k++) PetscCall(MatAssemblyEnd  (all[k], MAT_FINAL_ASSEMBLY));
      }

      PetscReal A_norm;
      PetscCall(MatNorm(A_ilu, NORM_FROBENIUS, &A_norm));
      PetscReal threshold = parilu_tol * A_norm;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                            "||A||_F = %.6e, ParILU stencil-residual threshold = %.6e\n",
                            (double)A_norm, (double)threshold));

      PetscCall(MatCreateVecs(U, &inv_dU, NULL));

      /* ── Random RHS (shared across all solves) ───────────────────────────── */
      PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rnd));
      PetscCall(PetscRandomSetFromOptions(rnd));
      PetscCall(MatCreateVecs(A_ilu, &b_rand, &x_sol));
      PetscCall(VecSetRandom(b_rand, rnd));

      /* ── Baseline: A x = b  GMRES(30) + PCBJACOBI/ILU ──────────────────── */
      PetscCall(PetscLogStagePush(stage_A_pcilu));
      {
        KSP ksp_Apc;
        PC  pc_Apc;
        PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_Apc));
        PetscCall(KSPSetType(ksp_Apc, KSPGMRES));
        PetscCall(KSPGMRESSetRestart(ksp_Apc, 30));
        PetscCall(KSPSetNormType(ksp_Apc, KSP_NORM_UNPRECONDITIONED));
        PetscCall(KSPSetOperators(ksp_Apc, A_ilu, A_ilu));
        PetscCall(KSPSetTolerances(ksp_Apc, 1e-6, 1e-50, PETSC_DEFAULT, 2000));
        PetscCall(KSPGetPC(ksp_Apc, &pc_Apc));
        PetscCall(PCSetType(pc_Apc, PCBJACOBI));
        PetscCall(KSPSetOptionsPrefix(ksp_Apc, "Apc_"));
        PetscCall(KSPSetFromOptions(ksp_Apc));
        PetscCall(KSPSetUp(ksp_Apc));
        PetscCall(KSPSolve(ksp_Apc, b_rand, x_sol));
        PetscCall(ReportSolve("A x = b solve (gmres(30) + PCBJACOBI/ILU)",
                              ksp_Apc, A_ilu, b_rand, x_sol, &solves_converged));
        PetscCall(KSPDestroy(&ksp_Apc));
      }
      PetscCall(PetscLogStagePop());

      /* ── Baseline: A x = b  Richardson + PCAIR ──────────────────────────── */
      PetscCall(PetscLogStagePush(stage_A_air));
      {
        KSP ksp_Aair;
        PC  pc_Aair;
        PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_Aair));
        PetscCall(KSPSetType(ksp_Aair, KSPRICHARDSON));
        PetscCall(KSPSetNormType(ksp_Aair, KSP_NORM_UNPRECONDITIONED));
        PetscCall(KSPSetOperators(ksp_Aair, A_ilu, A_ilu));
        PetscCall(KSPSetTolerances(ksp_Aair, 1e-6, 1e-50, PETSC_DEFAULT, 2000));
        PetscCall(KSPGetPC(ksp_Aair, &pc_Aair));
        PetscCall(ConfigureInnerPC(pc_Aair, INNER_PC_AIR));
        PetscCall(KSPSetOptionsPrefix(ksp_Aair, "A_air_"));
        PetscCall(KSPSetFromOptions(ksp_Aair));
        PetscCall(KSPSetUp(ksp_Aair));
        {
          PetscReal cA = -1.0, sA = -1.0, gA = -1.0;
          PetscCall(PCAIRGetCycleComplexity(pc_Aair, &cA));
          PetscCall(PCAIRGetStorageComplexity(pc_Aair, &sA));
          PetscCall(PCAIRGetGridComplexity(pc_Aair, &gA));
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                                "AIR complexities (A): cycle=%.3f, storage=%.3f, grid=%.3f\n",
                                (double)cA, (double)sA, (double)gA));
        }
        PetscCall(KSPSolve(ksp_Aair, b_rand, x_sol));
        PetscCall(ReportSolve("A x = b solve (richardson + PCAIR)",
                              ksp_Aair, A_ilu, b_rand, x_sol, &solves_converged));
        PetscCall(KSPDestroy(&ksp_Aair));
      }
      PetscCall(PetscLogStagePop());

      /* ── ParILU sweep loop ───────────────────────────────────────────────── */
      PetscReal res_initial = -1.0;
      PetscCall(PetscLogStagePush(stage_parilu));
      for (sweep = 0; sweep < max_sweeps; sweep++) {
        if (sweep == 0) PetscCall(MatMatMult(L, U, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &M));
        else            PetscCall(MatMatMult(L, U, MAT_REUSE_MATRIX,   PETSC_DEFAULT, &M));

        PetscCall(MatCopy(A_L_strict, R_L, SAME_NONZERO_PATTERN));
        remove_from_sparse_match(M, R_L, 0, 1, -1.0);

        PetscCall(MatCopy(A_U, R_U, SAME_NONZERO_PATTERN));
        remove_from_sparse_match(M, R_U, 0, 1, -1.0);

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
      PetscCall(PetscLogStagePop());
      if (converged) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "ParILU converged in %" PetscInt_FMT " sweeps\n", sweep));
      } else {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "ParILU did NOT converge within %" PetscInt_FMT " sweeps\n", max_sweeps));
      }

      PetscCall(MatDestroy(&M));
      PetscCall(MatDestroy(&R_L));
      PetscCall(MatDestroy(&R_U));
      PetscCall(MatDestroy(&A_L_strict));
      PetscCall(MatDestroy(&A_U));
      PetscCall(VecDestroy(&inv_dU));

      if (parilu_diverged) {
        PetscCall(MatDestroy(&L));
        PetscCall(MatDestroy(&U));
        PetscCall(VecDestroy(&b_rand));
        PetscCall(VecDestroy(&x_sol));
        PetscCall(PetscRandomDestroy(&rnd));
        PetscCall(MatDestroy(&A_ilu));
        PetscCall(PetscFinalize());
        return 1;
      }

      /* ── Left-scale U by 1/diag(U_raw) ──────────────────────────────────── */
      PetscCall(MatCreateVecs(U, &inv_diag_U_raw, NULL));
      PetscCall(MatGetDiagonal(U, inv_diag_U_raw));
      PetscCall(VecReciprocal(inv_diag_U_raw));
      PetscCall(MatDiagonalScale(U, inv_diag_U_raw, NULL));

      PetscCall(VecDuplicate(inv_diag_U_raw, &inv_diag_U_raw_for_jac));
      PetscCall(VecCopy(inv_diag_U_raw, inv_diag_U_raw_for_jac));

      /* ── L solve: Richardson + PCAIR ─────────────────────────────────────── */
      PetscCall(PetscLogStagePush(stage_L_air));
      PetscCall(RunFactorSolve(PETSC_COMM_WORLD, L, b_rand, x_sol, INNER_PC_AIR,
                               "L solve (richardson + PCAIR)", "ilu_L_",
                               &solves_converged, &ksp_L_air));
      PetscCall(PetscLogStagePop());

      /* ── U solve: Richardson + PCAIR ─────────────────────────────────────── */
      PetscCall(PetscLogStagePush(stage_U_air));
      PetscCall(RunFactorSolve(PETSC_COMM_WORLD, U, b_rand, x_sol, INNER_PC_AIR,
                               "U solve (richardson + PCAIR)", "ilu_U_",
                               &solves_converged, &ksp_U_air));
      PetscCall(PetscLogStagePop());

      /* ── Ax=b: GMRES(30) + LU shell PC with PCAIR inner ─────────────────── */
      PetscCall(PetscLogStagePush(stage_A_shell));
      {
        KSP         ksp_A;
        PC          pc_A, pcL_inner, pcU_inner;
        LUShellCtx *shell_ctx;
        PetscReal   cL = -1.0, cU = -1.0, sL = -1.0, sU = -1.0, gL = -1.0, gU = -1.0;
        PetscCall(PetscNew(&shell_ctx));
        shell_ctx->ksp_L = ksp_L_air;
        shell_ctx->ksp_U = ksp_U_air;
        PetscCall(KSPSetType(shell_ctx->ksp_L, KSPPREONLY));
        PetscCall(KSPSetNormType(shell_ctx->ksp_L, KSP_NORM_NONE));
        PetscCall(KSPSetType(shell_ctx->ksp_U, KSPPREONLY));
        PetscCall(KSPSetNormType(shell_ctx->ksp_U, KSP_NORM_NONE));
        PetscCall(KSPSetUp(shell_ctx->ksp_L));
        PetscCall(KSPSetUp(shell_ctx->ksp_U));
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
        PetscCall(PetscOptionsGetInt(NULL, NULL, "-jac_max_it", &jac_max_it, NULL));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                              "AIR complexities: cycle L=%.3f U=%.3f, storage L=%.3f U=%.3f, grid L=%.3f U=%.3f\n",
                              (double)cL, (double)cU, (double)sL, (double)sU, (double)gL, (double)gU));
        PetscCall(MatCreateVecs(L, &shell_ctx->tmp, NULL));
        shell_ctx->inv_diag_U_raw = inv_diag_U_raw;
        inv_diag_U_raw            = NULL;

        PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_A));
        PetscCall(KSPSetType(ksp_A, KSPGMRES));
        PetscCall(KSPGMRESSetRestart(ksp_A, 30));
        PetscCall(KSPSetNormType(ksp_A, KSP_NORM_UNPRECONDITIONED));
        PetscCall(KSPSetOperators(ksp_A, A_ilu, A_ilu));
        PetscCall(KSPSetTolerances(ksp_A, 1e-6, 1e-50, PETSC_DEFAULT, 2000));
        PetscCall(KSPGetPC(ksp_A, &pc_A));
        PetscCall(PCSetType(pc_A, PCSHELL));
        PetscCall(PCShellSetContext(pc_A, shell_ctx));
        PetscCall(PCShellSetApply(pc_A, LUShellApply));
        PetscCall(PCShellSetDestroy(pc_A, LUShellDestroy));
        PetscCall(PCShellSetName(pc_A, "LU_AIR_shell"));
        PetscCall(KSPSetOptionsPrefix(ksp_A, "ilu_A_"));
        PetscCall(KSPSetFromOptions(ksp_A));
        PetscCall(KSPSetUp(ksp_A));
        PetscCall(KSPSolve(ksp_A, b_rand, x_sol));
        PetscCall(ReportSolve("A x = b solve (gmres(30) + LU shell PC, AIRG inner)",
                              ksp_A, A_ilu, b_rand, x_sol, &solves_converged));
        PetscCall(KSPDestroy(&ksp_A));
      }
      PetscCall(PetscLogStagePop());

      /* ── L/U standalone solves: Richardson + {GMRES poly, Neumann poly,
            ISAI, Jacobi} inner PC, mirroring ilu_factors.c ─────────────────── */
      PetscCall(PetscLogStagePush(stage_L_gmres));
      PetscCall(RunFactorSolve(PETSC_COMM_WORLD, L, b_rand, x_sol, INNER_PC_GMRES_POLY,
                               "L solve (richardson + GMRES poly)", "ilu_L_gmres_",
                               &solves_converged, NULL));
      PetscCall(PetscLogStagePop());

      PetscCall(PetscLogStagePush(stage_U_gmres));
      PetscCall(RunFactorSolve(PETSC_COMM_WORLD, U, b_rand, x_sol, INNER_PC_GMRES_POLY,
                               "U solve (richardson + GMRES poly)", "ilu_U_gmres_",
                               &solves_converged, NULL));
      PetscCall(PetscLogStagePop());

      PetscCall(PetscLogStagePush(stage_L_neumann));
      PetscCall(RunFactorSolve(PETSC_COMM_WORLD, L, b_rand, x_sol, INNER_PC_NEUMANN_POLY,
                               "L solve (richardson + Neumann poly)", "ilu_L_neumann_",
                               &solves_converged, NULL));
      PetscCall(PetscLogStagePop());

      PetscCall(PetscLogStagePush(stage_U_neumann));
      PetscCall(RunFactorSolve(PETSC_COMM_WORLD, U, b_rand, x_sol, INNER_PC_NEUMANN_POLY,
                               "U solve (richardson + Neumann poly)", "ilu_U_neumann_",
                               &solves_converged, NULL));
      PetscCall(PetscLogStagePop());

      PetscCall(PetscLogStagePush(stage_L_isai));
      PetscCall(RunFactorSolve(PETSC_COMM_WORLD, L, b_rand, x_sol, INNER_PC_ISAI,
                               "L solve (richardson + ISAI)", "ilu_L_isai_",
                               &solves_converged, NULL));
      PetscCall(PetscLogStagePop());

      PetscCall(PetscLogStagePush(stage_U_isai));
      PetscCall(RunFactorSolve(PETSC_COMM_WORLD, U, b_rand, x_sol, INNER_PC_ISAI,
                               "U solve (richardson + ISAI)", "ilu_U_isai_",
                               &solves_converged, NULL));
      PetscCall(PetscLogStagePop());

      PetscCall(PetscLogStagePush(stage_L_jac));
      PetscCall(RunFactorSolve(PETSC_COMM_WORLD, L, b_rand, x_sol, INNER_PC_JACOBI,
                               "L solve (richardson + PCJACOBI)", "ilu_L_jac_",
                               &solves_converged, NULL));
      PetscCall(PetscLogStagePop());

      PetscCall(PetscLogStagePush(stage_U_jac));
      PetscCall(RunFactorSolve(PETSC_COMM_WORLD, U, b_rand, x_sol, INNER_PC_JACOBI,
                               "U solve (richardson + PCJACOBI)", "ilu_U_jac_",
                               &solves_converged, NULL));
      PetscCall(PetscLogStagePop());

      /* ── Ax=b: GMRES(30) + LU shell PC with Jacobi inner ─────────────────── */
      PetscCall(PetscLogStagePush(stage_A_shell_jac));
      {
        KSP         ksp_Ajac;
        PC          pc_Ajac;
        LUShellCtx *shell_ctx;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Jacobi inner max_it=%" PetscInt_FMT "\n", jac_max_it));
        PetscCall(PetscNew(&shell_ctx));
        PetscCall(CreateInnerKSP(PETSC_COMM_WORLD, L, "ilu_A_jac_pc_L_", INNER_PC_JACOBI, jac_max_it, &shell_ctx->ksp_L));
        PetscCall(CreateInnerKSP(PETSC_COMM_WORLD, U, "ilu_A_jac_pc_U_", INNER_PC_JACOBI, jac_max_it, &shell_ctx->ksp_U));
        PetscCall(MatCreateVecs(L, &shell_ctx->tmp, NULL));
        shell_ctx->inv_diag_U_raw = inv_diag_U_raw_for_jac;
        inv_diag_U_raw_for_jac    = NULL;

        PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_Ajac));
        PetscCall(KSPSetType(ksp_Ajac, KSPGMRES));
        PetscCall(KSPGMRESSetRestart(ksp_Ajac, 30));
        PetscCall(KSPSetNormType(ksp_Ajac, KSP_NORM_UNPRECONDITIONED));
        PetscCall(KSPSetOperators(ksp_Ajac, A_ilu, A_ilu));
        PetscCall(KSPSetTolerances(ksp_Ajac, 1e-6, 1e-50, PETSC_DEFAULT, 2000));
        PetscCall(KSPGetPC(ksp_Ajac, &pc_Ajac));
        PetscCall(PCSetType(pc_Ajac, PCSHELL));
        PetscCall(PCShellSetContext(pc_Ajac, shell_ctx));
        PetscCall(PCShellSetApply(pc_Ajac, LUShellApply));
        PetscCall(PCShellSetDestroy(pc_Ajac, LUShellDestroy));
        PetscCall(PCShellSetName(pc_Ajac, "LU_Jacobi_shell"));
        PetscCall(KSPSetOptionsPrefix(ksp_Ajac, "ilu_A_jac_"));
        PetscCall(KSPSetFromOptions(ksp_Ajac));
        PetscCall(KSPSetUp(ksp_Ajac));
        PetscCall(KSPSolve(ksp_Ajac, b_rand, x_sol));
        PetscCall(ReportSolve("A x = b solve (gmres(30) + LU shell PC, Jacobi inner)",
                              ksp_Ajac, A_ilu, b_rand, x_sol, &solves_converged));
        PetscCall(KSPDestroy(&ksp_Ajac));
      }
      PetscCall(PetscLogStagePop());

      /* ── Cleanup ─────────────────────────────────────────────────────────── */
      PetscCall(MatDestroy(&L));
      PetscCall(MatDestroy(&U));
      PetscCall(VecDestroy(&b_rand));
      PetscCall(VecDestroy(&x_sol));
      PetscCall(PetscRandomDestroy(&rnd));
      PetscCall(MatDestroy(&A_ilu));
      PetscCall(PetscFinalize());
      return (converged && solves_converged) ? 0 : 1;

      /* suppress unused-variable warnings when PETSC_USE_LOG is off */
      (void)npe;
    } /* end if (ilu_test) */
  } /* end ilu_test scope */

  PetscCall(PetscLogStageRegister("Setup", &setup));
  PetscCall(PetscLogStageRegister("GPU copy stage - triggered by a prelim KSPSolve", &gpu_copy));

  // Dimension - default to 2D
  dim = 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));
  PetscCheck(dim == 2 || dim == 3, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "dim must be 2 or 3");

  // Dimensions of box - default to [0, 1]^dim
  L_x = 1.0; L_y = 1.0; L_z = 1.0;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-L_x", &L_x_test, &option_found_u));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-L_y", &L_y_test, &option_found_v));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-L_z", &L_z_test, &option_found_w));

  if (option_found_u) PetscCheck(L_x_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "L_x must be positive");
  if (option_found_v) PetscCheck(L_y_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "L_y must be positive");
  if (option_found_w) PetscCheck(L_z_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "L_z must be positive");

  if (option_found_u) L_x = L_x_test;
  if (option_found_v) L_y = L_y_test;
  if (option_found_w) L_z = L_z_test;

  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  if (dim == 2) {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                           11, 11, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  } else {
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                           11, 11, 11, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, NULL, &da));
  }
  PetscCall(DMSetFromOptions(da));
  // This stops the zero entries in the stencil from being added to the matrix
  // It still allocates the memory for a 5/7 point stencil but the sparsity doesn't include the zero entries
  // We do this instead of calling MatFilter as there is no Kokkos implementation so its very slow
  PetscCall(DMSetMatrixPreallocateOnly(da,PETSC_TRUE));
  PetscCall(DMSetUp(da));
  if (dim == 2) {
    PetscCall(DMDASetUniformCoordinates(da, 0.0, L_x, 0.0, L_y, 0.0, 0.0));
  } else {
    PetscCall(DMDASetUniformCoordinates(da, 0.0, L_x, 0.0, L_y, 0.0, L_z));
  }
  PetscCall(KSPSetDM(ksp,(DM)da));
  // We generate the matrix ourselves
  PetscCall(KSPSetDMActive(ksp, KSP_DMACTIVE_ALL, PETSC_FALSE));

  // Create empty matrix and vectors
  PetscCall(DMCreateMatrix(da, &A));
  PetscCall(MatSetOption(A,MAT_IGNORE_ZERO_ENTRIES ,PETSC_TRUE));
  PetscCall(DMCreateGlobalVector(da, &x));
  PetscCall(PetscObjectSetName((PetscObject)x, "adv_diff"));
  PetscCall(DMCreateGlobalVector(da, &b));

  // Zero rhs
  PetscCall(VecSet(b, 0.0));

  // ~~~~~~~~~~~~~~
  // Get command line options
  // ~~~~~~~~~~~~~~

  PetscBool second_solve= PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-second_solve", &second_solve, NULL));

  // Advection velocities
  // Default in 2D/3D: (1,1,1), normalised when unit_velocity is true
  // If -theta is provided, set (u,v,w) = (cos(theta), sin(theta), 0)
  PetscReal pi = 4*atan(1.0);
  theta = pi/4.0;
  PetscBool theta_found;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-theta", &theta, &theta_found));
  u = 1.0;
  v = 1.0;
  w = 1.0;
  if (theta_found) {
    PetscCheck(theta <= pi/2.0 && theta >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "Theta must be between 0 and pi/2");
    u = cos(theta);
    v = sin(theta);
    w = 0.0;
  }

  // Or the user can pass in the individual advection velocities
  // This will override theta and unit velocity options if they are set
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-u", &u_test, &option_found_u));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-v", &v_test, &option_found_v));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-w", &w_test, &option_found_w));

  if (option_found_u) PetscCheck(u_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "u must be positive");
  if (option_found_v) PetscCheck(v_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "v must be positive");
  if (option_found_w) PetscCheck(w_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "w must be positive");

  if (option_found_u) u = u_test;
  if (option_found_v) v = v_test;
  if (option_found_w) w = w_test;

  // Curved velocity field option (u(x,y) = y, v(x,y) = 1-x)
  // Default is straight line (false)
  curved_velocity = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-curved_velocity", &curved_velocity, NULL));

  // Unit velocity normalization option - default to unit velocity
  unit_velocity = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-unit_velocity", &unit_velocity, NULL));

  // Don't normalise if user has explicitly set a velocity
  if (option_found_u || option_found_v || option_found_w) {
    unit_velocity = PETSC_FALSE;
  }

  // Diffusion coefficient
  // Default alpha is 0 - pure advection
  alpha = 0.0;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-alpha", &alpha, NULL));

  // If we just have advection, rather than scaling by Hx * Hy, we can just have 
  // a dimensionless advection problem - this is enabled by default
  // If we have any diffusion this is turned off by default
  adv_nondim = PETSC_TRUE;
  if (alpha != 0.0)
  {
   adv_nondim = PETSC_FALSE;
  }  
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-adv_nondim", &adv_nondim, NULL));
  // We can only nondimensionalise the advection if we don't have any diffusion
  check_nondim = PETSC_TRUE;
  if (alpha != 0.0)
  {
   if (adv_nondim)
   {
      check_nondim = PETSC_FALSE;
   }
  }

  PetscCheck(check_nondim, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "Non-dimensional advection only applies without diffusion");

  // Do we diagonally scale our matrix before solving
  // Defaults to false
  diag_scale = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-diag_scale", &diag_scale, NULL));

  // Set only the bottom face inflow to u=1, all other inflow faces to u=0
  // In 2D: bottom face is j==0; in 3D: bottom face is k==0 (z=0)
  // Defaults to false (all inflow faces are u=0)
  bottom_only_inflow_one = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-bottom_only_inflow_one", &bottom_only_inflow_one, NULL));

  // If bottom_only_inflow_one is set, set the RHS to 1 on the bottom boundary
  if (bottom_only_inflow_one) {
    PetscInt xs_b, ys_b, zs_b, xm_b, ym_b, zm_b;
    if (dim == 2) {
      PetscScalar **b_arr;
      PetscCall(DMDAGetCorners(da, &xs_b, &ys_b, NULL, &xm_b, &ym_b, NULL));
      PetscCall(DMDAVecGetArray(da, b, &b_arr));
      if (ys_b == 0) {
        for (PetscInt i_b = xs_b; i_b < xs_b + xm_b; i_b++) b_arr[0][i_b] = 1.0;
      }
      PetscCall(DMDAVecRestoreArray(da, b, &b_arr));
    } else {
      PetscScalar ***b_arr;
      PetscCall(DMDAGetCorners(da, &xs_b, &ys_b, &zs_b, &xm_b, &ym_b, &zm_b));
      PetscCall(DMDAVecGetArray(da, b, &b_arr));
      if (zs_b == 0) {
        for (PetscInt j_b = ys_b; j_b < ys_b + ym_b; j_b++)
          for (PetscInt i_b = xs_b; i_b < xs_b + xm_b; i_b++)
            b_arr[0][j_b][i_b] = 1.0;
      }
      PetscCall(DMDAVecRestoreArray(da, b, &b_arr));
    }
  }

  // ~~~~~~~~~~~~~~
  // ~~~~~~~~~~~~~~

  // Compute our matrix
  PetscCall(ComputeMat(da, A, dim, u, v, w, L_x, L_y, L_z, alpha, adv_nondim, curved_velocity, unit_velocity));
  // This will compress out the extra memory
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &A_temp));
  PetscCall(MatDestroy(&A));
  A = A_temp;

  // Set the operator and options
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));

  // Diagonally scale our matrix 
  if (diag_scale) {
   PetscCall(VecDuplicate(x, &diag_vec));
   PetscCall(MatGetDiagonal(A, diag_vec));
   PetscCall(VecReciprocal(diag_vec));
   PetscCall(MatDiagonalScale(A, diag_vec, PETSC_NULLPTR));
   PetscCall(VecPointwiseMult(b, diag_vec, b));
   PetscCall(VecDestroy(&diag_vec));
  }

  // Setup the ksp
  PetscCall(PetscLogStagePush(setup));
  PetscCall(KSPSetUp(ksp));
  PetscCall(PetscLogStagePop());

  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(VecSet(x, 1.0));

  // Do a preliminary KSPSolve so all the vecs and mats get copied to the gpu
  // before the solve we're trying to time
  PetscCall(PetscLogStagePush(gpu_copy));
  PetscCall(KSPSolve(ksp,b,x));
  PetscCall(PetscLogStagePop());

  // Solve
  // We set x to 1 rather than random as the vecrandom doesn't yet have a
  // gpu implementation and we don't want a copy occuring back to the cpu
  if (second_solve)
  {
   PetscCall(VecSet(x, 1.0));
   PetscCall(KSPSolve(ksp,b,x));
  }

  // Write out the iteration count
  PetscCall(KSPGetConvergedReason(ksp,&reason));

  // Optionally write the solution to a VTK structured grid file.
  // Use -vec_view vtk:solution.vts on the command line to enable.
  PetscCall(VecViewFromOptions(x, NULL, "-vec_view"));
   
  // ~~~~~~~~~~~~~~
  // ~~~~~~~~~~~~~~  

  // Cleanup
  PetscCall(DMDestroy(&da));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  if (reason < 0)
  {
   return 1;
  }
  return 0;
}

PetscErrorCode ComputeMat(DM da, Mat A, PetscInt dim,
                          PetscScalar u, PetscScalar v, PetscScalar w,
                          PetscScalar L_x, PetscScalar L_y, PetscScalar L_z,
                          PetscScalar alpha, PetscBool adv_nondim,
                          PetscBool curved_velocity, PetscBool unit_velocity)
{
  PetscInt    i, j, k, M, N, P, xs, ys, zs, xm, ym, zm;
  PetscScalar val[7], Hx, Hy, Hz;
  PetscScalar HxdHy, HydHx, adv_x_scale, adv_y_scale;
  PetscScalar HyHz_Hx, HxHz_Hy, HxHy_Hz, adv_yz_scale, adv_xz_scale, adv_xy_scale;
  MatStencil  row, col[7];

  PetscCall(DMDAGetInfo(da, NULL, &M, &N, &P, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  PetscCheck(M > 1 && N > 1 && (dim == 2 || P > 1), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
             "Grid dimensions must be at least 2 points in each active direction");

  // Node-centred spacing: domain length divided by number of intervals
  Hx = L_x / (PetscReal)(M - 1);
  Hy = L_y / (PetscReal)(N - 1);
  Hz = (dim == 3) ? L_z / (PetscReal)(P - 1) : 1.0;

  HxdHy = Hx / Hy;
  HydHx = Hy / Hx;

  // 2D advection scale factors: multiply upwinded equation by Hx*Hy
  //   adv_y_scale goes with u (west neighbour, x-direction upwinding)
  //   adv_x_scale goes with v (south neighbour, y-direction upwinding)
  // For dimensionless pure advection (adv_nondim), divide through by Hx
  adv_x_scale = Hx;
  adv_y_scale = Hy;
  if (adv_nondim) {
    adv_x_scale = 1.0;
    adv_y_scale = HydHx;
  }

  // 3D advection scale factors: multiply upwinded equation by Hx*Hy*Hz
  //   adv_yz_scale goes with u (west neighbour)
  //   adv_xz_scale goes with v (south neighbour)
  //   adv_xy_scale goes with w (bottom neighbour)
  // For dimensionless pure advection (adv_nondim), divide through by Hx
  HyHz_Hx    = Hy * Hz / Hx;
  HxHz_Hy    = Hx * Hz / Hy;
  HxHy_Hz    = Hx * Hy / Hz;
  adv_yz_scale = Hy * Hz;
  adv_xz_scale = Hx * Hz;
  adv_xy_scale = Hx * Hy;
  if (adv_nondim) {
    // For 3D to be dimensionless like 2D, we divide by Hx*Hx
    adv_yz_scale = HyHz_Hx / Hx;
    adv_xz_scale = Hz / Hx;
    adv_xy_scale = Hy / Hx;
  }

  if (dim == 2) {

    PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

    for (j = ys; j < ys+ym; j++) {
      for (i = xs; i < xs+xm; i++) {
        row.i = i; row.j = j;

        // Compute velocity at this node
        PetscReal x_node[3] = {i * Hx, j * Hy, 0.0};
        PetscReal vel[3];
        GetVelocity(dim, (PetscReal)u, (PetscReal)v, (PetscReal)w, x_node, curved_velocity, unit_velocity, vel);
        PetscScalar u_loc = vel[0], v_loc = vel[1];

        // Boundary values
        if (i==0 || j==0 || i==M-1 || j==N-1) {

          // Dirichlets left and bottom
          if (i==0 || j==0) {
            val[0] = 1.0; col[0].i = i; col[0].j = j;
            PetscCall(MatSetValuesStencil(A, 1, &row, 1, col, val, ADD_VALUES));

          // Top or right - depends on if we have any diffusion
          } else {

            // If we have no diffusion, the top and right nodes just get the normal
            // upwinded stencil, representing an outflow bc
            if (alpha == 0.0) {
              // Upwind advection with theta between 0 and pi/2
              // south (lower y)
              val[0] = -v_loc * adv_x_scale;                  col[0].i = i;   col[0].j = j-1;
              // west (lower x)
              val[1] = -u_loc * adv_y_scale;                  col[1].i = i-1; col[1].j = j;
              // centre
              val[2] = u_loc*adv_y_scale + v_loc*adv_x_scale; col[2].i = i;   col[2].j = j;
              PetscCall(MatSetValuesStencil(A, 1, &row, 3, col, val, ADD_VALUES));

            // If we have diffusion we have dirichlet bcs on the top and right
            } else {
              val[0] = 1.0; col[0].i = i; col[0].j = j;
              PetscCall(MatSetValuesStencil(A, 1, &row, 1, col, val, ADD_VALUES));
            }
          }

        // interior stencil
        } else {

          // If we have diffusion
          if (alpha != 0.0) {
            // south
            val[0] = -alpha * HxdHy;              col[0].i = i;   col[0].j = j-1;
            // west
            val[1] = -alpha * HydHx;              col[1].i = i-1; col[1].j = j;
            // centre
            val[2] =  alpha * 2.0*(HxdHy+HydHx);  col[2].i = i;   col[2].j = j;
            // east
            val[3] = -alpha * HydHx;              col[3].i = i+1; col[3].j = j;
            // north
            val[4] = -alpha * HxdHy;              col[4].i = i;   col[4].j = j+1;
            PetscCall(MatSetValuesStencil(A, 1, &row, 5, col, val, ADD_VALUES));
          }

          // Upwind advection with theta between 0 and pi/2
          if (u_loc != 0.0 || v_loc != 0.0) {
            // south (lower y)
            val[0] = -v_loc * adv_x_scale;                  col[0].i = i;   col[0].j = j-1;
            // west (lower x)
            val[1] = -u_loc * adv_y_scale;                  col[1].i = i-1; col[1].j = j;
            // centre
            val[2] = u_loc*adv_y_scale + v_loc*adv_x_scale; col[2].i = i;   col[2].j = j;
            PetscCall(MatSetValuesStencil(A, 1, &row, 3, col, val, ADD_VALUES));
          }
        }
      }
    }

  } else { // dim == 3

    PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));

    for (k = zs; k < zs+zm; k++) {
      for (j = ys; j < ys+ym; j++) {
        for (i = xs; i < xs+xm; i++) {
          row.i = i; row.j = j; row.k = k;

          // Compute velocity at this node
          PetscReal x_node[3] = {i * Hx, j * Hy, k * Hz};
          PetscReal vel[3];
          GetVelocity(dim, (PetscReal)u, (PetscReal)v, (PetscReal)w, x_node, curved_velocity, unit_velocity, vel);
          PetscScalar u_loc = vel[0], v_loc = vel[1], w_loc = vel[2];

          // Boundary values
          if (i==0 || j==0 || k==0 || i==M-1 || j==N-1 || k==P-1) {

            // Dirichlets west, south and bottom (inflow faces)
            if (i==0 || j==0 || k==0) {
              val[0] = 1.0; col[0].i = i; col[0].j = j; col[0].k = k;
              PetscCall(MatSetValuesStencil(A, 1, &row, 1, col, val, ADD_VALUES));

            // East, north or top - depends on if we have any diffusion
            } else {

              // If we have no diffusion, the outflow nodes just get the normal
              // upwinded stencil, representing an outflow bc
              if (alpha == 0.0) {
                // Upwind advection with velocity in [0,1]^3 quadrant
                // bottom (lower z)
                val[0] = -w_loc * adv_xy_scale;                                        col[0].i = i;   col[0].j = j;   col[0].k = k-1;
                // south (lower y)
                val[1] = -v_loc * adv_xz_scale;                                        col[1].i = i;   col[1].j = j-1; col[1].k = k;
                // west (lower x)
                val[2] = -u_loc * adv_yz_scale;                                        col[2].i = i-1; col[2].j = j;   col[2].k = k;
                // centre
                val[3] = u_loc*adv_yz_scale + v_loc*adv_xz_scale + w_loc*adv_xy_scale; col[3].i = i;   col[3].j = j;   col[3].k = k;
                PetscCall(MatSetValuesStencil(A, 1, &row, 4, col, val, ADD_VALUES));

              // If we have diffusion we have dirichlet bcs on the east, north and top
              } else {
                val[0] = 1.0; col[0].i = i; col[0].j = j; col[0].k = k;
                PetscCall(MatSetValuesStencil(A, 1, &row, 1, col, val, ADD_VALUES));
              }
            }

          // interior stencil
          } else {

            // If we have diffusion
            if (alpha != 0.0) {
              // bottom
              val[0] = -alpha * HxHy_Hz; col[0].i = i;   col[0].j = j;   col[0].k = k-1;
              // south
              val[1] = -alpha * HxHz_Hy; col[1].i = i;   col[1].j = j-1; col[1].k = k;
              // west
              val[2] = -alpha * HyHz_Hx; col[2].i = i-1; col[2].j = j;   col[2].k = k;
              // centre
              val[3] =  alpha * 2.0*(HyHz_Hx + HxHz_Hy + HxHy_Hz); col[3].i = i; col[3].j = j; col[3].k = k;
              // east
              val[4] = -alpha * HyHz_Hx; col[4].i = i+1; col[4].j = j;   col[4].k = k;
              // north
              val[5] = -alpha * HxHz_Hy; col[5].i = i;   col[5].j = j+1; col[5].k = k;
              // top
              val[6] = -alpha * HxHy_Hz; col[6].i = i;   col[6].j = j;   col[6].k = k+1;
              PetscCall(MatSetValuesStencil(A, 1, &row, 7, col, val, ADD_VALUES));
            }

            // Upwind advection with velocity in [0,1]^3 quadrant
            if (u_loc != 0.0 || v_loc != 0.0 || w_loc != 0.0) {
              // bottom (lower z)
              val[0] = -w_loc * adv_xy_scale;                                        col[0].i = i;   col[0].j = j;   col[0].k = k-1;
              // south (lower y)
              val[1] = -v_loc * adv_xz_scale;                                        col[1].i = i;   col[1].j = j-1; col[1].k = k;
              // west (lower x)
              val[2] = -u_loc * adv_yz_scale;                                        col[2].i = i-1; col[2].j = j;   col[2].k = k;
              // centre
              val[3] = u_loc*adv_yz_scale + v_loc*adv_xz_scale + w_loc*adv_xy_scale; col[3].i = i;   col[3].j = j;   col[3].k = k;
              PetscCall(MatSetValuesStencil(A, 1, &row, 4, col, val, ADD_VALUES));
            }
          }
        }
      }
    }

  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  return PETSC_SUCCESS;
}
