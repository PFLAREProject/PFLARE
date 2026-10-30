/*   DMDA/TS solving time dependent 2D advection with upwind finite differences
     and implicit (adaptive) time stepping.

     Integrates the DAE   M du/dt + L u = b
     where L is the dimensionless upwind advection operator from adv_diff_fd (ie the
     upwinded equation multiplied by Hx), with Dirichlet rows on the inflow faces, so
     M = Hx on every non-Dirichlet row and M = 0 on the Dirichlet rows.
     The implicit Jacobian is therefore J = shift * M + L, so changing the time step
     only changes the diagonal of J.

     BCs: bottom (j=0) and left (i=0) faces dirichlet inflow u=1, top and right outflow.
     The initial condition is u=0, so a front advects through the domain until it
     reaches the steady state u=1.

     ./adv_diff_fd_ts -pc_type air -ts_adapt_type basic
             : backward Euler with adaptive time steps on a 2D grid of size 11 x 11

     Can control grid size with -da_grid_x, -da_grid_y (default 11)
     Can control the direction of advection with -theta (pi/4 default)
     Timestep / step count / final time / adaptivity are all controlled by the standard
     PETSc TS options, e.g., -ts_type, -ts_time_step, -ts_max_time, -ts_adapt_type
     The default is backward Euler, dt = 1e-3 and final time 2.0
     Can print the dt and the iteration count of every step with -print_steps
     Can error if the total number of linear iterations is above -max_total_its (default -1, don't check)
     Can check the final solution is within 1e-3 of the steady state u=1 with -verify_solution

     The preconditioner behaviour across steps is controlled by standard options:
       rebuilt every time J changes : the default
       never rebuilt (frozen)       : -snes_lag_preconditioner -2 -snes_lag_preconditioner_persists
       updated with the diagonal    : -pc_air_diag_update
*/

static char help[] = "Solves time dependent 2D advection with finite differences and adaptive time stepping.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>

#include "pflare.h"

typedef struct {
  Mat       L;
  Vec       m_diag, b, work;
  PetscReal shift;      // the shift currently in J
  PetscBool print_steps;
  PetscInt  its_last;
} AppCtx;

static PetscErrorCode ComputeL(DM da, Mat L, PetscReal u, PetscReal v, Vec m_diag, Vec b)
{
  PetscInt      i, j, mx, my, xs, ys, xm, ym;
  PetscScalar   vals[3], **m_arr, **b_arr;
  MatStencil    row, cols[3];
  PetscReal     Hx, Hy;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da, 0, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0));
  Hx = 1.0 / (PetscReal)(mx - 1);
  Hy = 1.0 / (PetscReal)(my - 1);

  PetscCall(DMDAVecGetArray(da, m_diag, &m_arr));
  PetscCall(DMDAVecGetArray(da, b, &b_arr));
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      row.i = i; row.j = j;
      // Dirichlet inflow of one on the left and bottom
      if (i == 0 || j == 0) {
        cols[0] = row; vals[0] = 1.0;
        PetscCall(MatSetValuesStencil(L, 1, &row, 1, cols, vals, INSERT_VALUES));
        m_arr[j][i] = 0.0;
        b_arr[j][i] = 1.0;
      // Upwinded stencil everywhere else, including the outflow faces
      } else {
        cols[0].i = i;     cols[0].j = j - 1; vals[0] = -v;
        cols[1].i = i - 1; cols[1].j = j;     vals[1] = -u * Hy / Hx;
        cols[2].i = i;     cols[2].j = j;     vals[2] = u * Hy / Hx + v;
        PetscCall(MatSetValuesStencil(L, 1, &row, 3, cols, vals, INSERT_VALUES));
        m_arr[j][i] = Hx;
        b_arr[j][i] = 0.0;
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da, m_diag, &m_arr));
  PetscCall(DMDAVecRestoreArray(da, b, &b_arr));
  PetscCall(MatAssemblyBegin(L, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(L, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// F = M udot + L u - b
static PetscErrorCode IFunction(TS ts, PetscReal t, Vec u, Vec udot, Vec F, void *ctx_void)
{
  AppCtx *ctx = (AppCtx *)ctx_void;

  PetscFunctionBeginUser;
  (void)ts; (void)t;
  PetscCall(MatMult(ctx->L, u, F));
  PetscCall(VecPointwiseMult(ctx->work, ctx->m_diag, udot));
  PetscCall(VecAXPY(F, 1.0, ctx->work));
  PetscCall(VecAXPY(F, -1.0, ctx->b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// J = shift * M + L
// We only touch J if the shift has changed, so the PC only sees a change to
// the Pmat when the time step changes
static PetscErrorCode IJacobian(TS ts, PetscReal t, Vec u, Vec udot, PetscReal shift, Mat J, Mat Jpre, void *ctx_void)
{
  AppCtx *ctx = (AppCtx *)ctx_void;

  PetscFunctionBeginUser;
  (void)ts; (void)t; (void)u; (void)udot; (void)Jpre;
  if (shift == ctx->shift) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatCopy(ctx->L, J, SAME_NONZERO_PATTERN));
  PetscCall(VecCopy(ctx->m_diag, ctx->work));
  PetscCall(VecScale(ctx->work, shift));
  PetscCall(MatDiagonalSet(J, ctx->work, ADD_VALUES));
  ctx->shift = shift;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Prints the time step and the linear iterations of each step
static PetscErrorCode MonitorSteps(TS ts, PetscInt step, PetscReal time, Vec u, void *ctx_void)
{
  AppCtx   *ctx = (AppCtx *)ctx_void;
  PetscInt  its;
  PetscReal dt;

  PetscFunctionBeginUser;
  (void)u;
  PetscCall(TSGetKSPIterations(ts, &its));
  PetscCall(TSGetTimeStep(ts, &dt));
  if (ctx->print_steps && step > 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "step %4" PetscInt_FMT " time %10.4e next dt %10.4e its %4" PetscInt_FMT "\n", step, (double)time, (double)dt, its - ctx->its_last));
  ctx->its_last = its;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  TS                ts;
  DM                da;
  Mat               J;
  Vec               x;
  AppCtx            ctx;
  PetscInt          steps, rejects, its, max_total_its = -1;
  PetscReal         theta = PETSC_PI / 4.0;
  PetscBool         verify_solution = PETSC_FALSE;
  TSConvergedReason ts_reason;
  PetscLogDouble    time_start, time_end;

  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PCRegister_PFLARE();

  ctx.print_steps = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-print_steps", &ctx.print_steps, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-theta", &theta, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-max_total_its", &max_total_its, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-verify_solution", &verify_solution, NULL));
  PetscCheck(theta <= PETSC_PI / 2.0 && theta >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "Theta must be between 0 and pi/2");

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 11, 11, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  // Only the nonzeros we set go in the sparsity
  PetscCall(DMSetMatrixPreallocateOnly(da, PETSC_TRUE));
  PetscCall(DMSetUp(da));

  PetscCall(DMCreateMatrix(da, &ctx.L));
  PetscCall(DMCreateGlobalVector(da, &x));
  PetscCall(VecDuplicate(x, &ctx.m_diag));
  PetscCall(VecDuplicate(x, &ctx.b));
  PetscCall(VecDuplicate(x, &ctx.work));
  PetscCall(ComputeL(da, ctx.L, PetscCosReal(theta), PetscSinReal(theta), ctx.m_diag, ctx.b));
  PetscCall(MatDuplicate(ctx.L, MAT_COPY_VALUES, &J));
  ctx.shift    = 0.0;
  ctx.its_last = 0;

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetDM(ts, da));
  PetscCall(TSSetType(ts, TSBEULER));
  PetscCall(TSSetProblemType(ts, TS_LINEAR));
  PetscCall(TSSetEquationType(ts, TS_EQ_DAE_IMPLICIT_INDEX1));
  PetscCall(TSSetIFunction(ts, NULL, IFunction, &ctx));
  PetscCall(TSSetIJacobian(ts, J, J, IJacobian, &ctx));
  PetscCall(TSSetTimeStep(ts, 1e-3));
  PetscCall(TSSetMaxTime(ts, 2.0));
  PetscCall(TSSetMaxSteps(ts, 100000));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSMonitorSet(ts, MonitorSteps, &ctx, NULL));
  PetscCall(TSSetFromOptions(ts));

  // Zero initial condition
  PetscCall(VecSet(x, 0.0));
  PetscCall(PetscTime(&time_start));
  PetscCall(TSSolve(ts, x));
  PetscCall(PetscTime(&time_end));

  PetscCall(TSGetConvergedReason(ts, &ts_reason));
  PetscCall(TSGetStepNumber(ts, &steps));
  PetscCall(TSGetStepRejections(ts, &rejects));
  PetscCall(TSGetKSPIterations(ts, &its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Steps %" PetscInt_FMT ", rejected steps %" PetscInt_FMT ", total linear iterations %" PetscInt_FMT ", average its per solve %.2f, TSSolve time %.4e s\n", steps, rejects, its, (double)its / (double)(steps + rejects), (double)(time_end - time_start)));

  if (verify_solution) {
    Vec       x_ones;
    PetscReal err_norm;
    PetscCall(VecDuplicate(x, &x_ones));
    PetscCall(VecSet(x_ones, 1.0));
    PetscCall(VecAXPY(x_ones, -1.0, x));
    PetscCall(VecNorm(x_ones, NORM_INFINITY, &err_norm));
    PetscCall(VecDestroy(&x_ones));
    PetscCheck(err_norm < 1e-3, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Solution verification FAILED: max|u - 1| = %g > 1e-3", (double)err_norm);
  }

  PetscCall(TSDestroy(&ts));
  PetscCall(MatDestroy(&J));
  PetscCall(MatDestroy(&ctx.L));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&ctx.m_diag));
  PetscCall(VecDestroy(&ctx.b));
  PetscCall(VecDestroy(&ctx.work));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  if (ts_reason < 0) return 1;
  if (max_total_its >= 0 && its > max_total_its) return 1;
  return 0;
}
