/*   Tests PCAIRSetDiagonalUpdate on a series of diagonally perturbed systems
     (L + D_g) x = b, where L is a steady 2D upwind finite difference advection(-diffusion)
     operator and D_g is diagonal, D_g = c_g (1 + 0.5 sin(row)), with c_g sweeping over
     decades. This mimics a multigroup transport solve, where L is the streaming operator
     and D_g the per-group removal.

     For each D_g we solve with:
       - fresh:  PCAIR built from L + D_g
       - update: PCAIR built once from L, with PCAIRSetDiagonalUpdate(pc, D_g) and
                 the Amat set to L + D_g
       - frozen: (with -print_frozen) PCAIR built once from L with no update

     ./adv_diff_fd_diag_update -pc_type air
             : sweep with the default c_g, on a 2D grid of size 50 x 50

     Can control the grid size with -n (default 50)
     Can add diffusion with -alpha (default 0.0)
     Can give the list of c_g with -c_values c1,c2,...
     Can error if the update takes more than -max_extra_its extra iterations than fresh (default -1, don't check)

     The checks always done are:
       - applying the same D_g twice gives the same iteration count
       - after the sweep, applying D = 0 gives the iteration count of the hierarchy built from L
       - removing the update with PCAIRSetDiagonalUpdate(pc, NULL) does the same
*/

static char help[] = "Tests a diagonal update to PCAIR with 2D advection.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

#include "pflare.h"

static PetscErrorCode ComputeMat(DM da, Mat A, PetscReal alpha)
{
  PetscInt    i, j, mx, my, xs, ys, xm, ym, ncols;
  PetscScalar vals[5];
  MatStencil  row, cols[5];

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da, 0, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0));

  // Unit velocity at theta = pi/4, dimensionless upwinding
  const PetscReal u = PetscCosReal(PETSC_PI/4.0), v = PetscSinReal(PETSC_PI/4.0);

  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      row.i = i; row.j = j;
      ncols = 0;
      // Dirichlet on the inflow faces
      if (i == 0 || j == 0) {
        cols[0] = row; vals[0] = 1.0; ncols = 1;
      } else {
        cols[ncols].i = i; cols[ncols].j = j; vals[ncols++] = u + v + 4.0 * alpha;
        cols[ncols].i = i - 1; cols[ncols].j = j; vals[ncols++] = -u - alpha;
        cols[ncols].i = i; cols[ncols].j = j - 1; vals[ncols++] = -v - alpha;
        if (alpha != 0.0) {
          if (i < mx - 1) { cols[ncols].i = i + 1; cols[ncols].j = j; vals[ncols++] = -alpha; }
          if (j < my - 1) { cols[ncols].i = i; cols[ncols].j = j + 1; vals[ncols++] = -alpha; }
        }
      }
      PetscCall(MatSetValuesStencil(A, 1, &row, ncols, cols, vals, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// D = c (1 + 0.5 sin(row)) on the diagonal
static PetscErrorCode ComputeDiag(Vec d, PetscReal c)
{
  PetscInt     start, end, i;
  PetscScalar *d_array;

  PetscFunctionBeginUser;
  PetscCall(VecGetOwnershipRange(d, &start, &end));
  PetscCall(VecGetArray(d, &d_array));
  for (i = start; i < end; i++) d_array[i - start] = c * (1.0 + 0.5 * PetscSinReal((PetscReal)i));
  PetscCall(VecRestoreArray(d, &d_array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Checks the multiple rhs apply of the PC matches the single rhs apply after an update
// The block smoothers attach products to the inverses, so this catches any stale ones
static PetscErrorCode CheckMatApply(PC pc, Vec b)
{
  Mat       X, Y;
  Vec       y, y_col;
  PetscInt  local_rows, global_rows;
  PetscReal norm_diff, norm_y;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(b, &local_rows));
  PetscCall(VecGetSize(b, &global_rows));
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)b), local_rows, PETSC_DECIDE, global_rows, 2, NULL, &X));
  PetscCall(MatDuplicate(X, MAT_DO_NOT_COPY_VALUES, &Y));
  PetscCall(MatDenseGetColumnVecWrite(X, 0, &y_col));
  PetscCall(VecCopy(b, y_col));
  PetscCall(MatDenseRestoreColumnVecWrite(X, 0, &y_col));
  PetscCall(MatDenseGetColumnVecWrite(X, 1, &y_col));
  PetscCall(VecCopy(b, y_col));
  PetscCall(VecScale(y_col, 2.0));
  PetscCall(MatDenseRestoreColumnVecWrite(X, 1, &y_col));

  PetscCall(PCMatApply(pc, X, Y));
  PetscCall(VecDuplicate(b, &y));
  PetscCall(PCApply(pc, b, y));
  PetscCall(MatDenseGetColumnVecRead(Y, 0, &y_col));
  PetscCall(VecAXPY(y, -1.0, y_col));
  PetscCall(VecNorm(y, NORM_2, &norm_diff));
  PetscCall(VecNorm(y_col, NORM_2, &norm_y));
  PetscCall(MatDenseRestoreColumnVecRead(Y, 0, &y_col));
  PetscCheck(norm_diff <= 1e-4 * norm_y, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PCMatApply differs from PCApply by %g relative", (double)(norm_diff / norm_y));

  PetscCall(VecDestroy(&y));
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Solves and returns the iteration count, erroring if we didn't converge
static PetscErrorCode Solve(KSP ksp, Vec b, Vec x, PetscInt *its)
{
  KSPConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(VecSet(x, 0.0));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(KSPGetIterationNumber(ksp, its));
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  PetscCheck(reason > 0, PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, "Solve did not converge, reason %d", (int)reason);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  KSP       ksp_fresh, ksp_update, ksp_frozen = NULL;
  PC        pc_update;
  DM        da;
  Mat       L, A_fresh, A_update, D;
  Vec       x, b, d_vec;
  PetscInt  n = 50, n_c = 16, i_loc, its_fresh, its_update, its_repeat, its_frozen, its_base, its_zero, its_null;
  PetscInt  max_extra_its = -1;
  PetscReal alpha = 0.0;
  PetscReal c_values[16] = {0.0, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3};
  PetscBool print_frozen = PETSC_FALSE, flg;

  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PCRegister_PFLARE();

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-alpha", &alpha, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-max_extra_its", &max_extra_its, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-print_frozen", &print_frozen, NULL));
  PetscCall(PetscOptionsGetRealArray(NULL, NULL, "-c_values", c_values, &n_c, &flg));
  if (!flg) n_c = 8;

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, n, n, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  PetscCall(DMCreateMatrix(da, &L));
  PetscCall(ComputeMat(da, L, alpha));
  PetscCall(MatDuplicate(L, MAT_COPY_VALUES, &A_fresh));
  PetscCall(MatDuplicate(L, MAT_COPY_VALUES, &A_update));
  PetscCall(MatCreateVecs(L, &x, &b));
  PetscCall(VecDuplicate(x, &d_vec));
  PetscCall(VecSet(b, 1.0));
  // D only needs its diagonal
  PetscCall(MatCreateDiagonal(d_vec, &D));

  // The fresh solve, rebuilt from L + D_g for every g
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_fresh));
  PetscCall(KSPSetOperators(ksp_fresh, A_fresh, A_fresh));
  PetscCall(KSPSetFromOptions(ksp_fresh));

  // The updated solve, the Pmat is always L
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_update));
  PetscCall(KSPSetOperators(ksp_update, A_update, L));
  PetscCall(KSPSetFromOptions(ksp_update));
  PetscCall(KSPGetPC(ksp_update, &pc_update));

  // The frozen solve, the hierarchy built from L with no update
  if (print_frozen) {
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_frozen));
    PetscCall(KSPSetOperators(ksp_frozen, A_update, L));
    PetscCall(KSPSetFromOptions(ksp_frozen));
  }

  // The iteration count with the hierarchy built from L
  PetscCall(KSPSetOperators(ksp_update, L, L));
  PetscCall(Solve(ksp_update, b, x, &its_base));
  PetscCall(KSPSetOperators(ksp_update, A_update, L));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Iterations with the hierarchy built from L: %" PetscInt_FMT "\n", its_base));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%12s %8s %8s %8s\n", "c", "fresh", "update", print_frozen ? "frozen" : ""));

  for (i_loc = 0; i_loc < n_c; i_loc++) {

    // D_g, with the matdiagonal D sharing d_vec
    PetscCall(ComputeDiag(d_vec, c_values[i_loc]));
    PetscCall(PetscObjectStateIncrease((PetscObject)D));

    // L + D_g
    PetscCall(MatCopy(L, A_fresh, SAME_NONZERO_PATTERN));
    PetscCall(MatDiagonalSet(A_fresh, d_vec, ADD_VALUES));
    PetscCall(MatCopy(L, A_update, SAME_NONZERO_PATTERN));
    PetscCall(MatDiagonalSet(A_update, d_vec, ADD_VALUES));

    PetscCall(Solve(ksp_fresh, b, x, &its_fresh));

    PetscCall(PCAIRSetDiagonalUpdate(pc_update, D));
    PetscCall(Solve(ksp_update, b, x, &its_update));
    PetscCall(CheckMatApply(pc_update, b));
    // Applying the same update again has to give the same answer
    PetscCall(PCAIRSetDiagonalUpdate(pc_update, D));
    PetscCall(Solve(ksp_update, b, x, &its_repeat));
    PetscCheck(its_repeat == its_update, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Repeating the update gave %" PetscInt_FMT " iterations rather than %" PetscInt_FMT, its_repeat, its_update);

    its_frozen = -1;
    if (print_frozen) PetscCall(Solve(ksp_frozen, b, x, &its_frozen));

    if (print_frozen) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%12.3e %8" PetscInt_FMT " %8" PetscInt_FMT " %8" PetscInt_FMT "\n", (double)c_values[i_loc], its_fresh, its_update, its_frozen));
    else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%12.3e %8" PetscInt_FMT " %8" PetscInt_FMT "\n", (double)c_values[i_loc], its_fresh, its_update));

    if (max_extra_its >= 0) PetscCheck(its_update - its_fresh <= max_extra_its, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "The update took %" PetscInt_FMT " iterations compared to %" PetscInt_FMT " fresh", its_update, its_fresh);
  }

  // Going back to D = 0 has to give back the hierarchy built from L
  PetscCall(MatCopy(L, A_update, SAME_NONZERO_PATTERN));
  PetscCall(VecSet(d_vec, 0.0));
  PetscCall(PetscObjectStateIncrease((PetscObject)D));
  PetscCall(PCAIRSetDiagonalUpdate(pc_update, D));
  PetscCall(Solve(ksp_update, b, x, &its_zero));
  PetscCheck(its_zero == its_base, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "An update with D = 0 gave %" PetscInt_FMT " iterations rather than %" PetscInt_FMT, its_zero, its_base);

  // As does removing the update, after applying a non-zero one
  PetscCall(ComputeDiag(d_vec, 1.0));
  PetscCall(PetscObjectStateIncrease((PetscObject)D));
  PetscCall(PCAIRSetDiagonalUpdate(pc_update, D));
  PetscCall(PCAIRSetDiagonalUpdate(pc_update, NULL));
  PetscCall(Solve(ksp_update, b, x, &its_null));
  PetscCheck(its_null == its_base, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Removing the update gave %" PetscInt_FMT " iterations rather than %" PetscInt_FMT, its_null, its_base);

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Iterations after going back to D = 0: %" PetscInt_FMT ", after removing the update: %" PetscInt_FMT "\n", its_zero, its_null));

  PetscCall(KSPDestroy(&ksp_fresh));
  PetscCall(KSPDestroy(&ksp_update));
  PetscCall(KSPDestroy(&ksp_frozen));
  PetscCall(MatDestroy(&L));
  PetscCall(MatDestroy(&A_fresh));
  PetscCall(MatDestroy(&A_update));
  PetscCall(MatDestroy(&D));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&d_vec));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}
