/*   Tests the (prototype) multi-shift block V-cycle of PCAIR, PCAIRSetMultiShift.
     L is a steady 2D upwind finite difference advection(-diffusion) operator and the k
     shifted systems are (L + D_j) x_j = b_j, with D_j = c_j (1 + 0.5 sin(row)), as in
     adv_diff_fd_diag_update. The hierarchy is built once from L.

     The checks always done are (with -pc_air_reuse_poly_coeffs, which is what the
     multi-shift reproduces):
       - column j of one multi-shift PCMatApply on k columns matches PCApply after a single
         rhs diagonal update to L + D_j
       - the multi-shift still gives the same answer after the hierarchy has been updated
         (the multi-shift data is recomputed relative to the updated hierarchy)
       - a multi-shift with zero shifts matches the plain PCMG PCMatApply
       - clearing the multi-shift gives back the plain PCMG block apply
     Errors if any relative difference is above -tol (default 1e-8, 1e-3 in single precision)

     With -solve, also solves all k systems with
       - sequential: for each j a diagonal update to L + D_j and a right preconditioned GMRES
       - multi-shift: one batched right preconditioned (F)GMRES over the k columns, whose matvecs
         are a single SpMM plus a Hadamard product and whose preconditioner is one multi-shift
         PCMatApply
     and prints the iteration counts per column and the wall times, erroring if any column of the
     multi-shift takes more than -max_extra_its (default 1) more iterations than sequential
     With -frozen as well, also solves with the hierarchy built from the base applied unchanged to
     every column (the plain PCMatApply of PCAIR), with at most -frozen_max_it (default 300) iterations
     The frozen hierarchy is built from L + D(c_frozen), -c_frozen defaults to c_base
     Can build the hierarchy from L + D(c_base) with -c_base (default 0), the shifts are then relative to it
     With -richardson as well, also solves with an outer Richardson, sequentially with an update and
     a KSPSolve for each shift, and with a single KSPMatSolve whose Amat is a matshell applying the
     shifted operators to a dense block, ie PETSc's block Richardson with no changes

     ./adv_diff_fd_multishift -pc_type air
     Can control the grid size with -n (default 50), diffusion with -alpha (default 0.0),
     the shifts with -c_values c1,c2,... (default 0,1e-3,...,1e3), the GMRES tolerance with -ksp_rtol
*/

static char help[] = "Tests the multi-shift block V-cycle of PCAIR with 2D advection.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petsctime.h>

#include "pflare.h"

#define MAX_SHIFTS 128

static PetscErrorCode ComputeMat(DM da, Mat A, PetscReal alpha)
{
  PetscInt    i, j, mx, my, xs, ys, xm, ym, ncols;
  PetscScalar vals[5];
  MatStencil  row, cols[5];

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da, 0, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0));
  const PetscReal u = PetscCosReal(PETSC_PI / 4.0), v = PetscSinReal(PETSC_PI / 4.0);
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      row.i = i; row.j = j;
      ncols = 0;
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Column-wise dense helpers, host loops
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Y = Y + D .* X
static PetscErrorCode DenseHadamardAdd(Mat Y, Mat D, Mat X)
{
  PetscScalar       *y;
  const PetscScalar *d, *x;
  PetscInt           m, k, ldy, ldd, ldx, i, j;

  PetscFunctionBeginUser;
  PetscCall(MatGetLocalSize(Y, &m, NULL));
  PetscCall(MatGetSize(Y, NULL, &k));
  PetscCall(MatDenseGetLDA(Y, &ldy));
  PetscCall(MatDenseGetLDA(D, &ldd));
  PetscCall(MatDenseGetLDA(X, &ldx));
  PetscCall(MatDenseGetArray(Y, &y));
  PetscCall(MatDenseGetArrayRead(D, &d));
  PetscCall(MatDenseGetArrayRead(X, &x));
  for (j = 0; j < k; j++)
    for (i = 0; i < m; i++) y[i + j * ldy] += d[i + j * ldd] * x[i + j * ldx];
  PetscCall(MatDenseRestoreArrayRead(X, &x));
  PetscCall(MatDenseRestoreArrayRead(D, &d));
  PetscCall(MatDenseRestoreArray(Y, &y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// out_j = X(:,j) . Y(:,j)
static PetscErrorCode ColumnDots(Mat X, Mat Y, PetscScalar *out)
{
  const PetscScalar *x, *y;
  PetscInt           m, k, ldx, ldy, i, j;

  PetscFunctionBeginUser;
  PetscCall(MatGetLocalSize(X, &m, NULL));
  PetscCall(MatGetSize(X, NULL, &k));
  PetscCall(MatDenseGetLDA(X, &ldx));
  PetscCall(MatDenseGetLDA(Y, &ldy));
  PetscCall(MatDenseGetArrayRead(X, &x));
  PetscCall(MatDenseGetArrayRead(Y, &y));
  for (j = 0; j < k; j++) {
    out[j] = 0.0;
    for (i = 0; i < m; i++) out[j] += x[i + j * ldx] * y[i + j * ldy];
  }
  PetscCall(MatDenseRestoreArrayRead(Y, &y));
  PetscCall(MatDenseRestoreArrayRead(X, &x));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, out, (PetscMPIInt)k, MPIU_SCALAR, MPIU_SUM, PetscObjectComm((PetscObject)X)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Y(:,j) = Y(:,j) + alpha_j X(:,j)
static PetscErrorCode ColumnAXPY(Mat Y, const PetscScalar *alpha, Mat X)
{
  PetscScalar       *y;
  const PetscScalar *x;
  PetscInt           m, k, ldx, ldy, i, j;

  PetscFunctionBeginUser;
  PetscCall(MatGetLocalSize(Y, &m, NULL));
  PetscCall(MatGetSize(Y, NULL, &k));
  PetscCall(MatDenseGetLDA(X, &ldx));
  PetscCall(MatDenseGetLDA(Y, &ldy));
  PetscCall(MatDenseGetArray(Y, &y));
  PetscCall(MatDenseGetArrayRead(X, &x));
  for (j = 0; j < k; j++)
    for (i = 0; i < m; i++) y[i + j * ldy] += alpha[j] * x[i + j * ldx];
  PetscCall(MatDenseRestoreArrayRead(X, &x));
  PetscCall(MatDenseRestoreArray(Y, &y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Y(:,j) = alpha_j Y(:,j)
static PetscErrorCode ColumnScale(Mat Y, const PetscScalar *alpha)
{
  PetscScalar *y;
  PetscInt     m, k, ldy, i, j;

  PetscFunctionBeginUser;
  PetscCall(MatGetLocalSize(Y, &m, NULL));
  PetscCall(MatGetSize(Y, NULL, &k));
  PetscCall(MatDenseGetLDA(Y, &ldy));
  PetscCall(MatDenseGetArray(Y, &y));
  for (j = 0; j < k; j++)
    for (i = 0; i < m; i++) y[i + j * ldy] *= alpha[j];
  PetscCall(MatDenseRestoreArray(Y, &y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Y = (L + diag(D(:,j))) X column by column, ie one SpMM and a Hadamard product
static PetscErrorCode ShiftedMatMat(Mat L, Mat D, Mat X, Mat Y)
{
  PetscFunctionBeginUser;
  PetscCall(MatMatMult(L, X, MAT_REUSE_MATRIX, PETSC_DETERMINE, &Y));
  PetscCall(DenseHadamardAdd(Y, D, X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// A matshell Amat for the k shifted operators, whose product with a dense block is
// (L + diag(D(:,j))) X(:,j) column by column, ie one SpMM and a Hadamard product
// With it KSPMatSolve with KSPRICHARDSON is a batched multi-shift solve, as the block
// Richardson in PETSc only ever applies Amat through MatProduct
// A MatMult has no column to pick the shift from, so it is an error
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
typedef struct {
  Mat L, D, LX;
} ShiftedCtx;

static PetscErrorCode ShiftedMult(Mat S, Vec x, Vec y)
{
  PetscFunctionBeginUser;
  SETERRQ(PetscObjectComm((PetscObject)S), PETSC_ERR_SUP, "The multi-shift operator can only be applied to a block of k columns");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ShiftedProductNumeric(Mat S, Mat X, Mat Y, void *data)
{
  ShiftedCtx *ctx;
  PetscInt    kx, kl = -1;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(S, &ctx));
  // L X goes into our own block, Y carries the product of the shell
  PetscCall(MatGetSize(X, NULL, &kx));
  if (ctx->LX) PetscCall(MatGetSize(ctx->LX, NULL, &kl));
  if (kl != kx) {
    PetscCall(MatDestroy(&ctx->LX));
    PetscCall(MatMatMult(ctx->L, X, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &ctx->LX));
  } else PetscCall(MatMatMult(ctx->L, X, MAT_REUSE_MATRIX, PETSC_DETERMINE, &ctx->LX));
  PetscCall(MatCopy(ctx->LX, Y, SAME_NONZERO_PATTERN));
  PetscCall(DenseHadamardAdd(Y, ctx->D, X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateShiftedOperator(Mat L, Mat D, ShiftedCtx *ctx, Mat *S)
{
  PetscInt m, n, M, N;

  PetscFunctionBeginUser;
  ctx->L  = L;
  ctx->D  = D;
  ctx->LX = NULL;
  PetscCall(MatGetLocalSize(L, &m, &n));
  PetscCall(MatGetSize(L, &M, &N));
  PetscCall(MatCreateShell(PetscObjectComm((PetscObject)L), m, n, M, N, ctx, S));
  PetscCall(MatShellSetOperation(*S, MATOP_MULT, (PetscErrorCodeFn *)ShiftedMult));
  PetscCall(MatShellSetMatProductOperation(*S, MATPRODUCT_AB, NULL, ShiftedProductNumeric, NULL, MATDENSE, MATDENSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Batched right preconditioned FGMRES(m), k independent Arnoldi processes whose
// matvecs and preconditioner applies are batched over the columns
// With a fixed preconditioner this is right preconditioned GMRES in exact arithmetic
// Each column stops counting iterations once ||r_j|| <= rtol ||b_j||; converged
// columns are still carried through the block operations (frozen, not deflated)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
static PetscErrorCode BatchedGMRES(Mat L, Mat D, PC pc, Mat B, Mat X, PetscReal rtol, PetscInt restart, PetscInt maxit, PetscInt *its, PetscInt *n_applies)
{
  PetscInt     k, j, i, l, total = 0, *n_done;
  Mat         *V, *Z, R;
  PetscScalar *h, *g, *cs, *sn, *coef, *beta, *bnorm, *y;
  PetscBool   *active, *conv, any;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(B, NULL, &k));
  PetscCall(PetscMalloc2(restart + 1, &V, restart, &Z));
  for (i = 0; i <= restart; i++) PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &V[i]));
  for (i = 0; i < restart; i++) PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &Z[i]));
  PetscCall(MatMatMult(L, X, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &R));
  // h[j][row][col] with (restart + 1) x restart per column
  PetscCall(PetscCalloc4(k * (restart + 1) * restart, &h, k * (restart + 1), &g, k * restart, &cs, k * restart, &sn));
  PetscCall(PetscMalloc4(k, &coef, k, &beta, k, &bnorm, restart, &y));
  PetscCall(PetscMalloc3(k, &active, k, &conv, k, &n_done));
#define H(j, r, c) h[(j) * (restart + 1) * restart + (r) * restart + (c)]

  PetscCall(MatZeroEntries(X));
  PetscCall(ColumnDots(B, B, bnorm));
  for (j = 0; j < k; j++) {
    bnorm[j] = PetscSqrtReal(PetscRealPart(bnorm[j]));
    its[j]    = 0;
    active[j] = PETSC_TRUE;
  }
  *n_applies = 0;

  while (total < maxit) {
    // R = B - A X
    PetscCall(ShiftedMatMat(L, D, X, R));
    PetscCall(MatAYPX(R, -1.0, B, SAME_NONZERO_PATTERN));
    PetscCall(ColumnDots(R, R, beta));
    any = PETSC_FALSE;
    for (j = 0; j < k; j++) {
      beta[j] = PetscSqrtReal(PetscRealPart(beta[j]));
      if (PetscRealPart(beta[j]) <= rtol * PetscRealPart(bnorm[j])) active[j] = PETSC_FALSE;
      if (active[j]) any = PETSC_TRUE;
      // Inactive columns are zeroed so they carry no work into the basis
      coef[j]   = active[j] ? 1.0 / beta[j] : 0.0;
      conv[j]   = PETSC_FALSE;
      n_done[j] = 0;
      for (i = 0; i <= restart; i++) g[j * (restart + 1) + i] = 0.0;
      g[j * (restart + 1)] = active[j] ? beta[j] : 0.0;
    }
    if (!any) break;
    PetscCall(MatCopy(R, V[0], SAME_NONZERO_PATTERN));
    PetscCall(ColumnScale(V[0], coef));

    for (i = 0; i < restart && total < maxit; i++) {
      // Z_i = M^-1 V_i, one multi-shift block apply
      PetscCall(PCMatApply(pc, V[i], Z[i]));
      (*n_applies)++;
      // V_{i+1} = A Z_i
      PetscCall(ShiftedMatMat(L, D, Z[i], V[i + 1]));
      // Modified Gram-Schmidt, column by column
      for (l = 0; l <= i; l++) {
        PetscCall(ColumnDots(V[l], V[i + 1], coef));
        for (j = 0; j < k; j++) H(j, l, i) = coef[j];
        for (j = 0; j < k; j++) coef[j] = -coef[j];
        PetscCall(ColumnAXPY(V[i + 1], coef, V[l]));
      }
      PetscCall(ColumnDots(V[i + 1], V[i + 1], coef));
      for (j = 0; j < k; j++) {
        coef[j]          = PetscSqrtReal(PetscRealPart(coef[j]));
        H(j, i + 1, i)   = coef[j];
        coef[j]          = PetscRealPart(coef[j]) > 0.0 ? 1.0 / coef[j] : 0.0;
      }
      PetscCall(ColumnScale(V[i + 1], coef));
      total++;

      // Givens rotations on the active, unconverged columns
      any = PETSC_FALSE;
      for (j = 0; j < k; j++) {
        PetscScalar *gj = &g[j * (restart + 1)], temp, denom;
        if (!active[j] || conv[j]) continue;
        for (l = 0; l < i; l++) {
          temp           = cs[j * restart + l] * H(j, l, i) + sn[j * restart + l] * H(j, l + 1, i);
          H(j, l + 1, i) = -sn[j * restart + l] * H(j, l, i) + cs[j * restart + l] * H(j, l + 1, i);
          H(j, l, i)     = temp;
        }
        denom = PetscSqrtReal(PetscRealPart(H(j, i, i) * H(j, i, i) + H(j, i + 1, i) * H(j, i + 1, i)));
        cs[j * restart + i] = H(j, i, i) / denom;
        sn[j * restart + i] = H(j, i + 1, i) / denom;
        H(j, i, i)          = denom;
        H(j, i + 1, i)      = 0.0;
        gj[i + 1]           = -sn[j * restart + i] * gj[i];
        gj[i]               = cs[j * restart + i] * gj[i];
        its[j]++;
        n_done[j] = i + 1;
        if (PetscAbsScalar(gj[i + 1]) <= rtol * PetscRealPart(bnorm[j])) conv[j] = PETSC_TRUE;
        else any = PETSC_TRUE;
      }
      if (!any) { i++; break; }
    }

    // X_j += sum_l y_l Z_l(:,j), with y from the triangular solve of column j
    for (l = 0; l < i; l++) {
      for (j = 0; j < k; j++) coef[j] = 0.0;
      for (j = 0; j < k; j++) {
        PetscInt n = n_done[j], r, c;
        if (!active[j] || l >= n) continue;
        for (r = n - 1; r >= 0; r--) {
          y[r] = g[j * (restart + 1) + r];
          for (c = r + 1; c < n; c++) y[r] -= H(j, r, c) * y[c];
          y[r] /= H(j, r, r);
        }
        coef[j] = y[l];
      }
      PetscCall(ColumnAXPY(X, coef, Z[l]));
    }
  }
#undef H

  for (i = 0; i <= restart; i++) PetscCall(MatDestroy(&V[i]));
  for (i = 0; i < restart; i++) PetscCall(MatDestroy(&Z[i]));
  PetscCall(MatDestroy(&R));
  PetscCall(PetscFree2(V, Z));
  PetscCall(PetscFree4(h, g, cs, sn));
  PetscCall(PetscFree4(coef, beta, bnorm, y));
  PetscCall(PetscFree3(active, conv, n_done));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Relative difference between column j of Y and y
static PetscErrorCode ColumnDiff(Mat Y, PetscInt j, Vec y, PetscReal *rel)
{
  Vec       col, diff;
  PetscReal norm_diff, norm_y;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(y, &diff));
  PetscCall(MatDenseGetColumnVecRead(Y, j, &col));
  PetscCall(VecWAXPY(diff, -1.0, col, y));
  PetscCall(VecNorm(col, NORM_2, &norm_y));
  PetscCall(MatDenseRestoreColumnVecRead(Y, j, &col));
  PetscCall(VecNorm(diff, NORM_2, &norm_diff));
  *rel = norm_diff / norm_y;
  PetscCall(VecDestroy(&diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// The relative difference between two dense blocks, max over columns
static PetscErrorCode BlockDiff(Mat Y1, Mat Y2, PetscReal *rel)
{
  PetscInt  k, j;
  Vec       c1;
  PetscReal r;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(Y1, NULL, &k));
  *rel = 0.0;
  for (j = 0; j < k; j++) {
    PetscCall(MatDenseGetColumnVecRead(Y2, j, &c1));
    PetscCall(ColumnDiff(Y1, j, c1, &r));
    PetscCall(MatDenseRestoreColumnVecRead(Y2, j, &c1));
    *rel = PetscMax(*rel, r);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Sets up a KSP with PCAIR configured for diagonal updates with kept coefficients and
// matrix-free, diagonally scaled polynomials, which is what the multi-shift reproduces
static PetscErrorCode CreateKSP(Mat Amat, Mat A, KSPType type, PetscReal rtol, KSP *ksp)
{
  PC        pc;
  PetscBool is_gmres;

  PetscFunctionBeginUser;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, ksp));
  PetscCall(KSPSetOperators(*ksp, Amat, A));
  PetscCall(KSPSetType(*ksp, type));
  PetscCall(PetscStrcmp(type, KSPGMRES, &is_gmres));
  if (is_gmres) PetscCall(KSPSetPCSide(*ksp, PC_RIGHT));
  PetscCall(KSPSetNormType(*ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCall(KSPSetTolerances(*ksp, rtol, 1e-50, PETSC_DETERMINE, 1000));
  PetscCall(KSPGetPC(*ksp, &pc));
  PetscCall(PCSetType(pc, PCAIR));
  PetscCall(PCAIRSetDiagUpdate(pc, PETSC_TRUE));
  PetscCall(PCAIRSetReusePolyCoeffs(pc, PETSC_TRUE));
  PetscCall(PCAIRSetMatrixFreePolys(pc, PETSC_TRUE));
  PetscCall(PCAIRSetCoarsestMatrixFreePolys(pc, PETSC_TRUE));
  // Kept coefficients are only sensible with diagonally scaled polynomials, unscaled
  // ones overflow for large shifts in the single rhs update too
  PetscCall(PCAIRSetDiagScalePolys(pc, PETSC_TRUE));
  PetscCall(PCAIRSetCoarsestDiagScalePolys(pc, PETSC_TRUE));
  PetscCall(KSPSetFromOptions(*ksp));
  PetscCall(KSPSetUp(*ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  KSP           ksp_ms, ksp_seq;
  PC            pc_ms, pc_seq;
  DM            da;
  Mat           L, A_ms, A_fr, A_seq, Dshift, B, Y, Y2, X_ms, X_fr;
  Vec           x, b, d_vec, col, y;
  PetscRandom   rand;
  PetscInt      n = 50, k = MAX_SHIFTS, j, i, its_seq[MAX_SHIFTS], its_ms[MAX_SHIFTS], max_extra_its = 1, n_applies = 0, n_reps = 5, frozen_max_it = 300, its_fr[MAX_SHIFTS], local_rows, global_rows;
  PetscReal     rel_clear, c_base = 0.0, c_frozen, alpha = 0.0, tol = 1e-8, rtol = 1e-10, rel, max_rel = 0.0, max_rel_after, rel_zero;
  PetscReal     c_values[MAX_SHIFTS] = {0.0, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3};
  PetscBool     flg, solve = PETSC_FALSE, print_cols = PETSC_FALSE, richardson = PETSC_FALSE, frozen = PETSC_FALSE;
  PetscReal     rnorm_fr[MAX_SHIFTS];
  PetscLogDouble t_fr_solve = 0, t0, t1, t_seq_solve = 0, t_ms_solve, t_ms_set, t_ms_apply, t_mrhs_apply, t_seq_apply, t_seq_update;
  KSPConvergedReason reason;

  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PCRegister_PFLARE();

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-alpha", &alpha, NULL));
#if defined(PETSC_USE_REAL_SINGLE)
  tol = 1e-3;
#endif
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-tol", &tol, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-ksp_rtol", &rtol, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-max_extra_its", &max_extra_its, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_reps", &n_reps, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-solve", &solve, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-print_cols", &print_cols, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-richardson", &richardson, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-frozen", &frozen, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-frozen_max_it", &frozen_max_it, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-c_base", &c_base, NULL));
  c_frozen = c_base;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-c_frozen", &c_frozen, NULL));
  PetscCall(PetscOptionsGetRealArray(NULL, NULL, "-c_values", c_values, &k, &flg));
  if (!flg) k = 8;

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, n, n, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMCreateMatrix(da, &L));
  PetscCall(ComputeMat(da, L, alpha));
  // From here on L is the base the hierarchy is built from, L + D(c_base)
  if (c_base != 0.0) {
    Vec d_base;
    PetscCall(MatCreateVecs(L, &d_base, NULL));
    PetscCall(ComputeDiag(d_base, c_base));
    PetscCall(MatDiagonalSet(L, d_base, ADD_VALUES));
    PetscCall(VecDestroy(&d_base));
  }
  PetscCall(MatCreateVecs(L, &x, &b));
  PetscCall(VecDuplicate(x, &d_vec));
  PetscCall(VecDuplicate(x, &y));
  PetscCall(MatDuplicate(L, MAT_COPY_VALUES, &A_ms));
  // The matrix a frozen hierarchy is built from, L + D(c_frozen)
  PetscCall(MatDuplicate(L, MAT_COPY_VALUES, &A_fr));
  if (c_frozen != c_base) {
    PetscCall(ComputeDiag(d_vec, c_frozen - c_base));
    PetscCall(MatDiagonalSet(A_fr, d_vec, ADD_VALUES));
  }
  PetscCall(MatDuplicate(L, MAT_COPY_VALUES, &A_seq));
  PetscCall(VecGetLocalSize(b, &local_rows));
  PetscCall(VecGetSize(b, &global_rows));

  // The shifts and a random block of rhs
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, local_rows, PETSC_DECIDE, global_rows, k, NULL, &Dshift));
  PetscCall(MatDuplicate(Dshift, MAT_DO_NOT_COPY_VALUES, &B));
  PetscCall(MatDuplicate(Dshift, MAT_DO_NOT_COPY_VALUES, &Y));
  PetscCall(MatDuplicate(Dshift, MAT_DO_NOT_COPY_VALUES, &Y2));
  PetscCall(MatDuplicate(Dshift, MAT_DO_NOT_COPY_VALUES, &X_ms));
  PetscCall(MatDuplicate(Dshift, MAT_DO_NOT_COPY_VALUES, &X_fr));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(MatGetDiagonal(L, y));
  for (j = 0; j < k; j++) {
    // The shift exactly as a diagonal update sees it, diag(L + D_j) - diag(L), which
    // differs from D_j by rounding
    PetscCall(MatDenseGetColumnVecWrite(Dshift, j, &col));
    // D(c) is linear in c, so the shift relative to the base is D(c_j - c_base)
    PetscCall(ComputeDiag(col, c_values[j] - c_base));
    PetscCall(VecAXPY(col, 1.0, y));
    PetscCall(VecAXPY(col, -1.0, y));
    PetscCall(MatDenseRestoreColumnVecWrite(Dshift, j, &col));
    PetscCall(MatDenseGetColumnVecWrite(B, j, &col));
    PetscCall(VecSetRandom(col, rand));
    PetscCall(MatDenseRestoreColumnVecWrite(B, j, &col));
  }

  // Both hierarchies are built from L
  PetscCall(CreateKSP(A_ms, A_ms, KSPGMRES, rtol, &ksp_ms));
  PetscCall(CreateKSP(A_seq, A_seq, KSPGMRES, rtol, &ksp_seq));
  PetscCall(KSPGetPC(ksp_ms, &pc_ms));
  PetscCall(KSPGetPC(ksp_seq, &pc_seq));

  // ~~~~~~~~~~~~
  // A multi-shift with zero shifts has to match the PCMG block apply
  // ~~~~~~~~~~~~
  PetscCall(PCMatApply(pc_ms, B, Y2));
  PetscCall(MatZeroEntries(Y));
  PetscCall(PCAIRSetMultiShift(pc_ms, Y));
  PetscCall(PCMatApply(pc_ms, B, Y));
  PetscCall(BlockDiff(Y, Y2, &rel_zero));

  // ~~~~~~~~~~~~
  // Column j of the multi-shift has to match PCApply after a diagonal update to L + D_j
  // ~~~~~~~~~~~~
  PetscCall(PCAIRSetMultiShift(pc_ms, Dshift));
  PetscCall(PCMatApply(pc_ms, B, Y));
  for (j = 0; j < k; j++) {
    PetscCall(MatDenseGetColumnVecRead(Dshift, j, &col));
    PetscCall(VecCopy(col, d_vec));
    PetscCall(MatDenseRestoreColumnVecRead(Dshift, j, &col));
    PetscCall(MatCopy(L, A_seq, SAME_NONZERO_PATTERN));
    PetscCall(MatDiagonalSet(A_seq, d_vec, ADD_VALUES));
    PetscCall(PCSetUp(pc_seq));
    PetscCall(MatDenseGetColumnVecRead(B, j, &col));
    PetscCall(PCApply(pc_seq, col, y));
    PetscCall(MatDenseRestoreColumnVecRead(B, j, &col));
    PetscCall(ColumnDiff(Y, j, y, &rel));
    PetscCall(PetscInfo(NULL, "Column %" PetscInt_FMT " c %g relative difference %g\n", j, (double)c_values[j], (double)rel));
    if (print_cols) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  column %" PetscInt_FMT " c = %10.3e: relative difference %.3e\n", j, (double)c_values[j], (double)rel));
    max_rel = PetscMax(max_rel, rel);
  }

  // ~~~~~~~~~~~~
  // After a diagonal update of the multi-shift hierarchy itself, it has to give the same answer
  // ~~~~~~~~~~~~
  PetscCall(MatCopy(L, A_ms, SAME_NONZERO_PATTERN));
  PetscCall(MatDiagonalSet(A_ms, d_vec, ADD_VALUES));
  PetscCall(PCSetUp(pc_ms));
  PetscCall(PCMatApply(pc_ms, B, Y2));
  PetscCall(BlockDiff(Y, Y2, &max_rel_after));
  // Go back to the hierarchy built from L
  PetscCall(MatCopy(L, A_ms, SAME_NONZERO_PATTERN));
  PetscCall(PCSetUp(pc_ms));

  // ~~~~~~~~~~~~
  // Clearing the multi-shift has to give back the plain PCMG block apply, ie the
  // original restrictors are back in the PCMG
  // ~~~~~~~~~~~~
  PetscCall(PCAIRSetMultiShift(pc_ms, NULL));
  PetscCall(PCMatApply(pc_ms, B, Y2));
  PetscCall(MatCopy(L, A_seq, SAME_NONZERO_PATTERN));
  PetscCall(PCSetUp(pc_seq));
  PetscCall(PCMatApply(pc_seq, B, Y));
  PetscCall(BlockDiff(Y, Y2, &rel_clear));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Max relative difference: zero shift vs PCMG %.3e, multi-shift vs single rhs updates %.3e, after an update %.3e, cleared vs PCMG %.3e\n", (double)rel_zero, (double)max_rel, (double)max_rel_after, (double)rel_clear));
  PetscCheck(rel_zero <= tol && max_rel <= tol && max_rel_after <= tol && rel_clear <= tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Multi-shift apply differs from the single rhs applies");

  // ~~~~~~~~~~~~
  // Solve all the shifted systems both ways
  // ~~~~~~~~~~~~
  if (solve) {
    PetscInt restart = 30;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-ksp_gmres_restart", &restart, NULL));

    // Sequential, an update and a solve for each shift
    for (j = 0; j < k; j++) {
      PetscCall(MatDenseGetColumnVecRead(Dshift, j, &col));
      PetscCall(VecCopy(col, d_vec));
      PetscCall(MatDenseRestoreColumnVecRead(Dshift, j, &col));
      PetscCall(MatDenseGetColumnVecRead(B, j, &col));
      PetscCall(VecCopy(col, b));
      PetscCall(MatDenseRestoreColumnVecRead(B, j, &col));
      PetscCall(PetscBarrier(NULL));
      PetscCall(PetscTime(&t0));
      PetscCall(MatCopy(L, A_seq, SAME_NONZERO_PATTERN));
      PetscCall(MatDiagonalSet(A_seq, d_vec, ADD_VALUES));
      PetscCall(VecSet(x, 0.0));
      PetscCall(KSPSolve(ksp_seq, b, x));
      PetscCall(PetscTime(&t1));
      t_seq_solve += t1 - t0;
      PetscCall(KSPGetIterationNumber(ksp_seq, &its_seq[j]));
      PetscCall(KSPGetConvergedReason(ksp_seq, &reason));
      PetscCheck(reason > 0, PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, "Sequential solve %" PetscInt_FMT " did not converge", j);
    }

    // Multi-shift, one batched solve (the setup of the shifts is included)
    PetscCall(PetscBarrier(NULL));
    PetscCall(PetscTime(&t0));
    PetscCall(PCAIRSetMultiShift(pc_ms, Dshift));
    PetscCall(PetscTime(&t1));
    t_ms_set = t1 - t0;
    PetscCall(BatchedGMRES(L, Dshift, pc_ms, B, X_ms, rtol, restart, 1000, its_ms, &n_applies));
    PetscCall(PetscTime(&t1));
    t_ms_solve = t1 - t0;

    // The frozen comparison, a hierarchy built from L + D(c_frozen) applied to every
    // column with no update, which is just the plain PCMatApply of PCAIR
    if (frozen) {
      PetscInt n_applies_fr;
      KSP      ksp_fr;
      PC       pc_fr;
      PetscCall(CreateKSP(A_fr, A_fr, KSPGMRES, rtol, &ksp_fr));
      PetscCall(KSPGetPC(ksp_fr, &pc_fr));
      PetscCall(PetscBarrier(NULL));
      PetscCall(PetscTime(&t0));
      PetscCall(BatchedGMRES(L, Dshift, pc_fr, B, X_fr, rtol, restart, frozen_max_it, its_fr, &n_applies_fr));
      PetscCall(PetscTime(&t1));
      PetscCall(KSPDestroy(&ksp_fr));
      t_fr_solve = t1 - t0;
      PetscCall(ShiftedMatMat(L, Dshift, X_fr, Y));
      PetscCall(MatAYPX(Y, -1.0, B, SAME_NONZERO_PATTERN));
      PetscCall(MatGetColumnNorms(Y, NORM_2, rnorm_fr));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Frozen solve time: batched GMRES with the hierarchy built at c_frozen %g, %" PetscInt_FMT " block applies, %.4e s\n", (double)c_frozen, n_applies_fr, t_fr_solve));
    }

    // Check the true residuals of the batched solve
    PetscCall(ShiftedMatMat(L, Dshift, X_ms, Y));
    PetscCall(MatAYPX(Y, -1.0, B, SAME_NONZERO_PATTERN));
    {
      PetscReal rnorm[MAX_SHIFTS], bnorm[MAX_SHIFTS];
      PetscCall(MatGetColumnNorms(Y, NORM_2, rnorm));
      PetscCall(MatGetColumnNorms(B, NORM_2, bnorm));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%12s %10s %12s %12s %12s %12s\n", "c", "seq its", "multi its", "multi ||r||/||b||", "frozen its", "frozen ||r||/||b||"));
      for (j = 0; j < k; j++) {
        if (frozen) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%12.3e %10" PetscInt_FMT " %12" PetscInt_FMT " %12.3e %12" PetscInt_FMT " %12.3e\n", (double)c_values[j], its_seq[j], its_ms[j], (double)(rnorm[j] / bnorm[j]), its_fr[j], (double)(rnorm_fr[j] / bnorm[j])));
        else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%12.3e %10" PetscInt_FMT " %12" PetscInt_FMT " %12.3e\n", (double)c_values[j], its_seq[j], its_ms[j], (double)(rnorm[j] / bnorm[j])));
        PetscCheck(rnorm[j] <= 10 * rtol * bnorm[j], PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, "Multi-shift column %" PetscInt_FMT " did not converge", j);
        if (max_extra_its >= 0) PetscCheck(its_ms[j] - its_seq[j] <= max_extra_its, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Multi-shift column %" PetscInt_FMT " took %" PetscInt_FMT " iterations, sequential %" PetscInt_FMT, j, its_ms[j], its_seq[j]);
      }
    }

    // ~~~~~~~~~~~~
    // The same with an outer Richardson. The sequential solves are an update and a
    // KSPSolve for each shift, the multi-shift one is a single KSPMatSolve with the
    // shifted matshell as Amat, ie entirely through PETSc's block Richardson
    // ~~~~~~~~~~~~
    if (richardson) {
      KSP        ksp_rseq, ksp_rms;
      PC         pc_rms;
      Mat        A_rseq, A_rms, S;
      ShiftedCtx sctx;
      PetscInt   its_rseq[MAX_SHIFTS], its_rms, max_rseq = 0;
      PetscReal  rnorm[MAX_SHIFTS], bnorm[MAX_SHIFTS], max_rel_res = 0.0;
      PetscLogDouble t_rseq = 0, t_rms;

      PetscCall(MatDuplicate(L, MAT_COPY_VALUES, &A_rseq));
      PetscCall(MatDuplicate(L, MAT_COPY_VALUES, &A_rms));
      PetscCall(CreateKSP(A_rseq, A_rseq, KSPRICHARDSON, rtol, &ksp_rseq));
      PetscCall(CreateShiftedOperator(L, Dshift, &sctx, &S));
      PetscCall(CreateKSP(S, A_rms, KSPRICHARDSON, rtol, &ksp_rms));
      PetscCall(KSPGetPC(ksp_rms, &pc_rms));

      for (j = 0; j < k; j++) {
        PetscCall(MatDenseGetColumnVecRead(Dshift, j, &col));
        PetscCall(VecCopy(col, d_vec));
        PetscCall(MatDenseRestoreColumnVecRead(Dshift, j, &col));
        PetscCall(MatDenseGetColumnVecRead(B, j, &col));
        PetscCall(VecCopy(col, b));
        PetscCall(MatDenseRestoreColumnVecRead(B, j, &col));
        PetscCall(PetscBarrier(NULL));
        PetscCall(PetscTime(&t0));
        PetscCall(MatCopy(L, A_rseq, SAME_NONZERO_PATTERN));
        PetscCall(MatDiagonalSet(A_rseq, d_vec, ADD_VALUES));
        PetscCall(VecSet(x, 0.0));
        PetscCall(KSPSolve(ksp_rseq, b, x));
        PetscCall(PetscTime(&t1));
        t_rseq += t1 - t0;
        PetscCall(KSPGetIterationNumber(ksp_rseq, &its_rseq[j]));
        PetscCall(KSPGetConvergedReason(ksp_rseq, &reason));
        PetscCheck(reason > 0, PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, "Sequential Richardson %" PetscInt_FMT " did not converge, reason %d", j, (int)reason);
        max_rseq = PetscMax(max_rseq, its_rseq[j]);
      }

      PetscCall(PetscBarrier(NULL));
      PetscCall(PetscTime(&t0));
      PetscCall(PCAIRSetMultiShift(pc_rms, Dshift));
      PetscCall(MatZeroEntries(X_ms));
      PetscCall(KSPMatSolve(ksp_rms, B, X_ms));
      PetscCall(PetscTime(&t1));
      t_rms = t1 - t0;
      PetscCall(KSPGetIterationNumber(ksp_rms, &its_rms));
      PetscCall(KSPGetConvergedReason(ksp_rms, &reason));
      PetscCheck(reason > 0, PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, "Multi-shift Richardson did not converge, reason %d", (int)reason);

      // The block Richardson stops on the Frobenius norm of the whole block, so look at
      // the true residual of each column
      PetscCall(ShiftedMatMat(L, Dshift, X_ms, Y));
      PetscCall(MatAYPX(Y, -1.0, B, SAME_NONZERO_PATTERN));
      PetscCall(MatGetColumnNorms(Y, NORM_2, rnorm));
      PetscCall(MatGetColumnNorms(B, NORM_2, bnorm));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Richardson: %12s %10s %12s\n", "c", "seq its", "multi ||r||/||b||"));
      for (j = 0; j < k; j++) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "            %12.3e %10" PetscInt_FMT " %12.3e\n", (double)c_values[j], its_rseq[j], (double)(rnorm[j] / bnorm[j])));
        max_rel_res = PetscMax(max_rel_res, rnorm[j] / bnorm[j]);
      }
      // What the Frobenius norm stopping test guarantees for each column
      {
        PetscReal bnorm_f = 0.0;
        for (j = 0; j < k; j++) bnorm_f += bnorm[j] * bnorm[j];
        bnorm_f = PetscSqrtReal(bnorm_f);
        for (j = 0; j < k; j++) PetscCheck(rnorm[j] <= 1.01 * rtol * bnorm_f, PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, "Multi-shift Richardson column %" PetscInt_FMT " has residual %g above rtol ||B||_F", j, (double)rnorm[j]);
      }
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Richardson: multi-shift block its %" PetscInt_FMT ", max sequential its %" PetscInt_FMT ", max column ||r||/||b|| %.3e\n", its_rms, max_rseq, (double)max_rel_res));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Richardson solve time: sequential (k updates + k KSPSolve) %.4e s, multi-shift (set + KSPMatSolve) %.4e s\n", t_rseq, t_rms));
      // The block stops when ||R||_F <= rtol ||B||_F, it iterates for as long as the slowest
      // column needs to within a few iterations
      if (max_extra_its >= 0) PetscCheck(its_rms <= max_rseq + max_extra_its, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "The multi-shift Richardson took %" PetscInt_FMT " iterations, the slowest sequential %" PetscInt_FMT, its_rms, max_rseq);

      // The frozen comparison, a fresh PCAIR built from L + D(c_frozen) and never updated, with
      // the same shifted Amat, so column j is solved for L + D_j preconditioned with that
      if (frozen) {
        KSP            ksp_rfr;
        Mat            A_rfr;
        PetscInt       its_rfr;
        PetscLogDouble t_rfr;
        PetscReal      max_fr = 0.0;

        PetscCall(MatDuplicate(A_fr, MAT_COPY_VALUES, &A_rfr));
        PetscCall(CreateKSP(S, A_rfr, KSPRICHARDSON, rtol, &ksp_rfr));
        PetscCall(KSPSetTolerances(ksp_rfr, rtol, 1e-50, PETSC_DETERMINE, frozen_max_it));
        PetscCall(PetscBarrier(NULL));
        PetscCall(PetscTime(&t0));
        PetscCall(MatZeroEntries(X_fr));
        PetscCall(KSPMatSolve(ksp_rfr, B, X_fr));
        PetscCall(PetscTime(&t1));
        t_rfr = t1 - t0;
        PetscCall(KSPGetIterationNumber(ksp_rfr, &its_rfr));
        PetscCall(KSPGetConvergedReason(ksp_rfr, &reason));
        PetscCall(ShiftedMatMat(L, Dshift, X_fr, Y));
        PetscCall(MatAYPX(Y, -1.0, B, SAME_NONZERO_PATTERN));
        PetscCall(MatGetColumnNorms(Y, NORM_2, rnorm));
        for (j = 0; j < k; j++) max_fr = PetscMax(max_fr, rnorm[j] / bnorm[j]);
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Richardson: frozen block its %" PetscInt_FMT " (reason %d), max column ||r||/||b|| %.3e, time %.4e s\n", its_rfr, (int)reason, (double)max_fr, t_rfr));
        PetscCall(KSPDestroy(&ksp_rfr));
        PetscCall(MatDestroy(&A_rfr));
      }

      PetscCall(KSPDestroy(&ksp_rseq));
      PetscCall(KSPDestroy(&ksp_rms));
      PetscCall(MatDestroy(&S));
      PetscCall(MatDestroy(&sctx.LX));
      PetscCall(MatDestroy(&A_rseq));
      PetscCall(MatDestroy(&A_rms));
    }

    // ~~~~~~~~~~~~
    // Timings of the applies on their own
    // ~~~~~~~~~~~~
    // One multi-shift block apply on k columns
    PetscCall(PetscBarrier(NULL));
    PetscCall(PetscTime(&t0));
    for (i = 0; i < n_reps; i++) PetscCall(PCMatApply(pc_ms, B, Y));
    PetscCall(PetscTime(&t1));
    t_ms_apply = (t1 - t0) / n_reps;
    // One plain multiple rhs apply with a single shared shift, the upper bound on speed
    PetscCall(PetscBarrier(NULL));
    PetscCall(PetscTime(&t0));
    for (i = 0; i < n_reps; i++) PetscCall(PCMatApply(pc_seq, B, Y));
    PetscCall(PetscTime(&t1));
    t_mrhs_apply = (t1 - t0) / n_reps;
    // k single rhs applies
    PetscCall(PetscBarrier(NULL));
    PetscCall(PetscTime(&t0));
    for (i = 0; i < n_reps; i++) {
      for (j = 0; j < k; j++) {
        PetscCall(MatDenseGetColumnVecRead(B, j, &col));
        PetscCall(PCApply(pc_seq, col, y));
        PetscCall(MatDenseRestoreColumnVecRead(B, j, &col));
      }
    }
    PetscCall(PetscTime(&t1));
    t_seq_apply = (t1 - t0) / n_reps;
    // k diagonal updates
    PetscCall(PetscBarrier(NULL));
    PetscCall(PetscTime(&t0));
    for (j = 0; j < k; j++) {
      PetscCall(MatDenseGetColumnVecRead(Dshift, j, &col));
      PetscCall(MatCopy(L, A_seq, SAME_NONZERO_PATTERN));
      PetscCall(MatDiagonalSet(A_seq, col, ADD_VALUES));
      PetscCall(MatDenseRestoreColumnVecRead(Dshift, j, &col));
      PetscCall(PCSetUp(pc_seq));
    }
    PetscCall(PetscTime(&t1));
    t_seq_update = t1 - t0;

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "k = %" PetscInt_FMT ", n = %" PetscInt_FMT "\n", k, global_rows));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Solve time: sequential (k updates + k GMRES) %.4e s, multi-shift (set + batched GMRES, %" PetscInt_FMT " block applies) %.4e s\n", t_seq_solve, n_applies, t_ms_solve));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Setup time: k diagonal updates %.4e s, PCAIRSetMultiShift %.4e s\n", t_seq_update, t_ms_set));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Apply time: k PCApply %.4e s, multi-shift PCMatApply %.4e s, shared-shift PCMatApply %.4e s\n", t_seq_apply, t_ms_apply, t_mrhs_apply));
  }

  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(KSPDestroy(&ksp_ms));
  PetscCall(KSPDestroy(&ksp_seq));
  PetscCall(MatDestroy(&L));
  PetscCall(MatDestroy(&A_ms));
  PetscCall(MatDestroy(&A_fr));
  PetscCall(MatDestroy(&A_seq));
  PetscCall(MatDestroy(&Dshift));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&Y));
  PetscCall(MatDestroy(&Y2));
  PetscCall(MatDestroy(&X_ms));
  PetscCall(MatDestroy(&X_fr));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&d_vec));
  PetscCall(VecDestroy(&y));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}
