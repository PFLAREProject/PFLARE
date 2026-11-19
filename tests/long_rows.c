static char help[] = "Solves an upwind advection matrix with a few long rows, so the long row Kokkos kernels run.\n\n";

/*
  An upwind advection (plus a little diffusion) matrix on an n x n grid, with some of its
  rows turned into "hubs". Hub k keeps its grid entries and also couples to about
  hub_lengths[k] unknowns spread across the whole grid (and so across every rank). With
  -hub_cols (the default) each of those unknowns couples back to the hub as well, so the
  matrix has long columns too, which gives long rows in A_cf and the coarse matrices.

  Every extra entry is also added to the diagonal, so the matrix stays a diagonally
  dominant M-matrix that AIR converges on.

  The default hub lengths are picked to land in each of the row length regimes of the
  Kokkos kernels: the DDC ratios (more than 64 entries), the fixed sparsity powers
  kernel (more than 8 times the vector length) and the thread, team, dense block and
  iterative solves in lAIR/ISAI (16, 40 and 1024 entries).

  With -second_solve the matrix gets new values and is solved again with the same KSP,
  so the setup runs again on a matrix with a new state.
  Returns non-zero if either solve diverges.

    ./long_rows
    ./long_rows -pc_air_z_type sai
    mpiexec -n 2 ./long_rows -pc_type pflareinv -pc_pflareinv_type sai
*/
#include <petscksp.h>
#include "pflare.h"

#define MAX_HUBS 16

int main(int argc, char **args)
{
  Mat                A;
  KSP                ksp;
  PC                 pc;
  Vec                x, b;
  PetscInt           n = 64, N, i, j, k, m, row, col, row_start, row_end, local_rows;
  PetscInt           hub_lengths[MAX_HUBS] = {30, 150, 400, 1100}, n_hubs = 4, n_given;
  PetscInt           hub_rows[MAX_HUBS], stride[MAX_HUBS], *d_nnz, *o_nnz;
  PetscReal          diffusion = 0.01, hub_value = 0.5, arrow_value = 0.1;
  PetscBool          hub_cols = PETSC_TRUE, second_solve = PETSC_FALSE, own_hub, flg;
  KSPConvergedReason reason, reason_second = KSP_CONVERGED_ITS;

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  // n_given comes back as 0 when the option isn't given, so only take it when it is
  n_given = MAX_HUBS;
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-hub_lengths", hub_lengths, &n_given, &flg));
  if (flg) n_hubs = n_given;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-hub_cols", &hub_cols, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-second_solve", &second_solve, NULL));
  N = n * n;

  // Hubs spread evenly through the rows, each coupling to every stride-th unknown
  for (k = 0; k < n_hubs; k++) {
    PetscCheck(hub_lengths[k] > 0 && hub_lengths[k] <= N, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, \
               "Hub lengths must be in [1, n^2]");
    hub_rows[k] = ((2 * k + 1) * N) / (2 * n_hubs);
    stride[k]   = N / hub_lengths[k];
  }

  // Register the PFLARE types
  PCRegister_PFLARE();

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A, &row_start, &row_end));
  local_rows = row_end - row_start;

  // Upper bounds on the entries of each row in the local and nonlocal blocks
  PetscCall(PetscCalloc2(local_rows, &d_nnz, local_rows, &o_nnz));
  for (i = 0; i < local_rows; i++) {
    d_nnz[i] = PetscMin(5, local_rows);
    o_nnz[i] = PetscMin(4, N - local_rows);
  }
  for (k = 0; k < n_hubs; k++) {
    own_hub = (PetscBool)(hub_rows[k] >= row_start && hub_rows[k] < row_end);
    if (own_hub) {
      d_nnz[hub_rows[k] - row_start] = local_rows;
      o_nnz[hub_rows[k] - row_start] = N - local_rows;
    }
    if (!hub_cols) continue;
    for (m = 0; m < hub_lengths[k]; m++) {
      row = (k + m * stride[k]) % N;
      if (row < row_start || row >= row_end) continue;
      if (own_hub) d_nnz[row - row_start] = PetscMin(d_nnz[row - row_start] + 1, local_rows);
      else o_nnz[row - row_start] = PetscMin(o_nnz[row - row_start] + 1, N - local_rows);
    }
  }
  PetscCall(MatXAIJSetPreallocation(A, 1, d_nnz, o_nnz, NULL, NULL));
  PetscCall(PetscFree2(d_nnz, o_nnz));

  // Grid rows: upwind in x and y, with the boundary neighbours dropped
  for (row = row_start; row < row_end; row++) {
    PetscInt    cols[5], n_cols = 0;
    PetscScalar vals[5];

    i              = row % n;
    j              = row / n;
    cols[n_cols]   = row;
    vals[n_cols++] = 2.0 + 4.0 * diffusion;
    if (i > 0) {
      cols[n_cols]   = row - 1;
      vals[n_cols++] = -1.0 - diffusion;
    }
    if (j > 0) {
      cols[n_cols]   = row - n;
      vals[n_cols++] = -1.0 - diffusion;
    }
    if (i < n - 1) {
      cols[n_cols]   = row + 1;
      vals[n_cols++] = -diffusion;
    }
    if (j < n - 1) {
      cols[n_cols]   = row + n;
      vals[n_cols++] = -diffusion;
    }
    PetscCall(MatSetValues(A, 1, &row, n_cols, cols, vals, ADD_VALUES));
  }

  // Hub entries, each with its magnitude added to the diagonal of its row
  for (k = 0; k < n_hubs; k++) {
    const PetscScalar hub_entry = -hub_value / hub_lengths[k], arrow_entry = -arrow_value;

    own_hub = (PetscBool)(hub_rows[k] >= row_start && hub_rows[k] < row_end);
    for (m = 0; m < hub_lengths[k]; m++) {
      col = (k + m * stride[k]) % N;
      if (col == hub_rows[k]) continue;
      if (own_hub) {
        PetscCall(MatSetValue(A, hub_rows[k], col, hub_entry, ADD_VALUES));
        PetscCall(MatSetValue(A, hub_rows[k], hub_rows[k], -hub_entry, ADD_VALUES));
      }
      if (hub_cols && col >= row_start && col < row_end) {
        PetscCall(MatSetValue(A, col, hub_rows[k], arrow_entry, ADD_VALUES));
        PetscCall(MatSetValue(A, col, col, -arrow_entry, ADD_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecSet(b, 1.0));
  PetscCall(VecSet(x, 0.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPGMRES));
  PetscCall(KSPSetPCSide(ksp, PC_RIGHT));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCAIR));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(KSPGetConvergedReason(ksp, &reason));

  if (second_solve) {
    // New values on the same matrix, so the setup runs again on a matrix with a new state
    PetscCall(MatScale(A, 2.0));
    PetscCall(MatShift(A, 0.1));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(VecSet(x, 0.0));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(KSPGetConvergedReason(ksp, &reason_second));
  }

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  if (reason < 0 || reason_second < 0) return 1;
  return 0;
}
