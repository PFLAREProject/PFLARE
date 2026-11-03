static char help[] = "Records and replays the polynomial coefficients of PCAIR across a series of Pmats.\n\
\n\
The Pmats are L + c_k D, with L a 2D upwind advection-diffusion operator and D a\n\
varying diagonal, for each c_k in -c_values. Changing c_k changes the spectrum, so\n\
with the Newton basis the number of roots stored can change between them.\n\
\n\
Record: for each c_k in turn the Pmat values are changed in place and PCAIR is set up\n\
(with -pc_air_reuse_sparsity or -pc_air_diag_update), computing new coefficients. The\n\
iteration count, PCApply of a fixed vector and the coefficients of every inverse on every\n\
level are recorded.\n\
\n\
Replay: with PCAIRSetReusePolyCoeffs, the Pmats are visited again in reverse and then\n\
forward order, each time setting the recorded coefficients before the solve. The\n\
iteration count, PCApply and the stored coefficients must match those recorded.\n\
\n\
If -pc_air_reuse_poly_coeffs is given, it applies from the first setup, so the record\n\
must keep the coefficients from the first Pmat for every later one, and the first setup\n\
must compute the same coefficients as a PC without it. The number of\n\
levels must not change between setups, at any -pc_air_reuse_amount. Only the inverses\n\
that are computed as polynomials can have coefficients returned.\n\
\n\
  -c_values c1,c2,...  : the c_k (default 0.5,8,50)\n\
  -require_resize      : error if no replay changed the number of stored roots\n\
  -print_roots         : print the recorded roots\n\
  -n                   : grid size (default 10)\n\
  -alpha               : diffusion (default 1)\n\
The polynomial order defaults to 16, high enough that the number of Newton roots\n\
changes between the default c_k; it can be changed with -pc_air_poly_order.\n\n";

#include <petscksp.h>
#include "pflare.h"

#define MAX_C   16
#define MAX_INV 256

typedef struct {
  PetscInt  level;
  int       which;
  PetscInt  rows[MAX_C], cols[MAX_C];
  PetscReal *coeffs[MAX_C];
} InvRecord;

static PetscErrorCode BuildL(PetscInt n, PetscReal alpha, Mat *L, Vec *d)
{
  PetscInt Istart, Iend, row, i, j;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(PETSC_COMM_WORLD, L));
  PetscCall(MatSetSizes(*L, PETSC_DECIDE, PETSC_DECIDE, n * n, n * n));
  PetscCall(MatSetFromOptions(*L));
  PetscCall(MatSeqAIJSetPreallocation(*L, 5, NULL));
  PetscCall(MatMPIAIJSetPreallocation(*L, 5, NULL, 5, NULL));
  PetscCall(MatGetOwnershipRange(*L, &Istart, &Iend));
  for (row = Istart; row < Iend; row++) {
    i = row % n;
    j = row / n;
    PetscCall(MatSetValue(*L, row, row, 2.0 + 4.0 * alpha, INSERT_VALUES));
    if (i > 0) PetscCall(MatSetValue(*L, row, row - 1, -1.0 - alpha, INSERT_VALUES));
    if (j > 0) PetscCall(MatSetValue(*L, row, row - n, -1.0 - alpha, INSERT_VALUES));
    if (i < n - 1) PetscCall(MatSetValue(*L, row, row + 1, -alpha, INSERT_VALUES));
    if (j < n - 1) PetscCall(MatSetValue(*L, row, row + n, -alpha, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*L, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*L, MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateVecs(*L, d, NULL));
  for (row = Istart; row < Iend; row++) PetscCall(VecSetValue(*d, row, 1.0 + 0.5 * PetscSinReal((PetscReal)row), INSERT_VALUES));
  PetscCall(VecAssemblyBegin(*d));
  PetscCall(VecAssemblyEnd(*d));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* P = L + c D, changing the values of P in place */
static PetscErrorCode SetPmat(Mat L, Vec d, PetscReal c, Mat P, Vec work)
{
  PetscFunctionBeginUser;
  PetscCall(MatCopy(L, P, SAME_NONZERO_PATTERN));
  PetscCall(VecCopy(d, work));
  PetscCall(VecScale(work, c));
  PetscCall(MatDiagonalSet(P, work, ADD_VALUES));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat         L, P;
  Vec         d, work, x, b_solve, b, y;
  Vec         y_rec[MAX_C];
  KSP         ksp, ksp_fresh = NULL;
  PC          pc, pc_fresh = NULL;
  PetscInt    n = 10, n_c = MAX_C, k, k_idx, i, num_levels, lvl, its, its_rec[MAX_C], n_inv = 0, n_resized = 0, n_sets = 0, rows, cols, pass;
  PetscReal   alpha = 1.0, c_values[MAX_C] = {0.5, 8.0, 50.0}, norm_diff, norm_ref, max_rel_diff = 0.0;
  PetscBool   flg, require_resize = PETSC_FALSE, print_roots = PETSC_FALSE, reuse_sparsity, diag_update, same, reuse_from_start;
  PetscRandom rctx;
  PetscReal  *ptr;
  PetscErrorCode ierr;
  InvRecord  *inv;
  PCAIRZType  z_type;
  PetscReal   strong_r;
  PetscBool   full_smooth, dropped_used, aff_poly;
  PetscInt    num_levels_k;
  PCPFLAREINVType inv_type;
  int         which_list[3] = {COEFFS_INV_AFF, COEFFS_INV_AFF_DROPPED, COEFFS_INV_ACC};

  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PCRegister_PFLARE();
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-alpha", &alpha, NULL));
  PetscCall(PetscOptionsGetRealArray(NULL, NULL, "-c_values", c_values, &n_c, &flg));
  if (!flg) n_c = 3;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-require_resize", &require_resize, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-print_roots", &print_roots, NULL));
  PetscCall(PetscCalloc1(MAX_INV, &inv));

  PetscCall(BuildL(n, alpha, &L, &d));
  PetscCall(MatDuplicate(L, MAT_COPY_VALUES, &P));
  PetscCall(VecDuplicate(d, &work));
  PetscCall(MatCreateVecs(L, &x, &b_solve));
  PetscCall(VecDuplicate(b_solve, &b));
  PetscCall(VecDuplicate(b_solve, &y));
  PetscCall(VecSet(b_solve, 1.0));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(PetscRandomSetSeed(rctx, 7));
  PetscCall(PetscRandomSeed(rctx));
  PetscCall(VecSetRandom(b, rctx));
  PetscCall(PetscRandomDestroy(&rctx));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, P, P));
  PetscCall(KSPSetType(ksp, KSPGMRES));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCAIR));
  PetscCall(PCAIRSetInverseType(pc, PFLAREINV_NEWTON));
  PetscCall(PCAIRSetPolyOrder(pc, 16));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(PCAIRGetReusePolyCoeffs(pc, &reuse_from_start));

  /* ~~~~~~~~ Record ~~~~~~~~ */
  for (k = 0; k < n_c; k++) {
    PetscCall(SetPmat(L, d, c_values[k], P, work));
    PetscCall(VecSet(x, 0.0));
    PetscCall(KSPSolve(ksp, b_solve, x));
    PetscCall(KSPGetIterationNumber(ksp, &its_rec[k]));
    PetscCall(VecDuplicate(b, &y_rec[k]));
    PetscCall(PCApply(pc, b, y_rec[k]));

    if (k == 0) {
      PetscCall(PCAIRGetReuseSparsity(pc, &reuse_sparsity));
      PetscCall(PCAIRGetDiagUpdate(pc, &diag_update));
      PetscCheck(reuse_sparsity || diag_update, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Use -pc_air_reuse_sparsity or -pc_air_diag_update");
      PetscCall(PCAIRGetNumLevels(pc, &num_levels));
      /* The dropped A_ff inverse is only computed with AIRG and a strong R threshold (or
         full smoothing); otherwise its storage exists but is never written, so skip it */
      PetscCall(PCAIRGetZType(pc, &z_type));
      PetscCall(PCAIRGetStrongRThreshold(pc, &strong_r));
      PetscCall(PCAIRGetFullSmoothingUpAndDown(pc, &full_smooth));
      dropped_used = (PetscBool)(z_type == AIR_Z_PRODUCT && (strong_r != 0.0 || full_smooth));
      PetscCall(PCAIRGetInverseType(pc, &inv_type));
      aff_poly = (PetscBool)(inv_type != PFLAREINV_SAI && inv_type != PFLAREINV_ISAI && inv_type != PFLAREINV_WJACOBI &&
                             inv_type != PFLAREINV_JACOBI && inv_type != PFLAREINV_L1JACOBI);
      /* Find every inverse that stores coefficients; errors are returned while probing */
      for (lvl = 0; lvl < num_levels; lvl++) {
        for (i = 0; i < 3; i++) {
          PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
          ierr = PCAIRGetPolyCoeffs(pc, lvl, which_list[i], &ptr, &rows, &cols);
          PetscCall(PetscPopErrorHandler());
          /* Nothing computes these, so there must be no coefficients to return */
          if (which_list[i] == COEFFS_INV_AFF_DROPPED && !dropped_used) PetscCheck(ierr != PETSC_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Coefficients returned for the dropped A_ff inverse on petsc_level %" PetscInt_FMT ", which isn't computed", lvl);
          if (which_list[i] != COEFFS_INV_ACC && !aff_poly) PetscCheck(ierr != PETSC_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Coefficients returned for which_inverse %d on petsc_level %" PetscInt_FMT ", which isn't a polynomial", which_list[i], lvl);
          if (ierr == PETSC_SUCCESS) {
            PetscCheck(n_inv < MAX_INV, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Too many inverses");
            inv[n_inv].level = lvl;
            inv[n_inv++].which = which_list[i];
          }
        }
      }
      PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
      ierr = PCAIRGetPolyCoeffs(pc, 0, COEFFS_INV_COARSE, &ptr, &rows, &cols);
      PetscCall(PetscPopErrorHandler());
      if (ierr == PETSC_SUCCESS) {
        inv[n_inv].level = 0;
        inv[n_inv++].which = COEFFS_INV_COARSE;
      }
    }
    /* The reuse keeps the hierarchy, so the number of levels can't change */
    PetscCall(PCAIRGetNumLevels(pc, &num_levels_k));
    PetscCheck(num_levels_k == num_levels || !reuse_sparsity, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "With reuse the number of levels changed from %" PetscInt_FMT " to %" PetscInt_FMT " for c = %g", num_levels, num_levels_k, (double)c_values[k]);
    for (i = 0; i < n_inv; i++) {
      PetscCall(PCAIRGetPolyCoeffs(pc, inv[i].level, inv[i].which, &ptr, &inv[i].rows[k], &inv[i].cols[k]));
      if (print_roots) {
        PetscInt r;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "c=%g petsc_level %" PetscInt_FMT " which %d roots:", (double)c_values[k], inv[i].level, inv[i].which));
        for (r = 0; r < inv[i].rows[k]; r++) PetscCall(PetscPrintf(PETSC_COMM_WORLD, " (%.4g,%.4g)", (double)ptr[r], inv[i].cols[k] == 2 ? (double)ptr[r + inv[i].rows[k]] : 0.0));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
      }
      PetscCall(PetscMalloc1(inv[i].rows[k] * inv[i].cols[k], &inv[i].coeffs[k]));
      PetscCall(PetscArraycpy(inv[i].coeffs[k], ptr, inv[i].rows[k] * inv[i].cols[k]));
      /* The first setup has nothing to reuse, so it must compute exactly what a PC without reuse does */
      if (reuse_from_start && k == 0) {
        PetscReal *ptr_fresh;
        PetscInt   rows_fresh, cols_fresh;
        if (i == 0) {
          PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_fresh));
          PetscCall(KSPSetOperators(ksp_fresh, P, P));
          PetscCall(KSPSetType(ksp_fresh, KSPGMRES));
          PetscCall(KSPGetPC(ksp_fresh, &pc_fresh));
          PetscCall(PCSetType(pc_fresh, PCAIR));
          PetscCall(PCAIRSetInverseType(pc_fresh, PFLAREINV_NEWTON));
          PetscCall(PCAIRSetPolyOrder(pc_fresh, 16));
          PetscCall(KSPSetFromOptions(ksp_fresh));
          PetscCall(PCAIRSetReusePolyCoeffs(pc_fresh, PETSC_FALSE));
          PetscCall(KSPSetUp(ksp_fresh));
        }
        PetscCall(PCAIRGetPolyCoeffs(pc_fresh, inv[i].level, inv[i].which, &ptr_fresh, &rows_fresh, &cols_fresh));
        PetscCheck(rows_fresh == inv[i].rows[0] && cols_fresh == inv[i].cols[0], PETSC_COMM_SELF, PETSC_ERR_PLIB, "First setup stored a different size on petsc_level %" PetscInt_FMT " which_inverse %d than without -pc_air_reuse_poly_coeffs", inv[i].level, inv[i].which);
        PetscCall(PetscArraycmp(ptr, ptr_fresh, rows_fresh * cols_fresh, &same));
        PetscCheck(same, PETSC_COMM_SELF, PETSC_ERR_PLIB, "First setup computed different coefficients on petsc_level %" PetscInt_FMT " which_inverse %d than without -pc_air_reuse_poly_coeffs", inv[i].level, inv[i].which);
        if (i == n_inv - 1) PetscCall(KSPDestroy(&ksp_fresh));
      }
      /* Reusing the coefficients from the first setup, so none of the later setups can have computed them */
      if (reuse_from_start && k > 0) {
        PetscCheck(inv[i].rows[k] == inv[i].rows[0] && inv[i].cols[k] == inv[i].cols[0], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stored size on petsc_level %" PetscInt_FMT " which_inverse %d changed with -pc_air_reuse_poly_coeffs", inv[i].level, inv[i].which);
        PetscCall(PetscArraycmp(ptr, inv[i].coeffs[0], inv[i].rows[k] * inv[i].cols[k], &same));
        PetscCheck(same, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Coefficients on petsc_level %" PetscInt_FMT " which_inverse %d changed with -pc_air_reuse_poly_coeffs, c = %g", inv[i].level, inv[i].which, (double)c_values[k]);
      }
    }
  }

  /* Print the number of rows of each inverse for each c_k */
  for (i = 0; i < n_inv; i++) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  petsc_level %2" PetscInt_FMT " which_inverse %d rows:", inv[i].level, inv[i].which));
    for (k = 0; k < n_c; k++) PetscCall(PetscPrintf(PETSC_COMM_WORLD, " %" PetscInt_FMT, inv[i].rows[k]));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
  }

  /* ~~~~~~~~ Replay, in reverse and then forward order ~~~~~~~~ */
  PetscCall(PCAIRSetReusePolyCoeffs(pc, PETSC_TRUE));
  for (pass = 0; pass < 2; pass++) {
    for (k_idx = 0; k_idx < n_c; k_idx++) {
      k = pass == 0 ? n_c - 1 - k_idx : k_idx;
      /* Skip replaying the set we are already on at the start */
      if (pass == 0 && k_idx == 0) continue;

      for (i = 0; i < n_inv; i++) {
        PetscCall(PCAIRGetPolyCoeffs(pc, inv[i].level, inv[i].which, &ptr, &rows, &cols));
        if (rows != inv[i].rows[k]) n_resized++;
        n_sets++;
        PetscCall(PCAIRSetPolyCoeffs(pc, inv[i].level, inv[i].which, inv[i].coeffs[k], inv[i].rows[k], inv[i].cols[k]));
      }
      PetscCall(SetPmat(L, d, c_values[k], P, work));
      PetscCall(VecSet(x, 0.0));
      PetscCall(KSPSolve(ksp, b_solve, x));
      PetscCall(KSPGetIterationNumber(ksp, &its));
      PetscCheck(its == its_rec[k], PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Replay of c = %g (pass %" PetscInt_FMT ") took %" PetscInt_FMT " iterations, recorded %" PetscInt_FMT, (double)c_values[k], pass, its, its_rec[k]);

      PetscCall(PCApply(pc, b, y));
      PetscCall(VecAXPY(y, -1.0, y_rec[k]));
      PetscCall(VecNorm(y, NORM_2, &norm_diff));
      PetscCall(VecNorm(y_rec[k], NORM_2, &norm_ref));
      max_rel_diff = PetscMax(max_rel_diff, norm_diff / norm_ref);
      PetscCheck(norm_diff <= 1e3 * PETSC_MACHINE_EPSILON * norm_ref, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Replay of c = %g (pass %" PetscInt_FMT ") PCApply differs from recorded, relative difference %g", (double)c_values[k], pass, (double)(norm_diff / norm_ref));

      /* The setup must have used, not recomputed, the coefficients */
      for (i = 0; i < n_inv; i++) {
        PetscCall(PCAIRGetPolyCoeffs(pc, inv[i].level, inv[i].which, &ptr, &rows, &cols));
        PetscCheck(rows == inv[i].rows[k] && cols == inv[i].cols[k], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stored size changed by the setup");
        PetscCall(PetscArraycmp(ptr, inv[i].coeffs[k], rows * cols, &same));
        PetscCheck(same, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stored coefficients changed by the setup on petsc_level %" PetscInt_FMT " which_inverse %d", inv[i].level, inv[i].which);
      }
    }
  }

  /* n_resized can differ between ranks only if the sizes do, which they don't */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "OK: %" PetscInt_FMT " inverses, %" PetscInt_FMT " of %" PetscInt_FMT " sets resized, max relative PCApply difference %.3g, %s\n", n_inv, n_resized, n_sets, (double)max_rel_diff,
                        reuse_sparsity ? "reuse sparsity" : "diagonal update"));
  PetscCheck(!require_resize || n_resized > 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "No replay changed the number of stored roots, so this does not test resizing");

  for (i = 0; i < n_inv; i++)
    for (k = 0; k < n_c; k++) PetscCall(PetscFree(inv[i].coeffs[k]));
  PetscCall(PetscFree(inv));
  for (k = 0; k < n_c; k++) PetscCall(VecDestroy(&y_rec[k]));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b_solve));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&d));
  PetscCall(VecDestroy(&work));
  PetscCall(MatDestroy(&P));
  PetscCall(MatDestroy(&L));
  PetscCall(PetscFinalize());
  return 0;
}
