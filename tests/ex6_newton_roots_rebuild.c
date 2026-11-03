static char help[] = "Rebuilds Newton polynomial inverses from sets of roots of different lengths.\n\
\n\
Sets up PCPFLAREINV with a Newton basis on a 2D upwind advection-diffusion matrix, then\n\
builds variants of the computed roots with more or fewer roots: extra copies of real\n\
roots and complex pairs (as the newton type adds for stability), zero roots at the end\n\
and in the middle, and the roots truncated back to the polynomial order + 1.\n\
For each variant the inverse is rebuilt (reusing the existing inverse matrix, which was\n\
built from a different number of roots) and checked:\n\
  - exactness, when the inverse should be exactly the polynomial (matrix-free, or\n\
    assembled with no more roots than the sparsity order, capped at the polynomial\n\
    order, plus one):\n\
        A p(A) b = b - prod_i (I - A/theta_i) b\n\
    with the product computed here with matvecs, skipping zero roots and taking\n\
    complex conjugate pairs together\n\
  - history independence: rebuilding from the same roots gives the same inverse,\n\
    whatever set of roots the reused inverse was built from before\n\
  - extra roots: extra copies of roots change the inverse, while zero roots don't\n\
  - fixed sparsity: the nonzeros in an assembled inverse are those of the matrix to the\n\
    power of the sparsity order, capped at the polynomial order, whatever the roots\n\
\n\
With -dense the matrix has a full nonzero pattern, so any sparsity order of at least 1\n\
holds every power of the matrix and all the variants must be exact. It also adds variants\n\
with fixed roots, so the layouts tested don't depend on the computed roots.\n\
\n\
  -n : grid size (default 10)\n\
  -c : diagonal shift (default 1)\n\
  -dense : use a matrix with a full nonzero pattern\n\
  -zero_root_first : the first setup is on a badly scaled matrix, which gives zero roots at\n\
                     the end. The inverse must still have its full sparsity, as the setups\n\
                     after it use the unscaled matrix and reuse the inverse\n\
  -verbose : print each rebuild\n\n";

#include <petscksp.h>
#include "pflare.h"

#define MAX_VARIANTS 11

static PetscErrorCode BuildMatrix(PetscInt n, PetscReal c, PetscBool dense, Mat *A)
{
  PetscInt  Istart, Iend, row, i, j;
  PetscReal alpha = 1.0;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(PETSC_COMM_WORLD, A));
  PetscCall(MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, n * n, n * n));
  PetscCall(MatSetFromOptions(*A));
  PetscCall(MatSeqAIJSetPreallocation(*A, dense ? n * n : 5, NULL));
  PetscCall(MatMPIAIJSetPreallocation(*A, dense ? n * n : 5, NULL, dense ? n * n : 5, NULL));
  PetscCall(MatGetOwnershipRange(*A, &Istart, &Iend));
  for (row = Istart; row < Iend; row++) {
    if (dense) {
      /* Non-symmetric and diagonally dominant */
      for (j = 0; j < n * n; j++) PetscCall(MatSetValue(*A, row, j, j == row ? 6.0 + 0.3 * row : (j < row ? -0.4 : -0.1) / (1.0 + PetscAbsInt(row - j)), INSERT_VALUES));
      continue;
    }
    i = row % n;
    j = row / n;
    PetscCall(MatSetValue(*A, row, row, 2.0 + 4.0 * alpha + c * (1.0 + 0.5 * PetscSinReal((PetscReal)row)), INSERT_VALUES));
    if (i > 0) PetscCall(MatSetValue(*A, row, row - 1, -1.0 - alpha, INSERT_VALUES));
    if (j > 0) PetscCall(MatSetValue(*A, row, row - n, -1.0 - alpha, INSERT_VALUES));
    if (i < n - 1) PetscCall(MatSetValue(*A, row, row + 1, -alpha, INSERT_VALUES));
    if (j < n - 1) PetscCall(MatSetValue(*A, row, row + n, -alpha, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The number of nonzeros in A^power, which needs A to have a full diagonal */
static PetscErrorCode NnzOfPower(Mat A, PetscInt power, PetscLogDouble *nz)
{
  Mat     B, C;
  MatInfo info;

  PetscFunctionBeginUser;
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  for (PetscInt k = 1; k < power; k++) {
    PetscCall(MatMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C));
    PetscCall(MatDestroy(&B));
    B = C;
  }
  PetscCall(MatGetInfo(B, MAT_GLOBAL_SUM, &info));
  *nz = info.nz_used;
  PetscCall(MatDestroy(&B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* r = prod_i (I - A/theta_i) b, skipping zero roots, with conjugate pairs (a +- ib) as
   I - 2a/(a^2+b^2) A + A^2/(a^2+b^2) */
static PetscErrorCode ResidualPoly(Mat A, const PetscReal *roots, PetscInt nr, Vec b, Vec r)
{
  Vec       t, t2;
  PetscInt  i = 0;
  PetscReal a, im, s;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(b, &t));
  PetscCall(VecDuplicate(b, &t2));
  PetscCall(VecCopy(b, r));
  while (i < nr) {
    a  = roots[i];
    im = roots[nr + i];
    if (im == 0.0) {
      if (a != 0.0) {
        /* r = r - A r / a */
        PetscCall(MatMult(A, r, t));
        PetscCall(VecAXPY(r, -1.0 / a, t));
      }
      i++;
    } else {
      s = a * a + im * im;
      PetscCall(MatMult(A, r, t));
      PetscCall(MatMult(A, t, t2));
      PetscCall(VecAXPBYPCZ(r, -2.0 * a / s, 1.0 / s, 1.0, t, t2));
      i += 2;
    }
  }
  PetscCall(VecDestroy(&t));
  PetscCall(VecDestroy(&t2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Sets row i of the roots in dst, which has nd rows */
static void Put(PetscReal *dst, PetscInt nd, PetscInt i, PetscReal re, PetscReal im)
{
  dst[i]      = re;
  dst[nd + i] = im;
}

int main(int argc, char **args)
{
  Mat              A, A_unscaled = NULL, inv_mat, inv_ref[MAX_VARIANTS];
  Vec              b, y, r, Ay, y_ref[MAX_VARIANTS];
  PC               pc;
  PetscInt         rstart, rend, n = 10, n0, cols0, i, v, w, nv = 0, po, sp, nr[MAX_VARIANTS], first_real = -1, first_complex = -1, k;
  PetscReal        c = 1.0, *roots0, *ptr, *var[MAX_VARIANTS], norm_b, norm_e, max_exact = 0.0, max_hist = 0.0, tol;
  PetscBool        matrix_free, exact, is_diag, dense = PETSC_FALSE, zero_root_first = PETSC_FALSE, verbose = PETSC_FALSE;
  PetscLogDouble   nz_power = 0.0;
  MatInfo          info;
  PCPFLAREINVType  type;
  PetscRandom      rctx;
  const char      *names[MAX_VARIANTS];

  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PCRegister_PFLARE();
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-c", &c, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-verbose", &verbose, NULL));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-dense", &dense, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-zero_root_first", &zero_root_first, NULL));

  PetscCall(BuildMatrix(n, c, dense, &A));
  PetscCall(MatCreateVecs(A, &y, &b));
  PetscCall(VecDuplicate(b, &r));
  PetscCall(VecDuplicate(b, &Ay));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(PetscRandomSetSeed(rctx, 3));
  PetscCall(PetscRandomSeed(rctx));
  PetscCall(VecSetRandom(b, rctx));
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(VecNorm(b, NORM_2, &norm_b));

  PetscCall(PCCreate(PETSC_COMM_WORLD, &pc));
  PetscCall(PCSetOperators(pc, A, A));
  PetscCall(PCSetType(pc, PCPFLAREINV));
  PetscCall(PCPFLAREINVSetType(pc, PFLAREINV_NEWTON));
  PetscCall(PCSetFromOptions(pc));
  if (zero_root_first) {
    /* Scale most of the rows so the Krylov space is too small for all the roots */
    Vec scale;

    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &A_unscaled));
    PetscCall(VecDuplicate(b, &scale));
    PetscCall(VecGetOwnershipRange(scale, &rstart, &rend));
    for (i = rstart; i < rend; i++) PetscCall(VecSetValue(scale, i, i % 3 == 0 ? 1.0 : 1e-10, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(scale));
    PetscCall(VecAssemblyEnd(scale));
    PetscCall(MatDiagonalScale(A, scale, NULL));
    PetscCall(VecDestroy(&scale));
  }
  PetscCall(PCSetUp(pc));
  PetscCall(PCPFLAREINVGetType(pc, &type));
  PetscCall(PCPFLAREINVGetMatrixFree(pc, &matrix_free));
  PetscCall(PCPFLAREINVGetPolyOrder(pc, &po));
  PetscCall(PCPFLAREINVGetSparsityOrder(pc, &sp));
  PetscCheck(type == PFLAREINV_NEWTON || type == PFLAREINV_NEWTON_NO_EXTRA, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Needs a Newton type");
  if (!matrix_free && PetscMin(sp, po) > 0) PetscCall(NnzOfPower(A, PetscMin(sp, po), &nz_power));

  if (zero_root_first) {
    PetscCall(PCPFLAREINVGetPolyCoeffs(pc, &ptr, &n0, &cols0));
    PetscCheck(ptr[n0 - 1] == 0.0 && ptr[2 * n0 - 1] == 0.0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "The first setup needs a zero root at the end to test this");
    if (verbose) {
      for (i = 0, k = 0; i < n0; i++) k += (ptr[i] == 0.0 && ptr[n0 + i] == 0.0) ? 1 : 0;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "first setup has %" PetscInt_FMT " zero roots of %" PetscInt_FMT "\n", k, n0));
    }
    if (!matrix_free && PetscMin(sp, po) > 0) {
      PetscCall(PCPFLAREINVGetInverseMat(pc, &inv_mat));
      PetscCall(MatGetInfo(inv_mat, MAT_GLOBAL_SUM, &info));
      PetscCheck(info.nz_used == nz_power, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Zero roots gave the inverse %g nonzeros rather than %g", (double)info.nz_used, (double)nz_power);
    }
    /* The next setup has non-zero roots and reuses the inverse */
    PetscCall(MatCopy(A_unscaled, A, SAME_NONZERO_PATTERN));
    PetscCall(MatDestroy(&A_unscaled));
    PetscCall(PCSetOperators(pc, A, A));
    PetscCall(PCSetUp(pc));
  }

  PetscCall(PCPFLAREINVGetPolyCoeffs(pc, &ptr, &n0, &cols0));
  PetscCall(PetscMalloc1(2 * n0, &roots0));
  PetscCall(PetscArraycpy(roots0, ptr, 2 * n0));
  for (i = 0; i < n0; i++) {
    if (roots0[n0 + i] == 0.0 && roots0[i] != 0.0 && first_real < 0) first_real = i;
    if (roots0[n0 + i] != 0.0 && first_complex < 0) first_complex = i;
  }

  /* ~~~~~~~~ The variants ~~~~~~~~ */
  /* 0: the roots as computed */
  names[nv] = "computed";
  nr[nv]    = n0;
  PetscCall(PetscMalloc1(2 * nr[nv], &var[nv]));
  PetscCall(PetscArraycpy(var[nv], roots0, 2 * n0));
  nv++;
  /* 1: two exact-zero roots appended */
  names[nv] = "zeros appended";
  nr[nv]    = n0 + 2;
  PetscCall(PetscCalloc1(2 * nr[nv], &var[nv]));
  for (i = 0; i < n0; i++) Put(var[nv], nr[nv], i, roots0[i], roots0[n0 + i]);
  nv++;
  /* 2: extra copies, as newton adds: a real root and a complex pair (if there are any) */
  names[nv] = "extra copies appended";
  nr[nv]    = n0 + (first_real >= 0 ? 1 : 0) + (first_complex >= 0 ? 2 : 0);
  PetscCall(PetscCalloc1(2 * nr[nv], &var[nv]));
  for (i = 0; i < n0; i++) Put(var[nv], nr[nv], i, roots0[i], roots0[n0 + i]);
  k = n0;
  if (first_complex >= 0) {
    Put(var[nv], nr[nv], k++, roots0[first_complex], PetscAbsReal(roots0[n0 + first_complex]));
    Put(var[nv], nr[nv], k++, roots0[first_complex], -PetscAbsReal(roots0[n0 + first_complex]));
  }
  if (first_real >= 0) Put(var[nv], nr[nv], k++, roots0[first_real], 0.0);
  nv++;
  /* 3: extra copies inserted at the front, so every later root moves */
  names[nv] = "extra copies at the front";
  nr[nv]    = nr[nv - 1];
  PetscCall(PetscCalloc1(2 * nr[nv], &var[nv]));
  k = 0;
  if (first_complex >= 0) {
    Put(var[nv], nr[nv], k++, roots0[first_complex], PetscAbsReal(roots0[n0 + first_complex]));
    Put(var[nv], nr[nv], k++, roots0[first_complex], -PetscAbsReal(roots0[n0 + first_complex]));
  }
  if (first_real >= 0) Put(var[nv], nr[nv], k++, roots0[first_real], 0.0);
  for (i = 0; i < n0; i++) Put(var[nv], nr[nv], k++, roots0[i], roots0[n0 + i]);
  nv++;
  /* 4: a zero root in the middle (after the first root, keeping conjugate pairs together) */
  names[nv] = "zero root in the middle";
  nr[nv]    = n0 + 1;
  PetscCall(PetscCalloc1(2 * nr[nv], &var[nv]));
  k = (roots0[n0] != 0.0) ? 2 : 1;
  if (k > n0) k = n0;
  for (i = 0; i < k; i++) Put(var[nv], nr[nv], i, roots0[i], roots0[n0 + i]);
  Put(var[nv], nr[nv], k, 0.0, 0.0);
  for (i = k; i < n0; i++) Put(var[nv], nr[nv], i + 1, roots0[i], roots0[n0 + i]);
  nv++;
  /* 5: twice as many roots, the computed set repeated */
  names[nv] = "computed set twice";
  nr[nv]    = 2 * n0;
  PetscCall(PetscCalloc1(2 * nr[nv], &var[nv]));
  for (i = 0; i < n0; i++) {
    Put(var[nv], nr[nv], i, roots0[i], roots0[n0 + i]);
    Put(var[nv], nr[nv], n0 + i, roots0[i], roots0[n0 + i]);
  }
  nv++;
  /* 6: truncated to the polynomial order + 1 roots, if that is fewer and doesn't split a pair */
  /* The last root kept must be real or the second of a conjugate pair */
  if (n0 > po + 1 && (roots0[n0 + po] == 0.0 || (po > 0 && roots0[po - 1] == roots0[po] && roots0[n0 + po - 1] == -roots0[n0 + po]))) {
    names[nv] = "truncated to order + 1";
    nr[nv]    = po + 1;
    PetscCall(PetscCalloc1(2 * nr[nv], &var[nv]));
    for (i = 0; i <= po; i++) Put(var[nv], nr[nv], i, roots0[i], roots0[n0 + i]);
    nv++;
  }
  /* A single root, which only has to work with a sparsity order of 0 as the matrix then stays diagonal */
  if (sp == 0 && po > 0) {
    names[nv] = "single root";
    nr[nv]    = 1;
    PetscCall(PetscCalloc1(2 * nr[nv], &var[nv]));
    Put(var[nv], nr[nv], 0, roots0[first_real >= 0 ? first_real : 0], 0.0);
    nv++;
  }
  if (dense && sp > 0) {
    /* A zero root straight after a real root */
    names[nv] = "real root then zero root";
    nr[nv]    = po + 3;
    PetscCall(PetscCalloc1(2 * nr[nv], &var[nv]));
    Put(var[nv], nr[nv], 0, 6.0, 0.0);
    for (i = 2; i < nr[nv]; i++) Put(var[nv], nr[nv], i, 6.0 + 0.5 * i, 0.0);
    nv++;
    /* The first root of a conjugate pair falls on the power of the sparsity order */
    names[nv] = "pair across the sparsity order";
    nr[nv]    = PetscMax(po, sp) + 3;
    PetscCall(PetscCalloc1(2 * nr[nv], &var[nv]));
    for (i = 0; i < sp; i++) Put(var[nv], nr[nv], i, 6.0 + 0.5 * i, 0.0);
    Put(var[nv], nr[nv], sp, 7.0, 1.0);
    Put(var[nv], nr[nv], sp + 1, 7.0, -1.0);
    for (i = sp + 2; i < nr[nv]; i++) Put(var[nv], nr[nv], i, 6.0 + 0.5 * i, 0.0);
    nv++;
    /* Starting with a pair, with a real root between it and the next pair */
    names[nv] = "pairs first";
    nr[nv]    = po + 6;
    PetscCall(PetscCalloc1(2 * nr[nv], &var[nv]));
    Put(var[nv], nr[nv], 0, 7.0, 1.0);
    Put(var[nv], nr[nv], 1, 7.0, -1.0);
    Put(var[nv], nr[nv], 2, 6.0, 0.0);
    Put(var[nv], nr[nv], 3, 8.0, 2.0);
    Put(var[nv], nr[nv], 4, 8.0, -2.0);
    for (i = 5; i < nr[nv]; i++) Put(var[nv], nr[nv], i, 6.0 + 0.5 * i, 0.0);
    nv++;
  }

  PetscCall(PCPFLAREINVSetReusePolyCoeffs(pc, PETSC_TRUE));
  tol = 1e4 * PETSC_MACHINE_EPSILON;

  /* ~~~~~~~~ Pass 1: rebuild from each variant in turn, check exactness and save the result ~~~~~~~~ */
  for (v = 0; v < nv; v++) {
    if (verbose) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "pass 1: rebuild '%s' (%" PetscInt_FMT " roots)\n", names[v], nr[v]));
    PetscCall(PCPFLAREINVSetPolyCoeffs(pc, var[v], nr[v], 2));
    PetscCall(PetscObjectStateIncrease((PetscObject)A));
    PetscCall(PCSetUp(pc));
    PetscCall(PCApply(pc, b, y));
    exact = (PetscBool)(matrix_free || nr[v] - 1 <= PetscMin(sp, po) || (dense && sp > 0));
    if (exact) {
      PetscCall(ResidualPoly(A, var[v], nr[v], b, r));
      /* e = A y - (b - r) */
      PetscCall(MatMult(A, y, Ay));
      PetscCall(VecAXPY(Ay, -1.0, b));
      PetscCall(VecAXPY(Ay, 1.0, r));
      PetscCall(VecNorm(Ay, NORM_2, &norm_e));
      max_exact = PetscMax(max_exact, norm_e / norm_b);
      PetscCheck(norm_e <= tol * norm_b, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Variant '%s' (%" PetscInt_FMT " roots) is not the polynomial: ||A p(A) b - (b - r)|| / ||b|| = %g", names[v], nr[v], (double)(norm_e / norm_b));
    }
    PetscCall(VecDuplicate(y, &y_ref[v]));
    PetscCall(VecCopy(y, y_ref[v]));
    inv_ref[v] = NULL;
    if (!matrix_free) {
      PetscCall(PCPFLAREINVGetInverseMat(pc, &inv_mat));
      PetscCall(MatDuplicate(inv_mat, MAT_COPY_VALUES, &inv_ref[v]));
      PetscCall(PetscObjectTypeCompare((PetscObject)inv_mat, MATDIAGONAL, &is_diag));
      if (!is_diag) {
        PetscCall(MatGetInfo(inv_mat, MAT_GLOBAL_SUM, &info));
        PetscCheck(info.nz_used == nz_power, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Variant '%s' (%" PetscInt_FMT " roots) gave the inverse %g nonzeros rather than %g", names[v], nr[v], (double)info.nz_used, (double)nz_power);
      }
    }
  }

  /* Zero roots are skipped so don't change the inverse, but extra copies of roots must */
  PetscCall(VecNorm(y_ref[0], NORM_2, &norm_b));
  PetscCall(VecWAXPY(r, -1.0, y_ref[0], y_ref[1]));
  PetscCall(VecNorm(r, NORM_2, &norm_e));
  PetscCheck(norm_e <= tol * norm_b, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Appending zero roots changed the inverse, relative %g", (double)(norm_e / norm_b));
  PetscCall(VecWAXPY(r, -1.0, y_ref[0], y_ref[2]));
  PetscCall(VecNorm(r, NORM_2, &norm_e));
  PetscCheck(norm_e > tol * norm_b, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Appending extra copies of roots didn't change the inverse, relative %g", (double)(norm_e / norm_b));

  /* ~~~~~~~~ Pass 2: every ordered pair (w then v), check rebuilding v gives the same as in pass 1 ~~~~~~~~ */
  for (w = 0; w < nv; w++) {
    for (v = 0; v < nv; v++) {
      if (v == w) continue;
      if (verbose) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "pass 2: '%s' then '%s'\n", names[w], names[v]));
      PetscCall(PCPFLAREINVSetPolyCoeffs(pc, var[w], nr[w], 2));
      PetscCall(PetscObjectStateIncrease((PetscObject)A));
      PetscCall(PCSetUp(pc));
      PetscCall(PCPFLAREINVSetPolyCoeffs(pc, var[v], nr[v], 2));
      PetscCall(PetscObjectStateIncrease((PetscObject)A));
      PetscCall(PCSetUp(pc));
      PetscCall(PCApply(pc, b, y));
      PetscCall(VecAXPY(y, -1.0, y_ref[v]));
      PetscCall(VecNorm(y, NORM_2, &norm_e));
      PetscCall(VecNorm(y_ref[v], NORM_2, &norm_b));
      max_hist = PetscMax(max_hist, norm_e / norm_b);
      PetscCheck(norm_e <= tol * norm_b, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Rebuilding '%s' after '%s' differs from rebuilding it before, relative %g", names[v], names[w], (double)(norm_e / norm_b));
      if (!matrix_free) {
        PetscReal nrm_d, nrm_r;
        Mat       diff;
        PetscCall(PCPFLAREINVGetInverseMat(pc, &inv_mat));
        PetscCall(MatDuplicate(inv_mat, MAT_COPY_VALUES, &diff));
        PetscCall(MatAXPY(diff, -1.0, inv_ref[v], DIFFERENT_NONZERO_PATTERN));
        PetscCall(MatNorm(diff, NORM_FROBENIUS, &nrm_d));
        PetscCall(MatNorm(inv_ref[v], NORM_FROBENIUS, &nrm_r));
        PetscCall(MatDestroy(&diff));
        max_hist = PetscMax(max_hist, nrm_d / nrm_r);
        PetscCheck(nrm_d <= tol * nrm_r, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Assembled inverse from '%s' after '%s' differs, relative %g", names[v], names[w], (double)(nrm_d / nrm_r));
      }
    }
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "OK: order %" PetscInt_FMT " sparsity %" PetscInt_FMT " %s, %" PetscInt_FMT " computed roots, %" PetscInt_FMT " variants (rows", po, sp, matrix_free ? "matrix-free" : "assembled", n0, nv));
  for (v = 0; v < nv; v++) PetscCall(PetscPrintf(PETSC_COMM_WORLD, " %" PetscInt_FMT, nr[v]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "), max exactness error %.2g, max history difference %.2g\n", (double)max_exact, (double)max_hist));

  for (v = 0; v < nv; v++) {
    PetscCall(PetscFree(var[v]));
    PetscCall(VecDestroy(&y_ref[v]));
    PetscCall(MatDestroy(&inv_ref[v]));
  }
  PetscCall(PetscFree(roots0));
  PetscCall(PCDestroy(&pc));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&Ay));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}
