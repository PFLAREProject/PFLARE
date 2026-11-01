static char help[] = "Rebuilds Newton polynomial inverses from sets of roots of different lengths.\n\
\n\
Sets up PCPFLAREINV with a Newton basis on a 2D upwind advection-diffusion matrix, then\n\
builds variants of the computed roots with more or fewer roots: extra copies of real\n\
roots and complex pairs (as the newton type adds for stability), zero roots at the end\n\
and in the middle, and the roots truncated back to the polynomial order + 1.\n\
For each variant the inverse is rebuilt (reusing the existing inverse matrix, which was\n\
built from a different number of roots) and checked:\n\
  - exactness, when the inverse should be exactly the polynomial (matrix-free, or\n\
    assembled with -pc_pflareinv_sparsity_order >= -pc_pflareinv_poly_order):\n\
        A p(A) b = b - prod_i (I - A/theta_i) b\n\
    with the product computed here with matvecs, skipping zero roots and taking\n\
    complex conjugate pairs together\n\
  - history independence: rebuilding from the same roots gives the same inverse,\n\
    whatever set of roots the reused inverse was built from before\n\
\n\
  -n : grid size (default 10)\n\
  -c : diagonal shift (default 1)\n\n";

#include <petscksp.h>
#include "pflare.h"

#define MAX_VARIANTS 8

static PetscErrorCode BuildMatrix(PetscInt n, PetscReal c, Mat *A)
{
  PetscInt  Istart, Iend, row, i, j;
  PetscReal alpha = 1.0;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(PETSC_COMM_WORLD, A));
  PetscCall(MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, n * n, n * n));
  PetscCall(MatSetFromOptions(*A));
  PetscCall(MatSeqAIJSetPreallocation(*A, 5, NULL));
  PetscCall(MatMPIAIJSetPreallocation(*A, 5, NULL, 5, NULL));
  PetscCall(MatGetOwnershipRange(*A, &Istart, &Iend));
  for (row = Istart; row < Iend; row++) {
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

/* Copies the set src (nsrc rows) into dst, which has room for nmax rows */
static void Put(PetscReal *dst, PetscInt nd, PetscInt i, PetscReal re, PetscReal im)
{
  dst[i]      = re;
  dst[nd + i] = im;
}

int main(int argc, char **args)
{
  Mat              A, inv_mat, inv_ref[MAX_VARIANTS];
  Vec              b, y, r, Ay, y_ref[MAX_VARIANTS];
  PC               pc;
  PetscInt         n = 10, n0, cols0, i, v, w, nv = 0, po, sp, nr[MAX_VARIANTS], first_real = -1, first_complex = -1, k;
  PetscReal        c = 1.0, *roots0, *ptr, *var[MAX_VARIANTS], norm_b, norm_e, max_exact = 0.0, max_hist = 0.0, tol;
  PetscBool        matrix_free, exact, verbose = PETSC_FALSE;
  PCPFLAREINVType  type;
  PetscRandom      rctx;
  const char      *names[MAX_VARIANTS];

  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PCRegister_PFLARE();
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-c", &c, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-verbose", &verbose, NULL));

  PetscCall(BuildMatrix(n, c, &A));
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
  PetscCall(PCSetUp(pc));
  PetscCall(PCPFLAREINVGetType(pc, &type));
  PetscCall(PCPFLAREINVGetMatrixFree(pc, &matrix_free));
  PetscCall(PCPFLAREINVGetPolyOrder(pc, &po));
  PetscCall(PCPFLAREINVGetSparsityOrder(pc, &sp));
  PetscCheck(type == PFLAREINV_NEWTON || type == PFLAREINV_NEWTON_NO_EXTRA, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Needs a Newton type");
  exact = (PetscBool)(matrix_free || sp >= po);

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

  PetscCall(PCPFLAREINVSetReusePolyCoeffs(pc, PETSC_TRUE));
  tol = 1e4 * PETSC_MACHINE_EPSILON;

  /* ~~~~~~~~ Pass 1: rebuild from each variant in turn, check exactness and save the result ~~~~~~~~ */
  for (v = 0; v < nv; v++) {
    if (verbose) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "pass 1: rebuild '%s' (%" PetscInt_FMT " roots)\n", names[v], nr[v]));
    PetscCall(PCPFLAREINVSetPolyCoeffs(pc, var[v], nr[v], 2));
    PetscCall(PetscObjectStateIncrease((PetscObject)A));
    PetscCall(PCSetUp(pc));
    PetscCall(PCApply(pc, b, y));
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
    }
  }

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
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "), max exactness error %.2g%s, max history difference %.2g\n", (double)max_exact, exact ? "" : " (not checked)", (double)max_hist));

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
