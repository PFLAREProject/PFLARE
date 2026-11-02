/*
  Capability study for the PCAIR diagonal update (-pc_air_diag_update).

  Solves a sequence of systems (L + s_k W) x = b, k = 1..K, where L is a fixed upwind
  finite difference advection(-diffusion) operator on a 2D or 3D DMDA grid (grid units),
  W is a fixed positive diagonal ("material" weight) and s_k >= 0 a list of shifts.

  This covers several applications with one fixed transport operator and a changing diagonal:
    multigroup transport          s_k = group removal scale, W = heterogeneous cross section
    multi-species transport       s_k = R_i/dt + lambda_i (retardation, decay), W = lumped mass
    implicit time stepping / IRK  s_k = gamma_j/dt_n

  For every s_k we time:
    rebuild : PCAIR built from L + s_k W (KSPSetUp) then KSPSolve
    update  : PCAIR built once from L + s_build W with PCAIRSetDiagUpdate, then the diagonal
              of the same Pmat is changed in place and KSPSolve updates the hierarchy
    frozen  : (-frozen) PCAIR built once from L + s_build W and never touched
  and, with -bands f, the update restores stored polynomial coefficients for bands of 1/s
  (the dt bands of tests/dt_bands.h).

  Options
    -dim 2|3, -n (grid points per dim)
    -vel const|curved|recirc     velocity field
    -alpha a                     diffusion (grid units)
    -contrast c                  W is a checkerboard of blocks with values 1 and c (default 1)
    -nblocks m                   number of checkerboard blocks per dimension (default 4)
    -shifts s1,s2,...            the sequence of shifts (up to 4096)
    -s_build s                   the shift the update hierarchy is built at (default min s_k)
    -rebuild 0|1, -frozen 0|1, -bands f
*/
static char help[] = "Capability study of the PCAIR diagonal update.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petsctime.h>
#include "pflare.h"
#include "../../tests/dt_bands.h"

typedef enum { VEL_CONST, VEL_CURVED, VEL_RECIRC } VelType;

static void Velocity(VelType vt, PetscInt dim, PetscReal x, PetscReal y, PetscReal z, PetscReal *u)
{
  (void)z;
  if (vt == VEL_CONST) {
    if (dim == 2) { u[0] = PetscCosReal(PETSC_PI / 4); u[1] = PetscSinReal(PETSC_PI / 4); u[2] = 0; }
    else { u[0] = 1.0 / PetscSqrtReal(3.0); u[1] = u[0]; u[2] = u[0]; }
  } else if (vt == VEL_CURVED) {
    // bends from +x to +y across the domain, no closed streamlines
    PetscReal th = 0.5 * PETSC_PI * y;
    u[0] = PetscCosReal(th); u[1] = PetscSinReal(th); u[2] = (dim == 3) ? 0.3 : 0.0;
  } else {
    // "double glazing" recirculating flow on [-1,1]^d, closed streamlines
    PetscReal X = 2 * x - 1, Y = 2 * y - 1;
    u[0] = 2 * Y * (1 - X * X); u[1] = -2 * X * (1 - Y * Y); u[2] = (dim == 3) ? 0.2 : 0.0;
  }
}

static PetscErrorCode ComputeL(DM da, Mat A, VelType vt, PetscReal alpha)
{
  PetscInt    dim, mx, my, mz, xs, ys, zs, xm, ym, zm, i, j, k, nc, d;
  PetscScalar vals[7];
  MatStencil  row, cols[7];
  PetscReal   u[3];

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da, &dim, &mx, &my, &mz, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
  for (k = zs; k < zs + zm; k++) {
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        row.i = i; row.j = j; row.k = k; row.c = 0;
        PetscBool bnd = (i == 0 || j == 0 || i == mx - 1 || j == my - 1);
        if (dim == 3) bnd = (PetscBool)(bnd || k == 0 || k == mz - 1);
        if (bnd) {
          cols[0] = row; vals[0] = 1.0;
          PetscCall(MatSetValuesStencil(A, 1, &row, 1, cols, vals, INSERT_VALUES));
          continue;
        }
        Velocity(vt, dim, (PetscReal)i / (mx - 1), (PetscReal)j / (my - 1), dim == 3 ? (PetscReal)k / (mz - 1) : 0, u);
        nc = 0;
        cols[nc] = row; vals[nc] = 2 * dim * alpha; nc++;
        for (d = 0; d < dim; d++) {
          MatStencil lo = row, hi = row;
          if (d == 0) { lo.i--; hi.i++; }
          if (d == 1) { lo.j--; hi.j++; }
          if (d == 2) { lo.k--; hi.k++; }
          vals[0] += PetscAbsReal(u[d]);
          // upwind neighbour gets -|u|, both get -alpha
          cols[nc] = lo; vals[nc] = -alpha - (u[d] > 0 ? u[d] : 0); nc++;
          cols[nc] = hi; vals[nc] = -alpha - (u[d] < 0 ? -u[d] : 0); nc++;
        }
        PetscCall(MatSetValuesStencil(A, 1, &row, nc, cols, vals, INSERT_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Checkerboard material weight, zero on Dirichlet rows
static PetscErrorCode ComputeW(DM da, Vec w, PetscReal contrast, PetscInt nblocks)
{
  PetscInt     dim, mx, my, mz, xs, ys, zs, xm, ym, zm, i, j, k;
  PetscScalar ***w3, **w2;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da, &dim, &mx, &my, &mz, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
  if (dim == 2) PetscCall(DMDAVecGetArray(da, w, &w2));
  else PetscCall(DMDAVecGetArray(da, w, &w3));
  for (k = zs; k < zs + zm; k++) {
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        PetscBool bnd = (i == 0 || j == 0 || i == mx - 1 || j == my - 1);
        if (dim == 3) bnd = (PetscBool)(bnd || k == 0 || k == mz - 1);
        PetscInt  bi = (i * nblocks) / mx, bj = (j * nblocks) / my, bk = dim == 3 ? (k * nblocks) / mz : 0;
        PetscReal val = bnd ? 0.0 : (((bi + bj + bk) % 2) ? contrast : 1.0);
        if (dim == 2) w2[j][i] = val;
        else w3[k][j][i] = val;
      }
    }
  }
  if (dim == 2) PetscCall(DMDAVecRestoreArray(da, w, &w2));
  else PetscCall(DMDAVecRestoreArray(da, w, &w3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// A = L + s W
static PetscErrorCode SetShift(Mat L, Vec w, Vec tmp, PetscReal s, Mat A)
{
  PetscFunctionBeginUser;
  PetscCall(MatCopy(L, A, SAME_NONZERO_PATTERN));
  PetscCall(VecCopy(w, tmp));
  PetscCall(VecScale(tmp, s));
  PetscCall(MatDiagonalSet(A, tmp, ADD_VALUES));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TimedSolve(KSP ksp, Vec b, Vec x, PetscInt *its, double *t_setup, double *t_solve, PetscBool *ok)
{
  PetscLogDouble     t0, t1, t2;
  KSPConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(VecSet(x, 0.0));
  PetscCall(PetscBarrier(NULL));
  PetscCall(PetscTime(&t0));
  PetscCall(KSPSetUp(ksp));
  PetscCall(PetscBarrier(NULL));
  PetscCall(PetscTime(&t1));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(PetscBarrier(NULL));
  PetscCall(PetscTime(&t2));
  PetscCall(KSPGetIterationNumber(ksp, its));
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  *ok      = (PetscBool)(reason > 0);
  *t_setup = t1 - t0;
  *t_solve = t2 - t1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        da;
  Mat       L, A_up, A_re;
  Vec       x, b, w, tmp;
  KSP       ksp_up, ksp_fr = NULL;
  PC        pc;
  PetscInt  dim = 2, n = 100, nblocks = 4, n_s = 4096, s, its;
  PetscReal alpha = 0.0, contrast = 1.0, s_build = 0, bands_f = 0.0, *shifts;
  PetscBool do_rebuild = PETSC_TRUE, do_frozen = PETSC_FALSE, flg, sb_set, ok;
  char      vel[32] = "const";
  VelType   vt = VEL_CONST;
  double    ts, tv;
  double    tot[3][2] = {{0}}; // rebuild/update/frozen x setup/solve
  PetscInt  tot_its[3] = {0}, fails[3] = {0};
  DtBands   bands;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PCRegister_PFLARE();
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nblocks", &nblocks, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-alpha", &alpha, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-contrast", &contrast, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-bands", &bands_f, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-rebuild", &do_rebuild, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-frozen", &do_frozen, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-vel", vel, sizeof(vel), NULL));
  if (!strcmp(vel, "curved")) vt = VEL_CURVED;
  else if (!strcmp(vel, "recirc")) vt = VEL_RECIRC;
  PetscCall(PetscMalloc1(n_s, &shifts));
  PetscCall(PetscOptionsGetRealArray(NULL, NULL, "-shifts", shifts, &n_s, &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER, "Need -shifts");
  s_build = shifts[0];
  for (s = 1; s < n_s; s++) s_build = PetscMin(s_build, shifts[s]);
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-s_build", &s_build, &sb_set));

  if (dim == 2) PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, n, n, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  else PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, n, n, n, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMCreateMatrix(da, &L));
  PetscCall(ComputeL(da, L, vt, alpha));
  PetscCall(DMCreateGlobalVector(da, &x));
  PetscCall(VecDuplicate(x, &b));
  PetscCall(VecDuplicate(x, &w));
  PetscCall(VecDuplicate(x, &tmp));
  PetscCall(VecSet(b, 1.0));
  PetscCall(ComputeW(da, w, contrast, nblocks));
  PetscCall(MatDuplicate(L, MAT_COPY_VALUES, &A_up));
  PetscCall(MatDuplicate(L, MAT_COPY_VALUES, &A_re));

  // Update KSP, built at s_build
  PetscCall(SetShift(L, w, tmp, s_build, A_up));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_up));
  PetscCall(KSPSetOperators(ksp_up, A_up, A_up));
  PetscCall(KSPGetPC(ksp_up, &pc));
  PetscCall(PCSetType(pc, PCAIR));
  PetscCall(KSPSetFromOptions(ksp_up));
  PetscCall(PCAIRSetDiagUpdate(pc, PETSC_TRUE));
  PetscCall(DtBandsCreate(&bands, bands_f));
  PetscCall(DtBandsSetKSP(&bands, ksp_up));
  PetscCall(VecSet(x, 0.0));
  {
    PetscLogDouble t0, t1;
    PetscCall(PetscTime(&t0));
    PetscCall(KSPSetUp(ksp_up));
    PetscCall(PetscTime(&t1));
    tot[1][0] += t1 - t0;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "initial build at s=%g: %.3f s\n", (double)s_build, t1 - t0));
  }
  if (do_frozen) {
    Mat A_fr;
    PetscCall(MatDuplicate(A_up, MAT_COPY_VALUES, &A_fr));
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_fr));
    PetscCall(KSPSetOperators(ksp_fr, A_fr, A_fr));
    PetscCall(KSPGetPC(ksp_fr, &pc));
    PetscCall(PCSetType(pc, PCAIR));
    PetscCall(KSPSetFromOptions(ksp_fr));
    PetscCall(KSPSetUp(ksp_fr));
    PetscCall(MatDestroy(&A_fr));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%4s %12s | %5s %8s %8s | %5s %8s %8s%s\n", "k", "shift", "re_it", "re_setup", "re_solve", "up_it", "up_setup", "up_solve", do_frozen ? " | fr_it" : ""));
  for (s = 0; s < n_s; s++) {
    PetscInt its_re = -1, its_up, its_fr = -1;
    double   rs = 0, rv = 0;
    // rebuild
    if (do_rebuild) {
      KSP ksp_re;
      PC  pc_re;
      PetscCall(SetShift(L, w, tmp, shifts[s], A_re));
      PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_re));
      PetscCall(KSPSetOperators(ksp_re, A_re, A_re));
      PetscCall(KSPGetPC(ksp_re, &pc_re));
      PetscCall(PCSetType(pc_re, PCAIR));
      PetscCall(KSPSetFromOptions(ksp_re));
      PetscCall(TimedSolve(ksp_re, b, x, &its_re, &rs, &rv, &ok));
      if (!ok) fails[0]++;
      tot[0][0] += rs; tot[0][1] += rv; tot_its[0] += its_re;
      PetscCall(KSPDestroy(&ksp_re));
    }
    // update in place
    PetscCall(SetShift(L, w, tmp, shifts[s], A_up));
    if (bands_f > 1.0 && shifts[s] > 0) PetscCall(DtBandsPrepare(&bands, 1.0 / shifts[s]));
    PetscCall(TimedSolve(ksp_up, b, x, &its_up, &ts, &tv, &ok));
    if (!ok) fails[1]++;
    tot[1][0] += ts; tot[1][1] += tv; tot_its[1] += its_up;
    if (do_frozen) {
      Mat A_fr;
      double fs, fv;
      PetscCall(KSPGetOperators(ksp_fr, &A_fr, NULL));
      // change the Amat only, so the PC is not rebuilt
      PetscCall(KSPSetReusePreconditioner(ksp_fr, PETSC_TRUE));
      PetscCall(SetShift(L, w, tmp, shifts[s], A_fr));
      PetscCall(TimedSolve(ksp_fr, b, x, &its_fr, &fs, &fv, &ok));
      if (!ok) fails[2]++;
      tot[2][1] += fv; tot_its[2] += its_fr;
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%4d %12.4e | %5d %8.4f %8.4f | %5d %8.4f %8.4f", (int)s, (double)shifts[s], (int)its_re, rs, rv, (int)its_up, ts, tv));
    if (do_frozen) PetscCall(PetscPrintf(PETSC_COMM_WORLD, " | %5d", (int)its_fr));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "SUMMARY nsys %d | rebuild its/solve %.2f setup %.3f solve %.3f total %.3f fails %d | update its/solve %.2f setup %.3f solve %.3f total %.3f fails %d | frozen its/solve %.2f solve %.3f fails %d | bands stored %d restored %d\n",
                        (int)n_s, (double)tot_its[0] / n_s, tot[0][0], tot[0][1], tot[0][0] + tot[0][1], (int)fails[0], (double)tot_its[1] / n_s, tot[1][0], tot[1][1], tot[1][0] + tot[1][1], (int)fails[1], (double)tot_its[2] / n_s, tot[2][1], (int)fails[2], (int)bands.n_stored, (int)bands.n_restored));
  (void)its;
  // Multiple right-hand sides on the last updated operator: one KSPMatSolve against nrhs KSPSolves
  {
    PetscInt nrhs = 0, j;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-nrhs", &nrhs, NULL));
    if (nrhs > 0) {
      Mat            B, X;
      PetscRandom    rnd;
      PetscLogDouble t0, t1, t2;
      PetscInt       nloc, nglob, its_block;
      Vec            col;
      PetscCall(KSPSetType(ksp_up, KSPRICHARDSON));
      PetscCall(KSPSetTolerances(ksp_up, 1e-8, PETSC_CURRENT, PETSC_CURRENT, 300));
      PetscCall(VecGetLocalSize(b, &nloc));
      PetscCall(VecGetSize(b, &nglob));
      PetscCall(MatCreateDense(PETSC_COMM_WORLD, nloc, PETSC_DECIDE, nglob, nrhs, NULL, &B));
      PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &X));
      PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rnd));
      PetscCall(PetscRandomSetFromOptions(rnd));
      PetscCall(MatSetRandom(B, rnd));
      // warm up the block scratch
      PetscCall(KSPMatSolve(ksp_up, B, X));
      PetscCall(PetscTime(&t0));
      PetscCall(KSPMatSolve(ksp_up, B, X));
      PetscCall(PetscTime(&t1));
      PetscCall(KSPGetIterationNumber(ksp_up, &its_block));
      its = 0;
      for (j = 0; j < nrhs; j++) {
        PetscInt its_j;
        PetscCall(MatDenseGetColumnVecRead(B, j, &col));
        PetscCall(VecCopy(col, tmp));
        PetscCall(MatDenseRestoreColumnVecRead(B, j, &col));
        PetscCall(VecSet(x, 0.0));
        PetscCall(KSPSolve(ksp_up, tmp, x));
        PetscCall(KSPGetIterationNumber(ksp_up, &its_j));
        its += its_j;
      }
      PetscCall(PetscTime(&t2));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MRHS nrhs %d | block its %d time %.4f | single total its %d time %.4f | speedup %.2f\n", (int)nrhs, (int)its_block, t1 - t0, (int)its, t2 - t1, (t2 - t1) / (t1 - t0)));
      PetscCall(PetscRandomDestroy(&rnd));
      PetscCall(MatDestroy(&B));
      PetscCall(MatDestroy(&X));
    }
  }
  PetscCall(DtBandsDestroy(&bands));
  PetscCall(KSPDestroy(&ksp_up));
  PetscCall(KSPDestroy(&ksp_fr));
  PetscCall(MatDestroy(&L));
  PetscCall(MatDestroy(&A_up));
  PetscCall(MatDestroy(&A_re));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&w));
  PetscCall(VecDestroy(&tmp));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFree(shifts));
  PetscCall(PetscFinalize());
  return 0;
}
