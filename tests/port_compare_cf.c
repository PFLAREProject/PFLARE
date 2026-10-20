static char help[] = "Compares the CF splittings against their Fortran reference implementations.\n\n";

/*
  Differential test used while the Fortran core of PFLARE is ported to C.
  Built only with make PFLARE_PORT_DEBUG=1, which also compiles the Fortran
  reference implementations in src/fref. See section 6 of
  docs/dev/fortran_removal_plan.md.

  For every CFSplittingType and both values of skip_symmetrize (plus a few
  variants of the PMISR DDC options), this computes the CF splitting with the
  public compute_cf_splitting, which calls the C port, and with the Fortran
  reference through compute_cf_splitting_c_fref (C_Fortran_Bindings.F90), and
  checks the fine and coarse ISs are the same. It then compares the matrix from
  compute_diag_dom_submatrix against compute_diag_dom_submatrix_c_fref row by row
  (global columns and bitwise values). Any difference is an error.

  The random numbers both use come from the same seeded generator, so the
  splittings must be identical.

  The matrix is read from a PETSc binary file with -f (as tests/ex6_cf_splitting.c
  does), or is otherwise a steady upwinded advection-diffusion operator on a 2D
  structured grid (as tests/adv_diff_fd.c builds, without its scaling options):
  -da_grid_x, -da_grid_y set the grid (default 30 x 30), -theta the direction of
  the velocity (default pi/4) and -alpha the diffusion coefficient (default 0,
  pure advection). The left and bottom faces are dirichlet rows.
  -mat_type / -dm_mat_type are honoured, so with aijkokkos both the C and the
  Fortran run their Kokkos paths (and their CPU/Kokkos compares with
  PFLARE_KOKKOS_DEBUG=1).
  With -empty_last_rank the rows of the last rank are dropped (the comparisons
  run on the submatrix of the other ranks' rows), so that rank has no rows.
*/
#include <petscksp.h>
#include <petscdmda.h>
#include "pflare.h"

// The C bindings of the Fortran reference implementations in C_Fortran_Bindings.F90
// (only built with PFLARE_PORT_DEBUG=1)
PETSC_EXTERN void compute_cf_splitting_c_fref(Mat *input_mat, int skip_symmetrize_int,
   PetscReal strong_threshold, int max_luby_steps, int cf_splitting_type,
   int ddc_its, PetscReal fraction_swap,
   IS *is_fine, IS *is_coarse);
PETSC_EXTERN void compute_diag_dom_submatrix_c_fref(Mat *input_mat, PetscReal max_dd_ratio, Mat *output_mat);

static const char *const cf_names[] = {"CF_PMISR_DDC", "CF_DIAG_DOM", "CF_PMIS", "CF_PMIS_DIST2", "CF_AGG", "CF_PMIS_AGG", "CF_CR"};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Builds the upwinded advection-diffusion operator on the DMDA
static PetscErrorCode ComputeMat(DM da, PetscReal theta, PetscReal alpha, Mat A)
{
  PetscInt    i, j, mx, my, xm, ym, xs, ys, n;
  PetscReal   u, v;
  PetscScalar vals[5];
  MatStencil  row, cols[5];

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da, NULL, &mx, &my, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));
  u = PetscCosReal(theta);
  v = PetscSinReal(theta);

  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      row.i = i;
      row.j = j;
      n     = 0;
      // Dirichlet on the inflow (left and bottom) faces
      if (i == 0 || j == 0) {
        cols[n].i = i; cols[n].j = j; vals[n++] = 1.0;
      } else {
        // In increasing global column order (south, west, centre, east, north), as
        // tests/adv_diff_fd.c sets them
        cols[n].i = i; cols[n].j = j - 1; vals[n++] = -v - alpha;
        cols[n].i = i - 1; cols[n].j = j; vals[n++] = -u - alpha;
        cols[n].i = i; cols[n].j = j; vals[n++] = u + v + 4.0 * alpha;
        // Outflow on the right and top faces, no neighbour there
        if (alpha != 0.0 && i < mx - 1) {
          cols[n].i = i + 1; cols[n].j = j; vals[n++] = -alpha;
        }
        if (alpha != 0.0 && j < my - 1) {
          cols[n].i = i; cols[n].j = j + 1; vals[n++] = -alpha;
        }
      }
      PetscCall(MatSetValuesStencil(A, 1, &row, n, cols, vals, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Compares the CF splitting from the C port and the Fortran reference
static PetscErrorCode CompareSplitting(Mat A, int skip_symmetrize, PetscReal strong_threshold, int max_luby_steps, int cf_splitting_type, int ddc_its, PetscReal ddc_fraction, const char *label, PetscInt *n_compared)
{
  IS        is_fine, is_coarse, is_fine_fref, is_coarse_fref;
  PetscBool fine_equal, coarse_equal, fine_equal_order, coarse_equal_order;
  PetscInt  n_fine, n_coarse;
  MPI_Comm  comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));

  compute_cf_splitting(A, skip_symmetrize, strong_threshold, max_luby_steps, cf_splitting_type, ddc_its, ddc_fraction, &is_fine, &is_coarse);
  compute_cf_splitting_c_fref(&A, skip_symmetrize, strong_threshold, max_luby_steps, cf_splitting_type, ddc_its, ddc_fraction, &is_fine_fref, &is_coarse_fref);

  PetscCall(ISEqual(is_fine, is_fine_fref, &fine_equal));
  PetscCall(ISEqual(is_coarse, is_coarse_fref, &coarse_equal));
  // and in the same order
  PetscCall(ISEqualUnsorted(is_fine, is_fine_fref, &fine_equal_order));
  PetscCall(ISEqualUnsorted(is_coarse, is_coarse_fref, &coarse_equal_order));
  PetscCall(ISGetSize(is_fine, &n_fine));
  PetscCall(ISGetSize(is_coarse, &n_coarse));

  PetscCall(PetscPrintf(comm, "%-14s %-28s skip_symmetrize %d: F %6" PetscInt_FMT " C %6" PetscInt_FMT " : %s\n", cf_names[cf_splitting_type], label, skip_symmetrize, n_fine, n_coarse,
                        fine_equal && coarse_equal && fine_equal_order && coarse_equal_order ? "match" : "MISMATCH"));
  PetscCheck(fine_equal && coarse_equal && fine_equal_order && coarse_equal_order, comm, PETSC_ERR_PLIB, "C and Fortran reference CF splittings differ for %s %s skip_symmetrize %d", cf_names[cf_splitting_type], label, skip_symmetrize);

  PetscCall(ISDestroy(&is_fine));
  PetscCall(ISDestroy(&is_coarse));
  PetscCall(ISDestroy(&is_fine_fref));
  PetscCall(ISDestroy(&is_coarse_fref));
  (*n_compared)++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Compares two matrices row by row: the same sizes and, in every row, the same global
// column indices (explicit zeros included) and bitwise the same values
static PetscErrorCode CompareMats(Mat A, Mat B, PetscBool *equal)
{
  PetscInt           m, n, M, N, mb, nb, Mb, Nb, rstart, rend, row, ncols, ncols_b, k;
  const PetscInt    *cols, *cols_b;
  const PetscScalar *vals, *vals_b;
  PetscBool          local_equal;
  MPI_Comm           comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(MatGetLocalSize(B, &mb, &nb));
  PetscCall(MatGetSize(B, &Mb, &Nb));
  local_equal = (m == mb && n == nb && M == Mb && N == Nb) ? PETSC_TRUE : PETSC_FALSE;

  if (local_equal) {
    PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
    for (row = rstart; row < rend && local_equal; row++) {
      PetscCall(MatGetRow(A, row, &ncols, &cols, &vals));
      PetscCall(MatGetRow(B, row, &ncols_b, &cols_b, &vals_b));
      if (ncols != ncols_b) local_equal = PETSC_FALSE;
      for (k = 0; k < ncols && local_equal; k++) {
        if (cols[k] != cols_b[k] || vals[k] != vals_b[k]) local_equal = PETSC_FALSE;
      }
      PetscCall(MatRestoreRow(B, row, &ncols_b, &cols_b, &vals_b));
      PetscCall(MatRestoreRow(A, row, &ncols, &cols, &vals));
    }
  }
  PetscCallMPI(MPI_Allreduce(&local_equal, equal, 1, MPI_C_BOOL, MPI_LAND, comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Compares the diagonally dominant submatrix from the C port and the Fortran reference
static PetscErrorCode CompareDiagDomSubmatrix(Mat A, PetscReal max_dd_ratio, PetscInt *n_compared)
{
  Mat       A_dd, A_dd_fref;
  PetscBool equal;
  PetscInt  M, N;
  MPI_Comm  comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));

  compute_diag_dom_submatrix(A, max_dd_ratio, &A_dd);
  compute_diag_dom_submatrix_c_fref(&A, max_dd_ratio, &A_dd_fref);

  PetscCall(CompareMats(A_dd, A_dd_fref, &equal));
  PetscCall(MatGetSize(A_dd, &M, &N));
  PetscCall(PetscPrintf(comm, "compute_diag_dom_submatrix max_dd_ratio %g: %" PetscInt_FMT " x %" PetscInt_FMT " : %s\n", (double)max_dd_ratio, M, N, equal ? "match" : "MISMATCH"));
  PetscCheck(equal, comm, PETSC_ERR_PLIB, "C and Fortran reference diagonally dominant submatrices differ for max_dd_ratio %g", (double)max_dd_ratio);

  PetscCall(MatDestroy(&A_dd));
  PetscCall(MatDestroy(&A_dd_fref));
  (*n_compared)++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char **argv)
{
  Mat         A, A_sub;
  IS          rows;
  DM          da = NULL;
  PetscMPIInt rank, size;
  PetscInt    rstart, rend;
  PetscBool   empty_last_rank = PETSC_FALSE;
  PetscViewer fd;
  char        file[PETSC_MAX_PATH_LEN];
  PetscBool   flg;
  PetscReal   theta, alpha, strong_threshold, cr_rate, ddc_fraction;
  PetscInt    max_luby_steps_in, ddc_its_in, n_compared = 0;
  int         cf_splitting_type, skip_symmetrize, max_luby_steps, ddc_its;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  // The Fortran reference is called from this C main, so the Fortran PETSc
  // constants it uses have to be set up
  PetscCall(PetscInitializeFortran());

  // ~~~~~~~~~~~~
  // The matrix
  // ~~~~~~~~~~~~
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", file, sizeof(file), &flg));
  if (flg) {
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatLoad(A, fd));
    PetscCall(PetscViewerDestroy(&fd));
  } else {
    theta = PETSC_PI / 4.0;
    alpha = 0.0;
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-theta", &theta, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-alpha", &alpha, NULL));
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 30, 30, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
    PetscCall(DMSetFromOptions(da));
    // Only the stencil entries we set are in the sparsity
    PetscCall(DMSetMatrixPreallocateOnly(da, PETSC_TRUE));
    PetscCall(DMSetUp(da));
    PetscCall(DMCreateMatrix(da, &A));
    PetscCall(ComputeMat(da, theta, alpha, A));
  }

  // Drop the rows of the last rank, so it has none
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-empty_last_rank", &empty_last_rank, NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  if (empty_last_rank && size > 1) {
    PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
    if (rank == size - 1) rend = rstart;
    PetscCall(ISCreateStride(PETSC_COMM_WORLD, rend - rstart, rstart, 1, &rows));
    PetscCall(MatCreateSubMatrix(A, rows, rows, MAT_INITIAL_MATRIX, &A_sub));
    PetscCall(ISDestroy(&rows));
    PetscCall(MatDestroy(&A));
    A = A_sub;
  }

  // ~~~~~~~~~~~~
  // The CF splitting options
  // ~~~~~~~~~~~~
  // Threshold for a strong connection (and the target ratio for CF_DIAG_DOM)
  strong_threshold = 0.5;
  // The target CR rate for CF_CR
  cr_rate = 0.1;
  // Fraction of F points to convert to C per ddc it
  ddc_fraction = 0.1;
  // As many steps as needed
  max_luby_steps_in = -1;
  // Second pass cleanup - one iteration
  ddc_its_in = 1;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-strong_threshold", &strong_threshold, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-cr_rate", &cr_rate, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-ddc_fraction", &ddc_fraction, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-max_luby_steps", &max_luby_steps_in, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ddc_its", &ddc_its_in, NULL));
  max_luby_steps = (int)max_luby_steps_in;
  ddc_its        = (int)ddc_its_in;

  // ~~~~~~~~~~~~
  // Every CF splitting type, with and without symmetrizing the strength matrix
  // ~~~~~~~~~~~~
  for (cf_splitting_type = CF_PMISR_DDC; cf_splitting_type <= CF_CR; cf_splitting_type++) {
    for (skip_symmetrize = 0; skip_symmetrize <= 1; skip_symmetrize++) {
      PetscCall(CompareSplitting(A, skip_symmetrize, cf_splitting_type == CF_CR ? cr_rate : strong_threshold, max_luby_steps, cf_splitting_type, ddc_its, ddc_fraction, "", &n_compared));
    }
  }

  // ~~~~~~~~~~~~
  // Variants of the PMISR DDC options
  // ~~~~~~~~~~~~
  for (skip_symmetrize = 0; skip_symmetrize <= 1; skip_symmetrize++) {
    // A fixed number of Luby steps
    PetscCall(CompareSplitting(A, skip_symmetrize, strong_threshold, 2, CF_PMISR_DDC, ddc_its, ddc_fraction, "max_luby_steps 2", &n_compared));
    PetscCall(CompareSplitting(A, skip_symmetrize, strong_threshold, 2, CF_PMIS, ddc_its, ddc_fraction, "max_luby_steps 2", &n_compared));
    // A fixed diagonal dominance threshold rather than a fraction
    PetscCall(CompareSplitting(A, skip_symmetrize, strong_threshold, max_luby_steps, CF_PMISR_DDC, ddc_its, -0.5, "ddc_fraction -0.5", &n_compared));
    // More DDC iterations
    PetscCall(CompareSplitting(A, skip_symmetrize, strong_threshold, max_luby_steps, CF_PMISR_DDC, 3, ddc_fraction, "ddc_its 3", &n_compared));
    // An exact independent set (no DDC)
    PetscCall(CompareSplitting(A, skip_symmetrize, 0.0, max_luby_steps, CF_PMISR_DDC, ddc_its, ddc_fraction, "strong_threshold 0", &n_compared));
    // A different target diagonal dominance ratio
    PetscCall(CompareSplitting(A, skip_symmetrize, 0.25, max_luby_steps, CF_DIAG_DOM, ddc_its, ddc_fraction, "strong_threshold 0.25", &n_compared));
  }

  // ~~~~~~~~~~~~
  // The diagonally dominant submatrix
  // ~~~~~~~~~~~~
  PetscCall(CompareDiagDomSubmatrix(A, 0.5, &n_compared));
  PetscCall(CompareDiagDomSubmatrix(A, 0.25, &n_compared));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "port compare CF: all %" PetscInt_FMT " comparisons match\n", n_compared));

  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}
