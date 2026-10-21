static char help[] = "Compares PCPFLAREINV and the approximate inverses against their Fortran reference implementations.\n\n";

/*
  Differential test used while the Fortran core of PFLARE is ported to C.
  Built only with make PFLARE_PORT_DEBUG=1, which also compiles the Fortran
  reference implementations in src/fref. See section 6 of
  docs/dev/fortran_removal_plan.md.

  1) The PC: for every PCPFLAREINVType, assembled and matrix-free (except SAI/ISAI,
     which can't be matrix-free), polynomial orders 2, 6 and 12 and sparsity orders
     0, 1 and 2 (assembled), this sets up a pflareinv PC, which calls the C port, and
     a pflareinv_fref PC (src/fref/PCPFLAREINV_fref.c), which is PCPFLAREINV as it was
     before the port and calls the Fortran reference, on the same operator and checks
     - the polynomial coefficients from PCPFLAREINVGetPolyCoeffs are bitwise the same
     - an assembled inverse from PCPFLAREINVGetInverseMat is the same row by row
       (global columns and bitwise values), a diagonal one has the same diagonal
     - PCApply and PCApplyTranspose of a random vector and PCMatApply of a random
       block are bitwise the same
     and then does the same after the reuse paths: the operator values change
     (SAME_NONZERO_PATTERN) with PCPFLAREINVSetReusePolyCoeffs, so the stored
     coefficients are reused, and then with coefficients set by
     PCPFLAREINVSetPolyCoeffs.
  2) The diagonal scaling and subcomm, which PCPFLAREINV never turns on: for every
     polynomial type, assembled and matrix-free, with and without diagonal scaling
     and with and without the subcomm, this builds the inverse with
     calculate_and_build_approximate_inverse_c (PCPFLAREINV.c, the C port) and
     calculate_and_build_approximate_inverse_c_fref (src/fref/C_Fortran_Bindings.F90, the
     Fortran reference) and checks the coefficients and the inverse as above, with
     the matrix-free inverses compared through MatMult, MatMultTranspose and the
     block apply (pflareinv_shell_block_matapply_c and its _fref).
  Any difference is an error. The C and the Fortran do the same floating point
  operations in the same order (the random start vector of the GMRES polynomials
  comes from the same seeded generator), so everything is compared bitwise.

  The matrix is read from a PETSc binary file with -f, or is otherwise a steady
  upwinded advection-diffusion operator on a 2D structured grid (as in
  tests/port_compare_cf.c): -da_grid_x, -da_grid_y set the grid (default 30 x 30),
  -theta the direction of the velocity (default pi/4) and -alpha the diffusion
  coefficient (default 0, pure advection). The left and bottom faces are
  dirichlet rows. -mat_type / -dm_mat_type are honoured, so with aijkokkos both
  the C and the Fortran run their Kokkos paths (and their CPU/Kokkos compares
  with PFLARE_KOKKOS_DEBUG=1).
  With -empty_last_rank the rows of the last rank are dropped (the comparisons
  run on the submatrix of the other ranks' rows), so that rank has no rows.
*/
#include <petscksp.h>
#include <petscdmda.h>
#include "pflare.h"

// The C entry points of the C port in PCPFLAREINV.c and of the Fortran reference in
// src/fref/C_Fortran_Bindings.F90 (only built with PFLARE_PORT_DEBUG=1). The int arguments
// are 1 for true
PETSC_EXTERN void calculate_and_build_approximate_inverse_c(Mat *input_mat, int inverse_type, int order,
   int sparsity_order, int matrix_free_int, int diag_scale_polys_int, int subcomm_int,
   PetscReal **coeffs_ptr, PetscInt *row_size, PetscInt *col_size, Mat *inv_matrix);
PETSC_EXTERN void reset_inverse_mat_c(Mat *mat);
PETSC_EXTERN void pflareinv_shell_block_matapply_c(Mat *mat, Mat *X, Mat *Y, int *applied);
PETSC_EXTERN void calculate_and_build_approximate_inverse_c_fref(Mat *input_mat, int inverse_type, int order,
   int sparsity_order, int matrix_free_int, int diag_scale_polys_int, int subcomm_int,
   PetscReal **coeffs_ptr, PetscInt *row_size, PetscInt *col_size, Mat *inv_matrix);
PETSC_EXTERN void reset_inverse_mat_c_fref(Mat *mat);
PETSC_EXTERN void pflareinv_shell_block_matapply_c_fref(Mat *mat, Mat *X, Mat *Y, int *applied);

static const char *const inv_names[] = {"power", "arnoldi", "newton", "newton_no_extra", "neumann", "sai", "isai", "wjacobi", "jacobi", "l1jacobi"};

// The number of right hand sides in the blocks for PCMatApply
#define N_RHS 3

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Builds the upwinded advection-diffusion operator on the DMDA (as tests/port_compare_cf.c)
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
        // In increasing global column order (south, west, centre, east, north)
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

// Errors unless a local condition holds on every rank
static PetscErrorCode CheckAll(MPI_Comm comm, PetscBool local_ok, const char *what, const char *label)
{
  PetscBool ok;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Allreduce(&local_ok, &ok, 1, MPI_C_BOOL, MPI_LAND, comm));
  PetscCheck(ok, comm, PETSC_ERR_PLIB, "C and Fortran reference differ in the %s for %s", what, label);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The coefficients must be the same size and bitwise the same (they are the same on
// every rank)
static PetscErrorCode CompareCoeffs(MPI_Comm comm, const PetscReal *coeffs, PetscInt rows, PetscInt cols, const PetscReal *coeffs_fref, PetscInt rows_fref, PetscInt cols_fref, const char *label)
{
  PetscBool ok;
  PetscInt  i;

  PetscFunctionBeginUser;
  ok = (rows == rows_fref && cols == cols_fref) ? PETSC_TRUE : PETSC_FALSE;
  if (ok && rows * cols > 0) {
    ok = (coeffs && coeffs_fref) ? PETSC_TRUE : PETSC_FALSE;
    for (i = 0; i < rows * cols && ok; i++) {
      if (coeffs[i] != coeffs_fref[i]) ok = PETSC_FALSE;
    }
  }
  PetscCall(CheckAll(comm, ok, "coefficients", label));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Compares two assembled matrices: the same type and sizes and a MATDIAGONAL the same
// diagonal, anything else, in every row, the same global column indices (explicit
// zeros included) and bitwise the same values
static PetscErrorCode CompareAssembled(Mat A, Mat B, const char *label)
{
  PetscInt           m, n, M, N, mb, nb, Mb, Nb, rstart, rend, row, ncols, ncols_b, k;
  const PetscInt    *cols, *cols_b;
  const PetscScalar *vals, *vals_b;
  PetscBool          local_equal, is_diag, is_diag_b, vec_equal;
  MPI_Comm           comm;
  Vec                d, d_b;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATDIAGONAL, &is_diag));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATDIAGONAL, &is_diag_b));
  PetscCheck(is_diag == is_diag_b, comm, PETSC_ERR_PLIB, "C and Fortran reference differ in the inverse type for %s", label);
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(MatGetLocalSize(B, &mb, &nb));
  PetscCall(MatGetSize(B, &Mb, &Nb));
  local_equal = (m == mb && n == nb && M == Mb && N == Nb) ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(CheckAll(comm, local_equal, "inverse sizes", label));

  if (is_diag) {
    PetscCall(MatCreateVecs(A, NULL, &d));
    PetscCall(VecDuplicate(d, &d_b));
    PetscCall(MatGetDiagonal(A, d));
    PetscCall(MatGetDiagonal(B, d_b));
    PetscCall(VecEqual(d, d_b, &vec_equal));
    PetscCheck(vec_equal, comm, PETSC_ERR_PLIB, "C and Fortran reference differ in the diagonal inverse for %s", label);
    PetscCall(VecDestroy(&d));
    PetscCall(VecDestroy(&d_b));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

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
  PetscCall(CheckAll(comm, local_equal, "assembled inverse", label));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Bitwise comparison of two vectors
static PetscErrorCode CompareVecs(Vec x, Vec y, const char *what, const char *label)
{
  PetscBool equal;
  MPI_Comm  comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)x, &comm));
  PetscCall(VecEqual(x, y, &equal));
  PetscCheck(equal, comm, PETSC_ERR_PLIB, "C and Fortran reference differ in the %s for %s", what, label);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Bitwise comparison of two dense blocks with the same layout
static PetscErrorCode CompareDense(Mat X, Mat Y, const char *what, const char *label)
{
  const PetscScalar *x, *y;
  PetscInt           m, n, ldx, ldy, i, j;
  PetscBool          local_equal = PETSC_TRUE;
  MPI_Comm           comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)X, &comm));
  PetscCall(MatGetLocalSize(X, &m, NULL));
  PetscCall(MatGetSize(X, NULL, &n));
  PetscCall(MatDenseGetLDA(X, &ldx));
  PetscCall(MatDenseGetLDA(Y, &ldy));
  PetscCall(MatDenseGetArrayRead(X, &x));
  PetscCall(MatDenseGetArrayRead(Y, &y));
  for (j = 0; j < n && local_equal; j++) {
    for (i = 0; i < m && local_equal; i++) {
      if (x[i + j * ldx] != y[i + j * ldy]) local_equal = PETSC_FALSE;
    }
  }
  PetscCall(MatDenseRestoreArrayRead(Y, &y));
  PetscCall(MatDenseRestoreArrayRead(X, &x));
  PetscCall(CheckAll(comm, local_equal, what, label));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// The random vector and block the applies are compared on, from a seeded generator
static PetscErrorCode CreateRandom(Mat A, Vec *x, Mat *B)
{
  PetscRandom rctx;
  VecType     vtype;
  PetscInt    m, M;
  MPI_Comm    comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCall(PetscRandomCreate(comm, &rctx));
  PetscCall(PetscRandomSetSeed(rctx, 0x5eed));
  PetscCall(PetscRandomSeed(rctx));
  PetscCall(MatCreateVecs(A, x, NULL));
  PetscCall(VecSetRandom(*x, rctx));
  PetscCall(MatGetVecType(A, &vtype));
  PetscCall(MatGetLocalSize(A, &m, NULL));
  PetscCall(MatGetSize(A, &M, NULL));
  PetscCall(MatCreateDenseFromVecType(comm, vtype, m, PETSC_DECIDE, M, N_RHS, PETSC_DECIDE, NULL, B));
  PetscCall(MatSetRandom(*B, rctx));
  PetscCall(PetscRandomDestroy(&rctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Compares the pflareinv and pflareinv_fref PCs after a setup: the coefficients, an
// assembled inverse and PCApply, PCApplyTranspose and PCMatApply
static PetscErrorCode ComparePCs(PC pc, PC pc_fref, Vec x, Mat B, PetscBool matrix_free, const char *label)
{
  PetscReal *coeffs, *coeffs_fref;
  PetscInt   rows, cols, rows_fref, cols_fref;
  Mat        inv, inv_fref, Y, Y_fref;
  Vec        y, y_fref;
  MPI_Comm   comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)pc, &comm));

  // The coefficients (NULL for the non-polynomial types)
  PetscCall(PCPFLAREINVGetPolyCoeffs(pc, &coeffs, &rows, &cols));
  PetscCall(PCPFLAREINVGetPolyCoeffs(pc_fref, &coeffs_fref, &rows_fref, &cols_fref));
  PetscCall(CompareCoeffs(comm, coeffs, rows, cols, coeffs_fref, rows_fref, cols_fref, label));

  // An assembled inverse (a matrix-free one is compared through the applies)
  PetscCall(PCPFLAREINVGetInverseMat(pc, &inv));
  PetscCall(PCPFLAREINVGetInverseMat(pc_fref, &inv_fref));
  if (!matrix_free) PetscCall(CompareAssembled(inv, inv_fref, label));

  // The applies
  PetscCall(VecDuplicate(x, &y));
  PetscCall(VecDuplicate(x, &y_fref));
  PetscCall(PCApply(pc, x, y));
  PetscCall(PCApply(pc_fref, x, y_fref));
  PetscCall(CompareVecs(y, y_fref, "PCApply", label));
  PetscCall(PCApplyTranspose(pc, x, y));
  PetscCall(PCApplyTranspose(pc_fref, x, y_fref));
  PetscCall(CompareVecs(y, y_fref, "PCApplyTranspose", label));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&y_fref));

  PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &Y));
  PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &Y_fref));
  PetscCall(PCMatApply(pc, B, Y));
  PetscCall(PCMatApply(pc_fref, B, Y_fref));
  PetscCall(CompareDense(Y, Y_fref, "PCMatApply", label));
  PetscCall(MatDestroy(&Y));
  PetscCall(MatDestroy(&Y_fref));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Creates a PC of the given type on A with the PCPFLAREINV options
static PetscErrorCode CreatePC(Mat A, const char *type, PCPFLAREINVType inverse_type, PetscInt poly_order, PetscInt sparsity_order, PetscBool matrix_free, PC *pc)
{
  MPI_Comm comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCall(PCCreate(comm, pc));
  PetscCall(PCSetType(*pc, type));
  PetscCall(PCPFLAREINVSetType(*pc, inverse_type));
  PetscCall(PCPFLAREINVSetPolyOrder(*pc, poly_order));
  PetscCall(PCPFLAREINVSetSparsityOrder(*pc, sparsity_order));
  PetscCall(PCPFLAREINVSetMatrixFree(*pc, matrix_free));
  PetscCall(PCSetOperators(*pc, A, A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Compares pflareinv against pflareinv_fref for one set of options, through the
// fresh setup and the two reuse paths
static PetscErrorCode CompareOnePC(Mat A_in, PCPFLAREINVType inverse_type, PetscInt poly_order, PetscInt sparsity_order, PetscBool matrix_free, PetscInt *n_compared)
{
  Mat        A, B;
  Vec        x;
  PC         pc, pc_fref;
  PetscReal *coeffs, *user_coeffs;
  PetscInt   rows, cols, i;
  PetscBool  reuse;
  char       label[256];
  MPI_Comm   comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)A_in, &comm));
  PetscCall(PetscSNPrintf(label, sizeof(label), "%-15s %-9s poly_order %2" PetscInt_FMT " sparsity_order %" PetscInt_FMT, inv_names[inverse_type], matrix_free ? "mf" : "assembled", poly_order, sparsity_order));

  // Our own copy of the operator, as the reuse paths change its values
  PetscCall(MatDuplicate(A_in, MAT_COPY_VALUES, &A));
  PetscCall(CreateRandom(A, &x, &B));

  PetscCall(CreatePC(A, PCPFLAREINV, inverse_type, poly_order, sparsity_order, matrix_free, &pc));
  PetscCall(CreatePC(A, "pflareinv_fref", inverse_type, poly_order, sparsity_order, matrix_free, &pc_fref));

  // ~~~~~~~~~~~~
  // Fresh setup
  // ~~~~~~~~~~~~
  PetscCall(PCSetUp(pc));
  PetscCall(PCSetUp(pc_fref));
  PetscCall(ComparePCs(pc, pc_fref, x, B, matrix_free, label));

  // The assembled Newton polynomials with a sparsity order of 2 or more can't be rebuilt
  // in place on the sparsity of the first build: on some operators an entry outside
  // it is set ("New nonzero ... caused a malloc"), in the C and the Fortran reference
  // alike, so they skip the reuse paths
  reuse = (!matrix_free && sparsity_order >= 2 && (inverse_type == PFLAREINV_NEWTON || inverse_type == PFLAREINV_NEWTON_NO_EXTRA)) ? PETSC_FALSE : PETSC_TRUE;

  if (reuse) {
    // ~~~~~~~~~~~~
    // Reuse the stored coefficients when the operator values change
    // (SAME_NONZERO_PATTERN), so the polynomial is rebuilt with the old coefficients
    // The operator is scaled down rather than up: the CPU/Kokkos compare of the assembled
    // power basis polynomial (PFLARE_KOKKOS_DEBUG=1) has an absolute tolerance, which the
    // rounding differences in the entries of a high power of a scaled up operator exceed
    // (in the C and the Fortran reference alike)
    // ~~~~~~~~~~~~
    PetscCall(PCPFLAREINVSetReusePolyCoeffs(pc, PETSC_TRUE));
    PetscCall(PCPFLAREINVSetReusePolyCoeffs(pc_fref, PETSC_TRUE));
    PetscCall(MatScale(A, 0.5));
    PetscCall(PCSetUp(pc));
    PetscCall(PCSetUp(pc_fref));
    PetscCall(ComparePCs(pc, pc_fref, x, B, matrix_free, label));

    // ~~~~~~~~~~~~
    // Coefficients set by the user (a scaled copy of the computed ones), reused when
    // the operator values change again (back to the original operator)
    // ~~~~~~~~~~~~
    PetscCall(PCPFLAREINVGetPolyCoeffs(pc, &coeffs, &rows, &cols));
    if (coeffs && rows * cols > 0) {
      // SetPolyCoeffs takes a copy
      PetscCall(PetscMalloc1(rows * cols, &user_coeffs));
      for (i = 0; i < rows * cols; i++) user_coeffs[i] = 0.5 * coeffs[i];
      PetscCall(PCPFLAREINVSetPolyCoeffs(pc, user_coeffs, rows, cols));
      PetscCall(PCPFLAREINVSetPolyCoeffs(pc_fref, user_coeffs, rows, cols));
      PetscCall(PetscFree(user_coeffs));
      PetscCall(MatScale(A, 2.0));
      PetscCall(PCSetUp(pc));
      PetscCall(PCSetUp(pc_fref));
      PetscCall(ComparePCs(pc, pc_fref, x, B, matrix_free, label));
    }
  }

  PetscCall(PetscPrintf(comm, "PC     %s%s : match\n", label, reuse ? "" : " (no reuse)"));
  (*n_compared)++;

  PetscCall(PCDestroy(&pc));
  PetscCall(PCDestroy(&pc_fref));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Compares calculate_and_build_approximate_inverse_c against its Fortran reference
// for one set of options, including the diagonal scaling and subcomm that PCPFLAREINV
// never turns on
static PetscErrorCode CompareOneDirect(Mat A, PCPFLAREINVType inverse_type, PetscInt poly_order, PetscInt sparsity_order, PetscBool matrix_free, PetscBool diag_scale, PetscBool subcomm, PetscInt *n_compared)
{
  Mat        inv = NULL, inv_fref = NULL, B, Y, Y_fref;
  Vec        x, y, y_fref;
  PetscReal *coeffs = NULL, *coeffs_fref = NULL;
  PetscInt   rows = 0, cols = 0, rows_fref = 0, cols_fref = 0;
  int        applied = 0, applied_fref = 0;
  char       label[256];
  MPI_Comm   comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCall(PetscSNPrintf(label, sizeof(label), "%-15s %-9s poly_order %2" PetscInt_FMT " sparsity_order %" PetscInt_FMT " diag_scale %d subcomm %d", inv_names[inverse_type], matrix_free ? "mf" : "assembled", poly_order, sparsity_order, (int)diag_scale, (int)subcomm));

  calculate_and_build_approximate_inverse_c(&A, (int)inverse_type, (int)poly_order, (int)sparsity_order, matrix_free ? 1 : 0, diag_scale ? 1 : 0, subcomm ? 1 : 0, &coeffs, &rows, &cols, &inv);
  calculate_and_build_approximate_inverse_c_fref(&A, (int)inverse_type, (int)poly_order, (int)sparsity_order, matrix_free ? 1 : 0, diag_scale ? 1 : 0, subcomm ? 1 : 0, &coeffs_fref, &rows_fref, &cols_fref, &inv_fref);

  PetscCall(CompareCoeffs(comm, coeffs, rows, cols, coeffs_fref, rows_fref, cols_fref, label));

  PetscCall(CreateRandom(A, &x, &B));
  if (!matrix_free) {
    PetscCall(CompareAssembled(inv, inv_fref, label));
  } else {
    PetscCall(VecDuplicate(x, &y));
    PetscCall(VecDuplicate(x, &y_fref));
    PetscCall(MatMult(inv, x, y));
    PetscCall(MatMult(inv_fref, x, y_fref));
    PetscCall(CompareVecs(y, y_fref, "MatMult", label));
    PetscCall(MatMultTranspose(inv, x, y));
    PetscCall(MatMultTranspose(inv_fref, x, y_fref));
    PetscCall(CompareVecs(y, y_fref, "MatMultTranspose", label));
    PetscCall(VecDestroy(&y));
    PetscCall(VecDestroy(&y_fref));

    PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &Y));
    PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &Y_fref));
    pflareinv_shell_block_matapply_c(&inv, &B, &Y, &applied);
    pflareinv_shell_block_matapply_c_fref(&inv_fref, &B, &Y_fref, &applied_fref);
    PetscCheck(applied == applied_fref, comm, PETSC_ERR_PLIB, "C and Fortran reference differ in whether the block apply was done for %s", label);
    if (applied) PetscCall(CompareDense(Y, Y_fref, "block apply", label));
    PetscCall(MatDestroy(&Y));
    PetscCall(MatDestroy(&Y_fref));
  }

  PetscCall(PetscPrintf(comm, "Direct %s : match\n", label));
  (*n_compared)++;

  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&B));
  reset_inverse_mat_c(&inv);
  reset_inverse_mat_c_fref(&inv_fref);
  free(coeffs);
  free(coeffs_fref);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char **argv)
{
  Mat             A, A_sub;
  IS              rows, zero_rows;
  DM              da = NULL;
  PetscMPIInt     rank, size;
  PetscInt        rstart, rend, n_compared = 0, i_order, sparsity_order, n_zero_rows;
  PetscBool       empty_last_rank = PETSC_FALSE, flg, zero_diagonal;
  PetscViewer     fd;
  char            file[PETSC_MAX_PATH_LEN];
  PetscReal       theta, alpha;
  PCPFLAREINVType inverse_type;
  int             mf, diag_scale, subcomm;
  const PetscInt  poly_orders[] = {2, 6, 12};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  // The Fortran reference is called from this C main, so the Fortran PETSc
  // constants it uses have to be set up
  PetscCall(PetscInitializeFortran());

  // Register the pflare types (pflareinv_fref too, with PFLARE_PORT_DEBUG)
  PCRegister_PFLARE();

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

  // Does the matrix have any zero diagonal entries
  PetscCall(MatFindZeroDiagonals(A, &zero_rows));
  PetscCall(ISGetSize(zero_rows, &n_zero_rows));
  PetscCall(ISDestroy(&zero_rows));
  zero_diagonal = n_zero_rows > 0 ? PETSC_TRUE : PETSC_FALSE;

  // ~~~~~~~~~~~~
  // The PC, pflareinv against pflareinv_fref
  // ~~~~~~~~~~~~
  for (inverse_type = PFLAREINV_POWER; inverse_type <= PFLAREINV_L1JACOBI; inverse_type = (PCPFLAREINVType)(inverse_type + 1)) {
    for (mf = 0; mf <= 1; mf++) {
      // SAI/ISAI can't be applied matrix-free
      if (mf && (inverse_type == PFLAREINV_SAI || inverse_type == PFLAREINV_ISAI)) continue;
      if (inverse_type == PFLAREINV_WJACOBI || inverse_type == PFLAREINV_JACOBI || inverse_type == PFLAREINV_L1JACOBI) {
        // Neither the order nor the sparsity matter
        PetscCall(CompareOnePC(A, inverse_type, 6, 1, mf ? PETSC_TRUE : PETSC_FALSE, &n_compared));
      } else if (inverse_type == PFLAREINV_SAI || inverse_type == PFLAREINV_ISAI) {
        // Only the sparsity matters. ISAI with the sparsity of the diagonal inverts
        // the diagonal entries, so it needs them all nonzero
        for (sparsity_order = 0; sparsity_order <= 2; sparsity_order++) {
          if (inverse_type == PFLAREINV_ISAI && sparsity_order == 0 && zero_diagonal) continue;
          PetscCall(CompareOnePC(A, inverse_type, 6, sparsity_order, PETSC_FALSE, &n_compared));
        }
      } else {
        for (i_order = 0; i_order < 3; i_order++) {
          if (mf) {
            // The sparsity doesn't matter matrix-free
            PetscCall(CompareOnePC(A, inverse_type, poly_orders[i_order], 1, PETSC_TRUE, &n_compared));
          } else {
            for (sparsity_order = 0; sparsity_order <= 2; sparsity_order++) PetscCall(CompareOnePC(A, inverse_type, poly_orders[i_order], sparsity_order, PETSC_FALSE, &n_compared));
          }
        }
      }
    }
  }

  // ~~~~~~~~~~~~
  // The diagonal scaling and the subcomm, for the polynomials
  // ~~~~~~~~~~~~
  for (inverse_type = PFLAREINV_POWER; inverse_type <= PFLAREINV_NEUMANN; inverse_type = (PCPFLAREINVType)(inverse_type + 1)) {
    for (mf = 0; mf <= 1; mf++) {
      for (diag_scale = 0; diag_scale <= 1; diag_scale++) {
        for (subcomm = 0; subcomm <= 1; subcomm++) {
          // The subcomm is only for a parallel (MATMPIAIJ) matrix, and the power basis
          // refuses one
          if (subcomm && (size == 1 || inverse_type == PFLAREINV_POWER)) continue;
          PetscCall(CompareOneDirect(A, inverse_type, 6, 1, mf ? PETSC_TRUE : PETSC_FALSE, diag_scale ? PETSC_TRUE : PETSC_FALSE, subcomm ? PETSC_TRUE : PETSC_FALSE, &n_compared));
        }
      }
    }
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "port compare PC: all %" PetscInt_FMT " comparisons match\n", n_compared));

  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}
