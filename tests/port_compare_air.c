static char help[] = "Compares PCAIR against its Fortran reference implementation.\n\n";

/*
  Differential test used while the Fortran core of PFLARE is ported to C.
  Built only with make PFLARE_PORT_DEBUG=1, which also compiles the Fortran
  reference implementations in src/fref. See section 6 of
  docs/dev/fortran_removal_plan.md.

  For every set of options in the list below (each CF splitting type, each inverse
  type for Aff and for the coarsest grid, the Z types, the smooth types, the
  constraints and improvements of W and Z, processor agglomeration on/off, the
  subcomms, symmetric, auto truncation, matrix-free polynomials, diagonal scaling,
  the strong R threshold and others) this sets up an air PC, which calls the C port,
  and an air_fref PC (src/fref/PCAIR_fref.c), which is PCAIR as it was before the
  port and calls the Fortran reference, on the same operator, with the options set
  through PCSetFromOptions on both, and checks
  - every PCAIRGet* returns the same (bitwise for the reals), including the number
    of levels and the five complexities
  - PCAIRGetPolyCoeffs gives bitwise the same coefficients (or errors on both) for
    every level and every COEFFS_INV_*
  - PCApply and PCApplyTranspose of a random vector and PCMatApply of a random
    block are bitwise the same
  It does that after
  1) the first setup
  2) a second setup after the operator values change (SAME_NONZERO_PATTERN), with
     the reuse of the sparsity off, and on with each reuse amount
  3) with the reuse of the sparsity on, a third setup with the polynomial
     coefficients set by PCAIRSetPolyCoeffs (a scaled copy of the computed ones on
     every level) and PCAIRSetReusePolyCoeffs, after the values change again
  4) PCReset and a setup again
  5) changing options through the setters (which reset the hierarchy) and a setup
  Any difference is an error. The C and the Fortran do the same floating point
  operations in the same order, so everything is compared bitwise.

  The matrix is read from a PETSc binary file with -f, or is otherwise a steady
  upwinded advection-diffusion operator on a 2D structured grid (as in
  tests/port_compare_pc.c): -da_grid_x, -da_grid_y set the grid (default 30 x 30),
  -theta the direction of the velocity (default pi/4) and -alpha the diffusion
  coefficient (default 0, pure advection). -mat_type / -dm_mat_type are honoured.
  With -empty_last_rank the rows of the last rank are dropped, so that rank has no
  rows. -configs <n,m,...> compares only those sets of options (0-based
  indices into the list below).
*/
#include <petscksp.h>
#include <petscdmda.h>
#include "pflare.h"

// The C API of the air_fref reference PC in src/fref/PCAIR_fref.c (only built with
// PFLARE_PORT_DEBUG=1)
#define FREF_GET(name, type) PETSC_EXTERN PetscErrorCode PCAIRGet##name##_fref(PC, type *);
#define FREF_SET(name, type) PETSC_EXTERN PetscErrorCode PCAIRSet##name##_fref(PC, type);
FREF_GET(PrintStatsTimings, PetscBool)
FREF_GET(MaxLevels, PetscInt)
FREF_GET(CoarseEqLimit, PetscInt)
FREF_GET(AutoTruncateStartLevel, PetscInt)
FREF_GET(AutoTruncateTol, PetscReal)
FREF_GET(NumLevels, PetscInt)
FREF_GET(ProcessorAgglom, PetscBool)
FREF_GET(ProcessorAgglomRatio, PetscReal)
FREF_GET(ProcessorAgglomFactor, PetscInt)
FREF_GET(ProcessEqLimit, PetscInt)
FREF_GET(Subcomm, PetscBool)
FREF_GET(StrongThreshold, PetscReal)
FREF_GET(DDCIts, PetscInt)
FREF_GET(DDCFraction, PetscReal)
FREF_GET(CFSplittingType, CFSplittingType)
FREF_GET(MaxLubySteps, PetscInt)
FREF_GET(DiagScalePolys, PetscBool)
FREF_GET(MatrixFreePolys, PetscBool)
FREF_GET(OnePointClassicalProlong, PetscBool)
FREF_GET(FullSmoothingUpAndDown, PetscBool)
FREF_GET(Symmetric, PetscBool)
FREF_GET(ConstrainW, PetscBool)
FREF_GET(ConstrainZ, PetscBool)
FREF_GET(ImproveWIts, PetscInt)
FREF_GET(ImproveZIts, PetscInt)
FREF_GET(StrongRThreshold, PetscReal)
FREF_GET(InverseType, PCPFLAREINVType)
FREF_GET(ZType, PCAIRZType)
FREF_GET(PolyOrder, PetscInt)
FREF_GET(LairDistance, PetscInt)
FREF_GET(InverseSparsityOrder, PetscInt)
FREF_GET(CInverseType, PCPFLAREINVType)
FREF_GET(CInverseSparsityOrder, PetscInt)
FREF_GET(CPolyOrder, PetscInt)
FREF_GET(CoarsestInverseType, PCPFLAREINVType)
FREF_GET(CoarsestPolyOrder, PetscInt)
FREF_GET(CoarsestInverseSparsityOrder, PetscInt)
FREF_GET(CoarsestMatrixFreePolys, PetscBool)
FREF_GET(CoarsestDiagScalePolys, PetscBool)
FREF_GET(CoarsestSubcomm, PetscBool)
FREF_GET(RDrop, PetscReal)
FREF_GET(ADrop, PetscReal)
FREF_GET(ALump, PetscBool)
FREF_GET(ReuseSparsity, PetscBool)
FREF_GET(ReusePolyCoeffs, PetscBool)
FREF_GET(ReuseAmount, PetscInt)
FREF_GET(GridComplexity, PetscReal)
FREF_GET(OperatorComplexity, PetscReal)
FREF_GET(CycleComplexity, PetscReal)
FREF_GET(StorageComplexity, PetscReal)
FREF_GET(ReuseStorageComplexity, PetscReal)
FREF_SET(ReuseSparsity, PetscBool)
FREF_SET(ReuseAmount, PetscInt)
FREF_SET(ReusePolyCoeffs, PetscBool)
FREF_SET(StrongThreshold, PetscReal)
FREF_SET(PolyOrder, PetscInt)
FREF_SET(SmoothType, const char *)
PETSC_EXTERN PetscErrorCode PCAIRGetSmoothType_fref(PC, char *);
PETSC_EXTERN PetscErrorCode PCAIRGetPolyCoeffs_fref(PC, PetscInt, int, PetscReal **, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode PCAIRSetPolyCoeffs_fref(PC, PetscInt, int, PetscReal *, PetscInt, PetscInt);

// The sets of options compared. Each is applied on top of the defaults
// max_reuse is the largest reuse amount compared (with the reuse of the sparsity on),
// and parallel_only sets of options are only compared on more than one rank
// Two sets of options fail with a reuse amount of 3 in the Fortran reference as well
// as the C, so they skip it:
// - symmetric: the second setup (the Fortran segfaults, the C errors in MatSetOption
//   on a NULL matrix in compute_R_from_Z_cpu)
// - auto truncation: the setup after PCAIRSetPolyCoeffs and PCAIRSetReusePolyCoeffs
//   ("The smoothing temporaries can only be created when no level has them")
// serial_only sets of options are only compared on one rank: -pc_air_symmetric is not
// deterministic in parallel (the PtAP gives different roundoff from run to run, in the
// Fortran before the port too) and its reuse errors in the C there (the Fortran ignored
// the error from MatPtAP)
// no_kokkos sets of options are not compared on Kokkos matrices: -pc_air_subcomm errors
// there in the C and the Fortran reference alike (MatMPICreateNonemptySubcomm_c calls
// MatMPIAIJGetSeqAIJ on the MATMPIAIJKOKKOS Aff). It is compared with the PMIS splitting
// and the Arnoldi basis, as it errors the same way on the diagonal (MATDIAGONAL) Aff of
// the PMISR DDC splitting, and the power basis refuses a subcomm
// The highest polynomial order is 8: with orders 10 and 12 (and sparsity order 2) the absolute
// tolerance of the CPU/Kokkos compare of mat_mult_powers_share_sparsity trips on
// mat_stream_2364 (see docs/dev/kokkos.md), in the C and the Fortran reference alike
typedef struct {
  const char *options;
  PetscInt    max_reuse;
  PetscBool   parallel_only;
  PetscBool   serial_only;
  PetscBool   no_kokkos;
} air_config;

static const air_config configs[] = {
  {"", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_cf_splitting_type diag_dom", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_cf_splitting_type pmis", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_cf_splitting_type pmis_dist2", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_cf_splitting_type agg", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_cf_splitting_type pmis_agg", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_cf_splitting_type cr", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_inverse_type power", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_inverse_type arnoldi", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_inverse_type newton", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_inverse_type newton_no_extra", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_inverse_type neumann", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_inverse_type sai", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_inverse_type isai", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_inverse_type wjacobi", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_inverse_type jacobi", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_coarsest_inverse_type power", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_coarsest_inverse_type arnoldi", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_coarsest_inverse_type newton", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_coarsest_inverse_type newton_no_extra", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_coarsest_inverse_type neumann", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_coarsest_inverse_type sai", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_coarsest_inverse_type isai", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_coarsest_inverse_type wjacobi", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_coarsest_inverse_type jacobi", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_z_type product", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_z_type lair", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_z_type lair_sai", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_z_type lair -pc_air_lair_distance 1", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_smooth_type ff", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_smooth_type fc", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_smooth_type fcf", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_smooth_type fcf -pc_air_c_inverse_type arnoldi -pc_air_c_poly_order 3 -pc_air_c_inverse_sparsity_order 2", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_smooth_type fc -pc_air_full_smoothing_up_and_down", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_constrain_w", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_constrain_z", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_improve_w_its 3", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_improve_z_its 3", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_processor_agglom 0", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_processor_agglom_ratio 1.1 -pc_air_processor_agglom_factor 2 -pc_air_process_eq_limit 200", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_subcomm -pc_air_cf_splitting_type pmis -pc_air_inverse_type arnoldi", 3, PETSC_TRUE, PETSC_FALSE, PETSC_TRUE},
  {"-pc_air_coarsest_subcomm", 3, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_symmetric", 2, PETSC_FALSE, PETSC_TRUE, PETSC_FALSE},
  {"-pc_air_auto_truncate_start_level 1 -pc_air_auto_truncate_tol 1e-1", 2, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_matrix_free_polys", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_coarsest_matrix_free_polys", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_diag_scale_polys", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_coarsest_diag_scale_polys", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_strong_r_threshold 0.1", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_one_point_classical_prolong 0", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_a_drop 1e-3 -pc_air_r_drop 0.1", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_a_lump", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_max_levels 3", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_coarse_eq_limit 200", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_poly_order 8 -pc_air_inverse_sparsity_order 2", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_coarsest_poly_order 8 -pc_air_coarsest_inverse_sparsity_order 2", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_ddc_its 2 -pc_air_ddc_fraction 0.2 -pc_air_max_luby_steps 3", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_strong_threshold 0.3", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
  {"-pc_air_inverse_type newton -pc_air_matrix_free_polys -pc_air_smooth_type fcf -pc_air_z_type lair", 3, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE},
};
#define N_CONFIGS ((PetscInt)(sizeof(configs) / sizeof(configs[0])))

// The number of right hand sides in the blocks for PCMatApply
#define N_RHS 3

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Builds the upwinded advection-diffusion operator on the DMDA (as tests/port_compare_pc.c)
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

// Changes the values of the operator but not its sparsity: scales the rows by
// factors between 0.75 and 1 (down rather than up, see tests/port_compare_pc.c; with
// factors down to 0.5 the absolute tolerance of the CPU/Kokkos compare of
// mat_mult_powers_share_sparsity trips on mat_stream_2364, in the C and the Fortran
// reference alike)
static PetscErrorCode ChangeValues(Mat A, PetscInt step)
{
  PetscRandom rctx;
  Vec         l;
  MPI_Comm    comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCall(PetscRandomCreate(comm, &rctx));
  PetscCall(PetscRandomSetInterval(rctx, 0.75, 1.0));
  PetscCall(PetscRandomSetSeed(rctx, 0x1234 + (unsigned long)step));
  PetscCall(PetscRandomSeed(rctx));
  PetscCall(MatCreateVecs(A, NULL, &l));
  PetscCall(VecSetRandom(l, rctx));
  PetscCall(MatDiagonalScale(A, l, NULL));
  PetscCall(VecDestroy(&l));
  PetscCall(PetscRandomDestroy(&rctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Compares one getter of the air PC and the air_fref PC (bitwise for the reals)
#define COMPARE_GET(name, type) \
  do { \
    type val_c, val_f; \
    PetscCall(PCAIRGet##name(pc, &val_c)); \
    PetscCall(PCAIRGet##name##_fref(pc_fref, &val_f)); \
    PetscCheck(val_c == val_f, comm, PETSC_ERR_PLIB, "C and Fortran reference differ in PCAIRGet" #name " for %s", label); \
  } while (0)

// Compares every getter
static PetscErrorCode CompareGetters(PC pc, PC pc_fref, const char *label)
{
  char     smooth_c[16], smooth_f[16];
  MPI_Comm comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)pc, &comm));
  COMPARE_GET(PrintStatsTimings, PetscBool);
  COMPARE_GET(MaxLevels, PetscInt);
  COMPARE_GET(CoarseEqLimit, PetscInt);
  COMPARE_GET(AutoTruncateStartLevel, PetscInt);
  COMPARE_GET(AutoTruncateTol, PetscReal);
  COMPARE_GET(NumLevels, PetscInt);
  COMPARE_GET(ProcessorAgglom, PetscBool);
  COMPARE_GET(ProcessorAgglomRatio, PetscReal);
  COMPARE_GET(ProcessorAgglomFactor, PetscInt);
  COMPARE_GET(ProcessEqLimit, PetscInt);
  COMPARE_GET(Subcomm, PetscBool);
  COMPARE_GET(StrongThreshold, PetscReal);
  COMPARE_GET(DDCIts, PetscInt);
  COMPARE_GET(DDCFraction, PetscReal);
  COMPARE_GET(CFSplittingType, CFSplittingType);
  COMPARE_GET(MaxLubySteps, PetscInt);
  COMPARE_GET(DiagScalePolys, PetscBool);
  COMPARE_GET(MatrixFreePolys, PetscBool);
  COMPARE_GET(OnePointClassicalProlong, PetscBool);
  COMPARE_GET(FullSmoothingUpAndDown, PetscBool);
  COMPARE_GET(Symmetric, PetscBool);
  COMPARE_GET(ConstrainW, PetscBool);
  COMPARE_GET(ConstrainZ, PetscBool);
  COMPARE_GET(ImproveWIts, PetscInt);
  COMPARE_GET(ImproveZIts, PetscInt);
  COMPARE_GET(StrongRThreshold, PetscReal);
  COMPARE_GET(InverseType, PCPFLAREINVType);
  COMPARE_GET(ZType, PCAIRZType);
  COMPARE_GET(PolyOrder, PetscInt);
  COMPARE_GET(LairDistance, PetscInt);
  COMPARE_GET(InverseSparsityOrder, PetscInt);
  COMPARE_GET(CInverseType, PCPFLAREINVType);
  COMPARE_GET(CInverseSparsityOrder, PetscInt);
  COMPARE_GET(CPolyOrder, PetscInt);
  COMPARE_GET(CoarsestInverseType, PCPFLAREINVType);
  COMPARE_GET(CoarsestPolyOrder, PetscInt);
  COMPARE_GET(CoarsestInverseSparsityOrder, PetscInt);
  COMPARE_GET(CoarsestMatrixFreePolys, PetscBool);
  COMPARE_GET(CoarsestDiagScalePolys, PetscBool);
  COMPARE_GET(CoarsestSubcomm, PetscBool);
  COMPARE_GET(RDrop, PetscReal);
  COMPARE_GET(ADrop, PetscReal);
  COMPARE_GET(ALump, PetscBool);
  COMPARE_GET(ReuseSparsity, PetscBool);
  COMPARE_GET(ReusePolyCoeffs, PetscBool);
  COMPARE_GET(ReuseAmount, PetscInt);
  // The complexities count the matvecs of every polynomial (compute_mf_gmres_poly_num_matvecs),
  // which reads the uninitialised coefficients of the inverses that are not GMRES polynomials
  // (the NaNs of the debug malloc trap with -fp_trap, in the Fortran before the port too), so
  // the floating point traps are off around them
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  COMPARE_GET(GridComplexity, PetscReal);
  COMPARE_GET(OperatorComplexity, PetscReal);
  COMPARE_GET(CycleComplexity, PetscReal);
  COMPARE_GET(StorageComplexity, PetscReal);
  COMPARE_GET(ReuseStorageComplexity, PetscReal);
  PetscCall(PetscFPTrapPop());
  PetscCall(PCAIRGetSmoothType(pc, smooth_c));
  PetscCall(PCAIRGetSmoothType_fref(pc_fref, smooth_f));
  PetscCheck(strcmp(smooth_c, smooth_f) == 0, comm, PETSC_ERR_PLIB, "C and Fortran reference differ in PCAIRGetSmoothType for %s", label);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Whether an inverse type is a GMRES polynomial, the only ones that compute coefficients
static PetscBool IsGmresPoly(PCPFLAREINVType type)
{
  return (type == PFLAREINV_POWER || type == PFLAREINV_ARNOLDI || type == PFLAREINV_NEWTON || type == PFLAREINV_NEWTON_NO_EXTRA) ? PETSC_TRUE : PETSC_FALSE;
}

// Gets the coefficients of one level and inverse from the air (fref false) or the
// air_fref PC, returning whether there are any rather than erroring
static PetscErrorCode GetPolyCoeffsOrNone(PC pc, PetscBool fref, PetscInt petsc_level, int which_inverse, PetscReal **coeffs, PetscInt *rows, PetscInt *cols, PetscBool *found)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  *coeffs = NULL;
  *rows   = 0;
  *cols   = 0;
  // Both error if there are no coefficients, without printing with this handler
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  if (fref) ierr = PCAIRGetPolyCoeffs_fref(pc, petsc_level, which_inverse, coeffs, rows, cols);
  else ierr = PCAIRGetPolyCoeffs(pc, petsc_level, which_inverse, coeffs, rows, cols);
  PetscCall(PetscPopErrorHandler());
  *found = ierr == PETSC_SUCCESS ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Compares the coefficients of every level and inverse, returning how many sets were
// found. With set_scaled, it sets a scaled copy of the coefficients on both PCs
static PetscErrorCode ComparePolyCoeffs(PC pc, PC pc_fref, PetscBool set_scaled, const char *label, PetscInt *n_found)
{
  PetscInt        num_levels, petsc_level, rows, cols, rows_f, cols_f, i;
  PetscReal      *coeffs, *coeffs_f, *user_coeffs, strong_r_threshold;
  PetscBool       found, found_f, ok, full_smoothing, computed[4];
  PCPFLAREINVType inv_type, c_inv_type, coarsest_inv_type;
  PCAIRZType      z_type;
  int        which_inverse;
  const int  which[] = {COEFFS_INV_AFF, COEFFS_INV_AFF_DROPPED, COEFFS_INV_ACC, COEFFS_INV_COARSE};
  MPI_Comm   comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)pc, &comm));
  *n_found = 0;
  PetscCall(PCAIRGetNumLevels(pc, &num_levels));
  // The coefficients are stored for every inverse that is set up, but are only
  // computed by the GMRES polynomials, and for the inverse of the dropped Aff only
  // when it is used to build Z. The others are left uninitialised (in the C and the
  // Fortran reference alike), so only their sizes are compared
  PetscCall(PCAIRGetInverseType(pc, &inv_type));
  PetscCall(PCAIRGetCInverseType(pc, &c_inv_type));
  PetscCall(PCAIRGetCoarsestInverseType(pc, &coarsest_inv_type));
  PetscCall(PCAIRGetZType(pc, &z_type));
  PetscCall(PCAIRGetStrongRThreshold(pc, &strong_r_threshold));
  PetscCall(PCAIRGetFullSmoothingUpAndDown(pc, &full_smoothing));
  computed[0] = IsGmresPoly(inv_type);
  computed[1] = (IsGmresPoly(inv_type) && z_type == AIR_Z_PRODUCT && (strong_r_threshold != 0.0 || full_smoothing)) ? PETSC_TRUE : PETSC_FALSE;
  computed[2] = IsGmresPoly(c_inv_type);
  // A single level is a Jacobi PC, without a coarse grid inverse
  computed[3] = (IsGmresPoly(coarsest_inv_type) && num_levels > 1) ? PETSC_TRUE : PETSC_FALSE;
  // Out of range levels too (-1 and num_levels), which must error on both
  for (petsc_level = -1; petsc_level <= num_levels; petsc_level++) {
    for (i = 0; i < 4; i++) {
      which_inverse = which[i];
      PetscCall(GetPolyCoeffsOrNone(pc, PETSC_FALSE, petsc_level, which_inverse, &coeffs, &rows, &cols, &found));
      PetscCall(GetPolyCoeffsOrNone(pc_fref, PETSC_TRUE, petsc_level, which_inverse, &coeffs_f, &rows_f, &cols_f, &found_f));
      PetscCheck(found == found_f, comm, PETSC_ERR_PLIB, "C and Fortran reference differ in whether there are coefficients on petsc_level %" PetscInt_FMT " for inverse %d for %s", petsc_level, which_inverse, label);
      if (!found) continue;
      ok = (rows == rows_f && cols == cols_f) ? PETSC_TRUE : PETSC_FALSE;
      if (!ok) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Coefficients on petsc_level %" PetscInt_FMT " for inverse %d are %" PetscInt_FMT " x %" PetscInt_FMT " in C and %" PetscInt_FMT " x %" PetscInt_FMT " in the Fortran reference\n", petsc_level, which_inverse, rows, cols, rows_f, cols_f));
      for (PetscInt k = 0; k < rows * cols && ok && computed[i]; k++) {
        if (coeffs[k] != coeffs_f[k]) {
          ok = PETSC_FALSE;
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "Coefficient %" PetscInt_FMT " on petsc_level %" PetscInt_FMT " for inverse %d is %.17g in C and %.17g in the Fortran reference\n", k, petsc_level, which_inverse, (double)coeffs[k], (double)coeffs_f[k]));
        }
      }
      PetscCall(CheckAll(comm, ok, "polynomial coefficients", label));
      (*n_found)++;
      // The coarse grid coefficients don't depend on the level, so set them once
      // (only those computed, the others are uninitialised)
      if (set_scaled && computed[i] && rows * cols > 0 && (which_inverse != COEFFS_INV_COARSE || petsc_level == 0)) {
        PetscCall(PetscMalloc1(rows * cols, &user_coeffs));
        // Scaled by 0.9: with Newton roots scaled by 0.5 on the advection-diffusion operator
        // the reused (reuse amount 3) assembled inverse no longer matches the fresh build the
        // stage 5 check in port_interfaces_poly runs for the Fortran reference
        for (PetscInt k = 0; k < rows * cols; k++) user_coeffs[k] = 0.9 * coeffs[k];
        PetscCall(PCAIRSetPolyCoeffs(pc, petsc_level, which_inverse, user_coeffs, rows, cols));
        PetscCall(PCAIRSetPolyCoeffs_fref(pc_fref, petsc_level, which_inverse, user_coeffs, rows, cols));
        PetscCall(PetscFree(user_coeffs));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Compares the air and air_fref PCs after a setup: the getters, the coefficients
// and PCApply, PCApplyTranspose and PCMatApply
static PetscErrorCode ComparePCs(PC pc, PC pc_fref, Vec x, Mat B, const char *label)
{
  Mat      Y, Y_fref;
  Vec      y, y_fref;
  PetscInt n_found;

  PetscFunctionBeginUser;
  PetscCall(CompareGetters(pc, pc_fref, label));
  PetscCall(ComparePolyCoeffs(pc, pc_fref, PETSC_FALSE, label, &n_found));

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

// Creates a PC of the given type on A with the options in the string, set through
// PCSetFromOptions with an options database of its own
static PetscErrorCode CreatePC(Mat A, const char *type, const char *options, PC *pc)
{
  PetscOptions opts;
  MPI_Comm     comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCall(PetscOptionsCreate(&opts));
  PetscCall(PetscOptionsInsertString(opts, options));
  PetscCall(PCCreate(comm, pc));
  PetscCall(PetscObjectSetOptions((PetscObject)*pc, opts));
  PetscCall(PCSetType(*pc, type));
  PetscCall(PCSetOperators(*pc, A, A));
  PetscCall(PCSetFromOptions(*pc));
  // Back to the global database, so the sub objects use it as they would normally
  PetscCall(PetscObjectSetOptions((PetscObject)*pc, NULL));
  PetscCall(PetscOptionsDestroy(&opts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Compares air against air_fref for one set of options and one reuse amount
// (0 for the reuse of the sparsity off), through the setups listed at the top
static PetscErrorCode CompareOne(Mat A_in, PetscInt config, PetscInt reuse_amount, PetscInt *n_compared)
{
  Mat      A, B;
  Vec      x;
  PC       pc, pc_fref;
  PetscInt n_found, step = 0;
  char     label[512], label_step[600];
  MPI_Comm comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)A_in, &comm));
  PetscCall(PetscSNPrintf(label, sizeof(label), "config %2" PetscInt_FMT " reuse_amount %" PetscInt_FMT " \"%s\"", config, reuse_amount, configs[config].options));

  // Our own copy of the operator, as the values are changed
  PetscCall(MatDuplicate(A_in, MAT_COPY_VALUES, &A));
  PetscCall(CreateRandom(A, &x, &B));

  PetscCall(CreatePC(A, PCAIR, configs[config].options, &pc));
  PetscCall(CreatePC(A, "air_fref", configs[config].options, &pc_fref));
  if (reuse_amount > 0) {
    PetscCall(PCAIRSetReuseSparsity(pc, PETSC_TRUE));
    PetscCall(PCAIRSetReuseSparsity_fref(pc_fref, PETSC_TRUE));
    PetscCall(PCAIRSetReuseAmount(pc, reuse_amount));
    PetscCall(PCAIRSetReuseAmount_fref(pc_fref, reuse_amount));
  }
  // Nothing has been built yet
  PetscCall(CompareGetters(pc, pc_fref, label));

  // ~~~~~~~~~~~~
  // 1) Fresh setup
  // ~~~~~~~~~~~~
  PetscCall(PetscSNPrintf(label_step, sizeof(label_step), "%s, first setup", label));
  PetscCall(PCSetUp(pc));
  PetscCall(PCSetUp(pc_fref));
  PetscCall(ComparePCs(pc, pc_fref, x, B, label_step));

  // ~~~~~~~~~~~~
  // 2) The operator values change (SAME_NONZERO_PATTERN)
  // ~~~~~~~~~~~~
  PetscCall(PetscSNPrintf(label_step, sizeof(label_step), "%s, second setup", label));
  PetscCall(ChangeValues(A, step++));
  PetscCall(PCSetUp(pc));
  PetscCall(PCSetUp(pc_fref));
  PetscCall(ComparePCs(pc, pc_fref, x, B, label_step));

  // ~~~~~~~~~~~~
  // 3) The coefficients set by the user and reused when the values change again
  // ~~~~~~~~~~~~
  if (reuse_amount > 0) {
    PetscCall(PetscSNPrintf(label_step, sizeof(label_step), "%s, set coefficients", label));
    PetscCall(ComparePolyCoeffs(pc, pc_fref, PETSC_TRUE, label_step, &n_found));
    PetscCall(PCAIRSetReusePolyCoeffs(pc, PETSC_TRUE));
    PetscCall(PCAIRSetReusePolyCoeffs_fref(pc_fref, PETSC_TRUE));
    PetscCall(ChangeValues(A, step++));
    PetscCall(PCSetUp(pc));
    PetscCall(PCSetUp(pc_fref));
    PetscCall(ComparePCs(pc, pc_fref, x, B, label_step));
    PetscCall(PCAIRSetReusePolyCoeffs(pc, PETSC_FALSE));
    PetscCall(PCAIRSetReusePolyCoeffs_fref(pc_fref, PETSC_FALSE));
  }

  // ~~~~~~~~~~~~
  // 4) PCReset and setup again
  // ~~~~~~~~~~~~
  PetscCall(PetscSNPrintf(label_step, sizeof(label_step), "%s, after PCReset", label));
  PetscCall(PCReset(pc));
  PetscCall(PCReset(pc_fref));
  PetscCall(CompareGetters(pc, pc_fref, label_step));
  PetscCall(PCSetOperators(pc, A, A));
  PetscCall(PCSetOperators(pc_fref, A, A));
  PetscCall(PCSetUp(pc));
  PetscCall(PCSetUp(pc_fref));
  PetscCall(ComparePCs(pc, pc_fref, x, B, label_step));

  // ~~~~~~~~~~~~
  // 5) Options changed through the setters, which reset the hierarchy
  // ~~~~~~~~~~~~
  if (reuse_amount == 0) {
    PetscCall(PetscSNPrintf(label_step, sizeof(label_step), "%s, after the setters", label));
    PetscCall(PCAIRSetPolyOrder(pc, 4));
    PetscCall(PCAIRSetPolyOrder_fref(pc_fref, 4));
    PetscCall(PCAIRSetSmoothType(pc, "fcf"));
    PetscCall(PCAIRSetSmoothType_fref(pc_fref, "fcf"));
    PetscCall(CompareGetters(pc, pc_fref, label_step));
    PetscCall(PCSetUp(pc));
    PetscCall(PCSetUp(pc_fref));
    PetscCall(ComparePCs(pc, pc_fref, x, B, label_step));
  }

  PetscCall(PetscPrintf(comm, "AIR %s : match\n", label));
  (*n_compared)++;

  PetscCall(PCDestroy(&pc));
  PetscCall(PCDestroy(&pc_fref));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char **argv)
{
  Mat         A, A_sub;
  IS          rows;
  DM          da = NULL;
  PetscMPIInt rank, size;
  PetscInt    rstart, rend, n_compared = 0, config, reuse_amount, config_list[N_CONFIGS], n_list = N_CONFIGS, i_list;
  PetscBool   in_list;
  PetscBool   empty_last_rank = PETSC_FALSE, flg, is_kokkos;
  MatType     mat_type;
  char       *kokkos_str;
  PetscViewer fd;
  char        file[PETSC_MAX_PATH_LEN];
  PetscReal   theta, alpha;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  // The Fortran reference is called from this C main, so the Fortran PETSc
  // constants it uses have to be set up
  PetscCall(PetscInitializeFortran());

  // Register the pflare types (air_fref too, with PFLARE_PORT_DEBUG)
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

  // ~~~~~~~~~~~~
  // air against air_fref, for every set of options and reuse amount
  // ~~~~~~~~~~~~
  // -configs <n,m,...> compares only those sets of options
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-configs", config_list, &n_list, &flg));
  if (!flg) n_list = -1;
  PetscCall(MatGetType(A, &mat_type));
  PetscCall(PetscStrstr(mat_type, "kokkos", &kokkos_str));
  is_kokkos = kokkos_str ? PETSC_TRUE : PETSC_FALSE;
  for (config = 0; config < N_CONFIGS; config++) {
    if (n_list >= 0) {
      in_list = PETSC_FALSE;
      for (i_list = 0; i_list < n_list; i_list++) {
        if (config_list[i_list] == config) in_list = PETSC_TRUE;
      }
      if (!in_list) continue;
    }
    // The subcomms need a parallel matrix
    if (configs[config].parallel_only && size == 1) continue;
    if (configs[config].serial_only && size > 1) continue;
    if (configs[config].no_kokkos && is_kokkos) continue;
    for (reuse_amount = 0; reuse_amount <= configs[config].max_reuse; reuse_amount++) PetscCall(CompareOne(A, config, reuse_amount, &n_compared));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "port compare AIR: all %" PetscInt_FMT " comparisons match\n", n_compared));

  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}
