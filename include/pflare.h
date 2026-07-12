#if !defined (PFLARE_C_DEF_H)
#define PFLARE_C_DEF_H

#include "petsc.h"

/* SUBMANSEC = PC */

/*E
  PCPFLAREINVType - The type of approximate inverse applied by `PCPFLAREINV`, and used as the
  smoother and coarse-grid solver by `PCAIR`

  Values:
+ `PFLAREINV_POWER`           - GMRES polynomial with the power basis
. `PFLAREINV_ARNOLDI`         - GMRES polynomial with the Arnoldi basis
. `PFLAREINV_NEWTON`          - GMRES polynomial with the Newton basis, with extra roots added for stability
. `PFLAREINV_NEWTON_NO_EXTRA` - GMRES polynomial with the Newton basis, without extra roots
. `PFLAREINV_NEUMANN`         - Neumann polynomial
. `PFLAREINV_SAI`             - sparse approximate inverse; cannot be applied matrix-free
. `PFLAREINV_ISAI`            - incomplete sparse approximate inverse; cannot be applied matrix-free
. `PFLAREINV_WJACOBI`         - weighted Jacobi
- `PFLAREINV_JACOBI`          - unweighted Jacobi

  Level: intermediate

  Note:
  The two Jacobi types exist mainly for use as smoothers in `PCAIR`; for a standalone Jacobi
  preconditioner use PETSc's `PCJACOBI`.

.seealso: [](ch_ksp), `PCPFLAREINV`, `PCAIR`, `PCPFLAREINVSetType()`, `PCPFLAREINVGetType()`, `PCAIRSetInverseType()`, `PCAIRGetInverseType()`
E*/
typedef enum {
   PFLAREINV_POWER,
   PFLAREINV_ARNOLDI,
   PFLAREINV_NEWTON,
   PFLAREINV_NEWTON_NO_EXTRA,
   PFLAREINV_NEUMANN,
   PFLAREINV_SAI,
   PFLAREINV_ISAI,
   PFLAREINV_WJACOBI,
   PFLAREINV_JACOBI,
}  PCPFLAREINVType;

/*E
  PCAIRZType - The type of grid-transfer (restriction) operator built by `PCAIR`

  Values:
+ `AIR_Z_PRODUCT`  - approximate ideal restriction built from a matrix product (gives AIRG when combined with a polynomial inverse type)
. `AIR_Z_LAIR`     - local approximate ideal restriction (lAIR)
- `AIR_Z_LAIR_SAI` - local approximate ideal restriction built with a sparse approximate inverse

  Level: intermediate

  Note:
  Together with `PCAIRSetInverseType()` this selects the reduction multigrid method: for example
  product + arnoldi gives AIRG, and lair + wjacobi gives lAIR.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetZType()`, `PCAIRGetZType()`, `PCAIRSetInverseType()`, `PCAIRSetLairDistance()`
E*/
typedef enum {
   AIR_Z_PRODUCT,
   AIR_Z_LAIR,
   AIR_Z_LAIR_SAI,
}  PCAIRZType;

/*E
  CFSplittingType - The coarse/fine (CF) splitting algorithm used by `PCAIR`

  Values:
+ `CF_PMISR_DDC`  - two-pass splitting giving a diagonally dominant fine-fine block (PMISR DDC)
. `CF_DIAG_DOM`   - two-pass splitting enforcing a fixed diagonal dominance ratio (set by the strong threshold) in the fine-fine block
. `CF_PMIS`       - PMIS with a symmetrised strength matrix
. `CF_PMIS_DIST2` - distance-2 PMIS, with strength matrix formed from S'S + S and then symmetrised
. `CF_AGG`        - aggregation with root-nodes as C points; processor-local aggregation in parallel
- `CF_PMIS_AGG`   - PMIS on boundary nodes with a symmetrised strength matrix, then processor-local aggregation

  Level: intermediate

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetCFSplittingType()`, `PCAIRGetCFSplittingType()`, `PCAIRSetStrongThreshold()`
E*/
typedef enum {
   CF_PMISR_DDC,
   CF_DIAG_DOM,
   CF_PMIS,
   CF_PMIS_DIST2,
   CF_AGG,
   CF_PMIS_AGG,
}  CFSplittingType;

#define PCAIR "air"
#define PCPFLAREINV "pflareinv"

typedef enum {
   COEFFS_INV_AFF,
   COEFFS_INV_AFF_DROPPED,
   COEFFS_INV_ACC,
   COEFFS_INV_COARSE,
}  WhichInverseType;

/* This should be called to register the new PC types */
PETSC_EXTERN void PCRegister_PFLARE();

/* Can call the CF splitting separate to everything */
PETSC_EXTERN void compute_cf_splitting(Mat, int, PetscReal, int, int, int, PetscReal, IS*, IS*);
PETSC_EXTERN void compute_diag_dom_submatrix(Mat, PetscReal, Mat*);

/* Restrict input_mat onto output_mat's existing sparsity pattern.
   With alpha_int = 1, performs output_mat += alpha * input_mat on output_mat's
   pattern; with alpha_int = 0, performs output_mat = input_mat on output's pattern.
   lump_int = 1 adds dropped entries to the diagonal. Auto-dispatches between the
   CPU and Kokkos implementations based on the input matrix type. */
PETSC_EXTERN void remove_from_sparse_match(Mat, Mat, int, int, PetscReal);

/* Define PCPFLAREINV get routines */
PETSC_EXTERN PetscErrorCode PCPFLAREINVGetPolyOrder(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCPFLAREINVGetSparsityOrder(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCPFLAREINVGetType(PC, PCPFLAREINVType *);
PETSC_EXTERN PetscErrorCode PCPFLAREINVGetMatrixFree(PC, PetscBool *);
/* Returns the underlying approximate-inverse matrix (borrowed reference - do not
   destroy; only valid after PCSetUp and until the next setup/reset) */
PETSC_EXTERN PetscErrorCode PCPFLAREINVGetInverseMat(PC, Mat *);
/* Define PCPFLAREINV set routines */
PETSC_EXTERN PetscErrorCode PCPFLAREINVSetPolyOrder(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCPFLAREINVSetSparsityOrder(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCPFLAREINVSetType(PC, PCPFLAREINVType);
PETSC_EXTERN PetscErrorCode PCPFLAREINVSetMatrixFree(PC, PetscBool);

/* Get/Set polynomial coefficients stored after PCSetUp */
PETSC_EXTERN PetscErrorCode PCPFLAREINVGetPolyCoeffs(PC, PetscReal **, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode PCPFLAREINVSetPolyCoeffs(PC, PetscReal *, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PCPFLAREINVGetReusePolyCoeffs(PC, PetscBool *);
PETSC_EXTERN PetscErrorCode PCPFLAREINVSetReusePolyCoeffs(PC, PetscBool);

/* Define PCAIR get routines */
PETSC_EXTERN PetscErrorCode PCAIRGetPrintStatsTimings(PC, PetscBool *);
PETSC_EXTERN PetscErrorCode PCAIRGetMaxLevels(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCAIRGetCoarseEqLimit(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCAIRGetAutoTruncateStartLevel(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCAIRGetAutoTruncateTol(PC, PetscReal *);
PETSC_EXTERN PetscErrorCode PCAIRGetNumLevels(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCAIRGetProcessorAgglom(PC, PetscBool *);
PETSC_EXTERN PetscErrorCode PCAIRGetProcessorAgglomRatio(PC, PetscReal *);
PETSC_EXTERN PetscErrorCode PCAIRGetProcessorAgglomFactor(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCAIRGetProcessEqLimit(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCAIRGetSubcomm(PC, PetscBool *);
PETSC_EXTERN PetscErrorCode PCAIRGetStrongThreshold(PC, PetscReal *);
PETSC_EXTERN PetscErrorCode PCAIRGetDDCIts(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCAIRGetDDCFraction(PC, PetscReal *);
PETSC_EXTERN PetscErrorCode PCAIRGetCFSplittingType(PC, CFSplittingType *);
PETSC_EXTERN PetscErrorCode PCAIRGetMaxLubySteps(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCAIRGetSmoothType(PC, char *);
PETSC_EXTERN PetscErrorCode PCAIRGetDiagScalePolys(PC, PetscBool *);
PETSC_EXTERN PetscErrorCode PCAIRGetMatrixFreePolys(PC, PetscBool *);
PETSC_EXTERN PetscErrorCode PCAIRGetOnePointClassicalProlong(PC, PetscBool *);
PETSC_EXTERN PetscErrorCode PCAIRGetFullSmoothingUpAndDown(PC, PetscBool *);
PETSC_EXTERN PetscErrorCode PCAIRGetSymmetric(PC, PetscBool *);
PETSC_EXTERN PetscErrorCode PCAIRGetConstrainW(PC, PetscBool *);
PETSC_EXTERN PetscErrorCode PCAIRGetConstrainZ(PC, PetscBool *);
PETSC_EXTERN PetscErrorCode PCAIRGetImproveWIts(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCAIRGetImproveZIts(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCAIRGetStrongRThreshold(PC, PetscReal *);
PETSC_EXTERN PetscErrorCode PCAIRGetInverseType(PC, PCPFLAREINVType *);
PETSC_EXTERN PetscErrorCode PCAIRGetZType(PC, PCAIRZType *);
PETSC_EXTERN PetscErrorCode PCAIRGetPolyOrder(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCAIRGetLairDistance(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCAIRGetInverseSparsityOrder(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCAIRGetCInverseType(PC, PCPFLAREINVType *);
PETSC_EXTERN PetscErrorCode PCAIRGetCInverseSparsityOrder(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCAIRGetCPolyOrder(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestInverseType(PC, PCPFLAREINVType *);
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestPolyOrder(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestInverseSparsityOrder(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestMatrixFreePolys(PC, PetscBool *);
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestDiagScalePolys(PC, PetscBool *);
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestSubcomm(PC, PetscBool *);
PETSC_EXTERN PetscErrorCode PCAIRGetRDrop(PC, PetscReal *);
PETSC_EXTERN PetscErrorCode PCAIRGetADrop(PC, PetscReal *);
PETSC_EXTERN PetscErrorCode PCAIRGetALump(PC, PetscBool *);
PETSC_EXTERN PetscErrorCode PCAIRGetReuseSparsity(PC, PetscBool *);
PETSC_EXTERN PetscErrorCode PCAIRGetReusePolyCoeffs(PC, PetscBool *);
PETSC_EXTERN PetscErrorCode PCAIRGetReuseAmount(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCAIRGetPolyCoeffs(PC, PetscInt, int, PetscReal **, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode PCAIRGetGridComplexity(PC, PetscReal *);
PETSC_EXTERN PetscErrorCode PCAIRGetOperatorComplexity(PC, PetscReal *);
PETSC_EXTERN PetscErrorCode PCAIRGetCycleComplexity(PC, PetscReal *);
PETSC_EXTERN PetscErrorCode PCAIRGetStorageComplexity(PC, PetscReal *);
PETSC_EXTERN PetscErrorCode PCAIRGetReuseStorageComplexity(PC, PetscReal *);
/* Define PCAIR set routines */
PETSC_EXTERN PetscErrorCode PCAIRSetPrintStatsTimings(PC, PetscBool);
PETSC_EXTERN PetscErrorCode PCAIRSetMaxLevels(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCAIRSetCoarseEqLimit(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCAIRSetAutoTruncateStartLevel(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCAIRSetAutoTruncateTol(PC, PetscReal);
PETSC_EXTERN PetscErrorCode PCAIRSetProcessorAgglom(PC, PetscBool);
PETSC_EXTERN PetscErrorCode PCAIRSetProcessorAgglomRatio(PC, PetscReal);
PETSC_EXTERN PetscErrorCode PCAIRSetProcessorAgglomFactor(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCAIRSetProcessEqLimit(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCAIRSetSubcomm(PC, PetscBool);
PETSC_EXTERN PetscErrorCode PCAIRSetStrongThreshold(PC, PetscReal);
PETSC_EXTERN PetscErrorCode PCAIRSetDDCIts(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCAIRSetDDCFraction(PC, PetscReal);
PETSC_EXTERN PetscErrorCode PCAIRSetCFSplittingType(PC, CFSplittingType);
PETSC_EXTERN PetscErrorCode PCAIRSetMaxLubySteps(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCAIRSetSmoothType(PC, const char *);
PETSC_EXTERN PetscErrorCode PCAIRSetDiagScalePolys(PC, PetscBool);
PETSC_EXTERN PetscErrorCode PCAIRSetMatrixFreePolys(PC, PetscBool);
PETSC_EXTERN PetscErrorCode PCAIRSetOnePointClassicalProlong(PC, PetscBool);
PETSC_EXTERN PetscErrorCode PCAIRSetFullSmoothingUpAndDown(PC, PetscBool);
PETSC_EXTERN PetscErrorCode PCAIRSetSymmetric(PC, PetscBool);
PETSC_EXTERN PetscErrorCode PCAIRSetConstrainW(PC, PetscBool);
PETSC_EXTERN PetscErrorCode PCAIRSetConstrainZ(PC, PetscBool);
PETSC_EXTERN PetscErrorCode PCAIRSetImproveWIts(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCAIRSetImproveZIts(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCAIRSetStrongRThreshold(PC, PetscReal);
PETSC_EXTERN PetscErrorCode PCAIRSetInverseType(PC, PCPFLAREINVType);
PETSC_EXTERN PetscErrorCode PCAIRSetZType(PC, PCAIRZType);
PETSC_EXTERN PetscErrorCode PCAIRSetLairDistance(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCAIRSetPolyOrder(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCAIRSetInverseSparsityOrder(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCAIRSetCInverseType(PC, PCPFLAREINVType);
PETSC_EXTERN PetscErrorCode PCAIRSetCPolyOrder(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCAIRSetCInverseSparsityOrder(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestInverseType(PC, PCPFLAREINVType);
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestPolyOrder(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestInverseSparsityOrder(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestMatrixFreePolys(PC, PetscBool);
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestDiagScalePolys(PC, PetscBool);
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestSubcomm(PC, PetscBool);
PETSC_EXTERN PetscErrorCode PCAIRSetRDrop(PC, PetscReal);
PETSC_EXTERN PetscErrorCode PCAIRSetADrop(PC, PetscReal);
PETSC_EXTERN PetscErrorCode PCAIRSetALump(PC, PetscBool);
PETSC_EXTERN PetscErrorCode PCAIRSetReuseSparsity(PC, PetscBool);
PETSC_EXTERN PetscErrorCode PCAIRSetReusePolyCoeffs(PC, PetscBool);
PETSC_EXTERN PetscErrorCode PCAIRSetReuseAmount(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCAIRSetPolyCoeffs(PC, PetscInt, int, PetscReal *, PetscInt, PetscInt);

#endif