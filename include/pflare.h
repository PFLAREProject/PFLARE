#if !defined (PFLARE_C_DEF_H)
#define PFLARE_C_DEF_H

#include "petsc.h"

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
typedef enum {
   AIR_Z_PRODUCT,
   AIR_Z_LAIR,
   AIR_Z_LAIR_SAI,
}  PCAIRZType;
typedef enum {
   CF_PMISR_DDC,
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
PETSC_EXTERN void compute_cf_splitting_c(Mat *, int, double, int, int, int, double, double, IS*, IS*);

/* Define PCPFLAREINV get routines */
PETSC_EXTERN PetscErrorCode PCPFLAREINVGetOrder(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCPFLAREINVGetType(PC, PCPFLAREINVType *);
PETSC_EXTERN PetscErrorCode PCPFLAREINVGetMatrixFree(PC, PetscBool *);
/* Define PCPFLAREINV set routines */
PETSC_EXTERN PetscErrorCode PCPFLAREINVSetOrder(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCPFLAREINVSetType(PC, PCPFLAREINVType);
PETSC_EXTERN PetscErrorCode PCPFLAREINVSetMatrixFree(PC, PetscBool);

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
PETSC_EXTERN PetscErrorCode PCAIRGetMaxDDRatio(PC, PetscReal *);
PETSC_EXTERN PetscErrorCode PCAIRGetDDCFraction(PC, PetscReal *);
PETSC_EXTERN PetscErrorCode PCAIRGetCFSplittingType(PC, CFSplittingType *);
PETSC_EXTERN PetscErrorCode PCAIRGetMaxLubySteps(PC, PetscInt *);
PETSC_EXTERN PetscErrorCode PCAIRGetSmoothType(PC, char *);
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
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestSubcomm(PC, PetscBool *);
PETSC_EXTERN PetscErrorCode PCAIRGetRDrop(PC, PetscReal *);
PETSC_EXTERN PetscErrorCode PCAIRGetADrop(PC, PetscReal *);
PETSC_EXTERN PetscErrorCode PCAIRGetALump(PC, PetscBool *);
PETSC_EXTERN PetscErrorCode PCAIRGetReuseSparsity(PC, PetscBool *);
PETSC_EXTERN PetscErrorCode PCAIRGetReusePolyCoeffs(PC, PetscBool *);
PETSC_EXTERN PetscErrorCode PCAIRGetPolyCoeffs(PC, PetscInt, int, PetscReal **, PetscInt *, PetscInt *);
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
PETSC_EXTERN PetscErrorCode PCAIRSetMaxDDRatio(PC, PetscReal);
PETSC_EXTERN PetscErrorCode PCAIRSetDDCFraction(PC, PetscReal);
PETSC_EXTERN PetscErrorCode PCAIRSetCFSplittingType(PC, CFSplittingType);
PETSC_EXTERN PetscErrorCode PCAIRSetMaxLubySteps(PC, PetscInt);
PETSC_EXTERN PetscErrorCode PCAIRSetSmoothType(PC, const char *);
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
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestSubcomm(PC, PetscBool);
PETSC_EXTERN PetscErrorCode PCAIRSetRDrop(PC, PetscReal);
PETSC_EXTERN PetscErrorCode PCAIRSetADrop(PC, PetscReal);
PETSC_EXTERN PetscErrorCode PCAIRSetALump(PC, PetscBool);
PETSC_EXTERN PetscErrorCode PCAIRSetReuseSparsity(PC, PetscBool);
PETSC_EXTERN PetscErrorCode PCAIRSetReusePolyCoeffs(PC, PetscBool);
PETSC_EXTERN PetscErrorCode PCAIRSetPolyCoeffs(PC, PetscInt, int, PetscReal *, PetscInt, PetscInt);

#endif