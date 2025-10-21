/*
  Definition and registration of the PC AIR type
  Largely just a wrapper around all the fortran 
  using PCShell
*/

// Include the petsc header files
#include <petsc/private/pcimpl.h>
#include "pflare.h"
#include <string.h>

// Defined in C_Fortran_Bindings.F90
PETSC_EXTERN void PCReset_AIR_Shell_c(PC *pc);
PETSC_EXTERN void create_pc_air_data_c(void **pc_air_data);
PETSC_EXTERN void create_pc_air_shell_c(void **pc_air_data, PC *pc);
// Defined in PCAIR_C_Fortran_Bindings.F90 
// External users should use the get/set routines without _c which have 
// PetscErrorCode defined as return type, those routines are defined below this
// e.g., PCAIRGetPrintStatsTimings 
PETSC_EXTERN void PCAIRGetPrintStatsTimings_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetMaxLevels_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetCoarseEqLimit_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetAutoTruncateStartLevel_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetAutoTruncateTol_c(PC *pc, PetscReal *input_real);
PETSC_EXTERN void PCAIRGetNumLevels_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetProcessorAgglom_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetProcessorAgglomRatio_c(PC *pc, PetscReal *input_real);
PETSC_EXTERN void PCAIRGetProcessorAgglomFactor_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetProcessEqLimit_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetSubcomm_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetStrongThreshold_c(PC *pc, PetscReal *input_real);
PETSC_EXTERN void PCAIRGetDDCIts_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetMaxDDRatio_c(PC *pc, PetscReal *input_real);
PETSC_EXTERN void PCAIRGetDDCFraction_c(PC *pc, PetscReal *input_real);
PETSC_EXTERN void PCAIRGetCFSplittingType_c(PC *pc, CFSplittingType *input_int);
PETSC_EXTERN void PCAIRGetMaxLubySteps_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetSmoothType_c(PC *pc, char* input_string);
PETSC_EXTERN void PCAIRGetMatrixFreePolys_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetOnePointClassicalProlong_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetFullSmoothingUpAndDown_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetSymmetric_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetConstrainW_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetConstrainZ_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetImproveWIts_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetImproveZIts_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetStrongRThreshold_c(PC *pc, PetscReal *input_real);
PETSC_EXTERN void PCAIRGetInverseType_c(PC *pc, PCPFLAREINVType *input_int);
PETSC_EXTERN void PCAIRGetCInverseType_c(PC *pc, PCPFLAREINVType *input_int);
PETSC_EXTERN void PCAIRGetZType_c(PC *pc, PCAIRZType *input_int);
PETSC_EXTERN void PCAIRGetPolyOrder_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetLairDistance_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetInverseSparsityOrder_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetCPolyOrder_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetCInverseSparsityOrder_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetCoarsestInverseType_c(PC *pc, PCPFLAREINVType *input_int);
PETSC_EXTERN void PCAIRGetCoarsestPolyOrder_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetCoarsestInverseSparsityOrder_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetCoarsestMatrixFreePolys_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetCoarsestSubcomm_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetRDrop_c(PC *pc, PetscReal *input_real);
PETSC_EXTERN void PCAIRGetADrop_c(PC *pc, PetscReal *input_real);
PETSC_EXTERN void PCAIRGetALump_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetReuseSparsity_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetReusePolyCoeffs_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetPolyCoeffs_c(PC *pc, PetscInt petsc_level, int which_inverse, PetscReal **coeffs_ptr, PetscInt *row_size, PetscInt *col_size);

// Setters
PETSC_EXTERN void PCAIRSetPrintStatsTimings_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetMaxLevels_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetCoarseEqLimit_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetAutoTruncateStartLevel_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetAutoTruncateTol_c(PC *pc, PetscReal input_real);
PETSC_EXTERN void PCAIRSetProcessorAgglom_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetProcessorAgglomRatio_c(PC *pc, PetscReal input_real);
PETSC_EXTERN void PCAIRSetProcessorAgglomFactor_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetProcessEqLimit_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetSubcomm_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetStrongThreshold_c(PC *pc, PetscReal input_real);
PETSC_EXTERN void PCAIRSetDDCIts_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetMaxDDRatio_c(PC *pc, PetscReal input_real);
PETSC_EXTERN void PCAIRSetDDCFraction_c(PC *pc, PetscReal input_real);
PETSC_EXTERN void PCAIRSetCFSplittingType_c(PC *pc, CFSplittingType input_int);
PETSC_EXTERN void PCAIRSetMaxLubySteps_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetSmoothType_c(PC *pc, const char* input_string);
PETSC_EXTERN void PCAIRSetMatrixFreePolys_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetOnePointClassicalProlong_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetFullSmoothingUpAndDown_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetSymmetric_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetConstrainW_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetConstrainZ_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetImproveWIts_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetImproveZIts_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetStrongRThreshold_c(PC *pc, PetscReal input_real);
PETSC_EXTERN void PCAIRSetInverseType_c(PC *pc, PCPFLAREINVType input_int);
PETSC_EXTERN void PCAIRSetZType_c(PC *pc, PCAIRZType input_int);
PETSC_EXTERN void PCAIRSetLairDistance_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetPolyOrder_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetInverseSparsityOrder_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetCInverseType_c(PC *pc, PCPFLAREINVType input_int);
PETSC_EXTERN void PCAIRSetCPolyOrder_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetCInverseSparsityOrder_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetCoarsestInverseType_c(PC *pc, PCPFLAREINVType input_int);
PETSC_EXTERN void PCAIRSetCoarsestPolyOrder_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetCoarsestInverseSparsityOrder_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetCoarsestMatrixFreePolys_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetCoarsestSubcomm_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetRDrop_c(PC *pc, PetscReal input_real);
PETSC_EXTERN void PCAIRSetADrop_c(PC *pc, PetscReal input_real);
PETSC_EXTERN void PCAIRSetALump_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetReuseSparsity_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetReusePolyCoeffs_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetPolyCoeffs_c(PC *pc, PetscInt petsc_level, int which_inverse, PetscReal *coeffs_ptr, PetscInt row_size, PetscInt col_size);

// ~~~~~~~~~~~~~

static PetscErrorCode PCReset_AIR_c(PC pc)
{
   PetscFunctionBegin;

   PC *pc_air_shell = (PC *)pc->data;
   // Call the underlying reset - this won't touch the context 
   // in our pcshell though as pcshell doesn't offer a way to 
   // give a custom reset function
   PetscCall(PCReset(*pc_air_shell));

   // So now we manually reset the underlying data
   PCReset_AIR_Shell_c(pc_air_shell);  

   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_AIR_c(PC pc, Vec x, Vec y)
{
   PetscFunctionBegin;
   PC *pc_air_shell = (PC *)pc->data;

   // Just call the underlying pcshell apply
   PetscCall(PCApply(*pc_air_shell, x, y));
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_AIR_c(PC pc)
{
   PetscFunctionBegin;
   PC *pc_air_shell = (PC *)pc->data;
   // Just call the underlying pcshell destroy, this also 
   // destroys the pc_air_data
   PetscCall(PCDestroy(pc_air_shell));
   // Then destroy the heap pointer
   PetscCall(PetscFree(pc_air_shell));
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

static PetscErrorCode PCSetUp_AIR_c(PC pc)
{
   PetscFunctionBegin;
   PC *pc_air_shell = (PC *)pc->data;

   // The pc_air_shell doesn't have any operators yet 
   // as they are not available yet in pccreate
   // so we have to set them
   PetscCall(PCSetOperators(*pc_air_shell, pc->mat, pc->pmat));

   // Now we should be able to call the pcshell setup
   // that builds the air hierarchy
   PetscCall(PCSetUp(*pc_air_shell));
   PetscFunctionReturn(PETSC_SUCCESS);
}

// Get the underlying PCShell
PETSC_EXTERN PetscErrorCode c_PCAIRGetPCShell(PC *pc, PC *pc_air_shell)
{
   PetscFunctionBegin;
   PC *pc_shell = (PC *)(*pc)->data;
   *pc_air_shell = *pc_shell;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~
// Now all the get/set routines for options
// Most of the explanation are in the comments above the set routines
// ~~~~~~~~~~~~~~~~~~~~~

// Get routines

PETSC_EXTERN PetscErrorCode PCAIRGetPrintStatsTimings(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetPrintStatsTimings_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetMaxLevels(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetMaxLevels_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCoarseEqLimit(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetCoarseEqLimit_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetAutoTruncateStartLevel(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetAutoTruncateStartLevel_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetAutoTruncateTol(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetAutoTruncateTol_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Returns the number of levels in the underlying PCMG
// -1 if the mg is not setup yet
PETSC_EXTERN PetscErrorCode PCAIRGetNumLevels(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetNumLevels_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetProcessorAgglom(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetProcessorAgglom_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetProcessorAgglomRatio(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetProcessorAgglomRatio_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetProcessorAgglomFactor(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetProcessorAgglomFactor_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetProcessEqLimit(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetProcessEqLimit_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetSubcomm(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetSubcomm_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetStrongThreshold(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetStrongThreshold_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetMaxDDRatio(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetMaxDDRatio_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetDDCFraction(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetDDCFraction_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCFSplittingType(PC pc, CFSplittingType *input_int)
{
   PetscFunctionBegin;
   PCAIRGetCFSplittingType_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetDDCIts(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetDDCIts_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetMaxLubySteps(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetMaxLubySteps_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetSmoothType(PC pc, char *input_string)
{
   PetscFunctionBegin;
   PCAIRGetSmoothType_c(&pc, input_string);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetMatrixFreePolys(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetMatrixFreePolys_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetOnePointClassicalProlong(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetOnePointClassicalProlong_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetFullSmoothingUpAndDown(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetFullSmoothingUpAndDown_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetSymmetric(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetSymmetric_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetConstrainW(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetConstrainW_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetConstrainZ(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetConstrainZ_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetImproveWIts(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetImproveWIts_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetImproveZIts(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetImproveZIts_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetStrongRThreshold(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetStrongRThreshold_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetInverseType(PC pc, PCPFLAREINVType *input_int)
{
   PetscFunctionBegin;
   PCAIRGetInverseType_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCInverseType(PC pc, PCPFLAREINVType *input_int)
{
   PetscFunctionBegin;
   PCAIRGetCInverseType_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetZType(PC pc, PCAIRZType *input_int)
{
   PetscFunctionBegin;
   PCAIRGetZType_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetPolyOrder(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetPolyOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetLairDistance(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetLairDistance_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetInverseSparsityOrder(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetInverseSparsityOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCPolyOrder(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetCPolyOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCInverseSparsityOrder(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetCInverseSparsityOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestInverseType(PC pc, PCPFLAREINVType *input_int)
{
   PetscFunctionBegin;
   PCAIRGetCoarsestInverseType_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestPolyOrder(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetCoarsestPolyOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestInverseSparsityOrder(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetCoarsestInverseSparsityOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestMatrixFreePolys(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetCoarsestMatrixFreePolys_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestSubcomm(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetCoarsestSubcomm_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetRDrop(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetRDrop_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetADrop(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetADrop_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetALump(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetALump_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetReuseSparsity(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetReuseSparsity_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetReusePolyCoeffs(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetReusePolyCoeffs_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// This routine returns a pointer to the coefficients in the PCAIR object
// If you want to save/restore them later them you will need to copy them yourself
// the size of the coeff pointer array is also returned 
// This is different to the fortran interface to this routine, which returns a copy
// in an allocatable object (which knows its own size)
PETSC_EXTERN PetscErrorCode PCAIRGetPolyCoeffs(PC pc, PetscInt petsc_level, int which_inverse, PetscReal **coeffs_ptr, PetscInt *row_size, PetscInt *col_size)
{
   PetscFunctionBegin;
   PCAIRGetPolyCoeffs_c(&pc,petsc_level, which_inverse, \
      coeffs_ptr, row_size, col_size);
   PetscFunctionReturn(PETSC_SUCCESS);
}

// Set routines

// Print out stats and timings
// These require some parallel reductions to compute
// so they are off by default
// Default: false
// -pc_air_print_stats_timings
PETSC_EXTERN PetscErrorCode PCAIRSetPrintStatsTimings(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   // No need to reset if this changes
   PCAIRSetPrintStatsTimings_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}

// Maximum number of levels in the multigrid hierarchy
// Default: 300
// -pc_air_max_levels
PETSC_EXTERN PetscErrorCode PCAIRSetMaxLevels(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PetscCall(PCAIRGetMaxLevels(pc, &old_int));
   if (old_int == input_int) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetMaxLevels_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Minimum number of global unknowns on the coarse grid
// Default: 6
// -pc_air_coarse_eq_limit
PETSC_EXTERN PetscErrorCode PCAIRSetCoarseEqLimit(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PetscCall(PCAIRGetCoarseEqLimit(pc, &old_int));
   if (old_int == input_int) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetCoarseEqLimit_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// From this level onwards, build and then evaluate if the coarse grid solver
// is good enough and use that to determine if we should truncate on that level
// Default: -1
// -pc_air_auto_truncate_start_level
PETSC_EXTERN PetscErrorCode PCAIRSetAutoTruncateStartLevel(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PetscCall(PCAIRGetAutoTruncateStartLevel(pc, &old_int));
   if (old_int == input_int) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetAutoTruncateStartLevel_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// What relative tolerance to use to determine if a coarse grid solver is good enough
// Default: 1e-14
// -pc_air_auto_truncate_tol
PETSC_EXTERN PetscErrorCode PCAIRSetAutoTruncateTol(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PetscReal old_real;
   PetscCall(PCAIRGetAutoTruncateTol(pc, &old_real));
   if (old_real == input_real) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetAutoTruncateTol_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Perform processor agglomeration throughout the hierarchy
// This reduces the number of active MPI ranks as we coarsen
// by a factor of processor_agglom_factor, whenever the 
// local to non-local ratio of nnzs is processor_agglom_ratio
// The entire hierarchy stays on comm_world however
// Only happens where necessary, not on every level
// Default: true
// -pc_air_processor_agglom
PETSC_EXTERN PetscErrorCode PCAIRSetProcessorAgglom(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscBool old_bool;
   PetscCall(PCAIRGetProcessorAgglom(pc, &old_bool));
   if (old_bool == input_bool) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetProcessorAgglom_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// The local to nonlocal ratio of nnzs that is used to 
// trigger processor agglomeration on all level
// Default: 2.0
// -pc_air_processor_agglom_ratio
PETSC_EXTERN PetscErrorCode PCAIRSetProcessorAgglomRatio(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PetscReal old_real;
   PetscCall(PCAIRGetProcessorAgglomRatio(pc, &old_real));
   if (old_real == input_real) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetProcessorAgglomRatio_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// What factor to reduce the number of active MPI ranks by
// each time when doing processor agglomeration
// Default: 2
// -pc_air_processor_agglom_factor
PETSC_EXTERN PetscErrorCode PCAIRSetProcessorAgglomFactor(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PetscCall(PCAIRGetProcessorAgglomFactor(pc, &old_int));
   if (old_int == input_int) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetProcessorAgglomFactor_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// If on average there are fewer than this number of equations per rank
// processor agglomeration will be triggered
// Default: 50
// -pc_air_process_eq_limit
PETSC_EXTERN PetscErrorCode PCAIRSetProcessEqLimit(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PetscCall(PCAIRGetProcessEqLimit(pc, &old_int));
   if (old_int == input_int) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetProcessEqLimit_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// If we are doing processor agglomeration, then we have 
// some ranks with no rows
// If computing a gmres polynomial inverse 
// with inverse_type arnoldi or newton, then we can have 
// the reductions occur on a subcomm if we want to reduce the cost
// Default: false
// -pc_air_subcomm
PETSC_EXTERN PetscErrorCode PCAIRSetSubcomm(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   // No need to reset if this changes
   PCAIRSetSubcomm_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// This is used in the CF splitting to define strong dependencies/influences
// Default: 0.5
// -pc_air_strong_threshold
PETSC_EXTERN PetscErrorCode PCAIRSetStrongThreshold(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PetscReal old_real;
   PetscCall(PCAIRGetStrongThreshold(pc, &old_real));
   if (old_real == input_real) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetStrongThreshold_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// How many passes of DDC to do
// Default: 1
// -pc_air_ddc_its
PETSC_EXTERN PetscErrorCode PCAIRSetDDCIts(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PetscCall(PCAIRGetDDCIts(pc, &old_int));
   if (old_int == input_int) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetDDCIts_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// If using CF splitting type pmisr_ddc, do as many DDC iterations as necessary to 
// hit this diagonal dominance ratio. If 0.0 do the number in -pc_air_ddc_its
// Default: 0.0
// -pc_air_max_dd_ratio
PETSC_EXTERN PetscErrorCode PCAIRSetMaxDDRatio(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PetscReal old_real;
   PetscCall(PCAIRGetMaxDDRatio(pc, &old_real));
   if (old_real == input_real) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetMaxDDRatio_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Second pass in the PMISR DDC CF splitting converts 
// this fraction of local F points to C based on diagonal dominance
// Default: 0.1
// -pc_air_ddc_fraction
PETSC_EXTERN PetscErrorCode PCAIRSetDDCFraction(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PetscReal old_real;
   PetscCall(PCAIRGetDDCFraction(pc, &old_real));
   if (old_real == input_real) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));     
   PCAIRSetDDCFraction_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// What CF splitting algorithm to use
// 0 - PMISR DDC
// 1 - PMIS distance 1
// 2 - PMIS distance 2 - uses S^T S + S 
// Default: 0
// -pc_air_cf_splitting_type
PETSC_EXTERN PetscErrorCode PCAIRSetCFSplittingType(PC pc, CFSplittingType input_int)
{
   PetscFunctionBegin;
   CFSplittingType old_int;
   PetscCall(PCAIRGetCFSplittingType(pc, &old_int));
   if (old_int == input_int) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetCFSplittingType_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Maximum number of Luby steps to do in CF splitting
// Negative means do as many as needed (at the cost of a parallel
// reduction everytime we finish a Luby step)
// Default: -1
// -pc_air_max_luby_steps
PETSC_EXTERN PetscErrorCode PCAIRSetMaxLubySteps(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PetscCall(PCAIRGetMaxLubySteps(pc, &old_int));
   if (old_int == input_int) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetMaxLubySteps_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Set the type/its of the smoothing
// Default: ff
// -pc_air_smooth_type
PETSC_EXTERN PetscErrorCode PCAIRSetSmoothType(PC pc, const char* input_string)
{
   PetscFunctionBegin;
   char old_string[PETSC_MAX_PATH_LEN];
   PetscCall(PCAIRGetSmoothType(pc, old_string));
   if (strcmp(input_string, old_string) == 0) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetSmoothType_c(&pc, input_string);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Do we apply our polynomials matrix free when smoothing?
// Default: false
// -pc_air_matrix_free_polys
PETSC_EXTERN PetscErrorCode PCAIRSetMatrixFreePolys(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscBool old_bool;
   PetscCall(PCAIRGetMatrixFreePolys(pc, &old_bool));
   if (old_bool == input_bool) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetMatrixFreePolys_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Do we use a one point injection classical prolongator or an AIR-style prolongator
// Default: true
// -pc_air_one_point_classical_prolong
PETSC_EXTERN PetscErrorCode PCAIRSetOnePointClassicalProlong(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscBool old_bool;
   PetscCall(PCAIRGetOnePointClassicalProlong(pc, &old_bool));
   if (old_bool == input_bool) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetOnePointClassicalProlong_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Do we do full smoothing up and down rather than FF or FFC
// Default: false
// -pc_air_full_smoothing_up_and_down
PETSC_EXTERN PetscErrorCode PCAIRSetFullSmoothingUpAndDown(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscBool old_bool;
   PetscCall(PCAIRGetFullSmoothingUpAndDown(pc, &old_bool));
   if (old_bool == input_bool) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetFullSmoothingUpAndDown_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Do we define our prolongator as R^T?
// Default: false
// -pc_air_symmetric
PETSC_EXTERN PetscErrorCode PCAIRSetSymmetric(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscBool old_bool;
   PetscCall(PCAIRGetSymmetric(pc, &old_bool));
   if (old_bool == input_bool) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetSymmetric_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Use a smoothed version of the near-nullspace vectors when building 
// the prolongator 
// If the operator matrix doesn't have a near-nullspace attached to it
// the constant will be used by default
// You can set near-nullspace vectors with MatSetNearNullSpace
// Default: false
// -pc_air_constrain_w
PETSC_EXTERN PetscErrorCode PCAIRSetConstrainW(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscBool old_bool;
   PetscCall(PCAIRGetConstrainW(pc, &old_bool));
   if (old_bool == input_bool) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetConstrainW_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Use a smoothed version of the near-nullspace vectors when building 
// the restrictor 
// If the operator matrix doesn't have a near-nullspace attached to it
// the constant will be used by default
// You can set near-nullspace vectors with MatSetNearNullSpace
// Default: false
// -pc_air_constrain_z
PETSC_EXTERN PetscErrorCode PCAIRSetConstrainZ(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscBool old_bool;
   PetscCall(PCAIRGetConstrainZ(pc, &old_bool));
   if (old_bool == input_bool) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetConstrainZ_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Maximum number of iterations to do when improving W
// Uses a Richardson and Aff^-1 to precondition
// Default: 0
// -pc_air_improve_w_its
PETSC_EXTERN PetscErrorCode PCAIRSetImproveWIts(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PetscCall(PCAIRGetImproveWIts(pc, &old_int));
   if (old_int == input_int) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetImproveWIts_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Maximum number of iterations to do when improving Z
// Uses a Richardson and the diagonal of Aff^-1 to precondition
// Default: 0
// -pc_air_improve_z_its
PETSC_EXTERN PetscErrorCode PCAIRSetImproveZIts(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PetscCall(PCAIRGetImproveZIts(pc, &old_int));
   if (old_int == input_int) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetImproveZIts_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Strong R threshold to apply dropping prior to computing Z
// This only applies when computing Z, ie if you build a GMRES polynomial approximation
// to Aff^-1, it applies this dropping, then computes Z, then rebuilds 
// an Aff^-1 approximation without the dropping for smoothing
// Default: 0.0
// -pc_air_strong_r_threshold
PETSC_EXTERN PetscErrorCode PCAIRSetStrongRThreshold(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PetscReal old_real;
   PetscCall(PCAIRGetStrongRThreshold(pc, &old_real));
   if (old_real == input_real) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetStrongRThreshold_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// What type of approximation do we use for Aff^-1 
// This is used both for Z (if z_type == AIR_Z_PRODUCT, see below) and for F smoothing
// These are defined by PCPFLAREINVType 
// "power" - PFLAREINV_POWER - GMRES polynomial with the power basis 
// "arnoldi" - PFLAREINV_ARNOLDI - GMRES polynomial with the arnoldi basis 
// "newton" - PFLAREINV_NEWTON - GMRES polynomial with the newton basis with extra roots for stability - can only be used matrix-free atm   
// "newton_no_extra" - PFLAREINV_NEWTON_NO_EXTRA - GMRES polynomial with the newton basis with no extra roots - can only be used matrix-free atm      
// "neumann" - PFLAREINV_NEUMANN - Neumann polynomial
// "sai" - PFLAREINV_SAI - SAI
// "isai" - PFLAREINV_ISAI - Incomplete SAI (ie a restricted additive schwartz)
// "wjacobi" - PFLAREINV_WJACOBI - Weighted Jacobi with weight 3 / ( 4 * || Dff^(-1/2) * Aff * Dff^(-1/2) ||_inf )
// "jacobi" - PFLAREINV_JACOBI - Unweighted Jacobi
// Default: power
// -pc_air_inverse_type
PETSC_EXTERN PetscErrorCode PCAIRSetInverseType(PC pc, PCPFLAREINVType input_int)
{
   PetscFunctionBegin;
   PCPFLAREINVType old_int;
   PetscCall(PCAIRGetInverseType(pc, &old_int));
   if (old_int == input_int) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetInverseType_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// What type of approximation do we use for Acc^-1 
// If unset, this defaults to whatever the F point smoother is atm
// Default: pc_air_inverse_type
// -pc_air_c_inverse_type
PETSC_EXTERN PetscErrorCode PCAIRSetCInverseType(PC pc, PCPFLAREINVType input_int)
{
   PetscFunctionBegin;
   PCPFLAREINVType old_int;
   PetscCall(PCAIRGetCInverseType(pc, &old_int));
   if (old_int == input_int) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetCInverseType_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// What type of approximation do we use for Z?
// "product" - AIR_Z_PRODUCT - Aff^-1 approximation determined by inverse type (above) and then Z computed with matmatmult
// "lair" - AIR_Z_LAIR - lAIR computes Z directly
// "lair_sai" - AIR_Z_LAIR_SAI - SAI version of lAIR computes Z directly
// Default: product
// -pc_air_z_type
PETSC_EXTERN PetscErrorCode PCAIRSetZType(PC pc, PCAIRZType input_int)
{
   PetscFunctionBegin;
   PCAIRZType old_int;
   PetscCall(PCAIRGetZType(pc, &old_int));
   if (old_int == input_int) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetZType_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// If z_type == 1 or 2, this is the distance the grid-transfer operators go out to
// This is so we can have lair out to some distance, and then a different sparsity 
// for our smoothers
// If z_type == 0 this is ignored, and the distance is determined by inverse_sparsity_order + 1
// Default: 2
// -pc_air_lair_distance 
PETSC_EXTERN PetscErrorCode PCAIRSetLairDistance(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PetscCall(PCAIRGetLairDistance(pc, &old_int));
   if (old_int == input_int) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetLairDistance_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// This is the order of polynomial we use in air if inverse_type is 
// power, arnoldi, newton or neumann
// Default: 6
// -pc_air_poly_order
PETSC_EXTERN PetscErrorCode PCAIRSetPolyOrder(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PetscCall(PCAIRGetPolyOrder(pc, &old_int));
   if (old_int == input_int) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetPolyOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// This is the order of sparsity we use if we assemble our approximate inverses
// This (hence also) determines what distance our grid-transfer operators are
// distance = inverse_sparsity_order + 1
// Default: 1
// -pc_air_inverse_sparsity_order
PETSC_EXTERN PetscErrorCode PCAIRSetInverseSparsityOrder(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PetscCall(PCAIRGetInverseSparsityOrder(pc, &old_int));
   if (old_int == input_int) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetInverseSparsityOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// This is the order of polynomial we use in air if inverse_type is 
// power, arnoldi, newton or neumann but on the C points
// If unset, this defaults to whatever the F point smoother is atm
// Default: pc_air_poly_order
// -pc_air_c_poly_order
PETSC_EXTERN PetscErrorCode PCAIRSetCPolyOrder(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PetscCall(PCAIRGetCPolyOrder(pc, &old_int));
   if (old_int == input_int) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetCPolyOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// This is the order of sparsity we use if we assemble our approximate inverses
// but on the C points
// If unset, this defaults to whatever the F point smoother is atm
// Default: pc_air_inverse_sparsity_order
// -pc_air_c_inverse_sparsity_order
PETSC_EXTERN PetscErrorCode PCAIRSetCInverseSparsityOrder(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PetscCall(PCAIRGetCInverseSparsityOrder(pc, &old_int));
   if (old_int == input_int) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetCInverseSparsityOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Coarse grid inverse type (see PCAIRSetInverseType)
// Default: power
// -pc_air_coarsest_inverse_type
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestInverseType(PC pc, PCPFLAREINVType input_int)
{
   PetscFunctionBegin;
   PCPFLAREINVType old_int;
   PetscCall(PCAIRGetCoarsestInverseType(pc, &old_int));
   if (old_int == input_int) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetCoarsestInverseType_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Coarse grid polynomial order (see PCAIRSetPolyOrder)
// Default: 6
// -pc_air_coarsest_poly_order
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestPolyOrder(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PetscCall(PCAIRGetCoarsestPolyOrder(pc, &old_int));
   if (old_int == input_int) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetCoarsestPolyOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Coarse grid polynomial sparsity order (see PCAIRSetInverseSparsityOrder)
// Default: 1
// -pc_air_coarsest_inverse_sparsity_order
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestInverseSparsityOrder(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PetscCall(PCAIRGetCoarsestInverseSparsityOrder(pc, &old_int));
   if (old_int == input_int) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetCoarsestInverseSparsityOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Coarse grid matrix-free application (see PCAIRSetMatrixFreePolys)
// Default: false
// -pc_air_coarsest_matrix_free_polys
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestMatrixFreePolys(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscBool old_bool;
   PetscCall(PCAIRGetCoarsestMatrixFreePolys(pc, &old_bool));
   if (old_bool == input_bool) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetCoarsestMatrixFreePolys_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Coarse grid subcomm (see PCAIRSetSubcomm)
// Default: false
// -pc_air_coarsest_subcomm
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestSubcomm(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   // No need to reset if this changes
   PCAIRSetCoarsestSubcomm_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Relative drop tolerances (inf norm) on R 
// Default: 0.01
// -pc_air_r_drop
PETSC_EXTERN PetscErrorCode PCAIRSetRDrop(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PetscReal old_real;
   PetscCall(PCAIRGetRDrop(pc, &old_real));
   if (old_real == input_real) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetRDrop_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Relative drop tolerances (inf norm) on A 
// Default: 0.001
// -pc_air_a_drop
PETSC_EXTERN PetscErrorCode PCAIRSetADrop(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PetscReal old_real;
   PetscCall(PCAIRGetADrop(pc, &old_real));
   if (old_real == input_real) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetADrop_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Whether to lump in A or drop
// Default: false
// -pc_air_a_lump
PETSC_EXTERN PetscErrorCode PCAIRSetALump(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscBool old_bool;
   PetscCall(PCAIRGetALump(pc, &old_bool));
   if (old_bool == input_bool) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_AIR_c(pc));
   PCAIRSetALump_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
// Whether or not to re-use the existing sparsity when PCSetup is called
// with SAME_NONZERO_PATTERN
// This involves re-using the CF splitting, the symbolic mat-mat mults, 
// the repartitioning, the structure of the matrices with drop tolerances applied, etc
// This will take more memory but 
// will make the setup much cheaper on subsequent calls. If the matrix has 
// changed entries convergence may suffer if the matrix is sufficiently different
// Default: false
// -pc_air_reuse_sparsity
PETSC_EXTERN PetscErrorCode PCAIRSetReuseSparsity(PC pc, PetscBool input_bool)
{  
   PetscFunctionBegin; 
   PCAIRSetReuseSparsity_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS); 
}
// Whether or not to also re-use the gmres polynomial coefficients when 
// reuse_sparsity is set to true
// If the matrix has been changed the reused coefficients won't be correct, 
// and the coefficients are very sensitive to changes in the matrix
// This is really only a useful option if you are regenerating 
// the hierarchy for the exact same matrix where you have stored 
// the gmres polynomial coefficients externally and restore them
// using PCAIRGetPolyCoeffs/PCAIRSetPolyCoeffs
// Default: false
// -pc_air_reuse_poly_coeffs
PETSC_EXTERN PetscErrorCode PCAIRSetReusePolyCoeffs(PC pc, PetscBool input_bool)
{   
   PetscFunctionBegin;
   PCAIRSetReusePolyCoeffs_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS); 
}
// This routine sets the polynomial coefficients in the PCAIR object
// row_size and col_size are the size of the coeffs_ptr array
PETSC_EXTERN PetscErrorCode PCAIRSetPolyCoeffs(PC pc, PetscInt petsc_level, int which_inverse, PetscReal *coeffs_ptr, PetscInt row_size, PetscInt col_size)
{
   PetscFunctionBegin;
   PCAIRSetPolyCoeffs_c(&pc,petsc_level, which_inverse, \
      coeffs_ptr, row_size, col_size);
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~

static PetscErrorCode PCSetFromOptions_AIR_c(PC pc, PetscOptionItems PetscOptionsObject)
{
   PetscFunctionBegin;
   
   PetscBool    flg, old_flag;
   PetscInt input_int, old_int;
   PetscReal input_real, old_real;
   PCPFLAREINVType old_type, type;
   PCAIRZType old_z_type, z_type;
   CFSplittingType old_cf_type, cf_type;
   char old_string[PETSC_MAX_PATH_LEN] ,input_string[PETSC_MAX_PATH_LEN];

   PetscOptionsHeadBegin(PetscOptionsObject, "PCAIR options");
   // ~~~~
   PetscCall(PCAIRGetPrintStatsTimings(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_print_stats_timings", "Print statistics and timings", "PCAIRSetPrintStatsTimings", old_flag, &flg, NULL));
   PetscCall(PCAIRSetPrintStatsTimings(pc, flg));
   // ~~~~
   PetscCall(PCAIRGetSubcomm(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_subcomm", "Computes polynomial coefficients on subcomm", "PCAIRSetSubcomm", old_flag, &flg, NULL));
   PetscCall(PCAIRSetSubcomm(pc, flg));
   // ~~~~
   PetscCall(PCAIRGetProcessorAgglom(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_processor_agglom", "Processor agglomeration", "PCAIRSetProcessorAgglom", old_flag, &flg, NULL));
   PetscCall(PCAIRSetProcessorAgglom(pc, flg));
   // ~~~~
   PetscCall(PCAIRGetMatrixFreePolys(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_matrix_free_polys", "Applies polynomial smoothers matrix-free", "PCAIRSetMatrixFreePolys", old_flag, &flg, NULL));
   PetscCall(PCAIRSetMatrixFreePolys(pc, flg));
   // ~~~~   
   PetscCall(PCAIRGetOnePointClassicalProlong(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_one_point_classical_prolong", "One-point classical prolongator", "PCAIRSetOnePointClassicalProlong", old_flag, &flg, NULL));
   PetscCall(PCAIRSetOnePointClassicalProlong(pc, flg));
   // ~~~~  
   PetscCall(PCAIRGetFullSmoothingUpAndDown(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_full_smoothing_up_and_down", "Full up and down smoothing", "PCAIRSetFullSmoothingUpAndDown", old_flag, &flg, NULL));
   PetscCall(PCAIRSetFullSmoothingUpAndDown(pc, flg));
   // ~~~~  
   PetscCall(PCAIRGetSymmetric(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_symmetric", "Use symmetric grid-transfer operators", "PCAIRSetSymmetric", old_flag, &flg, NULL));
   PetscCall(PCAIRSetSymmetric(pc, flg));
   // ~~~~
   PetscCall(PCAIRGetConstrainW(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_constrain_w", "Use constraints on prolongator", "PCAIRSetConstrainW", old_flag, &flg, NULL));
   PetscCall(PCAIRSetConstrainW(pc, flg));
   // ~~~~
   PetscCall(PCAIRGetConstrainZ(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_constrain_z", "Use constraints on restrictor", "PCAIRSetConstrainZ", old_flag, &flg, NULL));
   PetscCall(PCAIRSetConstrainZ(pc, flg));
   // ~~~~  
   PetscCall(PCAIRGetCoarsestMatrixFreePolys(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_coarsest_matrix_free_polys", "Applies polynomial coarse grid solver matrix-free", "PCAIRSetCoarsestMatrixFreePolys", old_flag, &flg, NULL));
   PetscCall(PCAIRSetCoarsestMatrixFreePolys(pc, flg));
   // ~~~~  
   PetscCall(PCAIRGetCoarsestSubcomm(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_coarsest_subcomm", "Computes polynomial coefficients on the coarse grid on subcomm", "PCAIRGetCoarsestSubcomm", old_flag, &flg, NULL));
   PetscCall(PCAIRSetCoarsestSubcomm(pc, flg));
   // ~~~~ 
   PetscCall(PCAIRGetALump(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_a_lump", "Uses lumping on A", "PCAIRSetALump", old_flag, &flg, NULL));
   PetscCall(PCAIRSetALump(pc, flg));
   // ~~~~ 
   PetscCall(PCAIRGetReuseSparsity(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_reuse_sparsity", "Reuses sparsity during setup", "PCAIRSetReuseSparsity", old_flag, &flg, NULL));
   PetscCall(PCAIRSetReuseSparsity(pc, flg));
   // ~~~~
   PetscCall(PCAIRGetReusePolyCoeffs(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_reuse_poly_coeffs", "Reuses gmres polynomial coefficients during setup", "PCAIRSetReusePolyCoeffs", old_flag, &flg, NULL));
   PetscCall(PCAIRSetReusePolyCoeffs(pc, flg));   
   // ~~~~
   PetscCall(PCAIRGetProcessorAgglomRatio(pc, &old_real));
   input_real = old_real;
   PetscCall(PetscOptionsReal("-pc_air_processor_agglom_ratio", "Ratio to trigger processor agglomeration", "PCAIRSetProcessorAgglomRatio", old_real, &input_real, NULL));
   PetscCall(PCAIRSetProcessorAgglomRatio(pc, input_real)); 
   // ~~~~
   PetscCall(PCAIRGetAutoTruncateTol(pc, &old_real));
   input_real = old_real;
   PetscCall(PetscOptionsReal("-pc_air_auto_truncate_tol", "Tolerance to use with auto truncation", "PCAIRSetAutoTruncateTol", old_real, &input_real, NULL));
   PetscCall(PCAIRSetAutoTruncateTol(pc, input_real));
   // ~~~~
   PetscCall(PCAIRGetStrongThreshold(pc, &old_real));
   input_real = old_real;
   PetscCall(PetscOptionsReal("-pc_air_strong_threshold", "Strong threshold for CF splitting", "PCAIRSetStrongThreshold", old_real, &input_real, NULL));
   PetscCall(PCAIRSetStrongThreshold(pc, input_real));
   // ~~~~ 
   PetscCall(PCAIRGetMaxDDRatio(pc, &old_real));
   input_real = old_real;
   PetscCall(PetscOptionsReal("-pc_air_max_dd_ratio", "Max DDC ratio for CF splitting", "PCAIRGetMaxDDRatio", old_real, &input_real, NULL));
   PetscCall(PCAIRSetMaxDDRatio(pc, input_real));
   // ~~~~ 
   PetscCall(PCAIRGetDDCFraction(pc, &old_real));
   input_real = old_real;
   PetscCall(PetscOptionsReal("-pc_air_ddc_fraction", "DDC fraction for CF splitting", "PCAIRGetDDCFraction", old_real, &input_real, NULL));
   PetscCall(PCAIRSetDDCFraction(pc, input_real));
   // ~~~~
   PetscCall(PCAIRGetStrongRThreshold(pc, &old_real));
   input_real = old_real;
   PetscCall(PetscOptionsReal("-pc_air_strong_r_threshold", "Strong R threshold for grid-transfer operators", "PCAIRSetStrongRThreshold", old_real, &input_real, NULL));
   PetscCall(PCAIRSetStrongRThreshold(pc, input_real));
   // ~~~~ 
   PetscCall(PCAIRGetRDrop(pc, &old_real));
   input_real = old_real;
   PetscCall(PetscOptionsReal("-pc_air_r_drop", "Drop tolerance for R", "PCAIRSetRDrop", old_real, &input_real, NULL));
   PetscCall(PCAIRSetRDrop(pc, input_real));
   // ~~~~  
   PetscCall(PCAIRGetADrop(pc, &old_real));
   input_real = old_real;
   PetscCall(PetscOptionsReal("-pc_air_a_drop", "Drop tolerance for A", "PCAIRSetADrop", old_real, &input_real, NULL));
   PetscCall(PCAIRSetADrop(pc, input_real));
   // ~~~~  
   const char *const CFSplittingTypes[] = {"PMISR_DDC", "PMIS", "PMIS_DIST2", "AGG", "PMIS_AGG", "CFSplittingType", "CF_", NULL};
   PetscCall(PCAIRGetCFSplittingType(pc, &old_cf_type));
   cf_type = old_cf_type;
   PetscCall(PetscOptionsEnum("-pc_air_cf_splitting_type", "CF splitting algorithm", "PCAIRSetCFSplittingType", CFSplittingTypes, (PetscEnum)old_cf_type, (PetscEnum *)&cf_type, &flg));
   PetscCall(PCAIRSetCFSplittingType(pc, cf_type));
   // ~~~~ 
   PetscCall(PCAIRGetDDCIts(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_ddc_its", "DDC iterations for CF splitting", "PCAIRGetDDCIts", old_int, &input_int, NULL));
   PetscCall(PCAIRSetDDCIts(pc, input_int));
   // ~~~~   
   PetscCall(PCAIRGetMaxLubySteps(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_max_luby_steps", "Max Luby steps in CF splitting algorithm", "PCAIRSetMaxLubySteps", old_int, &input_int, NULL));
   PetscCall(PCAIRSetMaxLubySteps(pc, input_int));
   // ~~~~   
   PetscCall(PCAIRGetImproveWIts(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_improve_w_its", "Number of iterations to improve W", "PCAIRSetImproveWIts", old_int, &input_int, NULL));
   PetscCall(PCAIRSetImproveWIts(pc, input_int));
   // ~~~~   
   PetscCall(PCAIRGetImproveZIts(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_improve_z_its", "Number of iterations to improve Z", "PCAIRSetImproveZIts", old_int, &input_int, NULL));
   PetscCall(PCAIRSetImproveZIts(pc, input_int));
   // ~~~~ 
   PetscCall(PCAIRGetSmoothType(pc, old_string));
   strcpy(input_string, old_string);
   PetscCall(PetscOptionsString("-pc_air_smooth_type", "The smoothing types/its", "PCAIRSetSmoothType", input_string, input_string, sizeof(input_string), &flg));
   PetscCall(PCAIRSetSmoothType(pc, input_string));
   // ~~~~    
   PetscCall(PCAIRGetMaxLevels(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_max_levels", "Maximum number of levels", "PCAIRSetMaxLevels", old_int, &input_int, NULL));
   PetscCall(PCAIRSetMaxLevels(pc, input_int));
   // ~~~~    
   PetscCall(PCAIRGetCoarseEqLimit(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_coarse_eq_limit", "Minimum number of global unknowns on the coarse grid", "PCAIRSetCoarseEqLimit", old_int, &input_int, NULL));
   PetscCall(PCAIRSetCoarseEqLimit(pc, input_int));
   // ~~~~    
   PetscCall(PCAIRGetAutoTruncateStartLevel(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_auto_truncate_start_level", "Use auto truncation from this level", "PCAIRSetAutoTruncateStartLevel", old_int, &input_int, NULL));
   PetscCall(PCAIRSetAutoTruncateStartLevel(pc, input_int));
   // ~~~~ 
   const char *const PCPFLAREINVTypes[] = {"POWER", "ARNOLDI", "NEWTON", "NEWTON_NO_EXTRA", "NEUMANN", "SAI", "ISAI", "WJACOBI", "JACOBI", "PCPFLAREINVType", "PFLAREINV_", NULL};
   PetscCall(PCAIRGetInverseType(pc, &old_type));
   type = old_type;
   PetscCall(PetscOptionsEnum("-pc_air_inverse_type", "Inverse type", "PCPFLAREINVSetType", PCPFLAREINVTypes, (PetscEnum)old_type, (PetscEnum *)&type, &flg));
   PetscCall(PCAIRSetInverseType(pc, type));
   // ~~~~ 
   // Defaults to whatever the F point smoother is atm
   PetscCall(PCAIRGetInverseType(pc, &old_type));
   type = old_type;
   PetscCall(PetscOptionsEnum("-pc_air_c_inverse_type", "C point inverse type", "PCPFLAREINVSetType", PCPFLAREINVTypes, (PetscEnum)old_type, (PetscEnum *)&type, &flg));
   PetscCall(PCAIRSetCInverseType(pc, type));
   // ~~~~
   const char *const PCAIRZTypes[] = {"PRODUCT", "LAIR", "LAIR_SAI", "PCAIRZType", "AIR_Z_", NULL};
   PetscCall(PCAIRGetZType(pc, &old_z_type));
   z_type = old_z_type;
   PetscCall(PetscOptionsEnum("-pc_air_z_type", "Z type", "PCAIRSetZType", PCAIRZTypes, (PetscEnum)old_z_type, (PetscEnum *)&z_type, &flg));
   PetscCall(PCAIRSetZType(pc, z_type));
   // ~~~~ 
   PetscCall(PCAIRGetLairDistance(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_lair_distance", "lAIR distance", "PCAIRSetLairDistance", old_int, &input_int, NULL));
   PetscCall(PCAIRSetLairDistance(pc, input_int));   
   // ~~~~ 
   PetscCall(PCAIRGetPolyOrder(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_poly_order", "Polynomial order", "PCAIRSetPolyOrder", old_int, &input_int, NULL));
   PetscCall(PCAIRSetPolyOrder(pc, input_int));
   // ~~~~ 
   PetscCall(PCAIRGetInverseSparsityOrder(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_inverse_sparsity_order", "Inverse sparsity order", "PCAIRSetInverseSparsityOrder", old_int, &input_int, NULL));
   PetscCall(PCAIRSetInverseSparsityOrder(pc, input_int));
   // ~~~~ 
   // Defaults to whatever the F point smoother is atm
   PetscCall(PCAIRGetPolyOrder(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_c_poly_order", "C point polynomial order", "PCAIRSetCPolyOrder", old_int, &input_int, NULL));
   PetscCall(PCAIRSetCPolyOrder(pc, input_int));
   // ~~~~ 
   // Defaults to whatever the F point smoother is atm
   PetscCall(PCAIRGetInverseSparsityOrder(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_c_inverse_sparsity_order", "C point inverse sparsity order", "PCAIRSetCInverseSparsityOrder", old_int, &input_int, NULL));
   PetscCall(PCAIRSetCInverseSparsityOrder(pc, input_int));
   // ~~~~ 
   PetscCall(PCAIRGetCoarsestInverseType(pc, &old_type));
   type = old_type;
   PetscCall(PetscOptionsEnum("-pc_air_coarsest_inverse_type", "Inverse type on the coarse grid", "PCPFLAREINVSetType", PCPFLAREINVTypes, (PetscEnum)old_type, (PetscEnum *)&type, &flg));
   PetscCall(PCAIRSetCoarsestInverseType(pc, type));
   // ~~~~ 
   PetscCall(PCAIRGetCoarsestPolyOrder(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_coarsest_poly_order", "Polynomial order on the coarse grid", "PCAIRSetCoarsestPolyOrder", old_int, &input_int, NULL));
   PetscCall(PCAIRSetCoarsestPolyOrder(pc, input_int));
   // ~~~~ 
   PetscCall(PCAIRGetCoarsestInverseSparsityOrder(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_coarsest_inverse_sparsity_order", "Inverse sparsity order on the coarse grid", "PCAIRSetCoarsestInverseSparsityOrder", old_int, &input_int, NULL));
   PetscCall(PCAIRSetCoarsestInverseSparsityOrder(pc, input_int));
   // ~~~~ 
   PetscCall(PCAIRGetProcessorAgglomFactor(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_processor_agglom_factor", "Factor to reduce MPI ranks by", "PCAIRSetProcessorAgglomFactor", old_int, &input_int, NULL));
   PetscCall(PCAIRSetProcessorAgglomFactor(pc, input_int));
   // ~~~~        
   PetscCall(PCAIRGetProcessEqLimit(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_process_eq_limit", "Trigger process agglomeration if fewer eqs/core", "PCAIRSetProcessEqLimit", old_int, &input_int, NULL));
   PetscCall(PCAIRSetProcessEqLimit(pc, input_int));
   // ~~~~                                                       

   PetscOptionsHeadEnd();
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~

static PetscErrorCode PCView_AIR_c(PC pc, PetscViewer viewer)
{
   PetscFunctionBegin;
   
   PC *pc_air_shell = (PC *)pc->data;

   PetscInt input_int, input_int_two, input_int_three, input_int_four;
   PetscBool flg, flg_f_smooth, flg_c_smooth;
   PetscReal input_real, input_real_two, input_real_three;
   PCPFLAREINVType input_type;
   PCAIRZType z_type;
   CFSplittingType cf_type;
   char input_string[PETSC_MAX_PATH_LEN];

   // Print out details
   PetscBool  iascii;
   PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));

   if (iascii) {

      PetscCall(PCAIRGetMaxLevels(pc, &input_int));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Max number of levels=%" PetscInt_FMT " \n", input_int));
      PetscCall(PCAIRGetCoarseEqLimit(pc, &input_int));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Coarse eq limit=%" PetscInt_FMT " \n", input_int));

      PetscCall(PCAIRGetAutoTruncateStartLevel(pc, &input_int));
      PetscCall(PCAIRGetAutoTruncateTol(pc, &input_real));
      if (input_int != -1)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  Auto truncate start level=%" PetscInt_FMT ", with tolerance %.2e \n", input_int, input_real));
      }

      PetscCall(PCAIRGetRDrop(pc, &input_real));
      PetscCall(PCAIRGetADrop(pc, &input_real_two));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  A drop tolerance=%f, R drop tolerance %f \n", input_real_two, input_real));
      PetscCall(PCAIRGetALump(pc, &flg));
      if (flg) PetscCall(PetscViewerASCIIPrintf(viewer, "  Lumping \n"));
      PetscCall(PCAIRGetReuseSparsity(pc, &flg));
      if (flg) PetscCall(PetscViewerASCIIPrintf(viewer, "  Reusing sparsity during setup \n"));
      PetscCall(PCAIRGetReusePolyCoeffs(pc, &flg));
      if (flg) PetscCall(PetscViewerASCIIPrintf(viewer, "  Reusing gmres polynomial coefficients during setup \n"));

      PetscCall(PCAIRGetProcessorAgglom(pc, &flg));
      PetscCall(PCAIRGetProcessorAgglomRatio(pc, &input_real));
      PetscCall(PCAIRGetProcessorAgglomFactor(pc, &input_int));
      PetscCall(PCAIRGetProcessEqLimit(pc, &input_int_two));
      if (flg) PetscCall(PetscViewerASCIIPrintf(viewer, "  Processor agglomeration with factor=%" PetscInt_FMT ", ratio %f and eq limit=%" PetscInt_FMT " \n", input_int, input_real, input_int_two));
      PetscCall(PCAIRGetSubcomm(pc, &flg));
      if (flg) PetscCall(PetscViewerASCIIPrintf(viewer, "  Polynomial coefficients calculated on subcomm \n"));
      
      PetscCall(PCAIRGetCFSplittingType(pc, &cf_type));
      PetscCall(PCAIRGetStrongThreshold(pc, &input_real));
      PetscCall(PCAIRGetDDCIts(pc, &input_int_three));
      PetscCall(PCAIRGetMaxDDRatio(pc, &input_real_three));
      PetscCall(PCAIRGetDDCFraction(pc, &input_real_two));
      PetscCall(PCAIRGetMaxLubySteps(pc, &input_int_two));
      if (cf_type == CF_PMISR_DDC)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  CF splitting algorithm=PMISR_DDC \n"));
         if (input_real_three == 0.0)
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "    %" PetscInt_FMT " Luby steps \n      Strong threshold=%f, DDC its=%" PetscInt_FMT ", DDC fraction=%f \n", \
                  input_int_two, input_real, input_int_three, input_real_two));
         }
         else
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "    %" PetscInt_FMT " Luby steps \n      Strong threshold=%f, Max DD Ratio=%f, DDC fraction=%f \n", \
                  input_int_two, input_real, input_real_three, input_real_two));
         }     
      }
      else if (cf_type == CF_PMIS)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  CF splitting algorithm=PMIS \n"));
         PetscCall(PetscViewerASCIIPrintf(viewer, "    %" PetscInt_FMT " Luby steps \n      Strong threshold=%f \n", \
                  input_int_two, input_real));          
      }
      else if (cf_type == CF_PMIS_DIST2)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  CF splitting algorithm=PMIS_DIST2 \n"));
         PetscCall(PetscViewerASCIIPrintf(viewer, "    %" PetscInt_FMT " Luby steps \n      Strong threshold=%f \n", \
                  input_int_two, input_real));
      }
      else if (cf_type == CF_AGG)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  CF splitting algorithm=AGG \n"));
         PetscCall(PetscViewerASCIIPrintf(viewer, "    Strong threshold=%f \n", \
                  input_real));
      }

      PetscCall(PCAIRGetFullSmoothingUpAndDown(pc, &flg));
      if (flg) 
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  Full smoothing up & down \n"));
         PetscCall(PCAIRGetInverseType(pc, &input_type));
         PetscCall(PCAIRGetPolyOrder(pc, &input_int_two));
         PetscCall(PCAIRGetInverseSparsityOrder(pc, &input_int_three));
         PetscCall(PCAIRGetMatrixFreePolys(pc, &flg));

         // What type of inverse
         if (input_type == PFLAREINV_POWER)
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, power basis, order %" PetscInt_FMT " \n", input_int_two));
         }
         else if (input_type == PFLAREINV_ARNOLDI)
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, arnoldi basis, order %" PetscInt_FMT " \n", input_int_two));
         }
         else if (input_type == PFLAREINV_NEWTON)
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, newton basis with extra roots, order %" PetscInt_FMT " \n", input_int_two));
         }
         else if (input_type == PFLAREINV_NEWTON_NO_EXTRA)
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, newton basis without extra roots, order %" PetscInt_FMT " \n", input_int_two));
         }
         else if (input_type == PFLAREINV_SAI)
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "    SAI \n"));
         }
         else if (input_type == PFLAREINV_ISAI)
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "    ISAI \n"));
         }
         else if (input_type == PFLAREINV_NEUMANN)
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "    Neumann polynomial, order %" PetscInt_FMT " \n", input_int_two));
         }
         else if (input_type == PFLAREINV_WJACOBI)
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "    Weighted Jacobi \n"));
         }
         else if (input_type == PFLAREINV_JACOBI)
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "    Unweighted Jacobi \n"));
         }
         if (input_type != PFLAREINV_WJACOBI && input_type != PFLAREINV_JACOBI)
         {
            // If matrix-free or not
            if (flg)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "      matrix-free inverse \n"));
            }
            else
            {
               // Only print out if the sparsity is less than the poly order
               if (input_int_three < input_int_two)
               {
                  PetscCall(PetscViewerASCIIPrintf(viewer, "      assembled inverse, sparsity order %" PetscInt_FMT "\n", input_int_three));
               }
               else
               {
                  PetscCall(PetscViewerASCIIPrintf(viewer, "      assembled inverse\n"));
               }                 
            }
         }             
      }
      else
      {
         PetscCall(PCAIRGetSmoothType(pc, input_string));
         // Set flags if 'f' or 'c' present in input_string
         flg_f_smooth = PETSC_FALSE;
         flg_c_smooth = PETSC_FALSE;
         for (int i = 0; input_string[i] != '\0'; i++) {
            if (input_string[i] == 'f' || input_string[i] == 'F') flg_f_smooth = PETSC_TRUE;
            if (input_string[i] == 'c' || input_string[i] == 'C') flg_c_smooth = PETSC_TRUE;
         }
         PetscCall(PetscViewerASCIIPrintf(viewer, "  Up smoothing of type=%s \n", input_string));

         if (flg_f_smooth)
         {
            PetscCall(PCAIRGetInverseType(pc, &input_type));
            PetscCall(PCAIRGetPolyOrder(pc, &input_int_two));
            PetscCall(PCAIRGetInverseSparsityOrder(pc, &input_int_three));
            PetscCall(PCAIRGetMatrixFreePolys(pc, &flg));

            // What type of inverse
            if (input_type == PFLAREINV_POWER)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    F smooth: GMRES polynomial, power basis, order %" PetscInt_FMT " \n", input_int_two));
            }
            else if (input_type == PFLAREINV_ARNOLDI)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    F smooth: GMRES polynomial, arnoldi basis, order %" PetscInt_FMT " \n", input_int_two));
            }
            else if (input_type == PFLAREINV_NEWTON)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    F smooth: GMRES polynomial, newton basis with extra roots, order %" PetscInt_FMT " \n", input_int_two));
            }
            else if (input_type == PFLAREINV_NEWTON_NO_EXTRA)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    F smooth: GMRES polynomial, newton basis without extra roots, order %" PetscInt_FMT " \n", input_int_two));
            }
            else if (input_type == PFLAREINV_SAI)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    F smooth: SAI \n"));      
            }
            else if (input_type == PFLAREINV_ISAI)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    F smooth: ISAI \n"));
            }
            else if (input_type == PFLAREINV_NEUMANN)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    F smooth: Neumann polynomial, order %" PetscInt_FMT " \n", input_int_two));
            }
            else if (input_type == PFLAREINV_WJACOBI)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    F smooth: Weighted Jacobi \n"));      
            }
            else if (input_type == PFLAREINV_JACOBI)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    F smooth: Unweighted Jacobi \n"));      
            }                  
            if (input_type != PFLAREINV_WJACOBI && input_type != PFLAREINV_JACOBI)
            {
               // If matrix-free or not
               if (flg)
               {
                  PetscCall(PetscViewerASCIIPrintf(viewer, "      matrix-free inverse \n"));
               }
               else
               {
                  // Only print out if the sparsity is less than the poly order
                  if (input_int_three < input_int_two)
                  {
                     PetscCall(PetscViewerASCIIPrintf(viewer, "      assembled inverse, sparsity order %" PetscInt_FMT "\n", input_int_three));
                  }
                  else
                  {
                     PetscCall(PetscViewerASCIIPrintf(viewer, "      assembled inverse\n"));
                  }
               }
            }  
         }

         // If C smoothing
         if (flg_c_smooth) {

            PetscCall(PCAIRGetCInverseType(pc, &input_type));
            PetscCall(PCAIRGetCPolyOrder(pc, &input_int_two));
            PetscCall(PCAIRGetCInverseSparsityOrder(pc, &input_int_three));
            PetscCall(PCAIRGetMatrixFreePolys(pc, &flg));

            // What type of inverse
            if (input_type == PFLAREINV_POWER)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    C smooth: GMRES polynomial, power basis, order %" PetscInt_FMT " \n", input_int_two));
            }
            else if (input_type == PFLAREINV_ARNOLDI)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    C smooth: GMRES polynomial, arnoldi basis, order %" PetscInt_FMT " \n", input_int_two));
            }
            else if (input_type == PFLAREINV_NEWTON)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    C smooth: GMRES polynomial, newton basis with extra roots, order %" PetscInt_FMT " \n", input_int_two));
            }
            else if (input_type == PFLAREINV_NEWTON_NO_EXTRA)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    C smooth: GMRES polynomial, newton basis without extra roots, order %" PetscInt_FMT " \n", input_int_two));
            }
            else if (input_type == PFLAREINV_SAI)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    C smooth: SAI \n"));
            }
            else if (input_type == PFLAREINV_ISAI)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    C smooth: ISAI \n"));
            }
            else if (input_type == PFLAREINV_NEUMANN)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    C smooth: Neumann polynomial, order %" PetscInt_FMT " \n", input_int_two));
            }     
            else if (input_type == PFLAREINV_WJACOBI)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    C smooth: Weighted Jacobi \n"));
            }
            else if (input_type == PFLAREINV_JACOBI)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    C smooth: Unweighted Jacobi \n"));
            }
            if (input_type != PFLAREINV_WJACOBI && input_type != PFLAREINV_JACOBI)
            {
               // If matrix-free or not
               if (flg)
               {
                  PetscCall(PetscViewerASCIIPrintf(viewer, "      matrix-free inverse \n"));
               }
               else
               {
                  // Only print out if the sparsity is less than the poly order
                  if (input_int_three < input_int_two)
                  {
                     PetscCall(PetscViewerASCIIPrintf(viewer, "      assembled inverse, sparsity order %" PetscInt_FMT "\n", input_int_three));
                  }
                  else
                  {
                     PetscCall(PetscViewerASCIIPrintf(viewer, "      assembled inverse\n"));
                  }
               }
            }  
         }           
      }        

      PetscCall(PetscViewerASCIIPrintf(viewer, "  Grid transfer operators: \n")); 
      PetscCall(PCAIRGetZType(pc, &z_type));
      PetscCall(PCAIRGetLairDistance(pc, &input_int_two));
      PetscCall(PCAIRGetImproveWIts(pc, &input_int_three));
      PetscCall(PCAIRGetImproveZIts(pc, &input_int_four));
      if (z_type == AIR_Z_PRODUCT)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    Mat-Mat product used to form Z \n"));      
      }
      else if (z_type == AIR_Z_LAIR)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    lAIR Z, distance %" PetscInt_FMT " \n", input_int_two));            
      }
      else
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    lAIR SAI Z, distance %" PetscInt_FMT " \n", input_int_two));
      }
      PetscCall(PCAIRGetStrongRThreshold(pc, &input_real));
      PetscCall(PetscViewerASCIIPrintf(viewer, "    Strong R threshold=%f \n", input_real));
      if (input_int_three > 0)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    Improve W %" PetscInt_FMT " iterations \n", input_int_three));
      }
      if (input_int_four > 0)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    Improve Z %" PetscInt_FMT " iterations \n", input_int_four));      
      }
      PetscCall(PCAIRGetConstrainZ(pc, &flg));
      if (flg) 
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    Constraints applied to restrictor \n"));      
      }     

      PetscCall(PCAIRGetSymmetric(pc, &flg));
      if (flg) 
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    Approximate ideal prolongator \n"));      
         PetscCall(PetscViewerASCIIPrintf(viewer, "      Symmetric - transpose of restrictor \n"));      
      }        
      else
      {
         PetscCall(PCAIRGetOnePointClassicalProlong(pc, &flg));
         if (flg) 
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "    One point classical prolongator \n"));      
         }
         else
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "    Approximate ideal prolongator \n"));      
         }       
      }

      PetscCall(PCAIRGetConstrainW(pc, &flg));
      if (flg) 
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "      Constraints applied to prolongator \n"));      
      }      

      PetscCall(PetscViewerASCIIPrintf(viewer, "  Coarse grid solver: \n"));
      PetscCall(PCAIRGetCoarsestInverseType(pc, &input_type));
      PetscCall(PCAIRGetCoarsestPolyOrder(pc, &input_int_two));
      PetscCall(PCAIRGetCoarsestInverseSparsityOrder(pc, &input_int_three));
      PetscCall(PCAIRGetCoarsestMatrixFreePolys(pc, &flg));

      // What type of inverse
      if (input_type == PFLAREINV_POWER)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, power basis, order %" PetscInt_FMT " \n", input_int_two));
      }
      else if (input_type == PFLAREINV_ARNOLDI)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, arnoldi basis, order %" PetscInt_FMT " \n", input_int_two));
      }
      else if (input_type == PFLAREINV_NEWTON)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, newton basis with extra roots, order %" PetscInt_FMT " \n", input_int_two));      
      }
      else if (input_type == PFLAREINV_NEWTON_NO_EXTRA)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, newton basis without extra roots, order %" PetscInt_FMT " \n", input_int_two));
      }
      else if (input_type == PFLAREINV_SAI)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    SAI \n"));
      }
      else if (input_type == PFLAREINV_ISAI)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    ISAI \n"));
      }
      else if (input_type == PFLAREINV_NEUMANN)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    Neumann polynomial, order %" PetscInt_FMT " \n", input_int_two));
      }
      else if (input_type == PFLAREINV_WJACOBI)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    Weighted Jacobi \n"));
      }
      else if (input_type == PFLAREINV_JACOBI)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    Unweighted Jacobi \n"));
      }
      if (input_type != PFLAREINV_WJACOBI && input_type != PFLAREINV_JACOBI)
      {
         // If matrix-free or not
         if (flg)
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "    matrix-free inverse \n"));
         }
         else
         {
            // Only print out if the sparsity is less than the poly order
            if (input_int_three < input_int_two)
            {            
               PetscCall(PetscViewerASCIIPrintf(viewer, "    assembled inverse, sparsity order %" PetscInt_FMT "\n", input_int_three));
            }
            else
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    assembled inverse\n"));
            }
         }
      }      
     
      PetscCall(PCAIRGetCoarsestSubcomm(pc, &flg));
      if (flg) PetscCall(PetscViewerASCIIPrintf(viewer, "   Polynomial coefficients calculated on subcomm \n"));

      PetscCall(PetscViewerASCIIPrintf(viewer, "The underlying PCMG: \n"));
      // Call the underlying pcshell view
      PetscCall(PCView(*pc_air_shell, viewer));
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~

// Creates the structure we need for this PC
PETSC_EXTERN PetscErrorCode PCCreate_AIR(PC pc)
{
   PetscFunctionBegin;

   // Now we call petsc fortran routines from this PC
   // so we have to have made sure this is called
   PetscCall(PetscInitializeFortran());
      
   // This is annoying, as our PCAIR type is just a wrapper
   // around a PCShell that implements everything
   // I tried to just write a PCAIR that called (basically) the 
   // same routines as the PCShell, but I could not get a reliable
   // way to give a void* representing the pc_air_data object from fortran
   // into the pc->data pointer (given name mangling etc)
   // So instead that is just defined as the context in a pcshell
   // and PCShellSetContext/PCShellGetContext handles everything
   void *pc_air_data;
   PC *pc_air_shell;
   // Create a pointer on the heap
   PetscCall(PetscNew(&pc_air_shell));
   MPI_Comm comm;

   // We need to create our new pcshell
   PetscCall(PetscObjectGetComm((PetscObject)pc, &comm));
   PetscCall(PCCreate(comm, pc_air_shell));

   // Create the memory for our pc_air_data which holds all the air data
   create_pc_air_data_c(&pc_air_data);
   // The type and rest of the creation of the pcshell
   // is done in here
   create_pc_air_shell_c(&pc_air_data, pc_air_shell);
   // Set the pc data
   pc->data = (void*)pc_air_shell;

   // Set the method functions
   pc->ops->apply               = PCApply_AIR_c;
   pc->ops->setup               = PCSetUp_AIR_c;
   pc->ops->destroy             = PCDestroy_AIR_c;
   pc->ops->view                = PCView_AIR_c;  
   pc->ops->reset               = PCReset_AIR_c;
   pc->ops->setfromoptions      = PCSetFromOptions_AIR_c;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~

// Registers the PC type
PETSC_EXTERN void PCRegister_AIR()
{
   PetscCallVoid(PCRegister("air", PCCreate_AIR));
}

// This is called automatically when libpflare is loaded by 
// petsc as a shared library - this enables --download-pflare in the petsc
// configure to just work
// Is unused if static linking
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscpflare(void)
{
  PetscFunctionBegin;
  PCRegister_AIR();
  PetscFunctionReturn(PETSC_SUCCESS);
}