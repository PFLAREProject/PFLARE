/*
  Definition and registration of the PC AIR type
  Largely just a wrapper around all the fortran 
  using PCShell
*/

// Include the petsc header files
#include <petsc/private/pcimpl.h>
#include "pflare.h"

// Defined in C_Fortran_Bindings.F90
PETSC_EXTERN void PCReset_AIR_Shell_c(PC *pc);
PETSC_EXTERN void create_pc_air_data_c(void **pc_air_data);
PETSC_EXTERN void create_pc_air_shell_c(void **pc_air_data, PC *pc);
/* Defined in PCAIR_C_Fortran_Bindings.F90 */
PETSC_EXTERN PetscErrorCode PCAIRGetPrintStatsTimings_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN PetscErrorCode PCAIRGetMaxLevels_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN PetscErrorCode PCAIRGetCoarseEqLimit_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN PetscErrorCode PCAIRGetAutoTruncateStartLevel_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN PetscErrorCode PCAIRGetAutoTruncateTol_c(PC *pc, PetscReal *input_real);
PETSC_EXTERN PetscErrorCode PCAIRGetNumLevels_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN PetscErrorCode PCAIRGetProcessorAgglom_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN PetscErrorCode PCAIRGetProcessorAgglomRatio_c(PC *pc, PetscReal *input_real);
PETSC_EXTERN PetscErrorCode PCAIRGetProcessorAgglomFactor_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN PetscErrorCode PCAIRGetProcessEqLimit_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN PetscErrorCode PCAIRGetSubcomm_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN PetscErrorCode PCAIRGetStrongThreshold_c(PC *pc, PetscReal *input_real);
PETSC_EXTERN PetscErrorCode PCAIRGetDDCFraction_c(PC *pc, PetscReal *input_real);
PETSC_EXTERN PetscErrorCode PCAIRGetCFSplittingType_c(PC *pc, CFSplittingType *input_int);
PETSC_EXTERN PetscErrorCode PCAIRGetMaxLubySteps_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN PetscErrorCode PCAIRGetMaxitsAff_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN PetscErrorCode PCAIRGetOneCSmooth_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN PetscErrorCode PCAIRGetMatrixFreePolys_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN PetscErrorCode PCAIRGetOnePointClassicalProlong_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN PetscErrorCode PCAIRGetFullSmoothingUpAndDown_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN PetscErrorCode PCAIRGetSymmetric_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN PetscErrorCode PCAIRGetConstrainW_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN PetscErrorCode PCAIRGetConstrainZ_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN PetscErrorCode PCAIRGetStrongRThreshold_c(PC *pc, PetscReal *input_real);
PETSC_EXTERN PetscErrorCode PCAIRGetInverseType_c(PC *pc, PCPFLAREINVType *input_int);
PETSC_EXTERN PetscErrorCode PCAIRGetCInverseType_c(PC *pc, PCPFLAREINVType *input_int);
PETSC_EXTERN PetscErrorCode PCAIRGetZType_c(PC *pc, PCAIRZType *input_int);
PETSC_EXTERN PetscErrorCode PCAIRGetPolyOrder_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN PetscErrorCode PCAIRGetLairDistance_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN PetscErrorCode PCAIRGetInverseSparsityOrder_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN PetscErrorCode PCAIRGetCPolyOrder_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN PetscErrorCode PCAIRGetCInverseSparsityOrder_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestInverseType_c(PC *pc, PCPFLAREINVType *input_int);
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestPolyOrder_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestInverseSparsityOrder_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestMatrixFreePolys_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestSubcomm_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN PetscErrorCode PCAIRGetRDrop_c(PC *pc, PetscReal *input_real);
PETSC_EXTERN PetscErrorCode PCAIRGetADrop_c(PC *pc, PetscReal *input_real);
PETSC_EXTERN PetscErrorCode PCAIRGetALump_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN PetscErrorCode PCAIRGetReuseSparsity_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN PetscErrorCode PCAIRGetReusePolyCoeffs_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN PetscErrorCode PCAIRGetPolyCoeffs_c(PC *pc, PetscInt petsc_level, int which_inverse, PetscReal **coeffs_ptr, PetscInt *row_size, PetscInt *col_size);

// Setters
PETSC_EXTERN PetscErrorCode PCAIRSetPrintStatsTimings_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN PetscErrorCode PCAIRSetMaxLevels_c(PC *pc, PetscInt input_int);
PETSC_EXTERN PetscErrorCode PCAIRSetCoarseEqLimit_c(PC *pc, PetscInt input_int);
PETSC_EXTERN PetscErrorCode PCAIRSetAutoTruncateStartLevel_c(PC *pc, PetscInt input_int);
PETSC_EXTERN PetscErrorCode PCAIRSetAutoTruncateTol_c(PC *pc, PetscReal input_real);
PETSC_EXTERN PetscErrorCode PCAIRSetProcessorAgglom_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN PetscErrorCode PCAIRSetProcessorAgglomRatio_c(PC *pc, PetscReal input_real);
PETSC_EXTERN PetscErrorCode PCAIRSetProcessorAgglomFactor_c(PC *pc, PetscInt input_int);
PETSC_EXTERN PetscErrorCode PCAIRSetProcessEqLimit_c(PC *pc, PetscInt input_int);
PETSC_EXTERN PetscErrorCode PCAIRSetSubcomm_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN PetscErrorCode PCAIRSetStrongThreshold_c(PC *pc, PetscReal input_real);
PETSC_EXTERN PetscErrorCode PCAIRSetDDCFraction_c(PC *pc, PetscReal input_real);
PETSC_EXTERN PetscErrorCode PCAIRSetCFSplittingType_c(PC *pc, CFSplittingType input_int);
PETSC_EXTERN PetscErrorCode PCAIRSetMaxLubySteps_c(PC *pc, PetscInt input_int);
PETSC_EXTERN PetscErrorCode PCAIRSetMaxitsAff_c(PC *pc, PetscInt input_int);
PETSC_EXTERN PetscErrorCode PCAIRSetOneCSmooth_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN PetscErrorCode PCAIRSetMatrixFreePolys_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN PetscErrorCode PCAIRSetOnePointClassicalProlong_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN PetscErrorCode PCAIRSetFullSmoothingUpAndDown_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN PetscErrorCode PCAIRSetSymmetric_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN PetscErrorCode PCAIRSetConstrainW_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN PetscErrorCode PCAIRSetConstrainZ_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN PetscErrorCode PCAIRSetStrongRThreshold_c(PC *pc, PetscReal input_real);
PETSC_EXTERN PetscErrorCode PCAIRSetInverseType_c(PC *pc, PCPFLAREINVType input_int);
PETSC_EXTERN PetscErrorCode PCAIRSetZType_c(PC *pc, PCAIRZType input_int);
PETSC_EXTERN PetscErrorCode PCAIRSetLairDistance_c(PC *pc, PetscInt input_int);
PETSC_EXTERN PetscErrorCode PCAIRSetPolyOrder_c(PC *pc, PetscInt input_int);
PETSC_EXTERN PetscErrorCode PCAIRSetInverseSparsityOrder_c(PC *pc, PetscInt input_int);
PETSC_EXTERN PetscErrorCode PCAIRSetCInverseType_c(PC *pc, PCPFLAREINVType input_int);
PETSC_EXTERN PetscErrorCode PCAIRSetCPolyOrder_c(PC *pc, PetscInt input_int);
PETSC_EXTERN PetscErrorCode PCAIRSetCInverseSparsityOrder_c(PC *pc, PetscInt input_int);
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestInverseType_c(PC *pc, PCPFLAREINVType input_int);
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestPolyOrder_c(PC *pc, PetscInt input_int);
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestInverseSparsityOrder_c(PC *pc, PetscInt input_int);
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestMatrixFreePolys_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestSubcomm_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN PetscErrorCode PCAIRSetRDrop_c(PC *pc, PetscReal input_real);
PETSC_EXTERN PetscErrorCode PCAIRSetADrop_c(PC *pc, PetscReal input_real);
PETSC_EXTERN PetscErrorCode PCAIRSetALump_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN PetscErrorCode PCAIRSetReuseSparsity_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN PetscErrorCode PCAIRSetReusePolyCoeffs_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN PetscErrorCode PCAIRSetPolyCoeffs_c(PC *pc, PetscInt petsc_level, int which_inverse, PetscReal *coeffs_ptr, PetscInt row_size, PetscInt col_size);

// ~~~~~~~~~~~~~

static PetscErrorCode PCReset_AIR_c(PC pc)
{
   PetscFunctionBegin;

   PC *pc_air_shell = (PC *)pc->data;
   // Call the underlying reset - this won't touch the context 
   // in our pcshell though as pcshell doesn't offer a way to 
   // give a custom reset function
   PCReset(*pc_air_shell);

   // So now we manually reset the underlying data
   PCReset_AIR_Shell_c(pc_air_shell);  

   PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_AIR_c(PC pc, Vec x, Vec y)
{
   PetscFunctionBegin;
   PC *pc_air_shell = (PC *)pc->data;

   VecType vtypex, vtypey;
   VecGetType(x, &vtypex);
   VecGetType(y, &vtypey);

   // Just call the underlying pcshell apply
   PCApply(*pc_air_shell, x, y); 
   PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_AIR_c(PC pc)
{
   PetscFunctionBegin;
   PC *pc_air_shell = (PC *)pc->data;
   // Just call the underlying pcshell destroy, this also 
   // destroys the pc_air_data
   PCDestroy(pc_air_shell);
   // Then destroy the heap pointer
   PetscFree(pc_air_shell);
   PetscFunctionReturn(0);
}

// ~~~~~~~~~~

static PetscErrorCode PCSetUp_AIR_c(PC pc)
{
   PetscFunctionBegin;
   PC *pc_air_shell = (PC *)pc->data;

   // The pc_air_shell doesn't have any operators yet 
   // as they are not available yet in pccreate
   // so we have to set them
   PCSetOperators(*pc_air_shell, pc->mat, pc->pmat);

   // Now we should be able to call the pcshell setup
   // that builds the air hierarchy
   PCSetUp(*pc_air_shell);
   PetscFunctionReturn(0);
}

// Get the underlying PCShell
PETSC_EXTERN PetscErrorCode c_PCAIRGetPCShell(PC *pc, PC *pc_air_shell)
{
   PetscFunctionBegin;
   PC *pc_shell = (PC *)(*pc)->data;
   *pc_air_shell = *pc_shell;
   PetscFunctionReturn(0);
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
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetMaxLevels(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetMaxLevels_c(&pc, input_int);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCoarseEqLimit(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetCoarseEqLimit_c(&pc, input_int);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetAutoTruncateStartLevel(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetAutoTruncateStartLevel_c(&pc, input_int);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetAutoTruncateTol(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetAutoTruncateTol_c(&pc, input_real);
   PetscFunctionReturn(0);
}
// Returns the number of levels in the underlying PCMG
// -1 if the mg is not setup yet
PETSC_EXTERN PetscErrorCode PCAIRGetNumLevels(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetNumLevels_c(&pc, input_int);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetProcessorAgglom(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetProcessorAgglom_c(&pc, input_bool);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetProcessorAgglomRatio(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetProcessorAgglomRatio_c(&pc, input_real);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetProcessorAgglomFactor(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetProcessorAgglomFactor_c(&pc, input_int);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetProcessEqLimit(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetProcessEqLimit_c(&pc, input_int);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetSubcomm(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetSubcomm_c(&pc, input_bool);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetStrongThreshold(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetStrongThreshold_c(&pc, input_real);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetDDCFraction(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetDDCFraction_c(&pc, input_real);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCFSplittingType(PC pc, CFSplittingType *input_int)
{
   PetscFunctionBegin;
   PCAIRGetCFSplittingType_c(&pc, input_int);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetMaxLubySteps(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetMaxLubySteps_c(&pc, input_int);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetMaxitsAff(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetMaxitsAff_c(&pc, input_int);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetOneCSmooth(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetOneCSmooth_c(&pc, input_bool);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetMatrixFreePolys(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetMatrixFreePolys_c(&pc, input_bool);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetOnePointClassicalProlong(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetOnePointClassicalProlong_c(&pc, input_bool);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetFullSmoothingUpAndDown(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetFullSmoothingUpAndDown_c(&pc, input_bool);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetSymmetric(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetSymmetric_c(&pc, input_bool);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetConstrainW(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetConstrainW_c(&pc, input_bool);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetConstrainZ(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetConstrainZ_c(&pc, input_bool);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetStrongRThreshold(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetStrongRThreshold_c(&pc, input_real);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetInverseType(PC pc, PCPFLAREINVType *input_int)
{
   PetscFunctionBegin;
   PCAIRGetInverseType_c(&pc, input_int);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCInverseType(PC pc, PCPFLAREINVType *input_int)
{
   PetscFunctionBegin;
   PCAIRGetCInverseType_c(&pc, input_int);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetZType(PC pc, PCAIRZType *input_int)
{
   PetscFunctionBegin;
   PCAIRGetZType_c(&pc, input_int);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetPolyOrder(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetPolyOrder_c(&pc, input_int);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetLairDistance(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetLairDistance_c(&pc, input_int);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetInverseSparsityOrder(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetInverseSparsityOrder_c(&pc, input_int);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCPolyOrder(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetCPolyOrder_c(&pc, input_int);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCInverseSparsityOrder(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetCInverseSparsityOrder_c(&pc, input_int);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestInverseType(PC pc, PCPFLAREINVType *input_int)
{
   PetscFunctionBegin;
   PCAIRGetCoarsestInverseType_c(&pc, input_int);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestPolyOrder(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetCoarsestPolyOrder_c(&pc, input_int);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestInverseSparsityOrder(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetCoarsestInverseSparsityOrder_c(&pc, input_int);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestMatrixFreePolys(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetCoarsestMatrixFreePolys_c(&pc, input_bool);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestSubcomm(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetCoarsestSubcomm_c(&pc, input_bool);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetRDrop(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetRDrop_c(&pc, input_real);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetADrop(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetADrop_c(&pc, input_real);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetALump(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetALump_c(&pc, input_bool);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetReuseSparsity(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetReuseSparsity_c(&pc, input_bool);
   PetscFunctionReturn(0);
}
PETSC_EXTERN PetscErrorCode PCAIRGetReusePolyCoeffs(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetReusePolyCoeffs_c(&pc, input_bool);
   PetscFunctionReturn(0);
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
   PetscFunctionReturn(0);
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
   PetscFunctionReturn(0);
}

// Maximum number of levels in the multigrid hierarchy
// Default: 300
// -pc_air_max_levels
PETSC_EXTERN PetscErrorCode PCAIRSetMaxLevels(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PCAIRGetMaxLevels(pc, &old_int);
   if (old_int == input_int) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);
   PCAIRSetMaxLevels_c(&pc, input_int);
   PetscFunctionReturn(0);
}
// Minimum number of global unknowns on the coarse grid
// Default: 6
// -pc_air_coarse_eq_limit
PETSC_EXTERN PetscErrorCode PCAIRSetCoarseEqLimit(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PCAIRGetCoarseEqLimit(pc, &old_int);
   if (old_int == input_int) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);
   PCAIRSetCoarseEqLimit_c(&pc, input_int);
   PetscFunctionReturn(0);
}
// From this level onwards, build and then evaluate if the coarse grid solver
// is good enough and use that to determine if we should truncate on that level
// Default: -1
// -pc_air_auto_truncate_start_level
PETSC_EXTERN PetscErrorCode PCAIRSetAutoTruncateStartLevel(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PCAIRGetAutoTruncateStartLevel(pc, &old_int);
   if (old_int == input_int) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);
   PCAIRSetAutoTruncateStartLevel_c(&pc, input_int);
   PetscFunctionReturn(0);
}
// What relative tolerance to use to determine if a coarse grid solver is good enough
// Default: 1e-14
// -pc_air_auto_truncate_tol
PETSC_EXTERN PetscErrorCode PCAIRSetAutoTruncateTol(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PetscReal old_real;
   PCAIRGetAutoTruncateTol(pc, &old_real);
   if (old_real == input_real) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetAutoTruncateTol_c(&pc, input_real);
   PetscFunctionReturn(0);
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
   PCAIRGetProcessorAgglom(pc, &old_bool);
   if (old_bool == input_bool) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);   
   PCAIRSetProcessorAgglom_c(&pc, input_bool);
   PetscFunctionReturn(0);
}
// The local to nonlocal ratio of nnzs that is used to 
// trigger processor agglomeration on all level
// Default: 2.0
// -pc_air_processor_agglom_ratio
PETSC_EXTERN PetscErrorCode PCAIRSetProcessorAgglomRatio(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PetscReal old_real;
   PCAIRGetProcessorAgglomRatio(pc, &old_real);
   if (old_real == input_real) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetProcessorAgglomRatio_c(&pc, input_real);
   PetscFunctionReturn(0);
}
// What factor to reduce the number of active MPI ranks by
// each time when doing processor agglomeration
// Default: 2
// -pc_air_processor_agglom_factor
PETSC_EXTERN PetscErrorCode PCAIRSetProcessorAgglomFactor(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PCAIRGetProcessorAgglomFactor(pc, &old_int);
   if (old_int == input_int) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetProcessorAgglomFactor_c(&pc, input_int);
   PetscFunctionReturn(0);
}
// If on average there are fewer than this number of equations per rank
// processor agglomeration will be triggered
// Default: 50
// -pc_air_process_eq_limit
PETSC_EXTERN PetscErrorCode PCAIRSetProcessEqLimit(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PCAIRGetProcessEqLimit(pc, &old_int);
   if (old_int == input_int) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetProcessEqLimit_c(&pc, input_int);
   PetscFunctionReturn(0);
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
   PetscFunctionReturn(0);
}
// This is used in the CF splitting to define strong dependencies/influences
// Default: 0.5
// -pc_air_strong_threshold
PETSC_EXTERN PetscErrorCode PCAIRSetStrongThreshold(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PetscReal old_real;
   PCAIRGetStrongThreshold(pc, &old_real);
   if (old_real == input_real) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetStrongThreshold_c(&pc, input_real);
   PetscFunctionReturn(0);
}
// Second pass in the PMISR DDC CF splitting converts 
// this fraction of local F points to C based on diagonal dominance
// Default: 0.1
// -pc_air_ddc_fraction
PETSC_EXTERN PetscErrorCode PCAIRSetDDCFraction(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PetscReal old_real;
   PCAIRGetDDCFraction(pc, &old_real);
   if (old_real == input_real) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);     
   PCAIRSetDDCFraction_c(&pc, input_real);
   PetscFunctionReturn(0);
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
   PCAIRGetCFSplittingType(pc, &old_int);
   if (old_int == input_int) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetCFSplittingType_c(&pc, input_int);
   PetscFunctionReturn(0);
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
   PCAIRGetMaxLubySteps(pc, &old_int);
   if (old_int == input_int) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetMaxLubySteps_c(&pc, input_int);
   PetscFunctionReturn(0);
}
// How many iterations of F point smoothing to do 
// Default: 2
// -pc_air_maxits_a_ff
PETSC_EXTERN PetscErrorCode PCAIRSetMaxitsAff(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PCAIRGetMaxitsAff(pc, &old_int);
   if (old_int == input_int) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetMaxitsAff_c(&pc, input_int);
   PetscFunctionReturn(0);
}
// Do we do maxits_a_ff F smoothing or maxits_a_ff F smooths then a single C?
// Default: false
// -pc_air_one_c_smooth
PETSC_EXTERN PetscErrorCode PCAIRSetOneCSmooth(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscBool old_bool;
   PCAIRGetOneCSmooth(pc, &old_bool);
   if (old_bool == input_bool) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetOneCSmooth_c(&pc, input_bool);
   PetscFunctionReturn(0);
}
// Do we apply our polynomials matrix free when smoothing?
// Default: false
// -pc_air_matrix_free_polys
PETSC_EXTERN PetscErrorCode PCAIRSetMatrixFreePolys(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscBool old_bool;
   PCAIRGetMatrixFreePolys(pc, &old_bool);
   if (old_bool == input_bool) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetMatrixFreePolys_c(&pc, input_bool);
   PetscFunctionReturn(0);
}
// Do we use a one point injection classical prolongator or an AIR-style prolongator
// Default: true
// -pc_air_one_point_classical_prolong
PETSC_EXTERN PetscErrorCode PCAIRSetOnePointClassicalProlong(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscBool old_bool;
   PCAIRGetOnePointClassicalProlong(pc, &old_bool);
   if (old_bool == input_bool) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetOnePointClassicalProlong_c(&pc, input_bool);
   PetscFunctionReturn(0);
}
// Do we do full smoothing up and down rather than FF or FFC
// Default: false
// -pc_air_full_smoothing_up_and_down
PETSC_EXTERN PetscErrorCode PCAIRSetFullSmoothingUpAndDown(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscBool old_bool;
   PCAIRGetFullSmoothingUpAndDown(pc, &old_bool);
   if (old_bool == input_bool) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetFullSmoothingUpAndDown_c(&pc, input_bool);
   PetscFunctionReturn(0);
}
// Do we define our prolongator as R^T?
// Default: false
// -pc_air_symmetric
PETSC_EXTERN PetscErrorCode PCAIRSetSymmetric(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscBool old_bool;
   PCAIRGetSymmetric(pc, &old_bool);
   if (old_bool == input_bool) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetSymmetric_c(&pc, input_bool);
   PetscFunctionReturn(0);
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
   PCAIRGetConstrainW(pc, &old_bool);
   if (old_bool == input_bool) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);   
   PCAIRSetConstrainW_c(&pc, input_bool);
   PetscFunctionReturn(0);
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
   PCAIRGetConstrainZ(pc, &old_bool);
   if (old_bool == input_bool) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);     
   PCAIRSetConstrainZ_c(&pc, input_bool);
   PetscFunctionReturn(0);
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
   PCAIRGetStrongRThreshold(pc, &old_real);
   if (old_real == input_real) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);     
   PCAIRSetStrongRThreshold_c(&pc, input_real);
   PetscFunctionReturn(0);
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
   PCAIRGetInverseType(pc, &old_int);
   if (old_int == input_int) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetInverseType_c(&pc, input_int);
   PetscFunctionReturn(0);
}
// What type of approximation do we use for Acc^-1 
// If unset, this defaults to whatever the F point smoother is atm
// Default: pc_air_inverse_type
// -pc_air_c_inverse_type
PETSC_EXTERN PetscErrorCode PCAIRSetCInverseType(PC pc, PCPFLAREINVType input_int)
{
   PetscFunctionBegin;
   PCPFLAREINVType old_int;
   PCAIRGetCInverseType(pc, &old_int);
   if (old_int == input_int) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetCInverseType_c(&pc, input_int);
   PetscFunctionReturn(0);
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
   PCAIRGetZType(pc, &old_int);
   if (old_int == input_int) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);   
   PCAIRSetZType_c(&pc, input_int);
   PetscFunctionReturn(0);
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
   PCAIRGetLairDistance(pc, &old_int);
   if (old_int == input_int) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);   
   PCAIRSetLairDistance_c(&pc, input_int);
   PetscFunctionReturn(0);
}
// This is the order of polynomial we use in air if inverse_type is 
// power, arnoldi, newton or neumann
// Default: 6
// -pc_air_poly_order
PETSC_EXTERN PetscErrorCode PCAIRSetPolyOrder(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PCAIRGetPolyOrder(pc, &old_int);
   if (old_int == input_int) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);   
   PCAIRSetPolyOrder_c(&pc, input_int);
   PetscFunctionReturn(0);
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
   PCAIRGetInverseSparsityOrder(pc, &old_int);
   if (old_int == input_int) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetInverseSparsityOrder_c(&pc, input_int);
   PetscFunctionReturn(0);
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
   PCAIRGetCPolyOrder(pc, &old_int);
   if (old_int == input_int) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);   
   PCAIRSetCPolyOrder_c(&pc, input_int);
   PetscFunctionReturn(0);
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
   PCAIRGetCInverseSparsityOrder(pc, &old_int);
   if (old_int == input_int) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetCInverseSparsityOrder_c(&pc, input_int);
   PetscFunctionReturn(0);
}
// Coarse grid inverse type (see PCAIRSetInverseType)
// Default: power
// -pc_air_coarsest_inverse_type
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestInverseType(PC pc, PCPFLAREINVType input_int)
{
   PetscFunctionBegin;
   PCPFLAREINVType old_int;
   PCAIRGetCoarsestInverseType(pc, &old_int);
   if (old_int == input_int) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);   
   PCAIRSetCoarsestInverseType_c(&pc, input_int);
   PetscFunctionReturn(0);
}
// Coarse grid polynomial order (see PCAIRSetPolyOrder)
// Default: 6
// -pc_air_coarsest_poly_order
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestPolyOrder(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PCAIRGetCoarsestPolyOrder(pc, &old_int);
   if (old_int == input_int) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetCoarsestPolyOrder_c(&pc, input_int);
   PetscFunctionReturn(0);
}
// Coarse grid polynomial sparsity order (see PCAIRSetInverseSparsityOrder)
// Default: 1
// -pc_air_coarsest_inverse_sparsity_order
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestInverseSparsityOrder(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscInt old_int;
   PCAIRGetCoarsestInverseSparsityOrder(pc, &old_int);
   if (old_int == input_int) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetCoarsestInverseSparsityOrder_c(&pc, input_int);
   PetscFunctionReturn(0);
}
// Coarse grid matrix-free application (see PCAIRSetMatrixFreePolys)
// Default: false
// -pc_air_coarsest_matrix_free_polys
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestMatrixFreePolys(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscBool old_bool;
   PCAIRGetCoarsestMatrixFreePolys(pc, &old_bool);
   if (old_bool == input_bool) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetCoarsestMatrixFreePolys_c(&pc, input_bool);
   PetscFunctionReturn(0);
}
// Coarse grid subcomm (see PCAIRSetSubcomm)
// Default: false
// -pc_air_coarsest_subcomm
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestSubcomm(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   // No need to reset if this changes
   PCAIRSetCoarsestSubcomm_c(&pc, input_bool);
   PetscFunctionReturn(0);
}
// Relative drop tolerances (inf norm) on R 
// Default: 0.01
// -pc_air_r_drop
PETSC_EXTERN PetscErrorCode PCAIRSetRDrop(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PetscReal old_real;
   PCAIRGetRDrop(pc, &old_real);
   if (old_real == input_real) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetRDrop_c(&pc, input_real);
   PetscFunctionReturn(0);
}
// Relative drop tolerances (inf norm) on A 
// Default: 0.001
// -pc_air_a_drop
PETSC_EXTERN PetscErrorCode PCAIRSetADrop(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PetscReal old_real;
   PCAIRGetADrop(pc, &old_real);
   if (old_real == input_real) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);     
   PCAIRSetADrop_c(&pc, input_real);
   PetscFunctionReturn(0);   
}
// Whether to lump in A or drop
// Default: false
// -pc_air_a_lump
PETSC_EXTERN PetscErrorCode PCAIRSetALump(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscBool old_bool;
   PCAIRGetALump(pc, &old_bool);
   if (old_bool == input_bool) PetscFunctionReturn(0);
   PCReset_AIR_c(pc);    
   PCAIRSetALump_c(&pc, input_bool);
   PetscFunctionReturn(0); 
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
   PetscFunctionReturn(0); 
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
   PetscFunctionReturn(0); 
}
// This routine sets the polynomial coefficients in the PCAIR object
// row_size and col_size are the size of the coeffs_ptr array
PETSC_EXTERN PetscErrorCode PCAIRSetPolyCoeffs(PC pc, PetscInt petsc_level, int which_inverse, PetscReal *coeffs_ptr, PetscInt row_size, PetscInt col_size)
{
   PetscFunctionBegin;
   PCAIRSetPolyCoeffs_c(&pc,petsc_level, which_inverse, \
      coeffs_ptr, row_size, col_size);
   PetscFunctionReturn(0);
}

// ~~~~~~~~~~~~~~~~

#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR >= 18)
static PetscErrorCode PCSetFromOptions_AIR_c(PC pc, PetscOptionItems *PetscOptionsObject)
#else
static PetscErrorCode PCSetFromOptions_AIR_c(PetscOptionItems *PetscOptionsObject,PC pc)
#endif
{
   PetscFunctionBegin;
   
   PetscBool    flg, old_flag;
   PetscInt input_int, old_int;
   PetscReal input_real, old_real;
   PCPFLAREINVType old_type, type;
   PCAIRZType old_z_type, z_type;
   CFSplittingType old_cf_type, cf_type;

#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR >= 18)
   PetscOptionsHeadBegin(PetscOptionsObject, "PCAIR options");
#else
   PetscOptionsHead(PetscOptionsObject, "PCAIR options");
#endif
   // ~~~~
   PCAIRGetPrintStatsTimings(pc, &old_flag);
   flg = old_flag;
   PetscOptionsBool("-pc_air_print_stats_timings", "Print statistics and timings", "PCAIRSetPrintStatsTimings", old_flag, &flg, NULL);
   PCAIRSetPrintStatsTimings(pc, flg);
   // ~~~~
   PCAIRGetSubcomm(pc, &old_flag);
   flg = old_flag;
   PetscOptionsBool("-pc_air_subcomm", "Computes polynomial coefficients on subcomm", "PCAIRSetSubcomm", old_flag, &flg, NULL);
   PCAIRSetSubcomm(pc, flg);   
   // ~~~~
   PCAIRGetProcessorAgglom(pc, &old_flag);
   flg = old_flag;
   PetscOptionsBool("-pc_air_processor_agglom", "Processor agglomeration", "PCAIRSetProcessorAgglom", old_flag, &flg, NULL);
   PCAIRSetProcessorAgglom(pc, flg);
   // ~~~~
   PCAIRGetOneCSmooth(pc, &old_flag);
   flg = old_flag;
   PetscOptionsBool("-pc_air_one_c_smooth", "Does one C smooth", "PCAIRSetOneCSmooth", old_flag, &flg, NULL);
   PCAIRSetOneCSmooth(pc, flg);
   // ~~~~
   PCAIRGetMatrixFreePolys(pc, &old_flag);
   flg = old_flag;
   PetscOptionsBool("-pc_air_matrix_free_polys", "Applies polynomial smoothers matrix-free", "PCAIRSetMatrixFreePolys", old_flag, &flg, NULL);
   PCAIRSetMatrixFreePolys(pc, flg);
   // ~~~~   
   PCAIRGetOnePointClassicalProlong(pc, &old_flag);
   flg = old_flag;
   PetscOptionsBool("-pc_air_one_point_classical_prolong", "One-point classical prolongator", "PCAIRSetOnePointClassicalProlong", old_flag, &flg, NULL);
   PCAIRSetOnePointClassicalProlong(pc, flg);
   // ~~~~  
   PCAIRGetFullSmoothingUpAndDown(pc, &old_flag);
   flg = old_flag;
   PetscOptionsBool("-pc_air_full_smoothing_up_and_down", "Full up and down smoothing", "PCAIRSetFullSmoothingUpAndDown", old_flag, &flg, NULL);
   PCAIRSetFullSmoothingUpAndDown(pc, flg);
   // ~~~~  
   PCAIRGetSymmetric(pc, &old_flag);
   flg = old_flag;
   PetscOptionsBool("-pc_air_symmetric", "Use symmetric grid-transfer operators", "PCAIRSetSymmetric", old_flag, &flg, NULL);
   PCAIRSetSymmetric(pc, flg);
   // ~~~~
   PCAIRGetConstrainW(pc, &old_flag);
   flg = old_flag;
   PetscOptionsBool("-pc_air_constrain_w", "Use constraints on prolongator", "PCAIRSetConstrainW", old_flag, &flg, NULL);
   PCAIRSetConstrainW(pc, flg);
   // ~~~~
   PCAIRGetConstrainZ(pc, &old_flag);
   flg = old_flag;
   PetscOptionsBool("-pc_air_constrain_z", "Use constraints on restrictor", "PCAIRSetConstrainZ", old_flag, &flg, NULL);
   PCAIRSetConstrainZ(pc, flg);
   // ~~~~  
   PCAIRGetCoarsestMatrixFreePolys(pc, &old_flag);
   flg = old_flag;
   PetscOptionsBool("-pc_air_coarsest_matrix_free_polys", "Applies polynomial coarse grid solver matrix-free", "PCAIRSetCoarsestMatrixFreePolys", old_flag, &flg, NULL);
   PCAIRSetCoarsestMatrixFreePolys(pc, flg);
   // ~~~~  
   PCAIRGetCoarsestSubcomm(pc, &old_flag);
   flg = old_flag;
   PetscOptionsBool("-pc_air_coarsest_subcomm", "Computes polynomial coefficients on the coarse grid on subcomm", "PCAIRGetCoarsestSubcomm", old_flag, &flg, NULL);
   PCAIRSetCoarsestSubcomm(pc, flg);
   // ~~~~ 
   PCAIRGetALump(pc, &old_flag);
   flg = old_flag;
   PetscOptionsBool("-pc_air_a_lump", "Uses lumping on A", "PCAIRSetALump", old_flag, &flg, NULL);
   PCAIRSetALump(pc, flg);
   // ~~~~ 
   PCAIRGetReuseSparsity(pc, &old_flag);
   flg = old_flag;
   PetscOptionsBool("-pc_air_reuse_sparsity", "Reuses sparsity during setup", "PCAIRSetReuseSparsity", old_flag, &flg, NULL);
   PCAIRSetReuseSparsity(pc, flg);   
   // ~~~~
   PCAIRGetReusePolyCoeffs(pc, &old_flag);
   flg = old_flag;
   PetscOptionsBool("-pc_air_reuse_poly_coeffs", "Reuses gmres polynomial coefficients during setup", "PCAIRSetReusePolyCoeffs", old_flag, &flg, NULL);
   PCAIRSetReusePolyCoeffs(pc, flg);   
   // ~~~~
   PCAIRGetProcessorAgglomRatio(pc, &old_real);
   input_real = old_real;
   PetscOptionsReal("-pc_air_processor_agglom_ratio", "Ratio to trigger processor agglomeration", "PCAIRSetProcessorAgglomRatio", old_real, &input_real, NULL);
   PCAIRSetProcessorAgglomRatio(pc, input_real);   
   // ~~~~
   PCAIRGetAutoTruncateTol(pc, &old_real);
   input_real = old_real;
   PetscOptionsReal("-pc_air_auto_truncate_tol", "Tolerance to use with auto truncation", "PCAIRSetAutoTruncateTol", old_real, &input_real, NULL);
   PCAIRSetAutoTruncateTol(pc, input_real);
   // ~~~~
   PCAIRGetStrongThreshold(pc, &old_real);
   input_real = old_real;
   PetscOptionsReal("-pc_air_strong_threshold", "Strong threshold for CF splitting", "PCAIRSetStrongThreshold", old_real, &input_real, NULL);
   PCAIRSetStrongThreshold(pc, input_real);
   // ~~~~ 
   PCAIRGetDDCFraction(pc, &old_real);
   input_real = old_real;
   PetscOptionsReal("-pc_air_ddc_fraction", "DDC fraction for CF splitting", "PCAIRGetDDCFraction", old_real, &input_real, NULL);
   PCAIRSetDDCFraction(pc, input_real);
   // ~~~~
   PCAIRGetStrongRThreshold(pc, &old_real);
   input_real = old_real;
   PetscOptionsReal("-pc_air_strong_r_threshold", "Strong R threshold for grid-transfer operators", "PCAIRSetStrongRThreshold", old_real, &input_real, NULL);
   PCAIRSetStrongRThreshold(pc, input_real);
   // ~~~~ 
   PCAIRGetRDrop(pc, &old_real);
   input_real = old_real;
   PetscOptionsReal("-pc_air_r_drop", "Drop tolerance for R", "PCAIRSetRDrop", old_real, &input_real, NULL);
   PCAIRSetRDrop(pc, input_real);
   // ~~~~  
   PCAIRGetADrop(pc, &old_real);
   input_real = old_real;
   PetscOptionsReal("-pc_air_a_drop", "Drop tolerance for A", "PCAIRSetADrop", old_real, &input_real, NULL);
   PCAIRSetADrop(pc, input_real);
   // ~~~~  
   const char *const CFSplittingTypes[] = {"PMISR_DDC", "PMIS", "PMIS_DIST2", "AGG", "PMIS_AGG", "CFSplittingType", "CF_", NULL};
   PCAIRGetCFSplittingType(pc, &old_cf_type);
   cf_type = old_cf_type;
   PetscOptionsEnum("-pc_air_cf_splitting_type", "CF splitting algorithm", "PCAIRSetCFSplittingType", CFSplittingTypes, (PetscEnum)old_cf_type, (PetscEnum *)&cf_type, &flg);
   PCAIRSetCFSplittingType(pc, cf_type);
   // ~~~~   
   PCAIRGetMaxLubySteps(pc, &old_int);
   input_int = old_int;
   PetscOptionsInt("-pc_air_max_luby_steps", "Max Luby steps in CF splitting algorithm", "PCAIRSetMaxLubySteps", old_int, &input_int, NULL);
   PCAIRSetMaxLubySteps(pc, input_int);
   // ~~~~ 
   PCAIRGetMaxitsAff(pc, &old_int);
   input_int = old_int;
   PetscOptionsInt("-pc_air_maxits_a_ff", "Iterations of F smoothing", "PCAIRSetMaxitsAff", old_int, &input_int, NULL);
   PCAIRSetMaxitsAff(pc, input_int);
   // ~~~~    
   PCAIRGetMaxLevels(pc, &old_int);
   input_int = old_int;
   PetscOptionsInt("-pc_air_max_levels", "Maximum number of levels", "PCAIRSetMaxLevels", old_int, &input_int, NULL);
   PCAIRSetMaxLevels(pc, input_int);
   // ~~~~    
   PCAIRGetCoarseEqLimit(pc, &old_int);
   input_int = old_int;
   PetscOptionsInt("-pc_air_coarse_eq_limit", "Minimum number of global unknowns on the coarse grid", "PCAIRSetCoarseEqLimit", old_int, &input_int, NULL);
   PCAIRSetCoarseEqLimit(pc, input_int);   
   // ~~~~    
   PCAIRGetAutoTruncateStartLevel(pc, &old_int);
   input_int = old_int;
   PetscOptionsInt("-pc_air_auto_truncate_start_level", "Use auto truncation from this level", "PCAIRSetAutoTruncateStartLevel", old_int, &input_int, NULL);
   PCAIRSetAutoTruncateStartLevel(pc, input_int);
   // ~~~~ 
   const char *const PCPFLAREINVTypes[] = {"POWER", "ARNOLDI", "NEWTON", "NEWTON_NO_EXTRA", "NEUMANN", "SAI", "ISAI", "WJACOBI", "JACOBI", "PCPFLAREINVType", "PFLAREINV_", NULL};
   PCAIRGetInverseType(pc, &old_type);
   type = old_type;
   PetscOptionsEnum("-pc_air_inverse_type", "Inverse type", "PCPFLAREINVSetType", PCPFLAREINVTypes, (PetscEnum)old_type, (PetscEnum *)&type, &flg);
   PCAIRSetInverseType(pc, type);
   // ~~~~ 
   // Defaults to whatever the F point smoother is atm
   PCAIRGetInverseType(pc, &old_type);
   type = old_type;
   PetscOptionsEnum("-pc_air_c_inverse_type", "C point inverse type", "PCPFLAREINVSetType", PCPFLAREINVTypes, (PetscEnum)old_type, (PetscEnum *)&type, &flg);
   PCAIRSetCInverseType(pc, type);
   // ~~~~
   const char *const PCAIRZTypes[] = {"PRODUCT", "LAIR", "LAIR_SAI", "PCAIRZType", "AIR_Z_", NULL};
   PCAIRGetZType(pc, &old_z_type);
   z_type = old_z_type;
   PetscOptionsEnum("-pc_air_z_type", "Z type", "PCAIRSetZType", PCAIRZTypes, (PetscEnum)old_z_type, (PetscEnum *)&z_type, &flg);
   PCAIRSetZType(pc, z_type);
   // ~~~~ 
   PCAIRGetLairDistance(pc, &old_int);
   input_int = old_int;
   PetscOptionsInt("-pc_air_lair_distance", "lAIR distance", "PCAIRSetLairDistance", old_int, &input_int, NULL);
   PCAIRSetLairDistance(pc, input_int);   
   // ~~~~ 
   PCAIRGetPolyOrder(pc, &old_int);
   input_int = old_int;
   PetscOptionsInt("-pc_air_poly_order", "Polynomial order", "PCAIRSetPolyOrder", old_int, &input_int, NULL);
   PCAIRSetPolyOrder(pc, input_int);
   // ~~~~ 
   PCAIRGetInverseSparsityOrder(pc, &old_int);
   input_int = old_int;
   PetscOptionsInt("-pc_air_inverse_sparsity_order", "Inverse sparsity order", "PCAIRSetInverseSparsityOrder", old_int, &input_int, NULL);
   PCAIRSetInverseSparsityOrder(pc, input_int);
   // ~~~~ 
   // Defaults to whatever the F point smoother is atm
   PCAIRGetPolyOrder(pc, &old_int);
   input_int = old_int;
   PetscOptionsInt("-pc_air_c_poly_order", "C point polynomial order", "PCAIRSetCPolyOrder", old_int, &input_int, NULL);
   PCAIRSetCPolyOrder(pc, input_int);
   // ~~~~ 
   // Defaults to whatever the F point smoother is atm
   PCAIRGetInverseSparsityOrder(pc, &old_int);
   input_int = old_int;
   PetscOptionsInt("-pc_air_c_inverse_sparsity_order", "C point inverse sparsity order", "PCAIRSetCInverseSparsityOrder", old_int, &input_int, NULL);
   PCAIRSetCInverseSparsityOrder(pc, input_int);   
   // ~~~~ 
   PCAIRGetCoarsestInverseType(pc, &old_type);
   type = old_type;
   PetscOptionsEnum("-pc_air_coarsest_inverse_type", "Inverse type on the coarse grid", "PCPFLAREINVSetType", PCPFLAREINVTypes, (PetscEnum)old_type, (PetscEnum *)&type, &flg);
   PCAIRSetCoarsestInverseType(pc, type);
   // ~~~~ 
   PCAIRGetCoarsestPolyOrder(pc, &old_int);
   input_int = old_int;
   PetscOptionsInt("-pc_air_coarsest_poly_order", "Polynomial order on the coarse grid", "PCAIRSetCoarsestPolyOrder", old_int, &input_int, NULL);
   PCAIRSetCoarsestPolyOrder(pc, input_int);
   // ~~~~ 
   PCAIRGetCoarsestInverseSparsityOrder(pc, &old_int);
   input_int = old_int;
   PetscOptionsInt("-pc_air_coarsest_inverse_sparsity_order", "Inverse sparsity order on the coarse grid", "PCAIRSetCoarsestInverseSparsityOrder", old_int, &input_int, NULL);
   PCAIRSetCoarsestInverseSparsityOrder(pc, input_int);
   // ~~~~ 
   PCAIRGetProcessorAgglomFactor(pc, &old_int);
   input_int = old_int;
   PetscOptionsInt("-pc_air_processor_agglom_factor", "Factor to reduce MPI ranks by", "PCAIRSetProcessorAgglomFactor", old_int, &input_int, NULL);
   PCAIRSetProcessorAgglomFactor(pc, input_int);   
   // ~~~~        
   PCAIRGetProcessEqLimit(pc, &old_int);
   input_int = old_int;
   PetscOptionsInt("-pc_air_process_eq_limit", "Trigger process agglomeration if fewer eqs/core", "PCAIRSetProcessEqLimit", old_int, &input_int, NULL);
   PCAIRSetProcessEqLimit(pc, input_int);   
   // ~~~~                                                       

#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR >= 18)
   PetscOptionsHeadEnd();
#else
   PetscOptionsTail();
#endif
   PetscFunctionReturn(0);
}

// ~~~~~~~~~~~~~~~~

static PetscErrorCode PCView_AIR_c(PC pc, PetscViewer viewer)
{
   PetscFunctionBegin;
   
   PC *pc_air_shell = (PC *)pc->data;

   PetscInt input_int, input_int_two, input_int_three;
   PetscErrorCode ierr = 0;
   PetscBool flg, flg_two;
   PetscReal input_real, input_real_two;
   PCPFLAREINVType input_type;
   PCAIRZType z_type;
   CFSplittingType cf_type;

   // Print out details
   PetscBool  iascii;
   PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii);   

   if (iascii) {

      ierr =  PCAIRGetMaxLevels(pc, &input_int);
      PetscViewerASCIIPrintf(viewer, "  Max number of levels=%"PetscInt_FMT" \n", input_int);
      ierr =  PCAIRGetCoarseEqLimit(pc, &input_int);
      PetscViewerASCIIPrintf(viewer, "  Coarse eq limit=%"PetscInt_FMT" \n", input_int);  
      
      ierr =  PCAIRGetAutoTruncateStartLevel(pc, &input_int);
      ierr =  PCAIRGetAutoTruncateTol(pc, &input_real);
      if (input_int != -1) 
      {
         PetscViewerASCIIPrintf(viewer, "  Auto truncate start level=%"PetscInt_FMT", with tolerance %.2e \n", input_int, input_real);      
      }

      ierr =  PCAIRGetRDrop(pc, &input_real);
      ierr =  PCAIRGetADrop(pc, &input_real_two);
      PetscViewerASCIIPrintf(viewer, "  A drop tolerance=%f, R drop tolerance %f \n", input_real_two, input_real);
      ierr =  PCAIRGetALump(pc, &flg);
      if (flg) PetscViewerASCIIPrintf(viewer, "  Lumping \n");
      ierr =  PCAIRGetReuseSparsity(pc, &flg);
      if (flg) PetscViewerASCIIPrintf(viewer, "  Reusing sparsity during setup \n");    
      ierr =  PCAIRGetReusePolyCoeffs(pc, &flg);
      if (flg) PetscViewerASCIIPrintf(viewer, "  Reusing gmres polynomial coefficients during setup \n");         

      ierr =  PCAIRGetProcessorAgglom(pc, &flg);
      ierr =  PCAIRGetProcessorAgglomRatio(pc, &input_real);
      ierr =  PCAIRGetProcessorAgglomFactor(pc, &input_int);
      ierr =  PCAIRGetProcessEqLimit(pc, &input_int_two);
      if (flg) PetscViewerASCIIPrintf(viewer, "  Processor agglomeration with factor=%"PetscInt_FMT", ratio %f and eq limit=%"PetscInt_FMT" \n", input_int, input_real, input_int_two);      
      ierr =  PCAIRGetSubcomm(pc, &flg);
      if (flg) PetscViewerASCIIPrintf(viewer, "  Polynomial coefficients calculated on subcomm \n");      
      
      ierr =  PCAIRGetCFSplittingType(pc, &cf_type);
      ierr =  PCAIRGetStrongThreshold(pc, &input_real);
      ierr =  PCAIRGetDDCFraction(pc, &input_real_two);
      ierr =  PCAIRGetMaxLubySteps(pc, &input_int_two);
      if (cf_type == CF_PMISR_DDC)
      {
         PetscViewerASCIIPrintf(viewer, "  CF splitting algorithm=PMISR_DDC \n");
      }
      else if (cf_type == CF_PMIS)
      {
         PetscViewerASCIIPrintf(viewer, "  CF splitting algorithm=PMIS \n");
      }
      else if (cf_type == CF_PMIS_DIST2)
      {
         PetscViewerASCIIPrintf(viewer, "  CF splitting algorithm=PMIS_DIST2 \n");         
      }      
      else if (cf_type == CF_AGG)
      {
         PetscViewerASCIIPrintf(viewer, "  CF splitting algorithm=AGG \n");         
      }
         PetscViewerASCIIPrintf(viewer, "    %"PetscInt_FMT" Luby steps \n      Strong threshold=%f, DDC fraction=%f \n", \
                  input_int_two, input_real, input_real_two);            
      
      ierr =  PCAIRGetFullSmoothingUpAndDown(pc, &flg);
      if (flg) 
      {
         PetscViewerASCIIPrintf(viewer, "  Full smoothing up & down \n");   
         ierr =  PCAIRGetInverseType(pc, &input_type);
         ierr =  PCAIRGetPolyOrder(pc, &input_int_two);
         ierr =  PCAIRGetInverseSparsityOrder(pc, &input_int_three);
         ierr =  PCAIRGetMatrixFreePolys(pc, &flg);

         // What type of inverse
         if (input_type == PFLAREINV_POWER)
         {
            PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, power basis, order %"PetscInt_FMT" \n", input_int_two);
         }
         else if (input_type == PFLAREINV_ARNOLDI)
         {
            PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, arnoldi basis, order %"PetscInt_FMT" \n", input_int_two);
         }
         else if (input_type == PFLAREINV_NEWTON)
         {
            PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, newton basis with extra roots, order %"PetscInt_FMT" \n", input_int_two);      
         }
         else if (input_type == PFLAREINV_NEWTON_NO_EXTRA)
         {
            PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, newton basis without extra roots, order %"PetscInt_FMT" \n", input_int_two);             
         }
         else if (input_type == PFLAREINV_SAI)
         {
            PetscViewerASCIIPrintf(viewer, "    SAI \n");      
         }
         else if (input_type == PFLAREINV_ISAI)
         {
            PetscViewerASCIIPrintf(viewer, "    ISAI \n");      
         }      
         else if (input_type == PFLAREINV_NEUMANN)
         {
            PetscViewerASCIIPrintf(viewer, "    Neumann polynomial, order %"PetscInt_FMT" \n", input_int_two);      
         }     
         else if (input_type == PFLAREINV_WJACOBI)
         {
            PetscViewerASCIIPrintf(viewer, "    Weighted Jacobi \n");      
         }
         else if (input_type == PFLAREINV_JACOBI)
         {
            PetscViewerASCIIPrintf(viewer, "    Unweighted Jacobi \n");      
         }                  
         if (input_type != PFLAREINV_WJACOBI && input_type != PFLAREINV_JACOBI)
         {
            // If matrix-free or not
            if (flg)
            {
               PetscViewerASCIIPrintf(viewer, "      matrix-free inverse \n");
            }
            else
            {
               // Only print out if the sparsity is less than the poly order
               if (input_int_three < input_int_two)
               {
                  PetscViewerASCIIPrintf(viewer, "      assembled inverse, sparsity order %"PetscInt_FMT"\n", input_int_three);
               }
               else
               {
                  PetscViewerASCIIPrintf(viewer, "      assembled inverse\n");
               }                 
            }
         }             
      }
      else
      {
         ierr =  PCAIRGetMaxitsAff(pc, &input_int);
         ierr =  PCAIRGetOneCSmooth(pc, &flg_two);
         if (flg_two) 
         {
            PetscViewerASCIIPrintf(viewer, "  Up FC smoothing with number of F smooths=%"PetscInt_FMT" and number of C smooths=1: \n", input_int);      
         }
         else
         {
            PetscViewerASCIIPrintf(viewer, "  Up F smoothing with number of F smooths=%"PetscInt_FMT": \n", input_int);      
         }
         ierr =  PCAIRGetInverseType(pc, &input_type);
         ierr =  PCAIRGetPolyOrder(pc, &input_int_two);
         ierr =  PCAIRGetInverseSparsityOrder(pc, &input_int_three);
         ierr =  PCAIRGetMatrixFreePolys(pc, &flg);

         // What type of inverse
         if (input_type == PFLAREINV_POWER)
         {
            PetscViewerASCIIPrintf(viewer, "    F smooth: GMRES polynomial, power basis, order %"PetscInt_FMT" \n", input_int_two);
         }
         else if (input_type == PFLAREINV_ARNOLDI)
         {
            PetscViewerASCIIPrintf(viewer, "    F smooth: GMRES polynomial, arnoldi basis, order %"PetscInt_FMT" \n", input_int_two);
         }
         else if (input_type == PFLAREINV_NEWTON)
         {
            PetscViewerASCIIPrintf(viewer, "    F smooth: GMRES polynomial, newton basis with extra roots, order %"PetscInt_FMT" \n", input_int_two);      
         }
         else if (input_type == PFLAREINV_NEWTON_NO_EXTRA)
         {
            PetscViewerASCIIPrintf(viewer, "    F smooth: GMRES polynomial, newton basis without extra roots, order %"PetscInt_FMT" \n", input_int_two);             
         }
         else if (input_type == PFLAREINV_SAI)
         {
            PetscViewerASCIIPrintf(viewer, "    F smooth: SAI \n");      
         }
         else if (input_type == PFLAREINV_ISAI)
         {
            PetscViewerASCIIPrintf(viewer, "    F smooth: ISAI \n");      
         }      
         else if (input_type == PFLAREINV_NEUMANN)
         {
            PetscViewerASCIIPrintf(viewer, "    F smooth: Neumann polynomial, order %"PetscInt_FMT" \n", input_int_two);      
         }     
         else if (input_type == PFLAREINV_WJACOBI)
         {
            PetscViewerASCIIPrintf(viewer, "    F smooth: Weighted Jacobi \n");      
         }
         else if (input_type == PFLAREINV_JACOBI)
         {
            PetscViewerASCIIPrintf(viewer, "    F smooth: Unweighted Jacobi \n");      
         }                  
         if (input_type != PFLAREINV_WJACOBI && input_type != PFLAREINV_JACOBI)
         {
            // If matrix-free or not
            if (flg)
            {
               PetscViewerASCIIPrintf(viewer, "      matrix-free inverse \n");
            }
            else
            {
               // Only print out if the sparsity is less than the poly order
               if (input_int_three < input_int_two)
               {               
                  PetscViewerASCIIPrintf(viewer, "      assembled inverse, sparsity order %"PetscInt_FMT"\n", input_int_three);
               }
               else
               {
                  PetscViewerASCIIPrintf(viewer, "      assembled inverse\n");
               }                
            }
         }  

         // If C smoothing
         if (flg_two) {

            ierr =  PCAIRGetCInverseType(pc, &input_type);
            ierr =  PCAIRGetCPolyOrder(pc, &input_int_two);
            ierr =  PCAIRGetCInverseSparsityOrder(pc, &input_int_three);
            ierr =  PCAIRGetMatrixFreePolys(pc, &flg);

            // What type of inverse
            if (input_type == PFLAREINV_POWER)
            {
               PetscViewerASCIIPrintf(viewer, "    C smooth: GMRES polynomial, power basis, order %"PetscInt_FMT" \n", input_int_two);
            }
            else if (input_type == PFLAREINV_ARNOLDI)
            {
               PetscViewerASCIIPrintf(viewer, "    C smooth: GMRES polynomial, arnoldi basis, order %"PetscInt_FMT" \n", input_int_two);
            }
            else if (input_type == PFLAREINV_NEWTON)
            {
               PetscViewerASCIIPrintf(viewer, "    C smooth: GMRES polynomial, newton basis with extra roots, order %"PetscInt_FMT" \n", input_int_two);      
            }
            else if (input_type == PFLAREINV_NEWTON_NO_EXTRA)
            {
               PetscViewerASCIIPrintf(viewer, "    C smooth: GMRES polynomial, newton basis without extra roots, order %"PetscInt_FMT" \n", input_int_two);                
            }
            else if (input_type == PFLAREINV_SAI)
            {
               PetscViewerASCIIPrintf(viewer, "    C smooth: SAI \n");      
            }
            else if (input_type == PFLAREINV_ISAI)
            {
               PetscViewerASCIIPrintf(viewer, "    C smooth: ISAI \n");      
            }      
            else if (input_type == PFLAREINV_NEUMANN)
            {
               PetscViewerASCIIPrintf(viewer, "    C smooth: Neumann polynomial, order %"PetscInt_FMT" \n", input_int_two);      
            }     
            else if (input_type == PFLAREINV_WJACOBI)
            {
               PetscViewerASCIIPrintf(viewer, "    C smooth: Weighted Jacobi \n");      
            }
            else if (input_type == PFLAREINV_JACOBI)
            {
               PetscViewerASCIIPrintf(viewer, "    C smooth: Unweighted Jacobi \n");      
            }                  
            if (input_type != PFLAREINV_WJACOBI && input_type != PFLAREINV_JACOBI)
            {
               // If matrix-free or not
               if (flg)
               {
                  PetscViewerASCIIPrintf(viewer, "      matrix-free inverse \n");
               }
               else
               {
                  // Only print out if the sparsity is less than the poly order
                  if (input_int_three < input_int_two)
                  {                  
                     PetscViewerASCIIPrintf(viewer, "      assembled inverse, sparsity order %"PetscInt_FMT"\n", input_int_three);
                  }
                  else
                  {
                     PetscViewerASCIIPrintf(viewer, "      assembled inverse\n");
                  }                  
               }
            }  
         }           
      }        

      PetscViewerASCIIPrintf(viewer, "  Grid transfer operators: \n");      
      ierr =  PCAIRGetZType(pc, &z_type);
      ierr =  PCAIRGetLairDistance(pc, &input_int_two);
      if (z_type == AIR_Z_PRODUCT)
      {
         PetscViewerASCIIPrintf(viewer, "    Mat-Mat product used to form Z \n");      
      }
      else if (z_type == AIR_Z_LAIR)
      {
         PetscViewerASCIIPrintf(viewer, "    lAIR Z, distance %"PetscInt_FMT" \n", input_int_two);            
      }
      else
      {
         PetscViewerASCIIPrintf(viewer, "    lAIR SAI Z, distance %"PetscInt_FMT" \n", input_int_two);            
      }
      ierr =  PCAIRGetStrongRThreshold(pc, &input_real);
      PetscViewerASCIIPrintf(viewer, "    Strong R threshold=%f \n", input_real);            

      ierr =  PCAIRGetConstrainZ(pc, &flg);
      if (flg) 
      {
         PetscViewerASCIIPrintf(viewer, "    Constraints applied to restrictor \n");      
      }     

      ierr =  PCAIRGetSymmetric(pc, &flg);
      if (flg) 
      {
         PetscViewerASCIIPrintf(viewer, "    Approximate ideal prolongator \n");      
         PetscViewerASCIIPrintf(viewer, "      Symmetric - transpose of restrictor \n");      
      }        
      else
      {
         ierr =  PCAIRGetOnePointClassicalProlong(pc, &flg);
         if (flg) 
         {
            PetscViewerASCIIPrintf(viewer, "    One point classical prolongator \n");      
         }
         else
         {
            PetscViewerASCIIPrintf(viewer, "    Approximate ideal prolongator \n");      
         }       
      }

      ierr =  PCAIRGetConstrainW(pc, &flg);
      if (flg) 
      {
         PetscViewerASCIIPrintf(viewer, "      Constraints applied to prolongator \n");      
      }      

      PetscViewerASCIIPrintf(viewer, "  Coarse grid solver: \n");      
      ierr =  PCAIRGetCoarsestInverseType(pc, &input_type);
      ierr =  PCAIRGetCoarsestPolyOrder(pc, &input_int_two);
      ierr =  PCAIRGetCoarsestInverseSparsityOrder(pc, &input_int_three);
      ierr =  PCAIRGetCoarsestMatrixFreePolys(pc, &flg);

      // What type of inverse
      if (input_type == PFLAREINV_POWER)
      {
         PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, power basis, order %"PetscInt_FMT" \n", input_int_two);
      }
      else if (input_type == PFLAREINV_ARNOLDI)
      {
         PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, arnoldi basis, order %"PetscInt_FMT" \n", input_int_two);
      }
      else if (input_type == PFLAREINV_NEWTON)
      {
         PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, newton basis with extra roots, order %"PetscInt_FMT" \n", input_int_two);      
      }
      else if (input_type == PFLAREINV_NEWTON_NO_EXTRA)
      {
         PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, newton basis without extra roots, order %"PetscInt_FMT" \n", input_int_two);          
      }
      else if (input_type == PFLAREINV_SAI)
      {
         PetscViewerASCIIPrintf(viewer, "    SAI \n");      
      }
      else if (input_type == PFLAREINV_ISAI)
      {
         PetscViewerASCIIPrintf(viewer, "    ISAI \n");      
      }      
      else if (input_type == PFLAREINV_NEUMANN)
      {
         PetscViewerASCIIPrintf(viewer, "    Neumann polynomial, order %"PetscInt_FMT" \n", input_int_two);      
      }     
      else if (input_type == PFLAREINV_WJACOBI)
      {
         PetscViewerASCIIPrintf(viewer, "    Weighted Jacobi \n");      
      }
      else if (input_type == PFLAREINV_JACOBI)
      {
         PetscViewerASCIIPrintf(viewer, "    Unweighted Jacobi \n");      
      }                  
      if (input_type != PFLAREINV_WJACOBI && input_type != PFLAREINV_JACOBI)
      {
         // If matrix-free or not
         if (flg)
         {
            PetscViewerASCIIPrintf(viewer, "    matrix-free inverse \n");
         }
         else
         {
            // Only print out if the sparsity is less than the poly order
            if (input_int_three < input_int_two)
            {            
               PetscViewerASCIIPrintf(viewer, "    assembled inverse, sparsity order %"PetscInt_FMT"\n", input_int_three);
            }
            else
            {
               PetscViewerASCIIPrintf(viewer, "    assembled inverse\n");
            }
         }
      }      
     
      ierr =  PCAIRGetCoarsestSubcomm(pc, &flg);
      if (flg) PetscViewerASCIIPrintf(viewer, "    Polynomial coefficients calculated on subcomm \n");            

      PetscViewerASCIIPrintf(viewer, "The underlying PCMG: \n");      
      // Call the underlying pcshell view
      PCView(*pc_air_shell, viewer);
   }
   PetscFunctionReturn(ierr);
}

// ~~~~~~~~~~~~~~~~

// Creates the structure we need for this PC
PETSC_EXTERN PetscErrorCode PCCreate_AIR(PC pc)
{
   PetscFunctionBegin;

   // Now we call petsc fortran routines from this PC
   // so we have to have made sure this is called
   PetscInitializeFortran();
      
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
   PetscNew(&pc_air_shell);
   MPI_Comm comm;

   // We need to create our new pcshell
   PetscObjectGetComm((PetscObject)pc, &comm);
   PCCreate(comm, pc_air_shell);

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

   PetscFunctionReturn(0);
}

// ~~~~~~~~~~~~~~~~

// Registers the PC type
PETSC_EXTERN void PCRegister_AIR()
{
   PCRegister("air", PCCreate_AIR);
}