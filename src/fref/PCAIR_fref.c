/*
  The reference PC type air_fref, registered only in PFLARE_PORT_DEBUG builds
  This is src/PCAIR.c as it was before the AIR hierarchy was ported to C (stage 7
  of docs/dev/fortran_removal_plan.md), a wrapper around a PCShell implemented by
  the Fortran reference implementations in src/fref, reached through the _fref
  bindings in src/fref/C_Fortran_Bindings.F90 and src/fref/PCAIR_C_Fortran_Bindings.F90,
  so tests/port_compare_pc.c can compare the air PC against it
  Only the names have changed: the type name, the statics, the bindings and the
  get/set routines have an _fref suffix (PCAIRGetNumLevels_fref etc, which take an
  air_fref PC), the C API that was already C (compute_cf_splitting,
  compute_diag_dom_submatrix, remove_from_sparse_match) and the library
  registration are left out, as is the documentation
*/

// Include the petsc header files
#include <petsc/private/pcimpl.h>
#include "pflare.h"
#include <string.h>

// Defined in src/fref/C_Fortran_Bindings.F90
PETSC_EXTERN void PCReset_AIR_Shell_c_fref(PC *pc);
PETSC_EXTERN void create_pc_air_data_c_fref(void **pc_air_data);
PETSC_EXTERN void create_pc_air_shell_c_fref(void **pc_air_data, PC *pc);
PETSC_EXTERN void pcair_shell_block_matapply_c_fref(PC *pc, Mat *X, Mat *Y, int *applied, int *error_code);
PETSC_EXTERN void pcair_shell_get_pcmg_c_fref(PC *pc, PC *pcmg);
// Defined in src/fref/PCAIR_C_Fortran_Bindings.F90
// External users should use the get/set routines without _c which have 
// PetscErrorCode defined as return type, those routines are defined below this
// e.g., PCAIRGetPrintStatsTimings_fref 
PETSC_EXTERN void PCAIRGetPrintStatsTimings_c_fref(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetMaxLevels_c_fref(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetCoarseEqLimit_c_fref(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetAutoTruncateStartLevel_c_fref(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetAutoTruncateTol_c_fref(PC *pc, PetscReal *input_real);
PETSC_EXTERN void PCAIRGetNumLevels_c_fref(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetProcessorAgglom_c_fref(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetProcessorAgglomRatio_c_fref(PC *pc, PetscReal *input_real);
PETSC_EXTERN void PCAIRGetProcessorAgglomFactor_c_fref(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetProcessEqLimit_c_fref(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetSubcomm_c_fref(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetStrongThreshold_c_fref(PC *pc, PetscReal *input_real);
PETSC_EXTERN void PCAIRGetDDCIts_c_fref(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetDDCFraction_c_fref(PC *pc, PetscReal *input_real);
PETSC_EXTERN void PCAIRGetCFSplittingType_c_fref(PC *pc, CFSplittingType *cf_splitting_type);
PETSC_EXTERN void PCAIRGetMaxLubySteps_c_fref(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetSmoothType_c_fref(PC *pc, char* input_string);
PETSC_EXTERN void PCAIRGetDiagScalePolys_c_fref(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetMatrixFreePolys_c_fref(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetOnePointClassicalProlong_c_fref(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetFullSmoothingUpAndDown_c_fref(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetSymmetric_c_fref(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetConstrainW_c_fref(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetConstrainZ_c_fref(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetImproveWIts_c_fref(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetImproveZIts_c_fref(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetStrongRThreshold_c_fref(PC *pc, PetscReal *input_real);
PETSC_EXTERN void PCAIRGetInverseType_c_fref(PC *pc, PCPFLAREINVType *inverse_type);
PETSC_EXTERN void PCAIRGetCInverseType_c_fref(PC *pc, PCPFLAREINVType *inverse_type);
PETSC_EXTERN void PCAIRGetZType_c_fref(PC *pc, PCAIRZType *z_type);
PETSC_EXTERN void PCAIRGetPolyOrder_c_fref(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetLairDistance_c_fref(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetInverseSparsityOrder_c_fref(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetCPolyOrder_c_fref(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetCInverseSparsityOrder_c_fref(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetCoarsestInverseType_c_fref(PC *pc, PCPFLAREINVType *inverse_type);
PETSC_EXTERN void PCAIRGetCoarsestPolyOrder_c_fref(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetCoarsestInverseSparsityOrder_c_fref(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetCoarsestMatrixFreePolys_c_fref(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetCoarsestDiagScalePolys_c_fref(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetCoarsestSubcomm_c_fref(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetRDrop_c_fref(PC *pc, PetscReal *input_real);
PETSC_EXTERN void PCAIRGetADrop_c_fref(PC *pc, PetscReal *input_real);
PETSC_EXTERN void PCAIRGetALump_c_fref(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetReuseSparsity_c_fref(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetReusePolyCoeffs_c_fref(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetReuseAmount_c_fref(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetPolyCoeffs_c_fref(PC *pc, PetscInt petsc_level, int which_inverse, PetscReal **coeffs_ptr, PetscInt *row_size, PetscInt *col_size);
PETSC_EXTERN void PCAIRGetGridComplexity_c_fref(PC *pc, PetscReal *complexity);
PETSC_EXTERN void PCAIRGetOperatorComplexity_c_fref(PC *pc, PetscReal *complexity);
PETSC_EXTERN void PCAIRGetCycleComplexity_c_fref(PC *pc, PetscReal *complexity);
PETSC_EXTERN void PCAIRGetStorageComplexity_c_fref(PC *pc, PetscReal *complexity);
PETSC_EXTERN void PCAIRGetReuseStorageComplexity_c_fref(PC *pc, PetscReal *complexity);

// Setters
PETSC_EXTERN void PCAIRSetPrintStatsTimings_c_fref(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetMaxLevels_c_fref(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetCoarseEqLimit_c_fref(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetAutoTruncateStartLevel_c_fref(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetAutoTruncateTol_c_fref(PC *pc, PetscReal input_real);
PETSC_EXTERN void PCAIRSetProcessorAgglom_c_fref(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetProcessorAgglomRatio_c_fref(PC *pc, PetscReal input_real);
PETSC_EXTERN void PCAIRSetProcessorAgglomFactor_c_fref(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetProcessEqLimit_c_fref(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetSubcomm_c_fref(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetStrongThreshold_c_fref(PC *pc, PetscReal input_real);
PETSC_EXTERN void PCAIRSetDDCIts_c_fref(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetDDCFraction_c_fref(PC *pc, PetscReal input_real);
PETSC_EXTERN void PCAIRSetCFSplittingType_c_fref(PC *pc, CFSplittingType cf_splitting_type);
PETSC_EXTERN void PCAIRSetMaxLubySteps_c_fref(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetSmoothType_c_fref(PC *pc, const char* input_string);
PETSC_EXTERN void PCAIRSetDiagScalePolys_c_fref(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetMatrixFreePolys_c_fref(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetOnePointClassicalProlong_c_fref(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetFullSmoothingUpAndDown_c_fref(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetSymmetric_c_fref(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetConstrainW_c_fref(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetConstrainZ_c_fref(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetImproveWIts_c_fref(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetImproveZIts_c_fref(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetStrongRThreshold_c_fref(PC *pc, PetscReal input_real);
PETSC_EXTERN void PCAIRSetInverseType_c_fref(PC *pc, PCPFLAREINVType inverse_type);
PETSC_EXTERN void PCAIRSetZType_c_fref(PC *pc, PCAIRZType z_type);
PETSC_EXTERN void PCAIRSetLairDistance_c_fref(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetPolyOrder_c_fref(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetInverseSparsityOrder_c_fref(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetCInverseType_c_fref(PC *pc, PCPFLAREINVType inverse_type);
PETSC_EXTERN void PCAIRSetCPolyOrder_c_fref(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetCInverseSparsityOrder_c_fref(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetCoarsestInverseType_c_fref(PC *pc, PCPFLAREINVType inverse_type);
PETSC_EXTERN void PCAIRSetCoarsestPolyOrder_c_fref(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetCoarsestInverseSparsityOrder_c_fref(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetCoarsestMatrixFreePolys_c_fref(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetCoarsestDiagScalePolys_c_fref(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetCoarsestSubcomm_c_fref(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetRDrop_c_fref(PC *pc, PetscReal input_real);
PETSC_EXTERN void PCAIRSetADrop_c_fref(PC *pc, PetscReal input_real);
PETSC_EXTERN void PCAIRSetALump_c_fref(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetReuseSparsity_c_fref(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetReusePolyCoeffs_c_fref(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetReuseAmount_c_fref(PC *pc, PetscInt input_int);

// ~~~~~~~~~~~~~

static PetscErrorCode PCReset_AIR_c_fref(PC pc)
{
   PetscFunctionBegin;

   PC *pc_air_shell = (PC *)pc->data;
   // Call the underlying reset - this won't touch the context 
   // in our pcshell though as pcshell doesn't offer a way to 
   // give a custom reset function
   PetscCall(PCReset(*pc_air_shell));

   // So now we manually reset the underlying data
   PCReset_AIR_Shell_c_fref(pc_air_shell);  

   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_AIR_c_fref(PC pc, Vec x, Vec y)
{
   PetscFunctionBegin;
   PC *pc_air_shell = (PC *)pc->data;

   // The shell tracks the pmat state itself, so it must also see the
   // reusepreconditioner flag: when it is set the outer PCSetUp is skipped
   // entirely, but the PCApply below calls PCSetUp on the shell, which would
   // otherwise rebuild the hierarchy after any change to the pmat
   // This mirrors petsc semantics: once set up, a frozen pc stays frozen
   // even if the sparsity pattern changes
   PetscCall(PCSetReusePreconditioner(*pc_air_shell, pc->reusepreconditioner));

   // Just call the underlying pcshell apply
   PetscCall(PCApply(*pc_air_shell, x, y));
   PetscFunctionReturn(PETSC_SUCCESS);
}

// Transposed apply - this applies the exact transpose of what PCApply applies, so
// the two are adjoints and KSPSolveTranspose works
static PetscErrorCode PCApplyTranspose_AIR_c_fref(PC pc, Vec x, Vec y)
{
   PetscFunctionBegin;
   PC *pc_air_shell = (PC *)pc->data;

   // Same reasoning as in PCApply_AIR_c_fref - the shell tracks the pmat state itself
   // so it has to see the reusepreconditioner flag
   PetscCall(PCSetReusePreconditioner(*pc_air_shell, pc->reusepreconditioner));

   // Just call the underlying pcshell transposed apply
   PetscCall(PCApplyTranspose(*pc_air_shell, x, y));
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

// Multi-RHS apply: applies the air multigrid to a whole block of dense right
// hand sides, so KSPMatSolve does real sparse matrix by dense matrix products
// throughout the cycle rather than PETSc's column-by-column PCApply fallback
static PetscErrorCode PCMatApply_AIR_c_fref(PC pc, Mat X, Mat Y)
{
   PetscFunctionBegin;
   PC *pc_air_shell = (PC *)pc->data;
   int applied = 0, error_code = 0;

   // Same reasoning as in PCApply_AIR_c_fref - the shell tracks the pmat state itself
   // so it has to see the reusepreconditioner flag
   PetscCall(PCSetReusePreconditioner(*pc_air_shell, pc->reusepreconditioner));

   // PCApply on the shell would do this for us, but the block path below goes
   // straight to the underlying PCMG so we have to set the shell up ourselves
   PetscCall(PCSetUp(*pc_air_shell));

   pcair_shell_block_matapply_c_fref(pc_air_shell, &X, &Y, &applied, &error_code);
   PetscCall((PetscErrorCode)error_code);

   if (applied) {
      PetscCall(PetscInfo(pc, "PCMatApply used the block air multigrid\n"));
   } else {
      // Anything we can't do blockwise is applied column-by-column
      PetscInt n_cols, j;
      Vec cx, cy;
      PetscCall(PetscInfo(pc, "PCMatApply solving column by column\n"));
      PetscCall(MatGetSize(X, NULL, &n_cols));
      for (j = 0; j < n_cols; j++) {
         PetscCall(MatDenseGetColumnVecRead(X, j, &cx));
         PetscCall(MatDenseGetColumnVecWrite(Y, j, &cy));
         PetscCall(PCApply(*pc_air_shell, cx, cy));
         PetscCall(MatDenseRestoreColumnVecWrite(Y, j, &cy));
         PetscCall(MatDenseRestoreColumnVecRead(X, j, &cx));
      }
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_AIR_c_fref(PC pc)
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

static PetscErrorCode PCSetUp_AIR_c_fref(PC pc)
{
   PetscFunctionBegin;
   PC *pc_air_shell = (PC *)pc->data;
   PC pcmg;
   const char *prefix;

   // The pc_air_shell doesn't have any operators yet
   // as they are not available yet in pccreate
   // so we have to set them
   PetscCall(PCSetOperators(*pc_air_shell, pc->mat, pc->pmat));

   // Pass our options prefix down to the shell and the underlying PCMG
   // PCMGSetLevels uses the prefix of the PCMG when it names the level KSPs
   // and PCSetUp_MG then calls KSPSetFromOptions on them, so this is what
   // makes -prefix_mg_coarse_* and -prefix_mg_levels_* reach this PCAIR only
   // rather than the unprefixed -mg_coarse_* hitting every PCAIR
   PetscCall(PCGetOptionsPrefix(pc, &prefix));
   PetscCall(PCSetOptionsPrefix(*pc_air_shell, prefix));
   pcair_shell_get_pcmg_c_fref(pc_air_shell, &pcmg);
   PetscCall(PCSetOptionsPrefix(pcmg, prefix));

   // Keep the shell's reusepreconditioner flag in sync - if the flag has
   // been unset after a frozen solve the shell must be allowed
   // to rebuild again
   PetscCall(PCSetReusePreconditioner(*pc_air_shell, pc->reusepreconditioner));

   // Now we should be able to call the pcshell setup
   // that builds the air hierarchy
   PetscCall(PCSetUp(*pc_air_shell));
   PetscFunctionReturn(PETSC_SUCCESS);
}

// Get the underlying PCShell
PETSC_EXTERN PetscErrorCode c_PCAIRGetPCShell_fref(PC *pc, PC *pc_air_shell)
{
   PetscFunctionBegin;
   PC *pc_shell = (PC *)(*pc)->data;
   *pc_air_shell = *pc_shell;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// The PCAIRGet_fref/Set routines below forward to Fortran routines that cast
// pc->data unconditionally, so the type must be checked here on the way in
static PetscErrorCode PCAIRCheckType_fref(PC pc)
{
   PetscBool is_air;

   PetscFunctionBegin;
   PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
   PetscCall(PetscObjectTypeCompare((PetscObject)pc, "air_fref", &is_air));
   PetscCheck(is_air, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "PC is not of type air_fref");
   PetscFunctionReturn(PETSC_SUCCESS);
}

// Returns a pointer to the polynomial coefficients stored internally by PCAIR for a given
// level and inverse, erroring if there are none. Used by both PCAIRGetPolyCoeffs_fref and
// PCAIRSetPolyCoeffs_fref; petsc_level is ignored for COEFFS_INV_COARSE
static PetscErrorCode PCAIRGetPolyCoeffsPointer_Private_fref(PC pc, PetscInt petsc_level, int which_inverse, PetscReal **coeffs_ptr, PetscInt *row_size, PetscInt *col_size)
{
   PetscInt num_levels;

   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PetscCheck(which_inverse == COEFFS_INV_AFF || which_inverse == COEFFS_INV_AFF_DROPPED || which_inverse == COEFFS_INV_ACC || which_inverse == COEFFS_INV_COARSE, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unknown which_inverse %d", which_inverse);
   // The number of levels is -1 before PCSetUp() and after the hierarchy is reset
   PCAIRGetNumLevels_c_fref(&pc, &num_levels);
   PetscCheck(num_levels > 0, PETSC_COMM_SELF, PETSC_ERR_ORDER, "PCAIR has no multigrid hierarchy; call PCSetUp() first");
   PetscCheck(which_inverse == COEFFS_INV_COARSE || (petsc_level >= 0 && petsc_level < num_levels), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "petsc_level %" PetscInt_FMT " out of range [0, %" PetscInt_FMT "]", petsc_level, num_levels - 1);
   PCAIRGetPolyCoeffs_c_fref(&pc, petsc_level, which_inverse, coeffs_ptr, row_size, col_size);
   PetscCheck(*coeffs_ptr, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "PCAIR stores no polynomial coefficients for which_inverse %d on petsc_level %" PetscInt_FMT "; that inverse is not built as a polynomial there with the current options", which_inverse, petsc_level);
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~
// Now all the get/set routines for options
// Most of the explanation are in the comments above the set routines
// ~~~~~~~~~~~~~~~~~~~~~

// Get routines

PETSC_EXTERN PetscErrorCode PCAIRGetPrintStatsTimings_fref(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetPrintStatsTimings_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetMaxLevels_fref(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetMaxLevels_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCoarseEqLimit_fref(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetCoarseEqLimit_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetAutoTruncateStartLevel_fref(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetAutoTruncateStartLevel_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetAutoTruncateTol_fref(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetAutoTruncateTol_c_fref(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetNumLevels_fref(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetNumLevels_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetProcessorAgglom_fref(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetProcessorAgglom_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetProcessorAgglomRatio_fref(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetProcessorAgglomRatio_c_fref(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetProcessorAgglomFactor_fref(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetProcessorAgglomFactor_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetProcessEqLimit_fref(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetProcessEqLimit_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetSubcomm_fref(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetSubcomm_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetStrongThreshold_fref(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetStrongThreshold_c_fref(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetDDCFraction_fref(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetDDCFraction_c_fref(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCFSplittingType_fref(PC pc, CFSplittingType *cf_splitting_type)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetCFSplittingType_c_fref(&pc, cf_splitting_type);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetDDCIts_fref(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetDDCIts_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetMaxLubySteps_fref(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetMaxLubySteps_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetSmoothType_fref(PC pc, char *input_string)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetSmoothType_c_fref(&pc, input_string);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetDiagScalePolys_fref(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetDiagScalePolys_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetMatrixFreePolys_fref(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetMatrixFreePolys_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetOnePointClassicalProlong_fref(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetOnePointClassicalProlong_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetFullSmoothingUpAndDown_fref(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetFullSmoothingUpAndDown_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetSymmetric_fref(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetSymmetric_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetConstrainW_fref(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetConstrainW_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetConstrainZ_fref(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetConstrainZ_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetImproveWIts_fref(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetImproveWIts_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetImproveZIts_fref(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetImproveZIts_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetStrongRThreshold_fref(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetStrongRThreshold_c_fref(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetInverseType_fref(PC pc, PCPFLAREINVType *inverse_type)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetInverseType_c_fref(&pc, inverse_type);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCInverseType_fref(PC pc, PCPFLAREINVType *inverse_type)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetCInverseType_c_fref(&pc, inverse_type);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetZType_fref(PC pc, PCAIRZType *z_type)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetZType_c_fref(&pc, z_type);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetPolyOrder_fref(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetPolyOrder_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetLairDistance_fref(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetLairDistance_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetInverseSparsityOrder_fref(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetInverseSparsityOrder_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCPolyOrder_fref(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetCPolyOrder_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCInverseSparsityOrder_fref(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetCInverseSparsityOrder_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestInverseType_fref(PC pc, PCPFLAREINVType *inverse_type)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetCoarsestInverseType_c_fref(&pc, inverse_type);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestPolyOrder_fref(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetCoarsestPolyOrder_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestInverseSparsityOrder_fref(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetCoarsestInverseSparsityOrder_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestMatrixFreePolys_fref(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetCoarsestMatrixFreePolys_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestDiagScalePolys_fref(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetCoarsestDiagScalePolys_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestSubcomm_fref(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetCoarsestSubcomm_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetRDrop_fref(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetRDrop_c_fref(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetADrop_fref(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetADrop_c_fref(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetALump_fref(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetALump_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetReuseSparsity_fref(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetReuseSparsity_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetReusePolyCoeffs_fref(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetReusePolyCoeffs_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetPolyCoeffs_fref(PC pc, PetscInt petsc_level, int which_inverse, PetscReal **coeffs_ptr, PetscInt *row_size, PetscInt *col_size)
{
   PetscFunctionBegin;
   PetscAssertPointer(coeffs_ptr, 4);
   PetscAssertPointer(row_size, 5);
   PetscAssertPointer(col_size, 6);
   PetscCall(PCAIRGetPolyCoeffsPointer_Private_fref(pc, petsc_level, which_inverse, coeffs_ptr, row_size, col_size));
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetGridComplexity_fref(PC pc, PetscReal *complexity)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetGridComplexity_c_fref(&pc, complexity);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetOperatorComplexity_fref(PC pc, PetscReal *complexity)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetOperatorComplexity_c_fref(&pc, complexity);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetCycleComplexity_fref(PC pc, PetscReal *complexity)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetCycleComplexity_c_fref(&pc, complexity);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetStorageComplexity_fref(PC pc, PetscReal *complexity)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetStorageComplexity_c_fref(&pc, complexity);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetReuseStorageComplexity_fref(PC pc, PetscReal *complexity)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetReuseStorageComplexity_c_fref(&pc, complexity);
   PetscFunctionReturn(PETSC_SUCCESS);
}

// Set routines

PETSC_EXTERN PetscErrorCode PCAIRSetPrintStatsTimings_fref(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   // No need to reset if this changes
   PCAIRSetPrintStatsTimings_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PCAIRSetMaxLevels_fref(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetMaxLevels_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetCoarseEqLimit_fref(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetCoarseEqLimit_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetAutoTruncateStartLevel_fref(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetAutoTruncateStartLevel_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetAutoTruncateTol_fref(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetAutoTruncateTol_c_fref(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetProcessorAgglom_fref(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetProcessorAgglom_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetProcessorAgglomRatio_fref(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetProcessorAgglomRatio_c_fref(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetProcessorAgglomFactor_fref(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetProcessorAgglomFactor_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetProcessEqLimit_fref(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetProcessEqLimit_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetSubcomm_fref(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   // No need to reset if this changes
   PCAIRSetSubcomm_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetStrongThreshold_fref(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetStrongThreshold_c_fref(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetDDCIts_fref(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetDDCIts_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetDDCFraction_fref(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetDDCFraction_c_fref(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetCFSplittingType_fref(PC pc, CFSplittingType cf_splitting_type)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetCFSplittingType_c_fref(&pc, cf_splitting_type);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetMaxLubySteps_fref(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetMaxLubySteps_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetSmoothType_fref(PC pc, const char* input_string)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   // The Fortran side only stores (and reads) at most 10 characters
   PetscCheck(strlen(input_string) <= 10, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_OUTOFRANGE, "Smooth type %s has %d characters, at most 10 are supported", input_string, (int)strlen(input_string));
   PCAIRSetSmoothType_c_fref(&pc, input_string);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetDiagScalePolys_fref(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetDiagScalePolys_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetMatrixFreePolys_fref(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetMatrixFreePolys_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetOnePointClassicalProlong_fref(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetOnePointClassicalProlong_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetFullSmoothingUpAndDown_fref(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetFullSmoothingUpAndDown_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetSymmetric_fref(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetSymmetric_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetConstrainW_fref(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetConstrainW_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetConstrainZ_fref(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetConstrainZ_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetImproveWIts_fref(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetImproveWIts_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetImproveZIts_fref(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetImproveZIts_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetStrongRThreshold_fref(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetStrongRThreshold_c_fref(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetInverseType_fref(PC pc, PCPFLAREINVType inverse_type)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetInverseType_c_fref(&pc, inverse_type);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetCInverseType_fref(PC pc, PCPFLAREINVType inverse_type)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetCInverseType_c_fref(&pc, inverse_type);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetZType_fref(PC pc, PCAIRZType z_type)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetZType_c_fref(&pc, z_type);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetLairDistance_fref(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetLairDistance_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetPolyOrder_fref(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetPolyOrder_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetInverseSparsityOrder_fref(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetInverseSparsityOrder_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetCPolyOrder_fref(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetCPolyOrder_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetCInverseSparsityOrder_fref(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetCInverseSparsityOrder_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestInverseType_fref(PC pc, PCPFLAREINVType inverse_type)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetCoarsestInverseType_c_fref(&pc, inverse_type);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestPolyOrder_fref(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetCoarsestPolyOrder_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestInverseSparsityOrder_fref(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetCoarsestInverseSparsityOrder_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestMatrixFreePolys_fref(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetCoarsestMatrixFreePolys_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestDiagScalePolys_fref(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetCoarsestDiagScalePolys_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestSubcomm_fref(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   // No need to reset if this changes
   PCAIRSetCoarsestSubcomm_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetRDrop_fref(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetRDrop_c_fref(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetADrop_fref(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetADrop_c_fref(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetALump_fref(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetALump_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetReuseSparsity_fref(PC pc, PetscBool input_bool)
{  
   PetscFunctionBegin; 
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetReuseSparsity_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS); 
}
PETSC_EXTERN PetscErrorCode PCAIRSetReusePolyCoeffs_fref(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRSetReusePolyCoeffs_c_fref(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRGetReuseAmount_fref(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   PCAIRGetReuseAmount_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetReuseAmount_fref(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PetscCall(PCAIRCheckType_fref(pc));
   // The amount indexes the reuse tables in AIR_Data_Type.F90 so must be 1, 2 or 3
   PetscCheck(input_int >= 1 && input_int <= 3, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_OUTOFRANGE, "Reuse amount %" PetscInt_FMT " must be 1, 2 or 3", input_int);
   PCAIRSetReuseAmount_c_fref(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
PETSC_EXTERN PetscErrorCode PCAIRSetPolyCoeffs_fref(PC pc, PetscInt petsc_level, int which_inverse, PetscReal *coeffs_ptr, PetscInt row_size, PetscInt col_size)
{
   PetscReal *coeffs_internal;
   PetscInt   rows_internal, cols_internal;

   PetscFunctionBegin;
   PetscCall(PCAIRGetPolyCoeffsPointer_Private_fref(pc, petsc_level, which_inverse, &coeffs_internal, &rows_internal, &cols_internal));
   PetscCheck(row_size == rows_internal && col_size == cols_internal, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Coefficient array is %" PetscInt_FMT " x %" PetscInt_FMT " but PCAIR stores %" PetscInt_FMT " x %" PetscInt_FMT " for which_inverse %d on petsc_level %" PetscInt_FMT, row_size, col_size, rows_internal, cols_internal, which_inverse, petsc_level);
   PetscAssertPointer(coeffs_ptr, 4);
   PetscCall(PetscArraycpy(coeffs_internal, coeffs_ptr, rows_internal * cols_internal));
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~

static PetscErrorCode PCSetFromOptions_AIR_c_fref(PC pc, PetscOptionItems PetscOptionsObject)
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
   PetscCall(PCAIRGetPrintStatsTimings_fref(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_print_stats_timings", "Print statistics and timings", "PCAIRSetPrintStatsTimings_fref", old_flag, &flg, NULL));
   PetscCall(PCAIRSetPrintStatsTimings_fref(pc, flg));
   // ~~~~
   PetscCall(PCAIRGetSubcomm_fref(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_subcomm", "Computes polynomial coefficients on subcomm", "PCAIRSetSubcomm_fref", old_flag, &flg, NULL));
   PetscCall(PCAIRSetSubcomm_fref(pc, flg));
   // ~~~~
   PetscCall(PCAIRGetProcessorAgglom_fref(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_processor_agglom", "Processor agglomeration", "PCAIRSetProcessorAgglom_fref", old_flag, &flg, NULL));
   PetscCall(PCAIRSetProcessorAgglom_fref(pc, flg));
   // ~~~~
   PetscCall(PCAIRGetDiagScalePolys_fref(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_diag_scale_polys", "Diagonally scale before computing polynomial inverse", "PCAIRSetDiagScalePolys_fref", old_flag, &flg, NULL));
   PetscCall(PCAIRSetDiagScalePolys_fref(pc, flg));   
   // ~~~~
   PetscCall(PCAIRGetMatrixFreePolys_fref(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_matrix_free_polys", "Applies polynomial smoothers matrix-free", "PCAIRSetMatrixFreePolys_fref", old_flag, &flg, NULL));
   PetscCall(PCAIRSetMatrixFreePolys_fref(pc, flg));
   // ~~~~   
   PetscCall(PCAIRGetOnePointClassicalProlong_fref(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_one_point_classical_prolong", "One-point classical prolongator", "PCAIRSetOnePointClassicalProlong_fref", old_flag, &flg, NULL));
   PetscCall(PCAIRSetOnePointClassicalProlong_fref(pc, flg));
   // ~~~~  
   PetscCall(PCAIRGetFullSmoothingUpAndDown_fref(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_full_smoothing_up_and_down", "Full up and down smoothing", "PCAIRSetFullSmoothingUpAndDown_fref", old_flag, &flg, NULL));
   PetscCall(PCAIRSetFullSmoothingUpAndDown_fref(pc, flg));
   // ~~~~  
   PetscCall(PCAIRGetSymmetric_fref(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_symmetric", "Use symmetric grid-transfer operators", "PCAIRSetSymmetric_fref", old_flag, &flg, NULL));
   PetscCall(PCAIRSetSymmetric_fref(pc, flg));
   // ~~~~
   PetscCall(PCAIRGetConstrainW_fref(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_constrain_w", "Use constraints on prolongator", "PCAIRSetConstrainW_fref", old_flag, &flg, NULL));
   PetscCall(PCAIRSetConstrainW_fref(pc, flg));
   // ~~~~
   PetscCall(PCAIRGetConstrainZ_fref(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_constrain_z", "Use constraints on restrictor", "PCAIRSetConstrainZ_fref", old_flag, &flg, NULL));
   PetscCall(PCAIRSetConstrainZ_fref(pc, flg));
   // ~~~~  
   PetscCall(PCAIRGetCoarsestMatrixFreePolys_fref(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_coarsest_matrix_free_polys", "Applies polynomial coarse grid solver matrix-free", "PCAIRSetCoarsestMatrixFreePolys_fref", old_flag, &flg, NULL));
   PetscCall(PCAIRSetCoarsestMatrixFreePolys_fref(pc, flg));
   // ~~~~  
   PetscCall(PCAIRGetCoarsestDiagScalePolys_fref(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_coarsest_diag_scale_polys", "Diagonally scale before computing coarse grid polynomial inverse", "PCAIRSetCoarsestDiagScalePolys_fref", old_flag, &flg, NULL));
   PetscCall(PCAIRSetCoarsestDiagScalePolys_fref(pc, flg));   
   // ~~~~  
   PetscCall(PCAIRGetCoarsestSubcomm_fref(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_coarsest_subcomm", "Computes polynomial coefficients on the coarse grid on subcomm", "PCAIRSetCoarsestSubcomm_fref", old_flag, &flg, NULL));
   PetscCall(PCAIRSetCoarsestSubcomm_fref(pc, flg));
   // ~~~~ 
   PetscCall(PCAIRGetALump_fref(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_a_lump", "Uses lumping on A", "PCAIRSetALump_fref", old_flag, &flg, NULL));
   PetscCall(PCAIRSetALump_fref(pc, flg));
   // ~~~~ 
   PetscCall(PCAIRGetReuseSparsity_fref(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_reuse_sparsity", "Reuses sparsity during setup", "PCAIRSetReuseSparsity_fref", old_flag, &flg, NULL));
   PetscCall(PCAIRSetReuseSparsity_fref(pc, flg));
   // ~~~~
   PetscCall(PCAIRGetReusePolyCoeffs_fref(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_reuse_poly_coeffs", "Reuses gmres polynomial coefficients during setup", "PCAIRSetReusePolyCoeffs_fref", old_flag, &flg, NULL));
   PetscCall(PCAIRSetReusePolyCoeffs_fref(pc, flg));
   // ~~~~
   PetscCall(PCAIRGetReuseAmount_fref(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsRangeInt("-pc_air_reuse_amount", "Amount of data to reuse during setup with reuse_sparsity (1, 2, or 3 - 3 is store everything)", "PCAIRSetReuseAmount_fref", old_int, &input_int, NULL, 1, 3));
   PetscCall(PCAIRSetReuseAmount_fref(pc, input_int));
   // ~~~~
   PetscCall(PCAIRGetProcessorAgglomRatio_fref(pc, &old_real));
   input_real = old_real;
   PetscCall(PetscOptionsReal("-pc_air_processor_agglom_ratio", "Ratio to trigger processor agglomeration", "PCAIRSetProcessorAgglomRatio_fref", old_real, &input_real, NULL));
   PetscCall(PCAIRSetProcessorAgglomRatio_fref(pc, input_real)); 
   // ~~~~
   PetscCall(PCAIRGetAutoTruncateTol_fref(pc, &old_real));
   input_real = old_real;
   PetscCall(PetscOptionsReal("-pc_air_auto_truncate_tol", "Tolerance to use with auto truncation", "PCAIRSetAutoTruncateTol_fref", old_real, &input_real, NULL));
   PetscCall(PCAIRSetAutoTruncateTol_fref(pc, input_real));
   // ~~~~
   PetscCall(PCAIRGetStrongThreshold_fref(pc, &old_real));
   input_real = old_real;
   PetscCall(PetscOptionsReal("-pc_air_strong_threshold", "Strong threshold for CF splitting", "PCAIRSetStrongThreshold_fref", old_real, &input_real, NULL));
   PetscCall(PCAIRSetStrongThreshold_fref(pc, input_real));
   // ~~~~ 
   PetscCall(PCAIRGetDDCFraction_fref(pc, &old_real));
   input_real = old_real;
   PetscCall(PetscOptionsReal("-pc_air_ddc_fraction", "DDC fraction for CF splitting", "PCAIRSetDDCFraction_fref", old_real, &input_real, NULL));
   PetscCall(PCAIRSetDDCFraction_fref(pc, input_real));
   // ~~~~
   PetscCall(PCAIRGetStrongRThreshold_fref(pc, &old_real));
   input_real = old_real;
   PetscCall(PetscOptionsReal("-pc_air_strong_r_threshold", "Strong R threshold for grid-transfer operators", "PCAIRSetStrongRThreshold_fref", old_real, &input_real, NULL));
   PetscCall(PCAIRSetStrongRThreshold_fref(pc, input_real));
   // ~~~~ 
   PetscCall(PCAIRGetRDrop_fref(pc, &old_real));
   input_real = old_real;
   PetscCall(PetscOptionsReal("-pc_air_r_drop", "Drop tolerance for R", "PCAIRSetRDrop_fref", old_real, &input_real, NULL));
   PetscCall(PCAIRSetRDrop_fref(pc, input_real));
   // ~~~~  
   PetscCall(PCAIRGetADrop_fref(pc, &old_real));
   input_real = old_real;
   PetscCall(PetscOptionsReal("-pc_air_a_drop", "Drop tolerance for A", "PCAIRSetADrop_fref", old_real, &input_real, NULL));
   PetscCall(PCAIRSetADrop_fref(pc, input_real));
   // ~~~~  
   const char *const CFSplittingTypes[] = {"PMISR_DDC", "DIAG_DOM", "PMIS", "PMIS_DIST2", "AGG", "PMIS_AGG", "CR", "CFSplittingType", "CF_", NULL};
   PetscCall(PCAIRGetCFSplittingType_fref(pc, &old_cf_type));
   cf_type = old_cf_type;
   PetscCall(PetscOptionsEnum("-pc_air_cf_splitting_type", "CF splitting algorithm", "PCAIRSetCFSplittingType_fref", CFSplittingTypes, (PetscEnum)old_cf_type, (PetscEnum *)&cf_type, &flg));
   PetscCall(PCAIRSetCFSplittingType_fref(pc, cf_type));
   // ~~~~ 
   PetscCall(PCAIRGetDDCIts_fref(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_ddc_its", "DDC iterations for CF splitting", "PCAIRSetDDCIts_fref", old_int, &input_int, NULL));
   PetscCall(PCAIRSetDDCIts_fref(pc, input_int));
   // ~~~~   
   PetscCall(PCAIRGetMaxLubySteps_fref(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_max_luby_steps", "Max Luby steps in CF splitting algorithm", "PCAIRSetMaxLubySteps_fref", old_int, &input_int, NULL));
   PetscCall(PCAIRSetMaxLubySteps_fref(pc, input_int));
   // ~~~~   
   PetscCall(PCAIRGetImproveWIts_fref(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_improve_w_its", "Number of iterations to improve W", "PCAIRSetImproveWIts_fref", old_int, &input_int, NULL));
   PetscCall(PCAIRSetImproveWIts_fref(pc, input_int));
   // ~~~~   
   PetscCall(PCAIRGetImproveZIts_fref(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_improve_z_its", "Number of iterations to improve Z", "PCAIRSetImproveZIts_fref", old_int, &input_int, NULL));
   PetscCall(PCAIRSetImproveZIts_fref(pc, input_int));
   // ~~~~ 
   PetscCall(PCAIRGetSmoothType_fref(pc, old_string));
   strcpy(input_string, old_string);
   PetscCall(PetscOptionsString("-pc_air_smooth_type", "The smoothing types/its", "PCAIRSetSmoothType_fref", input_string, input_string, sizeof(input_string), &flg));
   PetscCall(PCAIRSetSmoothType_fref(pc, input_string));
   // ~~~~    
   PetscCall(PCAIRGetMaxLevels_fref(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_max_levels", "Maximum number of levels", "PCAIRSetMaxLevels_fref", old_int, &input_int, NULL));
   PetscCall(PCAIRSetMaxLevels_fref(pc, input_int));
   // ~~~~    
   PetscCall(PCAIRGetCoarseEqLimit_fref(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_coarse_eq_limit", "Minimum number of global unknowns on the coarse grid", "PCAIRSetCoarseEqLimit_fref", old_int, &input_int, NULL));
   PetscCall(PCAIRSetCoarseEqLimit_fref(pc, input_int));
   // ~~~~    
   PetscCall(PCAIRGetAutoTruncateStartLevel_fref(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_auto_truncate_start_level", "Use auto truncation from this level", "PCAIRSetAutoTruncateStartLevel_fref", old_int, &input_int, NULL));
   PetscCall(PCAIRSetAutoTruncateStartLevel_fref(pc, input_int));
   // ~~~~ 
   const char *const PCPFLAREINVTypes[] = {"POWER", "ARNOLDI", "NEWTON", "NEWTON_NO_EXTRA", "NEUMANN", "SAI", "ISAI", "WJACOBI", "JACOBI", "PCPFLAREINVType", "PFLAREINV_", NULL};
   PetscCall(PCAIRGetInverseType_fref(pc, &old_type));
   type = old_type;
   PetscCall(PetscOptionsEnum("-pc_air_inverse_type", "Inverse type", "PCPFLAREINVSetType", PCPFLAREINVTypes, (PetscEnum)old_type, (PetscEnum *)&type, &flg));
   PetscCall(PCAIRSetInverseType_fref(pc, type));
   // ~~~~ 
   // If not explicitly set, the getter returns the F point smoother value
   // Only set if given, so an unset C value keeps following the F point smoother
   PetscCall(PCAIRGetCInverseType_fref(pc, &old_type));
   type = old_type;
   PetscCall(PetscOptionsEnum("-pc_air_c_inverse_type", "C point inverse type", "PCPFLAREINVSetType", PCPFLAREINVTypes, (PetscEnum)old_type, (PetscEnum *)&type, &flg));
   if (flg) PetscCall(PCAIRSetCInverseType_fref(pc, type));
   // ~~~~
   const char *const PCAIRZTypes[] = {"PRODUCT", "LAIR", "LAIR_SAI", "PCAIRZType", "AIR_Z_", NULL};
   PetscCall(PCAIRGetZType_fref(pc, &old_z_type));
   z_type = old_z_type;
   PetscCall(PetscOptionsEnum("-pc_air_z_type", "Z type", "PCAIRSetZType_fref", PCAIRZTypes, (PetscEnum)old_z_type, (PetscEnum *)&z_type, &flg));
   PetscCall(PCAIRSetZType_fref(pc, z_type));
   // ~~~~ 
   PetscCall(PCAIRGetLairDistance_fref(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_lair_distance", "lAIR distance", "PCAIRSetLairDistance_fref", old_int, &input_int, NULL));
   PetscCall(PCAIRSetLairDistance_fref(pc, input_int));   
   // ~~~~ 
   PetscCall(PCAIRGetPolyOrder_fref(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_poly_order", "Polynomial order", "PCAIRSetPolyOrder_fref", old_int, &input_int, NULL));
   PetscCall(PCAIRSetPolyOrder_fref(pc, input_int));
   // ~~~~ 
   PetscCall(PCAIRGetInverseSparsityOrder_fref(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_inverse_sparsity_order", "Inverse sparsity order", "PCAIRSetInverseSparsityOrder_fref", old_int, &input_int, NULL));
   PetscCall(PCAIRSetInverseSparsityOrder_fref(pc, input_int));
   // ~~~~ 
   // If not explicitly set, the getter returns the F point smoother value
   PetscCall(PCAIRGetCPolyOrder_fref(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_c_poly_order", "C point polynomial order", "PCAIRSetCPolyOrder_fref", old_int, &input_int, &flg));
   if (flg) PetscCall(PCAIRSetCPolyOrder_fref(pc, input_int));
   // ~~~~ 
   // If not explicitly set, the getter returns the F point smoother value
   PetscCall(PCAIRGetCInverseSparsityOrder_fref(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_c_inverse_sparsity_order", "C point inverse sparsity order", "PCAIRSetCInverseSparsityOrder_fref", old_int, &input_int, &flg));
   if (flg) PetscCall(PCAIRSetCInverseSparsityOrder_fref(pc, input_int));
   // ~~~~ 
   PetscCall(PCAIRGetCoarsestInverseType_fref(pc, &old_type));
   type = old_type;
   PetscCall(PetscOptionsEnum("-pc_air_coarsest_inverse_type", "Inverse type on the coarse grid", "PCPFLAREINVSetType", PCPFLAREINVTypes, (PetscEnum)old_type, (PetscEnum *)&type, &flg));
   PetscCall(PCAIRSetCoarsestInverseType_fref(pc, type));
   // ~~~~ 
   PetscCall(PCAIRGetCoarsestPolyOrder_fref(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_coarsest_poly_order", "Polynomial order on the coarse grid", "PCAIRSetCoarsestPolyOrder_fref", old_int, &input_int, NULL));
   PetscCall(PCAIRSetCoarsestPolyOrder_fref(pc, input_int));
   // ~~~~ 
   PetscCall(PCAIRGetCoarsestInverseSparsityOrder_fref(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_coarsest_inverse_sparsity_order", "Inverse sparsity order on the coarse grid", "PCAIRSetCoarsestInverseSparsityOrder_fref", old_int, &input_int, NULL));
   PetscCall(PCAIRSetCoarsestInverseSparsityOrder_fref(pc, input_int));
   // ~~~~ 
   PetscCall(PCAIRGetProcessorAgglomFactor_fref(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_processor_agglom_factor", "Factor to reduce MPI ranks by", "PCAIRSetProcessorAgglomFactor_fref", old_int, &input_int, NULL));
   PetscCall(PCAIRSetProcessorAgglomFactor_fref(pc, input_int));
   // ~~~~        
   PetscCall(PCAIRGetProcessEqLimit_fref(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_process_eq_limit", "Trigger process agglomeration if fewer eqs/core", "PCAIRSetProcessEqLimit_fref", old_int, &input_int, NULL));
   PetscCall(PCAIRSetProcessEqLimit_fref(pc, input_int));
   // ~~~~                                                       

   PetscOptionsHeadEnd();
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~

static PetscErrorCode PCView_AIR_c_fref(PC pc, PetscViewer viewer)
{
   PetscFunctionBegin;
   
   PC *pc_air_shell = (PC *)pc->data;

   PetscInt input_int, input_int_two, input_int_three, input_int_four;
   PetscBool flg, flg_f_smooth, flg_c_smooth, flg_diag_scale;
   PetscReal input_real, input_real_two;
   PCPFLAREINVType input_type;
   PCAIRZType z_type;
   CFSplittingType cf_type;
   char input_string[PETSC_MAX_PATH_LEN];

   // Print out details
   PetscBool  iascii;
   PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));

   if (iascii) {

      PetscCall(PCAIRGetMaxLevels_fref(pc, &input_int));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Max number of levels=%" PetscInt_FMT " \n", input_int));
      PetscCall(PCAIRGetCoarseEqLimit_fref(pc, &input_int));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Coarse eq limit=%" PetscInt_FMT " \n", input_int));

      PetscCall(PCAIRGetAutoTruncateStartLevel_fref(pc, &input_int));
      PetscCall(PCAIRGetAutoTruncateTol_fref(pc, &input_real));
      if (input_int != -1)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  Auto truncate start level=%" PetscInt_FMT ", with tolerance=%.2e \n", input_int, input_real));
      }

      PetscCall(PCAIRGetRDrop_fref(pc, &input_real));
      PetscCall(PCAIRGetADrop_fref(pc, &input_real_two));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  A drop tolerance=%.2e, R drop tolerance=%.2e \n", input_real_two, input_real));
      PetscCall(PCAIRGetALump_fref(pc, &flg));
      if (flg) PetscCall(PetscViewerASCIIPrintf(viewer, "  Lumping \n"));
      PetscCall(PCAIRGetReuseSparsity_fref(pc, &flg));
      if (flg)
      {
         PetscCall(PCAIRGetReuseAmount_fref(pc, &input_int));
         PetscCall(PetscViewerASCIIPrintf(viewer, "  Reusing sparsity during setup (reuse amount=%" PetscInt_FMT ")\n", input_int));
      }
      PetscCall(PCAIRGetReusePolyCoeffs_fref(pc, &flg));
      if (flg) PetscCall(PetscViewerASCIIPrintf(viewer, "  Reusing gmres polynomial coefficients during setup \n"));

      PetscCall(PCAIRGetProcessorAgglom_fref(pc, &flg));
      PetscCall(PCAIRGetProcessorAgglomRatio_fref(pc, &input_real));
      PetscCall(PCAIRGetProcessorAgglomFactor_fref(pc, &input_int));
      PetscCall(PCAIRGetProcessEqLimit_fref(pc, &input_int_two));
      if (flg) PetscCall(PetscViewerASCIIPrintf(viewer, "  Processor agglomeration with factor=%" PetscInt_FMT ", ratio=%f and eq limit=%" PetscInt_FMT " \n", input_int, input_real, input_int_two));
      PetscCall(PCAIRGetSubcomm_fref(pc, &flg));
      if (flg) PetscCall(PetscViewerASCIIPrintf(viewer, "  Polynomial coefficients calculated on subcomm \n"));
      
      PetscCall(PCAIRGetCFSplittingType_fref(pc, &cf_type));
      PetscCall(PCAIRGetStrongThreshold_fref(pc, &input_real));
      PetscCall(PCAIRGetDDCIts_fref(pc, &input_int_three));
      PetscCall(PCAIRGetDDCFraction_fref(pc, &input_real_two));
      PetscCall(PCAIRGetMaxLubySteps_fref(pc, &input_int_two));
      if (cf_type == CF_PMISR_DDC)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  CF splitting algorithm=PMISR_DDC \n"));
         PetscCall(PetscViewerASCIIPrintf(viewer, "    %" PetscInt_FMT " Luby steps \n      Strong threshold=%f, DDC its=%" PetscInt_FMT ", DDC fraction=%f \n", \
               input_int_two, input_real, input_int_three, input_real_two));
      }
      else if (cf_type == CF_DIAG_DOM)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  CF splitting algorithm=DIAG_DOM \n"));
         PetscCall(PetscViewerASCIIPrintf(viewer, "    %" PetscInt_FMT " Luby steps \n      Diagonal dominance target (strong threshold)=%f \n", \
               input_int_two, input_real));
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
      else if (cf_type == CF_PMIS_AGG)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  CF splitting algorithm=PMIS_AGG \n"));
         PetscCall(PetscViewerASCIIPrintf(viewer, "    Strong threshold=%f \n", \
                  input_real));
      }
      else if (cf_type == CF_CR)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  CF splitting algorithm=CR \n"));
         PetscCall(PetscViewerASCIIPrintf(viewer, "    Target CR rate (strong threshold)=%f \n", \
                  input_real));
      }

      PetscCall(PCAIRGetFullSmoothingUpAndDown_fref(pc, &flg));
      if (flg) 
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  Full smoothing up & down \n"));
         PetscCall(PCAIRGetInverseType_fref(pc, &input_type));
         PetscCall(PCAIRGetPolyOrder_fref(pc, &input_int_two));
         PetscCall(PCAIRGetInverseSparsityOrder_fref(pc, &input_int_three));
         PetscCall(PCAIRGetDiagScalePolys_fref(pc, &flg_diag_scale));
         PetscCall(PCAIRGetMatrixFreePolys_fref(pc, &flg));

         // What type of inverse
         if (input_type == PFLAREINV_POWER)
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, power basis, order %" PetscInt_FMT " \n", input_int_two));
            if (flg_diag_scale)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
            }
         }
         else if (input_type == PFLAREINV_ARNOLDI)
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, arnoldi basis, order %" PetscInt_FMT " \n", input_int_two));
            if (flg_diag_scale)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
            }            
         }
         else if (input_type == PFLAREINV_NEWTON)
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, newton basis with extra roots, order %" PetscInt_FMT " \n", input_int_two));
            if (flg_diag_scale)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
            }            
         }
         else if (input_type == PFLAREINV_NEWTON_NO_EXTRA)
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, newton basis without extra roots, order %" PetscInt_FMT " \n", input_int_two));
            if (flg_diag_scale)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
            }            
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
            // Neumann always diagonally scales
            PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
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
         PetscCall(PCAIRGetSmoothType_fref(pc, input_string));
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
            PetscCall(PCAIRGetInverseType_fref(pc, &input_type));
            PetscCall(PCAIRGetPolyOrder_fref(pc, &input_int_two));
            PetscCall(PCAIRGetInverseSparsityOrder_fref(pc, &input_int_three));
            PetscCall(PCAIRGetDiagScalePolys_fref(pc, &flg_diag_scale));
            PetscCall(PCAIRGetMatrixFreePolys_fref(pc, &flg));

            // What type of inverse
            if (input_type == PFLAREINV_POWER)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    F smooth: GMRES polynomial, power basis, order %" PetscInt_FMT " \n", input_int_two));
               if (flg_diag_scale)
               {
                  PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
               }
            }
            else if (input_type == PFLAREINV_ARNOLDI)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    F smooth: GMRES polynomial, arnoldi basis, order %" PetscInt_FMT " \n", input_int_two));
               if (flg_diag_scale)
               {
                  PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
               }
            }
            else if (input_type == PFLAREINV_NEWTON)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    F smooth: GMRES polynomial, newton basis with extra roots, order %" PetscInt_FMT " \n", input_int_two));
               if (flg_diag_scale)
               {
                  PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
               }
            }
            else if (input_type == PFLAREINV_NEWTON_NO_EXTRA)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    F smooth: GMRES polynomial, newton basis without extra roots, order %" PetscInt_FMT " \n", input_int_two));
               if (flg_diag_scale)
               {
                  PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
               }
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
               // Neumann always diagonally scales
               PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
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

            PetscCall(PCAIRGetCInverseType_fref(pc, &input_type));
            PetscCall(PCAIRGetCPolyOrder_fref(pc, &input_int_two));
            PetscCall(PCAIRGetCInverseSparsityOrder_fref(pc, &input_int_three));
            PetscCall(PCAIRGetDiagScalePolys_fref(pc, &flg_diag_scale));
            PetscCall(PCAIRGetMatrixFreePolys_fref(pc, &flg));

            // What type of inverse
            if (input_type == PFLAREINV_POWER)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    C smooth: GMRES polynomial, power basis, order %" PetscInt_FMT " \n", input_int_two));
               if (flg_diag_scale)
               {
                  PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
               }               
            }
            else if (input_type == PFLAREINV_ARNOLDI)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    C smooth: GMRES polynomial, arnoldi basis, order %" PetscInt_FMT " \n", input_int_two));
               if (flg_diag_scale)
               {
                  PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
               }
            }
            else if (input_type == PFLAREINV_NEWTON)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    C smooth: GMRES polynomial, newton basis with extra roots, order %" PetscInt_FMT " \n", input_int_two));
               if (flg_diag_scale)
               {
                  PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
               }
            }
            else if (input_type == PFLAREINV_NEWTON_NO_EXTRA)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "    C smooth: GMRES polynomial, newton basis without extra roots, order %" PetscInt_FMT " \n", input_int_two));
               if (flg_diag_scale)
               {
                  PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
               }
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
               // Neumann always diagonally scales
               PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
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
      PetscCall(PCAIRGetZType_fref(pc, &z_type));
      PetscCall(PCAIRGetLairDistance_fref(pc, &input_int_two));
      PetscCall(PCAIRGetImproveWIts_fref(pc, &input_int_three));
      PetscCall(PCAIRGetImproveZIts_fref(pc, &input_int_four));
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
      PetscCall(PCAIRGetStrongRThreshold_fref(pc, &input_real));
      PetscCall(PetscViewerASCIIPrintf(viewer, "    Strong R threshold=%.2e \n", input_real));
      if (input_int_three > 0)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    Improve W %" PetscInt_FMT " iterations \n", input_int_three));
      }
      if (input_int_four > 0)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    Improve Z %" PetscInt_FMT " iterations \n", input_int_four));      
      }
      PetscCall(PCAIRGetConstrainZ_fref(pc, &flg));
      if (flg) 
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    Constraints applied to restrictor \n"));      
      }     

      PetscCall(PCAIRGetSymmetric_fref(pc, &flg));
      if (flg) 
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    Approximate ideal prolongator \n"));      
         PetscCall(PetscViewerASCIIPrintf(viewer, "      Symmetric - transpose of restrictor \n"));      
      }        
      else
      {
         PetscCall(PCAIRGetOnePointClassicalProlong_fref(pc, &flg));
         if (flg) 
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "    One point classical prolongator \n"));      
         }
         else
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "    Approximate ideal prolongator \n"));      
         }       
      }

      PetscCall(PCAIRGetConstrainW_fref(pc, &flg));
      if (flg) 
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "      Constraints applied to prolongator \n"));      
      }      

      PetscCall(PetscViewerASCIIPrintf(viewer, "  Coarse grid solver: \n"));
      PetscCall(PCAIRGetCoarsestInverseType_fref(pc, &input_type));
      PetscCall(PCAIRGetCoarsestPolyOrder_fref(pc, &input_int_two));
      PetscCall(PCAIRGetCoarsestInverseSparsityOrder_fref(pc, &input_int_three));
      PetscCall(PCAIRGetCoarsestDiagScalePolys_fref(pc, &flg_diag_scale));
      PetscCall(PCAIRGetCoarsestMatrixFreePolys_fref(pc, &flg));

      // What type of inverse
      if (input_type == PFLAREINV_POWER)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, power basis, order %" PetscInt_FMT " \n", input_int_two));
         if (flg_diag_scale)
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
         }          
      }
      else if (input_type == PFLAREINV_ARNOLDI)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, arnoldi basis, order %" PetscInt_FMT " \n", input_int_two));
         if (flg_diag_scale)
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
         }          
      }
      else if (input_type == PFLAREINV_NEWTON)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, newton basis with extra roots, order %" PetscInt_FMT " \n", input_int_two));      
         if (flg_diag_scale)
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
         }          
      }
      else if (input_type == PFLAREINV_NEWTON_NO_EXTRA)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "    GMRES polynomial, newton basis without extra roots, order %" PetscInt_FMT " \n", input_int_two));
         if (flg_diag_scale)
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
         }          
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
     
      PetscCall(PCAIRGetCoarsestSubcomm_fref(pc, &flg));
      if (flg) PetscCall(PetscViewerASCIIPrintf(viewer, "   Polynomial coefficients calculated on subcomm \n"));

      PetscCall(PetscViewerASCIIPrintf(viewer, "The underlying PCMG: \n"));
      // Call the underlying pcshell view
      PetscCall(PCView(*pc_air_shell, viewer));
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~


// Creates the structure we need for this PC
static PetscErrorCode PCCreate_AIR_fref(PC pc)
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
   create_pc_air_data_c_fref(&pc_air_data);
   // The type and rest of the creation of the pcshell
   // is done in here
   create_pc_air_shell_c_fref(&pc_air_data, pc_air_shell);
   // Set the pc data
   pc->data = (void*)pc_air_shell;

   // Set the method functions
   pc->ops->apply               = PCApply_AIR_c_fref;
   pc->ops->matapply            = PCMatApply_AIR_c_fref;
   // We deliberately don't set matapplytranspose - PCMatApplyTranspose falls back to
   // applying PCApplyTranspose column by column when it is NULL, which is correct
   pc->ops->applytranspose      = PCApplyTranspose_AIR_c_fref;
   pc->ops->setup               = PCSetUp_AIR_c_fref;
   pc->ops->destroy             = PCDestroy_AIR_c_fref;
   pc->ops->view                = PCView_AIR_c_fref;  
   pc->ops->reset               = PCReset_AIR_c_fref;
   pc->ops->setfromoptions      = PCSetFromOptions_AIR_c_fref;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~

// Registers the reference PC type (called by PCRegister_AIR in PFLARE_PORT_DEBUG builds)
PETSC_INTERN void PCRegister_AIR_fref(void)
{
   PetscCallVoid(PCRegister("air_fref", PCCreate_AIR_fref));
}
