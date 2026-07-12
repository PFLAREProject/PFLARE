/*
  Definition and registration of the PC AIR type
  Largely just a wrapper around all the fortran 
  using PCShell
*/

// Include the petsc header files
#include <petsc/private/pcimpl.h>
#include "pflare.h"
#include <string.h>

/* SUBMANSEC = PC */

// Defined in C_Fortran_Bindings.F90
PETSC_EXTERN void PCReset_AIR_Shell_c(PC *pc);
PETSC_EXTERN void create_pc_air_data_c(void **pc_air_data);
PETSC_EXTERN void create_pc_air_shell_c(void **pc_air_data, PC *pc);
PETSC_EXTERN void compute_cf_splitting_c(Mat *input_mat, int symmetric_int,
   PetscReal strong_threshold, int max_luby_steps, int cf_splitting_type,
   int ddc_its, PetscReal fraction_swap,
   IS *is_fine, IS *is_coarse);
PETSC_EXTERN void compute_diag_dom_submatrix_c(Mat *input_mat, PetscReal max_dd_ratio, Mat *output_mat);
PETSC_EXTERN void remove_from_sparse_match_c(Mat *input_mat, Mat *output_mat,
   int lump_int, int alpha_int, PetscReal alpha);
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
PETSC_EXTERN void PCAIRGetDDCFraction_c(PC *pc, PetscReal *input_real);
PETSC_EXTERN void PCAIRGetCFSplittingType_c(PC *pc, CFSplittingType *cf_splitting_type);
PETSC_EXTERN void PCAIRGetMaxLubySteps_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetSmoothType_c(PC *pc, char* input_string);
PETSC_EXTERN void PCAIRGetDiagScalePolys_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetMatrixFreePolys_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetOnePointClassicalProlong_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetFullSmoothingUpAndDown_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetSymmetric_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetConstrainW_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetConstrainZ_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetImproveWIts_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetImproveZIts_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetStrongRThreshold_c(PC *pc, PetscReal *input_real);
PETSC_EXTERN void PCAIRGetInverseType_c(PC *pc, PCPFLAREINVType *inverse_type);
PETSC_EXTERN void PCAIRGetCInverseType_c(PC *pc, PCPFLAREINVType *inverse_type);
PETSC_EXTERN void PCAIRGetZType_c(PC *pc, PCAIRZType *z_type);
PETSC_EXTERN void PCAIRGetPolyOrder_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetLairDistance_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetInverseSparsityOrder_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetCPolyOrder_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetCInverseSparsityOrder_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetCoarsestInverseType_c(PC *pc, PCPFLAREINVType *inverse_type);
PETSC_EXTERN void PCAIRGetCoarsestPolyOrder_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetCoarsestInverseSparsityOrder_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetCoarsestMatrixFreePolys_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetCoarsestDiagScalePolys_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetCoarsestSubcomm_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetRDrop_c(PC *pc, PetscReal *input_real);
PETSC_EXTERN void PCAIRGetADrop_c(PC *pc, PetscReal *input_real);
PETSC_EXTERN void PCAIRGetALump_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetReuseSparsity_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetReusePolyCoeffs_c(PC *pc, PetscBool *input_bool);
PETSC_EXTERN void PCAIRGetReuseAmount_c(PC *pc, PetscInt *input_int);
PETSC_EXTERN void PCAIRGetPolyCoeffs_c(PC *pc, PetscInt petsc_level, int which_inverse, PetscReal **coeffs_ptr, PetscInt *row_size, PetscInt *col_size);
PETSC_EXTERN void PCAIRGetGridComplexity_c(PC *pc, PetscReal *complexity);
PETSC_EXTERN void PCAIRGetOperatorComplexity_c(PC *pc, PetscReal *complexity);
PETSC_EXTERN void PCAIRGetCycleComplexity_c(PC *pc, PetscReal *complexity);
PETSC_EXTERN void PCAIRGetStorageComplexity_c(PC *pc, PetscReal *complexity);
PETSC_EXTERN void PCAIRGetReuseStorageComplexity_c(PC *pc, PetscReal *complexity);

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
PETSC_EXTERN void PCAIRSetDDCFraction_c(PC *pc, PetscReal input_real);
PETSC_EXTERN void PCAIRSetCFSplittingType_c(PC *pc, CFSplittingType cf_splitting_type);
PETSC_EXTERN void PCAIRSetMaxLubySteps_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetSmoothType_c(PC *pc, const char* input_string);
PETSC_EXTERN void PCAIRSetDiagScalePolys_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetMatrixFreePolys_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetOnePointClassicalProlong_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetFullSmoothingUpAndDown_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetSymmetric_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetConstrainW_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetConstrainZ_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetImproveWIts_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetImproveZIts_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetStrongRThreshold_c(PC *pc, PetscReal input_real);
PETSC_EXTERN void PCAIRSetInverseType_c(PC *pc, PCPFLAREINVType inverse_type);
PETSC_EXTERN void PCAIRSetZType_c(PC *pc, PCAIRZType z_type);
PETSC_EXTERN void PCAIRSetLairDistance_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetPolyOrder_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetInverseSparsityOrder_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetCInverseType_c(PC *pc, PCPFLAREINVType inverse_type);
PETSC_EXTERN void PCAIRSetCPolyOrder_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetCInverseSparsityOrder_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetCoarsestInverseType_c(PC *pc, PCPFLAREINVType inverse_type);
PETSC_EXTERN void PCAIRSetCoarsestPolyOrder_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetCoarsestInverseSparsityOrder_c(PC *pc, PetscInt input_int);
PETSC_EXTERN void PCAIRSetCoarsestMatrixFreePolys_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetCoarsestDiagScalePolys_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetCoarsestSubcomm_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetRDrop_c(PC *pc, PetscReal input_real);
PETSC_EXTERN void PCAIRSetADrop_c(PC *pc, PetscReal input_real);
PETSC_EXTERN void PCAIRSetALump_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetReuseSparsity_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetReusePolyCoeffs_c(PC *pc, PetscBool input_bool);
PETSC_EXTERN void PCAIRSetReuseAmount_c(PC *pc, PetscInt input_int);
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

// CF splitting
PETSC_EXTERN void compute_cf_splitting(Mat input_mat, int symmetric_int,
   PetscReal strong_threshold, int max_luby_steps, int cf_splitting_type,
   int ddc_its, PetscReal fraction_swap,
   IS *is_fine, IS *is_coarse)
{
   // Now we call petsc fortran routines in compute_cf_splitting_c from this C file
   // so we have to have made sure this is called
   // Otherwise things like PETSC_NULL_INTEGER_ARRAY aren't defined
   PetscCallVoid(PetscInitializeFortran());
   compute_cf_splitting_c(&input_mat, symmetric_int, strong_threshold,
      max_luby_steps, cf_splitting_type, ddc_its, fraction_swap,
      is_fine, is_coarse);
}

PETSC_EXTERN void compute_diag_dom_submatrix(Mat input_mat, PetscReal max_dd_ratio, Mat *output_mat)
{
   // Now we call petsc fortran routines in compute_cf_splitting_c from this C file
   // so we have to have made sure this is called
   // Otherwise things like PETSC_NULL_INTEGER_ARRAY aren't defined
   PetscCallVoid(PetscInitializeFortran());
   compute_diag_dom_submatrix_c(&input_mat, max_dd_ratio, output_mat);
}

// Restrict input_mat onto output_mat's existing sparsity pattern.
// Auto-dispatches between the CPU and Kokkos implementations based on the
// matrix type, so callers can mix kokkos and non-kokkos matrices freely.
PETSC_EXTERN void remove_from_sparse_match(Mat input_mat, Mat output_mat,
   int lump_int, int alpha_int, PetscReal alpha)
{
   remove_from_sparse_match_c(&input_mat, &output_mat, lump_int, alpha_int, alpha);
}

// Get routines

/*@
  PCAIRGetPrintStatsTimings - Returns whether `PCAIR` prints statistics about the multigrid hierarchy and timings

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_bool - `PETSC_TRUE` if statistics about the multigrid hierarchy and timings are printed

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetPrintStatsTimings()`, `PCAIRGetNumLevels()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetPrintStatsTimings(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetPrintStatsTimings_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetMaxLevels - Returns the maximum number of levels allowed in the `PCAIR` multigrid hierarchy

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_int - the maximum number of levels in the multigrid hierarchy

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetMaxLevels()`, `PCAIRGetCoarseEqLimit()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetMaxLevels(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetMaxLevels_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetCoarseEqLimit - Returns the minimum number of global unknowns allowed on the coarsest grid of the `PCAIR` hierarchy

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_int - the minimum number of global unknowns allowed on the coarse grid

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetCoarseEqLimit()`, `PCAIRGetMaxLevels()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetCoarseEqLimit(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetCoarseEqLimit_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetAutoTruncateStartLevel - Returns the level from which `PCAIR` builds and evaluates a coarse-grid solver to decide whether the hierarchy can be truncated there

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_int - the level from which to build a coarse solver and test whether the hierarchy can be truncated

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetAutoTruncateStartLevel()`, `PCAIRGetAutoTruncateTol()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetAutoTruncateStartLevel(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetAutoTruncateStartLevel_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetAutoTruncateTol - Returns the relative tolerance used by `PCAIR` to decide if a coarse-grid solver is good enough to truncate the hierarchy

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_real - the relative tolerance used to determine if a coarse-grid solver is good enough to truncate the hierarchy

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetAutoTruncateTol()`, `PCAIRGetAutoTruncateStartLevel()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetAutoTruncateTol(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetAutoTruncateTol_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetNumLevels - Returns the number of levels in the `PCAIR` multigrid hierarchy

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_int - the number of levels in the hierarchy, or -1 if `PCSetUp()` has not yet been called

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCSetUp()`, `PCAIRGetGridComplexity()`, `PCAIRGetOperatorComplexity()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetNumLevels(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetNumLevels_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetProcessorAgglom - Returns whether `PCAIR` uses a graph partitioner to repartition coarse grids and reduce the number of active MPI ranks

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_bool - `PETSC_TRUE` if the coarse grids are repartitioned onto fewer MPI ranks as the hierarchy coarsens

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetProcessorAgglom()`, `PCAIRGetProcessorAgglomRatio()`, `PCAIRGetProcessorAgglomFactor()`, `PCAIRGetProcessEqLimit()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetProcessorAgglom(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetProcessorAgglom_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetProcessorAgglomRatio - Returns the local to non-local nonzero ratio that triggers processor agglomeration in `PCAIR`

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_real - the local to non-local nonzero ratio that triggers processor agglomeration

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetProcessorAgglomRatio()`, `PCAIRGetProcessorAgglom()`, `PCAIRGetProcessorAgglomFactor()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetProcessorAgglomRatio(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetProcessorAgglomRatio_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetProcessorAgglomFactor - Returns the factor by which `PCAIR` reduces the number of active MPI ranks each time processor agglomeration occurs

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_int - the factor by which the number of active MPI ranks is reduced

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetProcessorAgglomFactor()`, `PCAIRGetProcessorAgglom()`, `PCAIRGetProcessorAgglomRatio()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetProcessorAgglomFactor(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetProcessorAgglomFactor_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetProcessEqLimit - Returns the average number of equations per MPI rank below which `PCAIR` triggers processor agglomeration

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_int - the average number of equations per rank below which processor agglomeration is triggered

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetProcessEqLimit()`, `PCAIRGetProcessorAgglom()`, `PCAIRGetProcessorAgglomRatio()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetProcessEqLimit(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetProcessEqLimit_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetSubcomm - Returns whether `PCAIR` performs reductions for arnoldi or newton polynomial inverses on a subcommunicator that excludes empty MPI ranks

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_bool - `PETSC_TRUE` if the reductions are performed on a subcommunicator excluding empty ranks

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetSubcomm()`, `PCAIRGetProcessorAgglom()`, `PCAIRGetInverseType()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetSubcomm(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetSubcomm_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetStrongThreshold - Returns the strong threshold used in the `PCAIR` CF splitting

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_real - the strong threshold used in the CF splitting

  Level: intermediate

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetStrongThreshold()`, `PCAIRGetCFSplittingType()`, `CFSplittingType`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetStrongThreshold(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetStrongThreshold_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetDDCFraction - Returns the fraction of local F points converted to C points by diagonal dominance in the `PCAIR` `CF_PMISR_DDC` CF splitting

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_real - the local fraction of F points converted to C points, or (if negative) minus the diagonal-dominance ratio threshold

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetDDCFraction()`, `PCAIRGetDDCIts()`, `PCAIRGetCFSplittingType()`, `CFSplittingType`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetDDCFraction(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetDDCFraction_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetCFSplittingType - Returns the coarse/fine (CF) splitting algorithm used by `PCAIR`

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. cf_splitting_type - the CF splitting algorithm, one of `CF_PMISR_DDC`, `CF_DIAG_DOM`, `CF_PMIS`, `CF_PMIS_DIST2`, `CF_AGG`, or `CF_PMIS_AGG`

  Level: intermediate

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetCFSplittingType()`, `CFSplittingType`, `PCAIRGetStrongThreshold()`, `PCAIRGetMaxLubySteps()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetCFSplittingType(PC pc, CFSplittingType *cf_splitting_type)
{
   PetscFunctionBegin;
   PCAIRGetCFSplittingType_c(&pc, cf_splitting_type);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetDDCIts - Returns the number of diagonal-dominance-conversion (DDC) iterations used by the `PCAIR` `CF_PMISR_DDC` CF splitting

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_int - the number of DDC iterations

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetDDCIts()`, `PCAIRGetDDCFraction()`, `PCAIRGetCFSplittingType()`, `CFSplittingType`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetDDCIts(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetDDCIts_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetMaxLubySteps - Returns the maximum number of Luby steps used by the `PCAIR` CF splitting

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_int - the maximum number of Luby steps; a negative value means as many steps as necessary

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetMaxLubySteps()`, `PCAIRGetCFSplittingType()`, `CFSplittingType`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetMaxLubySteps(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetMaxLubySteps_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@C
  PCAIRGetSmoothType - Returns the type and number of smooths used by the `PCAIR` reduction multigrid

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_string - the smoothing pattern, a sequence of `f` and `c` characters giving the type and number of smooths (for example `ff`, `fc`, `fcf`, `ffc`, ...)

  Level: intermediate

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetSmoothType()`, `PCAIRGetInverseType()`, `PCAIRGetFullSmoothingUpAndDown()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetSmoothType(PC pc, char *input_string)
{
   PetscFunctionBegin;
   PCAIRGetSmoothType_c(&pc, input_string);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetDiagScalePolys - Returns whether `PCAIR` diagonally scales before computing a polynomial approximate inverse

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_bool - `PETSC_TRUE` if diagonally scaling before computing a polynomial inverse

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetDiagScalePolys()`, `PCAIRGetInverseType()`, `PCAIRGetMatrixFreePolys()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetDiagScalePolys(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetDiagScalePolys_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetMatrixFreePolys - Returns whether `PCAIR` applies polynomial smoothers matrix-free where possible

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_bool - `PETSC_TRUE` if polynomial smoothers are applied matrix-free where possible

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetMatrixFreePolys()`, `PCAIRGetInverseType()`, `PCAIRGetDiagScalePolys()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetMatrixFreePolys(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetMatrixFreePolys_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetOnePointClassicalProlong - Returns whether `PCAIR` uses a one-point classical prolongator instead of an approximate ideal prolongator

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_bool - `PETSC_TRUE` if a one-point classical prolongator is used instead of an approximate ideal prolongator

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetOnePointClassicalProlong()`, `PCAIRGetSymmetric()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetOnePointClassicalProlong(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetOnePointClassicalProlong_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetFullSmoothingUpAndDown - Returns whether `PCAIR` smooths all points on the up and down sweeps, instead of the default down F and C smoothing

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_bool - `PETSC_TRUE` if all points are smoothed on both the up and down sweeps

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetFullSmoothingUpAndDown()`, `PCAIRGetSmoothType()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetFullSmoothingUpAndDown(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetFullSmoothingUpAndDown_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetSymmetric - Returns whether `PCAIR` defines the prolongator as the transpose of the restrictor

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_bool - `PETSC_TRUE` if symmetric grid-transfer operators are used, defining the prolongator as R^T

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetSymmetric()`, `PCAIRGetOnePointClassicalProlong()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetSymmetric(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetSymmetric_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetConstrainW - Returns whether `PCAIR` applies constraints to the prolongator

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_bool - `PETSC_TRUE` if constraints are applied to the prolongator

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetConstrainW()`, `PCAIRGetConstrainZ()`, `MatSetNearNullSpace()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetConstrainW(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetConstrainW_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetConstrainZ - Returns whether `PCAIR` applies constraints to the restrictor

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_bool - `PETSC_TRUE` if constraints are applied to the restrictor

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetConstrainZ()`, `PCAIRGetConstrainW()`, `MatSetNearNullSpace()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetConstrainZ(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetConstrainZ_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetImproveWIts - Returns the number of Richardson iterations used by `PCAIR` to improve the approximate prolongator

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_int - the number of Richardson iterations

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetImproveWIts()`, `PCAIRGetImproveZIts()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetImproveWIts(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetImproveWIts_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetImproveZIts - Returns the number of Richardson iterations used by `PCAIR` to improve the approximate restrictor

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_int - the number of Richardson iterations

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetImproveZIts()`, `PCAIRGetImproveWIts()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetImproveZIts(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetImproveZIts_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetStrongRThreshold - Returns the threshold used by `PCAIR` to drop entries when forming the grid-transfer operators

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_real - the strong R threshold

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetStrongRThreshold()`, `PCAIRGetZType()`, `PCAIRGetStrongThreshold()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetStrongRThreshold(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetStrongRThreshold_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetInverseType - Returns the type of approximate inverse used as the `PCAIR` F-point smoother

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. inverse_type - the approximate inverse type, one of `PFLAREINV_POWER`, `PFLAREINV_ARNOLDI`, `PFLAREINV_NEWTON`, `PFLAREINV_NEWTON_NO_EXTRA`, `PFLAREINV_NEUMANN`, `PFLAREINV_SAI`, `PFLAREINV_ISAI`, `PFLAREINV_WJACOBI`, or `PFLAREINV_JACOBI`

  Level: intermediate

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetInverseType()`, `PCPFLAREINVType`, `PCAIRGetCInverseType()`, `PCAIRGetZType()`, `PCAIRGetPolyOrder()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetInverseType(PC pc, PCPFLAREINVType *inverse_type)
{
   PetscFunctionBegin;
   PCAIRGetInverseType_c(&pc, inverse_type);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetCInverseType - Returns the type of approximate inverse used as the `PCAIR` C-point smoother

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. inverse_type - the approximate inverse type, one of `PFLAREINV_POWER`, `PFLAREINV_ARNOLDI`, `PFLAREINV_NEWTON`, `PFLAREINV_NEWTON_NO_EXTRA`, `PFLAREINV_NEUMANN`, `PFLAREINV_SAI`, `PFLAREINV_ISAI`, `PFLAREINV_WJACOBI`, or `PFLAREINV_JACOBI`

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetCInverseType()`, `PCPFLAREINVType`, `PCAIRGetInverseType()`, `PCAIRGetCPolyOrder()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetCInverseType(PC pc, PCPFLAREINVType *inverse_type)
{
   PetscFunctionBegin;
   PCAIRGetCInverseType_c(&pc, inverse_type);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetZType - Returns the type of grid-transfer (restriction) operator used by `PCAIR`

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. z_type - the grid-transfer operator type, one of `AIR_Z_PRODUCT`, `AIR_Z_LAIR`, or `AIR_Z_LAIR_SAI`

  Level: intermediate

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetZType()`, `PCAIRZType`, `PCAIRGetInverseType()`, `PCAIRGetLairDistance()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetZType(PC pc, PCAIRZType *z_type)
{
   PetscFunctionBegin;
   PCAIRGetZType_c(&pc, z_type);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetPolyOrder - Returns the polynomial order used by the `PCAIR` F-point smoother inverse

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_int - the polynomial order

  Level: intermediate

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetPolyOrder()`, `PCAIRGetInverseType()`, `PCAIRGetCPolyOrder()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetPolyOrder(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetPolyOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetLairDistance - Returns the distance of the grid-transfer operators used by `PCAIR` when the `PCAIRZType` is `AIR_Z_LAIR` or `AIR_Z_LAIR_SAI`

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_int - the lAIR distance

  Level: intermediate

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetLairDistance()`, `PCAIRZType`, `PCAIRGetZType()`, `PCAIRGetInverseType()`, `PCAIRGetInverseSparsityOrder()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetLairDistance(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetLairDistance_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetInverseSparsityOrder - Returns the power of the operator matrix used as the sparsity pattern of assembled `PCAIR` approximate inverses

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_int - the power of the operator matrix used as the sparsity of assembled inverses

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetInverseSparsityOrder()`, `PCAIRGetInverseType()`, `PCAIRGetCInverseSparsityOrder()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetInverseSparsityOrder(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetInverseSparsityOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetCPolyOrder - Returns the polynomial order used by the `PCAIR` C-point smoother inverse

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_int - the polynomial order

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetCPolyOrder()`, `PCAIRGetPolyOrder()`, `PCAIRGetCInverseType()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetCPolyOrder(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetCPolyOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetCInverseSparsityOrder - Returns the power of the operator matrix used as the sparsity pattern of assembled `PCAIR` C-point approximate inverses

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_int - the power of the operator matrix used as the sparsity of assembled inverses for the C-point smooth

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetCInverseSparsityOrder()`, `PCAIRGetInverseSparsityOrder()`, `PCAIRGetCInverseType()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetCInverseSparsityOrder(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetCInverseSparsityOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetCoarsestInverseType - Returns the type of approximate inverse used as the `PCAIR` coarse-grid solver

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. inverse_type - the approximate inverse type, one of `PFLAREINV_POWER`, `PFLAREINV_ARNOLDI`, `PFLAREINV_NEWTON`, `PFLAREINV_NEWTON_NO_EXTRA`, `PFLAREINV_NEUMANN`, `PFLAREINV_SAI`, `PFLAREINV_ISAI`, `PFLAREINV_WJACOBI`, or `PFLAREINV_JACOBI`

  Level: intermediate

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetCoarsestInverseType()`, `PCPFLAREINVType`, `PCAIRGetInverseType()`, `PCAIRGetCoarsestPolyOrder()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestInverseType(PC pc, PCPFLAREINVType *inverse_type)
{
   PetscFunctionBegin;
   PCAIRGetCoarsestInverseType_c(&pc, inverse_type);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetCoarsestPolyOrder - Returns the polynomial order used by the `PCAIR` coarse-grid solver

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_int - the polynomial order

  Level: intermediate

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetCoarsestPolyOrder()`, `PCAIRGetPolyOrder()`, `PCAIRGetCoarsestInverseType()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestPolyOrder(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetCoarsestPolyOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetCoarsestInverseSparsityOrder - Returns the power of the operator matrix used as the sparsity pattern of the assembled `PCAIR` coarse-grid inverse

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_int - the power of the operator matrix used as the sparsity of the assembled coarse-grid inverse

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetCoarsestInverseSparsityOrder()`, `PCAIRGetInverseSparsityOrder()`, `PCAIRGetCoarsestInverseType()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestInverseSparsityOrder(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetCoarsestInverseSparsityOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetCoarsestMatrixFreePolys - Returns whether `PCAIR` applies the coarse-grid polynomial solver matrix-free

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_bool - `PETSC_TRUE` if the coarse-grid polynomial solver is applied matrix-free

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetCoarsestMatrixFreePolys()`, `PCAIRGetMatrixFreePolys()`, `PCAIRGetCoarsestInverseType()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestMatrixFreePolys(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetCoarsestMatrixFreePolys_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetCoarsestDiagScalePolys - Returns whether `PCAIR` diagonally scales before computing the coarse-grid polynomial approximate inverse

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_bool - `PETSC_TRUE` if diagonally scaling before computing the coarse-grid polynomial inverse

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetCoarsestDiagScalePolys()`, `PCAIRGetDiagScalePolys()`, `PCAIRGetCoarsestInverseType()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestDiagScalePolys(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetCoarsestDiagScalePolys_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetCoarsestSubcomm - Returns whether `PCAIR` computes the coarse-grid polynomial coefficients on a subcommunicator

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_bool - `PETSC_TRUE` if the coarse-grid polynomial coefficients are computed on a subcommunicator

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetCoarsestSubcomm()`, `PCAIRGetSubcomm()`, `PCAIRGetCoarsestInverseType()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetCoarsestSubcomm(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetCoarsestSubcomm_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetRDrop - Returns the relative drop tolerance applied to R by `PCAIR` on each level after it is built

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_real - the relative (inf norm) drop tolerance applied to R

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetRDrop()`, `PCAIRGetADrop()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetRDrop(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetRDrop_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetADrop - Returns the relative drop tolerance applied to the coarse matrix by `PCAIR` on each level after it is built

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_real - the relative (inf norm) drop tolerance applied to the coarse matrix

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetADrop()`, `PCAIRGetRDrop()`, `PCAIRGetALump()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetADrop(PC pc, PetscReal *input_real)
{
   PetscFunctionBegin;
   PCAIRGetADrop_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetALump - Returns whether `PCAIR` lumps to the diagonal rather than drops when forming the coarse matrix

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_bool - `PETSC_TRUE` if lumping to the diagonal rather than dropping when forming the coarse matrix

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetALump()`, `PCAIRGetADrop()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetALump(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetALump_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetReuseSparsity - Returns whether `PCAIR` reuses the sparsity of the multigrid hierarchy during setup

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_bool - `PETSC_TRUE` if the sparsity of the hierarchy is reused during setup

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetReuseSparsity()`, `PCAIRGetReusePolyCoeffs()`, `PCAIRGetReuseAmount()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetReuseSparsity(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetReuseSparsity_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetReusePolyCoeffs - Returns whether `PCAIR` also reuses the GMRES polynomial coefficients when reusing the sparsity of the hierarchy

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_bool - `PETSC_TRUE` if the GMRES polynomial coefficients are also reused

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetReusePolyCoeffs()`, `PCAIRGetReuseSparsity()`, `PCAIRGetPolyCoeffs()`, `PCAIRSetPolyCoeffs()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetReusePolyCoeffs(PC pc, PetscBool *input_bool)
{
   PetscFunctionBegin;
   PCAIRGetReusePolyCoeffs_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@C
  PCAIRGetPolyCoeffs - Returns the polynomial coefficients stored by `PCAIR` for a given level and inverse after the last `PCSetUp()`

  Not Collective

  Input Parameters:
+ pc            - the `PCAIR` preconditioner context
. petsc_level   - the level in the hierarchy
- which_inverse - which inverse's coefficients to return, one of `COEFFS_INV_AFF`, `COEFFS_INV_AFF_DROPPED`, `COEFFS_INV_ACC`, or `COEFFS_INV_COARSE`

  Output Parameters:
+ coeffs_ptr - pointer to the array of polynomial coefficients
. row_size   - the number of rows in the coefficient array
- col_size   - the number of columns in the coefficient array

  Level: advanced

  Note:
  This routine returns a pointer into the `PCAIR` object itself, valid only until the next `PCSetUp()` or
  `PCReset()` call; copy the coefficients yourself if you need to save or restore them later. This differs from
  the Fortran interface to this routine, which returns a copy in an allocatable array that knows its own size.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetPolyCoeffs()`, `PCAIRGetReusePolyCoeffs()`, `WhichInverseType`, `PCSetUp()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetPolyCoeffs(PC pc, PetscInt petsc_level, int which_inverse, PetscReal **coeffs_ptr, PetscInt *row_size, PetscInt *col_size)
{
   PetscFunctionBegin;
   PCAIRGetPolyCoeffs_c(&pc,petsc_level, which_inverse, \
      coeffs_ptr, row_size, col_size);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetGridComplexity - Returns the grid complexity of the `PCAIR` multigrid hierarchy

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. complexity - the grid complexity of the hierarchy, or -1 if `PCSetUp()` has not yet been called

  Level: advanced

  Note:
  Grid complexity is the total number of unknowns summed over all levels of the hierarchy, divided by the number of unknowns on the finest level.

.seealso: [](ch_ksp), `PCAIR`, `PCSetUp()`, `PCAIRGetNumLevels()`, `PCAIRGetOperatorComplexity()`, `PCAIRGetCycleComplexity()`, `PCAIRGetStorageComplexity()`, `PCAIRGetReuseStorageComplexity()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetGridComplexity(PC pc, PetscReal *complexity)
{
   PetscFunctionBegin;
   PCAIRGetGridComplexity_c(&pc, complexity);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetOperatorComplexity - Returns the operator complexity of the `PCAIR` multigrid hierarchy

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. complexity - the operator complexity of the hierarchy, or -1 if `PCSetUp()` has not yet been called

  Level: advanced

  Note:
  Operator complexity is the total number of nonzeros summed over the coarse-grid matrices on all levels of the hierarchy, divided by the number of nonzeros in the finest-level matrix.

.seealso: [](ch_ksp), `PCAIR`, `PCSetUp()`, `PCAIRGetNumLevels()`, `PCAIRGetGridComplexity()`, `PCAIRGetCycleComplexity()`, `PCAIRGetStorageComplexity()`, `PCAIRGetReuseStorageComplexity()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetOperatorComplexity(PC pc, PetscReal *complexity)
{
   PetscFunctionBegin;
   PCAIRGetOperatorComplexity_c(&pc, complexity);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetCycleComplexity - Returns the cycle complexity of the `PCAIR` multigrid hierarchy

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. complexity - the cycle complexity of the hierarchy, or -1 if `PCSetUp()` has not yet been called

  Level: advanced

  Note:
  Cycle complexity is the total number of nonzeros touched while applying a single multigrid V-cycle, divided by the number of nonzeros in the finest-level matrix.

.seealso: [](ch_ksp), `PCAIR`, `PCSetUp()`, `PCAIRGetNumLevels()`, `PCAIRGetGridComplexity()`, `PCAIRGetOperatorComplexity()`, `PCAIRGetStorageComplexity()`, `PCAIRGetReuseStorageComplexity()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetCycleComplexity(PC pc, PetscReal *complexity)
{
   PetscFunctionBegin;
   PCAIRGetCycleComplexity_c(&pc, complexity);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetStorageComplexity - Returns the storage complexity of the `PCAIR` multigrid hierarchy

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. complexity - the storage complexity of the hierarchy, or -1 if `PCSetUp()` has not yet been called

  Level: advanced

  Note:
  Storage complexity is the total number of nonzeros actually stored by the hierarchy (accounting for the reduced storage possible with F-point-only up smoothing), divided by the number of nonzeros in the finest-level matrix.

.seealso: [](ch_ksp), `PCAIR`, `PCSetUp()`, `PCAIRGetNumLevels()`, `PCAIRGetGridComplexity()`, `PCAIRGetOperatorComplexity()`, `PCAIRGetCycleComplexity()`, `PCAIRGetReuseStorageComplexity()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetStorageComplexity(PC pc, PetscReal *complexity)
{
   PetscFunctionBegin;
   PCAIRGetStorageComplexity_c(&pc, complexity);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetReuseStorageComplexity - Returns the reuse storage complexity of the `PCAIR` multigrid hierarchy

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. complexity - the reuse storage complexity of the hierarchy (0 when reuse is disabled), or -1 if `PCSetUp()` has not yet been called

  Level: advanced

  Note:
  Reuse storage complexity is the total number of nonzeros in the matrices and index sets kept in memory to accelerate a subsequent setup, divided by the number of nonzeros in the finest-level matrix.

.seealso: [](ch_ksp), `PCAIR`, `PCSetUp()`, `PCAIRGetNumLevels()`, `PCAIRGetGridComplexity()`, `PCAIRGetOperatorComplexity()`, `PCAIRGetCycleComplexity()`, `PCAIRGetStorageComplexity()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetReuseStorageComplexity(PC pc, PetscReal *complexity)
{
   PetscFunctionBegin;
   PCAIRGetReuseStorageComplexity_c(&pc, complexity);
   PetscFunctionReturn(PETSC_SUCCESS);
}

// Set routines

/*@
  PCAIRSetPrintStatsTimings - Sets whether `PCAIR` prints statistics about the multigrid hierarchy and timings

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_bool - `PETSC_TRUE` to print statistics about the multigrid hierarchy and timings

  Options Database Key:
. -pc_air_print_stats_timings (true|false) - print statistics about the multigrid hierarchy and timings; defaults to false

  Level: advanced

  Note:
  Computing these statistics requires some parallel reductions, so this is disabled by default.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetPrintStatsTimings()`, `PCAIRGetNumLevels()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetPrintStatsTimings(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   // No need to reset if this changes
   PCAIRSetPrintStatsTimings_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCAIRSetMaxLevels - Sets the maximum number of levels allowed in the `PCAIR` multigrid hierarchy

  Logically Collective

  Input Parameters:
+ pc        - the `PCAIR` preconditioner context
- input_int - the maximum number of levels in the multigrid hierarchy

  Options Database Key:
. -pc_air_max_levels input_int - the maximum number of levels in the multigrid hierarchy; defaults to 300

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetMaxLevels()`, `PCAIRSetCoarseEqLimit()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetMaxLevels(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PCAIRSetMaxLevels_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetCoarseEqLimit - Sets the minimum number of global unknowns allowed on the coarsest grid of the `PCAIR` hierarchy

  Logically Collective

  Input Parameters:
+ pc        - the `PCAIR` preconditioner context
- input_int - the minimum number of global unknowns allowed on the coarse grid

  Options Database Key:
. -pc_air_coarse_eq_limit input_int - the minimum number of global unknowns on the coarse grid; defaults to 6

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetCoarseEqLimit()`, `PCAIRSetMaxLevels()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetCoarseEqLimit(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PCAIRSetCoarseEqLimit_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetAutoTruncateStartLevel - Sets the level from which `PCAIR` builds and evaluates a coarse-grid solver to decide whether the hierarchy can be truncated there

  Logically Collective

  Input Parameters:
+ pc        - the `PCAIR` preconditioner context
- input_int - the level from which to build a coarse solver and test whether the hierarchy can be truncated

  Options Database Key:
. -pc_air_auto_truncate_start_level input_int - use auto truncation from this level onwards; defaults to -1

  Level: advanced

  Note:
  A value of -1 disables auto truncation, so the hierarchy is always built down to `PCAIRSetCoarseEqLimit()`.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetAutoTruncateStartLevel()`, `PCAIRSetAutoTruncateTol()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetAutoTruncateStartLevel(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PCAIRSetAutoTruncateStartLevel_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetAutoTruncateTol - Sets the relative tolerance used by `PCAIR` to decide if a coarse-grid solver is good enough to truncate the hierarchy

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_real - the relative tolerance used to determine if a coarse-grid solver is good enough to truncate the hierarchy

  Options Database Key:
. -pc_air_auto_truncate_tol input_real - tolerance to use with auto truncation; defaults to 1e-14

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetAutoTruncateTol()`, `PCAIRSetAutoTruncateStartLevel()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetAutoTruncateTol(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PCAIRSetAutoTruncateTol_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetProcessorAgglom - Sets whether `PCAIR` uses a graph partitioner to repartition coarse grids and reduce the number of active MPI ranks

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_bool - `PETSC_TRUE` to repartition the coarse grids onto fewer MPI ranks as the hierarchy coarsens

  Options Database Key:
. -pc_air_processor_agglom (true|false) - use a graph partitioner to repartition coarse grids and reduce the number of active MPI ranks; defaults to true

  Level: advanced

  Note:
  Processor agglomeration reduces the number of active MPI ranks by `PCAIRSetProcessorAgglomFactor()` whenever the local to non-local nonzero ratio drops below `PCAIRSetProcessorAgglomRatio()`, or the average number of equations per rank drops below `PCAIRSetProcessEqLimit()`; it is only performed where necessary, not on every level, and the entire hierarchy remains on the original communicator.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetProcessorAgglom()`, `PCAIRSetProcessorAgglomRatio()`, `PCAIRSetProcessorAgglomFactor()`, `PCAIRSetProcessEqLimit()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetProcessorAgglom(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PCAIRSetProcessorAgglom_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetProcessorAgglomRatio - Sets the local to non-local nonzero ratio that triggers processor agglomeration in `PCAIR`

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_real - the local to non-local nonzero ratio that triggers processor agglomeration

  Options Database Key:
. -pc_air_processor_agglom_ratio ratio - the local to non-local nonzero ratio that triggers processor agglomeration; defaults to 2.0

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetProcessorAgglomRatio()`, `PCAIRSetProcessorAgglom()`, `PCAIRSetProcessorAgglomFactor()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetProcessorAgglomRatio(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PCAIRSetProcessorAgglomRatio_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetProcessorAgglomFactor - Sets the factor by which `PCAIR` reduces the number of active MPI ranks each time processor agglomeration occurs

  Logically Collective

  Input Parameters:
+ pc        - the `PCAIR` preconditioner context
- input_int - the factor by which to reduce the number of active MPI ranks

  Options Database Key:
. -pc_air_processor_agglom_factor factor - the factor by which to reduce the number of active MPI ranks; defaults to 2

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetProcessorAgglomFactor()`, `PCAIRSetProcessorAgglom()`, `PCAIRSetProcessorAgglomRatio()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetProcessorAgglomFactor(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PCAIRSetProcessorAgglomFactor_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetProcessEqLimit - Sets the average number of equations per MPI rank below which `PCAIR` triggers processor agglomeration

  Logically Collective

  Input Parameters:
+ pc        - the `PCAIR` preconditioner context
- input_int - the average number of equations per rank below which processor agglomeration is triggered

  Options Database Key:
. -pc_air_process_eq_limit eq_limit - trigger processor agglomeration if the average number of equations per rank drops below this value; defaults to 50

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetProcessEqLimit()`, `PCAIRSetProcessorAgglom()`, `PCAIRSetProcessorAgglomRatio()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetProcessEqLimit(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PCAIRSetProcessEqLimit_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetSubcomm - Sets whether `PCAIR` performs reductions for arnoldi or newton polynomial inverses on a subcommunicator that excludes empty MPI ranks

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_bool - `PETSC_TRUE` to perform the reductions on a subcommunicator excluding ranks with no rows

  Options Database Key:
. -pc_air_subcomm (true|false) - perform reductions for arnoldi/newton polynomial inverses on a subcommunicator excluding empty ranks; defaults to false

  Level: advanced

  Note:
  Only relevant after processor agglomeration (see `PCAIRSetProcessorAgglom()`) has left some MPI ranks with no rows, and only affects the basis-building reductions of the `PFLAREINV_ARNOLDI`, `PFLAREINV_NEWTON`, and `PFLAREINV_NEWTON_NO_EXTRA` polynomial inverse types (see `PCAIRSetInverseType()`). It does not apply to the `PFLAREINV_POWER` basis, which needs no such reductions; enabling it with the power basis is an error.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetSubcomm()`, `PCAIRSetProcessorAgglom()`, `PCAIRSetInverseType()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetSubcomm(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   // No need to reset if this changes
   PCAIRSetSubcomm_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetStrongThreshold - Sets the strong threshold used in the `PCAIR` CF splitting

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_real - the strong threshold used in the CF splitting

  Options Database Key:
. -pc_air_strong_threshold strong_threshold - the strong threshold used in the CF splitting; defaults to 0.5

  Level: intermediate

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetStrongThreshold()`, `PCAIRSetCFSplittingType()`, `CFSplittingType`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetStrongThreshold(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PCAIRSetStrongThreshold_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetDDCIts - Sets the number of diagonal-dominance-conversion (DDC) iterations used by the `PCAIR` `CF_PMISR_DDC` CF splitting

  Logically Collective

  Input Parameters:
+ pc        - the `PCAIR` preconditioner context
- input_int - the number of DDC iterations

  Options Database Key:
. -pc_air_ddc_its ddc_its - the number of DDC iterations; defaults to 1

  Level: advanced

  Note:
  Only used by the `CF_PMISR_DDC` CF splitting type (see `PCAIRSetCFSplittingType()`).

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetDDCIts()`, `PCAIRSetDDCFraction()`, `PCAIRSetCFSplittingType()`, `CFSplittingType`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetDDCIts(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PCAIRSetDDCIts_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetDDCFraction - Sets the fraction of local F points converted to C points by diagonal dominance in the `PCAIR` `CF_PMISR_DDC` CF splitting

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_real - the local fraction of F points to convert to C points, or (if negative) minus the diagonal-dominance ratio threshold

  Options Database Key:
. -pc_air_ddc_fraction fraction - local fraction of F points converted to C by diagonal dominance; if negative, convert any row whose diagonal-dominance ratio is less than the absolute value; defaults to 0.1

  Level: advanced

  Note:
  Only used by the `CF_PMISR_DDC` CF splitting type (see `PCAIRSetCFSplittingType()`).

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetDDCFraction()`, `PCAIRSetDDCIts()`, `PCAIRSetCFSplittingType()`, `CFSplittingType`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetDDCFraction(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PCAIRSetDDCFraction_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetCFSplittingType - Sets the coarse/fine (CF) splitting algorithm used by `PCAIR`

  Logically Collective

  Input Parameters:
+ pc                - the `PCAIR` preconditioner context
- cf_splitting_type - the CF splitting algorithm, one of `CF_PMISR_DDC`, `CF_DIAG_DOM`, `CF_PMIS`, `CF_PMIS_DIST2`, `CF_AGG`, or `CF_PMIS_AGG`

  Options Database Key:
. -pc_air_cf_splitting_type (pmisr_ddc|diag_dom|pmis|pmis_dist2|agg|pmis_agg) - the CF splitting algorithm; defaults to pmisr_ddc

  Level: intermediate

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetCFSplittingType()`, `CFSplittingType`, `PCAIRSetStrongThreshold()`, `PCAIRSetMaxLubySteps()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetCFSplittingType(PC pc, CFSplittingType cf_splitting_type)
{
   PetscFunctionBegin;
   PCAIRSetCFSplittingType_c(&pc, cf_splitting_type);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetMaxLubySteps - Sets the maximum number of Luby steps used by the `PCAIR` CF splitting

  Logically Collective

  Input Parameters:
+ pc        - the `PCAIR` preconditioner context
- input_int - the maximum number of Luby steps

  Options Database Key:
. -pc_air_max_luby_steps max_luby_steps - the maximum number of Luby steps; a negative value uses as many steps as necessary; defaults to -1

  Level: advanced

  Note:
  Only used by the `CF_PMISR_DDC`, `CF_DIAG_DOM`, `CF_PMIS`, and `CF_PMIS_DIST2` CF splitting types (see `PCAIRSetCFSplittingType()`). A negative value performs as many Luby steps as necessary, at the cost of a parallel reduction after every step.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetMaxLubySteps()`, `PCAIRSetCFSplittingType()`, `CFSplittingType`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetMaxLubySteps(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PCAIRSetMaxLubySteps_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@C
  PCAIRSetSmoothType - Sets the type and number of smooths used by the `PCAIR` reduction multigrid

  Logically Collective

  Input Parameters:
+ pc           - the `PCAIR` preconditioner context
- input_string - the smoothing pattern, any sequence of `f` and `c` characters giving the type and number of smooths (for example `ff`, `fc`, `fcf`, `ffc`, ...)

  Options Database Key:
. -pc_air_smooth_type input_string - the type and number of smooths, any sequence of f and c characters (for example ff, fc, fcf); defaults to ff

  Level: intermediate

  Note:
  Each `f` performs a smooth on the F points and each `c` a smooth on the C points; the string may be any combination and length, not only `ff`, `fc`, or `fcf`.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetSmoothType()`, `PCAIRSetInverseType()`, `PCAIRSetFullSmoothingUpAndDown()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetSmoothType(PC pc, const char* input_string)
{
   PetscFunctionBegin;
   PCAIRSetSmoothType_c(&pc, input_string);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetDiagScalePolys - Sets whether `PCAIR` diagonally scales before computing a polynomial approximate inverse

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_bool - `PETSC_TRUE` to diagonally scale before computing a polynomial inverse

  Options Database Key:
. -pc_air_diag_scale_polys (true|false) - diagonally scale before computing a polynomial inverse; defaults to false

  Level: advanced

  Note:
  Only relevant if using a polynomial inverse type (see `PCAIRSetInverseType()`); if the inverse type is `PFLAREINV_NEUMANN` this is always forced true and cannot be overridden.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetDiagScalePolys()`, `PCAIRSetInverseType()`, `PCAIRSetMatrixFreePolys()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetDiagScalePolys(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PCAIRSetDiagScalePolys_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetMatrixFreePolys - Sets whether `PCAIR` applies polynomial smoothers matrix-free where possible

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_bool - `PETSC_TRUE` to apply polynomial smoothers matrix-free where possible

  Options Database Key:
. -pc_air_matrix_free_polys (true|false) - apply polynomial smoothers matrix-free where possible; defaults to false

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetMatrixFreePolys()`, `PCAIRSetInverseType()`, `PCAIRSetDiagScalePolys()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetMatrixFreePolys(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PCAIRSetMatrixFreePolys_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetOnePointClassicalProlong - Sets whether `PCAIR` uses a one-point classical prolongator instead of an approximate ideal prolongator

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_bool - `PETSC_TRUE` to use a one-point classical prolongator

  Options Database Key:
. -pc_air_one_point_classical_prolong (true|false) - use a one-point classical prolongator instead of an approximate ideal prolongator; defaults to true

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetOnePointClassicalProlong()`, `PCAIRSetSymmetric()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetOnePointClassicalProlong(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PCAIRSetOnePointClassicalProlong_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetFullSmoothingUpAndDown - Sets whether `PCAIR` smooths all points on the up and down sweeps, instead of the default down F and C smoothing

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_bool - `PETSC_TRUE` to smooth all points on both the up and down sweeps

  Options Database Key:
. -pc_air_full_smoothing_up_and_down (true|false) - smooth all points up and down, instead of the default down F and C smoothing; defaults to false

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetFullSmoothingUpAndDown()`, `PCAIRSetSmoothType()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetFullSmoothingUpAndDown(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PCAIRSetFullSmoothingUpAndDown_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetSymmetric - Sets whether `PCAIR` defines the prolongator as the transpose of the restrictor

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_bool - `PETSC_TRUE` to use symmetric grid-transfer operators, defining the prolongator as R^T

  Options Database Key:
. -pc_air_symmetric (true|false) - define the prolongator as R^T, giving symmetric grid-transfer operators; defaults to false

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetSymmetric()`, `PCAIRSetOnePointClassicalProlong()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetSymmetric(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PCAIRSetSymmetric_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetConstrainW - Sets whether `PCAIR` applies constraints to the prolongator

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_bool - `PETSC_TRUE` to apply constraints to the prolongator

  Options Database Key:
. -pc_air_constrain_w (true|false) - apply constraints to the prolongator; defaults to false

  Level: advanced

  Note:
  By default this smooths the constant vector and forces the prolongator to interpolate it exactly. Use `MatSetNearNullSpace()` on the operator matrix to supply other vectors to constrain instead.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetConstrainW()`, `PCAIRSetConstrainZ()`, `MatSetNearNullSpace()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetConstrainW(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PCAIRSetConstrainW_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetConstrainZ - Sets whether `PCAIR` applies constraints to the restrictor

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_bool - `PETSC_TRUE` to apply constraints to the restrictor

  Options Database Key:
. -pc_air_constrain_z (true|false) - apply constraints to the restrictor; defaults to false

  Level: advanced

  Note:
  By default this smooths the constant vector and forces the restrictor to restrict it exactly. Use `MatSetNearNullSpace()` on the operator matrix to supply other vectors to constrain instead.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetConstrainZ()`, `PCAIRSetConstrainW()`, `MatSetNearNullSpace()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetConstrainZ(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PCAIRSetConstrainZ_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetImproveWIts - Sets the number of Richardson iterations used by `PCAIR` to improve the approximate prolongator

  Logically Collective

  Input Parameters:
+ pc        - the `PCAIR` preconditioner context
- input_int - the number of Richardson iterations

  Options Database Key:
. -pc_air_improve_w_its its - number of Richardson iterations to improve the approximate prolongator, using the existing W as the initial guess; defaults to 0

  Level: advanced

  Note:
  The Richardson iteration is preconditioned with the approximate inverse of Aff.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetImproveWIts()`, `PCAIRSetImproveZIts()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetImproveWIts(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PCAIRSetImproveWIts_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetImproveZIts - Sets the number of Richardson iterations used by `PCAIR` to improve the approximate restrictor

  Logically Collective

  Input Parameters:
+ pc        - the `PCAIR` preconditioner context
- input_int - the number of Richardson iterations

  Options Database Key:
. -pc_air_improve_z_its its - number of Richardson iterations to improve the approximate restrictor, using the existing Z as the initial guess; defaults to 0

  Level: advanced

  Note:
  The Richardson iteration is preconditioned with the diagonal of the approximate inverse of Aff.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetImproveZIts()`, `PCAIRSetImproveWIts()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetImproveZIts(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PCAIRSetImproveZIts_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetStrongRThreshold - Sets the threshold used by `PCAIR` to drop entries when forming the grid-transfer operators

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_real - the strong R threshold

  Options Database Key:
. -pc_air_strong_r_threshold threshold - threshold to drop when forming the grid-transfer operators; defaults to 0.0

  Level: advanced

  Note:
  This only applies when computing Z; for example, if a GMRES polynomial approximation to Aff^-1 is built, this dropping is applied, Z is computed, and then an Aff^-1 approximation without the dropping is rebuilt for smoothing.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetStrongRThreshold()`, `PCAIRSetZType()`, `PCAIRSetStrongThreshold()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetStrongRThreshold(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PCAIRSetStrongRThreshold_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetInverseType - Sets the type of approximate inverse used as the `PCAIR` F-point smoother

  Logically Collective

  Input Parameters:
+ pc           - the `PCAIR` preconditioner context
- inverse_type - the approximate inverse type, one of `PFLAREINV_POWER`, `PFLAREINV_ARNOLDI`, `PFLAREINV_NEWTON`, `PFLAREINV_NEWTON_NO_EXTRA`, `PFLAREINV_NEUMANN`, `PFLAREINV_SAI`, `PFLAREINV_ISAI`, `PFLAREINV_WJACOBI`, or `PFLAREINV_JACOBI`

  Options Database Key:
. -pc_air_inverse_type (power|arnoldi|newton|newton_no_extra|neumann|sai|isai|wjacobi|jacobi) - the approximate inverse type used for the F-point smoother; defaults to arnoldi

  Level: intermediate

  Note:
  Together with `PCAIRSetZType()` this selects the reduction multigrid method, for example product + arnoldi gives AIRG and lair + wjacobi gives lAIR; see `PCPFLAREINVType` for what each value means.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetInverseType()`, `PCPFLAREINVType`, `PCAIRSetCInverseType()`, `PCAIRSetZType()`, `PCAIRSetPolyOrder()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetInverseType(PC pc, PCPFLAREINVType inverse_type)
{
   PetscFunctionBegin;
   PCAIRSetInverseType_c(&pc, inverse_type);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetCInverseType - Sets the type of approximate inverse used as the `PCAIR` C-point smoother

  Logically Collective

  Input Parameters:
+ pc           - the `PCAIR` preconditioner context
- inverse_type - the approximate inverse type, one of `PFLAREINV_POWER`, `PFLAREINV_ARNOLDI`, `PFLAREINV_NEWTON`, `PFLAREINV_NEWTON_NO_EXTRA`, `PFLAREINV_NEUMANN`, `PFLAREINV_SAI`, `PFLAREINV_ISAI`, `PFLAREINV_WJACOBI`, or `PFLAREINV_JACOBI`

  Options Database Key:
. -pc_air_c_inverse_type (power|arnoldi|newton|newton_no_extra|neumann|sai|isai|wjacobi|jacobi) - the approximate inverse type used for the C-point smooth; if unset, defaults to the same as `-pc_air_inverse_type`

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetCInverseType()`, `PCPFLAREINVType`, `PCAIRSetInverseType()`, `PCAIRSetCPolyOrder()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetCInverseType(PC pc, PCPFLAREINVType inverse_type)
{
   PetscFunctionBegin;
   PCAIRSetCInverseType_c(&pc, inverse_type);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetZType - Sets the type of grid-transfer (restriction) operator used by `PCAIR`

  Logically Collective

  Input Parameters:
+ pc     - the `PCAIR` preconditioner context
- z_type - the grid-transfer operator type, one of `AIR_Z_PRODUCT`, `AIR_Z_LAIR`, or `AIR_Z_LAIR_SAI`

  Options Database Key:
. -pc_air_z_type (product|lair|lair_sai) - the grid-transfer operator type; defaults to product

  Level: intermediate

  Note:
  Together with `PCAIRSetInverseType()` this selects the reduction multigrid method, for example product + arnoldi gives AIRG and lair + wjacobi gives lAIR; see `PCAIRZType` for what each value means.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetZType()`, `PCAIRZType`, `PCAIRSetInverseType()`, `PCAIRSetLairDistance()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetZType(PC pc, PCAIRZType z_type)
{
   PetscFunctionBegin;
   PCAIRSetZType_c(&pc, z_type);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetLairDistance - Sets the distance of the grid-transfer operators used by `PCAIR` when the `PCAIRZType` is `AIR_Z_LAIR` or `AIR_Z_LAIR_SAI`

  Logically Collective

  Input Parameters:
+ pc        - the `PCAIR` preconditioner context
- input_int - the lAIR distance

  Options Database Key:
. -pc_air_lair_distance lair_distance - if the Z type is lair or lair_sai, the distance of the grid-transfer operators; defaults to 2

  Level: intermediate

  Note:
  This allows lAIR to be computed out to a given distance while using a different sparsity for the smoothers. If the `PCAIRZType` is `AIR_Z_PRODUCT` this option is ignored, and the distance is instead determined by `-pc_air_inverse_sparsity_order` + 1.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetLairDistance()`, `PCAIRZType`, `PCAIRSetZType()`, `PCAIRSetInverseType()`, `PCAIRSetInverseSparsityOrder()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetLairDistance(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PCAIRSetLairDistance_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetPolyOrder - Sets the polynomial order used by the `PCAIR` F-point smoother inverse

  Logically Collective

  Input Parameters:
+ pc        - the `PCAIR` preconditioner context
- input_int - the polynomial order

  Options Database Key:
. -pc_air_poly_order poly_order - the polynomial order if using a polynomial inverse type; defaults to 6

  Level: intermediate

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetPolyOrder()`, `PCAIRSetInverseType()`, `PCAIRSetCPolyOrder()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetPolyOrder(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PCAIRSetPolyOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetInverseSparsityOrder - Sets the power of the operator matrix used as the sparsity pattern of assembled `PCAIR` approximate inverses

  Logically Collective

  Input Parameters:
+ pc        - the `PCAIR` preconditioner context
- input_int - the power of the operator matrix used as the sparsity of assembled inverses

  Options Database Key:
. -pc_air_inverse_sparsity_order sparsity_order - the power of the operator matrix used as the sparsity of assembled inverses; defaults to 1

  Level: advanced

  Note:
  When `PCAIRSetZType()` is `AIR_Z_PRODUCT` this also sets the distance of the grid-transfer operators, since that distance is `-pc_air_inverse_sparsity_order` + 1.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetInverseSparsityOrder()`, `PCAIRSetInverseType()`, `PCAIRSetCInverseSparsityOrder()`, `PCAIRSetLairDistance()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetInverseSparsityOrder(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PCAIRSetInverseSparsityOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetCPolyOrder - Sets the polynomial order used by the `PCAIR` C-point smoother inverse

  Logically Collective

  Input Parameters:
+ pc        - the `PCAIR` preconditioner context
- input_int - the polynomial order

  Options Database Key:
. -pc_air_c_poly_order poly_order - the polynomial order for the C-point smooth if using a polynomial inverse type; if unset, defaults to the same as `-pc_air_poly_order`

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetCPolyOrder()`, `PCAIRSetPolyOrder()`, `PCAIRSetCInverseType()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetCPolyOrder(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PCAIRSetCPolyOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetCInverseSparsityOrder - Sets the power of the operator matrix used as the sparsity pattern of assembled `PCAIR` C-point approximate inverses

  Logically Collective

  Input Parameters:
+ pc        - the `PCAIR` preconditioner context
- input_int - the power of the operator matrix used as the sparsity of assembled inverses for the C-point smooth

  Options Database Key:
. -pc_air_c_inverse_sparsity_order sparsity_order - the power of the operator matrix used as the sparsity of assembled inverses for the C-point smooth; if unset, defaults to the same as `-pc_air_inverse_sparsity_order`

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetCInverseSparsityOrder()`, `PCAIRSetInverseSparsityOrder()`, `PCAIRSetCInverseType()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetCInverseSparsityOrder(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PCAIRSetCInverseSparsityOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetCoarsestInverseType - Sets the type of approximate inverse used as the `PCAIR` coarse-grid solver

  Logically Collective

  Input Parameters:
+ pc           - the `PCAIR` preconditioner context
- inverse_type - the approximate inverse type, one of `PFLAREINV_POWER`, `PFLAREINV_ARNOLDI`, `PFLAREINV_NEWTON`, `PFLAREINV_NEWTON_NO_EXTRA`, `PFLAREINV_NEUMANN`, `PFLAREINV_SAI`, `PFLAREINV_ISAI`, `PFLAREINV_WJACOBI`, or `PFLAREINV_JACOBI`

  Options Database Key:
. -pc_air_coarsest_inverse_type (power|arnoldi|newton|newton_no_extra|neumann|sai|isai|wjacobi|jacobi) - the approximate inverse type used as the coarse-grid solver; defaults to arnoldi

  Level: intermediate

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetCoarsestInverseType()`, `PCPFLAREINVType`, `PCAIRSetInverseType()`, `PCAIRSetCoarsestPolyOrder()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestInverseType(PC pc, PCPFLAREINVType inverse_type)
{
   PetscFunctionBegin;
   PCAIRSetCoarsestInverseType_c(&pc, inverse_type);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetCoarsestPolyOrder - Sets the polynomial order used by the `PCAIR` coarse-grid solver

  Logically Collective

  Input Parameters:
+ pc        - the `PCAIR` preconditioner context
- input_int - the polynomial order

  Options Database Key:
. -pc_air_coarsest_poly_order poly_order - the polynomial order of the coarse-grid solver if using a polynomial inverse type; defaults to 6

  Level: intermediate

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetCoarsestPolyOrder()`, `PCAIRSetPolyOrder()`, `PCAIRSetCoarsestInverseType()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestPolyOrder(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PCAIRSetCoarsestPolyOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetCoarsestInverseSparsityOrder - Sets the power of the operator matrix used as the sparsity pattern of the assembled `PCAIR` coarse-grid inverse

  Logically Collective

  Input Parameters:
+ pc        - the `PCAIR` preconditioner context
- input_int - the power of the operator matrix used as the sparsity of the assembled coarse-grid inverse

  Options Database Key:
. -pc_air_coarsest_inverse_sparsity_order sparsity_order - the power of the operator matrix used as the sparsity of the assembled coarse-grid inverse; defaults to 1

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetCoarsestInverseSparsityOrder()`, `PCAIRSetInverseSparsityOrder()`, `PCAIRSetCoarsestInverseType()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestInverseSparsityOrder(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PCAIRSetCoarsestInverseSparsityOrder_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetCoarsestMatrixFreePolys - Sets whether `PCAIR` applies the coarse-grid polynomial solver matrix-free

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_bool - `PETSC_TRUE` to apply the coarse-grid polynomial solver matrix-free

  Options Database Key:
. -pc_air_coarsest_matrix_free_polys (true|false) - apply the coarse-grid polynomial solver matrix-free; defaults to false

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetCoarsestMatrixFreePolys()`, `PCAIRSetMatrixFreePolys()`, `PCAIRSetCoarsestInverseType()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestMatrixFreePolys(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PCAIRSetCoarsestMatrixFreePolys_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetCoarsestDiagScalePolys - Sets whether `PCAIR` diagonally scales before computing the coarse-grid polynomial approximate inverse

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_bool - `PETSC_TRUE` to diagonally scale before computing the coarse-grid polynomial inverse

  Options Database Key:
. -pc_air_coarsest_diag_scale_polys (true|false) - diagonally scale before computing the coarse-grid polynomial inverse; defaults to false

  Level: advanced

  Note:
  Only relevant if using a polynomial inverse type (see `PCAIRSetCoarsestInverseType()`); if the coarsest inverse type is `PFLAREINV_NEUMANN` this is always forced true and cannot be overridden.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetCoarsestDiagScalePolys()`, `PCAIRSetDiagScalePolys()`, `PCAIRSetCoarsestInverseType()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestDiagScalePolys(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PCAIRSetCoarsestDiagScalePolys_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetCoarsestSubcomm - Sets whether `PCAIR` computes the coarse-grid polynomial coefficients on a subcommunicator

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_bool - `PETSC_TRUE` to compute the coarse-grid polynomial coefficients on a subcommunicator

  Options Database Key:
. -pc_air_coarsest_subcomm (true|false) - compute the coarse-grid polynomial coefficients on a subcommunicator; defaults to false

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetCoarsestSubcomm()`, `PCAIRSetSubcomm()`, `PCAIRSetCoarsestInverseType()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetCoarsestSubcomm(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   // No need to reset if this changes
   PCAIRSetCoarsestSubcomm_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetRDrop - Sets the relative drop tolerance applied to R by `PCAIR` on each level after it is built

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_real - the relative (inf norm) drop tolerance applied to R

  Options Database Key:
. -pc_air_r_drop input_real - drop tolerance for R; defaults to 0.01

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetRDrop()`, `PCAIRSetADrop()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetRDrop(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PCAIRSetRDrop_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetADrop - Sets the relative drop tolerance applied to the coarse matrix by `PCAIR` on each level after it is built

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_real - the relative (inf norm) drop tolerance applied to the coarse matrix

  Options Database Key:
. -pc_air_a_drop input_real - drop tolerance for the coarse matrix; defaults to 1e-4

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetADrop()`, `PCAIRSetRDrop()`, `PCAIRSetALump()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetADrop(PC pc, PetscReal input_real)
{
   PetscFunctionBegin;
   PCAIRSetADrop_c(&pc, input_real);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetALump - Sets whether `PCAIR` lumps to the diagonal rather than drops when forming the coarse matrix

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_bool - `PETSC_TRUE` to lump to the diagonal rather than drop when forming the coarse matrix

  Options Database Key:
. -pc_air_a_lump (true|false) - lump to the diagonal rather than drop for the coarse matrix; defaults to false

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetALump()`, `PCAIRSetADrop()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetALump(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PCAIRSetALump_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetReuseSparsity - Sets whether `PCAIR` reuses the sparsity of the multigrid hierarchy during setup

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_bool - `PETSC_TRUE` to reuse the sparsity of the hierarchy during setup

  Options Database Key:
. -pc_air_reuse_sparsity (true|false) - reuse the sparsity of the hierarchy (CF splitting, repartitioning, symbolic matrix-matrix products) during setup; defaults to false

  Level: advanced

  Note:
  If the matrix has changed too much, convergence may suffer.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetReuseSparsity()`, `PCAIRSetReusePolyCoeffs()`, `PCAIRSetReuseAmount()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetReuseSparsity(PC pc, PetscBool input_bool)
{  
   PetscFunctionBegin; 
   PCAIRSetReuseSparsity_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS); 
}
/*@
  PCAIRSetReusePolyCoeffs - Sets whether `PCAIR` also reuses the GMRES polynomial coefficients when reusing the sparsity of the hierarchy

  Logically Collective

  Input Parameters:
+ pc         - the `PCAIR` preconditioner context
- input_bool - `PETSC_TRUE` to also reuse the GMRES polynomial coefficients

  Options Database Key:
. -pc_air_reuse_poly_coeffs (true|false) - also reuse the GMRES polynomial coefficients when `-pc_air_reuse_sparsity` is set; defaults to false

  Level: advanced

  Note:
  Only useful when regenerating the hierarchy for the same matrix, with coefficients stored and restored using `PCAIRGetPolyCoeffs()` and `PCAIRSetPolyCoeffs()`; the coefficients are very sensitive to changes in the matrix.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetReusePolyCoeffs()`, `PCAIRSetReuseSparsity()`, `PCAIRGetPolyCoeffs()`, `PCAIRSetPolyCoeffs()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetReusePolyCoeffs(PC pc, PetscBool input_bool)
{
   PetscFunctionBegin;
   PCAIRSetReusePolyCoeffs_c(&pc, input_bool);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRGetReuseAmount - Returns how much data `PCAIR` stores when reusing the sparsity of the multigrid hierarchy

  Not Collective

  Input Parameter:
. pc - the `PCAIR` preconditioner context

  Output Parameter:
. input_int - the amount of data stored when `-pc_air_reuse_sparsity` is enabled

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRSetReuseAmount()`, `PCAIRGetReuseSparsity()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRGetReuseAmount(PC pc, PetscInt *input_int)
{
   PetscFunctionBegin;
   PCAIRGetReuseAmount_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PCAIRSetReuseAmount - Sets how much data `PCAIR` stores when reusing the sparsity of the multigrid hierarchy

  Logically Collective

  Input Parameters:
+ pc        - the `PCAIR` preconditioner context
- input_int - the amount of data to store when `-pc_air_reuse_sparsity` is enabled

  Options Database Key:
. -pc_air_reuse_amount amount - how much data to store when `-pc_air_reuse_sparsity` is enabled: 1 stores only the CF splitting and parallel repartitioning, 2 additionally stores everything needed to reuse sparsity in the SpGEMMs, and 3 stores everything; defaults to 3

  Level: advanced

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetReuseAmount()`, `PCAIRSetReuseSparsity()`
@*/
PETSC_EXTERN PetscErrorCode PCAIRSetReuseAmount(PC pc, PetscInt input_int)
{
   PetscFunctionBegin;
   PCAIRSetReuseAmount_c(&pc, input_int);
   PetscFunctionReturn(PETSC_SUCCESS);
}
/*@C
  PCAIRSetPolyCoeffs - Sets (copies in) the polynomial coefficients used by `PCAIR` for a given level and inverse

  Logically Collective

  Input Parameters:
+ pc            - the `PCAIR` preconditioner context
. petsc_level   - the level in the hierarchy
. which_inverse - which inverse's coefficients to set, one of `COEFFS_INV_AFF`, `COEFFS_INV_AFF_DROPPED`, `COEFFS_INV_ACC`, or `COEFFS_INV_COARSE`
. coeffs_ptr    - the array of polynomial coefficients to copy in
. row_size      - the number of rows in coeffs_ptr
- col_size      - the number of columns in coeffs_ptr

  Level: advanced

  Note:
  This routine copies the data from coeffs_ptr into the `PCAIR` object; the caller's array is not referenced after
  this call and may be freed or modified.

.seealso: [](ch_ksp), `PCAIR`, `PCAIRGetPolyCoeffs()`, `PCAIRSetReusePolyCoeffs()`, `WhichInverseType`
@*/
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
   PetscCall(PCAIRGetDiagScalePolys(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_diag_scale_polys", "Diagonally scale before computing polynomial inverse", "PCAIRSetDiagScalePolys", old_flag, &flg, NULL));
   PetscCall(PCAIRSetDiagScalePolys(pc, flg));   
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
   PetscCall(PCAIRGetCoarsestDiagScalePolys(pc, &old_flag));
   flg = old_flag;
   PetscCall(PetscOptionsBool("-pc_air_coarsest_diag_scale_polys", "Diagonally scale before computing coarse grid polynomial inverse", "PCAIRSetCoarsestDiagScalePolys", old_flag, &flg, NULL));
   PetscCall(PCAIRSetCoarsestDiagScalePolys(pc, flg));   
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
   PetscCall(PCAIRGetReuseAmount(pc, &old_int));
   input_int = old_int;
   PetscCall(PetscOptionsInt("-pc_air_reuse_amount", "Amount of data to reuse during setup with reuse_sparsity (1, 2, or 3 - 3 is store everything)", "PCAIRSetReuseAmount", old_int, &input_int, NULL));
   PetscCall(PCAIRSetReuseAmount(pc, input_int));
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
   const char *const CFSplittingTypes[] = {"PMISR_DDC", "DIAG_DOM", "PMIS", "PMIS_DIST2", "AGG", "PMIS_AGG", "CFSplittingType", "CF_", NULL};
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

      PetscCall(PCAIRGetMaxLevels(pc, &input_int));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Max number of levels=%" PetscInt_FMT " \n", input_int));
      PetscCall(PCAIRGetCoarseEqLimit(pc, &input_int));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Coarse eq limit=%" PetscInt_FMT " \n", input_int));

      PetscCall(PCAIRGetAutoTruncateStartLevel(pc, &input_int));
      PetscCall(PCAIRGetAutoTruncateTol(pc, &input_real));
      if (input_int != -1)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  Auto truncate start level=%" PetscInt_FMT ", with tolerance=%.2e \n", input_int, input_real));
      }

      PetscCall(PCAIRGetRDrop(pc, &input_real));
      PetscCall(PCAIRGetADrop(pc, &input_real_two));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  A drop tolerance=%.2e, R drop tolerance=%.2e \n", input_real_two, input_real));
      PetscCall(PCAIRGetALump(pc, &flg));
      if (flg) PetscCall(PetscViewerASCIIPrintf(viewer, "  Lumping \n"));
      PetscCall(PCAIRGetReuseSparsity(pc, &flg));
      if (flg)
      {
         PetscCall(PCAIRGetReuseAmount(pc, &input_int));
         PetscCall(PetscViewerASCIIPrintf(viewer, "  Reusing sparsity during setup (reuse amount=%" PetscInt_FMT ")\n", input_int));
      }
      PetscCall(PCAIRGetReusePolyCoeffs(pc, &flg));
      if (flg) PetscCall(PetscViewerASCIIPrintf(viewer, "  Reusing gmres polynomial coefficients during setup \n"));

      PetscCall(PCAIRGetProcessorAgglom(pc, &flg));
      PetscCall(PCAIRGetProcessorAgglomRatio(pc, &input_real));
      PetscCall(PCAIRGetProcessorAgglomFactor(pc, &input_int));
      PetscCall(PCAIRGetProcessEqLimit(pc, &input_int_two));
      if (flg) PetscCall(PetscViewerASCIIPrintf(viewer, "  Processor agglomeration with factor=%" PetscInt_FMT ", ratio=%f and eq limit=%" PetscInt_FMT " \n", input_int, input_real, input_int_two));
      PetscCall(PCAIRGetSubcomm(pc, &flg));
      if (flg) PetscCall(PetscViewerASCIIPrintf(viewer, "  Polynomial coefficients calculated on subcomm \n"));
      
      PetscCall(PCAIRGetCFSplittingType(pc, &cf_type));
      PetscCall(PCAIRGetStrongThreshold(pc, &input_real));
      PetscCall(PCAIRGetDDCIts(pc, &input_int_three));
      PetscCall(PCAIRGetDDCFraction(pc, &input_real_two));
      PetscCall(PCAIRGetMaxLubySteps(pc, &input_int_two));
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

      PetscCall(PCAIRGetFullSmoothingUpAndDown(pc, &flg));
      if (flg) 
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  Full smoothing up & down \n"));
         PetscCall(PCAIRGetInverseType(pc, &input_type));
         PetscCall(PCAIRGetPolyOrder(pc, &input_int_two));
         PetscCall(PCAIRGetInverseSparsityOrder(pc, &input_int_three));
         PetscCall(PCAIRGetDiagScalePolys(pc, &flg_diag_scale));
         PetscCall(PCAIRGetMatrixFreePolys(pc, &flg));

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
            if (flg_diag_scale)
            {
               PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
            }            
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
            PetscCall(PCAIRGetDiagScalePolys(pc, &flg_diag_scale));
            PetscCall(PCAIRGetMatrixFreePolys(pc, &flg));

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
               if (flg_diag_scale)
               {
                  PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
               }
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
            PetscCall(PCAIRGetDiagScalePolys(pc, &flg_diag_scale));
            PetscCall(PCAIRGetMatrixFreePolys(pc, &flg));

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
               if (flg_diag_scale)
               {
                  PetscCall(PetscViewerASCIIPrintf(viewer, "      with diagonal scaling \n"));
               }
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
      PetscCall(PetscViewerASCIIPrintf(viewer, "    Strong R threshold=%.2e \n", input_real));
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
      PetscCall(PCAIRGetCoarsestDiagScalePolys(pc, &flg_diag_scale));
      PetscCall(PCAIRGetCoarsestMatrixFreePolys(pc, &flg));

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
     
      PetscCall(PCAIRGetCoarsestSubcomm(pc, &flg));
      if (flg) PetscCall(PetscViewerASCIIPrintf(viewer, "   Polynomial coefficients calculated on subcomm \n"));

      PetscCall(PetscViewerASCIIPrintf(viewer, "The underlying PCMG: \n"));
      // Call the underlying pcshell view
      PetscCall(PCView(*pc_air_shell, viewer));
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~

/*MC
  PCAIR - Reduction-based algebraic multigrid using approximate ideal restriction (AIR),
  providing the AIRG, nAIR and lAIR methods for nonsymmetric/asymmetric linear systems

  Options Database Keys:
+ -pc_air_z_type            (product|lair|lair_sai) - grid-transfer operator type
. -pc_air_inverse_type      (power|arnoldi|newton|neumann|sai|isai|wjacobi|jacobi) - approximate inverse used for smoothing
. -pc_air_poly_order        poly_order - polynomial order if using a polynomial inverse type
. -pc_air_smooth_type       smooth_type - type and number of smooths, any sequence of f and c characters (for example ff, fc, fcf)
. -pc_air_cf_splitting_type (pmisr_ddc|diag_dom|pmis|pmis_dist2|agg|pmis_agg) - CF splitting to use
. -pc_air_strong_threshold  strong_threshold - strong threshold used in the CF splitting
. -pc_air_r_drop            drop_tol - drop tolerance applied to R on each level after it is built
. -pc_air_a_drop            drop_tol - drop tolerance applied to the coarse matrix on each level after it is built
- -pc_air_print_stats_timings (true|false) - print statistics about the multigrid hierarchy and timings

  Level: intermediate

  Notes:
  `PCAIR` requires configuring PETSc with `--download-pflare`.

  The combination of `-pc_air_z_type` and `-pc_air_inverse_type` selects the reduction
  multigrid: product + arnoldi gives AIRG, lair + wjacobi gives lAIR, and so on.

  Only the most common options are listed here. See <https://github.com/PFLAREProject/PFLARE>
  and its `docs/options.md` for the complete set, and the many `PCAIRSetXXX()` routines.

  `PCAIR` can use GPUs in both its setup and solve: configure PETSc with Kokkos and the
  relevant GPU backend (CUDA, HIP or SYCL), then specify the matrix/vector types as the
  PETSc Kokkos types (for example `-mat_type aijkokkos -vec_type kokkos`, or
  `-dm_mat_type aijkokkos -dm_vec_type kokkos` if using a `DM`).

  If you use `PCAIR` please cite S. Dargaville et al., "AIR multigrid with GMRES polynomials
  (AIRG) and additive preconditioners for Boltzmann transport", J. Comput. Phys. 518 (2024) 113342.

.seealso: [](ch_ksp), `PCCreate()`, `PCSetType()`, `PCType`, `PC`, `PCPFLAREINV`, `PCHYPRE`, `PCGAMG`, `PCMG`
M*/

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