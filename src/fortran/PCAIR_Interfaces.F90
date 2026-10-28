module pcair_interfaces

   use iso_c_binding
   use petscksp

#include "petsc/finclude/petscksp.h"
#include "finclude/pflare_types.h"

   implicit none

   public

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! Contains Fortran interfaces to the PCAIR options
   ! The options are stored in, and the get/set routines are implemented by,
   ! the C routines in PCAIR.c
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Get routines - interfaces to the C routines in PCAIR
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function PCAIRGetPrintStatsTimings_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetPrintStatsTimings")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool                              :: b
         PetscErrorCode  :: PCAIRGetPrintStatsTimings_mine
      end function PCAIRGetPrintStatsTimings_mine
   end interface

   interface
      function PCAIRGetMaxLevels_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetMaxLevels")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt                               :: b
         PetscErrorCode  :: PCAIRGetMaxLevels_mine
      end function PCAIRGetMaxLevels_mine
   end interface

   interface
      function PCAIRGetCoarseEqLimit_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetCoarseEqLimit")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt                               :: b
         PetscErrorCode  :: PCAIRGetCoarseEqLimit_mine
      end function PCAIRGetCoarseEqLimit_mine
   end interface

   interface
      function PCAIRGetAutoTruncateStartLevel_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetAutoTruncateStartLevel")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt                               :: b
         PetscErrorCode  :: PCAIRGetAutoTruncateStartLevel_mine
      end function PCAIRGetAutoTruncateStartLevel_mine
   end interface

   interface
      function PCAIRGetAutoTruncateTol_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetAutoTruncateTol")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscReal                              :: b
         PetscErrorCode  :: PCAIRGetAutoTruncateTol_mine
      end function PCAIRGetAutoTruncateTol_mine
   end interface

   interface
      function PCAIRGetNumLevels_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetNumLevels")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt                               :: b
         PetscErrorCode  :: PCAIRGetNumLevels_mine
      end function PCAIRGetNumLevels_mine
   end interface

   interface
      function PCAIRGetGridComplexity_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetGridComplexity")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscReal                              :: b
         PetscErrorCode  :: PCAIRGetGridComplexity_mine
      end function PCAIRGetGridComplexity_mine
   end interface

   interface
      function PCAIRGetOperatorComplexity_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetOperatorComplexity")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscReal                              :: b
         PetscErrorCode  :: PCAIRGetOperatorComplexity_mine
      end function PCAIRGetOperatorComplexity_mine
   end interface

   interface
      function PCAIRGetCycleComplexity_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetCycleComplexity")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscReal                              :: b
         PetscErrorCode  :: PCAIRGetCycleComplexity_mine
      end function PCAIRGetCycleComplexity_mine
   end interface

   interface
      function PCAIRGetStorageComplexity_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetStorageComplexity")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscReal                              :: b
         PetscErrorCode  :: PCAIRGetStorageComplexity_mine
      end function PCAIRGetStorageComplexity_mine
   end interface

   interface
      function PCAIRGetReuseStorageComplexity_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetReuseStorageComplexity")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscReal                              :: b
         PetscErrorCode  :: PCAIRGetReuseStorageComplexity_mine
      end function PCAIRGetReuseStorageComplexity_mine
   end interface

   interface
      function PCAIRGetProcessorAgglom_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetProcessorAgglom")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool                              :: b
         PetscErrorCode  :: PCAIRGetProcessorAgglom_mine
      end function PCAIRGetProcessorAgglom_mine
   end interface

   interface
      function PCAIRGetProcessorAgglomRatio_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetProcessorAgglomRatio")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscReal                              :: b
         PetscErrorCode  :: PCAIRGetProcessorAgglomRatio_mine
      end function PCAIRGetProcessorAgglomRatio_mine
   end interface

   interface
      function PCAIRGetProcessorAgglomFactor_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetProcessorAgglomFactor")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt                               :: b
         PetscErrorCode  :: PCAIRGetProcessorAgglomFactor_mine
      end function PCAIRGetProcessorAgglomFactor_mine
   end interface

   interface
      function PCAIRGetProcessEqLimit_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetProcessEqLimit")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt                               :: b
         PetscErrorCode  :: PCAIRGetProcessEqLimit_mine
      end function PCAIRGetProcessEqLimit_mine
   end interface

   interface
      function PCAIRGetSubcomm_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetSubcomm")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool                              :: b
         PetscErrorCode  :: PCAIRGetSubcomm_mine
      end function PCAIRGetSubcomm_mine
   end interface

   interface
      function PCAIRGetStrongThreshold_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetStrongThreshold")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscReal                              :: b
         PetscErrorCode  :: PCAIRGetStrongThreshold_mine
      end function PCAIRGetStrongThreshold_mine
   end interface

   interface
      function PCAIRGetDDCIts_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetDDCIts")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt                               :: b
         PetscErrorCode  :: PCAIRGetDDCIts_mine
      end function PCAIRGetDDCIts_mine
   end interface

   interface
      function PCAIRGetDDCFraction_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetDDCFraction")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscReal                              :: b
         PetscErrorCode  :: PCAIRGetDDCFraction_mine
      end function PCAIRGetDDCFraction_mine
   end interface

   interface
      function PCAIRGetCFSplittingType_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetCFSplittingType")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         integer(c_int)                         :: b
         PetscErrorCode  :: PCAIRGetCFSplittingType_mine
      end function PCAIRGetCFSplittingType_mine
   end interface

   interface
      function PCAIRGetMaxLubySteps_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetMaxLubySteps")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt                               :: b
         PetscErrorCode  :: PCAIRGetMaxLubySteps_mine
      end function PCAIRGetMaxLubySteps_mine
   end interface

   interface
      function PCAIRGetDiagScalePolys_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetDiagScalePolys")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool                              :: b
         PetscErrorCode  :: PCAIRGetDiagScalePolys_mine
      end function PCAIRGetDiagScalePolys_mine
   end interface

   interface
      function PCAIRGetMatrixFreePolys_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetMatrixFreePolys")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool                              :: b
         PetscErrorCode  :: PCAIRGetMatrixFreePolys_mine
      end function PCAIRGetMatrixFreePolys_mine
   end interface

   interface
      function PCAIRGetOnePointClassicalProlong_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetOnePointClassicalProlong")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool                              :: b
         PetscErrorCode  :: PCAIRGetOnePointClassicalProlong_mine
      end function PCAIRGetOnePointClassicalProlong_mine
   end interface

   interface
      function PCAIRGetFullSmoothingUpAndDown_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetFullSmoothingUpAndDown")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool                              :: b
         PetscErrorCode  :: PCAIRGetFullSmoothingUpAndDown_mine
      end function PCAIRGetFullSmoothingUpAndDown_mine
   end interface

   interface
      function PCAIRGetSymmetric_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetSymmetric")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool                              :: b
         PetscErrorCode  :: PCAIRGetSymmetric_mine
      end function PCAIRGetSymmetric_mine
   end interface

   interface
      function PCAIRGetConstrainW_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetConstrainW")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool                              :: b
         PetscErrorCode  :: PCAIRGetConstrainW_mine
      end function PCAIRGetConstrainW_mine
   end interface

   interface
      function PCAIRGetConstrainZ_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetConstrainZ")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool                              :: b
         PetscErrorCode  :: PCAIRGetConstrainZ_mine
      end function PCAIRGetConstrainZ_mine
   end interface

   interface
      function PCAIRGetImproveWIts_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetImproveWIts")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt                               :: b
         PetscErrorCode  :: PCAIRGetImproveWIts_mine
      end function PCAIRGetImproveWIts_mine
   end interface

   interface
      function PCAIRGetImproveZIts_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetImproveZIts")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt                               :: b
         PetscErrorCode  :: PCAIRGetImproveZIts_mine
      end function PCAIRGetImproveZIts_mine
   end interface

   interface
      function PCAIRGetStrongRThreshold_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetStrongRThreshold")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscReal                              :: b
         PetscErrorCode  :: PCAIRGetStrongRThreshold_mine
      end function PCAIRGetStrongRThreshold_mine
   end interface

   interface
      function PCAIRGetInverseType_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetInverseType")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         integer(c_int)                         :: b
         PetscErrorCode  :: PCAIRGetInverseType_mine
      end function PCAIRGetInverseType_mine
   end interface

   interface
      function PCAIRGetCInverseType_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetCInverseType")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         integer(c_int)                         :: b
         PetscErrorCode  :: PCAIRGetCInverseType_mine
      end function PCAIRGetCInverseType_mine
   end interface

   interface
      function PCAIRGetZType_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetZType")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         integer(c_int)                         :: b
         PetscErrorCode  :: PCAIRGetZType_mine
      end function PCAIRGetZType_mine
   end interface

   interface
      function PCAIRGetLairDistance_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetLairDistance")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt                               :: b
         PetscErrorCode  :: PCAIRGetLairDistance_mine
      end function PCAIRGetLairDistance_mine
   end interface

   interface
      function PCAIRGetCPolyOrder_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetCPolyOrder")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt                               :: b
         PetscErrorCode  :: PCAIRGetCPolyOrder_mine
      end function PCAIRGetCPolyOrder_mine
   end interface

   interface
      function PCAIRGetCInverseSparsityOrder_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetCInverseSparsityOrder")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt                               :: b
         PetscErrorCode  :: PCAIRGetCInverseSparsityOrder_mine
      end function PCAIRGetCInverseSparsityOrder_mine
   end interface

   interface
      function PCAIRGetPolyOrder_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetPolyOrder")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt                               :: b
         PetscErrorCode  :: PCAIRGetPolyOrder_mine
      end function PCAIRGetPolyOrder_mine
   end interface

   interface
      function PCAIRGetInverseSparsityOrder_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetInverseSparsityOrder")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt                               :: b
         PetscErrorCode  :: PCAIRGetInverseSparsityOrder_mine
      end function PCAIRGetInverseSparsityOrder_mine
   end interface

   interface
      function PCAIRGetCoarsestInverseType_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetCoarsestInverseType")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         integer(c_int)                         :: b
         PetscErrorCode  :: PCAIRGetCoarsestInverseType_mine
      end function PCAIRGetCoarsestInverseType_mine
   end interface

   interface
      function PCAIRGetCoarsestPolyOrder_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetCoarsestPolyOrder")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt                               :: b
         PetscErrorCode  :: PCAIRGetCoarsestPolyOrder_mine
      end function PCAIRGetCoarsestPolyOrder_mine
   end interface

   interface
      function PCAIRGetCoarsestInverseSparsityOrder_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetCoarsestInverseSparsityOrder")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt                               :: b
         PetscErrorCode  :: PCAIRGetCoarsestInverseSparsityOrder_mine
      end function PCAIRGetCoarsestInverseSparsityOrder_mine
   end interface

   interface
      function PCAIRGetCoarsestMatrixFreePolys_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetCoarsestMatrixFreePolys")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool                              :: b
         PetscErrorCode  :: PCAIRGetCoarsestMatrixFreePolys_mine
      end function PCAIRGetCoarsestMatrixFreePolys_mine
   end interface

   interface
      function PCAIRGetCoarsestDiagScalePolys_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetCoarsestDiagScalePolys")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool                              :: b
         PetscErrorCode  :: PCAIRGetCoarsestDiagScalePolys_mine
      end function PCAIRGetCoarsestDiagScalePolys_mine
   end interface

   interface
      function PCAIRGetCoarsestSubcomm_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetCoarsestSubcomm")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool                              :: b
         PetscErrorCode  :: PCAIRGetCoarsestSubcomm_mine
      end function PCAIRGetCoarsestSubcomm_mine
   end interface

   interface
      function PCAIRGetRDrop_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetRDrop")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscReal                              :: b
         PetscErrorCode  :: PCAIRGetRDrop_mine
      end function PCAIRGetRDrop_mine
   end interface

   interface
      function PCAIRGetADrop_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetADrop")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscReal                              :: b
         PetscErrorCode  :: PCAIRGetADrop_mine
      end function PCAIRGetADrop_mine
   end interface

   interface
      function PCAIRGetALump_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetALump")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool                              :: b
         PetscErrorCode  :: PCAIRGetALump_mine
      end function PCAIRGetALump_mine
   end interface

   interface
      function PCAIRGetReuseSparsity_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetReuseSparsity")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool                              :: b
         PetscErrorCode  :: PCAIRGetReuseSparsity_mine
      end function PCAIRGetReuseSparsity_mine
   end interface

   interface
      function PCAIRGetReusePolyCoeffs_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetReusePolyCoeffs")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool                              :: b
         PetscErrorCode  :: PCAIRGetReusePolyCoeffs_mine
      end function PCAIRGetReusePolyCoeffs_mine
   end interface

   interface
      function PCAIRGetReuseAmount_mine(pc_ptr, b) &
         bind(c, name="PCAIRGetReuseAmount")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt                               :: b
         PetscErrorCode  :: PCAIRGetReuseAmount_mine
      end function PCAIRGetReuseAmount_mine
   end interface

   interface
      function PCAIRGetSmoothType_mine(pc_ptr, output_string) &
         bind(c, name="PCAIRGetSmoothType")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         character(kind=c_char, len=1), dimension(*) :: output_string
         PetscErrorCode  :: PCAIRGetSmoothType_mine
      end function PCAIRGetSmoothType_mine
   end interface

   interface
      function PCAIRGetPolyCoeffs_mine(pc_ptr, petsc_level, which_inverse, coeffs_ptr, rows, cols) &
         bind(c, name="PCAIRGetPolyCoeffs")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value               :: pc_ptr
         PetscInt, value                           :: petsc_level
         integer(c_int), value                     :: which_inverse
         type(c_ptr), intent(out)                  :: coeffs_ptr
         PetscInt, intent(out) :: rows
         PetscInt, intent(out) :: cols
         PetscErrorCode     :: PCAIRGetPolyCoeffs_mine
      end function PCAIRGetPolyCoeffs_mine
   end interface

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Set routines - interfaces to the C routines in PCAIR
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function PCAIRSetPrintStatsTimings_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetPrintStatsTimings")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool, value                       :: b
         PetscErrorCode  :: PCAIRSetPrintStatsTimings_mine
      end function PCAIRSetPrintStatsTimings_mine
   end interface

   interface
      function PCAIRSetMaxLevels_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetMaxLevels")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt, value                        :: b
         PetscErrorCode  :: PCAIRSetMaxLevels_mine
      end function PCAIRSetMaxLevels_mine
   end interface

   interface
      function PCAIRSetCoarseEqLimit_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetCoarseEqLimit")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt, value                        :: b
         PetscErrorCode  :: PCAIRSetCoarseEqLimit_mine
      end function PCAIRSetCoarseEqLimit_mine
   end interface

   interface
      function PCAIRSetAutoTruncateStartLevel_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetAutoTruncateStartLevel")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt, value                        :: b
         PetscErrorCode  :: PCAIRSetAutoTruncateStartLevel_mine
      end function PCAIRSetAutoTruncateStartLevel_mine
   end interface

   interface
      function PCAIRSetAutoTruncateTol_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetAutoTruncateTol")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscReal, value                       :: b
         PetscErrorCode  :: PCAIRSetAutoTruncateTol_mine
      end function PCAIRSetAutoTruncateTol_mine
   end interface

   interface
      function PCAIRSetProcessorAgglom_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetProcessorAgglom")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool, value                       :: b
         PetscErrorCode  :: PCAIRSetProcessorAgglom_mine
      end function PCAIRSetProcessorAgglom_mine
   end interface

   interface
      function PCAIRSetProcessorAgglomRatio_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetProcessorAgglomRatio")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscReal, value                       :: b
         PetscErrorCode  :: PCAIRSetProcessorAgglomRatio_mine
      end function PCAIRSetProcessorAgglomRatio_mine
   end interface

   interface
      function PCAIRSetProcessorAgglomFactor_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetProcessorAgglomFactor")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt, value                        :: b
         PetscErrorCode  :: PCAIRSetProcessorAgglomFactor_mine
      end function PCAIRSetProcessorAgglomFactor_mine
   end interface

   interface
      function PCAIRSetProcessEqLimit_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetProcessEqLimit")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt, value                        :: b
         PetscErrorCode  :: PCAIRSetProcessEqLimit_mine
      end function PCAIRSetProcessEqLimit_mine
   end interface

   interface
      function PCAIRSetSubcomm_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetSubcomm")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool, value                       :: b
         PetscErrorCode  :: PCAIRSetSubcomm_mine
      end function PCAIRSetSubcomm_mine
   end interface

   interface
      function PCAIRSetStrongThreshold_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetStrongThreshold")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscReal, value                       :: b
         PetscErrorCode  :: PCAIRSetStrongThreshold_mine
      end function PCAIRSetStrongThreshold_mine
   end interface

   interface
      function PCAIRSetDDCIts_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetDDCIts")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt, value                        :: b
         PetscErrorCode  :: PCAIRSetDDCIts_mine
      end function PCAIRSetDDCIts_mine
   end interface

   interface
      function PCAIRSetDDCFraction_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetDDCFraction")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscReal, value                       :: b
         PetscErrorCode  :: PCAIRSetDDCFraction_mine
      end function PCAIRSetDDCFraction_mine
   end interface

   interface
      function PCAIRSetCFSplittingType_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetCFSplittingType")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         integer(c_int), value                  :: b
         PetscErrorCode  :: PCAIRSetCFSplittingType_mine
      end function PCAIRSetCFSplittingType_mine
   end interface

   interface
      function PCAIRSetMaxLubySteps_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetMaxLubySteps")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt, value                        :: b
         PetscErrorCode  :: PCAIRSetMaxLubySteps_mine
      end function PCAIRSetMaxLubySteps_mine
   end interface

   interface
      function PCAIRSetDiagScalePolys_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetDiagScalePolys")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool, value                       :: b
         PetscErrorCode  :: PCAIRSetDiagScalePolys_mine
      end function PCAIRSetDiagScalePolys_mine
   end interface

   interface
      function PCAIRSetMatrixFreePolys_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetMatrixFreePolys")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool, value                       :: b
         PetscErrorCode  :: PCAIRSetMatrixFreePolys_mine
      end function PCAIRSetMatrixFreePolys_mine
   end interface

   interface
      function PCAIRSetOnePointClassicalProlong_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetOnePointClassicalProlong")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool, value                       :: b
         PetscErrorCode  :: PCAIRSetOnePointClassicalProlong_mine
      end function PCAIRSetOnePointClassicalProlong_mine
   end interface

   interface
      function PCAIRSetFullSmoothingUpAndDown_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetFullSmoothingUpAndDown")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool, value                       :: b
         PetscErrorCode  :: PCAIRSetFullSmoothingUpAndDown_mine
      end function PCAIRSetFullSmoothingUpAndDown_mine
   end interface

   interface
      function PCAIRSetSymmetric_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetSymmetric")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool, value                       :: b
         PetscErrorCode  :: PCAIRSetSymmetric_mine
      end function PCAIRSetSymmetric_mine
   end interface

   interface
      function PCAIRSetConstrainW_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetConstrainW")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool, value                       :: b
         PetscErrorCode  :: PCAIRSetConstrainW_mine
      end function PCAIRSetConstrainW_mine
   end interface

   interface
      function PCAIRSetConstrainZ_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetConstrainZ")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool, value                       :: b
         PetscErrorCode  :: PCAIRSetConstrainZ_mine
      end function PCAIRSetConstrainZ_mine
   end interface

   interface
      function PCAIRSetImproveWIts_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetImproveWIts")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt, value                        :: b
         PetscErrorCode  :: PCAIRSetImproveWIts_mine
      end function PCAIRSetImproveWIts_mine
   end interface

   interface
      function PCAIRSetImproveZIts_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetImproveZIts")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt, value                        :: b
         PetscErrorCode  :: PCAIRSetImproveZIts_mine
      end function PCAIRSetImproveZIts_mine
   end interface

   interface
      function PCAIRSetStrongRThreshold_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetStrongRThreshold")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscReal, value                       :: b
         PetscErrorCode  :: PCAIRSetStrongRThreshold_mine
      end function PCAIRSetStrongRThreshold_mine
   end interface

   interface
      function PCAIRSetInverseType_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetInverseType")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         integer(c_int), value                  :: b
         PetscErrorCode  :: PCAIRSetInverseType_mine
      end function PCAIRSetInverseType_mine
   end interface

   interface
      function PCAIRSetCInverseType_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetCInverseType")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         integer(c_int), value                  :: b
         PetscErrorCode  :: PCAIRSetCInverseType_mine
      end function PCAIRSetCInverseType_mine
   end interface

   interface
      function PCAIRSetZType_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetZType")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         integer(c_int), value                  :: b
         PetscErrorCode  :: PCAIRSetZType_mine
      end function PCAIRSetZType_mine
   end interface

   interface
      function PCAIRSetLairDistance_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetLairDistance")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt, value                        :: b
         PetscErrorCode  :: PCAIRSetLairDistance_mine
      end function PCAIRSetLairDistance_mine
   end interface

   interface
      function PCAIRSetPolyOrder_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetPolyOrder")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt, value                        :: b
         PetscErrorCode  :: PCAIRSetPolyOrder_mine
      end function PCAIRSetPolyOrder_mine
   end interface

   interface
      function PCAIRSetInverseSparsityOrder_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetInverseSparsityOrder")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt, value                        :: b
         PetscErrorCode  :: PCAIRSetInverseSparsityOrder_mine
      end function PCAIRSetInverseSparsityOrder_mine
   end interface

   interface
      function PCAIRSetCPolyOrder_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetCPolyOrder")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt, value                        :: b
         PetscErrorCode  :: PCAIRSetCPolyOrder_mine
      end function PCAIRSetCPolyOrder_mine
   end interface

   interface
      function PCAIRSetCInverseSparsityOrder_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetCInverseSparsityOrder")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt, value                        :: b
         PetscErrorCode  :: PCAIRSetCInverseSparsityOrder_mine
      end function PCAIRSetCInverseSparsityOrder_mine
   end interface

   interface
      function PCAIRSetCoarsestInverseType_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetCoarsestInverseType")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         integer(c_int), value                  :: b
         PetscErrorCode  :: PCAIRSetCoarsestInverseType_mine
      end function PCAIRSetCoarsestInverseType_mine
   end interface

   interface
      function PCAIRSetCoarsestPolyOrder_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetCoarsestPolyOrder")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt, value                        :: b
         PetscErrorCode  :: PCAIRSetCoarsestPolyOrder_mine
      end function PCAIRSetCoarsestPolyOrder_mine
   end interface

   interface
      function PCAIRSetCoarsestInverseSparsityOrder_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetCoarsestInverseSparsityOrder")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt, value                        :: b
         PetscErrorCode  :: PCAIRSetCoarsestInverseSparsityOrder_mine
      end function PCAIRSetCoarsestInverseSparsityOrder_mine
   end interface

   interface
      function PCAIRSetCoarsestMatrixFreePolys_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetCoarsestMatrixFreePolys")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool, value                       :: b
         PetscErrorCode  :: PCAIRSetCoarsestMatrixFreePolys_mine
      end function PCAIRSetCoarsestMatrixFreePolys_mine
   end interface

   interface
      function PCAIRSetCoarsestDiagScalePolys_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetCoarsestDiagScalePolys")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool, value                       :: b
         PetscErrorCode  :: PCAIRSetCoarsestDiagScalePolys_mine
      end function PCAIRSetCoarsestDiagScalePolys_mine
   end interface

   interface
      function PCAIRSetCoarsestSubcomm_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetCoarsestSubcomm")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool, value                       :: b
         PetscErrorCode  :: PCAIRSetCoarsestSubcomm_mine
      end function PCAIRSetCoarsestSubcomm_mine
   end interface

   interface
      function PCAIRSetRDrop_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetRDrop")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscReal, value                       :: b
         PetscErrorCode  :: PCAIRSetRDrop_mine
      end function PCAIRSetRDrop_mine
   end interface

   interface
      function PCAIRSetADrop_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetADrop")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscReal, value                       :: b
         PetscErrorCode  :: PCAIRSetADrop_mine
      end function PCAIRSetADrop_mine
   end interface

   interface
      function PCAIRSetALump_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetALump")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool, value                       :: b
         PetscErrorCode  :: PCAIRSetALump_mine
      end function PCAIRSetALump_mine
   end interface

   interface
      function PCAIRSetReuseSparsity_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetReuseSparsity")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool, value                       :: b
         PetscErrorCode  :: PCAIRSetReuseSparsity_mine
      end function PCAIRSetReuseSparsity_mine
   end interface

   interface
      function PCAIRSetReusePolyCoeffs_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetReusePolyCoeffs")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscBool, value                       :: b
         PetscErrorCode  :: PCAIRSetReusePolyCoeffs_mine
      end function PCAIRSetReusePolyCoeffs_mine
   end interface

   interface
      function PCAIRSetReuseAmount_mine(pc_ptr, b) &
         bind(c, name="PCAIRSetReuseAmount")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         PetscInt, value                        :: b
         PetscErrorCode  :: PCAIRSetReuseAmount_mine
      end function PCAIRSetReuseAmount_mine
   end interface

   interface
      function PCAIRSetSmoothType_mine(pc_ptr, input_string) &
         bind(c, name="PCAIRSetSmoothType")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: pc_ptr
         character(kind=c_char, len=1), dimension(*) :: input_string
         PetscErrorCode  :: PCAIRSetSmoothType_mine
      end function PCAIRSetSmoothType_mine
   end interface

   interface
      function PCAIRSetPolyCoeffs_mine(pc_ptr, petsc_level, which_inverse, coeffs_ptr, rows, cols) &
         bind(c, name="PCAIRSetPolyCoeffs")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value               :: pc_ptr
         PetscInt, value                           :: petsc_level
         integer(c_int), value                     :: which_inverse
         type(c_ptr), value                        :: coeffs_ptr
         PetscInt, value    :: rows
         PetscInt, value    :: cols
         PetscErrorCode     :: PCAIRSetPolyCoeffs_mine
      end function PCAIRSetPolyCoeffs_mine
   end interface

   ! ~~~~~~~~~~~~~~~~

   contains

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Get routines - Fortran versions of the C routines in PCAIR
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetPrintStatsTimings(pc, print_stats, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: print_stats
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetPrintStatsTimings_mine(pc_ptr, print_stats)

   end subroutine PCAIRGetPrintStatsTimings

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetMaxLevels(pc, max_levels, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: max_levels
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetMaxLevels_mine(pc_ptr, max_levels)

   end subroutine PCAIRGetMaxLevels

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCoarseEqLimit(pc, coarse_eq_limit, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: coarse_eq_limit
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetCoarseEqLimit_mine(pc_ptr, coarse_eq_limit)

   end subroutine PCAIRGetCoarseEqLimit

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetAutoTruncateStartLevel(pc, start_level, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: start_level
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetAutoTruncateStartLevel_mine(pc_ptr, start_level)

   end subroutine PCAIRGetAutoTruncateStartLevel

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetAutoTruncateTol(pc, tol, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(out)        :: tol
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetAutoTruncateTol_mine(pc_ptr, tol)

   end subroutine PCAIRGetAutoTruncateTol

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetNumLevels(pc, num_levels, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: num_levels
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetNumLevels_mine(pc_ptr, num_levels)

   end subroutine PCAIRGetNumLevels

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetGridComplexity(pc, complexity, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(out)        :: complexity
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetGridComplexity_mine(pc_ptr, complexity)

   end subroutine PCAIRGetGridComplexity

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetOperatorComplexity(pc, complexity, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(out)        :: complexity
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetOperatorComplexity_mine(pc_ptr, complexity)

   end subroutine PCAIRGetOperatorComplexity

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCycleComplexity(pc, complexity, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(out)        :: complexity
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetCycleComplexity_mine(pc_ptr, complexity)

   end subroutine PCAIRGetCycleComplexity

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetStorageComplexity(pc, complexity, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(out)        :: complexity
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetStorageComplexity_mine(pc_ptr, complexity)

   end subroutine PCAIRGetStorageComplexity

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetReuseStorageComplexity(pc, complexity, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(out)        :: complexity
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetReuseStorageComplexity_mine(pc_ptr, complexity)

   end subroutine PCAIRGetReuseStorageComplexity

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetProcessorAgglom(pc, processor_agglom, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: processor_agglom
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetProcessorAgglom_mine(pc_ptr, processor_agglom)

   end subroutine PCAIRGetProcessorAgglom

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetProcessorAgglomRatio(pc, ratio, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(out)        :: ratio
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetProcessorAgglomRatio_mine(pc_ptr, ratio)

   end subroutine PCAIRGetProcessorAgglomRatio

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetProcessorAgglomFactor(pc, factor, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: factor
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetProcessorAgglomFactor_mine(pc_ptr, factor)

   end subroutine PCAIRGetProcessorAgglomFactor

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetProcessEqLimit(pc, limit, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: limit
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetProcessEqLimit_mine(pc_ptr, limit)

   end subroutine PCAIRGetProcessEqLimit

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetSubcomm(pc, subcomm, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: subcomm
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetSubcomm_mine(pc_ptr, subcomm)

   end subroutine PCAIRGetSubcomm

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetStrongThreshold(pc, thresh, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(out)        :: thresh
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetStrongThreshold_mine(pc_ptr, thresh)

   end subroutine PCAIRGetStrongThreshold

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetDDCIts(pc, its, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: its
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetDDCIts_mine(pc_ptr, its)

   end subroutine PCAIRGetDDCIts

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetDDCFraction(pc, frac, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(out)        :: frac
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetDDCFraction_mine(pc_ptr, frac)

   end subroutine PCAIRGetDDCFraction

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCFSplittingType(pc, algo, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      CFSplittingType, intent(out)  :: algo
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetCFSplittingType_mine(pc_ptr, algo)

   end subroutine PCAIRGetCFSplittingType

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetMaxLubySteps(pc, steps, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: steps
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetMaxLubySteps_mine(pc_ptr, steps)

   end subroutine PCAIRGetMaxLubySteps

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetDiagScalePolys(pc, scale, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: scale
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetDiagScalePolys_mine(pc_ptr, scale)

   end subroutine PCAIRGetDiagScalePolys

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetMatrixFreePolys(pc, mf, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: mf
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetMatrixFreePolys_mine(pc_ptr, mf)

   end subroutine PCAIRGetMatrixFreePolys

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetOnePointClassicalProlong(pc, onep, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: onep
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetOnePointClassicalProlong_mine(pc_ptr, onep)

   end subroutine PCAIRGetOnePointClassicalProlong

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetFullSmoothingUpAndDown(pc, full, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: full
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetFullSmoothingUpAndDown_mine(pc_ptr, full)

   end subroutine PCAIRGetFullSmoothingUpAndDown

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetSymmetric(pc, sym, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: sym
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetSymmetric_mine(pc_ptr, sym)

   end subroutine PCAIRGetSymmetric

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetConstrainW(pc, constrain, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: constrain
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetConstrainW_mine(pc_ptr, constrain)

   end subroutine PCAIRGetConstrainW

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetConstrainZ(pc, constrain, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: constrain
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetConstrainZ_mine(pc_ptr, constrain)

   end subroutine PCAIRGetConstrainZ

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetImproveWIts(pc, its, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: its
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetImproveWIts_mine(pc_ptr, its)

   end subroutine PCAIRGetImproveWIts

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetImproveZIts(pc, its, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: its
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetImproveZIts_mine(pc_ptr, its)

   end subroutine PCAIRGetImproveZIts

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetStrongRThreshold(pc, thresh, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(out)        :: thresh
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetStrongRThreshold_mine(pc_ptr, thresh)

   end subroutine PCAIRGetStrongRThreshold

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetInverseType(pc, inv_type, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PCPFLAREINVType, intent(out)         :: inv_type
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetInverseType_mine(pc_ptr, inv_type)

   end subroutine PCAIRGetInverseType

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCInverseType(pc, inv_type, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PCPFLAREINVType, intent(out)         :: inv_type
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetCInverseType_mine(pc_ptr, inv_type)

   end subroutine PCAIRGetCInverseType

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetZType(pc, z_type, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PCAIRZType, intent(out)         :: z_type
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetZType_mine(pc_ptr, z_type)

   end subroutine PCAIRGetZType

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetLairDistance(pc, distance, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: distance
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetLairDistance_mine(pc_ptr, distance)

   end subroutine PCAIRGetLairDistance

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCPolyOrder(pc, order, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: order
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetCPolyOrder_mine(pc_ptr, order)

   end subroutine PCAIRGetCPolyOrder

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCInverseSparsityOrder(pc, order, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: order
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetCInverseSparsityOrder_mine(pc_ptr, order)

   end subroutine PCAIRGetCInverseSparsityOrder

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetPolyOrder(pc, order, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: order
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetPolyOrder_mine(pc_ptr, order)

   end subroutine PCAIRGetPolyOrder

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetInverseSparsityOrder(pc, order, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: order
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetInverseSparsityOrder_mine(pc_ptr, order)

   end subroutine PCAIRGetInverseSparsityOrder

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCoarsestInverseType(pc, inv_type, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PCPFLAREINVType, intent(out)         :: inv_type
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetCoarsestInverseType_mine(pc_ptr, inv_type)

   end subroutine PCAIRGetCoarsestInverseType

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCoarsestPolyOrder(pc, order, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: order
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetCoarsestPolyOrder_mine(pc_ptr, order)

   end subroutine PCAIRGetCoarsestPolyOrder

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCoarsestInverseSparsityOrder(pc, order, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: order
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetCoarsestInverseSparsityOrder_mine(pc_ptr, order)

   end subroutine PCAIRGetCoarsestInverseSparsityOrder

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCoarsestMatrixFreePolys(pc, mf, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: mf
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetCoarsestMatrixFreePolys_mine(pc_ptr, mf)

   end subroutine PCAIRGetCoarsestMatrixFreePolys

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCoarsestDiagScalePolys(pc, scale, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: scale
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetCoarsestDiagScalePolys_mine(pc_ptr, scale)

   end subroutine PCAIRGetCoarsestDiagScalePolys

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCoarsestSubcomm(pc, subcomm, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: subcomm
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetCoarsestSubcomm_mine(pc_ptr, subcomm)

   end subroutine PCAIRGetCoarsestSubcomm

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetRDrop(pc, rdrop, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(out)        :: rdrop
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetRDrop_mine(pc_ptr, rdrop)

   end subroutine PCAIRGetRDrop

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetADrop(pc, adrop, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(out)        :: adrop
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetADrop_mine(pc_ptr, adrop)

   end subroutine PCAIRGetADrop

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetALump(pc, lump, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: lump
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetALump_mine(pc_ptr, lump)

   end subroutine PCAIRGetALump

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetReuseSparsity(pc, reuse, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: reuse
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetReuseSparsity_mine(pc_ptr, reuse)

   end subroutine PCAIRGetReuseSparsity

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetReusePolyCoeffs(pc, reuse, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: reuse
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetReusePolyCoeffs_mine(pc_ptr, reuse)

   end subroutine PCAIRGetReusePolyCoeffs

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetReuseAmount(pc, amount, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: amount
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetReuseAmount_mine(pc_ptr, amount)

   end subroutine PCAIRGetReuseAmount

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetSmoothType(pc, output_string, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)         :: pc
      character(len=*), intent(out)    :: output_string
      PetscErrorCode, intent(out)      :: ierr

      integer(c_long_long) :: pc_ptr
      ! The smoothing pattern is at most 10 characters, plus the null terminator
      character(kind=c_char, len=1), dimension(11) :: c_string
      integer :: i
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRGetSmoothType_mine(pc_ptr, c_string)

      ! Copy the null-terminated C string to output, trimming or padding as needed
      output_string = ''
      do i = 1, size(c_string)
         if (c_string(i) == c_null_char) exit
         if (i > len(output_string)) exit
         output_string(i:i) = c_string(i)
      end do

   end subroutine PCAIRGetSmoothType

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetPolyCoeffs(pc, petsc_level, which_inverse, coeffs, ierr)

      ! This routine returns a copy of the polynomial coefficients stored in the PC.
      ! coeffs is allocated/reallocated as needed; it is safe to keep and use after
      ! the next PCSetUp or PCReset call (unlike the C interface, which returns a
      ! raw pointer into internal storage that is only valid until the next setup).

      ! ~~~~~~~~
      type(tPC), intent(inout)                           :: pc
      PetscInt, intent(in)                               :: petsc_level
      integer, intent(in)                                :: which_inverse
      PetscReal, dimension(:,:), pointer, intent(inout)       :: coeffs
      PetscErrorCode, intent(out)                        :: ierr

      integer(c_long_long) :: pc_ptr
      type(c_ptr) :: coeffs_c_ptr
      PetscInt :: rows, cols
      PetscReal, pointer :: coeffs_internal(:,:)
      ! ~~~~~~~~

      ! The C routine errors if there are no coefficients stored for this
      ! petsc_level and which_inverse
      pc_ptr = pc%v
      ierr = PCAIRGetPolyCoeffs_mine(pc_ptr, petsc_level, int(which_inverse, c_int), coeffs_c_ptr, rows, cols)
      if (ierr /= 0) return

      ! Check sizes
      if (associated(coeffs)) then
         if (.NOT. (size(coeffs,1) == int(rows) .AND. &
             size(coeffs,2) == int(cols))) then

            deallocate(coeffs)
            coeffs => null()
         end if
      end if

      if (.NOT. associated(coeffs)) then
         allocate(coeffs(rows, cols))
      end if

      ! Create a temporary Fortran view of the C-owned buffer and copy
      call c_f_pointer(coeffs_c_ptr, coeffs_internal, [rows, cols])
      coeffs = coeffs_internal

   end subroutine PCAIRGetPolyCoeffs

! -------------------------------------------------------------------------------------------------------------------------------

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Set routines - Fortran versions of the C routines in PCAIR
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetPrintStatsTimings(pc, print_stats, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: print_stats
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetPrintStatsTimings_mine(pc_ptr, print_stats)

   end subroutine PCAIRSetPrintStatsTimings

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetMaxLevels(pc, max_levels, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: max_levels
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetMaxLevels_mine(pc_ptr, max_levels)

   end subroutine PCAIRSetMaxLevels

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCoarseEqLimit(pc, coarse_eq_limit, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: coarse_eq_limit
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetCoarseEqLimit_mine(pc_ptr, coarse_eq_limit)

   end subroutine PCAIRSetCoarseEqLimit

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetAutoTruncateStartLevel(pc, start_level, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: start_level
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetAutoTruncateStartLevel_mine(pc_ptr, start_level)

   end subroutine PCAIRSetAutoTruncateStartLevel

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetAutoTruncateTol(pc, tol, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(in)         :: tol
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetAutoTruncateTol_mine(pc_ptr, tol)

   end subroutine PCAIRSetAutoTruncateTol

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetProcessorAgglom(pc, processor_agglom, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: processor_agglom
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetProcessorAgglom_mine(pc_ptr, processor_agglom)

   end subroutine PCAIRSetProcessorAgglom

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetProcessorAgglomRatio(pc, ratio, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(in)         :: ratio
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetProcessorAgglomRatio_mine(pc_ptr, ratio)

   end subroutine PCAIRSetProcessorAgglomRatio

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetProcessorAgglomFactor(pc, factor, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: factor
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetProcessorAgglomFactor_mine(pc_ptr, factor)

   end subroutine PCAIRSetProcessorAgglomFactor

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetProcessEqLimit(pc, limit, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: limit
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetProcessEqLimit_mine(pc_ptr, limit)

   end subroutine PCAIRSetProcessEqLimit

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetSubcomm(pc, subcomm, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: subcomm
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetSubcomm_mine(pc_ptr, subcomm)

   end subroutine PCAIRSetSubcomm

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetStrongThreshold(pc, thresh, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(in)         :: thresh
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetStrongThreshold_mine(pc_ptr, thresh)

   end subroutine PCAIRSetStrongThreshold

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetDDCIts(pc, its, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: its
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetDDCIts_mine(pc_ptr, its)

   end subroutine PCAIRSetDDCIts

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetDDCFraction(pc, frac, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(in)         :: frac
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetDDCFraction_mine(pc_ptr, frac)

   end subroutine PCAIRSetDDCFraction

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCFSplittingType(pc, algo, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      CFSplittingType, intent(in)   :: algo
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetCFSplittingType_mine(pc_ptr, algo)

   end subroutine PCAIRSetCFSplittingType

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetMaxLubySteps(pc, steps, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: steps
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetMaxLubySteps_mine(pc_ptr, steps)

   end subroutine PCAIRSetMaxLubySteps

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetDiagScalePolys(pc, scale, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: scale
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetDiagScalePolys_mine(pc_ptr, scale)

   end subroutine PCAIRSetDiagScalePolys

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetMatrixFreePolys(pc, mf, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: mf
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetMatrixFreePolys_mine(pc_ptr, mf)

   end subroutine PCAIRSetMatrixFreePolys

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetOnePointClassicalProlong(pc, onep, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: onep
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetOnePointClassicalProlong_mine(pc_ptr, onep)

   end subroutine PCAIRSetOnePointClassicalProlong

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetFullSmoothingUpAndDown(pc, full, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: full
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetFullSmoothingUpAndDown_mine(pc_ptr, full)

   end subroutine PCAIRSetFullSmoothingUpAndDown

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetSymmetric(pc, sym, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: sym
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetSymmetric_mine(pc_ptr, sym)

   end subroutine PCAIRSetSymmetric

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetConstrainW(pc, constrain, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: constrain
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetConstrainW_mine(pc_ptr, constrain)

   end subroutine PCAIRSetConstrainW

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetConstrainZ(pc, constrain, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: constrain
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetConstrainZ_mine(pc_ptr, constrain)

   end subroutine PCAIRSetConstrainZ

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetImproveWIts(pc, its, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: its
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetImproveWIts_mine(pc_ptr, its)

   end subroutine PCAIRSetImproveWIts

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetImproveZIts(pc, its, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: its
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetImproveZIts_mine(pc_ptr, its)

   end subroutine PCAIRSetImproveZIts

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetStrongRThreshold(pc, thresh, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(in)         :: thresh
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetStrongRThreshold_mine(pc_ptr, thresh)

   end subroutine PCAIRSetStrongRThreshold

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetInverseType(pc, inv_type, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PCPFLAREINVType, intent(in)          :: inv_type
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetInverseType_mine(pc_ptr, inv_type)

   end subroutine PCAIRSetInverseType

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCInverseType(pc, inv_type, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PCPFLAREINVType, intent(in)          :: inv_type
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetCInverseType_mine(pc_ptr, inv_type)

   end subroutine PCAIRSetCInverseType

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetZType(pc, z_type, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PCAIRZType, intent(in)          :: z_type
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetZType_mine(pc_ptr, z_type)

   end subroutine PCAIRSetZType

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetLairDistance(pc, distance, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: distance
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetLairDistance_mine(pc_ptr, distance)

   end subroutine PCAIRSetLairDistance

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetPolyOrder(pc, order, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: order
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetPolyOrder_mine(pc_ptr, order)

   end subroutine PCAIRSetPolyOrder

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetInverseSparsityOrder(pc, order, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: order
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetInverseSparsityOrder_mine(pc_ptr, order)

   end subroutine PCAIRSetInverseSparsityOrder

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCPolyOrder(pc, order, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: order
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetCPolyOrder_mine(pc_ptr, order)

   end subroutine PCAIRSetCPolyOrder

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCInverseSparsityOrder(pc, order, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: order
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetCInverseSparsityOrder_mine(pc_ptr, order)

   end subroutine PCAIRSetCInverseSparsityOrder

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCoarsestInverseType(pc, inv_type, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PCPFLAREINVType, intent(in)          :: inv_type
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetCoarsestInverseType_mine(pc_ptr, inv_type)

   end subroutine PCAIRSetCoarsestInverseType

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCoarsestPolyOrder(pc, order, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: order
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetCoarsestPolyOrder_mine(pc_ptr, order)

   end subroutine PCAIRSetCoarsestPolyOrder

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCoarsestInverseSparsityOrder(pc, order, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: order
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetCoarsestInverseSparsityOrder_mine(pc_ptr, order)

   end subroutine PCAIRSetCoarsestInverseSparsityOrder

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCoarsestMatrixFreePolys(pc, mf, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: mf
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetCoarsestMatrixFreePolys_mine(pc_ptr, mf)

   end subroutine PCAIRSetCoarsestMatrixFreePolys

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCoarsestDiagScalePolys(pc, scale, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: scale
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetCoarsestDiagScalePolys_mine(pc_ptr, scale)

   end subroutine PCAIRSetCoarsestDiagScalePolys

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCoarsestSubcomm(pc, subcomm, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: subcomm
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetCoarsestSubcomm_mine(pc_ptr, subcomm)

   end subroutine PCAIRSetCoarsestSubcomm

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetRDrop(pc, rdrop, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(in)         :: rdrop
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetRDrop_mine(pc_ptr, rdrop)

   end subroutine PCAIRSetRDrop

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetADrop(pc, adrop, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(in)         :: adrop
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetADrop_mine(pc_ptr, adrop)

   end subroutine PCAIRSetADrop

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetALump(pc, lump, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: lump
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetALump_mine(pc_ptr, lump)

   end subroutine PCAIRSetALump

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetReuseSparsity(pc, reuse, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: reuse
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetReuseSparsity_mine(pc_ptr, reuse)

   end subroutine PCAIRSetReuseSparsity

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetReusePolyCoeffs(pc, reuse, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: reuse
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetReusePolyCoeffs_mine(pc_ptr, reuse)

   end subroutine PCAIRSetReusePolyCoeffs

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetReuseAmount(pc, amount, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: amount
      PetscErrorCode, intent(out)   :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v
      ierr = PCAIRSetReuseAmount_mine(pc_ptr, amount)

   end subroutine PCAIRSetReuseAmount

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetSmoothType(pc, input_string, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)       :: pc
      character(len = *), intent(in) :: input_string
      PetscErrorCode, intent(out)    :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~

      ! Trailing blanks are ignored
      pc_ptr = pc%v
      ierr = PCAIRSetSmoothType_mine(pc_ptr, trim(input_string) // c_null_char)

   end subroutine PCAIRSetSmoothType

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetPolyCoeffs(pc, petsc_level, which_inverse, coeffs, ierr)

      ! This routine copies coeffs into the PC's internal storage.
      ! The caller's array is not referenced after this call and can be
      ! deallocated or modified freely.
      ! Does not trigger a rebuild; after setting these coefficients call
      ! PCAIRSetReusePolyCoeffs(pc, PETSC_TRUE) and then KSPSolve to apply them.
      type(tPC), intent(inout)                           :: pc
      PetscInt, intent(in)                               :: petsc_level
      integer, intent(in)                                :: which_inverse
      PetscReal, dimension(:,:), pointer, intent(in)          :: coeffs
      PetscErrorCode, intent(out)                        :: ierr

      integer(c_long_long) :: pc_ptr
      PetscInt :: rows, cols
      integer :: errorcode
      ! ~~~~~~~~

      if (.NOT. associated(coeffs)) then
         print *, "PCAIRSetPolyCoeffs: coeffs is not associated"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! The C routine errors if there are no coefficients stored for this
      ! petsc_level and which_inverse, or if the sizes are wrong
      pc_ptr = pc%v
      rows   = int(size(coeffs, 1), PETSC_INT_KIND)
      cols   = int(size(coeffs, 2), PETSC_INT_KIND)
      ierr   = PCAIRSetPolyCoeffs_mine(pc_ptr, petsc_level, int(which_inverse, c_int), c_loc(coeffs(1,1)), rows, cols)

   end subroutine PCAIRSetPolyCoeffs

! -------------------------------------------------------------------------------------------------------------------------------

end module pcair_interfaces
