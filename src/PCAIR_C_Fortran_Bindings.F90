module pcair_c_fortran_bindings

   use iso_c_binding
   use petscksp
   use pcair_interfaces

#include "petsc/finclude/petscksp.h"
#include "finclude/pflare_types.h"

   implicit none

   public

   contains

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! Contains C bindings to the PCAIR options 
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------      
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetPrintStatsTimings_c(pc_ptr, print_stats) bind(C, name='PCAIRGetPrintStatsTimings_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, intent(out)              :: print_stats

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetPrintStatsTimings(pc, print_stats, ierr)

   end subroutine PCAIRGetPrintStatsTimings_c   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetMaxLevels_c(pc_ptr, max_levels) bind(C, name='PCAIRGetMaxLevels_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, intent(out)               :: max_levels

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetMaxLevels(pc, max_levels, ierr)

   end subroutine PCAIRGetMaxLevels_c

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCoarseEqLimit_c(pc_ptr, coarse_eq_limit) bind(C, name='PCAIRGetCoarseEqLimit_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, intent(out)               :: coarse_eq_limit

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetCoarseEqLimit(pc, coarse_eq_limit, ierr)

   end subroutine PCAIRGetCoarseEqLimit_c   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetAutoTruncateStartLevel_c(pc_ptr, start_level) bind(C, name='PCAIRGetAutoTruncateStartLevel_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, intent(out)               :: start_level

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetAutoTruncateStartLevel(pc, start_level, ierr)

   end subroutine PCAIRGetAutoTruncateStartLevel_c   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetAutoTruncateTol_c(pc_ptr, tol) bind(C, name='PCAIRGetAutoTruncateTol_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscReal, intent(out)        :: tol

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetAutoTruncateTol(pc, tol, ierr)

   end subroutine PCAIRGetAutoTruncateTol_c   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetNumLevels_c(pc_ptr, num_levels) bind(C, name='PCAIRGetNumLevels_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, intent(out)               :: num_levels

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetNumLevels(pc, num_levels, ierr)

   end subroutine PCAIRGetNumLevels_c   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetProcessorAgglom_c(pc_ptr, processor_agglom) bind(C, name='PCAIRGetProcessorAgglom_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, intent(out)        :: processor_agglom

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetProcessorAgglom(pc, processor_agglom, ierr)

   end subroutine PCAIRGetProcessorAgglom_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetProcessorAgglomRatio_c(pc_ptr, ratio) bind(C, name='PCAIRGetProcessorAgglomRatio_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscReal, intent(out)        :: ratio

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetProcessorAgglomRatio(pc, ratio, ierr)

   end subroutine PCAIRGetProcessorAgglomRatio_c

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetProcessorAgglomFactor_c(pc_ptr, factor) bind(C, name='PCAIRGetProcessorAgglomFactor_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, intent(out)        :: factor

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetProcessorAgglomFactor(pc, factor, ierr)

   end subroutine PCAIRGetProcessorAgglomFactor_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetProcessEqLimit_c(pc_ptr, limit) bind(C, name='PCAIRGetProcessEqLimit_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, intent(out)        :: limit

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetProcessEqLimit(pc, limit, ierr)

   end subroutine PCAIRGetProcessEqLimit_c   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetSubcomm_c(pc_ptr, subcomm) bind(C, name='PCAIRGetSubcomm_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, intent(out)        :: subcomm
      
      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetSubcomm(pc, subcomm, ierr)

   end subroutine PCAIRGetSubcomm_c   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetStrongThreshold_c(pc_ptr, thresh) bind(C, name='PCAIRGetStrongThreshold_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscReal, intent(out)        :: thresh

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetStrongThreshold(pc, thresh, ierr)

   end subroutine PCAIRGetStrongThreshold_c   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetDDCIts_c(pc_ptr, its) bind(C, name='PCAIRGetDDCIts_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, intent(out)               :: its

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetDDCIts(pc, its, ierr)

   end subroutine PCAIRGetDDCIts_c

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetDDCFraction_c(pc_ptr, frac) bind(C, name='PCAIRGetDDCFraction_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscReal, intent(out)        :: frac

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetDDCFraction(pc, frac, ierr)

   end subroutine PCAIRGetDDCFraction_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCFSplittingType_c(pc_ptr, algo) bind(C, name='PCAIRGetCFSplittingType_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      CFSplittingType, intent(out)         :: algo

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetCFSplittingType(pc, algo, ierr)

   end subroutine PCAIRGetCFSplittingType_c

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetMaxLubySteps_c(pc_ptr, steps) bind(C, name='PCAIRGetMaxLubySteps_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, intent(out)         :: steps

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetMaxLubySteps(pc, steps, ierr)

   end subroutine PCAIRGetMaxLubySteps_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetMatrixFreePolys_c(pc_ptr, mf) bind(C, name='PCAIRGetMatrixFreePolys_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, intent(out)        :: mf

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetMatrixFreePolys(pc, mf, ierr)

   end subroutine PCAIRGetMatrixFreePolys_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetOnePointClassicalProlong_c(pc_ptr, onep) bind(C, name='PCAIRGetOnePointClassicalProlong_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, intent(out)        :: onep

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetOnePointClassicalProlong(pc, onep, ierr)

   end subroutine PCAIRGetOnePointClassicalProlong_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetFullSmoothingUpAndDown_c(pc_ptr, full) bind(C, name='PCAIRGetFullSmoothingUpAndDown_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, intent(out)        :: full

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetFullSmoothingUpAndDown(pc, full, ierr)

   end subroutine PCAIRGetFullSmoothingUpAndDown_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetSymmetric_c(pc_ptr, sym) bind(C, name='PCAIRGetSymmetric_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, intent(out)        :: sym

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetSymmetric(pc, sym, ierr)

   end subroutine PCAIRGetSymmetric_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetConstrainW_c(pc_ptr, constrain) bind(C, name='PCAIRGetConstrainW_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, intent(out)        :: constrain

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetConstrainW(pc, constrain, ierr)

   end subroutine PCAIRGetConstrainW_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetConstrainZ_c(pc_ptr, constrain) bind(C, name='PCAIRGetConstrainZ_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, intent(out)        :: constrain

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetConstrainZ(pc, constrain, ierr)

   end subroutine PCAIRGetConstrainZ_c

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetImproveWIts_c(pc_ptr, its) bind(C, name='PCAIRGetImproveWIts_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, intent(out)         :: its

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetImproveWIts(pc, its, ierr)

   end subroutine PCAIRGetImproveWIts_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetImproveZIts_c(pc_ptr, its) bind(C, name='PCAIRGetImproveZIts_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, intent(out)         :: its

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetImproveZIts(pc, its, ierr)

   end subroutine PCAIRGetImproveZIts_c    
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetStrongRThreshold_c(pc_ptr, thresh) bind(C, name='PCAIRGetStrongRThreshold_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscReal, intent(out)        :: thresh

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetStrongRThreshold(pc, thresh, ierr)

   end subroutine PCAIRGetStrongRThreshold_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetInverseType_c(pc_ptr, inv_type) bind(C, name='PCAIRGetInverseType_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PCPFLAREINVType, intent(out)         :: inv_type

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetInverseType(pc, inv_type, ierr)

   end subroutine PCAIRGetInverseType_c

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCInverseType_c(pc_ptr, inv_type) bind(C, name='PCAIRGetCInverseType_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PCPFLAREINVType, intent(out)         :: inv_type

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetCInverseType(pc, inv_type, ierr)

   end subroutine PCAIRGetCInverseType_c   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetZType_c(pc_ptr, z_type) bind(C, name='PCAIRGetZType_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PCAIRZType, intent(out)         :: z_type

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetZType(pc, z_type, ierr)

   end subroutine PCAIRGetZType_c

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetLairDistance_c(pc_ptr, distance) bind(C, name='PCAIRGetLairDistance_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, intent(out)         :: distance

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetLairDistance(pc, distance, ierr)

   end subroutine PCAIRGetLairDistance_c   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetPolyOrder_c(pc_ptr, order) bind(C, name='PCAIRGetPolyOrder_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, intent(out)         :: order

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetPolyOrder(pc, order, ierr)

   end subroutine PCAIRGetPolyOrder_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetInverseSparsityOrder_c(pc_ptr, order) bind(C, name='PCAIRGetInverseSparsityOrder_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, intent(out)         :: order

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetInverseSparsityOrder(pc, order, ierr)

   end subroutine PCAIRGetInverseSparsityOrder_c

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCPolyOrder_c(pc_ptr, order) bind(C, name='PCAIRGetCPolyOrder_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, intent(out)         :: order

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetCPolyOrder(pc, order, ierr)

   end subroutine PCAIRGetCPolyOrder_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCInverseSparsityOrder_c(pc_ptr, order) bind(C, name='PCAIRGetCInverseSparsityOrder_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, intent(out)         :: order

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetCInverseSparsityOrder(pc, order, ierr)

   end subroutine PCAIRGetCInverseSparsityOrder_c   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCoarsestInverseType_c(pc_ptr, inv_type) bind(C, name='PCAIRGetCoarsestInverseType_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PCPFLAREINVType, intent(out)         :: inv_type

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetCoarsestInverseType(pc, inv_type, ierr)

   end subroutine PCAIRGetCoarsestInverseType_c   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCoarsestPolyOrder_c(pc_ptr, order) bind(C, name='PCAIRGetCoarsestPolyOrder_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, intent(out)         :: order

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetCoarsestPolyOrder(pc, order, ierr)

   end subroutine PCAIRGetCoarsestPolyOrder_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCoarsestInverseSparsityOrder_c(pc_ptr, order) bind(C, name='PCAIRGetCoarsestInverseSparsityOrder_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, intent(out)         :: order

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetCoarsestInverseSparsityOrder(pc, order, ierr)

   end subroutine PCAIRGetCoarsestInverseSparsityOrder_c 
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCoarsestMatrixFreePolys_c(pc_ptr, mf) bind(C, name='PCAIRGetCoarsestMatrixFreePolys_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, intent(out)        :: mf

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetCoarsestMatrixFreePolys(pc, mf, ierr)

   end subroutine PCAIRGetCoarsestMatrixFreePolys_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCoarsestSubcomm_c(pc_ptr, subcomm) bind(C, name='PCAIRGetCoarsestSubcomm_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, intent(out)        :: subcomm

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetCoarsestSubcomm(pc, subcomm, ierr)

   end subroutine PCAIRGetCoarsestSubcomm_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetRDrop_c(pc_ptr, rdrop) bind(C, name='PCAIRGetRDrop_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscReal, intent(out)        :: rdrop

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetRDrop(pc, rdrop, ierr)

   end subroutine PCAIRGetRDrop_c  
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetADrop_c(pc_ptr, adrop) bind(C, name='PCAIRGetADrop_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscReal, intent(out)        :: adrop

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetADrop(pc, adrop, ierr)

   end subroutine PCAIRGetADrop_c 
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetALump_c(pc_ptr, lump) bind(C, name='PCAIRGetALump_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, intent(out)        :: lump

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetALump(pc, lump, ierr)

   end subroutine PCAIRGetALump_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetReuseSparsity_c(pc_ptr, reuse) bind(C, name='PCAIRGetReuseSparsity_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, intent(out)        :: reuse

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetReuseSparsity(pc, reuse, ierr)

   end subroutine PCAIRGetReuseSparsity_c

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetReusePolyCoeffs_c(pc_ptr, reuse) bind(C, name='PCAIRGetReusePolyCoeffs_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, intent(out)        :: reuse

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRGetReusePolyCoeffs(pc, reuse, ierr)

   end subroutine PCAIRGetReusePolyCoeffs_c    
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetPolyCoeffs_c(pc_ptr, petsc_level, which_inverse, &   
      coeffs_ptr, row_size, col_size) bind(C, name='PCAIRGetPolyCoeffs_c')

      ! The C interface to this differs by sending out a pointer, rather than a copy of the coefficients
      ! in an allocatable array - hence you will have to take a copy of the values externally if you want 
      ! to save them

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, value, intent(in)         :: petsc_level
      integer(c_int), value, intent(in)   :: which_inverse 
      PetscInt, intent(out)               :: row_size, col_size
      type(c_ptr), intent(out)            :: coeffs_ptr

      type(tPC)                              :: pc
      PetscErrorCode                         :: ierr
      integer                                :: our_level, errorcode
      PetscInt                               :: num_levels
      type(tPC)                              :: pc_shell
      type(pc_air_multigrid_data), pointer   :: pc_air_data      

      ! ~~~~~~~~

      pc%v = pc_ptr

      ! Get the underlying PCShell
      call PCAIRGetPCShell(pc, pc_shell)

      ! Get the PC shell context
      call PCShellGetContext(pc_shell, pc_air_data, ierr)

      ! Get the number of levels in our mg
      call PCAIRGetNumLevels(pc, num_levels, ierr) 

      ! We order our levels from 1 to num_levels
      our_level = int(num_levels - petsc_level)
      
      ! Inverse Aff
      if (which_inverse == COEFFS_INV_AFF) then

         row_size = size(pc_air_data%air_data%inv_A_ff_poly_data(our_level)%coefficients, 1)
         col_size = size(pc_air_data%air_data%inv_A_ff_poly_data(our_level)%coefficients, 2)

         coeffs_ptr = c_loc(pc_air_data%air_data%inv_A_ff_poly_data(our_level)%coefficients)

      ! Inverse dropped Aff
      else if (which_inverse == COEFFS_INV_AFF_DROPPED) then

         row_size = size(pc_air_data%air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients, 1)
         col_size = size(pc_air_data%air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients, 2)       

         coeffs_ptr = c_loc(pc_air_data%air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients)

      ! Inverse Acc
      else if (which_inverse == COEFFS_INV_ACC) then

         row_size = size(pc_air_data%air_data%inv_A_cc_poly_data(our_level)%coefficients, 1)
         col_size = size(pc_air_data%air_data%inv_A_cc_poly_data(our_level)%coefficients, 2)            

         coeffs_ptr = c_loc(pc_air_data%air_data%inv_A_cc_poly_data(our_level)%coefficients)

      ! Coarsest grid matrix
      else if (which_inverse == COEFFS_INV_COARSE) then

         row_size = size(pc_air_data%air_data%inv_coarsest_poly_data%coefficients, 1)
         col_size = size(pc_air_data%air_data%inv_coarsest_poly_data%coefficients, 2)         

         coeffs_ptr = c_loc(pc_air_data%air_data%inv_coarsest_poly_data%coefficients)

      else
         print *, "Unknown which_inverse in PCAIRGetPolyCoeffs_c"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if      


   end subroutine PCAIRGetPolyCoeffs_c      
   
! -------------------------------------------------------------------------------------------------------------------------------   

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Set routines - Fortran versions of the C routines in PCAIR
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~    

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetPrintStatsTimings_c(pc_ptr, print_stats) bind(C, name='PCAIRSetPrintStatsTimings_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, value, intent(in)         :: print_stats

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetPrintStatsTimings(pc, print_stats, ierr)

   end subroutine PCAIRSetPrintStatsTimings_c   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetMaxLevels_c(pc_ptr, max_levels) bind(C, name='PCAIRSetMaxLevels_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, value, intent(in)          :: max_levels

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetMaxLevels(pc, max_levels, ierr)

   end subroutine PCAIRSetMaxLevels_c

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCoarseEqLimit_c(pc_ptr, coarse_eq_limit) bind(C, name='PCAIRSetCoarseEqLimit_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, value, intent(in)          :: coarse_eq_limit

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetCoarseEqLimit(pc, coarse_eq_limit, ierr)

   end subroutine PCAIRSetCoarseEqLimit_c   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetAutoTruncateStartLevel_c(pc_ptr, start_level) bind(C, name='PCAIRSetAutoTruncateStartLevel_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, value, intent(in)          :: start_level

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetAutoTruncateStartLevel(pc, start_level, ierr)

   end subroutine PCAIRSetAutoTruncateStartLevel_c   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetAutoTruncateTol_c(pc_ptr, tol) bind(C, name='PCAIRSetAutoTruncateTol_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscReal, value, intent(in)         :: tol

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetAutoTruncateTol(pc, tol, ierr)

   end subroutine PCAIRSetAutoTruncateTol_c   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetProcessorAgglom_c(pc_ptr, processor_agglom) bind(C, name='PCAIRSetProcessorAgglom_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, value, intent(in)         :: processor_agglom

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetProcessorAgglom(pc, processor_agglom, ierr)

   end subroutine PCAIRSetProcessorAgglom_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetProcessorAgglomRatio_c(pc_ptr, ratio) bind(C, name='PCAIRSetProcessorAgglomRatio_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscReal, value, intent(in)         :: ratio

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetProcessorAgglomRatio(pc, ratio, ierr)

   end subroutine PCAIRSetProcessorAgglomRatio_c

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetProcessorAgglomFactor_c(pc_ptr, factor) bind(C, name='PCAIRSetProcessorAgglomFactor_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, value, intent(in)         :: factor

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetProcessorAgglomFactor(pc, factor, ierr)

   end subroutine PCAIRSetProcessorAgglomFactor_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetProcessEqLimit_c(pc_ptr, limit) bind(C, name='PCAIRSetProcessEqLimit_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, value, intent(in)         :: limit

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetProcessEqLimit(pc, limit, ierr)

   end subroutine PCAIRSetProcessEqLimit_c     
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetSubcomm_c(pc_ptr, subcomm) bind(C, name='PCAIRSetSubcomm_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, value, intent(in)         :: subcomm

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetSubcomm(pc, subcomm, ierr)

   end subroutine PCAIRSetSubcomm_c   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetStrongThreshold_c(pc_ptr, thresh) bind(C, name='PCAIRSetStrongThreshold_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscReal, value, intent(in)         :: thresh

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetStrongThreshold(pc, thresh, ierr)

   end subroutine PCAIRSetStrongThreshold_c  
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetDDCIts_c(pc_ptr, its) bind(C, name='PCAIRSetDDCIts_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, value, intent(in)         :: its

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetDDCIts(pc, its, ierr)

   end subroutine PCAIRSetDDCIts_c

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetDDCFraction_c(pc_ptr, frac) bind(C, name='PCAIRSetDDCFraction_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscReal, value, intent(in)         :: frac

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetDDCFraction(pc, frac, ierr)

   end subroutine PCAIRSetDDCFraction_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCFSplittingType_c(pc_ptr, algo) bind(C, name='PCAIRSetCFSplittingType_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      CFSplittingType, value, intent(in)          :: algo

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetCFSplittingType(pc, algo, ierr)

   end subroutine PCAIRSetCFSplittingType_c

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetMaxLubySteps_c(pc_ptr, steps) bind(C, name='PCAIRSetMaxLubySteps_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, value, intent(in)          :: steps

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetMaxLubySteps(pc, steps, ierr)

   end subroutine PCAIRSetMaxLubySteps_c   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetSmoothType_c(pc_ptr, input_string ) bind (C, name="PCAIRSetSmoothType_c")

      use iso_c_binding, only: C_CHAR, c_null_char
      implicit none

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      character (kind=c_char, len=1), dimension (10), intent (in) :: input_string

      type(tPC)              :: pc
      PetscErrorCode         :: ierr
      character (len=10)     :: regular_string
      integer :: i      
      ! ~~~~~~~~

      pc%v = pc_ptr      
   
      ! Convert the string to fortran
      regular_string = " "
      loop_string: do i=1, 10
         if ( input_string (i) == c_null_char ) then
            exit loop_string
         else
            regular_string (i:i) = input_string (i)
         end if
      end do loop_string

      call PCAIRSetSmoothType(pc, regular_string, ierr)
      
   end subroutine PCAIRSetSmoothType_c   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetSmoothType_c(pc_ptr, output_string) bind(C, name="PCAIRGetSmoothType_c")
      use iso_c_binding, only: c_char, c_null_char
      implicit none
   
      integer(c_long_long), intent(inout) :: pc_ptr
      character(kind=c_char, len=1), dimension(*), intent(out) :: output_string
   
      type(tPC) :: pc
      PetscErrorCode :: ierr
      character(len=10) :: fortran_string
      integer :: i, n
   
      pc%v = pc_ptr
   
      ! Get the Fortran string (trimmed)
      call PCAIRGetSmoothType(pc, fortran_string, ierr)
      n = len_trim(fortran_string)
   
      ! Copy Fortran string to C array, null-terminate
      do i = 1, n
         output_string(i) = fortran_string(i:i)
      end do
      output_string(n+1) = c_null_char
   
      ! Fill the rest with c_null_char if output_string is longer
      do i = n+2, 256
         output_string(i) = c_null_char
      end do
   
   end subroutine PCAIRGetSmoothType_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetMatrixFreePolys_c(pc_ptr, mf) bind(C, name='PCAIRSetMatrixFreePolys_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, value, intent(in)         :: mf

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetMatrixFreePolys(pc, mf, ierr)

   end subroutine PCAIRSetMatrixFreePolys_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetOnePointClassicalProlong_c(pc_ptr, onep) bind(C, name='PCAIRSetOnePointClassicalProlong_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, value, intent(in)         :: onep

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetOnePointClassicalProlong(pc, onep, ierr)

   end subroutine PCAIRSetOnePointClassicalProlong_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetFullSmoothingUpAndDown_c(pc_ptr, full) bind(C, name='PCAIRSetFullSmoothingUpAndDown_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, value, intent(in)         :: full

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetFullSmoothingUpAndDown(pc, full, ierr)

   end subroutine PCAIRSetFullSmoothingUpAndDown_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetSymmetric_c(pc_ptr, sym) bind(C, name='PCAIRSetSymmetric_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, value, intent(in)         :: sym

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetSymmetric(pc, sym, ierr)

   end subroutine PCAIRSetSymmetric_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetConstrainW_c(pc_ptr, constrain) bind(C, name='PCAIRSetConstrainW_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, value, intent(in)         :: constrain

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetConstrainW(pc, constrain, ierr)

   end subroutine PCAIRSetConstrainW_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetConstrainZ_c(pc_ptr, constrain) bind(C, name='PCAIRSetConstrainZ_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, value, intent(in)         :: constrain

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetConstrainZ(pc, constrain, ierr)

   end subroutine PCAIRSetConstrainZ_c

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetImproveWIts_c(pc_ptr, its) bind(C, name='PCAIRSetImproveWIts_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, value, intent(in)          :: its

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetImproveWIts(pc, its, ierr)

   end subroutine PCAIRSetImproveWIts_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetImproveZIts_c(pc_ptr, its) bind(C, name='PCAIRSetImproveZIts_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, value, intent(in)          :: its

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetImproveZIts(pc, its, ierr)

   end subroutine PCAIRSetImproveZIts_c    
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetStrongRThreshold_c(pc_ptr, thresh) bind(C, name='PCAIRSetStrongRThreshold_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscReal, value, intent(in)         :: thresh

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetStrongRThreshold(pc, thresh, ierr)

   end subroutine PCAIRSetStrongRThreshold_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetInverseType_c(pc_ptr, inv_type) bind(C, name='PCAIRSetInverseType_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PCPFLAREINVType, value, intent(in)          :: inv_type

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetInverseType(pc, inv_type, ierr)

   end subroutine PCAIRSetInverseType_c

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCInverseType_c(pc_ptr, inv_type) bind(C, name='PCAIRSetCInverseType_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PCPFLAREINVType, value, intent(in)          :: inv_type

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetCInverseType(pc, inv_type, ierr)

   end subroutine PCAIRSetCInverseType_c   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetZType_c(pc_ptr, z_type) bind(C, name='PCAIRSetZType_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PCAIRZType, value, intent(in)          :: z_type

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetZType(pc, z_type, ierr)

   end subroutine PCAIRSetZType_c

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetLairDistance_c(pc_ptr, distance) bind(C, name='PCAIRSetLairDistance_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, value, intent(in)          :: distance

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetLairDistance(pc, distance, ierr)

   end subroutine PCAIRSetLairDistance_c   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetPolyOrder_c(pc_ptr, order) bind(C, name='PCAIRSetPolyOrder_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, value, intent(in)          :: order

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetPolyOrder(pc, order, ierr)

   end subroutine PCAIRSetPolyOrder_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetInverseSparsityOrder_c(pc_ptr, order) bind(C, name='PCAIRSetInverseSparsityOrder_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, value, intent(in)          :: order

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetInverseSparsityOrder(pc, order, ierr)

   end subroutine PCAIRSetInverseSparsityOrder_c

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCPolyOrder_c(pc_ptr, order) bind(C, name='PCAIRSetCPolyOrder_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, value, intent(in)          :: order

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetCPolyOrder(pc, order, ierr)

   end subroutine PCAIRSetCPolyOrder_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCInverseSparsityOrder_c(pc_ptr, order) bind(C, name='PCAIRSetCInverseSparsityOrder_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, value, intent(in)          :: order

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetCInverseSparsityOrder(pc, order, ierr)

   end subroutine PCAIRSetCInverseSparsityOrder_c   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCoarsestInverseType_c(pc_ptr, inv_type) bind(C, name='PCAIRSetCoarsestInverseType_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PCPFLAREINVType, value, intent(in)          :: inv_type

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetCoarsestInverseType(pc, inv_type, ierr)

   end subroutine PCAIRSetCoarsestInverseType_c   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCoarsestPolyOrder_c(pc_ptr, order) bind(C, name='PCAIRSetCoarsestPolyOrder_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, value, intent(in)          :: order

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetCoarsestPolyOrder(pc, order, ierr)

   end subroutine PCAIRSetCoarsestPolyOrder_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCoarsestInverseSparsityOrder_c(pc_ptr, order) bind(C, name='PCAIRSetCoarsestInverseSparsityOrder_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, value, intent(in)          :: order

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetCoarsestInverseSparsityOrder(pc, order, ierr)

   end subroutine PCAIRSetCoarsestInverseSparsityOrder_c 
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCoarsestMatrixFreePolys_c(pc_ptr, mf) bind(C, name='PCAIRSetCoarsestMatrixFreePolys_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, value, intent(in)         :: mf

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetCoarsestMatrixFreePolys(pc, mf, ierr)

   end subroutine PCAIRSetCoarsestMatrixFreePolys_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCoarsestSubcomm_c(pc_ptr, subcomm) bind(C, name='PCAIRSetCoarsestSubcomm_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, value, intent(in)         :: subcomm

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetCoarsestSubcomm(pc, subcomm, ierr)

   end subroutine PCAIRSetCoarsestSubcomm_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetRDrop_c(pc_ptr, rdrop) bind(C, name='PCAIRSetRDrop_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscReal, value, intent(in)         :: rdrop

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetRDrop(pc, rdrop, ierr)

   end subroutine PCAIRSetRDrop_c  
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetADrop_c(pc_ptr, adrop) bind(C, name='PCAIRSetADrop_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscReal, value, intent(in)         :: adrop

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetADrop(pc, adrop, ierr)

   end subroutine PCAIRSetADrop_c 
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetALump_c(pc_ptr, lump) bind(C, name='PCAIRSetALump_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, value, intent(in)         :: lump

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetALump(pc, lump, ierr)

   end subroutine PCAIRSetALump_c

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetReuseSparsity_c(pc_ptr, reuse) bind(C, name='PCAIRSetReuseSparsity_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, value, intent(in)         :: reuse

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetReuseSparsity(pc, reuse, ierr)

   end subroutine PCAIRSetReuseSparsity_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetReusePolyCoeffs_c(pc_ptr, reuse) bind(C, name='PCAIRSetReusePolyCoeffs_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscBool, value, intent(in)         :: reuse

      type(tPC)                  :: pc
      PetscErrorCode         :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr
      call PCAIRSetReusePolyCoeffs(pc, reuse, ierr)

   end subroutine PCAIRSetReusePolyCoeffs_c
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetPolyCoeffs_c(pc_ptr, petsc_level, which_inverse, &
               coeffs_ptr, row_size, col_size) bind(C, name='PCAIRSetPolyCoeffs_c')

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr
      PetscInt, value, intent(in)         :: petsc_level
      integer(c_int), value, intent(in)   :: which_inverse 
      PetscInt, value, intent(in)         :: row_size, col_size
      type(c_ptr), value, intent(in)      :: coeffs_ptr

      type(tPC)                              :: pc
      PetscErrorCode                         :: ierr
      integer                                :: our_level, errorcode
      PetscInt                               :: num_levels, i_loc, j_loc
      type(tPC)                              :: pc_shell
      type(pc_air_multigrid_data), pointer   :: pc_air_data   
      PetscReal, pointer :: coeffs_c(:,:)   

      ! ~~~~~~~~

      pc%v = pc_ptr

      ! Get the underlying PCShell
      call PCAIRGetPCShell(pc, pc_shell)

      ! Get the PC shell context
      call PCShellGetContext(pc_shell, pc_air_data, ierr)

      ! Get the number of levels in our mg
      call PCAIRGetNumLevels(pc, num_levels, ierr) 

      ! We order our levels from 1 to num_levels
      our_level = int(num_levels) - int(petsc_level)

      call c_f_pointer(coeffs_ptr, coeffs_c, shape=[row_size, col_size])
      
      ! Inverse Aff
      if (which_inverse == COEFFS_INV_AFF) then

         do i_loc = 1, col_size
            do j_loc = 1, row_size
               pc_air_data%air_data%inv_A_ff_poly_data(our_level)%coefficients(j_loc, i_loc) = &
                     coeffs_c(j_loc, i_loc)
            end do
         end do

      ! Inverse dropped Aff
      else if (which_inverse == COEFFS_INV_AFF_DROPPED) then

         do i_loc = 1, col_size
            do j_loc = 1, row_size
               pc_air_data%air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients(j_loc, i_loc) = &
                     coeffs_c(j_loc, i_loc)
            end do
         end do     

      ! Inverse Acc
      else if (which_inverse == COEFFS_INV_ACC) then

         do i_loc = 1, col_size
            do j_loc = 1, row_size
               pc_air_data%air_data%inv_A_cc_poly_data(our_level)%coefficients(j_loc, i_loc) = &
                     coeffs_c(j_loc, i_loc)
            end do
         end do          

      ! Coarsest grid matrix
      else if (which_inverse == COEFFS_INV_COARSE) then

         do i_loc = 1, col_size
            do j_loc = 1, row_size
               pc_air_data%air_data%inv_coarsest_poly_data%coefficients(j_loc, i_loc) = &
                     coeffs_c(j_loc, i_loc)
            end do
         end do        

      else
         print *, "Unknown which_inverse in PCAIRSetPolyCoeffs_c"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if      


   end subroutine PCAIRSetPolyCoeffs_c      

! -------------------------------------------------------------------------------------------------------------------------------

end module pcair_c_fortran_bindings