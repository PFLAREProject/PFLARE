module pcair_interfaces

   use iso_c_binding
   use petscksp
   use pcair_shell

#include "petsc/finclude/petscksp.h"
#include "finclude/pflare_types.h"

   implicit none

   public

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! Contains Fortran interfaces to the PCAIR options 
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------      

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Get routines - interfaces to the C routines in PCAIR
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   interface   
      
      subroutine c_PCAIRGetPCShell(A_array, B_array) &
         bind(c, name="c_PCAIRGetPCShell")
         use iso_c_binding
         integer(c_long_long) :: A_array, B_array
      end subroutine c_PCAIRGetPCShell         
 
   end interface   

   ! Which polynomial coefficients to get/set in 
   ! PCAIRGetPolyCoeffs/PCAIRSetPolyCoeffs
   integer, parameter :: COEFFS_INV_AFF = 0
   integer, parameter :: COEFFS_INV_AFF_DROPPED = 1
   integer, parameter :: COEFFS_INV_ACC = 2
   integer, parameter :: COEFFS_INV_COARSE = 3
   
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Set routines - interfaces to the C routines in PCAIR
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   ! ~~~~~~~~~~~~~~~~

   contains

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Get routines - Fortran versions of the C routines in PCAIR
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetPCShell(pc, pc_shell) 

      ! Gets the PCShell that lives in our PCAIR

      ! ~~~~~~~~
      type(tPC), intent(inout) :: pc, pc_shell

      integer(c_long_long) :: pc_ptr, pc_shell_ptr
      ! ~~~~~~~~

      pc_ptr = pc%v

      ! Call the c routine
      call c_PCAIRGetPCShell(pc_ptr, pc_shell_ptr)
      pc_shell%v = pc_shell_ptr

   end subroutine PCAIRGetPCShell

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetOptions(pc, options) 

      ! Gets the air options that lives in our PCAIR

      ! ~~~~~~~~
      type(tPC), intent(inout)                  :: pc
      type(air_options), pointer, intent(inout) :: options

      type(tPC)                             :: pc_shell
      type(pc_air_multigrid_data), pointer  :: pc_air_data=>null()
      PetscErrorCode :: ierr
      ! ~~~~~~~~

      ! Get the underlying PCShell
      call PCAIRGetPCShell(pc, pc_shell)

      ! Get the PC shell context
      call PCShellGetContext(pc_shell, pc_air_data, ierr)
      
      ! Return the options
      options => pc_air_data%air_data%options

   end subroutine PCAIRGetOptions 


! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetPrintStatsTimings(pc, print_stats, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: print_stats
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      print_stats = options%print_stats_timings
      ierr = 0

   end subroutine PCAIRGetPrintStatsTimings   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetMaxLevels(pc, max_levels, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: max_levels
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      max_levels = options%max_levels
      ierr = 0

   end subroutine PCAIRGetMaxLevels

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCoarseEqLimit(pc, coarse_eq_limit, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: coarse_eq_limit
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      coarse_eq_limit = options%coarse_eq_limit
      ierr = 0

   end subroutine PCAIRGetCoarseEqLimit   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetAutoTruncateStartLevel(pc, start_level, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: start_level
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      start_level = options%auto_truncate_start_level
      ierr = 0

   end subroutine PCAIRGetAutoTruncateStartLevel   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetAutoTruncateTol(pc, tol, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(out)        :: tol
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      tol = options%auto_truncate_tol
      ierr = 0

   end subroutine PCAIRGetAutoTruncateTol   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetNumLevels(pc, num_levels, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: num_levels
      PetscErrorCode, intent(out)   :: ierr

      type(tPC)                             :: pc_shell
      type(pc_air_multigrid_data), pointer  :: pc_air_data=>null()
      ! ~~~~~~~~

      ! Get the underlying PCShell
      call PCAIRGetPCShell(pc, pc_shell)

      ! Get the PC shell context
      call PCShellGetContext(pc_shell, pc_air_data, ierr)
  
      num_levels = pc_air_data%air_data%no_levels
      ierr = 0

   end subroutine PCAIRGetNumLevels
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetPolyCoeffs(pc, petsc_level, which_inverse, coeffs, ierr) 

      ! This routine returns a copy of the coefficients in coeffs
      ! The C version of this routine instead just returns a pointer to the coefficients
      ! in the PCAIR object and hence should be copied externally 

      ! ~~~~~~~~
      type(tPC), intent(inout)                           :: pc
      PetscInt, intent(in)                               :: petsc_level
      integer, intent(in)                                :: which_inverse
      PetscReal, dimension(:,:), pointer, intent(inout)       :: coeffs
      PetscErrorCode, intent(out)                        :: ierr

      type(tPC)                             :: pc_shell
      type(pc_air_multigrid_data), pointer  :: pc_air_data=>null()
      PetscInt                              :: num_levels
      integer                               :: our_level, errorcode
      ! ~~~~~~~~

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

         ! Check sizes
         if (associated(coeffs)) then
            if (.NOT. (size(coeffs,1) == size(pc_air_data%air_data%inv_A_ff_poly_data(our_level)%coefficients, 1) .AND. &
                size(coeffs,2) == size(pc_air_data%air_data%inv_A_ff_poly_data(our_level)%coefficients, 2))) then

               deallocate(coeffs)
               coeffs => null()
            end if
         end if

         if (.NOT. associated(coeffs)) then
            allocate(coeffs(size(pc_air_data%air_data%inv_A_ff_poly_data(our_level)%coefficients, 1), &
                            size(pc_air_data%air_data%inv_A_ff_poly_data(our_level)%coefficients, 2)))
         end if

         coeffs = pc_air_data%air_data%inv_A_ff_poly_data(our_level)%coefficients

      ! Inverse dropped Aff
      else if (which_inverse == COEFFS_INV_AFF_DROPPED) then

         ! Check sizes
         if (associated(coeffs)) then
            if (.NOT. (size(coeffs,1) == size(pc_air_data%air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients, 1) .AND. &
                size(coeffs,2) == size(pc_air_data%air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients, 2))) then

               deallocate(coeffs)
               coeffs => null()
            end if
         end if

         if (.NOT. associated(coeffs)) then
            allocate(coeffs(size(pc_air_data%air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients, 1), &
                            size(pc_air_data%air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients, 2)))
         end if         

         coeffs = pc_air_data%air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients

      ! Inverse Acc
      else if (which_inverse == COEFFS_INV_ACC) then

         ! Check sizes
         if (associated(coeffs)) then
            if (.NOT. (size(coeffs,1) == size(pc_air_data%air_data%inv_A_cc_poly_data(our_level)%coefficients, 1) .AND. &
                size(coeffs,2) == size(pc_air_data%air_data%inv_A_cc_poly_data(our_level)%coefficients, 2))) then

               deallocate(coeffs)
               coeffs => null()
            end if
         end if

         if (.NOT. associated(coeffs)) then
            allocate(coeffs(size(pc_air_data%air_data%inv_A_cc_poly_data(our_level)%coefficients, 1), &
                            size(pc_air_data%air_data%inv_A_cc_poly_data(our_level)%coefficients, 2)))
         end if          

         coeffs = pc_air_data%air_data%inv_A_cc_poly_data(our_level)%coefficients

      ! Coarsest grid matrix
      else if (which_inverse == COEFFS_INV_COARSE) then

         ! Check sizes
         if (associated(coeffs)) then
            if (.NOT. (size(coeffs,1) == size(pc_air_data%air_data%inv_coarsest_poly_data%coefficients, 1) .AND. &
                size(coeffs,2) == size(pc_air_data%air_data%inv_coarsest_poly_data%coefficients, 2))) then

               deallocate(coeffs)
               coeffs => null()
            end if
         end if

         if (.NOT. associated(coeffs)) then
            allocate(coeffs(size(pc_air_data%air_data%inv_coarsest_poly_data%coefficients, 1), &
                            size(pc_air_data%air_data%inv_coarsest_poly_data%coefficients, 2)))
         end if         

         coeffs = pc_air_data%air_data%inv_coarsest_poly_data%coefficients

      else
         print *, "Unknown which_inverse in PCAIRGetPolyCoeffs"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ierr = 0

   end subroutine PCAIRGetPolyCoeffs 
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetPolyCoeffs(pc, petsc_level, which_inverse, coeffs, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)                           :: pc
      PetscInt, intent(in)                               :: petsc_level
      integer, intent(in)                                :: which_inverse
      PetscReal, dimension(:,:), pointer, intent(in)          :: coeffs
      PetscErrorCode, intent(out)                        :: ierr

      type(tPC)                             :: pc_shell
      type(pc_air_multigrid_data), pointer  :: pc_air_data=>null()
      PetscInt                              :: num_levels
      integer                               :: our_level, errorcode
      ! ~~~~~~~~

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

         if (.NOT. associated(pc_air_data%air_data%inv_A_ff_poly_data(our_level)%coefficients)) then
            print *, "PCAIR MG not setup yet"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)            
         end if

         ! Check sizes
         if (.NOT. (size(coeffs,1) == size(pc_air_data%air_data%inv_A_ff_poly_data(our_level)%coefficients, 1) .AND. &
               size(coeffs,2) == size(pc_air_data%air_data%inv_A_ff_poly_data(our_level)%coefficients, 2))) then

            print *, "Sizes wrong in PCAIRSetPolyCoeffs"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
         end if

         pc_air_data%air_data%inv_A_ff_poly_data(our_level)%coefficients = coeffs

      ! Inverse dropped Aff
      else if (which_inverse == COEFFS_INV_AFF_DROPPED) then

         if (.NOT. associated(pc_air_data%air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients)) then
            print *, "PCAIR MG not setup yet"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)            
         end if         

         ! Check sizes
         if (.NOT. (size(coeffs,1) == size(pc_air_data%air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients, 1) .AND. &
               size(coeffs,2) == size(pc_air_data%air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients, 2))) then

            print *, "Sizes wrong in PCAIRSetPolyCoeffs"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
         end if
       
         pc_air_data%air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients = coeffs

      ! Inverse Acc
      else if (which_inverse == COEFFS_INV_ACC) then

         if (.NOT. associated(pc_air_data%air_data%inv_A_cc_poly_data(our_level)%coefficients)) then
            print *, "PCAIR MG not setup yet"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)            
         end if          

         ! Check sizes
         if (.NOT. (size(coeffs,1) == size(pc_air_data%air_data%inv_A_cc_poly_data(our_level)%coefficients, 1) .AND. &
               size(coeffs,2) == size(pc_air_data%air_data%inv_A_cc_poly_data(our_level)%coefficients, 2))) then

            print *, "Sizes wrong in PCAIRSetPolyCoeffs"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
         end if

         pc_air_data%air_data%inv_A_cc_poly_data(our_level)%coefficients = coeffs

      ! Coarsest grid matrix
      else if (which_inverse == COEFFS_INV_COARSE) then

         if (.NOT. associated(pc_air_data%air_data%inv_coarsest_poly_data%coefficients)) then
            print *, "PCAIR MG not setup yet"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)            
         end if          

         ! Check sizes
         if (.NOT. (size(coeffs,1) == size(pc_air_data%air_data%inv_coarsest_poly_data%coefficients, 1) .AND. &
               size(coeffs,2) == size(pc_air_data%air_data%inv_coarsest_poly_data%coefficients, 2))) then

            print *, "Sizes wrong in PCAIRSetPolyCoeffs"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
         end if        

         pc_air_data%air_data%inv_coarsest_poly_data%coefficients = coeffs

      else
         print *, "Unknown which_inverse in PCAIRSetPolyCoeffs"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ierr = 0

   end subroutine PCAIRSetPolyCoeffs     
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetProcessorAgglom(pc, processor_agglom, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: processor_agglom
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      processor_agglom = options%processor_agglom
      ierr = 0

   end subroutine PCAIRGetProcessorAgglom
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetProcessorAgglomRatio(pc, ratio, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(out)        :: ratio
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      ratio = options%processor_agglom_ratio
      ierr = 0

   end subroutine PCAIRGetProcessorAgglomRatio

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetProcessorAgglomFactor(pc, factor, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: factor
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      factor = options%processor_agglom_factor
      ierr = 0

   end subroutine PCAIRGetProcessorAgglomFactor  
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetProcessEqLimit(pc, limit, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: limit
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      limit = options%process_eq_limit
      ierr = 0

   end subroutine PCAIRGetProcessEqLimit      
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetSubcomm(pc, subcomm, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: subcomm
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      subcomm = options%subcomm
      ierr = 0

   end subroutine PCAIRGetSubcomm   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetStrongThreshold(pc, thresh, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(out)        :: thresh
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      thresh = options%strong_threshold
      ierr = 0

   end subroutine PCAIRGetStrongThreshold   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetDDCIts(pc, its, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: its
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      its = options%ddc_its
      ierr = 0

   end subroutine PCAIRGetDDCIts 

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetMaxDDRatio(pc, ratio, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(out)        :: ratio
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      ratio = options%max_dd_ratio
      ierr = 0

   end subroutine PCAIRGetMaxDDRatio

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetDDCFraction(pc, frac, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(out)        :: frac
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      frac = options%ddc_fraction
      ierr = 0

   end subroutine PCAIRGetDDCFraction
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCFSplittingType(pc, algo, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      CFSplittingType, intent(out)  :: algo
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      algo = options%cf_splitting_type
      ierr = 0

   end subroutine PCAIRGetCFSplittingType

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetMaxLubySteps(pc, steps, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: steps
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      steps = options%max_luby_steps
      ierr = 0

   end subroutine PCAIRGetMaxLubySteps
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetMatrixFreePolys(pc, mf, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: mf
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      mf = options%matrix_free_polys
      ierr = 0

   end subroutine PCAIRGetMatrixFreePolys
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetOnePointClassicalProlong(pc, onep, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: onep
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      onep = options%one_point_classical_prolong
      ierr = 0

   end subroutine PCAIRGetOnePointClassicalProlong
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetFullSmoothingUpAndDown(pc, full, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: full
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      full = options%full_smoothing_up_and_down
      ierr = 0

   end subroutine PCAIRGetFullSmoothingUpAndDown
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetSymmetric(pc, sym, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: sym
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      sym = options%symmetric
      ierr = 0

   end subroutine PCAIRGetSymmetric
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetConstrainW(pc, constrain, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: constrain
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      constrain = options%constrain_w
      ierr = 0

   end subroutine PCAIRGetConstrainW
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetConstrainZ(pc, constrain, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: constrain
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      constrain = options%constrain_z
      ierr = 0

   end subroutine PCAIRGetConstrainZ

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetImproveWIts(pc, its, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: its
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      its = options%improve_w_its
      ierr = 0

   end subroutine PCAIRGetImproveWIts     

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetImproveZIts(pc, its, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: its
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      its = options%improve_z_its
      ierr = 0

   end subroutine PCAIRGetImproveZIts   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetStrongRThreshold(pc, thresh, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(out)        :: thresh
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      thresh = options%strong_r_threshold
      ierr = 0

   end subroutine PCAIRGetStrongRThreshold
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetInverseType(pc, inv_type, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PCPFLAREINVType, intent(out)         :: inv_type
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      inv_type = options%inverse_type
      ierr = 0

   end subroutine PCAIRGetInverseType

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCInverseType(pc, inv_type, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PCPFLAREINVType, intent(out)         :: inv_type
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      inv_type = options%c_inverse_type
      ierr = 0

   end subroutine PCAIRGetCInverseType   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetZType(pc, z_type, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PCAIRZType, intent(out)         :: z_type
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      z_type = options%z_type
      ierr = 0

   end subroutine PCAIRGetZType

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetLairDistance(pc, distance, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: distance
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      distance = options%lair_distance
      ierr = 0

   end subroutine PCAIRGetLairDistance
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCPolyOrder(pc, order, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: order
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      order = options%c_poly_order
      ierr = 0

   end subroutine PCAIRGetCPolyOrder
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCInverseSparsityOrder(pc, order, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: order
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      order = options%c_inverse_sparsity_order
      ierr = 0

   end subroutine PCAIRGetCInverseSparsityOrder   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetPolyOrder(pc, order, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: order
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      order = options%poly_order
      ierr = 0

   end subroutine PCAIRGetPolyOrder
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetInverseSparsityOrder(pc, order, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: order
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      order = options%inverse_sparsity_order
      ierr = 0

   end subroutine PCAIRGetInverseSparsityOrder

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCoarsestInverseType(pc, inv_type, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PCPFLAREINVType, intent(out)         :: inv_type
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      inv_type = options%coarsest_inverse_type
      ierr = 0

   end subroutine PCAIRGetCoarsestInverseType   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCoarsestPolyOrder(pc, order, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: order
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      order = options%coarsest_poly_order
      ierr = 0

   end subroutine PCAIRGetCoarsestPolyOrder
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCoarsestInverseSparsityOrder(pc, order, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(out)         :: order
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      order = options%coarsest_inverse_sparsity_order
      ierr = 0

   end subroutine PCAIRGetCoarsestInverseSparsityOrder 
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCoarsestMatrixFreePolys(pc, mf, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: mf
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      mf = options%coarsest_matrix_free_polys
      ierr = 0

   end subroutine PCAIRGetCoarsestMatrixFreePolys
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetCoarsestSubcomm(pc, subcomm, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: subcomm
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      subcomm = options%coarsest_subcomm
      ierr = 0

   end subroutine PCAIRGetCoarsestSubcomm
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetRDrop(pc, rdrop, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(out)        :: rdrop
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      rdrop = options%r_drop
      ierr = 0

   end subroutine PCAIRGetRDrop  
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetADrop(pc, adrop, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(out)        :: adrop
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      adrop = options%a_drop
      ierr = 0

   end subroutine PCAIRGetADrop 
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetALump(pc, lump, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: lump
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      lump = options%a_lump
      ierr = 0

   end subroutine PCAIRGetALump
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetReuseSparsity(pc, reuse, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: reuse
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      reuse = options%reuse_sparsity
      ierr = 0

   end subroutine PCAIRGetReuseSparsity
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetReusePolyCoeffs(pc, reuse, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(out)        :: reuse
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Get the options
      call PCAIRGetOptions(pc, options)    
      reuse = options%reuse_poly_coeffs
      ierr = 0

   end subroutine PCAIRGetReusePolyCoeffs    
   
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

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%print_stats_timings = print_stats
      ierr = 0

   end subroutine PCAIRSetPrintStatsTimings   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetMaxLevels(pc, max_levels, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: max_levels
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%max_levels = int(max_levels)
      ierr = 0

   end subroutine PCAIRSetMaxLevels

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCoarseEqLimit(pc, coarse_eq_limit, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: coarse_eq_limit
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%coarse_eq_limit = coarse_eq_limit
      ierr = 0

   end subroutine PCAIRSetCoarseEqLimit   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetAutoTruncateStartLevel(pc, start_level, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: start_level
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%auto_truncate_start_level = int(start_level)
      ierr = 0

   end subroutine PCAIRSetAutoTruncateStartLevel   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetAutoTruncateTol(pc, tol, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(in)         :: tol
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%auto_truncate_tol = tol
      ierr = 0

   end subroutine PCAIRSetAutoTruncateTol   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetProcessorAgglom(pc, processor_agglom, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: processor_agglom
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%processor_agglom = processor_agglom
      ierr = 0

   end subroutine PCAIRSetProcessorAgglom
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetProcessorAgglomRatio(pc, ratio, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(in)         :: ratio
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%processor_agglom_ratio = ratio
      ierr = 0

   end subroutine PCAIRSetProcessorAgglomRatio

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetProcessorAgglomFactor(pc, factor, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: factor
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%processor_agglom_factor = int(factor)
      ierr = 0

   end subroutine PCAIRSetProcessorAgglomFactor
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetProcessEqLimit(pc, limit, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: limit
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%process_eq_limit = limit
      ierr = 0

   end subroutine PCAIRSetProcessEqLimit    
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetSubcomm(pc, subcomm, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: subcomm
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%subcomm = subcomm
      ierr = 0

   end subroutine PCAIRSetSubcomm   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetStrongThreshold(pc, thresh, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(in)         :: thresh
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%strong_threshold = thresh
      ierr = 0

   end subroutine PCAIRSetStrongThreshold   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetDDCIts(pc, its, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: its
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%ddc_its = int(its)
      ierr = 0

   end subroutine PCAIRSetDDCIts

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetMaxDDRatio(pc, ratio, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(in)         :: ratio
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%max_dd_ratio = ratio
      ierr = 0

   end subroutine PCAIRSetMaxDDRatio   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetDDCFraction(pc, frac, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(in)         :: frac
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%ddc_fraction = frac
      ierr = 0

   end subroutine PCAIRSetDDCFraction
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCFSplittingType(pc, algo, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      CFSplittingType, intent(in)   :: algo
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%cf_splitting_type = int(algo)
      ierr = 0

   end subroutine PCAIRSetCFSplittingType

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetMaxLubySteps(pc, steps, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: steps
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%max_luby_steps = int(steps)
      ierr = 0

   end subroutine PCAIRSetMaxLubySteps

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetSmoothType(pc, input_string, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)       :: pc
      character(len = *), intent(in) :: input_string
      PetscErrorCode, intent(out)    :: ierr

      type(air_options), pointer :: options
      character(len = 1) :: current_char, old_char
      integer :: smooth_order_index
      integer :: errorcode, n, i
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    

      n = len_trim(input_string)
      if (n == 0) then
         print *, "Smooth order must be non-zero length"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)         
      end if
      
      old_char = "0"
      smooth_order_index = 1
      options%smooth_order = 0
      options%any_c_smooths = .FALSE.
      do i = 1, n

         current_char = input_string(i:i)

         if (current_char /= old_char .AND. i /= 1) then
            smooth_order_index = smooth_order_index + 1
         end if

         select case (current_char)
         case ('f', 'F')
            options%smooth_order(smooth_order_index) = options%smooth_order(smooth_order_index) + 1
         case ('c', 'C')
            options%smooth_order(smooth_order_index) = options%smooth_order(smooth_order_index) - 1
            options%any_c_smooths = .TRUE.
         case default
            print *, "Smooth order must be only f or c characters"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode) 
         end select           

         old_char = current_char
      end do      

      ierr = 0

   end subroutine PCAIRSetSmoothType   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRGetSmoothType(pc, output_string, ierr)

      ! ~~~~~~~~
      type(tPC), intent(inout)         :: pc
      character(len=*), intent(out)    :: output_string
      PetscErrorCode, intent(out)      :: ierr
   
      type(air_options), pointer :: options
      integer :: i, j, idx, n, absval, total_len
      character(len=1) :: ch
      character(len=:), allocatable :: temp_string
      ! ~~~~~~~~
   
      call PCAIRGetOptions(pc, options)
   
      ! First, compute the total length needed
      total_len = 0
      n = size(options%smooth_order)
      do i = 1, n
         if (options%smooth_order(i) /= 0) then
            total_len = total_len + abs(options%smooth_order(i))
         end if
      end do
   
      ! Allocate a temporary string of the exact needed length
      if (total_len > 0) then
         allocate(character(len=total_len) :: temp_string)
         idx = 1
         do i = 1, n
            if (options%smooth_order(i) > 0) then
               ch = 'f'
               absval = options%smooth_order(i)
            else if (options%smooth_order(i) < 0) then
               ch = 'c'
               absval = -options%smooth_order(i)
            else
               cycle
            end if
            do j = 1, absval
               temp_string(idx:idx) = ch
               idx = idx + 1
            end do
         end do
         ! Copy to output, trimming or padding as needed
         output_string = temp_string
         deallocate(temp_string)
      else
         output_string = ''
      end if
   
      ierr = 0
   end subroutine PCAIRGetSmoothType
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetMatrixFreePolys(pc, mf, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: mf
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%matrix_free_polys = mf
      ierr = 0

   end subroutine PCAIRSetMatrixFreePolys
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetOnePointClassicalProlong(pc, onep, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: onep
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%one_point_classical_prolong = onep
      ierr = 0

   end subroutine PCAIRSetOnePointClassicalProlong
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetFullSmoothingUpAndDown(pc, full, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: full
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%full_smoothing_up_and_down = full
      ierr = 0

   end subroutine PCAIRSetFullSmoothingUpAndDown
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetSymmetric(pc, sym, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: sym
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%symmetric = sym
      ierr = 0

   end subroutine PCAIRSetSymmetric
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetConstrainW(pc, constrain, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: constrain
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%constrain_w = constrain
      ierr = 0

   end subroutine PCAIRSetConstrainW
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetConstrainZ(pc, constrain, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: constrain
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%constrain_z = constrain
      ierr = 0

   end subroutine PCAIRSetConstrainZ

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetImproveWIts(pc, its, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: its
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%improve_w_its = int(its)
      ierr = 0

   end subroutine PCAIRSetImproveWIts    

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetImproveZIts(pc, its, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: its
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%improve_z_its = int(its)
      ierr = 0

   end subroutine PCAIRSetImproveZIts   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetStrongRThreshold(pc, thresh, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(in)         :: thresh
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%strong_r_threshold = thresh
      ierr = 0

   end subroutine PCAIRSetStrongRThreshold
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetInverseType(pc, inv_type, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PCPFLAREINVType, intent(in)          :: inv_type
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%inverse_type = inv_type
      ierr = 0

   end subroutine PCAIRSetInverseType

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCInverseType(pc, inv_type, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PCPFLAREINVType, intent(in)          :: inv_type
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%c_inverse_type = inv_type
      ierr = 0

   end subroutine PCAIRSetCInverseType   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetZType(pc, z_type, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PCAIRZType, intent(in)          :: z_type
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%z_type = z_type
      ierr = 0

   end subroutine PCAIRSetZType

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetLairDistance(pc, distance, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: distance
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%lair_distance = int(distance)
      ierr = 0

   end subroutine PCAIRSetLairDistance   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetPolyOrder(pc, order, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: order
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%poly_order = int(order)
      ierr = 0

   end subroutine PCAIRSetPolyOrder
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetInverseSparsityOrder(pc, order, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: order
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%inverse_sparsity_order = int(order)
      ierr = 0

   end subroutine PCAIRSetInverseSparsityOrder

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCPolyOrder(pc, order, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: order
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%c_poly_order = int(order)
      ierr = 0

   end subroutine PCAIRSetCPolyOrder
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCInverseSparsityOrder(pc, order, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: order
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%c_inverse_sparsity_order = int(order)
      ierr = 0

   end subroutine PCAIRSetCInverseSparsityOrder   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCoarsestInverseType(pc, inv_type, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PCPFLAREINVType, intent(in)          :: inv_type
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%coarsest_inverse_type = inv_type
      ierr = 0

   end subroutine PCAIRSetCoarsestInverseType   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCoarsestPolyOrder(pc, order, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: order
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%coarsest_poly_order = int(order)
      ierr = 0

   end subroutine PCAIRSetCoarsestPolyOrder
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCoarsestInverseSparsityOrder(pc, order, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscInt, intent(in)          :: order
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%coarsest_inverse_sparsity_order = int(order)
      ierr = 0

   end subroutine PCAIRSetCoarsestInverseSparsityOrder 
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCoarsestMatrixFreePolys(pc, mf, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: mf
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%coarsest_matrix_free_polys = mf
      ierr = 0

   end subroutine PCAIRSetCoarsestMatrixFreePolys
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetCoarsestSubcomm(pc, subcomm, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: subcomm
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%coarsest_subcomm = subcomm
      ierr = 0

   end subroutine PCAIRSetCoarsestSubcomm
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetRDrop(pc, rdrop, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(in)         :: rdrop
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%r_drop = rdrop
      ierr = 0

   end subroutine PCAIRSetRDrop  
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetADrop(pc, adrop, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscReal, intent(in)         :: adrop
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%a_drop = adrop
      ierr = 0

   end subroutine PCAIRSetADrop 
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetALump(pc, lump, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: lump
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%a_lump = lump
      ierr = 0

   end subroutine PCAIRSetALump

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetReuseSparsity(pc, reuse, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: reuse
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%reuse_sparsity = reuse
      ierr = 0

   end subroutine PCAIRSetReuseSparsity
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCAIRSetReusePolyCoeffs(pc, reuse, ierr) 

      ! ~~~~~~~~
      type(tPC), intent(inout)      :: pc
      PetscBool, intent(in)         :: reuse
      PetscErrorCode, intent(out)   :: ierr

      type(air_options), pointer :: options
      ! ~~~~~~~~

      ! Set the options
      call PCAIRGetOptions(pc, options)    
      options%reuse_poly_coeffs = reuse
      ierr = 0

   end subroutine PCAIRSetReusePolyCoeffs    

! -------------------------------------------------------------------------------------------------------------------------------

end module pcair_interfaces

