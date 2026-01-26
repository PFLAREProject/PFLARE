module pcair_shell

   use petscksp
   use c_petsc_interfaces
   use pcair_data_type
   use air_mg_setup
   use air_data_type_routines

#include "petsc/finclude/petscksp.h"

   implicit none

   public
   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine create_pc_air_shell(pc_air_data, pc)
      
      ! Sets the PCShell that knows how to setup itself up and re-use
      ! pc should have been obtained from a call to KSPGetPC
      ! pc_air_data should have been created outside this routine

      ! ~~~~~~
      type(pc_air_multigrid_data), intent(inout)   :: pc_air_data
      type(tPC), intent(inout)                     :: pc

      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX

      ! ~~~~~~  

      ! Set the pc as a shell
      call PCSetType(pc, PCSHELL, ierr)
      ! Set the pc_air_data
      call PCShellSetContext(pc, pc_air_data, ierr)
      ! Set the apply routine
      call PCShellSetApply(pc, PCApply_AIR_Shell, ierr)
      ! Set the destroy routine
      call PCShellSetDestroy(pc, PCDestroy_AIR_Shell, ierr)
      ! Set the setup routine
      call PCShellSetSetUp(pc, PCSetUp_AIR_Shell, ierr)
      ! Set the view routine
      call PCShellSetView(pc, PCView_AIR_Shell, ierr)
      ! Annoyingly there is no ability to set a reset routine
      ! in fortran, so if we want to call the equivalent of a 
      ! reset on our pcshell we need to manually call reset_pc_air_data

      ! Create our underlying PCMG object
      call PetscObjectGetComm(pc, MPI_COMM_MATRIX, ierr)    
      call PCCreate(MPI_COMM_MATRIX, pc_air_data%pcmg, ierr)

   end subroutine create_pc_air_shell

! -------------------------------------------------------------------------------------------------------------------------------   

   subroutine destroy_pc_air_data(pc_air_data)

      ! ~~~~~~
      type(pc_air_multigrid_data), intent(inout)    :: pc_air_data

      PetscErrorCode :: ierr
      ! ~~~~~~    

      call destroy_air_data(pc_air_data%air_data)
      ! Destroy the underlying pcmg
      call PCDestroy(pc_air_data%pcmg, ierr)  

   end subroutine destroy_pc_air_data
   
! -------------------------------------------------------------------------------------------------------------------------------   

   subroutine reset_pc_air_data(pc_air_data)

      ! ~~~~~~
      type(pc_air_multigrid_data), intent(inout)    :: pc_air_data

      PetscErrorCode :: ierr
      ! ~~~~~~    

      ! Reset the air data
      call reset_air_data(pc_air_data%air_data)
      ! We reset the underlying pcmg
      call PCReset(pc_air_data%pcmg, ierr)  

   end subroutine reset_pc_air_data
   
! -------------------------------------------------------------------------------------------------------------------------------   

   subroutine reset_pc_air_data_reuse(pc_air_data)

      ! ~~~~~~
      type(pc_air_multigrid_data), intent(inout)    :: pc_air_data

      PetscErrorCode :: ierr
      ! ~~~~~~    

      ! Reset the air data but keep the reuse
      call reset_air_data(pc_air_data%air_data, keep_reuse = .TRUE.)
      ! We reset the underlying pcmg
      call PCReset(pc_air_data%pcmg, ierr)  

   end subroutine reset_pc_air_data_reuse    

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCSetUp_AIR_Shell(pc, ierr)
      
      ! Setup our shell PC

      ! ~~~~~~
      type(tPC), intent(inout) :: pc
      PetscErrorCode, intent(inout)   :: ierr

      type(tMat)                 :: amat, pmat
      type(pc_air_multigrid_data), pointer  :: pc_air_data=>null()
      integer                    :: structure_flag
      PetscInt                   :: setupcalled

      ! ~~~~~~  

      ! Get the preconditioner matrix
      call PCGetOperators(pc, amat, pmat, ierr)
      ! Get the PC structure flag, either SAME_NONZERO_PATTERN or DIFFERENT_NONZERO_PATTERN
      call c_PCGetStructureFlag(pc%v, structure_flag)
      call PCGetSetupCalled_c(pc%v, setupcalled)
      ! Get the PC shell context
      call PCShellGetContext(pc, pc_air_data, ierr)

      ! We need to call the setup routines
      if (setupcalled == 0) then

         call setup_air_pcmg(amat, pmat, pc_air_data%air_data, pc_air_data%pcmg)

      ! We can possibly re-use some of the setup
      else

         ! If we've got a different non-zero pattern we've got to 
         ! start again 
         ! DIFFERENT_NONZERO_PATTERN = 0
         if (structure_flag == 0) then

            ! Reset the air data
            call reset_pc_air_data(pc_air_data)
            ! Set it up again
            call setup_air_pcmg(amat, pmat, pc_air_data%air_data, pc_air_data%pcmg)

         ! We can re-use the sparsity
         ! SAME_NONZERO_PATTERN = 2
         else if (structure_flag == 2) then

            if (pc_air_data%air_data%options%reuse_sparsity) then
               ! Reset the air data but keep any reuse data we need
               call reset_pc_air_data_reuse(pc_air_data)
            else
               ! Otherwise we can't reuse anything so reset the air data
               call reset_pc_air_data(pc_air_data)               
            end if
            ! Set it up again
            call setup_air_pcmg(amat, pmat, pc_air_data%air_data, pc_air_data%pcmg)

         end if
      end if

   end subroutine PCSetUp_AIR_Shell


! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCApply_AIR_Shell(pc, x, y, ierr)
      
      ! Apply our shell PC

      ! ~~~~~~
      type(tPC), intent(in)    :: pc
      type(tVec), intent(in)   :: x, y
      PetscErrorCode, intent(inout)   :: ierr

      type(pc_air_multigrid_data), pointer  :: pc_air_data=>null()

      ! ~~~~~~  

      ! Get the PC context
      call PCShellGetContext(pc, pc_air_data, ierr)
      ! Just apply the pcmg
      call PCApply(pc_air_data%pcmg, x, y, ierr)

   end subroutine PCApply_AIR_Shell   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCDestroy_AIR_Shell(pc, ierr)
      
      ! Destroy our shell PC

      ! ~~~~~~
      type(tPC), intent(in)    :: pc
      PetscErrorCode, intent(inout)   :: ierr

      type(pc_air_multigrid_data), pointer  :: pc_air_data=>null()

      ! ~~~~~~  

      ! Get the PC context
      call PCShellGetContext(pc, pc_air_data, ierr)
      ! Destroy the air data
      call destroy_pc_air_data(pc_air_data)
      ! Destroy the data structure
      deallocate(pc_air_data)

   end subroutine PCDestroy_AIR_Shell

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCReset_AIR_Shell(pc, ierr)
      
      ! Reset our shell PC

      ! ~~~~~~
      type(tPC), intent(in)    :: pc
      PetscErrorCode, intent(inout)   :: ierr

      type(pc_air_multigrid_data), pointer  :: pc_air_data=>null()

      ! ~~~~~~  

      ! Get the PC context
      call PCShellGetContext(pc, pc_air_data, ierr)
      ! Destroy the air data
      call reset_pc_air_data(pc_air_data)

   end subroutine PCReset_AIR_Shell   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCView_AIR_Shell(pc, viewer, ierr)
      
      ! Views our shell PC

      ! ~~~~~~
      type(tPC), intent(in)            :: pc
      type(tPetscViewer), intent(in)   :: viewer
      PetscErrorCode, intent(inout)           :: ierr

      type(pc_air_multigrid_data), pointer  :: pc_air_data=>null()

      ! ~~~~~~  

      ! Get the PC context
      call PCShellGetContext(pc, pc_air_data, ierr)
      ! Just call the pcmg view
      call PCView(pc_air_data%pcmg, viewer, ierr)

   end subroutine PCView_AIR_Shell    

! -------------------------------------------------------------------------------------------------------------------------------

end module pcair_shell
