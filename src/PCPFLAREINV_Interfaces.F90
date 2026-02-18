module pcpflareinv_interfaces

   use iso_c_binding
   use petscksp

#include "petsc/finclude/petscksp.h"
#include "finclude/pflare_types.h"
#include "finclude/PETSc_ISO_Types.h"

   implicit none

   public

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! Contains Fortran interfaces to the PCPFLAREINV options 
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------      

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Get routines - interfaces to the C routines in PCPFLAREINV
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface 
      function PCPFLAREINVGetPolyOrder_mine(A_array,b) &
         bind(c, name="PCPFLAREINVGetPolyOrder")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: A_array
         integer(PFLARE_PETSCINT_C_KIND)        :: b
         integer(PFLARE_PETSCERRORCODE_C_KIND)  :: PCPFLAREINVGetPolyOrder_mine
      end function PCPFLAREINVGetPolyOrder_mine 
   end interface
   
   interface 
      function PCPFLAREINVGetSparsityOrder_mine(A_array,b) &
         bind(c, name="PCPFLAREINVGetSparsityOrder")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: A_array
         integer(PFLARE_PETSCINT_C_KIND)        :: b
         integer(PFLARE_PETSCERRORCODE_C_KIND)  :: PCPFLAREINVGetSparsityOrder_mine
      end function PCPFLAREINVGetSparsityOrder_mine 
   end interface    

   interface 
      function PCPFLAREINVGetType_mine(A_array,b) &
         bind(c, name="PCPFLAREINVGetType")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: A_array
         integer(c_int)                         :: b
         integer(PFLARE_PETSCERRORCODE_C_KIND)  :: PCPFLAREINVGetType_mine
      end function PCPFLAREINVGetType_mine 
   end interface

   interface 
      function PCPFLAREINVGetMatrixFree_mine(A_array,b) &
         bind(c, name="PCPFLAREINVGetMatrixFree")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: A_array
         integer(c_int)                         :: b
         integer(PFLARE_PETSCERRORCODE_C_KIND)  :: PCPFLAREINVGetMatrixFree_mine
      end function PCPFLAREINVGetMatrixFree_mine 
   end interface    
   
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Set routines - interfaces to the C routines in PCPFLAREINV
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface 
      function PCPFLAREINVSetPolyOrder_mine(A_array,b) &
         bind(c, name="PCPFLAREINVSetPolyOrder")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: A_array
         integer(PFLARE_PETSCINT_C_KIND), value :: b
         integer(PFLARE_PETSCERRORCODE_C_KIND)  :: PCPFLAREINVSetPolyOrder_mine
      end function PCPFLAREINVSetPolyOrder_mine 
   end interface
   
   interface 
      function PCPFLAREINVSetSparsityOrder_mine(A_array,b) &
         bind(c, name="PCPFLAREINVSetSparsityOrder")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: A_array
         integer(PFLARE_PETSCINT_C_KIND), value :: b
         integer(PFLARE_PETSCERRORCODE_C_KIND)  :: PCPFLAREINVSetSparsityOrder_mine
      end function PCPFLAREINVSetSparsityOrder_mine 
   end interface    

   interface 
      function PCPFLAREINVSetType_mine(A_array,b) &
         bind(c, name="PCPFLAREINVSetType")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: A_array
         integer(c_int), value                  :: b
         integer(PFLARE_PETSCERRORCODE_C_KIND)  :: PCPFLAREINVSetType_mine
      end function PCPFLAREINVSetType_mine 
   end interface

   interface 
      function PCPFLAREINVSetMatrixFree_mine(A_array,b) &
         bind(c, name="PCPFLAREINVSetMatrixFree")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: A_array
         integer(c_int), value                  :: b
         integer(PFLARE_PETSCERRORCODE_C_KIND)  :: PCPFLAREINVSetMatrixFree_mine
      end function PCPFLAREINVSetMatrixFree_mine 
   end interface  

   ! ~~~~~~~~~~~~~~~~

   contains

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Get routines - Fortran versions of the C routines in PCPFLAREINV
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCPFLAREINVGetPolyOrder(pc, order, ierr)

      ! ~~~~~~~~~~
      type(tPC), intent(in)         :: pc
      PetscInt, intent(inout)       :: order
      PetscErrorCode, intent(inout) :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~~~

      pc_ptr= pc%v
      ierr = PCPFLAREINVGetPolyOrder_mine(pc_ptr, order)

   end subroutine PCPFLAREINVGetPolyOrder    

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCPFLAREINVGetSparsityOrder(pc, order, ierr)

      ! ~~~~~~~~~~
      type(tPC), intent(in)         :: pc
      PetscInt, intent(inout)       :: order
      PetscErrorCode, intent(inout) :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~~~

      pc_ptr= pc%v
      ierr = PCPFLAREINVGetSparsityOrder_mine(pc_ptr, order)

   end subroutine PCPFLAREINVGetSparsityOrder     

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCPFLAREINVGetType(pc, pflare_type, ierr)

      ! ~~~~~~~~~~
      type(tPC), intent(in)            :: pc
      PCPFLAREINVType, intent(inout)   :: pflare_type
      PetscErrorCode, intent(inout)    :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~~~

      pc_ptr= pc%v
      ierr = PCPFLAREINVGetType_mine(pc_ptr, pflare_type)

   end subroutine PCPFLAREINVGetType

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCPFLAREINVGetMatrixFree(pc, flag, ierr)

      ! ~~~~~~~~~~
      type(tPC), intent(in)         :: pc
      PetscBool, intent(inout)      :: flag
      PetscErrorCode, intent(inout) :: ierr

      integer(c_long_long) :: pc_ptr
      integer :: flag_int
      ! ~~~~~~~~~~

      pc_ptr= pc%v
      flag_int = 0
      if (flag) flag_int = 1
      ierr = PCPFLAREINVGetMatrixFree_mine(pc_ptr, flag_int)

   end subroutine PCPFLAREINVGetMatrixFree   

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Set routines - Fortran versions of the C routines in PCPFLAREINV
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~    

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCPFLAREINVSetPolyOrder(pc, order, ierr)

      ! ~~~~~~~~~~
      type(tPC), intent(in)         :: pc
      PetscInt, intent(in)          :: order
      PetscErrorCode, intent(inout) :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~~~

      pc_ptr= pc%v
      ierr = PCPFLAREINVSetPolyOrder_mine(pc_ptr, order)

   end subroutine PCPFLAREINVSetPolyOrder    

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCPFLAREINVSetSparsityOrder(pc, order, ierr)

      ! ~~~~~~~~~~
      type(tPC), intent(in)         :: pc
      PetscInt, intent(in)          :: order
      PetscErrorCode, intent(inout) :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~~~

      pc_ptr= pc%v
      ierr = PCPFLAREINVSetSparsityOrder_mine(pc_ptr, order)

   end subroutine PCPFLAREINVSetSparsityOrder     

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCPFLAREINVSetType(pc, pflare_type, ierr)

      ! ~~~~~~~~~~
      type(tPC), intent(in)         :: pc
      PCPFLAREINVType, intent(in)   :: pflare_type
      PetscErrorCode, intent(inout) :: ierr

      integer(c_long_long) :: pc_ptr
      ! ~~~~~~~~~~

      pc_ptr= pc%v
      ierr = PCPFLAREINVSetType_mine(pc_ptr, pflare_type)

   end subroutine PCPFLAREINVSetType

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCPFLAREINVSetMatrixFree(pc, flag, ierr)

      ! ~~~~~~~~~~
      type(tPC), intent(in)         :: pc
      PetscBool, intent(in)         :: flag
      PetscErrorCode, intent(inout) :: ierr

      integer(c_long_long) :: pc_ptr
      integer :: flag_int
      ! ~~~~~~~~~~

      pc_ptr= pc%v
      flag_int = 0
      if (flag) flag_int = 1
      ierr = PCPFLAREINVSetMatrixFree_mine(pc_ptr, flag_int)

   end subroutine PCPFLAREINVSetMatrixFree    

! -------------------------------------------------------------------------------------------------------------------------------

end module pcpflareinv_interfaces

