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

   interface 
      function PCPFLAREINVGetPolyCoeffs_mine(A_array, coeffs_ptr, rows, cols) &
         bind(c, name="PCPFLAREINVGetPolyCoeffs")
         use iso_c_binding
#include "finclude/pflare_types.h"
#include "finclude/PETSc_ISO_Types.h"
         integer(c_long_long), value               :: A_array
         type(c_ptr), intent(out)                  :: coeffs_ptr
         integer(PFLARE_PETSCINT_C_KIND), intent(out) :: rows
         integer(PFLARE_PETSCINT_C_KIND), intent(out) :: cols
         integer(PFLARE_PETSCERRORCODE_C_KIND)     :: PCPFLAREINVGetPolyCoeffs_mine
      end function PCPFLAREINVGetPolyCoeffs_mine
   end interface

   interface 
      function PCPFLAREINVGetReusePolyCoeffs_mine(A_array, b) &
         bind(c, name="PCPFLAREINVGetReusePolyCoeffs")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: A_array
         integer(c_int), intent(out)            :: b
         integer(PFLARE_PETSCERRORCODE_C_KIND)  :: PCPFLAREINVGetReusePolyCoeffs_mine
      end function PCPFLAREINVGetReusePolyCoeffs_mine
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

   interface 
      function PCPFLAREINVSetPolyCoeffs_mine(A_array, coeffs_ptr, rows, cols) &
         bind(c, name="PCPFLAREINVSetPolyCoeffs")
         use iso_c_binding
#include "finclude/pflare_types.h"
#include "finclude/PETSc_ISO_Types.h"
         integer(c_long_long), value               :: A_array
         type(c_ptr), value                        :: coeffs_ptr
         integer(PFLARE_PETSCINT_C_KIND), value    :: rows
         integer(PFLARE_PETSCINT_C_KIND), value    :: cols
         integer(PFLARE_PETSCERRORCODE_C_KIND)     :: PCPFLAREINVSetPolyCoeffs_mine
      end function PCPFLAREINVSetPolyCoeffs_mine
   end interface

   interface 
      function PCPFLAREINVSetReusePolyCoeffs_mine(A_array, b) &
         bind(c, name="PCPFLAREINVSetReusePolyCoeffs")
         use iso_c_binding
#include "finclude/pflare_types.h"
         integer(c_long_long), value            :: A_array
         integer(c_int), value                  :: b
         integer(PFLARE_PETSCERRORCODE_C_KIND)  :: PCPFLAREINVSetReusePolyCoeffs_mine
      end function PCPFLAREINVSetReusePolyCoeffs_mine
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

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCPFLAREINVGetPolyCoeffs(pc, coeffs, ierr)

      ! This routine returns a copy of the polynomial coefficients stored in the PC.
      ! coeffs is allocated/reallocated as needed; it is safe to keep and use after
      ! the next PCSetUp or PCReset call (unlike the C interface, which returns a
      ! raw pointer into internal storage that is only valid until the next setup).
      ! coeffs(:,1) = real roots (or power/arnoldi coefficients)
      ! coeffs(:,2) = imaginary roots (Newton basis only)

#include "finclude/PETSc_ISO_Types.h"

      ! ~~~~~~~~~~
      type(tPC), intent(in)                                :: pc
      PetscReal, dimension(:,:), pointer, intent(inout)   :: coeffs
      PetscErrorCode, intent(inout)                       :: ierr

      integer(c_long_long) :: pc_ptr
      type(c_ptr) :: coeffs_c_ptr
      integer(PFLARE_PETSCINT_C_KIND) :: rows, cols
      real(PFLARE_PETSCREAL_C_KIND), pointer :: coeffs_f(:,:)
      ! ~~~~~~~~~~

      pc_ptr = pc%v
      ierr = PCPFLAREINVGetPolyCoeffs_mine(pc_ptr, coeffs_c_ptr, rows, cols)

      if (.NOT. c_associated(coeffs_c_ptr)) then
         print *, "PCPFLAREINVGetPolyCoeffs: no coefficients available; call PCSetUp first"
         error stop 1
      end if

      ! (Re-)allocate output array if size has changed
      if (associated(coeffs)) then
         if (.NOT. (size(coeffs,1) == int(rows) .AND. size(coeffs,2) == int(cols))) then
            deallocate(coeffs)
            coeffs => null()
         end if
      end if
      if (.NOT. associated(coeffs)) then
         allocate(coeffs(rows, cols))
      end if

      ! Create a temporary Fortran view of the C-owned buffer and copy
      call c_f_pointer(coeffs_c_ptr, coeffs_f, [rows, cols])
      coeffs = coeffs_f

   end subroutine PCPFLAREINVGetPolyCoeffs

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCPFLAREINVGetReusePolyCoeffs(pc, flag, ierr)

      ! ~~~~~~~~~~
      type(tPC), intent(in)         :: pc
      PetscBool, intent(inout)      :: flag
      PetscErrorCode, intent(inout) :: ierr

      integer(c_long_long) :: pc_ptr
      integer :: flag_int
      ! ~~~~~~~~~~

      pc_ptr   = pc%v
      flag_int = 0
      ierr     = PCPFLAREINVGetReusePolyCoeffs_mine(pc_ptr, flag_int)
      flag     = PETSC_FALSE
      if (flag_int /= 0) flag = PETSC_TRUE

   end subroutine PCPFLAREINVGetReusePolyCoeffs

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

   subroutine PCPFLAREINVSetPolyCoeffs(pc, coeffs, ierr)

      ! This routine copies coeffs into the PC's internal storage.
      ! The caller's array is not referenced after this call and can be
      ! deallocated or modified freely.
      ! Does not trigger a rebuild; after setting these coefficients call
      ! PCPFLAREINVSetReusePolyCoeffs(pc, PETSC_TRUE) and then KSPSolve
      ! to apply them.

#include "finclude/PETSc_ISO_Types.h"

      ! ~~~~~~~~~~
      type(tPC), intent(in)                                :: pc
      PetscReal, dimension(:,:), pointer, intent(in)      :: coeffs
      PetscErrorCode, intent(inout)                       :: ierr

      integer(c_long_long) :: pc_ptr
      integer(PFLARE_PETSCINT_C_KIND) :: rows, cols
      ! ~~~~~~~~~~

      pc_ptr = pc%v
      rows   = int(size(coeffs, 1), PFLARE_PETSCINT_C_KIND)
      cols   = int(size(coeffs, 2), PFLARE_PETSCINT_C_KIND)
      ierr   = PCPFLAREINVSetPolyCoeffs_mine(pc_ptr, c_loc(coeffs(1,1)), rows, cols)

   end subroutine PCPFLAREINVSetPolyCoeffs

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCPFLAREINVSetReusePolyCoeffs(pc, flag, ierr)

      ! ~~~~~~~~~~
      type(tPC), intent(in)         :: pc
      PetscBool, intent(in)         :: flag
      PetscErrorCode, intent(inout) :: ierr

      integer(c_long_long) :: pc_ptr
      integer :: flag_int
      ! ~~~~~~~~~~

      pc_ptr   = pc%v
      flag_int = 0
      if (flag) flag_int = 1
      ierr = PCPFLAREINVSetReusePolyCoeffs_mine(pc_ptr, flag_int)

   end subroutine PCPFLAREINVSetReusePolyCoeffs

! -------------------------------------------------------------------------------------------------------------------------------

end module pcpflareinv_interfaces

