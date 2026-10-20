module port_interfaces

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! The Fortran that is still to be ported calls the modules that have been
   ! ported to C (see docs/dev/fortran_removal_plan.md) through this module
   ! Each C routine has a bind(C) interface, named after the routine with a _c suffix,
   ! and a module procedure with the name and arguments of the Fortran routine it
   ! replaces, so the callers only change the module they use
   ! The module procedures convert the arguments (PETSc objects cross as their
   ! handles, arrays with their sizes, 0-based indices become 1-based), check the
   ! error code the C returns and, with PFLARE_PORT_DEBUG, also run the Fortran
   ! reference implementation in src/fref on the same input and abort if the
   ! results differ
   ! This module grows as modules are ported and is deleted once all the callers are in C
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------

   use iso_c_binding
   use iso_fortran_env, only: output_unit
   use petscmat

#include "petsc/finclude/petscmat.h"

   implicit none

   private

   ! Timers.c
   public :: timer_start, timer_finish, timer_reset, timer_clear, timer_time, print_timers

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Timers.c
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function print_timers_c() &
         bind(c, name="print_timers")
         use iso_c_binding
         PetscErrorCode :: print_timers_c
      end function print_timers_c
   end interface

   interface
      function timer_start_c(id) &
         bind(c, name="timer_start")
         use iso_c_binding
         integer(c_int), value :: id
         PetscErrorCode :: timer_start_c
      end function timer_start_c
   end interface

   interface
      function timer_finish_c(id) &
         bind(c, name="timer_finish")
         use iso_c_binding
         integer(c_int), value :: id
         PetscErrorCode :: timer_finish_c
      end function timer_finish_c
   end interface

   interface
      function timer_reset_c() &
         bind(c, name="timer_reset")
         use iso_c_binding
         PetscErrorCode :: timer_reset_c
      end function timer_reset_c
   end interface

   interface
      function timer_clear_c(id) &
         bind(c, name="timer_clear")
         use iso_c_binding
         integer(c_int), value :: id
         PetscErrorCode :: timer_clear_c
      end function timer_clear_c
   end interface

   interface
      function timer_time_c(id, timer_time) &
         bind(c, name="timer_time")
         use iso_c_binding
         integer(c_int), value :: id
         PetscReal :: timer_time
         PetscErrorCode :: timer_time_c
      end function timer_time_c
   end interface

   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine port_check(ierr, routine)

      ! The C routines return a PETSc error code, and PETSc has already printed
      ! the error and traceback if it is not zero. The Fortran callers have no
      ! error code to pass it back through, so abort

      ! ~~~~~~
      PetscErrorCode, intent(in)   :: ierr
      character(len=*), intent(in) :: routine

      integer :: errorcode
      ! ~~~~~~

      if (ierr /= 0) then
         print *, routine, " failed"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

   end subroutine port_check

! -------------------------------------------------------------------------------------------------------------------------------
! Timers.c
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine print_timers()

      ! ~~~~~~
      PetscErrorCode :: ierr
      ! ~~~~~~

      ! The C prints through PetscPrintf, which flushes the C stdout straight away,
      ! so flush what the Fortran has printed first to keep the output in order
      flush(output_unit)
      ierr = print_timers_c()
      call port_check(ierr, "print_timers")

   end subroutine print_timers

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine timer_start(id)

      ! ~~~~~~
      integer, intent(in) :: id

      PetscErrorCode :: ierr
      ! ~~~~~~

      ierr = timer_start_c(id)
      call port_check(ierr, "timer_start")

   end subroutine timer_start

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine timer_finish(id)

      ! ~~~~~~
      integer, intent(in) :: id

      PetscErrorCode :: ierr
      ! ~~~~~~

      ierr = timer_finish_c(id)
      call port_check(ierr, "timer_finish")

   end subroutine timer_finish

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine timer_reset()

      ! ~~~~~~
      PetscErrorCode :: ierr
      ! ~~~~~~

      ierr = timer_reset_c()
      call port_check(ierr, "timer_reset")

   end subroutine timer_reset

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine timer_clear(id)

      ! ~~~~~~
      integer, intent(in) :: id

      PetscErrorCode :: ierr
      ! ~~~~~~

      ierr = timer_clear_c(id)
      call port_check(ierr, "timer_clear")

   end subroutine timer_clear

! -------------------------------------------------------------------------------------------------------------------------------

   PetscReal function timer_time(id)

      ! ~~~~~~
      integer, intent(in) :: id

      PetscErrorCode :: ierr
      ! ~~~~~~

      ierr = timer_time_c(id, timer_time)
      call port_check(ierr, "timer_time")

   end function timer_time

! -------------------------------------------------------------------------------------------------------------------------------

end module port_interfaces
