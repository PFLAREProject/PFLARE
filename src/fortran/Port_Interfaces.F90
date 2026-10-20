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
   ! The Fortran reference implementations, under #ifdef rather than #if defined
   ! so that make depend (makedepf90 -D PFLARE_PORT_DEBUG) sees these uses only
   ! in the port debug build
#ifdef PFLARE_PORT_DEBUG
   use sorting, only: intersect_pre_sorted_indices_only_fref => intersect_pre_sorted_indices_only, &
         merge_pre_sorted_fref => merge_pre_sorted
   use aggregation, only: generate_serial_aggregation_fref => generate_serial_aggregation
#endif

#include "petsc/finclude/petscmat.h"

   implicit none

   private

   ! Timers.c
   public :: timer_start, timer_finish, timer_reset, timer_clear, timer_time, print_timers
   ! Sorting.c
   public :: sorted_unique, intersect_pre_sorted_indices_only, merge_pre_sorted
   ! Aggregation.c
   public :: generate_serial_aggregation

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! C_PETSc_Routines.c
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function PetscFree_c(ptr) &
         bind(c, name="PetscFree_c")
         use iso_c_binding
         type(c_ptr) :: ptr
         PetscErrorCode :: PetscFree_c
      end function PetscFree_c
   end interface

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

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Sorting.c
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function sorted_unique_c(array, n) &
         bind(c, name="sorted_unique")
         use iso_c_binding
         PetscInt :: n
         PetscInt, dimension(n) :: array
         PetscErrorCode :: sorted_unique_c
      end function sorted_unique_c
   end interface

   interface
      function intersect_pre_sorted_indices_only_c(array1, size_array1, array2, size_array2, &
                  indices_array1, indices_array2, intersect_count) &
         bind(c, name="intersect_pre_sorted_indices_only")
         use iso_c_binding
         PetscInt, value :: size_array1, size_array2
         PetscInt, dimension(size_array1) :: array1
         PetscInt, dimension(size_array2) :: array2
         ! Sized by the caller, at least min(size_array1, size_array2)
         integer(c_int), dimension(*) :: indices_array1, indices_array2
         integer(c_int) :: intersect_count
         PetscErrorCode :: intersect_pre_sorted_indices_only_c
      end function intersect_pre_sorted_indices_only_c
   end interface

   interface
      function merge_pre_sorted_c(array1, size_array1, array2, size_array2, sorted_array) &
         bind(c, name="merge_pre_sorted")
         use iso_c_binding
         PetscInt, value :: size_array1, size_array2
         PetscInt, dimension(size_array1) :: array1
         PetscInt, dimension(size_array2) :: array2
         PetscInt, dimension(size_array1 + size_array2) :: sorted_array
         PetscErrorCode :: merge_pre_sorted_c
      end function merge_pre_sorted_c
   end interface

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Aggregation.c
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function generate_serial_aggregation_c(strength_mat, cf_markers, aggregates) &
         bind(c, name="generate_serial_aggregation")
         use iso_c_binding
         integer(c_long_long), value :: strength_mat
         type(c_ptr) :: cf_markers, aggregates
         PetscErrorCode :: generate_serial_aggregation_c
      end function generate_serial_aggregation_c
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
! Sorting.c
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine sorted_unique(array, n)

      ! Sorts array(1:n) and removes the duplicates, in place
      ! On return n is the number of unique entries, which are in array(1:n) in ascending order
      ! The port debug comparison against the binary tree this replaces is at the call site

      ! ~~~~~~
      PetscInt, dimension(:), intent(inout) :: array
      PetscInt, intent(inout)               :: n

      PetscErrorCode :: ierr
      ! ~~~~~~

      ierr = sorted_unique_c(array, n)
      call port_check(ierr, "sorted_unique")

   end subroutine sorted_unique

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine intersect_pre_sorted_indices_only(array1, array2, indices_array1, indices_array2, intersect_count)

      ! ~~~~~~
      PetscInt, dimension(:), intent(in) :: array1, array2
      integer, dimension(:), intent(out)  :: indices_array1, indices_array2
      integer, intent(out) :: intersect_count

      PetscInt :: size_array1, size_array2
      PetscErrorCode :: ierr
#if defined(PFLARE_PORT_DEBUG)
      integer, dimension(size(indices_array1)) :: indices_array1_fref
      integer, dimension(size(indices_array2)) :: indices_array2_fref
      integer :: intersect_count_fref, errorcode
#endif
      ! ~~~~~~

      size_array1 = size(array1)
      size_array2 = size(array2)
      ierr = intersect_pre_sorted_indices_only_c(array1, size_array1, array2, size_array2, &
                  indices_array1, indices_array2, intersect_count)
      call port_check(ierr, "intersect_pre_sorted_indices_only")
      ! The C returns 0-based indices
      indices_array1(1:intersect_count) = indices_array1(1:intersect_count) + 1
      indices_array2(1:intersect_count) = indices_array2(1:intersect_count) + 1

#if defined(PFLARE_PORT_DEBUG)
      call intersect_pre_sorted_indices_only_fref(array1, array2, indices_array1_fref, indices_array2_fref, &
                  intersect_count_fref)
      if (intersect_count /= intersect_count_fref) then
         print *, "C and Fortran reference versions of intersect_pre_sorted_indices_only do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      if (any(indices_array1(1:intersect_count) /= indices_array1_fref(1:intersect_count)) .OR. &
          any(indices_array2(1:intersect_count) /= indices_array2_fref(1:intersect_count))) then
         print *, "C and Fortran reference versions of intersect_pre_sorted_indices_only do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
#endif

   end subroutine intersect_pre_sorted_indices_only

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine merge_pre_sorted(array1, array2, sorted_array)

      ! ~~~~~~
      PetscInt, dimension(:), intent(in) :: array1, array2
      PetscInt, dimension(:), intent(out)  :: sorted_array

      PetscInt :: size_array1, size_array2
      PetscErrorCode :: ierr
#if defined(PFLARE_PORT_DEBUG)
      PetscInt, dimension(size(sorted_array)) :: sorted_array_fref
      integer :: errorcode
#endif
      ! ~~~~~~

      size_array1 = size(array1)
      size_array2 = size(array2)
      ierr = merge_pre_sorted_c(array1, size_array1, array2, size_array2, sorted_array)
      call port_check(ierr, "merge_pre_sorted")

#if defined(PFLARE_PORT_DEBUG)
      call merge_pre_sorted_fref(array1, array2, sorted_array_fref)
      if (any(sorted_array /= sorted_array_fref)) then
         print *, "C and Fortran reference versions of merge_pre_sorted do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
#endif

   end subroutine merge_pre_sorted

! -------------------------------------------------------------------------------------------------------------------------------
! Aggregation.c
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine generate_serial_aggregation(strength_mat, cf_markers, aggregates)

      ! The C takes cf_markers as input if it is allocated, and otherwise allocates it,
      ! and always allocates aggregates (with PetscMalloc). The arrays the C allocates
      ! are copied into the Fortran allocatables and freed

      ! ~~~~~~
      type(tMat), target, intent(in)      :: strength_mat
      integer, dimension(:), allocatable, target, intent(inout) :: cf_markers
      PetscInt, dimension(:), allocatable, intent(inout) :: aggregates

      PetscInt :: local_rows, local_cols
      PetscErrorCode :: ierr
      logical :: cf_markers_allocated
      type(c_ptr) :: cf_markers_ptr, aggregates_ptr
      integer, dimension(:), pointer :: cf_markers_c
      PetscInt, dimension(:), pointer :: aggregates_c
#if defined(PFLARE_PORT_DEBUG)
      integer, dimension(:), allocatable :: cf_markers_fref
      PetscInt, dimension(:), allocatable :: aggregates_fref
      integer :: errorcode
#endif
      ! ~~~~~~

      call MatGetLocalSize(strength_mat, local_rows, local_cols, ierr)

#if defined(PFLARE_PORT_DEBUG)
      ! The C modifies cf_markers in place, so give the reference a copy of the input
      if (allocated(cf_markers)) then
         allocate(cf_markers_fref(size(cf_markers)))
         cf_markers_fref = cf_markers
      end if
      call generate_serial_aggregation_fref(strength_mat, cf_markers_fref, aggregates_fref)
#endif

      cf_markers_allocated = allocated(cf_markers)
      if (cf_markers_allocated) then
         cf_markers_ptr = c_loc(cf_markers)
      else
         cf_markers_ptr = c_null_ptr
      end if
      aggregates_ptr = c_null_ptr

      ierr = generate_serial_aggregation_c(strength_mat%v, cf_markers_ptr, aggregates_ptr)
      call port_check(ierr, "generate_serial_aggregation")

      ! Copy what the C allocated into the Fortran allocatables
      if (.NOT. cf_markers_allocated) then
         allocate(cf_markers(local_rows))
         if (local_rows > 0) then
            call c_f_pointer(cf_markers_ptr, cf_markers_c, [local_rows])
            cf_markers = cf_markers_c
         end if
         ierr = PetscFree_c(cf_markers_ptr)
         call port_check(ierr, "PetscFree_c")
      end if
      allocate(aggregates(local_rows))
      if (local_rows > 0) then
         call c_f_pointer(aggregates_ptr, aggregates_c, [local_rows])
         aggregates = aggregates_c
      end if
      ierr = PetscFree_c(aggregates_ptr)
      call port_check(ierr, "PetscFree_c")

#if defined(PFLARE_PORT_DEBUG)
      if (size(cf_markers) /= size(cf_markers_fref) .OR. size(aggregates) /= size(aggregates_fref)) then
         print *, "C and Fortran reference versions of generate_serial_aggregation do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      if (any(cf_markers /= cf_markers_fref) .OR. any(aggregates /= aggregates_fref)) then
         print *, "C and Fortran reference versions of generate_serial_aggregation do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      deallocate(cf_markers_fref, aggregates_fref)
#endif

   end subroutine generate_serial_aggregation

! -------------------------------------------------------------------------------------------------------------------------------

end module port_interfaces
