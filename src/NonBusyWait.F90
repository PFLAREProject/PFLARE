module nonbusywait

   use petscsys
   use iso_c_binding

#include "petsc/finclude/petscsys.h"   

   implicit none

   public

   ! Interface to the sleep routien in C
   interface
      ! int usleep(useconds_t useconds)
      function c_usleep(useconds) bind(c, name='usleep')
         import :: c_int, c_int32_t
         integer(kind=c_int32_t), value :: useconds
         integer(kind=c_int)            :: c_usleep
      end function c_usleep
   end interface     

contains

  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

subroutine non_busy_wait(comm, local_rows)

   ! Non-busy version of a waitall, where ranks sleep
   ! We use this to enable omp on idle ranks, without needing
   ! an mpi library that supports non-busy waits

   ! ~~~~~~~~~~~~~~
   integer, intent(in)  :: comm
   PetscInt, intent(in) :: local_rows

   integer :: sleep_time, comm_size, errorcode, ierr
   integer :: int_send, int_receive
   integer, dimension(:), allocatable :: request
   integer, dimension(:,:), allocatable :: status
   logical :: allreduce_finished
   ! ~~~~~~~~~~~~~~

   call MPI_Comm_size(comm, comm_size, errorcode)

   ! Return if not parallel
   if (comm_size == 1) return

   allocate(request(1))
   request = MPI_REQUEST_NULL
   ! Careful as this status is 2D, we're accessing the old style fortran interface for testall
   allocate(status(MPI_STATUS_SIZE, size(request)))
   ! Start some collective comms (doesn't matter what, here we're just sending 
   ! an int)
   call MPI_IAllreduce(int_send, int_receive, 1, MPI_INTEGER, &
               MPI_MAX, comm, request(1), errorcode)
   if (errorcode /= MPI_SUCCESS) then
      print *, "MPI_IAllreduce failed"
      call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
   end if                

   allreduce_finished = .FALSE.
   ! Sleep for 100 microseconds if the allreduce isn't finished yet
   ! This needs to be big enough that we can do some omp work in that time
   ! but not so big that we end up waiting around for ranks to be done sleeping
   sleep_time = 100
   do while (.NOT. allreduce_finished)
      ! Check if the collective comms is done yet - when it is the status is completed
      call MPI_Testall(size(request), request, allreduce_finished, status, errorcode)

      ! Only sleep on processors that aren't involved in the subcomm
      if (.NOT. allreduce_finished .AND. local_rows == 0) then
         ierr = c_usleep(sleep_time)
      end if
   end do
   deallocate(request, status)   

end subroutine non_busy_wait

! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

end module nonbusywait
