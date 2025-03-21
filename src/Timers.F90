module timers

   ! Just need the mpi types 
   ! and it is difficult to pick either the old school 
   ! include mpif.h or the newer use mpi_f08   
   use petscsys
   use iso_c_binding, only: c_double

#include "petsc/finclude/petscsys.h"   

   implicit none

   ! Timer IDs - Numbers must be between 1 and 1024
   integer, parameter      :: TIMER_ID_AIR_SETUP = 1, &
                              TIMER_ID_AIR_INVERSE = 2, &
                              TIMER_ID_AIR_DROP = 3, &
                              TIMER_ID_AIR_RAP = 4, &
                              TIMER_ID_AIR_EXTRACT = 5, &
                              TIMER_ID_AIR_PROLONG = 6, &
                              TIMER_ID_AIR_RESTRICT = 7, &
                              TIMER_ID_AIR_PROC_AGGLOM = 8, &
                              TIMER_ID_AIR_COARSEN = 9, &
                              TIMER_ID_AIR_CONSTRAIN = 10, &
                              TIMER_ID_AIR_IDENTITY = 11, &
                              TIMER_ID_AIR_TRUNCATE = 12

   public :: timer_start, timer_finish, timer_reset, timer_clear

      ! -------------------------------------------------------------------------------------------------------------------------------
      ! -------------------------------------------------------------------------------------------------------------------------------
      ! Stores a globally accessible array of times along with some ids
      ! Basically a copy of fluidity's tictoc.F90 and Timers.F90
      ! Makes it easy to time things across different routines
      ! -------------------------------------------------------------------------------------------------------------------------------
      ! -------------------------------------------------------------------------------------------------------------------------------  

   integer, parameter :: MAX_TIMER_ID = 1024
   PetscReal :: starttimers(MAX_TIMER_ID) = 0d0, totaltimers(MAX_TIMER_ID) = 0d0

contains

subroutine print_timers()

   ! Print the values of the timers

   print *,  "coarsen time     :", timer_time(TIMER_ID_AIR_COARSEN)
   print *,  "extract time     :", timer_time(TIMER_ID_AIR_EXTRACT)
   print *,  "proc agglom time :", timer_time(TIMER_ID_AIR_PROC_AGGLOM)       
   print *,  "inverse time     :", timer_time(TIMER_ID_AIR_INVERSE)            
   print *,  "restrict time    :", timer_time(TIMER_ID_AIR_RESTRICT)
   print *,  "prolong time     :", timer_time(TIMER_ID_AIR_PROLONG)
   print *,  "constrain time   :", timer_time(TIMER_ID_AIR_CONSTRAIN)       
   print *,  "rap time         :", timer_time(TIMER_ID_AIR_RAP)
   print *,  "identity time    :", timer_time(TIMER_ID_AIR_IDENTITY)
   print *,  "drop time        :", timer_time(TIMER_ID_AIR_DROP)
   print *,  "truncate time    :", timer_time(TIMER_ID_AIR_TRUNCATE)

end subroutine print_timers

function wall_time()

   ! This function returns the wall clock time from when the
   ! simulation started.

   real(kind = c_double):: wall_time
   logical, save :: started=.false.
   real(kind = c_double), save :: wall_time0

   wall_time = MPI_Wtime()
   if(.not.started) then
      wall_time0 = wall_time
      wall_time = 0d0
      started=.true.
   else
      wall_time = wall_time - wall_time0
   endif
 end function wall_time

 ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  subroutine timer_start(id)
    integer, intent(in) :: id

    starttimers(id) = wall_time()

  end subroutine timer_start

  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  subroutine timer_finish(id)
    integer, intent(in) :: id

    PetscReal :: finish_time

    finish_time = wall_time()
    totaltimers(id) = totaltimers(id) + (finish_time - starttimers(id))

  end subroutine timer_finish

  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  subroutine timer_reset()
    starttimers = 0d0
    totaltimers = 0d0
  end subroutine timer_reset

  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  subroutine timer_clear(id)
    integer, intent(in) :: id

    starttimers(id) = 0d0
    totaltimers(id) = 0d0

  end subroutine timer_clear

  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  PetscReal function timer_time(id)
    integer, intent(in) :: id

    timer_time = totaltimers(id)

  end function timer_time

  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

end module timers
