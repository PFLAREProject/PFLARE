! TEMPORARY (stage 9 of docs/dev/fortran_removal_plan.md): the retired Fortran
! generator, renamed so tests/random_compare.c can check src/Pflare_Random.c
! against it bitwise. Deleted once the check has passed

module pflare_random

   ! The random numbers used by PFLARE (the PMISR measure, the DDC tie-breaking,
   ! the CR initial error and the GMRES polynomial start vector) all come from
   ! the Fortran intrinsic random_number, seeded the same way on every call.
   ! They are generated through this one routine so that C code can draw exactly
   ! the same sequence as the Fortran does.

   use iso_c_binding
   use petscsys

#include "petsc/finclude/petscsys.h"

   implicit none

   public

   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine pflare_random_fill(comm_rank, n, out) bind(C, name='pflare_random_fill_fortran')

      ! Seeds the intrinsic random number generator with a seed that depends on
      ! comm_rank (so each rank gets different random numbers, but repeated runs
      ! get the same ones) and then fills out(1:n) with uniform random numbers
      ! in 0 <= u < 1

      ! ~~~~~~
      integer(c_int), value, intent(in)     :: comm_rank
      PetscInt, value, intent(in)           :: n
      PetscReal, dimension(n), intent(out)  :: out

      integer :: seed_size, i_loc
      integer, dimension(:), allocatable :: seed
      ! ~~~~~~

      call random_seed(size=seed_size)
      allocate(seed(seed_size))
      ! Ensure we seed the same so subsequent runs get the same random
      ! Seed different on each process so we don't have the same random numbers on each processor
      do i_loc = 1, seed_size
         seed(i_loc) = comm_rank + 1 + i_loc
      end do
      call random_seed(put=seed)
      ! Fill with random numbers
      call random_number(out)
      deallocate(seed)

   end subroutine pflare_random_fill

! -------------------------------------------------------------------------------------------------------------------------------

end module pflare_random
