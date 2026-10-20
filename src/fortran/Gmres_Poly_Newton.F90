module gmres_poly_newton

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! The Fortran interface to modified_leja, which is implemented in C in
   ! src/Gmres_Poly_Newton.c with the rest of the Newton basis polynomials (see
   ! docs/dev/fortran_removal_plan.md)
   ! modified_leja keeps its Fortran signature: the roots are passed to the C by
   ! base address and size, and the 0-based indices the C returns are handed back
   ! 1-based in the allocatable indices
   ! With PFLARE_PORT_DEBUG it also runs the Fortran reference implementation in
   ! src/fref/Gmres_Poly_Newton.F90 on the same input and aborts if the orderings differ
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------

   use iso_c_binding
   use petscsys
   ! The Fortran reference implementation, under #ifdef rather than #if defined
   ! so that make depend (makedepf90 -D PFLARE_PORT_DEBUG) sees this use only
   ! in the port debug build
#ifdef PFLARE_PORT_DEBUG
   use gmres_poly_newton_fref, only: modified_leja_fref => modified_leja
#endif

#include "petsc/finclude/petscsys.h"

   implicit none

   private
   public :: modified_leja

   interface
      function modified_leja_c(real_roots, imag_roots, roots_size, indices) &
         bind(c, name="modified_leja")
         use iso_c_binding
         type(c_ptr), value :: real_roots, imag_roots
         PetscInt, value :: roots_size
         type(c_ptr) :: indices
         PetscErrorCode :: modified_leja_c
      end function modified_leja_c
   end interface

   interface
      function PetscFree_c(ptr) &
         bind(c, name="PetscFree_c")
         use iso_c_binding
         type(c_ptr) :: ptr
         PetscErrorCode :: PetscFree_c
      end function PetscFree_c
   end interface

   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine modified_leja(real_roots, imag_roots, indices)

      ! Computes a modified leja ordering of the eigenvalues
      ! The roots must be passed in with complex conjugate pairs next to each other
      ! with positive imag e'vals first
      ! indices is allocated here and comes back with the (1-based) ordering

      ! ~~~~~~
      PetscReal, dimension(:), intent(inout)  :: real_roots, imag_roots
      integer, dimension(:), allocatable, intent(inout) :: indices

      ! Contiguous copies of the roots, as the dummies may be strided
      PetscReal, dimension(size(real_roots)), target :: real_copy, imag_copy
      PetscInt, dimension(:), pointer :: indices_c
      type(c_ptr) :: indices_ptr
      PetscInt :: roots_size
      PetscErrorCode :: ierr
      integer :: errorcode
#if defined(PFLARE_PORT_DEBUG)
      integer, dimension(:), allocatable :: indices_fref
      PetscReal, dimension(size(real_roots)) :: real_fref, imag_fref
#endif
      ! ~~~~~~

      roots_size = size(real_roots)
      real_copy = real_roots
      imag_copy = imag_roots

      indices_ptr = c_null_ptr
      ierr = modified_leja_c(c_loc(real_copy), c_loc(imag_copy), roots_size, indices_ptr)
      if (ierr /= 0) then
         print *, "modified_leja failed"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      if (allocated(indices)) deallocate(indices)
      allocate(indices(roots_size))
      if (roots_size > 0) then
         call c_f_pointer(indices_ptr, indices_c, [roots_size])
         indices = int(indices_c) + 1
      end if
      ierr = PetscFree_c(indices_ptr)
      if (ierr /= 0) then
         print *, "PetscFree_c failed"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

#if defined(PFLARE_PORT_DEBUG)
      real_fref = real_roots
      imag_fref = imag_roots
      call modified_leja_fref(real_fref, imag_fref, indices_fref)
      if (size(indices_fref) /= size(indices)) then
         print *, "C and Fortran reference versions of modified_leja do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      if (any(indices_fref /= indices)) then
         print *, "C and Fortran reference versions of modified_leja do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      deallocate(indices_fref)
#endif

   end subroutine modified_leja

! -------------------------------------------------------------------------------------------------------------------------------

end module gmres_poly_newton
