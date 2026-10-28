module gmres_poly_newton

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! The Fortran interface to modified_leja, cluster_eigenvalues_stable and
   ! compute_extra_roots, which are implemented in C in src/Gmres_Poly_Newton.c
   ! with the rest of the Newton basis polynomials (see docs/dev/fortran_removal_plan.md)
   ! They keep their Fortran signatures: the roots are passed to the C by base
   ! address and size, the 0-based indices modified_leja returns are handed back
   ! 1-based in the allocatable indices, and the roots compute_extra_roots allocates
   ! in C are copied into the allocatable outputs
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------

   use iso_c_binding
   use petscsys

#include "petsc/finclude/petscsys.h"

   implicit none

   private
   public :: modified_leja, cluster_eigenvalues_stable, compute_extra_roots

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
      function cluster_eigenvalues_stable_c(real_roots, imag_roots, roots_size, rel_tol, abs_tol) &
         bind(c, name="cluster_eigenvalues_stable")
         use iso_c_binding
         type(c_ptr), value :: real_roots, imag_roots
         PetscInt, value :: roots_size
         PetscReal, value :: rel_tol, abs_tol
         PetscErrorCode :: cluster_eigenvalues_stable_c
      end function cluster_eigenvalues_stable_c
   end interface

   interface
      function compute_extra_roots_c(real_roots, imag_roots, roots_size, &
         real_roots_output, imag_roots_output, roots_output_size) &
         bind(c, name="compute_extra_roots")
         use iso_c_binding
         type(c_ptr), value :: real_roots, imag_roots
         PetscInt, value :: roots_size
         type(c_ptr) :: real_roots_output, imag_roots_output
         PetscInt :: roots_output_size
         PetscErrorCode :: compute_extra_roots_c
      end function compute_extra_roots_c
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

   end subroutine modified_leja

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine cluster_eigenvalues_stable(real_roots, imag_roots, rel_tol, abs_tol)

      ! Clustering of (possibly complex) harmonic Ritz values.
      ! Numerically distinct clusters are moved to the front.
      ! Remaining entries are set to zero.
      ! The roots must be passed in with complex conjugate pairs next to each other
      ! (as dgeev returns them) and every complex cluster is output as a conjugate
      ! pair, with the positive imaginary root first

      ! ~~~~~~
      PetscReal, dimension(:), intent(inout) :: real_roots, imag_roots
      PetscReal, intent(in)                  :: rel_tol, abs_tol

      ! Contiguous copies of the roots, as the dummies may be strided
      PetscReal, dimension(size(real_roots)), target :: real_copy, imag_copy
      PetscInt :: roots_size
      PetscErrorCode :: ierr
      integer :: errorcode
      ! ~~~~~~

      roots_size = size(real_roots)
      real_copy = real_roots
      imag_copy = imag_roots

      ierr = cluster_eigenvalues_stable_c(c_loc(real_copy), c_loc(imag_copy), roots_size, &
               rel_tol, abs_tol)
      if (ierr /= 0) then
         print *, "cluster_eigenvalues_stable failed"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      real_roots = real_copy
      imag_roots = imag_copy

   end subroutine cluster_eigenvalues_stable

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine compute_extra_roots(real_roots, imag_roots, real_roots_output, imag_roots_output)

      ! Add extra roots for stability
      ! Computes the product of factors for each eigenvalue and adds extra copies
      ! of roots that have large products (to improve polynomial stability)
      ! Only non-zero eigenvalues should be passed in
      ! real_roots_output, imag_roots_output are allocated and filled with the original
      ! roots plus any extra

      ! ~~~~~~
      PetscReal, dimension(:), intent(inout)              :: real_roots, imag_roots
      PetscReal, dimension(:), allocatable, intent(inout) :: real_roots_output, imag_roots_output

      ! Contiguous copies of the roots, as the dummies may be strided
      PetscReal, dimension(size(real_roots)), target :: real_copy, imag_copy
      PetscReal, dimension(:), pointer :: real_output_c, imag_output_c
      type(c_ptr) :: real_output_ptr, imag_output_ptr
      PetscInt :: roots_size, roots_output_size
      PetscErrorCode :: ierr
      integer :: errorcode
      ! ~~~~~~

      roots_size = size(real_roots)
      real_copy = real_roots
      imag_copy = imag_roots

      real_output_ptr = c_null_ptr
      imag_output_ptr = c_null_ptr
      roots_output_size = 0
      ierr = compute_extra_roots_c(c_loc(real_copy), c_loc(imag_copy), roots_size, &
               real_output_ptr, imag_output_ptr, roots_output_size)
      if (ierr /= 0) then
         print *, "compute_extra_roots failed"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      if (allocated(real_roots_output)) deallocate(real_roots_output)
      if (allocated(imag_roots_output)) deallocate(imag_roots_output)
      allocate(real_roots_output(roots_output_size))
      allocate(imag_roots_output(roots_output_size))
      if (roots_output_size > 0) then
         call c_f_pointer(real_output_ptr, real_output_c, [roots_output_size])
         call c_f_pointer(imag_output_ptr, imag_output_c, [roots_output_size])
         real_roots_output = real_output_c
         imag_roots_output = imag_output_c
      end if
      ierr = PetscFree_c(real_output_ptr)
      if (ierr == 0) ierr = PetscFree_c(imag_output_ptr)
      if (ierr /= 0) then
         print *, "PetscFree_c failed"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

   end subroutine compute_extra_roots

! -------------------------------------------------------------------------------------------------------------------------------

end module gmres_poly_newton
