!
!  Tests the modified Leja ordering used by the Newton basis GMRES polynomials
!  by calling modified_leja directly.
!
!  A Leja ordering depends only on ratios of products of distances between
!  the roots, so scaling every root by the same constant must not change it.
!  This checks that on a tightly clustered real spectrum (where the product of
!  distances is small) and on a spectrum with a complex conjugate pair, across
!  a range of scalings, and also checks the known Leja order of the real spectrum.
!
      program main

      use petscsys
      use gmres_poly_newton, only: modified_leja

#include "petsc/finclude/petscsys.h"

      implicit none

      PetscErrorCode :: ierr
      integer, parameter :: n_scales = 5
      PetscReal, dimension(n_scales), parameter :: scales = &
            [real(1d-3, kind=PETSC_REAL_KIND), real(1d-2, kind=PETSC_REAL_KIND), &
             real(1d0, kind=PETSC_REAL_KIND), real(1d2, kind=PETSC_REAL_KIND), &
             real(1d3, kind=PETSC_REAL_KIND)]
      ! Tightly clustered real spectrum
      PetscReal, dimension(5), parameter :: real_one = &
            [real(0.1d0, kind=PETSC_REAL_KIND), real(0.12d0, kind=PETSC_REAL_KIND), &
             real(0.15d0, kind=PETSC_REAL_KIND), real(0.2d0, kind=PETSC_REAL_KIND), &
             real(0.3d0, kind=PETSC_REAL_KIND)]
      PetscReal, dimension(5), parameter :: imag_one = 0
      ! The Leja order of real_one: start at the biggest (0.3), then the furthest
      ! from it (0.1), then maximise the product of distances each time (0.2, 0.15, 0.12)
      integer, dimension(5), parameter :: expected_one = [5, 1, 4, 3, 2]
      ! Spectrum with a complex conjugate pair (positive imaginary part first)
      PetscReal, dimension(6), parameter :: real_two = &
            [real(0.1d0, kind=PETSC_REAL_KIND), real(0.13d0, kind=PETSC_REAL_KIND), &
             real(0.13d0, kind=PETSC_REAL_KIND), real(0.16d0, kind=PETSC_REAL_KIND), &
             real(0.2d0, kind=PETSC_REAL_KIND), real(0.25d0, kind=PETSC_REAL_KIND)]
      PetscReal, dimension(6), parameter :: imag_two = &
            [real(0d0, kind=PETSC_REAL_KIND), real(0.02d0, kind=PETSC_REAL_KIND), &
             real(-0.02d0, kind=PETSC_REAL_KIND), real(0d0, kind=PETSC_REAL_KIND), &
             real(0d0, kind=PETSC_REAL_KIND), real(0d0, kind=PETSC_REAL_KIND)]
      integer :: errors

      call PetscInitialize(PETSC_NULL_CHARACTER, ierr)

      errors = 0
      call check_spectrum(real_one, imag_one, expected_one, errors)
      call check_spectrum(real_two, imag_two, [integer ::], errors)

      if (errors /= 0) then
         print *, "Leja ordering test failed with", errors, "errors"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, ierr)
      end if

      call PetscFinalize(ierr)

      contains

      subroutine check_spectrum(real_in, imag_in, expected, errors)

         ! Leja orders real_in/imag_in under each scaling and checks the orderings
         ! are identical, match expected (if given) and keep conjugate pairs together

         PetscReal, dimension(:), intent(in) :: real_in, imag_in
         integer, dimension(:), intent(in)   :: expected
         integer, intent(inout)              :: errors

         PetscReal, dimension(size(real_in)) :: real_roots, imag_roots
         integer, dimension(:), allocatable  :: indices
         integer, dimension(size(real_in))   :: reference
         integer :: i_scale, i_loc

         do i_scale = 1, n_scales

            real_roots = scales(i_scale) * real_in
            imag_roots = scales(i_scale) * imag_in
            call modified_leja(real_roots, imag_roots, indices)

            if (i_scale == 1) then
               reference = indices
               if (size(expected) > 0) then
                  if (any(indices /= expected)) then
                     print *, "Leja order", indices, "does not match expected", expected
                     errors = errors + 1
                  end if
               end if
            else if (any(indices /= reference)) then
               print *, "Leja order", indices, "with scaling", scales(i_scale), &
                     "differs from", reference, "with scaling", scales(1)
               errors = errors + 1
            end if

            ! Complex conjugates must be next to each other in the ordering
            ! (the positive imaginary root is at k and its conjugate at k+1 on input)
            do i_loc = 1, size(indices)
               if (imag_in(i_loc) > 0) then
                  if (abs(findloc(indices, i_loc, dim=1) - findloc(indices, i_loc+1, dim=1)) /= 1) then
                     print *, "Complex conjugate pair split in Leja order", indices
                     errors = errors + 1
                  end if
               end if
            end do

            deallocate(indices)
         end do

      end subroutine check_spectrum

      end program main
