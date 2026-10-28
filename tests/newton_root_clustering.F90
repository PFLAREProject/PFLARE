!
!  Tests the clustering of the Newton basis roots by calling
!  cluster_eigenvalues_stable directly, then passing the result through
!  compute_extra_roots and modified_leja as the Newton setup does.
!
!  Every complex root output by the clustering must be followed by its
!  conjugate, with the positive imaginary part first. The constructed inputs
!  have a complex pair (1, +-y) next to the real root 1, with y within the
!  clustering tolerance of the real root but 2y not. Clustering the complex
!  root with the real root but not its conjugate splits the pair. The
!  tolerances are relative to sqrt(epsilon), so this triggers in any precision.
!
      program main

      use petscsys
      use gmres_poly_newton, only: cluster_eigenvalues_stable, compute_extra_roots, &
            modified_leja

#include "petsc/finclude/petscsys.h"

      implicit none

      PetscErrorCode :: ierr
      PetscReal :: rel_tol, abs_tol, y
      PetscReal, dimension(3) :: real_roots, imag_roots
      PetscReal, dimension(6) :: real_six, imag_six
      integer :: errors

      call PetscInitialize(PETSC_NULL_CHARACTER, ierr)

      ! The tolerances used by calculate_gmres_polynomial_roots_newton, with ||H|| = 1
      rel_tol = sqrt(epsilon(rel_tol))
      abs_tol = epsilon(abs_tol)
      errors = 0

      ! The pair and the real root form one cluster, whose centroid is within
      ! tolerance of its own conjugate, so it should be a single real root
      y = 0.7 * rel_tol
      real_roots = [1, 1, 1]
      imag_roots = [y, -y, real(0, kind=PETSC_REAL_KIND)]
      call check_clustering(real_roots, imag_roots, 1, errors)

      ! The pair and the real root form one cluster, whose centroid is not within
      ! tolerance of its conjugate, so it should be a single conjugate pair
      y = 0.9 * rel_tol
      real_roots = [1, 1, 1]
      imag_roots = [y, -y, real(0, kind=PETSC_REAL_KIND)]
      call check_clustering(real_roots, imag_roots, 2, errors)

      ! The same with the real root first
      real_roots = [1, 1, 1]
      imag_roots = [real(0, kind=PETSC_REAL_KIND), y, -y]
      call check_clustering(real_roots, imag_roots, 2, errors)

      ! Well separated roots, with an exactly zero root and a pair close to the
      ! real root 1 as above, should give the same roots in the same order
      y = 0.9 * rel_tol
      real_six = [real(2, kind=PETSC_REAL_KIND), real(2, kind=PETSC_REAL_KIND), &
            real(0, kind=PETSC_REAL_KIND), real(1, kind=PETSC_REAL_KIND), &
            real(1, kind=PETSC_REAL_KIND), real(1, kind=PETSC_REAL_KIND)]
      imag_six = [real(1, kind=PETSC_REAL_KIND), real(-1, kind=PETSC_REAL_KIND), &
            real(0, kind=PETSC_REAL_KIND), y, -y, real(0, kind=PETSC_REAL_KIND)]
      call check_clustering(real_six, imag_six, 4, errors)
      if (any(real_six(1:2) /= 2) .OR. imag_six(1) /= 1 .OR. imag_six(2) /= -1) then
         print *, "Well separated pair changed by the clustering", real_six(1:2), imag_six(1:2)
         errors = errors + 1
      end if

      if (errors /= 0) then
         print *, "Newton root clustering test failed with", errors, "errors"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, ierr)
      end if

      call PetscFinalize(ierr)

      contains

      subroutine check_clustering(real_in, imag_in, expected_unique, errors)

         ! Clusters real_in/imag_in in place and checks the number of non-zero roots,
         ! that the non-zero roots are at the front with conjugate pairs adjacent,
         ! and that the extra roots and the Leja ordering keep the pairs together

         PetscReal, dimension(:), intent(inout) :: real_in, imag_in
         integer, intent(in)                    :: expected_unique
         integer, intent(inout)                 :: errors

         PetscReal, dimension(:), allocatable :: real_added, imag_added
         integer, dimension(:), allocatable   :: indices
         integer :: n_unique, i_loc

         call cluster_eigenvalues_stable(real_in, imag_in, rel_tol, abs_tol)

         n_unique = count(real_in /= 0 .OR. imag_in /= 0)
         if (n_unique /= expected_unique) then
            print *, "Clustering gave", n_unique, "roots, expected", expected_unique
            errors = errors + 1
         end if
         if (any(real_in(n_unique+1:) /= 0 .OR. imag_in(n_unique+1:) /= 0)) then
            print *, "Zero roots are not at the end after clustering"
            errors = errors + 1
         end if
         if (.NOT. pairs_adjacent(real_in(1:n_unique), imag_in(1:n_unique), .TRUE.)) then
            print *, "Complex conjugate pair split by the clustering"
            do i_loc = 1, size(real_in)
               print *, i_loc, real_in(i_loc), imag_in(i_loc)
            end do
            errors = errors + 1
            return
         end if

         ! Add the extra roots and Leja order, as the Newton setup does
         call compute_extra_roots(real_in(1:n_unique), imag_in(1:n_unique), &
               real_added, imag_added)
         if (.NOT. pairs_adjacent(real_added, imag_added, .TRUE.)) then
            print *, "Complex conjugate pair split by the extra roots"
            errors = errors + 1
         end if
         call modified_leja(real_added, imag_added, indices)
         do i_loc = 1, size(indices)
            if (count(indices == i_loc) /= 1) then
               print *, "Leja order is not a permutation", indices
               errors = errors + 1
               exit
            end if
         end do
         if (.NOT. pairs_adjacent(real_added(indices), imag_added(indices), .FALSE.)) then
            print *, "Complex conjugate pair split in Leja order", indices
            errors = errors + 1
         end if

         deallocate(real_added, imag_added, indices)

      end subroutine check_clustering

      logical function pairs_adjacent(real_in, imag_in, positive_first)

         ! Every complex root must be next to its conjugate, with the positive
         ! imaginary root first if positive_first (the Leja order can put either first)

         PetscReal, dimension(:), intent(in) :: real_in, imag_in
         logical, intent(in)                 :: positive_first
         integer :: i_loc

         pairs_adjacent = .TRUE.
         i_loc = 1
         do while (i_loc <= size(real_in))
            if (imag_in(i_loc) /= 0) then
               if (i_loc == size(real_in) .OR. &
                     (positive_first .AND. imag_in(i_loc) < 0)) then
                  pairs_adjacent = .FALSE.
                  return
               end if
               if (real_in(i_loc+1) /= real_in(i_loc) .OR. &
                     imag_in(i_loc+1) /= -imag_in(i_loc)) then
                  pairs_adjacent = .FALSE.
                  return
               end if
               i_loc = i_loc + 2
            else
               i_loc = i_loc + 1
            end if
         end do

      end function pairs_adjacent

      end program main
