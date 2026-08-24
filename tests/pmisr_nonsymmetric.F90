!
!  Tests the CF splitting on a structurally nonsymmetric strength
!  matrix, by calling compute_cf_splitting with skip_symmetrize true.
!  PCAIR always symmetrizes the strength matrix it builds, so this path
!  is only reachable through compute_cf_splitting.
!
!  There is no explicit check of the splitting here - the point of this
!  test is running under PFLARE_KOKKOS_DEBUG=1 with -mat_type aijkokkos,
!  where the CF splitting runs both the CPU and Kokkos implementations
!  and aborts if they differ.
!
!  The operator is a 1D chain (unit diagonal, -1 in columns i-2 and
!  i+1), so with skip_symmetrize the strength matrix has no diagonal
!  and row i holds entries in columns i-2 and i+1 only, which is
!  structurally nonsymmetric.  Having strong connections in both
!  directions matters in parallel: a node at a rank boundary can then
!  be touched by the halo exchanges and by local updates in the same
!  Luby round, which is what surfaces inconsistent halo handling.  A
!  one-sided chain (all entries below the diagonal) cannot do that, and
!  hides these bugs.
!
      program main

      use petscmat
      use cf_splitting, only: compute_cf_splitting, CF_PMISR_DDC

#include "petsc/finclude/petscmat.h"

      implicit none

      Mat            :: A
      IS             :: is_fine, is_coarse
      PetscInt       :: n, i, rstart, rend
      PetscInt, parameter :: one = 1, two = 2, three = 3
      PetscScalar, parameter :: s_one = 1d0
      PetscReal, parameter :: strong_threshold = 0.5d0, ddc_fraction = 0.1d0
      integer, parameter  :: max_luby_steps = -1, ddc_its = 1
      PetscErrorCode :: ierr
      PetscBool      :: flg

      call PetscInitialize(PETSC_NULL_CHARACTER, ierr)

      n = 700
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-n', n, flg, ierr)

      ! ~~~~~~~~~~~~
      ! Build the operator
      ! MatSetFromOptions means -mat_type aijkokkos is honoured
      ! ~~~~~~~~~~~~
      call MatCreate(PETSC_COMM_WORLD, A, ierr)
      call MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n, ierr)
      call MatSetFromOptions(A, ierr)
      call MatSeqAIJSetPreallocation(A, three, PETSC_NULL_INTEGER_ARRAY, ierr)
      call MatMPIAIJSetPreallocation(A, three, PETSC_NULL_INTEGER_ARRAY, two, PETSC_NULL_INTEGER_ARRAY, ierr)
      call MatSetUp(A, ierr)

      call MatGetOwnershipRange(A, rstart, rend, ierr)
      do i = rstart, rend-1
         call MatSetValue(A, i, i, s_one, INSERT_VALUES, ierr)
         if (i > 1) call MatSetValue(A, i, i-two, -s_one, INSERT_VALUES, ierr)
         if (i < n-1) call MatSetValue(A, i, i+one, -s_one, INSERT_VALUES, ierr)
      end do
      call MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr)

      ! ~~~~~~~~~~~~
      ! The PMISR DDC splitting on the unsymmetrized strength matrix
      ! ~~~~~~~~~~~~
      call compute_cf_splitting(A, .TRUE., &
               strong_threshold, max_luby_steps, &
               CF_PMISR_DDC, ddc_its, ddc_fraction, &
               is_fine, is_coarse)

      call ISDestroy(is_fine, ierr)
      call ISDestroy(is_coarse, ierr)
      call MatDestroy(A, ierr)

      call PetscFinalize(ierr)

      end program main
