!
!  Tests the pmisr CF splitting directly on a structurally nonsymmetric
!  strength matrix.
!
!  PCAIR always symmetrizes the strength matrix it feeds to the first pass
!  CF splitting, so the nonsymmetric input path through pmisr - and in
!  particular the Kokkos kernel pmisr_existing_measure_cf_markers_kokkos -
!  is not reachable through any PCAIR option.  This driver calls pmisr
!  directly to keep that path covered.
!
!  The strength matrix here is the shape a wider 1D upwind stencil
!  produces: an n point chain with unit entries, no diagonal, and row i
!  holding entries in columns i-1 and i-2 only.  That is completely
!  structurally nonsymmetric, and the in-degree of 2 means a corrupted
!  marker in the halo exchange can sum into a wrong sign, which an
!  in-degree 1 chain can hide.
!
!  The assertion that matters is not made here - it is made inside pmisr.
!  Under PFLARE_KOKKOS_DEBUG=1 with -mat_type aijkokkos the wrapper runs
!  both the Kokkos and the CPU implementations and aborts if the resulting
!  cf markers differ.  In parallel the row i -> col i-1 entry of the first
!  local row of every rank but the first crosses the rank boundary, so the
!  MPI halo branch of the kernel is exercised as well as the serial one.
!
!  The checks below are only cheap sanity checks on the host result, and
!  they are only made when the host cf markers are actually valid - the
!  Kokkos pmisr leaves its result on the device and only copies it back to
!  the host when the debug comparison asks for it.
!
!  The same nonsymmetric path is also exercised through the public
!  compute_cf_splitting entry point with skip_symmetrize true.
!
      program main
      use petscmat
      use pmisr_module, only: pmisr
      use cf_splitting, only: compute_cf_splitting, CF_PMIS, CF_PMISR_DDC
      use pflare_parameters, only: C_POINT, F_POINT
      use petsc_helper, only: kokkos_debug
#include "petsc/finclude/petscmat.h"
      implicit none

      Mat            :: S
      PetscInt       :: n, i, rstart, rend
      PetscInt, parameter :: one = 1, two = 2
      PetscScalar, parameter :: s_one = 1d0
      PetscErrorCode :: ierr
      PetscBool      :: flg
      PetscMPIInt    :: rank
      MatType        :: mat_type
      logical        :: host_markers_valid, kokkos_mat

      call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD, rank, ierr)

      n = 1000
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-n', n, flg, ierr)

      ! ~~~~~~~~~~~~
      ! Build the strength matrix directly - unit entries, no diagonal and
      ! row i has entries in columns i-1 and i-2 only (row 0 is empty)
      ! MatSetFromOptions means -mat_type aijkokkos is honoured
      ! ~~~~~~~~~~~~
      call MatCreate(PETSC_COMM_WORLD, S, ierr)
      call MatSetSizes(S, PETSC_DECIDE, PETSC_DECIDE, n, n, ierr)
      call MatSetFromOptions(S, ierr)
      call MatSeqAIJSetPreallocation(S, two, PETSC_NULL_INTEGER_ARRAY, ierr)
      call MatMPIAIJSetPreallocation(S, two, PETSC_NULL_INTEGER_ARRAY, two, PETSC_NULL_INTEGER_ARRAY, ierr)
      call MatSetUp(S, ierr)

      call MatGetOwnershipRange(S, rstart, rend, ierr)
      do i = rstart, rend-1
         ! In parallel these entries are in the off-diagonal block for the
         ! first local rows of every rank but the first
         if (i > 0) call MatSetValue(S, i, i-one, s_one, INSERT_VALUES, ierr)
         if (i > 1) call MatSetValue(S, i, i-two, s_one, INSERT_VALUES, ierr)
      end do
      call MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY, ierr)

      ! ~~~~~~~~~~~~
      ! The Kokkos pmisr leaves the cf markers on the device, so the host
      ! copy is only meaningful if the debug comparison copied it back
      ! ~~~~~~~~~~~~
      kokkos_mat = .FALSE.
      call MatGetType(S, mat_type, ierr)
      if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) kokkos_mat = .TRUE.

      host_markers_valid = .TRUE.
      if (kokkos_mat) host_markers_valid = kokkos_debug()

      ! Negative max_luby_steps runs the Luby loop to completion
      ! PMISR (pmis false) is the variant PCAIR uses
      call do_splitting(S, .FALSE., host_markers_valid, "PMISR")
      call do_splitting(S, .TRUE., host_markers_valid, "PMIS ")

      ! ~~~~~~~~~~~~
      ! With a Kokkos matrix pmisr leaves its result in a device view that
      ! belongs to the library, and a Kokkos view that is still alive when
      ! Kokkos is finalised aborts.  Nothing outside the library can free it
      ! directly, but compute_cf_splitting owns that view for its whole
      ! lifetime - it runs pmisr itself and then releases the device markers
      ! once it has built its ISs - so one splitting here takes over the
      ! markers our direct calls left behind and releases them
      ! It also checks the normal CF splitting entry point still copes with a
      ! matrix like this one
      ! ~~~~~~~~~~~~
      if (kokkos_mat) call release_device_cf_markers(S)

      call MatDestroy(S, ierr)

      ! Also go through the public compute_cf_splitting entry point with
      ! skip_symmetrize true, which reaches pmisr with a nonsymmetric
      ! strength matrix through the public API
      call public_api_skip_symmetrize(n)

      call PetscFinalize(ierr)

      contains

! -------------------------------------------------------------------------------------------------------------------------------

      subroutine do_splitting(strength_mat, pmis, host_valid, label)

         ! Calls pmisr and does some cheap sanity checks on the result

         Mat, intent(in)      :: strength_mat
         logical, intent(in)  :: pmis
         logical, intent(in)  :: host_valid
         character(len=*), intent(in) :: label

         integer, allocatable, dimension(:) :: cf_markers
         integer, parameter :: max_luby_steps = -1
         PetscInt :: local_rows, local_cols
         PetscInt :: n_c, n_f, n_bad
         PetscInt :: n_c_global, n_f_global, n_bad_global
         PetscErrorCode :: ierr_local
         integer :: errorcode
         MPIU_Comm :: comm

         call PetscObjectGetComm(strength_mat, comm, ierr_local)

         ! cf_markers is allocated inside pmisr, so it must come in unallocated
         call pmisr(strength_mat, max_luby_steps, pmis, cf_markers)

         call MatGetLocalSize(strength_mat, local_rows, local_cols, ierr_local)
         if (size(cf_markers) /= local_rows) then
            print *, "pmisr did not return one cf marker per local row"
            error stop 1
         end if

         if (host_valid) then
            n_c = count(cf_markers == C_POINT)
            n_f = count(cf_markers == F_POINT)
            n_bad = local_rows - n_c - n_f

            ! Reduce so every rank makes the same pass/fail decision
            call MPI_Allreduce(n_c, n_c_global, 1, MPIU_INTEGER, MPI_SUM, comm, errorcode)
            call MPI_Allreduce(n_f, n_f_global, 1, MPIU_INTEGER, MPI_SUM, comm, errorcode)
            call MPI_Allreduce(n_bad, n_bad_global, 1, MPIU_INTEGER, MPI_SUM, comm, errorcode)

            if (n_bad_global /= 0) then
               print *, "pmisr returned markers that are neither C nor F points"
               error stop 1
            end if
            ! A chain of this size must produce some of both
            if (n_c_global == 0 .OR. n_f_global == 0) then
               print *, "pmisr returned an empty C or F set"
               error stop 1
            end if

            if (rank == 0) print *, label, " nonsymmetric chain splitting: C points ", &
                     n_c_global, " F points ", n_f_global
         else
            ! The markers are on the device and we have no way of getting at
            ! them from here - pmisr itself has done the checking
            if (rank == 0) print *, label, " nonsymmetric chain splitting: done on the device"
         end if

         deallocate(cf_markers)

      end subroutine do_splitting

! -------------------------------------------------------------------------------------------------------------------------------

      subroutine release_device_cf_markers(strength_mat)

         ! Runs one CF splitting through the normal entry point purely so the
         ! device cf markers left behind by our direct pmisr calls are freed
         ! by the library that allocated them
         ! CF_PMIS is used because it skips the DDC pass, which would want
         ! diagonal dominance ratios of a matrix that has no diagonal

         Mat, intent(in) :: strength_mat

         IS :: is_fine, is_coarse
         PetscReal, parameter :: strong_threshold = 0.5d0, fraction_swap = 0d0
         integer, parameter :: max_luby_steps = -1, ddc_its = 0
         PetscErrorCode :: ierr_local

         ! is_fine and is_coarse are created inside, exactly as PCAIR passes
         ! them in - they must not be set to PETSC_NULL_IS first

         ! Don't skip symmetrizing the strength matrix (though CF_PMIS
         ! always symmetrizes regardless)
         call compute_cf_splitting(strength_mat, .FALSE., &
                  strong_threshold, max_luby_steps, &
                  CF_PMIS, ddc_its, fraction_swap, &
                  is_fine, is_coarse)

         call ISDestroy(is_fine, ierr_local)
         call ISDestroy(is_coarse, ierr_local)

      end subroutine release_device_cf_markers

! -------------------------------------------------------------------------------------------------------------------------------

      subroutine public_api_skip_symmetrize(n_global)

         ! Goes through the public compute_cf_splitting entry point with
         ! skip_symmetrize true on a wide-stencil 1D upwind advection
         ! operator (unit diagonal, -1 at columns i-1 and i-2) - the
         ! strength matrix generated internally is then the same
         ! nonsymmetric chain as the direct pmisr calls above use, so
         ! under the kokkos debug comparison this covers the nonsymmetric
         ! path through the public API - the CF splitting uses the
         ! pmisr_ddc default that PCAIR uses

         PetscInt, intent(in) :: n_global

         Mat :: A
         IS  :: is_fine, is_coarse
         PetscInt :: rstart_l, rend_l, il
         PetscInt, parameter :: three = 3
         PetscReal, parameter :: strong_threshold = 0.5d0, fraction_swap = 0.1d0
         integer, parameter :: max_luby_steps = -1, ddc_its = 1
         PetscErrorCode :: ierr_local

         call MatCreate(PETSC_COMM_WORLD, A, ierr_local)
         call MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n_global, n_global, ierr_local)
         call MatSetFromOptions(A, ierr_local)
         call MatSeqAIJSetPreallocation(A, three, PETSC_NULL_INTEGER_ARRAY, ierr_local)
         call MatMPIAIJSetPreallocation(A, three, PETSC_NULL_INTEGER_ARRAY, two, PETSC_NULL_INTEGER_ARRAY, ierr_local)
         call MatSetUp(A, ierr_local)

         call MatGetOwnershipRange(A, rstart_l, rend_l, ierr_local)
         do il = rstart_l, rend_l-1
            call MatSetValue(A, il, il, s_one, INSERT_VALUES, ierr_local)
            if (il > 0) call MatSetValue(A, il, il-one, -s_one, INSERT_VALUES, ierr_local)
            if (il > 1) call MatSetValue(A, il, il-two, -s_one, INSERT_VALUES, ierr_local)
         end do
         call MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr_local)
         call MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr_local)

         ! skip_symmetrize true - pmisr sees the nonsymmetric strength matrix
         call compute_cf_splitting(A, .TRUE., &
                  strong_threshold, max_luby_steps, &
                  CF_PMISR_DDC, ddc_its, fraction_swap, &
                  is_fine, is_coarse)

         call ISDestroy(is_fine, ierr_local)
         call ISDestroy(is_coarse, ierr_local)
         call MatDestroy(A, ierr_local)

         if (rank == 0) print *, "compute_cf_splitting with skip_symmetrize: done"

      end subroutine public_api_skip_symmetrize

      end program main
