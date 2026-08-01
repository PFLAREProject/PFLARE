module cr_splitting

   use petscmat
   use petsc_helper, only: MatCreateSubMatrixWrapper
   use pmisr_module, only: pmisr_existing_measure_implicit_transpose
   use approx_inverse_setup, only: calculate_and_build_approximate_inverse
   use pflare_parameters, only: C_POINT, F_POINT, PFLARE_CR_NU, &
            PFLARE_CR_NU_POLY, PFLARE_CR_POLY_ORDER, &
            PFLAREINV_WJACOBI, PFLAREINV_JACOBI, &
            PFLARE_CR_CANDIDATE, PFLARE_MINUS_ONE, PFLARE_ONE, PFLARE_REAL_KIND

#include "petsc/finclude/petscmat.h"

   implicit none

   public

   ! The CR relaxation whose error decides the splitting is the approximate
   ! inverse of Aff built with the same inverse type, polynomial order, sparsity
   ! order and diagonal scaling that PCAIR uses for its Aff inverse, so the CR
   ! rate certifies the one-application contraction of the F-solve actually
   ! applied in the hierarchy
   ! It is ALWAYS built assembled, never matrix-free, even when
   ! -pc_air_matrix_free_polys is on: the ideal restrictor is always built from
   ! the assembled sparsified inverse (see AIR_Operators_Setup), which is
   ! weaker than the matrix-free polynomial, so the assembled inverse is the
   ! binding constraint on the splitting - certifying with the matrix-free
   ! polynomial under-coarsens and the restrictor quality degrades with
   ! problem size, while certifying with the assembled inverse guarantees the
   ! restrictor and the matrix-free smoother simply does better than predicted

   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine cr_pass(input_mat, is_fine, target_cr_rate, &
                  cr_inverse_type, cr_poly_order, cr_sparsity_order, cr_diag_scale, &
                  cr_rate_achieved, n_swapped_global, cf_markers_local)

      ! One pass of compatible relaxation (CR) coarsening - the CR splitting is
      ! built entirely from repeated calls to this, starting from all points F
      ! Runs nu applications of the CR relaxation on Aff with a zero rhs and a random
      ! initial error (habituated CR - the C points are held at zero so the relaxation
      ! never touches the whole matrix), then promotes an independent set of the
      ! slowest converging F rows to C
      ! The relaxation is the assembled approximate inverse with the settings
      ! given by cr_inverse_type/cr_poly_order/cr_sparsity_order/cr_diag_scale
      ! (mirroring AIRG's F-point smoothing) - see the module comment above for
      ! why it is always assembled. Jacobi types and tiny Aff blocks (where an
      ! Arnoldi of the polynomial order doesn't fit) use the inline Jacobi path,
      ! which sanitizes zero diagonals
      ! Candidates for promotion are the rows where the relaxed error remains
      ! large relative to the biggest remaining error (the hypre CR candidate
      ! measure), so rows where the relaxation diverges or stalls are promoted first
      ! cr_rate_achieved returns the estimated CR rate rho = (||e_nu||/||e_0||)^(1/nu)
      ! and the caller iterates until rho <= target_cr_rate or nothing is swapped

      ! ~~~~~~
      type(tMat), target, intent(in)      :: input_mat
      type(tIS), intent(in)               :: is_fine
      PetscReal, intent(in)               :: target_cr_rate
      integer, intent(in)                 :: cr_inverse_type, cr_poly_order, cr_sparsity_order
      logical, intent(in)                 :: cr_diag_scale
      PetscReal, intent(out)              :: cr_rate_achieved
      PetscInt, intent(out)               :: n_swapped_global
      integer, dimension(:), allocatable, target, intent(inout) :: cf_markers_local

      ! Local
      type(tMat) :: Aff, temp_mat, inv_Aff
      type(tVec) :: diag_vec, scale_vec, e_vec, work_vec, work2_vec
      PetscInt :: fine_local, fine_global, global_rows, global_cols
      PetscInt :: input_row_start, input_row_end_plus_one, idx, ifree
      PetscInt :: n_swapped_local
      PetscInt, dimension(:), pointer :: is_pointer
      PetscScalar, dimension(:), pointer :: vec_array
      PetscScalar, dimension(:), pointer :: e_array
      PetscReal, dimension(:), allocatable :: cr_measure
      PetscReal, dimension(:), allocatable :: rand_full
      integer, dimension(:), allocatable :: cf_markers_local_aff
      logical, dimension(:), allocatable :: forced_c
      logical :: extracted
      PetscReal :: norm_inf, weight, norm_e0, norm_enu, e_norm_inf
      integer :: seed_size, comm_rank, errorcode, i_loc, sweep, max_luby_steps, nu
      logical :: use_poly
      integer, dimension(:), allocatable :: seed
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX

      ! ~~~~~~

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)
      call MPI_Comm_rank(MPI_COMM_MATRIX, comm_rank, errorcode)

      cr_rate_achieved = 0
      n_swapped_global = 0

      call ISGetLocalSize(is_fine, fine_local, ierr)
      call ISGetSize(is_fine, fine_global, ierr)

      ! Nothing to relax on - every point is C
      ! This is on the global size so all ranks agree, everything below is collective
      if (fine_global == 0) return

      call MatGetOwnershipRange(input_mat, input_row_start, input_row_end_plus_one, ierr)
      call MatGetSize(input_mat, global_rows, global_cols, ierr)
      call ISGetIndices(is_fine, is_pointer, ierr)

      ! While every point is still F (the initial rounds, as CR starts with an
      ! empty C set) Aff is just the input matrix so we can skip the extraction
      extracted = fine_global /= global_rows
      if (extracted) then
         call MatCreateSubMatrixWrapper(input_mat, is_fine, is_fine, &
                  MAT_INITIAL_MATRIX, Aff)
      else
         Aff = input_mat
      end if

      allocate(forced_c(fine_local))
      forced_c = .FALSE.

      ! Jacobi inverse types use the inline path below (which sanitizes zero
      ! diagonals), as does any Aff too tiny to fit an Arnoldi of the
      ! polynomial order
      use_poly = cr_inverse_type /= PFLAREINV_WJACOBI .AND. &
                 cr_inverse_type /= PFLAREINV_JACOBI .AND. &
                 fine_global > cr_poly_order + 1
      nu = PFLARE_CR_NU
      if (use_poly) nu = PFLARE_CR_NU_POLY

      if (use_poly) then

      ! ~~~~~~~~
      ! Assembled approximate inverse of Aff with the same settings as the
      ! F-point smoothing AIRG applies in the hierarchy, so the CR rate
      ! certifies the contraction of the actual F-solve including the
      ! sparsity truncation
      ! Always assembled, never matrix-free - see the module comment
      ! ~~~~~~~~
      call calculate_and_build_approximate_inverse(Aff, cr_inverse_type, &
               cr_poly_order, cr_sparsity_order, &
               .FALSE., cr_diag_scale, .FALSE., &
               inv_Aff)

      else

      ! ~~~~~~~~
      ! Build the weighted Jacobi inverse of the diagonal of Aff
      ! This uses the same weight as calculate_and_build_weighted_jacobi_inverse
      ! (ie the hypre weight) but we sanitize exactly zero diagonals so they can't
      ! poison the weight with infs/nans - those rows can never be relaxed so they
      ! are force promoted to C below
      ! ~~~~~~~~
      call MatCreateVecs(Aff, PETSC_NULL_VEC, diag_vec, ierr)
      call MatGetDiagonal(Aff, diag_vec, ierr)
      call VecGetArray(diag_vec, vec_array, ierr)
      do ifree = 1, fine_local
         if (vec_array(ifree) == 0.0) then
            vec_array(ifree) = 1.0
            forced_c(ifree) = .TRUE.
         end if
      end do
      call VecRestoreArray(diag_vec, vec_array, ierr)

      ! 3 / ( 4 * || D^(-1/2) * Aff * D^(-1/2) ||_inf )
      ! Unweighted plain Jacobi if that's what the hierarchy smooths with
      weight = 1.0
      if (cr_inverse_type /= PFLAREINV_JACOBI) then
         call MatDuplicate(Aff, MAT_COPY_VALUES, temp_mat, ierr)
         call VecDuplicate(diag_vec, scale_vec, ierr)
         call VecCopy(diag_vec, scale_vec, ierr)
         call VecSqrtAbs(scale_vec, ierr)
         call VecReciprocal(scale_vec, ierr)
         call MatDiagonalScale(temp_mat, scale_vec, scale_vec, ierr)
         call MatNorm(temp_mat, NORM_INFINITY, norm_inf, ierr)
         call MatDestroy(temp_mat, ierr)
         call VecDestroy(scale_vec, ierr)
         if (norm_inf /= 0.0) weight = 3.0/(4.0 * norm_inf)
      end if

      ! Invert and apply the weight, zeroing the rows we can't relax
      call VecReciprocal(diag_vec, ierr)
      call VecScale(diag_vec, weight, ierr)
      if (any(forced_c)) then
         call VecGetArray(diag_vec, vec_array, ierr)
         do ifree = 1, fine_local
            if (forced_c(ifree)) vec_array(ifree) = 0.0
         end do
         call VecRestoreArray(diag_vec, vec_array, ierr)
      end if

      ! The matrix takes ownership of diag_vec
      call MatCreateDiagonal(diag_vec, inv_Aff, ierr)
      call VecDestroy(diag_vec, ierr)

      end if

      ! ~~~~~~~~
      ! Random initial error, generated on the host over all the local rows of A
      ! with the same seeding convention as the pmisr/ddc, so any future device
      ! port of this routine is bit-identical
      ! ~~~~~~~~
      allocate(rand_full(input_row_end_plus_one - input_row_start))
      call random_seed(size=seed_size)
      allocate(seed(seed_size))
      do i_loc = 1, seed_size
         seed(i_loc) = comm_rank + 1 + i_loc
      end do
      call random_seed(put=seed)
      call random_number(rand_full)
      deallocate(seed)

      call MatCreateVecs(Aff, e_vec, work_vec, ierr)
      call VecDuplicate(e_vec, work2_vec, ierr)

      ! Scatter the random error into the F rows
      call VecGetArray(e_vec, e_array, ierr)
      do ifree = 1, fine_local
         e_array(ifree) = rand_full(is_pointer(ifree) - input_row_start + 1)
      end do
      call VecRestoreArray(e_vec, e_array, ierr)

      call VecNorm(e_vec, NORM_2, norm_e0, ierr)

      ! ~~~~~~~~
      ! The relaxation - nu applications of e <- e - M Aff e, where M is the
      ! approximate inverse of Aff built above
      ! All default petsc mat/vec ops so this runs on the device for kokkos types
      ! ~~~~~~~~
      do sweep = 1, nu
         call MatMult(Aff, e_vec, work_vec, ierr)
         call MatMult(inv_Aff, work_vec, work2_vec, ierr)
         call VecAXPY(e_vec, PFLARE_MINUS_ONE, work2_vec, ierr)
      end do

      call VecNorm(e_vec, NORM_2, norm_enu, ierr)
      if (norm_e0 /= 0.0) then
         cr_rate_achieved = (norm_enu/norm_e0)**(PFLARE_ONE/real(nu, kind=PFLARE_REAL_KIND))
      end if

      ! Only promote F points if we haven't already hit the target rate
      ! The rate is computed from global norms so all ranks take the same branch
      if (cr_rate_achieved > target_cr_rate) then

         ! ~~~~~~~~
         ! Candidate F rows are those where the relaxed error is still big
         ! relative to the biggest remaining error anywhere - the hypre CR
         ! candidate measure. Wherever error persists the relaxation is failing,
         ! and promoting the biggest entries cuts the slowly converging modes -
         ! a per-row convergence factor can look fine on every row while a
         ! global slow mode remains, which stalls the coarsening
         ! ~~~~~~~~
         call VecNorm(e_vec, NORM_INFINITY, e_norm_inf, ierr)

         allocate(cr_measure(fine_local))
         allocate(cf_markers_local_aff(fine_local))
         cf_markers_local_aff = 0

         call VecGetArrayRead(e_vec, e_array, ierr)
         do ifree = 1, fine_local
            if (forced_c(ifree)) then
               ! Rows we couldn't relax (zero diagonal) always get promoted first
               cr_measure(ifree) = 2.0
            else if (e_norm_inf /= 0.0) then
               cr_measure(ifree) = abs(e_array(ifree))/e_norm_inf
            else
               cr_measure(ifree) = 0.0
            end if
         end do
         call VecRestoreArrayRead(e_vec, e_array, ierr)

         ! ~~~~~~~~
         ! Same trick as the ddc trigger path - the pmisr tags the points with the
         ! smallest measure as "F", so feed in 10 - candidate measure and it picks
         ! an independent set of the biggest remaining errors
         ! The base of 10 keeps abs(measure) .ge. 1, which the pmisr requires
         ! (the candidate measure is in [0, 2]), and the scaled down random
         ! breaks ties without changing the ordering very much
         ! There is always at least one candidate when the rate is above the
         ! target (the row holding the biggest error), so every pass promotes
         ! something and the outer loop can't stall short of the target
         ! ~~~~~~~~
         do ifree = 1, fine_local
            if (cr_measure(ifree) < PFLARE_CR_CANDIDATE) then
               ! Error already small here - never swap these
               cr_measure(ifree) = PETSC_MAX_REAL
               cf_markers_local_aff(ifree) = C_POINT
            else
               cr_measure(ifree) = real(10d0 - (cr_measure(ifree) - &
                     rand_full(is_pointer(ifree) - input_row_start + 1)/1d10), &
                     kind=PFLARE_REAL_KIND)
            end if
         end do

         ! Call PMISR with as many steps as necessary
         ! Uses the implicit transpose version which takes Aff directly
         ! and handles Aff+Aff^T internally without forming the explicit sum
         max_luby_steps = -1
         call pmisr_existing_measure_implicit_transpose(Aff, &
                  max_luby_steps, .FALSE., &
                  cr_measure, cf_markers_local_aff)

         ! Swap the slowest converging rows to C points
         n_swapped_local = 0
         do ifree = 1, fine_local

            ! The pmisr marked the points we want to swap as F
            if (cf_markers_local_aff(ifree) == F_POINT) then
               ! This is the actual numbering in A, rather than Aff
               ! Careful here to minus away the row_start of A, not Aff
               ! as cf_markers_local is as big as A
               idx = is_pointer(ifree) - input_row_start + 1

               ! Swap by multiplying by -1
               cf_markers_local(idx) = cf_markers_local(idx) * (-1)
               n_swapped_local = n_swapped_local + 1
            end if
         end do

         ! The caller needs to know if anything changed globally to keep all
         ! ranks consistent in the outer loop
         call MPI_Allreduce(n_swapped_local, n_swapped_global, 1, MPIU_INTEGER, &
                  MPI_SUM, MPI_COMM_MATRIX, errorcode)

         deallocate(cr_measure, cf_markers_local_aff)
      end if

      ! Cleanup
      call VecDestroy(e_vec, ierr)
      call VecDestroy(work_vec, ierr)
      call VecDestroy(work2_vec, ierr)
      call MatDestroy(inv_Aff, ierr)
      if (extracted) call MatDestroy(Aff, ierr)
      call ISRestoreIndices(is_fine, is_pointer, ierr)
      deallocate(rand_full, forced_c)

   end subroutine cr_pass

! -------------------------------------------------------------------------------------------------------------------------------

end module cr_splitting
