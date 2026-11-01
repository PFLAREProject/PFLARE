module air_diagonal_update

   use petscksp
   use air_data_type, only: air_multigrid_data, REUSE_MAT_ACTIVE
   use gmres_poly_data_type, only: gmres_poly_data
   use pflare_parameters, only: MAT_INV_AFF, MAT_INV_ACC, PFLARE_ONE, PFLARE_MINUS_ONE, PFLARE_HALF, &
         TIMER_ID_AIR_SETUP, TIMER_ID_AIR_INVERSE, TIMER_ID_AIR_EXTRACT, TIMER_ID_AIR_RESTRICT, &
         TIMER_ID_AIR_RAP
   use timers, only: timer_start, timer_finish, timer_time, print_timers
   use approx_inverse_setup, only: start_approximate_inverse, finish_approximate_inverse, &
         destroy_matrix_reuse
   use fc_smooth, only: VecISCopyLocalWrapper

#include "petsc/finclude/petscksp.h"

   implicit none
   public

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! Applies a diagonal update to an existing AIR hierarchy
   ! Given the hierarchy built from A and a diagonal matrix D, this modifies the hierarchy
   ! so that it approximates the one built from A + D, without needing A + D or any
   ! matrix-matrix products
   ! The CF splittings, the grid-transfer operators and the off-diagonal blocks are frozen,
   ! only the diagonals of A_ff, A_cc and the coarsest matrix change on each level,
   ! along with the approximate inverses built from them
   ! The diagonal perturbation is carried down to the coarse grids by row sum lumping 
   ! R diag(d) P, and the F point columns of the restrictors are rescaled to account for 
   ! the change in A_ff, though only ever shrunk
   ! The update is always relative to the hierarchy built from A (ie it is not cumulative),
   ! so applying D = 0 gives back the original hierarchy
   ! -------------------------------------------------------------------------------------------------------------------------------

   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine update_air_diag(air_data, pmat)

      ! Updates the hierarchy in air_data with the change in the diagonal of pmat 
      ! since the hierarchy was built
      ! The only vectors we keep between updates are the baseline diagonals
      ! The F and C point scratch on each level are the smoothing temporaries, and the 
      ! vectors with the layout of each level are subvectors of one pool that only 
      ! exists during the update

      ! ~~~~~~
      type(air_multigrid_data), intent(inout) :: air_data
      type(tMat), intent(in)                  :: pmat

      integer :: our_level, no_levels, errorcode, comm_rank
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX
      type(tVec) :: pool, temp_vec, d_vec, d_coarse, work, f_new, f_old, c_vec
      type(tIS) :: d_is, d_coarse_is, work_is
      logical :: update_acc
      ! ~~~~~~

      no_levels = air_data%no_levels

      ! ~~~~~~~~~~~~
      ! Check we can apply an update to this hierarchy
      ! ~~~~~~~~~~~~
      ! With one level we either precondition with jacobi on the top grid matrix
      ! itself (which we don't own) or an approximate inverse built from it
      if (no_levels < 2) then
         print *, "A diagonal update requires a hierarchy with at least two levels"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      ! When smoothing all the unknowns the smoother is built from the full level matrices
      ! which we don't keep
      if (air_data%options%full_smoothing_up_and_down) then
         print *, "A diagonal update is not supported with full smoothing up and down"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      call timer_start(TIMER_ID_AIR_SETUP)

      ! The first time through we capture the diagonals of the hierarchy we were given
      if (.NOT. air_data%diag_update_initialised) call setup_diag_update_data(air_data)

      ! The pool holds the perturbation on two levels at once and a work vector
      call create_pool(air_data, pool)

      ! The perturbation on the top grid
      call get_level_vec(air_data, pool, 1, 1, d_is, d_vec)
      call MatGetDiagonal(pmat, d_vec, ierr)
      call VecAXPY(d_vec, PFLARE_MINUS_ONE, air_data%diag_update_base_pmat, ierr)

      ! ~~~~~~~~~~~~
      ! Go down the hierarchy
      ! ~~~~~~~~~~~~
      do our_level = 1, no_levels - 1

         ! The perturbations on this level and the next alternate between the first
         ! two parts of the pool
         call get_level_vec(air_data, pool, our_level + 1, mod(our_level, 2) + 1, d_coarse_is, d_coarse)
         call get_level_vec(air_data, pool, our_level, 3, work_is, work)
         ! F point scratch
         f_new = air_data%temp_vecs_fine(1)%array(our_level)
         f_old = air_data%temp_vecs_fine(2)%array(our_level)

         ! ~~~~~~~~~~~~
         ! Update the diagonal of A_ff
         ! ~~~~~~~~~~~~
         call timer_start(TIMER_ID_AIR_EXTRACT)
         ! The diagonal of A_ff from the last update, which we need to rescale R
         call MatGetDiagonal(air_data%A_ff(our_level), f_old, ierr)
         ! f_new = baseline + d_f
         call VecISCopyLocalWrapper(air_data, our_level, .TRUE., d_vec, SCATTER_REVERSE, f_new)
         call VecAYPX(f_new, PFLARE_ONE, air_data%diag_update_base_aff(our_level), ierr)
         call set_diagonal(air_data%A_ff(our_level), f_new)
         call timer_finish(TIMER_ID_AIR_EXTRACT)

         ! ~~~~~~~~~~~~
         ! Rescale the F point columns of the restrictor
         ! ~~~~~~~~~~~~
         ! Z = -A_cf A_ff^-1, so if A_ff were diagonal the updated Z would just be
         ! the original with columns scaled by a_ff / (a_ff + d_f)
         ! Without this the frozen Z is far too big when d is large, and with only F point 
         ! smoothing the error that introduces at the C points is never corrected
         ! We only ever shrink Z though, so the scaling is min(1, a_ff / (a_ff + d_f))
         ! When the diagonal decreases, if we don't clip the iteration count grows substantially
         ! If we're symmetric we don't have R (it is P^T) and scaling it would also scale P
         if (.NOT. air_data%options%symmetric) then

            call timer_start(TIMER_ID_AIR_RESTRICT)
            ! The scaling a_ff / (a_ff + d_f) clipped at one
            call VecPointwiseDivide(f_new, air_data%diag_update_base_aff(our_level), f_new, ierr)
            call clip_at_one(f_new, air_data%temp_vecs_fine(3)%array(our_level), &
                     air_data%temp_vecs_fine(4)%array(our_level))
            ! R has already been scaled by the clipped a_ff / (a_ff + d_f_old), so scale by 
            ! the ratio of the new scaling to that
            call VecPointwiseDivide(f_old, air_data%diag_update_base_aff(our_level), f_old, ierr)
            call clip_at_one(f_old, air_data%temp_vecs_fine(3)%array(our_level), &
                     air_data%temp_vecs_fine(4)%array(our_level))
            call VecPointwiseDivide(f_old, f_new, f_old, ierr)
            ! The scaling on all the points on this level, one on the C points
            call VecSet(work, PFLARE_ONE, ierr)
            ! On gpus without kokkos this just goes through the host
            if (air_data%fast_veciscopy_exists) then
               call VecISCopyLocalWrapper(air_data, our_level, .TRUE., work, SCATTER_FORWARD, f_old)
            else
               call VecISCopy(work, air_data%IS_fine_index(our_level), SCATTER_FORWARD, f_old, ierr)
            end if
            call MatDiagonalScale(air_data%restrictors(our_level), PETSC_NULL_VEC, work, ierr)
            call timer_finish(TIMER_ID_AIR_RESTRICT)
         end if

         ! ~~~~~~~~~~~~
         ! Rebuild the approximate inverse of A_ff we smooth with
         ! ~~~~~~~~~~~~
         call rebuild_approximate_inverse(air_data, air_data%A_ff(our_level), &
                  air_data%inv_A_ff_poly_data(our_level), &
                  air_data%options%matrix_free_polys, air_data%options%diag_scale_polys, &
                  our_level, MAT_INV_AFF, air_data%inv_A_ff(our_level))

         ! ~~~~~~~~~~~~
         ! Update A_cc and its inverse if we're C smoothing
         ! ~~~~~~~~~~~~
         update_acc = .FALSE.
         if (allocated(air_data%diag_update_base_acc)) then
            temp_vec = air_data%diag_update_base_acc(our_level)
            update_acc = .NOT. PetscObjectIsNull(temp_vec)
         end if
         if (update_acc) then
            call timer_start(TIMER_ID_AIR_EXTRACT)
            c_vec = air_data%temp_vecs_coarse(1)%array(our_level)
            call VecISCopyLocalWrapper(air_data, our_level, .FALSE., d_vec, SCATTER_REVERSE, c_vec)
            call VecAYPX(c_vec, PFLARE_ONE, air_data%diag_update_base_acc(our_level), ierr)
            call set_diagonal(air_data%A_cc(our_level), c_vec)
            call timer_finish(TIMER_ID_AIR_EXTRACT)

            call rebuild_approximate_inverse(air_data, air_data%A_cc(our_level), &
                     air_data%inv_A_cc_poly_data(our_level), &
                     air_data%options%matrix_free_polys, air_data%options%diag_scale_polys, &
                     our_level, MAT_INV_ACC, air_data%inv_A_cc(our_level))
         end if

         ! ~~~~~~~~~~~~
         ! Compute the perturbation on the coarse grid
         ! ~~~~~~~~~~~~
         ! Row sum lumping of R diag(d) P, ie R (d .* P 1), which keeps the action 
         ! of R diag(d) P on the constant
         ! P 1 is one matvec with P, so we just recompute it rather than store it
         call timer_start(TIMER_ID_AIR_RAP)
         call VecSet(d_coarse, PFLARE_ONE, ierr)
         call MatMult(air_data%prolongators(our_level), d_coarse, work, ierr)
         call VecPointwiseMult(work, work, d_vec, ierr)
         ! If we're symmetric the restrictor is P^T
         if (air_data%options%symmetric) then
            call MatMultTranspose(air_data%prolongators(our_level), work, d_coarse, ierr)
         else
            call MatMult(air_data%restrictors(our_level), work, d_coarse, ierr)
         end if
         call timer_finish(TIMER_ID_AIR_RAP)

         call VecRestoreSubVector(pool, work_is, work, ierr)
         call ISDestroy(work_is, ierr)
         call VecRestoreSubVector(pool, d_is, d_vec, ierr)
         call ISDestroy(d_is, ierr)
         ! The coarse perturbation becomes the perturbation on the next level
         d_vec = d_coarse
         d_is = d_coarse_is
      end do

      ! ~~~~~~~~~~~~
      ! Update the coarse grid matrix and rebuild the coarse grid solver
      ! ~~~~~~~~~~~~
      call timer_start(TIMER_ID_AIR_RAP)
      call VecAYPX(d_vec, PFLARE_ONE, air_data%diag_update_base_aff(no_levels), ierr)
      call set_diagonal(air_data%coarse_matrix(no_levels), d_vec)
      call timer_finish(TIMER_ID_AIR_RAP)
      call VecRestoreSubVector(pool, d_is, d_vec, ierr)
      call ISDestroy(d_is, ierr)
      call VecDestroy(pool, ierr)

      call rebuild_approximate_inverse(air_data, air_data%coarse_matrix(no_levels), &
               air_data%inv_coarsest_poly_data, &
               air_data%options%coarsest_matrix_free_polys, air_data%options%coarsest_diag_scale_polys, &
               no_levels, MAT_INV_AFF, air_data%inv_A_ff(no_levels))

      call timer_finish(TIMER_ID_AIR_SETUP)

      ! ~~~~~~~~~~~~
      ! Output some timing results, as at the end of a setup
      ! ~~~~~~~~~~~~
      if (air_data%options%print_stats_timings) then
         call PetscObjectGetComm(air_data%prolongators(1), MPI_COMM_MATRIX, ierr)
         call MPI_Comm_rank(MPI_COMM_MATRIX, comm_rank, errorcode)
         if (comm_rank == 0) then
            print *, "~~~~~~~~~~~~ Diagonal update"
            call print_timers()
            print *, "~~~~~~~~~~~~ "
            print *,  "Total cumulative setup time :", timer_time(TIMER_ID_AIR_SETUP)
            print *, "~~~~~~~~~~~~ "
         end if
      end if

   end subroutine update_air_diag

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine clip_at_one(vec, temp_a, temp_b)

      ! Replaces each entry of vec with min(1, vec), written as 
      ! vec - (vec - 1 + |vec - 1|) / 2 with vector operations that run on the device
      ! Where an entry is at most one, vec - 1 + |vec - 1| is exactly zero, so it doesn't change
      ! The temporaries must have the same layout as vec

      ! ~~~~~~
      type(tVec), intent(inout) :: vec, temp_a, temp_b

      PetscErrorCode :: ierr
      ! ~~~~~~

      call VecCopy(vec, temp_a, ierr)
      call VecShift(temp_a, PFLARE_MINUS_ONE, ierr)
      call VecCopy(temp_a, temp_b, ierr)
      call VecAbs(temp_b, ierr)
      call VecAXPY(temp_a, PFLARE_ONE, temp_b, ierr)
      call VecAXPY(vec, -PFLARE_HALF, temp_a, ierr)

   end subroutine clip_at_one

! -------------------------------------------------------------------------------------------------------------------------------

   PetscInt function level_local_size(air_data, our_level)

      ! The local size of a level, from the rows of its prolongator or on the 
      ! coarsest level the columns of the prolongator above it
      ! The prolongators have been repartitioned along with the coarse matrices

      ! ~~~~~~
      type(air_multigrid_data), intent(in) :: air_data
      integer, intent(in)                  :: our_level

      PetscInt :: local_rows, local_cols
      PetscErrorCode :: ierr
      ! ~~~~~~

      if (our_level < air_data%no_levels) then
         call MatGetLocalSize(air_data%prolongators(our_level), local_rows, local_cols, ierr)
         level_local_size = local_rows
      else
         call MatGetLocalSize(air_data%prolongators(our_level - 1), local_rows, local_cols, ierr)
         level_local_size = local_cols
      end if

   end function level_local_size

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine create_pool(air_data, pool)

      ! Creates a vector with three parts, each big enough to hold any level on this rank

      ! ~~~~~~
      type(air_multigrid_data), intent(in) :: air_data
      type(tVec), intent(inout)            :: pool

      integer :: our_level
      PetscInt :: max_local
      PetscInt, parameter :: three = 3
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX
      VecType :: vec_type
      ! ~~~~~~

      max_local = 0
      do our_level = 1, air_data%no_levels
         max_local = max(max_local, level_local_size(air_data, our_level))
      end do

      call PetscObjectGetComm(air_data%prolongators(1), MPI_COMM_MATRIX, ierr)
      call MatGetVecType(air_data%prolongators(1), vec_type, ierr)
      call VecCreate(MPI_COMM_MATRIX, pool, ierr)
      call VecSetSizes(pool, three * max_local, PETSC_DECIDE, ierr)
      call VecSetType(pool, vec_type, ierr)

   end subroutine create_pool

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine get_level_vec(air_data, pool, our_level, part, is_part, level_vec)

      ! Gets a vector with the layout of our_level that shares the memory of 
      ! part (1, 2 or 3) of the pool
      ! The IS is contiguous so this doesn't copy, and works on the device

      ! ~~~~~~
      type(air_multigrid_data), intent(in) :: air_data
      type(tVec), intent(inout)            :: pool
      integer, intent(in)                  :: our_level, part
      type(tIS), intent(inout)             :: is_part
      type(tVec), intent(inout)            :: level_vec

      PetscInt :: pool_start, pool_end_plus_one, part_size, offset
      PetscInt, parameter :: one = 1, three = 3
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX
      ! ~~~~~~

      call VecGetOwnershipRange(pool, pool_start, pool_end_plus_one, ierr)
      part_size = (pool_end_plus_one - pool_start) / three
      offset = part - 1
      call PetscObjectGetComm(pool, MPI_COMM_MATRIX, ierr)
      call ISCreateStride(MPI_COMM_MATRIX, level_local_size(air_data, our_level), &
               pool_start + offset * part_size, one, is_part, ierr)
      call VecGetSubVector(pool, is_part, level_vec, ierr)

   end subroutine get_level_vec

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine setup_diag_update_data(air_data)

      ! Captures the baseline diagonals from the hierarchy, which are all we need to 
      ! keep between updates

      ! ~~~~~~
      type(air_multigrid_data), intent(inout) :: air_data

      integer :: our_level, no_levels
      PetscErrorCode :: ierr
      ! ~~~~~~

      no_levels = air_data%no_levels

      allocate(air_data%diag_update_base_aff(no_levels))
      if (air_data%options%any_c_smooths) then
         allocate(air_data%diag_update_base_acc(no_levels - 1))
      end if

      do our_level = 1, no_levels - 1

         ! Baseline diagonal of A_ff - this works for matdiagonal too
         call MatCreateVecs(air_data%A_ff(our_level), PETSC_NULL_VEC, &
                  air_data%diag_update_base_aff(our_level), ierr)
         call MatGetDiagonal(air_data%A_ff(our_level), air_data%diag_update_base_aff(our_level), ierr)

         ! Baseline diagonal of A_cc
         if (air_data%options%any_c_smooths) then
            if (air_data%allocated_matrices_A_cc(our_level)) then
               call MatCreateVecs(air_data%A_cc(our_level), PETSC_NULL_VEC, &
                        air_data%diag_update_base_acc(our_level), ierr)
               call MatGetDiagonal(air_data%A_cc(our_level), air_data%diag_update_base_acc(our_level), ierr)
            end if
         end if
      end do

      ! Baseline diagonal of the coarsest matrix
      call MatCreateVecs(air_data%coarse_matrix(no_levels), PETSC_NULL_VEC, &
               air_data%diag_update_base_aff(no_levels), ierr)
      call MatGetDiagonal(air_data%coarse_matrix(no_levels), air_data%diag_update_base_aff(no_levels), ierr)

      air_data%diag_update_initialised = .TRUE.

   end subroutine setup_diag_update_data

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine set_diagonal(matrix, diag_vec)

      ! Sets the diagonal of a matrix, which may be a matdiagonal

      ! ~~~~~~
      type(tMat), intent(inout) :: matrix
      type(tVec), intent(in)    :: diag_vec

      PetscErrorCode :: ierr
      MatType :: mat_type
      type(tVec) :: mat_diag
      ! ~~~~~~

      call MatGetType(matrix, mat_type, ierr)
      if (mat_type == MATDIAGONAL) then
         call MatDiagonalGetDiagonal(matrix, mat_diag, ierr)
         call VecCopy(diag_vec, mat_diag, ierr)
         call MatDiagonalRestoreDiagonal(matrix, mat_diag, ierr)
      else
         ! The diagonal is always present in the matrices of the hierarchy
         call MatDiagonalSet(matrix, diag_vec, INSERT_VALUES, ierr)
      end if

   end subroutine set_diagonal

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine rebuild_approximate_inverse(air_data, matrix, poly_data, matrix_free, diag_scale_polys, &
                  our_level, reuse_index, inv_matrix)

      ! Rebuilds an approximate inverse in place after the values of matrix have changed
      ! The inverse keeps its sparsity as only the diagonal of matrix changes
      ! The inverse matrix is rebuilt in place rather than recreated as the pcmg and
      ! any products attached by the block smoothers hold references to it

      ! ~~~~~~
      type(air_multigrid_data), intent(inout) :: air_data
      type(tMat), intent(in)                  :: matrix
      type(gmres_poly_data), intent(inout)    :: poly_data
      logical, intent(in)                     :: matrix_free, diag_scale_polys
      integer, intent(in)                     :: our_level, reuse_index
      type(tMat), intent(inout)               :: inv_matrix
      ! ~~~~~~

      call timer_start(TIMER_ID_AIR_INVERSE)

      ! If we are re-using the polynomial coefficients then we don't have to compute
      ! them, exactly as the setup does
      if (.NOT. air_data%options%reuse_poly_coeffs) then
         call start_approximate_inverse(matrix, &
                  poly_data%inverse_type, &
                  poly_data%gmres_poly_order, &
                  diag_scale_polys, &
                  poly_data%buffers, &
                  poly_data%coefficients)
      end if

      call finish_approximate_inverse(matrix, &
               poly_data%inverse_type, &
               poly_data%gmres_poly_order, &
               poly_data%gmres_poly_sparsity_order, &
               poly_data%buffers, &
               poly_data%coefficients, &
               matrix_free, &
               diag_scale_polys, &
               air_data%reuse(our_level)%reuse_mat(reuse_index), &
               air_data%reuse(our_level)%reuse_submatrices(reuse_index)%array, &
               inv_matrix)

      call timer_finish(TIMER_ID_AIR_INVERSE)

      ! Delete temporary if not reusing, exactly as the setup does
      if (.NOT. air_data%options%reuse_sparsity .OR. &
          .NOT. REUSE_MAT_ACTIVE(reuse_index, air_data%options%reuse_amount)) then
         call destroy_matrix_reuse(air_data%reuse(our_level)%reuse_mat(reuse_index), &
                  air_data%reuse(our_level)%reuse_submatrices(reuse_index)%array)
      end if

   end subroutine rebuild_approximate_inverse

! -------------------------------------------------------------------------------------------------------------------------------

end module air_diagonal_update
