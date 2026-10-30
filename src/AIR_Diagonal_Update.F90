module air_diagonal_update

   use petscksp
   use air_data_type, only: air_multigrid_data, REUSE_MAT_ACTIVE
   use gmres_poly_data_type, only: gmres_poly_data
   use pflare_parameters, only: MAT_INV_AFF, MAT_INV_ACC, PFLARE_ONE
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
   ! the change in A_ff
   ! The update is always relative to the hierarchy built from A (ie it is not cumulative),
   ! so applying D = 0 gives back the original hierarchy
   ! -------------------------------------------------------------------------------------------------------------------------------

   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine update_air_diag(air_data, diag_update)

      ! Applies the diagonal perturbation diag_update (which has the layout of the
      ! top grid matrix) to the hierarchy in air_data

      ! ~~~~~~
      type(air_multigrid_data), intent(inout) :: air_data
      type(tVec), intent(in)                  :: diag_update

      integer :: our_level, no_levels, errorcode
      PetscErrorCode :: ierr
      type(tVec) :: s_f, temp_vec
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

      ! The first time through we capture the diagonals of the hierarchy we were given
      if (.NOT. air_data%diag_update_initialised) call setup_diag_update_data(air_data)

      call VecCopy(diag_update, air_data%diag_update_d(1), ierr)

      ! ~~~~~~~~~~~~
      ! Go down the hierarchy
      ! ~~~~~~~~~~~~
      do our_level = 1, no_levels - 1

         ! ~~~~~~~~~~~~
         ! Update the diagonal of A_ff
         ! ~~~~~~~~~~~~
         ! Pull out the F points of the perturbation
         call VecISCopyLocalWrapper(air_data, our_level, .TRUE., air_data%diag_update_d(our_level), &
                  SCATTER_REVERSE, air_data%diag_update_f(our_level), air_data%diag_update_work(our_level))
         ! diag_update_f = baseline + d_f
         call VecAYPX(air_data%diag_update_f(our_level), PFLARE_ONE, &
                  air_data%diag_update_base_aff(our_level), ierr)
         call set_diagonal(air_data%A_ff(our_level), air_data%diag_update_f(our_level))

         ! ~~~~~~~~~~~~
         ! Rescale the F point columns of the restrictor
         ! ~~~~~~~~~~~~
         ! Z = -A_cf A_ff^-1, so if A_ff were diagonal the updated Z would just be
         ! the original with columns scaled by a_ff / (a_ff + d_f)
         ! Without this the frozen Z is far too big when d is large, and with only F point 
         ! smoothing the error that introduces at the C points is never corrected
         ! If we're symmetric we don't have R (it is P^T) and scaling it would also scale P
         if (.NOT. air_data%options%symmetric) then

            call VecDuplicate(air_data%diag_update_f(our_level), s_f, ierr)
            call VecPointwiseDivide(s_f, air_data%diag_update_base_aff(our_level), &
                     air_data%diag_update_f(our_level), ierr)
            ! The scaling on all the points on this level, one on the C points
            call VecSet(air_data%diag_update_work(our_level), PFLARE_ONE, ierr)
            ! On gpus without kokkos this just goes through the host
            if (air_data%fast_veciscopy_exists) then
               call VecISCopyLocalWrapper(air_data, our_level, .TRUE., air_data%diag_update_work(our_level), &
                        SCATTER_FORWARD, s_f)
            else
               call VecISCopy(air_data%diag_update_work(our_level), air_data%IS_fine_index(our_level), &
                        SCATTER_FORWARD, s_f, ierr)
            end if
            call VecDestroy(s_f, ierr)
            ! We only store the current scaled R, so scale by the ratio of the new scaling
            ! to the current one
            call VecPointwiseDivide(air_data%diag_update_r_scale(our_level), &
                     air_data%diag_update_work(our_level), air_data%diag_update_r_scale(our_level), ierr)
            call MatDiagonalScale(air_data%restrictors(our_level), PETSC_NULL_VEC, &
                     air_data%diag_update_r_scale(our_level), ierr)
            call VecCopy(air_data%diag_update_work(our_level), air_data%diag_update_r_scale(our_level), ierr)
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
            call VecISCopyLocalWrapper(air_data, our_level, .FALSE., air_data%diag_update_d(our_level), &
                     SCATTER_REVERSE, air_data%diag_update_c(our_level), air_data%diag_update_work(our_level))
            call VecAYPX(air_data%diag_update_c(our_level), PFLARE_ONE, &
                     air_data%diag_update_base_acc(our_level), ierr)
            call set_diagonal(air_data%A_cc(our_level), air_data%diag_update_c(our_level))

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
         call VecPointwiseMult(air_data%diag_update_work(our_level), air_data%diag_update_d(our_level), &
                  air_data%diag_update_p_one(our_level), ierr)
         call restrict_vec(air_data, our_level, air_data%diag_update_work(our_level), &
                  air_data%diag_update_d(our_level + 1))
      end do

      ! ~~~~~~~~~~~~
      ! Update the coarse grid matrix and rebuild the coarse grid solver
      ! ~~~~~~~~~~~~
      call VecAYPX(air_data%diag_update_d(no_levels), PFLARE_ONE, &
               air_data%diag_update_base_aff(no_levels), ierr)
      call set_diagonal(air_data%coarse_matrix(no_levels), air_data%diag_update_d(no_levels))

      call rebuild_approximate_inverse(air_data, air_data%coarse_matrix(no_levels), &
               air_data%inv_coarsest_poly_data, &
               air_data%options%coarsest_matrix_free_polys, air_data%options%coarsest_diag_scale_polys, &
               no_levels, MAT_INV_AFF, air_data%inv_A_ff(no_levels))

   end subroutine update_air_diag

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine setup_diag_update_data(air_data)

      ! Captures the baseline diagonals from the hierarchy and builds the vectors
      ! (and matrices) we need to apply diagonal updates

      ! ~~~~~~
      type(air_multigrid_data), intent(inout) :: air_data

      integer :: our_level, no_levels
      PetscErrorCode :: ierr
      type(tVec) :: ones
      ! ~~~~~~

      no_levels = air_data%no_levels

      allocate(air_data%diag_update_base_aff(no_levels))
      allocate(air_data%diag_update_d(no_levels))
      allocate(air_data%diag_update_f(no_levels - 1))
      allocate(air_data%diag_update_work(no_levels - 1))
      if (air_data%options%any_c_smooths) then
         allocate(air_data%diag_update_base_acc(no_levels - 1))
         allocate(air_data%diag_update_c(no_levels - 1))
      end if
      allocate(air_data%diag_update_p_one(no_levels - 1))
      if (.NOT. air_data%options%symmetric) then
         allocate(air_data%diag_update_r_scale(no_levels - 1))
      end if

      ! The perturbation on each level has the layout of the rows of the prolongator
      ! (which is the layout of the level) and on the coarsest the layout of its columns
      ! The prolongators always exist and have been repartitioned along with the coarse matrices
      do our_level = 1, no_levels - 1
         call MatCreateVecs(air_data%prolongators(our_level), PETSC_NULL_VEC, &
                  air_data%diag_update_d(our_level), ierr)
         call VecDuplicate(air_data%diag_update_d(our_level), air_data%diag_update_work(our_level), ierr)
      end do
      call MatCreateVecs(air_data%prolongators(no_levels - 1), air_data%diag_update_d(no_levels), &
               PETSC_NULL_VEC, ierr)

      do our_level = 1, no_levels - 1

         ! Baseline diagonal of A_ff - this works for matdiagonal too
         call MatCreateVecs(air_data%A_ff(our_level), PETSC_NULL_VEC, &
                  air_data%diag_update_base_aff(our_level), ierr)
         call MatGetDiagonal(air_data%A_ff(our_level), air_data%diag_update_base_aff(our_level), ierr)
         call VecDuplicate(air_data%diag_update_base_aff(our_level), air_data%diag_update_f(our_level), ierr)

         ! Baseline diagonal of A_cc
         if (air_data%options%any_c_smooths) then
            if (air_data%allocated_matrices_A_cc(our_level)) then
               call MatCreateVecs(air_data%A_cc(our_level), PETSC_NULL_VEC, &
                        air_data%diag_update_base_acc(our_level), ierr)
               call MatGetDiagonal(air_data%A_cc(our_level), air_data%diag_update_base_acc(our_level), ierr)
               call VecDuplicate(air_data%diag_update_base_acc(our_level), air_data%diag_update_c(our_level), ierr)
            end if
         end if

         ! P 1 for row sum lumping
         call VecDuplicate(air_data%diag_update_d(our_level + 1), ones, ierr)
         call VecSet(ones, PFLARE_ONE, ierr)
         call VecDuplicate(air_data%diag_update_d(our_level), air_data%diag_update_p_one(our_level), ierr)
         call MatMult(air_data%prolongators(our_level), ones, air_data%diag_update_p_one(our_level), ierr)
         call VecDestroy(ones, ierr)

         ! The restrictor starts unscaled
         if (.NOT. air_data%options%symmetric) then
            call VecDuplicate(air_data%diag_update_d(our_level), air_data%diag_update_r_scale(our_level), ierr)
            call VecSet(air_data%diag_update_r_scale(our_level), PFLARE_ONE, ierr)
         end if
      end do

      ! Baseline diagonal of the coarsest matrix
      call VecDuplicate(air_data%diag_update_d(no_levels), air_data%diag_update_base_aff(no_levels), ierr)
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

      call start_approximate_inverse(matrix, &
               poly_data%inverse_type, &
               poly_data%gmres_poly_order, &
               diag_scale_polys, &
               poly_data%buffers, &
               poly_data%coefficients)

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

      ! Delete temporary if not reusing, exactly as the setup does
      if (.NOT. air_data%options%reuse_sparsity .OR. &
          .NOT. REUSE_MAT_ACTIVE(reuse_index, air_data%options%reuse_amount)) then
         call destroy_matrix_reuse(air_data%reuse(our_level)%reuse_mat(reuse_index), &
                  air_data%reuse(our_level)%reuse_submatrices(reuse_index)%array)
      end if

   end subroutine rebuild_approximate_inverse

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine restrict_vec(air_data, our_level, fine_vec, coarse_vec)

      ! Applies the restrictor on our_level, which is P^T if we're symmetric

      ! ~~~~~~
      type(air_multigrid_data), intent(inout) :: air_data
      integer, intent(in)                     :: our_level
      type(tVec), intent(inout)               :: fine_vec, coarse_vec

      PetscErrorCode :: ierr
      ! ~~~~~~

      if (air_data%options%symmetric) then
         call MatMultTranspose(air_data%prolongators(our_level), fine_vec, coarse_vec, ierr)
      else
         call MatMult(air_data%restrictors(our_level), fine_vec, coarse_vec, ierr)
      end if

   end subroutine restrict_vec


! -------------------------------------------------------------------------------------------------------------------------------

end module air_diagonal_update
