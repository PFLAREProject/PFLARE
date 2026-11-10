module air_diagonal_update

   use petscksp
   use air_data_type, only: air_multigrid_data, REUSE_MAT_ACTIVE
   use gmres_poly_data_type, only: gmres_poly_data
   use pflare_parameters, only: MAT_INV_AFF, MAT_INV_ACC, PFLARE_ONE, PFLARE_MINUS_ONE, PFLARE_HALF, &
         TIMER_ID_AIR_SETUP, TIMER_ID_AIR_INVERSE, TIMER_ID_AIR_EXTRACT, TIMER_ID_AIR_RESTRICT, &
         TIMER_ID_AIR_RAP
   use timers, only: timer_start, timer_finish, timer_time, print_timers
   use approx_inverse_setup, only: start_approximate_inverse, finish_approximate_inverse, &
         destroy_matrix_reuse, inverse_type_is_polynomial
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
   ! R diag(d) P, though never below the perturbation on that C point, and the F point 
   ! columns of the restrictors are rescaled to account for the change in A_ff, though 
   ! only ever shrunk
   ! The update is always relative to the hierarchy built from A (ie it is not cumulative),
   ! so applying D = 0 gives back the original hierarchy
   ! -------------------------------------------------------------------------------------------------------------------------------

   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine update_air_diag(air_data, pmat)

      ! Updates the hierarchy in air_data with the change in the diagonal of pmat 
      ! since the hierarchy was built
      ! The only vectors we keep just for updates are the baseline diagonals
      ! The F and C point scratch on each level are the smoothing temporaries, and the 
      ! vectors with the layout of each level are the x and r we gave to PCMG, which it 
      ! only uses during a PCApply

      ! ~~~~~~
      type(air_multigrid_data), intent(inout) :: air_data
      type(tMat), intent(in)                  :: pmat

      integer :: our_level, no_levels, errorcode, comm_rank
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX
      type(tVec) :: temp_vec, d_vec, d_coarse, work, f_new, f_old, c_vec
      logical :: update_acc
      PetscReal, parameter :: zero = 0.0
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

      ! The perturbation on the top grid
      d_vec = air_data%diag_update_r(1)
      call MatGetDiagonal(pmat, d_vec, ierr)
      call VecAXPY(d_vec, PFLARE_MINUS_ONE, air_data%diag_update_base_pmat, ierr)

      ! ~~~~~~~~~~~~
      ! Go down the hierarchy
      ! ~~~~~~~~~~~~
      do our_level = 1, no_levels - 1

         ! The perturbation on the next level is in its x
         d_coarse = air_data%diag_update_x(our_level + 1)
         ! The top grid only has the r, which has the perturbation in it. That is fine as we can
         ! always recompute the perturbation on the top grid
         work = air_data%diag_update_r(our_level)
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
            temp_vec = air_data%temp_vecs_fine(3)%array(our_level)
            call VecSet(temp_vec, PFLARE_ONE, ierr)
            call VecPointwiseDivide(f_new, air_data%diag_update_base_aff(our_level), f_new, ierr)
            call VecPointwiseMin(f_new, f_new, temp_vec, ierr)
            ! R has already been scaled by the clipped a_ff / (a_ff + d_f_old), so scale by 
            ! the ratio of the new scaling to that
            call VecPointwiseDivide(f_old, air_data%diag_update_base_aff(our_level), f_old, ierr)
            call VecPointwiseMin(f_old, f_old, temp_vec, ierr)
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
         ! Compute the perturbation on the coarse grid
         ! ~~~~~~~~~~~~
         ! Row sum lumping of R diag(d) P, ie R (d .* P 1), which keeps the action 
         ! of R diag(d) P on the constant
         ! P 1 is one matvec with P, so we just recompute it rather than store it
         ! The off-diagonals of the coarse matrix are frozen, so on a decrease they are too small
         ! and the lumping has to make up for that on the diagonal, which can take it through zero
         ! on DG, where the diagonal blocks of the level aren't dominant. So the perturbation is 
         ! never less than the injected one, which is d on the C points
         call timer_start(TIMER_ID_AIR_RAP)
         call VecSet(d_coarse, PFLARE_ONE, ierr)
         call MatMult(air_data%prolongators(our_level), d_coarse, work, ierr)
         if (our_level == 1) then
            ! Work and d_vec are the same here, so take the F points of P 1 and 
            ! recompute the perturbation. P has the identity on the C points, so
            ! d .* P 1 is just d there and we only have to fix the F points
            f_new = air_data%temp_vecs_fine(3)%array(our_level)
            f_old = air_data%temp_vecs_fine(4)%array(our_level)
            call VecISCopyLocalWrapper(air_data, our_level, .TRUE., work, SCATTER_REVERSE, f_old)
            call MatGetDiagonal(pmat, d_vec, ierr)
            call VecAXPY(d_vec, PFLARE_MINUS_ONE, air_data%diag_update_base_pmat, ierr)
            call VecISCopyLocalWrapper(air_data, our_level, .TRUE., d_vec, SCATTER_REVERSE, f_new)
            call VecPointwiseMult(f_new, f_new, f_old, ierr)
            if (air_data%fast_veciscopy_exists) then
               call VecISCopyLocalWrapper(air_data, our_level, .TRUE., work, SCATTER_FORWARD, f_new)
            else
               call VecISCopy(work, air_data%IS_fine_index(our_level), SCATTER_FORWARD, f_new, ierr)
            end if
         else
            call VecPointwiseMult(work, work, d_vec, ierr)
         end if
         ! If we're symmetric the restrictor is P^T
         if (air_data%options%symmetric) then
            call MatMultTranspose(air_data%prolongators(our_level), work, d_coarse, ierr)
         else
            call MatMult(air_data%restrictors(our_level), work, d_coarse, ierr)
         end if
         ! P has the identity on the C points so work has d there
         c_vec = air_data%temp_vecs_coarse(1)%array(our_level)
         call VecISCopyLocalWrapper(air_data, our_level, .FALSE., work, SCATTER_REVERSE, c_vec)
         if (air_data%repartitioned(our_level)) then
            ! The C points aren't in the layout of the next level, but R has the identity 
            ! on the C points so we can apply it to get them there
            call VecSet(work, zero, ierr)
            if (air_data%fast_veciscopy_exists) then
               call VecISCopyLocalWrapper(air_data, our_level, .FALSE., work, SCATTER_FORWARD, c_vec)
            else
               call VecISCopy(work, air_data%IS_coarse_index(our_level), SCATTER_FORWARD, c_vec, ierr)
            end if
            c_vec = air_data%diag_update_r(our_level + 1)
            if (air_data%options%symmetric) then
               call MatMultTranspose(air_data%prolongators(our_level), work, c_vec, ierr)
            else
               call MatMult(air_data%restrictors(our_level), work, c_vec, ierr)
            end if
         end if
         call VecPointwiseMax(d_coarse, d_coarse, c_vec, ierr)
         call timer_finish(TIMER_ID_AIR_RAP)

         ! The coarse perturbation becomes the perturbation on the next level
         d_vec = d_coarse
      end do

      ! ~~~~~~~~~~~~
      ! Update the coarse grid matrix and rebuild the coarse grid solver
      ! ~~~~~~~~~~~~
      call timer_start(TIMER_ID_AIR_RAP)
      call VecAYPX(d_vec, PFLARE_ONE, air_data%diag_update_base_aff(no_levels), ierr)
      call set_diagonal(air_data%coarse_matrix(no_levels), d_vec)
      call timer_finish(TIMER_ID_AIR_RAP)

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
      if (.NOT. (poly_data%coefficients_set .AND. air_data%options%reuse_poly_coeffs)) then
         call start_approximate_inverse(matrix, &
                  poly_data%inverse_type, &
                  poly_data%gmres_poly_order, &
                  diag_scale_polys, &
                  poly_data%buffers, &
                  poly_data%coefficients)
         poly_data%coefficients_set = inverse_type_is_polynomial(poly_data%inverse_type)
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
