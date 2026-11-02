module air_multishift

   use petscksp
   use air_data_type, only: air_multigrid_data
   use air_data_type_routines, only: destroy_multishift_data
   use air_diagonal_update, only: setup_diag_update_data, clip_at_one
   use fc_smooth_block, only: create_air_block_temp, ensure_air_block_temps, MatISCopyLocalWrapper, &
         mat_product_block
   use gmres_poly_apply, only: mat_dense_pointwise_mult
   use pflare_parameters, only: PFLARE_ONE, PFLARE_MINUS_ONE

#include "petsc/finclude/petscksp.h"

   implicit none
   public

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! The setup of a multi-shift block V-cycle (prototype)
   ! Given a hierarchy built from A and a dense n x k block Dshift, a block apply on k
   ! columns preconditions column j for A + diag(Dshift(:,j)), sharing every sparse
   ! matrix by dense matrix product between the columns
   ! Column j gets exactly the hierarchy that a diagonal update (update_air_diag) to
   ! A + diag(Dshift(:,j)) would build with -pc_air_reuse_poly_coeffs, ie:
   !  - A_ff (and A_cc) of column j are the current matrices plus a per column diagonal
   !    shift, applied as A_ff x + S .* x
   !  - the F point columns of R are scaled per column by the clipped a_ff / (a_ff + d_f),
   !    applied as R (C .* x), with the scaling relative to the current R
   !  - the coarse grid shifts are the row sum lumping R^(j) (d .* P 1), one SpMM per level
   !  - the smoothers are the matrix-free polynomials of the shifted matrices with the
   !    coefficients of the current hierarchy (shared between the columns)
   ! The shifts are stored relative to the current state of the hierarchy, so if the
   ! hierarchy is updated afterwards setup_air_multishift has to be called again
   ! (PCAIR_Shell does this after every diagonal update)
   ! The cycle itself is the PCMG kaskade cycle PCAIR always uses: the smoothers and coarse
   ! grid solver are our own pcshells, which pick up the shifts from the air data, and the
   ! per column R scaling is done by matshell restrictors swapped into the PCMG (see
   ! install_multishift_pcmg in PCAIR_Shell.F90), so no change to PETSc is needed
   ! -------------------------------------------------------------------------------------------------------------------------------

   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine dense_add_vec_cols(y_mat, vec, alpha)

      ! y_mat(:,j) = y_mat(:,j) + alpha * vec for every column j

      ! ~~~~~~
      type(tMat), intent(inout) :: y_mat
      type(tVec), intent(in)    :: vec
      PetscScalar, intent(in)   :: alpha

      PetscErrorCode :: ierr
      PetscInt :: local_rows, local_cols, j_loc
      PetscScalar, pointer :: y_array(:,:)
      PetscScalar, pointer :: v_array(:)
      ! ~~~~~~

      call MatGetLocalSize(y_mat, local_rows, local_cols, ierr)
      if (local_rows == 0) return
      call MatDenseGetArray(y_mat, y_array, ierr)
      call VecGetArrayRead(vec, v_array, ierr)
      do j_loc = 1, size(y_array, 2)
         y_array(:, j_loc) = y_array(:, j_loc) + alpha * v_array
      end do
      call VecRestoreArrayRead(vec, v_array, ierr)
      call MatDenseRestoreArray(y_mat, y_array, ierr)

   end subroutine dense_add_vec_cols

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine dense_elementwise(y_mat, op)

      ! Elementwise operations on a dense block
      ! op = 1: y = 1 / y
      ! op = 2: y = min(1, y)
      ! op = 3: y = 1

      ! ~~~~~~
      type(tMat), intent(inout) :: y_mat
      integer, intent(in)       :: op

      PetscErrorCode :: ierr
      PetscInt :: local_rows, local_cols
      PetscScalar, pointer :: y_array(:,:)
      ! ~~~~~~

      call MatGetLocalSize(y_mat, local_rows, local_cols, ierr)
      if (local_rows == 0) return
      call MatDenseGetArray(y_mat, y_array, ierr)
      if (op == 1) then
         y_array = PFLARE_ONE / y_array
      else if (op == 2) then
         y_array = min(PFLARE_ONE, y_array)
      else
         y_array = PFLARE_ONE
      end if
      call MatDenseRestoreArray(y_mat, y_array, ierr)

   end subroutine dense_elementwise

! -------------------------------------------------------------------------------------------------------------------------------

   logical function is_shell_inverse(inv_mat)

      ! Is this approximate inverse a matshell, which a multi-shift needs

      ! ~~~~~~
      type(tMat), intent(in) :: inv_mat

      MatType :: inv_type
      PetscErrorCode :: ierr
      ! ~~~~~~

      call MatGetType(inv_mat, inv_type, ierr)
      is_shell_inverse = inv_type == MATSHELL

   end function is_shell_inverse

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine setup_air_multishift(air_data, dshift)

      ! Computes the per level multi-shift data from the dense n x k block of shifts dshift,
      ! where column j is diag(D_j) - diag(Pmat the hierarchy was built from)
      ! This is the batched analogue of update_air_diag, with SpMMs in place of SpMVs
      ! and nothing in the hierarchy itself is changed

      ! ~~~~~~
      type(air_multigrid_data), intent(inout) :: air_data
      type(tMat), intent(in)                  :: dshift

      integer :: our_level, no_levels, errorcode
      PetscErrorCode :: ierr
      PetscInt :: local_rows, local_cols, global_rows, global_cols
      type(tMat) :: d_copy, d_cur, d_next, w_mat, af_mat, temp_mat, null_mat
      type(tVec) :: f_vec, c_vec, ones_coarse, p_one, temp_vec
      logical :: nonsym, scaled, c_update
      ! ~~~~~~

      no_levels = air_data%no_levels

      if (no_levels < 2) then
         print *, "A multi-shift requires a hierarchy with at least two levels"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      if (air_data%options%full_smoothing_up_and_down) then
         print *, "A multi-shift is not supported with full smoothing up and down"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      ! The polynomials of the shifted matrices can only be applied matrix-free
      do our_level = 1, no_levels
         if (.NOT. is_shell_inverse(air_data%inv_A_ff(our_level))) then
            print *, "A multi-shift needs matrix-free polynomial smoothers and coarse grid solver, ", &
                     "set -pc_air_matrix_free_polys and -pc_air_coarsest_matrix_free_polys"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
         end if
      end do

      ! Copy the shifts first, the dshift we're given may be our own copy
      call MatDuplicate(dshift, MAT_COPY_VALUES, d_copy, ierr)
      call destroy_multishift_data(air_data)
      air_data%ms_dshift = d_copy

      ! The baselines of the hierarchy, if a diagonal update hasn't captured them yet
      ! then the hierarchy is still the one built from the base pmat
      if (.NOT. air_data%diag_update_initialised) call setup_diag_update_data(air_data)

      call MatGetSize(dshift, global_rows, global_cols, ierr)
      call MatGetLocalSize(dshift, local_rows, local_cols, ierr)

      ! The smoothing scratch, which the IS copies may need
      call ensure_air_block_temps(air_data, dshift, ierr)

      allocate(air_data%ms_shift_f(no_levels), air_data%ms_recip_f(no_levels), &
               air_data%ms_shift_c(no_levels), air_data%ms_recip_c(no_levels), &
               air_data%ms_rscale(no_levels))
      ! The handles are default initialised to null by the allocate - we can't assign
      ! PETSC_NULL_MAT, the fortran stubs pass that through as a null pointer

      nonsym = .NOT. air_data%options%symmetric
      scaled = air_data%options%diag_scale_polys

      ! The shifts on the top grid
      call MatDuplicate(dshift, MAT_COPY_VALUES, d_cur, ierr)

      do our_level = 1, no_levels - 1

         f_vec = air_data%temp_vecs_fine(1)%array(our_level)

         ! ~~~~~~~~~~~~
         ! The shifted diagonal of A_ff for each column, af = base + d_f
         ! ~~~~~~~~~~~~
         af_mat = null_mat
         call create_air_block_temp(air_data%A_ff(our_level), .FALSE., dshift, &
                  local_cols, global_cols, af_mat, ierr)
         call MatISCopyLocalWrapper(air_data, our_level, .TRUE., d_cur, SCATTER_REVERSE, af_mat)
         call dense_add_vec_cols(af_mat, air_data%diag_update_base_aff(our_level), PFLARE_ONE)

         ! The reciprocal used by the diagonally scaled polynomials
         if (scaled) then
            call MatDuplicate(af_mat, MAT_COPY_VALUES, air_data%ms_recip_f(our_level), ierr)
            call dense_elementwise(air_data%ms_recip_f(our_level), 1)
         end if

         ! ~~~~~~~~~~~~
         ! The per column scaling of the F point columns of R
         ! ~~~~~~~~~~~~
         ! The current R has been scaled by c_cur = min(1, base / current), and column j
         ! wants c_j = min(1, base / af_j), so we scale by c_j / c_cur
         if (nonsym) then
            ! temp = min(1, base / af)
            call MatDuplicate(af_mat, MAT_COPY_VALUES, temp_mat, ierr)
            call dense_elementwise(temp_mat, 1)
            call MatDiagonalScale(temp_mat, air_data%diag_update_base_aff(our_level), PETSC_NULL_VEC, ierr)
            call dense_elementwise(temp_mat, 2)
            ! f_vec = 1 / min(1, base / current)
            call MatGetDiagonal(air_data%A_ff(our_level), f_vec, ierr)
            call VecPointwiseDivide(f_vec, air_data%diag_update_base_aff(our_level), f_vec, ierr)
            call clip_at_one(f_vec, air_data%temp_vecs_fine(3)%array(our_level), &
                     air_data%temp_vecs_fine(4)%array(our_level))
            call VecReciprocal(f_vec, ierr)
            call MatDiagonalScale(temp_mat, f_vec, PETSC_NULL_VEC, ierr)
            ! Put it in the F rows of a full level block of ones
            call create_air_block_temp(air_data%prolongators(our_level), .FALSE., dshift, &
                     local_cols, global_cols, air_data%ms_rscale(our_level), ierr)
            call dense_elementwise(air_data%ms_rscale(our_level), 3)
            call MatISCopyLocalWrapper(air_data, our_level, .TRUE., air_data%ms_rscale(our_level), &
                     SCATTER_FORWARD, temp_mat, air_data%block_temp_full(1)%array(our_level))
            call MatDestroy(temp_mat, ierr)
         end if

         ! ~~~~~~~~~~~~
         ! The shift relative to the current A_ff, af - current
         ! ~~~~~~~~~~~~
         call MatGetDiagonal(air_data%A_ff(our_level), f_vec, ierr)
         call dense_add_vec_cols(af_mat, f_vec, PFLARE_MINUS_ONE)
         air_data%ms_shift_f(our_level) = af_mat

         ! ~~~~~~~~~~~~
         ! The same for A_cc if we're C smoothing
         ! ~~~~~~~~~~~~
         c_update = .FALSE.
         if (allocated(air_data%diag_update_base_acc)) then
            temp_vec = air_data%diag_update_base_acc(our_level)
            c_update = .NOT. PetscObjectIsNull(temp_vec)
         end if
         if (c_update) then
            if (.NOT. is_shell_inverse(air_data%inv_A_cc(our_level))) then
               print *, "A multi-shift needs matrix-free polynomial C point smoothers"
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
            end if
            c_vec = air_data%temp_vecs_coarse(1)%array(our_level)
            af_mat = null_mat
            call create_air_block_temp(air_data%A_fc(our_level), .TRUE., dshift, &
                     local_cols, global_cols, af_mat, ierr)
            call MatISCopyLocalWrapper(air_data, our_level, .FALSE., d_cur, SCATTER_REVERSE, af_mat)
            call dense_add_vec_cols(af_mat, air_data%diag_update_base_acc(our_level), PFLARE_ONE)
            if (scaled) then
               call MatDuplicate(af_mat, MAT_COPY_VALUES, air_data%ms_recip_c(our_level), ierr)
               call dense_elementwise(air_data%ms_recip_c(our_level), 1)
            end if
            call MatGetDiagonal(air_data%A_cc(our_level), c_vec, ierr)
            call dense_add_vec_cols(af_mat, c_vec, PFLARE_MINUS_ONE)
            air_data%ms_shift_c(our_level) = af_mat
         end if

         ! ~~~~~~~~~~~~
         ! The shifts on the coarse grid, R^(j) (d_j .* P 1)
         ! ~~~~~~~~~~~~
         call MatCreateVecs(air_data%prolongators(our_level), ones_coarse, p_one, ierr)
         call VecSet(ones_coarse, PFLARE_ONE, ierr)
         call MatMult(air_data%prolongators(our_level), ones_coarse, p_one, ierr)
         call MatDuplicate(d_cur, MAT_COPY_VALUES, w_mat, ierr)
         call MatDiagonalScale(w_mat, p_one, PETSC_NULL_VEC, ierr)
         call VecDestroy(ones_coarse, ierr)
         call VecDestroy(p_one, ierr)

         d_next = null_mat
         call create_air_block_temp(air_data%prolongators(our_level), .TRUE., dshift, &
                  local_cols, global_cols, d_next, ierr)
         if (nonsym) then
            call mat_dense_pointwise_mult(w_mat, air_data%ms_rscale(our_level))
            call mat_product_block(air_data%restrictors(our_level), w_mat, d_next, .FALSE., ierr)
         else
            call mat_product_block(air_data%prolongators(our_level), w_mat, d_next, .TRUE., ierr)
         end if
         call MatDestroy(w_mat, ierr)
         call MatDestroy(d_cur, ierr)
         d_cur = d_next
      end do

      ! ~~~~~~~~~~~~
      ! The coarsest grid
      ! ~~~~~~~~~~~~
      call dense_add_vec_cols(d_cur, air_data%diag_update_base_aff(no_levels), PFLARE_ONE)
      if (air_data%options%coarsest_diag_scale_polys) then
         call MatDuplicate(d_cur, MAT_COPY_VALUES, air_data%ms_recip_f(no_levels), ierr)
         call dense_elementwise(air_data%ms_recip_f(no_levels), 1)
      end if
      call MatCreateVecs(air_data%coarse_matrix(no_levels), PETSC_NULL_VEC, temp_vec, ierr)
      call MatGetDiagonal(air_data%coarse_matrix(no_levels), temp_vec, ierr)
      call dense_add_vec_cols(d_cur, temp_vec, PFLARE_MINUS_ONE)
      call VecDestroy(temp_vec, ierr)
      air_data%ms_shift_f(no_levels) = d_cur

      air_data%multishift_active = .TRUE.


   end subroutine setup_air_multishift

end module air_multishift
