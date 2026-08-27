module fc_smooth_block

   use petscksp
   use c_petsc_interfaces, only: mat_iscopy_local_kokkos
   use air_data_type, only: air_multigrid_data
   use petsc_helper, only: kokkos_debug
   use matshell_data_type, only: mat_ctxtype, attach_mat_product
   use gmres_poly_newton, only: shell_poly_block_apply
   use pflare_parameters, only: PFLARE_TOL_MATFREE_13, PFLARE_MINUS_ONE, PFLARE_ONE, &
         AIR_MAT_SOL, AIR_MAT_TEMP, AIR_MAT_RESIDUAL, AIR_MAT_RHS, AIR_MAT_OFF_DIAG

#include "petsc/finclude/petscksp.h"
#include "petscconf.h"

   implicit none

   public

   ! The multiple rhs (block) twins of the FC smoothing in fc_smooth, plus the
   ! lifecycle of the dense scratch blocks they smooth in. Everything here is
   ! reached through the PCMatApply of PCAIR - the dense blocks of right hand
   ! sides come down the PCMG hierarchy and every product below is a sparse
   ! matrix by dense matrix product through the MatProduct API, with the
   ! products the smooths run kept attached to the per-level scratch

   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine create_air_block_temp(op_mat, from_cols, x_mat, local_cols, global_cols, block_mat, ierr)

      ! Creates one dense scratch block with the row layout of op_mat (or its
      ! column layout if from_cols is true) and the column layout (and dense type)
      ! of the block of rhs we've been given, x_mat
      ! Does nothing if the block already exists

      ! ~~~~~~
      type(tMat), intent(in)        :: op_mat, x_mat
      logical, intent(in)           :: from_cols
      PetscInt, intent(in)          :: local_cols, global_cols
      type(tMat), intent(inout)     :: block_mat
      PetscErrorCode, intent(inout) :: ierr

      PetscInt :: local_rows, global_rows, local_op_cols, global_op_cols
      VecType :: vec_type
      MPIU_Comm :: MPI_COMM_MATRIX
      type(tMat) :: temp_mat
      ! ~~~~~~

      temp_mat = block_mat
      if (.NOT. PetscObjectIsNull(temp_mat)) return

      call PetscObjectGetComm(op_mat, MPI_COMM_MATRIX, ierr)
      call MatGetLocalSize(op_mat, local_rows, local_op_cols, ierr)
      call MatGetSize(op_mat, global_rows, global_op_cols, ierr)
      if (from_cols) then
         local_rows  = local_op_cols
         global_rows = global_op_cols
      end if
      ! Take the type from the block of rhs, so a block that lives on the device
      ! gives us device scratch
      call MatGetVecType(x_mat, vec_type, ierr)

      call MatCreateDenseFromVecType(MPI_COMM_MATRIX, vec_type, local_rows, local_cols, &
               global_rows, global_cols, PETSC_DECIDE, PETSC_NULL_SCALAR, &
               block_mat, ierr)

   end subroutine create_air_block_temp

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine destroy_air_block_temps(air_data)

      ! Destroys the dense scratch blocks the multiple rhs (block) smooths use
      ! ensure_air_block_temps rebuilds any of them we need on the next block apply

      ! ~~~~~~
      type(air_multigrid_data), intent(inout) :: air_data

      integer :: our_level, i_loc
      PetscErrorCode :: ierr
      type(tMat) :: temp_mat
      ! ~~~~~~

      ! The outer arrays are only allocated in create_air_data
      if (.NOT. allocated(air_data%block_temp_full(1)%array)) return

      do our_level = 1, size(air_data%block_temp_full(1)%array)

         do i_loc = 1, size(air_data%block_temp_fine)

            ! MatDestroy nulls the local copy of the handle, not the slot in the
            ! array, so we have to copy it back or the slot would be left dangling
            ! and ensure_air_block_temps would skip rebuilding it
            temp_mat = air_data%block_temp_fine(i_loc)%array(our_level)
            if (.NOT. PetscObjectIsNull(temp_mat)) then
               call MatDestroy(temp_mat, ierr)
               air_data%block_temp_fine(i_loc)%array(our_level) = temp_mat
            end if

            temp_mat = air_data%block_temp_coarse(i_loc)%array(our_level)
            if (.NOT. PetscObjectIsNull(temp_mat)) then
               call MatDestroy(temp_mat, ierr)
               air_data%block_temp_coarse(i_loc)%array(our_level) = temp_mat
            end if
         end do

         temp_mat = air_data%block_temp_full(1)%array(our_level)
         if (.NOT. PetscObjectIsNull(temp_mat)) then
            call MatDestroy(temp_mat, ierr)
            air_data%block_temp_full(1)%array(our_level) = temp_mat
         end if
      end do

      air_data%block_ncols = -1
      air_data%block_local_ncols = -1

   end subroutine destroy_air_block_temps

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine ensure_air_block_temps(air_data, x_mat, ierr)

      ! Makes sure the cached dense scratch blocks match the block of rhs, x_mat,
      ! we've been given
      ! This is the air equivalent of ensure_block_temp_mats in matshell_data_type
      ! The blocks have the row layouts of the operators on each level (which we
      ! only know after the setup) and the column layout of x_mat (which we only
      ! know once a block apply happens), hence building them lazily here

      ! ~~~~~~
      type(air_multigrid_data), intent(inout) :: air_data
      type(tMat), intent(in)                  :: x_mat
      PetscErrorCode, intent(inout)           :: ierr

      integer :: our_level, i_loc, n_coarse
      logical :: rebuild
      PetscInt :: global_rows, global_cols, local_rows, local_cols
      MatType :: x_type, temp_type
      type(tMat) :: temp_mat
      ! ~~~~~~

      ! With full smoothing up and down we never do an FC smooth, so none of these
      ! blocks are used - the level smoothers and the coarse grid solver apply
      ! their inverses to the whole block directly
      if (air_data%options%full_smoothing_up_and_down) return
      ! Nothing to smooth on if there's only a single level
      if (air_data%no_levels < 2) return
      if (.NOT. allocated(air_data%block_temp_full(1)%array)) return

      call MatGetSize(x_mat, global_rows, global_cols, ierr)
      call MatGetLocalSize(x_mat, local_rows, local_cols, ierr)
      call MatGetType(x_mat, x_type, ierr)

      ! If the number of columns has changed we have to start again - the local
      ! column layout also has to match or the products would have incompatible
      ! layouts
      rebuild = air_data%block_ncols /= global_cols .OR. &
                  air_data%block_local_ncols /= local_cols

      ! The type of block we're given can also change between applies
      if (.NOT. rebuild) then
         do our_level = 1, air_data%no_levels - 1
            temp_mat = air_data%block_temp_fine(1)%array(our_level)
            if (.NOT. PetscObjectIsNull(temp_mat)) then
               call MatGetType(temp_mat, temp_type, ierr)
               if (temp_type /= x_type) rebuild = .TRUE.
               exit
            end if
         end do
      end if

      if (rebuild) call destroy_air_block_temps(air_data)

      ! We only need the extra coarse blocks if we're C point smoothing, in the
      ! same way as the extra temp_vecs_coarse
      n_coarse = 1
      if (air_data%options%any_c_smooths) n_coarse = size(air_data%block_temp_coarse)

      ! Now create any of the blocks we need that don't exist
      do our_level = 1, air_data%no_levels - 1

         do i_loc = 1, size(air_data%block_temp_fine)
            call create_air_block_temp(air_data%A_ff(our_level), .FALSE., x_mat, &
                     local_cols, global_cols, &
                     air_data%block_temp_fine(i_loc)%array(our_level), ierr)
         end do

         ! A_fc has the coarse column layout - we take it from A_fc rather than
         ! A_cf as A_cf is destroyed after the setup unless we're C smoothing,
         ! and this is where temp_vecs_coarse gets its layout from too
         do i_loc = 1, n_coarse
            call create_air_block_temp(air_data%A_fc(our_level), .TRUE., x_mat, &
                     local_cols, global_cols, &
                     air_data%block_temp_coarse(i_loc)%array(our_level), ierr)
         end do

         ! Only the injector version of MatISCopyLocalWrapper needs a full size
         ! block, in the same way as temp_vecs
         if (.NOT. air_data%fast_veciscopy_exists) then
            call create_air_block_temp(air_data%coarse_matrix(our_level), .FALSE., x_mat, &
                     local_cols, global_cols, &
                     air_data%block_temp_full(1)%array(our_level), ierr)
         end if

         ! The blocks on this level have just been (re)built, so attach the
         ! products the block smooths run to their scratch targets - the smooths
         ! then only run the numeric phase of each product every iteration
         ! rebuild is always true when the blocks didn't exist yet (block_ncols
         ! starts and is reset to -1), so the products can't be attached twice
         if (rebuild) call setup_air_block_products(air_data, our_level, ierr)
      end do

      air_data%block_ncols = global_cols
      air_data%block_local_ncols = local_cols

   end subroutine ensure_air_block_temps

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine setup_air_block_products(air_data, our_level, ierr)

      ! Attaches the products the block FC smooths compute on a level to their
      ! dense scratch targets, so each smoother iteration only has to run the
      ! numeric phase of each product
      ! Called by ensure_air_block_temps whenever the scratch on this level has
      ! just been (re)built - the products stay attached until the scratch is
      ! destroyed, and the scratch outlives every operator it references only
      ! briefly (destroy_air_block_temps runs first in reset_air_data)

      ! ~~~~~~
      type(air_multigrid_data), intent(inout) :: air_data
      integer, intent(in)                     :: our_level
      PetscErrorCode, intent(inout)           :: ierr

      logical :: any_f_smooths, any_c_smooths
      MatType :: inv_type
      ! ~~~~~~

      any_f_smooths = any(air_data%smooth_order_levels(our_level)%array > 0)
      any_c_smooths = any(air_data%smooth_order_levels(our_level)%array < 0)

      if (any_f_smooths) then

         ! A_ff * x_f - computed every F richardson iteration
         call attach_mat_product(air_data%A_ff(our_level), &
                  air_data%block_temp_fine(AIR_MAT_SOL)%array(our_level), .FALSE., &
                  air_data%block_temp_fine(AIR_MAT_RESIDUAL)%array(our_level), ierr)

         ! A_fc * x_c - computed once per F smooth
         call attach_mat_product(air_data%A_fc(our_level), &
                  air_data%block_temp_coarse(AIR_MAT_SOL)%array(our_level), .FALSE., &
                  air_data%block_temp_fine(AIR_MAT_OFF_DIAG)%array(our_level), ierr)

         ! The assembled (or diagonal) inverses are applied with a product too -
         ! the matrix-free matshells go through the blockwise shell apply instead
         ! and never have a product attached
         call MatGetType(air_data%inv_A_ff(our_level), inv_type, ierr)
         if (inv_type /= MATSHELL) then
            call attach_mat_product(air_data%inv_A_ff(our_level), &
                     air_data%block_temp_fine(AIR_MAT_RESIDUAL)%array(our_level), .FALSE., &
                     air_data%block_temp_fine(AIR_MAT_TEMP)%array(our_level), ierr)
         end if
      end if

      if (any_c_smooths) then

         ! A_cc * x_c - computed every C richardson iteration
         call attach_mat_product(air_data%A_cc(our_level), &
                  air_data%block_temp_coarse(AIR_MAT_SOL)%array(our_level), .FALSE., &
                  air_data%block_temp_coarse(AIR_MAT_RESIDUAL)%array(our_level), ierr)

         ! A_cf * x_f - computed once per C smooth
         call attach_mat_product(air_data%A_cf(our_level), &
                  air_data%block_temp_fine(AIR_MAT_SOL)%array(our_level), .FALSE., &
                  air_data%block_temp_coarse(AIR_MAT_OFF_DIAG)%array(our_level), ierr)

         call MatGetType(air_data%inv_A_cc(our_level), inv_type, ierr)
         if (inv_type /= MATSHELL) then
            call attach_mat_product(air_data%inv_A_cc(our_level), &
                     air_data%block_temp_coarse(AIR_MAT_RESIDUAL)%array(our_level), .FALSE., &
                     air_data%block_temp_coarse(AIR_MAT_TEMP)%array(our_level), ierr)
         end if
      end if

   end subroutine setup_air_block_products

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine mat_product_block(mat, x_mat, y_mat, transpose_mat, ierr)

      ! Computes y_mat = mat * x_mat (or mat^T * x_mat) for a dense block of
      ! right hand sides, ie the multiple rhs version of MatMult/MatMultTranspose
      ! The product bookkeeping is always cleared afterwards, or y_mat would keep
      ! holding references to mat and x_mat until it is destroyed

      ! ~~~~~~
      type(tMat), intent(in)        :: mat
      type(tMat)                    :: x_mat, y_mat
      logical, intent(in)           :: transpose_mat
      PetscErrorCode, intent(inout) :: ierr
      ! ~~~~~~

      call attach_mat_product(mat, x_mat, transpose_mat, y_mat, ierr)
      call MatProductNumeric(y_mat, ierr)
      call MatProductClear(y_mat, ierr)

   end subroutine mat_product_block

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine shell_block_apply_or_columns(inv_mat, x_mat, y_mat, ierr)

      ! Applies a matrix-free matshell inverse to a dense block of right hand
      ! sides - shell_poly_block_apply knows how to do the polynomial matshells
      ! blockwise by doing the products with the underlying matrix instead, and
      ! reports back if it has been handed something it doesn't know about, in
      ! which case we apply column by column

      ! ~~~~~~
      type(tMat), intent(in)        :: inv_mat
      type(tMat)                    :: x_mat, y_mat
      PetscErrorCode, intent(inout) :: ierr

      logical :: block_applied
      PetscInt :: global_rows, global_cols, i_loc
      type(tVec) :: col_x, col_y
      ! ~~~~~~

      call shell_poly_block_apply(inv_mat, x_mat, y_mat, block_applied)

      ! Anything we can't do blockwise has to be applied column by column
      if (.NOT. block_applied) then
         call MatGetSize(x_mat, global_rows, global_cols, ierr)
         do i_loc = 0, global_cols-1
            call MatDenseGetColumnVecRead(x_mat, i_loc, col_x, ierr)
            call MatDenseGetColumnVecWrite(y_mat, i_loc, col_y, ierr)
            call MatMult(inv_mat, col_x, col_y, ierr)
            call MatDenseRestoreColumnVecWrite(y_mat, i_loc, col_y, ierr)
            call MatDenseRestoreColumnVecRead(x_mat, i_loc, col_x, ierr)
         end do
      end if

   end subroutine shell_block_apply_or_columns

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine apply_inverse_block(inv_mat, x_mat, y_mat, ierr)

      ! Applies one of our approximate inverses to a dense block of right hand
      ! sides, ie the multiple rhs version of MatMult(inv_mat, x, y)
      ! The matrix-free polynomial inverses are matshells with only a MATOP_MULT,
      ! so a product on them would fail - they go through the blockwise shell
      ! apply instead

      ! ~~~~~~
      type(tMat), intent(in)        :: inv_mat
      type(tMat)                    :: x_mat, y_mat
      PetscErrorCode, intent(inout) :: ierr

      MatType :: inv_type
      ! ~~~~~~

      call MatGetType(inv_mat, inv_type, ierr)

      if (inv_type == MATSHELL) then
         call shell_block_apply_or_columns(inv_mat, x_mat, y_mat, ierr)

      ! Assembled (or diagonal) inverses just do a real product
      else
         call mat_product_block(inv_mat, x_mat, y_mat, .FALSE., ierr)
      end if

   end subroutine apply_inverse_block

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine apply_inverse_block_cached(inv_mat, x_mat, y_mat, ierr)

      ! The cached version of apply_inverse_block used inside the block FC
      ! smooths - for assembled (or diagonal) inverses the product has already
      ! been attached to y_mat by setup_air_block_products, so only the numeric
      ! phase runs here. The matrix-free matshells go through the same blockwise
      ! shell apply as apply_inverse_block

      ! ~~~~~~
      type(tMat), intent(in)        :: inv_mat
      type(tMat)                    :: x_mat, y_mat
      PetscErrorCode, intent(inout) :: ierr

      MatType :: inv_type
      ! ~~~~~~

      call MatGetType(inv_mat, inv_type, ierr)

      if (inv_type == MATSHELL) then
         call shell_block_apply_or_columns(inv_mat, x_mat, y_mat, ierr)
      else
         call MatProductNumeric(y_mat, ierr)
      end if

   end subroutine apply_inverse_block_cached

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine MatISCopyLocalWrapper(air_data, our_level, fine, xfull_mat, mode, xreduced_mat, &
               x_temp_full_mat)

      ! The multiple rhs version of VecISCopyLocalWrapper - pulls the fine or
      ! coarse rows out of a dense block (SCATTER_REVERSE) or puts them back
      ! (SCATTER_FORWARD)
      ! Relies on having pre-built some things with the routine create_VecISCopyLocalWrapper

      ! ~~~~~~~~~~
      ! Input
      type(air_multigrid_data), intent(in) :: air_data
      integer, intent(in)                  :: our_level
      logical, intent(in)                  :: fine
      type(tMat), intent(inout)            :: xfull_mat, xreduced_mat
      type(tMat), optional, intent(inout)  :: x_temp_full_mat
      ScatterMode, intent(in)              :: mode

      PetscErrorCode :: ierr
#if defined(PETSC_HAVE_KOKKOS)
      integer(c_long_long) :: xfull_array, xreduced_array
      integer :: fine_int, mode_int, errorcode
      type(tMat) :: temp_mat
      PetscReal :: normy
#endif
      ! ~~~~~~~~~~

      ! On gpus without kokkos we can't touch the local arrays, so instead we use
      ! the identity restrictors/prolongators built in create_VecISCopyLocalWrapper
      ! These are the block versions of the matmults the single rhs version does
      if (.NOT. air_data%fast_veciscopy_exists) then

         if (mode == SCATTER_REVERSE) then

            if (fine) then
               call mat_product_block(air_data%i_fine_full(our_level), xfull_mat, &
                        xreduced_mat, .FALSE., ierr)
            else
               call mat_product_block(air_data%i_coarse_full(our_level), xfull_mat, &
                        xreduced_mat, .FALSE., ierr)
            end if

         ! SCATTER FORWARD
         else

            if (fine) then
               ! Copy x but only the non-fine rows of x are non-zero
               ! ie get x_c but in a block of full size
               call mat_product_block(air_data%i_coarse_full_full(our_level), xfull_mat, &
                        x_temp_full_mat, .FALSE., ierr)
               ! There is no MatMultTransposeAdd equivalent for a product, so we do
               ! the transpose product on its own and add the untouched rows back
               call mat_product_block(air_data%i_fine_full(our_level), xreduced_mat, &
                        xfull_mat, .TRUE., ierr)
            else
               ! Copy x but only the non-coarse rows of x are non-zero
               ! ie get x_f but in a block of full size
               call mat_product_block(air_data%i_fine_full_full(our_level), xfull_mat, &
                        x_temp_full_mat, .FALSE., ierr)
               call mat_product_block(air_data%i_coarse_full(our_level), xreduced_mat, &
                        xfull_mat, .TRUE., ierr)
            end if

            call MatAXPY(xfull_mat, PFLARE_ONE, x_temp_full_mat, SAME_NONZERO_PATTERN, ierr)
         end if

      ! Otherwise copy the rows we want directly out of the dense arrays
      else

#if defined(PETSC_HAVE_KOKKOS)

         ! The single rhs version decides by the vec type of the vec it is handed,
         ! but we can't do that here - the dense blocks in the mg hierarchy are
         ! built by petsc with MatDuplicate, which does not propagate the vec type,
         ! so a block can be backed by kokkos data and still report a standard vec
         ! type. The device IS views existing is the condition that matters (they
         ! are only built for a kokkos mat type in create_VecISCopyLocalWrapper) and
         ! the dense arrays we get back below always live in the default kokkos
         ! memory space - there is no MATDENSEKOKKOS, the blocks are host MATDENSE
         ! on a host kokkos backend and MATDENSECUDA/HIP on a device one
         if (c_associated(air_data%kokkos_is_views_handle)) then

            if (mode == SCATTER_REVERSE) then
               mode_int = 1
            else
               mode_int = 0
            end if
            fine_int = 0
            if (fine) fine_int = 1

            ! SCATTER FORWARD only writes the fine (or coarse) rows of the full
            ! block, so we have to keep a copy of it to run the cpu version on
            if (kokkos_debug() .AND. mode /= SCATTER_REVERSE) then
               call MatDuplicate(xfull_mat, MAT_COPY_VALUES, temp_mat, ierr)
            end if

            xfull_array = xfull_mat%v
            xreduced_array = xreduced_mat%v
            call mat_iscopy_local_kokkos(air_data%kokkos_is_views_handle, our_level, fine_int, xfull_array, &
                     mode_int, xreduced_array)

            ! If debugging do a comparison between CPU and Kokkos results
            if (kokkos_debug()) then

               if (mode == SCATTER_REVERSE) then

                  call MatDuplicate(xreduced_mat, MAT_DO_NOT_COPY_VALUES, temp_mat, ierr)
                  call mat_iscopy_local_host(air_data, our_level, fine, xfull_mat, mode, temp_mat)
                  call MatAXPY(temp_mat, PFLARE_MINUS_ONE, xreduced_mat, SAME_NONZERO_PATTERN, ierr)

               else

                  call mat_iscopy_local_host(air_data, our_level, fine, temp_mat, mode, xreduced_mat)
                  call MatAXPY(temp_mat, PFLARE_MINUS_ONE, xfull_mat, SAME_NONZERO_PATTERN, ierr)

               end if

               call MatNorm(temp_mat, NORM_FROBENIUS, normy, ierr)
               if (normy .gt. PFLARE_TOL_MATFREE_13) then
                  print *, "Kokkos and CPU versions of MatISCopyLocalWrapper do not match"
                  call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
               end if
               call MatDestroy(temp_mat, ierr)

            end if

         else
            call mat_iscopy_local_host(air_data, our_level, fine, xfull_mat, mode, xreduced_mat)
         end if
#else
         call mat_iscopy_local_host(air_data, our_level, fine, xfull_mat, mode, xreduced_mat)
#endif
      end if

   end subroutine MatISCopyLocalWrapper

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine mat_iscopy_local_host(air_data, our_level, fine, xfull_mat, mode, xreduced_mat)

      ! The host version of MatISCopyLocalWrapper - copies the fine or coarse rows
      ! we want directly out of the local dense arrays

      ! ~~~~~~~~~~
      ! Input
      type(air_multigrid_data), intent(in) :: air_data
      integer, intent(in)                  :: our_level
      logical, intent(in)                  :: fine
      type(tMat), intent(inout)            :: xfull_mat, xreduced_mat
      ScatterMode, intent(in)              :: mode

      PetscErrorCode :: ierr
      PetscInt :: global_row_start, global_row_end_plus_one
      PetscInt :: i_loc, j_loc, local_row
      PetscInt, pointer :: is_pointer(:)
      PetscScalar, pointer :: full_array(:,:), reduced_array(:,:)
      type(tIS) :: is_local
      ! ~~~~~~~~~~

      if (fine) then
         is_local = air_data%is_fine_index(our_level)
      else
         is_local = air_data%is_coarse_index(our_level)
      end if

      ! The IS holds global indices
      call MatGetOwnershipRange(xfull_mat, global_row_start, global_row_end_plus_one, ierr)
      call ISGetIndices(is_local, is_pointer, ierr)

      ! The petsc fortran interface only hands back a dense array when the
      ! leading dimension matches the number of local rows, so the arrays here
      ! are already (local rows, global columns) and need no separate lda
      if (mode == SCATTER_REVERSE) then

         call MatDenseGetArrayRead(xfull_mat, full_array, ierr)
         call MatDenseGetArray(xreduced_mat, reduced_array, ierr)

         do j_loc = 1, size(reduced_array, 2)
            do i_loc = 1, size(reduced_array, 1)
               local_row = is_pointer(i_loc) - global_row_start + 1
               reduced_array(i_loc, j_loc) = full_array(local_row, j_loc)
            end do
         end do

         call MatDenseRestoreArray(xreduced_mat, reduced_array, ierr)
         call MatDenseRestoreArrayRead(xfull_mat, full_array, ierr)

      ! SCATTER FORWARD - only the fine (or coarse) rows of the full block
      ! are written, everything else is left alone
      else

         call MatDenseGetArrayRead(xreduced_mat, reduced_array, ierr)
         call MatDenseGetArray(xfull_mat, full_array, ierr)

         do j_loc = 1, size(reduced_array, 2)
            do i_loc = 1, size(reduced_array, 1)
               local_row = is_pointer(i_loc) - global_row_start + 1
               full_array(local_row, j_loc) = reduced_array(i_loc, j_loc)
            end do
         end do

         call MatDenseRestoreArray(xfull_mat, full_array, ierr)
         call MatDenseRestoreArrayRead(xreduced_mat, reduced_array, ierr)

      end if

      call ISRestoreIndices(is_local, is_pointer, ierr)

   end subroutine mat_iscopy_local_host

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine mg_coarse_shell_block_apply(pc, x, y, ierr)

      ! The multiple rhs version of mg_coarse_shell_apply - applies our polynomial
      ! approximate inverse of the coarse matrix to a whole dense block

      ! ~~~~~~
      type(tPC)                             :: pc
      type(tMat)                            :: x, y
      PetscErrorCode, intent(out)           :: ierr

      type(air_multigrid_data), pointer     :: air_data => null()
      ! ~~~~~~

      ierr = 0
      call PCShellGetContext(pc, air_data, ierr)
      call apply_inverse_block(air_data%inv_A_ff(air_data%no_levels), x, y, ierr)

   end subroutine mg_coarse_shell_block_apply

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine mg_smooth_shell_block_apply(pc, x, y, ierr)

      ! The multiple rhs version of mg_smooth_shell_apply

      ! ~~~~~~
      type(tPC)                             :: pc
      type(tMat)                            :: x, y
      PetscErrorCode, intent(out)           :: ierr

      type(tMat) :: mat, pmat
      ! ~~~~~~

      ierr = 0
      call PCGetOperators(pc, mat, pmat, ierr)
      call apply_inverse_block(pmat, x, y, ierr)

   end subroutine mg_smooth_shell_block_apply

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine mg_FC_block_richardson(pc, b, x, r, rtol, abstol, dtol, maxits, guess_zero, its, conv_reason, ierr)

      ! The multiple rhs version of mg_FC_point_richardson - applies an FC block
      ! richardson to a whole dense block of right hand sides
      ! r is the block of work vectors and may be null, we never use it (just like
      ! the single rhs version never uses its work vector)

      ! ~~~~~~
      type(tPC) :: pc
      type(tMat) :: b, x, r
      PetscReal :: rtol, abstol, dtol
      PetscInt :: maxits, its
      PetscBool :: guess_zero
      PCRichardsonConvergedReason :: conv_reason
      PetscErrorCode :: ierr

      type(tMat) :: mat, pmat
      integer :: our_level, errorcode, i, smooth_its
      type(mat_ctxtype), pointer :: mat_ctx=>null()
      type(air_multigrid_data), pointer :: air_data
      PetscBool :: first_smooth

      ! ~~~~~~

      ! Set these for output
      ! have to return zero here!
      ierr = 0
      its = maxits
      conv_reason = PCRICHARDSON_CONVERGED_ITS;

      ! Can come in here with zero maxits, have to do nothing
      if (maxits == 0) return
      if (maxits /= 1) then
         print *, "To change the number of smooths adjust smooth_order"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! Get the level
      call PCGetOperators(pc, mat, pmat, ierr)
      ! Get what level we are on
      call MatShellGetContext(mat, mat_ctx, ierr)
      our_level = mat_ctx%our_level
      air_data => mat_ctx%air_data

      ! The first time we go through any smooth, we need to pull out x_f and/or x_c
      first_smooth = PETSC_TRUE

      ! Loop over all the smooths we need to do
      do i = 1, size(air_data%smooth_order_levels(our_level)%array)

         smooth_its = air_data%smooth_order_levels(our_level)%array(i)

         if (smooth_its == 0) exit

         ! Do consecutive F point smooths
         if (smooth_its > 0) then

            call f_smooths_block(b, x, guess_zero, first_smooth, air_data, our_level, smooth_its)

         ! Do consecutive C point smooths
         else

            call c_smooths_block(b, x, guess_zero, first_smooth, air_data, our_level, abs(smooth_its))
         end if

         ! Once we've done our first smooth, we can use the existing values
         first_smooth = PETSC_FALSE

      end do

      ! Now technically there should be a new residual that we put into r after this is done
      ! but I don't think it matters, as it is the solution that is interpolated up
      ! and the richardson on the next level up computes its own F-point residual
      ! and the norm type is none on the mg levels, as we just do maxits

      ! have to return zero here!
      ierr = 0

   end subroutine mg_FC_block_richardson

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine f_smooths_block(b, x, guess_zero, first_smooth, air_data, our_level, its)

      ! The multiple rhs version of f_smooths - applies consecutive F smooths to a
      ! whole dense block of right hand sides
      ! This is the same arithmetic as f_smooths with the matvecs replaced by
      ! sparse matrix - dense matrix products

      ! ~~~~~~
      type(tMat), intent(inout)               :: b, x
      type(air_multigrid_data), intent(inout) :: air_data
      integer, intent(in)                     :: our_level, its
      PetscBool, intent(in)                   :: guess_zero, first_smooth

      PetscErrorCode :: ierr
      integer :: f_its

      ! ~~~~~~

      ! Get out just the fine rows from b - this is b_f
      call MatISCopyLocalWrapper(air_data, our_level, .TRUE., b, &
               SCATTER_REVERSE, air_data%block_temp_fine(AIR_MAT_RHS)%array(our_level))

      ! If we haven't done any smooth before calling this F point smooth
      ! we need to pull out x_c^0 and x_f^0
      if (first_smooth) then

         ! Get out just the fine rows from x - this is x_f^0
         call MatISCopyLocalWrapper(air_data, our_level, .TRUE., x, &
                  SCATTER_REVERSE, air_data%block_temp_fine(AIR_MAT_SOL)%array(our_level))

         ! Get the coarse rows from x - this is x_c^0
         call MatISCopyLocalWrapper(air_data, our_level, .FALSE., x, &
                  SCATTER_REVERSE, air_data%block_temp_coarse(AIR_MAT_SOL)%array(our_level))

      end if

      ! Compute Afc * x_c^0 - this never changes
      ! The product was attached to the scratch by setup_air_block_products so
      ! only the numeric phase runs, here and below
      call MatProductNumeric(air_data%block_temp_fine(AIR_MAT_OFF_DIAG)%array(our_level), ierr)

      ! This is b_f - A_fc * x_c^0 - this never changes
      call MatAXPY(air_data%block_temp_fine(AIR_MAT_RHS)%array(our_level), PFLARE_MINUS_ONE, &
               air_data%block_temp_fine(AIR_MAT_OFF_DIAG)%array(our_level), SAME_NONZERO_PATTERN, ierr)

      ! Do all the consecutive F smooths
      do f_its = 1, its

         ! Then A_ff * x_f^n - this changes at each richardson iteration
         call MatProductNumeric(air_data%block_temp_fine(AIR_MAT_RESIDUAL)%array(our_level), ierr)

         ! This is b_f - A_fc * x_c - A_ff * x_f^n
         call MatAYPX(air_data%block_temp_fine(AIR_MAT_RESIDUAL)%array(our_level), PFLARE_MINUS_ONE, &
                  air_data%block_temp_fine(AIR_MAT_RHS)%array(our_level), SAME_NONZERO_PATTERN, ierr)

         ! ! Compute A_ff^{-1} ( b_f - A_fc * x_c - A_ff * x_f^n)
         call apply_inverse_block_cached(air_data%inv_A_ff(our_level), &
                  air_data%block_temp_fine(AIR_MAT_RESIDUAL)%array(our_level), &
                  air_data%block_temp_fine(AIR_MAT_TEMP)%array(our_level), ierr)

         ! Compute x_f^n + A_ff^{-1} ( b_f - A_fc * x_c - A_ff * x_f^n)
         call MatAXPY(air_data%block_temp_fine(AIR_MAT_SOL)%array(our_level), PFLARE_ONE, &
                  air_data%block_temp_fine(AIR_MAT_TEMP)%array(our_level), SAME_NONZERO_PATTERN, ierr)

      end do

      ! ~~~~~~~~
      ! Reverse put fine x_f back into x
      ! ~~~~~~~~
      call MatISCopyLocalWrapper(air_data, our_level, .TRUE., x, &
               SCATTER_FORWARD, air_data%block_temp_fine(AIR_MAT_SOL)%array(our_level), &
               air_data%block_temp_full(1)%array(our_level))

   end subroutine f_smooths_block

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine c_smooths_block(b, x, guess_zero, first_smooth, air_data, our_level, its)

      ! The multiple rhs version of c_smooths - applies consecutive C smooths to a
      ! whole dense block of right hand sides

      ! ~~~~~~
      type(tMat), intent(inout)               :: b, x
      type(air_multigrid_data), intent(inout) :: air_data
      integer, intent(in)                     :: our_level, its
      PetscBool, intent(in)                   :: guess_zero, first_smooth

      PetscErrorCode :: ierr
      integer :: c_its

      ! ~~~~~~

      ! Get out just the coarse rows from b - this is b_c
      call MatISCopyLocalWrapper(air_data, our_level, .FALSE., b, &
               SCATTER_REVERSE, air_data%block_temp_coarse(AIR_MAT_RHS)%array(our_level))

      ! If we haven't done any smooth before calling this C point smooth
      ! we need to pull out x_c^0 and x_f^0
      if (first_smooth) then

            ! Get out just the fine rows from x - this is x_f^0
         call MatISCopyLocalWrapper(air_data, our_level, .TRUE., x, &
                  SCATTER_REVERSE, air_data%block_temp_fine(AIR_MAT_SOL)%array(our_level))

         ! Get the coarse rows from x - this is x_c^0
         call MatISCopyLocalWrapper(air_data, our_level, .FALSE., x, &
                  SCATTER_REVERSE, air_data%block_temp_coarse(AIR_MAT_SOL)%array(our_level))

      end if

      ! Compute Acf * x_f^0 - this never changes
      ! The product was attached to the scratch by setup_air_block_products so
      ! only the numeric phase runs, here and below
      call MatProductNumeric(air_data%block_temp_coarse(AIR_MAT_OFF_DIAG)%array(our_level), ierr)
      ! This is b_c - A_cf * x_f^0 - this never changes
      call MatAXPY(air_data%block_temp_coarse(AIR_MAT_RHS)%array(our_level), PFLARE_MINUS_ONE, &
               air_data%block_temp_coarse(AIR_MAT_OFF_DIAG)%array(our_level), SAME_NONZERO_PATTERN, ierr)

      ! Do all the consecutive C smooths
      do c_its = 1, its

         ! Then A_cc * x_c^n - this changes at each richardson iteration
         call MatProductNumeric(air_data%block_temp_coarse(AIR_MAT_RESIDUAL)%array(our_level), ierr)

         ! This is b_c - A_cf * x_f^0 - A_cc * x_c^n
         call MatAYPX(air_data%block_temp_coarse(AIR_MAT_RESIDUAL)%array(our_level), PFLARE_MINUS_ONE, &
                  air_data%block_temp_coarse(AIR_MAT_RHS)%array(our_level), SAME_NONZERO_PATTERN, ierr)

         ! ! Compute A_cc^{-1} (b_c - A_cf * x_f^0 - A_cc * x_c^n)
         call apply_inverse_block_cached(air_data%inv_A_cc(our_level), &
                  air_data%block_temp_coarse(AIR_MAT_RESIDUAL)%array(our_level), &
                  air_data%block_temp_coarse(AIR_MAT_TEMP)%array(our_level), ierr)

         ! Compute x_c^n + A_cc^{-1} (b_c - A_cf * x_f^0 - A_cc * x_c^n)
         call MatAXPY(air_data%block_temp_coarse(AIR_MAT_SOL)%array(our_level), PFLARE_ONE, &
                  air_data%block_temp_coarse(AIR_MAT_TEMP)%array(our_level), SAME_NONZERO_PATTERN, ierr)

      end do

      ! ~~~~~~~~
      ! Reverse put coarse x_c back into x
      ! ~~~~~~~~
      call MatISCopyLocalWrapper(air_data, our_level, .FALSE., x, &
               SCATTER_FORWARD, air_data%block_temp_coarse(AIR_MAT_SOL)%array(our_level), &
               air_data%block_temp_full(1)%array(our_level))

   end subroutine c_smooths_block

! -------------------------------------------------------------------------------------------------------------------------------

end module fc_smooth_block
