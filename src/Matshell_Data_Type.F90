module matshell_data_type

   use petscmat
   use air_data_type
   use pflare_parameters, only: MF_VEC_TEMP, MF_VEC_TEMP_TWO, MF_VEC_TEMP_THREE, &
         MF_VEC_DIAG, MF_VEC_RHS, MF_MAT_TEMP, MF_MAT_TEMP_TWO, MF_MAT_TEMP_THREE, &
         MF_MAT_RHS

#include "petsc/finclude/petscmat.h"   

   implicit none

   public

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! This is our context type for matshells
   ! This has to be in a separate file to matshell

   type :: mat_ctxtype
      integer :: our_level = -1
      PetscReal, dimension(:), pointer :: coefficients => null()
      logical                     :: own_coefficients = .FALSE.
      PetscReal, dimension(:), pointer :: real_roots => null()
      PetscReal, dimension(:), pointer :: imag_roots => null()
      type(tMat) :: mat, mat_scaled
      ! Whether the inner mat_scaled shell applies I - D^-1 A (the Neumann polynomial)
      ! rather than D^-1 A (the diagonally scaled gmres polynomials)
      logical :: neumann_inner = .FALSE.
      ! Temporary vectors we use
      type(tVec), dimension(5) :: mf_temp_vec
      ! Temporary dense matrices we use during a multiple rhs (block) apply
      ! These are lazily created from the input block of rhs, as the number of
      ! columns isn't known until the apply happens
      type(tMat), dimension(4) :: mf_temp_mat
      ! The number of global columns the cached mf_temp_mat's were built with
      PetscInt :: mf_temp_mat_ncols = -1
      ! Whether the block apply kernels have their products attached to the
      ! mf_temp_mat scratch, and the mat they were attached with - the products
      ! themselves keep that mat alive, so the handle comparison in the kernels
      ! is safe even if the context is later pointed at a different mat
      logical :: mf_products_attached = .FALSE.
      type(tMat) :: mf_product_mat
      ! The reciprocal of mf_temp_vec(MF_VEC_DIAG), only used by the block applies
      ! of the diagonally scaled polynomials
      type(tVec) :: mf_vec_diag_recip
      type(air_multigrid_data), pointer :: air_data => null()

   end type mat_ctxtype

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine ensure_block_temp_mats(mat_ctx, x_mat, n_temps, ierr, need_rhs)

      ! Makes sure the cached dense scratch blocks in mat_ctx%mf_temp_mat
      ! match the block of rhs, x_mat, we've been given
      ! Slots 1 to n_temps are created (MF_MAT_TEMP, MF_MAT_TEMP_TWO, MF_MAT_TEMP_THREE)
      ! and the optional need_rhs also creates the MF_MAT_RHS slot, which is where the
      ! diagonally scaled block applies store D^-1 X
      ! We can't just always create all four, as the number of temporaries needed
      ! depends on the polynomial being applied and these blocks aren't small

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! Input
      type(mat_ctxtype), intent(inout) :: mat_ctx
      type(tMat), intent(in)           :: x_mat
      integer, intent(in)              :: n_temps
      PetscErrorCode, intent(inout)    :: ierr
      logical, optional, intent(in)    :: need_rhs

      ! Local
      integer :: i_loc
      logical :: rebuild, want_rhs
      PetscInt :: global_rows, global_cols
      MatType :: x_type, temp_type
      type(tMat) :: temp_mat

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      want_rhs = .FALSE.
      if (present(need_rhs)) want_rhs = need_rhs

      call MatGetSize(x_mat, global_rows, global_cols, ierr)
      call MatGetType(x_mat, x_type, ierr)

      ! If the number of columns has changed we have to start again
      rebuild = mat_ctx%mf_temp_mat_ncols /= global_cols

      ! The type of block we're given can also change between applies
      if (.NOT. rebuild) then
         do i_loc = 1, size(mat_ctx%mf_temp_mat)
            temp_mat = mat_ctx%mf_temp_mat(i_loc)
            if (.NOT. PetscObjectIsNull(temp_mat)) then
               call MatGetType(temp_mat, temp_type, ierr)
               if (temp_type /= x_type) rebuild = .TRUE.
               exit
            end if
         end do
      end if

      if (rebuild) then
         do i_loc = 1, size(mat_ctx%mf_temp_mat)
            temp_mat = mat_ctx%mf_temp_mat(i_loc)
            if (.NOT. PetscObjectIsNull(temp_mat)) then
               ! The block kernels attach products that hold references between the
               ! scratch mats - the horner ping-pong pair reference each other, so
               ! plain destroys would leave a reference cycle alive - clear the
               ! products first to break it
               call MatProductClear(temp_mat, ierr)
               call MatDestroy(temp_mat, ierr)
               ! MatDestroy nulls the local copy of the handle, not the slot in the
               ! context - copy it back or the slot would be left dangling and the
               ! creation below would skip it
               mat_ctx%mf_temp_mat(i_loc) = temp_mat
            end if
         end do
         mat_ctx%mf_temp_mat_ncols = -1
         ! Any products the block apply kernels had attached died with the
         ! scratch, so they have to attach them again
         mat_ctx%mf_products_attached = .FALSE.
      end if

      ! Now create any of the slots we need that don't exist
      do i_loc = 1, n_temps
         temp_mat = mat_ctx%mf_temp_mat(i_loc)
         if (PetscObjectIsNull(temp_mat)) then
            call MatDuplicate(x_mat, MAT_DO_NOT_COPY_VALUES, mat_ctx%mf_temp_mat(i_loc), ierr)
         end if
      end do
      if (want_rhs) then
         temp_mat = mat_ctx%mf_temp_mat(MF_MAT_RHS)
         if (PetscObjectIsNull(temp_mat)) then
            call MatDuplicate(x_mat, MAT_DO_NOT_COPY_VALUES, mat_ctx%mf_temp_mat(MF_MAT_RHS), ierr)
         end if
      end if

      mat_ctx%mf_temp_mat_ncols = global_cols

   end subroutine ensure_block_temp_mats

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine attach_mat_product(mat, x_mat, transpose_mat, y_mat, ierr, has_product)

      ! Attaches y_mat = mat * x_mat (or mat^T * x_mat) as a product and runs the
      ! symbolic phase, leaving the product bookkeeping alive on y_mat so repeat
      ! computations only have to run MatProductNumeric
      ! y_mat therefore keeps references to mat and x_mat until the product is
      ! cleared or y_mat is destroyed
      ! If has_product is present it comes back false (with the product cleared
      ! and no symbolic run) when there is no product implementation for these
      ! types and the caller has to fall back; without it the product is assumed
      ! to exist
      ! This lives here rather than in petsc_helper as this file is compiled
      ! before petsc_helper

      ! ~~~~~~
      type(tMat), intent(in)         :: mat
      type(tMat)                     :: x_mat, y_mat
      logical, intent(in)            :: transpose_mat
      PetscErrorCode, intent(inout)  :: ierr
      logical, optional, intent(out) :: has_product

      PetscBool :: has_op
      ! ~~~~~~

      call MatProductCreateWithMat(mat, x_mat, PETSC_NULL_MAT, y_mat, ierr)
      if (transpose_mat) then
         call MatProductSetType(y_mat, MATPRODUCT_AtB, ierr)
      else
         call MatProductSetType(y_mat, MATPRODUCT_AB, ierr)
      end if
      call MatProductSetFromOptions(y_mat, ierr)
      if (present(has_product)) then
         call MatHasOperation(y_mat, MATOP_PRODUCTSYMBOLIC, has_op, ierr)
         has_product = has_op .eqv. PETSC_TRUE
         if (.NOT. has_product) then
            call MatProductClear(y_mat, ierr)
            return
         end if
      end if
      call MatProductSymbolic(y_mat, ierr)

   end subroutine attach_mat_product

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine ensure_block_kernel_products(mat_ctx, second_target, attached)

      ! Makes sure the two products a matrix-free block apply kernel runs are
      ! attached to the dense temporaries in the context:
      !    mf_temp_mat(MF_MAT_TEMP_TWO) = mat * mf_temp_mat(MF_MAT_TEMP)
      !    mf_temp_mat(second_target)   = mat * mf_temp_mat(MF_MAT_TEMP_TWO)
      ! The horner kernel ping-pongs between its two temporaries so its second
      ! target is MF_MAT_TEMP, while the newton kernel's is MF_MAT_TEMP_THREE
      ! The products stay attached between applies - they die with the scratch
      ! whenever ensure_block_temp_mats rebuilds it (which resets
      ! mf_products_attached), and are rebuilt here if the mat in the context has
      ! changed identity (the attached products keep the old mat alive, so the
      ! handle comparison is safe)

      ! attached comes back false if there is no product for these types and the
      ! caller has to fall back to a column by column apply

      ! ~~~~~~
      type(mat_ctxtype), intent(inout) :: mat_ctx
      integer, intent(in)              :: second_target
      logical, intent(out)             :: attached

      logical :: has_product
      PetscErrorCode :: ierr
      ! ~~~~~~

      attached = .TRUE.

      ! If the mat has been swapped out from under us the attached products
      ! would still run with the old one - clear them and attach again
      if (mat_ctx%mf_products_attached) then
         if (mat_ctx%mf_product_mat%v /= mat_ctx%mat%v) then
            call MatProductClear(mat_ctx%mf_temp_mat(MF_MAT_TEMP_TWO), ierr)
            call MatProductClear(mat_ctx%mf_temp_mat(second_target), ierr)
            mat_ctx%mf_products_attached = .FALSE.
         end if
      end if

      if (.NOT. mat_ctx%mf_products_attached) then

         ! temp_two = mat * temp - if there is no product available for these
         ! types the caller does a column by column apply instead
         call attach_mat_product(mat_ctx%mat, mat_ctx%mf_temp_mat(MF_MAT_TEMP), .FALSE., &
                  mat_ctx%mf_temp_mat(MF_MAT_TEMP_TWO), ierr, has_product=has_product)
         if (.NOT. has_product) then
            attached = .FALSE.
            return
         end if

         ! second_target = mat * temp_two
         ! The types are the same as the product above, so we know this one exists
         call attach_mat_product(mat_ctx%mat, mat_ctx%mf_temp_mat(MF_MAT_TEMP_TWO), .FALSE., &
                  mat_ctx%mf_temp_mat(second_target), ierr)

         mat_ctx%mf_products_attached = .TRUE.
         mat_ctx%mf_product_mat = mat_ctx%mat
      end if

   end subroutine ensure_block_kernel_products

! -------------------------------------------------------------------------------------------------------------------------------

end module matshell_data_type

