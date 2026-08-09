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
               call MatDestroy(temp_mat, ierr)
            end if
         end do
         mat_ctx%mf_temp_mat_ncols = -1
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

end module matshell_data_type

