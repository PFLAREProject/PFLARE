module port_interfaces_grid

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! The Fortran that is still to be ported calls the grid transfer routines and the
   ! near nullspace constraints, which have been ported to C (see
   ! docs/dev/fortran_removal_plan.md), through this module, in the same way as through
   ! port_interfaces
   ! This is a separate module because the PFLARE_PORT_DEBUG comparisons use the
   ! Fortran reference implementations in src/fref, which themselves use
   ! port_interfaces (so port_interfaces cannot use them)
   ! The routines keep the names and arguments of the Fortran routines they replace
   ! The arrays of Vecs cross as arrays of handles with their sizes, and get_near_nullspace
   ! copies the arrays the C allocates into the Fortran allocatables
   ! With PFLARE_PORT_DEBUG the routines also run the Fortran reference on a copy of
   ! anything the C changes and abort if the results differ: matrices row by row and
   ! bitwise (port_compare_mats), index sets with ISEqualUnsorted and vectors with VecEqual
   ! This module is deleted with port_interfaces once all the callers are in C
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------

   use iso_c_binding
   use petscksp
   use port_interfaces, only: port_check, mat_to_c, mat_from_c
   ! The Fortran reference implementations, under #ifdef rather than #if defined
   ! so that make depend (makedepf90 -D PFLARE_PORT_DEBUG) sees these uses only
   ! in the port debug build
#ifdef PFLARE_PORT_DEBUG
   use grid_transfer, only: &
         generate_one_point_with_one_entry_from_sparse_fref => generate_one_point_with_one_entry_from_sparse, &
         compute_P_from_W_fref => compute_P_from_W, compute_R_from_Z_fref => compute_R_from_Z
   use grid_transfer_improve, only: improve_w_fref => improve_w, improve_z_fref => improve_z
   use constrain_z_or_w, only: get_near_nullspace_fref => get_near_nullspace, &
         smooth_near_nullspace_fref => smooth_near_nullspace, &
         constrain_grid_transfer_fref => constrain_grid_transfer
   use port_interfaces, only: port_compare_mats, port_copy_mat
#endif

#include "petsc/finclude/petscksp.h"

   implicit none

   private

   ! Grid_Transfer.c
   public :: generate_one_point_with_one_entry_from_sparse, compute_P_from_W, compute_R_from_Z
   ! Grid_Transfer_Improve.c
   public :: improve_w, improve_z
   ! Constrain_Z_or_W.c
   public :: get_near_nullspace, smooth_near_nullspace, constrain_grid_transfer

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! C_PETSc_Routines.c
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function PetscFree_c(ptr) &
         bind(c, name="PetscFree_c")
         use iso_c_binding
         type(c_ptr) :: ptr
         PetscErrorCode :: PetscFree_c
      end function PetscFree_c
   end interface

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Grid_Transfer.c
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function generate_one_point_with_one_entry_from_sparse_c(input_mat, output_mat) &
         bind(c, name="generate_one_point_with_one_entry_from_sparse")
         use iso_c_binding
         integer(c_long_long), value :: input_mat
         integer(c_long_long) :: output_mat
         PetscErrorCode :: generate_one_point_with_one_entry_from_sparse_c
      end function generate_one_point_with_one_entry_from_sparse_c
   end interface

   interface
      function compute_P_from_W_c(W, global_row_start, is_fine, is_coarse, identity, reuse, P) &
         bind(c, name="compute_P_from_W")
         use iso_c_binding
         integer(c_long_long), value :: W, is_fine, is_coarse
         PetscInt, value :: global_row_start
         PetscBool, value :: identity, reuse
         integer(c_long_long) :: P
         PetscErrorCode :: compute_P_from_W_c
      end function compute_P_from_W_c
   end interface

   interface
      function compute_R_from_Z_c(Z, global_row_start, is_fine, is_coarse, orig_fine_col_indices, &
                  identity, reuse, R) &
         bind(c, name="compute_R_from_Z")
         use iso_c_binding
         integer(c_long_long), value :: Z, is_fine, is_coarse
         PetscInt, value :: global_row_start
         integer(c_long_long) :: orig_fine_col_indices
         PetscBool, value :: identity, reuse
         integer(c_long_long) :: R
         PetscErrorCode :: compute_R_from_Z_c
      end function compute_R_from_Z_c
   end interface

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Grid_Transfer_Improve.c
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function improve_w_c(W, A_ff, A_fc, A_ff_inv, reuse_mat, reuse_mat_two, its, reuse_sparsity) &
         bind(c, name="improve_w")
         use iso_c_binding
         integer(c_long_long), value :: W, A_ff, A_fc, A_ff_inv
         integer(c_long_long) :: reuse_mat, reuse_mat_two
         integer(c_int), value :: its
         PetscBool, value :: reuse_sparsity
         PetscErrorCode :: improve_w_c
      end function improve_w_c
   end interface

   interface
      function improve_z_c(Z, A_ff, A_cf, A_ff_inv, reuse_mat, reuse_mat_two, its, reuse_sparsity) &
         bind(c, name="improve_z")
         use iso_c_binding
         integer(c_long_long), value :: Z, A_ff, A_cf, A_ff_inv
         integer(c_long_long) :: reuse_mat, reuse_mat_two
         integer(c_int), value :: its
         PetscBool, value :: reuse_sparsity
         PetscErrorCode :: improve_z_c
      end function improve_z_c
   end interface

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Constrain_Z_or_W.c
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function get_near_nullspace_c(input_mat, left, right, left_null_vecs, right_null_vecs, null_vecs_size) &
         bind(c, name="get_near_nullspace")
         use iso_c_binding
         integer(c_long_long), value :: input_mat
         PetscBool, value :: left, right
         ! The arrays of Vecs the C allocates
         type(c_ptr) :: left_null_vecs, right_null_vecs
         PetscInt :: null_vecs_size
         PetscErrorCode :: get_near_nullspace_c
      end function get_near_nullspace_c
   end interface

   interface
      function smooth_near_nullspace_c(input_mat, left, right, left_null_vecs, left_null_vecs_size, &
                  right_null_vecs, right_null_vecs_size) &
         bind(c, name="smooth_near_nullspace")
         use iso_c_binding
         integer(c_long_long), value :: input_mat
         PetscBool, value :: left, right
         PetscInt, value :: left_null_vecs_size, right_null_vecs_size
         integer(c_long_long), dimension(*) :: left_null_vecs, right_null_vecs
         PetscErrorCode :: smooth_near_nullspace_c
      end function smooth_near_nullspace_c
   end interface

   interface
      function constrain_grid_transfer_c(z_or_w, is_z, null_vecs_f, null_vecs_c, null_vecs_size) &
         bind(c, name="constrain_grid_transfer")
         use iso_c_binding
         integer(c_long_long) :: z_or_w
         PetscBool, value :: is_z
         PetscInt, value :: null_vecs_size
         integer(c_long_long), dimension(*) :: null_vecs_f, null_vecs_c
         PetscErrorCode :: constrain_grid_transfer_c
      end function constrain_grid_transfer_c
   end interface

   contains

! -------------------------------------------------------------------------------------------------------------------------------

   logical function mat_is_null(m)

      ! PetscObjectIsNull of a matrix (the macro expands past the line length
      ! with the longer names of the callers)

      ! ~~~~~~
      type(tMat), intent(in) :: m
      ! ~~~~~~

      mat_is_null = PetscObjectIsNull(m)

   end function mat_is_null

! -------------------------------------------------------------------------------------------------------------------------------

   logical function vec_is_null(v)

      ! PetscObjectIsNull of a vector

      ! ~~~~~~
      type(tVec), intent(in) :: v
      ! ~~~~~~

      vec_is_null = PetscObjectIsNull(v)

   end function vec_is_null

! -------------------------------------------------------------------------------------------------------------------------------

   logical function is_is_null(s)

      ! PetscObjectIsNull of an index set

      ! ~~~~~~
      type(tIS), intent(in) :: s
      ! ~~~~~~

      is_is_null = PetscObjectIsNull(s)

   end function is_is_null

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine vecs_to_c(vecs, handles)

      ! The handles of an array of Vecs to pass to the C, NULL for a Vec that has not
      ! been created

      ! ~~~~~~
      type(tVec), dimension(:), intent(in)                 :: vecs
      integer(c_long_long), dimension(size(vecs)), intent(out) :: handles

      integer :: i_loc
      ! ~~~~~~

      do i_loc = 1, size(vecs)
         if (vec_is_null(vecs(i_loc))) then
            handles(i_loc) = 0
         else
            handles(i_loc) = vecs(i_loc)%v
         end if
      end do

   end subroutine vecs_to_c

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine vecs_from_c(vecs_ptr, vecs_size, vecs)

      ! Allocates vecs with vecs_size entries and copies in the handles from the array
      ! the C allocated, which is freed. A NULL handle (a Vec the C did not create)
      ! leaves the entry as the null Fortran object it is after the allocate

      ! ~~~~~~
      type(c_ptr), intent(inout)                          :: vecs_ptr
      PetscInt, intent(in)                                :: vecs_size
      type(tVec), dimension(:), allocatable, intent(inout) :: vecs

      PetscErrorCode :: ierr
      integer :: i_loc
      integer(c_long_long), dimension(:), pointer :: handles
      ! ~~~~~~

      allocate(vecs(vecs_size))
      if (vecs_size > 0) then
         call c_f_pointer(vecs_ptr, handles, [vecs_size])
         do i_loc = 1, int(vecs_size)
            if (handles(i_loc) /= 0) vecs(i_loc)%v = handles(i_loc)
         end do
      end if
      ierr = PetscFree_c(vecs_ptr)
      call port_check(ierr, "PetscFree_c")

   end subroutine vecs_from_c

#if defined(PFLARE_PORT_DEBUG)
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine port_compare_reuse_mats(mat, mat_fref, routine)

      ! Aborts unless both matrices are null or both exist and are the same
      ! (port_compare_mats)

      ! ~~~~~~
      type(tMat), intent(in)       :: mat, mat_fref
      character(len=*), intent(in) :: routine

      integer :: errorcode
      ! ~~~~~~

      if (mat_is_null(mat) .neqv. mat_is_null(mat_fref)) then
         print *, "C and Fortran reference versions of ", routine, " do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      if (.NOT. mat_is_null(mat)) call port_compare_mats(mat, mat_fref, routine)

   end subroutine port_compare_reuse_mats

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine port_compare_null_vecs(vecs, vecs_fref, routine)

      ! Aborts unless the arrays of Vecs have the same size and each pair is either
      ! null in both or bitwise the same (VecEqual)

      ! ~~~~~~
      type(tVec), dimension(:), intent(in) :: vecs, vecs_fref
      character(len=*), intent(in)         :: routine

      PetscErrorCode :: ierr
      PetscBool :: vec_equal
      logical :: vecs_equal
      integer :: i_loc, errorcode
      ! ~~~~~~

      vecs_equal = size(vecs) == size(vecs_fref)
      if (vecs_equal) then
         do i_loc = 1, size(vecs)
            if (vec_is_null(vecs(i_loc)) .neqv. vec_is_null(vecs_fref(i_loc))) then
               vecs_equal = .FALSE.
            else if (.NOT. vec_is_null(vecs(i_loc))) then
               call VecEqual(vecs(i_loc), vecs_fref(i_loc), vec_equal, ierr)
               if (.NOT. vec_equal) vecs_equal = .FALSE.
            end if
         end do
      end if

      if (.NOT. vecs_equal) then
         print *, "C and Fortran reference versions of ", routine, " do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

   end subroutine port_compare_null_vecs

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine port_copy_null_vecs(vecs, copy, vecs_copy)

      ! Allocates vecs_copy with the size of vecs and, if copy, fills it with copies
      ! of the Vecs in vecs that exist (the others stay null)

      ! ~~~~~~
      type(tVec), dimension(:), intent(in)                 :: vecs
      logical, intent(in)                                  :: copy
      type(tVec), dimension(:), allocatable, intent(inout) :: vecs_copy

      PetscErrorCode :: ierr
      integer :: i_loc
      ! ~~~~~~

      allocate(vecs_copy(size(vecs)))
      if (.NOT. copy) return
      do i_loc = 1, size(vecs)
         if (vec_is_null(vecs(i_loc))) cycle
         call VecDuplicate(vecs(i_loc), vecs_copy(i_loc), ierr)
         call VecCopy(vecs(i_loc), vecs_copy(i_loc), ierr)
      end do

   end subroutine port_copy_null_vecs

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine port_destroy_null_vecs(vecs)

      ! Destroys the Vecs that exist in vecs and deallocates it

      ! ~~~~~~
      type(tVec), dimension(:), allocatable, intent(inout) :: vecs

      PetscErrorCode :: ierr
      integer :: i_loc
      ! ~~~~~~

      do i_loc = 1, size(vecs)
         if (.NOT. vec_is_null(vecs(i_loc))) call VecDestroy(vecs(i_loc), ierr)
      end do
      deallocate(vecs)

   end subroutine port_destroy_null_vecs
#endif

! -------------------------------------------------------------------------------------------------------------------------------
! Grid_Transfer.c
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine generate_one_point_with_one_entry_from_sparse(input_mat, output_mat)

      ! ~~~~~~
      type(tMat), intent(in) :: input_mat
      type(tMat), intent(inout) :: output_mat

      PetscErrorCode :: ierr
      integer(c_long_long) :: output_mat_handle
#if defined(PFLARE_PORT_DEBUG)
      type(tMat) :: output_mat_fref
#endif
      ! ~~~~~~

      output_mat_handle = mat_to_c(output_mat)
      ierr = generate_one_point_with_one_entry_from_sparse_c(input_mat%v, output_mat_handle)
      call port_check(ierr, "generate_one_point_with_one_entry_from_sparse")
      call mat_from_c(output_mat_handle, output_mat)

#if defined(PFLARE_PORT_DEBUG)
      PetscObjectNullify(output_mat_fref)
      call generate_one_point_with_one_entry_from_sparse_fref(input_mat, output_mat_fref)
      call port_compare_mats(output_mat, output_mat_fref, "generate_one_point_with_one_entry_from_sparse")
      call MatDestroy(output_mat_fref, ierr)
#endif

   end subroutine generate_one_point_with_one_entry_from_sparse

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine compute_P_from_W(W, global_row_start, is_fine, is_coarse, identity, reuse, P)

      ! ~~~~~~
      type(tMat), intent(inout) :: W, P
      type(tIS), intent(in)     :: is_fine, is_coarse
      PetscInt, intent(in)      :: global_row_start
      logical, intent(in) :: identity, reuse

      PetscErrorCode :: ierr
      PetscBool :: identity_bool, reuse_bool
      integer(c_long_long) :: P_handle
#if defined(PFLARE_PORT_DEBUG)
      type(tMat) :: P_fref
#endif
      ! ~~~~~~

#if defined(PFLARE_PORT_DEBUG)
      ! When reusing the C refills P in place, so give the reference a copy of the input
      PetscObjectNullify(P_fref)
      if (reuse) call port_copy_mat(P, P_fref)
#endif

      identity_bool = identity
      reuse_bool = reuse
      P_handle = mat_to_c(P)
      ierr = compute_P_from_W_c(W%v, global_row_start, is_fine%v, is_coarse%v, identity_bool, reuse_bool, P_handle)
      call port_check(ierr, "compute_P_from_W")
      call mat_from_c(P_handle, P)

#if defined(PFLARE_PORT_DEBUG)
      call compute_P_from_W_fref(W, global_row_start, is_fine, is_coarse, identity, reuse, P_fref)
      call port_compare_mats(P, P_fref, "compute_P_from_W")
      call MatDestroy(P_fref, ierr)
#endif

   end subroutine compute_P_from_W

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine compute_R_from_Z(Z, global_row_start, is_fine, is_coarse, &
                     orig_fine_col_indices, &
                     identity, reuse, &
                     R)

      ! ~~~~~~
      type(tMat), intent(inout) :: Z, R
      PetscInt, intent(in)      :: global_row_start
      type(tIS), intent(in)     :: is_fine, is_coarse
      type(tIS), intent(inout)  :: orig_fine_col_indices
      logical, intent(in) :: identity, reuse

      PetscErrorCode :: ierr
      PetscBool :: identity_bool, reuse_bool
      integer(c_long_long) :: R_handle, orig_fine_col_indices_handle
#if defined(PFLARE_PORT_DEBUG)
      type(tMat) :: R_fref
      type(tIS) :: orig_fine_col_indices_fref
      logical :: orig_fine_col_indices_null
      PetscBool :: is_equal
      integer :: errorcode
#endif
      ! ~~~~~~

#if defined(PFLARE_PORT_DEBUG)
      ! When reusing the C refills R in place, so give the reference a copy of the input
      PetscObjectNullify(R_fref)
      if (reuse) call port_copy_mat(R, R_fref)
      ! The C creates orig_fine_col_indices if it is null, which the reference then has to
      ! do too, and otherwise only reads it
      orig_fine_col_indices_null = is_is_null(orig_fine_col_indices)
#endif

      identity_bool = identity
      reuse_bool = reuse
      R_handle = mat_to_c(R)
      if (is_is_null(orig_fine_col_indices)) then
         orig_fine_col_indices_handle = 0
      else
         orig_fine_col_indices_handle = orig_fine_col_indices%v
      end if
      ierr = compute_R_from_Z_c(Z%v, global_row_start, is_fine%v, is_coarse%v, orig_fine_col_indices_handle, &
                  identity_bool, reuse_bool, R_handle)
      call port_check(ierr, "compute_R_from_Z")
      call mat_from_c(R_handle, R)
      if (orig_fine_col_indices_handle /= 0) orig_fine_col_indices%v = orig_fine_col_indices_handle

#if defined(PFLARE_PORT_DEBUG)
      if (orig_fine_col_indices_null) then
         PetscObjectNullify(orig_fine_col_indices_fref)
         call compute_R_from_Z_fref(Z, global_row_start, is_fine, is_coarse, &
                     orig_fine_col_indices_fref, &
                     identity, reuse, &
                     R_fref)
         call ISEqualUnsorted(orig_fine_col_indices, orig_fine_col_indices_fref, is_equal, ierr)
         if (ierr /= 0 .OR. .NOT. is_equal) then
            print *, "C and Fortran reference versions of compute_R_from_Z do not match"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
         end if
         call ISDestroy(orig_fine_col_indices_fref, ierr)
      else
         call compute_R_from_Z_fref(Z, global_row_start, is_fine, is_coarse, &
                     orig_fine_col_indices, &
                     identity, reuse, &
                     R_fref)
      end if
      call port_compare_mats(R, R_fref, "compute_R_from_Z")
      call MatDestroy(R_fref, ierr)
#endif

   end subroutine compute_R_from_Z

! -------------------------------------------------------------------------------------------------------------------------------
! Grid_Transfer_Improve.c
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine improve_w(W, A_ff, A_fc, A_ff_inv, reuse_mat, reuse_mat_two, its, reuse_sparsity)

      ! With PFLARE_PORT_DEBUG the reference starts without the reuse matrices: they keep
      ! the products the C reuses, which a copy would not have, and reusing a product gives
      ! the values computing it again does. They are compared when the C created them too

      ! ~~~~~~
      type(tMat), intent(inout) :: W, A_ff, A_fc, A_ff_inv, reuse_mat, reuse_mat_two
      integer, intent(in) :: its
      logical, intent(in) :: reuse_sparsity

      PetscErrorCode :: ierr
      PetscBool :: reuse_sparsity_bool
      integer(c_long_long) :: reuse_mat_handle, reuse_mat_two_handle
#if defined(PFLARE_PORT_DEBUG)
      type(tMat) :: W_fref, reuse_mat_fref, reuse_mat_two_fref
      logical :: reuse_mat_null, reuse_mat_two_null
#endif
      ! ~~~~~~

#if defined(PFLARE_PORT_DEBUG)
      ! The C refills W in place, so give the reference a copy of the input
      reuse_mat_null = mat_is_null(reuse_mat)
      reuse_mat_two_null = mat_is_null(reuse_mat_two)
      PetscObjectNullify(W_fref)
      if (its /= 0) call port_copy_mat(W, W_fref)
#endif

      reuse_sparsity_bool = reuse_sparsity
      reuse_mat_handle = mat_to_c(reuse_mat)
      reuse_mat_two_handle = mat_to_c(reuse_mat_two)
      ierr = improve_w_c(W%v, A_ff%v, A_fc%v, A_ff_inv%v, reuse_mat_handle, reuse_mat_two_handle, &
                  its, reuse_sparsity_bool)
      call port_check(ierr, "improve_w")
      call mat_from_c(reuse_mat_handle, reuse_mat)
      call mat_from_c(reuse_mat_two_handle, reuse_mat_two)

#if defined(PFLARE_PORT_DEBUG)
      if (its /= 0) then
         PetscObjectNullify(reuse_mat_fref)
         PetscObjectNullify(reuse_mat_two_fref)
         call improve_w_fref(W_fref, A_ff, A_fc, A_ff_inv, reuse_mat_fref, reuse_mat_two_fref, its, reuse_sparsity)
         call port_compare_mats(W, W_fref, "improve_w")
         if (reuse_mat_null) call port_compare_reuse_mats(reuse_mat, reuse_mat_fref, "improve_w")
         if (reuse_mat_two_null) call port_compare_reuse_mats(reuse_mat_two, reuse_mat_two_fref, "improve_w")
         call MatDestroy(W_fref, ierr)
         if (.NOT. mat_is_null(reuse_mat_fref)) call MatDestroy(reuse_mat_fref, ierr)
         if (.NOT. mat_is_null(reuse_mat_two_fref)) call MatDestroy(reuse_mat_two_fref, ierr)
      end if
#endif

   end subroutine improve_w

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine improve_z(Z, A_ff, A_cf, A_ff_inv, reuse_mat, reuse_mat_two, its, reuse_sparsity)

      ! With PFLARE_PORT_DEBUG the reference starts without the reuse matrices, as in improve_w

      ! ~~~~~~
      type(tMat), intent(inout) :: Z, A_ff, A_cf, A_ff_inv, reuse_mat, reuse_mat_two
      integer, intent(in) :: its
      logical, intent(in) :: reuse_sparsity

      PetscErrorCode :: ierr
      PetscBool :: reuse_sparsity_bool
      integer(c_long_long) :: reuse_mat_handle, reuse_mat_two_handle
#if defined(PFLARE_PORT_DEBUG)
      type(tMat) :: Z_fref, reuse_mat_fref, reuse_mat_two_fref
      logical :: reuse_mat_null, reuse_mat_two_null
#endif
      ! ~~~~~~

#if defined(PFLARE_PORT_DEBUG)
      ! The C refills Z in place, so give the reference a copy of the input
      reuse_mat_null = mat_is_null(reuse_mat)
      reuse_mat_two_null = mat_is_null(reuse_mat_two)
      PetscObjectNullify(Z_fref)
      if (its /= 0) call port_copy_mat(Z, Z_fref)
#endif

      reuse_sparsity_bool = reuse_sparsity
      reuse_mat_handle = mat_to_c(reuse_mat)
      reuse_mat_two_handle = mat_to_c(reuse_mat_two)
      ierr = improve_z_c(Z%v, A_ff%v, A_cf%v, A_ff_inv%v, reuse_mat_handle, reuse_mat_two_handle, &
                  its, reuse_sparsity_bool)
      call port_check(ierr, "improve_z")
      call mat_from_c(reuse_mat_handle, reuse_mat)
      call mat_from_c(reuse_mat_two_handle, reuse_mat_two)

#if defined(PFLARE_PORT_DEBUG)
      if (its /= 0) then
         PetscObjectNullify(reuse_mat_fref)
         PetscObjectNullify(reuse_mat_two_fref)
         call improve_z_fref(Z_fref, A_ff, A_cf, A_ff_inv, reuse_mat_fref, reuse_mat_two_fref, its, reuse_sparsity)
         call port_compare_mats(Z, Z_fref, "improve_z")
         if (reuse_mat_null) call port_compare_reuse_mats(reuse_mat, reuse_mat_fref, "improve_z")
         if (reuse_mat_two_null) call port_compare_reuse_mats(reuse_mat_two, reuse_mat_two_fref, "improve_z")
         call MatDestroy(Z_fref, ierr)
         if (.NOT. mat_is_null(reuse_mat_fref)) call MatDestroy(reuse_mat_fref, ierr)
         if (.NOT. mat_is_null(reuse_mat_two_fref)) call MatDestroy(reuse_mat_two_fref, ierr)
      end if
#endif

   end subroutine improve_z

! -------------------------------------------------------------------------------------------------------------------------------
! Constrain_Z_or_W.c
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine get_near_nullspace(input_mat, left, right, left_null_vecs, right_null_vecs)

      ! ~~~~~~
      type(tMat), intent(in)                                :: input_mat
      logical, intent(in)                                   :: left, right
      type(tVec), dimension(:), allocatable, intent(inout)  :: left_null_vecs, right_null_vecs

      PetscErrorCode :: ierr
      PetscBool :: left_bool, right_bool
      type(c_ptr) :: left_null_vecs_ptr, right_null_vecs_ptr
      PetscInt :: null_vecs_size
#if defined(PFLARE_PORT_DEBUG)
      type(tVec), dimension(:), allocatable :: left_null_vecs_fref, right_null_vecs_fref
#endif
      ! ~~~~~~

      left_bool = left
      right_bool = right
      left_null_vecs_ptr = c_null_ptr
      right_null_vecs_ptr = c_null_ptr
      ierr = get_near_nullspace_c(input_mat%v, left_bool, right_bool, left_null_vecs_ptr, right_null_vecs_ptr, &
                  null_vecs_size)
      call port_check(ierr, "get_near_nullspace")
      call vecs_from_c(left_null_vecs_ptr, null_vecs_size, left_null_vecs)
      call vecs_from_c(right_null_vecs_ptr, null_vecs_size, right_null_vecs)

#if defined(PFLARE_PORT_DEBUG)
      call get_near_nullspace_fref(input_mat, left, right, left_null_vecs_fref, right_null_vecs_fref)
      call port_compare_null_vecs(left_null_vecs, left_null_vecs_fref, "get_near_nullspace")
      call port_compare_null_vecs(right_null_vecs, right_null_vecs_fref, "get_near_nullspace")
      call port_destroy_null_vecs(left_null_vecs_fref)
      call port_destroy_null_vecs(right_null_vecs_fref)
#endif

   end subroutine get_near_nullspace

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine smooth_near_nullspace(input_mat, left, right, left_null_vecs, right_null_vecs)

      ! ~~~~~~
      type(tMat), intent(in)                            :: input_mat
      logical, intent(in)                               :: left, right
      type(tVec), dimension(:), intent(inout)           :: left_null_vecs, right_null_vecs

      PetscErrorCode :: ierr
      PetscBool :: left_bool, right_bool
      PetscInt :: left_null_vecs_size, right_null_vecs_size
      integer(c_long_long), dimension(size(left_null_vecs)) :: left_null_vecs_handles
      integer(c_long_long), dimension(size(right_null_vecs)) :: right_null_vecs_handles
#if defined(PFLARE_PORT_DEBUG)
      type(tVec), dimension(:), allocatable :: left_null_vecs_fref, right_null_vecs_fref
#endif
      ! ~~~~~~

#if defined(PFLARE_PORT_DEBUG)
      ! The C smooths the vectors in place, so give the reference copies of the inputs
      call port_copy_null_vecs(left_null_vecs, left, left_null_vecs_fref)
      call port_copy_null_vecs(right_null_vecs, right, right_null_vecs_fref)
#endif

      left_bool = left
      right_bool = right
      left_null_vecs_size = size(left_null_vecs)
      right_null_vecs_size = size(right_null_vecs)
      call vecs_to_c(left_null_vecs, left_null_vecs_handles)
      call vecs_to_c(right_null_vecs, right_null_vecs_handles)
      ierr = smooth_near_nullspace_c(input_mat%v, left_bool, right_bool, left_null_vecs_handles, left_null_vecs_size, &
                  right_null_vecs_handles, right_null_vecs_size)
      call port_check(ierr, "smooth_near_nullspace")

#if defined(PFLARE_PORT_DEBUG)
      call smooth_near_nullspace_fref(input_mat, left, right, left_null_vecs_fref, right_null_vecs_fref)
      if (left) call port_compare_null_vecs(left_null_vecs, left_null_vecs_fref, "smooth_near_nullspace")
      if (right) call port_compare_null_vecs(right_null_vecs, right_null_vecs_fref, "smooth_near_nullspace")
      call port_destroy_null_vecs(left_null_vecs_fref)
      call port_destroy_null_vecs(right_null_vecs_fref)
#endif

   end subroutine smooth_near_nullspace

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine constrain_grid_transfer(z_or_w, is_z, null_vecs_f, null_vecs_c)

      ! The C uses size(null_vecs_c) entries of both arrays, as the Fortran did

      ! ~~~~~~
      type(tMat), intent(inout)                         :: z_or_w
      logical, intent(in)                               :: is_z
      type(tVec), dimension(:), intent(inout)           :: null_vecs_f, null_vecs_c

      PetscErrorCode :: ierr
      PetscBool :: is_z_bool
      PetscInt :: null_vecs_size
      integer(c_long_long) :: z_or_w_handle
      integer(c_long_long), dimension(size(null_vecs_f)) :: null_vecs_f_handles
      integer(c_long_long), dimension(size(null_vecs_c)) :: null_vecs_c_handles
#if defined(PFLARE_PORT_DEBUG)
      type(tMat) :: z_or_w_fref
      type(tVec), dimension(:), allocatable :: null_vecs_f_fref, null_vecs_c_fref
#endif
      ! ~~~~~~

#if defined(PFLARE_PORT_DEBUG)
      ! The C destroys z_or_w and returns the constrained matrix in its place, so give the
      ! reference a copy of the input
      call port_copy_mat(z_or_w, z_or_w_fref)
      ! The reference also gets copies of the vectors, taken before the C runs: the
      ! VecGetArray in the C drops the cached norms of the vectors only on the ranks
      ! with rows to constrain, and the VecNorm of the reference would then abort on
      ! the cached norms that differ between ranks (a copy keeps the cached norms)
      call port_copy_null_vecs(null_vecs_f, .TRUE., null_vecs_f_fref)
      call port_copy_null_vecs(null_vecs_c, .TRUE., null_vecs_c_fref)
#endif

      is_z_bool = is_z
      null_vecs_size = size(null_vecs_c)
      call vecs_to_c(null_vecs_f, null_vecs_f_handles)
      call vecs_to_c(null_vecs_c, null_vecs_c_handles)
      z_or_w_handle = mat_to_c(z_or_w)
      ierr = constrain_grid_transfer_c(z_or_w_handle, is_z_bool, null_vecs_f_handles, null_vecs_c_handles, &
                  null_vecs_size)
      call port_check(ierr, "constrain_grid_transfer")
      call mat_from_c(z_or_w_handle, z_or_w)

#if defined(PFLARE_PORT_DEBUG)
      call constrain_grid_transfer_fref(z_or_w_fref, is_z, null_vecs_f_fref, null_vecs_c_fref)
      call port_compare_mats(z_or_w, z_or_w_fref, "constrain_grid_transfer")
      call MatDestroy(z_or_w_fref, ierr)
      call port_destroy_null_vecs(null_vecs_f_fref)
      call port_destroy_null_vecs(null_vecs_c_fref)
#endif

   end subroutine constrain_grid_transfer

! -------------------------------------------------------------------------------------------------------------------------------

end module port_interfaces_grid
