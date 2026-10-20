module port_interfaces_poly

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! The Fortran that is still to be ported calls the polynomials and approximate
   ! inverses, which have been ported to C (see docs/dev/fortran_removal_plan.md),
   ! through this module, in the same way as through port_interfaces
   ! This is a separate module because the PFLARE_PORT_DEBUG comparisons use the
   ! Fortran reference implementations in src/fref, which themselves use
   ! port_interfaces (so port_interfaces cannot use them)
   ! The routines keep the names and arguments of the Fortran routines they replace,
   ! except that start_approximate_inverse and finish_approximate_inverse take the
   ! gmres_poly_data in place of its buffers and coefficients, which are C memory in the
   ! bind(C) gmres_poly_data of src/fortran/Gmres_Poly_Data_Type.F90
   ! With PFLARE_PORT_DEBUG the routines with logic of their own also run the Fortran
   ! reference on the same input and abort if the results differ: the coefficients
   ! bitwise, assembled matrices row by row (port_compare_mats), and matrix-free
   ! inverses by applying both (and their transposes) to the same random vector
   ! This module is deleted with port_interfaces once all the callers are in C
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------

   use iso_c_binding
   use petscmat
   use gmres_poly_data_type, only: gmres_poly_data, poly_coeffs
   use port_interfaces, only: port_check, mat_to_c, mat_from_c
   ! The Fortran reference implementations, under #ifdef rather than #if defined
   ! so that make depend (makedepf90 -D PFLARE_PORT_DEBUG) sees these uses only
   ! in the port debug build
#ifdef PFLARE_PORT_DEBUG
   use tsqr, only: tsqr_buffers_fref => tsqr_buffers
   use gmres_poly_data_type_fref, only: gmres_poly_data_fref => gmres_poly_data
   use gmres_poly, only: setup_gmres_poly_data_fref => setup_gmres_poly_data, &
         compute_mf_gmres_poly_num_matvecs_fref => compute_mf_gmres_poly_num_matvecs
   use gmres_poly_apply, only: &
         petsc_matvec_poly_newton_residual_mf_fref => petsc_matvec_poly_newton_residual_mf, &
         petsc_matvec_right_scale_poly_newton_residual_mf_fref => petsc_matvec_right_scale_poly_newton_residual_mf
   use approx_inverse_setup, only: start_approximate_inverse_fref => start_approximate_inverse, &
         finish_approximate_inverse_fref => finish_approximate_inverse, &
         reset_inverse_mat_fref => reset_inverse_mat
   use sai_z, only: calculate_and_build_sai_z_fref => calculate_and_build_sai_z
   use port_interfaces, only: port_compare_mats, port_copy_mat, destroy_matrix_reuse
   use pflare_parameters, only: PFLAREINV_POWER, PFLAREINV_ARNOLDI, PFLAREINV_NEWTON, &
         PFLAREINV_NEWTON_NO_EXTRA, PFLAREINV_NEUMANN
#endif

#include "petsc/finclude/petscmat.h"

   implicit none

   private

   ! Gmres_Poly.c
   public :: setup_gmres_poly_data, compute_mf_gmres_poly_num_matvecs
   ! Approx_Inverse_Setup.c
   public :: start_approximate_inverse, finish_approximate_inverse, reset_inverse_mat
   ! SAI_Z.c
   public :: calculate_and_build_sai_z
   ! Gmres_Poly_Apply.c
   public :: shell_poly_block_apply, petsc_matvec_poly_newton_residual_mf, &
         petsc_matvec_right_scale_poly_newton_residual_mf
   ! Matshell_Data_Type.c
   public :: attach_mat_product
#if defined(PFLARE_PORT_DEBUG)
   ! The matshell context accessors are only used by the Fortran reference
   ! implementations in src/fref, so they are only built with PFLARE_PORT_DEBUG
   public :: mat_ctx_create_shell, mat_ctx_get_our_level, mat_ctx_get_air_data
#endif

#if defined(PFLARE_PORT_DEBUG)
   ! The gmres_poly_data whose coefficients a start_approximate_inverse has started,
   ! and the size of their coefficients then, so finish_approximate_inverse knows
   ! whether to run the whole reference (start and finish) or only its finish with
   ! the coefficients it was given
   type :: started_poly_data
      type(c_ptr) :: key = c_null_ptr
      PetscInt :: rows = 0, cols = 0
   end type started_poly_data
   type(started_poly_data), dimension(:), allocatable :: started
#endif

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Gmres_Poly.c
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function setup_gmres_poly_data_c(global_rows, inverse_type, poly_order, &
                  poly_sparsity_order, subcomm, proc_stride, poly_data) &
         bind(c, name="setup_gmres_poly_data")
         use iso_c_binding
         import :: gmres_poly_data
         PetscInt, value :: global_rows, proc_stride
         integer(c_int), value :: inverse_type, poly_order, poly_sparsity_order
         PetscBool, value :: subcomm
         type(gmres_poly_data) :: poly_data
         PetscErrorCode :: setup_gmres_poly_data_c
      end function setup_gmres_poly_data_c
   end interface

   interface
      function compute_mf_gmres_poly_num_matvecs_c(inverse_type, coefficients, coefficients_rows, &
                  coefficients_cols, matvecs) &
         bind(c, name="compute_mf_gmres_poly_num_matvecs")
         use iso_c_binding
         integer(c_int), value :: inverse_type
         type(c_ptr), value :: coefficients
         PetscInt, value :: coefficients_rows, coefficients_cols
         integer(c_int) :: matvecs
         PetscErrorCode :: compute_mf_gmres_poly_num_matvecs_c
      end function compute_mf_gmres_poly_num_matvecs_c
   end interface

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Gmres_Poly_Data_Type.c
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function poly_data_buffers_c(poly_data, buffers) &
         bind(c, name="poly_data_buffers")
         use iso_c_binding
         import :: gmres_poly_data
         type(gmres_poly_data) :: poly_data
         type(c_ptr) :: buffers
         PetscErrorCode :: poly_data_buffers_c
      end function poly_data_buffers_c
   end interface

#if defined(PFLARE_PORT_DEBUG)
   interface
      function poly_data_get_subcomm_c(poly_data, subcomm) &
         bind(c, name="poly_data_get_subcomm")
         use iso_c_binding
         import :: gmres_poly_data
         type(gmres_poly_data) :: poly_data
         PetscBool :: subcomm
         PetscErrorCode :: poly_data_get_subcomm_c
      end function poly_data_get_subcomm_c
   end interface
#endif

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Approx_Inverse_Setup.c
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function start_approximate_inverse_c(matrix, inverse_type, poly_order, diag_scale_polys, &
                  buffers, coefficients, coefficients_rows, coefficients_cols) &
         bind(c, name="start_approximate_inverse")
         use iso_c_binding
         integer(c_long_long), value :: matrix
         integer(c_int), value :: inverse_type, poly_order
         PetscBool, value :: diag_scale_polys
         type(c_ptr), value :: buffers
         type(c_ptr) :: coefficients
         PetscInt :: coefficients_rows, coefficients_cols
         PetscErrorCode :: start_approximate_inverse_c
      end function start_approximate_inverse_c
   end interface

   interface
      function finish_approximate_inverse_c(matrix, inverse_type, poly_order, inverse_sparsity_order, &
                  buffers, coefficients, coefficients_rows, coefficients_cols, &
                  matrix_free, diag_scale_polys, reuse_mat, reuse_submatrices, inv_matrix) &
         bind(c, name="finish_approximate_inverse")
         use iso_c_binding
         integer(c_long_long), value :: matrix
         integer(c_int), value :: inverse_type, poly_order, inverse_sparsity_order
         type(c_ptr), value :: buffers, coefficients
         PetscInt, value :: coefficients_rows, coefficients_cols
         PetscBool, value :: matrix_free, diag_scale_polys
         integer(c_long_long) :: reuse_mat, inv_matrix
         type(c_ptr) :: reuse_submatrices
         PetscErrorCode :: finish_approximate_inverse_c
      end function finish_approximate_inverse_c
   end interface

   interface
      function reset_inverse_mat_c(matrix) &
         bind(c, name="reset_inverse_mat")
         use iso_c_binding
         integer(c_long_long) :: matrix
         PetscErrorCode :: reset_inverse_mat_c
      end function reset_inverse_mat_c
   end interface

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! SAI_Z.c
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function calculate_and_build_sai_z_c(A_ff_input, A_cf, sparsity_mat_cf, incomplete, &
                  reuse_mat, reuse_submatrices, z) &
         bind(c, name="calculate_and_build_sai_z")
         use iso_c_binding
         integer(c_long_long), value :: A_ff_input, A_cf, sparsity_mat_cf
         PetscBool, value :: incomplete
         integer(c_long_long) :: reuse_mat, z
         type(c_ptr) :: reuse_submatrices
         PetscErrorCode :: calculate_and_build_sai_z_c
      end function calculate_and_build_sai_z_c
   end interface

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Gmres_Poly_Apply.c
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function shell_poly_block_apply_c(shell_mat, x_mat, y_mat, block_applied) &
         bind(c, name="shell_poly_block_apply")
         use iso_c_binding
         integer(c_long_long), value :: shell_mat, x_mat, y_mat
         PetscBool :: block_applied
         PetscErrorCode :: shell_poly_block_apply_c
      end function shell_poly_block_apply_c
   end interface

   interface
      function petsc_matvec_poly_newton_residual_mf_c(mat, x, y) &
         bind(c, name="petsc_matvec_poly_newton_residual_mf")
         use iso_c_binding
         integer(c_long_long), value :: mat, x, y
         PetscErrorCode :: petsc_matvec_poly_newton_residual_mf_c
      end function petsc_matvec_poly_newton_residual_mf_c
   end interface

   interface
      function petsc_matvec_right_scale_poly_newton_residual_mf_c(mat, x, y) &
         bind(c, name="petsc_matvec_right_scale_poly_newton_residual_mf")
         use iso_c_binding
         integer(c_long_long), value :: mat, x, y
         PetscErrorCode :: petsc_matvec_right_scale_poly_newton_residual_mf_c
      end function petsc_matvec_right_scale_poly_newton_residual_mf_c
   end interface

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Matshell_Data_Type.c
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function attach_mat_product_c(mat, x_mat, transpose_mat, y_mat, has_product) &
         bind(c, name="attach_mat_product")
         use iso_c_binding
         integer(c_long_long), value :: mat, x_mat, y_mat
         PetscBool, value :: transpose_mat
         type(c_ptr), value :: has_product
         PetscErrorCode :: attach_mat_product_c
      end function attach_mat_product_c
   end interface

#if defined(PFLARE_PORT_DEBUG)
   interface
      function mat_ctx_create_shell_c(comm_mat, local_rows, local_cols, global_rows, global_cols, &
                  our_level, air_data, mat) &
         bind(c, name="mat_ctx_create_shell")
         use iso_c_binding
         integer(c_long_long), value :: comm_mat
         PetscInt, value :: local_rows, local_cols, global_rows, global_cols
         integer(c_int), value :: our_level
         type(c_ptr), value :: air_data
         integer(c_long_long) :: mat
         PetscErrorCode :: mat_ctx_create_shell_c
      end function mat_ctx_create_shell_c
   end interface

   interface
      function mat_ctx_get_our_level_c(mat, our_level) &
         bind(c, name="mat_ctx_get_our_level")
         use iso_c_binding
         integer(c_long_long), value :: mat
         integer(c_int) :: our_level
         PetscErrorCode :: mat_ctx_get_our_level_c
      end function mat_ctx_get_our_level_c
   end interface

   interface
      function mat_ctx_get_air_data_c(mat, air_data) &
         bind(c, name="mat_ctx_get_air_data")
         use iso_c_binding
         integer(c_long_long), value :: mat
         type(c_ptr) :: air_data
         PetscErrorCode :: mat_ctx_get_air_data_c
      end function mat_ctx_get_air_data_c
   end interface
#endif

   contains

! -------------------------------------------------------------------------------------------------------------------------------
! Helpers
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine submatrices_to_c(submatrices, submatrices_ptr)

      ! The address of the C array of submatrices behind the Fortran pointer
      ! (MatCreateSubMatrices allocated it, and the Fortran pointer is over it),
      ! or NULL when it isn't associated

      ! ~~~~~~
      type(tMat), dimension(:), pointer, intent(in) :: submatrices
      type(c_ptr), intent(out)                      :: submatrices_ptr
      ! ~~~~~~

      if (associated(submatrices)) then
         submatrices_ptr = c_loc(submatrices(1))
      else
         submatrices_ptr = c_null_ptr
      end if

   end subroutine submatrices_to_c

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine submatrices_from_c(submatrices_ptr, submatrices)

      ! Points the Fortran pointer at the C array of (one) submatrix the C returned,
      ! or nullifies it

      ! ~~~~~~
      type(c_ptr), intent(in)                          :: submatrices_ptr
      type(tMat), dimension(:), pointer, intent(inout) :: submatrices
      ! ~~~~~~

      if (c_associated(submatrices_ptr)) then
         call c_f_pointer(submatrices_ptr, submatrices, [1])
      else
         submatrices => null()
      end if

   end subroutine submatrices_from_c

#if defined(PFLARE_PORT_DEBUG)
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine port_copy_inverse(mat, mat_copy)

      ! A copy of an assembled inverse for the reference to reuse: the diagonal of a
      ! MATDIAGONAL is copied with VecDuplicate (as in the weighted jacobi wrapper in
      ! port_interfaces), anything else with port_copy_mat

      ! ~~~~~~
      type(tMat), intent(in)    :: mat
      type(tMat), intent(inout) :: mat_copy

      PetscErrorCode :: ierr
      MatType :: mat_type
      type(tVec) :: diag_vec, diag_vec_copy
      ! ~~~~~~

      call MatGetType(mat, mat_type, ierr)
      if (mat_type == MATDIAGONAL) then
         call MatDiagonalGetDiagonal(mat, diag_vec, ierr)
         call VecDuplicate(diag_vec, diag_vec_copy, ierr)
         call VecCopy(diag_vec, diag_vec_copy, ierr)
         call MatDiagonalRestoreDiagonal(mat, diag_vec, ierr)
         ! The matrix takes ownership of diag_vec_copy
         call MatCreateDiagonal(diag_vec_copy, mat_copy, ierr)
         call VecDestroy(diag_vec_copy, ierr)
      else
         call port_copy_mat(mat, mat_copy)
      end if

   end subroutine port_copy_inverse

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine port_compare_vecs(x, x_fref, routine)

      ! Aborts unless the two vectors are bitwise the same

      ! ~~~~~~
      type(tVec), intent(in)       :: x, x_fref
      character(len=*), intent(in) :: routine

      PetscErrorCode :: ierr
      PetscBool :: vec_equal
      integer :: errorcode
      ! ~~~~~~

      call VecEqual(x, x_fref, vec_equal, ierr)
      if (.NOT. vec_equal) then
         print *, "C and Fortran reference versions of ", routine, " do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

   end subroutine port_compare_vecs

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine port_compare_inverses(inv_matrix, inv_matrix_fref, routine)

      ! Aborts unless the inverses the C and the Fortran reference built are the same:
      ! a matrix-free inverse by applying both, and their transposes, to the same random
      ! vector (bitwise, as the shells run the same PETSc operations in the same order),
      ! a MATDIAGONAL by its diagonal and anything else row by row with port_compare_mats

      ! ~~~~~~
      type(tMat), intent(in)       :: inv_matrix, inv_matrix_fref
      character(len=*), intent(in) :: routine

      PetscErrorCode :: ierr
      MatType :: mat_type, mat_type_fref
      MPIU_Comm :: MPI_COMM_MATRIX
      type(tVec) :: x, y, y_fref
      type(tPetscRandom) :: rctx
      integer :: errorcode
      ! ~~~~~~

      call MatGetType(inv_matrix, mat_type, ierr)
      call MatGetType(inv_matrix_fref, mat_type_fref, ierr)
      if (mat_type /= mat_type_fref) then
         print *, "C and Fortran reference versions of ", routine, " do not match (types)"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      if (mat_type == MATSHELL) then
         call PetscObjectGetComm(inv_matrix, MPI_COMM_MATRIX, ierr)
         call PetscRandomCreate(MPI_COMM_MATRIX, rctx, ierr)
         call MatCreateVecs(inv_matrix, x, y, ierr)
         call VecDuplicate(y, y_fref, ierr)
         call VecSetRandom(x, rctx, ierr)
         call PetscRandomDestroy(rctx, ierr)
         call MatMult(inv_matrix, x, y, ierr)
         call MatMult(inv_matrix_fref, x, y_fref, ierr)
         call port_compare_vecs(y, y_fref, routine)
         call MatMultTranspose(inv_matrix, x, y, ierr)
         call MatMultTranspose(inv_matrix_fref, x, y_fref, ierr)
         call port_compare_vecs(y, y_fref, routine)
         call VecDestroy(x, ierr)
         call VecDestroy(y, ierr)
         call VecDestroy(y_fref, ierr)
      else if (mat_type == MATDIAGONAL) then
         call MatCreateVecs(inv_matrix, PETSC_NULL_VEC, y, ierr)
         call VecDuplicate(y, y_fref, ierr)
         call MatGetDiagonal(inv_matrix, y, ierr)
         call MatGetDiagonal(inv_matrix_fref, y_fref, ierr)
         call port_compare_vecs(y, y_fref, routine)
         call VecDestroy(y, ierr)
         call VecDestroy(y_fref, ierr)
      else
         call port_compare_mats(inv_matrix, inv_matrix_fref, routine)
      end if

   end subroutine port_compare_inverses

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine port_compare_coefficients(coefficients, coefficients_fref, routine)

      ! Aborts unless the coefficients are the same size and bitwise the same

      ! ~~~~~~
      PetscReal, dimension(:, :), intent(in) :: coefficients, coefficients_fref
      character(len=*), intent(in)           :: routine

      integer :: errorcode
      ! ~~~~~~

      if (size(coefficients, 1) /= size(coefficients_fref, 1) .OR. &
          size(coefficients, 2) /= size(coefficients_fref, 2)) then
         print *, "C and Fortran reference versions of ", routine, " do not match (coefficient sizes)"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      if (any(coefficients /= coefficients_fref)) then
         print *, "C and Fortran reference versions of ", routine, " do not match (coefficients)"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

   end subroutine port_compare_coefficients
#endif

! -------------------------------------------------------------------------------------------------------------------------------
! Gmres_Poly.c
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine setup_gmres_poly_data(global_rows, inverse_type, poly_order, &
                  poly_sparsity_order, subcomm, proc_stride, poly_data)

      ! ~~~~~~
      PetscInt, intent(in)                              :: global_rows, proc_stride
      integer, intent(in)                               :: inverse_type, poly_order
      integer, intent(in)                               :: poly_sparsity_order
      logical, intent(in)                               :: subcomm
      type(gmres_poly_data), intent(inout)              :: poly_data

      PetscErrorCode :: ierr
      PetscBool :: subcomm_bool
#if defined(PFLARE_PORT_DEBUG)
      type(gmres_poly_data_fref) :: poly_data_fref
      integer :: errorcode
#endif
      ! ~~~~~~

#if defined(PFLARE_PORT_DEBUG)
      ! The reference only allocates the coefficients if they aren't already
      if (c_associated(poly_data%coefficients)) then
         allocate(poly_data_fref%coefficients(poly_data%coefficients_rows, poly_data%coefficients_cols))
      end if
#endif

      subcomm_bool = subcomm
      ierr = setup_gmres_poly_data_c(global_rows, int(inverse_type, c_int), int(poly_order, c_int), &
                  int(poly_sparsity_order, c_int), subcomm_bool, proc_stride, poly_data)
      call port_check(ierr, "setup_gmres_poly_data")

#if defined(PFLARE_PORT_DEBUG)
      call setup_gmres_poly_data_fref(global_rows, inverse_type, poly_order, &
                  poly_sparsity_order, subcomm, proc_stride, poly_data_fref)
      if (poly_data_fref%inverse_type /= poly_data%inverse_type .OR. &
          poly_data_fref%gmres_poly_order /= poly_data%gmres_poly_order .OR. &
          poly_data_fref%gmres_poly_sparsity_order /= poly_data%gmres_poly_sparsity_order .OR. &
          size(poly_data_fref%coefficients, 1) /= poly_data%coefficients_rows .OR. &
          size(poly_data_fref%coefficients, 2) /= poly_data%coefficients_cols) then
         print *, "C and Fortran reference versions of setup_gmres_poly_data do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      deallocate(poly_data_fref%coefficients)
#endif

   end subroutine setup_gmres_poly_data

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine compute_mf_gmres_poly_num_matvecs(inverse_type, coefficients, matvecs)

      ! ~~~~~~
      integer, intent(in)                                       :: inverse_type
      PetscReal, dimension(:, :), contiguous, target, intent(in) :: coefficients
      integer, intent(out)                                      :: matvecs

      PetscErrorCode :: ierr
      integer(c_int) :: matvecs_c
      type(c_ptr) :: coefficients_ptr
#if defined(PFLARE_PORT_DEBUG)
      integer :: matvecs_fref, errorcode
#endif
      ! ~~~~~~

      coefficients_ptr = c_null_ptr
      if (size(coefficients) > 0) coefficients_ptr = c_loc(coefficients)
      ierr = compute_mf_gmres_poly_num_matvecs_c(int(inverse_type, c_int), coefficients_ptr, &
                  size(coefficients, 1, kind=PETSC_INT_KIND), &
                  size(coefficients, 2, kind=PETSC_INT_KIND), matvecs_c)
      call port_check(ierr, "compute_mf_gmres_poly_num_matvecs")
      matvecs = int(matvecs_c)

#if defined(PFLARE_PORT_DEBUG)
      call compute_mf_gmres_poly_num_matvecs_fref(inverse_type, coefficients, matvecs_fref)
      if (matvecs /= matvecs_fref) then
         print *, "C and Fortran reference versions of compute_mf_gmres_poly_num_matvecs do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
#endif

   end subroutine compute_mf_gmres_poly_num_matvecs

! -------------------------------------------------------------------------------------------------------------------------------
! Approx_Inverse_Setup.c
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine start_approximate_inverse(matrix, inverse_type, poly_order, diag_scale_polys, &
                  poly_data)

      ! Takes the gmres_poly_data in place of the Fortran's buffers and coefficients arguments
      ! The comparison against the reference happens in finish_approximate_inverse,
      ! once the coefficients are known (the power basis only finishes them there)

      ! ~~~~~~
      type(tMat), intent(in)                       :: matrix
      integer, intent(in)                          :: inverse_type, poly_order
      logical, intent(in)                          :: diag_scale_polys
      type(gmres_poly_data), target, intent(inout) :: poly_data

      PetscErrorCode :: ierr
      PetscBool :: diag_scale_polys_bool
      type(c_ptr) :: buffers
#if defined(PFLARE_PORT_DEBUG)
      integer :: i_loc, i_found
      type(started_poly_data), dimension(:), allocatable :: started_grow
#endif
      ! ~~~~~~

#if defined(PFLARE_PORT_DEBUG)
      ! Record that this one has been started, and the size of its coefficients now
      if (.NOT. allocated(started)) allocate(started(0))
      i_found = 0
      do i_loc = 1, size(started)
         if (c_associated(started(i_loc)%key, c_loc(poly_data))) i_found = i_loc
      end do
      if (i_found == 0) then
         allocate(started_grow(size(started) + 1))
         started_grow(1:size(started)) = started
         call move_alloc(started_grow, started)
         i_found = size(started)
      end if
      started(i_found)%key = c_loc(poly_data)
      started(i_found)%rows = poly_data%coefficients_rows
      started(i_found)%cols = poly_data%coefficients_cols
#endif

      ierr = poly_data_buffers_c(poly_data, buffers)
      call port_check(ierr, "poly_data_buffers")

      diag_scale_polys_bool = diag_scale_polys
      ierr = start_approximate_inverse_c(matrix%v, int(inverse_type, c_int), int(poly_order, c_int), &
                  diag_scale_polys_bool, buffers, poly_data%coefficients, &
                  poly_data%coefficients_rows, poly_data%coefficients_cols)
      call port_check(ierr, "start_approximate_inverse")

   end subroutine start_approximate_inverse

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine finish_approximate_inverse(matrix, inverse_type, &
                  poly_order, inverse_sparsity_order, &
                  poly_data, &
                  matrix_free, diag_scale_polys, reuse_mat, reuse_submatrices, inv_matrix)

      ! Takes the gmres_poly_data in place of the Fortran's buffers and coefficients arguments
      ! With PFLARE_PORT_DEBUG, if this poly_data was started by start_approximate_inverse
      ! the reference runs its start and finish on the same matrix, otherwise its finish
      ! with a copy of the coefficients; then the coefficients and the inverses are compared

      ! ~~~~~~
      type(tMat), intent(in)                            :: matrix
      integer, intent(in)                               :: inverse_type, poly_order
      integer, intent(in)                               :: inverse_sparsity_order
      type(gmres_poly_data), target, intent(inout)      :: poly_data
      logical, intent(in)                               :: matrix_free, diag_scale_polys
      type(tMat), intent(inout)                         :: reuse_mat, inv_matrix
      type(tMat), dimension(:), pointer, intent(inout)  :: reuse_submatrices

      PetscErrorCode :: ierr
      PetscBool :: matrix_free_bool, diag_scale_polys_bool
      type(c_ptr) :: buffers, submatrices_ptr
      integer(c_long_long) :: reuse_mat_handle, inv_matrix_handle
#if defined(PFLARE_PORT_DEBUG)
      integer :: i_loc, i_found
      type(started_poly_data), dimension(:), allocatable :: started_shrink
      type(tsqr_buffers_fref) :: buffers_fref
      PetscReal, dimension(:, :), pointer, contiguous :: coefficients_fref, coefficients
      type(tMat) :: inv_matrix_fref, reuse_mat_fref
      type(tMat), dimension(:), pointer :: reuse_submatrices_fref
      PetscBool :: subcomm_bool
      MatType :: mat_type
      MPIU_Comm :: MPI_COMM_MATRIX
      type(tVec) :: x, y, y_fref
      type(tPetscRandom) :: rctx
#endif
      ! ~~~~~~

#if defined(PFLARE_PORT_DEBUG)
      ! The reference reuses a copy of an assembled inverse; a matshell it builds afresh
      PetscObjectNullify(inv_matrix_fref)
      if (.NOT. PetscObjectIsNull(inv_matrix)) then
         call MatGetType(inv_matrix, mat_type, ierr)
         if (mat_type /= MATSHELL) call port_copy_inverse(inv_matrix, inv_matrix_fref)
      end if
#endif

      ierr = poly_data_buffers_c(poly_data, buffers)
      call port_check(ierr, "poly_data_buffers")

      matrix_free_bool = matrix_free
      diag_scale_polys_bool = diag_scale_polys
      reuse_mat_handle = mat_to_c(reuse_mat)
      inv_matrix_handle = mat_to_c(inv_matrix)
      call submatrices_to_c(reuse_submatrices, submatrices_ptr)
      ierr = finish_approximate_inverse_c(matrix%v, int(inverse_type, c_int), int(poly_order, c_int), &
                  int(inverse_sparsity_order, c_int), &
                  buffers, poly_data%coefficients, poly_data%coefficients_rows, poly_data%coefficients_cols, &
                  matrix_free_bool, diag_scale_polys_bool, reuse_mat_handle, submatrices_ptr, inv_matrix_handle)
      call port_check(ierr, "finish_approximate_inverse")
      call mat_from_c(reuse_mat_handle, reuse_mat)
      call mat_from_c(inv_matrix_handle, inv_matrix)
      call submatrices_from_c(submatrices_ptr, reuse_submatrices)

#if defined(PFLARE_PORT_DEBUG)
      ! Was this poly_data started?
      i_found = 0
      if (allocated(started)) then
         do i_loc = 1, size(started)
            if (c_associated(started(i_loc)%key, c_loc(poly_data))) i_found = i_loc
         end do
      end if

      PetscObjectNullify(reuse_mat_fref)
      reuse_submatrices_fref => null()
      if (i_found /= 0) then
         ! Run the whole reference, starting from coefficients of the size the C started with
         allocate(coefficients_fref(started(i_found)%rows, started(i_found)%cols))
         ierr = poly_data_get_subcomm_c(poly_data, subcomm_bool)
         call port_check(ierr, "poly_data_get_subcomm")
         buffers_fref%subcomm = subcomm_bool
         call start_approximate_inverse_fref(matrix, inverse_type, poly_order, diag_scale_polys, &
                  buffers_fref, coefficients_fref)

         ! Forget it
         allocate(started_shrink(size(started) - 1))
         started_shrink(1:i_found-1) = started(1:i_found-1)
         started_shrink(i_found:) = started(i_found+1:)
         call move_alloc(started_shrink, started)
      else
         ! The coefficients were supplied, so set up the buffers as
         ! calculate_and_build_approximate_inverse does in that case
         allocate(coefficients_fref(poly_data%coefficients_rows, poly_data%coefficients_cols))
         coefficients_fref = reshape(poly_coeffs(poly_data), shape(coefficients_fref))
         buffers_fref%on_subcomm = .FALSE.
         buffers_fref%matrix = matrix
         call PetscObjectReference(matrix, ierr)
      end if
      call finish_approximate_inverse_fref(matrix, inverse_type, &
                  poly_order, inverse_sparsity_order, &
                  buffers_fref, coefficients_fref, &
                  matrix_free, diag_scale_polys, reuse_mat_fref, reuse_submatrices_fref, inv_matrix_fref)

      ! The coefficients (only the polynomial types set them, for the others they
      ! are left uninitialised by both)
      coefficients => poly_coeffs(poly_data)
      if (associated(coefficients) .AND. &
            (inverse_type == PFLAREINV_POWER .OR. inverse_type == PFLAREINV_ARNOLDI .OR. &
             inverse_type == PFLAREINV_NEWTON .OR. inverse_type == PFLAREINV_NEWTON_NO_EXTRA .OR. &
             inverse_type == PFLAREINV_NEUMANN)) then
         call port_compare_coefficients(coefficients, coefficients_fref, "finish_approximate_inverse")
      end if

      ! The inverse
      call port_compare_inverses(inv_matrix, inv_matrix_fref, "finish_approximate_inverse")

      ! The residual applies of the matrix-free Newton polynomials
      if ((inverse_type == PFLAREINV_NEWTON .OR. inverse_type == PFLAREINV_NEWTON_NO_EXTRA) .AND. &
            matrix_free) then
         call PetscObjectGetComm(inv_matrix, MPI_COMM_MATRIX, ierr)
         call PetscRandomCreate(MPI_COMM_MATRIX, rctx, ierr)
         call MatCreateVecs(inv_matrix, x, y, ierr)
         call VecDuplicate(y, y_fref, ierr)
         call VecSetRandom(x, rctx, ierr)
         call PetscRandomDestroy(rctx, ierr)
         if (diag_scale_polys) then
            call petsc_matvec_right_scale_poly_newton_residual_mf(inv_matrix, x, y)
            call petsc_matvec_right_scale_poly_newton_residual_mf_fref(inv_matrix_fref, x, y_fref)
         else
            call petsc_matvec_poly_newton_residual_mf(inv_matrix, x, y)
            call petsc_matvec_poly_newton_residual_mf_fref(inv_matrix_fref, x, y_fref)
         end if
         call port_compare_vecs(y, y_fref, "petsc_matvec_poly_newton_residual_mf")
         call VecDestroy(x, ierr)
         call VecDestroy(y, ierr)
         call VecDestroy(y_fref, ierr)
      end if

      ! The reference matshell points at coefficients_fref, so it goes first
      call reset_inverse_mat_fref(inv_matrix_fref)
      call destroy_matrix_reuse(reuse_mat_fref, reuse_submatrices_fref)
      deallocate(coefficients_fref)
#endif

   end subroutine finish_approximate_inverse


! -------------------------------------------------------------------------------------------------------------------------------

   subroutine reset_inverse_mat(matrix)

      ! No port debug comparison, this destroys the matrix

      ! ~~~~~~
      type(tMat), intent(inout) :: matrix

      PetscErrorCode :: ierr
      integer(c_long_long) :: matrix_handle
      ! ~~~~~~

      matrix_handle = mat_to_c(matrix)
      ierr = reset_inverse_mat_c(matrix_handle)
      call port_check(ierr, "reset_inverse_mat")
      call mat_from_c(matrix_handle, matrix)

   end subroutine reset_inverse_mat

! -------------------------------------------------------------------------------------------------------------------------------
! SAI_Z.c
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine calculate_and_build_sai_z(A_ff_input, A_cf, sparsity_mat_cf, incomplete, reuse_mat, reuse_submatrices, z)

      ! ~~~~~~
      type(tMat), intent(in)                            :: A_ff_input, A_cf, sparsity_mat_cf
      logical, intent(in)                               :: incomplete
      type(tMat), intent(inout)                         :: reuse_mat, z
      type(tMat), dimension(:), pointer, intent(inout)  :: reuse_submatrices

      PetscErrorCode :: ierr
      PetscBool :: incomplete_bool
      type(c_ptr) :: submatrices_ptr
      integer(c_long_long) :: reuse_mat_handle, z_handle
#if defined(PFLARE_PORT_DEBUG)
      type(tMat) :: z_fref, reuse_mat_fref
      type(tMat), dimension(:), pointer :: reuse_submatrices_fref
#endif
      ! ~~~~~~

#if defined(PFLARE_PORT_DEBUG)
      ! The C reuses z in place, so give the reference a copy of the input
      PetscObjectNullify(z_fref)
      if (.NOT. PetscObjectIsNull(z)) call port_copy_mat(z, z_fref)
#endif

      incomplete_bool = incomplete
      reuse_mat_handle = mat_to_c(reuse_mat)
      z_handle = mat_to_c(z)
      call submatrices_to_c(reuse_submatrices, submatrices_ptr)
      ierr = calculate_and_build_sai_z_c(A_ff_input%v, A_cf%v, sparsity_mat_cf%v, incomplete_bool, &
                  reuse_mat_handle, submatrices_ptr, z_handle)
      call port_check(ierr, "calculate_and_build_sai_z")
      call mat_from_c(reuse_mat_handle, reuse_mat)
      call mat_from_c(z_handle, z)
      call submatrices_from_c(submatrices_ptr, reuse_submatrices)

#if defined(PFLARE_PORT_DEBUG)
      PetscObjectNullify(reuse_mat_fref)
      reuse_submatrices_fref => null()
      call calculate_and_build_sai_z_fref(A_ff_input, A_cf, sparsity_mat_cf, incomplete, &
                  reuse_mat_fref, reuse_submatrices_fref, z_fref)
      call port_compare_mats(z, z_fref, "calculate_and_build_sai_z")
      call MatDestroy(z_fref, ierr)
      call destroy_matrix_reuse(reuse_mat_fref, reuse_submatrices_fref)
#endif

   end subroutine calculate_and_build_sai_z

! -------------------------------------------------------------------------------------------------------------------------------
! Gmres_Poly_Apply.c
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine shell_poly_block_apply(shell_mat, x_mat, y_mat, block_applied)

      ! No port debug comparison: the reference applies a matshell the reference built,
      ! so the block applies are compared through PCMatApply in tests/port_compare_pc.c

      ! ~~~~~~
      type(tMat), intent(in)    :: shell_mat
      type(tMat)                :: x_mat
      type(tMat)                :: y_mat
      logical, intent(out)      :: block_applied

      PetscErrorCode :: ierr
      PetscBool :: block_applied_bool
      ! ~~~~~~

      ierr = shell_poly_block_apply_c(shell_mat%v, x_mat%v, y_mat%v, block_applied_bool)
      call port_check(ierr, "shell_poly_block_apply")
      block_applied = block_applied_bool

   end subroutine shell_poly_block_apply

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine petsc_matvec_poly_newton_residual_mf(mat, x, y)

      ! Compared against the reference in finish_approximate_inverse, on the matshells
      ! the C and the reference built there

      ! ~~~~~~
      type(tMat), intent(in)    :: mat
      type(tVec) :: x
      type(tVec) :: y

      PetscErrorCode :: ierr
      ! ~~~~~~

      ierr = petsc_matvec_poly_newton_residual_mf_c(mat%v, x%v, y%v)
      call port_check(ierr, "petsc_matvec_poly_newton_residual_mf")

   end subroutine petsc_matvec_poly_newton_residual_mf

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine petsc_matvec_right_scale_poly_newton_residual_mf(mat, x, y)

      ! Compared against the reference in finish_approximate_inverse, on the matshells
      ! the C and the reference built there

      ! ~~~~~~
      type(tMat), intent(in)    :: mat
      type(tVec) :: x
      type(tVec) :: y

      PetscErrorCode :: ierr
      ! ~~~~~~

      ierr = petsc_matvec_right_scale_poly_newton_residual_mf_c(mat%v, x%v, y%v)
      call port_check(ierr, "petsc_matvec_right_scale_poly_newton_residual_mf")

   end subroutine petsc_matvec_right_scale_poly_newton_residual_mf

! -------------------------------------------------------------------------------------------------------------------------------
! Matshell_Data_Type.c
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine attach_mat_product(mat, x_mat, transpose_mat, y_mat, ierr, has_product)

      ! No port debug comparison, this is a sequence of PETSc calls

      ! ~~~~~~
      type(tMat), intent(in)         :: mat
      type(tMat)                     :: x_mat, y_mat
      logical, intent(in)            :: transpose_mat
      PetscErrorCode, intent(inout)  :: ierr
      logical, optional, intent(out) :: has_product

      PetscBool :: transpose_mat_bool
      PetscBool, target :: has_product_bool
      type(c_ptr) :: has_product_ptr
      ! ~~~~~~

      transpose_mat_bool = transpose_mat
      has_product_ptr = c_null_ptr
      if (present(has_product)) has_product_ptr = c_loc(has_product_bool)
      ierr = attach_mat_product_c(mat%v, x_mat%v, transpose_mat_bool, y_mat%v, has_product_ptr)
      call port_check(ierr, "attach_mat_product")
      if (present(has_product)) has_product = has_product_bool

   end subroutine attach_mat_product

! -------------------------------------------------------------------------------------------------------------------------------

#if defined(PFLARE_PORT_DEBUG)
   subroutine mat_ctx_create_shell(comm_mat, local_rows, local_cols, global_rows, global_cols, &
                  our_level, air_data, mat)

      ! Creates a matshell on the comm of comm_mat whose context records our_level and
      ! the (Fortran) air_data, which replaces allocating a Fortran mat_ctxtype and
      ! creating the matshell with it

      ! ~~~~~~
      type(tMat), intent(in)    :: comm_mat
      PetscInt, intent(in)      :: local_rows, local_cols, global_rows, global_cols
      integer, intent(in)       :: our_level
      type(c_ptr), intent(in)   :: air_data
      type(tMat), intent(inout) :: mat

      PetscErrorCode :: ierr
      integer(c_long_long) :: mat_handle
      ! ~~~~~~

      mat_handle = 0
      ierr = mat_ctx_create_shell_c(comm_mat%v, local_rows, local_cols, global_rows, global_cols, &
                  int(our_level, c_int), air_data, mat_handle)
      call port_check(ierr, "mat_ctx_create_shell")
      mat%v = mat_handle

   end subroutine mat_ctx_create_shell

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine mat_ctx_get_our_level(mat, our_level)

      ! Returns our_level from the context of a matshell

      ! ~~~~~~
      type(tMat), intent(in) :: mat
      integer, intent(out)   :: our_level

      PetscErrorCode :: ierr
      integer(c_int) :: our_level_c
      ! ~~~~~~

      ierr = mat_ctx_get_our_level_c(mat%v, our_level_c)
      call port_check(ierr, "mat_ctx_get_our_level")
      our_level = int(our_level_c)

   end subroutine mat_ctx_get_our_level

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine mat_ctx_get_air_data(mat, air_data)

      ! Returns the air_data pointer from the context of a matshell, which the caller
      ! turns into a Fortran pointer with c_f_pointer

      ! ~~~~~~
      type(tMat), intent(in)   :: mat
      type(c_ptr), intent(out) :: air_data

      PetscErrorCode :: ierr
      ! ~~~~~~

      ierr = mat_ctx_get_air_data_c(mat%v, air_data)
      call port_check(ierr, "mat_ctx_get_air_data")

   end subroutine mat_ctx_get_air_data
#endif

! -------------------------------------------------------------------------------------------------------------------------------

end module port_interfaces_poly
