module port_interfaces

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! The Fortran that is still to be ported calls the modules that have been
   ! ported to C (see docs/dev/fortran_removal_plan.md) through this module
   ! Each C routine has a bind(C) interface, named after the routine with a _c suffix,
   ! and a module procedure with the name and arguments of the Fortran routine it
   ! replaces, so the callers only change the module they use
   ! The module procedures convert the arguments (PETSc objects cross as their
   ! handles, arrays with their sizes, 0-based indices become 1-based), check the
   ! error code the C returns and, with PFLARE_PORT_DEBUG, also run the Fortran
   ! reference implementation in src/fref on the same input and abort if the
   ! results differ
   ! This module grows as modules are ported and is deleted once all the callers are in C
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------

   use iso_c_binding
   use iso_fortran_env, only: output_unit
   use petscmat
   ! The Fortran reference implementations, under #ifdef rather than #if defined
   ! so that make depend (makedepf90 -D PFLARE_PORT_DEBUG) sees these uses only
   ! in the port debug build
#ifdef PFLARE_PORT_DEBUG
   use sorting, only: intersect_pre_sorted_indices_only_fref => intersect_pre_sorted_indices_only, &
         merge_pre_sorted_fref => merge_pre_sorted
   use aggregation, only: generate_serial_aggregation_fref => generate_serial_aggregation
   use weighted_jacobi, only: &
         calculate_and_build_weighted_jacobi_inverse_fref => calculate_and_build_weighted_jacobi_inverse, &
         calculate_l1_jacobi_inverse_diagonal_fref => calculate_l1_jacobi_inverse_diagonal
   use petsc_helper, only: kokkos_debug_fref => kokkos_debug, &
         remove_small_from_sparse_fref => remove_small_from_sparse, &
         remove_from_sparse_match_fref => remove_from_sparse_match, &
         MatSetAllValues_fref => MatSetAllValues, &
         mat_duplicate_copy_plus_diag_fref => mat_duplicate_copy_plus_diag, &
         MatCreateSubMatrixWrapper_fref => MatCreateSubMatrixWrapper, &
         generate_identity_fref => generate_identity, &
         generate_identity_rect_fref => generate_identity_rect, &
         generate_identity_is_fref => generate_identity_is, &
         get_nnzs_petsc_sparse_fref => get_nnzs_petsc_sparse, &
         MatGetNNZs_both_fref => MatGetNNZs_both, &
         svd_fref => svd, pseudo_inv_fref => pseudo_inv
   use repartition, only: &
         compute_mat_ratio_local_nonlocal_nnzs_fref => compute_mat_ratio_local_nonlocal_nnzs, &
         calculate_repartition_fref => calculate_repartition, &
         MatMPICreateNonemptySubcomm_fref => MatMPICreateNonemptySubcomm
#endif

#include "petsc/finclude/petscmat.h"

   implicit none

   private

   ! Timers.c
   public :: timer_start, timer_finish, timer_reset, timer_clear, timer_time, print_timers
   ! Sorting.c
   public :: sorted_unique, intersect_pre_sorted_indices_only, merge_pre_sorted
   ! Aggregation.c
   public :: generate_serial_aggregation
   ! Weighted_Jacobi.c
   public :: calculate_and_build_weighted_jacobi_inverse, calculate_l1_jacobi_inverse_diagonal
   ! PETSc_Helper.c
   public :: kokkos_debug, destroy_matrix_reuse, remove_small_from_sparse, remove_from_sparse_match, &
         MatSetAllValues, mat_duplicate_copy_plus_diag, MatAXPYWrapper, MatCreateSubMatrixWrapper, &
         generate_identity, generate_identity_rect, generate_identity_is, get_nnzs_petsc_sparse, &
         MatGetNNZs_both, svd, pseudo_inv, mat_mat_symbolic
   ! Repartition.c
   public :: compute_mat_ratio_local_nonlocal_nnzs, calculate_repartition, MatMPICreateNonemptySubcomm
   ! The handle conversions and error check, which the wrappers in
   ! src/fortran/Port_Interfaces_Poly.F90 use too
   public :: port_check, mat_to_c, mat_from_c
#if defined(PFLARE_PORT_DEBUG)
   ! The row by row matrix comparison and copy, which the Fortran wrappers in src/fortran use too
   public :: port_compare_mats, port_copy_mat
#endif

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
   ! Timers.c
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function print_timers_c() &
         bind(c, name="print_timers")
         use iso_c_binding
         PetscErrorCode :: print_timers_c
      end function print_timers_c
   end interface

   interface
      function timer_start_c(id) &
         bind(c, name="timer_start")
         use iso_c_binding
         integer(c_int), value :: id
         PetscErrorCode :: timer_start_c
      end function timer_start_c
   end interface

   interface
      function timer_finish_c(id) &
         bind(c, name="timer_finish")
         use iso_c_binding
         integer(c_int), value :: id
         PetscErrorCode :: timer_finish_c
      end function timer_finish_c
   end interface

   interface
      function timer_reset_c() &
         bind(c, name="timer_reset")
         use iso_c_binding
         PetscErrorCode :: timer_reset_c
      end function timer_reset_c
   end interface

   interface
      function timer_clear_c(id) &
         bind(c, name="timer_clear")
         use iso_c_binding
         integer(c_int), value :: id
         PetscErrorCode :: timer_clear_c
      end function timer_clear_c
   end interface

   interface
      function timer_time_c(id, timer_time) &
         bind(c, name="timer_time")
         use iso_c_binding
         integer(c_int), value :: id
         PetscReal :: timer_time
         PetscErrorCode :: timer_time_c
      end function timer_time_c
   end interface

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Sorting.c
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function sorted_unique_c(array, n) &
         bind(c, name="sorted_unique")
         use iso_c_binding
         PetscInt :: n
         PetscInt, dimension(n) :: array
         PetscErrorCode :: sorted_unique_c
      end function sorted_unique_c
   end interface

   interface
      function intersect_pre_sorted_indices_only_c(array1, size_array1, array2, size_array2, &
                  indices_array1, indices_array2, intersect_count) &
         bind(c, name="intersect_pre_sorted_indices_only")
         use iso_c_binding
         PetscInt, value :: size_array1, size_array2
         PetscInt, dimension(size_array1) :: array1
         PetscInt, dimension(size_array2) :: array2
         ! Sized by the caller, at least min(size_array1, size_array2)
         integer(c_int), dimension(*) :: indices_array1, indices_array2
         integer(c_int) :: intersect_count
         PetscErrorCode :: intersect_pre_sorted_indices_only_c
      end function intersect_pre_sorted_indices_only_c
   end interface

   interface
      function merge_pre_sorted_c(array1, size_array1, array2, size_array2, sorted_array) &
         bind(c, name="merge_pre_sorted")
         use iso_c_binding
         PetscInt, value :: size_array1, size_array2
         PetscInt, dimension(size_array1) :: array1
         PetscInt, dimension(size_array2) :: array2
         PetscInt, dimension(size_array1 + size_array2) :: sorted_array
         PetscErrorCode :: merge_pre_sorted_c
      end function merge_pre_sorted_c
   end interface

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Aggregation.c
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function generate_serial_aggregation_c(strength_mat, cf_markers, aggregates) &
         bind(c, name="generate_serial_aggregation")
         use iso_c_binding
         integer(c_long_long), value :: strength_mat
         type(c_ptr) :: cf_markers, aggregates
         PetscErrorCode :: generate_serial_aggregation_c
      end function generate_serial_aggregation_c
   end interface

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Weighted_Jacobi.c
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function calculate_and_build_weighted_jacobi_inverse_c(matrix, inverse_type, inv_matrix) &
         bind(c, name="calculate_and_build_weighted_jacobi_inverse")
         use iso_c_binding
         integer(c_long_long), value :: matrix
         integer(c_int), value :: inverse_type
         integer(c_long_long) :: inv_matrix
         PetscErrorCode :: calculate_and_build_weighted_jacobi_inverse_c
      end function calculate_and_build_weighted_jacobi_inverse_c
   end interface

   interface
      function calculate_l1_jacobi_inverse_diagonal_c(matrix, diag_vec) &
         bind(c, name="calculate_l1_jacobi_inverse_diagonal")
         use iso_c_binding
         integer(c_long_long), value :: matrix, diag_vec
         PetscErrorCode :: calculate_l1_jacobi_inverse_diagonal_c
      end function calculate_l1_jacobi_inverse_diagonal_c
   end interface

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! PETSc_Helper.c
   ! The optional arguments are passed as pointers, null when absent
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function kokkos_debug_c() &
         bind(c, name="kokkos_debug")
         use iso_c_binding
         PetscBool :: kokkos_debug_c
      end function kokkos_debug_c
   end interface

   interface
      function destroy_matrix_reuse_c(mat, submatrices) &
         bind(c, name="destroy_matrix_reuse")
         use iso_c_binding
         integer(c_long_long) :: mat
         type(c_ptr) :: submatrices
         PetscErrorCode :: destroy_matrix_reuse_c
      end function destroy_matrix_reuse_c
   end interface

   interface
      function remove_small_from_sparse_c(input_mat, tol, output_mat, relative_max_row_tol_int, &
                  lump, drop_diagonal_int, diag_strength_int) &
         bind(c, name="remove_small_from_sparse")
         use iso_c_binding
         integer(c_long_long), value :: input_mat
         PetscReal, value :: tol
         integer(c_long_long) :: output_mat
         type(c_ptr), value :: relative_max_row_tol_int, lump, drop_diagonal_int, diag_strength_int
         PetscErrorCode :: remove_small_from_sparse_c
      end function remove_small_from_sparse_c
   end interface

   ! The C is called remove_from_sparse_match_c, as remove_from_sparse_match is the
   ! public C API in include/pflare.h
   interface
      function remove_from_sparse_match_c(input_mat, output_mat, lump, alpha) &
         bind(c, name="remove_from_sparse_match_c")
         use iso_c_binding
         integer(c_long_long), value :: input_mat, output_mat
         type(c_ptr), value :: lump, alpha
         PetscErrorCode :: remove_from_sparse_match_c
      end function remove_from_sparse_match_c
   end interface

   interface
      function MatSetAllValues_c(input_mat, val) &
         bind(c, name="MatSetAllValues")
         use iso_c_binding
         integer(c_long_long), value :: input_mat
         PetscScalar, value :: val
         PetscErrorCode :: MatSetAllValues_c
      end function MatSetAllValues_c
   end interface

   interface
      function mat_duplicate_copy_plus_diag_c(input_mat, reuse, output_mat) &
         bind(c, name="mat_duplicate_copy_plus_diag")
         use iso_c_binding
         integer(c_long_long), value :: input_mat
         PetscBool, value :: reuse
         integer(c_long_long) :: output_mat
         PetscErrorCode :: mat_duplicate_copy_plus_diag_c
      end function mat_duplicate_copy_plus_diag_c
   end interface

   interface
      function MatAXPYWrapper_c(y_mat, alpha, x_mat) &
         bind(c, name="MatAXPYWrapper")
         use iso_c_binding
         integer(c_long_long), value :: y_mat, x_mat
         PetscScalar, value :: alpha
         PetscErrorCode :: MatAXPYWrapper_c
      end function MatAXPYWrapper_c
   end interface

   interface
      function MatCreateSubMatrixWrapper_c(input_mat, is_row, is_col, reuse, output_mat, &
                  our_level, is_row_fine, is_col_fine, kokkos_is_views_handle) &
         bind(c, name="MatCreateSubMatrixWrapper")
         use iso_c_binding
         integer(c_long_long), value :: input_mat, is_row, is_col
         ! The value of the MatReuse
         integer(c_int), value :: reuse
         integer(c_long_long) :: output_mat
         type(c_ptr), value :: our_level, is_row_fine, is_col_fine, kokkos_is_views_handle
         PetscErrorCode :: MatCreateSubMatrixWrapper_c
      end function MatCreateSubMatrixWrapper_c
   end interface

   interface
      function generate_identity_c(input_mat, output_mat) &
         bind(c, name="generate_identity")
         use iso_c_binding
         integer(c_long_long), value :: input_mat
         integer(c_long_long) :: output_mat
         PetscErrorCode :: generate_identity_c
      end function generate_identity_c
   end interface

   interface
      function generate_identity_rect_c(full_mat, rect_indices, output_mat) &
         bind(c, name="generate_identity_rect")
         use iso_c_binding
         integer(c_long_long), value :: full_mat, rect_indices
         integer(c_long_long) :: output_mat
         PetscErrorCode :: generate_identity_rect_c
      end function generate_identity_rect_c
   end interface

   interface
      function generate_identity_is_c(input_mat, indices, output_mat) &
         bind(c, name="generate_identity_is")
         use iso_c_binding
         integer(c_long_long), value :: input_mat, indices
         integer(c_long_long) :: output_mat
         PetscErrorCode :: generate_identity_is_c
      end function generate_identity_is_c
   end interface

   interface
      function get_nnzs_petsc_sparse_c(input_mat, nnzs) &
         bind(c, name="get_nnzs_petsc_sparse")
         use iso_c_binding
         integer(c_long_long), value :: input_mat
         integer(c_int64_t) :: nnzs
         PetscErrorCode :: get_nnzs_petsc_sparse_c
      end function get_nnzs_petsc_sparse_c
   end interface

   interface
      function MatGetNNZs_both_c(input_mat, nnzs_local, nnzs_nonlocal) &
         bind(c, name="MatGetNNZs_both")
         use iso_c_binding
         integer(c_long_long), value :: input_mat
         PetscInt :: nnzs_local, nnzs_nonlocal
         PetscErrorCode :: MatGetNNZs_both_c
      end function MatGetNNZs_both_c
   end interface

   interface
      function svd_c(input, input_rows, input_cols, U, sigma, VT) &
         bind(c, name="svd")
         use iso_c_binding
         PetscInt, value :: input_rows, input_cols
         PetscScalar, dimension(input_rows, input_cols) :: input
         PetscScalar, dimension(input_rows, input_rows) :: U
         PetscReal, dimension(min(input_rows, input_cols)) :: sigma
         PetscScalar, dimension(input_cols, input_cols) :: VT
         PetscErrorCode :: svd_c
      end function svd_c
   end interface

   interface
      function pseudo_inv_c(input, input_rows, input_cols, output) &
         bind(c, name="pseudo_inv")
         use iso_c_binding
         PetscInt, value :: input_rows, input_cols
         PetscScalar, dimension(input_rows, input_cols) :: input
         PetscScalar, dimension(input_cols, input_rows) :: output
         PetscErrorCode :: pseudo_inv_c
      end function pseudo_inv_c
   end interface

   interface
      function mat_mat_symbolic_c(A, B, C) &
         bind(c, name="mat_mat_symbolic")
         use iso_c_binding
         integer(c_long_long), value :: A, B
         integer(c_long_long) :: C
         PetscErrorCode :: mat_mat_symbolic_c
      end function mat_mat_symbolic_c
   end interface

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! Repartition.c
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function compute_mat_ratio_local_nonlocal_nnzs_c(input_mat, no_active_cores, ratio) &
         bind(c, name="compute_mat_ratio_local_nonlocal_nnzs")
         use iso_c_binding
         integer(c_long_long), value :: input_mat
         PetscInt, value :: no_active_cores
         PetscReal :: ratio
         PetscErrorCode :: compute_mat_ratio_local_nonlocal_nnzs_c
      end function compute_mat_ratio_local_nonlocal_nnzs_c
   end interface

   interface
      function calculate_repartition_c(input_mat, proc_stride, no_active_cores, simple, index) &
         bind(c, name="calculate_repartition")
         use iso_c_binding
         integer(c_long_long), value :: input_mat
         PetscInt, value :: proc_stride
         PetscInt :: no_active_cores
         PetscBool, value :: simple
         integer(c_long_long) :: index
         PetscErrorCode :: calculate_repartition_c
      end function calculate_repartition_c
   end interface

   ! Binds to the C MatMPICreateNonemptySubcomm, which calls the helper whose C name
   ! is MatMPICreateNonemptySubcomm_c
   interface
      function MatMPICreateNonemptySubcomm_c(input_mat, on_subcomm, output_mat) &
         bind(c, name="MatMPICreateNonemptySubcomm")
         use iso_c_binding
         integer(c_long_long), value :: input_mat
         PetscBool :: on_subcomm
         integer(c_long_long) :: output_mat
         PetscErrorCode :: MatMPICreateNonemptySubcomm_c
      end function MatMPICreateNonemptySubcomm_c
   end interface

   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine port_check(ierr, routine)

      ! The C routines return a PETSc error code, and PETSc has already printed
      ! the error and traceback if it is not zero. The Fortran callers have no
      ! error code to pass it back through, so abort

      ! ~~~~~~
      PetscErrorCode, intent(in)   :: ierr
      character(len=*), intent(in) :: routine

      integer :: errorcode
      ! ~~~~~~

      if (ierr /= 0) then
         print *, routine, " failed"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

   end subroutine port_check

! -------------------------------------------------------------------------------------------------------------------------------

   function mat_to_c(mat) result(handle)

      ! The handle of a matrix to pass to the C: a null Fortran object (0 for
      ! PETSC_NULL_MAT, or PETSC_FORTRAN_TYPE_INITIALIZE when it was never created
      ! or has been destroyed) is NULL in C

      ! ~~~~~~
      type(tMat), intent(in) :: mat
      integer(c_long_long)   :: handle
      ! ~~~~~~

      if (PetscObjectIsNull(mat)) then
         handle = 0
      else
         handle = mat%v
      end if

   end function mat_to_c

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine mat_from_c(handle, mat)

      ! Sets a matrix from a handle the C returned. A NULL handle leaves a null
      ! Fortran object as it is and otherwise sets it to the destroyed state, which
      ! is what the Fortran MatDestroy does

      ! ~~~~~~
      integer(c_long_long), intent(in) :: handle
      type(tMat), intent(inout)        :: mat
      ! ~~~~~~

      if (handle == 0) then
         if (.NOT. PetscObjectIsNull(mat)) then
            PetscObjectNullify(mat)
         end if
      else
         mat%v = handle
      end if

   end subroutine mat_from_c

#if defined(PFLARE_PORT_DEBUG)
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine port_copy_mat(mat, mat_copy)

      ! Returns a copy of mat with the same entries (explicit zeros included), built with
      ! MatSetValues and assembled, for a reference that sets values in a matrix it is given
      ! MatDuplicate does not do here: the duplicate of a parallel AIJ matrix is not marked as
      ! assembled before (was_assembled), so MatSetValues then takes the global column indices
      ! of the off-diagonal block as the local ones of its compressed storage

      ! ~~~~~~
      type(tMat), intent(in)    :: mat
      type(tMat), intent(inout) :: mat_copy

      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX
      MatType :: mat_type
      PetscInt :: local_rows, local_cols, global_rows, global_cols
      PetscInt :: global_row_start, global_row_end_plus_one, global_col_start, global_col_end_plus_one
      PetscInt :: ifree, ncols, col, i_loc
      PetscInt, parameter :: one = 1
      PetscInt, dimension(:), pointer :: cols => null()
      PetscScalar, dimension(:), pointer :: vals => null()
      PetscInt, dimension(:), allocatable :: d_nnz, o_nnz
      ! ~~~~~~

      call PetscObjectGetComm(mat, MPI_COMM_MATRIX, ierr)
      call MatGetLocalSize(mat, local_rows, local_cols, ierr)
      call MatGetSize(mat, global_rows, global_cols, ierr)
      call MatGetOwnershipRange(mat, global_row_start, global_row_end_plus_one, ierr)
      call MatGetOwnershipRangeColumn(mat, global_col_start, global_col_end_plus_one, ierr)

      call MatCreate(MPI_COMM_MATRIX, mat_copy, ierr)
      call MatSetSizes(mat_copy, local_rows, local_cols, global_rows, global_cols, ierr)
      call MatGetType(mat, mat_type, ierr)
      call MatSetType(mat_copy, mat_type, ierr)

      allocate(d_nnz(local_rows), o_nnz(local_rows))
      d_nnz = 0
      o_nnz = 0
      do ifree = global_row_start, global_row_end_plus_one-1
         call MatGetRow(mat, ifree, ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)
         i_loc = ifree - global_row_start + 1
         do col = 1, ncols
            if (cols(col) >= global_col_start .AND. cols(col) < global_col_end_plus_one) then
               d_nnz(i_loc) = d_nnz(i_loc) + 1
            else
               o_nnz(i_loc) = o_nnz(i_loc) + 1
            end if
         end do
         call MatRestoreRow(mat, ifree, ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)
      end do
      call MatXAIJSetPreallocation(mat_copy, one, d_nnz, o_nnz, &
               PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_INTEGER_ARRAY, ierr)
      deallocate(d_nnz, o_nnz)

      do ifree = global_row_start, global_row_end_plus_one-1
         call MatGetRow(mat, ifree, ncols, cols, vals, ierr)
         if (ncols > 0) then
            call MatSetValues(mat_copy, one, [ifree], ncols, cols(1:ncols), vals(1:ncols), INSERT_VALUES, ierr)
         end if
         call MatRestoreRow(mat, ifree, ncols, cols, vals, ierr)
      end do
      call MatAssemblyBegin(mat_copy, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(mat_copy, MAT_FINAL_ASSEMBLY, ierr)

   end subroutine port_copy_mat

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine port_compare_mats(mat, mat_fref, routine)

      ! Aborts unless the matrices the C and the Fortran reference built are the same:
      ! the same sizes and, row by row, the same global column indices (explicit zeros
      ! included) and bitwise the same values
      ! This is compared through MatGetRow rather than with MatEqual, which compares the
      ! compressed off-diagonal block of a parallel AIJ matrix without its column map

      ! ~~~~~~
      type(tMat), intent(in)       :: mat, mat_fref
      character(len=*), intent(in) :: routine

      PetscErrorCode :: ierr
      PetscInt :: local_rows, local_cols, global_rows, global_cols
      PetscInt :: local_rows_fref, local_cols_fref, global_rows_fref, global_cols_fref
      PetscInt :: global_row_start, global_row_end_plus_one, ifree, ncols, ncols_fref
      PetscInt, dimension(:), pointer :: cols => null(), cols_fref => null()
      PetscScalar, dimension(:), pointer :: vals => null(), vals_fref => null()
      logical :: mats_equal
      integer :: errorcode
      ! ~~~~~~

      ! Both can be the same matrix (eg MatMPICreateNonemptySubcomm returns a reference
      ! to its input when no rank is empty), which only one MatGetRow can be active on
      if (mat%v == mat_fref%v) return

      call MatGetLocalSize(mat, local_rows, local_cols, ierr)
      call MatGetSize(mat, global_rows, global_cols, ierr)
      call MatGetLocalSize(mat_fref, local_rows_fref, local_cols_fref, ierr)
      call MatGetSize(mat_fref, global_rows_fref, global_cols_fref, ierr)
      mats_equal = local_rows == local_rows_fref .AND. local_cols == local_cols_fref .AND. &
                   global_rows == global_rows_fref .AND. global_cols == global_cols_fref

      if (mats_equal) then
         call MatGetOwnershipRange(mat, global_row_start, global_row_end_plus_one, ierr)
         do ifree = global_row_start, global_row_end_plus_one-1
            call MatGetRow(mat, ifree, ncols, cols, vals, ierr)
            call MatGetRow(mat_fref, ifree, ncols_fref, cols_fref, vals_fref, ierr)
            if (ncols /= ncols_fref) then
               mats_equal = .FALSE.
            else if (ncols > 0) then
               if (any(cols(1:ncols) /= cols_fref(1:ncols)) .OR. any(vals(1:ncols) /= vals_fref(1:ncols))) then
                  mats_equal = .FALSE.
               end if
            end if
            call MatRestoreRow(mat_fref, ifree, ncols_fref, cols_fref, vals_fref, ierr)
            call MatRestoreRow(mat, ifree, ncols, cols, vals, ierr)
            if (.NOT. mats_equal) exit
         end do
      end if

      if (.NOT. mats_equal) then
         print *, "C and Fortran reference versions of ", routine, " do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

   end subroutine port_compare_mats
#endif

! -------------------------------------------------------------------------------------------------------------------------------
! Timers.c
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine print_timers()

      ! ~~~~~~
      PetscErrorCode :: ierr
      ! ~~~~~~

      ! The C prints through PetscPrintf, which flushes the C stdout straight away,
      ! so flush what the Fortran has printed first to keep the output in order
      flush(output_unit)
      ierr = print_timers_c()
      call port_check(ierr, "print_timers")

   end subroutine print_timers

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine timer_start(id)

      ! ~~~~~~
      integer, intent(in) :: id

      PetscErrorCode :: ierr
      ! ~~~~~~

      ierr = timer_start_c(id)
      call port_check(ierr, "timer_start")

   end subroutine timer_start

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine timer_finish(id)

      ! ~~~~~~
      integer, intent(in) :: id

      PetscErrorCode :: ierr
      ! ~~~~~~

      ierr = timer_finish_c(id)
      call port_check(ierr, "timer_finish")

   end subroutine timer_finish

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine timer_reset()

      ! ~~~~~~
      PetscErrorCode :: ierr
      ! ~~~~~~

      ierr = timer_reset_c()
      call port_check(ierr, "timer_reset")

   end subroutine timer_reset

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine timer_clear(id)

      ! ~~~~~~
      integer, intent(in) :: id

      PetscErrorCode :: ierr
      ! ~~~~~~

      ierr = timer_clear_c(id)
      call port_check(ierr, "timer_clear")

   end subroutine timer_clear

! -------------------------------------------------------------------------------------------------------------------------------

   PetscReal function timer_time(id)

      ! ~~~~~~
      integer, intent(in) :: id

      PetscErrorCode :: ierr
      ! ~~~~~~

      ierr = timer_time_c(id, timer_time)
      call port_check(ierr, "timer_time")

   end function timer_time

! -------------------------------------------------------------------------------------------------------------------------------
! Sorting.c
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine sorted_unique(array, n)

      ! Sorts array(1:n) and removes the duplicates, in place
      ! On return n is the number of unique entries, which are in array(1:n) in ascending order
      ! The port debug comparison against the binary tree this replaces is at the call site

      ! ~~~~~~
      PetscInt, dimension(:), intent(inout) :: array
      PetscInt, intent(inout)               :: n

      PetscErrorCode :: ierr
      ! ~~~~~~

      ierr = sorted_unique_c(array, n)
      call port_check(ierr, "sorted_unique")

   end subroutine sorted_unique

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine intersect_pre_sorted_indices_only(array1, array2, indices_array1, indices_array2, intersect_count)

      ! ~~~~~~
      PetscInt, dimension(:), intent(in) :: array1, array2
      integer, dimension(:), intent(out)  :: indices_array1, indices_array2
      integer, intent(out) :: intersect_count

      PetscInt :: size_array1, size_array2
      PetscErrorCode :: ierr
#if defined(PFLARE_PORT_DEBUG)
      integer, dimension(size(indices_array1)) :: indices_array1_fref
      integer, dimension(size(indices_array2)) :: indices_array2_fref
      integer :: intersect_count_fref, errorcode
#endif
      ! ~~~~~~

      size_array1 = size(array1)
      size_array2 = size(array2)
      ierr = intersect_pre_sorted_indices_only_c(array1, size_array1, array2, size_array2, &
                  indices_array1, indices_array2, intersect_count)
      call port_check(ierr, "intersect_pre_sorted_indices_only")
      ! The C returns 0-based indices
      indices_array1(1:intersect_count) = indices_array1(1:intersect_count) + 1
      indices_array2(1:intersect_count) = indices_array2(1:intersect_count) + 1

#if defined(PFLARE_PORT_DEBUG)
      call intersect_pre_sorted_indices_only_fref(array1, array2, indices_array1_fref, indices_array2_fref, &
                  intersect_count_fref)
      if (intersect_count /= intersect_count_fref) then
         print *, "C and Fortran reference versions of intersect_pre_sorted_indices_only do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      if (any(indices_array1(1:intersect_count) /= indices_array1_fref(1:intersect_count)) .OR. &
          any(indices_array2(1:intersect_count) /= indices_array2_fref(1:intersect_count))) then
         print *, "C and Fortran reference versions of intersect_pre_sorted_indices_only do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
#endif

   end subroutine intersect_pre_sorted_indices_only

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine merge_pre_sorted(array1, array2, sorted_array)

      ! ~~~~~~
      PetscInt, dimension(:), intent(in) :: array1, array2
      PetscInt, dimension(:), intent(out)  :: sorted_array

      PetscInt :: size_array1, size_array2
      PetscErrorCode :: ierr
#if defined(PFLARE_PORT_DEBUG)
      PetscInt, dimension(size(sorted_array)) :: sorted_array_fref
      integer :: errorcode
#endif
      ! ~~~~~~

      size_array1 = size(array1)
      size_array2 = size(array2)
      ierr = merge_pre_sorted_c(array1, size_array1, array2, size_array2, sorted_array)
      call port_check(ierr, "merge_pre_sorted")

#if defined(PFLARE_PORT_DEBUG)
      call merge_pre_sorted_fref(array1, array2, sorted_array_fref)
      if (any(sorted_array /= sorted_array_fref)) then
         print *, "C and Fortran reference versions of merge_pre_sorted do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
#endif

   end subroutine merge_pre_sorted

! -------------------------------------------------------------------------------------------------------------------------------
! Aggregation.c
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine generate_serial_aggregation(strength_mat, cf_markers, aggregates)

      ! The C takes cf_markers as input if it is allocated, and otherwise allocates it,
      ! and always allocates aggregates (with PetscMalloc). The arrays the C allocates
      ! are copied into the Fortran allocatables and freed

      ! ~~~~~~
      type(tMat), target, intent(in)      :: strength_mat
      integer, dimension(:), allocatable, target, intent(inout) :: cf_markers
      PetscInt, dimension(:), allocatable, intent(inout) :: aggregates

      PetscInt :: local_rows, local_cols
      PetscErrorCode :: ierr
      logical :: cf_markers_allocated
      type(c_ptr) :: cf_markers_ptr, aggregates_ptr
      integer, dimension(:), pointer :: cf_markers_c
      PetscInt, dimension(:), pointer :: aggregates_c
#if defined(PFLARE_PORT_DEBUG)
      integer, dimension(:), allocatable :: cf_markers_fref
      PetscInt, dimension(:), allocatable :: aggregates_fref
      integer :: errorcode
#endif
      ! ~~~~~~

      call MatGetLocalSize(strength_mat, local_rows, local_cols, ierr)

#if defined(PFLARE_PORT_DEBUG)
      ! The C modifies cf_markers in place, so give the reference a copy of the input
      if (allocated(cf_markers)) then
         allocate(cf_markers_fref(size(cf_markers)))
         cf_markers_fref = cf_markers
      end if
      call generate_serial_aggregation_fref(strength_mat, cf_markers_fref, aggregates_fref)
#endif

      cf_markers_allocated = allocated(cf_markers)
      if (cf_markers_allocated) then
         cf_markers_ptr = c_loc(cf_markers)
      else
         cf_markers_ptr = c_null_ptr
      end if
      aggregates_ptr = c_null_ptr

      ierr = generate_serial_aggregation_c(strength_mat%v, cf_markers_ptr, aggregates_ptr)
      call port_check(ierr, "generate_serial_aggregation")

      ! Copy what the C allocated into the Fortran allocatables
      if (.NOT. cf_markers_allocated) then
         allocate(cf_markers(local_rows))
         if (local_rows > 0) then
            call c_f_pointer(cf_markers_ptr, cf_markers_c, [local_rows])
            cf_markers = cf_markers_c
         end if
         ierr = PetscFree_c(cf_markers_ptr)
         call port_check(ierr, "PetscFree_c")
      end if
      allocate(aggregates(local_rows))
      if (local_rows > 0) then
         call c_f_pointer(aggregates_ptr, aggregates_c, [local_rows])
         aggregates = aggregates_c
      end if
      ierr = PetscFree_c(aggregates_ptr)
      call port_check(ierr, "PetscFree_c")

#if defined(PFLARE_PORT_DEBUG)
      if (size(cf_markers) /= size(cf_markers_fref) .OR. size(aggregates) /= size(aggregates_fref)) then
         print *, "C and Fortran reference versions of generate_serial_aggregation do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      if (any(cf_markers /= cf_markers_fref) .OR. any(aggregates /= aggregates_fref)) then
         print *, "C and Fortran reference versions of generate_serial_aggregation do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      deallocate(cf_markers_fref, aggregates_fref)
#endif

   end subroutine generate_serial_aggregation

! -------------------------------------------------------------------------------------------------------------------------------
! Weighted_Jacobi.c
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine calculate_and_build_weighted_jacobi_inverse(matrix, inverse_type, inv_matrix)

      ! ~~~~~~
      type(tMat), target, intent(in)                    :: matrix
      integer, intent(in)                               :: inverse_type
      type(tMat), intent(inout)                         :: inv_matrix

      PetscErrorCode :: ierr
      integer(c_int) :: inverse_type_int
      integer(c_long_long) :: matrix_array, inv_matrix_array
#if defined(PFLARE_PORT_DEBUG)
      type(tMat) :: inv_matrix_fref
      type(tVec) :: diag_vec, diag_vec_fref
      PetscBool :: vec_equal
      integer :: errorcode
      MatType :: mat_type
#endif
      ! ~~~~~~

#if defined(PFLARE_PORT_DEBUG)
      ! The C reuses inv_matrix in place, so give the reference a copy of the input
      ! The diagonal of a MATDIAGONAL is copied with VecDuplicate rather than with
      ! MatDuplicate, whose new diagonal does not keep the vector type (eg Kokkos)
      PetscObjectNullify(inv_matrix_fref)
      if (.NOT. PetscObjectIsNull(inv_matrix)) then
         call MatGetType(inv_matrix, mat_type, ierr)
         if (mat_type == MATDIAGONAL) then
            call MatDiagonalGetDiagonal(inv_matrix, diag_vec, ierr)
            call VecDuplicate(diag_vec, diag_vec_fref, ierr)
            call VecCopy(diag_vec, diag_vec_fref, ierr)
            call MatDiagonalRestoreDiagonal(inv_matrix, diag_vec, ierr)
            ! The matrix takes ownership of diag_vec_fref
            call MatCreateDiagonal(diag_vec_fref, inv_matrix_fref, ierr)
            call VecDestroy(diag_vec_fref, ierr)
         else
            call MatDuplicate(inv_matrix, MAT_COPY_VALUES, inv_matrix_fref, ierr)
         end if
      end if
#endif

      inverse_type_int = inverse_type
      matrix_array = matrix%v
      ! A null Fortran handle is not necessarily 0, so pass NULL explicitly
      if (PetscObjectIsNull(inv_matrix)) then
         inv_matrix_array = 0
      else
         inv_matrix_array = inv_matrix%v
      end if
      ierr = calculate_and_build_weighted_jacobi_inverse_c(matrix_array, inverse_type_int, inv_matrix_array)
      call port_check(ierr, "calculate_and_build_weighted_jacobi_inverse")
      inv_matrix%v = inv_matrix_array

#if defined(PFLARE_PORT_DEBUG)
      call calculate_and_build_weighted_jacobi_inverse_fref(matrix, inverse_type, inv_matrix_fref)
      call MatCreateVecs(matrix, PETSC_NULL_VEC, diag_vec, ierr)
      call VecDuplicate(diag_vec, diag_vec_fref, ierr)
      call MatGetDiagonal(inv_matrix, diag_vec, ierr)
      call MatGetDiagonal(inv_matrix_fref, diag_vec_fref, ierr)
      call VecEqual(diag_vec, diag_vec_fref, vec_equal, ierr)
      if (.NOT. vec_equal) then
         print *, "C and Fortran reference versions of calculate_and_build_weighted_jacobi_inverse do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      call VecDestroy(diag_vec, ierr)
      call VecDestroy(diag_vec_fref, ierr)
      call MatDestroy(inv_matrix_fref, ierr)
#endif

   end subroutine calculate_and_build_weighted_jacobi_inverse

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine calculate_l1_jacobi_inverse_diagonal(matrix, diag_vec)

      ! ~~~~~~
      type(tMat), intent(in)                            :: matrix
      type(tVec), intent(inout)                         :: diag_vec

      PetscErrorCode :: ierr
#if defined(PFLARE_PORT_DEBUG)
      type(tVec) :: diag_vec_fref
      PetscBool :: vec_equal
      integer :: errorcode
#endif
      ! ~~~~~~

#if defined(PFLARE_PORT_DEBUG)
      ! The C overwrites diag_vec in place, so give the reference a copy of the input
      call VecDuplicate(diag_vec, diag_vec_fref, ierr)
      call VecCopy(diag_vec, diag_vec_fref, ierr)
#endif

      ierr = calculate_l1_jacobi_inverse_diagonal_c(matrix%v, diag_vec%v)
      call port_check(ierr, "calculate_l1_jacobi_inverse_diagonal")

#if defined(PFLARE_PORT_DEBUG)
      call calculate_l1_jacobi_inverse_diagonal_fref(matrix, diag_vec_fref)
      call VecEqual(diag_vec, diag_vec_fref, vec_equal, ierr)
      if (.NOT. vec_equal) then
         print *, "C and Fortran reference versions of calculate_l1_jacobi_inverse_diagonal do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      call VecDestroy(diag_vec_fref, ierr)
#endif

   end subroutine calculate_l1_jacobi_inverse_diagonal

! -------------------------------------------------------------------------------------------------------------------------------
! PETSc_Helper.c
! -------------------------------------------------------------------------------------------------------------------------------

   function kokkos_debug()

      ! The C reads the environment variable once and caches it, as the Fortran did

      ! ~~~~~~
      logical :: kokkos_debug

#if defined(PFLARE_PORT_DEBUG)
      integer :: errorcode
#endif
      ! ~~~~~~

      kokkos_debug = kokkos_debug_c()

#if defined(PFLARE_PORT_DEBUG)
      if (kokkos_debug .neqv. kokkos_debug_fref()) then
         print *, "C and Fortran reference versions of kokkos_debug do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
#endif

   end function kokkos_debug

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine destroy_matrix_reuse(mat, submatrices)

      ! The array of submatrices is the C array that MatCreateSubMatrices allocated
      ! (the Fortran pointer is created over it without a copy), so the C gets its
      ! address and destroys it with MatDestroySubMatrices
      ! No port debug comparison, this is a sequence of PETSc calls

      ! ~~~~~~
      type(tMat), intent(inout)                          :: mat
      type(tMat), dimension(:), pointer, intent(inout)   :: submatrices

      PetscErrorCode :: ierr
      integer(c_long_long) :: mat_handle
      type(c_ptr) :: submatrices_ptr
      logical :: submatrices_associated
      ! ~~~~~~

      submatrices_associated = associated(submatrices)
      if (submatrices_associated) then
         submatrices_ptr = c_loc(submatrices(1))
      else
         submatrices_ptr = c_null_ptr
      end if
      mat_handle = mat_to_c(mat)

      ierr = destroy_matrix_reuse_c(mat_handle, submatrices_ptr)
      call port_check(ierr, "destroy_matrix_reuse")

      if (submatrices_associated) then
         submatrices => null()
         ! MatDestroySubMatrices doesn't leave the mat in the same null state
         ! as calling matdestroy
         PetscObjectNullify(mat)
      else
         call mat_from_c(mat_handle, mat)
      end if

   end subroutine destroy_matrix_reuse

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine remove_small_from_sparse(input_mat, tol, output_mat, relative_max_row_tol_int, &
                  lump, drop_diagonal_int, diag_strength_int)

      ! ~~~~~~
      type(tMat), intent(in) :: input_mat
      type(tMat), intent(inout) :: output_mat
      PetscReal, intent(in) :: tol
      logical, intent(in), optional :: lump
      integer, intent(in), optional :: relative_max_row_tol_int, drop_diagonal_int, diag_strength_int

      PetscErrorCode :: ierr
      integer(c_long_long) :: output_mat_handle
      integer(c_int), target :: relative_max_row_tol_int_c, drop_diagonal_int_c, diag_strength_int_c
      PetscBool, target :: lump_c
      type(c_ptr) :: relative_max_row_tol_int_ptr, lump_ptr, drop_diagonal_int_ptr, diag_strength_int_ptr
#if defined(PFLARE_PORT_DEBUG)
      type(tMat) :: output_mat_fref
#endif
      ! ~~~~~~

      relative_max_row_tol_int_ptr = c_null_ptr
      lump_ptr = c_null_ptr
      drop_diagonal_int_ptr = c_null_ptr
      diag_strength_int_ptr = c_null_ptr
      if (present(relative_max_row_tol_int)) then
         relative_max_row_tol_int_c = relative_max_row_tol_int
         relative_max_row_tol_int_ptr = c_loc(relative_max_row_tol_int_c)
      end if
      if (present(lump)) then
         lump_c = lump
         lump_ptr = c_loc(lump_c)
      end if
      if (present(drop_diagonal_int)) then
         drop_diagonal_int_c = drop_diagonal_int
         drop_diagonal_int_ptr = c_loc(drop_diagonal_int_c)
      end if
      if (present(diag_strength_int)) then
         diag_strength_int_c = diag_strength_int
         diag_strength_int_ptr = c_loc(diag_strength_int_c)
      end if

      output_mat_handle = mat_to_c(output_mat)
      ierr = remove_small_from_sparse_c(input_mat%v, tol, output_mat_handle, relative_max_row_tol_int_ptr, &
                  lump_ptr, drop_diagonal_int_ptr, diag_strength_int_ptr)
      call port_check(ierr, "remove_small_from_sparse")
      call mat_from_c(output_mat_handle, output_mat)

#if defined(PFLARE_PORT_DEBUG)
      call remove_small_from_sparse_fref(input_mat, tol, output_mat_fref, relative_max_row_tol_int, &
                  lump, drop_diagonal_int, diag_strength_int)
      call port_compare_mats(output_mat, output_mat_fref, "remove_small_from_sparse")
      call MatDestroy(output_mat_fref, ierr)
#endif

   end subroutine remove_small_from_sparse

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine remove_from_sparse_match(input_mat, output_mat, lump, alpha)

      ! ~~~~~~
      type(tMat), intent(in) :: input_mat
      type(tMat), intent(inout) :: output_mat
      logical, intent(in), optional :: lump
      PetscReal, intent(in), optional :: alpha

      PetscErrorCode :: ierr
      PetscBool, target :: lump_c
      PetscReal, target :: alpha_c
      type(c_ptr) :: lump_ptr, alpha_ptr
#if defined(PFLARE_PORT_DEBUG)
      type(tMat) :: output_mat_fref
#endif
      ! ~~~~~~

#if defined(PFLARE_PORT_DEBUG)
      ! The C changes the values of output_mat, so give the reference a copy of the input
      call port_copy_mat(output_mat, output_mat_fref)
#endif

      lump_ptr = c_null_ptr
      alpha_ptr = c_null_ptr
      if (present(lump)) then
         lump_c = lump
         lump_ptr = c_loc(lump_c)
      end if
      if (present(alpha)) then
         alpha_c = alpha
         alpha_ptr = c_loc(alpha_c)
      end if

      ierr = remove_from_sparse_match_c(input_mat%v, output_mat%v, lump_ptr, alpha_ptr)
      call port_check(ierr, "remove_from_sparse_match")

#if defined(PFLARE_PORT_DEBUG)
      call remove_from_sparse_match_fref(input_mat, output_mat_fref, lump, alpha)
      call port_compare_mats(output_mat, output_mat_fref, "remove_from_sparse_match")
      call MatDestroy(output_mat_fref, ierr)
#endif

   end subroutine remove_from_sparse_match

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine MatSetAllValues(input_mat, val)

      ! ~~~~~~
      type(tMat), intent(in)  :: input_mat
      PetscScalar, intent(in) :: val

      PetscErrorCode :: ierr
#if defined(PFLARE_PORT_DEBUG)
      type(tMat) :: input_mat_fref
#endif
      ! ~~~~~~

#if defined(PFLARE_PORT_DEBUG)
      ! The C changes the values of input_mat, so give the reference a copy of the input
      call port_copy_mat(input_mat, input_mat_fref)
#endif

      ierr = MatSetAllValues_c(input_mat%v, val)
      call port_check(ierr, "MatSetAllValues")

#if defined(PFLARE_PORT_DEBUG)
      call MatSetAllValues_fref(input_mat_fref, val)
      call port_compare_mats(input_mat, input_mat_fref, "MatSetAllValues")
      call MatDestroy(input_mat_fref, ierr)
#endif

   end subroutine MatSetAllValues

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine mat_duplicate_copy_plus_diag(input_mat, reuse, output_mat)

      ! ~~~~~~
      type(tMat), intent(in) :: input_mat
      logical, intent(in) :: reuse
      type(tMat), intent(inout) :: output_mat

      PetscErrorCode :: ierr
      PetscBool :: reuse_c
      integer(c_long_long) :: output_mat_handle
#if defined(PFLARE_PORT_DEBUG)
      type(tMat) :: output_mat_fref
#endif
      ! ~~~~~~

#if defined(PFLARE_PORT_DEBUG)
      ! When reusing the C changes the values of output_mat, so give the reference a copy
      PetscObjectNullify(output_mat_fref)
      if (reuse) call port_copy_mat(output_mat, output_mat_fref)
#endif

      reuse_c = reuse
      output_mat_handle = mat_to_c(output_mat)
      ierr = mat_duplicate_copy_plus_diag_c(input_mat%v, reuse_c, output_mat_handle)
      call port_check(ierr, "mat_duplicate_copy_plus_diag")
      call mat_from_c(output_mat_handle, output_mat)

#if defined(PFLARE_PORT_DEBUG)
      call mat_duplicate_copy_plus_diag_fref(input_mat, reuse, output_mat_fref)
      call port_compare_mats(output_mat, output_mat_fref, "mat_duplicate_copy_plus_diag")
      call MatDestroy(output_mat_fref, ierr)
#endif

   end subroutine mat_duplicate_copy_plus_diag

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine MatAXPYWrapper(y_mat, alpha, x_mat)

      ! No port debug comparison, this is a wrapper around MatAXPY

      ! ~~~~~~
      type(tMat), intent(inout) :: y_mat
      type(tMat), intent(in)    :: x_mat
      PetscScalar, intent(in)   :: alpha

      PetscErrorCode :: ierr
      ! ~~~~~~

      ierr = MatAXPYWrapper_c(y_mat%v, alpha, x_mat%v)
      call port_check(ierr, "MatAXPYWrapper")

   end subroutine MatAXPYWrapper

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine MatCreateSubMatrixWrapper(input_mat, is_row, is_col, &
                  reuse, output_mat, &
                  our_level, is_row_fine, is_col_fine, kokkos_is_views_handle)

      ! With PFLARE_PORT_DEBUG the result is compared with the Fortran reference only
      ! when a new matrix is created, as reusing needs the matrix that MatCreateSubMatrix
      ! created before (it is a wrapper around MatCreateSubMatrix and its Kokkos version)

      ! ~~~~~~
      type(tMat), intent(in)        :: input_mat
      IS, intent(in)                :: is_row, is_col
      integer, intent(in), optional :: our_level
      logical, intent(in), optional :: is_row_fine, is_col_fine
      type(c_ptr), intent(in), optional :: kokkos_is_views_handle
      MatReuse, intent(in)          :: reuse
      type(tMat), intent(inout)     :: output_mat

      PetscErrorCode :: ierr
      integer(c_long_long) :: output_mat_handle
      integer(c_int), target :: our_level_c
      PetscBool, target :: is_row_fine_c, is_col_fine_c
      type(c_ptr) :: our_level_ptr, is_row_fine_ptr, is_col_fine_ptr, handle_local
#if defined(PFLARE_PORT_DEBUG)
      type(tMat) :: output_mat_fref
#endif
      ! ~~~~~~

      our_level_ptr = c_null_ptr
      is_row_fine_ptr = c_null_ptr
      is_col_fine_ptr = c_null_ptr
      handle_local = c_null_ptr
      if (present(our_level)) then
         our_level_c = our_level
         our_level_ptr = c_loc(our_level_c)
      end if
      if (present(is_row_fine)) then
         is_row_fine_c = is_row_fine
         is_row_fine_ptr = c_loc(is_row_fine_c)
      end if
      if (present(is_col_fine)) then
         is_col_fine_c = is_col_fine
         is_col_fine_ptr = c_loc(is_col_fine_c)
      end if
      if (present(kokkos_is_views_handle)) then
         handle_local = kokkos_is_views_handle
      end if

      output_mat_handle = mat_to_c(output_mat)
      ierr = MatCreateSubMatrixWrapper_c(input_mat%v, is_row%v, is_col%v, reuse%v, output_mat_handle, &
                  our_level_ptr, is_row_fine_ptr, is_col_fine_ptr, handle_local)
      call port_check(ierr, "MatCreateSubMatrixWrapper")
      call mat_from_c(output_mat_handle, output_mat)

#if defined(PFLARE_PORT_DEBUG)
      if (reuse == MAT_INITIAL_MATRIX) then
         call MatCreateSubMatrixWrapper_fref(input_mat, is_row, is_col, &
                  reuse, output_mat_fref, &
                  our_level, is_row_fine, is_col_fine, kokkos_is_views_handle)
         call port_compare_mats(output_mat, output_mat_fref, "MatCreateSubMatrixWrapper")
         call MatDestroy(output_mat_fref, ierr)
      end if
#endif

   end subroutine MatCreateSubMatrixWrapper

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine generate_identity(input_mat, output_mat)

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
      ierr = generate_identity_c(input_mat%v, output_mat_handle)
      call port_check(ierr, "generate_identity")
      call mat_from_c(output_mat_handle, output_mat)

#if defined(PFLARE_PORT_DEBUG)
      call generate_identity_fref(input_mat, output_mat_fref)
      call port_compare_mats(output_mat, output_mat_fref, "generate_identity")
      call MatDestroy(output_mat_fref, ierr)
#endif

   end subroutine generate_identity

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine generate_identity_rect(full_mat, rect_indices, output_mat)

      ! ~~~~~~
      type(tMat), intent(in)    :: full_mat
      type(tIS), intent(in)     :: rect_indices
      type(tMat), intent(inout) :: output_mat

      PetscErrorCode :: ierr
      integer(c_long_long) :: output_mat_handle
#if defined(PFLARE_PORT_DEBUG)
      type(tMat) :: output_mat_fref
#endif
      ! ~~~~~~

      output_mat_handle = mat_to_c(output_mat)
      ierr = generate_identity_rect_c(full_mat%v, rect_indices%v, output_mat_handle)
      call port_check(ierr, "generate_identity_rect")
      call mat_from_c(output_mat_handle, output_mat)

#if defined(PFLARE_PORT_DEBUG)
      call generate_identity_rect_fref(full_mat, rect_indices, output_mat_fref)
      call port_compare_mats(output_mat, output_mat_fref, "generate_identity_rect")
      call MatDestroy(output_mat_fref, ierr)
#endif

   end subroutine generate_identity_rect

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine generate_identity_is(input_mat, indices, output_mat)

      ! ~~~~~~
      type(tMat), intent(in)     :: input_mat
      type(tIS), intent(in)      :: indices
      type(tMat), intent(inout)  :: output_mat

      PetscErrorCode :: ierr
      integer(c_long_long) :: output_mat_handle
#if defined(PFLARE_PORT_DEBUG)
      type(tMat) :: output_mat_fref
#endif
      ! ~~~~~~

      output_mat_handle = mat_to_c(output_mat)
      ierr = generate_identity_is_c(input_mat%v, indices%v, output_mat_handle)
      call port_check(ierr, "generate_identity_is")
      call mat_from_c(output_mat_handle, output_mat)

#if defined(PFLARE_PORT_DEBUG)
      call generate_identity_is_fref(input_mat, indices, output_mat_fref)
      call port_compare_mats(output_mat, output_mat_fref, "generate_identity_is")
      call MatDestroy(output_mat_fref, ierr)
#endif

   end subroutine generate_identity_is

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine get_nnzs_petsc_sparse(input_mat, nnzs)

      ! ~~~~~~
      type(tMat), intent(in) :: input_mat
      integer(kind=8), intent(out) :: nnzs

      PetscErrorCode :: ierr
      integer(c_int64_t) :: nnzs_c
#if defined(PFLARE_PORT_DEBUG)
      integer(kind=8) :: nnzs_fref
      integer :: errorcode
#endif
      ! ~~~~~~

      ierr = get_nnzs_petsc_sparse_c(input_mat%v, nnzs_c)
      call port_check(ierr, "get_nnzs_petsc_sparse")
      nnzs = nnzs_c

#if defined(PFLARE_PORT_DEBUG)
      call get_nnzs_petsc_sparse_fref(input_mat, nnzs_fref)
      if (nnzs /= nnzs_fref) then
         print *, "C and Fortran reference versions of get_nnzs_petsc_sparse do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
#endif

   end subroutine get_nnzs_petsc_sparse

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine MatGetNNZs_both(input_mat, nnzs_local, nnzs_nonlocal)

      ! ~~~~~~
      type(tMat), intent(in)  :: input_mat
      PetscInt, intent(out)   :: nnzs_local, nnzs_nonlocal

      PetscErrorCode :: ierr
#if defined(PFLARE_PORT_DEBUG)
      PetscInt :: nnzs_local_fref, nnzs_nonlocal_fref
      integer :: errorcode
#endif
      ! ~~~~~~

      ierr = MatGetNNZs_both_c(input_mat%v, nnzs_local, nnzs_nonlocal)
      call port_check(ierr, "MatGetNNZs_both")

#if defined(PFLARE_PORT_DEBUG)
      call MatGetNNZs_both_fref(input_mat, nnzs_local_fref, nnzs_nonlocal_fref)
      if (nnzs_local /= nnzs_local_fref .OR. nnzs_nonlocal /= nnzs_nonlocal_fref) then
         print *, "C and Fortran reference versions of MatGetNNZs_both do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
#endif

   end subroutine MatGetNNZs_both

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine svd(input, U, sigma, VT)

      ! ~~~~~~
      PetscScalar, dimension(:, :), intent(in) :: input
      PetscScalar, dimension(size(input, 1), size(input, 1)), intent(out) :: U
      PetscReal, dimension(min(size(input, 1), size(input, 2))), intent(out) :: sigma
      PetscScalar, dimension(size(input, 2), size(input, 2)), intent(out) :: VT

      PetscErrorCode :: ierr
      PetscInt :: input_rows, input_cols
#if defined(PFLARE_PORT_DEBUG)
      PetscScalar, dimension(size(input, 1), size(input, 1)) :: U_fref
      PetscReal, dimension(min(size(input, 1), size(input, 2))) :: sigma_fref
      PetscScalar, dimension(size(input, 2), size(input, 2)) :: VT_fref
      integer :: errorcode
#endif
      ! ~~~~~~

      input_rows = size(input, 1)
      input_cols = size(input, 2)
      ierr = svd_c(input, input_rows, input_cols, U, sigma, VT)
      call port_check(ierr, "svd")

#if defined(PFLARE_PORT_DEBUG)
      call svd_fref(input, U_fref, sigma_fref, VT_fref)
      if (any(U /= U_fref) .OR. any(sigma /= sigma_fref) .OR. any(VT /= VT_fref)) then
         print *, "C and Fortran reference versions of svd do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
#endif

   end subroutine svd

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine pseudo_inv(input, output)

      ! ~~~~~~
      PetscScalar, dimension(:, :), intent(in) :: input
      PetscScalar, dimension(size(input, 2), size(input, 1)), intent(out) :: output

      PetscErrorCode :: ierr
      PetscInt :: input_rows, input_cols
#if defined(PFLARE_PORT_DEBUG)
      PetscScalar, dimension(size(input, 2), size(input, 1)) :: output_fref
      integer :: errorcode
#endif
      ! ~~~~~~

      input_rows = size(input, 1)
      input_cols = size(input, 2)
      ierr = pseudo_inv_c(input, input_rows, input_cols, output)
      call port_check(ierr, "pseudo_inv")

#if defined(PFLARE_PORT_DEBUG)
      call pseudo_inv_fref(input, output_fref)
      if (any(output /= output_fref)) then
         print *, "C and Fortran reference versions of pseudo_inv do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
#endif

   end subroutine pseudo_inv

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine mat_mat_symbolic(A, B, C)

      ! No port debug comparison, this is a sequence of PETSc calls

      ! ~~~~~~
      type(tMat), intent(in)  :: A, B
      type(tMat), intent(out) :: C

      PetscErrorCode :: ierr
      integer(c_long_long) :: C_handle
      ! ~~~~~~

      C_handle = 0
      ierr = mat_mat_symbolic_c(A%v, B%v, C_handle)
      call port_check(ierr, "mat_mat_symbolic")
      call mat_from_c(C_handle, C)

   end subroutine mat_mat_symbolic

! -------------------------------------------------------------------------------------------------------------------------------
! Repartition.c
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine compute_mat_ratio_local_nonlocal_nnzs(input_mat, no_active_cores, ratio)

      ! ~~~~~~
      type(tMat), target, intent(in)      :: input_mat
      PetscInt, intent(in)                :: no_active_cores
      PetscReal, intent(out)              :: ratio

      PetscErrorCode :: ierr
#if defined(PFLARE_PORT_DEBUG)
      PetscReal :: ratio_fref
      integer :: errorcode
#endif
      ! ~~~~~~

      ierr = compute_mat_ratio_local_nonlocal_nnzs_c(input_mat%v, no_active_cores, ratio)
      call port_check(ierr, "compute_mat_ratio_local_nonlocal_nnzs")

#if defined(PFLARE_PORT_DEBUG)
      call compute_mat_ratio_local_nonlocal_nnzs_fref(input_mat, no_active_cores, ratio_fref)
      if (ratio /= ratio_fref) then
         print *, "C and Fortran reference versions of compute_mat_ratio_local_nonlocal_nnzs do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
#endif

   end subroutine compute_mat_ratio_local_nonlocal_nnzs

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine calculate_repartition(input_mat, proc_stride, no_active_cores, simple, index)

      ! ~~~~~~
      type(tMat), target, intent(in)      :: input_mat
      PetscInt, intent(in)                :: proc_stride
      PetscInt, intent(inout)             :: no_active_cores
      logical, intent(in)                 :: simple
      type(tIS), intent(out)              :: index

      PetscErrorCode :: ierr
      PetscBool :: simple_c
      integer(c_long_long) :: index_handle
#if defined(PFLARE_PORT_DEBUG)
      PetscInt :: no_active_cores_fref
      type(tIS) :: index_fref
      PetscBool :: is_equal
      integer :: errorcode
#endif
      ! ~~~~~~

#if defined(PFLARE_PORT_DEBUG)
      ! no_active_cores is inout, so give the reference a copy of the input
      no_active_cores_fref = no_active_cores
#endif

      simple_c = simple
      index_handle = 0
      ierr = calculate_repartition_c(input_mat%v, proc_stride, no_active_cores, simple_c, index_handle)
      call port_check(ierr, "calculate_repartition")
      index%v = index_handle

#if defined(PFLARE_PORT_DEBUG)
      call calculate_repartition_fref(input_mat, proc_stride, no_active_cores_fref, simple, index_fref)
      if (no_active_cores /= no_active_cores_fref) then
         print *, "C and Fortran reference versions of calculate_repartition do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      ! The order of the indices is the new ordering, so it has to match too
      call ISEqualUnsorted(index, index_fref, is_equal, ierr)
      if (ierr /= 0 .OR. .NOT. is_equal) then
         print *, "C and Fortran reference versions of calculate_repartition do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      call ISDestroy(index_fref, ierr)
#endif

   end subroutine calculate_repartition

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine MatMPICreateNonemptySubcomm(input_mat, on_subcomm, output_mat)

      ! ~~~~~~
      type(tMat), target, intent(in)      :: input_mat
      logical, intent(out)                :: on_subcomm
      type(tMat), target, intent(inout)   :: output_mat

      PetscErrorCode :: ierr
      PetscBool :: on_subcomm_c
      integer(c_long_long) :: B_array
#if defined(PFLARE_PORT_DEBUG)
      logical :: on_subcomm_fref, output_mat_null, output_mat_fref_null
      type(tMat) :: output_mat_fref
      integer :: errorcode
#endif
      ! ~~~~~~

      ierr = MatMPICreateNonemptySubcomm_c(input_mat%v, on_subcomm_c, B_array)
      call port_check(ierr, "MatMPICreateNonemptySubcomm")
      on_subcomm = on_subcomm_c
      ! Assign the index to the mat we get back from c, which is NULL on the ranks
      ! that are not on the subcomm, as the Fortran did
      output_mat%v = B_array

#if defined(PFLARE_PORT_DEBUG)
      call MatMPICreateNonemptySubcomm_fref(input_mat, on_subcomm_fref, output_mat_fref)
      output_mat_null = PetscObjectIsNull(output_mat)
      output_mat_fref_null = PetscObjectIsNull(output_mat_fref)
      if ((on_subcomm .neqv. on_subcomm_fref) .OR. (output_mat_null .neqv. output_mat_fref_null)) then
         print *, "C and Fortran reference versions of MatMPICreateNonemptySubcomm do not match"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      ! Only the ranks on the subcomm have a matrix
      if (.NOT. output_mat_null) then
         call port_compare_mats(output_mat, output_mat_fref, "MatMPICreateNonemptySubcomm")
      end if
      call MatDestroy(output_mat_fref, ierr)
#endif

   end subroutine MatMPICreateNonemptySubcomm

! -------------------------------------------------------------------------------------------------------------------------------

end module port_interfaces
