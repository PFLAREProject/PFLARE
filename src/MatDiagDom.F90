module matdiagdom

   use iso_c_binding
   use petscmat
   use petsc_helper, only: kokkos_debug
      use c_petsc_interfaces, only: copy_diag_dom_ratio_d2h, MatDiagDomRatio_kokkos, &
         vecscatter_mat_begin_c, vecscatter_mat_end_c, vecscatter_mat_restore_c, MatSeqAIJGetArrayF90_mine
   use pflare_parameters, only: C_POINT, F_POINT

#include "petsc/finclude/petscmat.h"
#include "finclude/PETSc_ISO_Types.h"

   implicit none

   PetscReal, parameter :: dd_ratio_abs_tol = 1d-12
   PetscReal, parameter :: dd_ratio_rel_tol = 1d-10

   public   
   
   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine MatDiagDomRatio(input_mat, is_fine, cf_markers_local, diag_dom_ratio, max_dd_ratio_achieved)

      ! Wrapper for diagonal-dominance ratio computation.
      ! Chooses Kokkos or CPU implementation and optionally compares the
      ! resulting host ratios in debug mode.

      type(tMat), target, intent(in)      :: input_mat
      type(tIS), intent(in)               :: is_fine
      integer, dimension(:), intent(in)   :: cf_markers_local
      PetscReal, dimension(:), allocatable, target, intent(out) :: diag_dom_ratio
      PetscReal, intent(out)              :: max_dd_ratio_achieved

      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX

#if defined(PETSC_HAVE_KOKKOS)
      integer :: errorcode, ifree
      MatType :: mat_type
      PetscReal :: tol, diff
      integer(c_long_long) :: A_array
      PetscInt :: local_rows_aff_kokkos
      type(c_ptr) :: diag_dom_ratio_ptr
      PetscReal, dimension(:), allocatable :: diag_dom_ratio_cpu
      PetscReal :: max_dd_ratio_cpu
#endif

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)

#if defined(PETSC_HAVE_KOKKOS)
      call MatGetType(input_mat, mat_type, ierr)
      if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) then

         A_array = input_mat%v
         local_rows_aff_kokkos = 0
         max_dd_ratio_achieved = 0d0
         call MatDiagDomRatio_kokkos(A_array, max_dd_ratio_achieved, local_rows_aff_kokkos)

         if (kokkos_debug()) then
            allocate(diag_dom_ratio(local_rows_aff_kokkos))
            if (local_rows_aff_kokkos > 0) then
               diag_dom_ratio_ptr = c_loc(diag_dom_ratio)
               call copy_diag_dom_ratio_d2h(diag_dom_ratio_ptr)
            end if

            call MatDiagDomRatio_cpu(input_mat, is_fine, cf_markers_local, diag_dom_ratio_cpu, max_dd_ratio_cpu)

            tol = dd_ratio_abs_tol + dd_ratio_rel_tol * max(abs(max_dd_ratio_cpu), abs(max_dd_ratio_achieved))
            if (abs(max_dd_ratio_cpu - max_dd_ratio_achieved) > tol) then
               print *, "Kokkos and CPU MatDiagDomRatio global max do not match"
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
            end if

            do ifree = 1, size(diag_dom_ratio)
               diff = abs(diag_dom_ratio(ifree) - diag_dom_ratio_cpu(ifree))
               tol = dd_ratio_abs_tol + dd_ratio_rel_tol * max(abs(diag_dom_ratio(ifree)), abs(diag_dom_ratio_cpu(ifree)))
               if (diff > tol) then
                  print *, "Kokkos and CPU MatDiagDomRatio entries do not match"
                  call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
               end if
            end do

            deallocate(diag_dom_ratio_cpu)
         end if
      else
         call MatDiagDomRatio_cpu(input_mat, is_fine, cf_markers_local, diag_dom_ratio, max_dd_ratio_achieved)
      end if
#else
      call MatDiagDomRatio_cpu(input_mat, is_fine, cf_markers_local, diag_dom_ratio, max_dd_ratio_achieved)
#endif

   end subroutine MatDiagDomRatio

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine MatDiagDomRatio_cpu(input_mat, is_fine, cf_markers_local, diag_dom_ratio, max_dd_ratio_achieved)

      ! Compute diagonal dominance ratio over local fine rows of input_mat
      ! without extracting Aff. This mirrors the Kokkos MatDiagDomRatio path:
      ! sum abs(F-neighbour off-diagonals) / abs(F-diagonal), with nonlocal
      ! F markers obtained from the matrix halo scatter.

      ! ~~~~~~

      type(tMat), target, intent(in)      :: input_mat
      type(tIS), intent(in)               :: is_fine
      integer, dimension(:), intent(in)   :: cf_markers_local
      PetscReal, dimension(:), allocatable, intent(out) :: diag_dom_ratio
      PetscReal, intent(out)              :: max_dd_ratio_achieved

      ! Local
      PetscInt :: local_rows, local_cols, global_rows, global_cols, fine_size
      PetscInt :: input_row_start, input_row_end_plus_one
      PetscInt :: ifree, jfree, local_row, target_col, rows_ao, cols_ao
      PetscInt :: n_ad, n_ao
      PetscInt, parameter :: one = 1
      PetscErrorCode :: ierr
      integer :: errorcode, comm_size
      MPIU_Comm :: MPI_COMM_MATRIX
      integer(c_long_long) :: A_array, vec_long, Ad_array, Ao_array
      type(tMat) :: Ad, Ao
      type(tVec) :: cf_markers_vec
      PetscInt, dimension(:), pointer :: is_pointer => null(), colmap => null()
      PetscInt, dimension(:), pointer :: ad_ia => null(), ad_ja => null(), ao_ia => null(), ao_ja => null()
      PetscReal, dimension(:), pointer :: ad_vals => null(), ao_vals => null(), cf_markers_nonlocal => null()
      PetscReal, dimension(:), allocatable, target :: cf_markers_local_real
      type(c_ptr) :: ad_vals_c_ptr, ao_vals_c_ptr, cf_markers_nonlocal_ptr
      PetscInt :: shift = 0
      PetscBool :: symmetric = PETSC_FALSE, inodecompressed = PETSC_FALSE, done
      PetscReal :: diag_val, off_diag_sum
      PetscReal :: max_dd_ratio_local
      logical :: mpi

      ! ~~~~~~

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)
      call MatGetLocalSize(input_mat, local_rows, local_cols, ierr)
      call MatGetSize(input_mat, global_rows, global_cols, ierr)
      call MatGetOwnershipRange(input_mat, input_row_start, input_row_end_plus_one, ierr)
      call ISGetLocalSize(is_fine, fine_size, ierr)
      call ISGetIndices(is_fine, is_pointer, ierr)

      allocate(diag_dom_ratio(fine_size))
      diag_dom_ratio = 0d0

      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)
      mpi = comm_size /= 1

      if (mpi) then
         call MatMPIAIJGetSeqAIJ(input_mat, Ad, Ao, colmap, ierr)
         call MatGetSize(Ao, rows_ao, cols_ao, ierr)
      else
         Ad = input_mat
      end if

      ! Get pointers to the local/off-diagonal CSR structures.
      ! This mirrors the Kokkos path, which accesses local and nonlocal CSR directly.
      call MatGetRowIJ(Ad, shift, symmetric, inodecompressed, n_ad, ad_ia, ad_ja, done, ierr)
      if (.NOT. done) then
         print *, "Pointers not set in call to MatGetRowIJ"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      if (mpi) then
         call MatGetRowIJ(Ao, shift, symmetric, inodecompressed, n_ao, ao_ia, ao_ja, done, ierr)
         if (.NOT. done) then
            print *, "Pointers not set in call to MatGetRowIJ"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
         end if
      end if

      Ad_array = Ad%v
      call MatSeqAIJGetArrayF90_mine(Ad_array, ad_vals_c_ptr)
      call c_f_pointer(ad_vals_c_ptr, ad_vals, shape=[size(ad_ja)])

      ! Off-diagonal rows require a halo exchange of cf markers.
      ! Start and finish the scatter once, then reuse the received nonlocal markers
      ! while looping over all local fine rows.
      if (mpi) then
         Ao_array = Ao%v
         call MatSeqAIJGetArrayF90_mine(Ao_array, ao_vals_c_ptr)
         call c_f_pointer(ao_vals_c_ptr, ao_vals, shape=[size(ao_ja)])

         allocate(cf_markers_local_real(local_rows))
         if (local_rows > 0) cf_markers_local_real = dble(cf_markers_local(1:local_rows))

         call VecCreateMPIWithArray(MPI_COMM_MATRIX, one, local_rows, global_rows, &
               cf_markers_local_real, cf_markers_vec, ierr)
         A_array = input_mat%v
         vec_long = cf_markers_vec%v
         call vecscatter_mat_begin_c(A_array, vec_long, cf_markers_nonlocal_ptr)
         call vecscatter_mat_end_c(A_array, vec_long, cf_markers_nonlocal_ptr)
         call c_f_pointer(cf_markers_nonlocal_ptr, cf_markers_nonlocal, shape=[cols_ao])
      end if

      ! Compute diagonal-dominance sums over the local fine-row list.
      ! For each row: accumulate abs(off-diagonal) over F neighbors only,
      ! store abs(diagonal) for an F diagonal entry, then form ratio.
      do ifree = 1, fine_size
         local_row = is_pointer(ifree) - input_row_start + 1
         diag_val = 0d0
         off_diag_sum = 0d0

         do jfree = ad_ia(local_row) + 1, ad_ia(local_row + 1)
            target_col = ad_ja(jfree) + 1

            if (cf_markers_local(target_col) /= F_POINT) cycle

            if (target_col == local_row) then
               diag_val = abs(ad_vals(jfree))
            else
               off_diag_sum = off_diag_sum + abs(ad_vals(jfree))
            end if
         end do

         if (mpi) then
            do jfree = ao_ia(local_row) + 1, ao_ia(local_row + 1)
               target_col = ao_ja(jfree) + 1

               if (nint(cf_markers_nonlocal(target_col)) /= F_POINT) cycle

               off_diag_sum = off_diag_sum + abs(ao_vals(jfree))
            end do
         end if

         ! If no diagonal was found, keep ratio at zero.
         ! This matches the Kokkos behavior for rows without a diagonal entry.
         if (diag_val /= 0d0) then
            diag_dom_ratio(ifree) = off_diag_sum / diag_val
         end if
      end do

      ! Cleanup for halo scatter resources.
      if (mpi) then
         call vecscatter_mat_restore_c(A_array, cf_markers_nonlocal_ptr)
         call VecDestroy(cf_markers_vec, ierr)
         deallocate(cf_markers_local_real)
      end if

      ! Restore CSR pointers before returning.
      call MatRestoreRowIJ(Ad, shift, symmetric, inodecompressed, n_ad, ad_ia, ad_ja, done, ierr)
      if (mpi) then
         call MatRestoreRowIJ(Ao, shift, symmetric, inodecompressed, n_ao, ao_ia, ao_ja, done, ierr)
      end if

      call ISRestoreIndices(is_fine, is_pointer, ierr)

      if (fine_size == 0) then
         max_dd_ratio_local = 0d0
      else
         max_dd_ratio_local = maxval(diag_dom_ratio)
      end if
      call MPI_Allreduce(max_dd_ratio_local, max_dd_ratio_achieved, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_MATRIX, errorcode)

   end subroutine MatDiagDomRatio_cpu

! -------------------------------------------------------------------------------------------------------------------------------

end module matdiagdom

