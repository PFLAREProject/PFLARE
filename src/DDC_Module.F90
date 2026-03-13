module ddc_module

   use iso_c_binding
   use petscmat
   use petsc_helper, only: kokkos_debug, remove_small_from_sparse, MatCreateSubMatrixWrapper
      use c_petsc_interfaces, only: copy_cf_markers_d2h, ddc_kokkos, create_cf_is_kokkos, &
         vecscatter_mat_begin_c, vecscatter_mat_end_c, vecscatter_mat_restore_c, MatSeqAIJGetArrayF90_mine
   use sabs, only: generate_sabs
   use pmisr_module, only: pmisr_existing_measure_cf_markers
   use pflare_parameters, only: C_POINT, F_POINT

#include "petsc/finclude/petscmat.h"
#include "finclude/PETSc_ISO_Types.h"

   implicit none

   public   
   
   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine ddc(input_mat, is_fine, fraction_swap, max_dd_ratio, cf_markers_local)

      ! Second pass diagonal dominance cleanup 
      ! Flips the F definitions to C based on least diagonally dominant local rows
      ! If fraction_swap = 0 this does nothing
      ! If fraction_swap < 0 it uses abs(fraction_swap) to be a threshold 
      !  for swapping C to F based on row-wise diagonal dominance (ie alpha_diag)
      ! If fraction_swap > 0 it uses fraction_swap as the local fraction of worst C points to swap to F
      !  though it won't hit that fraction exactly as we bin the diag dom ratios for speed, it will be close to the fraction

      ! ~~~~~~
      type(tMat), target, intent(in)      :: input_mat
      type(tIS), intent(in)               :: is_fine
      PetscReal, intent(in)               :: fraction_swap
      PetscReal, intent(inout)            :: max_dd_ratio
      integer, dimension(:), allocatable, target, intent(inout) :: cf_markers_local

      type(tMat) :: Aff_ddc, Aff_transpose_ddc
      PetscErrorCode :: ierr
      logical :: trigger_dd_ratio_compute_local
      PetscInt :: local_rows, local_cols
      integer :: seed_size_ddc, comm_rank_ddc, errorcode_ddc, i_loc
      integer, dimension(:), allocatable :: seed_ddc
      PetscReal, dimension(:), allocatable, target :: diag_dom_ratio_random
      type(c_ptr) :: random_numbers_ptr
      MPIU_Comm :: MPI_COMM_MATRIX

#if defined(PETSC_HAVE_KOKKOS)
      integer(c_long_long) :: A_array, Aff_transpose_array, is_fine_array, is_coarse_array
      MatType :: mat_type
      type(c_ptr)  :: cf_markers_local_ptr
   integer :: errorcode
      !integer :: kfree
      integer, dimension(:), allocatable :: cf_markers_local_two
      PetscReal :: max_dd_ratio_cpu, max_dd_ratio_kokkos
      type(tIS) :: is_fine_temp, is_coarse_temp
#endif
      ! ~~~~~~

      ! If we don't need to swap anything, return
      if (fraction_swap == 0d0) then
         return
      end if

      trigger_dd_ratio_compute_local = max_dd_ratio > 0

      ! Generate random numbers for the PMIS tie-breaking in the trigger path
      ! These are generated here so both CPU and Kokkos use the same randoms
      random_numbers_ptr = c_null_ptr
      if (trigger_dd_ratio_compute_local) then
         
         call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)
         call MPI_Comm_rank(MPI_COMM_MATRIX, comm_rank_ddc, errorcode_ddc)
         call MatGetLocalSize(input_mat, local_rows, local_cols, ierr)
         ! We allocate randoms here to be the size of input, rather than 
         ! just F points as if we are on the device the is_fine won't be allocated
         ! on the host yet
         allocate(diag_dom_ratio_random(local_rows))
         
         call random_seed(size=seed_size_ddc)
         allocate(seed_ddc(seed_size_ddc))
         
         do i_loc = 1, seed_size_ddc
            seed_ddc(i_loc) = comm_rank_ddc + 1 + i_loc
         end do
         call random_seed(put=seed_ddc)
         call random_number(diag_dom_ratio_random)

         deallocate(seed_ddc)
         random_numbers_ptr = c_loc(diag_dom_ratio_random)
      end if

#if defined(PETSC_HAVE_KOKKOS)

      call MatGetType(input_mat, mat_type, ierr)
      if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) then

         ! Kokkos path: only extract Aff and build sabs if trigger_dd_ratio_compute
         ! as the kokkos ddc computes diag dominance ratio without needing Aff
         Aff_transpose_array = 0
         A_array = input_mat%v
         if (trigger_dd_ratio_compute_local) then

            ! Create the host is_fine and is_coarse based on device cf_markers
            call create_cf_is_kokkos(A_array, is_fine_array, is_coarse_array)            
            is_fine_temp%v = is_fine_array
            is_coarse_temp%v = is_coarse_array

            call MatCreateSubMatrixWrapper(input_mat, &
                        is_fine_temp, is_fine_temp, MAT_INITIAL_MATRIX, &
                        Aff_ddc) 

            call generate_sabs(Aff_ddc, 0d0, .TRUE., .FALSE., Aff_transpose_ddc)
            call MatDestroy(Aff_ddc, ierr)
            Aff_transpose_array = Aff_transpose_ddc%v
            call ISDestroy(is_fine_temp, ierr)
            call ISDestroy(is_coarse_temp, ierr)
         end if

         cf_markers_local_ptr = c_loc(cf_markers_local)

         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then
            allocate(cf_markers_local_two(size(cf_markers_local)))
            cf_markers_local_two = cf_markers_local
         end if

         ! Modifies the existing device cf_markers created by the pmisr
         max_dd_ratio_kokkos = max_dd_ratio
         call ddc_kokkos(A_array, fraction_swap, max_dd_ratio_kokkos, Aff_transpose_array, &
               random_numbers_ptr)

         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then

            ! Kokkos DDC by default now doesn't copy back to the host, as any subsequent ddc calls
            ! use the existing device data
            call copy_cf_markers_d2h(cf_markers_local_ptr)
            max_dd_ratio_cpu = max_dd_ratio
            if (trigger_dd_ratio_compute_local) then
               call ddc_cpu(input_mat, is_fine, fraction_swap, max_dd_ratio_cpu, &
                  cf_markers_local_two, Aff_transpose=Aff_transpose_ddc, &
                  diag_dom_ratio_random=diag_dom_ratio_random)
            else
               call ddc_cpu(input_mat, is_fine, fraction_swap, max_dd_ratio_cpu, &
                  cf_markers_local_two)
            end if

            if (any(cf_markers_local /= cf_markers_local_two)) then

               ! do kfree = 1, size(cf_markers_local)
               !    if (cf_markers_local(kfree) /= cf_markers_local_two(kfree)) then
               !       print *, kfree-1, "no match", cf_markers_local(kfree), cf_markers_local_two(kfree)
               !    end if
               ! end do
               print *, "Kokkos and CPU versions of ddc do not match"
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
            end if
            deallocate(cf_markers_local_two)
         end if
         max_dd_ratio = max_dd_ratio_kokkos

         ! Cleanup
         if (trigger_dd_ratio_compute_local) then
            call MatDestroy(Aff_transpose_ddc, ierr)
         end if

      else
         ! CPU path: only extract Aff if trigger_dd_ratio_compute
         if (trigger_dd_ratio_compute_local) then
            call MatCreateSubMatrix(input_mat, &
                  is_fine, is_fine, MAT_INITIAL_MATRIX, &
                  Aff_ddc, ierr)
            call generate_sabs(Aff_ddc, 0d0, .TRUE., .FALSE., Aff_transpose_ddc)
            call ddc_cpu(input_mat, is_fine, fraction_swap, max_dd_ratio, cf_markers_local, &
                  Aff_transpose=Aff_transpose_ddc, &
                  diag_dom_ratio_random=diag_dom_ratio_random)
            call MatDestroy(Aff_ddc, ierr)
            call MatDestroy(Aff_transpose_ddc, ierr)
         else
            call ddc_cpu(input_mat, is_fine, fraction_swap, max_dd_ratio, cf_markers_local)
         end if
      end if
#else
      ! CPU path: only extract Aff if trigger_dd_ratio_compute
      if (trigger_dd_ratio_compute_local) then
         call MatCreateSubMatrix(input_mat, &
               is_fine, is_fine, MAT_INITIAL_MATRIX, &
               Aff_ddc, ierr)
         call generate_sabs(Aff_ddc, 0d0, .TRUE., .FALSE., Aff_transpose_ddc)
         call ddc_cpu(input_mat, is_fine, fraction_swap, max_dd_ratio, cf_markers_local, &
               Aff_transpose=Aff_transpose_ddc, &
               diag_dom_ratio_random=diag_dom_ratio_random)
         call MatDestroy(Aff_ddc, ierr)
         call MatDestroy(Aff_transpose_ddc, ierr)
      else
         call ddc_cpu(input_mat, is_fine, fraction_swap, max_dd_ratio, cf_markers_local)
      end if
#endif

      if (allocated(diag_dom_ratio_random)) deallocate(diag_dom_ratio_random)

   end subroutine ddc

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine MatDiagDomRatio_cpu(input_mat, is_fine, cf_markers_local, diag_dom_ratio)

      ! Compute diagonal dominance ratio over local fine rows of input_mat
      ! without extracting Aff. This mirrors the Kokkos MatDiagDomRatio path:
      ! sum abs(F-neighbour off-diagonals) / abs(F-diagonal), with nonlocal
      ! F markers obtained from the matrix halo scatter.

      ! ~~~~~~

      type(tMat), target, intent(in)      :: input_mat
      type(tIS), intent(in)               :: is_fine
      integer, dimension(:), intent(in)   :: cf_markers_local
      PetscReal, dimension(:), allocatable, intent(out) :: diag_dom_ratio

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

   end subroutine MatDiagDomRatio_cpu

! -------------------------------------------------------------------------------------------------------------------------------

      subroutine ddc_cpu(input_mat, is_fine, fraction_swap, max_dd_ratio, cf_markers_local, Aff_transpose, &
         diag_dom_ratio_random)

      ! Second pass diagonal dominance cleanup
      ! Flips the F definitions to C based on least diagonally dominant local rows
      ! If fraction_swap = 0 this does nothing
      ! If fraction_swap < 0 it uses abs(fraction_swap) to be a threshold
      !  for swapping C to F based on row-wise diagonal dominance (ie alpha_diag)
      ! If fraction_swap > 0 it uses fraction_swap as the local fraction of worst C points to swap to F
      !  though it won't hit that fraction exactly as we bin the diag dom ratios for speed, it will be close to the fraction

      ! ~~~~~~
      type(tMat), target, intent(in)      :: input_mat
      type(tIS), intent(in)               :: is_fine
      PetscReal, intent(in)               :: fraction_swap
      PetscReal, intent(inout)            :: max_dd_ratio
      integer, dimension(:), allocatable, intent(inout) :: cf_markers_local
      type(tMat), intent(in), optional    :: Aff_transpose
      PetscReal, dimension(:), intent(in), optional :: diag_dom_ratio_random

      ! Local
      PetscInt :: local_rows, one=1
      PetscInt :: ifree
      PetscInt :: input_row_start, input_row_end_plus_one
      PetscInt :: idx, search_size, fine_size, frac_size
      integer :: bin_sum, bin_boundary, bin, errorcode
      integer :: max_luby_steps
      PetscErrorCode :: ierr
      PetscReal, dimension(:), allocatable :: diag_dom_ratio, diag_dom_ratio_measure
      integer, dimension(:), allocatable :: cf_markers_local_aff
      PetscInt, dimension(:), pointer :: is_pointer
      PetscReal :: max_dd_ratio_local, max_dd_ratio_achieved
      real(c_double) :: swap_dom_val
      integer, dimension(1000) :: dom_bins
      MPIU_Comm :: MPI_COMM_MATRIX
      logical :: trigger_dd_ratio_compute

      ! ~~~~~~  

      ! Get the communicator
      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)    

      ! The indices are the numbering in the local fine row set
      call ISGetIndices(is_fine, is_pointer, ierr)  
      call ISGetLocalSize(is_fine, fine_size, ierr) 

      trigger_dd_ratio_compute = max_dd_ratio > 0

      ! Trigger path requires Aff_transpose and pre-generated random numbers
      if (trigger_dd_ratio_compute) then
         if (.NOT. present(Aff_transpose) .OR. .NOT. present(diag_dom_ratio_random)) then
            print *, "ddc_cpu missing Aff_transpose/randoms for trigger path"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
         end if
      end if
      
      ! Do a fixed alpha_diag
      if (fraction_swap < 0) then
         ! We have to look through all the local rows
         search_size = fine_size

      ! Or pick alpha_diag based on the worst % of rows
      else
         ! Only need to go through the biggest % of indices
         frac_size = int(dble(fine_size) * fraction_swap)

         ! If we are trying to hit a given max_dd_ratio, then we need to continue coarsening, even
         ! if we only change one dof at a time, otherwise we could get stuck         
         if (trigger_dd_ratio_compute) then
            search_size = max(one, frac_size)
         ! If we're not trying to hit a given max_dd_ratio, then if fraction_swap is small
         ! we allow it to just not swap anything if the number of local rows is small
         ! This stops many lower levels in parallel where we are only changing one dof at a time            
         else
            search_size = frac_size
         end if
      end if
      
      call MatGetOwnershipRange(input_mat, input_row_start, input_row_end_plus_one, ierr)                                    

      ! ~~~~~~~~~~~~~
      ! Compute diagonal dominance ratio
      ! ~~~~~~~~~~~~~
      call MatDiagDomRatio_cpu(input_mat, is_fine, cf_markers_local, diag_dom_ratio)
      local_rows = fine_size
      dom_bins = 0

      ! ~~~~~~~~
      ! Get the maximum diagonal dominance ratio
      ! ~~~~~~~~
      if (trigger_dd_ratio_compute) then
         if (local_rows == 0) then
            max_dd_ratio_local = 0
         else
            max_dd_ratio_local = maxval(diag_dom_ratio)
         end if
         call MPI_Allreduce(max_dd_ratio_local, max_dd_ratio_achieved, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_MATRIX, errorcode)
         ! If we have hit the required diagonal dominance ratio, then return without swapping any F points
         if (max_dd_ratio_achieved < max_dd_ratio) then
            max_dd_ratio = max_dd_ratio_achieved
            deallocate(diag_dom_ratio)
            call ISRestoreIndices(is_fine, is_pointer, ierr)

            ! Return if we are < max_dd_ratio
            return
         end if

         ! ~~~~~~~~
         ! If we haven't hit the required diagonal dominance ratio, 
         ! then we need to swap some F points to C points, and we will do that with a 
         ! PMIS style algorithm
         ! This lets us swap many points at once, without just picking every point that 
         ! is above the max ratio and swapping all of them - this would coarsen faster than 
         ! necessary as the removal of any one F point changes the diag dom of any connected 
         ! F points
         ! We also don't want to do it one F point at a time as that would be slow
         ! Hence the independent set is the best of both worlds and very parallel
         !
         ! We go over all existing F points and compute an independent set 
         ! in Aff + Aff^T with a measure given by the diagonal dominance ratio
         ! This will build an independent set of the biggest diagonal dominance ratio
         ! We then swap all of those to C points and then the outer loop outside 
         ! this routine can recompute the diagonal dominace ratio and decide if we 
         ! want to do this again
         ! If there are F points > max_ratio but with neighbours all < max_ratio this 
         !    point will be swapped 
         ! If there are F points > max_ratio and with some neighbours > max_ratio then
         !    only one of those points in the neighbourhood will be swapped, namely the one
         !    with the worst diagonal dominance ratio (this is a heuristic). 
         !    MacLachlan & Saad (2007) page 2120 for example multiply the diagonal dominance ratio
         !    by the 1 on the number of neighbours in S - this would prioritise swapping bad entries
         !    with many F-F connections (ie keeping Aff sparse)
         !    The ratio of all neighbouring
         !    rows will change and may be below the max_ratio after the swap
         !    That will be picked up in the next outer loop. 
         ! ~~~~~~~~
         allocate(diag_dom_ratio_measure(local_rows))

         ! Use the random numbers passed in from the wrapper
         ! so that CPU and Kokkos use the same randoms for PMIS tie-breaking
         diag_dom_ratio_measure = diag_dom_ratio_random(1:local_rows)

         ! ~~~~~~~~
         ! pmisr_existing_measure_cf_markers tags the points with the smallest
         ! measure as F points
         ! So if we feed in a measure that is like 10 - diag_dom_ratio, it will
         ! pick the points with the biggest diagonal dominace ratio
         ! If a point is already below the requested ratio, we set it to be 
         ! PETSC_MAX_REAL so it will never be picked 
         ! ~~~~~~~~

         ! Now we take the existing random number and scale it down
         ! to break ties but not change the diagonal dominance very much
         ! PMISR sets the smallest measure as F points (which is what 
         ! we're going to use to denote points that need to swap in the loop below)
         ! We feed in only F points and a zero cf_markers_local_aff and then 
         ! after the PMISR we take any points tagged as "F" from that result 
         ! and swap them.
         ! The reason we feed in something like 10 - diag_dominance_ratio is not only 
         ! as we want it to pick the biggest diag dominance ratio but also
         ! we have to ensure abs(measure) .ge. 1 
         ! as the PMISR has a step where it sets anything with measure < 1 as F directly
         ! given PMISR is normally called with the measure being the number of strong neighbours
         diag_dom_ratio_measure = max(10d0, max_dd_ratio_achieved*2d0) - (diag_dom_ratio - diag_dom_ratio_measure/1d10)

         allocate(cf_markers_local_aff(local_rows))
         cf_markers_local_aff = 0         

         ! And then any points with diagonal dominance ratio already below
         ! the minimum, we set the measure to PETSC_MAX_REAL and assign them as "C" already
         ! so they won't be swapped
         do ifree = 1, local_rows
            ! Check against the diag_dom_ratio that we haven't modified 
            if (diag_dom_ratio(ifree) < max_dd_ratio) then
               diag_dom_ratio_measure(ifree) = PETSC_MAX_REAL
               cf_markers_local_aff(ifree) = C_POINT
            end if
         end do

         ! Call PMISR with as many steps as necessary
         max_luby_steps = -1
         call pmisr_existing_measure_cf_markers(Aff_transpose, max_luby_steps, .FALSE., &
                  diag_dom_ratio_measure, cf_markers_local_aff)

         ! Let's go and swap the badly diagonally dominant rows to F points
         do ifree = 1, local_rows

            ! The pmisr_existing_measure_cf_markers marked the points we want to swap as F
            if (cf_markers_local_aff(ifree) == F_POINT) then
               ! This is the actual numbering in A, rather than Aff
               ! Careful here to minus away the row_start of A, not Aff
               ! as cf_markers_local is as big as A
               idx = is_pointer(ifree) - input_row_start + 1

               ! Swap by multiplying by -1
               cf_markers_local(idx) = cf_markers_local(idx) * (-1)
            end if
         end do   
         
         deallocate(cf_markers_local_aff, diag_dom_ratio_measure, diag_dom_ratio)
         call ISRestoreIndices(is_fine, is_pointer, ierr)

         ! Return as we're done
         return
      end if

      ! ~~~~~~~~~~~~~
      ! If we got here then the user doesn't want us to hit a given diagonal dominance ratio
      ! So we just swap a fixed fraction of the worst F points to C
      ! ~~~~~~~~~~~~~

      ! If we have local points to swap
      if (search_size > 0) then     

         ! If we reach here then we want to swap some local F points to C points

         do ifree = 1, local_rows

            ! Bin the entries between 0 and 1
            ! The top bin has entries greater than 0.9 (including greater than 1)
            bin = min(floor(diag_dom_ratio(ifree) * size(dom_bins)) + 1, size(dom_bins))
            ! If the diagonal dominance ratio is really large the expression above will overflow
            ! the int to negative, so we just stick that in the top bin            
            if (bin < 0) then
               bin = size(dom_bins)
            end if
            dom_bins(bin) = dom_bins(bin) + 1

         end do      
      
         ! Do a fixed alpha_diag
         if (fraction_swap< 0) then    
            swap_dom_val = -fraction_swap

         ! Otherwise swap everything bigger than a fixed fraction
         else

            ! In order to reduce the size of the sort required, we have binned the entries into 1000 bins
            ! Let's count backwards from the biggest entries to find which bin we know the nth_element is in
            ! and then we only include those bins and higher into the sort
            bin_sum = 0
            do bin_boundary = size(dom_bins), 1, -1
               bin_sum = bin_sum + dom_bins(bin_boundary)
               if (bin_sum .ge. search_size) exit
            end do
            ! Now bin_boundary holds the bin whose lower boundary is guaranteed to be <= the n_th element

            ! Rather than do any type of sort, just swap everything above that bin boundary
            ! This will give a fraction_swap that is very close to that passed in as long as the 
            ! size of the bins is small
            swap_dom_val = dble(bin_boundary-1)/dble(size(dom_bins))

         end if

         ! Let's go and swap F points to C points
         do ifree = 1, local_rows

            ! If this row only has a single diagonal entry, or is below the threshold we swap, skip
            if (diag_dom_ratio(ifree) == 0 .OR. diag_dom_ratio(ifree) < swap_dom_val) cycle

            ! This is the actual numbering in A, rather than Aff
            ! Careful here to minus away the row_start of A, not Aff, as cf_markers_local is as big as A
            idx = is_pointer(ifree) - input_row_start + 1

            ! Swap by multiplying by -1
            cf_markers_local(idx) = cf_markers_local(idx) * (-1)
         end do
      end if

      deallocate(diag_dom_ratio)
      call ISRestoreIndices(is_fine, is_pointer, ierr)

   end subroutine ddc_cpu      
   
! -------------------------------------------------------------------------------------------------------------------------------

end module ddc_module

