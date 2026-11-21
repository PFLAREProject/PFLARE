module ddc_module

   use iso_c_binding
   use petscmat
   use petsc_helper, only: kokkos_debug, MatCreateSubMatrixWrapper, MatGetTypeEffective
   use c_petsc_interfaces, only: copy_cf_markers_d2h, ddc_kokkos
   use pmisr_module, only: pmisr_existing_measure_implicit_transpose, pmisr_random_numbers
   use pflare_parameters, only: C_POINT, F_POINT
   use matdiagdom, only: MatDiagDomRatio

#include "petsc/finclude/petscmat.h"

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

      PetscErrorCode :: ierr
      logical :: trigger_dd_ratio_compute_local, on_device
      PetscInt :: local_rows, local_cols
      PetscReal :: max_dd_ratio_achieved
      integer :: comm_rank_ddc, errorcode_ddc
      PetscReal, dimension(:), allocatable :: diag_dom_ratio
      PetscReal, dimension(:), allocatable, target :: randoms_owned
      PetscReal, dimension(:), pointer :: randoms
      type(c_ptr) :: random_numbers_ptr
      MPIU_Comm :: MPI_COMM_MATRIX

#if defined(PETSC_HAVE_KOKKOS)
      integer(c_long_long) :: A_array
      MatType :: mat_type
      type(c_ptr)  :: cf_markers_local_ptr
      integer :: errorcode
      integer, dimension(:), allocatable :: cf_markers_local_two
#endif
      ! ~~~~~~

      ! Not with an initialiser in the declaration, which would give it an implicit save
      nullify(randoms)
      trigger_dd_ratio_compute_local = max_dd_ratio > 0

      ! If we don't need to swap anything, return
      if (fraction_swap == 0d0 .AND. .NOT. trigger_dd_ratio_compute_local) then
         return
      end if

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)
      call MPI_Comm_rank(MPI_COMM_MATRIX, comm_rank_ddc, errorcode_ddc)

      ! Compute the diagonal dominance ratio - either returned in diag_dom_ratio
      ! or stored in a device copy for kokkos
      ! max_dd_ratio_achieved is always returned and is the max diag dom ratio across
      ! all ranks
      call MatDiagDomRatio(input_mat, is_fine, cf_markers_local, diag_dom_ratio, max_dd_ratio_achieved)

      ! If we have hit the required diagonal dominance ratio, return
      if (trigger_dd_ratio_compute_local .AND. max_dd_ratio_achieved < max_dd_ratio) then
         max_dd_ratio = max_dd_ratio_achieved
         if (allocated(diag_dom_ratio)) deallocate(diag_dom_ratio)
         return
      end if

      ! The random numbers for the PMIS tie-breaking in the trigger path, the same ones
      ! for the CPU and Kokkos (and those of the PMISR, kept between the calls in a PCAIR setup)
      ! We take as many as the local rows, rather than just the F points, as if we are on the
      ! device the is_fine won't be allocated on the host yet
      random_numbers_ptr = c_null_ptr
      if (trigger_dd_ratio_compute_local) then
         call MatGetLocalSize(input_mat, local_rows, local_cols, ierr)
         call pmisr_random_numbers(comm_rank_ddc, local_rows, randoms_owned, randoms)
         random_numbers_ptr = c_loc(randoms)
      end if

      on_device = .FALSE.
#if defined(PETSC_HAVE_KOKKOS)
      call MatGetTypeEffective(input_mat, mat_type, ierr)
      on_device = mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS
#endif

      if (on_device) then
#if defined(PETSC_HAVE_KOKKOS)
         ! The kokkos ddc works on the input matrix and the device cf_markers,
         ! it never needs Aff or the host ISs
         A_array = input_mat%v
         cf_markers_local_ptr = c_loc(cf_markers_local)

         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then
            allocate(cf_markers_local_two(size(cf_markers_local)))
            cf_markers_local_two = cf_markers_local
         end if

         ! Modifies the existing device cf_markers created by the pmisr
         call ddc_kokkos(A_array, fraction_swap, max_dd_ratio, max_dd_ratio_achieved, &
            random_numbers_ptr)

         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then

            ! Kokkos DDC by default now doesn't copy back to the host, as any subsequent ddc calls
            ! use the existing device data
            call copy_cf_markers_d2h(cf_markers_local_ptr)
            ! When debugging the host is_fine is kept up to date
            call ddc_host(input_mat, is_fine, fraction_swap, max_dd_ratio, max_dd_ratio_achieved, &
                  diag_dom_ratio, cf_markers_local_two, randoms)

            if (any(cf_markers_local /= cf_markers_local_two)) then
               print *, "Kokkos and CPU versions of ddc do not match"
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
            end if
            deallocate(cf_markers_local_two)
         end if
#endif
      else
         call ddc_host(input_mat, is_fine, fraction_swap, max_dd_ratio, max_dd_ratio_achieved, &
               diag_dom_ratio, cf_markers_local, randoms)
      end if

      if (allocated(randoms_owned)) deallocate(randoms_owned)
      if (allocated(diag_dom_ratio)) deallocate(diag_dom_ratio)

   end subroutine ddc

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine ddc_host(input_mat, is_fine, fraction_swap, max_dd_ratio, max_dd_ratio_achieved, diag_dom_ratio, &
         cf_markers_local, randoms)

      ! ddc_cpu, extracting Aff first if the trigger path needs it

      ! ~~~~~~
      type(tMat), target, intent(in)      :: input_mat
      type(tIS), intent(in)               :: is_fine
      PetscReal, intent(in)               :: fraction_swap
      PetscReal, intent(in)               :: max_dd_ratio
      PetscReal, intent(in)               :: max_dd_ratio_achieved
      PetscReal, dimension(:), intent(in) :: diag_dom_ratio
      integer, dimension(:), allocatable, intent(inout) :: cf_markers_local
      PetscReal, dimension(:), pointer, intent(in) :: randoms

      type(tMat) :: Aff_ddc
      PetscErrorCode :: ierr

      ! ~~~~~~

      if (max_dd_ratio > 0) then
         call MatCreateSubMatrixWrapper(input_mat, &
               is_fine, is_fine, MAT_INITIAL_MATRIX, &
               Aff_ddc)
         call ddc_cpu(input_mat, is_fine, fraction_swap, max_dd_ratio, max_dd_ratio_achieved, &
               diag_dom_ratio, cf_markers_local, &
               Aff=Aff_ddc, &
               diag_dom_ratio_random=randoms)
         call MatDestroy(Aff_ddc, ierr)
      else
         call ddc_cpu(input_mat, is_fine, fraction_swap, max_dd_ratio, max_dd_ratio_achieved, &
               diag_dom_ratio, cf_markers_local)
      end if

   end subroutine ddc_host

! -------------------------------------------------------------------------------------------------------------------------------

      subroutine ddc_cpu(input_mat, is_fine, fraction_swap, max_dd_ratio, max_dd_ratio_achieved, diag_dom_ratio, &
         cf_markers_local, Aff, diag_dom_ratio_random)

      ! Second pass diagonal dominance cleanup
      ! Flips the F definitions to C based on least diagonally dominant local rows
      ! If fraction_swap = 0 this does nothing
      ! If fraction_swap < 0 it uses abs(fraction_swap) to be a threshold
      !  for swapping C to F based on row-wise diagonal dominance (ie alpha_diag)
      ! If fraction_swap > 0 it uses fraction_swap as the local fraction of worst C points to swap to F
      !  though it won't hit that fraction exactly as we bin the diag dom ratios for speed, it will be close to the fraction
      !
      ! When Aff is present and the trigger path runs, the halo scatters inside
      ! pmisr_existing_measure_implicit_transpose use Aff's own mult PetscSF
      ! (obtained via MatGetMultPetscSF), so nothing needs to be forwarded here.

      ! ~~~~~~
      type(tMat), target, intent(in)      :: input_mat
      type(tIS), intent(in)               :: is_fine
      PetscReal, intent(in)               :: fraction_swap
      PetscReal, intent(in)               :: max_dd_ratio
      PetscReal, intent(in)               :: max_dd_ratio_achieved
      PetscReal, dimension(:), intent(in) :: diag_dom_ratio
      integer, dimension(:), allocatable, intent(inout) :: cf_markers_local
      type(tMat), intent(in), optional    :: Aff
      PetscReal, dimension(:), intent(in), optional :: diag_dom_ratio_random

      ! Local
      PetscInt :: local_rows, one=1
      PetscInt :: ifree
      PetscInt :: input_row_start, input_row_end_plus_one
      PetscInt :: idx, search_size, fine_size, frac_size
      integer :: bin_sum, bin_boundary, bin, errorcode
      integer :: max_luby_steps
      PetscErrorCode :: ierr
      PetscReal, dimension(:), allocatable :: diag_dom_ratio_measure
      integer, dimension(:), allocatable :: cf_markers_local_aff
      PetscInt, dimension(:), pointer :: is_pointer
      PetscReal :: swap_dom_val, scaled_ratio
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

      ! Trigger path requires Aff and pre-generated random numbers
      if (trigger_dd_ratio_compute) then
         if (.NOT. present(Aff) .OR. .NOT. present(diag_dom_ratio_random)) then
            print *, "ddc_cpu missing Aff/randoms for trigger path"
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

      local_rows = fine_size
      dom_bins = 0

      ! ~~~~~~~~
      ! Trigger path: use PMIS-based independent set to swap F points
      ! ~~~~~~~~
      if (trigger_dd_ratio_compute) then
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
         diag_dom_ratio_measure = real(max(10d0, max_dd_ratio_achieved*2d0) - &
            (diag_dom_ratio - diag_dom_ratio_measure/1d10), kind=kind(diag_dom_ratio_measure))

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
         ! Uses the implicit transpose version which takes Aff directly
         ! and handles Aff+Aff^T internally without forming the explicit sum
         max_luby_steps = -1
         call pmisr_existing_measure_implicit_transpose(Aff, &
                  max_luby_steps, .FALSE., &
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
         
         deallocate(cf_markers_local_aff, diag_dom_ratio_measure)
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
            ! The top bin has entries greater than 0.999, along with ratios greater than 1
            ! and negative or NaN ratios
            ! We test in floating point first so we never convert NaN or a huge value to an integer
            scaled_ratio = diag_dom_ratio(ifree) * size(dom_bins)
            if (scaled_ratio >= 0 .AND. scaled_ratio < size(dom_bins) - 1) then
               bin = floor(scaled_ratio) + 1
            else
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
            swap_dom_val = real(bin_boundary-1, kind=kind(swap_dom_val))/real(size(dom_bins), kind=kind(swap_dom_val))

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

      call ISRestoreIndices(is_fine, is_pointer, ierr)

   end subroutine ddc_cpu      
   
! -------------------------------------------------------------------------------------------------------------------------------

end module ddc_module

