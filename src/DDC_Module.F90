module ddc_module

   use iso_c_binding
   use petscmat
   use petsc_helper, only: kokkos_debug, remove_small_from_sparse
   use c_petsc_interfaces, only: copy_cf_markers_d2h, &
         vecscatter_mat_begin_c, vecscatter_mat_end_c, vecscatter_mat_restore_c, &
         allreducesum_petscint_mine, boolscatter_mat_begin_c, boolscatter_mat_end_c, &
         boolscatter_mat_reverse_begin_c, boolscatter_mat_reverse_end_c, ddc_kokkos
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

#if defined(PETSC_HAVE_KOKKOS)                     
      integer(c_long_long) :: A_array
      PetscErrorCode :: ierr
      MatType :: mat_type
      type(c_ptr)  :: cf_markers_local_ptr
      integer :: errorcode
      !integer :: kfree
      integer, dimension(:), allocatable :: cf_markers_local_two
      PetscReal :: max_dd_ratio_cpu, max_dd_ratio_kokkos
#endif 
      ! ~~~~~~  

      ! If we don't need to swap anything, return
      if (fraction_swap == 0d0) then
         return
      end if      

#if defined(PETSC_HAVE_KOKKOS)    

      call MatGetType(input_mat, mat_type, ierr)
      if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) then  

         A_array = input_mat%v  
         cf_markers_local_ptr = c_loc(cf_markers_local)

         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then         
            allocate(cf_markers_local_two(size(cf_markers_local)))
            cf_markers_local_two = cf_markers_local
         end if

         ! Modifies the existing device cf_markers created by the pmisr
         max_dd_ratio_kokkos = max_dd_ratio
         call ddc_kokkos(A_array, fraction_swap, max_dd_ratio_kokkos)

         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then  

            ! Kokkos DDC by default now doesn't copy back to the host, as any subsequent ddc calls 
            ! use the existing device data
            call copy_cf_markers_d2h(cf_markers_local_ptr)       
            max_dd_ratio_cpu = max_dd_ratio     
            call ddc_cpu(input_mat, is_fine, fraction_swap, max_dd_ratio_cpu, cf_markers_local_two)  

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

      else
         call ddc_cpu(input_mat, is_fine, fraction_swap, max_dd_ratio, cf_markers_local)     
      end if
#else
      call ddc_cpu(input_mat, is_fine, fraction_swap, max_dd_ratio, cf_markers_local)
#endif        

   end subroutine ddc
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine ddc_cpu(input_mat, is_fine, fraction_swap, max_dd_ratio, cf_markers_local)

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

      ! Local
      PetscInt :: local_rows, local_cols, one=1, global_rows, global_cols
      PetscInt :: a_global_row_start, a_global_row_end_plus_one, ifree, ncols
      PetscInt :: input_row_start, input_row_end_plus_one
      PetscInt :: jfree, idx, search_size, diag_index, fine_size, frac_size
      integer :: bin_sum, bin_boundary, bin, errorcode, seed_size, comm_size, comm_rank
      integer :: max_luby_steps
      integer, dimension(:), allocatable :: seed
      PetscErrorCode :: ierr
      PetscInt, dimension(:), pointer :: cols => null()
      PetscReal, dimension(:), pointer :: vals => null()
      PetscReal, dimension(:), allocatable :: diag_dom_ratio, diag_dom_ratio_measure
      integer, dimension(:), allocatable :: cf_markers_local_aff
      PetscInt, dimension(:), pointer :: is_pointer
      type(tMat) :: Aff, Aff_transpose
      PetscReal :: diag_val, max_dd_ratio_local, max_dd_ratio_achieved
      real(c_double) :: swap_dom_val
      integer, dimension(1000) :: dom_bins
      MPIU_Comm :: MPI_COMM_MATRIX      
      logical :: trigger_dd_ratio_compute

      ! ~~~~~~  

      ! Get the comm size 
      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)    
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)
      ! Get the comm rank 
      call MPI_Comm_rank(MPI_COMM_MATRIX, comm_rank, errorcode)       

      ! The indices are the numbering in Aff matrix
      call ISGetIndices(is_fine, is_pointer, ierr)  
      call ISGetLocalSize(is_fine, fine_size, ierr) 

      trigger_dd_ratio_compute = max_dd_ratio > 0
      
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
      
      ! ~~~~~~~~~~~~~

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)      
 
      ! Pull out Aff for ease of use
      call MatCreateSubMatrix(input_mat, &
            is_fine, is_fine, MAT_INITIAL_MATRIX, &
            Aff, ierr)     
            
      ! Get the local sizes
      call MatGetLocalSize(Aff, local_rows, local_cols, ierr)
      call MatGetSize(Aff, global_rows, global_cols, ierr)
      call MatGetOwnershipRange(Aff, a_global_row_start, a_global_row_end_plus_one, ierr)
      call MatGetOwnershipRange(input_mat, input_row_start, input_row_end_plus_one, ierr)                                    

      ! ~~~~~~~~~~~~~
      ! Compute diagonal dominance ratio
      ! ~~~~~~~~~~~~~
      allocate(diag_dom_ratio(local_rows))   
      diag_dom_ratio = 0
      dom_bins = 0
      
      ! Sum the rows and find the diagonal entry in each local row
      do ifree = a_global_row_start, a_global_row_end_plus_one-1                  
         call MatGetRow(Aff, ifree, ncols, cols, vals, ierr)

         ! Index of the diagonal
         diag_index = -1
         diag_val = 1.0d0

         do jfree = 1, ncols
            ! Store the diagonal
            if (cols(jfree) == ifree) then
               diag_val = abs(vals(jfree))
               diag_index = jfree
            else
               ! Row sum of off-diagonals
               diag_dom_ratio(ifree - a_global_row_start + 1) = diag_dom_ratio(ifree - a_global_row_start + 1) + abs(vals(jfree))
            end if
         end do

         ! If we don't have a diagonal entry in this row there is no point trying to 
         ! compute a diagonal dominance ratio
         ! We set diag_dom_ratio to zero and that means this row will stay as an F point
         if (diag_index == -1) then
            diag_dom_ratio(ifree - a_global_row_start + 1) = 0.0
            call MatRestoreRow(Aff, ifree, ncols, cols, vals, ierr)    
            cycle
         end if 

         ! If we have non-diagonal entries
         if (diag_dom_ratio(ifree - a_global_row_start + 1) /= 0d0) then
            ! Compute the diagonal dominance ratio
            diag_dom_ratio(ifree - a_global_row_start + 1) = diag_dom_ratio(ifree - a_global_row_start + 1) / diag_val
         end if

         call MatRestoreRow(Aff, ifree, ncols, cols, vals, ierr) 
      end do

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
            call ISRestoreIndices(is_fine, is_pointer, ierr)
            call MatDestroy(Aff, ierr)     

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

         ! Let's fill diag_dom_ratio_measure with random numbers
         call random_seed(size=seed_size)
         allocate(seed(seed_size))
         do ifree = 1, seed_size
            seed(ifree) = comm_rank + 1 + ifree
         end do   
         call random_seed(put=seed) 
         ! Fill the measure with random numbers
         call random_number(diag_dom_ratio_measure)
         deallocate(seed)

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

         ! Generate strength matrix Aff + Aff^T - keep everything but drop the diagonal
         call generate_sabs(Aff, 0d0, .TRUE., .FALSE., Aff_transpose)

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
         
         deallocate(cf_markers_local_aff, diag_dom_ratio_measure)
         call MatDestroy(Aff_transpose, ierr)         

         ! Return as we're done
         return
      end if  

      ! ~~~~~~~~~~~~~
      ! If we got here then the user doesn't want us to hit a given diagonal dominance ratio
      ! So we just swap a fixed fraction of the worst F points to C
      ! ~~~~~~~~~~~~~

      ! Can't put this above because of collective operations in parallel (namely the getsubmatrix)
      ! If we have local points to swap
      if (search_size > 0) then     

         ! If we reach here then we want to swap some local F points to C points

         do ifree = a_global_row_start, a_global_row_end_plus_one-1           

            ! Bin the entries between 0 and 1
            ! The top bin has entries greater than 0.9 (including greater than 1)
            bin = min(floor(diag_dom_ratio(ifree - a_global_row_start + 1) * size(dom_bins)) + 1, size(dom_bins))
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
      call MatDestroy(Aff, ierr)     

   end subroutine ddc_cpu      
   
! -------------------------------------------------------------------------------------------------------------------------------

end module ddc_module

