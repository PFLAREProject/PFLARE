module aggregation

   use petscmat

#include "petsc/finclude/petscmat.h"

   implicit none

#include "petsc_legacy.h"

   public   
   
   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine generate_serial_aggregation(strength_mat, cf_markers, aggregates)

      ! Do an aggregation algorithm - this is currently an exact copy of the aggregation 
      ! algorithm in PyAMG
      ! Only in serial

      ! ~~~~~~
      type(tMat), target, intent(in)      :: strength_mat
      integer, dimension(:), allocatable, intent(inout) :: cf_markers
      PetscInt, dimension(:), allocatable, intent(inout) :: aggregates

      ! Local
      PetscInt :: local_rows, local_cols, global_rows, global_cols
      PetscInt :: a_global_row_start, a_global_row_end_plus_one, ifree, ncols, kfree
      PetscInt :: aggregate, max_neighbour_index, max_neighbour_value, jfree
      integer :: errorcode, comm_size
      PetscErrorCode :: ierr
      MPI_Comm :: MPI_COMM_MATRIX      
      PetscInt, dimension(:), allocatable :: indices
      PetscInt, dimension(:), pointer :: cols
      logical :: mark_neigh

      ! ~~~~~~   

      call PetscObjectGetComm(strength_mat, MPI_COMM_MATRIX, ierr)       
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)      
      ! Serial only!
      if (comm_size /= 1) then
         print *, "Fix me - aggregation algorithm is serial only"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! Get the local sizes
      call MatGetLocalSize(strength_mat, local_rows, local_cols, ierr)
      call MatGetSize(strength_mat, global_rows, global_cols, ierr)      
      call MatGetOwnershipRange(strength_mat, a_global_row_start, a_global_row_end_plus_one, ierr)       

      ! Get the number of connections in S
      allocate(indices(local_rows))
      allocate(aggregates(local_rows))  
      aggregates = 0

      ! If we've passed in a partially completed cf splitting
      if (allocated(cf_markers)) then
         ! we assign them a negative aggregate, which stops any new 
         ! points being added to those "aggregates" in step 2
         do ifree = 1, local_rows
            if (cf_markers(ifree) /= 0) then
               aggregates(ifree) = -1
            end if
         end do         
      else
         allocate(cf_markers(local_rows)) 
         cf_markers = 0 
      end if  
   
      ! Serial ordering to match pyamg in serial
      do ifree = 1, local_rows
         indices(ifree) = ifree
      end do

      aggregate = 1

      ! Loop over all the rows - Step 1 - initial covering
      do ifree = 1, size(indices)

         ! Get S_i - distance 1 neighbours
         call MatGetRow(strength_mat, a_global_row_start + indices(ifree)-1, ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)

         if (ncols == 0) then

            ! No neighbours it stays fine
            cf_markers(indices(ifree)) = -1

         else       

            ! Check if this node or any of the strongly connected neighbours are already in an aggregate
            mark_neigh = any(cf_markers(cols(1:ncols)+1) /= 0 .OR. cf_markers(indices(ifree)) /= 0)      
         
            ! Skip if this node or any strong neighbours are already in an aggregate
            if (.NOT. mark_neigh) then
   
               ! This is the root node
               cf_markers(indices(ifree)) = 1
               aggregates(indices(ifree)) = aggregate
               ! It's neighbours are fine
               do jfree = 1, ncols
                  cf_markers(cols(jfree)+1) = -1
                  aggregates(cols(jfree)+1) = aggregate
               end do         
               ! Advance to a new aggregate
               aggregate = aggregate + 1
            end if            

         end if

         call MatRestoreRow(strength_mat, a_global_row_start + indices(ifree)-1, ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)

      end do

      ! Step 2 - enlarging the sets
      do ifree = 1, size(indices)

         ! Check if this node has been assigned
         if (cf_markers(indices(ifree)) /= 0) cycle

         ! Get S_i - distance 1 neighbours
         call MatGetRow(strength_mat, a_global_row_start + indices(ifree)-1, ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)

         max_neighbour_index = -1
         max_neighbour_value = 0

         ! Loop over strongly connected neighbours
         j_loop: do jfree = 1, ncols

            ! Have we hit a strongly connected neighbour in an aggregate
            ! that wasn't added as part of stage 2
            if (aggregates(cols(jfree)+1) .le. 0) cycle
            ! The pyamg version doesn't test for the actual strength of this connection
            ! It just finds the first strongly connected aggregate neighbour and appends to that
            k_loop: do kfree = 1, ncols
               if (cols(kfree) == cols(jfree)) then
                  max_neighbour_index = jfree
                  exit j_loop
               end if
            end do k_loop
         end do j_loop

         ! If we didn't have a strongly connected neighbour, cycle
         if (max_neighbour_index /= -1) then

            ! Otherwise we have identified the strongest connection this unassigned node has
            ! And so this node becomes part of that aggregate (ie an F point)
            cf_markers(indices(ifree)) = -1
            ! Make this negative to record that it was added as part of stage 2
            aggregates(indices(ifree)) = -aggregates(cols(max_neighbour_index)+1)
         
            ! Advance to a new aggregate
            aggregate = aggregate + 1
         end if

         call MatRestoreRow(strength_mat, a_global_row_start + indices(ifree)-1, ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)
      end do      

      ! Swap the negative ones back to positive
      do ifree = 1, size(indices)
         if (aggregates(ifree) < 0) aggregates(ifree) = aggregates(ifree) * (-1)
      end do

      ! Step 3 - any remnants
      do ifree = 1, size(indices)

         ! Check if this node has been assigned
         if (cf_markers(indices(ifree)) /= 0) cycle

         ! Get S_i - distance 1 neighbours
         call MatGetRow(strength_mat, a_global_row_start + indices(ifree)-1, ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)

         ! This is the root node
         cf_markers(indices(ifree)) = 1
         aggregates(indices(ifree)) = aggregate

         ! Any unassigned strong neighbours are fine and in this aggregate
         do jfree = 1, ncols
            if (cf_markers(cols(jfree)+1) /= 0) cycle            
            cf_markers(cols(jfree)+1) = -1
            aggregates(cols(jfree)+1) = aggregate
         end do         
         ! Advance to a new aggregate
         aggregate = aggregate + 1

         call MatRestoreRow(strength_mat, a_global_row_start + indices(ifree)-1, ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)
      end do

      deallocate(indices)

   end subroutine generate_serial_aggregation   

! -------------------------------------------------------------------------------------------------------------------------------

end module aggregation

