module sorting
   
   use binary_tree   
   use petscsys

#include "petsc/finclude/petscsys.h"

   implicit none

   public

   contains 
   
       
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! Some routines to do with sorting arrays
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------   
   
   ! ------------------------------------------------------------------------------------------------------------------------------- 
    
   ! Do a Knuth shuffle of the numbers 1..max_no
   
   subroutine create_knuth_shuffle(max_no, shuffle)
   
      !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      integer, dimension(:), intent(inout) :: shuffle
      integer, intent(in)                               :: max_no

      integer                                           :: i, j, temp_int
      real(8)                                           :: rand_temp
      !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      
      do i = 1, max_no
         shuffle(i) = i
      end do  
      
      ! Knuth shuffle
      do i = max_no, 1, -1
      
         call RANDOM_NUMBER(rand_temp)
         j = int(rand_temp * dble(i) + 1)
         temp_int = shuffle(j)
         shuffle(j) = shuffle(i)
         shuffle(i) = temp_int         
      end do
   
   end subroutine create_knuth_shuffle  
   
   ! ------------------------------------------------------------------------------------------------------------------------------- 
   
   ! Do a Knuth shuffle of the entries in the array and insert them into the tree
   
   subroutine create_knuth_shuffle_tree_array(input_array, tree)
   
      !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      type(itree), intent(inout)                :: tree
      PetscInt, dimension(:), intent(in)        :: input_array
      
      integer, allocatable, dimension(:)        :: shuffle
      integer                                   :: i
      !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      
      allocate(shuffle(size(input_array)))
      
      call create_knuth_shuffle(size(input_array), shuffle)

      do i = 1, size(input_array)
         call insert(tree, input_array(shuffle(i)))
      end do
      
      deallocate(shuffle)
   
   end subroutine create_knuth_shuffle_tree_array       
   

   ! -------------------------------------------------------------------------------------------------------------------------------
      
   
   ! Binary search of a sorted array, returns -1 if not found
   
   subroutine sorted_binary_search(array1, x, location)
   
      !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      PetscInt, dimension(:), intent(in) :: array1
      PetscInt, intent(in) :: x
      integer, intent(out) :: location
      
      integer :: high, low, mid
      !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      location = -1
      low = 1
      high = size(array1)
      do while (low /= high) 

         mid = (low + high)/2

         if (array1(mid) == x) then
            location = mid
            return

         else if (x > array1(mid)) then
            low = mid + 1
         else
            high = mid - 1
         end if
      end do

      ! It should be low or high now, and if not we haven't found it
      if (array1(low) == x) then
         location = low
         return
      end if

   end subroutine sorted_binary_search     

   ! -------------------------------------------------------------------------------------------------------------------------------
      
   
   ! Intersection of two arrays - Assume they are sorted and only return the matching
   ! indices for both arrays
   
   subroutine intersect_pre_sorted_indices_only(array1, array2, indices_array1, indices_array2, intersect_count)
   
      !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      PetscInt, dimension(:), intent(in) :: array1, array2
      integer, dimension(:), intent(out)  :: indices_array1, indices_array2
      integer, intent(out) :: intersect_count

      integer :: index1, index2
      !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      
      index1 = 1
      index2 = 1
      intersect_count = 0   
      
      ! Now loop through and do the intersection
      do while (index1 .le. size(array1) .AND. index2 .le. size(array2)) 
         if (array1(index1) == array2(index2)) then
            intersect_count = intersect_count +1 
            indices_array1(intersect_count) = index1
            indices_array2(intersect_count) = index2
            index1 = index1 + 1
            index2 = index2 + 1
         elseif (array1(index1) < array2(index2)) then
            index1 = index1 + 1
         else
            index2 = index2 + 1
         end if  
      end do

   end subroutine intersect_pre_sorted_indices_only
   
   ! -------------------------------------------------------------------------------------------------------------------------------
   
   ! Combine and sort the contents of two pre-sorted arrays
   
   subroutine merge_pre_sorted(array1, array2, sorted_array)
   
      !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      PetscInt, dimension(:), intent(in) :: array1, array2
      PetscInt, dimension(:), intent(out)  :: sorted_array

      integer :: index1, index2, counter
      !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      
      index1 = 1
      index2 = 1
      counter = 0   
      
      ! Now loop through
      do while (index1 .le. size(array1) .AND. index2 .le. size(array2)) 
         counter = counter + 1
         if (array1(index1) .le. array2(index2)) then
            sorted_array(counter) = array1(index1)
            index1 = index1 + 1
         else
            sorted_array(counter) = array2(index2)
            index2 = index2 + 1
         end if  
      end do

      ! If we've gotten here we need to check if we hit the end of both arrays
      if (index1 .le. size(array1)) then
         sorted_array(counter+1:size(sorted_array)) = array1(index1:size(array1))
      end if
      if (index2 .le. size(array2)) then
         sorted_array(counter+1:size(sorted_array)) = array2(index2:size(array2))
      end if

   end subroutine merge_pre_sorted   

   !-------------------------------------------------------------------------------------------------------------------------------

end module sorting

