module binary_tree

   use petscsys
   use iso_c_binding

#include "petsc/finclude/petscsys.h"      

   implicit none

  ! Define a binary tree node for integers
  TYPE tree_inode
     PetscInt :: value = 0               
     TYPE (tree_inode), pointer :: left=>null() ! next node
     TYPE (tree_inode), pointer :: right=>null() ! next node
     TYPE (tree_inode), pointer :: parent=>null() ! next node
  END TYPE tree_inode
  
  TYPE itree
     PetscInt :: length=0
     TYPE (tree_inode), pointer :: headnode=>null()
  END TYPE itree
  
   TYPE tree_pointer
      type(itree), pointer :: tree => null()
   END TYPE  
 
  interface insert
     module procedure iinsert_tree
  end interface
  
    interface has_value
     module procedure ihas_value_tree
  end interface
  
  interface remove
     module procedure iremove_tree
  end interface
  
  interface print_tree
     module procedure print_tree
  end interface


  contains

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function ihas_value_tree ( tree, value)

  type ( itree ), intent(in) :: tree
  PetscInt, intent(in)  :: value
  
  type(tree_inode), pointer :: current_node
  logical      :: ihas_value_tree
  
  ihas_value_tree = .FALSE.
 
 !  If the tree is empty
  if ( .not. associated ( tree%headnode ) ) then
    return
  end if
  
  !If the tree has at least one node in it
  current_node => tree%headnode
  
  !Iterate until we find where the value belongs
  !Ignore duplicate entries
  do
  
   if ( value < current_node%value) then

      if ( .not. associated ( current_node%left ) ) then
        return
      else
        current_node => current_node%left
      end if

   elseif ( value > current_node%value) then

      if ( .not. associated ( current_node%right ) ) then
        return
      else
        current_node => current_node%right
      end if
      
   else
      ihas_value_tree = .TRUE.
      return
   end if  
  
  end do
  
  
end function ihas_value_tree

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

!This insert method ignores duplicate values
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

subroutine iinsert_tree ( tree, value )

  type ( itree ), intent(inout) :: tree
  PetscInt, intent(in)  :: value
  
  type(tree_inode), pointer :: node
  type(tree_inode), pointer :: current_node
 
 !  If the tree is empty
  if ( .not. associated ( tree%headnode ) ) then
    allocate(node)
    node%value = value
    tree%headnode => node
    tree%length = tree%length + 1
    return
  end if
  
  !If the tree has at least one node in it
  current_node => tree%headnode
  
  !Iterate until we find where the value belongs
  !Ignore duplicate entries
  do
  
   if ( value < current_node%value) then

      if ( .not. associated ( current_node%left ) ) then
        allocate(node)
        node%value = value
        current_node%left => node       
        node%parent => current_node
        tree%length = tree%length + 1
        return
      else
        current_node => current_node%left
      end if

   elseif ( value > current_node%value) then

      if ( .not. associated ( current_node%right ) ) then

        allocate(node)
        node%value = value
        current_node%right => node       
        node%parent => current_node
        tree%length = tree%length + 1
        return
      else
        current_node => current_node%right
      end if
      
   else
      return
   end if  
  
  end do
  
  
end subroutine iinsert_tree

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

!This insert method ignores duplicate values
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

subroutine iremove_tree ( tree, value )

  type ( itree ), intent(inout) :: tree
  PetscInt, intent(in)  :: value
  
  type(tree_inode), pointer :: current_node
  type(tree_inode), pointer :: set_to => null()
   
 !  If the tree is empty
  if ( .not. associated ( tree%headnode ) ) then
    return
  end if
  
  !If the tree has at least one node in it
  current_node => tree%headnode
  
  !Iterate until we find where the value belongs
  !Ignore duplicate entries
  do
  
   if ( value < current_node%value) then

      if ( .not. associated ( current_node%left ) ) then
        return
      else
        current_node => current_node%left
      end if

   elseif ( value > current_node%value ) then

      if ( .not. associated ( current_node%right ) ) then
        return
      else
        current_node => current_node%right
      end if
      
   ! If we've found the value we want to remove   
   else    
   
      ! If the node has two children
      if ( associated ( current_node%right ) .AND. associated ( current_node%left ) ) then
      
         ! Choose the in-order successor
         call find_min_node(current_node%right, set_to)

         current_node%value = set_to%value
         call remove_node(tree, set_to)
         
      else
         call remove_node(tree, current_node)
      
      end if
      tree%length = tree%length - 1
      return

   end if    
  end do
  
  
end subroutine iremove_tree


!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


subroutine remove_node (tree, node )
  
  type ( itree ), intent(inout) :: tree
  type(tree_inode), pointer, intent(inout) :: node
  type(tree_inode), pointer :: set_to => null()
  
  if (.not. associated(node)) then
      return
  end if
  
   ! If the node has no children
   if ( .not. associated ( node%right ) .AND. .not. associated ( node%left ) ) then
   
      set_to => null()
      
   ! If the node has only a left child
   elseif ( .not. associated ( node%right ) .AND.  associated ( node%left ) ) then
   
      set_to => node%left
   
   ! If the node has only a right child
   elseif (associated ( node%right ) .AND.  .not. associated ( node%left ) ) then
   
      set_to => node%right
      
   ! If both children are present, special case, must be dealt with separately
   ! Shouldn't ever be called
   else
      return    
   end if
  
   ! If we're removing the head node
   if (associated(node, tree%headnode)) then
   
      tree%headnode => set_to
      
      if (associated(tree%headnode)) then
         tree%headnode%parent => null()
      end if

      deallocate(node)
      return
   
   end if  
   
   ! Nullify the parent pointer (ie musn't be the head node)
   if (associated(node%parent%right, node)) then
      node%parent%right => set_to 
   else
      node%parent%left => set_to
   end if
   
   if (associated(set_to)) then
      set_to%parent => node%parent
   end if

   deallocate(node)
   return

end subroutine remove_node


!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! Finds the minimum of all the children of a node
! Returns the input node if it has no left children
subroutine find_min_node ( node, current_node )


  type ( tree_inode ), pointer, intent(in)      :: node
  type ( tree_inode ), pointer, intent(inout)   :: current_node
  
  current_node => node

  do while (associated(current_node%left))
      current_node => current_node%left
  
  end do

end subroutine find_min_node

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! Finds the maximum of all the children of a node
! Returns the input node if it has no right children
subroutine find_max_node ( node, current_node )


  type ( tree_inode ), pointer, intent(in)      :: node
  type ( tree_inode ), pointer, intent(inout)   :: current_node
  
  current_node => node

  do while (associated(current_node%right))
      current_node => current_node%right
  
  end do

end subroutine find_max_node

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


subroutine find_max ( tree, value )

  type ( itree ), intent(inout) :: tree
  PetscInt, intent(out)          :: value
  type ( tree_inode ), pointer  :: node => null()
  
  value = 0
  call find_max_node(tree%headnode, node)
  value = node%value

end subroutine find_max

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This routine will automatically deallocate vector and reallocate it if 
! the vector passed in is not big enough
subroutine itree2vector ( tree, vector, ascending )

  type ( itree ), intent(in)                            :: tree
  PetscInt, dimension(:), intent(inout)                 :: vector
  logical, intent(in), optional                         :: ascending
  logical                                               :: asc
  PetscInt                                              :: current_loc
  
  current_loc = 1
  
  if (tree%length == 0) then
      return
  end if
  
  asc = .TRUE.
  
  if (present(ascending)) then
   asc = ascending
  end if
  
   if (asc) then
      call morris_ascending (tree, vector)
   else
      ! should prob bother writing a morris for the descending, but I don't think we ever use the descending
      ! did actually see a stack overflow on the ascending
      call recursive_assign_descending ( tree%headnode, vector, current_loc )
   end if
  
end subroutine itree2vector

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


recursive subroutine recursive_assign_descending ( node, vector, current_loc )

  type ( tree_inode ), pointer, intent(in)        :: node
  PetscInt, dimension(:), intent(inout)           :: vector
  PetscInt, intent(inout)                         :: current_loc

  if ( associated ( node ) ) then

    call recursive_assign_descending ( node%right, vector, current_loc )
    vector(current_loc) = node%value
    current_loc = current_loc + 1
    call recursive_assign_descending ( node%left, vector, current_loc )

  end if
 
  return
end subroutine recursive_assign_descending

! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

subroutine morris_ascending (tree, vector)

   type (itree), target, intent(in)              :: tree
   PetscInt, dimension(:), intent(inout)         :: vector

   type(tree_inode), pointer :: current, pre
   PetscInt :: counter

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   current => tree%headnode
   counter = 1

   do while (associated(current))

      if (.NOT. associated(current%left)) then
         vector(counter) = current%value
         counter = counter + 1
         current => current%right
      else
         pre => current%left
         ! The second associated is a test if they are equal
         do while (associated(pre%right) .AND. .NOT. associated(pre%right, current))
            pre => pre%right
         end do

         if (.NOT. associated(pre%right)) then
            pre%right => current
            current => current%left
         else
            pre%right => null()
            vector(counter) = current%value
            counter = counter + 1            
            current => current%right
         end if
      end if
   end do

end subroutine morris_ascending


!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

recursive subroutine recursive_assign_ascending ( node, vector, current_loc )

  type ( tree_inode ), pointer, intent(in)        :: node
  PetscInt, dimension(:), intent(inout)           :: vector
  PetscInt, intent(inout)                         :: current_loc

  if ( associated ( node ) ) then

    call recursive_assign_ascending ( node%left, vector, current_loc )
    vector(current_loc) = node%value
    current_loc = current_loc + 1
    call recursive_assign_ascending ( node%right, vector, current_loc )

  end if
 
  return
end subroutine recursive_assign_ascending

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

subroutine print_tree ( tree, ascending )

  type ( itree ), intent(in)    :: tree
  logical, intent(in), optional :: ascending
  logical                       :: asc
  
  asc = .FALSE.
  
  if (present(ascending)) then
   asc = ascending
  end if
  
   if (asc) then
      call recursive_print_ascending ( tree%headnode )
   else
      call recursive_print_descending ( tree%headnode )
   end if

end subroutine print_tree


!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

recursive subroutine recursive_print_descending ( node )

  type ( tree_inode ), pointer, intent(in) :: node

  if ( associated ( node ) ) then

    call recursive_print_descending ( node%right )
    print *, node%value
    call recursive_print_descending ( node%left )

  end if
 
  return
end subroutine recursive_print_descending

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

recursive subroutine recursive_print_ascending ( node )

  type ( tree_inode ), pointer, intent(in) :: node

  if ( associated ( node ) ) then

    call recursive_print_ascending ( node%left )
    print *, node%value
    call recursive_print_ascending ( node%right )

  end if
 
  return
end subroutine recursive_print_ascending

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


subroutine flush_tree ( tree )

  type ( itree ), intent(inout) :: tree
 
  if (tree%length == 0) return
  call recursive_flush ( tree%headnode )
  tree%length = 0

end subroutine flush_tree

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

recursive subroutine recursive_flush ( node )

  type ( tree_inode ), pointer, intent(inout) :: node

  if ( associated ( node ) ) then

    call recursive_flush ( node%left )
    call recursive_flush ( node%right )
    deallocate(node)

  end if
 
  return
end subroutine recursive_flush


!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

end module binary_tree
