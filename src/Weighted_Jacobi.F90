module weighted_jacobi

   use petscmat

#include "petsc/finclude/petscmat.h"

   implicit none
   public
   
   PetscEnum, parameter :: PFLAREINV_WJACOBI=7
   PetscEnum, parameter :: PFLAREINV_JACOBI=8   
   
   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine calculate_and_build_weighted_jacobi_inverse(matrix, weighted, inv_matrix)


      ! Builds an assembled weighted jacobi approximate inverse
      ! We could just apply the weighted diagonal without storing it in a petsc mat
      ! but its just convenient to have it like that so we don't have to change any code

      ! ~~~~~~
      type(tMat), target, intent(in)                    :: matrix
      logical, intent(in)                               :: weighted
      type(tMat), intent(inout)                         :: inv_matrix

      ! Local variables
      integer :: comm_size, errorcode
      MPI_Comm :: MPI_COMM_MATRIX 
      PetscInt :: local_rows, local_cols, global_row_start, global_row_end_plus_one
      PetscInt :: global_rows, global_cols, i_loc
      PetscCount :: counter
      PetscErrorCode :: ierr
      PetscInt, allocatable, dimension(:) :: row_indices, col_indices
      type(tMat) :: temp_mat
      type(tVec) :: rhs_copy, diag_vec
      PetscReal :: norm_inf, weight
      PetscInt, parameter :: one=1, zero=0
      MatType:: mat_type
      logical :: reuse_triggered

      ! ~~~~~~    

      ! We might want to call this on a sub communicator
      ! so let's get the comm attached to the matrix and make sure to use that 
      call PetscObjectGetComm(matrix, MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)    
      
      ! Get the local sizes
      call MatGetLocalSize(matrix, local_rows, local_cols, ierr)
      call MatGetSize(matrix, global_rows, global_cols, ierr)
      ! This returns the global index of the local portion of the matrix
      call MatGetOwnershipRange(matrix, global_row_start, global_row_end_plus_one, ierr)        

      call MatCreateVecs(matrix, rhs_copy, diag_vec, ierr)
      call MatGetDiagonal(matrix, diag_vec, ierr)

      ! If weighting the Jacobi
      if (weighted) then

         call MatDuplicate(matrix, MAT_COPY_VALUES, temp_mat, ierr)          

         ! D^(1/2)
         call VecSqrtAbs(diag_vec, ierr)
         ! D^(-1/2)
         call VecReciprocal(diag_vec, ierr)
         ! D^(-1/2) * A * D^(-1/2)
         call MatDiagonalScale(temp_mat, diag_vec, diag_vec, ierr)          
         ! || D^(-1/2) * A * D^(-1/2) ||_inf
         call MatNorm(temp_mat, NORM_INFINITY, norm_inf, ierr)
         call MatDestroy(temp_mat, ierr)

         call MatGetDiagonal(matrix, diag_vec, ierr)

         ! This is the weight that hypre uses, even in assymetric problems
         ! 3 / ( 4 * || D^(-1/2) * A * D^(-1/2) ||_inf )
         weight = 3.0/(4.0 * norm_inf)

      ! Unweighted
      else

         weight = 1d0

      end if

      call VecReciprocal(diag_vec, ierr)
      call VecScale(diag_vec, weight, ierr)      

      ! Let's create a matrix to represent the inverse diagonal
      ! Can't use matdiagonal as we want to do symbolic matmat products
      ! and don't want to have to define how that is done
      reuse_triggered = .NOT. PetscObjectIsNull(inv_matrix) 

      ! We may be reusing with the same sparsity
      if (.NOT. reuse_triggered) then
         call MatCreate(MPI_COMM_MATRIX, inv_matrix, ierr)
         call MatSetSizes(inv_matrix, local_rows, local_cols, &
                           global_rows, global_cols, ierr)
         ! Match the output type
         call MatGetType(matrix, mat_type, ierr)
         call MatSetType(inv_matrix, mat_type, ierr)
         call MatSetUp(inv_matrix, ierr)   
      end if       

      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(inv_matrix, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)
      call MatSetOption(inv_matrix, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)          

      if (.NOT. reuse_triggered) then
         allocate(row_indices(local_rows))
         allocate(col_indices(local_rows))

         ! Set the diagonal
         counter = 1
         do i_loc = global_row_start, global_row_end_plus_one-1
            row_indices(counter) = i_loc
            counter = counter + 1
         end do
         ! MatSetPreallocationCOO could modify the values in either row_indices or col_indices
         col_indices = row_indices
         ! Set the diagonal
         ! Don't need to set the values as we do that directly with MatDiagonalSet
         counter = local_rows
         call MatSetPreallocationCOO(inv_matrix, counter, row_indices, col_indices, ierr)
         deallocate(row_indices, col_indices)         
      end if  

      ! Set the diagonal to our weighted jacobi
      call MatDiagonalSet(inv_matrix, diag_vec, INSERT_VALUES, ierr)
   
      call VecDestroy(diag_vec, ierr)
      call VecDestroy(rhs_copy, ierr)     

   end subroutine calculate_and_build_weighted_jacobi_inverse


! -------------------------------------------------------------------------------------------------------------------------------

end module weighted_jacobi

