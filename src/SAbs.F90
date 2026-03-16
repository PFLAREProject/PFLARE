module sabs

   use petscmat
   use petsc_helper, only: MatAXPYWrapper, MatSetAllValues, remove_small_from_sparse

#include "petsc/finclude/petscmat.h"

   implicit none
   public   
   
   contains
   
!------------------------------------------------------------------------------------------------------------------------
   
   subroutine generate_sabs(input_mat, strong_threshold, symmetrize, square, output_mat, &
                  allow_drop_diagonal, allow_diag_strength)
      
      ! Generate strength of connection matrix with absolute value 
      ! Output has no diagonal entries
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in)     :: input_mat
      type(tMat), intent(inout)  :: output_mat
      PetscReal, intent(in)           :: strong_threshold
      logical, intent(in)        :: symmetrize, square
      logical, intent(in), optional :: allow_drop_diagonal, allow_diag_strength
      
      PetscInt :: ifree
      PetscInt :: local_rows, local_cols, global_rows, global_cols
      PetscInt :: global_row_start, global_row_end_plus_one
      PetscInt :: global_col_start, global_col_end_plus_one, counter
      integer :: errorcode, comm_size
      PetscErrorCode :: ierr
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      MPIU_Comm :: MPI_COMM_MATRIX
      type(tMat) :: transpose_mat
      type(tIS) :: zero_diags
      PetscInt, dimension(:), pointer :: zero_diags_pointer
      logical :: drop_diag, diag_strength
      
      ! ~~~~~~~~~~

      drop_diag = .TRUE.
      diag_strength = .FALSE.
      if (present(allow_drop_diagonal)) drop_diag = allow_drop_diagonal
      if (present(allow_diag_strength)) diag_strength = allow_diag_strength

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)

      ! Get the local sizes
      call MatGetLocalSize(input_mat, local_rows, local_cols, ierr)
      call MatGetSize(input_mat, global_rows, global_cols, ierr)
      ! This returns the global index of the local portion of the matrix
      call MatGetOwnershipRange(input_mat, global_row_start, global_row_end_plus_one, ierr)  
      call MatGetOwnershipRangeColumn(input_mat, global_col_start, global_col_end_plus_one, ierr)  
      
      ! Drop entries smaller than the strong_threshold, with a relative tolerance measured 
      ! against the biggest abs non-diagonal entry, don't lump and always drop the diagonal
      if (.NOT. diag_strength) then
         call remove_small_from_sparse(input_mat, strong_threshold, output_mat, &
               relative_max_row_tol_int = -1, lump=.FALSE., drop_diagonal_int=-1)
      else
         ! Measure the strength of connection relative to the diagonal entry, 
         ! not the max row value excluding the diagonal
         call remove_small_from_sparse(input_mat, strong_threshold, output_mat, &
               relative_max_row_tol_int = -1, lump=.FALSE., drop_diagonal_int=-1, &
               diag_strength_int = 1)
      end if

      ! Now symmetrize if desired
      if (symmetrize) then

         ! We could just do a symbolic transpose and add the two sets of indices together, 
         ! but its so much simpler to just add the two together - and the symbolic will be the expensive part
         ! anyway
         call MatTranspose(output_mat, MAT_INITIAL_MATRIX, transpose_mat, ierr)
         ! Kokkos + MPI doesn't have a gpu mataxpy yet, so we have a wrapper around our own version
         call MatAXPYWrapper(output_mat, 1d0, transpose_mat)

         ! Don't forget to destroy the explicit transpose
         call MatDestroy(transpose_mat, ierr)

      end if

      ! Square the strength matrix to aggressively coarsen (gives a distance 2 MIS)
      if (square) then

         if (symmetrize) then
            call MatMatMult(output_mat, output_mat, &
                        MAT_INITIAL_MATRIX, 1d0, transpose_mat, ierr)     
         else
            call MatTransposeMatMult(output_mat, output_mat, &
                        MAT_INITIAL_MATRIX, 1d0, transpose_mat, ierr)          
         endif     

         ! Also have to add in the original distance 1 connections to the square
         ! as the dist 1 strength matrix has had the diagonals removed, so the square won't 
         ! have the dist 1 connetions in it
         call MatAXPYWrapper(transpose_mat, 1d0, output_mat)
         call MatDestroy(output_mat, ierr)

         ! Can end up with diagonal entries we have to remove
         ! Let's get the diagonals that are zero or unassigned
         call MatFindZeroDiagonals(transpose_mat, zero_diags, ierr)
         call ISGetIndices(zero_diags, zero_diags_pointer, ierr)
         ! Then let's just set every other row to have a zero diagonal
         ! as we know they're already preallocated
         counter = 1
         do ifree = 1, local_rows

            if (counter .le. size(zero_diags_pointer)) then
               ! Skip over any rows that don't have diagonals or are already zero
               if (zero_diags_pointer(counter) - global_row_start + 1 == ifree) then
                  counter = counter + 1 
                  cycle
               end if
            end if
         
            ! Set the diagonal to 0
            call MatSetValue(transpose_mat, ifree - 1 + global_row_start, ifree - 1 + global_row_start, 0d0, INSERT_VALUES, ierr)
         end do

         call ISRestoreIndices(zero_diags, zero_diags_pointer, ierr)
         
         call MatAssemblyBegin(transpose_mat, MAT_FINAL_ASSEMBLY, ierr)
         call MatAssemblyEnd(transpose_mat, MAT_FINAL_ASSEMBLY, ierr)

         ! Could call MatEliminateZeros in later versions of petsc, but for here
         ! given we know the entries are ==1, we will just create a copy with "small" stuff removed
         ! ie the zero diagonal
         call remove_small_from_sparse(transpose_mat, 1d-100, output_mat, drop_diagonal_int = 1) 
         call MatDestroy(transpose_mat, ierr)

      end if   
      
      ! Reset the entries in the strength matrix back to 1
      if (symmetrize .OR. square) call MatSetAllValues(output_mat, 1d0)

   end subroutine generate_sabs     

! -------------------------------------------------------------------------------------------------------------------------------

end module sabs

