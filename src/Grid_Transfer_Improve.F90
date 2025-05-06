module grid_transfer_improve

   use petscmat
   use petsc_helper

#include "petsc/finclude/petscmat.h"
#include "petscconf.h"
                
   implicit none

   public

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! Grid transfer improvement routines
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------      

   contains  

  !------------------------------------------------------------------------------------------------------------------------
   
   subroutine improve_z(Z, A_ff, A_cf, A_ff_inv, its)

      ! Does a richardson iteration to improve the ideal Z
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(inout) :: Z, A_ff, A_cf, A_ff_inv
      PetscInt, intent(in) :: its

      PetscInt :: i
      PetscErrorCode :: ierr
      PetscReal :: residual
      type(tMat) :: residual_mat, Z_temp_no_sparsity, Z_temp_sparsity
      type(tMat) :: temp_mat
      type(tVec) :: right_vec_aff, right_vec_inv_aff
      MatType :: mat_type_aff, mat_type_inv_aff            
      ! ~~~~~~~~~~

      ! Return if nothing to do
      if (its == 0) return

      ! Copy the sparsity pattern of Z
      call MatDuplicate(Z, MAT_DO_NOT_COPY_VALUES, Z_temp_sparsity, ierr)

      ! Check if Aff is diagonal
      call MatGetType(A_ff, mat_type_aff, ierr)
      ! Pull out the diagonals if Aff is diagonal
      if (mat_type_aff == MATDIAGONAL) then
         call MatCreateVecs(Z, right_vec_aff, PETSC_NULL_VEC, ierr)
         ! Should be able to call MatDiagonalGetDiagonal but it returns
         ! the wrong vector type with kokkos, even when it is set correctly
         ! when calling matcreatediagonal
         call MatGetDiagonal(A_ff, right_vec_aff, ierr)  
      end if

      ! Check if inv Aff is diagonal
      call MatGetType(A_ff_inv, mat_type_inv_aff, ierr)
      ! Pull out the diagonals
      if (mat_type_inv_aff == MATDIAGONAL) then
         call MatCreateVecs(Z, right_vec_inv_aff, PETSC_NULL_VEC, ierr)
         call MatGetDiagonal(A_ff_inv, right_vec_inv_aff, ierr)    
      end if      

      ! Do the number of iterations requested
      ! Can reuse the same sparsity if doing multiple iterations
      do i = 1, its

         ! Compute the residual - Z^n Aff + Acf
         temp_mat = Z_temp_no_sparsity
         if (mat_type_aff == MATDIAGONAL) then

            if (.NOT. PetscObjectIsNull(temp_mat)) then
               call MatCopy(Z, &
                        residual_mat, &
                        SAME_NONZERO_PATTERN, ierr)
            else
               call MatDuplicate(Z, &
                        MAT_COPY_VALUES, &
                        residual_mat, ierr)          
            end if

            ! Right multiply
            call MatDiagonalScale(residual_mat, &
                     PETSC_NULL_VEC, right_vec_aff, ierr)          

         ! If not matdiagonal
         else

            if (i == 1) then
               call MatMatMult(Z, A_ff, MAT_INITIAL_MATRIX, 1d0, &
                     residual_mat, ierr)
            else
               call MatMatMult(Z, A_ff, MAT_REUSE_MATRIX, 1d0, &
                     residual_mat, ierr)            
            end if
         end if

         ! Acf should have a subset of the sparsity of Z
         call MatAXPY(residual_mat, 1d0, A_cf, SUBSET_NONZERO_PATTERN, ierr)

         ! If you want to output the residual
         call MatNorm(residual_mat, NORM_FROBENIUS, residual, ierr)
         print *, i, "residual = ", residual

         ! Multiply on the right by the preconditioner
         ! (Z^n Aff - Acf) * Aff_inv

         ! Special case if Aff inv is matdiagonal
         temp_mat = Z_temp_no_sparsity
         if (mat_type_inv_aff == MATDIAGONAL) then

            if (.NOT. PetscObjectIsNull(temp_mat)) then
               call MatCopy(residual_mat, &
                        Z_temp_no_sparsity, &
                        SAME_NONZERO_PATTERN, ierr)
            else
               call MatDuplicate(residual_mat, &
                        MAT_COPY_VALUES, &
                        Z_temp_no_sparsity, ierr)          
            end if

            ! Right multiply
            call MatDiagonalScale(Z_temp_no_sparsity, &
                     PETSC_NULL_VEC, right_vec_inv_aff, ierr)          

         ! If not matdiagonal
         else   
            if (i == 1) then
               call MatMatMult(residual_mat, A_ff_inv, MAT_INITIAL_MATRIX, 1d0, &
                     Z_temp_no_sparsity, ierr)
            else
               call MatMatMult(residual_mat, A_ff_inv, MAT_REUSE_MATRIX, 1d0, &
                     Z_temp_no_sparsity, ierr)            
            end if
         end if
         
         ! Drop any non-zeros outside of the sparsity pattern of Z
         call remove_from_sparse_match(Z_temp_no_sparsity, Z_temp_sparsity)
         ! Z^n+1 = Z^n - (Z^n Aff + Acf) * Aff_inv
         call MatAXPY(Z, -1d0, Z_temp_sparsity, SAME_NONZERO_PATTERN, ierr)

      end do

      ! Destroy the temporaries
      call MatDestroy(residual_mat, ierr)
      call MatDestroy(Z_temp_no_sparsity, ierr)      
      call MatDestroy(Z_temp_sparsity, ierr)     
      call VecDestroy(right_vec_aff, ierr) 
      call VecDestroy(right_vec_inv_aff, ierr) 
         
   end subroutine improve_z   
   
   !-------------------------------------------------------------------------------------------------------------------------------

end module grid_transfer_improve

