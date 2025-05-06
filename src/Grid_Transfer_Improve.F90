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
            
      ! ~~~~~~~~~~

      ! Return if nothing to do
      if (its == 0) return

      ! Copy the sparsity pattern of Z
      call MatDuplicate(Z, MAT_DO_NOT_COPY_VALUES, Z_temp_sparsity, ierr)

      ! Do the number of iterations requested
      do i = 1, its

         ! Compute the residual - Z^n Aff - Acf
         ! Can reuse the same sparsity if doing multiple iterations
         if (i == 1) then
            call MatMatMult(Z, A_ff, MAT_INITIAL_MATRIX, 1d0, &
                  residual_mat, ierr)
         else
            call MatMatMult(Z, A_ff, MAT_REUSE_MATRIX, 1d0, &
                  residual_mat, ierr)            
         end if

         ! Acf should have a subset of the sparsity of Z
         call MatAXPY(residual_mat, -1d0, A_cf, SUBSET_NONZERO_PATTERN, ierr)

         call MatNorm(residual_mat, NORM_FROBENIUS, residual, ierr)
         print *, i, "residual = ", residual

         ! Multiply by the preconditioner
         ! Aff_inv * (Z^n Aff - Acf)

         ! @@@ need the case if aff inv is matdiagonal
         if (i == 1) then
            call MatMatMult(residual_mat, A_ff_inv, MAT_INITIAL_MATRIX, 1d0, &
                  Z_temp_no_sparsity, ierr)
         else
            call MatMatMult(residual_mat, A_ff_inv, MAT_REUSE_MATRIX, 1d0, &
                  Z_temp_no_sparsity, ierr)            
         end if
         
         ! Drop any non-zeros outside of the sparsity pattern of Z
         call remove_from_sparse_match(Z_temp_no_sparsity, Z_temp_sparsity)
         ! Z^n+1 = Z^n + Aff_inv * (Z^n Aff - Acf)
         call MatAXPY(Z, 1d0, Z_temp_sparsity, SAME_NONZERO_PATTERN, ierr)

      end do

      ! Destroy the temporaries
      call MatDestroy(residual_mat, ierr)
      call MatDestroy(Z_temp_no_sparsity, ierr)      
      call MatDestroy(Z_temp_sparsity, ierr)      
         
   end subroutine improve_z   
   
   !-------------------------------------------------------------------------------------------------------------------------------

end module grid_transfer_improve

