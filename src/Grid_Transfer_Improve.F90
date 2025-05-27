module grid_transfer_improve

   use petscmat
   use timers
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
   
   subroutine improve_w(W, A_ff, A_fc, A_ff_inv, reuse_mat, reuse_mat_two, its, reuse_sparsity)

      ! Does a richardson iteration to improve the ideal W
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(inout) :: W, A_ff, A_fc, A_ff_inv, reuse_mat, reuse_mat_two
      integer, intent(in) :: its
      logical, intent(in) :: reuse_sparsity

      integer :: i
      PetscErrorCode :: ierr
      ! PetscReal :: residual
      type(tMat) :: temp_mat_residual_copy, temp_mat, temp_mat_axpy, temp_mat_sparsity
      type(tVec) :: left_vec_aff, left_vec_inv_aff
      MatType :: mat_type_aff, mat_type_inv_aff      
      logical :: diag_aff_inv, trigger_delete

      ! ~~~~~~~~~~

      ! Return if nothing to do
      if (its == 0) return 

      ! Check if Aff is diagonal
      call MatGetType(A_ff, mat_type_aff, ierr)
      ! Pull out the diagonals if Aff is diagonal
      if (mat_type_aff == MATDIAGONAL) then
         call MatCreateVecs(W, PETSC_NULL_VEC, left_vec_aff, ierr)
         ! Should be able to call MatDiagonalGetDiagonal but it returns
         ! the wrong vector type with kokkos, even when it is set correctly
         ! when calling matcreatediagonal
         call MatGetDiagonal(A_ff, left_vec_aff, ierr)  
      end if

      ! Check if inv Aff is diagonal
      call MatGetType(A_ff_inv, mat_type_inv_aff, ierr)
      ! Pull out the diagonals
      diag_aff_inv = .FALSE.
      ! We now just always pull out the diagonal of Aff^-1, it works very well
      ! and saves a matmatmult if Aff^-1 is not diagonal
      !if (mat_type_inv_aff == MATDIAGONAL) then
         call MatCreateVecs(W, PETSC_NULL_VEC, left_vec_inv_aff, ierr)
         call MatGetDiagonal(A_ff_inv, left_vec_inv_aff, ierr)    
         diag_aff_inv = .TRUE.
      !end if      

      trigger_delete = .FALSE.

      ! Do the number of iterations requested
      ! Can reuse the same sparsity if doing multiple iterations
      do i = 1, its

         ! Compute the residual - Aff W^n + Afc
         if (mat_type_aff == MATDIAGONAL) then

            temp_mat = reuse_mat
            if (.NOT. PetscObjectIsNull(temp_mat)) then

               ! The residual will have a different sparsity pattern to W
               ! after Aff W^n + Afc occurs in the first iteration
               ! (even if Aff is diagonal due to the A_fc)
               call MatCopy(W, &
                        reuse_mat, &
                        DIFFERENT_NONZERO_PATTERN, ierr)
            else
               call MatDuplicate(W, &
                        MAT_COPY_VALUES, &
                        reuse_mat, ierr)
            end if

            ! Left multiply
            call MatDiagonalScale(reuse_mat, &
                     left_vec_aff, PETSC_NULL_VEC, ierr)  
                     
            temp_mat_axpy = reuse_mat                     

         ! If not matdiagonal
         else
            temp_mat = reuse_mat

            ! Now because kokkos is picky about having the exact
            ! same matrix in the MatMatMult when reusing
            ! We cannot just use reuse_mat everywhere
            ! as MatAXPY rebuilds a new matrix internally and replaces it
            ! Therefore have to keep around an extra copy just 
            ! for reuse in kokkos  
            ! But if we are only doing a max of 1 iteration and not doing external reuse            
            if (PetscObjectIsNull(temp_mat)) then
               call MatMatMult(A_ff, W, MAT_INITIAL_MATRIX, 1d0, &
                     reuse_mat, ierr)
               temp_mat_axpy = reuse_mat

               ! Only need to duplicate if we are doing more than 1 iteration
               if (its > 1 .OR. reuse_sparsity) then
                  call MatDuplicate(reuse_mat, &
                        MAT_COPY_VALUES, &
                        temp_mat_residual_copy, ierr)

                  trigger_delete = .TRUE.
                  temp_mat_axpy = temp_mat_residual_copy
               end if               

            else
               call MatMatMult(A_ff, W, MAT_REUSE_MATRIX, 1d0, &
                     reuse_mat, ierr)      
               if (i == 1) then
                  call MatDuplicate(reuse_mat, &
                        MAT_COPY_VALUES, &
                        temp_mat_residual_copy, ierr)                  
               else
                  call MatCopy(reuse_mat, &
                        temp_mat_residual_copy, &
                        DIFFERENT_NONZERO_PATTERN, ierr)     
               end if    

               trigger_delete = .TRUE.      
               temp_mat_axpy = temp_mat_residual_copy               
            end if
         end if

         ! Afc should have a subset of the sparsity of W if Aff 
         ! has a diagonal, but just to be safe lets 
         ! say its different
         call MatAXPYWrapper(temp_mat_axpy, 1d0, A_fc)

         ! If you want to print the residual
         ! call MatNorm(temp_mat_axpy, NORM_FROBENIUS, residual, ierr)
         ! print *, i, "residual = ", residual

         ! Multiply on the left by the preconditioner
         ! Aff_inv (Aff W^n - Afc)

         ! Special case if Aff inv is matdiagonal
         if (diag_aff_inv) then

            ! Left multiply
            call MatDiagonalScale(temp_mat_axpy, &
                     left_vec_inv_aff, PETSC_NULL_VEC, ierr)      
                     
            temp_mat_sparsity = temp_mat_axpy                     

         ! If not matdiagonal
         else   

            temp_mat = reuse_mat_two
            if (PetscObjectIsNull(temp_mat)) then
               call MatMatMult(A_ff_inv, temp_mat_axpy, MAT_INITIAL_MATRIX, 1d0, &
                     reuse_mat_two, ierr)
            else
               call MatMatMult(A_ff_inv, temp_mat_axpy, MAT_REUSE_MATRIX, 1d0, &
                     reuse_mat_two, ierr)            
            end if
            temp_mat_sparsity = reuse_mat_two
         end if
         
         call timer_start(TIMER_ID_AIR_DROP) 
         ! Compute
         ! W^n+1 = W^n - Aff_inv * (Aff W^n + Afc)
         ! and drop any non-zeros outside of the sparsity pattern of W
         call remove_from_sparse_match(temp_mat_sparsity, W, alpha=-1d0)
         call timer_finish(TIMER_ID_AIR_DROP)

      end do

      ! Destroy the temporaries
      if (trigger_delete) call MatDestroy(temp_mat_residual_copy, ierr)
      call VecDestroy(left_vec_aff, ierr) 
      call VecDestroy(left_vec_inv_aff, ierr) 
         
   end subroutine improve_w   

  !------------------------------------------------------------------------------------------------------------------------
   
   subroutine improve_z(Z, A_ff, A_cf, A_ff_inv, reuse_mat, reuse_mat_two, its, reuse_sparsity)

      ! Does a richardson iteration to improve the ideal Z
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(inout) :: Z, A_ff, A_cf, A_ff_inv, reuse_mat, reuse_mat_two
      integer, intent(in) :: its
      logical, intent(in) :: reuse_sparsity

      integer :: i
      PetscErrorCode :: ierr
      ! PetscReal :: residual
      type(tMat) :: temp_mat_residual_copy, temp_mat, temp_mat_axpy, temp_mat_sparsity
      type(tVec) :: right_vec_aff, right_vec_inv_aff
      MatType :: mat_type_aff, mat_type_inv_aff      
      logical :: diag_aff_inv, trigger_delete

      ! ~~~~~~~~~~

      ! Return if nothing to do
      if (its == 0) return 

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
      diag_aff_inv = .FALSE.
      ! We now just always pull out the diagonal of Aff^-1, it works very well
      ! and saves a matmatmult if Aff^-1 is not diagonal
      !if (mat_type_inv_aff == MATDIAGONAL) then
         call MatCreateVecs(Z, right_vec_inv_aff, PETSC_NULL_VEC, ierr)
         call MatGetDiagonal(A_ff_inv, right_vec_inv_aff, ierr)    
         diag_aff_inv = .TRUE.
      !end if      

      trigger_delete = .FALSE.

      ! Do the number of iterations requested
      ! Can reuse the same sparsity if doing multiple iterations
      do i = 1, its

         ! Compute the residual - Z^n Aff + Acf
         if (mat_type_aff == MATDIAGONAL) then

            temp_mat = reuse_mat            
            if (.NOT. PetscObjectIsNull(temp_mat)) then

               ! The residual will have a different sparsity pattern to Z
               ! after Z * Aff + A_cf occurs in the first iteration
               ! (even if Aff is diagonal due to the A_cf)
               call MatCopy(Z, &
                        reuse_mat, &
                        DIFFERENT_NONZERO_PATTERN, ierr)
            else
               call MatDuplicate(Z, &
                        MAT_COPY_VALUES, &
                        reuse_mat, ierr)
            end if

            ! Right multiply
            call MatDiagonalScale(reuse_mat, &
                     PETSC_NULL_VEC, right_vec_aff, ierr)         
                     
            temp_mat_axpy = reuse_mat

         ! If not matdiagonal
         else
            temp_mat = reuse_mat

            ! Now because kokkos is picky about having the exact
            ! same matrix in the MatMatMult when reusing
            ! We cannot just use reuse_mat everywhere
            ! as MatAXPY rebuilds a new matrix internally and replaces it
            ! Therefore have to keep around an extra copy just 
            ! for reuse in kokkos  
            ! But if we are only doing a max of 1 iteration and not doing external reuse
            if (PetscObjectIsNull(temp_mat)) then
               call MatMatMult(Z, A_ff, MAT_INITIAL_MATRIX, 1d0, &
                     reuse_mat, ierr)
               temp_mat_axpy = reuse_mat

               ! Only need to duplicate if we are doing more than 1 iteration
               if (its > 1 .OR. reuse_sparsity) then
                  call MatDuplicate(reuse_mat, &
                        MAT_COPY_VALUES, &
                        temp_mat_residual_copy, ierr)

                  trigger_delete = .TRUE.
                  temp_mat_axpy = temp_mat_residual_copy
               end if

            else
               call MatMatMult(Z, A_ff, MAT_REUSE_MATRIX, 1d0, &
                     reuse_mat, ierr)      
               if (i == 1) then
                  call MatDuplicate(reuse_mat, &
                        MAT_COPY_VALUES, &
                        temp_mat_residual_copy, ierr)                  
               else
                  call MatCopy(reuse_mat, &
                        temp_mat_residual_copy, &
                        DIFFERENT_NONZERO_PATTERN, ierr)     
               end if  

               trigger_delete = .TRUE.      
               temp_mat_axpy = temp_mat_residual_copy
            end if
         end if

         ! Acf should have a subset of the sparsity of Z if Aff 
         ! has a diagonal, but just to be safe lets 
         ! say its different
         call MatAXPYWrapper(temp_mat_axpy, 1d0, A_cf)

         ! If you want to print the residual
         ! call MatNorm(temp_mat_axpy, NORM_FROBENIUS, residual, ierr)
         ! print *, i, "residual = ", residual

         ! Multiply on the right by the preconditioner
         ! (Z^n Aff - Acf) * Aff_inv

         ! Special case if Aff inv is matdiagonal
         if (diag_aff_inv) then

            ! Right multiply
            call MatDiagonalScale(temp_mat_axpy, &
                     PETSC_NULL_VEC, right_vec_inv_aff, ierr)          

            temp_mat_sparsity = temp_mat_axpy

         ! If not matdiagonal
         else   

            temp_mat = reuse_mat_two
            if (PetscObjectIsNull(temp_mat)) then
               call MatMatMult(temp_mat_axpy, A_ff_inv, MAT_INITIAL_MATRIX, 1d0, &
                     reuse_mat_two, ierr)
            else
               call MatMatMult(temp_mat_axpy, A_ff_inv, MAT_REUSE_MATRIX, 1d0, &
                     reuse_mat_two, ierr)            
            end if
            temp_mat_sparsity = reuse_mat_two
         end if
         
         call timer_start(TIMER_ID_AIR_DROP) 
         ! Compute 
         ! Z^n+1 = Z^n - (Z^n Aff + Acf) * Aff_inv
         ! and drop any non-zeros outside of the sparsity pattern of Z
         call remove_from_sparse_match(temp_mat_sparsity, Z, alpha=-1d0)
         call timer_finish(TIMER_ID_AIR_DROP) 

      end do

      ! Destroy the temporaries
      if (trigger_delete) call MatDestroy(temp_mat_residual_copy, ierr)
      call VecDestroy(right_vec_aff, ierr) 
      call VecDestroy(right_vec_inv_aff, ierr) 
         
   end subroutine improve_z   
   
   !-------------------------------------------------------------------------------------------------------------------------------

end module grid_transfer_improve

