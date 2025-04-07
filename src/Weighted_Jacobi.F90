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

      ! ~~~~~~
      type(tMat), target, intent(in)                    :: matrix
      logical, intent(in)                               :: weighted
      type(tMat), intent(inout)                         :: inv_matrix

      ! Local variables
      PetscErrorCode :: ierr
      type(tMat) :: temp_mat
      type(tVec) :: diag_vec
      PetscReal :: norm_inf, weight
      logical :: reuse_triggered

      ! ~~~~~~    

      call MatCreateVecs(matrix, PETSC_NULL_VEC, diag_vec, ierr)
      call MatGetDiagonal(matrix, diag_vec, ierr)

      ! If weighting the Jacobi
      if (weighted) then

         call MatDuplicate(matrix, MAT_COPY_VALUES, temp_mat, ierr)          

         ! D^(1/2)
         ! Currently happens on the host
         call VecSqrtAbs(diag_vec, ierr)
         ! D^(-1/2)
         call VecReciprocal(diag_vec, ierr)
         ! D^(-1/2) * A * D^(-1/2)
         call MatDiagonalScale(temp_mat, diag_vec, diag_vec, ierr)          
         ! || D^(-1/2) * A * D^(-1/2) ||_inf
         ! Currently happens on the host
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
         ! The matrix takes ownership of diag_vec
         call MatCreateDiagonal(diag_vec, inv_matrix, ierr)
      else
         ! Should be able to call MatDiagonalGetDiagonal but it returns
         ! the wrong vector type with kokkos
         call MatDiagonalSet(inv_matrix, diag_vec, INSERT_VALUES, ierr)
      end if       
      call VecDestroy(diag_vec, ierr)
   
   end subroutine calculate_and_build_weighted_jacobi_inverse


! -------------------------------------------------------------------------------------------------------------------------------

end module weighted_jacobi

