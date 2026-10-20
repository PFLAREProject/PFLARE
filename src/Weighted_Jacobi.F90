module weighted_jacobi

   use petscmat
   use pflare_parameters, only: PFLAREINV_WJACOBI, PFLAREINV_JACOBI, PFLAREINV_L1JACOBI

#include "petsc/finclude/petscmat.h"

   implicit none
   public

   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine calculate_and_build_weighted_jacobi_inverse(matrix, inverse_type, inv_matrix)


      ! Builds an assembled weighted jacobi approximate inverse
      ! inverse_type is one of PFLAREINV_WJACOBI, PFLAREINV_JACOBI or PFLAREINV_L1JACOBI

      ! ~~~~~~
      type(tMat), target, intent(in)                    :: matrix
      integer, intent(in)                               :: inverse_type
      type(tMat), intent(inout)                         :: inv_matrix

      ! Local variables
      PetscErrorCode :: ierr
      type(tMat) :: temp_mat
      type(tVec) :: diag_vec
      PetscReal :: norm_inf, weight
      logical :: reuse_triggered

      ! ~~~~~~    

      ! Let's create a matrix to represent the inverse diagonal
      reuse_triggered = .NOT. PetscObjectIsNull(inv_matrix)       

      if (.NOT. reuse_triggered) then
         call MatCreateVecs(matrix, PETSC_NULL_VEC, diag_vec, ierr)
      else
         call MatDiagonalGetDiagonal(inv_matrix, diag_vec, ierr)
      end if
      call MatGetDiagonal(matrix, diag_vec, ierr)

      ! l1 Jacobi has its own weight on each row
      if (inverse_type == PFLAREINV_L1JACOBI) then

         call calculate_l1_jacobi_inverse_diagonal(matrix, diag_vec)

      ! If weighting the Jacobi
      else if (inverse_type == PFLAREINV_WJACOBI) then

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

         ! This is the weight that hypre's BoomerAMG computes with relax_weight = 0
         ! (see par_amg_setup.c), even in asymmetric problems
         ! 4 / ( 3 * || D^(-1/2) * A * D^(-1/2) ||_inf )
         ! Note the hypre docs for HYPRE_BoomerAMGSetRelaxWt quote 3 / (4 * ||.||)
         ! but the code uses 4 / (3 * ||.||)
         weight = 4.0/(3.0 * norm_inf)

      ! Unweighted
      else

         weight = 1d0

      end if

      if (inverse_type /= PFLAREINV_L1JACOBI) then
         call VecReciprocal(diag_vec, ierr)
         call VecScale(diag_vec, weight, ierr)
      end if

      ! We may be reusing with the same sparsity
      if (.NOT. reuse_triggered) then
         ! The matrix takes ownership of diag_vec
         call MatCreateDiagonal(diag_vec, inv_matrix, ierr)
         call VecDestroy(diag_vec, ierr)
      else
         call MatDiagonalRestoreDiagonal(inv_matrix, diag_vec, ierr)
      end if       
   
   end subroutine calculate_and_build_weighted_jacobi_inverse

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine calculate_l1_jacobi_inverse_diagonal(matrix, diag_vec)

      ! Overwrites diag_vec with the l1 Jacobi inverse diagonal
      ! sign(a_ii) / sum_j |a_ij|, where the sum is over the whole (parallel) row
      ! This is hypre's hypre_ParCSRComputeL1Norms with option 1, which also
      ! flips the sign of the l1 norm on rows with a negative diagonal
      ! Rows with a zero diagonal are set to zero (ie they are never relaxed)

      ! ~~~~~~
      type(tMat), intent(in)                            :: matrix
      type(tVec), intent(inout)                         :: diag_vec

      ! Local variables
      PetscErrorCode :: ierr
      type(tVec) :: abs_diag_vec, row_sum_vec
      MatType :: mat_type

      ! ~~~~~~

      call MatGetDiagonal(matrix, diag_vec, ierr)

      ! MATDIAGONAL has no MatGetRowSumAbs, but there sum_j |a_ij| = |a_ii|
      ! so the l1 Jacobi inverse is just the inverse of the diagonal
      call MatGetType(matrix, mat_type, ierr)
      if (mat_type == MATDIAGONAL) then
         call VecReciprocal(diag_vec, ierr)
         return
      end if
      call VecDuplicate(diag_vec, abs_diag_vec, ierr)
      call VecDuplicate(diag_vec, row_sum_vec, ierr)

      ! sign(a_ii) = a_ii / |a_ii|
      ! VecReciprocal leaves zeros as zero, so a zero diagonal gives a zero sign
      call VecCopy(diag_vec, abs_diag_vec, ierr)
      call VecAbs(abs_diag_vec, ierr)
      call VecReciprocal(abs_diag_vec, ierr)
      call VecPointwiseMult(diag_vec, diag_vec, abs_diag_vec, ierr)

      ! 1 / sum_j |a_ij|
      call MatGetRowSumAbs(matrix, row_sum_vec, ierr)
      call VecReciprocal(row_sum_vec, ierr)
      call VecPointwiseMult(diag_vec, diag_vec, row_sum_vec, ierr)

      call VecDestroy(abs_diag_vec, ierr)
      call VecDestroy(row_sum_vec, ierr)

   end subroutine calculate_l1_jacobi_inverse_diagonal


! -------------------------------------------------------------------------------------------------------------------------------

end module weighted_jacobi

