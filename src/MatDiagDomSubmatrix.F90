module matdiagdomsubmatrix

   use petscmat
   use cf_splitting, only: compute_cf_splitting, CF_PMISR_DDC
   use petsc_helper, only: MatCreateSubMatrixWrapper
   use pflare_parameters, only: 

#include "petsc/finclude/petscmat.h"

   implicit none
   public   
   
   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine compute_diag_dom_submatrix(input_mat, max_dd_ratio, output_mat)

      ! Returns a diagonally dominant submatrix taken from input_mat where every row's
      ! diagonal dominance ratio is < max_dd_ratio 
      ! Not guaranteed to be the optimal (ie largest) submatrix but should be close
      ! This works for symmetric and asymmetric input_mat
      ! Works in serial, parallel and kokkos (and hence gpus)
      ! This is just a convenience wrapper around compute_cf_splitting and matcreatesubmatrixwrapper

      ! ~~~~~~
      type(tMat), target, intent(in)      :: input_mat
      PetscReal, intent(in)               :: max_dd_ratio
      type(tMat), intent(inout)           :: output_mat

      PetscErrorCode :: ierr
      type(tIS) :: is_fine, is_coarse
      integer :: ddc_its, max_luby_steps, algorithm, errorcode
      PetscReal :: ddc_fraction
      logical :: symmetric       
 
      ! ~~~~~~  

      if (max_dd_ratio .le. 0d0 .or. max_dd_ratio .ge. 1d0) then
         print *, "max_dd_ratio input to compute_diag_dom_submatrix must be (0.0, 1.0)"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)         
      end if

     ! Ignored as we pass in non-zero max_dd_ratio 
     ddc_its = 1
     ! Ignored as we pass in non-zero max_dd_ratio - but can't be sent in as 0
     ddc_fraction = 0.1
     ! As many steps as needed
     max_luby_steps = -1
     ! PMISR DDC
     algorithm = CF_PMISR_DDC
     ! Assume asymmetric - still works for symmetric
     symmetric = .FALSE.

     ! Calll the PMISR_DDC
     ! We call with max_dd_ratio as the strong_threshold
     call compute_cf_splitting(input_mat, &
           symmetric, &
           max_dd_ratio, max_luby_steps, &
           algorithm, &
           ddc_its, &
           ddc_fraction, &
           max_dd_ratio, &
           is_fine, is_coarse)  

     call ISDestroy(is_coarse, ierr)

     ! The input_mat(is_fine, is_fine) is the diagonally dominant
     ! submatrix - this is the wrapper around the kokkos gpu code
      call MatCreateSubMatrixWrapper(input_mat, &
                  is_fine, is_fine, MAT_INITIAL_MATRIX, &
                  output_mat) 

      call ISDestroy(is_fine, ierr)

   end subroutine compute_diag_dom_submatrix 
   
! -------------------------------------------------------------------------------------------------------------------------------

end module matdiagdomsubmatrix

