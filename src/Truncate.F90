module truncate

   use petscksp
   use approx_inverse_setup
   use petsc_helper
   use c_petsc_interfaces

#include "petsc/finclude/petscksp.h"

   implicit none
   public

   contains
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine test_truncate(our_level, proc_stride, comm, air_data, auto_truncate)

      ! Compute a coarse grid solver on this level and test if it is good enough

      ! ~~~~~~
      integer, intent(in)                                :: our_level
      PetscInt, intent(in)                               :: proc_stride
      MPI_Comm, intent(in)                               :: comm
      type(air_multigrid_data), target, intent(inout)    :: air_data
      logical, intent(inout)                             :: auto_truncate

      ! Local
      PetscErrorCode      :: ierr
      type(tVec)          :: rand_vec, sol_vec, temp_vec
      PetscInt, parameter :: one=1, zero=0
      PetscReal           :: achieved_rel_tol, norm_b
      PetscInt            :: global_rows, global_cols, local_rows, local_cols     

      ! ~~~~~~     

      ! Get matrix sizes
      call MatGetSize(air_data%coarse_matrix(our_level), global_rows, global_cols, ierr)
      call MatGetLocalSize(air_data%coarse_matrix(our_level), local_rows, local_cols, ierr)       

      ! Set up our coarse inverse data
      call setup_gmres_poly_data(global_rows, &
               air_data%options%coarsest_inverse_type, &
               air_data%options%coarsest_poly_order, &
               air_data%options%coarsest_inverse_sparsity_order, &
               air_data%options%coarsest_subcomm, &
               proc_stride, &
               air_data%inv_coarsest_poly_data)  

      ! Start the approximate inverse we'll use on this level
      call start_approximate_inverse(air_data%coarse_matrix(our_level), &
            air_data%inv_coarsest_poly_data%inverse_type, &
            air_data%inv_coarsest_poly_data%gmres_poly_order, &
            air_data%inv_coarsest_poly_data%buffers, &
            air_data%inv_coarsest_poly_data%coefficients)                       

      call MatCreateVecs(air_data%coarse_matrix(our_level), &
               rand_vec, PETSC_NULL_VEC, ierr)              

      ! This will be a vec of randoms that differ from those used to create the gmres polynomials
      ! We will solve Ax = rand_vec to test how good our coarse solver is
      call vec_random(comm, rand_vec)

      call VecDuplicate(rand_vec, sol_vec, ierr)
      call VecDuplicate(rand_vec, temp_vec, ierr)

      ! Finish our approximate inverse
      call finish_approximate_inverse(air_data%coarse_matrix(our_level), &
            air_data%inv_coarsest_poly_data%inverse_type, &
            air_data%inv_coarsest_poly_data%gmres_poly_order, &
            air_data%inv_coarsest_poly_data%gmres_poly_sparsity_order, &
            air_data%inv_coarsest_poly_data%buffers, &
            air_data%inv_coarsest_poly_data%coefficients, &
            air_data%options%coarsest_matrix_free_polys, &
            air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF), &
            air_data%inv_A_ff(our_level))             

      ! sol_vec = A^-1 * rand_vec
      call MatMult(air_data%inv_A_ff(our_level), rand_vec, sol_vec, ierr)
      ! Now calculate a residual
      ! A * sol_vec
      call MatMult(air_data%coarse_matrix(our_level), sol_vec, temp_vec, ierr)
      ! Now A * sol_vec - rand_vec
      call VecAXPY(temp_vec, -1d0, rand_vec, ierr)
      call VecNorm(temp_vec, NORM_2, achieved_rel_tol, ierr)    
      call VecNorm(rand_vec, NORM_2, norm_b, ierr)    

      ! If it's good enough we can truncate on this level and our coarse solver has been computed
      if (achieved_rel_tol/norm_b < air_data%options%auto_truncate_tol) then
         auto_truncate = .TRUE.

         ! Delete temporary if not reusing
         if (.NOT. air_data%options%reuse_sparsity) then
            call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF), ierr)        
         end if                  

      ! If this isn't good enough, destroy everything we used - no chance for reuse
      else
         call MatDestroy(air_data%inv_A_ff(our_level), ierr)               
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF), ierr)             
      end if      

      call VecDestroy(rand_vec, ierr)
      call VecDestroy(sol_vec, ierr)
      call VecDestroy(temp_vec, ierr)

   end subroutine test_truncate       
   
! -------------------------------------------------------------------------------------------------------------------------------

end module truncate

