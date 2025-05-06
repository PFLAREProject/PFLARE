module air_data_type_routines

   use air_data_type
   use approx_inverse_setup
   use fc_smooth
   
   ! PETSc
   use petscmat
   use petscvec
   use petscis

#include "petsc/finclude/petscmat.h"
#include "petsc/finclude/petscvec.h"
#include "petsc/finclude/petscis.h"

   implicit none
   public
  
   contains    

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine create_air_data(air_data)

      ! Setup the data structures for air and reads options from the 
      ! command line

      ! ~~~~~~
      type(air_multigrid_data), intent(inout)    :: air_data
      ! ~~~~~~    

      air_data%no_levels = -1    

      ! Allocate the AIR specific data structures
      allocate(air_data%IS_fine_index(air_data%options%max_levels))
      allocate(air_data%IS_coarse_index(air_data%options%max_levels)) 

      allocate(air_data%restrictors(air_data%options%max_levels))
      allocate(air_data%prolongators(air_data%options%max_levels))

      allocate(air_data%i_fine_full(air_data%options%max_levels))
      allocate(air_data%i_coarse_full(air_data%options%max_levels))
      allocate(air_data%i_fine_full_full(air_data%options%max_levels))
      allocate(air_data%i_coarse_full_full(air_data%options%max_levels))             

      allocate(air_data%coarse_matrix(air_data%options%max_levels))
      allocate(air_data%A_ff(air_data%options%max_levels))
      allocate(air_data%inv_A_ff(air_data%options%max_levels))
      allocate(air_data%inv_A_ff_poly_data(air_data%options%max_levels))
      allocate(air_data%inv_A_ff_poly_data_dropped(air_data%options%max_levels))
      allocate(air_data%inv_A_cc(air_data%options%max_levels))
      allocate(air_data%inv_A_cc_poly_data(air_data%options%max_levels))
      allocate(air_data%A_fc(air_data%options%max_levels))
      allocate(air_data%A_cf(air_data%options%max_levels))
      allocate(air_data%A_cc(air_data%options%max_levels))         

      allocate(air_data%prolongator_nnzs(air_data%options%max_levels))
      allocate(air_data%restrictor_nnzs(air_data%options%max_levels))
      allocate(air_data%A_ff_nnzs(air_data%options%max_levels))  
      allocate(air_data%A_fc_nnzs(air_data%options%max_levels))       
      allocate(air_data%A_cf_nnzs(air_data%options%max_levels))       
      allocate(air_data%A_cc_nnzs(air_data%options%max_levels))       
      allocate(air_data%inv_A_ff_nnzs(air_data%options%max_levels))          
      allocate(air_data%inv_A_cc_nnzs(air_data%options%max_levels))         
      allocate(air_data%coarse_matrix_nnzs(air_data%options%max_levels))

      allocate(air_data%allocated_matrices_A_ff(air_data%options%max_levels))
      allocate(air_data%allocated_matrices_A_cc(air_data%options%max_levels))
      allocate(air_data%allocated_is(air_data%options%max_levels))
      allocate(air_data%allocated_coarse_matrix(air_data%options%max_levels))   
      
      allocate(air_data%smooth_order_levels(air_data%options%max_levels))      

      ! Temporary vectors
      allocate(air_data%temp_vecs_fine(1)%array(air_data%options%max_levels))
      allocate(air_data%temp_vecs_fine(2)%array(air_data%options%max_levels))
      allocate(air_data%temp_vecs_fine(3)%array(air_data%options%max_levels))
      allocate(air_data%temp_vecs_fine(4)%array(air_data%options%max_levels))
      allocate(air_data%temp_vecs_coarse(1)%array(air_data%options%max_levels))
      allocate(air_data%temp_vecs_coarse(2)%array(air_data%options%max_levels))
      allocate(air_data%temp_vecs_coarse(3)%array(air_data%options%max_levels))
      allocate(air_data%temp_vecs_coarse(4)%array(air_data%options%max_levels))
      allocate(air_data%temp_vecs(1)%array(air_data%options%max_levels))

      ! Reuse 
      allocate(air_data%reuse(air_data%options%max_levels))
      
      ! nnzs counts
      air_data%restrictor_nnzs      = 0
      air_data%prolongator_nnzs     = 0
      air_data%inv_A_ff_nnzs        = 0
      air_data%A_fc_nnzs            = 0
      air_data%A_ff_nnzs            = 0
      air_data%A_cf_nnzs            = 0     
      air_data%A_cc_nnzs            = 0       
      air_data%inv_A_cc_nnzs        = 0  
      air_data%coarse_matrix_nnzs   = 0   
      air_data%allocated_matrices_A_ff = .FALSE.
      air_data%allocated_is = .FALSE.
      air_data%allocated_matrices_A_cc = .FALSE. 
      air_data%allocated_coarse_matrix = .FALSE.
     
   end subroutine create_air_data    

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine reset_air_data(air_data, keep_reuse)

      ! Resets the data structures for air

      ! ~~~~~~
      type(air_multigrid_data), intent(inout) :: air_data
      logical, optional :: keep_reuse

      integer :: our_level
      PetscErrorCode :: ierr
      integer :: i_loc
      logical :: reuse
      type(tMat) :: temp_mat
      type(tIS)  :: temp_is
      ! ~~~~~~    

      reuse = .FALSE.
      if (present(keep_reuse)) reuse = keep_reuse

      ! Use if this data structure is allocated to determine if we setup anything
      if (allocated(air_data%allocated_matrices_A_ff)) then

         ! Loop over the levels
         do our_level = 1, size(air_data%allocated_matrices_A_ff)

            ! If we setup Aff
            if (air_data%allocated_matrices_A_ff(our_level)) then

               if (.NOT. reuse) then
                  call MatDestroy(air_data%A_ff(our_level), ierr)
                  call MatDestroy(air_data%A_fc(our_level), ierr)
                  call MatDestroy(air_data%A_cf(our_level), ierr)                  
                  call MatDestroy(air_data%prolongators(our_level), ierr)
                  if (.NOT. air_data%options%symmetric) then
                     call MatDestroy(air_data%restrictors(our_level), ierr)
                  end if                  

                  call destroy_VecISCopyLocalWrapper(air_data, our_level)

                  air_data%allocated_matrices_A_ff(our_level) = .FALSE.
                  call reset_inverse_mat(air_data%inv_A_ff(our_level))
                  if (associated(air_data%inv_A_ff_poly_data(our_level)%coefficients)) then
                     deallocate(air_data%inv_A_ff_poly_data(our_level)%coefficients)
                     air_data%inv_A_ff_poly_data(our_level)%coefficients => null()
                  end if         
                  if (associated(air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients)) then
                     deallocate(air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients)
                     air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients => null()
                  end if

                  call VecDestroy(air_data%temp_vecs(1)%array(our_level), ierr)
                  call VecDestroy(air_data%temp_vecs_fine(1)%array(our_level), ierr)
                  call VecDestroy(air_data%temp_vecs_fine(2)%array(our_level), ierr)
                  call VecDestroy(air_data%temp_vecs_fine(3)%array(our_level), ierr)
                  call VecDestroy(air_data%temp_vecs_fine(4)%array(our_level), ierr)             
                  call VecDestroy(air_data%temp_vecs_coarse(1)%array(our_level), ierr)
                  if (air_data%options%any_c_smooths .AND. &
                        .NOT. air_data%options%full_smoothing_up_and_down) then               
   
                     call VecDestroy(air_data%temp_vecs_coarse(2)%array(our_level), ierr)
                     call VecDestroy(air_data%temp_vecs_coarse(3)%array(our_level), ierr)
                     call VecDestroy(air_data%temp_vecs_coarse(4)%array(our_level), ierr) 
                  end if                   
               end if                                       
            end if  

            if (air_data%allocated_is(our_level)) then
               if (.NOT. reuse) then
                  call ISDestroy(air_data%IS_fine_index(our_level), ierr)
                  call ISDestroy(air_data%IS_coarse_index(our_level), ierr)
                  air_data%allocated_is(our_level) = .FALSE.
               end if
            end if            
            
            ! Did we do C point smoothing?
            if (air_data%allocated_matrices_A_cc(our_level)) then
               if (.NOT. reuse) then
                  call MatDestroy(air_data%A_cc(our_level), ierr)
                  call reset_inverse_mat(air_data%inv_A_cc(our_level))
                  if (associated(air_data%inv_A_cc_poly_data(our_level)%coefficients)) then
                     deallocate(air_data%inv_A_cc_poly_data(our_level)%coefficients)
                     air_data%inv_A_cc_poly_data(our_level)%coefficients => null()
                  end if
                  air_data%allocated_matrices_A_cc(our_level) = .FALSE.
               end if
            end if            
            ! Did we create a coarse grid on this level
            if (air_data%allocated_coarse_matrix(our_level)) then
               call reset_inverse_mat(air_data%coarse_matrix(our_level))
            end if

            ! Destroy the reuse data if needed
            if (.NOT. reuse) then
               do i_loc = 1, size(air_data%reuse(our_level)%reuse_mat)
                  temp_mat = air_data%reuse(our_level)%reuse_mat(i_loc)
                  if (.NOT. PetscObjectIsNull(temp_mat)) then
                     call MatDestroy(air_data%reuse(our_level)%reuse_mat(i_loc), ierr)                  
                  end if
               end do

               do i_loc = 1, size(air_data%reuse(our_level)%reuse_is)
                  temp_is = air_data%reuse(our_level)%reuse_is(i_loc)
                  if (.NOT. PetscObjectIsNull(temp_is)) then
                     call ISDestroy(air_data%reuse(our_level)%reuse_is(i_loc), ierr)                    
                  end if
               end do
            end if
         end do

         if (air_data%no_levels /= -1) then
            ! We also build some things on the coarse grid that must be destroyed
            call VecDestroy(air_data%temp_vecs_fine(1)%array(air_data%no_levels), ierr)
            ! Coarse grid solver
            if (.NOT. reuse) then
               call reset_inverse_mat(air_data%inv_A_ff(air_data%no_levels))
               if (associated(air_data%inv_coarsest_poly_data%coefficients)) then
                  deallocate(air_data%inv_coarsest_poly_data%coefficients)
                  air_data%inv_coarsest_poly_data%coefficients => null()
               end if         
            end if
            ! If we're not doing full smoothing, we have built a matshell on the top grid
            ! we use in the fc smoothing that needs to be destroyed
            if (.NOT. air_data%options%full_smoothing_up_and_down) then
               call reset_inverse_mat(air_data%coarse_matrix(1))
            end if
         end if
      end if 

      ! Reset data
      air_data%no_levels = -1
      air_data%restrictor_nnzs      = 0
      air_data%prolongator_nnzs     = 0
      air_data%inv_A_ff_nnzs        = 0
      air_data%A_fc_nnzs            = 0
      air_data%A_ff_nnzs            = 0
      air_data%A_cf_nnzs            = 0     
      air_data%A_cc_nnzs            = 0       
      air_data%inv_A_cc_nnzs        = 0  
      air_data%coarse_matrix_nnzs   = 0   
      air_data%allocated_coarse_matrix = .FALSE.   

   end subroutine reset_air_data     

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine destroy_air_data(air_data)

      ! Destroys the data structures for air

      ! ~~~~~~
      type(air_multigrid_data), intent(inout) :: air_data

      integer :: our_level
      ! ~~~~~~    

      call reset_air_data(air_data)

      ! Now set the options back to the default
      air_data%options%print_stats_timings = .FALSE.

      air_data%options%max_levels = 300
      air_data%options%coarse_eq_limit = 6
      air_data%options%auto_truncate_start_level = -1
      air_data%options%auto_truncate_tol = 1e-14
      air_data%options%processor_agglom = .TRUE.
      air_data%options%processor_agglom_ratio = 2
      air_data%options%processor_agglom_factor = 2
      air_data%options%process_eq_limit = 50
      air_data%options%subcomm = .FALSE.

      air_data%options%strong_threshold = 0.5
      air_data%options%ddc_fraction = 0.1
      air_data%options%cf_splitting_type = 0
      air_data%options%max_luby_steps = -1

      air_data%options%smooth_order = 0
      air_data%options%smooth_order(1) = 2
      air_data%options%any_c_smooths = .FALSE.
      air_data%options%matrix_free_polys = .FALSE.
      air_data%options%one_point_classical_prolong = .TRUE.
      air_data%options%full_smoothing_up_and_down = .FALSE.
      air_data%options%symmetric = .FALSE.
      air_data%options%constrain_w = .FALSE.
      air_data%options%constrain_z = .FALSE.  
      air_data%options%improve_z_its = 0
      air_data%options%improve_w_its = 0     

      air_data%options%strong_r_threshold = 0d0

      air_data%options%inverse_type = PFLAREINV_POWER

      air_data%options%z_type = AIR_Z_PRODUCT

      air_data%options%lair_distance = 2

      air_data%options%poly_order = 6
      air_data%options%inverse_sparsity_order = 1

      air_data%options%c_inverse_type = PFLAREINV_POWER
      air_data%options%c_poly_order = 6
      air_data%options%c_inverse_sparsity_order = 1
      
      air_data%options%coarsest_inverse_type = PFLAREINV_POWER
      air_data%options%coarsest_poly_order = 6
      air_data%options%coarsest_inverse_sparsity_order = 1
      air_data%options%coarsest_matrix_free_polys = .FALSE.
      air_data%options%coarsest_subcomm = .FALSE.

      air_data%options%r_drop = 0.01
      air_data%options%a_drop = 0.001
      air_data%options%a_lump = .FALSE.    

      air_data%options%reuse_sparsity = .FALSE.     
      air_data%options%reuse_poly_coeffs = .FALSE.           

      ! Use if this data structure is allocated to determine if we setup anything
      if (allocated(air_data%allocated_matrices_A_ff)) then
         
         ! Deallocate the allocated structures
         deallocate(air_data%IS_fine_index)
         deallocate(air_data%IS_coarse_index) 

         deallocate(air_data%restrictors)
         deallocate(air_data%prolongators)  

         deallocate(air_data%i_fine_full)
         deallocate(air_data%i_coarse_full) 
         deallocate(air_data%i_fine_full_full)
         deallocate(air_data%i_coarse_full_full)                   

         deallocate(air_data%coarse_matrix)
         deallocate(air_data%A_ff)
         deallocate(air_data%inv_A_ff)
         deallocate(air_data%inv_A_ff_poly_data)
         deallocate(air_data%inv_A_ff_poly_data_dropped)         
         deallocate(air_data%inv_A_cc)
         deallocate(air_data%inv_A_cc_poly_data)
         deallocate(air_data%A_fc)
         deallocate(air_data%A_cf)
         deallocate(air_data%A_cc) 

         deallocate(air_data%allocated_matrices_A_ff)
         deallocate(air_data%allocated_matrices_A_cc)      
         deallocate(air_data%allocated_is)
         deallocate(air_data%allocated_coarse_matrix)    
         
         do our_level = 1, size(air_data%smooth_order_levels)
            if (allocated(air_data%smooth_order_levels(our_level)%array)) then
               deallocate(air_data%smooth_order_levels(our_level)%array)
            end if
         end do
         deallocate(air_data%smooth_order_levels)         
    
         deallocate(air_data%temp_vecs(1)%array)
         deallocate(air_data%temp_vecs_fine(1)%array)
         deallocate(air_data%temp_vecs_fine(2)%array)
         deallocate(air_data%temp_vecs_fine(3)%array)
         deallocate(air_data%temp_vecs_fine(4)%array)
         deallocate(air_data%temp_vecs_coarse(1)%array)
         deallocate(air_data%temp_vecs_coarse(2)%array)
         deallocate(air_data%temp_vecs_coarse(3)%array)
         deallocate(air_data%temp_vecs_coarse(4)%array)
         
         deallocate(air_data%reuse)
         
         ! Delete the nnzs
         if (allocated(air_data%restrictor_nnzs)) deallocate(air_data%restrictor_nnzs)          
         if (allocated(air_data%prolongator_nnzs)) deallocate(air_data%prolongator_nnzs)        
         if (allocated(air_data%A_ff_nnzs)) deallocate(air_data%A_ff_nnzs) 
         if (allocated(air_data%A_fc_nnzs)) deallocate(air_data%A_fc_nnzs) 
         if (allocated(air_data%A_cf_nnzs)) deallocate(air_data%A_cf_nnzs) 
         if (allocated(air_data%A_cc_nnzs)) deallocate(air_data%A_cc_nnzs) 
         if (allocated(air_data%inv_A_ff_nnzs)) deallocate(air_data%inv_A_ff_nnzs)
         if (allocated(air_data%inv_A_cc_nnzs)) deallocate(air_data%inv_A_cc_nnzs) 
         if (allocated(air_data%coarse_matrix_nnzs)) deallocate(air_data%coarse_matrix_nnzs)         

      end if 
      
      air_data%no_levels = -1

   end subroutine destroy_air_data      

! -------------------------------------------------------------------------------------------------------------------------------
      
end module air_data_type_routines
