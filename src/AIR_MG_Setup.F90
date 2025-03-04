module air_mg_setup

   use petsc
   use constrain_z_or_w
   use cf_splitting
   use matshell_data_type
   use approx_inverse_setup
   use timers
   use air_mg_stats
   use fc_smooth
   use c_petsc_interfaces

#include "petsc/finclude/petsc.h"
      
   implicit none

#include "petsc_legacy.h"   

   public

   contains

   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine get_submatrices_start_poly_coeff_comms(input_mat, our_level, air_data)

      ! Gets the submatrices we need for our multigrid and starts off the comms required 
      ! to compute the approximate inverse of A_ff
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in)                    :: input_mat
      integer, intent(in)                       :: our_level
      type(air_multigrid_data), intent(inout)   :: air_data

      PetscErrorCode :: ierr
      type(tMat) :: smoothing_mat, temp_mat

      ! ~~~~~~~~~~   

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      ! Pull out each of the sub-matrices
      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      call timer_start(TIMER_ID_AIR_EXTRACT)             

      ! Pull out A_ff
      if (air_data%allocated_matrices_A_ff(our_level)) then
         call MatCreateSubMatrix(input_mat, &
               air_data%IS_fine_index(our_level), air_data%IS_fine_index(our_level), MAT_REUSE_MATRIX, &
               air_data%A_ff(our_level), ierr)         
      else
         call MatCreateSubMatrix(input_mat, &
               air_data%IS_fine_index(our_level), air_data%IS_fine_index(our_level), MAT_INITIAL_MATRIX, &
               air_data%A_ff(our_level), ierr)
      end if
               
      call timer_finish(TIMER_ID_AIR_EXTRACT)           

      ! ~~~~~~~~~~~~~
      ! Now to apply a strong R tolerance as lAIR in hypre does, we have to drop entries 
      ! from A_cf and A_ff according to the strong R tolerance on A 
      ! ~~~~~~~~~~~~~
      if (air_data%options%strong_r_threshold == 0d0) then

         ! Copy the original pointer
         air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP) = air_data%A_ff(our_level)
         ! Increase the reference counter
         call PetscObjectReference(air_data%A_ff(our_level), ierr) 
         
      ! If we're dropping
      else

         call timer_start(TIMER_ID_AIR_DROP)  

         ! If we want to reuse, we have to match the original sparsity
         ! which might be different if the matrix has changed (but the sparsity is the same)
         ! so we can't just drop with a drop tolerance    
         temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_A_DROP)
         if (.NOT. PetscMatIsNull(temp_mat)) then

            call remove_from_sparse_match_no_lump(input_mat, air_data%reuse(our_level)%reuse_mat(MAT_A_DROP))     

         else
         
            ! Drop entries smaller than the strong R threshold
            ! but make sure not to drop the diagonal entry!
            call remove_small_from_sparse(input_mat, air_data%options%strong_r_threshold, &
                           air_data%reuse(our_level)%reuse_mat(MAT_A_DROP), &
                           relative_max_row_tol_int= 1, drop_diagonal_int = 0)   
         end if       

         call timer_finish(TIMER_ID_AIR_DROP)                

         call timer_start(TIMER_ID_AIR_EXTRACT)             

         ! Pull out A_ff
         temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP)
         if (.NOT. PetscMatIsNull(temp_mat)) then
            call MatCreateSubMatrix(air_data%reuse(our_level)%reuse_mat(MAT_A_DROP), &
                        air_data%IS_fine_index(our_level), air_data%IS_fine_index(our_level), MAT_REUSE_MATRIX, &
                        air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP), ierr)              
         else
            call MatCreateSubMatrix(air_data%reuse(our_level)%reuse_mat(MAT_A_DROP), &
                        air_data%IS_fine_index(our_level), air_data%IS_fine_index(our_level), MAT_INITIAL_MATRIX, &
                        air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP), ierr)  
         end if
                     
         call timer_finish(TIMER_ID_AIR_EXTRACT)                             

      end if          
      
      ! ~~~~~~~~~~~~~~
      ! Start building approximate inverse - A_ff^-1
      ! ~~~~~~~~~~~~~~       
      call timer_start(TIMER_ID_AIR_INVERSE)    
      
      ! This is for the smoother
      if (.NOT. air_data%options%full_smoothing_up_and_down) then
         smoothing_mat = air_data%A_ff(our_level)
      else
         smoothing_mat = input_mat
      end if

      ! Compute the inverse for smoothing
      ! If we are re-using the polynomial coefficients then we don't have to do this
      temp_mat = air_data%inv_A_ff(our_level)
      if (.NOT. (.NOT. PetscMatIsNull(temp_mat) .AND. &
                  air_data%options%reuse_poly_coeffs)) then
         call start_approximate_inverse(smoothing_mat, &
                  air_data%options%inverse_type, &
                  air_data%inv_A_ff_poly_data(our_level)%gmres_poly_order, &
                  air_data%inv_A_ff_poly_data(our_level)%buffers, &
                  air_data%inv_A_ff_poly_data(our_level)%coefficients)        
      end if

      ! If we are doing AIRG
      ! then we need an A_ff^-1 that we use to build the grid-transfer operators
      ! but if the strong R threshold is zero we just use the one computed above
      if (air_data%options%z_type == AIR_Z_PRODUCT .AND. &
               (air_data%options%strong_r_threshold /= 0d0 .OR. &
                air_data%options%full_smoothing_up_and_down)) then

         temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP)
         if (.NOT. (.NOT. PetscMatIsNull(temp_mat) .AND. &
                     air_data%options%reuse_poly_coeffs)) then
            call start_approximate_inverse(air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP), &
                  air_data%options%inverse_type, &
                  air_data%inv_A_ff_poly_data_dropped(our_level)%gmres_poly_order, &
                  air_data%inv_A_ff_poly_data_dropped(our_level)%buffers, &
                  air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients)          
         end if
      end if

      call timer_finish(TIMER_ID_AIR_INVERSE)
      
      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      ! If we are doing C point smoothing then we need to pull out Acc 
      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
      if (air_data%options%one_c_smooth .AND. .NOT. air_data%options%full_smoothing_up_and_down) then

         if (air_data%allocated_matrices_A_cc(our_level)) then
            call MatCreateSubMatrix(input_mat, &
                  air_data%IS_coarse_index(our_level), air_data%IS_coarse_index(our_level), MAT_REUSE_MATRIX, &
                  air_data%A_cc(our_level), ierr)             
         else
            call MatCreateSubMatrix(input_mat, &
                  air_data%IS_coarse_index(our_level), air_data%IS_coarse_index(our_level), MAT_INITIAL_MATRIX, &
                  air_data%A_cc(our_level), ierr)   
         end if

         call timer_start(TIMER_ID_AIR_INVERSE)    

         ! Compute the inverse for smoothing
         temp_mat = air_data%inv_A_cc(our_level)
         if (.NOT. (.NOT. PetscMatIsNull(temp_mat) &
                     .AND. air_data%options%reuse_poly_coeffs)) then
            call start_approximate_inverse(air_data%A_cc(our_level), &
                  air_data%options%inverse_type, air_data%inv_A_cc_poly_data(our_level)%gmres_poly_order, &
                  air_data%inv_A_cc_poly_data(our_level)%buffers, air_data%inv_A_cc_poly_data(our_level)%coefficients)  
         end if

         call timer_finish(TIMER_ID_AIR_INVERSE)

      end if        
         
      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      ! Pull out the rest of the sub-matrices
      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
      call timer_start(TIMER_ID_AIR_EXTRACT)             
                        
      if (air_data%allocated_matrices_A_ff(our_level)) then
         call MatCreateSubMatrix(input_mat, &
               air_data%IS_fine_index(our_level), air_data%IS_coarse_index(our_level), MAT_REUSE_MATRIX, &
               air_data%A_fc(our_level), ierr)   
      call MatCreateSubMatrix(input_mat, &
               air_data%IS_coarse_index(our_level), air_data%IS_fine_index(our_level), MAT_REUSE_MATRIX, &
               air_data%A_cf(our_level), ierr)                     
      else
         call MatCreateSubMatrix(input_mat, &
               air_data%IS_fine_index(our_level), air_data%IS_coarse_index(our_level), MAT_INITIAL_MATRIX, &
               air_data%A_fc(our_level), ierr) 
         call MatCreateSubMatrix(input_mat, &
               air_data%IS_coarse_index(our_level), air_data%IS_fine_index(our_level), MAT_INITIAL_MATRIX, &
               air_data%A_cf(our_level), ierr)                                                          
      end if

      call timer_finish(TIMER_ID_AIR_EXTRACT)   

      ! ~~~~~~~~~~~~~~
      ! Apply the strong R threshold to A_cf
      ! ~~~~~~~~~~~~~~
      if (air_data%options%strong_r_threshold == 0d0) then

         ! Copy the original pointer
         air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP) = air_data%A_cf(our_level)
         ! Increase the reference counter
         call PetscObjectReference(air_data%A_cf(our_level), ierr)
         air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP) = air_data%A_fc(our_level)
         ! Increase the reference counter
         call PetscObjectReference(air_data%A_fc(our_level), ierr)        
         
      else

         call timer_start(TIMER_ID_AIR_EXTRACT)        
         
         ! Drop the entries from A_cf
         temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP)
         if (.NOT. PetscMatIsNull(temp_mat)) then
            call MatCreateSubMatrix(air_data%reuse(our_level)%reuse_mat(MAT_A_DROP), &
                        air_data%IS_coarse_index(our_level), air_data%IS_fine_index(our_level), MAT_REUSE_MATRIX, &
                        air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), ierr)              
         else

            call MatCreateSubMatrix(air_data%reuse(our_level)%reuse_mat(MAT_A_DROP), &
                        air_data%IS_coarse_index(our_level), air_data%IS_fine_index(our_level), MAT_INITIAL_MATRIX, &
                        air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), ierr)  
         end if

         if (.NOT. air_data%options%one_point_classical_prolong) then
            ! Drop the entries from A_fc
            temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP)
            if (.NOT. PetscMatIsNull(temp_mat)) then
               call MatCreateSubMatrix(air_data%reuse(our_level)%reuse_mat(MAT_A_DROP), &
                        air_data%IS_fine_index(our_level), air_data%IS_coarse_index(our_level), MAT_REUSE_MATRIX, &
                        air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP), ierr)
            else
               call MatCreateSubMatrix(air_data%reuse(our_level)%reuse_mat(MAT_A_DROP), &
                        air_data%IS_fine_index(our_level), air_data%IS_coarse_index(our_level), MAT_INITIAL_MATRIX, &
                        air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP), ierr)  
            end if
         end if                  
         
         call timer_finish(TIMER_ID_AIR_EXTRACT)    
      end if               

      ! Delete temporary if not reusing
      if (.NOT. air_data%options%reuse_sparsity) then
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_A_DROP), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
         air_data%reuse(our_level)%reuse_mat(MAT_A_DROP) = PETSC_NULL_MAT
#endif
      end if
                        
   end subroutine get_submatrices_start_poly_coeff_comms 
   
   
   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine finish_comms_compute_restrict_prolong(A, our_level, air_data, &
                     left_null_vecs, right_null_vecs, &
                     left_null_vecs_c, right_null_vecs_c)

      ! Finishes off the comms required to compute the A_ff inverse
      ! then form Z, W and the full R and P
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(inout)                             :: A
      integer, intent(in)                                   :: our_level
      type(air_multigrid_data), intent(inout)               :: air_data
      type(tVec), dimension(:), intent(inout)               :: left_null_vecs, right_null_vecs
      type(tVec), dimension(:), allocatable, intent(inout)  :: left_null_vecs_c, right_null_vecs_c

      PetscErrorCode :: ierr
      type(tMat) :: minus_mat, sparsity_mat_cf, A_ff_power, inv_dropped_Aff, smoothing_mat
      type(tMat) :: temp_mat
      type(tIS)  :: temp_is
      type(tVec), dimension(:), allocatable   :: left_null_vecs_f, right_null_vecs_f
      integer :: comm_size, errorcode, order, i_loc
      MPI_Comm :: MPI_COMM_MATRIX
      integer(c_long_long) :: A_array, B_array, C_array
      PetscInt :: global_row_start, global_row_end_plus_one
      PetscInt, parameter :: nz_ignore = -1
      logical :: destroy_mat, reuse_grid_transfer
      MatType:: mat_type

      ! ~~~~~~~~~~

      call PetscObjectGetComm(air_data%A_ff(our_level), MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)

      ! ~~~~~~~~~~~
      ! Get some sizes
      ! ~~~~~~~~~~~

      call MatGetOwnershipRange(A, global_row_start, global_row_end_plus_one, ierr)  
      
      ! ~~~~~~~~~~~~~~~~~~~
      ! ~~~~~~~~~~~~~~~~~~~
      ! Calculate one point W if needed
      ! ~~~~~~~~~~~~~~~~~~~
      ! ~~~~~~~~~~~~~~~~~~~        

      ! If we're doing a one-point classical prolongator, we can do this before A_ff^-1 is finished
      if (air_data%options%one_point_classical_prolong .AND. .NOT. air_data%options%symmetric) then

         ! Classical one-point prolongator
         call timer_start(TIMER_ID_AIR_PROLONG)       
         
         ! If we want to reuse, we have to match the original sparsity
         ! which might be different if the matrix has changed (but the sparsity is the same)
         ! so we can't just drop with a drop tolerance
         temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_W_DROP)
         if (.NOT. PetscMatIsNull(temp_mat)) then

            ! The one point classical prolongator will never change if 
            ! we are reusing
            
         ! First time
         else
            call generate_one_point_with_one_entry_from_sparse(air_data%A_fc(our_level), &
                     air_data%reuse(our_level)%reuse_mat(MAT_W_DROP)) 
         end if         
         
         call timer_finish(TIMER_ID_AIR_PROLONG)                

      end if

      ! ~~~~~~~~~~~~~~~~~~~
      ! ~~~~~~~~~~~~~~~~~~~
      ! Finish the calculation of the inverse for the smoother 
      ! and/or the inverse for the grid-transfers (they may be the same)
      ! ~~~~~~~~~~~~~~~~~~~
      ! ~~~~~~~~~~~~~~~~~~~        
      
      ! This is for the smoother
      if (.NOT. air_data%options%full_smoothing_up_and_down) then
         smoothing_mat = air_data%A_ff(our_level)
      else
         smoothing_mat = A
      end if      

      ! Resolve the calculation for the smoother
      ! This may build a matrix-free version of inv_A_ff
      call timer_start(TIMER_ID_AIR_INVERSE)   

      ! Regardless of if we are fc smoothing or smoothing all unknowns we store the inverse
      ! in inv_A_ff
      call finish_approximate_inverse(smoothing_mat, air_data%options%inverse_type, &
            air_data%inv_A_ff_poly_data(our_level)%gmres_poly_order, &
            air_data%inv_A_ff_poly_data(our_level)%gmres_poly_sparsity_order, &
            air_data%inv_A_ff_poly_data(our_level)%buffers, &
            air_data%inv_A_ff_poly_data(our_level)%coefficients, &
            air_data%options%matrix_free_polys, air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF), &
            air_data%inv_A_ff(our_level)) 
            
      ! Delete temporary if not reusing
      if (.NOT. air_data%options%reuse_sparsity) then
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
         air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF) = PETSC_NULL_MAT
#endif
      end if              
      
      destroy_mat = .FALSE.
      ! If we are doing AIRG, then we also need an A_ff^-1 for the grid
      ! transfer operators (may be the same used for the smoother)
      if (air_data%options%z_type == AIR_Z_PRODUCT) then

         ! If we have applied a strong R tolerance or we are not doing fc smoothing
         ! then we have also started an inverse for dropped A_ff, let's finish it
         if (air_data%options%strong_r_threshold /= 0d0 .OR. air_data%options%full_smoothing_up_and_down) then

#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
            inv_dropped_Aff = PETSC_NULL_MAT
#endif
            ! Now we always build an assembled inv_A_ff here as we need it, hence the .false. 
            ! given to matrix_free_polys
            ! If we aren't doing matrix-free smooths, then we keep the assembled inv_A_ff
            call finish_approximate_inverse(air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP), &
                     air_data%options%inverse_type, &
                     air_data%inv_A_ff_poly_data_dropped(our_level)%gmres_poly_order, &
                     air_data%inv_A_ff_poly_data_dropped(our_level)%gmres_poly_sparsity_order, &
                     air_data%inv_A_ff_poly_data_dropped(our_level)%buffers, &
                     air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients, &
                     .FALSE., air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF_DROPPED), &
                     inv_dropped_Aff)
            destroy_mat = .TRUE.

            ! Delete temporary if not reusing
            if (.NOT. air_data%options%reuse_sparsity) then
               call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF_DROPPED), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
               air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF_DROPPED) = PETSC_NULL_MAT
#endif               
            end if             

         ! If we have a strong R tolerance of 0, we can re-use the 
         ! A_ff^-1 we computed for the smoother
         else             
            ! If we requested matrix-free smoothing, then we will have called the finish
            ! and computed the coefficients but inv_A_ff will be a matshell
            ! and hence we want to build an assembled version of A_ff^-1 to use here
            ! We can just re-call finish_approximate_inverse, as the requests will have been resolved 
            ! and hence it will just build an assembled version, which we store in inv_dropped_Aff
            if (air_data%options%matrix_free_polys) then
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)                     
               inv_dropped_Aff = PETSC_NULL_MAT
#endif               
               ! Making sure to give it the non dropped A_ff here
               call finish_approximate_inverse(air_data%A_ff(our_level), &
                        air_data%options%inverse_type, &
                        air_data%inv_A_ff_poly_data(our_level)%gmres_poly_order, &
                        air_data%inv_A_ff_poly_data(our_level)%gmres_poly_sparsity_order, &
                        air_data%inv_A_ff_poly_data(our_level)%buffers, &
                        air_data%inv_A_ff_poly_data(our_level)%coefficients, &
                        .FALSE., air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF_DROPPED), &
                        inv_dropped_Aff)
               destroy_mat = .TRUE.

               ! Delete temporary if not reusing
               if (.NOT. air_data%options%reuse_sparsity) then
                  call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF_DROPPED), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
                  air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF_DROPPED) = PETSC_NULL_MAT
#endif                  
               end if               

            ! Just re-use the already assembled one            
            else
               inv_dropped_Aff = air_data%inv_A_ff(our_level)
            end if
         end if

         ! Copy inv_A_ff and multiply by -1
         call MatConvert(inv_dropped_Aff, MATSAME, MAT_INITIAL_MATRIX, minus_mat, ierr)
         call MatScale(minus_mat, -1d0, ierr)

         if (destroy_mat) then
            call MatDestroy(inv_dropped_Aff, ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
            inv_dropped_Aff = PETSC_NULL_MAT
#endif            
         end if
      end if

      call timer_finish(TIMER_ID_AIR_INVERSE)           
      
      ! Need a routine to calculate W with lAIR
      if (air_data%options%z_type /= AIR_Z_PRODUCT .AND. &
               (.NOT. air_data%options%one_point_classical_prolong .AND. .NOT. air_data%options%symmetric)) then
         print *, "Fix me - calculation of W with lair"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! ~~~~~~~~~~~~~~~~~~~
      ! ~~~~~~~~~~~~~~~~~~~
      ! Pull out the constraints on F and C points
      ! ~~~~~~~~~~~~~~~~~~~
      ! ~~~~~~~~~~~~~~~~~~~    
      if (air_data%options%constrain_w) then
         if (.NOT. allocated(right_null_vecs_c)) allocate(right_null_vecs_c(size(right_null_vecs)))
         if (.NOT. allocated(right_null_vecs_f)) allocate(right_null_vecs_f(size(right_null_vecs)))

         ! Create space for the C and F constraints
         call MatCreateVecs(air_data%A_fc(our_level), right_null_vecs_c(1), right_null_vecs_f(1), ierr)       
         do i_loc = 2, size(right_null_vecs) 
            call VecDuplicate(right_null_vecs_c(1), right_null_vecs_c(i_loc), ierr)
            call VecDuplicate(right_null_vecs_f(1), right_null_vecs_f(i_loc), ierr)
         end do
         ! Pull out the F and C points
         do i_loc = 1, size(right_null_vecs) 
            call VecISCopy(right_null_vecs(i_loc), air_data%IS_fine_index(our_level), &
                     SCATTER_REVERSE, right_null_vecs_f(i_loc), ierr)       
            call VecISCopy(right_null_vecs(i_loc), air_data%IS_coarse_index(our_level), &
                     SCATTER_REVERSE, right_null_vecs_c(i_loc), ierr) 
         end do         
      end if
      if (air_data%options%constrain_z) then
         if (.NOT. allocated(left_null_vecs_c)) allocate(left_null_vecs_c(size(left_null_vecs)))
         if (.NOT. allocated(left_null_vecs_f)) allocate(left_null_vecs_f(size(left_null_vecs)))
         
         ! Create space for the C and F constraints
         call MatCreateVecs(air_data%A_fc(our_level), left_null_vecs_c(1), left_null_vecs_f(1), ierr)       
         do i_loc = 2, size(left_null_vecs) 
            call VecDuplicate(left_null_vecs_c(1), left_null_vecs_c(i_loc), ierr)
            call VecDuplicate(left_null_vecs_f(1), left_null_vecs_f(i_loc), ierr)
         end do
         ! Pull out the F and C points
         do i_loc = 1, size(left_null_vecs) 
            call VecISCopy(left_null_vecs(i_loc), air_data%IS_fine_index(our_level), &
                     SCATTER_REVERSE, left_null_vecs_f(i_loc), ierr)       
            call VecISCopy(left_null_vecs(i_loc), air_data%IS_coarse_index(our_level), &
                     SCATTER_REVERSE, left_null_vecs_c(i_loc), ierr) 
         end do         
      end if      

      ! ~~~~~~~~~~~~~~~~~~~
      ! ~~~~~~~~~~~~~~~~~~~
      ! Calculate W if needed
      ! ~~~~~~~~~~~~~~~~~~~
      ! ~~~~~~~~~~~~~~~~~~~     
      reuse_grid_transfer = air_data%allocated_matrices_A_ff(our_level)   

      ! We do this a little backwards if symmetric, we build R and then compute P^T, then delete R
      ! It's just because we have code to do different version of Z, and I haven't rewritten those 
      ! for W
      if (.NOT. air_data%options%symmetric) then

         ! Calculate the W component of the prolongator
         call timer_start(TIMER_ID_AIR_PROLONG)                

         ! If we want an ideal prolongator
         if (.NOT. air_data%options%one_point_classical_prolong) then

            ! Do the multiplication
            temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_W)
            if (.NOT. PetscMatIsNull(temp_mat)) then
               call MatMatMult(minus_mat, air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP), &
                        MAT_REUSE_MATRIX, 1.58d0, air_data%reuse(our_level)%reuse_mat(MAT_W), ierr)
            else
               call MatMatMult(minus_mat, air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP), &
                        MAT_INITIAL_MATRIX, 1.58d0, air_data%reuse(our_level)%reuse_mat(MAT_W), ierr)  
            end if

            call timer_start(TIMER_ID_AIR_DROP)  

            ! If we want to reuse, we have to match the original sparsity
            ! which might be different if the matrix has changed (but the sparsity is the same)
            ! so we can't just drop with a drop tolerance
            temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_W_DROP)
            if (.NOT. PetscMatIsNull(temp_mat)) then
   
               call remove_from_sparse_match_no_lump(air_data%reuse(our_level)%reuse_mat(MAT_W), &
                        air_data%reuse(our_level)%reuse_mat(MAT_W_DROP))  
               
            ! First time so just drop according to a tolerance 
            else
               call remove_small_from_sparse(air_data%reuse(our_level)%reuse_mat(MAT_W), &
                              air_data%options%r_drop, &
                              air_data%reuse(our_level)%reuse_mat(MAT_W_DROP), &
                              relative_max_row_tol_int= 1)  
            end if
            
            ! Delete temporary if not reusing
            if (.NOT. air_data%options%reuse_sparsity) then
               call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_W), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
               air_data%reuse(our_level)%reuse_mat(MAT_W) = PETSC_NULL_MAT
#endif               
            end if            

            call timer_finish(TIMER_ID_AIR_DROP)                    
         end if      
         
         ! ~~~~~~~~~
         ! Apply constraints to W if needed
         ! ~~~~~~~~~
         if (air_data%options%constrain_w) then
            call timer_start(TIMER_ID_AIR_CONSTRAIN)
            call constrain_grid_transfer(air_data%reuse(our_level)%reuse_mat(MAT_W_DROP), .FALSE., &
                     right_null_vecs_f, right_null_vecs_c)
            call timer_finish(TIMER_ID_AIR_CONSTRAIN)

            do i_loc = 1, size(right_null_vecs_f) 
               call VecDestroy(right_null_vecs_f(i_loc), ierr)       
            end do             
            deallocate(right_null_vecs_f)
         end if      

         ! Now we have W
         ! Build a copy of P with the identity block in it
         ! This is to save having to do Z A_ff W + A_cf W + Z A_fc + A_cc

         ! ~~~~~~~~~~~~~~~~~~~
         ! ~~~~~~~~~~~~~~~~~~~
         ! Calculate the prolongator
         ! ~~~~~~~~~~~~~~~~~~~
         ! ~~~~~~~~~~~~~~~~~~~
         air_data%reuse_one_point_classical_prolong = air_data%options%one_point_classical_prolong .AND. &
               .NOT. air_data%options%symmetric .AND. &
               .NOT. air_data%options%constrain_w .AND. &
               air_data%allocated_matrices_A_ff(our_level)

         ! If we are doing reuse and processor agglomeration on this level, 
         ! then we can't reuse the sparsity of R or P as it gets repartitioned during the setup            
         temp_is = air_data%reuse(our_level)%reuse_is(IS_REPARTITION)
         if (air_data%allocated_matrices_A_ff(our_level) .AND. &
                  .NOT. PetscISIsNull(temp_is)) then

            ! Now when we're doing processor agglomeration, we have to be careful with 
            ! Kokkos, as it gets fussy
            ! about the exact same pointers being passed into spgemm_numeric
            ! (rather than just having the same sparsity) once we've 
            ! repartitioned. We have to force it to repartition to get around this
            ! so the exact same matrices are used in every case
            call MatGetType(air_data%A_ff(our_level), mat_type, ierr)
            if (mat_type == MATMPIAIJKOKKOS) then
               air_data%reuse_one_point_classical_prolong = .FALSE.
            end if

            ! Destroy the grid transfer operators and rebuild them
            call MatDestroy(air_data%restrictors(our_level), ierr)
            if (.NOT. air_data%reuse_one_point_classical_prolong) then
               call MatDestroy(air_data%prolongators(our_level), ierr)
            end if
            reuse_grid_transfer = .FALSE.
         end if

         ! If we've got a one point classical prolongator computed already we can just reuse the prolongator
         ! without change
         if (.NOT. air_data%reuse_one_point_classical_prolong) then
            call compute_P_from_W(air_data%reuse(our_level)%reuse_mat(MAT_W_DROP), &
                     global_row_start, &
                     air_data%IS_fine_index(our_level), air_data%IS_coarse_index(our_level), &
                     .TRUE., &
                     reuse_grid_transfer, &
                     air_data%prolongators(our_level))
         end if

         ! Delete temporary if not reusing
         if (.NOT. air_data%options%reuse_sparsity) then
            call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_W_DROP), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
            air_data%reuse(our_level)%reuse_mat(MAT_W_DROP) = PETSC_NULL_MAT
#endif            
         end if          

         call timer_finish(TIMER_ID_AIR_PROLONG)   
         
      end if

      ! ~~~~~~~~~~~~~~~~~~~
      ! ~~~~~~~~~~~~~~~~~~~
      ! Calculate Z
      ! ~~~~~~~~~~~~~~~~~~~
      ! ~~~~~~~~~~~~~~~~~~~

      ! Calculate the Z component of the restrictor  
      call timer_start(TIMER_ID_AIR_RESTRICT)        

      ! For lAIR
      if (air_data%options%z_type /= AIR_Z_PRODUCT) then

         ! Compute the strongly connected F neighbourhood that each C point
         ! is going to use in lAIR
         ! We use the dropped A_ff, A_cf here as they have had the strong R tolerance applied
         ! to them
         if (air_data%options%lair_distance == 1) then

            sparsity_mat_cf = air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP)
         
         ! Distance 2 is sparsity_mat = A_cf * A_ff
         ! Distance 3 is sparsity_mat = A_cf * A_ff^2         
         ! etc
         else

            ! If we are doing reuse, we already know the sparsity we want
            temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_Z)
            if (.NOT. PetscMatIsNull(temp_mat)) then

               ! We should just be able to use the pointer to air_data%reuse(our_level)%reuse_mat(MAT_Z)
               ! as sparsity_mat_cf, but it gives me errors about unassembled matrices
               ! I think that has to do with the old interface for MatMPIAIJGetSeqAIJ that we are using
               ! in calculate_and_build_sai_z (and the fact it doesn't have a restore)
               ! If we enforce a minimum of petsc 3.19 we could use the new MatMPIAIJGetSeqAIJF90
               ! For now we just take an extra copy
               call MatDuplicate(air_data%reuse(our_level)%reuse_mat(MAT_Z), &
                        MAT_DO_NOT_COPY_VALUES, sparsity_mat_cf, ierr)
               !sparsity_mat_cf = air_data%reuse(our_level)%reuse_mat(MAT_Z)

            else
   
               ! Copy the pointer
               A_ff_power = air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP)
               destroy_mat = .FALSE.
      
               ! Compute A_ff^(distance - 1)
               do order = 3, air_data%options%lair_distance
                  
                  ! Call a symbolic mult as we don't need the values, just the resulting sparsity  
                  A_array = air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP)%v
                  B_array = A_ff_power%v
                  call mat_mat_symbolic_c(A_array, B_array, C_array)
                  ! Don't delete the original power - ie A_ff
                  if (destroy_mat) call MatDestroy(A_ff_power, ierr)
                  A_ff_power%v = C_array  
                  destroy_mat = .TRUE.
      
               end do
      
               ! Call a symbolic mult as we don't need the values, just the resulting sparsity  
               ! A_cf * A_ff^(distance - 1)
               A_array = air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP)%v
               B_array = A_ff_power%v
               call mat_mat_symbolic_c(A_array, B_array, C_array)
               if (destroy_mat) call MatDestroy(A_ff_power, ierr)

               sparsity_mat_cf%v = C_array    

            end if
         end if
   
         ! lAIR
         ! We delibrately have to give it A_ff and A_cf as input, not the dropped versions
         ! The sparsity is controlled by sparsity_mat_cf which has used the dropped versions
         if (air_data%options%z_type == AIR_Z_LAIR) then
            call calculate_and_build_sai_z(air_data%A_ff(our_level), air_data%A_cf(our_level), &
                        sparsity_mat_cf, .TRUE., &
                        air_data%reuse(our_level)%reuse_mat(MAT_SAI_SUB), &
                        air_data%reuse(our_level)%reuse_mat(MAT_Z))
         ! SAI Z
         else
            call calculate_and_build_sai_z(air_data%A_ff(our_level), air_data%A_cf(our_level), &
                        sparsity_mat_cf, .FALSE., &
                        air_data%reuse(our_level)%reuse_mat(MAT_SAI_SUB), &
                        air_data%reuse(our_level)%reuse_mat(MAT_Z))
         end if        
         ! Delete temporary if not reusing
         if (.NOT. air_data%options%reuse_sparsity) then
            call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_SAI_SUB), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
            air_data%reuse(our_level)%reuse_mat(MAT_SAI_SUB) = PETSC_NULL_MAT
#endif            
         end if 
         if (air_data%options%lair_distance .ge. 2) then
            call MatDestroy(sparsity_mat_cf, ierr)        
         end if

      ! For AIRG - we do a matmatmult with our approximate A_ff inverse
      else         
         temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_Z)
         if (.NOT. PetscMatIsNull(temp_mat)) then
            call MatMatMult(air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), minus_mat, &
                  MAT_REUSE_MATRIX, 1.58d0, air_data%reuse(our_level)%reuse_mat(MAT_Z), ierr)            
         else
            call MatMatMult(air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), minus_mat, &
                  MAT_INITIAL_MATRIX, 1.58d0, air_data%reuse(our_level)%reuse_mat(MAT_Z), ierr) 
         end if
      end if

      ! Destroy the copies, this only decrements the reference counter
      if (air_data%options%strong_r_threshold == 0d0) then
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP), ierr)      
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), ierr)
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP), ierr)         

#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
         air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP) = PETSC_NULL_MAT
         air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP) = PETSC_NULL_MAT
         air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP) = PETSC_NULL_MAT
#endif         
      end if

      ! Delete temporaries if not reusing
      if (.NOT. air_data%options%reuse_sparsity) then
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP), ierr)      
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), ierr)
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP), ierr)

#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
         air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP) = PETSC_NULL_MAT
         air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP) = PETSC_NULL_MAT
         air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP) = PETSC_NULL_MAT
#endif
      end if        
      
      ! ~~~~~~~~~~~~
      ! ~~~~~~~~~~~~

      call timer_start(TIMER_ID_AIR_DROP)    

      ! If we want to reuse, we have to match the original sparsity
      ! which might be different if the matrix has changed (but the sparsity is the same)
      ! so we can't just drop with a drop tolerance
      temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_Z_DROP)
      if (.NOT. PetscMatIsNull(temp_mat)) then

         call remove_from_sparse_match_no_lump(air_data%reuse(our_level)%reuse_mat(MAT_Z), &
                  air_data%reuse(our_level)%reuse_mat(MAT_Z_DROP))     
         
      ! First time so just drop according to a tolerance 
      else
         call remove_small_from_sparse(air_data%reuse(our_level)%reuse_mat(MAT_Z), &
                     air_data%options%r_drop, air_data%reuse(our_level)%reuse_mat(MAT_Z_DROP), &
                     relative_max_row_tol_int= 1)  
      end if

      call timer_finish(TIMER_ID_AIR_DROP)   
      ! Delete temporary if not reusing
      if (.NOT. air_data%options%reuse_sparsity) then
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_Z), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
         air_data%reuse(our_level)%reuse_mat(MAT_Z) = PETSC_NULL_MAT
#endif         
      end if       
      if (air_data%options%z_type == AIR_Z_PRODUCT) then
         call MatDestroy(minus_mat, ierr)
      end if

      ! ~~~~~~~~~
      ! Apply constraints to Z if needed
      ! ~~~~~~~~~
      if (air_data%options%constrain_z) then

         call timer_start(TIMER_ID_AIR_CONSTRAIN)
         call constrain_grid_transfer(air_data%reuse(our_level)%reuse_mat(MAT_Z_DROP), .TRUE., &
                     left_null_vecs_f, left_null_vecs_c)
         call timer_finish(TIMER_ID_AIR_CONSTRAIN)

         do i_loc = 1, size(left_null_vecs_f) 
            call VecDestroy(left_null_vecs_f(i_loc), ierr)       
         end do             
         deallocate(left_null_vecs_f)
      end if                  

      ! ~~~~~~~~~~~~~~~~~~~
      ! ~~~~~~~~~~~~~~~~~~~
      ! Calculate R
      ! ~~~~~~~~~~~~~~~~~~~
      ! ~~~~~~~~~~~~~~~~~~~    
      call compute_R_from_Z(air_data%reuse(our_level)%reuse_mat(MAT_Z_DROP), global_row_start, &
               air_data%IS_fine_index(our_level), air_data%IS_coarse_index(our_level), &
               air_data%reuse(our_level)%reuse_is(IS_R_Z_FINE_COLS), &
               .TRUE., &
               reuse_grid_transfer, &
               air_data%restrictors(our_level))

      call timer_finish(TIMER_ID_AIR_RESTRICT)      

      call timer_start(TIMER_ID_AIR_IDENTITY)            
           
      ! We need to calculate any data we need during the fc smooth which 
      ! uses (something like) a VecISCopy
      if (.NOT. air_data%allocated_matrices_A_ff(our_level)) then
         call create_VecISCopyLocalWrapper(air_data, our_level, A)
      end if       
      
      call timer_finish(TIMER_ID_AIR_IDENTITY)            
      
      ! Delete temporary if not reusing
      if (.NOT. air_data%options%reuse_sparsity) then
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_Z_DROP), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
         air_data%reuse(our_level)%reuse_mat(MAT_Z_DROP) = PETSC_NULL_MAT
#endif         

         call ISDestroy(air_data%reuse(our_level)%reuse_is(IS_R_Z_FINE_COLS), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)         
         air_data%reuse(our_level)%reuse_is(IS_R_Z_FINE_COLS) = PETSC_NULL_IS
#endif
      end if      
      
      ! Delete temporary if not reusing
      if (.NOT. air_data%options%one_c_smooth .AND. .NOT. air_data%options%reuse_sparsity) then      
         call MatDestroy(air_data%A_cf(our_level), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
         air_data%A_cf(our_level) = PETSC_NULL_MAT
#endif         
      end if
            
      ! Transpose the restrictor if needed
      if (air_data%options%symmetric) then
         call MatTranspose(air_data%restrictors(our_level), MAT_INITIAL_MATRIX, air_data%prolongators(our_level), ierr)
         call MatDestroy(air_data%restrictors(our_level), ierr)
      end if

      ! ~~~~~~~~~~~
      ! If we are doing C-point smoothing, finish the comms
      ! ~~~~~~~~~~~
      if (air_data%options%one_c_smooth .AND. &
               .NOT. air_data%options%full_smoothing_up_and_down) then      

         call timer_start(TIMER_ID_AIR_INVERSE)           
                  
         call finish_approximate_inverse(air_data%A_cc(our_level), &
                  air_data%options%inverse_type, &
                  air_data%inv_A_cc_poly_data(our_level)%gmres_poly_order, &
                  air_data%inv_A_cc_poly_data(our_level)%gmres_poly_sparsity_order, &
                  air_data%inv_A_cc_poly_data(our_level)%buffers, &
                  air_data%inv_A_cc_poly_data(our_level)%coefficients, &
                  air_data%options%matrix_free_polys, &
                  air_data%reuse(our_level)%reuse_mat(MAT_INV_ACC), &
                  air_data%inv_A_cc(our_level))
                  
         call timer_finish(TIMER_ID_AIR_INVERSE) 
         
         ! Delete temporary if not reusing
         if (.NOT. air_data%options%reuse_sparsity) then
            call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_INV_ACC), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
            air_data%reuse(our_level)%reuse_mat(MAT_INV_ACC) = PETSC_NULL_MAT
#endif            
         end if         

      end if
         
   end subroutine finish_comms_compute_restrict_prolong  

       
   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine compute_coarse_matrix(A, our_level, air_data, &
                     coarse_matrix)

      ! Computes the coarse grid matrix with dropping
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(inout)                 :: A
      integer, intent(in)                       :: our_level
      type(air_multigrid_data), intent(inout)   :: air_data      
      type(tMat), intent(inout)                 :: coarse_matrix

      PetscErrorCode :: ierr
      type(tMat) :: temp_mat

      ! ~~~~~~~~~~

      ! ~~~~~~~~~~~~~~~~~~~
      ! ~~~~~~~~~~~~~~~~~~~
      ! Now build our coarse grid matrix
      ! Can be computed from either RAP, or the smaller
      ! Z A_ff W + A_cf W + Z A_fc + A_cc
      ! ~~~~~~~~~~~~~~~~~~~
      ! ~~~~~~~~~~~~~~~~~~~
      call timer_start(TIMER_ID_AIR_RAP)                          

      ! Can just do PtAP
      if (air_data%options%symmetric) then  

         ! If we've done this before we can reuse
         temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_RAP)
         if (.NOT. PetscMatIsNull(temp_mat)) then

            call MatPtap(A, air_data%prolongators(our_level), &
                     MAT_REUSE_MATRIX, 1.58d0, air_data%reuse(our_level)%reuse_mat(MAT_RAP), ierr)             

         ! First time
         else

            call MatPtap(A, air_data%prolongators(our_level), &
                     MAT_INITIAL_MATRIX, 1.58d0, air_data%reuse(our_level)%reuse_mat(MAT_RAP), ierr)          
         end if

      ! ~~~~~~~~~~~
      ! Do two matmatmults rather than the triple product
      ! ~~~~~~~~~~~            
      else

         ! If we've done this before we can reuse
         temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_AP)
         if (.NOT. PetscMatIsNull(temp_mat)) then

            call MatMatMult(A, air_data%prolongators(our_level), &
                     MAT_REUSE_MATRIX, 1.58d0, air_data%reuse(our_level)%reuse_mat(MAT_AP), ierr)     
                     
            call MatMatMult(air_data%restrictors(our_level), air_data%reuse(our_level)%reuse_mat(MAT_AP), &
                     MAT_REUSE_MATRIX, 1.58d0, air_data%reuse(our_level)%reuse_mat(MAT_RAP), ierr)             
         
         ! First time
         else

            call MatMatMult(A, air_data%prolongators(our_level), &
                     MAT_INITIAL_MATRIX, 1.58d0, air_data%reuse(our_level)%reuse_mat(MAT_AP), ierr)     
                     
            call MatMatMult(air_data%restrictors(our_level), air_data%reuse(our_level)%reuse_mat(MAT_AP), &
                     MAT_INITIAL_MATRIX, 1.58d0, air_data%reuse(our_level)%reuse_mat(MAT_RAP), ierr) 
         end if
         
         ! Delete temporary if not reusing
         if (.NOT. air_data%options%reuse_sparsity) then
            call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_AP), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
            air_data%reuse(our_level)%reuse_mat(MAT_AP) = PETSC_NULL_MAT
#endif            
         end if          
      end if

      call timer_finish(TIMER_ID_AIR_RAP)        
               
      ! Drop relative small entries         
      call timer_start(TIMER_ID_AIR_DROP)   

      ! If we want to reuse, we have to match the original sparsity
      ! which might be different if the matrix has changed (but the sparsity is the same)
      ! so we can't just drop with a drop tolerance
      temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_RAP_DROP)
      if (.NOT. PetscMatIsNull(temp_mat)) then

         call remove_from_sparse_match(air_data%reuse(our_level)%reuse_mat(MAT_RAP), &
                  air_data%reuse(our_level)%reuse_mat(MAT_RAP_DROP), coarse_matrix, &
                  lump=air_data%options%a_lump)

      ! First time so just drop according to a tolerance 
      else
         call remove_small_from_sparse(air_data%reuse(our_level)%reuse_mat(MAT_RAP), &
                  air_data%options%a_drop, air_data%reuse(our_level)%reuse_mat(MAT_RAP_DROP), &
                  relative_max_row_tol_int = 1, lump=air_data%options%a_lump)

         call MatDuplicate(air_data%reuse(our_level)%reuse_mat(MAT_RAP_DROP), &
                     MAT_COPY_VALUES, coarse_matrix, ierr)
      end if

      ! Delete temporary if not reusing
      if (.NOT. air_data%options%reuse_sparsity) then
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_RAP), ierr)
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_RAP_DROP), ierr)

#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
         air_data%reuse(our_level)%reuse_mat(MAT_RAP) = PETSC_NULL_MAT
         air_data%reuse(our_level)%reuse_mat(MAT_RAP_DROP) = PETSC_NULL_MAT
#endif
      end if       

      call timer_finish(TIMER_ID_AIR_DROP)    

         
   end subroutine compute_coarse_matrix     

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine setup_air_pcmg(amat, pmat, air_data, pcmg)

      ! Setup AIR by computing the hierarchy and returning a PETSC PCMG object 
      ! Have to have called create_air_data before this routine
      ! The options for AIR are stored in air_data%options

      ! ~~~~~~
      type(tMat), target, intent(in)                     :: amat, pmat
      type(air_multigrid_data), target, intent(inout)    :: air_data
      type(tPC), intent(inout)                           :: pcmg

      ! Local
      PetscInt            :: local_rows, local_cols, global_rows, global_cols
      PetscInt            :: local_fine_is_size, local_coarse_is_size
      PetscInt            :: global_coarse_is_size, global_fine_is_size, global_row_start
      PetscInt            :: global_row_end_plus_one, no_active_cores
      PetscInt            :: prolongator_start, prolongator_end_plus_one, proc_stride
      PetscInt            :: petsc_level, no_levels_petsc_int
      PetscInt            :: local_vec_size, ystart, yend, local_rows_repart, local_cols_repart
      PetscInt            :: global_rows_repart, global_cols_repart
      integer             :: i_loc
      integer             :: no_levels, our_level, our_level_coarse, errorcode, comm_rank, comm_size
      PetscErrorCode      :: ierr
      MPI_Comm            :: MPI_COMM_MATRIX
      PetscReal           :: ratio_local_nnzs_off_proc, achieved_rel_tol, norm_b
      logical             :: continue_coarsening, trigger_proc_agglom
      type(tMat)          :: temp_mat
      type(tKSP)          :: ksp_smoother_up, ksp_smoother_down, ksp_coarse_solver
      type(tPC)           :: pc_smoother_up, pc_smoother_down, pc_coarse_solver
      type(tVec)          :: temp_coarse_vec, rand_vec, sol_vec, temp_vec
      type(tIS)           :: is_unchanged, is_full, temp_is
      type(mat_ctxtype), pointer :: mat_ctx
      PetscInt, parameter :: one=1, zero=0
      type(tVec), dimension(:), allocatable :: left_null_vecs, right_null_vecs
      type(tVec), dimension(:), allocatable :: left_null_vecs_c, right_null_vecs_c
      VecScatter :: vec_scatter
      VecType :: vec_type
      logical :: auto_truncated
      PetscRandom :: rctx
      MatType:: mat_type

      ! ~~~~~~     

      ! Start timing the setup
      call timer_start(TIMER_ID_AIR_SETUP)    
      
      ! Get the communicator the input matrix is on, we build everything on that
      call PetscObjectGetComm(pmat, MPI_COMM_MATRIX, ierr)
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)        
      ! Get the comm rank
      call MPI_Comm_rank(MPI_COMM_MATRIX, comm_rank, errorcode)      
      
      ! The max number of levels
      no_levels = air_data%options%max_levels    
      ! Keep track of how many times we've done processor agglomeration
      ! ie the stride between active mpi ranks
      proc_stride = 1
      ! Copy the top grid matrix pointer
      air_data%coarse_matrix(1) = pmat

      ! If on the cpu we have a veciscopy which is fast
      ! If on the gpu with kokkos we have a veciscopy which is fast
      ! If on the gpu without kokkos the veciscopy involves the host which is slow
      ! For the slow ones we instead create some extra matrices to use during the smoothing
      call MatGetType(air_data%coarse_matrix(1), mat_type, ierr)
      air_data%fast_veciscopy_exists = .TRUE.
      if (mat_type == MATSEQAIJCUSPARSE .OR. mat_type == MATMPIAIJCUSPARSE .OR. mat_type == MATAIJCUSPARSE .OR. &  
          mat_type == MATSEQAIJHIPSPARSE .OR. mat_type == MATMPIAIJHIPSPARSE .OR. mat_type == MATAIJHIPSPARSE .OR. &
          mat_type == MATSEQAIJVIENNACL .OR. mat_type == MATMPIAIJVIENNACL .OR. mat_type == MATAIJVIENNACL .OR. &
          mat_type == MATDENSECUDA .OR. mat_type == MATDENSEHIP .OR. &
          mat_type == MATSEQDENSECUDA .OR. mat_type == MATSEQDENSEHIP .OR. &
          mat_type == MATMPIDENSECUDA .OR. mat_type == MATMPIDENSEHIP) then

         air_data%fast_veciscopy_exists = .FALSE.
      end if

      ! ~~~~~~~~~~~~~~~~~~~~~
      ! Check if the user has provided a near nullspace before we do anything
      ! ~~~~~~~~~~~~~~~~~~~~~
      call get_near_nullspace(amat, air_data%options%constrain_z, air_data%options%constrain_w, &
               left_null_vecs, right_null_vecs)

      ! ~~~~~~~~~~~~~~~~~~~~~

      ! If we have an exact IS, we know Aff is diagonal and our inverse is exact
      if (air_data%options%strong_threshold == 0d0) then   
         ! Only have to do one "smooth" to apply the exact inverse
         air_data%options%maxits_a_ff = 1        
      end if                 

      if (air_data%options%print_stats_timings .AND. comm_rank == 0) print *, "Timers are cumulative"

      auto_truncated = .FALSE.

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~
      ! Loop over the number of levels
      ! We will exit this loop once we coarsen far enough
      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~
      level_loop: do our_level = 1, no_levels-1

         ! This is our coarse level
         our_level_coarse = our_level + 1

         ! Get matrix sizes
         call MatGetSize(air_data%coarse_matrix(our_level), global_rows, global_cols, ierr)
         call MatGetLocalSize(air_data%coarse_matrix(our_level), local_rows, local_cols, ierr)
         ! This returns the global index of the local portion of the matrix
         call MatGetOwnershipRange(air_data%coarse_matrix(our_level), global_row_start, global_row_end_plus_one, ierr)           

         continue_coarsening = .TRUE.

         ! ~~~~~~~~~~
         ! We can also check if our coarse grid approximations are good enough to work as a coarse grid solver
         ! If so we can stop coarsening here   
         ! This is really only a sensible idea when using a matrix-free polynomial for the coarse grid solve
         ! Otherwise building assembled approximation inverses can be very expensive!      
         ! ~~~~~~~~~~
         ! We already know how many coarse levels we have if we are re-using
         if (.NOT. air_data%allocated_matrices_A_ff(our_level) .AND. &
                     our_level .ge. air_data%options%auto_truncate_start_level .AND. &
                     air_data%options%auto_truncate_start_level /= -1) then         

            call timer_start(TIMER_ID_AIR_TRUNCATE)   

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
                  air_data%options%coarsest_inverse_type, &
                  air_data%inv_coarsest_poly_data%gmres_poly_order, &
                  air_data%inv_coarsest_poly_data%buffers, air_data%inv_coarsest_poly_data%coefficients)                       

            ! This will be a vec of randoms that differ from those used to create the gmres polynomials
            ! We will solve Ax = rand_vec to test how good our coarse solver is
            call PetscRandomCreate(MPI_COMM_MATRIX, rctx, ierr)
            call MatCreateVecs(air_data%coarse_matrix(our_level), &
                     rand_vec, PETSC_NULL_VEC, ierr)            

            call VecSetRandom(rand_vec, rctx, ierr)
            call PetscRandomDestroy(rctx, ierr)

            call VecDuplicate(rand_vec, sol_vec, ierr)
            call VecDuplicate(rand_vec, temp_vec, ierr)

            ! Finish our approximate inverse
            call finish_approximate_inverse(air_data%coarse_matrix(our_level), &
                  air_data%options%coarsest_inverse_type, &
                  air_data%inv_coarsest_poly_data%gmres_poly_order, air_data%inv_coarsest_poly_data%gmres_poly_sparsity_order, &
                  air_data%inv_coarsest_poly_data%buffers, air_data%inv_coarsest_poly_data%coefficients, &
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
               auto_truncated = .TRUE.

               ! Delete temporary if not reusing
               if (.NOT. air_data%options%reuse_sparsity) then
                  call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
                  air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF) = PETSC_NULL_MAT
#endif            
               end if                  

            ! If this isn't good enough, destroy everything we used - no chance for reuse
            else
               call MatDestroy(air_data%inv_A_ff(our_level), ierr)               
               call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
               air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF) = PETSC_NULL_MAT
#endif                
            end if      

            call VecDestroy(rand_vec, ierr)
            call VecDestroy(sol_vec, ierr)
            call VecDestroy(temp_vec, ierr)

            call timer_finish(TIMER_ID_AIR_TRUNCATE)   
         end if

         ! ~~~~~~~~~~~~
         ! Compute the coarsening
         ! ~~~~~~~~~~~~     
         call timer_start(TIMER_ID_AIR_COARSEN)     
         
         ! Are we reusing our CF splitting
         if (.NOT. air_data%allocated_is(our_level) .AND. .NOT. auto_truncated) then

            ! Do the CF splitting
            call compute_cf_splitting(air_data%coarse_matrix(our_level), &
                  air_data%options%symmetric, &
                  air_data%options%strong_threshold, &
                  air_data%options%max_luby_steps, &
                  air_data%options%cf_splitting_type, &
                  air_data%options%ddc_fraction, &
                  air_data%IS_fine_index(our_level), air_data%IS_coarse_index(our_level))      
            air_data%allocated_is(our_level) = .TRUE.
         end if
         
         call timer_finish(TIMER_ID_AIR_COARSEN)   

         ! ~~~~~~~~~~~~~~
         ! Get the sizes of C and F points
         ! ~~~~~~~~~~~~~~
         if (.NOT. auto_truncated) then

            call ISGetSize(air_data%IS_fine_index(our_level), global_fine_is_size, ierr)
            call ISGetLocalSize(air_data%IS_fine_index(our_level), local_fine_is_size, ierr)
            call ISGetSize(air_data%IS_coarse_index(our_level), global_coarse_is_size, ierr)
            call ISGetLocalSize(air_data%IS_coarse_index(our_level), local_coarse_is_size, ierr)  

         ! We're not continuing the coarsening anyway, this is just to ensure the continue_coarsening
         ! test below doesn't break
         else
            global_fine_is_size = 0
            global_coarse_is_size = 0
         end if             

         ! Do we want to keep coarsening?
         ! We check if our coarse grid solve is already good enough and
         ! if the problem is still big enough and
         ! that the coarsening resulted in any fine points, sometimes
         ! you can have it such that no fine points are selected                  
         continue_coarsening = .NOT. auto_truncated .AND. &
                  (global_coarse_is_size > air_data%options%coarse_eq_limit .AND. global_fine_is_size /= 0)  

         ! Did we end up with a coarse grid we still want to coarsen?
         if (continue_coarsening) then

            ! Output stats on the coarsening
            if (air_data%options%print_stats_timings .AND. comm_rank == 0) then
               print *, "~~~~~~~~~~~~ Level ", our_level
               print *, "Global rows", global_rows, "Global F-points", global_fine_is_size, "Global C-points", global_coarse_is_size   
            end if
       
         ! If this coarse grid is smaller than our minimum, then we are done coarsening
         else

            ! Get the size of the previous coarse grid as that is now our bottom grid
            ! If we're on the top grid and we have coarsened fast enough to not have a second level
            ! should only happen on very small problems               
            if (our_level == 1) then
               global_fine_is_size = global_rows
               local_fine_is_size = local_rows
            else
               call ISGetSize(air_data%IS_coarse_index(our_level-1), global_fine_is_size, ierr)
               call ISGetLocalSize(air_data%IS_coarse_index(our_level-1), local_fine_is_size, ierr)

               call MatCreateVecs(air_data%A_fc(our_level-1), &
                        air_data%temp_vecs_fine(1)%array(our_level), PETSC_NULL_VEC, ierr)               
            end if

            if (air_data%options%constrain_z) then
               ! Destroy our copy of the left near nullspace vectors
               do i_loc = 1, size(left_null_vecs)
                  call VecDestroy(left_null_vecs(i_loc), ierr)
               end do
               deallocate(left_null_vecs)
               if (allocated(left_null_vecs_c)) deallocate(left_null_vecs_c)
            end if
            if (air_data%options%constrain_w) then
               ! Destroy our copy of the right near nullspace vectors
               do i_loc = 1, size(right_null_vecs)
                  call VecDestroy(right_null_vecs(i_loc), ierr)
               end do
               deallocate(right_null_vecs)
               if (allocated(right_null_vecs_c)) deallocate(right_null_vecs_c)
            end if            

            no_levels = our_level

            ! Exit out of the coarsening loop
            exit level_loop

         end if    

         ! ~~~~~~~~~~~~~~     
         ! Now let's go and build all our operators
         ! ~~~~~~~~~~~~~~                         

         ! ~~~~~~~~~
         ! Let's smooth near null-space vectors if needed
         ! ~~~~~~~~~
         if (air_data%options%constrain_z .OR. air_data%options%constrain_w) then
            call timer_start(TIMER_ID_AIR_CONSTRAIN)
            call smooth_near_nullspace(air_data%coarse_matrix(our_level), &
               air_data%options%constrain_z, &
               air_data%options%constrain_w, &
               left_null_vecs, right_null_vecs)
            call timer_finish(TIMER_ID_AIR_CONSTRAIN)
         end if


         ! ~~~~~~~~~
         ! Setup the details of our gmres polynomials
         ! ~~~~~~~~~         

         call setup_gmres_poly_data(global_fine_is_size, &
                  air_data%options%inverse_type, &
                  air_data%options%poly_order, &
                  air_data%options%inverse_sparsity_order, &
                  air_data%options%subcomm, &
                  proc_stride, &
                  air_data%inv_A_ff_poly_data(our_level))

         ! Setup the same structure for the inv_A_ff made from dropped Aff 
         call setup_gmres_poly_data(global_fine_is_size, &
                  air_data%options%inverse_type, &
                  air_data%options%poly_order, &
                  air_data%options%inverse_sparsity_order, &
                  air_data%options%subcomm, &
                  proc_stride, &
                  air_data%inv_A_ff_poly_data_dropped(our_level)) 

         ! ~~~~~~~~~
         ! If we're doing C-point smoothing we may have a gmres polynomial on C points
         ! ~~~~~~~~~
         if (air_data%options%one_c_smooth .AND. &
                  .NOT. air_data%options%full_smoothing_up_and_down) then                  
                  
            call setup_gmres_poly_data(global_coarse_is_size, &
                     air_data%options%c_inverse_type, &
                     air_data%options%c_poly_order, &
                     air_data%options%c_inverse_sparsity_order, &
                     air_data%options%subcomm, &
                     proc_stride, &
                     air_data%inv_A_cc_poly_data(our_level))   
         end if            

         ! ~~~~~~~~~
         ! Extract the submatrices and start the comms to compute the approximate inverses
         ! ~~~~~~~~~         
         call get_submatrices_start_poly_coeff_comms(air_data%coarse_matrix(our_level), &
               our_level, air_data)         
               
         ! ~~~~~~~
         ! Temporary vecs we use in the smoother
         ! ~~~~~~~
         ! If we haven't built them already
         if (.NOT. air_data%allocated_matrices_A_ff(our_level)) then
            call MatCreateVecs(air_data%A_ff(our_level), &
                     air_data%temp_vecs_fine(1)%array(our_level), PETSC_NULL_VEC, ierr)
            call MatCreateVecs(air_data%A_fc(our_level), &
                     air_data%temp_vecs_coarse(1)%array(our_level), PETSC_NULL_VEC, ierr)  
            call MatCreateVecs(air_data%coarse_matrix(our_level), &
                     air_data%temp_vecs(1)%array(our_level), PETSC_NULL_VEC, ierr)                                                  

            if (our_level == no_levels - 1) then
               call MatCreateVecs(air_data%A_fc(our_level), PETSC_NULL_VEC, &
                        air_data%temp_vecs_fine(1)%array(our_level+1), ierr)                                         
            end if                  
         end if   
         
         ! ~~~~~~~~~~~~~~
         ! If we got here then we want to generate a new coarse matrix
         ! ~~~~~~~~~~~~~~

         ! Generate the temporary vectors we use to smooth with
         ! If we haven't built them already
         if (.NOT. air_data%allocated_matrices_A_ff(our_level)) then         
            call VecDuplicate(air_data%temp_vecs_fine(1)%array(our_level), air_data%temp_vecs_fine(2)%array(our_level), ierr)
            call VecDuplicate(air_data%temp_vecs_fine(1)%array(our_level), air_data%temp_vecs_fine(3)%array(our_level), ierr)
            call VecDuplicate(air_data%temp_vecs_fine(1)%array(our_level), air_data%temp_vecs_fine(4)%array(our_level), ierr)        

            ! If we're doing C point smoothing we need some extra temporaries
            if (air_data%options%one_c_smooth .AND. &
                     .NOT. air_data%options%full_smoothing_up_and_down) then
               call VecDuplicate(air_data%temp_vecs_coarse(1)%array(our_level), air_data%temp_vecs_coarse(2)%array(our_level), ierr)
               call VecDuplicate(air_data%temp_vecs_coarse(1)%array(our_level), air_data%temp_vecs_coarse(3)%array(our_level), ierr)
               call VecDuplicate(air_data%temp_vecs_coarse(1)%array(our_level), air_data%temp_vecs_coarse(4)%array(our_level), ierr)         
            end if
         end if         

         ! ~~~~~~~~~
         ! Finish the non-blocking comms and build the approximate inverse, then the 
         ! restrictor and prolongator
         ! ~~~~~~~~~
         call finish_comms_compute_restrict_prolong(air_data%coarse_matrix(our_level), &
               our_level, air_data, &
               left_null_vecs, right_null_vecs, &
               left_null_vecs_c, right_null_vecs_c)

         if (air_data%options%constrain_z) then
            ! Destroy our copy of the left near nullspace vectors
            do i_loc = 1, size(left_null_vecs)
               call VecDestroy(left_null_vecs(i_loc), ierr)
            end do
            left_null_vecs = left_null_vecs_c
         end if
         if (air_data%options%constrain_w) then
            ! Destroy our copy of the right near nullspace vectors
            do i_loc = 1, size(right_null_vecs)
               call VecDestroy(right_null_vecs(i_loc), ierr)
            end do
            right_null_vecs = right_null_vecs_c
         end if

         ! ~~~~~~~~~~~~~~
         ! Build the coarse matrix
         ! ~~~~~~~~~~~~~~

         call compute_coarse_matrix(air_data%coarse_matrix(our_level), our_level, air_data, &
                  air_data%coarse_matrix(our_level_coarse))  

         ! ~~~~~~~~~~~
         ! ~~~~~~~~~~~            

         ! ~~~~~~~~~~~~
         ! Do processor agglomeration if desired
         ! We stay on the existing communicator with some cores just having zero dofs
         ! ~~~~~~~~~~~~
         if (air_data%options%processor_agglom) then

            ! If we're in parallel and we haven't already agglomerated down to one processor
            if (comm_size /= 1 .AND. proc_stride /= comm_size) then

               ! Number of cores we have dofs on
               ! Stolen from calculate_repartition, make sure they match!
               no_active_cores = floor(dble(comm_size)/dble(proc_stride))
               ! Be careful of rounding!
               if (no_active_cores == 0) no_active_cores = 1                 
               ratio_local_nnzs_off_proc = 0d0

               ! If we have already setup our hierarchy, then we know what levels need to be repartitioned
               ! If not, then we have to compute the ratio on each level to check if they need to be 
               if (.NOT. air_data%allocated_matrices_A_ff(our_level)) then
                  call compute_mat_ratio_local_nonlocal_nnzs(air_data%coarse_matrix(our_level_coarse), &
                           no_active_cores, ratio_local_nnzs_off_proc)
               end if

               ! Let's check the size of the coarse matrix
               call MatGetSize(air_data%coarse_matrix(our_level_coarse), global_rows_repart, global_cols_repart, ierr)               

               ! If we are reusing and we know we have to repartition this level, or 
               ! we have not very many unknowns per core (on average) or
               ! if we get a local to off-processor ratio of less than processor_agglom_ratio
               temp_is = air_data%reuse(our_level)%reuse_is(IS_REPARTITION)
               trigger_proc_agglom = .NOT. PetscISIsNull(temp_is) .OR. &
                           (global_rows_repart/no_active_cores < air_data%options%process_eq_limit &
                              .AND. no_active_cores /= 1) .OR. &
                           (ratio_local_nnzs_off_proc .le. air_data%options%processor_agglom_ratio &
                           .AND. ratio_local_nnzs_off_proc /= 0d0)
                                 
               ! Start the process agglomeration 
               if (trigger_proc_agglom) then

                  call timer_start(TIMER_ID_AIR_PROC_AGGLOM)
                  ! air_data%options%processor_agglom_factor tells us how much we're reducing the number 
                  ! of active mpi ranks by each time
                  
                  ! proc_stride is fed into the polynomial inverse assembly to tell it how many
                  ! idle ranks we have, so we can use them as threads in omp if it is enabled
                  proc_stride = proc_stride * air_data%options%processor_agglom_factor

                  ! If we don't have at least process_eq_limit unknowns per core (on average)
                  ! then we need to be more aggressive with our processor agglomeration
                  ! We'll just keep increasing the stride until we have more than process_eq_limit unknowns per core
                  stride_loop: do while (global_rows_repart < air_data%options%process_eq_limit * no_active_cores)
                     proc_stride = proc_stride * air_data%options%processor_agglom_factor
                     ! Stolen from calculate_repartition, make sure they match!
                     no_active_cores = floor(dble(comm_size)/dble(proc_stride))     
                     ! Be careful of rounding!
                     if (no_active_cores == 0) no_active_cores = 1   
                     ! If we can't agglomerate any more and we still haven't hit the desired
                     ! process_eq_limit we'll just have to live with it
                     if (no_active_cores == 1) exit stride_loop             
                  end do stride_loop               

                  ! If we agglomerate down to one processor
                  if (proc_stride > comm_size) proc_stride = comm_size

                  ! Calculate the IS with the repartitioning if we haven't already
                  ! This is expensive as it calls the graph partitioner (e.g., parmetis)
                  temp_is = air_data%reuse(our_level)%reuse_is(IS_REPARTITION)
                  if (PetscISIsNull(temp_is)) then

                     call calculate_repartition(air_data%coarse_matrix(our_level_coarse), &
                                 proc_stride, no_active_cores, .FALSE., &
                                 air_data%reuse(our_level)%reuse_is(IS_REPARTITION))
                  end if

                  if (air_data%options%print_stats_timings .AND. comm_rank == 0) then
                     print *, "Doing processor agglomeration onto no cores:", no_active_cores
                  end if

                  ! ~~~~~~~~~~~~~~~~~~
                  ! Repartition the coarse matrix
                  ! ~~~~~~~~~~~~~~~~~~
                  temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_COARSE_REPARTITIONED)
                  if (.NOT. PetscMatIsNull(temp_mat)) then

                     call MatCreateSubMatrix(air_data%coarse_matrix(our_level_coarse), &
                                 air_data%reuse(our_level)%reuse_is(IS_REPARTITION), &
                                 air_data%reuse(our_level)%reuse_is(IS_REPARTITION), &
                                 MAT_REUSE_MATRIX, &
                                 air_data%reuse(our_level)%reuse_mat(MAT_COARSE_REPARTITIONED), ierr)                     
                  else
                     call MatCreateSubMatrix(air_data%coarse_matrix(our_level_coarse), &
                                 air_data%reuse(our_level)%reuse_is(IS_REPARTITION), &
                                 air_data%reuse(our_level)%reuse_is(IS_REPARTITION), &
                                 MAT_INITIAL_MATRIX, &
                                 air_data%reuse(our_level)%reuse_mat(MAT_COARSE_REPARTITIONED), ierr)
                  end if

                  call MatDestroy(air_data%coarse_matrix(our_level_coarse), ierr)
                  call MatDuplicate(air_data%reuse(our_level)%reuse_mat(MAT_COARSE_REPARTITIONED), &
                           MAT_COPY_VALUES, air_data%coarse_matrix(our_level_coarse), ierr)

                  ! Delete temporary if not reusing
                  if (.NOT. air_data%options%reuse_sparsity) then
                     call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_COARSE_REPARTITIONED), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
                     air_data%reuse(our_level)%reuse_mat(MAT_COARSE_REPARTITIONED) = PETSC_NULL_MAT
#endif                     
                  end if                   

                  ! Create an IS to represent the row indices of the prolongator 
                  ! as we are only repartitioning the coarse matrix not the matrix on this level
                  ! ie the columns of the prolongator 
                  call MatGetOwnershipRange(air_data%prolongators(our_level), prolongator_start, &
                           prolongator_end_plus_one, ierr)                     
                  call ISCreateStride(MPI_COMM_MATRIX, prolongator_end_plus_one - prolongator_start, &
                           prolongator_start, one, is_unchanged, ierr)

                  ! ~~~~~~~~~~~~~~~~~~
                  ! Repartition the prolongator matrix
                  ! ~~~~~~~~~~~~~~~~~~
                  temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_P_REPARTITIONED)
                  if (.NOT. PetscMatIsNull(temp_mat)) then

                     ! If we've got a one point classical prolongator then we just use the existing repartitioned
                     ! one so we don't need to repartition
                     if (.NOT. air_data%reuse_one_point_classical_prolong) then

                        call MatCreateSubMatrix(air_data%prolongators(our_level), &
                                    is_unchanged, air_data%reuse(our_level)%reuse_is(IS_REPARTITION), &
                                    MAT_REUSE_MATRIX, &
                                    air_data%reuse(our_level)%reuse_mat(MAT_P_REPARTITIONED), ierr)                     

                        call MatDestroy(air_data%prolongators(our_level), ierr)
                        call MatDuplicate(air_data%reuse(our_level)%reuse_mat(MAT_P_REPARTITIONED), &
                                 MAT_COPY_VALUES, air_data%prolongators(our_level), ierr)                                 
                     end if
                  else
                     call MatCreateSubMatrix(air_data%prolongators(our_level), &
                                 is_unchanged, air_data%reuse(our_level)%reuse_is(IS_REPARTITION), &
                                 MAT_INITIAL_MATRIX, &
                                 air_data%reuse(our_level)%reuse_mat(MAT_P_REPARTITIONED), ierr)

                     call MatDestroy(air_data%prolongators(our_level), ierr)
                     call MatDuplicate(air_data%reuse(our_level)%reuse_mat(MAT_P_REPARTITIONED), &
                                 MAT_COPY_VALUES, air_data%prolongators(our_level), ierr)                                 
                  end if

                  ! Delete temporary if not reusing
                  if (.NOT. air_data%options%reuse_sparsity) then
                     call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_P_REPARTITIONED), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
                     air_data%reuse(our_level)%reuse_mat(MAT_P_REPARTITIONED) = PETSC_NULL_MAT
#endif                     
                  end if                   
                  
                  ! ~~~~~~~~~~~~~~~~~~
                  ! If need to repartition a restrictor
                  ! ~~~~~~~~~~~~~~~~~~
                  if (.NOT. air_data%options%symmetric) then

                     ! Repartition the restrictor matrix
                     temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_R_REPARTITIONED)
                     if (.NOT. PetscMatIsNull(temp_mat)) then

                        call MatCreateSubMatrix(air_data%restrictors(our_level), &
                                    air_data%reuse(our_level)%reuse_is(IS_REPARTITION), &
                                    is_unchanged, MAT_REUSE_MATRIX, &
                                    air_data%reuse(our_level)%reuse_mat(MAT_R_REPARTITIONED), ierr)                        
                     else
                        call MatCreateSubMatrix(air_data%restrictors(our_level), &
                                    air_data%reuse(our_level)%reuse_is(IS_REPARTITION), &
                                    is_unchanged, MAT_INITIAL_MATRIX, &
                                    air_data%reuse(our_level)%reuse_mat(MAT_R_REPARTITIONED), ierr)
                     end if

                     call MatDestroy(air_data%restrictors(our_level), ierr)
                     call MatDuplicate(air_data%reuse(our_level)%reuse_mat(MAT_R_REPARTITIONED), &
                              MAT_COPY_VALUES, air_data%restrictors(our_level), ierr)

                     ! Delete temporary if not reusing
                     if (.NOT. air_data%options%reuse_sparsity) then
                        call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_R_REPARTITIONED), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
                        air_data%reuse(our_level)%reuse_mat(MAT_R_REPARTITIONED) = PETSC_NULL_MAT
#endif                        
                     end if                     
                  end if

                  ! ~~~~~~~~~~~~~~~~~~
                  ! If need to repartition the coarse right and left near-nullspace vectors
                  ! ~~~~~~~~~~~~~~~~~~     
                  if (air_data%options%constrain_z .OR. air_data%options%constrain_w) then 

                     call MatGetSize(air_data%coarse_matrix(our_level_coarse), &
                           global_rows_repart, global_cols_repart, ierr)
                     call MatGetLocalSize(air_data%coarse_matrix(our_level_coarse), &
                           local_rows_repart, local_cols_repart, ierr)
                     ! Can't use matcreatevecs here on coarse_matrix(our_level_coarse)
                     ! if the coarser matrix has gone down to one process it returns a serial vector
                     ! but we have to have the same type for the scatter (ie mpi and mpi)
                     if (air_data%options%constrain_z) then
                        call VecGetType(left_null_vecs(1), vec_type, ierr)
                     else if (air_data%options%constrain_w) then
                        call VecGetType(right_null_vecs(1), vec_type, ierr)
                     end if
                     call VecCreate(MPI_COMM_MATRIX, temp_coarse_vec, ierr)
                     call VecSetSizes(temp_coarse_vec, local_rows_repart, global_rows_repart, ierr)
                     call VecSetType(temp_coarse_vec, vec_type, ierr)
                     call VecSetUp(temp_coarse_vec, ierr)

                     ! Can't seem to pass in PETSC_NULL_IS to the vecscattercreate in petsc 3.14
                     call VecGetLocalSize(temp_coarse_vec, local_vec_size, ierr)
                     call VecGetOwnershipRange(temp_coarse_vec, ystart, yend, ierr)
                     call ISCreateStride(PETSC_COMM_SELF, local_vec_size, ystart, one, is_full, ierr) 

                     if (air_data%options%constrain_z) then
                        call VecScatterCreate(left_null_vecs(1), &
                                 air_data%reuse(our_level)%reuse_is(IS_REPARTITION), &
                                 temp_coarse_vec, is_full, vec_scatter,ierr)
                     else if (air_data%options%constrain_w) then
                        call VecScatterCreate(right_null_vecs(1), &
                                 air_data%reuse(our_level)%reuse_is(IS_REPARTITION), &
                                 temp_coarse_vec, is_full, vec_scatter,ierr)
                     end if
                  end if

                  ! Could overlap the comms if we stored a copy of the number of vectors
                  
                  ! Do the vec scatters for the left nullspace vecs
                  if (air_data%options%constrain_z) then
                     do i_loc = 1, size(left_null_vecs) 
                        call VecScatterBegin(vec_scatter, left_null_vecs(i_loc), temp_coarse_vec, &
                                    INSERT_VALUES, SCATTER_FORWARD, ierr)
                        call VecScatterEnd(vec_scatter, left_null_vecs(i_loc), temp_coarse_vec, &
                                    INSERT_VALUES, SCATTER_FORWARD, ierr)                                         
                        call VecDestroy(left_null_vecs(i_loc), ierr)
                        call VecDuplicate(temp_coarse_vec, left_null_vecs(i_loc), ierr)
                        call VecCopy(temp_coarse_vec, left_null_vecs(i_loc), ierr)
                     end do
                  end if
                  
                  ! Do the vec scatters for the right nullspace vecs
                  if (air_data%options%constrain_w) then
                     do i_loc = 1, size(right_null_vecs) 
                        call VecScatterBegin(vec_scatter, right_null_vecs(i_loc), temp_coarse_vec, &
                                    INSERT_VALUES, SCATTER_FORWARD, ierr)
                        call VecScatterEnd(vec_scatter, right_null_vecs(i_loc), temp_coarse_vec, &
                                    INSERT_VALUES, SCATTER_FORWARD, ierr)        
                        call VecDestroy(right_null_vecs(i_loc), ierr)
                        call VecDuplicate(temp_coarse_vec, right_null_vecs(i_loc), ierr)
                        call VecCopy(temp_coarse_vec, right_null_vecs(i_loc), ierr)
                     end do                     
                  end if    
                  
                  if (air_data%options%constrain_z .OR. air_data%options%constrain_w) then 
                     call VecDestroy(temp_coarse_vec, ierr)
                     call VecScatterDestroy(vec_scatter, ierr)
                     call ISDestroy(is_full, ierr)
                  end if

                  ! ~~~~~~~~~~~~~~~~~~

                  call ISDestroy(is_unchanged, ierr)
                  ! Delete temporary if not reusing
                  if (.NOT. air_data%options%reuse_sparsity) then
                     call ISDestroy(air_data%reuse(our_level)%reuse_is(IS_REPARTITION), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)                     
                     air_data%reuse(our_level)%reuse_is(IS_REPARTITION) = PETSC_NULL_IS
#endif                     
                  end if                  

                  call timer_finish(TIMER_ID_AIR_PROC_AGGLOM)

               end if             
            end if
         end if

         ! ~~~~~~~~~~~~~~

         air_data%allocated_matrices_A_ff(our_level) = .TRUE.
         if (air_data%options%one_c_smooth .AND. &
                  .NOT. air_data%options%full_smoothing_up_and_down) then          
            air_data%allocated_matrices_A_cc(our_level) = .TRUE.
         end if         

         ! ~~~~~~~~~~~~ 

         ! Get the nnzs for these matrices here, in case we destroy them below
         if (air_data%options%print_stats_timings) then
            call get_nnzs_petsc_sparse(air_data%coarse_matrix(our_level), air_data%coarse_matrix_nnzs(our_level))
         end if

         ! On every level but the top and the bottom we can destroy the full operator matrix
         if (our_level /= 1) then
            if (.NOT. air_data%options%full_smoothing_up_and_down) then
               call MatDestroy(air_data%coarse_matrix(our_level), ierr)
            end if
         end if         
         
         ! If we are just doing F point smoothing, we no longer have our coarse matrix
         ! But we use the mat_ctx in our F-point smoother to tell what level 
         ! we're on, so let's just create an empty matshell to pass in that has the right sizes         
         allocate(mat_ctx)
         mat_ctx%our_level = our_level
         mat_ctx%air_data => air_data
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
         mat_ctx%mat_ida = PETSC_NULL_MAT
#endif         

         if (.NOT. air_data%options%full_smoothing_up_and_down) then
            call MatCreateShell(MPI_COMM_MATRIX, local_rows, local_cols, global_rows, global_cols, &
                        mat_ctx, air_data%coarse_matrix(our_level), ierr)
            call MatAssemblyBegin(air_data%coarse_matrix(our_level), MAT_FINAL_ASSEMBLY, ierr)
            call MatAssemblyEnd(air_data%coarse_matrix(our_level), MAT_FINAL_ASSEMBLY, ierr)   
            
            ! Have to make sure to set the type of vectors the shell creates
            ! Input can be any matrix, we just need the correct type
            call ShellSetVecType(air_data%A_fc(our_level), air_data%coarse_matrix(our_level))                   
         end if
         
         air_data%allocated_coarse_matrix(our_level_coarse) = .TRUE.

         ! ~~~~~~~~~~~~
         ! Output some timing results
         ! ~~~~~~~~~~~~

         if (air_data%options%print_stats_timings .AND. comm_rank == 0) call print_timers()

      end do level_loop

      ! Record how many levels we have
      air_data%no_levels = no_levels

      ! ~~~~~~~~~~~
      ! We can now start the comms for the coarse grid solver
      ! ~~~~~~~~~~~

      call timer_start(TIMER_ID_AIR_INVERSE)   
      call MatGetSize(air_data%coarse_matrix(no_levels), global_rows, global_cols, ierr)

      ! Set up a GMRES polynomial inverse data for the coarse grid solve
      call setup_gmres_poly_data(global_rows, &
               air_data%options%coarsest_inverse_type, &
               air_data%options%coarsest_poly_order, &
               air_data%options%coarsest_inverse_sparsity_order, &
               air_data%options%coarsest_subcomm, &
               proc_stride, &
               air_data%inv_coarsest_poly_data)      
               
      ! Start the inverse for the coarse grid
      ! These comms will be finished below
      temp_mat = air_data%inv_A_ff(air_data%no_levels)
      if (.NOT. (.NOT. PetscMatIsNull(temp_mat) &
                  .AND. air_data%options%reuse_poly_coeffs)) then

         ! We've already created our coarse solver if we've auto truncated         
         if (.NOT. auto_truncated) then
            call start_approximate_inverse(air_data%coarse_matrix(no_levels), &
                  air_data%options%coarsest_inverse_type, &
                  air_data%inv_coarsest_poly_data%gmres_poly_order, &
                  air_data%inv_coarsest_poly_data%buffers, air_data%inv_coarsest_poly_data%coefficients)                   
         end if
      end if
      call timer_finish(TIMER_ID_AIR_INVERSE)  

      ! ~~~~~~~~~~~~~~~
      ! Let's setup the PETSc pc we need
      ! ~~~~~~~~~~~~~~~
      if (no_levels > 1) then

         call PCSetOperators(pcmg, amat, pmat, ierr)
         call PCSetType(pcmg, PCMG, ierr)
         no_levels_petsc_int = no_levels
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR >= 15)
         call PCMGSetLevels(pcmg, no_levels_petsc_int, PETSC_NULL_MPI_COMM, ierr)
#else
         call PCMGSetLevels(pcmg, no_levels_petsc_int, PETSC_NULL_KSP, ierr)
#endif      

         ! If we're doing fc smoothing always have zero down smooths, which petsc calls kaskade
         ! This prevents any unnecessary residual calculations on the way down
         ! Annoyingly kaskade calls the "down" smoother on the way up, so 
         ! we have to set the options carefully for the down smoother
         if (.NOT. air_data%options%full_smoothing_up_and_down) then
            call PCMGSetType(pcmg, PC_MG_KASKADE, ierr)
         end if

         ! PETSc MG levels work in the opposite order to those in our code. If there are N levels:
         !           PETSc   our_level
         ! Fine     N - 1      1
         !          N - 2      2
         !            .        .
         !            1      N - 1
         ! Coarse     0        N

         ! Therefore  ---  petsc_level = N - our_level

         ! Set up the petsc objects on each level
         ! Loop over all levels except the bottom
         do petsc_level = no_levels_petsc_int - 1, 1, -1

            ! Level is reverse ordering
            our_level = no_levels - int(petsc_level)
            our_level_coarse = our_level + 1

            ! Set the restrictor/prolongator
            if (.NOT. air_data%options%symmetric) then
               ! If restrictor is not set petsc will use transpose of prolongator
               call PCMGSetRestriction(pcmg, petsc_level, air_data%restrictors(our_level), ierr)
            end if
            call PCMGSetInterpolation(pcmg, petsc_level, air_data%prolongators(our_level), ierr)

            ! Get smoother for this level
            ! The up smoother is never used or called when doing kaskade, but we set it as a richardson so that petsc doesn't default
            ! to chebychev and hence try and calculate eigenvalues on each grid
            call PCMGGetSmootherUp(pcmg, petsc_level, ksp_smoother_up, ierr)
            call PCMGGetSmootherDown(pcmg, petsc_level, ksp_smoother_down, ierr)                                
            
            ! Set the operators for smoothing on this level
            if (.NOT. air_data%options%full_smoothing_up_and_down) then
               call KSPSetOperators(ksp_smoother_up, air_data%coarse_matrix(our_level), &
                           air_data%coarse_matrix(our_level), ierr)
               call KSPSetOperators(ksp_smoother_down, air_data%coarse_matrix(our_level), &
                           air_data%coarse_matrix(our_level), ierr)
            ! The smoother for all the unknowns is stored in inv_A_ff
            else
               call KSPSetOperators(ksp_smoother_up, air_data%coarse_matrix(our_level), &
                           air_data%inv_A_ff(our_level), ierr)
               call KSPSetOperators(ksp_smoother_down, air_data%coarse_matrix(our_level), &
                           air_data%inv_A_ff(our_level), ierr)               
            end if
            
            ! Set no norm
            call KSPSetNormType(ksp_smoother_up, KSP_NORM_NONE, ierr)
            call KSPSetNormType(ksp_smoother_down, KSP_NORM_NONE, ierr)  
            
            ! Now here is where we have to be careful as we are calling kaskade mg type
            ! It uses the down smoother as the up smoother (with no calls to the up)
            ! And on the way up we have a nonzero initial guess 
            if (.NOT. air_data%options%full_smoothing_up_and_down) then
               call KSPSetInitialGuessNonzero(ksp_smoother_down, PETSC_TRUE, ierr)         
            end if

            ! Get the PC for each smoother
            call KSPGetPC(ksp_smoother_up, pc_smoother_up, ierr)
            call KSPGetPC(ksp_smoother_down, pc_smoother_down, ierr)

            ! Set the smoother
            if (.NOT. air_data%options%full_smoothing_up_and_down) then
               call PCSetType(pc_smoother_up, PCSHELL, ierr)      
               call PCSetType(pc_smoother_down, PCSHELL, ierr)      
            else
               call PCSetType(pc_smoother_up, PCMAT, ierr)
               call PCSetType(pc_smoother_down, PCMAT, ierr)    
            end if

            ! Set richardson
            call KSPSetType(ksp_smoother_up, KSPRICHARDSON, ierr)
            call KSPSetType(ksp_smoother_down, KSPRICHARDSON, ierr)               

            ! We're overwriting the richardson for the fc smoothing
            ! This is automatically disabled if you run with -mg_levels_ksp_monitor fyi!
            if (.NOT. air_data%options%full_smoothing_up_and_down) then
               call PCShellSetApplyRichardson(pc_smoother_up, mg_FC_point_richardson, ierr)
               call PCShellSetApplyRichardson(pc_smoother_down, mg_FC_point_richardson, ierr)
            end if

            ! Zero up smooths for kaskade
            if (.NOT. air_data%options%full_smoothing_up_and_down) then
               ! This is never called anyway with kaskade
               call KSPSetTolerances(ksp_smoother_up, 1d-10, &
                     & 1d-10, &
                     & PETSC_DEFAULT_REAL, &
                     & zero, ierr)                 
            else
               call KSPSetTolerances(ksp_smoother_up, 1d-10, &
                     & 1d-10, &
                     & PETSC_DEFAULT_REAL, &
                     & one, ierr)  
            end if

            ! One up smooth (but we have to set it as the down given kaskade)
            call KSPSetTolerances(ksp_smoother_down, 1d-10, &
                  & 1d-10, &
                  & PETSC_DEFAULT_REAL, &
                  & one, ierr)   
                  
            ! Set up the smoothers on this level
            call PCSetUp(pc_smoother_up, ierr)
            call PCSetUp(pc_smoother_down, ierr)               
            call KSPSetUp(ksp_smoother_up, ierr)
            call KSPSetUp(ksp_smoother_down, ierr)               
            
         end do  

         ! ~~~~~~~~~~~
         ! Then at the bottom of the PCMG we need to build the coarse grid solve
         ! ~~~~~~~~~~~    

         ! Let's do a Richardson
         call PCMGGetCoarseSolve(pcmg, ksp_coarse_solver, ierr)
         ! If you want to apply more iterations of the coarse solver, change this to 
         ! a richardson (can do via command line -mg_coarse_ksp_type richardson)
         call KSPSetType(ksp_coarse_solver, KSPPREONLY, ierr)
         ! Apply one iteration of the coarse solver by default
         call KSPSetTolerances(ksp_coarse_solver, 1d-3, 1d-13, PETSC_DEFAULT_REAL, one, ierr)

         ! Set no norm
         call KSPSetNormType(ksp_coarse_solver, KSP_NORM_NONE, ierr)
         call KSPGetPC(ksp_coarse_solver, pc_coarse_solver, ierr)

         ! ~~~~~~~~~~~~~~~
         ! Finish the comms for the coarse grid solver
         ! ~~~~~~~~~~~~~~~
         ! Coarse grid polynomial coefficients
         call timer_start(TIMER_ID_AIR_INVERSE) 

         ! We've already created our coarse solver if we've auto truncated
         if (.NOT. auto_truncated) then

            call finish_approximate_inverse(air_data%coarse_matrix(no_levels), &
                  air_data%options%coarsest_inverse_type, &
                  air_data%inv_coarsest_poly_data%gmres_poly_order, &
                  air_data%inv_coarsest_poly_data%gmres_poly_sparsity_order, &
                  air_data%inv_coarsest_poly_data%buffers, &
                  air_data%inv_coarsest_poly_data%coefficients, &
                  air_data%options%coarsest_matrix_free_polys, &
                  air_data%reuse(air_data%no_levels)%reuse_mat(MAT_INV_AFF), &
                  air_data%inv_A_ff(air_data%no_levels))           

            ! Delete temporary if not reusing
            if (.NOT. air_data%options%reuse_sparsity) then
               call MatDestroy(air_data%reuse(air_data%no_levels)%reuse_mat(MAT_INV_AFF), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
               air_data%reuse(air_data%no_levels)%reuse_mat(MAT_INV_AFF) = PETSC_NULL_MAT
#endif            
            end if                      
         end if  

         ! Use the mf coarse grid solver or not
         ! Let's store the coarse grid solver in inv_A_ff(no_levels)
         if (.NOT. air_data%options%coarsest_matrix_free_polys) then
            if (air_data%options%print_stats_timings) then
               call get_nnzs_petsc_sparse(air_data%inv_A_ff(air_data%no_levels), &
                        air_data%inv_A_ff_nnzs(air_data%no_levels))
            end if
         end if      

         ! Now we've finished the coarse grid solver, output the time
         call timer_finish(TIMER_ID_AIR_INVERSE)               

         ! This has to be called after we've built the coarse grid inverse
         call KSPSetOperators(ksp_coarse_solver, air_data%coarse_matrix(no_levels), &
                     air_data%inv_A_ff(no_levels), ierr)

         ! ~~~~~~~~~~~~
         ! Set our coarse grid solver
         ! ~~~~~~~~~~~~
         ! Just apply the approximate inverse matrix
         call PCSetType(pc_coarse_solver, PCMAT, ierr)      
         call PCSetUp(pc_coarse_solver, ierr)
         call KSPSetUp(ksp_coarse_solver, ierr)   

      ! If we've only got one level 
      else
         ! Precondition with the "coarse grid" solver we used to determine auto truncation
         if (auto_truncated) then
            call PetscObjectReference(amat, ierr) 
            call PCSetOperators(pcmg, amat, &
                        air_data%inv_A_ff(no_levels), ierr)         
            call PCSetType(pcmg, PCMAT, ierr)

         ! Otherwise just do a jacobi and tell the user
         else
            
            ! If we've only got one level just precondition with jacobi
            call PCSetOperators(pcmg, amat, pmat, ierr)
            call PCSetType(pcmg, PCJACOBI, ierr)
            if (comm_rank == 0) print *, "Only a single level, defaulting to Jacobi PC"
         end if
      end if      

      ! Call the setup on our PC
      call PCSetUp(pcmg, ierr)

      call timer_finish(TIMER_ID_AIR_SETUP)                
      ! Print out the coarse grid info
      if (air_data%options%print_stats_timings .AND. comm_rank == 0) then
         print *, "~~~~~~~~~~~~ Coarse grid ", no_levels
         print *, "Global rows", global_rows
         call print_timers()
         print *, "~~~~~~~~~~~~ "      
         print *,  "Total cumulative setup time :", timer_time(TIMER_ID_AIR_SETUP)
         print *, "~~~~~~~~~~~~ "      
      end if
      ! Print out stats on the hierarchy - collective so make sure to call 
      ! this on all ranks      
      if (air_data%options%print_stats_timings) call print_stats(air_data, pcmg)   

   end subroutine setup_air_pcmg       
   
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
                  ! If the strong threshold is zero we just copy the pointers for 
                  ! aff, afc and acf so we wouldn't want to destroy them below
                  if (air_data%options%strong_r_threshold == 0d0) then
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
                     air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP) = PETSC_NULL_MAT
                     air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP) = PETSC_NULL_MAT
                     air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP) = PETSC_NULL_MAT
#endif                     
                  end if                    
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
                  if (air_data%options%one_c_smooth .AND. &
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
                  if (.NOT. PetscMatIsNull(temp_mat)) then
                     call MatDestroy(air_data%reuse(our_level)%reuse_mat(i_loc), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
                     air_data%reuse(our_level)%reuse_mat(i_loc) = PETSC_NULL_MAT
#endif                     
                  end if
               end do

               do i_loc = 1, size(air_data%reuse(our_level)%reuse_is)
                  temp_is = air_data%reuse(our_level)%reuse_is(i_loc)
                  if (.NOT. PetscISIsNull(temp_is)) then
                     call ISDestroy(air_data%reuse(our_level)%reuse_is(i_loc), ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)                     
                     air_data%reuse(our_level)%reuse_is(i_loc) = PETSC_NULL_IS
#endif                     
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

      air_data%options%maxits_a_ff = 2
      air_data%options%one_c_smooth = .FALSE.
      air_data%options%matrix_free_polys = .FALSE.
      air_data%options%one_point_classical_prolong = .TRUE.
      air_data%options%full_smoothing_up_and_down = .FALSE.
      air_data%options%symmetric = .FALSE.
      air_data%options%constrain_w = .FALSE.
      air_data%options%constrain_z = .FALSE.       

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

end module air_mg_setup

