module air_operators_setup

   use petscksp
   use constrain_z_or_w
   use approx_inverse_setup
   use timers
   use fc_smooth
   use c_petsc_interfaces
   use grid_transfer
   use grid_transfer_improve

#include "petsc/finclude/petscksp.h"

   implicit none
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
         if (.NOT. PetscObjectIsNull(temp_mat)) then

            call remove_from_sparse_match(input_mat, air_data%reuse(our_level)%reuse_mat(MAT_A_DROP))     

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
         if (.NOT. PetscObjectIsNull(temp_mat)) then
            call MatCreateSubMatrixWrapper(air_data%reuse(our_level)%reuse_mat(MAT_A_DROP), &
                        air_data%IS_fine_index(our_level), air_data%IS_fine_index(our_level), MAT_REUSE_MATRIX, &
                        air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP), &
                        our_level = our_level, is_row_fine = .TRUE., is_col_fine = .TRUE.)              
         else
            call MatCreateSubMatrixWrapper(air_data%reuse(our_level)%reuse_mat(MAT_A_DROP), &
                        air_data%IS_fine_index(our_level), air_data%IS_fine_index(our_level), MAT_INITIAL_MATRIX, &
                        air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP), &
                        our_level = our_level, is_row_fine = .TRUE., is_col_fine = .TRUE.)                              
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
      if (.NOT. (.NOT. PetscObjectIsNull(temp_mat) .AND. &
                  air_data%options%reuse_poly_coeffs)) then
         call start_approximate_inverse(smoothing_mat, &
                  air_data%inv_A_ff_poly_data(our_level)%inverse_type, &
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
         if (.NOT. (.NOT. PetscObjectIsNull(temp_mat) .AND. &
                     air_data%options%reuse_poly_coeffs)) then
            call start_approximate_inverse(air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP), &
                  air_data%inv_A_ff_poly_data_dropped(our_level)%inverse_type, &
                  air_data%inv_A_ff_poly_data_dropped(our_level)%gmres_poly_order, &
                  air_data%inv_A_ff_poly_data_dropped(our_level)%buffers, &
                  air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients)          
         end if
      end if

      call timer_finish(TIMER_ID_AIR_INVERSE)
      
      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      ! If we are doing C point smoothing then we need to pull out Acc 
      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
      if (air_data%options%any_c_smooths .AND. .NOT. air_data%options%full_smoothing_up_and_down) then

         if (air_data%allocated_matrices_A_cc(our_level)) then
            call MatCreateSubMatrixWrapper(input_mat, &
                  air_data%IS_coarse_index(our_level), air_data%IS_coarse_index(our_level), MAT_REUSE_MATRIX, &
                  air_data%A_cc(our_level), &
                  our_level = our_level, is_row_fine = .FALSE., is_col_fine = .FALSE.)            
         else
            call MatCreateSubMatrixWrapper(input_mat, &
                  air_data%IS_coarse_index(our_level), air_data%IS_coarse_index(our_level), MAT_INITIAL_MATRIX, &
                  air_data%A_cc(our_level), &
                  our_level = our_level, is_row_fine = .FALSE., is_col_fine = .FALSE.)                
         end if

         call timer_start(TIMER_ID_AIR_INVERSE)    

         ! Compute the inverse for smoothing
         temp_mat = air_data%inv_A_cc(our_level)
         if (.NOT. (.NOT. PetscObjectIsNull(temp_mat) &
                     .AND. air_data%options%reuse_poly_coeffs)) then
            call start_approximate_inverse(air_data%A_cc(our_level), &
                  air_data%inv_A_cc_poly_data(our_level)%inverse_type, &
                  air_data%inv_A_cc_poly_data(our_level)%gmres_poly_order, &
                  air_data%inv_A_cc_poly_data(our_level)%buffers, &
                  air_data%inv_A_cc_poly_data(our_level)%coefficients)  
         end if

         call timer_finish(TIMER_ID_AIR_INVERSE)

      end if        
         
      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      ! Pull out the rest of the sub-matrices
      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
      call timer_start(TIMER_ID_AIR_EXTRACT)             
                        
      if (air_data%allocated_matrices_A_ff(our_level)) then
         call MatCreateSubMatrixWrapper(input_mat, &
               air_data%IS_fine_index(our_level), air_data%IS_coarse_index(our_level), MAT_REUSE_MATRIX, &
               air_data%A_fc(our_level), &
               our_level = our_level, is_row_fine = .TRUE., is_col_fine = .FALSE.)   
      call MatCreateSubMatrixWrapper(input_mat, &
               air_data%IS_coarse_index(our_level), air_data%IS_fine_index(our_level), MAT_REUSE_MATRIX, &
               air_data%A_cf(our_level), &
               our_level = our_level, is_row_fine = .FALSE., is_col_fine = .TRUE.)                      
      else 
         call MatCreateSubMatrixWrapper(input_mat, &
               air_data%IS_fine_index(our_level), air_data%IS_coarse_index(our_level), MAT_INITIAL_MATRIX, &
               air_data%A_fc(our_level), &
               our_level = our_level, is_row_fine = .TRUE., is_col_fine = .FALSE.) 
         call MatCreateSubMatrixWrapper(input_mat, &
               air_data%IS_coarse_index(our_level), air_data%IS_fine_index(our_level), MAT_INITIAL_MATRIX, &
               air_data%A_cf(our_level), &
               our_level = our_level, is_row_fine = .FALSE., is_col_fine = .TRUE.)                                                                    
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
         if (.NOT. PetscObjectIsNull(temp_mat)) then
            call MatCreateSubMatrixWrapper(air_data%reuse(our_level)%reuse_mat(MAT_A_DROP), &
                        air_data%IS_coarse_index(our_level), air_data%IS_fine_index(our_level), MAT_REUSE_MATRIX, &
                        air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), &
                        our_level = our_level, is_row_fine = .FALSE., is_col_fine = .TRUE.)               
         else
            call MatCreateSubMatrixWrapper(air_data%reuse(our_level)%reuse_mat(MAT_A_DROP), &
                        air_data%IS_coarse_index(our_level), air_data%IS_fine_index(our_level), MAT_INITIAL_MATRIX, &
                        air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), &
                        our_level = our_level, is_row_fine = .FALSE., is_col_fine = .TRUE.)             
         end if

         if (.NOT. air_data%options%one_point_classical_prolong) then
            ! Drop the entries from A_fc
            temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP)
            if (.NOT. PetscObjectIsNull(temp_mat)) then
               call MatCreateSubMatrixWrapper(air_data%reuse(our_level)%reuse_mat(MAT_A_DROP), &
                        air_data%IS_fine_index(our_level), air_data%IS_coarse_index(our_level), MAT_REUSE_MATRIX, &
                        air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP), &
                        our_level = our_level, is_row_fine = .TRUE., is_col_fine = .FALSE.)
            else
               call MatCreateSubMatrixWrapper(air_data%reuse(our_level)%reuse_mat(MAT_A_DROP), &
                        air_data%IS_fine_index(our_level), air_data%IS_coarse_index(our_level), MAT_INITIAL_MATRIX, &
                        air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP), &
                        our_level = our_level, is_row_fine = .TRUE., is_col_fine = .FALSE.)
            end if
         end if                  
         
         call timer_finish(TIMER_ID_AIR_EXTRACT)    
      end if               

      ! Delete temporary if not reusing
      if (.NOT. air_data%options%reuse_sparsity) then
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_A_DROP), ierr)
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
      type(tVec), dimension(:), intent(inout)               :: left_null_vecs_c, right_null_vecs_c

      PetscErrorCode :: ierr
      type(tMat) :: sparsity_mat_cf, A_ff_power, inv_dropped_Aff, smoothing_mat
      type(tMat) :: temp_mat
      type(tIS)  :: temp_is
      type(tVec) :: diag_vec
      type(tVec), dimension(:), allocatable   :: left_null_vecs_f, right_null_vecs_f
      integer :: comm_size, errorcode, order, i_loc
      MPI_Comm :: MPI_COMM_MATRIX
      integer(c_long_long) :: A_array, B_array, C_array
      PetscInt :: global_row_start, global_row_end_plus_one
      PetscInt, parameter :: nz_ignore = -1
      logical :: destroy_mat, reuse_grid_transfer
      MatType:: mat_type, mat_type_inv_aff

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
         if (.NOT. PetscObjectIsNull(temp_mat)) then

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
      call finish_approximate_inverse(smoothing_mat, &
            air_data%inv_A_ff_poly_data(our_level)%inverse_type, &
            air_data%inv_A_ff_poly_data(our_level)%gmres_poly_order, &
            air_data%inv_A_ff_poly_data(our_level)%gmres_poly_sparsity_order, &
            air_data%inv_A_ff_poly_data(our_level)%buffers, &
            air_data%inv_A_ff_poly_data(our_level)%coefficients, &
            air_data%options%matrix_free_polys, air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF), &
            air_data%inv_A_ff(our_level)) 
            
      ! Delete temporary if not reusing
      if (.NOT. air_data%options%reuse_sparsity) then
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF), ierr)
      end if              
      
      destroy_mat = .FALSE.
      ! If we are doing AIRG, then we also need an A_ff^-1 for the grid
      ! transfer operators (may be the same used for the smoother)
      if (air_data%options%z_type == AIR_Z_PRODUCT) then

         ! If we have applied a strong R tolerance or we are not doing fc smoothing
         ! then we have also started an inverse for dropped A_ff, let's finish it
         if (air_data%options%strong_r_threshold /= 0d0 .OR. air_data%options%full_smoothing_up_and_down) then

            ! Now we always build an assembled inv_A_ff here as we need it, hence the .false. 
            ! given to matrix_free_polys
            ! If we aren't doing matrix-free smooths, then we keep the assembled inv_A_ff
            call finish_approximate_inverse(air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP), &
                     air_data%inv_A_ff_poly_data_dropped(our_level)%inverse_type, &
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
               ! Making sure to give it the non dropped A_ff here
               call finish_approximate_inverse(air_data%A_ff(our_level), &
                        air_data%inv_A_ff_poly_data(our_level)%inverse_type, &
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
               end if               

            ! Just re-use the already assembled one            
            else
               inv_dropped_Aff = air_data%inv_A_ff(our_level)
            end if
         end if

         call MatGetType(inv_dropped_Aff, mat_type_inv_aff, ierr)
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
            ! If we know our inv_aff is diagonal we don't have to do a matmatmult
            ! It is just a row/column scaling
            if (mat_type_inv_aff == MATDIAGONAL) then
               if (.NOT. PetscObjectIsNull(temp_mat)) then
                  call MatCopy(air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP), &
                           air_data%reuse(our_level)%reuse_mat(MAT_W), &
                           SAME_NONZERO_PATTERN, ierr)
               else
                  call MatDuplicate(air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP), &
                           MAT_COPY_VALUES, &
                           air_data%reuse(our_level)%reuse_mat(MAT_W), ierr)          
               end if
               call MatCreateVecs(inv_dropped_Aff, PETSC_NULL_VEC, diag_vec, ierr)
               ! Should be able to call MatDiagonalGetDiagonal but it returns
               ! the wrong vector type with kokkos, even when it is set correctly
               ! when calling matcreatediagonal
               call MatGetDiagonal(inv_dropped_Aff, diag_vec, ierr)
               call VecScale(diag_vec, -1d0, ierr)
               ! Left multiply
               call MatDiagonalScale(air_data%reuse(our_level)%reuse_mat(MAT_W), &
                        diag_vec, PETSC_NULL_VEC, ierr)          
               call VecDestroy(diag_vec, ierr)
            else
               if (.NOT. PetscObjectIsNull(temp_mat)) then
                  call MatMatMult(inv_dropped_Aff, air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP), &
                           MAT_REUSE_MATRIX, 1.58d0, air_data%reuse(our_level)%reuse_mat(MAT_W), ierr)
               else
                  call MatMatMult(inv_dropped_Aff, air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP), &
                           MAT_INITIAL_MATRIX, 1.58d0, air_data%reuse(our_level)%reuse_mat(MAT_W), ierr)  
               end if
               call MatScale(air_data%reuse(our_level)%reuse_mat(MAT_W), -1d0, ierr)                       
            end if   

            ! ~~~~~~~~~
            ! Improve W if needed
            ! ~~~~~~~~~    

            call improve_w(air_data%reuse(our_level)%reuse_mat(MAT_W), &
                           air_data%A_ff(our_level), &
                           air_data%A_fc(our_level), &
                           air_data%inv_A_ff(our_level), &
                           air_data%reuse(our_level)%reuse_mat(MAT_W_AFF), &
                           air_data%reuse(our_level)%reuse_mat(MAT_W_NO_SPARSITY), &
                           air_data%options%improve_w_its, &
                           air_data%options%reuse_sparsity) 

            ! Delete temporaries if not reusing
            if (.NOT. air_data%options%reuse_sparsity) then   
               call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_W_AFF), ierr)
               call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_W_NO_SPARSITY), ierr)                           
            end if         
            
            ! ~~~~~~~~~~~~
            ! ~~~~~~~~~~~~            

            call timer_start(TIMER_ID_AIR_DROP)  

            ! If we want to reuse, we have to match the original sparsity
            ! which might be different if the matrix has changed (but the sparsity is the same)
            ! so we can't just drop with a drop tolerance
            temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_W_DROP)
            if (.NOT. PetscObjectIsNull(temp_mat)) then
   
               call remove_from_sparse_match(air_data%reuse(our_level)%reuse_mat(MAT_W), &
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
                  .NOT. PetscObjectIsNull(temp_is)) then

            ! Now when we're doing processor agglomeration, we have to be careful with 
            ! Kokkos, as it gets fussy
            ! about the exact same pointers being passed into spgemm_numeric
            ! (rather than just having the same sparsity) once we've 
            ! repartitioned. We have to force it to repartition to get around this
            ! so the exact same matrices are used in every case
            call MatGetType(air_data%coarse_matrix(our_level), mat_type, ierr)
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
            if (.NOT. PetscObjectIsNull(temp_mat)) then

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
         end if 
         if (air_data%options%lair_distance .ge. 2) then
            call MatDestroy(sparsity_mat_cf, ierr)        
         end if

      ! For AIRG - we do a matmatmult with our approximate A_ff inverse
      else         
         temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_Z)

         ! If we know our inv_aff is diagonal we don't have to do a matmatmult
         ! It is just a row/column scaling
         if (mat_type_inv_aff == MATDIAGONAL) then
            if (.NOT. PetscObjectIsNull(temp_mat)) then
               call MatCopy(air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), &
                        air_data%reuse(our_level)%reuse_mat(MAT_Z), &
                        SAME_NONZERO_PATTERN, ierr)
            else
               call MatDuplicate(air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), &
                        MAT_COPY_VALUES, &
                        air_data%reuse(our_level)%reuse_mat(MAT_Z), ierr)
            end if
            call MatCreateVecs(inv_dropped_Aff, PETSC_NULL_VEC, diag_vec, ierr)
            call MatGetDiagonal(inv_dropped_Aff, diag_vec, ierr)
            call VecScale(diag_vec, -1d0, ierr)
            ! Right multiply
            call MatDiagonalScale(air_data%reuse(our_level)%reuse_mat(MAT_Z), &
                     PETSC_NULL_VEC, diag_vec, ierr)          
            call VecDestroy(diag_vec, ierr)
         else         
            if (.NOT. PetscObjectIsNull(temp_mat)) then
               call MatMatMult(air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), inv_dropped_Aff, &
                     MAT_REUSE_MATRIX, 1.58d0, air_data%reuse(our_level)%reuse_mat(MAT_Z), ierr)            
            else
               call MatMatMult(air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), inv_dropped_Aff, &
                     MAT_INITIAL_MATRIX, 1.58d0, air_data%reuse(our_level)%reuse_mat(MAT_Z), ierr) 
            end if
            call MatScale(air_data%reuse(our_level)%reuse_mat(MAT_Z), -1d0, ierr)
         end if
      end if

      ! Destroy the copies, this only decrements the reference counter
      if (air_data%options%strong_r_threshold == 0d0) then
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP), ierr)      
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), ierr)
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP), ierr)                 
      end if

      ! Delete temporaries if not reusing
      if (.NOT. air_data%options%reuse_sparsity) then
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP), ierr)      
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), ierr)
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP), ierr)
      end if        

      ! ~~~~~~~~~
      ! Improve Z if needed
      ! ~~~~~~~~~

      call improve_z(air_data%reuse(our_level)%reuse_mat(MAT_Z), &
                     air_data%A_ff(our_level), &
                     air_data%A_cf(our_level), &
                     air_data%inv_A_ff(our_level), &
                     air_data%reuse(our_level)%reuse_mat(MAT_Z_AFF), &
                     air_data%reuse(our_level)%reuse_mat(MAT_Z_NO_SPARSITY), &
                     air_data%options%improve_z_its, &
                     air_data%options%reuse_sparsity)

      ! Delete temporaries if not reusing
      if (.NOT. air_data%options%reuse_sparsity) then   
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_Z_AFF), ierr)
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_Z_NO_SPARSITY), ierr)                           
      end if

      ! Delete temporary if not reusing
      if (.NOT. air_data%options%any_c_smooths .AND. .NOT. air_data%options%reuse_sparsity) then      
         call MatDestroy(air_data%A_cf(our_level), ierr)       
      end if      
      
      ! ~~~~~~~~~~~~
      ! ~~~~~~~~~~~~

      call timer_start(TIMER_ID_AIR_DROP)    

      ! If we want to reuse, we have to match the original sparsity
      ! which might be different if the matrix has changed (but the sparsity is the same)
      ! so we can't just drop with a drop tolerance
      temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_Z_DROP)
      if (.NOT. PetscObjectIsNull(temp_mat)) then

         call remove_from_sparse_match(air_data%reuse(our_level)%reuse_mat(MAT_Z), &
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
      end if       
      if (air_data%options%z_type == AIR_Z_PRODUCT .AND. destroy_mat) then
         call MatDestroy(inv_dropped_Aff, ierr)
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
      
      ! Delete temporary if not reusing
      if (.NOT. air_data%options%reuse_sparsity) then
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_Z_DROP), ierr)      
         call ISDestroy(air_data%reuse(our_level)%reuse_is(IS_R_Z_FINE_COLS), ierr)
      end if        
            
      ! Transpose the restrictor if needed
      if (air_data%options%symmetric) then
         call MatTransposeWrapper(air_data%restrictors(our_level), air_data%prolongators(our_level))
         call MatDestroy(air_data%restrictors(our_level), ierr)
      end if

      ! ~~~~~~~~~~~
      ! If we are doing C-point smoothing, finish the comms
      ! ~~~~~~~~~~~
      if (air_data%options%any_c_smooths .AND. &
               .NOT. air_data%options%full_smoothing_up_and_down) then      

         call timer_start(TIMER_ID_AIR_INVERSE)           
                  
         call finish_approximate_inverse(air_data%A_cc(our_level), &
                  air_data%inv_A_cc_poly_data(our_level)%inverse_type, &
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
         if (.NOT. PetscObjectIsNull(temp_mat)) then

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
         if (.NOT. PetscObjectIsNull(temp_mat)) then

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
         end if          
      end if

      call timer_finish(TIMER_ID_AIR_RAP)        
               
      ! Drop relative small entries         
      call timer_start(TIMER_ID_AIR_DROP)   

      ! If we want to reuse, we have to match the original sparsity
      ! which might be different if the matrix has changed (but the sparsity is the same)
      ! so we can't just drop with a drop tolerance
      temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_RAP_DROP)
      if (.NOT. PetscObjectIsNull(temp_mat)) then

         ! Duplicate the sparsity
         call MatDuplicate(air_data%reuse(our_level)%reuse_mat(MAT_RAP_DROP), &
                  MAT_DO_NOT_COPY_VALUES, coarse_matrix, ierr)

         call remove_from_sparse_match(air_data%reuse(our_level)%reuse_mat(MAT_RAP), &
                  coarse_matrix, &
                  lump=air_data%options%a_lump)

      ! First time so just drop according to a tolerance 
      else
         ! If we know we're not reusing save a copy
         if (.NOT. air_data%options%reuse_sparsity) then
            call remove_small_from_sparse(air_data%reuse(our_level)%reuse_mat(MAT_RAP), &
                     air_data%options%a_drop, coarse_matrix, &
                     relative_max_row_tol_int = 1, lump=air_data%options%a_lump)            
         else
            call remove_small_from_sparse(air_data%reuse(our_level)%reuse_mat(MAT_RAP), &
                     air_data%options%a_drop, air_data%reuse(our_level)%reuse_mat(MAT_RAP_DROP), &
                     relative_max_row_tol_int = 1, lump=air_data%options%a_lump)

            call MatDuplicate(air_data%reuse(our_level)%reuse_mat(MAT_RAP_DROP), &
                        MAT_COPY_VALUES, coarse_matrix, ierr)
         end if
      end if

      ! Delete temporary if not reusing
      if (.NOT. air_data%options%reuse_sparsity) then
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_RAP), ierr)
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_RAP_DROP), ierr)
      end if       

      call timer_finish(TIMER_ID_AIR_DROP)    

         
   end subroutine compute_coarse_matrix  

! -------------------------------------------------------------------------------------------------------------------------------

end module air_operators_setup

