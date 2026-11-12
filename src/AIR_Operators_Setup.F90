module air_operators_setup

   use petscksp
   use constrain_z_or_w, only: constrain_grid_transfer
   use approx_inverse_setup, only: &
         start_approximate_inverse, finish_approximate_inverse, destroy_matrix_reuse, &
         inverse_type_is_polynomial
   use pflare_parameters, only: &
         TIMER_ID_AIR_DROP, TIMER_ID_AIR_EXTRACT, TIMER_ID_AIR_INVERSE, &
         TIMER_ID_AIR_CONSTRAIN, TIMER_ID_AIR_PROLONG, TIMER_ID_AIR_RAP, &
         TIMER_ID_AIR_RESTRICT, &
         IS_REPARTITION, IS_R_Z_FINE_COLS, &
         MAT_ACF_DROP, MAT_AFC_DROP, MAT_AFF_DROP, MAT_AP, MAT_A_DROP, &
         MAT_INV_ACC, MAT_INV_AFF, MAT_INV_AFF_DROPPED, MAT_RAP, MAT_RAP_DROP, &
         MAT_SAI_SUB, MAT_W, MAT_W_AFF, MAT_W_DROP, MAT_W_NO_SPARSITY, &
         MAT_Z, MAT_Z_AFF, MAT_Z_DROP, MAT_Z_NO_SPARSITY, &
         AIR_Z_PRODUCT, AIR_Z_LAIR, &
         PFLARE_MINUS_ONE, PFLARE_PTAP_FILL
   use timers, only: timer_start, timer_finish
   use air_data_type, only: air_multigrid_data, REUSE_MAT_ACTIVE, REUSE_IS_ACTIVE
   use grid_transfer, only: generate_one_point_with_one_entry_from_sparse, &
         compute_P_from_W, compute_R_from_Z, generate_one_point_P_from_sparse, compute_R_from_A_diag, &
         compute_R_from_raw_Z
   use grid_transfer_improve, only: improve_w, improve_z
   use sai_z, only: calculate_and_build_sai_z
   use petsc_helper, only: MatCreateSubMatrixWrapper, remove_small_from_sparse, &
         remove_from_sparse_match, mat_mat_symbolic, mat_mult_one_entry_per_row, &
         mat_mult_r_one_point_ap, mat_mult_r_one_point_ap_raw, rap_raw_drop, mat_mult_raw, &
         mat_mult_mat_raw, raw_free
   use gmres_poly, only: build_gmres_polynomial_inverse_raw

#include "petsc/finclude/petscksp.h"

   implicit none
   public

   contains

!------------------------------------------------------------------------------------------------------------------------

   logical function r_from_a_diag_possible(input_mat, our_level, air_data)

      ! Can R be built straight from A on this level, without A_cf or Z
      ! (compute_R_from_A_diag) if the Aff inverse turns out to be a matdiagonal
      ! That needs a Kokkos matrix, a diagonal Aff, Z = -A_cf * Aff^-1 with A_cf itself
      ! and nothing else that needs Z or the dropped Z

      type(tMat), intent(in)                    :: input_mat
      integer, intent(in)                       :: our_level
      type(air_multigrid_data), intent(in)      :: air_data

      PetscErrorCode :: ierr
      MatType :: mat_type, mat_type_aff

      call MatGetType(input_mat, mat_type, ierr)
      call MatGetType(air_data%A_ff(our_level), mat_type_aff, ierr)
      r_from_a_diag_possible = (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
                  mat_type == MATAIJKOKKOS) .AND. &
               mat_type_aff == MATDIAGONAL .AND. &
               air_data%options%z_type == AIR_Z_PRODUCT .AND. &
               air_data%options%strong_r_threshold == 0d0 .AND. &
               .NOT. air_data%options%full_smoothing_up_and_down .AND. &
               .NOT. air_data%options%reuse_sparsity .AND. &
               .NOT. air_data%options%constrain_z .AND. &
               air_data%options%improve_z_its == 0

   end function r_from_a_diag_possible

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
                        our_level = our_level, is_row_fine = .TRUE., is_col_fine = .TRUE., &
                        kokkos_is_views_handle = air_data%kokkos_is_views_handle)              
         else
            call MatCreateSubMatrixWrapper(air_data%reuse(our_level)%reuse_mat(MAT_A_DROP), &
                        air_data%IS_fine_index(our_level), air_data%IS_fine_index(our_level), MAT_INITIAL_MATRIX, &
                        air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP), &
                        our_level = our_level, is_row_fine = .TRUE., is_col_fine = .TRUE., &
                        kokkos_is_views_handle = air_data%kokkos_is_views_handle)                              
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
      ! If we are re-using the polynomial coefficients and we have some then we don't have to do this
      if (.NOT. (air_data%inv_A_ff_poly_data(our_level)%coefficients_set .AND. &
                  air_data%options%reuse_poly_coeffs)) then
         call start_approximate_inverse(smoothing_mat, &
                  air_data%inv_A_ff_poly_data(our_level)%inverse_type, &
                  air_data%inv_A_ff_poly_data(our_level)%gmres_poly_order, &
                  air_data%options%diag_scale_polys, &
                  air_data%inv_A_ff_poly_data(our_level)%buffers, &
                  air_data%inv_A_ff_poly_data(our_level)%coefficients)        
         air_data%inv_A_ff_poly_data(our_level)%coefficients_set = &
                  inverse_type_is_polynomial(air_data%inv_A_ff_poly_data(our_level)%inverse_type)
      end if

      ! If we are doing AIRG
      ! then we need an A_ff^-1 that we use to build the grid-transfer operators
      ! but if the strong R threshold is zero we just use the one computed above
      if (air_data%options%z_type == AIR_Z_PRODUCT .AND. &
               (air_data%options%strong_r_threshold /= 0d0 .OR. &
                air_data%options%full_smoothing_up_and_down)) then

         ! The dropped A_ff has its own polynomial, so check its own coefficients
         if (.NOT. (air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients_set .AND. &
                     air_data%options%reuse_poly_coeffs)) then
            call start_approximate_inverse(air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP), &
                  air_data%inv_A_ff_poly_data_dropped(our_level)%inverse_type, &
                  air_data%inv_A_ff_poly_data_dropped(our_level)%gmres_poly_order, &
                  air_data%options%diag_scale_polys, &
                  air_data%inv_A_ff_poly_data_dropped(our_level)%buffers, &
                  air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients)          
            air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients_set = &
                  inverse_type_is_polynomial(air_data%inv_A_ff_poly_data_dropped(our_level)%inverse_type)
         end if
      end if

      call timer_finish(TIMER_ID_AIR_INVERSE)
      
      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      ! If we are doing C point smoothing then we need to pull out Acc 
      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
      if (air_data%options%any_c_smooths .AND. .NOT. air_data%options%full_smoothing_up_and_down) then

         ! Only reuse when coarse matrix structure is stable (amount>=2 stores MAT_RAP_DROP)
         if (air_data%allocated_matrices_A_cc(our_level) .AND. &
             air_data%options%reuse_sparsity .AND. &
             REUSE_MAT_ACTIVE(MAT_RAP_DROP, air_data%options%reuse_amount)) then
            call MatCreateSubMatrixWrapper(input_mat, &
                  air_data%IS_coarse_index(our_level), air_data%IS_coarse_index(our_level), MAT_REUSE_MATRIX, &
                  air_data%A_cc(our_level), &
                  our_level = our_level, is_row_fine = .FALSE., is_col_fine = .FALSE., &
                  kokkos_is_views_handle = air_data%kokkos_is_views_handle)
         else
            call MatCreateSubMatrixWrapper(input_mat, &
                  air_data%IS_coarse_index(our_level), air_data%IS_coarse_index(our_level), MAT_INITIAL_MATRIX, &
                  air_data%A_cc(our_level), &
                  our_level = our_level, is_row_fine = .FALSE., is_col_fine = .FALSE., &
                  kokkos_is_views_handle = air_data%kokkos_is_views_handle)
         end if

         call timer_start(TIMER_ID_AIR_INVERSE)    

         ! Compute the inverse for smoothing
         if (.NOT. (air_data%inv_A_cc_poly_data(our_level)%coefficients_set &
                     .AND. air_data%options%reuse_poly_coeffs)) then
            call start_approximate_inverse(air_data%A_cc(our_level), &
                  air_data%inv_A_cc_poly_data(our_level)%inverse_type, &
                  air_data%inv_A_cc_poly_data(our_level)%gmres_poly_order, &
                  air_data%options%diag_scale_polys, &
                  air_data%inv_A_cc_poly_data(our_level)%buffers, &
                  air_data%inv_A_cc_poly_data(our_level)%coefficients)  
            air_data%inv_A_cc_poly_data(our_level)%coefficients_set = &
                  inverse_type_is_polynomial(air_data%inv_A_cc_poly_data(our_level)%inverse_type)
         end if

         call timer_finish(TIMER_ID_AIR_INVERSE)

      end if        
         
      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      ! Pull out the rest of the sub-matrices
      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
      call timer_start(TIMER_ID_AIR_EXTRACT)             
                        
      ! Only reuse when coarse matrix structure is stable (amount>=2 stores MAT_RAP_DROP)
      if (air_data%allocated_matrices_A_ff(our_level) .AND. &
          air_data%options%reuse_sparsity .AND. &
          REUSE_MAT_ACTIVE(MAT_RAP_DROP, air_data%options%reuse_amount)) then
         call MatCreateSubMatrixWrapper(input_mat, &
               air_data%IS_fine_index(our_level), air_data%IS_coarse_index(our_level), MAT_REUSE_MATRIX, &
               air_data%A_fc(our_level), &
               our_level = our_level, is_row_fine = .TRUE., is_col_fine = .FALSE., &
               kokkos_is_views_handle = air_data%kokkos_is_views_handle)
         call MatCreateSubMatrixWrapper(input_mat, &
               air_data%IS_coarse_index(our_level), air_data%IS_fine_index(our_level), MAT_REUSE_MATRIX, &
               air_data%A_cf(our_level), &
               our_level = our_level, is_row_fine = .FALSE., is_col_fine = .TRUE., &
               kokkos_is_views_handle = air_data%kokkos_is_views_handle)
      else
         call MatCreateSubMatrixWrapper(input_mat, &
               air_data%IS_fine_index(our_level), air_data%IS_coarse_index(our_level), MAT_INITIAL_MATRIX, &
               air_data%A_fc(our_level), &
               our_level = our_level, is_row_fine = .TRUE., is_col_fine = .FALSE., &
               kokkos_is_views_handle = air_data%kokkos_is_views_handle)
         ! If R can be built straight from A, A_cf is only needed for C smoothing
         ! If the Aff inverse turns out not to be diagonal, A_cf is extracted when Z is built
         if (air_data%options%any_c_smooths .OR. &
               .NOT. r_from_a_diag_possible(input_mat, our_level, air_data)) then
            call MatCreateSubMatrixWrapper(input_mat, &
                  air_data%IS_coarse_index(our_level), air_data%IS_fine_index(our_level), MAT_INITIAL_MATRIX, &
                  air_data%A_cf(our_level), &
                  our_level = our_level, is_row_fine = .FALSE., is_col_fine = .TRUE., &
                  kokkos_is_views_handle = air_data%kokkos_is_views_handle)
         end if
      end if

      call timer_finish(TIMER_ID_AIR_EXTRACT)   

      ! ~~~~~~~~~~~~~~
      ! Apply the strong R threshold to A_cf
      ! ~~~~~~~~~~~~~~
      if (air_data%options%strong_r_threshold == 0d0) then

         ! Copy the original pointer (A_cf may not have been extracted)
         temp_mat = air_data%A_cf(our_level)
         if (.NOT. PetscObjectIsNull(temp_mat)) then
            air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP) = air_data%A_cf(our_level)
            ! Increase the reference counter
            call PetscObjectReference(air_data%A_cf(our_level), ierr)
         end if
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
                        our_level = our_level, is_row_fine = .FALSE., is_col_fine = .TRUE., &
                        kokkos_is_views_handle = air_data%kokkos_is_views_handle)               
         else
            call MatCreateSubMatrixWrapper(air_data%reuse(our_level)%reuse_mat(MAT_A_DROP), &
                        air_data%IS_coarse_index(our_level), air_data%IS_fine_index(our_level), MAT_INITIAL_MATRIX, &
                        air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), &
                        our_level = our_level, is_row_fine = .FALSE., is_col_fine = .TRUE., &
                        kokkos_is_views_handle = air_data%kokkos_is_views_handle)             
         end if

         if (.NOT. air_data%options%one_point_classical_prolong) then
            ! Drop the entries from A_fc
            temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP)
            if (.NOT. PetscObjectIsNull(temp_mat)) then
               call MatCreateSubMatrixWrapper(air_data%reuse(our_level)%reuse_mat(MAT_A_DROP), &
                        air_data%IS_fine_index(our_level), air_data%IS_coarse_index(our_level), MAT_REUSE_MATRIX, &
                        air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP), &
                        our_level = our_level, is_row_fine = .TRUE., is_col_fine = .FALSE., &
                        kokkos_is_views_handle = air_data%kokkos_is_views_handle)
            else
               call MatCreateSubMatrixWrapper(air_data%reuse(our_level)%reuse_mat(MAT_A_DROP), &
                        air_data%IS_fine_index(our_level), air_data%IS_coarse_index(our_level), MAT_INITIAL_MATRIX, &
                        air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP), &
                        our_level = our_level, is_row_fine = .TRUE., is_col_fine = .FALSE., &
                        kokkos_is_views_handle = air_data%kokkos_is_views_handle)
            end if
         end if                  
         
         call timer_finish(TIMER_ID_AIR_EXTRACT)    
      end if               

      ! Delete temporary if not reusing
      if (.NOT. air_data%options%reuse_sparsity .OR. &
          .NOT. REUSE_MAT_ACTIVE(MAT_A_DROP, air_data%options%reuse_amount)) then
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
      type(tVec), dimension(:), allocatable   :: left_null_vecs_f, right_null_vecs_f
      integer :: comm_size, errorcode, order, i_loc
      MPIU_Comm :: MPI_COMM_MATRIX
      type(tMat) :: C_mat
      PetscInt :: global_row_start, global_row_end_plus_one
      PetscInt, parameter :: nz_ignore = -1
      logical :: destroy_mat, reuse_grid_transfer, one_point_p_direct, fuse_z_drop, r_from_a_diag
      logical :: z_raw, z_raw_possible, inv_raw, inv_raw_possible
      integer(kind=8) :: z_raw_handle, inv_raw_handle
      MatType:: mat_type, mat_type_inv_aff

      ! ~~~~~~~~~~

      call PetscObjectGetComm(air_data%A_ff(our_level), MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)

      ! ~~~~~~~~~~~
      ! Get some sizes
      ! ~~~~~~~~~~~

      call MatGetOwnershipRange(A, global_row_start, global_row_end_plus_one, ierr)

      ! A one point classical prolongator can be built straight from A_fc without building
      ! W, as long as nothing needs W: no constraints on W and W isn't kept for reuse
      one_point_p_direct = air_data%options%one_point_classical_prolong .AND. &
               .NOT. air_data%options%symmetric .AND. &
               .NOT. air_data%options%constrain_w .AND. &
               .NOT. air_data%options%reuse_sparsity
      
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
            
         ! If nothing needs W, build P straight from A_fc without building W
         ! We don't reuse here, so the reset has already destroyed any old P
         else if (one_point_p_direct) then
            call generate_one_point_P_from_sparse(air_data%A_fc(our_level), &
                     global_row_start, &
                     air_data%IS_fine_index(our_level), air_data%IS_coarse_index(our_level), &
                     air_data%prolongators(our_level))
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
            air_data%options%matrix_free_polys, &
            air_data%options%diag_scale_polys, &
            air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF), &
            air_data%reuse(our_level)%reuse_submatrices(MAT_INV_AFF)%array, &
            air_data%inv_A_ff(our_level)) 
            
      ! Delete temporary if not reusing
      if (.NOT. air_data%options%reuse_sparsity .OR. &
          .NOT. REUSE_MAT_ACTIVE(MAT_INV_AFF, air_data%options%reuse_amount)) then
         ! Because inv_A_ff might have come from a gmres polynomial that had to use matcreatesubmatrices
         ! have to be careful about how we destroy it
         call destroy_matrix_reuse(air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF), &
                  air_data%reuse(our_level)%reuse_submatrices(MAT_INV_AFF)%array)
      end if              
      
      destroy_mat = .FALSE.
      inv_raw = .FALSE.
      inv_raw_handle = 0
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
                     .FALSE., &
                     air_data%options%diag_scale_polys, &
                     air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF_DROPPED), &
                     air_data%reuse(our_level)%reuse_submatrices(MAT_INV_AFF_DROPPED)%array, &
                     inv_dropped_Aff)
            destroy_mat = .TRUE.

            ! Delete temporary if not reusing
            if (.NOT. air_data%options%reuse_sparsity .OR. &
                .NOT. REUSE_MAT_ACTIVE(MAT_INV_AFF_DROPPED, air_data%options%reuse_amount)) then
               call destroy_matrix_reuse(air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF_DROPPED), &
                        air_data%reuse(our_level)%reuse_submatrices(MAT_INV_AFF_DROPPED)%array)
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
               ! If this inverse only feeds Z, and R is built from Z without Z being a matrix
               ! (see z_raw_possible below), it doesn't have to be a matrix either
               ! PetscObjectIsNull is a macro, so keep it on a short line
               temp_is = air_data%reuse(our_level)%reuse_is(IS_R_Z_FINE_COLS)
               inv_raw_possible = PetscObjectIsNull(temp_is)
               inv_raw_possible = inv_raw_possible .AND. air_data%options%one_point_classical_prolong .AND. &
                     .NOT. air_data%options%symmetric .AND. .NOT. air_data%options%reuse_sparsity .AND. &
                     .NOT. air_data%options%constrain_z .AND. air_data%options%improve_z_its == 0
               if (inv_raw_possible) then
                  call build_gmres_polynomial_inverse_raw(air_data%A_ff(our_level), &
                        air_data%inv_A_ff_poly_data(our_level)%inverse_type, &
                        air_data%inv_A_ff_poly_data(our_level)%gmres_poly_order, &
                        air_data%inv_A_ff_poly_data(our_level)%gmres_poly_sparsity_order, &
                        air_data%inv_A_ff_poly_data(our_level)%buffers, &
                        air_data%inv_A_ff_poly_data(our_level)%coefficients, &
                        air_data%options%diag_scale_polys, &
                        air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF_DROPPED), &
                        air_data%reuse(our_level)%reuse_submatrices(MAT_INV_AFF_DROPPED)%array, &
                        inv_raw_handle, inv_raw)
               end if
               ! Making sure to give it the non dropped A_ff here
               if (.NOT. inv_raw) then
                  call finish_approximate_inverse(air_data%A_ff(our_level), &
                        air_data%inv_A_ff_poly_data(our_level)%inverse_type, &
                        air_data%inv_A_ff_poly_data(our_level)%gmres_poly_order, &
                        air_data%inv_A_ff_poly_data(our_level)%gmres_poly_sparsity_order, &
                        air_data%inv_A_ff_poly_data(our_level)%buffers, &
                        air_data%inv_A_ff_poly_data(our_level)%coefficients, &
                        .FALSE., &
                        air_data%options%diag_scale_polys, &
                        air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF_DROPPED), &
                        air_data%reuse(our_level)%reuse_submatrices(MAT_INV_AFF_DROPPED)%array, &
                        inv_dropped_Aff)
               end if
               destroy_mat = .TRUE.

               ! Delete temporary if not reusing
               if (.NOT. air_data%options%reuse_sparsity .OR. &
                   .NOT. REUSE_MAT_ACTIVE(MAT_INV_AFF_DROPPED, air_data%options%reuse_amount)) then
                  call destroy_matrix_reuse(air_data%reuse(our_level)%reuse_mat(MAT_INV_AFF_DROPPED), &
                           air_data%reuse(our_level)%reuse_submatrices(MAT_INV_AFF_DROPPED)%array)
               end if

            ! Just re-use the already assembled one            
            else
               inv_dropped_Aff = air_data%inv_A_ff(our_level)
            end if
         end if

         ! A raw inverse is assembled and has the sparsity of A_ff, so it isn't diagonal
         if (inv_raw) then
            mat_type_inv_aff = MATAIJKOKKOS
         else
            call MatGetType(inv_dropped_Aff, mat_type_inv_aff, ierr)
         end if
      end if

      ! Can we build R straight from A, without A_cf or Z
      r_from_a_diag = .FALSE.
      z_raw = .FALSE.
      z_raw_handle = 0
      if (air_data%options%z_type == AIR_Z_PRODUCT) then
         r_from_a_diag = mat_type_inv_aff == MATDIAGONAL .AND. &
               r_from_a_diag_possible(A, our_level, air_data)
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
      ! Grid transfer (R, P) reuse requires A_ff/A_fc/A_cf to be preserved
      ! across setups (only amount=3).  At amount=2, P and R are rebuilt fresh
      ! from stored W_DROP/Z_DROP; SpGEMM reuse still works via stored AP/RAP.
      reuse_grid_transfer = air_data%allocated_matrices_A_ff(our_level) .AND. &
            air_data%options%reuse_sparsity

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
            if (.NOT. PetscObjectIsNull(temp_mat)) then
               call MatMatMult(inv_dropped_Aff, air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP), &
                        MAT_REUSE_MATRIX, PFLARE_PTAP_FILL, air_data%reuse(our_level)%reuse_mat(MAT_W), ierr)
            else
               call MatMatMult(inv_dropped_Aff, air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP), &
                        MAT_INITIAL_MATRIX, PFLARE_PTAP_FILL, air_data%reuse(our_level)%reuse_mat(MAT_W), ierr)  
            if (.NOT. air_data%options%reuse_sparsity .OR. &
                .NOT. REUSE_MAT_ACTIVE(MAT_W, air_data%options%reuse_amount)) &
               call MatProductClear(air_data%reuse(our_level)%reuse_mat(MAT_W), ierr)
            end if
            call MatScale(air_data%reuse(our_level)%reuse_mat(MAT_W), PFLARE_MINUS_ONE, ierr)                         

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
                           air_data%options%reuse_sparsity .AND. &
                              REUSE_MAT_ACTIVE(MAT_W_AFF, air_data%options%reuse_amount))

            ! Delete temporaries if not reusing
            if (.NOT. air_data%options%reuse_sparsity .OR. &
                .NOT. REUSE_MAT_ACTIVE(MAT_W_AFF, air_data%options%reuse_amount)) then
               call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_W_AFF), ierr)
            end if
            if (.NOT. air_data%options%reuse_sparsity .OR. &
                .NOT. REUSE_MAT_ACTIVE(MAT_W_NO_SPARSITY, air_data%options%reuse_amount)) then
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
            if (.NOT. air_data%options%reuse_sparsity .OR. &
                .NOT. REUSE_MAT_ACTIVE(MAT_W, air_data%options%reuse_amount)) then
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
         end if      

         ! Now we have W
         ! Build a copy of P with the identity block in it
         ! This is to save having to do Z A_ff W + A_cf W + Z A_fc + A_cc

         ! ~~~~~~~~~~~~~~~~~~~
         ! ~~~~~~~~~~~~~~~~~~~
         ! Calculate the prolongator
         ! ~~~~~~~~~~~~~~~~~~~
         ! ~~~~~~~~~~~~~~~~~~~
         ! One-point classical prolongator reuse requires A_ff preserved (amount=3 only).
         ! At amount=2, P is rebuilt from stored W_DROP which has stable sparsity.
         air_data%reuse_one_point_classical_prolong = air_data%options%one_point_classical_prolong .AND. &
               .NOT. air_data%options%symmetric .AND. &
               .NOT. air_data%options%constrain_w .AND. &
               air_data%allocated_matrices_A_ff(our_level) .AND. &
               air_data%options%reuse_sparsity

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
         ! (or we built it straight from A_fc above)
         if (.NOT. air_data%reuse_one_point_classical_prolong .AND. .NOT. one_point_p_direct) then
            call compute_P_from_W(air_data%reuse(our_level)%reuse_mat(MAT_W_DROP), &
                     global_row_start, &
                     air_data%IS_fine_index(our_level), air_data%IS_coarse_index(our_level), &
                     .TRUE., &
                     reuse_grid_transfer, &
                     air_data%prolongators(our_level))
         end if

         ! Delete temporary if not reusing
         if (.NOT. air_data%options%reuse_sparsity .OR. &
             .NOT. REUSE_MAT_ACTIVE(MAT_W_DROP, air_data%options%reuse_amount)) then
            call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_W_DROP), ierr)
         end if

         call timer_finish(TIMER_ID_AIR_PROLONG)   
         
      end if

      ! Destroy the F point constraints on W
      ! If symmetric we never build W (P = R^T) so constrain_w is ignored and 
      ! these are unused, but they still have to be destroyed
      if (air_data%options%constrain_w) then
         do i_loc = 1, size(right_null_vecs_f) 
            call VecDestroy(right_null_vecs_f(i_loc), ierr)       
         end do             
         deallocate(right_null_vecs_f)
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
                  call mat_mat_symbolic(air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP), A_ff_power, C_mat)
                  ! Don't delete the original power - ie A_ff
                  if (destroy_mat) call MatDestroy(A_ff_power, ierr)
                  A_ff_power = C_mat
                  destroy_mat = .TRUE.
      
               end do
      
               ! Call a symbolic mult as we don't need the values, just the resulting sparsity  
               ! A_cf * A_ff^(distance - 1)
               call mat_mat_symbolic(air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), A_ff_power, C_mat)
               if (destroy_mat) call MatDestroy(A_ff_power, ierr)

               sparsity_mat_cf = C_mat

            end if
         end if
   
         ! lAIR
         ! We delibrately have to give it A_ff and A_cf as input, not the dropped versions
         ! The sparsity is controlled by sparsity_mat_cf which has used the dropped versions
         if (air_data%options%z_type == AIR_Z_LAIR) then
            call calculate_and_build_sai_z(air_data%A_ff(our_level), air_data%A_cf(our_level), &
                        sparsity_mat_cf, .TRUE., &
                        air_data%reuse(our_level)%reuse_mat(MAT_SAI_SUB), &
                        air_data%reuse(our_level)%reuse_submatrices(MAT_SAI_SUB)%array, &
                        air_data%reuse(our_level)%reuse_mat(MAT_Z))
         ! SAI Z
         else
            call calculate_and_build_sai_z(air_data%A_ff(our_level), air_data%A_cf(our_level), &
                        sparsity_mat_cf, .FALSE., &
                        air_data%reuse(our_level)%reuse_mat(MAT_SAI_SUB), &
                        air_data%reuse(our_level)%reuse_submatrices(MAT_SAI_SUB)%array, &                        
                        air_data%reuse(our_level)%reuse_mat(MAT_Z))
         end if        
         ! Delete temporary if not reusing
         if (.NOT. air_data%options%reuse_sparsity .OR. &
             .NOT. REUSE_MAT_ACTIVE(MAT_SAI_SUB, air_data%options%reuse_amount)) then
            call destroy_matrix_reuse(air_data%reuse(our_level)%reuse_mat(MAT_SAI_SUB), &
                     air_data%reuse(our_level)%reuse_submatrices(MAT_SAI_SUB)%array)
         end if 
         if (air_data%options%lair_distance .ge. 2) then
            call MatDestroy(sparsity_mat_cf, ierr)        
         end if

      ! With a diagonal Aff inverse we may not need Z at all, R is built straight from A below
      else if (r_from_a_diag) then

      ! For AIRG - we do a matmatmult with our approximate A_ff inverse
      else         

         ! If A_cf wasn't extracted as we hoped to build R straight from A, extract it now
         temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP)
         if (PetscObjectIsNull(temp_mat)) then
            call timer_start(TIMER_ID_AIR_EXTRACT)
            call MatCreateSubMatrixWrapper(A, &
                  air_data%IS_coarse_index(our_level), air_data%IS_fine_index(our_level), MAT_INITIAL_MATRIX, &
                  air_data%A_cf(our_level), &
                  our_level = our_level, is_row_fine = .FALSE., is_col_fine = .TRUE., &
                  kokkos_is_views_handle = air_data%kokkos_is_views_handle)
            call timer_finish(TIMER_ID_AIR_EXTRACT)
            air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP) = air_data%A_cf(our_level)
            call PetscObjectReference(air_data%A_cf(our_level), ierr)
         end if

         temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_Z)       
         if (.NOT. PetscObjectIsNull(temp_mat)) then
            call MatMatMult(air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), inv_dropped_Aff, &
                  MAT_REUSE_MATRIX, PFLARE_PTAP_FILL, air_data%reuse(our_level)%reuse_mat(MAT_Z), ierr)            
         else
            ! If R is built from Z with the drop in one pass (see fuse_z_drop below) and nothing else
            ! needs Z, Z doesn't have to be a matrix
            ! PetscObjectIsNull is a macro, so keep it on a short line
            temp_is = air_data%reuse(our_level)%reuse_is(IS_R_Z_FINE_COLS)
            z_raw_possible = PetscObjectIsNull(temp_is)
            z_raw_possible = z_raw_possible .AND. .NOT. air_data%options%reuse_sparsity .AND. &
                  .NOT. air_data%options%constrain_z .AND. air_data%options%improve_z_its == 0
            if (inv_raw) then
               ! The raw inverse is only built when z_raw_possible holds
               call mat_mult_mat_raw(air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), inv_raw_handle, &
                        z_raw_handle)
               z_raw = .TRUE.
               call raw_free(inv_raw_handle)
               inv_raw = .FALSE.
            else if (z_raw_possible) then
               call mat_mult_raw(air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), inv_dropped_Aff, &
                        z_raw_handle, z_raw)
            end if
            if (.NOT. z_raw) then
               call MatMatMult(air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), inv_dropped_Aff, &
                     MAT_INITIAL_MATRIX, PFLARE_PTAP_FILL, air_data%reuse(our_level)%reuse_mat(MAT_Z), ierr)
               if (.NOT. air_data%options%reuse_sparsity .OR. &
                   .NOT. REUSE_MAT_ACTIVE(MAT_Z, air_data%options%reuse_amount)) &
                  call MatProductClear(air_data%reuse(our_level)%reuse_mat(MAT_Z), ierr)
            end if
         end if
         ! A raw Z is scaled when R is built from it
         if (.NOT. z_raw) call MatScale(air_data%reuse(our_level)%reuse_mat(MAT_Z), PFLARE_MINUS_ONE, ierr)
      end if

      ! Destroy the copies, this only decrements the reference counter
      if (air_data%options%strong_r_threshold == 0d0) then
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP), ierr)      
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), ierr)
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_AFC_DROP), ierr)                 
      end if

      ! Delete temporaries if not reusing
      if (.NOT. air_data%options%reuse_sparsity .OR. &
          .NOT. REUSE_MAT_ACTIVE(MAT_AFF_DROP, air_data%options%reuse_amount)) then
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_AFF_DROP), ierr)
      end if
      if (.NOT. air_data%options%reuse_sparsity .OR. &
          .NOT. REUSE_MAT_ACTIVE(MAT_ACF_DROP, air_data%options%reuse_amount)) then
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_ACF_DROP), ierr)
      end if
      if (.NOT. air_data%options%reuse_sparsity .OR. &
          .NOT. REUSE_MAT_ACTIVE(MAT_AFC_DROP, air_data%options%reuse_amount)) then
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
                     air_data%options%reuse_sparsity .AND. &
                        REUSE_MAT_ACTIVE(MAT_Z_AFF, air_data%options%reuse_amount))

      ! Delete temporaries if not reusing
      if (.NOT. air_data%options%reuse_sparsity .OR. &
          .NOT. REUSE_MAT_ACTIVE(MAT_Z_AFF, air_data%options%reuse_amount)) then
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_Z_AFF), ierr)
      end if
      if (.NOT. air_data%options%reuse_sparsity .OR. &
          .NOT. REUSE_MAT_ACTIVE(MAT_Z_NO_SPARSITY, air_data%options%reuse_amount)) then
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_Z_NO_SPARSITY), ierr)
      end if

      ! Delete temporary if not reusing
      if (.NOT. air_data%options%any_c_smooths .AND. .NOT. air_data%options%reuse_sparsity) then      
         call MatDestroy(air_data%A_cf(our_level), ierr)       
      end if      
      
      ! ~~~~~~~~~~~~
      ! ~~~~~~~~~~~~

      ! Without reuse or constraints nothing needs the dropped Z, so the drop is done
      ! while building R, without building the dropped Z as a matrix
      fuse_z_drop = .NOT. air_data%options%reuse_sparsity .AND. .NOT. air_data%options%constrain_z

      if (.NOT. fuse_z_drop) then

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
            ! If nothing is dropped MAT_Z_DROP can just be MAT_Z, as long as we
            ! aren't keeping either for reuse
            call remove_small_from_sparse(air_data%reuse(our_level)%reuse_mat(MAT_Z), &
                        air_data%options%r_drop, air_data%reuse(our_level)%reuse_mat(MAT_Z_DROP), &
                        relative_max_row_tol_int= 1, &
                        allow_alias = .NOT. air_data%options%reuse_sparsity .OR. &
                           (.NOT. REUSE_MAT_ACTIVE(MAT_Z, air_data%options%reuse_amount) .AND. &
                            .NOT. REUSE_MAT_ACTIVE(MAT_Z_DROP, air_data%options%reuse_amount)))
         end if

         call timer_finish(TIMER_ID_AIR_DROP)
         ! Delete temporary if not reusing
         if (.NOT. air_data%options%reuse_sparsity .OR. &
             .NOT. REUSE_MAT_ACTIVE(MAT_Z, air_data%options%reuse_amount)) then
            call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_Z), ierr)
         end if
      end if
      ! If we build R straight from A we still need the Aff inverse
      if (air_data%options%z_type == AIR_Z_PRODUCT .AND. destroy_mat .AND. .NOT. r_from_a_diag) then
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
      if (r_from_a_diag) then
         call compute_R_from_A_diag(A, inv_dropped_Aff, our_level, air_data%kokkos_is_views_handle, &
                  global_row_start, &
                  air_data%IS_fine_index(our_level), air_data%IS_coarse_index(our_level), &
                  air_data%options%r_drop, &
                  air_data%restrictors(our_level))
         if (destroy_mat) call MatDestroy(inv_dropped_Aff, ierr)
      else if (z_raw) then
         call compute_R_from_raw_Z(z_raw_handle, global_row_start, &
                  air_data%IS_fine_index(our_level), air_data%IS_coarse_index(our_level), &
                  air_data%reuse(our_level)%reuse_is(IS_R_Z_FINE_COLS), &
                  air_data%restrictors(our_level), &
                  air_data%options%r_drop)
      else if (fuse_z_drop) then
         call compute_R_from_Z(air_data%reuse(our_level)%reuse_mat(MAT_Z), global_row_start, &
                  air_data%IS_fine_index(our_level), air_data%IS_coarse_index(our_level), &
                  air_data%reuse(our_level)%reuse_is(IS_R_Z_FINE_COLS), &
                  .TRUE., &
                  reuse_grid_transfer, &
                  air_data%restrictors(our_level), &
                  drop_tol = air_data%options%r_drop)
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_Z), ierr)
      else
         call compute_R_from_Z(air_data%reuse(our_level)%reuse_mat(MAT_Z_DROP), global_row_start, &
                  air_data%IS_fine_index(our_level), air_data%IS_coarse_index(our_level), &
                  air_data%reuse(our_level)%reuse_is(IS_R_Z_FINE_COLS), &
                  .TRUE., &
                  reuse_grid_transfer, &
                  air_data%restrictors(our_level))
      end if

      call timer_finish(TIMER_ID_AIR_RESTRICT) 
      
      ! Delete temporaries if not reusing
      if (.NOT. air_data%options%reuse_sparsity .OR. &
          .NOT. REUSE_MAT_ACTIVE(MAT_Z_DROP, air_data%options%reuse_amount)) then
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_Z_DROP), ierr)
      end if
      if (.NOT. air_data%options%reuse_sparsity .OR. &
          .NOT. REUSE_IS_ACTIVE(IS_R_Z_FINE_COLS, air_data%options%reuse_amount)) then
         call ISDestroy(air_data%reuse(our_level)%reuse_is(IS_R_Z_FINE_COLS), ierr)
      end if        
            
      ! Transpose the restrictor if needed
      if (air_data%options%symmetric) then

         ! We always rebuild P rather than use MatTranspose with MAT_REUSE_MATRIX, as that 
         ! gives a stale P with mpiaijkokkos once R has been updated in place on the device
         call MatDestroy(air_data%prolongators(our_level), ierr)
         call MatTranspose(air_data%restrictors(our_level), MAT_INITIAL_MATRIX, air_data%prolongators(our_level), ierr)
         ! The stored PtAP product refers to the old P, so it can't be reused
         temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_RAP)
         if (.NOT. PetscObjectIsNull(temp_mat)) then
            call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_RAP), ierr)
         end if

         ! Keep R if we're reusing the grid transfer operators (only amount=3)
         ! as the reuse in compute_R_from_Z needs it
         if (.NOT. air_data%options%reuse_sparsity .OR. air_data%options%reuse_amount < 3) then
            call MatDestroy(air_data%restrictors(our_level), ierr)
         end if
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
                  air_data%options%diag_scale_polys, &
                  air_data%reuse(our_level)%reuse_mat(MAT_INV_ACC), &
                  air_data%reuse(our_level)%reuse_submatrices(MAT_INV_ACC)%array, &
                  air_data%inv_A_cc(our_level))
                  
         call timer_finish(TIMER_ID_AIR_INVERSE) 
         
         ! Delete temporary if not reusing
         if (.NOT. air_data%options%reuse_sparsity .OR. &
             .NOT. REUSE_MAT_ACTIVE(MAT_INV_ACC, air_data%options%reuse_amount)) then
            call destroy_matrix_reuse(air_data%reuse(our_level)%reuse_mat(MAT_INV_ACC), &
                     air_data%reuse(our_level)%reuse_submatrices(MAT_INV_ACC)%array)
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
      logical :: rap_done, rap_raw, rap_drop_reused
      integer(kind=8) :: rap_raw_handle

      ! ~~~~~~~~~~

      ! ~~~~~~~~~~~~~~~~~~~
      ! ~~~~~~~~~~~~~~~~~~~
      ! Now build our coarse grid matrix
      ! Can be computed from either RAP, or the smaller
      ! Z A_ff W + A_cf W + Z A_fc + A_cc
      ! ~~~~~~~~~~~~~~~~~~~
      ! ~~~~~~~~~~~~~~~~~~~
      rap_raw = .FALSE.
      rap_raw_handle = 0
      ! Is the dropped coarse matrix kept (or to be kept) for sparsity reuse
      temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_RAP_DROP)
      rap_drop_reused = .NOT. PetscObjectIsNull(temp_mat)
      if (air_data%options%reuse_sparsity) then
         if (REUSE_MAT_ACTIVE(MAT_RAP_DROP, air_data%options%reuse_amount)) rap_drop_reused = .TRUE.
      end if

      call timer_start(TIMER_ID_AIR_RAP)

      ! Can just do PtAP
      if (air_data%options%symmetric) then  

         ! If we've done this before we can reuse
         temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_RAP)
         if (.NOT. PetscObjectIsNull(temp_mat)) then

            call MatPtap(A, air_data%prolongators(our_level), &
                     MAT_REUSE_MATRIX, PFLARE_PTAP_FILL, air_data%reuse(our_level)%reuse_mat(MAT_RAP), ierr)             

         ! First time
         else

            call MatPtap(A, air_data%prolongators(our_level), &
                     MAT_INITIAL_MATRIX, PFLARE_PTAP_FILL, air_data%reuse(our_level)%reuse_mat(MAT_RAP), ierr)
            ! The product data references A and P, so clear it if we're not reusing
            ! MAT_RAP may become the coarse matrix if nothing is dropped
            if (.NOT. air_data%options%reuse_sparsity .OR. &
                .NOT. REUSE_MAT_ACTIVE(MAT_RAP, air_data%options%reuse_amount)) &
               call MatProductClear(air_data%reuse(our_level)%reuse_mat(MAT_RAP), ierr)
         end if

      ! ~~~~~~~~~~~
      ! Do two matmatmults rather than the triple product
      ! ~~~~~~~~~~~            
      else

         ! If we've done this before we can reuse
         temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_AP)
         if (.NOT. PetscObjectIsNull(temp_mat)) then

            call MatMatMult(A, air_data%prolongators(our_level), &
                     MAT_REUSE_MATRIX, PFLARE_PTAP_FILL, air_data%reuse(our_level)%reuse_mat(MAT_AP), ierr)     
                     
            call MatMatMult(air_data%restrictors(our_level), air_data%reuse(our_level)%reuse_mat(MAT_AP), &
                     MAT_REUSE_MATRIX, PFLARE_PTAP_FILL, air_data%reuse(our_level)%reuse_mat(MAT_RAP), ierr)             
         
         ! First time
         else

            ! With Kokkos and a one point P, R * (A * P) can be computed without making A * P a
            ! matrix, as long as neither MAT_AP nor MAT_RAP has to be a product for reuse
            rap_done = .FALSE.
            if (air_data%options%one_point_classical_prolong .AND. &
                  (.NOT. air_data%options%reuse_sparsity .OR. &
                   (.NOT. REUSE_MAT_ACTIVE(MAT_AP, air_data%options%reuse_amount) .AND. &
                    .NOT. REUSE_MAT_ACTIVE(MAT_RAP, air_data%options%reuse_amount)))) then
               ! If the coarse matrix is just dropped from it (not kept for reuse), R * (A * P) doesn't
               ! have to be a matrix either
               if (.NOT. rap_drop_reused) then
                  call mat_mult_r_one_point_ap_raw(air_data%restrictors(our_level), A, &
                           air_data%prolongators(our_level), rap_raw_handle, rap_done)
                  rap_raw = rap_done
               else
                  call mat_mult_r_one_point_ap(air_data%restrictors(our_level), A, air_data%prolongators(our_level), &
                           air_data%reuse(our_level)%reuse_mat(MAT_RAP), rap_done)
               end if
            end if

            if (.NOT. rap_done) then
               ! A one point P has at most one entry per row, so A*P is a relabelling of the
               ! columns of A, as long as we don't need MAT_AP to be a product for reuse
               if (air_data%options%one_point_classical_prolong .AND. &
                     (.NOT. air_data%options%reuse_sparsity .OR. &
                      .NOT. REUSE_MAT_ACTIVE(MAT_AP, air_data%options%reuse_amount))) then
                  call mat_mult_one_entry_per_row(A, air_data%prolongators(our_level), &
                           air_data%reuse(our_level)%reuse_mat(MAT_AP))
               else
                  call MatMatMult(A, air_data%prolongators(our_level), &
                           MAT_INITIAL_MATRIX, PFLARE_PTAP_FILL, air_data%reuse(our_level)%reuse_mat(MAT_AP), ierr)
                  if (.NOT. air_data%options%reuse_sparsity .OR. &
                      .NOT. REUSE_MAT_ACTIVE(MAT_AP, air_data%options%reuse_amount)) &
                     call MatProductClear(air_data%reuse(our_level)%reuse_mat(MAT_AP), ierr)
               end if

               call MatMatMult(air_data%restrictors(our_level), air_data%reuse(our_level)%reuse_mat(MAT_AP), &
                        MAT_INITIAL_MATRIX, PFLARE_PTAP_FILL, air_data%reuse(our_level)%reuse_mat(MAT_RAP), ierr)
               if (.NOT. air_data%options%reuse_sparsity .OR. &
                   .NOT. REUSE_MAT_ACTIVE(MAT_RAP, air_data%options%reuse_amount)) &
                  call MatProductClear(air_data%reuse(our_level)%reuse_mat(MAT_RAP), ierr)
            end if
         end if
         
         ! Delete temporary if not reusing
         if (.NOT. air_data%options%reuse_sparsity .OR. &
             .NOT. REUSE_MAT_ACTIVE(MAT_AP, air_data%options%reuse_amount)) then
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
      ! R * (A * P) was left as device data, drop straight from it
      ! This is the same as the remove_small_from_sparse with allow_alias below
      if (rap_raw) then

         call rap_raw_drop(rap_raw_handle, air_data%restrictors(our_level), A, air_data%prolongators(our_level), &
                  air_data%options%a_drop, coarse_matrix, air_data%options%a_lump)

      else if (.NOT. PetscObjectIsNull(temp_mat)) then

         ! Duplicate the sparsity
         call MatDuplicate(air_data%reuse(our_level)%reuse_mat(MAT_RAP_DROP), &
                  MAT_DO_NOT_COPY_VALUES, coarse_matrix, ierr)

         call remove_from_sparse_match(air_data%reuse(our_level)%reuse_mat(MAT_RAP), &
                  coarse_matrix, &
                  lump=air_data%options%a_lump)

      ! First time so just drop according to a tolerance
      else
         ! If we know we're not reusing MAT_RAP_DROP, compute directly into coarse_matrix
         if (.NOT. air_data%options%reuse_sparsity .OR. &
             .NOT. REUSE_MAT_ACTIVE(MAT_RAP_DROP, air_data%options%reuse_amount)) then
            ! If nothing is dropped the coarse matrix can just be MAT_RAP, as long as
            ! we aren't keeping MAT_RAP for reuse
            call remove_small_from_sparse(air_data%reuse(our_level)%reuse_mat(MAT_RAP), &
                     air_data%options%a_drop, coarse_matrix, &
                     relative_max_row_tol_int = 1, lump=air_data%options%a_lump, &
                     allow_alias = .NOT. air_data%options%reuse_sparsity .OR. &
                        .NOT. REUSE_MAT_ACTIVE(MAT_RAP, air_data%options%reuse_amount))
         else
            call remove_small_from_sparse(air_data%reuse(our_level)%reuse_mat(MAT_RAP), &
                     air_data%options%a_drop, air_data%reuse(our_level)%reuse_mat(MAT_RAP_DROP), &
                     relative_max_row_tol_int = 1, lump=air_data%options%a_lump)

            call MatDuplicate(air_data%reuse(our_level)%reuse_mat(MAT_RAP_DROP), &
                        MAT_COPY_VALUES, coarse_matrix, ierr)
         end if
      end if

      ! Delete temporaries if not reusing
      if (.NOT. air_data%options%reuse_sparsity .OR. &
          .NOT. REUSE_MAT_ACTIVE(MAT_RAP, air_data%options%reuse_amount)) then
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_RAP), ierr)
      end if
      if (.NOT. air_data%options%reuse_sparsity .OR. &
          .NOT. REUSE_MAT_ACTIVE(MAT_RAP_DROP, air_data%options%reuse_amount)) then
         call MatDestroy(air_data%reuse(our_level)%reuse_mat(MAT_RAP_DROP), ierr)
      end if       

      call timer_finish(TIMER_ID_AIR_DROP)    

         
   end subroutine compute_coarse_matrix  

! -------------------------------------------------------------------------------------------------------------------------------

end module air_operators_setup

