module air_mg_setup

   use petscksp
   use constrain_z_or_w
   use cf_splitting
   use matshell_data_type
   use approx_inverse_setup
   use timers
   use air_mg_stats
   use fc_smooth
   use c_petsc_interfaces
   use grid_transfer
   use air_operators_setup

#include "petsc/finclude/petscksp.h"

   implicit none
   public

   contains
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine setup_air_pcmg(amat, pmat, air_data, pcmg_input)

      ! Setup AIR by computing the hierarchy and returning a PETSC PCMG object 
      ! Have to have called create_air_data before this routine
      ! The options for AIR are stored in air_data%options

      ! ~~~~~~
      type(tMat), target, intent(in)                     :: amat, pmat
      type(air_multigrid_data), target, intent(inout)    :: air_data
      type(tPC), intent(inout)                           :: pcmg_input

      ! Local
      PetscInt            :: local_rows, local_cols, global_rows, global_cols
      PetscInt            :: local_fine_is_size, local_coarse_is_size
      PetscInt            :: global_coarse_is_size, global_fine_is_size, global_row_start
      PetscInt            :: global_row_end_plus_one, no_active_cores
      PetscInt            :: prolongator_start, prolongator_end_plus_one, proc_stride
      PetscInt            :: petsc_level, no_levels_petsc_int
      PetscInt            :: local_vec_size, ystart, yend, local_rows_repart, local_cols_repart
      PetscInt            :: global_rows_repart, global_cols_repart
      integer             :: i_loc, inverse_type_aff, inverse_sparsity_aff
      integer             :: no_levels, our_level, our_level_coarse, errorcode, comm_rank, comm_size
      PetscErrorCode      :: ierr
      MPI_Comm            :: MPI_COMM_MATRIX
      PetscReal           :: ratio_local_nnzs_off_proc, achieved_rel_tol, norm_b
      logical             :: continue_coarsening, trigger_proc_agglom
      type(tMat)          :: temp_mat
      type(tKSP)          :: ksp_smoother_up, ksp_smoother_down, ksp_coarse_solver
      type(tPC)           :: pc_smoother_up, pc_smoother_down, pc_coarse_solver
      type(tVec)          :: temp_coarse_vec, rand_vec, sol_vec, temp_vec, diag_vec, diag_vec_aff
      type(tIS)           :: is_unchanged, is_full, temp_is
      type(mat_ctxtype), pointer :: mat_ctx
      PetscInt, parameter :: one=1, zero=0
      type(tVec), dimension(:), allocatable :: left_null_vecs, right_null_vecs
      type(tVec), dimension(:), allocatable :: left_null_vecs_c, right_null_vecs_c
      VecScatter :: vec_scatter
      VecType :: vec_type
      logical :: auto_truncated, aff_diag, check_diag_only
      PetscRandom :: rctx
      MatType:: mat_type, mat_type_aff
      integer(c_int) :: diag_only

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
                  air_data%inv_coarsest_poly_data%inverse_type, &
                  air_data%inv_coarsest_poly_data%gmres_poly_order, &
                  air_data%inv_coarsest_poly_data%buffers, &
                  air_data%inv_coarsest_poly_data%coefficients)                       

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
               auto_truncated = .TRUE.

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
                  air_data%options%ddc_its, &
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
            end if

            if (air_data%options%constrain_z) then
               ! Destroy our copy of the left near nullspace vectors
               do i_loc = 1, size(left_null_vecs)
                  call VecDestroy(left_null_vecs(i_loc), ierr)
               end do
            end if
            if (allocated(left_null_vecs)) deallocate(left_null_vecs)
            if (allocated(left_null_vecs_c)) deallocate(left_null_vecs_c)            
            if (air_data%options%constrain_w) then
               ! Destroy our copy of the right near nullspace vectors
               do i_loc = 1, size(right_null_vecs)
                  call VecDestroy(right_null_vecs(i_loc), ierr)
               end do
            end if   
            if (allocated(right_null_vecs)) deallocate(right_null_vecs)
            if (allocated(right_null_vecs_c)) deallocate(right_null_vecs_c)                     

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

         ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         ! We need to pull out Aff before we do anything
         ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~         

         if (.NOT. allocated(air_data%smooth_order_levels(our_level)%array)) then
            allocate(air_data%smooth_order_levels(our_level)%array(size(air_data%options%smooth_order)))
         end if
         air_data%smooth_order_levels(our_level)%array = air_data%options%smooth_order

         call timer_start(TIMER_ID_AIR_EXTRACT)             

         ! Pull out A_ff
         if (air_data%allocated_matrices_A_ff(our_level)) then
            call MatGetType(air_data%A_ff(our_level), mat_type_aff, ierr)

            ! If our Aff was previously converted to a matdiagonal, can't call matcreatesubmatrix 
            ! with MAT_REUSE_MATRIX
            if (mat_type_aff == MATDIAGONAL) then

               ! Easy to get out Aff if we know its diagonal
               call MatDiagonalGetDiagonal(air_data%A_ff(our_level), diag_vec_aff, ierr)
               call MatCreateVecs(air_data%coarse_matrix(our_level), diag_vec, PETSC_NULL_VEC, ierr)
               ! Get the matrix diagonal
               call MatGetDiagonal(air_data%coarse_matrix(our_level), diag_vec, ierr)
               ! Pull out the F points in the diagonal - this happens on the host
               ! if this becomes expensive could call something like our version of VecISCopyLocal_kokkos
               call VecISCopy(diag_vec, air_data%IS_fine_index(our_level), &
                        SCATTER_REVERSE, diag_vec_aff, ierr)
               call MatDiagonalRestoreDiagonal(air_data%A_ff(our_level), diag_vec_aff, ierr)
               call VecDestroy(diag_vec, ierr)                             

            ! If its not matdiagonal we can do reuse as normal
            else
               call MatCreateSubMatrix(air_data%coarse_matrix(our_level), &
                     air_data%IS_fine_index(our_level), air_data%IS_fine_index(our_level), MAT_REUSE_MATRIX, &
                     air_data%A_ff(our_level), ierr)         
            end if
         else
            call MatCreateSubMatrix(air_data%coarse_matrix(our_level), &
                  air_data%IS_fine_index(our_level), air_data%IS_fine_index(our_level), MAT_INITIAL_MATRIX, &
                  air_data%A_ff(our_level), ierr)
         end if
                  
         call timer_finish(TIMER_ID_AIR_EXTRACT)   
         
         ! ~~~~~~~~~
         ! Check if Aff is purely diagonal
         ! ~~~~~~~~~                 
         inverse_type_aff = air_data%options%inverse_type
         inverse_sparsity_aff = air_data%options%inverse_sparsity_order
         aff_diag = .FALSE.
         check_diag_only = .TRUE.
         ! Don't have to check if we have strong threshold of zero 
         ! or its already matdiagonal due to reuse
         call MatGetType(air_data%A_ff(our_level), mat_type_aff, ierr)
         if (mat_type_aff == MATDIAGONAL .OR. air_data%options%strong_threshold == 0d0) then
            check_diag_only = .FALSE.
         end if

         ! Check if Aff is only a diagonal
         if (check_diag_only) then      

            call MatGetDiagonalOnly_c(air_data%A_ff(our_level)%v, diag_only)
            ! If Aff is diagonal we can exploit this                
            if (diag_only == 1) then
               aff_diag = .TRUE.
            end if
         else
            aff_diag = .TRUE.
         end if

         ! Convert Aff to a matdiagonal type
         ! Haven't rewritten sai to take advantage of matdiagonal
         if (aff_diag .AND. &
                  inverse_type_aff /= PFLAREINV_SAI .AND. &
                  inverse_type_aff /= PFLAREINV_ISAI) then

            ! We've already updated Aff above if we are reusing and it is matdiagonal already
            if (.NOT. air_data%allocated_matrices_A_ff(our_level)) then
               call MatCreateVecs(air_data%A_ff(our_level), diag_vec_aff, PETSC_NULL_VEC, ierr)
               call MatGetDiagonal(air_data%A_ff(our_level), diag_vec_aff, ierr)               

               call MatDestroy(air_data%A_ff(our_level), ierr)
               ! The matrix takes ownership of diag_vec_aff and increases ref counter
               call MatCreateDiagonal(diag_vec_aff, air_data%A_ff(our_level), ierr)     
               call VecDestroy(diag_vec_aff, ierr)
            end if

            ! Use an exact inverse 
            !inverse_type_aff = PFLAREINV_JACOBI
            ! if (air_data%options%print_stats_timings .AND. comm_rank == 0) then
            !    print *, "Detected diagonal Aff - using exact inverse"
            ! end if   

            ! If diagonal we know the sparsity is "0th" order
            inverse_sparsity_aff = 0         
            ! Our approximation of diagonals is often an exact inverse
            ! So set the number of F smooths to 1
            if (inverse_type_aff /= PFLAREINV_WJACOBI .AND. &
                  air_data%options%poly_order > 2) then

               ! Any F smooths we just make 1 iteration
               do i_loc = 1, size(air_data%options%smooth_order)
                  if (air_data%smooth_order_levels(our_level)%array(i_loc) > 0) then
                     air_data%smooth_order_levels(our_level)%array(i_loc) = 1
                  end if
               end do
                     
               if (air_data%options%print_stats_timings .AND. comm_rank == 0) then
                  print *, "Detected diagonal Aff - setting any F smooth to 1 iteration on this level"
               end if                 
            end if
         end if             

         ! ~~~~~~~~~
         ! Setup the details of our gmres polynomials
         ! ~~~~~~~~~         

         call setup_gmres_poly_data(global_fine_is_size, &
                  inverse_type_aff, &
                  air_data%options%poly_order, &
                  inverse_sparsity_aff, &
                  air_data%options%subcomm, &
                  proc_stride, &
                  air_data%inv_A_ff_poly_data(our_level))

         ! Setup the same structure for the inv_A_ff made from dropped Aff 
         call setup_gmres_poly_data(global_fine_is_size, &
                  inverse_type_aff, &
                  air_data%options%poly_order, &
                  inverse_sparsity_aff, &
                  air_data%options%subcomm, &
                  proc_stride, &
                  air_data%inv_A_ff_poly_data_dropped(our_level)) 

         ! ~~~~~~~~~
         ! If we're doing C-point smoothing we may have a gmres polynomial on C points
         ! ~~~~~~~~~
         if (air_data%options%any_c_smooths .AND. &
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
               
         ! ~~~~~~~~~
         ! Finish the non-blocking comms and build the approximate inverse, then the 
         ! restrictor and prolongator
         ! ~~~~~~~~~        
               
         if (.NOT. allocated(left_null_vecs_c)) allocate(left_null_vecs_c(size(left_null_vecs)))
         if (.NOT. allocated(right_null_vecs_c)) allocate(right_null_vecs_c(size(right_null_vecs)))

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

         air_data%allocated_coarse_matrix(our_level_coarse) = .TRUE.                  

         ! ~~~~~~~~~~~
         ! We may be able to destroy the coarse matrix on our_level from here
         ! If so we build a shell as a placeholder
         ! ~~~~~~~~~~~
         
         ! Get the nnzs for these matrices here, in case we destroy them below
         if (air_data%options%print_stats_timings) then
            call get_nnzs_petsc_sparse(air_data%coarse_matrix(our_level), &
                     air_data%coarse_matrix_nnzs(our_level))
         end if

         ! On every level but the top we can destroy the full operator matrix
         if (our_level /= 1) then
            if (.NOT. air_data%options%full_smoothing_up_and_down) then
               call MatDestroy(air_data%coarse_matrix(our_level), ierr)
            end if
         end if         
         
         ! If we are just doing F point smoothing, we no longer have our coarse matrix
         ! But we use the mat_ctx in our F-point smoother to tell what level 
         ! we're on, so let's just create an empty matshell to pass in that has the right sizes           
         if (.NOT. air_data%options%full_smoothing_up_and_down) then
            
            allocate(mat_ctx)
            mat_ctx%our_level = our_level
            mat_ctx%air_data => air_data  

            call MatCreateShell(MPI_COMM_MATRIX, local_rows, local_cols, global_rows, global_cols, &
                        mat_ctx, air_data%coarse_matrix(our_level), ierr)
            call MatAssemblyBegin(air_data%coarse_matrix(our_level), MAT_FINAL_ASSEMBLY, ierr)
            call MatAssemblyEnd(air_data%coarse_matrix(our_level), MAT_FINAL_ASSEMBLY, ierr)   
            
            ! Have to make sure to set the type of vectors the shell creates
            ! Input can be any matrix, we just need the correct type
            call ShellSetVecType(air_data%A_fc(our_level), air_data%coarse_matrix(our_level))                   
         end if         

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
               trigger_proc_agglom = .NOT. PetscObjectIsNull(temp_is) .OR. &
                           (global_rows_repart/no_active_cores < air_data%options%process_eq_limit &
                              .AND. no_active_cores /= 1) .OR. &
                           (ratio_local_nnzs_off_proc .le. air_data%options%processor_agglom_ratio &
                           .AND. ratio_local_nnzs_off_proc /= 0d0)
                                 
               ! Start the process agglomeration 
               if (trigger_proc_agglom) then

                  call timer_start(TIMER_ID_AIR_PROC_AGGLOM)
                  ! air_data%options%processor_agglom_factor tells us how much we're reducing the number 
                  ! of active mpi ranks by each time
                  
                  ! can tell us how many idle threads we have on lower grids
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
                  if (PetscObjectIsNull(temp_is)) then

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
                  if (.NOT. PetscObjectIsNull(temp_mat)) then

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
                  if (.NOT. PetscObjectIsNull(temp_mat)) then

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
                  end if                   
                  
                  ! ~~~~~~~~~~~~~~~~~~
                  ! If need to repartition a restrictor
                  ! ~~~~~~~~~~~~~~~~~~
                  if (.NOT. air_data%options%symmetric) then

                     ! Repartition the restrictor matrix
                     temp_mat = air_data%reuse(our_level)%reuse_mat(MAT_R_REPARTITIONED)
                     if (.NOT. PetscObjectIsNull(temp_mat)) then

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
                  end if                  

                  call timer_finish(TIMER_ID_AIR_PROC_AGGLOM)

               end if             
            end if
         end if

         ! ~~~~~~~~~~~~~~

         if (air_data%options%any_c_smooths .AND. &
                  .NOT. air_data%options%full_smoothing_up_and_down) then          
            air_data%allocated_matrices_A_cc(our_level) = .TRUE.
         end if         

         ! ~~~~~~~~~~~~         

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
      if (.NOT. (.NOT. PetscObjectIsNull(temp_mat) &
                  .AND. air_data%options%reuse_poly_coeffs)) then

         ! We've already created our coarse solver if we've auto truncated         
         if (.NOT. auto_truncated) then
            call start_approximate_inverse(air_data%coarse_matrix(no_levels), &
                  air_data%inv_coarsest_poly_data%inverse_type, &
                  air_data%inv_coarsest_poly_data%gmres_poly_order, &
                  air_data%inv_coarsest_poly_data%buffers, &
                  air_data%inv_coarsest_poly_data%coefficients)                   
         end if
      end if
      call timer_finish(TIMER_ID_AIR_INVERSE)  

      ! ~~~~~~~~~~~~~~~
      ! Let's setup the PETSc pc we need
      ! ~~~~~~~~~~~~~~~
      if (no_levels > 1) then

         call PCSetOperators(pcmg_input, amat, pmat, ierr)
         call PCSetType(pcmg_input, PCMG, ierr)
         no_levels_petsc_int = no_levels
         call PCMGSetLevels(pcmg_input, no_levels_petsc_int, PETSC_NULL_MPI_COMM, ierr)  

         ! If we're doing fc smoothing always have zero down smooths, which petsc calls kaskade
         ! This prevents any unnecessary residual calculations on the way down
         ! Annoyingly kaskade calls the "down" smoother on the way up, so 
         ! we have to set the options carefully for the down smoother
         if (.NOT. air_data%options%full_smoothing_up_and_down) then
            call PCMGSetType(pcmg_input, PC_MG_KASKADE, ierr)
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
               call PCMGSetRestriction(pcmg_input, petsc_level, air_data%restrictors(our_level), ierr)
            end if
            call PCMGSetInterpolation(pcmg_input, petsc_level, air_data%prolongators(our_level), ierr)

            ! Get smoother for this level
            ! The up smoother is never used or called when doing kaskade, but we set it as a richardson so that petsc doesn't default
            ! to chebychev and hence try and calculate eigenvalues on each grid
            call PCMGGetSmootherUp(pcmg_input, petsc_level, ksp_smoother_up, ierr)
            call PCMGGetSmootherDown(pcmg_input, petsc_level, ksp_smoother_down, ierr)                                
            
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

         call PCMGGetCoarseSolve(pcmg_input, ksp_coarse_solver, ierr)
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
                  air_data%inv_coarsest_poly_data%inverse_type, &
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
            call PCSetOperators(pcmg_input, amat, &
                        air_data%inv_A_ff(no_levels), ierr)         
            call PCSetType(pcmg_input, PCMAT, ierr)

         ! Otherwise just do a jacobi and tell the user
         else
            
            ! If we've only got one level just precondition with jacobi
            call PCSetOperators(pcmg_input, amat, pmat, ierr)
            call PCSetType(pcmg_input, PCJACOBI, ierr)
            if (comm_rank == 0) print *, "Only a single level, defaulting to Jacobi PC"
         end if
      end if      

      ! Call the setup on our PC
      call PCSetUp(pcmg_input, ierr)

      ! ~~~~~~~~~
      ! Build the temporary vectors we use during smoothing
      ! ~~~~~~~~~   
      do our_level = 1, air_data%no_levels-1

         if (.NOT. air_data%allocated_matrices_A_ff(our_level)) then
            call MatCreateVecs(air_data%A_ff(our_level), &
                     air_data%temp_vecs_fine(1)%array(our_level), PETSC_NULL_VEC, ierr)
            call MatCreateVecs(air_data%A_fc(our_level), &
                     air_data%temp_vecs_coarse(1)%array(our_level), PETSC_NULL_VEC, ierr)  
            call MatCreateVecs(air_data%coarse_matrix(our_level), &
                     air_data%temp_vecs(1)%array(our_level), PETSC_NULL_VEC, ierr)
     
            call VecDuplicate(air_data%temp_vecs_fine(1)%array(our_level), air_data%temp_vecs_fine(2)%array(our_level), ierr)
            call VecDuplicate(air_data%temp_vecs_fine(1)%array(our_level), air_data%temp_vecs_fine(3)%array(our_level), ierr)
            call VecDuplicate(air_data%temp_vecs_fine(1)%array(our_level), air_data%temp_vecs_fine(4)%array(our_level), ierr)        

            ! If we're doing C point smoothing we need some extra temporaries
            if (air_data%options%any_c_smooths .AND. &
                     .NOT. air_data%options%full_smoothing_up_and_down) then
               call VecDuplicate(air_data%temp_vecs_coarse(1)%array(our_level), air_data%temp_vecs_coarse(2)%array(our_level), ierr)
               call VecDuplicate(air_data%temp_vecs_coarse(1)%array(our_level), air_data%temp_vecs_coarse(3)%array(our_level), ierr)
               call VecDuplicate(air_data%temp_vecs_coarse(1)%array(our_level), air_data%temp_vecs_coarse(4)%array(our_level), ierr)         
            end if
         end if     
         
         air_data%allocated_matrices_A_ff(our_level) = .TRUE.

      end do

      ! ~~~~~~~~~
      ! We are done, print out info
      ! ~~~~~~~~~       

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
      if (air_data%options%print_stats_timings) call print_stats(air_data, pcmg_input)   

   end subroutine setup_air_pcmg       
   
! -------------------------------------------------------------------------------------------------------------------------------

end module air_mg_setup

