module air_mg_stats

   use petscksp
   use petsc_helper
   use air_data_type
   use gmres_poly

#include "petsc/finclude/petscksp.h"
      
   implicit none

#include "petsc_legacy.h"   

   public

   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine compute_nnzs_air_matrices(air_data)
      
      ! Compute the nnzs of the matrices in air

      ! ~~~~~~
      type(air_multigrid_data), intent(inout) :: air_data

      integer :: our_level
      PetscErrorCode :: ierr
      MatType:: mat_type

      ! ~~~~~~    

      ! The coarse solver nnzs
      if (air_data%no_levels /= 1) then
         ! Are we mf?
         call MatGetType(air_data%inv_A_ff(air_data%no_levels), mat_type, ierr)
         if (mat_type/=MATSHELL) then
            call get_nnzs_petsc_sparse(air_data%inv_A_ff(air_data%no_levels), air_data%inv_A_ff_nnzs(air_data%no_levels))                                             
         end if
         call get_nnzs_petsc_sparse(air_data%coarse_matrix(air_data%no_levels), air_data%coarse_matrix_nnzs(air_data%no_levels))      
      end if
      
      ! Then go over all the levels except the coarse
      do our_level = 1, air_data%no_levels-1        
         
         ! The fine/coarse components of our matrix
         call get_nnzs_petsc_sparse(air_data%A_ff(our_level), air_data%A_ff_nnzs(our_level))                          
         call get_nnzs_petsc_sparse(air_data%A_fc(our_level), air_data%A_fc_nnzs(our_level))
         
         ! Are we mf?
         call MatGetType(air_data%inv_A_ff(our_level), mat_type, ierr)
         if (mat_type/=MATSHELL) then         
            call get_nnzs_petsc_sparse(air_data%inv_A_ff(our_level), air_data%inv_A_ff_nnzs(our_level))                
         end if
         ! Are we doing C point smoothing
         if (air_data%options%one_c_smooth .AND. &
                  .NOT. air_data%options%full_smoothing_up_and_down) then 
            
            call get_nnzs_petsc_sparse(air_data%A_cc(our_level), air_data%A_cc_nnzs(our_level))
            call get_nnzs_petsc_sparse(air_data%A_cf(our_level), air_data%A_cf_nnzs(our_level))

            ! Are we mf?
            call MatGetType(air_data%inv_A_cc(our_level), mat_type, ierr)
            if (mat_type/=MATSHELL) then         
               call get_nnzs_petsc_sparse(air_data%inv_A_cc(our_level), air_data%inv_A_cc_nnzs(our_level))                
            end if            
         end if
         
         ! Add in the nnzs from the restrictor
         if (.NOT. air_data%options%symmetric) then
            call get_nnzs_petsc_sparse(air_data%restrictors(our_level), air_data%restrictor_nnzs(our_level))                                             
         end if
         ! Add in the nnzs from the prolongator
         call get_nnzs_petsc_sparse(air_data%prolongators(our_level), air_data%prolongator_nnzs(our_level))                                             

      end do

   end subroutine compute_nnzs_air_matrices      

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine compute_nnzs_v_cycle_air(air_data, pcmg_input, nnzs)
      
      ! Compute the nnzs of a V cycle of air
      ! Have to input both the air data and the pcmg that comes from setup_air_pcmg

      ! ~~~~~~
      type(air_multigrid_data), intent(inout):: air_data
      type(tPC), intent(in)                  :: pcmg_input
      integer(kind=8), intent(out)           :: nnzs

      integer :: our_level, non_zero_order
      PetscInt :: maxits, petsc_level
      PetscErrorCode :: ierr
      PCType :: pc_type
      type(tKSP) :: ksp
      PetscReal :: rtol, atol, dtol
      integer(kind=8) maxits_long, maxits_aff_long, gmres_size_long, poly_order_long
      MatType :: mat_type

      ! ~~~~~~    
      
      ! Set to zero to start
      nnzs = 0

      ! Check we're doing mg
      call PCGetType(pcmg_input, pc_type, ierr)
      if (pc_type /= PCMG) return

      ! Coarse grid solve
      call PCMGGetCoarseSolve(pcmg_input, ksp, ierr)   
      call KSPGetTolerances(ksp, rtol, atol, dtol, maxits, ierr)
      ! Round to long
      maxits_long = int(maxits, kind=8)

      ! How many matvecs do we need to apply our coarse grid polynomial inverse
      ! Normally this is just the size of air_data%inv_coarsest_poly_data%coefficients, but for newton
      ! we may have zero roots that we skip
      call compute_mf_gmres_poly_num_matvecs(air_data%options%coarsest_inverse_type, &
                  air_data%inv_coarsest_poly_data%coefficients, &
                  non_zero_order)

      gmres_size_long = int(non_zero_order, kind=8)
      poly_order_long = int(air_data%options%poly_order, kind=8)

      ! Application of the coarse matrix inverse with richardson
      ! No need for an initial matvec here with the coarse matrix as we have just restricted the residual (which is just b) straight down here
      ! and the initial guess on the coarse grid is zero

      ! Are we mf?
      call MatGetType(air_data%inv_A_ff(air_data%no_levels), mat_type, ierr)
      if (mat_type==MATSHELL) then
         nnzs = maxits_long * gmres_size_long * air_data%coarse_matrix_nnzs(air_data%no_levels) + &
                  (maxits_long-1) * air_data%coarse_matrix_nnzs(air_data%no_levels)
      else
         nnzs = maxits_long * air_data%inv_A_ff_nnzs(air_data%no_levels) + &
                  (maxits_long-1) * air_data%coarse_matrix_nnzs(air_data%no_levels)         
      end if
      
      ! Then go over all the levels except the coarse
      do our_level = 1, air_data%no_levels-1

         ! Round to long
         maxits_aff_long = int(air_data%options%maxits_a_ff, kind=8)

         ! ~~~~~~~~~~~~~
         ! No down smooths
         ! ~~~~~~~~~~~~~         

         ! ~~~~~~~~~~~~~
         ! Up F point smooths
         ! ~~~~~~~~~~~~~
         ! Make sure to use the petsc level
         ! Now being careful here to actually pull out the down smoother from petsc
         ! as the kaskade cycle only calls the "down" smoother on the way up
         petsc_level = air_data%no_levels-our_level
         call PCMGGetSmootherDown(pcmg_input, petsc_level, ksp, ierr)      
         ! Get the number of F&C up smooths
         call KSPGetTolerances(ksp, rtol, atol, dtol, maxits, ierr)   

         ! Round to long
         maxits_long = maxits

         ! on the way up, we are solving the same equation A e = r, but this time the error is non-zero, so our richardson's will 
         ! have to do an initial F-point residual. The F-point smooths then proceed as normal, with the rhs being the residual from when we came down
         ! and we don't have to do a post residual

         ! Are we mf?
         call MatGetType(air_data%inv_A_ff(our_level), mat_type, ierr)         
         
         if (.NOT. air_data%options%full_smoothing_up_and_down) then

            ! MF polynomial order may be different on each level
            call compute_mf_gmres_poly_num_matvecs(air_data%options%inverse_type, &
                        air_data%inv_A_ff_poly_data(our_level)%coefficients, &
                        non_zero_order)    
            gmres_size_long = int(non_zero_order, kind=8)

            if (mat_type==MATSHELL) then
               nnzs = nnzs + maxits_long * maxits_aff_long * gmres_size_long * air_data%A_ff_nnzs(our_level) + &
                           maxits_long * (maxits_aff_long) * air_data%A_ff_nnzs(our_level)
            else
               nnzs = nnzs + maxits_long * maxits_aff_long * air_data%inv_A_ff_nnzs(our_level) + &
                           maxits_long * (maxits_aff_long) * air_data%A_ff_nnzs(our_level)
            end if

            ! Add in the minus Afc - this is done once before we F-point smooth
            nnzs = nnzs + maxits_long * air_data%A_fc_nnzs(our_level)    

            ! One C-point smooth         
            if (air_data%options%one_c_smooth) then

               ! MF polynomial order may be different on each level
               call compute_mf_gmres_poly_num_matvecs(air_data%options%c_inverse_type, &
                           air_data%inv_A_cc_poly_data(our_level)%coefficients, &
                           non_zero_order)  
               gmres_size_long = int(non_zero_order, kind=8)

               ! Are we mf?
               call MatGetType(air_data%inv_A_cc(our_level), mat_type, ierr)
               if (mat_type==MATSHELL) then  
                  nnzs = nnzs + maxits_long * gmres_size_long * air_data%A_cc_nnzs(our_level) + &
                           maxits_long * air_data%A_cc_nnzs(our_level)
               else          
                  nnzs = nnzs + maxits_long * air_data%inv_A_cc_nnzs(our_level) + &
                           maxits_long * air_data%A_cc_nnzs(our_level)
               end if
               ! Add in the minus Acf
               nnzs = nnzs + maxits_long * air_data%A_cf_nnzs(our_level)
            end if            

         ! Full smoothing up and down
         else
            
            ! Symmetric GS and up and down
            !nnzs = nnzs + maxits_long * air_data%coarse_matrix_nnzs(our_level) * 2 * 2            

            ! MF polynomial order may be different on each level
            call compute_mf_gmres_poly_num_matvecs(air_data%options%inverse_type, &
                        air_data%inv_A_ff_poly_data(our_level)%coefficients, &
                        non_zero_order)   
            gmres_size_long = int(non_zero_order, kind=8)                                   

            ! The two is because we do up and down smoothing
            if (mat_type==MATSHELL) then
               nnzs = nnzs + 2 * maxits_long * gmres_size_long * air_data%coarse_matrix_nnzs(our_level) + &
                           maxits_long  * air_data%coarse_matrix_nnzs(our_level)
            else
               nnzs = nnzs + 2 * maxits_long * air_data%inv_A_ff_nnzs(our_level) + &
                           maxits_long * air_data%coarse_matrix_nnzs(our_level)
            end if
         end if

         ! Restrictor
         if (air_data%options%symmetric) then
            nnzs = nnzs + air_data%prolongator_nnzs(our_level)
         else
            nnzs = nnzs + air_data%restrictor_nnzs(our_level)
         end if
         ! Prolongator
         nnzs = nnzs + air_data%prolongator_nnzs(our_level)
      end do

   end subroutine compute_nnzs_v_cycle_air   

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine compute_stats(air_data, pcmg_input, grid_complx, op_complx, cycle_complx, storage_complx, reuse_storage_complx)
      
      ! Compute the different stats for air
      ! Have to input both the air data and the pcmg that comes from setup_air_pcmg

      ! ~~~~~~
      type(air_multigrid_data), intent(inout) :: air_data
      type(tPC), intent(in)                   :: pcmg_input
      PetscReal, intent(out)  :: grid_complx, op_complx, cycle_complx, storage_complx, reuse_storage_complx

      integer :: our_level, i_loc
      PetscInt :: maxits_coarse
      PetscInt :: global_rows, global_cols
      PetscErrorCode :: ierr
      type(tKSP) :: ksp
      PetscReal :: rtol, atol, dtol
      integer(kind=8) :: nnzs_air_v, mat_storage_nnzs, op_complx_nnzs, mat_reuse_storage_nnzs, mat_nnzs
      type(tMat) :: temp_mat
      type(tIS) :: temp_is

      ! ~~~~~~    

      ! Compute the nnzs in each matrix - this has parallel reductions in it
      call compute_nnzs_air_matrices(air_data)
      ! Compute the nnzs in a single V cycle
      call compute_nnzs_v_cycle_air(air_data, &
                     pcmg_input, &
                     nnzs_air_v)   
            
      ! ~~~~~~~~~
      ! Compute the grid complexity
      ! ~~~~~~~~~
      grid_complx = 0
      do our_level = 1, air_data%no_levels-1
         call MatGetSize(air_data%coarse_matrix(our_level), global_rows, global_cols, ierr)
         grid_complx = grid_complx + dble(global_rows)
      end do
      ! Don't forget the bottom grid
      if (air_data%no_levels /= 1) then
         call MatGetSize(air_data%coarse_matrix(air_data%no_levels), global_rows, global_cols, ierr)
         grid_complx = grid_complx + dble(global_rows)
      end if

      call MatGetSize(air_data%coarse_matrix(1), global_rows, global_cols, ierr)
      grid_complx = grid_complx/dble(global_rows)      
      
      
      ! ~~~~~~~~~
      ! Compute the nnzs for operator complexity and storage complexity
      ! ~~~~~~~~~      
      ! Need a do loop so we don't overflow in the sum
      op_complx_nnzs = 0 
      do our_level = 1, air_data%no_levels
         op_complx_nnzs = op_complx_nnzs + air_data%coarse_matrix_nnzs(our_level)      
      end do        
      mat_storage_nnzs = 0
      do our_level = 1, air_data%no_levels-1
         ! Given we do F point up smoothing only, we don't actually have to store very much (e.g., the coarse matrix)
         ! So how much matrix storage do we actually need
         if (.NOT. air_data%options%full_smoothing_up_and_down) then
            mat_storage_nnzs = mat_storage_nnzs + air_data%A_ff_nnzs(our_level)
            mat_storage_nnzs = mat_storage_nnzs + air_data%A_fc_nnzs(our_level)
            ! If we are doing C point smoothing need the other matrices
            if (air_data%options%one_c_smooth) then
               mat_storage_nnzs = mat_storage_nnzs + air_data%A_cc_nnzs(our_level)
               mat_storage_nnzs = mat_storage_nnzs + air_data%inv_A_cc_nnzs(our_level) 
               mat_storage_nnzs = mat_storage_nnzs + air_data%A_cf_nnzs(our_level)                 
            end if            
         else
            mat_storage_nnzs = mat_storage_nnzs + air_data%coarse_matrix_nnzs(our_level)      
         end if
         mat_storage_nnzs = mat_storage_nnzs + air_data%inv_A_ff_nnzs(our_level)
         mat_storage_nnzs = mat_storage_nnzs + air_data%restrictor_nnzs(our_level)
         mat_storage_nnzs = mat_storage_nnzs + air_data%prolongator_nnzs(our_level)        
      end do  
      
      ! Coarse grid solve
      if (air_data%no_levels /= 1) then
         call PCMGGetCoarseSolve(pcmg_input, ksp, ierr)      
         call KSPGetTolerances(ksp, rtol, atol, dtol, maxits_coarse, ierr)       
         ! If we're doing matrix-free application of coarse grid, we need the coarse grid
         if (air_data%options%coarsest_matrix_free_polys) then
            mat_storage_nnzs = mat_storage_nnzs + air_data%coarse_matrix_nnzs(air_data%no_levels)
         ! If we're not doing matrix-free application of the coarse grid, then we need the inverse of the coarse grid matrix
         else
            mat_storage_nnzs = mat_storage_nnzs + air_data%inv_A_ff_nnzs(air_data%no_levels)

            ! If we're doing more than one mb iteration on the coarse grid, then we also need to store the coarse grid matrix
            ! in order to compute a residual in the richardson   
            if (maxits_coarse /= 1) then      
               mat_storage_nnzs = mat_storage_nnzs + air_data%coarse_matrix_nnzs(air_data%no_levels)      
            end if         
         end if 
      end if   
      
      ! ~~~~~~~~~
      ! Compute the nnzs used as part of the reuse
      ! ~~~~~~~~~  
      mat_reuse_storage_nnzs = 0
      do our_level = 1, air_data%no_levels-1
         ! Loop over all the reused matrices
         do i_loc = 1, size(air_data%reuse(our_level)%reuse_mat)
            temp_mat = air_data%reuse(our_level)%reuse_mat(i_loc)
            if (.NOT. PetscMatIsNull(temp_mat)) then
               call get_nnzs_petsc_sparse(air_data%reuse(our_level)%reuse_mat(i_loc), mat_nnzs)
               mat_reuse_storage_nnzs = mat_reuse_storage_nnzs + mat_nnzs
            end if
         end do
         ! Include any IS's
         do i_loc = 1, size(air_data%reuse(our_level)%reuse_is)
            temp_is = air_data%reuse(our_level)%reuse_is(i_loc)
            if (.NOT. PetscISIsNull(temp_is)) then
               call ISGetSize(air_data%reuse(our_level)%reuse_is(i_loc), global_rows, ierr)
               mat_reuse_storage_nnzs = mat_reuse_storage_nnzs + global_rows
            end if
         end do            
      end do

      ! Now compute the other complexities
      op_complx = dble(op_complx_nnzs)/dble(air_data%coarse_matrix_nnzs(1))
      cycle_complx = dble(nnzs_air_v)/dble(air_data%coarse_matrix_nnzs(1))
      storage_complx = dble(mat_storage_nnzs)/dble(air_data%coarse_matrix_nnzs(1))  
      reuse_storage_complx = dble(mat_reuse_storage_nnzs)/dble(air_data%coarse_matrix_nnzs(1))  

   end subroutine compute_stats

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine print_stats(air_data, pcmg_input)
      
      ! Print out stats on the mg hierarchy

      ! ~~~~~~
      type(air_multigrid_data), intent(inout) :: air_data
      type(tPC), intent(in)                   :: pcmg_input

      PetscReal :: grid_complx, op_complx, cycle_complx, storage_complx, reuse_storage_complx
      integer :: comm_rank, errorcode
      PetscErrorCode :: ierr
      MPI_Comm :: MPI_COMM_MATRIX

      ! ~~~~~~   
      
      ! Get the communicator the input matrix is on, we build everything on that
      call PetscObjectGetComm(pcmg_input, MPI_COMM_MATRIX, ierr)      
      ! Get the comm rank
      call MPI_Comm_rank(MPI_COMM_MATRIX, comm_rank, errorcode)       

      call compute_stats(air_data, pcmg_input, grid_complx, op_complx, cycle_complx, storage_complx, reuse_storage_complx)

      if (comm_rank == 0) print *, "~~~~~~~~~~~~"
      if (comm_rank == 0) print *, "Grid complexity            : ", grid_complx
      if (comm_rank == 0) print *, "Operator complexity        : ", op_complx
      if (comm_rank == 0) print *, "Cycle complexity           : ", cycle_complx
      if (comm_rank == 0) print *, "Storage complexity         : ", storage_complx
      if (comm_rank == 0) print *, "Reuse storage complexity   : ", reuse_storage_complx
      if (comm_rank == 0) print *, "~~~~~~~~~~~~"

   end subroutine print_stats   
   
   ! -------------------------------------------------------------------------------------------------------------------------------   

end module air_mg_stats

