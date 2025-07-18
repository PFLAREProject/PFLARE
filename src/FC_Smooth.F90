module fc_smooth

   use petscksp
   use c_petsc_interfaces
   use air_data_type
   use petsc_helper
   use matshell_pflare

#include "petsc/finclude/petscksp.h"
#include "petscconf.h"
                
   implicit none
   public

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! Functions involving the FC smoothing
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------      

   contains 

   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine create_VecISCopyLocalWrapper(air_data, our_level, mat_type, input_mat)

      ! Creates any data we might need in VecISCopyLocalWrapper for a given level
      ! air_data%fast_veciscopy_exists must have been set before the 
      ! first call to this routine
      
      ! ~~~~~~~~~~
      ! Input 
      type(air_multigrid_data), intent(inout) :: air_data
      integer, intent(in)                     :: our_level
      MatType, intent(in)                     :: mat_type
      type(tMat), intent(in)                  :: input_mat

#if defined(PETSC_HAVE_KOKKOS)                     
      PetscErrorCode :: ierr
      integer(c_long_long) :: is_fine_array, is_coarse_array
      PetscInt global_row_start, global_row_end_plus_one
#endif         
      ! ~~~~~~~~~~

      ! On cpus we use VecISCopy to pull out fine and coarse points
      ! That copies back to the cpu if doing gpu, so on the gpu we build
      ! identity restrictors/prolongators of various sizes and do matmults         
      if (.NOT. air_data%fast_veciscopy_exists) then

         ! Build fine to full injector
         call generate_identity_rect(input_mat, air_data%A_fc(our_level), &
                  air_data%IS_fine_index(our_level), &
                  air_data%i_fine_full(our_level))

         ! Build coarse to full injector
         call generate_identity_rect(input_mat, air_data%A_cf(our_level), &
                  air_data%IS_coarse_index(our_level), &
                  air_data%i_coarse_full(our_level))
                  
         ! Build identity that sets fine in full to zero
         call generate_identity_is(input_mat, air_data%IS_coarse_index(our_level), &
                  air_data%i_coarse_full_full(our_level))               

         ! If we're C point smoothing as well
         if (air_data%options%any_c_smooths .AND. &
                  .NOT. air_data%options%full_smoothing_up_and_down) then     
            
            ! Build identity that sets coarse in full to zero
            call generate_identity_is(input_mat, air_data%IS_fine_index(our_level), &
                  air_data%i_fine_full_full(our_level))                         
         end if 

      ! We're either on the cpu or on the gpu with kokkos
      else
#if defined(PETSC_HAVE_KOKKOS) 

         ! If our mat type is kokkos we need to build some things
         ! If not we just use the petsc veciscopy and don't have to setup anything
         if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
               mat_type == MATAIJKOKKOS) then

            ! Build in case not built yet
            call create_VecISCopyLocal_kokkos(air_data%options%max_levels)
            call MatGetOwnershipRange(input_mat, global_row_start, global_row_end_plus_one, ierr)

            ! Copy the IS's over to the device
            is_fine_array = air_data%IS_fine_index(our_level)%v
            is_coarse_array = air_data%IS_coarse_index(our_level)%v
            call set_VecISCopyLocal_kokkos_our_level(our_level, global_row_start, is_fine_array, is_coarse_array)

         end if
#endif
      end if
         
   end subroutine create_VecISCopyLocalWrapper     

   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine destroy_VecISCopyLocalWrapper(air_data, our_level)

      ! Destroy any data we might need in VecISCopyLocalWrapper for a given level
      
      ! ~~~~~~~~~~
      ! Input 
      type(air_multigrid_data), intent(inout) :: air_data
      integer, intent(in)                     :: our_level

      PetscErrorCode :: ierr
      ! ~~~~~~~~~~

      ! Destroys the matrices       
      if (.NOT. air_data%fast_veciscopy_exists) then

         call MatDestroy(air_data%i_fine_full(our_level), ierr)
         call MatDestroy(air_data%i_coarse_full(our_level), ierr)
         call MatDestroy(air_data%i_fine_full_full(our_level), ierr)
         if (air_data%options%any_c_smooths .AND. &
                  .NOT. air_data%options%full_smoothing_up_and_down) then     
            call MatDestroy(air_data%i_coarse_full_full(our_level), ierr)                       
         end if 

      else
#if defined(PETSC_HAVE_KOKKOS) 
         call destroy_VecISCopyLocal_kokkos()
#endif
      end if
         
   end subroutine destroy_VecISCopyLocalWrapper    

   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine VecISCopyLocalWrapper(air_data, our_level, fine, vfull, mode, vreduced, v_temp_mat)

      ! Wrapper around VecISCopy (currently cpu only), a kokkos version of that and 
      ! the matmult used on gpus when petsc isn't configured with kokkos 
      ! Relies on having pre-built some things with the routine create_VecISCopyLocalWrapper
      
      ! ~~~~~~~~~~
      ! Input 
      type(air_multigrid_data), intent(in) :: air_data
      integer, intent(in)                  :: our_level
      logical, intent(in)                  :: fine
      type(tVec), intent(inout)            :: vfull, vreduced
      type(tVec), optional, intent(inout)  :: v_temp_mat
      ScatterMode, intent(in)              :: mode  
      
      PetscErrorCode :: ierr
      integer :: mode_int
#if defined(PETSC_HAVE_KOKKOS)                     
      integer(c_long_long) :: vfull_array, vreduced_array
      integer :: fine_int, errorcode
      VecType :: vec_type
      Vec :: temp_vec
      PetscScalar normy;
#endif          
      ! ~~~~~~~~~~

      if (mode == SCATTER_REVERSE) then
         mode_int = 1
      else
         mode_int = 0
      end if
      ! FINE variables
      if (fine) then
         if (mode == SCATTER_REVERSE) then

            if (.NOT. air_data%fast_veciscopy_exists) then
               call MatMult(air_data%i_fine_full(our_level), vfull, &
                        vreduced, ierr)                          
            else

#if defined(PETSC_HAVE_KOKKOS)  

               call VecGetType(vfull, vec_type, ierr)
               if (vec_type == "seqkokkos" .OR. vec_type == "mpikokkos" .OR. &
                        vec_type == "kokkos") then

                  fine_int = 0
                  if (fine) fine_int = 1
                  vfull_array = vfull%v
                  vreduced_array = vreduced%v
                  call VecISCopyLocal_kokkos(our_level, fine_int, vfull_array, &
                           mode_int, vreduced_array)

                  ! If debugging do a comparison between CPU and Kokkos results
                  if (kokkos_debug()) then             
                     
                     call VecDuplicate(vreduced, temp_vec, ierr)
                     call VecISCopy(vfull, air_data%is_fine_index(our_level), mode, &
                              temp_vec, ierr)
                     call VecAXPY(temp_vec, -1d0, vreduced, ierr)
                     call VecNorm(temp_vec, NORM_2, normy, ierr)
                     if (normy .gt. 1d-13) then
                        print *, "Kokkos and CPU versions of VecISCopyLocalWrapper REV FINE do not match"
                        call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)  
                     end if
                     call VecDestroy(temp_vec, ierr)

                  end if

               else
                  call VecISCopy(vfull, air_data%is_fine_index(our_level), mode, &
                        vreduced, ierr)
               end if
#else
               call VecISCopy(vfull, air_data%is_fine_index(our_level), mode, &
                        vreduced, ierr)
#endif
            end if

         ! SCATTER FORWARD
         else
            if (.NOT. air_data%fast_veciscopy_exists) then

               ! Copy x but only the non-coarse points from x are non-zero
               ! ie get x_c but in a vec of full size 
               call MatMult(air_data%i_coarse_full_full(our_level), vfull, &
                                 v_temp_mat, ierr)        

               ! If we're just doing F point smoothing, don't change the coarse points 
               ! Not sure why we need the vecset, but on the gpu x is twice the size it should be if we don't
               ! x should be overwritten by the MatMultTransposeAdd
               call VecSet(vfull, 0d0, ierr)
               call MatMultTransposeAdd(air_data%i_fine_full(our_level), &
                     vreduced, &
                     v_temp_mat, &
                     vfull, ierr)               

            else

#if defined(PETSC_HAVE_KOKKOS)  

               call VecGetType(vfull, vec_type, ierr)
               if (vec_type == "seqkokkos" .OR. vec_type == "mpikokkos" .OR. &
                        vec_type == "kokkos") then

                  if (kokkos_debug()) then             
                     call VecDuplicate(vfull, temp_vec, ierr)
                     call VecCopy(vfull, temp_vec, ierr)                           
                  end if

                  fine_int = 0
                  if (fine) fine_int = 1
                  vfull_array = vfull%v
                  vreduced_array = vreduced%v
                  call VecISCopyLocal_kokkos(our_level, fine_int, vfull_array, &
                           mode_int, vreduced_array)

                  ! If debugging do a comparison between CPU and Kokkos results
                  if (kokkos_debug()) then             
                     
                     call VecISCopy(temp_vec, air_data%is_fine_index(our_level), mode, &
                              vreduced, ierr)  
                     call VecAXPY(temp_vec, -1d0, vfull, ierr)
                     call VecNorm(temp_vec, NORM_2, normy, ierr)
                     if (normy .gt. 1d-13) then
                        print *, "Kokkos and CPU versions of VecISCopyLocalWrapper FORW FINE do not match"
                        call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)  
                     end if
                     call VecDestroy(temp_vec, ierr)

                  end if                           

               else
                  call VecISCopy(vfull, air_data%is_fine_index(our_level), mode, &
                           vreduced, ierr)  
               end if
#else
               call VecISCopy(vfull, air_data%is_fine_index(our_level), mode, &
                        vreduced, ierr)                 
#endif                        
            end if
         end if

      ! COARSE variables
      else
         if (mode == SCATTER_REVERSE) then

            if (.NOT. air_data%fast_veciscopy_exists) then
               call MatMult(air_data%i_coarse_full(our_level), vfull, &
                        vreduced, ierr)                          
            else

#if defined(PETSC_HAVE_KOKKOS)  

               call VecGetType(vfull, vec_type, ierr)
               if (vec_type == "seqkokkos" .OR. vec_type == "mpikokkos" .OR. &
                        vec_type == "kokkos") then

                  fine_int = 0
                  if (fine) fine_int = 1
                  vfull_array = vfull%v
                  vreduced_array = vreduced%v
                  call VecISCopyLocal_kokkos(our_level, fine_int, vfull_array, &
                           mode_int, vreduced_array)

                  ! If debugging do a comparison between CPU and Kokkos results
                  if (kokkos_debug()) then             
                     
                     call VecDuplicate(vreduced, temp_vec, ierr)
                     call VecISCopy(vfull, air_data%is_coarse_index(our_level), mode, &
                           temp_vec, ierr) 
                     call VecAXPY(temp_vec, -1d0, vreduced, ierr)
                     call VecNorm(temp_vec, NORM_2, normy, ierr)
                     if (normy .gt. 1d-13) then
                        print *, "Kokkos and CPU versions of VecISCopyLocalWrapper REV COARSE do not match"
                        call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)  
                     end if
                     call VecDestroy(temp_vec, ierr)

                  end if                            

               else
                  call VecISCopy(vfull, air_data%is_coarse_index(our_level), mode, &
                           vreduced, ierr)
               end if
#else               
               call VecISCopy(vfull, air_data%is_coarse_index(our_level), mode, &
                        vreduced, ierr)
#endif                        
            end if

         ! SCATTER FORWARD
         else 

            if (.NOT. air_data%fast_veciscopy_exists) then

               ! Copy x but only the non-fine points from x are non-zero
               ! ie get x_f but in a vec of full size 
               call MatMult(air_data%i_fine_full_full(our_level), vfull, &
                                 v_temp_mat, ierr)        

               ! Not sure why we need the vecset, but on the gpu x is twice the size it should be if we don't
               ! x should be overwritten by the MatMultTransposeAdd
               call VecSet(vfull, 0d0, ierr)
               call MatMultTransposeAdd(air_data%i_coarse_full(our_level), &
                     vreduced, &
                     v_temp_mat, &
                     vfull, ierr)    

            else      
               
#if defined(PETSC_HAVE_KOKKOS)  

               call VecGetType(vfull, vec_type, ierr)
               if (vec_type == "seqkokkos" .OR. vec_type == "mpikokkos" .OR. &
                        vec_type == "kokkos") then

                  if (kokkos_debug()) then             
                     call VecDuplicate(vfull, temp_vec, ierr)
                     call VecCopy(vfull, temp_vec, ierr)                           
                  end if                           

                  fine_int = 0
                  if (fine) fine_int = 1
                  vfull_array = vfull%v
                  vreduced_array = vreduced%v
                  call VecISCopyLocal_kokkos(our_level, fine_int, vfull_array, &
                           mode_int, vreduced_array)

                  ! If debugging do a comparison between CPU and Kokkos results
                  if (kokkos_debug()) then             
                     
                     call VecISCopy(temp_vec, air_data%is_coarse_index(our_level), mode, &
                           vreduced, ierr)
                     call VecAXPY(temp_vec, -1d0, vfull, ierr)
                     call VecNorm(temp_vec, NORM_2, normy, ierr)
                     if (normy .gt. 1d-13) then
                        print *, "Kokkos and CPU versions of VecISCopyLocalWrapper FORW COARSE do not match"
                        call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)  
                     end if
                     call VecDestroy(temp_vec, ierr)

                  end if                            

               else
                  call VecISCopy(vfull, air_data%is_coarse_index(our_level), mode, &
                        vreduced, ierr)
               end if
#else                 
               call VecISCopy(vfull, air_data%is_coarse_index(our_level), mode, &
                        vreduced, ierr)
#endif                        
            end if            
         end if
      end if     
         
   end subroutine VecISCopyLocalWrapper   

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine mg_FC_point_richardson(pc, b, x, r, rtol, abstol, dtol, maxits, guess_zero, its, conv_reason, ierr)

      ! This applies an FC point richardson. This saves computing full residuals on each level
      ! This is automatically disabled if you run with -mg_levels_ksp_monitor fyi!

      ! ~~~~~~
      type(tPC) :: pc
      type(tVec) :: b, x, r
      PetscReal :: rtol, abstol, dtol
      PetscInt :: maxits, its
      PetscBool :: guess_zero
      PCRichardsonConvergedReason :: conv_reason
      PetscErrorCode :: ierr

      type(tMat) :: mat, pmat
      integer :: our_level, errorcode, i, smooth_its
      type(mat_ctxtype), pointer :: mat_ctx  
      type(air_multigrid_data), pointer :: air_data
      PetscBool :: first_smooth

      ! ~~~~~~

      ! Set these for output
      its = maxits
      conv_reason = PCRICHARDSON_CONVERGED_ITS;

      ! Can come in here with zero maxits, have to do nothing
      if (maxits == 0) return
      if (maxits /= 1) then
         print *, "To change the number of smooths adjust smooth_order"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! Get the level 
      call PCGetOperators(pc, mat, pmat, ierr)
      ! Get what level we are on
      call MatShellGetContext(mat, mat_ctx, ierr)
      our_level = mat_ctx%our_level
      air_data => mat_ctx%air_data  

      ! The first time we go through any smooth, we need to pull out x_f and/or x_c
      first_smooth = PETSC_TRUE

      ! Loop over all the smooths we need to do
      do i = 1, size(air_data%smooth_order_levels(our_level)%array)

         smooth_its = air_data%smooth_order_levels(our_level)%array(i)

         if (smooth_its == 0) exit

         ! Do consecutive F point smooths
         if (smooth_its > 0) then

            call f_smooths(b, x, guess_zero, first_smooth, air_data, our_level, smooth_its)            

         ! Do consecutive C point smooths
         else
            
            call c_smooths(b, x, guess_zero, first_smooth, air_data, our_level, abs(smooth_its))
         end if

         ! Once we've done our first smooth, we can use the existing values
         first_smooth = PETSC_FALSE

      end do
      
      ! Now technically there should be a new residual that we put into r after this is done
      ! but I don't think it matters, as it is the solution that is interpolated up 
      ! and the richardson on the next level up computes its own F-point residual 
      ! and the norm type is none on the mg levels, as we just do maxits        

      ! have to return zero here!
      ierr = 0
      
   end subroutine mg_FC_point_richardson     

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine f_smooths(b, x, guess_zero, first_smooth, air_data, our_level, its)

      ! This applies consecutive F smooths

      ! ~~~~~~
      type(tVec), intent(inout)               :: b, x
      type(air_multigrid_data), intent(inout) :: air_data
      integer, intent(in)                     :: our_level, its
      PetscBool, intent(in)                   :: guess_zero, first_smooth

      PetscErrorCode :: ierr
      integer :: f_its

      ! ~~~~~~

      ! Get out just the fine points from b - this is b_f
      call VecISCopyLocalWrapper(air_data, our_level, .TRUE., b, &
               SCATTER_REVERSE, air_data%temp_vecs_fine(4)%array(our_level))

      ! If we haven't done any smooth before calling this F point smooth
      ! we need to pull out x_c^0 and x_f^0              
      if (first_smooth) then        

         ! Get out just the fine points from x - this is x_f^0
         call VecISCopyLocalWrapper(air_data, our_level, .TRUE., x, &
                  SCATTER_REVERSE, air_data%temp_vecs_fine(1)%array(our_level))             

         ! Get the coarse points from x - this is x_c^0
         call VecISCopyLocalWrapper(air_data, our_level, .FALSE., x, &
                  SCATTER_REVERSE, air_data%temp_vecs_coarse(1)%array(our_level))   
                     
      end if

      ! Compute Afc * x_c^0 - this never changes
      call MatMult(air_data%A_fc(our_level), air_data%temp_vecs_coarse(1)%array(our_level), &
               air_data%temp_vecs_fine(2)%array(our_level), ierr)               
      
      ! This is b_f - A_fc * x_c^0 - this never changes
      call VecAXPY(air_data%temp_vecs_fine(4)%array(our_level), -1d0, &
               air_data%temp_vecs_fine(2)%array(our_level), ierr)                      

      ! Do all the consecutive F smooths
      do f_its = 1, its

         ! Then A_ff * x_f^n - this changes at each richardson iteration
         call MatMult(air_data%A_ff(our_level), air_data%temp_vecs_fine(1)%array(our_level), &
                     air_data%temp_vecs_fine(3)%array(our_level), ierr)          

         ! This is b_f - A_fc * x_c - A_ff * x_f^n
         call VecAYPX(air_data%temp_vecs_fine(3)%array(our_level), -1d0, &
                  air_data%temp_vecs_fine(4)%array(our_level), ierr)           

         ! ! Compute A_ff^{-1} ( b_f - A_fc * x_c - A_ff * x_f^n)
         call MatMult(air_data%inv_A_ff(our_level), air_data%temp_vecs_fine(3)%array(our_level), &
                     air_data%temp_vecs_fine(2)%array(our_level), ierr)    

         ! Compute x_f^n + A_ff^{-1} ( b_f - A_fc * x_c - A_ff * x_f^n)
         call VecAXPY(air_data%temp_vecs_fine(1)%array(our_level), 1d0, &
                  air_data%temp_vecs_fine(2)%array(our_level), ierr)                      

      end do

      ! ~~~~~~~~
      ! Reverse put fine x_f back into x
      ! ~~~~~~~~
      call VecISCopyLocalWrapper(air_data, our_level, .TRUE., x, &
               SCATTER_FORWARD, air_data%temp_vecs_fine(1)%array(our_level), &
               air_data%temp_vecs(1)%array(our_level))

   end subroutine f_smooths

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine c_smooths(b, x, guess_zero, first_smooth, air_data, our_level, its)

      ! This applies consecutive C smooths

      ! ~~~~~~
      type(tVec), intent(inout)               :: b, x
      type(air_multigrid_data), intent(inout) :: air_data
      integer, intent(in)                     :: our_level, its
      PetscBool, intent(in)                   :: guess_zero, first_smooth

      PetscErrorCode :: ierr
      integer :: c_its

      ! ~~~~~~  

      ! Get out just the coarse points from b - this is b_c
      call VecISCopyLocalWrapper(air_data, our_level, .FALSE., b, &
               SCATTER_REVERSE, air_data%temp_vecs_coarse(4)%array(our_level))

      ! If we haven't done any smooth before calling this C point smooth
      ! we need to pull out x_c^0 and x_f^0
      if (first_smooth) then

            ! Get out just the fine points from x - this is x_f^0
         call VecISCopyLocalWrapper(air_data, our_level, .TRUE., x, &
                  SCATTER_REVERSE, air_data%temp_vecs_fine(1)%array(our_level))             

         ! Get the coarse points from x - this is x_c^0
         call VecISCopyLocalWrapper(air_data, our_level, .FALSE., x, &
                  SCATTER_REVERSE, air_data%temp_vecs_coarse(1)%array(our_level))  
                  
      end if

      ! Compute Acf * x_f^0 - this never changes
      call MatMult(air_data%A_cf(our_level), air_data%temp_vecs_fine(1)%array(our_level), &
                  air_data%temp_vecs_coarse(2)%array(our_level), ierr)
      ! This is b_c - A_cf * x_f^0 - this never changes
      call VecAXPY(air_data%temp_vecs_coarse(4)%array(our_level), -1d0, &
               air_data%temp_vecs_coarse(2)%array(our_level), ierr)  

      ! Do all the consecutive C smooths
      do c_its = 1, its

         ! Then A_cc * x_c^n - this changes at each richardson iteration
         call MatMult(air_data%A_cc(our_level), air_data%temp_vecs_coarse(1)%array(our_level), &
                     air_data%temp_vecs_coarse(3)%array(our_level), ierr)       

         ! This is b_c - A_cf * x_f^0 - A_cc * x_c^n
         call VecAYPX(air_data%temp_vecs_coarse(3)%array(our_level), -1d0, &
                  air_data%temp_vecs_coarse(4)%array(our_level), ierr)          

         ! ! Compute A_cc^{-1} (b_c - A_cf * x_f^0 - A_cc * x_c^n)
         call MatMult(air_data%inv_A_cc(our_level), air_data%temp_vecs_coarse(3)%array(our_level), &
                     air_data%temp_vecs_coarse(2)%array(our_level), ierr)    

         ! Compute x_c^n + A_cc^{-1} (b_c - A_cf * x_f^0 - A_cc * x_c^n)
         call VecAXPY(air_data%temp_vecs_coarse(1)%array(our_level), 1d0, &
                     air_data%temp_vecs_coarse(2)%array(our_level), ierr)    
                     
      end do

      ! ~~~~~~~~
      ! Reverse put coarse x_c back into x
      ! ~~~~~~~~
      call VecISCopyLocalWrapper(air_data, our_level, .FALSE., x, &
               SCATTER_FORWARD, air_data%temp_vecs_coarse(1)%array(our_level), &
               air_data%temp_vecs(1)%array(our_level))          
      
   end subroutine c_smooths      
 
   !------------------------------------------------------------------------------------------------------------------------
   
end module fc_smooth

