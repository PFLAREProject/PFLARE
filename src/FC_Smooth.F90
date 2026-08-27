module fc_smooth

   use petscksp
   use c_petsc_interfaces, only: create_VecISCopyLocal_kokkos, &
         set_VecISCopyLocal_kokkos_our_level, &
         destroy_VecISCopyLocal_kokkos, VecISCopyLocal_kokkos, &
         mat_iscopy_local_kokkos
   use air_data_type, only: air_multigrid_data
   use petsc_helper, only: generate_identity_rect, generate_identity_is, kokkos_debug
   use matshell_data_type, only: mat_ctxtype
   use gmres_poly_newton, only: shell_poly_block_apply
   use pflare_parameters, only: PFLARE_TOL_MATFREE_13, PFLARE_MINUS_ONE, PFLARE_ZERO, &
         PFLARE_ONE, AIR_MAT_SOL, AIR_MAT_TEMP, AIR_MAT_RESIDUAL, AIR_MAT_RHS, &
         AIR_MAT_OFF_DIAG

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

   subroutine mg_coarse_shell_apply(pc, x, y, ierr)

      ! PCShell apply for the default coarse-grid solver: applies our polynomial
      ! approximate inverse of the coarse matrix, inv_A_ff(no_levels), as y = inv_A_ff x.
      ! Using a shell (rather than PCMAT with the inverse as the Pmat) lets us set the
      ! coarse KSP's Amat and Pmat to the actual coarse operator, so a user-supplied
      ! coarse PC (e.g. -mg_coarse_pc_type lu) factorises/solves the right matrix.
      ! The shell context is the air_multigrid_data (a target that outlives the PC).

      ! ~~~~~~
      type(tPC)                             :: pc
      type(tVec)                            :: x, y
      PetscErrorCode, intent(out)           :: ierr

      type(air_multigrid_data), pointer     :: air_data => null()
      ! ~~~~~~

      call PCShellGetContext(pc, air_data, ierr)
      call MatMult(air_data%inv_A_ff(air_data%no_levels), x, y, ierr)

   end subroutine mg_coarse_shell_apply

   !------------------------------------------------------------------------------------------------------------------------

   subroutine mg_coarse_shell_block_apply(pc, x, y, ierr)

      ! The multiple rhs version of mg_coarse_shell_apply - applies our polynomial
      ! approximate inverse of the coarse matrix to a whole dense block

      ! ~~~~~~
      type(tPC)                             :: pc
      type(tMat)                            :: x, y
      PetscErrorCode, intent(out)           :: ierr

      type(air_multigrid_data), pointer     :: air_data => null()
      ! ~~~~~~

      ierr = 0
      call PCShellGetContext(pc, air_data, ierr)
      call apply_inverse_block(air_data%inv_A_ff(air_data%no_levels), x, y, ierr)

   end subroutine mg_coarse_shell_block_apply

   !------------------------------------------------------------------------------------------------------------------------

   subroutine mg_smooth_shell_apply(pc, x, y, ierr)

      ! PCShell apply used for the level smoothers when doing full smoothing up
      ! and down with a matrix-free inverse. This does exactly what PCMAT does
      ! (the Pmat of the smoother is inv_A_ff on that level), we just can't use
      ! PCMAT as its multiple rhs apply does a MatMatMult on the Pmat, which fails
      ! on a matrix-free polynomial matshell

      ! ~~~~~~
      type(tPC)                             :: pc
      type(tVec)                            :: x, y
      PetscErrorCode, intent(out)           :: ierr

      type(tMat) :: mat, pmat
      ! ~~~~~~

      ierr = 0
      call PCGetOperators(pc, mat, pmat, ierr)
      call MatMult(pmat, x, y, ierr)

   end subroutine mg_smooth_shell_apply

   !------------------------------------------------------------------------------------------------------------------------

   subroutine mg_smooth_shell_block_apply(pc, x, y, ierr)

      ! The multiple rhs version of mg_smooth_shell_apply

      ! ~~~~~~
      type(tPC)                             :: pc
      type(tMat)                            :: x, y
      PetscErrorCode, intent(out)           :: ierr

      type(tMat) :: mat, pmat
      ! ~~~~~~

      ierr = 0
      call PCGetOperators(pc, mat, pmat, ierr)
      call apply_inverse_block(pmat, x, y, ierr)

   end subroutine mg_smooth_shell_block_apply

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
            call create_VecISCopyLocal_kokkos(air_data%options%max_levels, air_data%kokkos_is_views_handle)
            call MatGetOwnershipRange(input_mat, global_row_start, global_row_end_plus_one, ierr)

            ! Copy the IS's over to the device
            is_fine_array = air_data%IS_fine_index(our_level)%v
            is_coarse_array = air_data%IS_coarse_index(our_level)%v
            call set_VecISCopyLocal_kokkos_our_level(air_data%kokkos_is_views_handle, our_level, &
                     global_row_start, is_fine_array, is_coarse_array)

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
         call destroy_VecISCopyLocal_kokkos(air_data%kokkos_is_views_handle)
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
                  call VecISCopyLocal_kokkos(air_data%kokkos_is_views_handle, our_level, fine_int, vfull_array, &
                           mode_int, vreduced_array)

                  ! If debugging do a comparison between CPU and Kokkos results
                  if (kokkos_debug()) then             
                     
                     call VecDuplicate(vreduced, temp_vec, ierr)
                     call VecISCopy(vfull, air_data%is_fine_index(our_level), mode, &
                              temp_vec, ierr)
                     call VecAXPY(temp_vec, PFLARE_MINUS_ONE, vreduced, ierr)
                     call VecNorm(temp_vec, NORM_2, normy, ierr)
                     if (normy .gt. PFLARE_TOL_MATFREE_13) then
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
               call VecSet(vfull, PFLARE_ZERO, ierr)
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
                  call VecISCopyLocal_kokkos(air_data%kokkos_is_views_handle, our_level, fine_int, vfull_array, &
                           mode_int, vreduced_array)

                  ! If debugging do a comparison between CPU and Kokkos results
                  if (kokkos_debug()) then             
                     
                     call VecISCopy(temp_vec, air_data%is_fine_index(our_level), mode, &
                              vreduced, ierr)  
                     call VecAXPY(temp_vec, PFLARE_MINUS_ONE, vfull, ierr)
                     call VecNorm(temp_vec, NORM_2, normy, ierr)
                     if (normy .gt. PFLARE_TOL_MATFREE_13) then
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
                  call VecISCopyLocal_kokkos(air_data%kokkos_is_views_handle, our_level, fine_int, vfull_array, &
                           mode_int, vreduced_array)

                  ! If debugging do a comparison between CPU and Kokkos results
                  if (kokkos_debug()) then             
                     
                     call VecDuplicate(vreduced, temp_vec, ierr)
                     call VecISCopy(vfull, air_data%is_coarse_index(our_level), mode, &
                           temp_vec, ierr) 
                     call VecAXPY(temp_vec, PFLARE_MINUS_ONE, vreduced, ierr)
                     call VecNorm(temp_vec, NORM_2, normy, ierr)
                     if (normy .gt. PFLARE_TOL_MATFREE_13) then
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
               call VecSet(vfull, PFLARE_ZERO, ierr)
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
                  call VecISCopyLocal_kokkos(air_data%kokkos_is_views_handle, our_level, fine_int, vfull_array, &
                           mode_int, vreduced_array)

                  ! If debugging do a comparison between CPU and Kokkos results
                  if (kokkos_debug()) then             
                     
                     call VecISCopy(temp_vec, air_data%is_coarse_index(our_level), mode, &
                           vreduced, ierr)
                     call VecAXPY(temp_vec, PFLARE_MINUS_ONE, vfull, ierr)
                     call VecNorm(temp_vec, NORM_2, normy, ierr)
                     if (normy .gt. PFLARE_TOL_MATFREE_13) then
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

   subroutine mat_product_block(mat, x_mat, y_mat, transpose_mat, ierr)

      ! Computes y_mat = mat * x_mat (or mat^T * x_mat) for a dense block of
      ! right hand sides, ie the multiple rhs version of MatMult/MatMultTranspose
      ! The product bookkeeping is always cleared afterwards, or y_mat would keep
      ! holding references to mat and x_mat until it is destroyed

      ! ~~~~~~
      type(tMat), intent(in)        :: mat
      type(tMat)                    :: x_mat, y_mat
      logical, intent(in)           :: transpose_mat
      PetscErrorCode, intent(inout) :: ierr
      ! ~~~~~~

      call MatProductCreateWithMat(mat, x_mat, PETSC_NULL_MAT, y_mat, ierr)
      if (transpose_mat) then
         call MatProductSetType(y_mat, MATPRODUCT_AtB, ierr)
      else
         call MatProductSetType(y_mat, MATPRODUCT_AB, ierr)
      end if
      call MatProductSetFromOptions(y_mat, ierr)
      call MatProductSymbolic(y_mat, ierr)
      call MatProductNumeric(y_mat, ierr)
      call MatProductClear(y_mat, ierr)

   end subroutine mat_product_block

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine shell_block_apply_or_columns(inv_mat, x_mat, y_mat, ierr)

      ! Applies a matrix-free matshell inverse to a dense block of right hand
      ! sides - shell_poly_block_apply knows how to do the polynomial matshells
      ! blockwise by doing the products with the underlying matrix instead, and
      ! reports back if it has been handed something it doesn't know about, in
      ! which case we apply column by column

      ! ~~~~~~
      type(tMat), intent(in)        :: inv_mat
      type(tMat)                    :: x_mat, y_mat
      PetscErrorCode, intent(inout) :: ierr

      logical :: block_applied
      PetscInt :: global_rows, global_cols, i_loc
      type(tVec) :: col_x, col_y
      ! ~~~~~~

      call shell_poly_block_apply(inv_mat, x_mat, y_mat, block_applied)

      ! Anything we can't do blockwise has to be applied column by column
      if (.NOT. block_applied) then
         call MatGetSize(x_mat, global_rows, global_cols, ierr)
         do i_loc = 0, global_cols-1
            call MatDenseGetColumnVecRead(x_mat, i_loc, col_x, ierr)
            call MatDenseGetColumnVecWrite(y_mat, i_loc, col_y, ierr)
            call MatMult(inv_mat, col_x, col_y, ierr)
            call MatDenseRestoreColumnVecWrite(y_mat, i_loc, col_y, ierr)
            call MatDenseRestoreColumnVecRead(x_mat, i_loc, col_x, ierr)
         end do
      end if

   end subroutine shell_block_apply_or_columns

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine apply_inverse_block(inv_mat, x_mat, y_mat, ierr)

      ! Applies one of our approximate inverses to a dense block of right hand
      ! sides, ie the multiple rhs version of MatMult(inv_mat, x, y)
      ! The matrix-free polynomial inverses are matshells with only a MATOP_MULT,
      ! so a product on them would fail - they go through the blockwise shell
      ! apply instead

      ! ~~~~~~
      type(tMat), intent(in)        :: inv_mat
      type(tMat)                    :: x_mat, y_mat
      PetscErrorCode, intent(inout) :: ierr

      MatType :: inv_type
      ! ~~~~~~

      call MatGetType(inv_mat, inv_type, ierr)

      if (inv_type == MATSHELL) then
         call shell_block_apply_or_columns(inv_mat, x_mat, y_mat, ierr)

      ! Assembled (or diagonal) inverses just do a real product
      else
         call mat_product_block(inv_mat, x_mat, y_mat, .FALSE., ierr)
      end if

   end subroutine apply_inverse_block

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine apply_inverse_block_cached(inv_mat, x_mat, y_mat, ierr)

      ! The cached version of apply_inverse_block used inside the block FC
      ! smooths - for assembled (or diagonal) inverses the product has already
      ! been attached to y_mat by setup_air_block_products, so only the numeric
      ! phase runs here. The matrix-free matshells go through the same blockwise
      ! shell apply as apply_inverse_block

      ! ~~~~~~
      type(tMat), intent(in)        :: inv_mat
      type(tMat)                    :: x_mat, y_mat
      PetscErrorCode, intent(inout) :: ierr

      MatType :: inv_type
      ! ~~~~~~

      call MatGetType(inv_mat, inv_type, ierr)

      if (inv_type == MATSHELL) then
         call shell_block_apply_or_columns(inv_mat, x_mat, y_mat, ierr)
      else
         call MatProductNumeric(y_mat, ierr)
      end if

   end subroutine apply_inverse_block_cached

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine attach_block_product(mat, x_mat, y_mat, ierr)

      ! Attaches y_mat = mat * x_mat as a product so repeat computations only
      ! have to run the numeric phase - this is what mat_product_block does, but
      ! with the product bookkeeping deliberately left alive on y_mat
      ! y_mat therefore keeps references to mat and x_mat until it is destroyed

      ! ~~~~~~
      type(tMat), intent(in)        :: mat
      type(tMat)                    :: x_mat, y_mat
      PetscErrorCode, intent(inout) :: ierr
      ! ~~~~~~

      call MatProductCreateWithMat(mat, x_mat, PETSC_NULL_MAT, y_mat, ierr)
      call MatProductSetType(y_mat, MATPRODUCT_AB, ierr)
      call MatProductSetFromOptions(y_mat, ierr)
      call MatProductSymbolic(y_mat, ierr)

   end subroutine attach_block_product

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine setup_air_block_products(air_data, our_level, ierr)

      ! Attaches the products the block FC smooths compute on a level to their
      ! dense scratch targets, so each smoother iteration only has to run the
      ! numeric phase of each product
      ! Called by ensure_air_block_temps whenever the scratch on this level has
      ! just been (re)built - the products stay attached until the scratch is
      ! destroyed, and the scratch outlives every operator it references only
      ! briefly (destroy_air_block_temps runs first in reset_air_data)

      ! ~~~~~~
      type(air_multigrid_data), intent(inout) :: air_data
      integer, intent(in)                     :: our_level
      PetscErrorCode, intent(inout)           :: ierr

      logical :: any_f_smooths, any_c_smooths
      MatType :: inv_type
      ! ~~~~~~

      any_f_smooths = any(air_data%smooth_order_levels(our_level)%array > 0)
      any_c_smooths = any(air_data%smooth_order_levels(our_level)%array < 0)

      if (any_f_smooths) then

         ! A_ff * x_f - computed every F richardson iteration
         call attach_block_product(air_data%A_ff(our_level), &
                  air_data%block_temp_fine(AIR_MAT_SOL)%array(our_level), &
                  air_data%block_temp_fine(AIR_MAT_RESIDUAL)%array(our_level), ierr)

         ! A_fc * x_c - computed once per F smooth
         call attach_block_product(air_data%A_fc(our_level), &
                  air_data%block_temp_coarse(AIR_MAT_SOL)%array(our_level), &
                  air_data%block_temp_fine(AIR_MAT_OFF_DIAG)%array(our_level), ierr)

         ! The assembled (or diagonal) inverses are applied with a product too -
         ! the matrix-free matshells go through the blockwise shell apply instead
         ! and never have a product attached
         call MatGetType(air_data%inv_A_ff(our_level), inv_type, ierr)
         if (inv_type /= MATSHELL) then
            call attach_block_product(air_data%inv_A_ff(our_level), &
                     air_data%block_temp_fine(AIR_MAT_RESIDUAL)%array(our_level), &
                     air_data%block_temp_fine(AIR_MAT_TEMP)%array(our_level), ierr)
         end if
      end if

      if (any_c_smooths) then

         ! A_cc * x_c - computed every C richardson iteration
         call attach_block_product(air_data%A_cc(our_level), &
                  air_data%block_temp_coarse(AIR_MAT_SOL)%array(our_level), &
                  air_data%block_temp_coarse(AIR_MAT_RESIDUAL)%array(our_level), ierr)

         ! A_cf * x_f - computed once per C smooth
         call attach_block_product(air_data%A_cf(our_level), &
                  air_data%block_temp_fine(AIR_MAT_SOL)%array(our_level), &
                  air_data%block_temp_coarse(AIR_MAT_OFF_DIAG)%array(our_level), ierr)

         call MatGetType(air_data%inv_A_cc(our_level), inv_type, ierr)
         if (inv_type /= MATSHELL) then
            call attach_block_product(air_data%inv_A_cc(our_level), &
                     air_data%block_temp_coarse(AIR_MAT_RESIDUAL)%array(our_level), &
                     air_data%block_temp_coarse(AIR_MAT_TEMP)%array(our_level), ierr)
         end if
      end if

   end subroutine setup_air_block_products

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine MatISCopyLocalWrapper(air_data, our_level, fine, xfull_mat, mode, xreduced_mat, &
               x_temp_full_mat)

      ! The multiple rhs version of VecISCopyLocalWrapper - pulls the fine or
      ! coarse rows out of a dense block (SCATTER_REVERSE) or puts them back
      ! (SCATTER_FORWARD)
      ! Relies on having pre-built some things with the routine create_VecISCopyLocalWrapper

      ! ~~~~~~~~~~
      ! Input
      type(air_multigrid_data), intent(in) :: air_data
      integer, intent(in)                  :: our_level
      logical, intent(in)                  :: fine
      type(tMat), intent(inout)            :: xfull_mat, xreduced_mat
      type(tMat), optional, intent(inout)  :: x_temp_full_mat
      ScatterMode, intent(in)              :: mode

      PetscErrorCode :: ierr
#if defined(PETSC_HAVE_KOKKOS)
      integer(c_long_long) :: xfull_array, xreduced_array
      integer :: fine_int, mode_int, errorcode
      type(tMat) :: temp_mat
      PetscReal :: normy
#endif
      ! ~~~~~~~~~~

      ! On gpus without kokkos we can't touch the local arrays, so instead we use
      ! the identity restrictors/prolongators built in create_VecISCopyLocalWrapper
      ! These are the block versions of the matmults the single rhs version does
      if (.NOT. air_data%fast_veciscopy_exists) then

         if (mode == SCATTER_REVERSE) then

            if (fine) then
               call mat_product_block(air_data%i_fine_full(our_level), xfull_mat, &
                        xreduced_mat, .FALSE., ierr)
            else
               call mat_product_block(air_data%i_coarse_full(our_level), xfull_mat, &
                        xreduced_mat, .FALSE., ierr)
            end if

         ! SCATTER FORWARD
         else

            if (fine) then
               ! Copy x but only the non-fine rows of x are non-zero
               ! ie get x_c but in a block of full size
               call mat_product_block(air_data%i_coarse_full_full(our_level), xfull_mat, &
                        x_temp_full_mat, .FALSE., ierr)
               ! There is no MatMultTransposeAdd equivalent for a product, so we do
               ! the transpose product on its own and add the untouched rows back
               call mat_product_block(air_data%i_fine_full(our_level), xreduced_mat, &
                        xfull_mat, .TRUE., ierr)
            else
               ! Copy x but only the non-coarse rows of x are non-zero
               ! ie get x_f but in a block of full size
               call mat_product_block(air_data%i_fine_full_full(our_level), xfull_mat, &
                        x_temp_full_mat, .FALSE., ierr)
               call mat_product_block(air_data%i_coarse_full(our_level), xreduced_mat, &
                        xfull_mat, .TRUE., ierr)
            end if

            call MatAXPY(xfull_mat, PFLARE_ONE, x_temp_full_mat, SAME_NONZERO_PATTERN, ierr)
         end if

      ! Otherwise copy the rows we want directly out of the dense arrays
      else

#if defined(PETSC_HAVE_KOKKOS)

         ! The single rhs version decides by the vec type of the vec it is handed,
         ! but we can't do that here - the dense blocks in the mg hierarchy are
         ! built by petsc with MatDuplicate, which does not propagate the vec type,
         ! so a block can be backed by kokkos data and still report a standard vec
         ! type. The device IS views existing is the condition that matters (they
         ! are only built for a kokkos mat type in create_VecISCopyLocalWrapper) and
         ! the dense arrays we get back below always live in the default kokkos
         ! memory space - there is no MATDENSEKOKKOS, the blocks are host MATDENSE
         ! on a host kokkos backend and MATDENSECUDA/HIP on a device one
         if (c_associated(air_data%kokkos_is_views_handle)) then

            if (mode == SCATTER_REVERSE) then
               mode_int = 1
            else
               mode_int = 0
            end if
            fine_int = 0
            if (fine) fine_int = 1

            ! SCATTER FORWARD only writes the fine (or coarse) rows of the full
            ! block, so we have to keep a copy of it to run the cpu version on
            if (kokkos_debug() .AND. mode /= SCATTER_REVERSE) then
               call MatDuplicate(xfull_mat, MAT_COPY_VALUES, temp_mat, ierr)
            end if

            xfull_array = xfull_mat%v
            xreduced_array = xreduced_mat%v
            call mat_iscopy_local_kokkos(air_data%kokkos_is_views_handle, our_level, fine_int, xfull_array, &
                     mode_int, xreduced_array)

            ! If debugging do a comparison between CPU and Kokkos results
            if (kokkos_debug()) then

               if (mode == SCATTER_REVERSE) then

                  call MatDuplicate(xreduced_mat, MAT_DO_NOT_COPY_VALUES, temp_mat, ierr)
                  call mat_iscopy_local_host(air_data, our_level, fine, xfull_mat, mode, temp_mat)
                  call MatAXPY(temp_mat, PFLARE_MINUS_ONE, xreduced_mat, SAME_NONZERO_PATTERN, ierr)

               else

                  call mat_iscopy_local_host(air_data, our_level, fine, temp_mat, mode, xreduced_mat)
                  call MatAXPY(temp_mat, PFLARE_MINUS_ONE, xfull_mat, SAME_NONZERO_PATTERN, ierr)

               end if

               call MatNorm(temp_mat, NORM_FROBENIUS, normy, ierr)
               if (normy .gt. PFLARE_TOL_MATFREE_13) then
                  print *, "Kokkos and CPU versions of MatISCopyLocalWrapper do not match"
                  call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
               end if
               call MatDestroy(temp_mat, ierr)

            end if

         else
            call mat_iscopy_local_host(air_data, our_level, fine, xfull_mat, mode, xreduced_mat)
         end if
#else
         call mat_iscopy_local_host(air_data, our_level, fine, xfull_mat, mode, xreduced_mat)
#endif
      end if

   end subroutine MatISCopyLocalWrapper

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine mat_iscopy_local_host(air_data, our_level, fine, xfull_mat, mode, xreduced_mat)

      ! The host version of MatISCopyLocalWrapper - copies the fine or coarse rows
      ! we want directly out of the local dense arrays

      ! ~~~~~~~~~~
      ! Input
      type(air_multigrid_data), intent(in) :: air_data
      integer, intent(in)                  :: our_level
      logical, intent(in)                  :: fine
      type(tMat), intent(inout)            :: xfull_mat, xreduced_mat
      ScatterMode, intent(in)              :: mode

      PetscErrorCode :: ierr
      PetscInt :: global_row_start, global_row_end_plus_one
      PetscInt :: i_loc, j_loc, local_row
      PetscInt, pointer :: is_pointer(:)
      PetscScalar, pointer :: full_array(:,:), reduced_array(:,:)
      type(tIS) :: is_local
      ! ~~~~~~~~~~

      if (fine) then
         is_local = air_data%is_fine_index(our_level)
      else
         is_local = air_data%is_coarse_index(our_level)
      end if

      ! The IS holds global indices
      call MatGetOwnershipRange(xfull_mat, global_row_start, global_row_end_plus_one, ierr)
      call ISGetIndices(is_local, is_pointer, ierr)

      ! The petsc fortran interface only hands back a dense array when the
      ! leading dimension matches the number of local rows, so the arrays here
      ! are already (local rows, global columns) and need no separate lda
      if (mode == SCATTER_REVERSE) then

         call MatDenseGetArrayRead(xfull_mat, full_array, ierr)
         call MatDenseGetArray(xreduced_mat, reduced_array, ierr)

         do j_loc = 1, size(reduced_array, 2)
            do i_loc = 1, size(reduced_array, 1)
               local_row = is_pointer(i_loc) - global_row_start + 1
               reduced_array(i_loc, j_loc) = full_array(local_row, j_loc)
            end do
         end do

         call MatDenseRestoreArray(xreduced_mat, reduced_array, ierr)
         call MatDenseRestoreArrayRead(xfull_mat, full_array, ierr)

      ! SCATTER FORWARD - only the fine (or coarse) rows of the full block
      ! are written, everything else is left alone
      else

         call MatDenseGetArrayRead(xreduced_mat, reduced_array, ierr)
         call MatDenseGetArray(xfull_mat, full_array, ierr)

         do j_loc = 1, size(reduced_array, 2)
            do i_loc = 1, size(reduced_array, 1)
               local_row = is_pointer(i_loc) - global_row_start + 1
               full_array(local_row, j_loc) = reduced_array(i_loc, j_loc)
            end do
         end do

         call MatDenseRestoreArray(xfull_mat, full_array, ierr)
         call MatDenseRestoreArrayRead(xreduced_mat, reduced_array, ierr)

      end if

      call ISRestoreIndices(is_local, is_pointer, ierr)

   end subroutine mat_iscopy_local_host

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
      type(mat_ctxtype), pointer :: mat_ctx=>null()
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

   subroutine mg_FC_block_richardson(pc, b, x, r, rtol, abstol, dtol, maxits, guess_zero, its, conv_reason, ierr)

      ! The multiple rhs version of mg_FC_point_richardson - applies an FC block
      ! richardson to a whole dense block of right hand sides
      ! r is the block of work vectors and may be null, we never use it (just like
      ! the single rhs version never uses its work vector)

      ! ~~~~~~
      type(tPC) :: pc
      type(tMat) :: b, x, r
      PetscReal :: rtol, abstol, dtol
      PetscInt :: maxits, its
      PetscBool :: guess_zero
      PCRichardsonConvergedReason :: conv_reason
      PetscErrorCode :: ierr

      type(tMat) :: mat, pmat
      integer :: our_level, errorcode, i, smooth_its
      type(mat_ctxtype), pointer :: mat_ctx=>null()
      type(air_multigrid_data), pointer :: air_data
      PetscBool :: first_smooth

      ! ~~~~~~

      ! Set these for output
      ! have to return zero here!
      ierr = 0
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

            call f_smooths_block(b, x, guess_zero, first_smooth, air_data, our_level, smooth_its)

         ! Do consecutive C point smooths
         else

            call c_smooths_block(b, x, guess_zero, first_smooth, air_data, our_level, abs(smooth_its))
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

   end subroutine mg_FC_block_richardson

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
      call VecAXPY(air_data%temp_vecs_fine(4)%array(our_level), PFLARE_MINUS_ONE, &
               air_data%temp_vecs_fine(2)%array(our_level), ierr)                      

      ! Do all the consecutive F smooths
      do f_its = 1, its

         ! Then A_ff * x_f^n - this changes at each richardson iteration
         call MatMult(air_data%A_ff(our_level), air_data%temp_vecs_fine(1)%array(our_level), &
                     air_data%temp_vecs_fine(3)%array(our_level), ierr)          

         ! This is b_f - A_fc * x_c - A_ff * x_f^n
         call VecAYPX(air_data%temp_vecs_fine(3)%array(our_level), PFLARE_MINUS_ONE, &
                  air_data%temp_vecs_fine(4)%array(our_level), ierr)           

         ! ! Compute A_ff^{-1} ( b_f - A_fc * x_c - A_ff * x_f^n)
         call MatMult(air_data%inv_A_ff(our_level), air_data%temp_vecs_fine(3)%array(our_level), &
                     air_data%temp_vecs_fine(2)%array(our_level), ierr)    

         ! Compute x_f^n + A_ff^{-1} ( b_f - A_fc * x_c - A_ff * x_f^n)
         call VecAXPY(air_data%temp_vecs_fine(1)%array(our_level), PFLARE_ONE, &
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

   subroutine f_smooths_block(b, x, guess_zero, first_smooth, air_data, our_level, its)

      ! The multiple rhs version of f_smooths - applies consecutive F smooths to a
      ! whole dense block of right hand sides
      ! This is the same arithmetic as f_smooths with the matvecs replaced by
      ! sparse matrix - dense matrix products

      ! ~~~~~~
      type(tMat), intent(inout)               :: b, x
      type(air_multigrid_data), intent(inout) :: air_data
      integer, intent(in)                     :: our_level, its
      PetscBool, intent(in)                   :: guess_zero, first_smooth

      PetscErrorCode :: ierr
      integer :: f_its

      ! ~~~~~~

      ! Get out just the fine rows from b - this is b_f
      call MatISCopyLocalWrapper(air_data, our_level, .TRUE., b, &
               SCATTER_REVERSE, air_data%block_temp_fine(AIR_MAT_RHS)%array(our_level))

      ! If we haven't done any smooth before calling this F point smooth
      ! we need to pull out x_c^0 and x_f^0
      if (first_smooth) then

         ! Get out just the fine rows from x - this is x_f^0
         call MatISCopyLocalWrapper(air_data, our_level, .TRUE., x, &
                  SCATTER_REVERSE, air_data%block_temp_fine(AIR_MAT_SOL)%array(our_level))

         ! Get the coarse rows from x - this is x_c^0
         call MatISCopyLocalWrapper(air_data, our_level, .FALSE., x, &
                  SCATTER_REVERSE, air_data%block_temp_coarse(AIR_MAT_SOL)%array(our_level))

      end if

      ! Compute Afc * x_c^0 - this never changes
      ! The product was attached to the scratch by setup_air_block_products so
      ! only the numeric phase runs, here and below
      call MatProductNumeric(air_data%block_temp_fine(AIR_MAT_OFF_DIAG)%array(our_level), ierr)

      ! This is b_f - A_fc * x_c^0 - this never changes
      call MatAXPY(air_data%block_temp_fine(AIR_MAT_RHS)%array(our_level), PFLARE_MINUS_ONE, &
               air_data%block_temp_fine(AIR_MAT_OFF_DIAG)%array(our_level), SAME_NONZERO_PATTERN, ierr)

      ! Do all the consecutive F smooths
      do f_its = 1, its

         ! Then A_ff * x_f^n - this changes at each richardson iteration
         call MatProductNumeric(air_data%block_temp_fine(AIR_MAT_RESIDUAL)%array(our_level), ierr)

         ! This is b_f - A_fc * x_c - A_ff * x_f^n
         call MatAYPX(air_data%block_temp_fine(AIR_MAT_RESIDUAL)%array(our_level), PFLARE_MINUS_ONE, &
                  air_data%block_temp_fine(AIR_MAT_RHS)%array(our_level), SAME_NONZERO_PATTERN, ierr)

         ! ! Compute A_ff^{-1} ( b_f - A_fc * x_c - A_ff * x_f^n)
         call apply_inverse_block_cached(air_data%inv_A_ff(our_level), &
                  air_data%block_temp_fine(AIR_MAT_RESIDUAL)%array(our_level), &
                  air_data%block_temp_fine(AIR_MAT_TEMP)%array(our_level), ierr)

         ! Compute x_f^n + A_ff^{-1} ( b_f - A_fc * x_c - A_ff * x_f^n)
         call MatAXPY(air_data%block_temp_fine(AIR_MAT_SOL)%array(our_level), PFLARE_ONE, &
                  air_data%block_temp_fine(AIR_MAT_TEMP)%array(our_level), SAME_NONZERO_PATTERN, ierr)

      end do

      ! ~~~~~~~~
      ! Reverse put fine x_f back into x
      ! ~~~~~~~~
      call MatISCopyLocalWrapper(air_data, our_level, .TRUE., x, &
               SCATTER_FORWARD, air_data%block_temp_fine(AIR_MAT_SOL)%array(our_level), &
               air_data%block_temp_full(1)%array(our_level))

   end subroutine f_smooths_block

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
      call VecAXPY(air_data%temp_vecs_coarse(4)%array(our_level), PFLARE_MINUS_ONE, &
               air_data%temp_vecs_coarse(2)%array(our_level), ierr)  

      ! Do all the consecutive C smooths
      do c_its = 1, its

         ! Then A_cc * x_c^n - this changes at each richardson iteration
         call MatMult(air_data%A_cc(our_level), air_data%temp_vecs_coarse(1)%array(our_level), &
                     air_data%temp_vecs_coarse(3)%array(our_level), ierr)       

         ! This is b_c - A_cf * x_f^0 - A_cc * x_c^n
         call VecAYPX(air_data%temp_vecs_coarse(3)%array(our_level), PFLARE_MINUS_ONE, &
                  air_data%temp_vecs_coarse(4)%array(our_level), ierr)          

         ! ! Compute A_cc^{-1} (b_c - A_cf * x_f^0 - A_cc * x_c^n)
         call MatMult(air_data%inv_A_cc(our_level), air_data%temp_vecs_coarse(3)%array(our_level), &
                     air_data%temp_vecs_coarse(2)%array(our_level), ierr)    

         ! Compute x_c^n + A_cc^{-1} (b_c - A_cf * x_f^0 - A_cc * x_c^n)
         call VecAXPY(air_data%temp_vecs_coarse(1)%array(our_level), PFLARE_ONE, &
                     air_data%temp_vecs_coarse(2)%array(our_level), ierr)    
                     
      end do

      ! ~~~~~~~~
      ! Reverse put coarse x_c back into x
      ! ~~~~~~~~
      call VecISCopyLocalWrapper(air_data, our_level, .FALSE., x, &
               SCATTER_FORWARD, air_data%temp_vecs_coarse(1)%array(our_level), &
               air_data%temp_vecs(1)%array(our_level))          
      
   end subroutine c_smooths

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine c_smooths_block(b, x, guess_zero, first_smooth, air_data, our_level, its)

      ! The multiple rhs version of c_smooths - applies consecutive C smooths to a
      ! whole dense block of right hand sides

      ! ~~~~~~
      type(tMat), intent(inout)               :: b, x
      type(air_multigrid_data), intent(inout) :: air_data
      integer, intent(in)                     :: our_level, its
      PetscBool, intent(in)                   :: guess_zero, first_smooth

      PetscErrorCode :: ierr
      integer :: c_its

      ! ~~~~~~

      ! Get out just the coarse rows from b - this is b_c
      call MatISCopyLocalWrapper(air_data, our_level, .FALSE., b, &
               SCATTER_REVERSE, air_data%block_temp_coarse(AIR_MAT_RHS)%array(our_level))

      ! If we haven't done any smooth before calling this C point smooth
      ! we need to pull out x_c^0 and x_f^0
      if (first_smooth) then

            ! Get out just the fine rows from x - this is x_f^0
         call MatISCopyLocalWrapper(air_data, our_level, .TRUE., x, &
                  SCATTER_REVERSE, air_data%block_temp_fine(AIR_MAT_SOL)%array(our_level))

         ! Get the coarse rows from x - this is x_c^0
         call MatISCopyLocalWrapper(air_data, our_level, .FALSE., x, &
                  SCATTER_REVERSE, air_data%block_temp_coarse(AIR_MAT_SOL)%array(our_level))

      end if

      ! Compute Acf * x_f^0 - this never changes
      ! The product was attached to the scratch by setup_air_block_products so
      ! only the numeric phase runs, here and below
      call MatProductNumeric(air_data%block_temp_coarse(AIR_MAT_OFF_DIAG)%array(our_level), ierr)
      ! This is b_c - A_cf * x_f^0 - this never changes
      call MatAXPY(air_data%block_temp_coarse(AIR_MAT_RHS)%array(our_level), PFLARE_MINUS_ONE, &
               air_data%block_temp_coarse(AIR_MAT_OFF_DIAG)%array(our_level), SAME_NONZERO_PATTERN, ierr)

      ! Do all the consecutive C smooths
      do c_its = 1, its

         ! Then A_cc * x_c^n - this changes at each richardson iteration
         call MatProductNumeric(air_data%block_temp_coarse(AIR_MAT_RESIDUAL)%array(our_level), ierr)

         ! This is b_c - A_cf * x_f^0 - A_cc * x_c^n
         call MatAYPX(air_data%block_temp_coarse(AIR_MAT_RESIDUAL)%array(our_level), PFLARE_MINUS_ONE, &
                  air_data%block_temp_coarse(AIR_MAT_RHS)%array(our_level), SAME_NONZERO_PATTERN, ierr)

         ! ! Compute A_cc^{-1} (b_c - A_cf * x_f^0 - A_cc * x_c^n)
         call apply_inverse_block_cached(air_data%inv_A_cc(our_level), &
                  air_data%block_temp_coarse(AIR_MAT_RESIDUAL)%array(our_level), &
                  air_data%block_temp_coarse(AIR_MAT_TEMP)%array(our_level), ierr)

         ! Compute x_c^n + A_cc^{-1} (b_c - A_cf * x_f^0 - A_cc * x_c^n)
         call MatAXPY(air_data%block_temp_coarse(AIR_MAT_SOL)%array(our_level), PFLARE_ONE, &
                  air_data%block_temp_coarse(AIR_MAT_TEMP)%array(our_level), SAME_NONZERO_PATTERN, ierr)

      end do

      ! ~~~~~~~~
      ! Reverse put coarse x_c back into x
      ! ~~~~~~~~
      call MatISCopyLocalWrapper(air_data, our_level, .FALSE., x, &
               SCATTER_FORWARD, air_data%block_temp_coarse(AIR_MAT_SOL)%array(our_level), &
               air_data%block_temp_full(1)%array(our_level))

   end subroutine c_smooths_block

   !------------------------------------------------------------------------------------------------------------------------
   
end module fc_smooth

