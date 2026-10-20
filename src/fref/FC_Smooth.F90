module fc_smooth

   use petscksp
   use c_petsc_interfaces, only: create_VecISCopyLocal_kokkos, &
         set_VecISCopyLocal_kokkos_our_level, &
         destroy_VecISCopyLocal_kokkos, VecISCopyLocal_kokkos
   use air_data_type, only: air_multigrid_data
   use port_interfaces, only: generate_identity_rect, generate_identity_is, kokkos_debug
   use iso_c_binding, only: c_ptr, c_f_pointer
   use port_interfaces_poly, only: mat_ctx_get_our_level, mat_ctx_get_air_data
   use pflare_parameters, only: PFLARE_TOL_MATFREE_13, PFLARE_MINUS_ONE, PFLARE_ZERO, &
         PFLARE_ONE

#include "petsc/finclude/petscksp.h"
#include "petscconf.h"
                
   implicit none
   public

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! Functions involving the FC smoothing
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

   subroutine mg_coarse_shell_apply_transpose(pc, x, y, ierr)

      ! The transpose of mg_coarse_shell_apply, y = inv_A_ff(no_levels)^T x
      ! Our polynomial inverses all know how to do a transposed matvec, either
      ! natively when assembled or through a transposed twin when matrix-free

      ! ~~~~~~
      type(tPC)                             :: pc
      type(tVec)                            :: x, y
      PetscErrorCode, intent(out)           :: ierr

      type(air_multigrid_data), pointer     :: air_data => null()
      ! ~~~~~~

      call PCShellGetContext(pc, air_data, ierr)
      call MatMultTranspose(air_data%inv_A_ff(air_data%no_levels), x, y, ierr)

   end subroutine mg_coarse_shell_apply_transpose

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

   subroutine mg_smooth_shell_apply_transpose(pc, x, y, ierr)

      ! The transpose of mg_smooth_shell_apply - the Pmat of the smoother is
      ! inv_A_ff on that level, so this is just a transposed matvec with it
      ! This is what KSPSolveTranspose on the level smoothers needs when doing
      ! full smoothing up and down with a matrix-free inverse

      ! ~~~~~~
      type(tPC)                             :: pc
      type(tVec)                            :: x, y
      PetscErrorCode, intent(out)           :: ierr

      type(tMat) :: mat, pmat
      ! ~~~~~~

      ierr = 0
      call PCGetOperators(pc, mat, pmat, ierr)
      call MatMultTranspose(pmat, x, y, ierr)

   end subroutine mg_smooth_shell_apply_transpose

   !------------------------------------------------------------------------------------------------------------------------

   subroutine create_smooth_temp_vecs(air_data)

      ! Creates the temporary vectors we use during smoothing on every level that doesn't have them
      ! These are only ever scratch space within a single smooth (or residual) on one level,
      ! and we only smooth one level at a time, so rather than give every level its own memory
      ! each slot (e.g., temp_vecs_fine(2)) is a subvector of one pool shared across all the
      ! levels, sized for the biggest level on this rank
      ! The pools must not exist yet, ie they are destroyed whenever the temporary vectors are

      ! ~~~~~~~~~~
      ! Input
      type(air_multigrid_data), intent(inout) :: air_data

      integer :: our_level, i_loc, n_coarse, first_level, errorcode
      PetscInt :: local_rows, local_cols, max_fine, max_coarse, max_full
      PetscInt :: pool_start, pool_end_plus_one
      PetscInt, parameter :: one = 1
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX
      VecType :: vec_type
      type(tVec) :: temp_vec
      ! ~~~~~~~~~~

      ! Find the biggest local sizes on the levels we have to build
      first_level = -1
      max_fine = 0
      max_coarse = 0
      max_full = 0
      do our_level = 1, air_data%no_levels-1
         if (air_data%allocated_matrices_A_ff(our_level)) cycle
         if (first_level == -1) first_level = our_level

         call MatGetLocalSize(air_data%A_ff(our_level), local_rows, local_cols, ierr)
         max_fine = max(max_fine, local_cols)
         call MatGetLocalSize(air_data%A_fc(our_level), local_rows, local_cols, ierr)
         max_coarse = max(max_coarse, local_cols)
         if (.NOT. air_data%fast_veciscopy_exists) then
            call MatGetLocalSize(air_data%coarse_matrix(our_level), local_rows, local_cols, ierr)
            max_full = max(max_full, local_cols)
         end if
      end do
      if (first_level == -1) return

      ! The temporary vectors on every level are subvectors of the pools, so we can only
      ! build them when no level has them, ie the pools don't exist
      temp_vec = air_data%temp_vecs_fine_pool(1)
      if (.NOT. PetscObjectIsNull(temp_vec)) then
         print *, "The smoothing temporaries can only be created when no level has them"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! If we're doing C point smoothing we need some extra temporaries
      ! The transposed smooth and residual don't need any more than the
      ! forward ones do
      n_coarse = 1
      if (air_data%options%any_c_smooths .AND. &
               .NOT. air_data%options%full_smoothing_up_and_down) n_coarse = 4

      ! Create the pools with the vec type the matrices use
      call PetscObjectGetComm(air_data%A_fc(first_level), MPI_COMM_MATRIX, ierr)
      call MatGetVecType(air_data%A_fc(first_level), vec_type, ierr)
      do i_loc = 1, 4
         call VecCreate(MPI_COMM_MATRIX, air_data%temp_vecs_fine_pool(i_loc), ierr)
         call VecSetSizes(air_data%temp_vecs_fine_pool(i_loc), max_fine, PETSC_DECIDE, ierr)
         call VecSetType(air_data%temp_vecs_fine_pool(i_loc), vec_type, ierr)
      end do
      do i_loc = 1, n_coarse
         call VecCreate(MPI_COMM_MATRIX, air_data%temp_vecs_coarse_pool(i_loc), ierr)
         call VecSetSizes(air_data%temp_vecs_coarse_pool(i_loc), max_coarse, PETSC_DECIDE, ierr)
         call VecSetType(air_data%temp_vecs_coarse_pool(i_loc), vec_type, ierr)
      end do
      if (.NOT. air_data%fast_veciscopy_exists) then
         call VecCreate(MPI_COMM_MATRIX, air_data%temp_vecs_pool(1), ierr)
         call VecSetSizes(air_data%temp_vecs_pool(1), max_full, PETSC_DECIDE, ierr)
         call VecSetType(air_data%temp_vecs_pool(1), vec_type, ierr)
      end if

      ! On each level take the start of each pool as the temporary vectors
      ! The IS is contiguous so the subvector shares the pool's memory
      do our_level = 1, air_data%no_levels-1
         if (air_data%allocated_matrices_A_ff(our_level)) cycle

         call MatGetLocalSize(air_data%A_ff(our_level), local_rows, local_cols, ierr)
         call VecGetOwnershipRange(air_data%temp_vecs_fine_pool(1), pool_start, pool_end_plus_one, ierr)
         call ISCreateStride(MPI_COMM_MATRIX, local_cols, pool_start, one, &
                  air_data%temp_vecs_fine_is(our_level), ierr)
         do i_loc = 1, 4
            call VecGetSubVector(air_data%temp_vecs_fine_pool(i_loc), air_data%temp_vecs_fine_is(our_level), &
                     air_data%temp_vecs_fine(i_loc)%array(our_level), ierr)
         end do

         call MatGetLocalSize(air_data%A_fc(our_level), local_rows, local_cols, ierr)
         call VecGetOwnershipRange(air_data%temp_vecs_coarse_pool(1), pool_start, pool_end_plus_one, ierr)
         call ISCreateStride(MPI_COMM_MATRIX, local_cols, pool_start, one, &
                  air_data%temp_vecs_coarse_is(our_level), ierr)
         do i_loc = 1, n_coarse
            call VecGetSubVector(air_data%temp_vecs_coarse_pool(i_loc), air_data%temp_vecs_coarse_is(our_level), &
                     air_data%temp_vecs_coarse(i_loc)%array(our_level), ierr)
         end do

         if (.NOT. air_data%fast_veciscopy_exists) then
            call MatGetLocalSize(air_data%coarse_matrix(our_level), local_rows, local_cols, ierr)
            call VecGetOwnershipRange(air_data%temp_vecs_pool(1), pool_start, pool_end_plus_one, ierr)
            call ISCreateStride(MPI_COMM_MATRIX, local_cols, pool_start, one, &
                     air_data%temp_vecs_is(our_level), ierr)
            call VecGetSubVector(air_data%temp_vecs_pool(1), air_data%temp_vecs_is(our_level), &
                     air_data%temp_vecs(1)%array(our_level), ierr)
         end if
      end do

   end subroutine create_smooth_temp_vecs

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
         call generate_identity_rect(input_mat, &
                  air_data%IS_fine_index(our_level), &
                  air_data%i_fine_full(our_level))

         ! Build coarse to full injector
         call generate_identity_rect(input_mat, &
                  air_data%IS_coarse_index(our_level), &
                  air_data%i_coarse_full(our_level))
                  
         ! Build identity that sets fine in full to zero
         call generate_identity_is(input_mat, air_data%IS_coarse_index(our_level), &
                  air_data%i_coarse_full_full(our_level))               

         ! Build identity that sets coarse in full to zero
         ! This is needed whenever we scatter coarse values back into a full sized
         ! vector, which the C point smooths do and so does the transposed residual,
         ! as that has to write r_c back even with no C smooths
         call generate_identity_is(input_mat, air_data%IS_fine_index(our_level), &
               air_data%i_fine_full_full(our_level))

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
         call MatDestroy(air_data%i_coarse_full_full(our_level), ierr)
         call MatDestroy(air_data%i_fine_full_full(our_level), ierr)

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

               ! The device IS views only exist if the matrix type was kokkos in
               ! create_VecISCopyLocalWrapper, so fall back to VecISCopy if not
               ! (e.g., kokkos vecs with an aij matrix)
               call VecGetType(vfull, vec_type, ierr)
               if ((vec_type == "seqkokkos" .OR. vec_type == "mpikokkos" .OR. &
                        vec_type == "kokkos") .AND. &
                        c_associated(air_data%kokkos_is_views_handle)) then

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

               ! The device IS views only exist if the matrix type was kokkos in
               ! create_VecISCopyLocalWrapper, so fall back to VecISCopy if not
               ! (e.g., kokkos vecs with an aij matrix)
               call VecGetType(vfull, vec_type, ierr)
               if ((vec_type == "seqkokkos" .OR. vec_type == "mpikokkos" .OR. &
                        vec_type == "kokkos") .AND. &
                        c_associated(air_data%kokkos_is_views_handle)) then

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

               ! The device IS views only exist if the matrix type was kokkos in
               ! create_VecISCopyLocalWrapper, so fall back to VecISCopy if not
               ! (e.g., kokkos vecs with an aij matrix)
               call VecGetType(vfull, vec_type, ierr)
               if ((vec_type == "seqkokkos" .OR. vec_type == "mpikokkos" .OR. &
                        vec_type == "kokkos") .AND. &
                        c_associated(air_data%kokkos_is_views_handle)) then

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

               ! The device IS views only exist if the matrix type was kokkos in
               ! create_VecISCopyLocalWrapper, so fall back to VecISCopy if not
               ! (e.g., kokkos vecs with an aij matrix)
               call VecGetType(vfull, vec_type, ierr)
               if ((vec_type == "seqkokkos" .OR. vec_type == "mpikokkos" .OR. &
                        vec_type == "kokkos") .AND. &
                        c_associated(air_data%kokkos_is_views_handle)) then

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
      type(c_ptr) :: air_data_ptr
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
      call mat_ctx_get_our_level(mat, our_level)
      call mat_ctx_get_air_data(mat, air_data_ptr)
      call c_f_pointer(air_data_ptr, air_data)

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

   subroutine mg_FC_point_apply_transpose(pc, b, x, ierr)

      ! This applies the transpose of the F-C point smooth that mg_FC_point_richardson
      ! applies on this level, from a zero initial guess, x = N^T b
      ! This is the PCApplyTranspose of the smoother shell, which is what petsc's
      ! transposed cycle drives with a KSPSolveTranspose - a transposed solve skips
      ! the PCApplyRichardson fast path the forward smooth lives in, and with one
      ! iteration and no norms the richardson is then just x = PCApplyTranspose(b)
      !
      ! The forward smooth is a product of relaxations x += N_i (b - A x), so its
      ! transpose is the same relaxations with A^T and N_i^T in place of A and N_i,
      ! applied in the reverse order. The F row of A^T is [A_ff^T, A_cf^T] and the C
      ! row is [A_fc^T, A_cc^T], so the transposed F relaxation is
      !      x_f += inv_A_ff^T (b_f - A_ff^T x_f - A_cf^T x_c)
      ! and the transposed C relaxation is
      !      x_c += inv_A_cc^T (b_c - A_fc^T x_f - A_cc^T x_c)
      ! Just like the forward smooth this only ever touches the blocks of A, never
      ! the full operator on this level, which doesn't exist when we're F-C smoothing

      ! ~~~~~~
      type(tPC) :: pc
      type(tVec) :: b, x
      PetscErrorCode :: ierr

      type(tMat) :: mat, pmat
      integer :: our_level, i, no_blocks, smooth_its
      type(c_ptr) :: air_data_ptr
      type(air_multigrid_data), pointer :: air_data
      logical :: x_f_zero, x_c_zero

      ! ~~~~~~

      ! Get the level, exactly as mg_FC_point_richardson does
      call PCGetOperators(pc, mat, pmat, ierr)
      call mat_ctx_get_our_level(mat, our_level)
      call mat_ctx_get_air_data(mat, air_data_ptr)
      call c_f_pointer(air_data_ptr, air_data)

      ! How many blocks of smooths we do on this level - the ordering stops at
      ! the first zero entry
      no_blocks = 0
      do i = 1, size(air_data%smooth_order_levels(our_level)%array)
         if (air_data%smooth_order_levels(our_level)%array(i) == 0) exit
         no_blocks = i
      end do

      ! The solution starts at zero and we track which halves of it are still
      ! identically zero, as any term multiplying a zero half can just be skipped.
      ! With the default F point smoothing x_c never becomes nonzero, so we never
      ! touch A_cf (which doesn't exist then) and do no coarse sized work at all
      x_f_zero = .TRUE.
      x_c_zero = .TRUE.

      ! Reverse order compared to mg_FC_point_richardson
      do i = no_blocks, 1, -1

         smooth_its = air_data%smooth_order_levels(our_level)%array(i)

         ! Do consecutive F point smooths
         if (smooth_its > 0) then

            call f_smooths_transpose(b, air_data, our_level, smooth_its, x_f_zero, x_c_zero)

         ! Do consecutive C point smooths
         else

            call c_smooths_transpose(b, air_data, our_level, abs(smooth_its), x_f_zero, x_c_zero)
         end if

      end do

      ! ~~~~~~~~
      ! Put the solution back into the full sized vector. x is a ksp work vector so
      ! every entry has to be written - zero it and then scatter in whichever halves
      ! are nonzero
      ! ~~~~~~~~
      call VecSet(x, PFLARE_ZERO, ierr)
      if (.NOT. x_f_zero) then
         call VecISCopyLocalWrapper(air_data, our_level, .TRUE., x, &
                  SCATTER_FORWARD, air_data%temp_vecs_fine(1)%array(our_level), &
                  air_data%temp_vecs(1)%array(our_level))
      end if
      if (.NOT. x_c_zero) then
         call VecISCopyLocalWrapper(air_data, our_level, .FALSE., x, &
                  SCATTER_FORWARD, air_data%temp_vecs_coarse(1)%array(our_level), &
                  air_data%temp_vecs(1)%array(our_level))
      end if

      ! have to return zero here!
      ierr = 0

   end subroutine mg_FC_point_apply_transpose

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine f_smooths_transpose(b, air_data, our_level, its, x_f_zero, x_c_zero)

      ! This applies the transpose of consecutive F smooths
      !      x_f = x_f + inv_A_ff^T (b_f - A_ff^T x_f - A_cf^T x_c)
      ! x_c doesn't change in here, so b_f - A_cf^T x_c is computed once for the
      ! whole block, exactly as the forward f_smooths does with b_f - A_fc * x_c

      ! ~~~~~~
      type(tVec), intent(inout)               :: b
      type(air_multigrid_data), intent(inout) :: air_data
      integer, intent(in)                     :: our_level, its
      logical, intent(inout)                  :: x_f_zero
      logical, intent(in)                     :: x_c_zero

      PetscErrorCode :: ierr
      integer :: f_its

      ! ~~~~~~

      ! Get out just the fine points from b - this is b_f
      call VecISCopyLocalWrapper(air_data, our_level, .TRUE., b, &
               SCATTER_REVERSE, air_data%temp_vecs_fine(4)%array(our_level))

      ! This is b_f - A_cf^T * x_c - this never changes
      ! A_cf only exists when we're doing C point smooths, which is also the only
      ! way x_c can be nonzero
      if (.NOT. x_c_zero) then
         call MatMultTranspose(air_data%A_cf(our_level), air_data%temp_vecs_coarse(1)%array(our_level), &
                  air_data%temp_vecs_fine(2)%array(our_level), ierr)
         call VecAXPY(air_data%temp_vecs_fine(4)%array(our_level), PFLARE_MINUS_ONE, &
                  air_data%temp_vecs_fine(2)%array(our_level), ierr)
      end if

      ! Do all the consecutive F smooths
      do f_its = 1, its

         ! If x_f is still zero there is no A_ff^T * x_f term and nothing to add to,
         ! so the whole relaxation is just x_f = A_ff^{-T} (b_f - A_cf^T * x_c)
         if (x_f_zero) then

            call MatMultTranspose(air_data%inv_A_ff(our_level), air_data%temp_vecs_fine(4)%array(our_level), &
                     air_data%temp_vecs_fine(1)%array(our_level), ierr)
            x_f_zero = .FALSE.

         else

            ! Then A_ff^T * x_f^n - this changes at each richardson iteration
            call MatMultTranspose(air_data%A_ff(our_level), air_data%temp_vecs_fine(1)%array(our_level), &
                     air_data%temp_vecs_fine(3)%array(our_level), ierr)

            ! This is b_f - A_cf^T * x_c - A_ff^T * x_f^n
            call VecAYPX(air_data%temp_vecs_fine(3)%array(our_level), PFLARE_MINUS_ONE, &
                     air_data%temp_vecs_fine(4)%array(our_level), ierr)

            ! Compute A_ff^{-T} ( b_f - A_cf^T * x_c - A_ff^T * x_f^n)
            call MatMultTranspose(air_data%inv_A_ff(our_level), air_data%temp_vecs_fine(3)%array(our_level), &
                     air_data%temp_vecs_fine(2)%array(our_level), ierr)

            ! Compute x_f^n + A_ff^{-T} ( b_f - A_cf^T * x_c - A_ff^T * x_f^n)
            call VecAXPY(air_data%temp_vecs_fine(1)%array(our_level), PFLARE_ONE, &
                     air_data%temp_vecs_fine(2)%array(our_level), ierr)
         end if

      end do

   end subroutine f_smooths_transpose

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine c_smooths_transpose(b, air_data, our_level, its, x_f_zero, x_c_zero)

      ! This applies the transpose of consecutive C smooths - the C point mirror of
      ! f_smooths_transpose
      !      x_c = x_c + inv_A_cc^T (b_c - A_cc^T x_c - A_fc^T x_f)

      ! ~~~~~~
      type(tVec), intent(inout)               :: b
      type(air_multigrid_data), intent(inout) :: air_data
      integer, intent(in)                     :: our_level, its
      logical, intent(in)                     :: x_f_zero
      logical, intent(inout)                  :: x_c_zero

      PetscErrorCode :: ierr
      integer :: c_its

      ! ~~~~~~

      ! Get out just the coarse points from b - this is b_c
      call VecISCopyLocalWrapper(air_data, our_level, .FALSE., b, &
               SCATTER_REVERSE, air_data%temp_vecs_coarse(4)%array(our_level))

      ! This is b_c - A_fc^T * x_f - this never changes
      if (.NOT. x_f_zero) then
         call MatMultTranspose(air_data%A_fc(our_level), air_data%temp_vecs_fine(1)%array(our_level), &
                  air_data%temp_vecs_coarse(2)%array(our_level), ierr)
         call VecAXPY(air_data%temp_vecs_coarse(4)%array(our_level), PFLARE_MINUS_ONE, &
                  air_data%temp_vecs_coarse(2)%array(our_level), ierr)
      end if

      ! Do all the consecutive C smooths
      do c_its = 1, its

         ! As in f_smooths_transpose, a zero x_c makes the relaxation a single matmult
         if (x_c_zero) then

            call MatMultTranspose(air_data%inv_A_cc(our_level), air_data%temp_vecs_coarse(4)%array(our_level), &
                     air_data%temp_vecs_coarse(1)%array(our_level), ierr)
            x_c_zero = .FALSE.

         else

            ! Then A_cc^T * x_c^n - this changes at each richardson iteration
            call MatMultTranspose(air_data%A_cc(our_level), air_data%temp_vecs_coarse(1)%array(our_level), &
                     air_data%temp_vecs_coarse(3)%array(our_level), ierr)

            ! This is b_c - A_fc^T * x_f - A_cc^T * x_c^n
            call VecAYPX(air_data%temp_vecs_coarse(3)%array(our_level), PFLARE_MINUS_ONE, &
                     air_data%temp_vecs_coarse(4)%array(our_level), ierr)

            ! Compute A_cc^{-T} (b_c - A_fc^T * x_f - A_cc^T * x_c^n)
            call MatMultTranspose(air_data%inv_A_cc(our_level), air_data%temp_vecs_coarse(3)%array(our_level), &
                     air_data%temp_vecs_coarse(2)%array(our_level), ierr)

            ! Compute x_c^n + A_cc^{-T} (b_c - A_fc^T * x_f - A_cc^T * x_c^n)
            call VecAXPY(air_data%temp_vecs_coarse(1)%array(our_level), PFLARE_ONE, &
                     air_data%temp_vecs_coarse(2)%array(our_level), ierr)
         end if

      end do

   end subroutine c_smooths_transpose

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine mg_FC_point_residual_transpose(mat, b, x, r, ierr)

      ! Computes the transposed residual r = b - A^T x on a level, which is what
      ! petsc's transposed cycle needs after each transposed smooth
      ! The full operator on this level doesn't exist when we're F-C smoothing - mat
      ! is the empty placeholder matshell that just tells us which level we're on -
      ! so we build the residual out of the blocks of A instead
      !      r_f = b_f - A_ff^T x_f - A_cf^T x_c
      !      r_c = b_c - A_fc^T x_f - A_cc^T x_c
      ! The x_c terms are only there when we're doing C point smooths. Without them
      ! A_cf and A_cc don't exist, but neither are they needed: the only x this is
      ! ever called with is the output of mg_FC_point_apply_transpose, whose coarse
      ! half is then identically zero

      ! ~~~~~~
      type(tMat), intent(inout)     :: mat
      type(tVec), intent(inout)     :: b, x, r
      PetscErrorCode, intent(out)   :: ierr

      integer :: our_level
      type(c_ptr) :: air_data_ptr
      type(air_multigrid_data), pointer :: air_data

      ! ~~~~~~

      ierr = 0

      ! Get the level out of the placeholder matshell
      call mat_ctx_get_our_level(mat, our_level)
      call mat_ctx_get_air_data(mat, air_data_ptr)
      call c_f_pointer(air_data_ptr, air_data)

      ! We negate x and use MatMultTransposeAdd throughout, which keeps us to the
      ! temporaries the forward smooth already has

      ! Get out -x_f
      call VecISCopyLocalWrapper(air_data, our_level, .TRUE., x, &
               SCATTER_REVERSE, air_data%temp_vecs_fine(2)%array(our_level))
      call VecScale(air_data%temp_vecs_fine(2)%array(our_level), PFLARE_MINUS_ONE, ierr)

      ! Get out b_f and b_c
      call VecISCopyLocalWrapper(air_data, our_level, .TRUE., b, &
               SCATTER_REVERSE, air_data%temp_vecs_fine(4)%array(our_level))
      call VecISCopyLocalWrapper(air_data, our_level, .FALSE., b, &
               SCATTER_REVERSE, air_data%temp_vecs_coarse(1)%array(our_level))

      ! r_f = b_f - A_ff^T * x_f
      call MatMultTransposeAdd(air_data%A_ff(our_level), air_data%temp_vecs_fine(2)%array(our_level), &
               air_data%temp_vecs_fine(4)%array(our_level), air_data%temp_vecs_fine(3)%array(our_level), ierr)
      ! r_c = b_c - A_fc^T * x_f, built up in place
      call MatMultTransposeAdd(air_data%A_fc(our_level), air_data%temp_vecs_fine(2)%array(our_level), &
               air_data%temp_vecs_coarse(1)%array(our_level), air_data%temp_vecs_coarse(1)%array(our_level), ierr)

      if (air_data%options%any_c_smooths) then

         ! Get out -x_c
         call VecISCopyLocalWrapper(air_data, our_level, .FALSE., x, &
                  SCATTER_REVERSE, air_data%temp_vecs_coarse(2)%array(our_level))
         call VecScale(air_data%temp_vecs_coarse(2)%array(our_level), PFLARE_MINUS_ONE, ierr)

         ! r_f = r_f - A_cf^T * x_c
         call MatMultTransposeAdd(air_data%A_cf(our_level), air_data%temp_vecs_coarse(2)%array(our_level), &
                  air_data%temp_vecs_fine(3)%array(our_level), air_data%temp_vecs_fine(3)%array(our_level), ierr)
         ! r_c = r_c - A_cc^T * x_c
         call MatMultTransposeAdd(air_data%A_cc(our_level), air_data%temp_vecs_coarse(2)%array(our_level), &
                  air_data%temp_vecs_coarse(1)%array(our_level), air_data%temp_vecs_coarse(1)%array(our_level), ierr)
      end if

      ! ~~~~~~~~
      ! Put both halves back into the full sized residual - every entry of r has to
      ! be written, and the scatter of one half preserves whatever the other holds,
      ! so zero r first
      ! ~~~~~~~~
      call VecSet(r, PFLARE_ZERO, ierr)
      call VecISCopyLocalWrapper(air_data, our_level, .TRUE., r, &
               SCATTER_FORWARD, air_data%temp_vecs_fine(3)%array(our_level), &
               air_data%temp_vecs(1)%array(our_level))
      call VecISCopyLocalWrapper(air_data, our_level, .FALSE., r, &
               SCATTER_FORWARD, air_data%temp_vecs_coarse(1)%array(our_level), &
               air_data%temp_vecs(1)%array(our_level))

   end subroutine mg_FC_point_residual_transpose

   ! -------------------------------------------------------------------------------------------------------------------------------

end module fc_smooth

