module cf_splitting

   use petscmat
   use pflare_parameters, only: C_POINT, F_POINT, PFLARE_CR_MAX_ITS, &
            PFLARE_CR_POLY_ORDER, PFLAREINV_ARNOLDI
   use pmisr_module, only: pmisr
   use ddc_module, only: ddc
   use cr_splitting, only: cr_pass
   use sabs, only: generate_sabs
   use c_petsc_interfaces, only: create_cf_is_kokkos, delete_device_cf_markers, delete_device_diag_dom_ratio
   use aggregation, only: generate_serial_aggregation
   use petsc_helper, only: MatAXPYWrapper, MatSetAllValues, kokkos_debug, remove_small_from_sparse

#include "petsc/finclude/petscmat.h"

   implicit none
   public   

   PetscEnum, parameter :: CF_PMISR_DDC=0
   PetscEnum, parameter :: CF_DIAG_DOM=1
   PetscEnum, parameter :: CF_PMIS=2
   PetscEnum, parameter :: CF_PMIS_DIST2=3
   PetscEnum, parameter :: CF_AGG=4
   PetscEnum, parameter :: CF_PMIS_AGG=5
   PetscEnum, parameter :: CF_CR=6
   
   contains

! -------------------------------------------------------------------------------------------------------------------------------
   
   subroutine create_cf_is(input_mat, cf_markers_local, is_fine, is_coarse)

      ! Creates IS's for C and F points given a cf_markers_local array
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in) :: input_mat
      integer, allocatable, dimension(:), intent(in) :: cf_markers_local
      type(tIS), intent(inout) :: is_fine, is_coarse

      PetscInt, dimension(:), allocatable :: fine_indices, coarse_indices
      PetscInt :: global_row_start, global_row_end_plus_one
      PetscInt :: fine_counter, coarse_counter, i_loc
      PetscInt :: n, n_fine, n_coarse
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX      
      
      ! ~~~~~~~~~~

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)    
      call MatGetOwnershipRange(input_mat, global_row_start, global_row_end_plus_one, ierr)   

      ! This is the local_rows
      n = size(cf_markers_local)
      n_fine = count(cf_markers_local == F_POINT)
      n_coarse = count(cf_markers_local == C_POINT) 

      allocate(fine_indices(n_fine))
      allocate(coarse_indices(n_coarse))

      fine_counter = 0
      coarse_counter = 0
      do i_loc = 1, n
         if (cf_markers_local(i_loc) == F_POINT) then
            fine_counter = fine_counter + 1
            ! Remember petsc indices start at zero
            fine_indices(fine_counter) = global_row_start + i_loc-1
         else
            coarse_counter = coarse_counter + 1
            ! Remember petsc indices start at zero
            coarse_indices(coarse_counter) = global_row_start + i_loc-1
         end if
      end do   
      
      call ISCreateGeneral(MPI_COMM_MATRIX, fine_counter, &
            fine_indices(1:fine_counter), &
            PETSC_COPY_VALUES, is_fine, ierr)  
      call ISCreateGeneral(MPI_COMM_MATRIX, coarse_counter, &
            coarse_indices(1:coarse_counter), &
            PETSC_COPY_VALUES, is_coarse, ierr)   

      deallocate(fine_indices, coarse_indices)
         
   end subroutine create_cf_is
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine first_pass_splitting(input_mat, skip_symmetrize, strong_threshold, &
                  max_luby_steps, cf_splitting_type, cf_markers_local)

      ! Compute a strength matrix and then call the first pass of CF splitting
      ! skip_symmetrize skips symmetrizing the strength matrix for the PMISR DDC,
      ! diag dom and aggregation splittings - the PMIS-based splittings always
      ! symmetrize as they are defined on the symmetrized graph

      ! ~~~~~~
      type(tMat), target, intent(in)      :: input_mat
      logical, intent(in)                 :: skip_symmetrize
      PetscReal, intent(in)                    :: strong_threshold
      integer, intent(in)                 :: max_luby_steps, cf_splitting_type
      integer, dimension(:), allocatable, intent(inout) :: cf_markers_local

      ! Local
      PetscInt :: global_row_start, global_row_end_plus_one, i_loc
      ! PetscInt :: counter, local_c_size
      ! PetscInt :: global_row_start_dist2, global_row_end_plus_one_dist2
      PetscInt :: local_rows, local_cols, ncols
      integer :: errorcode, comm_size
      MPIU_Comm :: MPI_COMM_MATRIX
      PetscErrorCode :: ierr
      type(tMat) :: strength_mat
      ! type(tMat) :: prolongators, strength_mat_c, strength_mat_fc, temp_mat
      ! type(tIS) :: is_fine, is_coarse, zero_diags
      ! integer, dimension(:), allocatable :: cf_markers_local_c
      PetscInt, dimension(:), allocatable :: aggregates
      ! PetscInt, dimension(:), pointer :: is_pointer_coarse, is_pointer_fine, zero_diags_pointer
      PetscInt, parameter :: nz_ignore = -1
      type(tMat) :: Ad, Ao
      PetscInt, dimension(:), pointer :: colmap

      ! ~~~~~~  

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)  
      call MatGetLocalSize(input_mat, local_rows, local_cols, ierr)
      call MatGetOwnershipRange(input_mat, global_row_start, global_row_end_plus_one, ierr)   

      ! ~~~~~~~~~~~~
      ! Generate the strength matrix
      ! ~~~~~~~~~~~~

      ! Use classical strength of connection
      if (cf_splitting_type == CF_PMIS_DIST2) then

         ! Square the strength of connection graph - S'S + S
         ! Note we are symmetrizing the strength matrix here
         call generate_sabs(input_mat, strong_threshold, .TRUE., .TRUE., strength_mat)

      else if (cf_splitting_type == CF_PMIS) then

         ! Note we are symmetrizing the strength matrix here
         call generate_sabs(input_mat, strong_threshold, .TRUE., .FALSE., strength_mat)

      else if (cf_splitting_type == CF_DIAG_DOM) then
         ! Symmetrize the strength matrix unless asked to skip it

         ! Tried to generate a strength matrix based on the relative size compared to the
         ! diagonal, but it produces a worse initial coarsening when fed to PMISR, making
         ! the DDC cleanup take a lot more work
         !call generate_sabs(input_mat, strong_threshold, .NOT. skip_symmetrize, .FALSE., strength_mat, &
         !         allow_diag_strength = .TRUE.)
         call generate_sabs(input_mat, strong_threshold, .NOT. skip_symmetrize, .FALSE., strength_mat)

      ! PMISR DDC and Aggregation
      else
         ! Symmetrize the strength matrix unless asked to skip it
         call generate_sabs(input_mat, strong_threshold, .NOT. skip_symmetrize, .FALSE., strength_mat)
      end if

      ! ~~~~~~~~~~~~
      ! Do the first pass splitting
      ! ~~~~~~~~~~~~

      ! PMISR
      if (cf_splitting_type == CF_PMISR_DDC .OR. cf_splitting_type == CF_DIAG_DOM) then

         call pmisr(strength_mat, max_luby_steps, .FALSE., cf_markers_local)

      ! Distance 1 PMIS
      else if (cf_splitting_type == CF_PMIS) then

         call pmisr(strength_mat, max_luby_steps, .TRUE., cf_markers_local)

      ! Distance 2 PMIS
      else if (cf_splitting_type == CF_PMIS_DIST2) then

         ! As we have generated S'S + S for the strength matrix, this will do distance 2 PMIS
         call pmisr(strength_mat, max_luby_steps, .TRUE., cf_markers_local)

      ! PMIS on boundary nodes then processor local aggregation
      else if (cf_splitting_type == CF_PMIS_AGG) then
        
         ! In parallel we do distance 1 pmis on boundary nodes, then do serial aggregation 
         if (comm_size /= 1) then

            ! Do distance 1 PMIS
            call pmisr(strength_mat, max_luby_steps, .TRUE., cf_markers_local)

            ! Get the sequential part of the matrix
            call MatMPIAIJGetSeqAIJ(strength_mat, Ad, Ao, colmap, ierr) 
            
            ! For any local node that doesn't touch a boundary node, we set the 
            ! cf_markers back to unassigned and leave them to be done by the aggregation
            do i_loc = 1, local_rows

               ! Get how many non-zeros are in the off-diagonal 
               call MatGetRow(Ao, i_loc-1, ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
               ! If we don't touch anything off-processor, its local and set it to unassigned
               if (ncols == 0) cf_markers_local(i_loc) = 0
               call MatRestoreRow(Ao, i_loc-1, ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
            end do
            
         else
            Ad = strength_mat
         end if

         ! Call an aggregation algorithm but only on the local Ad
         call generate_serial_aggregation(Ad, cf_markers_local, aggregates)
         deallocate(aggregates)

      ! Processor local aggregation
      else if (cf_splitting_type == CF_AGG) then
        
         if (comm_size /= 1) then
            call MatMPIAIJGetSeqAIJ(strength_mat, Ad, Ao, colmap, ierr) 
         else
            Ad = strength_mat
         end if

         ! Call an aggregation algorithm but only on the local Ad
         call generate_serial_aggregation(Ad, cf_markers_local, aggregates)
         deallocate(aggregates)         

      else
         ! Just do the same thing as distance 2 multiple times if you want higher distance
         print *, "Unknown CF splitting algorithm"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! Destroy the strength matrix
      call MatDestroy(strength_mat, ierr)

   end subroutine first_pass_splitting


! -------------------------------------------------------------------------------------------------------------------------------

   subroutine compute_cf_splitting(input_mat, skip_symmetrize, &
                     strong_threshold, max_luby_steps, &
                     cf_splitting_type, ddc_its, fraction_swap, &
                     is_fine, is_coarse, &
                     cr_inverse_type, cr_poly_order, cr_inverse_sparsity_order, &
                     cr_diag_scale_polys)

      ! Computes a CF splitting and returns the F and C point ISs
      ! skip_symmetrize skips symmetrizing the strength matrix (it does not
      ! apply to the PMIS-based splittings, which are defined on the
      ! symmetrized graph and always symmetrize)
      ! The optional cr_* arguments only apply to CF_CR and set the
      ! approximate inverse used as the CR relaxation - PCAIR passes its
      ! Aff inverse settings so the CR rate certifies the F-solve actually
      ! applied in the hierarchy; when absent an assembled arnoldi-basis
      ! GMRES polynomial of order PFLARE_CR_POLY_ORDER with sparsity order 1
      ! and no diagonal scaling is used, matching the PCAIR defaults

      ! ~~~~~~
      type(tMat), target, intent(in)      :: input_mat
      logical, intent(in)                 :: skip_symmetrize
      PetscReal, intent(in)               :: strong_threshold
      integer, intent(in)                 :: max_luby_steps, cf_splitting_type, ddc_its
      PetscReal, intent(in)               :: fraction_swap
      type(tIS), intent(inout)            :: is_fine, is_coarse
      integer, intent(in), optional       :: cr_inverse_type, cr_poly_order
      integer, intent(in), optional       :: cr_inverse_sparsity_order
      logical, intent(in), optional       :: cr_diag_scale_polys

      PetscErrorCode :: ierr
      integer, dimension(:), allocatable, target :: cf_markers_local
      integer :: its, ddc_its_max
      logical :: need_intermediate_is
      PetscReal :: max_dd_ratio_achieved, cr_rate_achieved
      PetscInt :: local_rows, local_cols, n_swapped
      integer :: cr_inverse_type_use, cr_poly_order_use, cr_sparsity_order_use
      logical :: cr_diag_scale_use
#if defined(PETSC_HAVE_KOKKOS)                     
      MatType :: mat_type
      integer(c_long_long) :: A_array, is_fine_array, is_coarse_array
      type(tIS) :: is_fine_kokkos, is_coarse_kokkos
      PetscBool :: fine_equal, coarse_equal
#endif       

      ! ~~~~~~  

      ! In Kokkos the DDC and PMISR do everything on the device
      ! that means we don't need to create intermediate is_fine and is_coarse ISs
      need_intermediate_is = .TRUE.

#if defined(PETSC_HAVE_KOKKOS)    

      call MatGetType(input_mat, mat_type, ierr)
      if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) then 

            ! If kokkos debugging is on, the pmisr and ddc do
            ! copy to the host after they finish in order to do the comparisons
            ! and hence we do need the intermediate ISs
            ! If doing pmis agg, the initial pmis will be on the device so always
            ! trigger the d2h copy and build the intermediate is
            ! CR never runs the pmisr/ddc on the device (its heavy numerics are all
            ! default petsc ops that run on the device anyway) so its host
            ! cf_markers_local and ISs are always the authoritative ones
            if (.NOT. kokkos_debug() .AND. &
                  (cf_splitting_type /= CF_AGG .AND. cf_splitting_type /= CF_PMIS_AGG .AND. &
                   cf_splitting_type /= CF_CR)) then
               need_intermediate_is = .FALSE.
            end if
      end if                     
#endif

      ! CR builds its splitting from scratch - all points start as F and each
      ! cr_pass promotes an independent set of the slowest converging
      ! F rows to C until the target CR rate is reached
      ! There is no strength matrix or first pass, so for CR the strong_threshold
      ! carries the target CR rate (just like it carries the target diagonal
      ! dominance ratio for CF_DIAG_DOM)
      if (cf_splitting_type == CF_CR) then

         ! The CR relaxation defaults, overridden by PCAIR with its Aff
         ! inverse settings
         ! Matches the default -pc_air_inverse_type
         cr_inverse_type_use = PFLAREINV_ARNOLDI
         cr_poly_order_use = PFLARE_CR_POLY_ORDER
         cr_sparsity_order_use = 1
         cr_diag_scale_use = .FALSE.
         if (present(cr_inverse_type)) cr_inverse_type_use = cr_inverse_type
         if (present(cr_poly_order)) cr_poly_order_use = cr_poly_order
         if (present(cr_inverse_sparsity_order)) cr_sparsity_order_use = cr_inverse_sparsity_order
         if (present(cr_diag_scale_polys)) cr_diag_scale_use = cr_diag_scale_polys

         call MatGetLocalSize(input_mat, local_rows, local_cols, ierr)
         allocate(cf_markers_local(local_rows))
         cf_markers_local = F_POINT
         call create_cf_is(input_mat, cf_markers_local, is_fine, is_coarse)

         cr_its_loop: do its = 1, PFLARE_CR_MAX_ITS

            ! Directly modifies the values in cf_markers_local
            call cr_pass(input_mat, is_fine, strong_threshold, &
                     cr_inverse_type_use, cr_poly_order_use, &
                     cr_sparsity_order_use, cr_diag_scale_use, &
                     cr_rate_achieved, n_swapped, cf_markers_local)

            ! If we swapped anything the ISs are now outdated
            if (n_swapped > 0) then
               call ISDestroy(is_fine, ierr)
               call ISDestroy(is_coarse, ierr)
               call create_cf_is(input_mat, cf_markers_local, is_fine, is_coarse)
            end if

            ! Terminate if we've hit the target rate or we can't make progress
            if (cr_rate_achieved <= strong_threshold .OR. n_swapped == 0) exit cr_its_loop
         end do cr_its_loop

      else

         ! Generate the strength matrix and do the first pass CF splitting
         call first_pass_splitting(input_mat, skip_symmetrize, strong_threshold, &
                  max_luby_steps, cf_splitting_type, cf_markers_local)

         ! Create the IS for the CF splittings
         if (need_intermediate_is) call create_cf_is(input_mat, cf_markers_local, is_fine, is_coarse)
      end if

      ! Only do the DDC pass if we're doing PMISR_DDC
      ! and if we haven't requested an exact independent set, ie strong threshold is not zero
      ! as this gives diagonal Aff)
      if (strong_threshold /= 0d0 .AND. &
            (cf_splitting_type == CF_PMISR_DDC .OR. cf_splitting_type == CF_DIAG_DOM)) then

         ! Do a set number of DDC iterations for PMISR_DDC.
         ! For CF_DIAG_DOM, iterate until the requested strong_threshold ratio is reached.
         ddc_its_max = ddc_its
         if (cf_splitting_type == CF_DIAG_DOM) ddc_its_max = huge(ddc_its_max)

         ddc_its_loop: do its = 1, ddc_its_max

            ! Do the second pass cleanup - this will directly modify the values in cf_markers_local
            ! (or the equivalent device cf_markers, is_fine is ignored if on the device)
            max_dd_ratio_achieved = 0d0
            if (cf_splitting_type == CF_DIAG_DOM) max_dd_ratio_achieved = strong_threshold
            call ddc(input_mat, is_fine, fraction_swap, max_dd_ratio_achieved, cf_markers_local)

            ! If we did anything in our ddc second pass and hence need to rebuild
            ! the is_fine and is_coarse
            if ((fraction_swap /= 0d0 .OR. max_dd_ratio_achieved /= 0d0) &
                  .AND. need_intermediate_is) then
            
               ! These are now outdated
               call ISDestroy(is_fine, ierr)
               call ISDestroy(is_coarse, ierr)

               ! Create the new CF ISs
               call create_cf_is(input_mat, cf_markers_local, is_fine, is_coarse) 
            end if

            ! Terminate if we've reached the ratio
            if (cf_splitting_type == CF_DIAG_DOM .AND. max_dd_ratio_achieved < strong_threshold) exit ddc_its_loop
         end do ddc_its_loop
      end if

      ! The Kokkos PMISR and DDC no longer copy back to the host to save copies
      ! during an arbritrary number of iterations above     
      ! We need to explicitly copy the cf_markers_local back to the host once 
      ! we've finished both the PMISR and all the DDC iterations  
      ! And we also need to build the host IS's    
#if defined(PETSC_HAVE_KOKKOS)    

      if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) then

         ! The initial pmis will be on the device, so need to destroy
         ! Intermediate ISs are created in that case
         if (cf_splitting_type == CF_PMIS_AGG) then
            ! Destroys the device cf_markers_local
            call delete_device_cf_markers()
            call delete_device_diag_dom_ratio()
         end if
      
         ! Aggregation is not on the device at all, and CR never creates
         ! device cf_markers (its host ISs are already the final answer)
         if (cf_splitting_type /= CF_AGG .AND. cf_splitting_type /= CF_PMIS_AGG .AND. &
               cf_splitting_type /= CF_CR) then

            A_array = input_mat%v
            is_fine_array = is_fine_kokkos%v
            is_coarse_array = is_coarse_kokkos%v

            ! Create the host is_fine and is_coarse based on device cf_markers
            call create_cf_is_kokkos(A_array, is_fine_array, is_coarse_array)  
            is_fine_kokkos%v = is_fine_array
            is_coarse_kokkos%v = is_coarse_array         
            
            ! If we are doing debugging the IS's are already created above
            ! so let's compare them to our kokkos ones
            if (kokkos_debug()) then 

               call ISEqual(is_fine, is_fine_kokkos, fine_equal, ierr)
               if (.NOT. fine_equal) then
                  print *, "Kokkos and CPU versions of create_cf_is_kokkos for fine do not match"
                  call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, ierr)
               end if
               call ISEqual(is_coarse, is_coarse_kokkos, coarse_equal, ierr)
               if (.NOT. coarse_equal) then
                  print *, "Kokkos and CPU versions of create_cf_is_kokkos for coarse do not match"
                  call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, ierr)
               end if
               ! Destroy the extra IS's
               call ISDestroy(is_fine_kokkos, ierr)
               call ISDestroy(is_coarse_kokkos, ierr)

            ! If we're not debugging just copy over the ones we built on the device            
            else
               is_fine = is_fine_kokkos
               is_coarse = is_coarse_kokkos
            end if

            ! Destroys the device cf_markers_local
            call delete_device_cf_markers()
            call delete_device_diag_dom_ratio()
         end if

      end if    
#endif       

      deallocate(cf_markers_local)

   end subroutine compute_cf_splitting 
   
! -------------------------------------------------------------------------------------------------------------------------------

end module cf_splitting

