module air_data_type_routines

   use air_data_type, only: air_multigrid_data, REUSE_MAT_ACTIVE, REUSE_IS_ACTIVE
   use pflare_parameters, only: PFLAREINV_ARNOLDI, AIR_Z_PRODUCT, MAT_RAP_DROP, MAT_INV_AFF, &
         PFLARE_TOL_AUTO_TRUNCATE
   use approx_inverse_setup, only: reset_inverse_mat, destroy_matrix_reuse
   use fc_smooth, only: destroy_VecISCopyLocalWrapper, setup_air_block_products
   
   ! PETSc
   use petscmat

#include "petsc/finclude/petscmat.h"

   implicit none
   public
  
   contains    

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine create_air_data(air_data)

      ! Setup the data structures for air and reads options from the 
      ! command line

      ! ~~~~~~
      type(air_multigrid_data), intent(inout)    :: air_data

      integer :: i_loc
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

      ! Temporary dense blocks used during a multiple rhs (block) smooth
      ! Only the outer arrays are allocated here, the blocks themselves are
      ! built lazily in ensure_air_block_temps once we know how many columns
      ! we have been given
      do i_loc = 1, size(air_data%block_temp_fine)
         allocate(air_data%block_temp_fine(i_loc)%array(air_data%options%max_levels))
         allocate(air_data%block_temp_coarse(i_loc)%array(air_data%options%max_levels))
      end do
      allocate(air_data%block_temp_full(1)%array(air_data%options%max_levels))
      air_data%block_ncols = -1
      air_data%block_local_ncols = -1

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

      ! The dense scratch blocks are sized from A_ff/A_cf/coarse_matrix on each
      ! level, so they have to go whenever we reset regardless of whether we are
      ! reusing - the number of levels and the layouts on them can both change
      ! when we build again. They are cheap to rebuild, ensure_air_block_temps
      ! does that lazily on the next block apply
      call destroy_air_block_temps(air_data)

      ! Use if this data structure is allocated to determine if we setup anything
      if (allocated(air_data%allocated_matrices_A_ff)) then

         ! Loop over the levels
         do our_level = 1, size(air_data%allocated_matrices_A_ff)

            ! If we setup Aff
            if (air_data%allocated_matrices_A_ff(our_level)) then

               ! Destroy data that depends on the CF splitting and poly coefficients
               ! only when not reusing (the IS device copy and poly coefficients
               ! remain valid across setups when reusing)
               if (.NOT. reuse) then
                  call destroy_VecISCopyLocalWrapper(air_data, our_level)
                  if (associated(air_data%inv_A_ff_poly_data(our_level)%coefficients)) then
                     deallocate(air_data%inv_A_ff_poly_data(our_level)%coefficients)
                     air_data%inv_A_ff_poly_data(our_level)%coefficients => null()
                  end if
                  if (associated(air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients)) then
                     deallocate(air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients)
                     air_data%inv_A_ff_poly_data_dropped(our_level)%coefficients => null()
                  end if
               end if

               ! temp_vecs are sized from A_ff/A_fc; they must be destroyed whenever
               ! A_ff/A_fc are destroyed (amounts 1 and 2 destroy them even when reusing)
               if (.NOT. reuse .OR. air_data%options%reuse_amount < 3) then
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
               
               ! Only amount=3 preserves A_ff/A_fc/A_cf and grid-transfer operators
               ! between setups.  For amount<=2 these are destroyed and rebuilt
               ! from scratch (SpGEMM reuse at amount=2 works via stored W/Z/AP/RAP
               ! whose sparsity is guaranteed by stored RAP_DROP).
               if (.NOT. reuse .OR. air_data%options%reuse_amount < 3) then

                  call MatDestroy(air_data%prolongators(our_level), ierr)
                  if (.NOT. air_data%options%symmetric) then
                     call MatDestroy(air_data%restrictors(our_level), ierr)
                  end if                     
                  call reset_inverse_mat(air_data%inv_A_ff(our_level))
                  call MatDestroy(air_data%A_ff(our_level), ierr)
                  call MatDestroy(air_data%A_fc(our_level), ierr)
                  call MatDestroy(air_data%A_cf(our_level), ierr)
                  air_data%allocated_matrices_A_ff(our_level) = .FALSE.
               end if
            end if

            ! IS_fine_index and IS_coarse_index (the CF splitting) are always kept
            ! whenever reuse=.TRUE., regardless of reuse_amount.  The CF splitting
            ! is the basis for all other reuse at every amount level.
            if (air_data%allocated_is(our_level)) then
               if (.NOT. reuse) then
                  call ISDestroy(air_data%IS_fine_index(our_level), ierr)
                  call ISDestroy(air_data%IS_coarse_index(our_level), ierr)
                  air_data%allocated_is(our_level) = .FALSE.
               end if
            end if

            ! Did we do C point smoothing?
            if (air_data%allocated_matrices_A_cc(our_level)) then
               ! Same logic as A_ff: only amount=3 preserves A_cc between setups.
               if (.NOT. reuse .OR. air_data%options%reuse_amount < 3) then
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
                     call destroy_matrix_reuse(air_data%reuse(our_level)%reuse_mat(i_loc), &
                        air_data%reuse(our_level)%reuse_submatrices(i_loc)%array)
                  end if
               end do

               do i_loc = 1, size(air_data%reuse(our_level)%reuse_is)
                  temp_is = air_data%reuse(our_level)%reuse_is(i_loc)
                  if (.NOT. PetscObjectIsNull(temp_is)) then
                     call ISDestroy(air_data%reuse(our_level)%reuse_is(i_loc), ierr)
                  end if
               end do
            else
               ! When reusing, destroy any reuse entries not active at the
               ! current amount level (handles amount being lowered between setups)
               do i_loc = 1, size(air_data%reuse(our_level)%reuse_mat)
                  if (.NOT. REUSE_MAT_ACTIVE(i_loc, air_data%options%reuse_amount)) then
                     temp_mat = air_data%reuse(our_level)%reuse_mat(i_loc)
                     if (.NOT. PetscObjectIsNull(temp_mat)) then
                        call destroy_matrix_reuse(air_data%reuse(our_level)%reuse_mat(i_loc), &
                           air_data%reuse(our_level)%reuse_submatrices(i_loc)%array)
                     end if
                  end if
               end do

               do i_loc = 1, size(air_data%reuse(our_level)%reuse_is)
                  if (.NOT. REUSE_IS_ACTIVE(i_loc, air_data%options%reuse_amount)) then
                     temp_is = air_data%reuse(our_level)%reuse_is(i_loc)
                     if (.NOT. PetscObjectIsNull(temp_is)) then
                        call ISDestroy(air_data%reuse(our_level)%reuse_is(i_loc), ierr)
                     end if
                  end if
               end do
            end if
         end do

         if (air_data%no_levels /= -1) then
            ! Coarse grid solver
            ! Reset when not reusing, or when reusing but inv_A_ff is not stored
            ! at this reuse_amount level (amounts 1 and 2); otherwise reuse_triggered
            ! would be TRUE in build_gmres_polynomial_inverse even though the old
            ! inv_A_ff is stale relative to the rebuilt coarse matrix.
            if (.NOT. reuse .OR. &
                 .NOT. REUSE_MAT_ACTIVE(MAT_INV_AFF, air_data%options%reuse_amount)) then
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

   subroutine destroy_air_block_temps(air_data)

      ! Destroys the dense scratch blocks the multiple rhs (block) smooths use
      ! ensure_air_block_temps rebuilds any of them we need on the next block apply

      ! ~~~~~~
      type(air_multigrid_data), intent(inout) :: air_data

      integer :: our_level, i_loc
      PetscErrorCode :: ierr
      type(tMat) :: temp_mat
      ! ~~~~~~

      ! The outer arrays are only allocated in create_air_data
      if (.NOT. allocated(air_data%block_temp_full(1)%array)) return

      do our_level = 1, size(air_data%block_temp_full(1)%array)

         do i_loc = 1, size(air_data%block_temp_fine)

            ! MatDestroy nulls the local copy of the handle, not the slot in the
            ! array, so we have to copy it back or the slot would be left dangling
            ! and ensure_air_block_temps would skip rebuilding it
            temp_mat = air_data%block_temp_fine(i_loc)%array(our_level)
            if (.NOT. PetscObjectIsNull(temp_mat)) then
               call MatDestroy(temp_mat, ierr)
               air_data%block_temp_fine(i_loc)%array(our_level) = temp_mat
            end if

            temp_mat = air_data%block_temp_coarse(i_loc)%array(our_level)
            if (.NOT. PetscObjectIsNull(temp_mat)) then
               call MatDestroy(temp_mat, ierr)
               air_data%block_temp_coarse(i_loc)%array(our_level) = temp_mat
            end if
         end do

         temp_mat = air_data%block_temp_full(1)%array(our_level)
         if (.NOT. PetscObjectIsNull(temp_mat)) then
            call MatDestroy(temp_mat, ierr)
            air_data%block_temp_full(1)%array(our_level) = temp_mat
         end if
      end do

      air_data%block_ncols = -1
      air_data%block_local_ncols = -1

   end subroutine destroy_air_block_temps

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine create_air_block_temp(op_mat, from_cols, x_mat, local_cols, global_cols, block_mat, ierr)

      ! Creates one dense scratch block with the row layout of op_mat (or its
      ! column layout if from_cols is true) and the column layout (and dense type)
      ! of the block of rhs we've been given, x_mat
      ! Does nothing if the block already exists

      ! ~~~~~~
      type(tMat), intent(in)        :: op_mat, x_mat
      logical, intent(in)           :: from_cols
      PetscInt, intent(in)          :: local_cols, global_cols
      type(tMat), intent(inout)     :: block_mat
      PetscErrorCode, intent(inout) :: ierr

      PetscInt :: local_rows, global_rows, local_op_cols, global_op_cols
      VecType :: vec_type
      MPIU_Comm :: MPI_COMM_MATRIX
      type(tMat) :: temp_mat
      ! ~~~~~~

      temp_mat = block_mat
      if (.NOT. PetscObjectIsNull(temp_mat)) return

      call PetscObjectGetComm(op_mat, MPI_COMM_MATRIX, ierr)
      call MatGetLocalSize(op_mat, local_rows, local_op_cols, ierr)
      call MatGetSize(op_mat, global_rows, global_op_cols, ierr)
      if (from_cols) then
         local_rows  = local_op_cols
         global_rows = global_op_cols
      end if
      ! Take the type from the block of rhs, so a block that lives on the device
      ! gives us device scratch
      call MatGetVecType(x_mat, vec_type, ierr)

      call MatCreateDenseFromVecType(MPI_COMM_MATRIX, vec_type, local_rows, local_cols, &
               global_rows, global_cols, PETSC_DECIDE, PETSC_NULL_SCALAR, &
               block_mat, ierr)

   end subroutine create_air_block_temp

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine ensure_air_block_temps(air_data, x_mat, ierr)

      ! Makes sure the cached dense scratch blocks match the block of rhs, x_mat,
      ! we've been given
      ! This is the air equivalent of ensure_block_temp_mats in matshell_data_type
      ! The blocks have the row layouts of the operators on each level (which we
      ! only know after the setup) and the column layout of x_mat (which we only
      ! know once a block apply happens), hence building them lazily here

      ! ~~~~~~
      type(air_multigrid_data), intent(inout) :: air_data
      type(tMat), intent(in)                  :: x_mat
      PetscErrorCode, intent(inout)           :: ierr

      integer :: our_level, i_loc, n_coarse
      logical :: rebuild
      PetscInt :: global_rows, global_cols, local_rows, local_cols
      MatType :: x_type, temp_type
      type(tMat) :: temp_mat
      ! ~~~~~~

      ! With full smoothing up and down we never do an FC smooth, so none of these
      ! blocks are used - the level smoothers and the coarse grid solver apply
      ! their inverses to the whole block directly
      if (air_data%options%full_smoothing_up_and_down) return
      ! Nothing to smooth on if there's only a single level
      if (air_data%no_levels < 2) return
      if (.NOT. allocated(air_data%block_temp_full(1)%array)) return

      call MatGetSize(x_mat, global_rows, global_cols, ierr)
      call MatGetLocalSize(x_mat, local_rows, local_cols, ierr)
      call MatGetType(x_mat, x_type, ierr)

      ! If the number of columns has changed we have to start again - the local
      ! column layout also has to match or the products would have incompatible
      ! layouts
      rebuild = air_data%block_ncols /= global_cols .OR. &
                  air_data%block_local_ncols /= local_cols

      ! The type of block we're given can also change between applies
      if (.NOT. rebuild) then
         do our_level = 1, air_data%no_levels - 1
            temp_mat = air_data%block_temp_fine(1)%array(our_level)
            if (.NOT. PetscObjectIsNull(temp_mat)) then
               call MatGetType(temp_mat, temp_type, ierr)
               if (temp_type /= x_type) rebuild = .TRUE.
               exit
            end if
         end do
      end if

      if (rebuild) call destroy_air_block_temps(air_data)

      ! We only need the extra coarse blocks if we're C point smoothing, in the
      ! same way as the extra temp_vecs_coarse
      n_coarse = 1
      if (air_data%options%any_c_smooths) n_coarse = size(air_data%block_temp_coarse)

      ! Now create any of the blocks we need that don't exist
      do our_level = 1, air_data%no_levels - 1

         do i_loc = 1, size(air_data%block_temp_fine)
            call create_air_block_temp(air_data%A_ff(our_level), .FALSE., x_mat, &
                     local_cols, global_cols, &
                     air_data%block_temp_fine(i_loc)%array(our_level), ierr)
         end do

         ! A_fc has the coarse column layout - we take it from A_fc rather than
         ! A_cf as A_cf is destroyed after the setup unless we're C smoothing,
         ! and this is where temp_vecs_coarse gets its layout from too
         do i_loc = 1, n_coarse
            call create_air_block_temp(air_data%A_fc(our_level), .TRUE., x_mat, &
                     local_cols, global_cols, &
                     air_data%block_temp_coarse(i_loc)%array(our_level), ierr)
         end do

         ! Only the injector version of MatISCopyLocalWrapper needs a full size
         ! block, in the same way as temp_vecs
         if (.NOT. air_data%fast_veciscopy_exists) then
            call create_air_block_temp(air_data%coarse_matrix(our_level), .FALSE., x_mat, &
                     local_cols, global_cols, &
                     air_data%block_temp_full(1)%array(our_level), ierr)
         end if

         ! The blocks on this level have just been (re)built, so attach the
         ! products the block smooths run to their scratch targets - the smooths
         ! then only run the numeric phase of each product every iteration
         ! rebuild is always true when the blocks didn't exist yet (block_ncols
         ! starts and is reset to -1), so the products can't be attached twice
         if (rebuild) call setup_air_block_products(air_data, our_level, ierr)
      end do

      air_data%block_ncols = global_cols
      air_data%block_local_ncols = local_cols

   end subroutine ensure_air_block_temps

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine destroy_air_data(air_data)

      ! Destroys the data structures for air

      ! ~~~~~~
      type(air_multigrid_data), intent(inout) :: air_data

      integer :: our_level, i_loc
      ! ~~~~~~

      call reset_air_data(air_data)

      ! Now set the options back to the default
      air_data%options%print_stats_timings = .FALSE.

      air_data%options%max_levels = 300
      air_data%options%coarse_eq_limit = 6
      air_data%options%auto_truncate_start_level = -1
      air_data%options%auto_truncate_tol = PFLARE_TOL_AUTO_TRUNCATE
      air_data%options%processor_agglom = .TRUE.
      air_data%options%processor_agglom_ratio = 2
      air_data%options%processor_agglom_factor = 2
      air_data%options%process_eq_limit = 50
      air_data%options%subcomm = .FALSE.

      air_data%options%strong_threshold = 0.5
      air_data%options%ddc_its = 1
      air_data%options%ddc_fraction = 0.1
      air_data%options%cf_splitting_type = 0
      air_data%options%max_luby_steps = -1

      air_data%options%smooth_order = 0
      air_data%options%smooth_order(1) = 2
      air_data%options%any_c_smooths = .FALSE.
      air_data%options%diag_scale_polys = .FALSE.
      air_data%options%matrix_free_polys = .FALSE.
      air_data%options%one_point_classical_prolong = .TRUE.
      air_data%options%full_smoothing_up_and_down = .FALSE.
      air_data%options%symmetric = .FALSE.
      air_data%options%constrain_w = .FALSE.
      air_data%options%constrain_z = .FALSE.  
      air_data%options%improve_z_its = 0
      air_data%options%improve_w_its = 0     

      air_data%options%strong_r_threshold = 0d0

      air_data%options%inverse_type = PFLAREINV_ARNOLDI

      air_data%options%z_type = AIR_Z_PRODUCT

      air_data%options%lair_distance = 2

      air_data%options%poly_order = 6
      air_data%options%inverse_sparsity_order = 1

      air_data%options%c_inverse_type = PFLAREINV_ARNOLDI
      air_data%options%c_poly_order = 6
      air_data%options%c_inverse_sparsity_order = 1
      
      air_data%options%coarsest_inverse_type = PFLAREINV_ARNOLDI
      air_data%options%coarsest_poly_order = 6
      air_data%options%coarsest_inverse_sparsity_order = 1
      air_data%options%coarsest_matrix_free_polys = .FALSE.
      air_data%options%coarsest_diag_scale_polys = .FALSE.
      air_data%options%coarsest_subcomm = .FALSE.

      air_data%options%r_drop = 0.01
      air_data%options%a_drop = 1e-4
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

         ! The blocks themselves have already been destroyed in reset_air_data
         do i_loc = 1, size(air_data%block_temp_fine)
            deallocate(air_data%block_temp_fine(i_loc)%array)
            deallocate(air_data%block_temp_coarse(i_loc)%array)
         end do
         deallocate(air_data%block_temp_full(1)%array)

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
