module cf_splitting

   use petsc
   use pmisr_ddc
   use aggregation
   use petsc_helper

#include "petsc/finclude/petsc.h"

   implicit none

#include "petsc_legacy.h"

   public   

   PetscEnum, parameter :: CF_PMISR_DDC=0
   PetscEnum, parameter :: CF_PMIS=1
   PetscEnum, parameter :: CF_PMIS_DIST2=2
   PetscEnum, parameter :: CF_AGG=3
   PetscEnum, parameter :: CF_PMIS_AGG=4
   
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
      integer :: n, n_fine, n_coarse
      PetscErrorCode :: ierr
      MPI_Comm :: MPI_COMM_MATRIX      
      
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
   
!------------------------------------------------------------------------------------------------------------------------
   
   subroutine generate_sabs(input_mat, strong_threshold, symmetrize, square, output_mat, allow_drop_diagonal)
      
      ! Generate strength of connection matrix with absolute value 
      ! Output has no diagonal entries
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in)     :: input_mat
      type(tMat), intent(inout)  :: output_mat
      PetscReal, intent(in)           :: strong_threshold
      logical, intent(in)        :: symmetrize, square
      logical, intent(in), optional :: allow_drop_diagonal
      
      PetscInt :: ifree
      PetscInt :: local_rows, local_cols, global_rows, global_cols
      PetscInt :: global_row_start, global_row_end_plus_one
      PetscInt :: global_col_start, global_col_end_plus_one
      integer :: counter, errorcode, comm_size
      PetscErrorCode :: ierr
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      MPI_Comm :: MPI_COMM_MATRIX
      type(tMat) :: transpose_mat
      type(tIS) :: zero_diags
      PetscInt, dimension(:), pointer :: zero_diags_pointer
      logical :: drop_diag
      
      ! ~~~~~~~~~~

      drop_diag = .TRUE.
      if (present(allow_drop_diagonal)) drop_diag = allow_drop_diagonal

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)

      ! Get the local sizes
      call MatGetLocalSize(input_mat, local_rows, local_cols, ierr)
      call MatGetSize(input_mat, global_rows, global_cols, ierr)
      ! This returns the global index of the local portion of the matrix
      call MatGetOwnershipRange(input_mat, global_row_start, global_row_end_plus_one, ierr)  
      call MatGetOwnershipRangeColumn(input_mat, global_col_start, global_col_end_plus_one, ierr)  
      
      ! Drop entries smaller than the strong_threshold, with a relative tolerance measured 
      ! against the biggest abs non-diagonal entry, don't lump and always drop the diagonal
      call remove_small_from_sparse(input_mat, strong_threshold, output_mat, &
               relative_max_row_tol_int = -1, lump=.FALSE., drop_diagonal_int=-1)

      ! Now symmetrize if desired
      if (symmetrize) then

         ! We could just do a symbolic transpose and add the two sets of indices together, 
         ! but its so much simpler to just add the two together - and the symbolic will be the expensive part
         ! anyway
         call MatTranspose(output_mat, MAT_INITIAL_MATRIX, transpose_mat, ierr)
         call MatAXPY(output_mat, 1d0, transpose_mat, DIFFERENT_NONZERO_PATTERN, ierr)     

         ! Don't forget to destroy the explicit transpose
         call MatDestroy(transpose_mat, ierr)

      end if

      ! Square the strength matrix to aggressively coarsen (gives a distance 2 MIS)
      if (square) then

         if (symmetrize) then
            call MatMatMult(output_mat, output_mat, &
                        MAT_INITIAL_MATRIX, 1d0, transpose_mat, ierr)     
         else
            call MatTransposeMatMult(output_mat, output_mat, &
                        MAT_INITIAL_MATRIX, 1d0, transpose_mat, ierr)          
         endif     

         ! Also have to add in the original distance 1 connections to the square
         ! as the dist 1 strength matrix has had the diagonals removed, so the square won't 
         ! have the dist 1 connetions in it
         call MatAXPY(transpose_mat, 1d0, output_mat, DIFFERENT_NONZERO_PATTERN, ierr)     
         call MatDestroy(output_mat, ierr)

         ! Can end up with diagonal entries we have to remove
         ! Let's get the diagonals that are zero or unassigned
         call MatFindZeroDiagonals(transpose_mat, zero_diags, ierr)
         call ISGetIndicesF90(zero_diags, zero_diags_pointer, ierr)
         ! Then let's just set every other row to have a zero diagonal
         ! as we know they're already preallocated
         counter = 1
         do ifree = 1, local_rows

            if (counter .le. size(zero_diags_pointer)) then
               ! Skip over any rows that don't have diagonals or are already zero
               if (zero_diags_pointer(counter) - global_row_start + 1 == ifree) then
                  counter = counter + 1 
                  cycle
               end if
            end if
         
            ! Set the diagonal to 0
            call MatSetValue(transpose_mat, ifree - 1 + global_row_start, ifree - 1 + global_row_start, 0d0, INSERT_VALUES, ierr)
         end do
         
         call MatAssemblyBegin(transpose_mat, MAT_FINAL_ASSEMBLY, ierr)
         call MatAssemblyEnd(transpose_mat, MAT_FINAL_ASSEMBLY, ierr)

         ! Could call MatEliminateZeros in later versions of petsc, but for here
         ! given we know the entries are ==1, we will just create a copy with "small" stuff removed
         ! ie the zero diagonal
         call remove_small_from_sparse(transpose_mat, 1d-100, output_mat, drop_diagonal_int = 1) 
         call MatDestroy(transpose_mat, ierr)

      end if   
      
      ! Reset the entries in the strength matrix back to 1
      if (symmetrize .OR. square) call MatSetAllValues(output_mat, 1d0)

   end subroutine generate_sabs     

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine first_pass_splitting(input_mat, symmetric, strong_threshold, max_luby_steps, cf_splitting_type, cf_markers_local)

      ! Compute a symmetrized strength matrix and then call the first pass of CF splitting

      ! ~~~~~~
      type(tMat), target, intent(in)      :: input_mat
      logical, intent(in)                 :: symmetric
      PetscReal, intent(in)                    :: strong_threshold
      integer, intent(in)                 :: max_luby_steps, cf_splitting_type
      integer, dimension(:), allocatable, intent(inout) :: cf_markers_local

      ! Local
      PetscInt :: global_row_start, global_row_end_plus_one, i_loc
      ! PetscInt :: counter, local_c_size
      ! PetscInt :: global_row_start_dist2, global_row_end_plus_one_dist2
      PetscInt :: local_rows, local_cols, ncols
      integer :: errorcode, MPI_COMM_MATRIX, comm_size
      PetscErrorCode :: ierr
      type(tMat) :: strength_mat
      ! type(tMat) :: prolongators, strength_mat_c, strength_mat_fc, temp_mat
      ! type(tIS) :: is_fine, is_coarse, zero_diags
      ! integer, dimension(:), allocatable :: cf_markers_local_c
      PetscInt, dimension(:), allocatable :: aggregates
      ! PetscInt, dimension(:), pointer :: is_pointer_coarse, is_pointer_fine, zero_diags_pointer
      PetscInt, parameter :: nz_ignore = -1
      type(tMat) :: Ad, Ao
      PetscOffset :: iicol
      PetscInt :: icol(1)      

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

      ! PMISR DDC and Aggregation
      else 
         ! Only symmetrize if not already symmetric
         call generate_sabs(input_mat, strong_threshold, .NOT. symmetric, .FALSE., strength_mat)         
      end if

      ! ~~~~~~~~~~~~
      ! Do the first pass splitting
      ! ~~~~~~~~~~~~

      ! PMISR
      if (cf_splitting_type == CF_PMISR_DDC) then

         call pmisr(strength_mat, max_luby_steps, .FALSE., cf_markers_local)

      ! Distance 1 PMIS
      else if (cf_splitting_type == CF_PMIS) then

         call pmisr(strength_mat, max_luby_steps, .TRUE., cf_markers_local)

      ! Distance 2 PMIS
      else if (cf_splitting_type == CF_PMIS_DIST2) then

         ! As we have generated S'S + S for the strength matrix, this will do distance 2 PMIS
         call pmisr(strength_mat, max_luby_steps, .TRUE., cf_markers_local)

         ! ~~~~~~~~~~
         ! The code below instead relies on the distance 1 strength matrix and does
         ! MIS(MIS(S_dist1)), which coarsens quite a bit faster than MIS(S'S)
         ! ~~~~~~~~~~

         ! ! We need IS's for the C and F points
         ! call create_cf_is(strength_mat, cf_markers_local, is_fine, is_coarse) 

         ! ! Get the number of local C points
         ! call ISGetLocalSize(is_coarse, local_c_size, ierr)

         ! ! Extract S_fc
         ! call MatCreateSubMatrix(strength_mat, &
         !          is_fine, is_coarse, MAT_INITIAL_MATRIX, &
         !          strength_mat_fc, ierr)   
                  
         ! call MatGetOwnershipRange(strength_mat, global_row_start, global_row_end_plus_one, ierr)                        
         ! ! Create [S_fc; 0]'
         ! call compute_P_from_W(strength_mat_fc, global_row_start, &
         !          is_fine, is_coarse, .FALSE., &
         !          prolongators)                     

         ! ! Now the strength matrix for C points can be computed from
         ! ! [Scf; 0] * strength_mat * [Sfc; 0]'
         ! ! which is equivalent to [Scf Sff Sfc], but given Aff will be so big 
         ! ! we don't bother to extract Aff, we just explicitly create [Sfc; 0]' 
         ! ! and then do the PtaP on the full S
         ! ! Also S is symmetric so Scf = Sfc'
         ! call MatPtap(strength_mat, prolongators, &
         !          MAT_INITIAL_MATRIX, 1.58d0, temp_mat, ierr)   
         
         ! call ISGetIndicesF90(is_fine, is_pointer_fine, ierr)
         ! call ISGetIndicesF90(is_coarse, is_pointer_coarse, ierr)           

         ! ! Can end up with diagonal entries we have to remove
         ! ! Let's get the diagonals that are zero or unassigned
         ! call MatFindZeroDiagonals(temp_mat, zero_diags, ierr)
         ! call ISGetIndicesF90(zero_diags, zero_diags_pointer, ierr)
         ! ! Then let's just set every other row to have a zero diagonal
         ! ! as we know they're already preallocated
         ! counter = 1
         ! call MatGetOwnershipRange(temp_mat, global_row_start_dist2, global_row_end_plus_one_dist2, ierr)                    
         ! do i_loc = 1, size(is_pointer_coarse)

         !    if (counter .le. size(zero_diags_pointer)) then
         !       ! Skip over any rows that don't have diagonals or are already zero
         !       if (zero_diags_pointer(counter) - global_row_start_dist2 + 1 == i_loc) then
         !          counter = counter + 1 
         !          cycle
         !       end if
         !    end if
         
         !    ! Set the diagonal to 0
         !    call MatSetValue(temp_mat, i_loc - 1 + global_row_start_dist2, i_loc - 1 + global_row_start_dist2, 0d0, INSERT_VALUES, ierr)
         ! end do
         
         ! call MatAssemblyBegin(temp_mat, MAT_FINAL_ASSEMBLY, ierr)
         ! call MatAssemblyEnd(temp_mat, MAT_FINAL_ASSEMBLY, ierr)

         ! ! Could call MatEliminateZeros in later versions of petsc, but for here
         ! ! given we know the entries are ==1, we will just create a copy with "small" stuff removed
         ! ! ie the zero diagonal
         ! call remove_small_from_sparse(temp_mat, 1e-100, strength_mat_c, drop_diagonal_int = .TRUE.)                   
         ! call MatDestroy(temp_mat, ierr)
         ! call ISRestoreIndicesF90(zero_diags, zero_diags_pointer, ierr)
         ! call ISDestroy(zero_diags, ierr)

         ! ! Do the second distance 1 PMIS on just the C points
         ! ! Now any C-point that is not strongly connected to any other C-point we want to retain as C-points
         ! ! It is only C-points that are not distance 2 away from other C-points that we want to turn into F points
         ! call pmisr(strength_mat_c, -1, .TRUE., cf_markers_local_c, zero_measure_c_point=.TRUE.)

         ! ! Then make sure we put the new definitions back into the original cf_markers
         ! cf_markers_local(is_pointer_coarse - global_row_start + 1) = cf_markers_local_c

         ! call ISRestoreIndicesF90(is_coarse, is_pointer_coarse, ierr)
         ! call ISRestoreIndicesF90(is_fine, is_pointer_fine, ierr)         
         ! call ISDestroy(is_fine, ierr)
         ! call ISDestroy(is_coarse, ierr)
         ! deallocate(cf_markers_local_c)
         ! call MatDestroy(strength_mat_fc, ierr)
         ! call MatDestroy(strength_mat_c, ierr)
         ! call MatDestroy(prolongators, ierr)

      ! PMIS on boundary nodes then processor local aggregation
      else if (cf_splitting_type == CF_PMIS_AGG) then
        
         ! In parallel we do distance 1 pmis on boundary nodes, then do serial aggregation 
         if (comm_size /= 1) then

            ! Do distance 1 PMIS
            call pmisr(strength_mat, max_luby_steps, .TRUE., cf_markers_local)

            ! Get the sequential part of the matrix
            call MatMPIAIJGetSeqAIJ(strength_mat, Ad, Ao, icol, iicol, ierr)   
            
            ! For any local node that doesn't touch a boundary node, we set the 
            ! cf_markers back to unassigned and leave them to be done by the aggregation
            do i_loc = 1, local_rows

               ! Get how many non-zeros are in the off-diagonal 
               call MatGetRow(Ao, i_loc-1, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
               ! If we don't touch anything off-processor, its local and set it to unassigned
               if (ncols == 0) cf_markers_local(i_loc) = 0
               call MatRestoreRow(Ao, i_loc-1, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
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

            ! Get the sequential part of the matrix
            call MatMPIAIJGetSeqAIJ(strength_mat, Ad, Ao, icol, iicol, ierr)   
            
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

   subroutine compute_cf_splitting(input_mat, symmetric, &
                     strong_threshold, max_luby_steps, &
                     cf_splitting_type, fraction_swap, &
                     is_fine, is_coarse)

      ! Computes a CF splitting and returns the F and C point ISs

      ! ~~~~~~
      type(tMat), target, intent(in)      :: input_mat
      logical, intent(in)                 :: symmetric
      PetscReal, intent(in)                    :: strong_threshold
      integer, intent(in)                 :: max_luby_steps, cf_splitting_type
      PetscReal, intent(in)                    :: fraction_swap
      type(tIS), intent(inout)            :: is_fine, is_coarse

      PetscErrorCode :: ierr
      integer, dimension(:), allocatable :: cf_markers_local

      ! ~~~~~~  

      ! Call the first pass CF splitting with a symmetrized strength matrix
      call first_pass_splitting(input_mat, symmetric, strong_threshold, max_luby_steps, cf_splitting_type, cf_markers_local)

      ! Create the IS for the CF splittings
      call create_cf_is(input_mat, cf_markers_local, is_fine, is_coarse)   
      
      ! Only do the DDC pass if we're doing PMISR_DDC
      ! and if we haven't requested an exact independent set, ie strong threshold is not zero
      ! as this gives diagonal Aff)
      if (strong_threshold /= 0d0 .AND. cf_splitting_type == CF_PMISR_DDC) then

         ! Do the second pass cleanup - this will directly modify the values in cf_markers_local
         call ddc(input_mat, is_fine, fraction_swap, cf_markers_local)

         ! If we did anything in our ddc second pass
         if (fraction_swap /= 0d0) then
         
            ! These are now outdated
            call ISDestroy(is_fine, ierr)
            call ISDestroy(is_coarse, ierr)

            ! Create the new CF ISs
            call create_cf_is(input_mat, cf_markers_local, is_fine, is_coarse) 
         end if
      end if

      deallocate(cf_markers_local)

   end subroutine compute_cf_splitting 
   
! -------------------------------------------------------------------------------------------------------------------------------

end module cf_splitting

