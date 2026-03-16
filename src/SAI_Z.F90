module sai_z

   use iso_c_binding
   use petscksp
   use binary_tree, only: itree, itree2vector, flush_tree
   use sorting, only: create_knuth_shuffle_tree_array, intersect_pre_sorted_indices_only, &
         merge_pre_sorted, sorted_binary_search
   use c_petsc_interfaces, only: mat_mat_symbolic_c, calculate_and_build_sai_z_kokkos
   use petsc_helper, only: generate_identity, kokkos_debug, destroy_matrix_reuse, MatAXPYWrapper
   use pflare_parameters, only: AIR_Z_PRODUCT, AIR_Z_LAIR, AIR_Z_LAIR_SAI, PFLAREINV_SAI, PFLAREINV_ISAI

#include "petsc/finclude/petscksp.h"

   implicit none
   public
   
   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine calculate_and_build_sai_z_cpu(A_ff_input, A_cf, sparsity_mat_cf, incomplete, &
                  reuse_mat, reuse_submatrices, z, no_approx_solve)

      ! Computes an approximation to z using sai/isai
      ! If incomplete is true then this is lAIR
      ! Can also use this to compute an SAI (ie an inverse in the non rectangular case)
      ! by giving A_cf as the negative identity (of the size A_ff)

      ! ~~~~~~
      type(tMat), intent(in)                            :: A_ff_input, A_cf, sparsity_mat_cf
      logical, intent(in)                               :: incomplete
      type(tMat), intent(inout)                         :: reuse_mat, z
      type(tMat), dimension(:), pointer, intent(inout)  :: reuse_submatrices
      logical, intent(in), optional                     :: no_approx_solve

      ! Local variables 
      PetscInt :: local_rows, local_cols, ncols, global_row_start, global_row_end_plus_one, ncols_sparsity_max
      PetscInt :: global_rows, global_cols, iterations_taken
      PetscInt :: i_loc, j_loc, cols_ad, rows_ad
      PetscInt :: rows_ao, cols_ao, ifree, row_size, i_size, j_size
      PetscInt :: global_row_start_aff, global_row_end_plus_one_aff
      integer :: lwork, intersect_count
      integer :: errorcode, comm_size
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX      
      type(tMat) :: transpose_mat, A_ff
      type(tIS), dimension(1) :: all_cols_indices, row_indices
      type(tIS), dimension(1) :: col_indices
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0, maxits=1000
      PetscInt, dimension(:), allocatable, target :: j_rows
      PetscInt, dimension(:), allocatable :: i_rows, ad_indices
      integer, dimension(:), allocatable :: pivots, j_indices, i_indices
      PetscInt, dimension(:), pointer :: cols => null(), j_rows_ptr => null()
      PetscReal, dimension(:), pointer :: vals => null()
      PetscReal, dimension(:), allocatable :: e_row, j_vals
      PetscReal, dimension(:,:), allocatable :: submat_vals
      type(itree) :: i_rows_tree
      PetscReal, dimension(:), allocatable :: work
      type(tVec) :: solution, rhs, diag_vec
      logical :: approx_solve, disable_approx_solve
      type(tMat) :: Ao, Ad, temp_mat
      type(tKSP) :: ksp
      type(tPC) :: pc
      PetscInt, dimension(:), pointer :: colmap
      logical :: deallocate_submatrices = .FALSE.
      PetscInt, dimension(:), allocatable :: col_indices_off_proc_array
      integer(c_long_long) :: A_array
      MatType:: mat_type, mat_type_input
      PetscScalar, dimension(:), pointer :: vec_vals
      PetscInt :: row_index_into_submatrix, ncols_match
      PetscInt, dimension(:), allocatable :: cols_global_temp
      integer, dimension(:), allocatable :: row_i_match, row_col_match
      integer :: match_count_sub
      type(tMat) :: small_mat

      ! ~~~~~~

      disable_approx_solve = .FALSE.
      if (present(no_approx_solve)) disable_approx_solve = no_approx_solve

      call PetscObjectGetComm(A_ff_input, MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)

      ! Get the local sizes
      call MatGetLocalSize(A_cf, local_rows, local_cols, ierr)
      call MatGetSize(A_cf, global_rows, global_cols, ierr)   
      
      call MatGetType(A_ff_input, mat_type_input, ierr)
      if (mat_type_input == MATDIAGONAL) then
         ! Convert it to aij just for this routine 
         ! doesn't work in parallel for some reason
         !call MatConvert(A_ff_input, MATAIJ, MAT_INITIAL_MATRIX, A_ff, ierr)
         call MatCreate(MPI_COMM_MATRIX, A_ff, ierr)
         call MatSetSizes(A_ff, local_cols, local_cols, global_cols, global_cols, ierr)
         call MatSetType(A_ff, MATAIJ, ierr)
         call MatSeqAIJSetPreallocation(A_ff,one,PETSC_NULL_INTEGER_ARRAY, ierr)
         call MatMPIAIJSetPreallocation(A_ff,one,PETSC_NULL_INTEGER_ARRAY,&
                  zero,PETSC_NULL_INTEGER_ARRAY, ierr)
         call MatSetUp(A_ff, ierr)
         call MatSetOption(A_ff, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)                   
         call MatCreateVecs(A_ff_input, diag_vec, PETSC_NULL_VEC, ierr)
         call MatGetDiagonal(A_ff_input, diag_vec, ierr)
         call MatDiagonalSet(A_ff, diag_vec, INSERT_VALUES, ierr)
         call MatAssemblyBegin(A_ff, MAT_FINAL_ASSEMBLY, ierr)
         call MatAssemblyEnd(A_ff, MAT_FINAL_ASSEMBLY, ierr)             
         call VecDestroy(diag_vec, ierr)
      else
         A_ff = A_ff_input
      end if
      call MatGetType(A_ff, mat_type, ierr)

      ! ~~~~~~~~~~~~~
      ! ~~~~~~~~~~~~~

      ! We're enforcing the same sparsity 
      
      ! If not re-using
      if (PetscObjectIsNull(z)) then
         call MatDuplicate(sparsity_mat_cf, MAT_DO_NOT_COPY_VALUES, z, ierr)
      end if

      ! Just in case there are some zeros in the input mat, ignore them
      ! Now this won't do anything given we've imposed the sparsity of sparsity_mat_cf on 
      ! this matrix in advance with the MatDuplicate above
      ! Dropping zeros from Z happens outside this routine
      !call MatSetOption(z, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE, ierr)       
      ! We know we will never have non-zero locations outside of the sparsity power 
      call MatSetOption(z, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE,  ierr)     
      call MatSetOption(z, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr) 
      ! We know we are only going to insert local vals
      ! These options should turn off any reductions in the assembly
      call MatSetOption(z, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)   
      
      call MatGetOwnershipRange(A_ff, global_row_start_aff, global_row_end_plus_one_aff, ierr)              

      ! ~~~~~~~~~~~~
      ! If we're in parallel we need to get the off-process rows of matrix that correspond
      ! to the columns of matrix
      ! ~~~~~~~~~~~~
      ! Have to double check comm_size /= 1 as we might be on a subcommunicator and we can't call
      ! MatMPIAIJGetSeqAIJ specifically if that's the case
      if (comm_size /= 1) then

         ! ~~~~
         ! Get the cols from the sparsity_mat_cf, not from A_ff
         ! ~~~~
         ! Much more annoying in older petsc
         if (mat_type == "mpiaij") then
            call MatMPIAIJGetSeqAIJ(sparsity_mat_cf, Ad, Ao, colmap, ierr) 
            A_array = sparsity_mat_cf%v
         else
            call MatConvert(sparsity_mat_cf, MATMPIAIJ, MAT_INITIAL_MATRIX, temp_mat, ierr)
            call MatMPIAIJGetSeqAIJ(temp_mat, Ad, Ao, colmap, ierr) 
            A_array = temp_mat%v
         end if

         ! Have to be careful here as we don't have a square matrix, so rows_ao isn't equal to the number of local columns
         call MatGetSize(Ad, rows_ad, cols_ad, ierr)             
         ! We know the col size of Ao is the size of colmap, the number of non-zero offprocessor columns
         call MatGetSize(Ao, rows_ao, cols_ao, ierr)         

         ! ~~~~~~~
         ! Now we can pull out the chunk of matrix that we need
         ! ~~~~~~~

         ! Setting this is necessary to avoid an allreduce when calling createsubmatrices
         ! This will be reset to false after the call to createsubmatrices
         call MatSetOption(A_ff, MAT_SUBMAT_SINGLEIS, PETSC_TRUE, ierr)       
         
         ! Now this will be doing comms to get the non-local rows we want
         ! Only fetch the non-local rows - local rows are already in A_ff
         ! This scales much better than fetching both local and non-local rows
         ! This returns a sequential matrix
         call ISCreateGeneral(PETSC_COMM_SELF, cols_ao, colmap, &
                  PETSC_USE_POINTER, row_indices(1), ierr)

         if (incomplete) then

            ! These are the global indices of the columns we want
            ! Taking care here to use cols_ad and not rows_ao  
            allocate(col_indices_off_proc_array(cols_ad + cols_ao))
            allocate(ad_indices(cols_ad))
            ! Local rows (as global indices)
            do ifree = 1, cols_ad
               ad_indices(ifree) = global_row_start_aff + ifree - 1
            end do

            ! Do a sort on the indices
            ! Both ad_indices and colmap should already be sorted so we can merge them together quickly
            call merge_pre_sorted(ad_indices, colmap, col_indices_off_proc_array)
            deallocate(ad_indices)

            ! Create the sequential IS we want with the cols we want (written as global indices)
            call ISCreateGeneral(PETSC_COMM_SELF, cols_ad + cols_ao, col_indices_off_proc_array, &
                     PETSC_USE_POINTER, col_indices(1), ierr)             
         else
            ! For full SAI we need all columns with global indices preserved
            ! so the shadow I computation can see all non-zero columns
            call ISCreateStride(PETSC_COMM_SELF, global_cols, zero, one, all_cols_indices(1), ierr)
            call ISSetIdentity(all_cols_indices(1), ierr)
         end if

         if (.NOT. PetscObjectIsNull(reuse_mat)) then
            reuse_submatrices(1) = reuse_mat
            if (incomplete) then
               call MatCreateSubMatrices(A_ff, one, row_indices, col_indices, MAT_REUSE_MATRIX, reuse_submatrices, ierr)
            else
               call MatCreateSubMatrices(A_ff, one, row_indices, all_cols_indices, MAT_REUSE_MATRIX, reuse_submatrices, ierr)
            end if
         else
            if (incomplete) then
               call MatCreateSubMatrices(A_ff, one, row_indices, col_indices, MAT_INITIAL_MATRIX, reuse_submatrices, ierr)
            else
               call MatCreateSubMatrices(A_ff, one, row_indices, all_cols_indices, MAT_INITIAL_MATRIX, reuse_submatrices, ierr)
            end if
            reuse_mat = reuse_submatrices(1)
         end if

         call ISDestroy(row_indices(1), ierr)
         if (.NOT. incomplete) call ISDestroy(all_cols_indices(1), ierr)

         row_size = size(col_indices_off_proc_array)
         call ISDestroy(col_indices(1), ierr)

      ! Easy in serial as we have everything we neeed
      else
         
         allocate(reuse_submatrices(1))
         deallocate_submatrices = .TRUE.               
         reuse_submatrices(1) = A_ff
         ! local rows is the size of c, local cols is the size of f
         row_size = local_cols
      end if        

      call MatGetOwnershipRange(sparsity_mat_cf, global_row_start, global_row_end_plus_one, ierr)              

      ! Setup the options for iterative solve when the direct gets too big
      if (incomplete) then

         ! Sequential solve
         call KSPCreate(PETSC_COMM_SELF, ksp, ierr)
         call KSPSetType(ksp, KSPGMRES, ierr)
         ! Solve to relative 1e-3
         call KSPSetTolerances(ksp, 1d-3, &
                  & 1d-50, &
                  & PETSC_DEFAULT_REAL, &
                  & maxits, ierr) 
         call KSPGetPC(ksp,pc,ierr)
         ! Should be diagonally dominant
         call PCSetType(pc, PCJACOBI, ierr)   
         
      else

         ! Sequential solve
         call KSPCreate(PETSC_COMM_SELF, ksp, ierr)
         ! Use the LSQR to solve the least squares inexactly
         call KSPSetType(ksp, KSPLSQR, ierr)

         ! Solve to relative 1e-3
         call KSPSetTolerances(ksp, 1d-3, &
                  & 1d-50, &
                  & PETSC_DEFAULT_REAL, &
                  & maxits, ierr)      
                  
         call KSPGetPC(ksp,pc,ierr)
         ! We would have to form A' * A to precondition, but should be 
         ! very diagonally dominant anyway
         call PCSetType(pc, PCNONE, ierr)                  
         
      end if

      ! Pre-pass to find the maximum number of columns in any row of sparsity_mat_cf
      ncols_sparsity_max = 0
      do i_loc = global_row_start, global_row_end_plus_one-1
         call MatGetRow(sparsity_mat_cf, i_loc, ncols, &
                  PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
         if (ncols > ncols_sparsity_max) ncols_sparsity_max = ncols
         call MatRestoreRow(sparsity_mat_cf, i_loc, ncols, &
                  PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
      end do

      ! Pre-allocate arrays that are sized by ncols from sparsity_mat_cf
      allocate(j_rows(ncols_sparsity_max), j_vals(ncols_sparsity_max), j_indices(ncols_sparsity_max))

      ! Now go through each of the rows
      ! GetRow has to happen over the global indices
      do i_loc = global_row_start, global_row_end_plus_one-1

         ! We just want the F-indices of whatever distance we are going out to
         call MatGetRow(sparsity_mat_cf, i_loc, ncols, &
                  cols, PETSC_NULL_SCALAR_POINTER, ierr)

         j_size = ncols
         j_vals(1:j_size) = 0
         j_rows(1:j_size) = cols(1:j_size)
         j_rows_ptr => j_rows(1:j_size)

         call MatRestoreRow(sparsity_mat_cf, i_loc, ncols, &
                  cols, PETSC_NULL_SCALAR_POINTER, ierr) 

         ! If we have no non-zeros in this row skip it
         ! This means we have a c point with no neighbour f points
         if (j_size == 0) cycle

         ! ~~~~~~~~
         ! We need to stick the non-zero row of A_cf into the indices of
         ! A_cf, or A_cf A_ff, or A_cf A_ff^2, etc given whatever distance we're going out to
         call MatGetRow(A_cf, i_loc, ncols, &
                  cols, vals, ierr)

         allocate(i_indices(ncols))
         call intersect_pre_sorted_indices_only(j_rows_ptr, cols(1:ncols), j_indices, i_indices, intersect_count)
         j_vals(j_indices(1:intersect_count)) = vals(i_indices(1:intersect_count))
         deallocate(i_indices)

         call MatRestoreRow(A_cf, i_loc, ncols, &
                  cols, vals, ierr)                   

         ! ~~~~~~~~

         ! j_rows stays as global indices - we access local rows directly
         ! from A_ff and non-local rows from the submatrix                    

         approx_solve = .FALSE.

         ! This is the "incomplete" SAI (Antz 2018) which minimises for column j
         ! ie it doesn't use the shadow I so gives a square system that can be solved
         ! exactly for = 0
         ! This is equivalent to a restricted additive schwarz
         ! ||M(j, J) A(J, J) - eye(j, J)||_2             
         if (incomplete) then
            allocate(i_rows(j_size))
            i_rows = j_rows(1:j_size)

         ! This is the SAI which minimises for column j
         ! which gives the rectangular system that must be minimised
         ! ||M(j, J) A(J, I) - eye(j, I)||_2             
         else

            ! Loop over all the non zero column indices, and get the nonzero columns in those rows
            ! ie get the shadow I
            row_index_into_submatrix = 1
            do j_loc = 1, j_size

               ! Check if this is a non-local row
               if (comm_size /= 1 .AND. &
                   (j_rows(j_loc) < global_row_start_aff .OR. j_rows(j_loc) >= global_row_end_plus_one_aff)) then

                  ! Non-local row: find position in colmap via walk
                  do while (row_index_into_submatrix <= cols_ao .AND. &
                            colmap(row_index_into_submatrix) < j_rows(j_loc))
                     row_index_into_submatrix = row_index_into_submatrix + 1
                  end do

                  ! Cols are GLOBAL (all_cols_indices preserves them)
                  call MatGetRow(reuse_submatrices(1), row_index_into_submatrix - 1, ncols, &
                           cols, PETSC_NULL_SCALAR_POINTER, ierr)
                  call create_knuth_shuffle_tree_array(cols(1:ncols), i_rows_tree)
                  call MatRestoreRow(reuse_submatrices(1), row_index_into_submatrix - 1, ncols, &
                           cols, PETSC_NULL_SCALAR_POINTER, ierr)

               else

                  ! Local row: get directly from A_ff - cols are global
                  call MatGetRow(A_ff, j_rows(j_loc), ncols, &
                           cols, PETSC_NULL_SCALAR_POINTER, ierr)
                  call create_knuth_shuffle_tree_array(cols(1:ncols), i_rows_tree)
                  call MatRestoreRow(A_ff, j_rows(j_loc), ncols, &
                           cols, PETSC_NULL_SCALAR_POINTER, ierr)

               end if

            end do

            allocate(i_rows(i_rows_tree%length))
            call itree2vector(i_rows_tree, i_rows)
            call flush_tree(i_rows_tree)

         end if

         ! If we have a big system to solve, do its iteratively
         if ((size(i_rows) > 40 .OR. j_size > 40) .AND. .NOT. disable_approx_solve) approx_solve = .TRUE.

         ! This determines the indices of J^* in I*
         ! Both i_rows and j_rows are global indices
         i_size = size(i_rows)
         allocate(i_indices(i_size))
         call intersect_pre_sorted_indices_only(i_rows, j_rows_ptr, i_indices, j_indices, intersect_count)

         ! Build the dense block directly from A_ff (local rows) and submatrix (non-local rows)
         if (.NOT. approx_solve) then
            allocate(submat_vals(i_size, j_size))
            submat_vals = 0
         else
            ! Prevent uninitialised warning
            allocate(submat_vals(0, 0))
            ! Create small SeqAIJ for KSP
            call MatCreate(PETSC_COMM_SELF, small_mat, ierr)
            call MatSetSizes(small_mat, j_size, i_size, j_size, i_size, ierr)
            call MatSetType(small_mat, MATSEQAIJ, ierr)
            call MatSetUp(small_mat, ierr)
         end if

         row_index_into_submatrix = 1
         do j_loc = 1, j_size

            ! Check if this is a non-local row
            if (comm_size /= 1 .AND. &
                (j_rows(j_loc) < global_row_start_aff .OR. j_rows(j_loc) >= global_row_end_plus_one_aff)) then

               ! Non-local row: find position in colmap via walk
               do while (row_index_into_submatrix <= cols_ao .AND. &
                         colmap(row_index_into_submatrix) < j_rows(j_loc))
                  row_index_into_submatrix = row_index_into_submatrix + 1
               end do

               call MatGetRow(reuse_submatrices(1), row_index_into_submatrix - 1, ncols, &
                        cols, vals, ierr)

               if (incomplete) then
                  ! Incomplete: col_indices used for columns, so indices are local
                  ! Convert local col indices to global
                  allocate(cols_global_temp(ncols))
                  cols_global_temp = col_indices_off_proc_array(cols(1:ncols) + 1)

                  allocate(row_i_match(i_size))
                  allocate(row_col_match(ncols))
                  call intersect_pre_sorted_indices_only(i_rows, cols_global_temp, &
                           row_i_match, row_col_match, match_count_sub)
               else
                  ! Full SAI: all_cols_indices used for columns, so indices are already global
                  allocate(row_i_match(i_size))
                  allocate(row_col_match(ncols))
                  call intersect_pre_sorted_indices_only(i_rows, cols(1:ncols), &
                           row_i_match, row_col_match, match_count_sub)
               end if

               if (.NOT. approx_solve) then
                  ! Transpose: row of A_ff becomes column of submat_vals
                  ! as we solve A(J*, I*)^T z(j,J^*)^T = -A_cf(j,I*)^T
                  submat_vals(row_i_match(1:match_count_sub), j_loc) = vals(row_col_match(1:match_count_sub))
               else
                  ! Set row in small_mat (0-based indices)
                  ncols_match = match_count_sub
                  if (incomplete) then
                     cols_global_temp(1:match_count_sub) = row_i_match(1:match_count_sub) - 1
                  else
                     allocate(cols_global_temp(match_count_sub))
                     cols_global_temp = row_i_match(1:match_count_sub) - 1
                  end if
                  call MatSetValues(small_mat, one, [j_loc-1], ncols_match, &
                           cols_global_temp(1:match_count_sub), vals(row_col_match(1:match_count_sub)), &
                           INSERT_VALUES, ierr)
                  if (.NOT. incomplete) deallocate(cols_global_temp)
               end if

               if (incomplete) deallocate(cols_global_temp)
               deallocate(row_i_match, row_col_match)
               call MatRestoreRow(reuse_submatrices(1), row_index_into_submatrix - 1, ncols, &
                        cols, vals, ierr)

            else

               ! Local row: get directly from A_ff
               call MatGetRow(A_ff, j_rows(j_loc), ncols, cols, vals, ierr)

               ! Intersect with i_rows (global) to find matching columns
               allocate(row_i_match(i_size))
               allocate(row_col_match(ncols))
               call intersect_pre_sorted_indices_only(i_rows, cols(1:ncols), &
                        row_i_match, row_col_match, match_count_sub)

               if (.NOT. approx_solve) then
                  ! Transpose: row of A_ff becomes column of submat_vals
                  submat_vals(row_i_match(1:match_count_sub), j_loc) = vals(row_col_match(1:match_count_sub))
               else
                  ! Set row in small_mat (0-based indices)
                  ncols_match = match_count_sub
                  allocate(cols_global_temp(match_count_sub))
                  cols_global_temp = row_i_match(1:match_count_sub) - 1
                  call MatSetValues(small_mat, one, [j_loc-1], ncols_match, &
                           cols_global_temp, vals(row_col_match(1:match_count_sub)), &
                           INSERT_VALUES, ierr)
                  deallocate(cols_global_temp)
               end if

               deallocate(row_i_match, row_col_match)
               call MatRestoreRow(A_ff, j_rows(j_loc), ncols, cols, vals, ierr)

            end if
         end do

         if (approx_solve) then
            call MatAssemblyBegin(small_mat, MAT_FINAL_ASSEMBLY, ierr)
            call MatAssemblyEnd(small_mat, MAT_FINAL_ASSEMBLY, ierr)
         end if

         allocate(e_row(size(i_rows)))
         e_row = 0
         ! Have to stick J^* into the indices of I*
         ! be careful as there is also a minus here
         e_row(i_indices(1:intersect_count)) = -j_vals(j_indices(1:intersect_count))
         
         ! ~~~~~~~~~~~~~            
         ! Solve the square system
         ! ~~~~~~~~~~~~~            
         if (incomplete) then

            ! ~~~~~~~~~~~~~
            ! Sparse approximate solve
            ! ~~~~~~~~~~~~~
            if (approx_solve) then

               call KSPSetOperators(ksp, small_mat, small_mat, ierr)
               call KSPSetUp(ksp, ierr)

               call MatCreateVecs(small_mat, solution, PETSC_NULL_VEC, ierr)
               ! Have to restore the array before the solve in case this is kokkos
               call VecGetArray(solution, vec_vals, ierr)
               vec_vals(1:i_size) = e_row(1:i_size)
               call VecRestoreArray(solution, vec_vals, ierr)

               ! Do the solve - overwrite the rhs
               call KSPSolveTranspose(ksp, solution, solution, ierr)
               call KSPGetIterationNumber(ksp, iterations_taken, ierr)

               call VecGetArray(solution, vec_vals, ierr)
               e_row(1:i_size) = vec_vals(1:i_size)
               call VecRestoreArray(solution, vec_vals, ierr)

               call KSPReset(ksp, ierr)
               call VecDestroy(solution, ierr)
               call MatDestroy(small_mat, ierr)

            ! ~~~~~~~~~~~~~
            ! Exact dense solve
            ! about half the flops to do an LU rather than the QR 
            ! ~~~~~~~~~~~~~
            else

               allocate(pivots(size(i_rows)))
               call dgesv(size(i_rows), 1, submat_vals, size(i_rows), pivots, e_row, size(i_rows), errorcode)
               ! Rearrange given the row permutations done by the LU
               e_row(pivots) = e_row
               deallocate(pivots)

            end if

         ! ~~~~~~~~~~~~~            
         ! Solve the least-squares problem
         ! ~~~~~~~~~~~~~
         else

            ! ~~~~~~~~~~~~~
            ! Sparse approximate solve
            ! ~~~~~~~~~~~~~
            if (approx_solve) then

               ! We can't seem to call KSPSolveTranspose with LSQR, so we explicitly
               ! take a transpose here
               call MatTranspose(small_mat, MAT_INITIAL_MATRIX, transpose_mat, ierr)

               call KSPSetOperators(ksp, transpose_mat, transpose_mat, ierr)
               call KSPSetUp(ksp, ierr)

               call MatCreateVecs(small_mat, rhs, solution, ierr)
               ! Have to restore the array before the solve in case this is kokkos
               call VecGetArray(rhs, vec_vals, ierr)
               vec_vals(1:i_size) = e_row(1:i_size)
               call VecRestoreArray(rhs, vec_vals, ierr)

               ! Do the solve
               call KSPSolve(ksp, rhs, solution, ierr)
               call KSPGetIterationNumber(ksp, iterations_taken, ierr)

               ! Copy solution into e_row
               call VecGetArray(solution, vec_vals, ierr)
               e_row(1:j_size) = vec_vals(1:j_size)
               call VecRestoreArray(solution, vec_vals, ierr)

               call KSPReset(ksp, ierr)
               call VecDestroy(solution, ierr)
               call VecDestroy(rhs, ierr)
               call MatDestroy(transpose_mat, ierr)
               call MatDestroy(small_mat, ierr)

            ! ~~~~~~~~~~~~~
            ! Exact dense solve with QR
            ! ~~~~~~~~~~~~~
            else

               allocate(work(1))
               lwork = -1
               call dgels('N', i_size, j_size, 1, submat_vals, i_size, &
                           e_row, i_size, work, lwork, errorcode)
               lwork = int(work(1))
               deallocate(work)
               allocate(work(lwork))
               call dgels('N', i_size, j_size, 1, submat_vals, i_size, &
                           e_row, i_size, work, lwork, errorcode)
               deallocate(work)

            end if
         end if

         ! ~~~~~~~~~~~~~
         ! Set all the row values
         ! ~~~~~~~~~~~~~
         ! j_rows are global indices for both cases
         if (j_size /= 0) then
            call MatSetValues(z, one, [i_loc], &
                  j_size, j_rows, e_row, INSERT_VALUES, ierr)
         end if

         deallocate(i_rows, e_row, i_indices)
         if (allocated(submat_vals)) deallocate(submat_vals)
      end do

      ! Deallocate pre-allocated arrays
      deallocate(j_rows, j_vals, j_indices)
      if (allocated(col_indices_off_proc_array)) deallocate(col_indices_off_proc_array)

      if (comm_size /= 1 .AND. mat_type /= "mpiaij") then
         call MatDestroy(temp_mat, ierr)
      end if     
      if (deallocate_submatrices) then
         deallocate(reuse_submatrices)   
         reuse_submatrices => null()
      end if
      if (mat_type_input == MATDIAGONAL) then
         call MatDestroy(A_ff, ierr)
      end if      

      call KSPDestroy(ksp, ierr)
      call MatAssemblyBegin(z, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(z, MAT_FINAL_ASSEMBLY, ierr)       

   end subroutine calculate_and_build_sai_z_cpu

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine calculate_and_build_sai_z(A_ff_input, A_cf, sparsity_mat_cf, incomplete, reuse_mat, reuse_submatrices, z)

      ! Wrapper around calculate_and_build_sai_z_cpu and calculate_and_build_sai_z_kokkos
      ! Dispatches to Kokkos for the incomplete (lAIR & isai) case with aijkokkos matrices

      ! ~~~~~~
      type(tMat), intent(in)                            :: A_ff_input, A_cf, sparsity_mat_cf
      logical, intent(in)                               :: incomplete
      type(tMat), intent(inout)                         :: reuse_mat, z
      type(tMat), dimension(:), pointer, intent(inout)  :: reuse_submatrices

#if defined(PETSC_HAVE_KOKKOS)
      integer(c_long_long) :: A_ff_arr, A_cf_arr, sparsity_arr, reuse_arr, z_arr
      integer :: errorcode, reuse_int
      PetscErrorCode :: ierr
      MatType :: mat_type
      logical :: reuse_triggered
      type(tMat) :: temp_mat, temp_mat_reuse, temp_mat_compare
      type(tMat) :: reuse_mat_cpu
      type(tMat), dimension(:), pointer :: reuse_submatrices_cpu
      PetscScalar :: normy
      type(tVec) :: max_vec
      PetscInt :: row_loc
#endif
      ! ~~~~~~     

#if defined(PETSC_HAVE_KOKKOS)

      call MatGetType(A_ff_input, mat_type, ierr)
      if ((mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) .AND. incomplete) then

         ! We're enforcing the same sparsity 
         ! If not re-using
         if (PetscObjectIsNull(z)) then
            call MatDuplicate(sparsity_mat_cf, MAT_DO_NOT_COPY_VALUES, z, ierr)
         end if

         ! We know we will never have non-zero locations outside of the sparsity 
         call MatSetOption(z, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE,  ierr)
         call MatSetOption(z, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)
         call MatSetOption(z, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)                

         ! Extract opaque pointers for C interop
         A_ff_arr = A_ff_input%v
         A_cf_arr = A_cf%v
         sparsity_arr = sparsity_mat_cf%v
         reuse_arr = reuse_mat%v
         z_arr = z%v

         reuse_triggered = .NOT. PetscObjectIsNull(reuse_mat)
         reuse_int = 0
         if (reuse_triggered) reuse_int = 1

         ! Call the Kokkos implementation
         call calculate_and_build_sai_z_kokkos(A_ff_arr, A_cf_arr, sparsity_arr, &
                  reuse_int, reuse_arr, z_arr)

         ! Copy back the opaque pointers
         reuse_mat%v = reuse_arr
         z%v = z_arr

         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then

            ! If we're doing reuse and debug, then we have to always output the result
            ! from the cpu version, as it will have coo preallocation structures set
            if (reuse_triggered) then
               temp_mat = z
               call MatConvert(z, MATSAME, MAT_INITIAL_MATRIX, temp_mat_compare, ierr)
            else
               temp_mat_compare = z
            end if

            ! We send in an empty reuse_mat_cpu here always, as we can't pass through
            ! the same one Kokkos uses as it now only gets out the non-local rows we need
            ! We also disable the approximate solve if the size of any of the dense 
            ! matrices are greater than 40x40, as the kokkos code always does direct solves
            reuse_submatrices_cpu => null()
            call calculate_and_build_sai_z_cpu(A_ff_input, A_cf, sparsity_mat_cf, incomplete, &
                     reuse_mat_cpu, reuse_submatrices_cpu, temp_mat, &
                     no_approx_solve = .TRUE.)
            call destroy_matrix_reuse(reuse_mat_cpu, reuse_submatrices_cpu)

            call MatConvert(temp_mat, MATSAME, MAT_INITIAL_MATRIX, &
                        temp_mat_reuse, ierr)

            call MatAXPYWrapper(temp_mat_reuse, -1d0, temp_mat_compare)
            ! Find the biggest entry in the difference
            call MatCreateVecs(temp_mat_reuse, PETSC_NULL_VEC, max_vec, ierr)
            call MatGetRowMaxAbs(temp_mat_reuse, max_vec, PETSC_NULL_INTEGER_POINTER, ierr)
            call VecMax(max_vec, row_loc, normy, ierr)
            call VecDestroy(max_vec, ierr)

            ! There is floating point compute in these inverses, so we have to be a
            ! bit more tolerant to rounding differences
            if (normy .gt. 4d-11 .OR. normy/=normy) then
               print *, "Diff Kokkos and CPU calculate_and_build_sai_z", normy, "row", row_loc

               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
            end if
            call MatDestroy(temp_mat_reuse, ierr)
            if (.NOT. reuse_triggered) then
               call MatDestroy(z, ierr)
            else
               call MatDestroy(temp_mat_compare, ierr)
            end if
            z = temp_mat
         end if

      else

         call calculate_and_build_sai_z_cpu(A_ff_input, A_cf, sparsity_mat_cf, incomplete, &
                  reuse_mat, reuse_submatrices, z)

      end if
#else
      call calculate_and_build_sai_z_cpu(A_ff_input, A_cf, sparsity_mat_cf, incomplete, &
               reuse_mat, reuse_submatrices, z)
#endif

   end subroutine calculate_and_build_sai_z

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine calculate_and_build_sai(matrix, sparsity_order, incomplete, reuse_mat, reuse_submatrices, inv_matrix)

      ! Computes an approximate inverse with an SAI (or an ISAI)
      ! This just builds an identity and then calls the calculate_and_build_sai_z code

      ! ~~~~~~

      type(tMat), intent(in)                            :: matrix
      integer, intent(in)                               :: sparsity_order
      logical, intent(in)                               :: incomplete
      type(tMat), intent(inout)                         :: reuse_mat, inv_matrix   
      type(tMat), dimension(:), pointer, intent(inout)  :: reuse_submatrices

      type(tMat) :: minus_I, sparsity_mat_cf, A_power
      integer :: order
      PetscErrorCode :: ierr
      logical :: destroy_mat
      integer(c_long_long) :: A_array, B_array, C_array
      PetscInt, parameter ::  one=1, zero=0

      ! ~~~~~~

      ! ~~~~~~~~~~~
      ! Now computing a SAI for Aff is the same as computing Z with an
      ! SAI except we give it -I (the same size of Aff) instead of Acf
      ! So we are going to use the same code 
      ! Means assembling an identity, but that is trivial
      ! ~~~~~~~~~~~

      call generate_identity(matrix, minus_I)
      call MatScale(minus_I, -1d0, ierr)
      
      ! Calculate our approximate inverse
      ! Now given we are using the same code as SAI Z
      ! We have to feed it sparsity_mat_cf to get the approximate inverse sparsity we want
      ! which is just powers of A
      if (sparsity_order == 0) then

         ! Sparsity is just diagonal
         sparsity_mat_cf = minus_I

      else if (sparsity_order == 1) then

         ! Sparsity is just matrix
         sparsity_mat_cf = matrix

      else

         ! If we're not doing reuse
         if (PetscObjectIsNull(inv_matrix)) then

            ! Copy the pointer
            A_power = matrix
            destroy_mat = .FALSE.

            ! Sparsity is a power - A^sparsity_order
            do order = 2, sparsity_order
               
               ! Call a symbolic mult as we don't need the values, just the resulting sparsity  
               A_array = matrix%v
               B_array = A_power%v
               call mat_mat_symbolic_c(A_array, B_array, C_array)
               ! Don't delete the original power - ie matrix
               if (destroy_mat) call MatDestroy(A_power, ierr)
               A_power%v = C_array  
               destroy_mat = .TRUE.

            end do

            sparsity_mat_cf%v = C_array    

         ! Reuse
         else
            call MatDuplicate(inv_matrix, MAT_DO_NOT_COPY_VALUES, sparsity_mat_cf, ierr)

         end if
      end if

      ! Now compute our sparse approximate inverse
      call calculate_and_build_sai_z(matrix, minus_I, sparsity_mat_cf, incomplete, &
               reuse_mat, reuse_submatrices, inv_matrix)     
      call MatDestroy(minus_I, ierr)
      if (sparsity_order .ge. 2) call MatDestroy(sparsity_mat_cf, ierr)      

   end subroutine

end module sai_z

