module petsc_helper

   use iso_c_binding, only: c_ptr, c_null_ptr, c_associated
   use petscmat
   use c_petsc_interfaces, only: remove_small_from_sparse_kokkos, &
         remove_from_sparse_match_kokkos, &
         MatSetAllValues_kokkos, &
         mat_duplicate_copy_plus_diag_kokkos, &
         MatAXPY_kokkos, MatCreateSubMatrix_kokkos
   use pflare_parameters, only: PFLARE_TOL_MATFREE_12, PFLARE_TOL_MATFREE_13, &
         PFLARE_TOL_SIGMA_DROP, PFLARE_TOL_SIGMA_DROP_REL, PFLARE_MINUS_ONE, PFLARE_ZERO, PFLARE_ONE

#include "petsc/finclude/petscmat.h"
#include "petscconf.h"
#include "finclude/pflare_blaslapack.h"

   implicit none

logical, protected :: got_debug_kokkos_env = .FALSE.
logical, protected :: kokkos_debug_global = .FALSE.

   public

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! Some helper functions that involve PETSc matrices, mainly concerning sparsification
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------      

   contains 

   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   function kokkos_debug ()

      ! This checks if an environmental variable is defined 
      ! If it is we check if the Kokkos and CPU versions of routines
      ! are the same

      ! ~~~~~~~~~~~~
      logical      :: kokkos_debug

#if defined(PETSC_HAVE_KOKKOS)       
      integer :: env_val, length, status
      CHARACTER(len=255) :: env_char      
#endif      
      ! ~~~~~~~~~~~~
      
      kokkos_debug = .FALSE.
     
#if defined(PETSC_HAVE_KOKKOS)    

      ! Only get the environmental variable once
      if (got_debug_kokkos_env) then
         kokkos_debug = kokkos_debug_global
      else
         ! Check if the environment variable is set
         call get_environment_variable('PFLARE_KOKKOS_DEBUG', env_char, &
                  length=length, status=status)

         got_debug_kokkos_env = .TRUE.
         if (status /= 1 .AND. length /= 0) then
            read(env_char, '(I3)') env_val
            if (env_val == 1) then
               kokkos_debug_global = .TRUE.
               kokkos_debug = kokkos_debug_global
            end if
         end if
      end if

#endif
      
    end function kokkos_debug   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine destroy_matrix_reuse(mat, submatrices)

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! Input
      type(tMat), intent(inout)                          :: mat
      type(tMat), dimension(:), pointer, intent(inout)   :: submatrices

      ! Local
      PetscInt, parameter :: one = 1
      PetscErrorCode :: ierr      
      type(tMat) :: temp_mat

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
      
      if (associated(submatrices)) then
         submatrices(1) = mat
         ! We only ever do one matrix at a time in matcreatesubmatrices
         call MatDestroySubMatrices(one, submatrices, ierr)
         submatrices => null()
         ! MatDestroySubMatrices doesn't leave the mat in the same null state
         ! as calling matdestroy
         mat = temp_mat
      else
         call MatDestroy(mat, ierr)
      end if
      
   end subroutine     
 
   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine remove_small_from_sparse(input_mat, tol, output_mat, relative_max_row_tol_int, &
                  lump, drop_diagonal_int, diag_strength_int)

      ! Wrapper around remove_small_from_sparse_cpu and remove_small_from_sparse_kokkos
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in) :: input_mat
      type(tMat), intent(inout) :: output_mat
      PetscReal, intent(in) :: tol
      logical, intent(in), optional :: lump
      integer, intent(in), optional :: relative_max_row_tol_int, drop_diagonal_int, diag_strength_int
      
#if defined(PETSC_HAVE_KOKKOS)                     
      integer(c_long_long) :: A_array, B_array
      integer :: lump_int, allow_drop_diagonal_int, allow_diag_strength_int, rel_max_row_tol_int, errorcode
      PetscErrorCode :: ierr
      MatType :: mat_type
      Mat :: temp_mat
      PetscScalar :: normy
      type(tVec) :: max_vec
      PetscInt :: row_loc      
#endif      
      ! ~~~~~~~~~~

#if defined(PETSC_HAVE_KOKKOS)    

      call MatGetType(input_mat, mat_type, ierr)
      if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) then

         ! Absolute tolerance by default   
         rel_max_row_tol_int = 0
         if (present(relative_max_row_tol_int)) then
            rel_max_row_tol_int = relative_max_row_tol_int
         end if
         lump_int = 0
         if (present(lump)) then
            if (lump) then
               lump_int = 1
            end if
         end if   
         ! Never drop the diagonal by default
         allow_drop_diagonal_int = 0
         if (present(drop_diagonal_int)) then
            allow_drop_diagonal_int = drop_diagonal_int
         end if
         ! Whether we use a strength of connection defined by |a_ij| .ge. tol * |a_ii|
         allow_diag_strength_int = 0
         if (present(diag_strength_int)) then
            allow_diag_strength_int = diag_strength_int
         end if                    

         A_array = input_mat%v             
         call remove_small_from_sparse_kokkos(A_array, tol, &
                  B_array, rel_max_row_tol_int, &
                  lump_int, allow_drop_diagonal_int, &
                  allow_diag_strength_int) 
         output_mat%v = B_array

         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then

            ! Debug check if the CPU and Kokkos versions are the same
            call remove_small_from_sparse_cpu(input_mat, tol, temp_mat, relative_max_row_tol_int, &
                     lump, drop_diagonal_int, diag_strength_int)       

            call MatAXPY(temp_mat, PFLARE_MINUS_ONE, output_mat, DIFFERENT_NONZERO_PATTERN, ierr)
            ! Find the biggest entry in the difference
            call MatCreateVecs(temp_mat, PETSC_NULL_VEC, max_vec, ierr)
            call MatGetRowMaxAbs(temp_mat, max_vec, PETSC_NULL_INTEGER_POINTER, ierr)
            call VecMax(max_vec, row_loc, normy, ierr)
            call VecDestroy(max_vec, ierr)

            if (normy .gt. PFLARE_TOL_MATFREE_12 .OR. normy/=normy) then
               !call MatFilter(temp_mat, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
               !call MatView(temp_mat, PETSC_VIEWER_STDOUT_WORLD, ierr)
               print *, "Diff Kokkos and CPU remove_small_from_sparse", normy, "row", row_loc
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)  
            end if
            call MatDestroy(temp_mat, ierr)
         end if

      else

         call remove_small_from_sparse_cpu(input_mat, tol, output_mat, relative_max_row_tol_int, &
                  lump, drop_diagonal_int, diag_strength_int)          

      end if
#else
      call remove_small_from_sparse_cpu(input_mat, tol, output_mat, relative_max_row_tol_int, &
               lump, drop_diagonal_int, diag_strength_int)  
#endif  
     
         
   end subroutine remove_small_from_sparse

   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine remove_small_from_sparse_cpu(input_mat, tol, output_mat, relative_max_row_tol_int, &
                     lump, drop_diagonal_int, diag_strength_int)

      ! Returns a copy of a sparse matrix with entries below abs(val) < tol removed
      ! If rel_max_row_tol_int is 1, then the tol is taken to be a relative scaling 
      ! of the max row val on each row including the diagonal. If it's -1, it doesn't include the diagonal
      ! If lumped is true the removed entries are added to the diagonal
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in) :: input_mat
      type(tMat), intent(inout) :: output_mat
      PetscReal, intent(in) :: tol
      logical, intent(in), optional :: lump
      integer, intent(in), optional :: drop_diagonal_int, relative_max_row_tol_int, diag_strength_int

      PetscInt :: col, ncols, ifree, row_n, diag_pos, target_col, i_loc, max_ncols
      PetscInt :: local_rows, local_cols, global_rows, global_cols, global_row_start
      PetscInt :: global_row_end_plus_one, diag_index
      PetscInt :: global_col_start, global_col_end_plus_one
      PetscErrorCode :: ierr
      PetscInt, dimension(:), pointer :: cols => null()
      ! Matrix entries filled by MatGetRow are PetscScalar
      PetscScalar, dimension(:), pointer :: vals => null()
      ! The output row
      PetscInt, allocatable, dimension(:) :: row_cols, d_nnz, o_nnz
      PetscScalar, allocatable, dimension(:) :: row_vals
      PetscInt, dimension(1) :: row_array
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      logical :: lump_entries
      integer :: drop_diag_int, diag_stren_int, errorcode, rel_max_row_tol_int, pass
      ! Derived magnitude/tolerance stays PetscReal
      PetscReal :: rel_row_tol
      MPIU_Comm :: MPI_COMM_MATRIX
      MatType:: mat_type
      PetscScalar :: abs_biggest_entry, value
      
      ! ~~~~~~~~~~
      ! If the tolerance is 0 we still want to go through this routine and drop the zeros

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)    

      lump_entries = .FALSE.
      ! 1  - Allow drop diagonal
      ! 0  - Never drop diagonal
      ! -1 - Always drop diagonal
      ! Never drop the diagonal by default
      drop_diag_int = 0
      diag_stren_int = 0
      if (present(lump)) lump_entries = lump
      if (present(drop_diagonal_int)) drop_diag_int = drop_diagonal_int
      if (present(diag_strength_int)) diag_stren_int = diag_strength_int
      rel_row_tol = tol
      ! 1  - Relative row tolerance (including diagonal)
      ! 0  - Absolute tolerance
      ! -1 - Relative row tolerance (not including diagonal)
      ! Absolute tolerance by default    
      rel_max_row_tol_int = 0
      if (present(relative_max_row_tol_int)) then
         rel_max_row_tol_int = relative_max_row_tol_int
      end if

      if (lump_entries .AND. drop_diag_int == 1) then
         print *, "Error: Cannot lump and drop the diagonal"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)         
      end if

      ! Get the local sizes
      call MatGetLocalSize(input_mat, local_rows, local_cols, ierr)
      call MatGetSize(input_mat, global_rows, global_cols, ierr)
      ! This returns the global index of the local portion of the matrix
      call MatGetOwnershipRange(input_mat, global_row_start, global_row_end_plus_one, ierr)  
      call MatGetOwnershipRangeColumn(input_mat, global_col_start, global_col_end_plus_one, ierr)  

      call MatCreate(MPI_COMM_MATRIX, output_mat, ierr)
      call MatSetSizes(output_mat, local_rows, local_cols, &
                       global_rows, global_cols, ierr)
      ! Match the output type
      call MatGetType(input_mat, mat_type, ierr)
      call MatSetType(output_mat, mat_type, ierr)

      allocate(d_nnz(local_rows), o_nnz(local_rows))
      d_nnz = 0
      o_nnz = 0
      ! The output row is never longer than the input row plus a diagonal
      max_ncols = 0
      do ifree = global_row_start, global_row_end_plus_one-1
         call MatGetRow(input_mat, ifree, ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
         max_ncols = max(max_ncols, ncols)
         call MatRestoreRow(input_mat, ifree, ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
      end do
      allocate(row_cols(max_ncols + 1), row_vals(max_ncols + 1))
      
      ! We go through the rows twice, the first time to count the entries we keep so we 
      ! can exactly preallocate, the second to set them
      do pass = 1, 2
         do ifree = global_row_start, global_row_end_plus_one-1                  
         
            ! Get the row
            call MatGetRow(input_mat, ifree, ncols, cols, vals, ierr)     

            ! Find where the diagonal is
            diag_index = -1
            if (diag_stren_int == 1) then
               do col = 1, ncols
                  if (cols(col) == ifree) then
                     diag_index = col
                  end if 
               end do              
            end if

            ! If we want a relative row tolerance
            if (rel_max_row_tol_int /= 0) then
               ! Include the diagonal in the relative row tolerance
               if (rel_max_row_tol_int == 1) then
                  if (diag_stren_int == 1) then
                     ! If there is a zero diagonal
                     if (diag_index == -1) then
                        rel_row_tol = 0d0
                     else
                        ! We are measuring relative to the strength of the diagonal
                        rel_row_tol = tol * abs(vals(diag_index))
                     end if
                  else
                     rel_row_tol = tol * maxval(abs(vals(1:ncols)))
                  end if

               ! Don't include the diagonal in the relative row tolerance
               else if (rel_max_row_tol_int == -1) then
                  
                  ! If the user has specified they don't want to include the diagonal in the 
                  ! relative row tolerance, but also have said they want to measure
                  ! everything by the strength of the diagonal, it doesn't make
                  ! sense to exclude the diagonal!
                  if (diag_stren_int == 1) then
                     ! If there is a zero diagonal
                     if (diag_index == -1) then
                        rel_row_tol = 0d0
                     else
                        ! We are measuring relative to the strength of the diagonal
                        rel_row_tol = tol * abs(vals(diag_index))
                     end if                  
                  else
                     ! Be careful to take huge() of the typed target (not 0d0) so
                     ! single builds get the representable -huge, not -Inf
                     abs_biggest_entry = -huge(abs_biggest_entry)
                     ! Find the biggest entry in the row thats not the diagonal
                     do col = 1, ncols
                        if (cols(col) /= ifree .AND. abs(vals(col)) > abs_biggest_entry) then
                           abs_biggest_entry = abs(vals(col))
                        end if 
                     end do  
                     rel_row_tol = tol * abs_biggest_entry
                  end if
               end if
            end if 
                     
            ! Build the output row - anything that ends up on the diagonal (the diagonal itself 
            ! and any lumped entries) is summed into a single diagonal entry
            row_n = 0
            diag_pos = 0
            do col = 1, ncols

               ! Keep it if it is bigger than the tolerance
               if (abs(vals(col)) .ge. rel_row_tol) then

                  ! If this is the diagonal and we are always dropping it, then don't add it
                  if (drop_diag_int == -1 .AND. cols(col) == ifree) cycle
                  target_col = cols(col)

               ! If the entry is small and we are lumping, then add it to the diagonal
               ! or if this is the diagonal and it's small but we are not dropping it
               else if (lump_entries .OR. (drop_diag_int == 0 .AND. cols(col) == ifree)) then
                  target_col = ifree
               else
                  cycle
               end if

               value = vals(col)
               if (target_col == ifree) then
                  if (diag_pos == 0) then
                     row_n = row_n + 1
                     diag_pos = row_n
                     row_cols(row_n) = ifree
                     row_vals(row_n) = value
                  else
                     row_vals(diag_pos) = row_vals(diag_pos) + value
                  end if
               else
                  row_n = row_n + 1
                  row_cols(row_n) = target_col
                  row_vals(row_n) = value
               end if
            end do                       

            ! Must call otherwise petsc leaks memory
            call MatRestoreRow(input_mat, ifree, ncols, cols, vals, ierr)   

            if (pass == 1) then
               i_loc = ifree - global_row_start + 1
               do col = 1, row_n
                  if (row_cols(col) >= global_col_start .AND. row_cols(col) < global_col_end_plus_one) then
                     d_nnz(i_loc) = d_nnz(i_loc) + 1
                  else
                     o_nnz(i_loc) = o_nnz(i_loc) + 1
                  end if
               end do
            else if (row_n > 0) then
               row_array(1) = ifree
               call MatSetValues(output_mat, one, row_array, row_n, row_cols(1:row_n), row_vals(1:row_n), &
                        INSERT_VALUES, ierr)
            end if
         end do

         if (pass == 1) then
            call MatXAIJSetPreallocation(output_mat, one, d_nnz, o_nnz, &
                     PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_INTEGER_ARRAY, ierr)
            ! Don't set any off processor entries so no need for a reduction when assembling
            call MatSetOption(output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)      
            call MatSetOption(output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)     
         end if
      end do
      deallocate(d_nnz, o_nnz, row_cols, row_vals)

      call MatAssemblyBegin(output_mat, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(output_mat, MAT_FINAL_ASSEMBLY, ierr)
         
   end subroutine remove_small_from_sparse_cpu   

   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine remove_from_sparse_match_no_lump(input_mat, output_mat, alpha)

      ! Returns a copy of a sparse matrix with entries that don't match the sparsity
      ! of the other input matrix dropped
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in) :: input_mat
      type(tMat), intent(inout) :: output_mat
      PetscReal, intent(in), optional :: alpha

      PetscInt :: ncols, ifree
      PetscInt :: global_row_start, global_row_end_plus_one
      PetscErrorCode :: ierr
      PetscInt, dimension(:), pointer :: cols => null()
      ! Matrix entries filled by MatGetRow are PetscScalar
      PetscScalar, dimension(:), pointer :: vals => null()
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0

      ! ~~~~~~~~~~
      ! This returns the global index of the local portion of the matrix
      call MatGetOwnershipRange(input_mat, global_row_start, global_row_end_plus_one, ierr)  
       
      ! Just in case there are some zeros in the input mat, ignore them
      call MatSetOption(output_mat, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE, ierr)     
      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)      
      call MatSetOption(output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)    
      ! This ensures any entries outside the existing sparsity of output_mat are dropped 
      ! Not sure if this is respected by the new COO interface so we will keep the 
      ! matsetvalues use below for now
      call MatSetOption(output_mat, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE, ierr)

      ! Now go and fill the new matrix
      ! Loop over global row indices     
      
      ! Add values to existing matrix
      if (present(alpha)) then

         do ifree = global_row_start, global_row_end_plus_one-1                  
         
            ! Get the row
            call MatGetRow(input_mat, ifree, ncols, cols, vals, ierr)
            if (ncols /= 0) then
               call MatSetValues(output_mat, one, [ifree], ncols, cols, &
                     alpha * vals, ADD_VALUES, ierr)
            end if
            call MatRestoreRow(input_mat, ifree, ncols, cols, vals, ierr)                    
   
         end do           

      ! Replace
      else
         do ifree = global_row_start, global_row_end_plus_one-1                  
         
            ! Get the row
            call MatGetRow(input_mat, ifree, ncols, cols, vals, ierr)
            if (ncols /= 0) then
               call MatSetValues(output_mat, one, [ifree], ncols, cols, &
                     vals, INSERT_VALUES, ierr)
            end if
            call MatRestoreRow(input_mat, ifree, ncols, cols, vals, ierr)                    
   
         end do           
      end if
            
      call MatAssemblyBegin(output_mat, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(output_mat, MAT_FINAL_ASSEMBLY, ierr) 
         
   end subroutine remove_from_sparse_match_no_lump   


   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine remove_from_sparse_match(input_mat, output_mat, lump, alpha)

      ! Wrapper around remove_from_sparse_match_cpu and remove_from_sparse_match_kokkos
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in) :: input_mat
      type(tMat), intent(inout) :: output_mat
      logical, intent(in), optional :: lump
      PetscReal, intent(in), optional :: alpha

#if defined(PETSC_HAVE_KOKKOS)                     
      integer(c_long_long) :: A_array, B_array
      integer :: lump_int, errorcode, alpha_int
      PetscErrorCode :: ierr
      MatType :: mat_type
      Mat :: temp_mat
      PetscScalar :: normy
      PetscReal :: alpha_val
      type(tVec) :: max_vec
      PetscInt :: row_loc      
#endif      
      ! ~~~~~~~~~~

#if defined(PETSC_HAVE_KOKKOS)    

      call MatGetType(input_mat, mat_type, ierr)

      if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) then

         lump_int = 0
         if (present(lump)) then
            if (lump) then
               lump_int = 1
            end if
         end if        
         alpha_int = 0
         alpha_val = 1.0
         if (present(alpha)) then
            alpha_int = 1
            alpha_val = alpha
         end if

         if (kokkos_debug()) then
            ! Make sure to copy the values here as we may be doing with alpha
            call MatDuplicate(output_mat, &
                     MAT_COPY_VALUES, temp_mat, ierr)            
         end if

         A_array = input_mat%v             
         B_array = output_mat%v             
         call remove_from_sparse_match_kokkos(A_array, B_array, lump_int, alpha_int, alpha_val) 

         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then

            ! Debug check if the CPU and Kokkos versions are the same
            call remove_from_sparse_match_cpu(input_mat, temp_mat, lump, alpha)      

            call MatAXPY(temp_mat, PFLARE_MINUS_ONE, output_mat, DIFFERENT_NONZERO_PATTERN, ierr)
            ! Find the biggest entry in the difference
            call MatCreateVecs(temp_mat, PETSC_NULL_VEC, max_vec, ierr)
            call MatGetRowMaxAbs(temp_mat, max_vec, PETSC_NULL_INTEGER_POINTER, ierr)
            call VecMax(max_vec, row_loc, normy, ierr)
            call VecDestroy(max_vec, ierr)

            if (normy .gt. PFLARE_TOL_MATFREE_13 .OR. normy/=normy) then
               !call MatFilter(temp_mat, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
               !call MatView(temp_mat, PETSC_VIEWER_STDOUT_WORLD, ierr)
               print *, "Diff Kokkos and CPU remove_from_sparse_match", normy, "row", row_loc
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)  
            end if
            call MatDestroy(temp_mat, ierr)
         end if

      else

         call remove_from_sparse_match_cpu(input_mat, output_mat, lump, alpha)          

      end if
#else
      call remove_from_sparse_match_cpu(input_mat, output_mat, lump, alpha)   
#endif  
         
   end subroutine remove_from_sparse_match   

   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine remove_from_sparse_match_cpu(input_mat, output_mat, lump, alpha)

      ! Returns a copy of a sparse matrix with entries that don't match the sparsity
      ! of the other input matrix dropped
      ! If lumped is true the removed entries are added to the diagonal
      ! If alpha is present, it does output_mat += alpha * input_mat 
      ! on the given sparsity, rather than output_mat = input_mat
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in) :: input_mat
      type(tMat), intent(inout) :: output_mat
      logical, intent(in), optional :: lump
      PetscReal, intent(in), optional :: alpha

      PetscInt :: col, ncols, ifree, max_nnzs, ncols_mod, index1, index2
      PetscInt :: local_rows, local_cols, global_rows, global_cols, global_row_start 
      PetscInt :: global_row_end_plus_one
      PetscInt :: global_col_start, global_col_end_plus_one
      ! The total nnzs (and twice them) can overflow a 32-bit PetscInt
      PetscCount :: counter, max_nnzs_total, max_nnzs_total_two
      PetscErrorCode :: ierr
      integer :: errorcode, comm_size
      PetscInt, dimension(:), pointer :: cols => null(), cols_mod
      ! Matrix entries filled by MatGetRow are PetscScalar
      PetscScalar, dimension(:), pointer :: vals => null(), vals_copy
      PetscInt, allocatable, dimension(:) :: col_indices
      ! Value buffer feeding MatSetValues is PetscScalar
      PetscScalar, allocatable, dimension(:) :: v, lump_vals
      PetscCount, allocatable, dimension(:) :: row_ptr
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      logical :: lump_entries, alpha_present
      PetscReal :: lump_sum
      InsertMode :: insert_mode
      MPIU_Comm :: MPI_COMM_MATRIX
      
      ! ~~~~~~~~~~
      ! If the tolerance is 0 we still want to go through this routine and drop the zeros

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)

      lump_entries = .FALSE.
      if (present(lump)) lump_entries = lump
      alpha_present = .FALSE.
      if (present(alpha)) alpha_present = .TRUE.

      if (.NOT. lump_entries) then
         ! This version is faster
         call remove_from_sparse_match_no_lump(input_mat, output_mat, alpha)
         return
      end if

      ! Get the local sizes
      call MatGetLocalSize(input_mat, local_rows, local_cols, ierr)
      call MatGetSize(input_mat, global_rows, global_cols, ierr)
      ! This returns the global index of the local portion of the matrix
      call MatGetOwnershipRange(input_mat, global_row_start, global_row_end_plus_one, ierr)  
      call MatGetOwnershipRangeColumn(input_mat, global_col_start, global_col_end_plus_one, ierr)  

      max_nnzs = 0
      max_nnzs_total = 0
      max_nnzs_total_two = 0
      do ifree = global_row_start, global_row_end_plus_one-1                  

         call MatGetRow(input_mat, ifree, ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
         if (ncols > max_nnzs) max_nnzs = ncols
         max_nnzs_total = max_nnzs_total + ncols
         call MatRestoreRow(input_mat, ifree, ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
         call MatGetRow(output_mat, ifree, ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
         if (ncols > max_nnzs) max_nnzs = ncols
         max_nnzs_total_two = max_nnzs_total_two + ncols
         call MatRestoreRow(output_mat, ifree, ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)         
      end do

      max_nnzs_total = max(max_nnzs_total, max_nnzs_total_two)

      allocate(cols_mod(max_nnzs))
      allocate(vals_copy(max_nnzs))

      ! Times 2 here in case we are lumping
      allocate(col_indices(max_nnzs_total * 2))
      allocate(v(max_nnzs_total * 2))       
       
      ! Just in case there are some zeros in the input mat, ignore them
      call MatSetOption(output_mat, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE, ierr)     
      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)      
      call MatSetOption(output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)     
      ! This ensures any entries outside the existing sparsity of output_mat are dropped
      ! (e.g., a lumped diagonal entry when output_mat has no diagonal)
      call MatSetOption(output_mat, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE, ierr)

      allocate(lump_vals(local_rows))
      allocate(row_ptr(local_rows + 1))
      
      ! Now go and fill the new matrix
      ! Loop over global row indices
      counter = 1
      do ifree = global_row_start, global_row_end_plus_one-1                  

         row_ptr(ifree - global_row_start + 1) = counter
      
         ! Get the row
         call MatGetRow(input_mat, ifree, ncols, cols, vals, ierr)  
         
         lump_sum = 0
         ncols_mod = ncols
         cols_mod(1:ncols) = cols(1:ncols)
         if (alpha_present) then
            vals_copy(1:ncols) = alpha * vals(1:ncols)
         else
            vals_copy(1:ncols) = vals(1:ncols)
         end if

         ! Must call otherwise petsc leaks memory
         call MatRestoreRow(input_mat, ifree, ncols, cols, vals, ierr)  
         ! Get the sparsity row
         call MatGetRow(output_mat, ifree, ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)    
         
         ! Now loop through and do the intersection
         ! Anything that is not in both is not inserted
         ! cols should be a subset of cols_mod
         index1 = 1
         index2 = 1
         do while (index1 .le. ncols_mod .and. index2 .le. ncols) 
            if (cols_mod(index1) == cols(index2)) then
               index1 = index1 + 1
               index2 = index2 + 1
            elseif (cols_mod(index1) < cols(index2)) then
               ! This value won't be inserted into the matrix
               cols_mod(index1) = -1
               if (lump_entries) lump_sum = lump_sum + vals_copy(index1)
               index1 = index1 + 1
            else
               index2 = index2 + 1
            end if  
         end do
         ! Do the rest 
         do col = index1, ncols_mod
            cols_mod(col) = -1
            if (lump_entries) lump_sum = lump_sum + vals_copy(col)
         end do

         ! Must call otherwise petsc leaks memory
         call MatRestoreRow(output_mat, ifree, ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)      
         
         ! Stick in the intersecting values
         do col = 1, ncols_mod
            if (cols_mod(col) /= -1) then
               col_indices(counter) = cols_mod(col)
               v(counter) = vals_copy(col)
               counter = counter + 1
            end if
         end do

         ! Lumped term to add to the diagonal
         lump_vals(ifree - global_row_start + 1) = lump_sum
      end do  
      row_ptr(local_rows + 1) = counter
            
      deallocate(cols_mod, vals_copy)

      ! Set the values into the existing output_mat
      ! We can't use MatSetPreallocationCOO here, as that resets the sparsity and 
      ! values of output_mat, which would lose the existing values when adding with alpha 
      ! and any entries of output_mat that aren't in input_mat
      ! We also can't MatGetRow on output_mat once we start setting values, hence the 
      ! intersecting values have been stored above
      if (alpha_present) then
         ! If alpha is present, we add the values to the output matrix
         insert_mode = ADD_VALUES
      else
         ! Otherwise we just copy the values across
         insert_mode = INSERT_VALUES
      end if
      do ifree = 1, local_rows
         ncols = int(row_ptr(ifree + 1) - row_ptr(ifree), kind(ncols))
         if (ncols /= 0) then
            call MatSetValues(output_mat, one, [ifree + global_row_start - 1], ncols, &
                  col_indices(row_ptr(ifree):row_ptr(ifree + 1) - 1), &
                  v(row_ptr(ifree):row_ptr(ifree + 1) - 1), insert_mode, ierr)
         end if
      end do
      ! Can't mix insert and add without an assembly in between
      if (.NOT. alpha_present) then
         call MatAssemblyBegin(output_mat, MAT_FLUSH_ASSEMBLY, ierr)
         call MatAssemblyEnd(output_mat, MAT_FLUSH_ASSEMBLY, ierr)
      end if
      ! Add the lumped terms to the diagonal
      do ifree = global_row_start, global_row_end_plus_one-1
         call MatSetValues(output_mat, one, [ifree], one, [ifree], &
               [lump_vals(ifree - global_row_start + 1)], ADD_VALUES, ierr)
      end do

      call MatAssemblyBegin(output_mat, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(output_mat, MAT_FINAL_ASSEMBLY, ierr) 

      deallocate(col_indices, v, lump_vals, row_ptr)  
         
   end subroutine remove_from_sparse_match_cpu

   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine MatSetAllValues(input_mat, val)

      ! Sets all the values in the matrix to be val
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in)  :: input_mat
      PetscScalar, intent(in) :: val

#if defined(PETSC_HAVE_KOKKOS)
      integer(c_long_long) :: A_array                     
      PetscErrorCode :: ierr
      MatType :: mat_type
#endif        
      ! ~~~~~~~~~~


#if defined(PETSC_HAVE_KOKKOS)    

      call MatGetType(input_mat, mat_type, ierr)
      if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) then  

         A_array = input_mat%v
         call MatSetAllValues_kokkos(A_array, val)

      else
         call MatSetAllValues_cpu(input_mat, val)          
      end if
#else
      call MatSetAllValues_cpu(input_mat, val)    
#endif        


   end subroutine MatSetAllValues

   !------------------------------------------------------------------------------------------------------------------------

   subroutine MatSetAllValues_cpu(input_mat, val)

      ! Sets all the values in the local and nonlocal parts of input_mat to be val

      ! ~~~~~~~~~~
      ! Input
      type(tMat), intent(in)  :: input_mat
      PetscScalar, intent(in) :: val

      integer :: comm_size, errorcode
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX
      type(tMat) :: Ad, Ao
      PetscInt, dimension(:), pointer :: colmap => null()
      PetscScalar, dimension(:), pointer :: vals => null()
      ! ~~~~~~~~~~

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)

      if (comm_size /= 1) then
         call MatMPIAIJGetSeqAIJ(input_mat, Ad, Ao, colmap, ierr)
      else
         Ad = input_mat
      end if

      ! MatSeqAIJGetArrayWrite returns a pointer sized to exactly the nonzeros,
      ! so a whole-array assignment sets all the values in the local part
      call MatSeqAIJGetArrayWrite(Ad, vals, ierr)
      vals = val
      call MatSeqAIJRestoreArrayWrite(Ad, vals, ierr)

      ! Now the non-local part
      if (comm_size /= 1) then
         call MatSeqAIJGetArrayWrite(Ao, vals, ierr)
         vals = val
         call MatSeqAIJRestoreArrayWrite(Ao, vals, ierr)
      end if

   end subroutine MatSetAllValues_cpu

  !------------------------------------------------------------------------------------------------------------------------
   
   subroutine mat_duplicate_copy_plus_diag(input_mat, reuse, output_mat)

      ! Wrapper around mat_duplicate_copy_plus_diag_kokkos and mat_duplicate_copy_plus_diag_cpu
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in) :: input_mat
      logical, intent(in) :: reuse
      type(tMat), intent(inout) :: output_mat

#if defined(PETSC_HAVE_KOKKOS)                     
      integer(c_long_long) :: A_array, B_array
      integer :: reuse_int, errorcode
      PetscErrorCode :: ierr
      MatType :: mat_type
      Mat :: temp_mat, temp_mat_reuse, temp_mat_compare
      PetscScalar :: normy
      type(tVec) :: max_vec
      PetscInt :: row_loc      
#endif        
      ! ~~~~~~~~~~


#if defined(PETSC_HAVE_KOKKOS)    

      call MatGetType(input_mat, mat_type, ierr)
      if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) then

         reuse_int = 0
         if (reuse) reuse_int = 1

         A_array = input_mat%v             
         if (reuse) B_array = output_mat%v
         call mat_duplicate_copy_plus_diag_kokkos(A_array, reuse_int, B_array)
         output_mat%v = B_array
         
         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then

            ! If we're doing reuse and debug, then the cpu version reuses the output
            ! matrix itself (as that has the sparsity the cpu version expects) and so
            ! we compare against a copy of the kokkos result
            if (reuse) then
               temp_mat = output_mat
               call MatConvert(output_mat, MATSAME, MAT_INITIAL_MATRIX, temp_mat_compare, ierr)  
            else
               temp_mat_compare = output_mat                 
            end if

            ! Debug check if the CPU and Kokkos versions are the same
            call mat_duplicate_copy_plus_diag_cpu(input_mat, reuse, temp_mat)

            call MatConvert(temp_mat, MATSAME, MAT_INITIAL_MATRIX, &
                        temp_mat_reuse, ierr)                       

            call MatAXPY(temp_mat_reuse, PFLARE_MINUS_ONE, temp_mat_compare, DIFFERENT_NONZERO_PATTERN, ierr)
            ! Find the biggest entry in the difference
            call MatCreateVecs(temp_mat_reuse, PETSC_NULL_VEC, max_vec, ierr)
            call MatGetRowMaxAbs(temp_mat_reuse, max_vec, PETSC_NULL_INTEGER_POINTER, ierr)
            call VecMax(max_vec, row_loc, normy, ierr)
            call VecDestroy(max_vec, ierr)

            if (normy .gt. PFLARE_TOL_MATFREE_13 .OR. normy/=normy) then
               !call MatFilter(temp_mat_reuse, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
               !call MatView(temp_mat_reuse, PETSC_VIEWER_STDOUT_WORLD, ierr)
               print *, "Diff Kokkos and CPU compute_R_from_Z", normy, "row", row_loc
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)  
            end if
            call MatDestroy(temp_mat_reuse, ierr)
            if (.NOT. reuse) then
               call MatDestroy(output_mat, ierr)
            else
               call MatDestroy(temp_mat_compare, ierr)
            end if
            output_mat = temp_mat

         end if

      else

         call mat_duplicate_copy_plus_diag_cpu(input_mat, reuse, output_mat)     

      end if
#else
      call mat_duplicate_copy_plus_diag_cpu(input_mat, reuse, output_mat)
#endif       

         
   end subroutine mat_duplicate_copy_plus_diag  
   

  !------------------------------------------------------------------------------------------------------------------------
   
   subroutine mat_duplicate_copy_plus_diag_cpu(input_mat, reuse, output_mat)

      ! Duplicates and copies the values from input matrix into the output mat, but ensures
      ! there are always diagonal entries present that are set to zero if absent
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in) :: input_mat
      logical, intent(in) :: reuse
      type(tMat), intent(inout) :: output_mat

      PetscInt :: ncols, ifree, col, i_loc
      PetscInt :: local_rows, local_cols, global_rows, global_cols
      PetscInt :: global_row_start, global_row_end_plus_one
      PetscInt :: global_col_start, global_col_end_plus_one
      PetscErrorCode :: ierr
      PetscInt, dimension(:), pointer :: cols => null()
      ! Matrix entries filled by MatGetRow are PetscScalar
      PetscScalar, dimension(:), pointer :: vals => null()
      PetscInt, allocatable, dimension(:) :: d_nnz, o_nnz
      PetscInt, dimension(1) :: row_array
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      MPIU_Comm :: MPI_COMM_MATRIX
      MatType:: mat_type
      logical :: diag_present
      
      ! ~~~~~~~~~~

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)    

      ! Get the local sizes
      call MatGetLocalSize(input_mat, local_rows, local_cols, ierr)
      call MatGetSize(input_mat, global_rows, global_cols, ierr)
      ! This returns the global index of the local portion of the matrix
      call MatGetOwnershipRange(input_mat, global_row_start, global_row_end_plus_one, ierr)  
      call MatGetOwnershipRangeColumn(input_mat, global_col_start, global_col_end_plus_one, ierr)  
      
      ! We may be reusing with the same sparsity
      if (.NOT. reuse) then
         call MatCreate(MPI_COMM_MATRIX, output_mat, ierr)
         call MatSetSizes(output_mat, local_rows, local_cols, &
                           global_rows, global_cols, ierr)
         ! Match the output type
         call MatGetType(input_mat, mat_type, ierr)
         call MatSetType(output_mat, mat_type, ierr)

         ! Count the entries in each row, plus the diagonal if it's absent, so we can exactly preallocate
         allocate(d_nnz(local_rows), o_nnz(local_rows))
         d_nnz = 0
         o_nnz = 0
         do ifree = global_row_start, global_row_end_plus_one-1
            call MatGetRow(input_mat, ifree, ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)
            i_loc = ifree - global_row_start + 1
            diag_present = .FALSE.
            do col = 1, ncols
               if (cols(col) == ifree) diag_present = .TRUE.
               if (cols(col) >= global_col_start .AND. cols(col) < global_col_end_plus_one) then
                  d_nnz(i_loc) = d_nnz(i_loc) + 1
               else
                  o_nnz(i_loc) = o_nnz(i_loc) + 1
               end if
            end do
            if (.NOT. diag_present) d_nnz(i_loc) = d_nnz(i_loc) + 1
            call MatRestoreRow(input_mat, ifree, ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)
         end do
         call MatXAIJSetPreallocation(output_mat, one, d_nnz, o_nnz, &
                  PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_INTEGER_ARRAY, ierr)
         deallocate(d_nnz, o_nnz)
      end if          
       
      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)      
      call MatSetOption(output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)     
      
      ! Now go and fill the new matrix
      ! Loop over global row indices
      do ifree = global_row_start, global_row_end_plus_one-1                  
      
         ! Get the row
         call MatGetRow(input_mat, ifree, ncols, cols, vals, ierr)  
         row_array(1) = ifree
         if (ncols > 0) then
            call MatSetValues(output_mat, one, row_array, ncols, cols(1:ncols), vals(1:ncols), INSERT_VALUES, ierr)
         end if
         ! Add a zero on the diagonal if it's not present
         diag_present = .FALSE.
         do col = 1, ncols
            if (cols(col) == ifree) diag_present = .TRUE.
         end do
         if (.NOT. diag_present) then
            call MatSetValue(output_mat, ifree, ifree, PFLARE_ZERO, INSERT_VALUES, ierr)
         end if

         ! Must call otherwise petsc leaks memory
         call MatRestoreRow(input_mat, ifree, ncols, cols, vals, ierr)   
      end do      

      call MatAssemblyBegin(output_mat, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(output_mat, MAT_FINAL_ASSEMBLY, ierr)
         
   end subroutine mat_duplicate_copy_plus_diag_cpu   

!------------------------------------------------------------------------------------------------------------------------
   
   subroutine MatAXPYWrapper(y_mat, alpha, x_mat)

      ! Wrapper around MatAXPY_kokkos
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(inout) :: y_mat
      type(tMat), intent(in)    :: x_mat
      PetscScalar, intent(in)   :: alpha
      
      PetscErrorCode :: ierr
#if defined(PETSC_HAVE_KOKKOS)                     
      MPIU_Comm :: MPI_COMM_MATRIX
      integer :: comm_size, errorcode
      integer(c_long_long) :: A_array, B_array
      MatType :: mat_type
      Mat :: temp_mat
      PetscScalar :: normy
      type(tVec) :: max_vec
      PetscInt :: row_loc      
#endif      
      ! ~~~~~~~~~~

#if defined(PETSC_HAVE_KOKKOS)    

      call MatGetType(y_mat, mat_type, ierr)
      call PetscObjectGetComm(y_mat, MPI_COMM_MATRIX, ierr)  
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)       

      ! If doing parallel Kokkos
      if ((mat_type == MATMPIAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) .AND. comm_size /= 1) then

         if (kokkos_debug()) then    
            call MatDuplicate(y_mat, MAT_COPY_VALUES, temp_mat, ierr)            
         end if

         A_array = y_mat%v   
         B_array = x_mat%v      
         call MatAXPY_kokkos(A_array, alpha, B_array) 

         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then

            call MatAXPY(temp_mat, alpha, x_mat, DIFFERENT_NONZERO_PATTERN, ierr)    

            call MatAXPY(temp_mat, PFLARE_MINUS_ONE, y_mat, DIFFERENT_NONZERO_PATTERN, ierr)
            ! Find the biggest entry in the difference
            call MatCreateVecs(temp_mat, PETSC_NULL_VEC, max_vec, ierr)
            call MatGetRowMaxAbs(temp_mat, max_vec, PETSC_NULL_INTEGER_POINTER, ierr)
            call VecMax(max_vec, row_loc, normy, ierr)
            call VecDestroy(max_vec, ierr)

            if (normy .gt. PFLARE_TOL_MATFREE_12 .OR. normy/=normy) then
               !call MatFilter(temp_mat, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
               !call MatView(temp_mat, PETSC_VIEWER_STDOUT_WORLD, ierr)
               print *, "Diff Kokkos and CPU MatAXPY", normy, "row", row_loc
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)  
            end if
            call MatDestroy(temp_mat, ierr)
         end if

      else

         call MatAXPY(y_mat, alpha, x_mat, DIFFERENT_NONZERO_PATTERN, ierr)        

      end if
#else
         call MatAXPY(y_mat, alpha, x_mat, DIFFERENT_NONZERO_PATTERN, ierr) 
#endif  
     
         
   end subroutine MatAXPYWrapper   

!------------------------------------------------------------------------------------------------------------------------
   
   subroutine MatCreateSubMatrixWrapper(input_mat, is_row, is_col, &
                  reuse, output_mat, &
                  our_level, is_row_fine, is_col_fine, kokkos_is_views_handle)

      ! Wrapper around MatCreateSubMatrix_kokkos
      ! Only works if the input IS have the same parallel row/column distribution
      ! as the matrices
      ! is_col must be sorted
      ! When our_level is supplied (i.e. the kokkos path uses precomputed per-level
      ! fine/coarse IS views), kokkos_is_views_handle MUST also be supplied; it
      ! is the per-PCAIR handle stored on air_multigrid_data.

      ! ~~~~~~~~~~
      ! Input
      type(tMat), intent(in)        :: input_mat
      IS, intent(in)                :: is_row, is_col
      integer, intent(in), optional :: our_level
      logical, intent(in), optional :: is_row_fine, is_col_fine
      type(c_ptr), intent(in), optional :: kokkos_is_views_handle
      MatReuse, intent(in)          :: reuse
      type(tMat), intent(inout)     :: output_mat

      PetscErrorCode :: ierr
#if defined(PETSC_HAVE_KOKKOS)
      MPIU_Comm :: MPI_COMM_MATRIX
      integer :: comm_size, errorcode
      integer(c_long_long) :: A_array, B_array, is_row_ptr, is_col_ptr
      integer :: reuse_int, our_level_int, is_row_fine_int, is_col_fine_int
      logical :: reuse_logical
      MatType :: mat_type
      Mat :: temp_mat
      PetscScalar :: normy
      type(tVec) :: max_vec
      PetscInt :: row_loc
      type(c_ptr) :: handle_local
#endif
      ! ~~~~~~~~~~

#if defined(PETSC_HAVE_KOKKOS)    

      call MatGetType(input_mat, mat_type, ierr)
      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)  
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)       

      ! If doing parallel Kokkos
      if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) then

         ! Are we reusing
         reuse_logical = reuse == MAT_REUSE_MATRIX
         reuse_int = 0
         if (reuse_logical) reuse_int = 1

         A_array = input_mat%v   
         B_array = output_mat%v      
         is_row_ptr = is_row%v
         is_col_ptr = is_col%v

         our_level_int = -1
         is_row_fine_int = 0
         is_col_fine_int = 0
         handle_local = c_null_ptr

         if (present(our_level)) then
            our_level_int = our_level
         end if
         if (present(is_row_fine)) then
            if (is_row_fine) is_row_fine_int = 1
         end if
         if (present(is_col_fine)) then
            if (is_col_fine) is_col_fine_int = 1
         end if
         if (present(kokkos_is_views_handle)) then
            handle_local = kokkos_is_views_handle
         end if
         if (our_level_int /= -1 .AND. .NOT. c_associated(handle_local)) then
            print *, "MatCreateSubMatrixWrapper: our_level set but kokkos_is_views_handle is null"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
         end if

         call MatCreateSubMatrix_kokkos(A_array, is_row_ptr, is_col_ptr, &
                  reuse_int, B_array, &
                  handle_local, &
                  our_level_int, is_row_fine_int, is_col_fine_int)

         output_mat%v = B_array

         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then

            call MatCreateSubMatrix(input_mat, is_row, is_col, &
                  MAT_INITIAL_MATRIX, temp_mat, ierr)    

            call MatAXPY(temp_mat, PFLARE_MINUS_ONE, output_mat, DIFFERENT_NONZERO_PATTERN, ierr)
            ! Find the biggest entry in the difference
            call MatCreateVecs(temp_mat, PETSC_NULL_VEC, max_vec, ierr)
            call MatGetRowMaxAbs(temp_mat, max_vec, PETSC_NULL_INTEGER_POINTER, ierr)
            call VecMax(max_vec, row_loc, normy, ierr)
            call VecDestroy(max_vec, ierr)

            if (normy .gt. PFLARE_TOL_MATFREE_12 .OR. normy/=normy) then
               !call MatFilter(temp_mat, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
               !call MatView(temp_mat, PETSC_VIEWER_STDOUT_WORLD, ierr)
               print *, "Diff Kokkos and CPU MatCreateSubMatrix", normy, "row", row_loc
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)  
            end if
            call MatDestroy(temp_mat, ierr)
         end if

      else

         call MatCreateSubMatrix(input_mat, is_row, is_col, &
               reuse, output_mat, ierr)        

      end if
#else
         call MatCreateSubMatrix(input_mat, is_row, is_col, &
               reuse, output_mat, ierr) 
#endif  
         
   end subroutine MatCreateSubMatrixWrapper   

   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine generate_identity(input_mat, output_mat)

      ! Returns an assembled identity of matching dimension/type to the input
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in) :: input_mat
      type(tMat), intent(inout) :: output_mat
      
      PetscInt :: ifree, local_rows, local_cols, global_rows, global_cols
      PetscInt :: global_row_start, global_row_end_plus_one
      PetscErrorCode :: ierr
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      PetscInt, allocatable, dimension(:) :: d_nnz, o_nnz
      MPIU_Comm :: MPI_COMM_MATRIX
      MatType:: mat_type
      
      ! ~~~~~~~~~~

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)   

      ! Get the local sizes
      call MatGetLocalSize(input_mat, local_rows, local_cols, ierr)
      call MatGetSize(input_mat, global_rows, global_cols, ierr)
      ! This returns the global index of the local portion of the matrix
      call MatGetOwnershipRange(input_mat, global_row_start, global_row_end_plus_one, ierr)  

      call MatCreate(MPI_COMM_MATRIX, output_mat, ierr)
      call MatSetSizes(output_mat, local_rows, local_cols, &
                       global_rows, global_cols, ierr)
      ! Match the output type
      call MatGetType(input_mat, mat_type, ierr)
      call MatSetType(output_mat, mat_type, ierr)
      ! One entry in the diagonal block of each row
      allocate(d_nnz(local_rows), o_nnz(local_rows))
      d_nnz = 1
      o_nnz = 0
      call MatXAIJSetPreallocation(output_mat, one, d_nnz, o_nnz, &
               PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_INTEGER_ARRAY, ierr)
      deallocate(d_nnz, o_nnz)
      
      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)
      call MatSetOption(output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)    
      
      ! Set the diagonal
      do ifree = global_row_start, global_row_end_plus_one-1
         call MatSetValue(output_mat, ifree, ifree, PFLARE_ONE, INSERT_VALUES, ierr)
      end do
      call MatAssemblyBegin(output_mat, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(output_mat, MAT_FINAL_ASSEMBLY, ierr)
         
   end subroutine generate_identity   
   

   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine generate_identity_rect(full_mat, rect_indices, output_mat)

      ! Returns an assembled (rectangular) injector that pulls out points in the row_indices: looks like [I 0]
      ! The rows are sized from rect_indices (local and global size)
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in)    :: full_mat
      type(tIS), intent(in)     :: rect_indices
      type(tMat), intent(inout) :: output_mat
      
      PetscInt :: i_loc, local_rows, local_cols, global_rows, global_cols
      PetscInt :: global_col_start, global_col_end_plus_one
      PetscInt :: global_row_start_rect, global_row_end_plus_one_rect
      PetscInt :: local_indices_size, global_indices_size
      PetscErrorCode :: ierr
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      PetscInt, allocatable, dimension(:) :: d_nnz, o_nnz
      MPIU_Comm :: MPI_COMM_MATRIX
      MatType:: mat_type
      PetscInt, dimension(:), pointer :: is_pointer
      
      ! ~~~~~~~~~~

      call PetscObjectGetComm(full_mat, MPI_COMM_MATRIX, ierr)   

      ! Get the local sizes
      call MatGetLocalSize(full_mat, local_rows, local_cols, ierr)
      call MatGetSize(full_mat, global_rows, global_cols, ierr)
      call MatGetOwnershipRangeColumn(full_mat, global_col_start, global_col_end_plus_one, ierr)  

      ! Get the local and global sizes of the rows we're pulling out
      call ISGetLocalSize(rect_indices, local_indices_size, ierr)
      call ISGetSize(rect_indices, global_indices_size, ierr)

      call MatCreate(MPI_COMM_MATRIX, output_mat, ierr)
      ! Rectangular matrix 
      call MatSetSizes(output_mat, local_indices_size, local_cols, &
                  global_indices_size, global_cols, ierr)
      ! Match the output type
      call MatGetType(full_mat, mat_type, ierr)
      call MatSetType(output_mat, mat_type, ierr)

      ! Get the indices we need
      call ISGetIndices(rect_indices, is_pointer, ierr)

      ! One entry in each row
      allocate(d_nnz(local_indices_size), o_nnz(local_indices_size))
      d_nnz = 0
      o_nnz = 0
      do i_loc = 1, local_indices_size
         if (is_pointer(i_loc) >= global_col_start .AND. is_pointer(i_loc) < global_col_end_plus_one) then
            d_nnz(i_loc) = 1
         else
            o_nnz(i_loc) = 1
         end if
      end do
      call MatXAIJSetPreallocation(output_mat, one, d_nnz, o_nnz, &
               PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_INTEGER_ARRAY, ierr)
      deallocate(d_nnz, o_nnz)
      ! This returns the global index of the local portion of the output
      call MatGetOwnershipRange(output_mat, global_row_start_rect, global_row_end_plus_one_rect, ierr)       
      
      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)
      call MatSetOption(output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)          

      do i_loc = 1, local_indices_size
         call MatSetValue(output_mat, global_row_start_rect + i_loc-1, is_pointer(i_loc), &
                  PFLARE_ONE, INSERT_VALUES, ierr)
      end do
      call MatAssemblyBegin(output_mat, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(output_mat, MAT_FINAL_ASSEMBLY, ierr)

      call ISRestoreIndices(rect_indices, is_pointer, ierr)    
         
   end subroutine generate_identity_rect   

   !------------------------------------------------------------------------------------------------------------------------

   subroutine generate_identity_is(input_mat, indices, output_mat)

      ! Returns an assembled identity of matching dimension/type to the input
      ! but with ones only in the diagonals of the input IS
      ! We use this to do the equivalent of veciscopy that doesn't have to be 
      ! copied back to the cpu from the gpu
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in)     :: input_mat
      type(tIS), intent(in)      :: indices
      type(tMat), intent(inout)  :: output_mat
      
      PetscInt :: local_rows, local_cols, global_rows, global_cols
      PetscInt :: global_row_start, global_row_end_plus_one
      PetscInt :: local_indices_size, i_loc
      PetscErrorCode :: ierr
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      PetscInt, allocatable, dimension(:) :: d_nnz, o_nnz
      MPIU_Comm :: MPI_COMM_MATRIX
      MatType:: mat_type
      PetscInt, dimension(:), pointer :: is_pointer
      
      ! ~~~~~~~~~~

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)   

      ! Get the local sizes
      call IsGetLocalSize(indices, local_indices_size, ierr)

      call MatGetLocalSize(input_mat, local_rows, local_cols, ierr)
      call MatGetSize(input_mat, global_rows, global_cols, ierr)
      ! This returns the global index of the local portion of the matrix
      call MatGetOwnershipRange(input_mat, global_row_start, global_row_end_plus_one, ierr)  

      call MatCreate(MPI_COMM_MATRIX, output_mat, ierr)
      call MatSetSizes(output_mat, local_rows, local_cols, &
                       global_rows, global_cols, ierr)
      ! Match the output type
      call MatGetType(input_mat, mat_type, ierr)
      call MatSetType(output_mat, mat_type, ierr)
      
      ! Get the indices we need
      call ISGetIndices(indices, is_pointer, ierr)

      ! One entry in the diagonal block of each row in the IS
      allocate(d_nnz(local_rows), o_nnz(local_rows))
      d_nnz = 0
      o_nnz = 0
      do i_loc = 1, local_indices_size
         d_nnz(is_pointer(i_loc) - global_row_start + 1) = 1
      end do
      call MatXAIJSetPreallocation(output_mat, one, d_nnz, o_nnz, &
               PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_INTEGER_ARRAY, ierr)
      deallocate(d_nnz, o_nnz)
      
      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)
      call MatSetOption(output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)  

      ! Set the diagonal
      do i_loc = 1, local_indices_size
         call MatSetValue(output_mat, is_pointer(i_loc), is_pointer(i_loc), PFLARE_ONE, INSERT_VALUES, ierr)
      end do
      call MatAssemblyBegin(output_mat, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(output_mat, MAT_FINAL_ASSEMBLY, ierr)

      call ISRestoreIndices(indices, is_pointer, ierr)       
         
   end subroutine generate_identity_is

   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine get_nnzs_petsc_sparse(input_mat, nnzs)
      ! Returns nnzs in a sparse petsc matrix
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in) :: input_mat
      integer(kind=8), intent(out) :: nnzs

      integer :: comm_size, errorcode
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX
      PetscInt :: local_nnzs_petsc, offdiag_nnzs_petsc
      integer(kind=8) :: local_nnzs
      
      ! ~~~~~~~~~~

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)  
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)      

      ! Get nnzs without using getrow
      call MatGetNNZs_both(input_mat, local_nnzs_petsc, offdiag_nnzs_petsc)
      ! Widen before adding so this can't overflow with 32-bit PetscInt
      local_nnzs = int(local_nnzs_petsc, kind=8) + int(offdiag_nnzs_petsc, kind=8)

      ! Do an accumulate if in parallel
      if (comm_size/=1) then
         call MPI_Allreduce(local_nnzs, nnzs, 1, MPI_INTEGER8, MPI_SUM, MPI_COMM_MATRIX, errorcode)
      else
         nnzs = local_nnzs
      end if    
         
   end subroutine get_nnzs_petsc_sparse

   !-------------------------------------------------------------------------------------------------------------------------------

   subroutine MatGetNNZs_both(input_mat, nnzs_local, nnzs_nonlocal)

      ! Returns the number of nonzeros in the local and nonlocal parts of input_mat

      ! ~~~~~~~~~~
      ! Input
      type(tMat), intent(in)  :: input_mat
      PetscInt, intent(out)   :: nnzs_local, nnzs_nonlocal

      integer :: comm_size, errorcode
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX
      MatType :: mat_type
      type(tMat) :: Ad, Ao
      PetscInt, dimension(:), pointer :: colmap => null(), ia => null(), ja => null()
      PetscInt :: n, shift = 0
      PetscBool :: symmetric = PETSC_FALSE, inodecompressed = PETSC_FALSE, done
      ! ~~~~~~~~~~

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)
      nnzs_nonlocal = 0

      ! A diagonal matrix has one nonzero per row
      call MatGetType(input_mat, mat_type, ierr)
      if (mat_type == MATDIAGONAL) then
         call MatGetLocalSize(input_mat, nnzs_local, PETSC_NULL_INTEGER, ierr)
         return
      end if

      if (comm_size /= 1) then
         call MatMPIAIJGetSeqAIJ(input_mat, Ad, Ao, colmap, ierr)
      else
         Ad = input_mat
      end if

      ! The number of nonzeros is the last entry of the CSR row pointer
      call MatGetRowIJ(Ad, shift, symmetric, inodecompressed, n, ia, ja, done, ierr)
      nnzs_local = ia(n + 1)
      call MatRestoreRowIJ(Ad, shift, symmetric, inodecompressed, n, ia, ja, done, ierr)

      if (comm_size /= 1) then
         call MatGetRowIJ(Ao, shift, symmetric, inodecompressed, n, ia, ja, done, ierr)
         nnzs_nonlocal = ia(n + 1)
         call MatRestoreRowIJ(Ao, shift, symmetric, inodecompressed, n, ia, ja, done, ierr)
      end if

   end subroutine MatGetNNZs_both

   !-------------------------------------------------------------------------------------------------------------------------------

   subroutine svd(input, U, sigma, VT)

      ! ~~~~~~~~~~~~~~~~

      ! Matrix entries are PetscScalar; singular values (sigma) are genuinely real
      PetscScalar, dimension(:, :), intent(in) :: input
      PetscScalar, dimension(size(input, 1), size(input, 1)), intent(out) :: U
      PetscReal, dimension(min(size(input, 1), size(input, 2))), intent(out) :: sigma
      PetscScalar, dimension(size(input, 2), size(input, 2)), intent(out) :: VT

      PetscScalar, dimension(size(input, 1), size(input, 2)) :: tmp_input
      PetscScalar, dimension(:), allocatable :: WORK
      ! BLAS/LAPACK integer arguments must be PetscBLASInt
      PetscBLASInt :: LWORK, M, N, info
      integer :: errorcode

      ! ~~~~~~~~~~~~~~~~

     tmp_input = input
     M = size(input, 1)
     N = size(input, 2)
     LWORK = 2*MAX(1,3*MIN(M,N)+MAX(M,N),5*MIN(M,N))
     allocate(WORK(LWORK))

     call PFLAREgesvd('A', 'A', M, N, tmp_input, M, &
            sigma, U, M, VT, N, WORK, LWORK, info)

      if (info /= 0) then
         print *, "SVD fail"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)         
      end if             
     deallocate(WORK)

    end subroutine svd   

   !-------------------------------------------------------------------------------------------------------------------------------

    subroutine pseudo_inv(input, output)

      ! ~~~~~~~~~~~~~~~~

      ! Matrix entries are PetscScalar; singular values (sigma) are genuinely real
      ! The pseudo inverse of an m x n matrix is n x m
      PetscScalar, dimension(:, :), intent(in) :: input
      PetscScalar, dimension(size(input, 2), size(input, 1)), intent(out) :: output

      PetscScalar, dimension(size(input, 1), size(input, 1)) :: U
      PetscReal, dimension(min(size(input, 1), size(input, 2))) :: sigma
      PetscScalar, dimension(size(input, 2), size(input, 2)) :: VT

      integer :: iloc, errorcode
      PetscReal :: sigma_max, sigma_cutoff
      ! Kind-correct BLAS integer/real arguments for the dgemm call
      PetscBLASInt :: m_bl, n_bl, k_bl
      PetscScalar :: blas_one, blas_zero

      ! ~~~~~~~~~~~~~~~~

      ! Compute the svd
      call svd(input, U, sigma, VT)

      ! Now the pseudoinverse is V * inv(sigma) * U^T
      ! and sigma is diagonal
      ! So scale each column of U (given the transpose)
      ! Drop a singular value if it is below the absolute floor (as before) OR
      ! below the standard RELATIVE pseudo-inverse cutoff, ie a fixed fraction of
      ! this block's largest singular value (numpy.linalg.pinv rcond, pyamg
      ! pinv_array). Taking the max of the two keeps every drop the original
      ! absolute test made - removing the absolute floor regresses the
      ! ex6f -pc_air_constrain_z tests, which rely on it to discard blocks that
      ! are numerically zero - while also discarding directions that are
      ! negligible relative to the block, which matters once there is more than
      ! one constraint vector.
      ! Note neither test can protect a 1x1 block whose single value is small but
      ! well above the floor: the caller must decide whether the block carries
      ! information at all, see PFLARE_TOL_CONSTRAIN_REL in
      ! constrain_grid_transfer.
      sigma_max = 0d0
      do iloc = 1, size(sigma)
         if (abs(sigma(iloc)) > sigma_max) sigma_max = abs(sigma(iloc))
      end do
      sigma_cutoff = max(PFLARE_TOL_SIGMA_DROP, &
                         PFLARE_TOL_SIGMA_DROP_REL * sigma_max)
      ! There are only min(m, n) singular values
      do iloc = 1, size(sigma)
         if (abs(sigma(iloc)) > sigma_cutoff) then
            U(:, iloc) = U(:, iloc) / sigma(iloc)
         else
            U(:, iloc) = 0d0
         end if
      end do

      ! Do the matmatmult, making sure to transpose both
      ! output = V(:, 1:k) * inv(sigma) * U(:, 1:k)^T, with k = min(m, n),
      ! ie the first k rows of VT and the first k (scaled) columns of U
      m_bl = size(input,1)
      n_bl = size(input,2)
      k_bl = size(sigma)
      blas_one = 1d0
      blas_zero = 0d0
      call PFLAREgemm("T", "T", n_bl, m_bl, k_bl, &
               blas_one, VT, n_bl, &
               U, m_bl, &
               blas_zero, output, n_bl)

      ! nan check
      if (any(output /= output)) then
         print *, "NaN in pseudo inverse"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)         
      end if      
  
    end subroutine pseudo_inv

   !-------------------------------------------------------------------------------------------------------------------------------

   ! Computes the symbolic sparsity of A*B (values not needed).
   ! Replaces the old C shim mat_mat_symbolic_c now that the PETSc
   ! Fortran MatProduct interface (incl. MATPRODUCT_AB) exists.
   subroutine mat_mat_symbolic(A, B, C)

      type(tMat), intent(in)  :: A, B
      type(tMat), intent(out) :: C

      MatType :: mat_type_a, mat_type_b
      MPIU_Comm :: MPI_COMM_MATRIX
      integer :: comm_size, errorcode
      PetscErrorCode :: ierr

      ! ~~~~~~~~~~

      call PetscObjectGetComm(A, MPI_COMM_MATRIX, ierr)
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)

      call MatGetType(A, mat_type_a, ierr)
      call MatGetType(B, mat_type_b, ierr)

      if (comm_size == 1) then
         ! For some reason in serial matduplicate is not defined on unassembled matrices
         ! ie we call a matduplicate on the symbolic sparsity_mat returned from this
         ! So we just do an ordinary matmatmult in serial
         call MatMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, C, ierr)
      else
         call MatProductCreate(A, B, PETSC_NULL_MAT, C, ierr)
         call MatProductSetType(C, MATPRODUCT_AB, ierr)
         call MatProductSetAlgorithm(C, 'default', ierr)
         call MatProductSetFill(C, PETSC_DEFAULT_REAL, ierr)
         call MatProductSetFromOptions(C, ierr)
         call MatProductSymbolic(C, ierr)
         call MatProductClear(C, ierr)
      end if

   end subroutine mat_mat_symbolic

   !-------------------------------------------------------------------------------------------------------------------------------

end module petsc_helper

