module petsc_helper

   use petsc
   use c_petsc_interfaces

#include "petsc/finclude/petsc.h"
#include "petscconf.h"
                
   implicit none

#include "petsc_legacy.h"

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
 
   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine remove_small_from_sparse(input_mat, tol, output_mat, relative_max_row_tol_int, lump, drop_diagonal_int)

      ! Wrapper around remove_small_from_sparse_cpu and remove_small_from_sparse_kokkos
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in) :: input_mat
      type(tMat), intent(inout) :: output_mat
      PetscReal, intent(in) :: tol
      logical, intent(in), optional :: lump
      integer, intent(in), optional :: relative_max_row_tol_int, drop_diagonal_int
      
#if defined(PETSC_HAVE_KOKKOS)                     
      integer(c_long_long) :: A_array, B_array
      integer :: lump_int, allow_drop_diagonal_int, rel_max_row_tol_int, errorcode
      PetscErrorCode :: ierr
      MatType :: mat_type
      Mat :: temp_mat
      PetscScalar normy;
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

         A_array = input_mat%v             
         call remove_small_from_sparse_kokkos(A_array, tol, &
                  B_array, rel_max_row_tol_int, lump_int, allow_drop_diagonal_int) 
         output_mat%v = B_array

         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then

            ! Debug check if the CPU and Kokkos versions are the same
            call remove_small_from_sparse_cpu(input_mat, tol, temp_mat, relative_max_row_tol_int, &
                     lump, drop_diagonal_int)       

            call MatAXPY(temp_mat, -1d0, output_mat, DIFFERENT_NONZERO_PATTERN, ierr)
            call MatNorm(temp_mat, NORM_FROBENIUS, normy, ierr)
            if (normy .gt. 1d-13) then
               !call MatFilter(temp_mat, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
               !call MatView(temp_mat, PETSC_VIEWER_STDOUT_WORLD, ierr)
               print *, "Kokkos and CPU versions of remove_small_from_sparse do not match"
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)  
            end if
            call MatDestroy(temp_mat, ierr)
         end if

      else

         call remove_small_from_sparse_cpu(input_mat, tol, output_mat, relative_max_row_tol_int, &
                  lump, drop_diagonal_int)          

      end if
#else
      call remove_small_from_sparse_cpu(input_mat, tol, output_mat, relative_max_row_tol_int, &
               lump, drop_diagonal_int)  
#endif  
     
         
   end subroutine remove_small_from_sparse

   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine remove_small_from_sparse_cpu(input_mat, tol, output_mat, relative_max_row_tol_int, lump, drop_diagonal_int)

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
      integer, intent(in), optional :: drop_diagonal_int, relative_max_row_tol_int

      PetscInt :: col, ncols, ifree, max_nnzs
      PetscInt :: local_rows, local_cols, global_rows, global_cols, global_row_start
      PetscInt :: global_row_end_plus_one, max_nnzs_total
      PetscInt :: counter
      PetscErrorCode :: ierr
      PetscInt, dimension(:), allocatable :: cols
      PetscReal, dimension(:), allocatable :: vals
      PetscInt, allocatable, dimension(:) :: row_indices, col_indices
      PetscReal, allocatable, dimension(:) :: v          
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      logical :: lump_entries
      integer :: drop_diag_int, errorcode, rel_max_row_tol_int
      PetscReal :: rel_row_tol
      MPI_Comm :: MPI_COMM_MATRIX
      MatType:: mat_type
      PetscScalar :: abs_biggest_entry
      
      ! ~~~~~~~~~~
      ! If the tolerance is 0 we still want to go through this routine and drop the zeros

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)    

      lump_entries = .FALSE.
      ! 1  - Allow drop diagonal
      ! 0  - Never drop diagonal
      ! -1 - Always drop diagonal
      ! Never drop the diagonal by default
      drop_diag_int = 0
      if (present(lump)) lump_entries = lump
      if (present(drop_diagonal_int)) drop_diag_int = drop_diagonal_int
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
      
      max_nnzs = 0
      max_nnzs_total = 0
      do ifree = global_row_start, global_row_end_plus_one-1                  

         call MatGetRow(input_mat, ifree, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
         if (ncols > max_nnzs) max_nnzs = ncols
         max_nnzs_total = max_nnzs_total + ncols
         call MatRestoreRow(input_mat, ifree, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
      end do

      allocate(cols(max_nnzs))
      allocate(vals(max_nnzs)) 

      ! We know we never have more to do than the original nnzs
      allocate(row_indices(max_nnzs_total))
      allocate(col_indices(max_nnzs_total))
      ! By default drop everything
      row_indices = -1
      col_indices = -1
      allocate(v(max_nnzs_total))      

      call MatCreate(MPI_COMM_MATRIX, output_mat, ierr)
      call MatSetSizes(output_mat, local_rows, local_cols, &
                       global_rows, global_cols, ierr)
      ! Match the output type
      call MatGetType(input_mat, mat_type, ierr)
      call MatSetType(output_mat, mat_type, ierr)
      call MatSetUp(output_mat, ierr) 
       
      ! Just in case there are some zeros in the input mat, ignore them
      call MatSetOption(output_mat, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE, ierr)     
      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)      
      call MatSetOption(output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)     
      
      ! Now go and fill the new matrix
      ! These loops just set the row and col indices to not be -1
      ! if we are including it in the matrix
      ! Loop over global row indices
      counter = 1
      do ifree = global_row_start, global_row_end_plus_one-1                  
      
         ! Get the row
         call MatGetRow(input_mat, ifree, ncols, cols, vals, ierr)     
         ! Copy in all the values
         v(counter:counter + ncols - 1) = vals(1:ncols)

         ! If we want a relative row tolerance
         if (rel_max_row_tol_int /= 0) then
            ! Include the diagonal in the relative row tolerance
            if (rel_max_row_tol_int == 1) then
               rel_row_tol = tol * maxval(abs(vals(1:ncols)))

            ! Don't include the diagonal in the relative row tolerance
            else if (rel_max_row_tol_int == -1) then
               
               ! Be careful here to use huge(0d0) rather than huge(0)!
               abs_biggest_entry = -huge(0d0)
               ! Find the biggest entry in the row thats not the diagonal
               do col = 1, ncols
                  if (cols(col) /= ifree .AND. abs(vals(col)) > abs_biggest_entry) then
                     abs_biggest_entry = abs(vals(col))
                  end if 
               end do  
               rel_row_tol = tol * abs_biggest_entry
            end if
         end if 
                  
         do col = 1, ncols

            ! Set the row/col to be included (ie not -1) 
            ! if it is bigger than the tolerance
            if (abs(vals(col)) .ge. rel_row_tol) then

               ! If this is the diagonal and we are always dropping it, then don't add it
               if (drop_diag_int == -1 .AND. cols(col) == ifree) cycle
                                 
               row_indices(counter + col - 1) = ifree
               col_indices(counter + col - 1) = cols(col)

            ! If the entry is small and we are lumping, then add it to the diagonal
            ! or if this is the diagonal and it's small but we are not dropping it
            else if (lump_entries .OR. (drop_diag_int == 0 .AND. cols(col) == ifree)) then

               row_indices(counter + col - 1) = ifree
               col_indices(counter + col - 1) = ifree

            end if
         end do                       
         counter = counter + ncols

         ! Must call otherwise petsc leaks memory
         call MatRestoreRow(input_mat, ifree, ncols, cols, vals, ierr)   
      end do           

      deallocate(cols, vals)
      ! Set the values
      call MatSetPreallocationCOO(output_mat, counter-1, row_indices, col_indices, ierr)
      deallocate(row_indices, col_indices)
      call MatSetValuesCOO(output_mat, v, INSERT_VALUES, ierr)    
      deallocate(v)        
         
   end subroutine remove_small_from_sparse_cpu   

   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine remove_from_sparse_match_no_lump(input_mat, output_mat)

      ! Returns a copy of a sparse matrix with entries that don't match the sparsity
      ! of the other input matrix dropped
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in) :: input_mat
      type(tMat), intent(inout) :: output_mat

      PetscInt :: ncols, ifree, max_nnzs
      PetscInt :: global_row_start, global_row_end_plus_one
      PetscErrorCode :: ierr
      PetscInt, dimension(:), allocatable :: cols
      PetscReal, dimension(:), allocatable :: vals
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      
      ! ~~~~~~~~~~
      ! This returns the global index of the local portion of the matrix
      call MatGetOwnershipRange(input_mat, global_row_start, global_row_end_plus_one, ierr)  

      max_nnzs = 0
      do ifree = global_row_start, global_row_end_plus_one-1                  

         call MatGetRow(input_mat, ifree, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
         if (ncols > max_nnzs) max_nnzs = ncols
         call MatRestoreRow(input_mat, ifree, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
         call MatGetRow(output_mat, ifree, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
         if (ncols > max_nnzs) max_nnzs = ncols
         call MatRestoreRow(output_mat, ifree, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)         
      end do

      allocate(cols(max_nnzs))
      allocate(vals(max_nnzs)) 
       
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
      do ifree = global_row_start, global_row_end_plus_one-1                  
      
         ! Get the row
         call MatGetRow(input_mat, ifree, ncols, cols, vals, ierr)
         call MatSetValues(output_mat, one, [ifree], ncols, cols, &
                  vals, INSERT_VALUES, ierr)
         call MatRestoreRow(input_mat, ifree, ncols, cols, vals, ierr)                    
 
      end do  
            
      call MatAssemblyBegin(output_mat, MAT_FINAL_ASSEMBLY, ierr)
      deallocate(cols, vals)
      call MatAssemblyEnd(output_mat, MAT_FINAL_ASSEMBLY, ierr) 
         
   end subroutine remove_from_sparse_match_no_lump   


   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine remove_from_sparse_match(input_mat, sparsity_mat, output_mat, lump)

      ! Returns a copy of a sparse matrix with entries that don't match the sparsity
      ! of the other input matrix dropped
      ! If lumped is true the removed entries are added to the diagonal
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in) :: input_mat, sparsity_mat
      type(tMat), intent(inout) :: output_mat
      logical, intent(in), optional :: lump

      PetscInt :: col, ncols, ifree, max_nnzs, ncols_mod, index1, index2, counter
      PetscInt :: local_rows, local_cols, global_rows, global_cols, global_row_start 
      PetscInt :: global_row_end_plus_one, max_nnzs_total_two
      PetscInt :: global_col_start, global_col_end_plus_one, max_nnzs_total
      PetscErrorCode :: ierr
      integer :: errorcode, comm_size
      PetscInt, dimension(:), allocatable :: cols, cols_mod
      PetscReal, dimension(:), allocatable :: vals, vals_copy
      PetscInt, allocatable, dimension(:) :: row_indices, col_indices
      PetscReal, allocatable, dimension(:) :: v        
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      logical :: lump_entries
      PetscReal :: lump_sum
      MPI_Comm :: MPI_COMM_MATRIX
      
      ! ~~~~~~~~~~
      ! If the tolerance is 0 we still want to go through this routine and drop the zeros

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)

      lump_entries = .FALSE.
      if (present(lump)) lump_entries = lump

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

         call MatGetRow(input_mat, ifree, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
         if (ncols > max_nnzs) max_nnzs = ncols
         max_nnzs_total = max_nnzs_total + ncols
         call MatRestoreRow(input_mat, ifree, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
         call MatGetRow(sparsity_mat, ifree, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
         if (ncols > max_nnzs) max_nnzs = ncols
         max_nnzs_total_two = max_nnzs_total_two + ncols
         call MatRestoreRow(sparsity_mat, ifree, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)         
      end do

      max_nnzs_total = max(max_nnzs_total, max_nnzs_total_two)

      allocate(cols(max_nnzs))
      allocate(cols_mod(max_nnzs))
      allocate(vals(max_nnzs)) 
      allocate(vals_copy(max_nnzs))

      ! Times 2 here in case we are lumping
      allocate(row_indices(max_nnzs_total * 2))
      allocate(col_indices(max_nnzs_total * 2))
      allocate(v(max_nnzs_total * 2))       

      ! Duplicate the sparsity
      call MatDuplicate(sparsity_mat, MAT_DO_NOT_COPY_VALUES, output_mat, ierr)
       
      ! Just in case there are some zeros in the input mat, ignore them
      call MatSetOption(output_mat, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE, ierr)     
      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)      
      call MatSetOption(output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)     
      
      ! Now go and fill the new matrix
      ! Loop over global row indices
      counter = 1
      do ifree = global_row_start, global_row_end_plus_one-1                  
      
         ! Get the row
         call MatGetRow(input_mat, ifree, ncols, cols, vals, ierr)  
         
         lump_sum = 0
         ncols_mod = ncols
         cols_mod(1:ncols) = cols(1:ncols)
         vals_copy(1:ncols) = vals(1:ncols)

         ! Must call otherwise petsc leaks memory
         call MatRestoreRow(input_mat, ifree, ncols, cols, vals, ierr)  
         ! Get the sparsity row
         call MatGetRow(sparsity_mat, ifree, ncols, cols, PETSC_NULL_SCALAR_ARRAY, ierr)    
         
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
         call MatRestoreRow(sparsity_mat, ifree, ncols, cols, PETSC_NULL_SCALAR_ARRAY, ierr)      
         
         ! Stick in the intersecting values
         do col = 1, ncols_mod
            if (cols_mod(col) /= -1) then
               row_indices(counter) = ifree
               col_indices(counter) = cols_mod(col)
               v(counter) = vals_copy(col)
               counter = counter + 1
            end if
         end do

         ! Add lumped terms to the diagonal
         if (lump_entries) then
            row_indices(counter) = ifree
            col_indices(counter) = ifree
            v(counter) = lump_sum
            counter = counter + 1
         end if              
      end do  
            
      deallocate(cols, vals, cols_mod, vals_copy)
      ! Set the values
      call MatSetPreallocationCOO(output_mat, counter-1, row_indices, col_indices, ierr)
      deallocate(row_indices, col_indices)
      call MatSetValuesCOO(output_mat, v, INSERT_VALUES, ierr)    
      deallocate(v)  
         
   end subroutine remove_from_sparse_match

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

      ! Sets all the values in the matrix to be val
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in)  :: input_mat
      PetscScalar, intent(in) :: val

      MPI_Comm :: MPI_COMM_MATRIX
      integer :: errorcode, comm_size
      PetscErrorCode :: ierr
      type(tMat) :: Ad, Ao
      PetscOffset :: iicol
      PetscInt :: icol(1)      
      PetscScalar, pointer :: xx_v(:)
      PetscInt, dimension(:), pointer :: ad_ia, ad_ja, ao_ia, ao_ja
      PetscInt :: shift = 0, n_ad, n_ao, local_rows, local_cols
      logical :: symmetric = .false., inodecompressed=.false., done      

      ! ~~~~~~~~~~

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)      
      call MatGetLocalSize(input_mat, local_rows, local_cols, ierr)

      if (comm_size /= 1) then
         ! Let's get the diagonal and off-diagonal parts of the strength matrix
         call MatMPIAIJGetSeqAIJ(input_mat, Ad, Ao, icol, iicol, ierr) 
      else
         Ad = input_mat    
      end if      

      ! Need to know how many nnzs in xx_v, as its size isn't set
      call MatGetRowIJF90(Ad,shift,symmetric,inodecompressed,n_ad,ad_ia,ad_ja,done,ierr) 
      if (.NOT. done) then
         print *, "Pointers not set in call to MatGetRowIJF90"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      if (comm_size /= 1) then
         call MatGetRowIJF90(Ao,shift,symmetric,inodecompressed,n_ao,ao_ia,ao_ja,done,ierr) 
         if (.NOT. done) then
            print *, "Pointers not set in call to MatGetRowIJF90"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
         end if
      end if       

      ! Sequential part
      call MatSeqAIJGetArrayF90(Ad,xx_v,ierr)
      xx_v(1:ad_ia(local_rows+1)) = val
      call MatSeqAIJRestoreArrayF90(Ad,xx_v,ierr)
     
      ! MPI part
      if (comm_size /= 1) then
         call MatSeqAIJGetArrayF90(Ao,xx_v,ierr)
         xx_v(1:ao_ia(local_rows+1)) = val
         call MatSeqAIJRestoreArrayF90(Ao,xx_v,ierr)         
      end if   
      
      ! Restore the sequantial pointers once we're done
      call MatRestoreRowIJF90(Ad,shift,symmetric,inodecompressed,n_ad,ad_ia,ad_ja,done,ierr) 
      if (comm_size /= 1) then
         call MatRestoreRowIJF90(Ao,shift,symmetric,inodecompressed,n_ao,ao_ia,ao_ja,done,ierr) 
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
      PetscScalar normy;
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

            ! If we're doing reuse and debug, then we have to always output the result 
            ! from the cpu version, as it will have coo preallocation structures set
            ! They aren't copied over if you do a matcopy (or matconvert)
            ! If we didn't do that the next time we come through this routine 
            ! and try to call the cpu version with reuse, it will segfault
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

            call MatAXPY(temp_mat_reuse, -1d0, temp_mat_compare, DIFFERENT_NONZERO_PATTERN, ierr)
            call MatNorm(temp_mat_reuse, NORM_FROBENIUS, normy, ierr)
            if (normy .gt. 1d-13) then
               !call MatFilter(temp_mat_reuse, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
               !call MatView(temp_mat_reuse, PETSC_VIEWER_STDOUT_WORLD, ierr)
               print *, "Kokkos and CPU versions of compute_R_from_Z do not match"
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

      PetscInt :: ncols, ifree, max_nnzs, max_nnzs_total
      PetscInt :: local_rows, local_cols, global_rows, global_cols
      PetscInt :: global_row_start, global_row_end_plus_one
      PetscInt :: counter
      PetscErrorCode :: ierr
      integer :: errorcode, comm_size
      PetscInt, dimension(:), allocatable :: cols
      PetscReal, dimension(:), allocatable :: vals
      PetscInt, allocatable, dimension(:) :: row_indices, col_indices
      PetscReal, allocatable, dimension(:) :: v         
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      MPI_Comm :: MPI_COMM_MATRIX
      MatType:: mat_type
      
      ! ~~~~~~~~~~
      ! If the tolerance is 0 we still want to go through this routine and drop the zeros

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)

      ! Get the local sizes
      call MatGetLocalSize(input_mat, local_rows, local_cols, ierr)
      call MatGetSize(input_mat, global_rows, global_cols, ierr)
      ! This returns the global index of the local portion of the matrix
      call MatGetOwnershipRange(input_mat, global_row_start, global_row_end_plus_one, ierr)  
      
      ! Get the max number of nnzs
      max_nnzs_total = 0
      max_nnzs = -1
      do ifree = global_row_start, global_row_end_plus_one-1
         call MatGetRow(input_mat, ifree, &
                  ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
         max_nnzs_total = max_nnzs_total + ncols
         if (ncols > max_nnzs) max_nnzs = ncols                  
         call MatRestoreRow(input_mat, ifree, &
                  ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)          
      end do
      ! Adding for potential missing diagonals
      max_nnzs_total = max_nnzs_total + local_rows
      max_nnzs = max_nnzs + 1

      allocate(cols(max_nnzs))
      allocate(vals(max_nnzs))

      allocate(row_indices(max_nnzs_total))
      allocate(col_indices(max_nnzs_total))
      allocate(v(max_nnzs_total))

      ! We may be reusing with the same sparsity
      if (.NOT. reuse) then
         call MatCreate(MPI_COMM_MATRIX, output_mat, ierr)
         call MatSetSizes(output_mat, local_rows, local_cols, &
                           global_rows, global_cols, ierr)
         ! Match the output type
         call MatGetType(input_mat, mat_type, ierr)
         call MatSetType(output_mat, mat_type, ierr)
         call MatSetUp(output_mat, ierr)   
      end if          
       
      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)      
      call MatSetOption(output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)     
      
      ! Now go and fill the new matrix
      ! Loop over global row indices
      counter = 1
      do ifree = global_row_start, global_row_end_plus_one-1                  
      
         ! Get the row
         call MatGetRow(input_mat, ifree, ncols, cols, vals, ierr)  

         row_indices(counter:counter+ncols-1) = ifree
         col_indices(counter:counter+ncols-1) = cols(1:ncols)
         v(counter:counter+ncols-1) = vals(1:ncols)    
         counter = counter + ncols           

         ! Must call otherwise petsc leaks memory
         call MatRestoreRow(input_mat, ifree, ncols, cols, vals, ierr)   
      end do      
      
      ! Go and add a zero on the diagonal just in case it's not present
      do ifree = global_row_start, global_row_end_plus_one-1 
         row_indices(counter) = ifree
         col_indices(counter) = ifree
         v(counter) = 0d0
         counter = counter + 1
      end do

      ! Set the values
      if (.NOT. reuse) then
         call MatSetPreallocationCOO(output_mat, counter-1, row_indices, col_indices, ierr)
      end if
      deallocate(row_indices, col_indices)
      ! Remember the COO format does the add of all the values in v that share an index
      ! So zero gets added to every diagonal entry (that way they're always present)
      ! even though we're calling INSERT_VALUES
      call MatSetValuesCOO(output_mat, v, INSERT_VALUES, ierr)    
      deallocate(v)       
      
      deallocate(cols, vals)

         
   end subroutine mat_duplicate_copy_plus_diag_cpu   

   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine generate_identity(input_mat, output_mat)

      ! Returns an assembled identity of matching dimension/type to the input
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in) :: input_mat
      type(tMat), intent(inout) :: output_mat
      
      PetscInt :: i_loc, local_rows, local_cols, global_rows, global_cols
      PetscInt :: global_row_start, global_row_end_plus_one
      PetscInt, allocatable, dimension(:) :: indices
      PetscReal, allocatable, dimension(:) :: v
      PetscErrorCode :: ierr
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      MPI_Comm :: MPI_COMM_MATRIX
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
      call MatSetUp(output_mat, ierr) 
      
      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)
      call MatSetOption(output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)    
      
      allocate(indices(local_rows))
      allocate(v(local_rows))
      do i_loc = 1, local_rows
         indices(i_loc) = global_row_start + i_loc-1
      end do
      v = 1d0
      ! Set the diagonal
      call MatSetPreallocationCOO(output_mat, local_rows, indices, indices, ierr)
      deallocate(indices)
      call MatSetValuesCOO(output_mat, v, INSERT_VALUES, ierr)    
      deallocate(v)
         
   end subroutine generate_identity   
   

   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine generate_identity_rect(full_mat, rect_mat, rect_indices, output_mat)

      ! Returns an assembled (rectangular) injector that pulls out points in the row_indices: looks like [I 0]
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in)    :: full_mat, rect_mat
      type(tIS), intent(in)     :: rect_indices
      type(tMat), intent(inout) :: output_mat
      
      PetscInt :: i_loc, local_rows, local_cols, global_rows, global_cols
      PetscInt :: global_row_start, global_row_end_plus_one
      PetscInt :: local_rows_rect, local_cols_rect, global_rows_rect, global_cols_rect
      PetscInt :: global_row_start_rect, global_row_end_plus_one_rect
      PetscInt :: local_indices_size
      PetscInt, allocatable, dimension(:) :: row_indices, col_indices
      PetscReal, allocatable, dimension(:) :: v      
      PetscErrorCode :: ierr
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      MPI_Comm :: MPI_COMM_MATRIX
      MatType:: mat_type
      PetscInt, dimension(:), pointer :: is_pointer
      
      ! ~~~~~~~~~~

      call PetscObjectGetComm(full_mat, MPI_COMM_MATRIX, ierr)   

      ! Get the local sizes
      call MatGetLocalSize(full_mat, local_rows, local_cols, ierr)
      call MatGetSize(full_mat, global_rows, global_cols, ierr)
      ! This returns the global index of the local portion of the matrix
      call MatGetOwnershipRange(full_mat, global_row_start, global_row_end_plus_one, ierr)  

      call MatGetLocalSize(rect_mat, local_rows_rect, local_cols_rect, ierr)
      call MatGetSize(rect_mat, global_rows_rect, global_cols_rect, ierr)
      ! This returns the global index of the local portion of the matrix
      call MatGetOwnershipRange(rect_mat, global_row_start_rect, global_row_end_plus_one_rect, ierr)       

      ! Get the local sizes
      call IsGetLocalSize(rect_indices, local_indices_size, ierr)

      call MatCreate(MPI_COMM_MATRIX, output_mat, ierr)
      ! Rectangular matrix 
      call MatSetSizes(output_mat, local_rows_rect, local_cols, &
                  global_rows_rect, global_cols, ierr)
      ! Match the output type
      call MatGetType(full_mat, mat_type, ierr)
      call MatSetType(output_mat, mat_type, ierr)
      call MatSetUp(output_mat, ierr) 
      
      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)
      call MatSetOption(output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)          

      ! Get the indices we need
      call ISGetIndicesF90(rect_indices, is_pointer, ierr)

      allocate(row_indices(local_indices_size))
      allocate(col_indices(local_indices_size))
      allocate(v(local_indices_size))
      do i_loc = 1, local_indices_size
         row_indices(i_loc) = global_row_start_rect + i_loc-1
      end do
      v = 1d0
      ! MatSetPreallocationCOO could modify the values in is_pointer
      col_indices = is_pointer
      ! Set the diagonal
      call MatSetPreallocationCOO(output_mat, local_indices_size, row_indices, col_indices, ierr)
      deallocate(row_indices, col_indices)
      call MatSetValuesCOO(output_mat, v, INSERT_VALUES, ierr)    
      deallocate(v)      

      call ISRestoreIndicesF90(rect_indices, is_pointer, ierr)    
         
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
      PetscInt :: local_indices_size
      PetscReal, allocatable, dimension(:) :: v      
      PetscErrorCode :: ierr
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      MPI_Comm :: MPI_COMM_MATRIX
      MatType:: mat_type
      PetscInt, dimension(:), pointer :: is_pointer
      PetscInt, allocatable, dimension(:) :: row_indices, col_indices
      
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
      call MatSetUp(output_mat, ierr) 
      
      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)
      call MatSetOption(output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)  
      
      ! Get the indices we need
      call ISGetIndicesF90(indices, is_pointer, ierr)

      allocate(v(local_indices_size))
      allocate(row_indices(local_indices_size))
      allocate(col_indices(local_indices_size))
      ! MatSetPreallocationCOO could modify the values in is_pointer
      row_indices = is_pointer
      col_indices = row_indices
      v = 1d0
      ! Set the diagonal
      call MatSetPreallocationCOO(output_mat, local_indices_size, row_indices, col_indices, ierr)
      deallocate(row_indices, col_indices)
      call MatSetValuesCOO(output_mat, v, INSERT_VALUES, ierr)    
      deallocate(v)  

      call ISRestoreIndicesF90(indices, is_pointer, ierr)       
         
   end subroutine generate_identity_is   

   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine generate_one_point_with_one_entry_from_sparse(input_mat, output_mat)

      ! Wrapper around generate_one_point_with_one_entry_from_sparse_kokkos and 
      ! generate_one_point_with_one_entry_from_sparse_cpu
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in) :: input_mat
      type(tMat), intent(inout) :: output_mat
      
#if defined(PETSC_HAVE_KOKKOS)                     
      integer(c_long_long) :: A_array, B_array
      integer :: errorcode
      PetscErrorCode :: ierr
      MatType :: mat_type
      Mat :: temp_mat
      PetscScalar normy;
#endif      
      ! ~~~~~~~~~~

#if defined(PETSC_HAVE_KOKKOS)    

      call MatGetType(input_mat, mat_type, ierr)
      if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) then      

         A_array = input_mat%v             
         call generate_one_point_with_one_entry_from_sparse_kokkos(A_array, B_array) 
         output_mat%v = B_array

         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then

            ! Debug check if the CPU and Kokkos versions are the same
            call generate_one_point_with_one_entry_from_sparse_cpu(input_mat, temp_mat)      

            call MatAXPY(temp_mat, -1d0, output_mat, DIFFERENT_NONZERO_PATTERN, ierr)
            call MatNorm(temp_mat, NORM_FROBENIUS, normy, ierr)
            if (normy .gt. 1d-13) then
               !call MatFilter(temp_mat, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
               !call MatView(temp_mat, PETSC_VIEWER_STDOUT_WORLD, ierr)
               print *, "Kokkos and CPU versions of generate_one_point_with_one_entry_from_sparse do not match"
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)  
            end if
            call MatDestroy(temp_mat, ierr)
         end if

      else

         call generate_one_point_with_one_entry_from_sparse_cpu(input_mat, output_mat)         

      end if
#else
      call generate_one_point_with_one_entry_from_sparse_cpu(input_mat, output_mat)  
#endif 

         
   end subroutine generate_one_point_with_one_entry_from_sparse    

  !------------------------------------------------------------------------------------------------------------------------
   
   subroutine generate_one_point_with_one_entry_from_sparse_cpu(input_mat, output_mat)

      ! Returns a copy of a sparse matrix, but with only one in the spot of the biggest entry
      ! This can be used to generate a classical one point prolongator for example
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in) :: input_mat
      type(tMat), intent(inout) :: output_mat
      
      PetscInt :: ncols, ifree, max_nnzs
      PetscInt :: local_rows, local_cols, global_rows, global_cols
      PetscInt :: global_row_start, global_row_end_plus_one
      PetscInt :: global_col_start, global_col_end_plus_one, counter
      PetscInt, allocatable, dimension(:) :: row_indices, col_indices
      PetscReal, allocatable, dimension(:) :: v
      PetscErrorCode :: ierr
      PetscInt, dimension(:), allocatable :: cols
      PetscReal, dimension(:), allocatable :: vals
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      integer :: max_loc(1)
      integer :: comm_size, errorcode
      MPI_Comm :: MPI_COMM_MATRIX
      MatType:: mat_type
      
      ! ~~~~~~~~~~

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)  
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)

      ! Get the local sizes
      call MatGetLocalSize(input_mat, local_rows, local_cols, ierr)
      call MatGetSize(input_mat, global_rows, global_cols, ierr)
      ! This returns the global index of the local portion of the matrix
      call MatGetOwnershipRange(input_mat, global_row_start, global_row_end_plus_one, ierr)  
      call MatGetOwnershipRangeColumn(input_mat, global_col_start, global_col_end_plus_one, ierr)  

      max_nnzs = 0
      do ifree = global_row_start, global_row_end_plus_one-1                  

         call MatGetRow(input_mat, ifree, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
         if (ncols > max_nnzs) max_nnzs = ncols
         call MatRestoreRow(input_mat, ifree, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
      end do

      allocate(cols(max_nnzs))
      allocate(vals(max_nnzs))       

      ! ! Create the output matrix
      call MatCreate(MPI_COMM_MATRIX, output_mat, ierr)
      call MatSetSizes(output_mat, local_rows, local_cols, &
                       global_rows, global_cols, ierr)
      ! Match the output type
      call MatGetType(input_mat, mat_type, ierr)
      call MatSetType(output_mat, mat_type, ierr)
      call MatSetUp(output_mat, ierr)      
      
      ! Just in case there are some zeros in the input mat, ignore them
      call MatSetOption(output_mat, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE, ierr)   
      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)        
      call MatSetOption(output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)    

      ! We know we only have one entry per row
      allocate(row_indices(local_rows))
      allocate(col_indices(local_rows))
      allocate(v(local_rows))
      v = 1d0
      
      ! Now go and fill the new matrix
      ! Loop over global row indices
      counter = 1
      do ifree = global_row_start, global_row_end_plus_one-1                  
      
         ! Get the row
         call MatGetRow(input_mat, ifree, ncols, cols, vals, ierr)   
         if (ncols /= 0) then
            max_loc = maxloc(abs(vals(1:ncols)))
            row_indices(counter) = ifree
            col_indices(counter) = cols(max_loc(1))         
            counter = counter + 1
         end if    
   
         ! Must call otherwise petsc leaks memory
         call MatRestoreRow(input_mat, ifree, ncols, cols, vals, ierr)   
      end do         
      
      ! Set the values
      call MatSetPreallocationCOO(output_mat, counter-1, row_indices, col_indices, ierr)
      deallocate(row_indices, col_indices)
      call MatSetValuesCOO(output_mat, v, INSERT_VALUES, ierr)    
      deallocate(v)      

      deallocate(cols, vals)
         
   end subroutine generate_one_point_with_one_entry_from_sparse_cpu   
   
!------------------------------------------------------------------------------------------------------------------------
   
   subroutine compute_P_from_W(W, global_row_start, is_fine, is_coarse, identity, reuse, P)

      ! Wrapper around compute_P_from_W_cpu and compute_P_from_W_kokkos 
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(inout) :: W, P
      type(tIS), intent(in)     :: is_fine, is_coarse
      PetscInt, intent(in)      :: global_row_start
      logical, intent(in) :: identity, reuse

#if defined(PETSC_HAVE_KOKKOS)                     
      integer(c_long_long) :: A_array, B_array, indices_fine, indices_coarse
      integer :: identity_int, reuse_int, errorcode
      PetscErrorCode :: ierr
      MatType :: mat_type
      Mat :: temp_mat, temp_mat_reuse, temp_mat_compare
      PetscScalar normy;
#endif        
      ! ~~~~~~~~~~


#if defined(PETSC_HAVE_KOKKOS)    

      call MatGetType(W, mat_type, ierr)
      if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) then

         identity_int = 0
         if (identity) identity_int = 1
         reuse_int = 0
         if (reuse) reuse_int = 1

         A_array = W%v             
         indices_fine = is_fine%v
         indices_coarse = is_coarse%v
         if (reuse) B_array = P%v
         call compute_P_from_W_kokkos(A_array, global_row_start, &
                     indices_fine, indices_coarse, &
                     identity_int, reuse_int, B_array)
         P%v = B_array
         
         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then

            ! If we're doing reuse and debug, then we have to always output the result 
            ! from the cpu version, as it will have coo preallocation structures set
            ! They aren't copied over if you do a matcopy (or matconvert)
            ! If we didn't do that the next time we come through this routine 
            ! and try to call the cpu version with reuse, it will segfault
            if (reuse) then
               temp_mat = P
               call MatConvert(P, MATSAME, MAT_INITIAL_MATRIX, temp_mat_compare, ierr)  
            else
               temp_mat_compare = P                         
            end if

            ! Debug check if the CPU and Kokkos versions are the same
            call compute_P_from_W_cpu(W, global_row_start, is_fine, is_coarse, &
                     identity, reuse, temp_mat)   

            call MatConvert(temp_mat, MATSAME, MAT_INITIAL_MATRIX, &
                        temp_mat_reuse, ierr)                       

            call MatAXPY(temp_mat_reuse, -1d0, temp_mat_compare, DIFFERENT_NONZERO_PATTERN, ierr)
            call MatNorm(temp_mat_reuse, NORM_FROBENIUS, normy, ierr)
            if (normy .gt. 1d-13) then
               !call MatFilter(temp_mat_reuse, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
               !call MatView(temp_mat_reuse, PETSC_VIEWER_STDOUT_WORLD, ierr)
               print *, "Kokkos and CPU versions of compute_P_from_W do not match"
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)  
            end if
            call MatDestroy(temp_mat_reuse, ierr)
            if (.NOT. reuse) then
               call MatDestroy(P, ierr)
            else
               call MatDestroy(temp_mat_compare, ierr)
            end if
            P = temp_mat

         end if

      else

         call compute_P_from_W_cpu(W, global_row_start, is_fine, is_coarse, &
                  identity, reuse, P)         

      end if
#else
      call compute_P_from_W_cpu(W, global_row_start, is_fine, is_coarse, &
                  identity, reuse, P)
#endif  

         
   end subroutine compute_P_from_W         

  !------------------------------------------------------------------------------------------------------------------------
   
   subroutine compute_P_from_W_cpu(W, global_row_start, is_fine, is_coarse, identity, reuse, P)

      ! Pass in W and get out P = [W I]' (or [W 0] if identity is false)
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(inout) :: W, P
      type(tIS), intent(in)     :: is_fine, is_coarse
      PetscInt, intent(in)      :: global_row_start
      logical, intent(in) :: identity, reuse

      PetscInt :: global_row_start_W, global_row_end_plus_one_W
      PetscInt :: global_col_start_W, global_col_end_plus_one_W
      PetscInt :: local_rows_coarse, local_rows, local_cols, local_cols_coarse
      PetscInt :: max_nnzs, i_loc, ncols, max_nnzs_total
      PetscInt :: global_cols, global_rows, global_rows_coarse, global_cols_coarse
      PetscInt :: cols_z, rows_z, local_rows_fine, counter
      integer :: errorcode, comm_size, comm_size_world
      PetscErrorCode :: ierr
      MPI_Comm :: MPI_COMM_MATRIX
      PetscInt, dimension(:), allocatable :: cols
      PetscReal, dimension(:), allocatable :: vals
      PetscInt, allocatable, dimension(:) :: row_indices, col_indices
      PetscReal, allocatable, dimension(:) :: v      
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      PetscInt, dimension(:), pointer :: is_pointer_coarse, is_pointer_fine
      MatType:: mat_type

      ! ~~~~~~~~~~

      call PetscObjectGetComm(W, MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)
      call MPI_Comm_size(MPI_COMM_WORLD, comm_size_world, errorcode)      

      call MatGetOwnershipRange(W, global_row_start_W, global_row_end_plus_one_W, ierr)  
      call MatGetOwnershipRangeColumn(W, global_col_start_W, global_col_end_plus_one_W, ierr)                  

      call MatGetSize(W, cols_z, rows_z, ierr) 

      call ISGetIndicesF90(is_fine, is_pointer_fine, ierr)
      call ISGetIndicesF90(is_coarse, is_pointer_coarse, ierr)      

      call IsGetLocalSize(is_coarse, local_rows_coarse, ierr)
      call IsGetLocalSize(is_fine, local_rows_fine, ierr)

      local_cols_coarse = local_rows_coarse
      local_cols = local_rows_coarse + local_rows_fine
      local_rows = local_cols 
      
      global_cols = rows_z + cols_z
      global_rows = global_cols
      global_rows_coarse = rows_z
      global_cols_coarse = rows_z      

      ! Get the max number of nnzs
      max_nnzs_total = 0
      max_nnzs = -1
      do i_loc = global_row_start_W, global_row_end_plus_one_W-1
         call MatGetRow(W, i_loc, &
                  ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
         max_nnzs_total = max_nnzs_total + ncols
         if (ncols > max_nnzs) max_nnzs = ncols                  
         call MatRestoreRow(W, i_loc, &
                  ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)          
      end do
      if (identity) max_nnzs_total = max_nnzs_total + local_rows_coarse
      max_nnzs = max_nnzs + 1

      allocate(cols(max_nnzs))
      allocate(vals(max_nnzs))

      allocate(row_indices(max_nnzs_total))
      allocate(col_indices(max_nnzs_total))
      allocate(v(max_nnzs_total))

      ! We may be reusing with the same sparsity
      if (.NOT. reuse) then
      
         call MatCreate(MPI_COMM_MATRIX, P, ierr)
         call MatSetSizes(P, local_rows, local_cols_coarse, &
                          global_rows, global_cols_coarse, ierr)
         ! Match the output type
         call MatGetType(W, mat_type, ierr)
         call MatSetType(P, mat_type, ierr)
         call MatSetUp(P, ierr)         
      end if

      ! Just in case there are some zeros in the input mat, ignore them
      call MatSetOption(P, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE, ierr)     
      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(P, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)      
      call MatSetOption(P, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)           

      ! Copy in the values of W
      counter = 1
      do i_loc = global_row_start_W, global_row_end_plus_one_W-1
         call MatGetRow(W, i_loc, &
                  ncols, cols, vals, ierr)

         row_indices(counter:counter+ncols-1) = is_pointer_fine(i_loc - global_row_start_W + 1)
         col_indices(counter:counter+ncols-1) = cols(1:ncols)
         v(counter:counter+ncols-1) = vals(1:ncols)    
         counter = counter + ncols                                

         call MatRestoreRow(W, i_loc, &
                  ncols, cols, vals, ierr)          
      end do

      ! If we want the identity block or just leave it zero
      if (identity) then
         do i_loc = 1, local_rows_coarse

            row_indices(counter) = is_pointer_coarse(i_loc)
            col_indices(counter) = i_loc - 1 + global_col_start_W
            v(counter) = 1d0
            counter = counter + 1

         end do     
      end if
      deallocate(cols, vals) 
      
      ! Set the values
      if (.NOT. reuse) then
         call MatSetPreallocationCOO(P, counter-1, row_indices, col_indices, ierr)
      end if
      deallocate(row_indices, col_indices)
      call MatSetValuesCOO(P, v, INSERT_VALUES, ierr)    
      deallocate(v)  
      
      call ISRestoreIndicesF90(is_coarse, is_pointer_coarse, ierr)
      call ISRestoreIndicesF90(is_fine, is_pointer_fine, ierr)       
         
   end subroutine compute_P_from_W_cpu      

!------------------------------------------------------------------------------------------------------------------------
   
   subroutine compute_R_from_Z(Z, global_row_start, is_fine, is_coarse, &
                     orig_fine_col_indices, &
                     identity, reuse, &
                     R)

      ! Wrapper around compute_R_from_Z_kokkos and compute_R_from_Z_cpu
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(inout) :: Z, R
      PetscInt, intent(in)      :: global_row_start
      type(tIS), intent(in)     :: is_fine, is_coarse
      type(tIS), intent(inout)  :: orig_fine_col_indices
      logical, intent(in) :: identity, reuse

#if defined(PETSC_HAVE_KOKKOS)                     
      integer(c_long_long) :: A_array, B_array, indices_fine, indices_coarse, orig_indices
      integer :: identity_int, reuse_int, reuse_indices_int, errorcode
      PetscErrorCode :: ierr
      MatType :: mat_type
      Mat :: temp_mat, temp_mat_reuse, temp_mat_compare
      PetscScalar normy;
#endif        
      ! ~~~~~~~~~~


#if defined(PETSC_HAVE_KOKKOS)    

      call MatGetType(Z, mat_type, ierr)
      if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) then

         identity_int = 0
         if (identity) identity_int = 1
         reuse_int = 0
         if (reuse) reuse_int = 1
         reuse_indices_int = 0;
         if (.NOT. PetscISIsNull(orig_fine_col_indices)) reuse_indices_int = 1

         A_array = Z%v             
         indices_fine = is_fine%v
         indices_coarse = is_coarse%v
         orig_indices = orig_fine_col_indices%v
         if (reuse) B_array = R%v
         call compute_R_from_Z_kokkos(A_array, global_row_start, indices_fine, indices_coarse, &
                        orig_indices, &
                        identity_int, reuse_int, reuse_indices_int, &
                        B_array)
         R%v = B_array
         orig_fine_col_indices%v = orig_indices
         
         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then

            ! If we're doing reuse and debug, then we have to always output the result 
            ! from the cpu version, as it will have coo preallocation structures set
            ! They aren't copied over if you do a matcopy (or matconvert)
            ! If we didn't do that the next time we come through this routine 
            ! and try to call the cpu version with reuse, it will segfault
            if (reuse) then
               temp_mat = R
               call MatConvert(R, MATSAME, MAT_INITIAL_MATRIX, temp_mat_compare, ierr)  
            else
               temp_mat_compare = R                       
            end if

            ! Debug check if the CPU and Kokkos versions are the same
            call compute_R_from_Z_cpu(Z, global_row_start, is_fine, is_coarse, &
                           orig_fine_col_indices, &
                           identity, reuse, &
                           temp_mat)  

            call MatConvert(temp_mat, MATSAME, MAT_INITIAL_MATRIX, &
                        temp_mat_reuse, ierr)                       

            call MatAXPY(temp_mat_reuse, -1d0, temp_mat_compare, DIFFERENT_NONZERO_PATTERN, ierr)
            call MatNorm(temp_mat_reuse, NORM_FROBENIUS, normy, ierr)
            if (normy .gt. 1d-13) then
               !call MatFilter(temp_mat_reuse, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
               !call MatView(temp_mat_reuse, PETSC_VIEWER_STDOUT_WORLD, ierr)
               print *, "Kokkos and CPU versions of compute_R_from_Z do not match"
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)  
            end if
            call MatDestroy(temp_mat_reuse, ierr)
            if (.NOT. reuse) then
               call MatDestroy(R, ierr)
            else
               call MatDestroy(temp_mat_compare, ierr)
            end if
            R = temp_mat

         end if

      else

         call compute_R_from_Z_cpu(Z, global_row_start, is_fine, is_coarse, &
                        orig_fine_col_indices, &
                        identity, reuse, &
                        R)       

      end if
#else
      call compute_R_from_Z_cpu(Z, global_row_start, is_fine, is_coarse, &
                     orig_fine_col_indices, &
                     identity, reuse, &
                     R)
#endif 
         
         
   end subroutine compute_R_from_Z   


 !------------------------------------------------------------------------------------------------------------------------
   
   subroutine compute_R_from_Z_cpu(Z, global_row_start, is_fine, is_coarse, &
                     orig_fine_col_indices, &
                     identity, reuse, &
                     R)

      ! Pass in Z and get out R = [Z I] (or [Z 0] if identity is false)
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(inout) :: Z, R
      PetscInt, intent(in)      :: global_row_start
      type(tIS), intent(in)     :: is_fine, is_coarse
      type(tIS), intent(inout)  :: orig_fine_col_indices
      logical, intent(in) :: identity, reuse

      PetscInt :: global_row_start_Z, global_row_end_plus_one_Z
      PetscInt :: global_col_start_Z, global_col_end_plus_one_Z
      PetscInt :: local_coarse_size, local_fine_size, local_full_cols
      PetscInt :: i_loc, ncols, max_nnzs_total, max_nnzs
      PetscInt :: global_coarse_size, global_fine_size, global_full_cols
      PetscInt :: rows_ao, cols_ao, rows_ad, cols_ad, size_cols
      PetscInt :: global_rows_z, global_cols_z
      PetscInt :: local_rows_z, local_cols_z, counter
      integer :: comm_size, errorcode
      PetscErrorCode :: ierr
      MPI_Comm :: MPI_COMM_MATRIX      
      PetscInt, dimension(:), allocatable :: cols
      PetscReal, dimension(:), allocatable :: vals
      PetscInt, allocatable, dimension(:) :: row_indices_coo, col_indices_coo
      PetscReal, allocatable, dimension(:) :: v      
      PetscOffset :: iicol
      PetscInt :: icol(1)       
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      type(tMat) :: Ad, Ao, temp_mat
      type(c_ptr) :: colmap_c_ptr
      PetscInt, pointer :: colmap_c(:)
      PetscInt, dimension(:), pointer :: col_indices_off_proc_array
      type(tIS) :: col_indices
      PetscInt, dimension(:), pointer :: is_pointer_orig_fine_col, is_pointer_coarse, is_pointer_fine
      integer(c_long_long) :: A_array
      MatType:: mat_type

      ! ~~~~~~~~~~

      call PetscObjectGetComm(Z, MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)
      
      call ISGetIndicesF90(is_fine, is_pointer_fine, ierr)
      call ISGetIndicesF90(is_coarse, is_pointer_coarse, ierr)      

      call ISGetLocalSize(is_coarse, local_coarse_size, ierr)
      call ISGetLocalSize(is_fine, local_fine_size, ierr)
      call ISGetSize(is_coarse, global_coarse_size, ierr)
      call ISGetSize(is_fine, global_fine_size, ierr)      

      local_full_cols = local_coarse_size + local_fine_size
      global_full_cols = global_coarse_size + global_fine_size

      call MatGetLocalSize(Z, local_rows_z, local_cols_z, ierr) 
      call MatGetSize(Z, global_rows_z, global_cols_z, ierr) 
      
      call MatGetOwnershipRange(Z, global_row_start_Z, global_row_end_plus_one_Z, ierr)  
      call MatGetOwnershipRangeColumn(Z, global_col_start_Z, global_col_end_plus_one_Z, ierr)   

      call MatGetType(Z, mat_type, ierr)

      ! Get the local non-local components and sizes
      if (comm_size /= 1) then

         if (mat_type == "mpiaij") then
            call MatMPIAIJGetSeqAIJ(Z, Ad, Ao, icol, iicol, ierr)
            A_array = Z%v

         ! If on the gpu, just do a convert to mpiaij format first
         ! This will be expensive but the best we can do for now without writing our 
         ! own version of this subroutine in cuda/kokkos            
         else
            call MatConvert(Z, MATMPIAIJ, MAT_INITIAL_MATRIX, temp_mat, ierr)
            call MatMPIAIJGetSeqAIJ(temp_mat, Ad, Ao, icol, iicol, ierr)             
            A_array = temp_mat%v
         end if

         ! We know the col size of Ao is the size of colmap, the number of non-zero offprocessor columns
         call MatGetSize(Ao, rows_ao, cols_ao, ierr)    
         call MatGetSize(Ad, rows_ad, cols_ad, ierr)  
         
      else
         call MatGetSize(Z, rows_ad, cols_ad, ierr)    
         Ad = Z         
      end if

      ! We can reuse the orig_fine_col_indices as they can be expensive to generate in parallel
      if (PetscISIsNull(orig_fine_col_indices)) then
            
         ! Now we need the global off-processor column indices in Z
         if (comm_size /= 1) then 

            call get_colmap_c(A_array, colmap_c_ptr)
            call c_f_pointer(colmap_c_ptr, colmap_c, shape=[cols_ao])
            
            ! These are the global indices of the columns we want
            allocate(col_indices_off_proc_array(cols_ad + cols_ao))
            size_cols = cols_ad + cols_ao
            ! Local rows (as global indices)
            do i_loc = 1, cols_ad
               col_indices_off_proc_array(i_loc) = global_col_start_Z + i_loc - 1
            end do
            ! Off diagonal rows we want (as global indices)
            do i_loc = 1, cols_ao
               col_indices_off_proc_array(cols_ad + i_loc) = colmap_c(i_loc)
            end do            
            ! These indices are not sorted, we deliberately have the local ones first then the 
            ! off processor ones, as we insert Z into the full matrix in pieces
            ! with Ad, then Ao, so that way we can index into orig_fine_col_indices in that order

         else

            ! These are the global indices of the columns we want
            allocate(col_indices_off_proc_array(cols_ad))
            size_cols = cols_ad
            ! Local rows (as global indices)
            do i_loc = 1, cols_ad
               col_indices_off_proc_array(i_loc) = global_col_start_Z + i_loc - 1
            end do
         end if

         ! Create the IS we want with the cols we want (written as global indices)
         call ISCreateGeneral(MPI_COMM_MATRIX, size_cols, col_indices_off_proc_array, PETSC_USE_POINTER, col_indices, ierr)

         ! Now let's do the comms to get what the original column indices in the full matrix are, given these indices for all 
         ! the columns of Z - ie we need to check in the original fine indices at the positions given by col_indices_off_proc_array
         ! This could be expensive as the number of off-processor columns in Z grows!
         call ISCreateSubIS(is_fine, col_indices, orig_fine_col_indices, ierr)

         ! We've now built the original fine indices
         call ISDestroy(col_indices, ierr)
         deallocate(col_indices_off_proc_array)  

      end if

      ! Get the indices
      call ISGetIndicesF90(orig_fine_col_indices, is_pointer_orig_fine_col, ierr)

      ! Z
      ! Get the max number of nnzs
      max_nnzs_total = 0
      max_nnzs = -1      
      do i_loc = global_row_start_Z, global_row_end_plus_one_Z-1
         call MatGetRow(Z, i_loc, &
                  ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
         max_nnzs_total = max_nnzs_total + ncols
         if (ncols > max_nnzs) max_nnzs = ncols
         call MatRestoreRow(Z, i_loc, &
                  ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)          
      end do
      if (identity) max_nnzs_total = max_nnzs_total + local_rows_z
      max_nnzs = max_nnzs + 1

      allocate(cols(max_nnzs))
      allocate(vals(max_nnzs))

      allocate(row_indices_coo(max_nnzs_total))
      allocate(col_indices_coo(max_nnzs_total))
      allocate(v(max_nnzs_total))    
      
      ! We may be reusing with the same sparsity
      if (.NOT. reuse) then
      
         call MatCreate(MPI_COMM_MATRIX, R, ierr)
         ! Taking care here to use the row sizes of Z, but the full size for cols
         ! as we can use this routine to stick Afc and Aff in full sizes column arrays
         ! and hence the row size of the input Z won't always equal the same sizes as the input is's
         call MatSetSizes(R, local_rows_z, local_full_cols, &
                             global_rows_z, global_full_cols, ierr)
         ! Match the output type
         call MatSetType(R, mat_type, ierr)
         call MatSetUp(R, ierr)         
      end if      

      ! Just in case there are some zeros in the input mat, ignore them
      call MatSetOption(R, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE, ierr)     
      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(R, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)      
      call MatSetOption(R, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr) 

      ! Z - do Ad and Ao separately, as that way we have local indices into is_pointer_orig_fine_col
      ! to give us the original column numbers 
      ! Let's start with Ad - remember Ad and Ao are serial
      counter = 1
      do i_loc = 1, local_rows_z
         call MatGetRow(Ad, i_loc-1, &
                  ncols, cols, vals, ierr)

         row_indices_coo(counter:counter+ncols-1) = i_loc -1 + global_row_start_Z
         col_indices_coo(counter:counter+ncols-1) = is_pointer_orig_fine_col(cols(1:ncols)+1)
         v(counter:counter+ncols-1) = vals(1:ncols)    
         counter = counter + ncols                                

         call MatRestoreRow(Ad, i_loc-1, &
                  ncols, cols, vals, ierr)         
      end do

      if (comm_size /= 1) then
         ! Then Ao
         do i_loc = 1, local_rows_z
            call MatGetRow(Ao, i_loc-1, &
                     ncols, cols, vals, ierr)

            row_indices_coo(counter:counter+ncols-1) = i_loc -1 + global_row_start_Z
            col_indices_coo(counter:counter+ncols-1) = is_pointer_orig_fine_col(cols(1:ncols)+1 + cols_ad)
            v(counter:counter+ncols-1) = vals(1:ncols)    
            counter = counter + ncols                       

            call MatRestoreRow(Ao, i_loc-1, &
                     ncols, cols, vals, ierr)          
         end do
      end if

      ! If we want the identity block or just leave it zero
      if (identity) then
         do i_loc = 1, local_rows_z

            row_indices_coo(counter) = i_loc - 1 + global_row_start_Z
            col_indices_coo(counter) = is_pointer_coarse(i_loc)
            v(counter) = 1d0
            counter = counter + 1

         end do  
      end if                                   
      deallocate(cols, vals)

      ! Set the values
      if (.NOT. reuse) then
         call MatSetPreallocationCOO(R, counter-1, row_indices_coo, col_indices_coo, ierr)
      end if
      deallocate(row_indices_coo, col_indices_coo)
      call MatSetValuesCOO(R, v, INSERT_VALUES, ierr)    
      deallocate(v)   

      if (comm_size /= 1 .AND. mat_type /= "mpiaij") then
         call MatDestroy(temp_mat, ierr)
      end if      

      call ISRestoreIndicesF90(orig_fine_col_indices, is_pointer_orig_fine_col, ierr)
      call ISRestoreIndicesF90(is_coarse, is_pointer_coarse, ierr)
      call ISRestoreIndicesF90(is_fine, is_pointer_fine, ierr)       
         
   end subroutine compute_R_from_Z_cpu
   
   !------------------------------------------------------------------------------------------------------------------------
   
   subroutine get_nnzs_petsc_sparse(input_mat, nnzs)
      ! Returns nnzs in a sparse petsc matrix
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), intent(in) :: input_mat
      integer(kind=8), intent(out) :: nnzs

      PetscInt :: ncols, i_loc
      PetscInt :: global_row_start, global_row_end_plus_one
      integer :: comm_size, errorcode
      PetscErrorCode :: ierr
      MPI_Comm :: MPI_COMM_MATRIX
      integer(kind=8) :: local_nnzs
      
      ! ~~~~~~~~~~

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)  
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)

      ! This returns the global index of the local portion of the matrix
      call MatGetOwnershipRange(input_mat, global_row_start, global_row_end_plus_one, ierr)  
      
      local_nnzs = 0

      ! This will be the nnzs associated with the local rows
      do i_loc = global_row_start, global_row_end_plus_one-1                  
         call MatGetRow(input_mat, i_loc, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
         local_nnzs = local_nnzs + ncols
         call MatRestoreRow(input_mat, i_loc, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
      end do          

      ! Do an accumulate if in parallel
      if (comm_size/=1) then
         call MPI_Allreduce(local_nnzs, nnzs, 1, MPI_INTEGER8, MPI_SUM, MPI_COMM_MATRIX, errorcode)
      else
         nnzs = local_nnzs
      end if    
         
   end subroutine get_nnzs_petsc_sparse

   !-------------------------------------------------------------------------------------------------------------------------------

   subroutine svd(input, U, sigma, VT)

      ! ~~~~~~~~~~~~~~~~

      PetscReal, dimension(:, :), intent(in) :: input
      PetscReal, dimension(size(input, 1), size(input, 1)), intent(out) :: U
      PetscReal, dimension(min(size(input, 1), size(input, 2))), intent(out) :: sigma
      PetscReal, dimension(size(input, 2), size(input, 2)), intent(out) :: VT
  
      PetscReal, dimension(size(input, 1), size(input, 2)) :: tmp_input
      PetscReal, dimension(:), allocatable :: WORK
      integer :: LWORK, M, N, info, errorcode

      ! ~~~~~~~~~~~~~~~~
  
     tmp_input = input
     M = size(input, 1)
     N = size(input, 2)
     LWORK = 2*MAX(1,3*MIN(M,N)+MAX(M,N),5*MIN(M,N))
     allocate(WORK(LWORK))
  
     call DGESVD('A', 'A', M, N, tmp_input, M, &
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

      PetscReal, dimension(:, :), intent(in) :: input
      PetscReal, dimension(min(size(input, 1), size(input, 2))), intent(out) :: output

      PetscReal, dimension(size(input, 1), size(input, 1)) :: U
      PetscReal, dimension(min(size(input, 1), size(input, 2))) :: sigma
      PetscReal, dimension(size(input, 2), size(input, 2)) :: VT

      integer :: iloc, errorcode

      ! ~~~~~~~~~~~~~~~~

      ! Compute the svd
      call svd(input, U, sigma, VT)

      ! Now the pseudoinverse is V * inv(sigma) * U^T
      ! and sigma is diagonal 
      ! So scale each column of U (given the transpose)
      do iloc = 1, size(input,1)
         if (abs(sigma(iloc)) > 1e-13) then
            U(:, iloc) = U(:, iloc) * 1d0/sigma(iloc)
         else
            U(:, iloc) = 0d0
         end if
      end do

      ! Do the matmatmult, making sure to transpose both
      call dgemm("T", "T", size(input,1), size(input,1), size(input,1), &
               1d0, VT, size(input,1), &
               U, size(input,1), &
               0d0, output, size(input,1))          

      ! nan check
      if (any(output /= output)) then
         print *, "NaN in pseudo inverse"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)         
      end if      
  
    end subroutine pseudo_inv 
    
   !-------------------------------------------------------------------------------------------------------------------------------

    subroutine ShellSetVecType(input_mat, input_mat_shell)

      ! ~~~~~~~~~~~~~~~~
      type(tMat), intent(in) :: input_mat, input_mat_shell

      integer(c_long_long) :: A_array, B_array
      ! ~~~~~~~~~~~~~~~~

      A_array = input_mat%v
      B_array = input_mat_shell%v

      call ShellSetVecType_c(A_array, B_array) 
  
    end subroutine ShellSetVecType      

   !-------------------------------------------------------------------------------------------------------------------------------

end module petsc_helper

