module grid_transfer

   use petscmat
   use c_petsc_interfaces
   use petsc_helper

#include "petsc/finclude/petscmat.h"
#include "petscconf.h"
                
   implicit none

   public

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! Grid transfer routines
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------      

   contains 

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
      PetscScalar :: normy
      type(tVec) :: max_vec
      PetscInt :: row_loc      
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
            ! Find the biggest entry in the difference
            call MatCreateVecs(temp_mat, PETSC_NULL_VEC, max_vec, ierr)
            call MatGetRowMaxAbs(temp_mat, max_vec, PETSC_NULL_INTEGER_POINTER, ierr)
            call VecMax(max_vec, row_loc, normy, ierr)
            call VecDestroy(max_vec, ierr)
            
            if (normy .gt. 1d-13 .OR. normy/=normy) then
               !call MatFilter(temp_mat, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
               !call MatView(temp_mat, PETSC_VIEWER_STDOUT_WORLD, ierr)
               print *, "Diff Kokkos and CPU generate_one_point_with_one_entry_from_sparse", normy, "row", row_loc
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
      
      PetscInt :: ncols, ifree
      PetscInt :: local_rows, local_cols, global_rows, global_cols
      PetscInt :: global_row_start, global_row_end_plus_one
      PetscInt :: global_col_start, global_col_end_plus_one
      PetscInt :: max_local_col, max_nonlocal_col
      PetscCount :: counter
      PetscInt, allocatable, dimension(:) :: row_indices, col_indices
      PetscReal, allocatable, dimension(:) :: v
      PetscErrorCode :: ierr
      PetscInt, dimension(:), pointer :: cols => null()
      PetscReal, dimension(:), pointer :: vals => null()
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      integer :: max_loc(1)
      integer :: comm_size, errorcode
      MPIU_Comm :: MPI_COMM_MATRIX
      MatType:: mat_type
      PetscInt, dimension(:), pointer :: colmap
      type(tMat) :: Ad, Ao
      PetscReal :: max_local_val, max_nonlocal_val
      
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

      ! Get the local/nonlocal components
      if (comm_size /= 1) then
         call MatMPIAIJGetSeqAIJ(input_mat, Ad, Ao, colmap, ierr)
      else
         Ad = input_mat
      end if
      
      ! Now go and fill the new matrix
      ! Loop over global row indices
      counter = 1
      do ifree = global_row_start, global_row_end_plus_one-1             
         
         max_local_val = -huge(0d0)
         max_nonlocal_val = -huge(0d0)
         max_local_col = -1
         max_nonlocal_col = -1

         ! Let's get the biggest local component
         call MatGetRow(Ad, ifree - global_row_start, ncols, cols, vals, ierr) 
         if (ncols /= 0) then
            max_loc = maxloc(abs(vals(1:ncols)))
            max_local_val = abs(vals(max_loc(1)))
            ! Global column index
            max_local_col = cols(max_loc(1)) + global_col_start
         end if
         call MatRestoreRow(Ad, ifree - global_row_start, ncols, cols, vals, ierr)   

         ! And the biggest non local component
         if (comm_size /= 1) then
            call MatGetRow(Ao, ifree - global_row_start, ncols, cols, vals, ierr) 
            if (ncols /= 0) then
               max_loc = maxloc(abs(vals(1:ncols)))
               max_nonlocal_val = abs(vals(max_loc(1)))
               ! Global column index - need colmap
               max_nonlocal_col = colmap(cols(max_loc(1)) + 1)
            end if
            call MatRestoreRow(Ao, ifree - global_row_start, ncols, cols, vals, ierr)            
         end if

         ! If there is a bigger nonlocal entry we have to use that
         ! If the biggest entry is local or the biggest local or nonlocal entries are equal
         ! we use the local entry
         if (max_nonlocal_val > max_local_val) max_local_col = max_nonlocal_col

         if (max_local_col /= -1) then
            row_indices(counter) = ifree
            col_indices(counter) = max_local_col  
            counter = counter + 1
         end if     
      end do         
      
      ! Set the values
      call MatSetPreallocationCOO(output_mat, counter-1, row_indices, col_indices, ierr)
      deallocate(row_indices, col_indices)
      call MatSetValuesCOO(output_mat, v, INSERT_VALUES, ierr)    
      deallocate(v)      
         
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
      PetscScalar :: normy
      type(tVec) :: max_vec
      PetscInt :: row_loc       
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
            ! Find the biggest entry in the difference
            call MatCreateVecs(temp_mat_reuse, PETSC_NULL_VEC, max_vec, ierr)
            call MatGetRowMaxAbs(temp_mat_reuse, max_vec, PETSC_NULL_INTEGER_POINTER, ierr)
            call VecMax(max_vec, row_loc, normy, ierr)
            call VecDestroy(max_vec, ierr)

            if (normy .gt. 1d-13 .OR. normy/=normy) then
               !call MatFilter(temp_mat_reuse, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
               !call MatView(temp_mat_reuse, PETSC_VIEWER_STDOUT_WORLD, ierr)
               print *, "Diff Kokkos and CPU compute_P_from_W", normy, "row", row_loc
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
      PetscInt :: cols_z, rows_z, local_rows_fine
      PetscCount :: counter
      integer :: errorcode, comm_size, comm_size_world
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX
      PetscInt, dimension(:), pointer :: cols => null()
      PetscReal, dimension(:), pointer :: vals => null()
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

      call ISGetIndices(is_fine, is_pointer_fine, ierr)
      call ISGetIndices(is_coarse, is_pointer_coarse, ierr)      

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
                  ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
         max_nnzs_total = max_nnzs_total + ncols
         if (ncols > max_nnzs) max_nnzs = ncols                  
         call MatRestoreRow(W, i_loc, &
                  ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)          
      end do
      if (identity) max_nnzs_total = max_nnzs_total + local_rows_coarse
      max_nnzs = max_nnzs + 1

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
      
      ! Set the values
      if (.NOT. reuse) then
         call MatSetPreallocationCOO(P, counter-1, row_indices, col_indices, ierr)
      end if
      deallocate(row_indices, col_indices)
      call MatSetValuesCOO(P, v, INSERT_VALUES, ierr)    
      deallocate(v)  
      
      call ISRestoreIndices(is_coarse, is_pointer_coarse, ierr)
      call ISRestoreIndices(is_fine, is_pointer_fine, ierr)       
         
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
      PetscScalar :: normy
      type(tVec) :: max_vec
      PetscInt :: row_loc      
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
         if (.NOT. PetscObjectIsNull(orig_fine_col_indices)) then
            reuse_indices_int = 1
         end if

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
            ! Find the biggest entry in the difference
            call MatCreateVecs(temp_mat_reuse, PETSC_NULL_VEC, max_vec, ierr)
            call MatGetRowMaxAbs(temp_mat_reuse, max_vec, PETSC_NULL_INTEGER_POINTER, ierr)
            call VecMax(max_vec, row_loc, normy, ierr)
            call VecDestroy(max_vec, ierr)

            if (normy .gt. 1d-13 .OR. normy/=normy) then
               !call MatFilter(temp_mat_reuse, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
               !call MatView(temp_mat_reuse, PETSC_VIEWER_STDOUT_WORLD, ierr)
               print *, "Diff Kokkos and CPU compute_R_from_Z", normy, "row", row_loc
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
      PetscInt :: local_rows_z, local_cols_z
      PetscCount :: counter
      integer :: comm_size, errorcode
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX      
      PetscInt, dimension(:), pointer :: cols => null()
      PetscReal, dimension(:), pointer :: vals => null()
      PetscInt, allocatable, dimension(:) :: row_indices_coo, col_indices_coo
      PetscReal, allocatable, dimension(:) :: v      
      PetscInt, dimension(:), pointer :: colmap
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      type(tMat) :: Ad, Ao
      PetscInt, dimension(:), pointer :: col_indices_off_proc_array
      type(tIS) :: col_indices
      PetscInt, dimension(:), pointer :: is_pointer_orig_fine_col, is_pointer_coarse, is_pointer_fine
      integer(c_long_long) :: A_array
      MatType:: mat_type

      ! ~~~~~~~~~~

      call PetscObjectGetComm(Z, MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)
      
      call ISGetIndices(is_fine, is_pointer_fine, ierr)
      call ISGetIndices(is_coarse, is_pointer_coarse, ierr)      

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

         call MatMPIAIJGetSeqAIJ(Z, Ad, Ao, colmap, ierr)
         A_array = Z%v

         ! We know the col size of Ao is the size of colmap, the number of non-zero offprocessor columns
         call MatGetSize(Ao, rows_ao, cols_ao, ierr)    
         call MatGetSize(Ad, rows_ad, cols_ad, ierr)  
         
      else
         call MatGetSize(Z, rows_ad, cols_ad, ierr)    
         Ad = Z         
      end if

      ! We can reuse the orig_fine_col_indices as they can be expensive to generate in parallel
      if (PetscObjectIsNull(orig_fine_col_indices)) then
            
         ! Now we need the global off-processor column indices in Z
         if (comm_size /= 1) then 
            
            ! These are the global indices of the columns we want
            allocate(col_indices_off_proc_array(cols_ad + cols_ao))
            size_cols = cols_ad + cols_ao
            ! Local rows (as global indices)
            do i_loc = 1, cols_ad
               col_indices_off_proc_array(i_loc) = global_col_start_Z + i_loc - 1
            end do
            ! Off diagonal rows we want (as global indices)
            do i_loc = 1, cols_ao
               col_indices_off_proc_array(cols_ad + i_loc) = colmap(i_loc)
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
      call ISGetIndices(orig_fine_col_indices, is_pointer_orig_fine_col, ierr)

      ! Z
      ! Get the max number of nnzs
      max_nnzs_total = 0
      max_nnzs = -1      
      do i_loc = global_row_start_Z, global_row_end_plus_one_Z-1
         call MatGetRow(Z, i_loc, &
                  ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
         max_nnzs_total = max_nnzs_total + ncols
         if (ncols > max_nnzs) max_nnzs = ncols
         call MatRestoreRow(Z, i_loc, &
                  ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)          
      end do
      if (identity) max_nnzs_total = max_nnzs_total + local_rows_z
      max_nnzs = max_nnzs + 1

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

      ! Set the values
      if (.NOT. reuse) then
         call MatSetPreallocationCOO(R, counter-1, row_indices_coo, col_indices_coo, ierr)
      end if
      deallocate(row_indices_coo, col_indices_coo)
      call MatSetValuesCOO(R, v, INSERT_VALUES, ierr)    
      deallocate(v)     

      call ISRestoreIndices(orig_fine_col_indices, is_pointer_orig_fine_col, ierr)
      call ISRestoreIndices(is_coarse, is_pointer_coarse, ierr)
      call ISRestoreIndices(is_fine, is_pointer_fine, ierr)       
         
   end subroutine compute_R_from_Z_cpu

   !-------------------------------------------------------------------------------------------------------------------------------

end module grid_transfer

