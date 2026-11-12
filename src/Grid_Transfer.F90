module grid_transfer

   use petscmat
   use c_petsc_interfaces, only: generate_one_point_with_one_entry_from_sparse_kokkos, &
         compute_P_from_W_kokkos, compute_R_from_Z_kokkos, generate_one_point_P_from_sparse_kokkos
   use petsc_helper, only: kokkos_debug, assemble_one_point_shell
   use pflare_parameters, only: PFLARE_TOL_MATFREE_12, PFLARE_TOL_MATFREE_13, PFLARE_MINUS_ONE, PFLARE_ONE, &
         PFLARE_TWO

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

            call MatAXPY(temp_mat, PFLARE_MINUS_ONE, output_mat, DIFFERENT_NONZERO_PATTERN, ierr)
            ! Find the biggest entry in the difference
            call MatCreateVecs(temp_mat, PETSC_NULL_VEC, max_vec, ierr)
            call MatGetRowMaxAbs(temp_mat, max_vec, PETSC_NULL_INTEGER_POINTER, ierr)
            call VecMax(max_vec, row_loc, normy, ierr)
            call VecDestroy(max_vec, ierr)
            
            if (normy .gt. PFLARE_TOL_MATFREE_13 .OR. normy/=normy) then
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
      
      PetscInt :: ncols, ifree, i_loc
      PetscInt :: local_rows, local_cols, global_rows, global_cols
      PetscInt :: global_row_start, global_row_end_plus_one
      PetscInt :: global_col_start, global_col_end_plus_one
      PetscInt :: max_local_col, max_nonlocal_col
      ! The column of the biggest entry in each local row (-1 if the row is empty)
      PetscInt, allocatable, dimension(:) :: biggest_cols, d_nnz, o_nnz
      PetscErrorCode :: ierr
      PetscInt, dimension(:), pointer :: cols => null()
      ! Matrix entries filled by MatGetRow are PetscScalar
      PetscScalar, dimension(:), pointer :: vals => null()
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      integer :: max_loc(1)
      integer :: comm_size, errorcode
      MPIU_Comm :: MPI_COMM_MATRIX
      MatType:: mat_type
      PetscInt, dimension(:), pointer :: colmap
      type(tMat) :: Ad, Ao
      ! Derived magnitudes stay PetscReal
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

      ! We know we only have one entry per row
      allocate(biggest_cols(local_rows))
      biggest_cols = -1

      ! Get the local/nonlocal components
      if (comm_size /= 1) then
         call MatMPIAIJGetSeqAIJ(input_mat, Ad, Ao, colmap, ierr)
      else
         Ad = input_mat
      end if
      
      ! Find the biggest entry in each row
      ! Loop over global row indices
      do ifree = global_row_start, global_row_end_plus_one-1             
         
         ! huge() of the PetscReal target (not 0d0) so single builds get the
         ! representable -huge, not -Inf (which traps under -ffpe-trap=overflow)
         max_local_val = -huge(max_local_val)
         max_nonlocal_val = -huge(max_nonlocal_val)
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

         biggest_cols(ifree - global_row_start + 1) = max_local_col
      end do         

      ! Create the output matrix, exactly preallocated
      allocate(d_nnz(local_rows), o_nnz(local_rows))
      d_nnz = 0
      o_nnz = 0
      do i_loc = 1, local_rows
         if (biggest_cols(i_loc) == -1) cycle
         if (biggest_cols(i_loc) >= global_col_start .AND. biggest_cols(i_loc) < global_col_end_plus_one) then
            d_nnz(i_loc) = 1
         else
            o_nnz(i_loc) = 1
         end if
      end do

      call MatCreate(MPI_COMM_MATRIX, output_mat, ierr)
      call MatSetSizes(output_mat, local_rows, local_cols, &
                       global_rows, global_cols, ierr)
      ! Match the output type
      call MatGetType(input_mat, mat_type, ierr)
      call MatSetType(output_mat, mat_type, ierr)
      call MatXAIJSetPreallocation(output_mat, one, d_nnz, o_nnz, &
               PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_INTEGER_ARRAY, ierr)
      deallocate(d_nnz, o_nnz)
      
      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)        
      call MatSetOption(output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)    

      do i_loc = 1, local_rows
         if (biggest_cols(i_loc) == -1) cycle
         call MatSetValue(output_mat, global_row_start + i_loc - 1, biggest_cols(i_loc), &
                  PFLARE_ONE, INSERT_VALUES, ierr)
      end do
      deallocate(biggest_cols)
      call MatAssemblyBegin(output_mat, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(output_mat, MAT_FINAL_ASSEMBLY, ierr)
         
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

            ! If we're doing reuse and debug, then the cpu version reuses the output
            ! matrix itself (as that has the sparsity the cpu version expects) and so
            ! we compare against a copy of the kokkos result
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

            call MatAXPY(temp_mat_reuse, PFLARE_MINUS_ONE, temp_mat_compare, DIFFERENT_NONZERO_PATTERN, ierr)
            ! Find the biggest entry in the difference
            call MatCreateVecs(temp_mat_reuse, PETSC_NULL_VEC, max_vec, ierr)
            call MatGetRowMaxAbs(temp_mat_reuse, max_vec, PETSC_NULL_INTEGER_POINTER, ierr)
            call VecMax(max_vec, row_loc, normy, ierr)
            call VecDestroy(max_vec, ierr)

            if (normy .gt. PFLARE_TOL_MATFREE_13 .OR. normy/=normy) then
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

   subroutine generate_one_point_P_from_sparse(A_fc, global_row_start, is_fine, is_coarse, P)

      ! Builds the one point classical prolongator P = [W I]' straight from A_fc, where W has
      ! a one in the column of the biggest entry in each row of A_fc
      ! The same as generate_one_point_with_one_entry_from_sparse followed by compute_P_from_W
      ! with the identity, but with Kokkos it doesn't build W, and P is matrix-free
      ! (a MATSHELL, see assemble_one_point_shell for the places that need it assembled)
      ! Always creates a new P (no reuse)

      ! ~~~~~~~~~~
      ! Input
      type(tMat), intent(in)    :: A_fc
      type(tMat), intent(inout) :: P
      type(tIS), intent(in)     :: is_fine, is_coarse
      PetscInt, intent(in)      :: global_row_start

      type(tMat) :: W
      PetscErrorCode :: ierr
#if defined(PETSC_HAVE_KOKKOS)
      integer(c_long_long) :: A_array, B_array, indices_fine, indices_coarse
      integer :: errorcode
      MatType :: mat_type
      type(tMat) :: temp_mat, P_assembled
      PetscScalar :: normy
      type(tVec) :: max_vec
      PetscInt :: row_loc
      integer(c_int) :: shell_int
      type(tVec) :: x_coarse, x_fine, y_coarse, y_fine, x_coarse_ref, x_fine_ref
      PetscRandom :: rctx
      PetscReal :: norm_fine, norm_coarse
#endif
      ! ~~~~~~~~~~

#if defined(PETSC_HAVE_KOKKOS)

      call MatGetType(A_fc, mat_type, ierr)
      if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) then

         A_array = A_fc%v
         indices_fine = is_fine%v
         indices_coarse = is_coarse%v
         shell_int = 1
         call generate_one_point_P_from_sparse_kokkos(A_array, global_row_start, &
                     indices_fine, indices_coarse, shell_int, B_array)
         P%v = B_array

         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then

            call generate_one_point_with_one_entry_from_sparse_cpu(A_fc, W)
            call compute_P_from_W_cpu(W, global_row_start, is_fine, is_coarse, &
                     .TRUE., .FALSE., temp_mat)
            call MatDestroy(W, ierr)

            call assemble_one_point_shell(P, P_assembled)
            call MatAXPY(temp_mat, PFLARE_MINUS_ONE, P_assembled, DIFFERENT_NONZERO_PATTERN, ierr)
            ! Find the biggest entry in the difference
            call MatCreateVecs(temp_mat, PETSC_NULL_VEC, max_vec, ierr)
            call MatGetRowMaxAbs(temp_mat, max_vec, PETSC_NULL_INTEGER_POINTER, ierr)
            call VecMax(max_vec, row_loc, normy, ierr)
            call VecDestroy(max_vec, ierr)

            if (normy .gt. PFLARE_TOL_MATFREE_13 .OR. normy/=normy) then
               print *, "Diff Kokkos and CPU generate_one_point_P_from_sparse", normy, "row", row_loc
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
            end if

            ! The matrix-free P applied to a random vector (forward, adding and transposed)
            ! should match the assembled P
            call PetscRandomCreate(MPI_COMM_WORLD, rctx, ierr)
            call MatCreateVecs(P_assembled, x_coarse, x_fine, ierr)
            call VecDuplicate(x_coarse, y_coarse, ierr)
            call VecDuplicate(x_fine, y_fine, ierr)
            call VecDuplicate(x_coarse, x_coarse_ref, ierr)
            call VecDuplicate(x_fine, x_fine_ref, ierr)
            call VecSetRandom(x_coarse, rctx, ierr)
            call VecSetRandom(x_fine, rctx, ierr)
            call MatMult(P, x_coarse, y_fine, ierr)
            call MatMultAdd(P, x_coarse, y_fine, y_fine, ierr)
            call MatMultTranspose(P, x_fine, y_coarse, ierr)
            call MatMultTransposeAdd(P, x_fine, y_coarse, y_coarse, ierr)
            ! y_fine should be 2 P x_coarse, y_coarse 2 P^T x_fine
            call VecScale(x_coarse, PFLARE_TWO, ierr)
            call VecScale(x_fine, PFLARE_TWO, ierr)
            call MatMult(P_assembled, x_coarse, x_fine_ref, ierr)
            call MatMultTranspose(P_assembled, x_fine, x_coarse_ref, ierr)
            call VecAXPY(y_fine, PFLARE_MINUS_ONE, x_fine_ref, ierr)
            call VecAXPY(y_coarse, PFLARE_MINUS_ONE, x_coarse_ref, ierr)
            call VecNorm(y_fine, NORM_INFINITY, norm_fine, ierr)
            call VecNorm(y_coarse, NORM_INFINITY, norm_coarse, ierr)
            if (norm_fine .gt. PFLARE_TOL_MATFREE_12 .OR. norm_fine/=norm_fine .OR. &
                  norm_coarse .gt. PFLARE_TOL_MATFREE_12 .OR. norm_coarse/=norm_coarse) then
               print *, "Diff matrix-free and assembled one point P", norm_fine, norm_coarse
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
            end if
            call VecDestroy(x_coarse, ierr)
            call VecDestroy(x_fine, ierr)
            call VecDestroy(y_coarse, ierr)
            call VecDestroy(y_fine, ierr)
            call VecDestroy(x_coarse_ref, ierr)
            call VecDestroy(x_fine_ref, ierr)
            call PetscRandomDestroy(rctx, ierr)
            call MatDestroy(P_assembled, ierr)
            call MatDestroy(temp_mat, ierr)
         end if

      else

         call generate_one_point_with_one_entry_from_sparse(A_fc, W)
         call compute_P_from_W(W, global_row_start, is_fine, is_coarse, .TRUE., .FALSE., P)
         call MatDestroy(W, ierr)

      end if
#else
      call generate_one_point_with_one_entry_from_sparse(A_fc, W)
      call compute_P_from_W(W, global_row_start, is_fine, is_coarse, .TRUE., .FALSE., P)
      call MatDestroy(W, ierr)
#endif

   end subroutine generate_one_point_P_from_sparse

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
      PetscInt :: i_loc, ncols, col, p_row
      PetscInt :: global_cols, global_rows, global_rows_coarse, global_cols_coarse
      PetscInt :: cols_z, rows_z, local_rows_fine
      PetscInt, dimension(1) :: row_array
      integer :: errorcode, comm_size, comm_size_world
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX
      PetscInt, dimension(:), pointer :: cols => null()
      ! Matrix entries filled by MatGetRow are PetscScalar
      PetscScalar, dimension(:), pointer :: vals => null()
      PetscInt, allocatable, dimension(:) :: d_nnz, o_nnz
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

      ! We may be reusing with the same sparsity
      if (.NOT. reuse) then
      
         call MatCreate(MPI_COMM_MATRIX, P, ierr)
         call MatSetSizes(P, local_rows, local_cols_coarse, &
                          global_rows, global_cols_coarse, ierr)
         ! Match the output type
         call MatGetType(W, mat_type, ierr)
         call MatSetType(P, mat_type, ierr)

         ! Count the entries in each row so we can exactly preallocate
         ! The columns of P have the same layout as the columns of W
         allocate(d_nnz(local_rows), o_nnz(local_rows))
         d_nnz = 0
         o_nnz = 0
         do i_loc = global_row_start_W, global_row_end_plus_one_W-1
            call MatGetRow(W, i_loc, ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)
            p_row = is_pointer_fine(i_loc - global_row_start_W + 1) - global_row_start + 1
            do col = 1, ncols
               if (cols(col) >= global_col_start_W .AND. cols(col) < global_col_end_plus_one_W) then
                  d_nnz(p_row) = d_nnz(p_row) + 1
               else
                  o_nnz(p_row) = o_nnz(p_row) + 1
               end if
            end do
            call MatRestoreRow(W, i_loc, ncols, cols, PETSC_NULL_SCALAR_POINTER, ierr)
         end do
         ! The identity block is always local
         if (identity) then
            do i_loc = 1, local_rows_coarse
               p_row = is_pointer_coarse(i_loc) - global_row_start + 1
               d_nnz(p_row) = d_nnz(p_row) + 1
            end do
         end if
         call MatXAIJSetPreallocation(P, one, d_nnz, o_nnz, &
                  PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_INTEGER_ARRAY, ierr)
         deallocate(d_nnz, o_nnz)
      end if

      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(P, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)      
      call MatSetOption(P, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)           

      ! Copy in the values of W
      do i_loc = global_row_start_W, global_row_end_plus_one_W-1
         call MatGetRow(W, i_loc, ncols, cols, vals, ierr)
         if (ncols > 0) then
            row_array(1) = is_pointer_fine(i_loc - global_row_start_W + 1)
            call MatSetValues(P, one, row_array, ncols, cols(1:ncols), vals(1:ncols), INSERT_VALUES, ierr)
         end if
         call MatRestoreRow(W, i_loc, ncols, cols, vals, ierr)          
      end do

      ! If we want the identity block or just leave it zero
      if (identity) then
         do i_loc = 1, local_rows_coarse
            call MatSetValue(P, is_pointer_coarse(i_loc), i_loc - 1 + global_col_start_W, &
                     PFLARE_ONE, INSERT_VALUES, ierr)
         end do     
      end if
      call MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY, ierr)
      
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

            ! If we're doing reuse and debug, then the cpu version reuses the output
            ! matrix itself (as that has the sparsity the cpu version expects) and so
            ! we compare against a copy of the kokkos result
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
      PetscInt :: local_coarse_size, local_fine_size, local_full_cols
      PetscInt :: i_loc, ncols, max_ncols
      PetscInt :: global_coarse_size, global_fine_size, global_full_cols
      PetscInt :: rows_ao, cols_ao, rows_ad, cols_ad, size_cols
      PetscInt :: global_rows_z, global_cols_z
      PetscInt :: local_rows_z, local_cols_z
      integer :: comm_size, errorcode
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX      
      PetscInt, dimension(:), pointer :: cols => null()
      ! Matrix entries filled by MatGetRow are PetscScalar
      PetscScalar, dimension(:), pointer :: vals => null()
      ! The columns of a row of Z in the full matrix
      PetscInt, allocatable, dimension(:) :: row_cols, d_nnz, o_nnz
      PetscInt, dimension(1) :: row_array
      PetscInt, dimension(:), pointer :: colmap
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0
      type(tMat) :: Ad, Ao
      PetscInt, allocatable, dimension(:) :: orig_indices_array
      type(tIS) :: col_indices, orig_nonlocal
      PetscInt, dimension(:), pointer :: is_pointer_orig_fine_col, is_pointer_coarse, is_pointer_fine
      PetscInt, dimension(:), pointer :: is_pointer_orig_nonlocal
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
            
         ! The local columns of Z are the local fine points in order, so their original
         ! indices are just the local entries of is_fine and need no comms
         ! We deliberately have the local ones first then the off processor ones, as we
         ! insert Z into the full matrix in pieces with Ad, then Ao, so that way we can
         ! index into orig_fine_col_indices in that order
         size_cols = cols_ad
         if (comm_size /= 1) size_cols = cols_ad + cols_ao
         allocate(orig_indices_array(size_cols))
         orig_indices_array(1:cols_ad) = is_pointer_fine(1:cols_ad)

         ! Only the off-processor columns have to be looked up on other ranks
         if (comm_size /= 1) then

            ! Has to be called on every rank, even those without off-processor columns, as it is collective
            call ISCreateGeneral(MPI_COMM_MATRIX, cols_ao, colmap, PETSC_USE_POINTER, col_indices, ierr)
            ! This could be expensive as the number of off-processor columns in Z grows!
            call ISCreateSubIS(is_fine, col_indices, orig_nonlocal, ierr)
            call ISGetIndices(orig_nonlocal, is_pointer_orig_nonlocal, ierr)
            orig_indices_array(cols_ad + 1:size_cols) = is_pointer_orig_nonlocal(1:cols_ao)
            call ISRestoreIndices(orig_nonlocal, is_pointer_orig_nonlocal, ierr)
            call ISDestroy(orig_nonlocal, ierr)
            call ISDestroy(col_indices, ierr)

         end if

         ! The original fine indices of all the columns of Z, local then off-processor
         call ISCreateGeneral(MPI_COMM_MATRIX, size_cols, orig_indices_array, PETSC_COPY_VALUES, &
                  orig_fine_col_indices, ierr)
         deallocate(orig_indices_array)

      end if

      ! Get the indices
      call ISGetIndices(orig_fine_col_indices, is_pointer_orig_fine_col, ierr)

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
      end if      

      ! Get the longest row of Ad or Ao and if we're not reusing, the number of entries in each 
      ! row of R so we can exactly preallocate
      ! The columns of Ad are local fine points and so are in the diagonal block of R, the
      ! columns of Ao are fine points on other ranks and so are in the off-diagonal block, and
      ! the identity is local
      if (.NOT. reuse) allocate(d_nnz(local_rows_z), o_nnz(local_rows_z))
      max_ncols = 0
      do i_loc = 1, local_rows_z
         call MatGetRow(Ad, i_loc-1, ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
         max_ncols = max(max_ncols, ncols)
         if (.NOT. reuse) then
            d_nnz(i_loc) = ncols
            if (identity) d_nnz(i_loc) = d_nnz(i_loc) + 1
            o_nnz(i_loc) = 0
         end if
         call MatRestoreRow(Ad, i_loc-1, ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
         if (comm_size /= 1) then
            call MatGetRow(Ao, i_loc-1, ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
            max_ncols = max(max_ncols, ncols)
            if (.NOT. reuse) o_nnz(i_loc) = ncols
            call MatRestoreRow(Ao, i_loc-1, ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
         end if
      end do
      allocate(row_cols(max_ncols))

      if (.NOT. reuse) then
         call MatXAIJSetPreallocation(R, one, d_nnz, o_nnz, &
                  PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_INTEGER_ARRAY, ierr)
         deallocate(d_nnz, o_nnz)
      end if

      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(R, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)      
      call MatSetOption(R, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr) 

      ! Z - do Ad and Ao separately, as that way we have local indices into is_pointer_orig_fine_col
      ! to give us the original column numbers 
      ! Remember Ad and Ao are serial
      do i_loc = 1, local_rows_z
         row_array(1) = i_loc - 1 + global_row_start_Z

         ! Ad
         call MatGetRow(Ad, i_loc-1, ncols, cols, vals, ierr)
         if (ncols > 0) then
            row_cols(1:ncols) = is_pointer_orig_fine_col(cols(1:ncols)+1)
            call MatSetValues(R, one, row_array, ncols, row_cols(1:ncols), vals(1:ncols), INSERT_VALUES, ierr)
         end if
         call MatRestoreRow(Ad, i_loc-1, ncols, cols, vals, ierr)         

         ! Ao
         if (comm_size /= 1) then
            call MatGetRow(Ao, i_loc-1, ncols, cols, vals, ierr)
            if (ncols > 0) then
               row_cols(1:ncols) = is_pointer_orig_fine_col(cols(1:ncols)+1 + cols_ad)
               call MatSetValues(R, one, row_array, ncols, row_cols(1:ncols), vals(1:ncols), INSERT_VALUES, ierr)
            end if
            call MatRestoreRow(Ao, i_loc-1, ncols, cols, vals, ierr)          
         end if

         ! If we want the identity block or just leave it zero
         if (identity) then
            call MatSetValue(R, row_array(1), is_pointer_coarse(i_loc), PFLARE_ONE, INSERT_VALUES, ierr)
         end if
      end do
      deallocate(row_cols)

      call MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY, ierr)

      call ISRestoreIndices(orig_fine_col_indices, is_pointer_orig_fine_col, ierr)
      call ISRestoreIndices(is_coarse, is_pointer_coarse, ierr)
      call ISRestoreIndices(is_fine, is_pointer_fine, ierr)       
         
   end subroutine compute_R_from_Z_cpu

   !-------------------------------------------------------------------------------------------------------------------------------

end module grid_transfer

