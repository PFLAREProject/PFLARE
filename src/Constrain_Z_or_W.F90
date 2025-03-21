module constrain_z_or_w

   use iso_c_binding
   use petscksp
   use c_petsc_interfaces
   use petsc_helper

#include "petsc/finclude/petscksp.h"

   implicit none

#include "petsc_legacy.h"

   public     
   
   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine get_near_nullspace(input_mat, left, right, left_null_vecs, right_null_vecs)

      ! Gets left and/or right near nullspace vectors set by the user with 
      ! matsetnearnullspace

      ! ~~~~~~
      type(tMat), intent(in)                                :: input_mat
      logical, intent(in)                                   :: left, right
      type(tVec), dimension(:), allocatable, intent(inout)  :: left_null_vecs, right_null_vecs

      integer :: comm_size, errorcode
      MatNullSpace :: nullspace
      PetscErrorCode :: ierr
      MPI_Comm :: MPI_COMM_MATRIX
      PetscBool :: has_constant
      PetscInt :: no_nullspace, no_nullspace_vecs, i_loc
      type(tVec), dimension(:), pointer :: null_vecs      
      logical :: cst_nullspace

      ! ~~~~~

      ! Get the communicator the input matrix is on, we build everything on that
      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)      

      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)      

      ! Get the near nullspace if the user has set one 
      call MatGetNearNullSpace(input_mat, nullspace, ierr)
      ! If the user hasn't set a near nullspace and still wants constraints
      ! we use the constant as a default
      cst_nullspace = .FALSE.
      no_nullspace_vecs = 0
      no_nullspace = 0
      if (PetscNullspaceIsNull(nullspace) .AND. &
            (left .OR. right)) then

         cst_nullspace = .TRUE.
         no_nullspace_vecs = 1
         
      ! If the user has provided a near null-space
      else if (.NOT. PetscNullspaceIsNull(nullspace) .AND. &
                  (left .OR. right)) then    

         ! Get the nullspace vectors
         call MatNullSpaceGetVecs(nullspace, has_constant, no_nullspace, null_vecs, ierr)       
         no_nullspace_vecs = no_nullspace
         
         ! If has_constant is true, then null_vecs won't contain the constant vector
         ! but it is still part of the nullspace - we explicitly create it below
         if (has_constant .eqv. PETSC_TRUE) then
            cst_nullspace = .TRUE.
            no_nullspace_vecs = no_nullspace_vecs + 1
         end if
      end if

      ! Create some storage for all the near nullspace vectors
      ! constant included
      if (left) allocate(left_null_vecs(no_nullspace_vecs))
      if (right) allocate(right_null_vecs(no_nullspace_vecs))
      ! Have to make copies of the nullspace vecs passed in, as MatNullSpaceCreate locks the vectors
      ! to be read-only in newer petsc
      do i_loc = 1, no_nullspace
         if (left) then
            call VecDuplicate(null_vecs(i_loc), left_null_vecs(i_loc), ierr)
            call VecCopy(null_vecs(i_loc), left_null_vecs(i_loc), ierr)
         end if
         if (right) then
            call VecDuplicate(null_vecs(i_loc), right_null_vecs(i_loc), ierr)
            call VecCopy(null_vecs(i_loc), right_null_vecs(i_loc), ierr)            
         end if
      end do

      ! If we want to create a constant nullspace vector on the end of left_null_vecs or right_null_vecs
      if (cst_nullspace) then           

         if (left) then
            call MatCreateVecs(input_mat, left_null_vecs(no_nullspace_vecs), PETSC_NULL_VEC, ierr)
            ! Set to the constant     
            call VecSet(left_null_vecs(no_nullspace_vecs), 1d0, ierr)
         end if

         if (right) then
            call MatCreateVecs(input_mat, right_null_vecs(no_nullspace_vecs), PETSC_NULL_VEC, ierr)
            ! Set to the constant
            call VecSet(right_null_vecs(no_nullspace_vecs), 1d0, ierr)
         end if         
      end if      
      
   end subroutine get_near_nullspace

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine smooth_near_nullspace(input_mat, left, right, left_null_vecs, right_null_vecs)

      ! Smooths left and/or right near nullspace vectors

      ! ~~~~~~
      type(tMat), intent(in)                            :: input_mat
      logical, intent(in)                               :: left, right
      type(tVec), dimension(:), intent(inout)           :: left_null_vecs, right_null_vecs

      integer :: comm_size, errorcode, i_loc
      PetscInt :: local_rows, global_rows, local_cols, global_cols
      PetscErrorCode :: ierr
      MPI_Comm :: MPI_COMM_MATRIX
      type(tVec) :: vec_rhs
      type(tKSP) :: ksp
      type(tPC) :: pc
      PetscInt, parameter :: maxits=15

      ! ~~~~~~
      ! Just return if we don't want either left or right (or both)
      if (.NOT. (left .OR. right)) then
         return
      end if

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)
      call MatGetSize(input_mat, global_rows, global_cols, ierr)    
      call MatGetLocalSize(input_mat, local_rows, local_cols, ierr)
      
      ! Create the vecs
      call MatCreateVecs(input_mat, vec_rhs, PETSC_NULL_VEC, ierr)

      ! ~~~~~~~~~~~~
      ! Setup a petsc ksp to solve Ax = 0
      ! ~~~~~~~~~~~~

      ! We want the nullspace so solve homog problem
      call VecSet(vec_rhs, 0d0, ierr)     

      call KSPCreate(MPI_COMM_MATRIX, ksp, ierr)
      !call KSPSetType(ksp, KSPGMRES, ierr)
      call KSPSetType(ksp, KSPRICHARDSON, ierr)
      ! Scale the richardson with a dampening parameter
      call KSPRichardsonSetSelfScale(ksp, PETSC_TRUE, ierr)

      call KSPGetPC(ksp, pc, ierr)
      call PCSetType(pc, PCJACOBI, ierr)    
      ! Use the L1 Jacobi
      !call PCJacobiSetType(pc, 1, ierr)   

      ! Do 15 iterations
      call KSPSetTolerances(ksp, 1d-14, &
               & 1d-50, &
               & PETSC_DEFAULT_REAL, &
               & maxits, ierr) 

      call KSPSetOperators(ksp, input_mat, input_mat, ierr)             
      !call KSPSetFromOptions(ksp_coarse_solver, ierr)    
      call KSPSetInitialGuessNonzero(ksp, PETSC_TRUE, ierr)
      call KSPSetNormType(ksp, KSP_NORM_NONE, ierr)
      call KSPSetUp(ksp, ierr)    
      
      ! ~~~~~~~
      ! Compute the left near nullspace vec
      ! ~~~~~~~
      if (left) then    
         
         ! Smooth each vector
         do i_loc = 1, size(left_null_vecs)           

            ! To get at the left near nullspace vectors we solve the transposed problem
            call KSPSolveTranspose(ksp, vec_rhs, left_null_vecs(i_loc), ierr)

         end do
      end if

      ! ~~~~~~~
      ! Compute the right near nullspace vec
      ! ~~~~~~~
      if (right) then     
         
         ! Smooth each vector
         do i_loc = 1, size(right_null_vecs)           

            ! To get at the right near nullspace vectors we solve the untransposed problem
            call KSPSolve(ksp, vec_rhs, right_null_vecs(i_loc), ierr)         
         end do
      end if      

      call VecDestroy(vec_rhs, ierr)
      call KSPDestroy(ksp, ierr)          
      
      
   end subroutine smooth_near_nullspace

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine constrain_grid_transfer(z_or_w, is_z, null_vecs_f, null_vecs_c)

      ! Applies constraints based on near nullspace vectors to Z or W passed in
      ! For Z, we need to apply constraints to the columns, so is_z should be passed in true
      ! so that we can take a transpose to make things easy
      ! For W, we need to apply constraints to the rows

      ! ~~~~~~
      type(tMat), intent(inout)                         :: z_or_w
      logical, intent(in)                               :: is_z
      type(tVec), dimension(:), intent(inout)           :: null_vecs_f, null_vecs_c

      ! Local variables
      PetscInt :: local_rows, local_cols, global_row_start
      PetscInt :: global_row_end_plus_one, global_col_end_plus_one
      PetscInt :: global_rows, global_cols, i_loc
      PetscInt :: rows_ao, cols_ao, global_col_start, ncols
      PetscInt :: cols_ad, rows_ad, ncols_ad, ncols_ao, ncols_temp
      integer :: errorcode, comm_size, null_vec
      PetscErrorCode :: ierr      
      MPI_Comm :: MPI_COMM_MATRIX
      type(tMat) :: row_mat, temp_mat_aij
      PetscInt, dimension(:), allocatable :: col_indices_off_proc_array
      PetscInt, dimension(:), pointer :: cols
      PetscReal, dimension(:), pointer :: vals
      PetscReal, dimension(:), allocatable :: row_vals, diff
      type(tMat) :: new_z_or_w
      type(c_ptr) :: colmap_c_ptr, b_c_nonlocal_c_ptr
      integer(c_long_long) :: A_array, vec_long
      PetscInt, pointer :: colmap_c(:) => null()
      type(tMat) :: Ad, Ao
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<23)      
      PetscOffset :: iicol
      PetscInt :: icol(1) 
#else
      PetscInt, dimension(:), pointer :: colmap
#endif
      real(c_double), pointer :: b_c_nonlocal(:)
      PetscScalar, dimension(:), pointer :: b_c_local, b_f_vals
      PetscReal, dimension(:,:), allocatable :: b_c_nonlocal_alloc, b_c_vals, bctbc, pseudo, temp_mat
      PetscInt, parameter :: one = 1, zero = 0
      MatType:: mat_type

      ! ~~~~~~

      call PetscObjectGetComm(z_or_w, MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)   

      ! Have to do an explicit transpose if applying constraints to columns of Z
      if (is_z) then
         call MatTranspose(z_or_w, MAT_INITIAL_MATRIX, row_mat, ierr)
      else
         row_mat = z_or_w
      end if

      ! Duplicate the sparsity pattern for our new Z or W
      call MatDuplicate(row_mat, MAT_COPY_VALUES, new_z_or_w, ierr)

      ! We know we will never have non-zero locations outside of the sparsity power 
      call MatSetOption(new_z_or_w, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE,  ierr)     
      call MatSetOption(new_z_or_w, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr) 
      ! We know we are only going to insert local vals
      ! These options should turn off any reductions in the assembly
      call MatSetOption(new_z_or_w, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)  
      
      ! ~~~~~~~~~

      ! Get the local sizes
      call MatGetLocalSize(row_mat, local_rows, local_cols, ierr)
      call MatGetSize(row_mat, global_rows, global_cols, ierr)

      ! Get the nnzs in row_mat
      call MatGetOwnershipRange(row_mat, global_row_start, global_row_end_plus_one, ierr) 
      call MatGetOwnershipRangeColumn(row_mat, global_col_start, global_col_end_plus_one, ierr)    

      ! ~~~~~
      ! What we are doing, with W for example:
      ! The constraints we need to enforce are P B_c^R = B^R
      ! ie preservation of the right near nullspace vectors
      ! which given our ideal prolongator is equivalent to 
      ! W B_c^R = B_f^R     - (1)
      ! This is enforced by computing an orthgonal projector and just doing 
      ! a one step method
      ! W = W - (W * B_c^R - B_f^R ) * inv((B_c^R)^T * B_c^R) * (B_c^R)^T
      ! which we can do one row at a time
      ! See Eq 3.2 in Olson 2011 or 
      ! filter_operator in utils.py in PyAMG
      ! If we wanted we could modify this to be done iteratively like clAIR, but 
      ! our approx Aff inverses are so good we don't really need to
      ! If B is a vector of 1 for example, all the constraint does is scales by a row sum
      ! We take a transpose of Z at the start so it is easy to access the columns
      ! via the rows in petsc, we then just transpose at the end  
      ! ~~~~~

      ! ~~~~
      ! Now for each row, we need to get the values of B_c for both the local and nonlocal col indices
      ! The nonlocal col indices are colmap in petsc
      ! We can do the comms for this once by using the VecScatter built into the mat
      ! ~~~~

      call MatGetType(row_mat, mat_type, ierr)

      ! Nonlocal if needed
      if (comm_size /= 1) then

         ! Much more annoying in older petsc
         if (mat_type == "mpiaij") then
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<23)
            ! Let's get the diagonal and off-diagonal parts
            call MatMPIAIJGetSeqAIJ(row_mat, Ad, Ao, icol, iicol, ierr) 
#else
            call MatMPIAIJGetSeqAIJ(row_mat, Ad, Ao, colmap, ierr) 
#endif
            A_array = row_mat%v

         ! If on the gpu, just do a convert to mpiaij format first
         ! This will be expensive but the best we can do for now without writing our 
         ! own version of this subroutine in cuda/kokkos               
         else
            call MatConvert(row_mat, MATMPIAIJ, MAT_INITIAL_MATRIX, temp_mat_aij, ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<23)
            ! Let's get the diagonal and off-diagonal parts
            call MatMPIAIJGetSeqAIJ(temp_mat_aij, Ad, Ao, icol, iicol, ierr) 
#else
            call MatMPIAIJGetSeqAIJ(temp_mat_aij, Ad, Ao, colmap, ierr) 
#endif            
            A_array = temp_mat_aij%v
         end if

         call MatGetSize(Ad, rows_ad, cols_ad, ierr)             
         ! We know the col size of Ao is the size of colmap, the number of non-zero offprocessor columns
         call MatGetSize(Ao, rows_ao, cols_ao, ierr)         

         ! For the column indices we need to take all the columns of row_mat
         call get_colmap_c(A_array, colmap_c_ptr)
         call c_f_pointer(colmap_c_ptr, colmap_c, shape=[cols_ao])

         allocate(b_c_nonlocal_alloc(cols_ao, size(null_vecs_c)))

         ! Loop over all the near nullspace vectors and get the nonlocal components
         do null_vec = 1, size(null_vecs_c)
         
            ! We want the nonlocal values in B_c
            vec_long = null_vecs_c(null_vec)%v

            ! Do the comms
            ! Have to call restore after we're done with lvec (ie null_vecs_c(null_vec))
            call vecscatter_mat_begin_c(A_array, vec_long, b_c_nonlocal_c_ptr)
            call vecscatter_mat_end_c(A_array, vec_long, b_c_nonlocal_c_ptr)
            ! Nonlocal vals only pointer
            ! b_c_nonlocal now contains all the nonlocal values of B_c we need for all the nonlocal columns
            ! in every local row
            call c_f_pointer(b_c_nonlocal_c_ptr, b_c_nonlocal, shape=[cols_ao])

            ! Copy the values
            b_c_nonlocal_alloc(:, null_vec) = b_c_nonlocal
            ! Make sure to restore as soon as we're done with it
            call vecscatter_mat_restore_c(A_array, b_c_nonlocal_c_ptr)

         end do
         
      ! In serial this is simple
      else

         Ad = row_mat

      end if

      ! Now go through each of the rows
      ! GetRow has to happen over the global indices
      do i_loc = global_row_start, global_row_end_plus_one-1                  

         ! Get how many non-zero columsn we have for this row
         ! Convenience given we are going to interact with Ad and Ao separately
         call MatGetRow(row_mat, i_loc, ncols_temp, &
                  cols, PETSC_NULL_SCALAR_POINTER, ierr) 
         ncols = ncols_temp
         ! We're done with this row
         call MatRestoreRow(row_mat, i_loc, ncols_temp, &
            cols, PETSC_NULL_SCALAR_POINTER, ierr)   
            
         ! Skip this row if there are no non-zeros
         if (ncols == 0) cycle

         ! This is the rhs
         allocate(row_vals(ncols))
         allocate(b_c_vals(ncols, size(null_vecs_c)))
         allocate(bctbc(size(null_vecs_c), size(null_vecs_c)))
         allocate(pseudo(size(null_vecs_c), size(null_vecs_c)))
         allocate(temp_mat(size(null_vecs_c), ncols))

         allocate(diff(size(null_vecs_c)))
         ! This stores the global indices, with local indices first then global
         allocate(col_indices_off_proc_array(ncols))

         ! ~~~~~~~~~~~
         ! Let's pull out existing entries in W and the coarse nullspace components
         ! We have the local values first then the nonlocal
         ! This is so the ordering is easy with colmap and b_c_nonlocal_alloc above
         ! ~~~~~~~~~~~
         ! We do the local values first - remember Ad and Ao need sequential indices
         call MatGetRow(Ad, i_loc - global_row_start, ncols_temp, &
                  cols, vals, ierr) 
         ncols_ad = ncols_temp

         ! this is the local values of W
         row_vals(1:ncols_ad) = vals(1:ncols_ad) 
         ! Global indices of our local cols - be careful here to use col_start not row_start
         col_indices_off_proc_array(1:ncols_ad) = cols(1:ncols_ad) + global_col_start

         ! Get the local b_c values 
         ! Loop over all the near nullspace vectors and get the local components
         do null_vec = 1, size(null_vecs_c)

            call VecGetArray(null_vecs_c(null_vec), b_c_local, ierr)

            ! cols are local indices
            b_c_vals(1:ncols_ad, null_vec) = b_c_local(cols(1:ncols_ad)+1)

            call VecRestoreArray(null_vecs_c(null_vec), b_c_local, ierr)
         end do
         call MatRestoreRow(Ad, i_loc - global_row_start, ncols_temp, &
                  cols, vals, ierr)          

         ! ~~~~~~~~~
         ! Nonlocal if needed
         if (comm_size /= 1) then

            ! Now the nonlocal
            call MatGetRow(Ao, i_loc - global_row_start, ncols_temp, &
                     cols, vals, ierr) 
            
            ncols_ao = ncols_temp
            ! this is the nonlocal values of W
            ! after the local values in row_vals
            row_vals(ncols_ad+1:ncols_ad + ncols_ao) = vals(1:ncols_ao) 
            ! Global indices of our nonlocal cols
            col_indices_off_proc_array(ncols_ad+1:ncols_ad + ncols_ao) = colmap_c(cols(1:ncols_ao)+1) 

            ! Get the nonlocal b_c values
            ! Remembering cols are the local indices into colmap (or equivalently the first dim of b_c_nonlocal_alloc)
            do null_vec = 1, size(null_vecs_c) 
               b_c_vals(ncols_ad+1:ncols_ad + ncols_ao, null_vec) =  b_c_nonlocal_alloc(cols(1:ncols_ao)+1, null_vec) 
            end do

            call MatRestoreRow(Ao, i_loc - global_row_start, ncols_temp, &
                     cols, vals, ierr)  
                     
         end if

         ! ~~~~~~~~~
         ! Now we pull out the fine near nullspace component
         ! We are then going to compute the residual for this row
         ! ie W B_c^R - B_f^R      
         ! ~~~~~~~~~
         ! W B_c^R can be done as (B_c^R)^T W
         call dgemv("T", size(b_c_vals, 1), size(b_c_vals,2), &
                  1d0, b_c_vals, size(b_c_vals,1), &
                  row_vals, 1, &
                  0d0, diff, 1)    

         ! Compute W * B_c^R - B_f^R
         ! Loop over near-nullspace vecs
         do null_vec = 1, size(null_vecs_c)

            call VecGetArray(null_vecs_f(null_vec), b_f_vals, ierr)
            ! b_f_vals is local (we want b_f for this row) but so is the value we want
            diff(null_vec) = diff(null_vec) - b_f_vals(i_loc - global_row_start + 1)    
            call VecRestoreArray(null_vecs_f(null_vec), b_f_vals, ierr)               
         end do                  
               
         ! Compute (B_c^R)^T * B_c^R
         call dgemm("T", "N", size(b_c_vals, 2), size(b_c_vals,2), size(b_c_vals, 1), &
                  1d0, b_c_vals, size(b_c_vals,1), &
                  b_c_vals, size(b_c_vals,1), &
                  0d0, bctbc, size(bctbc,1)) 
                  
         ! Compute the pseudo inverse of (B_c^R)^T * B_c^R
         call pseudo_inv(bctbc, pseudo)

         ! Compute inv((B_c^R)^T * B_c^R) * (B_c^R)^T
         call dgemm("N", "T", size(pseudo, 1), size(b_c_vals,1), size(pseudo, 2), &
                  1d0, pseudo, size(pseudo,1), &
                  b_c_vals, size(b_c_vals,1), &
                  0d0, temp_mat, size(temp_mat,1))          

         ! Now compute -(W * B_c^R - B_f^R ) * inv((B_c^R)^T * B_c^R) * (B_c^R)^T
         ! Again we're doing the left vec mat mult with a transpose
         call dgemv("T", size(temp_mat, 1), size(temp_mat,2), &
                  -1d0, temp_mat, size(temp_mat,1), &
                  diff, 1, &
                  0d0, b_c_vals, 1) 

         ! ~~~~~~~~~~~~~
         ! Set all the row values, same sparsity pattern
         ! ~~~~~~~~~~~~~
         call MatSetValues(new_z_or_w, one, [i_loc], ncols, col_indices_off_proc_array, b_c_vals(:, 1), ADD_VALUES, ierr)            
         deallocate(row_vals, col_indices_off_proc_array, b_c_vals)
         deallocate(diff, bctbc, pseudo, temp_mat)

      end do

      call MatAssemblyBegin(new_z_or_w, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(new_z_or_w, MAT_FINAL_ASSEMBLY, ierr)     
      
      ! Don't forget to restore
      if (comm_size /= 1) then
         deallocate(b_c_nonlocal_alloc)
      end if

      ! Destroy the old input matrix
      call MatDestroy(z_or_w, ierr)      

      if (is_z) then
         ! Destroy the transposed original matrix 
         call MatDestroy(row_mat, ierr)
         ! Now go and transpose the new one back to the original sizes
         call MatTranspose(new_z_or_w, MAT_INITIAL_MATRIX, row_mat, ierr)
         ! And set the output to the newly constrained mat
         z_or_w = row_mat
         call MatDestroy(new_z_or_w, ierr)

      else
         ! And set the output to the newly constrained mat
         z_or_w = new_z_or_w
      end if

      if (comm_size /= 1 .AND. mat_type /= "mpiaij") then
         call MatDestroy(temp_mat_aij, ierr)
      end if

   end subroutine constrain_grid_transfer

   ! -------------------------------------------------------------------------------------------------------------------------------

end module constrain_z_or_w

