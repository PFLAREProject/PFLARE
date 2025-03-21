module sai_z

   use iso_c_binding
   use petscksp
   use sorting
   use c_petsc_interfaces
   use petsc_helper

#include "petsc/finclude/petscksp.h"

   implicit none

#include "petsc_legacy.h"

   public

   PetscEnum, parameter :: AIR_Z_PRODUCT=0
   PetscEnum, parameter :: AIR_Z_LAIR=1
   PetscEnum, parameter :: AIR_Z_LAIR_SAI=2
   
   PetscEnum, parameter :: PFLAREINV_SAI=5
   PetscEnum, parameter :: PFLAREINV_ISAI=6    
   
   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine calculate_and_build_sai_z(A_ff, A_cf, sparsity_mat_cf, incomplete, reuse_mat, z)

      ! Computes an approximation to z using sai/isai
      ! If incomplete is true then this is lAIR
      ! Can also use this to compute an SAI (ie an inverse in the non rectangular case)
      ! by giving A_cf as the negative identity (of the size A_ff)

      ! ~~~~~~
      type(tMat), intent(in)                            :: A_ff, A_cf, sparsity_mat_cf
      logical, intent(in)                               :: incomplete
      type(tMat), intent(inout)                         :: reuse_mat, z

      ! Local variables 
      PetscInt :: local_rows, local_cols, ncols, global_row_start, global_row_end_plus_one
      PetscInt :: global_rows, global_cols, iterations_taken
      PetscInt :: i_loc, j_loc, cols_ad, rows_ad
      PetscInt :: rows_ao, cols_ao, ifree, row_size, i_size, j_size
      PetscInt :: global_row_start_aff, global_row_end_plus_one_aff
      integer :: lwork, intersect_count, location
      integer :: errorcode, comm_size
      PetscErrorCode :: ierr
      MPI_Comm :: MPI_COMM_MATRIX      
      type(tMat) :: transpose_mat
      type(tIS), dimension(1) :: i_row_is, j_col_is, all_cols_indices
      type(tIS), dimension(1) :: col_indices
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0, maxits=1000
      PetscInt, dimension(:), allocatable :: j_rows, i_rows, ad_indices
      integer, dimension(:), allocatable :: pivots, j_indices, i_indices
      PetscInt, dimension(:), pointer :: cols
      PetscReal, dimension(:), pointer :: vals
      PetscReal, dimension(:), allocatable :: e_row, j_vals
      PetscReal, dimension(:,:), allocatable :: submat_vals
      type(itree) :: i_rows_tree
      PetscReal, dimension(:), allocatable :: work
      type(tVec) :: solution, rhs
      logical :: approx_solve
      type(tMat) :: Ao, Ad, temp_mat
      type(tKSP) :: ksp
      type(tPC) :: pc
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<23)      
      PetscOffset :: iicol
      PetscInt :: icol(1)
      ! In fortran this needs to be of size n+1 where n is the number of submatrices we want
      type(tMat), dimension(2) :: submatrices, submatrices_full
#else
      PetscInt, dimension(:), pointer :: colmap
      type(tMat), dimension(:), pointer :: submatrices, submatrices_full
#endif
      type(c_ptr) :: colmap_c_ptr
      PetscInt, pointer :: colmap_c(:)
      PetscInt, dimension(:), allocatable :: col_indices_off_proc_array
      integer(c_long_long) :: A_array
      MatType:: mat_type
      PetscScalar, dimension(:), pointer :: vec_vals

      ! ~~~~~~

      call PetscObjectGetComm(A_ff, MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)

      ! Get the local sizes
      call MatGetLocalSize(A_cf, local_rows, local_cols, ierr)
      call MatGetSize(A_cf, global_rows, global_cols, ierr)     

      ! ~~~~~~~~~~~~~
      ! ~~~~~~~~~~~~~

      ! We're enforcing the same sparsity 
      
      ! If not re-using
      if (PetscMatIsNull(z)) then
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
      call MatGetType(A_ff, mat_type, ierr)

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
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<23)
            ! Let's get the diagonal and off-diagonal parts
            call MatMPIAIJGetSeqAIJ(sparsity_mat_cf, Ad, Ao, icol, iicol, ierr) 
#else
            call MatMPIAIJGetSeqAIJ(sparsity_mat_cf, Ad, Ao, colmap, ierr) 
#endif
            A_array = sparsity_mat_cf%v
         else
            call MatConvert(sparsity_mat_cf, MATMPIAIJ, MAT_INITIAL_MATRIX, temp_mat, ierr)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<23)
            ! Let's get the diagonal and off-diagonal parts
            call MatMPIAIJGetSeqAIJ(temp_mat, Ad, Ao, icol, iicol, ierr) 
#else
            call MatMPIAIJGetSeqAIJ(temp_mat, Ad, Ao, colmap, ierr) 
#endif            
            A_array = temp_mat%v
         end if

         ! Have to be careful here as we don't have a square matrix, so rows_ao isn't equal to the number of local columns
         call MatGetSize(Ad, rows_ad, cols_ad, ierr)             
         ! We know the col size of Ao is the size of colmap, the number of non-zero offprocessor columns
         call MatGetSize(Ao, rows_ao, cols_ao, ierr)         

         ! For the column indices we need to take all the columns
         call get_colmap_c(A_array, colmap_c_ptr)
         call c_f_pointer(colmap_c_ptr, colmap_c, shape=[cols_ao])

         ! These are the global indices of the columns we want
         ! Taking care here to use cols_ad and not rows_ao  
         allocate(col_indices_off_proc_array(cols_ad + cols_ao))
         allocate(ad_indices(cols_ad))
         ! Local rows (as global indices)
         do ifree = 1, cols_ad
            ad_indices(ifree) = global_row_start_aff + ifree - 1
         end do

         ! Do a sort on the indices
         ! Both ad_indices and colmap_c should already be sorted so we can merge them together quickly
         call merge_pre_sorted(ad_indices, colmap_c, col_indices_off_proc_array)
         deallocate(ad_indices)

         ! Create the sequential IS we want with the cols we want (written as global indices)
         call ISCreateGeneral(PETSC_COMM_SELF, cols_ad + cols_ao, col_indices_off_proc_array, PETSC_USE_POINTER, col_indices(1), ierr) 

         ! ~~~~~~~
         ! Now we can pull out the chunk of matrix that we need
         ! ~~~~~~~

         ! Setting this is necessary to avoid an allreduce when calling createsubmatrices
         ! This will be reset to false after the call to createsubmatrices
         call MatSetOption(A_ff, MAT_SUBMAT_SINGLEIS, PETSC_TRUE, ierr)       
         
         ! Now this will be doing comms to get the non-local rows we want
         ! This matrix has the local rows and the non-local rows in it
         ! We could just request the non-local rows, but it's easier to just get the whole slab
         ! as then the row indices match colmap
         ! This returns a sequential matrix
         if (incomplete) then
            
            if (.NOT. PetscMatIsNull(reuse_mat)) then
               submatrices_full(1) = reuse_mat
               call MatCreateSubMatrices(A_ff, one, col_indices, col_indices, MAT_REUSE_MATRIX, submatrices_full, ierr)
            else
               call MatCreateSubMatrices(A_ff, one, col_indices, col_indices, MAT_INITIAL_MATRIX, submatrices_full, ierr)
               reuse_mat = submatrices_full(1)
            end if

         else

            ! Now we need to create a sequential IS that points at all the columns of the full matrix
            ! The IS stride doesn't actually store all the column integers, it just stores the start, end and stride
            ! So no need to worry about memory use
            ! The size and identity are what MatCreateSubMatrices uses to match allcolumns
            call ISCreateStride(PETSC_COMM_SELF, global_cols, zero, one, all_cols_indices(1), ierr) 
            ! This plus MAT_SUBMAT_SINGLEIS above tells the matcreatesubmatrices that we have an identity for the columns
            call ISSetIdentity(all_cols_indices(1), ierr)            

            ! We need to get all the column entries in the nonlocal rows 
            ! Have to be careful though, when we trigger allcolumns, the column indices are returned without 
            ! change, ie they are not rewritten to be the local row indices
            ! This means we will have to map any column indices we use 
            ! This is very slow in parallel and doesn't scale well! 
            ! There is no easy way in petsc to return only the non-zero columns for a given set of rows
            if (.NOT. PetscMatIsNull(reuse_mat)) then
               submatrices_full(1) = reuse_mat        
               call MatCreateSubMatrices(A_ff, one, col_indices, all_cols_indices, MAT_REUSE_MATRIX, submatrices_full, ierr)
            else
               call MatCreateSubMatrices(A_ff, one, col_indices, all_cols_indices, MAT_INITIAL_MATRIX, submatrices_full, ierr)
               reuse_mat = submatrices_full(1)
            end if
            call ISDestroy(all_cols_indices(1), ierr)

         end if

         row_size = size(col_indices_off_proc_array)
         call ISDestroy(col_indices(1), ierr)

      ! Easy in serial as we have everything we neeed
      else
         
         submatrices_full(1) = A_ff
         ! local rows is the size of c, local cols is the size of f
         row_size = local_cols
         allocate(col_indices_off_proc_array(local_cols))
         do ifree = 1, local_cols
            col_indices_off_proc_array(ifree) = ifree-1
         end do
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


      ! Now go through each of the rows
      ! GetRow has to happen over the global indices
      do i_loc = global_row_start, global_row_end_plus_one-1                  

         ! We just want the F-indices of whatever distance we are going out to
         call MatGetRow(sparsity_mat_cf, i_loc, ncols, &
                  cols, PETSC_NULL_SCALAR_POINTER, ierr) 
                  
         allocate(j_rows(ncols))
         allocate(j_vals(ncols))
         j_vals = 0
         j_rows = cols(1:ncols)

         call MatRestoreRow(sparsity_mat_cf, i_loc, ncols, &
                  cols, PETSC_NULL_SCALAR_POINTER, ierr) 

         ! If we have no non-zeros in this row skip it
         ! This means we have a c point with no neighbour f points
         if (size(j_rows) == 0) then
            deallocate(j_rows, j_vals)
            cycle
         end if                  

         ! ~~~~~~~~
         ! We need to stick the non-zero row of A_cf into the indices of 
         ! A_cf, or A_cf A_ff, or A_cf A_ff^2, etc given whatever distance we're going out to
         call MatGetRow(A_cf, i_loc, ncols, &
                  cols, vals, ierr) 

         allocate(j_indices(size(j_rows)))
         allocate(i_indices(ncols))
         call intersect_pre_sorted_indices_only(j_rows, cols(1:ncols), j_indices, i_indices, intersect_count)                   
         j_vals(j_indices(1:intersect_count)) = vals(i_indices(1:intersect_count))
         deallocate(j_indices, i_indices)

         call MatRestoreRow(A_cf, i_loc, ncols, &
                  cols, vals, ierr)                   

         ! ~~~~~~~~

         ! We already have the global indices, we need the local ones
         ! This is why we sort the col_indices_off_proc_array above, to make this search easy
         do j_loc = 1, size(j_rows)
            call sorted_binary_search(col_indices_off_proc_array, j_rows(j_loc), location)
            if (location == -1) then
               print *, "Couldn't find location"
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
            end if
            j_rows(j_loc) = location-1
         end do                    

         approx_solve = .FALSE.

         ! This is the "incomplete" SAI (Antz 2018) which minimises for column j
         ! ie it doesn't use the shadow I so gives a square system that can be solved
         ! exactly for = 0
         ! This is equivalent to a restricted additive schwarz
         ! ||M(j, J) A(J, J) - eye(j, J)||_2             
         if (incomplete) then
            allocate(i_rows(size(j_rows)))
            i_rows = j_rows

         ! This is the SAI which minimises for column j
         ! which gives the rectangular system that must be minimised
         ! ||M(j, J) A(J, I) - eye(j, I)||_2             
         else

            ! Loop over all the non zero column indices, and get the nonzero columns in those rows
            ! ie get the shadow I
            do j_loc = 1, size(j_rows)

               ! We just want the indices
               call MatGetRow(submatrices_full(1), j_rows(j_loc), ncols, &
                        cols, PETSC_NULL_SCALAR_POINTER, ierr) 
   
               call create_knuth_shuffle_tree_array(cols(1:ncols), &
                        i_rows_tree)                       
   
               call MatRestoreRow(submatrices_full(1), j_rows(j_loc), ncols, &
                        cols, PETSC_NULL_SCALAR_POINTER, ierr)             
   
            end do

            allocate(i_rows(i_rows_tree%length))
            call itree2vector(i_rows_tree, i_rows)
            call flush_tree(i_rows_tree)

         end if

         ! If we have a big system to solve, do its iteratively
         if (size(i_rows) > 40 .OR. size(j_rows) > 40) approx_solve = .TRUE.

         ! This determines the indices of J^* in I*
         allocate(j_indices(size(j_rows)))
         allocate(i_indices(size(i_rows)))
         if (incomplete) then
            call intersect_pre_sorted_indices_only(i_rows, j_rows, i_indices, j_indices, intersect_count)       
         else
            ! The i_rows are the global column indexes, the j_rows are the local
            call intersect_pre_sorted_indices_only(i_rows, col_indices_off_proc_array(j_rows+1), &
                     i_indices, j_indices, intersect_count)       
         end if

         ! Create the sequential IS we want with the cols we want (written as global indices)
         i_size = size(i_rows)
         j_size = size(j_rows)
         call ISCreateGeneral(PETSC_COMM_SELF, i_size, &
               i_rows, &
               PETSC_COPY_VALUES, i_row_is(1), ierr)  
         call ISCreateGeneral(PETSC_COMM_SELF, j_size, &
               j_rows, &
               PETSC_COPY_VALUES, j_col_is(1), ierr) 

         ! Setting this is necessary to avoid an allreduce when calling createsubmatrices
         ! This will be reset to false after the call to createsubmatrices
         ! Shouldn't be needed given submatrices_full(1) is sequential, but whats the harm
         call MatSetOption(submatrices_full(1), MAT_SUBMAT_SINGLEIS, PETSC_TRUE, ierr)                             

         ! This should just be an entirely local operation
         call MatCreateSubMatrices(submatrices_full(1), one, j_col_is, i_row_is, MAT_INITIAL_MATRIX, submatrices, ierr)        
         
         ! Pull out the entries of the submatrix into a dense mat
         if (.NOT. approx_solve) then

            ! This is the correct size as we are going to transpose directly
            allocate(submat_vals(size(i_rows), size(j_rows)))
            submat_vals = 0            
               
            ! Pull out the submat into a dense matrix
            do j_loc = 1, size(j_rows)
                  
               call MatGetRow(submatrices(1), j_loc - 1, ncols, &
                        cols, vals, ierr) 

               ! There is a transpose here! As we want to solve A(J*, I*)^T z(j,J^*)^T = -A_cf(j,I*)^T
               submat_vals(cols(1:ncols)+1, j_loc) = vals(1:ncols)

               call MatRestoreRow(submatrices(1), j_loc - 1, ncols, &
                        cols, vals, ierr)             
            end do
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

               call KSPSetOperators(ksp, submatrices(1), submatrices(1), ierr)             
               call KSPSetUp(ksp, ierr) 

               call MatCreateVecs(submatrices(1), solution, PETSC_NULL_VEC, ierr)
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
               call MatTranspose(submatrices(1), MAT_INITIAL_MATRIX, transpose_mat, ierr)

               call KSPSetOperators(ksp, transpose_mat, transpose_mat, ierr)                           
               call KSPSetUp(ksp, ierr)

               call MatCreateVecs(submatrices(1), solution, rhs, ierr)  
               ! Have to restore the array before the solve in case this is kokkos
               call VecGetArray(rhs, vec_vals, ierr)
               vec_vals(1:i_size) = e_row(1:i_size)
               call VecRestoreArray(rhs, vec_vals, ierr)                                

               ! Do the solve
               call KSPSolve(ksp, rhs, solution, ierr)
               call KSPGetIterationNumber(ksp, iterations_taken, ierr)

               ! Copy solution into e_row
               call VecGetArray(solution, vec_vals, ierr)
               e_row(1:size(j_rows)) = vec_vals(1:size(j_rows))
               call VecRestoreArray(solution, vec_vals, ierr) 

               call KSPReset(ksp, ierr)
               call VecDestroy(solution, ierr)
               call VecDestroy(rhs, ierr)
               call MatDestroy(transpose_mat, ierr)

            ! ~~~~~~~~~~~~~
            ! Exact dense solve with QR
            ! ~~~~~~~~~~~~~
            else

               allocate(work(1))
               lwork = -1
               call dgels('N', size(i_rows), size(j_rows), 1, submat_vals, size(i_rows), &
                           e_row, size(i_rows), work, lwork, errorcode)
               lwork = int(work(1))
               deallocate(work)
               allocate(work(lwork))  
               call dgels('N', size(i_rows), size(j_rows), 1, submat_vals, size(i_rows), &
                           e_row, size(i_rows), work, lwork, errorcode)
               deallocate(work)

            end if
         end if

         call MatDestroySubMatrices(one, submatrices, ierr)

         ! ~~~~~~~~~~~~~
         ! Set all the row values
         ! ~~~~~~~~~~~~~
         call MatSetValues(z, one, [i_loc], j_size, col_indices_off_proc_array(j_rows+1), e_row, INSERT_VALUES, ierr)            

         deallocate(j_rows, i_rows, e_row, j_vals, j_indices, i_indices)  
         if (allocated(submat_vals)) deallocate(submat_vals)
         call ISDestroy(i_row_is(1), ierr)
         call ISDestroy(j_col_is(1), ierr)               
      end do  

      if (comm_size /= 1 .AND. mat_type /= "mpiaij") then
         call MatDestroy(temp_mat, ierr)
      end if        

      call KSPDestroy(ksp, ierr)
      call MatAssemblyBegin(z, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(z, MAT_FINAL_ASSEMBLY, ierr)       

   end subroutine calculate_and_build_sai_z

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine calculate_and_build_sai(matrix, sparsity_order, incomplete, reuse_mat, inv_matrix)

      ! Computes an approximate inverse with an SAI (or an ISAI)
      ! This just builds an identity and then calls the calculate_and_build_sai_z code

      ! ~~~~~~

      type(tMat), intent(in)                            :: matrix
      integer, intent(in)                               :: sparsity_order
      logical, intent(in)                               :: incomplete
      type(tMat), intent(inout)                         :: reuse_mat, inv_matrix    

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
         if (PetscMatIsNull(inv_matrix)) then

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
      call calculate_and_build_sai_z(matrix, minus_I, sparsity_mat_cf, incomplete, reuse_mat, inv_matrix)     
      call MatDestroy(minus_I, ierr)
      if (sparsity_order .ge. 2) call MatDestroy(sparsity_mat_cf, ierr)      

   end subroutine

end module sai_z

