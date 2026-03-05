module approx_inverse_setup

   use petscmat
   use tsqr, only: tsqr_buffers
   use pflare_parameters, only: &
         PFLAREINV_POWER, PFLAREINV_ARNOLDI, PFLAREINV_NEWTON, PFLAREINV_NEWTON_NO_EXTRA, &
         PFLAREINV_NEUMANN, PFLAREINV_WJACOBI, PFLAREINV_JACOBI, &
         PFLAREINV_SAI, PFLAREINV_ISAI, AIR_Z_PRODUCT, &
         MF_VEC_DIAG, MF_VEC_TEMP, MF_VEC_RHS, MF_VEC_TEMP_TWO, MF_VEC_TEMP_THREE
   use gmres_poly, only: &
         start_gmres_polynomial_coefficients_power, &
         calculate_gmres_polynomial_coefficients_arnoldi, &
         build_gmres_polynomial_inverse, petsc_matvec_da_poly_mf
   use gmres_poly_newton, only: &
         build_gmres_polynomial_newton_inverse, &
         calculate_gmres_polynomial_roots_newton
   use neumann_poly, only: calculate_and_build_neumann_polynomial_inverse
   use weighted_jacobi, only: calculate_and_build_weighted_jacobi_inverse
   use sai_z, only: calculate_and_build_sai
   use repartition, only: MatMPICreateNonemptySubcomm
   use petsc_helper, only: destroy_matrix_reuse, ShellSetVecType
   use matshell_data_type, only: mat_ctxtype

#include "petsc/finclude/petscmat.h"
      
   implicit none
   public

   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine calculate_and_build_approximate_inverse(matrix, inverse_type, &
                  poly_order, inverse_sparsity_order, &
                  matrix_free, diag_scale_polys, subcomm, &
                  inv_matrix, coefficients)

      ! Builds an approximate inverse
      ! inverse_type:
      ! PFLAREINV_POWER - GMRES polynomial with the power basis 
      ! PFLAREINV_ARNOLDI - GMRES polynomial with the arnoldi basis 
      ! PFLAREINV_NEWTON - GMRES polynomial with the newton basis with extra roots for stability - can only be used matrix-free atm      
      ! PFLAREINV_NEWTON_NO_EXTRA - GMRES polynomial with the newton basis with no extra roots - can only be used matrix-free atm      
      ! PFLAREINV_NEUMANN - Neumann polynomial
      ! PFLAREINV_SAI - SAI
      ! PFLAREINV_ISAI - Incomplete SAI (ie a restricted additive schwartz)
      ! PFLAREINV_WJACOBI - Weighted Jacobi with weight 3 / ( 4 * || D^(-1/2) * A * D^(-1/2) ||_inf )
      ! PFLAREINV_JACOBI - Unweighted Jacobi                  
      ! This is just a wrapper around the start and finish routines below
      ! No opportunity to put work between the async comms
      ! If you want to do that, you will have to call the individual routines
      !
      ! Optional coefficients argument:
      !   - If present and associated on entry: the existing coefficients are reused and
      !     start_approximate_inverse is skipped entirely. buffers is set up manually
      !     (subcomm=.FALSE., buffers%matrix = matrix with ref incremented) so that
      !     finish_approximate_inverse can clean up normally. For a matrix-free inverse
      !     the matshell does NOT take ownership (the caller owns them).
      !   - If present but not associated on entry: fresh coefficients are computed, a
      !     heap allocation is made, and on return the pointer is associated with that
      !     allocation. For matrix-free the matshell also does NOT take ownership (the
      !     caller takes ownership via the returned pointer).
      !   - If absent: existing behaviour. The matshell takes ownership of heap-allocated
      !     coefficients (matrix-free case) or stack storage is used (assembled case).

      ! ~~~~~~
      type(tMat), intent(in)                                        :: matrix
      integer, intent(in)                                           :: inverse_type, poly_order
      integer, intent(in)                                           :: inverse_sparsity_order
      logical, intent(in)                                           :: matrix_free, diag_scale_polys, subcomm
      type(tMat), intent(inout)                                     :: inv_matrix
      ! This pointer must be declared contiguous, as we require coefficients
      ! to be contiguous in build_gmres_polynomial_newton_inverse as 
      ! we muck about with the rank and if this pointer is not declared
      ! contiguous it creates a temporary copy which then disappears and
      ! we segfault when trying to apply mf
      PetscReal, dimension(:, :), contiguous, pointer, optional, intent(inout) :: coefficients

      type(tsqr_buffers)                           :: buffers
      PetscReal, dimension(:, :), contiguous, pointer   :: work_coefficients
      PetscReal, dimension(poly_order + 1, 1), target   :: coefficients_stack
      type(mat_ctxtype), pointer :: mat_ctx => null()
      PetscErrorCode :: ierr
      type(tMat) :: reuse_mat, inv_matrix_temp
      type(tMat), dimension(:), pointer :: reuse_submatrices => null()
      logical :: heap_allocated

      ! ~~~~~~  

      ! This is diabolical - In petsc 3.22, they changed the way to test for 
      ! a null matrix in fortran
      ! Fortran variables are initialized such that they are not PETSC_NULL_MAT
      ! but such that PetscObjectIsNull returns true
      ! Unfortunately, we call this routine from C
      ! I can't seem to find a way to have the C matrix not be PETSC_NULL_MAT
      ! the first time into this routine - I've tried different ways of setting 
      ! the matrix to null in C (or not setting it at all)
      ! If it is PETSC_NULL_MAT then this triggers a null pointer check 
      ! in the mat creation routines  
      ! So we test here if we were given a null matrix
      ! If we weren't, then we copy the pointer to matrix we were given
      ! If we were, then inv_matrix_temp will be null, but in the Fortran way
      ! We then copy the pointer of inv_matrix_temp back into inv_matrix after we're done
      if (inv_matrix /= PETSC_NULL_MAT) then
         inv_matrix_temp = inv_matrix
      end if

      ! The inverse is always returned on the same comm as the input matrix
      ! but some methods benefit from having their intermediate calculations 
      ! done on a subcomm
      buffers%subcomm = subcomm
      ! Don't do any overlapping of comms, if you want that call the start/finish yourself   

      if (present(coefficients) .AND. associated(coefficients)) then

         ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         ! Reuse path: valid coefficients supplied by the caller;
         ! skip start_approximate_inverse entirely to avoid calling
         ! PetscObjectReference on a potential null matrix.
         ! Replicate the non-subcomm setup that start_approximate_inverse
         ! would have done so that finish_approximate_inverse can clean up normally.
         ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         work_coefficients => coefficients
         buffers%on_subcomm = .FALSE.
         buffers%matrix     = matrix
         call PetscObjectReference(matrix, ierr)
         call finish_approximate_inverse(matrix, inverse_type, &
                     poly_order, inverse_sparsity_order, &
                     buffers, work_coefficients, &
                     matrix_free, diag_scale_polys, &
                     reuse_mat, reuse_submatrices, inv_matrix_temp)
         ! Caller owns the coefficient memory; matshell must not free it
         if (matrix_free) then
            call MatShellGetContext(inv_matrix_temp, mat_ctx, ierr)
            mat_ctx%own_coefficients = .FALSE.
         end if

      else

         ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         ! Fresh computation path
         ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         ! Heap-allocate when: matrix-free (matshell will hold the pointer),
         ! coefficients are being returned to the caller, or Newton basis (2 columns).
         ! Otherwise, stack storage suffices.
         heap_allocated = matrix_free .OR. &
                          (present(coefficients) .AND. .NOT. associated(coefficients)) .OR. &
                          inverse_type == PFLAREINV_NEWTON .OR. inverse_type == PFLAREINV_NEWTON_NO_EXTRA
         if (heap_allocated) then
            if (inverse_type == PFLAREINV_NEWTON .OR. inverse_type == PFLAREINV_NEWTON_NO_EXTRA) then
               ! Newton basis needs storage for real and imaginary roots
               allocate(work_coefficients(poly_order + 1, 2))
            else
               allocate(work_coefficients(poly_order + 1, 1))
            end if
         else
            work_coefficients => coefficients_stack
         end if

         ! Start the calculation
         call start_approximate_inverse(matrix, inverse_type, &
                     poly_order, diag_scale_polys, &
                     buffers, work_coefficients)
         ! Finish it
         call finish_approximate_inverse(matrix, inverse_type, &
                     poly_order, inverse_sparsity_order, &
                     buffers, work_coefficients, &
                     matrix_free, diag_scale_polys, &
                     reuse_mat, reuse_submatrices, inv_matrix_temp)

         if (matrix_free) then
            call MatShellGetContext(inv_matrix_temp, mat_ctx, ierr)
            if (present(coefficients)) then
               ! Caller takes ownership; matshell must not free the allocation
               mat_ctx%own_coefficients = .FALSE.
               coefficients => work_coefficients
            else
               ! No external owner: matshell owns and will free on reset
               mat_ctx%own_coefficients = .TRUE.
            end if
         else if (present(coefficients)) then
            ! Return heap-allocated coefficients to caller
            coefficients => work_coefficients
         else if (heap_allocated) then
            ! Heap-allocated (Newton assembled path) but not needed further: free
            deallocate(work_coefficients)
         end if

      end if

      if (.NOT. PetscObjectIsNull(reuse_mat)) then
         call destroy_matrix_reuse(reuse_mat, reuse_submatrices)
      end if

      ! Copy the pointer - see comment above about petsc 3.22
      inv_matrix = inv_matrix_temp

   end subroutine calculate_and_build_approximate_inverse    

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine start_approximate_inverse(matrix, inverse_type, poly_order, diag_scale_polys, &
                  buffers, coefficients)

      ! Starts the assembly of an approximate inverse
      ! We have different types of inverses we can use
      ! Have to call finish_approximate_inverse before it can be used

      ! ~~~~~~
      type(tMat), intent(in)                             :: matrix
      integer, intent(in)                                :: inverse_type, poly_order
      logical, intent(in)                                :: diag_scale_polys
      type(tsqr_buffers), intent(inout)                  :: buffers 
      PetscReal, dimension(:, :), pointer, intent(inout) :: coefficients

      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX, MPI_COMM_BUFFERS_MATRIX
      PetscInt :: local_rows, local_cols, global_rows, global_cols
      integer :: errorcode
      type(mat_ctxtype), pointer :: mat_ctx_da=>null()
      ! ~~~~~~    

      if (buffers%subcomm .AND. inverse_type == PFLAREINV_POWER) then
         print *, "There is no reason to use a subcomm with the power basis, turn off subcomm"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)         
      end if

      if (buffers%subcomm .AND. &
            (inverse_type == PFLAREINV_ARNOLDI .OR. &
            inverse_type == PFLAREINV_NEWTON .OR. &
            inverse_type == PFLAREINV_NEWTON_NO_EXTRA)) then

         ! Create a version of matrix on a subcomm and store it in buffers%matrix
         ! If all ranks have entries, then this just returns a pointer to the original 
         ! matrix (ie buffers%matrix = matrix) with the reference counter incremented
         ! If not, then a copy is taken on a subcomm with ranks that have entries
         ! That is the matrix we use to start our polynomial coefficient calculation
         ! But *importantly* the inverse matrix we create is always on MPI_COMM_WORLD
         ! I did try building a mg hierarchy that progressively moved onto smaller 
         ! subcomms with processor agglomeration, but it ended up expensive
         ! This way, only the reductions done in the coefficient calculation happen on 
         ! the subcomm
         ! On any rank that isn't part of this subcomm, we then need to comm the coefficients 
         ! to those ranks, as petsc in debug mode checks that matscale (and mataxpy) all use
         ! the same coefficient, even if the matrix doesn't have any rows and hence wouldn't use them    
         call MatMPICreateNonemptySubcomm(matrix, buffers%on_subcomm, buffers%matrix)     
      else

         buffers%matrix = matrix
         ! Increase the reference counter
         call PetscObjectReference(matrix, ierr)          
      end if     

      ! ~~~~~~~~~~~~~~~
      ! ~~~~~~~~~~~~~~~

      ! If we're on the subcomm, we don't need to do anything
      if (.NOT. PetscObjectIsNull(buffers%matrix)) then

         ! Have to dynamically allocate this
         allocate(mat_ctx_da)
         mat_ctx_da%mat_scaled = buffers%matrix           

         ! If we want to diagonally scale (ie apply Jacobi preconditioned gmres polynomial)
         if ((inverse_type == PFLAREINV_POWER .OR. &
             inverse_type == PFLAREINV_ARNOLDI .OR. &
             inverse_type == PFLAREINV_NEWTON .OR. &
             inverse_type == PFLAREINV_NEWTON_NO_EXTRA) .AND. &
             diag_scale_polys) then             

            ! ~~~~~~~~~~~~~
            ! Now we allocate a new matshell that applies a diagonally scaled version of 
            ! the matrix to compute the coefficients
            ! ~~~~~~~~~~~~~       
            ! Get the comm on the buffers matrix (which may be a subcomm)
            call PetscObjectGetComm(buffers%matrix, MPI_COMM_BUFFERS_MATRIX, ierr)                
      
            ! Get the sizes
            call MatGetLocalSize(buffers%matrix, local_rows, local_cols, ierr)
            call MatGetSize(buffers%matrix, global_rows, global_cols, ierr)                
            
            ! Create the matshell
            call MatCreateShell(MPI_COMM_BUFFERS_MATRIX, local_rows, local_cols, global_rows, global_cols, &
                        mat_ctx_da, mat_ctx_da%mat_scaled, ierr)
            ! The subroutine petsc_matvec_da_poly_mf applies D^-1 A
            call MatShellSetOperation(mat_ctx_da%mat_scaled, &
                        MATOP_MULT, petsc_matvec_da_poly_mf, ierr)

            call MatAssemblyBegin(mat_ctx_da%mat_scaled, MAT_FINAL_ASSEMBLY, ierr)
            call MatAssemblyEnd(mat_ctx_da%mat_scaled, MAT_FINAL_ASSEMBLY, ierr)   
            ! Have to make sure to set the type of vectors the shell creates
            call ShellSetVecType(buffers%matrix, mat_ctx_da%mat_scaled)   
            
            ! Create temporary vector we use during horner
            ! Make sure to use matrix here to get the right type (as the shell doesn't know about gpus)            
            call MatCreateVecs(buffers%matrix, PETSC_NULL_VEC, mat_ctx_da%mf_temp_vec(MF_VEC_DIAG), ierr) 
            ! Get the diagonal
            call MatGetDiagonal(buffers%matrix, mat_ctx_da%mf_temp_vec(MF_VEC_DIAG), ierr)    
            ! This is the matrix whose inverse we are applying (just copying the pointer here)
            mat_ctx_da%mat = buffers%matrix

         end if

         ! ~~~~~~~~~
         ! mat_ctx_da%mat_scaled now contains a shell that just does D^-1 A
         ! We compute the coefficients with that and then finish_approximate_inverse builds
         ! whatever matrix it needs to compute the appropriate inverse with those coeffs
         ! ~~~~~~~~~

         ! Gmres poylnomial with power basis
         if (inverse_type == PFLAREINV_POWER) then

            ! Only does one reduction in parallel
            ! Want to start the coefficient calculation asap, as we have a non-blocking
            ! all reduce in it
            call start_gmres_polynomial_coefficients_power(mat_ctx_da%mat_scaled, poly_order, &
                  buffers, coefficients(:, 1))         

         ! Gmres polynomial with arnoldi basis
         else if (inverse_type == PFLAREINV_ARNOLDI) then

            ! Does lots of reductions in parallel
            call calculate_gmres_polynomial_coefficients_arnoldi(mat_ctx_da%mat_scaled, poly_order, coefficients(:, 1))

         ! Gmres polynomial with Newton basis with extra added roots for stability - Can only use matrix-free
         else if (inverse_type == PFLAREINV_NEWTON) then

            ! Does lots of reductions in parallel
            call calculate_gmres_polynomial_roots_newton(mat_ctx_da%mat_scaled, poly_order, .TRUE., coefficients)

         ! Gmres polynomial with Newton basis without extra added roots - Can only use matrix-free
         else if (inverse_type == PFLAREINV_NEWTON_NO_EXTRA) then

            ! Does lots of reductions in parallel
            call calculate_gmres_polynomial_roots_newton(mat_ctx_da%mat_scaled, poly_order, .FALSE., coefficients)            
         end if

         ! Destroy the shell matrix
         if ((inverse_type == PFLAREINV_POWER .OR. &
             inverse_type == PFLAREINV_ARNOLDI .OR. &
             inverse_type == PFLAREINV_NEWTON .OR. &
             inverse_type == PFLAREINV_NEWTON_NO_EXTRA) .AND. &
             diag_scale_polys) then

            call VecDestroy(mat_ctx_da%mf_temp_vec(MF_VEC_DIAG), ierr)             
            call MatDestroy(mat_ctx_da%mat_scaled, ierr)
         end if
         deallocate(mat_ctx_da)           
      end if

      ! If we ended up on a subcomm, we need to comm the coefficients back to 
      ! MPI_COMM_MATRIX
      ! Both the arnoldi and newton basis have already finished their comms, so we can start an 
      ! all reduce here and conclude it in finish_approximate_inverse
      ! The power basis however does not yet have its coefficients finished, so we have to do all the 
      ! comms in finish (although it doesn't really make sense to move onto a subcomm for the power basis
      ! given it only does one reduction on MPI_COMM_MATRIX anyway)
      ! Gmres polynomial with arnoldi basis
      if (buffers%on_subcomm) then

         ! Get the comm
         call PetscObjectGetComm(matrix, MPI_COMM_MATRIX, ierr)       
         buffers%request = MPI_REQUEST_NULL

         if (inverse_type == PFLAREINV_ARNOLDI) then

            ! We know rank 0 will always have the coefficients, just broadcast them to everyone on MPI_COMM_MATRIX
            ! Some of the ranks on MPI_COMM_MATRIX will already have the coefficients, but I'm not going 
            ! to bother creating an intercommunicator to send the coefficients from any rank on the comm of buffers%matrix
            ! to the comm of ranks which aren't in MPI_COMM_MATRIX but are in buffers%matrix
            call MPI_IBcast(coefficients, size(coefficients, 1), MPI_DOUBLE, 0, &
                     MPI_COMM_MATRIX, buffers%request, errorcode)

         ! Gmres polynomial with Newton basis - Can only use matrix-free
         else if (inverse_type == PFLAREINV_NEWTON .OR. inverse_type == PFLAREINV_NEWTON_NO_EXTRA) then

            ! Have to broadcast the 2D real and imaginary roots
            call MPI_IBcast(coefficients, size(coefficients, 1) * size(coefficients, 2), MPI_DOUBLE, 0, &
                     MPI_COMM_MATRIX, buffers%request, errorcode)
         end if    
      end if  

   end subroutine start_approximate_inverse

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine finish_approximate_inverse(matrix, inverse_type, &
                  poly_order, inverse_sparsity_order, &
                  buffers, coefficients, &
                  matrix_free, diag_scale_polys, reuse_mat, reuse_submatrices, inv_matrix)

      ! Finish the assembly of an approximate inverse

      ! ~~~~~~
      type(tMat), intent(in)                            :: matrix
      integer, intent(in)                               :: inverse_type, poly_order
      integer, intent(in)                               :: inverse_sparsity_order
      type(tsqr_buffers), intent(inout)                 :: buffers      
      PetscReal, dimension(:, :), pointer, contiguous, intent(inout) :: coefficients
      logical, intent(in)                               :: matrix_free, diag_scale_polys
      type(tMat), intent(inout)                         :: reuse_mat, inv_matrix
      type(tMat), dimension(:), pointer, intent(inout)  :: reuse_submatrices

      logical :: incomplete
      PetscErrorCode :: ierr
      integer :: errorcode
#if defined(PETSC_USE_MPI_F08)
      MPIU_Status :: status
#else
      MPIU_Status, dimension(MPI_STATUS_SIZE) :: status
#endif

      ! ~~~~~~    

      ! ~~~~~~~~~~~~
      ! For any calculations started in start_approximate_inverse that happen on 
      ! a subcomm, we need to finish the broadcasts
      ! ~~~~~~~~~~~~
      if (buffers%on_subcomm .AND. buffers%request /= MPI_REQUEST_NULL) then

         ! Finish the non-blocking comms
         call mpi_wait(buffers%request, &
                        status, errorcode)
            
         if (errorcode /= MPI_SUCCESS) then
            print *, "mpi_wait failed"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)         
         end if    
         buffers%request = MPI_REQUEST_NULL           
      end if

      ! ~~~~~~~~~~~~

      ! Gmres poylnomial with power or arnoldi basis
      if (inverse_type == PFLAREINV_POWER .OR. inverse_type == PFLAREINV_ARNOLDI) then

         call build_gmres_polynomial_inverse(matrix, poly_order, &
                  buffers, coefficients(:, 1), &
                  inverse_sparsity_order, matrix_free, diag_scale_polys, &
                  reuse_mat, reuse_submatrices, inv_matrix)  

      ! Gmres polynomial with newton basis
      else if (inverse_type == PFLAREINV_NEWTON .OR. inverse_type == PFLAREINV_NEWTON_NO_EXTRA) then

         call build_gmres_polynomial_newton_inverse(matrix, poly_order, &
                           coefficients, &
                           inverse_sparsity_order, matrix_free, diag_scale_polys, reuse_mat, reuse_submatrices, &
                           inv_matrix)         

      ! Neumann polynomial
      else if (inverse_type == PFLAREINV_NEUMANN) then

         coefficients = 1d0
         call calculate_and_build_neumann_polynomial_inverse(matrix, poly_order, &
                     buffers, inverse_sparsity_order, matrix_free, reuse_mat, reuse_submatrices, inv_matrix)        
                 
      ! Sparse approximate inverse
      else if (inverse_type == PFLAREINV_SAI .OR. inverse_type == PFLAREINV_ISAI) then

         ! ~~~~~~~~~~~
         ! ~~~~~~~~~~~               

         if (inverse_type == PFLAREINV_SAI) then
            incomplete = .FALSE.
         
         ! Incomplete sparse approximate inverse
         ! This is equivalent to a one-level restricted additive schwartz
         ! where each subdomain is a single unknown with the overlap 
         ! given by neighbouring unknowns            
         else
            incomplete = .TRUE.
         end if

         call calculate_and_build_sai(matrix, inverse_sparsity_order, incomplete, &
                  reuse_mat, reuse_submatrices, inv_matrix)
         
      ! Weighted jacobi
      else if (inverse_type == PFLAREINV_WJACOBI) then

         call calculate_and_build_weighted_jacobi_inverse(matrix, .TRUE., inv_matrix)          

      ! Unweighted jacobi
      else if (inverse_type == PFLAREINV_JACOBI) then

         call calculate_and_build_weighted_jacobi_inverse(matrix, .FALSE., inv_matrix)           

      end if

      ! If we were on a subcomm we created a copy of our matrix, if not we incremented 
      ! the reference counter
      call MatDestroy(buffers%matrix, ierr)            

   end subroutine finish_approximate_inverse
   
   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine reset_inverse_mat(matrix)

      ! Resets the matrix

      ! ~~~~~~
      type(tMat), intent(inout) :: matrix

      PetscErrorCode :: ierr
      MatType:: mat_type
      type(mat_ctxtype), pointer :: mat_ctx=>null(), mat_ctx_scaled=>null()
      ! ~~~~~~

      if (.NOT. PetscObjectIsNull(matrix)) then
         call MatGetType(matrix, mat_type, ierr)
         ! If its a matshell, make sure to delete its ctx
         if (mat_type==MATSHELL) then
            call MatShellGetContext(matrix, mat_ctx, ierr)
            if (mat_ctx%own_coefficients) then
               deallocate(mat_ctx%coefficients)
               mat_ctx%coefficients => null()
            end if
            call VecDestroy(mat_ctx%mf_temp_vec(MF_VEC_TEMP), ierr)

            ! Both newton and neumann polynomials use some extra temporary vectors
            if (.NOT. PetscObjectIsNull(mat_ctx%mat_scaled) .OR. &
                     associated(mat_ctx%real_roots)) then
               
               call VecDestroy(mat_ctx%mf_temp_vec(MF_VEC_RHS), ierr)
               call VecDestroy(mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)
               call VecDestroy(mat_ctx%mf_temp_vec(MF_VEC_TEMP_TWO), ierr)
               call VecDestroy(mat_ctx%mf_temp_vec(MF_VEC_TEMP_THREE), ierr)
            end if

            ! Neumann polynomial has extra context that needs deleting
            if (.NOT. PetscObjectIsNull(mat_ctx%mat_scaled)) then
               call MatShellGetContext(mat_ctx%mat_scaled, mat_ctx_scaled, ierr)
               deallocate(mat_ctx_scaled)
               call MatDestroy(mat_ctx%mat_scaled, ierr)
            end if
            deallocate(mat_ctx)
         end if               
         call MatDestroy(matrix, ierr)       
      end if  
      
   end subroutine reset_inverse_mat

   ! -------------------------------------------------------------------------------------------------------------------------------

end module approx_inverse_setup

