module approx_inverse_setup

   use petscmat
   use gmres_poly
   use gmres_poly_newton
   use neumann_poly
   use weighted_jacobi
   use sai_z
   use repartition
   use matshell_pflare

#include "petsc/finclude/petscmat.h"
      
   implicit none
   public

   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine calculate_and_build_approximate_inverse(matrix, inverse_type, &
                  poly_order, inverse_sparsity_order, &
                  matrix_free, subcomm, &
                  inv_matrix)

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

      ! ~~~~~~
      type(tMat), intent(in)                            :: matrix
      integer, intent(in)                               :: inverse_type, poly_order
      integer, intent(in)                               :: inverse_sparsity_order
      logical, intent(in)                               :: matrix_free, subcomm
      type(tMat), intent(inout)                         :: inv_matrix

      type(tsqr_buffers)                           :: buffers 
      ! This pointer must be declared contiguous, as we require coefficients
      ! to be contiguous in build_gmres_polynomial_newton_inverse as 
      ! we muck about with the rank and if this pointer is not declared
      ! contiguous it creates a temporary copy which then disappears and
      ! we segfault when trying to apply mf
      PetscReal, dimension(:, :), contiguous, pointer   :: coefficients
      PetscReal, dimension(poly_order + 1, 1), target   :: coefficients_stack
      type(mat_ctxtype), pointer :: mat_ctx => null()
      PetscErrorCode :: ierr
      type(tMat) :: reuse_mat, inv_matrix_temp

      ! ~~~~~~  
      
      ! Have to allocate heap memory if matrix-free as the context in inv_matrix
      ! just points at the coefficients
      if (matrix_free) then
         if (inverse_type == PFLAREINV_NEWTON .OR. inverse_type == PFLAREINV_NEWTON_NO_EXTRA) then
            ! Newton basis needs storage for real and imaginary roots
            allocate(coefficients(poly_order + 1, 2))
         else 
            allocate(coefficients(poly_order + 1, 1))
         end if
      else
         coefficients => coefficients_stack
      end if

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
      ! Don't do any re-use, if you want reuse call the start/finish yourself   

      ! Start the calculation
      call start_approximate_inverse(matrix, inverse_type, &
                  poly_order, &
                  buffers, coefficients)
      ! Finish it
      call finish_approximate_inverse(matrix, inverse_type, &
                  poly_order, inverse_sparsity_order, &
                  buffers, coefficients, &
                  matrix_free, &
                  reuse_mat, &
                  inv_matrix_temp)
            
      ! Have to tell the matshell to own the coefficient pointers
      if (matrix_free) then
         call MatShellGetContext(inv_matrix_temp, mat_ctx, ierr)
         mat_ctx%own_coefficients = .TRUE.               
      end if
      if (.NOT. PetscObjectIsNull(reuse_mat)) then
         call MatDestroy(reuse_mat, ierr)
      end if

      ! Copy the pointer - see comment above about petsc 3.22
      inv_matrix = inv_matrix_temp

   end subroutine calculate_and_build_approximate_inverse    

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine start_approximate_inverse(matrix, inverse_type, poly_order, &
                  buffers, coefficients)

      ! Starts the assembly of an approximate inverse
      ! We have different types of inverses we can use
      ! Have to call finish_approximate_inverse before it can be used

      ! ~~~~~~
      type(tMat), intent(in)                            :: matrix
      integer, intent(in)                               :: inverse_type, poly_order
      type(tsqr_buffers), intent(inout)                 :: buffers 
      PetscReal, dimension(:, :), pointer, intent(inout)     :: coefficients

      PetscErrorCode :: ierr
      integer :: MPI_COMM_MATRIX, errorcode
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

         ! Gmres poylnomial with power basis
         if (inverse_type == PFLAREINV_POWER) then

            ! Only does one reduction in parallel
            ! Want to start the coefficient calculation asap, as we have a non-blocking
            ! all reduce in it
            call start_gmres_polynomial_coefficients_power(buffers%matrix, poly_order, &
                  buffers, coefficients(:, 1))         

         ! Gmres polynomial with arnoldi basis
         else if (inverse_type == PFLAREINV_ARNOLDI) then

            ! Does lots of reductions in parallel
            call calculate_gmres_polynomial_coefficients_arnoldi(buffers%matrix, poly_order, coefficients(:, 1))

         ! Gmres polynomial with Newton basis with extra added roots for stability - Can only use matrix-free
         else if (inverse_type == PFLAREINV_NEWTON) then

            ! Does lots of reductions in parallel
            call calculate_gmres_polynomial_roots_newton(buffers%matrix, poly_order, .TRUE., coefficients)

         ! Gmres polynomial with Newton basis without extra added roots - Can only use matrix-free
         else if (inverse_type == PFLAREINV_NEWTON_NO_EXTRA) then

            ! Does lots of reductions in parallel
            call calculate_gmres_polynomial_roots_newton(buffers%matrix, poly_order, .FALSE., coefficients)            
         end if
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
                  matrix_free, reuse_mat, inv_matrix)

      ! Finish the assembly of an approximate inverse

      ! ~~~~~~
      type(tMat), intent(in)                            :: matrix
      integer, intent(in)                               :: inverse_type, poly_order
      integer, intent(in)                               :: inverse_sparsity_order
      type(tsqr_buffers), intent(inout)                 :: buffers      
      PetscReal, dimension(:, :), pointer, contiguous, intent(inout) :: coefficients
      logical, intent(in)                               :: matrix_free
      type(tMat), intent(inout)                         :: reuse_mat, inv_matrix

      logical :: incomplete
      PetscErrorCode :: ierr
      integer :: errorcode
      integer, dimension(MPI_STATUS_SIZE) :: status

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
                  inverse_sparsity_order, matrix_free, reuse_mat, inv_matrix)  

      ! Gmres polynomial with newton basis
      else if (inverse_type == PFLAREINV_NEWTON .OR. inverse_type == PFLAREINV_NEWTON_NO_EXTRA) then

         if (.NOT. matrix_free) then
            print *, "GMRES polynomial with Newton basis must be applied matrix-free"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
         end if

         call build_gmres_polynomial_newton_inverse(matrix, poly_order, &
                           coefficients, &
                           inv_matrix)         

      ! Neumann polynomial
      else if (inverse_type == PFLAREINV_NEUMANN) then

         call calculate_and_build_neumann_polynomial_inverse(matrix, poly_order, &
                     buffers, inverse_sparsity_order, matrix_free, reuse_mat, inv_matrix)        
                 
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

         call calculate_and_build_sai(matrix, inverse_sparsity_order, incomplete, reuse_mat, inv_matrix)
         
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
      type(mat_ctxtype), pointer :: mat_ctx, mat_ctx_ida
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
            if (.NOT. PetscObjectIsNull(mat_ctx%mat_ida) .OR. &
                     associated(mat_ctx%real_roots)) then
               
               call VecDestroy(mat_ctx%mf_temp_vec(MF_VEC_RHS), ierr)
               call VecDestroy(mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)
            end if

            ! Neumann polynomial has extra context that needs deleting
            if (.NOT. PetscObjectIsNull(mat_ctx%mat_ida)) then
               call MatShellGetContext(mat_ctx%mat_ida, mat_ctx_ida, ierr)
               deallocate(mat_ctx_ida)
               call MatDestroy(mat_ctx%mat_ida, ierr)

            end if
            deallocate(mat_ctx)
         end if               
         call MatDestroy(matrix, ierr)       
      end if  
      
   end subroutine reset_inverse_mat

   ! -------------------------------------------------------------------------------------------------------------------------------

end module approx_inverse_setup

