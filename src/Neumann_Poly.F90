module neumann_poly

   use petscmat
   use gmres_poly
   use matshell_pflare
   use c_petsc_interfaces

#include "petsc/finclude/petscmat.h"

   implicit none

#include "petsc_legacy.h"   

   public

   PetscEnum, parameter :: PFLAREINV_NEUMANN=4
   
   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine petsc_matvec_ida_neumann_poly_mf(mat, x, y)

      ! Applies I-D^-1 A matrix-free
      ! y = A x

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! Input
      type(tMat), intent(in)    :: mat
      type(tVec) :: x
      type(tVec) :: y

      ! Local
      PetscErrorCode :: ierr
      type(mat_ctxtype), pointer :: mat_ctx_ida => null()

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      call MatShellGetContext(mat, mat_ctx_ida, ierr)

      ! ~~~~~~~~~~~~
      ! We want to apply (I-D^-1 A) x
      ! ~~~~~~~~~~~~

      ! Multiply by A
      call MatMult(mat_ctx_ida%mat, x, y, ierr)

      ! Doing D^-1 on the result
      call VecPointwiseDivide(y, y, mat_ctx_ida%mf_temp_vec(MF_VEC_DIAG), ierr)

      ! Now do x - D^-1 A x
      call VecAXPBY(y, &
               1d0, &
               -1d0, &
               x, ierr)      

   end subroutine petsc_matvec_ida_neumann_poly_mf     


! -------------------------------------------------------------------------------------------------------------------------------

   subroutine petsc_matvec_neumann_poly_mf(mat, x, y)

      ! Applies a Neumann polynomial matrix-free, q(I-D^-1 A) D^-1, as an inverse
      ! y = A x

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! Input
      type(tMat), intent(in)    :: mat
      type(tVec) :: x
      type(tVec) :: y

      ! Local
      integer :: errorcode
      PetscErrorCode :: ierr      
      type(mat_ctxtype), pointer :: mat_ctx => null()

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      call MatShellGetContext(mat, mat_ctx, ierr)
      if (.NOT. associated(mat_ctx%coefficients)) then
         print *, "Polynomial coefficients in context aren't found"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! ~~~~~~~~~~~~
      ! We want to apply q(I-D^-1 A) D^-1 with coefficients=1
      ! ~~~~~~~~~~~~

      ! Doing rhs_copy = D^-1 x 
      call VecPointwiseDivide(mat_ctx%mf_temp_vec(MF_VEC_RHS), x, &
               mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)

      ! and now we call the horner method to apply our polynomial
      ! q(I - D^-1 A) to rhs_copy (D^-1 x)
      call petsc_horner(mat_ctx%mat_ida, mat_ctx%coefficients, mat_ctx%mf_temp_vec(MF_VEC_TEMP), &
                  mat_ctx%mf_temp_vec(MF_VEC_RHS), y)      

   end subroutine petsc_matvec_neumann_poly_mf      


! -------------------------------------------------------------------------------------------------------------------------------

   subroutine calculate_and_build_neumann_polynomial_inverse(matrix, poly_order, &
                  buffers, poly_sparsity_order, matrix_free, reuse_mat, inv_matrix)


      ! Builds an assembled neumann polynomial approximate inverse
      ! If poly_sparsity_order < poly_order it will build a fixed sparsity approximation
      ! Scales by the diagonal and applies the scaled diagonal to the rhs of the output
      ! so it can be used

      ! ~~~~~~
      type(tMat), target, intent(in)      :: matrix
      integer, intent(in)                 :: poly_order
      type(tsqr_buffers), intent(inout)   :: buffers
      integer, intent(in)                 :: poly_sparsity_order
      logical, intent(in)                 :: matrix_free
      type(tMat), intent(inout)           :: reuse_mat, inv_matrix

      ! Local variables
      PetscReal, dimension(:), pointer :: coefficients
      PetscReal, dimension(poly_order + 1), target :: coefficients_stack
      integer :: comm_size, errorcode
      PetscErrorCode :: ierr      
      MPI_Comm :: MPI_COMM_MATRIX
      PetscInt :: local_rows, local_cols, global_rows, global_cols
      type(tMat) :: temp_mat
      type(tVec) :: rhs_copy, diag_vec
      type(mat_ctxtype), pointer :: mat_ctx, mat_ctx_ida

      ! ~~~~~~    

      ! Have to allocate heap memory if matrix-free as the context in inv_matrix
      ! just points at the coefficients
      if (matrix_free) then

         ! We might want to call the gmres poly creation on a sub communicator
         ! so let's get the comm attached to the matrix and make sure to use that 
         call PetscObjectGetComm(matrix, MPI_COMM_MATRIX, ierr)    
         ! Get the comm size 
         call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)        
   
         ! Get the local sizes
         call MatGetLocalSize(matrix, local_rows, local_cols, ierr)
         call MatGetSize(matrix, global_rows, global_cols, ierr)      
         
         ! If not re-using
         if (PetscMatIsNull(inv_matrix)) then

            allocate(coefficients(poly_order + 1))
            coefficients = 1d0 

            ! Have to dynamically allocate this
            allocate(mat_ctx)
            ! A Neumann polynomial has coefficients of 1
            mat_ctx%coefficients => coefficients                      

            ! Create the matshell
            call MatCreateShell(MPI_COMM_MATRIX, local_rows, local_cols, global_rows, global_cols, &
                        mat_ctx, inv_matrix, ierr)
            ! The subroutine petsc_matvec_neumann_poly_mf applies the neumann polynomial inverse
            call MatShellSetOperation(inv_matrix, &
                        MATOP_MULT, petsc_matvec_neumann_poly_mf, ierr)

            call MatAssemblyBegin(inv_matrix, MAT_FINAL_ASSEMBLY, ierr)
            call MatAssemblyEnd(inv_matrix, MAT_FINAL_ASSEMBLY, ierr)
            ! Have to make sure to set the type of vectors the shell creates
            call ShellSetVecType(matrix, inv_matrix)              
            
            ! Create temporary vector we use during horner
            ! Make sure to use matrix here to get the right type (as the shell doesn't know about gpus)            
            call MatCreateVecs(matrix, mat_ctx%mf_temp_vec(MF_VEC_TEMP), PETSC_NULL_VEC, ierr) 

            ! ~~~~~~~~~~~~~
            ! Now we allocate a new matshell that applies a diagonally scaled version of 
            ! the matrix minus I
            ! ~~~~~~~~~~~~~

            ! Have to dynamically allocate this
            allocate(mat_ctx_ida)
            mat_ctx_ida%coefficients => coefficients
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
            mat_ctx_ida%mat_ida = PETSC_NULL_MAT
#endif                        
            
            ! Create the matshell
            call MatCreateShell(MPI_COMM_MATRIX, local_rows, local_cols, global_rows, global_cols, &
                        mat_ctx_ida, mat_ctx%mat_ida, ierr)
            ! The subroutine petsc_matvec_ida_neumann_poly_mf applies I - D^-1 A
            call MatShellSetOperation(mat_ctx%mat_ida, &
                        MATOP_MULT, petsc_matvec_ida_neumann_poly_mf, ierr)

            call MatAssemblyBegin(mat_ctx%mat_ida, MAT_FINAL_ASSEMBLY, ierr)
            call MatAssemblyEnd(mat_ctx%mat_ida, MAT_FINAL_ASSEMBLY, ierr)   
            ! Have to make sure to set the type of vectors the shell creates
            call ShellSetVecType(matrix, mat_ctx%mat_ida)   
            
            ! Create temporary vector we use during horner
            ! Make sure to use matrix here to get the right type (as the shell doesn't know about gpus)            
            call MatCreateVecs(matrix, mat_ctx%mf_temp_vec(MF_VEC_RHS), mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)       

         ! Reusing 
         else
            call MatShellGetContext(inv_matrix, mat_ctx, ierr)
            call MatShellGetContext(mat_ctx%mat_ida, mat_ctx_ida, ierr)

         end if

         ! This is the matrix whose inverse we are applying (just copying the pointer here)
         mat_ctx%mat = matrix 
         mat_ctx_ida%mat = matrix 

         ! Get the diagonal
         call MatGetDiagonal(matrix, mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)    
         mat_ctx_ida%mf_temp_vec(MF_VEC_DIAG) = mat_ctx%mf_temp_vec(MF_VEC_DIAG)

      ! If not matrix free
      else

         coefficients => coefficients_stack
         ! A Neumann polynomial has coefficients of 1
         coefficients = 1d0 

         ! Need to build an assembled I - D^-1 A
         call MatConvert(matrix, MATSAME, MAT_INITIAL_MATRIX, temp_mat, ierr)
         call MatCreateVecs(matrix, rhs_copy, diag_vec, ierr)
         call MatGetDiagonal(matrix, diag_vec, ierr)
         call VecReciprocal(diag_vec, ierr)
         call MatDiagonalScale(temp_mat, diag_vec, PETSC_NULL_VEC, ierr) 
   
         ! Computes: I - D^-1 A
         call MatScale(temp_mat, -1d0, ierr)
         call MatShift(temp_mat, 1d0, ierr)
         
         ! If we feed in coefficients=1 and leave buffers%R_buffer_receive unallocated as it will just skip 
         ! the "gmres" coefficient calculation and just calculate the sum of (potentailly fixed sparsity) matrix powers
         call build_gmres_polynomial_inverse(temp_mat, poly_order, buffers, coefficients, &
               poly_sparsity_order, .FALSE., reuse_mat, inv_matrix)
               
         ! Now this computes (I - D^-1 A)^-1 D^-1
         ! For the F-point smoothing and grid-transfer operators in our air multigrid, 
         ! this is equivalent to using (I - Dff^-1 Aff)^-1 Dff^-1 everywhere we normally use 
         ! Aff^-1
         call MatDiagonalScale(inv_matrix, PETSC_NULL_VEC, diag_vec, ierr) 
         
         ! Cleanup
         call VecDestroy(rhs_copy, ierr)
         call VecDestroy(diag_vec, ierr)  
         call MatDestroy(temp_mat, ierr)

      end if      

   end subroutine calculate_and_build_neumann_polynomial_inverse


! -------------------------------------------------------------------------------------------------------------------------------

end module neumann_poly

