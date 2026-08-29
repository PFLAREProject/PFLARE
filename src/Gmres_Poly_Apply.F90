module gmres_poly_apply

   use petscmat
   use matshell_data_type, only: mat_ctxtype, ensure_block_temp_mats, ensure_block_kernel_products
   use pflare_parameters, only: PFLARE_TOL_ZERO, PFLARE_ZERO, PFLARE_ONE, PFLARE_MINUS_ONE, PFLARE_TWO, &
         MF_VEC_TEMP, MF_VEC_TEMP_TWO, MF_VEC_TEMP_THREE, MF_VEC_DIAG, MF_VEC_RHS, &
         MF_MAT_TEMP, MF_MAT_TEMP_TWO, MF_MAT_TEMP_THREE, MF_MAT_RHS

#include "petsc/finclude/petscmat.h"

   implicit none

   public

   ! The apply side of the matrix-free gmres polynomial inverses - everything
   ! that runs when a polynomial matshell built by gmres_poly/gmres_poly_newton
   ! is applied: the scalar matvec callbacks the matshells register with
   ! MatShellSetOperation, the horner/newton kernels they call, their multiple
   ! rhs (block) twins, and the block dispatch shell_poly_block_apply
   ! The construction of the polynomials stays in gmres_poly/gmres_poly_newton

   contains

   subroutine petsc_matvec_da_poly_mf(mat, x, y)

      ! Applies D^-1 A as a shell
      ! y = A x

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! Input
      type(tMat), intent(in)    :: mat
      type(tVec) :: x
      type(tVec) :: y

      ! Local
      PetscErrorCode :: ierr
      type(mat_ctxtype), pointer :: mat_ctx => null()

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      call MatShellGetContext(mat, mat_ctx, ierr)

      ! ~~~~~~~~~~~~
      ! We want to apply (D^-1 A) x
      ! ~~~~~~~~~~~~

      ! Multiply by A
      call MatMult(mat_ctx%mat, x, y, ierr)

      ! Doing D^-1 on the result
      call VecPointwiseDivide(y, y, mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)    

   end subroutine petsc_matvec_da_poly_mf   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine petsc_matvec_poly_mf(mat, x, y)

      ! Applies a matrix polynomial matrix-free as an inverse
      ! Just uses a Horner iteration to apply the mat_ctx%coefficients
      ! to mat_ctx%mat in the input matshell
      ! y = A x

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! Input
      type(tMat), intent(in)    :: mat
      type(tVec) :: x
      type(tVec) :: y

      ! Local
      PetscErrorCode :: ierr
      integer :: errorcode
      type(mat_ctxtype), pointer :: mat_ctx => null()

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      call MatShellGetContext(mat, mat_ctx, ierr)
      if (.NOT. associated(mat_ctx%coefficients)) then
         print *, "Polynomial coefficients in context aren't found"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! Call the Horner iteration
      call petsc_horner(mat_ctx%mat, mat_ctx%coefficients, mat_ctx%mf_temp_vec(MF_VEC_TEMP), x, y)

   end subroutine petsc_matvec_poly_mf      

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine petsc_matvec_right_scale_poly_mf(mat, x, y)

      ! Applies a polynomial matrix-free with a right diagonal scaling added
      ! q(mat_ctx%mat_scaled) D^-1, as an inverse
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
      ! We want to apply q(mat_ctx%mat_scaled) D^-1
      ! ~~~~~~~~~~~~

      ! Doing rhs_copy = D^-1 x 
      call VecPointwiseDivide(mat_ctx%mf_temp_vec(MF_VEC_RHS), x, &
               mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)            

      ! and now we call the horner method to apply our polynomial
      ! q(mat_ctx%mat_scaled) to rhs_copy (D^-1 x)
      call petsc_horner(mat_ctx%mat_scaled, mat_ctx%coefficients, mat_ctx%mf_temp_vec(MF_VEC_TEMP), &
                  mat_ctx%mf_temp_vec(MF_VEC_RHS), y)      

   end subroutine petsc_matvec_right_scale_poly_mf      

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine petsc_horner(mat, coefficients, temp_vec, x, y)

      ! Uses a horner iteration to apply
      ! y = (coeff(1) + coeff(2) * A + coeff(3) * A^2 + ...) x

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! Input
      type(tMat), intent(in)    :: mat
      PetscReal, dimension(:)   :: coefficients
      type(tVec)                :: x, temp_vec
      type(tVec)                :: y

      ! Local
      integer :: order
      PetscErrorCode :: ierr      

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! ~~~~~~~
      ! This applies the polynomial of order n as:
      ! xn - x0 = q_n-1(A) r_0 
      ! where 
      ! q_n-1(z) = alpha_0 + alpha_1 * z + alpha_2 * z^2 + ...
      ! So the q_n-1(A) r_0 term is just the repeated application of 
      ! y = alpha_n-1 r_0
      ! do i = 1, n-1
      !   y = A * y + alpha_n-i-1 r_0
      ! where the output y is xn - x0.
      ! Practically, if we choose ksprichardson and then use this as a preconditioner B, 
      ! we are doing (where these n are the iteration count, not the order of the polynomial we're 
      ! applying above)
      ! x^n+1 = x^n + B * r^n
      ! so the x passed in should be the residual r^n, and we don't need to add x^n to
      ! the solution, as the richardson is doing that for us. We have to ensure the richardson scale is one though.
      ! ~~~~~~~

      ! Let's do the first y = alpha_n-1 r_0 (ie the highest order term first)
      call VecAXPBY(y, &
               coefficients(size(coefficients)), &
               PFLARE_ZERO, &
               x, ierr)

      ! If we are doing a first order polynomial or above, we have to do an extra matvec per order
      if (size(coefficients, 1) > 1) then     

         ! Loop down from the second highest order term down to the constant
         do order = size(coefficients, 1)-1, 1, -1

            ! Skip this coefficient if zero
            if (coefficients(order) == 0d0) cycle

            ! Copy y into temp_vec
            call VecCopy(y, temp_vec, ierr)             

            ! Now do y = A * temp_vec
            call MatMult(mat, temp_vec, y, ierr)

            ! Compute y = A * temp_vec + alpha_n-i-1 r_0
            call VecAXPBY(y, &
                     coefficients(order), &
                     PFLARE_ONE, &
                     x, ierr)
         end do
      end if

   end subroutine petsc_horner

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine petsc_horner_block(mat_ctx, x_mat, y_mat, recip_diag, &
                  neumann_inner, block_applied)

      ! Uses a horner iteration to apply
      ! y_mat = (coeff(1) + coeff(2) * B + coeff(3) * B^2 + ...) x_mat
      ! for a block of right hand sides, x_mat, ie the multiple rhs version of petsc_horner
      ! The matvecs of petsc_horner become sparse matrix-dense matrix products (SpMM)

      ! The iteration ping-pongs between the two dense temporaries in the context,
      ! so the products always target our own scratch (never the caller's y_mat)
      ! and can stay attached between applies - only the numeric phase runs each
      ! order, and the result is copied into y_mat once at the end

      ! There are three different inner operators, B, we can apply, all of them built
      ! out of real products with the *unscaled* mat in the context. We do this rather
      ! than running the products on the matshells the scalar versions use, as every
      ! product on a shell degrades to a column by column matvec
      ! 1) recip_diag null                : B = A, the plain polynomial q(A)
      ! 2) recip_diag non-null            : B = D^-1 A, the diagonally scaled polynomial
      !                                     q(D^-1 A), with recip_diag = D^-1
      ! 3) neumann_inner (implies scaled) : B = I - D^-1 A, the Neumann polynomial
      ! In the scaled cases the caller must have already scaled the block of rhs,
      ! ie x_mat = D^-1 X

      ! block_applied comes back false (with y_mat untouched) if the block of rhs we've
      ! been given has no product with mat, so the caller can fall back to a column by
      ! column apply

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! Input
      type(mat_ctxtype), intent(inout) :: mat_ctx
      type(tMat)                       :: x_mat
      type(tMat)                       :: y_mat
      type(tVec)                       :: recip_diag
      logical, intent(in)              :: neumann_inner
      logical, intent(out)             :: block_applied

      ! Local
      integer :: order
      logical :: scaled
      type(tMat) :: cur_mat, other_mat, swap_mat
      PetscErrorCode :: ierr

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      block_applied = .FALSE.
      scaled = .NOT. PetscObjectIsNull(recip_diag)

      ! A zeroth order polynomial has no products, just the scaled copy
      if (size(mat_ctx%coefficients, 1) == 1) then
         call MatCopy(x_mat, y_mat, SAME_NONZERO_PATTERN, ierr)
         call MatScale(y_mat, mat_ctx%coefficients(1), ierr)
         block_applied = .TRUE.
         return
      end if

      ! For a first order polynomial or above we do an extra product per order,
      ! alternating between the two temporaries - make sure both products are
      ! attached (they stay attached between applies)
      call ensure_block_kernel_products(mat_ctx, MF_MAT_TEMP, block_applied)
      if (.NOT. block_applied) return
      block_applied = .FALSE.

      ! Let's do the first cur = alpha_n-1 r_0 (ie the highest order term first)
      cur_mat = mat_ctx%mf_temp_mat(MF_MAT_TEMP)
      other_mat = mat_ctx%mf_temp_mat(MF_MAT_TEMP_TWO)
      call MatCopy(x_mat, cur_mat, SAME_NONZERO_PATTERN, ierr)
      call MatScale(cur_mat, mat_ctx%coefficients(size(mat_ctx%coefficients)), ierr)

      ! Loop down from the second highest order term down to the constant
      do order = size(mat_ctx%coefficients, 1)-1, 1, -1

         ! Skip this coefficient if zero
         if (mat_ctx%coefficients(order) == 0d0) cycle

         ! other = A * cur - each temporary's attached product reads the other
         ! temporary, so which product runs is decided by which one is cur
         call MatProductNumeric(other_mat, ierr)

         ! This is the arithmetic of the D^-1 A matshell, done blockwise
         if (scaled) call MatDiagonalScale(other_mat, recip_diag, PETSC_NULL_VEC, ierr)

         ! The inner operator is I - D^-1 A rather than D^-1 A, so finish
         ! other = cur - D^-1 A cur
         ! cur still holds the value we did the product with
         ! This is the arithmetic of the I - D^-1 A matshell, done blockwise
         if (neumann_inner) then
            call MatScale(other_mat, PFLARE_MINUS_ONE, ierr)
            call MatAXPY(other_mat, PFLARE_ONE, cur_mat, SAME_NONZERO_PATTERN, ierr)
         end if

         ! Compute other = B * cur + alpha_n-i-1 r_0
         call MatAXPY(other_mat, mat_ctx%coefficients(order), x_mat, SAME_NONZERO_PATTERN, ierr)

         ! The result of this order is the input of the next
         swap_mat = cur_mat
         cur_mat = other_mat
         other_mat = swap_mat
      end do

      call MatCopy(cur_mat, y_mat, SAME_NONZERO_PATTERN, ierr)

      block_applied = .TRUE.

   end subroutine petsc_horner_block

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine petsc_matvec_poly_newton_mf(mat, x, y)

      ! Applies a matrix polynomial matrix-free as an inverse
      ! y = A x

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! Input
      type(tMat), intent(in)    :: mat
      type(tVec) :: x
      type(tVec) :: y

      ! Local
      PetscErrorCode :: ierr
      integer :: errorcode
      type(mat_ctxtype), pointer :: mat_ctx => null()

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      call MatShellGetContext(mat, mat_ctx, ierr)
      if (.NOT. associated(mat_ctx%real_roots)) then
         print *, "Polynomial coefficients in context aren't found"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! Apply polynomial
      call petsc_newton(mat_ctx%mat, &
               mat_ctx%real_roots, mat_ctx%imag_roots, &
               mat_ctx%mf_temp_vec(MF_VEC_TEMP), &
               mat_ctx%mf_temp_vec(MF_VEC_TEMP_TWO), &
               mat_ctx%mf_temp_vec(MF_VEC_TEMP_THREE), &               
               x, y) 

   end subroutine petsc_matvec_poly_newton_mf     

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine petsc_matvec_right_scale_poly_newton_mf(mat, x, y)

      ! Applies a Newton polynomial matrix-free with a right diagonal scaling added
      ! q(mat_ctx%mat_scaled) D^-1, as an inverse
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
      if (.NOT. associated(mat_ctx%real_roots)) then
         print *, "Polynomial coefficients in context aren't found"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! ~~~~~~~~~~~~
      ! We want to apply q(mat_ctx%mat_scaled) D^-1
      ! ~~~~~~~~~~~~

      ! Doing MF_VEC_RHS = D^-1 x 
      call VecPointwiseDivide(mat_ctx%mf_temp_vec(MF_VEC_RHS), x, &
               mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)            

      ! and now we apply our polynomial
      ! q(mat_ctx%mat_scaled) to MF_VEC_RHS (D^-1 x)
      call petsc_newton(mat_ctx%mat_scaled, &
               mat_ctx%real_roots, mat_ctx%imag_roots, &
               mat_ctx%mf_temp_vec(MF_VEC_TEMP), &
               mat_ctx%mf_temp_vec(MF_VEC_TEMP_TWO), &
               mat_ctx%mf_temp_vec(MF_VEC_TEMP_THREE), &
               mat_ctx%mf_temp_vec(MF_VEC_RHS), y)      

   end subroutine petsc_matvec_right_scale_poly_newton_mf    

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine petsc_matvec_poly_newton_residual_mf(mat, x, y)

      ! Applies a matrix polynomial matrix-free as an inverse and computes the residual
      ! y = A x

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! Input
      type(tMat), intent(in)    :: mat
      type(tVec) :: x
      type(tVec) :: y

      ! Local
      PetscErrorCode :: ierr
      integer :: errorcode
      type(mat_ctxtype), pointer :: mat_ctx => null()

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      call MatShellGetContext(mat, mat_ctx, ierr)
      if (.NOT. associated(mat_ctx%real_roots)) then
         print *, "Polynomial coefficients in context aren't found"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! Apply polynomial
      call petsc_newton_residual(mat_ctx%mat, &
               mat_ctx%real_roots, mat_ctx%imag_roots, &
               mat_ctx%mf_temp_vec(MF_VEC_TEMP), &
               mat_ctx%mf_temp_vec(MF_VEC_TEMP_TWO), &
               mat_ctx%mf_temp_vec(MF_VEC_TEMP_THREE), &               
               x, y) 

   end subroutine petsc_matvec_poly_newton_residual_mf   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine petsc_matvec_right_scale_poly_newton_residual_mf(mat, x, y)

      ! Applies a Newton polynomial matrix-free with a right diagonal scaling added
      ! q(mat_ctx%mat_scaled) D^-1, as an inverse and computes the residual
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
      if (.NOT. associated(mat_ctx%real_roots)) then
         print *, "Polynomial coefficients in context aren't found"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! ~~~~~~~~~~~~
      ! We want to apply q(mat_ctx%mat_scaled) D^-1 and then compute a residual
      ! ~~~~~~~~~~~~

      ! Doing MF_VEC_RHS = D^-1 x 
      call VecPointwiseDivide(mat_ctx%mf_temp_vec(MF_VEC_RHS), x, &
               mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)            

      ! and now we apply our polynomial
      ! q(mat_ctx%mat_scaled) to MF_VEC_RHS (D^-1 x)
      call petsc_newton_residual(mat_ctx%mat_scaled, &
               mat_ctx%real_roots, mat_ctx%imag_roots, &
               mat_ctx%mf_temp_vec(MF_VEC_TEMP), &
               mat_ctx%mf_temp_vec(MF_VEC_TEMP_TWO), &
               mat_ctx%mf_temp_vec(MF_VEC_TEMP_THREE), &
               mat_ctx%mf_temp_vec(MF_VEC_RHS), y)   
               
      call VecPointwiseMult(y, y, mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)                   

   end subroutine petsc_matvec_right_scale_poly_newton_residual_mf     

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine petsc_newton(mat, real_roots, imag_roots, temp_vec, temp_vec_two, temp_vec_three, x, y)

      ! Applies a gmres polynomial in the newton basis matrix-free as an inverse
      ! The roots are stored in real_roots, imag_roots in the input matshell
      ! Based on Loe 2021 Toward efficient polynomial preconditioning for GMRES
      ! This is Algorithm 3 in Loe
      ! y = A x

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! Input
      type(tMat), intent(in)              :: mat
      PetscReal, dimension(:), intent(in) :: real_roots, imag_roots
      type(tVec), intent(in)              :: temp_vec, temp_vec_two, temp_vec_three
      type(tVec)                          :: x
      type(tVec)                          :: y

      ! Local
      integer :: i
      PetscErrorCode :: ierr      

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! temp_vec = x
      call VecCopy(x, temp_vec, ierr)
      ! y = 0
      call VecSet(y, PFLARE_ZERO, ierr)

      ! ~~~~~~~~~~~~
      ! Iterate over the i
      ! ~~~~~~~~~~~~
      i = 1
      do while (i .le. size(real_roots) - 1)

         ! If real this is easy
         if (imag_roots(i) == 0d0) then

            ! Skips eigenvalues that are numerically zero - see 
            ! the comment in calculate_gmres_polynomial_roots_newton 
            if (abs(real_roots(i)) < PFLARE_TOL_ZERO) then
               i = i + 1
               cycle
            end if

            ! y = y + theta_i * temp_vec
            call VecAXPY(y, &
                     PFLARE_ONE/real_roots(i), &
                     temp_vec, ierr)   
                                          
            ! temp_vec_two = A * temp_vec
            call MatMult(mat, temp_vec, temp_vec_two, ierr)
            ! temp_vec = temp_vec - theta_i * temp_vec_two
            call VecAXPY(temp_vec, &
                     PFLARE_MINUS_ONE/real_roots(i), &
                     temp_vec_two, ierr) 

            i = i + 1

         ! If imaginary, then have to combine the e'val and its
         ! complex conjugate to keep the arithmetic real
         ! Relies on the complex conjugate being next to each other
         else

            ! Skips eigenvalues that are numerically zero
            if (real_roots(i)**2 + imag_roots(i)**2 < PFLARE_TOL_ZERO) then
               i = i + 2
               cycle
            end if            

            ! temp_vec_two = A * temp_vec
            call MatMult(mat, temp_vec, temp_vec_two, ierr)    
            ! temp_vec_two = 2 * Re(theta_i) * temp_vec - temp_vec_two
            call VecAXPBY(temp_vec_two, &
                  2 * real_roots(i), &
                  PFLARE_MINUS_ONE, &
                  temp_vec, ierr)

            ! y = y + 1/(Re(theta_i)^2 + Imag(theta_i)^2) * temp_vec_two
            call VecAXPY(y, &
                     PFLARE_ONE/(real_roots(i)**2 + imag_roots(i)**2), &
                     temp_vec_two, ierr)  
                     
            if (i .le. size(real_roots) - 2) then
               ! temp_vec_three = A * temp_vec_two
               call MatMult(mat, temp_vec_two, temp_vec_three, ierr)    

               ! temp_vec = temp_vec - 1/(Re(theta_i)^2 + Imag(theta_i)^2) * temp_vec_three
               call VecAXPY(temp_vec, &
                        PFLARE_MINUS_ONE/(real_roots(i)**2 + imag_roots(i)**2), &
                        temp_vec_three, ierr)               
            end if

            ! Skip two evals
            i = i + 2

         end if
      end do

      ! Final step if last root is real
      if (imag_roots(size(real_roots)) == 0d0) then

         ! Skips eigenvalues that are numerically zero
         if (abs(real_roots(size(real_roots))) > PFLARE_TOL_ZERO) then

            ! y = y + theta_i * temp_vec
            call VecAXPBY(y, &
                     PFLARE_ONE/real_roots(size(real_roots)), &
                     PFLARE_ONE, &
                     temp_vec, ierr) 
         end if
      end if

   end subroutine petsc_newton

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine petsc_newton_residual(mat, real_roots, imag_roots, temp_vec, temp_vec_two, temp_vec_three, x, y)

      ! Applies a gmres residual polynomial in the newton basis matrix-free as an inverse
      ! This is different than petsc_newton which applies p(A)v, 
      ! whereas this routine applies pi(A)v
      ! This is (a slightly modified) Algorithm 1 in Loe and saves some flops when we don't need the solution
      ! just the residual
      ! y = A x

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! Input
      type(tMat), intent(in)              :: mat
      PetscReal, dimension(:), intent(in) :: real_roots, imag_roots
      type(tVec), intent(in)              :: temp_vec, temp_vec_two, temp_vec_three
      type(tVec)                          :: x
      type(tVec)                          :: y

      ! Local
      integer :: order
      PetscErrorCode :: ierr      

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! y = x
      call VecCopy(x, y, ierr)

      ! ~~~~~~~~~~~~
      ! Iterate over the order
      ! ~~~~~~~~~~~~
      order = 1
      ! Does every e'val in this loop unlike when we apply p(A)v
      do while (order .le. size(real_roots))

         ! If real this is easy
         if (imag_roots(order) == 0d0) then

            ! Skips eigenvalues that are numerically zero - see 
            ! the comment in calculate_gmres_polynomial_roots_newton 
            if (abs(real_roots(order)) < PFLARE_TOL_ZERO) then
               order = order + 1
               cycle
            end if

            ! temp_vec_two = A * y
            call MatMult(mat, y, temp_vec_two, ierr)            

            ! y = y - theta_i * temp_vec_two
            call VecAXPY(y, &
                     PFLARE_MINUS_ONE/real_roots(order), &
                     temp_vec_two, ierr)

            order = order + 1

         ! If imaginary, then have to combine the e'val and its
         ! complex conjugate to keep the arithmetic real
         ! Relies on the complex conjugate being next to each other
         else

            ! Skips eigenvalues that are numerically zero
            if (real_roots(order)**2 + imag_roots(order)**2 < PFLARE_TOL_ZERO) then
               order = order + 2
               cycle
            end if           
            
            ! temp_vec_two = A * y
            call MatMult(mat, y, temp_vec_two, ierr)   

            ! temp_vec = A * temp_vec_two
            call MatMult(mat, temp_vec_two, temp_vec, ierr)              

            ! temp_vec = temp_vec - 2 * Re(theta_i) * temp_vec_two
            call VecAXPY(temp_vec, &
                  -2 * real_roots(order), &
                  temp_vec_two, ierr)

            ! y = y + 1/(Re(theta_i)^2 + Imag(theta_i)^2) * temp_vec
            call VecAXPY(y, &
                     PFLARE_ONE/(real_roots(order)**2 + imag_roots(order)**2), &
                     temp_vec, ierr)

            ! Skip two evals
            order = order + 2

         end if
      end do

   end subroutine petsc_newton_residual

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine petsc_newton_block(mat_ctx, x_mat, y_mat, recip_diag, block_applied)

      ! Applies a gmres polynomial in the newton basis matrix-free as an inverse
      ! for a block of right hand sides, x_mat, ie the multiple rhs version of petsc_newton
      ! The roots are stored in real_roots, imag_roots in the input matshell
      ! Based on Loe 2021 Toward efficient polynomial preconditioning for GMRES
      ! This is Algorithm 3 in Loe
      ! The matvecs of petsc_newton become sparse matrix-dense matrix products (SpMM)
      ! y_mat = A x_mat

      ! If recip_diag is not null we are applying the diagonally scaled polynomial
      ! q(D^-1 A), with the mat passed in the *unscaled* A and recip_diag = D^-1.
      ! We do this rather than running the products on the D^-1 A matshell, as every
      ! product on a shell degrades to a column by column matvec. The caller must have
      ! already scaled the block of rhs, ie x_mat = D^-1 X
      ! Every product below is therefore scaled by D^-1 immediately after it is computed
      ! and before the result is used for anything else - that is exactly the arithmetic
      ! of the inner D^-1 A matshell, done blockwise

      ! block_applied comes back false (with y_mat untouched) if the block of rhs we've
      ! been given has no product with mat, so the caller can fall back to a column by
      ! column apply

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! Input
      type(mat_ctxtype), intent(inout)    :: mat_ctx
      type(tMat)                          :: x_mat
      type(tMat)                          :: y_mat
      type(tVec)                          :: recip_diag
      logical, intent(out)                :: block_applied

      ! Local
      PetscReal, dimension(:), pointer :: real_roots, imag_roots
      type(tMat) :: temp_mat, temp_mat_two, temp_mat_three
      integer :: i, nroots
      logical :: scaled
      PetscErrorCode :: ierr

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      block_applied = .FALSE.
      scaled = .NOT. PetscObjectIsNull(recip_diag)
      real_roots => mat_ctx%real_roots
      imag_roots => mat_ctx%imag_roots
      nroots = size(real_roots)

      temp_mat       = mat_ctx%mf_temp_mat(MF_MAT_TEMP)
      temp_mat_two   = mat_ctx%mf_temp_mat(MF_MAT_TEMP_TWO)
      temp_mat_three = mat_ctx%mf_temp_mat(MF_MAT_TEMP_THREE)

      ! If we have more than one root we have to do products as we iterate over
      ! the roots - make sure they are attached (they stay attached between
      ! applies, only the numeric phase runs below)
      ! This has to happen before we write any values into the temporaries, as the
      ! symbolic product may set them up
      if (nroots > 1) then
         call ensure_block_kernel_products(mat_ctx, MF_MAT_TEMP_THREE, block_applied)
         if (.NOT. block_applied) return
         block_applied = .FALSE.
      end if

      ! temp_mat = x_mat
      call MatCopy(x_mat, temp_mat, SAME_NONZERO_PATTERN, ierr)
      ! y_mat = 0
      call MatZeroEntries(y_mat, ierr)

      ! ~~~~~~~~~~~~
      ! Iterate over the i
      ! ~~~~~~~~~~~~
      i = 1
      do while (i .le. nroots - 1)

         ! If real this is easy
         if (imag_roots(i) == 0d0) then

            ! Skips eigenvalues that are numerically zero - see
            ! the comment in calculate_gmres_polynomial_roots_newton
            if (abs(real_roots(i)) < PFLARE_TOL_ZERO) then
               i = i + 1
               cycle
            end if

            ! y_mat = y_mat + theta_i * temp_mat
            call MatAXPY(y_mat, &
                     PFLARE_ONE/real_roots(i), &
                     temp_mat, SAME_NONZERO_PATTERN, ierr)

            ! temp_mat_two = A * temp_mat
            call MatProductNumeric(temp_mat_two, ierr)
            if (scaled) call MatDiagonalScale(temp_mat_two, recip_diag, PETSC_NULL_VEC, ierr)

            ! temp_mat = temp_mat - theta_i * temp_mat_two
            call MatAXPY(temp_mat, &
                     PFLARE_MINUS_ONE/real_roots(i), &
                     temp_mat_two, SAME_NONZERO_PATTERN, ierr)

            i = i + 1

         ! If imaginary, then have to combine the e'val and its
         ! complex conjugate to keep the arithmetic real
         ! Relies on the complex conjugate being next to each other
         else

            ! Skips eigenvalues that are numerically zero
            if (real_roots(i)**2 + imag_roots(i)**2 < PFLARE_TOL_ZERO) then
               i = i + 2
               cycle
            end if

            ! temp_mat_two = A * temp_mat
            call MatProductNumeric(temp_mat_two, ierr)
            if (scaled) call MatDiagonalScale(temp_mat_two, recip_diag, PETSC_NULL_VEC, ierr)

            ! temp_mat_two = 2 * Re(theta_i) * temp_mat - temp_mat_two
            call MatScale(temp_mat_two, PFLARE_MINUS_ONE, ierr)
            call MatAXPY(temp_mat_two, &
                  PFLARE_TWO * real_roots(i), &
                  temp_mat, SAME_NONZERO_PATTERN, ierr)

            ! y_mat = y_mat + 1/(Re(theta_i)^2 + Imag(theta_i)^2) * temp_mat_two
            call MatAXPY(y_mat, &
                     PFLARE_ONE/(real_roots(i)**2 + imag_roots(i)**2), &
                     temp_mat_two, SAME_NONZERO_PATTERN, ierr)

            if (i .le. nroots - 2) then
               ! temp_mat_three = A * temp_mat_two
               call MatProductNumeric(temp_mat_three, ierr)
               if (scaled) call MatDiagonalScale(temp_mat_three, recip_diag, PETSC_NULL_VEC, ierr)

               ! temp_mat = temp_mat - 1/(Re(theta_i)^2 + Imag(theta_i)^2) * temp_mat_three
               call MatAXPY(temp_mat, &
                        PFLARE_MINUS_ONE/(real_roots(i)**2 + imag_roots(i)**2), &
                        temp_mat_three, SAME_NONZERO_PATTERN, ierr)
            end if

            ! Skip two evals
            i = i + 2

         end if
      end do

      ! Final step if last root is real
      if (imag_roots(nroots) == 0d0) then

         ! Skips eigenvalues that are numerically zero
         if (abs(real_roots(nroots)) > PFLARE_TOL_ZERO) then

            ! y_mat = y_mat + theta_i * temp_mat
            call MatAXPY(y_mat, &
                     PFLARE_ONE/real_roots(nroots), &
                     temp_mat, SAME_NONZERO_PATTERN, ierr)
         end if
      end if

      block_applied = .TRUE.

   end subroutine petsc_newton_block

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine shell_poly_block_apply(shell_mat, x_mat, y_mat, block_applied)

      ! Applies one of the matrix-free gmres polynomial inverses to a block of
      ! right hand sides, x_mat, ie the multiple rhs version of the matvecs
      ! petsc_matvec_poly_mf / petsc_matvec_right_scale_poly_mf and their newton equivalents
      ! The shell_mat passed in is the matshell built by build_gmres_polynomial_inverse
      ! (or its newton equivalent) and we dispatch on what its context contains

      ! block_applied comes back false (with y_mat untouched) whenever we can't do a
      ! block apply, and the caller must then apply column by column instead

      ! The polynomials with a right diagonal scaling store an inner matshell in
      ! mat_ctx%mat_scaled and the block kernels below deliberately bypass it - they do
      ! real products with the unscaled mat_ctx%mat and then apply the rest of the inner
      ! shell's arithmetic themselves (a product on a shell would degrade to a column by
      ! column matvec). That means the kernels have to be told which inner operator they
      ! are emulating - mat_ctx%neumann_inner picks I - D^-1 A (the Neumann polynomial,
      ! see Neumann_Poly.F90) rather than D^-1 A (the diagonally scaled gmres polynomials)
      ! Any new inner operator variant must add to that flag rather than reuse it,
      ! otherwise the bypass will silently give the wrong answer

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! Input
      type(tMat), intent(in)    :: shell_mat
      type(tMat)                :: x_mat
      type(tMat)                :: y_mat
      logical, intent(out)      :: block_applied

      ! Local
      PetscErrorCode :: ierr
      integer :: n_temps
      logical :: scaled
      type(mat_ctxtype), pointer :: mat_ctx => null()
      ! The block of rhs and the D^-1 we hand to the kernels
      ! kernel_recip is left null in the unscaled case
      type(tMat) :: kernel_x, temp_mat
      type(tVec) :: kernel_recip, temp_vec

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      block_applied = .FALSE.
      call MatShellGetContext(shell_mat, mat_ctx, ierr)

      ! Are we applying q(D^-1 A) D^-1 rather than just q(A)?
      temp_mat = mat_ctx%mat_scaled
      scaled = .NOT. PetscObjectIsNull(temp_mat)

      kernel_x = x_mat

      if (scaled) then

         ! Lazily build somewhere to store D^-1
         temp_vec = mat_ctx%mf_vec_diag_recip
         if (PetscObjectIsNull(temp_vec)) then
            call VecDuplicate(mat_ctx%mf_temp_vec(MF_VEC_DIAG), temp_vec, ierr)
            ! The new handle has to go back into the context, or we would create
            ! (and leak) a new vec on every apply
            mat_ctx%mf_vec_diag_recip = temp_vec
         end if
         ! We have to refresh the values every apply - the diagonal in MF_VEC_DIAG is
         ! updated in place whenever the matshell is reused with the same nonzero
         ! pattern, and there is no hook that would let us know that has happened
         call VecCopy(mat_ctx%mf_temp_vec(MF_VEC_DIAG), temp_vec, ierr)
         call VecReciprocal(temp_vec, ierr)
         kernel_recip = temp_vec
      end if

      ! ~~~~~~~~~~~~
      ! Dispatch on what sort of polynomial is in the context
      ! The newton basis needs three dense temporaries, the power/arnoldi/neumann
      ! basis two (the horner iteration ping-pongs between them)
      ! The Neumann polynomial has to be checked first, as it also has coefficients
      ! ~~~~~~~~~~~~
      if (mat_ctx%neumann_inner) then
         ! The Neumann polynomial is q(I - D^-1 A) D^-1 with coefficients of one and is
         ! always built with its inner matshell, so we should always have a scaled apply
         ! here - don't rely on that though, we can't emulate the inner operator without
         ! the diagonal
         if (.NOT. scaled) return
         if (.NOT. associated(mat_ctx%coefficients)) return
         n_temps = 2
      else if (associated(mat_ctx%real_roots)) then
         n_temps = 3
      else if (associated(mat_ctx%coefficients)) then
         n_temps = 2
      else
         ! Nothing we know how to apply blockwise
         return
      end if

      ! Make the dense temporaries, plus somewhere to put D^-1 X if we need it
      call ensure_block_temp_mats(mat_ctx, x_mat, n_temps, ierr, need_rhs=scaled)

      ! Do the right diagonal scaling on the whole block, ie MF_MAT_RHS = D^-1 X
      if (scaled) then
         call MatCopy(x_mat, mat_ctx%mf_temp_mat(MF_MAT_RHS), SAME_NONZERO_PATTERN, ierr)
         call MatDiagonalScale(mat_ctx%mf_temp_mat(MF_MAT_RHS), kernel_recip, PETSC_NULL_VEC, ierr)
         kernel_x = mat_ctx%mf_temp_mat(MF_MAT_RHS)
      end if

      if (mat_ctx%neumann_inner) then

         ! Neumann polynomial - horner with an inner operator of I - D^-1 A
         call petsc_horner_block(mat_ctx, kernel_x, y_mat, kernel_recip, &
                  .TRUE., block_applied)

      else if (associated(mat_ctx%real_roots)) then

         ! Newton basis
         call petsc_newton_block(mat_ctx, kernel_x, y_mat, kernel_recip, block_applied)

      else

         ! Power/arnoldi basis
         call petsc_horner_block(mat_ctx, kernel_x, y_mat, kernel_recip, &
                  .FALSE., block_applied)

      end if

   end subroutine shell_poly_block_apply

end module gmres_poly_apply
