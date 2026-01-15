module gmres_poly_newton

   use petscmat
   use gmres_poly
   use c_petsc_interfaces

#include "petsc/finclude/petscmat.h"   

   implicit none
   public 
   
   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine modified_leja(real_roots, imag_roots, indices)

      ! Computes a modified leja ordering of the eigenvalues
      ! and re-orders in place
      ! The roots must be passed in with complex conjugate pairs next to each other
      ! with positive imag e'vals first

      ! ~~~~~~
      PetscReal, dimension(:), intent(inout)  :: real_roots, imag_roots
      integer, dimension(:), allocatable, intent(inout) :: indices

      ! Local variables
      integer :: i_loc, k_loc, counter
      integer :: max_loc(1)
      PetscReal, dimension(:), allocatable :: magnitude
      PetscReal :: a, b, squares, max_mag
      logical, dimension(size(real_roots)) :: sorted

      ! ~~~~~~    

      allocate(magnitude(size(real_roots)))
      allocate(indices(size(real_roots)))

      ! Compute the magnitudes of the evals
      do i_loc = 1, size(real_roots)
         magnitude(i_loc) = sqrt(real_roots(i_loc)**2 + imag_roots(i_loc)**2)
      end do
      ! Find the biggest
      max_loc = maxloc(magnitude)
      counter = 1

      sorted = .FALSE.

      ! That is our first entry
      indices(counter) = max_loc(1)
      sorted(indices(counter)) = .TRUE.
      counter = counter + 1
      ! If it was imaginary its complex conjugate is next to this one
      if (imag_roots(indices(counter-1)) /= 0d0) then
         ! dgeev returns roots with the positive imaginary part first
         if (imag_roots(indices(counter-1)) > 0) then
            ! So if positive we know the conjugate is ahead one
            indices(counter) = indices(counter - 1) + 1
         else
            ! If negative we know the conjugate is one behind
            indices(counter) = indices(counter - 1) - 1
         end if
         sorted(indices(counter)) = .TRUE.
         counter = counter + 1
      end if

      ! Do while we still have some sorting to do
      do while (counter-1 < size(real_roots))

         max_mag = -huge(0d0)

         ! For each value compute a product of differences
         do i_loc = 1, size(real_roots)

            ! Skip any we've already sorted
            if (sorted(i_loc)) cycle
            magnitude(i_loc) = 1d0

            ! Loop over all those we've sorted so far
            do k_loc = 1, counter-1
               
               ! Distance
               a = real_roots(i_loc) - real_roots(indices(k_loc))
               b = imag_roots(i_loc) - imag_roots(indices(k_loc))

               ! If we have repeated roots this will be exactly zero, and magnitude will be -infinity
               squares = a**2 + b**2
               magnitude(i_loc) = magnitude(i_loc) + &
                  log10(sqrt(squares))
            end do
            
            ! Store the biggest as we're going
            if (magnitude(i_loc) > max_mag) then
               max_mag = magnitude(i_loc)
               max_loc(1) = i_loc
            end if            
         end do

         ! If we found nothing (ie the only unsorted are repeated roots with zero distance)
         ! just have the next entry in the list
         if (max_mag < 0) then
            do i_loc = 1, size(real_roots)
               if (.NOT. sorted(i_loc)) then
                  max_loc(1) = i_loc
                  exit
               endif
            end do
         end if      

         ! The new index is the biggest in distance 
         indices(counter) = max_loc(1)
         sorted(indices(counter)) = .TRUE.
         counter = counter + 1
         ! If it was imaginary its complex conjugate is next to this one
         if (imag_roots(indices(counter - 1)) /= 0d0) then
            ! dgeev returns roots with the positive imaginary part first
            ! We did actually find a situation where due to rounding differnces, the 
            ! conjugate with negative imaginary root had a bigger product of factors
            ! (by 1e-14) than the positive imaginary root, so we can't guarantee we hit 
            ! the positive one first
            if (imag_roots(indices(counter-1)) > 0) then
               ! So if positive we know the conjugate is ahead one

               indices(counter) = indices(counter - 1) + 1
            else
               ! If negative we know the conjugate is one behind
               indices(counter) = indices(counter - 1) - 1
            end if    
            sorted(indices(counter)) = .TRUE.
            counter = counter + 1
         end if
      end do

      deallocate(magnitude)

   end subroutine modified_leja   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine calculate_gmres_polynomial_roots_newton(matrix, poly_order, add_roots, coefficients)

      ! Computes a fixed order gmres polynomial for the matrix passed in
      ! and outputs the Harmonic Ritz values (ie the roots) which we can use to apply 
      ! a polynomial in the Newton basis. This should be stable at high order
      ! The cost of computing the newton basis is many reductions in parallel.
      ! We don't provide a way to compute the monomial polynomial coefficients from the roots 
      ! (although we could by rearranging the newton polynomial) as I don't think this would be 
      ! stable at high order anyway. The only reason we use the monomial coefficients 
      ! is to easily build an assembled (approximate) matrix inverse, and 
      ! f you want that with these roots you could build one directly (although it would not 
      ! be very sparse at high order, and a fixed sparsity version at higher order than either the 
      ! power basis or Arnoldi basis could compute is likely not any better)

      ! ~~~~~~
      type(tMat), intent(in)                            :: matrix
      integer, intent(in)                               :: poly_order
      logical, intent(in)                               :: add_roots
      PetscReal, dimension(:, :), pointer, intent(inout)     :: coefficients

      ! Local variables
      PetscInt :: global_rows, global_cols, local_rows, local_cols
      integer :: lwork, subspace_size, rank, i_loc, comm_size, comm_rank, errorcode, iwork_size, j_loc
      integer :: total_extra, counter, k_loc, m
      PetscErrorCode :: ierr      
      MPIU_Comm :: MPI_COMM_MATRIX
      PetscReal, dimension(poly_order+2,poly_order+1) :: H_n
      PetscReal, dimension(poly_order+1,poly_order+2) :: H_n_T
      PetscReal, dimension(poly_order+1) :: e_d, solution, s, pof
      integer, dimension(poly_order+1) :: extra_pair_roots, overflow
      integer, dimension(:), allocatable :: iwork_allocated, indices
      PetscReal, dimension(:), allocatable :: work
      PetscReal, dimension(:,:), allocatable :: VL, VR
      PetscReal :: beta, div_real, div_imag, a, b, c, d, div_mag
      PetscReal, dimension(:, :), allocatable :: coefficients_temp
      type(tVec) :: w_j
      type(tVec), dimension(poly_order+2) :: V_n
      logical :: use_harmonic_ritz = .TRUE.
      PetscReal :: rcond = 1e-12

      ! ~~~~~~    

      ! This is how many columns we have in K_m
      subspace_size = poly_order + 1

      ! We might want to call the gmres poly creation on a sub communicator
      ! so let's get the comm attached to the matrix and make sure to use that 
      call PetscObjectGetComm(matrix, MPI_COMM_MATRIX, ierr)  
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)     
      ! Get the comm rank
      call MPI_Comm_rank(MPI_COMM_MATRIX, comm_rank, errorcode)     
      ! Get the matrix sizes
      call MatGetSize(matrix, global_rows, global_cols, ierr)
      call MatGetLocalSize(matrix, local_rows, local_cols, ierr)

      if (subspace_size > global_rows) then
         print *, "The input subspace size is greater than the matrix size"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! ~~~~~~~~~~
      ! Allocate space and create random numbers 
      ! The first vec has random numbers in it
      ! ~~~~~~~~~~ 
      call create_temp_space_box_muller(matrix, subspace_size, V_n)
      
      ! Create an extra vector for storage
      call VecDuplicate(V_n(1), w_j, ierr)      
      
      ! Do the Arnoldi and compute H_n
      ! Use the same lucky tolerance as petsc
      call arnoldi(matrix, poly_order, 1d-30, V_n, w_j, beta, H_n, m)

      ! ~~~~~~~~~~~
      ! Now the Ritz values are just the eigenvalues of the square part of H_n
      ! We're actually going to use the Harmonic Ritz values which are the reciprocals
      ! of the (ordinary) Ritz values of A^-1
      ! see Embree - Polynomial preconditioned arnoldi with stability control 
      ! and Goossens - Ritz and Harmonic Ritz Values and the Convergence of FOM and GMRES
      ! ~~~~~~~~~~~
      if (use_harmonic_ritz) then

         e_d = 0
         e_d(poly_order + 1) = 1

         ! Now we have to be careful here 
         ! PETSc and Trilinos both just use an LU factorisation here to solve
         ! H_d^-H e_d
         ! But we can have the situation where the user has requested a high 
         ! polynomial order and H_d ends up not full rank
         ! In that case the LU factorisation fails 
         ! Previously we just used to abort and tell the user to use lower order
         ! But that is not super helpful
         ! Instead we use a rank revealing factorisation that gives the 
         ! minimum norm solution
         ! What we find is that when use this to compute eigenvalues we find e-vals 
         ! as we might expect up to the rank
         ! but then we have some eigenvalues that are numerically zero
         ! We keep those and our application of the newton polynomial in 
         ! petsc_matvec_gmres_newton_mf and petsc_matvec_gmres_newton_mf_residual
         ! just skips them and hence we don't do any 
         ! extra work in the application phase than we would have done with lower order     
         
         ! ~~~~~~~~~~~
         ! ~~~~~~~~~~~

         ! Rank revealing min norm solution
         H_n_T = transpose(H_n)

         allocate(work(1))
         allocate(iwork_allocated(1))
         lwork = -1         

         ! We have rcond = 1e-12 which is used to decide what singular values to drop
         ! Matlab uses max(size(A))*eps(norm(A)) in their pinv
         call dgelsd(poly_order + 1, poly_order + 1, 1, H_n_T, size(H_n_T, 1), &
                        e_d, size(e_d), s, rcond, rank, &
                        work, lwork, iwork_allocated, errorcode)
         lwork = int(work(1))
         iwork_size = iwork_allocated(1)
         deallocate(work, iwork_allocated)
         allocate(work(lwork)) 
         allocate(iwork_allocated(iwork_size))
         call dgelsd(poly_order + 1, poly_order + 1, 1, H_n_T, size(H_n_T, 1), &
                        e_d, size(e_d), s, rcond, rank, &
                        work, lwork, iwork_allocated, errorcode)
         deallocate(work, iwork_allocated)         

         ! Copy in the solution
         solution = e_d

         if (errorcode /= 0) then
            print *, "Harmonic Ritz solve failed"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
         end if

         ! Scale f by H(d+1,d)^2
         solution = solution * H_n(poly_order + 2, poly_order + 1)**2
         ! Add to the last column of H_n
         H_n(1:poly_order + 1, poly_order + 1) = &
            H_n(1:poly_order + 1, poly_order + 1) + solution
      end if

      ! ~~~~~~~~~~~~~~
      ! Now compute the eigenvalues of the square part of H_n
      ! ie either compute the Ritz or Harmonic Ritz values
      ! ~~~~~~~~~~~~~~

      allocate(work(1))
      ! Not used but we have to allocate
      allocate(VL(0, 0))
      allocate(VR(0, 0))
      lwork = -1      

      ! Compute the eigenvalues
      call dgeev('N', 'N', poly_order + 1, H_n, size(H_n, 1), &
                     coefficients(1, 1), coefficients(1, 2), VL, 1, VR, 1, &
                     work, lwork, errorcode)
      lwork = int(work(1))
      deallocate(work)
      allocate(work(lwork)) 
      call dgeev('N', 'N', poly_order + 1, H_n, size(H_n, 1), &
                     coefficients(1, 1), coefficients(1, 2), VL, 1, VR, 1, &
                     work, lwork, errorcode)
      deallocate(work, VL, VR)

      if (errorcode /= 0) then
         print *, "Eig decomposition failed"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if  

      ! ~~~~~~~~~~~~~~
      ! Add roots for stability
      ! ~~~~~~~~~~~~~~         
      if (add_roots) then 

         ! Compute the product of factors
         pof = 1   
         extra_pair_roots = 0
         overflow = 0
         total_extra = 0
         do k_loc = 1, poly_order + 1

            a = coefficients(k_loc, 1)
            b = coefficients(k_loc, 2)      
            
            ! We have already computed pof for the positive imaginary complex conjugate
            if (b < 0) cycle

            ! Skips eigenvalues that are numerically zero
            if (abs(a) < 1e-12) cycle
            if (a**2 + b**2 < 1e-12) cycle

            ! Compute product(k)_{i, j/=i} * | 1 - theta_j/theta_i|
            do i_loc = 1, poly_order + 1

               ! Skip
               if (k_loc == i_loc) cycle

               c = coefficients(i_loc, 1)
               d = coefficients(i_loc, 2)

               ! Skips eigenvalues that are numerically zero
               if (abs(c) < 1e-12) cycle
               if (c**2 + d**2 < 1e-12) cycle

               ! theta_k/theta_i
               div_real = (a * c + b * d)/(c**2 + d**2)
               div_imag = (b * c - a * d)/(c**2 + d**2)

               ! |1 - theta_k/theta_i|
               div_mag = sqrt((1 - div_real)**2 + div_imag**2)

               ! Pof is about to overflow, store the exponent and 
               ! reset pof back to one
               ! We can hit this for very high order polynomials, where we have to 
               ! add more roots than 22 (ie pof > 1e308)
               if (log10(pof(k_loc)) + log10(div_mag) > 307) then
                  overflow(k_loc) = overflow(k_loc) + int(log10(pof(k_loc)))
                  pof(k_loc) = 1
               end if            

               ! Product
               pof(k_loc) = pof(k_loc) * div_mag

            end do

            ! If pof > 1e4, we add an extra root, plus one extra for every 1e14
            if (log10(pof(k_loc)) > 4 .OR. overflow(k_loc) /= 0) then

               ! if real extra_pair_roots counts each distinct real root we're adding
               ! if imaginary it only counts a pair as one
               extra_pair_roots(k_loc) = ceiling((log10(pof(k_loc)) + overflow(k_loc) - 4.0)/14.0)
               total_extra = total_extra + extra_pair_roots(k_loc)

               ! If imaginary, the pof is the same for the conjugate, let's just set it to -1
               if (b > 0) then
                  ! We know the positive imaginary value is first, so the conjugate follows it
                  pof(k_loc+1) = -1
                  ! We need the conjugates as well
                  total_extra = total_extra + extra_pair_roots(k_loc)

               end if            
            end if
         end do

         ! If we have extra roots we need to resize the coefficients storage
         if (total_extra > 0) then
            allocate(coefficients_temp(size(coefficients, 1), size(coefficients, 2)))
            coefficients_temp(1:size(coefficients, 1), 1:size(coefficients, 2)) = coefficients
            deallocate(coefficients)
            allocate(coefficients(size(coefficients_temp, 1) + total_extra, 2))
            coefficients = 0
            coefficients(1:size(coefficients_temp, 1), :) = coefficients_temp
            deallocate(coefficients_temp)
         end if
      end if

      ! Take a copy of the existing roots
      coefficients_temp = coefficients

      if (add_roots) then
         
         ! Add the extra copies of roots, ensuring conjugate pairs we add 
         ! are next to each other
         counter = size(extra_pair_roots)+1
         do i_loc = 1, size(extra_pair_roots)

            ! For each extra root pair to add
            do j_loc = 1, extra_pair_roots(i_loc)

               coefficients(counter, :) = coefficients(i_loc, :)
               ! Add in the conjugate
               if (coefficients(i_loc, 2) > 0) then
                  coefficients(counter+1, 1) = coefficients(i_loc, 1)
                  coefficients(counter+1, 2) = -coefficients(i_loc, 2)
               end if

               ! Store a perturbed root so we have unique values for the leja sort below
               ! Just peturbing the real value
               coefficients_temp(counter, 1) = coefficients(i_loc, 1) + j_loc * 5e-8
               coefficients_temp(counter, 2) = coefficients(i_loc, 2)
               ! Add in the conjugate
               if (coefficients(i_loc, 2) > 0) then
                  coefficients_temp(counter+1, 1) = coefficients(i_loc, 1) + j_loc * 5e-8
                  coefficients_temp(counter+1, 2) = -coefficients(i_loc, 2)
               end if            

               counter = counter + 1
               if (coefficients(i_loc, 2) > 0) counter = counter + 1
            end do
         end do
      end if

      ! ~~~~~~~~~~~~~~
      ! Now compute a modified leja ordering for stability
      ! ~~~~~~~~~~~~~~      
      ! Called with the peturbed extra roots
      call modified_leja(coefficients_temp(:,1), coefficients_temp(:,2), indices)   

      ! Reorder the (non-peturbed) roots 
      coefficients(:,1) = coefficients(indices,1)
      coefficients(:,2) = coefficients(indices,2)

      ! Cleanup
      deallocate(coefficients_temp)
      do i_loc = 1, subspace_size+1
         call VecDestroy(V_n(i_loc), ierr)
      end do
      call VecDestroy(w_j, ierr)

   end subroutine calculate_gmres_polynomial_roots_newton   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine petsc_matvec_gmres_newton_mf(mat, x, y)

      ! Applies a gmres polynomial in the newton basis matrix-free as an inverse
      ! The roots are stored in mat_ctx%real_roots, mat_ctx%imag_roots in the input matshell
      ! Based on Loe 2021 Toward efficient polynomial preconditioning for GMRES
      ! This is Algorithm 3 in Loe
      ! y = A x

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! Input
      type(tMat), intent(in)    :: mat
      type(tVec) :: x
      type(tVec) :: y

      ! Local
      integer :: order, errorcode
      PetscErrorCode :: ierr      
      type(mat_ctxtype), pointer :: mat_ctx => null()

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      call MatShellGetContext(mat, mat_ctx, ierr)
      if (.NOT. associated(mat_ctx%real_roots)) then
         print *, "Roots in context aren't found"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! MF_VEC_TEMP = x
      call VecCopy(x, mat_ctx%mf_temp_vec(MF_VEC_TEMP), ierr)
      ! y = 0
      call VecSet(y, 0d0, ierr)

      ! ~~~~~~~~~~~~
      ! Iterate over the order
      ! ~~~~~~~~~~~~
      order = 1
      do while (order .le. size(mat_ctx%real_roots) - 1)

         ! If real this is easy
         if (mat_ctx%imag_roots(order) == 0d0) then

            ! Skips eigenvalues that are numerically zero - see 
            ! the comment in calculate_gmres_polynomial_roots_newton 
            if (abs(mat_ctx%real_roots(order)) < 1e-12) then
               order = order + 1
               cycle
            end if

            ! y = y + theta_i * MF_VEC_TEMP
            call VecAXPY(y, &
                     1d0/mat_ctx%real_roots(order), &
                     mat_ctx%mf_temp_vec(MF_VEC_TEMP), ierr)   
                                          
            ! MF_VEC_DIAG = A * MF_VEC_TEMP
            ! MF_VEC_DIAG isn't actually a diagonal here, we're just using this vec as temporary storage
            call MatMult(mat_ctx%mat, mat_ctx%mf_temp_vec(MF_VEC_TEMP), mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)
            ! MF_VEC_TEMP = MF_VEC_TEMP - theta_i * MF_VEC_DIAG
            call VecAXPY(mat_ctx%mf_temp_vec(MF_VEC_TEMP), &
                     -1d0/mat_ctx%real_roots(order), &
                     mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr) 

            order = order + 1

         ! If imaginary, then have to combine the e'val and its
         ! complex conjugate to keep the arithmetic real
         ! Relies on the complex conjugate being next to each other
         else

            ! Skips eigenvalues that are numerically zero
            if (mat_ctx%real_roots(order)**2 + mat_ctx%imag_roots(order)**2 < 1e-12) then
               order = order + 2
               cycle
            end if            

            ! MF_VEC_DIAG = A * MF_VEC_TEMP
            call MatMult(mat_ctx%mat, mat_ctx%mf_temp_vec(MF_VEC_TEMP), mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)    
            ! MF_VEC_DIAG = 2 * Re(theta_i) * MF_VEC_TEMP - MF_VEC_DIAG
            call VecAXPBY(mat_ctx%mf_temp_vec(MF_VEC_DIAG), &
                  2 * mat_ctx%real_roots(order), &
                  -1d0, &
                  mat_ctx%mf_temp_vec(MF_VEC_TEMP), ierr)

            ! y = y + 1/(Re(theta_i)^2 + Imag(theta_i)^2) * MF_VEC_DIAG
            call VecAXPY(y, &
                     1d0/(mat_ctx%real_roots(order)**2 + mat_ctx%imag_roots(order)**2), &
                     mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)  
                     
            if (order .le. size(mat_ctx%real_roots) - 2) then
               ! MF_VEC_RHS = A * MF_VEC_DIAG
               call MatMult(mat_ctx%mat, mat_ctx%mf_temp_vec(MF_VEC_DIAG), mat_ctx%mf_temp_vec(MF_VEC_RHS), ierr)    

               ! MF_VEC_TEMP = MF_VEC_TEMP - 1/(Re(theta_i)^2 + Imag(theta_i)^2) * MF_VEC_RHS
               call VecAXPY(mat_ctx%mf_temp_vec(MF_VEC_TEMP), &
                        -1d0/(mat_ctx%real_roots(order)**2 + mat_ctx%imag_roots(order)**2), &
                        mat_ctx%mf_temp_vec(MF_VEC_RHS), ierr)               
            end if

            ! Skip two evals
            order = order + 2

         end if
      end do

      ! Final step if last root is real
      if (mat_ctx%imag_roots(size(mat_ctx%real_roots)) == 0d0) then

         ! Skips eigenvalues that are numerically zero
         if (abs(mat_ctx%real_roots(order)) > 1e-12) then

            ! y = y + theta_i * MF_VEC_TEMP
            call VecAXPBY(y, &
                     1d0/mat_ctx%real_roots(order), &
                     1d0, &
                     mat_ctx%mf_temp_vec(MF_VEC_TEMP), ierr) 
         end if
      end if

   end subroutine petsc_matvec_gmres_newton_mf      


! -------------------------------------------------------------------------------------------------------------------------------

   subroutine petsc_matvec_gmres_newton_mf_residual(mat, x, y)

      ! Applies a gmres residual polynomial in the newton basis matrix-free as an inverse
      ! This is different than petsc_matvec_gmres_newton_mf which applies p(A)v, 
      ! whereas this routine applies pi(A)v
      ! This is (a slightly modified) Algorithm 1 in Loe and saves some flops when we don't need the solution
      ! just the residual
      ! y = A x

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! Input
      type(tMat), intent(in)    :: mat
      type(tVec) :: x
      type(tVec) :: y

      ! Local
      integer :: order, errorcode
      PetscErrorCode :: ierr      
      type(mat_ctxtype), pointer :: mat_ctx => null()

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      call MatShellGetContext(mat, mat_ctx, ierr)
      if (.NOT. associated(mat_ctx%real_roots)) then
         print *, "Roots in context aren't found"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! y = x
      call VecCopy(x, y, ierr)

      ! ~~~~~~~~~~~~
      ! Iterate over the order
      ! ~~~~~~~~~~~~
      order = 1
      ! Does every e'val in this loop unlike when we apply p(A)v
      do while (order .le. size(mat_ctx%real_roots))

         ! If real this is easy
         if (mat_ctx%imag_roots(order) == 0d0) then

            ! Skips eigenvalues that are numerically zero - see 
            ! the comment in calculate_gmres_polynomial_roots_newton 
            if (abs(mat_ctx%real_roots(order)) < 1e-12) then
               order = order + 1
               cycle
            end if

            ! MF_VEC_DIAG = A * y
            call MatMult(mat_ctx%mat, y, mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)            

            ! y = y - theta_i * MF_VEC_DIAG
            call VecAXPY(y, &
                     -1d0/mat_ctx%real_roots(order), &
                     mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)

            order = order + 1

         ! If imaginary, then have to combine the e'val and its
         ! complex conjugate to keep the arithmetic real
         ! Relies on the complex conjugate being next to each other
         else

            ! Skips eigenvalues that are numerically zero
            if (mat_ctx%real_roots(order)**2 + mat_ctx%imag_roots(order)**2 < 1e-12) then
               order = order + 2
               cycle
            end if           
            
            ! MF_VEC_DIAG = A * y
            call MatMult(mat_ctx%mat, y, mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)   

            ! MF_VEC_TEMP = A * MF_VEC_DIAG
            call MatMult(mat_ctx%mat, mat_ctx%mf_temp_vec(MF_VEC_DIAG), mat_ctx%mf_temp_vec(MF_VEC_TEMP), ierr)              

            ! MF_VEC_TEMP = MF_VEC_TEMP - 2 * Re(theta_i) * MF_VEC_DIAG
            call VecAXPY(mat_ctx%mf_temp_vec(MF_VEC_TEMP), &
                  -2 * mat_ctx%real_roots(order), &
                  mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)

            ! y = y + 1/(Re(theta_i)^2 + Imag(theta_i)^2) * MF_VEC_TEMP
            call VecAXPY(y, &
                     1d0/(mat_ctx%real_roots(order)**2 + mat_ctx%imag_roots(order)**2), &
                     mat_ctx%mf_temp_vec(MF_VEC_TEMP), ierr)

            ! Skip two evals
            order = order + 2

         end if
      end do

   end subroutine petsc_matvec_gmres_newton_mf_residual

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine build_gmres_polynomial_newton_inverse(matrix, poly_order, &
                  coefficients, &
                  poly_sparsity_order, matrix_free, reuse_mat, reuse_submatrices, &
                  inv_matrix)

      ! Builds a matrix which is an approximation to the inverse of a matrix using the 
      ! gmres polynomial in the newton basis
      ! Can only be applied matrix-free

      ! ~~~~~~
      type(tMat), intent(in)                                      :: matrix
      integer, intent(in)                                         :: poly_order
      PetscReal, dimension(:, :), target, contiguous, intent(inout)    :: coefficients
      integer, intent(in)                                         :: poly_sparsity_order
      logical, intent(in)                                         :: matrix_free      
      type(tMat), intent(inout)                                   :: reuse_mat, inv_matrix
      type(tMat), dimension(:), pointer, intent(inout)            :: reuse_submatrices

      ! Local variables
      PetscInt :: global_rows, global_cols, local_rows, local_cols
      integer :: comm_size, errorcode, order
      PetscErrorCode :: ierr      
      MPIU_Comm :: MPI_COMM_MATRIX
      type(mat_ctxtype), pointer :: mat_ctx=>null()
      logical :: reuse_triggered      
      PetscReal :: square_sum
      type(tMat) :: mat_product, temp_mat_A, temp_mat_two, temp_mat_three, mat_product_k_plus_1

      ! ~~~~~~       

      ! We might want to call the gmres poly creation on a sub communicator
      ! so let's get the comm attached to the matrix and make sure to use that 
      call PetscObjectGetComm(matrix, MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)        

      ! Get the local sizes
      call MatGetLocalSize(matrix, local_rows, local_cols, ierr)
      call MatGetSize(matrix, global_rows, global_cols, ierr)   
      
      ! ~~~~~~~
      ! Just build a matshell that applies our polynomial matrix-free
      ! ~~~~~~~
      if (matrix_free) then      

         ! If not re-using
         if (PetscObjectIsNull(inv_matrix)) then

            ! Have to dynamically allocate this
            allocate(mat_ctx)      

            ! We pass in the polynomial coefficients as the context
            call MatCreateShell(MPI_COMM_MATRIX, local_rows, local_cols, global_rows, global_cols, &
                        mat_ctx, inv_matrix, ierr)
            ! The subroutine petsc_matvec_gmres_newton_mf applies the polynomial inverse
            call MatShellSetOperation(inv_matrix, &
                        MATOP_MULT, petsc_matvec_gmres_newton_mf, ierr)

            call MatAssemblyBegin(inv_matrix, MAT_FINAL_ASSEMBLY, ierr)
            call MatAssemblyEnd(inv_matrix, MAT_FINAL_ASSEMBLY, ierr)
            ! Have to make sure to set the type of vectors the shell creates
            call ShellSetVecType(matrix, inv_matrix)          
            
            ! Create temporary vectors we use during application
            ! Make sure to use matrix here to get the right type (as the shell doesn't know about gpus)
            call MatCreateVecs(matrix, mat_ctx%mf_temp_vec(MF_VEC_TEMP), PETSC_NULL_VEC, ierr)          
            call MatCreateVecs(matrix, mat_ctx%mf_temp_vec(MF_VEC_RHS), mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)                

         ! Reusing 
         else
            call MatShellGetContext(inv_matrix, mat_ctx, ierr)

         end if

         mat_ctx%real_roots => coefficients(:, 1)
         mat_ctx%imag_roots => coefficients(:, 2)
         ! Now because the context reset deallocates the coefficient pointer 
         ! we want to make sure we don't leak memory, so we use pointer remapping here 
         ! to turn the 2D coefficient pointer into a 1D that we can store in mat_ctx%coefficients
         ! and then the deallocate on mat_ctx%coefficients should still delete all the memory
         mat_ctx%coefficients(1:2*size(coefficients,1)) => coefficients(:, :)
         ! This is the matrix whose inverse we are applying (just copying the pointer here)
         mat_ctx%mat = matrix     
         
         ! We're done
         return
      endif

      ! ~~~~~~~~~~~~
      ! If we're here then we want an assembled approximate inverse
      ! ~~~~~~~~~~~~         
      reuse_triggered = .NOT. PetscObjectIsNull(inv_matrix)   
      
      ! For the 0th and 1st order assembled polynomial we just combine the coefficients
      ! to get the mononomial form and assemble it, which should be stable for such low order
      ! For higher order we use the actual Newton form 

      ! If we're zeroth order poly this is trivial as it's just 1/theta_1 I
      if (poly_order == 0) then

         call build_gmres_polynomial_newton_inverse_0th_order(matrix, poly_order, coefficients, &
               inv_matrix) 

         ! Then just return
         return      

      ! For poly_order 1 and poly_sparsity_order 1 this is easy
      else if (poly_order == 1 .AND. poly_sparsity_order == 1) then
         
         ! Duplicate & copy the matrix, but ensure there is a diagonal present
         call mat_duplicate_copy_plus_diag(matrix, reuse_triggered, inv_matrix)

         ! Flags to prevent reductions when assembling (there are assembles in the shift)
         call MatSetOption(inv_matrix, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr) 
         call MatSetOption(inv_matrix, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE,  ierr)     
         call MatSetOption(inv_matrix, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)

         ! We only have two coefficients, so they are either both real or complex conjugates
         ! If real
         if (coefficients(1,2) == 0d0) then

            ! Have to be careful here, as we may be first order, but the second eigenvaule
            ! might have been set to zero thanks to the rank reducing solve 
            ! So we just check if the second imaginary part is zero and if it is
            ! we just compute a 0th order inverse - annoyingly we can't call 
            ! build_gmres_polynomial_newton_inverse_0th_order as that builds a MATDIAGONAL
            ! and in the tests there is a problem where we reuse the sparsity, in the first
            ! solve we don't have a zero coefficient but in the second solve we do
            ! So the mat type needs to remain consistent
            ! This can't happen in the complex case
            if (coefficients(2,1) == 0d0) then

               ! Set to zero
               call MatScale(inv_matrix, 0d0, ierr)
               ! Then add in the 0th order inverse
               call MatShift(inv_matrix, 1d0/coefficients(1,1), ierr)
               
               ! Then just return
               return  
            end if

            ! result = -A_ff/(theta_1 * theta_2)
            call MatScale(inv_matrix, -1d0/(coefficients(1, 1) * coefficients(2, 1)), ierr)

            ! result = I * (1/theta_1 + 1/theta_2) - A_ff/(theta_1 * theta_2)
            ! Don't need an assemble as there is one called in this
            call MatShift(inv_matrix, 1d0/(coefficients(1, 1)) + 1d0/(coefficients(2, 1)), ierr)       

         ! Complex conjugate roots, a +- ib
         else
            ! a^2 + b^2
            square_sum = coefficients(1,1)**2 + coefficients(1,2)**2

            ! Complex conjugate roots
            ! result = -A_ff / (a^2 + b^2)
            call MatScale(inv_matrix, -1d0/square_sum, ierr)
            ! result = 2a/(a^2 + b^2) I - A_ff / (a^2 + b^2)
            ! Don't need an assemble as there is one called in this
            call MatShift(inv_matrix, 2d0 * coefficients(1,1)/square_sum, ierr)       
         end if    

         ! Then just return
         return

      end if

      ! If we're constraining sparsity we've built a custom matrix-powers that assumes fixed sparsity
      if (poly_sparsity_order < poly_order) then    

         ! ! This routine is a custom one that builds our matrix powers and assumes fixed sparsity
         ! ! so that it doen't have to do much comms
         ! ! This also finishes off the asyn comms and computes the coefficients
         ! call mat_mult_powers_share_sparsity(matrix, poly_order, poly_sparsity_order, buffers, coefficients, &
         !          reuse_mat, reuse_submatrices, inv_matrix)

         ! ! Then just return
         return         
         
      end if

      ! ~~~~~~~~~~
      ! We are only here if we don't constrain_sparsity
      ! ~~~~~~~~~~

      ! If not re-using
      ! Copy in the initial matrix
      if (.NOT. reuse_triggered) then
         ! Duplicate & copy the matrix, but ensure there is a diagonal present
         call mat_duplicate_copy_plus_diag(matrix, .FALSE., inv_matrix)
      else
         ! For the powers > 1 the pattern of the original matrix will be different
         ! to the resulting inverse
         call MatCopy(matrix, inv_matrix, DIFFERENT_NONZERO_PATTERN, ierr)
      end if

      ! Set to zero as we add in each product of terms
      call MatScale(inv_matrix, 0d0, ierr)

      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(inv_matrix, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)  

      ! We start with an identity in mat_product
      call generate_identity(matrix, mat_product)

      ! ~~~~~~~~~~~~
      ! Iterate over the order
      ! This is basically the same as the MF application but we have to build the powers
      ! ~~~~~~~~~~~~      
      order = 1
      do while (order .le. poly_order - 1)

         ! Duplicate & copy the matrix, but ensure there is a diagonal present
         ! temp_mat_A is going to store things with the sparsity of A
         if (PetscObjectIsNull(temp_mat_A)) then
            call mat_duplicate_copy_plus_diag(matrix, .FALSE., temp_mat_A)     
         else
            ! Can reuse the sparsity 
            call mat_duplicate_copy_plus_diag(matrix, .TRUE., temp_mat_A)     
         end if         

         ! If real this is easy
         if (coefficients(order,2) == 0d0) then

            ! Skips eigenvalues that are numerically zero - see 
            ! the comment in calculate_gmres_polynomial_roots_newton 
            if (abs(coefficients(order,1)) < 1e-12) then
               order = order + 1
               cycle
            end if        

            ! Then add the scaled version of each product
            if (reuse_triggered) then
               ! If doing reuse we know our nonzeros are a subset
               call MatAXPY(inv_matrix, 1d0/coefficients(order,1), mat_product, SUBSET_NONZERO_PATTERN, ierr)
            else
               ! Have to use the DIFFERENT_NONZERO_PATTERN here
               call MatAXPYWrapper(inv_matrix, 1d0/coefficients(order,1), mat_product)
            end if

            ! temp_mat_A = A_ff/theta_k       
            call MatScale(temp_mat_A, -1d0/coefficients(order,1), ierr)
            ! temp_mat_A = I - A_ff/theta_k
            call MatShift(temp_mat_A, 1d0, ierr)    
            
            ! mat_product_k_plus_1 = mat_product * temp_mat_A
            call MatMatMult(temp_mat_A, mat_product, &
                  MAT_INITIAL_MATRIX, 1.5d0, mat_product_k_plus_1, ierr)      
            call MatDestroy(mat_product, ierr)  
            mat_product = mat_product_k_plus_1     
            
            order = order + 1

         ! Complex 
         else

            ! Skips eigenvalues that are numerically zero
            if (coefficients(order,1)**2 + coefficients(order,2)**2 < 1e-12) then
               order = order + 2
               cycle
            end if

            ! Compute 2a I - A
            ! Have to use the DIFFERENT_NONZERO_PATTERN here
            ! temp_mat_A = -A    
            call MatScale(temp_mat_A, -1d0, ierr)
            ! temp_mat_A = 2a I - A_ff
            call MatShift(temp_mat_A, 2d0 * coefficients(order,1), ierr)   
            ! temp_mat_A = (2a I - A_ff)/(a^2 + b^2)
            call MatScale(temp_mat_A, 1d0/(coefficients(order,1)**2 + coefficients(order,2)**2), ierr) 

            call MatMatMult(temp_mat_A, mat_product, &
                  MAT_INITIAL_MATRIX, 1.5d0, temp_mat_two, ierr)      

            ! Then add the scaled version of each product
            if (reuse_triggered) then
               ! If doing reuse we know our nonzeros are a subset
               call MatAXPY(inv_matrix, 1d0, temp_mat_two, SUBSET_NONZERO_PATTERN, ierr)
            else
               ! Have to use the DIFFERENT_NONZERO_PATTERN here
               call MatAXPYWrapper(inv_matrix, 1d0, temp_mat_two)
            end if            

            if (order .le. size(coefficients, 1) - 2) then
               ! temp_mat_three = matrix * temp_mat_two
               call MatMatMult(matrix, temp_mat_two, &
                     MAT_INITIAL_MATRIX, 1.5d0, temp_mat_three, ierr)     
               call MatDestroy(temp_mat_two, ierr)  

               ! Then add the scaled version of each product
               if (reuse_triggered) then
                  ! If doing reuse we know our nonzeros are a subset
                  call MatAXPY(mat_product, -1d0, temp_mat_three, SUBSET_NONZERO_PATTERN, ierr)
               else
                  ! Have to use the DIFFERENT_NONZERO_PATTERN here
                  call MatAXPYWrapper(mat_product, -1d0, temp_mat_three)
               end if               
               call MatDestroy(temp_mat_three, ierr) 
            else
               call MatDestroy(temp_mat_two, ierr)  
            end if

            ! Skip two evals
            order = order + 2

         end if       
      end do

      ! Final step if last root is real
      if (coefficients(order,2) == 0d0) then
         ! Add in the final term multiplied by 1/theta_poly_order

         ! Skips eigenvalues that are numerically zero
         if (abs(coefficients(order,1)) > 1e-12) then            
            if (reuse_triggered) then
               ! If doing reuse we know our nonzeros are a subset
               call MatAXPY(inv_matrix, 1d0/coefficients(order,1), mat_product, SUBSET_NONZERO_PATTERN, ierr)
            else
               ! Have to use the DIFFERENT_NONZERO_PATTERN here
               call MatAXPYWrapper(inv_matrix, 1d0/coefficients(order,1), mat_product)
            end if     
         end if       
      end if        

      call MatDestroy(temp_mat_A, ierr)
      call MatDestroy(mat_product, ierr)

   end subroutine build_gmres_polynomial_newton_inverse     
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine build_gmres_polynomial_newton_inverse_0th_order(matrix, poly_order, coefficients, &
                  inv_matrix)

      ! Specific 0th order inverse

      ! ~~~~~~
      type(tMat), intent(in)                            :: matrix
      integer, intent(in)                               :: poly_order
      PetscReal, dimension(:, :), target, contiguous, intent(inout)    :: coefficients
      type(tMat), intent(inout)                         :: inv_matrix

      ! Local variables
      integer :: errorcode
      PetscErrorCode :: ierr      
      logical :: reuse_triggered
      type(tVec) :: diag_vec   

      ! ~~~~~~      
      
      if (poly_order /= 0) then
         print *, "This is a 0th order inverse, but poly_order is not 0"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! Let's create a matrix to represent the inverse diagonal
      reuse_triggered = .NOT. PetscObjectIsNull(inv_matrix)       

      if (.NOT. reuse_triggered) then
         call MatCreateVecs(matrix, PETSC_NULL_VEC, diag_vec, ierr)
      else
         call MatDiagonalGetDiagonal(inv_matrix, diag_vec, ierr)
      end if

      ! Must be real as we only have one coefficient
      call VecSet(diag_vec, 1d0/coefficients(1, 1), ierr)

      ! We may be reusing with the same sparsity
      if (.NOT. reuse_triggered) then
         ! The matrix takes ownership of diag_vec and increases ref counter
         call MatCreateDiagonal(diag_vec, inv_matrix, ierr)
         call VecDestroy(diag_vec, ierr)
      else
         call MatDiagonalRestoreDiagonal(inv_matrix, diag_vec, ierr)
      end if             

   end subroutine build_gmres_polynomial_newton_inverse_0th_order   

! -------------------------------------------------------------------------------------------------------------------------------


end module gmres_poly_newton

