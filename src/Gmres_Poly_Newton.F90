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

   ! -------------------------------------------------------------------------------------------------

   subroutine cluster_eigenvalues_stable(real_roots, imag_roots, rel_tol, abs_tol)

      ! Clustering of (possibly complex) harmonic Ritz values.
      ! Numerically distinct clusters are moved to the front.
      ! Remaining entries are set to zero.
      ! Skips eigenvalues that are exactly zero (both real and imag parts).
      !
      ! Inputs:
      !   real_roots, imag_roots : eigenvalues (length k)
      !   rel_tol               : relative tolerance (suggest sqrt(eps) ~ 1e-8)
      !   abs_tol               : absolute tolerance (suggest eps * ||H|| ~ 1e-15)
      !
      ! Outputs:
      !   real_roots, imag_roots : clustered eigenvalues at front, zeros after

      PetscReal, dimension(:), intent(inout) :: real_roots, imag_roots
      PetscReal, intent(in)                  :: rel_tol, abs_tol
      integer                                :: i, j, n, n_unique, cluster_size
      logical, allocatable                   :: used(:)
      PetscReal                              :: dist, mag_i, mag_j, scale
      PetscReal                              :: sum_real, sum_imag
      PetscReal, allocatable                 :: rtmp(:), itmp(:)

      n = size(real_roots)

      allocate(used(n))
      allocate(rtmp(n), itmp(n))

      used = .false.
      n_unique = 0

      ! ---------------------------------------------------------
      ! All-pairs clustering
      ! ---------------------------------------------------------
      do i = 1, n

         if (used(i)) cycle

         ! Skip eigenvalues that are exactly zero
         if (real_roots(i) == 0.0d0 .AND. imag_roots(i) == 0.0d0) then
            used(i) = .true.
            cycle
         end if

         ! Start new cluster with eigenvalue i
         sum_real = real_roots(i)
         sum_imag = imag_roots(i)
         cluster_size = 1
         used(i) = .true.
         
         mag_i = sqrt(real_roots(i)**2 + imag_roots(i)**2)

         ! Look for all other eigenvalues close to this one
         do j = i + 1, n

            if (used(j)) cycle

            ! Skip exactly zero eigenvalues
            if (real_roots(j) == 0.0d0 .AND. imag_roots(j) == 0.0d0) then
               used(j) = .true.
               cycle
            end if

            mag_j = sqrt(real_roots(j)**2 + imag_roots(j)**2)
            
            ! Distance between eigenvalues
            dist = sqrt((real_roots(j) - real_roots(i))**2 + &
                        (imag_roots(j) - imag_roots(i))**2)

            ! Use the larger magnitude for relative scaling
            scale = max(mag_i, mag_j, 1.0d0)

            ! Check if within tolerance
            if (dist <= abs_tol + rel_tol * scale) then
               sum_real = sum_real + real_roots(j)
               sum_imag = sum_imag + imag_roots(j)
               cluster_size = cluster_size + 1
               used(j) = .true.
            end if

         end do

         ! Compute cluster centroid (mean)
         n_unique = n_unique + 1
         rtmp(n_unique) = sum_real / dble(cluster_size)
         itmp(n_unique) = sum_imag / dble(cluster_size)

      end do

      ! ---------------------------------------------------------
      ! Output compact form
      ! ---------------------------------------------------------
      real_roots = 0.0d0
      imag_roots = 0.0d0

      real_roots(1:n_unique) = rtmp(1:n_unique)
      imag_roots(1:n_unique) = itmp(1:n_unique)

      deallocate(used, rtmp, itmp)

   end subroutine cluster_eigenvalues_stable


   ! -------------------------------------------------------------------------------------------------

   subroutine compute_extra_roots(real_roots, imag_roots, real_roots_output, imag_roots_output)

      ! Add extra roots for stability
      ! Computes the product of factors for each eigenvalue and adds extra copies
      ! of roots that have large products (to improve polynomial stability)
      ! Only non-zero eigenvalues should be passed in
      ! real_roots_output, imag_roots_output are allocated and filled with the original
      ! roots plus any extra

      ! ~~~~~~
      PetscReal, dimension(:), intent(inout)              :: real_roots, imag_roots
      PetscReal, dimension(:), allocatable, intent(inout) :: real_roots_output, imag_roots_output

      ! Local variables
      integer :: i_loc, j_loc, k_loc, n_roots, total_extra, counter
      PetscReal :: a, b, c, d, div_real, div_imag, div_mag
      PetscReal, dimension(size(real_roots)) :: pof
      integer, dimension(size(real_roots)) :: extra_pair_roots, overflow

      ! ~~~~~~

      n_roots = size(real_roots)

      ! Compute the product of factors
      pof = 1   
      extra_pair_roots = 0
      overflow = 0
      total_extra = 0
      do k_loc = 1, n_roots

         a = real_roots(k_loc)
         b = imag_roots(k_loc)      
         
         ! We have already computed pof for the positive imaginary complex conjugate
         if (b < 0) cycle

         ! Skips eigenvalues that are numerically zero
         if (abs(a) < 1e-12) cycle
         if (a**2 + b**2 < 1e-12) cycle

         ! Compute product(k)_{i, j/=i} * | 1 - theta_j/theta_i|
         do i_loc = 1, n_roots

            ! Skip
            if (k_loc == i_loc) cycle

            c = real_roots(i_loc)
            d = imag_roots(i_loc)

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

      ! Allocate output arrays (original roots + extra roots)
      allocate(real_roots_output(n_roots + total_extra))
      allocate(imag_roots_output(n_roots + total_extra))
      real_roots_output = 0d0
      imag_roots_output = 0d0

      ! Copy in original roots
      real_roots_output(1:n_roots) = real_roots(1:n_roots)
      imag_roots_output(1:n_roots) = imag_roots(1:n_roots)

      ! Add the extra copies of roots, ensuring conjugate pairs we add 
      ! are next to each other
      counter = n_roots + 1
      do i_loc = 1, n_roots

         ! For each extra root pair to add
         do j_loc = 1, extra_pair_roots(i_loc)

            real_roots_output(counter) = real_roots(i_loc)
            imag_roots_output(counter) = imag_roots(i_loc)
            ! Add in the conjugate
            if (imag_roots(i_loc) > 0) then
               real_roots_output(counter+1) = real_roots(i_loc)
               imag_roots_output(counter+1) = -imag_roots(i_loc)
            end if

            counter = counter + 1
            if (imag_roots(i_loc) > 0) counter = counter + 1
         end do
      end do

   end subroutine compute_extra_roots   


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
      integer :: total_extra, counter, k_loc, m, numerical_order
      PetscErrorCode :: ierr      
      MPIU_Comm :: MPI_COMM_MATRIX
      PetscReal, dimension(poly_order+2,poly_order+1) :: H_n
      PetscReal, dimension(poly_order+1,poly_order+2) :: H_n_T
      PetscReal, dimension(poly_order+1) :: e_d, solution, s
      integer, dimension(:), allocatable :: iwork_allocated, indices
      PetscReal, dimension(:), allocatable :: work, real_roots_added, imag_roots_added
      PetscReal, dimension(:), allocatable :: perturbed_real, perturbed_imag
      PetscReal, dimension(:,:), allocatable :: VL, VR
      PetscReal :: beta
      PetscReal, dimension(:, :), allocatable :: coefficients_temp
      type(tVec) :: w_j
      type(tVec), dimension(poly_order+2) :: V_n
      logical :: use_harmonic_ritz = .TRUE.
      PetscReal :: rcond = 1e-12, rel_tol, abs_tol, H_norm

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
         ! Given the way the outside code is structured, we can't lower the poly_order
         ! in this routine and return
         ! Instead we keep the "zero" eigenvalues our application of the newton polynomial
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
      ! Now we have to check the output eigenvalues
      ! ~~~~~~~~~~~~~~

      ! These are the tolerances that control the clustering
      H_norm = norm2(H_n(1:m,1:m))
      rel_tol = 1.0d0 * sqrt(epsilon(1.0d0))
      abs_tol = epsilon(1.0d0) * max(H_norm, beta)

      print *, "coeffs", coefficients(:,1), "imag", coefficients(:,2)

      ! In some cases with numerical rank deficiency, we can still
      ! end up with non-zero (or negative) eigenvalues that
      ! are trivially small - we set them explicitly to zero
      do i_loc = 1, poly_order + 1
         if (coefficients(i_loc,1)**2 + coefficients(i_loc,2)**2 < &
            (abs_tol + rel_tol*H_norm)**2) then
            coefficients(i_loc, 1) = 0d0
            coefficients(i_loc, 2) = 0d0
         end if
      end do       
      
      print *, "coeffs after zero", coefficients(:,1), "imag", coefficients(:,2)
     
      ! ~~~~~~~~~~~~~~
      ! Cluster close eigenvalues together to improve stability of the polynomial evaluation
      ! For example when computing the e'vals of a constant diagonal matrix
      ! the rank revealing factorisation above doesn't always report a rank of 1 given roundoff
      ! Instead it returns multiple eigenvalues that are very close to each other, 
      ! and we want to cluster those together and treat them as one root
      ! ~~~~~~~~~~~~~~      

      ! This places all exactly zero eigenvalues at the end of coefficients
      call cluster_eigenvalues_stable(coefficients(:, 1), coefficients(:, 2), rel_tol, abs_tol)      

      print *, "coeffs after cluster", coefficients(:,1), "imag", coefficients(:,2)

      ! ~~~~~~~~~~~~~~
      ! Extract the non-zero eigenvalues for root adding and leja ordering
      ! Zero eigenvalues will be appended at the end
      ! ~~~~~~~~~~~~~~
      ! Count the number of non-zero eigenvalues after clustering
      numerical_order = 0
      do i_loc = 1, poly_order + 1
         if (coefficients(i_loc, 1) /= 0d0 .OR. coefficients(i_loc, 2) /= 0d0) then
            numerical_order = numerical_order + 1
         end if
      end do

      ! ~~~~~~~~~~~~~~
      ! Add roots for stability (only on non-zero eigenvalues)
      ! ~~~~~~~~~~~~~~         
      if (add_roots .AND. numerical_order > 0) then 

         ! Extract non-zero eigenvalues into a temporary array
         allocate(coefficients_temp(numerical_order, 2))
         counter = 0
         do i_loc = 1, poly_order + 1
            if (coefficients(i_loc, 1) /= 0d0 .OR. coefficients(i_loc, 2) /= 0d0) then
               counter = counter + 1
               coefficients_temp(counter, 1) = coefficients(i_loc, 1)
               coefficients_temp(counter, 2) = coefficients(i_loc, 2)
            end if
         end do

         ! Call compute_extra_roots only on the non-zero eigenvalues
         ! This allocates real_roots_added/imag_roots_added with the original + extra roots
         call compute_extra_roots(coefficients_temp(:, 1), coefficients_temp(:, 2), &
                  real_roots_added, imag_roots_added)

         ! total number of non-zero roots after adding extras
         total_extra = size(real_roots_added) - numerical_order

         ! Resize coefficients to hold non-zero roots (with extras) + zero roots at end
         deallocate(coefficients)
         print *, "size(real_roots_added)", size(real_roots_added), "poly_order + 1", poly_order + 1, "numerical_order", numerical_order
         print *, "new size", size(real_roots_added) + (poly_order + 1 - numerical_order)
         allocate(coefficients(size(real_roots_added) + (poly_order + 1 - numerical_order), 2))
         coefficients = 0d0

         ! Create perturbed copy for leja ordering
         allocate(perturbed_real(size(real_roots_added)))
         allocate(perturbed_imag(size(real_roots_added)))
         perturbed_real = real_roots_added
         perturbed_imag = imag_roots_added

         ! Perturb the extra roots so they have unique values for the leja sort
         counter = numerical_order + 1
         do i_loc = 1, numerical_order
            k_loc = 0
            do j_loc = counter, size(real_roots_added)
               ! Check if this extra root matches the original
               if (real_roots_added(j_loc) == coefficients_temp(i_loc, 1) .AND. &
                   abs(imag_roots_added(j_loc)) == abs(coefficients_temp(i_loc, 2))) then
                  k_loc = k_loc + 1
                  perturbed_real(j_loc) = real_roots_added(j_loc) + k_loc * 5e-8
               end if
            end do
         end do

         ! Leja order only the non-zero eigenvalues (with extras)
         call modified_leja(perturbed_real, perturbed_imag, indices)

         ! Reorder the (non-perturbed) roots using the leja ordering
         coefficients(1:size(real_roots_added), 1) = real_roots_added(indices)
         coefficients(1:size(real_roots_added), 2) = imag_roots_added(indices)

         ! Zero eigenvalues are already zero at the end from the coefficients = 0d0 above

         ! Cleanup
         deallocate(coefficients_temp, real_roots_added, imag_roots_added)
         deallocate(perturbed_real, perturbed_imag, indices)

      else

         ! No root adding - just leja order the non-zero eigenvalues
         ! and put zeros at the end
         if (numerical_order > 0) then

            ! Extract non-zero eigenvalues
            allocate(coefficients_temp(numerical_order, 2))
            counter = 0
            do i_loc = 1, poly_order + 1
               if (coefficients(i_loc, 1) /= 0d0 .OR. coefficients(i_loc, 2) /= 0d0) then
                  counter = counter + 1
                  coefficients_temp(counter, 1) = coefficients(i_loc, 1)
                  coefficients_temp(counter, 2) = coefficients(i_loc, 2)
               end if
            end do

            ! Leja order the non-zero eigenvalues
            call modified_leja(coefficients_temp(:, 1), coefficients_temp(:, 2), indices)

            ! Reorder and put zeros at the end
            coefficients = 0d0
            coefficients(1:numerical_order, 1) = coefficients_temp(indices, 1)
            coefficients(1:numerical_order, 2) = coefficients_temp(indices, 2)

            deallocate(coefficients_temp, indices)

         end if

      end if     

      ! Cleanup
      do i_loc = 1, subspace_size+1
         call VecDestroy(V_n(i_loc), ierr)
      end do
      call VecDestroy(w_j, ierr)

      print *, "coeffs after leja", coefficients(:,1), "imag", coefficients(:,2)
       

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
      integer :: i, errorcode
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
      ! Iterate over the i
      ! ~~~~~~~~~~~~
      i = 1
      do while (i .le. size(mat_ctx%real_roots) - 1)

         ! If real this is easy
         if (mat_ctx%imag_roots(i) == 0d0) then

            ! Skips eigenvalues that are numerically zero - see 
            ! the comment in calculate_gmres_polynomial_roots_newton 
            if (abs(mat_ctx%real_roots(i)) < 1e-12) then
               i = i + 1
               cycle
            end if

            ! y = y + theta_i * MF_VEC_TEMP
            call VecAXPY(y, &
                     1d0/mat_ctx%real_roots(i), &
                     mat_ctx%mf_temp_vec(MF_VEC_TEMP), ierr)   
                                          
            ! MF_VEC_DIAG = A * MF_VEC_TEMP
            ! MF_VEC_DIAG isn't actually a diagonal here, we're just using this vec as temporary storage
            call MatMult(mat_ctx%mat, mat_ctx%mf_temp_vec(MF_VEC_TEMP), mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)
            ! MF_VEC_TEMP = MF_VEC_TEMP - theta_i * MF_VEC_DIAG
            call VecAXPY(mat_ctx%mf_temp_vec(MF_VEC_TEMP), &
                     -1d0/mat_ctx%real_roots(i), &
                     mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr) 

            i = i + 1

         ! If imaginary, then have to combine the e'val and its
         ! complex conjugate to keep the arithmetic real
         ! Relies on the complex conjugate being next to each other
         else

            ! Skips eigenvalues that are numerically zero
            if (mat_ctx%real_roots(i)**2 + mat_ctx%imag_roots(i)**2 < 1e-12) then
               i = i + 2
               cycle
            end if            

            ! MF_VEC_DIAG = A * MF_VEC_TEMP
            call MatMult(mat_ctx%mat, mat_ctx%mf_temp_vec(MF_VEC_TEMP), mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)    
            ! MF_VEC_DIAG = 2 * Re(theta_i) * MF_VEC_TEMP - MF_VEC_DIAG
            call VecAXPBY(mat_ctx%mf_temp_vec(MF_VEC_DIAG), &
                  2 * mat_ctx%real_roots(i), &
                  -1d0, &
                  mat_ctx%mf_temp_vec(MF_VEC_TEMP), ierr)

            ! y = y + 1/(Re(theta_i)^2 + Imag(theta_i)^2) * MF_VEC_DIAG
            call VecAXPY(y, &
                     1d0/(mat_ctx%real_roots(i)**2 + mat_ctx%imag_roots(i)**2), &
                     mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)  
                     
            if (i .le. size(mat_ctx%real_roots) - 2) then
               ! MF_VEC_RHS = A * MF_VEC_DIAG
               call MatMult(mat_ctx%mat, mat_ctx%mf_temp_vec(MF_VEC_DIAG), mat_ctx%mf_temp_vec(MF_VEC_RHS), ierr)    

               ! MF_VEC_TEMP = MF_VEC_TEMP - 1/(Re(theta_i)^2 + Imag(theta_i)^2) * MF_VEC_RHS
               call VecAXPY(mat_ctx%mf_temp_vec(MF_VEC_TEMP), &
                        -1d0/(mat_ctx%real_roots(i)**2 + mat_ctx%imag_roots(i)**2), &
                        mat_ctx%mf_temp_vec(MF_VEC_RHS), ierr)               
            end if

            ! Skip two evals
            i = i + 2

         end if
      end do

      ! Final step if last root is real
      if (mat_ctx%imag_roots(size(mat_ctx%real_roots)) == 0d0) then

         ! Skips eigenvalues that are numerically zero
         if (abs(mat_ctx%real_roots(size(mat_ctx%real_roots))) > 1e-12) then

            ! y = y + theta_i * MF_VEC_TEMP
            call VecAXPBY(y, &
                     1d0/mat_ctx%real_roots(size(mat_ctx%real_roots)), &
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
!------------------------------------------------------------------------------------------------------------------------
   
   subroutine mat_mult_powers_share_sparsity_newton(matrix, poly_order, poly_sparsity_order, coefficients, &
                  reuse_mat, reuse_submatrices, cmat)

      ! Wrapper around mat_mult_powers_share_sparsity_cpu and mat_mult_powers_share_sparsity_kokkos     
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), target, intent(in)                     :: matrix
      integer, intent(in)                                :: poly_order, poly_sparsity_order
      PetscReal, dimension(:, :), target, contiguous, intent(inout)    :: coefficients
      type(tMat), intent(inout)                          :: reuse_mat, cmat
      type(tMat), dimension(:), pointer, intent(inout)   :: reuse_submatrices

! #if defined(PETSC_HAVE_KOKKOS)                     
!       integer(c_long_long) :: A_array, B_array, reuse_array
!       integer :: errorcode, reuse_int_cmat, reuse_int_reuse_mat
!       PetscErrorCode :: ierr
!       MatType :: mat_type
!       Mat :: temp_mat, temp_mat_reuse, temp_mat_compare
!       PetscScalar normy;
!       logical :: reuse_triggered_cmat, reuse_triggered_reuse_mat
!       type(c_ptr)  :: coefficients_ptr
!       type(tMat) :: reuse_mat_cpu
!       type(tMat), dimension(:), pointer :: reuse_submatrices_cpu
! #endif      
      ! ~~~~~~~~~~

      ! ~~~~~~~~~~
      ! Special case if we just want to return a gmres polynomial with the sparsity of the diagonal
      ! This is like a damped Jacobi
      ! ~~~~~~~~~~
if (poly_sparsity_order == 0) then

      call build_gmres_polynomial_inverse_0th_order_sparsity_newton(matrix, poly_order, &
               coefficients, cmat)     

      return
end if

! #if defined(PETSC_HAVE_KOKKOS)    

!       call MatGetType(matrix, mat_type, ierr)
!       if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
!             mat_type == MATAIJKOKKOS) then                  

!          A_array = matrix%v             
!          reuse_triggered_cmat = .NOT. PetscObjectIsNull(cmat) 
!          reuse_triggered_reuse_mat = .NOT. PetscObjectIsNull(reuse_mat) 
!          reuse_int_cmat = 0
!          if (reuse_triggered_cmat) then
!             reuse_int_cmat = 1
!             B_array = cmat%v
!          end if
!          reuse_int_reuse_mat = 0
!          if (reuse_triggered_reuse_mat) then
!             reuse_int_reuse_mat = 1
!          end if         
!          reuse_array = reuse_mat%v
!          coefficients_ptr = c_loc(coefficients)

!          ! call mat_mult_powers_share_sparsity_newton_kokkos(A_array, poly_order, poly_sparsity_order, &
!          !         coefficients_ptr, reuse_int_reuse_mat, reuse_array, reuse_int_cmat, B_array)
                         
!          reuse_mat%v = reuse_array
!          cmat%v = B_array

!          ! If debugging do a comparison between CPU and Kokkos results
!          if (kokkos_debug()) then

!             ! If we're doing reuse and debug, then we have to always output the result 
!             ! from the cpu version, as it will have coo preallocation structures set
!             ! They aren't copied over if you do a matcopy (or matconvert)
!             ! If we didn't do that the next time we come through this routine 
!             ! and try to call the cpu version with reuse, it will segfault
!             if (reuse_triggered_cmat) then
!                temp_mat = cmat
!                call MatConvert(cmat, MATSAME, MAT_INITIAL_MATRIX, temp_mat_compare, ierr)  
!             else
!                temp_mat_compare = cmat                         
!             end if            

!             ! Debug check if the CPU and Kokkos versions are the same
!             ! We send in an empty reuse_mat_cpu here always, as we can't pass through
!             ! the same one Kokkos uses as it now only gets out the non-local rows we need
!             ! (ie reuse_mat and reuse_mat_cpu are no longer the same size)
!             reuse_submatrices_cpu => null()
!             call mat_mult_powers_share_sparsity_newton_cpu(matrix, poly_order, poly_sparsity_order, &
!                      coefficients, reuse_mat_cpu, reuse_submatrices_cpu, temp_mat)
!             call destroy_matrix_reuse(reuse_mat_cpu, reuse_submatrices_cpu)         
                     
!             call MatConvert(temp_mat, MATSAME, MAT_INITIAL_MATRIX, &
!                         temp_mat_reuse, ierr)                        

!             call MatAXPYWrapper(temp_mat_reuse, -1d0, temp_mat_compare)
!             call MatNorm(temp_mat_reuse, NORM_FROBENIUS, normy, ierr)
!             ! There is floating point compute in these inverses, so we have to be a 
!             ! bit more tolerant to rounding differences
!             if (normy .gt. 1d-11 .OR. normy/=normy) then
!                !call MatFilter(temp_mat_reuse, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
!                !call MatView(temp_mat_reuse, PETSC_VIEWER_STDOUT_WORLD, ierr)
!                print *, "Kokkos and CPU versions of mat_mult_powers_share_sparsity do not match"

!                call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)  
!             end if
!             call MatDestroy(temp_mat_reuse, ierr)
!             if (.NOT. reuse_triggered_cmat) then
!                call MatDestroy(cmat, ierr)
!             else
!                call MatDestroy(temp_mat_compare, ierr)
!             end if
!             cmat = temp_mat
!          end if

!       else

!          call mat_mult_powers_share_sparsity_newton_cpu(matrix, poly_order, poly_sparsity_order, &
!                   coefficients, reuse_mat, reuse_submatrices, cmat)       

!       end if
! #else
      call mat_mult_powers_share_sparsity_newton_cpu(matrix, poly_order, poly_sparsity_order, &
                  coefficients, reuse_mat, reuse_submatrices, cmat)
!#endif         

      ! ~~~~~~~~~~
      
   end subroutine mat_mult_powers_share_sparsity_newton

!------------------------------------------------------------------------------------------------------------------------
   
   subroutine mat_mult_powers_share_sparsity_newton_cpu(matrix, poly_order, poly_sparsity_order, coefficients, &
                  reuse_mat, reuse_submatrices, cmat)

      ! Compute newton powers with the same sparsity
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), target, intent(in)                     :: matrix
      integer, intent(in)                                :: poly_order, poly_sparsity_order
      PetscReal, dimension(:, :), target, contiguous, intent(inout)    :: coefficients
      type(tMat), intent(inout)                          :: reuse_mat, cmat
      type(tMat), dimension(:), pointer, intent(inout)   :: reuse_submatrices
      
      PetscInt :: local_rows, local_cols, global_rows, global_cols
      PetscInt :: global_row_start, global_row_end_plus_one, row_index_into_submatrix
      PetscInt :: global_col_start, global_col_end_plus_one, n, ncols, ncols_two, ifree, max_nnzs
      PetscInt :: i_loc, j_loc, row_size, rows_ao, cols_ao, rows_ad, cols_ad, shift = 0, diag_index
      integer :: errorcode, match_counter, term
      integer :: comm_size
      PetscErrorCode :: ierr      
      integer, dimension(:), allocatable :: cols_index_one, cols_index_two
      PetscInt, dimension(:), allocatable :: col_indices_off_proc_array, ad_indices, cols
      PetscReal, dimension(:), allocatable :: vals
      type(tIS), dimension(1) :: col_indices, row_indices
      type(tMat) :: Ad, Ao, mat_sparsity_match, mat_product_save
      PetscInt, dimension(:), pointer :: colmap
      logical :: deallocate_submatrices = .FALSE.
      type(c_ptr) :: vals_c_ptr
      type(int_vec), dimension(:), allocatable :: symbolic_ones
      type(real_vec), dimension(:), allocatable :: symbolic_vals
      integer(c_long_long) A_array
      MPIU_Comm :: MPI_COMM_MATRIX
      PetscReal, dimension(:), allocatable :: vals_power_temp, vals_previous_power_temp, temp
      PetscInt, dimension(:), pointer :: submatrices_ia, submatrices_ja, cols_two_ptr, cols_ptr
      PetscReal, dimension(:), pointer :: vals_two_ptr, vals_ptr
      real(c_double), pointer :: submatrices_vals(:)
      logical :: reuse_triggered
      PetscBool :: symmetric = PETSC_FALSE, inodecompressed = PETSC_FALSE, done
      PetscInt, parameter :: one = 1, zero = 0
      logical :: output_first_complex, skip_add
      PetscReal :: square_sum
      integer, dimension(poly_order + 1, 2) :: status_output
      
      ! ~~~~~~~~~~  

      if (poly_sparsity_order .ge. size(coefficients)-1) then      
         print *, "Requested sparsity is greater than or equal to the order"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if      

      call PetscObjectGetComm(matrix, MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)

      ! Get the local sizes
      call MatGetLocalSize(matrix, local_rows, local_cols, ierr)
      call MatGetSize(matrix, global_rows, global_cols, ierr)
      ! This returns the global index of the local portion of the matrix
      call MatGetOwnershipRange(matrix, global_row_start, global_row_end_plus_one, ierr)  
      call MatGetOwnershipRangeColumn(matrix, global_col_start, global_col_end_plus_one, ierr)  

      reuse_triggered = .NOT. PetscObjectIsNull(cmat) 

      ! ~~~~~~~~~~
      ! Compute cmat for all powers up to poly_sparsity_order
      ! We have to be more careful here than in the monomial case
      ! In the mononomial case we just compute the matrix powers up to poly_sparsity_order
      ! and add them times the coefficients to cmat
      ! Here though we have to build the Newton basis polynomials
      ! As a rule, the value input here into cmat is correct up to the power
      ! of poly_sparsity_order, with all other terms being added in the fixed sparsity loops
      ! below
      !
      ! For complex conjugate roots, two terms are computed at a time, but if the sparsity order
      ! falls in between those two roots, only part of the output cmat is included
      ! Any remaining terms are output in either mat_sparsity_match or mat_product_save depending 
      ! on the case
      ! mat_sparsity_match has either temp or prod in it depending on the case
      ! mat_product_save only exists in some cases and stores prod from the previous term
      ! mat_sparsity_match and mat_product_save are always output with the sparsity of sparsity order
      !
      ! status_output is an array of length the number of roots
      ! and has a 1 in each position if that term has been added to the output
      ! It just helps us keep track of what has gone into cmat and what hasn't
      ! It breaks up complex conjugate pairs into the first root (i) which has the same power as prod
      !   tmp = 2 * a * prod
      !   p = p + 1/(a^2 + b^2) * tmp
      ! and the second root (i+1) which has the same power as prod * A:
      !   tmp = -A * prod
      !   p = p + 1/(a^2 + b^2) * tmp
      ! ~~~~~~~~~~

      output_first_complex = .FALSE.
      if (poly_sparsity_order == 1) then

         ! If we have a real first coefficient and a second complex
         ! we can't call build_gmres_polynomial_newton_inverse_1st_1st as it is only correct
         ! for valid coefficients up to 1st order (ie both real or both complex)
         if (coefficients(1,2) == 0d0 .AND. coefficients(2,2) /= 0d0) then

            call build_gmres_polynomial_newton_inverse_full(matrix, poly_order, coefficients, &
                  cmat, mat_sparsity_match, poly_sparsity_order, output_first_complex, &
                  status_output, mat_product_save)    

         ! Valid 1st order polynomial, so this case is easy
         else
         
            ! Duplicate & copy the matrix, but ensure there is a diagonal present
            call mat_duplicate_copy_plus_diag(matrix, reuse_triggered, cmat)  

            call build_gmres_polynomial_newton_inverse_1st_1st(matrix, 1, &
                     coefficients(1:poly_sparsity_order + 1, 1:2), &
                     cmat, mat_sparsity_match, &
                     status_output)    
         end if     
      else

         ! If we're any higher, then we build cmat up to that order
         ! But we have to be careful because the last root we want to explicitly
         ! build up to here (ie the power of the matrix given by sparsity_order)
         ! might be the first root of a complex conjugate pair
         call build_gmres_polynomial_newton_inverse_full(matrix, poly_order, &
                  coefficients(1:poly_sparsity_order + 1, 1:2), &
                  cmat, mat_sparsity_match, poly_sparsity_order, output_first_complex, &
                  status_output, mat_product_save)
      end if 
      
      ! We know we will never have non-zero locations outside of the highest constrained sparsity power 
      call MatSetOption(cmat, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE,  ierr)     
      call MatSetOption(cmat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr) 
      ! We know we are only going to insert local vals
      ! These options should turn off any reductions in the assembly
      call MatSetOption(cmat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)     
      
      ! ~~~~~~~~~~~~
      ! If we're in parallel we need to get the off-process rows of matrix that correspond
      ! to the columns of mat_sparsity_match
      ! We can therefore do the matmult for every constrained power locally with just that data
      ! ~~~~~~~~~~~~
      ! Have to double check comm_size /= 1 as we might be on a subcommunicator and we can't call
      ! MatMPIAIJGetSeqAIJ specifically if that's the case
      if (comm_size /= 1) then

         ! ~~~~
         ! Get the cols
         ! ~~~~
         call MatMPIAIJGetSeqAIJ(mat_sparsity_match, Ad, Ao, colmap, ierr)

         call MatGetSize(Ad, rows_ad, cols_ad, ierr)             
         ! We know the col size of Ao is the size of colmap, the number of non-zero offprocessor columns
         call MatGetSize(Ao, rows_ao, cols_ao, ierr)         

         ! For the column indices we need to take all the columns of mat_sparsity_match
         A_array = mat_sparsity_match%v

         ! These are the global indices of the columns we want
         allocate(col_indices_off_proc_array(cols_ad + cols_ao))
         allocate(ad_indices(cols_ad))
         ! Local rows (as global indices)
         do ifree = 1, cols_ad
            ad_indices(ifree) = global_row_start + ifree - 1
         end do

         ! col_indices_off_proc_array is now sorted, which are the global indices of the columns we want
         call merge_pre_sorted(ad_indices, colmap, col_indices_off_proc_array)
         deallocate(ad_indices)

         ! Create the sequential IS we want with the cols we want (written as global indices)
         call ISCreateGeneral(PETSC_COMM_SELF, cols_ad + cols_ao, &
                     col_indices_off_proc_array, PETSC_USE_POINTER, col_indices(1), ierr) 
         call ISCreateGeneral(PETSC_COMM_SELF, cols_ao, &
                     colmap, PETSC_USE_POINTER, row_indices(1), ierr)                      

         ! ~~~~~~~
         ! Now we can pull out the chunk of matrix that we need
         ! ~~~~~~~

         ! We need off-processor rows to compute matrix powers   
         ! Setting this is necessary to avoid an allreduce when calling createsubmatrices
         ! This will be reset to false after the call to createsubmatrices
         call MatSetOption(matrix, MAT_SUBMAT_SINGLEIS, PETSC_TRUE, ierr)       
         
         ! Now this will be doing comms to get the non-local rows we want
         ! But only including the columns of the local fixed sparsity, as we don't need all the 
         ! columns of the non-local entries unless we are doing a full matmatmult
         ! This returns a sequential matrix
         if (.NOT. PetscObjectIsNull(reuse_mat)) then
            reuse_submatrices(1) = reuse_mat
            call MatCreateSubMatrices(matrix, one, row_indices, col_indices, MAT_REUSE_MATRIX, reuse_submatrices, ierr)
         else
            call MatCreateSubMatrices(matrix, one, row_indices, col_indices, MAT_INITIAL_MATRIX, reuse_submatrices, ierr)
            reuse_mat = reuse_submatrices(1)
         end if
         row_size = size(col_indices_off_proc_array)
         call ISDestroy(col_indices(1), ierr)
         call ISDestroy(row_indices(1), ierr)

      ! Easy in serial as we have everything we neeed
      else

         Ad = mat_sparsity_match
         cols_ad = local_cols
         allocate(reuse_submatrices(1))
         deallocate_submatrices = .TRUE.
         reuse_submatrices(1) = matrix
         row_size = local_rows
         allocate(col_indices_off_proc_array(local_rows))
         do ifree = 1, local_rows
            col_indices_off_proc_array(ifree) = ifree-1
         end do
      end if   
      
      ! ~~~~~~~~~
      ! Now that we are here, reuse_submatrices(1) contains A^poly_sparsity_order with all of the rows
      ! that correspond to the non-zero columns of matrix
      ! ~~~~~~~~~      

      ! Have to get the max nnzs of the local and off-local rows we've just retrieved
      max_nnzs = 0
      do ifree = global_row_start, global_row_end_plus_one-1     
         call MatGetRow(matrix, ifree, ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
         if (ncols > max_nnzs) max_nnzs = ncols
         call MatRestoreRow(matrix, ifree, ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
      end do      
      if (comm_size /= 1) then
         do ifree = 1, cols_ao            
            call MatGetRow(reuse_submatrices(1), ifree-1, ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
            if (ncols > max_nnzs) max_nnzs = ncols
            call MatRestoreRow(reuse_submatrices(1), ifree-1, ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
         end do
      end if
      ! and also the sparsity power
      do ifree = global_row_start, global_row_end_plus_one-1     
         call MatGetRow(mat_sparsity_match, ifree, ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
         if (ncols > max_nnzs) max_nnzs = ncols
         call MatRestoreRow(mat_sparsity_match, ifree, ncols, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr)
      end do 
      
      ! ~~~~~~~~
      ! Get pointers to the sequential aij structure so we don't have to put critical regions
      ! around the matgetrow
      ! ~~~~~~~~
      call MatGetRowIJ(reuse_submatrices(1),shift,symmetric,inodecompressed,n,submatrices_ia,submatrices_ja,done,ierr) 
      if (.NOT. done) then
         print *, "Pointers not set in call to MatGetRowIJF"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      ! Returns the wrong size pointer and can break if that size goes negative??
      !call MatSeqAIJGetArrayF90(reuse_submatrices(1),submatrices_vals,ierr);
      A_array = reuse_submatrices(1)%v
      ! Now we must never overwrite the values in this pointer, and we must 
      ! never call restore on it, see comment on top of the commented out
      ! MatSeqAIJRestoreArray below
      call MatSeqAIJGetArrayF90_mine(A_array, vals_c_ptr)
      call c_f_pointer(vals_c_ptr, submatrices_vals, shape=[size(submatrices_ja)])
      
      ! ~~~~~~~~~~
      
      allocate(cols(max_nnzs))
      allocate(vals(max_nnzs))
      allocate(vals_power_temp(max_nnzs))
      allocate(vals_previous_power_temp(max_nnzs))
      allocate(temp(max_nnzs))
      allocate(cols_index_one(max_nnzs))
      allocate(cols_index_two(max_nnzs))      

      ! ~~~~~~~~~~~~
      ! From here we now have cmat with the correct values up to the power poly_sparsity_order
      ! and hence we want to add in the sparsity constrained powers
      ! ~~~~~~~~~~~~
      
      ! Now go through and compute the sum of the matrix powers
      ! We're doing row-wise matmatmults here assuming the fixed sparsity
      ! We exploit the fact that the subsequent matrix powers can be done
      ! one row at a time, so we only have to retrieve the needed vals from mat_sparsity_match once
      do i_loc = 1, local_rows 
                          
         ! Get the row of mat_sparsity_match
         call MatGetRow(mat_sparsity_match, i_loc - 1 + global_row_start, ncols_two, &
                  cols_ptr, vals_ptr, ierr)
         ! Copying here because mat_sparsity_match and matrix are often the same matrix
         ! and hence we can only have one active matgetrow
         ncols = ncols_two
         cols(1:ncols) = cols_ptr(1:ncols)
         vals(1:ncols) = vals_ptr(1:ncols)
         call MatRestoreRow(mat_sparsity_match, i_loc - 1 + global_row_start, ncols_two, &
                  cols_ptr, vals_ptr, ierr)   
         diag_index = -1
         ! Find the diagonal index in this row     
         do j_loc = 1, ncols
            if (cols(j_loc) == i_loc - 1 + global_row_start) then
               diag_index = j_loc
               exit
            end if
         end do 

         ! This is just a symbolic for the set of rows given in cols
         ! Let's just do all the column matching and extraction of the values once
            
         ! Allocate some space to store the matching indices
         allocate(symbolic_ones(ncols))
         allocate(symbolic_vals(ncols))
         row_index_into_submatrix = 1

         ! This is a row-wise product
         do j_loc = 1, ncols

            ! If we're trying to access a local row in matrix
            if (cols(j_loc) .ge. global_row_start .AND. cols(j_loc) < global_row_end_plus_one) then

               call MatGetRow(matrix, cols(j_loc), ncols_two, &
                        cols_two_ptr, vals_two_ptr, ierr)

            ! If we're trying to access a non-local row in matrix
            else

               ! this is local row index we want into reuse_submatrices(1) (as row_indices used to extract are just colmap)  
               ! We know cols is sorted, so every non-local index will be greater than the last one
               ! (it's just that cols could have some local ones between different non-local)
               ! colmap is also sorted and we know every single non-local entry in cols(j_loc) is in colmap
               do while (row_index_into_submatrix .le. cols_ao .AND. colmap(row_index_into_submatrix) .lt. cols(j_loc))
                  row_index_into_submatrix = row_index_into_submatrix + 1
               end do

               ! This is the number of columns
               ncols_two = submatrices_ia(row_index_into_submatrix+1) - submatrices_ia(row_index_into_submatrix)
               allocate(cols_two_ptr(ncols_two))
               ! This is the local column indices in reuse_submatrices(1)
               cols_two_ptr = submatrices_ja(submatrices_ia(row_index_into_submatrix)+1:submatrices_ia(row_index_into_submatrix+1))
               ! Because col_indices_off_proc_array (and hence the column indices in reuse_submatrices(1) is sorted, 
               ! then cols_two_ptr contains the sorted global column indices
               cols_two_ptr = col_indices_off_proc_array(cols_two_ptr+1)

               ! This is the values
               vals_two_ptr => &
                submatrices_vals(submatrices_ia(row_index_into_submatrix)+1:submatrices_ia(row_index_into_submatrix+1))
            end if
            
            ! Search for the matching column
            ! We're intersecting the global column indices of mat_sparsity_match (cols) and matrix (cols_two_ptr)
            call intersect_pre_sorted_indices_only(cols(1:ncols), cols_two_ptr, cols_index_one, cols_index_two, match_counter)      
            
            ! Don't need to do anything if we have no matches
            if (match_counter == 0) then 
               ! Store that we can skip this entry
               symbolic_ones(j_loc)%ptr => null()
               symbolic_vals(j_loc)%ptr => null()                        
            else

               ! These are the matching local column indices for this row of mat_sparsity_match
               allocate(symbolic_ones(j_loc)%ptr(match_counter))
               symbolic_ones(j_loc)%ptr = cols_index_one(1:match_counter)

               ! These are the matching values of matrix
               allocate(symbolic_vals(j_loc)%ptr(match_counter))
               symbolic_vals(j_loc)%ptr = vals_two_ptr(cols_index_two(1:match_counter)) 
            end if   
            
            ! Restore local row of matrix
            if (cols(j_loc) .ge. global_row_start .AND. cols(j_loc) < global_row_end_plus_one) then
               call MatRestoreRow(matrix, cols(j_loc), ncols_two, &
                        cols_two_ptr, vals_two_ptr, ierr)
            else
               deallocate(cols_two_ptr)
            end if            
         end do
         
         ! Start with the values of mat_sparsity_match in it
         vals_previous_power_temp(1:ncols) = vals(1:ncols)
                     
         ! Loop over any matrix powers
         ! vals_power_temp stores the prod for this row, and we update this as we go through 
         ! the term loop
         term = poly_sparsity_order + 1
         skip_add = .FALSE.
         ! If the fixed sparsity root is the second of a complex pair, we start one term earlier
         ! so that we can compute the correct part of the fixed sparsity product, we just make sure not to add
         ! anything to cmat as it is already correct up to the fixed sparsity order
         if (coefficients(term,2) /= 0d0 .AND. .NOT. output_first_complex) then
            term = term - 1
            skip_add = .TRUE.
         end if        

         ! This loop skips the last coefficient
         do while (term .le. size(coefficients, 1) - 1)

            ! If real
            if (coefficients(term,2) == 0d0) then

               ! Skips eigenvalues that are numerically zero - see 
               ! the comment in calculate_gmres_polynomial_roots_newton                
               if (abs(coefficients(term,1)) < 1e-12) then
                  term = term + 1
                  cycle
               end if

               ! ~~~~~~~~~~~
               ! Now can add the value to our matrix
               ! ~~~~~~~~~~~               
               if (ncols /= 0 .AND. status_output(term, 1) /= 1) then
                  call MatSetValues(cmat, one, [global_row_start + i_loc-1], ncols, cols, &
                        1d0/coefficients(term, 1) * vals_previous_power_temp(1:ncols), ADD_VALUES, ierr)   
               end if          
               
               ! Initialize with previous product before the A*prod subtraction
               vals_power_temp(1:ncols) = vals_previous_power_temp(1:ncols)        
               
               ! This is the 1/theta_i * A * prod but where A * prod has fixed sparsity
               do j_loc = 1, ncols

                  ! If we have no matching columns cycle this row
                  if (.NOT. associated(symbolic_ones(j_loc)%ptr)) cycle

                  ! symbolic_vals(j_loc)%ptr has the matching values of A in it
                  vals_power_temp(symbolic_ones(j_loc)%ptr) = vals_power_temp(symbolic_ones(j_loc)%ptr) - &
                           1d0/coefficients(term, 1) * &
                           symbolic_vals(j_loc)%ptr * vals_previous_power_temp(j_loc)
               end do

               term = term + 1

            ! If complex
            else

               ! Skips eigenvalues that are numerically zero - see 
               ! the comment in calculate_gmres_polynomial_roots_newton                
               if (coefficients(term,1)**2 + coefficients(term,2)**2 < 1e-12) then
                  term = term + 2
                  cycle
               end if

               square_sum = 1d0/(coefficients(term,1)**2 + coefficients(term,2)**2)

               ! If our fixed sparsity order falls on the first of a complex conjugate pair
               if (.NOT. skip_add) then

                  ! We skip the 2 * a * prod from the first root of a complex pair if that has already
                  ! been included in the cmat from build_gmres_polynomial_newton_inverse_full
                  if (status_output(term, 2) /= 1) then
                     temp(1:ncols) = 2 * coefficients(term, 1) * vals_previous_power_temp(1:ncols)
                  else
                     temp(1:ncols) = 0d0
                  end if

                  ! This is the -A * prod
                  do j_loc = 1, ncols

                     ! If we have no matching columns cycle this row
                     if (.NOT. associated(symbolic_ones(j_loc)%ptr)) cycle

                     ! symbolic_vals(j_loc)%ptr has the matching values of A in it
                     temp(symbolic_ones(j_loc)%ptr) = temp(symbolic_ones(j_loc)%ptr) - &
                              symbolic_vals(j_loc)%ptr * vals_previous_power_temp(j_loc)
                  end do           

                  ! This is the p = p + 1/(a^2 + b^2) * temp
                  if (ncols /= 0) then
                     call MatSetValues(cmat, one, [global_row_start + i_loc-1], ncols, cols, &
                           square_sum * temp(1:ncols), ADD_VALUES, ierr)   
                  end if       

                  ! Here we need to go back in and ensure 2 *a * prod is in temp if we skipped it 
                  ! above. We know it is already in cmat, but it has to be in temp when we 
                  ! do the next product 
                  if (status_output(term, 2) == 1) then
                     if (output_first_complex) then
                        temp(1:ncols) = temp(1:ncols) + 2d0 * coefficients(term, 1) * vals_previous_power_temp(1:ncols)
                     end if                     
                  end if                  

               ! If our fixed sparsity order falls on the second of a complex conjugate pair
               else

                  ! In this case we have already included both 2*a*prod - A * prod into cmat
                  ! But we still have to compute the product for the next term
                  ! The problem here is that mat_sparsity_match has temp in it in this case, not 
                  ! the old prod from whatever the previous loop is
                  ! In that case build_gmres_polynomial_newton_inverse_full also outputs
                  ! mat_product_save which is the old value of prod but with the sparsity of 
                  ! mat_sparsity_match (with zeros if needed)
                  
                  ! This case only occurs once for each row, so once we've hit this 
                  ! we will always have our correct prod 
                  skip_add = .FALSE.
                  ! temp is output into mat_sparsity_match in this case
                  temp(1:ncols) = vals_previous_power_temp(1:ncols)

                  ! If sparsity order is 1, the previous product will have been the identity
                  ! and we don't output it into mat_product_save because that is a trivial case 
                  ! we can do ourselves
                  if (term == 1) then
                     vals_previous_power_temp(1:ncols) = 0d0
                     if (diag_index /= -1) then
                        vals_previous_power_temp(diag_index) = 1d0
                     end if

                  ! In the case the mat_product_save is not the identity, we need to pull it's value out
                  else

                     call MatGetRow(mat_product_save, i_loc - 1 + global_row_start, ncols_two, &
                              cols_two_ptr, vals_two_ptr, ierr)
                     
                     ! We have guaranteed in the build_gmres_polynomial_newton_inverse_full
                     ! version that mat_product_save has fixed sparsity
                     vals_previous_power_temp(1:ncols_two) = vals_two_ptr(1:ncols_two)
                     
                     call MatRestoreRow(mat_product_save, i_loc - 1 + global_row_start, ncols_two, &
                              cols_two_ptr, vals_two_ptr, ierr)                     
                  end if
               end if

               ! Now we compute the next product
               if (term .le. size(coefficients, 1)- 2) then

                  vals_power_temp(1:ncols) = vals_previous_power_temp(1:ncols)

                  ! This is prod = prod - 1/(a^2 + b^2) * A * temp
                  do j_loc = 1, ncols

                     ! If we have no matching columns cycle this row
                     if (.NOT. associated(symbolic_ones(j_loc)%ptr)) cycle

                     ! symbolic_vals(j_loc)%ptr has the matching values of A in it
                     vals_power_temp(symbolic_ones(j_loc)%ptr) = vals_power_temp(symbolic_ones(j_loc)%ptr) - &
                              square_sum * &
                              symbolic_vals(j_loc)%ptr * temp(j_loc)
                  end do                  
               end if

               term = term + 2

            end if

            ! This should now have the value of prod in it
            vals_previous_power_temp(1:ncols) = vals_power_temp(1:ncols)
         end do    

         ! Final step if last root is real
         if (coefficients(size(coefficients, 1),2) == 0d0) then
            if (ncols /= 0 .AND. abs(coefficients(size(coefficients, 1),1)) > 1e-12) then
               call MatSetValues(cmat, one, [global_row_start + i_loc-1], ncols, cols, &
                     1d0/coefficients(size(coefficients, 1), 1) * vals_power_temp(1:ncols), ADD_VALUES, ierr)
            end if             
         end if

         ! ~~~~~~~~~~~~~~~

         do j_loc = 1, ncols
            if (associated(symbolic_ones(j_loc)%ptr)) then
               deallocate(symbolic_ones(j_loc)%ptr)
               deallocate(symbolic_vals(j_loc)%ptr)
            end if      
         end do  
         deallocate(symbolic_vals, symbolic_ones)  
      end do

      call MatRestoreRowIJ(reuse_submatrices(1),shift,symmetric,inodecompressed,n,submatrices_ia,submatrices_ja,done,ierr) 
      ! We very deliberately don't call restorearray here!
      ! There is no matseqaijgetarrayread or matseqaijrestorearrayread in Fortran
      ! Those routines don't increment the PetscObjectStateGet which tells petsc
      ! the mat has changed. Hence above we directly access the data pointer with 
      ! a call to MatSeqAIJGetArrayF90_mine and then never write into it
      ! If we call the restorearrayf90, that does increment the object state
      ! even though we only read from the array
      ! That would mean if we pass in a pc->pmat for example, just setting up a pc
      ! would trigger petsc setting up the pc on every iteration of the pc
      ! call MatSeqAIJRestoreArray(reuse_submatrices(1),submatrices_vals,ierr);

      ! ~~~~~~~~~~~

      ! Do the assembly, should need zero reductions in this given we've set the 
      ! flags above
      call MatAssemblyBegin(cmat, MAT_FINAL_ASSEMBLY, ierr)

      ! Delete temporaries
      call MatDestroy(mat_sparsity_match, ierr)
      call MatDestroy(mat_product_save, ierr)
      if (deallocate_submatrices) then
         deallocate(reuse_submatrices)
         reuse_submatrices => null()
      end if

      deallocate(col_indices_off_proc_array)
      deallocate(cols, vals, vals_power_temp, vals_previous_power_temp, temp, cols_index_one, cols_index_two)

      ! Finish assembly
      call MatAssemblyEnd(cmat, MAT_FINAL_ASSEMBLY, ierr) 

         
   end subroutine mat_mult_powers_share_sparsity_newton_cpu   

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
      integer :: comm_size, errorcode
      PetscErrorCode :: ierr      
      MPIU_Comm :: MPI_COMM_MATRIX
      type(mat_ctxtype), pointer :: mat_ctx=>null()
      logical :: reuse_triggered

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

      reuse_triggered = .NOT. PetscObjectIsNull(inv_matrix)         

      ! ~~~~~~~~~~~~
      ! If we're here then we want an assembled approximate inverse
      ! ~~~~~~~~~~~~         

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
         
         call build_gmres_polynomial_newton_inverse_1st_1st(matrix, &
               poly_order, coefficients, inv_matrix)

         ! Then just return
         return

      end if

      ! If we're constraining sparsity we've built a custom matrix-powers that assumes fixed sparsity
      if (poly_sparsity_order < poly_order) then    

         ! This routine is a custom one that builds our matrix powers and assumes fixed sparsity
         ! so that it doen't have to do much comms
         call mat_mult_powers_share_sparsity_newton(matrix, poly_order, poly_sparsity_order, coefficients, &
                  reuse_mat, reuse_submatrices, inv_matrix)

         ! ! Then just return
         return         
         
      end if

      ! ~~~~~~~~~~
      ! We are only here if we don't constrain_sparsity
      ! ~~~~~~~~~~
      call build_gmres_polynomial_newton_inverse_full(matrix, poly_order, coefficients, &
                  inv_matrix)
      

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

   subroutine build_gmres_polynomial_inverse_0th_order_sparsity_newton(matrix, poly_order, coefficients, &
                  inv_matrix)

      ! Specific inverse with 0th order sparsity

      ! ~~~~~~
      type(tMat), intent(in)                            :: matrix
      integer, intent(in)                               :: poly_order
      PetscReal, dimension(:, :), target, contiguous, intent(inout)    :: coefficients
      type(tMat), intent(inout)                         :: inv_matrix

      ! Local variables
      integer :: i
      PetscErrorCode :: ierr      
      logical :: reuse_triggered
      type(tVec) :: inv_vec, diag_vec, product_vec, temp_vec_A, one_vec, temp_vec_two
      ! ~~~~~~

      reuse_triggered = .NOT. PetscObjectIsNull(inv_matrix)

       ! Our matrix has to be square
      call MatCreateVecs(matrix, product_vec, diag_vec, ierr)
      call MatGetDiagonal(matrix, diag_vec, ierr)

      if (.NOT. reuse_triggered) then
         call VecDuplicate(diag_vec, inv_vec, ierr)
      else
         call MatDiagonalGetDiagonal(inv_matrix, inv_vec, ierr)
      end if
      call VecDuplicate(diag_vec, temp_vec_A, ierr)
      call VecDuplicate(diag_vec, one_vec, ierr)
      call VecDuplicate(diag_vec, temp_vec_two, ierr)
      
      ! Set to zero as we add to it
      call VecSet(inv_vec, 0d0, ierr) 
      ! We start with an identity in product_vec    
      call VecSet(product_vec, 1d0, ierr)
      call VecSet(one_vec, 1d0, ierr)

      i = 1
      do while (i .le. size(coefficients, 1) - 1)

         ! temp_vec_A is going to store things with the sparsity of A
         call VecCopy(diag_vec, temp_vec_A, ierr)     

         ! If real this is easy
         if (coefficients(i,2) == 0d0) then

            ! Skips eigenvalues that are numerically zero - see 
            ! the comment in calculate_gmres_polynomial_roots_newton 
            if (abs(coefficients(i,1)) < 1e-12) then
               i = i + 1
               cycle
            end if        

            call VecAXPY(inv_vec, 1d0/coefficients(i,1), product_vec, ierr)

            ! temp_vec_A = A_ff/theta_k       
            call VecScale(temp_vec_A, -1d0/coefficients(i,1), ierr)
            ! temp_vec_A = I - A_ff/theta_k
            call VecAXPY(temp_vec_A, 1d0, one_vec, ierr)
            
            ! product_vec = product_vec * temp_vec_A
            call VecPointwiseMult(product_vec, product_vec, temp_vec_A, ierr)   
            
            i = i + 1

         ! Complex 
         else

            ! Skips eigenvalues that are numerically zero
            if (coefficients(i,1)**2 + coefficients(i,2)**2 < 1e-12) then
               i = i + 2
               cycle
            end if

            ! Compute 2a I - A
            ! temp_vec_A = -A    
            call VecScale(temp_vec_A, -1d0, ierr)
            ! temp_vec_A = 2a I - A_ff
            call VecAXPY(temp_vec_A, 2d0 * coefficients(i,1), one_vec, ierr) 
            ! temp_vec_A = (2a I - A_ff)/(a^2 + b^2)
            call VecScale(temp_vec_A, 1d0/(coefficients(i,1)**2 + coefficients(i,2)**2), ierr) 

            ! temp_vec_two = temp_vec_A * product_vec
            call VecPointwiseMult(temp_vec_two, temp_vec_A, product_vec, ierr)   
            call VecAXPY(inv_vec, 1d0, temp_vec_two, ierr)         

            if (i .le. size(coefficients, 1) - 2) then
               ! temp_vec_two = A * temp_vec_two
               call VecPointwiseMult(temp_vec_two, diag_vec, temp_vec_two, ierr) 
               call VecAXPY(product_vec, -1d0, temp_vec_two, ierr)
            end if

            ! Skip two evals
            i = i + 2

         end if       
      end do

      ! Final step if last root is real
      if (coefficients(size(coefficients,1),2) == 0d0) then
         ! Add in the final term multiplied by 1/theta_poly_order

         ! Skips eigenvalues that are numerically zero
         if (abs(coefficients(size(coefficients, 1),1)) > 1e-12) then      
            call VecAXPY(inv_vec, 1d0/coefficients(size(coefficients, 1),1), product_vec, ierr)  
         end if       
      end if

      ! We may be reusing with the same sparsity
      if (.NOT. reuse_triggered) then
         ! The matrix takes ownership of inv_vec and increases ref counter
         call MatCreateDiagonal(inv_vec, inv_matrix, ierr)
         call VecDestroy(inv_vec, ierr)
      else
         call MatDiagonalRestoreDiagonal(inv_matrix, inv_vec, ierr)
      end if  

      call VecDestroy(diag_vec, ierr)
      call VecDestroy(product_vec, ierr)     
      call VecDestroy(temp_vec_A, ierr)   
      call VecDestroy(one_vec, ierr)                      
      call VecDestroy(temp_vec_two, ierr)

   end subroutine build_gmres_polynomial_inverse_0th_order_sparsity_newton     

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine build_gmres_polynomial_newton_inverse_1st_1st(matrix, poly_order, coefficients, &
                  inv_matrix, mat_prod_or_temp, status_output)

      ! Specific 1st order with 1st order sparsity

      ! ~~~~~~
      type(tMat), intent(in)                            :: matrix
      integer, intent(in)                               :: poly_order
      PetscReal, dimension(:, :), target, contiguous, intent(inout)    :: coefficients
      type(tMat), intent(inout)                         :: inv_matrix
      type(tMat), intent(inout), optional               :: mat_prod_or_temp
      integer, dimension(:, :), intent(inout), optional :: status_output

      ! Local variables
      PetscErrorCode :: ierr      
      logical :: reuse_triggered, output_product
      PetscReal :: square_sum

      ! ~~~~~~      

      reuse_triggered = .NOT. PetscObjectIsNull(inv_matrix)     
      output_product = present(mat_prod_or_temp)    

      ! Flags to prevent reductions when assembling (there are assembles in the shift)
      call MatSetOption(inv_matrix, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr) 
      call MatSetOption(inv_matrix, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE,  ierr)     
      call MatSetOption(inv_matrix, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)

      if (output_product) then
         status_output = 0
      end if

      ! We only have two coefficients, so they are either both real or complex conjugates
      ! If real
      if (coefficients(1,2) == 0d0) then

         ! Have to be careful here, as we may be first order, but the second eigenvaule
         ! might have been set to zero thanks to the rank reducing solve 
         ! So we just check if the second real part is zero and if it is
         ! we just compute a 0th order inverse - annoyingly we can't call 
         ! build_gmres_polynomial_newton_inverse_0th_order as that builds a MATDIAGONAL
         ! and in the tests there is a problem where we reuse the sparsity, in the first
         ! solve we don't have a zero coefficient but in the second solve we do
         ! So the mat type needs to remain consistent
         ! This can't happen in the complex case
         if (abs(coefficients(2,1)) < 1e-12) then

            ! Set to zero
            call MatScale(inv_matrix, 0d0, ierr)

            ! Tricky case here as we want to pass out the identity with the 
            ! sparsity of A
            if (output_product) then
               call MatConvert(inv_matrix, MATSAME, MAT_INITIAL_MATRIX, mat_prod_or_temp, ierr)  
               call MatShift(mat_prod_or_temp, 1d0, ierr)
               status_output(1:2, 1) = 1
            end if                

            ! Then add in the 0th order inverse
            call MatShift(inv_matrix, 1d0/coefficients(1,1), ierr)       
            
            ! Then just return
            return  
         end if

         ! Could just compute the equivalent mononomial here to save some flops
         ! but the whole point to doing the Newton form is to avoid the 
         ! theta_1 * theta_2 that would result

         ! result = -A_ff/theta_1
         call MatScale(inv_matrix, -1d0/(coefficients(1, 1)), ierr)
         ! result = I -A_ff/theta_1
         call MatShift(inv_matrix, 1d0, ierr) 
         ! If we're doing this as part of fixed sparsity multiply, 
         ! we need to return mat_prod_or_temp
         if (output_product) then
            call MatConvert(inv_matrix, MATSAME, MAT_INITIAL_MATRIX, mat_prod_or_temp, ierr)  
         end if

         ! result = 1/theta_2 * (I -A_ff/theta_1)
         call MatScale(inv_matrix, 1d0/(coefficients(2, 1)), ierr)      

         ! result = 1/theta_1 + 1/theta_2 * (I -A_ff/theta_1)
         ! Don't need an assemble as there is one called in this
         call MatShift(inv_matrix, 1d0/(coefficients(1, 1)), ierr)     
         
         if (output_product) then
            status_output(1:2, 1) = 1
         end if

      ! Complex conjugate roots, a +- ib
      else
         ! a^2 + b^2
         square_sum = coefficients(1,1)**2 + coefficients(1,2)**2

         ! Complex conjugate roots
         ! result = -A_ff
         call MatScale(inv_matrix, -1d0, ierr)
         ! result = 2a I - A_ff
         ! Don't need an assemble as there is one called in this
         call MatShift(inv_matrix, 2d0 * coefficients(1,1), ierr)      
         ! If we're doing this as part of fixed sparsity multiply, 
         ! we need to return mat_prod_or_temp         
         if (output_product) then
            call MatConvert(inv_matrix, MATSAME, MAT_INITIAL_MATRIX, mat_prod_or_temp, ierr)  
         end if      
         ! result = 2a I - A_ff/(a^2 + b^2)
         call MatScale(inv_matrix, 1d0/square_sum, ierr)

         if (output_product) then
            status_output(1:2, 2) = 1
         end if
      end if               

   end subroutine build_gmres_polynomial_newton_inverse_1st_1st     
   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine build_gmres_polynomial_newton_inverse_full(matrix, poly_order, coefficients, &
                  inv_matrix, mat_prod_or_temp, poly_sparsity_order, output_first_complex, &
                  status_output, mat_product_save)

      ! No constrained sparsity by default
      ! If you pass in mat_prod_or_temp, poly_sparsity_order, output_first_complex, status_output and 
      ! mat_product_save
      ! then it will build part of the terms, up to poly_sparsity_order, and return the product
      ! in mat_prod_or_temp and mat_product_save that you need to compute the rest of the fixed sparsity terms

      ! ~~~~~~
      type(tMat), intent(in)                            :: matrix
      integer, intent(in)                               :: poly_order
      PetscReal, dimension(:, :), target, contiguous, intent(inout)    :: coefficients
      type(tMat), intent(inout)                         :: inv_matrix
      type(tMat), intent(inout), optional               :: mat_prod_or_temp, mat_product_save
      integer, intent(in), optional                     :: poly_sparsity_order
      logical, intent(inout), optional                  :: output_first_complex
      integer, dimension(:, :), intent(inout), optional :: status_output

      ! Local variables
      PetscErrorCode :: ierr      
      logical :: reuse_triggered, output_product, first_complex
      integer :: i, i_sparse, sparsity_order
      type(tMat) :: mat_product, temp_mat_A, temp_mat_two, temp_mat_three, mat_product_k_plus_1
      PetscReal :: square_sum, a_coeff

      ! ~~~~~~      

      reuse_triggered = .NOT. PetscObjectIsNull(inv_matrix)  
      output_product = present(mat_prod_or_temp)

      if (.NOT. reuse_triggered) then
         ! Duplicate & copy the matrix, but ensure there is a diagonal present
         call mat_duplicate_copy_plus_diag(matrix, .FALSE., inv_matrix)
      end if      

      ! Set to zero as we add in each product of terms
      call MatScale(inv_matrix, 0d0, ierr)

      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(inv_matrix, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)  

      ! We start with an identity in mat_product
      call generate_identity(matrix, mat_product)
      if (output_product) then
         status_output = 0
      end if
      sparsity_order = poly_order
      if (present(poly_sparsity_order)) then
         sparsity_order = poly_sparsity_order
      end if

      ! If we're going to output the product as part of a fixed sparsity multiply,
      ! we may be asking to constrain the sparsity to a power in between order and order + 2
      ! if there is a complex root at poly_sparsity_order
      ! ie if we have roots (theta_1^r, theta_2^c, theta_3^c, theta_4^r) 
      ! where ^r means a purely real root and ^c means a complex root 
      ! want poly_sparsity_order = 1, we can't process all the way up to theta_3^c as that would 
      ! compute up to an A^2 term which is beyond our sparsity constraint

      ! i_sparse tells us how many roots we are going to process
      ! Normally this would just be size(coefficients, 1) and the loop below goes up 
      ! to size(coefficients, 1) - 1. The last real root gets its final term added outside the loop
      ! and if the last root is complex then we only have to hit the first of the pair in the loop
      !
      ! If we have fixed sparsity:
      !
      ! if the fixed sparsity root is real then we want to set i_sparse to poly_sparsity_order+1
      ! so we hit the roots up to poly_sparsity_order in the loop and then we take care of the 
      ! poly_sparsity_order + 1 root outside the loop
      !
      ! if the fixed sparsity root is complex but poly_sparsity_order + 1 hits the second of the pair
      !    then we only need to set i_sparse to poly_sparsity_order + 1 so we only hit the first
      !    pair in the loop below
      ! 
      ! if the fixed sparsity root is complex but poly_sparsity_order + 1 hits the first of the pair
      !    then we need to set i_sparse to poly_sparsity_order + 2
      !    otherwise we would never hit the first pair

      i_sparse = size(coefficients, 1)
      first_complex = .FALSE.

      if (output_product) then

         output_first_complex = .FALSE.
         if (output_product) then
            i_sparse = sparsity_order + 1

            ! If the last root is real we don't have to do anything
            if (coefficients(i_sparse,2) /= 0d0) then

               ! If the one before is real, then we know we're on the first
               if (coefficients(i_sparse-1,2) == 0d0) then
                  output_first_complex = .TRUE.
                  ! See discussion above
                  i_sparse = i_sparse + 1

               ! If the one before is complex
               else

                  ! Check if the distance between the fixed sparsity root and the one before
                  ! If > zero then they are not complex conjugates and hence we are on the first of the pair         
                  if (abs(coefficients(i_sparse,1) - coefficients(i_sparse-1,1))/coefficients(i_sparse,1) > 1e-14 .AND. &
                        abs(coefficients(i_sparse,2) + coefficients(i_sparse-1,2))/coefficients(i_sparse,2) > 1e-14) then
                     output_first_complex = .TRUE.
                     i_sparse = i_sparse + 1
                  end if            
               end if
            end if
         end if 
         first_complex = output_first_complex
      end if

      ! ~~~~~~~~~~~~
      ! Iterate over the i
      ! This is basically the same as the MF application but we have to build the powers
      ! ~~~~~~~~~~~~      
      i = 1
      ! Loop through to one fewer than the number of roots
      ! We're always building up the next product
      do while (i .le. i_sparse - 1)

         ! Duplicate & copy the matrix, but ensure there is a diagonal present
         ! temp_mat_A is going to store things with the sparsity of A
         if (PetscObjectIsNull(temp_mat_A)) then
            call mat_duplicate_copy_plus_diag(matrix, .FALSE., temp_mat_A)     
         else
            ! Can reuse the sparsity 
            call mat_duplicate_copy_plus_diag(matrix, .TRUE., temp_mat_A)     
         end if         

         ! If real this is easy
         if (coefficients(i,2) == 0d0) then

            ! Skips eigenvalues that are numerically zero
            ! We still compute the entries as as zero because we need the sparsity
            ! to be correct for the next iteration
            if (abs(coefficients(i,1)) < 1e-12) then
               square_sum = 0
            else
               square_sum = 1d0/coefficients(i,1)
            end if        

            ! Then add the scaled version of each product
            if (i == 1) then
               ! If i == 1 then we know mat_product is identity so we can do it directly
               call MatShift(inv_matrix, 1d0/coefficients(i,1), ierr)  
            else
               if (reuse_triggered) then
                  ! If doing reuse we know our nonzeros are a subset
                  call MatAXPY(inv_matrix, square_sum, mat_product, SUBSET_NONZERO_PATTERN, ierr)
               else
                  ! Have to use the DIFFERENT_NONZERO_PATTERN here
                  call MatAXPYWrapper(inv_matrix, square_sum, mat_product)
               end if
            end if
            if (output_product) status_output(i, 1) = 1

            ! temp_mat_A = A_ff/theta_k       
            call MatScale(temp_mat_A, -square_sum, ierr)
            ! temp_mat_A = I - A_ff/theta_k
            call MatShift(temp_mat_A, 1d0, ierr)    
            
            ! mat_product_k_plus_1 = mat_product * temp_mat_A
            if (i == 1) then
               ! If i == 1 then we know mat_product is identity so we can just copy
               call MatDestroy(mat_product, ierr)
               call MatConvert(temp_mat_A, MATSAME, MAT_INITIAL_MATRIX, mat_product, ierr)  
            else
               call MatMatMult(temp_mat_A, mat_product, &
                     MAT_INITIAL_MATRIX, 1.5d0, mat_product_k_plus_1, ierr)      
               call MatDestroy(mat_product, ierr)  
               mat_product = mat_product_k_plus_1  
            end if
            
            ! We copy out the last product if we're doing this as part of a fixed sparsity multiply
            if (output_product .AND. i == i_sparse - 1) then
               call MatConvert(mat_product, MATSAME, MAT_INITIAL_MATRIX, mat_prod_or_temp, ierr)  
            end if
            
            i = i + 1

         ! Complex 
         else

            ! Skips eigenvalues that are numerically zero
            if (coefficients(i,1)**2 + coefficients(i,2)**2 < 1e-12) then
               square_sum = 0
               a_coeff = 0
             else  
               square_sum = 1d0/(coefficients(i,1)**2 + coefficients(i,2)**2)
               a_coeff = 2d0 * coefficients(i,1)
            end if

            ! If our fixed sparsity root is the first of a complex conjugate pair
            ! We want to pass out mat_product and only add that to inv_matrix
            ! This is equivalent to only part of tmp on Line 9 of Loe
            ! The fixed sparsity loop will then finish the tmp with the term -A * prod/(a^2+b^2)
            ! as this is the part that would increase the sparsity beyond poly_sparsity_order            
            if (i == sparsity_order + 1 .AND. first_complex) then

               ! Copy mat_product into temp_mat_two
               call MatConvert(mat_product, MATSAME, MAT_INITIAL_MATRIX, temp_mat_two, ierr)

               ! temp_mat_two = 2a * mat_product
               call MatScale(temp_mat_two, a_coeff, ierr)   
               if (output_product) status_output(i, 2) = 1

               ! We copy out the last part of the product if we're doing this as part of a fixed sparsity multiply
               if (output_product .AND. i > i_sparse - 2) then
                  call MatConvert(mat_product, MATSAME, MAT_INITIAL_MATRIX, mat_prod_or_temp, ierr)  
               end if  
               
            ! Just do the normal loop
            else

               ! temp_mat_A = -A    
               call MatScale(temp_mat_A, -1d0, ierr)
               ! temp_mat_A = 2a I - A_ff
               call MatShift(temp_mat_A, a_coeff, ierr)   
               if (output_product) then
                  status_output(i, 2) = 1
                  status_output(i+1, 2) = 1
               end if

               if (i == 1) then
                  ! If i == 1 then we know mat_product is identity so we can do it directly
                  call MatConvert(temp_mat_A, MATSAME, MAT_INITIAL_MATRIX, temp_mat_two, ierr)  
               else
                  ! temp_mat_two = temp_mat_A * mat_product
                  call MatMatMult(temp_mat_A, mat_product, &
                        MAT_INITIAL_MATRIX, 1.5d0, temp_mat_two, ierr)     
               end if    
               
               ! We copy out the last part of the old product if we're doing this as part of a fixed sparsity multiply
               if (output_product .AND. i > i_sparse - 2) then

                  call MatConvert(temp_mat_two, MATSAME, MAT_INITIAL_MATRIX, mat_prod_or_temp, ierr) 
                  ! If i == 1 then we know mat_product is the identity and we don't bother 
                  ! to write it out, we just have some custom code in the product given its trivial
                  if (i /= 1) then 
                     ! This ensures it has the matching sparsity
                     call MatConvert(mat_prod_or_temp, MATSAME, MAT_INITIAL_MATRIX, mat_product_save, ierr)  
                     ! This zeros mat_product_save and then puts mat_product into the sparsity pattern 
                     ! of mat_prod_or_temp
                     call MatCopy(mat_product, mat_product_save, DIFFERENT_NONZERO_PATTERN, ierr)     
                  end if              
               end if                 
            end if

            ! Then add the scaled version of each product
            if (reuse_triggered) then
               ! If doing reuse we know our nonzeros are a subset
               call MatAXPY(inv_matrix, square_sum, &
                        temp_mat_two, SUBSET_NONZERO_PATTERN, ierr)
            else
               ! Have to use the DIFFERENT_NONZERO_PATTERN here
               call MatAXPYWrapper(inv_matrix, square_sum, temp_mat_two)
            end if            

            if (i .le. i_sparse - 2) then

               ! temp_mat_three = matrix * temp_mat_two
               call MatMatMult(matrix, temp_mat_two, &
                     MAT_INITIAL_MATRIX, 1.5d0, temp_mat_three, ierr)     
               call MatDestroy(temp_mat_two, ierr)   
               if (output_product) then               
                  status_output(i, 2) = 1
               end if

               ! Then add the scaled version of each product
               if (reuse_triggered) then
                  ! If doing reuse we know our nonzeros are a subset
                  call MatAXPY(mat_product, -square_sum, &
                           temp_mat_three, SUBSET_NONZERO_PATTERN, ierr)
               else
                  ! Have to use the DIFFERENT_NONZERO_PATTERN here
                  call MatAXPYWrapper(mat_product, -square_sum, temp_mat_three)
               end if               

               ! We copy out the last part of the product if we're doing this as part of a fixed sparsity multiply
               if (output_product .AND. .NOT. first_complex) then
                  call MatConvert(mat_product, MATSAME, MAT_INITIAL_MATRIX, mat_prod_or_temp, ierr)      
               end if                 

               call MatDestroy(temp_mat_three, ierr) 
            else
               call MatDestroy(temp_mat_two, ierr)  
            end if

            ! Skip two evals
            i = i + 2

         end if       
      end do

      ! Final step if last root is real
      if (.NOT. first_complex) then
         if (coefficients(i_sparse,2) == 0d0) then
            ! Add in the final term multiplied by 1/theta_poly_order

            ! Skips eigenvalues that are numerically zero
            if (abs(coefficients(i_sparse,1)) > 1e-12) then     
               
               if (reuse_triggered) then
                  ! If doing reuse we know our nonzeros are a subset
                  call MatAXPY(inv_matrix, 1d0/coefficients(i_sparse,1), mat_product, SUBSET_NONZERO_PATTERN, ierr)
               else
                  ! Have to use the DIFFERENT_NONZERO_PATTERN here
                  call MatAXPYWrapper(inv_matrix, 1d0/coefficients(i_sparse,1), mat_product)
               end if     
               if (output_product) status_output(i_sparse, 1) = 1
            end if       
         end if   
      end if     

      call MatDestroy(temp_mat_A, ierr)
      call MatDestroy(mat_product, ierr)       

   end subroutine build_gmres_polynomial_newton_inverse_full   

! -------------------------------------------------------------------------------------------------------------------------------


end module gmres_poly_newton

