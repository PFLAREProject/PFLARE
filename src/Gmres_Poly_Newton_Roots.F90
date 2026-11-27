module gmres_poly_newton_roots

   use petscsys
   use pflare_parameters, only: PFLARE_TOL_RCOND, PFLARE_EPS, PFLARE_TOL_LEJA_PERTURB

#include "petsc/finclude/petscsys.h"
#include "finclude/pflare_blaslapack.h"

   implicit none
   public

   ! Dense routines that choose the roots of the Newton basis GMRES polynomial from the
   ! Arnoldi Hessenberg matrix: harmonic Ritz values, clustering, extra roots, Leja ordering
   ! and the adaptive degree and root rule

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
      PetscReal :: max_mag
      logical, dimension(size(real_roots)) :: sorted, repeated
      logical :: found

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

      ! magnitude(i) is one plus the log of the product of the distances from root i to every
      ! root sorted so far, added up in the order they were sorted. It is kept as a running
      ! sum, so each newly sorted root costs one pass over the unsorted ones
      ! repeated(i) is set once root i is exactly on top of a sorted root, where the product
      ! of distances is zero - we skip those rather than computing log10(0) = -infinity,
      ! which raises a floating point exception
      magnitude = 1d0
      repeated = .FALSE.
      do k_loc = 1, counter - 1
         call add_sorted_root(indices(k_loc))
      end do

      ! Do while we still have some sorting to do
      do while (counter-1 < size(real_roots))

         max_mag = -huge(max_mag)
         found = .FALSE.

         ! Find the unsorted root with the biggest product of differences
         do i_loc = 1, size(real_roots)

            ! Skip any we've already sorted, or that are on top of a sorted root
            if (sorted(i_loc) .OR. repeated(i_loc)) cycle

            ! Store the biggest as we're going
            if (magnitude(i_loc) > max_mag) then
               max_mag = magnitude(i_loc)
               max_loc(1) = i_loc
               found = .TRUE.
            end if            
         end do

         ! If we found nothing (ie the only unsorted are repeated roots with zero distance)
         ! just have the next entry in the list
         ! Note we can't test max_mag < 0 here, as the log of the product of distances
         ! is legitimately negative whenever that product is small (e.g., tightly 
         ! clustered or small magnitude eigenvalues), which would make the ordering scale dependent
         if (.NOT. found) then
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
         call add_sorted_root(indices(counter))
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
            call add_sorted_root(indices(counter))
            counter = counter + 1
         end if
      end do

      deallocate(magnitude)

      contains

      subroutine add_sorted_root(new_index)

         ! Adds the log of the distance to a newly sorted root to the running sum
         ! of every root

         integer, intent(in) :: new_index
         integer :: i_root
         PetscReal :: a_root, b_root, squares_root

         do i_root = 1, size(real_roots)

            if (repeated(i_root)) cycle

            ! Distance
            a_root = real_roots(i_root) - real_roots(new_index)
            b_root = imag_roots(i_root) - imag_roots(new_index)
            squares_root = a_root**2 + b_root**2
            if (squares_root == 0d0) then
               repeated(i_root) = .TRUE.
            else
               magnitude(i_root) = magnitude(i_root) + log10(sqrt(squares_root))
            end if
         end do

      end subroutine add_sorted_root

   end subroutine modified_leja   

   ! -------------------------------------------------------------------------------------------------

   subroutine cluster_eigenvalues_stable(real_roots, imag_roots, rel_tol, abs_tol)

      ! Clustering of (possibly complex) harmonic Ritz values.
      ! Numerically distinct clusters are moved to the front.
      ! Remaining entries are set to zero.
      ! Skips eigenvalues that are exactly zero (both real and imag parts).
      !
      ! The roots must be passed in with complex conjugate pairs next to each other
      ! (as dgeev returns them). The clustering is done only on the roots with 
      ! imag >= 0 and every complex cluster is output as a conjugate pair, with the
      ! positive imaginary root first, as everything downstream relies on that.
      ! Clustering all the roots instead can split a pair, e.g., if a real root is 
      ! within tolerance of a complex root but not of its conjugate
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
      integer                                :: i, j, n, n_unique, weight, cluster_weight
      logical, allocatable                   :: used(:)
      PetscReal                              :: dist, mag_i, mag_j, scale
      PetscReal                              :: sum_real, sum_imag, centroid_real, centroid_imag
      PetscReal, allocatable                 :: rtmp(:), itmp(:)

      n = size(real_roots)

      allocate(used(n))
      allocate(rtmp(n), itmp(n))

      used = .false.
      n_unique = 0

      ! ---------------------------------------------------------
      ! All-pairs clustering of the roots with imag >= 0
      ! ---------------------------------------------------------
      do i = 1, n

         if (used(i)) cycle

         ! Skip eigenvalues that are exactly zero and the negative imaginary
         ! conjugates (they are output with their positive imaginary root)
         if ((real_roots(i) == 0.0d0 .AND. imag_roots(i) == 0.0d0) .OR. &
               imag_roots(i) < 0.0d0) then
            used(i) = .true.
            cycle
         end if

         ! Start new cluster with eigenvalue i
         ! Each member is weighted by how many roots it stands for, ie a complex root 
         ! counts twice (itself and its conjugate) and a real root once, so the 
         ! real part of the centroid is the mean over the cluster and its mirror image
         weight = 1
         if (imag_roots(i) > 0.0d0) weight = 2
         sum_real = weight * real_roots(i)
         sum_imag = weight * imag_roots(i)
         cluster_weight = weight
         used(i) = .true.
         
         mag_i = sqrt(real_roots(i)**2 + imag_roots(i)**2)

         ! Look for all other eigenvalues close to this one
         do j = i + 1, n

            if (used(j)) cycle

            ! Skip exactly zero eigenvalues and negative imaginary conjugates
            if ((real_roots(j) == 0.0d0 .AND. imag_roots(j) == 0.0d0) .OR. &
                  imag_roots(j) < 0.0d0) then
               used(j) = .true.
               cycle
            end if

            mag_j = sqrt(real_roots(j)**2 + imag_roots(j)**2)
            
            ! Distance between eigenvalues
            ! Both have imag >= 0, so the conjugate of j is never closer to i than j
            dist = sqrt((real_roots(j) - real_roots(i))**2 + &
                        (imag_roots(j) - imag_roots(i))**2)

            ! Use the larger magnitude for relative scaling
            ! This is purely relative so the clustering does not depend on the scale 
            ! of the matrix, abs_tol handles roots near zero
            scale = max(mag_i, mag_j)

            ! Check if within tolerance
            if (dist <= abs_tol + rel_tol * scale) then
               weight = 1
               if (imag_roots(j) > 0.0d0) weight = 2
               sum_real = sum_real + weight * real_roots(j)
               sum_imag = sum_imag + weight * imag_roots(j)
               cluster_weight = cluster_weight + weight
               used(j) = .true.
            end if

         end do

         ! Compute cluster centroid (weighted mean)
         centroid_real = sum_real / real(cluster_weight, kind=kind(rtmp))
         centroid_imag = sum_imag / real(cluster_weight, kind=kind(itmp))

         ! If the centroid is within tolerance of its own conjugate, the pair
         ! is numerically a single real root
         if (centroid_imag > 0.0d0) then
            if (2 * centroid_imag <= abs_tol + &
                  rel_tol * sqrt(centroid_real**2 + centroid_imag**2)) then
               centroid_imag = 0.0d0
            end if
         end if

         n_unique = n_unique + 1
         rtmp(n_unique) = centroid_real
         itmp(n_unique) = centroid_imag
         ! Output a complex centroid as a conjugate pair, positive imaginary first
         if (centroid_imag > 0.0d0) then
            n_unique = n_unique + 1
            rtmp(n_unique) = centroid_real
            itmp(n_unique) = -centroid_imag
         end if

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

         ! Skips zero eigenvalues (we should only be passed non-zero roots)
         ! This matches the zero test used when applying the polynomial, ie 
         ! a purely imaginary root (a == 0, b /= 0) is not zero
         if (a == 0d0 .AND. b == 0d0) cycle

         ! Compute product(k)_{i, j/=i} * | 1 - theta_j/theta_i|
         do i_loc = 1, n_roots

            ! Skip
            if (k_loc == i_loc) cycle

            c = real_roots(i_loc)
            d = imag_roots(i_loc)

            ! Skips zero eigenvalues
            if (c == 0d0 .AND. d == 0d0) cycle

            ! theta_k/theta_i
            div_real = (a * c + b * d)/(c**2 + d**2)
            div_imag = (b * c - a * d)/(c**2 + d**2)

            ! |1 - theta_k/theta_i|
            div_mag = sqrt((1 - div_real)**2 + div_imag**2)

            ! Pof is about to overflow, store the exponent and 
            ! reset pof back to one
            ! We can hit this for very high order polynomials, where we have to 
            ! add more roots than 22 (ie pof > 1e308)
            ! range gives the decimal exponent range of pof, ie 307 in double and 37 in single
            ! In single pof can overflow at much lower order
            if (log10(pof(k_loc)) + log10(div_mag) > range(pof(k_loc))) then
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

   subroutine compute_extra_roots_adaptive(real_roots, imag_roots, rho, copy_all, sensitive, over_cap, &
                  real_roots_output, imag_roots_output)

      ! Add extra roots for stability, as in compute_extra_roots, but the product of factors
      ! of every root is recomputed after each copy is added, as a copy of one root changes
      ! the product of factors of all the others
      !
      ! If copy_all is false, copies are only added to roots where the extra factor does not
      ! amplify the polynomial at any of the other roots, ie where every other root z has
      ! |1 - z/theta_k| <= 1. Copies of small roots (or of roots with other roots outside
      ! that disc) can make the polynomial much worse
      ! sensitive is returned as true if a root we could not add copies to is too
      ! sensitive compared with the GMRES residual rho of this degree
      !
      ! If copy_all is true, copies can be added to any root. over_cap is returned as true
      ! (and nothing is allocated) if that needs more copies than there are roots
      !
      ! Only non-zero eigenvalues should be passed in, with complex conjugate pairs
      ! next to each other and the positive imaginary root first

      ! ~~~~~~
      PetscReal, dimension(:), intent(in)                 :: real_roots, imag_roots
      PetscReal, intent(in)                               :: rho
      logical, intent(in)                                 :: copy_all
      logical, intent(out)                                :: sensitive, over_cap
      PetscReal, dimension(:), allocatable, intent(inout) :: real_roots_output, imag_roots_output

      ! Local variables
      integer :: i_loc, j_loc, k_loc, n_roots, total_extra, counter, k_max, max_extra
      PetscReal :: a, b, c, d, div_real, div_imag, div_mag, amplify, sens, sens_max, sens_bad
      PetscReal :: decades_per_copy
      PetscReal, dimension(:, :), allocatable :: log_factor
      PetscReal, dimension(size(real_roots)) :: log_pof
      integer, dimension(size(real_roots)) :: copies, partner
      logical, dimension(size(real_roots)) :: allowed

      ! ~~~~~~

      n_roots = size(real_roots)
      sensitive = .FALSE.
      over_cap = .FALSE.

      ! Each copy reduces the sensitivity by around 1/eps, we take two decades off
      ! This is 14 decades in double, matching compute_extra_roots
      decades_per_copy = -log10(PFLARE_EPS) - 2

      ! log_factor(i,k) = log10|1 - theta_i/theta_k|
      allocate(log_factor(n_roots, n_roots))
      do k_loc = 1, n_roots

         c = real_roots(k_loc)
         d = imag_roots(k_loc)

         ! The complex conjugate of each root
         partner(k_loc) = k_loc
         if (d > 0) partner(k_loc) = k_loc + 1
         if (d < 0) partner(k_loc) = k_loc - 1

         do i_loc = 1, n_roots

            if (i_loc == k_loc) then
               log_factor(i_loc, k_loc) = 0
               cycle
            end if

            a = real_roots(i_loc)
            b = imag_roots(i_loc)

            ! theta_i/theta_k
            div_real = (a * c + b * d)/(c**2 + d**2)
            div_imag = (b * c - a * d)/(c**2 + d**2)

            ! |1 - theta_i/theta_k|
            div_mag = sqrt((1 - div_real)**2 + div_imag**2)
            log_factor(i_loc, k_loc) = log10(max(div_mag, tiny(div_mag)))
         end do
      end do

      ! A copy of root k (and its conjugate) multiplies the polynomial at every other
      ! root z by |1 - z/theta_k| (times |1 - z/conj(theta_k)|)
      ! Unless we are copying everything, we only allow copies that don't amplify at any other root
      ! Copies are counted on the positive imaginary root of a pair
      allowed = .FALSE.
      do k_loc = 1, n_roots
         if (imag_roots(k_loc) < 0) cycle

         if (copy_all) then
            allowed(k_loc) = .TRUE.
            cycle
         end if

         amplify = -huge(amplify)
         do i_loc = 1, n_roots
            if (i_loc == k_loc .OR. i_loc == partner(k_loc)) cycle
            sens = log_factor(i_loc, k_loc)
            if (partner(k_loc) /= k_loc) sens = sens + log_factor(i_loc, partner(k_loc))
            amplify = max(amplify, sens)
         end do
         allowed(k_loc) = amplify <= 1e4 * PFLARE_EPS
      end do

      ! More copies than this means this degree is not worth using
      max_extra = 4 * n_roots
      if (copy_all) max_extra = n_roots

      ! Add one copy at a time to the most sensitive root we are allowed to copy
      copies = 0
      total_extra = 0
      do

         ! log10 of the product of factors of every root given the copies added so far
         ! This doesn't include a root's own copies (or those of its conjugate)
         do i_loc = 1, n_roots
            log_pof(i_loc) = 0
            do k_loc = 1, n_roots
               log_pof(i_loc) = log_pof(i_loc) + (1 + copies(k_loc)) * log_factor(i_loc, k_loc)
            end do
            log_pof(i_loc) = log_pof(i_loc) - copies(i_loc) * log_factor(i_loc, partner(i_loc))
         end do

         ! As in compute_extra_roots, a root needs a copy if its pof is greater than 1e4
         k_max = 0
         sens_max = 0
         do k_loc = 1, n_roots
            if (.NOT. allowed(k_loc)) cycle
            sens = log_pof(k_loc) - decades_per_copy * copies(k_loc) - 4
            if (sens > sens_max) then
               sens_max = sens
               k_max = k_loc
            end if
         end do
         if (k_max == 0) exit

         counter = 1
         if (partner(k_max) /= k_max) counter = 2
         if (total_extra + counter > max_extra) then
            if (copy_all) then
               over_cap = .TRUE.
               deallocate(log_factor)
               return
            end if
            exit
         end if

         copies(k_max) = copies(k_max) + 1
         copies(partner(k_max)) = copies(k_max)
         total_extra = total_extra + counter
      end do

      ! The roots we couldn't copy have an error of around eps * pof
      ! If that is not well below the GMRES residual, this set of roots is not stable
      if (.NOT. copy_all) then
         sens_bad = -huge(sens_bad)
         do k_loc = 1, n_roots
            if (imag_roots(k_loc) < 0 .OR. allowed(k_loc)) cycle
            sens_bad = max(sens_bad, log10(PFLARE_EPS) + log_pof(k_loc))
         end do
         sensitive = sens_bad > log10(max(rho, tiny(rho))) - 1
      end if

      ! Allocate output arrays (original roots + extra roots)
      allocate(real_roots_output(n_roots + total_extra))
      allocate(imag_roots_output(n_roots + total_extra))

      ! Copy in original roots
      real_roots_output(1:n_roots) = real_roots(1:n_roots)
      imag_roots_output(1:n_roots) = imag_roots(1:n_roots)

      ! Add the extra copies of roots, ensuring conjugate pairs we add
      ! are next to each other
      counter = n_roots + 1
      do i_loc = 1, n_roots

         if (imag_roots(i_loc) < 0) cycle

         do j_loc = 1, copies(i_loc)

            real_roots_output(counter) = real_roots(i_loc)
            imag_roots_output(counter) = imag_roots(i_loc)
            counter = counter + 1
            ! Add in the conjugate
            if (imag_roots(i_loc) > 0) then
               real_roots_output(counter) = real_roots(i_loc)
               imag_roots_output(counter) = -imag_roots(i_loc)
               counter = counter + 1
            end if
         end do
      end do

      deallocate(log_factor)

   end subroutine compute_extra_roots_adaptive

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine leja_order_with_extras(real_roots, imag_roots, real_roots_added, imag_roots_added, indices)

      ! Computes the leja ordering of a set of roots with extra copies added
      ! real_roots, imag_roots are the original (non-zero) roots
      ! real_roots_added, imag_roots_added are the original roots followed by the extra copies
      ! indices is allocated and gives the ordering of real_roots_added, imag_roots_added

      ! ~~~~~~
      PetscReal, dimension(:), intent(in)               :: real_roots, imag_roots
      PetscReal, dimension(:), intent(in)               :: real_roots_added, imag_roots_added
      integer, dimension(:), allocatable, intent(inout) :: indices

      ! Local variables
      integer :: i_loc, j_loc, k_loc, counter, numerical_order
      PetscReal, dimension(:), allocatable :: perturbed_real, perturbed_imag
      PetscReal :: perturb_scale

      ! ~~~~~~

      numerical_order = size(real_roots)

      ! Create perturbed copy for leja ordering
      allocate(perturbed_real(size(real_roots_added)))
      allocate(perturbed_imag(size(real_roots_added)))
      perturbed_real = real_roots_added
      perturbed_imag = imag_roots_added

      ! Perturb the extra roots so they have unique values for the leja sort
      ! The perturbation is relative to the smallest (non-zero) root, so the ordering
      ! doesn't depend on the scale of the matrix, while keeping the perturbation
      ! negligible relative to every root
      ! A larger perturbation (e.g., relative to the biggest root, ||H|| or each root)
      ! makes the copies look well separated, so the leja ordering pulls them
      ! forward, which destabilises high order polynomials
      ! If the perturbation is lost to rounding on a big root, the copies are exact
      ! duplicates, which modified_leja handles
      perturb_scale = minval(sqrt(real_roots_added**2 + imag_roots_added**2))
      counter = numerical_order + 1
      do i_loc = 1, numerical_order
         k_loc = 0
         do j_loc = counter, size(real_roots_added)
            ! Check if this extra root matches the original
            if (real_roots_added(j_loc) == real_roots(i_loc) .AND. &
                abs(imag_roots_added(j_loc)) == abs(imag_roots(i_loc))) then
               k_loc = k_loc + 1
               perturbed_real(j_loc) = real_roots_added(j_loc) + k_loc * PFLARE_TOL_LEJA_PERTURB * perturb_scale
            end if
         end do
      end do

      ! Leja order only the non-zero eigenvalues (with extras)
      call modified_leja(perturbed_real, perturbed_imag, indices)

      deallocate(perturbed_real, perturbed_imag)

   end subroutine leja_order_with_extras

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine hessenberg_newton_replay(H, degree, real_roots, imag_roots, residual)

      ! Applies the Newton polynomial with the given (ordered, non-zero) roots to the first
      ! unit vector in the space of the Arnoldi Hessenberg matrix, in the same way as
      ! petsc_newton applies it to a vector, and returns the norm of the residual
      ! Uses the leading (degree+1) x degree block of H, padded with a zero column
      ! This is a cheap model of what the polynomial does to the vector the Arnoldi was
      ! started with, including the rounding in the apply. It is exact only while the
      ! number of roots is at most degree
      ! The residual is returned as huge if the apply blows up
      ! Everything tested here is independent of the scale of the matrix

      ! ~~~~~~
      PetscReal, dimension(:,:), intent(in) :: H
      integer, intent(in)                   :: degree
      PetscReal, dimension(:), intent(in)   :: real_roots, imag_roots
      PetscReal, intent(out)                :: residual

      ! Local variables
      integer :: i, n_roots
      PetscReal, dimension(degree+1, degree+1) :: H_ext
      PetscReal, dimension(degree+1) :: t, t_two, y
      PetscReal :: square_sum, blow_up

      ! ~~~~~~

      n_roots = size(real_roots)
      residual = huge(residual)
      ! Past this the rounding error swamps any residual we could accept
      blow_up = 1/PFLARE_EPS**2

      H_ext = 0
      H_ext(1:degree+1, 1:degree) = H(1:degree+1, 1:degree)

      t = 0
      t(1) = 1
      y = 0

      i = 1
      do while (i .le. n_roots - 1)

         if (imag_roots(i) == 0d0) then

            y = y + t/real_roots(i)
            t = t - matmul(H_ext, t)/real_roots(i)
            i = i + 1

         else

            square_sum = real_roots(i)**2 + imag_roots(i)**2
            t_two = 2 * real_roots(i) * t - matmul(H_ext, t)
            y = y + t_two/square_sum
            if (i .le. n_roots - 2) then
               t = t - matmul(H_ext, t_two)/square_sum
            end if
            i = i + 2
         end if

         ! The comparison is false for a NaN
         if (.NOT. (maxval(abs(t)) < blow_up)) return
      end do

      ! Final step if last root is real
      if (imag_roots(n_roots) == 0d0) then
         y = y + t/real_roots(n_roots)
      end if
      ! Residual of e_1 - H y
      t = -matmul(H_ext, y)
      t(1) = t(1) + 1
      residual = norm2(t)
      ! The comparison is false for a NaN
      if (.NOT. (residual < huge(residual))) residual = huge(residual)

   end subroutine hessenberg_newton_replay

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine adaptive_newton_candidate(H, degree, m, rho, accepted, real_roots_out, imag_roots_out, n_base)

      ! Builds the roots of the Newton polynomial of a given degree for the adaptive rule
      ! and decides if they are acceptable
      ! If accepted, real_roots_out, imag_roots_out are allocated and hold the non-zero roots
      ! with any extras in leja order, and n_base is the number of non-zero roots without the extras

      ! ~~~~~~
      PetscReal, dimension(:,:), intent(in)               :: H
      integer, intent(in)                                 :: degree, m
      PetscReal, intent(in)                               :: rho
      logical, intent(out)                                :: accepted
      PetscReal, dimension(:), allocatable, intent(inout) :: real_roots_out, imag_roots_out
      integer, intent(out)                                :: n_base

      ! Local variables
      integer :: i_loc
      PetscReal, dimension(degree, 2) :: roots
      PetscReal, dimension(:), allocatable :: real_added, imag_added
      integer, dimension(:), allocatable :: indices
      logical :: sensitive, over_cap, copied_all
      PetscReal :: residual
      integer :: replay_limit

      ! ~~~~~~

      if (allocated(real_roots_out)) deallocate(real_roots_out, imag_roots_out)
      accepted = .TRUE.

      ! Harmonic Ritz values of the leading block, with the non-zero roots first
      call harmonic_ritz_roots(H, degree, m, roots(:, 1), roots(:, 2))
      n_base = 0
      do i_loc = 1, degree
         if (roots(i_loc, 1) /= 0d0 .OR. roots(i_loc, 2) /= 0d0) n_base = n_base + 1
      end do

      if (n_base == 0) then
         allocate(real_roots_out(0), imag_roots_out(0))
         return
      end if

      ! Extra copies only of the roots where a copy cannot amplify the polynomial
      call compute_extra_roots_adaptive(roots(1:n_base, 1), roots(1:n_base, 2), rho, .FALSE., &
               sensitive, over_cap, real_added, imag_added)

      ! If one of the roots we didn't copy is too sensitive, see if copying
      ! every sensitive root gives a polynomial that is still good
      ! We can't lower the degree any further than one, so we just take what we have there
      copied_all = sensitive .AND. degree > 1
      if (copied_all) then

         deallocate(real_added, imag_added)
         call compute_extra_roots_adaptive(roots(1:n_base, 1), roots(1:n_base, 2), rho, .TRUE., &
                  sensitive, over_cap, real_added, imag_added)
         if (over_cap) then
            accepted = .FALSE.
            return
         end if
      end if

      call leja_order_with_extras(roots(1:n_base, 1), roots(1:n_base, 2), real_added, imag_added, indices)
      allocate(real_roots_out(size(real_added)), imag_roots_out(size(real_added)))
      real_roots_out = real_added(indices)
      imag_roots_out = imag_added(indices)
      deallocate(real_added, imag_added, indices)

      ! The copies we were not sure about are only kept if the polynomial
      ! still does about as well as GMRES on the Arnoldi start vector
      ! Every other candidate is replayed too, but is only rejected if the polynomial is clearly
      ! broken, ie its residual on the start vector is more than 100 times that of GMRES
      ! A candidate that is only a little worse than GMRES is kept, as the degree below it can
      ! pass on the start vector and be a much worse preconditioner on other vectors
      ! The root sets this is for (eg, close roots merged by the clustering, or an apply that
      ! has lost its accuracy to rounding) are worse than GMRES by many orders of magnitude
      replay_limit = 2
      if (.NOT. copied_all) replay_limit = root_env_int('PFLARE_ROOT_REPLAY_FACTOR', 100)
      if (copied_all .OR. root_env_int('PFLARE_ROOT_REPLAY_ALWAYS', 1) == 1) then
         call hessenberg_newton_replay(H, degree, real_roots_out, imag_roots_out, residual)
         if (residual > replay_limit * rho) then
            accepted = .FALSE.
            deallocate(real_roots_out, imag_roots_out)
         end if
      end if

   end subroutine adaptive_newton_candidate

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine adaptive_newton_roots(H, m, real_roots_out, imag_roots_out, n_base, real_roots_alt, imag_roots_alt)

      ! Chooses the degree and the roots (with extras) of the Newton polynomial
      ! using only the (m+1) x m Arnoldi Hessenberg matrix H
      ! real_roots_out, imag_roots_out are allocated and hold the non-zero roots
      ! with any extras in leja order and n_base is the number of non-zero roots without
      ! the extras

      ! ~~~~~~
      PetscReal, dimension(:,:), intent(in)               :: H
      integer, intent(in)                                 :: m
      PetscReal, dimension(:), allocatable, intent(inout) :: real_roots_out, imag_roots_out
      integer, intent(out)                                :: n_base
      ! If greedy_head_tail_order finds an order of the same roots that passes its tests,
      ! that order is returned here (allocated) for the caller to use
      ! real_roots_out, imag_roots_out are always in the leja order
      PetscReal, dimension(:), allocatable, intent(inout), optional :: real_roots_alt, imag_roots_alt

      ! Local variables
      integer :: degree, bad, good, mid, step, n_base_trial
      PetscReal, dimension(m) :: rho_h
      PetscReal, dimension(:), allocatable :: real_trial, imag_trial
      logical :: accepted, taken

      ! ~~~~~~

      ! The relative GMRES residual at every degree up to m comes for free
      ! from the Hessenberg matrix
      call hessenberg_gmres_residuals(H, m, rho_h)

      ! Arnoldi is nested, so the leading (j+1) x j block of H gives the
      ! GMRES polynomial of degree j without any extra work
      ! We first drop the degree that buys nothing, ie we take the smallest degree
      ! whose residual is as small as the smallest we see, where any
      ! residual at rounding level counts as converged
      do degree = 1, m
         if (rho_h(degree) <= max(minval(rho_h), 10 * PFLARE_EPS)) exit
      end do
      degree = min(degree, m)

      call adaptive_newton_candidate(H, degree, m, rho_h(degree), accepted, real_roots_out, imag_roots_out, n_base)
      if (accepted) then
         call greedy_candidate(degree)
         return
      end if

      ! Otherwise we have to lower the degree until the roots are acceptable
      ! Step down geometrically until we find a degree that is (degree one always is)
      bad = degree
      step = 1
      do
         good = max(1, degree - step)
         call adaptive_newton_candidate(H, good, m, rho_h(good), accepted, real_roots_out, imag_roots_out, n_base)
         if (accepted) exit
         bad = good
         step = 2 * step
      end do

      ! Then bisect between the acceptable degree and the one that is not
      do while (bad - good > 1)
         mid = (bad + good)/2
         call adaptive_newton_candidate(H, mid, m, rho_h(mid), accepted, real_trial, imag_trial, n_base_trial)
         if (accepted) then
            good = mid
            call move_alloc(real_trial, real_roots_out)
            call move_alloc(imag_trial, imag_roots_out)
            n_base = n_base_trial
         else
            bad = mid
         end if
      end do

      call greedy_candidate(good)

      contains

      subroutine greedy_candidate(degree_final)

         ! Returns the greedy order of the accepted roots in real_roots_alt, imag_roots_alt
         ! if it is switched on and estimated to be better
         integer, intent(in) :: degree_final
         PetscReal, dimension(:), allocatable :: real_new, imag_new

         if (.NOT. present(real_roots_alt)) return
         if (allocated(real_roots_alt)) deallocate(real_roots_alt, imag_roots_alt)
         if (root_env_int('PFLARE_ROOT_ORDER_GREEDY', 1) /= 1) return
         if (size(real_roots_out) == 0) return

         allocate(real_new(size(real_roots_out)), imag_new(size(real_roots_out)))
         call greedy_head_tail_order(H, degree_final, real_roots_out, imag_roots_out, real_new, imag_new, taken)
         if (taken) then
            call move_alloc(real_new, real_roots_alt)
            call move_alloc(imag_new, imag_roots_alt)
         else
            deallocate(real_new, imag_new)
         end if

      end subroutine greedy_candidate

   end subroutine adaptive_newton_roots

! -------------------------------------------------------------------------------------------------------------------------------

   integer function root_env_int(name, default_value)

      ! Reads an integer switch for the root rules from the environment
      ! Returns default_value if the variable is not set or cannot be read
      ! These are for debugging and for comparisons and are not user options:
      !   PFLARE_ROOT_REPLAY_ALWAYS = 0   the adaptive rule only replays the candidates where every
      !                                   sensitive root was copied (default 1: replay every candidate)
      !   PFLARE_ROOT_REPLAY_FACTOR = f   reject the other candidates if the replay is f times the
      !                                   GMRES residual (default 100)
      !   PFLARE_ROOT_ORDER_GREEDY  = 0   the adaptive rule keeps the leja order of its roots
      !                                   (default 1: reorder with greedy_head_tail_order if its tests pass)
      !   PFLARE_ROOT_ALG4          = c   the pof rule becomes compute_extra_roots_pof_update with cutoff c
      !                                   (default 0: off)
      !   PFLARE_ROOT_VERBOSE       = 1   print what the rules above did

      ! ~~~~~~
      character(len=*), intent(in) :: name
      integer, intent(in)          :: default_value

      character(len=32) :: env_char
      integer :: length, status, env_val, io_status
      ! ~~~~~~

      root_env_int = default_value
      call get_environment_variable(name, env_char, length=length, status=status)
      if (status /= 0 .OR. length == 0) return
      read(env_char, *, iostat=io_status) env_val
      if (io_status == 0) root_env_int = env_val

   end function root_env_int

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine head_tail_estimate(H, degree, real_roots, imag_roots, estimate)

      ! Estimates the rounding error of the product form apply (petsc_newton) of the Newton
      ! polynomial with the given ordered, non-zero roots, using only the Arnoldi Hessenberg matrix
      ! With head_g the product of the first g factors and tail_g the product of the rest,
      ! the first order error is the sum over the steps of the local rounding error, which is
      ! proportional to the vector the step is applied to (head_{g-1} applied to the start vector),
      ! propagated by the factors still to come (I - tail_g)
      ! This is the criterion of Lebedev & Finogenov (1971) for the ordering of the parameters
      ! of a cyclic Richardson iteration, where the local error is also taken to be
      ! proportional to the current iterate
      ! The heads are computed as in hessenberg_newton_replay and the tails by sweeping
      ! three fixed row vectors backwards through the factors, so this costs about
      ! as much as four replays
      ! estimate = eps/2 * sqrt( sum_g ( ||head_{g-1} e_1|| * (1 + max_z ||z^T tail_g||) )^2 )
      ! and is returned as huge if anything overflows
      ! It is independent of the scale of the matrix

      ! ~~~~~~
      PetscReal, dimension(:,:), intent(in) :: H
      integer, intent(in)                   :: degree
      PetscReal, dimension(:), intent(in)   :: real_roots, imag_roots
      PetscReal, intent(out)                :: estimate

      ! Local variables
      integer, parameter :: n_probe = 3
      integer :: i, k, n_roots, n_units, seed
      integer, dimension(size(real_roots)) :: unit_start
      PetscReal, dimension(degree+1, degree+1) :: H_ext
      PetscReal, dimension(n_probe, degree+1) :: z, z_two
      PetscReal, dimension(degree+1) :: t, t_two
      PetscReal, dimension(size(real_roots)) :: head, tail
      PetscReal :: square_sum, total, term

      ! ~~~~~~

      n_roots = size(real_roots)
      estimate = huge(estimate)
      if (n_roots == 0) then
         estimate = 0
         return
      end if

      H_ext = 0
      H_ext(1:degree+1, 1:degree) = H(1:degree+1, 1:degree)

      ! A conjugate pair is one step of the apply
      n_units = 0
      i = 1
      do while (i .le. n_roots)
         n_units = n_units + 1
         unit_start(n_units) = i
         if (imag_roots(i) /= 0d0 .AND. i < n_roots) then
            i = i + 2
         else
            i = i + 1
         end if
      end do

      ! Fixed pseudo-random row vectors with unit norm (a linear congruential generator,
      ! so the result is the same on every rank and every run)
      seed = 12345
      do k = 1, n_probe
         do i = 1, degree + 1
            seed = mod(seed * 1103 + 12347, 65521)
            z(k, i) = 2 * real(seed, kind(estimate)) / 65521 - 1
         end do
         z(k, :) = z(k, :) / norm2(z(k, :))
      end do

      ! Tails: tail(g) is the norm of z^T times the product of the factors after step g
      do k = n_units, 1, -1
         tail(k) = 0
         do i = 1, n_probe
            tail(k) = max(tail(k), norm2(z(i, :)))
         end do
         ! The comparison is false for a NaN
         if (.NOT. (tail(k) < sqrt(huge(estimate)))) return
         i = unit_start(k)
         if (imag_roots(i) == 0d0) then
            z = z - matmul(z, H_ext)/real_roots(i)
         else
            square_sum = real_roots(i)**2 + imag_roots(i)**2
            z_two = 2 * real_roots(i) * z - matmul(z, H_ext)
            z = z - matmul(z_two, H_ext)/square_sum
         end if
      end do

      ! Heads: head(g) is the norm of the product of the factors before step g applied to e_1
      t = 0
      t(1) = 1
      do k = 1, n_units
         head(k) = norm2(t)
         if (.NOT. (head(k) < sqrt(huge(estimate)))) return
         i = unit_start(k)
         if (imag_roots(i) == 0d0) then
            t = t - matmul(H_ext, t)/real_roots(i)
         else
            square_sum = real_roots(i)**2 + imag_roots(i)**2
            t_two = 2 * real_roots(i) * t - matmul(H_ext, t)
            t = t - matmul(H_ext, t_two)/square_sum
         end if
      end do

      total = 0
      do k = 1, n_units
         term = head(k) * (1 + tail(k))
         if (.NOT. (term < sqrt(huge(estimate)))) return
         total = total + term**2
      end do
      if (.NOT. (total < huge(estimate))) return
      estimate = PFLARE_EPS/2 * sqrt(total)

   end subroutine head_tail_estimate

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine greedy_head_tail_order(H, degree, real_roots, imag_roots, real_new, imag_new, taken)

      ! Computes another order for the (non-zero, already ordered) roots of the Newton polynomial,
      ! in which the products of the factors applied so far (heads) and of the factors still to
      ! come (tails) are both kept small, and says if that is estimated to reduce the
      ! rounding error of the apply
      ! The polynomial is not changed, only the order in which its factors are applied
      ! real_new, imag_new always return the new order, real_roots, imag_roots are not modified
      !
      ! At every step the root is taken that minimises
      !   max_S |head| * (1 + max_S |tail|)
      ! where S are the Ritz values of the leading degree x degree block of H, a stand-in
      ! for the spectrum, and the distance from a point of S to a root is never taken
      ! smaller than 100 eps ||H||, as a computed root is not closer than that to the
      ! eigenvalue it approximates even when a Ritz value coincides with it
      ! A leja ordering only controls the heads; with outlying roots and extra copies
      ! the tails can be very large
      !
      ! The new order is not always better, so taken is only returned as true if two tests
      ! that cost no matvecs both pass
      ! 1) The measure that was minimised, max over the steps of max_S |head| * (1 + max_S |tail|),
      !    is at least five decades smaller for the new order than for the order we were given
      !    When it barely falls the greedy has found nothing: this happens when a root without a
      !    copy has a large product of other factors (no order can then do better than the square
      !    root of that product), or when the Ritz values are not close to the spectrum
      !    (non-normal matrices). In those cases the new order has been seen to be much worse,
      !    eg on a diagonal matrix with many more eigenvalues than roots, where the heads grow
      !    between the Ritz values and only a leja ordering keeps them small
      ! 2) head_tail_estimate is at least 10 times smaller for the new order

      ! ~~~~~~
      PetscReal, dimension(:,:), intent(in)  :: H
      integer, intent(in)                    :: degree
      PetscReal, dimension(:), intent(in)    :: real_roots, imag_roots
      PetscReal, dimension(:), intent(out)   :: real_new, imag_new
      logical, intent(out)                   :: taken

      ! Local variables
      integer :: i, k, r, step, n_roots, n_units, best, errorcode, counter
      PetscBLASInt :: lwork, info, n_bl, lda_bl, one_bl
      integer, dimension(size(real_roots)) :: unit_start, unit_size, order
      logical, dimension(size(real_roots)) :: left
      PetscReal, dimension(degree, degree) :: H_square
      PetscReal, dimension(degree) :: ritz_real, ritz_imag, log_head, log_tail
      PetscReal, dimension(:,:), allocatable :: log_factor, VL, VR
      PetscReal, dimension(:), allocatable :: work
      PetscReal :: floor_dist, root_mag, dist, score, best_score, new_head, new_tail
      PetscReal :: estimate_old, estimate_new, measure_old, measure_new
      PetscReal, parameter :: log_clamp = -700

      ! ~~~~~~

      taken = .FALSE.
      real_new = real_roots
      imag_new = imag_roots
      n_roots = size(real_roots)
      if (n_roots < 3 .OR. degree < 2) return

      ! A conjugate pair is one unit
      n_units = 0
      i = 1
      do while (i .le. n_roots)
         n_units = n_units + 1
         unit_start(n_units) = i
         if (imag_roots(i) /= 0d0 .AND. i < n_roots) then
            unit_size(n_units) = 2
            i = i + 2
         else
            unit_size(n_units) = 1
            i = i + 1
         end if
      end do
      if (n_units < 3) return

      ! ~~~~~~~~~~~
      ! Ritz values of the leading block
      ! ~~~~~~~~~~~
      H_square = H(1:degree, 1:degree)
      floor_dist = 100 * PFLARE_EPS * norm2(H(1:degree+1, 1:degree))

      allocate(work(1), VL(0, 0), VR(0, 0))
      lwork = -1
      n_bl = degree
      lda_bl = size(H_square, 1)
      one_bl = 1
      call PFLAREgeev('N', 'N', n_bl, H_square, lda_bl, &
                     ritz_real, ritz_imag, VL, one_bl, VR, one_bl, &
                     work, lwork, info)
      lwork = int(work(1))
      deallocate(work)
      allocate(work(lwork))
      call PFLAREgeev('N', 'N', n_bl, H_square, lda_bl, &
                     ritz_real, ritz_imag, VL, one_bl, VR, one_bl, &
                     work, lwork, info)
      deallocate(work, VL, VR)
      if (info /= 0) then
         print *, "Eig decomposition failed"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! ~~~~~~~~~~~
      ! log of the magnitude of every factor at every Ritz value
      ! ~~~~~~~~~~~
      allocate(log_factor(degree, n_units))
      do r = 1, n_units
         i = unit_start(r)
         root_mag = sqrt(real_roots(i)**2 + imag_roots(i)**2)
         do k = 1, degree
            dist = sqrt((ritz_real(k) - real_roots(i))**2 + (ritz_imag(k) - imag_roots(i))**2)
            log_factor(k, r) = log(max(dist, floor_dist)/root_mag)
            if (unit_size(r) == 2) then
               dist = sqrt((ritz_real(k) - real_roots(i))**2 + (ritz_imag(k) + imag_roots(i))**2)
               log_factor(k, r) = log_factor(k, r) + log(max(dist, floor_dist)/root_mag)
            end if
            log_factor(k, r) = max(log_factor(k, r), log_clamp)
         end do
      end do

      ! ~~~~~~~~~~~
      ! Greedy ordering
      ! ~~~~~~~~~~~
      left = .FALSE.
      left(1:n_units) = .TRUE.
      log_head = 0
      do step = 1, n_units

         ! The tail is recomputed from the roots that are left, so that the clamp
         ! and rounding in a running difference cannot build up
         log_tail = 0
         do r = 1, n_units
            if (left(r)) log_tail = log_tail + log_factor(:, r)
         end do

         best = 0
         best_score = huge(best_score)
         do r = 1, n_units
            if (.NOT. left(r)) cycle
            new_head = maxval(log_head + log_factor(:, r))
            new_tail = maxval(log_tail - log_factor(:, r))
            ! log(1 + exp(new_tail)) without overflow
            if (new_tail > 30) then
               score = new_head + new_tail
            else
               score = new_head + log(1 + exp(new_tail))
            end if
            if (score < best_score) then
               best_score = score
               best = r
            end if
         end do
         ! Only if every score was a NaN
         if (best == 0) then
            deallocate(log_factor)
            return
         end if

         order(step) = best
         left(best) = .FALSE.
         log_head = log_head + log_factor(:, best)
      end do

      ! The measure that was minimised, for the order we were given and for the new one
      call head_tail_measure(.FALSE., measure_old)
      call head_tail_measure(.TRUE., measure_new)
      deallocate(log_factor)

      counter = 0
      do step = 1, n_units
         i = unit_start(order(step))
         do k = 0, unit_size(order(step)) - 1
            counter = counter + 1
            real_new(counter) = real_roots(i + k)
            imag_new(counter) = imag_roots(i + k)
         end do
      end do

      ! ~~~~~~~~~~~
      ! Only take the new order if it is estimated to be clearly better
      ! ~~~~~~~~~~~
      call head_tail_estimate(H, degree, real_roots, imag_roots, estimate_old)
      call head_tail_estimate(H, degree, real_new, imag_new, estimate_new)

      ! The comparison is false for a NaN
      if (estimate_new < huge(estimate_new) .AND. 10 * estimate_new < estimate_old .AND. &
            measure_old - measure_new >= 5 * log(real(10, kind(measure_old)))) then
         taken = .TRUE.
      end if

      if (root_env_int('PFLARE_ROOT_VERBOSE', 0) == 1) then
         print '(A,I5,A,I5,A,ES10.2,A,ES10.2,A,F7.1,A,F7.1,A,L1)', " PFLARE root order: degree ", degree, &
               " roots ", n_roots, " estimate given order ", estimate_old, " greedy order ", estimate_new, &
               " log10 measure given ", measure_old/log(real(10, kind(measure_old))), &
               " greedy ", measure_new/log(real(10, kind(measure_new))), " taken ", taken
      end if

      contains

      subroutine head_tail_measure(new_order, measure)

         ! log of max over the steps of max_S |head before the step| * (1 + max_S |tail after it|)
         logical, intent(in)    :: new_order
         PetscReal, intent(out) :: measure

         integer :: step_loc, r_loc
         PetscReal, dimension(degree) :: head_loc, tail_loc
         PetscReal :: max_tail, term_loc

         head_loc = 0
         tail_loc = sum(log_factor, dim=2)
         measure = -huge(measure)
         do step_loc = 1, n_units
            r_loc = step_loc
            if (new_order) r_loc = order(step_loc)
            tail_loc = tail_loc - log_factor(:, r_loc)
            max_tail = maxval(tail_loc)
            if (max_tail > 30) then
               term_loc = maxval(head_loc) + max_tail
            else
               term_loc = maxval(head_loc) + log(1 + exp(max_tail))
            end if
            measure = max(measure, term_loc)
            head_loc = head_loc + log_factor(:, r_loc)
         end do

      end subroutine head_tail_measure

   end subroutine greedy_head_tail_order

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine compute_extra_roots_pof_update(real_roots, imag_roots, cutoff, real_roots_output, imag_roots_output, indices)

      ! Adds extra copies of roots for stability with the rule of
      ! Embree, Henningsen, Jackson & Morgan, "Polynomial approximation to the inverse of a
      ! large matrix", arXiv:2502.18317 (2025), Algorithm 4
      ! This differs from compute_extra_roots (Loe & Morgan 2022, Algorithm 2) in that the roots are
      ! tested in increasing order of magnitude and the pof of the roots not yet tested is
      ! updated every time copies are added
      ! A root gets ceiling((log10(pof) - cutoff)/14) copies; they use cutoff = 8, Loe & Morgan use 4
      ! The roots are then leja ordered with the copies perturbed by a relative 1e-12,
      ! so that they are distinct for the ordering
      ! real_roots, imag_roots are the non-zero roots with conjugate pairs next to each other,
      ! positive imaginary part first
      ! real_roots_output, imag_roots_output are allocated and hold the roots followed by the copies
      ! and indices is allocated and gives their ordering
      ! This is a baseline for comparisons and is only for double precision

      ! ~~~~~~
      PetscReal, dimension(:), intent(in)                 :: real_roots, imag_roots
      PetscReal, intent(in)                               :: cutoff
      PetscReal, dimension(:), allocatable, intent(inout) :: real_roots_output, imag_roots_output
      integer, dimension(:), allocatable, intent(inout)   :: indices

      ! Local variables
      integer :: i, j, k, n, n_copy, partner, total, counter, seed, step, n_order
      integer, dimension(size(real_roots)) :: copies, sorted
      logical, dimension(size(real_roots)) :: done
      PetscReal, dimension(size(real_roots)) :: log_pof, mag
      PetscReal, dimension(:,:), allocatable :: log_factor
      PetscReal, dimension(:), allocatable :: perturbed_real, perturbed_imag
      PetscReal :: den, diff_real, diff_imag, factor

      ! ~~~~~~

      n = size(real_roots)

      ! log_factor(k, i) = log10|1 - theta_k/theta_i|
      allocate(log_factor(n, n))
      do i = 1, n
         den = real_roots(i)**2 + imag_roots(i)**2
         do k = 1, n
            if (k == i) then
               log_factor(k, i) = 0
            else
               ! theta_k/theta_i
               diff_real = (real_roots(k) * real_roots(i) + imag_roots(k) * imag_roots(i))/den
               diff_imag = (imag_roots(k) * real_roots(i) - real_roots(k) * imag_roots(i))/den
               log_factor(k, i) = log10(max(sqrt((1 - diff_real)**2 + diff_imag**2), tiny(den)))
            end if
         end do
      end do
      log_pof = sum(log_factor, dim=2)

      ! The roots with non-negative imaginary part in increasing order of magnitude
      mag = sqrt(real_roots**2 + imag_roots**2)
      n_order = 0
      do i = 1, n
         if (imag_roots(i) < 0d0) cycle
         n_order = n_order + 1
         sorted(n_order) = i
      end do
      do i = 2, n_order
         k = sorted(i)
         j = i - 1
         do while (j >= 1)
            if (mag(sorted(j)) <= mag(k)) exit
            sorted(j + 1) = sorted(j)
            j = j - 1
         end do
         sorted(j + 1) = k
      end do

      copies = 0
      done = .FALSE.
      do step = 1, n_order
         k = sorted(step)
         partner = k
         if (imag_roots(k) > 0d0) partner = k + 1
         done(k) = .TRUE.
         done(partner) = .TRUE.
         if (log_pof(k) > cutoff) then
            n_copy = ceiling((log_pof(k) - cutoff)/14)
            copies(k) = n_copy
            copies(partner) = n_copy
            ! Update the pof of the roots we haven't tested yet
            do i = 1, n
               if (done(i)) cycle
               log_pof(i) = log_pof(i) + n_copy * log_factor(i, k)
               if (partner /= k) log_pof(i) = log_pof(i) + n_copy * log_factor(i, partner)
            end do
         end if
      end do
      deallocate(log_factor)

      total = n
      do i = 1, n
         if (imag_roots(i) >= 0d0) then
            total = total + copies(i)
            if (imag_roots(i) > 0d0) total = total + copies(i)
         end if
      end do

      allocate(real_roots_output(total), imag_roots_output(total))
      allocate(perturbed_real(total), perturbed_imag(total))
      real_roots_output(1:n) = real_roots
      imag_roots_output(1:n) = imag_roots
      perturbed_real(1:n) = real_roots
      perturbed_imag(1:n) = imag_roots

      ! The copies, perturbed for the ordering only
      ! They use a normal random number, we use a fixed sequence in (-1, 1) so the
      ! result is the same on every rank and every run
      seed = 4321
      counter = n
      do i = 1, n
         if (imag_roots(i) < 0d0) cycle
         do j = 1, copies(i)
            seed = mod(seed * 1103 + 12347, 65521)
            factor = 1 + (2 * real(seed, kind(factor)) / 65521 - 1) / 1000000 / 1000000
            counter = counter + 1
            real_roots_output(counter) = real_roots(i)
            imag_roots_output(counter) = imag_roots(i)
            perturbed_real(counter) = real_roots(i) * factor
            perturbed_imag(counter) = imag_roots(i) * factor
            if (imag_roots(i) > 0d0) then
               counter = counter + 1
               real_roots_output(counter) = real_roots(i)
               imag_roots_output(counter) = -imag_roots(i)
               perturbed_real(counter) = real_roots(i) * factor
               perturbed_imag(counter) = -imag_roots(i) * factor
            end if
         end do
      end do

      call modified_leja(perturbed_real, perturbed_imag, indices)
      deallocate(perturbed_real, perturbed_imag)

      if (root_env_int('PFLARE_ROOT_VERBOSE', 0) == 1) then
         print '(A,I5,A,I6)', " PFLARE root rule: pof update, roots ", n, " copies ", total - n
      end if

   end subroutine compute_extra_roots_pof_update

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine hessenberg_gmres_residuals(H, m, rho)

      ! Computes the relative GMRES residual after each of the first m Arnoldi steps
      ! from the (m+1) x m Hessenberg matrix, with Givens rotations
      ! rho(j) is the residual of the GMRES polynomial of degree j
      ! H is not modified

      ! ~~~~~~
      PetscReal, dimension(:,:), intent(in) :: H
      integer, intent(in)                   :: m
      PetscReal, dimension(:), intent(out)  :: rho

      ! Local variables
      integer :: i_loc, j_loc
      PetscReal, dimension(m+1) :: col, g
      PetscReal, dimension(m) :: cs, sn
      PetscReal :: temp, denom

      ! ~~~~~~

      g = 0
      g(1) = 1

      do j_loc = 1, m

         col(1:j_loc+1) = H(1:j_loc+1, j_loc)

         ! Apply the previous rotations to this column
         do i_loc = 1, j_loc - 1
            temp = cs(i_loc) * col(i_loc) + sn(i_loc) * col(i_loc+1)
            col(i_loc+1) = -sn(i_loc) * col(i_loc) + cs(i_loc) * col(i_loc+1)
            col(i_loc) = temp
         end do

         ! The rotation that zeros the subdiagonal
         denom = sqrt(col(j_loc)**2 + col(j_loc+1)**2)
         if (denom == 0) then
            cs(j_loc) = 1
            sn(j_loc) = 0
         else
            cs(j_loc) = col(j_loc)/denom
            sn(j_loc) = col(j_loc+1)/denom
         end if

         g(j_loc+1) = -sn(j_loc) * g(j_loc)
         g(j_loc) = cs(j_loc) * g(j_loc)
         rho(j_loc) = abs(g(j_loc+1))
      end do

   end subroutine hessenberg_gmres_residuals

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine harmonic_ritz_roots(H_in, degree, m, real_roots, imag_roots)

      ! Computes the harmonic Ritz values of the leading (degree+1) x degree block
      ! of the Arnoldi Hessenberg matrix H_in, ie the roots of the GMRES residual 
      ! polynomial of that degree
      ! m is the number of Arnoldi steps that were completed
      ! Numerically zero roots are set to exactly zero, close roots are 
      ! clustered together and any zero roots are placed at the end
      ! H_in is not modified

      ! ~~~~~~
      PetscReal, dimension(:,:), intent(in)  :: H_in
      integer, intent(in)                    :: degree, m
      PetscReal, dimension(:), intent(inout) :: real_roots, imag_roots

      ! Local variables
      integer :: i_loc, errorcode, iwork_size
      ! BLAS/LAPACK integer arguments must be PetscBLASInt
      PetscBLASInt :: lwork, rank, info
      PetscBLASInt :: np1_bl, nrhs_bl, lda_hnt_bl, ldb_bl, lda_hn_bl, one_bl
      PetscReal, dimension(degree+1,degree) :: H
      PetscReal, dimension(degree,degree+1) :: H_T
      PetscReal, dimension(degree) :: e_d, solution, s
      PetscBLASInt, dimension(:), allocatable :: iwork_allocated
      PetscReal, dimension(:), allocatable :: work
      PetscReal, dimension(:,:), allocatable :: VL, VR
      logical :: use_harmonic_ritz = .TRUE.
      PetscReal :: rcond = PFLARE_TOL_RCOND, rel_tol, abs_tol, H_norm

      ! ~~~~~~

      H = H_in(1:degree+1, 1:degree)

      ! ~~~~~~~~~~~
      ! Now the Ritz values are just the eigenvalues of the square part of H
      ! We're actually going to use the Harmonic Ritz values which are the reciprocals
      ! of the (ordinary) Ritz values of A^-1
      ! see Embree - Polynomial preconditioned arnoldi with stability control 
      ! and Goossens - Ritz and Harmonic Ritz Values and the Convergence of FOM and GMRES
      ! ~~~~~~~~~~~
      if (use_harmonic_ritz) then

         e_d = 0
         e_d(degree) = 1

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
         H_T = transpose(H)

         allocate(work(1))
         allocate(iwork_allocated(1))
         lwork = -1         

         ! We have rcond = PFLARE_TOL_RCOND which is used to decide what singular values to drop
         ! Matlab uses max(size(A))*eps(norm(A)) in their pinv
         ! Kind-correct BLAS integer dimensions
         np1_bl = degree
         nrhs_bl = 1
         lda_hnt_bl = size(H_T, 1)
         ldb_bl = size(e_d)
         call PFLAREgelsd(np1_bl, np1_bl, nrhs_bl, H_T, lda_hnt_bl, &
                        e_d, ldb_bl, s, rcond, rank, &
                        work, lwork, iwork_allocated, info)
         ! Workspace query returns optimal sizes in work(1)/iwork(1); int() is exact
         ! for all plausible sizes < 2^24 even when work is single precision
         lwork = int(work(1))
         iwork_size = iwork_allocated(1)
         deallocate(work, iwork_allocated)
         allocate(work(lwork))
         allocate(iwork_allocated(iwork_size))
         call PFLAREgelsd(np1_bl, np1_bl, nrhs_bl, H_T, lda_hnt_bl, &
                        e_d, ldb_bl, s, rcond, rank, &
                        work, lwork, iwork_allocated, info)
         deallocate(work, iwork_allocated)

         ! Copy in the solution
         solution = e_d

         if (info /= 0) then
            print *, "Harmonic Ritz solve failed"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
         end if

         ! Scale f by H(d+1,d)^2
         solution = solution * H(degree + 1, degree)**2
         ! Add to the last column of H
         H(1:degree, degree) = &
            H(1:degree, degree) + solution
      end if

      ! ~~~~~~~~~~~~~~
      ! Now compute the eigenvalues of the square part of H
      ! ie either compute the Ritz or Harmonic Ritz values
      ! ~~~~~~~~~~~~~~

      ! The tolerances that control the clustering below are relative to ||H||
      ! This has to be computed before the eigenvalue solve, as PFLAREgeev overwrites H
      ! (after balancing, and at higher order with values that don't scale with the matrix)
      H_norm = norm2(H(1:min(m, degree),1:min(m, degree)))

      allocate(work(1))
      ! Not used but we have to allocate
      allocate(VL(0, 0))
      allocate(VR(0, 0))
      lwork = -1      

      ! Compute the eigenvalues
      ! Kind-correct BLAS integer dimensions
      np1_bl = degree
      lda_hn_bl = size(H, 1)
      one_bl = 1
      ! DEFERRED (complex-prep): coefficients kept PetscReal by design; see docs/plan.
      ! PFLAREgeev writes the strictly-real eigenvalue parts wr/wi directly into
      ! coefficients(:,1)/(:,2) - these are LAPACK real arrays, so PetscReal is correct.
      call PFLAREgeev('N', 'N', np1_bl, H, lda_hn_bl, &
                     real_roots, imag_roots, VL, one_bl, VR, one_bl, &
                     work, lwork, info)
      ! int() is exact for all plausible sizes < 2^24 even when work is single precision
      lwork = int(work(1))
      deallocate(work)
      allocate(work(lwork))
      call PFLAREgeev('N', 'N', np1_bl, H, lda_hn_bl, &
                     real_roots, imag_roots, VL, one_bl, VR, one_bl, &
                     work, lwork, info)
      deallocate(work, VL, VR)

      if (info /= 0) then
         print *, "Eig decomposition failed"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! ~~~~~~~~~~~~~~
      ! Now we have to check the output eigenvalues
      ! ~~~~~~~~~~~~~~

      ! These are the tolerances that control the clustering
      ! These are all relative to ||H|| (computed above, before H is overwritten), 
      ! which scales with the matrix, so the roots (and hence the polynomial) 
      ! do not depend on the scale of the matrix
      rel_tol = sqrt(PFLARE_EPS)
      abs_tol = PFLARE_EPS * H_norm

      ! In some cases with numerical rank deficiency, we can still
      ! end up with non-zero (or negative) eigenvalues that
      ! are trivially small - we set them explicitly to zero
      ! Everywhere downstream (the matrix-free apply, the assembled builds, 
      ! the matvec counts) then only skips roots that are exactly zero
      ! A complex root that survives this is never skipped
      do i_loc = 1, degree
         if (real_roots(i_loc)**2 + imag_roots(i_loc)**2 < &
            (abs_tol + rel_tol*H_norm)**2) then
            real_roots(i_loc) = 0d0
            imag_roots(i_loc) = 0d0
         end if
      end do       
           
      ! ~~~~~~~~~~~~~~
      ! Cluster close eigenvalues together to improve stability of the polynomial evaluation
      ! For example when computing the e'vals of a constant diagonal matrix
      ! the rank revealing factorisation above doesn't always report a rank of 1 given roundoff
      ! Instead it returns multiple eigenvalues that are very close to each other, 
      ! and we want to cluster those together and treat them as one root
      ! ~~~~~~~~~~~~~~      

      ! This places all exactly zero eigenvalues at the end of coefficients
      call cluster_eigenvalues_stable(real_roots, imag_roots, rel_tol, abs_tol)      

   end subroutine harmonic_ritz_roots

! -------------------------------------------------------------------------------------------------------------------------------

end module gmres_poly_newton_roots
