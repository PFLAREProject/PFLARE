module gmres_poly_newton

   use petscmat
   use gmres_poly
   use c_petsc_interfaces, only: mat_mult_powers_share_sparsity_newton_kokkos, &
         sstep_project_gram_kokkos, sstep_apply_inverse_kokkos, sstep_gemm_warmup_kokkos, &
         sstep_fused_dots_kokkos, sstep_project_kokkos
   use pflare_parameters, only: PFLARE_TOL_RCOND, &
         PFLARE_TOL_CONSISTENCY, PFLARE_EPS, PFLARE_TOL_LUCKY, &
         PFLARE_ONE, PFLARE_ZERO, PFLARE_MINUS_ONE, PFLARE_TWO, PFLARE_MATMULT_FILL, &
         PFLARE_TOL_MATFREE_NEWTON, PFLARE_TOL_LEJA_PERTURB
   use gmres_poly_apply, only: petsc_matvec_poly_newton_mf, &
         petsc_matvec_right_scale_poly_newton_mf, petsc_matvec_da_poly_mf, &
         petsc_matvec_poly_transpose_mf

#include "petsc/finclude/petscmat.h"
#include "finclude/pflare_blaslapack.h"

   implicit none
   public

   ! How the roots of the Newton polynomial are chosen in calculate_gmres_polynomial_roots_newton
   ! NEWTON_ROOTS_NONE     - no extra roots
   ! NEWTON_ROOTS_POF      - extra roots from the product of factors of each root
   ! NEWTON_ROOTS_ADAPTIVE - the degree and the extra roots are chosen from the Arnoldi Hessenberg
   integer, parameter :: NEWTON_ROOTS_NONE = 0
   integer, parameter :: NEWTON_ROOTS_POF = 1
   integer, parameter :: NEWTON_ROOTS_ADAPTIVE = 2

   ! TEMPORARY: timers for profiling the degree search of the adaptive rule
   PetscLogDouble :: prof_ritz = 0, prof_extra = 0, prof_leja = 0, prof_replay = 0
   integer :: prof_candidates = 0, prof_replays = 0, prof_copy_all = 0
   
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

   subroutine adaptive_newton_candidate_inner(H, degree, m, rho, accepted, real_roots_out, imag_roots_out, n_base)

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
      PetscLogDouble :: t0, t1
      PetscErrorCode :: ierr

      ! ~~~~~~

      if (allocated(real_roots_out)) deallocate(real_roots_out, imag_roots_out)
      accepted = .TRUE.

      ! Harmonic Ritz values of the leading block, with the non-zero roots first
      prof_candidates = prof_candidates + 1
      call PetscTime(t0, ierr)
      call harmonic_ritz_roots(H, degree, m, roots(:, 1), roots(:, 2))
      call PetscTime(t1, ierr)
      prof_ritz = prof_ritz + t1 - t0
      n_base = 0
      do i_loc = 1, degree
         if (roots(i_loc, 1) /= 0d0 .OR. roots(i_loc, 2) /= 0d0) n_base = n_base + 1
      end do

      if (n_base == 0) then
         allocate(real_roots_out(0), imag_roots_out(0))
         return
      end if

      ! Extra copies only of the roots where a copy cannot amplify the polynomial
      call PetscTime(t0, ierr)
      call compute_extra_roots_adaptive(roots(1:n_base, 1), roots(1:n_base, 2), rho, .FALSE., &
               sensitive, over_cap, real_added, imag_added)
      call PetscTime(t1, ierr)
      prof_extra = prof_extra + t1 - t0

      ! If one of the roots we didn't copy is too sensitive, see if copying
      ! every sensitive root gives a polynomial that is still good
      ! We can't lower the degree any further than one, so we just take what we have there
      copied_all = sensitive .AND. degree > 1
      if (copied_all) then

         deallocate(real_added, imag_added)
         prof_copy_all = prof_copy_all + 1
         call PetscTime(t0, ierr)
         call compute_extra_roots_adaptive(roots(1:n_base, 1), roots(1:n_base, 2), rho, .TRUE., &
                  sensitive, over_cap, real_added, imag_added)
         call PetscTime(t1, ierr)
         prof_extra = prof_extra + t1 - t0
         if (over_cap) then
            accepted = .FALSE.
            return
         end if
      end if

      call PetscTime(t0, ierr)
      call leja_order_with_extras(roots(1:n_base, 1), roots(1:n_base, 2), real_added, imag_added, indices)
      call PetscTime(t1, ierr)
      prof_leja = prof_leja + t1 - t0
      allocate(real_roots_out(size(real_added)), imag_roots_out(size(real_added)))
      real_roots_out = real_added(indices)
      imag_roots_out = imag_added(indices)
      deallocate(real_added, imag_added, indices)

      ! The copies we were not sure about are only kept if the polynomial
      ! still does about as well as GMRES on the Arnoldi start vector
      if (copied_all) then
         prof_replays = prof_replays + 1
         call PetscTime(t0, ierr)
         call hessenberg_newton_replay(H, degree, real_roots_out, imag_roots_out, residual)
         call PetscTime(t1, ierr)
         prof_replay = prof_replay + t1 - t0
         if (residual > 2 * rho) then
            accepted = .FALSE.
            deallocate(real_roots_out, imag_roots_out)
         end if
      end if

   end subroutine adaptive_newton_candidate_inner

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine adaptive_newton_candidate(H, degree, m, rho, accepted, real_roots_out, imag_roots_out, n_base)

      ! TEMPORARY wrapper that prints each candidate degree of the search

      PetscReal, dimension(:,:), intent(in)               :: H
      integer, intent(in)                                 :: degree, m
      PetscReal, intent(in)                               :: rho
      logical, intent(out)                                :: accepted
      PetscReal, dimension(:), allocatable, intent(inout) :: real_roots_out, imag_roots_out
      integer, intent(out)                                :: n_base

      PetscLogDouble :: t0, t1
      PetscErrorCode :: ierr
      PetscBool :: view_choice

      call PetscTime(t0, ierr)
      call adaptive_newton_candidate_inner(H, degree, m, rho, accepted, real_roots_out, imag_roots_out, n_base)
      call PetscTime(t1, ierr)
      call PetscOptionsHasName(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, &
               '-pflare_newton_adaptive_view', view_choice, ierr)
      if (view_choice) print '(A,I0,A,L1,A,ES9.2,A,ES9.2)', "Search candidate: degree ", degree, " accepted ", accepted, &
            " residual ", rho, " time ", t1 - t0

   end subroutine adaptive_newton_candidate

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine adaptive_newton_roots(H, m, real_roots_out, imag_roots_out, n_base)

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

      ! Local variables
      integer :: degree, bad, good, mid, step, n_base_trial, degree_floor
      PetscBool :: view_choice
      PetscErrorCode :: ierr
      PetscReal, dimension(m) :: rho_h
      PetscReal, dimension(:), allocatable :: real_trial, imag_trial
      logical :: accepted

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

      ! Temporary option: print how the degree was chosen (on every rank)
      call PetscOptionsHasName(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, &
               '-pflare_newton_adaptive_view', view_choice, ierr)
      degree_floor = degree

      call adaptive_newton_candidate(H, degree, m, rho_h(degree), accepted, real_roots_out, imag_roots_out, n_base)
      if (accepted) then
         if (view_choice) print '(A,I0,A,I0,A,I0,A,ES9.2,A,ES9.2,A,ES9.2,A,ES9.2)', "Search totals: candidates ", prof_candidates, &
               " copy-all passes ", prof_copy_all, " replays ", prof_replays, " harmonic Ritz ", prof_ritz, &
               " extra roots ", prof_extra, " leja ", prof_leja, " replay ", prof_replay
         if (view_choice) print '(A,I0,A,I0,A,ES9.2,A,ES9.2,A,I0,A,I0)', "Adaptive degree: steps ", m, &
               " residual stops improving at ", degree_floor, " residual there ", rho_h(degree_floor), &
               " at the last step ", rho_h(m), " lowered for stability to ", degree_floor, " roots ", n_base
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

      if (view_choice) print '(A,I0,A,I0,A,I0,A,ES9.2,A,ES9.2,A,ES9.2,A,ES9.2)', "Search totals: candidates ", prof_candidates, &
            " copy-all passes ", prof_copy_all, " replays ", prof_replays, " harmonic Ritz ", prof_ritz, &
            " extra roots ", prof_extra, " leja ", prof_leja, " replay ", prof_replay
      if (view_choice) print '(A,I0,A,I0,A,ES9.2,A,ES9.2,A,I0,A,I0,A,ES9.2)', "Adaptive degree: steps ", m, &
            " residual stops improving at ", degree_floor, " residual there ", rho_h(degree_floor), &
            " at the last step ", rho_h(m), " lowered for stability to ", good, " roots ", n_base, &
            " residual there ", rho_h(good)

   end subroutine adaptive_newton_roots

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

   subroutine partial_cholesky(G, n, omega, p, R, t_diag)

      ! Column by column Cholesky G = R^T R of the leading n x n block of G, that stops before
      ! the first column that gives a non-positive pivot or a 2-norm condition number of
      ! R bigger than omega
      ! Returns the number of columns kept in p and the factor in R(1:p, 1:p)
      ! If t_diag is given, G is the Gram matrix of a block after projection that was computed
      ! by Pythagoras, and t_diag the squared norms of its columns before projection. The error
      ! in G is then relative to t_diag, so we also stop once the biggest of those norms over
      ! the smallest singular value of R passes omega

      ! ~~~~~~
      real(kind=kind(1d0)), dimension(:,:), intent(in)    :: G
      integer, intent(in)                                 :: n
      real(kind=kind(1d0)), intent(in)                    :: omega
      integer, intent(out)                                :: p
      real(kind=kind(1d0)), dimension(:,:), intent(inout) :: R
      real(kind=kind(1d0)), dimension(:), intent(in), optional :: t_diag

      ! Local variables
      integer :: i_loc, j_loc, errorcode
      real(kind=kind(1d0)) :: pivot, norm_before
      PetscReal, dimension(n, n) :: R_copy
      PetscReal, dimension(n) :: sigma
      PetscReal, dimension(1, 1) :: U, VT
      PetscReal, dimension(:), allocatable :: work
      ! BLAS/LAPACK integer arguments must be PetscBLASInt
      PetscBLASInt :: lwork, j_bl, lda_bl, one_bl, info

      ! ~~~~~~

      R(1:n, 1:n) = 0
      p = 0
      lwork = 10 * max(1, n)
      allocate(work(lwork))
      lda_bl = n
      one_bl = 1

      do j_loc = 1, n

         do i_loc = 1, j_loc - 1
            R(i_loc, j_loc) = (G(i_loc, j_loc) - &
                     dot_product(R(1:i_loc-1, i_loc), R(1:i_loc-1, j_loc)))/R(i_loc, i_loc)
         end do
         pivot = G(j_loc, j_loc) - dot_product(R(1:j_loc-1, j_loc), R(1:j_loc-1, j_loc))
         ! The comparisons are false for a NaN
         if (.NOT. (pivot > 0 .AND. pivot < huge(pivot))) exit
         R(j_loc, j_loc) = sqrt(pivot)

         ! The condition number of the factor so far
         R_copy(1:j_loc, 1:j_loc) = real(R(1:j_loc, 1:j_loc), kind=kind(sigma))
         j_bl = int(j_loc, kind=kind(j_bl))
         call PFLAREgesvd('N', 'N', j_bl, j_bl, R_copy, lda_bl, &
                  sigma, U, one_bl, VT, one_bl, work, lwork, info)
         if (info /= 0) then
            print *, "SVD in the partial Cholesky failed"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
         end if
         if (.NOT. (sigma(j_loc) > 0)) exit
         if (.NOT. (sigma(1)/sigma(j_loc) <= omega)) exit
         if (present(t_diag)) then
            norm_before = sqrt(max(maxval(t_diag(1:j_loc)), real(sigma(1), kind=kind(1d0))**2))
            if (.NOT. (norm_before/sigma(j_loc) <= omega)) exit
         end if

         p = j_loc
      end do

      deallocate(work)

   end subroutine partial_cholesky

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine sstep_estimate_block_size(shift_real, shift_imag, point_real, point_imag, mean, ritz_max, &
                  s_max, omega, s_est)

      ! Estimates the biggest first block the s-step Arnoldi can keep, after Section 3.5 of
      ! Xu, Alonso & Darve, with no matvecs or reductions
      ! The points stand in for the spectrum. We follow the size of the scaled Newton polynomial
      ! with the given shifts (cycled as in arnoldi_sstep) at each point as its degree grows, and
      ! s_est is the last degree before the norm of those sizes, relative to degree zero, passes omega
      ! A shift that is exactly one of the points cancels it down to rounding level, once
      ! This is only printed, not used: in testing it overestimated the block kept by up to several times

      ! ~~~~~~
      PetscReal, dimension(:), intent(in)  :: shift_real, shift_imag, point_real, point_imag
      real(kind=kind(1d0)), intent(in)     :: mean, ritz_max, omega
      integer, intent(in)                  :: s_max
      integer, intent(out)                 :: s_est

      ! Local variables
      integer, parameter :: dp = kind(1d0)
      integer :: j, i_loc, n_shift, n_point
      real(kind=dp) :: x, y, gam, dist, dist_conj, eps
      real(kind=dp), dimension(size(point_real)) :: E, E_mid
      logical, dimension(size(point_real)) :: cancelled

      ! ~~~~~~

      n_shift = size(shift_real)
      n_point = size(point_real)
      eps = real(PFLARE_EPS, kind=dp)
      E = 1
      cancelled = .FALSE.
      s_est = s_max

      j = 0
      do while (j < s_max)

         x = real(shift_real(mod(j, n_shift) + 1), kind=dp)
         y = real(shift_imag(mod(j, n_shift) + 1), kind=dp)
         gam = max(sqrt((mean - x)**2 + y**2), 1d-3 * ritz_max)

         ! The column made with the real part of the shift
         do i_loc = 1, n_point
            dist = sqrt((real(point_real(i_loc), kind=dp) - x)**2 + real(point_imag(i_loc), kind=dp)**2)
            E_mid(i_loc) = E(i_loc) * dist/gam
         end do

         if (y == 0d0 .OR. j + 2 > s_max) then

            ! A real shift
            do i_loc = 1, n_point
               if (E_mid(i_loc) == 0d0 .AND. y == 0d0) then
                  if (.NOT. cancelled(i_loc)) E(i_loc) = E(i_loc) * eps
                  cancelled(i_loc) = .TRUE.
               else
                  E(i_loc) = E_mid(i_loc)
               end if
            end do
            j = j + 1
            if (.NOT. (norm2(E)/sqrt(real(n_point, kind=dp)) < omega)) then
               s_est = j - 1
               return
            end if

         else

            ! A conjugate pair, which is two columns
            if (.NOT. (norm2(E_mid)/sqrt(real(n_point, kind=dp)) < omega)) then
               s_est = j
               return
            end if
            do i_loc = 1, n_point
               dist = sqrt((real(point_real(i_loc), kind=dp) - x)**2 + (real(point_imag(i_loc), kind=dp) - y)**2)
               dist_conj = sqrt((real(point_real(i_loc), kind=dp) - x)**2 + (real(point_imag(i_loc), kind=dp) + y)**2)
               if (dist == 0d0 .OR. dist_conj == 0d0) then
                  if (.NOT. cancelled(i_loc)) E(i_loc) = E(i_loc) * eps
                  cancelled(i_loc) = .TRUE.
               else
                  E(i_loc) = E(i_loc) * dist * dist_conj/gam**2
               end if
            end do
            j = j + 2
            if (.NOT. (norm2(E)/sqrt(real(n_point, kind=dp)) < omega)) then
               s_est = j - 1
               return
            end if
         end if
      end do

   end subroutine sstep_estimate_block_size

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine arnoldi_extend(matrix, lucky_tol, V_n, w_j, H_n, m_done, m_max, m, passes, floor_stop)

      ! Continues an Arnoldi that has done m_done steps up to m_max steps, one step at a time
      ! in the same way as the arnoldi routine
      ! V_n(1:m_done+1) must hold the basis and H_n(1:m_done+1, 1:m_done) the Hessenberg matrix
      ! Returns the number of steps done in m, which is less than m_max only if we hit
      ! the lucky tolerance
      ! passes is the number of Gram-Schmidt passes in each step (one if not given)
      ! If floor_stop is given and true we also stop once the relative GMRES residual is at
      ! rounding level, with m the steps done and V_n(m+1) set. Further steps are only noise

      ! ~~~~~~
      type(tMat), intent(in)                    :: matrix
      PetscReal, intent(in)                     :: lucky_tol
      type(tVec), dimension(:), intent(in)      :: V_n
      type(tVec), intent(in)                    :: w_j
      PetscReal, dimension(:,:), intent(inout)  :: H_n
      integer, intent(in)                       :: m_done, m_max
      integer, intent(out)                      :: m
      integer, intent(in), optional             :: passes
      logical, intent(in), optional             :: floor_stop

      ! Local variables
      PetscErrorCode :: ierr
      PetscReal, dimension(m_max) :: neg_h
      PetscInt :: m_petscint
      integer :: n_passes, i_pass, j_loc
      logical :: stop_at_floor
      PetscReal, dimension(m_max) :: cs, sn
      PetscReal :: g_last, rho

      ! ~~~~~~

      n_passes = 1
      if (present(passes)) n_passes = passes
      stop_at_floor = .FALSE.
      if (present(floor_stop)) stop_at_floor = floor_stop

      ! The Givens rotations of the columns of H we already have
      g_last = 1
      if (stop_at_floor) then
         do j_loc = 1, m_done
            call rotate_column(j_loc, rho)
         end do
      end if

      do m = m_done + 1, m_max

         m_petscint = m
         call MatMult(matrix, V_n(m), w_j, ierr)

         H_n(1:m, m) = 0
         do i_pass = 1, n_passes
            call VecMDot(w_j, m_petscint, V_n, neg_h, ierr)
            H_n(1:m, m) = H_n(1:m, m) + neg_h(1:m)
            neg_h(1:m) = -neg_h(1:m)
            call VecMAXPY(w_j, m_petscint, neg_h, V_n, ierr)
         end do

         call VecNorm(w_j, NORM_2, H_n(m+1, m), ierr)
         if (H_n(m+1, m) < lucky_tol) return

         call VecAXPBY(V_n(m + 1), PFLARE_ONE/H_n(m+1, m), PFLARE_ZERO, w_j, ierr)

         if (stop_at_floor) then
            call rotate_column(m, rho)
            if (rho <= 10 * PFLARE_EPS) return
         end if
      end do

      ! Fortran leaves the loop variable one bigger if we get to the end of the loop
      m = m_max

      contains

      subroutine rotate_column(j_col, rho_col)

         ! Applies the Givens rotations so far to column j_col of H, adds the one that zeros
         ! its subdiagonal and returns the relative GMRES residual after j_col steps
         ! (as hessenberg_gmres_residuals does for the whole matrix)

         integer, intent(in)    :: j_col
         PetscReal, intent(out) :: rho_col

         integer :: i_col
         PetscReal, dimension(j_col + 1) :: col
         PetscReal :: temp, denom

         col = H_n(1:j_col + 1, j_col)
         do i_col = 1, j_col - 1
            temp = cs(i_col) * col(i_col) + sn(i_col) * col(i_col + 1)
            col(i_col + 1) = -sn(i_col) * col(i_col) + cs(i_col) * col(i_col + 1)
            col(i_col) = temp
         end do
         denom = sqrt(col(j_col)**2 + col(j_col + 1)**2)
         if (denom == 0) then
            cs(j_col) = 1
            sn(j_col) = 0
         else
            cs(j_col) = col(j_col)/denom
            sn(j_col) = col(j_col + 1)/denom
         end if
         g_last = -sn(j_col) * g_last
         rho_col = abs(g_last)

      end subroutine rotate_column

   end subroutine arnoldi_extend

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine arnoldi_sstep(matrix, poly_order, s_max, m_start, lucky_tol, V_n, w_j, beta, H_n, m, view, kernels, &
                  floor_stop)

      ! An s-step Arnoldi that returns the same Hessenberg matrix as the arnoldi routine
      ! (to rounding) with far fewer reductions. This is the adaptive s-step method of
      ! Xu, Alonso & Darve, "A numerically stable communication-avoiding s-step GMRES algorithm"
      !
      ! The first m_start steps are an ordinary Arnoldi. The Ritz values of those, leja ordered,
      ! are the shifts of a scaled Newton basis. After that we build a block of up to s new
      ! Krylov vectors with matvecs only, and orthogonalise the block in two passes. Each pass
      ! projects out the existing basis and does a Cholesky QR of the block, which is two
      ! reductions, so a block costs four reductions however big it is
      ! The Cholesky stops at the first column where the block becomes too ill-conditioned.
      ! The columns after that are thrown away and the block size is reduced to the number
      ! kept. It is never increased again
      ! If the block size goes below three the rest of the steps are an ordinary Arnoldi,
      ! which costs no more reductions than the blocks would
      !
      ! The block is built in place in V_n, so this needs no more vectors than the arnoldi routine
      ! If kernels is not zero and the vectors in V_n are slices of one device buffer (see
      ! create_temp_space_box_muller), each pass is done as dense matrix products on the device,
      ! with KokkosBlas (1) or our own kernels (2). KokkosBlas with a rocBLAS backend was no faster
      ! and its first big gemm cost 0.2 s
      ! Otherwise it is done with PETSc vector operations one column at a time

      ! ~~~~~~
      type(tMat), intent(in)                    :: matrix
      integer, intent(in)                       :: poly_order, s_max, m_start
      PetscReal, intent(in)                     :: lucky_tol
      type(tVec), dimension(:), intent(in)      :: V_n
      type(tVec), intent(in)                    :: w_j
      PetscReal, intent(out)                    :: beta
      PetscReal, dimension(:,:), intent(inout)  :: H_n
      integer, intent(out)                      :: m
      logical, intent(in)                       :: view, floor_stop
      integer, intent(in)                       :: kernels

      ! Local variables
      integer, parameter :: dp = kind(1d0)
      logical :: use_kernels
      integer(c_long_long), dimension(:), allocatable :: vec_handles
      integer :: k, m0, j, s, sb, p, p_two, i_loc, col_loc, n_old, errorcode
      integer :: n_blocks, n_wasted, n_reductions, n_single, s_est_ritz, s_est_harmonic, p_keep, n_cut
      real(kind=kind(1d0)) :: worst_ratio, ratio_max, worst_relation, ratio_before
      real(kind=kind(1d0)), dimension(:), allocatable :: column_ratio
      PetscReal :: relation_error, relation_tol
      integer :: n_tightened, n_extra_matvecs, p_block, n_resume, n_resume_start, m_target, n_stretches
      integer :: n_tightened_before
      logical :: lucky
      PetscReal, dimension(poly_order + 1) :: rho_block
      PetscReal :: ratio_option
      PetscBool :: ratio_found, discard_start, check_blocks
      integer, dimension(:), allocatable :: kept_history
      PetscLogDouble :: time_start, time_end, time_block, time_host, time_mark
      PetscLogDouble, dimension(:), allocatable :: time_history
      PetscErrorCode :: ierr
      PetscInt :: n_petscint
      PetscReal, dimension(:), allocatable :: ritz_real, ritz_imag, shift_real, shift_imag, work, dots
      PetscReal, dimension(:,:), allocatable :: H_copy, VL, VR
      integer, dimension(:), allocatable :: indices
      real(kind=dp), dimension(:,:), allocatable :: B, X_one, X_two, G, Z, Z_two, C, R_new
      real(kind=dp) :: x, y, y_sq, gam, mean, ritz_max, omega
      PetscReal :: alpha_r, beta_r, gamma_r
      PetscBLASInt :: m_bl, lda_bl, one_bl, lwork, info

      ! ~~~~~~

      ! The number of Arnoldi steps we want
      k = poly_order + 1
      m0 = max(1, min(m_start, k))
      s = max(1, s_max)
      allocate(kept_history(k))
      call PetscTime(time_start, ierr)
      use_kernels = kernels /= 0
      time_host = 0
      allocate(time_history(k))
      allocate(vec_handles(size(V_n)))
      do i_loc = 1, size(V_n)
         vec_handles(i_loc) = V_n(i_loc)%v
      end do

      ! ~~~~~~~~~
      ! Ordinary Arnoldi for the first m0 steps
      ! ~~~~~~~~~
      call arnoldi(matrix, m0 - 1, lucky_tol, V_n, w_j, beta, H_n, m)
      n_reductions = 1 + 2 * m
      n_single = m
      n_blocks = 0
      n_wasted = 0
      n_cut = 0
      worst_ratio = 0
      n_tightened = 0
      n_stretches = 0
      lucky = .FALSE.
      ! The number of ordinary steps before going back to blocks
      n_resume_start = 10
      n_resume = n_resume_start
      n_extra_matvecs = 0
      worst_relation = 0
      ! A block is cut shorter if the relative error in A v = V H for its last column passes this
      relation_tol = 1d-1 * sqrt(PFLARE_EPS)
      ! Temporary option for the bound on the ratio that controls the accuracy of H
      ! Temporary option: without the check the blocks are only cut by ratio_max, and with a big
      ! -pflare_newton_sstep_ratio as well this is the algorithm of the paper
      check_blocks = PETSC_TRUE
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, &
               '-pflare_newton_sstep_check', check_blocks, ratio_found, ierr)
      ratio_option = -1
      call PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, &
               '-pflare_newton_sstep_ratio', ratio_option, ratio_found, ierr)
      ! The error made in one block is about eps times this ratio, so it starts where that is
      ! well below the tolerance
      ratio_max = 1d-3/sqrt(real(PFLARE_EPS, kind=dp))
      if (ratio_option > 0) ratio_max = real(ratio_option, kind=dp)
      ! Have we finished or hit the lucky tolerance
      if (m == k .OR. H_n(m+1, m) < lucky_tol) then
         if (view) call view_counts()
         return
      end if

      ! ~~~~~~~~~
      ! The Ritz values of the first m0 steps are the shifts
      ! ~~~~~~~~~
      allocate(H_copy(m0, m0), ritz_real(m0), ritz_imag(m0), VL(0, 0), VR(0, 0))
      H_copy = H_n(1:m0, 1:m0)
      m_bl = int(m0, kind=kind(m_bl))
      lda_bl = m_bl
      one_bl = 1
      lwork = int(10 * m0, kind=kind(lwork))
      allocate(work(lwork))
      call PFLAREgeev('N', 'N', m_bl, H_copy, lda_bl, &
                     ritz_real, ritz_imag, VL, one_bl, VR, one_bl, &
                     work, lwork, info)
      deallocate(work, VL, VR, H_copy)
      if (info /= 0) then
         print *, "Eig decomposition failed"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! Leja order, which keeps the conjugate pairs next to each other
      allocate(shift_real(m0), shift_imag(m0))
      shift_real = ritz_real
      shift_imag = ritz_imag
      call modified_leja(shift_real, shift_imag, indices)
      shift_real = ritz_real(indices)
      shift_imag = ritz_imag(indices)
      deallocate(indices)

      ! The scaling of each basis vector is the distance of its shift from the centre
      mean = sum(real(ritz_real, kind=dp))/m0
      ritz_max = maxval(sqrt(real(ritz_real, kind=dp)**2 + real(ritz_imag, kind=dp)**2))

      ! The block is kept if its condition number is below this, which keeps the
      ! Cholesky QR stable in this precision
      omega = 0.1d0/sqrt(real(PFLARE_EPS, kind=dp))

      ! Estimates of the first block size, with the Ritz values and the harmonic Ritz values
      ! standing in for the spectrum
      call sstep_estimate_block_size(shift_real, shift_imag, ritz_real, ritz_imag, mean, ritz_max, &
               k - m, omega, s_est_ritz)
      call harmonic_ritz_roots(H_n, m0, m0, ritz_real, ritz_imag)
      call sstep_estimate_block_size(shift_real, shift_imag, ritz_real, ritz_imag, mean, ritz_max, &
               k - m, omega, s_est_harmonic)
      deallocate(ritz_real, ritz_imag)

      allocate(dots(k + 1))
      allocate(B(s + 1, s), X_one(k + 1, s), X_two(k + 1, s), G(s, s), Z(s, s), Z_two(s, s))

      ! Temporary option: throw the start up steps away once we have their Ritz values and build
      ! the whole basis in blocks from the start vector, as the paper does within a restart cycle
      discard_start = PETSC_FALSE
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, &
               '-pflare_newton_sstep_discard_start', discard_start, ratio_found, ierr)
      if (discard_start) then
         m = 0
         H_n = 0
         n_single = 0
      end if

      ! j is the number of Arnoldi steps finished, so V_n(1:j+1) is the basis so far
      j = m
      do while (j < k)

         ! Blocks this small cost as many reductions as ordinary steps
         if (s < 3) exit

         sb = min(s, k - j)
         n_old = j + 1
         n_blocks = n_blocks + 1
         call PetscTime(time_block, ierr)

         ! ~~~~~~~~~
         ! Build the block with the scaled Newton recurrence
         ! Column i of the block lives in V_n(j+1+i), with column 0 the last basis vector
         ! B holds the recurrence: A W(:, 0:sb-1) = W(:, 0:sb) B
         ! ~~~~~~~~~
         B = 0
         i_loc = 0
         do while (i_loc < sb)

            ! We cycle through the shifts if there are fewer than the block size
            x = real(shift_real(mod(i_loc, m0) + 1), kind=dp)
            y = real(shift_imag(mod(i_loc, m0) + 1), kind=dp)
            gam = max(sqrt((mean - x)**2 + y**2), 1d-3 * ritz_max)

            ! w_{i+1} = (A - x) w_i / gam
            call MatMult(matrix, V_n(j + 1 + i_loc), V_n(j + 2 + i_loc), ierr)
            alpha_r = real(-x/gam, kind=kind(alpha_r))
            beta_r = real(1d0/gam, kind=kind(beta_r))
            call VecAXPBY(V_n(j + 2 + i_loc), alpha_r, beta_r, V_n(j + 1 + i_loc), ierr)
            B(i_loc + 1, i_loc + 1) = x
            B(i_loc + 2, i_loc + 1) = gam

            ! A real shift, or a conjugate pair cut in half by the end of the block
            ! in which case we have just used its real part
            if (y == 0d0 .OR. i_loc + 2 > sb) then
               i_loc = i_loc + 1
               cycle
            end if

            ! The second member of a conjugate pair in real arithmetic
            ! w_{i+2} = ((A - x) w_{i+1} + (y^2/gam) w_i) / gam
            y_sq = y**2
            call MatMult(matrix, V_n(j + 2 + i_loc), V_n(j + 3 + i_loc), ierr)
            alpha_r = real(-x/gam, kind=kind(alpha_r))
            beta_r = real(y_sq/gam**2, kind=kind(beta_r))
            gamma_r = real(1d0/gam, kind=kind(gamma_r))
            call VecAXPBYPCZ(V_n(j + 3 + i_loc), alpha_r, beta_r, gamma_r, &
                     V_n(j + 2 + i_loc), V_n(j + 1 + i_loc), ierr)
            B(i_loc + 2, i_loc + 2) = x
            B(i_loc + 3, i_loc + 2) = gam
            B(i_loc + 1, i_loc + 2) = -y_sq/gam
            i_loc = i_loc + 2
         end do

         ! ~~~~~~~~~
         ! First pass: project out the existing basis and Cholesky QR the block
         ! ~~~~~~~~~
         call project_and_gram(sb, X_one)
         call PetscTime(time_mark, ierr)
         call partial_cholesky(G, sb, omega, p, Z)
         call PetscTime(time_end, ierr)
         time_host = time_host + time_end - time_mark
         n_reductions = n_reductions + 2
         if (p == 0) then
            n_wasted = n_wasted + sb
            kept_history(n_blocks) = 0
            time_history(n_blocks) = 0
            exit
         end if
         call apply_inverse_factor(p, Z)

         ! ~~~~~~~~~
         ! Second pass on the columns we kept
         ! ~~~~~~~~~
         call project_and_gram(p, X_two)
         call PetscTime(time_mark, ierr)
         call partial_cholesky(G, p, omega, p_two, Z_two)
         call PetscTime(time_end, ierr)
         time_host = time_host + time_end - time_mark
         n_reductions = n_reductions + 2
         if (p_two == 0) then
            n_wasted = n_wasted + sb
            kept_history(n_blocks) = 0
            time_history(n_blocks) = 0
            exit
         end if
         p = p_two
         call apply_inverse_factor(p, Z_two)
         n_wasted = n_wasted + sb - p
         kept_history(n_blocks) = p

         ! ~~~~~~~~~
         ! Recover the new columns of the Hessenberg matrix
         ! The block before orthogonalising is W(:, 1:p) = V_n(1:j+1) C + V_n(j+2:j+p+1) R_new
         ! ~~~~~~~~~
         call PetscTime(time_mark, ierr)
         allocate(C(n_old, p), R_new(p, p), column_ratio(p))
         C = X_one(1:n_old, 1:p) + matmul(X_two(1:n_old, 1:p), Z(1:p, 1:p))
         R_new = matmul(Z_two(1:p, 1:p), Z(1:p, 1:p))

         ! The Hessenberg columns come from dividing by the diagonal of R_new, the part of each
         ! block vector that is new. The Cholesky only bounds how that varies across the block,
         ! not how small it is against the whole vector, which mostly lies in the existing basis
         ! when the Newton basis is poor. The new columns of H are derived from the existing
         ! ones, whose errors are multiplied by about that ratio, so the error in H can grow from
         ! block to block (to order one on ill-conditioned matrices if nothing is done)
         ! So we only keep the columns where the ratio is below ratio_max, and then check the
         ! last column kept against the basis, as the error is biggest there: one matvec and
         ! one reduction to measure how well A v = V H holds for it
         ! If that has lost accuracy we keep half as many columns of this block, which needs no
         ! more work on the vectors as the columns are nested, and check again
         do col_loc = 1, p
            column_ratio(col_loc) = max(1d0, sqrt(sum(C(:, col_loc)**2) + sum(R_new(1:col_loc, col_loc)**2))) &
                     /R_new(col_loc, col_loc)
         end do
         p_block = p
         ratio_before = ratio_max
         n_tightened_before = n_tightened
         call PetscTime(time_end, ierr)
         time_host = time_host + time_end - time_mark

         do
            ! The columns within the bound
            p_keep = p
            do col_loc = 1, p
               if (.NOT. column_ratio(col_loc) <= ratio_max) then
                  p_keep = col_loc - 1
                  exit
               end if
            end do
            if (p_keep < p) n_cut = n_cut + 1
            p = p_keep
            if (p == 0) exit

            call PetscTime(time_mark, ierr)
            call recover_hessenberg(p)
            call PetscTime(time_end, ierr)
            time_host = time_host + time_end - time_mark

            n_petscint = j + p + 1
            call MatMult(matrix, V_n(j + p), w_j, ierr)
            dots(1:j + p + 1) = -H_n(1:j + p + 1, j + p)
            call VecMAXPY(w_j, n_petscint, dots, V_n, ierr)
            call VecNorm(w_j, NORM_2, relation_error, ierr)
            relation_error = relation_error/norm2(H_n(1:j + p + 1, j + p))
            n_reductions = n_reductions + 1
            n_extra_matvecs = n_extra_matvecs + 1
            if (relation_error <= relation_tol .OR. .NOT. check_blocks) then
               worst_relation = max(worst_relation, real(relation_error, kind=dp))
               worst_ratio = max(worst_ratio, maxval(column_ratio(1:p)))
               exit
            end if

            ! Not accurate enough: keep half as many columns and bound the ratio by what
            ! those have, so the blocks after this one are cut to match
            n_tightened = n_tightened + 1
            H_n(:, j + 1:j + p) = 0
            if (p == 1) then
               p = 0
               exit
            end if
            p = (p + 1)/2
            ratio_max = min(ratio_max, maxval(column_ratio(1:p)))
         end do

         ! If the check cut this block to fewer than three columns the ratio is not what is
         ! wrong either, so treat it in the same way as keeping none
         if (p > 0 .AND. p < 3 .AND. p_block >= 3 .AND. n_tightened > n_tightened_before) p = 0

         n_wasted = n_wasted + p_block - p
         kept_history(n_blocks) = p
         deallocate(C, R_new, column_ratio)
         if (p == 0) then

            ! Not even one column of this block could be kept accurately, so the error is in the
            ! columns of H we already have. Some ordinary steps measure their columns of H
            ! directly, after which the blocks no longer build on the inaccurate ones (exactly
            ! so if H is banded). We do twice as many each time this happens in a row
            time_history(n_blocks) = 0
            ! The bound goes back to what it was, as it was not the ratio that failed this block
            ratio_max = ratio_before
            H_n(:, j + 1:j + p_block) = 0
            m_target = min(k, j + n_resume)
            call arnoldi_extend(matrix, lucky_tol, V_n, w_j, H_n, j, m_target, m, 2, floor_stop)
            n_reductions = n_reductions + 3 * (m - j)
            n_single = n_single + m - j
            n_stretches = n_stretches + 1
            n_resume = 2 * n_resume
            ! Have we hit the lucky tolerance
            if (m < m_target .OR. H_n(m + 1, m) < lucky_tol) then
               j = m
               lucky = .TRUE.
               exit
            end if
            j = m
            cycle
         end if
         n_resume = n_resume_start

         call PetscTime(time_end, ierr)
         time_history(n_blocks) = time_end - time_block
         if (use_kernels .AND. kokkos_debug()) call check_block_orthogonal(p)

         j = j + p
         ! The block size never grows back
         if (p < sb) s = p

         ! Stop once the relative GMRES residual is at rounding level, as further steps are
         ! only noise. We can only test this after a block, so we may do up to a block too many
         if (floor_stop) then
            call hessenberg_gmres_residuals(H_n, j, rho_block)
            if (minval(rho_block(1:j)) <= 10 * PFLARE_EPS) then
               lucky = .TRUE.
               exit
            end if
         end if

      end do

      deallocate(shift_real, shift_imag, dots, B, X_one, X_two, G, Z, Z_two)

      ! Any steps left are an ordinary Arnoldi
      m = j
      if (j < k .AND. .NOT. lucky) then
         ! With two passes, as the blocks are orthogonalised twice
         call arnoldi_extend(matrix, lucky_tol, V_n, w_j, H_n, j, k, m, 2, floor_stop)
         n_reductions = n_reductions + 3 * (m - j)
         n_single = n_single + m - j
      end if

      if (view) call view_counts()

      contains

      subroutine project_and_gram(n_cols, X)

         ! Removes the existing basis from the first n_cols columns of the block and
         ! computes the Gram matrix of the result into G, in two reductions
         ! X returns the components removed

         integer, intent(in) :: n_cols
         real(kind=dp), dimension(:,:), intent(inout) :: X

         integer :: col
         PetscReal, dimension(k + 1, n_cols) :: X_work
#if defined(PETSC_HAVE_KOKKOS)
         PetscReal, dimension(:,:), allocatable, target :: X_k, G_k
         integer(c_int) :: done

         if (use_kernels) then
            allocate(X_k(n_old, n_cols), G_k(n_cols, n_cols))
            call sstep_project_gram_kokkos(vec_handles, int(n_old, c_int), int(n_cols, c_int), &
                     c_loc(X_k), c_loc(G_k), int(kernels - 1, c_int), done)
            if (done /= 0) then
               X(1:n_old, 1:n_cols) = real(X_k, kind=dp)
               G(1:n_cols, 1:n_cols) = real(G_k, kind=dp)
               deallocate(X_k, G_k)
               return
            end if
            deallocate(X_k, G_k)
            use_kernels = .FALSE.
         end if
#endif

         ! All the dot products against the existing basis in one reduction
         n_petscint = n_old
         do col = 1, n_cols
            call VecMDotBegin(V_n(j + 1 + col), n_petscint, V_n, dots, ierr)
         end do
         do col = 1, n_cols
            call VecMDotEnd(V_n(j + 1 + col), n_petscint, V_n, dots, ierr)
            X_work(1:n_old, col) = dots(1:n_old)
         end do
         do col = 1, n_cols
            X(1:n_old, col) = real(X_work(1:n_old, col), kind=dp)
            dots(1:n_old) = -X_work(1:n_old, col)
            call VecMAXPY(V_n(j + 1 + col), n_petscint, dots, V_n, ierr)
         end do

         ! The upper triangle of the Gram matrix of the block in one reduction
         do col = 1, n_cols
            n_petscint = col
            call VecMDotBegin(V_n(j + 1 + col), n_petscint, V_n(j + 2:j + 1 + col), dots, ierr)
         end do
         do col = 1, n_cols
            n_petscint = col
            call VecMDotEnd(V_n(j + 1 + col), n_petscint, V_n(j + 2:j + 1 + col), dots, ierr)
            G(1:col, col) = real(dots(1:col), kind=dp)
            G(col, 1:col) = G(1:col, col)
         end do

      end subroutine project_and_gram

      subroutine apply_inverse_factor(n_cols, R)

         ! Overwrites the first n_cols columns of the block with the block times R^-1
         ! for upper triangular R, one column at a time

         integer, intent(in) :: n_cols
         real(kind=dp), dimension(:,:), intent(in) :: R

         integer :: col
#if defined(PETSC_HAVE_KOKKOS)
         PetscReal, dimension(:,:), allocatable, target :: R_inv
         integer(c_int) :: done
         integer :: row

         if (use_kernels) then
            ! The inverse of the upper triangular factor
            allocate(R_inv(n_cols, n_cols))
            R_inv = 0
            do col = 1, n_cols
               R_inv(col, col) = real(1d0/R(col, col), kind=kind(R_inv))
               do row = 1, col - 1
                  R_inv(row, col) = real(-dot_product(real(R_inv(row, row:col - 1), kind=dp), R(row:col - 1, col)) &
                           /R(col, col), kind=kind(R_inv))
               end do
            end do
            call sstep_apply_inverse_kokkos(vec_handles, int(n_old, c_int), int(n_cols, c_int), &
                     c_loc(R_inv), done)
            deallocate(R_inv)
            if (done /= 0) return
            use_kernels = .FALSE.
         end if
#endif

         do col = 1, n_cols
            if (col > 1) then
               n_petscint = col - 1
               dots(1:col - 1) = real(-R(1:col - 1, col), kind=kind(dots))
               call VecMAXPY(V_n(j + 1 + col), n_petscint, dots, V_n(j + 2:j + col), ierr)
            end if
            alpha_r = real(1d0/R(col, col), kind=kind(alpha_r))
            call VecScale(V_n(j + 1 + col), alpha_r, ierr)
         end do

      end subroutine apply_inverse_factor

      subroutine recover_hessenberg(n_cols)

         ! The columns of H for the first n_cols columns of the block, from the recurrence B
         ! of the block and the factors C and R_new of its vectors in the new basis

         integer, intent(in) :: n_cols

         integer :: col, col_two
         real(kind=dp), dimension(:,:), allocatable :: R_full, rhs

         ! W(:, 0:n_cols) = V_n(1:j+n_cols+1) R_full
         allocate(R_full(j + n_cols + 1, n_cols + 1), rhs(j + n_cols + 1, n_cols))
         R_full = 0
         R_full(j + 1, 1) = 1
         R_full(1:n_old, 2:n_cols + 1) = C(:, 1:n_cols)
         R_full(n_old + 1:n_old + n_cols, 2:n_cols + 1) = R_new(1:n_cols, 1:n_cols)

         ! A V_n(1:j+n_cols) R_full(1:j+n_cols, 1:n_cols) = V_n(1:j+n_cols+1) R_full B
         ! and we know A V_n(1:j) already, so take that part across
         rhs = matmul(R_full, B(1:n_cols + 1, 1:n_cols))
         if (j > 0) then
            rhs(1:j + 1, :) = rhs(1:j + 1, :) - &
                     matmul(real(H_n(1:j + 1, 1:j), kind=dp), R_full(1:j, 1:n_cols))
         end if

         ! Solve against the upper triangular R_full(j+1:j+n_cols, 1:n_cols) from the right
         do col = 1, n_cols
            do col_two = 1, col - 1
               rhs(:, col) = rhs(:, col) - rhs(:, col_two) * R_full(j + col_two, col)
            end do
            rhs(:, col) = rhs(:, col)/R_full(j + col, col)
         end do
         H_n(1:j + n_cols + 1, j + 1:j + n_cols) = real(rhs, kind=kind(H_n))
         deallocate(R_full, rhs)

      end subroutine recover_hessenberg

      subroutine check_block_orthogonal(n_cols)

         ! Debug check of the device kernels with PETSc vector operations: the first n_cols
         ! columns of the block must be orthonormal and orthogonal to the existing basis

         integer, intent(in) :: n_cols

         integer :: col
         PetscReal :: tol

         tol = sqrt(PFLARE_EPS)
         do col = 1, n_cols
            n_petscint = n_old + col
            call VecMDot(V_n(j + 1 + col), n_petscint, V_n, dots, ierr)
            dots(n_old + col) = dots(n_old + col) - 1
            if (maxval(abs(dots(1:n_old + col))) > tol) then
               print *, "Kokkos s-step Arnoldi block is not orthonormal", maxval(abs(dots(1:n_old + col))), "column", col
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
            end if
         end do

      end subroutine check_block_orthogonal

      subroutine view_counts()

         integer :: comm_rank
         MPIU_Comm :: MPI_COMM_MATRIX

         call PetscObjectGetComm(matrix, MPI_COMM_MATRIX, ierr)
         call MPI_Comm_rank(MPI_COMM_MATRIX, comm_rank, errorcode)
         call PetscTime(time_end, ierr)
         if (comm_rank == 0) then
            if (use_kernels) print '(A,I0)', "s-step Arnoldi: device block kernels ", kernels
            print '(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A,ES10.3)', "s-step Arnoldi: steps ", m, &
                  " single steps ", n_single, &
                  " blocks ", n_blocks, " final block size ", s, " wasted matvecs ", n_wasted, &
                  " reductions ", n_reductions, " time ", time_end - time_start
            if (n_blocks > 0) print '(A,I0,A,I0)', "s-step Arnoldi: estimated first block, Ritz points ", &
                  s_est_ritz, " harmonic Ritz points ", s_est_harmonic
            if (n_blocks > 0) print '(A,*(1X,I0))', "s-step Arnoldi: columns kept per block", kept_history(1:n_blocks)
            if (n_blocks > 0) print '(A,ES10.3)', "s-step Arnoldi: host time in the Cholesky and Hessenberg recovery ", time_host
            if (n_blocks > 0) print '(A,I0,A,ES9.2,A,I0,A,I0,A,I0,A,ES9.2,A,ES9.2)', &
                  "s-step Arnoldi: blocks cut to keep H accurate ", n_cut, &
                  " largest ratio kept ", worst_ratio, " times the bound was tightened ", n_tightened, &
                  " stretches of ordinary steps ", n_stretches, &
                  " extra matvecs ", n_extra_matvecs, " final ratio bound ", ratio_max, &
                  " worst relation error seen ", worst_relation
            if (n_blocks > 0 .AND. n_blocks <= 12) print '(A,*(1X,ES9.2))', "s-step Arnoldi: time per block", &
                  time_history(1:n_blocks)
         end if

      end subroutine view_counts

   end subroutine arnoldi_sstep

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine arnoldi_sstep_low_sync(matrix, poly_order, s_max, m_start, lucky_tol, V_n, w_j, beta, H_n, m, view, kernels, &
                  floor_stop, mode, adaptive)

      ! The s-step Arnoldi of arnoldi_sstep with fewer reductions in each block, after
      ! Carson & Ma, "A stable one-synchronization variant of reorthogonalized block
      ! classical Gram-Schmidt" (2025). arnoldi_sstep uses five reductions a block: project and
      ! Gram matrix for each of the two passes, and one for the check of the Hessenberg matrix
      !
      ! Three things remove reductions here:
      ! 1) A pass can get the components along the existing basis and the Gram matrix of the block
      !    in one reduction, [Q W]^T W, with the Gram matrix of the projected block from
      !    Pythagoras as W^T W - (Q^T W)^T (Q^T W). That is always done for the second pass, where
      !    the block is already nearly orthonormal. For the first pass it loses accuracy when the
      !    block vectors lie mostly in the existing basis, so the partial Cholesky then also
      !    drops a column once the norm of the unprojected vectors over the smallest singular
      !    value of the factor passes the condition bound
      ! 2) The second pass of a block can share its reduction with the first projection of the
      !    next block (delayed). The next block then has to be built before this one is final,
      !    from its last vector after the first pass, and its components along the final vectors
      !    come from the same reduction without touching the vectors again
      ! 3) The check of the last column of the Hessenberg matrix is made after the first pass, and
      !    the norm it needs rides in the reduction of the second pass
      !
      ! mode is
      ! 1 - Pythagorean first pass, delayed: one reduction a block (BCGSI+P-1S)
      ! 2 - Pythagorean first pass, not delayed: two reductions (BCGS-PIPI+)
      ! 3 - explicit Gram matrix in the first pass, not delayed: three reductions
      ! 4 - explicit Gram matrix in the first pass, delayed: two reductions (BCGSI+P-2S)
      ! If adaptive, a block whose Pythagorean Cholesky keeps fewer columns than asked is factored
      ! again from an explicit Gram matrix, for one more reduction
      !
      ! Everything else is as in arnoldi_sstep: the block size never grows, a block that fails
      ! the check is halved, and a block that keeps nothing hands over to ordinary two-pass steps
      ! With the delayed modes a block that is cut after the next one was built wastes the
      ! matvecs of that next block

      ! ~~~~~~
      type(tMat), intent(in)                    :: matrix
      integer, intent(in)                       :: poly_order, s_max, m_start
      PetscReal, intent(in)                     :: lucky_tol
      type(tVec), dimension(:), intent(in)      :: V_n
      type(tVec), intent(in)                    :: w_j
      PetscReal, intent(out)                    :: beta
      PetscReal, dimension(:,:), intent(inout)  :: H_n
      integer, intent(out)                      :: m
      logical, intent(in)                       :: view, floor_stop, adaptive
      integer, intent(in)                       :: kernels, mode

      ! Local variables
      integer, parameter :: dp = kind(1d0)
      logical :: use_kernels, pyth_first, delayed, pending, have_first, have_next, stop_after, lucky
      integer(c_long_long), dimension(:), allocatable :: vec_handles
      integer(c_long_long) :: w_handle
      integer :: k, m0, j, s, sb, p, p_two, i_loc, col_loc, n_old, errorcode, sb_next, s_new, p_gen
      integer :: n_blocks, n_wasted, n_reductions, n_single, p_block, n_cut, n_left, n_right
      integer :: n_tightened, n_extra_matvecs, n_resume, n_resume_start, m_target, n_stretches
      integer :: n_tightened_before, n_late_waste, n_redo
      real(kind=dp) :: worst_ratio, ratio_max, worst_relation, ratio_before
      real(kind=dp), dimension(:), allocatable :: column_ratio, c_start, c_next, t_diag
      PetscReal :: relation_error, relation_tol, w_norm_sq, ratio_option
      PetscReal, dimension(poly_order + 1) :: rho_block
      PetscBool :: option_found, check_blocks
      integer, dimension(:), allocatable :: kept_history
      PetscLogDouble :: time_start, time_end
      PetscErrorCode :: ierr
      PetscInt :: n_petscint
      PetscReal, dimension(:), allocatable :: ritz_real, ritz_imag, shift_real, shift_imag, work, dots
      PetscReal, dimension(:,:), allocatable :: H_copy, VL, VR
      integer, dimension(:), allocatable :: indices
      real(kind=dp), dimension(:,:), allocatable :: B, B_next, X_one, X_next, T_one, T_next, G, Z, Z_two
      real(kind=dp), dimension(:,:), allocatable :: Y_two, D, C, R_new, Z_cross, P_cross
      real(kind=dp) :: mean, ritz_max, omega
      PetscReal :: alpha_r, beta_r, gamma_r
      PetscBLASInt :: m_bl, lda_bl, one_bl, lwork, info

      ! ~~~~~~

      pyth_first = mode == 1 .OR. mode == 2
      delayed = mode == 1 .OR. mode == 4

      ! The number of Arnoldi steps we want
      k = poly_order + 1
      m0 = max(1, min(m_start, k))
      s = max(1, s_max)
      allocate(kept_history(k))
      call PetscTime(time_start, ierr)
      use_kernels = kernels /= 0
#if !defined(PETSC_HAVE_KOKKOS)
      use_kernels = .FALSE.
#endif
      allocate(vec_handles(size(V_n)))
      do i_loc = 1, size(V_n)
         vec_handles(i_loc) = V_n(i_loc)%v
      end do
      w_handle = w_j%v

      ! ~~~~~~~~~
      ! Ordinary Arnoldi for the first m0 steps
      ! ~~~~~~~~~
      call arnoldi(matrix, m0 - 1, lucky_tol, V_n, w_j, beta, H_n, m)
      n_reductions = 1 + 2 * m
      n_single = m
      n_blocks = 0
      n_wasted = 0
      n_late_waste = 0
      n_redo = 0
      n_cut = 0
      worst_ratio = 0
      n_tightened = 0
      n_stretches = 0
      lucky = .FALSE.
      n_resume_start = 10
      n_resume = n_resume_start
      n_extra_matvecs = 0
      worst_relation = 0
      ! The same tolerance, bound and temporary options as arnoldi_sstep
      relation_tol = 1d-1 * sqrt(PFLARE_EPS)
      check_blocks = PETSC_TRUE
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, &
               '-pflare_newton_sstep_check', check_blocks, option_found, ierr)
      ratio_option = -1
      call PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, &
               '-pflare_newton_sstep_ratio', ratio_option, option_found, ierr)
      ratio_max = 1d-3/sqrt(real(PFLARE_EPS, kind=dp))
      if (ratio_option > 0) ratio_max = real(ratio_option, kind=dp)
      ! Have we finished or hit the lucky tolerance
      if (m == k .OR. H_n(m+1, m) < lucky_tol) then
         if (view) call view_counts()
         return
      end if

      ! ~~~~~~~~~
      ! The Ritz values of the first m0 steps are the shifts
      ! ~~~~~~~~~
      allocate(H_copy(m0, m0), ritz_real(m0), ritz_imag(m0), VL(0, 0), VR(0, 0))
      H_copy = H_n(1:m0, 1:m0)
      m_bl = int(m0, kind=kind(m_bl))
      lda_bl = m_bl
      one_bl = 1
      lwork = int(10 * m0, kind=kind(lwork))
      allocate(work(lwork))
      call PFLAREgeev('N', 'N', m_bl, H_copy, lda_bl, &
                     ritz_real, ritz_imag, VL, one_bl, VR, one_bl, &
                     work, lwork, info)
      deallocate(work, VL, VR, H_copy)
      if (info /= 0) then
         print *, "Eig decomposition failed"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! Leja order, which keeps the conjugate pairs next to each other
      allocate(shift_real(m0), shift_imag(m0))
      shift_real = ritz_real
      shift_imag = ritz_imag
      call modified_leja(shift_real, shift_imag, indices)
      shift_real = ritz_real(indices)
      shift_imag = ritz_imag(indices)
      deallocate(indices)

      ! The scaling of each basis vector is the distance of its shift from the centre
      mean = sum(real(ritz_real, kind=dp))/m0
      ritz_max = maxval(sqrt(real(ritz_real, kind=dp)**2 + real(ritz_imag, kind=dp)**2))
      deallocate(ritz_real, ritz_imag)

      ! The block is kept if its condition number is below this
      omega = 0.1d0/sqrt(real(PFLARE_EPS, kind=dp))

      allocate(dots(k + 1), c_start(k + 1), c_next(k + 1), t_diag(s), column_ratio(s))
      allocate(B(s + 1, s), B_next(s + 1, s), X_one(k + 1, s), X_next(k + 1, s), T_one(s, s), T_next(s, s))
      allocate(G(s, s), Z(s, s), Z_two(s, s), Y_two(k + 1, s), D(k + 1, 2 * s), Z_cross(k + 1, s), P_cross(s, s))

      ! j is the number of Arnoldi steps finished, so V_n(1:j+1) is the basis so far
      ! If pending, the next block is already in V_n(j+2:j+1+sb_next), built from V_n(j+1) before
      ! that had its second pass. c_next holds that start vector in the basis, X_next the
      ! components of the block along the basis and T_next its Gram matrix
      j = m
      pending = .FALSE.
      sb_next = 0
      do while (j < k)

         ! Blocks this small cost as many reductions as ordinary steps
         if (s < 3) exit

         n_old = j + 1
         n_blocks = n_blocks + 1
         if (pending) then
            sb = sb_next
            B = B_next
            c_start(1:n_old) = c_next(1:n_old)
            X_one(1:n_old, 1:sb) = X_next(1:n_old, 1:sb)
            T_one(1:sb, 1:sb) = T_next(1:sb, 1:sb)
            have_first = .TRUE.
            pending = .FALSE.
         else
            sb = min(s, k - j)
            call build_block(j, sb, B)
            c_start(1:n_old) = 0
            c_start(n_old) = 1
            have_first = .FALSE.
         end if

         ! ~~~~~~~~~
         ! First pass: the components along the existing basis (and with them the Gram matrix
         ! of the block, for the Pythagorean form), unless they came with the last reduction
         ! ~~~~~~~~~
         if (.NOT. have_first) then
            n_left = n_old
            if (pyth_first) n_left = n_old + sb
            call fused_dots(1, n_left, j + 2, sb, .FALSE.)
            n_reductions = n_reductions + 1
            X_one(1:n_old, 1:sb) = D(1:n_old, 1:sb)
            if (pyth_first) T_one(1:sb, 1:sb) = D(n_old + 1:n_old + sb, 1:sb)
         end if
         call project_block(sb, X_one)

         if (pyth_first) then
            G(1:sb, 1:sb) = T_one(1:sb, 1:sb) - matmul(transpose(X_one(1:n_old, 1:sb)), X_one(1:n_old, 1:sb))
            do col_loc = 1, sb
               t_diag(col_loc) = T_one(col_loc, col_loc)
            end do
            call partial_cholesky(G, sb, omega, p, Z, t_diag)
            if (adaptive .AND. p < sb) then
               call fused_dots(j + 2, sb, j + 2, sb, .FALSE.)
               n_reductions = n_reductions + 1
               n_redo = n_redo + 1
               G(1:sb, 1:sb) = D(1:sb, 1:sb)
               call partial_cholesky(G, sb, omega, p, Z)
            end if
         else
            call fused_dots(j + 2, sb, j + 2, sb, .FALSE.)
            n_reductions = n_reductions + 1
            G(1:sb, 1:sb) = D(1:sb, 1:sb)
            call partial_cholesky(G, sb, omega, p, Z)
         end if
         if (p == 0) then
            n_wasted = n_wasted + sb
            kept_history(n_blocks) = 0
            exit
         end if
         call apply_inverse_factor(p, Z)

         ! The ratio of each block vector to its new part, as in arnoldi_sstep
         do col_loc = 1, p
            column_ratio(col_loc) = max(1d0, sqrt(sum(X_one(1:n_old, col_loc)**2) + sum(Z(1:col_loc, col_loc)**2))) &
                     /Z(col_loc, col_loc)
         end do
         p_block = p
         ratio_before = ratio_max
         n_tightened_before = n_tightened
         call cut_by_ratio()

         have_next = .FALSE.
         stop_after = .FALSE.
         p_gen = 0
         if (p > 0) then

            ! The Hessenberg columns from the first pass alone, for the check and the floor
            call recover_hessenberg(p, X_one, Z)
            if (floor_stop) then
               call hessenberg_gmres_residuals(H_n, j + p, rho_block)
               stop_after = minval(rho_block(1:j + p)) <= 10 * PFLARE_EPS
            end if

            ! ~~~~~~~~~
            ! Delayed: build the next block now, from the last vector we expect to keep
            ! ~~~~~~~~~
            if (delayed .AND. j + p < k .AND. .NOT. stop_after) then
               s_new = s
               if (p < sb) s_new = p
               if (s_new >= 3) then
                  sb_next = min(s_new, k - j - p)
                  call build_block(j + p, sb_next, B_next)
                  have_next = .TRUE.
                  p_gen = p
               end if
            end if

            ! The check of the last column, whose norm rides in the next reduction
            if (check_blocks) call relation_vector(p)

            ! ~~~~~~~~~
            ! Second pass, in one reduction: the block against the existing basis and itself
            ! and for the delayed modes the next block against all of those as well
            ! ~~~~~~~~~
            n_right = p
            if (have_next) n_right = p + sb_next
            n_left = n_old + p
            if (have_next .AND. pyth_first) n_left = n_old + p + sb_next
            call fused_dots(1, n_left, j + 2, n_right, logical(check_blocks .eqv. PETSC_TRUE))
            n_reductions = n_reductions + 1
            Y_two(1:n_old, 1:p) = D(1:n_old, 1:p)
            T_one(1:p, 1:p) = D(n_old + 1:n_old + p, 1:p)
            if (have_next) then
               Z_cross(1:n_old, 1:sb_next) = D(1:n_old, p + 1:p + sb_next)
               P_cross(1:p, 1:sb_next) = D(n_old + 1:n_old + p, p + 1:p + sb_next)
               if (pyth_first) T_next(1:sb_next, 1:sb_next) = D(n_old + p + 1:n_old + p + sb_next, p + 1:p + sb_next)
            end if

            ! If the check fails keep half as many columns and check again, which then
            ! costs a reduction of its own
            if (check_blocks) then
               relation_error = sqrt(w_norm_sq)/norm2(H_n(1:j + p + 1, j + p))
               do
                  if (relation_error <= relation_tol) then
                     worst_relation = max(worst_relation, real(relation_error, kind=dp))
                     exit
                  end if
                  n_tightened = n_tightened + 1
                  if (p == 1) then
                     p = 0
                     exit
                  end if
                  p = (p + 1)/2
                  ratio_max = min(ratio_max, maxval(column_ratio(1:p)))
                  call cut_by_ratio()
                  if (p == 0) exit
                  call recover_hessenberg(p, X_one, Z)
                  call relation_vector(p)
                  call VecNorm(w_j, NORM_2, relation_error, ierr)
                  relation_error = relation_error/norm2(H_n(1:j + p + 1, j + p))
                  n_reductions = n_reductions + 1
               end do
            end if

            ! If the check cut this block to fewer than three columns the ratio is not what is
            ! wrong either, so treat it in the same way as keeping none
            if (p > 0 .AND. p < 3 .AND. p_block >= 3 .AND. n_tightened > n_tightened_before) p = 0
         end if

         if (p > 0) then
            G(1:p, 1:p) = T_one(1:p, 1:p) - matmul(transpose(Y_two(1:n_old, 1:p)), Y_two(1:n_old, 1:p))
            call partial_cholesky(G, p, omega, p_two, Z_two)
            if (p_two == 0) then
               n_wasted = n_wasted + sb
               if (have_next) n_late_waste = n_late_waste + sb_next
               kept_history(n_blocks) = 0
               H_n(:, j + 1:j + p_block) = 0
               exit
            end if
            p = p_two
            call project_block(p, Y_two)
            call apply_inverse_factor(p, Z_two)
            worst_ratio = max(worst_ratio, maxval(column_ratio(1:p)))

            ! The block before orthogonalising is W(:, 1:p) = V_n(1:j+1) C + V_n(j+2:j+p+1) R_new
            allocate(C(n_old, p), R_new(p, p))
            C = X_one(1:n_old, 1:p) + matmul(Y_two(1:n_old, 1:p), Z(1:p, 1:p))
            R_new = matmul(Z_two(1:p, 1:p), Z(1:p, 1:p))
            call recover_hessenberg(p, C, R_new)
            deallocate(C, R_new)
         end if

         n_wasted = n_wasted + sb - p
         kept_history(n_blocks) = p
         if (p == 0) then

            ! Not even one column of this block could be kept accurately: a stretch of ordinary
            ! two-pass steps and then blocks again, as in arnoldi_sstep
            if (have_next) n_late_waste = n_late_waste + sb_next
            ratio_max = ratio_before
            H_n(:, j + 1:j + p_block) = 0
            m_target = min(k, j + n_resume)
            call arnoldi_extend(matrix, lucky_tol, V_n, w_j, H_n, j, m_target, m, 2, floor_stop)
            n_reductions = n_reductions + 3 * (m - j)
            n_single = n_single + m - j
            n_stretches = n_stretches + 1
            n_resume = 2 * n_resume
            if (m < m_target .OR. H_n(m + 1, m) < lucky_tol) then
               j = m
               lucky = .TRUE.
               exit
            end if
            j = m
            cycle
         end if
         n_resume = n_resume_start

         if (use_kernels .AND. kokkos_debug()) call check_block_orthogonal(p)

         ! The next block was built from a vector that has now had its second pass. Its
         ! components along the final vectors follow from the reduction we already did
         if (have_next) then
            if (p == p_gen) then
               c_next(1:n_old) = Y_two(1:n_old, p)
               c_next(n_old + 1:n_old + p) = Z_two(1:p, p)
               X_next(1:n_old, 1:sb_next) = Z_cross(1:n_old, 1:sb_next)
               P_cross(1:p, 1:sb_next) = P_cross(1:p, 1:sb_next) - &
                        matmul(transpose(Y_two(1:n_old, 1:p)), Z_cross(1:n_old, 1:sb_next))
               ! Solve against the lower triangular transpose of Z_two
               do i_loc = 1, p
                  do col_loc = 1, i_loc - 1
                     P_cross(i_loc, 1:sb_next) = P_cross(i_loc, 1:sb_next) - &
                              Z_two(col_loc, i_loc) * P_cross(col_loc, 1:sb_next)
                  end do
                  P_cross(i_loc, 1:sb_next) = P_cross(i_loc, 1:sb_next)/Z_two(i_loc, i_loc)
               end do
               X_next(n_old + 1:n_old + p, 1:sb_next) = P_cross(1:p, 1:sb_next)
               pending = .TRUE.
            else
               n_late_waste = n_late_waste + sb_next
            end if
         end if

         j = j + p
         ! The block size never grows back
         if (p < sb) s = p

         ! Stop once the relative GMRES residual is at rounding level
         if (floor_stop) then
            call hessenberg_gmres_residuals(H_n, j, rho_block)
            if (minval(rho_block(1:j)) <= 10 * PFLARE_EPS) then
               if (pending) n_late_waste = n_late_waste + sb_next
               lucky = .TRUE.
               exit
            end if
         end if

      end do

      deallocate(shift_real, shift_imag, dots, c_start, c_next, t_diag, column_ratio)
      deallocate(B, B_next, X_one, X_next, T_one, T_next, G, Z, Z_two, Y_two, D, Z_cross, P_cross)

      ! Any steps left are an ordinary Arnoldi, with two passes
      m = j
      if (j < k .AND. .NOT. lucky) then
         call arnoldi_extend(matrix, lucky_tol, V_n, w_j, H_n, j, k, m, 2, floor_stop)
         n_reductions = n_reductions + 3 * (m - j)
         n_single = n_single + m - j
      end if

      if (view) call view_counts()

      contains

      subroutine build_block(j_at, n_cols, B_out)

         ! Builds a block of n_cols vectors with the scaled Newton recurrence from V_n(j_at+1)
         ! into V_n(j_at+2:j_at+1+n_cols). B_out holds the recurrence: A W(:, 0:n-1) = W(:, 0:n) B

         integer, intent(in) :: j_at, n_cols
         real(kind=dp), dimension(:,:), intent(inout) :: B_out

         integer :: i_b
         real(kind=dp) :: x, y, y_sq, gam

         B_out = 0
         i_b = 0
         do while (i_b < n_cols)

            ! We cycle through the shifts if there are fewer than the block size
            x = real(shift_real(mod(i_b, m0) + 1), kind=dp)
            y = real(shift_imag(mod(i_b, m0) + 1), kind=dp)
            gam = max(sqrt((mean - x)**2 + y**2), 1d-3 * ritz_max)

            ! w_{i+1} = (A - x) w_i / gam
            call MatMult(matrix, V_n(j_at + 1 + i_b), V_n(j_at + 2 + i_b), ierr)
            alpha_r = real(-x/gam, kind=kind(alpha_r))
            beta_r = real(1d0/gam, kind=kind(beta_r))
            call VecAXPBY(V_n(j_at + 2 + i_b), alpha_r, beta_r, V_n(j_at + 1 + i_b), ierr)
            B_out(i_b + 1, i_b + 1) = x
            B_out(i_b + 2, i_b + 1) = gam

            ! A real shift, or a conjugate pair cut in half by the end of the block
            if (y == 0d0 .OR. i_b + 2 > n_cols) then
               i_b = i_b + 1
               cycle
            end if

            ! The second member of a conjugate pair in real arithmetic
            y_sq = y**2
            call MatMult(matrix, V_n(j_at + 2 + i_b), V_n(j_at + 3 + i_b), ierr)
            alpha_r = real(-x/gam, kind=kind(alpha_r))
            beta_r = real(y_sq/gam**2, kind=kind(beta_r))
            gamma_r = real(1d0/gam, kind=kind(gamma_r))
            call VecAXPBYPCZ(V_n(j_at + 3 + i_b), alpha_r, beta_r, gamma_r, &
                     V_n(j_at + 2 + i_b), V_n(j_at + 1 + i_b), ierr)
            B_out(i_b + 2, i_b + 2) = x
            B_out(i_b + 3, i_b + 2) = gam
            B_out(i_b + 1, i_b + 2) = -y_sq/gam
            i_b = i_b + 2
         end do

      end subroutine build_block

      subroutine cut_by_ratio()

         ! Keeps only the leading columns of the block whose ratio is within the bound

         integer :: col

         do col = 1, p
            if (.NOT. column_ratio(col) <= ratio_max) then
               p = col - 1
               n_cut = n_cut + 1
               exit
            end if
         end do

      end subroutine cut_by_ratio

      subroutine fused_dots(l_start, n_l, r_start, n_r, with_w)

         ! D(1:n_l, 1:n_r) = V_n(l_start:l_start+n_l-1)^T V_n(r_start:r_start+n_r-1) in one
         ! reduction, which also returns the squared norm of w_j in w_norm_sq if with_w

         integer, intent(in) :: l_start, n_l, r_start, n_r
         logical, intent(in) :: with_w

         integer :: col
         PetscReal :: w_dot
#if defined(PETSC_HAVE_KOKKOS)
         PetscReal, dimension(n_l, n_r), target :: D_k
         integer(c_int) :: done, with_w_int
         PetscReal :: w_norm_sq_k

         D_k = 0
         w_norm_sq_k = 0
         if (use_kernels) then
            with_w_int = 0
            if (with_w) with_w_int = 1
            call sstep_fused_dots_kokkos(vec_handles, int(l_start - 1, c_int), int(n_l, c_int), &
                     int(r_start - 1, c_int), int(n_r, c_int), w_handle, with_w_int, &
                     c_loc(D_k), w_norm_sq_k, int(kernels - 1, c_int), done)
            if (done == 0) use_kernels = .FALSE.
         end if
#endif

         ! With PETSc vector operations, which is also the debug check of the kernel
         if (.NOT. use_kernels .OR. kokkos_debug()) then
            n_petscint = n_l
            do col = 1, n_r
               call VecMDotBegin(V_n(r_start + col - 1), n_petscint, V_n(l_start:l_start + n_l - 1), dots, ierr)
            end do
            if (with_w) call VecDotBegin(w_j, w_j, w_dot, ierr)
            do col = 1, n_r
               call VecMDotEnd(V_n(r_start + col - 1), n_petscint, V_n(l_start:l_start + n_l - 1), dots, ierr)
               D(1:n_l, col) = real(dots(1:n_l), kind=dp)
            end do
            w_norm_sq = 0
            if (with_w) then
               call VecDotEnd(w_j, w_j, w_dot, ierr)
               w_norm_sq = w_dot
            end if
         end if

#if defined(PETSC_HAVE_KOKKOS)
         if (use_kernels) then
            if (kokkos_debug()) then
               if (maxval(abs(D(1:n_l, 1:n_r) - real(D_k, kind=dp))) > &
                        sqrt(PFLARE_EPS) * max(1d0, maxval(abs(D(1:n_l, 1:n_r)))) .OR. &
                        (with_w .AND. abs(w_norm_sq - w_norm_sq_k) > sqrt(PFLARE_EPS) * max(PFLARE_ONE, w_norm_sq))) then
                  print *, "Kokkos s-step fused dot products do not match", &
                        maxval(abs(D(1:n_l, 1:n_r) - real(D_k, kind=dp)))
                  call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
               end if
            end if
            D(1:n_l, 1:n_r) = real(D_k, kind=dp)
            w_norm_sq = w_norm_sq_k
         end if
#endif

      end subroutine fused_dots

      subroutine project_block(n_cols, X)

         ! Removes the components X along the existing basis from the first n_cols columns
         ! of the block, with no reduction

         integer, intent(in) :: n_cols
         real(kind=dp), dimension(:,:), intent(in) :: X

         integer :: col
#if defined(PETSC_HAVE_KOKKOS)
         PetscReal, dimension(:,:), allocatable, target :: X_k
         integer(c_int) :: done

         if (use_kernels) then
            allocate(X_k(n_old, n_cols))
            X_k = real(X(1:n_old, 1:n_cols), kind=kind(X_k))
            call sstep_project_kokkos(vec_handles, int(n_old, c_int), int(n_cols, c_int), &
                     c_loc(X_k), int(kernels - 1, c_int), done)
            deallocate(X_k)
            if (done /= 0) return
            use_kernels = .FALSE.
         end if
#endif

         n_petscint = n_old
         do col = 1, n_cols
            dots(1:n_old) = real(-X(1:n_old, col), kind=kind(dots))
            call VecMAXPY(V_n(j + 1 + col), n_petscint, dots, V_n, ierr)
         end do

      end subroutine project_block

      subroutine apply_inverse_factor(n_cols, R)

         ! Overwrites the first n_cols columns of the block with the block times R^-1
         ! for upper triangular R

         integer, intent(in) :: n_cols
         real(kind=dp), dimension(:,:), intent(in) :: R

         integer :: col
#if defined(PETSC_HAVE_KOKKOS)
         PetscReal, dimension(:,:), allocatable, target :: R_inv
         integer(c_int) :: done
         integer :: row

         if (use_kernels) then
            allocate(R_inv(n_cols, n_cols))
            R_inv = 0
            do col = 1, n_cols
               R_inv(col, col) = real(1d0/R(col, col), kind=kind(R_inv))
               do row = 1, col - 1
                  R_inv(row, col) = real(-dot_product(real(R_inv(row, row:col - 1), kind=dp), R(row:col - 1, col)) &
                           /R(col, col), kind=kind(R_inv))
               end do
            end do
            call sstep_apply_inverse_kokkos(vec_handles, int(n_old, c_int), int(n_cols, c_int), &
                     c_loc(R_inv), done)
            deallocate(R_inv)
            if (done /= 0) return
            use_kernels = .FALSE.
         end if
#endif

         do col = 1, n_cols
            if (col > 1) then
               n_petscint = col - 1
               dots(1:col - 1) = real(-R(1:col - 1, col), kind=kind(dots))
               call VecMAXPY(V_n(j + 1 + col), n_petscint, dots, V_n(j + 2:j + col), ierr)
            end if
            alpha_r = real(1d0/R(col, col), kind=kind(alpha_r))
            call VecScale(V_n(j + 1 + col), alpha_r, ierr)
         end do

      end subroutine apply_inverse_factor

      subroutine recover_hessenberg(n_cols, C_in, R_in)

         ! The columns of H for the first n_cols columns of the block, from the recurrence B
         ! of the block and the factors of its vectors in the basis: the start vector of the
         ! block is V_n(1:j+1) c_start and W(:, 1:n) = V_n(1:j+1) C_in + V_n(j+2:j+n+1) R_in

         integer, intent(in) :: n_cols
         real(kind=dp), dimension(:,:), intent(in) :: C_in, R_in

         integer :: col, col_two
         real(kind=dp), dimension(:,:), allocatable :: R_full, rhs

         allocate(R_full(j + n_cols + 1, n_cols + 1), rhs(j + n_cols + 1, n_cols))
         R_full = 0
         R_full(1:n_old, 1) = c_start(1:n_old)
         R_full(1:n_old, 2:n_cols + 1) = C_in(1:n_old, 1:n_cols)
         R_full(n_old + 1:n_old + n_cols, 2:n_cols + 1) = R_in(1:n_cols, 1:n_cols)

         ! A V_n(1:j+n_cols) R_full(1:j+n_cols, 1:n_cols) = V_n(1:j+n_cols+1) R_full B
         ! and we know A V_n(1:j) already, so take that part across
         rhs = matmul(R_full, B(1:n_cols + 1, 1:n_cols))
         if (j > 0) then
            rhs(1:j + 1, :) = rhs(1:j + 1, :) - &
                     matmul(real(H_n(1:j + 1, 1:j), kind=dp), R_full(1:j, 1:n_cols))
         end if

         ! Solve against the upper triangular R_full(j+1:j+n_cols, 1:n_cols) from the right
         do col = 1, n_cols
            do col_two = 1, col - 1
               rhs(:, col) = rhs(:, col) - rhs(:, col_two) * R_full(j + col_two, col)
            end do
            rhs(:, col) = rhs(:, col)/R_full(j + col, col)
         end do
         H_n(:, j + 1:j + p_block) = 0
         H_n(1:j + n_cols + 1, j + 1:j + n_cols) = real(rhs, kind=kind(H_n))
         deallocate(R_full, rhs)

      end subroutine recover_hessenberg

      subroutine relation_vector(n_cols)

         ! w_j = A v - V H(:, j+n_cols) for the last of the n_cols columns of the block, with
         ! the Hessenberg column we have now. Its norm over that of the column is the check

         integer, intent(in) :: n_cols

         n_petscint = j + n_cols + 1
         call MatMult(matrix, V_n(j + n_cols), w_j, ierr)
         dots(1:j + n_cols + 1) = -H_n(1:j + n_cols + 1, j + n_cols)
         call VecMAXPY(w_j, n_petscint, dots, V_n, ierr)
         n_extra_matvecs = n_extra_matvecs + 1

      end subroutine relation_vector

      subroutine check_block_orthogonal(n_cols)

         ! Debug check of the device kernels with PETSc vector operations: the first n_cols
         ! columns of the block must be orthonormal and orthogonal to the existing basis

         integer, intent(in) :: n_cols

         integer :: col
         PetscReal :: tol

         tol = sqrt(PFLARE_EPS)
         do col = 1, n_cols
            n_petscint = n_old + col
            call VecMDot(V_n(j + 1 + col), n_petscint, V_n, dots, ierr)
            dots(n_old + col) = dots(n_old + col) - 1
            if (maxval(abs(dots(1:n_old + col))) > tol) then
               print *, "Kokkos s-step Arnoldi block is not orthonormal", maxval(abs(dots(1:n_old + col))), "column", col
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
            end if
         end do

      end subroutine check_block_orthogonal

      subroutine view_counts()

         integer :: comm_rank
         MPIU_Comm :: MPI_COMM_MATRIX

         call PetscObjectGetComm(matrix, MPI_COMM_MATRIX, ierr)
         call MPI_Comm_rank(MPI_COMM_MATRIX, comm_rank, errorcode)
         call PetscTime(time_end, ierr)
         if (comm_rank == 0) then
            if (use_kernels) print '(A,I0)', "s-step Arnoldi: device block kernels ", kernels
            print '(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A,ES10.3)', "s-step Arnoldi: steps ", m, &
                  " single steps ", n_single, &
                  " blocks ", n_blocks, " final block size ", s, " wasted matvecs ", n_wasted, &
                  " reductions ", n_reductions, " time ", time_end - time_start
            print '(A,I0,A,L1,A,I0,A,I0)', "s-step Arnoldi: low synchronisation mode ", mode, " adaptive ", adaptive, &
                  " matvecs of next blocks thrown away ", n_late_waste, " blocks factored again ", n_redo
            if (n_blocks > 0) print '(A,*(1X,I0))', "s-step Arnoldi: columns kept per block", kept_history(1:n_blocks)
            if (n_blocks > 0) print '(A,I0,A,ES9.2,A,I0,A,I0,A,I0,A,ES9.2,A,ES9.2)', &
                  "s-step Arnoldi: blocks cut to keep H accurate ", n_cut, &
                  " largest ratio kept ", worst_ratio, " times the bound was tightened ", n_tightened, &
                  " stretches of ordinary steps ", n_stretches, &
                  " extra matvecs ", n_extra_matvecs, " final ratio bound ", ratio_max, &
                  " worst relation error seen ", worst_relation
         end if

      end subroutine view_counts

   end subroutine arnoldi_sstep_low_sync

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine calculate_gmres_polynomial_roots_newton(matrix, poly_order, root_rule, coefficients)

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
      integer, intent(in)                               :: root_rule
      PetscReal, dimension(:, :), pointer, intent(inout)     :: coefficients

      ! Local variables
      PetscInt :: global_rows, global_cols, local_rows, local_cols
      integer :: subspace_size, i_loc, comm_size, comm_rank, errorcode
      integer :: counter, m, numerical_order
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX
      PetscReal, dimension(poly_order+2,poly_order+1) :: H_n
      integer, dimension(:), allocatable :: indices
      PetscReal, dimension(:), allocatable :: real_roots_added, imag_roots_added
      PetscReal :: beta
      PetscReal, dimension(:, :), allocatable :: coefficients_temp
      type(tVec) :: w_j
      type(tVec), pointer, dimension(:) :: V_n
      PetscInt :: sstep_size, sstep_start
      PetscLogDouble :: time_start, time_end
      PetscBool :: sstep_view, sstep_warm, sstep_check, option_found, floor_option
      logical :: floor_stop
      PetscInt :: sstep_kernels, arnoldi_passes, sstep_low_sync
      PetscBool :: sstep_low_sync_adaptive
      PetscLogDouble :: time_roots

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

      ! If we're reusing, coefficients may still be sized for the extra roots 
      ! added in a previous setup, so make sure we start from p+1 roots
      ! Otherwise the old roots would be included in the clustering below
      if (size(coefficients, 1) /= poly_order + 1) then
         deallocate(coefficients)
         allocate(coefficients(poly_order + 1, 2))
      end if

      ! ~~~~~~~~~~
      ! Allocate space and create random numbers 
      ! The first vec has random numbers in it
      ! ~~~~~~~~~~ 
      call create_temp_space_box_muller(matrix, subspace_size, V_n)
      
      ! Create an extra vector for storage
      ! Made from the matrix rather than duplicated from V_n(1), as VecMDot uses w_j's
      ! operations, and only a vector made this way has the Kokkos GEMV VecMDot
      call MatCreateVecs(matrix, w_j, PETSC_NULL_VEC, ierr)
      
      ! Do the Arnoldi and compute H_n
      ! Use the same lucky tolerance as petsc
      ! Temporary options while the s-step Arnoldi is being tested
      ! -pflare_newton_sstep is the block size, with zero the ordinary Arnoldi
      sstep_size = 0
      sstep_start = 10
      sstep_view = PETSC_FALSE
      sstep_kernels = 2
      sstep_warm = PETSC_FALSE
      arnoldi_passes = 1
      ! The adaptive rule only uses the steps up to where the Arnoldi residual reaches rounding
      ! level, so the Arnoldi can stop there (temporary option to switch that off)
      floor_option = PETSC_TRUE
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, &
               '-pflare_newton_floor_stop', floor_option, option_found, ierr)
      floor_stop = root_rule == NEWTON_ROOTS_ADAPTIVE .AND. floor_option
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, &
               '-pflare_newton_arnoldi_passes', arnoldi_passes, option_found, ierr)
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, &
               '-pflare_newton_sstep', sstep_size, option_found, ierr)
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, &
               '-pflare_newton_sstep_start', sstep_start, option_found, ierr)
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, &
               '-pflare_newton_sstep_view', sstep_view, option_found, ierr)
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, &
               '-pflare_newton_sstep_kernels', sstep_kernels, option_found, ierr)
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, &
               '-pflare_newton_sstep_warm', sstep_warm, option_found, ierr)
#if defined(PETSC_HAVE_KOKKOS)
      ! Temporary: time the first use of the device gemm on its own
      if (sstep_warm) then
         call PetscTime(time_start, ierr)
         call sstep_gemm_warmup_kokkos()
         call PetscTime(time_end, ierr)
         if (sstep_view .AND. comm_rank == 0) print '(A,ES10.3)', "s-step Arnoldi: gemm warm up time ", time_end - time_start
      end if
#endif

      ! Temporary option: the s-step Arnoldi with fewer reductions in each block
      ! (see arnoldi_sstep_low_sync for the modes), with zero the five reductions of arnoldi_sstep
      sstep_low_sync = 0
      sstep_low_sync_adaptive = PETSC_FALSE
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, &
               '-pflare_newton_sstep_low_sync', sstep_low_sync, option_found, ierr)
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, &
               '-pflare_newton_sstep_low_sync_adaptive', sstep_low_sync_adaptive, option_found, ierr)

      if (sstep_size > 0 .AND. sstep_low_sync >= 1 .AND. sstep_low_sync <= 4) then
         call arnoldi_sstep_low_sync(matrix, poly_order, int(sstep_size), int(sstep_start), PFLARE_TOL_LUCKY, &
                  V_n, w_j, beta, H_n, m, logical(sstep_view .eqv. PETSC_TRUE), &
                  int(sstep_kernels), floor_stop, int(sstep_low_sync), &
                  logical(sstep_low_sync_adaptive .eqv. PETSC_TRUE))
      else if (sstep_size > 0) then
         call arnoldi_sstep(matrix, poly_order, int(sstep_size), int(sstep_start), PFLARE_TOL_LUCKY, &
                  V_n, w_j, beta, H_n, m, logical(sstep_view .eqv. PETSC_TRUE), &
                  int(sstep_kernels), floor_stop)
      else
         call PetscTime(time_start, ierr)
         if (arnoldi_passes > 1 .OR. floor_stop) then
            ! The same Arnoldi one step at a time, which can stop at the noise floor and
            ! (temporary option) do more than one Gram-Schmidt pass in each step
            call arnoldi(matrix, 0, PFLARE_TOL_LUCKY, V_n, w_j, beta, H_n, m)
            if (poly_order > 0 .AND. .NOT. H_n(2, 1) < PFLARE_TOL_LUCKY) then
               call arnoldi_extend(matrix, PFLARE_TOL_LUCKY, V_n, w_j, H_n, 1, poly_order + 1, m, &
                        int(arnoldi_passes), floor_stop)
            end if
         else
            call arnoldi(matrix, poly_order, PFLARE_TOL_LUCKY, V_n, w_j, beta, H_n, m)
         end if
         call PetscTime(time_end, ierr)
         if (sstep_view .AND. comm_rank == 0) then
            print '(A,I0,A,I0,A,I0,A,ES10.3)', "Arnoldi: steps ", m, " passes ", arnoldi_passes, &
                  " reductions ", 1 + (1 + int(arnoldi_passes)) * m, " time ", time_end - time_start
         end if
      end if

      ! Temporary: check the basis and the Hessenberg matrix we were given
      sstep_check = PETSC_FALSE
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, &
               '-pflare_newton_check_arnoldi', sstep_check, option_found, ierr)
      if (sstep_check) call check_arnoldi_relation()

      call PetscTime(time_roots, ierr)

      if (root_rule == NEWTON_ROOTS_ADAPTIVE) then

         ! ~~~~~~~~~~~~~~
         ! Choose the degree and the extra roots from the Hessenberg matrix
         ! This returns the non-zero roots (with any extras) already leja ordered
         ! ~~~~~~~~~~~~~~
         call adaptive_newton_roots(H_n, m, real_roots_added, imag_roots_added, numerical_order)

         ! Any roots we don't use are zero roots at the end, which are skipped when applying
         ! We always store at least poly_order + 1 roots
         deallocate(coefficients)
         allocate(coefficients(size(real_roots_added) + (poly_order + 1 - numerical_order), 2))
         coefficients = 0d0
         coefficients(1:size(real_roots_added), 1) = real_roots_added
         coefficients(1:size(real_roots_added), 2) = imag_roots_added
         deallocate(real_roots_added, imag_roots_added)

      else

      ! Compute the harmonic Ritz values of H_n, with any numerically zero roots set 
      ! to exactly zero, close roots clustered and the zeros at the end
      call harmonic_ritz_roots(H_n, poly_order + 1, m, coefficients(:, 1), coefficients(:, 2))

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
      if (root_rule == NEWTON_ROOTS_POF .AND. numerical_order > 0) then 

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

         ! Resize coefficients to hold non-zero roots (with extras) + zero roots at end
         deallocate(coefficients)
         allocate(coefficients(size(real_roots_added) + (poly_order + 1 - numerical_order), 2))
         coefficients = 0d0

         ! Leja order only the non-zero eigenvalues (with extras)
         call leja_order_with_extras(coefficients_temp(:, 1), coefficients_temp(:, 2), &
                  real_roots_added, imag_roots_added, indices)

         ! Reorder the (non-perturbed) roots using the leja ordering
         coefficients(1:size(real_roots_added), 1) = real_roots_added(indices)
         coefficients(1:size(real_roots_added), 2) = imag_roots_added(indices)

         ! Zero eigenvalues are already zero at the end from the coefficients = 0d0 above

         ! Cleanup
         deallocate(coefficients_temp, real_roots_added, imag_roots_added, indices)

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

      end if

      call PetscTime(time_end, ierr)
      if (sstep_view .AND. comm_rank == 0) then
         print '(A,ES10.3)', "Newton roots: host time to choose and order the roots ", time_end - time_roots
      end if

      ! Cleanup
      call destroy_temp_space(V_n)
      call VecDestroy(w_j, ierr)       

      contains

      subroutine check_arnoldi_relation()

         ! Temporary diagnostic: for each Arnoldi step j measures how well A v_j = V_n(1:j+1) H_n(1:j+1, j)
         ! holds for the basis and Hessenberg matrix we have, relative to the size of that column of H,
         ! and how far v_{j+1} is from orthonormal against the basis before it
         ! Prints the worst of each, the first step where either passes 1e-6 and the GMRES residual there

         integer :: j_step, first_relation, first_orth, n_steps, n_bad
         PetscReal :: residual, worst_relation, worst_orth, orth
         PetscReal, dimension(poly_order + 2) :: work_h
         PetscReal, dimension(m) :: rho
         PetscInt :: n_vecs

         n_bad = 0
         worst_relation = 0
         worst_orth = 0
         first_relation = 0
         first_orth = 0
         ! The last basis vector does not exist if we hit the lucky tolerance
         n_steps = m
         if (H_n(m + 1, m) < PFLARE_TOL_LUCKY) n_steps = m - 1
         call hessenberg_gmres_residuals(H_n, m, rho)

         do j_step = 1, n_steps
            n_vecs = j_step + 1
            call MatMult(matrix, V_n(j_step), w_j, ierr)
            work_h(1:j_step + 1) = -H_n(1:j_step + 1, j_step)
            call VecMAXPY(w_j, n_vecs, work_h, V_n, ierr)
            call VecNorm(w_j, NORM_2, residual, ierr)
            residual = residual/norm2(H_n(1:j_step + 1, j_step))
            if (residual > worst_relation) worst_relation = residual
            if (first_relation == 0 .AND. residual > 1d-6) first_relation = j_step
            if (residual > 1d-6 .AND. comm_rank == 0 .AND. n_bad < 40) then
               n_bad = n_bad + 1
               print '(A,I0,A,ES9.2,A,ES9.2,A,ES9.2)', "   bad step ", j_step, " residual ", residual, &
                     " subdiagonal ", H_n(j_step + 1, j_step), " below subdiagonal ", &
                     maxval(abs(H_n(min(j_step + 2, m + 1):m + 1, j_step)))
            end if

            call VecMDot(V_n(j_step + 1), n_vecs, V_n, work_h, ierr)
            work_h(j_step + 1) = work_h(j_step + 1) - 1
            orth = maxval(abs(work_h(1:j_step + 1)))
            if (orth > worst_orth) worst_orth = orth
            if (first_orth == 0 .AND. orth > 1d-6) first_orth = j_step
         end do

         if (comm_rank == 0) then
            print '(A,ES9.2,A,I0,A,ES9.2,A,I0)', "Arnoldi check: worst relation residual ", worst_relation, &
                  " first above 1e-6 at step ", first_relation, " worst loss of orthogonality ", worst_orth, &
                  " first above 1e-6 at step ", first_orth
            print '(A,ES9.2,A,I0,A,ES9.2)', "Arnoldi check: smallest GMRES residual ", minval(rho), " at step ", &
                  minloc(rho, 1), " and at the first bad relation step ", rho(max(1, first_relation))
         end if

      end subroutine check_arnoldi_relation

   end subroutine calculate_gmres_polynomial_roots_newton   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine build_newton_fixed_sparsity_start(matrix, poly_order, poly_sparsity_order, coefficients, &
                  cmat, mat_sparsity_match, mat_product_save, status_output, output_first_complex)

      ! Builds cmat with values correct up to the power poly_sparsity_order, along with
      ! the products we need to compute the remaining fixed sparsity terms
      ! This is shared by the cpu and kokkos versions of mat_mult_powers_share_sparsity_newton
      ! and only involves petsc mat operations, so it runs on the device with kokkos mats

      ! ~~~~~~~~~~
      ! Input
      type(tMat), target, intent(in)                     :: matrix
      integer, intent(in)                                :: poly_order, poly_sparsity_order
      PetscReal, dimension(:, :), target, contiguous, intent(inout)    :: coefficients
      type(tMat), intent(inout)                          :: cmat
      type(tMat), intent(inout)                          :: mat_sparsity_match, mat_product_save
      integer, dimension(:, :), intent(inout)            :: status_output
      logical, intent(out)                               :: output_first_complex

      integer :: errorcode
      PetscErrorCode :: ierr
      logical :: reuse_triggered
      PetscReal, dimension(poly_sparsity_order+1,2) :: coeffs_contig

      ! ~~~~~~~~~~

      ! coefficients is (number of roots, 2), so check against the number of roots
      if (poly_sparsity_order .ge. size(coefficients, 1)-1) then
         print *, "Requested sparsity is greater than or equal to the order"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      reuse_triggered = .NOT. PetscObjectIsNull(cmat)

      ! ~~~~~~~~~~
      ! Compute cmat for all powers up to poly_sparsity_order
      ! We have to be more careful here than in the monomial case
      ! In the mononomial case we just compute the matrix powers up to poly_sparsity_order
      ! and add them times the coefficients to cmat
      ! Here though we have to build the Newton basis polynomials
      ! As a rule, the value input here into cmat is correct up to the power
      ! of poly_sparsity_order, with all other terms being added in the fixed sparsity loops
      ! that occur after this routine is called
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
            ! Have to be careful to pass in a contiguous piece of memory here
            coeffs_contig = coefficients(1:poly_sparsity_order + 1, 1:2)
            call build_gmres_polynomial_newton_inverse_1st_1st(matrix, 1, &
                     coeffs_contig, &
                     cmat, mat_sparsity_match, &
                     status_output)
         end if
      else

         ! If we're any higher, then we build cmat up to that order
         ! But we have to be careful because the last root we want to explicitly
         ! build up to here (ie the power of the matrix given by sparsity_order)
         ! might be the first root of a complex conjugate pair
         coeffs_contig = coefficients(1:poly_sparsity_order + 1, 1:2)
         call build_gmres_polynomial_newton_inverse_full(matrix, poly_order, &
                  coeffs_contig, &
                  cmat, mat_sparsity_match, poly_sparsity_order, output_first_complex, &
                  status_output, mat_product_save)
      end if

      ! We know we will never have non-zero locations outside of the highest constrained sparsity power
      call MatSetOption(cmat, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE,  ierr)
      call MatSetOption(cmat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)
      ! We know we are only going to insert local vals
      ! These options should turn off any reductions in the assembly
      call MatSetOption(cmat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)

   end subroutine build_newton_fixed_sparsity_start

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

#if defined(PETSC_HAVE_KOKKOS)
      integer(c_long_long) :: A_array, B_array, reuse_array
      integer(c_long_long) :: sparsity_array, prod_save_array
      integer :: errorcode, reuse_int_reuse_mat, num_terms
      integer :: prod_save_exists_int, output_first_complex_int
      PetscErrorCode :: ierr
      MatType :: mat_type
      Mat :: temp_mat, temp_mat_reuse, temp_mat_compare
      PetscScalar :: normy, max_cpu
      logical :: reuse_triggered_cmat, reuse_triggered_reuse_mat
      type(c_ptr)  :: coefficients_ptr, status_output_ptr
      type(tMat) :: reuse_mat_cpu
      type(tMat), dimension(:), pointer :: reuse_submatrices_cpu
      type(tVec) :: max_vec
      PetscInt :: row_loc
      type(tMat) :: mat_sparsity_match, mat_product_save
      integer, dimension(size(coefficients, 1), 2), target :: status_output
      logical :: output_first_complex
#endif
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

#if defined(PETSC_HAVE_KOKKOS)

      call MatGetType(matrix, mat_type, ierr)
      if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) then

         A_array = matrix%v
         reuse_triggered_cmat = .NOT. PetscObjectIsNull(cmat)
         reuse_triggered_reuse_mat = .NOT. PetscObjectIsNull(reuse_mat)
         reuse_int_reuse_mat = 0
         if (reuse_triggered_reuse_mat) then
            reuse_int_reuse_mat = 1
         end if
         reuse_array = reuse_mat%v
         coefficients_ptr = c_loc(coefficients)
         status_output_ptr = c_loc(status_output)
         num_terms = size(coefficients, 1)

         ! Build cmat with values correct up to the power poly_sparsity_order, along with
         ! mat_sparsity_match, mat_product_save, status_output and output_first_complex
         ! This is all petsc mat operations so runs on the device with kokkos mats
         call build_newton_fixed_sparsity_start(matrix, poly_order, poly_sparsity_order, coefficients, &
                  cmat, mat_sparsity_match, mat_product_save, status_output, output_first_complex)

         ! The build has always created cmat by this point, even if we're not reusing
         B_array = cmat%v
         sparsity_array = mat_sparsity_match%v
         prod_save_array = mat_product_save%v
         prod_save_exists_int = 0
         if (.NOT. PetscObjectIsNull(mat_product_save)) prod_save_exists_int = 1
         output_first_complex_int = 0
         if (output_first_complex) output_first_complex_int = 1

         ! Adds in all the fixed sparsity terms on the device
         call mat_mult_powers_share_sparsity_newton_kokkos(A_array, sparsity_array, prod_save_array, &
                  prod_save_exists_int, num_terms, poly_sparsity_order, &
                  coefficients_ptr, status_output_ptr, output_first_complex_int, &
                  reuse_int_reuse_mat, reuse_array, B_array)

         reuse_mat%v = reuse_array

         ! Delete the temporaries from the build
         call MatDestroy(mat_sparsity_match, ierr)
         call MatDestroy(mat_product_save, ierr)

         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then

            ! If we're doing reuse and debug, then we have to always output the result
            ! from the cpu version, as it will have coo preallocation structures set
            ! They aren't copied over if you do a matcopy (or matconvert)
            ! If we didn't do that the next time we come through this routine
            ! and try to call the cpu version with reuse, it will segfault
            if (reuse_triggered_cmat) then
               temp_mat = cmat
               call MatConvert(cmat, MATSAME, MAT_INITIAL_MATRIX, temp_mat_compare, ierr)
            else
               temp_mat_compare = cmat
            end if

            ! Debug check if the CPU and Kokkos versions are the same
            ! We send in an empty reuse_mat_cpu here always, as we can't pass through
            ! the same one Kokkos uses as it now only gets out the non-local rows we need
            ! (ie reuse_mat and reuse_mat_cpu are no longer the same size)
            reuse_submatrices_cpu => null()
            call mat_mult_powers_share_sparsity_newton_cpu(matrix, poly_order, poly_sparsity_order, &
                     coefficients, reuse_mat_cpu, reuse_submatrices_cpu, temp_mat)
            call destroy_matrix_reuse(reuse_mat_cpu, reuse_submatrices_cpu)

            call MatConvert(temp_mat, MATSAME, MAT_INITIAL_MATRIX, &
                        temp_mat_reuse, ierr)

            ! Find the biggest entry in the cpu result, the entries scale with 1/(the scale
            ! of the matrix), so the tolerance below is relative to this
            call MatCreateVecs(temp_mat_reuse, PETSC_NULL_VEC, max_vec, ierr)
            call MatGetRowMaxAbs(temp_mat_reuse, max_vec, PETSC_NULL_INTEGER_POINTER, ierr)
            call VecMax(max_vec, row_loc, max_cpu, ierr)

            call MatAXPYWrapper(temp_mat_reuse, PFLARE_MINUS_ONE, temp_mat_compare)
            ! Find the biggest entry in the difference
            call MatGetRowMaxAbs(temp_mat_reuse, max_vec, PETSC_NULL_INTEGER_POINTER, ierr)
            call VecMax(max_vec, row_loc, normy, ierr)
            call VecDestroy(max_vec, ierr)

            ! There is floating point compute in these inverses, so we have to be a
            ! bit more tolerant to rounding differences
            ! This recurrence is num_terms long, and per-term rounding differences between
            ! optimized and unoptimized builds (compiler reordering, not fp-contraction -
            ! this happens on the CPU and Kokkos sides independently) compound roughly
            ! quadratically, so scale the tolerance with num_terms rather than using a
            ! fixed bound that only holds at low order
            ! The tolerance is relative to the biggest entry, but never tighter than absolute
            if (normy .gt. PFLARE_TOL_MATFREE_NEWTON * dble(num_terms)**2 * max(PFLARE_ONE, max_cpu) &
                     .OR. normy/=normy) then
               !call MatFilter(temp_mat_reuse, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
               !call MatView(temp_mat_reuse, PETSC_VIEWER_STDOUT_WORLD, ierr)
               print *, "Diff Kokkos and CPU mat_mult_powers_share_sparsity", normy, "row", row_loc

               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
            end if
            call MatDestroy(temp_mat_reuse, ierr)
            if (.NOT. reuse_triggered_cmat) then
               call MatDestroy(cmat, ierr)
            else
               call MatDestroy(temp_mat_compare, ierr)
            end if
            cmat = temp_mat
         end if

      else

         call mat_mult_powers_share_sparsity_newton_cpu(matrix, poly_order, poly_sparsity_order, &
                  coefficients, reuse_mat, reuse_submatrices, cmat)

      end if
#else
      call mat_mult_powers_share_sparsity_newton_cpu(matrix, poly_order, poly_sparsity_order, &
                  coefficients, reuse_mat, reuse_submatrices, cmat)
#endif

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
      ! Matrix entries filled by MatGetRow are PetscScalar
      PetscScalar, dimension(:), allocatable :: vals
      type(tIS), dimension(1) :: col_indices, row_indices
      type(tMat) :: Ad, Ao, mat_sparsity_match, mat_product_save
      PetscInt, dimension(:), pointer :: colmap
      logical :: deallocate_submatrices
      type(int_vec), dimension(:), allocatable :: symbolic_ones
      type(real_vec), dimension(:), allocatable :: symbolic_vals
      integer(c_long_long) A_array
      MPIU_Comm :: MPI_COMM_MATRIX
      PetscReal, dimension(:), allocatable :: vals_power_temp, vals_previous_power_temp, temp
      PetscInt, dimension(:), pointer :: submatrices_ia, submatrices_ja, cols_two_ptr, cols_ptr
      ! Matrix entries filled by MatSeqAIJGetArray/MatGetRow are PetscScalar
      PetscScalar, dimension(:), pointer :: vals_two_ptr, vals_ptr
      PetscScalar, pointer :: submatrices_vals(:)
      PetscBool :: symmetric = PETSC_FALSE, inodecompressed = PETSC_FALSE, done
      PetscInt, parameter :: one = 1, zero = 0
      logical :: output_first_complex, skip_add
      PetscReal :: square_sum
      integer, dimension(size(coefficients, 1), 2) :: status_output

      ! ~~~~~~~~~~

      ! Must be set here rather than with an initialiser in the declaration, which would
      ! give it an implicit save and leave it .TRUE. for every call after a serial one
      deallocate_submatrices = .FALSE.

      call PetscObjectGetComm(matrix, MPI_COMM_MATRIX, ierr)
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)

      ! Get the local sizes
      call MatGetLocalSize(matrix, local_rows, local_cols, ierr)
      call MatGetSize(matrix, global_rows, global_cols, ierr)
      ! This returns the global index of the local portion of the matrix
      call MatGetOwnershipRange(matrix, global_row_start, global_row_end_plus_one, ierr)  
      call MatGetOwnershipRangeColumn(matrix, global_col_start, global_col_end_plus_one, ierr)

      ! Build cmat with values correct up to the power poly_sparsity_order, along with
      ! mat_sparsity_match, mat_product_save, status_output and output_first_complex
      ! which we need to compute the remaining fixed sparsity terms below
      call build_newton_fixed_sparsity_start(matrix, poly_order, poly_sparsity_order, coefficients, &
               cmat, mat_sparsity_match, mat_product_save, status_output, output_first_complex)

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
      ! Read-only access to the value array - the read variant doesn't bump
      ! the matrix object state, so passing in e.g. a pc->pmat won't trigger
      ! a spurious pc re-setup on every iteration
      call MatSeqAIJGetArrayRead(reuse_submatrices(1), submatrices_vals, ierr)
      
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
            ! An empty row of matrix leaves cols_two_ptr untouched by MatGetRow (ie stale),
            ! so we have to give its length rather than rely on its size
            call intersect_pre_sorted_indices_only(cols(1:ncols), cols_two_ptr(1:ncols_two), &
                     cols_index_one, cols_index_two, match_counter)
            
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

               ! Skips zero eigenvalues - see 
               ! the comment in calculate_gmres_polynomial_roots_newton                
               if (coefficients(term,1) == 0d0) then
                  term = term + 1
                  cycle
               end if

               ! ~~~~~~~~~~~
               ! Now can add the value to our matrix
               ! ~~~~~~~~~~~               
               if (ncols /= 0 .AND. status_output(term, 1) /= 1) then
                  call MatSetValues(cmat, one, [global_row_start + i_loc-1], ncols, cols, &
                        PFLARE_ONE/coefficients(term, 1) * vals_previous_power_temp(1:ncols), ADD_VALUES, ierr)   
               end if          
               
               ! Initialize with previous product before the A*prod subtraction
               vals_power_temp(1:ncols) = vals_previous_power_temp(1:ncols)        
               
               ! This is the 1/theta_i * A * prod but where A * prod has fixed sparsity
               do j_loc = 1, ncols

                  ! If we have no matching columns cycle this row
                  if (.NOT. associated(symbolic_ones(j_loc)%ptr)) cycle

                  ! symbolic_vals(j_loc)%ptr has the matching values of A in it
                  vals_power_temp(symbolic_ones(j_loc)%ptr) = vals_power_temp(symbolic_ones(j_loc)%ptr) - &
                           PFLARE_ONE/coefficients(term, 1) * &
                           symbolic_vals(j_loc)%ptr * vals_previous_power_temp(j_loc)
               end do

               term = term + 1

            ! If complex
            else

               ! A complex pair is never zero (its imaginary part is nonzero), so is
               ! never skipped. That matters here, as skipping a pair where the fixed
               ! sparsity root is the second of the pair would leave skip_add set
               ! and the wrong product in vals_previous_power_temp
               square_sum = PFLARE_ONE/(coefficients(term,1)**2 + coefficients(term,2)**2)

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
                        temp(1:ncols) = temp(1:ncols) + PFLARE_TWO * coefficients(term, 1) * vals_previous_power_temp(1:ncols)
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
            ! Skips zero eigenvalues
            ! vals_previous_power_temp has the final prod in it, even if the loop 
            ! above ended by skipping a zero root
            if (ncols /= 0 .AND. coefficients(size(coefficients, 1),1) /= 0d0) then
               call MatSetValues(cmat, one, [global_row_start + i_loc-1], ncols, cols, &
                     PFLARE_ONE/coefficients(size(coefficients, 1), 1) * vals_previous_power_temp(1:ncols), ADD_VALUES, ierr)
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
      call MatSeqAIJRestoreArrayRead(reuse_submatrices(1), submatrices_vals, ierr)

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

   subroutine point_mf_newton_roots(mat_ctx, coefficients)

      ! Points the context of a matrix-free Newton polynomial matshell at the roots
      ! in coefficients (real parts in column 1, imaginary parts in column 2)
      ! If the matshell has an inner diagonally scaled matshell (mat_ctx%mat_scaled)
      ! that is pointed at them too
      ! This must be called whenever the coefficients are reallocated, eg if the
      ! number of roots changes, otherwise the matshell is left pointing at freed memory
      ! Whether the matshell owns the coefficients is set by the caller in
      ! mat_ctx%own_coefficients, which this doesn't change (PCAIR never gives it ownership)

      ! ~~~~~~
      type(mat_ctxtype), intent(inout)                              :: mat_ctx
      PetscReal, dimension(:, :), target, contiguous, intent(inout) :: coefficients

      type(mat_ctxtype), pointer :: mat_ctx_scaled => null()
      PetscErrorCode :: ierr
      ! ~~~~~~

      mat_ctx%real_roots => coefficients(:, 1)
      mat_ctx%imag_roots => coefficients(:, 2)
      ! Pointer remapping turns the 2D coefficients into the 1D mat_ctx%coefficients
      ! If the caller sets mat_ctx%own_coefficients (PCPFLAREINV), the context reset
      ! deallocates through this pointer, which frees the whole 2D array
      mat_ctx%coefficients(1:2*size(coefficients,1)) => coefficients(:, :)

      if (.NOT. PetscObjectIsNull(mat_ctx%mat_scaled)) then
         call MatShellGetContext(mat_ctx%mat_scaled, mat_ctx_scaled, ierr)
         mat_ctx_scaled%coefficients(1:2*size(coefficients,1)) => coefficients(:, :)
         mat_ctx_scaled%real_roots => coefficients(:, 1)
         mat_ctx_scaled%imag_roots => coefficients(:, 2)
      end if

   end subroutine point_mf_newton_roots

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine build_gmres_polynomial_newton_inverse(matrix, poly_order, &
                  coefficients, &
                  poly_sparsity_order, matrix_free, diag_scale_polys, reuse_mat, reuse_submatrices, &
                  inv_matrix)

      ! Builds a matrix which is an approximation to the inverse of a matrix using the 
      ! gmres polynomial in the newton basis
      ! Can only be applied matrix-free

      ! ~~~~~~
      type(tMat), intent(in)                                      :: matrix
      integer, intent(in)                                         :: poly_order
      PetscReal, dimension(:, :), target, contiguous, intent(inout)    :: coefficients
      integer, intent(in)                                         :: poly_sparsity_order
      logical, intent(in)                                         :: matrix_free, diag_scale_polys
      type(tMat), intent(inout)                                   :: reuse_mat, inv_matrix
      type(tMat), dimension(:), pointer, intent(inout)            :: reuse_submatrices

      ! Local variables
      PetscInt :: global_rows, global_cols, local_rows, local_cols
      integer :: comm_size, errorcode, sparsity_order
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX
      VecType :: vtype
      type(mat_ctxtype), pointer :: mat_ctx=>null(), mat_ctx_scaled=>null()
      logical :: reuse_triggered
      type(tMat) :: diag_scaled_mat
      type(tVec) :: diag_inverse_vec

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
            if (diag_scale_polys) then
               ! The subroutine petsc_matvec_right_scale_poly_newton_mf applies
               ! q(mat_ctx%mat_scaled) D^-1
               call MatShellSetOperation(inv_matrix, &
                           MATOP_MULT, petsc_matvec_right_scale_poly_newton_mf, ierr)
               call mat_create_vec_device(matrix, .FALSE., mat_ctx%mf_temp_vec(MF_VEC_RHS))
               call mat_create_vec_device(matrix, .TRUE., mat_ctx%mf_temp_vec(MF_VEC_DIAG))
            else
               ! The subroutine petsc_matvec_poly_newton_mf applies the polynomial inverse, q(mat_ctx%mat)
            call MatShellSetOperation(inv_matrix, &
                        MATOP_MULT, petsc_matvec_poly_newton_mf, ierr)
            end if
            ! The subroutine petsc_matvec_poly_transpose_mf applies the transpose of
            ! whichever of the two above we just set - it builds a transposed twin of
            ! this matshell on demand, see ensure_transpose_mat
            call MatShellSetOperation(inv_matrix, &
                        MATOP_MULT_TRANSPOSE, petsc_matvec_poly_transpose_mf, ierr)

            call MatAssemblyBegin(inv_matrix, MAT_FINAL_ASSEMBLY, ierr)
            call MatAssemblyEnd(inv_matrix, MAT_FINAL_ASSEMBLY, ierr)
            ! Have to make sure to set the type of vectors the shell creates
            call MatGetVecType(matrix, vtype, ierr)
            call MatShellSetVecType(inv_matrix, vtype, ierr)
            
            ! Create temporary vectors we use during application
            ! Make sure to use matrix here to get the right type (as the shell doesn't know about gpus)
            call mat_create_vec_device(matrix, .FALSE., mat_ctx%mf_temp_vec(MF_VEC_TEMP))
            call mat_create_vec_device(matrix, .FALSE., mat_ctx%mf_temp_vec(MF_VEC_TEMP_THREE))
            call mat_create_vec_device(matrix, .TRUE., mat_ctx%mf_temp_vec(MF_VEC_TEMP_TWO))

            ! Need to build mat_ctx%mat_scaled, ie D^-1 A
            if (diag_scale_polys) then

               ! Have to dynamically allocate this
               allocate(mat_ctx_scaled)
               
               ! Create the matshell
               call MatCreateShell(MPI_COMM_MATRIX, local_rows, local_cols, global_rows, global_cols, &
                           mat_ctx_scaled, mat_ctx%mat_scaled, ierr)
               ! The subroutine petsc_matvec_da_poly_mf applies D^-1 A
               call MatShellSetOperation(mat_ctx%mat_scaled, &
                           MATOP_MULT, petsc_matvec_da_poly_mf, ierr)

               call MatAssemblyBegin(mat_ctx%mat_scaled, MAT_FINAL_ASSEMBLY, ierr)
               call MatAssemblyEnd(mat_ctx%mat_scaled, MAT_FINAL_ASSEMBLY, ierr)   
               ! Have to make sure to set the type of vectors the shell creates
               call MatGetVecType(matrix, vtype, ierr)
               call MatShellSetVecType(mat_ctx%mat_scaled, vtype, ierr)
            end if            

         ! Reusing 
         else
            call MatShellGetContext(inv_matrix, mat_ctx, ierr)
            if (diag_scale_polys) then
               call MatShellGetContext(mat_ctx%mat_scaled, mat_ctx_scaled, ierr)
            end if            
         end if

         ! Free old owned coefficients before reassigning.
         ! Avoids a memory leak when the same matshell is reused across PCSetUp calls
         ! (SAME_NONZERO_PATTERN) but fresh coefficients are being computed.
         ! For PCAIR, own_coefficients is always .FALSE., so this is a no-op there.
         if (mat_ctx%own_coefficients .AND. associated(mat_ctx%coefficients)) then
            deallocate(mat_ctx%coefficients)
            mat_ctx%coefficients => null()
         end if

         ! This is the matrix whose inverse we are applying (just copying the pointer here)
         mat_ctx%mat = matrix     
         if (diag_scale_polys) mat_ctx_scaled%mat = matrix

         ! Point the shell (and the inner diagonally scaled shell) at the roots
         ! The coefficients may have been reallocated since the last setup
         call point_mf_newton_roots(mat_ctx, coefficients)

         if (diag_scale_polys) then
            ! Get the diagonal
            call MatGetDiagonal(matrix, mat_ctx%mf_temp_vec(MF_VEC_DIAG), ierr)    
            mat_ctx_scaled%mf_temp_vec(MF_VEC_DIAG) = mat_ctx%mf_temp_vec(MF_VEC_DIAG)
         end if         
         
         ! We're done
         return
      endif

      ! ~~~~~~~~~~~~
      ! If we're here then we want an assembled approximate inverse
      ! ~~~~~~~~~~~~   
      reuse_triggered = .NOT. PetscObjectIsNull(inv_matrix)               

      ! If we want to diagonally scale (ie apply Jacobi preconditioned gmres polynomial)
      if (diag_scale_polys) then
         call mat_create_vec_device(matrix, .TRUE., diag_inverse_vec)
         call MatGetDiagonal(matrix, diag_inverse_vec, ierr)
         call VecReciprocal(diag_inverse_vec, ierr)
         call MatDuplicate(matrix, MAT_COPY_VALUES, diag_scaled_mat, ierr)
         call MatDiagonalScale(diag_scaled_mat, diag_inverse_vec, PETSC_NULL_VEC, ierr) 

      else
         diag_scaled_mat = matrix
      end if   

      ! The sparsity is set by the requested order, never by the number of roots, which can
      ! be more than poly_order + 1 and change between setups. This keeps the sparsity fixed
      ! when reusing, and matrix products of the inverse rely on that
      sparsity_order = min(poly_sparsity_order, poly_order)

      ! If we're zeroth order poly this is trivial as it's just 1/theta_1 I
      if (poly_order == 0 .AND. size(coefficients, 1) == 1) then

         call build_gmres_polynomial_newton_inverse_0th_order(diag_scaled_mat, poly_order, coefficients, &
               inv_matrix)
               
         if (diag_scale_polys) then
            call MatDestroy(diag_scaled_mat, ierr)
            call MatDiagonalScale(inv_matrix, PETSC_NULL_VEC, diag_inverse_vec, ierr)  
            call VecDestroy(diag_inverse_vec, ierr)      
         end if               

         ! Then just return
         return      

      ! For poly_order 1 and poly_sparsity_order 1 this is easy if there are only two roots
      else if (poly_order == 1 .AND. poly_sparsity_order == 1 .AND. size(coefficients, 1) == 2) then

         ! Duplicate & copy the matrix, but ensure there is a diagonal present
         call mat_duplicate_copy_plus_diag(diag_scaled_mat, reuse_triggered, inv_matrix)         
         
         call build_gmres_polynomial_newton_inverse_1st_1st(diag_scaled_mat, &
               poly_order, coefficients, inv_matrix)

         if (diag_scale_polys) then
            call MatDestroy(diag_scaled_mat, ierr)
            call MatDiagonalScale(inv_matrix, PETSC_NULL_VEC, diag_inverse_vec, ierr)  
            call VecDestroy(diag_inverse_vec, ierr)      
         end if               

         ! Then just return
         return

      end if

      ! If we're constraining sparsity we've built a custom matrix-powers that assumes fixed sparsity
      ! A sparsity order of 0 always goes through here, as that builds a matdiagonal
      if (sparsity_order == 0 .OR. sparsity_order < size(coefficients, 1) - 1) then

         ! This routine is a custom one that builds our matrix powers and assumes fixed sparsity
         ! so that it doen't have to do much comms
         call mat_mult_powers_share_sparsity_newton(diag_scaled_mat, poly_order, sparsity_order, coefficients, &
                  reuse_mat, reuse_submatrices, inv_matrix)

         if (diag_scale_polys) then
            call MatDestroy(diag_scaled_mat, ierr)
            call MatDiagonalScale(inv_matrix, PETSC_NULL_VEC, diag_inverse_vec, ierr)  
            call VecDestroy(diag_inverse_vec, ierr)      
         end if                  

         ! ! Then just return
         return         
         
      end if

      ! ~~~~~~~~~~
      ! We are only here if we don't constrain_sparsity
      ! ~~~~~~~~~~
      call build_gmres_polynomial_newton_inverse_full(diag_scaled_mat, poly_order, coefficients, &
                  inv_matrix)

      if (diag_scale_polys) then
         call MatDestroy(diag_scaled_mat, ierr)
         call MatDiagonalScale(inv_matrix, PETSC_NULL_VEC, diag_inverse_vec, ierr)  
         call VecDestroy(diag_inverse_vec, ierr)      
      end if                  

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
         call mat_create_vec_device(matrix, .TRUE., diag_vec)
      else
         call MatDiagonalGetDiagonal(inv_matrix, diag_vec, ierr)
      end if

      ! Must be real as we only have one coefficient
      ! Skips zero eigenvalues, like the matrix-free apply
      if (coefficients(1, 1) /= 0d0) then
         call VecSet(diag_vec, PFLARE_ONE/coefficients(1, 1), ierr)
      else
         call VecSet(diag_vec, PFLARE_ZERO, ierr)
      end if

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
      call mat_create_vec_device(matrix, .FALSE., product_vec)
      call mat_create_vec_device(matrix, .TRUE., diag_vec)
      call MatGetDiagonal(matrix, diag_vec, ierr)

      if (.NOT. reuse_triggered) then
         call mat_create_vec_device(matrix, .TRUE., inv_vec)
      else
         call MatDiagonalGetDiagonal(inv_matrix, inv_vec, ierr)
      end if
      call mat_create_vec_device(matrix, .TRUE., temp_vec_A)
      call mat_create_vec_device(matrix, .TRUE., one_vec)
      call mat_create_vec_device(matrix, .TRUE., temp_vec_two)
      
      ! Set to zero as we add to it
      call VecSet(inv_vec, PFLARE_ZERO, ierr) 
      ! We start with an identity in product_vec    
      call VecSet(product_vec, PFLARE_ONE, ierr)
      call VecSet(one_vec, PFLARE_ONE, ierr)

      i = 1
      do while (i .le. size(coefficients, 1) - 1)

         ! temp_vec_A is going to store things with the sparsity of A
         call VecCopy(diag_vec, temp_vec_A, ierr)     

         ! If real this is easy
         if (coefficients(i,2) == 0d0) then

            ! Skips zero eigenvalues - see 
            ! the comment in calculate_gmres_polynomial_roots_newton 
            if (coefficients(i,1) == 0d0) then
               i = i + 1
               cycle
            end if        

            call VecAXPY(inv_vec, PFLARE_ONE/coefficients(i,1), product_vec, ierr)

            ! temp_vec_A = A_ff/theta_k       
            call VecScale(temp_vec_A, PFLARE_MINUS_ONE/coefficients(i,1), ierr)
            ! temp_vec_A = I - A_ff/theta_k
            call VecAXPY(temp_vec_A, PFLARE_ONE, one_vec, ierr)
            
            ! product_vec = product_vec * temp_vec_A
            call VecPointwiseMult(product_vec, product_vec, temp_vec_A, ierr)   
            
            i = i + 1

         ! Complex 
         else

            ! A complex pair is never zero as its imaginary part is nonzero

            ! Compute 2a I - A
            ! temp_vec_A = -A    
            call VecScale(temp_vec_A, PFLARE_MINUS_ONE, ierr)
            ! temp_vec_A = 2a I - A_ff
            call VecAXPY(temp_vec_A, PFLARE_TWO * coefficients(i,1), one_vec, ierr) 
            ! temp_vec_A = (2a I - A_ff)/(a^2 + b^2)
            call VecScale(temp_vec_A, PFLARE_ONE/(coefficients(i,1)**2 + coefficients(i,2)**2), ierr) 

            ! temp_vec_two = temp_vec_A * product_vec
            call VecPointwiseMult(temp_vec_two, temp_vec_A, product_vec, ierr)   
            call VecAXPY(inv_vec, PFLARE_ONE, temp_vec_two, ierr)         

            if (i .le. size(coefficients, 1) - 2) then
               ! temp_vec_two = A * temp_vec_two
               call VecPointwiseMult(temp_vec_two, diag_vec, temp_vec_two, ierr) 
               call VecAXPY(product_vec, PFLARE_MINUS_ONE, temp_vec_two, ierr)
            end if

            ! Skip two evals
            i = i + 2

         end if       
      end do

      ! Final step if last root is real
      if (coefficients(size(coefficients,1),2) == 0d0) then
         ! Add in the final term multiplied by 1/theta_poly_order

         ! Skips zero eigenvalues
         if (coefficients(size(coefficients, 1),1) /= 0d0) then      
            call VecAXPY(inv_vec, PFLARE_ONE/coefficients(size(coefficients, 1),1), product_vec, ierr)  
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
      PetscReal :: square_sum, inv_theta_one

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

         ! Skips zero eigenvalues, like the matrix-free apply
         ! A skipped first root leaves the product as the identity
         if (coefficients(1,1) == 0d0) then
            inv_theta_one = 0
         else
            inv_theta_one = PFLARE_ONE/coefficients(1,1)
         end if

         ! Have to be careful here, as we may be first order, but the second eigenvaule
         ! might have been set to zero thanks to the rank reducing solve 
         ! So we just check if the second real part is zero and if it is
         ! we just compute a 0th order inverse - annoyingly we can't call 
         ! build_gmres_polynomial_newton_inverse_0th_order as that builds a MATDIAGONAL
         ! and in the tests there is a problem where we reuse the sparsity, in the first
         ! solve we don't have a zero coefficient but in the second solve we do
         ! So the mat type needs to remain consistent
         ! This can't happen in the complex case
         ! The matrix-free apply only applies the last root if it is non-zero
         if (coefficients(2,1) == 0d0) then

            ! Tricky case here as we want to pass out the product so far with the
            ! sparsity of A, as only the second root is skipped
            ! ie I - A_ff/theta_1, which is the identity if the first root is also zero
            if (output_product) then
               call MatConvert(inv_matrix, MATSAME, MAT_INITIAL_MATRIX, mat_prod_or_temp, ierr)  
               call MatScale(mat_prod_or_temp, -inv_theta_one, ierr)
               call MatShift(mat_prod_or_temp, PFLARE_ONE, ierr)
               status_output(1:2, 1) = 1
            end if                

            ! Set to zero
            call MatScale(inv_matrix, PFLARE_ZERO, ierr)

            ! Then add in the 0th order inverse
            call MatShift(inv_matrix, inv_theta_one, ierr)       
            
            ! Then just return
            return  
         end if

         ! Could just compute the equivalent mononomial here to save some flops
         ! but the whole point to doing the Newton form is to avoid the 
         ! theta_1 * theta_2 that would result

         ! result = -A_ff/theta_1
         call MatScale(inv_matrix, -inv_theta_one, ierr)
         ! result = I -A_ff/theta_1
         call MatShift(inv_matrix, PFLARE_ONE, ierr) 
         ! If we're doing this as part of fixed sparsity multiply, 
         ! we need to return mat_prod_or_temp
         if (output_product) then
            call MatConvert(inv_matrix, MATSAME, MAT_INITIAL_MATRIX, mat_prod_or_temp, ierr)  
         end if

         ! result = 1/theta_2 * (I -A_ff/theta_1)
         call MatScale(inv_matrix, PFLARE_ONE/(coefficients(2, 1)), ierr)      

         ! result = 1/theta_1 + 1/theta_2 * (I -A_ff/theta_1)
         ! Don't need an assemble as there is one called in this
         call MatShift(inv_matrix, inv_theta_one, ierr)     
         
         if (output_product) then
            status_output(1:2, 1) = 1
         end if

      ! Complex conjugate roots, a +- ib
      else
         ! 1/(a^2 + b^2)
         ! A complex pair is never zero as its imaginary part is nonzero
         square_sum = PFLARE_ONE/(coefficients(1,1)**2 + coefficients(1,2)**2)

         ! Complex conjugate roots
         ! result = -A_ff
         call MatScale(inv_matrix, PFLARE_MINUS_ONE, ierr)
         ! result = 2a I - A_ff
         ! Don't need an assemble as there is one called in this
         call MatShift(inv_matrix, PFLARE_TWO * coefficients(1,1), ierr)      
         ! If we're doing this as part of fixed sparsity multiply, 
         ! we need to return mat_prod_or_temp         
         if (output_product) then
            call MatConvert(inv_matrix, MATSAME, MAT_INITIAL_MATRIX, mat_prod_or_temp, ierr)  
         end if      
         ! result = 2a I - A_ff/(a^2 + b^2)
         call MatScale(inv_matrix, square_sum, ierr)

         if (output_product) then
            status_output(1:2, 2) = 1
         end if
      end if               

   end subroutine build_gmres_polynomial_newton_inverse_1st_1st     

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine add_sparsity_only(inv_matrix, mat_product)

      ! Adds the sparsity of mat_product to inv_matrix without changing any values
      ! MatAXPY with zero does nothing, so we have to add explicit zeros

      ! ~~~~~~
      type(tMat), intent(inout) :: inv_matrix
      type(tMat), intent(in)    :: mat_product

      ! Local variables
      PetscErrorCode :: ierr
      type(tMat) :: mat_zero

      ! ~~~~~~

      call MatConvert(mat_product, MATSAME, MAT_INITIAL_MATRIX, mat_zero, ierr)
      call MatScale(mat_zero, PFLARE_ZERO, ierr)
      call MatAXPYWrapper(inv_matrix, PFLARE_ONE, mat_zero)
      call MatDestroy(mat_zero, ierr)

   end subroutine add_sparsity_only

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
      call MatScale(inv_matrix, PFLARE_ZERO, ierr)

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
                  ! Don't divide by the root, as the real part may be zero (or negative)
                  if (abs(coefficients(i_sparse,1) - coefficients(i_sparse-1,1)) > &
                        PFLARE_TOL_CONSISTENCY * abs(coefficients(i_sparse,1)) .OR. &
                        abs(coefficients(i_sparse,2) + coefficients(i_sparse-1,2)) > &
                        PFLARE_TOL_CONSISTENCY * abs(coefficients(i_sparse,2))) then
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

            ! Skips zero eigenvalues
            ! We still compute the entries as as zero because we need the sparsity
            ! to be correct for the next iteration
            if (coefficients(i,1) == 0d0) then
               square_sum = 0
            else
               square_sum = PFLARE_ONE/coefficients(i,1)
            end if        

            ! Then add the scaled version of each product
            if (i == 1) then
               ! If i == 1 then we know mat_product is identity so we can do it directly
               call MatShift(inv_matrix, square_sum, ierr)  
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
            call MatShift(temp_mat_A, PFLARE_ONE, ierr)    
            
            ! mat_product_k_plus_1 = mat_product * temp_mat_A
            if (i == 1) then
               ! If i == 1 then we know mat_product is identity so we can just copy
               call MatDestroy(mat_product, ierr)
               call MatConvert(temp_mat_A, MATSAME, MAT_INITIAL_MATRIX, mat_product, ierr)  
            else
               call MatMatMult(temp_mat_A, mat_product, &
                     MAT_INITIAL_MATRIX, PFLARE_MATMULT_FILL, mat_product_k_plus_1, ierr)      
               call MatDestroy(mat_product, ierr)  
               mat_product = mat_product_k_plus_1  
            end if
            
            ! We copy out the last product if we're doing this as part of a fixed sparsity multiply
            if (output_product .AND. i == i_sparse - 1) then
               call MatDestroy(mat_prod_or_temp, ierr)
               call MatConvert(mat_product, MATSAME, MAT_INITIAL_MATRIX, mat_prod_or_temp, ierr)  
            end if
            
            i = i + 1

         ! Complex 
         else

            ! A complex pair is never zero as its imaginary part is nonzero
            ! The fixed sparsity continuation relies on this, as it never skips a complex pair
            square_sum = PFLARE_ONE/(coefficients(i,1)**2 + coefficients(i,2)**2)
            a_coeff = PFLARE_TWO * coefficients(i,1)

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
                  call MatDestroy(mat_prod_or_temp, ierr)
                  call MatConvert(mat_product, MATSAME, MAT_INITIAL_MATRIX, mat_prod_or_temp, ierr)  
               end if  
               
            ! Just do the normal loop
            else

               ! temp_mat_A = -A    
               call MatScale(temp_mat_A, PFLARE_MINUS_ONE, ierr)
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
                        MAT_INITIAL_MATRIX, PFLARE_MATMULT_FILL, temp_mat_two, ierr)     
               end if    
               
               ! We copy out the last part of the old product if we're doing this as part of a fixed sparsity multiply
               if (output_product .AND. i > i_sparse - 2) then

                  call MatDestroy(mat_prod_or_temp, ierr)
                  call MatConvert(temp_mat_two, MATSAME, MAT_INITIAL_MATRIX, mat_prod_or_temp, ierr) 
                  ! If i == 1 then we know mat_product is the identity and we don't bother 
                  ! to write it out, we just have some custom code in the product given its trivial
                  if (i /= 1) then 
                     ! This ensures it has the matching sparsity
                     call MatDestroy(mat_product_save, ierr)
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
                     MAT_INITIAL_MATRIX, PFLARE_MATMULT_FILL, temp_mat_three, ierr)     
               call MatDestroy(temp_mat_two, ierr)   
               if (output_product) then               
                  status_output(i, 2) = 1
               end if

               ! Then add the scaled version of each product
               ! Even if doing reuse this isn't a subset, as mat_product was built here and
               ! temp_mat_three has a higher power of A
               call MatAXPYWrapper(mat_product, -square_sum, temp_mat_three)

               ! We copy out the last part of the product if we're doing this as part of a fixed sparsity multiply
               if (output_product .AND. .NOT. first_complex) then
                  call MatDestroy(mat_prod_or_temp, ierr)
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

            ! Skips zero eigenvalues
            ! We still need the sparsity if we're not reusing, as this root may be
            ! non-zero when we reuse. This product has the sparsity of all the previous ones
            if (coefficients(i_sparse,1) == 0d0) then

               if (.NOT. reuse_triggered) call add_sparsity_only(inv_matrix, mat_product)

            else
               
               if (reuse_triggered) then
                  ! If doing reuse we know our nonzeros are a subset
                  call MatAXPY(inv_matrix, PFLARE_ONE/coefficients(i_sparse,1), mat_product, SUBSET_NONZERO_PATTERN, ierr)
               else
                  ! Have to use the DIFFERENT_NONZERO_PATTERN here
                  call MatAXPYWrapper(inv_matrix, PFLARE_ONE/coefficients(i_sparse,1), mat_product)
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

