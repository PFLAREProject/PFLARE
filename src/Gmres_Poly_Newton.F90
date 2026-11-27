module gmres_poly_newton

   use petscmat
   use gmres_poly, only: arnoldi, create_temp_space_box_muller, destroy_temp_space, int_vec, real_vec
   use petsc_helper, only: MatAXPYWrapper, destroy_matrix_reuse, mat_create_vec_device, &
         mat_duplicate_copy_plus_diag, generate_identity, kokkos_debug
   use matshell_data_type, only: mat_ctxtype
   use sorting, only: intersect_pre_sorted_indices_only, merge_pre_sorted
   use gmres_poly_newton_roots, only: modified_leja, compute_extra_roots, harmonic_ritz_roots, &
         leja_order_with_extras, adaptive_newton_roots
   use gmres_poly_arnoldi, only: arnoldi_extend, arnoldi_sstep
   use c_petsc_interfaces, only: mat_mult_powers_share_sparsity_newton_kokkos
   use pflare_parameters, only: PFLARE_TOL_CONSISTENCY, PFLARE_TOL_LUCKY, &
         PFLARE_ONE, PFLARE_ZERO, PFLARE_MINUS_ONE, PFLARE_TWO, PFLARE_MATMULT_FILL, &
         PFLARE_TOL_MATFREE_NEWTON, MF_VEC_DIAG, MF_VEC_RHS, MF_VEC_TEMP, MF_VEC_TEMP_TWO, &
         MF_VEC_TEMP_THREE
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
   
   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine calculate_gmres_polynomial_roots_newton(matrix, poly_order, root_rule, coefficients, arnoldi_block_size)

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

      ! If arnoldi_block_size is given and bigger than zero, the Arnoldi is an s-step one
      ! (see arnoldi_sstep) that builds that many Krylov vectors at a time

      ! ~~~~~~
      type(tMat), intent(in)                            :: matrix
      integer, intent(in)                               :: poly_order
      integer, intent(in)                               :: root_rule
      PetscReal, dimension(:, :), pointer, intent(inout)     :: coefficients
      integer, intent(in), optional                     :: arnoldi_block_size

      ! Local variables
      PetscInt :: global_rows, global_cols, local_rows, local_cols
      integer :: subspace_size, i_loc, comm_size, comm_rank, errorcode
      integer :: counter, m, numerical_order
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX
      PetscReal, dimension(poly_order+2,poly_order+1) :: H_n
      integer, dimension(:), allocatable :: indices
      PetscReal, dimension(:), allocatable :: real_roots_added, imag_roots_added
      PetscReal, dimension(:), allocatable :: real_roots_alt, imag_roots_alt
      PetscReal :: beta
      PetscReal, dimension(:, :), allocatable :: coefficients_temp
      type(tVec) :: w_j
      type(tVec), pointer, dimension(:) :: V_n
      integer :: sstep_size
      logical :: floor_stop

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
      ! The block size of the s-step Arnoldi, with zero the ordinary Arnoldi
      sstep_size = 0
      if (present(arnoldi_block_size)) sstep_size = arnoldi_block_size
      ! The adaptive rule only uses the steps up to where the Arnoldi residual reaches rounding
      ! level, so the Arnoldi can stop there
      floor_stop = root_rule == NEWTON_ROOTS_ADAPTIVE

      if (sstep_size > 0) then
         ! The first ten steps are ordinary, and give the shifts
         call arnoldi_sstep(matrix, poly_order, sstep_size, 10, PFLARE_TOL_LUCKY, &
                  V_n, w_j, beta, H_n, m, floor_stop)
      else if (floor_stop) then
         ! The same Arnoldi one step at a time, which can stop at the noise floor
         call arnoldi(matrix, 0, PFLARE_TOL_LUCKY, V_n, w_j, beta, H_n, m)
         if (poly_order > 0 .AND. .NOT. H_n(2, 1) < PFLARE_TOL_LUCKY) then
            call arnoldi_extend(matrix, PFLARE_TOL_LUCKY, V_n, w_j, H_n, 1, poly_order + 1, m, 1, floor_stop)
         end if
      else
         call arnoldi(matrix, poly_order, PFLARE_TOL_LUCKY, V_n, w_j, beta, H_n, m)
      end if

      if (root_rule == NEWTON_ROOTS_ADAPTIVE) then

         ! ~~~~~~~~~~~~~~
         ! Choose the degree and the extra roots from the Hessenberg matrix
         ! This returns the non-zero roots (with any extras) already leja ordered
         ! ~~~~~~~~~~~~~~
         call adaptive_newton_roots(H_n, m, real_roots_added, imag_roots_added, numerical_order, &
                  real_roots_alt, imag_roots_alt)

         ! Another order of the same roots is returned if it is expected to give a more
         ! accurate apply (see greedy_head_tail_order)
         if (allocated(real_roots_alt)) then
            real_roots_added = real_roots_alt
            imag_roots_added = imag_roots_alt
            deallocate(real_roots_alt, imag_roots_alt)
         end if

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

      ! Cleanup
      call destroy_temp_space(V_n)
      call VecDestroy(w_j, ierr)       

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

