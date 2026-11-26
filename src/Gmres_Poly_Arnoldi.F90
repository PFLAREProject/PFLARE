module gmres_poly_arnoldi

   use petscmat
   use gmres_poly, only: arnoldi
   use petsc_helper, only: kokkos_debug
   use gmres_poly_newton_roots, only: modified_leja, hessenberg_gmres_residuals
   use c_petsc_interfaces, only: sstep_fused_dots_kokkos, sstep_project_kokkos, sstep_apply_inverse_kokkos
   use pflare_parameters, only: PFLARE_EPS, PFLARE_ONE, PFLARE_ZERO

#include "petsc/finclude/petscmat.h"
#include "finclude/pflare_blaslapack.h"

   implicit none
   public

   ! Arnoldi used to build the Newton basis GMRES polynomial: continuing an Arnoldi one step
   ! at a time and the s-step (block) Arnoldi, with the kernels in Gmres_Poly_Arnoldik.kokkos.cxx

   contains


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
      PetscErrorCode :: ierr
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
         ! LAPACK's singular values only path probes the IEEE arithmetic with an Inf and a NaN
         ! (ieeeck), so floating point traps are turned off around it
         call PetscFPTrapPush(PETSC_FP_TRAP_OFF, ierr)
         call PFLAREgesvd('N', 'N', j_bl, j_bl, R_copy, lda_bl, &
                  sigma, U, one_bl, VT, one_bl, work, lwork, info)
         call PetscFPTrapPop(ierr)
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

   subroutine arnoldi_sstep(matrix, poly_order, s_max, m_start, lucky_tol, V_n, w_j, beta, H_n, m, floor_stop)

      ! An s-step Arnoldi that returns the same Hessenberg matrix as the arnoldi routine
      ! (to rounding) with far fewer reductions. This is the adaptive s-step method of
      ! Xu, Alonso & Darve, "A numerically stable communication-avoiding s-step GMRES algorithm"
      ! with the orthogonalisation of Carson, Lund, Ma & Oktay, "Reorthogonalized Pythagorean
      ! variants of block classical Gram-Schmidt" (BCGS-PIPI+), and a check of the Hessenberg
      ! matrix that neither has
      !
      ! The first m_start steps are an ordinary Arnoldi, orthogonalised twice. The Ritz values
      ! of those, leja ordered,
      ! are the shifts of a scaled Newton basis. After that we build a block of up to s new
      ! Krylov vectors with matvecs only, and orthogonalise the block in two passes
      ! Each pass gets the components along the existing basis and the Gram matrix of the
      ! block in one reduction, [Q W]^T W, with the Gram matrix of the projected block from
      ! Pythagoras as W^T W - (Q^T W)^T (Q^T W), and then a Cholesky QR of the block. So a
      ! block costs two reductions however big it is
      ! The Cholesky stops at the first column where the block becomes too ill-conditioned.
      ! The columns after that are thrown away and the block size is reduced to the number
      ! kept. It is never increased again
      ! In the first pass the Pythagorean difference loses accuracy when the block vectors lie
      ! mostly in the existing basis, so the Cholesky then also drops a column once the norm of
      ! the unprojected vectors over the smallest singular value of the factor passes the
      ! condition bound. The block is nearly orthonormal by the second pass
      !
      ! The new columns of H are derived from the existing ones, and their errors can grow from
      ! block to block (see the comment at column_ratio). So the last column of each block is
      ! checked after the first pass, with one matvec and a norm that rides in the reduction of
      ! the second pass. A block that fails is halved, and one that keeps nothing hands over to
      ! a stretch of ordinary two-pass steps before going back to blocks
      ! If the block size goes below three the rest of the steps are an ordinary Arnoldi,
      ! which costs no more reductions than the blocks would
      !
      ! Carson & Ma, "A stable one-synchronization variant of reorthogonalized block classical
      ! Gram-Schmidt" get to one reduction a block by sharing the second pass with the first
      ! projection of the next block. That was tried and works, but the next block then has to
      ! be built before this one is checked, and its matvecs are wasted if this one is cut
      !
      ! The block is built in place in V_n, so this needs no more vectors than the arnoldi routine
      ! If the vectors in V_n are slices of one device buffer (see create_temp_space_box_muller),
      ! the block work is dense matrix products on the device with our own kernels (KokkosBlas
      ! with a rocBLAS backend was no faster and its first big gemm cost 0.2 s)
      ! Otherwise it is done with PETSc vector operations

      ! ~~~~~~
      type(tMat), intent(in)                    :: matrix
      integer, intent(in)                       :: poly_order, s_max, m_start
      PetscReal, intent(in)                     :: lucky_tol
      type(tVec), dimension(:), intent(in)      :: V_n
      type(tVec), intent(in)                    :: w_j
      PetscReal, intent(out)                    :: beta
      PetscReal, dimension(:,:), intent(inout)  :: H_n
      integer, intent(out)                      :: m
      logical, intent(in)                       :: floor_stop

      ! Local variables
      integer, parameter :: dp = kind(1d0)
      logical :: use_kernels, lucky, at_floor
      integer(c_long_long), dimension(:), allocatable :: vec_handles
      integer(c_long_long) :: w_handle
      integer :: k, m0, j, s, sb, p, p_two, i_loc, col_loc, n_old, errorcode
      integer :: p_block, n_tightened, n_resume, n_resume_start, m_target
      integer :: n_tightened_before
      real(kind=dp) :: ratio_max, ratio_before
      real(kind=dp), dimension(:), allocatable :: column_ratio, t_diag
      PetscReal :: relation_error, relation_tol, w_norm_sq
      PetscReal, dimension(poly_order + 1) :: rho_block
      PetscErrorCode :: ierr
      PetscInt :: n_petscint
      PetscReal, dimension(:), allocatable :: ritz_real, ritz_imag, shift_real, shift_imag, work, dots
      PetscReal, dimension(:,:), allocatable :: H_copy, VL, VR
      integer, dimension(:), allocatable :: indices
      real(kind=dp), dimension(:,:), allocatable :: B, X_one, T_one, G, Z, Z_two, Y_two, D, C, R_new
      real(kind=dp) :: mean, ritz_max, omega
      PetscReal :: alpha_r, beta_r, gamma_r
      PetscBLASInt :: m_bl, lda_bl, one_bl, lwork, info

      ! ~~~~~~

      ! The number of Arnoldi steps we want
      k = poly_order + 1
      m0 = max(1, min(m_start, k))
      s = max(1, s_max)
      use_kernels = .TRUE.
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
      ! With two Gram-Schmidt passes from the second step on, as the blocks rely on the basis
      ! they are given being orthonormal: the Pythagorean Gram matrix of a block is wrong by
      ! the loss of orthogonality of the existing basis times the square of the block vectors,
      ! and one pass can lose 1e-7 in ten steps on an ill-conditioned matrix. A block vector
      ! whose new part is smaller than that then gets the wrong normalisation
      ! ~~~~~~~~~
      call arnoldi(matrix, 0, lucky_tol, V_n, w_j, beta, H_n, m)
      if (m0 > 1 .AND. .NOT. H_n(2, 1) < lucky_tol) then
         call arnoldi_extend(matrix, lucky_tol, V_n, w_j, H_n, 1, m0, m, 2, floor_stop)
      end if
      n_tightened = 0
      lucky = .FALSE.
      n_resume_start = 10
      n_resume = n_resume_start
      ! A block is cut shorter if the relative error in A v = V H for its last column passes this
      relation_tol = 1d-1 * sqrt(PFLARE_EPS)
      ! The error made in one block is about eps times this ratio, so it starts where that is
      ! well below the tolerance
      ratio_max = 1d-3/sqrt(real(PFLARE_EPS, kind=dp))
      ! Have we finished, hit the lucky tolerance or the noise floor
      if (m == k .OR. H_n(m+1, m) < lucky_tol) return
      if (floor_stop) then
         call hessenberg_gmres_residuals(H_n, m, rho_block)
         if (minval(rho_block(1:m)) <= 10 * PFLARE_EPS) return
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
      deallocate(ritz_real, ritz_imag)

      allocate(dots(k + 1), t_diag(s), column_ratio(s))
      allocate(B(s + 1, s), X_one(k + 1, s), T_one(s, s), G(s, s), Z(s, s), Z_two(s, s), Y_two(k + 1, s), D(k + 1, s))

      ! j is the number of Arnoldi steps finished, so V_n(1:j+1) is the basis so far
      j = m
      do while (j < k)

         ! Blocks this small cost as many reductions as ordinary steps
         if (s < 3) exit

         sb = min(s, k - j)
         n_old = j + 1
         call build_block(j, sb, B)

         ! ~~~~~~~~~
         ! First pass: the components along the existing basis and the Gram matrix of the
         ! block in one reduction
         ! ~~~~~~~~~
         call fused_dots(1, n_old + sb, j + 2, sb, .FALSE.)
         X_one(1:n_old, 1:sb) = D(1:n_old, 1:sb)
         T_one(1:sb, 1:sb) = D(n_old + 1:n_old + sb, 1:sb)
         call project_block(sb, X_one)

         G(1:sb, 1:sb) = T_one(1:sb, 1:sb) - matmul(transpose(X_one(1:n_old, 1:sb)), X_one(1:n_old, 1:sb))
         do col_loc = 1, sb
            t_diag(col_loc) = T_one(col_loc, col_loc)
         end do
         call partial_cholesky(G, sb, omega, p, Z, t_diag)
         if (p == 0) exit
         call apply_inverse_factor(p, Z)

         ! The Hessenberg columns come from dividing by the diagonal of the Cholesky factor, the
         ! part of each block vector that is new. The Cholesky only bounds how that varies
         ! across the block, not how small it is against the whole vector, which mostly lies in
         ! the existing basis when the Newton basis is poor. The new columns of H are derived
         ! from the existing ones, whose errors are multiplied by about that ratio, so the error
         ! in H can grow from block to block (to order one on ill-conditioned matrices if nothing
         ! is done). So we only keep the columns where the ratio is below ratio_max
         do col_loc = 1, p
            column_ratio(col_loc) = max(1d0, sqrt(sum(X_one(1:n_old, col_loc)**2) + sum(Z(1:col_loc, col_loc)**2))) &
                     /Z(col_loc, col_loc)
         end do
         p_block = p
         ratio_before = ratio_max
         n_tightened_before = n_tightened
         call cut_by_ratio()

         if (p > 0) then

            ! The Hessenberg columns from the first pass alone, and with them the check of the
            ! last column, whose norm rides in the next reduction
            call recover_hessenberg(p, X_one, Z)
            call relation_vector(p)

            ! ~~~~~~~~~
            ! Second pass, in one reduction: the block against the existing basis and itself
            ! ~~~~~~~~~
            call fused_dots(1, n_old + p, j + 2, p, .TRUE.)
               Y_two(1:n_old, 1:p) = D(1:n_old, 1:p)
            T_one(1:p, 1:p) = D(n_old + 1:n_old + p, 1:p)

            ! If the check fails keep half as many columns and check again, which then costs
            ! a reduction of its own. The columns are nested, so the second pass needs no rework
            relation_error = sqrt(w_norm_sq)/norm2(H_n(1:j + p + 1, j + p))
            do
               if (relation_error <= relation_tol) exit
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
                     end do

            ! If the check cut this block to fewer than three columns the ratio is not what is
            ! wrong either, so treat it in the same way as keeping none
            if (p > 0 .AND. p < 3 .AND. p_block >= 3 .AND. n_tightened > n_tightened_before) p = 0
         end if

         if (p > 0) then
            G(1:p, 1:p) = T_one(1:p, 1:p) - matmul(transpose(Y_two(1:n_old, 1:p)), Y_two(1:n_old, 1:p))
            call partial_cholesky(G, p, omega, p_two, Z_two)
            if (p_two == 0) then
               H_n(:, j + 1:j + p_block) = 0
               exit
            end if
            p = p_two
            call project_block(p, Y_two)
            call apply_inverse_factor(p, Z_two)

            ! The block before orthogonalising is W(:, 1:p) = V_n(1:j+1) C + V_n(j+2:j+p+1) R_new
            allocate(C(n_old, p), R_new(p, p))
            C = X_one(1:n_old, 1:p) + matmul(Y_two(1:n_old, 1:p), Z(1:p, 1:p))
            R_new = matmul(Z_two(1:p, 1:p), Z(1:p, 1:p))
            call recover_hessenberg(p, C, R_new)
            deallocate(C, R_new)
         end if

         if (p == 0) then

            ! Not even one column of this block could be kept accurately, so the error is in the
            ! columns of H we already have. Some ordinary steps measure their columns of H
            ! directly, after which the blocks no longer build on the inaccurate ones (exactly
            ! so if H is banded). We do twice as many each time this happens in a row
            ! The bound goes back to what it was, as it was not the ratio that failed this block
            ratio_max = ratio_before
            H_n(:, j + 1:j + p_block) = 0
            m_target = min(k, j + n_resume)
            call arnoldi_extend(matrix, lucky_tol, V_n, w_j, H_n, j, m_target, m, 2, floor_stop)
            n_resume = 2 * n_resume
            ! The ordinary steps may also reach the noise floor at their last step
            at_floor = .FALSE.
            if (floor_stop) then
               call hessenberg_gmres_residuals(H_n, m, rho_block)
               at_floor = minval(rho_block(1:m)) <= 10 * PFLARE_EPS
            end if
            if (m < m_target .OR. H_n(m + 1, m) < lucky_tol .OR. at_floor) then
               j = m
               lucky = .TRUE.
               exit
            end if
            j = m
            cycle
         end if
         n_resume = n_resume_start

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

      deallocate(shift_real, shift_imag, dots, t_diag, column_ratio)
      deallocate(B, X_one, T_one, G, Z, Z_two, Y_two, D)

      ! Any steps left are an ordinary Arnoldi, with two passes as the blocks are orthogonalised twice
      m = j
      if (j < k .AND. .NOT. lucky) then
         call arnoldi_extend(matrix, lucky_tol, V_n, w_j, H_n, j, k, m, 2, floor_stop)
      end if

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
                     c_loc(D_k), w_norm_sq_k, done)
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
                     c_loc(X_k), done)
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
         ! of the block and the factors of its vectors in the basis:
         ! W(:, 1:n) = V_n(1:j+1) C_in + V_n(j+2:j+n+1) R_in

         integer, intent(in) :: n_cols
         real(kind=dp), dimension(:,:), intent(in) :: C_in, R_in

         integer :: col, col_two
         real(kind=dp), dimension(:,:), allocatable :: R_full, rhs

         allocate(R_full(j + n_cols + 1, n_cols + 1), rhs(j + n_cols + 1, n_cols))
         R_full = 0
         R_full(j + 1, 1) = 1
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

   end subroutine arnoldi_sstep

! -------------------------------------------------------------------------------------------------------------------------------

end module gmres_poly_arnoldi
