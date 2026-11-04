!
!  Checks the coarse perturbations computed by the PCAIR diagonal update against an explicit
!  calculation on every level, d_{l+1} = R_l (d_l .* P_l 1)
!  The update on the top grid doesn't compute the C points of d_1 .* P_1 1, as it relies on
!  the C point rows of P_1 summing to exactly one, so this also checks that on every level
!  directly. Use with any PCAIR options on the command line, including -mat_type
!
!  Input arguments are:
!        -n <int>     : grid points in each direction (default 60)
!        -laplacian   : a symmetric problem rather than upwind advection diffusion
!
      program main
#include <petsc/finclude/petscksp.h>
      use petscksp
      use pcair_data_type, only: pc_air_multigrid_data
#include "finclude/pflare.h"
      implicit none

      PetscErrorCode :: ierr
      PetscInt :: n, nn, row, row_start, row_end, i, j, cols(5), ncols
      PetscInt :: no_updates, update
      PetscScalar :: vals(5), val
      PetscBool :: laplacian, flg
      PetscReal :: scale
      integer :: errorcode
      Mat :: A
      Vec :: x, b, d, diag
      KSP :: ksp
      PC :: pc, pc_shell
      type(pc_air_multigrid_data), pointer :: pc_air_data => null()
      PetscInt, parameter :: one = 1
      PetscReal, parameter :: r_one = 1.0_PETSC_REAL_KIND

      PetscCallA(PetscInitialize(ierr))
      n = 60
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-n', n, flg, ierr))
      PetscCallA(PetscOptionsHasName(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-laplacian', laplacian, ierr))
      nn = n * n

      ! ~~~~~~~~~~~~~~
      ! Assemble a 5 point operator
      ! ~~~~~~~~~~~~~~
      PetscCallA(MatCreate(PETSC_COMM_WORLD, A, ierr))
      PetscCallA(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, nn, nn, ierr))
      PetscCallA(MatSetFromOptions(A, ierr))
      PetscCallA(MatSeqAIJSetPreallocation(A, 5 * one, PETSC_NULL_INTEGER_ARRAY, ierr))
      PetscCallA(MatMPIAIJSetPreallocation(A, 5 * one, PETSC_NULL_INTEGER_ARRAY, 2 * one, PETSC_NULL_INTEGER_ARRAY, ierr))
      PetscCallA(MatGetOwnershipRange(A, row_start, row_end, ierr))
      do row = row_start, row_end - 1
         i = mod(row, n)
         j = row / n
         ncols = 1
         cols(1) = row
         if (laplacian) then
            vals(1) = 4 * r_one
            val = -r_one
         else
            ! Upwind advection in +x and +y with a little diffusion
            vals(1) = 2.04_PETSC_REAL_KIND
            val = -0.01_PETSC_REAL_KIND
         end if
         if (i > 0) then
            ncols = ncols + 1
            cols(ncols) = row - 1
            vals(ncols) = merge(-r_one, -1.01_PETSC_REAL_KIND, laplacian)
         end if
         if (j > 0) then
            ncols = ncols + 1
            cols(ncols) = row - n
            vals(ncols) = merge(-r_one, -1.01_PETSC_REAL_KIND, laplacian)
         end if
         if (i < n - 1) then
            ncols = ncols + 1
            cols(ncols) = row + 1
            vals(ncols) = val
         end if
         if (j < n - 1) then
            ncols = ncols + 1
            cols(ncols) = row + n
            vals(ncols) = val
         end if
         PetscCallA(MatSetValues(A, one, [row], ncols, cols, vals, INSERT_VALUES, ierr))
      end do
      PetscCallA(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr))
      PetscCallA(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr))

      PetscCallA(MatCreateVecs(A, x, b, ierr))
      PetscCallA(VecSet(b, r_one, ierr))
      PetscCallA(VecDuplicate(x, d, ierr))
      PetscCallA(VecDuplicate(x, diag, ierr))

      ! ~~~~~~~~~~~~~~
      ! Build the hierarchy
      ! ~~~~~~~~~~~~~~
      PetscCallA(KSPCreate(PETSC_COMM_WORLD, ksp, ierr))
      PetscCallA(KSPSetOperators(ksp, A, A, ierr))
      PetscCallA(KSPGetPC(ksp, pc, ierr))
      call PCRegister_PFLARE()
      PetscCallA(PCSetType(pc, PCAIR, ierr))
      PetscCallA(PCAIRSetDiagUpdate(pc, PETSC_TRUE, ierr))
      PetscCallA(KSPSetFromOptions(ksp, ierr))
      PetscCallA(KSPSetUp(ksp, ierr))

      call PCAIRGetPCShell(pc, pc_shell)
      PetscCallA(PCShellGetContext(pc_shell, pc_air_data, ierr))

      if (pc_air_data%air_data%no_levels < 2) then
         print *, "The hierarchy needs at least two levels for this test"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      call check_c_rows(pc_air_data)

      ! ~~~~~~~~~~~~~~
      ! Update and check, moving the diagonal up and down from what it was built with 
      ! The solves leave other values in the PCMG work vectors
      ! ~~~~~~~~~~~~~~
      no_updates = 3
      do update = 1, no_updates
         scale = merge(r_one, merge(0.01_PETSC_REAL_KIND, 10 * r_one, update == 2), update == 1)
         do row = row_start, row_end - 1
            val = scale * (1 + mod(row, 7 * one))
            PetscCallA(VecSetValues(d, one, [row], [val], INSERT_VALUES, ierr))
         end do
         PetscCallA(VecAssemblyBegin(d, ierr))
         PetscCallA(VecAssemblyEnd(d, ierr))
         PetscCallA(VecWAXPY(diag, r_one, pc_air_data%air_data%diag_update_base_pmat, d, ierr))
         PetscCallA(MatDiagonalSet(A, diag, INSERT_VALUES, ierr))
         PetscCallA(PCSetUp(pc, ierr))

         call check_perturbations(pc_air_data, d)
         call check_c_rows(pc_air_data)

         PetscCallA(KSPSolve(ksp, b, x, ierr))
      end do

      PetscCallA(PetscPrintf(PETSC_COMM_WORLD, "Diagonal update checks passed\n", ierr))

      PetscCallA(VecDestroy(x, ierr))
      PetscCallA(VecDestroy(b, ierr))
      PetscCallA(VecDestroy(d, ierr))
      PetscCallA(VecDestroy(diag, ierr))
      PetscCallA(MatDestroy(A, ierr))
      PetscCallA(KSPDestroy(ksp, ierr))
      PetscCallA(PetscFinalize(ierr))

      contains

      subroutine check_c_rows(pc_air_data)

         ! The C point rows of P on every level must sum to exactly one

         type(pc_air_multigrid_data), intent(in) :: pc_air_data

         integer :: our_level, errorcode
         PetscErrorCode :: ierr
         PetscReal :: norm
         Vec :: coarse_one, p_one, p_one_c, temp

         do our_level = 1, pc_air_data%air_data%no_levels - 1
            PetscCall(MatCreateVecs(pc_air_data%air_data%prolongators(our_level), coarse_one, p_one, ierr))
            PetscCall(VecSet(coarse_one, r_one, ierr))
            PetscCall(MatMult(pc_air_data%air_data%prolongators(our_level), coarse_one, p_one, ierr))
            PetscCall(VecGetSubVector(p_one, pc_air_data%air_data%IS_coarse_index(our_level), p_one_c, ierr))
            PetscCall(VecDuplicate(p_one_c, temp, ierr))
            PetscCall(VecCopy(p_one_c, temp, ierr))
            PetscCall(VecShift(temp, -r_one, ierr))
            PetscCall(VecNorm(temp, NORM_INFINITY, norm, ierr))
            PetscCall(VecDestroy(temp, ierr))
            PetscCall(VecRestoreSubVector(p_one, pc_air_data%air_data%IS_coarse_index(our_level), p_one_c, ierr))
            PetscCall(VecDestroy(coarse_one, ierr))
            PetscCall(VecDestroy(p_one, ierr))
            ! Exact, as these entries are a single one times a one
            if (norm /= 0.0) then
               print *, "C point rows of P don't sum to one on level", our_level, "error", norm
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
            end if
         end do

      end subroutine check_c_rows

      subroutine check_perturbations(pc_air_data, d_top)

         ! Walks down the hierarchy computing the coarse perturbations explicitly, keeping
         ! all of d .* P 1 rather than assuming anything about the C points

         type(pc_air_multigrid_data), intent(in) :: pc_air_data
         Vec, intent(in) :: d_top

         integer :: our_level, no_levels, errorcode
         PetscErrorCode :: ierr
         PetscReal :: norm, norm_exact, tol
         Vec :: d_level, d_coarse, coarse_one, p_one, computed

         no_levels = pc_air_data%air_data%no_levels
         tol = 1000 * epsilon(tol)
         PetscCall(VecDuplicate(d_top, d_level, ierr))
         PetscCall(VecCopy(d_top, d_level, ierr))

         do our_level = 1, no_levels - 1
            PetscCall(MatCreateVecs(pc_air_data%air_data%prolongators(our_level), coarse_one, p_one, ierr))
            PetscCall(VecDuplicate(coarse_one, d_coarse, ierr))
            PetscCall(VecSet(coarse_one, r_one, ierr))
            PetscCall(MatMult(pc_air_data%air_data%prolongators(our_level), coarse_one, p_one, ierr))
            PetscCall(VecPointwiseMult(p_one, p_one, d_level, ierr))
            if (pc_air_data%air_data%options%symmetric) then
               PetscCall(MatMultTranspose(pc_air_data%air_data%prolongators(our_level), p_one, d_coarse, ierr))
            else
               PetscCall(MatMult(pc_air_data%air_data%restrictors(our_level), p_one, d_coarse, ierr))
            end if

            ! The update leaves the perturbation in the x on each level, except the
            ! coarsest where it has the baseline added
            PetscCall(VecDuplicate(d_coarse, computed, ierr))
            PetscCall(VecCopy(pc_air_data%air_data%diag_update_x(our_level + 1), computed, ierr))
            if (our_level + 1 == no_levels) then
               PetscCall(VecAXPY(computed, -r_one, pc_air_data%air_data%diag_update_base_aff(no_levels), ierr))
            end if
            PetscCall(VecAXPY(computed, -r_one, d_coarse, ierr))
            PetscCall(VecNorm(computed, NORM_INFINITY, norm, ierr))
            PetscCall(VecNorm(d_coarse, NORM_INFINITY, norm_exact, ierr))
            ! The coarsest level has gone through a sum with the baseline
            if (norm > tol * max(norm_exact, r_one)) then
               print *, "Perturbation on level", our_level + 1, "error", norm, "size", norm_exact
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
            end if

            PetscCall(VecDestroy(computed, ierr))
            PetscCall(VecDestroy(coarse_one, ierr))
            PetscCall(VecDestroy(p_one, ierr))
            PetscCall(VecDestroy(d_level, ierr))
            d_level = d_coarse
         end do
         PetscCall(VecDestroy(d_level, ierr))

      end subroutine check_perturbations

      end program main
