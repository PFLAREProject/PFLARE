!
!  Verify PCMatApply on PCPFLAREINV: each column of Y = PCMatApply(pc, X)
!  must match PCApply(pc, X(:,j)) within tolerance. Exercises both the
!  assembled (MatProduct/SpMM) branch and the matrix-free (column-by-column)
!  branch in PCMatApply_PFLAREINV_c, for the POWER and ARNOLDI bases.
!

      program main
      use petscksp
#include "petsc/finclude/petscksp.h"
#include "finclude/pflare.h"
      implicit none

      PetscErrorCode :: ierr
      Mat :: A, X, Y
      PetscInt :: m, n, II, JJ, i, j, k
      PetscInt :: Istart, Iend
      PetscInt, parameter :: one = 1, ncols = 3
      PetscScalar :: v
      Vec :: y_ref, x_col, y_col
      PC :: pc
      PetscRandom :: rctx
      PetscReal :: norm_diff, norm_ref
      PetscReal, parameter :: tol = 1d-10
      PetscInt :: i_type, i_mf
      PCPFLAREINVType :: inv_types(2)
      PetscBool :: mf_flags(2)
      character(len=16) :: inv_label, mf_label

      call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
      if (ierr /= 0) then
         print *, 'Unable to initialize PETSc'
         stop
      end if

      ! Register the pflare PC types
      call PCRegister_PFLARE()

      m = 10
      n = 10

      ! 5-point 2D Laplacian on an m x n grid (same as ex6f.F90)
      call MatCreate(PETSC_COMM_WORLD, A, ierr)
      call MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m*n, m*n, ierr)
      call MatSetFromOptions(A, ierr)
      call MatSetUp(A, ierr)
      call MatGetOwnershipRange(A, Istart, Iend, ierr)
      do II = Istart, Iend - 1
         v = -1d0
         i = II/n
         j = II - i*n
         if (i > 0) then
            JJ = II - n
            call MatSetValues(A, one, [II], one, [JJ], [v], ADD_VALUES, ierr)
         end if
         if (i < m - 1) then
            JJ = II + n
            call MatSetValues(A, one, [II], one, [JJ], [v], ADD_VALUES, ierr)
         end if
         if (j > 0) then
            JJ = II - 1
            call MatSetValues(A, one, [II], one, [JJ], [v], ADD_VALUES, ierr)
         end if
         if (j < n - 1) then
            JJ = II + 1
            call MatSetValues(A, one, [II], one, [JJ], [v], ADD_VALUES, ierr)
         end if
         v = 4d0
         call MatSetValues(A, one, [II], one, [II], [v], ADD_VALUES, ierr)
      end do
      call MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr)

      ! Dense X with ncols random columns, matching A's row layout
      call MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,                &
     &                    m*n, ncols, PETSC_NULL_SCALAR_ARRAY, X, ierr)
      call PetscRandomCreate(PETSC_COMM_WORLD, rctx, ierr)
      call PetscRandomSetFromOptions(rctx, ierr)
      call MatSetRandom(X, rctx, ierr)
      call PetscRandomDestroy(rctx, ierr)
      call MatAssemblyBegin(X, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(X, MAT_FINAL_ASSEMBLY, ierr)

      ! Reusable reference vector for PCApply
      call MatCreateVecs(A, PETSC_NULL_VEC, y_ref, ierr)

      inv_types(1) = PFLAREINV_POWER
      inv_types(2) = PFLAREINV_ARNOLDI
      mf_flags(1) = PETSC_FALSE
      mf_flags(2) = PETSC_TRUE

      do i_type = 1, 2
         do i_mf = 1, 2

            if (inv_types(i_type) == PFLAREINV_POWER) then
               inv_label = "POWER"
            else
               inv_label = "ARNOLDI"
            end if
            if (mf_flags(i_mf)) then
               mf_label = "mf"
            else
               mf_label = "assembled"
            end if

            ! Fresh PC for each combination
            call PCCreate(PETSC_COMM_WORLD, pc, ierr)
            call PCSetOperators(pc, A, A, ierr)
            call PCSetType(pc, PCPFLAREINV, ierr)
            call PCPFLAREINVSetType(pc, inv_types(i_type), ierr)
            call PCPFLAREINVSetMatrixFree(pc, mf_flags(i_mf), ierr)
            call PCSetUp(pc, ierr)

            ! Y has same dense layout as X
            call MatDuplicate(X, MAT_DO_NOT_COPY_VALUES, Y, ierr)
            call MatAssemblyBegin(Y, MAT_FINAL_ASSEMBLY, ierr)
            call MatAssemblyEnd(Y, MAT_FINAL_ASSEMBLY, ierr)

            call PCMatApply(pc, X, Y, ierr)

            ! Check each column matches PCApply on the same input column
            do k = 0, ncols - 1
               call MatDenseGetColumnVecRead(X, k, x_col, ierr)
               call PCApply(pc, x_col, y_ref, ierr)
               call MatDenseGetColumnVecRead(Y, k, y_col, ierr)
               call VecNorm(y_ref, NORM_2, norm_ref, ierr)
               call VecAXPY(y_ref, -1d0, y_col, ierr)
               call VecNorm(y_ref, NORM_2, norm_diff, ierr)
               call MatDenseRestoreColumnVecRead(Y, k, y_col, ierr)
               call MatDenseRestoreColumnVecRead(X, k, x_col, ierr)

               if (norm_ref == 0d0) then
                  if (norm_diff > tol) then
                     print *, "FAIL ", trim(inv_label), " ", trim(mf_label),       &
     &                       " col=", k, " norm_diff=", norm_diff,                 &
     &                       " (norm_ref=0)"
                     error stop 1
                  end if
               else if (norm_diff / norm_ref > tol) then
                  print *, "FAIL ", trim(inv_label), " ", trim(mf_label),          &
     &                    " col=", k, " rel_err=", norm_diff / norm_ref
                  error stop 1
               end if
            end do

            call MatDestroy(Y, ierr)
            call PCDestroy(pc, ierr)
         end do
      end do

      call VecDestroy(y_ref, ierr)
      call MatDestroy(X, ierr)
      call MatDestroy(A, ierr)
      call PetscFinalize(ierr)
      end
