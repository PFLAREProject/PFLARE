!
!  Tests PCAIRSetReuseAmount / PCAIRGetReuseAmount.
!
!  For each reuse_amount value (1, 2, 3) two linear solves are performed with
!  reuse_sparsity enabled.  The matrix is perturbed between the solves to force
!  a new PCSetUp call; convergence of both solves is verified.
!  A round-trip get/set check is also performed.
!

      program main
      use petscksp
#include "petsc/finclude/petscksp.h"
#include "finclude/pflare.h"
      use pflare
      implicit none

      Vec            x, u, b
      Mat            A
      KSP            ksp
      PC             pc
      PetscInt       i, j, II, JJ, m, n
      PetscInt       Istart, Iend, one
      PetscErrorCode ierr
      PetscBool      flg
      PetscScalar    v
      PetscInt       amount, got_amount
      PetscInt       amount_start, amount_end
      KSPConvergedReason reason

      call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
      if (ierr .ne. 0) then
         print *, 'Unable to initialize PETSc'
         stop
      end if

      m   = 10
      n   = 10
      one = 1
      amount_start = 1
      amount_end   = 3
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-m', m, flg, ierr)
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-n', n, flg, ierr)
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, &
               '-test_reuse_amount_start', amount_start, flg, ierr)
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, &
               '-test_reuse_amount_end', amount_end, flg, ierr)

      ! Build 2-D five-point Laplacian
      call MatCreate(PETSC_COMM_WORLD, A, ierr)
      call MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m*n, m*n, ierr)
      call MatSetFromOptions(A, ierr)
      call MatSetUp(A, ierr)

      call MatGetOwnershipRange(A, Istart, Iend, ierr)
      do II = Istart, Iend-1
         v = -1d0
         i = II / n
         j = II - i*n
         if (i .gt. 0) then
            JJ = II - n
            call MatSetValues(A, one, [II], one, [JJ], [v], ADD_VALUES, ierr)
         end if
         if (i .lt. m-1) then
            JJ = II + n
            call MatSetValues(A, one, [II], one, [JJ], [v], ADD_VALUES, ierr)
         end if
         if (j .gt. 0) then
            JJ = II - 1
            call MatSetValues(A, one, [II], one, [JJ], [v], ADD_VALUES, ierr)
         end if
         if (j .lt. n-1) then
            JJ = II + 1
            call MatSetValues(A, one, [II], one, [JJ], [v], ADD_VALUES, ierr)
         end if
         v = 4d0
         call MatSetValues(A, one, [II], one, [II], [v], ADD_VALUES, ierr)
      end do
      call MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr)

      call VecCreate(PETSC_COMM_WORLD, u, ierr)
      call VecSetSizes(u, PETSC_DECIDE, m*n, ierr)
      call VecSetFromOptions(u, ierr)
      call VecDuplicate(u, b, ierr)
      call VecDuplicate(u, x, ierr)

      call PCRegister_PFLARE()

      ! -----------------------------------------------------------------------
      ! Round-trip get/set check (before any solve)
      ! -----------------------------------------------------------------------
      call KSPCreate(PETSC_COMM_WORLD, ksp, ierr)
      call KSPSetFromOptions(ksp, ierr)
      call KSPGetPC(ksp, pc, ierr)

      do amount = amount_start, amount_end
         call PCAIRSetReuseAmount(pc, amount, ierr)
         call PCAIRGetReuseAmount(pc, got_amount, ierr)
         if (got_amount .ne. amount) then
            print *, 'FAIL: round-trip reuse_amount mismatch: set', amount, 'got', got_amount
            error stop 1
         end if
      end do

      ! -----------------------------------------------------------------------
      ! For each reuse_amount, do two solves with reuse_sparsity and verify
      ! both converge.
      ! -----------------------------------------------------------------------
      do amount = amount_start, amount_end

         ! Reset matrix to original state by destroying and rebuilding
         call KSPDestroy(ksp, ierr)
         call MatDestroy(A, ierr)

         call MatCreate(PETSC_COMM_WORLD, A, ierr)
         call MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m*n, m*n, ierr)
         call MatSetFromOptions(A, ierr)
         call MatSetUp(A, ierr)

         call MatGetOwnershipRange(A, Istart, Iend, ierr)
         do II = Istart, Iend-1
            v = -1d0
            i = II / n
            j = II - i*n
            if (i .gt. 0) then
               JJ = II - n
               call MatSetValues(A, one, [II], one, [JJ], [v], ADD_VALUES, ierr)
            end if
            if (i .lt. m-1) then
               JJ = II + n
               call MatSetValues(A, one, [II], one, [JJ], [v], ADD_VALUES, ierr)
            end if
            if (j .gt. 0) then
               JJ = II - 1
               call MatSetValues(A, one, [II], one, [JJ], [v], ADD_VALUES, ierr)
            end if
            if (j .lt. n-1) then
               JJ = II + 1
               call MatSetValues(A, one, [II], one, [JJ], [v], ADD_VALUES, ierr)
            end if
            v = 4d0
            call MatSetValues(A, one, [II], one, [II], [v], ADD_VALUES, ierr)
         end do
         call MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr)
         call MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr)

         call KSPCreate(PETSC_COMM_WORLD, ksp, ierr)
         call KSPSetFromOptions(ksp, ierr)
         call KSPGetPC(ksp, pc, ierr)

         call PCAIRSetReuseSparsity(pc, PETSC_TRUE, ierr)
         call PCAIRSetReuseAmount(pc, amount, ierr)

         ! Solve 1
         call KSPSetOperators(ksp, A, A, ierr)
         call KSPSetReusePreconditioner(ksp, PETSC_FALSE, ierr)
         call KSPSetInitialGuessNonzero(ksp, PETSC_FALSE, ierr)
         v = 1d0
         call VecSet(u, v, ierr)
         call MatMult(A, u, b, ierr)
         call KSPSolve(ksp, b, x, ierr)

         call KSPGetConvergedReason(ksp, reason, ierr)
         if (reason%v .le. 0) then
            print *, 'FAIL: solve 1 did not converge for reuse_amount =', amount, &
                     ' (reason', reason%v, ')'
            error stop 1
         end if

         ! Perturb matrix to force a new setup
         call MatGetOwnershipRange(A, Istart, Iend, ierr)
         do II = Istart, Iend-1
            v = 0.5d0
            call MatSetValues(A, one, [II], one, [II], [v], ADD_VALUES, ierr)
         end do
         call MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr)
         call MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr)

         ! Solve 2 with reuse
         call KSPSetOperators(ksp, A, A, ierr)
         call KSPSetReusePreconditioner(ksp, PETSC_FALSE, ierr)
         call KSPSetInitialGuessNonzero(ksp, PETSC_FALSE, ierr)
         v = 1d0
         call VecSet(u, v, ierr)
         call MatMult(A, u, b, ierr)
         call KSPSolve(ksp, b, x, ierr)

         call KSPGetConvergedReason(ksp, reason, ierr)
         if (reason%v .le. 0) then
            print *, 'FAIL: solve 2 did not converge for reuse_amount =', amount, &
                     ' (reason', reason%v, ')'
            error stop 1
         end if

      end do

      call KSPDestroy(ksp, ierr)
      call VecDestroy(u, ierr)
      call VecDestroy(x, ierr)
      call VecDestroy(b, ierr)
      call MatDestroy(A, ierr)

      call PetscFinalize(ierr)
      end
