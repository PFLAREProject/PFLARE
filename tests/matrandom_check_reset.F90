#include <petsc/finclude/petscksp.h>
#include "finclude/pflare.h"
      use petscksp
      implicit none

!   Test changing parameters after a solve triggers a reset

      PetscErrorCode :: ierr
      Mat :: A
      PetscInt :: m, n, nnzs
      PetscInt, parameter :: one = 1, two = 2, three = 3
      Vec :: x,b
      KSP :: ksp
      PC :: pc
      PetscBool :: flg, check_airg, check
      KSPConvergedReason reason

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)    
      ! Register the pflare types
      call PCRegister_PFLARE()      
      
      m      = 5
      n      = 5
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-m',m,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',n,flg,ierr)
      check_airg = PETSC_TRUE
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER, &
               '-check_airg', check,flg,ierr)
      if (flg) check_airg = check                       

      ! Create matrix
      call MatCreate(PETSC_COMM_WORLD,A,ierr)
      call MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,ierr)
      call MatSetFromOptions(A,ierr)
      nnzs = m * 2;
      call MatSeqAIJSetPreallocation(A, nnzs, PETSC_NULL_INTEGER_ARRAY, ierr)
      call MatMPIAIJSetPreallocation(A, nnzs, PETSC_NULL_INTEGER_ARRAY, nnzs, PETSC_NULL_INTEGER_ARRAY, ierr)
      call MatSetUp(A,ierr)

      call MatCreateVecs(A,b,x,ierr)
      call VecSet(x, 0d0, ierr)
      call VecSet(b, 2d0, ierr)      

      ! Set random values in matrix
      call MatSetRandom(A, PETSC_NULL_RANDOM, ierr)
      call MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)
      call MatDiagonalSet(A, b, ADD_VALUES, ierr)  ! make it diagonally dominant

      call KSPCreate(PETSC_COMM_WORLD,ksp,ierr)
      call KSPSetOperators(ksp,A,A,ierr)
      call KSPGetPC(ksp,pc,ierr)
      ! Set AIRG as pc
      if (check_airg) then
         call PCSetType(pc, PCAIR, ierr)
         ! Set an initial option to test it persists after reset
         call PCAIRSetCoarseEqLimit(pc, 9, ierr)
      else
         call PCSetType(pc, PCPFLAREINV, ierr)
         call PCPFLAREINVSetOrder(pc, 2, ierr)
      end if

      call KSPSetPC(ksp, pc, ierr)       
      call KSPSetFromOptions(ksp,ierr)   
      call KSPSetUp(ksp,ierr)

      ! Do the solve
      call KSPSolve(ksp,b,x,ierr)
      call KSPGetConvergedReason(ksp, reason, ierr)
      if (reason%v < 0) then
         error stop 1
      end if      

      ! Do a second solve with some changed parameters to test reset
      if (check_airg) then
         call PCAIRSetMaxLevels(pc, 2, ierr)
         call PCAIRSetStrongThreshold(pc, 0.4d0, ierr)      
      else
         call PCPFLAREINVSetMatrixFree(pc, PETSC_TRUE, ierr)
      end if
      call VecSet(x, 0d0, ierr)
      call KSPSolve(ksp,b,x,ierr)
      call KSPGetConvergedReason(ksp, reason, ierr)
      if (reason%v < 0) then
         error stop 1
      end if          

      call MatDestroy(A, ierr)
      call VecDestroy(b, ierr)
      call VecDestroy(x, ierr)
      call KSPDestroy(ksp, ierr)
      call PetscFinalize(ierr)
      end