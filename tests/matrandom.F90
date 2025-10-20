#include <petsc/finclude/petscksp.h>
#include "finclude/pflare.h"
      use petscksp
      implicit none

!   Test fortran interface of pflare with small solve of random matrix

      PetscErrorCode :: ierr
      Mat :: A
      PetscInt :: m, n, nnzs
      PetscInt, parameter :: one = 1, two = 2, three = 3
      Vec :: x,b
      KSP :: ksp
      PC :: pc
      PetscBool :: flg
      KSPConvergedReason reason

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)    
      ! Register the pflare types
      call PCRegister_PFLARE()      
      
      m      = 5
      n      = 5
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-m',m,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',n,flg,ierr)

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
      call PCSetType(pc, PCAIR, ierr)
      call KSPSetPC(ksp, pc, ierr)       
      call KSPSetFromOptions(ksp,ierr)   
      call KSPSetUp(ksp,ierr)

      ! Do the solve
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