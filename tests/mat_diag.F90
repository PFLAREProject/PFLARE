#include <petsc/finclude/petscksp.h>
#include "finclude/pflare.h"
      use petscksp
      implicit none

      ! Test that the gmres polynomials can handle small solve of diagonal matrix
      ! We leave the polynomial order here as the default (which is 6), despite
      ! the fact that much lower polynomial order is an exact solve in this case
      ! This tests that the various gmres polynomial methods correctly
      ! identify we only need up to lower order

      PetscErrorCode :: ierr
      Mat :: A
      PetscInt :: m, n, nnzs
      PetscInt, parameter :: one = 1, two = 2, three = 3, zero = 0
      Vec :: x,b
      KSP :: ksp
      PC :: pc
      PetscBool :: flg
      KSPConvergedReason reason
      PetscRandom rctx

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)    
      ! Register the pflare types
      call PCRegister_PFLARE()      
      
      m      = 10
      n      = 10
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-m',m,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',n,flg,ierr)

      ! Create matrix
      call MatCreate(PETSC_COMM_WORLD,A,ierr)
      call MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,ierr)
      call MatSetFromOptions(A,ierr)
      nnzs = m;
      call MatSeqAIJSetPreallocation(A, nnzs, PETSC_NULL_INTEGER_ARRAY, ierr)
      call MatMPIAIJSetPreallocation(A, nnzs, PETSC_NULL_INTEGER_ARRAY, nnzs, PETSC_NULL_INTEGER_ARRAY, ierr)
      call MatSetUp(A,ierr)
      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)
      call MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr)

      call MatCreateVecs(A,b,x,ierr)
      call VecSet(x, 0d0, ierr)

      ! Random rhs
      call PetscRandomCreate(PETSC_COMM_WORLD, rctx, ierr)
      call PetscRandomSetFromOptions(rctx, ierr)
      call VecSetRandom(b, rctx, ierr)
      call PetscRandomDestroy(rctx, ierr)      

      ! ~~~~~~~~~~~~~~
      ! Set constant diagonal values in matrix
      ! In Newton form should only need a single root 
      ! (ie a 0th order polynomial) for an exact solve
      ! Starting with the identity, the inverse should also be the identity
      ! ~~~~~~~~~~~~~~
      call MatShift(A, 1d0, ierr)

      call KSPCreate(PETSC_COMM_WORLD,ksp,ierr)
      call KSPSetOperators(ksp,A,A,ierr)
      call KSPGetPC(ksp,pc,ierr)
      ! Set newton gmres polynomial as PC
      call PCSetType(pc, PCPFLAREINV, ierr)
      call PCPFLAREINVSetType(pc, PFLAREINV_NEWTON, ierr)
      call KSPSetPC(ksp, pc, ierr)       
      call KSPSetFromOptions(ksp,ierr)   
      call KSPSetUp(ksp,ierr)

      ! Do the solve
      call KSPSolve(ksp,b,x,ierr)
      call KSPGetConvergedReason(ksp, reason, ierr)
      if (reason%v < 0) then
         error stop 1
      end if    
      
      ! ~~~~~~~~~~~~~~
      ! Instead now set the diagonal to 1.5
      ! In Newton form should only need a single root 
      ! ~~~~~~~~~~~~~~
      call MatShift(A, 0.5d0, ierr)
      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)      
      call VecSet(x, 0d0, ierr)    

      ! Do another solve - this will automatically trigger the setup as the matrix
      ! has changed
      call KSPSolve(ksp,b,x,ierr)
      call KSPGetConvergedReason(ksp, reason, ierr)
      if (reason%v < 0) then
         error stop 1
      end if        
      
      ! ~~~~~~~~~~~~~~
      ! Instead now have two different constant values in the diagonal
      ! In Newton form should only need two roots
      ! (ie a 1st order polynomial) for an exact solve
      ! ~~~~~~~~~~~~~~
      ! Set one of the values to 2.5
      call MatSetValue(A, zero, zero, 2.5d0, INSERT_VALUES, ierr)
      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)      
      call VecSet(x, 0d0, ierr)    

      ! Do another solve - this will automatically trigger the setup as the matrix
      ! has changed
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