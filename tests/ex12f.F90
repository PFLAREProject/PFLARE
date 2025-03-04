!
      program main
#include <petsc/finclude/petscksp.h>
      use petscksp
#include "finclude/pflare.h"      
      implicit none

!   **** Modified example from PETSc that uses AIR as a PC
!
!  This example is the Fortran version of ex6.c.  The program reads a PETSc matrix
!  and vector from a file and solves a linear system.  Input arguments are:
!        -f <input_file> : file to load.  For example see $PETSC_DIR/share/petsc/datafiles/matrices
!

      PetscErrorCode  ierr
      PetscInt its,m,n,mlocal,nlocal
      PetscBool  flg, second_solve
      PetscScalar      none
      PetscReal        norm
      Vec              x,b,u, b_diff_type
      Mat              A, A_diff_type
      character*(128)  f
      PetscViewer      fd
      KSP              ksp
      PC               pc
      KSPConvergedReason reason
      PetscInt, parameter :: one=1
      MatType :: mtype, mtype_input
      PetscLogStage :: gpu_copy

      none = -1d0
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif

! Read in matrix and RHS
      call PetscOptionsGetString(PETSC_NULL_OPTIONS,                    &
     &        PETSC_NULL_CHARACTER,'-f',f,flg,ierr)
      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,f,FILE_MODE_READ,     &
     &     fd,ierr)
     second_solve = PETSC_FALSE
     call PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-second_solve',second_solve,flg,ierr)  

      call MatCreate(PETSC_COMM_WORLD,A,ierr)
      call MatLoad(A,fd,ierr)

      ! Get information about matrix
      call MatGetSize(A,m,n,ierr)
      call MatGetLocalSize(A,mlocal,nlocal,ierr)

      call VecCreate(PETSC_COMM_WORLD,b,ierr)
      call VecLoad(b,fd,ierr)
      call PetscViewerDestroy(fd,ierr)

      ! Test and see if the user wants us to use a different matrix type
      ! with -mat_type on the command line
      ! This lets us easily test our cpu and kokkos versions through our CI
      call MatCreateFromOptions(PETSC_COMM_WORLD,PETSC_NULL_CHARACTER,&
               one,mlocal,nlocal,m,n,A_diff_type,ierr)
      call MatAssemblyBegin(A_diff_type,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A_diff_type,MAT_FINAL_ASSEMBLY,ierr)               
      
      call MatGetType(A, mtype, ierr)
      call MatGetType(A_diff_type, mtype_input, ierr)

      if (mtype /= mtype_input) then
         ! Doesn't seem like there is a converter to kokkos
         ! So instead we just copy into the empty A_diff_type
         ! This will be slow as its not preallocated, but this is just for testing
         call MatCopy(A, A_diff_type, DIFFERENT_NONZERO_PATTERN, ierr)
         call MatDestroy(A,ierr)
         A = A_diff_type

         ! Mat and vec types have to match
         call VecCreateFromOptions(PETSC_COMM_WORLD,PETSC_NULL_CHARACTER, & 
                  one,nlocal,n,b_diff_type,ierr)
         call VecCopy(b,b_diff_type,ierr)
         call VecDestroy(b,ierr)
         b = b_diff_type
 
      else
         call MatDestroy(A_diff_type,ierr)
      end if

! Set up solution
      call VecDuplicate(b,x,ierr)
      call VecDuplicate(b,u,ierr)

! Solve system
      call KSPCreate(PETSC_COMM_WORLD,ksp,ierr)
      call KSPSetOperators(ksp,A,A,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!         Let's use AIRG as our PC
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -     

      call KSPGetPC(ksp, pc, ierr) 
      ! Register the pflare types
      call PCRegister_PFLARE()
      call PCSetType(pc, PCAIR, ierr)
      call KSPSetPC(ksp, pc, ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call PetscLogStageRegister("GPU copy stage - triggered by a prelim KSPSolve",gpu_copy, ierr)
      call PetscLogStagePush(gpu_copy, ierr)
      call KSPSetFromOptions(ksp,ierr)
      call KSPSolve(ksp,b,x,ierr)
      call PetscLogStagePop()

      if (second_solve) then
         call VecSet(x, 1d0, ierr)
         call KSPSolve(ksp,b,x,ierr)
      end if

! Show result
      call MatMult(A,x,u,ierr)
      call VecAXPY(u,none,b,ierr)
      call VecNorm(u,NORM_2,norm,ierr)
      call KSPGetIterationNumber(ksp,its,ierr)
      call KSPGetConvergedReason(ksp,reason,ierr)
      write(6,101) norm,its
 101  format('Residual norm ',1pe9.2,' iterations ',i5)

      call KSPDestroy(ksp,ierr)
      call VecDestroy(b,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(u,ierr)
      call MatDestroy(A,ierr)

      call PetscFinalize(ierr)

      if (reason < 0) then
         error stop 1
      end if
      end