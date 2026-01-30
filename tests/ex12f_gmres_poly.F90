!
      program main
#include <petsc/finclude/petscksp.h>
      use petscksp
#include "finclude/pflare.h"      
      implicit none

!   Comparison between different forms of GMRES polynomials

      PetscErrorCode  ierr
      PetscInt m,n,mlocal,nlocal
      PetscBool  flg
      PetscReal      norm_power, norm_rhs, norm_arnoldi, norm_newton
      PetscReal :: norm_diff_one, norm_diff_two
      Vec              x,b,u, b_diff_type
      Mat              A, A_diff_type
      character*(128)  f
      PetscViewer      fd
      KSP              ksp
      PC               pc
      KSPConvergedReason reason
      PetscInt, parameter :: one=1
      MatType :: mtype, mtype_input

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

      ! Register the pflare types
      call PCRegister_PFLARE()

      call VecNorm(b,NORM_2,norm_rhs,ierr)

      ! ~~~~~~~~~~~~~
      ! Do a solve with the power basis
      ! ~~~~~~~~~~~~~
      call KSPCreate(PETSC_COMM_WORLD,ksp,ierr)
      call KSPSetOperators(ksp,A,A,ierr)
      call KSPGetPC(ksp, pc, ierr)       
      call PCSetType(pc, PCAIR, ierr)      
      call PCAIRSetInverseType(pc, PFLAREINV_POWER, ierr)
      call KSPSetPC(ksp, pc, ierr)
      call KSPSetFromOptions(ksp,ierr)

      call VecSet(x, 0d0, ierr)
      call KSPSolve(ksp,b,x,ierr)
      call KSPGetConvergedReason(ksp,reason,ierr)      
      if (reason%v < 0) then
         error stop 1
      end if
      ! Compute the residual
      call MatMult(A,x,u,ierr)
      call VecAXPY(u,-1d0,b,ierr)
      call VecNorm(u,NORM_2,norm_power,ierr)
      norm_power = norm_power/norm_rhs

      ! ~~~~~~~~~~~~~
      ! Now do a solve with the Arnoldi basis
      ! ~~~~~~~~~~~~~      
      call PCAIRSetInverseType(pc, PFLAREINV_ARNOLDI, ierr)

      call VecSet(x, 0d0, ierr)
      call KSPSolve(ksp,b,x,ierr)
      call KSPGetConvergedReason(ksp,reason,ierr)      
      if (reason%v < 0) then
         error stop 1
      end if
      ! Compute the residual
      call MatMult(A,x,u,ierr)
      call VecAXPY(u,-1d0,b,ierr)
      call VecNorm(u,NORM_2,norm_arnoldi,ierr)
      norm_arnoldi = norm_arnoldi/norm_rhs

      ! ~~~~~~~~~~~~~
      ! Now do a solve with the Newton basis
      ! ~~~~~~~~~~~~~         
      call PCAIRSetInverseType(pc, PFLAREINV_NEWTON, ierr)   

      call VecSet(x, 0d0, ierr)
      call KSPSolve(ksp,b,x,ierr)
      call KSPGetConvergedReason(ksp,reason,ierr)      
      if (reason%v < 0) then
         error stop 1
      end if
      ! Compute the residual
      call MatMult(A,x,u,ierr)
      call VecAXPY(u,-1d0,b,ierr)
      call VecNorm(u,NORM_2,norm_newton,ierr)
      norm_newton = norm_newton/norm_rhs
      call KSPDestroy(ksp,ierr)

      ! ~~~~~~~~~~~~~
      ! Now check all the residuals are the same
      ! For low order polynomials on the diagonally dominant
      ! A_ff on each level they should be basically identical and hence
      ! we should have almost no difference in the resulting residual
      ! ~~~~~~~~~~~~~
      norm_diff_one = abs(norm_power - norm_newton)/norm_newton
      if (norm_diff_one > 1e-9) then
         print *, "Residuals differ between polynomial bases!", norm_diff_one
         print *, "Power basis residual:   ", norm_power
         print *, "Newton basis residual:  ", norm_newton
         error stop 1
      end if
      norm_diff_two = abs(norm_arnoldi - norm_power)/norm_power
      if (norm_diff_two > 1e-9) then
         print *, "Residuals differ between polynomial bases!", norm_diff_two
         print *, "Arnoldi basis residual: ", norm_arnoldi
         print *, "Newton basis residual:  ", norm_newton
         error stop 1
      end if      

      call VecDestroy(b,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(u,ierr)
      call MatDestroy(A,ierr)

      call PetscFinalize(ierr)

      end