!
!  Description: This example demonstrates repeated linear solves as
!  well as the use of different preconditioner and linear system
!  matrices.  This example also illustrates how to save PETSc objects
!  in common blocks.
!
!/*T
!  Concepts: KSP^repeatedly solving linear systems;
!  Concepts: KSP^different matrices for linear system and preconditioner;
!  Processors: n
!T*/
!

module solve_module

   use petscksp
#include "petsc/finclude/petscksp.h"   

   implicit none
  
   type real_coeffs
      ! Must be set to null to start
      PetscReal, dimension(:, :), pointer :: coeffs => null()
   end type real_coeffs   

contains 

   ! -----------------------------------------------------------------------
   !
   subroutine solve1(ksp,A,x,b,u,count,nsteps,coeffs_levels,ierr)
#include <petsc/finclude/petscksp.h>
      use petscksp
#include "finclude/pflare.h"  
      use pflare
      
   !
   !   solve1 - This routine is used for repeated linear system solves.
   !
   !      A - linear system matrix

      PetscScalar  v,val
      PetscInt II,Istart,Iend
      PetscInt count,nsteps,one,start, start_plus_one
      PetscErrorCode ierr
      PetscInt num_levels, petsc_level
      Mat     A
      KSP     ksp
      PC pc
      Vec     x,b,u
      KSPConvergedReason reason
      type(real_coeffs), dimension(:), allocatable :: coeffs_levels

      ! Use common block to retain stuff between successive subroutine calls
      PetscMPIInt      rank
      PetscBool        pflag
      PetscReal norm_first_solve, norm_third_solve
      common /my_data/ norm_first_solve,rank, pflag

      one = 1
      call KSPSetInitialGuessNonzero(ksp,PETSC_FALSE,ierr)
      call MatGetOwnershipRange(A, start, start_plus_one, ierr)
      if (start == 0) rank = 0      
      ! Explicitly tell it not to reuse the preconditioner
      ! This forces it to regenerate it every time with SAME_NON_ZERO
      call KSPSetReusePreconditioner(ksp,PETSC_FALSE,ierr)

      ! ~~~~~~~~~~~~~~~~~~

      ! Change the operator A
      ! On the first solve let's just actually solve a more diagonally dominant
      ! version of the matrix
      if (count .eq. 1) then

         ! Alter the matrix A a bit
         call MatGetOwnershipRange(A,Istart,Iend,ierr)
         do II=Istart,Iend-1
            v = 2
            call MatSetValues(A,one,[II],one,[II],[v],ADD_VALUES,ierr)
         end do

         call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
         call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)   

      ! On the second solve, we add 0.1 to the diagonal
      else if (count .eq. 2) then

         ! Alter the matrix A a bit
         call MatGetOwnershipRange(A,Istart,Iend,ierr)
         do II=Istart,Iend-1
            v = 0.1
            call MatSetValues(A,one,[II],one,[II],[v],ADD_VALUES,ierr)
         end do

         call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
         call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)   
      
      ! On the third solve, we minus 0.1 from the diagonal
      ! Making it the same linear system we solved in solve 1
      else if (count == 3) then

         ! Alter the matrix A a bit
         call MatGetOwnershipRange(A,Istart,Iend,ierr)
         do II=Istart,Iend-1
            v = -0.1
            call MatSetValues(A,one,[II],one,[II],[v],ADD_VALUES,ierr)
         end do

         call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
         call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)         
      endif     

      call KSPSetOperators(ksp,A,A,ierr)

      ! ~~~~~~~~~~~~~~~~~~

      ! Set the exact solution; compute the right-hand-side vector
      val = 1d0
      call VecSet(u,val,ierr)
      call MatMult(A,u,b,ierr)

      ! We have the first and third solve with the same linear system, but the second is different
      !
      ! We want to trigger a rebuild with same_nonzero_pattern every 
      ! time and reuse the sparsity
      !
      ! The first solve will generate poly coefficients (which we store)
      ! The second solve will generate poly coefficients for the changed operator
      ! but reusing the same sparsity in the hierarchy
      ! The third solve will restore the saved poly coefficients, reuse them and
      ! reuse the same sparsity in the hierarchy
      ! Hence in the third solve we should have exactly the same residual as in the first
      !
      ! This is like if we have an outer loop (e.g., eigenvalue solve) where we have to 
      ! repeatedly solve the same linear systems in the inner loop but don't have the memory to store
      ! the hierarchies for every inner solve
      !
      ! Here we are only storing the inv Aff coefficients (COEFFS_INV_AFF)
      ! and the coarse grid solver coefficients (COEFFS_INV_COARSE)
      ! If strong R threshold /= 0d0, you will also want to store the 
      ! inv Aff dropped coefficients (COEFFS_INV_AFF_DROPPED)
      ! If any C smooths are occuring, you will also want to store inv Acc coefficients (COEFFS_INV_COARSE)

      call KSPGetPC(ksp,pc,ierr)

      ! ~~~~~~~~~~~~~~~~~~

      ! In the first solve
      if (count == 1) then
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         !         Let's use AIRG as our PC
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -     
         call PCSetType(pc, PCAIR, ierr)
         ! Always reuse the sparsity
         call PCAIRSetReuseSparsity(pc, PETSC_TRUE, ierr)
         call KSPSetPC(ksp, pc, ierr) 
         call KSPSetFromOptions(ksp,ierr)         

      ! In the third solve
      else if (count == 3) then

         ! We need to know how many levels we have in the mg
         call PCAIRGetNumLevels(pc, num_levels, ierr)    
         ! On the third solve, restore the gmres polynomial coefficients from the first solve

         ! Loop over all levels except the coarse
         do petsc_level = num_levels - 1, 1, -1
            ! Set the inverse Aff coefficients
            call PCAIRSetPolyCoeffs(pc, petsc_level, COEFFS_INV_AFF, coeffs_levels(petsc_level+1)%coeffs, ierr) 
         end do 
         ! Get the coarse grid coefficients
         call PCAIRSetPolyCoeffs(pc, petsc_level, COEFFS_INV_COARSE, coeffs_levels(1)%coeffs, ierr)   

         ! Tell pcair to not regenerate the gmres polynomial coeffs
         call PCAIRSetReusePolyCoeffs(pc, PETSC_TRUE, ierr)   
         
      end if  
      
      ! ~~~~~~~~~~~~~~~~~~

      ! Solve linear system
      call KSPSolve(ksp,b,x,ierr)

      ! Compute the (non-preconditioned) residual
      call MatResidual(A, b, x, u, ierr)
      ! Store it after the first solve
      if (count == 1) then
         call VecNorm(u, NORM_2, norm_first_solve, ierr)
      ! and the third
      else if (count == 3) then
         call VecNorm(u, NORM_2, norm_third_solve, ierr)
      end if

      ! ~~~~~~~~~~~~~~~~~~

      ! We need to know how many levels we have in the mg
      call PCAIRGetNumLevels(pc, num_levels, ierr)
      ! Store the polynomial coefficients from the first solve
      if (count == 1) then

         allocate(coeffs_levels(num_levels))

         ! Loop over all levels except the coarse
         do petsc_level = num_levels - 1, 1, -1
            ! Get the inverse Aff coefficients
            ! Note that the fortran interface passes out a *copy* of the coefficeints in the pcair object
            ! as a pointer array
            ! The C inteface however just returns a pointer to the coefficients within the pcair object 
            ! so if you want to save/restore them later in C you will have to copy the values
            ! The C interface therefore also passes out the sizes of the coefficients array
            call PCAIRGetPolyCoeffs(pc, petsc_level, COEFFS_INV_AFF, coeffs_levels(petsc_level+1)%coeffs, ierr) 
         end do 
         ! Get the coarse grid coefficients
         call PCAIRGetPolyCoeffs(pc, petsc_level, COEFFS_INV_COARSE, coeffs_levels(1)%coeffs, ierr)   
      end if

      ! ~~~~~~~~~~~~~~~~~~

      call KSPGetConvergedReason(ksp, reason, ierr)
      if (reason%v > 0) then
      else
         error stop 1
      end if

      ! On the third solve, check the residuals of the first and third solve
      ! are the same
      if (count .eq. nsteps) then
         if (rank .eq. 0) then
            if (abs(norm_first_solve - norm_third_solve)/norm_first_solve < 1e-8) then
               print *, "Residuals OK"
            else
               print *, "Residuals WRONG"
               error stop 1
            end if
         end if
      end if

   end subroutine
  
  end module solve_module

  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      program main
      use solve_module
#include <petsc/finclude/petscksp.h>   
      use petscksp
#include "finclude/pflare.h"
      use pflare
      implicit none

     

!  Variables:
!
!  A       - matrix that defines linear system
!  ksp    - KSP context
!  ksp     - KSP context
!  x, b, u - approx solution, RHS, exact solution vectors
!
      Vec     x,u,b
      Mat     A
      KSP    ksp
      PetscInt i,j,II,JJ,m,n
      PetscInt Istart,Iend
      PetscInt nsteps,one
      PetscErrorCode ierr
      PetscBool  flg
      PetscScalar  v
      type(real_coeffs), dimension(:), allocatable :: coeffs_levels


      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      m      = 5
      n      = 5
      nsteps = 3
      one    = 1
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-m',m,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',n,flg,ierr)

!  Create parallel matrix, specifying only its global dimensions.
!  When using MatCreate(), the matrix format can be specified at
!  runtime. Also, the parallel partitioning of the matrix is
!  determined by PETSc at runtime.

      call MatCreate(PETSC_COMM_WORLD,A,ierr)
      call MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,ierr)
      call MatSetFromOptions(A,ierr)
      call MatSetUp(A,ierr)

!  The matrix is partitioned by contiguous chunks of rows across the
!  processors.  Determine which rows of the matrix are locally owned.

      call MatGetOwnershipRange(A,Istart,Iend,ierr)

!  Set matrix elements.
!   - Each processor needs to insert only elements that it owns
!     locally (but any non-local elements will be sent to the
!     appropriate processor during matrix assembly).
!   - Always specify global rows and columns of matrix entries.

      do 10, II=Istart,Iend-1
        v = -1d0
        i = II/n
        j = II - i*n
        if (i.gt.0) then
          JJ = II - n
          call MatSetValues(A,one,[II],one,[JJ],[v],ADD_VALUES,ierr)
        endif
        if (i.lt.m-1) then
          JJ = II + n
          call MatSetValues(A,one,[II],one,[JJ],[v],ADD_VALUES,ierr)
        endif
        if (j.gt.0) then
          JJ = II - 1
          call MatSetValues(A,one,[II],one,[JJ],[v],ADD_VALUES,ierr)
        endif
        if (j.lt.n-1) then
          JJ = II + 1
          call MatSetValues(A,one,[II],one,[JJ],[v],ADD_VALUES,ierr)
        endif
        v = 4.0
        call  MatSetValues(A,one,[II],one,[II],[v],ADD_VALUES,ierr)
 10   continue

!  Assemble matrix, using the 2-step process:
!       MatAssemblyBegin(), MatAssemblyEnd()
!  Computations can be done while messages are in transition
!  by placing code between these two statements.

      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)

!  Create parallel vectors.
!   - When using VecCreate(), the parallel partitioning of the vector
!     is determined by PETSc at runtime.
!   - Note: We form 1 vector from scratch and then duplicate as needed.

      call VecCreate(PETSC_COMM_WORLD,u,ierr)
      call VecSetSizes(u,PETSC_DECIDE,m*n,ierr)
      call VecSetFromOptions(u,ierr)
      call VecDuplicate(u,b,ierr)
      call VecDuplicate(b,x,ierr)

      ! Register the pflare types
      call PCRegister_PFLARE()

!  Create linear solver context

      call KSPCreate(PETSC_COMM_WORLD,ksp,ierr)

!  Set runtime options (e.g., -ksp_type <type> -pc_type <type>)

      call KSPSetFromOptions(ksp,ierr)

!  Solve several linear systems in succession

      do 100 i=1,nsteps
         call solve1(ksp,A,x,b,u,i,nsteps, coeffs_levels,ierr)
 100  continue

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.

      call VecDestroy(u,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(b,ierr)
      call MatDestroy(A,ierr)
      call KSPDestroy(ksp,ierr)

      call PetscFinalize(ierr)
      end


