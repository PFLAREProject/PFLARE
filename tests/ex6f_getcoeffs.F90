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
   subroutine test_solve_getcoeffs(ksp, A, x, b, u, count, nsteps, coeffs_levels, coeffs_pflareinv, ierr)
#include <petsc/finclude/petscksp.h>
      use petscksp
#include "finclude/pflare.h"
      use pflare
      use pcpflareinv_interfaces

   !  Runs three linear solves modifying the matrix between each, saving
   !  polynomial coefficients from solve 1 and restoring them on solve 3.
   !  Works for both PCAIR (multi-level) and PCPFLAREINV (single-level).
   !  The PC type is determined by how the KSP was configured from the
   !  command line (-pc_type air or -pc_type pflareinv).

      PetscScalar  v, val
      PetscInt II, Istart, Iend
      PetscInt count, nsteps, one, start, start_plus_one
      PetscErrorCode ierr
      PetscInt num_levels, petsc_level
      Mat A
      KSP ksp
      PC pc
      Vec x, b, u
      KSPConvergedReason reason
      type(real_coeffs), dimension(:), allocatable :: coeffs_levels
      PetscReal, dimension(:,:), pointer :: coeffs_pflareinv
      PCType pctype

      PetscMPIInt rank
      PetscReal norm_first, norm_third
      common /ksp_coeffs_data/ norm_first, rank

      one = 1
      call KSPSetInitialGuessNonzero(ksp, PETSC_FALSE, ierr)
      call MatGetOwnershipRange(A, start, start_plus_one, ierr)
      if (start == 0) rank = 0
      call KSPSetReusePreconditioner(ksp, PETSC_FALSE, ierr)

      ! Modify the operator between solves so that solve 3 reproduces solve 1
      if (count == 1) then
         call MatGetOwnershipRange(A, Istart, Iend, ierr)
         do II = Istart, Iend-1
            v = 2
            call MatSetValues(A, one, [II], one, [II], [v], ADD_VALUES, ierr)
         end do
         call MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr)
         call MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr)
      else if (count == 2) then
         call MatGetOwnershipRange(A, Istart, Iend, ierr)
         do II = Istart, Iend-1
            v = 0.1
            call MatSetValues(A, one, [II], one, [II], [v], ADD_VALUES, ierr)
         end do
         call MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr)
         call MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr)
      else if (count == 3) then
         call MatGetOwnershipRange(A, Istart, Iend, ierr)
         do II = Istart, Iend-1
            v = -0.1
            call MatSetValues(A, one, [II], one, [II], [v], ADD_VALUES, ierr)
         end do
         call MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr)
         call MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr)
      end if

      call KSPSetOperators(ksp, A, A, ierr)

      val = 1d0
      call VecSet(u, val, ierr)
      call MatMult(A, u, b, ierr)

      ! Detect which PC type is configured (set by the user via -pc_type)
      call KSPGetPC(ksp, pc, ierr)
      call PCGetType(pc, pctype, ierr)

      ! PC-type-specific first-call setup
      if (count == 1 .and. pctype == PCAIR) then
         call PCAIRSetReuseSparsity(pc, PETSC_TRUE, ierr)
      end if

      ! On solve 3, restore the saved coefficients
      if (count == 3) then
         ! Need to set the coefficients on each level for PCAIR
         if (pctype == PCAIR) then
            call PCAIRGetNumLevels(pc, num_levels, ierr)
            do petsc_level = num_levels-1, 1, -1
               call PCAIRSetPolyCoeffs(pc, petsc_level, COEFFS_INV_AFF, coeffs_levels(petsc_level+1)%coeffs, ierr)
            end do
            call PCAIRSetPolyCoeffs(pc, petsc_level, COEFFS_INV_COARSE, coeffs_levels(1)%coeffs, ierr)
            call PCAIRSetReusePolyCoeffs(pc, PETSC_TRUE, ierr)

         ! If PCPFLAREINV only one level
         else if (pctype == PCPFLAREINV) then
            call PCPFLAREINVSetPolyCoeffs(pc, coeffs_pflareinv, ierr)
            call PCPFLAREINVSetReusePolyCoeffs(pc, PETSC_TRUE, ierr)
         end if
      end if

      call KSPSolve(ksp, b, x, ierr)

      call MatResidual(A, b, x, u, ierr)
      if (count == 1) call VecNorm(u, NORM_2, norm_first, ierr)
      if (count == 3) call VecNorm(u, NORM_2, norm_third, ierr)

      ! After solve 1, save the polynomial coefficients
      if (count == 1) then
         ! Multiple levels
         if (pctype == PCAIR) then
            call PCAIRGetNumLevels(pc, num_levels, ierr)
            allocate(coeffs_levels(num_levels))
            do petsc_level = num_levels-1, 1, -1
               call PCAIRGetPolyCoeffs(pc, petsc_level, COEFFS_INV_AFF, coeffs_levels(petsc_level+1)%coeffs, ierr)
            end do
            call PCAIRGetPolyCoeffs(pc, petsc_level, COEFFS_INV_COARSE, coeffs_levels(1)%coeffs, ierr)

         ! Single level
         else if (pctype == PCPFLAREINV) then
            call PCPFLAREINVGetPolyCoeffs(pc, coeffs_pflareinv, ierr)
         end if
      end if

      call KSPGetConvergedReason(ksp, reason, ierr)
      if (reason%v > 0) then
      else
         error stop 1
      end if

      if (count == nsteps) then
         if (rank == 0) then
            if (abs(norm_first - norm_third) / norm_first > 1e-8) then
               print *, "Residuals WRONG"
               error stop 1
            end if
         end if
      end if

   end subroutine test_solve_getcoeffs
  
  end module solve_module

  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      program main
      use solve_module
#include <petsc/finclude/petscksp.h>   
      use petscksp
#include "finclude/pflare.h"
      use pflare
      use pcpflareinv_interfaces
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
      PC     pc
      PetscInt i,j,II,JJ,m,n
      PetscInt Istart,Iend
      PetscInt nsteps,one
      PetscErrorCode ierr
      PetscBool  flg
      PetscScalar  v
      type(real_coeffs), dimension(:), allocatable :: coeffs_levels
      PetscReal, dimension(:,:), pointer :: coeffs_pflareinv => null()
      PCPFLAREINVType :: pflare_type
      PetscBool :: no_power, skip
      PCType pctype


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

      ! Read no_power flag (disables power basis for Intel MPI CI)
      no_power = PETSC_FALSE
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, &
               '-no_power', no_power, flg, ierr)

!  Create KSP; the PC type is determined by -pc_type on the command line

      call KSPCreate(PETSC_COMM_WORLD, ksp, ierr)
      call KSPSetFromOptions(ksp, ierr)

!  If -no_power is set and the user configured PCPFLAREINV with power type, skip

      skip = PETSC_FALSE
      if (no_power) then
         call KSPGetPC(ksp, pc, ierr)
         call PCGetType(pc, pctype, ierr)
         if (pctype == PCPFLAREINV) then
            call PCPFLAREINVGetType(pc, pflare_type, ierr)
            if (pflare_type == PFLAREINV_POWER) skip = PETSC_TRUE
         end if
      end if

!  Solve several linear systems in succession

      if (.not. skip) then
         do 100 i=1,nsteps
            call test_solve_getcoeffs(ksp, A, x, b, u, i, nsteps, coeffs_levels, coeffs_pflareinv, ierr)
 100     continue
      end if

      if (associated(coeffs_pflareinv)) deallocate(coeffs_pflareinv)

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.

      call VecDestroy(u,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(b,ierr)
      call MatDestroy(A,ierr)
      call KSPDestroy(ksp,ierr)

      call PetscFinalize(ierr)
      end


