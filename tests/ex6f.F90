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

      program main
      use petscksp
#include "finclude/pflare.h"
#include "petsc/finclude/petscksp.h"
      implicit none

!  Variables:
!
!  A       - matrix that defines linear system
!  ksp    - KSP context
!  ksp     - KSP context
!  x, b, u - approx solution, RHS, exact solution vectors
!
      Vec     x,u,b
      Mat     A,A2
      KSP    ksp
      PetscInt i,j,II,JJ,m,n
      PetscInt Istart,Iend
      PetscInt nsteps,one
      PetscErrorCode ierr
      PetscBool  flg, check, regen, check_collectives
      PetscScalar  v


      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      m      = 5
      n      = 5
      nsteps = 2
      one    = 1
      regen  = PETSC_FALSE
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-m',m,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',n,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-nsteps',nsteps,flg,ierr)
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER, &
               '-regen', check,flg,ierr)  
      if (flg) regen = check

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
         call solve1(ksp,A,x,b,u,i,nsteps, regen,A2,ierr)
 100  continue

!  Check a v-cycle after the solves does no MPI collectives
!  Needs libcount_collectives.so preloaded, see make tests_collectives

      call PetscOptionsHasName(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER, &
               '-check_collectives',check_collectives,ierr)
      if (check_collectives) call check_vcycle_collectives(ksp,b,x)

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.

      call VecDestroy(u,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(b,ierr)
      call MatDestroy(A,ierr)
      call KSPDestroy(ksp,ierr)

      call PetscFinalize(ierr)
      end

! -----------------------------------------------------------------------
!
      subroutine solve1(ksp,A,x,b,u,count,nsteps,regen,A2,ierr)
#include "petsc/finclude/petscksp.h"        
      use petscksp
      use pflare
      implicit none

!
!   solve1 - This routine is used for repeated linear system solves.
!   We update the linear system matrix each time, but retain the same
!   preconditioning matrix for all linear solves.
!
!      A - linear system matrix
!      A2 - preconditioning matrix
!
      PetscScalar  v,val
      PetscInt II,Istart,Iend
      PetscInt count,nsteps,one
      PetscErrorCode ierr
      PetscInt start, start_plus_one
      PetscBool regen
      Mat     A
      KSP     ksp
      PC pc
      Vec     x,b,u
      KSPConvergedReason reason

! Use common block to retain matrix between successive subroutine calls
      Mat              A2
      PetscMPIInt      rank
      PetscBool        pflag
      common /my_data/ rank, pflag

      one = 1
! First time thorough: Create new matrix to define the linear system
      if (count .eq. 1) then

        call MatGetOwnershipRange(A, start, start_plus_one, ierr)
        if (start == 0) rank = 0
        pflag = .false.
        call PetscOptionsHasName(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-mat_view',pflag,ierr)
        if (pflag) then
          if (rank .eq. 0) write(6,100)
          call PetscFlush(6)
        endif
        call MatConvert(A,MATSAME,MAT_INITIAL_MATRIX,A2,ierr)
! All other times: Set previous solution as initial guess for next solve.
      else
        call KSPSetInitialGuessNonzero(ksp,PETSC_TRUE,ierr)
      endif

! Alter the matrix A a bit
      call MatGetOwnershipRange(A,Istart,Iend,ierr)
      do 20, II=Istart,Iend-1
        v = 2.0
        call MatSetValues(A,one,[II],one,[II],[v],ADD_VALUES,ierr)
 20   continue
      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
      if (pflag) then
        if (rank .eq. 0) write(6,110)
        call PetscFlush(6)
      endif
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)

! Set the exact solution; compute the right-hand-side vector
      val = real(count, kind=kind(val))
      call VecSet(u,val,ierr)
      call MatMult(A,u,b,ierr)

! Set operators, keeping the identical preconditioner matrix for
! all linear solves.  This approach is often effective when the
! linear systems do not change very much between successive steps.
! Note we do not need KSPSetReusePreconditioner here: A2 is a copy
! that is never modified after step 1, and petsc only triggers a PC
! re-setup off the pmat's object state, so the preconditioner is
! naturally reused. See reuse_preconditioner.c for a test of
! KSPSetReusePreconditioner with a pmat whose values change in place.
      if (.NOT. regen) then
         call KSPSetOperators(ksp,A,A2,ierr)
      else
         ! Otherwise this should trigger same_nonzero_pattern
         ! and the air pc will be rebuilt
         call KSPSetOperators(ksp,A,A,ierr)
      end if

      if (count == 1) then
         call KSPGetPC(ksp,pc,ierr)
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         !         Let's use AIRG as our PC
         ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -     
         call PCSetType(pc, PCAIR, ierr)
         call KSPSetPC(ksp, pc, ierr) 
         call KSPSetFromOptions(ksp,ierr)         
      end if         

! Solve linear system
      call KSPSolve(ksp,b,x,ierr)
      call KSPGetConvergedReason(ksp, reason, ierr)
      if (reason%v < 0) then
         error stop 1
      end if    

! Destroy the preconditioner matrix on the last time through
      if (count .eq. nsteps) then
         call MatDestroy(A2,ierr)
      end if

 100  format('previous matrix: preconditioning')
 110  format('next matrix: defines linear system')

      end

! -----------------------------------------------------------------------

      subroutine check_vcycle_collectives(ksp,b,x)
#include "petsc/finclude/petscksp.h"
      use petscksp
      implicit none

!
!   Fails the run if a PCApply calls any MPI collectives, with the counting
!   done by libcount_collectives.so (see tests/count_collectives.c), which
!   has to be preloaded. The windows it counts in are opened and closed with
!   MPI_Pcontrol, see tests/check_collectives.h for the C version
!   This is the Fortran driver, so the calibration window also makes sure
!   the library sees the collectives called from Fortran
!
      KSP ksp
      Vec b,x
      PC pc
      PetscErrorCode ierr
      PetscReal norm
      PetscMPIInt size
      integer :: status, a, a_sum, norm_reductions, err

      call get_environment_variable('PFLARE_COUNT_COLLECTIVES_LOADED', status=status)
      if (status /= 0) then
         print *, '-check_collectives needs libcount_collectives.so preloaded, see make tests_collectives'
         error stop 1
      end if
      call MPI_Comm_size(PETSC_COMM_WORLD, size, ierr)
      if (size < 2) then
         print *, '-check_collectives needs more than one rank'
         error stop 1
      end if

!     In debug builds MPIU_Allreduce does an extra reduction to check every
!     rank called it from the same line
#if defined(PETSC_USE_DEBUG)
      norm_reductions = 2
#else
      norm_reductions = 1
#endif

!     One reduction from Fortran and those in a VecNorm from C
      a = 1
      call collectives_window(norm_reductions + 2)
      call MPI_Allreduce(a, a_sum, 1, MPI_INTEGER, MPI_SUM, PETSC_COMM_WORLD, err)
      call VecNorm(b, NORM_2, norm, ierr)
      call collectives_window(0)

!     The v-cycle, after the solves have done any setup it needs
      call KSPGetPC(ksp, pc, ierr)
      call collectives_window(1)
      call PCApply(pc, b, x, ierr)
      call collectives_window(0)

      end

! ----------------------------------------------------------------------------
      subroutine collectives_window(level)
#include "petsc/finclude/petscsys.h"
      use petscsys
      implicit none

!
!   Opens (level > 0) or closes (level 0) a counting window with MPI_Pcontrol
!   MPIUNI has no Fortran MPI_Pcontrol, but the check needs two ranks, so it
!   never runs without MPI
!
      integer :: level

#if !defined(PETSC_HAVE_MPIUNI)
      call MPI_Pcontrol(level)
#endif

      end
