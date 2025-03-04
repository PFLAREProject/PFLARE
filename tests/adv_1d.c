static char help[] = "Solves a one-dimensional steady upwind advection system with KSP.\n\n";

/*
  This is a modified version of ex86 in PETSc
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h    - base PETSc routines   petscvec.h - vectors
     petscmat.h    - matrices              petscpc.h  - preconditioners
     petscis.h     - index sets
     petscviewer.h - viewers
*/
#include <petscksp.h>
#include "pflare.h"

int main(int argc, char **args)
{
  Vec         x, b; /* approx solution, RHS */
  Mat         A;              /* linear system matrix */
  KSP         ksp;            /* linear solver context */
  PC          pc;             /* preconditioner context */
  PetscInt    i, n = 100, its, global_row_start, global_row_end_plus_one, local_size;
  PetscInt    start_assign, counter;
  KSPConvergedReason reason;
  PetscLogStage setup, gpu_copy;

  PetscFunctionBeginUser;
  PetscInitialize(&argc, &args, (char*)0, help);

  PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL);
  PetscBool second_solve= PETSC_FALSE;
  PetscOptionsGetBool(NULL, NULL, "-second_solve", &second_solve, NULL);

  // Register the pflare types
  PCRegister_PFLARE();

  PetscLogStageRegister("Setup", &setup);
  PetscLogStageRegister("GPU copy stage - triggered by a prelim KSPSolve", &gpu_copy);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
  */
  VecCreate(PETSC_COMM_WORLD, &x);
  PetscObjectSetName((PetscObject)x, "Solution");
  VecSetSizes(x, PETSC_DECIDE, n);
  VecSetFromOptions(x);
  VecDuplicate(x, &b);
  VecGetLocalSize(x, &local_size);
  VecGetOwnershipRange(x, &global_row_start, &global_row_end_plus_one);

  // Row and column indices for COO assembly
  PetscInt *oor, *ooc;
  // Values for COO assembly
  PetscScalar *v;  

  /*
     Create matrix.  When using MatCreate(), the matrix format can
     be specified at runtime.

     Performance tuning note:  For problems of substantial size,
     preallocation of matrix memory is crucial for attaining good
     performance. See the matrix chapter of the users manual for details.
  */
  MatCreate(PETSC_COMM_WORLD, &A);
  MatSetSizes(A, local_size, local_size, n, n);
  MatSetFromOptions(A);

  // Going to do assembly in the COO interface so assembly happens on the gpu when needed
  // Allocate memory for the coordinates
  PetscMalloc2(2 * local_size, &oor, 2 * local_size, &ooc);
  PetscMalloc1(2 * local_size, &v);

  counter = 0;
  // Dirichlet condition on left boundary
  if (global_row_start == 0)
  {
      start_assign = 1;
      oor[counter] = 0;
      ooc[counter] = 0;
      v[counter] = 1;
      counter = counter + 1;  
  }
  else{
      start_assign = global_row_start;
  }

  // Let's write the COO format 
  for (i = start_assign; i < global_row_end_plus_one; i++) {
    // Upwinded dimensionless uniform grid finite difference operator
    oor[counter] = i;
    ooc[counter] = i - 1;
    v[counter] = -1.0;

    oor[counter + 1] = i;
    ooc[counter + 1] = i;
    v[counter + 1] = 1.0;

    counter = counter + 2;
  }

  // Set the indices
  MatSetPreallocationCOO(A, counter, oor, ooc);
  // Can delete oor and ooc now
  PetscFree2(oor, ooc);
  // Set the values
  MatSetValuesCOO(A, v, INSERT_VALUES);
  PetscFree(v);

  /*
     Create x, b - random initial guess and zero rhs
  */
  VecSet(b, 0.0);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the matrix that defines the preconditioner.
  */
  KSPSetOperators(ksp, A, A);

  /*
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the KSP context,
       we can then directly call any KSP and PC routines to set
       various options.
     - The following four statements are optional; all of these
       parameters could alternatively be specified at runtime via
       KSPSetFromOptions();
  */
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCAIR);

  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  KSPSetFromOptions(ksp);

  // Setup the ksp
  PetscLogStagePush(setup);
  KSPSetUp(ksp);
  PetscLogStagePop();

  // Do a preliminary KSPSolve so all the copies to the gpu happen
  KSPGetPC(ksp, &pc);
  VecSet(x, 1.0);

  PetscLogStagePush(gpu_copy);
  KSPSolve(ksp, b, x);
  PetscLogStagePop();

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  // We set x to 1 rather than random as the vecrandom doesn't yet have a
  // gpu implementation and we don't want a copy occuring back to the cpu    
  if (second_solve)
  { 
   VecSet(x, 1.0); 
   KSPSolve(ksp, b, x);
  }

  KSPGetIterationNumber(ksp,&its);
  KSPGetConvergedReason(ksp,&reason);

  PetscPrintf(PETSC_COMM_WORLD, "iterations %3" PetscInt_FMT "\n", its);

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  VecDestroy(&x);
  VecDestroy(&b);
  MatDestroy(&A);
  KSPDestroy(&ksp);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  PetscFinalize();
  if (reason < 0)
  {
   return 1;
  }
  return 0;
}
