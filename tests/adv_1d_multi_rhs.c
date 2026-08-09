static char help[] = "Solves a one-dimensional steady upwind advection system with multiple right-hand sides using KSPMatSolve.\n\n";

/*
  A multiple right-hand side version of adv_1d.c, built so the whole solve can
  stay on the device: the block of right-hand sides is created with
  MatCreateDenseFromVecType() using the vector type of the operator, so a
  -mat_type aijkokkos operator gives a device dense block.

    ./adv_1d_multi_rhs -n 100000 -nrhs 16 -mat_type aijkokkos -vec_type kokkos -log_view

  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h    - base PETSc routines   petscvec.h - vectors
     petscmat.h    - matrices              petscpc.h  - preconditioners
     petscis.h     - index sets
     petscviewer.h - viewers
*/
#include <petscksp.h>
#include "pflare.h"

// Tolerance for the comparison against the column-by-column reference solve
#if defined(PETSC_USE_REAL_SINGLE)
  #define DEFAULT_CHECK_TOL 1e-5
#else
  #define DEFAULT_CHECK_TOL 1e-10
#endif

int main(int argc, char **args)
{
  Vec         x;              /* only used to get the parallel layout */
  Mat         A;              /* linear system matrix */
  Mat         B, X, Xref;     /* block of rhs, block of solutions, reference solutions */
  KSP         ksp;            /* linear solver context */
  PC          pc;             /* preconditioner context */
  VecType     vtype;
  PetscInt    i, j, n = 100, nrhs = 5, global_row_start, global_row_end_plus_one, local_size;
  PetscInt    start_assign;
  PetscCount  counter;
  PetscReal   check_tol = DEFAULT_CHECK_TOL, diff_norm, x_norm;
  KSPConvergedReason reason;
  PetscLogStage setup, gpu_copy, matsolve, reference;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char*)0, help));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nrhs", &nrhs, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-check_tol", &check_tol, NULL));
  PetscBool second_solve = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-second_solve", &second_solve, NULL));
  PetscBool check = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-check", &check, NULL));

  // Register the pflare types
  PCRegister_PFLARE();

  PetscCall(PetscLogStageRegister("Setup", &setup));
  PetscCall(PetscLogStageRegister("GPU copy stage - triggered by a prelim KSPMatSolve", &gpu_copy));
  PetscCall(PetscLogStageRegister("MatSolve - the timed multiple rhs solve", &matsolve));
  PetscCall(PetscLogStageRegister("Column-by-column reference solve", &reference));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and the block of right-hand sides that define
         the linear systems, A X = B.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create a vector purely to get the parallel layout - the solve itself only
     uses the dense blocks below.
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecGetLocalSize(x, &local_size));
  PetscCall(VecGetOwnershipRange(x, &global_row_start, &global_row_end_plus_one));
  PetscCall(VecDestroy(&x));

  // Row and column indices for COO assembly
  PetscInt *oor, *ooc;
  // Values for COO assembly
  PetscScalar *v;

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, local_size, local_size, n, n));
  PetscCall(MatSetFromOptions(A));

  // Going to do assembly in the COO interface so assembly happens on the gpu when needed
  // Allocate memory for the coordinates
  PetscCall(PetscMalloc2(2 * local_size, &oor, 2 * local_size, &ooc));
  PetscCall(PetscMalloc1(2 * local_size, &v));

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
  PetscCall(MatSetPreallocationCOO(A, counter, oor, ooc));
  // Can delete oor and ooc now
  PetscCall(PetscFree2(oor, ooc));
  // Set the values
  PetscCall(MatSetValuesCOO(A, v, INSERT_VALUES));
  PetscCall(PetscFree(v));

  /*
     Create the dense blocks of right-hand sides and solutions. Taking the vector
     type from A means a device matrix gives device dense blocks, so nothing has
     to come back to the host. KSPMatSolve requires B and X to be different
     matrices of the same type.
  */
  PetscCall(MatGetVecType(A, &vtype));
  PetscCall(MatCreateDenseFromVecType(PETSC_COMM_WORLD, vtype, local_size, PETSC_DECIDE, n, nrhs, PETSC_DECIDE, NULL, &B));
  PetscCall(MatCreateDenseFromVecType(PETSC_COMM_WORLD, vtype, local_size, PETSC_DECIDE, n, nrhs, PETSC_DECIDE, NULL, &X));

  /*
     Give each column of B a different constant. We deliberately don't use
     MatSetRandom as there is no gpu implementation of the vector random and we
     don't want a copy occuring back to the cpu.
  */
  for (j = 0; j < nrhs; j++) {
    Vec cb;
    PetscCall(MatDenseGetColumnVecWrite(B, j, &cb));
    PetscCall(VecSet(cb, 1.0 + (PetscScalar)j));
    PetscCall(MatDenseRestoreColumnVecWrite(B, j, &cb));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));

  /*
     KSPMatSolve only does a genuine block solve - that is, one that reaches
     PCMatApply and hence a sparse matrix by dense matrix product - for
     KSPPREONLY and KSPHPDDM. Every other KSP type silently falls back to a
     loop of KSPSolve over the columns of B, so overriding this with -ksp_type
     turns the block solve back into a column-by-column solve.

     PCPFLAREINV is the default preconditioner here, but -pc_type air also does a
     genuine block solve - both implement PCMatApply.

     No KSPSetInitialGuessNonzero - preonly requires a zero initial guess and
     KSPMatSolve zeroes X for us anyway.
  */
  PetscCall(KSPSetType(ksp, KSPPREONLY));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCPFLAREINV));

  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  PetscCall(KSPSetFromOptions(ksp));

  // Setup the ksp
  PetscCall(PetscLogStagePush(setup));
  PetscCall(KSPSetUp(ksp));
  PetscCall(PetscLogStagePop());

  // Do a preliminary KSPMatSolve so all the copies to the gpu happen
  PetscCall(PetscLogStagePush(gpu_copy));
  PetscCall(KSPMatSolve(ksp, B, X));
  PetscCall(PetscLogStagePop());

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear systems
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscLogStagePush(matsolve));
  PetscCall(KSPMatSolve(ksp, B, X));
  if (second_solve) PetscCall(KSPMatSolve(ksp, B, X));
  PetscCall(PetscLogStagePop());

  PetscCall(KSPGetConvergedReason(ksp, &reason));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Check the block solve against a column-by-column reference solve.
       At the default preonly this is comparing PCMatApply against PCApply.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (check) {
    PetscCall(MatDuplicate(X, MAT_DO_NOT_COPY_VALUES, &Xref));

    PetscCall(PetscLogStagePush(reference));
    for (j = 0; j < nrhs; j++) {
      Vec cb, cx;
      PetscCall(MatDenseGetColumnVecRead(B, j, &cb));
      PetscCall(MatDenseGetColumnVecWrite(Xref, j, &cx));
      PetscCall(VecSet(cx, 0.0));
      PetscCall(KSPSolve(ksp, cb, cx));
      PetscCall(MatDenseRestoreColumnVecWrite(Xref, j, &cx));
      PetscCall(MatDenseRestoreColumnVecRead(B, j, &cb));
    }
    PetscCall(PetscLogStagePop());

    PetscCall(MatNorm(X, NORM_FROBENIUS, &x_norm));
    PetscCall(MatAXPY(Xref, -1.0, X, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(Xref, NORM_FROBENIUS, &diff_norm));
    PetscCheck(diff_norm <= check_tol * x_norm, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
               "Block solve differs from the column-by-column solve: ||X - Xref||_F = %g, ||X||_F = %g, tolerance %g",
               (double)diff_norm, (double)x_norm, (double)check_tol);

    PetscCall(MatDestroy(&Xref));
  }

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  PetscCall(PetscFinalize());
  if (reason < 0)
  {
   return 1;
  }
  return 0;
}
