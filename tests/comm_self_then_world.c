static char help[] = "Tests setting up PFLARE on PETSC_COMM_SELF and then on PETSC_COMM_WORLD in the same program.\n\n";

/*
  Sets up and solves with a PC on PETSC_COMM_SELF, and then with a new PC on
  PETSC_COMM_WORLD, each followed by re-setups with new values in the same
  nonzero pattern.

  This is the pattern you get with PFLARE as the sub-PC of a block Jacobi, or
  with a serial solve of some other system before the main parallel one. It
  catches any state that the serial (comm size 1) paths leave behind that then
  breaks the parallel paths, eg local variables with an implicit save. Run it
  with mpiexec -n 2 or more, as with one rank both solves take the serial paths.

    mpiexec -n 2 ./comm_self_then_world -pc_type pflareinv -pc_pflareinv_type arnoldi -pc_pflareinv_sparsity_order 1
    mpiexec -n 2 ./comm_self_then_world -pc_type air -pc_air_inverse_type newton -pc_air_reuse_sparsity
*/
#include <petscksp.h>
#include "pflare.h"

/*
   Writes a nonsymmetric 1D upwind advection-diffusion stencil into A, with a
   shift that keeps it strictly diagonally dominant. Called again with a
   different advection to re-setup with the same nonzero pattern.
*/
static PetscErrorCode SetOperatorValues(Mat A, PetscInt n, PetscReal advection)
{
  PetscInt    i, global_row_start, global_row_end_plus_one, cols[3], n_cols;
  PetscScalar vals[3];

  PetscFunctionBeginUser;
  PetscCall(MatGetOwnershipRange(A, &global_row_start, &global_row_end_plus_one));

  for (i = global_row_start; i < global_row_end_plus_one; i++) {
    n_cols = 0;
    if (i > 0) {
      cols[n_cols] = i - 1;
      vals[n_cols] = -1.0 - advection;
      n_cols++;
    }
    cols[n_cols] = i;
    vals[n_cols] = 2.0 + advection + 0.1;
    n_cols++;
    if (i < n - 1) {
      cols[n_cols] = i + 1;
      vals[n_cols] = -1.0;
      n_cols++;
    }
    PetscCall(MatSetValues(A, 1, &i, n_cols, cols, vals, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Builds the operator on comm and solves with it n_setups times, changing the
   values (but not the nonzero pattern) before each solve after the first so
   the PC is re-setup with SAME_NONZERO_PATTERN
*/
static PetscErrorCode SetupAndSolve(MPI_Comm comm, PetscInt n, PetscInt n_setups)
{
  Mat                A;
  Vec                x, b;
  KSP                ksp;
  PC                 pc;
  PetscInt           setup;
  KSPConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, 3, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 3, NULL, 2, NULL));
  PetscCall(MatSetUp(A));
  PetscCall(SetOperatorValues(A, n, 1.0));

  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecSet(b, 1.0));

  PetscCall(KSPCreate(comm, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPGetPC(ksp, &pc));
  // Default to PCPFLAREINV, can be overridden with -pc_type
  PetscCall(PCSetType(pc, PCPFLAREINV));
  PetscCall(KSPSetFromOptions(ksp));

  for (setup = 0; setup < n_setups; setup++) {
    // Change the values in the same nonzero pattern so the next solve triggers a re-setup
    if (setup > 0) PetscCall(SetOperatorValues(A, n, 1.0 + 0.5 * setup));
    PetscCall(VecSet(x, 0.0));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    PetscCheck(reason > 0, comm, PETSC_ERR_NOT_CONVERGED, "Solve %" PetscInt_FMT " did not converge, reason %d", setup, (int)reason);
  }

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  PetscInt n = 200, n_setups = 3;

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  // Register the pflare types
  PCRegister_PFLARE();

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_setups", &n_setups, NULL));

  // Serial first, so any state left behind by the comm size 1 paths is
  // then seen by the parallel setups
  PetscCall(SetupAndSolve(PETSC_COMM_SELF, n, n_setups));
  PetscCall(SetupAndSolve(PETSC_COMM_WORLD, n, n_setups));

  PetscCall(PetscFinalize());
  return 0;
}
