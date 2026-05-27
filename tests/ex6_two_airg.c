static char help[] = "Reads a PETSc matrix and sets up two concurrent PCAIR\n\
preconditioners on it, this checks the data structures are per PCAIR.\n\
\n\
Input arguments:\n\
  -f <input_file> : matrix to load (see $PETSC_DIR/share/petsc/datafiles/matrices)\n\n";

#include <petscksp.h>
#include "pflare.h"

/* Build a single KSP wrapping PCAIR on the given operator. KSPSetUp is called
   explicitly so that the per-PCAIR multigrid hierarchy (and the per-level
   fine/coarse IS views, on the Kokkos path) is constructed up front. Pass a
   distinct strong_threshold per instance so the two PCAIRs produce different
   CF splittings (and therefore different per-level IS views) — that is what
   makes the file-scope-globals bug observable; with identical splittings the
   overwritten view happens to have the same contents and the bug is silent. */
static PetscErrorCode BuildAIRKSP(MPI_Comm comm, Mat A, const char *prefix, PetscReal strong_threshold, KSP *ksp)
{
  PC pc;
  PetscFunctionBeginUser;
  PetscCall(KSPCreate(comm, ksp));
  PetscCall(KSPSetOperators(*ksp, A, A));
  PetscCall(KSPSetTolerances(*ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, 100));
  PetscCall(KSPGetPC(*ksp, &pc));
  PetscCall(PCSetType(pc, PCAIR));
  PetscCall(PCAIRSetStrongThreshold(pc, strong_threshold));
  if (prefix) PetscCall(KSPSetOptionsPrefix(*ksp, prefix));
  PetscCall(KSPSetFromOptions(*ksp));
  PetscCall(KSPSetUp(*ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat                A, A_diff_type;
  Vec                b, x1, x2;
  PetscRandom        rnd;
  PetscViewer        fd;
  char               file[PETSC_MAX_PATH_LEN];
  PetscBool          flg;
  PetscInt           m, n, M, N, one = 1;
  MatType            mtype, mtype_input;
  KSP                ksp1, ksp2;
  KSPConvergedReason reason1, reason2;
  int                npe;

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PCRegister_PFLARE();

  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", file, sizeof(file), &flg));
  if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Must indicate binary file with the -f option");
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatLoad(A, fd));
  PetscCall(PetscViewerDestroy(&fd));

  /* Partition the loaded matrix when in parallel (copy from ex6.c). */
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &npe));
  if (npe != 1) {
    MatPartitioning part;
    IS              is, isrows;
    Mat             A_partitioned;
    PetscCall(MatPartitioningCreate(PETSC_COMM_WORLD, &part));
    PetscCall(MatPartitioningSetAdjacency(part, A));
    PetscCall(MatPartitioningSetNParts(part, npe));
    PetscCall(MatPartitioningSetFromOptions(part));
    PetscCall(MatPartitioningApply(part, &is));
    PetscCall(ISBuildTwoSided(is, NULL, &isrows));
    PetscCall(MatCreateSubMatrix(A, isrows, isrows, MAT_INITIAL_MATRIX, &A_partitioned));
    PetscCall(MatDestroy(&A));
    PetscCall(MatPartitioningDestroy(&part));
    PetscCall(ISDestroy(&is));
    PetscCall(ISDestroy(&isrows));
    A = A_partitioned;
  }

  /* Convert A to the user-requested matrix type (e.g. -mat_type aijkokkos). */
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(MatCreateFromOptions(PETSC_COMM_WORLD, NULL, one, m, n, M, N, &A_diff_type));
  PetscCall(MatAssemblyBegin(A_diff_type, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A_diff_type, MAT_FINAL_ASSEMBLY));

  PetscCall(MatGetType(A, &mtype));
  PetscCall(MatGetType(A_diff_type, &mtype_input));
  if (strcmp(mtype, mtype_input) != 0) {
    PetscCall(MatCopy(A, A_diff_type, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatDestroy(&A));
    A = A_diff_type;
  } else {
    PetscCall(MatDestroy(&A_diff_type));
  }

  /* Build a random RHS that matches A's vec type. */
  PetscCall(MatCreateVecs(A, &b, &x1));
  PetscCall(VecDuplicate(x1, &x2));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rnd));
  PetscCall(PetscRandomSetFromOptions(rnd));
  PetscCall(VecSetRandom(b, rnd));

  /* Build BOTH KSPs (each with its own PCAIR on A) before applying either.
     Different strong thresholds give different CF splittings — without that,
     the two PCAIRs end up with identical per-level IS views and the bug is
     hidden behind matching contents. */
  PetscCall(BuildAIRKSP(PETSC_COMM_WORLD, A, "air1_", 0.3, &ksp1));
  PetscCall(BuildAIRKSP(PETSC_COMM_WORLD, A, "air2_", 0.8, &ksp2));

  PetscCall(KSPSolve(ksp1, b, x1));
  PetscCall(KSPGetConvergedReason(ksp1, &reason1));

  /* ksp2's solve, the second AIRG application. */
  PetscCall(KSPSolve(ksp2, b, x2));
  PetscCall(KSPGetConvergedReason(ksp2, &reason2));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "ksp1 reason = %d, ksp2 reason = %d\n",
                        (int)reason1, (int)reason2));

  int exit_code = (reason1 >= 0 && reason2 >= 0) ? 0 : 1;

  PetscCall(KSPDestroy(&ksp1));
  PetscCall(KSPDestroy(&ksp2));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x1));
  PetscCall(VecDestroy(&x2));
  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return exit_code;
}
