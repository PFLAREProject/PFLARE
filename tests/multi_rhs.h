/*
  Shared helpers for the multiple right-hand side (KSPMatSolve) test drivers.

  CheckBlockSolve compares a block solve against a column-by-column reference
  solve. MultiRhsSolve does the whole -nrhs path for a driver that already has
  a set up KSP and a single right-hand side, with the same log stage names as
  adv_1d_multi_rhs.c so one script can read the -log_view of all of them.
*/
#ifndef PFLARE_TESTS_MULTI_RHS_H
#define PFLARE_TESTS_MULTI_RHS_H

#include <petscksp.h>

// Tolerance for the comparison against the column-by-column reference solve
#if defined(PETSC_USE_REAL_SINGLE)
  #define DEFAULT_CHECK_TOL 1e-5
#else
  #define DEFAULT_CHECK_TOL 1e-10
#endif

/*
  Check the block solve in X against a column-by-column reference solve of the
  same systems. At preonly this is comparing PCMatApply against PCApply.
*/
static inline PetscErrorCode CheckBlockSolve(KSP ksp, Mat B, Mat X, PetscReal check_tol)
{
  Mat       A, Xref;
  Vec       b, x;
  PetscInt  j, nrhs;
  PetscReal diff_norm, x_norm;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(B, NULL, &nrhs));
  PetscCall(MatDuplicate(X, MAT_DO_NOT_COPY_VALUES, &Xref));

  // The single rhs solves use vecs of the operator's type, as with -host_blocks
  // the columns of the blocks are host vecs regardless of the operator's type
  PetscCall(KSPGetOperators(ksp, &A, NULL));
  PetscCall(MatCreateVecs(A, &x, &b));
  for (j = 0; j < nrhs; j++) {
    Vec cb, cx;
    PetscCall(MatDenseGetColumnVecRead(B, j, &cb));
    PetscCall(VecCopy(cb, b));
    PetscCall(MatDenseRestoreColumnVecRead(B, j, &cb));
    PetscCall(VecSet(x, 0.0));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(MatDenseGetColumnVecWrite(Xref, j, &cx));
    PetscCall(VecCopy(x, cx));
    PetscCall(MatDenseRestoreColumnVecWrite(Xref, j, &cx));
  }
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));

  PetscCall(MatNorm(X, NORM_FROBENIUS, &x_norm));
  PetscCall(MatAXPY(Xref, -1.0, X, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(Xref, NORM_FROBENIUS, &diff_norm));
  PetscCheck(diff_norm <= check_tol * x_norm, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
             "Block solve differs from the column-by-column solve: ||X - Xref||_F = %g, ||X||_F = %g, tolerance %g",
             (double)diff_norm, (double)x_norm, (double)check_tol);

  PetscCall(MatDestroy(&Xref));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Block solve of nrhs right-hand sides with the operators already set on ksp.
  Column j of the block is b + j, so column 0 is the driver's own problem and
  the others differ. Reads the options
    -host_blocks  : create the blocks as host MATDENSE regardless of the operator's type
    -check        : compare against a column-by-column reference solve (default true)
    -check_tol    : relative Frobenius tolerance of that comparison
    -check_copies : fail if a second block solve copies anything between the host and the device

  KSPMatSolve only does a genuine block solve - that is, one that reaches
  PCMatApply and hence a sparse matrix by dense matrix product - for
  KSPPREONLY, KSPHPDDM and KSPRICHARDSON (which iterates blockwise through
  PCMatApplyRichardson). Every other KSP type silently falls back to a loop
  of KSPSolve over the columns of B, so the KSP type is left to the caller.

  This turns off any nonzero initial guess on ksp: preonly requires a zero
  initial guess, and it makes every block solve and every reference solve
  start from zero so they are like for like.
*/
static inline PetscErrorCode MultiRhsSolve(KSP ksp, Vec b, PetscInt nrhs, KSPConvergedReason *reason)
{
  Mat           A, B, X;
  VecType       vtype;
  PetscInt      j, local_size, global_size;
  PetscReal     check_tol = DEFAULT_CHECK_TOL;
  PetscBool     check = PETSC_TRUE, check_copies = PETSC_FALSE, host_blocks = PETSC_FALSE;
  PetscLogStage gpu_copy, matsolve, reference;
  PetscLogEvent matsolve_event;

  PetscFunctionBeginUser;
  PetscCheck(nrhs >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-nrhs must be positive");
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-check_tol", &check_tol, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-check", &check, NULL));
  // Check the second block solve does not copy anything between the host and the device
  // This is only meaningful with device matrices and vectors (e.g. -mat_type aijkokkos
  // -vec_type kokkos on a gpu), everywhere else the counts are trivially zero
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-check_copies", &check_copies, NULL));
  // Create the dense blocks with MatCreateDense, so they are host MATDENSE
  // regardless of the type of the operator
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-host_blocks", &host_blocks, NULL));

  // The copy counts are read from the default log handler, which is only on with -log_view
  if (check_copies) PetscCall(PetscLogDefaultBegin());
  PetscCall(PetscLogStageRegister("GPU copy stage - triggered by a prelim KSPMatSolve", &gpu_copy));
  PetscCall(PetscLogStageRegister("MatSolve - the timed multiple rhs solve", &matsolve));
  PetscCall(PetscLogStageRegister("Column-by-column reference solve", &reference));
  PetscCall(PetscLogEventRegister("SecondBlockSolve", KSP_CLASSID, &matsolve_event));

  /*
     Create the dense blocks of right-hand sides and solutions. Taking the vector
     type from A means a device matrix gives device dense blocks, so nothing has
     to come back to the host. KSPMatSolve requires B and X to be different
     matrices of the same type.
     With -host_blocks the blocks are host MATDENSE even with a device matrix,
     which PCMG hands straight to the finest level of the preconditioner.
  */
  PetscCall(KSPGetOperators(ksp, &A, NULL));
  PetscCall(VecGetLocalSize(b, &local_size));
  PetscCall(VecGetSize(b, &global_size));
  if (host_blocks) {
    PetscCall(MatCreateDense(PETSC_COMM_WORLD, local_size, PETSC_DECIDE, global_size, nrhs, NULL, &B));
    PetscCall(MatCreateDense(PETSC_COMM_WORLD, local_size, PETSC_DECIDE, global_size, nrhs, NULL, &X));
  } else {
    PetscCall(MatGetVecType(A, &vtype));
    PetscCall(MatCreateDenseFromVecType(PETSC_COMM_WORLD, vtype, local_size, PETSC_DECIDE, global_size, nrhs, PETSC_DECIDE, NULL, &B));
    PetscCall(MatCreateDenseFromVecType(PETSC_COMM_WORLD, vtype, local_size, PETSC_DECIDE, global_size, nrhs, PETSC_DECIDE, NULL, &X));
  }

  /*
     Column j of B is b + j. We deliberately don't use MatSetRandom as there is
     no gpu implementation of the vector random and we don't want a copy
     occuring back to the cpu.
  */
  for (j = 0; j < nrhs; j++) {
    Vec cb;
    PetscCall(MatDenseGetColumnVecWrite(B, j, &cb));
    PetscCall(VecCopy(b, cb));
    PetscCall(VecShift(cb, (PetscScalar)j));
    PetscCall(MatDenseRestoreColumnVecWrite(B, j, &cb));
  }

  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_FALSE));

  // Do a preliminary KSPMatSolve so all the copies to the gpu happen
  PetscCall(PetscLogStagePush(gpu_copy));
  PetscCall(KSPMatSolve(ksp, B, X));
  PetscCall(PetscLogStagePop());

  PetscCall(PetscLogStagePush(matsolve));
  PetscCall(KSPMatSolve(ksp, B, X));
  if (check_copies) {
    PetscCall(PetscLogEventBegin(matsolve_event, 0, 0, 0, 0));
    PetscCall(KSPMatSolve(ksp, B, X));
    PetscCall(PetscLogEventEnd(matsolve_event, 0, 0, 0, 0));
  }
  PetscCall(PetscLogStagePop());

  PetscCall(KSPGetConvergedReason(ksp, reason));

#if PetscDefined(HAVE_DEVICE)
  // The earlier block solves have already moved everything the solve needs onto
  // the device, so with device matrices and dense blocks the second solve must
  // not copy anything between the host and the device in either direction
  if (check_copies) {
    PetscEventPerfInfo info;

    PetscCall(PetscLogEventGetPerfInfo(matsolve, matsolve_event, &info));
    PetscCheck(info.GpuToCpuCount == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%g unexpected GPU to CPU copies (%g bytes) in the second KSPMatSolve", info.GpuToCpuCount, info.GpuToCpuSize);
    PetscCheck(info.CpuToGpuCount == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%g unexpected CPU to GPU copies (%g bytes) in the second KSPMatSolve", info.CpuToGpuCount, info.CpuToGpuSize);
  }
#endif

  if (check) {
    PetscCall(PetscLogStagePush(reference));
    PetscCall(CheckBlockSolve(ksp, B, X, check_tol));
    PetscCall(PetscLogStagePop());
  }

  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
