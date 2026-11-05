/*
  Helpers for the -check_collectives option of the test drivers, which fails the
  run if a window of code (e.g., a PCApply) calls any MPI collectives on a
  communicator with more than one rank. The counting is done by
  libcount_collectives.so, which has to be preloaded (see count_collectives.c
  and make tests_collectives), the windows are opened and closed with
  MPI_Pcontrol
*/
#ifndef PFLARE_CHECK_COLLECTIVES_H
#define PFLARE_CHECK_COLLECTIVES_H

#include <stdlib.h>
#include <petscvec.h>

/*
  Checks the counting library is loaded and is seeing the collectives PETSc
  calls, by counting the reductions in a VecNorm
  Without this a run without the library preloaded, or one with an MPI whose
  calls the library can't see, would pass every check with a count of zero
*/
static inline PetscErrorCode CheckCollectivesCalibrate(Vec v)
{
  PetscReal   norm;
  PetscMPIInt size;

  PetscFunctionBeginUser;
  PetscCheck(getenv("PFLARE_COUNT_COLLECTIVES_LOADED"), PetscObjectComm((PetscObject)v), PETSC_ERR_USER,
             "-check_collectives needs libcount_collectives.so preloaded, see make tests_collectives");
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)v), &size));
  PetscCheck(size > 1, PetscObjectComm((PetscObject)v), PETSC_ERR_USER, "-check_collectives needs more than one rank");
  // Expect the one reduction in a VecNorm (level is the count + 1), and in
  // debug builds MPIU_Allreduce does an extra one to check every rank called
  // it from the same line
  PetscCallMPI(MPI_Pcontrol(PetscDefined(USE_DEBUG) ? 3 : 2));
  PetscCall(VecNorm(v, NORM_2, &norm));
  PetscCallMPI(MPI_Pcontrol(0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Open a window that has to see no collectives, and close it, which aborts
// with a backtrace of each collective if it saw any
#define CheckCollectivesBegin() MPI_Pcontrol(1)
#define CheckCollectivesEnd()   MPI_Pcontrol(0)

#endif
