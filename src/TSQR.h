/*
  Prototypes for TSQR.c, the C port of TSQR.F90 (module tsqr)

  The tsqr_buffers struct is in pflareimpl.h
*/

#if !defined(TSQR_H)
#define TSQR_H

#include <petscmat.h>
#include "pflareimpl.h"

PETSC_INTERN PetscErrorCode create_tsqr_buffers(tsqr_buffers **buffers);
PETSC_INTERN PetscErrorCode destroy_tsqr_buffers(tsqr_buffers **buffers);
PETSC_INTERN PetscErrorCode setup_custom_tsqr_reduction(void);
PETSC_INTERN PetscErrorCode start_tsqr(MPI_Comm MPI_COMM_MATRIX, PetscReal A[], PetscInt m_size, PetscInt n_size, tsqr_buffers *buffers);
PETSC_INTERN void custom_reduction_tsqr(void *invec, void *inoutvec, int *len, MPI_Datatype *type);
PETSC_INTERN PetscErrorCode finish_tsqr_parallel(tsqr_buffers *buffers);

#endif
