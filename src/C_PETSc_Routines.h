/*
  Prototypes for C_PETSc_Routines.c
*/

#if !defined(C_PETSC_ROUTINES_H)
#define C_PETSC_ROUTINES_H

#include <petscksp.h>

PETSC_INTERN PetscErrorCode MatGetDiagonalOnly_c(Mat *A, int *diag_only);
PETSC_INTERN PetscErrorCode PetscFree_c(void **ptr);

#endif
