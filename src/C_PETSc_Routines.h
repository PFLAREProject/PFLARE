/*
  Prototypes for C_PETSc_Routines.c
*/

#if !defined(C_PETSC_ROUTINES_H)
#define C_PETSC_ROUTINES_H

#include <petscmat.h>

PETSC_INTERN PetscErrorCode MatGetDiagonalOnly_c(Mat *A, int *diag_only);

#endif
