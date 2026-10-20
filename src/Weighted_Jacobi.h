/*
  Prototypes for Weighted_Jacobi.c, the C port of Weighted_Jacobi.F90 (module weighted_jacobi)
*/

#if !defined(WEIGHTED_JACOBI_H)
#define WEIGHTED_JACOBI_H

#include <petscmat.h>

PETSC_INTERN PetscErrorCode calculate_and_build_weighted_jacobi_inverse(Mat matrix, PetscBool weighted, Mat *inv_matrix);

#endif
