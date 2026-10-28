/*
  Prototypes for SAbs.c, the C port of SAbs.F90 (module sabs)

  The optional arguments of the Fortran are pointers that are NULL when the
  argument is absent
*/

#if !defined(SABS_H)
#define SABS_H

#include <petscmat.h>

PETSC_INTERN PetscErrorCode generate_sabs(Mat input_mat, PetscReal strong_threshold, PetscBool symmetrize, PetscBool square, Mat *output_mat,
   const PetscBool *allow_drop_diagonal, const PetscBool *allow_diag_strength);

#endif
