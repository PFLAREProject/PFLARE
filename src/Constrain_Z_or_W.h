/*
  Prototypes for Constrain_Z_or_W.c, the C port of Constrain_Z_or_W.F90
  (module constrain_z_or_w)
*/

#if !defined(CONSTRAIN_Z_OR_W_H)
#define CONSTRAIN_Z_OR_W_H

#include <petscmat.h>

PETSC_INTERN PetscErrorCode get_near_nullspace(Mat input_mat, PetscBool left, PetscBool right,
   Vec **left_null_vecs, Vec **right_null_vecs, PetscInt *null_vecs_size);
PETSC_INTERN PetscErrorCode smooth_near_nullspace(Mat input_mat, PetscBool left, PetscBool right,
   Vec left_null_vecs[], PetscInt left_null_vecs_size, Vec right_null_vecs[], PetscInt right_null_vecs_size);
PETSC_INTERN PetscErrorCode constrain_grid_transfer(Mat *z_or_w, PetscBool is_z, Vec null_vecs_f[], Vec null_vecs_c[],
   PetscInt null_vecs_size);

#endif
