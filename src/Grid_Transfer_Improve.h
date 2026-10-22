/*
  Prototypes for Grid_Transfer_Improve.c, the C port of Grid_Transfer_Improve.F90
  (module grid_transfer_improve)
*/

#if !defined(GRID_TRANSFER_IMPROVE_H)
#define GRID_TRANSFER_IMPROVE_H

#include <petscmat.h>

PETSC_INTERN PetscErrorCode improve_w(Mat W, Mat A_ff, Mat A_fc, Mat A_ff_inv, Mat *reuse_mat, Mat *reuse_mat_two,
   int its, PetscBool reuse_sparsity);
PETSC_INTERN PetscErrorCode improve_z(Mat Z, Mat A_ff, Mat A_cf, Mat A_ff_inv, Mat *reuse_mat, Mat *reuse_mat_two,
   int its, PetscBool reuse_sparsity);

#endif
