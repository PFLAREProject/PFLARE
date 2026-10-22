/*
  Prototypes for Grid_Transfer.c, the C port of Grid_Transfer.F90 (module grid_transfer)
*/

#if !defined(GRID_TRANSFER_H)
#define GRID_TRANSFER_H

#include <petscmat.h>

PETSC_INTERN PetscErrorCode generate_one_point_with_one_entry_from_sparse(Mat input_mat, Mat *output_mat);
PETSC_INTERN PetscErrorCode generate_one_point_with_one_entry_from_sparse_cpu(Mat input_mat, Mat *output_mat);
PETSC_INTERN PetscErrorCode compute_P_from_W(Mat W, PetscInt global_row_start, IS is_fine, IS is_coarse,
   PetscBool identity, PetscBool reuse, Mat *P);
PETSC_INTERN PetscErrorCode compute_P_from_W_cpu(Mat W, PetscInt global_row_start, IS is_fine, IS is_coarse,
   PetscBool identity, PetscBool reuse, Mat *P);
PETSC_INTERN PetscErrorCode compute_R_from_Z(Mat Z, PetscInt global_row_start, IS is_fine, IS is_coarse,
   IS *orig_fine_col_indices, PetscBool identity, PetscBool reuse, Mat *R);
PETSC_INTERN PetscErrorCode compute_R_from_Z_cpu(Mat Z, PetscInt global_row_start, IS is_fine, IS is_coarse,
   IS *orig_fine_col_indices, PetscBool identity, PetscBool reuse, Mat *R);

#endif
