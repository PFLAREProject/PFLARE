/*
  Prototypes for SAI_Z.c, the C port of SAI_Z.F90 (module sai_z)

  The optional arguments of the Fortran are pointers that are NULL when the
  argument is absent
*/

#if !defined(SAI_Z_H)
#define SAI_Z_H

#include <petscmat.h>

PETSC_INTERN PetscErrorCode calculate_and_build_sai_z_cpu(Mat A_ff_input, Mat A_cf, Mat sparsity_mat_cf, PetscBool incomplete,
   Mat *reuse_mat, Mat **reuse_submatrices, Mat *z, const PetscBool *no_approx_solve);
PETSC_INTERN PetscErrorCode calculate_and_build_sai_z(Mat A_ff_input, Mat A_cf, Mat sparsity_mat_cf, PetscBool incomplete,
   Mat *reuse_mat, Mat **reuse_submatrices, Mat *z);
PETSC_INTERN PetscErrorCode calculate_and_build_sai(Mat matrix, int sparsity_order, PetscBool incomplete,
   Mat *reuse_mat, Mat **reuse_submatrices, Mat *inv_matrix);

#endif
