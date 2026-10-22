/*
  Prototypes for Approx_Inverse_Setup.c, the C port of Approx_Inverse_Setup.F90
  (module approx_inverse_setup)

  The optional coefficients of calculate_and_build_approximate_inverse are a
  pointer that is NULL when the argument is absent; *coefficients is NULL when
  it is present but not associated. The coefficients are column-major with
  coefficients_rows rows and coefficients_cols columns
*/

#if !defined(APPROX_INVERSE_SETUP_H)
#define APPROX_INVERSE_SETUP_H

#include <petscmat.h>
#include "pflareimpl.h"

PETSC_INTERN PetscErrorCode calculate_and_build_approximate_inverse(Mat matrix, int inverse_type,
   int poly_order, int inverse_sparsity_order,
   PetscBool matrix_free, PetscBool diag_scale_polys, PetscBool subcomm,
   Mat *inv_matrix, PetscReal **coefficients, PetscInt *coefficients_rows, PetscInt *coefficients_cols);
PETSC_INTERN PetscBool inverse_type_is_polynomial(int inverse_type);
PETSC_INTERN PetscErrorCode start_approximate_inverse(Mat matrix, int inverse_type, int poly_order, PetscBool diag_scale_polys,
   tsqr_buffers *buffers, PetscReal **coefficients, PetscInt *coefficients_rows, PetscInt *coefficients_cols);
PETSC_INTERN PetscErrorCode finish_approximate_inverse(Mat matrix, int inverse_type,
   int poly_order, int inverse_sparsity_order,
   tsqr_buffers *buffers, PetscReal coefficients[], PetscInt coefficients_rows, PetscInt coefficients_cols,
   PetscBool matrix_free, PetscBool diag_scale_polys, Mat *reuse_mat, Mat **reuse_submatrices, Mat *inv_matrix);
PETSC_INTERN PetscErrorCode reset_inverse_mat(Mat *matrix);

#endif
