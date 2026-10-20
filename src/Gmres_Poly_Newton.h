/*
  Prototypes for Gmres_Poly_Newton.c, the C port of Gmres_Poly_Newton.F90
  (module gmres_poly_newton)

  The optional arguments of the Fortran are pointers that are NULL when the
  argument is absent. The coefficients (the real and imaginary parts of the
  roots in two columns) are column-major with coefficients_rows rows, and so is
  the status_output array (status_rows rows, two columns)
*/

#if !defined(GMRES_POLY_NEWTON_H)
#define GMRES_POLY_NEWTON_H

#include <petscmat.h>
#include "pflareimpl.h"

PETSC_INTERN PetscErrorCode modified_leja(PetscReal real_roots[], PetscReal imag_roots[], PetscInt roots_size, PetscInt **indices);
PETSC_INTERN PetscErrorCode cluster_eigenvalues_stable(PetscReal real_roots[], PetscReal imag_roots[], PetscInt roots_size, PetscReal rel_tol, PetscReal abs_tol);
PETSC_INTERN PetscErrorCode compute_extra_roots(PetscReal real_roots[], PetscReal imag_roots[], PetscInt roots_size,
   PetscReal **real_roots_output, PetscReal **imag_roots_output, PetscInt *roots_output_size);
PETSC_INTERN PetscErrorCode calculate_gmres_polynomial_roots_newton(Mat matrix, int poly_order, PetscBool add_roots,
   PetscReal **coefficients, PetscInt *coefficients_rows, PetscInt *coefficients_cols);
PETSC_INTERN PetscErrorCode build_newton_fixed_sparsity_start(Mat matrix, int poly_order, int poly_sparsity_order, PetscReal coefficients[], PetscInt coefficients_rows,
   Mat *cmat, Mat *mat_sparsity_match, Mat *mat_product_save, int status_output[], PetscInt status_rows, PetscBool *output_first_complex);
PETSC_INTERN PetscErrorCode mat_mult_powers_share_sparsity_newton(Mat matrix, int poly_order, int poly_sparsity_order, PetscReal coefficients[], PetscInt coefficients_rows,
   Mat *reuse_mat, Mat **reuse_submatrices, Mat *cmat);
PETSC_INTERN PetscErrorCode mat_mult_powers_share_sparsity_newton_cpu(Mat matrix, int poly_order, int poly_sparsity_order, PetscReal coefficients[], PetscInt coefficients_rows,
   Mat *reuse_mat, Mat **reuse_submatrices, Mat *cmat);
PETSC_INTERN PetscErrorCode build_gmres_polynomial_newton_inverse(Mat matrix, int poly_order, PetscReal coefficients[], PetscInt coefficients_rows,
   int poly_sparsity_order, PetscBool matrix_free, PetscBool diag_scale_polys, Mat *reuse_mat, Mat **reuse_submatrices, Mat *inv_matrix);
PETSC_INTERN PetscErrorCode build_gmres_polynomial_newton_inverse_0th_order(Mat matrix, int poly_order, PetscReal coefficients[], PetscInt coefficients_rows, Mat *inv_matrix);
PETSC_INTERN PetscErrorCode build_gmres_polynomial_inverse_0th_order_sparsity_newton(Mat matrix, int poly_order, PetscReal coefficients[], PetscInt coefficients_rows, Mat *inv_matrix);
PETSC_INTERN PetscErrorCode build_gmres_polynomial_newton_inverse_1st_1st(Mat matrix, int poly_order, PetscReal coefficients[], PetscInt coefficients_rows,
   Mat *inv_matrix, Mat *mat_prod_or_temp, int status_output[], PetscInt status_rows);
PETSC_INTERN PetscErrorCode build_gmres_polynomial_newton_inverse_full(Mat matrix, int poly_order, PetscReal coefficients[], PetscInt coefficients_rows,
   Mat *inv_matrix, Mat *mat_prod_or_temp, const int *poly_sparsity_order, PetscBool *output_first_complex,
   int status_output[], PetscInt status_rows, Mat *mat_product_save);

#endif
