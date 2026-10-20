/*
  Prototypes for Gmres_Poly.c, the C port of Gmres_Poly.F90 (module gmres_poly)

  The optional arguments of the Fortran are pointers that are NULL when the
  argument is absent. The dense arrays are column-major and are passed with
  their dimensions
*/

#if !defined(GMRES_POLY_H)
#define GMRES_POLY_H

#include <petscmat.h>
#include <petscblaslapack.h>
#include "pflareimpl.h"

// petscblaslapack.h has no gelsd, so it is declared here with PETSc's BLAS/LAPACK
// name mangling (the Fortran called the same routine through pflare_blaslapack.h)
#if !defined(LAPACKgelsd_)
  #define LAPACKgelsd_ PETSCBLAS(gelsd, GELSD)
#endif
BLAS_EXTERN void LAPACKgelsd_(const PetscBLASInt *, const PetscBLASInt *, const PetscBLASInt *, PetscReal *, const PetscBLASInt *, PetscReal *,
   const PetscBLASInt *, PetscReal *, const PetscReal *, PetscBLASInt *, PetscReal *, const PetscBLASInt *, PetscBLASInt *, PetscBLASInt *);

PETSC_INTERN PetscReal fortran_norm2(const PetscReal x[], PetscInt n);
PETSC_INTERN PetscErrorCode setup_gmres_poly_data(PetscInt global_rows, int inverse_type, int poly_order,
   int poly_sparsity_order, PetscBool subcomm, PetscInt proc_stride, gmres_poly_data *poly_data);
PETSC_INTERN PetscErrorCode compute_mf_gmres_poly_num_matvecs(int inverse_type, const PetscReal coefficients[], PetscInt coefficients_rows, PetscInt coefficients_cols, int *matvecs);
PETSC_INTERN PetscErrorCode create_temp_space_box_muller(Mat matrix, PetscInt subspace_size, Vec **V_n);
PETSC_INTERN PetscErrorCode ls_solve_arnoldi(PetscReal beta, PetscInt m, const PetscReal H_n[], PetscInt H_n_rows, PetscInt H_n_cols, PetscReal y[], PetscInt y_size);
PETSC_INTERN PetscErrorCode arnoldi(Mat matrix, int poly_order, PetscReal lucky_tol, Vec V_n[], Vec w_j, PetscReal *beta, PetscReal H_n[], PetscInt *m,
   PetscReal C_n[], PetscReal y[], PetscReal *user_rel_tol);
PETSC_INTERN PetscErrorCode calculate_gmres_polynomial_coefficients_arnoldi(Mat matrix, int poly_order, PetscReal coefficients[], PetscReal *user_rel_tol);
PETSC_INTERN PetscErrorCode start_gmres_polynomial_coefficients_power(Mat matrix, int poly_order, tsqr_buffers *buffers, PetscReal coefficients[]);
PETSC_INTERN PetscErrorCode finish_gmres_polynomial_coefficients_power(int poly_order, tsqr_buffers *buffers, PetscReal coefficients[]);
PETSC_INTERN PetscErrorCode mat_mult_powers_share_sparsity(Mat matrix, int poly_order, int poly_sparsity_order, tsqr_buffers *buffers,
   PetscReal coefficients[], PetscInt coefficients_size, Mat *reuse_mat, Mat **reuse_submatrices, Mat *cmat);
PETSC_INTERN PetscErrorCode mat_mult_powers_share_sparsity_cpu(Mat matrix, int poly_order, int poly_sparsity_order, tsqr_buffers *buffers,
   PetscReal coefficients[], PetscInt coefficients_size, Mat *reuse_mat, Mat **reuse_submatrices, Mat *cmat);
PETSC_INTERN PetscErrorCode build_gmres_polynomial_inverse(Mat matrix, int poly_order, tsqr_buffers *buffers, PetscReal coefficients[], PetscInt coefficients_size,
   int poly_sparsity_order, PetscBool matrix_free, PetscBool diag_scale_polys, Mat *reuse_mat, Mat **reuse_submatrices, Mat *inv_matrix);
PETSC_INTERN PetscErrorCode build_gmres_polynomial_inverse_0th_order(Mat matrix, int poly_order, tsqr_buffers *buffers, PetscReal coefficients[], Mat *inv_matrix);
PETSC_INTERN PetscErrorCode build_gmres_polynomial_inverse_0th_order_sparsity(Mat matrix, int poly_order, tsqr_buffers *buffers, PetscReal coefficients[], Mat *inv_matrix);

#endif
