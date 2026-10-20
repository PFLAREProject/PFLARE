/*
  Prototypes for Gmres_Poly_Apply.c, the C port of Gmres_Poly_Apply.F90
  (module gmres_poly_apply)

  The petsc_matvec_* routines are the MatShell callbacks (MATOP_MULT) of the
  matrix-free polynomial inverses, with the context the mat_ctxtype of
  pflareimpl.h
*/

#if !defined(GMRES_POLY_APPLY_H)
#define GMRES_POLY_APPLY_H

#include <petscmat.h>
#include "pflareimpl.h"

PETSC_INTERN PetscInt highest_nonzero_coeff(const PetscReal coefficients[], PetscInt coefficients_size);
PETSC_INTERN PetscErrorCode petsc_matvec_da_poly_mf(Mat mat, Vec x, Vec y);
PETSC_INTERN PetscErrorCode petsc_matvec_ida_neumann_poly_mf(Mat mat, Vec x, Vec y);
PETSC_INTERN PetscErrorCode petsc_matvec_poly_mf(Mat mat, Vec x, Vec y);
PETSC_INTERN PetscErrorCode petsc_matvec_right_scale_poly_mf(Mat mat, Vec x, Vec y);
PETSC_INTERN PetscErrorCode petsc_horner(Mat mat, const PetscReal coefficients[], PetscInt coefficients_size, Vec temp_vec, Vec x, Vec y);
PETSC_INTERN PetscErrorCode ensure_transpose_mat(mat_ctxtype *mat_ctx);
PETSC_INTERN PetscErrorCode destroy_transpose_mat(mat_ctxtype *mat_ctx);
PETSC_INTERN PetscErrorCode petsc_matvec_poly_transpose_mf(Mat mat, Vec x, Vec y);
PETSC_INTERN PetscErrorCode petsc_horner_block(mat_ctxtype *mat_ctx, Mat x_mat, Mat y_mat, Vec recip_diag, PetscBool neumann_inner, PetscBool *block_applied);
PETSC_INTERN PetscErrorCode petsc_matvec_poly_newton_mf(Mat mat, Vec x, Vec y);
PETSC_INTERN PetscErrorCode petsc_matvec_right_scale_poly_newton_mf(Mat mat, Vec x, Vec y);
PETSC_INTERN PetscErrorCode petsc_matvec_poly_newton_residual_mf(Mat mat, Vec x, Vec y);
PETSC_INTERN PetscErrorCode petsc_matvec_right_scale_poly_newton_residual_mf(Mat mat, Vec x, Vec y);
PETSC_INTERN PetscErrorCode petsc_newton(Mat mat, const PetscReal real_roots[], const PetscReal imag_roots[], PetscInt roots_size,
   Vec temp_vec, Vec temp_vec_two, Vec temp_vec_three, Vec x, Vec y);
PETSC_INTERN PetscErrorCode petsc_newton_residual(Mat mat, const PetscReal real_roots[], const PetscReal imag_roots[], PetscInt roots_size,
   Vec temp_vec, Vec temp_vec_two, Vec temp_vec_three, Vec x, Vec y);
PETSC_INTERN PetscErrorCode petsc_newton_block(mat_ctxtype *mat_ctx, Mat x_mat, Mat y_mat, Vec recip_diag, PetscBool *block_applied);
PETSC_INTERN PetscErrorCode shell_poly_block_apply(Mat shell_mat, Mat x_mat, Mat y_mat, PetscBool *block_applied);

#endif
