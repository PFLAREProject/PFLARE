/*
  Prototypes for PETSc_Helper.c, the C port of PETSc_Helper.F90 (module petsc_helper)

  The optional arguments of the Fortran are pointers that are NULL when the
  argument is absent
*/

#if !defined(PETSC_HELPER_H)
#define PETSC_HELPER_H

#include <petscmat.h>

PETSC_INTERN PetscBool kokkos_debug(void);
PETSC_INTERN PetscErrorCode destroy_matrix_reuse(Mat *mat, Mat **submatrices);
PETSC_INTERN PetscErrorCode remove_small_from_sparse(Mat input_mat, PetscReal tol, Mat *output_mat, const int *relative_max_row_tol_int,
   const PetscBool *lump, const int *drop_diagonal_int, const int *diag_strength_int);
PETSC_INTERN PetscErrorCode remove_small_from_sparse_cpu(Mat input_mat, PetscReal tol, Mat *output_mat, const int *relative_max_row_tol_int,
   const PetscBool *lump, const int *drop_diagonal_int, const int *diag_strength_int);
PETSC_INTERN PetscErrorCode remove_from_sparse_match_no_lump(Mat input_mat, Mat output_mat, const PetscReal *alpha);
PETSC_INTERN PetscErrorCode remove_from_sparse_match_c(Mat input_mat, Mat output_mat, const PetscBool *lump, const PetscReal *alpha);
PETSC_INTERN PetscErrorCode remove_from_sparse_match_cpu(Mat input_mat, Mat output_mat, const PetscBool *lump, const PetscReal *alpha);
PETSC_INTERN PetscErrorCode MatSetAllValues(Mat input_mat, PetscScalar val);
PETSC_INTERN PetscErrorCode MatSetAllValues_cpu(Mat input_mat, PetscScalar val);
PETSC_INTERN PetscErrorCode mat_duplicate_copy_plus_diag(Mat input_mat, PetscBool reuse, Mat *output_mat);
PETSC_INTERN PetscErrorCode mat_duplicate_copy_plus_diag_cpu(Mat input_mat, PetscBool reuse, Mat *output_mat);
PETSC_INTERN PetscErrorCode MatAXPYWrapper(Mat y_mat, PetscScalar alpha, Mat x_mat);
PETSC_INTERN PetscErrorCode MatCreateSubMatrixWrapper(Mat input_mat, IS is_row, IS is_col, MatReuse reuse, Mat *output_mat,
   const int *our_level, const PetscBool *is_row_fine, const PetscBool *is_col_fine, void *kokkos_is_views_handle);
PETSC_INTERN PetscErrorCode generate_identity(Mat input_mat, Mat *output_mat);
PETSC_INTERN PetscErrorCode generate_identity_rect(Mat full_mat, IS rect_indices, Mat *output_mat);
PETSC_INTERN PetscErrorCode generate_identity_is(Mat input_mat, IS indices, Mat *output_mat);
PETSC_INTERN PetscErrorCode get_nnzs_petsc_sparse(Mat input_mat, PetscInt64 *nnzs);
PETSC_INTERN PetscErrorCode MatGetNNZs_both(Mat input_mat, PetscInt *nnzs_local, PetscInt *nnzs_nonlocal);
PETSC_INTERN PetscErrorCode svd(const PetscScalar input[], PetscInt input_rows, PetscInt input_cols, PetscScalar U[], PetscReal sigma[], PetscScalar VT[]);
PETSC_INTERN PetscErrorCode pseudo_inv(const PetscScalar input[], PetscInt input_rows, PetscInt input_cols, PetscScalar output[]);
PETSC_INTERN PetscErrorCode mat_mat_symbolic(Mat A, Mat B, Mat *C);

// Prints to the C stdout like the Fortran print * did, see pflare_print in PETSc_Helper.c
PETSC_INTERN PetscErrorCode pflare_print(const char format[], ...) PETSC_ATTRIBUTE_FORMAT(1, 2);

#endif
