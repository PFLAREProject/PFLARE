/*
  Prototypes of the Kokkos kernels in PETSc_Helperk.kokkos.cxx that PETSc_Helper.c calls

  PETSc_Helperk.kokkos.cxx includes this header too, so the compiler checks
  that these match the definitions
*/

#if !defined(PETSC_HELPERK_H)
#define PETSC_HELPERK_H

#include <petscmat.h>

#if defined(PETSC_HAVE_KOKKOS)
PETSC_INTERN void remove_small_from_sparse_kokkos(Mat *input_mat, const PetscReal tol, Mat *output_mat,
   const int relative_max_row_tolerance_int, const int lump_int,
   const int allow_drop_diagonal_int, const int allow_diag_strength_int);
PETSC_INTERN void remove_from_sparse_match_kokkos(Mat *input_mat, Mat *output_mat, const int lump_int, const int alpha_int, const PetscReal alpha);
PETSC_INTERN void MatSetAllValues_kokkos(Mat *input_mat, PetscScalar val);
PETSC_INTERN void mat_duplicate_copy_plus_diag_kokkos(Mat *input_mat, const int reuse_int, Mat *output_mat);
PETSC_INTERN void MatAXPY_kokkos(Mat *Y, PetscScalar alpha, Mat *X);
PETSC_INTERN void MatCreateSubMatrix_kokkos(Mat *input_mat, IS *is_row, IS *is_col,
   const int reuse_int, Mat *output_mat,
   void *kokkos_is_views_handle,
   const int our_level, const int is_row_fine_int, const int is_col_fine_int);
#endif

#endif
