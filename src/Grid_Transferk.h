/*
  Prototypes of the Kokkos kernels in Grid_Transferk.kokkos.cxx that Grid_Transfer.c calls

  Grid_Transferk.kokkos.cxx includes this header too, so the compiler checks
  that these match the definitions
*/

#if !defined(GRID_TRANSFERK_H)
#define GRID_TRANSFERK_H

#include <petscmat.h>

#if defined(PETSC_HAVE_KOKKOS)
PETSC_INTERN void generate_one_point_with_one_entry_from_sparse_kokkos(Mat *input_mat, Mat *output_mat);
PETSC_INTERN void compute_P_from_W_kokkos(Mat *input_mat, PetscInt global_row_start, IS *is_fine,
                  IS *is_coarse, int identity_int, int reuse_int, Mat *output_mat);
PETSC_INTERN void compute_R_from_Z_kokkos(Mat *input_mat, PetscInt global_row_start, IS *is_fine,
                  IS *is_coarse, IS *orig_fine_col_indices, int identity_int, int reuse_int, int reuse_indices_int,
                  Mat *output_mat);
#endif

#endif
