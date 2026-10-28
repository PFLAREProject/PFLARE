/*
  Prototype of the Kokkos kernel in SAI_Zk.kokkos.cxx that SAI_Z.c calls

  SAI_Zk.kokkos.cxx includes this header too, so the compiler checks
  that it matches the definition
*/

#if !defined(SAI_ZK_H)
#define SAI_ZK_H

#include <petscmat.h>

#if defined(PETSC_HAVE_KOKKOS)
PETSC_INTERN void calculate_and_build_sai_z_kokkos(Mat *A_ff, Mat *A_cf, Mat *sparsity_mat_cf,
               const int reuse_int_reuse_mat, Mat *reuse_mat, Mat *z_mat,
               const int no_approx_solve_int);
#endif

#endif
