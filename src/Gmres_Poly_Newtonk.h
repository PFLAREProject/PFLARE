/*
  Prototype of the Kokkos kernel in Gmres_Poly_Newtonk.kokkos.cxx that Gmres_Poly_Newton.c calls

  Gmres_Poly_Newtonk.kokkos.cxx includes this header too, so the compiler checks
  that it matches the definition
*/

#if !defined(GMRES_POLY_NEWTONK_H)
#define GMRES_POLY_NEWTONK_H

#include <petscmat.h>

#if defined(PETSC_HAVE_KOKKOS)
PETSC_INTERN void mat_mult_powers_share_sparsity_newton_kokkos(Mat *input_mat, Mat *sparsity_mat, Mat *prod_save_mat,
               const int prod_save_exists_int, const int num_terms, const int poly_sparsity_order,
               PetscReal *coefficients, int *status_output, const int output_first_complex_int,
               const int reuse_int_reuse_mat, Mat *reuse_mat, Mat *output_mat);
#endif

#endif
