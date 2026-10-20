/*
  Prototype of the Kokkos kernel in Gmres_Polyk.kokkos.cxx that Gmres_Poly.c calls

  Gmres_Polyk.kokkos.cxx includes this header too, so the compiler checks
  that it matches the definition
*/

#if !defined(GMRES_POLYK_H)
#define GMRES_POLYK_H

#include <petscmat.h>

#if defined(PETSC_HAVE_KOKKOS)
PETSC_INTERN void mat_mult_powers_share_sparsity_kokkos(Mat *input_mat, const int poly_order, const int poly_sparsity_order, PetscReal *coefficients,
               const int reuse_int_reuse_mat, Mat *reuse_mat, const int reuse_int_cmat, Mat *output_mat);
#endif

#endif
