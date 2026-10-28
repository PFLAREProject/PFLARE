/*
  Prototype of the Kokkos kernel in DDC_Modulek.kokkos.cxx that DDC_Module.c calls

  DDC_Modulek.kokkos.cxx includes this header too, so the compiler checks
  that it matches the definition
*/

#if !defined(DDC_MODULEK_H)
#define DDC_MODULEK_H

#include <petscmat.h>

#if defined(PETSC_HAVE_KOKKOS)
PETSC_INTERN void ddc_kokkos(Mat *input_mat, const PetscReal fraction_swap, const PetscReal max_dd_ratio, const PetscReal max_dd_ratio_achieved, Mat *aff, PetscReal *random_numbers);
#endif

#endif
