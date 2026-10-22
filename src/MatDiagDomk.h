/*
  Prototype of the Kokkos kernel in MatDiagDomk.kokkos.cxx that MatDiagDom.c calls

  MatDiagDomk.kokkos.cxx includes this header too, so the compiler checks
  that it matches the definition
*/

#if !defined(MATDIAGDOMK_H)
#define MATDIAGDOMK_H

#include <petscmat.h>

#if defined(PETSC_HAVE_KOKKOS)
PETSC_INTERN void MatDiagDomRatio_kokkos(Mat *input_mat, PetscReal *max_dd_ratio_achieved, PetscInt *local_rows_aff);
#endif

#endif
