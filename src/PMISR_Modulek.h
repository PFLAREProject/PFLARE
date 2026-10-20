/*
  Prototype of the Kokkos kernel in PMISR_Modulek.kokkos.cxx that PMISR_Module.c calls

  PMISR_Modulek.kokkos.cxx includes this header too, so the compiler checks
  that it matches the definition
*/

#if !defined(PMISR_MODULEK_H)
#define PMISR_MODULEK_H

#include <petscmat.h>

#if defined(PETSC_HAVE_KOKKOS)
PETSC_INTERN void pmisr_kokkos(Mat *strength_mat, const int max_luby_steps, const int pmis_int, PetscReal *measure_local, const int zero_measure_c_point_int);
#endif

#endif
