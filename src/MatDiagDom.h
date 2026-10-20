/*
  Prototypes for MatDiagDom.c, the C port of MatDiagDom.F90 (module matdiagdom)

  diag_dom_ratio is allocated here (PetscMalloc1, freed by the caller with
  PetscFree) where the Fortran allocated it, and is left NULL where the Fortran
  left it unallocated (the Kokkos path, where the ratios stay on the device)
*/

#if !defined(MATDIAGDOM_H)
#define MATDIAGDOM_H

#include <petscmat.h>

PETSC_INTERN PetscErrorCode MatDiagDomRatio(Mat input_mat, IS is_fine, const int *cf_markers_local, PetscReal **diag_dom_ratio, PetscReal *max_dd_ratio_achieved);
PETSC_INTERN PetscErrorCode MatDiagDomRatio_cpu(Mat input_mat, IS is_fine, const int *cf_markers_local, PetscReal **diag_dom_ratio, PetscReal *max_dd_ratio_achieved);

#endif
