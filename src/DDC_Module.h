/*
  Prototypes for DDC_Module.c, the C port of DDC_Module.F90 (module ddc_module)

  The optional arguments of the Fortran are NULL when absent (Aff is a Mat and
  diag_dom_ratio_random an array, both already pointers). The CF markers are
  int arrays (the Fortran integer) with one entry per local row of input_mat
*/

#if !defined(DDC_MODULE_H)
#define DDC_MODULE_H

#include <petscmat.h>

PETSC_INTERN PetscErrorCode ddc(Mat input_mat, IS is_fine, PetscReal fraction_swap, PetscReal *max_dd_ratio, int *cf_markers_local);
PETSC_INTERN PetscErrorCode ddc_cpu(Mat input_mat, IS is_fine, PetscReal fraction_swap, PetscReal max_dd_ratio, PetscReal max_dd_ratio_achieved,
   const PetscReal *diag_dom_ratio, int *cf_markers_local, Mat Aff, const PetscReal *diag_dom_ratio_random);

#endif
