/*
  Prototypes for CR_Splitting.c, the C port of CR_Splitting.F90 (module cr_splitting)

  The CF markers are int arrays (the Fortran integer) with one entry per local
  row of input_mat
*/

#if !defined(CR_SPLITTING_H)
#define CR_SPLITTING_H

#include <petscmat.h>

PETSC_INTERN PetscErrorCode cr_pass(Mat input_mat, IS is_fine, PetscReal target_cr_rate,
   int cr_inverse_type, int cr_poly_order, int cr_sparsity_order, PetscBool cr_diag_scale,
   PetscReal *cr_rate_achieved, PetscInt *n_swapped_global, int *cf_markers_local);

#endif
