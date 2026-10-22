/*
  Prototypes for MatDiagDomSubmatrix.c, the C port of MatDiagDomSubmatrix.F90
  (module matdiagdomsubmatrix)

  compute_diag_dom_submatrix is called compute_diag_dom_submatrix_c here, as the
  public C API in include/pflare.h already has a routine of that name (in PCAIR.c,
  which calls this one)
*/

#if !defined(MATDIAGDOMSUBMATRIX_H)
#define MATDIAGDOMSUBMATRIX_H

#include <petscmat.h>

PETSC_INTERN PetscErrorCode compute_diag_dom_submatrix_c(Mat input_mat, PetscReal max_dd_ratio, Mat *output_mat);

#endif
