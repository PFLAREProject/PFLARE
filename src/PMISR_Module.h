/*
  Prototypes for PMISR_Module.c, the C port of PMISR_Module.F90 (module pmisr_module)

  The optional arguments of the Fortran are pointers that are NULL when the
  argument is absent. The CF markers are int arrays (the Fortran integer) with
  one entry per local row, the measures PetscReal arrays
*/

#if !defined(PMISR_MODULE_H)
#define PMISR_MODULE_H

#include <petscmat.h>

PETSC_INTERN PetscErrorCode pmisr(Mat strength_mat, int max_luby_steps, PetscBool pmis, int **cf_markers_local, const PetscBool *zero_measure_c_point);
PETSC_INTERN PetscErrorCode pmisr_cpu(Mat strength_mat, int max_luby_steps, PetscBool pmis, int **cf_markers_local, const PetscBool *zero_measure_c_point);
PETSC_INTERN PetscErrorCode pmisr_existing_measure_cf_markers(Mat strength_mat, int max_luby_steps, PetscBool pmis,
   PetscReal *measure_local, int *cf_markers_local, const PetscBool *zero_measure_c_point);
PETSC_INTERN PetscErrorCode pmisr_existing_measure_implicit_transpose(Mat strength_mat, int max_luby_steps, PetscBool pmis,
   PetscReal *measure_local, int *cf_markers_local, const PetscBool *zero_measure_c_point);

#endif
