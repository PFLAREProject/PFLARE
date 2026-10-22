/*
  Prototypes for Aggregation.c, the C port of Aggregation.F90 (module aggregation)
*/

#if !defined(AGGREGATION_H)
#define AGGREGATION_H

#include <petscmat.h>

PETSC_INTERN PetscErrorCode generate_serial_aggregation(Mat strength_mat, int **cf_markers, PetscInt **aggregates);

#endif
