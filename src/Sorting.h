/*
  Prototypes for Sorting.c, the C port of Sorting.F90 (module sorting)
*/

#if !defined(SORTING_H)
#define SORTING_H

#include <petscsys.h>

PETSC_INTERN PetscErrorCode sorted_unique(PetscInt array[], PetscInt *n);
PETSC_INTERN PetscErrorCode intersect_pre_sorted_indices_only(const PetscInt array1[], PetscInt size_array1, const PetscInt array2[], PetscInt size_array2, int indices_array1[], int indices_array2[], int *intersect_count);
PETSC_INTERN PetscErrorCode merge_pre_sorted(const PetscInt array1[], PetscInt size_array1, const PetscInt array2[], PetscInt size_array2, PetscInt sorted_array[]);

#endif
