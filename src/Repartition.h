/*
  Prototypes for Repartition.c, the C port of Repartition.F90 (module repartition)
*/

#if !defined(REPARTITION_H)
#define REPARTITION_H

#include <petscmat.h>

PETSC_INTERN PetscErrorCode compute_mat_ratio_local_nonlocal_nnzs(Mat input_mat, PetscInt no_active_cores, PetscReal *ratio);
PETSC_INTERN PetscErrorCode calculate_repartition(Mat input_mat, PetscInt proc_stride, PetscInt *no_active_cores, PetscBool simple, IS *index);
PETSC_INTERN PetscErrorCode MatMPICreateNonemptySubcomm(Mat input_mat, PetscBool *on_subcomm, Mat *output_mat);

// The helpers that used to be in C_PETSc_Routines.c
PETSC_INTERN PetscErrorCode MatPartitioning_c(Mat *adj, PetscInt new_size, PetscInt *proc_stride, IS *index);
PETSC_INTERN PetscErrorCode GenerateIS_ProcAgglomeration_c(PetscInt proc_stride, PetscInt global_size, PetscInt *local_size_reduced, PetscInt *start);
PETSC_INTERN PetscErrorCode MatMPICreateNonemptySubcomm_c(Mat *A, int *on_subcomm, Mat *B);

#endif
