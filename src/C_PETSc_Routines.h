/*
  Prototypes for C_PETSc_Routines.c
*/

#if !defined(C_PETSC_ROUTINES_H)
#define C_PETSC_ROUTINES_H

#include <petscksp.h>

PETSC_INTERN PetscErrorCode MatGetDiagonalOnly_c(Mat *A, int *diag_only);
PETSC_INTERN PetscErrorCode PetscFree_c(void **ptr);
#if defined(PFLARE_PORT_DEBUG)
// Only called by the reference Fortran in src/fref
PETSC_INTERN void c_PCGetStructureFlag(PC *pc, int *flag);
PETSC_INTERN void PCGetSetupCalled_c(PC *pc, PetscInt *setupcalled);
PETSC_INTERN void PCMarkNotSetUp_c(PC *pc);
#endif

#endif
