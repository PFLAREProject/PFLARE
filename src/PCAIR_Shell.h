/*
  Prototypes for PCAIR_Shell.c, the C port of PCAIR_Shell.F90 (module pcair_shell)
  The routines that take a PC take our PCAIR, whose pc->data is the pc_air_multigrid_data
*/

#if !defined(PCAIR_SHELL_H)
#define PCAIR_SHELL_H

#include <petscksp.h>
#include "pflareimpl.h"

PETSC_INTERN PetscErrorCode destroy_pc_air_data(pc_air_multigrid_data *pc_air_data);
PETSC_INTERN PetscErrorCode reset_pc_air_data(pc_air_multigrid_data *pc_air_data);
PETSC_INTERN PetscErrorCode reset_pc_air_data_reuse(pc_air_multigrid_data *pc_air_data);
PETSC_INTERN PetscErrorCode PCApply_AIR_Shell(PC pc, Vec x, Vec y);
PETSC_INTERN PetscErrorCode PCApplyTranspose_AIR_Shell(PC pc, Vec x, Vec y);
PETSC_INTERN PetscErrorCode PCDestroy_AIR_Shell(PC pc);
PETSC_INTERN PetscErrorCode PCReset_AIR_Shell(PC pc);
PETSC_INTERN PetscErrorCode PCView_AIR_Shell(PC pc, PetscViewer viewer);
PETSC_INTERN PetscErrorCode pcair_shell_block_matapply(PC pc, Mat X, Mat Y, int *applied);

#endif
