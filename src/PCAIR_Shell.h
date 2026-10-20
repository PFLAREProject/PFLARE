/*
  Prototypes for PCAIR_Shell.c, the C port of PCAIR_Shell.F90 (module pcair_shell)
*/

#if !defined(PCAIR_SHELL_H)
#define PCAIR_SHELL_H

#include <petscksp.h>
#include "pflareimpl.h"

PETSC_INTERN PetscErrorCode create_pc_air_shell(pc_air_multigrid_data *pc_air_data, PC pc);
PETSC_INTERN PetscErrorCode destroy_pc_air_data(pc_air_multigrid_data *pc_air_data);
PETSC_INTERN PetscErrorCode reset_pc_air_data(pc_air_multigrid_data *pc_air_data);
PETSC_INTERN PetscErrorCode reset_pc_air_data_reuse(pc_air_multigrid_data *pc_air_data);
PETSC_INTERN PetscErrorCode PCSetUp_AIR_Shell(PC pc);
PETSC_INTERN PetscErrorCode PCApply_AIR_Shell(PC pc, Vec x, Vec y);
PETSC_INTERN PetscErrorCode PCApplyTranspose_AIR_Shell(PC pc, Vec x, Vec y);
PETSC_INTERN PetscErrorCode PCDestroy_AIR_Shell(PC pc);
PETSC_INTERN PetscErrorCode PCReset_AIR_Shell(PC pc);
PETSC_INTERN PetscErrorCode PCView_AIR_Shell(PC pc, PetscViewer viewer);

// The entry points src/PCAIR.c used to reach through C_Fortran_Bindings.F90
PETSC_INTERN PetscErrorCode create_pc_air_data_c(pc_air_multigrid_data **pc_air_data);
PETSC_INTERN PetscErrorCode PCReset_AIR_Shell_c(PC *pc);
PETSC_INTERN PetscErrorCode create_pc_air_shell_c(pc_air_multigrid_data **pc_air_data, PC *pc);
PETSC_INTERN PetscErrorCode pcair_shell_get_pcmg_c(PC *pc, PC *pcmg);
PETSC_INTERN PetscErrorCode pcair_shell_block_matapply_c(PC *pc, Mat *X, Mat *Y, int *applied);

#endif
