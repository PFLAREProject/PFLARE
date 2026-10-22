/*
  Prototypes for AIR_MG_Stats.c, the C port of AIR_MG_Stats.F90 (module air_mg_stats)
*/

#if !defined(AIR_MG_STATS_H)
#define AIR_MG_STATS_H

#include <petscksp.h>
#include "pflareimpl.h"

PETSC_INTERN PetscErrorCode compute_nnzs_air_matrices(air_multigrid_data *air_data);
PETSC_INTERN PetscErrorCode compute_nnzs_v_cycle_air(air_multigrid_data *air_data, PC pcmg_input, PetscInt64 *nnzs);
PETSC_INTERN PetscErrorCode compute_stats(air_multigrid_data *air_data, PC pcmg_input, PetscReal *grid_complx, PetscReal *op_complx, PetscReal *cycle_complx,
   PetscReal *storage_complx, PetscReal *reuse_storage_complx);
PETSC_INTERN PetscErrorCode print_stats(air_multigrid_data *air_data, PC pcmg_input);

#endif
