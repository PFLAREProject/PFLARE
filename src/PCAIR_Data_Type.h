/*
  Prototypes for PCAIR_Data_Type.c, the C port of PCAIR_Data_Type.F90
  (module pcair_data_type)

  The pc_air_multigrid_data struct is in pflareimpl.h
*/

#if !defined(PCAIR_DATA_TYPE_H)
#define PCAIR_DATA_TYPE_H

#include <petscksp.h>
#include "pflareimpl.h"

PETSC_INTERN PetscErrorCode create_pc_air_multigrid_data(pc_air_multigrid_data **pc_air_data);

#endif
