/*
  Prototypes for AIR_Data_Type.c, the C port of AIR_Data_Type.F90
  (module air_data_type)

  The types of the module (air_options, air_reuse_data, air_multigrid_data and
  the small array holders) and the REUSE_MAT_ACTIVE/REUSE_IS_ACTIVE tables are
  in pflareimpl.h
*/

#if !defined(AIR_DATA_TYPE_H)
#define AIR_DATA_TYPE_H

#include <petscmat.h>
#include "pflareimpl.h"

PETSC_INTERN PetscErrorCode create_air_options(air_options *options);
PETSC_INTERN PetscErrorCode create_air_multigrid_data(air_multigrid_data *air_data);

#endif
