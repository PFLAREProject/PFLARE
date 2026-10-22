/*
  Prototypes for AIR_Data_Type_Routines.c, the C port of AIR_Data_Type_Routines.F90
  (module air_data_type_routines)

  The optional keep_reuse of reset_air_data is a pointer that is NULL when absent
*/

#if !defined(AIR_DATA_TYPE_ROUTINES_H)
#define AIR_DATA_TYPE_ROUTINES_H

#include <petscmat.h>
#include "pflareimpl.h"

PETSC_INTERN PetscErrorCode create_air_data(air_multigrid_data *air_data);
PETSC_INTERN PetscErrorCode reset_air_data(air_multigrid_data *air_data, const PetscBool *keep_reuse);
PETSC_INTERN PetscErrorCode destroy_air_data(air_multigrid_data *air_data);

#endif
