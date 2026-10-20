/*
  Prototypes for AIR_MG_Setup.c, the C port of AIR_MG_Setup.F90 (module air_mg_setup)
*/

#if !defined(AIR_MG_SETUP_H)
#define AIR_MG_SETUP_H

#include <petscksp.h>
#include "pflareimpl.h"

PETSC_INTERN PetscErrorCode setup_air_pcmg(Mat amat, Mat pmat, air_multigrid_data *air_data, PC pcmg_input);

#endif
