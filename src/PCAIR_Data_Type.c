/*
  C port of PCAIR_Data_Type.F90 (module pcair_data_type)

  The Fortran module only defined the context type of our pcshells, which is the
  pc_air_multigrid_data struct in pflareimpl.h. create_pc_air_multigrid_data does
  what the Fortran allocate of one did, giving the defaults of the type; it is
  freed with PetscFree (the Fortran deallocate)
*/

#include <petscksp.h>
#include "pflareimpl.h"
#include "AIR_Data_Type.h"
#include "PCAIR_Data_Type.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode create_pc_air_multigrid_data(pc_air_multigrid_data **pc_air_data)
{
   // Allocates the context of our pcshells with the defaults of the Fortran type

   PetscFunctionBegin;

   PetscCall(PetscNew(pc_air_data));
   PetscCall(create_air_multigrid_data(&(*pc_air_data)->air_data));
   // AIR under the hood is just a PCMG object, created in create_pc_air_shell
   (*pc_air_data)->pcmg = NULL;

   PetscFunctionReturn(PETSC_SUCCESS);
}
