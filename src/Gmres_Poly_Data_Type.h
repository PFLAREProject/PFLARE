/*
  Prototypes for Gmres_Poly_Data_Type.c, the C port of Gmres_Poly_Data_Type.F90
  (module gmres_poly_data_type)

  The gmres_poly_data struct is in pflareimpl.h
*/

#if !defined(GMRES_POLY_DATA_TYPE_H)
#define GMRES_POLY_DATA_TYPE_H

#include <petscmat.h>
#include "pflareimpl.h"

PETSC_INTERN PetscErrorCode create_poly_data(gmres_poly_data *poly_data);
PETSC_INTERN PetscErrorCode poly_data_buffers(gmres_poly_data *poly_data, tsqr_buffers **buffers);
PETSC_INTERN PetscErrorCode reset_poly_data(gmres_poly_data *poly_data);
PETSC_INTERN PetscErrorCode destroy_poly_data(gmres_poly_data *poly_data);

#endif
