/*
  C port of Gmres_Poly_Data_Type.F90 (module gmres_poly_data_type)

  The Fortran module only defined the type, which is the gmres_poly_data struct
  in pflareimpl.h. What the Fortran did through default initialisation and the
  automatic deallocation of the allocatable components is done here explicitly,
  in the create/reset/destroy trio of convention 3 of
  docs/dev/fortran_removal_plan.md. reset_poly_data and destroy_poly_data are also
  called by the Fortran AIR code (through the bind(C) mirror of the struct in
  src/fortran/Gmres_Poly_Data_Type.F90) where it used to deallocate the
  coefficients itself
*/

#include <petscmat.h>
#include "pflareimpl.h"
#include "TSQR.h"
#include "Gmres_Poly_Data_Type.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode create_poly_data(gmres_poly_data *poly_data)
{
   // Sets the defaults of the Fortran type

   PetscFunctionBegin;

   // Inverse type
   poly_data->inverse_type = 0;
   // Order of the gmres polynomial
   poly_data->gmres_poly_order = 6;
   // The sparsity order of the polynomial if assembled
   poly_data->gmres_poly_sparsity_order = 1;
   // Coefficients for the gmres polynomial
   poly_data->coefficients = NULL;
   poly_data->coefficients_rows = 0;
   poly_data->coefficients_cols = 0;
   // Asynchronous buffers for the TSQR
   PetscCall(create_tsqr_buffers(&poly_data->buffers));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode poly_data_buffers(gmres_poly_data *poly_data, tsqr_buffers **buffers)
{
   // Returns the asynchronous buffers for the TSQR, creating them if they don't exist yet
   // The Fortran type held its buffers directly, so they always existed - the
   // Fortran mirror of this struct starts with a null pointer instead

   PetscFunctionBegin;

   if (!poly_data->buffers) PetscCall(create_tsqr_buffers(&poly_data->buffers));
   *buffers = poly_data->buffers;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode reset_poly_data(gmres_poly_data *poly_data)
{
   // Frees the coefficients, if there are any

   PetscFunctionBegin;

   PetscCall(PetscFree(poly_data->coefficients));
   poly_data->coefficients_rows = 0;
   poly_data->coefficients_cols = 0;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode destroy_poly_data(gmres_poly_data *poly_data)
{
   // Frees the coefficients and the asynchronous buffers (with anything they still hold)

   PetscFunctionBegin;

   PetscCall(reset_poly_data(poly_data));
   PetscCall(destroy_tsqr_buffers(&poly_data->buffers));

   PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PFLARE_PORT_DEBUG)
// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode poly_data_get_subcomm(gmres_poly_data *poly_data, PetscBool *subcomm)
{
   // Returns whether the user has asked for the TSQR buffers to be on a subcomm
   // Only for the comparison against the Fortran reference in
   // src/fortran/Port_Interfaces_Poly.F90, which has to start the reference with the
   // same buffers

   // ~~~~~~
   tsqr_buffers *buffers = NULL;
   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(poly_data_buffers(poly_data, &buffers));
   *subcomm = buffers->subcomm;

   PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
