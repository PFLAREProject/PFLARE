/*
  Private header for the C implementation of PFLARE

  Holds what is shared between the C files of the library and is not part
  of the public API in include/pflare.h
*/

#if !defined(PFLAREIMPL_H)
#define PFLAREIMPL_H

#include <petscsys.h>
#include "pflare_parameters.h"

/* Complex is intentionally unsupported */
#if defined(PETSC_USE_COMPLEX)
#error "PFLARE BLAS/LAPACK dispatch does not support complex PETSc builds"
#endif

/* ~~~~~~~~~~~~~~~~~~~~~~~~
   Private structs
   Empty for now: the Fortran derived types become structs here as the
   modules that use them are ported to C (section 4 and convention 3 of
   docs/dev/fortran_removal_plan.md)
   ~~~~~~~~~~~~~~~~~~~~~~~~ */

#endif
