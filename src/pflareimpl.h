/*
  Private header for the C implementation of PFLARE

  Holds what is shared between the C files of the library and is not part
  of the public API in include/pflare.h
*/

#if !defined(PFLAREIMPL_H)
#define PFLAREIMPL_H

#include <petscmat.h>
#include "pflare_parameters.h"

/* Complex is intentionally unsupported */
#if defined(PETSC_USE_COMPLEX)
#error "PFLARE BLAS/LAPACK dispatch does not support complex PETSc builds"
#endif

/* ~~~~~~~~~~~~~~~~~~~~~~~~
   Private structs
   The Fortran derived types become structs here as the modules that use
   them are ported to C (section 4 and convention 3 of
   docs/dev/fortran_removal_plan.md)
   ~~~~~~~~~~~~~~~~~~~~~~~~ */

// ~~~~~~~~
// Stores data for the asynchronous comms (TSQR.c, tsqr_buffers in TSQR.F90)
// Created with create_tsqr_buffers (the defaults of the Fortran type) and
// destroyed with destroy_tsqr_buffers
// ~~~~~~~~
typedef struct {
   MPI_Request request;
   // Each is NULL when not allocated, both hold n_size * n_size + 1 entries
   // (n_size is in the first entry)
   PetscReal *R_buffer_send, *R_buffer_receive;
   // In case this comms request is done on a matrix on a subcomm, we
   // need to keep a pointer to it
   Mat matrix;
   // Has the user asked us to be on a subcomm
   PetscBool subcomm;
   // Did we actually end up on a subcomm
   PetscBool on_subcomm;
   int proc_stride;
} tsqr_buffers;

#endif
