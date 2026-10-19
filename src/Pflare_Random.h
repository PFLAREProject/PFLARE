/*
  C prototype for the random number routine in Pflare_Random.F90

  pflare_random_fill seeds the Fortran intrinsic random number generator with
  a seed that depends on comm_rank and fills out[0..n-1] with uniform random
  numbers in [0, 1). It is the only source of the random numbers PFLARE uses,
  so C code that calls it draws exactly the same sequence as the Fortran.
*/

#if !defined(PFLARE_RANDOM_H)
#define PFLARE_RANDOM_H

#include <petscsys.h>

PETSC_INTERN void pflare_random_fill(int comm_rank, PetscInt n, PetscReal *out);

#endif
