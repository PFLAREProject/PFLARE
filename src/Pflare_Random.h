/*
  Prototype for the random number routine in Pflare_Random.c

  pflare_random_fill seeds PFLARE's random number generator with a seed that
  depends on comm_rank and fills out[0..n-1] with uniform random numbers in
  [0, 1). It is the only source of the random numbers PFLARE uses. The
  generator reproduces gfortran's random_number bitwise (see Pflare_Random.c),
  so the numbers do not depend on the compiler
*/

#if !defined(PFLARE_RANDOM_H)
#define PFLARE_RANDOM_H

#include <petscsys.h>

PETSC_INTERN void pflare_random_fill(int comm_rank, PetscInt n, PetscReal *out);

#endif
