/*
  Prototypes for Timers.c, the C port of Timers.F90 (module timers)
*/

#if !defined(TIMERS_H)
#define TIMERS_H

#include <petscsys.h>

PETSC_INTERN PetscErrorCode print_timers(void);
PETSC_INTERN PetscErrorCode wall_time(double *wall_time);
PETSC_INTERN PetscErrorCode timer_start(int id);
PETSC_INTERN PetscErrorCode timer_finish(int id);
PETSC_INTERN PetscErrorCode timer_reset(void);
PETSC_INTERN PetscErrorCode timer_clear(int id);
PETSC_INTERN PetscErrorCode timer_time(int id, PetscReal *timer_time);

#endif
