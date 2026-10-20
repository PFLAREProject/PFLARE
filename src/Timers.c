/*
  C port of Timers.F90 (module timers)

  The timer ids (PFLARE_TIMER_ID_* in pflare_parameters.h) are 1-based, as in
  the Fortran, and the routines take them unchanged; the arrays are indexed with
  id - 1
*/

// Get MPI types from PETSc as it can control whether using mpi or mpif08
#include <petscsys.h>
#include <petsctime.h>
#include "pflareimpl.h"
#include "Timers.h"

// -------------------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------------------------------------
// Stores a globally accessible array of times along with some ids
// Basically a copy of fluidity's tictoc.F90 and Timers.F90
// Makes it easy to time things across different routines
// -------------------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------------------------------------

#define MAX_TIMER_ID 1024
static PetscReal starttimers[MAX_TIMER_ID] = {0}, totaltimers[MAX_TIMER_ID] = {0};

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode print_timers(void)
{
   PetscReal time;

   // Print the values of the timers
   // Printed with explicit formats rather than the list-directed print of the
   // Fortran, keeping the labels that tools/parse_pflare_output.py reads

   PetscFunctionBegin;

   PetscCall(timer_time(PFLARE_TIMER_ID_AIR_COARSEN, &time));
   PetscCall(PetscPrintf(PETSC_COMM_SELF, " coarsen time     : %.15g\n", (double)time));
   PetscCall(timer_time(PFLARE_TIMER_ID_AIR_EXTRACT, &time));
   PetscCall(PetscPrintf(PETSC_COMM_SELF, " extract time     : %.15g\n", (double)time));
   PetscCall(timer_time(PFLARE_TIMER_ID_AIR_PROC_AGGLOM, &time));
   PetscCall(PetscPrintf(PETSC_COMM_SELF, " proc agglom time : %.15g\n", (double)time));
   PetscCall(timer_time(PFLARE_TIMER_ID_AIR_INVERSE, &time));
   PetscCall(PetscPrintf(PETSC_COMM_SELF, " inverse time     : %.15g\n", (double)time));
   PetscCall(timer_time(PFLARE_TIMER_ID_AIR_RESTRICT, &time));
   PetscCall(PetscPrintf(PETSC_COMM_SELF, " restrict time    : %.15g\n", (double)time));
   PetscCall(timer_time(PFLARE_TIMER_ID_AIR_PROLONG, &time));
   PetscCall(PetscPrintf(PETSC_COMM_SELF, " prolong time     : %.15g\n", (double)time));
   PetscCall(timer_time(PFLARE_TIMER_ID_AIR_CONSTRAIN, &time));
   PetscCall(PetscPrintf(PETSC_COMM_SELF, " constrain time   : %.15g\n", (double)time));
   PetscCall(timer_time(PFLARE_TIMER_ID_AIR_RAP, &time));
   PetscCall(PetscPrintf(PETSC_COMM_SELF, " rap time         : %.15g\n", (double)time));
   PetscCall(timer_time(PFLARE_TIMER_ID_AIR_IDENTITY, &time));
   PetscCall(PetscPrintf(PETSC_COMM_SELF, " identity time    : %.15g\n", (double)time));
   PetscCall(timer_time(PFLARE_TIMER_ID_AIR_DROP, &time));
   PetscCall(PetscPrintf(PETSC_COMM_SELF, " drop time        : %.15g\n", (double)time));
   PetscCall(timer_time(PFLARE_TIMER_ID_AIR_TRUNCATE, &time));
   PetscCall(PetscPrintf(PETSC_COMM_SELF, " truncate time    : %.15g\n", (double)time));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode wall_time(double *wall_time)
{
   // This function returns the wall clock time from when the
   // simulation started.

   static PetscBool started = PETSC_FALSE;
   static double wall_time0;
   PetscLogDouble current_time;

   PetscFunctionBegin;

   // Use PetscTime rather than MPI_Wtime - MPIUNI (PETSc built without MPI) does not
   // provide a typed Fortran MPI_Wtime, whereas PetscTime works in serial and parallel
   PetscCall(PetscTime(&current_time));
   *wall_time = current_time;
   if (!started) {
      wall_time0 = *wall_time;
      *wall_time = 0.0;
      started = PETSC_TRUE;
   } else {
      *wall_time = *wall_time - wall_time0;
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PETSC_INTERN PetscErrorCode timer_start(int id)
{
   double time;

   PetscFunctionBegin;

   // Explicit conversion: wall_time() is c_double, timer storage is PetscReal
   PetscCall(wall_time(&time));
   starttimers[id - 1] = (PetscReal)time;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PETSC_INTERN PetscErrorCode timer_finish(int id)
{
   PetscReal finish_time;
   double time;

   PetscFunctionBegin;

   // Explicit conversion: wall_time() is c_double, timer storage is PetscReal
   PetscCall(wall_time(&time));
   finish_time = (PetscReal)time;
   totaltimers[id - 1] = totaltimers[id - 1] + (finish_time - starttimers[id - 1]);

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PETSC_INTERN PetscErrorCode timer_reset(void)
{
   PetscFunctionBegin;
   for (int i = 0; i < MAX_TIMER_ID; i++) {
      starttimers[i] = 0.0;
      totaltimers[i] = 0.0;
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PETSC_INTERN PetscErrorCode timer_clear(int id)
{
   PetscFunctionBegin;

   starttimers[id - 1] = 0.0;
   totaltimers[id - 1] = 0.0;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PETSC_INTERN PetscErrorCode timer_time(int id, PetscReal *timer_time)
{
   PetscFunctionBegin;

   *timer_time = totaltimers[id - 1];

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
