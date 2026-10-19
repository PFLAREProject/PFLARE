static char help[] = "Compares PCAIR and PCPFLAREINV against their Fortran reference implementations.\n\n";

/*
  Differential test used while the Fortran core of PFLARE is ported to C.
  Built only with make PFLARE_PORT_DEBUG=1, which also compiles the Fortran
  reference implementations in src/fref. See section 6 of
  docs/dev/fortran_removal_plan.md.

  Nothing has been ported yet, so this only checks that PFLARE registers and
  then exits successfully.
*/
#include <petscksp.h>
#include "pflare.h"

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  // Register the pflare types
  PCRegister_PFLARE();

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "port compare: nothing to compare yet\n"));

  PetscCall(PetscFinalize());
  return 0;
}
