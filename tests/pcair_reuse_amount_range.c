static char help[] = "Checks PCAIR rejects a reuse amount outside 1, 2 or 3.\n\n";

/*
  The reuse amount is used as an index into the constant reuse tables in
  AIR_Data_Type.F90, so anything other than 1, 2 or 3 would read out of bounds.
  This checks that PCAIRSetReuseAmount returns PETSC_ERR_ARG_OUTOFRANGE for a
  bad value and leaves the stored amount alone.

  The error handler is swapped for PetscReturnErrorHandler around the calls
  that are expected to fail, so this runs fine with -on_error_abort.
  -pc_air_reuse_amount isn't checked here: an error inside PCSetFromOptions
  returns before PetscOptionsEnd, which leaks under -malloc_dump.
*/

#include <petscksp.h>
#include "pflare.h"

// Check a set of the reuse amount fails with an out of range error and
// leaves the stored value unchanged
static PetscErrorCode CheckSetFails(PC pc, PetscInt amount)
{
  PetscErrorCode ierr;
  PetscInt       before, after;

  PetscFunctionBeginUser;
  PetscCall(PCAIRGetReuseAmount(pc, &before));
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = PCAIRSetReuseAmount(pc, amount);
  PetscCall(PetscPopErrorHandler());
  PetscCheck(ierr == PETSC_ERR_ARG_OUTOFRANGE, PETSC_COMM_WORLD, PETSC_ERR_LIB,
             "PCAIRSetReuseAmount(%" PetscInt_FMT ") returned error %d, expected PETSC_ERR_ARG_OUTOFRANGE", amount, (int)ierr);
  PetscCall(PCAIRGetReuseAmount(pc, &after));
  PetscCheck(after == before, PETSC_COMM_WORLD, PETSC_ERR_LIB,
             "Reuse amount changed from %" PetscInt_FMT " to %" PetscInt_FMT " after a failed set", before, after);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  PC       pc;
  PetscInt amount;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  // Register the pflare types
  PCRegister_PFLARE();

  PetscCall(PCCreate(PETSC_COMM_WORLD, &pc));
  PetscCall(PCSetType(pc, PCAIR));

  // The valid amounts round trip
  for (amount = 1; amount <= 3; amount++) {
    PetscInt got;
    PetscCall(PCAIRSetReuseAmount(pc, amount));
    PetscCall(PCAIRGetReuseAmount(pc, &got));
    PetscCheck(got == amount, PETSC_COMM_WORLD, PETSC_ERR_LIB, "Set reuse amount %" PetscInt_FMT " but got %" PetscInt_FMT, amount, got);
  }

  // Anything else is rejected
  PetscCall(CheckSetFails(pc, 0));
  PetscCall(CheckSetFails(pc, 4));
  PetscCall(CheckSetFails(pc, -1));

  PetscCall(PCDestroy(&pc));
  PetscCall(PetscFinalize());
  return 0;
}
