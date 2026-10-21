/*
  C port of PCAIR_Shell.F90 (module pcair_shell)

  This used to be the PCShell that PCAIR.c wrapped. The shell is gone: the
  pc_air_multigrid_data struct of pflareimpl.h is now the pc->data of the PCAIR
  itself, and the routines below that were the PCShell callbacks take that PCAIR
  and are called directly by the PC operations in PCAIR.c. The setup logic of
  PCSetUp_AIR_Shell (which read the structure flag and whether the setup has been
  called from the PC) and the creation of the underlying PCMG moved into
  PCSetUp_AIR_c and PCCreate_AIR
*/

#include <petsc/private/pcimpl.h>
#include "pflareimpl.h"
#include "AIR_Data_Type_Routines.h"
#include "FC_Smooth_Block.h"
#include "PCAIR_Shell.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode destroy_pc_air_data(pc_air_multigrid_data *pc_air_data)
{
   PetscFunctionBegin;

   PetscCall(destroy_air_data(&pc_air_data->air_data));
   // Destroy the underlying pcmg
   PetscCall(PCDestroy(&pc_air_data->pcmg));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode reset_pc_air_data(pc_air_multigrid_data *pc_air_data)
{
   PetscFunctionBegin;

   // Reset the air data
   PetscCall(reset_air_data(&pc_air_data->air_data, NULL));
   // We reset the underlying pcmg
   PetscCall(PCReset(pc_air_data->pcmg));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode reset_pc_air_data_reuse(pc_air_multigrid_data *pc_air_data)
{
   // ~~~~~~
   PetscBool keep_reuse = PETSC_TRUE;
   // ~~~~~~

   PetscFunctionBegin;

   // Reset the air data but keep the reuse
   PetscCall(reset_air_data(&pc_air_data->air_data, &keep_reuse));
   // We reset the underlying pcmg
   PetscCall(PCReset(pc_air_data->pcmg));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode PCApply_AIR_Shell(PC pc, Vec x, Vec y)
{
   // Apply our PCAIR

   // ~~~~~~
   pc_air_multigrid_data *pc_air_data = NULL;
   // ~~~~~~

   PetscFunctionBegin;

   // Get the PC context
   pc_air_data = (pc_air_multigrid_data *)pc->data;
   // Just apply the pcmg
   PetscCall(PCApply(pc_air_data->pcmg, x, y));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode PCApplyTranspose_AIR_Shell(PC pc, Vec x, Vec y)
{
   // Apply the transpose of our PCAIR
   // Just like the forward apply we hand this straight to the pcmg - petsc's
   // transposed cycles are the transpose of the forward ones, given the
   // transposed smoother and transposed residual setup_air_pcmg registers

   // ~~~~~~
   pc_air_multigrid_data *pc_air_data = NULL;
   // ~~~~~~

   PetscFunctionBegin;

   // Get the PC context
   pc_air_data = (pc_air_multigrid_data *)pc->data;
   // Just apply the transpose of the pcmg
   PetscCall(PCApplyTranspose(pc_air_data->pcmg, x, y));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode PCDestroy_AIR_Shell(PC pc)
{
   // Destroy our PCAIR

   // ~~~~~~
   pc_air_multigrid_data *pc_air_data = NULL;
   // ~~~~~~

   PetscFunctionBegin;

   // Get the PC context
   pc_air_data = (pc_air_multigrid_data *)pc->data;
   // Destroy the air data
   PetscCall(destroy_pc_air_data(pc_air_data));
   // Destroy the data structure
   PetscCall(PetscFree(pc->data));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode PCReset_AIR_Shell(PC pc)
{
   // Reset our PCAIR

   // ~~~~~~
   pc_air_multigrid_data *pc_air_data = NULL;
   // ~~~~~~

   PetscFunctionBegin;

   // Get the PC context
   pc_air_data = (pc_air_multigrid_data *)pc->data;
   // Destroy the air data
   PetscCall(reset_pc_air_data(pc_air_data));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode PCView_AIR_Shell(PC pc, PetscViewer viewer)
{
   // Views our PCAIR

   // ~~~~~~
   pc_air_multigrid_data *pc_air_data = NULL;
   // ~~~~~~

   PetscFunctionBegin;

   // Get the PC context
   pc_air_data = (pc_air_multigrid_data *)pc->data;
   // Just call the pcmg view
   PetscCall(PCView(pc_air_data->pcmg, viewer));

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode pcair_shell_block_matapply(PC pc, Mat X, Mat Y, int *applied)
{
   // Applies the air multigrid to a whole dense block of right hand sides
   // The pc handed in is our PCAIR (which must already have been set up, as
   // this doesn't go through PCApply)
   // applied comes back as 0 if we couldn't do a block apply, in which case
   // the caller has to apply column by column instead
   // Any petsc error is returned, where the Fortran handed it back in an argument

   // ~~~~~~~~
   pc_air_multigrid_data *pc_air_data = NULL;
   // ~~~~~~~~

   PetscFunctionBegin;

   *applied = 0;
   pc_air_data = (pc_air_multigrid_data *)pc->data;

   // Defensive - the hierarchy hasn't been built so we have nothing to apply
   if (pc_air_data->air_data.no_levels == -1) PetscFunctionReturn(PETSC_SUCCESS);
   if (!pc_air_data->pcmg) PetscFunctionReturn(PETSC_SUCCESS);

   // Build the dense scratch the block smooths need - we only know the number
   // of columns (and the type of block) once we get here
   PetscCall(ensure_air_block_temps(&pc_air_data->air_data, X));

   // PCMG (or the single level PCMAT/PCJACOBI) does the rest
   PetscCall(PCMatApply(pc_air_data->pcmg, X, Y));

   *applied = 1;

   PetscFunctionReturn(PETSC_SUCCESS);
}
