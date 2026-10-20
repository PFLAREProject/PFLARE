/*
  C port of PCAIR_Shell.F90 (module pcair_shell)

  The PCShell that PCAIR.c wraps: its context is the pc_air_multigrid_data struct
  of pflareimpl.h, and the setup reads the structure flag and whether the setup
  has been called straight from the PC (pc->flag, pc->setupcalled), which the
  Fortran got through C helpers
*/

#include <petsc/private/pcimpl.h>
#include "pflareimpl.h"
#include "AIR_MG_Setup.h"
#include "AIR_Data_Type_Routines.h"
#include "PCAIR_Shell.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode create_pc_air_shell(pc_air_multigrid_data *pc_air_data, PC pc)
{
   // Sets the PCShell that knows how to setup itself up and re-use
   // pc should have been obtained from a call to KSPGetPC
   // pc_air_data should have been created outside this routine

   // ~~~~~~
   MPI_Comm MPI_COMM_MATRIX;
   // ~~~~~~

   PetscFunctionBegin;

   // Set the pc as a shell
   PetscCall(PCSetType(pc, PCSHELL));
   // Set the pc_air_data
   PetscCall(PCShellSetContext(pc, pc_air_data));
   // Set the apply routine
   PetscCall(PCShellSetApply(pc, PCApply_AIR_Shell));
   // Set the transposed apply routine
   PetscCall(PCShellSetApplyTranspose(pc, PCApplyTranspose_AIR_Shell));
   // Set the destroy routine
   PetscCall(PCShellSetDestroy(pc, PCDestroy_AIR_Shell));
   // Set the setup routine
   PetscCall(PCShellSetSetUp(pc, PCSetUp_AIR_Shell));
   // Set the view routine
   PetscCall(PCShellSetView(pc, PCView_AIR_Shell));
   // Annoyingly there is no ability to set a reset routine
   // in fortran, so if we want to call the equivalent of a
   // reset on our pcshell we need to manually call reset_pc_air_data

   // Create our underlying PCMG object
   PetscCall(PetscObjectGetComm((PetscObject)pc, &MPI_COMM_MATRIX));
   PetscCall(PCCreate(MPI_COMM_MATRIX, &pc_air_data->pcmg));

   PetscFunctionReturn(PETSC_SUCCESS);
}

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

PETSC_INTERN PetscErrorCode PCSetUp_AIR_Shell(PC pc)
{
   // Setup our shell PC

   // ~~~~~~
   Mat amat, pmat;
   pc_air_multigrid_data *pc_air_data = NULL;
   MatStructure structure_flag;
   PetscBool setupcalled;
   // ~~~~~~

   PetscFunctionBegin;

   // Get the preconditioner matrix
   PetscCall(PCGetOperators(pc, &amat, &pmat));
   // Get the PC structure flag, either SAME_NONZERO_PATTERN or DIFFERENT_NONZERO_PATTERN
   structure_flag = pc->flag;
   setupcalled = pc->setupcalled;
   // Get the PC shell context
   PetscCall(PCShellGetContext(pc, &pc_air_data));

   // We need to call the setup routines
   if (!setupcalled) {

      PetscCall(setup_air_pcmg(amat, pmat, &pc_air_data->air_data, pc_air_data->pcmg));

   // We can possibly re-use some of the setup
   } else {

      // If we've got a different non-zero pattern we've got to
      // start again
      if (structure_flag == DIFFERENT_NONZERO_PATTERN) {

         // Reset the air data
         PetscCall(reset_pc_air_data(pc_air_data));
         // Set it up again
         PetscCall(setup_air_pcmg(amat, pmat, &pc_air_data->air_data, pc_air_data->pcmg));

      // We can re-use the sparsity
      } else if (structure_flag == SAME_NONZERO_PATTERN) {

         if (pc_air_data->air_data.options.reuse_sparsity) {
            // Reset the air data but keep any reuse data we need
            PetscCall(reset_pc_air_data_reuse(pc_air_data));
         } else {
            // Otherwise we can't reuse anything so reset the air data
            PetscCall(reset_pc_air_data(pc_air_data));
         }
         // Set it up again
         PetscCall(setup_air_pcmg(amat, pmat, &pc_air_data->air_data, pc_air_data->pcmg));

      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode PCApply_AIR_Shell(PC pc, Vec x, Vec y)
{
   // Apply our shell PC

   // ~~~~~~
   pc_air_multigrid_data *pc_air_data = NULL;
   // ~~~~~~

   PetscFunctionBegin;

   // Get the PC context
   PetscCall(PCShellGetContext(pc, &pc_air_data));
   // Just apply the pcmg
   PetscCall(PCApply(pc_air_data->pcmg, x, y));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode PCApplyTranspose_AIR_Shell(PC pc, Vec x, Vec y)
{
   // Apply the transpose of our shell PC
   // Just like the forward apply we hand this straight to the pcmg - petsc's
   // transposed cycles are the transpose of the forward ones, given the
   // transposed smoother and transposed residual setup_air_pcmg registers

   // ~~~~~~
   pc_air_multigrid_data *pc_air_data = NULL;
   // ~~~~~~

   PetscFunctionBegin;

   // Get the PC context
   PetscCall(PCShellGetContext(pc, &pc_air_data));
   // Just apply the transpose of the pcmg
   PetscCall(PCApplyTranspose(pc_air_data->pcmg, x, y));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode PCDestroy_AIR_Shell(PC pc)
{
   // Destroy our shell PC

   // ~~~~~~
   pc_air_multigrid_data *pc_air_data = NULL;
   // ~~~~~~

   PetscFunctionBegin;

   // Get the PC context
   PetscCall(PCShellGetContext(pc, &pc_air_data));
   // Destroy the air data
   PetscCall(destroy_pc_air_data(pc_air_data));
   // Destroy the data structure
   PetscCall(PetscFree(pc_air_data));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode PCReset_AIR_Shell(PC pc)
{
   // Reset our shell PC

   // ~~~~~~
   pc_air_multigrid_data *pc_air_data = NULL;
   // ~~~~~~

   PetscFunctionBegin;

   // Get the PC context
   PetscCall(PCShellGetContext(pc, &pc_air_data));
   // Destroy the air data
   PetscCall(reset_pc_air_data(pc_air_data));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode PCView_AIR_Shell(PC pc, PetscViewer viewer)
{
   // Views our shell PC

   // ~~~~~~
   pc_air_multigrid_data *pc_air_data = NULL;
   // ~~~~~~

   PetscFunctionBegin;

   // Get the PC context
   PetscCall(PCShellGetContext(pc, &pc_air_data));
   // Just call the pcmg view
   PetscCall(PCView(pc_air_data->pcmg, viewer));

   PetscFunctionReturn(PETSC_SUCCESS);
}
