/*
  C routines that are used to get around a lack of some PETSc fortran interfaces

  The helpers for the repartitioning (MatPartitioning_c, GenerateIS_ProcAgglomeration_c
  and MatMPICreateNonemptySubcomm_c) are now in Repartition.c, the C port of the
  Fortran that called them. MatGetDiagonalOnly_c is called by AIR_MG_Setup.c (and by
  the reference Fortran) and PetscFree_c by the Fortran wrappers in src/fortran. The
  PC helpers are only called by the reference Fortran in src/fref, so are only built
  with PFLARE_PORT_DEBUG (the C reads pc->flag and pc->setupcalled directly)
*/

// Include the petsc header files
#include <petsc.h>
#include <petsc/private/pcimpl.h>
#include "C_PETSc_Routines.h"

#if defined(PFLARE_PORT_DEBUG)
// Returns pc->flag so we can have access to the structure flag
// for re-use
PETSC_INTERN void c_PCGetStructureFlag(PC *pc, int *flag)
{
   *flag = (*pc)->flag;
}

// Returns pc->setupcalled
PETSC_INTERN void PCGetSetupCalled_c(PC *pc, PetscInt *setupcalled)
{
   *setupcalled = (*pc)->setupcalled;
}

// Mark that we need to do setup again (e.g. after a reset)
PETSC_INTERN void PCMarkNotSetUp_c(PC *pc)
{
   (*pc)->setupcalled = PETSC_FALSE;
}
#endif

PETSC_EXTERN PetscErrorCode MatGetDiagonalMarkers_SeqAIJ(Mat, const PetscInt **, PetscBool *);
// Returns true if a matrix is only the diagonal
PETSC_INTERN PetscErrorCode MatGetDiagonalOnly_c(Mat *A, int *diag_only)
{
  MPI_Comm        acomm;
  PetscMPIInt     size;
  PetscInt local_rows, local_cols;
  PetscInt rank_diag = 0, rank_diag_serial = 0;

  PetscFunctionBegin;

  PetscCall(PetscObjectGetComm((PetscObject)(*A), &acomm));
  PetscCallMPI(MPI_Comm_size(acomm, &size));

  Mat mat_local = NULL, mat_nonlocal = NULL; 

  // Get the existing output mats
  if (size != 1)
  {
     PetscCall(MatMPIAIJGetSeqAIJ(*A, &mat_local, &mat_nonlocal, NULL));
  }
  else
  {
     mat_local = *A;
  } 

  PetscCall(MatGetLocalSize(*A, &local_rows, &local_cols));
  *diag_only = 0;

  // Use MatGetRowIJ to read the SeqAIJ nnz count without touching the private struct.
  PetscInt local_nnz = 0, nonlocal_nnz = 0;
  {
     PetscInt n;
     PetscBool symmetric = PETSC_FALSE, inodecompressed = PETSC_FALSE, done;
     const PetscInt *ad_ia;
     PetscCall(MatGetRowIJ(mat_local, 0, symmetric, inodecompressed, &n, &ad_ia, NULL, &done));
     local_nnz = ad_ia[n];
     PetscCall(MatRestoreRowIJ(mat_local, 0, symmetric, inodecompressed, &n, &ad_ia, NULL, &done));
  }

  if (size != 1)
  {
      PetscBool diagDense;

      {
         PetscInt n;
         PetscBool symmetric = PETSC_FALSE, inodecompressed = PETSC_FALSE, done;
         const PetscInt *ao_ia;
         PetscCall(MatGetRowIJ(mat_nonlocal, 0, symmetric, inodecompressed, &n, &ao_ia, NULL, &done));
         nonlocal_nnz = ao_ia[n];
         PetscCall(MatRestoreRowIJ(mat_nonlocal, 0, symmetric, inodecompressed, &n, &ao_ia, NULL, &done));
      }

      // In parallel also have to check the nonlocal has nothing in it
      PetscCall(MatGetDiagonalMarkers_SeqAIJ(mat_local, NULL, &diagDense));
      if (diagDense && local_rows == local_nnz && nonlocal_nnz == 0)
      {
         rank_diag_serial++;
      }

      // Reduction to check every rank
      PetscCallMPI(MPI_Allreduce(&rank_diag_serial, &rank_diag, 1, MPIU_INT, MPI_SUM, acomm));
  }
  else
  {
      PetscBool diagDense;
      PetscCall(MatGetDiagonalMarkers_SeqAIJ(*A, NULL, &diagDense));
      // In serial easy
      if (diagDense && local_rows == local_nnz)
      {
         rank_diag++;
      }
  }

  // If every rank is diagonal only, the entire matrix is diagonal
  if (rank_diag == size)
  {
     *diag_only = 1;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Frees memory allocated with PetscMalloc in the C ports of the Fortran modules, for the
// Fortran callers (src/fortran/Gmres_Poly_Newton.F90 and Port_Interfaces*.F90) that copy
// such an array into a Fortran allocatable
// *ptr is NULL on return
PETSC_INTERN PetscErrorCode PetscFree_c(void **ptr)
{
   PetscFunctionBegin;
   PetscCall(PetscFree(*ptr));
   PetscFunctionReturn(PETSC_SUCCESS);
}
