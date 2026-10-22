/*
  C port of MatDiagDomSubmatrix.F90 (module matdiagdomsubmatrix)

  compute_diag_dom_submatrix is called compute_diag_dom_submatrix_c here, as the
  public C API in include/pflare.h already has a routine of that name (in PCAIR.c,
  which calls this one)
*/

#include <petscmat.h>
#include "pflare.h"
#include "pflareimpl.h"
#include "CF_Splitting.h"
#include "PETSc_Helper.h"
#include "MatDiagDomSubmatrix.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode compute_diag_dom_submatrix_c(Mat input_mat, PetscReal max_dd_ratio, Mat *output_mat)
{
   // Returns a diagonally dominant submatrix taken from input_mat where every row's
   // diagonal dominance ratio is < max_dd_ratio
   // Not guaranteed to be the optimal (ie largest) submatrix but should be close
   // This works for symmetric and asymmetric input_mat
   // Works in serial, parallel and kokkos (and hence gpus)
   // This is just a convenience wrapper around compute_cf_splitting and matcreatesubmatrixwrapper

   // ~~~~~~
   IS is_fine = NULL, is_coarse = NULL;
   int ddc_its, max_luby_steps, algorithm;
   PetscReal ddc_fraction;
   PetscBool skip_symmetrize;

   // ~~~~~~

   PetscFunctionBegin;

   PetscCheck(!(max_dd_ratio <= 0.0 || max_dd_ratio >= 1.0), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "max_dd_ratio input to compute_diag_dom_submatrix must be (0.0, 1.0)");

   // Ignored for CF_DIAG_DOM
   ddc_its = 0;
   // Ignored for CF_DIAG_DOM
   ddc_fraction = 0.0;
   // As many steps as needed
   max_luby_steps = -1;
   // PMISR DDC where strength of connection is given by
   // |a_ij| .ge. max_dd_ratio |a_ii|
   algorithm = CF_DIAG_DOM;
   // Symmetrize the strength matrix
   skip_symmetrize = PETSC_FALSE;

   // Call the CF splitting
   // We call with max_dd_ratio as the strong_threshold
   PetscCall(compute_cf_splitting_c(input_mat,
         skip_symmetrize,
         max_dd_ratio, max_luby_steps,
         algorithm,
         ddc_its,
         ddc_fraction,
         &is_fine, &is_coarse,
         NULL, NULL, NULL, NULL));

   PetscCall(ISDestroy(&is_coarse));

   // The input_mat(is_fine, is_fine) is the diagonally dominant
   // submatrix - this is the wrapper around the kokkos gpu code
   PetscCall(MatCreateSubMatrixWrapper(input_mat,
               is_fine, is_fine, MAT_INITIAL_MATRIX,
               output_mat, NULL, NULL, NULL, NULL));

   PetscCall(ISDestroy(&is_fine));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------
