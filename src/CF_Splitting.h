/*
  Prototypes for CF_Splitting.c, the C port of CF_Splitting.F90 (module cf_splitting)

  The optional arguments of the Fortran are pointers that are NULL when the
  argument is absent. The CF markers are int arrays (the Fortran integer) with
  one entry per local row of input_mat
  compute_cf_splitting is called compute_cf_splitting_c here, as the public C
  API in include/pflare.h already has a routine of that name (in PCAIR.c, which
  calls this one with the cr_* arguments absent); the Fortran module cf_splitting
  in src/fortran/CF_Splitting.F90 calls this one with the full argument list
*/

#if !defined(CF_SPLITTING_H)
#define CF_SPLITTING_H

#include <petscmat.h>

PETSC_INTERN PetscErrorCode create_cf_is(Mat input_mat, const int *cf_markers_local, IS *is_fine, IS *is_coarse);
PETSC_INTERN PetscErrorCode first_pass_splitting(Mat input_mat, PetscBool skip_symmetrize, PetscReal strong_threshold,
   int max_luby_steps, int cf_splitting_type, int **cf_markers_local);
PETSC_INTERN PetscErrorCode compute_cf_splitting_c(Mat input_mat, PetscBool skip_symmetrize,
   PetscReal strong_threshold, int max_luby_steps,
   int cf_splitting_type, int ddc_its, PetscReal fraction_swap,
   IS *is_fine, IS *is_coarse,
   const int *cr_inverse_type, const int *cr_poly_order, const int *cr_inverse_sparsity_order,
   const PetscBool *cr_diag_scale_polys);

#endif
