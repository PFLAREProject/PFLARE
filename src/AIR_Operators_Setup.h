/*
  Prototypes for AIR_Operators_Setup.c, the C port of AIR_Operators_Setup.F90
  (module air_operators_setup)

  The levels are 0-based (see air_multigrid_data in pflareimpl.h). The arrays of
  near nullspace vectors cross with their sizes (the null vectors on the C points
  have the sizes of the full ones)
*/

#if !defined(AIR_OPERATORS_SETUP_H)
#define AIR_OPERATORS_SETUP_H

#include <petscksp.h>
#include "pflareimpl.h"

PETSC_INTERN PetscErrorCode get_submatrices_start_poly_coeff_comms(Mat input_mat, int our_level, air_multigrid_data *air_data);
PETSC_INTERN PetscErrorCode finish_comms_compute_restrict_prolong(Mat A, int our_level, air_multigrid_data *air_data,
   Vec left_null_vecs[], PetscInt left_null_vecs_size, Vec right_null_vecs[], PetscInt right_null_vecs_size,
   Vec left_null_vecs_c[], Vec right_null_vecs_c[]);
PETSC_INTERN PetscErrorCode compute_coarse_matrix(Mat A, int our_level, air_multigrid_data *air_data,
   Mat *coarse_matrix);

#endif
