/*
  Prototypes for FC_Smooth.c, the C port of FC_Smooth.F90 (module fc_smooth)

  The levels are 0-based (see air_multigrid_data in pflareimpl.h). The optional
  v_temp_mat of VecISCopyLocalWrapper is NULL when absent
*/

#if !defined(FC_SMOOTH_H)
#define FC_SMOOTH_H

#include <petscksp.h>
#include "pflareimpl.h"

PETSC_INTERN PetscErrorCode mg_coarse_shell_apply(PC pc, Vec x, Vec y);
PETSC_INTERN PetscErrorCode mg_coarse_shell_apply_transpose(PC pc, Vec x, Vec y);
PETSC_INTERN PetscErrorCode mg_smooth_shell_apply(PC pc, Vec x, Vec y);
PETSC_INTERN PetscErrorCode mg_smooth_shell_apply_transpose(PC pc, Vec x, Vec y);
PETSC_INTERN PetscErrorCode create_smooth_temp_vecs(air_multigrid_data *air_data);
PETSC_INTERN PetscErrorCode create_VecISCopyLocalWrapper(air_multigrid_data *air_data, int our_level, MatType mat_type, Mat input_mat);
PETSC_INTERN PetscErrorCode destroy_VecISCopyLocalWrapper(air_multigrid_data *air_data, int our_level);
PETSC_INTERN PetscErrorCode VecISCopyLocalWrapper(air_multigrid_data *air_data, int our_level, PetscBool fine, Vec vfull, ScatterMode mode, Vec vreduced,
   Vec v_temp_mat);
PETSC_INTERN PetscErrorCode mg_FC_point_richardson(PC pc, Vec b, Vec x, Vec r, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt maxits,
   PetscBool guess_zero, PetscInt *its, PCRichardsonConvergedReason *conv_reason);
PETSC_INTERN PetscErrorCode f_smooths(Vec b, Vec x, PetscBool guess_zero, PetscBool first_smooth, air_multigrid_data *air_data, int our_level, int its);
PETSC_INTERN PetscErrorCode c_smooths(Vec b, Vec x, PetscBool guess_zero, PetscBool first_smooth, air_multigrid_data *air_data, int our_level, int its);
PETSC_INTERN PetscErrorCode mg_FC_point_apply_transpose(PC pc, Vec b, Vec x);
PETSC_INTERN PetscErrorCode f_smooths_transpose(Vec b, air_multigrid_data *air_data, int our_level, int its, PetscBool *x_f_zero, PetscBool x_c_zero);
PETSC_INTERN PetscErrorCode c_smooths_transpose(Vec b, air_multigrid_data *air_data, int our_level, int its, PetscBool x_f_zero, PetscBool *x_c_zero);
PETSC_INTERN PetscErrorCode mg_FC_point_residual_transpose(Mat mat, Vec b, Vec x, Vec r);

#endif
