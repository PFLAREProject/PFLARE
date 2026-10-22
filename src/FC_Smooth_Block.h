/*
  Prototypes for FC_Smooth_Block.c, the C port of FC_Smooth_Block.F90
  (module fc_smooth_block)

  The levels are 0-based (see air_multigrid_data in pflareimpl.h). The optional
  x_temp_full_mat of MatISCopyLocalWrapper is NULL when absent
*/

#if !defined(FC_SMOOTH_BLOCK_H)
#define FC_SMOOTH_BLOCK_H

#include <petscksp.h>
#include "pflareimpl.h"

PETSC_INTERN PetscErrorCode create_air_block_temp(Mat op_mat, PetscBool from_cols, Mat x_mat, PetscInt local_cols, PetscInt global_cols, Mat *block_mat);
PETSC_INTERN PetscErrorCode destroy_air_block_temps(air_multigrid_data *air_data);
PETSC_INTERN PetscErrorCode ensure_air_block_temps(air_multigrid_data *air_data, Mat x_mat);
PETSC_INTERN PetscErrorCode setup_air_block_products(air_multigrid_data *air_data, int our_level);
PETSC_INTERN PetscErrorCode mat_product_block(Mat mat, Mat x_mat, Mat y_mat, PetscBool transpose_mat);
PETSC_INTERN PetscErrorCode shell_block_apply_or_columns(Mat inv_mat, Mat x_mat, Mat y_mat);
PETSC_INTERN PetscErrorCode apply_inverse_block(Mat inv_mat, Mat x_mat, Mat y_mat);
PETSC_INTERN PetscErrorCode apply_inverse_block_cached(Mat inv_mat, Mat x_mat, Mat y_mat);
PETSC_INTERN PetscErrorCode MatISCopyLocalWrapper(air_multigrid_data *air_data, int our_level, PetscBool fine, Mat xfull_mat, ScatterMode mode, Mat xreduced_mat,
   Mat x_temp_full_mat);
PETSC_INTERN PetscErrorCode mat_iscopy_local_host(air_multigrid_data *air_data, int our_level, PetscBool fine, Mat xfull_mat, ScatterMode mode, Mat xreduced_mat);
PETSC_INTERN PetscErrorCode mg_coarse_shell_block_apply(PC pc, Mat x, Mat y);
PETSC_INTERN PetscErrorCode mg_smooth_shell_block_apply(PC pc, Mat x, Mat y);
PETSC_INTERN PetscErrorCode mg_FC_block_richardson(PC pc, Mat b, Mat x, Mat r, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt maxits,
   PetscBool guess_zero, PetscInt *its, PCRichardsonConvergedReason *conv_reason);
PETSC_INTERN PetscErrorCode f_smooths_block(Mat b, Mat x, PetscBool guess_zero, PetscBool first_smooth, air_multigrid_data *air_data, int our_level, int its);
PETSC_INTERN PetscErrorCode c_smooths_block(Mat b, Mat x, PetscBool guess_zero, PetscBool first_smooth, air_multigrid_data *air_data, int our_level, int its);

#endif
