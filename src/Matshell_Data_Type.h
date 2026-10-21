/*
  Prototypes for Matshell_Data_Type.c, the C port of Matshell_Data_Type.F90
  (module matshell_data_type)

  The mat_ctxtype struct is in pflareimpl.h. The optional arguments of the
  Fortran are pointers that are NULL when the argument is absent
*/

#if !defined(MATSHELL_DATA_TYPE_H)
#define MATSHELL_DATA_TYPE_H

#include <petscmat.h>
#include "pflareimpl.h"

PETSC_INTERN PetscErrorCode create_mat_ctx(mat_ctxtype **mat_ctx);
PETSC_INTERN PetscErrorCode ensure_block_temp_mats(mat_ctxtype *mat_ctx, Mat x_mat, PetscInt n_temps, const PetscBool *need_rhs);
PETSC_INTERN PetscErrorCode attach_mat_product(Mat mat, Mat x_mat, PetscBool transpose_mat, Mat y_mat, PetscBool *has_product);
PETSC_INTERN PetscErrorCode ensure_block_kernel_products(mat_ctxtype *mat_ctx, PetscInt second_target, PetscBool *attached);

#endif
