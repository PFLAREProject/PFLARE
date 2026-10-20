/*
  Prototypes for Neumann_Poly.c, the C port of Neumann_Poly.F90 (module neumann_poly)
*/

#if !defined(NEUMANN_POLY_H)
#define NEUMANN_POLY_H

#include <petscmat.h>
#include "pflareimpl.h"

PETSC_INTERN PetscErrorCode calculate_and_build_neumann_polynomial_inverse(Mat matrix, int poly_order,
   tsqr_buffers *buffers, PetscReal coefficients[], PetscInt coefficients_size, int poly_sparsity_order, PetscBool matrix_free,
   Mat *reuse_mat, Mat **reuse_submatrices, Mat *inv_matrix);

#endif
