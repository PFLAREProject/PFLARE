/*
  Private header for the C implementation of PFLARE

  Holds what is shared between the C files of the library and is not part
  of the public API in include/pflare.h
*/

#if !defined(PFLAREIMPL_H)
#define PFLAREIMPL_H

#include <petscmat.h>
#include "pflare_parameters.h"

/* Complex is intentionally unsupported */
#if defined(PETSC_USE_COMPLEX)
#error "PFLARE BLAS/LAPACK dispatch does not support complex PETSc builds"
#endif

/* ~~~~~~~~~~~~~~~~~~~~~~~~
   Private structs
   The Fortran derived types become structs here as the modules that use
   them are ported to C (section 4 and convention 3 of
   docs/dev/fortran_removal_plan.md)
   ~~~~~~~~~~~~~~~~~~~~~~~~ */

// ~~~~~~~~
// Stores data for the asynchronous comms (TSQR.c, tsqr_buffers in TSQR.F90)
// Created with create_tsqr_buffers (the defaults of the Fortran type) and
// destroyed with destroy_tsqr_buffers
// ~~~~~~~~
typedef struct {
   MPI_Request request;
   // Each is NULL when not allocated, both hold n_size * n_size + 1 entries
   // (n_size is in the first entry)
   PetscReal *R_buffer_send, *R_buffer_receive;
   // In case this comms request is done on a matrix on a subcomm, we
   // need to keep a pointer to it
   Mat matrix;
   // Has the user asked us to be on a subcomm
   PetscBool subcomm;
   // Did we actually end up on a subcomm
   PetscBool on_subcomm;
   int proc_stride;
} tsqr_buffers;

// ~~~~~~~~
// Stores the data needed for gmres polynomials (Gmres_Poly_Data_Type.c,
// gmres_poly_data in Gmres_Poly_Data_Type.F90)
// Until the AIR hierarchy is ported (stage 7 of docs/dev/fortran_removal_plan.md)
// the Fortran air_multigrid_data holds these, through the bind(C) Fortran type
// gmres_poly_data in src/fortran/Gmres_Poly_Data_Type.F90, which reads and writes
// this same memory: the two definitions must stay in step (the same fields, in the
// same order, with the same types)
// create_poly_data gives the defaults of the Fortran type, reset_poly_data frees the
// coefficients and destroy_poly_data frees the coefficients and the buffers
// ~~~~~~~~
typedef struct {
   // Coefficients for the gmres polynomial, column-major with coefficients_rows
   // rows and coefficients_cols columns (NULL when not allocated)
   // If using the newton basis this has two columns with the real
   // and imaginary roots
   // Allocated with PetscMalloc and freed with PetscFree
   PetscReal *coefficients;
   PetscInt coefficients_rows, coefficients_cols;
   // Inverse type
   int inverse_type;
   // Order of the gmres polynomial
   int gmres_poly_order;
   // The sparsity order of the polynomial if assembled
   int gmres_poly_sparsity_order;
   // Asynchronous buffers for the TSQR
   // The Fortran type held these directly, here they are created on first use
   // (see poly_data_buffers) as the Fortran mirror starts with a null pointer
   tsqr_buffers *buffers;
} gmres_poly_data;

// ~~~~~~~~
// This is our context type for matshells (Matshell_Data_Type.c, mat_ctxtype in
// Matshell_Data_Type.F90)
// Created with create_mat_ctx (the defaults of the Fortran type) and freed with
// PetscFree, as the Fortran deallocated it
// The Fortran pointer arrays carried their sizes, which are the _size fields here
// ~~~~~~~~
typedef struct {
   int our_level;
   PetscReal *coefficients;
   PetscInt coefficients_size;
   PetscBool own_coefficients;
   // real_roots and imag_roots both have roots_size entries
   PetscReal *real_roots;
   PetscReal *imag_roots;
   PetscInt roots_size;
   Mat mat, mat_scaled;
   // Whether the inner mat_scaled shell applies I - D^-1 A (the Neumann polynomial)
   // rather than D^-1 A (the diagonally scaled gmres polynomials)
   PetscBool neumann_inner;
   // Temporary vectors we use
   Vec mf_temp_vec[5];
   // Temporary dense matrices we use during a multiple rhs (block) apply
   // These are lazily created from the input block of rhs, as the number of
   // columns isn't known until the apply happens
   Mat mf_temp_mat[4];
   // The number of global columns the cached mf_temp_mat's were built with
   PetscInt mf_temp_mat_ncols;
   // Whether the block apply kernels have their products attached to the
   // mf_temp_mat scratch, and the mat they were attached with - the products
   // themselves keep that mat alive, so the handle comparison in the kernels
   // is safe even if the context is later pointed at a different mat
   PetscBool mf_products_attached;
   Mat mf_product_mat;
   // The reciprocal of mf_temp_vec(MF_VEC_DIAG), only used by the block applies
   // of the diagonally scaled polynomials
   Vec mf_vec_diag_recip;
   // The transposed twin of this polynomial, built lazily the first time a
   // transposed apply happens (MATOP_MULT_TRANSPOSE) - see ensure_transpose_mat
   // It is a matshell running the very same forward arithmetic as this one, but
   // against a MATTRANSPOSEVIRTUAL of mat, which is all the transpose needs as
   // our polynomials have real coefficients, so q(A)^T = q(A^T)
   Mat transpose_mat;
   // The mat the twin above was built from, so we can spot mat being swapped
   // out from under us - the same handle comparison as mf_product_mat
   Mat mf_transpose_source;
   // The Fortran air_multigrid_data, which the Fortran FC smoothers still set and
   // read until stage 7
   void *air_data;
} mat_ctxtype;

// The slots of mat_ctxtype mf_temp_vec and mf_temp_mat, 0-based (the Fortran slot
// indices PFLARE_MF_VEC_* and PFLARE_MF_MAT_* in pflare_parameters.h minus one)
#define MF_VEC_TEMP       (PFLARE_MF_VEC_TEMP - 1)
#define MF_VEC_TEMP_TWO   (PFLARE_MF_VEC_TEMP_TWO - 1)
#define MF_VEC_TEMP_THREE (PFLARE_MF_VEC_TEMP_THREE - 1)
#define MF_VEC_DIAG       (PFLARE_MF_VEC_DIAG - 1)
#define MF_VEC_RHS        (PFLARE_MF_VEC_RHS - 1)
#define MF_MAT_TEMP       (PFLARE_MF_MAT_TEMP - 1)
#define MF_MAT_TEMP_TWO   (PFLARE_MF_MAT_TEMP_TWO - 1)
#define MF_MAT_TEMP_THREE (PFLARE_MF_MAT_TEMP_THREE - 1)
#define MF_MAT_RHS        (PFLARE_MF_MAT_RHS - 1)

#endif
