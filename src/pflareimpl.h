/*
  Private header for the C implementation of PFLARE

  Holds what is shared between the C files of the library and is not part
  of the public API in include/pflare.h
*/

#if !defined(PFLAREIMPL_H)
#define PFLAREIMPL_H

#include <petscmat.h>
#include <petscksp.h>
#include "pflare.h"
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
// air_multigrid_data below holds these. The Fortran reference of the AIR hierarchy
// in src/fref (PFLARE_PORT_DEBUG only) holds them through the bind(C) Fortran type
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
   // The air_multigrid_data of the level this placeholder matshell is on, which the
   // FC smoothers read (struct air_multigrid_data_s is the air_multigrid_data below)
   struct air_multigrid_data_s *air_data;
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

// ~~~~~~~~
// The small array holders of AIR_Data_Type.F90 (module air_data_type)
// ~~~~~~~~

// A pointer to an array of Mats (petsc_mat_pointer_array), as MatCreateSubMatrices
// gives back - NULL when not associated
typedef struct {
   Mat *array;
} petsc_mat_pointer_array;
// An array of Vecs with one entry per level (petsc_vec_array) - NULL when not allocated
typedef struct {
   Vec *array;
} petsc_vec_array;
// The dense (block) equivalent of petsc_vec_array, one dense scratch block
// per level (petsc_dense_mat_array)
typedef struct {
   Mat *array;
} petsc_dense_mat_array;
// An integer array (int_vec_array) - NULL with size 0 when not allocated
typedef struct {
   int *array;
   PetscInt size;
} int_vec_array;

// ~~~~~~~~
// Stores the options needed for air multigrid (air_options in AIR_Data_Type.F90)
// These defaults are reasonable settings for time independent
// advection equations (ie purely hyperbolic) on a 2D unstructured
// triangular mesh
// Can be changed via command line arguments
// The defaults of the Fortran type are set by create_air_options (AIR_Data_Type.c)
// These defaults need to match those in destroy_air_data
// The fields take the type of the corresponding public getter in pflare.h
// (convention 4 of docs/dev/fortran_removal_plan.md)
// ~~~~~~~~

// The size of smooth_order
#define AIR_SMOOTH_ORDER_SIZE 10

typedef struct {

   // Print out stats and timings
   // These require some parallel reductions to compute
   // so they are off by default
   // -pc_air_print_stats_timings
   PetscBool print_stats_timings;

   // Maximum number of levels in the multigrid hierarchy
   // -pc_air_max_levels
   PetscInt max_levels;
   // Minimum number of global unknowns on the coarse grid
   // -pc_air_coarse_eq_limit
   PetscInt coarse_eq_limit;
   // From this level onwards, build and then evaluate if the coarse grid solver
   // is good enough and use that to determine if we should truncate on that level
   // This is a 1-based level number, as the user sees them
   // -pc_air_auto_truncate_start_level
   PetscInt auto_truncate_start_level;
   // What relative tolerance to use to determine if a coarse grid solver is good enough
   // -pc_air_auto_truncate_tol
   PetscReal auto_truncate_tol;

   // Perform processor agglomeration throughout the hierarchy
   // This reduces the number of active MPI ranks as we coarsen
   // by a factor of processor_agglom_factor, whenever either there
   // are fewer unkowns per core (on average) than process_eq_limit or
   // the local to non-local ratio of nnzs is < processor_agglom_ratio
   // The entire hierarchy stays on comm_world however
   // Only happens where necessary, not on every level
   // -pc_air_processor_agglom
   PetscBool processor_agglom;
   // The local to nonlocal ratio of nnzs that is used to
   // trigger processor agglomeration on all level
   // -pc_air_processor_agglom_ratio
   PetscReal processor_agglom_ratio;
   // What factor to reduce the number of active MPI ranks by
   // each time when doing processor agglomeration
   // -pc_air_processor_agglom_ratio
   PetscInt processor_agglom_factor;
   // If on average there are fewer than this number of equations per rank
   // processor agglomeration will be triggered
   // -pc_air_process_eq_limit
   PetscInt process_eq_limit;
   // If we are doing processor agglomeration, then we have
   // some ranks with no rows
   // If computing a gmres polynomial inverse
   // with inverse_type arnoldi or newton, then we can have
   // the reductions occur on a subcomm if we want to reduce the cost
   // -pc_air_subcomm
   PetscBool subcomm;

   // This is used in the CF splitting to define strong dependencies/influences
   // -pc_air_strong_threshold
   PetscReal strong_threshold;
   // How many passes of DDC to do
   // -pc_air_ddc_its
   PetscInt ddc_its;
   // Second pass in the PMISR DDC CF splitting converts
   // this fraction of local F points to C based on diagonal dominance
   // -pc_air_ddc_fraction
   PetscReal ddc_fraction;
   // What CF splitting algorithm to use
   // 0 - PMISR DDC
   // 1 - PMIS distance 1
   // 2 - PMIS distance 2 - uses S'S + S
   // 3 - Root node aggregation
   // -pc_air_cf_splitting_type
   CFSplittingType cf_splitting_type;
   // Maximum number of Luby steps to do in CF splitting
   // Negative means do as many as needed (at the cost of a parallel
   // reduction everytime we finish a Luby step)
   // -pc_air_max_luby_steps
   PetscInt max_luby_steps;

   // Stores what smoothing happens
   // Positive means F smooth, negative means C smooth
   // The size of the integer is how many consecutive smooths
   // e.g., (/ 2 -2 2 /) means 2 F, then 2 C, then 2 F
   // 2 F smooths by default
   int smooth_order[AIR_SMOOTH_ORDER_SIZE];
   // Do we do any C point smooths?
   PetscBool any_c_smooths;
   // If using a polynomial inverse type, do we diagonally scale before computing?
   // -pc_air_diag_scale_polys
   PetscBool diag_scale_polys;
   // Do we apply our polynomials matrix free when smoothing?
   // -pc_air_matrix_free_polys
   PetscBool matrix_free_polys;
   // Do we use a one point injection classical prolongator or an AIR-style prolongator
   // -pc_air_one_point_classical_prolong
   PetscBool one_point_classical_prolong;
   // Do we do full smoothing up and down rather than FF or FFC
   // -pc_air_full_smoothing_up_and_down
   PetscBool full_smoothing_up_and_down;
   // Do we define our prolongator as R^T?
   // -pc_air_symmetric
   PetscBool symmetric;
   // Do we compute a near-nullspace vector and apply it as a constaint
   // to the prolongator?
   // -pc_air_constrain_w
   PetscBool constrain_w;
   // Do we compute a near-nullspace vector and apply it as a constaint
   // to the restrictor?
   // -pc_air_constrain_z
   PetscBool constrain_z;
   // How many iterations of improvement do we want to do to Z
   // -pc_air_improve_z_its
   PetscInt improve_z_its;
   // How many iterations of improvement do we want to do to W
   // -pc_air_improve_w_its
   PetscInt improve_w_its;

   // Strong R threshold to apply dropping prior to computing Z
   // This only applies when computing Z, ie if you build a GMRES polynomial approximation
   // to Aff^-1, it applies this dropping, then computes Z, then rebuilds
   // an Aff^-1 approximation without the dropping for smoothing
   // -pc_air_strong_r_threshold
   PetscReal strong_r_threshold;

   // What type of approximation do we use for Z?
   // 0 - Aff^-1 approximation determined by inverse type (below) and then Z computed with matmatmult
   // 1 - lAIR computes Z directly
   // 2 - SAI version of lAIR computes Z directly
   // -pc_air_z_type
   PCAIRZType z_type;

   // If z_type == 1 or 2, this is the distance the grid-transfer operators go out to
   // This is so we can have lair out to some distance, and then a different sparsity
   // for our smoothers
   // If z_type == 0 this is ignored, and the distance is determined by inverse_sparsity_order + 1
   // -pc_air_lair_distance
   PetscInt lair_distance;

   // What type of approximation do we use for Aff^-1
   // This is used both for Z (if z_type == 0, see below) and for F smoothing
   // These are defined by PCPFLAREINVType
   // 0 - GMRES polynomial with the power basis
   // 1 - GMRES polynomial with the arnoldi basis
   // 2 - GMRES polynomial with the newton basis with extra roots for stability
   // 3 - GMRES polynomial with the newton basis without extra roots
   // 4 - Neumann polynomial
   // 5 - SAI
   // 6 - Incomplete SAI (ie a restricted additive schwartz)
   // 7 - Weighted Jacobi with weight 4 / ( 3 * || Dff^(-1/2) * Aff * Dff^(-1/2) ||_inf )
   // 8 - Unweighted Jacobi
   // 9 - l1 Jacobi with inverse sign(a_ii) / sum_j |a_ij|
   // -pc_air_inverse_type
   PCPFLAREINVType inverse_type;

   // This is the order of polynomial we use in air if inverse_type is
   // power, arnoldi, newton or neumann
   // -pc_air_poly_order
   PetscInt poly_order;
   // This is the order of sparsity we use if we assemble our approximate inverses
   // If z_type == 0 this also determines what distance our grid-transfer operators are
   // distance = inverse_sparsity_order + 1
   // -pc_air_inverse_sparsity_order
   PetscInt inverse_sparsity_order;

   // Inverse type for c smoothing
   // If unset (-1) this defaults to whatever the F point smoother is
   // Use the PCAIRGetC* routines to get the resolved value
   // An int rather than a PCPFLAREINVType as it can be -1
   // -pc_air_c_inverse_type
   int c_inverse_type;
   // Poly order for c smoothing
   // If unset (-1) this defaults to whatever the F point smoother is
   // -pc_air_c_poly_order
   PetscInt c_poly_order;
   // Inverse sparsity order for c smoothing
   // If unset (-1) this defaults to whatever the F point smoother is
   // -pc_air_c_inverse_sparsity_order
   PetscInt c_inverse_sparsity_order;

   // These are for the coarse grid solver
   // -pc_air_coarsest_inverse_type
   PCPFLAREINVType coarsest_inverse_type;
   // -pc_air_coarsest_poly_order
   PetscInt coarsest_poly_order;
   // -pc_air_coarsest_inverse_sparsity_order
   PetscInt coarsest_inverse_sparsity_order;
   // -pc_air_coarsest_matrix_free_polys
   PetscBool coarsest_matrix_free_polys;
   // -pc_air_coarsest_diag_scale_polys
   PetscBool coarsest_diag_scale_polys;
   // -pc_air_coarsest_subcomm
   PetscBool coarsest_subcomm;

   // These are the relative drop tolerances (inf norm) on R and A after they are built
   // -pc_air_r_drop
   PetscReal r_drop;
   // -pc_air_a_drop
   PetscReal a_drop;
   // Whether to lump in A or drop
   // -pc_air_a_lump
   PetscBool a_lump;

   // Whether or not to re-use the existing sparsity when PCSetup is called
   // with SAME_NONZERO_PATTERN
   // This involves re-using the CF splitting, the symbolic mat-mat mults,
   // the repartitioning, the structure of the matrices with drop tolerances applied, etc
   // This will take more memory but
   // will make the setup much cheaper on subsequent calls. If the matrix has
   // changed entries convergence may suffer if the matrix is sufficiently different
   // -pc_air_reuse_sparsity
   PetscBool reuse_sparsity;

   // Controls how much data is stored for reuse when reuse_sparsity is enabled.
   // Higher values store more data (less recomputation) at the cost of more memory.
   // The CF splitting (IS_fine_index/IS_coarse_index) is always kept regardless of
   // reuse_amount as it is the foundation on which all other reuse depends.
   // 1 - store nothing beyond the CF splitting; all matrices and derived IS are
   //     rebuilt from scratch each setup; saves only PMISR/DDC CF-splitting cost
   // 2 - additionally store IS_REPARTITION + IS_R_Z_FINE_COLS + all matrices needed
   //     to guarantee stable SpGEMM sparsity (MAT_RAP_DROP, MAT_Z, MAT_W,
   //     MAT_Z_DROP, MAT_W_DROP, MAT_AP, MAT_RAP, and when strong_r_threshold!=0
   //     also MAT_A_DROP, MAT_AFF_DROP, MAT_ACF_DROP, MAT_AFC_DROP);
   //     A_ff/A_fc/A_cf/prolongators/restrictors are rebuilt fresh each setup.
   //     reuse_grid_transfer and reuse_one_point_classical_prolong are FALSE.
   //     Saves CF splitting + SpGEMM symbolic cost
   // 3 - store everything (default, preserves previous behaviour)
   // -pc_air_reuse_amount
   PetscInt reuse_amount;

   // Whether or not to also re-use the gmres polynomial coefficients when
   // reuse_sparsity is set to true
   // If the matrix has been changed the reused coefficients won't be correct,
   // and the coefficients are very sensitive to changes in the matrix
   // This is really only a useful option if you are regenerating
   // the hierarchy for the exact same matrix where you have stored
   // the gmres polynomial coefficients externally and restore them
   // -pc_air_reuse_poly_coeffs
   PetscBool reuse_poly_coeffs;

} air_options;

// ~~~~~~~~
// The 0-based slots of air_reuse_data reuse_mat (and reuse_submatrices) and reuse_is,
// the Fortran indices MAT_* and IS_* of pflare_parameters.h minus one
// ~~~~~~~~
#define MAT_AP                   (PFLARE_MAT_AP - 1)
#define MAT_RAP                  (PFLARE_MAT_RAP - 1)
#define MAT_RAP_DROP             (PFLARE_MAT_RAP_DROP - 1)
#define MAT_Z_DROP               (PFLARE_MAT_Z_DROP - 1)
#define MAT_W_DROP               (PFLARE_MAT_W_DROP - 1)
#define MAT_COARSE_REPARTITIONED (PFLARE_MAT_COARSE_REPARTITIONED - 1)
#define MAT_P_REPARTITIONED      (PFLARE_MAT_P_REPARTITIONED - 1)
#define MAT_R_REPARTITIONED      (PFLARE_MAT_R_REPARTITIONED - 1)
#define MAT_AFF_DROP             (PFLARE_MAT_AFF_DROP - 1)
#define MAT_ACF_DROP             (PFLARE_MAT_ACF_DROP - 1)
#define MAT_AFC_DROP             (PFLARE_MAT_AFC_DROP - 1)
#define MAT_A_DROP               (PFLARE_MAT_A_DROP - 1)
#define MAT_W                    (PFLARE_MAT_W - 1)
#define MAT_Z                    (PFLARE_MAT_Z - 1)
#define MAT_INV_AFF              (PFLARE_MAT_INV_AFF - 1)
#define MAT_INV_AFF_DROPPED      (PFLARE_MAT_INV_AFF_DROPPED - 1)
#define MAT_INV_ACC              (PFLARE_MAT_INV_ACC - 1)
#define MAT_SAI_SUB              (PFLARE_MAT_SAI_SUB - 1)
#define MAT_Z_AFF                (PFLARE_MAT_Z_AFF - 1)
#define MAT_Z_NO_SPARSITY        (PFLARE_MAT_Z_NO_SPARSITY - 1)
#define MAT_W_AFF                (PFLARE_MAT_W_AFF - 1)
#define MAT_W_NO_SPARSITY        (PFLARE_MAT_W_NO_SPARSITY - 1)
#define IS_REPARTITION           (PFLARE_IS_REPARTITION - 1)
#define IS_R_Z_FINE_COLS         (PFLARE_IS_R_Z_FINE_COLS - 1)

// The 0-based slots of the dense scratch blocks block_temp_fine/block_temp_coarse,
// the Fortran indices AIR_MAT_* of pflare_parameters.h minus one
#define AIR_MAT_SOL      (PFLARE_AIR_MAT_SOL - 1)
#define AIR_MAT_TEMP     (PFLARE_AIR_MAT_TEMP - 1)
#define AIR_MAT_RESIDUAL (PFLARE_AIR_MAT_RESIDUAL - 1)
#define AIR_MAT_RHS      (PFLARE_AIR_MAT_RHS - 1)
#define AIR_MAT_OFF_DIAG (PFLARE_AIR_MAT_OFF_DIAG - 1)

// ~~~~~~~~
// Stores temporary data we use for re-use (air_reuse_data in AIR_Data_Type.F90)
// The things stored in these structures
// would normally be destroyed during the setup
// Indexed with the 0-based MAT_* and IS_* above
// ~~~~~~~~
typedef struct {

   Mat reuse_mat[22];
   petsc_mat_pointer_array reuse_submatrices[22];
   IS reuse_is[2];

} air_reuse_data;

// ~~~~~~~~
// Stores the data needed for air multigrid (air_multigrid_data in AIR_Data_Type.F90)
// The defaults of the Fortran type are set by create_air_multigrid_data (AIR_Data_Type.c)
// and the per level arrays are allocated by create_air_data (AIR_Data_Type_Routines.c),
// with levels_size entries each, and freed by destroy_air_data
// The levels are 0-based (our_level = 0 is the top grid), so the level arrays are
// indexed directly with our_level; the Fortran our_level is our_level + 1 here
// ~~~~~~~~
typedef struct air_multigrid_data_s {

   // Options for AIR multigrid - these are applied to all levels
   air_options options;

   // Actual number of levels in the hierarchy after coarsening
   int no_levels;
   // Do we have a gpu matrix or one on the cpu
   PetscBool fast_veciscopy_exists;
   // Can we reuse a one point classical prolongator
   // We only need this explicitly due to kokkos which is fussy
   // about the exact same pointers being passed into spgemm_numeric
   // (rather than just having the same sparsity) once we've
   // repartitioned
   PetscBool reuse_one_point_classical_prolong;

   // The number of entries in each of the per level arrays below (the Fortran
   // size() of them), 0 when they are not allocated
   PetscInt levels_size;

   // Indices of F and C points on each level
   IS *IS_fine_index, *IS_coarse_index;

   // Storage for ideal restrictors/prolongators that includes I block
   Mat *restrictors;
   Mat *prolongators;

   // Restrictor style injectors we use as gpus don't currently support veciscopy
   Mat *i_fine_full, *i_coarse_full;
   Mat *i_fine_full_full, *i_coarse_full_full;

   // Extracted matrices on each level
   Mat *A_ff, *A_fc, *A_cf, *A_cc;
   // Approximate inverse of Aff
   Mat *inv_A_ff;
   // GMRES polynomial data for inv_A_ff
   gmres_poly_data *inv_A_ff_poly_data;
   // GMRES polynomial data for inv_A_ff that has had entries dropped
   gmres_poly_data *inv_A_ff_poly_data_dropped;
   // Approximate inverse of Acc
   Mat *inv_A_cc;
   // GMRES polynomial data for inv_A_cc
   gmres_poly_data *inv_A_cc_poly_data;
   // Boolean to tell whether we have allocated petsc matrices
   PetscBool *allocated_matrices_A_ff, *allocated_matrices_A_cc;
   PetscBool *allocated_is;

   // Coarse matrices on each level
   Mat *coarse_matrix;
   // Boolean to tell whether we have allocated petsc matrices
   PetscBool *allocated_coarse_matrix;

   // The number of smooths we do on each level
   int_vec_array *smooth_order_levels;

   // The coarsest grid is stored in coarse_matrix(no_levels) and this is the
   // GMRES polynomial data for the coarsest grid solver
   // The approx inverse for the coarse grid solve is stored in inv_A_ff(no_levels)
   gmres_poly_data inv_coarsest_poly_data;

   // Temporary storage
   petsc_vec_array temp_vecs_fine[4], temp_vecs_coarse[4];
   petsc_vec_array temp_vecs[1];
   // The temporary vectors above are only ever scratch space within a single smooth
   // (or residual) on one level, so on every level each slot is a subvector of one pool,
   // sized for the biggest level - the IS's are the (contiguous) subsets of the pools
   // each level uses
   Vec temp_vecs_fine_pool[4], temp_vecs_coarse_pool[4];
   Vec temp_vecs_pool[1];
   IS *temp_vecs_fine_is, *temp_vecs_coarse_is, *temp_vecs_is;

   // Temporary dense storage used during a multiple rhs (block) smooth
   // These are the block twins of temp_vecs_fine/temp_vecs_coarse/temp_vecs,
   // with a fifth slot as the dedicated target of the cached off-diagonal
   // product (see setup_air_block_products)
   // They are lazily created from the block of rhs we're given, as the number
   // of columns isn't known until the block apply happens
   petsc_dense_mat_array block_temp_fine[5], block_temp_coarse[5];
   petsc_dense_mat_array block_temp_full[1];
   // The number of global (and local) columns the cached block temporaries
   // were built with - the local column layout has to match the block of rhs
   // we're given or the products would have incompatible layouts
   PetscInt block_ncols;
   PetscInt block_local_ncols;

   // Per-PCAIR opaque handle to the kokkos-side IS view storage. Set up by
   // create_VecISCopyLocal_kokkos, torn down by destroy_VecISCopyLocal_kokkos.
   // Stored here (instead of as a file-scope global in
   // src/VecISCopyLocalk.kokkos.cxx) so concurrent PCAIR instances do not
   // overwrite each other's per-level IS views.
   void *kokkos_is_views_handle;

   // Temporary reuse
   air_reuse_data *reuse;

   // ~~~~~~~~~~~~~~~~~~~~~~

   // Storage for nnzs of each matrix on each level
   PetscInt64 *coarse_matrix_nnzs, *A_ff_nnzs, *inv_A_ff_nnzs, *inv_A_cc_nnzs;
   PetscInt64 *restrictor_nnzs, *prolongator_nnzs, *A_fc_nnzs, *A_cf_nnzs, *A_cc_nnzs;

} air_multigrid_data;

// ~~~~~~~~
// 2D logical arrays controlling which reuse_mat and reuse_is entries are kept
// for each reuse_amount level (1, 2, or 3).  Dimension 1 is the mat/IS index,
// dimension 2 is the reuse_amount value.  Each row below corresponds to one
// MAT_*/IS_* index; the three columns give the active flag for amount=1/2/3.
// Indexed [MAT_*][reuse_amount - 1] (and [IS_*][reuse_amount - 1]), as reuse_amount
// is the 1, 2 or 3 the user sets
//
// Note: the CF splitting (IS_fine_index / IS_coarse_index, stored directly in
// air_multigrid_data and controlled by allocated_is) is ALWAYS kept whenever
// reuse_sparsity is enabled regardless of reuse_amount.  This is handled in
// reset_air_data: the IS block is gated only on (.NOT. reuse), not on reuse_amount.
// Without the CF splitting there is no basis for any other reuse.
//
// Amount 1: CF splitting + IS_REPARTITION
//           No reuse_mats are stored.  All matrices are rebuilt
//           from scratch each setup.  Saves only the PMISR/DDC CF-splitting cost.
//           and the ParMETIS calls for repartitioning in parallel
// Amount 2: CF splitting + IS_REPARTITION + SpGEMM matrices
//           (MAT_AP, MAT_RAP) + all matrices needed to guarantee stable SpGEMM
//           sparsity (MAT_RAP_DROP, MAT_Z_DROP, MAT_W_DROP).  MAT_RAP_DROP preserves coarse matrix
//           structure -> A_ff/A_fc/A_cf/A_cc have stable sparsity -> Z/W/R/P stable
//           -> MAT_AP and MAT_RAP sparsity is guaranteed unchanged on reuse.
// Amount 3: everything (preserves previous behaviour)
//
// Index order matches the MAT_* parameters in pflare_parameters:
// ~~~~~~~~
static const PetscBool REUSE_MAT_ACTIVE[22][3] = {
   //amt=1        amt=2        amt=3
   {PETSC_FALSE, PETSC_TRUE,  PETSC_TRUE},  //  1=AP
   {PETSC_FALSE, PETSC_TRUE,  PETSC_TRUE},  //  2=RAP
   {PETSC_FALSE, PETSC_TRUE,  PETSC_TRUE},  //  3=RAP_DROP
   {PETSC_FALSE, PETSC_TRUE,  PETSC_TRUE},  //  4=Z_DROP
   {PETSC_FALSE, PETSC_TRUE,  PETSC_TRUE},  //  5=W_DROP
   {PETSC_FALSE, PETSC_FALSE, PETSC_TRUE},  //  6=COARSE_REPARTITIONED
   {PETSC_FALSE, PETSC_FALSE, PETSC_TRUE},  //  7=P_REPARTITIONED
   {PETSC_FALSE, PETSC_FALSE, PETSC_TRUE},  //  8=R_REPARTITIONED
   {PETSC_FALSE, PETSC_TRUE,  PETSC_TRUE},  //  9=AFF_DROP
   {PETSC_FALSE, PETSC_TRUE,  PETSC_TRUE},  // 10=ACF_DROP
   {PETSC_FALSE, PETSC_TRUE,  PETSC_TRUE},  // 11=AFC_DROP
   {PETSC_FALSE, PETSC_TRUE,  PETSC_TRUE},  // 12=A_DROP
   {PETSC_FALSE, PETSC_TRUE,  PETSC_TRUE},  // 13=W
   {PETSC_FALSE, PETSC_TRUE,  PETSC_TRUE},  // 14=Z
   {PETSC_FALSE, PETSC_FALSE, PETSC_TRUE},  // 15=INV_AFF
   {PETSC_FALSE, PETSC_FALSE, PETSC_TRUE},  // 16=INV_AFF_DROPPED
   {PETSC_FALSE, PETSC_FALSE, PETSC_TRUE},  // 17=INV_ACC
   {PETSC_FALSE, PETSC_FALSE, PETSC_TRUE},  // 18=SAI_SUB
   {PETSC_FALSE, PETSC_FALSE, PETSC_TRUE},  // 19=Z_AFF
   {PETSC_FALSE, PETSC_FALSE, PETSC_TRUE},  // 20=Z_NO_SPARSITY
   {PETSC_FALSE, PETSC_FALSE, PETSC_TRUE},  // 21=W_AFF
   {PETSC_FALSE, PETSC_FALSE, PETSC_TRUE}   // 22=W_NO_SPARSITY
};

// Index order matches the IS_* parameters in pflare_parameters:
// IS_R_Z_FINE_COLS depends on Z's sparsity (see note above), so it is only safe
// to store when Z is also stored and its sparsity is guaranteed unchanged (amount>=2).
// IS_REPARTITION (graph partitioner output) is stored at all amounts.
static const PetscBool REUSE_IS_ACTIVE[2][3] = {
   //amt=1        amt=2        amt=3
   {PETSC_TRUE,  PETSC_TRUE,  PETSC_TRUE},  // 1=IS_REPARTITION
   {PETSC_FALSE, PETSC_TRUE,  PETSC_TRUE}   // 2=IS_R_Z_FINE_COLS
};

// ~~~~~~~~
// This is the pc->data of a PCAIR (pc_air_multigrid_data in PCAIR_Data_Type.F90, where
// it was the context type of our pcshells)
// Allocated by create_pc_air_multigrid_data (PCAIR_Data_Type.c) and freed with PetscFree
// ~~~~~~~~
typedef struct {

   air_multigrid_data air_data;
   // AIR under the hood is just a PCMG object
   PC pcmg;

} pc_air_multigrid_data;

#endif
