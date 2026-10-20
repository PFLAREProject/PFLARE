/*
  C port of AIR_Data_Type.F90 (module air_data_type)

  The Fortran module only defined the types, which are the structs in
  pflareimpl.h, and the REUSE_MAT_ACTIVE/REUSE_IS_ACTIVE tables, which are
  static const tables there. What the Fortran did through the default
  initialisation of the types is done here explicitly: create_air_options gives
  the defaults of air_options and create_air_multigrid_data those of
  air_multigrid_data (convention 3 of docs/dev/fortran_removal_plan.md). The per
  level arrays are allocated by create_air_data in AIR_Data_Type_Routines.c, as
  in the Fortran
*/

#include <petscmat.h>
#include "pflareimpl.h"
#include "AIR_Data_Type.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode create_air_options(air_options *options)
{
   // Sets the defaults of the Fortran type air_options
   // The real defaults that were default (single precision) real literals in the
   // Fortran keep that value, ie they are float literals converted to PetscReal

   // ~~~~~~
   PetscInt i_loc;
   // ~~~~~~

   PetscFunctionBegin;

   // Print out stats and timings
   options->print_stats_timings = PETSC_FALSE;

   // Maximum number of levels in the multigrid hierarchy
   options->max_levels = 300;
   // Minimum number of global unknowns on the coarse grid
   options->coarse_eq_limit = 6;
   // From this level onwards, build and then evaluate if the coarse grid solver
   // is good enough and use that to determine if we should truncate on that level
   options->auto_truncate_start_level = -1;
   // What relative tolerance to use to determine if a coarse grid solver is good enough
   options->auto_truncate_tol = (PetscReal)PFLARE_TOL_AUTO_TRUNCATE;

   // Perform processor agglomeration throughout the hierarchy
   options->processor_agglom = PETSC_TRUE;
   // The local to nonlocal ratio of nnzs that is used to
   // trigger processor agglomeration on all level
   options->processor_agglom_ratio = 2;
   // What factor to reduce the number of active MPI ranks by
   // each time when doing processor agglomeration
   options->processor_agglom_factor = 2;
   // If on average there are fewer than this number of equations per rank
   // processor agglomeration will be triggered
   options->process_eq_limit = 50;
   // Reductions on a subcomm
   options->subcomm = PETSC_FALSE;

   // This is used in the CF splitting to define strong dependencies/influences
   options->strong_threshold = (PetscReal)0.5f;
   // How many passes of DDC to do
   options->ddc_its = 1;
   // Second pass in the PMISR DDC CF splitting converts
   // this fraction of local F points to C based on diagonal dominance
   options->ddc_fraction = (PetscReal)0.1f;
   // What CF splitting algorithm to use
   options->cf_splitting_type = (CFSplittingType)0;
   // Maximum number of Luby steps to do in CF splitting
   options->max_luby_steps = -1;

   // Stores what smoothing happens
   // 2 F smooths by default
   for (i_loc = 0; i_loc < AIR_SMOOTH_ORDER_SIZE; i_loc++) options->smooth_order[i_loc] = 0;
   options->smooth_order[0] = 2;
   // Do we do any C point smooths?
   options->any_c_smooths = PETSC_FALSE;
   // If using a polynomial inverse type, do we diagonally scale before computing?
   options->diag_scale_polys = PETSC_FALSE;
   // Do we apply our polynomials matrix free when smoothing?
   options->matrix_free_polys = PETSC_FALSE;
   // Do we use a one point injection classical prolongator or an AIR-style prolongator
   options->one_point_classical_prolong = PETSC_TRUE;
   // Do we do full smoothing up and down rather than FF or FFC
   options->full_smoothing_up_and_down = PETSC_FALSE;
   // Do we define our prolongator as R^T?
   options->symmetric = PETSC_FALSE;
   // Do we compute a near-nullspace vector and apply it as a constaint
   // to the prolongator?
   options->constrain_w = PETSC_FALSE;
   // Do we compute a near-nullspace vector and apply it as a constaint
   // to the restrictor?
   options->constrain_z = PETSC_FALSE;
   // How many iterations of improvement do we want to do to Z
   options->improve_z_its = 0;
   // How many iterations of improvement do we want to do to W
   options->improve_w_its = 0;

   // Strong R threshold to apply dropping prior to computing Z
   options->strong_r_threshold = 0.0;

   // What type of approximation do we use for Z?
   options->z_type = (PCAIRZType)0;

   // If z_type == 1 or 2, this is the distance the grid-transfer operators go out to
   options->lair_distance = 2;

   // What type of approximation do we use for Aff^-1
   options->inverse_type = (PCPFLAREINVType)1;

   // This is the order of polynomial we use in air if inverse_type is
   // power, arnoldi, newton or neumann
   options->poly_order = 6;
   // This is the order of sparsity we use if we assemble our approximate inverses
   options->inverse_sparsity_order = 1;

   // Inverse type for c smoothing
   // If unset (-1) this defaults to whatever the F point smoother is
   options->c_inverse_type = -1;
   // Poly order for c smoothing
   options->c_poly_order = -1;
   // Inverse sparsity order for c smoothing
   options->c_inverse_sparsity_order = -1;

   // These are for the coarse grid solver
   options->coarsest_inverse_type = (PCPFLAREINVType)1;
   options->coarsest_poly_order = 6;
   options->coarsest_inverse_sparsity_order = 1;
   options->coarsest_matrix_free_polys = PETSC_FALSE;
   options->coarsest_diag_scale_polys = PETSC_FALSE;
   options->coarsest_subcomm = PETSC_FALSE;

   // These are the relative drop tolerances (inf norm) on R and A after they are built
   options->r_drop = (PetscReal)0.01f;
   options->a_drop = (PetscReal)1e-4f;
   // Whether to lump in A or drop
   options->a_lump = PETSC_FALSE;

   // Whether or not to re-use the existing sparsity when PCSetup is called
   // with SAME_NONZERO_PATTERN
   options->reuse_sparsity = PETSC_FALSE;

   // Controls how much data is stored for reuse when reuse_sparsity is enabled.
   options->reuse_amount = 3;

   // Whether or not to also re-use the gmres polynomial coefficients when
   // reuse_sparsity is set to true
   options->reuse_poly_coeffs = PETSC_FALSE;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode create_air_multigrid_data(air_multigrid_data *air_data)
{
   // Sets the defaults of the Fortran type air_multigrid_data
   // The per level arrays start unallocated (NULL), as do the matrices, vectors
   // and index sets that aren't in them, which the Fortran PETSc types start as null

   // ~~~~~~
   PetscInt i_loc;
   // ~~~~~~

   PetscFunctionBegin;

   // Everything starts zero, NULL or false
   PetscCall(PetscMemzero(air_data, sizeof(air_multigrid_data)));

   // Options for AIR multigrid - these are applied to all levels
   PetscCall(create_air_options(&air_data->options));

   // Actual number of levels in the hierarchy after coarsening
   air_data->no_levels = -1;
   // Do we have a gpu matrix or one on the cpu
   air_data->fast_veciscopy_exists = PETSC_FALSE;
   // Can we reuse a one point classical prolongator
   air_data->reuse_one_point_classical_prolong = PETSC_FALSE;

   // The GMRES polynomial data for the coarsest grid solver
   // These are the defaults of the bind(C) Fortran mirror of gmres_poly_data the
   // Fortran type held, which has no TSQR buffers until they are used (poly_data_buffers)
   air_data->inv_coarsest_poly_data.inverse_type = 0;
   air_data->inv_coarsest_poly_data.gmres_poly_order = 6;
   air_data->inv_coarsest_poly_data.gmres_poly_sparsity_order = 1;

   // The temporary vectors and their pools start null
   for (i_loc = 0; i_loc < 4; i_loc++) {
      air_data->temp_vecs_fine_pool[i_loc] = NULL;
      air_data->temp_vecs_coarse_pool[i_loc] = NULL;
   }
   air_data->temp_vecs_pool[0] = NULL;

   // The number of global (and local) columns the cached block temporaries
   // were built with
   air_data->block_ncols = -1;
   air_data->block_local_ncols = -1;

   // Per-PCAIR opaque handle to the kokkos-side IS view storage
   air_data->kokkos_is_views_handle = NULL;

   PetscFunctionReturn(PETSC_SUCCESS);
}
