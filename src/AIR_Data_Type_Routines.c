/*
  C port of AIR_Data_Type_Routines.F90 (module air_data_type_routines)

  The create/reset/destroy trio of the air_multigrid_data struct in pflareimpl.h
  (convention 3 of docs/dev/fortran_removal_plan.md). The per level arrays are
  allocated with options.max_levels entries (levels_size) and are 0-based, so the
  loops over the levels run from 0 and the Fortran our_level is our_level + 1
  The Fortran allocate of an array of PETSc objects gives null handles, which is
  PetscCalloc1 here; the gmres_poly_data in the arrays get the defaults of the
  bind(C) Fortran mirror of the struct the Fortran held (no TSQR buffers until
  they are used)
*/

#include <petscmat.h>
#include "pflareimpl.h"
#include "Approx_Inverse_Setup.h"
#include "PETSc_Helper.h"
#include "Gmres_Poly_Data_Type.h"
#include "FC_Smooth.h"
#include "FC_Smooth_Block.h"
#include "AIR_Data_Type_Routines.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode create_air_data(air_multigrid_data *air_data)
{
   // Setup the data structures for air and reads options from the
   // command line

   // ~~~~~~
   PetscInt i_loc, max_levels;
   // ~~~~~~

   PetscFunctionBegin;

   air_data->no_levels = -1;
   max_levels = air_data->options.max_levels;

   // Allocate the AIR specific data structures
   PetscCall(PetscCalloc1(max_levels, &air_data->IS_fine_index));
   PetscCall(PetscCalloc1(max_levels, &air_data->IS_coarse_index));
   PetscCall(PetscCalloc1(max_levels, &air_data->temp_vecs_fine_is));
   PetscCall(PetscCalloc1(max_levels, &air_data->temp_vecs_coarse_is));
   PetscCall(PetscCalloc1(max_levels, &air_data->temp_vecs_is));

   PetscCall(PetscCalloc1(max_levels, &air_data->restrictors));
   PetscCall(PetscCalloc1(max_levels, &air_data->prolongators));

   PetscCall(PetscCalloc1(max_levels, &air_data->i_fine_full));
   PetscCall(PetscCalloc1(max_levels, &air_data->i_coarse_full));
   PetscCall(PetscCalloc1(max_levels, &air_data->i_fine_full_full));
   PetscCall(PetscCalloc1(max_levels, &air_data->i_coarse_full_full));

   PetscCall(PetscCalloc1(max_levels, &air_data->coarse_matrix));
   PetscCall(PetscCalloc1(max_levels, &air_data->A_ff));
   PetscCall(PetscCalloc1(max_levels, &air_data->inv_A_ff));
   PetscCall(PetscCalloc1(max_levels, &air_data->inv_A_ff_poly_data));
   PetscCall(PetscCalloc1(max_levels, &air_data->inv_A_ff_poly_data_dropped));
   PetscCall(PetscCalloc1(max_levels, &air_data->inv_A_cc));
   PetscCall(PetscCalloc1(max_levels, &air_data->inv_A_cc_poly_data));
   PetscCall(PetscCalloc1(max_levels, &air_data->A_fc));
   PetscCall(PetscCalloc1(max_levels, &air_data->A_cf));
   PetscCall(PetscCalloc1(max_levels, &air_data->A_cc));
   // The defaults of the gmres_poly_data the Fortran allocated
   for (i_loc = 0; i_loc < max_levels; i_loc++) {
      air_data->inv_A_ff_poly_data[i_loc].gmres_poly_order = 6;
      air_data->inv_A_ff_poly_data[i_loc].gmres_poly_sparsity_order = 1;
      air_data->inv_A_ff_poly_data_dropped[i_loc].gmres_poly_order = 6;
      air_data->inv_A_ff_poly_data_dropped[i_loc].gmres_poly_sparsity_order = 1;
      air_data->inv_A_cc_poly_data[i_loc].gmres_poly_order = 6;
      air_data->inv_A_cc_poly_data[i_loc].gmres_poly_sparsity_order = 1;
   }

   PetscCall(PetscCalloc1(max_levels, &air_data->prolongator_nnzs));
   PetscCall(PetscCalloc1(max_levels, &air_data->restrictor_nnzs));
   PetscCall(PetscCalloc1(max_levels, &air_data->A_ff_nnzs));
   PetscCall(PetscCalloc1(max_levels, &air_data->A_fc_nnzs));
   PetscCall(PetscCalloc1(max_levels, &air_data->A_cf_nnzs));
   PetscCall(PetscCalloc1(max_levels, &air_data->A_cc_nnzs));
   PetscCall(PetscCalloc1(max_levels, &air_data->inv_A_ff_nnzs));
   PetscCall(PetscCalloc1(max_levels, &air_data->inv_A_cc_nnzs));
   PetscCall(PetscCalloc1(max_levels, &air_data->coarse_matrix_nnzs));

   PetscCall(PetscCalloc1(max_levels, &air_data->allocated_matrices_A_ff));
   PetscCall(PetscCalloc1(max_levels, &air_data->allocated_matrices_A_cc));
   PetscCall(PetscCalloc1(max_levels, &air_data->allocated_is));
   PetscCall(PetscCalloc1(max_levels, &air_data->allocated_coarse_matrix));

   PetscCall(PetscCalloc1(max_levels, &air_data->smooth_order_levels));

   // Temporary vectors
   for (i_loc = 0; i_loc < 4; i_loc++) {
      PetscCall(PetscCalloc1(max_levels, &air_data->temp_vecs_fine[i_loc].array));
   }
   for (i_loc = 0; i_loc < 4; i_loc++) {
      PetscCall(PetscCalloc1(max_levels, &air_data->temp_vecs_coarse[i_loc].array));
   }
   PetscCall(PetscCalloc1(max_levels, &air_data->temp_vecs[0].array));

   // Temporary dense blocks used during a multiple rhs (block) smooth
   // Only the outer arrays are allocated here, the blocks themselves are
   // built lazily in ensure_air_block_temps once we know how many columns
   // we have been given
   for (i_loc = 0; i_loc < (PetscInt)PETSC_STATIC_ARRAY_LENGTH(air_data->block_temp_fine); i_loc++) {
      PetscCall(PetscCalloc1(max_levels, &air_data->block_temp_fine[i_loc].array));
      PetscCall(PetscCalloc1(max_levels, &air_data->block_temp_coarse[i_loc].array));
   }
   PetscCall(PetscCalloc1(max_levels, &air_data->block_temp_full[0].array));
   air_data->block_ncols = -1;
   air_data->block_local_ncols = -1;

   // Reuse
   PetscCall(PetscCalloc1(max_levels, &air_data->reuse));

   // The size of all the per level arrays above
   air_data->levels_size = max_levels;

   // nnzs counts
   // The counts are zero and the allocated flags false from PetscCalloc1

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode reset_air_data(air_multigrid_data *air_data, const PetscBool *keep_reuse)
{
   // Resets the data structures for air

   // ~~~~~~
   PetscInt our_level;
   PetscInt i_loc;
   PetscBool reuse;
   // ~~~~~~

   PetscFunctionBegin;

   reuse = PETSC_FALSE;
   if (keep_reuse) reuse = *keep_reuse;

   // The dense scratch blocks are sized from A_ff/A_cf/coarse_matrix on each
   // level, so they have to go whenever we reset regardless of whether we are
   // reusing - the number of levels and the layouts on them can both change
   // when we build again. They are cheap to rebuild, ensure_air_block_temps
   // does that lazily on the next block apply
   PetscCall(destroy_air_block_temps(air_data));

   // Use if this data structure is allocated to determine if we setup anything
   if (air_data->allocated_matrices_A_ff) {

      // Loop over the levels
      for (our_level = 0; our_level < air_data->levels_size; our_level++) {

         // If we setup Aff
         if (air_data->allocated_matrices_A_ff[our_level]) {

            // Destroy data that depends on the CF splitting and poly coefficients
            // only when not reusing (the IS device copy and poly coefficients
            // remain valid across setups when reusing)
            if (!reuse) {
               PetscCall(destroy_VecISCopyLocalWrapper(air_data, (int)our_level));
               PetscCall(reset_poly_data(&air_data->inv_A_ff_poly_data[our_level]));
               PetscCall(reset_poly_data(&air_data->inv_A_ff_poly_data_dropped[our_level]));
            }

            // temp_vecs are sized from A_ff/A_fc; they must be destroyed whenever
            // A_ff/A_fc are destroyed (amounts 1 and 2 destroy them even when reusing)
            // They are subvectors of the pools, so restore them
            if (!reuse || air_data->options.reuse_amount < 3) {
               for (i_loc = 0; i_loc < 4; i_loc++) {
                  if (air_data->temp_vecs_fine[i_loc].array[our_level]) {
                     PetscCall(VecRestoreSubVector(air_data->temp_vecs_fine_pool[i_loc],
                              air_data->temp_vecs_fine_is[our_level],
                              &air_data->temp_vecs_fine[i_loc].array[our_level]));
                  }
                  if (air_data->temp_vecs_coarse[i_loc].array[our_level]) {
                     PetscCall(VecRestoreSubVector(air_data->temp_vecs_coarse_pool[i_loc],
                              air_data->temp_vecs_coarse_is[our_level],
                              &air_data->temp_vecs_coarse[i_loc].array[our_level]));
                  }
               }
               if (air_data->temp_vecs[0].array[our_level]) {
                  PetscCall(VecRestoreSubVector(air_data->temp_vecs_pool[0],
                           air_data->temp_vecs_is[our_level],
                           &air_data->temp_vecs[0].array[our_level]));
               }
               PetscCall(ISDestroy(&air_data->temp_vecs_fine_is[our_level]));
               PetscCall(ISDestroy(&air_data->temp_vecs_coarse_is[our_level]));
               PetscCall(ISDestroy(&air_data->temp_vecs_is[our_level]));
            }

            // Only amount=3 preserves A_ff/A_fc/A_cf and grid-transfer operators
            // between setups.  For amount<=2 these are destroyed and rebuilt
            // from scratch (SpGEMM reuse at amount=2 works via stored W/Z/AP/RAP
            // whose sparsity is guaranteed by stored RAP_DROP).
            if (!reuse || air_data->options.reuse_amount < 3) {

               PetscCall(MatDestroy(&air_data->prolongators[our_level]));
               // When symmetric R is only kept between setups at amount=3,
               // otherwise this is null already
               PetscCall(MatDestroy(&air_data->restrictors[our_level]));
               PetscCall(reset_inverse_mat(&air_data->inv_A_ff[our_level]));
               PetscCall(MatDestroy(&air_data->A_ff[our_level]));
               PetscCall(MatDestroy(&air_data->A_fc[our_level]));
               PetscCall(MatDestroy(&air_data->A_cf[our_level]));
               air_data->allocated_matrices_A_ff[our_level] = PETSC_FALSE;
            }
         }

         // IS_fine_index and IS_coarse_index (the CF splitting) are always kept
         // whenever reuse=.TRUE., regardless of reuse_amount.  The CF splitting
         // is the basis for all other reuse at every amount level.
         if (air_data->allocated_is[our_level]) {
            if (!reuse) {
               PetscCall(ISDestroy(&air_data->IS_fine_index[our_level]));
               PetscCall(ISDestroy(&air_data->IS_coarse_index[our_level]));
               air_data->allocated_is[our_level] = PETSC_FALSE;
            }
         }

         // Did we do C point smoothing?
         if (air_data->allocated_matrices_A_cc[our_level]) {
            // Same logic as A_ff: only amount=3 preserves A_cc between setups.
            if (!reuse || air_data->options.reuse_amount < 3) {
               PetscCall(MatDestroy(&air_data->A_cc[our_level]));
               PetscCall(reset_inverse_mat(&air_data->inv_A_cc[our_level]));
               PetscCall(reset_poly_data(&air_data->inv_A_cc_poly_data[our_level]));
               air_data->allocated_matrices_A_cc[our_level] = PETSC_FALSE;
            }
         }
         // Did we create a coarse grid on this level
         if (air_data->allocated_coarse_matrix[our_level]) {
            PetscCall(reset_inverse_mat(&air_data->coarse_matrix[our_level]));
         }

         // Destroy the reuse data if needed
         if (!reuse) {
            for (i_loc = 0; i_loc < (PetscInt)PETSC_STATIC_ARRAY_LENGTH(air_data->reuse[our_level].reuse_mat); i_loc++) {
               if (air_data->reuse[our_level].reuse_mat[i_loc]) {
                  PetscCall(destroy_matrix_reuse(&air_data->reuse[our_level].reuse_mat[i_loc],
                     &air_data->reuse[our_level].reuse_submatrices[i_loc].array));
               }
            }

            for (i_loc = 0; i_loc < (PetscInt)PETSC_STATIC_ARRAY_LENGTH(air_data->reuse[our_level].reuse_is); i_loc++) {
               if (air_data->reuse[our_level].reuse_is[i_loc]) {
                  PetscCall(ISDestroy(&air_data->reuse[our_level].reuse_is[i_loc]));
               }
            }
         } else {
            // When reusing, destroy any reuse entries not active at the
            // current amount level (handles amount being lowered between setups)
            for (i_loc = 0; i_loc < (PetscInt)PETSC_STATIC_ARRAY_LENGTH(air_data->reuse[our_level].reuse_mat); i_loc++) {
               if (!REUSE_MAT_ACTIVE[i_loc][air_data->options.reuse_amount - 1]) {
                  if (air_data->reuse[our_level].reuse_mat[i_loc]) {
                     PetscCall(destroy_matrix_reuse(&air_data->reuse[our_level].reuse_mat[i_loc],
                        &air_data->reuse[our_level].reuse_submatrices[i_loc].array));
                  }
               }
            }

            for (i_loc = 0; i_loc < (PetscInt)PETSC_STATIC_ARRAY_LENGTH(air_data->reuse[our_level].reuse_is); i_loc++) {
               if (!REUSE_IS_ACTIVE[i_loc][air_data->options.reuse_amount - 1]) {
                  if (air_data->reuse[our_level].reuse_is[i_loc]) {
                     PetscCall(ISDestroy(&air_data->reuse[our_level].reuse_is[i_loc]));
                  }
               }
            }
         }
      }

      // The pools the temporary vectors are subvectors of - these go
      // whenever the temporary vectors do, which have been restored above
      if (!reuse || air_data->options.reuse_amount < 3) {
         for (i_loc = 0; i_loc < 4; i_loc++) {
            PetscCall(VecDestroy(&air_data->temp_vecs_fine_pool[i_loc]));
            PetscCall(VecDestroy(&air_data->temp_vecs_coarse_pool[i_loc]));
         }
         PetscCall(VecDestroy(&air_data->temp_vecs_pool[0]));
      }

      if (air_data->no_levels != -1) {
         // Coarse grid solver
         // Reset when not reusing, or when reusing but inv_A_ff is not stored
         // at this reuse_amount level (amounts 1 and 2); otherwise reuse_triggered
         // would be TRUE in build_gmres_polynomial_inverse even though the old
         // inv_A_ff is stale relative to the rebuilt coarse matrix.
         if (!reuse ||
              !REUSE_MAT_ACTIVE[MAT_INV_AFF][air_data->options.reuse_amount - 1]) {
            PetscCall(reset_inverse_mat(&air_data->inv_A_ff[air_data->no_levels - 1]));
            PetscCall(reset_poly_data(&air_data->inv_coarsest_poly_data));
         }
      }
   }

   // Reset data
   air_data->no_levels = -1;
   for (our_level = 0; our_level < air_data->levels_size; our_level++) {
      air_data->restrictor_nnzs[our_level]      = 0;
      air_data->prolongator_nnzs[our_level]     = 0;
      air_data->inv_A_ff_nnzs[our_level]        = 0;
      air_data->A_fc_nnzs[our_level]            = 0;
      air_data->A_ff_nnzs[our_level]            = 0;
      air_data->A_cf_nnzs[our_level]            = 0;
      air_data->A_cc_nnzs[our_level]            = 0;
      air_data->inv_A_cc_nnzs[our_level]        = 0;
      air_data->coarse_matrix_nnzs[our_level]   = 0;
      air_data->allocated_coarse_matrix[our_level] = PETSC_FALSE;
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode destroy_air_data(air_multigrid_data *air_data)
{
   // Destroys the data structures for air

   // ~~~~~~
   PetscInt our_level, i_loc;
   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(reset_air_data(air_data, NULL));
   // The coarsest grid gmres_poly_data is C memory (the TSQR buffers), which the
   // Fortran type used to deallocate with air_data
   PetscCall(destroy_poly_data(&air_data->inv_coarsest_poly_data));

   // Now set the options back to the default
   air_data->options.print_stats_timings = PETSC_FALSE;

   air_data->options.max_levels = 300;
   air_data->options.coarse_eq_limit = 6;
   air_data->options.auto_truncate_start_level = -1;
   air_data->options.auto_truncate_tol = (PetscReal)PFLARE_TOL_AUTO_TRUNCATE;
   air_data->options.processor_agglom = PETSC_TRUE;
   air_data->options.processor_agglom_ratio = 2;
   air_data->options.processor_agglom_factor = 2;
   air_data->options.process_eq_limit = 50;
   air_data->options.subcomm = PETSC_FALSE;

   air_data->options.strong_threshold = (PetscReal)0.5f;
   air_data->options.ddc_its = 1;
   air_data->options.ddc_fraction = (PetscReal)0.1f;
   air_data->options.cf_splitting_type = (CFSplittingType)0;
   air_data->options.max_luby_steps = -1;

   for (i_loc = 0; i_loc < AIR_SMOOTH_ORDER_SIZE; i_loc++) air_data->options.smooth_order[i_loc] = 0;
   air_data->options.smooth_order[0] = 2;
   air_data->options.any_c_smooths = PETSC_FALSE;
   air_data->options.diag_scale_polys = PETSC_FALSE;
   air_data->options.matrix_free_polys = PETSC_FALSE;
   air_data->options.one_point_classical_prolong = PETSC_TRUE;
   air_data->options.full_smoothing_up_and_down = PETSC_FALSE;
   air_data->options.symmetric = PETSC_FALSE;
   air_data->options.constrain_w = PETSC_FALSE;
   air_data->options.constrain_z = PETSC_FALSE;
   air_data->options.improve_z_its = 0;
   air_data->options.improve_w_its = 0;

   air_data->options.strong_r_threshold = 0.0;

   air_data->options.inverse_type = (PCPFLAREINVType)PFLARE_PFLAREINV_ARNOLDI;

   air_data->options.z_type = (PCAIRZType)PFLARE_AIR_Z_PRODUCT;

   air_data->options.lair_distance = 2;

   air_data->options.poly_order = 6;
   air_data->options.inverse_sparsity_order = 1;

   // -1 means unset, i.e., use the same as the F point smoother
   air_data->options.c_inverse_type = -1;
   air_data->options.c_poly_order = -1;
   air_data->options.c_inverse_sparsity_order = -1;

   air_data->options.coarsest_inverse_type = (PCPFLAREINVType)PFLARE_PFLAREINV_ARNOLDI;
   air_data->options.coarsest_poly_order = 6;
   air_data->options.coarsest_inverse_sparsity_order = 1;
   air_data->options.coarsest_matrix_free_polys = PETSC_FALSE;
   air_data->options.coarsest_diag_scale_polys = PETSC_FALSE;
   air_data->options.coarsest_subcomm = PETSC_FALSE;

   air_data->options.r_drop = (PetscReal)0.01f;
   air_data->options.a_drop = (PetscReal)1e-4f;
   air_data->options.a_lump = PETSC_FALSE;

   air_data->options.reuse_sparsity = PETSC_FALSE;
   air_data->options.reuse_poly_coeffs = PETSC_FALSE;

   // Use if this data structure is allocated to determine if we setup anything
   if (air_data->allocated_matrices_A_ff) {

      // Deallocate the allocated structures
      PetscCall(PetscFree(air_data->IS_fine_index));
      PetscCall(PetscFree(air_data->IS_coarse_index));
      PetscCall(PetscFree(air_data->temp_vecs_fine_is));
      PetscCall(PetscFree(air_data->temp_vecs_coarse_is));
      PetscCall(PetscFree(air_data->temp_vecs_is));

      PetscCall(PetscFree(air_data->restrictors));
      PetscCall(PetscFree(air_data->prolongators));

      PetscCall(PetscFree(air_data->i_fine_full));
      PetscCall(PetscFree(air_data->i_coarse_full));
      PetscCall(PetscFree(air_data->i_fine_full_full));
      PetscCall(PetscFree(air_data->i_coarse_full_full));

      PetscCall(PetscFree(air_data->coarse_matrix));
      PetscCall(PetscFree(air_data->A_ff));
      PetscCall(PetscFree(air_data->inv_A_ff));
      // The gmres_poly_data are C memory (the TSQR buffers), which the Fortran
      // type used to deallocate with the arrays
      for (our_level = 0; our_level < air_data->levels_size; our_level++) {
         PetscCall(destroy_poly_data(&air_data->inv_A_ff_poly_data[our_level]));
         PetscCall(destroy_poly_data(&air_data->inv_A_ff_poly_data_dropped[our_level]));
         PetscCall(destroy_poly_data(&air_data->inv_A_cc_poly_data[our_level]));
      }
      PetscCall(PetscFree(air_data->inv_A_ff_poly_data));
      PetscCall(PetscFree(air_data->inv_A_ff_poly_data_dropped));
      PetscCall(PetscFree(air_data->inv_A_cc));
      PetscCall(PetscFree(air_data->inv_A_cc_poly_data));
      PetscCall(PetscFree(air_data->A_fc));
      PetscCall(PetscFree(air_data->A_cf));
      PetscCall(PetscFree(air_data->A_cc));

      PetscCall(PetscFree(air_data->allocated_matrices_A_ff));
      PetscCall(PetscFree(air_data->allocated_matrices_A_cc));
      PetscCall(PetscFree(air_data->allocated_is));
      PetscCall(PetscFree(air_data->allocated_coarse_matrix));

      for (our_level = 0; our_level < air_data->levels_size; our_level++) {
         if (air_data->smooth_order_levels[our_level].array) {
            PetscCall(PetscFree(air_data->smooth_order_levels[our_level].array));
            air_data->smooth_order_levels[our_level].size = 0;
         }
      }
      PetscCall(PetscFree(air_data->smooth_order_levels));

      PetscCall(PetscFree(air_data->temp_vecs[0].array));
      for (i_loc = 0; i_loc < 4; i_loc++) {
         PetscCall(PetscFree(air_data->temp_vecs_fine[i_loc].array));
      }
      for (i_loc = 0; i_loc < 4; i_loc++) {
         PetscCall(PetscFree(air_data->temp_vecs_coarse[i_loc].array));
      }

      // The blocks themselves have already been destroyed in reset_air_data
      for (i_loc = 0; i_loc < (PetscInt)PETSC_STATIC_ARRAY_LENGTH(air_data->block_temp_fine); i_loc++) {
         PetscCall(PetscFree(air_data->block_temp_fine[i_loc].array));
         PetscCall(PetscFree(air_data->block_temp_coarse[i_loc].array));
      }
      PetscCall(PetscFree(air_data->block_temp_full[0].array));

      PetscCall(PetscFree(air_data->reuse));

      // Delete the nnzs
      PetscCall(PetscFree(air_data->restrictor_nnzs));
      PetscCall(PetscFree(air_data->prolongator_nnzs));
      PetscCall(PetscFree(air_data->A_ff_nnzs));
      PetscCall(PetscFree(air_data->A_fc_nnzs));
      PetscCall(PetscFree(air_data->A_cf_nnzs));
      PetscCall(PetscFree(air_data->A_cc_nnzs));
      PetscCall(PetscFree(air_data->inv_A_ff_nnzs));
      PetscCall(PetscFree(air_data->inv_A_cc_nnzs));
      PetscCall(PetscFree(air_data->coarse_matrix_nnzs));

      // None of the per level arrays are allocated now
      air_data->levels_size = 0;
   }

   air_data->no_levels = -1;

   PetscFunctionReturn(PETSC_SUCCESS);
}
