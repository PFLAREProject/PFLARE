/*
  C port of AIR_Operators_Setup.F90 (module air_operators_setup)

  The levels are 0-based, so our_level indexes the per level arrays of
  air_multigrid_data directly (the Fortran our_level is our_level + 1). The
  Kokkos IS views that MatCreateSubMatrixWrapper can use number the levels from
  1, as the Fortran did, so it is given our_level_kokkos = our_level + 1
  The reuse slots are the 0-based MAT_* and IS_* of pflareimpl.h and the REUSE_*_ACTIVE
  tables are indexed with reuse_amount - 1
  The Fortran called start/finish_approximate_inverse with the buffers and the
  coefficients of a gmres_poly_data, which is what is passed here (poly_data_buffers
  creates the TSQR buffers on first use)
*/

#include <petscksp.h>
#include "pflareimpl.h"
#include "Constrain_Z_or_W.h"
#include "Approx_Inverse_Setup.h"
#include "Gmres_Poly_Data_Type.h"
#include "SAI_Z.h"
#include "PETSc_Helper.h"
#include "Timers.h"
#include "Grid_Transfer.h"
#include "Grid_Transfer_Improve.h"
#include "AIR_Operators_Setup.h"

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode get_submatrices_start_poly_coeff_comms(Mat input_mat, int our_level, air_multigrid_data *air_data)
{
   // Gets the submatrices we need for our multigrid and starts off the comms required
   // to compute the approximate inverse of A_ff

   // ~~~~~~~~~~
   Mat smoothing_mat;
   tsqr_buffers *buffers;
   // The Kokkos IS views number the levels from 1
   int our_level_kokkos = our_level + 1;
   int one = 1, zero = 0;
   PetscBool true_bool = PETSC_TRUE, false_bool = PETSC_FALSE;
   // ~~~~~~~~~~

   PetscFunctionBegin;

   // ~~~~~~~~~~~~~
   // Now to apply a strong R tolerance as lAIR in hypre does, we have to drop entries
   // from A_cf and A_ff according to the strong R tolerance on A
   // ~~~~~~~~~~~~~
   if (air_data->options.strong_r_threshold == 0.0) {

      // Copy the original pointer
      air_data->reuse[our_level].reuse_mat[MAT_AFF_DROP] = air_data->A_ff[our_level];
      // Increase the reference counter
      PetscCall(PetscObjectReference((PetscObject)air_data->A_ff[our_level]));

   // If we're dropping
   } else {

      PetscCall(timer_start(PFLARE_TIMER_ID_AIR_DROP));

      // If we want to reuse, we have to match the original sparsity
      // which might be different if the matrix has changed (but the sparsity is the same)
      // so we can't just drop with a drop tolerance
      if (air_data->reuse[our_level].reuse_mat[MAT_A_DROP]) {

         PetscCall(remove_from_sparse_match_c(input_mat, air_data->reuse[our_level].reuse_mat[MAT_A_DROP], NULL, NULL));

      } else {

         // Drop entries smaller than the strong R threshold
         // but make sure not to drop the diagonal entry!
         PetscCall(remove_small_from_sparse(input_mat, air_data->options.strong_r_threshold,
                        &air_data->reuse[our_level].reuse_mat[MAT_A_DROP],
                        &one, NULL, &zero, NULL));
      }

      PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_DROP));

      PetscCall(timer_start(PFLARE_TIMER_ID_AIR_EXTRACT));

      // Pull out A_ff
      if (air_data->reuse[our_level].reuse_mat[MAT_AFF_DROP]) {
         PetscCall(MatCreateSubMatrixWrapper(air_data->reuse[our_level].reuse_mat[MAT_A_DROP],
                     air_data->IS_fine_index[our_level], air_data->IS_fine_index[our_level], MAT_REUSE_MATRIX,
                     &air_data->reuse[our_level].reuse_mat[MAT_AFF_DROP],
                     &our_level_kokkos, &true_bool, &true_bool,
                     air_data->kokkos_is_views_handle));
      } else {
         PetscCall(MatCreateSubMatrixWrapper(air_data->reuse[our_level].reuse_mat[MAT_A_DROP],
                     air_data->IS_fine_index[our_level], air_data->IS_fine_index[our_level], MAT_INITIAL_MATRIX,
                     &air_data->reuse[our_level].reuse_mat[MAT_AFF_DROP],
                     &our_level_kokkos, &true_bool, &true_bool,
                     air_data->kokkos_is_views_handle));
      }

      PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_EXTRACT));

   }

   // ~~~~~~~~~~~~~~
   // Start building approximate inverse - A_ff^-1
   // ~~~~~~~~~~~~~~
   PetscCall(timer_start(PFLARE_TIMER_ID_AIR_INVERSE));

   // This is for the smoother
   if (!air_data->options.full_smoothing_up_and_down) {
      smoothing_mat = air_data->A_ff[our_level];
   } else {
      smoothing_mat = input_mat;
   }

   // Compute the inverse for smoothing
   // If we are re-using the polynomial coefficients then we don't have to do this
   if (!(air_data->inv_A_ff[our_level] &&
               air_data->options.reuse_poly_coeffs)) {
      PetscCall(poly_data_buffers(&air_data->inv_A_ff_poly_data[our_level], &buffers));
      PetscCall(start_approximate_inverse(smoothing_mat,
               air_data->inv_A_ff_poly_data[our_level].inverse_type,
               air_data->inv_A_ff_poly_data[our_level].gmres_poly_order,
               air_data->options.diag_scale_polys,
               buffers, &air_data->inv_A_ff_poly_data[our_level].coefficients,
               &air_data->inv_A_ff_poly_data[our_level].coefficients_rows,
               &air_data->inv_A_ff_poly_data[our_level].coefficients_cols));
   }

   // If we are doing AIRG
   // then we need an A_ff^-1 that we use to build the grid-transfer operators
   // but if the strong R threshold is zero we just use the one computed above
   if (air_data->options.z_type == PFLARE_AIR_Z_PRODUCT &&
            (air_data->options.strong_r_threshold != 0.0 ||
             air_data->options.full_smoothing_up_and_down)) {

      if (!(air_data->reuse[our_level].reuse_mat[MAT_AFF_DROP] &&
                  air_data->options.reuse_poly_coeffs)) {
         PetscCall(poly_data_buffers(&air_data->inv_A_ff_poly_data_dropped[our_level], &buffers));
         PetscCall(start_approximate_inverse(air_data->reuse[our_level].reuse_mat[MAT_AFF_DROP],
               air_data->inv_A_ff_poly_data_dropped[our_level].inverse_type,
               air_data->inv_A_ff_poly_data_dropped[our_level].gmres_poly_order,
               air_data->options.diag_scale_polys,
               buffers, &air_data->inv_A_ff_poly_data_dropped[our_level].coefficients,
               &air_data->inv_A_ff_poly_data_dropped[our_level].coefficients_rows,
               &air_data->inv_A_ff_poly_data_dropped[our_level].coefficients_cols));
      }
   }

   PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_INVERSE));

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   // If we are doing C point smoothing then we need to pull out Acc
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   if (air_data->options.any_c_smooths && !air_data->options.full_smoothing_up_and_down) {

      // Only reuse when coarse matrix structure is stable (amount>=2 stores MAT_RAP_DROP)
      if (air_data->allocated_matrices_A_cc[our_level] &&
          air_data->options.reuse_sparsity &&
          REUSE_MAT_ACTIVE[MAT_RAP_DROP][air_data->options.reuse_amount - 1]) {
         PetscCall(MatCreateSubMatrixWrapper(input_mat,
               air_data->IS_coarse_index[our_level], air_data->IS_coarse_index[our_level], MAT_REUSE_MATRIX,
               &air_data->A_cc[our_level],
               &our_level_kokkos, &false_bool, &false_bool,
               air_data->kokkos_is_views_handle));
      } else {
         PetscCall(MatCreateSubMatrixWrapper(input_mat,
               air_data->IS_coarse_index[our_level], air_data->IS_coarse_index[our_level], MAT_INITIAL_MATRIX,
               &air_data->A_cc[our_level],
               &our_level_kokkos, &false_bool, &false_bool,
               air_data->kokkos_is_views_handle));
      }

      PetscCall(timer_start(PFLARE_TIMER_ID_AIR_INVERSE));

      // Compute the inverse for smoothing
      if (!(air_data->inv_A_cc[our_level]
                  && air_data->options.reuse_poly_coeffs)) {
         PetscCall(poly_data_buffers(&air_data->inv_A_cc_poly_data[our_level], &buffers));
         PetscCall(start_approximate_inverse(air_data->A_cc[our_level],
               air_data->inv_A_cc_poly_data[our_level].inverse_type,
               air_data->inv_A_cc_poly_data[our_level].gmres_poly_order,
               air_data->options.diag_scale_polys,
               buffers, &air_data->inv_A_cc_poly_data[our_level].coefficients,
               &air_data->inv_A_cc_poly_data[our_level].coefficients_rows,
               &air_data->inv_A_cc_poly_data[our_level].coefficients_cols));
      }

      PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_INVERSE));

   }

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   // Pull out the rest of the sub-matrices
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PetscCall(timer_start(PFLARE_TIMER_ID_AIR_EXTRACT));

   // Only reuse when coarse matrix structure is stable (amount>=2 stores MAT_RAP_DROP)
   if (air_data->allocated_matrices_A_ff[our_level] &&
       air_data->options.reuse_sparsity &&
       REUSE_MAT_ACTIVE[MAT_RAP_DROP][air_data->options.reuse_amount - 1]) {
      PetscCall(MatCreateSubMatrixWrapper(input_mat,
            air_data->IS_fine_index[our_level], air_data->IS_coarse_index[our_level], MAT_REUSE_MATRIX,
            &air_data->A_fc[our_level],
            &our_level_kokkos, &true_bool, &false_bool,
            air_data->kokkos_is_views_handle));
      PetscCall(MatCreateSubMatrixWrapper(input_mat,
            air_data->IS_coarse_index[our_level], air_data->IS_fine_index[our_level], MAT_REUSE_MATRIX,
            &air_data->A_cf[our_level],
            &our_level_kokkos, &false_bool, &true_bool,
            air_data->kokkos_is_views_handle));
   } else {
      PetscCall(MatCreateSubMatrixWrapper(input_mat,
            air_data->IS_fine_index[our_level], air_data->IS_coarse_index[our_level], MAT_INITIAL_MATRIX,
            &air_data->A_fc[our_level],
            &our_level_kokkos, &true_bool, &false_bool,
            air_data->kokkos_is_views_handle));
      PetscCall(MatCreateSubMatrixWrapper(input_mat,
            air_data->IS_coarse_index[our_level], air_data->IS_fine_index[our_level], MAT_INITIAL_MATRIX,
            &air_data->A_cf[our_level],
            &our_level_kokkos, &false_bool, &true_bool,
            air_data->kokkos_is_views_handle));
   }

   PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_EXTRACT));

   // ~~~~~~~~~~~~~~
   // Apply the strong R threshold to A_cf
   // ~~~~~~~~~~~~~~
   if (air_data->options.strong_r_threshold == 0.0) {

      // Copy the original pointer
      air_data->reuse[our_level].reuse_mat[MAT_ACF_DROP] = air_data->A_cf[our_level];
      // Increase the reference counter
      PetscCall(PetscObjectReference((PetscObject)air_data->A_cf[our_level]));
      air_data->reuse[our_level].reuse_mat[MAT_AFC_DROP] = air_data->A_fc[our_level];
      // Increase the reference counter
      PetscCall(PetscObjectReference((PetscObject)air_data->A_fc[our_level]));

   } else {

      PetscCall(timer_start(PFLARE_TIMER_ID_AIR_EXTRACT));

      // Drop the entries from A_cf
      if (air_data->reuse[our_level].reuse_mat[MAT_ACF_DROP]) {
         PetscCall(MatCreateSubMatrixWrapper(air_data->reuse[our_level].reuse_mat[MAT_A_DROP],
                     air_data->IS_coarse_index[our_level], air_data->IS_fine_index[our_level], MAT_REUSE_MATRIX,
                     &air_data->reuse[our_level].reuse_mat[MAT_ACF_DROP],
                     &our_level_kokkos, &false_bool, &true_bool,
                     air_data->kokkos_is_views_handle));
      } else {
         PetscCall(MatCreateSubMatrixWrapper(air_data->reuse[our_level].reuse_mat[MAT_A_DROP],
                     air_data->IS_coarse_index[our_level], air_data->IS_fine_index[our_level], MAT_INITIAL_MATRIX,
                     &air_data->reuse[our_level].reuse_mat[MAT_ACF_DROP],
                     &our_level_kokkos, &false_bool, &true_bool,
                     air_data->kokkos_is_views_handle));
      }

      if (!air_data->options.one_point_classical_prolong) {
         // Drop the entries from A_fc
         if (air_data->reuse[our_level].reuse_mat[MAT_AFC_DROP]) {
            PetscCall(MatCreateSubMatrixWrapper(air_data->reuse[our_level].reuse_mat[MAT_A_DROP],
                     air_data->IS_fine_index[our_level], air_data->IS_coarse_index[our_level], MAT_REUSE_MATRIX,
                     &air_data->reuse[our_level].reuse_mat[MAT_AFC_DROP],
                     &our_level_kokkos, &true_bool, &false_bool,
                     air_data->kokkos_is_views_handle));
         } else {
            PetscCall(MatCreateSubMatrixWrapper(air_data->reuse[our_level].reuse_mat[MAT_A_DROP],
                     air_data->IS_fine_index[our_level], air_data->IS_coarse_index[our_level], MAT_INITIAL_MATRIX,
                     &air_data->reuse[our_level].reuse_mat[MAT_AFC_DROP],
                     &our_level_kokkos, &true_bool, &false_bool,
                     air_data->kokkos_is_views_handle));
         }
      }

      PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_EXTRACT));
   }

   // Delete temporary if not reusing
   if (!air_data->options.reuse_sparsity ||
       !REUSE_MAT_ACTIVE[MAT_A_DROP][air_data->options.reuse_amount - 1]) {
      PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_A_DROP]));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}


//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode finish_comms_compute_restrict_prolong(Mat A, int our_level, air_multigrid_data *air_data,
   Vec left_null_vecs[], PetscInt left_null_vecs_size, Vec right_null_vecs[], PetscInt right_null_vecs_size,
   Vec left_null_vecs_c[], Vec right_null_vecs_c[])
{
   // Finishes off the comms required to compute the A_ff inverse
   // then form Z, W and the full R and P

   // ~~~~~~~~~~
   Mat sparsity_mat_cf = NULL, A_ff_power, inv_dropped_Aff = NULL, smoothing_mat;
   Vec *left_null_vecs_f = NULL, *right_null_vecs_f = NULL;
   PetscMPIInt comm_size;
   PetscInt order, i_loc;
   MPI_Comm MPI_COMM_MATRIX;
   Mat C_mat;
   PetscInt global_row_start, global_row_end_plus_one;
   PetscBool destroy_mat, reuse_grid_transfer, is_kokkos;
   MatType mat_type, mat_type_inv_aff;
   tsqr_buffers *buffers;
   int one = 1;
   // ~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)air_data->A_ff[our_level], &MPI_COMM_MATRIX));
   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));

   // ~~~~~~~~~~~
   // Get some sizes
   // ~~~~~~~~~~~

   PetscCall(MatGetOwnershipRange(A, &global_row_start, &global_row_end_plus_one));

   // ~~~~~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~~~~~
   // Calculate one point W if needed
   // ~~~~~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~~~~~

   // If we're doing a one-point classical prolongator, we can do this before A_ff^-1 is finished
   if (air_data->options.one_point_classical_prolong && !air_data->options.symmetric) {

      // Classical one-point prolongator
      PetscCall(timer_start(PFLARE_TIMER_ID_AIR_PROLONG));

      // If we want to reuse, we have to match the original sparsity
      // which might be different if the matrix has changed (but the sparsity is the same)
      // so we can't just drop with a drop tolerance
      if (air_data->reuse[our_level].reuse_mat[MAT_W_DROP]) {

         // The one point classical prolongator will never change if
         // we are reusing

      // First time
      } else {
         PetscCall(generate_one_point_with_one_entry_from_sparse(air_data->A_fc[our_level],
                  &air_data->reuse[our_level].reuse_mat[MAT_W_DROP]));
      }

      PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_PROLONG));

   }

   // ~~~~~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~~~~~
   // Finish the calculation of the inverse for the smoother
   // and/or the inverse for the grid-transfers (they may be the same)
   // ~~~~~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~~~~~

   // This is for the smoother
   if (!air_data->options.full_smoothing_up_and_down) {
      smoothing_mat = air_data->A_ff[our_level];
   } else {
      smoothing_mat = A;
   }

   // Resolve the calculation for the smoother
   // This may build a matrix-free version of inv_A_ff
   PetscCall(timer_start(PFLARE_TIMER_ID_AIR_INVERSE));

   // Regardless of if we are fc smoothing or smoothing all unknowns we store the inverse
   // in inv_A_ff
   PetscCall(poly_data_buffers(&air_data->inv_A_ff_poly_data[our_level], &buffers));
   PetscCall(finish_approximate_inverse(smoothing_mat,
         air_data->inv_A_ff_poly_data[our_level].inverse_type,
         air_data->inv_A_ff_poly_data[our_level].gmres_poly_order,
         air_data->inv_A_ff_poly_data[our_level].gmres_poly_sparsity_order,
         buffers, air_data->inv_A_ff_poly_data[our_level].coefficients,
         air_data->inv_A_ff_poly_data[our_level].coefficients_rows,
         air_data->inv_A_ff_poly_data[our_level].coefficients_cols,
         air_data->options.matrix_free_polys,
         air_data->options.diag_scale_polys,
         &air_data->reuse[our_level].reuse_mat[MAT_INV_AFF],
         &air_data->reuse[our_level].reuse_submatrices[MAT_INV_AFF].array,
         &air_data->inv_A_ff[our_level]));

   // Delete temporary if not reusing
   if (!air_data->options.reuse_sparsity ||
       !REUSE_MAT_ACTIVE[MAT_INV_AFF][air_data->options.reuse_amount - 1]) {
      // Because inv_A_ff might have come from a gmres polynomial that had to use matcreatesubmatrices
      // have to be careful about how we destroy it
      PetscCall(destroy_matrix_reuse(&air_data->reuse[our_level].reuse_mat[MAT_INV_AFF],
               &air_data->reuse[our_level].reuse_submatrices[MAT_INV_AFF].array));
   }

   destroy_mat = PETSC_FALSE;
   // If we are doing AIRG, then we also need an A_ff^-1 for the grid
   // transfer operators (may be the same used for the smoother)
   if (air_data->options.z_type == PFLARE_AIR_Z_PRODUCT) {

      // If we have applied a strong R tolerance or we are not doing fc smoothing
      // then we have also started an inverse for dropped A_ff, let's finish it
      if (air_data->options.strong_r_threshold != 0.0 || air_data->options.full_smoothing_up_and_down) {

         // Now we always build an assembled inv_A_ff here as we need it, hence the .false.
         // given to matrix_free_polys
         // If we aren't doing matrix-free smooths, then we keep the assembled inv_A_ff
         PetscCall(poly_data_buffers(&air_data->inv_A_ff_poly_data_dropped[our_level], &buffers));
         PetscCall(finish_approximate_inverse(air_data->reuse[our_level].reuse_mat[MAT_AFF_DROP],
                  air_data->inv_A_ff_poly_data_dropped[our_level].inverse_type,
                  air_data->inv_A_ff_poly_data_dropped[our_level].gmres_poly_order,
                  air_data->inv_A_ff_poly_data_dropped[our_level].gmres_poly_sparsity_order,
                  buffers, air_data->inv_A_ff_poly_data_dropped[our_level].coefficients,
                  air_data->inv_A_ff_poly_data_dropped[our_level].coefficients_rows,
                  air_data->inv_A_ff_poly_data_dropped[our_level].coefficients_cols,
                  PETSC_FALSE,
                  air_data->options.diag_scale_polys,
                  &air_data->reuse[our_level].reuse_mat[MAT_INV_AFF_DROPPED],
                  &air_data->reuse[our_level].reuse_submatrices[MAT_INV_AFF_DROPPED].array,
                  &inv_dropped_Aff));
         destroy_mat = PETSC_TRUE;

         // Delete temporary if not reusing
         if (!air_data->options.reuse_sparsity ||
             !REUSE_MAT_ACTIVE[MAT_INV_AFF_DROPPED][air_data->options.reuse_amount - 1]) {
            PetscCall(destroy_matrix_reuse(&air_data->reuse[our_level].reuse_mat[MAT_INV_AFF_DROPPED],
                     &air_data->reuse[our_level].reuse_submatrices[MAT_INV_AFF_DROPPED].array));
         }

      // If we have a strong R tolerance of 0, we can re-use the
      // A_ff^-1 we computed for the smoother
      } else {
         // If we requested matrix-free smoothing, then we will have called the finish
         // and computed the coefficients but inv_A_ff will be a matshell
         // and hence we want to build an assembled version of A_ff^-1 to use here
         // We can just re-call finish_approximate_inverse, as the requests will have been resolved
         // and hence it will just build an assembled version, which we store in inv_dropped_Aff
         if (air_data->options.matrix_free_polys) {
            // Making sure to give it the non dropped A_ff here
            PetscCall(poly_data_buffers(&air_data->inv_A_ff_poly_data[our_level], &buffers));
            PetscCall(finish_approximate_inverse(air_data->A_ff[our_level],
                     air_data->inv_A_ff_poly_data[our_level].inverse_type,
                     air_data->inv_A_ff_poly_data[our_level].gmres_poly_order,
                     air_data->inv_A_ff_poly_data[our_level].gmres_poly_sparsity_order,
                     buffers, air_data->inv_A_ff_poly_data[our_level].coefficients,
                     air_data->inv_A_ff_poly_data[our_level].coefficients_rows,
                     air_data->inv_A_ff_poly_data[our_level].coefficients_cols,
                     PETSC_FALSE,
                     air_data->options.diag_scale_polys,
                     &air_data->reuse[our_level].reuse_mat[MAT_INV_AFF_DROPPED],
                     &air_data->reuse[our_level].reuse_submatrices[MAT_INV_AFF_DROPPED].array,
                     &inv_dropped_Aff));
            destroy_mat = PETSC_TRUE;

            // Delete temporary if not reusing
            if (!air_data->options.reuse_sparsity ||
                !REUSE_MAT_ACTIVE[MAT_INV_AFF_DROPPED][air_data->options.reuse_amount - 1]) {
               PetscCall(destroy_matrix_reuse(&air_data->reuse[our_level].reuse_mat[MAT_INV_AFF_DROPPED],
                        &air_data->reuse[our_level].reuse_submatrices[MAT_INV_AFF_DROPPED].array));
            }

         // Just re-use the already assembled one
         } else {
            inv_dropped_Aff = air_data->inv_A_ff[our_level];
         }
      }

      PetscCall(MatGetType(inv_dropped_Aff, &mat_type_inv_aff));
   }

   PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_INVERSE));

   // Need a routine to calculate W with lAIR
   PetscCheck(!(air_data->options.z_type != PFLARE_AIR_Z_PRODUCT &&
            (!air_data->options.one_point_classical_prolong && !air_data->options.symmetric)),
            MPI_COMM_MATRIX, PETSC_ERR_SUP, "Fix me - calculation of W with lair");

   // ~~~~~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~~~~~
   // Pull out the constraints on F and C points
   // ~~~~~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~~~~~
   if (air_data->options.constrain_w) {
      if (!right_null_vecs_f) PetscCall(PetscCalloc1(right_null_vecs_size, &right_null_vecs_f));
      // Create space for the C and F constraints
      PetscCall(MatCreateVecs(air_data->A_fc[our_level], &right_null_vecs_c[0], &right_null_vecs_f[0]));
      for (i_loc = 1; i_loc < right_null_vecs_size; i_loc++) {
         PetscCall(VecDuplicate(right_null_vecs_c[0], &right_null_vecs_c[i_loc]));
         PetscCall(VecDuplicate(right_null_vecs_f[0], &right_null_vecs_f[i_loc]));
      }
      // Pull out the F and C points
      for (i_loc = 0; i_loc < right_null_vecs_size; i_loc++) {
         PetscCall(VecISCopy(right_null_vecs[i_loc], air_data->IS_fine_index[our_level],
                  SCATTER_REVERSE, right_null_vecs_f[i_loc]));
         PetscCall(VecISCopy(right_null_vecs[i_loc], air_data->IS_coarse_index[our_level],
                  SCATTER_REVERSE, right_null_vecs_c[i_loc]));
      }
   }
   if (air_data->options.constrain_z) {
      if (!left_null_vecs_f) PetscCall(PetscCalloc1(left_null_vecs_size, &left_null_vecs_f));
      // Create space for the C and F constraints
      PetscCall(MatCreateVecs(air_data->A_fc[our_level], &left_null_vecs_c[0], &left_null_vecs_f[0]));
      for (i_loc = 1; i_loc < left_null_vecs_size; i_loc++) {
         PetscCall(VecDuplicate(left_null_vecs_c[0], &left_null_vecs_c[i_loc]));
         PetscCall(VecDuplicate(left_null_vecs_f[0], &left_null_vecs_f[i_loc]));
      }
      // Pull out the F and C points
      for (i_loc = 0; i_loc < left_null_vecs_size; i_loc++) {
         PetscCall(VecISCopy(left_null_vecs[i_loc], air_data->IS_fine_index[our_level],
                  SCATTER_REVERSE, left_null_vecs_f[i_loc]));
         PetscCall(VecISCopy(left_null_vecs[i_loc], air_data->IS_coarse_index[our_level],
                  SCATTER_REVERSE, left_null_vecs_c[i_loc]));
      }
   }

   // ~~~~~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~~~~~
   // Calculate W if needed
   // ~~~~~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~~~~~
   // Grid transfer (R, P) reuse requires A_ff/A_fc/A_cf to be preserved
   // across setups (only amount=3).  At amount=2, P and R are rebuilt fresh
   // from stored W_DROP/Z_DROP; SpGEMM reuse still works via stored AP/RAP.
   reuse_grid_transfer = (air_data->allocated_matrices_A_ff[our_level] &&
         air_data->options.reuse_sparsity) ? PETSC_TRUE : PETSC_FALSE;

   // We do this a little backwards if symmetric, we build R and then compute P^T, then delete R
   // It's just because we have code to do different version of Z, and I haven't rewritten those
   // for W
   if (!air_data->options.symmetric) {

      // Calculate the W component of the prolongator
      PetscCall(timer_start(PFLARE_TIMER_ID_AIR_PROLONG));

      // If we want an ideal prolongator
      if (!air_data->options.one_point_classical_prolong) {

         // Do the multiplication
         if (air_data->reuse[our_level].reuse_mat[MAT_W]) {
            PetscCall(MatMatMult(inv_dropped_Aff, air_data->reuse[our_level].reuse_mat[MAT_AFC_DROP],
                     MAT_REUSE_MATRIX, PFLARE_PTAP_FILL, &air_data->reuse[our_level].reuse_mat[MAT_W]));
         } else {
            PetscCall(MatMatMult(inv_dropped_Aff, air_data->reuse[our_level].reuse_mat[MAT_AFC_DROP],
                     MAT_INITIAL_MATRIX, PFLARE_PTAP_FILL, &air_data->reuse[our_level].reuse_mat[MAT_W]));
         if (!air_data->options.reuse_sparsity ||
             !REUSE_MAT_ACTIVE[MAT_W][air_data->options.reuse_amount - 1])
            PetscCall(MatProductClear(air_data->reuse[our_level].reuse_mat[MAT_W]));
         }
         PetscCall(MatScale(air_data->reuse[our_level].reuse_mat[MAT_W], PFLARE_MINUS_ONE));

         // ~~~~~~~~~
         // Improve W if needed
         // ~~~~~~~~~

         PetscCall(improve_w(air_data->reuse[our_level].reuse_mat[MAT_W],
                        air_data->A_ff[our_level],
                        air_data->A_fc[our_level],
                        air_data->inv_A_ff[our_level],
                        &air_data->reuse[our_level].reuse_mat[MAT_W_AFF],
                        &air_data->reuse[our_level].reuse_mat[MAT_W_NO_SPARSITY],
                        (int)air_data->options.improve_w_its,
                        (air_data->options.reuse_sparsity &&
                           REUSE_MAT_ACTIVE[MAT_W_AFF][air_data->options.reuse_amount - 1]) ? PETSC_TRUE : PETSC_FALSE));

         // Delete temporaries if not reusing
         if (!air_data->options.reuse_sparsity ||
             !REUSE_MAT_ACTIVE[MAT_W_AFF][air_data->options.reuse_amount - 1]) {
            PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_W_AFF]));
         }
         if (!air_data->options.reuse_sparsity ||
             !REUSE_MAT_ACTIVE[MAT_W_NO_SPARSITY][air_data->options.reuse_amount - 1]) {
            PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_W_NO_SPARSITY]));
         }

         // ~~~~~~~~~~~~
         // ~~~~~~~~~~~~

         PetscCall(timer_start(PFLARE_TIMER_ID_AIR_DROP));

         // If we want to reuse, we have to match the original sparsity
         // which might be different if the matrix has changed (but the sparsity is the same)
         // so we can't just drop with a drop tolerance
         if (air_data->reuse[our_level].reuse_mat[MAT_W_DROP]) {

            PetscCall(remove_from_sparse_match_c(air_data->reuse[our_level].reuse_mat[MAT_W],
                     air_data->reuse[our_level].reuse_mat[MAT_W_DROP], NULL, NULL));

         // First time so just drop according to a tolerance
         } else {
            PetscCall(remove_small_from_sparse(air_data->reuse[our_level].reuse_mat[MAT_W],
                           air_data->options.r_drop,
                           &air_data->reuse[our_level].reuse_mat[MAT_W_DROP],
                           &one, NULL, NULL, NULL));
         }

         // Delete temporary if not reusing
         if (!air_data->options.reuse_sparsity ||
             !REUSE_MAT_ACTIVE[MAT_W][air_data->options.reuse_amount - 1]) {
            PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_W]));
         }

         PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_DROP));
      }

      // ~~~~~~~~~
      // Apply constraints to W if needed
      // ~~~~~~~~~
      if (air_data->options.constrain_w) {
         PetscCall(timer_start(PFLARE_TIMER_ID_AIR_CONSTRAIN));
         PetscCall(constrain_grid_transfer(&air_data->reuse[our_level].reuse_mat[MAT_W_DROP], PETSC_FALSE,
                  right_null_vecs_f, right_null_vecs_c, right_null_vecs_size));
         PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_CONSTRAIN));
      }

      // Now we have W
      // Build a copy of P with the identity block in it
      // This is to save having to do Z A_ff W + A_cf W + Z A_fc + A_cc

      // ~~~~~~~~~~~~~~~~~~~
      // ~~~~~~~~~~~~~~~~~~~
      // Calculate the prolongator
      // ~~~~~~~~~~~~~~~~~~~
      // ~~~~~~~~~~~~~~~~~~~
      // One-point classical prolongator reuse requires A_ff preserved (amount=3 only).
      // At amount=2, P is rebuilt from stored W_DROP which has stable sparsity.
      air_data->reuse_one_point_classical_prolong = (air_data->options.one_point_classical_prolong &&
            !air_data->options.symmetric &&
            !air_data->options.constrain_w &&
            air_data->allocated_matrices_A_ff[our_level] &&
            air_data->options.reuse_sparsity) ? PETSC_TRUE : PETSC_FALSE;

      // If we are doing reuse and processor agglomeration on this level,
      // then we can't reuse the sparsity of R or P as it gets repartitioned during the setup
      if (air_data->allocated_matrices_A_ff[our_level] &&
               air_data->reuse[our_level].reuse_is[IS_REPARTITION]) {

         // Now when we're doing processor agglomeration, we have to be careful with
         // Kokkos, as it gets fussy
         // about the exact same pointers being passed into spgemm_numeric
         // (rather than just having the same sparsity) once we've
         // repartitioned. We have to force it to repartition to get around this
         // so the exact same matrices are used in every case
         PetscCall(MatGetType(air_data->coarse_matrix[our_level], &mat_type));
         PetscCall(PetscStrcmp(mat_type, MATMPIAIJKOKKOS, &is_kokkos));
         if (is_kokkos) {
            air_data->reuse_one_point_classical_prolong = PETSC_FALSE;
         }

         // Destroy the grid transfer operators and rebuild them
         PetscCall(MatDestroy(&air_data->restrictors[our_level]));
         if (!air_data->reuse_one_point_classical_prolong) {
            PetscCall(MatDestroy(&air_data->prolongators[our_level]));
         }
         reuse_grid_transfer = PETSC_FALSE;
      }

      // If we've got a one point classical prolongator computed already we can just reuse the prolongator
      // without change
      if (!air_data->reuse_one_point_classical_prolong) {
         PetscCall(compute_P_from_W(air_data->reuse[our_level].reuse_mat[MAT_W_DROP],
                  global_row_start,
                  air_data->IS_fine_index[our_level], air_data->IS_coarse_index[our_level],
                  PETSC_TRUE,
                  reuse_grid_transfer,
                  &air_data->prolongators[our_level]));
      }

      // Delete temporary if not reusing
      if (!air_data->options.reuse_sparsity ||
          !REUSE_MAT_ACTIVE[MAT_W_DROP][air_data->options.reuse_amount - 1]) {
         PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_W_DROP]));
      }

      PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_PROLONG));

   }

   // Destroy the F point constraints on W
   // If symmetric we never build W (P = R^T) so constrain_w is ignored and
   // these are unused, but they still have to be destroyed
   if (air_data->options.constrain_w) {
      for (i_loc = 0; i_loc < right_null_vecs_size; i_loc++) {
         PetscCall(VecDestroy(&right_null_vecs_f[i_loc]));
      }
      PetscCall(PetscFree(right_null_vecs_f));
   }

   // ~~~~~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~~~~~
   // Calculate Z
   // ~~~~~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~~~~~

   // Calculate the Z component of the restrictor
   PetscCall(timer_start(PFLARE_TIMER_ID_AIR_RESTRICT));

   // For lAIR
   if (air_data->options.z_type != PFLARE_AIR_Z_PRODUCT) {

      // Compute the strongly connected F neighbourhood that each C point
      // is going to use in lAIR
      // We use the dropped A_ff, A_cf here as they have had the strong R tolerance applied
      // to them
      if (air_data->options.lair_distance == 1) {

         sparsity_mat_cf = air_data->reuse[our_level].reuse_mat[MAT_ACF_DROP];

      // Distance 2 is sparsity_mat = A_cf * A_ff
      // Distance 3 is sparsity_mat = A_cf * A_ff^2
      // etc
      } else {

         // If we are doing reuse, we already know the sparsity we want
         if (air_data->reuse[our_level].reuse_mat[MAT_Z]) {

            // We should just be able to use the pointer to air_data%reuse(our_level)%reuse_mat(MAT_Z)
            // as sparsity_mat_cf, but it gives me errors about unassembled matrices
            // I think that has to do with the old interface for MatMPIAIJGetSeqAIJ that we are using
            // in calculate_and_build_sai_z (and the fact it doesn't have a restore)
            // If we enforce a minimum of petsc 3.19 we could use the new MatMPIAIJGetSeqAIJF90
            // For now we just take an extra copy
            PetscCall(MatDuplicate(air_data->reuse[our_level].reuse_mat[MAT_Z],
                     MAT_DO_NOT_COPY_VALUES, &sparsity_mat_cf));
            //sparsity_mat_cf = air_data%reuse(our_level)%reuse_mat(MAT_Z)

         } else {

            // Copy the pointer
            A_ff_power = air_data->reuse[our_level].reuse_mat[MAT_AFF_DROP];
            destroy_mat = PETSC_FALSE;

            // Compute A_ff^(distance - 1)
            for (order = 3; order <= air_data->options.lair_distance; order++) {

               // Call a symbolic mult as we don't need the values, just the resulting sparsity
               PetscCall(mat_mat_symbolic(air_data->reuse[our_level].reuse_mat[MAT_AFF_DROP], A_ff_power, &C_mat));
               // Don't delete the original power - ie A_ff
               if (destroy_mat) PetscCall(MatDestroy(&A_ff_power));
               A_ff_power = C_mat;
               destroy_mat = PETSC_TRUE;

            }

            // Call a symbolic mult as we don't need the values, just the resulting sparsity
            // A_cf * A_ff^(distance - 1)
            PetscCall(mat_mat_symbolic(air_data->reuse[our_level].reuse_mat[MAT_ACF_DROP], A_ff_power, &C_mat));
            if (destroy_mat) PetscCall(MatDestroy(&A_ff_power));

            sparsity_mat_cf = C_mat;

         }
      }

      // lAIR
      // We delibrately have to give it A_ff and A_cf as input, not the dropped versions
      // The sparsity is controlled by sparsity_mat_cf which has used the dropped versions
      if (air_data->options.z_type == PFLARE_AIR_Z_LAIR) {
         PetscCall(calculate_and_build_sai_z(air_data->A_ff[our_level], air_data->A_cf[our_level],
                     sparsity_mat_cf, PETSC_TRUE,
                     &air_data->reuse[our_level].reuse_mat[MAT_SAI_SUB],
                     &air_data->reuse[our_level].reuse_submatrices[MAT_SAI_SUB].array,
                     &air_data->reuse[our_level].reuse_mat[MAT_Z]));
      // SAI Z
      } else {
         PetscCall(calculate_and_build_sai_z(air_data->A_ff[our_level], air_data->A_cf[our_level],
                     sparsity_mat_cf, PETSC_FALSE,
                     &air_data->reuse[our_level].reuse_mat[MAT_SAI_SUB],
                     &air_data->reuse[our_level].reuse_submatrices[MAT_SAI_SUB].array,
                     &air_data->reuse[our_level].reuse_mat[MAT_Z]));
      }
      // Delete temporary if not reusing
      if (!air_data->options.reuse_sparsity ||
          !REUSE_MAT_ACTIVE[MAT_SAI_SUB][air_data->options.reuse_amount - 1]) {
         PetscCall(destroy_matrix_reuse(&air_data->reuse[our_level].reuse_mat[MAT_SAI_SUB],
                  &air_data->reuse[our_level].reuse_submatrices[MAT_SAI_SUB].array));
      }
      if (air_data->options.lair_distance >= 2) {
         PetscCall(MatDestroy(&sparsity_mat_cf));
      }

   // For AIRG - we do a matmatmult with our approximate A_ff inverse
   } else {
      if (air_data->reuse[our_level].reuse_mat[MAT_Z]) {
         PetscCall(MatMatMult(air_data->reuse[our_level].reuse_mat[MAT_ACF_DROP], inv_dropped_Aff,
               MAT_REUSE_MATRIX, PFLARE_PTAP_FILL, &air_data->reuse[our_level].reuse_mat[MAT_Z]));
      } else {
         PetscCall(MatMatMult(air_data->reuse[our_level].reuse_mat[MAT_ACF_DROP], inv_dropped_Aff,
               MAT_INITIAL_MATRIX, PFLARE_PTAP_FILL, &air_data->reuse[our_level].reuse_mat[MAT_Z]));
         if (!air_data->options.reuse_sparsity ||
             !REUSE_MAT_ACTIVE[MAT_Z][air_data->options.reuse_amount - 1])
            PetscCall(MatProductClear(air_data->reuse[our_level].reuse_mat[MAT_Z]));
      }
      PetscCall(MatScale(air_data->reuse[our_level].reuse_mat[MAT_Z], PFLARE_MINUS_ONE));
   }

   // Destroy the copies, this only decrements the reference counter
   if (air_data->options.strong_r_threshold == 0.0) {
      PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_AFF_DROP]));
      PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_ACF_DROP]));
      PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_AFC_DROP]));
   }

   // Delete temporaries if not reusing
   if (!air_data->options.reuse_sparsity ||
       !REUSE_MAT_ACTIVE[MAT_AFF_DROP][air_data->options.reuse_amount - 1]) {
      PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_AFF_DROP]));
   }
   if (!air_data->options.reuse_sparsity ||
       !REUSE_MAT_ACTIVE[MAT_ACF_DROP][air_data->options.reuse_amount - 1]) {
      PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_ACF_DROP]));
   }
   if (!air_data->options.reuse_sparsity ||
       !REUSE_MAT_ACTIVE[MAT_AFC_DROP][air_data->options.reuse_amount - 1]) {
      PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_AFC_DROP]));
   }

   // ~~~~~~~~~
   // Improve Z if needed
   // ~~~~~~~~~

   PetscCall(improve_z(air_data->reuse[our_level].reuse_mat[MAT_Z],
                  air_data->A_ff[our_level],
                  air_data->A_cf[our_level],
                  air_data->inv_A_ff[our_level],
                  &air_data->reuse[our_level].reuse_mat[MAT_Z_AFF],
                  &air_data->reuse[our_level].reuse_mat[MAT_Z_NO_SPARSITY],
                  (int)air_data->options.improve_z_its,
                  (air_data->options.reuse_sparsity &&
                     REUSE_MAT_ACTIVE[MAT_Z_AFF][air_data->options.reuse_amount - 1]) ? PETSC_TRUE : PETSC_FALSE));

   // Delete temporaries if not reusing
   if (!air_data->options.reuse_sparsity ||
       !REUSE_MAT_ACTIVE[MAT_Z_AFF][air_data->options.reuse_amount - 1]) {
      PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_Z_AFF]));
   }
   if (!air_data->options.reuse_sparsity ||
       !REUSE_MAT_ACTIVE[MAT_Z_NO_SPARSITY][air_data->options.reuse_amount - 1]) {
      PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_Z_NO_SPARSITY]));
   }

   // Delete temporary if not reusing
   if (!air_data->options.any_c_smooths && !air_data->options.reuse_sparsity) {
      PetscCall(MatDestroy(&air_data->A_cf[our_level]));
   }

   // ~~~~~~~~~~~~
   // ~~~~~~~~~~~~

   PetscCall(timer_start(PFLARE_TIMER_ID_AIR_DROP));

   // If we want to reuse, we have to match the original sparsity
   // which might be different if the matrix has changed (but the sparsity is the same)
   // so we can't just drop with a drop tolerance
   if (air_data->reuse[our_level].reuse_mat[MAT_Z_DROP]) {

      PetscCall(remove_from_sparse_match_c(air_data->reuse[our_level].reuse_mat[MAT_Z],
               air_data->reuse[our_level].reuse_mat[MAT_Z_DROP], NULL, NULL));

   // First time so just drop according to a tolerance
   } else {
      PetscCall(remove_small_from_sparse(air_data->reuse[our_level].reuse_mat[MAT_Z],
                  air_data->options.r_drop, &air_data->reuse[our_level].reuse_mat[MAT_Z_DROP],
                  &one, NULL, NULL, NULL));
   }

   PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_DROP));
   // Delete temporary if not reusing
   if (!air_data->options.reuse_sparsity ||
       !REUSE_MAT_ACTIVE[MAT_Z][air_data->options.reuse_amount - 1]) {
      PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_Z]));
   }
   if (air_data->options.z_type == PFLARE_AIR_Z_PRODUCT && destroy_mat) {
      PetscCall(MatDestroy(&inv_dropped_Aff));
   }

   // ~~~~~~~~~
   // Apply constraints to Z if needed
   // ~~~~~~~~~
   if (air_data->options.constrain_z) {

      PetscCall(timer_start(PFLARE_TIMER_ID_AIR_CONSTRAIN));
      PetscCall(constrain_grid_transfer(&air_data->reuse[our_level].reuse_mat[MAT_Z_DROP], PETSC_TRUE,
                  left_null_vecs_f, left_null_vecs_c, left_null_vecs_size));
      PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_CONSTRAIN));

      for (i_loc = 0; i_loc < left_null_vecs_size; i_loc++) {
         PetscCall(VecDestroy(&left_null_vecs_f[i_loc]));
      }
      PetscCall(PetscFree(left_null_vecs_f));
   }

   // ~~~~~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~~~~~
   // Calculate R
   // ~~~~~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~~~~~
   PetscCall(compute_R_from_Z(air_data->reuse[our_level].reuse_mat[MAT_Z_DROP], global_row_start,
            air_data->IS_fine_index[our_level], air_data->IS_coarse_index[our_level],
            &air_data->reuse[our_level].reuse_is[IS_R_Z_FINE_COLS],
            PETSC_TRUE,
            reuse_grid_transfer,
            &air_data->restrictors[our_level]));

   PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_RESTRICT));

   // Delete temporaries if not reusing
   if (!air_data->options.reuse_sparsity ||
       !REUSE_MAT_ACTIVE[MAT_Z_DROP][air_data->options.reuse_amount - 1]) {
      PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_Z_DROP]));
   }
   if (!air_data->options.reuse_sparsity ||
       !REUSE_IS_ACTIVE[IS_R_Z_FINE_COLS][air_data->options.reuse_amount - 1]) {
      PetscCall(ISDestroy(&air_data->reuse[our_level].reuse_is[IS_R_Z_FINE_COLS]));
   }

   // Transpose the restrictor if needed
   if (air_data->options.symmetric) {

      // P can only be updated in place if it was created as the transpose of
      // this R in a previous setup and hasn't been repartitioned since
      if (reuse_grid_transfer && air_data->prolongators[our_level] &&
               !air_data->reuse[our_level].reuse_is[IS_REPARTITION]) {
         PetscCall(MatTranspose(air_data->restrictors[our_level], MAT_REUSE_MATRIX, &air_data->prolongators[our_level]));
      } else {
         PetscCall(MatDestroy(&air_data->prolongators[our_level]));
         PetscCall(MatTranspose(air_data->restrictors[our_level], MAT_INITIAL_MATRIX, &air_data->prolongators[our_level]));
         // Any stored PtAP data refers to the old P, so it can't be reused
         if (air_data->reuse[our_level].reuse_mat[MAT_RAP]) {
            PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_RAP]));
         }
      }

      // Keep R if we're reusing the grid transfer operators (only amount=3)
      // as the reuse in compute_R_from_Z and the transpose above need it
      if (!air_data->options.reuse_sparsity || air_data->options.reuse_amount < 3) {
         PetscCall(MatDestroy(&air_data->restrictors[our_level]));
      }
   }

   // ~~~~~~~~~~~
   // If we are doing C-point smoothing, finish the comms
   // ~~~~~~~~~~~
   if (air_data->options.any_c_smooths &&
            !air_data->options.full_smoothing_up_and_down) {

      PetscCall(timer_start(PFLARE_TIMER_ID_AIR_INVERSE));

      PetscCall(poly_data_buffers(&air_data->inv_A_cc_poly_data[our_level], &buffers));
      PetscCall(finish_approximate_inverse(air_data->A_cc[our_level],
               air_data->inv_A_cc_poly_data[our_level].inverse_type,
               air_data->inv_A_cc_poly_data[our_level].gmres_poly_order,
               air_data->inv_A_cc_poly_data[our_level].gmres_poly_sparsity_order,
               buffers, air_data->inv_A_cc_poly_data[our_level].coefficients,
               air_data->inv_A_cc_poly_data[our_level].coefficients_rows,
               air_data->inv_A_cc_poly_data[our_level].coefficients_cols,
               air_data->options.matrix_free_polys,
               air_data->options.diag_scale_polys,
               &air_data->reuse[our_level].reuse_mat[MAT_INV_ACC],
               &air_data->reuse[our_level].reuse_submatrices[MAT_INV_ACC].array,
               &air_data->inv_A_cc[our_level]));

      PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_INVERSE));

      // Delete temporary if not reusing
      if (!air_data->options.reuse_sparsity ||
          !REUSE_MAT_ACTIVE[MAT_INV_ACC][air_data->options.reuse_amount - 1]) {
         PetscCall(destroy_matrix_reuse(&air_data->reuse[our_level].reuse_mat[MAT_INV_ACC],
                  &air_data->reuse[our_level].reuse_submatrices[MAT_INV_ACC].array));
      }

   }

   PetscFunctionReturn(PETSC_SUCCESS);
}


//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode compute_coarse_matrix(Mat A, int our_level, air_multigrid_data *air_data,
   Mat *coarse_matrix)
{
   // Computes the coarse grid matrix with dropping

   // ~~~~~~~~~~
   int one = 1;
   // ~~~~~~~~~~

   PetscFunctionBegin;

   // ~~~~~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~~~~~
   // Now build our coarse grid matrix
   // Can be computed from either RAP, or the smaller
   // Z A_ff W + A_cf W + Z A_fc + A_cc
   // ~~~~~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~~~~~
   PetscCall(timer_start(PFLARE_TIMER_ID_AIR_RAP));

   // Can just do PtAP
   if (air_data->options.symmetric) {

      // If we've done this before we can reuse
      if (air_data->reuse[our_level].reuse_mat[MAT_RAP]) {

         PetscCall(MatPtAP(A, air_data->prolongators[our_level],
                  MAT_REUSE_MATRIX, PFLARE_PTAP_FILL, &air_data->reuse[our_level].reuse_mat[MAT_RAP]));

      // First time
      } else {

         PetscCall(MatPtAP(A, air_data->prolongators[our_level],
                  MAT_INITIAL_MATRIX, PFLARE_PTAP_FILL, &air_data->reuse[our_level].reuse_mat[MAT_RAP]));
      }

   // ~~~~~~~~~~~
   // Do two matmatmults rather than the triple product
   // ~~~~~~~~~~~
   } else {

      // If we've done this before we can reuse
      if (air_data->reuse[our_level].reuse_mat[MAT_AP]) {

         PetscCall(MatMatMult(A, air_data->prolongators[our_level],
                  MAT_REUSE_MATRIX, PFLARE_PTAP_FILL, &air_data->reuse[our_level].reuse_mat[MAT_AP]));

         PetscCall(MatMatMult(air_data->restrictors[our_level], air_data->reuse[our_level].reuse_mat[MAT_AP],
                  MAT_REUSE_MATRIX, PFLARE_PTAP_FILL, &air_data->reuse[our_level].reuse_mat[MAT_RAP]));

      // First time
      } else {

         PetscCall(MatMatMult(A, air_data->prolongators[our_level],
                  MAT_INITIAL_MATRIX, PFLARE_PTAP_FILL, &air_data->reuse[our_level].reuse_mat[MAT_AP]));
         if (!air_data->options.reuse_sparsity ||
             !REUSE_MAT_ACTIVE[MAT_AP][air_data->options.reuse_amount - 1])
            PetscCall(MatProductClear(air_data->reuse[our_level].reuse_mat[MAT_AP]));

         PetscCall(MatMatMult(air_data->restrictors[our_level], air_data->reuse[our_level].reuse_mat[MAT_AP],
                  MAT_INITIAL_MATRIX, PFLARE_PTAP_FILL, &air_data->reuse[our_level].reuse_mat[MAT_RAP]));
         if (!air_data->options.reuse_sparsity ||
             !REUSE_MAT_ACTIVE[MAT_RAP][air_data->options.reuse_amount - 1])
            PetscCall(MatProductClear(air_data->reuse[our_level].reuse_mat[MAT_RAP]));
      }

      // Delete temporary if not reusing
      if (!air_data->options.reuse_sparsity ||
          !REUSE_MAT_ACTIVE[MAT_AP][air_data->options.reuse_amount - 1]) {
         PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_AP]));
      }
   }

   PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_RAP));

   // Drop relative small entries
   PetscCall(timer_start(PFLARE_TIMER_ID_AIR_DROP));

   // If we want to reuse, we have to match the original sparsity
   // which might be different if the matrix has changed (but the sparsity is the same)
   // so we can't just drop with a drop tolerance
   if (air_data->reuse[our_level].reuse_mat[MAT_RAP_DROP]) {

      // Duplicate the sparsity
      PetscCall(MatDuplicate(air_data->reuse[our_level].reuse_mat[MAT_RAP_DROP],
               MAT_DO_NOT_COPY_VALUES, coarse_matrix));

      PetscCall(remove_from_sparse_match_c(air_data->reuse[our_level].reuse_mat[MAT_RAP],
               *coarse_matrix,
               &air_data->options.a_lump, NULL));

   // First time so just drop according to a tolerance
   } else {
      // If we know we're not reusing MAT_RAP_DROP, compute directly into coarse_matrix
      if (!air_data->options.reuse_sparsity ||
          !REUSE_MAT_ACTIVE[MAT_RAP_DROP][air_data->options.reuse_amount - 1]) {
         PetscCall(remove_small_from_sparse(air_data->reuse[our_level].reuse_mat[MAT_RAP],
                  air_data->options.a_drop, coarse_matrix,
                  &one, &air_data->options.a_lump, NULL, NULL));
      } else {
         PetscCall(remove_small_from_sparse(air_data->reuse[our_level].reuse_mat[MAT_RAP],
                  air_data->options.a_drop, &air_data->reuse[our_level].reuse_mat[MAT_RAP_DROP],
                  &one, &air_data->options.a_lump, NULL, NULL));

         PetscCall(MatDuplicate(air_data->reuse[our_level].reuse_mat[MAT_RAP_DROP],
                     MAT_COPY_VALUES, coarse_matrix));
      }
   }

   // Delete temporaries if not reusing
   if (!air_data->options.reuse_sparsity ||
       !REUSE_MAT_ACTIVE[MAT_RAP][air_data->options.reuse_amount - 1]) {
      PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_RAP]));
   }
   if (!air_data->options.reuse_sparsity ||
       !REUSE_MAT_ACTIVE[MAT_RAP_DROP][air_data->options.reuse_amount - 1]) {
      PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_RAP_DROP]));
   }

   PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_DROP));

   PetscFunctionReturn(PETSC_SUCCESS);
}
