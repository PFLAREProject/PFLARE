/*
  C port of CF_Splitting.F90 (module cf_splitting)

  The optional arguments of the Fortran are pointers that are NULL when the
  argument is absent, so present(x) is x != NULL. The CF markers are int
  arrays (the Fortran integer) with one entry per local row of input_mat;
  cf_markers_local is an int ** where it is allocated (PetscMalloc1, freed with
  PetscFree), and NULL is the Fortran .not. allocated. The loops are 0-based
  The CF splitting types are the CFSplittingType values in include/pflare.h,
  which the Fortran module defined as parameters of the same names
  compute_cf_splitting is called compute_cf_splitting_c here, as the public C
  API in include/pflare.h already has a routine of that name (in PCAIR.c,
  which calls this one)
*/

#include <petscmat.h>
#include <limits.h>
#include "pflare.h"
#include "pflareimpl.h"
#include "PMISR_Module.h"
#include "DDC_Module.h"
#include "CR_Splitting.h"
#include "SAbs.h"
#include "Device_Datak.h"
#include "Aggregation.h"
#include "PETSc_Helper.h"
#include "CF_Splitting.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode create_cf_is(Mat input_mat, const int *cf_markers_local, IS *is_fine, IS *is_coarse)
{
   // Creates IS's for C and F points given a cf_markers_local array

   // ~~~~~~~~~~
   PetscInt *fine_indices, *coarse_indices;
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt fine_counter, coarse_counter, i_loc;
   PetscInt n, n_fine, n_coarse;
   MPI_Comm MPI_COMM_MATRIX;

   // ~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)input_mat, &MPI_COMM_MATRIX));
   PetscCall(MatGetOwnershipRange(input_mat, &global_row_start, &global_row_end_plus_one));

   // This is the local_rows
   // (the size of cf_markers_local, which the Fortran array carries)
   n = global_row_end_plus_one - global_row_start;
   n_fine = 0;
   n_coarse = 0;
   for (i_loc = 0; i_loc < n; i_loc++) {
      if (cf_markers_local[i_loc] == PFLARE_F_POINT) n_fine = n_fine + 1;
      if (cf_markers_local[i_loc] == PFLARE_C_POINT) n_coarse = n_coarse + 1;
   }

   PetscCall(PetscMalloc1(n_fine, &fine_indices));
   PetscCall(PetscMalloc1(n_coarse, &coarse_indices));

   fine_counter = 0;
   coarse_counter = 0;
   for (i_loc = 0; i_loc < n; i_loc++) {
      if (cf_markers_local[i_loc] == PFLARE_F_POINT) {
         fine_counter = fine_counter + 1;
         // Remember petsc indices start at zero
         fine_indices[fine_counter - 1] = global_row_start + i_loc;
      } else {
         coarse_counter = coarse_counter + 1;
         // Remember petsc indices start at zero
         coarse_indices[coarse_counter - 1] = global_row_start + i_loc;
      }
   }

   PetscCall(ISCreateGeneral(MPI_COMM_MATRIX, fine_counter,
         fine_indices,
         PETSC_COPY_VALUES, is_fine));
   PetscCall(ISCreateGeneral(MPI_COMM_MATRIX, coarse_counter,
         coarse_indices,
         PETSC_COPY_VALUES, is_coarse));

   PetscCall(PetscFree(fine_indices));
   PetscCall(PetscFree(coarse_indices));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode first_pass_splitting(Mat input_mat, PetscBool skip_symmetrize, PetscReal strong_threshold,
   int max_luby_steps, int cf_splitting_type, int **cf_markers_local)
{
   // Compute a strength matrix and then call the first pass of CF splitting
   // skip_symmetrize skips symmetrizing the strength matrix for the PMISR DDC,
   // diag dom and aggregation splittings - the PMIS-based splittings always
   // symmetrize as they are defined on the symmetrized graph

   // ~~~~~~

   // Local
   PetscInt global_row_start, global_row_end_plus_one, i_loc;
   PetscInt local_rows, local_cols, ncols;
   PetscMPIInt comm_size;
   MPI_Comm MPI_COMM_MATRIX;
   Mat strength_mat = NULL;
   PetscInt *aggregates = NULL;
   Mat Ad, Ao;
   const PetscInt *colmap;
#if defined(PETSC_HAVE_KOKKOS)
   MatType mat_type;
   PetscBool is_kokkos;
#endif

   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)input_mat, &MPI_COMM_MATRIX));
   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));
   PetscCall(MatGetLocalSize(input_mat, &local_rows, &local_cols));
   PetscCall(MatGetOwnershipRange(input_mat, &global_row_start, &global_row_end_plus_one));

   // ~~~~~~~~~~~~
   // Generate the strength matrix
   // ~~~~~~~~~~~~

   // Use classical strength of connection
   if (cf_splitting_type == CF_PMIS_DIST2) {

      // Square the strength of connection graph - S'S + S
      // Note we are symmetrizing the strength matrix here
      PetscCall(generate_sabs(input_mat, strong_threshold, PETSC_TRUE, PETSC_TRUE, &strength_mat, NULL, NULL));

   } else if (cf_splitting_type == CF_PMIS) {

      // Note we are symmetrizing the strength matrix here
      PetscCall(generate_sabs(input_mat, strong_threshold, PETSC_TRUE, PETSC_FALSE, &strength_mat, NULL, NULL));

   } else if (cf_splitting_type == CF_DIAG_DOM) {
      // Symmetrize the strength matrix unless asked to skip it

      // Tried to generate a strength matrix based on the relative size compared to the
      // diagonal, but it produces a worse initial coarsening when fed to PMISR, making
      // the DDC cleanup take a lot more work
      //call generate_sabs(input_mat, strong_threshold, .NOT. skip_symmetrize, .FALSE., strength_mat, &
      //         allow_diag_strength = .TRUE.)
      PetscCall(generate_sabs(input_mat, strong_threshold, !skip_symmetrize, PETSC_FALSE, &strength_mat, NULL, NULL));

   // PMISR DDC and Aggregation
   } else {
      // Symmetrize the strength matrix unless asked to skip it
      PetscCall(generate_sabs(input_mat, strong_threshold, !skip_symmetrize, PETSC_FALSE, &strength_mat, NULL, NULL));
   }

   // ~~~~~~~~~~~~
   // Do the first pass splitting
   // ~~~~~~~~~~~~

   // PMISR
   if (cf_splitting_type == CF_PMISR_DDC || cf_splitting_type == CF_DIAG_DOM) {

      PetscCall(pmisr(strength_mat, max_luby_steps, PETSC_FALSE, cf_markers_local, NULL));

   // Distance 1 PMIS
   } else if (cf_splitting_type == CF_PMIS) {

      PetscCall(pmisr(strength_mat, max_luby_steps, PETSC_TRUE, cf_markers_local, NULL));

   // Distance 2 PMIS
   } else if (cf_splitting_type == CF_PMIS_DIST2) {

      // As we have generated S'S + S for the strength matrix, this will do distance 2 PMIS
      PetscCall(pmisr(strength_mat, max_luby_steps, PETSC_TRUE, cf_markers_local, NULL));

   // PMIS on boundary nodes then processor local aggregation
   } else if (cf_splitting_type == CF_PMIS_AGG) {

      // In parallel we do distance 1 pmis on boundary nodes, then do serial aggregation
      if (comm_size != 1) {

         // Do distance 1 PMIS
         PetscCall(pmisr(strength_mat, max_luby_steps, PETSC_TRUE, cf_markers_local, NULL));

#if defined(PETSC_HAVE_KOKKOS)
         // The Kokkos pmisr leaves the cf_markers on the device (it only copies
         // them back to the host when debugging), but the boundary reset and
         // aggregation below read the host cf_markers_local
         PetscCall(MatGetType(strength_mat, &mat_type));
         PetscCall(PetscStrcmpAny(mat_type, &is_kokkos, MATMPIAIJKOKKOS, MATSEQAIJKOKKOS, MATAIJKOKKOS, ""));
         if (is_kokkos && !kokkos_debug()) {
            copy_cf_markers_d2h(*cf_markers_local);
         }
#endif

         // Get the sequential part of the matrix
         PetscCall(MatMPIAIJGetSeqAIJ(strength_mat, &Ad, &Ao, &colmap));

         // For any local node that doesn't touch a boundary node, we set the
         // cf_markers back to unassigned and leave them to be done by the aggregation
         for (i_loc = 0; i_loc < local_rows; i_loc++) {

            // Get how many non-zeros are in the off-diagonal
            PetscCall(MatGetRow(Ao, i_loc, &ncols, NULL, NULL));
            // If we don't touch anything off-processor, its local and set it to unassigned
            if (ncols == 0) (*cf_markers_local)[i_loc] = 0;
            PetscCall(MatRestoreRow(Ao, i_loc, &ncols, NULL, NULL));
         }

      } else {
         Ad = strength_mat;
      }

      // Call an aggregation algorithm but only on the local Ad
      PetscCall(generate_serial_aggregation(Ad, cf_markers_local, &aggregates));
      PetscCall(PetscFree(aggregates));

   // Processor local aggregation
   } else if (cf_splitting_type == CF_AGG) {

      if (comm_size != 1) {
         PetscCall(MatMPIAIJGetSeqAIJ(strength_mat, &Ad, &Ao, &colmap));
      } else {
         Ad = strength_mat;
      }

      // Call an aggregation algorithm but only on the local Ad
      PetscCall(generate_serial_aggregation(Ad, cf_markers_local, &aggregates));
      PetscCall(PetscFree(aggregates));

   } else {
      // Just do the same thing as distance 2 multiple times if you want higher distance
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unknown CF splitting algorithm");
   }

   // Destroy the strength matrix
   PetscCall(MatDestroy(&strength_mat));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode compute_cf_splitting_c(Mat input_mat, PetscBool skip_symmetrize,
   PetscReal strong_threshold, int max_luby_steps,
   int cf_splitting_type, int ddc_its, PetscReal fraction_swap,
   IS *is_fine, IS *is_coarse,
   const int *cr_inverse_type, const int *cr_poly_order, const int *cr_inverse_sparsity_order,
   const PetscBool *cr_diag_scale_polys)
{
   // Computes a CF splitting and returns the F and C point ISs
   // skip_symmetrize skips symmetrizing the strength matrix (it does not
   // apply to the PMIS-based splittings, which are defined on the
   // symmetrized graph and always symmetrize)
   // The optional cr_* arguments only apply to CF_CR and set the
   // approximate inverse used as the CR relaxation - PCAIR passes its
   // Aff inverse settings so the CR rate certifies the F-solve actually
   // applied in the hierarchy; when absent an assembled arnoldi-basis
   // GMRES polynomial of order PFLARE_CR_POLY_ORDER with sparsity order 1
   // and no diagonal scaling is used, matching the PCAIR defaults

   // ~~~~~~
   int *cf_markers_local = NULL;
   int its, ddc_its_max;
   PetscBool need_intermediate_is;
   PetscReal max_dd_ratio_achieved, cr_rate_achieved;
   PetscInt local_rows, local_cols, n_swapped, i_loc;
   int cr_inverse_type_use, cr_poly_order_use, cr_sparsity_order_use;
   PetscBool cr_diag_scale_use;
#if defined(PETSC_HAVE_KOKKOS)
   MatType mat_type;
   Mat A_array;
   IS is_fine_array, is_coarse_array;
   IS is_fine_kokkos = NULL, is_coarse_kokkos = NULL;
   PetscBool fine_equal, coarse_equal;
   PetscBool is_kokkos;
#endif

   // ~~~~~~

   PetscFunctionBegin;

   // In Kokkos the DDC and PMISR do everything on the device
   // that means we don't need to create intermediate is_fine and is_coarse ISs
   need_intermediate_is = PETSC_TRUE;

#if defined(PETSC_HAVE_KOKKOS)

   PetscCall(MatGetType(input_mat, &mat_type));
   PetscCall(PetscStrcmpAny(mat_type, &is_kokkos, MATMPIAIJKOKKOS, MATSEQAIJKOKKOS, MATAIJKOKKOS, ""));
   if (is_kokkos) {

         // If kokkos debugging is on, the pmisr and ddc do
         // copy to the host after they finish in order to do the comparisons
         // and hence we do need the intermediate ISs
         // If doing pmis agg, the initial pmis will be on the device so
         // first_pass_splitting always triggers the d2h copy and we build the intermediate is
         // CR never runs the pmisr/ddc on the device (its heavy numerics are all
         // default petsc ops that run on the device anyway) so its host
         // cf_markers_local and ISs are always the authoritative ones
         if (!kokkos_debug() &&
               (cf_splitting_type != CF_AGG && cf_splitting_type != CF_PMIS_AGG &&
                cf_splitting_type != CF_CR)) {
            need_intermediate_is = PETSC_FALSE;
         }
   }
#endif

   // CR builds its splitting from scratch - all points start as F and each
   // cr_pass promotes an independent set of the slowest converging
   // F rows to C until the target CR rate is reached
   // There is no strength matrix or first pass, so for CR the strong_threshold
   // carries the target CR rate (just like it carries the target diagonal
   // dominance ratio for CF_DIAG_DOM)
   if (cf_splitting_type == CF_CR) {

      // The CR relaxation defaults, overridden by PCAIR with its Aff
      // inverse settings
      // Matches the default -pc_air_inverse_type
      cr_inverse_type_use = PFLAREINV_ARNOLDI;
      cr_poly_order_use = PFLARE_CR_POLY_ORDER;
      cr_sparsity_order_use = 1;
      cr_diag_scale_use = PETSC_FALSE;
      if (cr_inverse_type) cr_inverse_type_use = *cr_inverse_type;
      if (cr_poly_order) cr_poly_order_use = *cr_poly_order;
      if (cr_inverse_sparsity_order) cr_sparsity_order_use = *cr_inverse_sparsity_order;
      if (cr_diag_scale_polys) cr_diag_scale_use = *cr_diag_scale_polys;

      PetscCall(MatGetLocalSize(input_mat, &local_rows, &local_cols));
      PetscCall(PetscMalloc1(local_rows, &cf_markers_local));
      for (i_loc = 0; i_loc < local_rows; i_loc++) cf_markers_local[i_loc] = PFLARE_F_POINT;
      PetscCall(create_cf_is(input_mat, cf_markers_local, is_fine, is_coarse));

      // cr_its_loop
      for (its = 0; its < PFLARE_CR_MAX_ITS; its++) {

         // Directly modifies the values in cf_markers_local
         PetscCall(cr_pass(input_mat, *is_fine, strong_threshold,
                  cr_inverse_type_use, cr_poly_order_use,
                  cr_sparsity_order_use, cr_diag_scale_use,
                  &cr_rate_achieved, &n_swapped, cf_markers_local));

         // If we swapped anything the ISs are now outdated
         if (n_swapped > 0) {
            PetscCall(ISDestroy(is_fine));
            PetscCall(ISDestroy(is_coarse));
            PetscCall(create_cf_is(input_mat, cf_markers_local, is_fine, is_coarse));
         }

         // Terminate if we've hit the target rate or we can't make progress
         if (cr_rate_achieved <= strong_threshold || n_swapped == 0) break;
      }

   } else {

      // Generate the strength matrix and do the first pass CF splitting
      PetscCall(first_pass_splitting(input_mat, skip_symmetrize, strong_threshold,
               max_luby_steps, cf_splitting_type, &cf_markers_local));

      // Create the IS for the CF splittings
      if (need_intermediate_is) PetscCall(create_cf_is(input_mat, cf_markers_local, is_fine, is_coarse));
   }

   // Only do the DDC pass if we're doing PMISR_DDC
   // and if we haven't requested an exact independent set, ie strong threshold is not zero
   // as this gives diagonal Aff)
   if (strong_threshold != 0.0 &&
         (cf_splitting_type == CF_PMISR_DDC || cf_splitting_type == CF_DIAG_DOM)) {

      // Do a set number of DDC iterations for PMISR_DDC.
      // For CF_DIAG_DOM, iterate until the requested strong_threshold ratio is reached.
      ddc_its_max = ddc_its;
      if (cf_splitting_type == CF_DIAG_DOM) ddc_its_max = INT_MAX;

      // ddc_its_loop
      for (its = 0; its < ddc_its_max; its++) {

         // Do the second pass cleanup - this will directly modify the values in cf_markers_local
         // (or the equivalent device cf_markers, is_fine is ignored if on the device)
         max_dd_ratio_achieved = 0.0;
         if (cf_splitting_type == CF_DIAG_DOM) max_dd_ratio_achieved = strong_threshold;
         PetscCall(ddc(input_mat, *is_fine, fraction_swap, &max_dd_ratio_achieved, cf_markers_local));

         // If we did anything in our ddc second pass and hence need to rebuild
         // the is_fine and is_coarse
         if ((fraction_swap != 0.0 || max_dd_ratio_achieved != 0.0)
               && need_intermediate_is) {

            // These are now outdated
            PetscCall(ISDestroy(is_fine));
            PetscCall(ISDestroy(is_coarse));

            // Create the new CF ISs
            PetscCall(create_cf_is(input_mat, cf_markers_local, is_fine, is_coarse));
         }

         // Terminate if we've reached the ratio
         if (cf_splitting_type == CF_DIAG_DOM && max_dd_ratio_achieved < strong_threshold) break;
      }
   }

   // The Kokkos PMISR and DDC no longer copy back to the host to save copies
   // during an arbritrary number of iterations above
   // We need to explicitly copy the cf_markers_local back to the host once
   // we've finished both the PMISR and all the DDC iterations
   // And we also need to build the host IS's
#if defined(PETSC_HAVE_KOKKOS)

   if (is_kokkos) {

      // The initial pmis will be on the device, so need to destroy
      // Intermediate ISs are created in that case
      if (cf_splitting_type == CF_PMIS_AGG) {
         // Destroys the device cf_markers_local
         delete_device_cf_markers();
         delete_device_diag_dom_ratio();
      }

      // Aggregation is not on the device at all, and CR never creates
      // device cf_markers (its host ISs are already the final answer)
      if (cf_splitting_type != CF_AGG && cf_splitting_type != CF_PMIS_AGG &&
            cf_splitting_type != CF_CR) {

         A_array = input_mat;
         is_fine_array = is_fine_kokkos;
         is_coarse_array = is_coarse_kokkos;

         // Create the host is_fine and is_coarse based on device cf_markers
         create_cf_is_kokkos(&A_array, &is_fine_array, &is_coarse_array);
         is_fine_kokkos = is_fine_array;
         is_coarse_kokkos = is_coarse_array;

         // If we are doing debugging the IS's are already created above
         // so let's compare them to our kokkos ones
         if (kokkos_debug()) {

            PetscCall(ISEqual(*is_fine, is_fine_kokkos, &fine_equal));
            PetscCheck(fine_equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Kokkos and CPU versions of create_cf_is_kokkos for fine do not match");
            PetscCall(ISEqual(*is_coarse, is_coarse_kokkos, &coarse_equal));
            PetscCheck(coarse_equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Kokkos and CPU versions of create_cf_is_kokkos for coarse do not match");
            // Destroy the extra IS's
            PetscCall(ISDestroy(&is_fine_kokkos));
            PetscCall(ISDestroy(&is_coarse_kokkos));

         // If we're not debugging just copy over the ones we built on the device
         } else {
            *is_fine = is_fine_kokkos;
            *is_coarse = is_coarse_kokkos;
         }

         // Destroys the device cf_markers_local
         delete_device_cf_markers();
         delete_device_diag_dom_ratio();
      }

   }
#endif

   PetscCall(PetscFree(cf_markers_local));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------
