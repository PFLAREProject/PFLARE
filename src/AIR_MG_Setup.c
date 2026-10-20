/*
  C port of AIR_MG_Setup.F90 (module air_mg_setup)

  The levels are 0-based, so our_level indexes the per level arrays of
  air_multigrid_data directly (the Fortran our_level is our_level + 1, and the
  Fortran no_levels index is no_levels - 1). The two conventions meet where the
  levels are visible to the user, the auto truncation start level and the stats
  printed with -pc_air_print_stats_timings, which are 1-based as before, and in
  the petsc level of each of our levels (petsc_level = no_levels - 1 - our_level)
  The placeholder matshells of the levels the FC smoothers run on have a
  mat_ctxtype context holding the 0-based our_level and air_data
  The Fortran called start/finish_approximate_inverse with the buffers and the
  coefficients of a gmres_poly_data, which is what is passed here (poly_data_buffers
  creates the TSQR buffers on first use)
  The prints use explicit formats rather than the list-directed print of the
  Fortran, keeping the labels tools/parse_pflare_output.py reads (convention 8 of
  docs/dev/fortran_removal_plan.md)
*/

#include <petscksp.h>
#include "pflareimpl.h"
#include "Constrain_Z_or_W.h"
#include "CF_Splitting.h"
#include "Approx_Inverse_Setup.h"
#include "PETSc_Helper.h"
#include "Timers.h"
#include "AIR_MG_Stats.h"
#include "FC_Smooth.h"
#include "FC_Smooth_Block.h"
#include "C_PETSc_Routines.h"
#include "AIR_Operators_Setup.h"
#include "Gmres_Poly.h"
#include "Gmres_Poly_Apply.h"
#include "Matshell_Data_Type.h"
#include "Gmres_Poly_Data_Type.h"
#include "Repartition.h"
#include "AIR_MG_Setup.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode setup_air_pcmg(Mat amat, Mat pmat, air_multigrid_data *air_data, PC pcmg_input)
{
   // Setup AIR by computing the hierarchy and returning a PETSC PCMG object
   // Have to have called create_air_data before this routine
   // The options for AIR are stored in air_data%options

   // ~~~~~~
   // Local
   PetscInt            local_rows, local_cols, global_rows, global_cols;
   PetscInt            local_fine_is_size, local_coarse_is_size;
   PetscInt            global_coarse_is_size, global_fine_is_size, global_row_start;
   PetscInt            global_row_end_plus_one, no_active_cores;
   PetscInt            prolongator_start, prolongator_end_plus_one, proc_stride;
   PetscInt            petsc_level, no_levels_petsc_int;
   PetscInt            local_vec_size, ystart, yend, local_rows_repart, local_cols_repart;
   PetscInt            global_rows_repart, global_cols_repart;
   PetscInt            i_loc, null_vecs_size;
   int                 inverse_type_aff, inverse_sparsity_aff;
   int                 no_levels, our_level, our_level_coarse;
   PetscMPIInt         comm_rank, comm_size;
   MPI_Comm            MPI_COMM_MATRIX;
   PetscReal           ratio_local_nnzs_off_proc, achieved_rel_tol, norm_b, time;
   PetscBool           continue_coarsening, trigger_proc_agglom, cf_split_reused;
   KSP                 ksp_smoother_up, ksp_smoother_down, ksp_coarse_solver;
   PC                  pc_smoother_up, pc_smoother_down, pc_coarse_solver;
   Vec                 temp_coarse_vec = NULL, rand_vec, sol_vec, temp_vec, diag_vec, diag_vec_aff;
   IS                  is_unchanged, is_full = NULL;
   PetscInt            one = 1, zero = 0;
   Vec                 *left_null_vecs = NULL, *right_null_vecs = NULL;
   Vec                 *left_null_vecs_c = NULL, *right_null_vecs_c = NULL;
   VecScatter          vec_scatter = NULL;
   VecType             vec_type = NULL;
   PetscBool           auto_truncated, aff_diag, check_diag_only;
   PetscRandom         rctx;
   MatType             mat_type, mat_type_aff, mat_type_inv_aff;
   int                 diag_only;
   PetscBool           slow_veciscopy, is_diagonal, is_shell;
   // The Kokkos IS views that MatCreateSubMatrixWrapper can use number the levels from 1
   int                 our_level_kokkos;
   PetscBool           true_bool = PETSC_TRUE;
   tsqr_buffers        *buffers;
   mat_ctxtype         *mat_ctx;
   // The cr_* arguments of compute_cf_splitting_c
   int                 cr_inverse_type, cr_poly_order, cr_inverse_sparsity_order;
   PetscBool           cr_diag_scale_polys;
   // ~~~~~~

   PetscFunctionBegin;

   // Start timing the setup
   PetscCall(timer_start(PFLARE_TIMER_ID_AIR_SETUP));

   // Get the communicator the input matrix is on, we build everything on that
   PetscCall(PetscObjectGetComm((PetscObject)pmat, &MPI_COMM_MATRIX));
   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));
   // Get the comm rank
   PetscCallMPI(MPI_Comm_rank(MPI_COMM_MATRIX, &comm_rank));

   // The max number of levels
   no_levels = (int)air_data->options.max_levels;
   // Keep track of how many times we've done processor agglomeration
   // ie the stride between active mpi ranks
   proc_stride = 1;
   // Copy the top grid matrix pointer
   air_data->coarse_matrix[0] = pmat;

   // If on the cpu we have a veciscopy which is fast
   // If on the gpu with kokkos we have a veciscopy which is fast
   // If on the gpu without kokkos the veciscopy involves the host which is slow
   // For the slow ones we instead create some extra matrices to use during the smoothing
   // mat_type points into pmat, which outlives the setup
   PetscCall(MatGetType(air_data->coarse_matrix[0], &mat_type));
   air_data->fast_veciscopy_exists = PETSC_TRUE;
   PetscCall(PetscStrcmpAny(mat_type, &slow_veciscopy,
       MATSEQAIJCUSPARSE, MATMPIAIJCUSPARSE, MATAIJCUSPARSE,
       MATSEQAIJHIPSPARSE, MATMPIAIJHIPSPARSE, MATAIJHIPSPARSE,
       MATSEQAIJVIENNACL, MATMPIAIJVIENNACL, MATAIJVIENNACL,
       MATDENSECUDA, MATDENSEHIP,
       MATSEQDENSECUDA, MATSEQDENSEHIP,
       MATMPIDENSECUDA, MATMPIDENSEHIP, ""));
   if (slow_veciscopy) {

      air_data->fast_veciscopy_exists = PETSC_FALSE;
   }

   // ~~~~~~~~~~~~~~~~~~~~~
   // Check if the user has provided a near nullspace before we do anything
   // ~~~~~~~~~~~~~~~~~~~~~
   PetscCall(get_near_nullspace(amat, air_data->options.constrain_z, air_data->options.constrain_w,
            &left_null_vecs, &right_null_vecs, &null_vecs_size));

   // ~~~~~~~~~~~~~~~~~~~~~

   if (air_data->options.print_stats_timings && comm_rank == 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, " Timers are cumulative\n"));

   auto_truncated = PETSC_FALSE;

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   // Loop over the number of levels
   // We will exit this loop once we coarsen far enough
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   for (our_level = 0; our_level < no_levels - 1; our_level++) {

      // This is our coarse level
      our_level_coarse = our_level + 1;

      // Get matrix sizes
      PetscCall(MatGetSize(air_data->coarse_matrix[our_level], &global_rows, &global_cols));
      PetscCall(MatGetLocalSize(air_data->coarse_matrix[our_level], &local_rows, &local_cols));
      // This returns the global index of the local portion of the matrix
      PetscCall(MatGetOwnershipRange(air_data->coarse_matrix[our_level], &global_row_start, &global_row_end_plus_one));

      continue_coarsening = PETSC_TRUE;

      // ~~~~~~~~~~
      // We can also check if our coarse grid approximations are good enough to work as a coarse grid solver
      // If so we can stop coarsening here
      // This is really only a sensible idea when using a matrix-free polynomial for the coarse grid solve
      // Otherwise building assembled approximation inverses can be very expensive!
      // ~~~~~~~~~~
      // We already know how many coarse levels we have if we are re-using
      // The auto truncation start level is the 1-based level the user sees
      if (!air_data->allocated_matrices_A_ff[our_level] &&
                  our_level + 1 >= air_data->options.auto_truncate_start_level &&
                  air_data->options.auto_truncate_start_level != -1) {

         PetscCall(timer_start(PFLARE_TIMER_ID_AIR_TRUNCATE));

         // Set up our coarse inverse data
         PetscCall(setup_gmres_poly_data(global_rows,
                  air_data->options.coarsest_inverse_type,
                  (int)air_data->options.coarsest_poly_order,
                  (int)air_data->options.coarsest_inverse_sparsity_order,
                  air_data->options.coarsest_subcomm,
                  proc_stride,
                  &air_data->inv_coarsest_poly_data));

         // Start the approximate inverse we'll use on this level
         PetscCall(poly_data_buffers(&air_data->inv_coarsest_poly_data, &buffers));
         PetscCall(start_approximate_inverse(air_data->coarse_matrix[our_level],
               air_data->inv_coarsest_poly_data.inverse_type,
               air_data->inv_coarsest_poly_data.gmres_poly_order,
               air_data->options.coarsest_diag_scale_polys,
               buffers, &air_data->inv_coarsest_poly_data.coefficients,
               &air_data->inv_coarsest_poly_data.coefficients_rows,
               &air_data->inv_coarsest_poly_data.coefficients_cols));

         // This will be a vec of randoms that differ from those used to create the gmres polynomials
         // We will solve Ax = rand_vec to test how good our coarse solver is
         PetscCall(PetscRandomCreate(MPI_COMM_MATRIX, &rctx));
         PetscCall(MatCreateVecs(air_data->coarse_matrix[our_level],
                  &rand_vec, NULL));

         PetscCall(VecSetRandom(rand_vec, rctx));
         PetscCall(PetscRandomDestroy(&rctx));
         PetscCall(VecNorm(rand_vec, NORM_2, &norm_b));

         PetscCall(VecDuplicate(rand_vec, &sol_vec));
         PetscCall(VecDuplicate(rand_vec, &temp_vec));

         // Finish our approximate inverse
         PetscCall(poly_data_buffers(&air_data->inv_coarsest_poly_data, &buffers));
         PetscCall(finish_approximate_inverse(air_data->coarse_matrix[our_level],
               air_data->inv_coarsest_poly_data.inverse_type,
               air_data->inv_coarsest_poly_data.gmres_poly_order,
               air_data->inv_coarsest_poly_data.gmres_poly_sparsity_order,
               buffers, air_data->inv_coarsest_poly_data.coefficients,
               air_data->inv_coarsest_poly_data.coefficients_rows,
               air_data->inv_coarsest_poly_data.coefficients_cols,
               air_data->options.coarsest_matrix_free_polys,
               air_data->options.coarsest_diag_scale_polys,
               &air_data->reuse[our_level].reuse_mat[MAT_INV_AFF],
               &air_data->reuse[our_level].reuse_submatrices[MAT_INV_AFF].array,
               &air_data->inv_A_ff[our_level]));

         // We have a custom MF routine for computing just the residual
         // when applying the Newton form of the gmres polynomial that saves some flops
         // We don't actually need the solution here as it's useless
         if ((air_data->inv_coarsest_poly_data.inverse_type == PFLARE_PFLAREINV_NEWTON ||
               air_data->inv_coarsest_poly_data.inverse_type == PFLARE_PFLAREINV_NEWTON_NO_EXTRA) &&
               air_data->options.coarsest_matrix_free_polys) {

            if (air_data->options.coarsest_diag_scale_polys) {
               PetscCall(petsc_matvec_right_scale_poly_newton_residual_mf(air_data->inv_A_ff[our_level], rand_vec, temp_vec));
            } else {
               PetscCall(petsc_matvec_poly_newton_residual_mf(air_data->inv_A_ff[our_level], rand_vec, temp_vec));
            }
        } else {

            // sol_vec = A^-1 * rand_vec
            PetscCall(MatMult(air_data->inv_A_ff[our_level], rand_vec, sol_vec));
            // Now calculate a residual
            // A * sol_vec
            PetscCall(MatMult(air_data->coarse_matrix[our_level], sol_vec, temp_vec));
            // Now A * sol_vec - rand_vec
            PetscCall(VecAXPY(temp_vec, PFLARE_MINUS_ONE, rand_vec));
         }

         // Get the achieved norm
         PetscCall(VecNorm(temp_vec, NORM_2, &achieved_rel_tol));

         // If it's good enough we can truncate on this level and our coarse solver has been computed
         if (achieved_rel_tol/norm_b < air_data->options.auto_truncate_tol) {
            auto_truncated = PETSC_TRUE;

            // Delete temporary if not reusing
            if (!air_data->options.reuse_sparsity ||
                !REUSE_MAT_ACTIVE[MAT_INV_AFF][air_data->options.reuse_amount - 1]) {
               PetscCall(destroy_matrix_reuse(&air_data->reuse[our_level].reuse_mat[MAT_INV_AFF],
                        &air_data->reuse[our_level].reuse_submatrices[MAT_INV_AFF].array));
            }

         // If this isn't good enough, destroy everything we used - no chance for reuse
         } else {
            PetscCall(reset_inverse_mat(&air_data->inv_A_ff[our_level]));
            PetscCall(destroy_matrix_reuse(&air_data->reuse[our_level].reuse_mat[MAT_INV_AFF],
                     &air_data->reuse[our_level].reuse_submatrices[MAT_INV_AFF].array));
            // The coefficients were sized for this level, but the coarsest grid may be
            // smaller than the polynomial order (setup_gmres_poly_data then lowers the order)
            // so throw them away and let them be reallocated with the right size
            PetscCall(reset_poly_data(&air_data->inv_coarsest_poly_data));
         }

         PetscCall(VecDestroy(&rand_vec));
         PetscCall(VecDestroy(&sol_vec));
         PetscCall(VecDestroy(&temp_vec));

         PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_TRUNCATE));
      }

      // ~~~~~~~~~~~~
      // Compute the coarsening
      // ~~~~~~~~~~~~
      PetscCall(timer_start(PFLARE_TIMER_ID_AIR_COARSEN));

      // Track whether the CF splitting was already present before this setup
      // (used below to avoid redundant device IS copies when reusing)
      cf_split_reused = air_data->allocated_is[our_level];

      // Are we reusing our CF splitting
      if (!cf_split_reused && !auto_truncated) {

         // Do the CF splitting
         // The cr_* arguments only apply to the CF_CR splitting - the CR
         // relaxation mirrors the settings of the Aff inverse used in the
         // hierarchy (though the CR always uses an assembled inverse
         // regardless of matrix_free_polys, as the ideal restrictor is
         // always built from the assembled inverse and that is the binding
         // constraint on the splitting quality)
         // Always symmetrize the strength matrix - the symmetric option
         // only controls whether the prolongator is defined as R^T
         cr_inverse_type = air_data->options.inverse_type;
         cr_poly_order = (int)air_data->options.poly_order;
         cr_inverse_sparsity_order = (int)air_data->options.inverse_sparsity_order;
         cr_diag_scale_polys = air_data->options.diag_scale_polys;
         PetscCall(compute_cf_splitting_c(air_data->coarse_matrix[our_level],
               PETSC_FALSE,
               air_data->options.strong_threshold,
               (int)air_data->options.max_luby_steps,
               air_data->options.cf_splitting_type,
               (int)air_data->options.ddc_its,
               air_data->options.ddc_fraction,
               &air_data->IS_fine_index[our_level], &air_data->IS_coarse_index[our_level],
               &cr_inverse_type,
               &cr_poly_order,
               &cr_inverse_sparsity_order,
               &cr_diag_scale_polys));
         air_data->allocated_is[our_level] = PETSC_TRUE;
      }

      PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_COARSEN));

      // ~~~~~~~~~~~~~~
      // Get the sizes of C and F points
      // ~~~~~~~~~~~~~~
      if (!auto_truncated) {

         PetscCall(ISGetSize(air_data->IS_fine_index[our_level], &global_fine_is_size));
         PetscCall(ISGetLocalSize(air_data->IS_fine_index[our_level], &local_fine_is_size));
         PetscCall(ISGetSize(air_data->IS_coarse_index[our_level], &global_coarse_is_size));
         PetscCall(ISGetLocalSize(air_data->IS_coarse_index[our_level], &local_coarse_is_size));

      // We're not continuing the coarsening anyway, this is just to ensure the continue_coarsening
      // test below doesn't break
      } else {
         global_fine_is_size = 0;
         global_coarse_is_size = 0;
      }

      // Do we want to keep coarsening?
      // We check if our coarse grid solve is already good enough and
      // if the problem is still big enough and
      // that the coarsening resulted in any fine points, sometimes
      // you can have it such that no fine points are selected
      continue_coarsening = (!auto_truncated &&
               (global_coarse_is_size > air_data->options.coarse_eq_limit && global_fine_is_size != 0)) ? PETSC_TRUE : PETSC_FALSE;

      // Did we end up with a coarse grid we still want to coarsen?
      if (continue_coarsening) {

         // Output stats on the coarsening
         // The level printed is the 1-based level the user sees
         if (air_data->options.print_stats_timings && comm_rank == 0) {
            PetscCall(PetscPrintf(PETSC_COMM_SELF, " ~~~~~~~~~~~~ Level %d\n", our_level + 1));
            PetscCall(PetscPrintf(PETSC_COMM_SELF, " Global rows %" PetscInt_FMT " Global F-points %" PetscInt_FMT " Global C-points %" PetscInt_FMT "\n",
                     global_rows, global_fine_is_size, global_coarse_is_size));
         }

      // If this coarse grid is smaller than our minimum, then we are done coarsening
      } else {

         // Get the size of the previous coarse grid as that is now our bottom grid
         // If we're on the top grid and we have coarsened fast enough to not have a second level
         // should only happen on very small problems
         if (our_level == 0) {
            global_fine_is_size = global_rows;
            local_fine_is_size = local_rows;
         } else {
            PetscCall(ISGetSize(air_data->IS_coarse_index[our_level-1], &global_fine_is_size));
            PetscCall(ISGetLocalSize(air_data->IS_coarse_index[our_level-1], &local_fine_is_size));
         }

         // This level is the coarsest, so we have one more level than its 0-based index
         no_levels = our_level + 1;

         // Exit out of the coarsening loop
         break;

      }

      // ~~~~~~~~~~~~~~
      // Copy over the fine/coarse IS's to the device, we use them
      // in our matrix extract to save h2d copies, and they are also used
      // in the fc smooth.
      // Skip when the CF splitting was reused: the device copy already exists
      // and the IS indices are unchanged.
      // ~~~~~~~~~~~~~~
      if (!cf_split_reused) {
         PetscCall(timer_start(PFLARE_TIMER_ID_AIR_IDENTITY));
         PetscCall(create_VecISCopyLocalWrapper(air_data, our_level,
               mat_type, air_data->coarse_matrix[our_level]));
         PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_IDENTITY));
      }

      // ~~~~~~~~~~~~~~
      // Now let's go and build all our operators
      // ~~~~~~~~~~~~~~

      // ~~~~~~~~~
      // Let's smooth near null-space vectors if needed
      // ~~~~~~~~~
      if (air_data->options.constrain_z || air_data->options.constrain_w) {
         PetscCall(timer_start(PFLARE_TIMER_ID_AIR_CONSTRAIN));
         PetscCall(smooth_near_nullspace(air_data->coarse_matrix[our_level],
            air_data->options.constrain_z,
            air_data->options.constrain_w,
            left_null_vecs, null_vecs_size, right_null_vecs, null_vecs_size));
         PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_CONSTRAIN));
      }

      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      // We need to pull out Aff before we do anything
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      if (!air_data->smooth_order_levels[our_level].array) {
         PetscCall(PetscMalloc1(AIR_SMOOTH_ORDER_SIZE, &air_data->smooth_order_levels[our_level].array));
         air_data->smooth_order_levels[our_level].size = AIR_SMOOTH_ORDER_SIZE;
      }
      for (i_loc = 0; i_loc < AIR_SMOOTH_ORDER_SIZE; i_loc++) {
         air_data->smooth_order_levels[our_level].array[i_loc] = air_data->options.smooth_order[i_loc];
      }

      PetscCall(timer_start(PFLARE_TIMER_ID_AIR_EXTRACT));

      // Pull out A_ff
      our_level_kokkos = our_level + 1;
      if (air_data->allocated_matrices_A_ff[our_level]) {
         PetscCall(MatGetType(air_data->A_ff[our_level], &mat_type_aff));
         PetscCall(PetscStrcmp(mat_type_aff, MATDIAGONAL, &is_diagonal));

         // If our Aff was previously converted to a matdiagonal, can't call matcreatesubmatrix
         // with MAT_REUSE_MATRIX.  However a diagonal matrix always has the same sparsity
         // structure regardless of any changes to the coarse matrix, so this path is safe
         // at all reuse_amount levels.
         if (is_diagonal) {

            // Easy to get out Aff if we know its diagonal
            PetscCall(MatDiagonalGetDiagonal(air_data->A_ff[our_level], &diag_vec_aff));
            PetscCall(MatCreateVecs(air_data->coarse_matrix[our_level], &diag_vec, NULL));
            // Get the matrix diagonal
            PetscCall(MatGetDiagonal(air_data->coarse_matrix[our_level], diag_vec));
            // Pull out the F points in the diagonal - this happens on the host
            // if this becomes expensive could call something like our version of VecISCopyLocal_kokkos
            PetscCall(VecISCopy(diag_vec, air_data->IS_fine_index[our_level],
                     SCATTER_REVERSE, diag_vec_aff));
            PetscCall(MatDiagonalRestoreDiagonal(air_data->A_ff[our_level], &diag_vec_aff));
            PetscCall(VecDestroy(&diag_vec));

         // For non-diagonal Aff, MAT_REUSE_MATRIX is only safe when the coarse matrix
         // structure is guaranteed unchanged.  allocated_matrices_A_ff is only TRUE at
         // amount=3 (A_ff preserved across setups), so this branch is amount=3 only.
         } else if (air_data->options.reuse_sparsity) {
            PetscCall(MatCreateSubMatrixWrapper(air_data->coarse_matrix[our_level],
                  air_data->IS_fine_index[our_level], air_data->IS_fine_index[our_level], MAT_REUSE_MATRIX,
                  &air_data->A_ff[our_level],
                  &our_level_kokkos, &true_bool, &true_bool,
                  air_data->kokkos_is_views_handle));
         } else {
            PetscCall(MatCreateSubMatrixWrapper(air_data->coarse_matrix[our_level],
                  air_data->IS_fine_index[our_level], air_data->IS_fine_index[our_level], MAT_INITIAL_MATRIX,
                  &air_data->A_ff[our_level],
                  &our_level_kokkos, &true_bool, &true_bool,
                  air_data->kokkos_is_views_handle));
         }
      } else {
         PetscCall(MatCreateSubMatrixWrapper(air_data->coarse_matrix[our_level],
                  air_data->IS_fine_index[our_level], air_data->IS_fine_index[our_level], MAT_INITIAL_MATRIX,
                  &air_data->A_ff[our_level],
                  &our_level_kokkos, &true_bool, &true_bool,
                  air_data->kokkos_is_views_handle));
      }

      PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_EXTRACT));

      // ~~~~~~~~~
      // Check if Aff is purely diagonal
      // ~~~~~~~~~
      inverse_type_aff = air_data->options.inverse_type;
      inverse_sparsity_aff = (int)air_data->options.inverse_sparsity_order;
      aff_diag = PETSC_FALSE;
      check_diag_only = PETSC_TRUE;
      // Don't have to check if its already matdiagonal due to reuse
      // or if we have strong threshold of zero with a PMISR based splitting,
      // as then the F points are an independent set in the (symmetrized)
      // graph of A and hence Aff is diagonal
      // That is not true for the other CF splittings (e.g., PMIS gives
      // independent C points, aggregation has connected F points and
      // CR uses the strong threshold as a target rate), so we have to check
      PetscCall(MatGetType(air_data->A_ff[our_level], &mat_type_aff));
      PetscCall(PetscStrcmp(mat_type_aff, MATDIAGONAL, &is_diagonal));
      if (is_diagonal ||
            (air_data->options.strong_threshold == 0.0 &&
               (air_data->options.cf_splitting_type == CF_PMISR_DDC ||
                air_data->options.cf_splitting_type == CF_DIAG_DOM))) {
         check_diag_only = PETSC_FALSE;
      }

      // Check if Aff is only a diagonal
      if (check_diag_only) {

         PetscCall(MatGetDiagonalOnly_c(&air_data->A_ff[our_level], &diag_only));
         // If Aff is diagonal we can exploit this
         if (diag_only == 1) {
            aff_diag = PETSC_TRUE;
         }
      } else {
         aff_diag = PETSC_TRUE;
      }

      // Convert Aff to a matdiagonal type
      // Haven't rewritten some inverse types to take advantage of matdiagonal
      if (aff_diag &&
               inverse_type_aff != PFLARE_PFLAREINV_SAI &&
               inverse_type_aff != PFLARE_PFLAREINV_ISAI) {

         // We've already updated Aff above if we are reusing and it is matdiagonal already
         if (!air_data->allocated_matrices_A_ff[our_level]) {
            PetscCall(MatCreateVecs(air_data->A_ff[our_level], &diag_vec_aff, NULL));
            PetscCall(MatGetDiagonal(air_data->A_ff[our_level], diag_vec_aff));

            PetscCall(MatDestroy(&air_data->A_ff[our_level]));
            // The matrix takes ownership of diag_vec_aff and increases ref counter
            PetscCall(MatCreateDiagonal(diag_vec_aff, &air_data->A_ff[our_level]));
            PetscCall(VecDestroy(&diag_vec_aff));
         }

         // Use an exact inverse
         //inverse_type_aff = PFLAREINV_JACOBI
         // if (air_data%options%print_stats_timings .AND. comm_rank == 0) then
         //    print *, "Detected diagonal Aff - using exact inverse"
         // end if

         // If diagonal we know the sparsity is "0th" order
         inverse_sparsity_aff = 0;
         // Our approximation of diagonals is often an exact inverse
         // So set the number of F smooths to 1
         // There are no F smooths if we're smoothing all the unknowns
         if (inverse_type_aff != PFLARE_PFLAREINV_WJACOBI &&
               air_data->options.poly_order > 2 &&
               !air_data->options.full_smoothing_up_and_down) {

            // Any F smooths we just make 1 iteration
            for (i_loc = 0; i_loc < AIR_SMOOTH_ORDER_SIZE; i_loc++) {
               if (air_data->smooth_order_levels[our_level].array[i_loc] > 0) {
                  air_data->smooth_order_levels[our_level].array[i_loc] = 1;
               }
            }

            if (air_data->options.print_stats_timings && comm_rank == 0) {
               PetscCall(PetscPrintf(PETSC_COMM_SELF, " Detected diagonal Aff - setting any F smooth to 1 iteration on this level\n"));
            }
         }
      }

      // ~~~~~~~~~
      // Setup the details of our gmres polynomials
      // ~~~~~~~~~

      // If we're smoothing all the unknowns, inv_A_ff_poly_data is not the inverse of A_ff, it is
      // the inverse of the full level matrix (see AIR_Operators_Setup.F90), so none of the
      // diagonal A_ff shortcuts above can apply to it - in particular the sparsity order has to
      // stay the one the user asked for, otherwise a diagonal A_ff would force the inverse of the
      // entire level matrix to be assembled as a diagonal
      if (air_data->options.full_smoothing_up_and_down) {
         PetscCall(setup_gmres_poly_data(global_rows,
                  air_data->options.inverse_type,
                  (int)air_data->options.poly_order,
                  (int)air_data->options.inverse_sparsity_order,
                  air_data->options.subcomm,
                  proc_stride,
                  &air_data->inv_A_ff_poly_data[our_level]));
      } else {
         PetscCall(setup_gmres_poly_data(global_fine_is_size,
                  inverse_type_aff,
                  (int)air_data->options.poly_order,
                  inverse_sparsity_aff,
                  air_data->options.subcomm,
                  proc_stride,
                  &air_data->inv_A_ff_poly_data[our_level]));
      }

      // Setup the same structure for the inv_A_ff made from dropped Aff
      // This genuinely is the inverse of (dropped) A_ff, used to build Z, so it keeps the
      // A_ff based values
      PetscCall(setup_gmres_poly_data(global_fine_is_size,
               inverse_type_aff,
               (int)air_data->options.poly_order,
               inverse_sparsity_aff,
               air_data->options.subcomm,
               proc_stride,
               &air_data->inv_A_ff_poly_data_dropped[our_level]));

      // ~~~~~~~~~
      // If we're doing C-point smoothing we may have a gmres polynomial on C points
      // ~~~~~~~~~
      if (air_data->options.any_c_smooths &&
               !air_data->options.full_smoothing_up_and_down) {

         // The C point options default to the F point values if unset (-1)
         PetscCall(setup_gmres_poly_data(global_coarse_is_size,
                  air_data->options.c_inverse_type == -1 ? (int)air_data->options.inverse_type : air_data->options.c_inverse_type,
                  (int)(air_data->options.c_poly_order == -1 ? air_data->options.poly_order : air_data->options.c_poly_order),
                  (int)(air_data->options.c_inverse_sparsity_order == -1 ? air_data->options.inverse_sparsity_order : air_data->options.c_inverse_sparsity_order),
                  air_data->options.subcomm,
                  proc_stride,
                  &air_data->inv_A_cc_poly_data[our_level]));
      }

      // ~~~~~~~~~
      // Extract the submatrices and start the comms to compute the approximate inverses
      // ~~~~~~~~~
      PetscCall(get_submatrices_start_poly_coeff_comms(air_data->coarse_matrix[our_level],
            our_level, air_data));

      // ~~~~~~~~~
      // Finish the non-blocking comms and build the approximate inverse, then the
      // restrictor and prolongator
      // ~~~~~~~~~

      if (!left_null_vecs_c) PetscCall(PetscCalloc1(null_vecs_size, &left_null_vecs_c));
      if (!right_null_vecs_c) PetscCall(PetscCalloc1(null_vecs_size, &right_null_vecs_c));

      PetscCall(finish_comms_compute_restrict_prolong(air_data->coarse_matrix[our_level],
            our_level, air_data,
            left_null_vecs, null_vecs_size, right_null_vecs, null_vecs_size,
            left_null_vecs_c, right_null_vecs_c));

      if (air_data->options.constrain_z) {
         // Destroy our copy of the left near nullspace vectors
         for (i_loc = 0; i_loc < null_vecs_size; i_loc++) {
            PetscCall(VecDestroy(&left_null_vecs[i_loc]));
         }
         for (i_loc = 0; i_loc < null_vecs_size; i_loc++) left_null_vecs[i_loc] = left_null_vecs_c[i_loc];
      }
      if (air_data->options.constrain_w) {
         // Destroy our copy of the right near nullspace vectors
         for (i_loc = 0; i_loc < null_vecs_size; i_loc++) {
            PetscCall(VecDestroy(&right_null_vecs[i_loc]));
         }
         for (i_loc = 0; i_loc < null_vecs_size; i_loc++) right_null_vecs[i_loc] = right_null_vecs_c[i_loc];
      }

      // ~~~~~~~~~~~~~~
      // Build the coarse matrix
      // ~~~~~~~~~~~~~~

      PetscCall(compute_coarse_matrix(air_data->coarse_matrix[our_level], our_level, air_data,
               &air_data->coarse_matrix[our_level_coarse]));

      air_data->allocated_coarse_matrix[our_level_coarse] = PETSC_TRUE;

      // ~~~~~~~~~~~
      // We may be able to destroy the coarse matrix on our_level from here
      // If so we build a shell as a placeholder
      // ~~~~~~~~~~~

      // Always save the nnzs before potential destruction below so that the
      // complexity getters (PCAIRGetCycleComplexity etc.) work on demand
      // without requiring print_stats_timings to be enabled.
      PetscCall(get_nnzs_petsc_sparse(air_data->coarse_matrix[our_level],
               &air_data->coarse_matrix_nnzs[our_level]));

      // On every level but the top we can destroy the full operator matrix
      if (our_level != 0) {
         if (!air_data->options.full_smoothing_up_and_down) {
            PetscCall(MatDestroy(&air_data->coarse_matrix[our_level]));
         }
      }

      // If we are just doing F point smoothing, we no longer have our coarse matrix
      // But we use the mat_ctx in our F-point smoother to tell what level
      // we're on, so let's just create an empty matshell to pass in that has the right sizes
      if (!air_data->options.full_smoothing_up_and_down) {

         // The context is a mat_ctxtype that records our level and air_data
         PetscCall(create_mat_ctx(&mat_ctx));
         mat_ctx->our_level = our_level;
         mat_ctx->air_data = air_data;
         PetscCall(MatCreateShell(MPI_COMM_MATRIX, local_rows, local_cols, global_rows, global_cols,
                     mat_ctx, &air_data->coarse_matrix[our_level]));
         PetscCall(MatAssemblyBegin(air_data->coarse_matrix[our_level], MAT_FINAL_ASSEMBLY));
         PetscCall(MatAssemblyEnd(air_data->coarse_matrix[our_level], MAT_FINAL_ASSEMBLY));

         // Have to make sure to set the type of vectors the shell creates
         // Input can be any matrix, we just need the correct type
         PetscCall(MatGetVecType(air_data->A_fc[our_level], &vec_type));
         PetscCall(MatShellSetVecType(air_data->coarse_matrix[our_level], vec_type));

         // We now own the matrix on this level - on the top level this replaces
         // the input pmat, which we must never destroy
         air_data->allocated_coarse_matrix[our_level] = PETSC_TRUE;
      }

      // ~~~~~~~~~~~~
      // Do processor agglomeration if desired
      // We stay on the existing communicator with some cores just having zero dofs
      // ~~~~~~~~~~~~
      if (air_data->options.processor_agglom) {

         // If we're in parallel and we haven't already agglomerated down to one processor
         if (comm_size != 1 && proc_stride != comm_size) {

            // Number of cores we have dofs on
            // Stolen from calculate_repartition, make sure they match!
            no_active_cores = (PetscInt)floor((double)comm_size/(double)proc_stride);
            // Be careful of rounding!
            if (no_active_cores == 0) no_active_cores = 1;
            ratio_local_nnzs_off_proc = 0.0;

            // If we have already setup our hierarchy, then we know what levels need to be repartitioned
            // If not, then we have to compute the ratio on each level to check if they need to be
            if (!air_data->allocated_matrices_A_ff[our_level]) {
               PetscCall(compute_mat_ratio_local_nonlocal_nnzs(air_data->coarse_matrix[our_level_coarse],
                        no_active_cores, &ratio_local_nnzs_off_proc));
            }

            // Let's check the size of the coarse matrix
            PetscCall(MatGetSize(air_data->coarse_matrix[our_level_coarse], &global_rows_repart, &global_cols_repart));

            // If we are reusing and we know we have to repartition this level, or
            // we have not very many unknowns per core (on average) or
            // if we get a local to off-processor ratio of less than processor_agglom_ratio
            trigger_proc_agglom = (air_data->reuse[our_level].reuse_is[IS_REPARTITION] ||
                        (global_rows_repart/no_active_cores < air_data->options.process_eq_limit
                           && no_active_cores != 1) ||
                        (ratio_local_nnzs_off_proc <= air_data->options.processor_agglom_ratio
                        && ratio_local_nnzs_off_proc != 0.0)) ? PETSC_TRUE : PETSC_FALSE;

            // Start the process agglomeration
            if (trigger_proc_agglom) {

               PetscCall(timer_start(PFLARE_TIMER_ID_AIR_PROC_AGGLOM));
               // air_data%options%processor_agglom_factor tells us how much we're reducing the number
               // of active mpi ranks by each time

               // can tell us how many idle threads we have on lower grids
               proc_stride = proc_stride * air_data->options.processor_agglom_factor;
               // Update the number of active cores to match the new stride
               // Stolen from calculate_repartition, make sure they match!
               no_active_cores = (PetscInt)floor((double)comm_size/(double)proc_stride);
               // Be careful of rounding!
               if (no_active_cores == 0) no_active_cores = 1;

               // If we don't have at least process_eq_limit unknowns per core (on average)
               // then we need to be more aggressive with our processor agglomeration
               // We'll just keep increasing the stride until we have more than process_eq_limit unknowns per core
               while (global_rows_repart < air_data->options.process_eq_limit * no_active_cores
                              && no_active_cores != 1) {
                  proc_stride = proc_stride * air_data->options.processor_agglom_factor;
                  // Stolen from calculate_repartition, make sure they match!
                  no_active_cores = (PetscInt)floor((double)comm_size/(double)proc_stride);
                  // Be careful of rounding!
                  if (no_active_cores == 0) no_active_cores = 1;
                  // If we can't agglomerate any more and we still haven't hit the desired
                  // process_eq_limit we'll just have to live with it
                  if (no_active_cores == 1) break;
               }

               // If we agglomerate down to one processor
               if (proc_stride > comm_size) proc_stride = comm_size;

               // Calculate the IS with the repartitioning if we haven't already
               // This is expensive as it calls the graph partitioner (e.g., parmetis)
               if (!air_data->reuse[our_level].reuse_is[IS_REPARTITION]) {

                  PetscCall(calculate_repartition(air_data->coarse_matrix[our_level_coarse],
                              proc_stride, &no_active_cores, PETSC_FALSE,
                              &air_data->reuse[our_level].reuse_is[IS_REPARTITION]));
               }

               if (air_data->options.print_stats_timings && comm_rank == 0) {
                  PetscCall(PetscPrintf(PETSC_COMM_SELF, " Doing processor agglomeration onto no cores: %" PetscInt_FMT "\n", no_active_cores));
               }

               // ~~~~~~~~~~~~~~~~~~
               // Repartition the coarse matrix
               // ~~~~~~~~~~~~~~~~~~
               if (air_data->reuse[our_level].reuse_mat[MAT_COARSE_REPARTITIONED]) {

                  PetscCall(MatCreateSubMatrix(air_data->coarse_matrix[our_level_coarse],
                              air_data->reuse[our_level].reuse_is[IS_REPARTITION],
                              air_data->reuse[our_level].reuse_is[IS_REPARTITION],
                              MAT_REUSE_MATRIX,
                              &air_data->reuse[our_level].reuse_mat[MAT_COARSE_REPARTITIONED]));
               } else {
                  PetscCall(MatCreateSubMatrix(air_data->coarse_matrix[our_level_coarse],
                              air_data->reuse[our_level].reuse_is[IS_REPARTITION],
                              air_data->reuse[our_level].reuse_is[IS_REPARTITION],
                              MAT_INITIAL_MATRIX,
                              &air_data->reuse[our_level].reuse_mat[MAT_COARSE_REPARTITIONED]));
               }

               PetscCall(MatDestroy(&air_data->coarse_matrix[our_level_coarse]));
               PetscCall(MatDuplicate(air_data->reuse[our_level].reuse_mat[MAT_COARSE_REPARTITIONED],
                        MAT_COPY_VALUES, &air_data->coarse_matrix[our_level_coarse]));

               // Delete temporary if not reusing
               if (!air_data->options.reuse_sparsity ||
                   !REUSE_MAT_ACTIVE[MAT_COARSE_REPARTITIONED][air_data->options.reuse_amount - 1]) {
                  PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_COARSE_REPARTITIONED]));
               }

               // Create an IS to represent the row indices of the prolongator
               // as we are only repartitioning the coarse matrix not the matrix on this level
               // ie the columns of the prolongator
               PetscCall(MatGetOwnershipRange(air_data->prolongators[our_level], &prolongator_start,
                        &prolongator_end_plus_one));
               PetscCall(ISCreateStride(MPI_COMM_MATRIX, prolongator_end_plus_one - prolongator_start,
                        prolongator_start, one, &is_unchanged));

               // ~~~~~~~~~~~~~~~~~~
               // Repartition the prolongator matrix
               // ~~~~~~~~~~~~~~~~~~
               if (air_data->reuse[our_level].reuse_mat[MAT_P_REPARTITIONED]) {

                  // If we've got a one point classical prolongator then we just use the existing repartitioned
                  // one so we don't need to repartition
                  if (!air_data->reuse_one_point_classical_prolong) {

                     PetscCall(MatCreateSubMatrix(air_data->prolongators[our_level],
                                 is_unchanged, air_data->reuse[our_level].reuse_is[IS_REPARTITION],
                                 MAT_REUSE_MATRIX,
                                 &air_data->reuse[our_level].reuse_mat[MAT_P_REPARTITIONED]));

                     PetscCall(MatDestroy(&air_data->prolongators[our_level]));
                     PetscCall(MatDuplicate(air_data->reuse[our_level].reuse_mat[MAT_P_REPARTITIONED],
                              MAT_COPY_VALUES, &air_data->prolongators[our_level]));
                  }
               } else {
                  PetscCall(MatCreateSubMatrix(air_data->prolongators[our_level],
                              is_unchanged, air_data->reuse[our_level].reuse_is[IS_REPARTITION],
                              MAT_INITIAL_MATRIX,
                              &air_data->reuse[our_level].reuse_mat[MAT_P_REPARTITIONED]));

                  PetscCall(MatDestroy(&air_data->prolongators[our_level]));
                  PetscCall(MatDuplicate(air_data->reuse[our_level].reuse_mat[MAT_P_REPARTITIONED],
                              MAT_COPY_VALUES, &air_data->prolongators[our_level]));
               }

               // Delete temporary if not reusing
               if (!air_data->options.reuse_sparsity ||
                   !REUSE_MAT_ACTIVE[MAT_P_REPARTITIONED][air_data->options.reuse_amount - 1]) {
                  PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_P_REPARTITIONED]));
               }

               // ~~~~~~~~~~~~~~~~~~
               // If need to repartition a restrictor
               // ~~~~~~~~~~~~~~~~~~
               if (!air_data->options.symmetric) {

                  // Repartition the restrictor matrix
                  if (air_data->reuse[our_level].reuse_mat[MAT_R_REPARTITIONED]) {

                     PetscCall(MatCreateSubMatrix(air_data->restrictors[our_level],
                                 air_data->reuse[our_level].reuse_is[IS_REPARTITION],
                                 is_unchanged, MAT_REUSE_MATRIX,
                                 &air_data->reuse[our_level].reuse_mat[MAT_R_REPARTITIONED]));
                  } else {
                     PetscCall(MatCreateSubMatrix(air_data->restrictors[our_level],
                                 air_data->reuse[our_level].reuse_is[IS_REPARTITION],
                                 is_unchanged, MAT_INITIAL_MATRIX,
                                 &air_data->reuse[our_level].reuse_mat[MAT_R_REPARTITIONED]));
                  }

                  PetscCall(MatDestroy(&air_data->restrictors[our_level]));
                  PetscCall(MatDuplicate(air_data->reuse[our_level].reuse_mat[MAT_R_REPARTITIONED],
                           MAT_COPY_VALUES, &air_data->restrictors[our_level]));

                  // Delete temporary if not reusing
                  if (!air_data->options.reuse_sparsity ||
                      !REUSE_MAT_ACTIVE[MAT_R_REPARTITIONED][air_data->options.reuse_amount - 1]) {
                     PetscCall(MatDestroy(&air_data->reuse[our_level].reuse_mat[MAT_R_REPARTITIONED]));
                  }
               }

               // ~~~~~~~~~~~~~~~~~~
               // If need to repartition the coarse right and left near-nullspace vectors
               // ~~~~~~~~~~~~~~~~~~
               if (air_data->options.constrain_z || air_data->options.constrain_w) {

                  PetscCall(MatGetSize(air_data->coarse_matrix[our_level_coarse],
                        &global_rows_repart, &global_cols_repart));
                  PetscCall(MatGetLocalSize(air_data->coarse_matrix[our_level_coarse],
                        &local_rows_repart, &local_cols_repart));
                  // Can't use matcreatevecs here on coarse_matrix(our_level_coarse)
                  // if the coarser matrix has gone down to one process it returns a serial vector
                  // but we have to have the same type for the scatter (ie mpi and mpi)
                  if (air_data->options.constrain_z) {
                     PetscCall(VecGetType(left_null_vecs[0], &vec_type));
                  } else if (air_data->options.constrain_w) {
                     PetscCall(VecGetType(right_null_vecs[0], &vec_type));
                  }
                  PetscCall(VecCreate(MPI_COMM_MATRIX, &temp_coarse_vec));
                  PetscCall(VecSetSizes(temp_coarse_vec, local_rows_repart, global_rows_repart));
                  PetscCall(VecSetType(temp_coarse_vec, vec_type));
                  PetscCall(VecSetUp(temp_coarse_vec));

                  // Can't seem to pass in PETSC_NULL_IS to the vecscattercreate in petsc 3.14
                  PetscCall(VecGetLocalSize(temp_coarse_vec, &local_vec_size));
                  PetscCall(VecGetOwnershipRange(temp_coarse_vec, &ystart, &yend));
                  PetscCall(ISCreateStride(PETSC_COMM_SELF, local_vec_size, ystart, one, &is_full));

                  if (air_data->options.constrain_z) {
                     PetscCall(VecScatterCreate(left_null_vecs[0],
                              air_data->reuse[our_level].reuse_is[IS_REPARTITION],
                              temp_coarse_vec, is_full, &vec_scatter));
                  } else if (air_data->options.constrain_w) {
                     PetscCall(VecScatterCreate(right_null_vecs[0],
                              air_data->reuse[our_level].reuse_is[IS_REPARTITION],
                              temp_coarse_vec, is_full, &vec_scatter));
                  }
               }

               // Could overlap the comms if we stored a copy of the number of vectors

               // Do the vec scatters for the left nullspace vecs
               if (air_data->options.constrain_z) {
                  for (i_loc = 0; i_loc < null_vecs_size; i_loc++) {
                     PetscCall(VecScatterBegin(vec_scatter, left_null_vecs[i_loc], temp_coarse_vec,
                                 INSERT_VALUES, SCATTER_FORWARD));
                     PetscCall(VecScatterEnd(vec_scatter, left_null_vecs[i_loc], temp_coarse_vec,
                                 INSERT_VALUES, SCATTER_FORWARD));
                     PetscCall(VecDestroy(&left_null_vecs[i_loc]));
                     PetscCall(VecDuplicate(temp_coarse_vec, &left_null_vecs[i_loc]));
                     PetscCall(VecCopy(temp_coarse_vec, left_null_vecs[i_loc]));
                  }
               }

               // Do the vec scatters for the right nullspace vecs
               if (air_data->options.constrain_w) {
                  for (i_loc = 0; i_loc < null_vecs_size; i_loc++) {
                     PetscCall(VecScatterBegin(vec_scatter, right_null_vecs[i_loc], temp_coarse_vec,
                                 INSERT_VALUES, SCATTER_FORWARD));
                     PetscCall(VecScatterEnd(vec_scatter, right_null_vecs[i_loc], temp_coarse_vec,
                                 INSERT_VALUES, SCATTER_FORWARD));
                     PetscCall(VecDestroy(&right_null_vecs[i_loc]));
                     PetscCall(VecDuplicate(temp_coarse_vec, &right_null_vecs[i_loc]));
                     PetscCall(VecCopy(temp_coarse_vec, right_null_vecs[i_loc]));
                  }
               }

               if (air_data->options.constrain_z || air_data->options.constrain_w) {
                  PetscCall(VecDestroy(&temp_coarse_vec));
                  PetscCall(VecScatterDestroy(&vec_scatter));
                  PetscCall(ISDestroy(&is_full));
               }

               // ~~~~~~~~~~~~~~~~~~

               PetscCall(ISDestroy(&is_unchanged));
               // Delete temporary if not reusing
               if (!air_data->options.reuse_sparsity ||
                   !REUSE_IS_ACTIVE[IS_REPARTITION][air_data->options.reuse_amount - 1]) {
                  PetscCall(ISDestroy(&air_data->reuse[our_level].reuse_is[IS_REPARTITION]));
               }

               PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_PROC_AGGLOM));

            }
         }
      }

      // ~~~~~~~~~~~~~~

      if (air_data->options.any_c_smooths &&
               !air_data->options.full_smoothing_up_and_down) {
         air_data->allocated_matrices_A_cc[our_level] = PETSC_TRUE;
      }

      // ~~~~~~~~~~~~

      // ~~~~~~~~~~~~
      // Output some timing results
      // ~~~~~~~~~~~~

      if (air_data->options.print_stats_timings && comm_rank == 0) PetscCall(print_timers());

   }

   // Destroy our copies of the near nullspace vectors
   // This has to happen after the loop, as we can leave it either by
   // coarsening far enough or by hitting the max number of levels
   if (air_data->options.constrain_z) {
      for (i_loc = 0; i_loc < null_vecs_size; i_loc++) {
         PetscCall(VecDestroy(&left_null_vecs[i_loc]));
      }
   }
   PetscCall(PetscFree(left_null_vecs));
   PetscCall(PetscFree(left_null_vecs_c));
   if (air_data->options.constrain_w) {
      for (i_loc = 0; i_loc < null_vecs_size; i_loc++) {
         PetscCall(VecDestroy(&right_null_vecs[i_loc]));
      }
   }
   PetscCall(PetscFree(right_null_vecs));
   PetscCall(PetscFree(right_null_vecs_c));

   // Record how many levels we have
   air_data->no_levels = no_levels;

   // ~~~~~~~~~~~
   // We can now start the comms for the coarse grid solver
   // ~~~~~~~~~~~

   PetscCall(MatGetSize(air_data->coarse_matrix[no_levels - 1], &global_rows, &global_cols));

   // Set up a GMRES polynomial inverse data for the coarse grid solve
   PetscCall(setup_gmres_poly_data(global_rows,
            air_data->options.coarsest_inverse_type,
            (int)air_data->options.coarsest_poly_order,
            (int)air_data->options.coarsest_inverse_sparsity_order,
            air_data->options.coarsest_subcomm,
            proc_stride,
            &air_data->inv_coarsest_poly_data));

   // Start the inverse for the coarse grid
   // These comms will be finished below
   if (!(air_data->inv_A_ff[air_data->no_levels - 1]
               && air_data->options.reuse_poly_coeffs)) {

      // We've already created our coarse solver if we've auto truncated
      // With only one level and no auto truncation we fall back to a jacobi PC
      // below and never call finish_approximate_inverse, so don't start one -
      // start_approximate_inverse takes a reference to the matrix that only
      // finish_approximate_inverse gives back
      if (!auto_truncated && no_levels > 1) {
         PetscCall(timer_start(PFLARE_TIMER_ID_AIR_INVERSE));

         PetscCall(poly_data_buffers(&air_data->inv_coarsest_poly_data, &buffers));
         PetscCall(start_approximate_inverse(air_data->coarse_matrix[no_levels - 1],
               air_data->inv_coarsest_poly_data.inverse_type,
               air_data->inv_coarsest_poly_data.gmres_poly_order,
               air_data->options.coarsest_diag_scale_polys,
               buffers, &air_data->inv_coarsest_poly_data.coefficients,
               &air_data->inv_coarsest_poly_data.coefficients_rows,
               &air_data->inv_coarsest_poly_data.coefficients_cols));
         PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_INVERSE));
      }
   }

   // ~~~~~~~~~~~~~~~
   // Let's setup the PETSc pc we need
   // ~~~~~~~~~~~~~~~
   if (no_levels > 1) {

      PetscCall(PCSetOperators(pcmg_input, amat, pmat));
      PetscCall(PCSetType(pcmg_input, PCMG));
      no_levels_petsc_int = no_levels;
      PetscCall(PCMGSetLevels(pcmg_input, no_levels_petsc_int, NULL));

      // If we're doing fc smoothing always have zero down smooths, which petsc calls kaskade
      // This prevents any unnecessary residual calculations on the way down
      // Annoyingly kaskade calls the "down" smoother on the way up, so
      // we have to set the options carefully for the down smoother
      if (!air_data->options.full_smoothing_up_and_down) {
         PetscCall(PCMGSetType(pcmg_input, PC_MG_KASKADE));
      }

      // PETSc MG levels work in the opposite order to those in our code. If there are N levels:
      //           PETSc   our_level (the 0-based C level is our_level - 1)
      // Fine     N - 1      1
      //          N - 2      2
      //            .        .
      //            1      N - 1
      // Coarse     0        N

      // Therefore  ---  petsc_level = N - our_level (N - 1 - our_level with the 0-based level)

      // Set up the petsc objects on each level
      // Loop over all levels except the bottom
      for (petsc_level = no_levels_petsc_int - 1; petsc_level >= 1; petsc_level--) {

         // Level is reverse ordering
         our_level = no_levels - 1 - (int)petsc_level;
         our_level_coarse = our_level + 1;

         // Set the restrictor/prolongator
         if (!air_data->options.symmetric) {
            // If restrictor is not set petsc will use transpose of prolongator
            PetscCall(PCMGSetRestriction(pcmg_input, petsc_level, air_data->restrictors[our_level]));
         }
         PetscCall(PCMGSetInterpolation(pcmg_input, petsc_level, air_data->prolongators[our_level]));

         // Get smoother for this level
         // Kaskade only ever calls the down smoother, so don't create a distinct up smoother
         // (and its work vectors) when we're fc smoothing - PCMGGetSmootherDown would
         // When fc smoothing the up and down smoothers are then the same ksp, which
         // we set up below as a richardson so that petsc doesn't default to chebychev
         // and hence try and calculate eigenvalues on each grid
         if (air_data->options.full_smoothing_up_and_down) {
            PetscCall(PCMGGetSmootherUp(pcmg_input, petsc_level, &ksp_smoother_up));
            PetscCall(PCMGGetSmootherDown(pcmg_input, petsc_level, &ksp_smoother_down));
         } else {
            PetscCall(PCMGGetSmoother(pcmg_input, petsc_level, &ksp_smoother_down));
            ksp_smoother_up = ksp_smoother_down;
         }

         // Set the operators for smoothing on this level
         if (!air_data->options.full_smoothing_up_and_down) {
            PetscCall(KSPSetOperators(ksp_smoother_up, air_data->coarse_matrix[our_level],
                        air_data->coarse_matrix[our_level]));
            PetscCall(KSPSetOperators(ksp_smoother_down, air_data->coarse_matrix[our_level],
                        air_data->coarse_matrix[our_level]));

            // The transposed cycle computes r = b - A^T x on each level, and the
            // default does a MatMultTranspose with the level operator, which is
            // only an empty placeholder matshell when we're F-C smoothing, so give
            // petsc our own that uses the blocks of A instead
            // Full smoothing up and down has real level matrices and uses the default
            PetscCall(PCMGSetResidualTranspose(pcmg_input, petsc_level, mg_FC_point_residual_transpose,
                     air_data->coarse_matrix[our_level]));
         // The smoother for all the unknowns is stored in inv_A_ff
         } else {
            PetscCall(KSPSetOperators(ksp_smoother_up, air_data->coarse_matrix[our_level],
                        air_data->inv_A_ff[our_level]));
            PetscCall(KSPSetOperators(ksp_smoother_down, air_data->coarse_matrix[our_level],
                        air_data->inv_A_ff[our_level]));
         }

         // Set no norm
         PetscCall(KSPSetNormType(ksp_smoother_up, KSP_NORM_NONE));
         PetscCall(KSPSetNormType(ksp_smoother_down, KSP_NORM_NONE));

         // Now here is where we have to be careful as we are calling kaskade mg type
         // It uses the down smoother as the up smoother (with no calls to the up)
         // And on the way up we have a nonzero initial guess
         if (!air_data->options.full_smoothing_up_and_down) {
            PetscCall(KSPSetInitialGuessNonzero(ksp_smoother_down, PETSC_TRUE));
         }

         // Get the PC for each smoother
         PetscCall(KSPGetPC(ksp_smoother_up, &pc_smoother_up));
         PetscCall(KSPGetPC(ksp_smoother_down, &pc_smoother_down));

         // Set the smoother
         if (!air_data->options.full_smoothing_up_and_down) {
            PetscCall(PCSetType(pc_smoother_up, PCSHELL));
            PetscCall(PCSetType(pc_smoother_down, PCSHELL));
         } else {
            // When doing full smoothing the smoother is just the application of
            // inv_A_ff, which is what PCMAT does with inv_A_ff as its Pmat
            // We can't use PCMAT with a matrix-free inverse though, as its
            // multiple rhs apply does a MatMatMult on the Pmat and the matshell
            // only has MATOP_MULT and MATOP_MULT_TRANSPOSE - use a shell that
            // does the same single rhs
            // arithmetic and knows how to do the block apply instead
            PetscCall(MatGetType(air_data->inv_A_ff[our_level], &mat_type_inv_aff));
            PetscCall(PetscStrcmp(mat_type_inv_aff, MATSHELL, &is_shell));
            if (is_shell) {
               PetscCall(PCSetType(pc_smoother_up, PCSHELL));
               PetscCall(PCSetType(pc_smoother_down, PCSHELL));
               PetscCall(PCShellSetApply(pc_smoother_up, mg_smooth_shell_apply));
               PetscCall(PCShellSetApply(pc_smoother_down, mg_smooth_shell_apply));
               PetscCall(PCShellSetMatApply(pc_smoother_up, mg_smooth_shell_block_apply));
               PetscCall(PCShellSetMatApply(pc_smoother_down, mg_smooth_shell_block_apply));
               // PCMAT has a transposed apply, so a shell standing in for it needs
               // one too or a transposed v-cycle can't drive these smoothers
               PetscCall(PCShellSetApplyTranspose(pc_smoother_up, mg_smooth_shell_apply_transpose));
               PetscCall(PCShellSetApplyTranspose(pc_smoother_down, mg_smooth_shell_apply_transpose));
            } else {
               PetscCall(PCSetType(pc_smoother_up, PCMAT));
               PetscCall(PCSetType(pc_smoother_down, PCMAT));
               // We want PCApply to be a matmult with inv_A_ff, as inv_A_ff is
               // already an approximate inverse. Since petsc 3.21 PCSetUp_Mat
               // otherwise defaults to MatSolve whenever the Pmat has one, and
               // among the matrix types we build matdiagonal does (unfactored
               // aij does not), which would smooth by multiplying with the
               // inverse of our approximate inverse
               PetscCall(PCMatSetApplyOperation(pc_smoother_up, MATOP_MULT));
               PetscCall(PCMatSetApplyOperation(pc_smoother_down, MATOP_MULT));
            }
         }

         // Set richardson
         PetscCall(KSPSetType(ksp_smoother_up, KSPRICHARDSON));
         PetscCall(KSPSetType(ksp_smoother_down, KSPRICHARDSON));

         // We're overwriting the richardson for the fc smoothing
         // This is automatically disabled if you run with -mg_levels_ksp_monitor fyi!
         if (!air_data->options.full_smoothing_up_and_down) {
            PetscCall(PCShellSetApplyRichardson(pc_smoother_up, mg_FC_point_richardson));
            PetscCall(PCShellSetApplyRichardson(pc_smoother_down, mg_FC_point_richardson));
            // The multiple rhs version - without this a KSPMatSolve would fall
            // back to calling the single rhs richardson on each column in turn
            PetscCall(PCShellSetMatApplyRichardson(pc_smoother_up, mg_FC_block_richardson));
            PetscCall(PCShellSetMatApplyRichardson(pc_smoother_down, mg_FC_block_richardson));
            // A KSPSolveTranspose skips the richardson fast path above, so the
            // transposed cycle drives the smoother through this instead
            PetscCall(PCShellSetApplyTranspose(pc_smoother_up, mg_FC_point_apply_transpose));
            PetscCall(PCShellSetApplyTranspose(pc_smoother_down, mg_FC_point_apply_transpose));
         }

         // Zero up smooths for kaskade
         if (!air_data->options.full_smoothing_up_and_down) {
            // This is never called anyway with kaskade
            PetscCall(KSPSetTolerances(ksp_smoother_up, PFLARE_KSP_ATOL_SMOOTH,
                  PFLARE_KSP_ATOL_SMOOTH,
                  PETSC_CURRENT,
                  zero));
         } else {
            PetscCall(KSPSetTolerances(ksp_smoother_up, PFLARE_KSP_ATOL_SMOOTH,
                  PFLARE_KSP_ATOL_SMOOTH,
                  PETSC_CURRENT,
                  one));
         }

         // One up smooth (but we have to set it as the down given kaskade)
         PetscCall(KSPSetTolerances(ksp_smoother_down, PFLARE_KSP_ATOL_SMOOTH,
               PFLARE_KSP_ATOL_SMOOTH,
               PETSC_CURRENT,
               one));

         // Set up the smoothers on this level
         PetscCall(PCSetUp(pc_smoother_up));
         PetscCall(PCSetUp(pc_smoother_down));
         PetscCall(KSPSetUp(ksp_smoother_up));
         PetscCall(KSPSetUp(ksp_smoother_down));

      }

      // ~~~~~~~~~~~
      // Then at the bottom of the PCMG we need to build the coarse grid solve
      // ~~~~~~~~~~~

      PetscCall(PCMGGetCoarseSolve(pcmg_input, &ksp_coarse_solver));
      // If you want to apply more iterations of the coarse solver, change this to
      // a richardson (can do via command line -mg_coarse_ksp_type richardson)
      PetscCall(KSPSetType(ksp_coarse_solver, KSPPREONLY));
      // Apply one iteration of the coarse solver by default
      PetscCall(KSPSetTolerances(ksp_coarse_solver, PFLARE_KSP_RTOL_COARSE, PFLARE_KSP_ATOL_COARSE, PETSC_CURRENT, one));

      // Set no norm
      PetscCall(KSPSetNormType(ksp_coarse_solver, KSP_NORM_NONE));
      PetscCall(KSPGetPC(ksp_coarse_solver, &pc_coarse_solver));

      // ~~~~~~~~~~~~~~~
      // Finish the comms for the coarse grid solver
      // ~~~~~~~~~~~~~~~

      // We've already created our coarse solver if we've auto truncated
      if (!auto_truncated) {

         // Coarse grid polynomial coefficients
         PetscCall(timer_start(PFLARE_TIMER_ID_AIR_INVERSE));

         PetscCall(poly_data_buffers(&air_data->inv_coarsest_poly_data, &buffers));
         PetscCall(finish_approximate_inverse(air_data->coarse_matrix[no_levels - 1],
               air_data->inv_coarsest_poly_data.inverse_type,
               air_data->inv_coarsest_poly_data.gmres_poly_order,
               air_data->inv_coarsest_poly_data.gmres_poly_sparsity_order,
               buffers, air_data->inv_coarsest_poly_data.coefficients,
               air_data->inv_coarsest_poly_data.coefficients_rows,
               air_data->inv_coarsest_poly_data.coefficients_cols,
               air_data->options.coarsest_matrix_free_polys,
               air_data->options.coarsest_diag_scale_polys,
               &air_data->reuse[air_data->no_levels - 1].reuse_mat[MAT_INV_AFF],
               &air_data->reuse[air_data->no_levels - 1].reuse_submatrices[MAT_INV_AFF].array,
               &air_data->inv_A_ff[air_data->no_levels - 1]));

         // Delete temporary if not reusing
         if (!air_data->options.reuse_sparsity ||
             !REUSE_MAT_ACTIVE[MAT_INV_AFF][air_data->options.reuse_amount - 1]) {
            PetscCall(destroy_matrix_reuse(&air_data->reuse[air_data->no_levels - 1].reuse_mat[MAT_INV_AFF],
                     &air_data->reuse[air_data->no_levels - 1].reuse_submatrices[MAT_INV_AFF].array));
         }

         // Now we've finished the coarse grid solver, output the time
         PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_INVERSE));
      }

      // Use the mf coarse grid solver or not
      // Let's store the coarse grid solver in inv_A_ff(no_levels)
      if (!air_data->options.coarsest_matrix_free_polys) {
         if (air_data->options.print_stats_timings) {
            PetscCall(get_nnzs_petsc_sparse(air_data->inv_A_ff[air_data->no_levels - 1],
                     &air_data->inv_A_ff_nnzs[air_data->no_levels - 1]));
         }
      }

      // ~~~~~~~~~~~~
      // Set our coarse grid solver
      // ~~~~~~~~~~~~
      PetscCall(KSPSetOperators(ksp_coarse_solver, air_data->coarse_matrix[no_levels - 1],
                  air_data->coarse_matrix[no_levels - 1]));
      PetscCall(PCSetType(pc_coarse_solver, PCSHELL));
      PetscCall(PCShellSetContext(pc_coarse_solver, air_data));
      PetscCall(PCShellSetApply(pc_coarse_solver, mg_coarse_shell_apply));
      // The multiple rhs version - the coarse ksp is a preonly, which does have
      // a matsolve, so this is what makes the coarse solve a real block apply
      PetscCall(PCShellSetMatApply(pc_coarse_solver, mg_coarse_shell_block_apply));
      // The transposed apply - this is what a KSPSolveTranspose on the coarse ksp
      // ends up calling in our transposed cycle
      PetscCall(PCShellSetApplyTranspose(pc_coarse_solver, mg_coarse_shell_apply_transpose));
      PetscCall(PCSetUp(pc_coarse_solver));
      PetscCall(KSPSetUp(ksp_coarse_solver));

   // If we've only got one level
   } else {
      // Precondition with the "coarse grid" solver we used to determine auto truncation
      if (auto_truncated) {
         // PCSetOperators takes its own references to both matrices and drops
         // them in PCReset/PCDestroy, so don't take one here
         PetscCall(PCSetOperators(pcmg_input, amat,
                     air_data->inv_A_ff[no_levels - 1]));
         PetscCall(PCSetType(pcmg_input, PCMAT));
         // Same as for the level smoothers above - the Pmat here is already an
         // approximate inverse so PCApply has to be a matmult with it, not the
         // MatSolve petsc defaults to when the Pmat has one
         PetscCall(PCMatSetApplyOperation(pcmg_input, MATOP_MULT));

      // Otherwise just do a jacobi and tell the user
      } else {

         // If we've only got one level just precondition with jacobi
         PetscCall(PCSetOperators(pcmg_input, amat, pmat));
         PetscCall(PCSetType(pcmg_input, PCJACOBI));
         if (comm_rank == 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, " Only a single level, defaulting to Jacobi PC\n"));
      }
   }

   // Call the setup on our PC
   PetscCall(PCSetUp(pcmg_input));

   // ~~~~~~~~~
   // Build the temporary data we use during smoothing
   // ~~~~~~~~~
   PetscCall(create_smooth_temp_vecs(air_data));
   for (our_level = 0; our_level < air_data->no_levels - 1; our_level++) {
      air_data->allocated_matrices_A_ff[our_level] = PETSC_TRUE;

   }

   // ~~~~~~~~~
   // We are done, print out info
   // ~~~~~~~~~

   PetscCall(timer_finish(PFLARE_TIMER_ID_AIR_SETUP));
   // Print out the coarse grid info
   // The level printed is the 1-based level the user sees, ie no_levels
   if (air_data->options.print_stats_timings && comm_rank == 0) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, " ~~~~~~~~~~~~ Coarse grid %d\n", no_levels));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, " Global rows %" PetscInt_FMT "\n", global_rows));
      PetscCall(print_timers());
      PetscCall(PetscPrintf(PETSC_COMM_SELF, " ~~~~~~~~~~~~ \n"));
      PetscCall(timer_time(PFLARE_TIMER_ID_AIR_SETUP, &time));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, " Total cumulative setup time : %.15g\n", (double)time));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, " ~~~~~~~~~~~~ \n"));
   }
   // Print out stats on the hierarchy - collective so make sure to call
   // this on all ranks
   if (air_data->options.print_stats_timings) PetscCall(print_stats(air_data, pcmg_input));

   PetscFunctionReturn(PETSC_SUCCESS);
}
