/*
  C port of AIR_MG_Stats.F90 (module air_mg_stats)

  The levels are 0-based (the Fortran our_level is our_level + 1, and the
  Fortran no_levels index is no_levels - 1 here). The Fortran integer(kind=8)
  counts are PetscInt64
  print_stats prints with explicit formats rather than the list-directed print
  of the Fortran, keeping the labels tools/parse_pflare_output.py reads
  (convention 8 of docs/dev/fortran_removal_plan.md)
*/

#include <petscksp.h>
#include "pflareimpl.h"
#include "PETSc_Helper.h"
#include "Gmres_Poly.h"
#include "AIR_MG_Stats.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode compute_nnzs_air_matrices(air_multigrid_data *air_data)
{
   // Compute the nnzs of the matrices in air

   // ~~~~~~
   int our_level;
   MatType mat_type;
   PetscBool is_shell;
   // ~~~~~~

   PetscFunctionBegin;

   // The coarse solver nnzs
   // With a single level the "coarse" matrix is the top grid matrix and we
   // only have a coarse solver (in inv_A_ff) if we auto truncated, otherwise
   // we used jacobi
   if (air_data->inv_A_ff[air_data->no_levels - 1]) {
      // Are we mf?
      PetscCall(MatGetType(air_data->inv_A_ff[air_data->no_levels - 1], &mat_type));
      PetscCall(PetscStrcmp(mat_type, MATSHELL, &is_shell));
      if (!is_shell) {
         PetscCall(get_nnzs_petsc_sparse(air_data->inv_A_ff[air_data->no_levels - 1], &air_data->inv_A_ff_nnzs[air_data->no_levels - 1]));
      }
   }
   PetscCall(get_nnzs_petsc_sparse(air_data->coarse_matrix[air_data->no_levels - 1], &air_data->coarse_matrix_nnzs[air_data->no_levels - 1]));

   // Then go over all the levels except the coarse
   for (our_level = 0; our_level < air_data->no_levels - 1; our_level++) {

      // The fine/coarse components of our matrix
      PetscCall(get_nnzs_petsc_sparse(air_data->A_ff[our_level], &air_data->A_ff_nnzs[our_level]));
      PetscCall(get_nnzs_petsc_sparse(air_data->A_fc[our_level], &air_data->A_fc_nnzs[our_level]));

      // Are we mf?
      PetscCall(MatGetType(air_data->inv_A_ff[our_level], &mat_type));
      PetscCall(PetscStrcmp(mat_type, MATSHELL, &is_shell));
      if (!is_shell) {
         PetscCall(get_nnzs_petsc_sparse(air_data->inv_A_ff[our_level], &air_data->inv_A_ff_nnzs[our_level]));
      }
      // Are we doing C point smoothing
      if (air_data->options.any_c_smooths &&
               !air_data->options.full_smoothing_up_and_down) {

         PetscCall(get_nnzs_petsc_sparse(air_data->A_cc[our_level], &air_data->A_cc_nnzs[our_level]));
         PetscCall(get_nnzs_petsc_sparse(air_data->A_cf[our_level], &air_data->A_cf_nnzs[our_level]));

         // Are we mf?
         PetscCall(MatGetType(air_data->inv_A_cc[our_level], &mat_type));
         PetscCall(PetscStrcmp(mat_type, MATSHELL, &is_shell));
         if (!is_shell) {
            PetscCall(get_nnzs_petsc_sparse(air_data->inv_A_cc[our_level], &air_data->inv_A_cc_nnzs[our_level]));
         }
      }

      // Add in the nnzs from the restrictor
      if (!air_data->options.symmetric) {
         PetscCall(get_nnzs_petsc_sparse(air_data->restrictors[our_level], &air_data->restrictor_nnzs[our_level]));
      }
      // Add in the nnzs from the prolongator
      PetscCall(get_nnzs_petsc_sparse(air_data->prolongators[our_level], &air_data->prolongator_nnzs[our_level]));

   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode compute_nnzs_v_cycle_air(air_multigrid_data *air_data, PC pcmg_input, PetscInt64 *nnzs)
{
   // Compute the nnzs of a V cycle of air
   // Have to input both the air data and the pcmg that comes from setup_air_pcmg

   // ~~~~~~
   int our_level, non_zero_order, i;
   PetscInt maxits, petsc_level, global_rows, global_cols;
   PCType pc_type;
   KSP ksp;
   PetscReal rtol, atol, dtol;
   PetscInt64 maxits_long, maxits_aff_long, gmres_size_long;
   PetscInt64 maxits_afc_long, maxits_acf_long, maxits_acc_long;
   MatType mat_type;
   PetscBool is_shell, is_mat, is_jacobi, is_mg;
   // ~~~~~~

   PetscFunctionBegin;

   // Set to zero to start
   *nnzs = 0;

   PetscCall(PCGetType(pcmg_input, &pc_type));
   PetscCall(PetscStrcmp(pc_type, PCMAT, &is_mat));
   PetscCall(PetscStrcmp(pc_type, PCJACOBI, &is_jacobi));
   PetscCall(PetscStrcmp(pc_type, PCMG, &is_mg));

   // If we only have a single level we either auto truncated and our PC
   // is a single application of the coarse grid solver, or we are doing jacobi
   if (air_data->no_levels == 1) {
      if (is_mat) {
         // Are we mf?
         PetscCall(MatGetType(air_data->inv_A_ff[0], &mat_type));
         PetscCall(PetscStrcmp(mat_type, MATSHELL, &is_shell));
         if (is_shell) {
            PetscCall(compute_mf_gmres_poly_num_matvecs(air_data->options.coarsest_inverse_type,
                        air_data->inv_coarsest_poly_data.coefficients,
                        air_data->inv_coarsest_poly_data.coefficients_rows,
                        air_data->inv_coarsest_poly_data.coefficients_cols,
                        &non_zero_order));
            *nnzs = (PetscInt64)non_zero_order * air_data->coarse_matrix_nnzs[0];
         } else {
            *nnzs = air_data->inv_A_ff_nnzs[0];
         }
      } else if (is_jacobi) {
         // Jacobi is a diagonal scaling, so one entry per row
         PetscCall(MatGetSize(air_data->coarse_matrix[0], &global_rows, &global_cols));
         *nnzs = (PetscInt64)global_rows;
      }
      PetscFunctionReturn(PETSC_SUCCESS);
   }

   // Check we're doing mg
   if (!is_mg) PetscFunctionReturn(PETSC_SUCCESS);

   // Coarse grid solve
   PetscCall(PCMGGetCoarseSolve(pcmg_input, &ksp));
   PetscCall(KSPGetTolerances(ksp, &rtol, &atol, &dtol, &maxits));
   // Round to long
   maxits_long = (PetscInt64)maxits;

   // How many matvecs do we need to apply our coarse grid polynomial inverse
   // Normally this is just the size of air_data%inv_coarsest_poly_data%coefficients, but for newton
   // we may have zero roots that we skip
   PetscCall(compute_mf_gmres_poly_num_matvecs(air_data->options.coarsest_inverse_type,
               air_data->inv_coarsest_poly_data.coefficients,
               air_data->inv_coarsest_poly_data.coefficients_rows,
               air_data->inv_coarsest_poly_data.coefficients_cols,
               &non_zero_order));

   gmres_size_long = (PetscInt64)non_zero_order;

   // Application of the coarse matrix inverse with richardson
   // No need for an initial matvec here with the coarse matrix as we have just restricted the residual (which is just b) straight down here
   // and the initial guess on the coarse grid is zero

   // Are we mf?
   PetscCall(MatGetType(air_data->inv_A_ff[air_data->no_levels - 1], &mat_type));
   PetscCall(PetscStrcmp(mat_type, MATSHELL, &is_shell));
   if (is_shell) {
      *nnzs = maxits_long * gmres_size_long * air_data->coarse_matrix_nnzs[air_data->no_levels - 1] +
               (maxits_long-1) * air_data->coarse_matrix_nnzs[air_data->no_levels - 1];
   } else {
      *nnzs = maxits_long * air_data->inv_A_ff_nnzs[air_data->no_levels - 1] +
               (maxits_long-1) * air_data->coarse_matrix_nnzs[air_data->no_levels - 1];
   }

   // Then go over all the levels except the coarse
   for (our_level = 0; our_level < air_data->no_levels - 1; our_level++) {

      maxits_aff_long = 0;
      maxits_acc_long = 0;
      maxits_afc_long = 0;
      maxits_acf_long = 0;
      // Round to long
      for (i = 0; i < air_data->smooth_order_levels[our_level].size; i++) {
         if (air_data->smooth_order_levels[our_level].array[i] > 0) {
            maxits_aff_long = maxits_aff_long + (PetscInt64)air_data->smooth_order_levels[our_level].array[i];
         } else {
            maxits_acc_long = maxits_acc_long + (PetscInt64)PetscAbsInt(air_data->smooth_order_levels[our_level].array[i]);
         }
         // count(array > 0) and count(array < 0)
         if (air_data->smooth_order_levels[our_level].array[i] > 0) maxits_afc_long = maxits_afc_long + 1;
         if (air_data->smooth_order_levels[our_level].array[i] < 0) maxits_acf_long = maxits_acf_long + 1;
      }

      // ~~~~~~~~~~~~~
      // No down smooths
      // ~~~~~~~~~~~~~

      // ~~~~~~~~~~~~~
      // Up F point smooths
      // ~~~~~~~~~~~~~
      // Make sure to use the petsc level
      // Now being careful here to actually pull out the down smoother from petsc
      // as the kaskade cycle only calls the "down" smoother on the way up
      // PCMGGetSmoother returns the down smoother without creating a distinct
      // up smoother, which PCMGGetSmootherDown would
      // The petsc level of our 0-based level (petsc numbers the levels from the coarsest)
      petsc_level = air_data->no_levels - 1 - our_level;
      PetscCall(PCMGGetSmoother(pcmg_input, petsc_level, &ksp));
      // Get the number of F&C up smooths
      PetscCall(KSPGetTolerances(ksp, &rtol, &atol, &dtol, &maxits));

      // Round to long
      maxits_long = maxits;

      // on the way up, we are solving the same equation A e = r, but this time the error is non-zero, so our richardson's will
      // have to do an initial F-point residual. The F-point smooths then proceed as normal, with the rhs being the residual from when we came down
      // and we don't have to do a post residual

      // Are we mf?
      PetscCall(MatGetType(air_data->inv_A_ff[our_level], &mat_type));
      PetscCall(PetscStrcmp(mat_type, MATSHELL, &is_shell));

      if (!air_data->options.full_smoothing_up_and_down) {

         // MF polynomial order may be different on each level
         PetscCall(compute_mf_gmres_poly_num_matvecs(air_data->inv_A_ff_poly_data[our_level].inverse_type,
                     air_data->inv_A_ff_poly_data[our_level].coefficients,
                     air_data->inv_A_ff_poly_data[our_level].coefficients_rows,
                     air_data->inv_A_ff_poly_data[our_level].coefficients_cols,
                     &non_zero_order));
         gmres_size_long = (PetscInt64)non_zero_order;

         if (is_shell) {
            *nnzs = *nnzs + maxits_long * maxits_aff_long * gmres_size_long * air_data->A_ff_nnzs[our_level] +
                        maxits_long * maxits_aff_long * air_data->A_ff_nnzs[our_level];
         } else {
            *nnzs = *nnzs + maxits_long * maxits_aff_long * air_data->inv_A_ff_nnzs[our_level] +
                        maxits_long * maxits_aff_long * air_data->A_ff_nnzs[our_level];
         }

         // Add in the minus Afc - this is done once before we consecutively F-point smooth
         *nnzs = *nnzs + maxits_long * maxits_afc_long * air_data->A_fc_nnzs[our_level];

         // One C-point smooth
         if (air_data->options.any_c_smooths) {

            // MF polynomial order may be different on each level
            PetscCall(compute_mf_gmres_poly_num_matvecs(air_data->inv_A_cc_poly_data[our_level].inverse_type,
                        air_data->inv_A_cc_poly_data[our_level].coefficients,
                        air_data->inv_A_cc_poly_data[our_level].coefficients_rows,
                        air_data->inv_A_cc_poly_data[our_level].coefficients_cols,
                        &non_zero_order));
            gmres_size_long = (PetscInt64)non_zero_order;

            // Are we mf?
            PetscCall(MatGetType(air_data->inv_A_cc[our_level], &mat_type));
            PetscCall(PetscStrcmp(mat_type, MATSHELL, &is_shell));
            if (is_shell) {
               *nnzs = *nnzs + maxits_long * maxits_acc_long * gmres_size_long * air_data->A_cc_nnzs[our_level] +
                        maxits_long * maxits_acc_long * air_data->A_cc_nnzs[our_level];
            } else {
               *nnzs = *nnzs + maxits_long * maxits_acc_long * air_data->inv_A_cc_nnzs[our_level] +
                        maxits_long * maxits_acc_long * air_data->A_cc_nnzs[our_level];
            }
            // Add in the minus Acf  - this is done once before we consecutively C-point smooth
            *nnzs = *nnzs + maxits_long * maxits_acf_long * air_data->A_cf_nnzs[our_level];
         }

      // Full smoothing up and down
      } else {

         // Symmetric GS and up and down
         //nnzs = nnzs + maxits_long * air_data%coarse_matrix_nnzs(our_level) * 2 * 2

         // MF polynomial order may be different on each level
         PetscCall(compute_mf_gmres_poly_num_matvecs(air_data->inv_A_ff_poly_data[our_level].inverse_type,
                     air_data->inv_A_ff_poly_data[our_level].coefficients,
                     air_data->inv_A_ff_poly_data[our_level].coefficients_rows,
                     air_data->inv_A_ff_poly_data[our_level].coefficients_cols,
                     &non_zero_order));
         gmres_size_long = (PetscInt64)non_zero_order;

         // The two is because we do up and down smoothing
         if (is_shell) {
            *nnzs = *nnzs + 2 * maxits_long * gmres_size_long * air_data->coarse_matrix_nnzs[our_level] +
                        maxits_long  * air_data->coarse_matrix_nnzs[our_level];
         } else {
            *nnzs = *nnzs + 2 * maxits_long * air_data->inv_A_ff_nnzs[our_level] +
                        maxits_long * air_data->coarse_matrix_nnzs[our_level];
         }
      }

      // Restrictor
      if (air_data->options.symmetric) {
         *nnzs = *nnzs + air_data->prolongator_nnzs[our_level];
      } else {
         *nnzs = *nnzs + air_data->restrictor_nnzs[our_level];
      }
      // Prolongator
      *nnzs = *nnzs + air_data->prolongator_nnzs[our_level];
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode compute_stats(air_multigrid_data *air_data, PC pcmg_input, PetscReal *grid_complx, PetscReal *op_complx, PetscReal *cycle_complx,
   PetscReal *storage_complx, PetscReal *reuse_storage_complx)
{
   // Compute the different stats for air
   // Have to input both the air data and the pcmg that comes from setup_air_pcmg

   // ~~~~~~
   int our_level, i_loc;
   PetscInt maxits_coarse;
   PetscInt global_rows, global_cols;
   KSP ksp;
   PetscReal rtol, atol, dtol;
   PetscInt64 nnzs_air_v, mat_storage_nnzs, op_complx_nnzs, mat_reuse_storage_nnzs, mat_nnzs;
   PCType pc_type;
   PetscBool is_mat, is_jacobi;
   // ~~~~~~

   PetscFunctionBegin;

   // Compute the nnzs in each matrix - this has parallel reductions in it
   PetscCall(compute_nnzs_air_matrices(air_data));
   // Compute the nnzs in a single V cycle
   PetscCall(compute_nnzs_v_cycle_air(air_data,
                  pcmg_input,
                  &nnzs_air_v));

   // ~~~~~~~~~
   // Compute the grid complexity
   // ~~~~~~~~~
   *grid_complx = 0;
   for (our_level = 0; our_level < air_data->no_levels - 1; our_level++) {
      PetscCall(MatGetSize(air_data->coarse_matrix[our_level], &global_rows, &global_cols));
      *grid_complx = *grid_complx + (PetscReal)global_rows;
   }
   // Don't forget the bottom grid (which is the top grid if we only have one level)
   PetscCall(MatGetSize(air_data->coarse_matrix[air_data->no_levels - 1], &global_rows, &global_cols));
   *grid_complx = *grid_complx + (PetscReal)global_rows;

   PetscCall(MatGetSize(air_data->coarse_matrix[0], &global_rows, &global_cols));
   *grid_complx = *grid_complx/(PetscReal)global_rows;


   // ~~~~~~~~~
   // Compute the nnzs for operator complexity and storage complexity
   // ~~~~~~~~~
   // Need a do loop so we don't overflow in the sum
   op_complx_nnzs = 0;
   for (our_level = 0; our_level < air_data->no_levels; our_level++) {
      op_complx_nnzs = op_complx_nnzs + air_data->coarse_matrix_nnzs[our_level];
   }
   mat_storage_nnzs = 0;
   for (our_level = 0; our_level < air_data->no_levels - 1; our_level++) {
      // Given we do F point up smoothing only, we don't actually have to store very much (e.g., the coarse matrix)
      // So how much matrix storage do we actually need
      if (!air_data->options.full_smoothing_up_and_down) {
         mat_storage_nnzs = mat_storage_nnzs + air_data->A_ff_nnzs[our_level];
         mat_storage_nnzs = mat_storage_nnzs + air_data->A_fc_nnzs[our_level];
         // If we are doing C point smoothing need the other matrices
         if (air_data->options.any_c_smooths) {
            mat_storage_nnzs = mat_storage_nnzs + air_data->A_cc_nnzs[our_level];
            mat_storage_nnzs = mat_storage_nnzs + air_data->inv_A_cc_nnzs[our_level];
            mat_storage_nnzs = mat_storage_nnzs + air_data->A_cf_nnzs[our_level];
         }
      } else {
         mat_storage_nnzs = mat_storage_nnzs + air_data->coarse_matrix_nnzs[our_level];
      }
      mat_storage_nnzs = mat_storage_nnzs + air_data->inv_A_ff_nnzs[our_level];
      mat_storage_nnzs = mat_storage_nnzs + air_data->restrictor_nnzs[our_level];
      mat_storage_nnzs = mat_storage_nnzs + air_data->prolongator_nnzs[our_level];
   }

   // Coarse grid solve
   if (air_data->no_levels != 1) {
      PetscCall(PCMGGetCoarseSolve(pcmg_input, &ksp));
      PetscCall(KSPGetTolerances(ksp, &rtol, &atol, &dtol, &maxits_coarse));
      // If we're doing matrix-free application of coarse grid, we need the coarse grid
      if (air_data->options.coarsest_matrix_free_polys) {
         mat_storage_nnzs = mat_storage_nnzs + air_data->coarse_matrix_nnzs[air_data->no_levels - 1];
      // If we're not doing matrix-free application of the coarse grid, then we need the inverse of the coarse grid matrix
      } else {
         mat_storage_nnzs = mat_storage_nnzs + air_data->inv_A_ff_nnzs[air_data->no_levels - 1];

         // If we're doing more than one mb iteration on the coarse grid, then we also need to store the coarse grid matrix
         // in order to compute a residual in the richardson
         if (maxits_coarse != 1) {
            mat_storage_nnzs = mat_storage_nnzs + air_data->coarse_matrix_nnzs[air_data->no_levels - 1];
         }
      }

   // If we only have a single level
   } else {
      PetscCall(PCGetType(pcmg_input, &pc_type));
      PetscCall(PetscStrcmp(pc_type, PCMAT, &is_mat));
      PetscCall(PetscStrcmp(pc_type, PCJACOBI, &is_jacobi));
      // If we auto truncated the PC is a single application of the coarse grid solver
      // so we never need the matrix to compute a residual unless we're mf
      if (is_mat) {
         if (air_data->options.coarsest_matrix_free_polys) {
            mat_storage_nnzs = mat_storage_nnzs + air_data->coarse_matrix_nnzs[0];
         } else {
            mat_storage_nnzs = mat_storage_nnzs + air_data->inv_A_ff_nnzs[0];
         }
      // Jacobi just needs the diagonal
      } else if (is_jacobi) {
         PetscCall(MatGetSize(air_data->coarse_matrix[0], &global_rows, &global_cols));
         mat_storage_nnzs = mat_storage_nnzs + (PetscInt64)global_rows;
      }
   }

   // ~~~~~~~~~
   // Compute the nnzs used as part of the reuse
   // ~~~~~~~~~
   mat_reuse_storage_nnzs = 0;
   for (our_level = 0; our_level < air_data->no_levels - 1; our_level++) {
      // Loop over all the reused matrices
      for (i_loc = 0; i_loc < (int)PETSC_STATIC_ARRAY_LENGTH(air_data->reuse[our_level].reuse_mat); i_loc++) {
         if (air_data->reuse[our_level].reuse_mat[i_loc]) {
            PetscCall(get_nnzs_petsc_sparse(air_data->reuse[our_level].reuse_mat[i_loc], &mat_nnzs));
            mat_reuse_storage_nnzs = mat_reuse_storage_nnzs + mat_nnzs;
         }
      }
      // Include any IS's
      for (i_loc = 0; i_loc < (int)PETSC_STATIC_ARRAY_LENGTH(air_data->reuse[our_level].reuse_is); i_loc++) {
         if (air_data->reuse[our_level].reuse_is[i_loc]) {
            PetscCall(ISGetSize(air_data->reuse[our_level].reuse_is[i_loc], &global_rows));
            mat_reuse_storage_nnzs = mat_reuse_storage_nnzs + global_rows;
         }
      }
   }

   // Now compute the other complexities
   *op_complx = (PetscReal)op_complx_nnzs/(PetscReal)air_data->coarse_matrix_nnzs[0];
   *cycle_complx = (PetscReal)nnzs_air_v/(PetscReal)air_data->coarse_matrix_nnzs[0];
   *storage_complx = (PetscReal)mat_storage_nnzs/
            (PetscReal)air_data->coarse_matrix_nnzs[0];
   *reuse_storage_complx = (PetscReal)mat_reuse_storage_nnzs/
            (PetscReal)air_data->coarse_matrix_nnzs[0];

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode print_stats(air_multigrid_data *air_data, PC pcmg_input)
{
   // Print out stats on the mg hierarchy

   // ~~~~~~
   PetscReal grid_complx, op_complx, cycle_complx, storage_complx, reuse_storage_complx;
   PetscMPIInt comm_rank;
   MPI_Comm MPI_COMM_MATRIX;
   // ~~~~~~

   PetscFunctionBegin;

   // Get the communicator the input matrix is on, we build everything on that
   PetscCall(PetscObjectGetComm((PetscObject)pcmg_input, &MPI_COMM_MATRIX));
   // Get the comm rank
   PetscCallMPI(MPI_Comm_rank(MPI_COMM_MATRIX, &comm_rank));

   PetscCall(compute_stats(air_data, pcmg_input, &grid_complx, &op_complx, &cycle_complx, &storage_complx, &reuse_storage_complx));

   if (comm_rank == 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, " ~~~~~~~~~~~~\n"));
   if (comm_rank == 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, " Grid complexity            : %.15g\n", (double)grid_complx));
   if (comm_rank == 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, " Operator complexity        : %.15g\n", (double)op_complx));
   if (comm_rank == 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, " Cycle complexity           : %.15g\n", (double)cycle_complx));
   if (comm_rank == 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, " Storage complexity         : %.15g\n", (double)storage_complx));
   if (comm_rank == 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, " Reuse storage complexity   : %.15g\n", (double)reuse_storage_complx));
   if (comm_rank == 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, " ~~~~~~~~~~~~\n"));

   PetscFunctionReturn(PETSC_SUCCESS);
}
