/*
  C port of FC_Smooth.F90 (module fc_smooth)

  The levels are 0-based, so our_level indexes the per level arrays of
  air_multigrid_data directly (the Fortran our_level is our_level + 1). The
  Kokkos IS views in VecISCopyLocalk.kokkos.cxx number the levels from 1, as the
  Fortran did, so they are given our_level + 1
  The level matshells the FC smoothers are handed are the placeholder matshells
  AIR_MG_Setup.c builds, whose mat_ctxtype context holds our_level and air_data
  The Fortran print + MPI_Abort checks are PetscCheck/SETERRQ with the same message
*/

#include <petscksp.h>
#include "pflareimpl.h"
#include "VecISCopyLocalk.h"
#include "PETSc_Helper.h"
#include "FC_Smooth.h"

// -------------------------------------------------------------------------------------------------------------------------------
// Functions involving the FC smoothing
// -------------------------------------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode mg_coarse_shell_apply(PC pc, Vec x, Vec y)
{
   // PCShell apply for the default coarse-grid solver: applies our polynomial
   // approximate inverse of the coarse matrix, inv_A_ff(no_levels), as y = inv_A_ff x.
   // Using a shell (rather than PCMAT with the inverse as the Pmat) lets us set the
   // coarse KSP's Amat and Pmat to the actual coarse operator, so a user-supplied
   // coarse PC (e.g. -mg_coarse_pc_type lu) factorises/solves the right matrix.
   // The shell context is the air_multigrid_data (a target that outlives the PC).

   // ~~~~~~
   air_multigrid_data *air_data = NULL;
   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(PCShellGetContext(pc, &air_data));
   PetscCall(MatMult(air_data->inv_A_ff[air_data->no_levels - 1], x, y));

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode mg_coarse_shell_apply_transpose(PC pc, Vec x, Vec y)
{
   // The transpose of mg_coarse_shell_apply, y = inv_A_ff(no_levels)^T x
   // Our polynomial inverses all know how to do a transposed matvec, either
   // natively when assembled or through a transposed twin when matrix-free

   // ~~~~~~
   air_multigrid_data *air_data = NULL;
   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(PCShellGetContext(pc, &air_data));
   PetscCall(MatMultTranspose(air_data->inv_A_ff[air_data->no_levels - 1], x, y));

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode mg_smooth_shell_apply(PC pc, Vec x, Vec y)
{
   // PCShell apply used for the level smoothers when doing full smoothing up
   // and down with a matrix-free inverse. This does exactly what PCMAT does
   // (the Pmat of the smoother is inv_A_ff on that level), we just can't use
   // PCMAT as its multiple rhs apply does a MatMatMult on the Pmat, which fails
   // on a matrix-free polynomial matshell

   // ~~~~~~
   Mat mat, pmat;
   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(PCGetOperators(pc, &mat, &pmat));
   PetscCall(MatMult(pmat, x, y));

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode mg_smooth_shell_apply_transpose(PC pc, Vec x, Vec y)
{
   // The transpose of mg_smooth_shell_apply - the Pmat of the smoother is
   // inv_A_ff on that level, so this is just a transposed matvec with it
   // This is what KSPSolveTranspose on the level smoothers needs when doing
   // full smoothing up and down with a matrix-free inverse

   // ~~~~~~
   Mat mat, pmat;
   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(PCGetOperators(pc, &mat, &pmat));
   PetscCall(MatMultTranspose(pmat, x, y));

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode create_smooth_temp_vecs(air_multigrid_data *air_data)
{
   // Creates the temporary vectors we use during smoothing on every level that doesn't have them
   // These are only ever scratch space within a single smooth (or residual) on one level,
   // and we only smooth one level at a time, so rather than give every level its own memory
   // each slot (e.g., temp_vecs_fine(2)) is a subvector of one pool shared across all the
   // levels, sized for the biggest level on this rank
   // The pools must not exist yet, ie they are destroyed whenever the temporary vectors are

   // ~~~~~~~~~~
   int our_level, i_loc, n_coarse, first_level;
   PetscInt local_rows, local_cols, max_fine, max_coarse, max_full;
   PetscInt pool_start, pool_end_plus_one;
   PetscInt one = 1;
   MPI_Comm MPI_COMM_MATRIX;
   VecType vec_type;
   // ~~~~~~~~~~

   PetscFunctionBegin;

   // Find the biggest local sizes on the levels we have to build
   first_level = -1;
   max_fine = 0;
   max_coarse = 0;
   max_full = 0;
   for (our_level = 0; our_level < air_data->no_levels - 1; our_level++) {
      if (air_data->allocated_matrices_A_ff[our_level]) continue;
      if (first_level == -1) first_level = our_level;

      PetscCall(MatGetLocalSize(air_data->A_ff[our_level], &local_rows, &local_cols));
      max_fine = PetscMax(max_fine, local_cols);
      PetscCall(MatGetLocalSize(air_data->A_fc[our_level], &local_rows, &local_cols));
      max_coarse = PetscMax(max_coarse, local_cols);
      if (!air_data->fast_veciscopy_exists) {
         PetscCall(MatGetLocalSize(air_data->coarse_matrix[our_level], &local_rows, &local_cols));
         max_full = PetscMax(max_full, local_cols);
      }
   }
   if (first_level == -1) PetscFunctionReturn(PETSC_SUCCESS);

   // The temporary vectors on every level are subvectors of the pools, so we can only
   // build them when no level has them, ie the pools don't exist
   PetscCheck(!air_data->temp_vecs_fine_pool[0], PETSC_COMM_SELF, PETSC_ERR_PLIB, "The smoothing temporaries can only be created when no level has them");

   // If we're doing C point smoothing we need some extra temporaries
   // The transposed smooth and residual don't need any more than the
   // forward ones do
   n_coarse = 1;
   if (air_data->options.any_c_smooths &&
            !air_data->options.full_smoothing_up_and_down) n_coarse = 4;

   // Create the pools with the vec type the matrices use
   PetscCall(PetscObjectGetComm((PetscObject)air_data->A_fc[first_level], &MPI_COMM_MATRIX));
   PetscCall(MatGetVecType(air_data->A_fc[first_level], &vec_type));
   for (i_loc = 0; i_loc < 4; i_loc++) {
      PetscCall(VecCreate(MPI_COMM_MATRIX, &air_data->temp_vecs_fine_pool[i_loc]));
      PetscCall(VecSetSizes(air_data->temp_vecs_fine_pool[i_loc], max_fine, PETSC_DECIDE));
      PetscCall(VecSetType(air_data->temp_vecs_fine_pool[i_loc], vec_type));
   }
   for (i_loc = 0; i_loc < n_coarse; i_loc++) {
      PetscCall(VecCreate(MPI_COMM_MATRIX, &air_data->temp_vecs_coarse_pool[i_loc]));
      PetscCall(VecSetSizes(air_data->temp_vecs_coarse_pool[i_loc], max_coarse, PETSC_DECIDE));
      PetscCall(VecSetType(air_data->temp_vecs_coarse_pool[i_loc], vec_type));
   }
   if (!air_data->fast_veciscopy_exists) {
      PetscCall(VecCreate(MPI_COMM_MATRIX, &air_data->temp_vecs_pool[0]));
      PetscCall(VecSetSizes(air_data->temp_vecs_pool[0], max_full, PETSC_DECIDE));
      PetscCall(VecSetType(air_data->temp_vecs_pool[0], vec_type));
   }

   // On each level take the start of each pool as the temporary vectors
   // The IS is contiguous so the subvector shares the pool's memory
   for (our_level = 0; our_level < air_data->no_levels - 1; our_level++) {
      if (air_data->allocated_matrices_A_ff[our_level]) continue;

      PetscCall(MatGetLocalSize(air_data->A_ff[our_level], &local_rows, &local_cols));
      PetscCall(VecGetOwnershipRange(air_data->temp_vecs_fine_pool[0], &pool_start, &pool_end_plus_one));
      PetscCall(ISCreateStride(MPI_COMM_MATRIX, local_cols, pool_start, one,
               &air_data->temp_vecs_fine_is[our_level]));
      for (i_loc = 0; i_loc < 4; i_loc++) {
         PetscCall(VecGetSubVector(air_data->temp_vecs_fine_pool[i_loc], air_data->temp_vecs_fine_is[our_level],
                  &air_data->temp_vecs_fine[i_loc].array[our_level]));
      }

      PetscCall(MatGetLocalSize(air_data->A_fc[our_level], &local_rows, &local_cols));
      PetscCall(VecGetOwnershipRange(air_data->temp_vecs_coarse_pool[0], &pool_start, &pool_end_plus_one));
      PetscCall(ISCreateStride(MPI_COMM_MATRIX, local_cols, pool_start, one,
               &air_data->temp_vecs_coarse_is[our_level]));
      for (i_loc = 0; i_loc < n_coarse; i_loc++) {
         PetscCall(VecGetSubVector(air_data->temp_vecs_coarse_pool[i_loc], air_data->temp_vecs_coarse_is[our_level],
                  &air_data->temp_vecs_coarse[i_loc].array[our_level]));
      }

      if (!air_data->fast_veciscopy_exists) {
         PetscCall(MatGetLocalSize(air_data->coarse_matrix[our_level], &local_rows, &local_cols));
         PetscCall(VecGetOwnershipRange(air_data->temp_vecs_pool[0], &pool_start, &pool_end_plus_one));
         PetscCall(ISCreateStride(MPI_COMM_MATRIX, local_cols, pool_start, one,
                  &air_data->temp_vecs_is[our_level]));
         PetscCall(VecGetSubVector(air_data->temp_vecs_pool[0], air_data->temp_vecs_is[our_level],
                  &air_data->temp_vecs[0].array[our_level]));
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode create_VecISCopyLocalWrapper(air_multigrid_data *air_data, int our_level, MatType mat_type, Mat input_mat)
{
   // Creates any data we might need in VecISCopyLocalWrapper for a given level
   // air_data%fast_veciscopy_exists must have been set before the
   // first call to this routine

   // ~~~~~~~~~~
#if defined(PETSC_HAVE_KOKKOS)
   PetscInt global_row_start, global_row_end_plus_one;
   PetscBool is_kokkos;
#endif
   // ~~~~~~~~~~

   PetscFunctionBegin;

   // On cpus we use VecISCopy to pull out fine and coarse points
   // That copies back to the cpu if doing gpu, so on the gpu we build
   // identity restrictors/prolongators of various sizes and do matmults
   if (!air_data->fast_veciscopy_exists) {

      // Build fine to full injector
      PetscCall(generate_identity_rect(input_mat,
               air_data->IS_fine_index[our_level],
               &air_data->i_fine_full[our_level]));

      // Build coarse to full injector
      PetscCall(generate_identity_rect(input_mat,
               air_data->IS_coarse_index[our_level],
               &air_data->i_coarse_full[our_level]));

      // Build identity that sets fine in full to zero
      PetscCall(generate_identity_is(input_mat, air_data->IS_coarse_index[our_level],
               &air_data->i_coarse_full_full[our_level]));

      // Build identity that sets coarse in full to zero
      // This is needed whenever we scatter coarse values back into a full sized
      // vector, which the C point smooths do and so does the transposed residual,
      // as that has to write r_c back even with no C smooths
      PetscCall(generate_identity_is(input_mat, air_data->IS_fine_index[our_level],
            &air_data->i_fine_full_full[our_level]));

   // We're either on the cpu or on the gpu with kokkos
   } else {
#if defined(PETSC_HAVE_KOKKOS)

      // If our mat type is kokkos we need to build some things
      // If not we just use the petsc veciscopy and don't have to setup anything
      PetscCall(PetscStrcmpAny(mat_type, &is_kokkos, MATMPIAIJKOKKOS, MATSEQAIJKOKKOS, MATAIJKOKKOS, ""));
      if (is_kokkos) {

         // Build in case not built yet
         create_VecISCopyLocal_kokkos((int)air_data->options.max_levels, &air_data->kokkos_is_views_handle);
         PetscCall(MatGetOwnershipRange(input_mat, &global_row_start, &global_row_end_plus_one));

         // Copy the IS's over to the device
         // The kokkos IS views number the levels from 1
         set_VecISCopyLocal_kokkos_our_level(air_data->kokkos_is_views_handle, our_level + 1,
                  global_row_start, &air_data->IS_fine_index[our_level], &air_data->IS_coarse_index[our_level]);

      }
#else
      (void)mat_type;
#endif
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode destroy_VecISCopyLocalWrapper(air_multigrid_data *air_data, int our_level)
{
   // Destroy any data we might need in VecISCopyLocalWrapper for a given level

   PetscFunctionBegin;

   // Destroys the matrices
   if (!air_data->fast_veciscopy_exists) {

      PetscCall(MatDestroy(&air_data->i_fine_full[our_level]));
      PetscCall(MatDestroy(&air_data->i_coarse_full[our_level]));
      PetscCall(MatDestroy(&air_data->i_coarse_full_full[our_level]));
      PetscCall(MatDestroy(&air_data->i_fine_full_full[our_level]));

   } else {
#if defined(PETSC_HAVE_KOKKOS)
      destroy_VecISCopyLocal_kokkos(&air_data->kokkos_is_views_handle);
#endif
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode VecISCopyLocalWrapper(air_multigrid_data *air_data, int our_level, PetscBool fine, Vec vfull, ScatterMode mode, Vec vreduced,
   Vec v_temp_mat)
{
   // Wrapper around VecISCopy (currently cpu only), a kokkos version of that and
   // the matmult used on gpus when petsc isn't configured with kokkos
   // Relies on having pre-built some things with the routine create_VecISCopyLocalWrapper
   // v_temp_mat is optional (NULL when absent), it is only used by SCATTER_FORWARD
   // with the injectors

   // ~~~~~~~~~~
   int mode_int;
#if defined(PETSC_HAVE_KOKKOS)
   int fine_int;
   VecType vec_type;
   Vec temp_vec = NULL;
   PetscReal normy;
   PetscBool is_kokkos;
#endif
   // ~~~~~~~~~~

   PetscFunctionBegin;

   if (mode == SCATTER_REVERSE) {
      mode_int = 1;
   } else {
      mode_int = 0;
   }
   // FINE variables
   if (fine) {
      if (mode == SCATTER_REVERSE) {

         if (!air_data->fast_veciscopy_exists) {
            PetscCall(MatMult(air_data->i_fine_full[our_level], vfull,
                     vreduced));
         } else {

#if defined(PETSC_HAVE_KOKKOS)

            // The device IS views only exist if the matrix type was kokkos in
            // create_VecISCopyLocalWrapper, so fall back to VecISCopy if not
            // (e.g., kokkos vecs with an aij matrix)
            PetscCall(VecGetType(vfull, &vec_type));
            PetscCall(PetscStrcmpAny(vec_type, &is_kokkos, "seqkokkos", "mpikokkos", "kokkos", ""));
            if (is_kokkos && air_data->kokkos_is_views_handle) {

               fine_int = 0;
               if (fine) fine_int = 1;
               // The kokkos IS views number the levels from 1
               VecISCopyLocal_kokkos(air_data->kokkos_is_views_handle, our_level + 1, fine_int, &vfull,
                        mode_int, &vreduced);

               // If debugging do a comparison between CPU and Kokkos results
               if (kokkos_debug()) {

                  PetscCall(VecDuplicate(vreduced, &temp_vec));
                  PetscCall(VecISCopy(vfull, air_data->IS_fine_index[our_level], mode,
                           temp_vec));
                  PetscCall(VecAXPY(temp_vec, PFLARE_MINUS_ONE, vreduced));
                  PetscCall(VecNorm(temp_vec, NORM_2, &normy));
                  PetscCheck(!(normy > PFLARE_TOL_MATFREE_13), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Kokkos and CPU versions of VecISCopyLocalWrapper REV FINE do not match");
                  PetscCall(VecDestroy(&temp_vec));

               }

            } else {
               PetscCall(VecISCopy(vfull, air_data->IS_fine_index[our_level], mode,
                     vreduced));
            }
#else
            PetscCall(VecISCopy(vfull, air_data->IS_fine_index[our_level], mode,
                     vreduced));
#endif
         }

      // SCATTER FORWARD
      } else {
         if (!air_data->fast_veciscopy_exists) {

            // Copy x but only the non-coarse points from x are non-zero
            // ie get x_c but in a vec of full size
            PetscCall(MatMult(air_data->i_coarse_full_full[our_level], vfull,
                              v_temp_mat));

            // If we're just doing F point smoothing, don't change the coarse points
            // Not sure why we need the vecset, but on the gpu x is twice the size it should be if we don't
            // x should be overwritten by the MatMultTransposeAdd
            PetscCall(VecSet(vfull, PFLARE_ZERO));
            PetscCall(MatMultTransposeAdd(air_data->i_fine_full[our_level],
                  vreduced,
                  v_temp_mat,
                  vfull));

         } else {

#if defined(PETSC_HAVE_KOKKOS)

            // The device IS views only exist if the matrix type was kokkos in
            // create_VecISCopyLocalWrapper, so fall back to VecISCopy if not
            // (e.g., kokkos vecs with an aij matrix)
            PetscCall(VecGetType(vfull, &vec_type));
            PetscCall(PetscStrcmpAny(vec_type, &is_kokkos, "seqkokkos", "mpikokkos", "kokkos", ""));
            if (is_kokkos && air_data->kokkos_is_views_handle) {

               if (kokkos_debug()) {
                  PetscCall(VecDuplicate(vfull, &temp_vec));
                  PetscCall(VecCopy(vfull, temp_vec));
               }

               fine_int = 0;
               if (fine) fine_int = 1;
               // The kokkos IS views number the levels from 1
               VecISCopyLocal_kokkos(air_data->kokkos_is_views_handle, our_level + 1, fine_int, &vfull,
                        mode_int, &vreduced);

               // If debugging do a comparison between CPU and Kokkos results
               if (kokkos_debug()) {

                  PetscCall(VecISCopy(temp_vec, air_data->IS_fine_index[our_level], mode,
                           vreduced));
                  PetscCall(VecAXPY(temp_vec, PFLARE_MINUS_ONE, vfull));
                  PetscCall(VecNorm(temp_vec, NORM_2, &normy));
                  PetscCheck(!(normy > PFLARE_TOL_MATFREE_13), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Kokkos and CPU versions of VecISCopyLocalWrapper FORW FINE do not match");
                  PetscCall(VecDestroy(&temp_vec));

               }

            } else {
               PetscCall(VecISCopy(vfull, air_data->IS_fine_index[our_level], mode,
                        vreduced));
            }
#else
            PetscCall(VecISCopy(vfull, air_data->IS_fine_index[our_level], mode,
                     vreduced));
#endif
         }
      }

   // COARSE variables
   } else {
      if (mode == SCATTER_REVERSE) {

         if (!air_data->fast_veciscopy_exists) {
            PetscCall(MatMult(air_data->i_coarse_full[our_level], vfull,
                     vreduced));
         } else {

#if defined(PETSC_HAVE_KOKKOS)

            // The device IS views only exist if the matrix type was kokkos in
            // create_VecISCopyLocalWrapper, so fall back to VecISCopy if not
            // (e.g., kokkos vecs with an aij matrix)
            PetscCall(VecGetType(vfull, &vec_type));
            PetscCall(PetscStrcmpAny(vec_type, &is_kokkos, "seqkokkos", "mpikokkos", "kokkos", ""));
            if (is_kokkos && air_data->kokkos_is_views_handle) {

               fine_int = 0;
               if (fine) fine_int = 1;
               // The kokkos IS views number the levels from 1
               VecISCopyLocal_kokkos(air_data->kokkos_is_views_handle, our_level + 1, fine_int, &vfull,
                        mode_int, &vreduced);

               // If debugging do a comparison between CPU and Kokkos results
               if (kokkos_debug()) {

                  PetscCall(VecDuplicate(vreduced, &temp_vec));
                  PetscCall(VecISCopy(vfull, air_data->IS_coarse_index[our_level], mode,
                        temp_vec));
                  PetscCall(VecAXPY(temp_vec, PFLARE_MINUS_ONE, vreduced));
                  PetscCall(VecNorm(temp_vec, NORM_2, &normy));
                  PetscCheck(!(normy > PFLARE_TOL_MATFREE_13), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Kokkos and CPU versions of VecISCopyLocalWrapper REV COARSE do not match");
                  PetscCall(VecDestroy(&temp_vec));

               }

            } else {
               PetscCall(VecISCopy(vfull, air_data->IS_coarse_index[our_level], mode,
                        vreduced));
            }
#else
            PetscCall(VecISCopy(vfull, air_data->IS_coarse_index[our_level], mode,
                     vreduced));
#endif
         }

      // SCATTER FORWARD
      } else {

         if (!air_data->fast_veciscopy_exists) {

            // Copy x but only the non-fine points from x are non-zero
            // ie get x_f but in a vec of full size
            PetscCall(MatMult(air_data->i_fine_full_full[our_level], vfull,
                              v_temp_mat));

            // Not sure why we need the vecset, but on the gpu x is twice the size it should be if we don't
            // x should be overwritten by the MatMultTransposeAdd
            PetscCall(VecSet(vfull, PFLARE_ZERO));
            PetscCall(MatMultTransposeAdd(air_data->i_coarse_full[our_level],
                  vreduced,
                  v_temp_mat,
                  vfull));

         } else {

#if defined(PETSC_HAVE_KOKKOS)

            // The device IS views only exist if the matrix type was kokkos in
            // create_VecISCopyLocalWrapper, so fall back to VecISCopy if not
            // (e.g., kokkos vecs with an aij matrix)
            PetscCall(VecGetType(vfull, &vec_type));
            PetscCall(PetscStrcmpAny(vec_type, &is_kokkos, "seqkokkos", "mpikokkos", "kokkos", ""));
            if (is_kokkos && air_data->kokkos_is_views_handle) {

               if (kokkos_debug()) {
                  PetscCall(VecDuplicate(vfull, &temp_vec));
                  PetscCall(VecCopy(vfull, temp_vec));
               }

               fine_int = 0;
               if (fine) fine_int = 1;
               // The kokkos IS views number the levels from 1
               VecISCopyLocal_kokkos(air_data->kokkos_is_views_handle, our_level + 1, fine_int, &vfull,
                        mode_int, &vreduced);

               // If debugging do a comparison between CPU and Kokkos results
               if (kokkos_debug()) {

                  PetscCall(VecISCopy(temp_vec, air_data->IS_coarse_index[our_level], mode,
                        vreduced));
                  PetscCall(VecAXPY(temp_vec, PFLARE_MINUS_ONE, vfull));
                  PetscCall(VecNorm(temp_vec, NORM_2, &normy));
                  PetscCheck(!(normy > PFLARE_TOL_MATFREE_13), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Kokkos and CPU versions of VecISCopyLocalWrapper FORW COARSE do not match");
                  PetscCall(VecDestroy(&temp_vec));

               }

            } else {
               PetscCall(VecISCopy(vfull, air_data->IS_coarse_index[our_level], mode,
                     vreduced));
            }
#else
            PetscCall(VecISCopy(vfull, air_data->IS_coarse_index[our_level], mode,
                     vreduced));
#endif
         }
      }
   }

   // The integer form of the mode is only used by the kokkos version
   (void)mode_int;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode mg_FC_point_richardson(PC pc, Vec b, Vec x, Vec r, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt maxits,
   PetscBool guess_zero, PetscInt *its, PCRichardsonConvergedReason *conv_reason)
{
   // This applies an FC point richardson. This saves computing full residuals on each level
   // This is automatically disabled if you run with -mg_levels_ksp_monitor fyi!

   // ~~~~~~
   Mat mat, pmat;
   int our_level, i, smooth_its;
   mat_ctxtype *mat_ctx = NULL;
   air_multigrid_data *air_data;
   PetscBool first_smooth;
   // ~~~~~~

   PetscFunctionBegin;

   // Set these for output
   *its = maxits;
   *conv_reason = PCRICHARDSON_CONVERGED_ITS;

   // Can come in here with zero maxits, have to do nothing
   if (maxits == 0) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCheck(maxits == 1, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_OUTOFRANGE, "To change the number of smooths adjust smooth_order");

   // Get the level
   PetscCall(PCGetOperators(pc, &mat, &pmat));
   // Get what level we are on
   PetscCall(MatShellGetContext(mat, &mat_ctx));
   our_level = mat_ctx->our_level;
   air_data = mat_ctx->air_data;

   // The first time we go through any smooth, we need to pull out x_f and/or x_c
   first_smooth = PETSC_TRUE;

   // Loop over all the smooths we need to do
   for (i = 0; i < air_data->smooth_order_levels[our_level].size; i++) {

      smooth_its = air_data->smooth_order_levels[our_level].array[i];

      if (smooth_its == 0) break;

      // Do consecutive F point smooths
      if (smooth_its > 0) {

         PetscCall(f_smooths(b, x, guess_zero, first_smooth, air_data, our_level, smooth_its));

      // Do consecutive C point smooths
      } else {

         PetscCall(c_smooths(b, x, guess_zero, first_smooth, air_data, our_level, PetscAbsInt(smooth_its)));
      }

      // Once we've done our first smooth, we can use the existing values
      first_smooth = PETSC_FALSE;

   }

   // Now technically there should be a new residual that we put into r after this is done
   // but I don't think it matters, as it is the solution that is interpolated up
   // and the richardson on the next level up computes its own F-point residual
   // and the norm type is none on the mg levels, as we just do maxits
   (void)r;
   (void)rtol;
   (void)abstol;
   (void)dtol;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode f_smooths(Vec b, Vec x, PetscBool guess_zero, PetscBool first_smooth, air_multigrid_data *air_data, int our_level, int its)
{
   // This applies consecutive F smooths

   // ~~~~~~
   int f_its;
   // ~~~~~~

   PetscFunctionBegin;

   (void)guess_zero;

   // Get out just the fine points from b - this is b_f
   PetscCall(VecISCopyLocalWrapper(air_data, our_level, PETSC_TRUE, b,
            SCATTER_REVERSE, air_data->temp_vecs_fine[3].array[our_level], NULL));

   // If we haven't done any smooth before calling this F point smooth
   // we need to pull out x_c^0 and x_f^0
   if (first_smooth) {

      // Get out just the fine points from x - this is x_f^0
      PetscCall(VecISCopyLocalWrapper(air_data, our_level, PETSC_TRUE, x,
               SCATTER_REVERSE, air_data->temp_vecs_fine[0].array[our_level], NULL));

      // Get the coarse points from x - this is x_c^0
      PetscCall(VecISCopyLocalWrapper(air_data, our_level, PETSC_FALSE, x,
               SCATTER_REVERSE, air_data->temp_vecs_coarse[0].array[our_level], NULL));

   }

   // Compute Afc * x_c^0 - this never changes
   PetscCall(MatMult(air_data->A_fc[our_level], air_data->temp_vecs_coarse[0].array[our_level],
            air_data->temp_vecs_fine[1].array[our_level]));

   // This is b_f - A_fc * x_c^0 - this never changes
   PetscCall(VecAXPY(air_data->temp_vecs_fine[3].array[our_level], PFLARE_MINUS_ONE,
            air_data->temp_vecs_fine[1].array[our_level]));

   // Do all the consecutive F smooths
   for (f_its = 1; f_its <= its; f_its++) {

      // Then A_ff * x_f^n - this changes at each richardson iteration
      PetscCall(MatMult(air_data->A_ff[our_level], air_data->temp_vecs_fine[0].array[our_level],
                  air_data->temp_vecs_fine[2].array[our_level]));

      // This is b_f - A_fc * x_c - A_ff * x_f^n
      PetscCall(VecAYPX(air_data->temp_vecs_fine[2].array[our_level], PFLARE_MINUS_ONE,
               air_data->temp_vecs_fine[3].array[our_level]));

      // ! Compute A_ff^{-1} ( b_f - A_fc * x_c - A_ff * x_f^n)
      PetscCall(MatMult(air_data->inv_A_ff[our_level], air_data->temp_vecs_fine[2].array[our_level],
                  air_data->temp_vecs_fine[1].array[our_level]));

      // Compute x_f^n + A_ff^{-1} ( b_f - A_fc * x_c - A_ff * x_f^n)
      PetscCall(VecAXPY(air_data->temp_vecs_fine[0].array[our_level], PFLARE_ONE,
               air_data->temp_vecs_fine[1].array[our_level]));

   }

   // ~~~~~~~~
   // Reverse put fine x_f back into x
   // ~~~~~~~~
   PetscCall(VecISCopyLocalWrapper(air_data, our_level, PETSC_TRUE, x,
            SCATTER_FORWARD, air_data->temp_vecs_fine[0].array[our_level],
            air_data->temp_vecs[0].array[our_level]));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode c_smooths(Vec b, Vec x, PetscBool guess_zero, PetscBool first_smooth, air_multigrid_data *air_data, int our_level, int its)
{
   // This applies consecutive C smooths

   // ~~~~~~
   int c_its;
   // ~~~~~~

   PetscFunctionBegin;

   (void)guess_zero;

   // Get out just the coarse points from b - this is b_c
   PetscCall(VecISCopyLocalWrapper(air_data, our_level, PETSC_FALSE, b,
            SCATTER_REVERSE, air_data->temp_vecs_coarse[3].array[our_level], NULL));

   // If we haven't done any smooth before calling this C point smooth
   // we need to pull out x_c^0 and x_f^0
   if (first_smooth) {

         // Get out just the fine points from x - this is x_f^0
      PetscCall(VecISCopyLocalWrapper(air_data, our_level, PETSC_TRUE, x,
               SCATTER_REVERSE, air_data->temp_vecs_fine[0].array[our_level], NULL));

      // Get the coarse points from x - this is x_c^0
      PetscCall(VecISCopyLocalWrapper(air_data, our_level, PETSC_FALSE, x,
               SCATTER_REVERSE, air_data->temp_vecs_coarse[0].array[our_level], NULL));

   }

   // Compute Acf * x_f^0 - this never changes
   PetscCall(MatMult(air_data->A_cf[our_level], air_data->temp_vecs_fine[0].array[our_level],
               air_data->temp_vecs_coarse[1].array[our_level]));
   // This is b_c - A_cf * x_f^0 - this never changes
   PetscCall(VecAXPY(air_data->temp_vecs_coarse[3].array[our_level], PFLARE_MINUS_ONE,
            air_data->temp_vecs_coarse[1].array[our_level]));

   // Do all the consecutive C smooths
   for (c_its = 1; c_its <= its; c_its++) {

      // Then A_cc * x_c^n - this changes at each richardson iteration
      PetscCall(MatMult(air_data->A_cc[our_level], air_data->temp_vecs_coarse[0].array[our_level],
                  air_data->temp_vecs_coarse[2].array[our_level]));

      // This is b_c - A_cf * x_f^0 - A_cc * x_c^n
      PetscCall(VecAYPX(air_data->temp_vecs_coarse[2].array[our_level], PFLARE_MINUS_ONE,
               air_data->temp_vecs_coarse[3].array[our_level]));

      // ! Compute A_cc^{-1} (b_c - A_cf * x_f^0 - A_cc * x_c^n)
      PetscCall(MatMult(air_data->inv_A_cc[our_level], air_data->temp_vecs_coarse[2].array[our_level],
                  air_data->temp_vecs_coarse[1].array[our_level]));

      // Compute x_c^n + A_cc^{-1} (b_c - A_cf * x_f^0 - A_cc * x_c^n)
      PetscCall(VecAXPY(air_data->temp_vecs_coarse[0].array[our_level], PFLARE_ONE,
                  air_data->temp_vecs_coarse[1].array[our_level]));

   }

   // ~~~~~~~~
   // Reverse put coarse x_c back into x
   // ~~~~~~~~
   PetscCall(VecISCopyLocalWrapper(air_data, our_level, PETSC_FALSE, x,
            SCATTER_FORWARD, air_data->temp_vecs_coarse[0].array[our_level],
            air_data->temp_vecs[0].array[our_level]));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode mg_FC_point_apply_transpose(PC pc, Vec b, Vec x)
{
   // This applies the transpose of the F-C point smooth that mg_FC_point_richardson
   // applies on this level, from a zero initial guess, x = N^T b
   // This is the PCApplyTranspose of the smoother shell, which is what petsc's
   // transposed cycle drives with a KSPSolveTranspose - a transposed solve skips
   // the PCApplyRichardson fast path the forward smooth lives in, and with one
   // iteration and no norms the richardson is then just x = PCApplyTranspose(b)
   //
   // The forward smooth is a product of relaxations x += N_i (b - A x), so its
   // transpose is the same relaxations with A^T and N_i^T in place of A and N_i,
   // applied in the reverse order. The F row of A^T is [A_ff^T, A_cf^T] and the C
   // row is [A_fc^T, A_cc^T], so the transposed F relaxation is
   //      x_f += inv_A_ff^T (b_f - A_ff^T x_f - A_cf^T x_c)
   // and the transposed C relaxation is
   //      x_c += inv_A_cc^T (b_c - A_fc^T x_f - A_cc^T x_c)
   // Just like the forward smooth this only ever touches the blocks of A, never
   // the full operator on this level, which doesn't exist when we're F-C smoothing

   // ~~~~~~
   Mat mat, pmat;
   int our_level, i, no_blocks, smooth_its;
   mat_ctxtype *mat_ctx = NULL;
   air_multigrid_data *air_data;
   PetscBool x_f_zero, x_c_zero;
   // ~~~~~~

   PetscFunctionBegin;

   // Get the level, exactly as mg_FC_point_richardson does
   PetscCall(PCGetOperators(pc, &mat, &pmat));
   PetscCall(MatShellGetContext(mat, &mat_ctx));
   our_level = mat_ctx->our_level;
   air_data = mat_ctx->air_data;

   // How many blocks of smooths we do on this level - the ordering stops at
   // the first zero entry
   no_blocks = 0;
   for (i = 0; i < air_data->smooth_order_levels[our_level].size; i++) {
      if (air_data->smooth_order_levels[our_level].array[i] == 0) break;
      no_blocks = i + 1;
   }

   // The solution starts at zero and we track which halves of it are still
   // identically zero, as any term multiplying a zero half can just be skipped.
   // With the default F point smoothing x_c never becomes nonzero, so we never
   // touch A_cf (which doesn't exist then) and do no coarse sized work at all
   x_f_zero = PETSC_TRUE;
   x_c_zero = PETSC_TRUE;

   // Reverse order compared to mg_FC_point_richardson
   for (i = no_blocks - 1; i >= 0; i--) {

      smooth_its = air_data->smooth_order_levels[our_level].array[i];

      // Do consecutive F point smooths
      if (smooth_its > 0) {

         PetscCall(f_smooths_transpose(b, air_data, our_level, smooth_its, &x_f_zero, x_c_zero));

      // Do consecutive C point smooths
      } else {

         PetscCall(c_smooths_transpose(b, air_data, our_level, PetscAbsInt(smooth_its), x_f_zero, &x_c_zero));
      }

   }

   // ~~~~~~~~
   // Put the solution back into the full sized vector. x is a ksp work vector so
   // every entry has to be written - zero it and then scatter in whichever halves
   // are nonzero
   // ~~~~~~~~
   PetscCall(VecSet(x, PFLARE_ZERO));
   if (!x_f_zero) {
      PetscCall(VecISCopyLocalWrapper(air_data, our_level, PETSC_TRUE, x,
               SCATTER_FORWARD, air_data->temp_vecs_fine[0].array[our_level],
               air_data->temp_vecs[0].array[our_level]));
   }
   if (!x_c_zero) {
      PetscCall(VecISCopyLocalWrapper(air_data, our_level, PETSC_FALSE, x,
               SCATTER_FORWARD, air_data->temp_vecs_coarse[0].array[our_level],
               air_data->temp_vecs[0].array[our_level]));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode f_smooths_transpose(Vec b, air_multigrid_data *air_data, int our_level, int its, PetscBool *x_f_zero, PetscBool x_c_zero)
{
   // This applies the transpose of consecutive F smooths
   //      x_f = x_f + inv_A_ff^T (b_f - A_ff^T x_f - A_cf^T x_c)
   // x_c doesn't change in here, so b_f - A_cf^T x_c is computed once for the
   // whole block, exactly as the forward f_smooths does with b_f - A_fc * x_c

   // ~~~~~~
   int f_its;
   // ~~~~~~

   PetscFunctionBegin;

   // Get out just the fine points from b - this is b_f
   PetscCall(VecISCopyLocalWrapper(air_data, our_level, PETSC_TRUE, b,
            SCATTER_REVERSE, air_data->temp_vecs_fine[3].array[our_level], NULL));

   // This is b_f - A_cf^T * x_c - this never changes
   // A_cf only exists when we're doing C point smooths, which is also the only
   // way x_c can be nonzero
   if (!x_c_zero) {
      PetscCall(MatMultTranspose(air_data->A_cf[our_level], air_data->temp_vecs_coarse[0].array[our_level],
               air_data->temp_vecs_fine[1].array[our_level]));
      PetscCall(VecAXPY(air_data->temp_vecs_fine[3].array[our_level], PFLARE_MINUS_ONE,
               air_data->temp_vecs_fine[1].array[our_level]));
   }

   // Do all the consecutive F smooths
   for (f_its = 1; f_its <= its; f_its++) {

      // If x_f is still zero there is no A_ff^T * x_f term and nothing to add to,
      // so the whole relaxation is just x_f = A_ff^{-T} (b_f - A_cf^T * x_c)
      if (*x_f_zero) {

         PetscCall(MatMultTranspose(air_data->inv_A_ff[our_level], air_data->temp_vecs_fine[3].array[our_level],
                  air_data->temp_vecs_fine[0].array[our_level]));
         *x_f_zero = PETSC_FALSE;

      } else {

         // Then A_ff^T * x_f^n - this changes at each richardson iteration
         PetscCall(MatMultTranspose(air_data->A_ff[our_level], air_data->temp_vecs_fine[0].array[our_level],
                  air_data->temp_vecs_fine[2].array[our_level]));

         // This is b_f - A_cf^T * x_c - A_ff^T * x_f^n
         PetscCall(VecAYPX(air_data->temp_vecs_fine[2].array[our_level], PFLARE_MINUS_ONE,
                  air_data->temp_vecs_fine[3].array[our_level]));

         // Compute A_ff^{-T} ( b_f - A_cf^T * x_c - A_ff^T * x_f^n)
         PetscCall(MatMultTranspose(air_data->inv_A_ff[our_level], air_data->temp_vecs_fine[2].array[our_level],
                  air_data->temp_vecs_fine[1].array[our_level]));

         // Compute x_f^n + A_ff^{-T} ( b_f - A_cf^T * x_c - A_ff^T * x_f^n)
         PetscCall(VecAXPY(air_data->temp_vecs_fine[0].array[our_level], PFLARE_ONE,
                  air_data->temp_vecs_fine[1].array[our_level]));
      }

   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode c_smooths_transpose(Vec b, air_multigrid_data *air_data, int our_level, int its, PetscBool x_f_zero, PetscBool *x_c_zero)
{
   // This applies the transpose of consecutive C smooths - the C point mirror of
   // f_smooths_transpose
   //      x_c = x_c + inv_A_cc^T (b_c - A_cc^T x_c - A_fc^T x_f)

   // ~~~~~~
   int c_its;
   // ~~~~~~

   PetscFunctionBegin;

   // Get out just the coarse points from b - this is b_c
   PetscCall(VecISCopyLocalWrapper(air_data, our_level, PETSC_FALSE, b,
            SCATTER_REVERSE, air_data->temp_vecs_coarse[3].array[our_level], NULL));

   // This is b_c - A_fc^T * x_f - this never changes
   if (!x_f_zero) {
      PetscCall(MatMultTranspose(air_data->A_fc[our_level], air_data->temp_vecs_fine[0].array[our_level],
               air_data->temp_vecs_coarse[1].array[our_level]));
      PetscCall(VecAXPY(air_data->temp_vecs_coarse[3].array[our_level], PFLARE_MINUS_ONE,
               air_data->temp_vecs_coarse[1].array[our_level]));
   }

   // Do all the consecutive C smooths
   for (c_its = 1; c_its <= its; c_its++) {

      // As in f_smooths_transpose, a zero x_c makes the relaxation a single matmult
      if (*x_c_zero) {

         PetscCall(MatMultTranspose(air_data->inv_A_cc[our_level], air_data->temp_vecs_coarse[3].array[our_level],
                  air_data->temp_vecs_coarse[0].array[our_level]));
         *x_c_zero = PETSC_FALSE;

      } else {

         // Then A_cc^T * x_c^n - this changes at each richardson iteration
         PetscCall(MatMultTranspose(air_data->A_cc[our_level], air_data->temp_vecs_coarse[0].array[our_level],
                  air_data->temp_vecs_coarse[2].array[our_level]));

         // This is b_c - A_fc^T * x_f - A_cc^T * x_c^n
         PetscCall(VecAYPX(air_data->temp_vecs_coarse[2].array[our_level], PFLARE_MINUS_ONE,
                  air_data->temp_vecs_coarse[3].array[our_level]));

         // Compute A_cc^{-T} (b_c - A_fc^T * x_f - A_cc^T * x_c^n)
         PetscCall(MatMultTranspose(air_data->inv_A_cc[our_level], air_data->temp_vecs_coarse[2].array[our_level],
                  air_data->temp_vecs_coarse[1].array[our_level]));

         // Compute x_c^n + A_cc^{-T} (b_c - A_fc^T * x_f - A_cc^T * x_c^n)
         PetscCall(VecAXPY(air_data->temp_vecs_coarse[0].array[our_level], PFLARE_ONE,
                  air_data->temp_vecs_coarse[1].array[our_level]));
      }

   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode mg_FC_point_residual_transpose(Mat mat, Vec b, Vec x, Vec r)
{
   // Computes the transposed residual r = b - A^T x on a level, which is what
   // petsc's transposed cycle needs after each transposed smooth
   // The full operator on this level doesn't exist when we're F-C smoothing - mat
   // is the empty placeholder matshell that just tells us which level we're on -
   // so we build the residual out of the blocks of A instead
   //      r_f = b_f - A_ff^T x_f - A_cf^T x_c
   //      r_c = b_c - A_fc^T x_f - A_cc^T x_c
   // The x_c terms are only there when we're doing C point smooths. Without them
   // A_cf and A_cc don't exist, but neither are they needed: the only x this is
   // ever called with is the output of mg_FC_point_apply_transpose, whose coarse
   // half is then identically zero

   // ~~~~~~
   int our_level;
   mat_ctxtype *mat_ctx = NULL;
   air_multigrid_data *air_data;
   // ~~~~~~

   PetscFunctionBegin;

   // Get the level out of the placeholder matshell
   PetscCall(MatShellGetContext(mat, &mat_ctx));
   our_level = mat_ctx->our_level;
   air_data = mat_ctx->air_data;

   // We negate x and use MatMultTransposeAdd throughout, which keeps us to the
   // temporaries the forward smooth already has

   // Get out -x_f
   PetscCall(VecISCopyLocalWrapper(air_data, our_level, PETSC_TRUE, x,
            SCATTER_REVERSE, air_data->temp_vecs_fine[1].array[our_level], NULL));
   PetscCall(VecScale(air_data->temp_vecs_fine[1].array[our_level], PFLARE_MINUS_ONE));

   // Get out b_f and b_c
   PetscCall(VecISCopyLocalWrapper(air_data, our_level, PETSC_TRUE, b,
            SCATTER_REVERSE, air_data->temp_vecs_fine[3].array[our_level], NULL));
   PetscCall(VecISCopyLocalWrapper(air_data, our_level, PETSC_FALSE, b,
            SCATTER_REVERSE, air_data->temp_vecs_coarse[0].array[our_level], NULL));

   // r_f = b_f - A_ff^T * x_f
   PetscCall(MatMultTransposeAdd(air_data->A_ff[our_level], air_data->temp_vecs_fine[1].array[our_level],
            air_data->temp_vecs_fine[3].array[our_level], air_data->temp_vecs_fine[2].array[our_level]));
   // r_c = b_c - A_fc^T * x_f, built up in place
   PetscCall(MatMultTransposeAdd(air_data->A_fc[our_level], air_data->temp_vecs_fine[1].array[our_level],
            air_data->temp_vecs_coarse[0].array[our_level], air_data->temp_vecs_coarse[0].array[our_level]));

   if (air_data->options.any_c_smooths) {

      // Get out -x_c
      PetscCall(VecISCopyLocalWrapper(air_data, our_level, PETSC_FALSE, x,
               SCATTER_REVERSE, air_data->temp_vecs_coarse[1].array[our_level], NULL));
      PetscCall(VecScale(air_data->temp_vecs_coarse[1].array[our_level], PFLARE_MINUS_ONE));

      // r_f = r_f - A_cf^T * x_c
      PetscCall(MatMultTransposeAdd(air_data->A_cf[our_level], air_data->temp_vecs_coarse[1].array[our_level],
               air_data->temp_vecs_fine[2].array[our_level], air_data->temp_vecs_fine[2].array[our_level]));
      // r_c = r_c - A_cc^T * x_c
      PetscCall(MatMultTransposeAdd(air_data->A_cc[our_level], air_data->temp_vecs_coarse[1].array[our_level],
               air_data->temp_vecs_coarse[0].array[our_level], air_data->temp_vecs_coarse[0].array[our_level]));
   }

   // ~~~~~~~~
   // Put both halves back into the full sized residual - every entry of r has to
   // be written, and the scatter of one half preserves whatever the other holds,
   // so zero r first
   // ~~~~~~~~
   PetscCall(VecSet(r, PFLARE_ZERO));
   PetscCall(VecISCopyLocalWrapper(air_data, our_level, PETSC_TRUE, r,
            SCATTER_FORWARD, air_data->temp_vecs_fine[2].array[our_level],
            air_data->temp_vecs[0].array[our_level]));
   PetscCall(VecISCopyLocalWrapper(air_data, our_level, PETSC_FALSE, r,
            SCATTER_FORWARD, air_data->temp_vecs_coarse[0].array[our_level],
            air_data->temp_vecs[0].array[our_level]));

   PetscFunctionReturn(PETSC_SUCCESS);
}
