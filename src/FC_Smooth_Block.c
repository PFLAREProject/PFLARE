/*
  C port of FC_Smooth_Block.F90 (module fc_smooth_block)

  The levels are 0-based, so our_level indexes the per level arrays of
  air_multigrid_data directly (the Fortran our_level is our_level + 1). The
  Kokkos IS views in VecISCopyLocalk.kokkos.cxx number the levels from 1, as the
  Fortran did, so they are given our_level + 1
  The slots of the dense scratch blocks are the 0-based AIR_MAT_* of pflareimpl.h
  The Fortran print + MPI_Abort checks are PetscCheck with the same message
*/

#include <petscksp.h>
#include "pflareimpl.h"
#include "VecISCopyLocalk.h"
#include "PETSc_Helper.h"
#include "Matshell_Data_Type.h"
#include "Gmres_Poly_Apply.h"
#include "FC_Smooth_Block.h"

// The multiple rhs (block) twins of the FC smoothing in fc_smooth, plus the
// lifecycle of the dense scratch blocks they smooth in. Everything here is
// reached through the PCMatApply of PCAIR - the dense blocks of right hand
// sides come down the PCMG hierarchy and every product below is a sparse
// matrix by dense matrix product through the MatProduct API, with the
// products the smooths run kept attached to the per-level scratch

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode create_air_block_temp(Mat op_mat, PetscBool from_cols, Mat x_mat, PetscInt local_cols, PetscInt global_cols, Mat *block_mat)
{
   // Creates one dense scratch block with the row layout of op_mat (or its
   // column layout if from_cols is true) and the column layout (and dense type)
   // of the block of rhs we've been given, x_mat
   // Does nothing if the block already exists

   // ~~~~~~
   PetscInt local_rows, global_rows, local_op_cols, global_op_cols;
   VecType vec_type;
   MPI_Comm MPI_COMM_MATRIX;
   // ~~~~~~

   PetscFunctionBegin;

   if (*block_mat) PetscFunctionReturn(PETSC_SUCCESS);

   PetscCall(PetscObjectGetComm((PetscObject)op_mat, &MPI_COMM_MATRIX));
   PetscCall(MatGetLocalSize(op_mat, &local_rows, &local_op_cols));
   PetscCall(MatGetSize(op_mat, &global_rows, &global_op_cols));
   if (from_cols) {
      local_rows  = local_op_cols;
      global_rows = global_op_cols;
   }
   // Take the type from the block of rhs, so a block that lives on the device
   // gives us device scratch
   PetscCall(MatGetVecType(x_mat, &vec_type));

   PetscCall(MatCreateDenseFromVecType(MPI_COMM_MATRIX, vec_type, local_rows, local_cols,
            global_rows, global_cols, PETSC_DECIDE, NULL,
            block_mat));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode destroy_air_block_temps(air_multigrid_data *air_data)
{
   // Destroys the dense scratch blocks the multiple rhs (block) smooths use
   // ensure_air_block_temps rebuilds any of them we need on the next block apply

   // ~~~~~~
   PetscInt our_level, i_loc;
   // ~~~~~~

   PetscFunctionBegin;

   // The outer arrays are only allocated in create_air_data
   if (!air_data->block_temp_full[0].array) PetscFunctionReturn(PETSC_SUCCESS);

   for (our_level = 0; our_level < air_data->levels_size; our_level++) {

      for (i_loc = 0; i_loc < (PetscInt)PETSC_STATIC_ARRAY_LENGTH(air_data->block_temp_fine); i_loc++) {

         // MatDestroy nulls the slot in the array, so ensure_air_block_temps
         // won't skip rebuilding it (the Fortran had to copy the nulled
         // handle back into the slot)
         if (air_data->block_temp_fine[i_loc].array[our_level]) {
            PetscCall(MatDestroy(&air_data->block_temp_fine[i_loc].array[our_level]));
         }

         if (air_data->block_temp_coarse[i_loc].array[our_level]) {
            PetscCall(MatDestroy(&air_data->block_temp_coarse[i_loc].array[our_level]));
         }
      }

      if (air_data->block_temp_full[0].array[our_level]) {
         PetscCall(MatDestroy(&air_data->block_temp_full[0].array[our_level]));
      }
   }

   air_data->block_ncols = -1;
   air_data->block_local_ncols = -1;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode ensure_air_block_temps(air_multigrid_data *air_data, Mat x_mat)
{
   // Makes sure the cached dense scratch blocks match the block of rhs, x_mat,
   // we've been given
   // This is the air equivalent of ensure_block_temp_mats in matshell_data_type
   // The blocks have the row layouts of the operators on each level (which we
   // only know after the setup) and the column layout of x_mat (which we only
   // know once a block apply happens), hence building them lazily here

   // ~~~~~~
   int our_level, i_loc, n_coarse;
   PetscBool rebuild, same_type;
   PetscInt global_rows, global_cols, local_rows, local_cols;
   MatType x_type, temp_type;
   // ~~~~~~

   PetscFunctionBegin;

   // With full smoothing up and down we never do an FC smooth, so none of these
   // blocks are used - the level smoothers and the coarse grid solver apply
   // their inverses to the whole block directly
   if (air_data->options.full_smoothing_up_and_down) PetscFunctionReturn(PETSC_SUCCESS);
   // Nothing to smooth on if there's only a single level
   if (air_data->no_levels < 2) PetscFunctionReturn(PETSC_SUCCESS);
   if (!air_data->block_temp_full[0].array) PetscFunctionReturn(PETSC_SUCCESS);

   PetscCall(MatGetSize(x_mat, &global_rows, &global_cols));
   PetscCall(MatGetLocalSize(x_mat, &local_rows, &local_cols));
   PetscCall(MatGetType(x_mat, &x_type));

   // If the number of columns has changed we have to start again - the local
   // column layout also has to match or the products would have incompatible
   // layouts
   rebuild = (air_data->block_ncols != global_cols ||
               air_data->block_local_ncols != local_cols) ? PETSC_TRUE : PETSC_FALSE;

   // The type of block we're given can also change between applies
   if (!rebuild) {
      for (our_level = 0; our_level < air_data->no_levels - 1; our_level++) {
         if (air_data->block_temp_fine[0].array[our_level]) {
            PetscCall(MatGetType(air_data->block_temp_fine[0].array[our_level], &temp_type));
            PetscCall(PetscStrcmp(temp_type, x_type, &same_type));
            if (!same_type) rebuild = PETSC_TRUE;
            break;
         }
      }
   }

   if (rebuild) PetscCall(destroy_air_block_temps(air_data));

   // We only need the extra coarse blocks if we're C point smoothing, in the
   // same way as the extra temp_vecs_coarse
   n_coarse = 1;
   if (air_data->options.any_c_smooths) n_coarse = (int)PETSC_STATIC_ARRAY_LENGTH(air_data->block_temp_coarse);

   // Now create any of the blocks we need that don't exist
   for (our_level = 0; our_level < air_data->no_levels - 1; our_level++) {

      for (i_loc = 0; i_loc < (int)PETSC_STATIC_ARRAY_LENGTH(air_data->block_temp_fine); i_loc++) {
         PetscCall(create_air_block_temp(air_data->A_ff[our_level], PETSC_FALSE, x_mat,
                  local_cols, global_cols,
                  &air_data->block_temp_fine[i_loc].array[our_level]));
      }

      // A_fc has the coarse column layout - we take it from A_fc rather than
      // A_cf as A_cf is destroyed after the setup unless we're C smoothing,
      // and this is where temp_vecs_coarse gets its layout from too
      for (i_loc = 0; i_loc < n_coarse; i_loc++) {
         PetscCall(create_air_block_temp(air_data->A_fc[our_level], PETSC_TRUE, x_mat,
                  local_cols, global_cols,
                  &air_data->block_temp_coarse[i_loc].array[our_level]));
      }

      // Only the injector version of MatISCopyLocalWrapper needs a full size
      // block, in the same way as temp_vecs
      if (!air_data->fast_veciscopy_exists) {
         PetscCall(create_air_block_temp(air_data->coarse_matrix[our_level], PETSC_FALSE, x_mat,
                  local_cols, global_cols,
                  &air_data->block_temp_full[0].array[our_level]));
      }

      // The blocks on this level have just been (re)built, so attach the
      // products the block smooths run to their scratch targets - the smooths
      // then only run the numeric phase of each product every iteration
      // rebuild is always true when the blocks didn't exist yet (block_ncols
      // starts and is reset to -1), so the products can't be attached twice
      if (rebuild) PetscCall(setup_air_block_products(air_data, our_level));
   }

   air_data->block_ncols = global_cols;
   air_data->block_local_ncols = local_cols;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode setup_air_block_products(air_multigrid_data *air_data, int our_level)
{
   // Attaches the products the block FC smooths compute on a level to their
   // dense scratch targets, so each smoother iteration only has to run the
   // numeric phase of each product
   // Called by ensure_air_block_temps whenever the scratch on this level has
   // just been (re)built - the products stay attached until the scratch is
   // destroyed, and the scratch outlives every operator it references only
   // briefly (destroy_air_block_temps runs first in reset_air_data)

   // ~~~~~~
   PetscBool any_f_smooths, any_c_smooths, is_shell;
   MatType inv_type;
   PetscInt i_loc;
   // ~~~~~~

   PetscFunctionBegin;

   // any(smooth_order_levels(our_level)%array > 0) and any(... < 0)
   any_f_smooths = PETSC_FALSE;
   any_c_smooths = PETSC_FALSE;
   for (i_loc = 0; i_loc < air_data->smooth_order_levels[our_level].size; i_loc++) {
      if (air_data->smooth_order_levels[our_level].array[i_loc] > 0) any_f_smooths = PETSC_TRUE;
      if (air_data->smooth_order_levels[our_level].array[i_loc] < 0) any_c_smooths = PETSC_TRUE;
   }

   if (any_f_smooths) {

      // A_ff * x_f - computed every F richardson iteration
      PetscCall(attach_mat_product(air_data->A_ff[our_level],
               air_data->block_temp_fine[AIR_MAT_SOL].array[our_level], PETSC_FALSE,
               air_data->block_temp_fine[AIR_MAT_RESIDUAL].array[our_level], NULL));

      // A_fc * x_c - computed once per F smooth
      PetscCall(attach_mat_product(air_data->A_fc[our_level],
               air_data->block_temp_coarse[AIR_MAT_SOL].array[our_level], PETSC_FALSE,
               air_data->block_temp_fine[AIR_MAT_OFF_DIAG].array[our_level], NULL));

      // The assembled (or diagonal) inverses are applied with a product too -
      // the matrix-free matshells go through the blockwise shell apply instead
      // and never have a product attached
      PetscCall(MatGetType(air_data->inv_A_ff[our_level], &inv_type));
      PetscCall(PetscStrcmp(inv_type, MATSHELL, &is_shell));
      if (!is_shell) {
         PetscCall(attach_mat_product(air_data->inv_A_ff[our_level],
                  air_data->block_temp_fine[AIR_MAT_RESIDUAL].array[our_level], PETSC_FALSE,
                  air_data->block_temp_fine[AIR_MAT_TEMP].array[our_level], NULL));
      }
   }

   if (any_c_smooths) {

      // A_cc * x_c - computed every C richardson iteration
      PetscCall(attach_mat_product(air_data->A_cc[our_level],
               air_data->block_temp_coarse[AIR_MAT_SOL].array[our_level], PETSC_FALSE,
               air_data->block_temp_coarse[AIR_MAT_RESIDUAL].array[our_level], NULL));

      // A_cf * x_f - computed once per C smooth
      PetscCall(attach_mat_product(air_data->A_cf[our_level],
               air_data->block_temp_fine[AIR_MAT_SOL].array[our_level], PETSC_FALSE,
               air_data->block_temp_coarse[AIR_MAT_OFF_DIAG].array[our_level], NULL));

      PetscCall(MatGetType(air_data->inv_A_cc[our_level], &inv_type));
      PetscCall(PetscStrcmp(inv_type, MATSHELL, &is_shell));
      if (!is_shell) {
         PetscCall(attach_mat_product(air_data->inv_A_cc[our_level],
                  air_data->block_temp_coarse[AIR_MAT_RESIDUAL].array[our_level], PETSC_FALSE,
                  air_data->block_temp_coarse[AIR_MAT_TEMP].array[our_level], NULL));
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode mat_product_block(Mat mat, Mat x_mat, Mat y_mat, PetscBool transpose_mat)
{
   // Computes y_mat = mat * x_mat (or mat^T * x_mat) for a dense block of
   // right hand sides, ie the multiple rhs version of MatMult/MatMultTranspose
   // The product bookkeeping is always cleared afterwards, or y_mat would keep
   // holding references to mat and x_mat until it is destroyed

   PetscFunctionBegin;

   PetscCall(attach_mat_product(mat, x_mat, transpose_mat, y_mat, NULL));
   PetscCall(MatProductNumeric(y_mat));
   PetscCall(MatProductClear(y_mat));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode shell_block_apply_or_columns(Mat inv_mat, Mat x_mat, Mat y_mat)
{
   // Applies a matrix-free matshell inverse to a dense block of right hand
   // sides - shell_poly_block_apply knows how to do the polynomial matshells
   // blockwise by doing the products with the underlying matrix instead, and
   // reports back if it has been handed something it doesn't know about, in
   // which case we apply column by column

   // ~~~~~~
   PetscBool block_applied;
   PetscInt global_rows, global_cols, i_loc;
   Vec col_x, col_y;
   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(shell_poly_block_apply(inv_mat, x_mat, y_mat, &block_applied));

   // Anything we can't do blockwise has to be applied column by column
   if (!block_applied) {
      PetscCall(MatGetSize(x_mat, &global_rows, &global_cols));
      for (i_loc = 0; i_loc < global_cols; i_loc++) {
         PetscCall(MatDenseGetColumnVecRead(x_mat, i_loc, &col_x));
         PetscCall(MatDenseGetColumnVecWrite(y_mat, i_loc, &col_y));
         PetscCall(MatMult(inv_mat, col_x, col_y));
         PetscCall(MatDenseRestoreColumnVecWrite(y_mat, i_loc, &col_y));
         PetscCall(MatDenseRestoreColumnVecRead(x_mat, i_loc, &col_x));
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode apply_inverse_block(Mat inv_mat, Mat x_mat, Mat y_mat)
{
   // Applies one of our approximate inverses to a dense block of right hand
   // sides, ie the multiple rhs version of MatMult(inv_mat, x, y)
   // The matrix-free polynomial inverses are matshells with only a MATOP_MULT
   // and a MATOP_MULT_TRANSPOSE, so a product on them would fail - they go
   // through the blockwise shell apply instead

   // ~~~~~~
   MatType inv_type;
   PetscBool is_shell;
   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(MatGetType(inv_mat, &inv_type));
   PetscCall(PetscStrcmp(inv_type, MATSHELL, &is_shell));

   if (is_shell) {
      PetscCall(shell_block_apply_or_columns(inv_mat, x_mat, y_mat));

   // Assembled (or diagonal) inverses just do a real product
   } else {
      PetscCall(mat_product_block(inv_mat, x_mat, y_mat, PETSC_FALSE));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode apply_inverse_block_cached(Mat inv_mat, Mat x_mat, Mat y_mat)
{
   // The cached version of apply_inverse_block used inside the block FC
   // smooths - for assembled (or diagonal) inverses the product has already
   // been attached to y_mat by setup_air_block_products, so only the numeric
   // phase runs here. The matrix-free matshells go through the same blockwise
   // shell apply as apply_inverse_block

   // ~~~~~~
   MatType inv_type;
   PetscBool is_shell;
   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(MatGetType(inv_mat, &inv_type));
   PetscCall(PetscStrcmp(inv_type, MATSHELL, &is_shell));

   if (is_shell) {
      PetscCall(shell_block_apply_or_columns(inv_mat, x_mat, y_mat));
   } else {
      PetscCall(MatProductNumeric(y_mat));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode MatISCopyLocalWrapper(air_multigrid_data *air_data, int our_level, PetscBool fine, Mat xfull_mat, ScatterMode mode, Mat xreduced_mat,
   Mat x_temp_full_mat)
{
   // The multiple rhs version of VecISCopyLocalWrapper - pulls the fine or
   // coarse rows out of a dense block (SCATTER_REVERSE) or puts them back
   // (SCATTER_FORWARD)
   // Relies on having pre-built some things with the routine create_VecISCopyLocalWrapper
   // x_temp_full_mat is optional (NULL when absent), it is only used by
   // SCATTER_FORWARD with the injectors

   // ~~~~~~~~~~
#if defined(PETSC_HAVE_KOKKOS)
   int fine_int, mode_int;
   int done_int;
   Mat temp_mat = NULL;
   PetscReal normy;
#endif
   // ~~~~~~~~~~

   PetscFunctionBegin;

   // On gpus without kokkos we can't touch the local arrays, so instead we use
   // the identity restrictors/prolongators built in create_VecISCopyLocalWrapper
   // These are the block versions of the matmults the single rhs version does
   if (!air_data->fast_veciscopy_exists) {

      if (mode == SCATTER_REVERSE) {

         if (fine) {
            PetscCall(mat_product_block(air_data->i_fine_full[our_level], xfull_mat,
                     xreduced_mat, PETSC_FALSE));
         } else {
            PetscCall(mat_product_block(air_data->i_coarse_full[our_level], xfull_mat,
                     xreduced_mat, PETSC_FALSE));
         }

      // SCATTER FORWARD
      } else {

         if (fine) {
            // Copy x but only the non-fine rows of x are non-zero
            // ie get x_c but in a block of full size
            PetscCall(mat_product_block(air_data->i_coarse_full_full[our_level], xfull_mat,
                     x_temp_full_mat, PETSC_FALSE));
            // There is no MatMultTransposeAdd equivalent for a product, so we do
            // the transpose product on its own and add the untouched rows back
            PetscCall(mat_product_block(air_data->i_fine_full[our_level], xreduced_mat,
                     xfull_mat, PETSC_TRUE));
         } else {
            // Copy x but only the non-coarse rows of x are non-zero
            // ie get x_f but in a block of full size
            PetscCall(mat_product_block(air_data->i_fine_full_full[our_level], xfull_mat,
                     x_temp_full_mat, PETSC_FALSE));
            PetscCall(mat_product_block(air_data->i_coarse_full[our_level], xreduced_mat,
                     xfull_mat, PETSC_TRUE));
         }

         PetscCall(MatAXPY(xfull_mat, PFLARE_ONE, x_temp_full_mat, SAME_NONZERO_PATTERN));
      }

   // Otherwise copy the rows we want directly out of the dense arrays
   } else {

#if defined(PETSC_HAVE_KOKKOS)

      // The single rhs version decides by the vec type of the vec it is handed,
      // but we can't do that here - the dense blocks in the mg hierarchy are
      // built by petsc with MatDuplicate, which does not propagate the vec type,
      // so a block can be backed by kokkos data and still report a standard vec
      // type. The device IS views existing is the condition that matters (they
      // are only built for a kokkos mat type in create_VecISCopyLocalWrapper)
      // The blocks are typically host MATDENSE on a host kokkos backend and
      // MATDENSECUDA/HIP on a device one, but the finest level uses the blocks
      // the user gave to KSPMatSolve, which can be host MATDENSE on a device
      // backend - mat_iscopy_local_kokkos tells us if it couldn't use them
      if (air_data->kokkos_is_views_handle) {

         if (mode == SCATTER_REVERSE) {
            mode_int = 1;
         } else {
            mode_int = 0;
         }
         fine_int = 0;
         if (fine) fine_int = 1;

         // SCATTER FORWARD only writes the fine (or coarse) rows of the full
         // block, so we have to keep a copy of it to run the cpu version on
         if (kokkos_debug() && mode != SCATTER_REVERSE) {
            PetscCall(MatDuplicate(xfull_mat, MAT_COPY_VALUES, &temp_mat));
         }

         // The kokkos IS views number the levels from 1
         mat_iscopy_local_kokkos(air_data->kokkos_is_views_handle, our_level + 1, fine_int, &xfull_mat,
                  mode_int, &xreduced_mat, &done_int);
         // If either block is in host memory the default kokkos execution space
         // can't access, the kokkos version doesn't do anything and we have
         // to copy on the host instead
         if (done_int == 0) {
            PetscCall(mat_iscopy_local_host(air_data, our_level, fine, xfull_mat, mode, xreduced_mat));
         }

         // If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) {

            if (mode == SCATTER_REVERSE) {

               PetscCall(MatDuplicate(xreduced_mat, MAT_DO_NOT_COPY_VALUES, &temp_mat));
               PetscCall(mat_iscopy_local_host(air_data, our_level, fine, xfull_mat, mode, temp_mat));
               PetscCall(MatAXPY(temp_mat, PFLARE_MINUS_ONE, xreduced_mat, SAME_NONZERO_PATTERN));

            } else {

               PetscCall(mat_iscopy_local_host(air_data, our_level, fine, temp_mat, mode, xreduced_mat));
               PetscCall(MatAXPY(temp_mat, PFLARE_MINUS_ONE, xfull_mat, SAME_NONZERO_PATTERN));

            }

            PetscCall(MatNorm(temp_mat, NORM_FROBENIUS, &normy));
            PetscCheck(!(normy > PFLARE_TOL_MATFREE_13), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Kokkos and CPU versions of MatISCopyLocalWrapper do not match");
            PetscCall(MatDestroy(&temp_mat));

         }

      } else {
         PetscCall(mat_iscopy_local_host(air_data, our_level, fine, xfull_mat, mode, xreduced_mat));
      }
#else
      PetscCall(mat_iscopy_local_host(air_data, our_level, fine, xfull_mat, mode, xreduced_mat));
#endif
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode mat_iscopy_local_host(air_multigrid_data *air_data, int our_level, PetscBool fine, Mat xfull_mat, ScatterMode mode, Mat xreduced_mat)
{
   // The host version of MatISCopyLocalWrapper - copies the fine or coarse rows
   // we want directly out of the local dense arrays

   // ~~~~~~~~~~
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt i_loc, j_loc, local_row, is_local_size, global_cols;
   PetscInt full_lda, reduced_lda;
   const PetscInt *is_pointer;
   const PetscScalar *full_array_read, *reduced_array_read;
   PetscScalar *full_array, *reduced_array;
   IS is_local;
   // ~~~~~~~~~~

   PetscFunctionBegin;

   if (fine) {
      is_local = air_data->IS_fine_index[our_level];
   } else {
      is_local = air_data->IS_coarse_index[our_level];
   }

   // If we have no fine (or coarse) rows locally there is nothing to copy
   // The reduced block then has no local rows, so petsc hands back a null
   // array for it (in the Fortran an unassociated pointer, which we couldn't
   // take the size of)
   PetscCall(ISGetLocalSize(is_local, &is_local_size));
   if (is_local_size == 0) PetscFunctionReturn(PETSC_SUCCESS);

   // The IS holds global indices
   PetscCall(MatGetOwnershipRange(xfull_mat, &global_row_start, &global_row_end_plus_one));
   PetscCall(ISGetIndices(is_local, &is_pointer));

   // The dense arrays are column-major with the leading dimensions petsc gives us
   // (the petsc fortran interface only handed back a dense array when the leading
   // dimension matched the number of local rows)
   // The reduced block has is_local_size local rows and every global column
   PetscCall(MatGetSize(xreduced_mat, NULL, &global_cols));
   PetscCall(MatDenseGetLDA(xfull_mat, &full_lda));
   PetscCall(MatDenseGetLDA(xreduced_mat, &reduced_lda));
   if (mode == SCATTER_REVERSE) {

      PetscCall(MatDenseGetArrayRead(xfull_mat, &full_array_read));
      PetscCall(MatDenseGetArray(xreduced_mat, &reduced_array));

      for (j_loc = 0; j_loc < global_cols; j_loc++) {
         for (i_loc = 0; i_loc < is_local_size; i_loc++) {
            local_row = is_pointer[i_loc] - global_row_start;
            reduced_array[i_loc + j_loc * reduced_lda] = full_array_read[local_row + j_loc * full_lda];
         }
      }

      PetscCall(MatDenseRestoreArray(xreduced_mat, &reduced_array));
      PetscCall(MatDenseRestoreArrayRead(xfull_mat, &full_array_read));

   // SCATTER FORWARD - only the fine (or coarse) rows of the full block
   // are written, everything else is left alone
   } else {

      PetscCall(MatDenseGetArrayRead(xreduced_mat, &reduced_array_read));
      PetscCall(MatDenseGetArray(xfull_mat, &full_array));

      for (j_loc = 0; j_loc < global_cols; j_loc++) {
         for (i_loc = 0; i_loc < is_local_size; i_loc++) {
            local_row = is_pointer[i_loc] - global_row_start;
            full_array[local_row + j_loc * full_lda] = reduced_array_read[i_loc + j_loc * reduced_lda];
         }
      }

      PetscCall(MatDenseRestoreArray(xfull_mat, &full_array));
      PetscCall(MatDenseRestoreArrayRead(xreduced_mat, &reduced_array_read));

   }

   PetscCall(ISRestoreIndices(is_local, &is_pointer));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode mg_coarse_shell_block_apply(PC pc, Mat x, Mat y)
{
   // The multiple rhs version of mg_coarse_shell_apply - applies our polynomial
   // approximate inverse of the coarse matrix to a whole dense block

   // ~~~~~~
   air_multigrid_data *air_data = NULL;
   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(PCShellGetContext(pc, &air_data));
   PetscCall(apply_inverse_block(air_data->inv_A_ff[air_data->no_levels - 1], x, y));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode mg_smooth_shell_block_apply(PC pc, Mat x, Mat y)
{
   // The multiple rhs version of mg_smooth_shell_apply

   // ~~~~~~
   Mat mat, pmat;
   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(PCGetOperators(pc, &mat, &pmat));
   PetscCall(apply_inverse_block(pmat, x, y));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode mg_FC_block_richardson(PC pc, Mat b, Mat x, Mat r, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt maxits,
   PetscBool guess_zero, PetscInt *its, PCRichardsonConvergedReason *conv_reason)
{
   // The multiple rhs version of mg_FC_point_richardson - applies an FC block
   // richardson to a whole dense block of right hand sides
   // r is the block of work vectors and may be null, we never use it (just like
   // the single rhs version never uses its work vector)

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

         PetscCall(f_smooths_block(b, x, guess_zero, first_smooth, air_data, our_level, smooth_its));

      // Do consecutive C point smooths
      } else {

         PetscCall(c_smooths_block(b, x, guess_zero, first_smooth, air_data, our_level, PetscAbsInt(smooth_its)));
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

PETSC_INTERN PetscErrorCode f_smooths_block(Mat b, Mat x, PetscBool guess_zero, PetscBool first_smooth, air_multigrid_data *air_data, int our_level, int its)
{
   // The multiple rhs version of f_smooths - applies consecutive F smooths to a
   // whole dense block of right hand sides
   // This is the same arithmetic as f_smooths with the matvecs replaced by
   // sparse matrix - dense matrix products

   // ~~~~~~
   int f_its;
   // ~~~~~~

   PetscFunctionBegin;

   (void)guess_zero;

   // Get out just the fine rows from b - this is b_f
   PetscCall(MatISCopyLocalWrapper(air_data, our_level, PETSC_TRUE, b,
            SCATTER_REVERSE, air_data->block_temp_fine[AIR_MAT_RHS].array[our_level], NULL));

   // If we haven't done any smooth before calling this F point smooth
   // we need to pull out x_c^0 and x_f^0
   if (first_smooth) {

      // Get out just the fine rows from x - this is x_f^0
      PetscCall(MatISCopyLocalWrapper(air_data, our_level, PETSC_TRUE, x,
               SCATTER_REVERSE, air_data->block_temp_fine[AIR_MAT_SOL].array[our_level], NULL));

      // Get the coarse rows from x - this is x_c^0
      PetscCall(MatISCopyLocalWrapper(air_data, our_level, PETSC_FALSE, x,
               SCATTER_REVERSE, air_data->block_temp_coarse[AIR_MAT_SOL].array[our_level], NULL));

   }

   // Compute Afc * x_c^0 - this never changes
   // The product was attached to the scratch by setup_air_block_products so
   // only the numeric phase runs, here and below
   PetscCall(MatProductNumeric(air_data->block_temp_fine[AIR_MAT_OFF_DIAG].array[our_level]));

   // This is b_f - A_fc * x_c^0 - this never changes
   PetscCall(MatAXPY(air_data->block_temp_fine[AIR_MAT_RHS].array[our_level], PFLARE_MINUS_ONE,
            air_data->block_temp_fine[AIR_MAT_OFF_DIAG].array[our_level], SAME_NONZERO_PATTERN));

   // Do all the consecutive F smooths
   for (f_its = 1; f_its <= its; f_its++) {

      // Then A_ff * x_f^n - this changes at each richardson iteration
      PetscCall(MatProductNumeric(air_data->block_temp_fine[AIR_MAT_RESIDUAL].array[our_level]));

      // This is b_f - A_fc * x_c - A_ff * x_f^n
      PetscCall(MatAYPX(air_data->block_temp_fine[AIR_MAT_RESIDUAL].array[our_level], PFLARE_MINUS_ONE,
               air_data->block_temp_fine[AIR_MAT_RHS].array[our_level], SAME_NONZERO_PATTERN));

      // ! Compute A_ff^{-1} ( b_f - A_fc * x_c - A_ff * x_f^n)
      PetscCall(apply_inverse_block_cached(air_data->inv_A_ff[our_level],
               air_data->block_temp_fine[AIR_MAT_RESIDUAL].array[our_level],
               air_data->block_temp_fine[AIR_MAT_TEMP].array[our_level]));

      // Compute x_f^n + A_ff^{-1} ( b_f - A_fc * x_c - A_ff * x_f^n)
      PetscCall(MatAXPY(air_data->block_temp_fine[AIR_MAT_SOL].array[our_level], PFLARE_ONE,
               air_data->block_temp_fine[AIR_MAT_TEMP].array[our_level], SAME_NONZERO_PATTERN));

   }

   // ~~~~~~~~
   // Reverse put fine x_f back into x
   // ~~~~~~~~
   PetscCall(MatISCopyLocalWrapper(air_data, our_level, PETSC_TRUE, x,
            SCATTER_FORWARD, air_data->block_temp_fine[AIR_MAT_SOL].array[our_level],
            air_data->block_temp_full[0].array[our_level]));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode c_smooths_block(Mat b, Mat x, PetscBool guess_zero, PetscBool first_smooth, air_multigrid_data *air_data, int our_level, int its)
{
   // The multiple rhs version of c_smooths - applies consecutive C smooths to a
   // whole dense block of right hand sides

   // ~~~~~~
   int c_its;
   // ~~~~~~

   PetscFunctionBegin;

   (void)guess_zero;

   // Get out just the coarse rows from b - this is b_c
   PetscCall(MatISCopyLocalWrapper(air_data, our_level, PETSC_FALSE, b,
            SCATTER_REVERSE, air_data->block_temp_coarse[AIR_MAT_RHS].array[our_level], NULL));

   // If we haven't done any smooth before calling this C point smooth
   // we need to pull out x_c^0 and x_f^0
   if (first_smooth) {

         // Get out just the fine rows from x - this is x_f^0
      PetscCall(MatISCopyLocalWrapper(air_data, our_level, PETSC_TRUE, x,
               SCATTER_REVERSE, air_data->block_temp_fine[AIR_MAT_SOL].array[our_level], NULL));

      // Get the coarse rows from x - this is x_c^0
      PetscCall(MatISCopyLocalWrapper(air_data, our_level, PETSC_FALSE, x,
               SCATTER_REVERSE, air_data->block_temp_coarse[AIR_MAT_SOL].array[our_level], NULL));

   }

   // Compute Acf * x_f^0 - this never changes
   // The product was attached to the scratch by setup_air_block_products so
   // only the numeric phase runs, here and below
   PetscCall(MatProductNumeric(air_data->block_temp_coarse[AIR_MAT_OFF_DIAG].array[our_level]));
   // This is b_c - A_cf * x_f^0 - this never changes
   PetscCall(MatAXPY(air_data->block_temp_coarse[AIR_MAT_RHS].array[our_level], PFLARE_MINUS_ONE,
            air_data->block_temp_coarse[AIR_MAT_OFF_DIAG].array[our_level], SAME_NONZERO_PATTERN));

   // Do all the consecutive C smooths
   for (c_its = 1; c_its <= its; c_its++) {

      // Then A_cc * x_c^n - this changes at each richardson iteration
      PetscCall(MatProductNumeric(air_data->block_temp_coarse[AIR_MAT_RESIDUAL].array[our_level]));

      // This is b_c - A_cf * x_f^0 - A_cc * x_c^n
      PetscCall(MatAYPX(air_data->block_temp_coarse[AIR_MAT_RESIDUAL].array[our_level], PFLARE_MINUS_ONE,
               air_data->block_temp_coarse[AIR_MAT_RHS].array[our_level], SAME_NONZERO_PATTERN));

      // ! Compute A_cc^{-1} (b_c - A_cf * x_f^0 - A_cc * x_c^n)
      PetscCall(apply_inverse_block_cached(air_data->inv_A_cc[our_level],
               air_data->block_temp_coarse[AIR_MAT_RESIDUAL].array[our_level],
               air_data->block_temp_coarse[AIR_MAT_TEMP].array[our_level]));

      // Compute x_c^n + A_cc^{-1} (b_c - A_cf * x_f^0 - A_cc * x_c^n)
      PetscCall(MatAXPY(air_data->block_temp_coarse[AIR_MAT_SOL].array[our_level], PFLARE_ONE,
               air_data->block_temp_coarse[AIR_MAT_TEMP].array[our_level], SAME_NONZERO_PATTERN));

   }

   // ~~~~~~~~
   // Reverse put coarse x_c back into x
   // ~~~~~~~~
   PetscCall(MatISCopyLocalWrapper(air_data, our_level, PETSC_FALSE, x,
            SCATTER_FORWARD, air_data->block_temp_coarse[AIR_MAT_SOL].array[our_level],
            air_data->block_temp_full[0].array[our_level]));

   PetscFunctionReturn(PETSC_SUCCESS);
}
