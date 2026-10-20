/*
  C port of Matshell_Data_Type.F90 (module matshell_data_type)

  The mat_ctxtype context of our matshells is the struct in pflareimpl.h.
  create_mat_ctx gives the defaults of the Fortran type (the Fortran
  allocate), and the context is freed with PetscFree (the Fortran deallocate).
  The slots of mf_temp_vec and mf_temp_mat are the 0-based MF_VEC_* and
  MF_MAT_* of pflareimpl.h

  The three accessors at the end are for the Fortran that is still to be
  ported (AIR_MG_Setup.F90 and the FC smoothers), which used to allocate the
  context of its level matshells and read our_level and air_data from it
  directly
*/

#include <petscmat.h>
#include "pflareimpl.h"
#include "Matshell_Data_Type.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode create_mat_ctx(mat_ctxtype **mat_ctx)
{
   // Allocates a matshell context with the defaults of the Fortran type

   PetscFunctionBegin;

   // Everything else is zero, NULL or false
   PetscCall(PetscNew(mat_ctx));
   (*mat_ctx)->our_level = -1;
   (*mat_ctx)->own_coefficients = PETSC_FALSE;
   (*mat_ctx)->neumann_inner = PETSC_FALSE;
   (*mat_ctx)->mf_temp_mat_ncols = -1;
   (*mat_ctx)->mf_products_attached = PETSC_FALSE;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode ensure_block_temp_mats(mat_ctxtype *mat_ctx, Mat x_mat, PetscInt n_temps, const PetscBool *need_rhs)
{
   // Makes sure the cached dense scratch blocks in mat_ctx->mf_temp_mat
   // match the block of rhs, x_mat, we've been given
   // Slots 1 to n_temps are created (MF_MAT_TEMP, MF_MAT_TEMP_TWO, MF_MAT_TEMP_THREE)
   // and the optional need_rhs also creates the MF_MAT_RHS slot, which is where the
   // diagonally scaled block applies store D^-1 X
   // We can't just always create all four, as the number of temporaries needed
   // depends on the polynomial being applied and these blocks aren't small

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   // Local
   PetscInt i_loc;
   PetscBool rebuild, want_rhs, same_type;
   PetscInt global_rows, global_cols;
   MatType x_type, temp_type;
   Mat temp_mat_local;

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   PetscFunctionBegin;

   want_rhs = PETSC_FALSE;
   if (need_rhs) want_rhs = *need_rhs;

   PetscCall(MatGetSize(x_mat, &global_rows, &global_cols));
   PetscCall(MatGetType(x_mat, &x_type));

   // If the number of columns has changed we have to start again
   rebuild = mat_ctx->mf_temp_mat_ncols != global_cols ? PETSC_TRUE : PETSC_FALSE;

   // The type of block we're given can also change between applies
   if (!rebuild) {
      for (i_loc = 0; i_loc < 4; i_loc++) {
         if (mat_ctx->mf_temp_mat[i_loc]) {
            PetscCall(MatGetType(mat_ctx->mf_temp_mat[i_loc], &temp_type));
            PetscCall(PetscStrcmp(temp_type, x_type, &same_type));
            if (!same_type) rebuild = PETSC_TRUE;
            break;
         }
      }
   }

   if (rebuild) {
      for (i_loc = 0; i_loc < 4; i_loc++) {
         if (mat_ctx->mf_temp_mat[i_loc]) {
            // The block kernels attach products that hold references between the
            // scratch mats - the horner ping-pong pair reference each other, so
            // plain destroys would leave a reference cycle alive - clear the
            // products first to break it
            // The parallel aij x dense numeric attaches its own inner product to
            // the local dense block, which recreates the same cycle between the
            // local blocks - clear that one too (for seq dense the local matrix
            // is the mat itself, so this is just a repeat clear)
            PetscCall(MatProductClear(mat_ctx->mf_temp_mat[i_loc]));
            PetscCall(MatDenseGetLocalMatrix(mat_ctx->mf_temp_mat[i_loc], &temp_mat_local));
            PetscCall(MatProductClear(temp_mat_local));
            // MatDestroy nulls the slot in the context, so the creation below
            // won't skip it
            PetscCall(MatDestroy(&mat_ctx->mf_temp_mat[i_loc]));
         }
      }
      mat_ctx->mf_temp_mat_ncols = -1;
      // Any products the block apply kernels had attached died with the
      // scratch, so they have to attach them again
      mat_ctx->mf_products_attached = PETSC_FALSE;
   }

   // Now create any of the slots we need that don't exist
   for (i_loc = 0; i_loc < n_temps; i_loc++) {
      if (!mat_ctx->mf_temp_mat[i_loc]) {
         PetscCall(MatDuplicate(x_mat, MAT_DO_NOT_COPY_VALUES, &mat_ctx->mf_temp_mat[i_loc]));
      }
   }
   if (want_rhs) {
      if (!mat_ctx->mf_temp_mat[MF_MAT_RHS]) {
         PetscCall(MatDuplicate(x_mat, MAT_DO_NOT_COPY_VALUES, &mat_ctx->mf_temp_mat[MF_MAT_RHS]));
      }
   }

   mat_ctx->mf_temp_mat_ncols = global_cols;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode attach_mat_product(Mat mat, Mat x_mat, PetscBool transpose_mat, Mat y_mat, PetscBool *has_product)
{
   // Attaches y_mat = mat * x_mat (or mat^T * x_mat) as a product and runs the
   // symbolic phase, leaving the product bookkeeping alive on y_mat so repeat
   // computations only have to run MatProductNumeric
   // y_mat therefore keeps references to mat and x_mat until the product is
   // cleared or y_mat is destroyed
   // If has_product is not NULL it comes back false (with the product cleared
   // and no symbolic run) when there is no product implementation for these
   // types and the caller has to fall back; without it the product is assumed
   // to exist
   // This lives here rather than in petsc_helper as this file is compiled
   // before petsc_helper

   // ~~~~~~
   PetscBool has_op;
   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(MatProductCreateWithMat(mat, x_mat, NULL, y_mat));
   if (transpose_mat) {
      PetscCall(MatProductSetType(y_mat, MATPRODUCT_AtB));
   } else {
      PetscCall(MatProductSetType(y_mat, MATPRODUCT_AB));
   }
   PetscCall(MatProductSetFromOptions(y_mat));
   if (has_product) {
      PetscCall(MatHasOperation(y_mat, MATOP_PRODUCTSYMBOLIC, &has_op));
      *has_product = has_op;
      if (!*has_product) {
         PetscCall(MatProductClear(y_mat));
         PetscFunctionReturn(PETSC_SUCCESS);
      }
   }
   PetscCall(MatProductSymbolic(y_mat));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode ensure_block_kernel_products(mat_ctxtype *mat_ctx, PetscInt second_target, PetscBool *attached)
{
   // Makes sure the two products a matrix-free block apply kernel runs are
   // attached to the dense temporaries in the context:
   //    mf_temp_mat(MF_MAT_TEMP_TWO) = mat * mf_temp_mat(MF_MAT_TEMP)
   //    mf_temp_mat(second_target)   = mat * mf_temp_mat(MF_MAT_TEMP_TWO)
   // The horner kernel ping-pongs between its two temporaries so its second
   // target is MF_MAT_TEMP, while the newton kernel's is MF_MAT_TEMP_THREE
   // The products stay attached between applies - they die with the scratch
   // whenever ensure_block_temp_mats rebuilds it (which resets
   // mf_products_attached), and are rebuilt here if the mat in the context has
   // changed identity (the attached products keep the old mat alive, so the
   // handle comparison is safe)

   // attached comes back false if there is no product for these types and the
   // caller has to fall back to a column by column apply

   // ~~~~~~
   PetscBool has_product;
   // ~~~~~~

   PetscFunctionBegin;

   *attached = PETSC_TRUE;

   // If the mat has been swapped out from under us the attached products
   // would still run with the old one - clear them and attach again
   if (mat_ctx->mf_products_attached) {
      if (mat_ctx->mf_product_mat != mat_ctx->mat) {
         PetscCall(MatProductClear(mat_ctx->mf_temp_mat[MF_MAT_TEMP_TWO]));
         PetscCall(MatProductClear(mat_ctx->mf_temp_mat[second_target]));
         mat_ctx->mf_products_attached = PETSC_FALSE;
      }
   }

   if (!mat_ctx->mf_products_attached) {

      // temp_two = mat * temp - if there is no product available for these
      // types the caller does a column by column apply instead
      PetscCall(attach_mat_product(mat_ctx->mat, mat_ctx->mf_temp_mat[MF_MAT_TEMP], PETSC_FALSE,
               mat_ctx->mf_temp_mat[MF_MAT_TEMP_TWO], &has_product));
      if (!has_product) {
         *attached = PETSC_FALSE;
         PetscFunctionReturn(PETSC_SUCCESS);
      }

      // second_target = mat * temp_two
      // The types are the same as the product above, so we know this one exists
      PetscCall(attach_mat_product(mat_ctx->mat, mat_ctx->mf_temp_mat[MF_MAT_TEMP_TWO], PETSC_FALSE,
               mat_ctx->mf_temp_mat[second_target], NULL));

      mat_ctx->mf_products_attached = PETSC_TRUE;
      mat_ctx->mf_product_mat = mat_ctx->mat;
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------
// The accessors for the Fortran that is still to be ported
// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode mat_ctx_create_shell(Mat comm_mat, PetscInt local_rows, PetscInt local_cols, PetscInt global_rows, PetscInt global_cols,
   int our_level, void *air_data, Mat *mat)
{
   // Creates an (empty) matshell whose context records the level it is on and
   // the Fortran air_multigrid_data, which is what AIR_MG_Setup.F90 gives the
   // FC smoothers to tell what level they're on
   // This replaces the Fortran allocate of the context and MatCreateShell
   // The shell is created on the comm of comm_mat (the Fortran passed the comm,
   // which would need converting from a Fortran handle)

   // ~~~~~~
   mat_ctxtype *mat_ctx;
   MPI_Comm comm;
   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)comm_mat, &comm));
   PetscCall(create_mat_ctx(&mat_ctx));
   mat_ctx->our_level = our_level;
   // The Fortran air_multigrid_data, held in the typed pointer of the C context
   mat_ctx->air_data = (air_multigrid_data *)air_data;
   PetscCall(MatCreateShell(comm, local_rows, local_cols, global_rows, global_cols, mat_ctx, mat));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode mat_ctx_get_our_level(Mat mat, int *our_level)
{
   // Returns our_level from the context of a matshell

   // ~~~~~~
   mat_ctxtype *mat_ctx;
   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(MatShellGetContext(mat, &mat_ctx));
   *our_level = mat_ctx->our_level;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode mat_ctx_get_air_data(Mat mat, void **air_data)
{
   // Returns the air_data pointer from the context of a matshell

   // ~~~~~~
   mat_ctxtype *mat_ctx;
   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(MatShellGetContext(mat, &mat_ctx));
   *air_data = (void *)mat_ctx->air_data;

   PetscFunctionReturn(PETSC_SUCCESS);
}
