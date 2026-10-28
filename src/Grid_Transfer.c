/*
  C port of Grid_Transfer.F90 (module grid_transfer)

  The matrices the Fortran returned in an intent(inout) handle are Mat *, as is
  the IS orig_fine_col_indices that compute_R_from_Z may create. The loops over
  the local rows and the row entries are 0-based, so the index sets and colmap
  are indexed without the Fortran +1
*/

#include <petscmat.h>
#include "pflareimpl.h"
#include "Grid_Transferk.h"
#include "PETSc_Helper.h"
#include "Grid_Transfer.h"

// -------------------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------------------------------------
// Grid transfer routines
// -------------------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode generate_one_point_with_one_entry_from_sparse(Mat input_mat, Mat *output_mat)
{
   // Wrapper around generate_one_point_with_one_entry_from_sparse_kokkos and
   // generate_one_point_with_one_entry_from_sparse_cpu

   // ~~~~~~~~~~
#if defined(PETSC_HAVE_KOKKOS)
   Mat A_array, B_array = NULL;
   MatType mat_type;
   Mat temp_mat = NULL;
   PetscReal normy;
   Vec max_vec;
   PetscInt row_loc;
   PetscBool is_kokkos;
#endif
   // ~~~~~~~~~~

   PetscFunctionBegin;

#if defined(PETSC_HAVE_KOKKOS)

   PetscCall(MatGetType(input_mat, &mat_type));
   PetscCall(PetscStrcmpAny(mat_type, &is_kokkos, MATMPIAIJKOKKOS, MATSEQAIJKOKKOS, MATAIJKOKKOS, ""));
   if (is_kokkos) {

      A_array = input_mat;
      generate_one_point_with_one_entry_from_sparse_kokkos(&A_array, &B_array);
      *output_mat = B_array;

      // If debugging do a comparison between CPU and Kokkos results
      if (kokkos_debug()) {

         // Debug check if the CPU and Kokkos versions are the same
         PetscCall(generate_one_point_with_one_entry_from_sparse_cpu(input_mat, &temp_mat));

         PetscCall(MatAXPY(temp_mat, PFLARE_MINUS_ONE, *output_mat, DIFFERENT_NONZERO_PATTERN));
         // Find the biggest entry in the difference
         PetscCall(MatCreateVecs(temp_mat, NULL, &max_vec));
         PetscCall(MatGetRowMaxAbs(temp_mat, max_vec, NULL));
         PetscCall(VecMax(max_vec, &row_loc, &normy));
         PetscCall(VecDestroy(&max_vec));

         if (normy > PFLARE_TOL_MATFREE_13 || PetscIsNanReal(normy)) {
            //call MatFilter(temp_mat, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
            //call MatView(temp_mat, PETSC_VIEWER_STDOUT_WORLD, ierr)
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Diff Kokkos and CPU generate_one_point_with_one_entry_from_sparse %g row %" PetscInt_FMT, (double)normy, row_loc);
         }
         PetscCall(MatDestroy(&temp_mat));
      }

   } else {

      PetscCall(generate_one_point_with_one_entry_from_sparse_cpu(input_mat, output_mat));

   }
#else
   PetscCall(generate_one_point_with_one_entry_from_sparse_cpu(input_mat, output_mat));
#endif

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode generate_one_point_with_one_entry_from_sparse_cpu(Mat input_mat, Mat *output_mat)
{
   // Returns a copy of a sparse matrix, but with only one in the spot of the biggest entry
   // This can be used to generate a classical one point prolongator for example

   // ~~~~~~~~~~
   PetscInt ncols, ifree, i_loc, j_loc;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt global_col_start, global_col_end_plus_one;
   PetscInt max_local_col, max_nonlocal_col;
   // The column of the biggest entry in each local row (-1 if the row is empty)
   PetscInt *biggest_cols, *d_nnz, *o_nnz;
   const PetscInt *cols = NULL;
   // Matrix entries filled by MatGetRow are PetscScalar
   const PetscScalar *vals = NULL;
   PetscInt one = 1;
   PetscInt max_loc;
   PetscMPIInt comm_size;
   MPI_Comm MPI_COMM_MATRIX;
   MatType mat_type;
   const PetscInt *colmap = NULL;
   Mat Ad, Ao = NULL;
   // Derived magnitudes stay PetscReal
   PetscReal max_local_val, max_nonlocal_val;
   // ~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)input_mat, &MPI_COMM_MATRIX));
   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));

   // Get the local sizes
   PetscCall(MatGetLocalSize(input_mat, &local_rows, &local_cols));
   PetscCall(MatGetSize(input_mat, &global_rows, &global_cols));
   // This returns the global index of the local portion of the matrix
   PetscCall(MatGetOwnershipRange(input_mat, &global_row_start, &global_row_end_plus_one));
   PetscCall(MatGetOwnershipRangeColumn(input_mat, &global_col_start, &global_col_end_plus_one));

   // We know we only have one entry per row
   // (at least one entry is allocated, as a Fortran zero-size array has an address)
   PetscCall(PetscMalloc1(PetscMax(local_rows, 1), &biggest_cols));
   for (i_loc = 0; i_loc < local_rows; i_loc++) biggest_cols[i_loc] = -1;

   // Get the local/nonlocal components
   if (comm_size != 1) {
      PetscCall(MatMPIAIJGetSeqAIJ(input_mat, &Ad, &Ao, &colmap));
   } else {
      Ad = input_mat;
   }

   // Find the biggest entry in each row
   // Loop over global row indices
   for (ifree = global_row_start; ifree < global_row_end_plus_one; ifree++) {

      // huge() of the PetscReal target (not 0d0) so single builds get the
      // representable -huge, not -Inf (which traps under -ffpe-trap=overflow)
      max_local_val = -PETSC_MAX_REAL;
      max_nonlocal_val = -PETSC_MAX_REAL;
      max_local_col = -1;
      max_nonlocal_col = -1;

      // Let's get the biggest local component
      PetscCall(MatGetRow(Ad, ifree - global_row_start, &ncols, &cols, &vals));
      if (ncols != 0) {
         // maxloc(abs(vals(1:ncols))): the first entry of largest magnitude, skipping
         // NaNs as gfortran's maxloc does (which gives the first entry if all are NaN)
         max_loc = 0;
         for (j_loc = 0; j_loc < ncols; j_loc++) {
            if (!PetscIsNanReal(PetscAbsScalar(vals[j_loc]))) {
               max_loc = j_loc;
               break;
            }
         }
         for (j_loc = max_loc + 1; j_loc < ncols; j_loc++) {
            if (PetscAbsScalar(vals[j_loc]) > PetscAbsScalar(vals[max_loc])) max_loc = j_loc;
         }
         max_local_val = PetscAbsScalar(vals[max_loc]);
         // Global column index
         max_local_col = cols[max_loc] + global_col_start;
      }
      PetscCall(MatRestoreRow(Ad, ifree - global_row_start, &ncols, &cols, &vals));

      // And the biggest non local component
      if (comm_size != 1) {
         PetscCall(MatGetRow(Ao, ifree - global_row_start, &ncols, &cols, &vals));
         if (ncols != 0) {
            // maxloc(abs(vals(1:ncols))), as above
            max_loc = 0;
            for (j_loc = 0; j_loc < ncols; j_loc++) {
               if (!PetscIsNanReal(PetscAbsScalar(vals[j_loc]))) {
                  max_loc = j_loc;
                  break;
               }
            }
            for (j_loc = max_loc + 1; j_loc < ncols; j_loc++) {
               if (PetscAbsScalar(vals[j_loc]) > PetscAbsScalar(vals[max_loc])) max_loc = j_loc;
            }
            max_nonlocal_val = PetscAbsScalar(vals[max_loc]);
            // Global column index - need colmap
            max_nonlocal_col = colmap[cols[max_loc]];
         }
         PetscCall(MatRestoreRow(Ao, ifree - global_row_start, &ncols, &cols, &vals));
      }

      // If there is a bigger nonlocal entry we have to use that
      // If the biggest entry is local or the biggest local or nonlocal entries are equal
      // we use the local entry
      if (max_nonlocal_val > max_local_val) max_local_col = max_nonlocal_col;

      biggest_cols[ifree - global_row_start] = max_local_col;
   }

   // Create the output matrix, exactly preallocated
   PetscCall(PetscCalloc1(PetscMax(local_rows, 1), &d_nnz));
   PetscCall(PetscCalloc1(PetscMax(local_rows, 1), &o_nnz));
   for (i_loc = 0; i_loc < local_rows; i_loc++) {
      if (biggest_cols[i_loc] == -1) continue;
      if (biggest_cols[i_loc] >= global_col_start && biggest_cols[i_loc] < global_col_end_plus_one) {
         d_nnz[i_loc] = 1;
      } else {
         o_nnz[i_loc] = 1;
      }
   }

   PetscCall(MatCreate(MPI_COMM_MATRIX, output_mat));
   PetscCall(MatSetSizes(*output_mat, local_rows, local_cols,
                    global_rows, global_cols));
   // Match the output type
   PetscCall(MatGetType(input_mat, &mat_type));
   PetscCall(MatSetType(*output_mat, mat_type));
   PetscCall(MatXAIJSetPreallocation(*output_mat, one, d_nnz, o_nnz,
            NULL, NULL));
   PetscCall(PetscFree(d_nnz));
   PetscCall(PetscFree(o_nnz));

   // Don't set any off processor entries so no need for a reduction when assembling
   PetscCall(MatSetOption(*output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
   PetscCall(MatSetOption(*output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));

   for (i_loc = 0; i_loc < local_rows; i_loc++) {
      if (biggest_cols[i_loc] == -1) continue;
      PetscCall(MatSetValue(*output_mat, global_row_start + i_loc, biggest_cols[i_loc],
               PFLARE_ONE, INSERT_VALUES));
   }
   PetscCall(PetscFree(biggest_cols));
   PetscCall(MatAssemblyBegin(*output_mat, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(*output_mat, MAT_FINAL_ASSEMBLY));

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode compute_P_from_W(Mat W, PetscInt global_row_start, IS is_fine, IS is_coarse,
   PetscBool identity, PetscBool reuse, Mat *P)
{
   // Wrapper around compute_P_from_W_cpu and compute_P_from_W_kokkos

   // ~~~~~~~~~~
#if defined(PETSC_HAVE_KOKKOS)
   Mat A_array, B_array = NULL;
   IS indices_fine, indices_coarse;
   int identity_int, reuse_int;
   MatType mat_type;
   Mat temp_mat = NULL, temp_mat_reuse = NULL, temp_mat_compare = NULL;
   PetscReal normy;
   Vec max_vec;
   PetscInt row_loc;
   PetscBool is_kokkos;
#endif
   // ~~~~~~~~~~

   PetscFunctionBegin;

#if defined(PETSC_HAVE_KOKKOS)

   PetscCall(MatGetType(W, &mat_type));
   PetscCall(PetscStrcmpAny(mat_type, &is_kokkos, MATMPIAIJKOKKOS, MATSEQAIJKOKKOS, MATAIJKOKKOS, ""));
   if (is_kokkos) {

      identity_int = 0;
      if (identity) identity_int = 1;
      reuse_int = 0;
      if (reuse) reuse_int = 1;

      A_array = W;
      indices_fine = is_fine;
      indices_coarse = is_coarse;
      if (reuse) B_array = *P;
      compute_P_from_W_kokkos(&A_array, global_row_start,
                  &indices_fine, &indices_coarse,
                  identity_int, reuse_int, &B_array);
      *P = B_array;

      // If debugging do a comparison between CPU and Kokkos results
      if (kokkos_debug()) {

         // If we're doing reuse and debug, then the cpu version reuses the output
         // matrix itself (as that has the sparsity the cpu version expects) and so
         // we compare against a copy of the kokkos result
         if (reuse) {
            temp_mat = *P;
            PetscCall(MatConvert(*P, MATSAME, MAT_INITIAL_MATRIX, &temp_mat_compare));
         } else {
            temp_mat_compare = *P;
         }

         // Debug check if the CPU and Kokkos versions are the same
         PetscCall(compute_P_from_W_cpu(W, global_row_start, is_fine, is_coarse,
                  identity, reuse, &temp_mat));

         PetscCall(MatConvert(temp_mat, MATSAME, MAT_INITIAL_MATRIX,
                     &temp_mat_reuse));

         PetscCall(MatAXPY(temp_mat_reuse, PFLARE_MINUS_ONE, temp_mat_compare, DIFFERENT_NONZERO_PATTERN));
         // Find the biggest entry in the difference
         PetscCall(MatCreateVecs(temp_mat_reuse, NULL, &max_vec));
         PetscCall(MatGetRowMaxAbs(temp_mat_reuse, max_vec, NULL));
         PetscCall(VecMax(max_vec, &row_loc, &normy));
         PetscCall(VecDestroy(&max_vec));

         if (normy > PFLARE_TOL_MATFREE_13 || PetscIsNanReal(normy)) {
            //call MatFilter(temp_mat_reuse, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
            //call MatView(temp_mat_reuse, PETSC_VIEWER_STDOUT_WORLD, ierr)
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Diff Kokkos and CPU compute_P_from_W %g row %" PetscInt_FMT, (double)normy, row_loc);
         }
         PetscCall(MatDestroy(&temp_mat_reuse));
         if (!reuse) {
            PetscCall(MatDestroy(P));
         } else {
            PetscCall(MatDestroy(&temp_mat_compare));
         }
         *P = temp_mat;

      }

   } else {

      PetscCall(compute_P_from_W_cpu(W, global_row_start, is_fine, is_coarse,
               identity, reuse, P));

   }
#else
   PetscCall(compute_P_from_W_cpu(W, global_row_start, is_fine, is_coarse,
               identity, reuse, P));
#endif

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode compute_P_from_W_cpu(Mat W, PetscInt global_row_start, IS is_fine, IS is_coarse,
   PetscBool identity, PetscBool reuse, Mat *P)
{
   // Pass in W and get out P = [W I]' (or [W 0] if identity is false)

   // ~~~~~~~~~~
   PetscInt global_row_start_W, global_row_end_plus_one_W;
   PetscInt global_col_start_W, global_col_end_plus_one_W;
   PetscInt local_rows_coarse, local_rows, local_cols, local_cols_coarse;
   PetscInt i_loc, ncols, col, p_row;
   PetscInt global_cols, global_rows, global_rows_coarse, global_cols_coarse;
   PetscInt cols_z, rows_z, local_rows_fine;
   PetscInt row_array[1];
   PetscMPIInt comm_size, comm_size_world;
   MPI_Comm MPI_COMM_MATRIX;
   const PetscInt *cols = NULL;
   // Matrix entries filled by MatGetRow are PetscScalar
   const PetscScalar *vals = NULL;
   PetscInt *d_nnz, *o_nnz;
   PetscInt one = 1;
   const PetscInt *is_pointer_coarse, *is_pointer_fine;
   MatType mat_type;
   // ~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)W, &MPI_COMM_MATRIX));
   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));
   PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD, &comm_size_world));

   PetscCall(MatGetOwnershipRange(W, &global_row_start_W, &global_row_end_plus_one_W));
   PetscCall(MatGetOwnershipRangeColumn(W, &global_col_start_W, &global_col_end_plus_one_W));

   PetscCall(MatGetSize(W, &cols_z, &rows_z));

   PetscCall(ISGetIndices(is_fine, &is_pointer_fine));
   PetscCall(ISGetIndices(is_coarse, &is_pointer_coarse));

   PetscCall(ISGetLocalSize(is_coarse, &local_rows_coarse));
   PetscCall(ISGetLocalSize(is_fine, &local_rows_fine));

   local_cols_coarse = local_rows_coarse;
   local_cols = local_rows_coarse + local_rows_fine;
   local_rows = local_cols;

   global_cols = rows_z + cols_z;
   global_rows = global_cols;
   global_rows_coarse = rows_z;
   global_cols_coarse = rows_z;
   (void)global_rows_coarse;

   // We may be reusing with the same sparsity
   if (!reuse) {

      PetscCall(MatCreate(MPI_COMM_MATRIX, P));
      PetscCall(MatSetSizes(*P, local_rows, local_cols_coarse,
                       global_rows, global_cols_coarse));
      // Match the output type
      PetscCall(MatGetType(W, &mat_type));
      PetscCall(MatSetType(*P, mat_type));

      // Count the entries in each row so we can exactly preallocate
      // The columns of P have the same layout as the columns of W
      // (at least one entry is allocated, as a Fortran zero-size array has an address)
      PetscCall(PetscCalloc1(PetscMax(local_rows, 1), &d_nnz));
      PetscCall(PetscCalloc1(PetscMax(local_rows, 1), &o_nnz));
      for (i_loc = global_row_start_W; i_loc < global_row_end_plus_one_W; i_loc++) {
         PetscCall(MatGetRow(W, i_loc, &ncols, &cols, NULL));
         p_row = is_pointer_fine[i_loc - global_row_start_W] - global_row_start;
         for (col = 0; col < ncols; col++) {
            if (cols[col] >= global_col_start_W && cols[col] < global_col_end_plus_one_W) {
               d_nnz[p_row] = d_nnz[p_row] + 1;
            } else {
               o_nnz[p_row] = o_nnz[p_row] + 1;
            }
         }
         PetscCall(MatRestoreRow(W, i_loc, &ncols, &cols, NULL));
      }
      // The identity block is always local
      if (identity) {
         for (i_loc = 0; i_loc < local_rows_coarse; i_loc++) {
            p_row = is_pointer_coarse[i_loc] - global_row_start;
            d_nnz[p_row] = d_nnz[p_row] + 1;
         }
      }
      PetscCall(MatXAIJSetPreallocation(*P, one, d_nnz, o_nnz,
               NULL, NULL));
      PetscCall(PetscFree(d_nnz));
      PetscCall(PetscFree(o_nnz));
   }

   // Don't set any off processor entries so no need for a reduction when assembling
   PetscCall(MatSetOption(*P, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
   PetscCall(MatSetOption(*P, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));

   // Copy in the values of W
   for (i_loc = global_row_start_W; i_loc < global_row_end_plus_one_W; i_loc++) {
      PetscCall(MatGetRow(W, i_loc, &ncols, &cols, &vals));
      if (ncols > 0) {
         row_array[0] = is_pointer_fine[i_loc - global_row_start_W];
         PetscCall(MatSetValues(*P, one, row_array, ncols, cols, vals, INSERT_VALUES));
      }
      PetscCall(MatRestoreRow(W, i_loc, &ncols, &cols, &vals));
   }

   // If we want the identity block or just leave it zero
   if (identity) {
      for (i_loc = 0; i_loc < local_rows_coarse; i_loc++) {
         PetscCall(MatSetValue(*P, is_pointer_coarse[i_loc], i_loc + global_col_start_W,
                  PFLARE_ONE, INSERT_VALUES));
      }
   }
   PetscCall(MatAssemblyBegin(*P, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(*P, MAT_FINAL_ASSEMBLY));

   PetscCall(ISRestoreIndices(is_coarse, &is_pointer_coarse));
   PetscCall(ISRestoreIndices(is_fine, &is_pointer_fine));

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode compute_R_from_Z(Mat Z, PetscInt global_row_start, IS is_fine, IS is_coarse,
   IS *orig_fine_col_indices,
   PetscBool identity, PetscBool reuse,
   Mat *R)
{
   // Wrapper around compute_R_from_Z_kokkos and compute_R_from_Z_cpu

   // ~~~~~~~~~~
#if defined(PETSC_HAVE_KOKKOS)
   Mat A_array, B_array = NULL;
   IS indices_fine, indices_coarse, orig_indices;
   int identity_int, reuse_int, reuse_indices_int;
   MatType mat_type;
   Mat temp_mat = NULL, temp_mat_reuse = NULL, temp_mat_compare = NULL;
   PetscReal normy;
   Vec max_vec;
   PetscInt row_loc;
   PetscBool is_kokkos;
#endif
   // ~~~~~~~~~~

   PetscFunctionBegin;

#if defined(PETSC_HAVE_KOKKOS)

   PetscCall(MatGetType(Z, &mat_type));
   PetscCall(PetscStrcmpAny(mat_type, &is_kokkos, MATMPIAIJKOKKOS, MATSEQAIJKOKKOS, MATAIJKOKKOS, ""));
   if (is_kokkos) {

      identity_int = 0;
      if (identity) identity_int = 1;
      reuse_int = 0;
      if (reuse) reuse_int = 1;
      reuse_indices_int = 0;
      if (*orig_fine_col_indices) {
         reuse_indices_int = 1;
      }

      A_array = Z;
      indices_fine = is_fine;
      indices_coarse = is_coarse;
      orig_indices = *orig_fine_col_indices;
      if (reuse) B_array = *R;
      compute_R_from_Z_kokkos(&A_array, global_row_start, &indices_fine, &indices_coarse,
                     &orig_indices,
                     identity_int, reuse_int, reuse_indices_int,
                     &B_array);
      *R = B_array;
      *orig_fine_col_indices = orig_indices;

      // If debugging do a comparison between CPU and Kokkos results
      if (kokkos_debug()) {

         // If we're doing reuse and debug, then the cpu version reuses the output
         // matrix itself (as that has the sparsity the cpu version expects) and so
         // we compare against a copy of the kokkos result
         if (reuse) {
            temp_mat = *R;
            PetscCall(MatConvert(*R, MATSAME, MAT_INITIAL_MATRIX, &temp_mat_compare));
         } else {
            temp_mat_compare = *R;
         }

         // Debug check if the CPU and Kokkos versions are the same
         PetscCall(compute_R_from_Z_cpu(Z, global_row_start, is_fine, is_coarse,
                        orig_fine_col_indices,
                        identity, reuse,
                        &temp_mat));

         PetscCall(MatConvert(temp_mat, MATSAME, MAT_INITIAL_MATRIX,
                     &temp_mat_reuse));

         PetscCall(MatAXPY(temp_mat_reuse, PFLARE_MINUS_ONE, temp_mat_compare, DIFFERENT_NONZERO_PATTERN));
         // Find the biggest entry in the difference
         PetscCall(MatCreateVecs(temp_mat_reuse, NULL, &max_vec));
         PetscCall(MatGetRowMaxAbs(temp_mat_reuse, max_vec, NULL));
         PetscCall(VecMax(max_vec, &row_loc, &normy));
         PetscCall(VecDestroy(&max_vec));

         if (normy > PFLARE_TOL_MATFREE_13 || PetscIsNanReal(normy)) {
            //call MatFilter(temp_mat_reuse, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
            //call MatView(temp_mat_reuse, PETSC_VIEWER_STDOUT_WORLD, ierr)
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Diff Kokkos and CPU compute_R_from_Z %g row %" PetscInt_FMT, (double)normy, row_loc);
         }
         PetscCall(MatDestroy(&temp_mat_reuse));
         if (!reuse) {
            PetscCall(MatDestroy(R));
         } else {
            PetscCall(MatDestroy(&temp_mat_compare));
         }
         *R = temp_mat;

      }

   } else {

      PetscCall(compute_R_from_Z_cpu(Z, global_row_start, is_fine, is_coarse,
                     orig_fine_col_indices,
                     identity, reuse,
                     R));

   }
#else
   PetscCall(compute_R_from_Z_cpu(Z, global_row_start, is_fine, is_coarse,
                  orig_fine_col_indices,
                  identity, reuse,
                  R));
#endif

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode compute_R_from_Z_cpu(Mat Z, PetscInt global_row_start, IS is_fine, IS is_coarse,
   IS *orig_fine_col_indices,
   PetscBool identity, PetscBool reuse,
   Mat *R)
{
   // Pass in Z and get out R = [Z I] (or [Z 0] if identity is false)

   // ~~~~~~~~~~
   PetscInt global_row_start_Z, global_row_end_plus_one_Z;
   PetscInt global_col_start_Z, global_col_end_plus_one_Z;
   PetscInt local_coarse_size, local_fine_size, local_full_cols;
   PetscInt i_loc, j_loc, ncols, max_ncols;
   PetscInt global_coarse_size, global_fine_size, global_full_cols;
   PetscInt rows_ao, cols_ao = 0, rows_ad, cols_ad, size_cols;
   PetscInt global_rows_z, global_cols_z;
   PetscInt local_rows_z, local_cols_z;
   PetscMPIInt comm_size;
   MPI_Comm MPI_COMM_MATRIX;
   const PetscInt *cols = NULL;
   // Matrix entries filled by MatGetRow are PetscScalar
   const PetscScalar *vals = NULL;
   // The columns of a row of Z in the full matrix
   PetscInt *row_cols, *d_nnz = NULL, *o_nnz = NULL;
   PetscInt row_array[1];
   const PetscInt *colmap = NULL;
   PetscInt one = 1;
   Mat Ad, Ao = NULL;
   PetscInt *col_indices_off_proc_array;
   IS col_indices;
   const PetscInt *is_pointer_orig_fine_col, *is_pointer_coarse, *is_pointer_fine;
   MatType mat_type;
   // ~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)Z, &MPI_COMM_MATRIX));
   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));

   PetscCall(ISGetIndices(is_fine, &is_pointer_fine));
   PetscCall(ISGetIndices(is_coarse, &is_pointer_coarse));

   PetscCall(ISGetLocalSize(is_coarse, &local_coarse_size));
   PetscCall(ISGetLocalSize(is_fine, &local_fine_size));
   PetscCall(ISGetSize(is_coarse, &global_coarse_size));
   PetscCall(ISGetSize(is_fine, &global_fine_size));

   local_full_cols = local_coarse_size + local_fine_size;
   global_full_cols = global_coarse_size + global_fine_size;

   PetscCall(MatGetLocalSize(Z, &local_rows_z, &local_cols_z));
   PetscCall(MatGetSize(Z, &global_rows_z, &global_cols_z));

   PetscCall(MatGetOwnershipRange(Z, &global_row_start_Z, &global_row_end_plus_one_Z));
   PetscCall(MatGetOwnershipRangeColumn(Z, &global_col_start_Z, &global_col_end_plus_one_Z));

   PetscCall(MatGetType(Z, &mat_type));

   // Get the local non-local components and sizes
   if (comm_size != 1) {

      PetscCall(MatMPIAIJGetSeqAIJ(Z, &Ad, &Ao, &colmap));

      // We know the col size of Ao is the size of colmap, the number of non-zero offprocessor columns
      PetscCall(MatGetSize(Ao, &rows_ao, &cols_ao));
      PetscCall(MatGetSize(Ad, &rows_ad, &cols_ad));

   } else {
      PetscCall(MatGetSize(Z, &rows_ad, &cols_ad));
      Ad = Z;
   }

   // We can reuse the orig_fine_col_indices as they can be expensive to generate in parallel
   if (!*orig_fine_col_indices) {

      // Now we need the global off-processor column indices in Z
      if (comm_size != 1) {

         // These are the global indices of the columns we want
         PetscCall(PetscMalloc1(cols_ad + cols_ao, &col_indices_off_proc_array));
         size_cols = cols_ad + cols_ao;
         // Local rows (as global indices)
         for (i_loc = 0; i_loc < cols_ad; i_loc++) {
            col_indices_off_proc_array[i_loc] = global_col_start_Z + i_loc;
         }
         // Off diagonal rows we want (as global indices)
         for (i_loc = 0; i_loc < cols_ao; i_loc++) {
            col_indices_off_proc_array[cols_ad + i_loc] = colmap[i_loc];
         }
         // These indices are not sorted, we deliberately have the local ones first then the
         // off processor ones, as we insert Z into the full matrix in pieces
         // with Ad, then Ao, so that way we can index into orig_fine_col_indices in that order

      } else {

         // These are the global indices of the columns we want
         PetscCall(PetscMalloc1(cols_ad, &col_indices_off_proc_array));
         size_cols = cols_ad;
         // Local rows (as global indices)
         for (i_loc = 0; i_loc < cols_ad; i_loc++) {
            col_indices_off_proc_array[i_loc] = global_col_start_Z + i_loc;
         }
      }

      // Create the IS we want with the cols we want (written as global indices)
      PetscCall(ISCreateGeneral(MPI_COMM_MATRIX, size_cols, col_indices_off_proc_array, PETSC_USE_POINTER, &col_indices));

      // Now let's do the comms to get what the original column indices in the full matrix are, given these indices for all
      // the columns of Z - ie we need to check in the original fine indices at the positions given by col_indices_off_proc_array
      // This could be expensive as the number of off-processor columns in Z grows!
      PetscCall(ISCreateSubIS(is_fine, col_indices, orig_fine_col_indices));

      // We've now built the original fine indices
      PetscCall(ISDestroy(&col_indices));
      PetscCall(PetscFree(col_indices_off_proc_array));

   }

   // Get the indices
   PetscCall(ISGetIndices(*orig_fine_col_indices, &is_pointer_orig_fine_col));

   // We may be reusing with the same sparsity
   if (!reuse) {

      PetscCall(MatCreate(MPI_COMM_MATRIX, R));
      // Taking care here to use the row sizes of Z, but the full size for cols
      // as we can use this routine to stick Afc and Aff in full sizes column arrays
      // and hence the row size of the input Z won't always equal the same sizes as the input is's
      PetscCall(MatSetSizes(*R, local_rows_z, local_full_cols,
                          global_rows_z, global_full_cols));
      // Match the output type
      PetscCall(MatSetType(*R, mat_type));
   }

   // Get the longest row of Ad or Ao and if we're not reusing, the number of entries in each
   // row of R so we can exactly preallocate
   // The columns of Ad are local fine points and so are in the diagonal block of R, the
   // columns of Ao are fine points on other ranks and so are in the off-diagonal block, and
   // the identity is local
   // (at least one entry is allocated, as a Fortran zero-size array has an address)
   if (!reuse) {
      PetscCall(PetscMalloc1(PetscMax(local_rows_z, 1), &d_nnz));
      PetscCall(PetscMalloc1(PetscMax(local_rows_z, 1), &o_nnz));
   }
   max_ncols = 0;
   for (i_loc = 0; i_loc < local_rows_z; i_loc++) {
      PetscCall(MatGetRow(Ad, i_loc, &ncols, NULL, NULL));
      max_ncols = PetscMax(max_ncols, ncols);
      if (!reuse) {
         d_nnz[i_loc] = ncols;
         if (identity) d_nnz[i_loc] = d_nnz[i_loc] + 1;
         o_nnz[i_loc] = 0;
      }
      PetscCall(MatRestoreRow(Ad, i_loc, &ncols, NULL, NULL));
      if (comm_size != 1) {
         PetscCall(MatGetRow(Ao, i_loc, &ncols, NULL, NULL));
         max_ncols = PetscMax(max_ncols, ncols);
         if (!reuse) o_nnz[i_loc] = ncols;
         PetscCall(MatRestoreRow(Ao, i_loc, &ncols, NULL, NULL));
      }
   }
   PetscCall(PetscMalloc1(PetscMax(max_ncols, 1), &row_cols));

   if (!reuse) {
      PetscCall(MatXAIJSetPreallocation(*R, one, d_nnz, o_nnz,
               NULL, NULL));
      PetscCall(PetscFree(d_nnz));
      PetscCall(PetscFree(o_nnz));
   }

   // Don't set any off processor entries so no need for a reduction when assembling
   PetscCall(MatSetOption(*R, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
   PetscCall(MatSetOption(*R, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));

   // Z - do Ad and Ao separately, as that way we have local indices into is_pointer_orig_fine_col
   // to give us the original column numbers
   // Remember Ad and Ao are serial
   for (i_loc = 0; i_loc < local_rows_z; i_loc++) {
      row_array[0] = i_loc + global_row_start_Z;

      // Ad
      PetscCall(MatGetRow(Ad, i_loc, &ncols, &cols, &vals));
      if (ncols > 0) {
         for (j_loc = 0; j_loc < ncols; j_loc++) row_cols[j_loc] = is_pointer_orig_fine_col[cols[j_loc]];
         PetscCall(MatSetValues(*R, one, row_array, ncols, row_cols, vals, INSERT_VALUES));
      }
      PetscCall(MatRestoreRow(Ad, i_loc, &ncols, &cols, &vals));

      // Ao
      if (comm_size != 1) {
         PetscCall(MatGetRow(Ao, i_loc, &ncols, &cols, &vals));
         if (ncols > 0) {
            for (j_loc = 0; j_loc < ncols; j_loc++) row_cols[j_loc] = is_pointer_orig_fine_col[cols[j_loc] + cols_ad];
            PetscCall(MatSetValues(*R, one, row_array, ncols, row_cols, vals, INSERT_VALUES));
         }
         PetscCall(MatRestoreRow(Ao, i_loc, &ncols, &cols, &vals));
      }

      // If we want the identity block or just leave it zero
      if (identity) {
         PetscCall(MatSetValue(*R, row_array[0], is_pointer_coarse[i_loc], PFLARE_ONE, INSERT_VALUES));
      }
   }
   PetscCall(PetscFree(row_cols));

   PetscCall(MatAssemblyBegin(*R, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(*R, MAT_FINAL_ASSEMBLY));

   PetscCall(ISRestoreIndices(*orig_fine_col_indices, &is_pointer_orig_fine_col));
   PetscCall(ISRestoreIndices(is_coarse, &is_pointer_coarse));
   PetscCall(ISRestoreIndices(is_fine, &is_pointer_fine));

   PetscFunctionReturn(PETSC_SUCCESS);
}

//-------------------------------------------------------------------------------------------------------------------------------
