/*
  C port of SAI_Z.F90 (module sai_z)

  The loops over rows and columns are 0-based and the indices that
  intersect_pre_sorted_indices_only returns are 0-based. The dense block of a
  row is column-major (i_size x j_size). The Fortran also compared the sorted
  set of the shadow I from sorted_unique against the binary tree it used to
  build it with (a check from stage 2 of docs/dev/fortran_removal_plan.md);
  the binary tree is Fortran only, so that check is not carried over
*/

#include <petscksp.h>
#include <petscblaslapack.h>
#include "pflare.h"
#include "pflareimpl.h"
#include "Sorting.h"
#include "SAI_Zk.h"
#include "PETSc_Helper.h"
#include "SAI_Z.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode calculate_and_build_sai_z_cpu(Mat A_ff_input, Mat A_cf, Mat sparsity_mat_cf, PetscBool incomplete,
   Mat *reuse_mat, Mat **reuse_submatrices, Mat *z, const PetscBool *no_approx_solve)
{
   // Computes an approximation to z using sai/isai
   // If incomplete is true then this is lAIR
   // Can also use this to compute an SAI (ie an inverse in the non rectangular case)
   // by giving A_cf as the negative identity (of the size A_ff)

   // ~~~~~~
   PetscInt local_rows, local_cols, ncols, global_row_start, global_row_end_plus_one, ncols_sparsity_max;
   PetscInt global_rows, global_cols, iterations_taken;
   PetscInt i_loc, j_loc, k_loc, cols_ad, rows_ad;
   PetscInt rows_ao, cols_ao = 0, ifree, row_size, i_size, j_size;
   PetscInt global_row_start_aff, global_row_end_plus_one_aff;
   int intersect_count;
   PetscMPIInt comm_size;
   // BLAS/LAPACK integer arguments must be PetscBLASInt
   PetscBLASInt lwork, info, m_bl, n_bl, nrhs_bl, lda_bl, ldb_bl;
   MPI_Comm MPI_COMM_MATRIX;
   Mat transpose_mat, A_ff;
   IS all_cols_indices, row_indices;
   IS col_indices;
   PetscInt one = 1, zero = 0, maxits = 1000;
   PetscInt *j_rows;
   PetscInt *i_rows, *ad_indices;
   int *j_indices, *i_indices;
   // LAPACK pivots must be PetscBLASInt
   PetscBLASInt *pivots;
   const PetscInt *cols = NULL;
   // Matrix entries (filled by MatGetRow / fed to MatSetValues and the dense
   // solves) are PetscScalar
   const PetscScalar *vals = NULL;
   PetscScalar *e_row, *j_vals;
   // Column-major i_size x j_size
   PetscScalar *submat_vals;
   // The columns of the rows in the shadow I, before sorted_unique, with
   // i_rows_size entries allocated
   PetscInt *i_rows_unsorted, *i_rows_grow;
   PetscInt i_rows_count, i_rows_size;
   PetscReal *work;
   Vec solution, rhs;
   PetscBool approx_solve, disable_approx_solve, is_mpiaij, is_diagonal;
   Mat Ao, Ad, temp_mat = NULL;
   KSP ksp;
   PC pc;
   const PetscInt *colmap = NULL;
   PetscBool deallocate_submatrices;
   PetscInt *col_indices_off_proc_array = NULL;
   MatType mat_type, mat_type_input;
   PetscScalar *vec_vals;
   PetscInt row_index_into_submatrix, ncols_match;
   PetscInt *cols_global_temp = NULL;
   int *row_i_match, *row_col_match;
   int match_count_sub;
   Mat small_mat;
   // The values handed to MatSetValues (the Fortran passed vals(row_col_match(1:match_count_sub)))
   PetscScalar *match_vals;
   PetscInt j_row;
   // ~~~~~~

   PetscFunctionBegin;

   disable_approx_solve = PETSC_FALSE;
   if (no_approx_solve) disable_approx_solve = *no_approx_solve;
   deallocate_submatrices = PETSC_FALSE;

   PetscCall(PetscObjectGetComm((PetscObject)A_ff_input, &MPI_COMM_MATRIX));
   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));

   // Get the local sizes
   PetscCall(MatGetLocalSize(A_cf, &local_rows, &local_cols));
   PetscCall(MatGetSize(A_cf, &global_rows, &global_cols));

   PetscCall(MatGetType(A_ff_input, &mat_type_input));
   PetscCall(PetscStrcmp(mat_type_input, MATDIAGONAL, &is_diagonal));
   if (is_diagonal) {
      // Convert it to aij just for this routine
      PetscCall(MatConvert(A_ff_input, MATAIJ, MAT_INITIAL_MATRIX, &A_ff));
   } else {
      A_ff = A_ff_input;
   }
   PetscCall(MatGetType(A_ff, &mat_type));
   PetscCall(PetscStrcmp(mat_type, "mpiaij", &is_mpiaij));

   // ~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~

   // We're enforcing the same sparsity

   // If not re-using
   if (!*z) {
      PetscCall(MatDuplicate(sparsity_mat_cf, MAT_DO_NOT_COPY_VALUES, z));
   }

   // Just in case there are some zeros in the input mat, ignore them
   // Now this won't do anything given we've imposed the sparsity of sparsity_mat_cf on
   // this matrix in advance with the MatDuplicate above
   // Dropping zeros from Z happens outside this routine
   //call MatSetOption(z, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE, ierr)
   // We know we will never have non-zero locations outside of the sparsity power
   PetscCall(MatSetOption(*z, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
   PetscCall(MatSetOption(*z, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
   // We know we are only going to insert local vals
   // These options should turn off any reductions in the assembly
   PetscCall(MatSetOption(*z, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));

   PetscCall(MatGetOwnershipRange(A_ff, &global_row_start_aff, &global_row_end_plus_one_aff));

   // ~~~~~~~~~~~~
   // If we're in parallel we need to get the off-process rows of matrix that correspond
   // to the columns of matrix
   // ~~~~~~~~~~~~
   // Have to double check comm_size /= 1 as we might be on a subcommunicator and we can't call
   // MatMPIAIJGetSeqAIJ specifically if that's the case
   if (comm_size != 1) {

      // ~~~~
      // Get the cols from the sparsity_mat_cf, not from A_ff
      // ~~~~
      // Much more annoying in older petsc
      if (is_mpiaij) {
         PetscCall(MatMPIAIJGetSeqAIJ(sparsity_mat_cf, &Ad, &Ao, &colmap));
      } else {
         PetscCall(MatConvert(sparsity_mat_cf, MATMPIAIJ, MAT_INITIAL_MATRIX, &temp_mat));
         PetscCall(MatMPIAIJGetSeqAIJ(temp_mat, &Ad, &Ao, &colmap));
      }

      // Have to be careful here as we don't have a square matrix, so rows_ao isn't equal to the number of local columns
      PetscCall(MatGetSize(Ad, &rows_ad, &cols_ad));
      // We know the col size of Ao is the size of colmap, the number of non-zero offprocessor columns
      PetscCall(MatGetSize(Ao, &rows_ao, &cols_ao));

      // ~~~~~~~
      // Now we can pull out the chunk of matrix that we need
      // ~~~~~~~

      // Setting this is necessary to avoid an allreduce when calling createsubmatrices
      // This will be reset to false after the call to createsubmatrices
      PetscCall(MatSetOption(A_ff, MAT_SUBMAT_SINGLEIS, PETSC_TRUE));

      // Now this will be doing comms to get the non-local rows we want
      // Only fetch the non-local rows - local rows are already in A_ff
      // This scales much better than fetching both local and non-local rows
      // This returns a sequential matrix
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF, cols_ao, colmap,
               PETSC_USE_POINTER, &row_indices));

      if (incomplete) {

         // These are the global indices of the columns we want
         // Taking care here to use cols_ad and not rows_ao
         PetscCall(PetscMalloc1(cols_ad + cols_ao, &col_indices_off_proc_array));
         PetscCall(PetscMalloc1(cols_ad, &ad_indices));
         // Local rows (as global indices)
         for (ifree = 0; ifree < cols_ad; ifree++) {
            ad_indices[ifree] = global_row_start_aff + ifree;
         }

         // Do a sort on the indices
         // Both ad_indices and colmap should already be sorted so we can merge them together quickly
         PetscCall(merge_pre_sorted(ad_indices, cols_ad, colmap, cols_ao, col_indices_off_proc_array));
         PetscCall(PetscFree(ad_indices));

         // Create the sequential IS we want with the cols we want (written as global indices)
         PetscCall(ISCreateGeneral(PETSC_COMM_SELF, cols_ad + cols_ao, col_indices_off_proc_array,
                  PETSC_USE_POINTER, &col_indices));
      } else {
         // For full SAI we need all columns with global indices preserved
         // so the shadow I computation can see all non-zero columns
         PetscCall(ISCreateStride(PETSC_COMM_SELF, global_cols, zero, one, &all_cols_indices));
         PetscCall(ISSetIdentity(all_cols_indices));
      }

      if (*reuse_mat) {
         (*reuse_submatrices)[0] = *reuse_mat;
         if (incomplete) {
            PetscCall(MatCreateSubMatrices(A_ff, one, &row_indices, &col_indices, MAT_REUSE_MATRIX, reuse_submatrices));
         } else {
            PetscCall(MatCreateSubMatrices(A_ff, one, &row_indices, &all_cols_indices, MAT_REUSE_MATRIX, reuse_submatrices));
         }
      } else {
         if (incomplete) {
            PetscCall(MatCreateSubMatrices(A_ff, one, &row_indices, &col_indices, MAT_INITIAL_MATRIX, reuse_submatrices));
         } else {
            PetscCall(MatCreateSubMatrices(A_ff, one, &row_indices, &all_cols_indices, MAT_INITIAL_MATRIX, reuse_submatrices));
         }
         *reuse_mat = (*reuse_submatrices)[0];
      }

      PetscCall(ISDestroy(&row_indices));
      if (!incomplete) PetscCall(ISDestroy(&all_cols_indices));

      // col_indices_off_proc_array is only allocated in the incomplete case
      if (incomplete) {
         row_size = cols_ad + cols_ao;
         PetscCall(ISDestroy(&col_indices));
      } else {
         row_size = global_cols;
      }

   // Easy in serial as we have everything we neeed
   } else {

      PetscCall(PetscMalloc1(1, reuse_submatrices));
      deallocate_submatrices = PETSC_TRUE;
      (*reuse_submatrices)[0] = A_ff;
      // local rows is the size of c, local cols is the size of f
      row_size = local_cols;
   }
   (void)row_size;

   PetscCall(MatGetOwnershipRange(sparsity_mat_cf, &global_row_start, &global_row_end_plus_one));

   // Setup the options for iterative solve when the direct gets too big
   if (incomplete) {

      // Sequential solve
      PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
      PetscCall(KSPSetType(ksp, KSPGMRES));
      // Solve to relative 1e-3
      // (PETSC_CURRENT is the value of the Fortran PETSC_DEFAULT_REAL)
      PetscCall(KSPSetTolerances(ksp, PFLARE_KSP_RTOL_COARSE,
               PFLARE_KSP_ATOL_OFF,
               PETSC_CURRENT,
               maxits));
      PetscCall(KSPGetPC(ksp, &pc));
      // Should be diagonally dominant
      PetscCall(PCSetType(pc, PCJACOBI));

   } else {

      // Sequential solve
      PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
      // Use the LSQR to solve the least squares inexactly
      PetscCall(KSPSetType(ksp, KSPLSQR));

      // Solve to relative 1e-3
      PetscCall(KSPSetTolerances(ksp, PFLARE_KSP_RTOL_COARSE,
               PFLARE_KSP_ATOL_OFF,
               PETSC_CURRENT,
               maxits));

      PetscCall(KSPGetPC(ksp, &pc));
      // We would have to form A' * A to precondition, but should be
      // very diagonally dominant anyway
      PetscCall(PCSetType(pc, PCNONE));
   }

   // Pre-pass to find the maximum number of columns in any row of sparsity_mat_cf
   ncols_sparsity_max = 0;
   for (i_loc = global_row_start; i_loc < global_row_end_plus_one; i_loc++) {
      PetscCall(MatGetRow(sparsity_mat_cf, i_loc, &ncols, NULL, NULL));
      if (ncols > ncols_sparsity_max) ncols_sparsity_max = ncols;
      PetscCall(MatRestoreRow(sparsity_mat_cf, i_loc, &ncols, NULL, NULL));
   }

   // Pre-allocate arrays that are sized by ncols from sparsity_mat_cf
   PetscCall(PetscMalloc1(ncols_sparsity_max, &j_rows));
   PetscCall(PetscMalloc1(ncols_sparsity_max, &j_vals));
   PetscCall(PetscMalloc1(ncols_sparsity_max, &j_indices));
   // Grown as needed
   i_rows_unsorted = NULL;
   i_rows_size = 0;

   // Now go through each of the rows
   // GetRow has to happen over the global indices
   for (i_loc = global_row_start; i_loc < global_row_end_plus_one; i_loc++) {

      // We just want the F-indices of whatever distance we are going out to
      PetscCall(MatGetRow(sparsity_mat_cf, i_loc, &ncols, &cols, NULL));

      j_size = ncols;
      for (j_loc = 0; j_loc < j_size; j_loc++) {
         j_vals[j_loc] = 0;
         j_rows[j_loc] = cols[j_loc];
      }

      PetscCall(MatRestoreRow(sparsity_mat_cf, i_loc, &ncols, &cols, NULL));

      // If we have no non-zeros in this row skip it
      // This means we have a c point with no neighbour f points
      if (j_size == 0) continue;

      // ~~~~~~~~
      // We need to stick the non-zero row of A_cf into the indices of
      // A_cf, or A_cf A_ff, or A_cf A_ff^2, etc given whatever distance we're going out to
      PetscCall(MatGetRow(A_cf, i_loc, &ncols, &cols, &vals));

      PetscCall(PetscMalloc1(ncols, &i_indices));
      PetscCall(intersect_pre_sorted_indices_only(j_rows, j_size, cols, ncols, j_indices, i_indices, &intersect_count));
      for (k_loc = 0; k_loc < intersect_count; k_loc++) j_vals[j_indices[k_loc]] = vals[i_indices[k_loc]];
      PetscCall(PetscFree(i_indices));

      PetscCall(MatRestoreRow(A_cf, i_loc, &ncols, &cols, &vals));

      // ~~~~~~~~

      // j_rows stays as global indices - we access local rows directly
      // from A_ff and non-local rows from the submatrix

      approx_solve = PETSC_FALSE;

      // This is the "incomplete" SAI (Antz 2018) which minimises for column j
      // ie it doesn't use the shadow I so gives a square system that can be solved
      // exactly for = 0
      // This is equivalent to a restricted additive schwarz
      // ||M(j, J) A(J, J) - eye(j, J)||_2
      if (incomplete) {
         i_size = j_size;
         PetscCall(PetscMalloc1(i_size, &i_rows));
         for (j_loc = 0; j_loc < j_size; j_loc++) i_rows[j_loc] = j_rows[j_loc];

      // This is the SAI which minimises for column j
      // which gives the rectangular system that must be minimised
      // ||M(j, J) A(J, I) - eye(j, I)||_2
      } else {

         // Loop over all the non zero column indices, and get the nonzero columns in those rows
         // ie get the shadow I
         // The columns of all the rows are collected in i_rows_unsorted and the sorted
         // set of them (without duplicates) is taken after the loop
         i_rows_count = 0;
         row_index_into_submatrix = 0;
         for (j_loc = 0; j_loc < j_size; j_loc++) {

            // Check if this is a non-local row
            if (comm_size != 1 &&
                (j_rows[j_loc] < global_row_start_aff || j_rows[j_loc] >= global_row_end_plus_one_aff)) {

               // Non-local row: find position in colmap via walk
               while (row_index_into_submatrix < cols_ao &&
                      colmap[row_index_into_submatrix] < j_rows[j_loc]) {
                  row_index_into_submatrix = row_index_into_submatrix + 1;
               }

               // Cols are GLOBAL (all_cols_indices preserves them)
               PetscCall(MatGetRow((*reuse_submatrices)[0], row_index_into_submatrix, &ncols, &cols, NULL));
               // Append the columns of this row to i_rows_unsorted, growing it if needed
               if (i_rows_count + ncols > i_rows_size) {
                  PetscCall(PetscMalloc1(PetscMax(2 * i_rows_size, i_rows_count + ncols), &i_rows_grow));
                  for (k_loc = 0; k_loc < i_rows_count; k_loc++) i_rows_grow[k_loc] = i_rows_unsorted[k_loc];
                  PetscCall(PetscFree(i_rows_unsorted));
                  i_rows_unsorted = i_rows_grow;
                  i_rows_size = PetscMax(2 * i_rows_size, i_rows_count + ncols);
               }
               for (k_loc = 0; k_loc < ncols; k_loc++) i_rows_unsorted[i_rows_count + k_loc] = cols[k_loc];
               i_rows_count = i_rows_count + ncols;
               PetscCall(MatRestoreRow((*reuse_submatrices)[0], row_index_into_submatrix, &ncols, &cols, NULL));

            } else {

               // Local row: get directly from A_ff - cols are global
               PetscCall(MatGetRow(A_ff, j_rows[j_loc], &ncols, &cols, NULL));
               // Append the columns of this row to i_rows_unsorted, growing it if needed
               if (i_rows_count + ncols > i_rows_size) {
                  PetscCall(PetscMalloc1(PetscMax(2 * i_rows_size, i_rows_count + ncols), &i_rows_grow));
                  for (k_loc = 0; k_loc < i_rows_count; k_loc++) i_rows_grow[k_loc] = i_rows_unsorted[k_loc];
                  PetscCall(PetscFree(i_rows_unsorted));
                  i_rows_unsorted = i_rows_grow;
                  i_rows_size = PetscMax(2 * i_rows_size, i_rows_count + ncols);
               }
               for (k_loc = 0; k_loc < ncols; k_loc++) i_rows_unsorted[i_rows_count + k_loc] = cols[k_loc];
               i_rows_count = i_rows_count + ncols;
               PetscCall(MatRestoreRow(A_ff, j_rows[j_loc], &ncols, &cols, NULL));
            }
         }

         // The sorted set of the columns, without duplicates
         PetscCall(sorted_unique(i_rows_unsorted, &i_rows_count));
         i_size = i_rows_count;
         PetscCall(PetscMalloc1(i_size, &i_rows));
         for (k_loc = 0; k_loc < i_size; k_loc++) i_rows[k_loc] = i_rows_unsorted[k_loc];
      }

      // If we have a big system to solve, do its iteratively
      if ((i_size > 40 || j_size > 40) && !disable_approx_solve) approx_solve = PETSC_TRUE;

      // This determines the indices of J^* in I*
      // Both i_rows and j_rows are global indices
      PetscCall(PetscMalloc1(i_size, &i_indices));
      PetscCall(intersect_pre_sorted_indices_only(i_rows, i_size, j_rows, j_size, i_indices, j_indices, &intersect_count));

      // Build the dense block directly from A_ff (local rows) and submatrix (non-local rows)
      if (!approx_solve) {
         PetscCall(PetscCalloc1(i_size * j_size, &submat_vals));
      } else {
         // Prevent uninitialised warning
         submat_vals = NULL;
         // Create small SeqAIJ for KSP
         PetscCall(MatCreate(PETSC_COMM_SELF, &small_mat));
         PetscCall(MatSetSizes(small_mat, j_size, i_size, j_size, i_size));
         PetscCall(MatSetType(small_mat, MATSEQAIJ));
         PetscCall(MatSetUp(small_mat));
      }

      row_index_into_submatrix = 0;
      for (j_loc = 0; j_loc < j_size; j_loc++) {

         // The row of the dense block (0-based) for MatSetValues
         j_row = j_loc;

         // Check if this is a non-local row
         if (comm_size != 1 &&
             (j_rows[j_loc] < global_row_start_aff || j_rows[j_loc] >= global_row_end_plus_one_aff)) {

            // Non-local row: find position in colmap via walk
            while (row_index_into_submatrix < cols_ao &&
                   colmap[row_index_into_submatrix] < j_rows[j_loc]) {
               row_index_into_submatrix = row_index_into_submatrix + 1;
            }

            PetscCall(MatGetRow((*reuse_submatrices)[0], row_index_into_submatrix, &ncols, &cols, &vals));

            if (incomplete) {
               // Incomplete: col_indices used for columns, so indices are local
               // Convert local col indices to global
               PetscCall(PetscMalloc1(ncols, &cols_global_temp));
               for (k_loc = 0; k_loc < ncols; k_loc++) cols_global_temp[k_loc] = col_indices_off_proc_array[cols[k_loc]];

               PetscCall(PetscMalloc1(i_size, &row_i_match));
               PetscCall(PetscMalloc1(ncols, &row_col_match));
               PetscCall(intersect_pre_sorted_indices_only(i_rows, i_size, cols_global_temp, ncols,
                        row_i_match, row_col_match, &match_count_sub));
            } else {
               // Full SAI: all_cols_indices used for columns, so indices are already global
               PetscCall(PetscMalloc1(i_size, &row_i_match));
               PetscCall(PetscMalloc1(ncols, &row_col_match));
               PetscCall(intersect_pre_sorted_indices_only(i_rows, i_size, cols, ncols,
                        row_i_match, row_col_match, &match_count_sub));
            }

            if (!approx_solve) {
               // Transpose: row of A_ff becomes column of submat_vals
               // as we solve A(J*, I*)^T z(j,J^*)^T = -A_cf(j,I*)^T
               for (k_loc = 0; k_loc < match_count_sub; k_loc++) {
                  submat_vals[row_i_match[k_loc] + j_loc * i_size] = vals[row_col_match[k_loc]];
               }
            } else {
               // Set row in small_mat (0-based indices)
               ncols_match = match_count_sub;
               PetscCall(PetscMalloc1(match_count_sub, &match_vals));
               if (incomplete) {
                  for (k_loc = 0; k_loc < match_count_sub; k_loc++) cols_global_temp[k_loc] = row_i_match[k_loc];
               } else {
                  PetscCall(PetscMalloc1(match_count_sub, &cols_global_temp));
                  for (k_loc = 0; k_loc < match_count_sub; k_loc++) cols_global_temp[k_loc] = row_i_match[k_loc];
               }
               for (k_loc = 0; k_loc < match_count_sub; k_loc++) match_vals[k_loc] = vals[row_col_match[k_loc]];
               PetscCall(MatSetValues(small_mat, one, &j_row, ncols_match,
                        cols_global_temp, match_vals,
                        INSERT_VALUES));
               PetscCall(PetscFree(match_vals));
               if (!incomplete) PetscCall(PetscFree(cols_global_temp));
            }

            if (incomplete) PetscCall(PetscFree(cols_global_temp));
            PetscCall(PetscFree(row_i_match));
            PetscCall(PetscFree(row_col_match));
            PetscCall(MatRestoreRow((*reuse_submatrices)[0], row_index_into_submatrix, &ncols, &cols, &vals));

         } else {

            // Local row: get directly from A_ff
            PetscCall(MatGetRow(A_ff, j_rows[j_loc], &ncols, &cols, &vals));

            // Intersect with i_rows (global) to find matching columns
            PetscCall(PetscMalloc1(i_size, &row_i_match));
            PetscCall(PetscMalloc1(ncols, &row_col_match));
            PetscCall(intersect_pre_sorted_indices_only(i_rows, i_size, cols, ncols,
                     row_i_match, row_col_match, &match_count_sub));

            if (!approx_solve) {
               // Transpose: row of A_ff becomes column of submat_vals
               for (k_loc = 0; k_loc < match_count_sub; k_loc++) {
                  submat_vals[row_i_match[k_loc] + j_loc * i_size] = vals[row_col_match[k_loc]];
               }
            } else {
               // Set row in small_mat (0-based indices)
               ncols_match = match_count_sub;
               PetscCall(PetscMalloc1(match_count_sub, &cols_global_temp));
               PetscCall(PetscMalloc1(match_count_sub, &match_vals));
               for (k_loc = 0; k_loc < match_count_sub; k_loc++) {
                  cols_global_temp[k_loc] = row_i_match[k_loc];
                  match_vals[k_loc] = vals[row_col_match[k_loc]];
               }
               PetscCall(MatSetValues(small_mat, one, &j_row, ncols_match,
                        cols_global_temp, match_vals,
                        INSERT_VALUES));
               PetscCall(PetscFree(cols_global_temp));
               PetscCall(PetscFree(match_vals));
            }

            PetscCall(PetscFree(row_i_match));
            PetscCall(PetscFree(row_col_match));
            PetscCall(MatRestoreRow(A_ff, j_rows[j_loc], &ncols, &cols, &vals));
         }
      }

      if (approx_solve) {
         PetscCall(MatAssemblyBegin(small_mat, MAT_FINAL_ASSEMBLY));
         PetscCall(MatAssemblyEnd(small_mat, MAT_FINAL_ASSEMBLY));
      }

      PetscCall(PetscCalloc1(i_size, &e_row));
      // Have to stick J^* into the indices of I*
      // be careful as there is also a minus here
      for (k_loc = 0; k_loc < intersect_count; k_loc++) e_row[i_indices[k_loc]] = -j_vals[j_indices[k_loc]];

      // ~~~~~~~~~~~~~
      // Solve the square system
      // ~~~~~~~~~~~~~
      if (incomplete) {

         // ~~~~~~~~~~~~~
         // Sparse approximate solve
         // ~~~~~~~~~~~~~
         if (approx_solve) {

            PetscCall(KSPSetOperators(ksp, small_mat, small_mat));
            PetscCall(KSPSetUp(ksp));

            PetscCall(MatCreateVecs(small_mat, &solution, NULL));
            // Have to restore the array before the solve in case this is kokkos
            PetscCall(VecGetArray(solution, &vec_vals));
            for (k_loc = 0; k_loc < i_size; k_loc++) vec_vals[k_loc] = e_row[k_loc];
            PetscCall(VecRestoreArray(solution, &vec_vals));

            // Do the solve - overwrite the rhs
            PetscCall(KSPSolveTranspose(ksp, solution, solution));
            PetscCall(KSPGetIterationNumber(ksp, &iterations_taken));

            PetscCall(VecGetArray(solution, &vec_vals));
            for (k_loc = 0; k_loc < i_size; k_loc++) e_row[k_loc] = vec_vals[k_loc];
            PetscCall(VecRestoreArray(solution, &vec_vals));

            PetscCall(KSPReset(ksp));
            PetscCall(VecDestroy(&solution));
            PetscCall(MatDestroy(&small_mat));

         // ~~~~~~~~~~~~~
         // Exact dense solve
         // about half the flops to do an LU rather than the QR
         // ~~~~~~~~~~~~~
         } else {

            PetscCall(PetscMalloc1(i_size, &pivots));
            // Kind-correct BLAS integer dimensions
            PetscCall(PetscBLASIntCast(i_size, &n_bl));
            nrhs_bl = 1;
            PetscCall(PetscBLASIntCast(i_size, &lda_bl));
            PetscCall(PetscBLASIntCast(i_size, &ldb_bl));
            // gesv applies the row interchanges from the LU to the rhs internally,
            // so e_row comes back as the solution in the original ordering
            PetscCallBLAS("LAPACKgesv", LAPACKgesv_(&n_bl, &nrhs_bl, submat_vals, &lda_bl, pivots, e_row, &ldb_bl, &info));
            PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "ISAI dense solve fail - singular local submatrix");
            PetscCall(PetscFree(pivots));
         }

      // ~~~~~~~~~~~~~
      // Solve the least-squares problem
      // ~~~~~~~~~~~~~
      } else {

         // ~~~~~~~~~~~~~
         // Sparse approximate solve
         // ~~~~~~~~~~~~~
         if (approx_solve) {

            // We can't seem to call KSPSolveTranspose with LSQR, so we explicitly
            // take a transpose here
            PetscCall(MatTranspose(small_mat, MAT_INITIAL_MATRIX, &transpose_mat));

            PetscCall(KSPSetOperators(ksp, transpose_mat, transpose_mat));
            PetscCall(KSPSetUp(ksp));

            PetscCall(MatCreateVecs(small_mat, &rhs, &solution));
            // Have to restore the array before the solve in case this is kokkos
            PetscCall(VecGetArray(rhs, &vec_vals));
            for (k_loc = 0; k_loc < i_size; k_loc++) vec_vals[k_loc] = e_row[k_loc];
            PetscCall(VecRestoreArray(rhs, &vec_vals));

            // Do the solve
            PetscCall(KSPSolve(ksp, rhs, solution));
            PetscCall(KSPGetIterationNumber(ksp, &iterations_taken));

            // Copy solution into e_row
            PetscCall(VecGetArray(solution, &vec_vals));
            for (k_loc = 0; k_loc < j_size; k_loc++) e_row[k_loc] = vec_vals[k_loc];
            PetscCall(VecRestoreArray(solution, &vec_vals));

            PetscCall(KSPReset(ksp));
            PetscCall(VecDestroy(&solution));
            PetscCall(VecDestroy(&rhs));
            PetscCall(MatDestroy(&transpose_mat));
            PetscCall(MatDestroy(&small_mat));

         // ~~~~~~~~~~~~~
         // Exact dense solve with QR
         // ~~~~~~~~~~~~~
         } else {

            PetscCall(PetscMalloc1(1, &work));
            lwork = -1;
            // Kind-correct BLAS integer dimensions
            PetscCall(PetscBLASIntCast(i_size, &m_bl));
            PetscCall(PetscBLASIntCast(j_size, &n_bl));
            nrhs_bl = 1;
            PetscCall(PetscBLASIntCast(i_size, &lda_bl));
            PetscCall(PetscBLASIntCast(i_size, &ldb_bl));
            PetscCallBLAS("LAPACKgels", LAPACKgels_("N", &m_bl, &n_bl, &nrhs_bl, submat_vals, &lda_bl,
                     e_row, &ldb_bl, work, &lwork, &info));
            // int() is exact for all plausible sizes < 2^24 even when work is single precision
            PetscCall(PetscBLASIntCast((PetscInt)work[0], &lwork));
            PetscCall(PetscFree(work));
            PetscCall(PetscMalloc1(lwork, &work));
            PetscCallBLAS("LAPACKgels", LAPACKgels_("N", &m_bl, &n_bl, &nrhs_bl, submat_vals, &lda_bl,
                     e_row, &ldb_bl, work, &lwork, &info));
            PetscCall(PetscFree(work));
         }
      }

      // ~~~~~~~~~~~~~
      // Set all the row values
      // ~~~~~~~~~~~~~
      // j_rows are global indices for both cases
      if (j_size != 0) {
         PetscCall(MatSetValues(*z, one, &i_loc,
                  j_size, j_rows, e_row, INSERT_VALUES));
      }

      PetscCall(PetscFree(i_rows));
      PetscCall(PetscFree(e_row));
      PetscCall(PetscFree(i_indices));
      PetscCall(PetscFree(submat_vals));
   }
   (void)iterations_taken;

   // Deallocate pre-allocated arrays
   PetscCall(PetscFree(j_rows));
   PetscCall(PetscFree(j_vals));
   PetscCall(PetscFree(j_indices));
   PetscCall(PetscFree(i_rows_unsorted));
   PetscCall(PetscFree(col_indices_off_proc_array));

   if (comm_size != 1 && !is_mpiaij) {
      PetscCall(MatDestroy(&temp_mat));
   }
   if (deallocate_submatrices) {
      PetscCall(PetscFree(*reuse_submatrices));
   }
   if (is_diagonal) {
      PetscCall(MatDestroy(&A_ff));
   }

   PetscCall(KSPDestroy(&ksp));
   PetscCall(MatAssemblyBegin(*z, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(*z, MAT_FINAL_ASSEMBLY));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode calculate_and_build_sai_z(Mat A_ff_input, Mat A_cf, Mat sparsity_mat_cf, PetscBool incomplete,
   Mat *reuse_mat, Mat **reuse_submatrices, Mat *z)
{
   // Wrapper around calculate_and_build_sai_z_cpu and calculate_and_build_sai_z_kokkos
   // Dispatches to Kokkos for the incomplete (lAIR & isai) case with aijkokkos matrices

   // ~~~~~~
#if defined(PETSC_HAVE_KOKKOS)
   Mat A_ff_arr, A_cf_arr, sparsity_arr, reuse_arr, z_arr;
   int reuse_int;
   int no_approx_solve_int;
   MatType mat_type;
   PetscBool reuse_triggered, is_kokkos, no_approx_solve_true = PETSC_TRUE;
   Mat temp_mat = NULL, temp_mat_reuse, temp_mat_compare;
   Mat reuse_mat_cpu = NULL;
   Mat *reuse_submatrices_cpu;
   PetscReal normy;
   Vec max_vec;
   PetscInt row_loc;
#endif
   // ~~~~~~

   PetscFunctionBegin;

#if defined(PETSC_HAVE_KOKKOS)

   PetscCall(MatGetType(A_ff_input, &mat_type));
   PetscCall(PetscStrcmpAny(mat_type, &is_kokkos, MATMPIAIJKOKKOS, MATSEQAIJKOKKOS, MATAIJKOKKOS, ""));
   if (is_kokkos && incomplete) {

      // We're enforcing the same sparsity
      // If not re-using
      if (!*z) {
         PetscCall(MatDuplicate(sparsity_mat_cf, MAT_DO_NOT_COPY_VALUES, z));
      }

      // We know we will never have non-zero locations outside of the sparsity
      PetscCall(MatSetOption(*z, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
      PetscCall(MatSetOption(*z, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
      PetscCall(MatSetOption(*z, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));

      // The handles for the kernel
      A_ff_arr = A_ff_input;
      A_cf_arr = A_cf;
      sparsity_arr = sparsity_mat_cf;
      reuse_arr = *reuse_mat;
      z_arr = *z;

      reuse_triggered = *reuse_mat ? PETSC_TRUE : PETSC_FALSE;
      reuse_int = 0;
      if (reuse_triggered) reuse_int = 1;

      // In debug mode force every row through the direct solve so the
      // CPU/Kokkos comparison below stays exact (CPU is forced to direct
      // via no_approx_solve=.TRUE. a few lines down).
      no_approx_solve_int = 0;
      if (kokkos_debug()) no_approx_solve_int = 1;

      // Call the Kokkos implementation
      calculate_and_build_sai_z_kokkos(&A_ff_arr, &A_cf_arr, &sparsity_arr,
               reuse_int, &reuse_arr, &z_arr, no_approx_solve_int);

      // Copy back the handles
      *reuse_mat = reuse_arr;
      *z = z_arr;

      // If debugging do a comparison between CPU and Kokkos results
      if (kokkos_debug()) {

         // If we're doing reuse and debug, then we have to always output the result
         // from the cpu version, as it will have coo preallocation structures set
         if (reuse_triggered) {
            temp_mat = *z;
            PetscCall(MatConvert(*z, MATSAME, MAT_INITIAL_MATRIX, &temp_mat_compare));
         } else {
            temp_mat_compare = *z;
         }

         // We send in an empty reuse_mat_cpu here always, as we can't pass through
         // the same one Kokkos uses as it now only gets out the non-local rows we need
         // We also disable the approximate solve if the size of any of the dense
         // matrices are greater than 40x40, as the kokkos code always does direct solves
         reuse_submatrices_cpu = NULL;
         PetscCall(calculate_and_build_sai_z_cpu(A_ff_input, A_cf, sparsity_mat_cf, incomplete,
                  &reuse_mat_cpu, &reuse_submatrices_cpu, &temp_mat,
                  &no_approx_solve_true));
         PetscCall(destroy_matrix_reuse(&reuse_mat_cpu, &reuse_submatrices_cpu));

         PetscCall(MatConvert(temp_mat, MATSAME, MAT_INITIAL_MATRIX, &temp_mat_reuse));

         PetscCall(MatAXPYWrapper(temp_mat_reuse, PFLARE_MINUS_ONE, temp_mat_compare));
         // Find the biggest entry in the difference
         PetscCall(MatCreateVecs(temp_mat_reuse, NULL, &max_vec));
         PetscCall(MatGetRowMaxAbs(temp_mat_reuse, max_vec, NULL));
         PetscCall(VecMax(max_vec, &row_loc, &normy));
         PetscCall(VecDestroy(&max_vec));

         // There is floating point compute in these inverses, so we have to be a
         // bit more tolerant to rounding differences
         if (normy > PFLARE_TOL_MATFREE_4EM11 || PetscIsNanReal(normy)) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Diff Kokkos and CPU calculate_and_build_sai_z %g row %" PetscInt_FMT, (double)normy, row_loc);
         }
         PetscCall(MatDestroy(&temp_mat_reuse));
         if (!reuse_triggered) {
            PetscCall(MatDestroy(z));
         } else {
            PetscCall(MatDestroy(&temp_mat_compare));
         }
         *z = temp_mat;
      }

   } else {

      PetscCall(calculate_and_build_sai_z_cpu(A_ff_input, A_cf, sparsity_mat_cf, incomplete,
               reuse_mat, reuse_submatrices, z, NULL));
   }
#else
   PetscCall(calculate_and_build_sai_z_cpu(A_ff_input, A_cf, sparsity_mat_cf, incomplete,
            reuse_mat, reuse_submatrices, z, NULL));
#endif

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode calculate_and_build_sai(Mat matrix, int sparsity_order, PetscBool incomplete,
   Mat *reuse_mat, Mat **reuse_submatrices, Mat *inv_matrix)
{
   // Computes an approximate inverse with an SAI (or an ISAI)
   // This just builds an identity and then calls the calculate_and_build_sai_z code

   // ~~~~~~
   Mat minus_I, sparsity_mat_cf, A_power;
   PetscInt order;
   PetscBool destroy_mat;
   Mat C_mat = NULL;
   // ~~~~~~

   PetscFunctionBegin;

   // ~~~~~~~~~~~
   // Now computing a SAI for Aff is the same as computing Z with an
   // SAI except we give it -I (the same size of Aff) instead of Acf
   // So we are going to use the same code
   // Means assembling an identity, but that is trivial
   // ~~~~~~~~~~~

   PetscCall(generate_identity(matrix, &minus_I));
   PetscCall(MatScale(minus_I, PFLARE_MINUS_ONE));

   // Calculate our approximate inverse
   // Now given we are using the same code as SAI Z
   // We have to feed it sparsity_mat_cf to get the approximate inverse sparsity we want
   // which is just powers of A
   if (sparsity_order == 0) {

      // Sparsity is just diagonal
      sparsity_mat_cf = minus_I;

   } else if (sparsity_order == 1) {

      // Sparsity is just matrix
      sparsity_mat_cf = matrix;

   } else {

      // If we're not doing reuse
      if (!*inv_matrix) {

         // Copy the pointer
         A_power = matrix;
         destroy_mat = PETSC_FALSE;

         // Sparsity is a power - A^sparsity_order
         for (order = 2; order <= sparsity_order; order++) {

            // Call a symbolic mult as we don't need the values, just the resulting sparsity
            PetscCall(mat_mat_symbolic(matrix, A_power, &C_mat));
            // Don't delete the original power - ie matrix
            if (destroy_mat) PetscCall(MatDestroy(&A_power));
            A_power = C_mat;
            destroy_mat = PETSC_TRUE;
         }

         sparsity_mat_cf = C_mat;

      // Reuse
      } else {
         PetscCall(MatDuplicate(*inv_matrix, MAT_DO_NOT_COPY_VALUES, &sparsity_mat_cf));
      }
   }

   // Now compute our sparse approximate inverse
   PetscCall(calculate_and_build_sai_z(matrix, minus_I, sparsity_mat_cf, incomplete,
            reuse_mat, reuse_submatrices, inv_matrix));
   PetscCall(MatDestroy(&minus_I));
   if (sparsity_order >= 2) PetscCall(MatDestroy(&sparsity_mat_cf));

   PetscFunctionReturn(PETSC_SUCCESS);
}
