/*
  C port of PETSc_Helper.F90 (module petsc_helper)

  The optional arguments of the Fortran are pointers that are NULL when the
  argument is absent, so present(x) is x != NULL. Matrices that the Fortran
  returned in an intent(inout) handle are Mat *, and the loops over the local
  rows and the row entries are 0-based
  The module variables got_debug_kokkos_env and kokkos_debug_global are file-static
  remove_from_sparse_match is called remove_from_sparse_match_c here, as the
  public C API in include/pflare.h already has a routine of that name (in PCAIR.c,
  which calls this one)
*/

#include <petscmat.h>
#include <stdarg.h>
#include <stdio.h>
#include <petscblaslapack.h>
#include <stdlib.h>
#include <string.h>
#include "pflareimpl.h"
#include "PETSc_Helper.h"
#include "PETSc_Helperk.h"

// Only used with Kokkos, so only defined with it (an unused static is a warning)
#if defined(PETSC_HAVE_KOKKOS)
static PetscBool got_debug_kokkos_env = PETSC_FALSE;
static PetscBool kokkos_debug_global  = PETSC_FALSE;
#endif

// -------------------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------------------------------------
// Some helper functions that involve PETSc matrices, mainly concerning sparsification
// -------------------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------------------------------------

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PETSC_INTERN PetscBool kokkos_debug(void)
{
   // This checks if an environmental variable is defined
   // If it is we check if the Kokkos and CPU versions of routines
   // are the same
   // This returns the flag rather than an error code, as it is only an environment
   // query and is used in conditions, as the Fortran function was

   // ~~~~~~~~~~~~
   PetscBool kokkos_debug_flag;

#if defined(PETSC_HAVE_KOKKOS)
   int env_val, length;
   const char *env_char;
   char env_digits[4];
   int i_loc, n_digits;
#endif
   // ~~~~~~~~~~~~

   kokkos_debug_flag = PETSC_FALSE;

#if defined(PETSC_HAVE_KOKKOS)

   // Only get the environmental variable once
   if (got_debug_kokkos_env) {
      kokkos_debug_flag = kokkos_debug_global;
   } else {
      // Check if the environment variable is set
      env_char = getenv("PFLARE_KOKKOS_DEBUG");
      length = env_char ? (int)strlen(env_char) : 0;

      got_debug_kokkos_env = PETSC_TRUE;
      if (env_char && length != 0) {
         // The Fortran reads the value with format (I3): the first three characters
         // as an integer, with blanks ignored
         n_digits = 0;
         for (i_loc = 0; i_loc < 3 && i_loc < length; i_loc++) {
            if (env_char[i_loc] != ' ') env_digits[n_digits++] = env_char[i_loc];
         }
         env_digits[n_digits] = '\0';
         env_val = atoi(env_digits);
         if (env_val == 1) {
            kokkos_debug_global = PETSC_TRUE;
            kokkos_debug_flag = kokkos_debug_global;
         }
      }
   }

#endif

   return kokkos_debug_flag;
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode destroy_matrix_reuse(Mat *mat, Mat **submatrices)
{
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   // Local
   PetscInt one = 1;

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   PetscFunctionBegin;

   if (*submatrices) {
      (*submatrices)[0] = *mat;
      // We only ever do one matrix at a time in matcreatesubmatrices
      // This also frees the array and sets *submatrices to NULL
      PetscCall(MatDestroySubMatrices(one, submatrices));
      // MatDestroySubMatrices doesn't leave the mat in the same null state
      // as calling matdestroy
      *mat = NULL;
   } else {
      PetscCall(MatDestroy(mat));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode remove_small_from_sparse(Mat input_mat, PetscReal tol, Mat *output_mat, const int *relative_max_row_tol_int,
   const PetscBool *lump, const int *drop_diagonal_int, const int *diag_strength_int)
{
   // Wrapper around remove_small_from_sparse_cpu and remove_small_from_sparse_kokkos

   // ~~~~~~~~~~
#if defined(PETSC_HAVE_KOKKOS)
   Mat A_array, B_array = NULL;
   int lump_int, allow_drop_diagonal_int, allow_diag_strength_int, rel_max_row_tol_int;
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

      // Absolute tolerance by default
      rel_max_row_tol_int = 0;
      if (relative_max_row_tol_int) {
         rel_max_row_tol_int = *relative_max_row_tol_int;
      }
      lump_int = 0;
      if (lump) {
         if (*lump) {
            lump_int = 1;
         }
      }
      // Never drop the diagonal by default
      allow_drop_diagonal_int = 0;
      if (drop_diagonal_int) {
         allow_drop_diagonal_int = *drop_diagonal_int;
      }
      // Whether we use a strength of connection defined by |a_ij| .ge. tol * |a_ii|
      allow_diag_strength_int = 0;
      if (diag_strength_int) {
         allow_diag_strength_int = *diag_strength_int;
      }

      A_array = input_mat;
      remove_small_from_sparse_kokkos(&A_array, tol,
               &B_array, rel_max_row_tol_int,
               lump_int, allow_drop_diagonal_int,
               allow_diag_strength_int);
      *output_mat = B_array;

      // If debugging do a comparison between CPU and Kokkos results
      if (kokkos_debug()) {

         // Debug check if the CPU and Kokkos versions are the same
         PetscCall(remove_small_from_sparse_cpu(input_mat, tol, &temp_mat, relative_max_row_tol_int,
                  lump, drop_diagonal_int, diag_strength_int));

         PetscCall(MatAXPY(temp_mat, PFLARE_MINUS_ONE, *output_mat, DIFFERENT_NONZERO_PATTERN));
         // Find the biggest entry in the difference
         PetscCall(MatCreateVecs(temp_mat, NULL, &max_vec));
         PetscCall(MatGetRowMaxAbs(temp_mat, max_vec, NULL));
         PetscCall(VecMax(max_vec, &row_loc, &normy));
         PetscCall(VecDestroy(&max_vec));

         if (normy > PFLARE_TOL_MATFREE_12 || PetscIsNanReal(normy)) {
            //call MatFilter(temp_mat, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
            //call MatView(temp_mat, PETSC_VIEWER_STDOUT_WORLD, ierr)
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Diff Kokkos and CPU remove_small_from_sparse %g row %" PetscInt_FMT, (double)normy, row_loc);
         }
         PetscCall(MatDestroy(&temp_mat));
      }

   } else {

      PetscCall(remove_small_from_sparse_cpu(input_mat, tol, output_mat, relative_max_row_tol_int,
               lump, drop_diagonal_int, diag_strength_int));

   }
#else
   PetscCall(remove_small_from_sparse_cpu(input_mat, tol, output_mat, relative_max_row_tol_int,
            lump, drop_diagonal_int, diag_strength_int));
#endif

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode remove_small_from_sparse_cpu(Mat input_mat, PetscReal tol, Mat *output_mat, const int *relative_max_row_tol_int,
   const PetscBool *lump, const int *drop_diagonal_int, const int *diag_strength_int)
{
   // Returns a copy of a sparse matrix with entries below abs(val) < tol removed
   // If rel_max_row_tol_int is 1, then the tol is taken to be a relative scaling
   // of the max row val on each row including the diagonal. If it's -1, it doesn't include the diagonal
   // If lumped is true the removed entries are added to the diagonal

   // ~~~~~~~~~~
   PetscInt col, ncols, ifree, row_n, diag_pos, target_col, i_loc, max_ncols;
   PetscInt local_rows, local_cols, global_rows, global_cols, global_row_start;
   PetscInt global_row_end_plus_one, diag_index;
   PetscInt global_col_start, global_col_end_plus_one;
   const PetscInt *cols = NULL;
   // Matrix entries filled by MatGetRow are PetscScalar
   const PetscScalar *vals = NULL;
   // The output row
   PetscInt *row_cols, *d_nnz, *o_nnz;
   PetscScalar *row_vals;
   PetscInt row_array[1];
   PetscInt one = 1;
   PetscBool lump_entries;
   int drop_diag_int, diag_stren_int, rel_max_row_tol_int, pass;
   // Derived magnitude/tolerance stays PetscReal
   PetscReal rel_row_tol;
   MPI_Comm MPI_COMM_MATRIX;
   MatType mat_type;
   PetscScalar abs_biggest_entry, value;
   PetscReal max_abs_val;

   // ~~~~~~~~~~
   // If the tolerance is 0 we still want to go through this routine and drop the zeros

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)input_mat, &MPI_COMM_MATRIX));

   lump_entries = PETSC_FALSE;
   // 1  - Allow drop diagonal
   // 0  - Never drop diagonal
   // -1 - Always drop diagonal
   // Never drop the diagonal by default
   drop_diag_int = 0;
   diag_stren_int = 0;
   if (lump) lump_entries = *lump;
   if (drop_diagonal_int) drop_diag_int = *drop_diagonal_int;
   if (diag_strength_int) diag_stren_int = *diag_strength_int;
   rel_row_tol = tol;
   // 1  - Relative row tolerance (including diagonal)
   // 0  - Absolute tolerance
   // -1 - Relative row tolerance (not including diagonal)
   // Absolute tolerance by default
   rel_max_row_tol_int = 0;
   if (relative_max_row_tol_int) {
      rel_max_row_tol_int = *relative_max_row_tol_int;
   }

   PetscCheck(!(lump_entries && drop_diag_int == 1), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Error: Cannot lump and drop the diagonal");

   // Get the local sizes
   PetscCall(MatGetLocalSize(input_mat, &local_rows, &local_cols));
   PetscCall(MatGetSize(input_mat, &global_rows, &global_cols));
   // This returns the global index of the local portion of the matrix
   PetscCall(MatGetOwnershipRange(input_mat, &global_row_start, &global_row_end_plus_one));
   PetscCall(MatGetOwnershipRangeColumn(input_mat, &global_col_start, &global_col_end_plus_one));

   PetscCall(MatCreate(MPI_COMM_MATRIX, output_mat));
   PetscCall(MatSetSizes(*output_mat, local_rows, local_cols,
                    global_rows, global_cols));
   // Match the output type
   PetscCall(MatGetType(input_mat, &mat_type));
   PetscCall(MatSetType(*output_mat, mat_type));

   PetscCall(PetscCalloc1(local_rows, &d_nnz));
   PetscCall(PetscCalloc1(local_rows, &o_nnz));
   // The output row is never longer than the input row plus a diagonal
   max_ncols = 0;
   for (ifree = global_row_start; ifree < global_row_end_plus_one; ifree++) {
      PetscCall(MatGetRow(input_mat, ifree, &ncols, NULL, NULL));
      max_ncols = PetscMax(max_ncols, ncols);
      PetscCall(MatRestoreRow(input_mat, ifree, &ncols, NULL, NULL));
   }
   PetscCall(PetscMalloc1(max_ncols + 1, &row_cols));
   PetscCall(PetscMalloc1(max_ncols + 1, &row_vals));

   // We go through the rows twice, the first time to count the entries we keep so we
   // can exactly preallocate, the second to set them
   for (pass = 1; pass <= 2; pass++) {
      for (ifree = global_row_start; ifree < global_row_end_plus_one; ifree++) {

         // Get the row
         PetscCall(MatGetRow(input_mat, ifree, &ncols, &cols, &vals));

         // Find where the diagonal is
         diag_index = -1;
         if (diag_stren_int == 1) {
            for (col = 0; col < ncols; col++) {
               if (cols[col] == ifree) {
                  diag_index = col;
               }
            }
         }

         // If we want a relative row tolerance
         if (rel_max_row_tol_int != 0) {
            // Include the diagonal in the relative row tolerance
            if (rel_max_row_tol_int == 1) {
               if (diag_stren_int == 1) {
                  // If there is a zero diagonal
                  if (diag_index == -1) {
                     rel_row_tol = 0.0;
                  } else {
                     // We are measuring relative to the strength of the diagonal
                     rel_row_tol = tol * PetscAbsScalar(vals[diag_index]);
                  }
               } else {
                  // maxval(abs(vals(1:ncols))), which is -huge for an empty row
                  max_abs_val = -PETSC_MAX_REAL;
                  for (col = 0; col < ncols; col++) {
                     if (PetscAbsScalar(vals[col]) > max_abs_val) max_abs_val = PetscAbsScalar(vals[col]);
                  }
                  rel_row_tol = tol * max_abs_val;
               }

            // Don't include the diagonal in the relative row tolerance
            } else if (rel_max_row_tol_int == -1) {

               // If the user has specified they don't want to include the diagonal in the
               // relative row tolerance, but also have said they want to measure
               // everything by the strength of the diagonal, it doesn't make
               // sense to exclude the diagonal!
               if (diag_stren_int == 1) {
                  // If there is a zero diagonal
                  if (diag_index == -1) {
                     rel_row_tol = 0.0;
                  } else {
                     // We are measuring relative to the strength of the diagonal
                     rel_row_tol = tol * PetscAbsScalar(vals[diag_index]);
                  }
               } else {
                  // Be careful to take huge() of the typed target (not 0d0) so
                  // single builds get the representable -huge, not -Inf
                  abs_biggest_entry = -PETSC_MAX_REAL;
                  // Find the biggest entry in the row thats not the diagonal
                  for (col = 0; col < ncols; col++) {
                     if (cols[col] != ifree && PetscAbsScalar(vals[col]) > abs_biggest_entry) {
                        abs_biggest_entry = PetscAbsScalar(vals[col]);
                     }
                  }
                  rel_row_tol = tol * abs_biggest_entry;
               }
            }
         }

         // Build the output row - anything that ends up on the diagonal (the diagonal itself
         // and any lumped entries) is summed into a single diagonal entry
         // diag_pos is the position of the diagonal in the output row, -1 until it is added
         row_n = 0;
         diag_pos = -1;
         for (col = 0; col < ncols; col++) {

            // Keep it if it is bigger than the tolerance
            if (PetscAbsScalar(vals[col]) >= rel_row_tol) {

               // If this is the diagonal and we are always dropping it, then don't add it
               if (drop_diag_int == -1 && cols[col] == ifree) continue;
               target_col = cols[col];

            // If the entry is small and we are lumping, then add it to the diagonal
            // or if this is the diagonal and it's small but we are not dropping it
            } else if (lump_entries || (drop_diag_int == 0 && cols[col] == ifree)) {
               target_col = ifree;
            } else {
               continue;
            }

            value = vals[col];
            if (target_col == ifree) {
               if (diag_pos == -1) {
                  diag_pos = row_n;
                  row_cols[row_n] = ifree;
                  row_vals[row_n] = value;
                  row_n = row_n + 1;
               } else {
                  row_vals[diag_pos] = row_vals[diag_pos] + value;
               }
            } else {
               row_cols[row_n] = target_col;
               row_vals[row_n] = value;
               row_n = row_n + 1;
            }
         }

         // Must call otherwise petsc leaks memory
         PetscCall(MatRestoreRow(input_mat, ifree, &ncols, &cols, &vals));

         if (pass == 1) {
            i_loc = ifree - global_row_start;
            for (col = 0; col < row_n; col++) {
               if (row_cols[col] >= global_col_start && row_cols[col] < global_col_end_plus_one) {
                  d_nnz[i_loc] = d_nnz[i_loc] + 1;
               } else {
                  o_nnz[i_loc] = o_nnz[i_loc] + 1;
               }
            }
         } else if (row_n > 0) {
            row_array[0] = ifree;
            PetscCall(MatSetValues(*output_mat, one, row_array, row_n, row_cols, row_vals,
                     INSERT_VALUES));
         }
      }

      if (pass == 1) {
         PetscCall(MatXAIJSetPreallocation(*output_mat, one, d_nnz, o_nnz,
                  NULL, NULL));
         // Don't set any off processor entries so no need for a reduction when assembling
         PetscCall(MatSetOption(*output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
         PetscCall(MatSetOption(*output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
      }
   }
   PetscCall(PetscFree(d_nnz));
   PetscCall(PetscFree(o_nnz));
   PetscCall(PetscFree(row_cols));
   PetscCall(PetscFree(row_vals));

   PetscCall(MatAssemblyBegin(*output_mat, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(*output_mat, MAT_FINAL_ASSEMBLY));

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode remove_from_sparse_match_no_lump(Mat input_mat, Mat output_mat, const PetscReal *alpha)
{
   // Returns a copy of a sparse matrix with entries that don't match the sparsity
   // of the other input matrix dropped

   // ~~~~~~~~~~
   PetscInt ncols, ifree, col;
   PetscInt global_row_start, global_row_end_plus_one;
   const PetscInt *cols = NULL;
   // Matrix entries filled by MatGetRow are PetscScalar
   const PetscScalar *vals = NULL;
   PetscInt one = 1;
   // alpha * vals, which is an array expression in the Fortran, grown as needed
   PetscScalar *alpha_vals = NULL;
   PetscInt alpha_vals_size = 0;

   // ~~~~~~~~~~
   PetscFunctionBegin;

   // This returns the global index of the local portion of the matrix
   PetscCall(MatGetOwnershipRange(input_mat, &global_row_start, &global_row_end_plus_one));

   // Just in case there are some zeros in the input mat, ignore them
   PetscCall(MatSetOption(output_mat, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
   // Don't set any off processor entries so no need for a reduction when assembling
   PetscCall(MatSetOption(output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
   PetscCall(MatSetOption(output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
   // This ensures any entries outside the existing sparsity of output_mat are dropped
   // Not sure if this is respected by the new COO interface so we will keep the
   // matsetvalues use below for now
   PetscCall(MatSetOption(output_mat, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));

   // Now go and fill the new matrix
   // Loop over global row indices

   // Add values to existing matrix
   if (alpha) {

      for (ifree = global_row_start; ifree < global_row_end_plus_one; ifree++) {

         // Get the row
         PetscCall(MatGetRow(input_mat, ifree, &ncols, &cols, &vals));
         if (ncols != 0) {
            if (ncols > alpha_vals_size) {
               PetscCall(PetscFree(alpha_vals));
               PetscCall(PetscMalloc1(ncols, &alpha_vals));
               alpha_vals_size = ncols;
            }
            for (col = 0; col < ncols; col++) alpha_vals[col] = *alpha * vals[col];
            PetscCall(MatSetValues(output_mat, one, &ifree, ncols, cols,
                  alpha_vals, ADD_VALUES));
         }
         PetscCall(MatRestoreRow(input_mat, ifree, &ncols, &cols, &vals));

      }
      PetscCall(PetscFree(alpha_vals));

   // Replace
   } else {
      for (ifree = global_row_start; ifree < global_row_end_plus_one; ifree++) {

         // Get the row
         PetscCall(MatGetRow(input_mat, ifree, &ncols, &cols, &vals));
         if (ncols != 0) {
            PetscCall(MatSetValues(output_mat, one, &ifree, ncols, cols,
                  vals, INSERT_VALUES));
         }
         PetscCall(MatRestoreRow(input_mat, ifree, &ncols, &cols, &vals));

      }
   }

   PetscCall(MatAssemblyBegin(output_mat, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(output_mat, MAT_FINAL_ASSEMBLY));

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode remove_from_sparse_match_c(Mat input_mat, Mat output_mat, const PetscBool *lump, const PetscReal *alpha)
{
   // Wrapper around remove_from_sparse_match_cpu and remove_from_sparse_match_kokkos
   // (remove_from_sparse_match in the Fortran)

   // ~~~~~~~~~~
#if defined(PETSC_HAVE_KOKKOS)
   Mat A_array, B_array;
   int lump_int, alpha_int;
   MatType mat_type;
   Mat temp_mat = NULL;
   PetscReal normy;
   PetscReal alpha_val;
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

      lump_int = 0;
      if (lump) {
         if (*lump) {
            lump_int = 1;
         }
      }
      alpha_int = 0;
      alpha_val = 1.0;
      if (alpha) {
         alpha_int = 1;
         alpha_val = *alpha;
      }

      if (kokkos_debug()) {
         // Make sure to copy the values here as we may be doing with alpha
         PetscCall(MatDuplicate(output_mat,
                  MAT_COPY_VALUES, &temp_mat));
      }

      A_array = input_mat;
      B_array = output_mat;
      remove_from_sparse_match_kokkos(&A_array, &B_array, lump_int, alpha_int, alpha_val);

      // If debugging do a comparison between CPU and Kokkos results
      if (kokkos_debug()) {

         // Debug check if the CPU and Kokkos versions are the same
         PetscCall(remove_from_sparse_match_cpu(input_mat, temp_mat, lump, alpha));

         PetscCall(MatAXPY(temp_mat, PFLARE_MINUS_ONE, output_mat, DIFFERENT_NONZERO_PATTERN));
         // Find the biggest entry in the difference
         PetscCall(MatCreateVecs(temp_mat, NULL, &max_vec));
         PetscCall(MatGetRowMaxAbs(temp_mat, max_vec, NULL));
         PetscCall(VecMax(max_vec, &row_loc, &normy));
         PetscCall(VecDestroy(&max_vec));

         if (normy > PFLARE_TOL_MATFREE_13 || PetscIsNanReal(normy)) {
            //call MatFilter(temp_mat, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
            //call MatView(temp_mat, PETSC_VIEWER_STDOUT_WORLD, ierr)
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Diff Kokkos and CPU remove_from_sparse_match %g row %" PetscInt_FMT, (double)normy, row_loc);
         }
         PetscCall(MatDestroy(&temp_mat));
      }

   } else {

      PetscCall(remove_from_sparse_match_cpu(input_mat, output_mat, lump, alpha));

   }
#else
   PetscCall(remove_from_sparse_match_cpu(input_mat, output_mat, lump, alpha));
#endif

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode remove_from_sparse_match_cpu(Mat input_mat, Mat output_mat, const PetscBool *lump, const PetscReal *alpha)
{
   // Returns a copy of a sparse matrix with entries that don't match the sparsity
   // of the other input matrix dropped
   // If lumped is true the removed entries are added to the diagonal
   // If alpha is present, it does output_mat += alpha * input_mat
   // on the given sparsity, rather than output_mat = input_mat

   // ~~~~~~~~~~
   PetscInt col, ncols, ifree, max_nnzs, ncols_mod, index1, index2;
   PetscInt local_rows, local_cols, global_rows, global_cols, global_row_start;
   PetscInt global_row_end_plus_one, max_nnzs_total_two;
   PetscInt global_col_start, global_col_end_plus_one, max_nnzs_total;
   PetscCount counter;
   PetscMPIInt comm_size;
   const PetscInt *cols = NULL;
   PetscInt *cols_mod;
   // Matrix entries filled by MatGetRow are PetscScalar
   const PetscScalar *vals = NULL;
   PetscScalar *vals_copy;
   PetscInt *col_indices;
   // Value buffer feeding MatSetValues is PetscScalar
   PetscScalar *v, *lump_vals;
   PetscCount *row_ptr;
   PetscInt one = 1;
   PetscBool lump_entries, alpha_present;
   PetscReal lump_sum;
   InsertMode insert_mode;
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt row_global;

   // ~~~~~~~~~~
   // If the tolerance is 0 we still want to go through this routine and drop the zeros

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)input_mat, &MPI_COMM_MATRIX));
   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));

   lump_entries = PETSC_FALSE;
   if (lump) lump_entries = *lump;
   alpha_present = PETSC_FALSE;
   if (alpha) alpha_present = PETSC_TRUE;

   if (!lump_entries) {
      // This version is faster
      PetscCall(remove_from_sparse_match_no_lump(input_mat, output_mat, alpha));
      PetscFunctionReturn(PETSC_SUCCESS);
   }

   // Get the local sizes
   PetscCall(MatGetLocalSize(input_mat, &local_rows, &local_cols));
   PetscCall(MatGetSize(input_mat, &global_rows, &global_cols));
   // This returns the global index of the local portion of the matrix
   PetscCall(MatGetOwnershipRange(input_mat, &global_row_start, &global_row_end_plus_one));
   PetscCall(MatGetOwnershipRangeColumn(input_mat, &global_col_start, &global_col_end_plus_one));

   max_nnzs = 0;
   max_nnzs_total = 0;
   max_nnzs_total_two = 0;
   for (ifree = global_row_start; ifree < global_row_end_plus_one; ifree++) {

      PetscCall(MatGetRow(input_mat, ifree, &ncols, NULL, NULL));
      if (ncols > max_nnzs) max_nnzs = ncols;
      max_nnzs_total = max_nnzs_total + ncols;
      PetscCall(MatRestoreRow(input_mat, ifree, &ncols, NULL, NULL));
      PetscCall(MatGetRow(output_mat, ifree, &ncols, NULL, NULL));
      if (ncols > max_nnzs) max_nnzs = ncols;
      max_nnzs_total_two = max_nnzs_total_two + ncols;
      PetscCall(MatRestoreRow(output_mat, ifree, &ncols, NULL, NULL));
   }

   max_nnzs_total = PetscMax(max_nnzs_total, max_nnzs_total_two);

   PetscCall(PetscMalloc1(max_nnzs, &cols_mod));
   PetscCall(PetscMalloc1(max_nnzs, &vals_copy));

   // Times 2 here in case we are lumping
   PetscCall(PetscMalloc1(max_nnzs_total * 2, &col_indices));
   PetscCall(PetscMalloc1(max_nnzs_total * 2, &v));

   // Just in case there are some zeros in the input mat, ignore them
   PetscCall(MatSetOption(output_mat, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
   // Don't set any off processor entries so no need for a reduction when assembling
   PetscCall(MatSetOption(output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
   PetscCall(MatSetOption(output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
   // This ensures any entries outside the existing sparsity of output_mat are dropped
   // (e.g., a lumped diagonal entry when output_mat has no diagonal)
   PetscCall(MatSetOption(output_mat, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));

   PetscCall(PetscMalloc1(local_rows, &lump_vals));
   PetscCall(PetscMalloc1(local_rows + 1, &row_ptr));

   // Now go and fill the new matrix
   // Loop over global row indices
   counter = 0;
   for (ifree = global_row_start; ifree < global_row_end_plus_one; ifree++) {

      row_ptr[ifree - global_row_start] = counter;

      // Get the row
      PetscCall(MatGetRow(input_mat, ifree, &ncols, &cols, &vals));

      lump_sum = 0;
      ncols_mod = ncols;
      for (col = 0; col < ncols; col++) cols_mod[col] = cols[col];
      if (alpha_present) {
         for (col = 0; col < ncols; col++) vals_copy[col] = *alpha * vals[col];
      } else {
         for (col = 0; col < ncols; col++) vals_copy[col] = vals[col];
      }

      // Must call otherwise petsc leaks memory
      PetscCall(MatRestoreRow(input_mat, ifree, &ncols, &cols, &vals));
      // Get the sparsity row
      PetscCall(MatGetRow(output_mat, ifree, &ncols, &cols, NULL));

      // Now loop through and do the intersection
      // Anything that is not in both is not inserted
      // cols should be a subset of cols_mod
      index1 = 0;
      index2 = 0;
      while (index1 < ncols_mod && index2 < ncols) {
         if (cols_mod[index1] == cols[index2]) {
            index1 = index1 + 1;
            index2 = index2 + 1;
         } else if (cols_mod[index1] < cols[index2]) {
            // This value won't be inserted into the matrix
            cols_mod[index1] = -1;
            if (lump_entries) lump_sum = lump_sum + vals_copy[index1];
            index1 = index1 + 1;
         } else {
            index2 = index2 + 1;
         }
      }
      // Do the rest
      for (col = index1; col < ncols_mod; col++) {
         cols_mod[col] = -1;
         if (lump_entries) lump_sum = lump_sum + vals_copy[col];
      }

      // Must call otherwise petsc leaks memory
      PetscCall(MatRestoreRow(output_mat, ifree, &ncols, &cols, NULL));

      // Stick in the intersecting values
      for (col = 0; col < ncols_mod; col++) {
         if (cols_mod[col] != -1) {
            col_indices[counter] = cols_mod[col];
            v[counter] = vals_copy[col];
            counter = counter + 1;
         }
      }

      // Lumped term to add to the diagonal
      lump_vals[ifree - global_row_start] = lump_sum;
   }
   row_ptr[local_rows] = counter;

   PetscCall(PetscFree(cols_mod));
   PetscCall(PetscFree(vals_copy));

   // Set the values into the existing output_mat
   // We can't use MatSetPreallocationCOO here, as that resets the sparsity and
   // values of output_mat, which would lose the existing values when adding with alpha
   // and any entries of output_mat that aren't in input_mat
   // We also can't MatGetRow on output_mat once we start setting values, hence the
   // intersecting values have been stored above
   if (alpha_present) {
      // If alpha is present, we add the values to the output matrix
      insert_mode = ADD_VALUES;
   } else {
      // Otherwise we just copy the values across
      insert_mode = INSERT_VALUES;
   }
   for (ifree = 0; ifree < local_rows; ifree++) {
      ncols = (PetscInt)(row_ptr[ifree + 1] - row_ptr[ifree]);
      if (ncols != 0) {
         row_global = ifree + global_row_start;
         PetscCall(MatSetValues(output_mat, one, &row_global, ncols,
               &col_indices[row_ptr[ifree]],
               &v[row_ptr[ifree]], insert_mode));
      }
   }
   // Can't mix insert and add without an assembly in between
   if (!alpha_present) {
      PetscCall(MatAssemblyBegin(output_mat, MAT_FLUSH_ASSEMBLY));
      PetscCall(MatAssemblyEnd(output_mat, MAT_FLUSH_ASSEMBLY));
   }
   // Add the lumped terms to the diagonal
   for (ifree = global_row_start; ifree < global_row_end_plus_one; ifree++) {
      PetscCall(MatSetValues(output_mat, one, &ifree, one, &ifree,
            &lump_vals[ifree - global_row_start], ADD_VALUES));
   }

   PetscCall(MatAssemblyBegin(output_mat, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(output_mat, MAT_FINAL_ASSEMBLY));

   PetscCall(PetscFree(col_indices));
   PetscCall(PetscFree(v));
   PetscCall(PetscFree(lump_vals));
   PetscCall(PetscFree(row_ptr));

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode MatSetAllValues(Mat input_mat, PetscScalar val)
{
   // Sets all the values in the matrix to be val

   // ~~~~~~~~~~
#if defined(PETSC_HAVE_KOKKOS)
   Mat A_array;
   MatType mat_type;
   PetscBool is_kokkos;
#endif
   // ~~~~~~~~~~

   PetscFunctionBegin;

#if defined(PETSC_HAVE_KOKKOS)

   PetscCall(MatGetType(input_mat, &mat_type));
   PetscCall(PetscStrcmpAny(mat_type, &is_kokkos, MATMPIAIJKOKKOS, MATSEQAIJKOKKOS, MATAIJKOKKOS, ""));
   if (is_kokkos) {

      A_array = input_mat;
      MatSetAllValues_kokkos(&A_array, val);

   } else {
      PetscCall(MatSetAllValues_cpu(input_mat, val));
   }
#else
   PetscCall(MatSetAllValues_cpu(input_mat, val));
#endif

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode MatSetAllValues_cpu(Mat input_mat, PetscScalar val)
{
   // Sets all the values in the local and nonlocal parts of input_mat to be val

   // ~~~~~~~~~~
   PetscMPIInt comm_size;
   MPI_Comm MPI_COMM_MATRIX;
   Mat Ad, Ao;
   const PetscInt *colmap = NULL;
   PetscScalar *vals = NULL;
   PetscInt nz, i_loc, n;
   const PetscInt *ia = NULL;
   PetscBool done;
   // ~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)input_mat, &MPI_COMM_MATRIX));
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));

   if (comm_size != 1) {
      PetscCall(MatMPIAIJGetSeqAIJ(input_mat, &Ad, &Ao, &colmap));
   } else {
      Ad = input_mat;
   }

   // MatSeqAIJGetArrayWrite returns a pointer sized to exactly the nonzeros,
   // so a whole-array assignment sets all the values in the local part
   // (in C the number of nonzeros is the last entry of the CSR row pointer, which
   // is how the Fortran interface sizes the array)
   PetscCall(MatGetRowIJ(Ad, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, NULL, &done));
   nz = ia[n];
   PetscCall(MatRestoreRowIJ(Ad, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, NULL, &done));
   PetscCall(MatSeqAIJGetArrayWrite(Ad, &vals));
   for (i_loc = 0; i_loc < nz; i_loc++) vals[i_loc] = val;
   PetscCall(MatSeqAIJRestoreArrayWrite(Ad, &vals));

   // Now the non-local part
   if (comm_size != 1) {
      PetscCall(MatGetRowIJ(Ao, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, NULL, &done));
      nz = ia[n];
      PetscCall(MatRestoreRowIJ(Ao, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, NULL, &done));
      PetscCall(MatSeqAIJGetArrayWrite(Ao, &vals));
      for (i_loc = 0; i_loc < nz; i_loc++) vals[i_loc] = val;
      PetscCall(MatSeqAIJRestoreArrayWrite(Ao, &vals));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode mat_duplicate_copy_plus_diag(Mat input_mat, PetscBool reuse, Mat *output_mat)
{
   // Wrapper around mat_duplicate_copy_plus_diag_kokkos and mat_duplicate_copy_plus_diag_cpu

   // ~~~~~~~~~~
#if defined(PETSC_HAVE_KOKKOS)
   Mat A_array, B_array = NULL;
   int reuse_int;
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

   PetscCall(MatGetType(input_mat, &mat_type));
   PetscCall(PetscStrcmpAny(mat_type, &is_kokkos, MATMPIAIJKOKKOS, MATSEQAIJKOKKOS, MATAIJKOKKOS, ""));
   if (is_kokkos) {

      reuse_int = 0;
      if (reuse) reuse_int = 1;

      A_array = input_mat;
      if (reuse) B_array = *output_mat;
      mat_duplicate_copy_plus_diag_kokkos(&A_array, reuse_int, &B_array);
      *output_mat = B_array;

      // If debugging do a comparison between CPU and Kokkos results
      if (kokkos_debug()) {

         // If we're doing reuse and debug, then the cpu version reuses the output
         // matrix itself (as that has the sparsity the cpu version expects) and so
         // we compare against a copy of the kokkos result
         if (reuse) {
            temp_mat = *output_mat;
            PetscCall(MatConvert(*output_mat, MATSAME, MAT_INITIAL_MATRIX, &temp_mat_compare));
         } else {
            temp_mat_compare = *output_mat;
         }

         // Debug check if the CPU and Kokkos versions are the same
         PetscCall(mat_duplicate_copy_plus_diag_cpu(input_mat, reuse, &temp_mat));

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
            PetscCall(MatDestroy(output_mat));
         } else {
            PetscCall(MatDestroy(&temp_mat_compare));
         }
         *output_mat = temp_mat;

      }

   } else {

      PetscCall(mat_duplicate_copy_plus_diag_cpu(input_mat, reuse, output_mat));

   }
#else
   PetscCall(mat_duplicate_copy_plus_diag_cpu(input_mat, reuse, output_mat));
#endif

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode mat_duplicate_copy_plus_diag_cpu(Mat input_mat, PetscBool reuse, Mat *output_mat)
{
   // Duplicates and copies the values from input matrix into the output mat, but ensures
   // there are always diagonal entries present that are set to zero if absent

   // ~~~~~~~~~~
   PetscInt ncols, ifree, col, i_loc;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt global_col_start, global_col_end_plus_one;
   const PetscInt *cols = NULL;
   // Matrix entries filled by MatGetRow are PetscScalar
   const PetscScalar *vals = NULL;
   PetscInt *d_nnz, *o_nnz;
   PetscInt row_array[1];
   PetscInt one = 1;
   MPI_Comm MPI_COMM_MATRIX;
   MatType mat_type;
   PetscBool diag_present;

   // ~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)input_mat, &MPI_COMM_MATRIX));

   // Get the local sizes
   PetscCall(MatGetLocalSize(input_mat, &local_rows, &local_cols));
   PetscCall(MatGetSize(input_mat, &global_rows, &global_cols));
   // This returns the global index of the local portion of the matrix
   PetscCall(MatGetOwnershipRange(input_mat, &global_row_start, &global_row_end_plus_one));
   PetscCall(MatGetOwnershipRangeColumn(input_mat, &global_col_start, &global_col_end_plus_one));

   // We may be reusing with the same sparsity
   if (!reuse) {
      PetscCall(MatCreate(MPI_COMM_MATRIX, output_mat));
      PetscCall(MatSetSizes(*output_mat, local_rows, local_cols,
                        global_rows, global_cols));
      // Match the output type
      PetscCall(MatGetType(input_mat, &mat_type));
      PetscCall(MatSetType(*output_mat, mat_type));

      // Count the entries in each row, plus the diagonal if it's absent, so we can exactly preallocate
      PetscCall(PetscCalloc1(local_rows, &d_nnz));
      PetscCall(PetscCalloc1(local_rows, &o_nnz));
      for (ifree = global_row_start; ifree < global_row_end_plus_one; ifree++) {
         PetscCall(MatGetRow(input_mat, ifree, &ncols, &cols, NULL));
         i_loc = ifree - global_row_start;
         diag_present = PETSC_FALSE;
         for (col = 0; col < ncols; col++) {
            if (cols[col] == ifree) diag_present = PETSC_TRUE;
            if (cols[col] >= global_col_start && cols[col] < global_col_end_plus_one) {
               d_nnz[i_loc] = d_nnz[i_loc] + 1;
            } else {
               o_nnz[i_loc] = o_nnz[i_loc] + 1;
            }
         }
         if (!diag_present) d_nnz[i_loc] = d_nnz[i_loc] + 1;
         PetscCall(MatRestoreRow(input_mat, ifree, &ncols, &cols, NULL));
      }
      PetscCall(MatXAIJSetPreallocation(*output_mat, one, d_nnz, o_nnz,
               NULL, NULL));
      PetscCall(PetscFree(d_nnz));
      PetscCall(PetscFree(o_nnz));
   }

   // Don't set any off processor entries so no need for a reduction when assembling
   PetscCall(MatSetOption(*output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
   PetscCall(MatSetOption(*output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));

   // Now go and fill the new matrix
   // Loop over global row indices
   for (ifree = global_row_start; ifree < global_row_end_plus_one; ifree++) {

      // Get the row
      PetscCall(MatGetRow(input_mat, ifree, &ncols, &cols, &vals));
      row_array[0] = ifree;
      if (ncols > 0) {
         PetscCall(MatSetValues(*output_mat, one, row_array, ncols, cols, vals, INSERT_VALUES));
      }
      // Add a zero on the diagonal if it's not present
      diag_present = PETSC_FALSE;
      for (col = 0; col < ncols; col++) {
         if (cols[col] == ifree) diag_present = PETSC_TRUE;
      }
      if (!diag_present) {
         PetscCall(MatSetValue(*output_mat, ifree, ifree, PFLARE_ZERO, INSERT_VALUES));
      }

      // Must call otherwise petsc leaks memory
      PetscCall(MatRestoreRow(input_mat, ifree, &ncols, &cols, &vals));
   }

   PetscCall(MatAssemblyBegin(*output_mat, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(*output_mat, MAT_FINAL_ASSEMBLY));

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode MatAXPYWrapper(Mat y_mat, PetscScalar alpha, Mat x_mat)
{
   // Wrapper around MatAXPY_kokkos

   // ~~~~~~~~~~
#if defined(PETSC_HAVE_KOKKOS)
   MPI_Comm MPI_COMM_MATRIX;
   PetscMPIInt comm_size;
   Mat A_array, B_array;
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

   PetscCall(MatGetType(y_mat, &mat_type));
   PetscCall(PetscObjectGetComm((PetscObject)y_mat, &MPI_COMM_MATRIX));
   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));

   // If doing parallel Kokkos
   PetscCall(PetscStrcmpAny(mat_type, &is_kokkos, MATMPIAIJKOKKOS, MATAIJKOKKOS, ""));
   if (is_kokkos && comm_size != 1) {

      if (kokkos_debug()) {
         PetscCall(MatDuplicate(y_mat, MAT_COPY_VALUES, &temp_mat));
      }

      A_array = y_mat;
      B_array = x_mat;
      MatAXPY_kokkos(&A_array, alpha, &B_array);

      // If debugging do a comparison between CPU and Kokkos results
      if (kokkos_debug()) {

         PetscCall(MatAXPY(temp_mat, alpha, x_mat, DIFFERENT_NONZERO_PATTERN));

         PetscCall(MatAXPY(temp_mat, PFLARE_MINUS_ONE, y_mat, DIFFERENT_NONZERO_PATTERN));
         // Find the biggest entry in the difference
         PetscCall(MatCreateVecs(temp_mat, NULL, &max_vec));
         PetscCall(MatGetRowMaxAbs(temp_mat, max_vec, NULL));
         PetscCall(VecMax(max_vec, &row_loc, &normy));
         PetscCall(VecDestroy(&max_vec));

         if (normy > PFLARE_TOL_MATFREE_12 || PetscIsNanReal(normy)) {
            //call MatFilter(temp_mat, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
            //call MatView(temp_mat, PETSC_VIEWER_STDOUT_WORLD, ierr)
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Diff Kokkos and CPU MatAXPY %g row %" PetscInt_FMT, (double)normy, row_loc);
         }
         PetscCall(MatDestroy(&temp_mat));
      }

   } else {

      PetscCall(MatAXPY(y_mat, alpha, x_mat, DIFFERENT_NONZERO_PATTERN));

   }
#else
   PetscCall(MatAXPY(y_mat, alpha, x_mat, DIFFERENT_NONZERO_PATTERN));
#endif

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode MatCreateSubMatrixWrapper(Mat input_mat, IS is_row, IS is_col, MatReuse reuse, Mat *output_mat,
   const int *our_level, const PetscBool *is_row_fine, const PetscBool *is_col_fine, void *kokkos_is_views_handle)
{
   // Wrapper around MatCreateSubMatrix_kokkos
   // Only works if the input IS have the same parallel row/column distribution
   // as the matrices
   // is_col must be sorted
   // When our_level is supplied (i.e. the kokkos path uses precomputed per-level
   // fine/coarse IS views), kokkos_is_views_handle MUST also be supplied; it
   // is the per-PCAIR handle stored on air_multigrid_data.
   // An absent kokkos_is_views_handle is NULL, which is the same as the Fortran
   // passing c_null_ptr when it is absent

   // ~~~~~~~~~~
#if defined(PETSC_HAVE_KOKKOS)
   MPI_Comm MPI_COMM_MATRIX;
   PetscMPIInt comm_size;
   Mat A_array, B_array;
   IS is_row_ptr, is_col_ptr;
   int reuse_int, our_level_int, is_row_fine_int, is_col_fine_int;
   PetscBool reuse_logical;
   MatType mat_type;
   Mat temp_mat = NULL;
   PetscReal normy;
   Vec max_vec;
   PetscInt row_loc;
   void *handle_local;
   PetscBool is_kokkos;
#endif
   // ~~~~~~~~~~

   PetscFunctionBegin;

#if defined(PETSC_HAVE_KOKKOS)

   PetscCall(MatGetType(input_mat, &mat_type));
   PetscCall(PetscObjectGetComm((PetscObject)input_mat, &MPI_COMM_MATRIX));
   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));

   // If doing parallel Kokkos
   PetscCall(PetscStrcmpAny(mat_type, &is_kokkos, MATMPIAIJKOKKOS, MATSEQAIJKOKKOS, MATAIJKOKKOS, ""));
   if (is_kokkos) {

      // Are we reusing
      reuse_logical = reuse == MAT_REUSE_MATRIX ? PETSC_TRUE : PETSC_FALSE;
      reuse_int = 0;
      if (reuse_logical) reuse_int = 1;

      A_array = input_mat;
      B_array = *output_mat;
      is_row_ptr = is_row;
      is_col_ptr = is_col;

      our_level_int = -1;
      is_row_fine_int = 0;
      is_col_fine_int = 0;
      handle_local = NULL;

      if (our_level) {
         our_level_int = *our_level;
      }
      if (is_row_fine) {
         if (*is_row_fine) is_row_fine_int = 1;
      }
      if (is_col_fine) {
         if (*is_col_fine) is_col_fine_int = 1;
      }
      if (kokkos_is_views_handle) {
         handle_local = kokkos_is_views_handle;
      }
      PetscCheck(!(our_level_int != -1 && !handle_local), PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "MatCreateSubMatrixWrapper: our_level set but kokkos_is_views_handle is null");

      MatCreateSubMatrix_kokkos(&A_array, &is_row_ptr, &is_col_ptr,
               reuse_int, &B_array,
               handle_local,
               our_level_int, is_row_fine_int, is_col_fine_int);

      *output_mat = B_array;

      // If debugging do a comparison between CPU and Kokkos results
      if (kokkos_debug()) {

         PetscCall(MatCreateSubMatrix(input_mat, is_row, is_col,
               MAT_INITIAL_MATRIX, &temp_mat));

         PetscCall(MatAXPY(temp_mat, PFLARE_MINUS_ONE, *output_mat, DIFFERENT_NONZERO_PATTERN));
         // Find the biggest entry in the difference
         PetscCall(MatCreateVecs(temp_mat, NULL, &max_vec));
         PetscCall(MatGetRowMaxAbs(temp_mat, max_vec, NULL));
         PetscCall(VecMax(max_vec, &row_loc, &normy));
         PetscCall(VecDestroy(&max_vec));

         if (normy > PFLARE_TOL_MATFREE_12 || PetscIsNanReal(normy)) {
            //call MatFilter(temp_mat, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
            //call MatView(temp_mat, PETSC_VIEWER_STDOUT_WORLD, ierr)
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Diff Kokkos and CPU MatCreateSubMatrix %g row %" PetscInt_FMT, (double)normy, row_loc);
         }
         PetscCall(MatDestroy(&temp_mat));
      }

   } else {

      PetscCall(MatCreateSubMatrix(input_mat, is_row, is_col,
            reuse, output_mat));

   }
#else
   PetscCall(MatCreateSubMatrix(input_mat, is_row, is_col,
         reuse, output_mat));
#endif

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode generate_identity(Mat input_mat, Mat *output_mat)
{
   // Returns an assembled identity of matching dimension/type to the input

   // ~~~~~~~~~~
   PetscInt ifree, local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt one = 1;
   PetscInt *d_nnz, *o_nnz;
   MPI_Comm MPI_COMM_MATRIX;
   MatType mat_type;
   PetscInt i_loc;

   // ~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)input_mat, &MPI_COMM_MATRIX));

   // Get the local sizes
   PetscCall(MatGetLocalSize(input_mat, &local_rows, &local_cols));
   PetscCall(MatGetSize(input_mat, &global_rows, &global_cols));
   // This returns the global index of the local portion of the matrix
   PetscCall(MatGetOwnershipRange(input_mat, &global_row_start, &global_row_end_plus_one));

   PetscCall(MatCreate(MPI_COMM_MATRIX, output_mat));
   PetscCall(MatSetSizes(*output_mat, local_rows, local_cols,
                    global_rows, global_cols));
   // Match the output type
   PetscCall(MatGetType(input_mat, &mat_type));
   PetscCall(MatSetType(*output_mat, mat_type));
   // One entry in the diagonal block of each row
   PetscCall(PetscMalloc1(local_rows, &d_nnz));
   PetscCall(PetscMalloc1(local_rows, &o_nnz));
   for (i_loc = 0; i_loc < local_rows; i_loc++) {
      d_nnz[i_loc] = 1;
      o_nnz[i_loc] = 0;
   }
   PetscCall(MatXAIJSetPreallocation(*output_mat, one, d_nnz, o_nnz,
            NULL, NULL));
   PetscCall(PetscFree(d_nnz));
   PetscCall(PetscFree(o_nnz));

   // Don't set any off processor entries so no need for a reduction when assembling
   PetscCall(MatSetOption(*output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
   PetscCall(MatSetOption(*output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));

   // Set the diagonal
   for (ifree = global_row_start; ifree < global_row_end_plus_one; ifree++) {
      PetscCall(MatSetValue(*output_mat, ifree, ifree, PFLARE_ONE, INSERT_VALUES));
   }
   PetscCall(MatAssemblyBegin(*output_mat, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(*output_mat, MAT_FINAL_ASSEMBLY));

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode generate_identity_rect(Mat full_mat, IS rect_indices, Mat *output_mat)
{
   // Returns an assembled (rectangular) injector that pulls out points in the row_indices: looks like [I 0]
   // The rows are sized from rect_indices (local and global size)

   // ~~~~~~~~~~
   PetscInt i_loc, local_rows, local_cols, global_rows, global_cols;
   PetscInt global_col_start, global_col_end_plus_one;
   PetscInt global_row_start_rect, global_row_end_plus_one_rect;
   PetscInt local_indices_size, global_indices_size;
   PetscInt one = 1;
   PetscInt *d_nnz, *o_nnz;
   MPI_Comm MPI_COMM_MATRIX;
   MatType mat_type;
   const PetscInt *is_pointer;

   // ~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)full_mat, &MPI_COMM_MATRIX));

   // Get the local sizes
   PetscCall(MatGetLocalSize(full_mat, &local_rows, &local_cols));
   PetscCall(MatGetSize(full_mat, &global_rows, &global_cols));
   PetscCall(MatGetOwnershipRangeColumn(full_mat, &global_col_start, &global_col_end_plus_one));

   // Get the local and global sizes of the rows we're pulling out
   PetscCall(ISGetLocalSize(rect_indices, &local_indices_size));
   PetscCall(ISGetSize(rect_indices, &global_indices_size));

   PetscCall(MatCreate(MPI_COMM_MATRIX, output_mat));
   // Rectangular matrix
   PetscCall(MatSetSizes(*output_mat, local_indices_size, local_cols,
               global_indices_size, global_cols));
   // Match the output type
   PetscCall(MatGetType(full_mat, &mat_type));
   PetscCall(MatSetType(*output_mat, mat_type));

   // Get the indices we need
   PetscCall(ISGetIndices(rect_indices, &is_pointer));

   // One entry in each row
   PetscCall(PetscCalloc1(local_indices_size, &d_nnz));
   PetscCall(PetscCalloc1(local_indices_size, &o_nnz));
   for (i_loc = 0; i_loc < local_indices_size; i_loc++) {
      if (is_pointer[i_loc] >= global_col_start && is_pointer[i_loc] < global_col_end_plus_one) {
         d_nnz[i_loc] = 1;
      } else {
         o_nnz[i_loc] = 1;
      }
   }
   PetscCall(MatXAIJSetPreallocation(*output_mat, one, d_nnz, o_nnz,
            NULL, NULL));
   PetscCall(PetscFree(d_nnz));
   PetscCall(PetscFree(o_nnz));
   // This returns the global index of the local portion of the output
   PetscCall(MatGetOwnershipRange(*output_mat, &global_row_start_rect, &global_row_end_plus_one_rect));

   // Don't set any off processor entries so no need for a reduction when assembling
   PetscCall(MatSetOption(*output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
   PetscCall(MatSetOption(*output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));

   for (i_loc = 0; i_loc < local_indices_size; i_loc++) {
      PetscCall(MatSetValue(*output_mat, global_row_start_rect + i_loc, is_pointer[i_loc],
               PFLARE_ONE, INSERT_VALUES));
   }
   PetscCall(MatAssemblyBegin(*output_mat, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(*output_mat, MAT_FINAL_ASSEMBLY));

   PetscCall(ISRestoreIndices(rect_indices, &is_pointer));

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode generate_identity_is(Mat input_mat, IS indices, Mat *output_mat)
{
   // Returns an assembled identity of matching dimension/type to the input
   // but with ones only in the diagonals of the input IS
   // We use this to do the equivalent of veciscopy that doesn't have to be
   // copied back to the cpu from the gpu

   // ~~~~~~~~~~
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt local_indices_size, i_loc;
   PetscInt one = 1;
   PetscInt *d_nnz, *o_nnz;
   MPI_Comm MPI_COMM_MATRIX;
   MatType mat_type;
   const PetscInt *is_pointer;

   // ~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)input_mat, &MPI_COMM_MATRIX));

   // Get the local sizes
   PetscCall(ISGetLocalSize(indices, &local_indices_size));

   PetscCall(MatGetLocalSize(input_mat, &local_rows, &local_cols));
   PetscCall(MatGetSize(input_mat, &global_rows, &global_cols));
   // This returns the global index of the local portion of the matrix
   PetscCall(MatGetOwnershipRange(input_mat, &global_row_start, &global_row_end_plus_one));

   PetscCall(MatCreate(MPI_COMM_MATRIX, output_mat));
   PetscCall(MatSetSizes(*output_mat, local_rows, local_cols,
                    global_rows, global_cols));
   // Match the output type
   PetscCall(MatGetType(input_mat, &mat_type));
   PetscCall(MatSetType(*output_mat, mat_type));

   // Get the indices we need
   PetscCall(ISGetIndices(indices, &is_pointer));

   // One entry in the diagonal block of each row in the IS
   PetscCall(PetscCalloc1(local_rows, &d_nnz));
   PetscCall(PetscCalloc1(local_rows, &o_nnz));
   for (i_loc = 0; i_loc < local_indices_size; i_loc++) {
      d_nnz[is_pointer[i_loc] - global_row_start] = 1;
   }
   PetscCall(MatXAIJSetPreallocation(*output_mat, one, d_nnz, o_nnz,
            NULL, NULL));
   PetscCall(PetscFree(d_nnz));
   PetscCall(PetscFree(o_nnz));

   // Don't set any off processor entries so no need for a reduction when assembling
   PetscCall(MatSetOption(*output_mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
   PetscCall(MatSetOption(*output_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));

   // Set the diagonal
   for (i_loc = 0; i_loc < local_indices_size; i_loc++) {
      PetscCall(MatSetValue(*output_mat, is_pointer[i_loc], is_pointer[i_loc], PFLARE_ONE, INSERT_VALUES));
   }
   PetscCall(MatAssemblyBegin(*output_mat, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(*output_mat, MAT_FINAL_ASSEMBLY));

   PetscCall(ISRestoreIndices(indices, &is_pointer));

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode get_nnzs_petsc_sparse(Mat input_mat, PetscInt64 *nnzs)
{
   // Returns nnzs in a sparse petsc matrix

   // ~~~~~~~~~~
   PetscMPIInt comm_size;
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_nnzs_petsc, offdiag_nnzs_petsc;
   PetscInt64 local_nnzs;

   // ~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)input_mat, &MPI_COMM_MATRIX));
   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));

   // Get nnzs without using getrow
   PetscCall(MatGetNNZs_both(input_mat, &local_nnzs_petsc, &offdiag_nnzs_petsc));
   // The sum is a PetscInt, as in the Fortran, and is then widened
   local_nnzs = local_nnzs_petsc + offdiag_nnzs_petsc;

   // Do an accumulate if in parallel
   if (comm_size != 1) {
      PetscCallMPI(MPI_Allreduce(&local_nnzs, nnzs, 1, MPIU_INT64, MPI_SUM, MPI_COMM_MATRIX));
   } else {
      *nnzs = local_nnzs;
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

//-------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode MatGetNNZs_both(Mat input_mat, PetscInt *nnzs_local, PetscInt *nnzs_nonlocal)
{
   // Returns the number of nonzeros in the local and nonlocal parts of input_mat

   // ~~~~~~~~~~
   PetscMPIInt comm_size;
   MPI_Comm MPI_COMM_MATRIX;
   MatType mat_type;
   Mat Ad, Ao;
   const PetscInt *colmap = NULL, *ia = NULL, *ja = NULL;
   PetscInt n, shift = 0;
   PetscBool symmetric = PETSC_FALSE, inodecompressed = PETSC_FALSE, done, is_diagonal;
   // ~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)input_mat, &MPI_COMM_MATRIX));
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));
   *nnzs_nonlocal = 0;

   // A diagonal matrix has one nonzero per row
   PetscCall(MatGetType(input_mat, &mat_type));
   PetscCall(PetscStrcmp(mat_type, MATDIAGONAL, &is_diagonal));
   if (is_diagonal) {
      PetscCall(MatGetLocalSize(input_mat, nnzs_local, NULL));
      PetscFunctionReturn(PETSC_SUCCESS);
   }

   if (comm_size != 1) {
      PetscCall(MatMPIAIJGetSeqAIJ(input_mat, &Ad, &Ao, &colmap));
   } else {
      Ad = input_mat;
   }

   // The number of nonzeros is the last entry of the CSR row pointer
   PetscCall(MatGetRowIJ(Ad, shift, symmetric, inodecompressed, &n, &ia, &ja, &done));
   *nnzs_local = ia[n];
   PetscCall(MatRestoreRowIJ(Ad, shift, symmetric, inodecompressed, &n, &ia, &ja, &done));

   if (comm_size != 1) {
      PetscCall(MatGetRowIJ(Ao, shift, symmetric, inodecompressed, &n, &ia, &ja, &done));
      *nnzs_nonlocal = ia[n];
      PetscCall(MatRestoreRowIJ(Ao, shift, symmetric, inodecompressed, &n, &ia, &ja, &done));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

//-------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode svd(const PetscScalar input[], PetscInt input_rows, PetscInt input_cols, PetscScalar U[], PetscReal sigma[], PetscScalar VT[])
{
   // The arrays are column-major, input is input_rows x input_cols, U is
   // input_rows x input_rows, sigma has min(input_rows, input_cols) entries and
   // VT is input_cols x input_cols

   // ~~~~~~~~~~~~~~~~

   // Matrix entries are PetscScalar; singular values (sigma) are genuinely real
   PetscScalar *tmp_input;
   PetscScalar *WORK;
   // BLAS/LAPACK integer arguments must be PetscBLASInt
   PetscBLASInt LWORK, M, N, info;
   PetscInt i_loc;

   // ~~~~~~~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscMalloc1(input_rows * input_cols, &tmp_input));
   for (i_loc = 0; i_loc < input_rows * input_cols; i_loc++) tmp_input[i_loc] = input[i_loc];
   PetscCall(PetscBLASIntCast(input_rows, &M));
   PetscCall(PetscBLASIntCast(input_cols, &N));
   LWORK = 2 * PetscMax(PetscMax(1, 3 * PetscMin(M, N) + PetscMax(M, N)), 5 * PetscMin(M, N));
   PetscCall(PetscMalloc1(LWORK, &WORK));

   PetscCallBLAS("LAPACKgesvd", LAPACKgesvd_("A", "A", &M, &N, tmp_input, &M,
         sigma, U, &M, VT, &N, WORK, &LWORK, &info));

   PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "SVD fail");
   PetscCall(PetscFree(WORK));
   PetscCall(PetscFree(tmp_input));

   PetscFunctionReturn(PETSC_SUCCESS);
}

//-------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode pseudo_inv(const PetscScalar input[], PetscInt input_rows, PetscInt input_cols, PetscScalar output[])
{
   // ~~~~~~~~~~~~~~~~

   // Matrix entries are PetscScalar; singular values (sigma) are genuinely real
   // The pseudo inverse of an m x n matrix is n x m (this routine assumes
   // square input, see the gemm below)
   // The arrays are column-major, input is input_rows x input_cols and
   // output is input_cols x input_rows
   PetscScalar *U;
   PetscReal *sigma;
   PetscScalar *VT;

   PetscInt iloc, jloc, n_sigma;
   PetscReal sigma_max, sigma_cutoff;
   // Kind-correct BLAS integer/real arguments for the dgemm call
   PetscBLASInt n_bl, lda_bl;
   PetscScalar blas_one, blas_zero;

   // ~~~~~~~~~~~~~~~~

   PetscFunctionBegin;

   n_sigma = PetscMin(input_rows, input_cols);
   PetscCall(PetscMalloc1(input_rows * input_rows, &U));
   PetscCall(PetscMalloc1(n_sigma, &sigma));
   PetscCall(PetscMalloc1(input_cols * input_cols, &VT));

   // Compute the svd
   PetscCall(svd(input, input_rows, input_cols, U, sigma, VT));

   // Now the pseudoinverse is V * inv(sigma) * U^T
   // and sigma is diagonal
   // So scale each column of U (given the transpose)
   // Drop a singular value if it is below the absolute floor (as before) OR
   // below the standard RELATIVE pseudo-inverse cutoff, ie a fixed fraction of
   // this block's largest singular value (numpy.linalg.pinv rcond, pyamg
   // pinv_array). Taking the max of the two keeps every drop the original
   // absolute test made - removing the absolute floor regresses the
   // ex6f -pc_air_constrain_z tests, which rely on it to discard blocks that
   // are numerically zero - while also discarding directions that are
   // negligible relative to the block, which matters once there is more than
   // one constraint vector.
   // Note neither test can protect a 1x1 block whose single value is small but
   // well above the floor: the caller must decide whether the block carries
   // information at all, see PFLARE_TOL_CONSTRAIN_REL in
   // constrain_grid_transfer.
   sigma_max = 0.0;
   for (iloc = 0; iloc < n_sigma; iloc++) {
      if (PetscAbsReal(sigma[iloc]) > sigma_max) sigma_max = PetscAbsReal(sigma[iloc]);
   }
   sigma_cutoff = PetscMax(PFLARE_TOL_SIGMA_DROP,
                      PFLARE_TOL_SIGMA_DROP_REL * sigma_max);
   for (iloc = 0; iloc < input_rows; iloc++) {
      if (PetscAbsReal(sigma[iloc]) > sigma_cutoff) {
         for (jloc = 0; jloc < input_rows; jloc++) U[jloc + iloc * input_rows] = U[jloc + iloc * input_rows] / sigma[iloc];
      } else {
         for (jloc = 0; jloc < input_rows; jloc++) U[jloc + iloc * input_rows] = 0.0;
      }
   }

   // Do the matmatmult, making sure to transpose both
   PetscCall(PetscBLASIntCast(input_rows, &n_bl));
   PetscCall(PetscBLASIntCast(input_rows, &lda_bl));
   blas_one = 1.0;
   blas_zero = 0.0;
   PetscCallBLAS("BLASgemm", BLASgemm_("T", "T", &n_bl, &n_bl, &n_bl,
            &blas_one, VT, &lda_bl,
            U, &lda_bl,
            &blas_zero, output, &lda_bl));

   // nan check
   for (iloc = 0; iloc < input_rows * input_cols; iloc++) {
      PetscCheck(!PetscIsNanScalar(output[iloc]), PETSC_COMM_SELF, PETSC_ERR_FP, "NaN in pseudo inverse");
   }

   PetscCall(PetscFree(U));
   PetscCall(PetscFree(sigma));
   PetscCall(PetscFree(VT));

   PetscFunctionReturn(PETSC_SUCCESS);
}

//-------------------------------------------------------------------------------------------------------------------------------

// Computes the symbolic sparsity of A*B (values not needed).
// Replaces the old C shim mat_mat_symbolic_c now that the PETSc
// Fortran MatProduct interface (incl. MATPRODUCT_AB) exists.
PETSC_INTERN PetscErrorCode mat_mat_symbolic(Mat A, Mat B, Mat *C)
{
   MatType mat_type_a, mat_type_b;
   MPI_Comm MPI_COMM_MATRIX;
   PetscMPIInt comm_size;

   // ~~~~~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)A, &MPI_COMM_MATRIX));
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));

   PetscCall(MatGetType(A, &mat_type_a));
   PetscCall(MatGetType(B, &mat_type_b));

   // The fill is PETSC_CURRENT, the value of PETSC_DEFAULT_REAL in the Fortran
   if (comm_size == 1) {
      // For some reason in serial matduplicate is not defined on unassembled matrices
      // ie we call a matduplicate on the symbolic sparsity_mat returned from this
      // So we just do an ordinary matmatmult in serial
      PetscCall(MatMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_CURRENT, C));
   } else {
      PetscCall(MatProductCreate(A, B, NULL, C));
      PetscCall(MatProductSetType(*C, MATPRODUCT_AB));
      PetscCall(MatProductSetAlgorithm(*C, "default"));
      PetscCall(MatProductSetFill(*C, PETSC_CURRENT));
      PetscCall(MatProductSetFromOptions(*C));
      PetscCall(MatProductSymbolic(*C));
      PetscCall(MatProductClear(*C));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

//-------------------------------------------------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------------------------------------------------

// Prints the stats and timings (the lines the Fortran printed with print *) to the
// C stdout, ie file descriptor 1, and flushes it
// This is deliberately not PetscPrintf: petsc4py replaces PetscVFPrintf with a
// routine that writes through Python's sys.stdout, which in Jupyter is not file
// descriptor 1, so output through PetscPrintf cannot be captured by redirecting
// that descriptor (as the notebooks in notebooks/ do), whereas the Fortran print *
// always wrote to it

PETSC_INTERN PetscErrorCode pflare_print(const char format[], ...)
{
   va_list args;
   int err;

   PetscFunctionBegin;
   va_start(args, format);
   err = vfprintf(stdout, format, args);
   va_end(args);
   PetscCheck(err >= 0, PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "vfprintf() returned error code %d", err);
   err = fflush(stdout);
   PetscCheck(err == 0, PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "fflush() failed");
   PetscFunctionReturn(PETSC_SUCCESS);
}
