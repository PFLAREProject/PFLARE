// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>

//------------------------------------------------------------------------------------------------------------------------

// Computes the fixed sparsity terms of a gmres polynomial inverse in the newton basis
// with kokkos - keeping everything on the device
// The build phase has already happened on the fortran side (see build_newton_fixed_sparsity_start),
// so the output_mat comes in with values correct up to the power poly_sparsity_order, along with
// sparsity_mat (mat_sparsity_match), prod_save_mat (mat_product_save), status_output and
// output_first_complex_int
// This routine then adds in all the fixed sparsity terms, matching
// mat_mult_powers_share_sparsity_newton_cpu
PETSC_INTERN void mat_mult_powers_share_sparsity_newton_kokkos(Mat *input_mat, Mat *sparsity_mat, Mat *prod_save_mat, \
               const int prod_save_exists_int, const int num_terms, const int poly_sparsity_order, \
               PetscReal *coefficients, int *status_output, const int output_first_complex_int, \
               const PetscReal tol_zero, const int reuse_int_reuse_mat, Mat *reuse_mat, Mat *output_mat)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols;
   PetscInt global_row_start_temp, global_row_end_plus_one_temp;
   PetscInt global_col_start_temp, global_col_end_plus_one_temp;
   PetscInt rows_ao, cols_ao, rows_ad, cols_ad, size_cols;
   MatType mat_type;
   PetscInt one = 1;
   bool deallocate_submatrices = false;

   PetscCallVoid(MatGetType(*input_mat, &mat_type));
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   const bool prod_save_exists = prod_save_exists_int != 0;
   const bool output_first_complex = output_first_complex_int != 0;

   Mat mat_local_sparsity = NULL, mat_nonlocal_sparsity = NULL;
   Mat mat_local_input = NULL, mat_nonlocal_input = NULL;
   Mat mat_local_prod_save = NULL, mat_nonlocal_prod_save = NULL;

   // Get the comm
   PetscCallVoid(PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX));
   PetscCallVoid(MatGetLocalSize(*input_mat, &local_rows, &local_cols));
   // This returns the global index of the local portion of the matrix
   PetscCallVoid(MatGetOwnershipRange(*input_mat, &global_row_start_temp, &global_row_end_plus_one_temp));
   PetscCallVoid(MatGetOwnershipRangeColumn(*input_mat, &global_col_start_temp, &global_col_end_plus_one_temp));
   const PetscInt global_row_start = global_row_start_temp;
   const PetscInt global_col_start = global_col_start_temp;

   auto exec = PetscGetKokkosExecutionSpace();

   // We also copy the coefficients and status flags over to the device as we need them
   // These are both column major with num_terms rows and two columns
   // For the coefficients the first column is the real part of each root
   // and the second column the imaginary part
   const PetscInt flat_size = 2 * num_terms;
   auto coefficients_h = PetscScalarKokkosViewHost(coefficients, flat_size);
   auto coefficients_d = PetscScalarKokkosView("coefficients_d", flat_size);
   Kokkos::deep_copy(exec, coefficients_d, coefficients_h);
   auto status_h = intKokkosViewHost(status_output, flat_size);
   auto status_d = intKokkosView("status_d", flat_size);
   Kokkos::deep_copy(exec, status_d, status_h);
   // Log copy with petsc
   size_t bytes = flat_size * (sizeof(PetscReal) + sizeof(int));
   PetscCallVoid(PetscLogCpuToGpu(bytes));

   PetscInt *col_indices_off_proc_array;
   const PetscInt *colmap_mat_sparsity_match;
   IS col_indices, row_indices;
   Mat *submatrices;

   // Pull out the nonlocal parts of the input mat we need
   const PetscInt *colmap_input_mat;
   PetscInt cols_ao_input = 0;
   cols_ao = 0;
   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*input_mat, &mat_local_input, &mat_nonlocal_input, &colmap_input_mat));

      PetscCallVoid(MatMPIAIJGetSeqAIJ(*sparsity_mat, &mat_local_sparsity, &mat_nonlocal_sparsity, &colmap_mat_sparsity_match));
      PetscCallVoid(MatGetSize(mat_nonlocal_sparsity, &rows_ao, &cols_ao));
      PetscCallVoid(MatGetSize(mat_local_sparsity, &rows_ad, &cols_ad));
      PetscInt rows_ao_input;
      PetscCallVoid(MatGetSize(mat_nonlocal_input, &rows_ao_input, &cols_ao_input));

      if (prod_save_exists) PetscCallVoid(MatMPIAIJGetSeqAIJ(*prod_save_mat, &mat_local_prod_save, &mat_nonlocal_prod_save, NULL));

      // We need to pull out all the columns in the sparsity mat
      // and the nonlocal rows that correspond to the nonlocal columns
      // from the input mat
      PetscCallVoid(PetscMalloc1(cols_ad + cols_ao, &col_indices_off_proc_array));
      size_cols = cols_ad + cols_ao;
      for (PetscInt i = 0; i < cols_ad; i++)
      {
         col_indices_off_proc_array[i] = global_row_start + i;
      }
      for (PetscInt i = 0; i < cols_ao; i++)
      {
         col_indices_off_proc_array[cols_ad + i] = colmap_mat_sparsity_match[i];
      }

      // Create the sequential IS we want with the cols we want (written as global indices)
      PetscCallVoid(ISCreateGeneral(PETSC_COMM_SELF, size_cols, \
                  col_indices_off_proc_array, PETSC_USE_POINTER, &col_indices));
      PetscCallVoid(ISCreateGeneral(PETSC_COMM_SELF, cols_ao, \
                  colmap_mat_sparsity_match, PETSC_USE_POINTER, &row_indices));

      PetscCallVoid(MatSetOption(*input_mat, MAT_SUBMAT_SINGLEIS, PETSC_TRUE));
      // Now this will be doing comms to get the non-local rows we want and returns in a sequential matrix
      if (!reuse_int_reuse_mat)
      {
         PetscCallVoid(MatCreateSubMatrices(*input_mat, one, &row_indices, &col_indices, MAT_INITIAL_MATRIX, &submatrices));
         *reuse_mat = submatrices[0];
      }
      else
      {
         submatrices = new Mat[1];
         deallocate_submatrices = true;
         submatrices[0] = *reuse_mat;
         PetscCallVoid(MatCreateSubMatrices(*input_mat, one, &row_indices, &col_indices, MAT_REUSE_MATRIX, &submatrices));
      }
      PetscCallVoid(ISDestroy(&col_indices));
      PetscCallVoid(ISDestroy(&row_indices));
   }
   // In serial
   else
   {
      submatrices = new Mat[1];
      deallocate_submatrices = true;
      submatrices[0] = *input_mat;
      mat_local_input = *input_mat;
      mat_local_sparsity = *sparsity_mat;
      if (prod_save_exists) mat_local_prod_save = *prod_save_mat;
      cols_ad = local_cols;
      PetscCallVoid(PetscMalloc1(local_rows, &col_indices_off_proc_array));
      for (PetscInt i = 0; i < local_rows; i++)
      {
         col_indices_off_proc_array[i] = i;
      }
   }

   // Get the existing output mat
   Mat mat_local_output = NULL, mat_nonlocal_output = NULL;
   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*output_mat, &mat_local_output, &mat_nonlocal_output, NULL));
   }
   else
   {
      mat_local_output = *output_mat;
   }

   // ~~~~~~~~~~~~
   // Get pointers to the i,j on the device and Kokkos views to the values
   // The build phase has finished all the (potentially) host modifications to the output mat
   // before this routine is called
   // ~~~~~~~~~~~~
   const PetscInt *device_submat_i = nullptr, *device_submat_j = nullptr;
   PetscMemType mtype;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(submatrices[0], &device_submat_i, &device_submat_j, NULL, &mtype));
   Kokkos::View<const PetscScalar *> device_submat_vals;
   PetscCallVoid(MatSeqAIJGetKokkosView(submatrices[0], &device_submat_vals));

   const PetscInt *device_local_i_input = nullptr, *device_local_j_input = nullptr, *device_nonlocal_i_input = nullptr, *device_nonlocal_j_input = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_input, &device_local_i_input, &device_local_j_input, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_input, &device_nonlocal_i_input, &device_nonlocal_j_input, NULL, &mtype));
   Kokkos::View<const PetscScalar *> device_local_vals_input;
   Kokkos::View<const PetscScalar *> device_nonlocal_vals_input;
   PetscCallVoid(MatSeqAIJGetKokkosView(mat_local_input, &device_local_vals_input));
   if (mpi) PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal_input, &device_nonlocal_vals_input));

   const PetscInt *device_local_i_sparsity = nullptr, *device_local_j_sparsity = nullptr, *device_nonlocal_i_sparsity = nullptr, *device_nonlocal_j_sparsity = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_sparsity, &device_local_i_sparsity, &device_local_j_sparsity, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_sparsity, &device_nonlocal_i_sparsity, &device_nonlocal_j_sparsity, NULL, &mtype));
   Kokkos::View<const PetscScalar *> device_local_vals_sparsity;
   Kokkos::View<const PetscScalar *> device_nonlocal_vals_sparsity;
   PetscCallVoid(MatSeqAIJGetKokkosView(mat_local_sparsity, &device_local_vals_sparsity));
   if (mpi) PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal_sparsity, &device_nonlocal_vals_sparsity));

   // mat_product_save is guaranteed by the build phase to have identical sparsity to
   // sparsity_mat, so we only need its values and i pointers
   const PetscInt *device_local_i_prod = nullptr, *device_nonlocal_i_prod = nullptr;
   Kokkos::View<const PetscScalar *> device_local_vals_prod;
   Kokkos::View<const PetscScalar *> device_nonlocal_vals_prod;
   if (prod_save_exists)
   {
      PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_prod_save, &device_local_i_prod, NULL, NULL, &mtype));
      if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_prod_save, &device_nonlocal_i_prod, NULL, NULL, &mtype));
      PetscCallVoid(MatSeqAIJGetKokkosView(mat_local_prod_save, &device_local_vals_prod));
      if (mpi) PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal_prod_save, &device_nonlocal_vals_prod));
   }

   const PetscInt *device_local_i_output = nullptr, *device_local_j_output = nullptr, *device_nonlocal_i_output = nullptr, *device_nonlocal_j_output = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_output, &device_local_i_output, &device_local_j_output, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_output, &device_nonlocal_i_output, &device_nonlocal_j_output, NULL, &mtype));
   // Output values are accumulated into via += so we need read-write access
   Kokkos::View<PetscScalar *> device_local_vals_output;
   Kokkos::View<PetscScalar *> device_nonlocal_vals_output;
   PetscCallVoid(MatSeqAIJGetKokkosView(mat_local_output, &device_local_vals_output));
   if (mpi) PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal_output, &device_nonlocal_vals_output));

   // ~~~~~~~~~~~~~~
   // Build a mapping from the input matrix's nonlocal column indices to the
   // sparsity matrix's column space ("local" submat column space), which is defined as:
   //   [0..cols_ad-1] for local columns, [cols_ad..cols_ad+cols_ao-1] for sparsity colmap columns
   //
   // When doing the matrix-matrix product:
   // 1. We need to compare local cols from local rows
   // We need to access the local input matrix and we
   // can do that directly given local indices are the same
   //
   // 2. We need to compare nonlocal cols from non-local rows
   // We need to access submat for this which now only has the non-local rows in it
   // Those will have a "local" column index that matches col_indices_off_proc_array given
   // we create it with MatCreateSubMatrices
   //
   // 3. We need to compare nonlocal cols from local rows
   // We need to access the input_matrix for this
   // But (for higher order fixed sparsity) the colmap of the input matrix is not the same
   // as the colmap of the sparsity matrix
   // So below we create a mapping that converts from the input matrix's nonlocal column indices
   // to the "local" column indices of the submat (which correspond to the sparsity matrix's column space) for the nonlocal columns
   // If there are not matching entries in the sparsity colmap, we use a large sentinel value that will never
   // match any col_orig and preserves sorted order.
   //
   // This mapping is needed because the input matrix and sparsity matrix may have
   // different colmaps when poly_sparsity_order >= 2.
   // ~~~~~~~~~~~~~~

   // Use a sentinel larger than any valid column index
   const PetscInt COLMAP_NOT_FOUND = cols_ad + cols_ao + 1;

   auto input_nonlocal_to_submat_col_d = PetscIntKokkosView("input_nonlocal_to_submat_col_d", mpi ? cols_ao_input : 1);
   if (mpi && cols_ao_input > 0)
   {
      // Build the mapping on the host
      // Both colmaps are sorted, so we can do a merge-style scan
      auto input_nonlocal_to_submat_col_h = Kokkos::create_mirror_view(input_nonlocal_to_submat_col_d);
      PetscInt sparsity_colmap_idx = 0;
      for (PetscInt k = 0; k < cols_ao_input; k++)
      {
         PetscInt global_col = colmap_input_mat[k];
         // Advance the sparsity colmap index (both are sorted)
         while (sparsity_colmap_idx < cols_ao && colmap_mat_sparsity_match[sparsity_colmap_idx] < global_col)
         {
            sparsity_colmap_idx++;
         }
         if (sparsity_colmap_idx < cols_ao && colmap_mat_sparsity_match[sparsity_colmap_idx] == global_col)
         {
            input_nonlocal_to_submat_col_h(k) = cols_ad + sparsity_colmap_idx;
         }
         else
         {
            // Not found — use sentinel value that preserves sort order
            // Since colmap_input is sorted and colmap_sparsity is sorted,
            // if an entry is missing it's between two found entries,
            // so we assign a value that maintains monotonicity
            input_nonlocal_to_submat_col_h(k) = COLMAP_NOT_FOUND;
         }
      }
      Kokkos::deep_copy(exec, input_nonlocal_to_submat_col_d, input_nonlocal_to_submat_col_h);
      // Log copy with petsc
      bytes = input_nonlocal_to_submat_col_h.extent(0) * sizeof(PetscInt);
      PetscCallVoid(PetscLogCpuToGpu(bytes));
   }

   // ~~~~~~~~~~~~~~
   // Find maximum non-zeros per row for sizing scratch memory
   // ~~~~~~~~~~~~~~
   PetscInt sparsity_max_nnz = 0, sparsity_max_nnz_local = 0, sparsity_max_nnz_nonlocal = 0;
   if (local_rows > 0) {
      // Also consider sparsity matrix row width if needed
      Kokkos::parallel_reduce("FindMaxNNZSparsity", Kokkos::RangePolicy<>(exec, 0, local_rows),
         KOKKOS_LAMBDA(const PetscInt i, PetscInt& thread_max) {
            PetscInt row_nnz = device_local_i_sparsity[i + 1] - device_local_i_sparsity[i];
            thread_max = (row_nnz > thread_max) ? row_nnz : thread_max;
         },
         Kokkos::Max<PetscInt>(sparsity_max_nnz_local)
      );
      if (mpi)
      {
         Kokkos::parallel_reduce("FindMaxNNZSparsityNonLocal", Kokkos::RangePolicy<>(exec, 0, local_rows),
            KOKKOS_LAMBDA(const PetscInt i, PetscInt& thread_max) {
               PetscInt row_nnz = device_nonlocal_i_sparsity[i + 1] - device_nonlocal_i_sparsity[i];
               thread_max = (row_nnz > thread_max) ? row_nnz : thread_max;
            },
            Kokkos::Max<PetscInt>(sparsity_max_nnz_nonlocal)
         );
      }
      sparsity_max_nnz = sparsity_max_nnz_local + sparsity_max_nnz_nonlocal;
   }

   // ~~~~~~~~~~~~~
   // Now we have to be careful
   // mat_sparsity_match may not have diagonal entries in some rows
   // but we know our gmres polynomial inverse must
   // and the build phase guarantees our output_mat has diagonals
   // But when we write out to output_mat below, we assume it has the same sparsity as
   // mat_sparsity_match
   // So we have to track which rows don't have diagonals in mat_sparsity_match
   // so we can increment the writing out by one in output_mat
   // The reason we don't have to do this in the cpu version is because we are calling
   // matsetvalues to write out to output_mat, rather than writing out to the csr directly
   // We also record the position of the diagonal within each row of the sparsity mat
   // as the newton recurrence needs an identity row in one of its cases
   // ~~~~~~~~~~~~~

   auto found_diag_row_d = PetscIntKokkosView("found_diag_row_d", local_rows);
   auto diag_pos_row_d = PetscIntKokkosView("diag_pos_row_d", local_rows);
   Kokkos::deep_copy(exec, found_diag_row_d, 0);
   Kokkos::deep_copy(exec, diag_pos_row_d, -1);

   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

      // Row
      const PetscInt i = t.league_rank();
      // ncols_local
      const PetscInt ncols_local = device_local_i_sparsity[i + 1] - device_local_i_sparsity[i];
      const PetscInt row_index_global = i + global_row_start;

      // Loop over all the columns in this row of sparsity mat
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

         // Is this column the diagonal
         const bool is_diagonal = (device_local_j_sparsity[device_local_i_sparsity[i] + j] + global_col_start == row_index_global);
         // This will only happen on a max of one thread per row
         // The local block comes first in our scratch ordering, so j is also
         // the position of the diagonal in the combined row
         if (is_diagonal)
         {
            found_diag_row_d(i) = 1;
            diag_pos_row_d(i) = j;
         }
      });
   });

   // ~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~

   // Create a team policy with scratch memory allocation
   // We want scratch space for each row of max size sparsity_max_nnz per view
   const size_t per_view = ScratchScalarView::shmem_size(sparsity_max_nnz);

   Kokkos::TeamPolicy<> policy(exec, local_rows, Kokkos::AUTO());
   // We're gonna use the level 1 scratch as our column data is probably bigger than the level 0
   // We want three views in our scratch space, vals_prev, vals_temp and temp_vals
   policy.set_scratch_size(1, Kokkos::PerTeam(3 * per_view));

   // Execute with scratch memory
   Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const KokkosTeamMemberType& t) {

      // Row
      const PetscInt i = t.league_rank();

      // ncols_row_i is the total number of columns in this row of the sparsity mat
      PetscInt ncols_row_i = device_local_i_sparsity[i + 1] - device_local_i_sparsity[i];
      if (mpi) ncols_row_i += device_nonlocal_i_sparsity[i + 1] - device_nonlocal_i_sparsity[i];

      // Allocate views directly on scratch memory
      // Have to use views here given alignment issues
      // These are the vals_previous_power_temp, vals_power_temp and temp arrays
      // from the cpu version
      ScratchScalarView vals_prev(t.team_scratch(1), ncols_row_i);
      ScratchScalarView vals_temp(t.team_scratch(1), ncols_row_i);
      ScratchScalarView temp_vals(t.team_scratch(1), ncols_row_i);

      // How many local columns do we have in row i
      const PetscInt local_cols_row_i = device_local_i_sparsity[i + 1] - device_local_i_sparsity[i];

      // Loop over all the columns in this row of sparsity mat
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_row_i), [&](const PetscInt j) {

         // Fill vals_prev with the values of the sparsity mat
         if (j < local_cols_row_i)
         {
            vals_prev[j] = device_local_vals_sparsity(device_local_i_sparsity[i] + j);
         }
         // Nonlocal part
         else
         {
            vals_prev[j] = device_nonlocal_vals_sparsity(device_nonlocal_i_sparsity[i] + (j - local_cols_row_i));
         }
      });

      // Team barrier to ensure all threads have finished filling the scratch space
      t.team_barrier();

      // Adds contribution into position j of this row of the output mat
      // The output mat has the sparsity of the sparsity mat plus a guaranteed diagonal,
      // hence the diag_increm below
      auto add_val_output = [&](const PetscInt j, const PetscScalar contribution) {

         // If we're in the local part of the matrix
         if (j < local_cols_row_i)
         {
            PetscInt diag_increm = 0;
            // We need to increment the index we access by one
            // if we don't have a diagonal in the sparsity matrix
            // as we have one in the output_mat
            if (found_diag_row_d(i) == 0 && device_local_j_output[device_local_i_output[i] + j] >= i)
            {
               diag_increm = 1;
            }
            device_local_vals_output(device_local_i_output[i] + j + diag_increm) += contribution;
         }
         // Nonlocal part
         else
         {
            device_nonlocal_vals_output(device_nonlocal_i_output[i] + (j - local_cols_row_i)) += contribution;
         }
      };

      // ~~~~~~~~~~~~~~~~~~~~~~
      // This is the row-wise matrix product with fixed sparsity
      // It computes dest[match] += factor * src[j] * A_val for all the matching
      // indices between row i of the sparsity mat and the row of the input mat
      // given by each column j
      // dest must be initialized by the caller before this is called, and the caller
      // must have team barriers around this
      // We recompute which indices match every time for each power
      // They never change and we used to compute them once and store them in scratch space
      // but this then meant we used lots of memory per team and hence fewer
      // teams/threads in teams were used and hence we had less parallelism
      // ~~~~~~~~~~~~~~~~~~~~~~
      auto fixed_sparsity_product = [&](const ScratchScalarView &dest, const ScratchScalarView &src, const PetscScalar factor) {

         // This goes over all the local and non-local columns in row i
         Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_row_i), [&](const PetscInt j) {

            // ~~~~~~~~~
            // Do a search through the sorted arrays to find matching indices
            // ~~~~~~~~~

            // Get the row index for this column in sparsity mat
            PetscInt row_of_col_j;
            // Do we have this row locally or have we retrieved it from other ranks?
            const bool row_of_col_j_local = j < local_cols_row_i;
            // This is only accesssed below if row_of_col_j_local is true
            PetscInt local_cols_row_of_col_j = 0;
            if (row_of_col_j_local)
            {
               row_of_col_j = device_local_j_sparsity[device_local_i_sparsity[i] + j];
               local_cols_row_of_col_j = device_local_i_input[row_of_col_j + 1] - device_local_i_input[row_of_col_j];
            }
            else
            {
               row_of_col_j = device_nonlocal_j_sparsity[device_nonlocal_i_sparsity[i] + (j - local_cols_row_i)];
            }

            // Get how many local and non-local columns there are in the row of column j
            PetscInt ncols_row_of_col_j = 0;
            if (row_of_col_j_local)
            {
               ncols_row_of_col_j = local_cols_row_of_col_j;
               if (mpi) ncols_row_of_col_j += device_nonlocal_i_input[row_of_col_j + 1] - device_nonlocal_i_input[row_of_col_j];
            }
            else
            {
               ncols_row_of_col_j = device_submat_i[row_of_col_j + 1] - device_submat_i[row_of_col_j];
            }

            // We'll perform a search to find matching indices
            // We're matching indices in sparsity mat to those in submat
            // This is just an intersection between row i in the sparsity mat
            // and the row of column j in the input mat
            // This assumes column indices are already sorted
            PetscInt idx_col_of_row_i = 0;  // Index into original row i columns
            PetscInt idx_col_of_row_j = 0;  // Index into target row of column j

            while (idx_col_of_row_i < ncols_row_i && idx_col_of_row_j < ncols_row_of_col_j) {

               // The col_target is the column we are trying to match in the row of column j
               // We convert everything to the submat "local" column space for comparison, ie
               // the column indexing of [0..cols_ad-1 for local cols; cols_ad+k for sparsity colmap[k]]
               // When the input matrix and sparsity matrix have different colmaps
               // (poly_sparsity_order >= 2), we use the input_nonlocal_to_submat_col_d mapping
               // to convert the input matrix's nonlocal column indices to the sparsity colmap space
               PetscInt col_target;
               if (row_of_col_j_local)
               {
                  if (idx_col_of_row_j < local_cols_row_of_col_j)
                  {
                     col_target = device_local_j_input[device_local_i_input[row_of_col_j] + idx_col_of_row_j];
                  }
                  else
                  {
                     // This is the case where we need to access non-local columns in local rows of input_matrix
                     // and hence we need our mapping
                     // Convert nonlocal column index from input matrix's colmap space
                     // to the to "local" column index of submat
                     col_target = input_nonlocal_to_submat_col_d(device_nonlocal_j_input[device_nonlocal_i_input[row_of_col_j] + idx_col_of_row_j - local_cols_row_of_col_j]);
                  }
               }
               else
               {
                  col_target = device_submat_j[device_submat_i[row_of_col_j] + idx_col_of_row_j];
               }

               PetscInt col_orig;
               // If we're in the local part of the matrix
               if (idx_col_of_row_i < local_cols_row_i)
               {
                  col_orig = device_local_j_sparsity[device_local_i_sparsity[i] + idx_col_of_row_i];
               }
               // Nonlocal part
               else
               {
                  // Convert to "local" column index of submat by adding cols_ad
                  col_orig = device_nonlocal_j_sparsity[device_nonlocal_i_sparsity[i] + (idx_col_of_row_i - local_cols_row_i)] + cols_ad;
               }

               // Skip entries where the input column doesn't exist in the sparsity pattern
               if (col_target == COLMAP_NOT_FOUND) {
                  idx_col_of_row_j++;
               } else {
                  if (col_orig < col_target) {
                     // Original column is smaller, move to next original column
                     idx_col_of_row_i++;
                  } else if (col_orig > col_target) {
                     // Target column is smaller, move to next target column
                     idx_col_of_row_j++;
                  // We've found a matching index and hence we can do our compute
                  } else {

                     PetscReal val_target;
                     if (row_of_col_j_local)
                     {
                        if (idx_col_of_row_j < local_cols_row_of_col_j)
                        {
                           val_target = device_local_vals_input(device_local_i_input[row_of_col_j] + idx_col_of_row_j);
                        }
                        else
                        {
                           val_target = device_nonlocal_vals_input(device_nonlocal_i_input[row_of_col_j] + idx_col_of_row_j - local_cols_row_of_col_j);
                        }
                     }
                     else
                     {
                        val_target = device_submat_vals(device_submat_i[row_of_col_j] + idx_col_of_row_j);
                     }

                     // Has to be atomic! Potentially lots of contention so maybe not
                     // the most performant way to do this
                     // The order of operations here matches the cpu version, ie
                     // (factor * A) * src, as the tiny rounding differences otherwise
                     // compound over the recurrence at high polynomial order
                     Kokkos::atomic_add(&dest[idx_col_of_row_i], factor * val_target * src[j]);

                     // Move forward in both arrays
                     idx_col_of_row_i++;
                     idx_col_of_row_j++;
                  }
               }
            }
         });
      };

      // ~~~~~~~~~~~~~~~~~~~~~~
      // Now loop over the terms in the newton polynomial
      // This exactly matches the term loop in mat_mult_powers_share_sparsity_newton_cpu
      // All of the branching below only depends on the coefficients and status flags
      // and hence is identical for every row/team
      // The term here is 1-based to keep parity with the fortran
      // ~~~~~~~~~~~~~~~~~~~~~~

      int term = poly_sparsity_order + 1;
      bool skip_add = false;
      // If the fixed sparsity root is the second of a complex pair, we start one term earlier
      // so that we can compute the correct part of the fixed sparsity product, we just make sure not to add
      // anything to output_mat as it is already correct up to the fixed sparsity order
      if (coefficients_d(num_terms + term - 1) != 0.0 && !output_first_complex)
      {
         term = term - 1;
         skip_add = true;
      }

      // This loop skips the last coefficient
      while (term <= num_terms - 1)
      {
         const PetscReal root_real = coefficients_d(term - 1);
         const PetscReal root_imag = coefficients_d(num_terms + term - 1);

         // If real
         if (root_imag == 0.0)
         {
            // Skips eigenvalues that are numerically zero - see
            // the comment in calculate_gmres_polynomial_roots_newton
            if (Kokkos::abs(root_real) < tol_zero)
            {
               term = term + 1;
               continue;
            }

            const PetscReal inv_theta = 1.0 / root_real;
            const bool do_add = status_d(term - 1) != 1;

            // ~~~~~~~~~~~
            // Now can add the value to our matrix and initialize vals_temp
            // with the previous product before the A*prod subtraction
            // ~~~~~~~~~~~
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_row_i), [&](const PetscInt j) {
               if (do_add) add_val_output(j, inv_theta * vals_prev[j]);
               vals_temp[j] = vals_prev[j];
            });
            t.team_barrier();

            // This is the 1/theta_i * A * prod but where A * prod has fixed sparsity
            fixed_sparsity_product(vals_temp, vals_prev, -inv_theta);
            t.team_barrier();

            term = term + 1;
         }
         // If complex
         else
         {
            // Skips eigenvalues that are numerically zero - see
            // the comment in calculate_gmres_polynomial_roots_newton
            if (root_real * root_real + root_imag * root_imag < tol_zero)
            {
               term = term + 2;
               continue;
            }

            const PetscReal square_sum = 1.0 / (root_real * root_real + root_imag * root_imag);

            // If our fixed sparsity order falls on the first of a complex conjugate pair
            if (!skip_add)
            {
               const bool two_a_prod_included = status_d(num_terms + term - 1) == 1;

               // We skip the 2 * a * prod from the first root of a complex pair if that has already
               // been included in the output mat from the build phase
               Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_row_i), [&](const PetscInt j) {
                  temp_vals[j] = !two_a_prod_included ? 2.0 * root_real * vals_prev[j] : 0.0;
               });
               t.team_barrier();

               // This is the -A * prod
               fixed_sparsity_product(temp_vals, vals_prev, -1.0);
               t.team_barrier();

               // This is the p = p + 1/(a^2 + b^2) * temp
               // Here we also need to go back in and ensure 2 * a * prod is in temp if we skipped it
               // above. We know it is already in the output mat, but it has to be in temp when we
               // do the next product
               Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_row_i), [&](const PetscInt j) {
                  add_val_output(j, square_sum * temp_vals[j]);
                  if (two_a_prod_included && output_first_complex)
                  {
                     temp_vals[j] += 2.0 * root_real * vals_prev[j];
                  }
               });
               t.team_barrier();
            }
            // If our fixed sparsity order falls on the second of a complex conjugate pair
            else
            {
               // In this case we have already included both 2*a*prod - A * prod into the output mat
               // But we still have to compute the product for the next term
               // The problem here is that the sparsity mat has temp in it in this case, not
               // the old prod from whatever the previous loop is
               // In that case the build phase also outputs mat_product_save which is the old value
               // of prod but with the sparsity of the sparsity mat (with zeros if needed)

               // This case only occurs once for each row, so once we've hit this
               // we will always have our correct prod
               skip_add = false;

               Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_row_i), [&](const PetscInt j) {
                  // temp is output into the sparsity mat in this case
                  temp_vals[j] = vals_prev[j];

                  // If sparsity order is 1, the previous product will have been the identity
                  // and it isn't output into mat_product_save because that is a trivial case
                  // we can do ourselves
                  if (term == 1)
                  {
                     vals_prev[j] = (j == diag_pos_row_d(i)) ? 1.0 : 0.0;
                  }
                  // In the case the mat_product_save is not the identity, we need to pull its value out
                  // The build phase has guaranteed that mat_product_save has fixed sparsity
                  else
                  {
                     if (j < local_cols_row_i)
                     {
                        vals_prev[j] = device_local_vals_prod(device_local_i_prod[i] + j);
                     }
                     else
                     {
                        vals_prev[j] = device_nonlocal_vals_prod(device_nonlocal_i_prod[i] + (j - local_cols_row_i));
                     }
                  }
               });
               t.team_barrier();
            }

            // Now we compute the next product
            if (term <= num_terms - 2)
            {
               Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_row_i), [&](const PetscInt j) {
                  vals_temp[j] = vals_prev[j];
               });
               t.team_barrier();

               // This is prod = prod - 1/(a^2 + b^2) * A * temp
               fixed_sparsity_product(vals_temp, temp_vals, -square_sum);
               t.team_barrier();
            }

            term = term + 2;
         }

         // This should now have the value of prod in it
         Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_row_i), [&](const PetscInt j) {
            vals_prev[j] = vals_temp[j];
         });
         t.team_barrier();
      }

      // Final step if last root is real
      const PetscReal last_real = coefficients_d(num_terms - 1);
      const PetscReal last_imag = coefficients_d(2 * num_terms - 1);
      if (last_imag == 0.0 && Kokkos::abs(last_real) > tol_zero)
      {
         const PetscReal inv_theta = 1.0 / last_real;
         Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_row_i), [&](const PetscInt j) {
            add_val_output(j, inv_theta * vals_temp[j]);
         });
      }
   });

   Kokkos::fence();

   // Restore the read-only input/sparsity/submat/prod_save views
   PetscCallVoid(MatSeqAIJRestoreKokkosView(submatrices[0], &device_submat_vals));
   PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local_input, &device_local_vals_input));
   if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal_input, &device_nonlocal_vals_input));
   PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local_sparsity, &device_local_vals_sparsity));
   if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal_sparsity, &device_nonlocal_vals_sparsity));
   if (prod_save_exists)
   {
      PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local_prod_save, &device_local_vals_prod));
      if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal_prod_save, &device_nonlocal_vals_prod));
   }

   // The matching restore handles MatSeqAIJKokkosModifyDevice (clears sync state,
   // marks device modified, invalidates transpose/hermitian, bumps object state).
   PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local_output, &device_local_vals_output));
   if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal_output, &device_nonlocal_vals_output));

   if (deallocate_submatrices) delete[] submatrices;
   // We don't explicitly call matdestroysubmatrices because lord knows how we pass out submatrices back
   // to fortran - to prevent leaking memory we destroy the submatrices pointer ourself, and if we come back
   // into this routine with reuse we just create a new submatrices array
   if (mpi && !reuse_int_reuse_mat) PetscCallVoid(PetscFree(submatrices));
   (void)PetscFree(col_indices_off_proc_array);

   return;
}
