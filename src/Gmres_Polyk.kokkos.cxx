// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

//------------------------------------------------------------------------------------------------------------------------

// Compute matrix-matrix product with fixed order sparsity but with kokkos - keeping everything on the device
PETSC_INTERN void mat_mult_powers_share_sparsity_kokkos(Mat *input_mat, int poly_order, int poly_sparsity_order, PetscReal *coefficients, \
               int reuse_int_reuse_mat, Mat *reuse_mat, int reuse_int_cmat, Mat *output_mat)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols;
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt global_col_start, global_col_end_plus_one;
   PetscInt rows_ao, cols_ao, rows_ad, cols_ad, size_cols;
   MatType mat_type;
   Mat *matrix_powers, *mat_sparsity_match;
   PetscInt one = 1;
   bool deallocate_submatrices = false;

   MatGetType(*input_mat, &mat_type);
   // Are we in parallel?
   bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat_MPIAIJ *mat_mpi = nullptr;
   Mat mat_local_sparsity, mat_nonlocal_sparsity;

   // Get the comm
   PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX);
   MatGetLocalSize(*input_mat, &local_rows, &local_cols);
   // This returns the global index of the local portion of the matrix
   MatGetOwnershipRange(*input_mat, &global_row_start, &global_row_end_plus_one);
   MatGetOwnershipRangeColumn(*input_mat, &global_col_start, &global_col_end_plus_one);   

   // We also copy the coefficients over to the device as we need it
   PetscInt coeff_size = poly_order + 1;
   auto coefficients_h = PetscScalarKokkosViewHost(coefficients, coeff_size);
   auto coefficients_d = PetscScalarKokkosView("coefficients_d", coeff_size);
   Kokkos::deep_copy(coefficients_d, coefficients_h);       
   // Log copy with petsc
   size_t bytes = coefficients_h.extent(0) * sizeof(PetscReal);
   PetscLogCpuToGpu(bytes);     
   
   // Let's build up our matrix powers
   matrix_powers = new Mat[coeff_size - 1];
   matrix_powers[0] = *input_mat;
   // Compute the matrix powers
   for (int i = 1; i < poly_sparsity_order; i++)
   {
      MatMatMult(*input_mat, matrix_powers[i-1], \
            MAT_INITIAL_MATRIX, 1.5, &(matrix_powers[i]));
   }
   // This is the matrix whose sparsity we want to match
   mat_sparsity_match = &(matrix_powers[poly_sparsity_order - 1]);

   // Copy in the highest unconstrained power
   // Duplicate & copy the matrix, but ensure there is a diagonal present
   mat_duplicate_copy_plus_diag_kokkos(mat_sparsity_match, reuse_int_cmat, output_mat);

   PetscInt *col_indices_off_proc_array;
   IS col_indices;
   Mat *submatrices;

   // Pull out the local and nonlocal parts of the sparsity match we need
   if (mpi)
   {
      mat_mpi = (Mat_MPIAIJ *)(*mat_sparsity_match)->data;
      mat_local_sparsity = mat_mpi->A;
      mat_nonlocal_sparsity = mat_mpi->B;
      MatGetSize(mat_nonlocal_sparsity, &rows_ao, &cols_ao); 
      MatGetSize(mat_local_sparsity, &rows_ad, &cols_ad);

      PetscMalloc1(cols_ad + cols_ao, &col_indices_off_proc_array);
      size_cols = cols_ad + cols_ao;
      for (int i = 0; i < cols_ad; i++)
      {
         col_indices_off_proc_array[i] = global_row_start + i;
      }
      for (int i = 0; i < cols_ao; i++)
      {
         col_indices_off_proc_array[cols_ad + i] = mat_mpi->garray[i];
      }           
      
      // Create the sequential IS we want with the cols we want (written as global indices)
      ISCreateGeneral(PETSC_COMM_SELF, size_cols, \
                  col_indices_off_proc_array, PETSC_USE_POINTER, &col_indices);

      MatSetOption(*input_mat, MAT_SUBMAT_SINGLEIS, PETSC_TRUE); 
      // Now this will be doing comms to get the non-local rows we want and returns in a sequential matrix
      if (!reuse_int_reuse_mat)
      {
         MatCreateSubMatrices(*input_mat, one, &col_indices, &col_indices, MAT_INITIAL_MATRIX, &submatrices);
         *reuse_mat = submatrices[0];
      }
      else
      {
         submatrices = new Mat[1];
         deallocate_submatrices = true;
         submatrices[0] = *reuse_mat;
         MatCreateSubMatrices(*input_mat, one, &col_indices, &col_indices, MAT_REUSE_MATRIX, &submatrices);         
      }
      ISDestroy(&col_indices);
   }
   // In serial
   else
   {
      submatrices = new Mat[1];
      deallocate_submatrices = true;      
      submatrices[0] = *input_mat;
      mat_local_sparsity = *mat_sparsity_match;
      cols_ad = local_cols;
      PetscMalloc1(local_rows, &col_indices_off_proc_array);
      for (int i = 0; i < local_rows; i++)
      {
         col_indices_off_proc_array[i] = i;
      }
   }

   // Get the existing output mat
   Mat mat_local_output, mat_nonlocal_output;   
   if (mpi)
   {
      Mat_MPIAIJ *mat_mpi_output = (Mat_MPIAIJ *)(*output_mat)->data;
      mat_local_output = mat_mpi_output->A;
      mat_nonlocal_output = mat_mpi_output->B;     
   }
   else
   {
      mat_local_output = *output_mat;
   }    

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_submat_i = nullptr, *device_submat_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_submat_vals = nullptr;  
   MatSeqAIJGetCSRAndMemType(submatrices[0], &device_submat_i, &device_submat_j, &device_submat_vals, &mtype);  

   const PetscInt *device_local_i_sparsity = nullptr, *device_local_j_sparsity = nullptr, *device_nonlocal_i_sparsity = nullptr, *device_nonlocal_j_sparsity = nullptr;;
   PetscScalar *device_local_vals_sparsity = nullptr, *device_nonlocal_vals_sparsity = nullptr;  
   MatSeqAIJGetCSRAndMemType(mat_local_sparsity, &device_local_i_sparsity, &device_local_j_sparsity, &device_local_vals_sparsity, &mtype);
   if (mpi) MatSeqAIJGetCSRAndMemType(mat_nonlocal_sparsity, &device_nonlocal_i_sparsity, &device_nonlocal_j_sparsity, &device_nonlocal_vals_sparsity, &mtype);

   const PetscInt *device_local_i_output = nullptr, *device_local_j_output = nullptr, *device_nonlocal_i_output = nullptr, *device_nonlocal_j_output = nullptr;
   PetscScalar *device_local_vals_output = nullptr, *device_nonlocal_vals_output = nullptr;
   MatSeqAIJGetCSRAndMemType(mat_local_output, &device_local_i_output, &device_local_j_output, &device_local_vals_output, &mtype);     
   if (mpi) MatSeqAIJGetCSRAndMemType(mat_nonlocal_output, &device_nonlocal_i_output, &device_nonlocal_j_output, &device_nonlocal_vals_output, &mtype);

   // Scale the highest constrained power
   MatScale(*output_mat, coefficients[poly_sparsity_order]);

   // Then go backwards and add in each of the coefficients * A^order from the second highest order down
   for (int i = poly_sparsity_order-1; i > 0; i--)
   {
      MatAXPY(*output_mat, coefficients[i], matrix_powers[i-1], SUBSET_NONZERO_PATTERN);
   }

   // Add in the 0th order term
   MatShift(*output_mat, coefficients[0]);

   // Find maximum non-zeros per row for sizing scratch memory
   PetscInt max_nnz = 0;
   if (local_rows > 0) {
      // First get max row width from submat
      Kokkos::parallel_reduce("FindMaxNNZ", local_rows,
         KOKKOS_LAMBDA(const PetscInt i, PetscInt& thread_max) {
            PetscInt row_nnz = device_submat_i[i + 1] - device_submat_i[i];
            thread_max = (row_nnz > thread_max) ? row_nnz : thread_max;
         },
         Kokkos::Max<PetscInt>(max_nnz)
      );
      
      // Also consider sparsity matrix row width if needed
      if(poly_sparsity_order != 1) {
         PetscInt sparsity_max_nnz = 0;
         Kokkos::parallel_reduce("FindMaxNNZSparsity", local_rows,
            KOKKOS_LAMBDA(const PetscInt i, PetscInt& thread_max) {
               PetscInt row_nnz = device_local_i_sparsity[i + 1] - device_local_i_sparsity[i];
               if (mpi) row_nnz += device_nonlocal_i_sparsity[i + 1] - device_nonlocal_i_sparsity[i];
               thread_max = (row_nnz > thread_max) ? row_nnz : thread_max;
            },
            Kokkos::Max<PetscInt>(sparsity_max_nnz)
         );
         
         // Take the larger of the two maxes
         if (sparsity_max_nnz > max_nnz) max_nnz = sparsity_max_nnz;
      }
   }   

   auto exec = PetscGetKokkosExecutionSpace();

   // ~~~~~~~~~~~~~
   // Now we have to be careful 
   // mat_sparsity_match may not have diagonal entries in some rows
   // but we know our gmres polynomial inverse must 
   // and after calling mat_duplicate_copy_plus_diag_kokkos we know our output_mat
   // has diagonals
   // But when we write out to output_mat below, we assume it has the same sparsity as 
   // mat_sparsity_match
   // So we have to track which rows don't have diagonals in mat_sparsity_match
   // so we can increment the writing out by one in output_mat
   // The reason we don't have to do this in the cpu version is because we are calling
   // matsetvalues to write out to output_mat, rather than writing out to the csr directly
   // ~~~~~~~~~~~~~

   auto found_diag_row_d = PetscIntKokkosView("found_diag_row_d", local_rows);    
   Kokkos::deep_copy(found_diag_row_d, 0); 

   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

      // Row
      PetscInt i = t.league_rank();
      // ncols_local
      PetscInt ncols_local = device_local_i_sparsity[i + 1] - device_local_i_sparsity[i];
      
      // Loop over all the columns in this row of sparsity mat
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

         // Is this column the diagonal
         bool is_diagonal = (device_local_j_sparsity[device_local_i_sparsity[i] + j] + global_col_start == i + global_row_start);         
         // This will only happen on a max of one thread per row
         if (is_diagonal) found_diag_row_d(i) = 1;

      });
   });
   
   // ~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~

   // Create a team policy with scratch memory allocation
   // We want scratch space for each row
   // We will have ncols of integers which tell us how many matching indices we have
   // We then want ncols space for each column (both indices and values)
   // We then want a vals_temp and vals_prev to store the accumulated matrix powers
   // We need to know if each column is local
   // the last bit of memory is to account for 8-byte alignment for each view
   size_t scratch_size_per_team = max_nnz * max_nnz * (sizeof(PetscInt) + sizeof(PetscScalar)) + \
               max_nnz * sizeof(PetscInt) + \
               max_nnz * 2 * sizeof(PetscScalar) + \
               8 * 5 * sizeof(PetscScalar);

   Kokkos::TeamPolicy<> policy(exec, local_rows, Kokkos::AUTO());
   // We're gonna use the level 1 scratch as our column data is probably bigger than the level 0
   policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size_per_team));

   // Execute with scratch memory
   Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const KokkosTeamMemberType& t) {

      // Row
      PetscInt i = t.league_rank();

      // ncols is the total number of columns in this row of the sparsity mat
      PetscInt ncols = device_local_i_sparsity[i + 1] - device_local_i_sparsity[i];
      if (mpi) ncols += device_nonlocal_i_sparsity[i + 1] - device_nonlocal_i_sparsity[i];

      // Allocate views directly on scratch memory
      // Have to use views here given alignment issues
      ScratchIntView scratch_match_counter(t.team_scratch(1), ncols);
      // Be careful as 2D views are column major by default
      Scratch2DIntView scratch_indices(t.team_scratch(1), max_nnz, ncols);
      Scratch2DScalarView scratch_vals(t.team_scratch(1), max_nnz, ncols);
      ScratchScalarView vals_prev(t.team_scratch(1), ncols);
      ScratchScalarView vals_temp(t.team_scratch(1), ncols);   

      // This is first nonlocal column of sparsity mat is in this row
      PetscInt start_nonlocal_idx = 0;
      if (mpi)
      {
         start_nonlocal_idx = device_local_i_sparsity[i + 1] - device_local_i_sparsity[i];  
      }         
      else
      {
         start_nonlocal_idx = ncols;
      }
      
      // Loop over all the columns in this row of sparsity mat
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols), [&](const PetscInt j) {

         // Set match to zero
         scratch_match_counter[j] = 0;

         // Get the row index into submat for this column in sparsity mat
         // and copy in this row of sparsity mat to vals_prev
         PetscInt row_idx;
         if (j < start_nonlocal_idx)
         {
            row_idx = device_local_j_sparsity[device_local_i_sparsity[i] + j];
            vals_prev[j] = device_local_vals_sparsity[device_local_i_sparsity[i] + j];
         }
         // Nonlocal part
         else
         {
            // We are matching the "local" column indices of the submat here
            row_idx = device_nonlocal_j_sparsity[device_nonlocal_i_sparsity[i] + (j - start_nonlocal_idx)] + cols_ad;
            vals_prev[j] = device_nonlocal_vals_sparsity[device_nonlocal_i_sparsity[i] + (j - start_nonlocal_idx)];
         }            
            
         // Get column indices for this row
         PetscInt target_start = device_submat_i[row_idx];
         PetscInt target_end = device_submat_i[row_idx + 1];
         PetscInt target_ncols = target_end - target_start;

         // We'll perform a search to find matching indices
         // We're matching indices in sparsity mat to those in submat
         // This is just an intersection between row i and the row of column j
         // This assumes column indices are already sorted 
         PetscInt idx_orig = 0;  // Index into original row i columns
         PetscInt idx_target = 0;  // Index into target row columns

         while (idx_orig < ncols && idx_target < target_ncols) {
            PetscInt col_target = device_submat_j[target_start + idx_target];

            PetscInt col_orig;
            // If we're in the local part of the matrix
            if (idx_orig < start_nonlocal_idx)
            {
               col_orig = device_local_j_sparsity[device_local_i_sparsity[i] + idx_orig];
            }
            // Nonlocal part
            else
            {
               // We are matching the "local" column indices of the submat here
               col_orig = device_nonlocal_j_sparsity[device_nonlocal_i_sparsity[i] + (idx_orig - start_nonlocal_idx)] + cols_ad;
            }
            
            if (col_orig < col_target) {
               // Original column is smaller, move to next original column
               idx_orig++;
            } else if (col_orig > col_target) {
               // Target column is smaller, move to next target column
               idx_target++;
            } else {
               // Match found - this ensures we insert in sorted order
               PetscInt match_idx = scratch_match_counter[j]++;
               // Local index into row i
               scratch_indices(match_idx, j) = idx_orig;
               // Values of matches 
               scratch_vals(match_idx, j) = device_submat_vals[target_start + idx_target];
               // Move forward in both arrays
               idx_orig++;
               idx_target++;
            }
         }
      });
      
      // Team barrier to ensure all threads have finished filling the scratch space
      t.team_barrier();

      // ~~~~~~~~~~~~~~~~~~~~~~
      // Now we have the matching indices and values for the shared sparsity 
      // matmatmult we will perform now between row i and all of the rows given by the 
      // columns in row i      
      // ~~~~~~~~~~~~~~~~~~~~~~   

      // Loop over any matrix powers
      // vals_power_temp stores the value of A^(term-1) for this row, and we update this as we go through 
      // the term loop
      for (int term = poly_sparsity_order + 1; term <= poly_order; term++)
      {
         // Skip any zero coefficients
         if (coefficients_d(term) != 0.0)
         {
            // Set vals_temp to zero
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols), [&](const PetscInt j) {
               vals_temp[j] = 0;
            });      
            
            // Team barrier to ensure initialization is complete before use
            t.team_barrier();                   
               
            // Now compute the sums in vals_temp
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols), [&](const PetscInt j) {

               // Loop over the matches
               for (int k = 0; k < scratch_match_counter[j]; k++)
               {
                  // Has to be atomic! Potentially lots of contention so maybe not 
                  // the most performant way to do this
                  Kokkos::atomic_add(&vals_temp[scratch_indices(k, j)], vals_prev[j] * scratch_vals(k, j));
               }
            });      
            
            // Team barrier to ensure initialization is complete before use
            t.team_barrier();     

            // ~~~~~~~~~~~
            // Now can add the value of coeff * A^(term-1) to our matrix
            // ~~~~~~~~~~~               
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols), [&](const PetscInt j) {

               PetscInt diag_increm = 0;

               // Do the mult with coeff
               // If we're in the local part of the matrix
               if (j < start_nonlocal_idx)
               {
                  // We need to increment the index we access by one
                  // if we don't have a diagonal in the sparsity matrix
                  // as we have one in the output_mat                  
                  if (found_diag_row_d(i) == 0 && device_local_j_output[device_local_i_output[i] + j] >= i) 
                  {
                     diag_increm = 1;
                  }
                  device_local_vals_output[device_local_i_output[i] + j + diag_increm] += coefficients_d(term) * vals_temp[j];
               }
               // Nonlocal part
               else
               {
                  device_nonlocal_vals_output[device_nonlocal_i_output[i] + (j - start_nonlocal_idx)] += coefficients_d(term) * vals_temp[j];                  
               }
               // This should now have the value of A^(term-1) in it
               vals_prev[j] = vals_temp[j];
            });

            // Team barrier to ensure initialization is complete before use
            t.team_barrier();                
         }
      }
   });
    
   Mat_SeqAIJKokkos *aijkok_local_output = static_cast<Mat_SeqAIJKokkos *>(mat_local_output->spptr);
   Mat_SeqAIJKokkos *aijkok_nonlocal_output;
   if (mpi) aijkok_nonlocal_output = static_cast<Mat_SeqAIJKokkos *>(mat_nonlocal_output->spptr);   

   // Have to specify we've modifed data on the device
   // Want to call MatSeqAIJKokkosModifyDevice but its PETSC_INTERN
   aijkok_local_output->a_dual.clear_sync_state();
   aijkok_local_output->a_dual.modify_device();
   aijkok_local_output->transpose_updated = PETSC_FALSE;
   aijkok_local_output->hermitian_updated = PETSC_FALSE;
   // Invalidate diagonals
   Mat_SeqAIJ *a = (Mat_SeqAIJ *)mat_local_output->data;
   a->idiagvalid  = PETSC_FALSE;
   a->ibdiagvalid = PETSC_FALSE;      
   a->inode.ibdiagvalid = PETSC_FALSE;        
   if (mpi)
   {
      aijkok_nonlocal_output->a_dual.clear_sync_state();
      aijkok_nonlocal_output->a_dual.modify_device();
      aijkok_nonlocal_output->transpose_updated = PETSC_FALSE;
      aijkok_nonlocal_output->hermitian_updated = PETSC_FALSE;
      a = (Mat_SeqAIJ *)mat_nonlocal_output->data;
      a->idiagvalid  = PETSC_FALSE;
      a->ibdiagvalid = PETSC_FALSE;   
      a->inode.ibdiagvalid = PETSC_FALSE; 
   }      
   PetscObjectStateIncrease((PetscObject)(*output_mat));   

   for (int i = 1; i < poly_sparsity_order; i++)
   {
      MatDestroy(&(matrix_powers[i]));
   }   
   delete[] matrix_powers;
   if (deallocate_submatrices) delete[] submatrices;
   PetscFree(col_indices_off_proc_array);

   return;
}