// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

//------------------------------------------------------------------------------------------------------------------------

// Compute matrix-matrix product with fixed order sparsity but with kokkos - keeping everything on the device
PETSC_INTERN void mat_mult_powers_share_sparsity_kokkos(Mat *input_mat, const int poly_order, const int poly_sparsity_order, PetscReal *coefficients, \
               const int reuse_int_reuse_mat, Mat *reuse_mat, const int reuse_int_cmat, Mat *output_mat)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols;
   PetscInt global_row_start_temp, global_row_end_plus_one_temp;
   PetscInt global_col_start_temp, global_col_end_plus_one_temp;
   PetscInt rows_ao, cols_ao, rows_ad, cols_ad, size_cols;
   MatType mat_type;
   Mat *matrix_powers, *mat_sparsity_match;
   PetscInt one = 1;
   bool deallocate_submatrices = false;

   MatGetType(*input_mat, &mat_type);
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat_MPIAIJ *mat_mpi = nullptr;
   Mat mat_local_sparsity = NULL, mat_nonlocal_sparsity = NULL;
   Mat_MPIAIJ *mat_mpi_input = nullptr;
   Mat mat_local_input = NULL, mat_nonlocal_input = NULL;   

   // Get the comm
   PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX);
   MatGetLocalSize(*input_mat, &local_rows, &local_cols);
   // This returns the global index of the local portion of the matrix
   MatGetOwnershipRange(*input_mat, &global_row_start_temp, &global_row_end_plus_one_temp);
   MatGetOwnershipRangeColumn(*input_mat, &global_col_start_temp, &global_col_end_plus_one_temp);
   const PetscInt global_row_start = global_row_start_temp;
   //const PetscInt global_row_end_plus_one = global_row_end_plus_one_temp;
   const PetscInt global_col_start = global_col_start_temp;
   //const PetscInt global_col_end_plus_one = global_col_end_plus_one_temp;

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
   IS col_indices, row_indices;
   Mat *submatrices;

   // Pull out the nonlocal parts of the input mat we need
   if (mpi)
   {
      mat_mpi_input = (Mat_MPIAIJ *)(*input_mat)->data;
      mat_local_input = mat_mpi_input->A;
      mat_nonlocal_input = mat_mpi_input->B;

      mat_mpi = (Mat_MPIAIJ *)(*mat_sparsity_match)->data;
      mat_local_sparsity = mat_mpi->A;
      mat_nonlocal_sparsity = mat_mpi->B;
      MatGetSize(mat_nonlocal_sparsity, &rows_ao, &cols_ao); 
      MatGetSize(mat_local_sparsity, &rows_ad, &cols_ad);

      // We need to pull out all the columns in the sparsity mat
      // and the nonlocal rows that correspond to the nonlocal columns
      // from the input mat      
      PetscMalloc1(cols_ad + cols_ao, &col_indices_off_proc_array);
      size_cols = cols_ad + cols_ao;
      for (PetscInt i = 0; i < cols_ad; i++)
      {
         col_indices_off_proc_array[i] = global_row_start + i;
      }
      for (PetscInt i = 0; i < cols_ao; i++)
      {
         col_indices_off_proc_array[cols_ad + i] = mat_mpi->garray[i];
      }           
      
      // Create the sequential IS we want with the cols we want (written as global indices)
      ISCreateGeneral(PETSC_COMM_SELF, size_cols, \
                  col_indices_off_proc_array, PETSC_USE_POINTER, &col_indices);
      ISCreateGeneral(PETSC_COMM_SELF, cols_ao, \
                  mat_mpi->garray, PETSC_USE_POINTER, &row_indices);

      MatSetOption(*input_mat, MAT_SUBMAT_SINGLEIS, PETSC_TRUE); 
      // Now this will be doing comms to get the non-local rows we want and returns in a sequential matrix
      if (!reuse_int_reuse_mat)
      {
         MatCreateSubMatrices(*input_mat, one, &row_indices, &col_indices, MAT_INITIAL_MATRIX, &submatrices);
         *reuse_mat = submatrices[0];
      }
      else
      {
         submatrices = new Mat[1];
         deallocate_submatrices = true;
         submatrices[0] = *reuse_mat;
         MatCreateSubMatrices(*input_mat, one, &row_indices, &col_indices, MAT_REUSE_MATRIX, &submatrices);         
      }
      ISDestroy(&col_indices);
      ISDestroy(&row_indices);
   }
   // In serial
   else
   {
      submatrices = new Mat[1];
      deallocate_submatrices = true;      
      submatrices[0] = *input_mat;
      mat_local_input = *input_mat;
      mat_local_sparsity = *mat_sparsity_match;
      cols_ad = local_cols;
      PetscMalloc1(local_rows, &col_indices_off_proc_array);
      for (PetscInt i = 0; i < local_rows; i++)
      {
         col_indices_off_proc_array[i] = i;
      }
   }

   // Get the existing output mat
   Mat mat_local_output = NULL, mat_nonlocal_output = NULL;   
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

   const PetscInt *device_local_i_input = nullptr, *device_local_j_input = nullptr, *device_nonlocal_i_input = nullptr, *device_nonlocal_j_input = nullptr;
   PetscScalar *device_local_vals_input = nullptr, *device_nonlocal_vals_input = nullptr;  
   MatSeqAIJGetCSRAndMemType(mat_local_input, &device_local_i_input, &device_local_j_input, &device_local_vals_input, &mtype);
   if (mpi) MatSeqAIJGetCSRAndMemType(mat_nonlocal_input, &device_nonlocal_i_input, &device_nonlocal_j_input, &device_nonlocal_vals_input, &mtype);

   const PetscInt *device_local_i_sparsity = nullptr, *device_local_j_sparsity = nullptr, *device_nonlocal_i_sparsity = nullptr, *device_nonlocal_j_sparsity = nullptr;
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

   // ~~~~~~~~~~~~~~
   // Find maximum non-zeros per row for sizing scratch memory
   // ~~~~~~~~~~~~~~
   PetscInt sparsity_max_nnz = 0, sparsity_max_nnz_local = 0, sparsity_max_nnz_nonlocal = 0;
   if (local_rows > 0) {        
      // Also consider sparsity matrix row width if needed
      Kokkos::parallel_reduce("FindMaxNNZSparsity", local_rows,
         KOKKOS_LAMBDA(const PetscInt i, PetscInt& thread_max) {
            PetscInt row_nnz = device_local_i_sparsity[i + 1] - device_local_i_sparsity[i];
            thread_max = (row_nnz > thread_max) ? row_nnz : thread_max;
         },
         Kokkos::Max<PetscInt>(sparsity_max_nnz_local)
      );
      if (mpi)
      {
         Kokkos::parallel_reduce("FindMaxNNZSparsityNonLocal", local_rows,
            KOKKOS_LAMBDA(const PetscInt i, PetscInt& thread_max) {
               PetscInt row_nnz = device_nonlocal_i_sparsity[i + 1] - device_nonlocal_i_sparsity[i];
               thread_max = (row_nnz > thread_max) ? row_nnz : thread_max;
            },
            Kokkos::Max<PetscInt>(sparsity_max_nnz_nonlocal)
         );   
      }  
      sparsity_max_nnz = sparsity_max_nnz_local + sparsity_max_nnz_nonlocal; 
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
      const PetscInt i = t.league_rank();
      // ncols_local
      const PetscInt ncols_local = device_local_i_sparsity[i + 1] - device_local_i_sparsity[i];
      const PetscInt row_index_global = i + global_row_start;
      
      // Loop over all the columns in this row of sparsity mat
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

         // Is this column the diagonal
         const bool is_diagonal = (device_local_j_sparsity[device_local_i_sparsity[i] + j] + global_col_start == row_index_global);         
         // This will only happen on a max of one thread per row
         if (is_diagonal) found_diag_row_d(i) = 1;

      });
   });
   
   // ~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~

   // Create a team policy with scratch memory allocation
   // We want scratch space for each row
   // We then want a vals_temp and vals_prev to store the accumulated matrix powers
   // the last bit of memory is to account for 8-byte alignment for each view
   size_t scratch_size_per_team = sparsity_max_nnz * 2 * sizeof(PetscScalar) + \
               8 * 2 * sizeof(PetscScalar);

   Kokkos::TeamPolicy<> policy(exec, local_rows, Kokkos::AUTO());
   // We're gonna use the level 1 scratch as our column data is probably bigger than the level 0
   policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size_per_team));

   // Execute with scratch memory
   Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const KokkosTeamMemberType& t) {

      // Row
      const PetscInt i = t.league_rank();

      // ncols_row_i is the total number of columns in this row of the sparsity mat
      PetscInt ncols_row_i = device_local_i_sparsity[i + 1] - device_local_i_sparsity[i];
      if (mpi) ncols_row_i += device_nonlocal_i_sparsity[i + 1] - device_nonlocal_i_sparsity[i];

      // Allocate views directly on scratch memory
      // Have to use views here given alignment issues
      ScratchScalarView vals_prev(t.team_scratch(1), ncols_row_i);
      ScratchScalarView vals_temp(t.team_scratch(1), ncols_row_i);   

      // How many local columns do we have in row i
      const PetscInt local_cols_row_i = device_local_i_sparsity[i + 1] - device_local_i_sparsity[i];
      
      // Loop over all the columns in this row of sparsity mat
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_row_i), [&](const PetscInt j) {

         // Fill vals_prev
         if (j < local_cols_row_i)
         {
            vals_prev[j] = device_local_vals_sparsity[device_local_i_sparsity[i] + j];
         }
         // Nonlocal part
         else
         {
            vals_prev[j] = device_nonlocal_vals_sparsity[device_nonlocal_i_sparsity[i] + (j - local_cols_row_i)];
         }
      });
      
      // Team barrier to ensure all threads have finished filling the scratch space
      t.team_barrier();

      // ~~~~~~~~~~~~~~~~~~~~~~
      // Now in the loop below we recompute which indices match every time for each power 
      // They never change and we used to compute them once and store them in scratch space
      // but this then meant we used lots of memory per team and hence fewer
      // teams/threads in teams were used and hence we had less parallelism
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
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_row_i), [&](const PetscInt j) {
               vals_temp[j] = 0;
            });      
            
            // Team barrier to ensure initialization is complete before use
            t.team_barrier();                   
               
            // Now compute the sums in vals_temp
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
                  // We always convert it to the "local" indexing as if it were in the columns of the submat, ie 
                  // the column indexing of [local cols; local cols + 0:cols_ao-1]
                  PetscInt col_target;
                  if (row_of_col_j_local)
                  {
                     if (idx_col_of_row_j < local_cols_row_of_col_j)
                     {
                        col_target = device_local_j_input[device_local_i_input[row_of_col_j] + idx_col_of_row_j];
                     }
                     else
                     {
                        // Convert to "local" column index of submat by adding cols_ad
                        col_target = device_nonlocal_j_input[device_nonlocal_i_input[row_of_col_j] + idx_col_of_row_j - local_cols_row_of_col_j] + cols_ad;
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
                           val_target = device_local_vals_input[device_local_i_input[row_of_col_j] + idx_col_of_row_j];
                        }
                        else
                        {
                           val_target = device_nonlocal_vals_input[device_nonlocal_i_input[row_of_col_j] + idx_col_of_row_j - local_cols_row_of_col_j];
                        }
                     }
                     else
                     {
                        val_target = device_submat_vals[device_submat_i[row_of_col_j] + idx_col_of_row_j];
                     }                     

                     // Has to be atomic! Potentially lots of contention so maybe not 
                     // the most performant way to do this
                     Kokkos::atomic_add(&vals_temp[idx_col_of_row_i], vals_prev[j] * val_target);

                     // Move forward in both arrays
                     idx_col_of_row_i++;
                     idx_col_of_row_j++;
                  }
               }
            });      
            
            // Team barrier to ensure initialization is complete before use
            t.team_barrier();     

            // ~~~~~~~~~~~
            // Now can add the value of coeff * A^(term-1) to our matrix
            // ~~~~~~~~~~~               
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_row_i), [&](const PetscInt j) {

               PetscInt diag_increm = 0;

               // Do the mult with coeff
               // If we're in the local part of the matrix
               if (j < local_cols_row_i)
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
                  device_nonlocal_vals_output[device_nonlocal_i_output[i] + (j - local_cols_row_i)] += coefficients_d(term) * vals_temp[j];                  
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
   Mat_SeqAIJKokkos *aijkok_nonlocal_output = NULL;
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
   (void)PetscFree(col_indices_off_proc_array);

   return;
}