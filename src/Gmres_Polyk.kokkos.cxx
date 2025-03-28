// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

//------------------------------------------------------------------------------------------------------------------------

// Build a 0th order gmres polynomial but with kokkos - keeping everything on the device
PETSC_INTERN void build_gmres_polynomial_inverse_0th_order_kokkos(Mat *input_mat, int poly_order, PetscReal *coefficients, \
                     int reuse_int, Mat *output_mat)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   MatType mat_type;
   PetscInt nnzs_match_local, nnzs_match_nonlocal;
   Mat output_mat_local, output_mat_nonlocal;

   MatGetType(*input_mat, &mat_type);
   // Are we in parallel?
   bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   // Get the comm
   PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX);
   MatGetLocalSize(*input_mat, &local_rows, &local_cols);
   MatGetSize(*input_mat, &global_rows, &global_cols);       

   // ~~~~~~~~~~~~
   // Get the number of nnzs
   // ~~~~~~~~~~~~
   nnzs_match_local = local_rows;
   nnzs_match_nonlocal = 0;

   // Get device views
   Kokkos::View<PetscScalar *> a_local_d;
   Kokkos::View<PetscInt *> i_local_d;    
   Kokkos::View<PetscInt *> j_local_d;

   // Nonlocal stuff 
   Kokkos::View<PetscScalar *> a_nonlocal_d;
   Kokkos::View<PetscInt *> i_nonlocal_d;          
   Kokkos::View<PetscInt *> j_nonlocal_d;  

   // ~~~~~~~~~~~~~~~~~  
   // We need to assemble our i,j, vals so we can build our matrix
   // ~~~~~~~~~~~~~~~~~   
   if (!reuse_int)
   {  
      // Create device & host memory
      a_local_d = Kokkos::View<PetscScalar *>("a_local_d", nnzs_match_local);
      i_local_d = Kokkos::View<PetscInt *>("i_local_d", local_rows+1);
      j_local_d = Kokkos::View<PetscInt *>("j_local_d", nnzs_match_local);           

      // we also have to go and build the a, i, j for the non-local off-diagonal block
      if (mpi) 
      {
         // Create non-local host and device memory
         a_nonlocal_d = Kokkos::View<PetscScalar *>("a_nonlocal_d", nnzs_match_nonlocal);
         i_nonlocal_d = Kokkos::View<PetscInt *>("i_nonlocal_d", local_rows+1);
         j_nonlocal_d = Kokkos::View<PetscInt *>("j_nonlocal_d", nnzs_match_nonlocal);  

         // All zero, no non-local entries
         Kokkos::deep_copy(i_nonlocal_d, 0);                
      }               

      // ~~~~~~~~~~~~~~~
      // Create i indices
      // ~~~~~~~~~~~~~~~
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows+1), KOKKOS_LAMBDA(int i) {

            i_local_d(i) = i;
      });  
      // ~~~~~~~~~~~~~~~
      // Create j indices
      // ~~~~~~~~~~~~~~~
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(int i) {

            j_local_d(i) = i;
      });    
      // 0th order polynomial is just the first coefficient on the diagonal
      // Copy it straight from the host
      Kokkos::deep_copy(a_local_d, coefficients[0]);  
      // Log copy with petsc
      size_t bytes = sizeof(PetscReal);
      PetscLogCpuToGpu(bytes);  
      
      // Let's make sure everything on the device is finished
      auto exec = PetscGetKokkosExecutionSpace();
      exec.fence();
      
      // We can create our local diagonal block matrix directly on the device
      MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, local_cols, i_local_d, j_local_d, a_local_d, &output_mat_local);        

      // we also have to go and build the a, i, j for the non-local off-diagonal block
      if (mpi) 
      {
         // Zero off-diagonal entries
         PetscInt *garray_host = NULL;
         PetscInt col_ao_output = 0;
         // Silly but depending on the compiler this may return a non-null pointer
         PetscMalloc1(col_ao_output, &garray_host);      
         
         // We can create our nonlocal diagonal block matrix directly on the device
         MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, col_ao_output, i_nonlocal_d, j_nonlocal_d, a_nonlocal_d, &output_mat_nonlocal);         

         // We can now create our MPI matrix
         MatCreateMPIAIJWithSeqAIJ(MPI_COMM_MATRIX, global_rows, global_cols, output_mat_local, output_mat_nonlocal, garray_host, output_mat);
      }     
      // If in serial 
      else
      {
         *output_mat = output_mat_local;
      }
   }
   // With re-use
   else
   {
      Mat_MPIAIJ *mat_mpi_output = nullptr;
      Mat mat_local_output; 

      // Get the existing output mats
      if (mpi)
      {
         mat_mpi_output = (Mat_MPIAIJ *)(*output_mat)->data;
         mat_local_output = mat_mpi_output->A;
      }
      else
      {
         mat_local_output = *output_mat;
      }     
      Mat_SeqAIJKokkos *aijkok_local_output = static_cast<Mat_SeqAIJKokkos *>(mat_local_output->spptr);
      // Annoying we can't just call MatSeqAIJGetKokkosView
      a_local_d = aijkok_local_output->a_dual.view_device();
      // Copy in the host value directly
      Kokkos::deep_copy(a_local_d, coefficients[0]);  
      // Log copy with petsc
      size_t bytes = sizeof(PetscReal);
      PetscLogCpuToGpu(bytes);         

      // Have to specify we've modifed local data on the device
      // Want to call MatSeqAIJKokkosModifyDevice but its PETSC_INTERN
      aijkok_local_output->a_dual.clear_sync_state();
      aijkok_local_output->a_dual.modify_device();
      // Transpose is the same
      //aijkok_local_output->transpose_updated = PETSC_FALSE;
      //aijkok_local_output->hermitian_updated = PETSC_FALSE;
      // Invalidate diagonals
      Mat_SeqAIJ *a = (Mat_SeqAIJ *)mat_local_output->data;
      a->idiagvalid  = PETSC_FALSE;
      a->ibdiagvalid = PETSC_FALSE;      
      a->inode.ibdiagvalid = PETSC_FALSE;           
      PetscObjectStateIncrease((PetscObject)(*output_mat));
   }

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Compute matrix-matrix product with fixed order sparsity but with kokkos - keeping everything on the device
PETSC_INTERN void mat_mult_powers_share_sparsity_kokkos(Mat *input_mat, int poly_order, int poly_sparsity_order, PetscReal *coefficients, \
               int reuse_int_reuse_mat, Mat *reuse_mat, int reuse_int_cmat, Mat *output_mat)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt global_col_start, global_col_end_plus_one;
   PetscInt rows_ao, cols_ao, rows_ad, cols_ad, size_cols;
   MatType mat_type;
   Mat *matrix_powers, *mat_sparsity_match;
   PetscInt one = 1;
   bool deallocate_submatrices = false;
   int errorcode;

   MatGetType(*input_mat, &mat_type);
   // Are we in parallel?
   bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat_MPIAIJ *mat_mpi = nullptr;
   Mat mat_local, mat_nonlocal;

   // Get the comm
   PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX);
   MatGetLocalSize(*input_mat, &local_rows, &local_cols);
   MatGetSize(*input_mat, &global_rows, &global_cols);  
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
      mat_local = mat_mpi->A;
      mat_nonlocal = mat_mpi->B;
      MatGetSize(mat_nonlocal, &rows_ao, &cols_ao); 
      MatGetSize(mat_local, &rows_ad, &cols_ad);

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

      if (poly_sparsity_order != 1)
      {
         ISSort(col_indices);
      }

      MatSetOption(*input_mat, MAT_SUBMAT_SINGLEIS, PETSC_TRUE); 
      // Now this will be doing comms to get the non-local rows we want and returns in a sequential matrix
      if (!reuse_int_reuse_mat)
      {
         MatCreateSubMatrices(*input_mat, one, &col_indices, &col_indices, MAT_INITIAL_MATRIX, &submatrices);
         reuse_mat = &submatrices[0];
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
      PetscMalloc1(local_rows, &col_indices_off_proc_array);
      for (int i = 0; i < local_rows; i++)
      {
         col_indices_off_proc_array[i] = i;
      }
   }

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr;  
   MatSeqAIJGetCSRAndMemType(submatrices[0], &device_local_i, &device_local_j, &device_local_vals, &mtype);  

   const PetscInt *device_local_i_sparsity = nullptr, *device_local_j_sparsity = nullptr;
   PetscScalar *device_local_vals_sparsity = nullptr;  
   MatSeqAIJGetCSRAndMemType(*mat_sparsity_match, &device_local_i_sparsity, &device_local_j_sparsity, &device_local_vals_sparsity, &mtype);     

   const PetscInt *device_local_i_output = nullptr, *device_local_j_output = nullptr;
   PetscScalar *device_local_vals_output = nullptr;  
   MatSeqAIJGetCSRAndMemType(*output_mat, &device_local_i_output, &device_local_j_output, &device_local_vals_output, &mtype);     

   // Scale the highest constrained power
   MatScale(*output_mat, coefficients[poly_sparsity_order]);

   // Then go backwards and add in each of the coefficients * A^order from the second highest order down
   for (int i = poly_sparsity_order-1; i > 0; i--)
   {
      MatAXPY(*output_mat, coefficients[i], matrix_powers[i-1], SUBSET_NONZERO_PATTERN);
   }

   // Add in the 0th order term
   MatShift(*output_mat, coefficients[0]);

   // Get global column indices
   if(poly_sparsity_order == 1)
   {
      // Find maximum non-zeros per row for sizing scratch memory using parallel reduction
      // @@@@@@@@@@@ this should also be over mat sparsity match 
      PetscInt max_nnz = 0;
      Kokkos::parallel_reduce("FindMaxNNZ", local_rows,
         KOKKOS_LAMBDA(const PetscInt i, PetscInt& thread_max) {
            PetscInt row_nnz = device_local_i[i + 1] - device_local_i[i];
            thread_max = (row_nnz > thread_max) ? row_nnz : thread_max;
         },
         Kokkos::Max<PetscInt>(max_nnz)
      );

      // Create a team policy with scratch memory allocation
      // We want scratch space for each row
      // We will have ncols of integers which tell us how many matching indices we have
      // We then want ncols space for each column (both indices and values)
      // We then want a vals_temp and vals_prev to store the accumulated matrix powers
      size_t scratch_size_per_team = max_nnz * max_nnz * (sizeof(PetscInt) + sizeof(PetscScalar)) + \
                  max_nnz * sizeof(PetscInt) + \
                  max_nnz * 2 * sizeof(PetscScalar);
      Kokkos::TeamPolicy<> policy(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO());
      // We're gonna use the level 1 scratch as our column data is probably bigger than the level 0
      policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size_per_team));

      // Execute with scratch memory
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const KokkosTeamMemberType& t) {

         // Row
         PetscInt i = t.league_rank();

         // ncols
         PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];  

         // Get scratch space for the match counter
         PetscInt* scratch_match_counter = (PetscInt*)t.team_scratch(1).get_shmem(ncols_local * (sizeof(PetscInt)));         
         // Get scratch space for the indices
         PetscInt* scratch_indices = (PetscInt*)t.team_scratch(1).get_shmem(ncols_local * max_nnz * (sizeof(PetscInt)));
         // Get scratch space for the values
         PetscScalar* scratch_vals = (PetscScalar*)t.team_scratch(1).get_shmem(ncols_local * max_nnz * (sizeof(PetscScalar)));       
         // Get scratch space for the previous temporary matrix powers sum
         PetscScalar* vals_prev = (PetscScalar*)t.team_scratch(1).get_shmem(ncols_local * (sizeof(PetscScalar)));     
         // Create view on scratch memory for atomic operations on vals_temp
         Kokkos::View<PetscScalar*, ScratchMemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> 
            vals_temp(t.team_scratch(1), ncols_local);           
         
         // Loop over all the columns in this row
         Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

            // Set match to zero
            scratch_match_counter[j] = 0;
            // Start with the values of mat_sparsity_match in it @@@ careful if modifying for not poly_sparsity_order == 1
            vals_prev[j] = device_local_vals[device_local_i[i] + j];

            // Get the row index that this column points to
            PetscInt row_idx = device_local_j[device_local_i[i] + j];
               
            // Get column indices for this row
            PetscInt target_start = device_local_i[row_idx];
            PetscInt target_end = device_local_i[row_idx + 1];
            PetscInt target_ncols = target_end - target_start;

            // We'll perform a binary search to find matching indices
            // This is just an intersection between row i and the row of column j
            // This assumes column indices are already sorted in CSR format
            PetscInt idx_orig = 0;  // Index into original row i columns
            PetscInt idx_target = 0;  // Index into target row columns

            while (idx_orig < ncols_local && idx_target < target_ncols) {
               PetscInt col_orig = device_local_j[device_local_i[i] + idx_orig];
               PetscInt col_target = device_local_j[target_start + idx_target];
               
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
                  scratch_indices[j * max_nnz + match_idx] = idx_orig;
                  // Values of matches 
                  scratch_vals[j * max_nnz + match_idx] = device_local_vals[target_start + idx_target];
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
               Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {
                  vals_temp(j) = 0;
               });      
               
               // Team barrier to ensure initialization is complete before use
               t.team_barrier();                   
                  
               // Now compute the sums in vals_temp
               Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

                  // Loop over the matches
                  for (int k = 0; k < scratch_match_counter[j]; k++)
                  {
                     // Has to be atomic! Potentially lots of contention so maybe not 
                     // the most performant way to do this
                     Kokkos::atomic_add(&vals_temp(scratch_indices[j * max_nnz + k]), vals_prev[j] * scratch_vals[j * max_nnz + k]);
                  }

               });      
               
               // Team barrier to ensure initialization is complete before use
               t.team_barrier();     

               // ~~~~~~~~~~~
               // Now can add the value of coeff * A^(term-1) to our matrix
               // ~~~~~~~~~~~               
               Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {
                  // Do the mult with coeff
                  device_local_vals_output[device_local_i_output[i] + j] += coefficients_d(term) * vals_temp(j);
                  // This should now have the value of A^(term-1) in it
                  vals_prev[j] = vals_temp(j);
               });

               // Team barrier to ensure initialization is complete before use
               t.team_barrier();                
            }
         }
      });
   }
   else
   {
      errorcode = 1;
      MPI_Abort(MPI_COMM_MATRIX, errorcode);
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

//------------------------------------------------------------------------------------------------------------------------

// Build a gmres polynomial with 0th order sparsity but with kokkos - keeping everything on the device
PETSC_INTERN void build_gmres_polynomial_inverse_0th_order_sparsity_kokkos(Mat *input_mat, int poly_order, PetscReal *coefficients, \
                     int reuse_int, Mat *output_mat)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   MatType mat_type;
   PetscInt nnzs_match_local, nnzs_match_nonlocal;
   Mat output_mat_local, output_mat_nonlocal;

   MatGetType(*input_mat, &mat_type);
   // Are we in parallel?
   bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   // Get the comm
   PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX);
   MatGetLocalSize(*input_mat, &local_rows, &local_cols);
   MatGetSize(*input_mat, &global_rows, &global_cols);  

   // We also copy the coefficients over to the device as we need it
   PetscInt coeff_size = poly_order + 1;
   auto coefficients_h = PetscScalarKokkosViewHost(coefficients, coeff_size);
   auto coefficients_d = PetscScalarKokkosView("coefficients_d", coeff_size);
   Kokkos::deep_copy(coefficients_d, coefficients_h);       
   // Log copy with petsc
   size_t bytes = coefficients_h.extent(0) * sizeof(PetscReal);
   PetscLogCpuToGpu(bytes);      

   // ~~~~~~~~~~~~
   // Get the number of nnzs
   // ~~~~~~~~~~~~
   nnzs_match_local = local_rows;
   nnzs_match_nonlocal = 0;

   // Get device views
   Kokkos::View<PetscScalar *> a_local_d;
   Kokkos::View<PetscInt *> i_local_d;    
   Kokkos::View<PetscInt *> j_local_d;

   // Nonlocal stuff 
   Kokkos::View<PetscScalar *> a_nonlocal_d;
   Kokkos::View<PetscInt *> i_nonlocal_d;          
   Kokkos::View<PetscInt *> j_nonlocal_d;  

   Mat_MPIAIJ *mat_mpi_output = nullptr;
   Mat mat_local_output; 
   Mat_SeqAIJKokkos *aijkok_local_output;

   // ~~~~~~~~~~~~~~~~~  
   // We need to assemble our i,j, vals so we can build our matrix
   // ~~~~~~~~~~~~~~~~~
   if (!reuse_int)
   {  
      // Create device & host memory
      a_local_d = Kokkos::View<PetscScalar *>("a_local_d", nnzs_match_local);
      i_local_d = Kokkos::View<PetscInt *>("i_local_d", local_rows+1);
      j_local_d = Kokkos::View<PetscInt *>("j_local_d", nnzs_match_local);      

      // we also have to go and build the a, i, j for the non-local off-diagonal block
      if (mpi) 
      {
         // Create non-local host and device memory
         a_nonlocal_d = Kokkos::View<PetscScalar *>("a_nonlocal_d", nnzs_match_nonlocal);
         i_nonlocal_d = Kokkos::View<PetscInt *>("i_nonlocal_d", local_rows+1);
         j_nonlocal_d = Kokkos::View<PetscInt *>("j_nonlocal_d", nnzs_match_nonlocal);  

         // All zero, no non-local entries
         Kokkos::deep_copy(i_nonlocal_d, 0);                
      }               

      // ~~~~~~~~~~~~~~~
      // Create i indices
      // ~~~~~~~~~~~~~~~
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows+1), KOKKOS_LAMBDA(int i) {

            i_local_d(i) = i;
      });  
      // ~~~~~~~~~~~~~~~
      // Create j indices
      // ~~~~~~~~~~~~~~~
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(int i) {

            j_local_d(i) = i;
      });    
   }
   // Reuse - get view to a 
   else
   {
      mat_mpi_output = nullptr;

      // Get the existing output mats
      if (mpi)
      {
         mat_mpi_output = (Mat_MPIAIJ *)(*output_mat)->data;
         mat_local_output = mat_mpi_output->A;
      }
      else
      {
         mat_local_output = *output_mat;
      }     
      aijkok_local_output = static_cast<Mat_SeqAIJKokkos *>(mat_local_output->spptr);
      // Annoying we can't just call MatSeqAIJGetKokkosView
      a_local_d = aijkok_local_output->a_dual.view_device();      
   }

   // ~~~~~~~~~~~~~~~~~~~~~~~
   // Compute the diagonal entries
   // ~~~~~~~~~~~~~~~~~~~~~~~

   // Get the matrix diagonal
   Vec diag_vec;
   MatCreateVecs(*input_mat, NULL, &diag_vec);
   MatGetDiagonal(*input_mat, diag_vec);
   ConstPetscScalarKokkosView diag_vec_d;
   VecGetKokkosView(diag_vec, &diag_vec_d);    

   // Loop over the rows
   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

      PetscInt i   = t.league_rank(); // row i
      PetscReal row_val = 0.0;

      // For this row, can do all the powers independently and then sum them
      // Add in the 1st order term to the last
      Kokkos::parallel_reduce(
         Kokkos::TeamThreadRange(t, 1, poly_order+1),
         [&](const PetscInt j, PetscReal& thread_data) {

            thread_data += coefficients_d[j] * pow(diag_vec_d(i), j);
         }, row_val
      );

      // Only want one thread in the team to write the result
      Kokkos::single(Kokkos::PerTeam(t), [&]() {     
         // Add the powers and the 0th order coefficient
         a_local_d(i) = row_val + coefficients_d[0];
      });      
   });    

   VecRestoreKokkosView(diag_vec, &diag_vec_d);    
   VecDestroy(&diag_vec);

   // ~~~~~~~~~~~~~~~~~~~~~~~

   // If we're not reusing we need to build our matrices
   if (!reuse_int)
   {
      // Let's make sure everything on the device is finished
      auto exec = PetscGetKokkosExecutionSpace();
      exec.fence();

      // We can create our local diagonal block matrix directly on the device
      MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, local_cols, i_local_d, j_local_d, a_local_d, &output_mat_local);  

      // we also have to go and build the a, i, j for the non-local off-diagonal block
      if (mpi) 
      {
         // Zero off-diagonal entries
         PetscInt *garray_host = NULL;
         PetscInt col_ao_output = 0;
         // Silly but depending on the compiler this may return a non-null pointer
         PetscMalloc1(col_ao_output, &garray_host);      
         
         // We can create our nonlocal diagonal block matrix directly on the device
         MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, col_ao_output, i_nonlocal_d, j_nonlocal_d, a_nonlocal_d, &output_mat_nonlocal);

         // We can now create our MPI matrix
         MatCreateMPIAIJWithSeqAIJ(MPI_COMM_MATRIX, global_rows, global_cols, output_mat_local, output_mat_nonlocal, garray_host, output_mat);
      }     
      // If in serial 
      else
      {
         *output_mat = output_mat_local;
      }
   }
   // With re-use
   else
   {
      // Have to specify we've modifed local data on the device
      // Want to call MatSeqAIJKokkosModifyDevice but its PETSC_INTERN
      aijkok_local_output->a_dual.clear_sync_state();
      aijkok_local_output->a_dual.modify_device();
      // Transpose is the same
      //aijkok_local_output->transpose_updated = PETSC_FALSE;
      //aijkok_local_output->hermitian_updated = PETSC_FALSE;
      // Invalidate diagonals
      Mat_SeqAIJ *a = (Mat_SeqAIJ *)mat_local_output->data;
      a->idiagvalid  = PETSC_FALSE;
      a->ibdiagvalid = PETSC_FALSE;      
      a->inode.ibdiagvalid = PETSC_FALSE;           
      PetscObjectStateIncrease((PetscObject)(*output_mat));
   }

   return;
}
