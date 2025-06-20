// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

//------------------------------------------------------------------------------------------------------------------------

// Generate the colmap and rewrite input global j indices to local given the calculated colmap
PETSC_INTERN void rewrite_j_global_to_local(PetscInt colmap_max_size, PetscInt &col_ao_output, PetscIntKokkosView j_nonlocal_d, PetscInt **garray_host)
{

   auto exec = PetscGetKokkosExecutionSpace();

   // Need to preallocate to the max size
   PetscIntKokkosView colmap_output_d("colmap_output_d", colmap_max_size);   
   col_ao_output = 0;

   // Take a copy of j and sort it and then build garray
   if (j_nonlocal_d.extent(0) > 0)
   {
      ptrdiff_t count_ptr_arith = -1;
      // Scoped so we don't keep the copy of j around very long
      {
         PetscIntKokkosView j_nonlocal_d_sorted("j_nonlocal_d_sorted", j_nonlocal_d.extent(0));
         Kokkos::deep_copy(j_nonlocal_d_sorted, j_nonlocal_d);
         Kokkos::sort(j_nonlocal_d_sorted);

         // Unique copy returns a copy of sorted j_nonlocal_d_sorted in order, but with all the duplicate entries removed
         auto unique_end_it = Kokkos::Experimental::unique_copy(exec, j_nonlocal_d_sorted, colmap_output_d);
         auto begin_it = Kokkos::Experimental::begin(colmap_output_d);
         count_ptr_arith = unique_end_it - begin_it;
      }
      col_ao_output = static_cast<PetscInt>(count_ptr_arith);

      // Create some host space for the output garray (that stays in scope) and copy it
      PetscMalloc1(col_ao_output, garray_host);
      PetscIntKokkosViewHost colmap_output_h = PetscIntKokkosViewHost(*garray_host, col_ao_output);
      PetscInt zero = 0;
      Kokkos::deep_copy(colmap_output_h, Kokkos::subview(colmap_output_d, Kokkos::make_pair(zero, col_ao_output)));
      // Log copy with petsc
      size_t bytes = col_ao_output * sizeof(PetscInt);
      PetscLogGpuToCpu(bytes);         
   }
   
   // ~~~~~~~~~~
   // Now we can go and overwrite the global indices in j with the local equivalents
   // ~~~~~~~~~~
   // Do we have any nonlocal columns
   if (col_ao_output == 0)
   {
      // Silly but depending on the compiler this may return a non-null pointer
      col_ao_output = 0;
      PetscMalloc1(col_ao_output, garray_host);
   }
   else
   {
      // Binary search sorted colmap to find our local index
      // Originally used Kokkos::UnorderedMap here but it only handles up to uint32_t
      // entries
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, j_nonlocal_d.extent(0)), KOKKOS_LAMBDA(const PetscInt i) { 

            PetscInt low = 0;
            PetscInt count = col_ao_output; // Number of elements in colmap_output_d
            PetscInt step = -1;
            PetscInt mid_idx = -1;
            
            while (count > 0) {
               step = count / 2;
               mid_idx = low + step;
               if (colmap_output_d(mid_idx) < j_nonlocal_d(i)) {
                  low = mid_idx + 1;
                  count -= (step + 1);
               } else {
                  count = step;
               }
            }
            j_nonlocal_d(i) = low;
      });         
   }
}

//------------------------------------------------------------------------------------------------------------------------

// Drop according to a tolerance but with kokkos - keeping everything on the device
PETSC_INTERN void remove_small_from_sparse_kokkos(Mat *input_mat, const PetscReal tol, Mat *output_mat, \
                  const int relative_max_row_tolerance_int, const int lump_int, const int allow_drop_diagonal_int)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start_temp, global_row_end_plus_one_temp;
   PetscInt global_col_start_temp, global_col_end_plus_one_temp;
   PetscInt rows_ao, cols_ao;
   MatType mat_type;
   PetscInt nnzs_match_local, nnzs_match_nonlocal;
   Mat output_mat_local, output_mat_nonlocal;

   MatGetType(*input_mat, &mat_type);
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat_MPIAIJ *mat_mpi = nullptr;
   Mat mat_local = NULL, mat_nonlocal = NULL;

   PetscIntKokkosViewHost colmap_input_h;
   PetscIntKokkosView colmap_input_d;   
   if (mpi)
   {
      mat_mpi = (Mat_MPIAIJ *)(*input_mat)->data;
      mat_local = mat_mpi->A;
      mat_nonlocal = mat_mpi->B;
      MatGetSize(mat_nonlocal, &rows_ao, &cols_ao); 

      // We also copy the input mat colmap over to the device as we need it
      colmap_input_h = PetscIntKokkosViewHost(mat_mpi->garray, cols_ao);
      colmap_input_d = PetscIntKokkosView("colmap_input_d", cols_ao);
      Kokkos::deep_copy(colmap_input_d, colmap_input_h);  
      // Log copy with petsc
      size_t bytes = colmap_input_h.extent(0) * sizeof(PetscInt);
      PetscLogCpuToGpu(bytes);              
   }
   else
   {
      mat_local = *input_mat;
   }

   // Get the comm
   PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX);
   MatGetLocalSize(*input_mat, &local_rows, &local_cols);
   MatGetSize(*input_mat, &global_rows, &global_cols);
   // This returns the global index of the local portion of the matrix
   MatGetOwnershipRange(*input_mat, &global_row_start_temp, &global_row_end_plus_one_temp);
   MatGetOwnershipRangeColumn(*input_mat, &global_col_start_temp, &global_col_end_plus_one_temp);
   const PetscInt global_row_start = global_row_start_temp;
   //const PetscInt global_row_end_plus_one = global_row_end_plus_one_temp;
   const PetscInt global_col_start = global_col_start_temp;
   const PetscInt global_col_end_plus_one = global_col_end_plus_one_temp;

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr, *device_nonlocal_vals = nullptr;  
   MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype);  
   if (mpi) MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype);          

   // ~~~~~~~~~~~~
   // Get the number of nnzs
   // ~~~~~~~~~~~~
   nnzs_match_local = 0;
   nnzs_match_nonlocal = 0;

   // ~~~~~~~~~~~~~~~~~~~~~~~
   // Let's build our i, j, and a on the device
   // ~~~~~~~~~~~~~~~~~~~~~~~
   // We need the relative row tolerance, let's create some device memory to store it
   PetscScalarKokkosView rel_row_tol_d("rel_row_tol_d", local_rows);      
   // Log copy with petsc
   size_t bytes = sizeof(PetscReal);
   PetscLogCpuToGpu(bytes);   

   // We need to know how many entries are in each row after our dropping  
   PetscIntKokkosView nnz_match_local_row_d("nnz_match_local_row_d", local_rows);             
   PetscIntKokkosView nnz_match_nonlocal_row_d;
   if (mpi) nnz_match_nonlocal_row_d = PetscIntKokkosView("nnz_match_nonlocal_row_d", local_rows);                  
   // Device memory for whether there is an existing diagonal in each row
   boolKokkosView existing_diag_d("existing_diag_d", local_rows);     
   Kokkos::deep_copy(existing_diag_d, false);
   const bool not_include_diag = relative_max_row_tolerance_int == -1;
   
   // Compute the relative row tolerances if needed
   if (relative_max_row_tolerance_int) 
   {       
      // Reduction over all rows
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            const PetscInt i = t.league_rank();
            const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];
            PetscScalar max_val = -1.0;
            const PetscInt row_index_global = i + global_row_start;

            // Reduce over local columns
            Kokkos::parallel_reduce(
               Kokkos::TeamVectorRange(t, ncols_local),
               [&](const PetscInt j, PetscScalar& thread_max) {

                  // Is this column the diagonal
                  const bool is_diagonal = (device_local_j[device_local_i[i] + j] + global_col_start == row_index_global);

                  // If our current tolerance is bigger than the max value we've seen so far
                  PetscScalar val = Kokkos::abs(device_local_vals[device_local_i[i] + j]);
                  // If we're not comparing against the diagonal when computing relative residual
                  if (not_include_diag && is_diagonal) val = -1.0;
                  if (val > thread_max) thread_max = val;

               },
               Kokkos::Max<PetscScalar>(max_val)
            );

            if (mpi) {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
               PetscScalar max_val_nonlocal = -1.0;
               
               // Reduce over nonlocal columns
               Kokkos::parallel_reduce(
                  Kokkos::TeamVectorRange(t, ncols_nonlocal),
                  [&](const PetscInt j, PetscScalar& thread_max) {

                     // Is this column the diagonal
                     const bool is_diagonal = (colmap_input_d(device_nonlocal_j[device_nonlocal_i[i] + j]) == row_index_global);

                     // If our current tolerance is bigger than the max value we've seen so far
                     PetscScalar val = Kokkos::abs(device_nonlocal_vals[device_nonlocal_i[i] + j]);
                     // If we're not comparing against the diagonal when computing relative residual
                     if (not_include_diag && is_diagonal) val = -1.0;                  
                     if (val > thread_max) thread_max = val;

                  },
                  Kokkos::Max<PetscScalar>(max_val_nonlocal)
               );
               // Take max of local and nonlocal
               if (max_val_nonlocal > max_val) max_val = max_val_nonlocal;               
            }

            // Only want one thread in the team to write the result
            Kokkos::single(Kokkos::PerTeam(t), [&]() {
               rel_row_tol_d(i) = tol * max_val;
            });
      });
   }
   // If we're using a constant tolerance, we can just copy it in
   else
   {
      // Copy in the tolerance
      Kokkos::deep_copy(rel_row_tol_d, tol);   
   }

   // ~~~~~~~~~~~~
   // Need to count the number of nnzs we end up with, on each row and in total
   // ~~~~~~~~~~~~
   // Reduce over all the rows
   Kokkos::parallel_reduce(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t, PetscInt& thread_total) {

      const PetscInt i   = t.league_rank(); // row i
      const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];
      const PetscInt row_index_global = i + global_row_start;

      // For this row, would we expect the diagonal to be in the local block or in the nonlocal?
      // Trivially true in the local block for square matrices
      const bool expect_local_diagonal = row_index_global >= global_col_start && \
                           row_index_global < global_col_end_plus_one;

      // We have a custom reduction type defined - ReduceData
      // Which has both a nnz count for this row, but also tracks whether we 
      // found the diagonal
      ReduceData row_result;

      // Reduce over all the columns
      Kokkos::parallel_reduce(
         Kokkos::TeamVectorRange(t, ncols_local),
         [&](const PetscInt j, ReduceData& thread_data) {

            // Is this column the diagonal
            const bool is_diagonal = (device_local_j[device_local_i[i] + j] + global_col_start == row_index_global);
            
            // We have found a diagonal in this row
            if (is_diagonal) {
               thread_data.found_diagonal = true;
               // Mark that there is a diagonal in this row
               // Will only happen for one thread
               existing_diag_d(i) = true;
            }
            
            // If the value is bigger than the tolerance, we keep it
            if (Kokkos::abs(device_local_vals[device_local_i[i] + j]) >= rel_row_tol_d(i)) {
               // If this is the diagonal and we're dropping all diagonals don't add it
               if (!(allow_drop_diagonal_int == -1 && is_diagonal)) thread_data.count++;
            }
            // Or if it's small but its the diagonal and we're keeping diagonals
            else if (allow_drop_diagonal_int == 0 && is_diagonal) {
               thread_data.count++;
            }
         }, row_result
      );

      // We're finished our parallel reduction for this row
      // If we're lumping but there was no diagonal in this row and there
      // should be we'll have to add in a diagonal to the local block
      // This will add one for every thread in this team, but all 
      // the threads in this team share the same result after the reduction
      if (lump_int && expect_local_diagonal && !row_result.found_diagonal) row_result.count++;
      // Only want one thread in the team to write the result
      Kokkos::single(Kokkos::PerTeam(t), [&]() {      
         nnz_match_local_row_d(i) = row_result.count;
         thread_total += row_result.count;
      });
      },
      nnzs_match_local
   );

   // ~~~~~~~~~~~~

   // Need to do a scan on nnz_match_local_row_d to get where each row starts
   Kokkos::parallel_scan (local_rows, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
      // Inclusive scan
      update += nnz_match_local_row_d(i);         
      if (final) {
         nnz_match_local_row_d(i) = update; // only update array on final pass
      }
   });            

   if (mpi) 
   {

      // ~~~~~~~~~~~~
      // Need to count the number of nnzs we end up with, on each row and in total
      // ~~~~~~~~~~~~
      // Reduce over all the rows
      Kokkos::parallel_reduce(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t, PetscInt& thread_total) {
            
            const PetscInt i = t.league_rank();
            const PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
            const PetscInt row_index_global = i + global_row_start;

            // For this row, would we expect the diagonal to be in the local block or in the nonlocal?
            const bool expect_local_diagonal = row_index_global >= global_col_start && \
                                 row_index_global < global_col_end_plus_one;

            ReduceData row_result;

            // Reduce over all the columns
            Kokkos::parallel_reduce(
               Kokkos::TeamVectorRange(t, ncols_nonlocal),
               [&](const PetscInt j, ReduceData& thread_data) {

                  // Is this column the diagonal
                  const bool is_diagonal = (colmap_input_d(device_nonlocal_j[device_nonlocal_i[i] + j]) == row_index_global);

                  // We have found a diagonal in this row
                  if (is_diagonal) {
                     thread_data.found_diagonal = true;
                     // Mark that there is a diagonal in this row
                     // Will only happen for one thread
                     existing_diag_d(i) = true;                     
                  }                  

                  // If the value is bigger than the tolerance, we keep it
                  if (Kokkos::abs(device_nonlocal_vals[device_nonlocal_i[i] + j]) >= rel_row_tol_d(i)) {
                     // If this is the diagonal and we're dropping all diagonals don't add it
                     if (!(allow_drop_diagonal_int == -1 && is_diagonal)) thread_data.count++;
                  }
                  // Or if it's small but its the diagonal and we're keeping diagonals
                  else if (allow_drop_diagonal_int == 0 && is_diagonal) {
                     thread_data.count++;
                  }                  
               },
               row_result
            );
            
            // We're finished our parallel reduction for this row
            // If we're lumping but there was no diagonal in this row and 
            // there should be (ie !expect_local_diagonal) we'll have to add in a diagonal
            // to the nonlocal block
            // This will add one for every thread in this team, but all 
            // the threads in this team share the same result after the reduction         
            if (lump_int && !expect_local_diagonal && !row_result.found_diagonal) row_result.count++;
            // Only want one thread in the team to write the result
            Kokkos::single(Kokkos::PerTeam(t), [&]() {      
               nnz_match_nonlocal_row_d(i) = row_result.count;
               thread_total += row_result.count;
            });
         },
         nnzs_match_nonlocal
      );

      // ~~~~~~~~~~~~

      // Need to do a scan on nnz_match_nonlocal_row_d to get where each row starts
      Kokkos::parallel_scan (local_rows, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
         // Inclusive scan
         update += nnz_match_nonlocal_row_d(i);         
         if (final) {
            nnz_match_nonlocal_row_d(i) = update; // only update array on final pass
         }
      });               
   }       

   // Find maximum non-zeros per row of the input mat for sizing scratch memory
   PetscInt max_nnz_local = 0, max_nnz_nonlocal = 0;
   if (local_rows > 0) {

      Kokkos::parallel_reduce("FindMaxNNZ", local_rows,
         KOKKOS_LAMBDA(const PetscInt i, PetscInt& thread_max) {
            PetscInt row_nnz = device_local_i[i + 1] - device_local_i[i];
            thread_max = (row_nnz > thread_max) ? row_nnz : thread_max;
         },
         Kokkos::Max<PetscInt>(max_nnz_local)
      );
      if (mpi)
      {
         Kokkos::parallel_reduce("FindMaxNNZNonLocal", local_rows,
            KOKKOS_LAMBDA(const PetscInt i, PetscInt& thread_max) {
               PetscInt row_nnz = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
               thread_max = (row_nnz > thread_max) ? row_nnz : thread_max;
            },
            Kokkos::Max<PetscInt>(max_nnz_nonlocal)
         );         
      }
   }      

   // ~~~~~~~~~~~~~~~~~  
   // We need to assemble our i,j, vals so we can build our matrix
   // ~~~~~~~~~~~~~~~~~
   // Create memory on the device and host
   Kokkos::View<PetscScalar *> a_local_d = Kokkos::View<PetscScalar *>("a_local_d", nnzs_match_local);
   Kokkos::View<PetscInt *> i_local_d = Kokkos::View<PetscInt *>("i_local_d", local_rows+1);
   Kokkos::View<PetscInt *> j_local_d = Kokkos::View<PetscInt *>("j_local_d", nnzs_match_local);

   // Initialize first entry to zero - the rest get set below
   Kokkos::deep_copy(Kokkos::subview(i_local_d, 0), 0);     

   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
      i_local_d(i + 1) = nnz_match_local_row_d(i);
   });  

   // Nonlocal stuff 
   Kokkos::View<PetscScalar *> a_nonlocal_d;
   Kokkos::View<PetscInt *> i_nonlocal_d;          
   Kokkos::View<PetscInt *> j_nonlocal_d;    

   // we also have to go and build the a, i, j for the non-local off-diagonal block
   if (mpi) 
   {
      // Non-local 
      a_nonlocal_d = Kokkos::View<PetscScalar *>("a_nonlocal_d", nnzs_match_nonlocal);
      i_nonlocal_d = Kokkos::View<PetscInt *>("i_nonlocal_d", local_rows+1);
      j_nonlocal_d = Kokkos::View<PetscInt *>("j_nonlocal_d", nnzs_match_nonlocal);  

      // Initialize first entry to zero - the rest get set below
      Kokkos::deep_copy(Kokkos::subview(i_nonlocal_d, 0), 0);                 

      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
         i_nonlocal_d(i + 1) = nnz_match_nonlocal_row_d(i);
      });      
   }           
   
   auto exec = PetscGetKokkosExecutionSpace();
   
   // Create a team policy with scratch memory allocation
   // We want scratch space for each row
   // We will ncols+1 of integers which tell us what the matching indices we have
   // the last bit of memory is to account for 8-byte alignment for each view
   size_t scratch_size_per_team = 1 * (max_nnz_local+1) * sizeof(PetscInt) + \
               1 * (max_nnz_nonlocal+1) * sizeof(PetscInt) +
               8 * 2 * sizeof(PetscScalar);

   Kokkos::TeamPolicy<> policy(exec, local_rows, Kokkos::AUTO());
   // We're gonna use the level 1 scratch as our column data is probably bigger than the level 0
   policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size_per_team));

   // Execute with scratch memory
   Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const KokkosTeamMemberType& t) {
         
      // Row
      const PetscInt i = t.league_rank();         
      // number of columns
      PetscInt ncols_local, ncols_nonlocal=-1;
      ncols_local = device_local_i[i + 1] - device_local_i[i];
      const PetscInt row_index_global = i + global_row_start;
      ScratchIntView scratch_indices, scratch_indices_nonlocal;

      // Allocate views directly on scratch memory
      // Have to use views here given alignment issues
      // We have of size ncols+1 to account for the exclusive scan
      scratch_indices = ScratchIntView(t.team_scratch(1), ncols_local+1); 
      
      // Lumped values we compute as we go for this row
      PetscScalar lump_val_local = 0.0, lump_val_nonlocal = 0.0;

      if (mpi) 
      {
         ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
         scratch_indices_nonlocal = ScratchIntView(t.team_scratch(1), ncols_nonlocal+1);

         Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_nonlocal+1), [&](const PetscInt j) {
            scratch_indices_nonlocal(j) = 0;
         });         
      }     
      
      // Initialize scratch
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_local+1), [&](const PetscInt j) {
         scratch_indices(j) = 0;
      });

      // Team barrier to ensure all threads have finished filling the scratch space
      t.team_barrier();

      // Now go and mark which values we're keeping and lumping
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j, PetscScalar& thread_sum) {

         bool keep_col = false;
         const bool is_diagonal = (device_local_j[device_local_i[i] + j] + global_col_start == row_index_global);

         // If we hit a diagonal put it in the lump'd value
         if (is_diagonal && lump_int) thread_sum += device_local_vals[device_local_i[i] + j];           
         
         // Check if we keep this column because of size
         if (Kokkos::abs(device_local_vals[device_local_i[i] + j]) >= rel_row_tol_d(i)) {
            // If this is the diagonal and we're dropping all diagonals don't add it
            if (!(allow_drop_diagonal_int == -1 && is_diagonal)) keep_col = true;
         }
         // Or if we keep it because we're not dropping diagonals
         else if (allow_drop_diagonal_int == 0 && is_diagonal) {
            keep_col = true;
         }
         
         // Mark this as a match
         if (keep_col) {
            scratch_indices(j) = 1;
         }
         // If we're not on the diagonal and we're small enough to lump
         else if (lump_int && !is_diagonal) thread_sum += device_local_vals[device_local_i[i] + j];       
         },
         Kokkos::Sum<PetscScalar>(lump_val_local)
      ); 

      if (mpi)
      {
         // Now go and mark which values we're keeping and lumping
         Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j, PetscScalar& thread_sum) {

            bool keep_col = false;
            // Remember we can have diagonals in the off-diagonal block if we're rectangular
            const bool is_diagonal = (colmap_input_d(device_nonlocal_j[device_nonlocal_i[i] + j]) == row_index_global);
      
            // If we hit a diagonal put it in the lump'd value
            if (is_diagonal && lump_int) thread_sum += device_nonlocal_vals[device_nonlocal_i[i] + j];

            // Check if we keep this column because of size
            if (Kokkos::abs(device_nonlocal_vals[device_nonlocal_i[i] + j]) >= rel_row_tol_d(i)) {
               // If this is the diagonal and we're dropping all diagonals don't add it
               if (!(allow_drop_diagonal_int == -1 && is_diagonal)) keep_col = true;
            }
            // Or if we keep it because we're not dropping diagonals
            else if (allow_drop_diagonal_int == 0 && is_diagonal) {
               keep_col = true;
            }
            
            // Mark this as a match
            if (keep_col) {
               scratch_indices_nonlocal(j) = 1;             
            }
            // If we're not on the diagonal and we're small enough to lump
            else if (lump_int && !is_diagonal) thread_sum += device_nonlocal_vals[device_nonlocal_i[i] + j];
            },
            Kokkos::Sum<PetscScalar>(lump_val_nonlocal)
         );
      }
      // Sum the local and nonlocal lumped values
      PetscScalar lump_val = lump_val_local + lump_val_nonlocal;      

      // Team barrier to ensure all threads have finished filling the scratch space
      t.team_barrier(); 
      
      // Perform exclusive scan over scratch_indices to get our output indices in this row
      // Have to be careful to go up to ncols_local+1
      Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, ncols_local+1), 
         [&](const PetscInt j, int& partial_sum, const bool is_final) {
            const int input_value = scratch_indices(j);
            if (is_final) {
                  scratch_indices(j) = partial_sum;  // Write exclusive prefix
            }
            partial_sum += input_value;  // Update running total
         }
      );     
      
      if (mpi)
      {
         Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, ncols_nonlocal+1), 
         [&](const PetscInt j, int& partial_sum, const bool is_final) {
            const int input_value = scratch_indices_nonlocal(j);
            if (is_final) {
                  scratch_indices_nonlocal(j) = partial_sum;  // Write exclusive prefix
            }
            partial_sum += input_value;  // Update running total
         }
      );          
      }

      // Team barrier to ensure all threads have finished scanning scratch_indices
      t.team_barrier();

      // Now go and write to the output
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {
         // We can tell if scratch_indices had 1 in it in this position by comparing the result
         // of the exclusive scan for this index and the next one
         if (scratch_indices(j+1) > scratch_indices(j))
         {
            j_local_d(i_local_d(i) + scratch_indices(j)) = device_local_j[device_local_i[i] + j];
            a_local_d(i_local_d(i) + scratch_indices(j)) = device_local_vals[device_local_i[i] + j];            
         }
      });

      if (mpi)
      {
         Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {
            if (scratch_indices_nonlocal(j+1) > scratch_indices_nonlocal(j))
            {
               // Writing the global column indices, this will get compactified below
               j_nonlocal_d(i_nonlocal_d(i) + scratch_indices_nonlocal(j)) = colmap_input_d(device_nonlocal_j[device_nonlocal_i[i] + j]);
               a_nonlocal_d(i_nonlocal_d(i) + scratch_indices_nonlocal(j)) = device_nonlocal_vals[device_nonlocal_i[i] + j];     
            }
         });         
      }

      // Team barrier to ensure all threads have finished updating the output
      t.team_barrier();      

      // Add in the lumped value to the diagonal
      if (lump_int)
      {
         // For this row, would we expect the diagonal to be in the local block or in the nonlocal?
         // Trivially true in the local block for square matrices
         const bool expect_local_diagonal = row_index_global >= global_col_start && \
                              row_index_global < global_col_end_plus_one;

         // Does this row already have a diagonal?
         if (existing_diag_d(i))
         {
            if (expect_local_diagonal)
            {
               // Only loop over the ncols in the local component - make sure this is over the output number of cols
               Kokkos::parallel_for(Kokkos::TeamThreadRange(t, i_local_d(i+1) - i_local_d(i)), [&](const PetscInt j) {

                  // Is this column the diagonal
                  const bool is_diagonal = j_local_d[i_local_d[i] + j] + global_col_start == row_index_global;

                  // Will only happen for one thread - lump_val contains the diagonal so we overwrite
                  if (is_diagonal) a_local_d[i_local_d[i] + j] = lump_val;
               });   
            }
            else
            {
               // Only loop over the ncols in the nonlocal component  - make sure this is over the output number of cols
               Kokkos::parallel_for(Kokkos::TeamThreadRange(t, i_nonlocal_d(i+1) - i_nonlocal_d(i)), [&](const PetscInt j) {

                  // Is this column the diagonal - j_nonlocal_d contains the global column index
                  const bool is_diagonal = j_nonlocal_d[i_nonlocal_d[i] + j] == row_index_global;

                  // Will only happen for one thread - lump_val contains the diagonal so we overwrite
                  if (is_diagonal) a_nonlocal_d[i_nonlocal_d[i] + j] = lump_val;
               });               
            }
         }
         // If this row doesn't have a diagonal already, add it to the end and we'll sort after
         // We have already correctly allocated extra space for this entry
         else
         {
            // We only need to write a single entry
            Kokkos::single(Kokkos::PerTeam(t), [&]() {
               // lump_val contains the diagonal so we overwrite
               if (expect_local_diagonal)
               {
                  // Has to be the local column index
                  j_local_d[i_local_d[i+1] - 1] = i;
                  a_local_d[i_local_d[i+1] - 1] = lump_val;
               }
               else
               {
                  // Has to be global column for now
                  j_nonlocal_d[i_nonlocal_d[i+1] - 1] = row_index_global;
                  a_nonlocal_d[i_nonlocal_d[i+1] - 1] = lump_val;
               }        
            });    
         }
      }    
   });  

   // Let's make sure everything on the device is finished
   exec.fence();

   // Now we may have to sort the column indices
   if (lump_int)
   {  
      // Reduce to see if we ever added a diagonal
      bool added_any_diagonal = false;
      Kokkos::parallel_reduce(
         Kokkos::RangePolicy<>(0, local_rows),
         KOKKOS_LAMBDA(const PetscInt i, bool& thread_result) {
            // If this row had a diagonal added, set the result to true
            if (!existing_diag_d(i)) thread_result = true;
         },
         Kokkos::LOr<bool>(added_any_diagonal)
      ); 

      // If we did add a diagonal, it got added to the end of the columns on each row, so will have to sort
      // It also could have been added to either the local or nonlocal components given not square
      if (added_any_diagonal) 
      {
         KokkosCsrMatrix csrmat_local = KokkosCsrMatrix("csrmat_local", local_rows, local_cols, a_local_d.extent(0), a_local_d, i_local_d, j_local_d);  
         KokkosSparse::sort_crs_matrix(csrmat_local); 
         
         if (mpi)
         {
            // The column size is not right here (it will be <= cols_ao)
            // but it shouldn't matter as we are only construting an explicit kokkos csr matrix here so it can sort
            KokkosCsrMatrix csrmat_nonlocal = KokkosCsrMatrix("csrmat_nonlocal", local_rows, cols_ao, a_nonlocal_d.extent(0), a_nonlocal_d, i_nonlocal_d, j_nonlocal_d);  
            KokkosSparse::sort_crs_matrix(csrmat_nonlocal);         
         }
      }
   }

   // We can create our local diagonal block matrix directly on the device
   MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, local_cols, i_local_d, j_local_d, a_local_d, &output_mat_local);  

   // we also have to go and build the a, i, j for the non-local off-diagonal block
   if (mpi) 
   {
      // Now we need to build garray on the host and rewrite the j_nonlocal_d indices so they are local
      // The default values here are for the case where we 
      // let petsc do it, it resets this internally in MatSetUpMultiply_MPIAIJ
      PetscInt *garray_host = NULL;
      PetscInt col_ao_output = 0;
      rewrite_j_global_to_local(cols_ao, col_ao_output, j_nonlocal_d, &garray_host);  

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

   return;
}


//------------------------------------------------------------------------------------------------------------------------

// Drop according to a existing sparsity in output_mat but with kokkos - keeping everything on the device
PETSC_INTERN void remove_from_sparse_match_kokkos(Mat *input_mat, Mat *output_mat, const int lump_int, const int alpha_int, const PetscReal alpha)
{

   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start_temp, global_row_end_plus_one_temp;
   PetscInt global_col_start_temp, global_col_end_plus_one_temp;
   PetscInt rows_ao_input, cols_ao_input, rows_ao_output, cols_ao_output;
   MatType mat_type;

   MatGetType(*input_mat, &mat_type);
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   // Get the comm
   PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX);
   MatGetLocalSize(*input_mat, &local_rows, &local_cols);
   MatGetSize(*input_mat, &global_rows, &global_cols);
   // This returns the global index of the local portion of the matrix
   MatGetOwnershipRange(*input_mat, &global_row_start_temp, &global_row_end_plus_one_temp);
   MatGetOwnershipRangeColumn(*input_mat, &global_col_start_temp, &global_col_end_plus_one_temp);
   const PetscInt global_row_start = global_row_start_temp;
   //const PetscInt global_row_end_plus_one = global_row_end_plus_one_temp;
   const PetscInt global_col_start = global_col_start_temp;
   const PetscInt global_col_end_plus_one = global_col_end_plus_one_temp;   

   Mat_MPIAIJ *mat_mpi = nullptr;
   Mat mat_local = NULL, mat_nonlocal = NULL;
   Mat_MPIAIJ *mat_mpi_output = nullptr;
   Mat mat_local_output = NULL, mat_nonlocal_output = NULL;   

   PetscIntKokkosViewHost colmap_input_h, colmap_output_h;
   PetscIntKokkosView colmap_input_d, colmap_output_d;   
   if (mpi)
   {
      mat_mpi = (Mat_MPIAIJ *)(*input_mat)->data;
      mat_local = mat_mpi->A;
      mat_nonlocal = mat_mpi->B;
      MatGetSize(mat_nonlocal, &rows_ao_input, &cols_ao_input); 

      // We also copy the input mat colmap over to the device as we need it
      colmap_input_h = PetscIntKokkosViewHost(mat_mpi->garray, cols_ao_input);
      colmap_input_d = PetscIntKokkosView("colmap_input_d", cols_ao_input);
      Kokkos::deep_copy(colmap_input_d, colmap_input_h);  

      // Log copy with petsc
      size_t bytes = colmap_input_h.extent(0) * sizeof(PetscInt);
      PetscLogCpuToGpu(bytes);
      
      // Same for output
      mat_mpi_output = (Mat_MPIAIJ *)(*output_mat)->data;
      mat_local_output = mat_mpi_output->A;
      mat_nonlocal_output = mat_mpi_output->B;  
      MatGetSize(mat_nonlocal_output, &rows_ao_output, &cols_ao_output); 

      colmap_output_h = PetscIntKokkosViewHost(mat_mpi_output->garray, cols_ao_output);
      colmap_output_d = PetscIntKokkosView("colmap_output_d", cols_ao_output);
      Kokkos::deep_copy(colmap_output_d, colmap_output_h); 

      bytes = colmap_output_h.extent(0) * sizeof(PetscInt);
      PetscLogCpuToGpu(bytes);      
 
   }
   else
   {
      mat_local = *input_mat;
      mat_local_output = *output_mat;
   }

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr, *device_nonlocal_vals = nullptr;  
   MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype);  
   if (mpi) MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype);          

   // Get the output pointers
   const PetscInt *device_local_i_output = nullptr, *device_local_j_output = nullptr, *device_nonlocal_i_output = nullptr, *device_nonlocal_j_output = nullptr;
   PetscScalar *device_local_vals_output = nullptr, *device_nonlocal_vals_output = nullptr;  
   MatSeqAIJGetCSRAndMemType(mat_local_output, &device_local_i_output, &device_local_j_output, &device_local_vals_output, &mtype);  
   if (mpi) MatSeqAIJGetCSRAndMemType(mat_nonlocal_output, &device_nonlocal_i_output, &device_nonlocal_j_output, &device_nonlocal_vals_output, &mtype); 

   Mat_SeqAIJKokkos *aijkok_local_output = static_cast<Mat_SeqAIJKokkos *>(mat_local_output->spptr);
   Mat_SeqAIJKokkos *aijkok_nonlocal_output = NULL;
   if (mpi) aijkok_nonlocal_output = static_cast<Mat_SeqAIJKokkos *>(mat_nonlocal_output->spptr);   

   // Find maximum non-zeros per row of the input mat for sizing scratch memory
   PetscInt max_nnz_local = 0, max_nnz_nonlocal = 0;
   if (local_rows > 0) {

      Kokkos::parallel_reduce("FindMaxNNZ", local_rows,
         KOKKOS_LAMBDA(const PetscInt i, PetscInt& thread_max) {
            PetscInt row_nnz = device_local_i[i + 1] - device_local_i[i];
            thread_max = (row_nnz > thread_max) ? row_nnz : thread_max;
         },
         Kokkos::Max<PetscInt>(max_nnz_local)
      );
      if (mpi)
      {
         Kokkos::parallel_reduce("FindMaxNNZNonLocal", local_rows,
            KOKKOS_LAMBDA(const PetscInt i, PetscInt& thread_max) {
               PetscInt row_nnz = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
               thread_max = (row_nnz > thread_max) ? row_nnz : thread_max;
            },
            Kokkos::Max<PetscInt>(max_nnz_nonlocal)
         );         
      }
   }   

   auto exec = PetscGetKokkosExecutionSpace();
   
   // Create a team policy with scratch memory allocation
   // We want scratch space for each row
   // We will have ncols of integers which tell us what the matching indices we have
   // the last bit of memory is to account for 8-byte alignment for each view
   size_t scratch_size_per_team = max_nnz_local * sizeof(PetscInt) + \
               max_nnz_nonlocal * sizeof(PetscInt) +
               8 * 2 * sizeof(PetscScalar);

   Kokkos::TeamPolicy<> policy(exec, local_rows, Kokkos::AUTO());
   // We're gonna use the level 1 scratch as our column data is probably bigger than the level 0
   policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size_per_team));

   // Execute with scratch memory
   Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const KokkosTeamMemberType& t) {

      // Row
      const PetscInt i = t.league_rank();
      const PetscInt row_index_global = i + global_row_start;

      // number of columns
      PetscInt ncols_local, ncols_nonlocal=-1, ncols_local_output, ncols_nonlocal_output=-1;
      ncols_local = device_local_i[i + 1] - device_local_i[i];
      // Allocate views directly on scratch memory
      // Have to use views here given alignment issues
      ScratchIntView scratch_indices, scratch_indices_nonlocal;
      scratch_indices = ScratchIntView(t.team_scratch(1), ncols_local); 

      if (mpi) 
      {
         ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
         ncols_nonlocal_output = device_nonlocal_i_output[i + 1] - device_nonlocal_i_output[i];
         scratch_indices_nonlocal = ScratchIntView(t.team_scratch(1), ncols_nonlocal);
         Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {
            scratch_indices_nonlocal(j) = -1;
         });          
      }
      ncols_local_output = device_local_i_output[i + 1] - device_local_i_output[i];

      // Initialize scratch
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {
            scratch_indices(j) = -1;
      });

      // Team barrier to ensure all threads have finished filling the scratch space
      t.team_barrier();      

      // Perform Sorted Merge using a single thread per team
      // Should be faster than having each input column loop through every output column in parallel

      // Do the local columns
      Kokkos::single(Kokkos::PerTeam(t), [&]() {
         PetscInt idx_input = 0;  // Pointer for input columns (0 to ncols_local-1)
         PetscInt idx_output = 0; // Pointer for output columns (0 to ncols_local_output-1)

         while (idx_input < ncols_local && idx_output < ncols_local_output) {
            // Get current global input column index
             PetscInt col_input = device_local_j[device_local_i[i] + idx_input] + global_col_start;

            // Get current global output column index
            PetscInt col_output = device_local_j_output[device_local_i_output[i] + idx_output] + global_col_start;

            // Compare and advance pointers
            if (col_input == col_output) {
               // Match found! Record output index
               scratch_indices(idx_input) = idx_output;
               idx_input++;
               idx_output++;
            } else if (col_input < col_output) {
               idx_input++;
            } else {
               idx_output++;
            }
         }
      });

      // Do the nonlocal columns
      if (mpi)
      {
         Kokkos::single(Kokkos::PerTeam(t), [&]() {
            PetscInt idx_input = 0;  // Pointer for input columns (0 to ncols_nonlocal-1)
            PetscInt idx_output = 0; // Pointer for output columns (0 to ncols_nonlocal_output-1)

            while (idx_input < ncols_nonlocal && idx_output < ncols_nonlocal_output) {
               // Get current global input column index
               PetscInt col_input = colmap_input_d(device_nonlocal_j[device_nonlocal_i[i] + idx_input]);

               // Get current global output column index
               PetscInt col_output = colmap_output_d(device_nonlocal_j_output[device_nonlocal_i_output[i] + idx_output]);

               // Compare and advance pointers
               if (col_input == col_output) {
                  // Match found! Record output index
                  scratch_indices_nonlocal(idx_input) = idx_output;
                  idx_input++;
                  idx_output++;
               } else if (col_input < col_output) {
                  idx_input++;
               } else {
                  idx_output++;
               }
            }
         });  
      }    

      // Team barrier to ensure all threads have finished filling the scratch space
      t.team_barrier();      

      PetscScalar lump_val = 0.0;
      // If lumping need to sum all the non-matching terms in input
      if (lump_int)
      {
         PetscScalar lump_val_local = 0.0, lump_val_nonlocal = 0.0;
         
         // Reduce over local columns
         Kokkos::parallel_reduce(
            Kokkos::TeamVectorRange(t, ncols_local),
            [&](const PetscInt j, PetscScalar& thread_sum) {          

               // If this is not being put into output then we lump it
               if (scratch_indices(j) == -1) thread_sum += alpha * device_local_vals[device_local_i[i] + j];
            },
            Kokkos::Sum<PetscScalar>(lump_val_local)
         );   

         if (mpi)
         {
            // Reduce over nonlocal columns
            Kokkos::parallel_reduce(
               Kokkos::TeamVectorRange(t, ncols_nonlocal),
               [&](const PetscInt j, PetscScalar& thread_sum) {           

                  // If this is not being put into output then we lump it
                  if (scratch_indices_nonlocal(j) == -1) thread_sum += alpha * device_nonlocal_vals[device_nonlocal_i[i] + j];
               },
               Kokkos::Sum<PetscScalar>(lump_val_nonlocal)
            );              
         }
         lump_val = lump_val_local + lump_val_nonlocal;
      }   

      // Now go and write to the output
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

         // If we have a match, copy the value
         if (scratch_indices(j) != -1)
         {
            if (alpha_int)
            {
               device_local_vals_output[device_local_i_output[i] + scratch_indices(j)] += alpha * device_local_vals[device_local_i[i] + j];
            }
            else
            {
               device_local_vals_output[device_local_i_output[i] + scratch_indices(j)] = device_local_vals[device_local_i[i] + j];
            }
         }
      }); 
      
      if (mpi)
      {
         Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) { 
   
            // If we have a match, copy the value
            if (scratch_indices_nonlocal(j) != -1)
            {
               // Writing the global column indices, this will get compactified below
               if (alpha_int)
               {
                  device_nonlocal_vals_output[device_nonlocal_i_output[i] + scratch_indices_nonlocal(j)] += alpha * device_nonlocal_vals[device_nonlocal_i[i] + j];
               }
               else
               {
                  device_nonlocal_vals_output[device_nonlocal_i_output[i] + scratch_indices_nonlocal(j)] = device_nonlocal_vals[device_nonlocal_i[i] + j];
               }
            }
         });          
      }      
      
      // Team barrier to ensure all threads have finished writing to output
      t.team_barrier();      

      // Add in the lumped value to the diagonal
      if (lump_int)
      {
         // For this row, would we expect the diagonal to be in the local block or in the nonlocal?
         // Trivially true in the local block for square matrices
         const bool expect_local_diagonal = row_index_global >= global_col_start && \
                              row_index_global < global_col_end_plus_one;

         if (expect_local_diagonal)
         {
            // Only loop over the ncols in the local component
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_local_output), [&](const PetscInt j) {

               // Is this column the diagonal
               const bool is_diagonal = device_local_j_output[device_local_i_output[i] + j] + global_col_start == row_index_global;

               // Will only happen for one thread
               if (is_diagonal) device_local_vals_output[device_local_i_output[i] + j] += lump_val;
            });   
         }
         else
         {
            // Only loop over the ncols in the nonlocal component
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_nonlocal_output), [&](const PetscInt j) {

               // Is this column the diagonal
               const bool is_diagonal = colmap_output_d(device_nonlocal_j_output[device_nonlocal_i_output[i] + j]) == row_index_global;

               // Will only happen for one thread
               if (is_diagonal) device_nonlocal_vals_output[device_nonlocal_i_output[i] + j] += lump_val;
            });               
         }
      }       
   });

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

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Set all the values of the matrix to val
PETSC_INTERN void MatSetAllValues_kokkos(Mat *input_mat, PetscReal val)
{
   MatType mat_type;

   MatGetType(*input_mat, &mat_type);
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat_MPIAIJ *mat_mpi = nullptr;
   Mat mat_local = NULL, mat_nonlocal = NULL;
  
   if (mpi)
   {
      mat_mpi = (Mat_MPIAIJ *)(*input_mat)->data;
      mat_local = mat_mpi->A;
      mat_nonlocal = mat_mpi->B;
   }
   else
   {
      mat_local = *input_mat;
   }
   PetscInt local_rows, local_cols;
   MatGetLocalSize(*input_mat, &local_rows, &local_cols);

   Mat_SeqAIJKokkos *aijkok_nonlocal = NULL;
   Mat_SeqAIJKokkos *aijkok_local = static_cast<Mat_SeqAIJKokkos *>(mat_local->spptr);
   if(mpi) aijkok_nonlocal = static_cast<Mat_SeqAIJKokkos *>(mat_nonlocal->spptr);
   
   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr, *device_nonlocal_vals = nullptr;  
   MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype);  
   if (mpi) MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype);          

   PetscScalarKokkosView a_local_d, a_nonlocal_d;
   a_local_d = PetscScalarKokkosView(device_local_vals, aijkok_local->csrmat.nnz());   
   if (mpi) a_nonlocal_d = PetscScalarKokkosView(device_nonlocal_vals, aijkok_nonlocal->csrmat.nnz()); 
   // Copy in the val
   Kokkos::deep_copy(a_local_d, val); 
   // Log copy with petsc
   size_t bytes = sizeof(PetscReal);
   PetscLogCpuToGpu(bytes);   
   if (mpi) 
   {  
      Kokkos::deep_copy(a_nonlocal_d, val); 
      PetscLogCpuToGpu(bytes);   
   }

   // Have to specify we've modifed data on the device
   // Want to call MatSeqAIJKokkosModifyDevice but its PETSC_INTERN

   aijkok_local->a_dual.clear_sync_state();
   aijkok_local->a_dual.modify_device();
   aijkok_local->transpose_updated = PETSC_FALSE;
   aijkok_local->hermitian_updated = PETSC_FALSE;
   // Invalidate diagonals
   Mat_SeqAIJ *a = (Mat_SeqAIJ *)mat_local->data;
   a->idiagvalid  = PETSC_FALSE;
   a->ibdiagvalid = PETSC_FALSE;      
   a->inode.ibdiagvalid = PETSC_FALSE;  

   if (mpi)
   {
      aijkok_nonlocal->a_dual.clear_sync_state();
      aijkok_nonlocal->a_dual.modify_device();
      aijkok_nonlocal->transpose_updated = PETSC_FALSE;
      aijkok_nonlocal->hermitian_updated = PETSC_FALSE;
      a = (Mat_SeqAIJ *)mat_nonlocal->data;
      a->idiagvalid  = PETSC_FALSE;
      a->ibdiagvalid = PETSC_FALSE;   
      a->inode.ibdiagvalid = PETSC_FALSE;    
   }
   PetscObjectStateIncrease((PetscObject)(*input_mat));

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Duplicate and copy a matrix ensuring it always has a diagonal but with kokkos - keeping everything on the device
PETSC_INTERN void mat_duplicate_copy_plus_diag_kokkos(Mat *input_mat, const int reuse_int, Mat *output_mat)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt global_row_start_temp, global_row_end_plus_one_temp;
   PetscInt global_col_start_temp, global_col_end_plus_one_temp;
   PetscInt rows_ao, cols_ao;
   PetscInt global_rows, global_cols;
   PetscInt local_rows, local_cols;
   MatType mat_type;
   PetscInt nnzs_match_local, nnzs_match_nonlocal;
   Mat output_mat_local, output_mat_nonlocal;

   MatGetType(*input_mat, &mat_type);
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat_MPIAIJ *mat_mpi = nullptr;
   Mat mat_local = NULL, mat_nonlocal = NULL;

   PetscIntKokkosViewHost colmap_input_h;
   PetscIntKokkosView colmap_input_d;   
   if (mpi)
   {
      mat_mpi = (Mat_MPIAIJ *)(*input_mat)->data;
      mat_local = mat_mpi->A;
      mat_nonlocal = mat_mpi->B;
      MatGetSize(mat_nonlocal, &rows_ao, &cols_ao); 

      // // We also copy the input mat colmap over to the device as we need it
      // colmap_input_h = PetscIntKokkosViewHost(mat_mpi->garray, cols_ao);
      // colmap_input_d = PetscIntKokkosView("colmap_input_d", cols_ao);
      // Kokkos::deep_copy(colmap_input_d, colmap_input_h);        
   }
   else
   {
      mat_local = *input_mat;
   }

   // Get the comm
   PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX);
   MatGetLocalSize(*input_mat, &local_rows, &local_cols); 
   MatGetSize(*input_mat, &global_rows, &global_cols);
   MatGetOwnershipRange(*input_mat, &global_row_start_temp, &global_row_end_plus_one_temp);
   MatGetOwnershipRangeColumn(*input_mat, &global_col_start_temp, &global_col_end_plus_one_temp);
   const PetscInt global_row_start = global_row_start_temp;
   //const PetscInt global_row_end_plus_one = global_row_end_plus_one_temp;
   const PetscInt global_col_start = global_col_start_temp;
   //const PetscInt global_col_end_plus_one = global_col_end_plus_one_temp;
   MatGetType(*input_mat, &mat_type);

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr, *device_nonlocal_vals = nullptr;  
   MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype);  
   if (mpi) MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype);          

   PetscIntKokkosView nnz_match_local_row_d;
   PetscIntKokkosView nnz_match_nonlocal_row_d;

   // Get device views
   Kokkos::View<PetscScalar *> a_local_d;
   Kokkos::View<PetscInt *> i_local_d;  
   Kokkos::View<PetscInt *> j_local_d;    

   // Nonlocal stuff 
   Kokkos::View<PetscScalar *> a_nonlocal_d;
   Kokkos::View<PetscInt *> i_nonlocal_d;          
   Kokkos::View<PetscInt *> j_nonlocal_d;  

   Mat_MPIAIJ *mat_mpi_output = nullptr;
   Mat mat_local_output = NULL, mat_nonlocal_output = NULL;   

   // We always need to know if we found a diagonal in each row of the input_matrix
   auto found_diag_row_d = PetscIntKokkosView("found_diag_row_d", local_rows);    
   Kokkos::deep_copy(found_diag_row_d, 0); 

   // ~~~~~~~~~~~~
   // Get the number of nnzs
   // ~~~~~~~~~~~~
   nnzs_match_local = 0;
   nnzs_match_nonlocal = 0;

   // We need to know how many entries are in each row 
   nnz_match_local_row_d = PetscIntKokkosView("nnz_match_local_row_d", local_rows);    
   // We may have identity
   Kokkos::deep_copy(nnz_match_local_row_d, 0);              
   if (mpi) nnz_match_nonlocal_row_d = PetscIntKokkosView("nnz_match_nonlocal_row_d", local_rows);      

   // Calculate if each row has a diagonal, we need to know this for both 
   // reuse and not reuse
   // For the local block we need to count the nnzs
   // but if there is no diagonal we need to add one in
   Kokkos::parallel_reduce(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t, PetscInt& thread_total) {

      const PetscInt i   = t.league_rank(); // row i
      const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];
      const PetscInt row_index_global = i + global_row_start;

      // We have a custom reduction type defined - ReduceData
      // Which has both a nnz count for this row, but also tracks whether we 
      // found the diagonal
      ReduceData row_result;

      // Reduce over all the columns
      Kokkos::parallel_reduce(
         Kokkos::TeamVectorRange(t, ncols_local),
         [&](const PetscInt j, ReduceData& thread_data) {

            // Is this column the diagonal
            const bool is_diagonal = (device_local_j[device_local_i[i] + j] + global_col_start == row_index_global);
            
            // We have found a diagonal in this row
            if (is_diagonal) {
               thread_data.found_diagonal = true;
            }
            thread_data.count++;
         }, row_result
      );
      
      // Only want one thread in the team to write the result
      Kokkos::single(Kokkos::PerTeam(t), [&]() {      
         if (!row_result.found_diagonal) 
         {
            row_result.count++;
         }
         else
         {
            found_diag_row_d(i) = 1;
         }
         nnz_match_local_row_d(i) = row_result.count;
         thread_total += row_result.count;            
      });
      },
      nnzs_match_local
   );         

   // Only need things to do with the sparsity pattern if we're not reusing
   if (!reuse_int)
   {
      // ~~~~~~~~~~~~
      // Need to count the number of nnzs we end up with, on each row and in total
      // ~~~~~~~~~~~~
      // Loop over the rows
      if (mpi)
      {      
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt i) {

               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
               nnz_match_nonlocal_row_d(i) = ncols_nonlocal;
         });
      }
      if (mpi)
      {
         Kokkos::parallel_reduce ("ReductionNonLocal", local_rows, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update) {
            update += nnz_match_nonlocal_row_d(i); 
         }, nnzs_match_nonlocal);       
      }

      // ~~~~~~~~~~~~

      // Need to do a scan on nnz_match_local_row_d to get where each row starts
      Kokkos::parallel_scan (local_rows, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
         // Inclusive scan
         update += nnz_match_local_row_d(i);         
         if (final) {
            nnz_match_local_row_d(i) = update; // only update array on final pass
         }
      });      
      if (mpi)
      { 
         // Need to do a scan on nnz_match_nonlocal_row_d to get where each row starts
         Kokkos::parallel_scan (local_rows, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
            // Inclusive scan
            update += nnz_match_nonlocal_row_d(i);         
            if (final) {
               nnz_match_nonlocal_row_d(i) = update; // only update array on final pass
            }
         });               
      }           

      // ~~~~~~~~~~~~~~~~~  
      // We need to assemble our i,j, vals so we can build our matrix
      // ~~~~~~~~~~~~~~~~~
      // Create memory on the device and host
      a_local_d = Kokkos::View<PetscScalar *>("a_local_d", nnzs_match_local);
      i_local_d = Kokkos::View<PetscInt *>("i_local_d", local_rows+1);
      j_local_d = Kokkos::View<PetscInt *>("j_local_d", nnzs_match_local);

      // Initialize first entry to zero - the rest get set below
      Kokkos::deep_copy(Kokkos::subview(i_local_d, 0), 0);       

      // we also have to go and build the a, i, j for the non-local off-diagonal block
      if (mpi) 
      {
         // Non-local 
         a_nonlocal_d = Kokkos::View<PetscScalar *>("a_nonlocal_d", nnzs_match_nonlocal);
         i_nonlocal_d = Kokkos::View<PetscInt *>("i_nonlocal_d", local_rows+1);
         j_nonlocal_d = Kokkos::View<PetscInt *>("j_nonlocal_d", nnzs_match_nonlocal);  

         // Initialize first entry to zero - the rest get set below
         Kokkos::deep_copy(Kokkos::subview(i_nonlocal_d, 0), 0);                
      }  

      // ~~~~~~~~~~~~~~~
      // Create i indices
      // ~~~~~~~~~~~~~~~
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt i) {      

            // The start of our row index comes from the scan
            i_local_d(i + 1) = nnz_match_local_row_d(i);   
            if (mpi) i_nonlocal_d(i + 1) = nnz_match_nonlocal_row_d(i);         
      });            


      // Loop over the rows
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            const PetscInt i = t.league_rank();

            // Still using i here (the local index into input)
            const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];         

            // For over local columns - copy in input
            Kokkos::parallel_for(
               Kokkos::TeamVectorRange(t, ncols_local), [&](const PetscInt j) {

               // Want the local col indices for the local block
               j_local_d(i_local_d(i) + j) = device_local_j[device_local_i[i] + j];
               a_local_d(i_local_d(i) + j) = device_local_vals[device_local_i[i] + j];
                     
            });     

            // For over nonlocal columns - copy in input
            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];         

               Kokkos::parallel_for(
                  Kokkos::TeamVectorRange(t, ncols_nonlocal), [&](const PetscInt j) {

                  // We keep the existing local indices in the off-diagonal block here
                  j_nonlocal_d(i_nonlocal_d(i) + j) = device_nonlocal_j[device_nonlocal_i[i] + j];
                  a_nonlocal_d(i_nonlocal_d(i) + j) = device_nonlocal_vals[device_nonlocal_i[i] + j];
                        
               });          
            }

            // Only want one thread to deal with the diagonal
            Kokkos::single(Kokkos::PerTeam(t), [&]() {            
               // If we didn't find a diagonal
               if (!found_diag_row_d(i))
               {
                  // Let's just stick it at the end and we will sort after
                  j_local_d(i_local_d(i) + ncols_local) = i;
                  a_local_d(i_local_d(i) + ncols_local) = 0.0;
               }
            });              
      }); 
   }
   // If we're reusing, we can just write directly to the existing views
   else
   {
      // Get the existing output mats
      if (mpi)
      {
         mat_mpi_output = (Mat_MPIAIJ *)(*output_mat)->data;
         mat_local_output = mat_mpi_output->A;
         mat_nonlocal_output = mat_mpi_output->B;     
      }
      else
      {
         mat_local_output = *output_mat;
      }     
      Mat_SeqAIJKokkos *aijkok_local_output = static_cast<Mat_SeqAIJKokkos *>(mat_local_output->spptr);
      Mat_SeqAIJKokkos *aijkok_nonlocal_output = NULL;
      if (mpi) aijkok_nonlocal_output = static_cast<Mat_SeqAIJKokkos *>(mat_nonlocal_output->spptr);

      // Annoying we can't just call MatSeqAIJGetKokkosView
      a_local_d = aijkok_local_output->a_dual.view_device();
      if (mpi) a_nonlocal_d = aijkok_nonlocal_output->a_dual.view_device();
      // Because we might be missing diagonals and we're going to skip some of them 
      // in the writing loop below
      Kokkos::deep_copy(a_local_d, 0.0);

      // Annoyingly there isn't currently the ability to get views for i (or j)
      const PetscInt *device_local_i_output = nullptr, *device_local_j_output = nullptr, *device_nonlocal_i_ouput = nullptr;
      PetscMemType mtype;
      MatSeqAIJGetCSRAndMemType(mat_local_output, &device_local_i_output, &device_local_j_output, NULL, &mtype);  
      if (mpi) MatSeqAIJGetCSRAndMemType(mat_nonlocal_output, &device_nonlocal_i_ouput, NULL, NULL, &mtype);  

      // Have these point at the existing i pointers - we only need the local j
      ConstMatRowMapKokkosView i_local_const_d = ConstMatRowMapKokkosView(device_local_i_output, local_rows+1);
      ConstMatRowMapKokkosView j_local_const_d = ConstMatRowMapKokkosView(device_local_j_output, aijkok_local_output->csrmat.nnz());
      ConstMatRowMapKokkosView i_nonlocal_const_d;
      if (mpi) i_nonlocal_const_d = ConstMatRowMapKokkosView(device_nonlocal_i_ouput, local_rows+1);         

      // Only have to write a but have to be careful as we may not have diagonals in some rows
      // in the input, but they are in the output
      // Loop over the rows - annoying we have const views as this is just the same loop as above
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            const PetscInt i = t.league_rank();

            // Still using i here (the local index into input)
            const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];         

            // For over local columns - copy in input
            // We have to skip over the identity entries, which we know are always C points
            Kokkos::parallel_for(
               Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

               PetscInt offset = 0;

               // If we're at or after the diagonal and there isn't actually a diagonal in the input
               // we know the output has a diagonal, so we skip ahead one in the output and 
               // leave it unassigned in the output (it gets set to zero above)
               if (j_local_const_d(i_local_const_d(i) + j) >= i && \
                     !found_diag_row_d(i)) offset = 1;

               a_local_d(i_local_const_d(i) + j + offset) = device_local_vals[device_local_i[i] + j];
            });  
           
            // For over nonlocal columns - copy in input - identical structure in the off-diag block
            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];         

               Kokkos::parallel_for(
                  Kokkos::TeamVectorRange(t, ncols_nonlocal), [&](const PetscInt j) {

                  // We keep the existing local indices in the off-diagonal block here
                  // we have all the same columns as input and hence the same garray
                  a_nonlocal_d(i_nonlocal_const_d(i) + j) = device_nonlocal_vals[device_nonlocal_i[i] + j];
                        
               });          
            }
      });   

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

    }

   // ~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~   

   if (!reuse_int)
   {
      // Let's make sure everything on the device is finished
      auto exec = PetscGetKokkosExecutionSpace();
      exec.fence();     

      // Now we have to sort the local column indices, as we add in the identity at the 
      // end of our local j indices      
      KokkosCsrMatrix csrmat_local = KokkosCsrMatrix("csrmat_local", local_rows, local_cols, a_local_d.extent(0), a_local_d, i_local_d, j_local_d);  
      KokkosSparse::sort_crs_matrix(csrmat_local);       

      // Let's make sure everything on the device is finished
      exec = PetscGetKokkosExecutionSpace();
      exec.fence();       

      // Create the matrix given the sorted csr
      MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, local_cols, i_local_d, j_local_d, a_local_d, &output_mat_local);                

      // we also have to go and build our off block matrix and then the output
      if (mpi) 
      {
         // We know the garray is just the original
         PetscInt *garray_host = NULL; 
         PetscMalloc1(cols_ao, &garray_host);
         for (PetscInt i = 0; i < cols_ao; i++)
         {
            garray_host[i] = mat_mpi->garray[i];
         }    
         
         // We can create our nonlocal diagonal block matrix directly on the device
         MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, cols_ao, i_nonlocal_d, j_nonlocal_d, a_nonlocal_d, &output_mat_nonlocal);      

         // We can now create our MPI matrix
         MatCreateMPIAIJWithSeqAIJ(MPI_COMM_MATRIX, global_rows, global_cols, output_mat_local, output_mat_nonlocal, garray_host, output_mat);         
      }    
      // If in serial 
      else
      {
         *output_mat = output_mat_local;
      }
   }  

   return;
}


//------------------------------------------------------------------------------------------------------------------------

// Does a MatAXPY for a MPIAIJ Kokkos matrix - the petsc version currently uses the host making it very slow
PETSC_INTERN void MatAXPY_kokkos(Mat *Y, PetscScalar alpha, Mat *X)
{

   Mat_MPIAIJ *mat_mpi_y = nullptr, *mat_mpi_x = nullptr;
   Mat mat_local_y = NULL, mat_nonlocal_y = NULL;
   Mat mat_local_x = NULL, mat_nonlocal_x = NULL;

   mat_mpi_y = (Mat_MPIAIJ *)(*Y)->data;
   mat_local_y = mat_mpi_y->A;
   mat_nonlocal_y = mat_mpi_y->B;

   mat_mpi_x = (Mat_MPIAIJ *)(*X)->data;
   mat_local_x = mat_mpi_x->A;
   mat_nonlocal_x = mat_mpi_x->B;

   Mat_SeqAIJKokkos *mat_local_ykok = static_cast<Mat_SeqAIJKokkos *>(mat_local_y->spptr);
   Mat_SeqAIJKokkos *mat_nonlocal_ykok = static_cast<Mat_SeqAIJKokkos *>(mat_nonlocal_y->spptr);
   Mat_SeqAIJKokkos *mat_local_xkok = static_cast<Mat_SeqAIJKokkos *>(mat_local_x->spptr);
   Mat_SeqAIJKokkos *mat_nonlocal_xkok = static_cast<Mat_SeqAIJKokkos *>(mat_nonlocal_x->spptr);

   // Equivalent to calling MatSeqAIJKokkosSyncDevice which is petsc intern
   // We have to make sure the device data is up to date before we do the axpy
   if (mat_local_ykok->a_dual.need_sync_device()) {
      mat_local_ykok->a_dual.sync_device();
      mat_local_ykok->transpose_updated = PETSC_FALSE; /* values of the transpose is out-of-date */
      mat_local_ykok->hermitian_updated = PETSC_FALSE;
    }  
    if (mat_nonlocal_ykok->a_dual.need_sync_device()) {
      mat_nonlocal_ykok->a_dual.sync_device();
      mat_nonlocal_ykok->transpose_updated = PETSC_FALSE; /* values of the transpose is out-of-date */
      mat_nonlocal_ykok->hermitian_updated = PETSC_FALSE;
    } 
    if (mat_local_xkok->a_dual.need_sync_device()) {
      mat_local_xkok->a_dual.sync_device();
      mat_local_xkok->transpose_updated = PETSC_FALSE; /* values of the transpose is out-of-date */
      mat_local_xkok->hermitian_updated = PETSC_FALSE;
    }           
    if (mat_nonlocal_xkok->a_dual.need_sync_device()) {
      mat_nonlocal_xkok->a_dual.sync_device();
      mat_nonlocal_xkok->transpose_updated = PETSC_FALSE; /* values of the transpose is out-of-date */
      mat_nonlocal_xkok->hermitian_updated = PETSC_FALSE;
    }  

   PetscInt rows_ao_y, cols_ao_y, rows_ao_x, cols_ao_x;

   MatGetSize(mat_nonlocal_y, &rows_ao_y, &cols_ao_y);
   MatGetSize(mat_nonlocal_x, &rows_ao_x, &cols_ao_x);
   
   // We also copy the colmaps over to the device as we need it
   PetscIntKokkosViewHost colmap_input_h_y = PetscIntKokkosViewHost(mat_mpi_y->garray, cols_ao_y);
   PetscIntKokkosView colmap_input_d_y = PetscIntKokkosView("colmap_input_d_y", cols_ao_y);
   Kokkos::deep_copy(colmap_input_d_y, colmap_input_h_y);  
   // Log copy with petsc
   size_t bytes = colmap_input_h_y.extent(0) * sizeof(PetscInt);
   PetscLogCpuToGpu(bytes);     

   PetscIntKokkosViewHost colmap_input_h_x = PetscIntKokkosViewHost(mat_mpi_x->garray, cols_ao_x);
   PetscIntKokkosView colmap_input_d_x = PetscIntKokkosView("colmap_input_d_x", cols_ao_x);
   Kokkos::deep_copy(colmap_input_d_x, colmap_input_h_x);  
   // Log copy with petsc
   bytes = colmap_input_h_x.extent(0) * sizeof(PetscInt);
   PetscLogCpuToGpu(bytes);  
   
   // Get the comm
   MPI_Comm MPI_COMM_MATRIX;
   PetscObjectGetComm((PetscObject)*Y, &MPI_COMM_MATRIX);
   PetscInt local_rows, local_cols, global_rows, global_cols;
   MatGetLocalSize(*Y, &local_rows, &local_cols);
   MatGetSize(*Y, &global_rows, &global_cols);   

   // ~~~~~~~~~~~~~~~
   // Let's go and add the local components together
   // ~~~~~~~~~~~~~~~

   Mat_SeqAIJKokkos *xkok_local, *ykok_local;
   ykok_local = static_cast<Mat_SeqAIJKokkos *>(mat_local_y->spptr);
   xkok_local = static_cast<Mat_SeqAIJKokkos *>(mat_local_x->spptr);   

   Kokkos::View<PetscScalar *> a_local_d_copy;
   Kokkos::View<PetscInt *> i_local_d_copy, j_local_d_copy;
   // Scope so the zcsr_local is destroyed once we copy 
   {
      KokkosCsrMatrix zcsr_local;
      KernelHandle    kh_local;      
      kh_local.create_spadd_handle(true); // X, Y are sorted

      KokkosSparse::spadd_symbolic(&kh_local, xkok_local->csrmat, ykok_local->csrmat, zcsr_local);
      KokkosSparse::spadd_numeric(&kh_local, alpha, xkok_local->csrmat, (PetscScalar)1.0, ykok_local->csrmat, zcsr_local);

      kh_local.destroy_spadd_handle();
      
      // Get the Kokkos Views from zcsr_local - annoyingly we can't just call MatCreateSeqAIJKokkosWithCSRMatrix
      // as it's petsc intern
      auto a_local_d_z = zcsr_local.values;
      auto i_local_d_z = zcsr_local.graph.row_map;
      auto j_local_d_z = zcsr_local.graph.entries;   

      a_local_d_copy = Kokkos::View<PetscScalar *>("a_local_d_copy", a_local_d_z.extent(0));
      i_local_d_copy = Kokkos::View<PetscInt *>("i_local_d_copy", i_local_d_z.extent(0));
      j_local_d_copy = Kokkos::View<PetscInt *>("j_local_d_copy", j_local_d_z.extent(0));   

      Kokkos::deep_copy(a_local_d_copy, a_local_d_z);
      Kokkos::deep_copy(i_local_d_copy, i_local_d_z);
      Kokkos::deep_copy(j_local_d_copy, j_local_d_z);
   }

   // We can create our local diagonal block matrix directly on the device
   Mat Z_local;
   MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, local_cols, i_local_d_copy, j_local_d_copy, a_local_d_copy, &Z_local);
   
   // ~~~~~~~~~~~~~~~
   // Now let's go and add the non-local components together
   // We first rewrite the j indices to be global as the nonlocal components of Y and X
   // might have different non-local non-zeros (and different numbers of non-local non-zeros)
   // ~~~~~~~~~~~~~~~

   // We need to duplicate the nonlocal part of x first as we are going to overwrite the 
   // column indices
   // Don't need to copy y as we destroy it anyway
   Mat mat_nonlocal_x_copy;
   MatDuplicate(mat_nonlocal_x, MAT_COPY_VALUES, &mat_nonlocal_x_copy);

   Mat_SeqAIJKokkos *xkok_nonlocal, *ykok_nonlocal; 
   ykok_nonlocal = static_cast<Mat_SeqAIJKokkos *>(mat_nonlocal_y->spptr);
   xkok_nonlocal = static_cast<Mat_SeqAIJKokkos *>(mat_nonlocal_x_copy->spptr);          

   PetscInt *device_nonlocal_x_j = xkok_nonlocal->j_device_data();
   PetscInt *device_nonlocal_y_j = ykok_nonlocal->j_device_data();

   // Rewrite the Y nonlocal indices to be global
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, ykok_nonlocal->csrmat.nnz()), KOKKOS_LAMBDA(PetscInt i) { 

         device_nonlocal_y_j[i] = colmap_input_d_y(device_nonlocal_y_j[i]);
   }); 

   // Rewrite the X nonlocal indices to be global
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, xkok_nonlocal->csrmat.nnz()), KOKKOS_LAMBDA(PetscInt i) { 

         device_nonlocal_x_j[i] = colmap_input_d_x(device_nonlocal_x_j[i]);
   });    

   // ~~~~~~~~~
   auto exec = PetscGetKokkosExecutionSpace();

   Kokkos::View<PetscScalar *> a_nonlocal_d_copy;
   Kokkos::View<PetscInt *> i_nonlocal_d_copy, j_nonlocal_d_copy;
   PetscInt *garray_host = NULL;
   PetscInt col_ao_output = 0;

   // Scope so the zcsr_nonlocal is destroyed once we copy 
   {
      // Now we can add the non-local components together
      KokkosCsrMatrix zcsr_nonlocal;
      // Global indices are sorted
      KernelHandle    kh_nonlocal;      
      kh_nonlocal.create_spadd_handle(true); 

      KokkosSparse::spadd_symbolic(&kh_nonlocal, xkok_nonlocal->csrmat, ykok_nonlocal->csrmat, zcsr_nonlocal);
      KokkosSparse::spadd_numeric(&kh_nonlocal, alpha, xkok_nonlocal->csrmat, (PetscScalar)1.0, ykok_nonlocal->csrmat, zcsr_nonlocal);

      kh_nonlocal.destroy_spadd_handle();

      // Can now destroy the copy
      MatDestroy(&mat_nonlocal_x_copy);

      // Get the Kokkos Views from zcsr_nonlocal - annoyingly we can't just call MatCreateSeqAIJKokkosWithCSRMatrix
      // as it's petsc intern
      auto a_nonlocal_d_z = zcsr_nonlocal.values;
      auto i_nonlocal_d_z = zcsr_nonlocal.graph.row_map;
      auto j_nonlocal_d_z = zcsr_nonlocal.graph.entries;

      // We know the most nonlocal indices we can have are the addition of x and y
      // (some might be the same)
      PetscInt cols_ao = cols_ao_x + cols_ao_y;

      // ~~~~~~~~~

      // Let's make sure everything on the device is finished
      exec.fence();   

      // Now we need to build garray on the host and rewrite the j_nonlocal_d_z indices so they are local
      rewrite_j_global_to_local(cols_ao, col_ao_output, j_nonlocal_d_z, &garray_host);  

      a_nonlocal_d_copy = Kokkos::View<PetscScalar *>("a_local_d_copy", a_nonlocal_d_z.extent(0));
      i_nonlocal_d_copy = Kokkos::View<PetscInt *>("i_local_d_copy", i_nonlocal_d_z.extent(0));
      j_nonlocal_d_copy = Kokkos::View<PetscInt *>("j_local_d_copy", j_nonlocal_d_z.extent(0));   

      Kokkos::deep_copy(a_nonlocal_d_copy, a_nonlocal_d_z);
      Kokkos::deep_copy(i_nonlocal_d_copy, i_nonlocal_d_z);
      Kokkos::deep_copy(j_nonlocal_d_copy, j_nonlocal_d_z);   
   }

   // We can create our nonlocal diagonal block matrix directly on the device
   Mat Z_nonlocal;
   MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, col_ao_output, i_nonlocal_d_copy, j_nonlocal_d_copy, a_nonlocal_d_copy, &Z_nonlocal);   

   // We can now create our MPI matrix
   Mat Z;
   MatCreateMPIAIJWithSeqAIJ(MPI_COMM_MATRIX, global_rows, global_cols, Z_local, Z_nonlocal, garray_host, &Z);    

   // Stick Z into the input Y (this destroys existing Y)
   MatHeaderReplace(*Y, &Z);

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Does a MatGetSubMatrix for a sequential Kokkos matrix - the petsc version currently uses the host making it very slow
// is_row_d_d and is_col_d_d must have the local indices in them
// is_col must be sorted
PETSC_INTERN void MatCreateSubMatrix_Seq_kokkos(Mat *input_mat, PetscIntKokkosView &is_row_d_d, PetscIntKokkosView &is_col_d_d, const int reuse_int, Mat *output_mat)
{
   PetscInt local_rows, local_cols;
   PetscInt nnzs_match_local;

   MatGetLocalSize(*input_mat, &local_rows, &local_cols); 
   PetscInt local_rows_row = is_row_d_d.extent(0), local_cols_col = is_col_d_d.extent(0);
   
   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr;
   MatSeqAIJGetCSRAndMemType(*input_mat, &device_local_i, &device_local_j, &device_local_vals, &mtype);  

   PetscIntKokkosView nnz_match_local_row_d;

   // Get device views
   Kokkos::View<PetscScalar *> a_local_d;
   Kokkos::View<PetscInt *> i_local_d;  
   Kokkos::View<PetscInt *> j_local_d;    

   // ~~~~~~~~~~~~
   // Get the number of nnzs
   // ~~~~~~~~~~~~
   nnzs_match_local = 0;

   // We need to know how many entries are in each row 
   nnz_match_local_row_d = PetscIntKokkosView("nnz_match_local_row_d", local_rows_row);    
   // We may have identity
   Kokkos::deep_copy(nnz_match_local_row_d, 0);              
   
   // Map which columns in the original mat are in is_col
   PetscIntKokkosView smap_d = PetscIntKokkosView("smap_d", local_cols);  
   Kokkos::deep_copy(smap_d, 0); 
   // Loop over all the cols in is_col
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_cols_col), KOKKOS_LAMBDA(PetscInt i) {      

         smap_d(is_col_d_d(i)) = i + 1; 
   });     
   
   // ~~~~~~~~~~~~
   // Need to count the number of nnzs we end up with, on each row and in total
   // ~~~~~~~~~~~~
   // Only loop over the number of rows in is_row
   if (!reuse_int)
   {
      Kokkos::parallel_reduce(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows_row, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t, PetscInt& thread_total) {

         const PetscInt i_idx_is_row = t.league_rank();
         const PetscInt i   = is_row_d_d(i_idx_is_row);
         const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];

         // nnz count for this row
         PetscInt row_result = 0;

         // Reduce over all the columns
         Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(t, ncols_local),
            [&](const PetscInt j, PetscInt& thread_data) {

               // Get this local column in the input_mat
               PetscInt target_col = device_local_j[device_local_i[i] + j];
               if (smap_d(target_col))
               {
                  thread_data++;
               }
            }, row_result
         );

         // We're finished our parallel reduction for this row
         // Only want one thread in the team to write the result
         Kokkos::single(Kokkos::PerTeam(t), [&]() {
            nnz_match_local_row_d(i_idx_is_row) = row_result;
            thread_total += row_result;
         });
         },
         nnzs_match_local
      );
   }

   // ~~~~~~~~~~~~  

   // Find maximum non-zeros over each of the is_row of the input mat for sizing scratch memory
   PetscInt max_nnz_local = 0;
   if (local_rows_row > 0) {

      Kokkos::parallel_reduce("FindMaxNNZ", local_rows_row,
         KOKKOS_LAMBDA(const PetscInt i_idx_is_row, PetscInt& thread_max) {
            // The indices in is_row will be global, but we want the local index
            const PetscInt i = is_row_d_d(i_idx_is_row);
            PetscInt row_nnz = device_local_i[i + 1] - device_local_i[i];
            thread_max = (row_nnz > thread_max) ? row_nnz : thread_max;
         },
         Kokkos::Max<PetscInt>(max_nnz_local)
      );
   }     

   auto exec = PetscGetKokkosExecutionSpace();

   // Create a team policy with scratch memory allocation
   // We want scratch space for each row
   // We will ncols+1 of integers which tell us what the matching indices we have
   // the last bit of memory is to account for 8-byte alignment for each view
   size_t scratch_size_per_team = (max_nnz_local+1) * sizeof(PetscInt) + \
               8 * 2 * sizeof(PetscScalar);

   Kokkos::TeamPolicy<> policy(exec, local_rows_row, Kokkos::AUTO());
   // We're gonna use the level 1 scratch as our column data is probably bigger than the level 0
   policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size_per_team));    

   // Only need things to do with the sparsity pattern if we're not reusing
   if (!reuse_int)
   {
      // Need to do a scan on nnz_match_local_row_d to get where each row starts
      Kokkos::parallel_scan (local_rows_row, KOKKOS_LAMBDA (const PetscInt i_idx_is_row, PetscInt& update, const bool final) {
         // Inclusive scan
         update += nnz_match_local_row_d(i_idx_is_row);         
         if (final) {
            nnz_match_local_row_d(i_idx_is_row) = update; // only update array on final pass
         }
      });

      // ~~~~~~~~~~~~~~~~~  
      // We need to assemble our i,j, vals so we can build our matrix
      // ~~~~~~~~~~~~~~~~~
      // Create memory on the device and host
      a_local_d = Kokkos::View<PetscScalar *>("a_local_d", nnzs_match_local);
      i_local_d = Kokkos::View<PetscInt *>("i_local_d", local_rows_row+1);
      j_local_d = Kokkos::View<PetscInt *>("j_local_d", nnzs_match_local);

      // Initialize first entry to zero - the rest get set below
      Kokkos::deep_copy(Kokkos::subview(i_local_d, 0), 0);       

      // ~~~~~~~~~~~~~~~
      // Create i indices
      // ~~~~~~~~~~~~~~~
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows_row), KOKKOS_LAMBDA(PetscInt i_idx_is_row) {      

            // The start of our row index comes from the scan
            i_local_d(i_idx_is_row + 1) = nnz_match_local_row_d(i_idx_is_row);   
      });    

      // Execute with scratch memory
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const KokkosTeamMemberType& t) {
         
         // i_idx_is_row is the row index into the output
         const PetscInt i_idx_is_row = t.league_rank();
         // i is the row index into the input
         const PetscInt i = is_row_d_d(i_idx_is_row);       

         // number of columns
         PetscInt ncols_local;
         ncols_local = device_local_i[i + 1] - device_local_i[i];
         ScratchIntView scratch_indices, scratch_indices_nonlocal;

         // Allocate views directly on scratch memory
         // Have to use views here given alignment issues
         // We have of size ncols+1 to account for the exclusive scan
         scratch_indices = ScratchIntView(t.team_scratch(1), ncols_local+1);  
         
         // Initialize scratch
         Kokkos::parallel_for(Kokkos::TeamVectorRange(t, ncols_local+1), [&](const PetscInt j) {
            scratch_indices(j) = 0;
         });

         // Team barrier to ensure all threads have finished filling the scratch space
         t.team_barrier();

         // Now go and mark which values we're keeping
         Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {
            // Mark this as a match
            if (smap_d(device_local_j[device_local_i[i] + j])){
               scratch_indices(j) = 1;
            }     
         });    

         // Team barrier to ensure all threads have finished filling the scratch space
         t.team_barrier(); 
         
         // Perform exclusive scan over scratch_indices to get our output indices in this row
         // Have to be careful to go up to ncols_local+1
         Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, ncols_local+1), 
            [&](const PetscInt j, int& partial_sum, const bool is_final) {
               const int input_value = scratch_indices(j);
               if (is_final) {
                     scratch_indices(j) = partial_sum;  // Write exclusive prefix
               }
               partial_sum += input_value;  // Update running total
         });

         // Team barrier to ensure all threads have finished scanning scratch_indices
         t.team_barrier();

         // Now go and write to the output
         Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {
            // We can tell if scratch_indices had 1 in it in this position by comparing the result
            // of the exclusive scan for this index and the next one
            if (scratch_indices(j+1) > scratch_indices(j))
            {
               // Be careful to use the correct i_idx_is_row index into i_local_d here
               j_local_d(i_local_d(i_idx_is_row) + scratch_indices(j)) = smap_d(device_local_j[device_local_i[i] + j]) - 1;
               a_local_d(i_local_d(i_idx_is_row) + scratch_indices(j)) = device_local_vals[device_local_i[i] + j];            
            }
         });
      });  
   }
   // If we're reusing, we can just write directly to the existing views
   else
   {
      Mat_SeqAIJKokkos *aijkok_local_output = static_cast<Mat_SeqAIJKokkos *>((*output_mat)->spptr);

      // Annoying we can't just call MatSeqAIJGetKokkosView
      a_local_d = aijkok_local_output->a_dual.view_device();

      // Annoyingly there isn't currently the ability to get views for i (or j)
      const PetscInt *device_local_i_output = nullptr;
      PetscMemType mtype;
      MatSeqAIJGetCSRAndMemType(*output_mat, &device_local_i_output, NULL, NULL, &mtype);  

      // Have these point at the existing i pointers
      ConstMatRowMapKokkosView i_local_const_d = ConstMatRowMapKokkosView(device_local_i_output, local_rows_row+1);     

      // Execute with scratch memory
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const KokkosTeamMemberType& t) {
            
         // i_idx_is_row is the row index into the output
         const PetscInt i_idx_is_row = t.league_rank();
         // i is the row index into the input
         const PetscInt i = is_row_d_d(i_idx_is_row);     

         // number of columns
         PetscInt ncols_local;
         ncols_local = device_local_i[i + 1] - device_local_i[i];
         ScratchIntView scratch_indices, scratch_indices_nonlocal;

         // Allocate views directly on scratch memory
         // Have to use views here given alignment issues
         // We have of size ncols+1 to account for the exclusive scan
         scratch_indices = ScratchIntView(t.team_scratch(1), ncols_local+1);  
         
         // Initialize scratch
         Kokkos::parallel_for(Kokkos::TeamVectorRange(t, ncols_local+1), [&](const PetscInt j) {
            scratch_indices(j) = 0;
         });

         // Team barrier to ensure all threads have finished filling the scratch space
         t.team_barrier();

         // Now go and mark which values we're keeping
         Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {
            // Mark this as a match
            if (smap_d(device_local_j[device_local_i[i] + j])){
               scratch_indices(j) = 1;
            }     
         });    

         // Team barrier to ensure all threads have finished filling the scratch space
         t.team_barrier(); 
         
         // Perform exclusive scan over scratch_indices to get our output indices in this row
         // Have to be careful to go up to ncols_local+1
         Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, ncols_local+1), 
            [&](const PetscInt j, int& partial_sum, const bool is_final) {
               const int input_value = scratch_indices(j);
               if (is_final) {
                     scratch_indices(j) = partial_sum;  // Write exclusive prefix
               }
               partial_sum += input_value;  // Update running total
         });

         // Team barrier to ensure all threads have finished scanning scratch_indices
         t.team_barrier();

         // Now go and write to the output
         Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {
            // We can tell if scratch_indices had 1 in it in this position by comparing the result
            // of the exclusive scan for this index and the next one
            if (scratch_indices(j+1) > scratch_indices(j))
            {
               // Be careful to use the correct i_idx_is_row index into i_local_const_d here
               a_local_d(i_local_const_d(i_idx_is_row) + scratch_indices(j)) = device_local_vals[device_local_i[i] + j];            
            }
         });
      });        

      // Have to specify we've modifed data on the device
      // Want to call MatSeqAIJKokkosModifyDevice but its PETSC_INTERN
      aijkok_local_output->a_dual.clear_sync_state();
      aijkok_local_output->a_dual.modify_device();
      aijkok_local_output->transpose_updated = PETSC_FALSE;
      aijkok_local_output->hermitian_updated = PETSC_FALSE;
      // Invalidate diagonals
      Mat_SeqAIJ *a = (Mat_SeqAIJ *)(*output_mat)->data;
      a->idiagvalid  = PETSC_FALSE;
      a->ibdiagvalid = PETSC_FALSE;      
      a->inode.ibdiagvalid = PETSC_FALSE;      
      PetscObjectStateIncrease((PetscObject)(*output_mat));    

   }

   // Let's make sure everything on the device is finished
   exec.fence();   

   // ~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~   

   if (!reuse_int)
   {   
      // Create the matrix given the sorted csr
      MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows_row, local_cols_col, i_local_d, j_local_d, a_local_d, output_mat);                
   }  

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Does a MatGetSubMatrix for a Kokkos matrix - the petsc version currently uses the host making it very slow
// This version only works  works if the input IS have the same parallel row/column distribution 
// as the matrices, ie equivalent to MatCreateSubMatrix_MPIAIJ_SameRowDist
// is_col must be sorted
// This one uses the views is_row_d_d and is_col_d_d directly, rewritten to be the local indices
PETSC_INTERN void MatCreateSubMatrix_kokkos_view(Mat *input_mat, PetscIntKokkosView &is_row_d_d, PetscInt global_rows_row, \
         PetscIntKokkosView &is_col_d_d, PetscInt global_cols_col, const int reuse_int, Mat *output_mat)
{
   PetscInt local_rows, local_cols;
   PetscInt global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one;
   MatGetOwnershipRange(*input_mat, &global_row_start, &global_row_end_plus_one);
   PetscInt local_cols_col = is_col_d_d.extent(0);

   // Are we in parallel?
   MatType mat_type;
   MPI_Comm MPI_COMM_MATRIX;
   MatGetType(*input_mat, &mat_type);

   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;   
   PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX);
   MatGetSize(*input_mat, &global_rows, &global_cols); 
   MatGetLocalSize(*input_mat, &local_rows, &local_cols); 
   
   Mat_MPIAIJ *mat_mpi = nullptr;
   Mat mat_local = NULL, mat_nonlocal = NULL;   
   Mat output_mat_local, output_mat_nonlocal;
  
   PetscInt rows_ao, cols_ao;
   if (mpi)
   {
      mat_mpi = (Mat_MPIAIJ *)(*input_mat)->data;
      mat_local = mat_mpi->A;
      mat_nonlocal = mat_mpi->B;
      MatGetSize(mat_nonlocal, &rows_ao, &cols_ao); 
      
      if (reuse_int)
      {
         Mat_MPIAIJ *mat_mpi_output = (Mat_MPIAIJ *)(*output_mat)->data;
         output_mat_local = mat_mpi_output->A;
         output_mat_nonlocal = mat_mpi_output->B;
      }
   }
   else
   {
      mat_local = *input_mat;
      if (reuse_int) output_mat_local = *output_mat;
   }
   size_t bytes = 0;

   // The diagonal component
   MatCreateSubMatrix_Seq_kokkos(&mat_local, is_row_d_d, is_col_d_d, reuse_int, &output_mat_local);

   // The off-diagonal component requires some comms
   // Basically a copy of MatCreateSubMatrix_MPIAIJ_SameRowColDist
   if (mpi)
   {
      PetscIntKokkosView is_col_o_d, garray_output_d;

      if (!reuse_int)
      {
         PetscInt isstart = 0;
         /* Get start indices on each rank for the new columns */
         MPI_Scan(&local_cols_col, &isstart, 1, MPIU_INT, MPI_SUM, MPI_COMM_MATRIX);
         isstart -= local_cols_col;

         // Basically a copy of ISGetSeqIS_SameColDist_Private
         /* (1) iscol is a sub-column vector of mat, pad it with '-1.' to form a full vector x */
         Vec x, cmap, lcmap;
         Vec lvec = mat_mpi->lvec;
         MatCreateVecs(*input_mat, &x, NULL);
         VecSet(x, -1.0);
         VecDuplicate(x, &cmap);
         VecSet(cmap, -1.0);

         // Use the vecs in the scatter provided by the input mat
         PetscScalarKokkosView x_d;
         VecGetKokkosView(x, &x_d);
         PetscScalarKokkosView cmap_d;
         VecGetKokkosView(cmap, &cmap_d);
         PetscScalarKokkosView lvec_d;
         VecGetKokkosView(lvec, &lvec_d);

         // Loop over all the cols in is_col
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, local_cols_col), KOKKOS_LAMBDA(PetscInt i) {      

               x_d(is_col_d_d(i)) = (PetscScalar)is_col_d_d(i); 
               cmap_d(is_col_d_d(i)) = i + isstart; /* global index of iscol[i] */
         });

         PetscScalar *x_d_ptr = NULL;
         x_d_ptr = x_d.data();      
         PetscScalar *cmap_d_ptr = NULL;
         cmap_d_ptr = cmap_d.data();
         PetscScalar *lvec_d_ptr = NULL;
         lvec_d_ptr = lvec_d.data();       

         // Start the scatter of the x - the kokkos memtype is set as PETSC_MEMTYPE_HOST or 
         // one of the kokkos backends like PETSC_MEMTYPE_HIP
         PetscMemType mem_type = PETSC_MEMTYPE_KOKKOS;      
         PetscSFBcastWithMemTypeBegin(mat_mpi->Mvctx, MPIU_SCALAR,
                     mem_type, x_d_ptr,
                     mem_type, lvec_d_ptr,
                     MPI_REPLACE);      
         
         VecDuplicate(lvec, &lcmap);
         PetscScalarKokkosView lcmap_d;
         VecGetKokkosView(lcmap, &lcmap_d);
         PetscScalar *lcmap_d_ptr = NULL;
         lcmap_d_ptr = lcmap_d.data();

         // Start the cmap scatter
         PetscSFBcastWithMemTypeBegin(mat_mpi->Mvctx, MPIU_SCALAR,
                     mem_type, cmap_d_ptr,
                     mem_type, lcmap_d_ptr,
                     MPI_REPLACE);      

         // Finish the x scatter
         PetscSFBcastEnd(mat_mpi->Mvctx, MPIU_SCALAR, x_d_ptr, lvec_d_ptr, MPI_REPLACE);      
         // We're done with x now
         VecRestoreKokkosView(x, &x_d);
         VecDestroy(&x);         

         // Let's count how many off-local columns we have
         PetscInt col_ao_output = 0;

         // One bigger for exclusive scan
         auto is_col_o_match_d = PetscIntKokkosView("is_col_o_match_d", cols_ao+1);
         Kokkos::deep_copy(is_col_o_match_d, 0);
         if (cols_ao > 0) 
         {
            Kokkos::parallel_reduce("FindMatches", cols_ao,
               KOKKOS_LAMBDA(const PetscInt i, PetscInt& thread_sum) {
                  // This is the scattered x for all of the non-local columns in the input mat
                  // It's not -1 if that column is present on another rank
                  if (lvec_d_ptr[i] > -1.0) {
                     thread_sum++;
                     is_col_o_match_d(i) = 1; // Mark this as a match
                  }
               },
               Kokkos::Sum<PetscInt>(col_ao_output)
            ); 
         }    

         VecRestoreKokkosView(lvec, &lvec_d);

         // Need to do an exclusive scan on is_col_o_match_d to get the new local indices
         // Have to remember to go up to cols_ao+1
         Kokkos::parallel_scan (cols_ao+1, KOKKOS_LAMBDA (const PetscInt i, PetscInt& partial_sum, const bool is_final) {
               const int input_value = is_col_o_match_d(i);
               if (is_final) {
                  is_col_o_match_d(i) = partial_sum;  // Write exclusive prefix
               }
               partial_sum += input_value;  // Update running total
         }); 

         // Local indices into input garray of the columns we want to keep
         // but remember this doesn't mean garray_output = garray_input(is_col_o_d)
         // as the of columns we have in the output has changed, ie we need 
         // the cmap_d given it has isstart 
         is_col_o_d = PetscIntKokkosView("is_col_o_d", col_ao_output);
         garray_output_d = PetscIntKokkosView("garray_output_d", col_ao_output);

         // Finish the cmap scatter
         PetscSFBcastEnd(mat_mpi->Mvctx, MPIU_SCALAR, cmap_d_ptr, lcmap_d_ptr, MPI_REPLACE);         

         // Loop over all the cols in the input matrix
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, cols_ao), KOKKOS_LAMBDA(PetscInt i) {     
               
               // We can tell if is_col_o_match_d had 1 in it in this position by comparing the result
               // of the exclusive scan for this index and the next one            
               if (is_col_o_match_d(i+1) > is_col_o_match_d(i))
               {
                  is_col_o_d(is_col_o_match_d(i)) = i;
                  garray_output_d(is_col_o_match_d(i)) = (PetscInt)lcmap_d_ptr[i];
               }
         });      

         VecRestoreKokkosView(cmap, &cmap_d);
         VecRestoreKokkosView(lcmap, &lcmap_d);

         VecDestroy(&cmap);
         VecDestroy(&lcmap);         
      }
      // If we're reusing we have the iscol_o associated with the output_mat
      else
      {
         // Get the iscol_o from the output_mat
         IS iscol_o;
         /* Retrieve isrow_d, iscol_d and iscol_o from output */
         PetscObjectQuery((PetscObject)(*output_mat), "iscol_o", (PetscObject *)&iscol_o);
         //PetscCheck(iscol_o, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "iscol_o passed in was not used before, cannot reuse");

         const PetscInt *iscol_o_indices_ptr;
         ISGetIndices(iscol_o, &iscol_o_indices_ptr);

         PetscInt local_cols_iscol_o;
         ISGetLocalSize(iscol_o, &local_cols_iscol_o);

         // Copy the iscol_o to the device
         auto iscol_o_view_h = PetscIntConstKokkosViewHost(iscol_o_indices_ptr, local_cols_iscol_o);    
         is_col_o_d = PetscIntKokkosView("is_col_o_d", local_cols_iscol_o);   
         Kokkos::deep_copy(is_col_o_d, iscol_o_view_h);
         // Log copy with petsc
         bytes = iscol_o_view_h.extent(0) * sizeof(PetscInt);
         PetscLogCpuToGpu(bytes);

         ISRestoreIndices(iscol_o, &iscol_o_indices_ptr);
      }

      // We can now create the off-diagonal component
      MatCreateSubMatrix_Seq_kokkos(&mat_nonlocal, is_row_d_d, is_col_o_d, reuse_int, &output_mat_nonlocal);

      // If it's our first time through we have to create our output matrix
      if (!reuse_int)
      {
         // Copy the garray output to the host
         PetscInt *garray_host = NULL; 
         PetscMalloc1(garray_output_d.extent(0), &garray_host);
         PetscIntKokkosViewHost colmap_output_h = PetscIntKokkosViewHost(garray_host, garray_output_d.extent(0));
         // Copy the garray output to the host
         Kokkos::deep_copy(colmap_output_h, garray_output_d);
         bytes = colmap_output_h.extent(0) * sizeof(PetscInt);
         PetscLogGpuToCpu(bytes);       
         
         // We can now create our MPI matrix
         MatCreateMPIAIJWithSeqAIJ(MPI_COMM_MATRIX, global_rows_row, global_cols_col, output_mat_local, output_mat_nonlocal, garray_host, output_mat);

         // ~~~~~~~~~~~~~~
         // If this is the first time through, we need to store the iscol_o in the output_mat
         // We don't store the is_row_d_d or is_col_d_d like the host version does as they're super cheap to rebuild
         // ~~~~~~~~~~~~~~
         // Copy the is_col_o_d to the host
         PetscInt *is_col_o_host = NULL; 
         PetscMalloc1(is_col_o_d.extent(0), &is_col_o_host);
         PetscIntKokkosViewHost is_col_o_h = PetscIntKokkosViewHost(is_col_o_host, is_col_o_d.extent(0));
         // Copy the is_col_o_d output to the host
         Kokkos::deep_copy(is_col_o_h, is_col_o_d);
         bytes = is_col_o_h.extent(0) * sizeof(PetscInt);
         PetscLogGpuToCpu(bytes);   
         // Now create an IS
         IS iscol_o;
         ISCreateGeneral(PETSC_COMM_SELF, is_col_o_h.extent(0), is_col_o_host, PETSC_OWN_POINTER, &iscol_o);      
         // Register it with the output_mat
         PetscObjectCompose((PetscObject)(*output_mat), "iscol_o", (PetscObject)iscol_o);
         // The ref counter is incremented by the compose
         ISDestroy(&iscol_o);
      }
   }
   else
   {
      *output_mat = output_mat_local;
   }

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Does a MatGetSubMatrix for a Kokkos matrix - the petsc version currently uses the host making it very slow
// This version only works  works if the input IS have the same parallel row/column distribution 
// as the matrices, ie equivalent to MatCreateSubMatrix_MPIAIJ_SameRowDist
// is_col must be sorted
// If you pass in our_level != -1 then it uses the fine/coarse indices stored in IS_fine_views_local
// and IS_coarse_views_local - they should match the passed in is_row and is_col though!
PETSC_INTERN void MatCreateSubMatrix_kokkos(Mat *input_mat, IS *is_row, IS *is_col, \
                     const int reuse_int, Mat *output_mat, \
                     const int our_level, const int is_row_fine_int, const int is_col_fine_int)
{

   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt global_col_start, global_col_end_plus_one;   
   MatGetOwnershipRange(*input_mat, &global_row_start, &global_row_end_plus_one);  
   MatGetOwnershipRangeColumn(*input_mat, &global_col_start, &global_col_end_plus_one); 
   PetscInt global_rows_row, global_cols_col;
   ISGetSize(*is_row, &global_rows_row);
   ISGetSize(*is_col, &global_cols_col);    
   
   PetscIntKokkosView is_row_d_d, is_col_d_d;

   // If we want the input is_row and is_col to be used
   if (our_level == -1)
   {
      // Get pointers to the indices on the host
      const PetscInt *is_row_indices_ptr, *is_col_indices_ptr;
      ISGetIndices(*is_row, &is_row_indices_ptr);   
      ISGetIndices(*is_col, &is_col_indices_ptr); 

      PetscInt local_rows_row, local_cols_col;
      ISGetLocalSize(*is_row, &local_rows_row);   
      ISGetLocalSize(*is_col, &local_cols_col);

      // Create a host view of the existing indices
      auto is_row_view_h = PetscIntConstKokkosViewHost(is_row_indices_ptr, local_rows_row);    
      is_row_d_d = PetscIntKokkosView("is_row_d_d", local_rows_row);   
      auto is_col_view_h = PetscIntConstKokkosViewHost(is_col_indices_ptr, local_cols_col);    
      is_col_d_d = PetscIntKokkosView("is_col_d_d", local_cols_col);      
      // Copy indices to the device
      Kokkos::deep_copy(is_row_d_d, is_row_view_h);     
      Kokkos::deep_copy(is_col_d_d, is_col_view_h);
      // Log copy with petsc
      size_t bytes = is_row_view_h.extent(0) * sizeof(PetscInt);
      PetscLogCpuToGpu(bytes);        
      bytes = is_col_view_h.extent(0) * sizeof(PetscInt);
      PetscLogCpuToGpu(bytes);  

      ISRestoreIndices(*is_row, &is_row_indices_ptr);   
      ISRestoreIndices(*is_col, &is_col_indices_ptr);   

      // ~~~~~~~~~~~~
      // Rewrite to local indices
      // ~~~~~~~~~~~~     
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, is_row_d_d.extent(0)), KOKKOS_LAMBDA(PetscInt i) {      

            is_row_d_d(i) -= global_row_start; // Make local
      });

      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, is_col_d_d.extent(0)), KOKKOS_LAMBDA(PetscInt i) {

            is_col_d_d(i) -= global_col_start; // Make local
      }); 
   }
   // Instead if we tell the routine that the is_row and is_col are fine/coarse local indices
   // that already are on the device
   else
   {
      if (is_row_fine_int)
      {
         is_row_d_d = *IS_fine_views_local[our_level];
      }
      else
      {
         is_row_d_d = *IS_coarse_views_local[our_level];
      }       
      if (is_col_fine_int)
      {
         is_col_d_d = *IS_fine_views_local[our_level];
      }
      else
      {
         is_col_d_d = *IS_coarse_views_local[our_level];
      }        
   }  

   MatCreateSubMatrix_kokkos_view(input_mat, is_row_d_d, global_rows_row, is_col_d_d, global_cols_col, reuse_int, output_mat);

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Parallel merge function
template <typename ViewType>
void parallel_merge(const ViewType& array1, const ViewType& array2, ViewType& output_array, ViewType& permutation_vector) {

    // Sizes of the input arrays
    const size_t size1 = array1.extent(0);
    const size_t size2 = array2.extent(0);

    // Resize the output array
    output_array = ViewType("output_array", size1 + size2);
    permutation_vector = ViewType("permutation_vector", size1 + size2);

    // Team policy for parallel merge
    const size_t total_size = size1 + size2;
    Kokkos::TeamPolicy<> policy(total_size / 256 + 1, Kokkos::AUTO());

    // Each team will handle a chunk of the output array
    // The chunk size is determined by the number of teams
    // and the total size of the output array
    // Each team will assign corresponding ranges in array1 and array2
    // and then merge the assigned ranges into the output array
    // Each team will handle a chunk of the output array
    Kokkos::parallel_for("ParallelMerge", policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        const size_t team_rank = team.league_rank();
        const size_t team_size = team.league_size();

        // Divide the output array among teams
        const size_t chunk_size = (total_size + team_size - 1) / team_size;
        const size_t start = team_rank * chunk_size;
        const size_t end = Kokkos::min(start + chunk_size, total_size);

        // Find the corresponding ranges in array1 and array2
        size_t start1 = Kokkos::min(size1, start);
        size_t start2 = start - start1;
        if (start2 > size2) {
            start2 = size2;
            start1 = start - start2;
        }

        size_t end1 = Kokkos::min(size1, end);
        size_t end2 = end - end1;
        if (end2 > size2) {
            end2 = size2;
            end1 = end - end2;
        }

        // Merge the assigned ranges
        size_t i = start1, j = start2, k = start;
        while (i < end1 && j < end2) {
            if (array1(i) <= array2(j)) {
                output_array(k) = array1(i);
                permutation_vector(k++) = i; // Store the index from array1
                i++;
            } else {
                output_array(k) = array2(j);
                permutation_vector(k++) = size1 + j; // Store the index from array2
                j++;
            }
        }
        while (i < end1) {
            output_array(k) = array1(i);
            permutation_vector(k++) = i; // Store the index from array1
            i++;
        }
        while (j < end2) {
            output_array(k) = array2(j);
            permutation_vector(k++) = size1 + j; // Store the index from array2
            j++;
        }
   });
}

//------------------------------------------------------------------------------------------------------------------------

// Merges sorted views together into one view
// Won't necessarily be very efficient if we have many input_arrays and they are small
// as the only parallelism is coming from the inside parallel_merge
template <typename ViewType>
void parallel_merge_tree(const std::vector<ViewType>& input_arrays, ViewType& output_array, ViewType& permutation_vector) {
    // If there are no input arrays, return an empty output array and permutation vector
    if (input_arrays.empty()) {
        output_array = ViewType("output_array", 0);
        permutation_vector = ViewType("permutation_vector", 0);
        return;
    }

    // If there is only one input array, copy it directly to the output
    if (input_arrays.size() == 1) {
        output_array = ViewType("output_array", input_arrays[0].extent(0));
        permutation_vector = ViewType("permutation_vector", input_arrays[0].extent(0));
        Kokkos::parallel_for("CopyPermutation", input_arrays[0].extent(0), KOKKOS_LAMBDA(const size_t i) {
            output_array(i) = input_arrays[0](i);
            permutation_vector(i) = i; // Identity permutation
        });
        return;
    }

    // Create a working vector to hold intermediate results
    std::vector<ViewType> current_level = input_arrays;
    std::vector<ViewType> current_permutations;
    std::vector<size_t> current_offsets;

    // Initialize permutation vectors and offsets for the input arrays
    size_t cumulative_offset = 0;
    for (const auto& array : input_arrays) {
        ViewType perm("perm", array.extent(0));
        Kokkos::parallel_for("InitPermutation", array.extent(0), KOKKOS_LAMBDA(const size_t i) {
            perm(i) = i; // Identity permutation for each input array
        });
        current_permutations.push_back(perm);
        current_offsets.push_back(cumulative_offset);
        cumulative_offset += array.extent(0);
    }

    // Perform the merge in a tree fashion
    while (current_level.size() > 1) {
        std::vector<ViewType> next_level;
        std::vector<ViewType> next_permutations;
        std::vector<size_t> next_offsets;

        // Merge pairs of arrays
        for (size_t i = 0; i < current_level.size(); i += 2) {
            if (i + 1 < current_level.size()) {
                // Merge two arrays
                ViewType merged_array;
                ViewType merged_permutation;
                parallel_merge(current_level[i], current_level[i + 1], merged_array, merged_permutation);

                // Map the merged permutation back to the original indices
                ViewType combined_permutation("combined_permutation", merged_permutation.extent(0));
                size_t offset1 = current_offsets[i];
                size_t offset2 = current_offsets[i + 1];
                
                Kokkos::parallel_for("CombinePermutation", merged_permutation.extent(0), KOKKOS_LAMBDA(const size_t j) {
                    if (static_cast<size_t>(merged_permutation(j)) < current_level[i].extent(0)) {
                        // Index from first array: add offset1 to the permutation from first array
                        combined_permutation(j) = current_permutations[i](merged_permutation(j)) + offset1;
                    } else {
                        // Index from second array: add offset2 to the permutation from second array
                        combined_permutation(j) = current_permutations[i + 1](merged_permutation(j) - current_level[i].extent(0)) + offset2;
                    }
                });

                next_level.push_back(merged_array);
                next_permutations.push_back(combined_permutation);
                next_offsets.push_back(offset1); // The merged array starts at the same offset as the first array
            } else {
                // If there's an odd array, move it to the next level as is
                next_level.push_back(current_level[i]);
                next_permutations.push_back(current_permutations[i]);
                next_offsets.push_back(current_offsets[i]);
            }
        }

        // Move to the next level
        current_level = std::move(next_level);
        current_permutations = std::move(next_permutations);
        current_offsets = std::move(next_offsets);
    }

    // The final merged array and permutation vector are the only ones left in the current level
    output_array = current_level[0];
    permutation_vector = current_permutations[0];
}

//------------------------------------------------------------------------------------------------------------------------

// Does a MatTranspose for a MPIAIJ Kokkos matrix - the petsc version currently uses the host making it slow
// Parts of this are taken from MatTranspose_MPIAIJ
PETSC_INTERN void MatTranspose_kokkos(Mat *X, Mat *Y, const int symbolic_int)
{
   Mat_MPIAIJ *mat_mpi_x = (Mat_MPIAIJ *)(*X)->data;
   Mat mat_local_x = mat_mpi_x->A;
   Mat mat_nonlocal_x = mat_mpi_x->B;

   Mat_SeqAIJKokkos *mat_local_xkok = static_cast<Mat_SeqAIJKokkos *>(mat_local_x->spptr);
   Mat_SeqAIJKokkos *mat_nonlocal_xkok = static_cast<Mat_SeqAIJKokkos *>(mat_nonlocal_x->spptr);

   // Equivalent to calling MatSeqAIJKokkosSyncDevice which is petsc intern
   // We have to make sure the device data is up to date before we do the transpose
   if (!symbolic_int)
   {
      if (mat_local_xkok->a_dual.need_sync_device()) {
      mat_local_xkok->a_dual.sync_device();
      mat_local_xkok->transpose_updated = PETSC_FALSE; /* values of the transpose is out-of-date */
      mat_local_xkok->hermitian_updated = PETSC_FALSE;
      }           
      if (mat_nonlocal_xkok->a_dual.need_sync_device()) {
      mat_nonlocal_xkok->a_dual.sync_device();
      mat_nonlocal_xkok->transpose_updated = PETSC_FALSE; /* values of the transpose is out-of-date */
      mat_nonlocal_xkok->hermitian_updated = PETSC_FALSE;
      }  
   }

   // ~~~~~~~~~~~~~~
   // Get the comm   
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscObjectGetComm((PetscObject)*X, &MPI_COMM_MATRIX);
   MatGetLocalSize(*X, &local_rows, &local_cols);
   MatGetSize(*X, &global_rows, &global_cols);   
   PetscInt cols_ao = mat_nonlocal_x->cmap->n;

   // This returns the global index of the local portion of the matrix
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt global_col_start, global_col_end_plus_one;
   MatGetOwnershipRange(*X, &global_row_start, &global_row_end_plus_one);
   MatGetOwnershipRangeColumn(*X, &global_col_start, &global_col_end_plus_one);

   // We also copy the input mat colmap over to the device as we need it
   PetscIntKokkosViewHost colmap_input_h = PetscIntKokkosViewHost(mat_mpi_x->garray, cols_ao);
   PetscIntKokkosView colmap_input_d = PetscIntKokkosView("colmap_input_d", cols_ao);
   Kokkos::deep_copy(colmap_input_d, colmap_input_h);      
   // Log copy with petsc
   size_t bytes = colmap_input_h.extent(0) * sizeof(PetscInt);
   PetscLogCpuToGpu(bytes);    

   // ~~~~~~~~~~~~~~
   // Let's do some comms to work out how many entries are coming/going
   // We are going to use an sf based on garray
   // The sf would normally be used for a matvec
   // ~~~~~~~~~~~~~~   

   // MatView(mat_nonlocal_x, PETSC_VIEWER_STDOUT_SELF);
   // for (PetscInt i = 0; i < cols_ao; i++) {
   //    fprintf(stderr,"garray[%d] = %d\n", i, mat_mpi_x->garray[i]);
   // }

   // Annoyingly there isn't currently the ability to get views for i (or j)
   const PetscInt *device_i_nonlocal = nullptr, *device_j_nonlocal = nullptr;
   PetscMemType mtype;
   MatSeqAIJGetCSRAndMemType(mat_nonlocal_x, &device_i_nonlocal, &device_j_nonlocal, NULL, &mtype);
   Kokkos::View<PetscInt *> g_nnz_d = PetscIntKokkosView("g_nnz_d", cols_ao);
   Kokkos::deep_copy(g_nnz_d, 0);

   // Work out how many entries in our non-local matrix per non-local column
   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

         // Row
         const PetscInt i = t.league_rank();

         // Still using i here (the local index into input)
         const PetscInt ncols = device_i_nonlocal[i + 1] - device_i_nonlocal[i];         

         // For over local columns
         Kokkos::parallel_for(
            Kokkos::TeamThreadRange(t, ncols), [&](const PetscInt j) {

            // Has to be atomic! Potentially lots of contention so maybe not 
            // the most performant way to do this
            Kokkos::atomic_add(&g_nnz_d(device_j_nonlocal[device_i_nonlocal[i] + j]), 1);    
         });  
   });   

   // We also need a copy of g_nnz on the host
   PetscInt *g_nnz;
   PetscMalloc1(cols_ao, &g_nnz);
   PetscIntKokkosViewHost g_nnz_h = PetscIntKokkosViewHost(g_nnz, cols_ao);
   // Copy g_nnz to the host
   Kokkos::deep_copy(g_nnz_h, g_nnz_d); 
   // Log copy with petsc
   bytes = g_nnz_h.extent(0) * sizeof(PetscInt);
   PetscLogGpuToCpu(bytes);        

   PetscSF sf;   
   PetscSFCreate(MPI_COMM_MATRIX, &sf);
   PetscSFSetGraphLayout(sf, (*X)->cmap, cols_ao, NULL, PETSC_USE_POINTER, mat_mpi_x->garray);

   PetscMemType mem_type = PETSC_MEMTYPE_KOKKOS;
   PetscInt *g_nnz_d_ptr = g_nnz_d.data();
   Kokkos::View<PetscInt *> i_transpose_d = PetscIntKokkosView("i_transpose_d", local_rows+1);
   PetscInt *i_transpose_d_ptr = i_transpose_d.data();

   // After this reduction, i_transpose_d will have how many entries we have in each row of the resulting
   // non-local portion of the transpose
   // A scan of this will tell us the i_indices   
   PetscSFReduceWithMemTypeBegin(sf, MPIU_INT,
      mem_type, g_nnz_d_ptr,
      mem_type, i_transpose_d_ptr,
      MPIU_SUM);   
   PetscSFReduceEnd(sf, MPIU_INT, g_nnz_d_ptr, i_transpose_d_ptr, MPIU_SUM);         

   // for (PetscInt i = 0; i < local_rows; i++) {
   //    fprintf(stderr,"i_transpose_d[%d] = %d\n", i, i_transpose_d[i]);
   // }      

   // Perform exclusive scan - this modifies i_transpose_d in-place
   Kokkos::parallel_scan(local_rows + 1, KOKKOS_LAMBDA(const PetscInt i, PetscInt& update, const bool final) {
      const PetscInt input_value = i_transpose_d(i);
      if (final) {
         i_transpose_d(i) = update; // Write exclusive prefix
      }
      update += input_value; // Update running total
   }); 

   // for (PetscInt i = 0; i < local_rows+1; i++) {
   //    fprintf(stderr,"i_transpose_d[%d] = %d\n", i, i_transpose_d(i));
   // }   

   const PetscInt    *roffset, *rmine;
   const PetscMPIInt *ranks;
   PetscMPIInt        rank, nranks, size;   
   MPI_Comm_rank(MPI_COMM_MATRIX, &rank);
   MPI_Comm_size(MPI_COMM_MATRIX, &size);

   // ~~~~~~~~~~~~~~
   // We can query the sf to get some information we need
   // The information below is talking about what we would receive during a matvec
   // ~~~~~~~~~~~~~~ 
   // nranks is how many ranks we receive from
   // roffset[rank+1] - roffset[rank] is how many items we expect to receive from this rank
   // if we have an array of length garray to receive into, rmine are the indices into that array for each rank
   //  ie garray[rmine] gives the non-local cols that we have just received
   // rremote are the local column indices on the remote ranks that we have received
   //  if we call MatGetOwnershipRangesColumns(*X, &ranges), then
   //  ie ranges[sender] + rremote gives the global column indices that we have just received, 
   //  ie garray[rmine] = ranges[sender] + rremote
   PetscSFGetRootRanks(sf, &nranks, &ranks, &roffset, &rmine, NULL);

   // We can use this information to tell us how many things we have to send during a transpose
   // As we now know where each non-local column is going, and we know how many of each non-local column
   // we have in g_nnz
   PetscInt *send_rank_no_vals;
   PetscMalloc1(nranks, &send_rank_no_vals);
   PetscArrayzero(send_rank_no_vals, nranks);

   for (int i = 0; i < nranks; i++) {
      //PetscMPIInt sender = ranks[i];
      PetscInt start = roffset[i];
      PetscInt end   = roffset[i+1];
      PetscInt nitems = end - start;

      // fprintf(stderr, "[%d] expecting %d items from rank %d\n",
      //             rank, nitems, sender);

      for (PetscInt j = 0; j < nitems; j++) {
         // fprintf(stderr, "[%d] expecting local index %d local remote index %d, remote index %d from rank %d\n",
         //             rank, rmine[start + j], rremote[start+j], mat_mpi_x->garray[rmine[start + j]], sender);

         // During the transpose we have to send this many things to rank sender
         send_rank_no_vals[i] += g_nnz[rmine[start + j]];                     
      }
   }

   // for (int i = 0; i < nranks; i++) {
   //    PetscMPIInt sender = ranks[i];
   //    fprintf(stderr, "[%d] in the transpose we are sending %d items to rank %d\n",
   //                rank, send_rank_no_vals[i], sender);
   // }

   // ~~~~~~~~~~~~~~
   // Let's start all our sends
   // ~~~~~~~~~~~~~~ 
   PetscInt no_send_entries = 0;
   PetscInt *send_rank_no_vals_scan;
   PetscMalloc1(nranks+1, &send_rank_no_vals_scan);
   send_rank_no_vals_scan[0] = 0;
   for (int i = 0; i < nranks; i++) {
      send_rank_no_vals_scan[i+1] = send_rank_no_vals_scan[i] + send_rank_no_vals[i];
      no_send_entries += send_rank_no_vals[i];
   }
   // This is the device memory we store our global i,j entries to send
   // We could just pack it up into 1 array and send that in one message, but then we need extra memory to unpack
   // There is also probably enough data that we will be bandwidth bound rather than latency so sending
   // two messages will probably not be slower
   Kokkos::View<PetscInt *> send_rows_d = Kokkos::View<PetscInt *>("send_rows_d", no_send_entries);
   Kokkos::View<PetscInt *> send_cols_d = Kokkos::View<PetscInt *>("send_cols_d", no_send_entries);

   // Now we pack up all the non-local entries into blocks of which rank we want to send it to
   // Let's use the sequential transpose on the non-local block to make that easier
   // This happens on the device
   Mat mat_nonlocal_x_transpose = NULL;
   MatTranspose(mat_nonlocal_x, MAT_INITIAL_MATRIX, &mat_nonlocal_x_transpose);   
   PetscInt local_rows_transpose, local_cols_transpose;
   MatGetLocalSize(mat_nonlocal_x_transpose, &local_rows_transpose, &local_cols_transpose);

   // Annoyingly there isn't currently the ability to get views for i (or j)
   const PetscInt *device_i_transpose = nullptr, *device_j_transpose = nullptr, *device_vals_transpose = nullptr;
   MatSeqAIJGetCSRAndMemType(mat_nonlocal_x_transpose, &device_i_transpose, &device_j_transpose, NULL, &mtype);  

   // Write the i,j indices into send_rows_d and send_cols_d
   // Start with the i indices
   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows_transpose, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

         // Row
         const PetscInt i = t.league_rank();

         // Still using i here (the local index into input)
         const PetscInt ncols = device_i_transpose[i + 1] - device_i_transpose[i];         

         // For over local columns
         Kokkos::parallel_for(
            Kokkos::TeamThreadRange(t, ncols), [&](const PetscInt j) {

            // Row index is the old column index
            send_rows_d(device_i_transpose[i] + j) = colmap_input_d(i);
            // Column index is the old row index
            send_cols_d(device_i_transpose[i] + j) = device_j_transpose[device_i_transpose[i] + j] + global_row_start;            
         });  
   });
   // We can destroy our local transpose
   MatDestroy(&mat_nonlocal_x_transpose);

   // for (PetscInt i = 0; i < no_send_entries; i++) {
   //    fprintf(stderr, "sending global row index %d global col index %d\n",
   //                send_rows_d[i], send_cols_d[i]);
   // }

   const PetscInt *iranks;
   PetscMPIInt niranks;

   // ~~~~~~~~~~~~~~
   // We can query the sf to get some information we need
   // The information below is talking about what we would send during a matvec
   // ~~~~~~~~~~~~~~    
   // niranks is how many ranks we send to
   // ioffset[i+1] - ioffset[i] is how many items we send to this rank
   // irootloc is the local column indices on this rank that we send
   //  ie the global column indices that we send are irootloc + global_col_start
   PetscSFGetLeafRanks(sf, &niranks, &iranks, NULL, NULL);

   // In the sf, we have a one to many relationship
   // Each of our local entries we may have to send to multiple ranks
   // So we when we think of the transpose we don't actually know how many entries we 
   // have to receive
   // ie we're not receiving ioffset[i+1] - ioffset[i] entries from iranks[i]
   // But we do know we are receiving from only the ranks in iranks
   // So we will just probe those ranks and find out how big the incoming messages are

   // Let's store how many things we are going to receive from each rank during
   // the transpose
   PetscInt *receive_rank_no_vals;
   PetscMalloc1(niranks, &receive_rank_no_vals);

   // for (int i = 0; i < niranks; i++) {
   //    PetscMPIInt receiver = iranks[i];
   //    PetscInt start = ioffset[i];
   //    PetscInt end   = ioffset[i+1];
   //    PetscInt nitems = end - start;

   //    fprintf(stderr, "[%d] sending %d items to rank %d\n",
   //                rank, nitems, receiver);
   //    for (PetscInt j = 0; j < nitems; j++) {

   //       fprintf(stderr, "[%d] sending local index %d remote index %d to rank %d\n",
   //                   rank, irootloc[start + j], irootloc[start + j] + global_col_start, receiver);
   //    }
   // }

   // ~~~~~~~~~~~~~~
   // Now let's start our non-blocking sends
   // ~~~~~~~~~~~~~~     

   // Now the start index for each rank into send_rows_d comes from send_rank_no_vals_scan
   MPI_Status *send_status;
   MPI_Request *send_request;   
   PetscMalloc1(nranks*2, &send_request);
   PetscMalloc1(nranks*2, &send_status);

   // We can give it the device pointer directly given gpu aware mpi
   for (int i = 0; i < nranks; i++) {      

      //fprintf(stderr, "rank %d start index into send %d \n", ranks[i], send_rank_no_vals_scan[i]);

      // Start an async send of our transposed data
      send_request[2*i] = MPI_REQUEST_NULL;
      send_request[2*i+1] = MPI_REQUEST_NULL;

      // Get the subset of rows we are sending to rank ranks[i]
      auto subview_send_rank_d = Kokkos::subview(send_rows_d, Kokkos::make_pair(send_rank_no_vals_scan[i], send_rank_no_vals_scan[i+1]));
      PetscInt *subview_send_rank_d_ptr = subview_send_rank_d.data();

      // Tag 0 for the i entries
      MPI_Isend(subview_send_rank_d_ptr, send_rank_no_vals[i], MPIU_INT, ranks[i], 0, MPI_COMM_MATRIX, &send_request[2 * i]);

      // Get the subset of cols we are sending to rank ranks[i]
      subview_send_rank_d = Kokkos::subview(send_cols_d, Kokkos::make_pair(send_rank_no_vals_scan[i], send_rank_no_vals_scan[i+1]));
      subview_send_rank_d_ptr = subview_send_rank_d.data();

      // Tag 1 for the j entries
      MPI_Isend(subview_send_rank_d_ptr, send_rank_no_vals[i], MPIU_INT, ranks[i], 1, MPI_COMM_MATRIX, &send_request[2 * i + 1]);      
   }   

   // ~~~~~~~~~~~~~~
   // We do our local transpose now to try and overlap work and comms
   // The tranpose of the local matrix happens on the device
   // ~~~~~~~~~~~~~~
   Mat mat_local_y = NULL, mat_nonlocal_y = NULL;
   MatTranspose(mat_local_x, MAT_INITIAL_MATRIX, &mat_local_y);   

   // ~~~~~~~~~~~~~~
   // Now let's post our non-blocking receives after our sends as we have a blocking 
   // probe
   // ~~~~~~~~~~~~~~   
   // First we have to find out how big our message is going to be
   // We can do this by probing the ranks we are going to receive from            
   for (int i = 0; i < niranks; i++)
   {
      // Tag 0 for the i,j entries
      MPI_Status probe_status;
      MPI_Probe(iranks[i], 0, MPI_COMM_MATRIX, &probe_status);
      // Get the message size
      MPI_Get_count(&probe_status, MPIU_INT, &receive_rank_no_vals[i]);
   }
   PetscInt no_receive_entries = 0;
   PetscInt *receive_rank_no_vals_scan;
   PetscMalloc1(niranks+1, &receive_rank_no_vals_scan);
   receive_rank_no_vals_scan[0] = 0;
   for (int i = 0; i < niranks; i++) {
      receive_rank_no_vals_scan[i+1] = receive_rank_no_vals_scan[i] + receive_rank_no_vals[i];
      no_receive_entries += receive_rank_no_vals[i];

      //fprintf(stderr, "rank %d start index into receive %d \n", iranks[i], receive_rank_no_vals_scan[i]);
   }

   // This is the device memory we store our received global i,j entries in
   Kokkos::View<PetscInt *> receive_rows_d = Kokkos::View<PetscInt *>("receive_rows_d", no_receive_entries);
   Kokkos::View<PetscInt *> receive_cols_d = Kokkos::View<PetscInt *>("receive_cols_d", no_receive_entries);

   MPI_Status *receive_status;
   MPI_Request *receive_request;
   PetscMalloc1(niranks*2, &receive_request);
   PetscMalloc1(niranks*2, &receive_status);

   // We can give it the device pointer directly given gpu aware mpi
   std::vector< Kokkos::View<PetscInt *> > sorted_views;
   for (int i = 0; i < niranks; i++)
   {
      // Start an async receive of our transposed data
      receive_request[2 * i] = MPI_REQUEST_NULL;
      receive_request[2 * i + 1] = MPI_REQUEST_NULL;

      // Get the subset of rows we are sending to rank ranks[i]
      auto subview_receive_rank_d = Kokkos::subview(receive_rows_d, Kokkos::make_pair(receive_rank_no_vals_scan[i], receive_rank_no_vals_scan[i+1]));
      // Store the subview
      sorted_views.push_back(subview_receive_rank_d);

      PetscInt *subview_receive_rank_d_ptr = subview_receive_rank_d.data();

      // Tag 0 for the i entries
      MPI_Irecv(subview_receive_rank_d_ptr, receive_rank_no_vals[i], MPIU_INT, iranks[i], 0, MPI_COMM_MATRIX, &receive_request[2 * i]);

      // Get the subset of rows we are sending to rank ranks[i]
      subview_receive_rank_d = Kokkos::subview(receive_cols_d, Kokkos::make_pair(receive_rank_no_vals_scan[i], receive_rank_no_vals_scan[i+1]));
      subview_receive_rank_d_ptr = subview_receive_rank_d.data();

      // Tag 1 for the j entries
      MPI_Irecv(subview_receive_rank_d_ptr, receive_rank_no_vals[i], MPIU_INT, iranks[i], 1, MPI_COMM_MATRIX, &receive_request[2 * i + 1]);
   }

   // Wait for all send/receives to complete
   MPI_Waitall(nranks, send_request, send_status);
   MPI_Waitall(niranks, receive_request, receive_status);

   // ~~~~~~~~~~~~~~
   // Now we should have our transposed data on the device
   // which are stored [[rows], [rows], ...] and [[cols], [cols], ...] from each rank we've received from
   // We now need to assemble our nonlocal_matrix
   // ~~~~~~~~~~~~~~  

   // fprintf(stderr, "global row start %d global row end %d\n",
   //                global_row_start, global_row_end_plus_one);                 
   // for (int i = 0; i < no_receive_entries; i++) {
   //    fprintf(stderr, "received global row index %d global col index %d\n",
   //                receive_rows_d[i], receive_cols_d[i]);
   // }

   // This sort should be reasonably efficient as long as we have a small number
   // of large arrays to merge, which is what our comms pattern looks like for a transpose
   Kokkos::View<PetscInt *> receive_rows_d_sorted;
   Kokkos::View<PetscInt *> receive_cols_d_sorted = Kokkos::View<PetscInt *>("receive_cols_d_sorted", no_receive_entries);
   Kokkos::View<PetscInt *> permutation_vector_d;
   parallel_merge_tree(sorted_views, receive_rows_d_sorted, permutation_vector_d);

   Kokkos::parallel_for("ApplyPermutation", receive_cols_d.extent(0), KOKKOS_LAMBDA(const int i) {
      receive_cols_d_sorted(i) = receive_cols_d(permutation_vector_d(i));
   });   

   // for (int i = 0; i < no_receive_entries; i++) {
   //    fprintf(stderr, "sorted receives global row index %d global col index %d\n",
   //                receive_rows_d_sorted[i], receive_cols_d_sorted[i]);
   // }    

   Kokkos::View<PetscScalar *> a_transpose_d_sorted = Kokkos::View<PetscScalar *>("a_transpose_d_sorted", no_receive_entries);
   Kokkos::deep_copy(a_transpose_d_sorted, 1);

   // Now all the row entries are sorted, but the columns within each row aren't sorted
   // So we call the sort_csr_matrix to do this
   // The column size is not right here (it will be <= receive_cols_d_sorted)
   // but it shouldn't matter as we are only construting an explicit kokkos csr matrix here so it can sort
   KokkosCsrMatrix csrmat_transpose = KokkosCsrMatrix("csrmat_local", local_rows, receive_cols_d_sorted.extent(0), receive_cols_d_sorted.extent(0), \
                                             a_transpose_d_sorted, i_transpose_d, receive_cols_d_sorted);  
   KokkosSparse::sort_crs_matrix(csrmat_transpose);

   // Let's make sure everything on the device is finished
   auto exec = PetscGetKokkosExecutionSpace();
   exec.fence();     

   // Now we need to build garray on the host and rewrite the receive_cols_d_sorted indices so they are local
   // The default values here are for the case where we 
   // let petsc do it, it resets this internally in MatSetUpMultiply_MPIAIJ
   PetscInt *garray_host = NULL;
   PetscInt col_ao_output = 0;
   // We don't have a good max bound on the number of unique non-local columns we received
   // so we just give how many non-local columns we received total as we know the unique number is less than that
   rewrite_j_global_to_local(receive_cols_d_sorted.extent(0), col_ao_output, receive_cols_d_sorted, &garray_host);  

   // for (int i = 0; i < col_ao_output; i++) {
   //    fprintf(stderr, "garray_host[%d] = %d\n", i, garray_host[i]);
   // }
   
   // We can create our nonlocal diagonal block matrix directly on the device
   MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, col_ao_output, i_transpose_d, receive_cols_d_sorted, a_transpose_d_sorted, &mat_nonlocal_y);

   //MatView(mat_nonlocal_y, PETSC_VIEWER_STDOUT_SELF);      

   // We can now create our MPI matrix - make sure dimensions are transposed
   MatCreateMPIAIJWithSeqAIJ(MPI_COMM_MATRIX, global_cols, global_rows, mat_local_y, mat_nonlocal_y, garray_host, Y);              

   // ~~~~~~~~~~~~~~
   // Cleanup
   // ~~~~~~~~~~~~~~ 

   PetscSFDestroy(&sf);
   (void)PetscFree(g_nnz);
   (void)PetscFree(send_rank_no_vals);
   (void)PetscFree(receive_rank_no_vals);
   (void)PetscFree(send_rank_no_vals_scan);
   (void)PetscFree(receive_rank_no_vals_scan);
   (void)PetscFree(send_request);
   (void)PetscFree(send_status);
   (void)PetscFree(receive_request);
   (void)PetscFree(receive_status);

   return;
}