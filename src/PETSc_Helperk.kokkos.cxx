// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

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
   bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

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
   // Copy in the tolerance
   Kokkos::deep_copy(rel_row_tol_d, tol);     
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
   
   // Compute the relative row tolerances if needed
   if (relative_max_row_tolerance_int) 
   {       
      // Reduction over all rows
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            PetscInt i = t.league_rank();
            PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];
            PetscScalar max_val = -1.0;
            const PetscInt row_index_global = i + global_row_start;

            // Reduce over local columns
            Kokkos::parallel_reduce(
               Kokkos::TeamVectorRange(t, ncols_local),
               [&](const PetscInt j, PetscScalar& thread_max) {

                  // Is this column the diagonal
                  bool is_diagonal = (device_local_j[device_local_i[i] + j] + global_col_start == row_index_global);

                  // If our current tolerance is bigger than the max value we've seen so far
                  PetscScalar val = Kokkos::abs(device_local_vals[device_local_i[i] + j]);
                  // If we're not comparing against the diagonal when computing relative residual
                  if (relative_max_row_tolerance_int == -1 && is_diagonal) val = -1.0;
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
                     bool is_diagonal = (colmap_input_d(device_nonlocal_j[device_nonlocal_i[i] + j]) == row_index_global);

                     // If our current tolerance is bigger than the max value we've seen so far
                     PetscScalar val = Kokkos::abs(device_nonlocal_vals[device_nonlocal_i[i] + j]);
                     // If we're not comparing against the diagonal when computing relative residual
                     if (relative_max_row_tolerance_int == -1 && is_diagonal) val = -1.0;                  
                     if (val > thread_max) thread_max = val;

                  },
                  Kokkos::Max<PetscScalar>(max_val_nonlocal)
               );
               // Take max of local and nonlocal
               if (max_val_nonlocal > max_val) max_val = max_val_nonlocal;               
            }

            // Only want one thread in the team to write the result
            Kokkos::single(Kokkos::PerTeam(t), [&]() {
               rel_row_tol_d(i) *= max_val;
            });
      });
   }

   // ~~~~~~~~~~~~
   // Need to count the number of nnzs we end up with, on each row and in total
   // ~~~~~~~~~~~~
   // Reduce over all the rows
   Kokkos::parallel_reduce(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t, PetscInt& thread_total) {

      PetscInt i   = t.league_rank(); // row i
      PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];
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
            bool is_diagonal = (device_local_j[device_local_i[i] + j] + global_col_start == row_index_global);
            
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
            
            PetscInt i = t.league_rank();
            PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
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
                  bool is_diagonal = (colmap_input_d(device_nonlocal_j[device_nonlocal_i[i] + j]) == row_index_global);

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
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(int i) {
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
         Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(int i) {
         i_nonlocal_d(i + 1) = nnz_match_nonlocal_row_d(i);
      });      
   }           
   
   auto exec = PetscGetKokkosExecutionSpace();
   
   // Create a team policy with scratch memory allocation
   // We want scratch space for each row
   // We will have ncols of integers which tell us what the matching indices we have
   // the last bit of memory is to account for 8-byte alignment for each view
   size_t scratch_size_per_team = 3 * max_nnz_local * sizeof(PetscInt) + \
               3 * max_nnz_nonlocal * sizeof(PetscInt) +
               8 * 6 * sizeof(PetscScalar);

   Kokkos::TeamPolicy<> policy(exec, local_rows, Kokkos::AUTO());
   // We're gonna use the level 1 scratch as our column data is probably bigger than the level 0
   policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size_per_team));

   // Execute with scratch memory
   Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const KokkosTeamMemberType& t) {
         
      // Row
      PetscInt i = t.league_rank();         
      // number of columns
      PetscInt ncols_local, ncols_nonlocal=-1;
      ncols_local = device_local_i[i + 1] - device_local_i[i];
      if (mpi) ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
      const PetscInt row_index_global = i + global_row_start;

      // Allocate views directly on scratch memory
      // Have to use views here given alignment issues
      ScratchIntView scratch_indices, scratch_indices_nonlocal, scratch_match, scratch_match_nonlocal;
      ScratchIntView scratch_lump, scratch_lump_nonlocal;
      scratch_indices = ScratchIntView(t.team_scratch(1), ncols_local); 
      if (mpi) scratch_indices_nonlocal = ScratchIntView(t.team_scratch(1), ncols_nonlocal);
      scratch_match = ScratchIntView(t.team_scratch(1), ncols_local); 
      if (mpi) scratch_match_nonlocal = ScratchIntView(t.team_scratch(1), ncols_nonlocal);      
      scratch_lump = ScratchIntView(t.team_scratch(1), ncols_local); 
      if (mpi) scratch_lump_nonlocal = ScratchIntView(t.team_scratch(1), ncols_nonlocal);      
      
      // Initialize scratch
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {
         scratch_match(j) = 0;
         scratch_lump(j) = 0;
      });
      if (mpi)
      {
         Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {
            scratch_match_nonlocal(j) = 0;
            scratch_lump_nonlocal(j) = 0;
         });      
      }
      // Team barrier to ensure all threads have finished filling the scratch space
      t.team_barrier();

      // Now go and mark which values we're keeping and lumping
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

         bool keep_col = false;
         bool is_diagonal = (device_local_j[device_local_i[i] + j] + global_col_start == row_index_global);

         // If we hit a diagonal put it in the lump'd value
         if (is_diagonal && lump_int) scratch_lump(j) = 1;            
         
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
            scratch_match(j) = 1;
         }
         // If we're not on the diagonal and we're small enough to lump
         else if (lump_int && !is_diagonal) {
            scratch_lump(j) = 1;
         }         
      }); 

      if (mpi)
      {
         // Now go and mark which values we're keeping and lumping
         Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {       

            bool keep_col = false;
            // Remember we can have diagonals in the off-diagonal block if we're rectangular
            bool is_diagonal = (colmap_input_d(device_nonlocal_j[device_nonlocal_i[i] + j]) == row_index_global);
      
            // If we hit a diagonal put it in the lump'd value
            if (is_diagonal && lump_int) scratch_lump_nonlocal(j) = 1;          
            
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
               scratch_match_nonlocal(j) = 1;             
            }
            // If we're not on the diagonal and we're small enough to lump
            else if (lump_int && !is_diagonal) {
               scratch_lump_nonlocal(j) = 1;
            }            
         });
      }

      // Team barrier to ensure all threads have finished filling the scratch space
      t.team_barrier(); 
      
      // Perform exclusive scan over scratch_match to get our output indices in this row
      Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, ncols_local), 
         [&](const PetscInt j, int& partial_sum, const bool is_final) {
            const int input_value = scratch_match(j);
            if (is_final) {
                  scratch_indices(j) = partial_sum;  // Write exclusive prefix
            }
            partial_sum += input_value;  // Update running total
         }
      );     
      
      if (mpi)
      {
         Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, ncols_nonlocal), 
         [&](const PetscInt j, int& partial_sum, const bool is_final) {
            const int input_value = scratch_match_nonlocal(j);
            if (is_final) {
                  scratch_indices_nonlocal(j) = partial_sum;  // Write exclusive prefix
            }
            partial_sum += input_value;  // Update running total
         }
      );          
      }

      PetscScalar lump_val = 0.0;
      // If lumping need to sum all the non-matching terms in input
      if (lump_int)
      {
         PetscScalar lump_val_local = 0.0, lump_val_nonlocal = 0.0;
         
         // Reduce over local columns
         Kokkos::parallel_reduce(
            Kokkos::TeamVectorRange(t, ncols_local),
            [&](const PetscInt j, PetscScalar& thread_sum) {          

               // If lumping
               if (scratch_lump(j) == 1) thread_sum += device_local_vals[device_local_i[i] + j];
            },
            Kokkos::Sum<PetscScalar>(lump_val_local)
         );   

         if (mpi)
         {
            // Reduce over nonlocal columns
            Kokkos::parallel_reduce(
               Kokkos::TeamVectorRange(t, ncols_nonlocal),
               [&](const PetscInt j, PetscScalar& thread_sum) {           

                  // If lumping
                  if (scratch_lump_nonlocal(j) == 1) thread_sum += device_nonlocal_vals[device_nonlocal_i[i] + j];
               },
               Kokkos::Sum<PetscScalar>(lump_val_nonlocal)
            );              
         }
         lump_val = lump_val_local + lump_val_nonlocal;
      } 

      // Team barrier to ensure all threads have finished scanning scratch_indices
      t.team_barrier();

      // Now go and write to the output
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {
         if (scratch_match(j) == 1)
         {
            j_local_d(i_local_d(i) + scratch_indices(j)) = device_local_j[device_local_i[i] + j];
            a_local_d(i_local_d(i) + scratch_indices(j)) = device_local_vals[device_local_i[i] + j];            
         }
      });

      if (mpi)
      {
         Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {
            if (scratch_match_nonlocal(j) == 1)
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
                  bool is_diagonal = j_local_d[i_local_d[i] + j] + global_col_start == row_index_global;

                  // Will only happen for one thread - lump_val contains the diagonal so we overwrite
                  if (is_diagonal) a_local_d[i_local_d[i] + j] = lump_val;
               });   
            }
            else
            {
               // Only loop over the ncols in the nonlocal component  - make sure this is over the output number of cols
               Kokkos::parallel_for(Kokkos::TeamThreadRange(t, i_nonlocal_d(i+1) - i_nonlocal_d(i)), [&](const PetscInt j) {

                  // Is this column the diagonal - j_nonlocal_d contains the global column index
                  bool is_diagonal = j_nonlocal_d[i_nonlocal_d[i] + j] == row_index_global;

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
      PetscInt col_ao_output = global_cols;
      if (cols_ao == 0)
      {
         // Silly but depending on the compiler this may return a non-null pointer
         col_ao_output = 0;
         PetscMalloc1(col_ao_output, &garray_host);
      }

      // We can use the Kokkos::UnorderedMap to do this if our 
      // off diagonal block has fewer than 4 billion non-zero columns (max capacity of uint32_t)
      // Otherwise we can just tell petsc to do do it on the host (in MatSetUpMultiply_MPIAIJ)
      // and rely on the hash tables in petsc on the host which can handle more than 4 billion entries
      // We trigger petsc doing it by passing in null as garray_host to MatSetMPIAIJKokkosWithSplitSeqAIJKokkosMatrices
      // If we have no off-diagonal entries (either we started with zero or we've dropped them all)
      // just skip all this and leave garray_host as null

      // If we have 4 bit ints, we know cols_ao can never be bigger than the capacity of uint32_t
      bool size_small_enough = sizeof(PetscInt) == 4 || \
                  (sizeof(PetscInt) > 4 && cols_ao < 4294967295);
      if (size_small_enough && cols_ao > 0 && nnzs_match_nonlocal > 0)
      {
         // Have to tell it the max capacity, we know we will have no more 
         // than the input off-diag columns
         Kokkos::UnorderedMap<PetscInt, PetscInt> hashmap((uint32_t)(cols_ao+1));

         // Let's insert all the existing global col indices as keys (with no value to start)
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, nnzs_match_nonlocal), KOKKOS_LAMBDA(int i) {      
            
            // Insert the key (global col indices) without a value
            // Duplicates will be ignored
            hashmap.insert(j_nonlocal_d(i));
         });

         // We now know how many unique global columns we have
         col_ao_output = hashmap.size();

         // Tag which of the original garray stick around  
         PetscIntKokkosView colmap_output_d_big("colmap_output_d_big", cols_ao);
         Kokkos::deep_copy(colmap_output_d_big, colmap_input_d);                

         // Mark which of the keys don't exist
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, cols_ao), KOKKOS_LAMBDA(int i) { 

            // If the key doesn't exist set the global index to -1
            if (!hashmap.exists(colmap_output_d_big(i))) colmap_output_d_big(i) = -1; 
         });         

         // Now sort the global columns indices
         // All the -1 should be at the start
         Kokkos::sort(colmap_output_d_big);

         // Count the number of -1 - this will be the index of the first entry
         // that isn't -1
         // It should never be equal to start index, because otherwise we
         // have dropped all nonlocal entries
         auto exec = PetscGetKokkosExecutionSpace();
         PetscInt start_index = Kokkos::Experimental::count(exec, colmap_output_d_big, -1);

         // Our final colmap_output_d is colmap_output_d_big(start_index:end)
         PetscIntKokkosView colmap_output_d = Kokkos::subview(colmap_output_d_big, \
                  Kokkos::make_pair(start_index, cols_ao));

         // Now we can clear the hash and instead stick in the global indices
         // but now with the local indices as values
         hashmap.clear();
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, colmap_output_d.extent(0)), KOKKOS_LAMBDA(int i) { 

            hashmap.insert(colmap_output_d(i), i);
         });          

         // And now we can overwrite j_nonlocal_d with the local indices
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, nnzs_match_nonlocal), KOKKOS_LAMBDA(int i) {     

            // Find where our global col index is at
            uint32_t loc = hashmap.find(j_nonlocal_d(i));
            // And get the value (the new local index)
            j_nonlocal_d(i) = hashmap.value_at(loc);
         });      
         hashmap.clear();

         // Create some host space for the output garray (that stays in scope) and copy it
         PetscMalloc1(colmap_output_d.extent(0), &garray_host);
         PetscIntKokkosViewHost colmap_output_h = PetscIntKokkosViewHost(garray_host, colmap_output_d.extent(0));
         Kokkos::deep_copy(colmap_output_h, colmap_output_d);
         // Log copy with petsc
         bytes = colmap_output_d.extent(0) * sizeof(PetscInt);
         PetscLogGpuToCpu(bytes);            
      } 
      
      // Let's make sure everything on the device is finished
      exec.fence();

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
PETSC_INTERN void remove_from_sparse_match_kokkos(Mat *input_mat, Mat *output_mat, const int lump_int)
{

   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start_temp, global_row_end_plus_one_temp;
   PetscInt global_col_start_temp, global_col_end_plus_one_temp;
   PetscInt rows_ao_input, cols_ao_input, rows_ao_output, cols_ao_output;
   MatType mat_type;

   MatGetType(*input_mat, &mat_type);
   // Are we in parallel?
   bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

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
      PetscInt i = t.league_rank();
      const PetscInt row_index_global = i + global_row_start;

      // number of columns
      PetscInt ncols_local, ncols_nonlocal=-1, ncols_local_output, ncols_nonlocal_output=-1;
      ncols_local = device_local_i[i + 1] - device_local_i[i];
      if (mpi) ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];

      ncols_local_output = device_local_i_output[i + 1] - device_local_i_output[i];
      if (mpi) ncols_nonlocal_output = device_nonlocal_i_output[i + 1] - device_nonlocal_i_output[i];      

      // Allocate views directly on scratch memory
      // Have to use views here given alignment issues
      ScratchIntView scratch_indices, scratch_indices_nonlocal;
      scratch_indices = ScratchIntView(t.team_scratch(1), ncols_local); 
      if (mpi) scratch_indices_nonlocal = ScratchIntView(t.team_scratch(1), ncols_nonlocal);   

      // Initialize scratch
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {
            scratch_indices(j) = -1;
      });
      if (mpi)
      {
         Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {
            scratch_indices_nonlocal(j) = -1;
         });      
      }
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
               if (scratch_indices(j) == -1) thread_sum += device_local_vals[device_local_i[i] + j];
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
                  if (scratch_indices_nonlocal(j) == -1) thread_sum += device_nonlocal_vals[device_nonlocal_i[i] + j];
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
            device_local_vals_output[device_local_i_output[i] + scratch_indices(j)] = device_local_vals[device_local_i[i] + j];
         }
      }); 
      
      if (mpi)
      {
         Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) { 
   
            // If we have a match, copy the value
            if (scratch_indices_nonlocal(j) != -1)
            {
               device_nonlocal_vals_output[device_nonlocal_i_output[i] + scratch_indices_nonlocal(j)] = device_nonlocal_vals[device_nonlocal_i[i] + j];                  
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
               bool is_diagonal = device_local_j_output[device_local_i_output[i] + j] + global_col_start == row_index_global;

               // Will only happen for one thread
               if (is_diagonal) device_local_vals_output[device_local_i_output[i] + j] += lump_val;
            });   
         }
         else
         {
            // Only loop over the ncols in the nonlocal component
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_nonlocal_output), [&](const PetscInt j) {

               // Is this column the diagonal
               bool is_diagonal = colmap_output_d(device_nonlocal_j_output[device_nonlocal_i_output[i] + j]) == row_index_global;

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
   bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

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
   bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

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

      PetscInt i   = t.league_rank(); // row i
      PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];
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
            bool is_diagonal = (device_local_j[device_local_i[i] + j] + global_col_start == row_index_global);
            
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
            Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(int i) {

               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
               nnz_match_nonlocal_row_d(i) = ncols_nonlocal;
         });
      }
      if (mpi)
      {
         Kokkos::parallel_reduce ("ReductionNonLocal", local_rows, KOKKOS_LAMBDA (const int i, PetscInt& update) {
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
         Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(int i) {

            // Row index is simple
            PetscInt row_index = i;       

            // The start of our row index comes from the scan
            i_local_d(row_index + 1) = nnz_match_local_row_d(row_index);   
            if (mpi) i_nonlocal_d(row_index + 1) = nnz_match_nonlocal_row_d(row_index);         
      });            


      // Loop over the rows
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            PetscInt i = t.league_rank();

            // Row index is simple
            PetscInt row_index = i;
            // Still using i here (the local index into input)
            PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];         

            // For over local columns - copy in input
            Kokkos::parallel_for(
               Kokkos::TeamVectorRange(t, ncols_local), [&](const PetscInt j) {

               // Want the local col indices for the local block
               j_local_d(i_local_d(row_index) + j) = device_local_j[device_local_i[i] + j];
               a_local_d(i_local_d(row_index) + j) = device_local_vals[device_local_i[i] + j];
                     
            });     

            // For over nonlocal columns - copy in input
            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];         

               Kokkos::parallel_for(
                  Kokkos::TeamVectorRange(t, ncols_nonlocal), [&](const PetscInt j) {

                  // We keep the existing local indices in the off-diagonal block here
                  j_nonlocal_d(i_nonlocal_d(row_index) + j) = device_nonlocal_j[device_nonlocal_i[i] + j];
                  a_nonlocal_d(i_nonlocal_d(row_index) + j) = device_nonlocal_vals[device_nonlocal_i[i] + j];
                        
               });          
            }

            // Only want one thread to deal with the diagonal
            Kokkos::single(Kokkos::PerTeam(t), [&]() {            
               // If we didn't find a diagonal
               if (!found_diag_row_d(i))
               {
                  // Let's just stick it at the end and we will sort after
                  j_local_d(i_local_d(row_index) + ncols_local) = i;
                  a_local_d(i_local_d(row_index) + ncols_local) = 0.0;
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
            PetscInt i = t.league_rank();

            // Simple row index
            PetscInt row_index = i;
            // Still using i here (the local index into input)
            PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];         

            // For over local columns - copy in input
            // We have to skip over the identity entries, which we know are always C points
            Kokkos::parallel_for(
               Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

               PetscInt offset = 0;

               // If we're at or after the diagonal and there isn't actually a diagonal in the input
               // we know the output has a diagonal, so we skip ahead one in the output and 
               // leave it unassigned in the output (it gets set to zero above)
               if (j_local_const_d(i_local_const_d(row_index) + j) >= i && \
                     !found_diag_row_d(i)) offset = 1;

               a_local_d(i_local_const_d(row_index) + j + offset) = device_local_vals[device_local_i[i] + j];
            });  
           
            // For over nonlocal columns - copy in input - identical structure in the off-diag block
            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];         

               Kokkos::parallel_for(
                  Kokkos::TeamVectorRange(t, ncols_nonlocal), [&](const PetscInt j) {

                  // We keep the existing local indices in the off-diagonal block here
                  // we have all the same columns as input and hence the same garray
                  a_nonlocal_d(i_nonlocal_const_d(row_index) + j) = device_nonlocal_vals[device_nonlocal_i[i] + j];
                        
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
         for (int i = 0; i < cols_ao; i++)
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

   Kokkos::View<PetscScalar *> a_local_d_copy = Kokkos::View<PetscScalar *>("a_local_d_copy", a_local_d_z.extent(0));
   Kokkos::View<PetscInt *> i_local_d_copy = Kokkos::View<PetscInt *>("i_local_d_copy", i_local_d_z.extent(0));
   Kokkos::View<PetscInt *> j_local_d_copy = Kokkos::View<PetscInt *>("j_local_d_copy", j_local_d_z.extent(0));   

   Kokkos::deep_copy(a_local_d_copy, a_local_d_z);
   Kokkos::deep_copy(i_local_d_copy, i_local_d_z);
   Kokkos::deep_copy(j_local_d_copy, j_local_d_z);

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
      Kokkos::RangePolicy<>(0, ykok_nonlocal->csrmat.nnz()), KOKKOS_LAMBDA(int i) { 

         device_nonlocal_y_j[i] = colmap_input_d_y(device_nonlocal_y_j[i]);
   }); 

   // Rewrite the X nonlocal indices to be global
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, xkok_nonlocal->csrmat.nnz()), KOKKOS_LAMBDA(int i) { 

         device_nonlocal_x_j[i] = colmap_input_d_x(device_nonlocal_x_j[i]);
   });    

   // ~~~~~~~~~

   // Now we can add the non-local components together
   KokkosCsrMatrix zcsr_nonlocal;
   // Not sure if the indices are sorted once we have replaced them with the global indices, 
   // let's just set to false
   KernelHandle    kh_nonlocal;
   kh_nonlocal.create_spadd_handle(false); 

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
   PetscInt nnzs_match_nonlocal = j_nonlocal_d_z.extent(0);

   // ~~~~~~~~~

   // Now we need to build garray on the host and rewrite the j_nonlocal_d_z indices so they are local
   auto exec = PetscGetKokkosExecutionSpace();
   PetscInt *garray_host = NULL;

   // We have all the global column indices in j_nonlocal_d_z
   // We can use unique_copy in kokkos to get us a copy of the unique global column indices
   // which gives us our garray

   // Need to preallocate to the max size, which we know is only as big as cols_ao_x + cols_ao_y
   PetscIntKokkosView colmap_output_d("colmap_output_d_big", cols_ao);
   Kokkos::deep_copy(colmap_output_d, -1); // initialize to -1

   // Take a copy of j and sort it
   PetscIntKokkosView j_nonlocal_d_z_sorted("j_nonlocal_d_z_sorted", j_nonlocal_d_z.extent(0));
   Kokkos::deep_copy(j_nonlocal_d_z_sorted, j_nonlocal_d_z);
   Kokkos::sort(j_nonlocal_d_z_sorted);

   // Unique copy returns a copy of sorted j_nonlocal_d_z in order, but with all the duplicate entries removed
   auto unique_end_it = Kokkos::Experimental::unique_copy(exec, j_nonlocal_d_z_sorted, colmap_output_d);
   auto begin_it = Kokkos::Experimental::begin(colmap_output_d);
   ptrdiff_t count_ptr_arith = unique_end_it - begin_it;
   PetscInt col_ao_output = static_cast<PetscInt>(count_ptr_arith);

   // Now we need to rewrite our global indices
   if (col_ao_output == 0)
   {
      // Silly but depending on the compiler this may return a non-null pointer
      col_ao_output = 0;
      PetscMalloc1(col_ao_output, &garray_host);
   }

   // We can use the Kokkos::UnorderedMap to do this if our 
   // off diagonal block has fewer than 4 billion non-zero columns (max capacity of uint32_t)
   // Otherwise we can just tell petsc to do do it on the host (in MatSetUpMultiply_MPIAIJ)
   // and rely on the hash tables in petsc on the host which can handle more than 4 billion entries
   // We trigger petsc doing it by passing in null as garray_host to MatSetMPIAIJKokkosWithSplitSeqAIJKokkosMatrices
   // If we have no off-diagonal entries (either we started with zero or we've dropped them all)
   // just skip all this and leave garray_host as null

   // If we have 4 bit ints, we know col_ao_output can never be bigger than the capacity of uint32_t
   bool size_small_enough = sizeof(PetscInt) == 4 || \
               (sizeof(PetscInt) > 4 && col_ao_output < 4294967295);
   if (size_small_enough && col_ao_output > 0 && nnzs_match_nonlocal > 0)
   {
      // Have to tell it the max capacity, we know we will have no more 
      // than the input off-diag columns
      Kokkos::UnorderedMap<PetscInt, PetscInt> hashmap((uint32_t)(col_ao_output+1));

      // Let's insert all the existing global col indices as keys (with no value to start)
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, col_ao_output), KOKKOS_LAMBDA(int i) {      
         
         // Insert the key (global col indices) with the local index
         hashmap.insert(colmap_output_d(i), i);
      });

      // And now we can overwrite j_nonlocal_d_z with the local indices
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, nnzs_match_nonlocal), KOKKOS_LAMBDA(int i) {     

         // Find where our global col index is at
         uint32_t loc = hashmap.find(j_nonlocal_d_z(i));
         // And get the value (the new local index)
         j_nonlocal_d_z(i) = hashmap.value_at(loc);
      });      
      hashmap.clear();

      // Create some host space for the output garray (that stays in scope) and copy it
      PetscMalloc1(col_ao_output, &garray_host);
      PetscIntKokkosViewHost colmap_output_h = PetscIntKokkosViewHost(garray_host, col_ao_output);
      PetscInt zero = 0;
      Kokkos::deep_copy(colmap_output_h, Kokkos::subview(colmap_output_d, Kokkos::make_pair(zero, col_ao_output)));

      // Log copy with petsc
      bytes = col_ao_output * sizeof(PetscInt);
      PetscLogGpuToCpu(bytes);
   }

   // Let's make sure everything on the device is finished
   exec.fence();  

   Kokkos::View<PetscScalar *> a_nonlocal_d_copy = Kokkos::View<PetscScalar *>("a_local_d_copy", a_nonlocal_d_z.extent(0));
   Kokkos::View<PetscInt *> i_nonlocal_d_copy = Kokkos::View<PetscInt *>("i_local_d_copy", i_nonlocal_d_z.extent(0));
   Kokkos::View<PetscInt *> j_nonlocal_d_copy = Kokkos::View<PetscInt *>("j_local_d_copy", j_nonlocal_d_z.extent(0));   

   Kokkos::deep_copy(a_nonlocal_d_copy, a_nonlocal_d_z);
   Kokkos::deep_copy(i_nonlocal_d_copy, i_nonlocal_d_z);
   Kokkos::deep_copy(j_nonlocal_d_copy, j_nonlocal_d_z);   

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
