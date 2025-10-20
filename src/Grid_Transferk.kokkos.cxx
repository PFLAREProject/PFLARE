// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>


//------------------------------------------------------------------------------------------------------------------------

// Generate one point classical prolongator but with kokkos - keeping everything on the device
PETSC_INTERN void generate_one_point_with_one_entry_from_sparse_kokkos(Mat *input_mat, Mat *output_mat)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt global_col_start, global_col_end_plus_one;
   PetscInt rows_ao, cols_ao;
   MatType mat_type;
   PetscInt nnzs_match_local, nnzs_match_nonlocal;
   Mat output_mat_local, output_mat_nonlocal;

   PetscCallVoid(MatGetType(*input_mat, &mat_type));
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
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao);); 

      // We also copy the input mat colmap over to the device as we need it
      colmap_input_h = PetscIntKokkosViewHost(mat_mpi->garray, cols_ao);
      colmap_input_d = PetscIntKokkosView("colmap_input_d", cols_ao);
      Kokkos::deep_copy(colmap_input_d, colmap_input_h);
      // Log copy with petsc
      size_t bytes = colmap_input_h.extent(0) * sizeof(PetscInt);
      PetscCallVoid(PetscLogCpuToGpu(bytes));              
   }
   else
   {
      mat_local = *input_mat;
   }

   // Get the comm
   PetscCallVoid(PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX));
   PetscCallVoid(MatGetLocalSize(*input_mat, &local_rows, &local_cols));
   PetscCallVoid(MatGetSize(*input_mat, &global_rows, &global_cols));
   // This returns the global index of the local portion of the matrix
   PetscCallVoid(MatGetOwnershipRange(*input_mat, &global_row_start, &global_row_end_plus_one));
   PetscCallVoid(MatGetOwnershipRangeColumn(*input_mat, &global_col_start, &global_col_end_plus_one));

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr, *device_nonlocal_vals = nullptr;  
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype));  
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype));          

   // ~~~~~~~~~~~~
   // Get the number of nnzs
   // ~~~~~~~~~~~~
   nnzs_match_local = 0;
   nnzs_match_nonlocal = 0;

   // ~~~~~~~~~~~~~~~~~~~~~~~
   // Let's build our i, j, and a on the device
   // ~~~~~~~~~~~~~~~~~~~~~~~   
   // We need to know where our max values are
   PetscIntKokkosView max_col_row_d("max_col_row_d", local_rows);    
   // We need to know how many entries are in each row  
   PetscIntKokkosView nnz_match_local_row_d("nnz_match_local_row_d", local_rows);             
   Kokkos::deep_copy(nnz_match_local_row_d, 0);
   PetscIntKokkosView nnz_match_nonlocal_row_d;
   if (mpi) 
   {
      nnz_match_nonlocal_row_d = PetscIntKokkosView("nnz_match_nonlocal_row_d", local_rows); 
      Kokkos::deep_copy(nnz_match_nonlocal_row_d, 0);
   }

   // Loop over the rows and find the biggest entry in each row
   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

      const PetscInt i   = t.league_rank(); // row i
      const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];

      // We have a custom reduction type defined - ReduceDataMaxRow
      ReduceDataMaxRow local_row_result, nonlocal_row_result;

      // Reduce over all the columns
      Kokkos::parallel_reduce(
         Kokkos::TeamThreadRange(t, ncols_local),
         [&](const PetscInt j, ReduceDataMaxRow& thread_data) {

            // If it's the biggest value keep it
            if (Kokkos::abs(device_local_vals[device_local_i[i] + j]) > thread_data.val) {
               thread_data.val = Kokkos::abs(device_local_vals[device_local_i[i] + j]);
               thread_data.col = device_local_j[device_local_i[i] + j];
            }
         }, local_row_result
      );

      if (mpi)
      {
         PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
         Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(t, ncols_nonlocal),
            [&](const PetscInt j, ReduceDataMaxRow& thread_data) {

               // If it's the biggest value keep it
               if (Kokkos::abs(device_nonlocal_vals[device_nonlocal_i[i] + j]) > thread_data.val) {
                  thread_data.val = Kokkos::abs(device_nonlocal_vals[device_nonlocal_i[i] + j]);
                  // Set the global index
                  thread_data.col = colmap_input_d(device_nonlocal_j[device_nonlocal_i[i] + j]);
               }
            }, nonlocal_row_result
         );         
      }

      // Only want one thread in the team to write the result
      Kokkos::single(Kokkos::PerTeam(t), [&]() {     

         // We know the entry is local
         if (!mpi)
         {
            // Check we found an entry
            if (local_row_result.col != -1) {
               max_col_row_d(i) = local_row_result.col;
               nnz_match_local_row_d(i)++;
            }
         }
         // If we have mpi we have to check both the local
         // and nonlocal block maxs
         else
         {
            // If our biggest entry is nonlocal
            if (nonlocal_row_result.val > local_row_result.val) {
               // Check we found an entry
               if (nonlocal_row_result.col != -1) {
                  max_col_row_d(i) = nonlocal_row_result.col;
                  nnz_match_nonlocal_row_d(i)++;
               }
            }
            // The local entry is the biggest
            else if (nonlocal_row_result.val < local_row_result.val) {
                  // Check we found an entry
                  if (local_row_result.col != -1) {
                     max_col_row_d(i) = local_row_result.col;
                     nnz_match_local_row_d(i)++;
                  }
            }        
            // If they are equal - let's check they're valid to start
            else if (local_row_result.col != -1 && nonlocal_row_result.col != -1)
            {
               // Always pick the local entry
               max_col_row_d(i) = local_row_result.col;
               nnz_match_local_row_d(i)++;
            }    
         }
      });      
   });      

   // Get number of nnzs
   Kokkos::parallel_reduce ("ReductionLocal", local_rows, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update) {
      update += nnz_match_local_row_d(i); 
   }, nnzs_match_local);   
   if (mpi)
   {
      Kokkos::parallel_reduce ("ReductionNonLocal", local_rows, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update) {
         update += nnz_match_nonlocal_row_d(i); 
      }, nnzs_match_nonlocal);       
   }   

   // ~~~~~~~~~~~~

   // Store original counts before scan
   PetscIntKokkosView has_entry_local_d("has_entry_local_d", local_rows);
   Kokkos::deep_copy(has_entry_local_d, nnz_match_local_row_d); 
   PetscIntKokkosView has_entry_nonlocal_d;
   if (mpi)
   {
      has_entry_nonlocal_d = PetscIntKokkosView ("has_entry_nonlocal_d", local_rows);
      Kokkos::deep_copy(has_entry_nonlocal_d, nnz_match_nonlocal_row_d);
   }  

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
   Kokkos::View<PetscScalar *> a_local_d = Kokkos::View<PetscScalar *>("a_local_d", nnzs_match_local);
   Kokkos::View<PetscInt *> i_local_d = Kokkos::View<PetscInt *>("i_local_d", local_rows+1);
   Kokkos::View<PetscInt *> j_local_d = Kokkos::View<PetscInt *>("j_local_d", nnzs_match_local);

   // Get device views
   // Initialize first entry to zero - the rest get set below
   Kokkos::deep_copy(Kokkos::subview(i_local_d, 0), 0);       

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
   }        

   // Initialize i_local_d row pointers (1 to local_rows) with cumulative sums from the scan
   PetscInt one = 1;
   auto i_local_range = Kokkos::subview(i_local_d, Kokkos::make_pair(one, local_rows+1));
   Kokkos::deep_copy(i_local_range, nnz_match_local_row_d);
   
   // Similarly for MPI nonlocal case if needed
   if (mpi) {
      auto i_nonlocal_range = Kokkos::subview(i_nonlocal_d, Kokkos::make_pair(one, local_rows+1));
      Kokkos::deep_copy(i_nonlocal_range, nnz_match_nonlocal_row_d);
   }          
   
   // Filling the matrix is easy as we know we only have one non-zero per row
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt i) {

      // If our max val is in the local block
      if (has_entry_local_d(i) > 0) {
         j_local_d(i_local_d(i)) = max_col_row_d(i);
         a_local_d(i_local_d(i)) = 1.0;
      }
      else if (mpi && has_entry_nonlocal_d(i) > 0)
      {
         j_nonlocal_d(i_nonlocal_d(i)) = max_col_row_d(i);
         a_nonlocal_d(i_nonlocal_d(i)) = 1.0;         
      }   
   });      

   // Let's make sure everything on the device is finished
   auto exec = PetscGetKokkosExecutionSpace();
   exec.fence();
   
   // We can create our local diagonal block matrix directly on the device
   PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, local_cols, i_local_d, j_local_d, a_local_d, &output_mat_local));

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
      PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, col_ao_output, i_nonlocal_d, j_nonlocal_d, a_nonlocal_d, &output_mat_nonlocal)); 

      // We can now create our MPI matrix
      PetscCallVoid(MatCreateMPIAIJWithSeqAIJ(MPI_COMM_MATRIX, global_rows, global_cols, output_mat_local, output_mat_nonlocal, garray_host, output_mat));
   }     
   // If in serial 
   else
   {
      *output_mat = output_mat_local;
   }

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Stick W in a full sized P but with kokkos - keeping everything on the device
PETSC_INTERN void compute_P_from_W_kokkos(Mat *input_mat, PetscInt global_row_start, IS *is_fine, \
                  IS *is_coarse, int identity_int, int reuse_int, Mat *output_mat)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt global_row_start_W, global_row_end_plus_one_W;
   PetscInt global_col_start_W, global_col_end_plus_one_W;
   PetscInt local_rows_coarse, local_rows, local_cols, local_cols_coarse;
   PetscInt cols_z, rows_z, local_rows_fine, global_cols_coarse, global_rows, global_cols;
   PetscInt rows_ao, cols_ao;
   MatType mat_type;
   PetscInt nnzs_match_local, nnzs_match_nonlocal;
   Mat output_mat_local, output_mat_nonlocal;

   PetscCallVoid(MatGetType(*input_mat, &mat_type));
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
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao);); 

      // We also copy the input mat colmap over to the device as we need it
      // Don't actually need on the device for this routine
      // colmap_input_h = PetscIntKokkosViewHost(mat_mpi->garray, cols_ao);
      //colmap_input_d = PetscIntKokkosView("colmap_input_d", cols_ao);
      //Kokkos::deep_copy(colmap_input_d, colmap_input_h);        
   }
   else
   {
      mat_local = *input_mat;
   }

   PetscCallVoid(MatGetOwnershipRange(*input_mat, &global_row_start_W, &global_row_end_plus_one_W));
   PetscCallVoid(MatGetOwnershipRangeColumn(*input_mat, &global_col_start_W, &global_col_end_plus_one_W));

   PetscCallVoid(MatGetSize(*input_mat, &cols_z, &rows_z));

   // Get pointers to the indices on the host
   const PetscInt *fine_indices_ptr, *coarse_indices_ptr;
   PetscCallVoid(ISGetIndices(*is_fine, &fine_indices_ptr));   
   PetscCallVoid(ISGetIndices(*is_coarse, &coarse_indices_ptr)); 

   PetscCallVoid(ISGetLocalSize(*is_coarse, &local_rows_coarse));
   PetscCallVoid(ISGetLocalSize(*is_fine, &local_rows_fine));

   // Create a host view of the existing indices
   auto fine_view_h = PetscIntConstKokkosViewHost(fine_indices_ptr, local_rows_fine);    
   auto fine_view_d = PetscIntKokkosView("fine_view_d", local_rows_fine);   
   auto coarse_view_h = PetscIntConstKokkosViewHost(coarse_indices_ptr, local_rows_coarse);    
   auto coarse_view_d = PetscIntKokkosView("coarse_view_d", local_rows_coarse);      
   // Copy indices to the device
   Kokkos::deep_copy(fine_view_d, fine_view_h);     
   Kokkos::deep_copy(coarse_view_d, coarse_view_h);
   // Log copy with petsc
   size_t bytes = fine_view_h.extent(0) * sizeof(PetscInt);
   PetscCallVoid(PetscLogCpuToGpu(bytes));        
   bytes = coarse_view_h.extent(0) * sizeof(PetscInt);
   PetscCallVoid(PetscLogCpuToGpu(bytes));      

   local_cols_coarse = local_rows_coarse;
   local_cols = local_rows_coarse + local_rows_fine;
   local_rows = local_cols; 
   
   global_cols = rows_z + cols_z;
   global_rows = global_cols;
   //global_rows_coarse = rows_z;
   global_cols_coarse = rows_z;    

   // Get the comm
   PetscCallVoid(PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX));

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr, *device_nonlocal_vals = nullptr;  
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype));  
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype));          

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

   // Only need things to do with the sparsity pattern if we're not reusing
   if (!reuse_int)
   {
      // ~~~~~~~~~~~~
      // Get the number of nnzs
      // ~~~~~~~~~~~~
      nnzs_match_local = 0;
      nnzs_match_nonlocal = 0;

      // ~~~~~~~~~~~~~~~~~~~~~~~
      // Let's build our i, j, and a on the device
      // ~~~~~~~~~~~~~~~~~~~~~~~ 
      // We need to know how many entries are in each row 
      nnz_match_local_row_d = PetscIntKokkosView("nnz_match_local_row_d", local_rows);    
      // We may have identity
      Kokkos::deep_copy(nnz_match_local_row_d, 0);         
      if (mpi) nnz_match_nonlocal_row_d = PetscIntKokkosView("nnz_match_nonlocal_row_d", local_rows);                  

      // ~~~~~~~~~~~~
      // Need to count the number of nnzs we end up with, on each row and in total
      // ~~~~~~~~~~~~
      // Loop over the rows of W
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows_fine), KOKKOS_LAMBDA(PetscInt i) {

            // Convert to global fine index into a local index in the full matrix
            PetscInt row_index = fine_view_d(i) - global_row_start;
            // Still using i here (the local index into W)
            const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];
            nnz_match_local_row_d(row_index) = ncols_local;

            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
               nnz_match_nonlocal_row_d(row_index) = ncols_nonlocal;
            }
      });

      // Loop over all the C points - we know they're in the local block
      if (identity_int) 
      {
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, local_rows_coarse), KOKKOS_LAMBDA(PetscInt i) {

            // Convert to global coarse index into a local index into the full matrix
            PetscInt row_index = coarse_view_d(i) - global_row_start;
            nnz_match_local_row_d(row_index)++;
         }); 
      }  

      // Get number of nnzs
      Kokkos::parallel_reduce ("ReductionLocal", local_rows, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update) {
         update += nnz_match_local_row_d(i); 
      }, nnzs_match_local);   
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

      // Get device views
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
      // Have to build i_local_d and the nonlocal for every row (F and C points)
      // regardless of if we are sticking 1 in (ie identity)
      // This has to happen before the main loop as the f and c 
      // points are placed in different orders (ie not in order as the index 
      // is row_index 
      // ~~~~~~~~~~~~~~~
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows_fine), KOKKOS_LAMBDA(PetscInt i) {

            // Convert to global fine index into a local index in the full matrix
            PetscInt row_index = fine_view_d(i) - global_row_start;       

            // The start of our row index comes from the scan
            i_local_d(row_index + 1) = nnz_match_local_row_d(row_index);   
            if (mpi) i_nonlocal_d(row_index + 1) = nnz_match_nonlocal_row_d(row_index);         
      });            

      // Always have to set the i_local_d for C points, regardless of if we are setting
      // 1 in the identity part for them
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows_coarse), KOKKOS_LAMBDA(PetscInt i) {

         // Convert to global coarse index into a local index into the full matrix
         PetscInt row_index = coarse_view_d(i) - global_row_start;

         // The start of our row index comes from the scan
         i_local_d(row_index + 1) = nnz_match_local_row_d(row_index);
         if (mpi) i_nonlocal_d(row_index + 1) = nnz_match_nonlocal_row_d(row_index);        

      });  

      // Loop over the rows of W
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows_fine, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            const PetscInt i = t.league_rank();

            // Convert to global fine index into a local index in the full matrix
            PetscInt row_index = fine_view_d(i) - global_row_start;
            // Still using i here (the local index into W)
            const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];         

            // For over local columns - copy in W
            Kokkos::parallel_for(
               Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

               j_local_d(i_local_d(row_index) + j) = device_local_j[device_local_i[i] + j];
               a_local_d(i_local_d(row_index) + j) = device_local_vals[device_local_i[i] + j];
                     
            });     

            // For over nonlocal columns - copy in W
            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];         

               Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {

                  // We keep the existing local indices in the off-diagonal block here
                  // we have all the same columns as W and hence the same garray
                  j_nonlocal_d(i_nonlocal_d(row_index) + j) = device_nonlocal_j[device_nonlocal_i[i] + j];
                  a_nonlocal_d(i_nonlocal_d(row_index) + j) = device_nonlocal_vals[device_nonlocal_i[i] + j];
                        
               });          
            }
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

      // Annoyingly there isn't currently the ability to get views for i (or j)
      const PetscInt *device_local_i_output = nullptr, *device_nonlocal_i_ouput = nullptr;
      PetscMemType mtype;
      PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_output, &device_local_i_output, NULL, NULL, &mtype));  
      if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_output, &device_nonlocal_i_ouput, NULL, NULL, &mtype));  

      // Have these point at the existing i pointers - we don't need j if we're reusing
      ConstMatRowMapKokkosView i_local_const_d = ConstMatRowMapKokkosView(device_local_i_output, local_rows+1);
      ConstMatRowMapKokkosView i_nonlocal_const_d;
      if (mpi) i_nonlocal_const_d = ConstMatRowMapKokkosView(device_nonlocal_i_ouput, local_rows+1);        

      // Only have to write W as the identity block cannot change
      // Loop over the rows of W - annoying we have const views as this is just the same loop as above
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows_fine, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            const PetscInt i = t.league_rank();

            // Convert to global fine index into a local index in the full matrix
            PetscInt row_index = fine_view_d(i) - global_row_start;
            // Still using i here (the local index into W)
            const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];         

            // For over local columns - copy in W
            Kokkos::parallel_for(
               Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

               a_local_d(i_local_const_d(row_index) + j) = device_local_vals[device_local_i[i] + j];
                     
            });     

            // For over nonlocal columns - copy in W
            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];         

               Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {

                  // We keep the existing local indices in the off-diagonal blocl here
                  // we have all the same columns as W and hence the same garray
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
      PetscCallVoid(PetscObjectStateIncrease((PetscObject)(*output_mat)));

   }

   // ~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~   

   if (!reuse_int)
   {
      // Loop over all the C points - we know they're in the local block
      if (identity_int) 
      {
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, local_rows_coarse), KOKKOS_LAMBDA(PetscInt i) {

            // Convert to global coarse index into a local index into the full matrix
            PetscInt row_index = coarse_view_d(i) - global_row_start;

            // Only a single column
            j_local_d(i_local_d(row_index)) = i;
            a_local_d(i_local_d(row_index)) = 1.0;         

         }); 
      }   
        
      // Let's make sure everything on the device is finished
      auto exec = PetscGetKokkosExecutionSpace();
      exec.fence();      

      // We can create our local diagonal block matrix directly on the device
      PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, local_cols_coarse, i_local_d, j_local_d, a_local_d, &output_mat_local));

      // we also have to go and build the a, i, j for the non-local off-diagonal block
      if (mpi) 
      {
         // We can create our nonlocal diagonal block matrix directly on the device
         // Same number of col_ao as W         
         PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, cols_ao, i_nonlocal_d, j_nonlocal_d, a_nonlocal_d, &output_mat_nonlocal));

         // We just take a copy of the original garray
         PetscInt *garray_host = NULL; 
         PetscCallVoid(PetscMalloc1(cols_ao, &garray_host));
         for (PetscInt i = 0; i < cols_ao; i++)
         {
            garray_host[i] = mat_mpi->garray[i];
         }

         // We can now create our MPI matrix
         PetscCallVoid(MatCreateMPIAIJWithSeqAIJ(MPI_COMM_MATRIX, global_rows, global_cols_coarse, output_mat_local, output_mat_nonlocal, garray_host, output_mat));
      }     
      // If in serial 
      else
      {
         *output_mat = output_mat_local;
      }
   }

   PetscCallVoid(ISRestoreIndices(*is_fine, &fine_indices_ptr));
   PetscCallVoid(ISRestoreIndices(*is_coarse, &coarse_indices_ptr));

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Stick Z in a full sized R but with kokkos - keeping everything on the device
PETSC_INTERN void compute_R_from_Z_kokkos(Mat *input_mat, PetscInt global_row_start, IS *is_fine, \
                  IS *is_coarse, IS *orig_fine_col_indices, int identity_int, int reuse_int, int reuse_indices_int, \
                  Mat *output_mat)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt global_row_start_Z, global_row_end_plus_one_Z;
   PetscInt global_col_start_Z, global_col_end_plus_one_Z;
   PetscInt local_coarse_size, local_fine_size, local_full_cols;
   PetscInt global_coarse_size, global_fine_size, global_full_cols;
   PetscInt rows_ao, cols_ao, rows_ad, cols_ad, size_cols;
   PetscInt global_rows_z, global_cols_z;
   PetscInt local_rows_z, local_cols_z;
   MatType mat_type;
   PetscInt nnzs_match_local, nnzs_match_nonlocal;
   Mat output_mat_local, output_mat_nonlocal;

   PetscCallVoid(MatGetType(*input_mat, &mat_type));
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
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao);); 

      // We also copy the input mat colmap over to the device as we need it
      colmap_input_h = PetscIntKokkosViewHost(mat_mpi->garray, cols_ao);
      colmap_input_d = PetscIntKokkosView("colmap_input_d", cols_ao);
      Kokkos::deep_copy(colmap_input_d, colmap_input_h);
      // Log copy with petsc
      size_t bytes = colmap_input_h.extent(0) * sizeof(PetscInt);
      PetscCallVoid(PetscLogCpuToGpu(bytes));              
   }
   else
   {
      mat_local = *input_mat;
   }

   // Get the comm
   PetscCallVoid(PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX));
   PetscCallVoid(ISGetLocalSize(*is_coarse, &local_coarse_size));
   PetscCallVoid(ISGetLocalSize(*is_fine, &local_fine_size));
   PetscCallVoid(ISGetSize(*is_coarse, &global_coarse_size));
   PetscCallVoid(ISGetSize(*is_fine, &global_fine_size));

   local_full_cols = local_coarse_size + local_fine_size;
   global_full_cols = global_coarse_size + global_fine_size;

   PetscCallVoid(MatGetLocalSize(*input_mat, &local_rows_z, &local_cols_z));
   PetscCallVoid(MatGetSize(*input_mat, &global_rows_z, &global_cols_z));

   PetscCallVoid(MatGetOwnershipRange(*input_mat, &global_row_start_Z, &global_row_end_plus_one_Z));
   PetscCallVoid(MatGetOwnershipRangeColumn(*input_mat, &global_col_start_Z, &global_col_end_plus_one_Z));

   PetscCallVoid(MatGetType(*input_mat, &mat_type));
   PetscCallVoid(MatGetSize(mat_local, &rows_ad, &cols_ad));

   // We can reuse the orig_fine_col_indices as they can be expensive to generate in parallel
   if (!reuse_indices_int)
   {
      PetscInt *col_indices_off_proc_array;
      IS col_indices;

      // Build these on the host as we need to call host routines 
      // on them anyway, we can transfer the result to the device
      if (mpi)
      {
         PetscCallVoid(PetscMalloc1(cols_ad + cols_ao, &col_indices_off_proc_array));
         size_cols = cols_ad + cols_ao;
         for (PetscInt i = 0; i < cols_ad; i++)
         {
            col_indices_off_proc_array[i] = global_col_start_Z + i;
         }
         for (PetscInt i = 0; i < cols_ao; i++)
         {
            col_indices_off_proc_array[cols_ad + i] = mat_mpi->garray[i];
         }                   
      }
      else
      {
         PetscCallVoid(PetscMalloc1(cols_ad, &col_indices_off_proc_array));
         size_cols = cols_ad;
         for (PetscInt i = 0; i < cols_ad; i++)
         {
            col_indices_off_proc_array[i] = global_col_start_Z + i;
         }
      }

      // Create the IS we want with the cols we want (written as global indices)
      PetscCallVoid(ISCreateGeneral(MPI_COMM_MATRIX, size_cols, col_indices_off_proc_array, PETSC_USE_POINTER, &col_indices));

      // Now let's do the comms to get what the original column indices in the full matrix are, given these indices for all 
      // the columns of Z - ie we need to check in the original fine indices at the positions given by col_indices_off_proc_array
      // This could be expensive as the number of off-processor columns in Z grows!
      PetscCallVoid(ISCreateSubIS(*is_fine, col_indices, orig_fine_col_indices));

      // We've now built the original fine indices
      PetscCallVoid(ISDestroy(&col_indices));
      (void)PetscFree(col_indices_off_proc_array);
   }
   else
   {
      PetscCallVoid(ISGetLocalSize(*orig_fine_col_indices, &size_cols));
   }

   // Get pointers to the indices on the host
   const PetscInt *fine_indices_ptr, *coarse_indices_ptr, *is_pointer_orig_fine_col;
   PetscCallVoid(ISGetIndices(*is_fine, &fine_indices_ptr));   
   PetscCallVoid(ISGetIndices(*is_coarse, &coarse_indices_ptr)); 
   PetscCallVoid(ISGetIndices(*orig_fine_col_indices, &is_pointer_orig_fine_col));     

   // Create a host view of the existing indices
   auto fine_view_h = PetscIntConstKokkosViewHost(fine_indices_ptr, local_fine_size);    
   auto fine_view_d = PetscIntKokkosView("fine_view_d", local_fine_size);   
   auto coarse_view_h = PetscIntConstKokkosViewHost(coarse_indices_ptr, local_coarse_size);    
   auto coarse_view_d = PetscIntKokkosView("coarse_view_d", local_coarse_size);    
   auto orig_view_h = PetscIntConstKokkosViewHost(is_pointer_orig_fine_col, size_cols);    
   auto orig_view_d = PetscIntKokkosView("orig_view_d", size_cols);       
   // Copy indices to the device
   Kokkos::deep_copy(fine_view_d, fine_view_h);     
   // Log copy with petsc
   size_t bytes = fine_view_h.extent(0) * sizeof(PetscInt);
   PetscCallVoid(PetscLogCpuToGpu(bytes));   
   Kokkos::deep_copy(coarse_view_d, coarse_view_h);
   bytes = coarse_view_h.extent(0) * sizeof(PetscInt);
   PetscCallVoid(PetscLogCpuToGpu(bytes));        
   Kokkos::deep_copy(orig_view_d, orig_view_h); 
   bytes = orig_view_h.extent(0) * sizeof(PetscInt);
   PetscCallVoid(PetscLogCpuToGpu(bytes));       

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr, *device_nonlocal_vals = nullptr;  
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype));  
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype));          

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

   // Only need things to do with the sparsity pattern if we're not reusing
   if (!reuse_int)
   {
      // ~~~~~~~~~~~~
      // Get the number of nnzs
      // ~~~~~~~~~~~~
      nnzs_match_local = 0;
      nnzs_match_nonlocal = 0;

      // ~~~~~~~~~~~~~~~~~~~~~~~
      // Let's build our i, j, and a on the device
      // ~~~~~~~~~~~~~~~~~~~~~~~ 
      // We need to know how many entries are in each row 
      nnz_match_local_row_d = PetscIntKokkosView("nnz_match_local_row_d", local_rows_z);    
      // We may have identity
      Kokkos::deep_copy(nnz_match_local_row_d, 0);         
      if (mpi) nnz_match_nonlocal_row_d = PetscIntKokkosView("nnz_match_nonlocal_row_d", local_rows_z);                  

      // ~~~~~~~~~~~~
      // Need to count the number of nnzs we end up with, on each row and in total
      // ~~~~~~~~~~~~
      // Loop over the rows of Z
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows_z), KOKKOS_LAMBDA(PetscInt i) {

            // Row index is simple
            PetscInt row_index = i;
            // Still using i here (the local index into Z)
            const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];
            nnz_match_local_row_d(row_index) = ncols_local;
            // Add one extra in this local block for the identity
            if (identity_int) nnz_match_local_row_d(row_index)++;

            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
               nnz_match_nonlocal_row_d(row_index) = ncols_nonlocal;
            }
      });

      // Get number of nnzs
      Kokkos::parallel_reduce ("ReductionLocal", local_rows_z, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update) {
         update += nnz_match_local_row_d(i); 
      }, nnzs_match_local);   
      if (mpi)
      {
         Kokkos::parallel_reduce ("ReductionNonLocal", local_rows_z, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update) {
            update += nnz_match_nonlocal_row_d(i); 
         }, nnzs_match_nonlocal);       
      }

      // ~~~~~~~~~~~~

      // Need to do a scan on nnz_match_local_row_d to get where each row starts
      Kokkos::parallel_scan (local_rows_z, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
         // Inclusive scan
         update += nnz_match_local_row_d(i);         
         if (final) {
            nnz_match_local_row_d(i) = update; // only update array on final pass
         }
      });      
      if (mpi)
      { 
         // Need to do a scan on nnz_match_nonlocal_row_d to get where each row starts
         Kokkos::parallel_scan (local_rows_z, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
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
      i_local_d = Kokkos::View<PetscInt *>("i_local_d", local_rows_z+1);
      j_local_d = Kokkos::View<PetscInt *>("j_local_d", nnzs_match_local);

      // Initialize first entry to zero - the rest get set below
      Kokkos::deep_copy(Kokkos::subview(i_local_d, 0), 0);       

      // we also have to go and build the a, i, j for the non-local off-diagonal block
      if (mpi) 
      {
         // Non-local 
         a_nonlocal_d = Kokkos::View<PetscScalar *>("a_nonlocal_d", nnzs_match_nonlocal);
         i_nonlocal_d = Kokkos::View<PetscInt *>("i_nonlocal_d", local_rows_z+1);
         j_nonlocal_d = Kokkos::View<PetscInt *>("j_nonlocal_d", nnzs_match_nonlocal);  

         // Initialize first entry to zero - the rest get set below
         Kokkos::deep_copy(Kokkos::subview(i_nonlocal_d, 0), 0);                
      }  

      // ~~~~~~~~~~~~~~~
      // Create i indices
      // ~~~~~~~~~~~~~~~
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows_z), KOKKOS_LAMBDA(PetscInt i) {

            // Row index is simple
            PetscInt row_index = i;       

            // The start of our row index comes from the scan
            i_local_d(row_index + 1) = nnz_match_local_row_d(row_index);   
            if (mpi) i_nonlocal_d(row_index + 1) = nnz_match_nonlocal_row_d(row_index);         
      });            


      // Loop over the rows of Z
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows_z, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            const PetscInt i = t.league_rank();

            // Row index is simple
            PetscInt row_index = i;
            // Still using i here (the local index into Z)
            const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];         

            // For over local columns - copy in Z
            Kokkos::parallel_for(
               Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

               // Want the local col indices for the local block
               // The orig_view_d contains the global indices for the original full matrix
               j_local_d(i_local_d(row_index) + j) = orig_view_d(device_local_j[device_local_i[i] + j]) - global_row_start;
               a_local_d(i_local_d(row_index) + j) = device_local_vals[device_local_i[i] + j];
                     
            });     

            // For over nonlocal columns - copy in Z
            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];         

               Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {

                  // We keep the existing local indices in the off-diagonal block here
                  // we have all the same non-local local column indices as Z (as the identity added is always local)
                  // The garray is the same size, its just the global indices that have changed
                  j_nonlocal_d(i_nonlocal_d(row_index) + j) = device_nonlocal_j[device_nonlocal_i[i] + j];
                  a_nonlocal_d(i_nonlocal_d(row_index) + j) = device_nonlocal_vals[device_nonlocal_i[i] + j];
                        
               });          
            }

            // Only want one thread to deal with the single identity value
            if (identity_int)
            {
               Kokkos::single(Kokkos::PerTeam(t), [&]() {
                  // Let's just stick it at the end and we will sort after
                  // The coarse_view_d contains the global indices for the original full matrix
                  j_local_d(i_local_d(row_index) + ncols_local) = coarse_view_d(i) - global_row_start;
                  a_local_d(i_local_d(row_index) + ncols_local) = 1.0;
               });     
            }         
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

      // Annoyingly there isn't currently the ability to get views for i (or j)
      const PetscInt *device_local_i_output = nullptr, *device_local_j_output = nullptr, *device_nonlocal_i_ouput = nullptr;
      PetscMemType mtype;
      PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_output, &device_local_i_output, &device_local_j_output, NULL, &mtype));  
      if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_output, &device_nonlocal_i_ouput, NULL, NULL, &mtype));  

      // Have these point at the existing i pointers - we only need the local j
      ConstMatRowMapKokkosView i_local_const_d = ConstMatRowMapKokkosView(device_local_i_output, local_rows_z+1);
      ConstMatRowMapKokkosView j_local_const_d = ConstMatRowMapKokkosView(device_local_j_output, aijkok_local_output->csrmat.nnz());
      ConstMatRowMapKokkosView i_nonlocal_const_d;
      if (mpi) i_nonlocal_const_d = ConstMatRowMapKokkosView(device_nonlocal_i_ouput, local_rows_z+1);        

      // Only have to write Z but have to be careful as we have the identity mixed 
      // in there
      // Loop over the rows of Z - annoying we have const views as this is just the same loop as above
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows_z, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            const PetscInt i = t.league_rank();

            // Simple row index
            PetscInt row_index = i;
            // Still using i here (the local index into Z)
            const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];         

            // For over local columns - copy in Z
            // We have to skip over the identity entries, which we know are always C points
            Kokkos::parallel_for(
               Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

               PetscInt offset = 0;

               // If we're at or after the C point identity, our index into R gets a +1
               // so we skip over writing to that index in R
               if (j_local_const_d(i_local_const_d(row_index) + j) >= coarse_view_d(i) - global_row_start) offset = 1;
               a_local_d(i_local_const_d(row_index) + j + offset) = device_local_vals[device_local_i[i] + j];
            });     

            // For over nonlocal columns - copy in Z - identical structure in the off-diag block
            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];         

               Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {

                  // We keep the existing local indices in the off-diagonal block here
                  // we have all the same columns as Z and hence the same garray
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
      PetscCallVoid(PetscObjectStateIncrease((PetscObject)(*output_mat)));

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
      KokkosCsrMatrix csrmat_local = KokkosCsrMatrix("csrmat_local", local_rows_z, local_full_cols, a_local_d.extent(0), a_local_d, i_local_d, j_local_d);  
      KokkosSparse::sort_crs_matrix(csrmat_local);
      
      // Let's make sure everything on the device is finished
      exec = PetscGetKokkosExecutionSpace();
      exec.fence();       
      
      // Create the matrix given the sorted csr
      PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows_z, local_full_cols, i_local_d, j_local_d, a_local_d, &output_mat_local));

      // we also have to go and build our off block matrix and then the output
      if (mpi) 
      {
         // We know the garray is just the original but rewritten to be 
         // the full indices, which we have in in is_pointer_orig_fine_col(cols_ad:end)
         PetscInt *garray_host = NULL; 
         PetscCallVoid(PetscMalloc1(cols_ao, &garray_host));
         for (PetscInt i = 0; i < cols_ao; i++)
         {
            garray_host[i] = is_pointer_orig_fine_col[i + cols_ad];
         }    
         
         // We can create our nonlocal diagonal block matrix directly on the device
         PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows_z, cols_ao, i_nonlocal_d, j_nonlocal_d, a_nonlocal_d, &output_mat_nonlocal));

         // We can now create our MPI matrix
         PetscCallVoid(MatCreateMPIAIJWithSeqAIJ(MPI_COMM_MATRIX, global_rows_z, global_full_cols, output_mat_local, output_mat_nonlocal, garray_host, output_mat));
      }    
      // If in serial 
      else
      {
         *output_mat = output_mat_local;
      }
   }

   PetscCallVoid(ISRestoreIndices(*is_fine, &fine_indices_ptr));
   PetscCallVoid(ISRestoreIndices(*is_coarse, &coarse_indices_ptr));
   PetscCallVoid(ISRestoreIndices(*orig_fine_col_indices, &is_pointer_orig_fine_col));

   return;
}