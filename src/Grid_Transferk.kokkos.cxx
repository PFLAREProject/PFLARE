// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>

struct PflareTraceScope {
   const char *func;
   explicit PflareTraceScope(const char *name) : func(name) {
      fprintf(stderr, "[PFLARE][TRACE] ENTER %s\n", func);
      fflush(stderr);
   }
   ~PflareTraceScope() {
      fprintf(stderr, "[PFLARE][TRACE] EXIT %s\n", func);
      fflush(stderr);
   }
};

static void pflare_guard_seq_csr(Mat seq_mat, PetscInt col_upper_bound, MPI_Comm comm, const char *func, const char *block)
{
   if (!seq_mat) return;

   PetscInt nrows = 0, ncols = 0;
   PetscCallVoid(MatGetLocalSize(seq_mat, &nrows, &ncols));

   const PetscInt *device_i = nullptr, *device_j = nullptr;
   PetscScalar *device_a = nullptr;
   PetscMemType mtype;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(seq_mat, &device_i, &device_j, &device_a, &mtype));

   auto exec = PetscGetKokkosExecutionSpace();
   ConstMatRowMapKokkosView i_d(device_i, nrows + 1);

   PetscInt bad_rowptr = 0;
   Kokkos::parallel_reduce(
      Kokkos::RangePolicy<>(exec, 0, nrows),
      KOKKOS_LAMBDA(const PetscInt i, PetscInt &thread_sum) {
         const PetscInt b = i_d(i), e = i_d(i + 1);
         if (b < 0 || e < b) thread_sum++;
      },
      bad_rowptr);

   auto nnz_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(i_d, nrows));
   const PetscInt nnz = nnz_h();

   PetscInt bad_colidx = 0;
   if (bad_rowptr == 0 && nnz > 0)
   {
      ConstMatColIdxKokkosView j_d(device_j, nnz);
      Kokkos::parallel_reduce(
         Kokkos::RangePolicy<>(exec, 0, nnz),
         KOKKOS_LAMBDA(const PetscInt k, PetscInt &thread_sum) {
            const PetscInt c = j_d(k);
            if (c < 0 || c >= col_upper_bound) thread_sum++;
         },
         bad_colidx);
   }

   if (bad_rowptr > 0 || bad_colidx > 0)
   {
      int rank = -1;
      MPI_Comm_rank(comm, &rank);
      fprintf(stderr,
         "[PFLARE][rank %d] %s CSR anomaly (%s): rowptr_bad=%" PetscInt_FMT ", colidx_bad=%" PetscInt_FMT ", nrows=%" PetscInt_FMT ", ncols=%" PetscInt_FMT ", nnz=%" PetscInt_FMT ", col_upper=%" PetscInt_FMT "\\n",
         rank, func, block, bad_rowptr, bad_colidx, nrows, ncols, nnz, col_upper_bound);
      fflush(stderr);
   }
}

//------------------------------------------------------------------------------------------------------------------------

// Generate one point classical prolongator but with kokkos - keeping everything on the device
PETSC_INTERN void generate_one_point_with_one_entry_from_sparse_kokkos(Mat *input_mat, Mat *output_mat)
{
   PflareTraceScope trace_scope("generate_one_point_with_one_entry_from_sparse_kokkos");
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

   Mat mat_local = NULL, mat_nonlocal = NULL;

   PetscIntConstKokkosViewHost colmap_input_h;
   PetscIntKokkosView colmap_input_d;   
   const PetscInt *colmap_input;
   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*input_mat, &mat_local, &mat_nonlocal, &colmap_input));
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao));

      // We also copy the input mat colmap over to the device as we need it
      colmap_input_h = PetscIntConstKokkosViewHost(colmap_input, cols_ao);
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

   pflare_guard_seq_csr(mat_local, local_cols, MPI_COMM_MATRIX, "generate_one_point_with_one_entry_from_sparse_kokkos", "local");
   if (mpi) pflare_guard_seq_csr(mat_nonlocal, cols_ao, MPI_COMM_MATRIX, "generate_one_point_with_one_entry_from_sparse_kokkos", "nonlocal");

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr, *device_nonlocal_vals = nullptr;  
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype));  
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype));          

   auto exec = PetscGetKokkosExecutionSpace();
   int rank = -1;
   MPI_Comm_rank(MPI_COMM_MATRIX, &rank);

   ConstMatRowMapKokkosView local_i_d(device_local_i, local_rows + 1);
   PetscInt bad_local_rowptr = 0, bad_local_colidx = 0;
   Kokkos::parallel_reduce(
      Kokkos::RangePolicy<>(exec, 0, local_rows),
      KOKKOS_LAMBDA(const PetscInt i, PetscInt &thread_sum) {
         const PetscInt b = local_i_d(i), e = local_i_d(i + 1);
         if (b < 0 || e < b) thread_sum++;
      }, bad_local_rowptr);
   auto local_nnz_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(local_i_d, local_rows));
   const PetscInt local_nnz = local_nnz_h();
   if (local_nnz > 0)
   {
      ConstMatColIdxKokkosView local_j_d(device_local_j, local_nnz);
      Kokkos::parallel_reduce(
         Kokkos::RangePolicy<>(exec, 0, local_nnz),
         KOKKOS_LAMBDA(const PetscInt k, PetscInt &thread_sum) {
            const PetscInt c = local_j_d(k);
            if (c < 0 || c >= local_cols) thread_sum++;
         }, bad_local_colidx);
   }

   PetscInt bad_nonlocal_rowptr = 0, bad_nonlocal_colidx = 0;
   if (mpi)
   {
      ConstMatRowMapKokkosView nonlocal_i_d(device_nonlocal_i, local_rows + 1);
      Kokkos::parallel_reduce(
         Kokkos::RangePolicy<>(exec, 0, local_rows),
         KOKKOS_LAMBDA(const PetscInt i, PetscInt &thread_sum) {
            const PetscInt b = nonlocal_i_d(i), e = nonlocal_i_d(i + 1);
            if (b < 0 || e < b) thread_sum++;
         }, bad_nonlocal_rowptr);
      auto nonlocal_nnz_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(nonlocal_i_d, local_rows));
      const PetscInt nonlocal_nnz = nonlocal_nnz_h();
      if (nonlocal_nnz > 0)
      {
         ConstMatColIdxKokkosView nonlocal_j_d(device_nonlocal_j, nonlocal_nnz);
         Kokkos::parallel_reduce(
            Kokkos::RangePolicy<>(exec, 0, nonlocal_nnz),
            KOKKOS_LAMBDA(const PetscInt k, PetscInt &thread_sum) {
               const PetscInt c = nonlocal_j_d(k);
               if (c < 0 || c >= cols_ao) thread_sum++;
            }, bad_nonlocal_colidx);
      }
   }

   if (bad_local_rowptr > 0 || bad_local_colidx > 0 || bad_nonlocal_rowptr > 0 || bad_nonlocal_colidx > 0)
   {
      fprintf(stderr,
         "[PFLARE][rank %d] generate_one_point_with_one_entry_from_sparse_kokkos CSR anomalies: local_rowptr=%" PetscInt_FMT ", local_colidx=%" PetscInt_FMT ", nonlocal_rowptr=%" PetscInt_FMT ", nonlocal_colidx=%" PetscInt_FMT "\n",
         rank, bad_local_rowptr, bad_local_colidx, bad_nonlocal_rowptr, bad_nonlocal_colidx);
      fflush(stderr);
   }

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

   PetscIntKokkosView invalid_local_read_count_d("generate_one_point_invalid_local_read_count_d", 1);
   PetscIntKokkosView invalid_nonlocal_read_count_d("generate_one_point_invalid_nonlocal_read_count_d", 1);
   PetscIntKokkosView invalid_colmap_read_count_d("generate_one_point_invalid_colmap_read_count_d", 1);
   Kokkos::deep_copy(exec, invalid_local_read_count_d, (PetscInt)0);
   Kokkos::deep_copy(exec, invalid_nonlocal_read_count_d, (PetscInt)0);
   Kokkos::deep_copy(exec, invalid_colmap_read_count_d, (PetscInt)0);

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

            const PetscInt idx = device_local_i[i] + j;
            if (idx < device_local_i[i] || idx >= device_local_i[i + 1])
            {
               Kokkos::atomic_add(&invalid_local_read_count_d(0), (PetscInt)1);
               return;
            }

            const PetscInt c = device_local_j[idx];
            if (c < 0 || c >= local_cols)
            {
               Kokkos::atomic_add(&invalid_local_read_count_d(0), (PetscInt)1);
               return;
            }

            // If it's the biggest value keep it
            if (Kokkos::abs(device_local_vals[idx]) > thread_data.val) {
               thread_data.val = Kokkos::abs(device_local_vals[idx]);
               thread_data.col = c;
            }
         }, local_row_result
      );

      if (mpi)
      {
         PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
         Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(t, ncols_nonlocal),
            [&](const PetscInt j, ReduceDataMaxRow& thread_data) {

               const PetscInt idx = device_nonlocal_i[i] + j;
               if (idx < device_nonlocal_i[i] || idx >= device_nonlocal_i[i + 1])
               {
                  Kokkos::atomic_add(&invalid_nonlocal_read_count_d(0), (PetscInt)1);
                  return;
               }

               const PetscInt c_local = device_nonlocal_j[idx];
               if (c_local < 0 || c_local >= cols_ao)
               {
                  Kokkos::atomic_add(&invalid_nonlocal_read_count_d(0), (PetscInt)1);
                  return;
               }

               // If it's the biggest value keep it
               if (Kokkos::abs(device_nonlocal_vals[idx]) > thread_data.val) {
                  thread_data.val = Kokkos::abs(device_nonlocal_vals[idx]);
                  // Set the global index
                  if (c_local >= 0 && c_local < (PetscInt)colmap_input_d.extent(0))
                  {
                     thread_data.col = colmap_input_d(c_local);
                  }
                  else
                  {
                     Kokkos::atomic_add(&invalid_colmap_read_count_d(0), (PetscInt)1);
                  }
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

   exec.fence();
   auto invalid_local_read_count_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), invalid_local_read_count_d);
   auto invalid_nonlocal_read_count_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), invalid_nonlocal_read_count_d);
   auto invalid_colmap_read_count_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), invalid_colmap_read_count_d);
   if (invalid_local_read_count_h(0) > 0 || invalid_nonlocal_read_count_h(0) > 0 || invalid_colmap_read_count_h(0) > 0)
   {
      fprintf(stderr,
         "[PFLARE][rank %d] generate_one_point_with_one_entry_from_sparse_kokkos invalid reads local=%" PetscInt_FMT ", nonlocal=%" PetscInt_FMT ", colmap=%" PetscInt_FMT "\\n",
         rank, invalid_local_read_count_h(0), invalid_nonlocal_read_count_h(0), invalid_colmap_read_count_h(0));
      fflush(stderr);
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
   PetscIntKokkosView invalid_fill_write_local_count_d("generate_one_point_invalid_fill_write_local_count_d", 1);
   PetscIntKokkosView invalid_fill_write_nonlocal_count_d("generate_one_point_invalid_fill_write_nonlocal_count_d", 1);
   Kokkos::deep_copy(exec, invalid_fill_write_local_count_d, (PetscInt)0);
   Kokkos::deep_copy(exec, invalid_fill_write_nonlocal_count_d, (PetscInt)0);

   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt i) {

      // If our max val is in the local block
      if (has_entry_local_d(i) > 0) {
         const PetscInt out_idx = i_local_d(i);
         const PetscInt out_col = max_col_row_d(i);
         if (out_idx < 0 || out_idx >= nnzs_match_local || out_col < 0 || out_col >= local_cols)
         {
            Kokkos::atomic_add(&invalid_fill_write_local_count_d(0), (PetscInt)1);
         }
         else
         {
            j_local_d(out_idx) = out_col;
            a_local_d(out_idx) = 1.0;
         }
      }
      else if (mpi && has_entry_nonlocal_d(i) > 0)
      {
         const PetscInt out_idx = i_nonlocal_d(i);
         const PetscInt out_col = max_col_row_d(i);
         if (out_idx < 0 || out_idx >= nnzs_match_nonlocal || out_col < 0 || out_col >= global_cols)
         {
            Kokkos::atomic_add(&invalid_fill_write_nonlocal_count_d(0), (PetscInt)1);
         }
         else
         {
            j_nonlocal_d(out_idx) = out_col;
            a_nonlocal_d(out_idx) = 1.0;
         }
      }   
   });      

   // Let's make sure everything on the device is finished
   exec.fence();
   auto invalid_fill_write_local_count_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), invalid_fill_write_local_count_d);
   auto invalid_fill_write_nonlocal_count_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), invalid_fill_write_nonlocal_count_d);
   if (invalid_fill_write_local_count_h(0) > 0 || invalid_fill_write_nonlocal_count_h(0) > 0)
   {
      fprintf(stderr,
         "[PFLARE][rank %d] generate_one_point_with_one_entry_from_sparse_kokkos invalid writes local=%" PetscInt_FMT ", nonlocal=%" PetscInt_FMT "\\n",
         rank, invalid_fill_write_local_count_h(0), invalid_fill_write_nonlocal_count_h(0));
      fflush(stderr);
   }
   
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
      // This routine fences internally
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
   PflareTraceScope trace_scope("compute_P_from_W_kokkos");
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
   Mat mat_local = NULL, mat_nonlocal = NULL;
   const PetscInt *colmap_input;
 
   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*input_mat, &mat_local, &mat_nonlocal, &colmap_input));
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao));
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

   PetscInt local_rows_w = 0, local_cols_w = 0;
   PetscCallVoid(MatGetLocalSize(*input_mat, &local_rows_w, &local_cols_w));
   pflare_guard_seq_csr(mat_local, local_cols_w, MPI_COMM_MATRIX, "compute_P_from_W_kokkos", "local");
   if (mpi) pflare_guard_seq_csr(mat_nonlocal, cols_ao, MPI_COMM_MATRIX, "compute_P_from_W_kokkos", "nonlocal");

   if (local_rows_w != local_rows_fine)
   {
      int rank = -1;
      MPI_Comm_rank(MPI_COMM_MATRIX, &rank);
      fprintf(stderr,
         "[PFLARE][rank %d] compute_P_from_W_kokkos row mismatch: W_local_rows=%" PetscInt_FMT ", local_rows_fine=%" PetscInt_FMT "\n",
         rank, local_rows_w, local_rows_fine);
      fflush(stderr);
   }

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
   Mat mat_local_output = NULL, mat_nonlocal_output = NULL;   

   auto exec = PetscGetKokkosExecutionSpace();

   int rank = -1;
   MPI_Comm_rank(MPI_COMM_MATRIX, &rank);
   PetscInt invalid_fine_rows = 0, invalid_coarse_rows = 0;
   if (local_rows_fine > 0)
   {
      Kokkos::parallel_reduce(
         Kokkos::RangePolicy<>(exec, 0, local_rows_fine),
         KOKKOS_LAMBDA(const PetscInt i, PetscInt &thread_sum) {
            const PetscInt row_index = fine_view_d(i) - global_row_start;
            if (row_index < 0 || row_index >= local_rows) thread_sum++;
         }, invalid_fine_rows);
   }
   if (local_rows_coarse > 0)
   {
      Kokkos::parallel_reduce(
         Kokkos::RangePolicy<>(exec, 0, local_rows_coarse),
         KOKKOS_LAMBDA(const PetscInt i, PetscInt &thread_sum) {
            const PetscInt row_index = coarse_view_d(i) - global_row_start;
            if (row_index < 0 || row_index >= local_rows) thread_sum++;
         }, invalid_coarse_rows);
   }
   if (invalid_fine_rows > 0 || invalid_coarse_rows > 0)
   {
      fprintf(stderr,
         "[PFLARE][rank %d] compute_P_from_W_kokkos invalid mapped rows: fine_bad=%" PetscInt_FMT ", coarse_bad=%" PetscInt_FMT " (local_rows=%" PetscInt_FMT ")\n",
         rank, invalid_fine_rows, invalid_coarse_rows, local_rows);
      fflush(stderr);
   }

   // Only need things to do with the sparsity pattern if we're not reusing
   if (!reuse_int)
   {
      PetscIntKokkosView invalid_row_map_count_d("compute_P_invalid_row_map_count_d", 1);
      PetscIntKokkosView invalid_write_count_d("compute_P_invalid_write_count_d", 1);
      Kokkos::deep_copy(exec, invalid_row_map_count_d, (PetscInt)0);
      Kokkos::deep_copy(exec, invalid_write_count_d, (PetscInt)0);

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
            if (row_index < 0 || row_index >= local_rows)
            {
               Kokkos::atomic_add(&invalid_row_map_count_d(0), (PetscInt)1);
               return;
            }
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
               if (row_index < 0 || row_index >= local_rows)
               {
                  Kokkos::atomic_add(&invalid_row_map_count_d(0), (PetscInt)1);
                  return;
               }
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
            if (row_index < 0 || row_index >= local_rows)
            {
               Kokkos::atomic_add(&invalid_row_map_count_d(0), (PetscInt)1);
               return;
            }

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
         if (row_index < 0 || row_index >= local_rows)
         {
            Kokkos::atomic_add(&invalid_row_map_count_d(0), (PetscInt)1);
            return;
         }

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
            if (row_index < 0 || row_index >= local_rows)
            {
               Kokkos::single(Kokkos::PerTeam(t), [&]() {
                  Kokkos::atomic_add(&invalid_row_map_count_d(0), (PetscInt)1);
               });
               return;
            }
            // Still using i here (the local index into W)
            const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];         

            // For over local columns - copy in W
            Kokkos::parallel_for(
               Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {
               const PetscInt out_idx = i_local_d(row_index) + j;
               const PetscInt out_col = device_local_j[device_local_i[i] + j];
               if (out_idx < 0 || out_idx >= nnzs_match_local || out_col < 0 || out_col >= local_cols)
               {
                  Kokkos::atomic_add(&invalid_write_count_d(0), (PetscInt)1);
               }
               else
               {
                  j_local_d(out_idx) = out_col;
                  a_local_d(out_idx) = device_local_vals[device_local_i[i] + j];
               }
                     
            });     

            // For over nonlocal columns - copy in W
            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];         

               Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {

                  // We keep the existing local indices in the off-diagonal block here
                  // we have all the same columns as W and hence the same garray
                  const PetscInt out_idx = i_nonlocal_d(row_index) + j;
                  const PetscInt out_col = device_nonlocal_j[device_nonlocal_i[i] + j];
                  if (out_idx < 0 || out_idx >= nnzs_match_nonlocal || out_col < 0 || out_col >= cols_ao)
                  {
                     Kokkos::atomic_add(&invalid_write_count_d(0), (PetscInt)1);
                  }
                  else
                  {
                     j_nonlocal_d(out_idx) = out_col;
                     a_nonlocal_d(out_idx) = device_nonlocal_vals[device_nonlocal_i[i] + j];
                  }
                        
               });          
            }
      }); 

      exec.fence();
      auto invalid_row_map_count_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), invalid_row_map_count_d);
      auto invalid_write_count_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), invalid_write_count_d);
      if (invalid_row_map_count_h(0) > 0 || invalid_write_count_h(0) > 0)
      {
         fprintf(stderr,
            "[PFLARE][rank %d] compute_P_from_W_kokkos invalid row maps=%" PetscInt_FMT ", invalid writes=%" PetscInt_FMT "\\n",
            rank, invalid_row_map_count_h(0), invalid_write_count_h(0));
         fflush(stderr);
      }
   }
   // If we're reusing, we can just write directly to the existing views
   else
   {
      // Get the existing output mats
      if (mpi)
      {  
         PetscCallVoid(MatMPIAIJGetSeqAIJ(*output_mat, &mat_local_output, &mat_nonlocal_output, NULL));
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
      pflare_guard_seq_csr(mat_local_output, local_cols, MPI_COMM_MATRIX, "compute_P_from_W_kokkos", "local_out_reuse");
      if (mpi) pflare_guard_seq_csr(mat_nonlocal_output, cols_ao, MPI_COMM_MATRIX, "compute_P_from_W_kokkos", "nonlocal_out_reuse");
      PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_output, &device_local_i_output, NULL, NULL, &mtype));  
      if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_output, &device_nonlocal_i_ouput, NULL, NULL, &mtype));  

      // Have these point at the existing i pointers - we don't need j if we're reusing
      ConstMatRowMapKokkosView i_local_const_d = ConstMatRowMapKokkosView(device_local_i_output, local_rows+1);
      ConstMatRowMapKokkosView i_nonlocal_const_d;
      if (mpi) i_nonlocal_const_d = ConstMatRowMapKokkosView(device_nonlocal_i_ouput, local_rows+1);        

      // Only have to write W as the identity block cannot change
      PetscIntKokkosView invalid_reuse_write_count_d("compute_P_reuse_invalid_write_count_d", 1);
      Kokkos::deep_copy(exec, invalid_reuse_write_count_d, (PetscInt)0);

      // Loop over the rows of W - annoying we have const views as this is just the same loop as above
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows_fine, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            const PetscInt i = t.league_rank();

            // Convert to global fine index into a local index in the full matrix
            PetscInt row_index = fine_view_d(i) - global_row_start;
            if (row_index < 0 || row_index >= local_rows)
            {
               Kokkos::single(Kokkos::PerTeam(t), [&]() {
                  Kokkos::atomic_add(&invalid_reuse_write_count_d(0), (PetscInt)1);
               });
               return;
            }
            // Still using i here (the local index into W)
            const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];         

            // For over local columns - copy in W
            Kokkos::parallel_for(
               Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {
               const PetscInt out_idx = i_local_const_d(row_index) + j;
               if (out_idx < 0 || out_idx >= (PetscInt)a_local_d.extent(0))
               {
                  Kokkos::atomic_add(&invalid_reuse_write_count_d(0), (PetscInt)1);
               }
               else
               {
                  a_local_d(out_idx) = device_local_vals[device_local_i[i] + j];
               }
                     
            });     

            // For over nonlocal columns - copy in W
            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];         

               Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {

                  // We keep the existing local indices in the off-diagonal blocl here
                  // we have all the same columns as W and hence the same garray
                  const PetscInt out_idx = i_nonlocal_const_d(row_index) + j;
                  if (out_idx < 0 || out_idx >= (PetscInt)a_nonlocal_d.extent(0))
                  {
                     Kokkos::atomic_add(&invalid_reuse_write_count_d(0), (PetscInt)1);
                  }
                  else
                  {
                     a_nonlocal_d(out_idx) = device_nonlocal_vals[device_nonlocal_i[i] + j];
                  }
                        
               });          
            }
      });   

      exec.fence();
      auto invalid_reuse_write_count_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), invalid_reuse_write_count_d);
      if (invalid_reuse_write_count_h(0) > 0)
      {
         fprintf(stderr,
            "[PFLARE][rank %d] compute_P_from_W_kokkos reuse invalid writes=%" PetscInt_FMT "\\n",
            rank, invalid_reuse_write_count_h(0));
         fflush(stderr);
      }

      // Let's make sure everything on the device is finished
      exec.fence();      

      // Have to specify we've modifed data on the device
      // Want to call MatSeqAIJKokkosModifyDevice but its PETSC_INTERN
      aijkok_local_output->a_dual.clear_sync_state();
      aijkok_local_output->a_dual.modify_device();
      aijkok_local_output->transpose_updated = PETSC_FALSE;
      aijkok_local_output->hermitian_updated = PETSC_FALSE;
      // Invalidate diagonals
      if (mpi)
      {
         aijkok_nonlocal_output->a_dual.clear_sync_state();
         aijkok_nonlocal_output->a_dual.modify_device();
         aijkok_nonlocal_output->transpose_updated = PETSC_FALSE;
         aijkok_nonlocal_output->hermitian_updated = PETSC_FALSE;
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
         PetscIntKokkosView invalid_identity_write_count_d("compute_P_invalid_identity_write_count_d", 1);
         Kokkos::deep_copy(exec, invalid_identity_write_count_d, (PetscInt)0);

         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, local_rows_coarse), KOKKOS_LAMBDA(PetscInt i) {

            // Convert to global coarse index into a local index into the full matrix
            PetscInt row_index = coarse_view_d(i) - global_row_start;
            if (row_index < 0 || row_index >= local_rows)
            {
               Kokkos::atomic_add(&invalid_identity_write_count_d(0), (PetscInt)1);
               return;
            }

            const PetscInt out_idx = i_local_d(row_index);
            if (out_idx < 0 || out_idx >= nnzs_match_local || i < 0 || i >= local_cols_coarse)
            {
               Kokkos::atomic_add(&invalid_identity_write_count_d(0), (PetscInt)1);
               return;
            }

            // Only a single column
            j_local_d(out_idx) = i;
            a_local_d(out_idx) = 1.0;         

         }); 

         exec.fence();
         auto invalid_identity_write_count_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), invalid_identity_write_count_d);
         if (invalid_identity_write_count_h(0) > 0)
         {
            fprintf(stderr,
               "[PFLARE][rank %d] compute_P_from_W_kokkos invalid identity writes=%" PetscInt_FMT "\\n",
               rank, invalid_identity_write_count_h(0));
            fflush(stderr);
         }
      }   
        
      // Let's make sure everything on the device is finished
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
            garray_host[i] = colmap_input[i];
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
   PflareTraceScope trace_scope("compute_R_from_Z_kokkos");
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
   Mat mat_local = NULL, mat_nonlocal = NULL;

   PetscIntConstKokkosViewHost colmap_input_h;
   PetscIntKokkosView colmap_input_d;   
   const PetscInt *colmap_input;
   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*input_mat, &mat_local, &mat_nonlocal, &colmap_input));
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao);); 

      // We also copy the input mat colmap over to the device as we need it
      colmap_input_h = PetscIntConstKokkosViewHost(colmap_input, cols_ao);
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
            col_indices_off_proc_array[cols_ad + i] = colmap_input[i];
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
   pflare_guard_seq_csr(mat_local, local_cols_z, MPI_COMM_MATRIX, "compute_R_from_Z_kokkos", "local");
   if (mpi) pflare_guard_seq_csr(mat_nonlocal, cols_ao, MPI_COMM_MATRIX, "compute_R_from_Z_kokkos", "nonlocal");

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
   Mat mat_local_output = NULL, mat_nonlocal_output = NULL;  
   
   auto exec = PetscGetKokkosExecutionSpace();   

   int rank = -1;
   MPI_Comm_rank(MPI_COMM_MATRIX, &rank);
   PetscInt invalid_orig_col_idx = 0, invalid_orig_local_map_idx = 0;
   if (size_cols > 0)
   {
      Kokkos::parallel_reduce(
         Kokkos::RangePolicy<>(exec, 0, size_cols),
         KOKKOS_LAMBDA(const PetscInt i, PetscInt &thread_sum) {
            const PetscInt idx = orig_view_d(i);
            if (idx < 0 || idx >= global_full_cols) thread_sum++;
         }, invalid_orig_col_idx);
   }
   if (local_cols_z > 0)
   {
      const PetscInt orig_local_check_n = PetscMin(local_cols_z, size_cols);
      if (local_cols_z > size_cols)
      {
         fprintf(stderr,
            "[PFLARE][rank %d] compute_R_from_Z_kokkos size mismatch: local_cols_z=%" PetscInt_FMT " > size_cols=%" PetscInt_FMT " (clamping local-map check)\n",
            rank, local_cols_z, size_cols);
         fflush(stderr);
      }
      Kokkos::parallel_reduce(
         Kokkos::RangePolicy<>(exec, 0, orig_local_check_n),
         KOKKOS_LAMBDA(const PetscInt i, PetscInt &thread_sum) {
            const PetscInt idx = orig_view_d(i);
            if (idx < global_row_start || idx >= global_row_start + local_full_cols) thread_sum++;
         }, invalid_orig_local_map_idx);
   }
   if (invalid_orig_col_idx > 0)
   {
      fprintf(stderr,
         "[PFLARE][rank %d] compute_R_from_Z_kokkos invalid orig column domain entries=%" PetscInt_FMT " (size_cols=%" PetscInt_FMT ", global_full_cols=%" PetscInt_FMT ")\n",
         rank, invalid_orig_col_idx, size_cols, global_full_cols);
      fflush(stderr);
   }
   if (invalid_orig_local_map_idx > 0)
   {
      fprintf(stderr,
         "[PFLARE][rank %d] compute_R_from_Z_kokkos invalid local mapped columns=%" PetscInt_FMT " (local_cols_z=%" PetscInt_FMT ", row_start=%" PetscInt_FMT ", local_full_cols=%" PetscInt_FMT ")\n",
         rank, invalid_orig_local_map_idx, local_cols_z, global_row_start, local_full_cols);
      fflush(stderr);
   }

   PetscIntKokkosView invalid_local_map_write_count_d("invalid_local_map_write_count_d", 1);
   Kokkos::deep_copy(exec, invalid_local_map_write_count_d, (PetscInt)0);
   PetscIntKokkosView invalid_orig_lookup_count_d("compute_R_invalid_orig_lookup_count_d", 1);
   Kokkos::deep_copy(exec, invalid_orig_lookup_count_d, (PetscInt)0);
   PetscIntKokkosView invalid_reuse_write_count_d("compute_R_reuse_invalid_write_count_d", 1);
   Kokkos::deep_copy(exec, invalid_reuse_write_count_d, (PetscInt)0);

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
               const PetscInt col_lookup = device_local_j[device_local_i[i] + j];
               if (col_lookup < 0 || col_lookup >= size_cols)
               {
                  Kokkos::atomic_add(&invalid_orig_lookup_count_d(0), (PetscInt)1);
                  j_local_d(i_local_d(row_index) + j) = 0;
                  a_local_d(i_local_d(row_index) + j) = 0.0;
                  return;
               }
               const PetscInt mapped_col_local = orig_view_d(col_lookup) - global_row_start;
               if (mapped_col_local < 0 || mapped_col_local >= local_full_cols)
               {
                  Kokkos::atomic_add(&invalid_local_map_write_count_d(0), (PetscInt)1);
                  j_local_d(i_local_d(row_index) + j) = 0;
                  a_local_d(i_local_d(row_index) + j) = 0.0;
               }
               else
               {
                  j_local_d(i_local_d(row_index) + j) = mapped_col_local;
                  a_local_d(i_local_d(row_index) + j) = device_local_vals[device_local_i[i] + j];
               }
                     
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
                  if (i >= 0 && i < local_coarse_size)
                  {
                     j_local_d(i_local_d(row_index) + ncols_local) = coarse_view_d(i) - global_row_start;
                     a_local_d(i_local_d(row_index) + ncols_local) = 1.0;
                  }
                  else
                  {
                     Kokkos::atomic_add(&invalid_orig_lookup_count_d(0), (PetscInt)1);
                  }
               });     
            }         
      }); 

      exec.fence();
      auto invalid_local_map_write_count_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), invalid_local_map_write_count_d);
      auto invalid_orig_lookup_count_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), invalid_orig_lookup_count_d);
      if (invalid_local_map_write_count_h(0) > 0 || invalid_orig_lookup_count_h(0) > 0)
      {
         fprintf(stderr,
            "[PFLARE][rank %d] compute_R_from_Z_kokkos invalid mapped writes=%" PetscInt_FMT ", invalid orig lookups=%" PetscInt_FMT "\n",
            rank, invalid_local_map_write_count_h(0), invalid_orig_lookup_count_h(0));
         fflush(stderr);
      }
   }
   // If we're reusing, we can just write directly to the existing views
   else
   {
      // Get the existing output mats
      if (mpi)
      {
         PetscCallVoid(MatMPIAIJGetSeqAIJ(*output_mat, &mat_local_output, &mat_nonlocal_output, NULL));
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
      pflare_guard_seq_csr(mat_local_output, local_full_cols, MPI_COMM_MATRIX, "compute_R_from_Z_kokkos", "local_out_reuse");
      if (mpi) pflare_guard_seq_csr(mat_nonlocal_output, cols_ao, MPI_COMM_MATRIX, "compute_R_from_Z_kokkos", "nonlocal_out_reuse");
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
               const PetscInt out_idx = i_local_const_d(row_index) + j + offset;
               if (out_idx < 0 || out_idx >= (PetscInt)a_local_d.extent(0))
               {
                  Kokkos::atomic_add(&invalid_reuse_write_count_d(0), (PetscInt)1);
               }
               else
               {
                  a_local_d(out_idx) = device_local_vals[device_local_i[i] + j];
               }
            });     

            // For over nonlocal columns - copy in Z - identical structure in the off-diag block
            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];         

               Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {

                  // We keep the existing local indices in the off-diagonal block here
                  // we have all the same columns as Z and hence the same garray
                  const PetscInt out_idx = i_nonlocal_const_d(row_index) + j;
                  if (out_idx < 0 || out_idx >= (PetscInt)a_nonlocal_d.extent(0))
                  {
                     Kokkos::atomic_add(&invalid_reuse_write_count_d(0), (PetscInt)1);
                  }
                  else
                  {
                     a_nonlocal_d(out_idx) = device_nonlocal_vals[device_nonlocal_i[i] + j];
                  }
                        
               });          
            }
      });   

      exec.fence();
      auto invalid_reuse_write_count_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), invalid_reuse_write_count_d);
      if (invalid_reuse_write_count_h(0) > 0)
      {
         fprintf(stderr,
            "[PFLARE][rank %d] compute_R_from_Z_kokkos reuse invalid writes=%" PetscInt_FMT "\n",
            rank, invalid_reuse_write_count_h(0));
         fflush(stderr);
      }

      exec.fence();

      // Have to specify we've modifed data on the device
      // Want to call MatSeqAIJKokkosModifyDevice but its PETSC_INTERN
      aijkok_local_output->a_dual.clear_sync_state();
      aijkok_local_output->a_dual.modify_device();
      aijkok_local_output->transpose_updated = PETSC_FALSE;
      aijkok_local_output->hermitian_updated = PETSC_FALSE;
      // Invalidate diagonals
      if (mpi)
      {
         aijkok_nonlocal_output->a_dual.clear_sync_state();
         aijkok_nonlocal_output->a_dual.modify_device();
         aijkok_nonlocal_output->transpose_updated = PETSC_FALSE;
         aijkok_nonlocal_output->hermitian_updated = PETSC_FALSE;
      }        
      PetscCallVoid(PetscObjectStateIncrease((PetscObject)(*output_mat)));

    }

   // ~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~   

   if (!reuse_int)
   {
      // Let's make sure everything on the device is finished
      exec.fence();   

      // Now we have to sort the local column indices, as we add in the identity at the 
      // end of our local j indices      
      KokkosCsrMatrix csrmat_local = KokkosCsrMatrix("csrmat_local", local_rows_z, local_full_cols, a_local_d.extent(0), a_local_d, i_local_d, j_local_d);  
      KokkosSparse::sort_crs_matrix(csrmat_local);
      
      // Let's make sure everything on the device is finished
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
