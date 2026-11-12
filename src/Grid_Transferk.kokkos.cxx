// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>

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

   Mat mat_local = NULL, mat_nonlocal = NULL;

   PetscIntConstKokkosViewHost colmap_input_h;
   PetscIntKokkosView colmap_input_d;   
   const PetscInt *colmap_input;
   auto exec = PetscGetKokkosExecutionSpace();

   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*input_mat, &mat_local, &mat_nonlocal, &colmap_input));
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao));

      // We also copy the input mat colmap over to the device as we need it
      colmap_input_h = PetscIntConstKokkosViewHost(colmap_input, cols_ao);
      colmap_input_d = PetscIntKokkosView("colmap_input_d", cols_ao);
      Kokkos::deep_copy(exec, colmap_input_d, colmap_input_h);
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
   // Get pointers to the i,j on the device and Kokkos views to the values
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, NULL, &mtype));
   Kokkos::View<const PetscScalar *> device_local_vals;
   Kokkos::View<const PetscScalar *> device_nonlocal_vals;
   PetscCallVoid(MatSeqAIJGetKokkosView(mat_local, &device_local_vals));
   if (mpi) PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal, &device_nonlocal_vals));

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
   Kokkos::deep_copy(exec, nnz_match_local_row_d, 0);
   PetscIntKokkosView nnz_match_nonlocal_row_d;
   if (mpi) 
   {
      nnz_match_nonlocal_row_d = PetscIntKokkosView("nnz_match_nonlocal_row_d", local_rows); 
      Kokkos::deep_copy(exec, nnz_match_nonlocal_row_d, 0);
   }

   // Loop over the rows and find the biggest entry in each row
   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

      const PetscInt i   = t.league_rank(); // row i
      const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];

      // We have a custom reduction type defined - ReduceDataMaxRow
      ReduceDataMaxRow local_row_result, nonlocal_row_result;

      // Reduce over all the columns
      Kokkos::parallel_reduce(
         Kokkos::TeamThreadRange(t, ncols_local),
         [&](const PetscInt j, ReduceDataMaxRow& thread_data) {

            // If it's the biggest value keep it - ties go to the smallest column
            // to match the CPU maxloc
            const PetscReal abs_val = Kokkos::abs(device_local_vals(device_local_i[i] + j));
            const PetscInt col = device_local_j[device_local_i[i] + j];
            if (thread_data.is_better(abs_val, col)) {
               thread_data.val = abs_val;
               thread_data.col = col;
            }
         }, local_row_result
      );

      if (mpi)
      {
         PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
         Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(t, ncols_nonlocal),
            [&](const PetscInt j, ReduceDataMaxRow& thread_data) {

               // If it's the biggest value keep it - ties go to the smallest column
               // (colmap is sorted, so this matches the CPU maxloc)
               const PetscReal abs_val = Kokkos::abs(device_nonlocal_vals(device_nonlocal_i[i] + j));
               // Global index
               const PetscInt col = colmap_input_d(device_nonlocal_j[device_nonlocal_i[i] + j]);
               if (thread_data.is_better(abs_val, col)) {
                  thread_data.val = abs_val;
                  thread_data.col = col;
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
   Kokkos::parallel_reduce ("ReductionLocal", Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA (const PetscInt i, PetscInt& update) {
      update += nnz_match_local_row_d(i);
   }, nnzs_match_local);
   if (mpi)
   {
      Kokkos::parallel_reduce ("ReductionNonLocal", Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA (const PetscInt i, PetscInt& update) {
         update += nnz_match_nonlocal_row_d(i);
      }, nnzs_match_nonlocal);
   }   

   // ~~~~~~~~~~~~

   // Store original counts before scan
   PetscIntKokkosView has_entry_local_d("has_entry_local_d", local_rows);
   Kokkos::deep_copy(exec, has_entry_local_d, nnz_match_local_row_d); 
   PetscIntKokkosView has_entry_nonlocal_d;
   if (mpi)
   {
      has_entry_nonlocal_d = PetscIntKokkosView ("has_entry_nonlocal_d", local_rows);
      Kokkos::deep_copy(exec, has_entry_nonlocal_d, nnz_match_nonlocal_row_d);
   }  

   // Need to do a scan on nnz_match_local_row_d to get where each row starts
   Kokkos::parallel_scan (Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
      // Inclusive scan
      update += nnz_match_local_row_d(i);
      if (final) {
         nnz_match_local_row_d(i) = update; // only update array on final pass
      }
   });

   if (mpi)
   {
      // Need to do a scan on nnz_match_nonlocal_row_d to get where each row starts
      Kokkos::parallel_scan (Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
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
   Kokkos::deep_copy(exec, Kokkos::subview(i_local_d, 0), 0);       

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
      Kokkos::deep_copy(exec, Kokkos::subview(i_nonlocal_d, 0), 0);                
   }        

   // Initialize i_local_d row pointers (1 to local_rows) with cumulative sums from the scan
   PetscInt one = 1;
   auto i_local_range = Kokkos::subview(i_local_d, Kokkos::make_pair(one, local_rows+1));
   Kokkos::deep_copy(exec, i_local_range, nnz_match_local_row_d);
   
   // Similarly for MPI nonlocal case if needed
   if (mpi) {
      auto i_nonlocal_range = Kokkos::subview(i_nonlocal_d, Kokkos::make_pair(one, local_rows+1));
      Kokkos::deep_copy(exec, i_nonlocal_range, nnz_match_nonlocal_row_d);
   }          
   
   // Filling the matrix is easy as we know we only have one non-zero per row
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {

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
   Kokkos::fence();
   
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
      // This fences internally
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

   PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local, &device_local_vals));
   if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal, &device_nonlocal_vals));

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Row kernel for generate_one_point_P_from_sparse_kokkos
// Each team handles team_size fine rows, one thread per row, with vector lanes over the
// entries in that row
struct OnePointMaxRowFunctor {
   PetscInt nrows = 0;
   PetscInt global_row_start = 0;
   bool mpi = false;

   // Input local and nonlocal blocks of A_fc, colmap_d gives the global column of a nonlocal column
   const PetscInt *l_i = nullptr, *l_j = nullptr, *nl_i = nullptr, *nl_j = nullptr;
   Kokkos::View<const PetscScalar *> l_a, nl_a;
   PetscIntKokkosView colmap_d;
   // Global row of each fine row in the full matrix
   PetscIntKokkosView fine_d;

   // Written for the row of P each fine row becomes: the column of the one entry (local
   // for the local block, global for the nonlocal block), and a count of 1 in
   // counts_l[row + 1] or counts_nl[row + 1] for the block it goes in
   PetscIntKokkosView col_d;
   PetscInt *counts_l = nullptr, *counts_nl = nullptr;

   KOKKOS_INLINE_FUNCTION void operator()(const KokkosTeamMemberType &t) const
   {
      const PetscInt row_start = t.league_rank() * t.team_size();
      const PetscInt row_end = row_start + t.team_size() < nrows ? row_start + t.team_size() : nrows;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, row_start, row_end), [&](const PetscInt i) {
         const PetscInt l_start = l_i[i], l_len = l_i[i + 1] - l_start;
         const PetscInt nl_start = mpi ? nl_i[i] : 0, nl_len = mpi ? nl_i[i + 1] - nl_start : 0;

         // The biggest entry in each block - ties go to the smallest column to match
         // the CPU maxloc, whatever order the lanes are joined in
         ReduceDataMaxRow local_row_result, nonlocal_row_result;
         Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, l_len),
            [&](const PetscInt k, ReduceDataMaxRow &lane_data) {
               const PetscReal abs_val = Kokkos::abs(l_a(l_start + k));
               const PetscInt col = l_j[l_start + k];
               if (lane_data.is_better(abs_val, col)) {
                  lane_data.val = abs_val;
                  lane_data.col = col;
               }
            }, local_row_result);
         if (mpi) {
            // colmap is sorted, so the smallest global column is the smallest local one
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, nl_len),
               [&](const PetscInt k, ReduceDataMaxRow &lane_data) {
                  const PetscReal abs_val = Kokkos::abs(nl_a(nl_start + k));
                  const PetscInt col = colmap_d(nl_j[nl_start + k]);
                  if (lane_data.is_better(abs_val, col)) {
                     lane_data.val = abs_val;
                     lane_data.col = col;
                  }
               }, nonlocal_row_result);
         }

         // Same choice as generate_one_point_with_one_entry_from_sparse_kokkos
         Kokkos::single(Kokkos::PerThread(t), [&]() {
            const PetscInt row_index = fine_d(i) - global_row_start;
            bool use_local = false, use_nonlocal = false;
            if (!mpi) {
               use_local = local_row_result.col != -1;
            }
            else if (nonlocal_row_result.val > local_row_result.val) {
               use_nonlocal = nonlocal_row_result.col != -1;
            }
            else if (nonlocal_row_result.val < local_row_result.val) {
               use_local = local_row_result.col != -1;
            }
            else {
               // Equal, pick the local entry if both are valid
               use_local = local_row_result.col != -1 && nonlocal_row_result.col != -1;
            }
            if (use_local) {
               col_d(row_index) = local_row_result.col;
               counts_l[row_index + 1] = 1;
            }
            else if (use_nonlocal) {
               col_d(row_index) = nonlocal_row_result.col;
               counts_nl[row_index + 1] = 1;
            }
         });
      });
   }
};

//------------------------------------------------------------------------------------------------------------------------

// Matrix-free one point classical prolongator
// Every row of P has at most one entry, a one, so applying P is a gather from the coarse vector
// and applying P^T is a scatter-add into it
// For each fine row the context stores the local coarse column of its entry (-1 if the row is
// empty or its entry is on another rank). The rows whose entry is on another rank are listed
// separately, with the index of their column in garray (the off-rank coarse columns, sorted),
// and sf moves the coarse values of those columns between their owners and ghost_d
struct OnePointShellCtx {
   // Whether P is an MPI matrix (from the type of A_fc, not the comm size)
   bool mpi = false;
   PetscInt local_rows = 0, local_cols = 0, col_start = 0;
   PetscIntKokkosView target_d;
   PetscInt n_ghost_rows = 0, n_ghost_cols = 0;
   PetscIntKokkosView ghost_rows_d, ghost_idx_d, garray_d;
   PetscScalarKokkosView ghost_d;
   PetscSF sf = NULL;
   // Host copies, only made if P is applied to vectors in host memory on a device build
   bool host_made = false;
   PetscIntKokkosViewHost target_h, ghost_rows_h, ghost_idx_h;
   PetscScalarKokkosViewHost ghost_h;
};

static PetscErrorCode one_point_shell_destroy(PetscCtxRt ctx)
{
   PetscFunctionBegin;
   OnePointShellCtx *c = *reinterpret_cast<OnePointShellCtx **>(ctx);
   if (c->sf) PetscCall(PetscSFDestroy(&c->sf));
   PetscCallCXX(delete c);
   PetscFunctionReturn(PETSC_SUCCESS);
}

// Can the default execution space use host memory directly (host backends)
static constexpr bool default_space_host_accessible = Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, Kokkos::HostSpace>::accessible;

static PetscErrorCode one_point_shell_make_host(OnePointShellCtx *c)
{
   PetscFunctionBegin;
   if (!c->host_made) {
      c->target_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c->target_d);
      c->ghost_rows_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c->ghost_rows_d);
      c->ghost_idx_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c->ghost_idx_d);
      c->ghost_h = PetscScalarKokkosViewHost("ghost_h", c->n_ghost_cols);
      size_t bytes = (c->target_d.extent(0) + c->ghost_rows_d.extent(0) + c->ghost_idx_d.extent(0)) * sizeof(PetscInt);
      PetscCall(PetscLogGpuToCpu(bytes));
      c->host_made = true;
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

// y = P x (add = false) or y += P x (add = true), on the given execution space
template <class ExecSpace>
static PetscErrorCode one_point_shell_gather(const ExecSpace &exec, OnePointShellCtx *c, const PetscInt *target, \
                  const PetscInt *ghost_rows, const PetscInt *ghost_idx, PetscScalar *ghost, PetscMemType ghost_mtype, \
                  const PetscScalar *x, PetscMemType x_mtype, PetscScalar *y, const bool add)
{
   PetscFunctionBegin;
   // Start sending the coarse values that other ranks need, the local rows don't need them
   if (c->sf) {
      if (PetscMemTypeDevice(x_mtype)) PetscCall(sf_sync_before_begin_kokkos());
      PetscCall(PetscSFBcastWithMemTypeBegin(c->sf, MPIU_SCALAR, x_mtype, x, ghost_mtype, ghost, MPI_REPLACE));
   }
   Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(exec, 0, c->local_rows), KOKKOS_LAMBDA(const PetscInt i) {
         const PetscInt t = target[i];
         const PetscScalar val = t >= 0 ? x[t] : 0.0;
         if (add) y[i] += val;
         else y[i] = val;
   });
   if (c->sf) {
      PetscCall(PetscSFBcastEnd(c->sf, MPIU_SCALAR, x, ghost, MPI_REPLACE));
      if (PetscMemTypeDevice(x_mtype)) PetscCall(sf_sync_after_end_kokkos());
      // Rows with off-rank entries were zero (or unchanged) above
      Kokkos::parallel_for(
         Kokkos::RangePolicy<ExecSpace>(exec, 0, c->n_ghost_rows), KOKKOS_LAMBDA(const PetscInt k) {
            y[ghost_rows[k]] += ghost[ghost_idx[k]];
      });
   }
   // On the device everything is ordered on the PETSc stream, so only the host waits
   if (PetscMemTypeHost(ghost_mtype)) exec.fence();
   PetscFunctionReturn(PETSC_SUCCESS);
}

// y += P^T x, on the given execution space
// Entries that land on the same coarse point are summed with atomics
template <class ExecSpace>
static PetscErrorCode one_point_shell_scatter(const ExecSpace &exec, OnePointShellCtx *c, const PetscInt *target, \
                  const PetscInt *ghost_rows, const PetscInt *ghost_idx, PetscScalar *ghost, PetscMemType ghost_mtype, \
                  const PetscScalar *x, PetscScalar *y, PetscMemType y_mtype)
{
   PetscFunctionBegin;
   if (c->sf) {
      const PetscInt n_ghost_cols = c->n_ghost_cols;
      Kokkos::parallel_for(
         Kokkos::RangePolicy<ExecSpace>(exec, 0, n_ghost_cols), KOKKOS_LAMBDA(const PetscInt k) {
            ghost[k] = 0.0;
      });
      Kokkos::parallel_for(
         Kokkos::RangePolicy<ExecSpace>(exec, 0, c->n_ghost_rows), KOKKOS_LAMBDA(const PetscInt k) {
            Kokkos::atomic_add(&ghost[ghost_idx[k]], x[ghost_rows[k]]);
      });
      // The sums have to be finished before they are sent
      if (PetscMemTypeHost(ghost_mtype)) exec.fence();
      else PetscCall(sf_sync_before_begin_kokkos());
      // Start sending the sums to their owners
      PetscCall(PetscSFReduceWithMemTypeBegin(c->sf, MPIU_SCALAR, ghost_mtype, ghost, y_mtype, y, MPI_SUM));
   }
   Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(exec, 0, c->local_rows), KOKKOS_LAMBDA(const PetscInt i) {
         const PetscInt t = target[i];
         if (t >= 0) Kokkos::atomic_add(&y[t], x[i]);
   });
   // On the device the unpack into y is ordered after this on the PETSc stream
   if (PetscMemTypeHost(ghost_mtype)) exec.fence();
   if (c->sf) {
      PetscCall(PetscSFReduceEnd(c->sf, MPIU_SCALAR, ghost, y, MPI_SUM));
      if (PetscMemTypeDevice(y_mtype)) PetscCall(sf_sync_after_end_kokkos());
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

// Applies P or P^T to vectors, with or without adding to y, on the device or on the host
// depending on where the vectors live
static PetscErrorCode one_point_shell_apply(Mat P, Vec x, Vec y, const bool add, const bool transpose)
{
   OnePointShellCtx *c;
   const PetscScalar *x_ptr;
   PetscScalar *y_ptr;
   PetscMemType x_mtype, y_mtype;

   PetscFunctionBegin;
   PetscCall(MatShellGetContext(P, &c));
   // The transpose always adds into y, so it needs the old values
   const bool read_y = add || transpose;
   PetscCall(VecGetArrayReadAndMemType(x, &x_ptr, &x_mtype));
   if (read_y) PetscCall(VecGetArrayAndMemType(y, &y_ptr, &y_mtype));
   else PetscCall(VecGetArrayWriteAndMemType(y, &y_ptr, &y_mtype));
   // If one vector is on the host and the other on the device, do it all on the host
   const bool mixed = PetscMemTypeHost(x_mtype) != PetscMemTypeHost(y_mtype);
   if (mixed) {
      PetscCall(VecRestoreArrayReadAndMemType(x, &x_ptr));
      if (read_y) PetscCall(VecRestoreArrayAndMemType(y, &y_ptr));
      else PetscCall(VecRestoreArrayWriteAndMemType(y, &y_ptr));
      PetscCall(VecGetArrayRead(x, &x_ptr));
      if (read_y) PetscCall(VecGetArray(y, &y_ptr));
      else PetscCall(VecGetArrayWrite(y, &y_ptr));
      x_mtype = y_mtype = PETSC_MEMTYPE_HOST;
   }

   if (!PetscMemTypeHost(x_mtype) || default_space_host_accessible) {
      // Device (or a host backend that can use the vectors directly)
      auto exec = PetscGetKokkosExecutionSpace();
      const PetscMemType ghost_mtype = default_space_host_accessible ? PETSC_MEMTYPE_HOST : PETSC_MEMTYPE_KOKKOS;
      if (transpose) {
         PetscCall(one_point_shell_scatter(exec, c, c->target_d.data(), c->ghost_rows_d.data(), c->ghost_idx_d.data(), \
                  c->ghost_d.data(), ghost_mtype, x_ptr, y_ptr, y_mtype));
      } else {
         PetscCall(one_point_shell_gather(exec, c, c->target_d.data(), c->ghost_rows_d.data(), c->ghost_idx_d.data(), \
                  c->ghost_d.data(), ghost_mtype, x_ptr, x_mtype, y_ptr, add));
      }
   }
   else {
      // Host vectors on a device build
      PetscCall(one_point_shell_make_host(c));
      Kokkos::DefaultHostExecutionSpace host_exec;
      if (transpose) {
         PetscCall(one_point_shell_scatter(host_exec, c, c->target_h.data(), c->ghost_rows_h.data(), c->ghost_idx_h.data(), \
                  c->ghost_h.data(), PETSC_MEMTYPE_HOST, x_ptr, y_ptr, y_mtype));
      } else {
         PetscCall(one_point_shell_gather(host_exec, c, c->target_h.data(), c->ghost_rows_h.data(), c->ghost_idx_h.data(), \
                  c->ghost_h.data(), PETSC_MEMTYPE_HOST, x_ptr, x_mtype, y_ptr, add));
      }
   }

   if (mixed) {
      PetscCall(VecRestoreArrayRead(x, &x_ptr));
      if (read_y) PetscCall(VecRestoreArray(y, &y_ptr));
      else PetscCall(VecRestoreArrayWrite(y, &y_ptr));
   } else {
      PetscCall(VecRestoreArrayReadAndMemType(x, &x_ptr));
      if (read_y) PetscCall(VecRestoreArrayAndMemType(y, &y_ptr));
      else PetscCall(VecRestoreArrayWriteAndMemType(y, &y_ptr));
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode one_point_shell_mult(Mat P, Vec x, Vec y)
{
   PetscFunctionBegin;
   PetscCall(one_point_shell_apply(P, x, y, false, false));
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode one_point_shell_mult_add(Mat P, Vec x, Vec y, Vec z)
{
   PetscFunctionBegin;
   if (z != y) PetscCall(VecCopy(y, z));
   PetscCall(one_point_shell_apply(P, x, z, true, false));
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode one_point_shell_mult_transpose(Mat P, Vec x, Vec y)
{
   PetscFunctionBegin;
   PetscCall(VecZeroEntries(y));
   PetscCall(one_point_shell_apply(P, x, y, true, true));
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode one_point_shell_mult_transpose_add(Mat P, Vec x, Vec y, Vec z)
{
   PetscFunctionBegin;
   if (z != y) PetscCall(VecCopy(y, z));
   PetscCall(one_point_shell_apply(P, x, z, true, true));
   PetscFunctionReturn(PETSC_SUCCESS);
}

// C = P B for dense B, one column at a time, wherever B lives
// (PETSc's default for shells would bind B and C to the CPU on CUDA builds)
static PetscErrorCode one_point_shell_mat_mult_dense(Mat P, Mat B, Mat C, void *)
{
   PetscInt K;

   PetscFunctionBegin;
   PetscCall(MatGetSize(B, NULL, &K));
   for (PetscInt k = 0; k < K; k++) {
      Vec x, y;
      PetscCall(MatDenseGetColumnVecRead(B, k, &x));
      PetscCall(MatDenseGetColumnVecWrite(C, k, &y));
      PetscCall(one_point_shell_apply(P, x, y, false, false));
      PetscCall(MatDenseRestoreColumnVecRead(B, k, &x));
      PetscCall(MatDenseRestoreColumnVecWrite(C, k, &y));
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

// Wrap the context as a MATSHELL
static PetscErrorCode one_point_shell_create(MPI_Comm comm, PetscInt global_rows, PetscInt global_cols, OnePointShellCtx *c, Mat *P)
{
   PetscFunctionBegin;
   PetscCall(MatCreateShell(comm, c->local_rows, c->local_cols, global_rows, global_cols, c, P));
   PetscCall(MatShellSetContextDestroy(*P, one_point_shell_destroy));
   PetscCall(MatShellSetOperation(*P, MATOP_MULT, (PetscErrorCodeFn *)one_point_shell_mult));
   PetscCall(MatShellSetOperation(*P, MATOP_MULT_ADD, (PetscErrorCodeFn *)one_point_shell_mult_add));
   PetscCall(MatShellSetOperation(*P, MATOP_MULT_TRANSPOSE, (PetscErrorCodeFn *)one_point_shell_mult_transpose));
   PetscCall(MatShellSetOperation(*P, MATOP_MULT_TRANSPOSE_ADD, (PetscErrorCodeFn *)one_point_shell_mult_transpose_add));
   PetscCall(MatShellSetVecType(*P, VECKOKKOS));
   const MatType dense_types[] = {MATSEQDENSE, MATMPIDENSE, MATSEQDENSECUDA, MATMPIDENSECUDA, MATSEQDENSEHIP, MATMPIDENSEHIP};
   for (const MatType t : dense_types) {
      PetscCall(MatShellSetMatProductOperation(*P, MATPRODUCT_AB, NULL, one_point_shell_mat_mult_dense, NULL, t, t));
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

// Is this one of our one point shells
static bool is_one_point_shell(Mat P)
{
   PetscBool shell;
   PetscCallAbort(PETSC_COMM_SELF, PetscObjectTypeCompare((PetscObject)P, MATSHELL, &shell));
   if (!shell) return false;
   PetscErrorCodeFn *f;
   PetscCallAbort(PETSC_COMM_SELF, MatShellGetOperation(P, MATOP_MULT, &f));
   return f == (PetscErrorCodeFn *)one_point_shell_mult;
}

// For Fortran: is this one of our one point shells
PETSC_INTERN void is_one_point_shell_kokkos(Mat *P, int *result)
{
   *result = is_one_point_shell(*P) ? 1 : 0;
}

//------------------------------------------------------------------------------------------------------------------------

// The global column and value of the entry in each local row of a one point shell P
// (the sentinel column and a zero for an empty row), for mat_mult_one_entry_per_row_kokkos
// Returns false if P isn't a one point shell
PETSC_VISIBILITY_INTERNAL bool one_point_shell_targets_kokkos(Mat P, const PetscInt sentinel, PetscIntKokkosView &target_global_d, PetscScalarKokkosView &weight_d)
{
   if (!is_one_point_shell(P)) return false;
   OnePointShellCtx *c;
   PetscCallAbort(PETSC_COMM_SELF, MatShellGetContext(P, &c));
   auto exec = PetscGetKokkosExecutionSpace();
   target_global_d = PetscIntKokkosView(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "target_global_d"), c->local_rows);
   weight_d = PetscScalarKokkosView(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "weight_d"), c->local_rows);
   auto target_d = c->target_d;
   auto ghost_rows_d = c->ghost_rows_d, ghost_idx_d = c->ghost_idx_d, garray_d = c->garray_d;
   const PetscInt col_start = c->col_start;
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, c->local_rows), KOKKOS_LAMBDA(const PetscInt i) {
         const PetscInt t = target_d(i);
         target_global_d(i) = t >= 0 ? t + col_start : sentinel;
         weight_d(i) = t >= 0 ? 1.0 : 0.0;
   });
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, c->n_ghost_rows), KOKKOS_LAMBDA(const PetscInt k) {
         target_global_d(ghost_rows_d(k)) = garray_d(ghost_idx_d(k));
         weight_d(ghost_rows_d(k)) = 1.0;
   });
   exec.fence();
   return true;
}

//------------------------------------------------------------------------------------------------------------------------

// The number of local and nonlocal nonzeros in a one point shell P
PETSC_INTERN void one_point_shell_nnzs_kokkos(Mat *P, PetscInt *nnzs_local, PetscInt *nnzs_nonlocal)
{
   OnePointShellCtx *c;
   PetscCallVoid(MatShellGetContext(*P, &c));
   auto exec = PetscGetKokkosExecutionSpace();
   auto target_d = c->target_d;
   PetscInt count = 0;
   Kokkos::parallel_reduce(
      Kokkos::RangePolicy<>(exec, 0, c->local_rows), KOKKOS_LAMBDA(const PetscInt i, PetscInt &update) {
         if (target_d(i) >= 0) update++;
   }, count);
   *nnzs_local = count;
   *nnzs_nonlocal = c->n_ghost_rows;
}

//------------------------------------------------------------------------------------------------------------------------

// Assemble a one point shell P as a MATAIJKOKKOS, for the places that need P as a matrix
PETSC_INTERN void one_point_shell_assemble_kokkos(Mat *P, Mat *output_mat)
{
   OnePointShellCtx *c;
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt global_rows, global_cols;
   Mat output_mat_local, output_mat_nonlocal;
   PetscCallVoid(MatShellGetContext(*P, &c));
   PetscCallVoid(PetscObjectGetComm((PetscObject)*P, &MPI_COMM_MATRIX));
   PetscCallVoid(MatGetSize(*P, &global_rows, &global_cols));
   auto exec = PetscGetKokkosExecutionSpace();
   const PetscInt local_rows = c->local_rows;
   auto target_d = c->target_d;
   auto ghost_rows_d = c->ghost_rows_d, ghost_idx_d = c->ghost_idx_d;

   // Zero initialised, the counts go into entry row + 1
   Kokkos::View<PetscInt *> i_local_d(Kokkos::view_alloc(exec, "i_local_d"), local_rows + 1);
   Kokkos::View<PetscInt *> i_nonlocal_d(Kokkos::view_alloc(exec, "i_nonlocal_d"), local_rows + 1);
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
         if (target_d(i) >= 0) i_local_d(i + 1) = 1;
   });
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, c->n_ghost_rows), KOKKOS_LAMBDA(const PetscInt k) {
         i_nonlocal_d(ghost_rows_d(k) + 1) = 1;
   });
   PetscInt nnzs_local = 0, nnzs_nonlocal = 0;
   Kokkos::parallel_scan(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i, PetscInt &update, const bool final) {
      update += i_local_d(i + 1);
      if (final) i_local_d(i + 1) = update;
   }, nnzs_local);
   Kokkos::parallel_scan(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i, PetscInt &update, const bool final) {
      update += i_nonlocal_d(i + 1);
      if (final) i_nonlocal_d(i + 1) = update;
   }, nnzs_nonlocal);

   Kokkos::View<PetscInt *> j_local_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "j_local_d"), nnzs_local);
   Kokkos::View<PetscScalar *> a_local_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "a_local_d"), nnzs_local);
   Kokkos::View<PetscInt *> j_nonlocal_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "j_nonlocal_d"), nnzs_nonlocal);
   Kokkos::View<PetscScalar *> a_nonlocal_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "a_nonlocal_d"), nnzs_nonlocal);
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
         if (target_d(i) >= 0) {
            j_local_d(i_local_d(i)) = target_d(i);
            a_local_d(i_local_d(i)) = 1.0;
         }
   });
   // The ghost rows are in increasing order, so entry k is the k-th nonlocal entry
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, c->n_ghost_rows), KOKKOS_LAMBDA(const PetscInt k) {
         j_nonlocal_d(k) = ghost_idx_d(k);
         a_nonlocal_d(k) = 1.0;
   });
   exec.fence();

   PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, c->local_cols, i_local_d, j_local_d, a_local_d, &output_mat_local));
   // The same type as an assembled P would have had
   if (c->mpi)
   {
      PetscInt *garray_host = NULL;
      PetscCallVoid(PetscMalloc1(c->n_ghost_cols, &garray_host));
      if (c->n_ghost_cols > 0)
      {
         PetscIntKokkosViewHost garray_h(garray_host, c->n_ghost_cols);
         Kokkos::deep_copy(garray_h, c->garray_d);
         PetscCallVoid(PetscLogGpuToCpu(c->n_ghost_cols * sizeof(PetscInt)));
      }
      PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, c->n_ghost_cols, i_nonlocal_d, j_nonlocal_d, a_nonlocal_d, &output_mat_nonlocal));
      PetscCallVoid(MatCreateMPIAIJWithSeqAIJ(MPI_COMM_MATRIX, global_rows, global_cols, output_mat_local, output_mat_nonlocal, garray_host, output_mat));
   }
   else
   {
      *output_mat = output_mat_local;
   }
}

//------------------------------------------------------------------------------------------------------------------------

// Build the one point classical prolongator P = [W I]' straight from A_fc, where W has a
// one in the column of the biggest entry in each row of A_fc
// This gives the same matrix as generate_one_point_with_one_entry_from_sparse_kokkos followed
// by compute_P_from_W_kokkos with the identity, without building W
// If shell_int is 1, P is matrix-free (a MATSHELL that applies P and P^T on the device)
PETSC_INTERN void generate_one_point_P_from_sparse_kokkos(Mat *input_mat, PetscInt global_row_start, IS *is_fine, \
                  IS *is_coarse, const int shell_int, Mat *output_mat)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows_fine, local_rows_coarse, local_rows, global_rows_fine, global_cols_coarse;
   PetscInt rows_ao, cols_ao = 0;
   MatType mat_type;
   Mat output_mat_local, output_mat_nonlocal;

   PetscCallVoid(MatGetType(*input_mat, &mat_type));
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;
   Mat mat_local = NULL, mat_nonlocal = NULL;

   PetscIntConstKokkosViewHost colmap_input_h;
   PetscIntKokkosView colmap_input_d;
   const PetscInt *colmap_input;
   auto exec = PetscGetKokkosExecutionSpace();

   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*input_mat, &mat_local, &mat_nonlocal, &colmap_input));
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao));

      // We also copy the input mat colmap over to the device as we need it
      colmap_input_h = PetscIntConstKokkosViewHost(colmap_input, cols_ao);
      colmap_input_d = PetscIntKokkosView(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "colmap_input_d"), cols_ao);
      Kokkos::deep_copy(exec, colmap_input_d, colmap_input_h);
      // Log copy with petsc
      size_t bytes = colmap_input_h.extent(0) * sizeof(PetscInt);
      PetscCallVoid(PetscLogCpuToGpu(bytes));
   }
   else
   {
      mat_local = *input_mat;
   }

   PetscCallVoid(PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX));
   // A_fc has the fine rows and the coarse columns
   PetscCallVoid(MatGetSize(*input_mat, &global_rows_fine, &global_cols_coarse));
   PetscCallVoid(ISGetLocalSize(*is_fine, &local_rows_fine));
   PetscCallVoid(ISGetLocalSize(*is_coarse, &local_rows_coarse));
   local_rows = local_rows_fine + local_rows_coarse;
   const PetscInt global_rows = global_rows_fine + global_cols_coarse;

   // Copy the fine and coarse indices to the device
   const PetscInt *fine_indices_ptr, *coarse_indices_ptr;
   PetscCallVoid(ISGetIndices(*is_fine, &fine_indices_ptr));
   PetscCallVoid(ISGetIndices(*is_coarse, &coarse_indices_ptr));
   auto fine_view_h = PetscIntConstKokkosViewHost(fine_indices_ptr, local_rows_fine);
   auto fine_view_d = PetscIntKokkosView(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "fine_view_d"), local_rows_fine);
   auto coarse_view_h = PetscIntConstKokkosViewHost(coarse_indices_ptr, local_rows_coarse);
   auto coarse_view_d = PetscIntKokkosView(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "coarse_view_d"), local_rows_coarse);
   Kokkos::deep_copy(exec, fine_view_d, fine_view_h);
   Kokkos::deep_copy(exec, coarse_view_d, coarse_view_h);
   size_t bytes = (fine_view_h.extent(0) + coarse_view_h.extent(0)) * sizeof(PetscInt);
   PetscCallVoid(PetscLogCpuToGpu(bytes));

   // ~~~~~~~~~~~~
   // Get pointers to the i,j on the device and Kokkos views to the values
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, NULL, &mtype));
   Kokkos::View<const PetscScalar *> device_local_vals;
   Kokkos::View<const PetscScalar *> device_nonlocal_vals;
   PetscCallVoid(MatSeqAIJGetKokkosView(mat_local, &device_local_vals));
   if (mpi) PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal, &device_nonlocal_vals));

   // ~~~~~~~~~~~~
   // Count the entries in each row of P, straight into the output row pointers at row + 1
   // Every row has at most one entry, col_d holds its column
   // ~~~~~~~~~~~~
   // Zero initialised on exec
   Kokkos::View<PetscInt *> i_local_d(Kokkos::view_alloc(exec, "i_local_d"), local_rows + 1);
   Kokkos::View<PetscInt *> i_nonlocal_d;
   if (mpi) i_nonlocal_d = Kokkos::View<PetscInt *>(Kokkos::view_alloc(exec, "i_nonlocal_d"), local_rows + 1);
   PetscIntKokkosView col_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "col_d"), local_rows);

   OnePointMaxRowFunctor functor;
   functor.nrows = local_rows_fine;
   functor.global_row_start = global_row_start;
   functor.mpi = mpi;
   functor.l_i = device_local_i;
   functor.l_j = device_local_j;
   functor.l_a = device_local_vals;
   if (mpi)
   {
      functor.nl_i = device_nonlocal_i;
      functor.nl_j = device_nonlocal_j;
      functor.nl_a = device_nonlocal_vals;
      functor.colmap_d = colmap_input_d;
      functor.counts_nl = i_nonlocal_d.data();
   }
   functor.fine_d = fine_view_d;
   functor.col_d = col_d;
   functor.counts_l = i_local_d.data();

   // The vector length comes from the average row length of A_fc (local plus nonlocal)
   // The host backends run the vector lanes serially, so there it makes no difference
   MatInfo info;
   PetscCallVoid(MatGetInfo(*input_mat, MAT_LOCAL, &info));
   const double avg_row_nnz = local_rows_fine > 0 ? info.nz_used / local_rows_fine : 0.0;
   int vector_length = 1;
   while (2 * vector_length <= avg_row_nnz && vector_length < 32) vector_length *= 2;
   vector_length = std::min(vector_length, Kokkos::TeamPolicy<>::vector_length_max());
   {
      Kokkos::TeamPolicy<> probe(exec, 1, 1, vector_length);
      int team_size = std::max(1, 256 / vector_length);
      team_size = std::min(team_size, probe.team_size_max(functor, Kokkos::ParallelForTag()));
      const int league_size = static_cast<int>((local_rows_fine + team_size - 1) / team_size);
      Kokkos::parallel_for(Kokkos::TeamPolicy<>(exec, league_size, team_size, vector_length), functor);
   }

   // The C points have the identity, which is always local
   // The local coarse index is the local column
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, local_rows_coarse), KOKKOS_LAMBDA(PetscInt i) {
         const PetscInt row_index = coarse_view_d(i) - global_row_start;
         col_d(row_index) = i;
         i_local_d(row_index + 1) = 1;
   });

   // ~~~~~~~~~~~~
   // Row pointers from an inclusive scan of the counts, in place (index i only touches i + 1)
   // The scan returns the total and Kokkos fences before returning it
   // ~~~~~~~~~~~~
   PetscInt nnzs_local = 0, nnzs_nonlocal = 0;
   Kokkos::parallel_scan(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i, PetscInt &update, const bool final) {
      update += i_local_d(i + 1);
      if (final) i_local_d(i + 1) = update;
   }, nnzs_local);
   if (mpi)
   {
      Kokkos::parallel_scan(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i, PetscInt &update, const bool final) {
         update += i_nonlocal_d(i + 1);
         if (final) i_nonlocal_d(i + 1) = update;
      }, nnzs_nonlocal);
   }

   // ~~~~~~~~~~~~
   // Fill j and the values, every entry is a one
   // The nonlocal j are written as global columns and rewritten below
   // ~~~~~~~~~~~~
   Kokkos::View<PetscScalar *> a_local_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "a_local_d"), nnzs_local);
   Kokkos::View<PetscInt *> j_local_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "j_local_d"), nnzs_local);
   Kokkos::View<PetscScalar *> a_nonlocal_d;
   Kokkos::View<PetscInt *> j_nonlocal_d;
   if (mpi)
   {
      a_nonlocal_d = Kokkos::View<PetscScalar *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "a_nonlocal_d"), nnzs_nonlocal);
      j_nonlocal_d = Kokkos::View<PetscInt *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "j_nonlocal_d"), nnzs_nonlocal);
   }
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
         if (i_local_d(i + 1) > i_local_d(i)) {
            j_local_d(i_local_d(i)) = col_d(i);
            a_local_d(i_local_d(i)) = 1.0;
         }
         else if (mpi && i_nonlocal_d(i + 1) > i_nonlocal_d(i)) {
            j_nonlocal_d(i_nonlocal_d(i)) = col_d(i);
            a_nonlocal_d(i_nonlocal_d(i)) = 1.0;
         }
   });

   // Let's make sure everything on the device is finished
   exec.fence();

   if (shell_int)
   {
      OnePointShellCtx *c = new OnePointShellCtx();
      c->mpi = mpi;
      c->local_rows = local_rows;
      c->local_cols = local_rows_coarse;
      PetscInt *garray_host = NULL;
      if (mpi)
      {
         // Build garray and rewrite the j_nonlocal_d indices so they index into it
         // This fences internally
         rewrite_j_global_to_local(cols_ao, c->n_ghost_cols, j_nonlocal_d, &garray_host);
         c->n_ghost_rows = nnzs_nonlocal;
         c->ghost_idx_d = j_nonlocal_d;
         c->garray_d = PetscIntKokkosView(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "garray_d"), c->n_ghost_cols);
         Kokkos::deep_copy(exec, c->garray_d, PetscIntConstKokkosViewHost(garray_host, c->n_ghost_cols));
         PetscCallVoid(PetscLogCpuToGpu(c->n_ghost_cols * sizeof(PetscInt)));
         c->ghost_d = PetscScalarKokkosView("ghost_d", c->n_ghost_cols);
      }
      // The local column of each row's entry, or -1, and the rows with an off-rank entry
      // in increasing order
      c->target_d = PetscIntKokkosView(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "target_d"), local_rows);
      c->ghost_rows_d = PetscIntKokkosView(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "ghost_rows_d"), c->n_ghost_rows);
      auto target_d = c->target_d;
      auto ghost_rows_d = c->ghost_rows_d;
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
            target_d(i) = i_local_d(i + 1) > i_local_d(i) ? j_local_d(i_local_d(i)) : -1;
            if (mpi && i_nonlocal_d(i + 1) > i_nonlocal_d(i)) ghost_rows_d(i_nonlocal_d(i)) = i;
      });
      exec.fence();

      PetscCallVoid(one_point_shell_create(MPI_COMM_MATRIX, global_rows, global_cols_coarse, c, output_mat));
      PetscLayout cmap;
      PetscCallVoid(MatGetLayouts(*output_mat, NULL, &cmap));
      PetscCallVoid(PetscLayoutGetRange(cmap, &c->col_start, NULL));
      if (mpi)
      {
         // The halo of coarse values, from their owners to the ranks with rows pointing at them
         PetscCallVoid(PetscSFCreate(MPI_COMM_MATRIX, &c->sf));
         PetscCallVoid(PetscSFSetGraphLayout(c->sf, cmap, c->n_ghost_cols, NULL, PETSC_COPY_VALUES, garray_host));
         PetscCallVoid(PetscSFSetUp(c->sf));
         PetscCallVoid(PetscFree(garray_host));
      }
   }
   else
   {

   // We can create our local diagonal block matrix directly on the device
   PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, local_rows_coarse, i_local_d, j_local_d, a_local_d, &output_mat_local));

   if (mpi)
   {
      // Build garray on the host and rewrite the j_nonlocal_d indices so they are local
      PetscInt *garray_host = NULL;
      PetscInt col_ao_output = 0;
      // This fences internally
      rewrite_j_global_to_local(cols_ao, col_ao_output, j_nonlocal_d, &garray_host);

      // We can create our nonlocal diagonal block matrix directly on the device
      PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, col_ao_output, i_nonlocal_d, j_nonlocal_d, a_nonlocal_d, &output_mat_nonlocal));

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
   PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local, &device_local_vals));
   if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal, &device_nonlocal_vals));

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
   Mat mat_local = NULL, mat_nonlocal = NULL;
   const PetscInt *colmap_input;

   auto exec = PetscGetKokkosExecutionSpace();

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
   Kokkos::deep_copy(exec, fine_view_d, fine_view_h);
   Kokkos::deep_copy(exec, coarse_view_d, coarse_view_h);
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
   // Get pointers to the i,j on the device and Kokkos views to the values
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, NULL, &mtype));
   Kokkos::View<const PetscScalar *> device_local_vals;
   Kokkos::View<const PetscScalar *> device_nonlocal_vals;
   PetscCallVoid(MatSeqAIJGetKokkosView(mat_local, &device_local_vals));
   if (mpi) PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal, &device_nonlocal_vals));

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
      Kokkos::deep_copy(exec, nnz_match_local_row_d, 0);         
      if (mpi) nnz_match_nonlocal_row_d = PetscIntKokkosView("nnz_match_nonlocal_row_d", local_rows);                  

      // ~~~~~~~~~~~~
      // Need to count the number of nnzs we end up with, on each row and in total
      // ~~~~~~~~~~~~
      // Loop over the rows of W
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, local_rows_fine), KOKKOS_LAMBDA(PetscInt i) {

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
            Kokkos::RangePolicy<>(exec, 0, local_rows_coarse), KOKKOS_LAMBDA(PetscInt i) {

            // Convert to global coarse index into a local index into the full matrix
            PetscInt row_index = coarse_view_d(i) - global_row_start;
            nnz_match_local_row_d(row_index)++;
         }); 
      }  

      // Get number of nnzs
      Kokkos::parallel_reduce ("ReductionLocal", Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA (const PetscInt i, PetscInt& update) {
         update += nnz_match_local_row_d(i);
      }, nnzs_match_local);
      if (mpi)
      {
         Kokkos::parallel_reduce ("ReductionNonLocal", Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA (const PetscInt i, PetscInt& update) {
            update += nnz_match_nonlocal_row_d(i);
         }, nnzs_match_nonlocal);
      }

      // ~~~~~~~~~~~~

      // Need to do a scan on nnz_match_local_row_d to get where each row starts
      Kokkos::parallel_scan (Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
         // Inclusive scan
         update += nnz_match_local_row_d(i);
         if (final) {
            nnz_match_local_row_d(i) = update; // only update array on final pass
         }
      });
      if (mpi)
      {
         // Need to do a scan on nnz_match_nonlocal_row_d to get where each row starts
         Kokkos::parallel_scan (Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
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
      Kokkos::deep_copy(exec, Kokkos::subview(i_local_d, 0), 0);       

      // we also have to go and build the a, i, j for the non-local off-diagonal block
      if (mpi) 
      {
         // Non-local 
         a_nonlocal_d = Kokkos::View<PetscScalar *>("a_nonlocal_d", nnzs_match_nonlocal);
         i_nonlocal_d = Kokkos::View<PetscInt *>("i_nonlocal_d", local_rows+1);
         j_nonlocal_d = Kokkos::View<PetscInt *>("j_nonlocal_d", nnzs_match_nonlocal);  

         // Initialize first entry to zero - the rest get set below
         Kokkos::deep_copy(exec, Kokkos::subview(i_nonlocal_d, 0), 0);                
      }  

      // ~~~~~~~~~~~~~~~
      // Have to build i_local_d and the nonlocal for every row (F and C points)
      // regardless of if we are sticking 1 in (ie identity)
      // This has to happen before the main loop as the f and c 
      // points are placed in different orders (ie not in order as the index 
      // is row_index 
      // ~~~~~~~~~~~~~~~
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, local_rows_fine), KOKKOS_LAMBDA(PetscInt i) {

            // Convert to global fine index into a local index in the full matrix
            PetscInt row_index = fine_view_d(i) - global_row_start;       

            // The start of our row index comes from the scan
            i_local_d(row_index + 1) = nnz_match_local_row_d(row_index);   
            if (mpi) i_nonlocal_d(row_index + 1) = nnz_match_nonlocal_row_d(row_index);         
      });            

      // Always have to set the i_local_d for C points, regardless of if we are setting
      // 1 in the identity part for them
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, local_rows_coarse), KOKKOS_LAMBDA(PetscInt i) {

         // Convert to global coarse index into a local index into the full matrix
         PetscInt row_index = coarse_view_d(i) - global_row_start;

         // The start of our row index comes from the scan
         i_local_d(row_index + 1) = nnz_match_local_row_d(row_index);
         if (mpi) i_nonlocal_d(row_index + 1) = nnz_match_nonlocal_row_d(row_index);        

      });  

      // Loop over the rows of W
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(exec, local_rows_fine, Kokkos::AUTO()),
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
               a_local_d(i_local_d(row_index) + j) = device_local_vals(device_local_i[i] + j);

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
                  a_nonlocal_d(i_nonlocal_d(row_index) + j) = device_nonlocal_vals(device_nonlocal_i[i] + j);
                        
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
         PetscCallVoid(MatMPIAIJGetSeqAIJ(*output_mat, &mat_local_output, &mat_nonlocal_output, NULL));
      }
      else
      {
         mat_local_output = *output_mat;
      }     
      // Get device pointers for the existing i array
      const PetscInt *device_local_i_output = nullptr, *device_nonlocal_i_output = nullptr;
      PetscMemType mtype;
      PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_output, &device_local_i_output, NULL, NULL, &mtype));
      if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_output, &device_nonlocal_i_output, NULL, NULL, &mtype));

      // Read-write Kokkos views to the values - we only update the W block, the identity block is preserved
      Kokkos::View<PetscScalar *> device_local_a_output;
      Kokkos::View<PetscScalar *> device_nonlocal_a_output;
      PetscCallVoid(MatSeqAIJGetKokkosView(mat_local_output, &device_local_a_output));
      if (mpi) PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal_output, &device_nonlocal_a_output));

      // Only have to write W as the identity block cannot change
      // Loop over the rows of W - annoying we have const views as this is just the same loop as above
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(exec, local_rows_fine, Kokkos::AUTO()),
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

               device_local_a_output(device_local_i_output[row_index] + j) = device_local_vals(device_local_i[i] + j);

            });

            // For over nonlocal columns - copy in W
            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];

               Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {

                  // We keep the existing local indices in the off-diagonal blocl here
                  // we have all the same columns as W and hence the same garray
                  device_nonlocal_a_output(device_nonlocal_i_output[row_index] + j) = device_nonlocal_vals(device_nonlocal_i[i] + j);

               });
            }
      });

      Kokkos::fence();

      // The matching restore handles MatSeqAIJKokkosModifyDevice (clears sync state,
      // marks device modified, invalidates transpose/hermitian, bumps object state).
      PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local_output, &device_local_a_output));
      if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal_output, &device_nonlocal_a_output));

   }

   // ~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~   

   if (!reuse_int)
   {
      // Loop over all the C points - we know they're in the local block
      if (identity_int) 
      {
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(exec, 0, local_rows_coarse), KOKKOS_LAMBDA(PetscInt i) {

            // Convert to global coarse index into a local index into the full matrix
            PetscInt row_index = coarse_view_d(i) - global_row_start;

            // Only a single column
            j_local_d(i_local_d(row_index)) = i;
            a_local_d(i_local_d(row_index)) = 1.0;         

         }); 
      }   
        
      // Let's make sure everything on the device is finished
      Kokkos::fence();      

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

   PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local, &device_local_vals));
   if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal, &device_nonlocal_vals));

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Row kernels for compute_R_from_Z_drop_kokkos
// Each team handles team_size rows of Z, one thread per row, with vector lanes over the
// entries in that row
// The drop is the one remove_small_from_sparse_kokkos does with relative_max_row_tolerance_int = 1,
// no lumping, allow_drop_diagonal_int = 0 and allow_diag_strength_int = 0: an entry is kept if
// abs(val) >= tol * the biggest abs(val) in the row, or if it is the "diagonal" of Z
// (global column of Z equal to the global row of Z), so the result is identical
struct RFromZDropFunctor {
   struct CountTag {};
   struct FillTag {};

   PetscInt nrows = 0;
   // Ownership starts of Z, used for the diagonal test
   PetscInt global_row_start_z = 0, global_col_start_z = 0;
   // Row start of R, its local columns are those of the full matrix
   PetscInt global_row_start = 0;
   PetscReal tol = 0.0;
   bool mpi = false;

   // Input local and nonlocal blocks of Z, colmap_d gives the global column of Z of a nonlocal column
   const PetscInt *l_i = nullptr, *l_j = nullptr, *nl_i = nullptr, *nl_j = nullptr;
   Kokkos::View<const PetscScalar *> l_a, nl_a;
   PetscIntKokkosView colmap_d;
   // The full matrix index of each local column of Z, and of each local coarse row
   PetscIntKokkosView orig_d, coarse_d;

   // Written by count: the tolerance for each row, and the number of entries of R in each
   // row in counts_l[row + 1] (kept entries plus the identity) and counts_nl[row + 1]
   PetscScalarKokkosView rel_row_tol_d;
   PetscInt *counts_l = nullptr, *counts_nl = nullptr;

   // Fill: start of each output row, and the output j and values
   // The local j are local columns of R, the nonlocal j are still the nonlocal columns of Z
   const PetscInt *out_l_i = nullptr, *out_nl_i = nullptr;
   PetscInt *out_l_j = nullptr, *out_nl_j = nullptr;
   PetscScalar *out_l_a = nullptr, *out_nl_a = nullptr;

   KOKKOS_INLINE_FUNCTION bool keep_local(const PetscInt k, const PetscScalar row_tol, const PetscInt row_global) const
   {
      return Kokkos::abs(l_a(k)) >= row_tol || l_j[k] + global_col_start_z == row_global;
   }
   KOKKOS_INLINE_FUNCTION bool keep_nonlocal(const PetscInt k, const PetscScalar row_tol, const PetscInt row_global) const
   {
      return Kokkos::abs(nl_a(k)) >= row_tol || colmap_d(nl_j[k]) == row_global;
   }

   KOKKOS_INLINE_FUNCTION void operator()(const CountTag &, const KokkosTeamMemberType &t) const
   {
      const PetscInt row_start = t.league_rank() * t.team_size();
      const PetscInt row_end = row_start + t.team_size() < nrows ? row_start + t.team_size() : nrows;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, row_start, row_end), [&](const PetscInt i) {
         const PetscInt row_global = i + global_row_start_z;
         const PetscInt l_start = l_i[i], l_len = l_i[i + 1] - l_start;
         const PetscInt nl_start = mpi ? nl_i[i] : 0, nl_len = mpi ? nl_i[i + 1] - nl_start : 0;

         // The biggest abs(val) in the row, local and nonlocal
         // The Max reducer starts from the lowest value, so an empty block gives a negative max
         PetscScalar max_l, max_nl = Kokkos::reduction_identity<PetscScalar>::max();
         Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, l_len),
            [&](const PetscInt k, PetscScalar &thread_max) {
               const PetscScalar val = Kokkos::abs(l_a(l_start + k));
               if (val > thread_max) thread_max = val;
            }, Kokkos::Max<PetscScalar>(max_l));
         if (mpi) {
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, nl_len),
               [&](const PetscInt k, PetscScalar &thread_max) {
                  const PetscScalar val = Kokkos::abs(nl_a(nl_start + k));
                  if (val > thread_max) thread_max = val;
               }, Kokkos::Max<PetscScalar>(max_nl));
         }
         const PetscScalar max_val = max_nl > max_l ? max_nl : max_l;
         const PetscScalar row_tol = tol * max_val;

         PetscInt count_l = 0, count_nl = 0;
         Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, l_len),
            [&](const PetscInt k, PetscInt &thread_count) {
               if (keep_local(l_start + k, row_tol, row_global)) thread_count++;
            }, count_l);
         if (mpi) {
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, nl_len),
               [&](const PetscInt k, PetscInt &thread_count) {
                  if (keep_nonlocal(nl_start + k, row_tol, row_global)) thread_count++;
               }, count_nl);
         }

         Kokkos::single(Kokkos::PerThread(t), [&]() {
            rel_row_tol_d(i) = row_tol;
            // One extra for the identity
            counts_l[i + 1] = count_l + 1;
            if (mpi) counts_nl[i + 1] = count_nl;
         });
      });
   }

   // The vector lanes do an exclusive scan over the kept flags to find their output position,
   // so the kept entries stay in the order of the input row
   KOKKOS_INLINE_FUNCTION void operator()(const FillTag &, const KokkosTeamMemberType &t) const
   {
      const PetscInt row_start = t.league_rank() * t.team_size();
      const PetscInt row_end = row_start + t.team_size() < nrows ? row_start + t.team_size() : nrows;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, row_start, row_end), [&](const PetscInt i) {
         const PetscInt row_global = i + global_row_start_z;
         const PetscInt l_start = l_i[i], l_len = l_i[i + 1] - l_start;
         const PetscInt nl_start = mpi ? nl_i[i] : 0, nl_len = mpi ? nl_i[i + 1] - nl_start : 0;
         const PetscScalar row_tol = rel_row_tol_d(i);

         const PetscInt out_l_offset = out_l_i[i];
         Kokkos::parallel_scan(Kokkos::ThreadVectorRange(t, l_len),
            [&](const PetscInt k, PetscInt &partial_sum, const bool is_final) {
               if (keep_local(l_start + k, row_tol, row_global)) {
                  if (is_final) {
                     // The orig_d contains the global indices for the original full matrix
                     out_l_j[out_l_offset + partial_sum] = orig_d(l_j[l_start + k]) - global_row_start;
                     out_l_a[out_l_offset + partial_sum] = l_a(l_start + k);
                  }
                  partial_sum++;
               }
            });
         if (mpi) {
            const PetscInt out_nl_offset = out_nl_i[i];
            Kokkos::parallel_scan(Kokkos::ThreadVectorRange(t, nl_len),
               [&](const PetscInt k, PetscInt &partial_sum, const bool is_final) {
                  if (keep_nonlocal(nl_start + k, row_tol, row_global)) {
                     if (is_final) {
                        out_nl_j[out_nl_offset + partial_sum] = nl_j[nl_start + k];
                        out_nl_a[out_nl_offset + partial_sum] = nl_a(nl_start + k);
                     }
                     partial_sum++;
                  }
               });
         }

         // The identity goes at the end of the local row and we sort after
         Kokkos::single(Kokkos::PerThread(t), [&]() {
            out_l_j[out_l_i[i + 1] - 1] = coarse_d(i) - global_row_start;
            out_l_a[out_l_i[i + 1] - 1] = 1.0;
         });
      });
   }
};

// Launches one of the RFromZDropFunctor kernels over all the rows
template <class Tag>
static void launch_r_from_z_drop(const RFromZDropFunctor &functor, const int vector_length)
{
   if (functor.nrows == 0) return;
   auto exec = PetscGetKokkosExecutionSpace();
   Kokkos::TeamPolicy<Tag> probe(exec, 1, 1, vector_length);
   int team_size = std::max(1, 256 / vector_length);
   team_size = std::min(team_size, probe.team_size_max(functor, Kokkos::ParallelForTag()));
   const int league_size = static_cast<int>((functor.nrows + team_size - 1) / team_size);
   Kokkos::parallel_for(Kokkos::TeamPolicy<Tag>(exec, league_size, team_size, vector_length), functor);
}

//------------------------------------------------------------------------------------------------------------------------

// Drop small entries from Z and stick it in a full sized R with the identity, in one pass and
// without building the dropped Z as a matrix
// Gives the same R as remove_small_from_sparse_kokkos on Z (relative row tolerance, no lumping)
// followed by compute_R_from_Z_kokkos with identity and without reuse
// Also creates orig_fine_col_indices for the columns of the dropped Z (local, then the nonlocal
// columns that are kept on this rank)
static void compute_R_from_Z_drop_kokkos(Mat *input_mat, PetscInt global_row_start, IS *is_fine, \
                  IS *is_coarse, IS *orig_fine_col_indices, const PetscReal tol, Mat *output_mat)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_coarse_size, local_fine_size, local_full_cols;
   PetscInt global_coarse_size, global_fine_size, global_full_cols;
   PetscInt rows_ao = 0, cols_ao = 0, rows_ad, cols_ad;
   PetscInt global_rows_z, global_cols_z;
   PetscInt local_rows_z, local_cols_z;
   PetscInt global_row_start_z, global_row_end_plus_one_z;
   PetscInt global_col_start_z, global_col_end_plus_one_z;
   MatType mat_type;
   Mat output_mat_local, output_mat_nonlocal;

   PetscCallVoid(MatGetType(*input_mat, &mat_type));
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;
   Mat mat_local = NULL, mat_nonlocal = NULL;

   PetscIntConstKokkosViewHost colmap_input_h;
   PetscIntKokkosView colmap_input_d;
   const PetscInt *colmap_input = NULL;

   auto exec = PetscGetKokkosExecutionSpace();

   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*input_mat, &mat_local, &mat_nonlocal, &colmap_input));
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao));

      // We also copy the input mat colmap over to the device as we need it for the diagonal test
      colmap_input_h = PetscIntConstKokkosViewHost(colmap_input, cols_ao);
      colmap_input_d = PetscIntKokkosView("colmap_input_d", cols_ao);
      Kokkos::deep_copy(exec, colmap_input_d, colmap_input_h);
      // Log copy with petsc
      size_t bytes = colmap_input_h.extent(0) * sizeof(PetscInt);
      PetscCallVoid(PetscLogCpuToGpu(bytes));
   }
   else
   {
      mat_local = *input_mat;
   }

   PetscCallVoid(PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX));
   PetscCallVoid(ISGetLocalSize(*is_coarse, &local_coarse_size));
   PetscCallVoid(ISGetLocalSize(*is_fine, &local_fine_size));
   PetscCallVoid(ISGetSize(*is_coarse, &global_coarse_size));
   PetscCallVoid(ISGetSize(*is_fine, &global_fine_size));

   local_full_cols = local_coarse_size + local_fine_size;
   global_full_cols = global_coarse_size + global_fine_size;

   PetscCallVoid(MatGetLocalSize(*input_mat, &local_rows_z, &local_cols_z));
   PetscCallVoid(MatGetSize(*input_mat, &global_rows_z, &global_cols_z));
   PetscCallVoid(MatGetOwnershipRange(*input_mat, &global_row_start_z, &global_row_end_plus_one_z));
   PetscCallVoid(MatGetOwnershipRangeColumn(*input_mat, &global_col_start_z, &global_col_end_plus_one_z));
   PetscCallVoid(MatGetSize(mat_local, &rows_ad, &cols_ad));

   // The local columns of Z are the local fine points in order, so their original
   // indices are just the local entries of is_fine
   const PetscInt *fine_indices_ptr, *coarse_indices_ptr;
   PetscCallVoid(ISGetIndices(*is_fine, &fine_indices_ptr));
   PetscCallVoid(ISGetIndices(*is_coarse, &coarse_indices_ptr));

   auto orig_view_h = PetscIntConstKokkosViewHost(fine_indices_ptr, cols_ad);
   auto orig_view_d = PetscIntKokkosView("orig_view_d", cols_ad);
   auto coarse_view_h = PetscIntConstKokkosViewHost(coarse_indices_ptr, local_coarse_size);
   auto coarse_view_d = PetscIntKokkosView("coarse_view_d", local_coarse_size);
   Kokkos::deep_copy(exec, orig_view_d, orig_view_h);
   size_t bytes = orig_view_h.extent(0) * sizeof(PetscInt);
   PetscCallVoid(PetscLogCpuToGpu(bytes));
   Kokkos::deep_copy(exec, coarse_view_d, coarse_view_h);
   bytes = coarse_view_h.extent(0) * sizeof(PetscInt);
   PetscCallVoid(PetscLogCpuToGpu(bytes));

   // ~~~~~~~~~~~~
   // Get pointers to the i,j on the device and Kokkos views to the values
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, NULL, &mtype));
   Kokkos::View<const PetscScalar *> device_local_vals;
   Kokkos::View<const PetscScalar *> device_nonlocal_vals;
   PetscCallVoid(MatSeqAIJGetKokkosView(mat_local, &device_local_vals));
   if (mpi) PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal, &device_nonlocal_vals));

   RFromZDropFunctor functor;
   functor.nrows = local_rows_z;
   functor.global_row_start_z = global_row_start_z;
   functor.global_col_start_z = global_col_start_z;
   functor.global_row_start = global_row_start;
   functor.tol = tol;
   functor.mpi = mpi;
   functor.l_i = device_local_i;
   functor.l_j = device_local_j;
   functor.l_a = device_local_vals;
   functor.orig_d = orig_view_d;
   functor.coarse_d = coarse_view_d;
   if (mpi)
   {
      functor.nl_i = device_nonlocal_i;
      functor.nl_j = device_nonlocal_j;
      functor.nl_a = device_nonlocal_vals;
      functor.colmap_d = colmap_input_d;
   }

   // The vector length comes from the average row length (local plus nonlocal)
   // The host backends run the vector lanes serially, so there it makes no difference
   MatInfo info;
   PetscCallVoid(MatGetInfo(*input_mat, MAT_LOCAL, &info));
   const double avg_row_nnz = local_rows_z > 0 ? info.nz_used / local_rows_z : 0.0;
   int vector_length = 1;
   while (2 * vector_length <= avg_row_nnz && vector_length < 32) vector_length *= 2;
   vector_length = std::min(vector_length, Kokkos::TeamPolicy<>::vector_length_max());

   // ~~~~~~~~~~~~
   // Count the entries of R in each row
   // ~~~~~~~~~~~~
   // Zero initialised on exec, the counts go into entry row + 1
   Kokkos::View<PetscInt *> i_local_d(Kokkos::view_alloc(exec, "i_local_d"), local_rows_z + 1);
   Kokkos::View<PetscInt *> i_nonlocal_d;
   if (mpi) i_nonlocal_d = Kokkos::View<PetscInt *>(Kokkos::view_alloc(exec, "i_nonlocal_d"), local_rows_z + 1);
   // Every entry is written by the count
   functor.rel_row_tol_d = PetscScalarKokkosView(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "rel_row_tol_d"), local_rows_z);
   functor.counts_l = i_local_d.data();
   if (mpi) functor.counts_nl = i_nonlocal_d.data();
   launch_r_from_z_drop<RFromZDropFunctor::CountTag>(functor, vector_length);

   // Inclusive scans of the row counts give the row pointers
   // Each index only reads and writes its own entry, so this is safe in place
   PetscInt nnzs_local = 0, nnzs_nonlocal = 0;
   Kokkos::parallel_scan(
      Kokkos::RangePolicy<>(exec, 0, local_rows_z),
      KOKKOS_LAMBDA(const PetscInt i, PetscInt &update, const bool final) {
         update += i_local_d(i + 1);
         if (final) i_local_d(i + 1) = update;
      },
      nnzs_local);
   if (mpi)
   {
      Kokkos::parallel_scan(
         Kokkos::RangePolicy<>(exec, 0, local_rows_z),
         KOKKOS_LAMBDA(const PetscInt i, PetscInt &update, const bool final) {
            update += i_nonlocal_d(i + 1);
            if (final) i_nonlocal_d(i + 1) = update;
         },
         nnzs_nonlocal);
   }

   // ~~~~~~~~~~~~
   // Fill the output i, j and values
   // ~~~~~~~~~~~~
   // Every entry is written by the fill
   Kokkos::View<PetscScalar *> a_local_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "a_local_d"), nnzs_local);
   Kokkos::View<PetscInt *> j_local_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "j_local_d"), nnzs_local);
   Kokkos::View<PetscScalar *> a_nonlocal_d;
   Kokkos::View<PetscInt *> j_nonlocal_d;
   if (mpi)
   {
      a_nonlocal_d = Kokkos::View<PetscScalar *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "a_nonlocal_d"), nnzs_nonlocal);
      j_nonlocal_d = Kokkos::View<PetscInt *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "j_nonlocal_d"), nnzs_nonlocal);
   }

   functor.out_l_i = i_local_d.data();
   functor.out_l_j = j_local_d.data();
   functor.out_l_a = a_local_d.data();
   if (mpi)
   {
      functor.out_nl_i = i_nonlocal_d.data();
      functor.out_nl_j = j_nonlocal_d.data();
      functor.out_nl_a = a_nonlocal_d.data();
   }
   launch_r_from_z_drop<RFromZDropFunctor::FillTag>(functor, vector_length);

   // Let's make sure everything on the device is finished
   exec.fence();

   // The identity was added at the end of each local row
   KokkosCsrMatrix csrmat_local = KokkosCsrMatrix("csrmat_local", local_rows_z, local_full_cols, a_local_d.extent(0), a_local_d, i_local_d, j_local_d);
   KokkosSparse::sort_crs_matrix(exec, csrmat_local);
   exec.fence();

   PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows_z, local_full_cols, i_local_d, j_local_d, a_local_d, &output_mat_local));

   PetscInt *orig_indices_array;
   PetscInt col_ao_output = 0;
   if (mpi)
   {
      // The nonlocal j are the nonlocal columns of Z, compact them to the ones we kept
      // This keeps their order, so the rows stay sorted
      // kept_cols_host holds which nonlocal columns of Z we kept, in increasing order
      PetscInt *kept_cols_host = NULL;
      // This fences internally
      rewrite_j_global_to_local(cols_ao, col_ao_output, j_nonlocal_d, &kept_cols_host);

      // The global columns of Z we kept
      PetscInt *kept_global_z = NULL;
      PetscCallVoid(PetscMalloc1(col_ao_output, &kept_global_z));
      for (PetscInt k = 0; k < col_ao_output; k++) kept_global_z[k] = colmap_input[kept_cols_host[k]];
      PetscCallVoid(PetscFree(kept_cols_host));

      // Look up the original fine indices of only the kept off-processor columns
      // Has to be called on every rank, even those without off-processor columns, as it is collective
      IS col_indices, orig_nonlocal;
      const PetscInt *orig_nonlocal_ptr;
      PetscCallVoid(ISCreateGeneral(MPI_COMM_MATRIX, col_ao_output, kept_global_z, PETSC_OWN_POINTER, &col_indices));
      PetscCallVoid(ISCreateSubIS(*is_fine, col_indices, &orig_nonlocal));
      PetscCallVoid(ISGetIndices(orig_nonlocal, &orig_nonlocal_ptr));

      // The garray of R is the original indices of the kept off-processor columns
      PetscInt *garray_host = NULL;
      PetscCallVoid(PetscMalloc1(col_ao_output, &garray_host));
      PetscCallVoid(PetscArraycpy(garray_host, orig_nonlocal_ptr, col_ao_output));

      PetscCallVoid(PetscMalloc1(cols_ad + col_ao_output, &orig_indices_array));
      PetscCallVoid(PetscArraycpy(orig_indices_array, fine_indices_ptr, cols_ad));
      PetscCallVoid(PetscArraycpy(orig_indices_array + cols_ad, orig_nonlocal_ptr, col_ao_output));
      PetscCallVoid(ISRestoreIndices(orig_nonlocal, &orig_nonlocal_ptr));
      PetscCallVoid(ISDestroy(&orig_nonlocal));
      PetscCallVoid(ISDestroy(&col_indices));

      PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows_z, col_ao_output, i_nonlocal_d, j_nonlocal_d, a_nonlocal_d, &output_mat_nonlocal));
      PetscCallVoid(MatCreateMPIAIJWithSeqAIJ(MPI_COMM_MATRIX, global_rows_z, global_full_cols, output_mat_local, output_mat_nonlocal, garray_host, output_mat));
   }
   else
   {
      PetscCallVoid(PetscMalloc1(cols_ad, &orig_indices_array));
      PetscCallVoid(PetscArraycpy(orig_indices_array, fine_indices_ptr, cols_ad));
      *output_mat = output_mat_local;
   }

   // The original fine indices of all the columns of the dropped Z, local then off-processor
   PetscCallVoid(ISCreateGeneral(MPI_COMM_MATRIX, cols_ad + col_ao_output, orig_indices_array, PETSC_OWN_POINTER, orig_fine_col_indices));

   PetscCallVoid(ISRestoreIndices(*is_fine, &fine_indices_ptr));
   PetscCallVoid(ISRestoreIndices(*is_coarse, &coarse_indices_ptr));

   PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local, &device_local_vals));
   if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal, &device_nonlocal_vals));
}

//------------------------------------------------------------------------------------------------------------------------

// Row kernels for compute_R_from_A_diag_kokkos
// Each team handles team_size coarse rows, one thread per row, with vector lanes over the
// entries in that row of A
// Z = -A_cf * D, with D the diagonal inverse of Aff, is never built: the entries of a coarse
// row of A in fine columns are Z's entries (in the same order), with the value -(a * d), which
// is what MatDiagonalScale followed by MatScale with -1 give
// The drop is the one in RFromZDropFunctor, with Z's global row the coarse index of the row
// and Z's global column the fine index of the column
struct RFromADiagFunctor {
   struct CountTag {};
   struct FillTag {};

   // Number of local coarse rows
   PetscInt nrows = 0;
   // Global coarse index of the first local coarse row (Z's row ownership start)
   PetscInt coarse_start = 0;
   PetscReal tol = 0.0;
   bool mpi = false;

   // Input local and nonlocal blocks of A
   const PetscInt *l_i = nullptr, *l_j = nullptr, *nl_i = nullptr, *nl_j = nullptr;
   Kokkos::View<const PetscScalar *> l_a, nl_a;
   // Local row of A of each local coarse row
   PetscIntKokkosView coarse_d;
   // For each local (l) and off-processor (nl) column of A, its global fine index or -1 if it
   // is a coarse point, and the entry of D for it
   PetscIntKokkosView fine_global_l_d, fine_global_nl_d;
   PetscScalarKokkosView d_l_d, d_nl_d;

   // Written by count, as in RFromZDropFunctor
   PetscScalarKokkosView rel_row_tol_d;
   PetscInt *counts_l = nullptr, *counts_nl = nullptr;

   // Fill: start of each output row, and the output j and values
   // The local j are local columns of A (and R), the nonlocal j are the nonlocal columns of A
   const PetscInt *out_l_i = nullptr, *out_nl_i = nullptr;
   PetscInt *out_l_j = nullptr, *out_nl_j = nullptr;
   PetscScalar *out_l_a = nullptr, *out_nl_a = nullptr;

   KOKKOS_INLINE_FUNCTION PetscScalar z_local(const PetscInt k) const
   {
      const PetscScalar val = l_a(k) * d_l_d(l_j[k]);
      return -val;
   }
   KOKKOS_INLINE_FUNCTION PetscScalar z_nonlocal(const PetscInt k) const
   {
      const PetscScalar val = nl_a(k) * d_nl_d(nl_j[k]);
      return -val;
   }
   // Is this entry of A in a fine column, and if so do we keep it
   KOKKOS_INLINE_FUNCTION bool keep_local(const PetscInt k, const PetscScalar row_tol, const PetscInt row_global) const
   {
      const PetscInt fine_col = fine_global_l_d(l_j[k]);
      if (fine_col < 0) return false;
      return Kokkos::abs(z_local(k)) >= row_tol || fine_col == row_global;
   }
   KOKKOS_INLINE_FUNCTION bool keep_nonlocal(const PetscInt k, const PetscScalar row_tol, const PetscInt row_global) const
   {
      const PetscInt fine_col = fine_global_nl_d(nl_j[k]);
      if (fine_col < 0) return false;
      return Kokkos::abs(z_nonlocal(k)) >= row_tol || fine_col == row_global;
   }

   KOKKOS_INLINE_FUNCTION void operator()(const CountTag &, const KokkosTeamMemberType &t) const
   {
      const PetscInt row_start = t.league_rank() * t.team_size();
      const PetscInt row_end = row_start + t.team_size() < nrows ? row_start + t.team_size() : nrows;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, row_start, row_end), [&](const PetscInt i) {
         const PetscInt row_global = i + coarse_start;
         const PetscInt row = coarse_d(i);
         const PetscInt l_start = l_i[row], l_len = l_i[row + 1] - l_start;
         const PetscInt nl_start = mpi ? nl_i[row] : 0, nl_len = mpi ? nl_i[row + 1] - nl_start : 0;

         // The biggest abs(val) of Z in the row, local and nonlocal
         // The Max reducer starts from the lowest value, so an empty block gives a negative max
         PetscScalar max_l, max_nl = Kokkos::reduction_identity<PetscScalar>::max();
         Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, l_len),
            [&](const PetscInt k, PetscScalar &thread_max) {
               if (fine_global_l_d(l_j[l_start + k]) < 0) return;
               const PetscScalar val = Kokkos::abs(z_local(l_start + k));
               if (val > thread_max) thread_max = val;
            }, Kokkos::Max<PetscScalar>(max_l));
         if (mpi) {
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, nl_len),
               [&](const PetscInt k, PetscScalar &thread_max) {
                  if (fine_global_nl_d(nl_j[nl_start + k]) < 0) return;
                  const PetscScalar val = Kokkos::abs(z_nonlocal(nl_start + k));
                  if (val > thread_max) thread_max = val;
               }, Kokkos::Max<PetscScalar>(max_nl));
         }
         const PetscScalar max_val = max_nl > max_l ? max_nl : max_l;
         const PetscScalar row_tol = tol * max_val;

         PetscInt count_l = 0, count_nl = 0;
         Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, l_len),
            [&](const PetscInt k, PetscInt &thread_count) {
               if (keep_local(l_start + k, row_tol, row_global)) thread_count++;
            }, count_l);
         if (mpi) {
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, nl_len),
               [&](const PetscInt k, PetscInt &thread_count) {
                  if (keep_nonlocal(nl_start + k, row_tol, row_global)) thread_count++;
               }, count_nl);
         }

         Kokkos::single(Kokkos::PerThread(t), [&]() {
            rel_row_tol_d(i) = row_tol;
            // One extra for the identity
            counts_l[i + 1] = count_l + 1;
            if (mpi) counts_nl[i + 1] = count_nl;
         });
      });
   }

   // The vector lanes do an exclusive scan over the kept flags to find their output position,
   // so the kept entries stay in the order of the row of A
   KOKKOS_INLINE_FUNCTION void operator()(const FillTag &, const KokkosTeamMemberType &t) const
   {
      const PetscInt row_start = t.league_rank() * t.team_size();
      const PetscInt row_end = row_start + t.team_size() < nrows ? row_start + t.team_size() : nrows;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, row_start, row_end), [&](const PetscInt i) {
         const PetscInt row_global = i + coarse_start;
         const PetscInt row = coarse_d(i);
         const PetscInt l_start = l_i[row], l_len = l_i[row + 1] - l_start;
         const PetscInt nl_start = mpi ? nl_i[row] : 0, nl_len = mpi ? nl_i[row + 1] - nl_start : 0;
         const PetscScalar row_tol = rel_row_tol_d(i);

         const PetscInt out_l_offset = out_l_i[i];
         Kokkos::parallel_scan(Kokkos::ThreadVectorRange(t, l_len),
            [&](const PetscInt k, PetscInt &partial_sum, const bool is_final) {
               if (keep_local(l_start + k, row_tol, row_global)) {
                  if (is_final) {
                     out_l_j[out_l_offset + partial_sum] = l_j[l_start + k];
                     out_l_a[out_l_offset + partial_sum] = z_local(l_start + k);
                  }
                  partial_sum++;
               }
            });
         if (mpi) {
            const PetscInt out_nl_offset = out_nl_i[i];
            Kokkos::parallel_scan(Kokkos::ThreadVectorRange(t, nl_len),
               [&](const PetscInt k, PetscInt &partial_sum, const bool is_final) {
                  if (keep_nonlocal(nl_start + k, row_tol, row_global)) {
                     if (is_final) {
                        out_nl_j[out_nl_offset + partial_sum] = nl_j[nl_start + k];
                        out_nl_a[out_nl_offset + partial_sum] = z_nonlocal(nl_start + k);
                     }
                     partial_sum++;
                  }
               });
         }

         // The identity goes at the end of the local row and we sort after
         Kokkos::single(Kokkos::PerThread(t), [&]() {
            out_l_j[out_l_i[i + 1] - 1] = row;
            out_l_a[out_l_i[i + 1] - 1] = 1.0;
         });
      });
   }
};

// Launches one of the RFromADiagFunctor kernels over all the coarse rows
template <class Tag>
static void launch_r_from_a_diag(const RFromADiagFunctor &functor, const int vector_length)
{
   if (functor.nrows == 0) return;
   auto exec = PetscGetKokkosExecutionSpace();
   Kokkos::TeamPolicy<Tag> probe(exec, 1, 1, vector_length);
   int team_size = std::max(1, 256 / vector_length);
   team_size = std::min(team_size, probe.team_size_max(functor, Kokkos::ParallelForTag()));
   const int league_size = static_cast<int>((functor.nrows + team_size - 1) / team_size);
   Kokkos::parallel_for(Kokkos::TeamPolicy<Tag>(exec, league_size, team_size, vector_length), functor);
}

//------------------------------------------------------------------------------------------------------------------------

// Build R straight from the square matrix A when Aff^-1 is the diagonal D (a fine-sized Kokkos
// vector of its entries), without building A_cf or Z
// Gives the same R as Z = -A_cf * D (MatMatMult with a MATDIAGONAL and MatScale), the drop
// with relative row tolerance tol, and compute_R_from_Z_kokkos with identity and without reuse
// The fine and coarse points of our_level come from the per-PCAIR IS views
PETSC_INTERN void compute_R_from_A_diag_kokkos(Mat *input_mat, void *kokkos_is_views_handle, const int our_level, \
                  Vec *inv_diag_vec, const PetscReal tol, Mat *output_mat)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols, rows_ao = 0, cols_ao = 0;
   MatType mat_type;
   Mat output_mat_local, output_mat_nonlocal;
   auto exec = PetscGetKokkosExecutionSpace();

   PetscCallVoid(MatGetType(*input_mat, &mat_type));
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;
   PetscCallVoid(PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX));
   PetscCallVoid(MatGetLocalSize(*input_mat, &local_rows, &local_cols));
   PetscCallVoid(MatGetSize(*input_mat, &global_rows, &global_cols));

   Mat mat_local = NULL, mat_nonlocal = NULL;
   const PetscInt *colmap_input = NULL;
   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*input_mat, &mat_local, &mat_nonlocal, &colmap_input));
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao));
   }
   else
   {
      mat_local = *input_mat;
   }

   // The local indices of the fine and coarse points, in the order of is_fine and is_coarse
   PetscIntKokkosView is_fine_d = VecISCopyLocal_kokkos_get_view(kokkos_is_views_handle, our_level, 1);
   PetscIntKokkosView is_coarse_d = VecISCopyLocal_kokkos_get_view(kokkos_is_views_handle, our_level, 0);
   const PetscInt local_fine = is_fine_d.extent(0), local_coarse = is_coarse_d.extent(0);

   // The global fine and coarse indices start where the previous ranks' end
   PetscInt local_counts[2] = {local_fine, local_coarse}, starts[2] = {0, 0}, global_counts[2] = {0, 0};
   PetscCallMPIAbort(MPI_COMM_MATRIX, MPI_Scan(local_counts, starts, 2, MPIU_INT, MPI_SUM, MPI_COMM_MATRIX));
   PetscCallMPIAbort(MPI_COMM_MATRIX, MPI_Allreduce(local_counts, global_counts, 2, MPIU_INT, MPI_SUM, MPI_COMM_MATRIX));
   const PetscInt fine_start = starts[0] - local_fine, coarse_start = starts[1] - local_coarse;
   const PetscInt global_coarse = global_counts[1];

   // The global fine index and the entry of D of each local column of A
   ConstPetscScalarKokkosView inv_diag_d;
   PetscCallVoid(VecGetKokkosView(*inv_diag_vec, &inv_diag_d));
   PetscIntKokkosView fine_global_l_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "fine_global_l_d"), local_cols);
   PetscScalarKokkosView d_l_d(Kokkos::view_alloc(exec, "d_l_d"), local_cols);
   Kokkos::deep_copy(exec, fine_global_l_d, -1);
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, local_fine), KOKKOS_LAMBDA(PetscInt i) {
         fine_global_l_d(is_fine_d(i)) = fine_start + i;
         d_l_d(is_fine_d(i)) = inv_diag_d(i);
   });

   // And of each off-processor column of A, through the matrix mult SF
   PetscIntKokkosView fine_global_nl_d;
   PetscScalarKokkosView d_nl_d;
   if (mpi)
   {
      PetscSF sf = NULL;
      PetscCallVoid(MatGetMultPetscSF(*input_mat, &sf));
      fine_global_nl_d = PetscIntKokkosView("fine_global_nl_d", cols_ao);
      d_nl_d = PetscScalarKokkosView("d_nl_d", cols_ao);
      // While Begin/End is in flight, do not touch the send/recv buffers
      PetscCallVoid(sf_sync_before_begin_kokkos());
      PetscCallVoid(PetscSFBcastWithMemTypeBegin(sf, MPIU_INT,
                  PETSC_MEMTYPE_KOKKOS, fine_global_l_d.data(),
                  PETSC_MEMTYPE_KOKKOS, fine_global_nl_d.data(),
                  MPI_REPLACE));
      PetscCallVoid(PetscSFBcastEnd(sf, MPIU_INT, fine_global_l_d.data(), fine_global_nl_d.data(), MPI_REPLACE));
      PetscCallVoid(PetscSFBcastWithMemTypeBegin(sf, MPIU_SCALAR,
                  PETSC_MEMTYPE_KOKKOS, d_l_d.data(),
                  PETSC_MEMTYPE_KOKKOS, d_nl_d.data(),
                  MPI_REPLACE));
      PetscCallVoid(PetscSFBcastEnd(sf, MPIU_SCALAR, d_l_d.data(), d_nl_d.data(), MPI_REPLACE));
      PetscCallVoid(sf_sync_after_end_kokkos());
   }

   // ~~~~~~~~~~~~
   // Get pointers to the i,j on the device and Kokkos views to the values
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, NULL, &mtype));
   Kokkos::View<const PetscScalar *> device_local_vals;
   Kokkos::View<const PetscScalar *> device_nonlocal_vals;
   PetscCallVoid(MatSeqAIJGetKokkosView(mat_local, &device_local_vals));
   if (mpi) PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal, &device_nonlocal_vals));

   RFromADiagFunctor functor;
   functor.nrows = local_coarse;
   functor.coarse_start = coarse_start;
   functor.tol = tol;
   functor.mpi = mpi;
   functor.l_i = device_local_i;
   functor.l_j = device_local_j;
   functor.l_a = device_local_vals;
   functor.coarse_d = is_coarse_d;
   functor.fine_global_l_d = fine_global_l_d;
   functor.d_l_d = d_l_d;
   if (mpi)
   {
      functor.nl_i = device_nonlocal_i;
      functor.nl_j = device_nonlocal_j;
      functor.nl_a = device_nonlocal_vals;
      functor.fine_global_nl_d = fine_global_nl_d;
      functor.d_nl_d = d_nl_d;
   }

   // The vector length comes from the average row length of A (local plus nonlocal)
   // The host backends run the vector lanes serially, so there it makes no difference
   MatInfo info;
   PetscCallVoid(MatGetInfo(*input_mat, MAT_LOCAL, &info));
   const double avg_row_nnz = local_rows > 0 ? info.nz_used / local_rows : 0.0;
   int vector_length = 1;
   while (2 * vector_length <= avg_row_nnz && vector_length < 32) vector_length *= 2;
   vector_length = std::min(vector_length, Kokkos::TeamPolicy<>::vector_length_max());

   // ~~~~~~~~~~~~
   // Count the entries of R in each row
   // ~~~~~~~~~~~~
   // Zero initialised on exec, the counts go into entry row + 1
   Kokkos::View<PetscInt *> i_local_d(Kokkos::view_alloc(exec, "i_local_d"), local_coarse + 1);
   Kokkos::View<PetscInt *> i_nonlocal_d;
   if (mpi) i_nonlocal_d = Kokkos::View<PetscInt *>(Kokkos::view_alloc(exec, "i_nonlocal_d"), local_coarse + 1);
   // Every entry is written by the count
   functor.rel_row_tol_d = PetscScalarKokkosView(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "rel_row_tol_d"), local_coarse);
   functor.counts_l = i_local_d.data();
   if (mpi) functor.counts_nl = i_nonlocal_d.data();
   launch_r_from_a_diag<RFromADiagFunctor::CountTag>(functor, vector_length);

   // Inclusive scans of the row counts give the row pointers
   // Each index only reads and writes its own entry, so this is safe in place
   PetscInt nnzs_local = 0, nnzs_nonlocal = 0;
   Kokkos::parallel_scan(
      Kokkos::RangePolicy<>(exec, 0, local_coarse),
      KOKKOS_LAMBDA(const PetscInt i, PetscInt &update, const bool final) {
         update += i_local_d(i + 1);
         if (final) i_local_d(i + 1) = update;
      },
      nnzs_local);
   if (mpi)
   {
      Kokkos::parallel_scan(
         Kokkos::RangePolicy<>(exec, 0, local_coarse),
         KOKKOS_LAMBDA(const PetscInt i, PetscInt &update, const bool final) {
            update += i_nonlocal_d(i + 1);
            if (final) i_nonlocal_d(i + 1) = update;
         },
         nnzs_nonlocal);
   }

   // ~~~~~~~~~~~~
   // Fill the output i, j and values
   // ~~~~~~~~~~~~
   // Every entry is written by the fill
   Kokkos::View<PetscScalar *> a_local_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "a_local_d"), nnzs_local);
   Kokkos::View<PetscInt *> j_local_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "j_local_d"), nnzs_local);
   Kokkos::View<PetscScalar *> a_nonlocal_d;
   Kokkos::View<PetscInt *> j_nonlocal_d;
   if (mpi)
   {
      a_nonlocal_d = Kokkos::View<PetscScalar *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "a_nonlocal_d"), nnzs_nonlocal);
      j_nonlocal_d = Kokkos::View<PetscInt *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "j_nonlocal_d"), nnzs_nonlocal);
   }

   functor.out_l_i = i_local_d.data();
   functor.out_l_j = j_local_d.data();
   functor.out_l_a = a_local_d.data();
   if (mpi)
   {
      functor.out_nl_i = i_nonlocal_d.data();
      functor.out_nl_j = j_nonlocal_d.data();
      functor.out_nl_a = a_nonlocal_d.data();
   }
   launch_r_from_a_diag<RFromADiagFunctor::FillTag>(functor, vector_length);

   // Let's make sure everything on the device is finished
   exec.fence();
   PetscCallVoid(VecRestoreKokkosView(*inv_diag_vec, &inv_diag_d));

   // The identity was added at the end of each local row
   KokkosCsrMatrix csrmat_local = KokkosCsrMatrix("csrmat_local", local_coarse, local_cols, a_local_d.extent(0), a_local_d, i_local_d, j_local_d);
   KokkosSparse::sort_crs_matrix(exec, csrmat_local);
   exec.fence();

   PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_coarse, local_cols, i_local_d, j_local_d, a_local_d, &output_mat_local));

   if (mpi)
   {
      // The nonlocal j are the nonlocal columns of A, compact them to the ones we kept
      // This keeps their order, so the rows stay in the order of A
      // kept_cols_host holds which nonlocal columns of A we kept, in increasing order
      PetscInt *kept_cols_host = NULL;
      PetscInt col_ao_output = 0;
      // This fences internally
      rewrite_j_global_to_local(cols_ao, col_ao_output, j_nonlocal_d, &kept_cols_host);

      // The garray of R is the global columns of A we kept
      PetscInt *garray_host = NULL;
      PetscCallVoid(PetscMalloc1(col_ao_output, &garray_host));
      for (PetscInt k = 0; k < col_ao_output; k++) garray_host[k] = colmap_input[kept_cols_host[k]];
      PetscCallVoid(PetscFree(kept_cols_host));

      PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_coarse, col_ao_output, i_nonlocal_d, j_nonlocal_d, a_nonlocal_d, &output_mat_nonlocal));
      PetscCallVoid(MatCreateMPIAIJWithSeqAIJ(MPI_COMM_MATRIX, global_coarse, global_cols, output_mat_local, output_mat_nonlocal, garray_host, output_mat));
   }
   else
   {
      *output_mat = output_mat_local;
   }

   PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local, &device_local_vals));
   if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal, &device_nonlocal_vals));
}

//------------------------------------------------------------------------------------------------------------------------

// Stick Z in a full sized R but with kokkos - keeping everything on the device
// If drop_int is set, small entries of Z are dropped first (see compute_R_from_Z_drop_kokkos)
// without building the dropped Z, only without reuse
PETSC_INTERN void compute_R_from_Z_kokkos(Mat *input_mat, PetscInt global_row_start, IS *is_fine, \
                  IS *is_coarse, IS *orig_fine_col_indices, int identity_int, int reuse_int, int reuse_indices_int, \
                  int drop_int, PetscReal tol, Mat *output_mat)
{
   if (drop_int)
   {
      compute_R_from_Z_drop_kokkos(input_mat, global_row_start, is_fine, is_coarse, orig_fine_col_indices, tol, output_mat);
      return;
   }

   MPI_Comm MPI_COMM_MATRIX;
   PetscInt global_row_start_Z, global_row_end_plus_one_Z;
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

   auto exec = PetscGetKokkosExecutionSpace();

   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*input_mat, &mat_local, &mat_nonlocal, &colmap_input));
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao));

      // We also copy the input mat colmap over to the device as we need it
      colmap_input_h = PetscIntConstKokkosViewHost(colmap_input, cols_ao);
      colmap_input_d = PetscIntKokkosView("colmap_input_d", cols_ao);
      Kokkos::deep_copy(exec, colmap_input_d, colmap_input_h);
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

   PetscCallVoid(MatGetType(*input_mat, &mat_type));
   PetscCallVoid(MatGetSize(mat_local, &rows_ad, &cols_ad));   

   // We can reuse the orig_fine_col_indices as they can be expensive to generate in parallel
   if (!reuse_indices_int)
   {
      // The local columns of Z are the local fine points in order, so their original
      // indices are just the local entries of is_fine and need no comms
      // Only the off-processor columns have to be looked up on other ranks
      PetscInt *orig_indices_array;
      const PetscInt *fine_indices_local;
      size_cols = mpi ? cols_ad + cols_ao : cols_ad;

      PetscCallVoid(PetscMalloc1(size_cols, &orig_indices_array));
      PetscCallVoid(ISGetIndices(*is_fine, &fine_indices_local));
      PetscCallVoid(PetscArraycpy(orig_indices_array, fine_indices_local, cols_ad));
      PetscCallVoid(ISRestoreIndices(*is_fine, &fine_indices_local));

      if (mpi)
      {
         IS col_indices, orig_nonlocal;
         const PetscInt *orig_nonlocal_ptr;

         // Has to be called on every rank, even those without off-processor columns, as it is collective
         PetscCallVoid(ISCreateGeneral(MPI_COMM_MATRIX, cols_ao, colmap_input, PETSC_USE_POINTER, &col_indices));
         // This could be expensive as the number of off-processor columns in Z grows!
         PetscCallVoid(ISCreateSubIS(*is_fine, col_indices, &orig_nonlocal));
         PetscCallVoid(ISGetIndices(orig_nonlocal, &orig_nonlocal_ptr));
         PetscCallVoid(PetscArraycpy(orig_indices_array + cols_ad, orig_nonlocal_ptr, cols_ao));
         PetscCallVoid(ISRestoreIndices(orig_nonlocal, &orig_nonlocal_ptr));
         PetscCallVoid(ISDestroy(&orig_nonlocal));
         PetscCallVoid(ISDestroy(&col_indices));
      }

      // The original fine indices of all the columns of Z, local then off-processor
      PetscCallVoid(ISCreateGeneral(MPI_COMM_MATRIX, size_cols, orig_indices_array, PETSC_OWN_POINTER, orig_fine_col_indices));
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
   Kokkos::deep_copy(exec, fine_view_d, fine_view_h);     
   // Log copy with petsc
   size_t bytes = fine_view_h.extent(0) * sizeof(PetscInt);
   PetscCallVoid(PetscLogCpuToGpu(bytes));   
   Kokkos::deep_copy(exec, coarse_view_d, coarse_view_h);
   bytes = coarse_view_h.extent(0) * sizeof(PetscInt);
   PetscCallVoid(PetscLogCpuToGpu(bytes));        
   Kokkos::deep_copy(exec, orig_view_d, orig_view_h); 
   bytes = orig_view_h.extent(0) * sizeof(PetscInt);
   PetscCallVoid(PetscLogCpuToGpu(bytes));

   // ~~~~~~~~~~~~
   // Get pointers to the i,j on the device and Kokkos views to the values
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, NULL, &mtype));
   Kokkos::View<const PetscScalar *> device_local_vals;
   Kokkos::View<const PetscScalar *> device_nonlocal_vals;
   PetscCallVoid(MatSeqAIJGetKokkosView(mat_local, &device_local_vals));
   if (mpi) PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal, &device_nonlocal_vals));

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
      Kokkos::deep_copy(exec, nnz_match_local_row_d, 0);         
      if (mpi) nnz_match_nonlocal_row_d = PetscIntKokkosView("nnz_match_nonlocal_row_d", local_rows_z);                  

      // ~~~~~~~~~~~~
      // Need to count the number of nnzs we end up with, on each row and in total
      // ~~~~~~~~~~~~
      // Loop over the rows of Z
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, local_rows_z), KOKKOS_LAMBDA(PetscInt i) {

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
      Kokkos::parallel_reduce ("ReductionLocal", Kokkos::RangePolicy<>(exec, 0, local_rows_z), KOKKOS_LAMBDA (const PetscInt i, PetscInt& update) {
         update += nnz_match_local_row_d(i);
      }, nnzs_match_local);
      if (mpi)
      {
         Kokkos::parallel_reduce ("ReductionNonLocal", Kokkos::RangePolicy<>(exec, 0, local_rows_z), KOKKOS_LAMBDA (const PetscInt i, PetscInt& update) {
            update += nnz_match_nonlocal_row_d(i);
         }, nnzs_match_nonlocal);
      }

      // ~~~~~~~~~~~~

      // Need to do a scan on nnz_match_local_row_d to get where each row starts
      Kokkos::parallel_scan (Kokkos::RangePolicy<>(exec, 0, local_rows_z), KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
         // Inclusive scan
         update += nnz_match_local_row_d(i);
         if (final) {
            nnz_match_local_row_d(i) = update; // only update array on final pass
         }
      });
      if (mpi)
      {
         // Need to do a scan on nnz_match_nonlocal_row_d to get where each row starts
         Kokkos::parallel_scan (Kokkos::RangePolicy<>(exec, 0, local_rows_z), KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
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
      Kokkos::deep_copy(exec, Kokkos::subview(i_local_d, 0), 0);       

      // we also have to go and build the a, i, j for the non-local off-diagonal block
      if (mpi) 
      {
         // Non-local 
         a_nonlocal_d = Kokkos::View<PetscScalar *>("a_nonlocal_d", nnzs_match_nonlocal);
         i_nonlocal_d = Kokkos::View<PetscInt *>("i_nonlocal_d", local_rows_z+1);
         j_nonlocal_d = Kokkos::View<PetscInt *>("j_nonlocal_d", nnzs_match_nonlocal);  

         // Initialize first entry to zero - the rest get set below
         Kokkos::deep_copy(exec, Kokkos::subview(i_nonlocal_d, 0), 0);                
      }  

      // ~~~~~~~~~~~~~~~
      // Create i indices
      // ~~~~~~~~~~~~~~~
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, local_rows_z), KOKKOS_LAMBDA(PetscInt i) {

            // Row index is simple
            PetscInt row_index = i;       

            // The start of our row index comes from the scan
            i_local_d(row_index + 1) = nnz_match_local_row_d(row_index);   
            if (mpi) i_nonlocal_d(row_index + 1) = nnz_match_nonlocal_row_d(row_index);         
      });            


      // Loop over the rows of Z
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(exec, local_rows_z, Kokkos::AUTO()),
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
               a_local_d(i_local_d(row_index) + j) = device_local_vals(device_local_i[i] + j);
                     
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
                  a_nonlocal_d(i_nonlocal_d(row_index) + j) = device_nonlocal_vals(device_nonlocal_i[i] + j);
                        
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
         PetscCallVoid(MatMPIAIJGetSeqAIJ(*output_mat, &mat_local_output, &mat_nonlocal_output, NULL));
      }
      else
      {
         mat_local_output = *output_mat;
      }     
      // Get device pointers for the existing i and j arrays
      const PetscInt *device_local_i_output = nullptr, *device_local_j_output = nullptr, *device_nonlocal_i_output = nullptr;
      PetscMemType mtype;
      PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_output, &device_local_i_output, &device_local_j_output, NULL, &mtype));
      if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_output, &device_nonlocal_i_output, NULL, NULL, &mtype));

      // Read-write Kokkos views to the values - we update the Z block, the identity entries are preserved
      Kokkos::View<PetscScalar *> device_local_a_output;
      Kokkos::View<PetscScalar *> device_nonlocal_a_output;
      PetscCallVoid(MatSeqAIJGetKokkosView(mat_local_output, &device_local_a_output));
      if (mpi) PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal_output, &device_nonlocal_a_output));

      // Only have to write Z but have to be careful as we have the identity mixed
      // in there
      // Loop over the rows of Z - annoying we have const views as this is just the same loop as above
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(exec, local_rows_z, Kokkos::AUTO()),
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
               // Only applies if the identity was included when the matrix was created
               if (identity_int && device_local_j_output[device_local_i_output[row_index] + j] >= coarse_view_d(i) - global_row_start) offset = 1;
               device_local_a_output(device_local_i_output[row_index] + j + offset) = device_local_vals(device_local_i[i] + j);
            });

            // For over nonlocal columns - copy in Z - identical structure in the off-diag block
            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];

               Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {

                  // We keep the existing local indices in the off-diagonal block here
                  // we have all the same columns as Z and hence the same garray
                  device_nonlocal_a_output(device_nonlocal_i_output[row_index] + j) = device_nonlocal_vals(device_nonlocal_i[i] + j);

               });
            }
      });

      Kokkos::fence();

      // The matching restore handles MatSeqAIJKokkosModifyDevice (clears sync state,
      // marks device modified, invalidates transpose/hermitian, bumps object state).
      PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local_output, &device_local_a_output));
      if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal_output, &device_nonlocal_a_output));

    }

   // ~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~   

   if (!reuse_int)
   {
      // Let's make sure everything on the device is finished
      Kokkos::fence();   

      // Now we have to sort the local column indices, as we add in the identity at the 
      // end of our local j indices      
      KokkosCsrMatrix csrmat_local = KokkosCsrMatrix("csrmat_local", local_rows_z, local_full_cols, a_local_d.extent(0), a_local_d, i_local_d, j_local_d);  
      KokkosSparse::sort_crs_matrix(exec, csrmat_local);
      
      // Let's make sure everything on the device is finished
      Kokkos::fence();       
      
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

   PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local, &device_local_vals));
   if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal, &device_nonlocal_vals));

   return;
}
