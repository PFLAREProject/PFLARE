// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"

//------------------------------------------------------------------------------------------------------------------------

// Compute lAIR Z matrix with kokkos - keeping everything on the device
// For each row i of Z:
//   1. Get J indices from sparsity_mat_cf row i (sorted global indices)
//   2. Build RHS from A_cf row i intersected with J
//   3. Build dense A_ff(J,J)^T
//   4. Solve A_ff(J,J)^T * z = -A_cf(i,J)^T
//   5. Write solution to Z row i (using permutation to map sorted→original order)
PETSC_INTERN void calculate_and_build_sai_z_kokkos(Mat *A_ff, Mat *A_cf, Mat *sparsity_mat_cf,
               const int reuse_int_reuse_mat, Mat *reuse_mat, Mat *z_mat,
               const int no_approx_solve_int)
{
   // Threshold above which we switch a row from dense direct solve (TeamGesv)
   // to dense Jacobi iteration. Mirrors the CPU code in src/SAI_Z.F90 which
   // switches at j_size > 40 (see calculate_and_build_sai_z_cpu).
   const PetscInt iter_threshold = 40;
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows_cf, local_cols_cf;
   PetscInt local_rows_ff, local_cols_ff;
   PetscInt global_row_start_ff_temp, global_row_end_plus_one_ff_temp;
   PetscInt global_row_start_cf_temp, global_row_end_plus_one_cf_temp;
   PetscInt rows_ao_sparsity, cols_ao_sparsity, rows_ad_sparsity, cols_ad_sparsity;
   PetscInt cols_ao_cf = 0, cols_ao_ff = 0;
   MatType mat_type;
   PetscInt one = 1;
   bool deallocate_submatrices = false;

   PetscCallVoid(MatGetType(*A_ff, &mat_type));
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   // Local and nonlocal parts of each matrix
   Mat mat_local_sparsity = NULL, mat_nonlocal_sparsity = NULL;
   Mat mat_local_ff = NULL, mat_nonlocal_ff = NULL;
   Mat mat_local_cf = NULL, mat_nonlocal_cf = NULL;
   Mat mat_local_z = NULL, mat_nonlocal_z = NULL;

   // Get the comm
   PetscCallVoid(PetscObjectGetComm((PetscObject)*A_ff, &MPI_COMM_MATRIX));
   // A_cf is C-rows x F-cols, A_ff is F-rows x F-cols
   PetscCallVoid(MatGetLocalSize(*A_cf, &local_rows_cf, &local_cols_cf));
   PetscCallVoid(MatGetLocalSize(*A_ff, &local_rows_ff, &local_cols_ff));
   PetscCallVoid(MatGetOwnershipRange(*A_ff, &global_row_start_ff_temp, &global_row_end_plus_one_ff_temp));
   PetscCallVoid(MatGetOwnershipRange(*sparsity_mat_cf, &global_row_start_cf_temp, &global_row_end_plus_one_cf_temp));
   const PetscInt global_row_start_ff = global_row_start_ff_temp;
   (void)global_row_start_cf_temp;

   // ~~~~~~~~~~~~~~
   // Communication setup: fetch non-local rows of A_ff
   // Same pattern as Gmres_Polyk.kokkos.cxx and SAI_Z.F90
   // ~~~~~~~~~~~~~~

   PetscInt *col_indices_off_proc_array;
   const PetscInt *colmap_sparsity;
   const PetscInt *colmap_ff;
   const PetscInt *colmap_cf;
   IS col_indices, row_indices;
   Mat *submatrices;
   PetscInt size_cols;

   cols_ao_sparsity = 0;
   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*sparsity_mat_cf, &mat_local_sparsity, &mat_nonlocal_sparsity, &colmap_sparsity));
      PetscCallVoid(MatGetSize(mat_nonlocal_sparsity, &rows_ao_sparsity, &cols_ao_sparsity));
      PetscCallVoid(MatGetSize(mat_local_sparsity, &rows_ad_sparsity, &cols_ad_sparsity));

      PetscCallVoid(MatMPIAIJGetSeqAIJ(*A_ff, &mat_local_ff, &mat_nonlocal_ff, &colmap_ff));
      PetscInt rows_ao_ff;
      PetscCallVoid(MatGetSize(mat_nonlocal_ff, &rows_ao_ff, &cols_ao_ff));

      PetscCallVoid(MatMPIAIJGetSeqAIJ(*A_cf, &mat_local_cf, &mat_nonlocal_cf, &colmap_cf));
      PetscInt rows_ao_cf;
      PetscCallVoid(MatGetSize(mat_nonlocal_cf, &rows_ao_cf, &cols_ao_cf));

      // Build col_indices_off_proc_array by merging local F-indices and colmap_sparsity
      // Local F-indices are [global_row_start_ff..global_row_start_ff+cols_ad_sparsity-1]
      // colmap_sparsity entries are outside the local range and sorted
      // We need a proper sorted merge since the submatrix expects sorted column indices
      PetscCallVoid(PetscMalloc1(cols_ad_sparsity + cols_ao_sparsity, &col_indices_off_proc_array));
      size_cols = cols_ad_sparsity + cols_ao_sparsity;

      // Merge the two sorted arrays: local indices and colmap
      {
         PetscInt idx_local = 0, idx_colmap = 0, idx_out = 0;
         while (idx_local < cols_ad_sparsity && idx_colmap < cols_ao_sparsity)
         {
            PetscInt local_val = global_row_start_ff + idx_local;
            if (local_val <= colmap_sparsity[idx_colmap])
            {
               col_indices_off_proc_array[idx_out++] = local_val;
               // Skip duplicate if colmap has the same value (shouldn't happen for off-proc)
               if (local_val == colmap_sparsity[idx_colmap]) idx_colmap++;
               idx_local++;
            }
            else
            {
               col_indices_off_proc_array[idx_out++] = colmap_sparsity[idx_colmap++];
            }
         }
         while (idx_local < cols_ad_sparsity)
         {
            col_indices_off_proc_array[idx_out++] = global_row_start_ff + idx_local;
            idx_local++;
         }
         while (idx_colmap < cols_ao_sparsity)
         {
            col_indices_off_proc_array[idx_out++] = colmap_sparsity[idx_colmap++];
         }
         size_cols = idx_out;
      }

      // Create IS for the submatrix extraction
      PetscCallVoid(ISCreateGeneral(PETSC_COMM_SELF, size_cols,
                  col_indices_off_proc_array, PETSC_USE_POINTER, &col_indices));
      PetscCallVoid(ISCreateGeneral(PETSC_COMM_SELF, cols_ao_sparsity,
                  colmap_sparsity, PETSC_USE_POINTER, &row_indices));

      // Fetch the non-local rows of A_ff
      PetscCallVoid(MatSetOption(*A_ff, MAT_SUBMAT_SINGLEIS, PETSC_TRUE));
      if (!reuse_int_reuse_mat)
      {
         PetscCallVoid(MatCreateSubMatrices(*A_ff, one, &row_indices, &col_indices, MAT_INITIAL_MATRIX, &submatrices));
         *reuse_mat = submatrices[0];
      }
      else
      {
         submatrices = new Mat[1];
         deallocate_submatrices = true;
         submatrices[0] = *reuse_mat;
         PetscCallVoid(MatCreateSubMatrices(*A_ff, one, &row_indices, &col_indices, MAT_REUSE_MATRIX, &submatrices));
      }
      PetscCallVoid(ISDestroy(&col_indices));
      PetscCallVoid(ISDestroy(&row_indices));
   }
   // Serial case
   else
   {
      submatrices = new Mat[1];
      deallocate_submatrices = true;
      submatrices[0] = *A_ff;
      mat_local_ff = *A_ff;
      mat_local_cf = *A_cf;
      mat_local_sparsity = *sparsity_mat_cf;
      cols_ad_sparsity = local_cols_ff;
      PetscCallVoid(PetscMalloc1(local_cols_ff, &col_indices_off_proc_array));
      for (PetscInt i = 0; i < local_cols_ff; i++)
      {
         col_indices_off_proc_array[i] = i;
      }
      size_cols = local_cols_ff;
   }

   // Get the Z output matrix parts
   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*z_mat, &mat_local_z, &mat_nonlocal_z, NULL));
   }
   else
   {
      mat_local_z = *z_mat;
   }

   auto exec = PetscGetKokkosExecutionSpace();

   // ~~~~~~~~~~~~~~
   // Copy colmaps to device for global index conversion in the kernel
   // colmap_cf: converts A_cf nonlocal column indices to global
   // colmap_ff: converts A_ff nonlocal column indices to global
   // colmap_sparsity: converts sparsity nonlocal column indices to global
   //                  and used to find non-local rows of A_ff in the submatrix
   // col_indices_off_proc: converts submatrix column indices to global
   // ~~~~~~~~~~~~~~

   auto colmap_cf_d = PetscIntKokkosView("colmap_cf_d", mpi ? cols_ao_cf : 1);
   if (mpi && cols_ao_cf > 0)
   {
      auto colmap_cf_h = PetscIntConstKokkosViewHost(colmap_cf, cols_ao_cf);
      Kokkos::deep_copy(exec, colmap_cf_d, colmap_cf_h);
      PetscCallVoid(PetscLogCpuToGpu(cols_ao_cf * sizeof(PetscInt)));
   }

   auto colmap_ff_d = PetscIntKokkosView("colmap_ff_d", mpi ? cols_ao_ff : 1);
   if (mpi && cols_ao_ff > 0)
   {
      auto colmap_ff_h = PetscIntConstKokkosViewHost(colmap_ff, cols_ao_ff);
      Kokkos::deep_copy(exec, colmap_ff_d, colmap_ff_h);
      PetscCallVoid(PetscLogCpuToGpu(cols_ao_ff * sizeof(PetscInt)));
   }

   // Copy col_indices_off_proc_array to device (for converting submatrix columns to global)
   auto col_indices_off_proc_d = PetscIntKokkosView("col_indices_off_proc_d", size_cols);
   {
      auto col_indices_off_proc_h = PetscIntConstKokkosViewHost(col_indices_off_proc_array, size_cols);
      Kokkos::deep_copy(exec, col_indices_off_proc_d, col_indices_off_proc_h);
      PetscCallVoid(PetscLogCpuToGpu(size_cols * sizeof(PetscInt)));
   }

   // Copy colmap_sparsity to device for non-local row lookup and global index conversion
   auto colmap_sparsity_d = PetscIntKokkosView("colmap_sparsity_d", mpi ? cols_ao_sparsity : 1);
   if (mpi && cols_ao_sparsity > 0)
   {
      auto colmap_sparsity_h = PetscIntConstKokkosViewHost(colmap_sparsity, cols_ao_sparsity);
      Kokkos::deep_copy(exec, colmap_sparsity_d, colmap_sparsity_h);
      PetscCallVoid(PetscLogCpuToGpu(cols_ao_sparsity * sizeof(PetscInt)));
   }

   // ~~~~~~~~~~~~~~
   // Get device CSR pointers for i,j and Kokkos views to the values
   // ~~~~~~~~~~~~~~
   PetscMemType mtype;

   // Submatrix (non-local rows of A_ff)
   const PetscInt *device_submat_i = nullptr, *device_submat_j = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(submatrices[0], &device_submat_i, &device_submat_j, NULL, &mtype));
   Kokkos::View<const PetscScalar *> device_submat_vals;
   PetscCallVoid(MatSeqAIJGetKokkosView(submatrices[0], &device_submat_vals));

   // A_ff local + nonlocal
   const PetscInt *device_local_i_ff = nullptr, *device_local_j_ff = nullptr;
   const PetscInt *device_nonlocal_i_ff = nullptr, *device_nonlocal_j_ff = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_ff, &device_local_i_ff, &device_local_j_ff, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_ff, &device_nonlocal_i_ff, &device_nonlocal_j_ff, NULL, &mtype));
   Kokkos::View<const PetscScalar *> device_local_vals_ff;
   Kokkos::View<const PetscScalar *> device_nonlocal_vals_ff;
   PetscCallVoid(MatSeqAIJGetKokkosView(mat_local_ff, &device_local_vals_ff));
   if (mpi) PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal_ff, &device_nonlocal_vals_ff));

   // A_cf local + nonlocal
   const PetscInt *device_local_i_cf = nullptr, *device_local_j_cf = nullptr;
   const PetscInt *device_nonlocal_i_cf = nullptr, *device_nonlocal_j_cf = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_cf, &device_local_i_cf, &device_local_j_cf, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_cf, &device_nonlocal_i_cf, &device_nonlocal_j_cf, NULL, &mtype));
   Kokkos::View<const PetscScalar *> device_local_vals_cf;
   Kokkos::View<const PetscScalar *> device_nonlocal_vals_cf;
   PetscCallVoid(MatSeqAIJGetKokkosView(mat_local_cf, &device_local_vals_cf));
   if (mpi) PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal_cf, &device_nonlocal_vals_cf));

   // Sparsity matrix local + nonlocal (values not used in this routine)
   const PetscInt *device_local_i_sparsity = nullptr, *device_local_j_sparsity = nullptr;
   const PetscInt *device_nonlocal_i_sparsity = nullptr, *device_nonlocal_j_sparsity = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_sparsity, &device_local_i_sparsity, &device_local_j_sparsity, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_sparsity, &device_nonlocal_i_sparsity, &device_nonlocal_j_sparsity, NULL, &mtype));

   // Z output local + nonlocal - every entry is overwritten by the solve below
   const PetscInt *device_local_i_z = nullptr, *device_local_j_z = nullptr;
   const PetscInt *device_nonlocal_i_z = nullptr, *device_nonlocal_j_z = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_z, &device_local_i_z, &device_local_j_z, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_z, &device_nonlocal_i_z, &device_nonlocal_j_z, NULL, &mtype));
   Kokkos::View<PetscScalar *> device_local_vals_z;
   Kokkos::View<PetscScalar *> device_nonlocal_vals_z;
   PetscCallVoid(MatSeqAIJGetKokkosViewWrite(mat_local_z, &device_local_vals_z));
   if (mpi) PetscCallVoid(MatSeqAIJGetKokkosViewWrite(mat_nonlocal_z, &device_nonlocal_vals_z));

   // ~~~~~~~~~~~~~~
   // Find per-row j_size = local_nnz + nonlocal_nnz and split into:
   //   sparsity_max_nnz_direct - max j_size over rows handled by TeamGesv
   //   sparsity_max_nnz_iter   - max j_size over rows handled by Jacobi
   //   count_iter              - number of rows above threshold
   // When no_approx_solve_int is set, all rows go to the direct kernel
   // (used by PFLARE_KOKKOS_DEBUG=1 so CPU and Kokkos sides both do direct).
   // ~~~~~~~~~~~~~~
   PetscInt sparsity_max_nnz_direct = 0;
   PetscInt sparsity_max_nnz_iter = 0;
   PetscInt count_iter = 0;
   const bool iter_enabled = (no_approx_solve_int == 0);
   if (local_rows_cf > 0)
   {
      Kokkos::parallel_reduce("FindMaxNNZSparsitySplit", Kokkos::RangePolicy<>(exec, 0, local_rows_cf),
         KOKKOS_LAMBDA(const PetscInt i, PetscInt& tmax_direct, PetscInt& tmax_iter, PetscInt& tcount_iter) {
            PetscInt row_nnz = device_local_i_sparsity[i + 1] - device_local_i_sparsity[i];
            if (mpi)
            {
               row_nnz += device_nonlocal_i_sparsity[i + 1] - device_nonlocal_i_sparsity[i];
            }
            if (iter_enabled && row_nnz > iter_threshold)
            {
               if (row_nnz > tmax_iter) tmax_iter = row_nnz;
               tcount_iter += 1;
            }
            else
            {
               if (row_nnz > tmax_direct) tmax_direct = row_nnz;
            }
         },
         Kokkos::Max<PetscInt>(sparsity_max_nnz_direct),
         Kokkos::Max<PetscInt>(sparsity_max_nnz_iter),
         Kokkos::Sum<PetscInt>(count_iter)
      );
      // Kokkos::Max identity is the minimum representable value; clamp to zero.
      if (sparsity_max_nnz_direct < 0) sparsity_max_nnz_direct = 0;
      if (sparsity_max_nnz_iter   < 0) sparsity_max_nnz_iter   = 0;
   }

   // Nothing to do if no rows
   if (local_rows_cf == 0 || (sparsity_max_nnz_direct == 0 && count_iter == 0))
   {
      if (deallocate_submatrices) delete[] submatrices;
      if (mpi && !reuse_int_reuse_mat) PetscCallVoid(PetscFree(submatrices));
      (void)PetscFree(col_indices_off_proc_array);
      return;
   }

   // ~~~~~~~~~~~~~~
   // TeamPolicy: one team per row, with per-team scratch memory
   // ~~~~~~~~~~~~~~
   using team_policy_t = Kokkos::TeamPolicy<>;
   using member_type = team_policy_t::member_type;

   // Direct kernel scratch is sized to the largest row it actually handles.
   // When iter_enabled is false, every row goes through the direct kernel.
   const PetscInt j_max = sparsity_max_nnz_direct;
   const PetscInt iter_threshold_dev = iter_threshold;

   if (j_max > 0)
   {

   // Level 1 scratch budget: dense_mat + rhs + sol + j_global + j_perm
   // Sized for j_max (worst case); inside the kernel views are created with actual j_size
   const size_t level1_scratch = Scratch2DScalarView::shmem_size(j_max, j_max)
                               + ScratchScalarView::shmem_size(j_max)
                               + ScratchScalarView::shmem_size(j_max)
                               + ScratchIntView::shmem_size(j_max)
                               + ScratchIntView::shmem_size(j_max);

   // Level 0 scratch budget for TeamGesv: it internally allocates n*(n+4) scalars
   // Disabling the level 0 scratch since we are using the nopivoting version of
   // teamgesv as it doesn't require temporary space
   //const size_t level0_scratch = Scratch2DScalarView::shmem_size(j_max, j_max + 4);

   auto policy = team_policy_t(exec, local_rows_cf, Kokkos::AUTO());
   // Disable 0 scratch budget
   //policy.set_scratch_size(0, Kokkos::PerTeam(level0_scratch));
   policy.set_scratch_size(1, Kokkos::PerTeam(level1_scratch));

   // ~~~~~~~~~~~~~~
   // Main kernel: one team per row, build dense system and solve
   // ~~~~~~~~~~~~~~
   Kokkos::parallel_for("SAI_Z_build_and_solve", policy,
      KOKKOS_LAMBDA(const member_type &member) {

      const PetscInt i = member.league_rank();

      // ~~~~~~~~
      // Compute j_size from CSR row pointers (all threads, no data dependency)
      // ~~~~~~~~
      const PetscInt ncols_local_sparsity = device_local_i_sparsity[i + 1] - device_local_i_sparsity[i];
      const PetscInt ncols_nonlocal_sparsity = mpi ?
         (device_nonlocal_i_sparsity[i + 1] - device_nonlocal_i_sparsity[i]) : 0;
      const PetscInt j_size = ncols_local_sparsity + ncols_nonlocal_sparsity;

      if (j_size == 0) return;
      // Large rows are handled by the iterative Jacobi kernel below
      if (iter_enabled && j_size > iter_threshold_dev) return;

      // Allocate per-team scratch views sized to j_size
      Scratch2DScalarView dense_mat(member.team_scratch(1), j_size, j_size);
      ScratchScalarView rhs(member.team_scratch(1), j_size);
      ScratchScalarView sol(member.team_scratch(1), j_size);
      ScratchIntView j_global(member.team_scratch(1), j_size);
      ScratchIntView j_perm(member.team_scratch(1), j_size);

      // Zero dense_mat and rhs (sol is overwritten by TeamGesv)
      Kokkos::parallel_for(Kokkos::TeamThreadRange(member, j_size), [&](const PetscInt k) {
         rhs(k) = 0.0;
      });
      Kokkos::parallel_for(Kokkos::TeamThreadRange(member, j_size * j_size), [&](const PetscInt k) {
         dense_mat.data()[k] = 0.0;
      });
      member.team_barrier();

      // ~~~~~~~~
      // Step A: Fill J indices from sparsity_mat_cf row i, then team sort
      // ~~~~~~~~
      Kokkos::parallel_for(Kokkos::TeamThreadRange(member, ncols_local_sparsity),
         [&](const PetscInt j) {
            PetscInt local_col = device_local_j_sparsity[device_local_i_sparsity[i] + j];
            j_global(j) = local_col + global_row_start_ff;
            j_perm(j) = j;
         });
      if (mpi) {
         Kokkos::parallel_for(Kokkos::TeamThreadRange(member, ncols_nonlocal_sparsity),
            [&](const PetscInt j) {
               PetscInt nonlocal_col = device_nonlocal_j_sparsity[device_nonlocal_i_sparsity[i] + j];
               j_global(ncols_local_sparsity + j) = colmap_sparsity_d(nonlocal_col);
               j_perm(ncols_local_sparsity + j) = ncols_local_sparsity + j;
            });
      }
      member.team_barrier();

      // Team-parallel sort of j_global with permutation j_perm to keep track of original positions
      Kokkos::Experimental::sort_by_key_team(member, j_global, j_perm);

      // ~~~~~~~~
      // Step B: Build RHS from A_cf row i (parallel over columns)
      // ~~~~~~~~
      const PetscInt ncols_local_cf = device_local_i_cf[i + 1] - device_local_i_cf[i];
      Kokkos::parallel_for(Kokkos::TeamThreadRange(member, ncols_local_cf),
         [&](const PetscInt k) {
            PetscInt col_local = device_local_j_cf[device_local_i_cf[i] + k];
            PetscScalar val = device_local_vals_cf(device_local_i_cf[i] + k);
            PetscInt global_col = col_local + global_row_start_ff;
            PetscInt pos = binary_search_sorted(j_global, j_size, global_col);
            if (pos >= 0) rhs(pos) = -val;
         });
      if (mpi) {
         const PetscInt ncols_nonlocal_cf = device_nonlocal_i_cf[i + 1] - device_nonlocal_i_cf[i];
         Kokkos::parallel_for(Kokkos::TeamThreadRange(member, ncols_nonlocal_cf),
            [&](const PetscInt k) {
               PetscInt col_nonlocal = device_nonlocal_j_cf[device_nonlocal_i_cf[i] + k];
               PetscScalar val = device_nonlocal_vals_cf(device_nonlocal_i_cf[i] + k);
               PetscInt global_col = colmap_cf_d(col_nonlocal);
               PetscInt pos = binary_search_sorted(j_global, j_size, global_col);
               if (pos >= 0) rhs(pos) = -val;
            });
      }
      member.team_barrier();

      // ~~~~~~~~
      // Step C: Build dense matrix A_ff(J,J)^T (parallel over J rows)
      // Each thread handles one j, writing to dense_mat(*, j) — no races.
      // On GPU (LayoutLeft), the transpose is cache-friendly: each thread reads
      // row J[j] of A_ff sequentially from CSR and writes down column j of
      // dense_mat, which is contiguous in column-major layout.
      // ~~~~~~~~
      Kokkos::parallel_for(Kokkos::TeamThreadRange(member, j_size),
         [&](const PetscInt j) {
            PetscInt global_row = j_global(j);
            bool is_local = (global_row >= global_row_start_ff &&
                             global_row < global_row_start_ff + local_rows_ff);

            if (is_local) {
               PetscInt local_row = global_row - global_row_start_ff;
               // Local A_ff columns
               PetscInt ncols = device_local_i_ff[local_row + 1] - device_local_i_ff[local_row];
               for (PetscInt k = 0; k < ncols; k++) {
                  PetscInt global_col = device_local_j_ff[device_local_i_ff[local_row] + k]
                                        + global_row_start_ff;
                  PetscScalar val = device_local_vals_ff(device_local_i_ff[local_row] + k);
                  PetscInt pos = binary_search_sorted(j_global, j_size, global_col);
                  if (pos >= 0) dense_mat(pos, j) = val;
               }
               // Nonlocal A_ff columns
               if (mpi) {
                  PetscInt ncols_nl = device_nonlocal_i_ff[local_row + 1]
                                      - device_nonlocal_i_ff[local_row];
                  for (PetscInt k = 0; k < ncols_nl; k++) {
                     PetscInt col_nonlocal = device_nonlocal_j_ff[
                        device_nonlocal_i_ff[local_row] + k];
                     PetscScalar val = device_nonlocal_vals_ff(
                        device_nonlocal_i_ff[local_row] + k);
                     PetscInt global_col = colmap_ff_d(col_nonlocal);
                     PetscInt pos = binary_search_sorted(j_global, j_size, global_col);
                     if (pos >= 0) dense_mat(pos, j) = val;
                  }
               }
            } else {
               // Non-local row: find in submatrix
               PetscInt submat_row = binary_search_sorted(
                  colmap_sparsity_d, cols_ao_sparsity, global_row);
               if (submat_row < 0) return;
               PetscInt ncols_sub = device_submat_i[submat_row + 1]
                                    - device_submat_i[submat_row];
               for (PetscInt k = 0; k < ncols_sub; k++) {
                  PetscInt submat_col = device_submat_j[
                     device_submat_i[submat_row] + k];
                  PetscScalar val = device_submat_vals(
                     device_submat_i[submat_row] + k);
                  PetscInt global_col = col_indices_off_proc_d(submat_col);
                  PetscInt pos = binary_search_sorted(j_global, j_size, global_col);
                  if (pos >= 0) dense_mat(pos, j) = val;
               }
            }
         });
      member.team_barrier();

      // ~~~~~~~~
      // Step D: Solve A_ff(J,J)^T * x = rhs using TeamGesv
      // ~~~~~~~~
      // Deliberately using the nopivoting version here as the pivoting
      // version uses level 0 scratch space and we can have the problem
      // where the j_size grows larger than the available level 0 scratch, causing a failure.
      // If you want to use the pivoting version you need to reenable the set_scratch_size
      // for level 0 outside the loop
      // The submatrices should not require pivoting given Aff is diagonally dominant
      KokkosBatched::TeamGesv<member_type, KokkosBatched::Gesv::NoPivoting>
         ::invoke(member, dense_mat, sol, rhs);
      member.team_barrier();

      // ~~~~~~~~
      // Step E: Write solution to Z (parallel over j_size)
      // j_perm[k] gives the original position in the (local, nonlocal) ordering
      // ~~~~~~~~
      Kokkos::parallel_for(Kokkos::TeamThreadRange(member, j_size),
         [&](const PetscInt k) {
            PetscInt orig_pos = j_perm(k);
            if (orig_pos < ncols_local_sparsity)
               device_local_vals_z(device_local_i_z[i] + orig_pos) = sol(k);
            else if (mpi)
               device_nonlocal_vals_z(device_nonlocal_i_z[i]
                  + (orig_pos - ncols_local_sparsity)) = sol(k);
         });
   });

   } // end if (j_max > 0)

   // ~~~~~~~~~~~~~~
   // Iterative Jacobi kernel for rows with j_size > iter_threshold.
   //
   // Two implementations, selected on the host:
   //  * COMPACT   - build the per-row sparse block C = A_ff(J,J) ONCE into team
   //                scratch as CSR (block_row_ptr / block_col / block_val, row j
   //                holding the entries of A_ff row J[j] whose column is in J) and
   //                reuse it across all Jacobi iterations. The mat-vec
   //                r = -A_ff(J,J)^T sol = -C^T sol is a scatter over the stored
   //                entries. This removes the per-iteration re-read of A_ff and the
   //                O(log j) binary search per nonzero, so it is much faster on
   //                dense rows (e.g. nd3k), at the cost of O(nnz(A_ff(J,J))) extra
   //                team scratch.
   //  * MATRIX-FREE - no block stored; the mat-vec is applied straight from the CSR
   //                rows of A_ff every iteration. O(j) scratch, never OOMs.
   //
   // Kokkos caps level-1 team scratch at an arbitrary 20 MiB, so COMPACT is only
   // used when the whole team scratch (the six length-j vectors plus the block CSR)
   // fits under that cap; otherwise we fall back to MATRIX-FREE. This keeps the
   // huge-j rows (e.g. trans4, j ~ millions) on the safe matrix-free path.
   // ~~~~~~~~~~~~~~
   if (iter_enabled && count_iter > 0)
   {
      const PetscInt j_max_iter = sparsity_max_nnz_iter;

      // Level-1 scratch used by both kernels: rhs, sol, r, diag (scalar) and
      // j_global, j_perm (int), all length j.
      const size_t base_scratch_iter = ScratchScalarView::shmem_size(j_max_iter)
                                      + ScratchScalarView::shmem_size(j_max_iter)
                                      + ScratchScalarView::shmem_size(j_max_iter)
                                      + ScratchScalarView::shmem_size(j_max_iter)
                                      + ScratchIntView::shmem_size(j_max_iter)
                                      + ScratchIntView::shmem_size(j_max_iter);

      // Kokkos hard-caps level-1 team scratch at 20 MiB; keep a small margin.
      const size_t kokkos_l1_cap = 20 * 1024 * 1024;
      const size_t l1_budget     = kokkos_l1_cap - 64 * 1024;

      // COMPACT also needs the block CSR: row pointer (length j+1) plus block_col
      // and block_val (length block_nnz_max). We can only decide it fits once we
      // know block_nnz_max, so first require the fixed part (base + row pointer) to
      // fit; only then run the pre-pass to size the block arrays.
      const size_t compact_fixed = base_scratch_iter
                                 + ScratchIntView::shmem_size(j_max_iter + 1);

      bool use_compact = (compact_fixed < l1_budget);
      PetscInt block_nnz_max = 0;

      if (use_compact)
      {
         // ~~~~~~~~
         // Pre-pass: for each iterative row, count the nnz of the sparse block
         // A_ff(J,J) (the entries the mat-vec touches). The max over rows sizes the
         // COMPACT block arrays. Uses only the two int length-j vectors, which fit
         // because compact_fixed (a superset) already fits.
         // ~~~~~~~~
         const size_t prepass_scratch = ScratchIntView::shmem_size(j_max_iter)
                                      + ScratchIntView::shmem_size(j_max_iter);
         auto policy_pre = team_policy_t(exec, local_rows_cf, Kokkos::AUTO());
         policy_pre.set_scratch_size(1, Kokkos::PerTeam(prepass_scratch));

         Kokkos::parallel_reduce("SAI_Z_block_nnz_prepass", policy_pre,
            KOKKOS_LAMBDA(const member_type &member, PetscInt &tmax) {

            const PetscInt i = member.league_rank();
            const PetscInt ncols_local_sparsity = device_local_i_sparsity[i + 1] - device_local_i_sparsity[i];
            const PetscInt ncols_nonlocal_sparsity = mpi ?
               (device_nonlocal_i_sparsity[i + 1] - device_nonlocal_i_sparsity[i]) : 0;
            const PetscInt j_size = ncols_local_sparsity + ncols_nonlocal_sparsity;
            if (j_size == 0) return;
            if (j_size <= iter_threshold_dev) return;

            ScratchIntView j_global(member.team_scratch(1), j_size);
            ScratchIntView j_perm(member.team_scratch(1), j_size);

            // Build J (same as Step A below), then count block nnz.
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, ncols_local_sparsity),
               [&](const PetscInt j) {
                  j_global(j) = device_local_j_sparsity[device_local_i_sparsity[i] + j] + global_row_start_ff;
                  j_perm(j) = j;
               });
            if (mpi) {
               Kokkos::parallel_for(Kokkos::TeamThreadRange(member, ncols_nonlocal_sparsity),
                  [&](const PetscInt j) {
                     j_global(ncols_local_sparsity + j) =
                        colmap_sparsity_d(device_nonlocal_j_sparsity[device_nonlocal_i_sparsity[i] + j]);
                     j_perm(ncols_local_sparsity + j) = ncols_local_sparsity + j;
                  });
            }
            member.team_barrier();
            Kokkos::Experimental::sort_by_key_team(member, j_global, j_perm);

            // Count nnz of A_ff row J[j] whose column is in J, summed over j.
            PetscInt row_block_nnz = 0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(member, j_size),
               [&](const PetscInt j, PetscInt &acc) {
                  const PetscInt global_row = j_global(j);
                  bool is_local = (global_row >= global_row_start_ff &&
                                   global_row < global_row_start_ff + local_rows_ff);
                  if (is_local) {
                     PetscInt lr = global_row - global_row_start_ff;
                     PetscInt nc = device_local_i_ff[lr + 1] - device_local_i_ff[lr];
                     for (PetscInt k = 0; k < nc; k++) {
                        PetscInt gc = device_local_j_ff[device_local_i_ff[lr] + k] + global_row_start_ff;
                        if (binary_search_sorted(j_global, j_size, gc) >= 0) acc++;
                     }
                     if (mpi) {
                        PetscInt ncnl = device_nonlocal_i_ff[lr + 1] - device_nonlocal_i_ff[lr];
                        for (PetscInt k = 0; k < ncnl; k++) {
                           PetscInt gc = colmap_ff_d(device_nonlocal_j_ff[device_nonlocal_i_ff[lr] + k]);
                           if (binary_search_sorted(j_global, j_size, gc) >= 0) acc++;
                        }
                     }
                  } else {
                     PetscInt sr = binary_search_sorted(colmap_sparsity_d, cols_ao_sparsity, global_row);
                     if (sr >= 0) {
                        PetscInt ncs = device_submat_i[sr + 1] - device_submat_i[sr];
                        for (PetscInt k = 0; k < ncs; k++) {
                           PetscInt gc = col_indices_off_proc_d(device_submat_j[device_submat_i[sr] + k]);
                           if (binary_search_sorted(j_global, j_size, gc) >= 0) acc++;
                        }
                     }
                  }
               }, row_block_nnz);

            Kokkos::single(Kokkos::PerTeam(member), [&]() {
               if (row_block_nnz > tmax) tmax = row_block_nnz;
            });
         }, Kokkos::Max<PetscInt>(block_nnz_max));
         if (block_nnz_max < 0) block_nnz_max = 0;

         // Does base + block CSR fit under the cap?
         const size_t compact_scratch = compact_fixed
                                      + ScratchIntView::shmem_size(block_nnz_max)
                                      + ScratchScalarView::shmem_size(block_nnz_max);
         if (block_nnz_max == 0 || compact_scratch > l1_budget) use_compact = false;
      }

      PetscCallVoid(PetscInfo(*A_ff,
         "SAI_Z iterative Jacobi kernel: %s (j_max_iter=%d, block_nnz_max=%d)\n",
         use_compact ? "COMPACT" : "MATRIX-FREE", (int)j_max_iter, (int)block_nnz_max));

      if (use_compact)
      {
      const PetscInt block_cap = block_nnz_max;
      const size_t level1_scratch_iter = compact_fixed
                                       + ScratchIntView::shmem_size(block_cap)
                                       + ScratchScalarView::shmem_size(block_cap);

      auto policy_iter = team_policy_t(exec, local_rows_cf, Kokkos::AUTO());
      policy_iter.set_scratch_size(1, Kokkos::PerTeam(level1_scratch_iter));

      Kokkos::parallel_for("SAI_Z_build_and_solve_jacobi_compact", policy_iter,
         KOKKOS_LAMBDA(const member_type &member) {

         const PetscInt i = member.league_rank();

         const PetscInt ncols_local_sparsity = device_local_i_sparsity[i + 1] - device_local_i_sparsity[i];
         const PetscInt ncols_nonlocal_sparsity = mpi ?
            (device_nonlocal_i_sparsity[i + 1] - device_nonlocal_i_sparsity[i]) : 0;
         const PetscInt j_size = ncols_local_sparsity + ncols_nonlocal_sparsity;

         if (j_size == 0) return;
         // Small rows are handled by the direct kernel above
         if (j_size <= iter_threshold_dev) return;

         ScratchScalarView rhs(member.team_scratch(1), j_size);
         ScratchScalarView sol(member.team_scratch(1), j_size);
         ScratchScalarView r(member.team_scratch(1), j_size);
         ScratchScalarView diag(member.team_scratch(1), j_size);
         ScratchIntView j_global(member.team_scratch(1), j_size);
         ScratchIntView j_perm(member.team_scratch(1), j_size);
         ScratchIntView block_row_ptr(member.team_scratch(1), j_size + 1);
         ScratchIntView block_col(member.team_scratch(1), block_cap);
         ScratchScalarView block_val(member.team_scratch(1), block_cap);

         // Zero rhs and diag (sol is initialised below as the Jacobi initial guess;
         // r is zeroed each iteration just before the scatter)
         Kokkos::parallel_for(Kokkos::TeamThreadRange(member, j_size), [&](const PetscInt k) {
            rhs(k) = 0.0;
            diag(k) = 0.0;
         });
         member.team_barrier();

         // ~~~~~~~~
         // Step A: Fill J indices from sparsity_mat_cf row i, then team sort
         // ~~~~~~~~
         Kokkos::parallel_for(Kokkos::TeamThreadRange(member, ncols_local_sparsity),
            [&](const PetscInt j) {
               PetscInt local_col = device_local_j_sparsity[device_local_i_sparsity[i] + j];
               j_global(j) = local_col + global_row_start_ff;
               j_perm(j) = j;
            });
         if (mpi) {
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, ncols_nonlocal_sparsity),
               [&](const PetscInt j) {
                  PetscInt nonlocal_col = device_nonlocal_j_sparsity[device_nonlocal_i_sparsity[i] + j];
                  j_global(ncols_local_sparsity + j) = colmap_sparsity_d(nonlocal_col);
                  j_perm(ncols_local_sparsity + j) = ncols_local_sparsity + j;
               });
         }
         member.team_barrier();

         Kokkos::Experimental::sort_by_key_team(member, j_global, j_perm);

         // ~~~~~~~~
         // Step B: Build RHS from A_cf row i (parallel over columns)
         // ~~~~~~~~
         const PetscInt ncols_local_cf = device_local_i_cf[i + 1] - device_local_i_cf[i];
         Kokkos::parallel_for(Kokkos::TeamThreadRange(member, ncols_local_cf),
            [&](const PetscInt k) {
               PetscInt col_local = device_local_j_cf[device_local_i_cf[i] + k];
               PetscScalar val = device_local_vals_cf[device_local_i_cf[i] + k];
               PetscInt global_col = col_local + global_row_start_ff;
               PetscInt pos = binary_search_sorted(j_global, j_size, global_col);
               if (pos >= 0) rhs(pos) = -val;
            });
         if (mpi) {
            const PetscInt ncols_nonlocal_cf = device_nonlocal_i_cf[i + 1] - device_nonlocal_i_cf[i];
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, ncols_nonlocal_cf),
               [&](const PetscInt k) {
                  PetscInt col_nonlocal = device_nonlocal_j_cf[device_nonlocal_i_cf[i] + k];
                  PetscScalar val = device_nonlocal_vals_cf[device_nonlocal_i_cf[i] + k];
                  PetscInt global_col = colmap_cf_d(col_nonlocal);
                  PetscInt pos = binary_search_sorted(j_global, j_size, global_col);
                  if (pos >= 0) rhs(pos) = -val;
               });
         }
         member.team_barrier();

         // ~~~~~~~~
         // Step C: Extract diag(j) = A_ff(J[j], J[j]) matrix-free (parallel over J
         // rows). The diagonal column J[j] equals this row's own global index, so
         // for a local row it is always in the LOCAL A_ff block.
         // ~~~~~~~~
         Kokkos::parallel_for(Kokkos::TeamThreadRange(member, j_size),
            [&](const PetscInt j) {
               const PetscInt global_diag = j_global(j);
               bool is_local = (global_diag >= global_row_start_ff &&
                                global_diag < global_row_start_ff + local_rows_ff);
               if (is_local) {
                  PetscInt local_row = global_diag - global_row_start_ff;
                  PetscInt ncols = device_local_i_ff[local_row + 1] - device_local_i_ff[local_row];
                  for (PetscInt k = 0; k < ncols; k++) {
                     PetscInt global_col = device_local_j_ff[device_local_i_ff[local_row] + k]
                                           + global_row_start_ff;
                     if (global_col == global_diag) {
                        diag(j) = device_local_vals_ff(device_local_i_ff[local_row] + k);
                        break;
                     }
                  }
               } else {
                  PetscInt submat_row = binary_search_sorted(
                     colmap_sparsity_d, cols_ao_sparsity, global_diag);
                  if (submat_row < 0) return;
                  PetscInt ncols_sub = device_submat_i[submat_row + 1]
                                       - device_submat_i[submat_row];
                  for (PetscInt k = 0; k < ncols_sub; k++) {
                     PetscInt global_col = col_indices_off_proc_d(
                        device_submat_j[device_submat_i[submat_row] + k]);
                     if (global_col == global_diag) {
                        diag(j) = device_submat_vals[device_submat_i[submat_row] + k];
                        break;
                     }
                  }
               }
            });
         member.team_barrier();

         // ~~~~~~~~
         // Step C2: Build the compact CSR of C = A_ff(J,J), once. Row j holds the
         // entries of A_ff row J[j] whose column is in J (column index = local
         // position pos in J, value = A_ff(J[j], J[pos])). Each thread j owns a
         // contiguous output range [block_row_ptr(j), block_row_ptr(j+1)), so the
         // fill needs no atomics.  Helper for_block_row walks those entries once;
         // it is reused for the count and the fill so the two stay in lockstep.
         // ~~~~~~~~
         auto for_block_row = [&](const PetscInt j, auto &&on_entry) {
            const PetscInt global_row = j_global(j);
            bool is_local = (global_row >= global_row_start_ff &&
                             global_row < global_row_start_ff + local_rows_ff);
            if (is_local) {
               PetscInt lr = global_row - global_row_start_ff;
               PetscInt nc = device_local_i_ff[lr + 1] - device_local_i_ff[lr];
               for (PetscInt k = 0; k < nc; k++) {
                  PetscInt gc = device_local_j_ff[device_local_i_ff[lr] + k] + global_row_start_ff;
                  PetscInt pos = binary_search_sorted(j_global, j_size, gc);
                  if (pos >= 0) on_entry(pos, device_local_vals_ff(device_local_i_ff[lr] + k));
               }
               if (mpi) {
                  PetscInt ncnl = device_nonlocal_i_ff[lr + 1] - device_nonlocal_i_ff[lr];
                  for (PetscInt k = 0; k < ncnl; k++) {
                     PetscInt gc = colmap_ff_d(device_nonlocal_j_ff[device_nonlocal_i_ff[lr] + k]);
                     PetscInt pos = binary_search_sorted(j_global, j_size, gc);
                     if (pos >= 0) on_entry(pos, device_nonlocal_vals_ff(device_nonlocal_i_ff[lr] + k));
                  }
               }
            } else {
               PetscInt sr = binary_search_sorted(colmap_sparsity_d, cols_ao_sparsity, global_row);
               if (sr >= 0) {
                  PetscInt ncs = device_submat_i[sr + 1] - device_submat_i[sr];
                  for (PetscInt k = 0; k < ncs; k++) {
                     PetscInt gc = col_indices_off_proc_d(device_submat_j[device_submat_i[sr] + k]);
                     PetscInt pos = binary_search_sorted(j_global, j_size, gc);
                     if (pos >= 0) on_entry(pos, device_submat_vals(device_submat_i[sr] + k));
                  }
               }
            }
         };

         // C2a: per-row counts into block_row_ptr(j+1)
         Kokkos::parallel_for(Kokkos::TeamThreadRange(member, j_size), [&](const PetscInt j) {
            PetscInt cnt = 0;
            for_block_row(j, [&](const PetscInt, const PetscScalar) { cnt++; });
            block_row_ptr(j + 1) = cnt;
         });
         member.team_barrier();

         // C2b: exclusive prefix sum -> CSR row pointer (single thread, O(j))
         Kokkos::single(Kokkos::PerTeam(member), [&]() {
            block_row_ptr(0) = 0;
            for (PetscInt j = 0; j < j_size; j++)
               block_row_ptr(j + 1) += block_row_ptr(j);
         });
         member.team_barrier();

         // C2c: fill block_col / block_val (each thread j owns a contiguous range)
         Kokkos::parallel_for(Kokkos::TeamThreadRange(member, j_size), [&](const PetscInt j) {
            PetscInt off = block_row_ptr(j);
            for_block_row(j, [&](const PetscInt pos, const PetscScalar val) {
               block_col(off) = pos;
               block_val(off) = val;
               off++;
            });
         });
         member.team_barrier();

         // ~~~~~~~~
         // Step D-Jacobi: solve C^T sol = rhs by Jacobi iteration. C = A_ff(J,J) is
         // diagonally dominant in the SAI/AIR setting, so Jacobi converges.
         // x_0 = 0 => initial residual r_0 = rhs, ||r_0||^2 = ||rhs||^2
         // ~~~~~~~~
         PetscScalar r0_sq = 0.0;
         Kokkos::parallel_reduce(Kokkos::TeamThreadRange(member, j_size),
            [&](const PetscInt k, PetscScalar &acc) {
               sol(k) = 0.0;
               acc += rhs(k) * rhs(k);
            }, r0_sq);
         member.team_barrier();

         const PetscScalar rtol_sq = 1.0e-6;       // (1e-3)^2
         const PetscScalar abs_floor_sq = 1.0e-100;
         const int max_iter = 100;

         if (r0_sq > abs_floor_sq) {
            const PetscScalar stop_sq = rtol_sq * r0_sq;
            PetscScalar rnorm_sq = 0.0;

            for (int it = 0; it < max_iter; ++it) {
               // r = -C^T sol from the prebuilt CSR: block row j (= A_ff row J[j])
               // holds entries (pos, C(j,pos)=A_ff(J[j],J[pos])), and
               // (C^T sol)(pos) = sum_j C(j,pos) sol(j), so scatter
               // C(j,pos)*sol(j) into r(pos). Several j hit one pos => atomic.
               Kokkos::parallel_for(Kokkos::TeamThreadRange(member, j_size),
                  [&](const PetscInt k) { r(k) = 0.0; });
               member.team_barrier();

               Kokkos::parallel_for(Kokkos::TeamThreadRange(member, j_size),
                  [&](const PetscInt j) {
                     const PetscScalar sj = sol(j);
                     for (PetscInt e = block_row_ptr(j); e < block_row_ptr(j + 1); e++)
                        Kokkos::atomic_add(&r(block_col(e)), -block_val(e) * sj);
                  });
               member.team_barrier();

               // r += rhs ; accumulate ||r||^2
               rnorm_sq = 0.0;
               Kokkos::parallel_reduce(Kokkos::TeamThreadRange(member, j_size),
                  [&](const PetscInt k, PetscScalar &acc) {
                     r(k) += rhs(k);
                     acc  += r(k) * r(k);
                  }, rnorm_sq);
               member.team_barrier();

               if (rnorm_sq < stop_sq) break;

               // Jacobi update: sol += r / diag. diag(A_ff(J,J)^T) == diag(A_ff(J,J)).
               Kokkos::parallel_for(Kokkos::TeamThreadRange(member, j_size),
                  [&](const PetscInt k) {
                     sol(k) += r(k) / diag(k);
                  });
               member.team_barrier();
            }
         }

         // ~~~~~~~~
         // Step E: Write solution to Z (parallel over j_size)
         // ~~~~~~~~
         Kokkos::parallel_for(Kokkos::TeamThreadRange(member, j_size),
            [&](const PetscInt k) {
               PetscInt orig_pos = j_perm(k);
               if (orig_pos < ncols_local_sparsity)
                  device_local_vals_z(device_local_i_z[i] + orig_pos) = sol(k);
               else if (mpi)
                  device_nonlocal_vals_z(device_nonlocal_i_z[i]
                     + (orig_pos - ncols_local_sparsity)) = sol(k);
            });
      });
      }
      else
      {
      // ~~~~~~~~
      // MATRIX-FREE fallback: no block stored, mat-vec applied straight from the
      // CSR rows of A_ff each iteration (O(j) scratch, used when the compact block
      // would exceed the 20 MiB level-1 scratch cap).
      // ~~~~~~~~
      const size_t level1_scratch_iter = base_scratch_iter;

      auto policy_iter = team_policy_t(exec, local_rows_cf, Kokkos::AUTO());
      policy_iter.set_scratch_size(1, Kokkos::PerTeam(level1_scratch_iter));

      Kokkos::parallel_for("SAI_Z_build_and_solve_jacobi", policy_iter,
         KOKKOS_LAMBDA(const member_type &member) {

         const PetscInt i = member.league_rank();

         const PetscInt ncols_local_sparsity = device_local_i_sparsity[i + 1] - device_local_i_sparsity[i];
         const PetscInt ncols_nonlocal_sparsity = mpi ?
            (device_nonlocal_i_sparsity[i + 1] - device_nonlocal_i_sparsity[i]) : 0;
         const PetscInt j_size = ncols_local_sparsity + ncols_nonlocal_sparsity;

         if (j_size == 0) return;
         // Small rows are handled by the direct kernel above
         if (j_size <= iter_threshold_dev) return;

         ScratchScalarView rhs(member.team_scratch(1), j_size);
         ScratchScalarView sol(member.team_scratch(1), j_size);
         ScratchScalarView r(member.team_scratch(1), j_size);
         ScratchScalarView diag(member.team_scratch(1), j_size);
         ScratchIntView j_global(member.team_scratch(1), j_size);
         ScratchIntView j_perm(member.team_scratch(1), j_size);

         // Zero rhs and diag (sol is initialised below as the Jacobi initial guess;
         // r is zeroed each iteration just before the matrix-free scatter)
         Kokkos::parallel_for(Kokkos::TeamThreadRange(member, j_size), [&](const PetscInt k) {
            rhs(k) = 0.0;
            diag(k) = 0.0;
         });
         member.team_barrier();

         // ~~~~~~~~
         // Step A: Fill J indices from sparsity_mat_cf row i, then team sort
         // ~~~~~~~~
         Kokkos::parallel_for(Kokkos::TeamThreadRange(member, ncols_local_sparsity),
            [&](const PetscInt j) {
               PetscInt local_col = device_local_j_sparsity[device_local_i_sparsity[i] + j];
               j_global(j) = local_col + global_row_start_ff;
               j_perm(j) = j;
            });
         if (mpi) {
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, ncols_nonlocal_sparsity),
               [&](const PetscInt j) {
                  PetscInt nonlocal_col = device_nonlocal_j_sparsity[device_nonlocal_i_sparsity[i] + j];
                  j_global(ncols_local_sparsity + j) = colmap_sparsity_d(nonlocal_col);
                  j_perm(ncols_local_sparsity + j) = ncols_local_sparsity + j;
               });
         }
         member.team_barrier();

         Kokkos::Experimental::sort_by_key_team(member, j_global, j_perm);

         // ~~~~~~~~
         // Step B: Build RHS from A_cf row i (parallel over columns)
         // ~~~~~~~~
         const PetscInt ncols_local_cf = device_local_i_cf[i + 1] - device_local_i_cf[i];
         Kokkos::parallel_for(Kokkos::TeamThreadRange(member, ncols_local_cf),
            [&](const PetscInt k) {
               PetscInt col_local = device_local_j_cf[device_local_i_cf[i] + k];
               PetscScalar val = device_local_vals_cf[device_local_i_cf[i] + k];
               PetscInt global_col = col_local + global_row_start_ff;
               PetscInt pos = binary_search_sorted(j_global, j_size, global_col);
               if (pos >= 0) rhs(pos) = -val;
            });
         if (mpi) {
            const PetscInt ncols_nonlocal_cf = device_nonlocal_i_cf[i + 1] - device_nonlocal_i_cf[i];
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, ncols_nonlocal_cf),
               [&](const PetscInt k) {
                  PetscInt col_nonlocal = device_nonlocal_j_cf[device_nonlocal_i_cf[i] + k];
                  PetscScalar val = device_nonlocal_vals_cf[device_nonlocal_i_cf[i] + k];
                  PetscInt global_col = colmap_cf_d(col_nonlocal);
                  PetscInt pos = binary_search_sorted(j_global, j_size, global_col);
                  if (pos >= 0) rhs(pos) = -val;
               });
         }
         member.team_barrier();

         // ~~~~~~~~
         // Step C: Extract diag(j) = A_ff(J[j], J[j]) matrix-free (parallel over
         // J rows). This is the diagonal of the (never materialised) dense block
         // A_ff(J,J)^T, needed for the Jacobi update; the off-diagonals are applied
         // on the fly in Step D. The diagonal column J[j] equals this row's own
         // global index, so for a local row it is always in the LOCAL A_ff block.
         // ~~~~~~~~
         Kokkos::parallel_for(Kokkos::TeamThreadRange(member, j_size),
            [&](const PetscInt j) {
               const PetscInt global_diag = j_global(j);
               bool is_local = (global_diag >= global_row_start_ff &&
                                global_diag < global_row_start_ff + local_rows_ff);
               if (is_local) {
                  PetscInt local_row = global_diag - global_row_start_ff;
                  PetscInt ncols = device_local_i_ff[local_row + 1] - device_local_i_ff[local_row];
                  for (PetscInt k = 0; k < ncols; k++) {
                     PetscInt global_col = device_local_j_ff[device_local_i_ff[local_row] + k]
                                           + global_row_start_ff;
                     if (global_col == global_diag) {
                        diag(j) = device_local_vals_ff(device_local_i_ff[local_row] + k);
                        break;
                     }
                  }
               } else {
                  PetscInt submat_row = binary_search_sorted(
                     colmap_sparsity_d, cols_ao_sparsity, global_diag);
                  if (submat_row < 0) return;
                  PetscInt ncols_sub = device_submat_i[submat_row + 1]
                                       - device_submat_i[submat_row];
                  for (PetscInt k = 0; k < ncols_sub; k++) {
                     PetscInt global_col = col_indices_off_proc_d(
                        device_submat_j[device_submat_i[submat_row] + k]);
                     if (global_col == global_diag) {
                        diag(j) = device_submat_vals[device_submat_i[submat_row] + k];
                        break;
                     }
                  }
               }
            });
         member.team_barrier();

         // ~~~~~~~~
         // Step D-Jacobi: solve mat * sol = rhs by Jacobi iteration.
         // mat is the row-wise transpose of A_ff(J,J), which is diagonally
         // dominant in the SAI/AIR setting, so Jacobi converges.
         // x_0 = 0 => initial residual r_0 = rhs, ||r_0||^2 = ||rhs||^2
         // ~~~~~~~~
         PetscScalar r0_sq = 0.0;
         Kokkos::parallel_reduce(Kokkos::TeamThreadRange(member, j_size),
            [&](const PetscInt k, PetscScalar &acc) {
               sol(k) = 0.0;
               acc += rhs(k) * rhs(k);
            }, r0_sq);
         member.team_barrier();

         const PetscScalar rtol_sq = 1.0e-6;       // (1e-3)^2
         const PetscScalar abs_floor_sq = 1.0e-100;
         const int max_iter = 100;

         if (r0_sq > abs_floor_sq) {
            const PetscScalar stop_sq = rtol_sq * r0_sq;
            PetscScalar rnorm_sq = 0.0;

            for (int it = 0; it < max_iter; ++it) {
               // r = -A_ff(J,J)^T * sol, applied MATRIX-FREE from the CSR rows of
               // A_ff (no dense block). dense_mat(pos,j) would be A_ff(J[j],J[pos]),
               // so (A_ff(J,J)^T sol)(pos) = sum_j A_ff(J[j],J[pos]) sol(j): iterate
               // rows J[j], and scatter A_ff(J[j],J[pos])*sol(j) into r(pos). Several
               // j map to one pos, so the scatter is atomic.
               Kokkos::parallel_for(Kokkos::TeamThreadRange(member, j_size),
                  [&](const PetscInt k) { r(k) = 0.0; });
               member.team_barrier();

               Kokkos::parallel_for(Kokkos::TeamThreadRange(member, j_size),
                  [&](const PetscInt j) {
                     const PetscScalar sj = sol(j);
                     const PetscInt global_row = j_global(j);
                     bool is_local = (global_row >= global_row_start_ff &&
                                      global_row < global_row_start_ff + local_rows_ff);
                     if (is_local) {
                        PetscInt local_row = global_row - global_row_start_ff;
                        PetscInt ncols = device_local_i_ff[local_row + 1] - device_local_i_ff[local_row];
                        for (PetscInt k = 0; k < ncols; k++) {
                           PetscInt global_col = device_local_j_ff[device_local_i_ff[local_row] + k]
                                                 + global_row_start_ff;
                           PetscInt pos = binary_search_sorted(j_global, j_size, global_col);
                           if (pos >= 0)
                              Kokkos::atomic_add(&r(pos),
                                 -device_local_vals_ff(device_local_i_ff[local_row] + k) * sj);
                        }
                        if (mpi) {
                           PetscInt ncols_nl = device_nonlocal_i_ff[local_row + 1]
                                               - device_nonlocal_i_ff[local_row];
                           for (PetscInt k = 0; k < ncols_nl; k++) {
                              PetscInt global_col = colmap_ff_d(
                                 device_nonlocal_j_ff[device_nonlocal_i_ff[local_row] + k]);
                              PetscInt pos = binary_search_sorted(j_global, j_size, global_col);
                              if (pos >= 0)
                                 Kokkos::atomic_add(&r(pos),
                                    -device_nonlocal_vals_ff(device_nonlocal_i_ff[local_row] + k) * sj);
                           }
                        }
                     } else {
                        PetscInt submat_row = binary_search_sorted(
                           colmap_sparsity_d, cols_ao_sparsity, global_row);
                        if (submat_row >= 0) {
                           PetscInt ncols_sub = device_submat_i[submat_row + 1]
                                                - device_submat_i[submat_row];
                           for (PetscInt k = 0; k < ncols_sub; k++) {
                              PetscInt global_col = col_indices_off_proc_d(
                                 device_submat_j[device_submat_i[submat_row] + k]);
                              PetscInt pos = binary_search_sorted(j_global, j_size, global_col);
                              if (pos >= 0)
                                 Kokkos::atomic_add(&r(pos),
                                    -device_submat_vals(device_submat_i[submat_row] + k) * sj);
                           }
                        }
                     }
                  });
               member.team_barrier();

               // r += rhs ; accumulate ||r||^2
               rnorm_sq = 0.0;
               Kokkos::parallel_reduce(Kokkos::TeamThreadRange(member, j_size),
                  [&](const PetscInt k, PetscScalar &acc) {
                     r(k) += rhs(k);
                     acc  += r(k) * r(k);
                  }, rnorm_sq);
               member.team_barrier();

               if (rnorm_sq < stop_sq) break;

               // Jacobi update: sol += r / diag. diag(A_ff(J,J)^T) == diag(A_ff(J,J)).
               Kokkos::parallel_for(Kokkos::TeamThreadRange(member, j_size),
                  [&](const PetscInt k) {
                     sol(k) += r(k) / diag(k);
                  });
               member.team_barrier();
            }
         }

         // ~~~~~~~~
         // Step E: Write solution to Z (parallel over j_size)
         // ~~~~~~~~
         Kokkos::parallel_for(Kokkos::TeamThreadRange(member, j_size),
            [&](const PetscInt k) {
               PetscInt orig_pos = j_perm(k);
               if (orig_pos < ncols_local_sparsity)
                  device_local_vals_z(device_local_i_z[i] + orig_pos) = sol(k);
               else if (mpi)
                  device_nonlocal_vals_z(device_nonlocal_i_z[i]
                     + (orig_pos - ncols_local_sparsity)) = sol(k);
            });
      });
      }
   }

   Kokkos::fence();

   // Restore the read-only input views
   PetscCallVoid(MatSeqAIJRestoreKokkosView(submatrices[0], &device_submat_vals));
   PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local_ff, &device_local_vals_ff));
   if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal_ff, &device_nonlocal_vals_ff));
   PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local_cf, &device_local_vals_cf));
   if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal_cf, &device_nonlocal_vals_cf));

   // The matching restore handles MatSeqAIJKokkosModifyDevice (clears sync state,
   // marks device modified, invalidates transpose/hermitian, bumps object state).
   PetscCallVoid(MatSeqAIJRestoreKokkosViewWrite(mat_local_z, &device_local_vals_z));
   if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosViewWrite(mat_nonlocal_z, &device_nonlocal_vals_z));

   // ~~~~~~~~~~~~~~
   // Cleanup
   // ~~~~~~~~~~~~~~
   if (deallocate_submatrices) delete[] submatrices;
   if (mpi && !reuse_int_reuse_mat) PetscCallVoid(PetscFree(submatrices));
   (void)PetscFree(col_indices_off_proc_array);

   return;
}
