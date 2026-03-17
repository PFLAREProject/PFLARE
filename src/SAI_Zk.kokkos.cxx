// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"

//------------------------------------------------------------------------------------------------------------------------

// Compute lAIR Z matrix with kokkos - keeping everything on the device
// This is the incomplete SAI case where I = J (square system)
// For each row i of Z:
//   1. Get J indices from sparsity_mat_cf row i (sorted global indices)
//   2. Build RHS from A_cf row i intersected with J
//   3. Build dense A_ff(J,J)^T
//   4. Solve A_ff(J,J)^T * z = -A_cf(i,J)^T with SerialGesv
//   5. Write solution to Z row i (using permutation to map sorted→original order)
PETSC_INTERN void calculate_and_build_sai_z_kokkos(Mat *A_ff, Mat *A_cf, Mat *sparsity_mat_cf,
               const int reuse_int_reuse_mat, Mat *reuse_mat, Mat *z_mat)
{
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
      Kokkos::deep_copy(colmap_cf_d, colmap_cf_h);
      PetscCallVoid(PetscLogCpuToGpu(cols_ao_cf * sizeof(PetscInt)));
   }

   auto colmap_ff_d = PetscIntKokkosView("colmap_ff_d", mpi ? cols_ao_ff : 1);
   if (mpi && cols_ao_ff > 0)
   {
      auto colmap_ff_h = PetscIntConstKokkosViewHost(colmap_ff, cols_ao_ff);
      Kokkos::deep_copy(colmap_ff_d, colmap_ff_h);
      PetscCallVoid(PetscLogCpuToGpu(cols_ao_ff * sizeof(PetscInt)));
   }

   // Copy col_indices_off_proc_array to device (for converting submatrix columns to global)
   auto col_indices_off_proc_d = PetscIntKokkosView("col_indices_off_proc_d", size_cols);
   {
      auto col_indices_off_proc_h = PetscIntConstKokkosViewHost(col_indices_off_proc_array, size_cols);
      Kokkos::deep_copy(col_indices_off_proc_d, col_indices_off_proc_h);
      PetscCallVoid(PetscLogCpuToGpu(size_cols * sizeof(PetscInt)));
   }

   // Copy colmap_sparsity to device for non-local row lookup and global index conversion
   auto colmap_sparsity_d = PetscIntKokkosView("colmap_sparsity_d", mpi ? cols_ao_sparsity : 1);
   if (mpi && cols_ao_sparsity > 0)
   {
      auto colmap_sparsity_h = PetscIntConstKokkosViewHost(colmap_sparsity, cols_ao_sparsity);
      Kokkos::deep_copy(colmap_sparsity_d, colmap_sparsity_h);
      PetscCallVoid(PetscLogCpuToGpu(cols_ao_sparsity * sizeof(PetscInt)));
   }

   // ~~~~~~~~~~~~~~
   // Get device CSR pointers for all matrices
   // ~~~~~~~~~~~~~~
   PetscMemType mtype;

   // Submatrix (non-local rows of A_ff)
   const PetscInt *device_submat_i = nullptr, *device_submat_j = nullptr;
   PetscScalar *device_submat_vals = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(submatrices[0], &device_submat_i, &device_submat_j, &device_submat_vals, &mtype));

   // A_ff local + nonlocal
   const PetscInt *device_local_i_ff = nullptr, *device_local_j_ff = nullptr;
   const PetscInt *device_nonlocal_i_ff = nullptr, *device_nonlocal_j_ff = nullptr;
   PetscScalar *device_local_vals_ff = nullptr, *device_nonlocal_vals_ff = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_ff, &device_local_i_ff, &device_local_j_ff, &device_local_vals_ff, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_ff, &device_nonlocal_i_ff, &device_nonlocal_j_ff, &device_nonlocal_vals_ff, &mtype));

   // A_cf local + nonlocal
   const PetscInt *device_local_i_cf = nullptr, *device_local_j_cf = nullptr;
   const PetscInt *device_nonlocal_i_cf = nullptr, *device_nonlocal_j_cf = nullptr;
   PetscScalar *device_local_vals_cf = nullptr, *device_nonlocal_vals_cf = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_cf, &device_local_i_cf, &device_local_j_cf, &device_local_vals_cf, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_cf, &device_nonlocal_i_cf, &device_nonlocal_j_cf, &device_nonlocal_vals_cf, &mtype));

   // Sparsity matrix local + nonlocal
   const PetscInt *device_local_i_sparsity = nullptr, *device_local_j_sparsity = nullptr;
   const PetscInt *device_nonlocal_i_sparsity = nullptr, *device_nonlocal_j_sparsity = nullptr;
   PetscScalar *device_local_vals_sparsity = nullptr, *device_nonlocal_vals_sparsity = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_sparsity, &device_local_i_sparsity, &device_local_j_sparsity, &device_local_vals_sparsity, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_sparsity, &device_nonlocal_i_sparsity, &device_nonlocal_j_sparsity, &device_nonlocal_vals_sparsity, &mtype));

   // Z output local + nonlocal
   const PetscInt *device_local_i_z = nullptr, *device_local_j_z = nullptr;
   const PetscInt *device_nonlocal_i_z = nullptr, *device_nonlocal_j_z = nullptr;
   PetscScalar *device_local_vals_z = nullptr, *device_nonlocal_vals_z = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_z, &device_local_i_z, &device_local_j_z, &device_local_vals_z, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_z, &device_nonlocal_i_z, &device_nonlocal_j_z, &device_nonlocal_vals_z, &mtype));

   // ~~~~~~~~~~~~~~
   // Find maximum j_size (max nnz per row in sparsity_mat_cf)
   // ~~~~~~~~~~~~~~
   PetscInt sparsity_max_nnz = 0, sparsity_max_nnz_local = 0, sparsity_max_nnz_nonlocal = 0;
   if (local_rows_cf > 0)
   {
      Kokkos::parallel_reduce("FindMaxNNZSparsity", local_rows_cf,
         KOKKOS_LAMBDA(const PetscInt i, PetscInt& thread_max) {
            PetscInt row_nnz = device_local_i_sparsity[i + 1] - device_local_i_sparsity[i];
            thread_max = (row_nnz > thread_max) ? row_nnz : thread_max;
         },
         Kokkos::Max<PetscInt>(sparsity_max_nnz_local)
      );
      if (mpi)
      {
         Kokkos::parallel_reduce("FindMaxNNZSparsityNonLocal", local_rows_cf,
            KOKKOS_LAMBDA(const PetscInt i, PetscInt& thread_max) {
               PetscInt row_nnz = device_nonlocal_i_sparsity[i + 1] - device_nonlocal_i_sparsity[i];
               thread_max = (row_nnz > thread_max) ? row_nnz : thread_max;
            },
            Kokkos::Max<PetscInt>(sparsity_max_nnz_nonlocal)
         );
      }
      sparsity_max_nnz = sparsity_max_nnz_local + sparsity_max_nnz_nonlocal;
   }

   // Nothing to do if no rows
   if (local_rows_cf == 0 || sparsity_max_nnz == 0)
   {
      if (deallocate_submatrices) delete[] submatrices;
      if (mpi && !reuse_int_reuse_mat) PetscCallVoid(PetscFree(submatrices));
      (void)PetscFree(col_indices_off_proc_array);
      return;
   }

   auto exec = PetscGetKokkosExecutionSpace();

   // ~~~~~~~~~~~~~~
   // TeamPolicy: one team per row, with per-team scratch memory
   // ~~~~~~~~~~~~~~
   using team_policy_t = Kokkos::TeamPolicy<>;
   using member_type = team_policy_t::member_type;

   const PetscInt j_max = sparsity_max_nnz;

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
      // Step C: Build dense matrix A_ff(J,J)^T (parallel over J rows)
      // Each thread handles one j, writing to dense_mat(*, j) — no races
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
                  PetscScalar val = device_local_vals_ff[device_local_i_ff[local_row] + k];
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
                     PetscScalar val = device_nonlocal_vals_ff[
                        device_nonlocal_i_ff[local_row] + k];
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
                  PetscScalar val = device_submat_vals[
                     device_submat_i[submat_row] + k];
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
               device_local_vals_z[device_local_i_z[i] + orig_pos] = sol(k);
            else if (mpi)
               device_nonlocal_vals_z[device_nonlocal_i_z[i]
                  + (orig_pos - ncols_local_sparsity)] = sol(k);
         });
   });

   // ~~~~~~~~~~~~~~
   // Mark Z's device data as modified
   // ~~~~~~~~~~~~~~
   Mat_SeqAIJKokkos *aijkok_local_z = static_cast<Mat_SeqAIJKokkos *>(mat_local_z->spptr);
   Mat_SeqAIJKokkos *aijkok_nonlocal_z = NULL;
   if (mpi) aijkok_nonlocal_z = static_cast<Mat_SeqAIJKokkos *>(mat_nonlocal_z->spptr);

   exec.fence();

   // Have to specify we've modified data on the device
   aijkok_local_z->a_dual.clear_sync_state();
   aijkok_local_z->a_dual.modify_device();
   aijkok_local_z->transpose_updated = PETSC_FALSE;
   aijkok_local_z->hermitian_updated = PETSC_FALSE;
   if (mpi)
   {
      aijkok_nonlocal_z->a_dual.clear_sync_state();
      aijkok_nonlocal_z->a_dual.modify_device();
      aijkok_nonlocal_z->transpose_updated = PETSC_FALSE;
      aijkok_nonlocal_z->hermitian_updated = PETSC_FALSE;
   }

   // ~~~~~~~~~~~~~~
   // Cleanup
   // ~~~~~~~~~~~~~~
   if (deallocate_submatrices) delete[] submatrices;
   if (mpi && !reuse_int_reuse_mat) PetscCallVoid(PetscFree(submatrices));
   (void)PetscFree(col_indices_off_proc_array);

   return;
}
