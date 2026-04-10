// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

//------------------------------------------------------------------------------------------------------------------------

static PetscErrorCode check_exact_petscint_to_scalar_encoding(PetscInt max_encoded_value, MPI_Comm comm)
{
   PetscFunctionBegin;
   //PflareKokkosTrace _trace("check_exact_petscint_to_scalar_encoding");
   if (max_encoded_value <= 0) PetscFunctionReturn(PETSC_SUCCESS);

   const int digits = std::numeric_limits<PetscScalar>::digits;
   const long double max_exact_ld = std::ldexp(1.0L, digits);

   PetscCheck((long double)max_encoded_value <= max_exact_ld, comm, PETSC_ERR_ARG_OUTOFRANGE,
              "MatCreateSubMatrix_kokkos_view encodes indices via PetscScalar, but max index %" PetscInt_FMT " exceeds exact integer range 2^%" PetscInt_FMT " = %.0Lf",
              max_encoded_value, (PetscInt)digits, max_exact_ld);

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

// Sync the kokkos parts of the matrix to make sure they're up to date
PETSC_INTERN void mat_sync(Mat *X)
{
   //PflareKokkosTrace _trace("mat_sync");
   MatType mat_type;
   PetscCallVoid(MatGetType(*X, &mat_type));
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;   
   Mat mat_local_x = NULL, mat_nonlocal_x = NULL;

   const PetscInt *colmap_x;
   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*X, &mat_local_x, &mat_nonlocal_x, &colmap_x));
   }
   else
   {
      mat_local_x = *X;
   }

    Mat_SeqAIJKokkos *mat_local_xkok = static_cast<Mat_SeqAIJKokkos *>(mat_local_x->spptr);
    if (mat_local_xkok->a_dual.need_sync_device()) {
      mat_local_xkok->a_dual.sync_device();
      mat_local_xkok->transpose_updated = PETSC_FALSE; /* values of the transpose is out-of-date */
      mat_local_xkok->hermitian_updated = PETSC_FALSE;
    }    
    if (mpi) 
    {       
      Mat_SeqAIJKokkos *mat_nonlocal_xkok = static_cast<Mat_SeqAIJKokkos *>(mat_nonlocal_x->spptr);
      if (mat_nonlocal_xkok->a_dual.need_sync_device()) {
         mat_nonlocal_xkok->a_dual.sync_device();
         mat_nonlocal_xkok->transpose_updated = PETSC_FALSE; /* values of the transpose is out-of-date */
         mat_nonlocal_xkok->hermitian_updated = PETSC_FALSE;
      }  
   }
}

//------------------------------------------------------------------------------------------------------------------------

// Remap each entry in j_d from a global index to its local index via binary search into garray_d.
// garray_d must be a sorted array of unique global indices.
// Fences internally.
static void remap_j_to_local_device(PetscIntKokkosView j_d, PetscIntKokkosView garray_d, PetscInt col_ao_output)
{
   //PflareKokkosTrace _trace("remap_j_to_local_device");
   auto exec = PetscGetKokkosExecutionSpace();

   if (j_d.extent(0) == 0) return;
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, j_d.extent(0)), KOKKOS_LAMBDA(const PetscInt i) {
         j_d(i) = binary_search_sorted(garray_d, col_ao_output, j_d(i));
   });
   Kokkos::fence();
}

//------------------------------------------------------------------------------------------------------------------------

// Build garray on device from global indices in j_nonlocal_d and remap j_nonlocal_d to local in-place.
// garray_d (out) is a device view of the sorted unique global column indices (size col_ao_output).
static void rewrite_j_global_to_local_device(PetscInt colmap_max_size, PetscInt &col_ao_output, PetscIntKokkosView j_nonlocal_d, PetscIntKokkosView &garray_d)
{
   //PflareKokkosTrace _trace("rewrite_j_global_to_local_device");
   auto exec = PetscGetKokkosExecutionSpace();

   // Need to preallocate to the max size
   PetscIntKokkosView colmap_output_d("colmap_output_d", colmap_max_size);
   col_ao_output = 0;

   if (j_nonlocal_d.extent(0) > 0)
   {
      ptrdiff_t count_ptr_arith = -1;
      // Scoped so we don't keep the sorted copy of j around very long
      {
         PetscIntKokkosView j_nonlocal_d_sorted("j_nonlocal_d_sorted", j_nonlocal_d.extent(0));
         Kokkos::deep_copy(exec, j_nonlocal_d_sorted, j_nonlocal_d);
         Kokkos::sort(exec, j_nonlocal_d_sorted);
         Kokkos::fence();

         // Unique copy returns a copy of sorted j_nonlocal_d_sorted in order, but with all the duplicate entries removed
         auto unique_end_it = Kokkos::Experimental::unique_copy(exec, j_nonlocal_d_sorted, colmap_output_d);
         Kokkos::fence();
         auto begin_it = Kokkos::Experimental::begin(colmap_output_d);
         count_ptr_arith = unique_end_it - begin_it;
      }
      col_ao_output = static_cast<PetscInt>(count_ptr_arith);

      PetscInt zero = 0;
      garray_d = Kokkos::subview(colmap_output_d, Kokkos::make_pair(zero, col_ao_output));

      // Remap j_nonlocal_d to local indices using binary search into garray_d
      // This fences internally
      remap_j_to_local_device(j_nonlocal_d, garray_d, col_ao_output);
   }
}

//------------------------------------------------------------------------------------------------------------------------

// Generate the colmap and rewrite input global j indices to local given the calculated colmap
PETSC_INTERN void rewrite_j_global_to_local(PetscInt colmap_max_size, PetscInt &col_ao_output, PetscIntKokkosView j_nonlocal_d, PetscInt **garray_host)
{
   //PflareKokkosTrace _trace("rewrite_j_global_to_local");
   auto exec = PetscGetKokkosExecutionSpace();
   PetscIntKokkosView garray_d;

   // This fences internally
   rewrite_j_global_to_local_device(colmap_max_size, col_ao_output, j_nonlocal_d, garray_d);

   // Always allocate host array (even zero-size)
   PetscCallVoid(PetscMalloc1(col_ao_output, garray_host));
   if (col_ao_output > 0)
   {
      PetscIntKokkosViewHost colmap_output_h = PetscIntKokkosViewHost(*garray_host, col_ao_output);
      // Device to host so don't need to specify exec space
      Kokkos::deep_copy(exec, colmap_output_h, garray_d);
      Kokkos::fence();
      // Log copy with petsc
      size_t bytes = col_ao_output * sizeof(PetscInt);
      PetscCallVoid(PetscLogGpuToCpu(bytes));
   }
}

//------------------------------------------------------------------------------------------------------------------------

// Drop according to a tolerance but with kokkos - keeping everything on the device
PETSC_INTERN void remove_small_from_sparse_kokkos(Mat *input_mat, const PetscReal tol, Mat *output_mat, \
                  const int relative_max_row_tolerance_int, const int lump_int, \
                  const int allow_drop_diagonal_int, const int allow_diag_strength_int)
{
   //PflareKokkosTrace _trace("remove_small_from_sparse_kokkos");
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start_temp, global_row_end_plus_one_temp;
   PetscInt global_col_start_temp, global_col_end_plus_one_temp;
   PetscInt rows_ao, cols_ao;
   MatType mat_type;
   PetscInt nnzs_match_local, nnzs_match_nonlocal;
   Mat output_mat_local, output_mat_nonlocal;

   // Equivalent to calling MatSeqAIJKokkosSyncDevice which is petsc intern
   mat_sync(input_mat);   

   PetscCallVoid(MatGetType(*input_mat, &mat_type));
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;
   Mat mat_local = NULL, mat_nonlocal = NULL;
   auto exec = PetscGetKokkosExecutionSpace();

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
   PetscCallVoid(MatGetOwnershipRange(*input_mat, &global_row_start_temp, &global_row_end_plus_one_temp));
   PetscCallVoid(MatGetOwnershipRangeColumn(*input_mat, &global_col_start_temp, &global_col_end_plus_one_temp));
   const PetscInt global_row_start = global_row_start_temp;
   //const PetscInt global_row_end_plus_one = global_row_end_plus_one_temp;
   const PetscInt global_col_start = global_col_start_temp;
   const PetscInt global_col_end_plus_one = global_col_end_plus_one_temp;

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   Kokkos::fence();
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
   // We need the relative row tolerance, let's create some device memory to store it
   PetscScalarKokkosView rel_row_tol_d("rel_row_tol_d", local_rows);
   // Log copy with petsc
   size_t bytes = sizeof(PetscReal);
   PetscCallVoid(PetscLogCpuToGpu(bytes));

   // We need to know how many entries are in each row after our dropping
   PetscIntKokkosView nnz_match_local_row_d("nnz_match_local_row_d", local_rows);
   PetscIntKokkosView nnz_match_nonlocal_row_d;
   if (mpi) nnz_match_nonlocal_row_d = PetscIntKokkosView("nnz_match_nonlocal_row_d", local_rows);
   // Device memory for whether there is an existing diagonal in each row
   boolKokkosView existing_diag_d("existing_diag_d", local_rows);
   Kokkos::deep_copy(exec, existing_diag_d, false);
   const bool not_include_diag = relative_max_row_tolerance_int == -1;
   
   // Compute the relative row tolerances if needed
   if (relative_max_row_tolerance_int) 
   {       
      // Reduction over all rows
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            const PetscInt i = t.league_rank();
            const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];
            const PetscInt row_index_global = i + global_row_start;

            // If we're measuring relative to the diagonal strength, find |a_ii| first.
            if (allow_diag_strength_int) {
               PetscScalar diag_val_abs = -1.0;

               Kokkos::parallel_reduce(
                  Kokkos::TeamVectorRange(t, ncols_local),
                  [&](const PetscInt j, PetscScalar &thread_diag_abs) {
                     const bool is_diagonal = (device_local_j[device_local_i[i] + j] + global_col_start == row_index_global);
                     if (is_diagonal) {
                        const PetscScalar val = Kokkos::abs(device_local_vals[device_local_i[i] + j]);
                        if (val > thread_diag_abs) thread_diag_abs = val;
                     }
                  },
                  Kokkos::Max<PetscScalar>(diag_val_abs)
               );

               if (mpi) {
                  const PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
                  PetscScalar diag_val_abs_nonlocal = -1.0;

                  // Diagonal can live in the off-diagonal block for some layouts.
                  Kokkos::parallel_reduce(
                     Kokkos::TeamVectorRange(t, ncols_nonlocal),
                     [&](const PetscInt j, PetscScalar &thread_diag_abs) {
                        const bool is_diagonal = (colmap_input_d(device_nonlocal_j[device_nonlocal_i[i] + j]) == row_index_global);
                        if (is_diagonal) {
                           const PetscScalar val = Kokkos::abs(device_nonlocal_vals[device_nonlocal_i[i] + j]);
                           if (val > thread_diag_abs) thread_diag_abs = val;
                        }
                     },
                     Kokkos::Max<PetscScalar>(diag_val_abs_nonlocal)
                  );

                  if (diag_val_abs_nonlocal > diag_val_abs) diag_val_abs = diag_val_abs_nonlocal;
               }

               Kokkos::single(Kokkos::PerTeam(t), [&]() {
                  // If the diagonal is missing, avoid over-dropping by using zero threshold.
                  rel_row_tol_d(i) = (diag_val_abs >= 0.0) ? tol * diag_val_abs : 0.0;
               });
            } else {
               PetscScalar max_val = -1.0;

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
            }
      });
   }
   // If we're using a constant tolerance, we can just copy it in
   else
   {
      // Copy in the tolerance
      Kokkos::deep_copy(exec, rel_row_tol_d, tol);
   }

   // ~~~~~~~~~~~~
   // Need to count the number of nnzs we end up with, on each row and in total
   // ~~~~~~~~~~~~
   // Reduce over all the rows
   Kokkos::parallel_reduce(
      Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
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
   Kokkos::parallel_scan (Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
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
         Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
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
      Kokkos::parallel_scan (Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
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

      Kokkos::parallel_reduce("FindMaxNNZ", Kokkos::RangePolicy<>(exec, 0, local_rows),
         KOKKOS_LAMBDA(const PetscInt i, PetscInt& thread_max) {
            PetscInt row_nnz = device_local_i[i + 1] - device_local_i[i];
            thread_max = (row_nnz > thread_max) ? row_nnz : thread_max;
         },
         Kokkos::Max<PetscInt>(max_nnz_local)
      );
      if (mpi)
      {
         Kokkos::parallel_reduce("FindMaxNNZNonLocal", Kokkos::RangePolicy<>(exec, 0, local_rows),
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
   Kokkos::deep_copy(exec, Kokkos::subview(i_local_d, 0), 0);

   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
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
      Kokkos::deep_copy(exec, Kokkos::subview(i_nonlocal_d, 0), 0);                 

      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
         i_nonlocal_d(i + 1) = nnz_match_nonlocal_row_d(i);
      });      
   }           
   
   // Create a team policy with scratch memory allocation
   // We want scratch space for each row
   // We will ncols+1 of integers which tell us what the matching indices we have
   // for both local and nonlocal indices
   const size_t per_view_local = ScratchIntView::shmem_size(max_nnz_local+1);
   const size_t per_view_nonlocal = ScratchIntView::shmem_size(max_nnz_nonlocal+1);

   Kokkos::TeamPolicy<> policy(exec, local_rows, Kokkos::AUTO());
   // We're gonna use the level 1 scratch as our column data is probably bigger than the level 0
   policy.set_scratch_size(1, Kokkos::PerTeam(per_view_local + per_view_nonlocal));

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
   Kokkos::fence();

   // Convert j_nonlocal_d from global to local indices now, before any sort below.
   // All global indices (including any diagonals added in the loop above) are finalised.
   // garray_d holds the sorted unique global column indices on device.
   PetscIntKokkosView garray_d;
   PetscInt col_ao_output = 0;
   if (mpi) {
      // This fences internally
      rewrite_j_global_to_local_device(cols_ao, col_ao_output, j_nonlocal_d, garray_d);
   }

   // Now we may have to sort the column indices
   if (lump_int)
   {
      // Reduce to see if we ever added a diagonal
      bool added_any_diagonal = false;
      Kokkos::parallel_reduce(
         Kokkos::RangePolicy<>(exec, 0, local_rows),
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
         Kokkos::fence();
         KokkosSparse::sort_crs_matrix(csrmat_local);

         if (mpi)
         {
            // j_nonlocal_d now contains local indices; use col_ao_output as numCols
            KokkosCsrMatrix csrmat_nonlocal = KokkosCsrMatrix("csrmat_nonlocal", local_rows, col_ao_output, a_nonlocal_d.extent(0), a_nonlocal_d, i_nonlocal_d, j_nonlocal_d);
            Kokkos::fence();
            KokkosSparse::sort_crs_matrix(csrmat_nonlocal);
         }
      }
   }

   Kokkos::fence();

   // We can create our local diagonal block matrix directly on the device
   PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, local_cols, i_local_d, j_local_d, a_local_d, &output_mat_local));

   // we also have to go and build the a, i, j for the non-local off-diagonal block
   if (mpi)
   {
      // Copy device garray to host
      PetscInt *garray_host = NULL;
      PetscCallVoid(PetscMalloc1(col_ao_output, &garray_host));
      if (col_ao_output > 0)
      {
         PetscIntKokkosViewHost garray_h(garray_host, col_ao_output);
         // Device to host so don't need to specify exec space
         Kokkos::deep_copy(exec, garray_h, garray_d);
         Kokkos::fence();
         size_t bytes = col_ao_output * sizeof(PetscInt);
         PetscCallVoid(PetscLogGpuToCpu(bytes));
      }

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

// Drop according to a existing sparsity in output_mat but with kokkos - keeping everything on the device
PETSC_INTERN void remove_from_sparse_match_kokkos(Mat *input_mat, Mat *output_mat, const int lump_int, const int alpha_int, const PetscReal alpha)
{
   //PflareKokkosTrace _trace("remove_from_sparse_match_kokkos");
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start_temp, global_row_end_plus_one_temp;
   PetscInt global_col_start_temp, global_col_end_plus_one_temp;
   PetscInt rows_ao_input, cols_ao_input, rows_ao_output, cols_ao_output;
   MatType mat_type;

   // Equivalent to calling MatSeqAIJKokkosSyncDevice which is petsc intern
   mat_sync(input_mat);   

   PetscCallVoid(MatGetType(*input_mat, &mat_type));
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;
   auto exec = PetscGetKokkosExecutionSpace();

   // Get the comm
   PetscCallVoid(PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX));
   PetscCallVoid(MatGetLocalSize(*input_mat, &local_rows, &local_cols));
   PetscCallVoid(MatGetSize(*input_mat, &global_rows, &global_cols));
   // This returns the global index of the local portion of the matrix
   PetscCallVoid(MatGetOwnershipRange(*input_mat, &global_row_start_temp, &global_row_end_plus_one_temp));
   PetscCallVoid(MatGetOwnershipRangeColumn(*input_mat, &global_col_start_temp, &global_col_end_plus_one_temp));
   const PetscInt global_row_start = global_row_start_temp;
   //const PetscInt global_row_end_plus_one = global_row_end_plus_one_temp;
   const PetscInt global_col_start = global_col_start_temp;
   const PetscInt global_col_end_plus_one = global_col_end_plus_one_temp;

   Mat mat_local = NULL, mat_nonlocal = NULL;
   Mat mat_local_output = NULL, mat_nonlocal_output = NULL;

   PetscIntConstKokkosViewHost colmap_input_h, colmap_output_h;
   PetscIntKokkosView colmap_input_d, colmap_output_d;
   const PetscInt *colmap_input, *colmap_output;
   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*input_mat, &mat_local, &mat_nonlocal, &colmap_input));
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao_input, &cols_ao_input));

      // We also copy the input mat colmap over to the device as we need it
      colmap_input_h = PetscIntConstKokkosViewHost(colmap_input, cols_ao_input);
      colmap_input_d = PetscIntKokkosView("colmap_input_d", cols_ao_input);
      Kokkos::deep_copy(exec, colmap_input_d, colmap_input_h);

      // Log copy with petsc
      size_t bytes = colmap_input_h.extent(0) * sizeof(PetscInt);
      PetscCallVoid(PetscLogCpuToGpu(bytes));

      // Same for output
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*output_mat, &mat_local_output, &mat_nonlocal_output, &colmap_output));
      PetscCallVoid(MatGetSize(mat_nonlocal_output, &rows_ao_output, &cols_ao_output));

      colmap_output_h = PetscIntConstKokkosViewHost(colmap_output, cols_ao_output);
      colmap_output_d = PetscIntKokkosView("colmap_output_d", cols_ao_output);
      Kokkos::deep_copy(exec, colmap_output_d, colmap_output_h);

      bytes = colmap_output_h.extent(0) * sizeof(PetscInt);
      PetscCallVoid(PetscLogCpuToGpu(bytes));

   }
   else
   {
      mat_local = *input_mat;
      mat_local_output = *output_mat;
   }

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   Kokkos::fence();
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr, *device_nonlocal_vals = nullptr;  
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype));  
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype));          

   // Get the output pointers
   const PetscInt *device_local_i_output = nullptr, *device_local_j_output = nullptr, *device_nonlocal_i_output = nullptr, *device_nonlocal_j_output = nullptr;
   PetscScalar *device_local_vals_output = nullptr, *device_nonlocal_vals_output = nullptr;  
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_output, &device_local_i_output, &device_local_j_output, &device_local_vals_output, &mtype));  
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_output, &device_nonlocal_i_output, &device_nonlocal_j_output, &device_nonlocal_vals_output, &mtype)); 

   Mat_SeqAIJKokkos *aijkok_local_output = static_cast<Mat_SeqAIJKokkos *>(mat_local_output->spptr);
   Mat_SeqAIJKokkos *aijkok_nonlocal_output = NULL;
   if (mpi) aijkok_nonlocal_output = static_cast<Mat_SeqAIJKokkos *>(mat_nonlocal_output->spptr);

   // Find maximum non-zeros per row of the input mat for sizing scratch memory
   PetscInt max_nnz_local = 0, max_nnz_nonlocal = 0;
   if (local_rows > 0) {

      Kokkos::parallel_reduce("FindMaxNNZ", Kokkos::RangePolicy<>(exec, 0, local_rows),
         KOKKOS_LAMBDA(const PetscInt i, PetscInt& thread_max) {
            PetscInt row_nnz = device_local_i[i + 1] - device_local_i[i];
            thread_max = (row_nnz > thread_max) ? row_nnz : thread_max;
         },
         Kokkos::Max<PetscInt>(max_nnz_local)
      );
      if (mpi)
      {
         Kokkos::parallel_reduce("FindMaxNNZNonLocal", Kokkos::RangePolicy<>(exec, 0, local_rows),
            KOKKOS_LAMBDA(const PetscInt i, PetscInt& thread_max) {
               PetscInt row_nnz = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
               thread_max = (row_nnz > thread_max) ? row_nnz : thread_max;
            },
            Kokkos::Max<PetscInt>(max_nnz_nonlocal)
         );         
      }
   }

   // Create a team policy with scratch memory allocation
   // We want scratch space for each row
   // We will have ncols of integers which tell us what the matching indices we have
   // for both local and nonlocal indices
   const size_t per_view_local = ScratchIntView::shmem_size(max_nnz_local);
   const size_t per_view_nonlocal = ScratchIntView::shmem_size(max_nnz_nonlocal);

   Kokkos::TeamPolicy<> policy(exec, local_rows, Kokkos::AUTO());
   // We're gonna use the level 1 scratch as our column data is probably bigger than the level 0
   policy.set_scratch_size(1, Kokkos::PerTeam(per_view_local + per_view_nonlocal));

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
   Kokkos::fence();

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

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Set all the values of the matrix to val
PETSC_INTERN void MatSetAllValues_kokkos(Mat *input_mat, PetscReal val)
{
   //PflareKokkosTrace _trace("MatSetAllValues_kokkos");
   MatType mat_type;

   PetscCallVoid(MatGetType(*input_mat, &mat_type));
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;
   Mat mat_local = NULL, mat_nonlocal = NULL;
  
   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*input_mat, &mat_local, &mat_nonlocal, NULL));
   }
   else
   {
      mat_local = *input_mat;
   }
   PetscInt local_rows, local_cols;
   PetscCallVoid(MatGetLocalSize(*input_mat, &local_rows, &local_cols));

   Mat_SeqAIJKokkos *aijkok_nonlocal = NULL;
   Mat_SeqAIJKokkos *aijkok_local = static_cast<Mat_SeqAIJKokkos *>(mat_local->spptr);
   if(mpi) aijkok_nonlocal = static_cast<Mat_SeqAIJKokkos *>(mat_nonlocal->spptr);
   auto exec = PetscGetKokkosExecutionSpace();

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   Kokkos::fence();
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr, *device_nonlocal_vals = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype));

   PetscScalarKokkosView a_local_d, a_nonlocal_d;
   a_local_d = PetscScalarKokkosView(device_local_vals, aijkok_local->csrmat.nnz());
   if (mpi) a_nonlocal_d = PetscScalarKokkosView(device_nonlocal_vals, aijkok_nonlocal->csrmat.nnz());
   // Copy in the val
   Kokkos::deep_copy(exec, a_local_d, val);
   // Log copy with petsc
   size_t bytes = sizeof(PetscReal);
   PetscCallVoid(PetscLogCpuToGpu(bytes));
   if (mpi)
   {
      Kokkos::deep_copy(exec, a_nonlocal_d, val); 
      PetscCallVoid(PetscLogCpuToGpu(bytes));   
   }
   Kokkos::fence();

   // Have to specify we've modifed data on the device
   // Want to call MatSeqAIJKokkosModifyDevice but its PETSC_INTERN

   aijkok_local->a_dual.clear_sync_state();
   aijkok_local->a_dual.modify_device();
   aijkok_local->transpose_updated = PETSC_FALSE;
   aijkok_local->hermitian_updated = PETSC_FALSE;
   // Invalidate diagonals

   if (mpi)
   {
      aijkok_nonlocal->a_dual.clear_sync_state();
      aijkok_nonlocal->a_dual.modify_device();
      aijkok_nonlocal->transpose_updated = PETSC_FALSE;
      aijkok_nonlocal->hermitian_updated = PETSC_FALSE;
   }
   PetscCallVoid(PetscObjectStateIncrease((PetscObject)(*input_mat)));

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Duplicate and copy a matrix ensuring it always has a diagonal but with kokkos - keeping everything on the device
PETSC_INTERN void mat_duplicate_copy_plus_diag_kokkos(Mat *input_mat, const int reuse_int, Mat *output_mat)
{
   //PflareKokkosTrace _trace("mat_duplicate_copy_plus_diag_kokkos");
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt global_row_start_temp, global_row_end_plus_one_temp;
   PetscInt global_col_start_temp, global_col_end_plus_one_temp;
   PetscInt rows_ao, cols_ao;
   PetscInt global_rows, global_cols;
   PetscInt local_rows, local_cols;
   MatType mat_type;
   PetscInt nnzs_match_local, nnzs_match_nonlocal;
   Mat output_mat_local, output_mat_nonlocal;

   // Equivalent to calling MatSeqAIJKokkosSyncDevice which is petsc intern
   mat_sync(input_mat);   

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

   // Get the comm
   PetscCallVoid(PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX));
   PetscCallVoid(MatGetLocalSize(*input_mat, &local_rows, &local_cols));
   PetscCallVoid(MatGetSize(*input_mat, &global_rows, &global_cols));
   PetscCallVoid(MatGetOwnershipRange(*input_mat, &global_row_start_temp, &global_row_end_plus_one_temp));
   PetscCallVoid(MatGetOwnershipRangeColumn(*input_mat, &global_col_start_temp, &global_col_end_plus_one_temp));
   const PetscInt global_row_start = global_row_start_temp;
   //const PetscInt global_row_end_plus_one = global_row_end_plus_one_temp;
   const PetscInt global_col_start = global_col_start_temp;
   //const PetscInt global_col_end_plus_one = global_col_end_plus_one_temp;
   PetscCallVoid(MatGetType(*input_mat, &mat_type));
   auto exec = PetscGetKokkosExecutionSpace();

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   Kokkos::fence();
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

   // We always need to know if we found a diagonal in each row of the input_matrix
   auto found_diag_row_d = PetscIntKokkosView("found_diag_row_d", local_rows);
   Kokkos::deep_copy(exec, found_diag_row_d, 0);

   // ~~~~~~~~~~~~
   // Get the number of nnzs
   // ~~~~~~~~~~~~
   nnzs_match_local = 0;
   nnzs_match_nonlocal = 0;

   // We need to know how many entries are in each row 
   nnz_match_local_row_d = PetscIntKokkosView("nnz_match_local_row_d", local_rows);
   // We may have identity
   Kokkos::deep_copy(exec, nnz_match_local_row_d, 0);
   if (mpi) nnz_match_nonlocal_row_d = PetscIntKokkosView("nnz_match_nonlocal_row_d", local_rows);

   // Calculate if each row has a diagonal, we need to know this for both 
   // reuse and not reuse
   // For the local block we need to count the nnzs
   // but if there is no diagonal we need to add one in
   Kokkos::parallel_reduce(
      Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
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
            Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {

               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
               nnz_match_nonlocal_row_d(i) = ncols_nonlocal;
         });

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
      // Create i indices
      // ~~~~~~~~~~~~~~~
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {      

            // The start of our row index comes from the scan
            i_local_d(i + 1) = nnz_match_local_row_d(i);   
            if (mpi) i_nonlocal_d(i + 1) = nnz_match_nonlocal_row_d(i);         
      });

      // Loop over the rows
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
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
      // Because we might be missing diagonals and we're going to skip some of them
      // in the writing loop below
      Kokkos::deep_copy(exec, a_local_d, 0.0);

      // Annoyingly there isn't currently the ability to get views for i (or j)
      Kokkos::fence();
      const PetscInt *device_local_i_output = nullptr, *device_local_j_output = nullptr, *device_nonlocal_i_ouput = nullptr;
      PetscMemType mtype;
      PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_output, &device_local_i_output, &device_local_j_output, NULL, &mtype));  
      if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_output, &device_nonlocal_i_ouput, NULL, NULL, &mtype));  

      // Have these point at the existing i pointers - we only need the local j
      ConstMatRowMapKokkosView i_local_const_d = ConstMatRowMapKokkosView(device_local_i_output, local_rows+1);
      ConstMatRowMapKokkosView j_local_const_d = ConstMatRowMapKokkosView(device_local_j_output, aijkok_local_output->csrmat.nnz());
      ConstMatRowMapKokkosView i_nonlocal_const_d;
      if (mpi) i_nonlocal_const_d = ConstMatRowMapKokkosView(device_nonlocal_i_ouput, local_rows+1);         

      // Only have to write a but have to be careful as we may not have diagonals in some rows
      // in the input, but they are in the output
      // Loop over the rows - annoying we have const views as this is just the same loop as above
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
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
      
      Kokkos::fence();

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
      Kokkos::fence();     

      // Now we have to sort the local column indices, as we add in the identity at the 
      // end of our local j indices      
      KokkosCsrMatrix csrmat_local = KokkosCsrMatrix("csrmat_local", local_rows, local_cols, a_local_d.extent(0), a_local_d, i_local_d, j_local_d);  
      Kokkos::fence();
      KokkosSparse::sort_crs_matrix(csrmat_local);       

      // Let's make sure everything on the device is finished
      Kokkos::fence();       

      // Create the matrix given the sorted csr
      PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, local_cols, i_local_d, j_local_d, a_local_d, &output_mat_local));

      // we also have to go and build our off block matrix and then the output
      if (mpi) 
      {
         // We know the garray is just the original
         PetscInt *garray_host = NULL; 
         PetscCallVoid(PetscMalloc1(cols_ao, &garray_host));
         for (PetscInt i = 0; i < cols_ao; i++)
         {
            garray_host[i] = colmap_input[i];
         }    
         
         // We can create our nonlocal diagonal block matrix directly on the device
         PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, cols_ao, i_nonlocal_d, j_nonlocal_d, a_nonlocal_d, &output_mat_nonlocal)); 

         // We can now create our MPI matrix
         PetscCallVoid(MatCreateMPIAIJWithSeqAIJ(MPI_COMM_MATRIX, global_rows, global_cols, output_mat_local, output_mat_nonlocal, garray_host, output_mat));
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
   //PflareKokkosTrace _trace("MatAXPY_kokkos");
   Mat mat_local_y = NULL, mat_nonlocal_y = NULL;
   Mat mat_local_x = NULL, mat_nonlocal_x = NULL;

   const PetscInt *colmap_y, *colmap_x;
   PetscCallVoid(MatMPIAIJGetSeqAIJ(*Y, &mat_local_y, &mat_nonlocal_y, &colmap_y));
   PetscCallVoid(MatMPIAIJGetSeqAIJ(*X, &mat_local_x, &mat_nonlocal_x, &colmap_x));

   // Equivalent to calling MatSeqAIJKokkosSyncDevice which is petsc intern
   // We have to make sure the device data is up to date before we do the axpy   
   mat_sync(X);
   mat_sync(Y);

   PetscInt rows_ao_y, cols_ao_y, rows_ao_x, cols_ao_x;
   auto exec = PetscGetKokkosExecutionSpace();

   PetscCallVoid(MatGetSize(mat_nonlocal_y, &rows_ao_y, &cols_ao_y));
   PetscCallVoid(MatGetSize(mat_nonlocal_x, &rows_ao_x, &cols_ao_x));

   // We also copy the colmaps over to the device as we need it
   PetscIntConstKokkosViewHost colmap_input_h_y = PetscIntConstKokkosViewHost(colmap_y, cols_ao_y);
   PetscIntKokkosView colmap_input_d_y = PetscIntKokkosView("colmap_input_d_y", cols_ao_y);
   Kokkos::deep_copy(exec, colmap_input_d_y, colmap_input_h_y);
   // Log copy with petsc
   size_t bytes = colmap_input_h_y.extent(0) * sizeof(PetscInt);
   PetscCallVoid(PetscLogCpuToGpu(bytes));

   PetscIntConstKokkosViewHost colmap_input_h_x = PetscIntConstKokkosViewHost(colmap_x, cols_ao_x);
   PetscIntKokkosView colmap_input_d_x = PetscIntKokkosView("colmap_input_d_x", cols_ao_x);
   Kokkos::deep_copy(exec, colmap_input_d_x, colmap_input_h_x);
   // Log copy with petsc
   bytes = colmap_input_h_x.extent(0) * sizeof(PetscInt);
   PetscCallVoid(PetscLogCpuToGpu(bytes));

   // Get the comm
   MPI_Comm MPI_COMM_MATRIX;
   PetscCallVoid(PetscObjectGetComm((PetscObject)*Y, &MPI_COMM_MATRIX));
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscCallVoid(MatGetLocalSize(*Y, &local_rows, &local_cols));
   PetscCallVoid(MatGetSize(*Y, &global_rows, &global_cols));

   // ~~~~~~~~~~~~~~~
   // Let's go and add the local components together
   // ~~~~~~~~~~~~~~~
   Kokkos::fence(); 

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

      KokkosSparse::spadd_symbolic(exec, &kh_local, xkok_local->csrmat, ykok_local->csrmat, zcsr_local);
      KokkosSparse::spadd_numeric(exec, &kh_local, alpha, xkok_local->csrmat, (PetscScalar)1.0, ykok_local->csrmat, zcsr_local);

      kh_local.destroy_spadd_handle();
      
      // Get the Kokkos Views from zcsr_local - annoyingly we can't just call MatCreateSeqAIJKokkosWithCSRMatrix
      // as it's petsc intern
      auto a_local_d_z = zcsr_local.values;
      auto i_local_d_z = zcsr_local.graph.row_map;
      auto j_local_d_z = zcsr_local.graph.entries;   

      a_local_d_copy = Kokkos::View<PetscScalar *>("a_local_d_copy", a_local_d_z.extent(0));
      i_local_d_copy = Kokkos::View<PetscInt *>("i_local_d_copy", i_local_d_z.extent(0));
      j_local_d_copy = Kokkos::View<PetscInt *>("j_local_d_copy", j_local_d_z.extent(0));   

      // Let's make sure everything on the device is finished
      Kokkos::fence();      

      Kokkos::deep_copy(exec, a_local_d_copy, a_local_d_z);
      Kokkos::deep_copy(exec, i_local_d_copy, i_local_d_z);
      Kokkos::deep_copy(exec, j_local_d_copy, j_local_d_z);
   }
   Kokkos::fence();

   // We can create our local diagonal block matrix directly on the device
   Mat Z_local;
   PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, local_cols, i_local_d_copy, j_local_d_copy, a_local_d_copy, &Z_local));
   
   // ~~~~~~~~~~~~~~~
   // Now let's go and add the non-local components together
   // We first rewrite the j indices to be global as the nonlocal components of Y and X
   // might have different non-local non-zeros (and different numbers of non-local non-zeros)
   // ~~~~~~~~~~~~~~~

   // We need to duplicate the nonlocal part of x first as we are going to overwrite the 
   // column indices
   // Don't need to copy y as we destroy it anyway
   Mat mat_nonlocal_x_copy;
   PetscCallVoid(MatDuplicate(mat_nonlocal_x, MAT_COPY_VALUES, &mat_nonlocal_x_copy));

   Mat_SeqAIJKokkos *xkok_nonlocal, *ykok_nonlocal; 
   ykok_nonlocal = static_cast<Mat_SeqAIJKokkos *>(mat_nonlocal_y->spptr);
   xkok_nonlocal = static_cast<Mat_SeqAIJKokkos *>(mat_nonlocal_x_copy->spptr);          

   PetscInt *device_nonlocal_x_j = xkok_nonlocal->j_device_data();
   PetscInt *device_nonlocal_y_j = ykok_nonlocal->j_device_data();

   // Rewrite the Y nonlocal indices to be global
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, ykok_nonlocal->csrmat.nnz()), KOKKOS_LAMBDA(PetscInt i) { 

         device_nonlocal_y_j[i] = colmap_input_d_y(device_nonlocal_y_j[i]);
   }); 

   // Rewrite the X nonlocal indices to be global
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, xkok_nonlocal->csrmat.nnz()), KOKKOS_LAMBDA(PetscInt i) { 

         device_nonlocal_x_j[i] = colmap_input_d_x(device_nonlocal_x_j[i]);
   });    
   // Ensure everything is finished before we hit the spadd below
   Kokkos::fence();

   // ~~~~~~~~~
   // Build merged garray from the union of X and Y global nonlocal column indices.
   // Then remap both to local indices so spadd sees correct column numbering.
   // ~~~~~~~~~

   PetscInt nnz_x = xkok_nonlocal->csrmat.nnz();
   PetscInt nnz_y = ykok_nonlocal->csrmat.nnz();
   PetscInt total_nnz_xy = nnz_x + nnz_y;

   // Non-owning views over the raw j data (already holds global indices at this point)
   PetscIntKokkosView j_x_view(device_nonlocal_x_j, nnz_x);
   PetscIntKokkosView j_y_view(device_nonlocal_y_j, nnz_y);

   PetscIntKokkosView garray_d;
   PetscInt col_ao_output = 0;

   if (total_nnz_xy > 0)
   {
      // Concatenate all global j indices into one array, sort, unique → merged garray
      PetscIntKokkosView combined_j_d("combined_j_d", total_nnz_xy);
      Kokkos::deep_copy(exec, Kokkos::subview(combined_j_d, Kokkos::make_pair((PetscInt)0, nnz_x)), j_x_view);
      Kokkos::deep_copy(exec, Kokkos::subview(combined_j_d, Kokkos::make_pair(nnz_x, total_nnz_xy)), j_y_view);
      Kokkos::sort(exec, combined_j_d);
      Kokkos::fence();

      PetscIntKokkosView garray_full_d("garray_full_d", total_nnz_xy);
      auto unique_end_it = Kokkos::Experimental::unique_copy(exec, combined_j_d, garray_full_d);
      Kokkos::fence();
      col_ao_output = static_cast<PetscInt>(unique_end_it - Kokkos::Experimental::begin(garray_full_d));
      PetscInt zero = 0;
      garray_d = Kokkos::subview(garray_full_d, Kokkos::make_pair(zero, col_ao_output));

      // Remap j_y and j_x from global to local indices into the merged garray
      // These fence internally
      remap_j_to_local_device(j_y_view, garray_d, col_ao_output);
      remap_j_to_local_device(j_x_view, garray_d, col_ao_output);
   }

   // ~~~~~~~~~

   Kokkos::View<PetscScalar *> a_nonlocal_d_copy;
   Kokkos::View<PetscInt *> i_nonlocal_d_copy, j_nonlocal_d_copy;
   PetscInt *garray_host = NULL;

   // Scope so the zcsr_nonlocal is destroyed once we copy
   {
      // Create csrmat wrappers for X and Y with the correct merged numCols
      KokkosCsrMatrix xcsrmat("x_nonlocal_remapped", local_rows, col_ao_output,
         nnz_x, xkok_nonlocal->csrmat.values,
         xkok_nonlocal->csrmat.graph.row_map, xkok_nonlocal->csrmat.graph.entries);
      KokkosCsrMatrix ycsrmat("y_nonlocal_remapped", local_rows, col_ao_output,
         nnz_y, ykok_nonlocal->csrmat.values,
         ykok_nonlocal->csrmat.graph.row_map, ykok_nonlocal->csrmat.graph.entries);

      Kokkos::fence();
      // Now we can add the non-local components together.
      // Local indices into the merged sorted garray preserve row-sort order.
      KokkosCsrMatrix zcsr_nonlocal;
      KernelHandle    kh_nonlocal;
      kh_nonlocal.create_spadd_handle(true);

      KokkosSparse::spadd_symbolic(exec, &kh_nonlocal, xcsrmat, ycsrmat, zcsr_nonlocal);
      KokkosSparse::spadd_numeric(exec, &kh_nonlocal, alpha, xcsrmat, (PetscScalar)1.0, ycsrmat, zcsr_nonlocal);

      kh_nonlocal.destroy_spadd_handle();

      Kokkos::fence();

      // Can now destroy the copy
      PetscCallVoid(MatDestroy(&mat_nonlocal_x_copy));

      // Get the Kokkos Views from zcsr_nonlocal - annoyingly we can't just call MatCreateSeqAIJKokkosWithCSRMatrix
      // as it's petsc intern
      auto a_nonlocal_d_z = zcsr_nonlocal.values;
      auto i_nonlocal_d_z = zcsr_nonlocal.graph.row_map;
      auto j_nonlocal_d_z = zcsr_nonlocal.graph.entries;

      // j_nonlocal_d_z already contains local indices; copy garray_d to host
      PetscCallVoid(PetscMalloc1(col_ao_output, &garray_host));
      if (col_ao_output > 0)
      {
         PetscIntKokkosViewHost garray_h(garray_host, col_ao_output);
         // Device to host so don't need to specify exec space
         Kokkos::deep_copy(exec, garray_h, garray_d);
         Kokkos::fence();
         size_t bytes = col_ao_output * sizeof(PetscInt);
         PetscCallVoid(PetscLogGpuToCpu(bytes));
      }

      // Let's make sure everything on the device is finished
      Kokkos::fence();

      a_nonlocal_d_copy = Kokkos::View<PetscScalar *>("a_local_d_copy", a_nonlocal_d_z.extent(0));
      i_nonlocal_d_copy = Kokkos::View<PetscInt *>("i_local_d_copy", i_nonlocal_d_z.extent(0));
      j_nonlocal_d_copy = Kokkos::View<PetscInt *>("j_local_d_copy", j_nonlocal_d_z.extent(0));

      Kokkos::deep_copy(exec, a_nonlocal_d_copy, a_nonlocal_d_z);
      Kokkos::deep_copy(exec, i_nonlocal_d_copy, i_nonlocal_d_z);
      Kokkos::deep_copy(exec, j_nonlocal_d_copy, j_nonlocal_d_z);
   }

   Kokkos::fence();

   // We can create our nonlocal diagonal block matrix directly on the device
   Mat Z_nonlocal;
   PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, col_ao_output, i_nonlocal_d_copy, j_nonlocal_d_copy, a_nonlocal_d_copy, &Z_nonlocal));

   // We can now create our MPI matrix
   Mat Z;
   PetscCallVoid(MatCreateMPIAIJWithSeqAIJ(MPI_COMM_MATRIX, global_rows, global_cols, Z_local, Z_nonlocal, garray_host, &Z));

   // Stick Z into the input Y (this destroys existing Y)
   PetscCallVoid(MatHeaderReplace(*Y, &Z));

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Does a MatGetSubMatrix for a sequential Kokkos matrix - the petsc version currently uses the host making it very slow
// is_row_d_d and is_col_d_d must have the local indices in them
// is_col must be sorted
PETSC_INTERN void MatCreateSubMatrix_Seq_kokkos(Mat *input_mat, PetscIntKokkosView &is_row_d_d, PetscIntKokkosView &is_col_d_d, const int reuse_int, Mat *output_mat)
{
   //PflareKokkosTrace _trace("MatCreateSubMatrix_Seq_kokkos");
   PetscInt local_rows, local_cols;
   PetscInt nnzs_match_local;
   auto exec = PetscGetKokkosExecutionSpace();   

   PetscCallVoid(MatGetLocalSize(*input_mat, &local_rows, &local_cols));
   PetscInt local_rows_row = is_row_d_d.extent(0), local_cols_col = is_col_d_d.extent(0);

   // ~~~~~~~~~~~~
   // DIAGNOSTIC (Step 1 of plan): verify is_row_d_d / is_col_d_d are in-bounds.
   // If a caller supplies out-of-range indices, smap_d / device_local_i accesses
   // below would silently clobber adjacent device allocations.
   // ~~~~~~~~~~~~
   {
      PetscInt row_min = 0, row_max = -1, col_min = 0, col_max = -1;
      if (local_rows_row > 0) {
         Kokkos::parallel_reduce("PFLARE_DBG_is_row_minmax",
            Kokkos::RangePolicy<>(exec, 0, local_rows_row),
            KOKKOS_LAMBDA(const PetscInt i, PetscInt &lmin) {
               const PetscInt v = is_row_d_d(i);
               if (v < lmin) lmin = v;
            }, Kokkos::Min<PetscInt>(row_min));
         Kokkos::parallel_reduce("PFLARE_DBG_is_row_max",
            Kokkos::RangePolicy<>(exec, 0, local_rows_row),
            KOKKOS_LAMBDA(const PetscInt i, PetscInt &lmax) {
               const PetscInt v = is_row_d_d(i);
               if (v > lmax) lmax = v;
            }, Kokkos::Max<PetscInt>(row_max));
      }
      if (local_cols_col > 0) {
         Kokkos::parallel_reduce("PFLARE_DBG_is_col_minmax",
            Kokkos::RangePolicy<>(exec, 0, local_cols_col),
            KOKKOS_LAMBDA(const PetscInt i, PetscInt &lmin) {
               const PetscInt v = is_col_d_d(i);
               if (v < lmin) lmin = v;
            }, Kokkos::Min<PetscInt>(col_min));
         Kokkos::parallel_reduce("PFLARE_DBG_is_col_max",
            Kokkos::RangePolicy<>(exec, 0, local_cols_col),
            KOKKOS_LAMBDA(const PetscInt i, PetscInt &lmax) {
               const PetscInt v = is_col_d_d(i);
               if (v > lmax) lmax = v;
            }, Kokkos::Max<PetscInt>(col_max));
      }
      Kokkos::fence();
      if (local_rows_row > 0) {
         PetscCheckAbort(row_min >= 0 && row_max < local_rows, PETSC_COMM_SELF,
            PETSC_ERR_ARG_OUTOFRANGE,
            "MatCreateSubMatrix_Seq_kokkos: is_row out of range [0,%" PetscInt_FMT ") got [%" PetscInt_FMT ",%" PetscInt_FMT "]",
            local_rows, row_min, row_max);
      }
      if (local_cols_col > 0) {
         PetscCheckAbort(col_min >= 0 && col_max < local_cols, PETSC_COMM_SELF,
            PETSC_ERR_ARG_OUTOFRANGE,
            "MatCreateSubMatrix_Seq_kokkos: is_col out of range [0,%" PetscInt_FMT ") got [%" PetscInt_FMT ",%" PetscInt_FMT "]",
            local_cols, col_min, col_max);
      }
   }

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   Kokkos::fence();
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(*input_mat, &device_local_i, &device_local_j, &device_local_vals, &mtype));

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
   Kokkos::deep_copy(exec, nnz_match_local_row_d, 0);

   // Map which columns in the original mat are in is_col
   PetscIntKokkosView smap_d = PetscIntKokkosView("smap_d", local_cols);
   Kokkos::deep_copy(exec, smap_d, 0);
   // Loop over all the cols in is_col
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, local_cols_col), KOKKOS_LAMBDA(PetscInt i) {      

         smap_d(is_col_d_d(i)) = i + 1; 
   });     
   
   // ~~~~~~~~~~~~
   // Need to count the number of nnzs we end up with, on each row and in total
   // ~~~~~~~~~~~~
   // Only loop over the number of rows in is_row
   if (!reuse_int)
   {
      Kokkos::parallel_reduce(
         Kokkos::TeamPolicy<>(exec, local_rows_row, Kokkos::AUTO()),
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

      Kokkos::parallel_reduce("FindMaxNNZ", Kokkos::RangePolicy<>(exec, 0, local_rows_row),
         KOKKOS_LAMBDA(const PetscInt i_idx_is_row, PetscInt& thread_max) {
            // The indices in is_row will be global, but we want the local index
            const PetscInt i = is_row_d_d(i_idx_is_row);
            PetscInt row_nnz = device_local_i[i + 1] - device_local_i[i];
            thread_max = (row_nnz > thread_max) ? row_nnz : thread_max;
         },
         Kokkos::Max<PetscInt>(max_nnz_local)
      );
   }     

   // Create a team policy with scratch memory allocation
   // We want scratch space for each row
   // We will ncols+1 of integers which tell us what the matching indices we have
   const size_t per_view_local = ScratchIntView::shmem_size(max_nnz_local+1);

   Kokkos::TeamPolicy<> policy(exec, local_rows_row, Kokkos::AUTO());
   // We're gonna use the level 1 scratch as our column data is probably bigger than the level 0
   policy.set_scratch_size(1, Kokkos::PerTeam(per_view_local));    

   // Only need things to do with the sparsity pattern if we're not reusing
   if (!reuse_int)
   {
      // Need to do a scan on nnz_match_local_row_d to get where each row starts
      Kokkos::parallel_scan (Kokkos::RangePolicy<>(exec, 0, local_rows_row), KOKKOS_LAMBDA (const PetscInt i_idx_is_row, PetscInt& update, const bool final) {
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
      Kokkos::deep_copy(exec, Kokkos::subview(i_local_d, 0), 0);

      // ~~~~~~~~~~~~~~~
      // Create i indices
      // ~~~~~~~~~~~~~~~
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, local_rows_row), KOKKOS_LAMBDA(PetscInt i_idx_is_row) {

            // The start of our row index comes from the scan
            i_local_d(i_idx_is_row + 1) = nnz_match_local_row_d(i_idx_is_row);
      });

      // ~~~~~~~~~~~~
      // DIAGNOSTIC (Step 1b of plan): verify i_local_d's final value equals
      // nnzs_match_local, and that device_local_j entries for the rows we touch
      // are all inside [0, local_cols). Either inconsistency would cause the
      // team kernel below to write j_local_d / a_local_d outside their bounds.
      // ~~~~~~~~~~~~
      if (local_rows_row > 0) {
         PetscInt i_local_last_h = 0;
         auto i_local_tail = Kokkos::subview(i_local_d, local_rows_row);
         Kokkos::View<PetscInt, Kokkos::HostSpace> i_local_tail_h("PFLARE_DBG_i_local_tail");
         Kokkos::deep_copy(exec, i_local_tail_h, i_local_tail);
         Kokkos::fence();
         i_local_last_h = i_local_tail_h();
         PetscCheckAbort(i_local_last_h == nnzs_match_local, PETSC_COMM_SELF,
            PETSC_ERR_PLIB,
            "MatCreateSubMatrix_Seq_kokkos: i_local_d tail (%" PetscInt_FMT ") != nnzs_match_local (%" PetscInt_FMT "), local_rows_row=%" PetscInt_FMT,
            i_local_last_h, nnzs_match_local, local_rows_row);

         PetscInt jmin = 0, jmax = -1;
         Kokkos::parallel_reduce("PFLARE_DBG_dev_j_min",
            Kokkos::RangePolicy<>(exec, 0, local_rows_row),
            KOKKOS_LAMBDA(const PetscInt ir, PetscInt &lmin) {
               const PetscInt i = is_row_d_d(ir);
               const PetscInt s = device_local_i[i];
               const PetscInt e = device_local_i[i + 1];
               for (PetscInt k = s; k < e; ++k) {
                  const PetscInt v = device_local_j[k];
                  if (v < lmin) lmin = v;
               }
            }, Kokkos::Min<PetscInt>(jmin));
         Kokkos::parallel_reduce("PFLARE_DBG_dev_j_max",
            Kokkos::RangePolicy<>(exec, 0, local_rows_row),
            KOKKOS_LAMBDA(const PetscInt ir, PetscInt &lmax) {
               const PetscInt i = is_row_d_d(ir);
               const PetscInt s = device_local_i[i];
               const PetscInt e = device_local_i[i + 1];
               for (PetscInt k = s; k < e; ++k) {
                  const PetscInt v = device_local_j[k];
                  if (v > lmax) lmax = v;
               }
            }, Kokkos::Max<PetscInt>(jmax));
         Kokkos::fence();
         if (jmax >= 0) {
            PetscCheckAbort(jmin >= 0 && jmax < local_cols, PETSC_COMM_SELF,
               PETSC_ERR_PLIB,
               "MatCreateSubMatrix_Seq_kokkos: device_local_j out of [0,%" PetscInt_FMT ") got [%" PetscInt_FMT ",%" PetscInt_FMT "]",
               local_cols, jmin, jmax);
         }
      }

      // Execute with scratch memory
      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const KokkosTeamMemberType& t) {

         // i_idx_is_row is the row index into the output
         const PetscInt i_idx_is_row = t.league_rank();
         // i is the row index into the input
         const PetscInt i = is_row_d_d(i_idx_is_row);

         // number of columns
         PetscInt ncols_local;
         ncols_local = device_local_i[i + 1] - device_local_i[i];
         ScratchIntView scratch_indices;

         // DIAGNOSTIC: ncols_local must not exceed max_nnz_local.
         // If it does the scratch allocation below overruns the per-team
         // budget and silently corrupts adjacent device memory.
         // Use Kokkos::abort (not KOKKOS_ASSERT) so this fires unconditionally
         // regardless of NDEBUG / KOKKOS_ENABLE_DEBUG build flags.
         if (ncols_local > max_nnz_local) Kokkos::abort("PFLARE: ncols_local > max_nnz_local in MatCreateSubMatrix_Seq_kokkos — scratch pool overflow");

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

      // ~~~~~~~~~~~~
      // DIAGNOSTIC (Step 1c of plan): post-team-kernel sanity check on the
      // produced j_local_d. Every column index handed to PETSc must be in
      // [0, local_cols_col); a value outside that range would either be a
      // smap_d corruption or a per-row scan / write-offset bug.
      // ~~~~~~~~~~~~
      if (nnzs_match_local > 0) {
         PetscInt jout_min = 0, jout_max = -1;
         Kokkos::parallel_reduce("PFLARE_DBG_jlocal_min",
            Kokkos::RangePolicy<>(exec, 0, nnzs_match_local),
            KOKKOS_LAMBDA(const PetscInt k, PetscInt &lmin) {
               const PetscInt v = j_local_d(k);
               if (v < lmin) lmin = v;
            }, Kokkos::Min<PetscInt>(jout_min));
         Kokkos::parallel_reduce("PFLARE_DBG_jlocal_max",
            Kokkos::RangePolicy<>(exec, 0, nnzs_match_local),
            KOKKOS_LAMBDA(const PetscInt k, PetscInt &lmax) {
               const PetscInt v = j_local_d(k);
               if (v > lmax) lmax = v;
            }, Kokkos::Max<PetscInt>(jout_max));
         Kokkos::fence();
         PetscCheckAbort(jout_min >= 0 && jout_max < local_cols_col, PETSC_COMM_SELF,
            PETSC_ERR_PLIB,
            "MatCreateSubMatrix_Seq_kokkos: j_local_d out of [0,%" PetscInt_FMT ") got [%" PetscInt_FMT ",%" PetscInt_FMT "], nnzs=%" PetscInt_FMT,
            local_cols_col, jout_min, jout_max, nnzs_match_local);
      }
   }
   // If we're reusing, we can just write directly to the existing views
   else
   {
      Mat_SeqAIJKokkos *aijkok_local_output = static_cast<Mat_SeqAIJKokkos *>((*output_mat)->spptr);

      // Annoying we can't just call MatSeqAIJGetKokkosView
      a_local_d = aijkok_local_output->a_dual.view_device();

      // Annoyingly there isn't currently the ability to get views for i (or j)
      Kokkos::fence();
      const PetscInt *device_local_i_output = nullptr;
      PetscMemType mtype;
      PetscCallVoid(MatSeqAIJGetCSRAndMemType(*output_mat, &device_local_i_output, NULL, NULL, &mtype));

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
         ScratchIntView scratch_indices;

         // DIAGNOSTIC: same scratch-overflow guard as in the non-reuse kernel above.
         KOKKOS_ASSERT(ncols_local <= max_nnz_local);

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
      
      Kokkos::fence();      

      // Have to specify we've modifed data on the device
      // Want to call MatSeqAIJKokkosModifyDevice but its PETSC_INTERN
      aijkok_local_output->a_dual.clear_sync_state();
      aijkok_local_output->a_dual.modify_device();
      aijkok_local_output->transpose_updated = PETSC_FALSE;
      aijkok_local_output->hermitian_updated = PETSC_FALSE;
      PetscObjectStateIncrease((PetscObject)(*output_mat));    

   }

   // Let's make sure everything on the device is finished
   Kokkos::fence();   

   // ~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~   

   if (!reuse_int)
   {   
      // Create the matrix given the sorted csr
      PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows_row, local_cols_col, i_local_d, j_local_d, a_local_d, output_mat));
   }  

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Does a MatGetSubMatrix for a Kokkos matrix - the petsc version currently uses the host making it very slow
// This version only works  works if the input IS have the same parallel row/column distribution 
// as the matrices, ie equivalent to MatCreateSubMatrix_MPIAIJ_SameRowDist
// is_col must be sorted
// This one uses the views is_row_d_d and is_col_d_d directly, rewritten to be the local indices
PETSC_INTERN void MatCreateSubMatrix_kokkos_view(Mat *input_mat, PetscIntKokkosView is_row_d_d, PetscInt global_rows_row, \
         PetscIntKokkosView is_col_d_d, PetscInt global_cols_col, const int reuse_int, Mat *output_mat, IS *rows_rows, IS *cols_cols)
{
//    //PflareKokkosTrace _trace("MatCreateSubMatrix_kokkos_view");
//    PetscInt local_rows, local_cols;
//    PetscInt global_rows, global_cols;
//    PetscInt global_row_start, global_row_end_plus_one;
//    // PetscCallVoid(MatGetOwnershipRange(*input_mat, &global_row_start, &global_row_end_plus_one));
//    PetscInt local_cols_col = is_col_d_d.extent(0);
//    auto exec = PetscGetKokkosExecutionSpace();

//    // // Are we in parallel?
//    // MatType mat_type;
//    MPI_Comm MPI_COMM_MATRIX;
//    // PetscCallVoid(MatGetType(*input_mat, &mat_type));

//    // const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;   
//    const bool mpi = true;
//    // PetscCallVoid(PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX));
//    // PetscCallVoid(MatGetSize(*input_mat, &global_rows, &global_cols));
//    // PetscCallVoid(MatGetLocalSize(*input_mat, &local_rows, &local_cols));

//    Mat_MPIAIJ *mat_mpi = nullptr;
//    Mat mat_local = NULL, mat_nonlocal = NULL;   
//    Mat output_mat_local, output_mat_nonlocal;
  
//    PetscInt rows_ao, cols_ao;
//    // if (mpi)
//    // {
//    //    mat_mpi = (Mat_MPIAIJ *)(*input_mat)->data;
//    //    PetscCallVoid(MatMPIAIJGetSeqAIJ(*input_mat, &mat_local, &mat_nonlocal, NULL));
//    //    PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao)); 
      
//    //    if (reuse_int)
//    //    {
//    //       PetscCallVoid(MatMPIAIJGetSeqAIJ(*output_mat, &output_mat_local, &output_mat_nonlocal, NULL));
//    //    }
//    // }
//    // else
//    // {
//    //    mat_local = *input_mat;
//    //    if (reuse_int) output_mat_local = *output_mat;
//    // }
//    size_t bytes = 0;

// // Ablation toggle (Step 2 of plan): when defined non-zero, the diagonal
// // MatCreateSubMatrix_Seq_kokkos call is replaced by PETSc's host-side
// // MatCreateSubMatrix on mat_local plus a MatConvert back to MATSEQAIJKOKKOS.
// // Used to test whether the intermittent GPU crash originates inside the
// // diag Seq_kokkos kernel chain. Reuse path is unchanged (crashes are
// // first-call only). Toggle off (set to 0) to restore the original path.
// #ifndef PFLARE_ABLATE_DIAG_SUBMAT
// #define PFLARE_ABLATE_DIAG_SUBMAT 0
// #endif

// //    // The diagonal component
// // #if PFLARE_ABLATE_DIAG_SUBMAT
// //    if (!reuse_int)
// //    {
// //       // Pull the (already-local) is_row / is_col indices back to the host so
// //       // PETSc's CPU MatCreateSubMatrix can consume them. mat_local is a
// //       // SeqAIJKokkos but PETSc's MatCreateSubMatrix dispatches to the host
// //       // SeqAIJ implementation, producing a SeqAIJ result that we then convert
// //       // back to SeqAIJKokkos for the downstream MatCreateMPIAIJWithSeqAIJ.
// //       const PetscInt n_row_h = is_row_d_d.extent(0);
// //       const PetscInt n_col_h = is_col_d_d.extent(0);
// //       PetscInt *is_row_host_arr = NULL, *is_col_host_arr = NULL;
// //       PetscCallVoid(PetscMalloc1(n_row_h > 0 ? n_row_h : 1, &is_row_host_arr));
// //       PetscCallVoid(PetscMalloc1(n_col_h > 0 ? n_col_h : 1, &is_col_host_arr));
// //       PetscIntKokkosViewHost is_row_h_view(is_row_host_arr, n_row_h);
// //       PetscIntKokkosViewHost is_col_h_view(is_col_host_arr, n_col_h);
// //       Kokkos::deep_copy(exec, is_row_h_view, is_row_d_d);
// //       Kokkos::deep_copy(exec, is_col_h_view, is_col_d_d);
// //       Kokkos::fence();

// //       IS is_row_temp = NULL, is_col_temp = NULL;
// //       PetscCallVoid(ISCreateGeneral(PETSC_COMM_SELF, n_row_h, is_row_host_arr, PETSC_COPY_VALUES, &is_row_temp));
// //       PetscCallVoid(ISCreateGeneral(PETSC_COMM_SELF, n_col_h, is_col_host_arr, PETSC_COPY_VALUES, &is_col_temp));

// //       Mat tmp_host_mat = NULL;
// //       PetscCallVoid(MatCreateSubMatrix(mat_local, is_row_temp, is_col_temp, MAT_INITIAL_MATRIX, &output_mat_local));
// //       // Convert the SeqAIJ host result to SeqAIJKokkos so the downstream
// //       // MatCreateMPIAIJWithSeqAIJ + reuse storage hand-off still get a Kokkos
// //       // seq block (matches what MatCreateSubMatrix_Seq_kokkos would have
// //       // produced).
// //       //PetscCallVoid(MatConvert(tmp_host_mat, MATSEQAIJKOKKOS, MAT_INITIAL_MATRIX, &output_mat_local));

// //       //PetscCallVoid(MatDestroy(&tmp_host_mat));
// //       PetscCallVoid(ISDestroy(&is_row_temp));
// //       PetscCallVoid(ISDestroy(&is_col_temp));
// //       PetscCallVoid(PetscFree(is_row_host_arr));
// //       PetscCallVoid(PetscFree(is_col_host_arr));
// //    }
// //    else
// //    {
// //       MatCreateSubMatrix_Seq_kokkos(&mat_local, is_row_d_d, is_col_d_d, reuse_int, &output_mat_local);
// //    }
// // #else
// //    MatCreateSubMatrix_Seq_kokkos(&mat_local, is_row_d_d, is_col_d_d, reuse_int, &output_mat_local);
// // #endif

//    // The off-diagonal component requires some comms
//    // Basically a copy of MatCreateSubMatrix_MPIAIJ_SameRowColDist

// // Off-diagonal ablation toggle (step 2a of plan): when non-zero, the entire
// // off-diag VecScatter + Seq_kokkos-nonlocal + MatCreateMPIAIJWithSeqAIJ path
// // is replaced by PETSc's CPU MatCreateSubMatrix on the full MPIAIJ input,
// // converted back to MATMPIAIJKOKKOS.  Combine with PFLARE_ABLATE_DIAG_SUBMAT=0
// // so that only the off-diag section is ablated while diag uses our Kokkos kernel.
// // Only the first-call (non-reuse) path is ablated, matching the observed failure mode.
// #ifndef PFLARE_ABLATE_OFFDIAG_SUBMAT
// #define PFLARE_ABLATE_OFFDIAG_SUBMAT 1
// #endif

//    if (mpi)
//    {
// #if PFLARE_ABLATE_OFFDIAG_SUBMAT
//       if (!reuse_int)
//       {
// //          // We need global IS indices (is_row/is_col on device are already LOCAL,
// //          // i.e. row_global - global_row_start; add back the offset before calling
// //          // PETSc's CPU MatCreateSubMatrix which expects global indices).
// //          PetscInt global_row_start_abl = 0, global_row_end_abl = 0;
// //          PetscInt global_col_start_abl = 0, global_col_end_abl = 0;
// //          PetscCallVoid(MatGetOwnershipRange(*input_mat, &global_row_start_abl, &global_row_end_abl));
// //          PetscCallVoid(MatGetOwnershipRangeColumn(*input_mat, &global_col_start_abl, &global_col_end_abl));

// //          const PetscInt n_row_abl = (PetscInt)is_row_d_d.extent(0);
// //          const PetscInt n_col_abl = (PetscInt)is_col_d_d.extent(0);
// //          PetscInt *is_row_g_arr = NULL, *is_col_g_arr = NULL;
// //          PetscCallVoid(PetscMalloc1(n_row_abl > 0 ? n_row_abl : 1, &is_row_g_arr));
// //          PetscCallVoid(PetscMalloc1(n_col_abl > 0 ? n_col_abl : 1, &is_col_g_arr));

// //          // Copy local device indices to host then shift back to global.
// //          PetscIntKokkosViewHost is_row_g_h(is_row_g_arr, n_row_abl);
// //          PetscIntKokkosViewHost is_col_g_h(is_col_g_arr, n_col_abl);
// //          Kokkos::deep_copy(exec, is_row_g_h, is_row_d_d);
// //          Kokkos::deep_copy(exec, is_col_g_h, is_col_d_d);
// //          Kokkos::fence();
// //          for (PetscInt ii = 0; ii < n_row_abl; ii++) is_row_g_arr[ii] += global_row_start_abl;
// //          for (PetscInt ii = 0; ii < n_col_abl; ii++) is_col_g_arr[ii] += global_col_start_abl;

// //          IS is_row_g_abl = NULL, is_col_g_abl = NULL;
// //          PetscCallVoid(ISCreateGeneral(MPI_COMM_MATRIX, n_row_abl, is_row_g_arr, PETSC_OWN_POINTER, &is_row_g_abl));
// //          PetscCallVoid(ISCreateGeneral(MPI_COMM_MATRIX, n_col_abl, is_col_g_arr, PETSC_OWN_POINTER, &is_col_g_abl));

// //          PetscBool equal_flag;
// //          PetscCallVoid(ISEqualUnsorted(is_row_g_abl, *rows_rows, &equal_flag));

// // PetscCheckAbort(equal_flag, MPI_COMM_MATRIX,
// //                PETSC_ERR_PLIB,
// //                "rows not equal");       
               
// //          PetscCallVoid(ISEqualUnsorted(is_col_g_abl, *cols_cols, &equal_flag));

// // PetscCheckAbort(equal_flag, MPI_COMM_MATRIX,
// //                PETSC_ERR_PLIB,
// //                "cols not equal");                

// //          Mat tmp_abl = NULL;
//          //PetscCallVoid(MatCreateSubMatrix(*input_mat, is_row_g_abl, is_col_g_abl, MAT_INITIAL_MATRIX, output_mat));
          PetscCallVoid(MatCreateSubMatrix(*input_mat, *rows_rows, *cols_cols, MAT_INITIAL_MATRIX, output_mat));
//          //PetscCallVoid(MatConvert(tmp_abl, MATMPIAIJKOKKOS, MAT_INITIAL_MATRIX, output_mat));
//          //PetscCallVoid(MatDestroy(&tmp_abl));
//          //PetscCallVoid(MatDestroy(&output_mat_local));   // diag mat no longer needed
//          //PetscCallVoid(ISDestroy(&is_row_g_abl));
//          //PetscCallVoid(ISDestroy(&is_col_g_abl));
          return;
//       }
// #endif
//       PetscIntKokkosView is_col_o_d, garray_output_d;

//       if (!reuse_int)
//       {
//          PetscInt isstart = 0;
//          /* Get start indices on each rank for the new columns */
//          MPI_Scan(&local_cols_col, &isstart, 1, MPIU_INT, MPI_SUM, MPI_COMM_MATRIX);
//          isstart -= local_cols_col;

//          // cmap values are encoded through PetscScalar and then cast back to PetscInt,
//          // so guard the exact integer range before using VecScatter transport.
//          // Anything larger than 9,000 trillion with 64 bit ints and 64 bit floats will break - should be fine for now
//          // Can't rely on PetscSFBcast with MPIU_INT as that was intermittently breaking
//          // on gpus so want to avoid
//          PetscInt max_encoded_value = global_cols_col > 0 ? global_cols_col - 1 : 0;
//          PetscCallVoid(check_exact_petscint_to_scalar_encoding(max_encoded_value, MPI_COMM_MATRIX));

//          // Kokkos version of ISGetSeqIS_SameColDist_Private (mpiaij.c)
//          // Uses VecScatter with PetscScalar Vecs (matching PETSc's own pattern)
//          // instead of direct PetscSFBcast with MPIU_INT on temporary views.

//          std::cerr << "one " << std::endl;

//          /* (1) iscol is a sub-column vector of mat, pad it with '-1.' to form a full vector x */
//          Vec x_vec, cmap_vec;
//          PetscCallVoid(MatCreateVecs(*input_mat, &x_vec, NULL));
//          PetscCallVoid(VecDuplicate(x_vec, &cmap_vec));

//          // Fill x_vec on device: x[is_col(i)] = is_col(i), rest = -1
         
//             PetscScalarKokkosView x_scalar_d;
//             PetscCallVoid(VecGetKokkosViewWrite(x_vec, &x_scalar_d));
//             Kokkos::deep_copy(exec, x_scalar_d, -1.0);
//             Kokkos::parallel_for(
//                Kokkos::RangePolicy<>(exec, 0, local_cols_col), KOKKOS_LAMBDA(PetscInt i) {
//                   x_scalar_d(is_col_d_d(i)) = (PetscScalar)is_col_d_d(i);
//             });
//             PetscCallVoid(VecRestoreKokkosViewWrite(x_vec, &x_scalar_d));
         

//                      std::cerr << "two " << std::endl;

//          /* (2) Scatter x and cmap using Mvctx to get their off-process portions */
//          // Keep at most one active communication on Mvctx at a time.
//          // While Begin/End is in flight, do not touch the corresponding send/recv buffers.
//          Vec x_leaf_vec;
//          PetscCallVoid(VecDuplicate(mat_mpi->lvec, &x_leaf_vec));
//          // Ensure send/receive buffers are stable before Begin.
//          Kokkos::fence();
//                      std::cerr << "two a " << std::endl;

//          PetscCallVoid(VecScatterBegin(mat_mpi->Mvctx, x_vec, x_leaf_vec, INSERT_VALUES, SCATTER_FORWARD));
//          // x scatter completed: x_leaf_vec is now safe to read.
//          PetscCallVoid(VecScatterEnd(mat_mpi->Mvctx, x_vec, x_leaf_vec, INSERT_VALUES, SCATTER_FORWARD));

//                      std::cerr << "two b" << std::endl;

//          // Fill cmap_vec on device: cmap[is_col(i)] = i + isstart, rest = -1
         
//             PetscScalarKokkosView cmap_scalar_d;
//             PetscCallVoid(VecGetKokkosViewWrite(cmap_vec, &cmap_scalar_d));
//             Kokkos::deep_copy(exec, cmap_scalar_d, -1.0);
//             Kokkos::parallel_for(
//                Kokkos::RangePolicy<>(exec, 0, local_cols_col), KOKKOS_LAMBDA(PetscInt i) {
//                   cmap_scalar_d(is_col_d_d(i)) = (PetscScalar)(i + isstart);
//             });
//             PetscCallVoid(VecRestoreKokkosViewWrite(cmap_vec, &cmap_scalar_d));
         
//          std::cerr << "three " << std::endl;

//          Vec lcmap_vec;
//          PetscCallVoid(VecDuplicate(mat_mpi->lvec, &lcmap_vec));

//          /* (3) Count how many off-local columns match */
//          PetscInt col_ao_output = 0;

//          // One bigger for exclusive scan
//          auto is_col_o_match_d = PetscIntKokkosView("is_col_o_match_d", cols_ao+1);
//          Kokkos::deep_copy(exec, is_col_o_match_d, 0);

//          // Start cmap scatter only after finishing x scatter on the same Mvctx.
//          // Ensure send/receive buffers are stable before Begin.
//          Kokkos::fence();         
//          PetscCallVoid(VecScatterBegin(mat_mpi->Mvctx, cmap_vec, lcmap_vec, INSERT_VALUES, SCATTER_FORWARD));
//          // cmap scatter completed: lcmap_vec is now safe to read.
//          PetscCallVoid(VecScatterEnd(mat_mpi->Mvctx, cmap_vec, lcmap_vec, INSERT_VALUES, SCATTER_FORWARD));         

//          //if (cols_ao > 0)
//          //{
//             ConstPetscScalarKokkosView lvec_scalar_d;
//             PetscCallVoid(VecGetKokkosView(x_leaf_vec, &lvec_scalar_d));

//             Kokkos::parallel_reduce("FindMatches", Kokkos::RangePolicy<>(exec, 0, cols_ao),
//                KOKKOS_LAMBDA(const PetscInt i, PetscInt& thread_sum) {
//                   // This is the scattered x for all of the non-local columns in the input mat
//                   // It's not -1.0 if that column is present on another rank
//                   if (lvec_scalar_d(i) > -1.0) {
//                      thread_sum++;
//                      is_col_o_match_d(i) = 1; // Mark this as a match
//                   }
//                },
//                Kokkos::Sum<PetscInt>(col_ao_output)
//             );

//             PetscCallVoid(VecRestoreKokkosView(x_leaf_vec, &lvec_scalar_d));
//          //}

//          std::cerr << "four " << std::endl;


//          // Need to do an exclusive scan on is_col_o_match_d to get the new local indices
//          // Have to remember to go up to cols_ao+1
//          Kokkos::parallel_scan(Kokkos::RangePolicy<>(exec, 0, cols_ao+1), KOKKOS_LAMBDA(const PetscInt i, PetscInt& partial_sum, const bool is_final) {
//                const int input_value = is_col_o_match_d(i);
//                if (is_final) {
//                   is_col_o_match_d(i) = partial_sum;  // Write exclusive prefix
//                }
//                partial_sum += input_value;  // Update running total
//          });

//          // ~~~~~~~~~~~~
//          // DIAGNOSTIC (Step 1 of plan): the parallel_reduce above produced
//          // col_ao_output on the host while the scan produced the per-index
//          // prefix sum on device. They must agree on the total count; if they
//          // don't, the size of is_col_o_d / garray_output_d below is wrong and
//          // the subsequent scatter kernel will write out of bounds.
//          // ~~~~~~~~~~~~
//          {
//             PetscInt scan_total_h = 0;
//             auto tail_sv = Kokkos::subview(is_col_o_match_d, cols_ao);
//             Kokkos::View<PetscInt, Kokkos::HostSpace> tail_h("PFLARE_DBG_scan_tail");
//             Kokkos::deep_copy(exec, tail_h, tail_sv);
//             Kokkos::fence();
//             scan_total_h = tail_h();
//             PetscCheckAbort(scan_total_h == col_ao_output, MPI_COMM_MATRIX,
//                PETSC_ERR_PLIB,
//                "MatCreateSubMatrix_kokkos_view: parallel_reduce count (%" PetscInt_FMT ") disagrees with scan total (%" PetscInt_FMT "), cols_ao=%" PetscInt_FMT,
//                col_ao_output, scan_total_h, cols_ao);
//             PetscCheckAbort(col_ao_output >= 0 && col_ao_output <= cols_ao, MPI_COMM_MATRIX,
//                PETSC_ERR_PLIB,
//                "MatCreateSubMatrix_kokkos_view: col_ao_output=%" PetscInt_FMT " outside [0,%" PetscInt_FMT "]",
//                col_ao_output, cols_ao);
//          }

//          // Local indices into input garray of the columns we want to keep
//          // but remember this doesn't mean garray_output = garray_input(is_col_o_d)
//          // as the of columns we have in the output has changed, ie we need
//          // the cmap_d given it has isstart
//          is_col_o_d = PetscIntKokkosView("is_col_o_d", col_ao_output);
//          garray_output_d = PetscIntKokkosView("garray_output_d", col_ao_output);

//          // Loop over all the cols in the input matrix
//          //{
//             ConstPetscScalarKokkosView lcmap_scalar_d;
//             PetscCallVoid(VecGetKokkosView(lcmap_vec, &lcmap_scalar_d));

//             Kokkos::parallel_for(
//                Kokkos::RangePolicy<>(exec, 0, cols_ao), KOKKOS_LAMBDA(PetscInt i) {

//                   // We can tell if is_col_o_match_d had 1 in it in this position by comparing the result
//                   // of the exclusive scan for this index and the next one
//                   if (is_col_o_match_d(i+1) > is_col_o_match_d(i))
//                   {
//                      is_col_o_d(is_col_o_match_d(i)) = i;
//                      garray_output_d(is_col_o_match_d(i)) = (PetscInt)lcmap_scalar_d(i);
//                   }
//             });
//             // Fence so the parallel for finishes
//             Kokkos::fence();

//             PetscCallVoid(VecRestoreKokkosView(lcmap_vec, &lcmap_scalar_d));
//          //}

//                   std::cerr << "five " << std::endl;


//          // Cleanup Vecs
//          PetscCallVoid(VecDestroy(&x_vec));
//          PetscCallVoid(VecDestroy(&x_leaf_vec));
//          PetscCallVoid(VecDestroy(&cmap_vec));
//          PetscCallVoid(VecDestroy(&lcmap_vec));
//       }
//       // If we're reusing we have the iscol_o associated with the output_mat
//       else
//       {
//          // Get the iscol_o from the output_mat
//          IS iscol_o;
//          /* Retrieve isrow_d, iscol_d and iscol_o from output */
//          PetscCallVoid(PetscObjectQuery((PetscObject)(*output_mat), "iscol_o", (PetscObject *)&iscol_o));
//          //PetscCheck(iscol_o, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "iscol_o passed in was not used before, cannot reuse");

//          const PetscInt *iscol_o_indices_ptr;
//          PetscCallVoid(ISGetIndices(iscol_o, &iscol_o_indices_ptr));

//          PetscInt local_cols_iscol_o;
//          PetscCallVoid(ISGetLocalSize(iscol_o, &local_cols_iscol_o));

//          // Copy the iscol_o to the device
//          auto iscol_o_view_h = PetscIntConstKokkosViewHost(iscol_o_indices_ptr, local_cols_iscol_o);    
//          is_col_o_d = PetscIntKokkosView("is_col_o_d", local_cols_iscol_o);
//          Kokkos::deep_copy(exec, is_col_o_d, iscol_o_view_h);
//          // Log copy with petsc
//          bytes = iscol_o_view_h.extent(0) * sizeof(PetscInt);
//          PetscCallVoid(PetscLogCpuToGpu(bytes));
//          Kokkos::fence();

//          PetscCallVoid(ISRestoreIndices(iscol_o, &iscol_o_indices_ptr));
//       }

//       // We can now create the off-diagonal component
//       Kokkos::fence();
//       MatCreateSubMatrix_Seq_kokkos(&mat_nonlocal, is_row_d_d, is_col_o_d, reuse_int, &output_mat_nonlocal);

//       // If it's our first time through we have to create our output matrix
//       if (!reuse_int)
//       {
//                   std::cerr << "six " << std::endl;

//          // Copy the garray output to the host
//          PetscInt *garray_host = NULL; 
//          PetscCallVoid(PetscMalloc1(garray_output_d.extent(0), &garray_host));
//          PetscIntKokkosViewHost colmap_output_h = PetscIntKokkosViewHost(garray_host, garray_output_d.extent(0));
//          // Copy the garray output to the host
//          Kokkos::deep_copy(exec, colmap_output_h, garray_output_d);
//          Kokkos::fence();
//          bytes = colmap_output_h.extent(0) * sizeof(PetscInt);
//          PetscCallVoid(PetscLogGpuToCpu(bytes));

//                   std::cerr << "seven " << std::endl;

         
//          // We can now create our MPI matrix
//          PetscCallVoid(MatCreateMPIAIJWithSeqAIJ(MPI_COMM_MATRIX, global_rows_row, global_cols_col, output_mat_local, output_mat_nonlocal, garray_host, output_mat));

//                   std::cerr << "eight " << std::endl;

//          // ~~~~~~~~~~~~~~
//          // If this is the first time through, we need to store the iscol_o in the output_mat
//          // We don't store the is_row_d_d or is_col_d_d like the host version does as they're super cheap to rebuild
//          // ~~~~~~~~~~~~~~
//          // Copy the is_col_o_d to the host
//          PetscInt *is_col_o_host = NULL; 
//          PetscCallVoid(PetscMalloc1(is_col_o_d.extent(0), &is_col_o_host));
//          PetscIntKokkosViewHost is_col_o_h = PetscIntKokkosViewHost(is_col_o_host, is_col_o_d.extent(0));
//          // Copy the is_col_o_d output to the host
//          Kokkos::deep_copy(exec, is_col_o_h, is_col_o_d);
//          Kokkos::fence();
//          bytes = is_col_o_h.extent(0) * sizeof(PetscInt);
//          PetscCallVoid(PetscLogGpuToCpu(bytes));
//          // Now create an IS
//          IS iscol_o;
//          PetscCallVoid(ISCreateGeneral(PETSC_COMM_SELF, is_col_o_h.extent(0), is_col_o_host, PETSC_COPY_VALUES, &iscol_o));      
//          // Register it with the output_mat
//          PetscCallVoid(PetscObjectCompose((PetscObject)(*output_mat), "iscol_o", (PetscObject)iscol_o));
//          // The ref counter is incremented by the compose
//          //PetscCallVoid(ISDestroy(&iscol_o));

//          std::cerr << "nine " << std::endl;

//       }
//    }
//    else
//    {
//       *output_mat = output_mat_local;
//    }

//    return;
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
   //PflareKokkosTrace _trace("MatCreateSubMatrix_kokkos");

   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt global_col_start, global_col_end_plus_one;   
   PetscCallVoid(MatGetOwnershipRange(*input_mat, &global_row_start, &global_row_end_plus_one));  
   PetscCallVoid(MatGetOwnershipRangeColumn(*input_mat, &global_col_start, &global_col_end_plus_one)); 
   PetscInt global_rows_row, global_cols_col;
   PetscCallVoid(ISGetSize(*is_row, &global_rows_row));
   PetscCallVoid(ISGetSize(*is_col, &global_cols_col));   
   
   // Equivalent to calling MatSeqAIJKokkosSyncDevice which is petsc intern
   mat_sync(input_mat);   
   
   PetscIntKokkosView is_row_d_d, is_col_d_d;
   const int level_idx = our_level - 1;
   auto exec = PetscGetKokkosExecutionSpace();

   // // If we want the input is_row and is_col to be used
   // if (our_level == -1)
   // {
   //    // Get pointers to the indices on the host
   //    const PetscInt *is_row_indices_ptr, *is_col_indices_ptr;
   //    PetscCallVoid(ISGetIndices(*is_row, &is_row_indices_ptr));   
   //    PetscCallVoid(ISGetIndices(*is_col, &is_col_indices_ptr)); 

   //    PetscInt local_rows_row, local_cols_col;
   //    PetscCallVoid(ISGetLocalSize(*is_row, &local_rows_row));   
   //    PetscCallVoid(ISGetLocalSize(*is_col, &local_cols_col));

   //    // Create a host view of the existing indices
   //    auto is_row_view_h = PetscIntConstKokkosViewHost(is_row_indices_ptr, local_rows_row);    
   //    is_row_d_d = PetscIntKokkosView("is_row_d_d", local_rows_row);   
   //    auto is_col_view_h = PetscIntConstKokkosViewHost(is_col_indices_ptr, local_cols_col);    
   //    is_col_d_d = PetscIntKokkosView("is_col_d_d", local_cols_col);      
   //    // Copy indices to the device
   //    Kokkos::deep_copy(exec, is_row_d_d, is_row_view_h);
   //    Kokkos::deep_copy(exec, is_col_d_d, is_col_view_h);
   //    // The source pointers come from ISGetIndices; ensure async copies complete
   //    // before restoring those host buffers.
   //    Kokkos::fence();
   //    // Log copy with petsc
   //    size_t bytes = is_row_view_h.extent(0) * sizeof(PetscInt);
   //    PetscCallVoid(PetscLogCpuToGpu(bytes));        
   //    bytes = is_col_view_h.extent(0) * sizeof(PetscInt);
   //    PetscCallVoid(PetscLogCpuToGpu(bytes));  

   //    PetscCallVoid(ISRestoreIndices(*is_row, &is_row_indices_ptr));   
   //    PetscCallVoid(ISRestoreIndices(*is_col, &is_col_indices_ptr));   

   //    // ~~~~~~~~~~~~
   //    // Rewrite to local indices
   //    // ~~~~~~~~~~~~     
   //    Kokkos::parallel_for(
   //       Kokkos::RangePolicy<>(exec, 0, is_row_d_d.extent(0)), KOKKOS_LAMBDA(PetscInt i) {      

   //          is_row_d_d(i) -= global_row_start; // Make local
   //    });

   //    Kokkos::parallel_for(
   //       Kokkos::RangePolicy<>(exec, 0, is_col_d_d.extent(0)), KOKKOS_LAMBDA(PetscInt i) {

   //          is_col_d_d(i) -= global_col_start; // Make local
   //    });
   //    Kokkos::fence(); 
   // }
   // // Instead if we tell the routine that the is_row and is_col are fine/coarse local indices
   // // that already are on the device
   // else
   // {
   //    if (is_row_fine_int)
   //    {
   //       is_row_d_d = *IS_fine_views_local[level_idx];
   //    }
   //    else
   //    {
   //       is_row_d_d = *IS_coarse_views_local[level_idx];
   //    }       
   //    if (is_col_fine_int)
   //    {
   //       is_col_d_d = *IS_fine_views_local[level_idx];
   //    }
   //    else
   //    {
   //       is_col_d_d = *IS_coarse_views_local[level_idx];
   //    }        
   // }  

   // ### path 2
   PetscCallVoid(MatCreateSubMatrix(*input_mat, *is_row, *is_col, MAT_INITIAL_MATRIX, output_mat));
   // return;   

   // ### path 1
   // MatCreateSubMatrix_kokkos_view(input_mat, is_row_d_d, global_rows_row, is_col_d_d, global_cols_col, reuse_int, output_mat, is_row, is_col);

   return;
}
