// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>
#include <vector>
#include <KokkosKernels_Handle.hpp>
#include <KokkosSparse_spgemm.hpp>
#include <petscpkg_version.h>

//------------------------------------------------------------------------------------------------------------------------

// Remap each entry in j_d from a global index to its local index via binary search into garray_d.
// garray_d must be a sorted array of unique global indices.
// Fences internally.
static void remap_j_to_local_device(PetscIntKokkosView j_d, PetscIntKokkosView garray_d, PetscInt col_ao_output)
{
   auto exec = PetscGetKokkosExecutionSpace();

   if (j_d.extent(0) == 0) return;
   // Binary search sorted garray to find our local index
   // Originally used Kokkos::UnorderedMap here but it only handles up to uint32_t
   // entries
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, j_d.extent(0)), KOKKOS_LAMBDA(const PetscInt i) {
         j_d(i) = binary_search_sorted(garray_d, col_ao_output, j_d(i));
   });
   // Ensure the rewrite is finished before we return
   Kokkos::fence();
}

//------------------------------------------------------------------------------------------------------------------------

// Build garray on device from global indices in j_nonlocal_d and remap j_nonlocal_d to local in-place.
// garray_d (out) is a device view of the sorted unique global column indices (size col_ao_output).
// Fences internally.
static void rewrite_j_global_to_local_device(PetscInt colmap_max_size, PetscInt &col_ao_output, PetscIntKokkosView j_nonlocal_d, PetscIntKokkosView &garray_d)
{
   auto exec = PetscGetKokkosExecutionSpace();

   // Need to preallocate to the max size
   PetscIntKokkosView colmap_output_d("colmap_output_d", colmap_max_size);
   col_ao_output = 0;

   // Take a copy of j and sort it and then build garray
   if (j_nonlocal_d.extent(0) > 0)
   {
      ptrdiff_t count_ptr_arith = -1;
      // Scoped so we don't keep the sorted copy of j around very long
      {
         PetscIntKokkosView j_nonlocal_d_sorted("j_nonlocal_d_sorted", j_nonlocal_d.extent(0));
         Kokkos::deep_copy(exec, j_nonlocal_d_sorted, j_nonlocal_d);
         Kokkos::sort(exec, j_nonlocal_d_sorted);

         // Unique copy returns a copy of sorted j_nonlocal_d_sorted in order, but with all the duplicate entries removed
         auto unique_end_it = Kokkos::Experimental::unique_copy(exec, j_nonlocal_d_sorted, colmap_output_d);
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
   PetscIntKokkosView garray_d;

   // This fences internally
   rewrite_j_global_to_local_device(colmap_max_size, col_ao_output, j_nonlocal_d, garray_d);

   // Always allocate host space for the output garray (that stays in scope), even
   // when zero-size, as depending on the compiler this may return a non-null pointer
   PetscCallVoid(PetscMalloc1(col_ao_output, garray_host));
   if (col_ao_output > 0)
   {
      PetscIntKokkosViewHost colmap_output_h = PetscIntKokkosViewHost(*garray_host, col_ao_output);
      // Device to host so don't need to specify exec space
      Kokkos::deep_copy(colmap_output_h, garray_d);
      // Log copy with petsc
      size_t bytes = col_ao_output * sizeof(PetscInt);
      PetscCallVoid(PetscLogGpuToCpu(bytes));
   }
}

//------------------------------------------------------------------------------------------------------------------------

// Row kernels for remove_small_from_sparse_kokkos
// Each team handles team_size rows, one thread per row, with vector lanes over the
// entries in that row
struct DropSmallRowsFunctor {
   struct CountTag {};
   struct FillTag {};

   PetscInt nrows = 0;
   PetscInt global_row_start = 0, global_col_start = 0, global_col_end_plus_one = 0;
   PetscReal tol = 0.0;
   int relative_max_row_tolerance_int = 0, lump_int = 0, allow_drop_diagonal_int = 0, allow_diag_strength_int = 0;
   bool mpi = false;

   // Input local and nonlocal blocks, colmap_d gives the global column of a nonlocal column
   const PetscInt *l_i = nullptr, *l_j = nullptr, *nl_i = nullptr, *nl_j = nullptr;
   Kokkos::View<const PetscScalar *> l_a, nl_a;
   PetscIntKokkosView colmap_d;

   // Written by count: the tolerance for each row, whether the diagonal is in the row
   // (in the block we expect it in), and the number of kept entries in each row
   // in counts_l[row + 1] and counts_nl[row + 1]
   PetscScalarKokkosView rel_row_tol_d;
   boolKokkosView existing_diag_d;
   PetscInt *counts_l = nullptr, *counts_nl = nullptr;

   // Fill: start of each output row, and the output j and values
   // The nonlocal j are written as global columns
   const PetscInt *out_l_i = nullptr, *out_nl_i = nullptr;
   PetscInt *out_l_j = nullptr, *out_nl_j = nullptr;
   PetscScalar *out_l_a = nullptr, *out_nl_a = nullptr;

   KOKKOS_INLINE_FUNCTION bool is_diag_local(const PetscInt k, const PetscInt row_global) const
   {
      return l_j[k] + global_col_start == row_global;
   }
   KOKKOS_INLINE_FUNCTION bool is_diag_nonlocal(const PetscInt k, const PetscInt row_global) const
   {
      return colmap_d(nl_j[k]) == row_global;
   }
   // Do we keep this entry
   KOKKOS_INLINE_FUNCTION bool keep(const PetscScalar val, const PetscScalar row_tol, const bool is_diagonal) const
   {
      if (Kokkos::abs(val) >= row_tol) {
         // If this is the diagonal and we're dropping all diagonals don't keep it
         return !(allow_drop_diagonal_int == -1 && is_diagonal);
      }
      // Or if it's small but it's the diagonal and we're keeping diagonals
      return allow_drop_diagonal_int == 0 && is_diagonal;
   }

   KOKKOS_INLINE_FUNCTION void operator()(const CountTag &, const KokkosTeamMemberType &t) const
   {
      const PetscInt row_start = t.league_rank() * t.team_size();
      const PetscInt row_end = row_start + t.team_size() < nrows ? row_start + t.team_size() : nrows;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, row_start, row_end), [&](const PetscInt i) {
         const PetscInt row_global = i + global_row_start;
         const PetscInt l_start = l_i[i], l_len = l_i[i + 1] - l_start;
         const PetscInt nl_start = mpi ? nl_i[i] : 0, nl_len = mpi ? nl_i[i + 1] - nl_start : 0;

         // Compute the tolerance for this row
         PetscScalar row_tol = tol;
         if (relative_max_row_tolerance_int) {
            // Measured relative to |a_ii|, or to the biggest |a_ij| in the row
            // (not including the diagonal if relative_max_row_tolerance_int == -1)
            const bool diag_only = allow_diag_strength_int;
            const bool not_include_diag = relative_max_row_tolerance_int == -1;
            // The Max reducer starts from the lowest value, so an empty range (or one where
            // every entry is excluded as -1) gives a negative max
            PetscScalar max_l, max_nl = Kokkos::reduction_identity<PetscScalar>::max();
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, l_len),
               [&](const PetscInt k, PetscScalar &thread_max) {
                  const bool is_diagonal = is_diag_local(l_start + k, row_global);
                  PetscScalar val = Kokkos::abs(l_a(l_start + k));
                  if ((diag_only && !is_diagonal) || (!diag_only && not_include_diag && is_diagonal)) val = -1.0;
                  if (val > thread_max) thread_max = val;
               }, Kokkos::Max<PetscScalar>(max_l));
            if (mpi) {
               // The diagonal can live in the off-diagonal block if we're rectangular
               Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, nl_len),
                  [&](const PetscInt k, PetscScalar &thread_max) {
                     const bool is_diagonal = is_diag_nonlocal(nl_start + k, row_global);
                     PetscScalar val = Kokkos::abs(nl_a(nl_start + k));
                     if ((diag_only && !is_diagonal) || (!diag_only && not_include_diag && is_diagonal)) val = -1.0;
                     if (val > thread_max) thread_max = val;
                  }, Kokkos::Max<PetscScalar>(max_nl));
            }
            const PetscScalar max_val = max_nl > max_l ? max_nl : max_l;
            if (diag_only) {
               // If the diagonal is missing, avoid over-dropping by using zero threshold
               row_tol = max_val >= 0.0 ? tol * max_val : 0.0;
            } else {
               row_tol = tol * max_val;
            }
         }

         // For this row, would we expect the diagonal to be in the local block or in the nonlocal?
         // Trivially true in the local block for square matrices
         const bool expect_local_diagonal = row_global >= global_col_start && row_global < global_col_end_plus_one;

         // Count the kept entries and whether we have a diagonal
         ReduceData result_l, result_nl;
         Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, l_len),
            [&](const PetscInt k, ReduceData &thread_data) {
               const bool is_diagonal = is_diag_local(l_start + k, row_global);
               if (is_diagonal) thread_data.found_diagonal = true;
               if (keep(l_a(l_start + k), row_tol, is_diagonal)) thread_data.count++;
            }, result_l);
         if (mpi) {
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, nl_len),
               [&](const PetscInt k, ReduceData &thread_data) {
                  const bool is_diagonal = is_diag_nonlocal(nl_start + k, row_global);
                  if (is_diagonal) thread_data.found_diagonal = true;
                  if (keep(nl_a(nl_start + k), row_tol, is_diagonal)) thread_data.count++;
               }, result_nl);
         }

         // The diagonal can only be in the block we expect it in
         const bool found_diagonal = expect_local_diagonal ? result_l.found_diagonal : result_nl.found_diagonal;
         // If we're lumping but there was no diagonal in this row, we add one to the
         // block it belongs in
         if (lump_int && !found_diagonal) {
            if (expect_local_diagonal) {
               result_l.count++;
            } else {
               result_nl.count++;
            }
         }

         Kokkos::single(Kokkos::PerThread(t), [&]() {
            rel_row_tol_d(i) = row_tol;
            existing_diag_d(i) = found_diagonal;
            counts_l[i + 1] = result_l.count;
            if (mpi) counts_nl[i + 1] = result_nl.count;
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
         const PetscInt row_global = i + global_row_start;
         const PetscInt l_start = l_i[i], l_len = l_i[i + 1] - l_start;
         const PetscInt nl_start = mpi ? nl_i[i] : 0, nl_len = mpi ? nl_i[i + 1] - nl_start : 0;
         const PetscScalar row_tol = rel_row_tol_d(i);
         const bool expect_local_diagonal = row_global >= global_col_start && row_global < global_col_end_plus_one;

         // The lumped value is the diagonal plus all the dropped off-diagonal entries
         PetscScalar lump_l = 0.0, lump_nl = 0.0;
         if (lump_int) {
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, l_len),
               [&](const PetscInt k, PetscScalar &thread_sum) {
                  const PetscScalar val = l_a(l_start + k);
                  const bool is_diagonal = is_diag_local(l_start + k, row_global);
                  if (is_diagonal || !keep(val, row_tol, is_diagonal)) thread_sum += val;
               }, Kokkos::Sum<PetscScalar>(lump_l));
            if (mpi) {
               Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, nl_len),
                  [&](const PetscInt k, PetscScalar &thread_sum) {
                     const PetscScalar val = nl_a(nl_start + k);
                     const bool is_diagonal = is_diag_nonlocal(nl_start + k, row_global);
                     if (is_diagonal || !keep(val, row_tol, is_diagonal)) thread_sum += val;
                  }, Kokkos::Sum<PetscScalar>(lump_nl));
            }
         }
         const PetscScalar lump_val = lump_l + lump_nl;

         // Write the kept entries, with the lumped value replacing the diagonal
         const PetscInt out_l_offset = out_l_i[i];
         Kokkos::parallel_scan(Kokkos::ThreadVectorRange(t, l_len),
            [&](const PetscInt k, PetscInt &partial_sum, const bool is_final) {
               const PetscScalar val = l_a(l_start + k);
               const bool is_diagonal = is_diag_local(l_start + k, row_global);
               if (keep(val, row_tol, is_diagonal)) {
                  if (is_final) {
                     out_l_j[out_l_offset + partial_sum] = l_j[l_start + k];
                     out_l_a[out_l_offset + partial_sum] = (lump_int && is_diagonal) ? lump_val : val;
                  }
                  partial_sum++;
               }
            });
         if (mpi) {
            const PetscInt out_nl_offset = out_nl_i[i];
            Kokkos::parallel_scan(Kokkos::ThreadVectorRange(t, nl_len),
               [&](const PetscInt k, PetscInt &partial_sum, const bool is_final) {
                  const PetscScalar val = nl_a(nl_start + k);
                  const bool is_diagonal = is_diag_nonlocal(nl_start + k, row_global);
                  if (keep(val, row_tol, is_diagonal)) {
                     if (is_final) {
                        out_nl_j[out_nl_offset + partial_sum] = colmap_d(nl_j[nl_start + k]);
                        out_nl_a[out_nl_offset + partial_sum] = (lump_int && is_diagonal) ? lump_val : val;
                     }
                     partial_sum++;
                  }
               });
         }

         // If we're lumping and this row doesn't have a diagonal, add it to the end of the row
         // in the block it belongs in and we sort after
         if (lump_int && !existing_diag_d(i)) {
            Kokkos::single(Kokkos::PerThread(t), [&]() {
               if (expect_local_diagonal) {
                  // Has to be the local column index - the row and column ownership
                  // starts can differ so we can't just use i
                  out_l_j[out_l_i[i + 1] - 1] = row_global - global_col_start;
                  out_l_a[out_l_i[i + 1] - 1] = lump_val;
               } else {
                  // Has to be global column for now
                  out_nl_j[out_nl_i[i + 1] - 1] = row_global;
                  out_nl_a[out_nl_i[i + 1] - 1] = lump_val;
               }
            });
         }
      });
   }
};

// Launches one of the DropSmallRowsFunctor kernels over all the rows
template <class Tag>
static void launch_drop_small_rows(const DropSmallRowsFunctor &functor, const int vector_length)
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

// A matrix's blocks as device data, the input of drop_small_rows_core
struct DropInput {
   bool mpi = false;
   MPI_Comm comm = MPI_COMM_NULL;
   PetscInt local_rows = 0, local_cols = 0, global_rows = 0, global_cols = 0;
   PetscInt row_start = 0, col_start = 0, col_end = 0;
   // The local block and the nonlocal block, whose columns index colmap_d (global columns)
   const PetscInt *l_i = nullptr, *l_j = nullptr, *nl_i = nullptr, *nl_j = nullptr;
   Kokkos::View<const PetscScalar *> l_a, nl_a;
   PetscIntKokkosView colmap_d;
   PetscInt cols_ao = 0;
   PetscInt nnz_l = 0, nnz_nl = 0;
};

// Drop according to a tolerance with kokkos, from a matrix's blocks on the device
// If allow_alias_int is set and nothing would be dropped (or added) on any rank, it returns true
// and doesn't create output_mat, as the output would be identical to the input; otherwise it
// returns false and creates output_mat
static bool drop_small_rows_core(const DropInput &in, const PetscReal tol, Mat *output_mat, \
                  const int relative_max_row_tolerance_int, const int lump_int, \
                  const int allow_drop_diagonal_int, const int allow_diag_strength_int, \
                  const int allow_alias_int)
{
   const bool mpi = in.mpi;
   MPI_Comm MPI_COMM_MATRIX = in.comm;
   const PetscInt local_rows = in.local_rows, local_cols = in.local_cols;
   const PetscInt global_rows = in.global_rows, global_cols = in.global_cols;
   const PetscInt cols_ao = in.cols_ao;
   Mat output_mat_local, output_mat_nonlocal;
   auto exec = PetscGetKokkosExecutionSpace();

   DropSmallRowsFunctor functor;
   functor.nrows = local_rows;
   functor.global_row_start = in.row_start;
   functor.global_col_start = in.col_start;
   functor.global_col_end_plus_one = in.col_end;
   functor.tol = tol;
   functor.relative_max_row_tolerance_int = relative_max_row_tolerance_int;
   functor.lump_int = lump_int;
   functor.allow_drop_diagonal_int = allow_drop_diagonal_int;
   functor.allow_diag_strength_int = allow_diag_strength_int;
   functor.mpi = mpi;
   functor.l_i = in.l_i;
   functor.l_j = in.l_j;
   functor.l_a = in.l_a;
   if (mpi)
   {
      functor.nl_i = in.nl_i;
      functor.nl_j = in.nl_j;
      functor.nl_a = in.nl_a;
      functor.colmap_d = in.colmap_d;
   }

   // The vector length comes from the average row length (local plus nonlocal), so short
   // rows don't leave most of the threads idle and long rows still get some parallelism
   // The host backends run the vector lanes serially, so there it makes no difference
   const double avg_row_nnz = local_rows > 0 ? static_cast<double>(in.nnz_l + (mpi ? in.nnz_nl : 0)) / local_rows : 0.0;
   int vector_length = 1;
   while (2 * vector_length <= avg_row_nnz && vector_length < 32) vector_length *= 2;
   vector_length = std::min(vector_length, Kokkos::TeamPolicy<>::vector_length_max());

   // ~~~~~~~~~~~~
   // Count the entries we keep in each row
   // ~~~~~~~~~~~~
   // Zero initialised on exec, the counts go into entry row + 1
   Kokkos::View<PetscInt *> i_local_d(Kokkos::view_alloc(exec, "i_local_d"), local_rows+1);
   Kokkos::View<PetscInt *> i_nonlocal_d;
   if (mpi) i_nonlocal_d = Kokkos::View<PetscInt *>(Kokkos::view_alloc(exec, "i_nonlocal_d"), local_rows+1);
   // Every entry is written by the count
   functor.rel_row_tol_d = PetscScalarKokkosView(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "rel_row_tol_d"), local_rows);
   functor.existing_diag_d = boolKokkosView(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "existing_diag_d"), local_rows);
   functor.counts_l = i_local_d.data();
   if (mpi) functor.counts_nl = i_nonlocal_d.data();
   launch_drop_small_rows<DropSmallRowsFunctor::CountTag>(functor, vector_length);

   // Inclusive scans of the row counts give the row pointers
   // Each index only reads and writes its own entry, so this is safe in place
   PetscInt nnzs_match_local = 0, nnzs_match_nonlocal = 0;
   Kokkos::parallel_scan(
      Kokkos::RangePolicy<>(exec, 0, local_rows),
      KOKKOS_LAMBDA(const PetscInt i, PetscInt &update, const bool final) {
         update += i_local_d(i + 1);
         if (final) i_local_d(i + 1) = update;
      },
      nnzs_match_local);
   if (mpi)
   {
      Kokkos::parallel_scan(
         Kokkos::RangePolicy<>(exec, 0, local_rows),
         KOKKOS_LAMBDA(const PetscInt i, PetscInt &update, const bool final) {
            update += i_nonlocal_d(i + 1);
            if (final) i_nonlocal_d(i + 1) = update;
         },
         nnzs_match_nonlocal);
   }

   // Did we add a diagonal to any row for lumping
   bool added_any_diagonal = false;
   if (lump_int)
   {
      boolKokkosView existing_diag_d = functor.existing_diag_d;
      Kokkos::parallel_reduce(
         Kokkos::RangePolicy<>(exec, 0, local_rows),
         KOKKOS_LAMBDA(const PetscInt i, bool& thread_result) {
            // If this row had a diagonal added, set the result to true
            if (!existing_diag_d(i)) thread_result = true;
         },
         Kokkos::LOr<bool>(added_any_diagonal)
      );
   }

   // ~~~~~~~~~~~~
   // If we're allowed, check whether the output would just be a copy of the input
   // ~~~~~~~~~~~~
   // If every entry is kept and no diagonal is added, the kept values are the input values and
   // a lumped diagonal is just the diagonal, so the output would be identical to the input
   // Every rank has to agree as the output is a parallel matrix
   if (allow_alias_int)
   {
      const PetscInt nnz_in_local = in.nnz_l;
      const PetscInt nnz_in_nonlocal = mpi ? in.nnz_nl : 0;
      int unchanged_local = !added_any_diagonal && nnzs_match_local == nnz_in_local && \
                  nnzs_match_nonlocal == nnz_in_nonlocal;
      int unchanged = 0;
      PetscCallMPIAbort(MPI_COMM_MATRIX, MPI_Allreduce(&unchanged_local, &unchanged, 1, MPI_INT, MPI_LAND, MPI_COMM_MATRIX));
      if (unchanged) return true;
   }

   // ~~~~~~~~~~~~
   // Fill the output i, j and values
   // ~~~~~~~~~~~~
   // Every entry is written by the fill
   Kokkos::View<PetscScalar *> a_local_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "a_local_d"), nnzs_match_local);
   Kokkos::View<PetscInt *> j_local_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "j_local_d"), nnzs_match_local);
   Kokkos::View<PetscScalar *> a_nonlocal_d;
   Kokkos::View<PetscInt *> j_nonlocal_d;
   if (mpi)
   {
      a_nonlocal_d = Kokkos::View<PetscScalar *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "a_nonlocal_d"), nnzs_match_nonlocal);
      j_nonlocal_d = Kokkos::View<PetscInt *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "j_nonlocal_d"), nnzs_match_nonlocal);
   }

   if (nnzs_match_local + nnzs_match_nonlocal > 0)
   {
      functor.out_l_i = i_local_d.data();
      functor.out_l_j = j_local_d.data();
      functor.out_l_a = a_local_d.data();
      if (mpi)
      {
         functor.out_nl_i = i_nonlocal_d.data();
         functor.out_nl_j = j_nonlocal_d.data();
         functor.out_nl_a = a_nonlocal_d.data();
      }
      launch_drop_small_rows<DropSmallRowsFunctor::FillTag>(functor, vector_length);
   }

   // Let's make sure everything on the device is finished
   exec.fence();

   // Convert j_nonlocal_d from global to local indices now, before any sort below.
   // All global indices (including any diagonals added in the fill above) are finalised.
   // garray_d holds the sorted unique global column indices on device.
   PetscIntKokkosView garray_d;
   PetscInt col_ao_output = 0;
   if (mpi)
   {
      // This fences internally
      rewrite_j_global_to_local_device(cols_ao, col_ao_output, j_nonlocal_d, garray_d);
   }

   // Now we may have to sort the column indices
   if (lump_int)
   {
      // If we did add a diagonal, it got added to the end of the columns on each row, so will have to sort
      // It also could have been added to either the local or nonlocal components given not square
      if (added_any_diagonal)
      {
         KokkosCsrMatrix csrmat_local = KokkosCsrMatrix("csrmat_local", local_rows, local_cols, a_local_d.extent(0), a_local_d, i_local_d, j_local_d);
         KokkosSparse::sort_crs_matrix(exec, csrmat_local);

         if (mpi)
         {
            // j_nonlocal_d now contains local indices; use col_ao_output as numCols
            KokkosCsrMatrix csrmat_nonlocal = KokkosCsrMatrix("csrmat_nonlocal", local_rows, col_ao_output, a_nonlocal_d.extent(0), a_nonlocal_d, i_nonlocal_d, j_nonlocal_d);
            KokkosSparse::sort_crs_matrix(exec, csrmat_nonlocal);
         }
      }
   }

   exec.fence();

   // We can create our local diagonal block matrix directly on the device
   PetscCallAbort(PETSC_COMM_SELF, MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, local_cols, i_local_d, j_local_d, a_local_d, &output_mat_local));

   // we also have to go and build the a, i, j for the non-local off-diagonal block
   if (mpi)
   {
      // Copy the device garray to the host
      PetscInt *garray_host = NULL;
      PetscCallAbort(PETSC_COMM_SELF, PetscMalloc1(col_ao_output, &garray_host));
      if (col_ao_output > 0)
      {
         PetscIntKokkosViewHost garray_h(garray_host, col_ao_output);
         // Device to host so don't need to specify exec space
         Kokkos::deep_copy(garray_h, garray_d);
         // Log copy with petsc
         size_t bytes = col_ao_output * sizeof(PetscInt);
         PetscCallAbort(PETSC_COMM_SELF, PetscLogGpuToCpu(bytes));
      }

      // We can create our nonlocal diagonal block matrix directly on the device
      PetscCallAbort(PETSC_COMM_SELF, MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, col_ao_output, i_nonlocal_d, j_nonlocal_d, a_nonlocal_d, &output_mat_nonlocal));

      // We can now create our MPI matrix
      PetscCallAbort(PETSC_COMM_SELF, MatCreateMPIAIJWithSeqAIJ(MPI_COMM_MATRIX, global_rows, global_cols, output_mat_local, output_mat_nonlocal, garray_host, output_mat));
   }
   // If in serial
   else
   {
      *output_mat = output_mat_local;
   }

   return false;
}

//------------------------------------------------------------------------------------------------------------------------

// Drop according to a tolerance but with kokkos - keeping everything on the device
// If allow_alias_int is set and nothing would be dropped (or added) on any rank, the output
// is the input itself with its reference count increased, rather than an identical copy
// Only set it if the caller doesn't modify or reuse either matrix in a way that needs them to be separate
PETSC_INTERN void remove_small_from_sparse_kokkos(Mat *input_mat, const PetscReal tol, Mat *output_mat, \
                  const int relative_max_row_tolerance_int, const int lump_int, \
                  const int allow_drop_diagonal_int, const int allow_diag_strength_int, \
                  const int allow_alias_int)
{
   MatType mat_type;
   PetscInt rows_ao, row_end, col_end;
   DropInput in;
   auto exec = PetscGetKokkosExecutionSpace();

   PetscCallVoid(MatGetType(*input_mat, &mat_type));
   // Are we in parallel?
   in.mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;
   Mat mat_local = NULL, mat_nonlocal = NULL;

   const PetscInt *colmap_input;
   if (in.mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*input_mat, &mat_local, &mat_nonlocal, &colmap_input));
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &in.cols_ao));

      // We also copy the input mat colmap over to the device as we need it
      PetscIntConstKokkosViewHost colmap_input_h(colmap_input, in.cols_ao);
      in.colmap_d = PetscIntKokkosView("colmap_input_d", in.cols_ao);
      Kokkos::deep_copy(exec, in.colmap_d, colmap_input_h);
      // Log copy with petsc
      size_t bytes = colmap_input_h.extent(0) * sizeof(PetscInt);
      PetscCallVoid(PetscLogCpuToGpu(bytes));
   }
   else
   {
      mat_local = *input_mat;
   }

   // Get the comm
   PetscCallVoid(PetscObjectGetComm((PetscObject)*input_mat, &in.comm));
   PetscCallVoid(MatGetLocalSize(*input_mat, &in.local_rows, &in.local_cols));
   PetscCallVoid(MatGetSize(*input_mat, &in.global_rows, &in.global_cols));
   // This returns the global index of the local portion of the matrix
   PetscCallVoid(MatGetOwnershipRange(*input_mat, &in.row_start, &row_end));
   PetscCallVoid(MatGetOwnershipRangeColumn(*input_mat, &in.col_start, &col_end));
   in.col_end = col_end;

   // ~~~~~~~~~~~~
   // Get pointers to the i,j on the device and Kokkos views to the values
   // ~~~~~~~~~~~~
   PetscMemType mtype;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &in.l_i, &in.l_j, NULL, &mtype));
   if (in.mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &in.nl_i, &in.nl_j, NULL, &mtype));
   PetscCallVoid(MatSeqAIJGetKokkosView(mat_local, &in.l_a));
   if (in.mpi) PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal, &in.nl_a));
   MatInfo info_local, info_nonlocal;
   PetscCallVoid(MatGetInfo(mat_local, MAT_LOCAL, &info_local));
   in.nnz_l = static_cast<PetscInt>(info_local.nz_used);
   if (in.mpi)
   {
      PetscCallVoid(MatGetInfo(mat_nonlocal, MAT_LOCAL, &info_nonlocal));
      in.nnz_nl = static_cast<PetscInt>(info_nonlocal.nz_used);
   }

   const bool unchanged = drop_small_rows_core(in, tol, output_mat, relative_max_row_tolerance_int, lump_int, \
                  allow_drop_diagonal_int, allow_diag_strength_int, allow_alias_int);

   PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local, &in.l_a));
   if (in.mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal, &in.nl_a));
   if (unchanged)
   {
      PetscCallVoid(PetscObjectReference((PetscObject)*input_mat));
      *output_mat = *input_mat;
   }
}


//------------------------------------------------------------------------------------------------------------------------

// Row kernels for mat_mult_one_entry_per_row_kokkos
// Each team handles team_size rows, one thread per row, with vector lanes over the
// entries in that row
struct OneEntryPerRowFunctor {
   struct RelabelTag {};
   struct FillTag {};

   PetscInt nrows = 0;
   bool mpi = false;

   // Relabel: the local and nonlocal blocks of A
   const PetscInt *l_i = nullptr, *l_j = nullptr, *nl_i = nullptr, *nl_j = nullptr;
   Kokkos::View<const PetscScalar *> l_a, nl_a;
   // The global column of P's entry in the row of P each local (l) and ghost (nl) column of A
   // refers to, and its value. A row of P with no entry gives the sentinel column and a zero
   PetscIntKokkosView target_l_d, target_nl_d;
   PetscScalarKokkosView weight_l_d, weight_nl_d;
   // Written by relabel: row i of A relabelled, starting at l_i[i] + nl_i[i]
   PetscInt *rel_j = nullptr;
   PetscScalar *rel_a = nullptr;

   // Fill: the sorted and merged rows, with the sentinel entries last in each row
   const PetscInt *m_i = nullptr, *m_j = nullptr;
   const PetscScalar *m_a = nullptr;
   // The local columns of the output (and P) are [col_start, col_end)
   PetscInt col_start = 0, col_end = 0;
   // How many entries in each merged row come before col_start
   const PetscInt *n_lower = nullptr;
   // Start of each output row, and the output j and values
   // The nonlocal j are written as global columns
   const PetscInt *out_l_i = nullptr, *out_nl_i = nullptr;
   PetscInt *out_l_j = nullptr, *out_nl_j = nullptr;
   PetscScalar *out_l_a = nullptr, *out_nl_a = nullptr;

   KOKKOS_INLINE_FUNCTION void operator()(const RelabelTag &, const KokkosTeamMemberType &t) const
   {
      const PetscInt row_start = t.league_rank() * t.team_size();
      const PetscInt row_end = row_start + t.team_size() < nrows ? row_start + t.team_size() : nrows;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, row_start, row_end), [&](const PetscInt i) {
         const PetscInt l_start = l_i[i], l_len = l_i[i + 1] - l_start;
         const PetscInt out_start = l_start + (mpi ? nl_i[i] : 0);
         Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, l_len), [&](const PetscInt k) {
            const PetscInt col = l_j[l_start + k];
            rel_j[out_start + k] = target_l_d(col);
            rel_a[out_start + k] = l_a(l_start + k) * weight_l_d(col);
         });
         if (mpi) {
            const PetscInt nl_start = nl_i[i], nl_len = nl_i[i + 1] - nl_start;
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, nl_len), [&](const PetscInt k) {
               const PetscInt col = nl_j[nl_start + k];
               rel_j[out_start + l_len + k] = target_nl_d(col);
               rel_a[out_start + l_len + k] = nl_a(nl_start + k) * weight_nl_d(col);
            });
         }
      });
   }

   KOKKOS_INLINE_FUNCTION void operator()(const FillTag &, const KokkosTeamMemberType &t) const
   {
      const PetscInt row_start = t.league_rank() * t.team_size();
      const PetscInt row_end = row_start + t.team_size() < nrows ? row_start + t.team_size() : nrows;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, row_start, row_end), [&](const PetscInt i) {
         // A merged row is sorted, so it is: nonlocal columns below col_start, the local
         // columns, nonlocal columns from col_end, then at most one sentinel entry
         const PetscInt lower = n_lower[i];
         const PetscInt diag = out_l_i[i + 1] - out_l_i[i];
         const PetscInt len = lower + diag + (mpi ? out_nl_i[i + 1] - out_nl_i[i] - lower : 0);
         const PetscInt m_start = m_i[i];
         Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, len), [&](const PetscInt k) {
            const PetscInt col = m_j[m_start + k];
            const PetscScalar val = m_a[m_start + k];
            if (k < lower) {
               out_nl_j[out_nl_i[i] + k] = col;
               out_nl_a[out_nl_i[i] + k] = val;
            }
            else if (k < lower + diag) {
               out_l_j[out_l_i[i] + k - lower] = col - col_start;
               out_l_a[out_l_i[i] + k - lower] = val;
            }
            else {
               out_nl_j[out_nl_i[i] + k - diag] = col;
               out_nl_a[out_nl_i[i] + k - diag] = val;
            }
         });
      });
   }
};

// Launches one of the OneEntryPerRowFunctor kernels over all the rows
template <class Tag>
static void launch_one_entry_per_row(const OneEntryPerRowFunctor &functor, const int vector_length)
{
   if (functor.nrows == 0) return;
   auto exec = PetscGetKokkosExecutionSpace();
   Kokkos::TeamPolicy<Tag> probe(exec, 1, 1, vector_length);
   int team_size = std::max(1, 256 / vector_length);
   team_size = std::min(team_size, probe.team_size_max(functor, Kokkos::ParallelForTag()));
   const int league_size = static_cast<int>((functor.nrows + team_size - 1) / team_size);
   Kokkos::parallel_for(Kokkos::TeamPolicy<Tag>(exec, league_size, team_size, vector_length), functor);
}

// The first position in the sorted j[start, end) that is not below key
KOKKOS_INLINE_FUNCTION PetscInt lower_bound_sorted(const PetscInt *j, PetscInt start, PetscInt end, const PetscInt key)
{
   while (start < end) {
      const PetscInt mid = start + (end - start) / 2;
      if (j[mid] < key) start = mid + 1;
      else end = mid;
   }
   return start;
}

//------------------------------------------------------------------------------------------------------------------------

// The blocks of A * P for a P with at most one entry per row, still as device views
// (built by build_one_point_ap_kokkos, not yet a PETSc matrix)
struct OnePointAP {
   bool mpi = false;
   MPI_Comm comm = MPI_COMM_NULL;
   PetscInt local_rows = 0, p_local_cols = 0, global_rows = 0, global_cols_p = 0;
   PetscInt col_start = 0, col_end = 0;
   // The local block, with local columns
   Kokkos::View<PetscInt *> i_l, j_l;
   Kokkos::View<PetscScalar *> a_l;
   PetscInt nnz_l = 0;
   // The nonlocal block, with local columns into garray_d (sorted global columns)
   Kokkos::View<PetscInt *> i_nl, j_nl;
   Kokkos::View<PetscScalar *> a_nl;
   PetscInt nnz_nl = 0;
   PetscIntKokkosView garray_d;
   PetscInt col_ao = 0;
   // The merged rows with global columns, sorted, before the split into blocks
   // The first (local + nonlocal) entries of each row are the entries of A * P, any
   // entries after them (columns of A whose row of P is empty) are not part of it
   Kokkos::View<PetscInt *> m_i, m_j;
   Kokkos::View<PetscScalar *> m_a;
};

// Builds the blocks of C = A * P as device views when every row of P has at most one entry
// (e.g., the one point classical prolongator), with kokkos - keeping everything on the device
// Each column of A is relabelled with the column of P's entry in that row of P and scaled
// by its value (columns whose row of P is empty are removed), then each row is sorted and
// entries in the same column are summed. This gives the same sparsity as MatMatMult(A, P)
// and the same values up to the order of the sums
// If any row of P has more than one entry on any rank, success_int is 0 and ap is not set;
// otherwise success_int is 1
static void build_one_point_ap_kokkos(Mat *input_mat, Mat *p_mat, int *success_int, OnePointAP &ap)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, global_rows, global_cols_p, p_local_rows, p_local_cols;
   PetscInt rows_ao, cols_ao = 0, p_rows_ao, p_cols_ao = 0;
   PetscInt col_start, col_end;
   MatType mat_type;

   PetscCallVoid(MatGetType(*input_mat, &mat_type));
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;
   Mat mat_local = NULL, mat_nonlocal = NULL, p_local = NULL, p_nonlocal = NULL;
   const PetscInt *colmap_input, *colmap_p;
   PetscSF sf = NULL;
   auto exec = PetscGetKokkosExecutionSpace();
   *success_int = 0;

   PetscCallVoid(PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX));
   PetscCallVoid(MatGetLocalSize(*input_mat, &local_rows, NULL));
   PetscCallVoid(MatGetSize(*input_mat, &global_rows, NULL));
   PetscCallVoid(MatGetLocalSize(*p_mat, &p_local_rows, &p_local_cols));
   PetscCallVoid(MatGetSize(*p_mat, NULL, &global_cols_p));
   PetscCallVoid(MatGetOwnershipRangeColumn(*p_mat, &col_start, &col_end));
   // Bigger than any column of the output, so entries whose row of P is empty sort last
   const PetscInt sentinel = global_cols_p;
   ap.mpi = mpi;
   ap.comm = MPI_COMM_MATRIX;
   ap.local_rows = local_rows;
   ap.p_local_cols = p_local_cols;
   ap.global_rows = global_rows;
   ap.global_cols_p = global_cols_p;
   ap.col_start = col_start;
   ap.col_end = col_end;

   // The global column and value of the entry in each local row of P
   PetscIntKokkosView target_l_d;
   PetscScalarKokkosView weight_l_d;
   // A matrix-free one point P gives these directly
   const bool p_shell = one_point_shell_targets_kokkos(*p_mat, sentinel, target_l_d, weight_l_d);

   PetscIntKokkosView colmap_p_d;
   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*input_mat, &mat_local, &mat_nonlocal, &colmap_input));
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao));
   }
   else
   {
      mat_local = *input_mat;
   }
   if (mpi && !p_shell)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*p_mat, &p_local, &p_nonlocal, &colmap_p));
      PetscCallVoid(MatGetSize(p_nonlocal, &p_rows_ao, &p_cols_ao));

      // P's colmap on the device, for the global column of a nonlocal entry
      colmap_p_d = PetscIntKokkosView(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "colmap_p_d"), p_cols_ao);
      Kokkos::deep_copy(exec, colmap_p_d, PetscIntConstKokkosViewHost(colmap_p, p_cols_ao));
      // Log copy with petsc
      size_t bytes = p_cols_ao * sizeof(PetscInt);
      PetscCallVoid(PetscLogCpuToGpu(bytes));
   }
   else if (!p_shell)
   {
      p_local = *p_mat;
   }
   // The halo of A, its leaves are A's nonlocal columns in order
   if (mpi) PetscCallVoid(MatGetMultPetscSF(*input_mat, &sf));

   // ~~~~~~~~~~~~
   // Get pointers to the i,j on the device and Kokkos views to the values
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   const PetscInt *p_local_i = nullptr, *p_local_j = nullptr, *p_nonlocal_i = nullptr, *p_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, NULL, &mtype));
   if (!p_shell)
   {
      PetscCallVoid(MatSeqAIJGetCSRAndMemType(p_local, &p_local_i, &p_local_j, NULL, &mtype));
      if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(p_nonlocal, &p_nonlocal_i, &p_nonlocal_j, NULL, &mtype));
   }
   Kokkos::View<const PetscScalar *> device_local_vals, device_nonlocal_vals, p_local_vals, p_nonlocal_vals;
   PetscCallVoid(MatSeqAIJGetKokkosView(mat_local, &device_local_vals));
   if (mpi) PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal, &device_nonlocal_vals));
   if (!p_shell)
   {
      PetscCallVoid(MatSeqAIJGetKokkosView(p_local, &p_local_vals));
      if (mpi) PetscCallVoid(MatSeqAIJGetKokkosView(p_nonlocal, &p_nonlocal_vals));
   }

   // ~~~~~~~~~~~~
   // The global column and value of the entry in each local row of P, and how many rows
   // have more than one entry
   // ~~~~~~~~~~~~
   PetscInt bad_rows = 0;
   if (!p_shell)
   {
      target_l_d = PetscIntKokkosView(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "target_l_d"), p_local_rows);
      weight_l_d = PetscScalarKokkosView(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "weight_l_d"), p_local_rows);
      Kokkos::parallel_reduce(
         Kokkos::RangePolicy<>(exec, 0, p_local_rows), KOKKOS_LAMBDA(const PetscInt k, PetscInt &update) {
            const PetscInt len_l = p_local_i[k + 1] - p_local_i[k];
            const PetscInt len_nl = mpi ? p_nonlocal_i[k + 1] - p_nonlocal_i[k] : 0;
            target_l_d(k) = sentinel;
            weight_l_d(k) = 0.0;
            if (len_l + len_nl > 1) {
               update++;
            }
            else if (len_l == 1) {
               target_l_d(k) = p_local_j[p_local_i[k]] + col_start;
               weight_l_d(k) = p_local_vals(p_local_i[k]);
            }
            else if (len_nl == 1) {
               target_l_d(k) = colmap_p_d(p_nonlocal_j[p_nonlocal_i[k]]);
               weight_l_d(k) = p_nonlocal_vals(p_nonlocal_i[k]);
            }
      }, bad_rows);
   }
   PetscInt bad_rows_global = bad_rows;
   if (mpi) PetscCallMPIAbort(MPI_COMM_MATRIX, MPI_Allreduce(&bad_rows, &bad_rows_global, 1, MPIU_INT, MPI_SUM, MPI_COMM_MATRIX));

   if (bad_rows_global == 0)
   {
      *success_int = 1;

      // ~~~~~~~~~~~~
      // Send the column and value to the ranks that have the row of P as a nonlocal column of A
      // ~~~~~~~~~~~~
      PetscIntKokkosView target_nl_d;
      PetscScalarKokkosView weight_nl_d;
      if (mpi)
      {
         target_nl_d = PetscIntKokkosView(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "target_nl_d"), cols_ao);
         weight_nl_d = PetscScalarKokkosView(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "weight_nl_d"), cols_ao);
         PetscCallVoid(sf_sync_before_begin_kokkos());
         PetscCallVoid(PetscSFBcastWithMemTypeBegin(sf, MPIU_INT,
                     PETSC_MEMTYPE_KOKKOS, target_l_d.data(),
                     PETSC_MEMTYPE_KOKKOS, target_nl_d.data(),
                     MPI_REPLACE));
         PetscCallVoid(PetscSFBcastWithMemTypeBegin(sf, MPIU_SCALAR,
                     PETSC_MEMTYPE_KOKKOS, weight_l_d.data(),
                     PETSC_MEMTYPE_KOKKOS, weight_nl_d.data(),
                     MPI_REPLACE));
      }

      // Start of each relabelled row, the local and nonlocal blocks of a row of A go together
      Kokkos::View<PetscInt *> rel_i_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "rel_i_d"), local_rows + 1);
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, local_rows + 1), KOKKOS_LAMBDA(const PetscInt i) {
            rel_i_d(i) = device_local_i[i] + (mpi ? device_nonlocal_i[i] : 0);
      });
      PetscInt nnzs_input = 0;
      Kokkos::deep_copy(exec, nnzs_input, Kokkos::subview(rel_i_d, local_rows));
      exec.fence();
      Kokkos::View<PetscInt *> rel_j_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "rel_j_d"), nnzs_input);
      Kokkos::View<PetscScalar *> rel_a_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "rel_a_d"), nnzs_input);

      if (mpi)
      {
         PetscCallVoid(PetscSFBcastEnd(sf, MPIU_INT, target_l_d.data(), target_nl_d.data(), MPI_REPLACE));
         PetscCallVoid(PetscSFBcastEnd(sf, MPIU_SCALAR, weight_l_d.data(), weight_nl_d.data(), MPI_REPLACE));
         PetscCallVoid(sf_sync_after_end_kokkos());
      }

      OneEntryPerRowFunctor functor;
      functor.nrows = local_rows;
      functor.mpi = mpi;
      functor.l_i = device_local_i;
      functor.l_j = device_local_j;
      functor.l_a = device_local_vals;
      functor.target_l_d = target_l_d;
      functor.weight_l_d = weight_l_d;
      if (mpi)
      {
         functor.nl_i = device_nonlocal_i;
         functor.nl_j = device_nonlocal_j;
         functor.nl_a = device_nonlocal_vals;
         functor.target_nl_d = target_nl_d;
         functor.weight_nl_d = weight_nl_d;
      }
      functor.rel_j = rel_j_d.data();
      functor.rel_a = rel_a_d.data();

      // The vector length comes from the average row length of A (local plus nonlocal)
      // The host backends run the vector lanes serially, so there it makes no difference
      const double avg_row_nnz = local_rows > 0 ? static_cast<double>(nnzs_input) / local_rows : 0.0;
      int vector_length = 1;
      while (2 * vector_length <= avg_row_nnz && vector_length < 32) vector_length *= 2;
      vector_length = std::min(vector_length, Kokkos::TeamPolicy<>::vector_length_max());

      // ~~~~~~~~~~~~
      // Relabel, then sort each row and sum entries in the same column
      // ~~~~~~~~~~~~
      launch_one_entry_per_row<OneEntryPerRowFunctor::RelabelTag>(functor, vector_length);

      Kokkos::View<PetscInt *> m_i_d, m_j_d;
      Kokkos::View<PetscScalar *> m_a_d;
      KokkosSparse::sort_and_merge_matrix(exec, Kokkos::View<const PetscInt *>(rel_i_d), rel_j_d, rel_a_d, \
                  m_i_d, m_j_d, m_a_d, sentinel + 1);

      // ~~~~~~~~~~~~
      // Count the local and nonlocal entries in each merged row
      // ~~~~~~~~~~~~
      // Zero initialised on exec, the counts go into entry row + 1
      Kokkos::View<PetscInt *> i_local_d(Kokkos::view_alloc(exec, "i_local_d"), local_rows + 1);
      Kokkos::View<PetscInt *> i_nonlocal_d;
      if (mpi) i_nonlocal_d = Kokkos::View<PetscInt *>(Kokkos::view_alloc(exec, "i_nonlocal_d"), local_rows + 1);
      PetscIntKokkosView n_lower_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "n_lower_d"), local_rows);
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
            const PetscInt *j = m_j_d.data();
            const PetscInt start = m_i_d(i), end = m_i_d(i + 1);
            const PetscInt diag_start = lower_bound_sorted(j, start, end, col_start);
            const PetscInt diag_end = lower_bound_sorted(j, diag_start, end, col_end);
            const PetscInt nonlocal_end = lower_bound_sorted(j, diag_end, end, sentinel);
            n_lower_d(i) = diag_start - start;
            i_local_d(i + 1) = diag_end - diag_start;
            if (mpi) i_nonlocal_d(i + 1) = (diag_start - start) + (nonlocal_end - diag_end);
      });

      // Inclusive scans of the row counts give the row pointers
      // Each index only reads and writes its own entry, so this is safe in place
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
      // Split the merged rows into the local and nonlocal blocks
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
      functor.m_i = m_i_d.data();
      functor.m_j = m_j_d.data();
      functor.m_a = m_a_d.data();
      functor.col_start = col_start;
      functor.col_end = col_end;
      functor.n_lower = n_lower_d.data();
      functor.out_l_i = i_local_d.data();
      functor.out_l_j = j_local_d.data();
      functor.out_l_a = a_local_d.data();
      if (mpi)
      {
         functor.out_nl_i = i_nonlocal_d.data();
         functor.out_nl_j = j_nonlocal_d.data();
         functor.out_nl_a = a_nonlocal_d.data();
      }
      launch_one_entry_per_row<OneEntryPerRowFunctor::FillTag>(functor, vector_length);

      // Let's make sure everything on the device is finished
      exec.fence();

      ap.i_l = i_local_d;
      ap.j_l = j_local_d;
      ap.a_l = a_local_d;
      ap.nnz_l = nnzs_local;
      ap.m_i = m_i_d;
      ap.m_j = m_j_d;
      ap.m_a = m_a_d;
      if (mpi)
      {
         // Build garray on the device and rewrite the j_nonlocal_d indices so they are local
         // The nonlocal entries in each row are sorted by global column, so they stay sorted
         // This fences internally
         rewrite_j_global_to_local_device(nnzs_nonlocal, ap.col_ao, j_nonlocal_d, ap.garray_d);
         ap.i_nl = i_nonlocal_d;
         ap.j_nl = j_nonlocal_d;
         ap.a_nl = a_nonlocal_d;
         ap.nnz_nl = nnzs_nonlocal;
      }
   }

   PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local, &device_local_vals));
   if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal, &device_nonlocal_vals));
   if (!p_shell)
   {
      PetscCallVoid(MatSeqAIJRestoreKokkosView(p_local, &p_local_vals));
      if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(p_nonlocal, &p_nonlocal_vals));
   }

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Computes C = A * P as a PETSc matrix when every row of P has at most one entry, see
// build_one_point_ap_kokkos
// If any row of P has more than one entry on any rank, success_int is 0 and output_mat is
// not set
PETSC_INTERN void mat_mult_one_entry_per_row_kokkos(Mat *input_mat, Mat *p_mat, int *success_int, Mat *output_mat)
{
   OnePointAP ap;
   build_one_point_ap_kokkos(input_mat, p_mat, success_int, ap);
   if (!*success_int) return;

   Mat output_mat_local, output_mat_nonlocal;
   // We can create our local diagonal block matrix directly on the device
   PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, ap.local_rows, ap.p_local_cols, ap.i_l, ap.j_l, ap.a_l, &output_mat_local));

   if (ap.mpi)
   {
      // Copy the device garray to the host
      PetscInt *garray_host = NULL;
      PetscCallVoid(PetscMalloc1(ap.col_ao, &garray_host));
      if (ap.col_ao > 0)
      {
         PetscIntKokkosViewHost garray_h(garray_host, ap.col_ao);
         // Device to host so don't need to specify exec space
         Kokkos::deep_copy(garray_h, ap.garray_d);
         // Log copy with petsc
         size_t bytes = ap.col_ao * sizeof(PetscInt);
         PetscCallVoid(PetscLogGpuToCpu(bytes));
      }

      // We can create our nonlocal diagonal block matrix directly on the device
      PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, ap.local_rows, ap.col_ao, ap.i_nl, ap.j_nl, ap.a_nl, &output_mat_nonlocal));

      // We can now create our MPI matrix
      PetscCallVoid(MatCreateMPIAIJWithSeqAIJ(ap.comm, ap.global_rows, ap.global_cols_p, output_mat_local, output_mat_nonlocal, garray_host, output_mat));
   }
   // If in serial
   else
   {
      *output_mat = output_mat_local;
   }
}

//------------------------------------------------------------------------------------------------------------------------

// The kernel handle PETSc uses for its device matrix-matrix products
using PflareKernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<PetscInt, PetscInt, PetscScalar, \
                  DefaultExecutionSpace, DefaultMemorySpace, DefaultMemorySpace>;

// A KokkosCsrMatrix that views (doesn't copy) the device CSR of a MATSEQAIJKOKKOS
// vals has to be restored with MatSeqAIJRestoreKokkosView once the views aren't needed
static KokkosCsrMatrix csr_view_seqaijkokkos(Mat mat, Kokkos::View<const PetscScalar *> &vals)
{
   PetscInt m, n;
   const PetscInt *i_d = nullptr, *j_d = nullptr;
   PetscMemType mtype;
   MatInfo info;
   PetscCallAbort(PETSC_COMM_SELF, MatGetSize(mat, &m, &n));
   PetscCallAbort(PETSC_COMM_SELF, MatGetInfo(mat, MAT_LOCAL, &info));
   const PetscInt nnz = static_cast<PetscInt>(info.nz_used);
   PetscCallAbort(PETSC_COMM_SELF, MatSeqAIJGetCSRAndMemType(mat, &i_d, &j_d, NULL, &mtype));
   PetscCallAbort(PETSC_COMM_SELF, MatSeqAIJGetKokkosView(mat, &vals));
   // Views made from pointers don't own the memory; spgemm only reads them
   Kokkos::View<PetscScalar *> a_v(const_cast<PetscScalar *>(vals.data()), nnz);
   Kokkos::View<const PetscInt *> i_v(i_d, m + 1);
   Kokkos::View<PetscInt *> j_v(const_cast<PetscInt *>(j_d), nnz);
   return KokkosCsrMatrix("csr_view", m, n, nnz, a_v, i_v, j_v);
}

// The row pointers, columns and values of a KokkosCsrMatrix as views we can hand to
// MatCreateSeqAIJKokkosWithKokkosViews (its row pointers are const, so they are copied)
static void csr_to_views(const KokkosCsrMatrix &csr, Kokkos::View<PetscInt *> &i_d, Kokkos::View<PetscInt *> &j_d, \
                  Kokkos::View<PetscScalar *> &a_d)
{
   auto exec = PetscGetKokkosExecutionSpace();
   i_d = Kokkos::View<PetscInt *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "csr_i_d"), csr.numRows() + 1);
   Kokkos::deep_copy(exec, i_d, csr.graph.row_map);
   j_d = csr.graph.entries;
   a_d = csr.values;
}

//------------------------------------------------------------------------------------------------------------------------

// The blocks of a parallel (or serial) AIJ matrix as device views, not yet a PETSc matrix
struct RapRaw {
   bool mpi = false;
   MPI_Comm comm = MPI_COMM_NULL;
   PetscInt local_rows = 0, local_cols = 0, global_rows = 0, global_cols = 0;
   PetscInt row_start = 0, col_start = 0, col_end = 0;
   // The local block, with local columns
   Kokkos::View<PetscInt *> i_l, j_l;
   Kokkos::View<PetscScalar *> a_l;
   // The nonlocal block, with local columns into garray_d (sorted global columns)
   Kokkos::View<PetscInt *> i_nl, j_nl;
   Kokkos::View<PetscScalar *> a_nl;
   PetscIntKokkosView garray_d;
   PetscInt col_ao = 0;
};

// Makes the PETSc matrix from a RapRaw, as MatProductSymbolic_MPIAIJKokkos does
static void wrap_rap_raw_kokkos(RapRaw &raw, Mat *output_mat)
{
   Mat output_mat_local, output_mat_nonlocal;
   PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, raw.local_rows, raw.local_cols, raw.i_l, raw.j_l, raw.a_l, &output_mat_local));
   if (raw.mpi)
   {
      // Copy the device garray to the host
      PetscInt *garray_host = NULL;
      PetscCallVoid(PetscMalloc1(raw.col_ao, &garray_host));
      if (raw.col_ao > 0)
      {
         PetscIntKokkosViewHost garray_h(garray_host, raw.col_ao);
         // Device to host so don't need to specify exec space
         Kokkos::deep_copy(garray_h, raw.garray_d);
         // Log copy with petsc
         size_t bytes = raw.col_ao * sizeof(PetscInt);
         PetscCallVoid(PetscLogGpuToCpu(bytes));
      }
      PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, raw.local_rows, raw.col_ao, raw.i_nl, raw.j_nl, raw.a_nl, &output_mat_nonlocal));
      PetscCallVoid(MatCreateMPIAIJWithSeqAIJ(raw.comm, raw.global_rows, raw.global_cols, output_mat_local, output_mat_nonlocal, garray_host, output_mat));
   }
   else
   {
      *output_mat = output_mat_local;
   }
}

// Computes C = R * (A * P) with kokkos when every row of P has at most one entry, without making
// A * P a PETSc matrix: A * P is built as device views by build_one_point_ap_kokkos and the product
// with R repeats the steps of PETSc's device MatMatMult for MPIAIJKOKKOS
// (MatProductSymbolic_MPIAIJKokkos_AB): R's local block times A * P's local and nonlocal blocks,
// the rows of A * P that R's nonlocal columns need gathered from their owners (F) and split into
// a local and nonlocal block, R's nonlocal block times those, then the sums. The KokkosKernels calls
// are the same as PETSc's in the same order, so the result is the same as MatMatMult(R, A * P)
// R and A must both be MPIAIJKOKKOS or both SEQAIJKOKKOS, with R's columns laid out like A's rows
// The output is returned as device views in out (see RapRaw), not yet a PETSc matrix
// If any row of P has more than one entry on any rank, success_int is 0 and out is not set
static void compute_r_one_point_ap_kokkos(Mat *r_mat, Mat *input_mat, Mat *p_mat, int *success_int, RapRaw &out)
{
   MatType r_type;
   PetscInt r_local_rows, r_local_cols, r_global_rows;
   auto exec = PetscGetKokkosExecutionSpace();
   const auto spgemm_alg = KokkosSparse::SPGEMMAlgorithm::SPGEMM_DEFAULT;

   *success_int = 0;
   PetscCallVoid(MatGetType(*r_mat, &r_type));
   const bool r_mpi = strcmp(r_type, MATMPIAIJKOKKOS) == 0;
   PetscCallVoid(MatGetLocalSize(*r_mat, &r_local_rows, &r_local_cols));
   PetscCallVoid(MatGetSize(*r_mat, &r_global_rows, NULL));

   // ~~~~~~~~~~~~
   // A * P as device views
   // ~~~~~~~~~~~~
   OnePointAP ap;
   build_one_point_ap_kokkos(input_mat, p_mat, success_int, ap);
   if (!*success_int) return;
   PetscCheckAbort(r_mpi == ap.mpi && r_local_cols == ap.local_rows, ap.comm, PETSC_ERR_ARG_SIZ, \
                  "R and A * P are not compatible");

   // R's blocks, viewed on the device
   Mat r_local = *r_mat, r_nonlocal = NULL;
   const PetscInt *colmap_r = nullptr;
   PetscInt n_ghost_r = 0;
   if (r_mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*r_mat, &r_local, &r_nonlocal, &colmap_r));
      PetscCallVoid(MatGetSize(r_nonlocal, NULL, &n_ghost_r));
   }
   Kokkos::View<const PetscScalar *> r_local_vals, r_nonlocal_vals;
   KokkosCsrMatrix Rd = csr_view_seqaijkokkos(r_local, r_local_vals), Ro;
   if (r_mpi) Ro = csr_view_seqaijkokkos(r_nonlocal, r_nonlocal_vals);

   KokkosCsrMatrix APd("APd", ap.local_rows, ap.p_local_cols, ap.nnz_l, ap.a_l, ap.i_l, ap.j_l);

   // The output blocks
   Kokkos::View<PetscInt *> c_i_l, c_j_l, c_i_nl, c_j_nl;
   Kokkos::View<PetscScalar *> c_a_l, c_a_nl;
   PetscInt c_col_ao = 0;
   PetscIntKokkosView c_garray_d;
   PetscSF sf_nz = NULL;

   if (!r_mpi)
   {
      // As MatProductSymbolic_SeqAIJKokkos
      PflareKernelHandle kh;
      kh.create_spgemm_handle(spgemm_alg);
      KokkosCsrMatrix C;
      KokkosSparse::spgemm_symbolic(kh, Rd, false, APd, false, C);
      KokkosSparse::spgemm_numeric(kh, Rd, false, APd, false, C);
      csr_to_views(C, c_i_l, c_j_l, c_a_l);
   }
   else
   {
      const PetscInt local_rows = ap.local_rows;
      const PetscInt col_start = ap.col_start, col_end = ap.col_end;
      KokkosCsrMatrix APo("APo", local_rows, ap.col_ao, ap.nnz_nl, ap.a_nl, ap.i_nl, ap.j_nl);

      // ~~~~~~~~~~~~
      // Gather the rows of A * P that R's nonlocal columns need (F)
      // The halo of R has a leaf for each nonlocal column of R, whose root is that row of A * P
      // ~~~~~~~~~~~~
      PetscSF sf_r;
      PetscInt nroots_r, nleaves_r;
      const PetscInt *ilocal_r;
      const PetscSFNode *iremote_r;
      PetscCallVoid(MatGetMultPetscSF(*r_mat, &sf_r));
      PetscCallVoid(PetscSFGetGraph(sf_r, &nroots_r, &nleaves_r, &ilocal_r, &iremote_r));

      // Each local row of A * P packed together, with global columns in order, which is what the
      // merged rows already hold (the entries after the first row_len are not part of A * P)
      auto i_l = ap.i_l, i_nl = ap.i_nl, m_i = ap.m_i, m_j = ap.m_j;
      auto m_a = ap.m_a;
      PetscIntKokkosView row_len_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "row_len_d"), local_rows);
      PetscIntKokkosView row_off_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "row_off_d"), local_rows);
      const PetscInt nroots_nz = ap.nnz_l + ap.nnz_nl;
      PetscIntKokkosView e_j_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "e_j_d"), nroots_nz);
      PetscScalarKokkosView e_a_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "e_a_d"), nroots_nz);
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
            const PetscInt len = (i_l(i + 1) - i_l(i)) + (i_nl(i + 1) - i_nl(i));
            const PetscInt off = i_l(i) + i_nl(i);
            row_len_d(i) = len;
            row_off_d(i) = off;
            for (PetscInt t = 0; t < len; t++)
            {
               e_j_d(off + t) = m_j(m_i(i) + t);
               e_a_d(off + t) = m_a(m_i(i) + t);
            }
      });
      // The merged rows aren't needed after this, so free them (Kokkos waits for the kernel)
      m_i = Kokkos::View<PetscInt *>();
      m_j = Kokkos::View<PetscInt *>();
      m_a = Kokkos::View<PetscScalar *>();
      ap.m_i = Kokkos::View<PetscInt *>();
      ap.m_j = Kokkos::View<PetscInt *>();
      ap.m_a = Kokkos::View<PetscScalar *>();

      // The length and start of each row of F on its owner
      PetscIntKokkosView f_len_d(Kokkos::view_alloc(exec, "f_len_d"), n_ghost_r);
      PetscIntKokkosView f_off_d(Kokkos::view_alloc(exec, "f_off_d"), n_ghost_r);
      PetscCallVoid(sf_sync_before_begin_kokkos());
      PetscCallVoid(PetscSFBcastWithMemTypeBegin(sf_r, MPIU_INT, PETSC_MEMTYPE_KOKKOS, row_len_d.data(), \
                  PETSC_MEMTYPE_KOKKOS, f_len_d.data(), MPI_REPLACE));
      PetscCallVoid(PetscSFBcastWithMemTypeBegin(sf_r, MPIU_INT, PETSC_MEMTYPE_KOKKOS, row_off_d.data(), \
                  PETSC_MEMTYPE_KOKKOS, f_off_d.data(), MPI_REPLACE));
      PetscCallVoid(PetscSFBcastEnd(sf_r, MPIU_INT, row_len_d.data(), f_len_d.data(), MPI_REPLACE));
      PetscCallVoid(PetscSFBcastEnd(sf_r, MPIU_INT, row_off_d.data(), f_off_d.data(), MPI_REPLACE));
      PetscCallVoid(sf_sync_after_end_kokkos());
      auto f_len_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f_len_d);
      auto f_off_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f_off_d);
      PetscCallVoid(PetscLogGpuToCpu(2 * n_ghost_r * sizeof(PetscInt)));

      // Row pointers of F, and an SF with a leaf for each entry of F, whose root is that entry
      // in the packed rows on its owner
      std::vector<PetscInt> f_rank(n_ghost_r, -1);
      for (PetscInt l = 0; l < nleaves_r; l++) f_rank[ilocal_r ? ilocal_r[l] : l] = iremote_r[l].rank;
      PetscIntKokkosViewHost f_i_h(Kokkos::view_alloc(Kokkos::WithoutInitializing, "f_i_h"), n_ghost_r + 1);
      f_i_h(0) = 0;
      for (PetscInt k = 0; k < n_ghost_r; k++) f_i_h(k + 1) = f_i_h(k) + f_len_h(k);
      const PetscInt f_nnz = f_i_h(n_ghost_r);
      PetscSFNode *iremote_nz;
      PetscCallVoid(PetscMalloc1(f_nnz, &iremote_nz));
      for (PetscInt k = 0; k < n_ghost_r; k++)
      {
         for (PetscInt t = 0; t < f_len_h(k); t++)
         {
            iremote_nz[f_i_h(k) + t].rank = f_rank[k];
            iremote_nz[f_i_h(k) + t].index = f_off_h(k) + t;
         }
      }
      PetscCallVoid(PetscSFCreate(ap.comm, &sf_nz));
      PetscCallVoid(PetscSFSetGraph(sf_nz, nroots_nz, f_nnz, NULL, PETSC_OWN_POINTER, iremote_nz, PETSC_OWN_POINTER));
      PetscCallVoid(PetscSFSetUp(sf_nz));

      // Start sending the entries of F, and overlap with R's local block times A * P
      PetscIntKokkosView f_j_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "f_j_d"), f_nnz);
      PetscScalarKokkosView f_a_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "f_a_d"), f_nnz);
      PetscCallVoid(sf_sync_before_begin_kokkos());
      PetscCallVoid(PetscSFBcastWithMemTypeBegin(sf_nz, MPIU_INT, PETSC_MEMTYPE_KOKKOS, e_j_d.data(), \
                  PETSC_MEMTYPE_KOKKOS, f_j_d.data(), MPI_REPLACE));
      PetscCallVoid(PetscSFBcastWithMemTypeBegin(sf_nz, MPIU_SCALAR, PETSC_MEMTYPE_KOKKOS, e_a_d.data(), \
                  PETSC_MEMTYPE_KOKKOS, f_a_d.data(), MPI_REPLACE));

      // ~~~~~~~~~~~~
      // R's local block times A * P's local (C1) and nonlocal (C2_mid) blocks
      // ~~~~~~~~~~~~
      PflareKernelHandle kh1, kh2, kh3, kh4;
      kh1.create_spgemm_handle(spgemm_alg);
      kh2.create_spgemm_handle(spgemm_alg);
      kh3.create_spgemm_handle(spgemm_alg);
      kh4.create_spgemm_handle(spgemm_alg);
      KokkosCsrMatrix C1, C2_mid, C3, C4;
      KokkosSparse::spgemm_symbolic(kh1, Rd, false, APd, false, C1);
      KokkosSparse::spgemm_symbolic(kh2, Rd, false, APo, false, C2_mid);
      KokkosSparse::spgemm_numeric(kh1, Rd, false, APd, false, C1);
      KokkosSparse::spgemm_numeric(kh2, Rd, false, APo, false, C2_mid);
#if PETSC_PKG_KOKKOS_KERNELS_VERSION_LT(4, 0, 0)
      // As PETSc, older KokkosKernels don't sort the output
      KokkosSparse::sort_crs_matrix(C1);
      KokkosSparse::sort_crs_matrix(C2_mid);
#endif

      PetscCallVoid(PetscSFBcastEnd(sf_nz, MPIU_INT, e_j_d.data(), f_j_d.data(), MPI_REPLACE));
      PetscCallVoid(PetscSFBcastEnd(sf_nz, MPIU_SCALAR, e_a_d.data(), f_a_d.data(), MPI_REPLACE));
      PetscCallVoid(sf_sync_after_end_kokkos());

      // ~~~~~~~~~~~~
      // Split F into its local block (columns in [col_start, col_end), local) and nonlocal block
      // (global columns for now). Each row of F is sorted by global column
      // ~~~~~~~~~~~~
      PetscIntKokkosView f_i_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), f_i_h);
      PetscCallVoid(PetscLogCpuToGpu((n_ghost_r + 1) * sizeof(PetscInt)));
      // Zero initialised on exec, the counts go into entry row + 1
      Kokkos::View<PetscInt *> fd_i_d(Kokkos::view_alloc(exec, "fd_i_d"), n_ghost_r + 1);
      Kokkos::View<PetscInt *> fo_i_d(Kokkos::view_alloc(exec, "fo_i_d"), n_ghost_r + 1);
      PetscIntKokkosView f_nzleft_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "f_nzleft_d"), n_ghost_r);
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, n_ghost_r), KOKKOS_LAMBDA(const PetscInt k) {
            const PetscInt *j = f_j_d.data();
            const PetscInt start = f_i_d(k), end = f_i_d(k + 1);
            const PetscInt diag_start = lower_bound_sorted(j, start, end, col_start);
            const PetscInt diag_end = lower_bound_sorted(j, diag_start, end, col_end);
            f_nzleft_d(k) = diag_start - start;
            fd_i_d(k + 1) = diag_end - diag_start;
            fo_i_d(k + 1) = (end - start) - (diag_end - diag_start);
      });
      PetscInt fd_nnz = 0, fo_nnz = 0;
      Kokkos::parallel_scan(Kokkos::RangePolicy<>(exec, 0, n_ghost_r), KOKKOS_LAMBDA(const PetscInt k, PetscInt &update, const bool final) {
         update += fd_i_d(k + 1);
         if (final) fd_i_d(k + 1) = update;
      }, fd_nnz);
      Kokkos::parallel_scan(Kokkos::RangePolicy<>(exec, 0, n_ghost_r), KOKKOS_LAMBDA(const PetscInt k, PetscInt &update, const bool final) {
         update += fo_i_d(k + 1);
         if (final) fo_i_d(k + 1) = update;
      }, fo_nnz);
      Kokkos::View<PetscInt *> fd_j_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "fd_j_d"), fd_nnz);
      Kokkos::View<PetscScalar *> fd_a_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "fd_a_d"), fd_nnz);
      Kokkos::View<PetscInt *> fo_j_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "fo_j_d"), fo_nnz);
      Kokkos::View<PetscScalar *> fo_a_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "fo_a_d"), fo_nnz);
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, n_ghost_r), KOKKOS_LAMBDA(const PetscInt k) {
            const PetscInt start = f_i_d(k), len = f_i_d(k + 1) - start;
            const PetscInt nzleft = f_nzleft_d(k), len_d = fd_i_d(k + 1) - fd_i_d(k);
            for (PetscInt t = 0; t < len; t++)
            {
               const PetscInt gid = f_j_d(start + t);
               const PetscScalar val = f_a_d(start + t);
               if (t < nzleft) {
                  fo_j_d(fo_i_d(k) + t) = gid;
                  fo_a_d(fo_i_d(k) + t) = val;
               } else if (t < nzleft + len_d) {
                  fd_j_d(fd_i_d(k) + t - nzleft) = gid - col_start;
                  fd_a_d(fd_i_d(k) + t - nzleft) = val;
               } else {
                  fo_j_d(fo_i_d(k) + t - len_d) = gid;
                  fo_a_d(fo_i_d(k) + t - len_d) = val;
               }
            }
      });

      // ~~~~~~~~~~~~
      // The nonlocal columns of the output are the sorted union of A * P's garray and F's nonlocal
      // columns (as ReduceTwoSetsOfGlobalIndices in PETSc). Rewrite both into that union
      // ~~~~~~~~~~~~
      const PetscInt n1 = ap.col_ao;
      PetscIntKokkosView concat_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "concat_d"), n1 + fo_nnz);
      const PetscInt zero = 0;
      if (n1 > 0) Kokkos::deep_copy(exec, Kokkos::subview(concat_d, Kokkos::make_pair(zero, n1)), ap.garray_d);
      Kokkos::deep_copy(exec, Kokkos::subview(concat_d, Kokkos::make_pair(n1, n1 + fo_nnz)), fo_j_d);
      // This fences internally
      rewrite_j_global_to_local_device(n1 + fo_nnz, c_col_ao, concat_d, c_garray_d);
      Kokkos::deep_copy(exec, fo_j_d, Kokkos::subview(concat_d, Kokkos::make_pair(n1, n1 + fo_nnz)));

      KokkosCsrMatrix Fd("Fd", n_ghost_r, ap.p_local_cols, fd_nnz, fd_a_d, fd_i_d, fd_j_d);
      KokkosCsrMatrix Fo("Fo", n_ghost_r, c_col_ao, fo_nnz, fo_a_d, fo_i_d, fo_j_d);

      // ~~~~~~~~~~~~
      // R's nonlocal block times F's local (C3) and nonlocal (C4) blocks
      // ~~~~~~~~~~~~
      KokkosSparse::spgemm_symbolic(kh3, Ro, false, Fd, false, C3);
      KokkosSparse::spgemm_numeric(kh3, Ro, false, Fd, false, C3);
      KokkosSparse::spgemm_symbolic(kh4, Ro, false, Fo, false, C4);
      KokkosSparse::spgemm_numeric(kh4, Ro, false, Fo, false, C4);
#if PETSC_PKG_KOKKOS_KERNELS_VERSION_LT(4, 0, 0)
      KokkosSparse::sort_crs_matrix(C3);
      KokkosSparse::sort_crs_matrix(C4);
#endif

      // C2 is C2_mid with its columns in the union garray
      auto map_d = Kokkos::subview(concat_d, Kokkos::make_pair(zero, n1));
      auto oldj = C2_mid.graph.entries;
      Kokkos::View<PetscInt *> newj(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "newj"), oldj.extent(0));
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, oldj.extent(0)), KOKKOS_LAMBDA(const PetscInt k) {
            newj(k) = map_d(oldj(k));
      });
      KokkosCsrMatrix C2("C2", C2_mid.numRows(), c_col_ao, C2_mid.nnz(), C2_mid.values, C2_mid.graph.row_map, newj);

      // ~~~~~~~~~~~~
      // The output blocks are C1 + C3 and C2 + C4
      // ~~~~~~~~~~~~
      KokkosCsrMatrix Cd, Co;
      kh1.create_spadd_handle(true);
      kh2.create_spadd_handle(true);
      KokkosSparse::spadd_symbolic(&kh1, C1, C3, Cd);
      KokkosSparse::spadd_symbolic(&kh2, C2, C4, Co);
      KokkosSparse::spadd_numeric(&kh1, 1.0, C1, 1.0, C3, Cd);
      KokkosSparse::spadd_numeric(&kh2, 1.0, C2, 1.0, C4, Co);
      csr_to_views(Cd, c_i_l, c_j_l, c_a_l);
      csr_to_views(Co, c_i_nl, c_j_nl, c_a_nl);
   }

   // Let's make sure everything on the device is finished
   exec.fence();

   PetscInt r_row_start, r_row_end;
   PetscCallVoid(MatGetOwnershipRange(*r_mat, &r_row_start, &r_row_end));
   out.mpi = r_mpi;
   out.comm = ap.comm;
   out.local_rows = r_local_rows;
   out.local_cols = ap.p_local_cols;
   out.global_rows = r_global_rows;
   out.global_cols = ap.global_cols_p;
   out.row_start = r_row_start;
   out.col_start = ap.col_start;
   out.col_end = ap.col_end;
   out.i_l = c_i_l;
   out.j_l = c_j_l;
   out.a_l = c_a_l;
   if (r_mpi)
   {
      out.i_nl = c_i_nl;
      out.j_nl = c_j_nl;
      out.a_nl = c_a_nl;
      out.garray_d = c_garray_d;
      out.col_ao = c_col_ao;
   }

   PetscCallVoid(PetscSFDestroy(&sf_nz));
   PetscCallVoid(MatSeqAIJRestoreKokkosView(r_local, &r_local_vals));
   if (r_mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(r_nonlocal, &r_nonlocal_vals));
}

//------------------------------------------------------------------------------------------------------------------------

// Computes C = R * (A * P) as a PETSc matrix with kokkos when every row of P has at most one entry,
// see compute_r_one_point_ap_kokkos
// If any row of P has more than one entry on any rank, success_int is 0 and output_mat is not set
PETSC_INTERN void mat_mult_r_one_point_ap_kokkos(Mat *r_mat, Mat *input_mat, Mat *p_mat, int *success_int, Mat *output_mat)
{
   RapRaw raw;
   compute_r_one_point_ap_kokkos(r_mat, input_mat, p_mat, success_int, raw);
   if (!*success_int) return;
   wrap_rap_raw_kokkos(raw, output_mat);
}

// Computes C = R * (A * P) with kokkos when every row of P has at most one entry, see
// compute_r_one_point_ap_kokkos, and returns it in raw_handle as device views, not as a PETSc matrix
// raw_handle has to be passed to rap_raw_drop_kokkos, which frees it
// If any row of P has more than one entry on any rank, success_int is 0 and raw_handle is NULL
PETSC_INTERN void mat_mult_r_one_point_ap_raw_kokkos(Mat *r_mat, Mat *input_mat, Mat *p_mat, int *success_int, void **raw_handle)
{
   *raw_handle = NULL;
   RapRaw *raw = new RapRaw();
   compute_r_one_point_ap_kokkos(r_mat, input_mat, p_mat, success_int, *raw);
   if (!*success_int)
   {
      delete raw;
      return;
   }
   *raw_handle = raw;
}

// Drops small entries from the matrix in raw_handle (from mat_mult_r_one_point_ap_raw_kokkos) with
// kokkos, as remove_small_from_sparse_kokkos does on a PETSc matrix, then frees raw_handle
// If allow_alias_int is set and nothing would be dropped (or added) on any rank, output_mat is the
// matrix in raw_handle itself, made into a PETSc matrix; otherwise that is never made into a matrix
PETSC_INTERN void rap_raw_drop_kokkos(void **raw_handle, const PetscReal tol, Mat *output_mat, \
                  const int relative_max_row_tolerance_int, const int lump_int, \
                  const int allow_drop_diagonal_int, const int allow_diag_strength_int, \
                  const int allow_alias_int)
{
   RapRaw *raw = static_cast<RapRaw *>(*raw_handle);
   DropInput in;
   in.mpi = raw->mpi;
   in.comm = raw->comm;
   in.local_rows = raw->local_rows;
   in.local_cols = raw->local_cols;
   in.global_rows = raw->global_rows;
   in.global_cols = raw->global_cols;
   in.row_start = raw->row_start;
   in.col_start = raw->col_start;
   in.col_end = raw->col_end;
   in.l_i = raw->i_l.data();
   in.l_j = raw->j_l.data();
   in.l_a = raw->a_l;
   in.nnz_l = raw->j_l.extent(0);
   if (raw->mpi)
   {
      in.nl_i = raw->i_nl.data();
      in.nl_j = raw->j_nl.data();
      in.nl_a = raw->a_nl;
      in.nnz_nl = raw->j_nl.extent(0);
      in.colmap_d = raw->garray_d;
      in.cols_ao = raw->col_ao;
   }
   const bool unchanged = drop_small_rows_core(in, tol, output_mat, relative_max_row_tolerance_int, lump_int, \
                  allow_drop_diagonal_int, allow_diag_strength_int, allow_alias_int);
   if (unchanged) wrap_rap_raw_kokkos(*raw, output_mat);
   delete raw;
   *raw_handle = NULL;
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

   // Get the output pointers
   const PetscInt *device_local_i_output = nullptr, *device_local_j_output = nullptr, *device_nonlocal_i_output = nullptr, *device_nonlocal_j_output = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_output, &device_local_i_output, &device_local_j_output, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_output, &device_nonlocal_i_output, &device_nonlocal_j_output, NULL, &mtype));
   // Output values are read-modify-write (+= and conditional =)
   Kokkos::View<PetscScalar *> device_local_vals_output;
   Kokkos::View<PetscScalar *> device_nonlocal_vals_output;
   PetscCallVoid(MatSeqAIJGetKokkosView(mat_local_output, &device_local_vals_output));
   if (mpi) PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal_output, &device_nonlocal_vals_output));

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
               if (scratch_indices(j) == -1) thread_sum += alpha * device_local_vals(device_local_i[i] + j);
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
                  if (scratch_indices_nonlocal(j) == -1) thread_sum += alpha * device_nonlocal_vals(device_nonlocal_i[i] + j);
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
               device_local_vals_output(device_local_i_output[i] + scratch_indices(j)) += alpha * device_local_vals(device_local_i[i] + j);
            }
            else
            {
               device_local_vals_output(device_local_i_output[i] + scratch_indices(j)) = device_local_vals(device_local_i[i] + j);
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
                  device_nonlocal_vals_output(device_nonlocal_i_output[i] + scratch_indices_nonlocal(j)) += alpha * device_nonlocal_vals(device_nonlocal_i[i] + j);
               }
               else
               {
                  device_nonlocal_vals_output(device_nonlocal_i_output[i] + scratch_indices_nonlocal(j)) = device_nonlocal_vals(device_nonlocal_i[i] + j);
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
               if (is_diagonal) device_local_vals_output(device_local_i_output[i] + j) += lump_val;
            });   
         }
         else
         {
            // Only loop over the ncols in the nonlocal component
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_nonlocal_output), [&](const PetscInt j) {

               // Is this column the diagonal
               const bool is_diagonal = colmap_output_d(device_nonlocal_j_output[device_nonlocal_i_output[i] + j]) == row_index_global;

               // Will only happen for one thread
               if (is_diagonal) device_nonlocal_vals_output(device_nonlocal_i_output[i] + j) += lump_val;
            });               
         }
      }       
   });
   Kokkos::fence();

   PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local, &device_local_vals));
   if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal, &device_nonlocal_vals));

   // The matching restore handles MatSeqAIJKokkosModifyDevice (clears sync state,
   // marks device modified, invalidates transpose/hermitian, bumps object state).
   PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local_output, &device_local_vals_output));
   if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal_output, &device_nonlocal_vals_output));

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Set all the values of the matrix to val
PETSC_INTERN void MatSetAllValues_kokkos(Mat *input_mat, PetscScalar val)
{
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

   auto exec = PetscGetKokkosExecutionSpace();

   Kokkos::View<PetscScalar *> device_local_vals;
   Kokkos::View<PetscScalar *> device_nonlocal_vals;
   PetscCallVoid(MatSeqAIJGetKokkosViewWrite(mat_local, &device_local_vals));
   if (mpi) PetscCallVoid(MatSeqAIJGetKokkosViewWrite(mat_nonlocal, &device_nonlocal_vals));

   // Set all values to val
   Kokkos::deep_copy(exec, device_local_vals, val);
   if (mpi)
   {
      Kokkos::deep_copy(exec, device_nonlocal_vals, val);
   }
   // Fence after the async fill of the matrix values - keep the convention of being finished before we exit
   Kokkos::fence();

   // The matching restore handles MatSeqAIJKokkosModifyDevice (clears sync state,
   // marks device modified, invalidates transpose/hermitian, bumps object state).
   PetscCallVoid(MatSeqAIJRestoreKokkosViewWrite(mat_local, &device_local_vals));
   if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosViewWrite(mat_nonlocal, &device_nonlocal_vals));

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
               a_local_d(i_local_d(i) + j) = device_local_vals(device_local_i[i] + j);
                     
            });     

            // For over nonlocal columns - copy in input
            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];         

               Kokkos::parallel_for(
                  Kokkos::TeamVectorRange(t, ncols_nonlocal), [&](const PetscInt j) {

                  // We keep the existing local indices in the off-diagonal block here
                  j_nonlocal_d(i_nonlocal_d(i) + j) = device_nonlocal_j[device_nonlocal_i[i] + j];
                  a_nonlocal_d(i_nonlocal_d(i) + j) = device_nonlocal_vals(device_nonlocal_i[i] + j);
                        
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
      // Get device pointers for the existing i and j arrays
      const PetscInt *device_local_i_output = nullptr, *device_local_j_output = nullptr, *device_nonlocal_i_output = nullptr;
      PetscMemType mtype;
      PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_output, &device_local_i_output, &device_local_j_output, NULL, &mtype));
      if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_output, &device_nonlocal_i_output, NULL, NULL, &mtype));

      // Output values are fully overwritten below (zeroed first, then assigned)
      Kokkos::View<PetscScalar *> device_local_a_output;
      Kokkos::View<PetscScalar *> device_nonlocal_a_output;
      PetscCallVoid(MatSeqAIJGetKokkosViewWrite(mat_local_output, &device_local_a_output));
      if (mpi) PetscCallVoid(MatSeqAIJGetKokkosViewWrite(mat_nonlocal_output, &device_nonlocal_a_output));

      // Because we might be missing diagonals and we're going to skip some of them
      // in the writing loop below
      Kokkos::deep_copy(exec, device_local_a_output, 0.0);

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
               if (device_local_j_output[device_local_i_output[i] + j] >= i && \
                     !found_diag_row_d(i)) offset = 1;

               device_local_a_output(device_local_i_output[i] + j + offset) = device_local_vals(device_local_i[i] + j);
            });

            // For over nonlocal columns - copy in input - identical structure in the off-diag block
            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];

               Kokkos::parallel_for(
                  Kokkos::TeamVectorRange(t, ncols_nonlocal), [&](const PetscInt j) {

                  // We keep the existing local indices in the off-diagonal block here
                  // we have all the same columns as input and hence the same garray
                  device_nonlocal_a_output(device_nonlocal_i_output[i] + j) = device_nonlocal_vals(device_nonlocal_i[i] + j);

               });
            }
      });

      Kokkos::fence();

      // The matching restore handles MatSeqAIJKokkosModifyDevice (clears sync state,
      // marks device modified, invalidates transpose/hermitian, bumps object state).
      PetscCallVoid(MatSeqAIJRestoreKokkosViewWrite(mat_local_output, &device_local_a_output));
      if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosViewWrite(mat_nonlocal_output, &device_nonlocal_a_output));

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
      KokkosSparse::sort_crs_matrix(exec, csrmat_local);

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

   PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local, &device_local_vals));
   if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal, &device_nonlocal_vals));

   return;
}


//------------------------------------------------------------------------------------------------------------------------

// Does a MatAXPY for a MPIAIJ Kokkos matrix - the petsc version currently uses the host making it very slow
PETSC_INTERN void MatAXPY_kokkos(Mat *Y, PetscScalar alpha, Mat *X)
{
   Mat mat_local_y = NULL, mat_nonlocal_y = NULL;
   Mat mat_local_x = NULL, mat_nonlocal_x = NULL;

   const PetscInt *colmap_y, *colmap_x;
   PetscCallVoid(MatMPIAIJGetSeqAIJ(*Y, &mat_local_y, &mat_nonlocal_y, &colmap_y));
   PetscCallVoid(MatMPIAIJGetSeqAIJ(*X, &mat_local_x, &mat_nonlocal_x, &colmap_x));

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
   // Local block: hand off to PETSc's MatAXPY on a duplicate of mat_local_y
   // (the duplicate becomes the diagonal block of the new MPI matrix)
   // ~~~~~~~~~~~~~~~
   Mat Z_local;
   PetscCallVoid(MatDuplicate(mat_local_y, MAT_COPY_VALUES, &Z_local));
   PetscCallVoid(MatAXPY(Z_local, alpha, mat_local_x, DIFFERENT_NONZERO_PATTERN));

   // ~~~~~~~~~~~~~~~
   // Nonlocal block: build the merged garray = union(garray_y, garray_x) and use it as
   // the column space of temporary SeqAIJKokkos matrices, then call PETSc's MatAXPY.
   // Because spadd produces a nonzero in column c iff X or Y had one there, the union
   // garray is exactly the output garray - no second remap is needed afterward.
   // ~~~~~~~~~~~~~~~

   // Concatenate the two garrays on the device, then run them through rewrite_j_global_to_local.
   // After it returns:
   //   concat[0 .. cols_ao_y)               = merged-local indices for each garray_y entry
   //   concat[cols_ao_y .. cols_ao_y+cols_ao_x) = merged-local indices for each garray_x entry
   //   garray_host[0 .. col_ao_output)      = sorted-unique global indices of garray_y, garray_x
   PetscInt cols_ao_concat = cols_ao_y + cols_ao_x;
   PetscIntKokkosView concat = PetscIntKokkosView("concat", cols_ao_concat);
   PetscInt zero = 0;
   Kokkos::deep_copy(exec, Kokkos::subview(concat, Kokkos::make_pair(zero, cols_ao_y)), colmap_input_d_y);
   Kokkos::deep_copy(exec, Kokkos::subview(concat, Kokkos::make_pair(cols_ao_y, cols_ao_concat)), colmap_input_d_x);

   PetscInt col_ao_output = 0;
   PetscInt *garray_host = NULL;
   rewrite_j_global_to_local(cols_ao_concat, col_ao_output, concat, &garray_host);

   // Pull device CSR pointers for the original nonlocal blocks (we never mutate them).
   const PetscInt *device_nonlocal_y_i = nullptr, *device_nonlocal_y_j = nullptr;
   const PetscInt *device_nonlocal_x_i = nullptr, *device_nonlocal_x_j = nullptr;
   PetscMemType mtype;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_y, &device_nonlocal_y_i, &device_nonlocal_y_j, NULL, &mtype));
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_x, &device_nonlocal_x_i, &device_nonlocal_x_j, NULL, &mtype));
   Kokkos::View<const PetscScalar *> a_y_orig;
   Kokkos::View<const PetscScalar *> a_x_orig;
   PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal_y, &a_y_orig));
   PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal_x, &a_x_orig));

   // Read the nnz of each nonlocal block from its host i pointer
   PetscInt nnz_y, nnz_x;
   {
      PetscInt n;
      PetscBool symmetric = PETSC_FALSE, inodecompressed = PETSC_FALSE, done;
      const PetscInt *ia;
      PetscCallVoid(MatGetRowIJ(mat_nonlocal_y, 0, symmetric, inodecompressed, &n, &ia, NULL, &done));
      nnz_y = ia[n];
      PetscCallVoid(MatRestoreRowIJ(mat_nonlocal_y, 0, symmetric, inodecompressed, &n, &ia, NULL, &done));
      PetscCallVoid(MatGetRowIJ(mat_nonlocal_x, 0, symmetric, inodecompressed, &n, &ia, NULL, &done));
      nnz_x = ia[n];
      PetscCallVoid(MatRestoreRowIJ(mat_nonlocal_x, 0, symmetric, inodecompressed, &n, &ia, NULL, &done));
   }

   // Build owned (i, j, a) copies for each side. j is rewritten from original-local
   // straight into merged-local via the concat lookup.
   Kokkos::View<PetscInt *> i_y_temp("i_y_temp", local_rows + 1);
   Kokkos::View<PetscInt *> j_y_temp("j_y_temp", nnz_y);
   Kokkos::View<PetscScalar *> a_y_temp("a_y_temp", nnz_y);
   Kokkos::View<const PetscInt *, DefaultMemorySpace, Kokkos::MemoryUnmanaged> i_y_orig(device_nonlocal_y_i, local_rows + 1);
   Kokkos::deep_copy(exec, i_y_temp, i_y_orig);
   Kokkos::deep_copy(exec, a_y_temp, a_y_orig);
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, nnz_y), KOKKOS_LAMBDA(const PetscInt k) {
         j_y_temp(k) = concat(device_nonlocal_y_j[k]);
      });

   Kokkos::View<PetscInt *> i_x_temp("i_x_temp", local_rows + 1);
   Kokkos::View<PetscInt *> j_x_temp("j_x_temp", nnz_x);
   Kokkos::View<PetscScalar *> a_x_temp("a_x_temp", nnz_x);
   Kokkos::View<const PetscInt *, DefaultMemorySpace, Kokkos::MemoryUnmanaged> i_x_orig(device_nonlocal_x_i, local_rows + 1);
   Kokkos::deep_copy(exec, i_x_temp, i_x_orig);
   Kokkos::deep_copy(exec, a_x_temp, a_x_orig);
   const PetscInt offset_x = cols_ao_y;
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, nnz_x), KOKKOS_LAMBDA(const PetscInt k) {
         j_x_temp(k) = concat(offset_x + device_nonlocal_x_j[k]);
      });

   Kokkos::fence();

   PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal_y, &a_y_orig));
   PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal_x, &a_x_orig));

   // concat is no longer needed - both j-rewrite kernels have completed.
   concat = PetscIntKokkosView();

   // Wrap the rewritten CSR as temporary SeqAIJKokkos matrices with col_ao_output cols
   // and let PETSc's MatAXPY run the spadd.
   Mat temp_y, temp_x;
   PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, col_ao_output, i_y_temp, j_y_temp, a_y_temp, &temp_y));
   PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, col_ao_output, i_x_temp, j_x_temp, a_x_temp, &temp_x));

   PetscCallVoid(MatAXPY(temp_y, alpha, temp_x, DIFFERENT_NONZERO_PATTERN));

   // temp_x is no longer needed; free it
   PetscCallVoid(MatDestroy(&temp_x));

   // Extract the result from temp_y so we can own it independently.
   const PetscInt *device_z_i = nullptr, *device_z_j = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(temp_y, &device_z_i, &device_z_j, NULL, &mtype));
   Kokkos::View<const PetscScalar *> a_z_unm;
   PetscCallVoid(MatSeqAIJGetKokkosView(temp_y, &a_z_unm));

   // Read the spadd-result nnz from its host i pointer
   PetscInt nnz_z;
   {
      PetscInt n;
      PetscBool symmetric = PETSC_FALSE, inodecompressed = PETSC_FALSE, done;
      const PetscInt *ia;
      PetscCallVoid(MatGetRowIJ(temp_y, 0, symmetric, inodecompressed, &n, &ia, NULL, &done));
      nnz_z = ia[n];
      PetscCallVoid(MatRestoreRowIJ(temp_y, 0, symmetric, inodecompressed, &n, &ia, NULL, &done));
   }

   Kokkos::View<PetscInt *> i_nonlocal_d_copy("i_nonlocal_d_copy", local_rows + 1);
   Kokkos::View<PetscInt *> j_nonlocal_d_copy("j_nonlocal_d_copy", nnz_z);
   Kokkos::View<PetscScalar *> a_nonlocal_d_copy("a_nonlocal_d_copy", nnz_z);
   Kokkos::View<const PetscInt *, DefaultMemorySpace, Kokkos::MemoryUnmanaged> i_z_unm(device_z_i, local_rows + 1);
   Kokkos::View<const PetscInt *, DefaultMemorySpace, Kokkos::MemoryUnmanaged> j_z_unm(device_z_j, nnz_z);
   Kokkos::deep_copy(exec, i_nonlocal_d_copy, i_z_unm);
   Kokkos::deep_copy(exec, j_nonlocal_d_copy, j_z_unm);
   Kokkos::deep_copy(exec, a_nonlocal_d_copy, a_z_unm);
   Kokkos::fence();

   PetscCallVoid(MatSeqAIJRestoreKokkosView(temp_y, &a_z_unm));

   // temp_y's data has been copied out; free it before allocating Z_nonlocal.
   PetscCallVoid(MatDestroy(&temp_y));

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

// Row kernels for MatCreateSubMatrix_Seq_kokkos
// Each team handles team_size rows, one thread per row, with vector lanes over the
// columns in that row
struct SubMatrixRowsFunctor {
   struct CountTag {};
   struct FillTag {};

   PetscIntKokkosView is_row;
   const PetscInt *in_i = nullptr, *in_j = nullptr;
   Kokkos::View<const PetscScalar *> in_a;
   // smap(col) is the output column + 1, or 0 if the column isn't kept
   PetscIntKokkosView smap;
   PetscInt nrows = 0;
   // Count: number of kept entries in each row goes into count_out[row + 1]
   PetscInt *count_out = nullptr;
   // Fill: start of each output row, and the output j (only written if not null) and values
   const PetscInt *out_i = nullptr;
   PetscInt *out_j = nullptr;
   PetscScalar *out_a = nullptr;

   KOKKOS_INLINE_FUNCTION void operator()(const CountTag &, const KokkosTeamMemberType &t) const
   {
      const PetscInt row_start = t.league_rank() * t.team_size();
      const PetscInt row_end = row_start + t.team_size() < nrows ? row_start + t.team_size() : nrows;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, row_start, row_end), [&](const PetscInt i_idx_is_row) {
         const PetscInt i = is_row(i_idx_is_row);
         const PetscInt row_offset = in_i[i];
         PetscInt row_count = 0;
         Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, in_i[i + 1] - row_offset),
            [&](const PetscInt k, PetscInt &count) {
               if (smap(in_j[row_offset + k])) count++;
            }, row_count);
         Kokkos::single(Kokkos::PerThread(t), [&]() { count_out[i_idx_is_row + 1] = row_count; });
      });
   }

   // The vector lanes do an exclusive scan over the kept flags to find their output position,
   // so the kept entries stay in the order of the input row
   KOKKOS_INLINE_FUNCTION void operator()(const FillTag &, const KokkosTeamMemberType &t) const
   {
      const PetscInt row_start = t.league_rank() * t.team_size();
      const PetscInt row_end = row_start + t.team_size() < nrows ? row_start + t.team_size() : nrows;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, row_start, row_end), [&](const PetscInt i_idx_is_row) {
         const PetscInt i = is_row(i_idx_is_row);
         const PetscInt row_offset = in_i[i];
         const PetscInt out_offset = out_i[i_idx_is_row];
         Kokkos::parallel_scan(Kokkos::ThreadVectorRange(t, in_i[i + 1] - row_offset),
            [&](const PetscInt k, PetscInt &partial_sum, const bool is_final) {
               const PetscInt col = smap(in_j[row_offset + k]);
               if (col) {
                  if (is_final) {
                     if (out_j) out_j[out_offset + partial_sum] = col - 1;
                     out_a[out_offset + partial_sum] = in_a(row_offset + k);
                  }
                  partial_sum++;
               }
            });
      });
   }
};

// Launches one of the SubMatrixRowsFunctor kernels over all the rows
template <class Tag>
static void launch_submatrix_rows(const SubMatrixRowsFunctor &functor, const int vector_length)
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

// Does a MatCreateSubMatrix on a sequential Kokkos matrix
// is_row_d_d and is_col_d_d must have the local indices in them
// is_col must be sorted
PETSC_INTERN void MatCreateSubMatrix_Seq_kokkos(Mat *input_mat, PetscIntKokkosView &is_row_d_d, PetscIntKokkosView &is_col_d_d, const int reuse_int, Mat *output_mat)
{
   PetscInt local_rows, local_cols;
   auto exec = PetscGetKokkosExecutionSpace();

   PetscCallVoid(MatGetLocalSize(*input_mat, &local_rows, &local_cols));
   const PetscInt local_rows_row = is_row_d_d.extent(0), local_cols_col = is_col_d_d.extent(0);

   // ~~~~~~~~~~~~
   // Get pointers to the i,j on the device and Kokkos view to the values
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr;
   PetscMemType mtype;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(*input_mat, &device_local_i, &device_local_j, NULL, &mtype));
   Kokkos::View<const PetscScalar *> device_local_vals;
   PetscCallVoid(MatSeqAIJGetKokkosView(*input_mat, &device_local_vals));

   // If no columns are kept the output has no entries, so there is nothing to compute
   // This is common for the off-diagonal block, where it saves launching kernels over every row
   const bool no_cols = local_cols_col == 0;

   SubMatrixRowsFunctor functor;
   functor.is_row = is_row_d_d;
   functor.in_i = device_local_i;
   functor.in_j = device_local_j;
   functor.in_a = device_local_vals;
   functor.nrows = local_rows_row;

   // Map which columns in the original mat are in is_col
   if (!no_cols)
   {
      // Zero initialised on exec
      PetscIntKokkosView smap_d(Kokkos::view_alloc(exec, "smap_d"), local_cols);
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, local_cols_col), KOKKOS_LAMBDA(PetscInt i) {
            smap_d(is_col_d_d(i)) = i + 1;
      });
      functor.smap = smap_d;
   }

   // The vector length comes from the average row length, so short rows don't leave most
   // of the threads idle and long rows still get some parallelism
   // The host backends run the vector lanes serially, so there it makes no difference
   MatInfo info;
   PetscCallVoid(MatGetInfo(*input_mat, MAT_LOCAL, &info));
   const double avg_row_nnz = local_rows > 0 ? info.nz_used / local_rows : 0.0;
   int vector_length = 1;
   while (2 * vector_length <= avg_row_nnz && vector_length < 32) vector_length *= 2;
   vector_length = std::min(vector_length, Kokkos::TeamPolicy<>::vector_length_max());

   if (!reuse_int)
   {
      // Zero initialised on exec, which is the output if no columns are kept
      Kokkos::View<PetscInt *> i_local_d(Kokkos::view_alloc(exec, "i_local_d"), local_rows_row+1);
      PetscInt nnzs_match_local = 0;

      if (!no_cols)
      {
         functor.count_out = i_local_d.data();
         launch_submatrix_rows<SubMatrixRowsFunctor::CountTag>(functor, vector_length);

         // Inclusive scan of the row counts gives the row pointers
         // Each index only reads and writes its own entry, so this is safe in place
         Kokkos::parallel_scan(
            Kokkos::RangePolicy<>(exec, 0, local_rows_row),
            KOKKOS_LAMBDA(const PetscInt i_idx_is_row, PetscInt &update, const bool final) {
               update += i_local_d(i_idx_is_row + 1);
               if (final) i_local_d(i_idx_is_row + 1) = update;
            },
            nnzs_match_local);
      }

      // Every entry is written by the fill below
      Kokkos::View<PetscScalar *> a_local_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "a_local_d"), nnzs_match_local);
      Kokkos::View<PetscInt *> j_local_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "j_local_d"), nnzs_match_local);

      if (nnzs_match_local > 0)
      {
         functor.out_i = i_local_d.data();
         functor.out_j = j_local_d.data();
         functor.out_a = a_local_d.data();
         launch_submatrix_rows<SubMatrixRowsFunctor::FillTag>(functor, vector_length);
      }

      // Let's make sure everything on the device is finished
      exec.fence();

      // Create the matrix given the sorted csr
      PetscCallVoid(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows_row, local_cols_col, i_local_d, j_local_d, a_local_d, output_mat));
   }
   // If we're reusing, we can just write directly to the existing values
   // With no kept columns the output has no values, so there is nothing to write
   else if (!no_cols)
   {
      // Get device pointers for the existing i array
      const PetscInt *device_local_i_output = nullptr;
      PetscCallVoid(MatSeqAIJGetCSRAndMemType(*output_mat, &device_local_i_output, NULL, NULL, &mtype));
      // Output values are fully overwritten below (one write per output entry)
      Kokkos::View<PetscScalar *> device_local_a_output;
      PetscCallVoid(MatSeqAIJGetKokkosViewWrite(*output_mat, &device_local_a_output));

      functor.out_i = device_local_i_output;
      functor.out_a = device_local_a_output.data();
      launch_submatrix_rows<SubMatrixRowsFunctor::FillTag>(functor, vector_length);

      exec.fence();

      // The matching restore handles MatSeqAIJKokkosModifyDevice (clears sync state,
      // marks device modified, invalidates transpose/hermitian, bumps object state).
      PetscCallVoid(MatSeqAIJRestoreKokkosViewWrite(*output_mat, &device_local_a_output));
   }

   PetscCallVoid(MatSeqAIJRestoreKokkosView(*input_mat, &device_local_vals));

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Does a MatGetSubMatrix for a Kokkos matrix - the petsc version currently uses the host making it very slow
// This version only works  works if the input IS have the same parallel row/column distribution
// as the matrices, ie equivalent to MatCreateSubMatrix_MPIAIJ_SameRowDist
// is_col must be sorted
// This one uses the views is_row_d_d and is_col_d_d directly, rewritten to be the local indices
PETSC_INTERN void MatCreateSubMatrix_kokkos_view(Mat *input_mat, \
         PetscIntKokkosView &is_row_d_d, PetscInt global_rows_row, \
         PetscIntKokkosView &is_col_d_d, PetscInt global_cols_col, const int reuse_int, Mat *output_mat)
{
   PetscInt local_rows, local_cols;
   PetscInt global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one;
   PetscCallVoid(MatGetOwnershipRange(*input_mat, &global_row_start, &global_row_end_plus_one));
   PetscInt local_cols_col = is_col_d_d.extent(0);
   auto exec = PetscGetKokkosExecutionSpace();

   // Are we in parallel?
   MatType mat_type;
   MPI_Comm MPI_COMM_MATRIX;
   PetscCallVoid(MatGetType(*input_mat, &mat_type));

   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;
   PetscCallVoid(PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX));
   PetscCallVoid(MatGetSize(*input_mat, &global_rows, &global_cols));
   PetscCallVoid(MatGetLocalSize(*input_mat, &local_rows, &local_cols));

   Mat mat_local = NULL, mat_nonlocal = NULL;
   Mat output_mat_local, output_mat_nonlocal;
   // The matrix's own mult SF (the borrowed mvctx) drives the halo broadcast
   PetscSF sf = NULL;

   PetscInt rows_ao, cols_ao;
   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*input_mat, &mat_local, &mat_nonlocal, NULL));
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao));

      if (reuse_int)
      {
         PetscCallVoid(MatMPIAIJGetSeqAIJ(*output_mat, &output_mat_local, &output_mat_nonlocal, NULL));
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
         PetscCallMPIAbort(MPI_COMM_MATRIX, MPI_Scan(&local_cols_col, &isstart, 1, MPIU_INT, MPI_SUM, MPI_COMM_MATRIX));
         isstart -= local_cols_col;

         // Kokkos version of ISGetSeqIS_SameColDist_Private (mpiaij.c)
         // The new global column indices are sent as PetscInt through the matrix mult SF,
         // as encoding them in PetscScalar loses exactness above 2^24 in single precision.
         PetscCallVoid(MatGetMultPetscSF(*input_mat, &sf));

         /* (1) cmap[is_col(i)] = i + isstart (global index of iscol[i]), rest = -1 */
         // Unlike PETSc we don't send a separate x, cmap > -1 already marks a column as present
         PetscIntKokkosView cmap_d("cmap_d", local_cols);
         PetscIntKokkosView lcmap_d("lcmap_d", cols_ao);
         Kokkos::deep_copy(exec, cmap_d, -1);
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(exec, 0, local_cols_col), KOKKOS_LAMBDA(PetscInt i) {
               cmap_d(is_col_d_d(i)) = i + isstart;
         });

         /* (2) Broadcast cmap to get the off-process portions */
         // While Begin/End is in flight, do not touch the send/recv buffers
         PetscCallVoid(sf_sync_before_begin_kokkos());
         PetscCallVoid(PetscSFBcastWithMemTypeBegin(sf, MPIU_INT,
                     PETSC_MEMTYPE_KOKKOS, cmap_d.data(),
                     PETSC_MEMTYPE_KOKKOS, lcmap_d.data(),
                     MPI_REPLACE));

         /* (3) Count how many off-local columns match */
         PetscInt col_ao_output = 0;

         // One bigger for exclusive scan
         auto is_col_o_match_d = PetscIntKokkosView("is_col_o_match_d", cols_ao+1);
         Kokkos::deep_copy(exec, is_col_o_match_d, 0);

         // cmap broadcast completed: lcmap_d is now safe to read
         PetscCallVoid(PetscSFBcastEnd(sf, MPIU_INT, cmap_d.data(), lcmap_d.data(), MPI_REPLACE));
         PetscCallVoid(sf_sync_after_end_kokkos());

         if (cols_ao > 0)
         {
            Kokkos::parallel_reduce("FindMatches", Kokkos::RangePolicy<>(exec, 0, cols_ao),
               KOKKOS_LAMBDA(const PetscInt i, PetscInt& thread_sum) {
                  // This is the scattered cmap for all of the non-local columns in the input mat
                  // It's not -1 if that column is present on another rank
                  if (lcmap_d(i) > -1) {
                     thread_sum++;
                     is_col_o_match_d(i) = 1; // Mark this as a match
                  }
               },
               Kokkos::Sum<PetscInt>(col_ao_output)
            );
         }

         // Need to do an exclusive scan on is_col_o_match_d to get the new local indices
         // Have to remember to go up to cols_ao+1
         Kokkos::parallel_scan(Kokkos::RangePolicy<>(exec, 0, cols_ao+1), KOKKOS_LAMBDA(const PetscInt i, PetscInt& partial_sum, const bool is_final) {
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

         // Loop over all the cols in the input matrix
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(exec, 0, cols_ao), KOKKOS_LAMBDA(PetscInt i) {

               // We can tell if is_col_o_match_d had 1 in it in this position by comparing the result
               // of the exclusive scan for this index and the next one
               if (is_col_o_match_d(i+1) > is_col_o_match_d(i))
               {
                  is_col_o_d(is_col_o_match_d(i)) = i;
                  garray_output_d(is_col_o_match_d(i)) = lcmap_d(i);
               }
         });
         // Fence so the parallel for finishes
         exec.fence();
      }
      // If we're reusing we have the iscol_o associated with the output_mat
      else
      {
         // Get the iscol_o from the output_mat
         IS iscol_o;
         /* Retrieve isrow_d, iscol_d and iscol_o from output */
         PetscCallVoid(PetscObjectQuery((PetscObject)(*output_mat), "iscol_o", (PetscObject *)&iscol_o));
         //PetscCheck(iscol_o, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "iscol_o passed in was not used before, cannot reuse");

         const PetscInt *iscol_o_indices_ptr;
         PetscCallVoid(ISGetIndices(iscol_o, &iscol_o_indices_ptr));

         PetscInt local_cols_iscol_o;
         PetscCallVoid(ISGetLocalSize(iscol_o, &local_cols_iscol_o));

         // Copy the iscol_o to the device
         auto iscol_o_view_h = PetscIntConstKokkosViewHost(iscol_o_indices_ptr, local_cols_iscol_o);    
         is_col_o_d = PetscIntKokkosView("is_col_o_d", local_cols_iscol_o);
         Kokkos::deep_copy(exec, is_col_o_d, iscol_o_view_h);
         // Log copy with petsc
         bytes = iscol_o_view_h.extent(0) * sizeof(PetscInt);
         PetscCallVoid(PetscLogCpuToGpu(bytes));

         // Fence after the async copy of the iscol_o indices to the device - ISRestoreIndices may free the host array
         Kokkos::fence();
         PetscCallVoid(ISRestoreIndices(iscol_o, &iscol_o_indices_ptr));
      }

      // We can now create the off-diagonal component
      MatCreateSubMatrix_Seq_kokkos(&mat_nonlocal, is_row_d_d, is_col_o_d, reuse_int, &output_mat_nonlocal);

      // If it's our first time through we have to create our output matrix
      if (!reuse_int)
      {
         // Copy the garray output to the host
         PetscInt *garray_host = NULL; 
         PetscCallVoid(PetscMalloc1(garray_output_d.extent(0), &garray_host));
         PetscIntKokkosViewHost colmap_output_h = PetscIntKokkosViewHost(garray_host, garray_output_d.extent(0));
         // Copy the garray output to the host
         Kokkos::deep_copy(colmap_output_h, garray_output_d);
         bytes = colmap_output_h.extent(0) * sizeof(PetscInt);
         PetscCallVoid(PetscLogGpuToCpu(bytes));
         
         // We can now create our MPI matrix
         PetscCallVoid(MatCreateMPIAIJWithSeqAIJ(MPI_COMM_MATRIX, global_rows_row, global_cols_col, output_mat_local, output_mat_nonlocal, garray_host, output_mat));

         // ~~~~~~~~~~~~~~
         // If this is the first time through, we need to store the iscol_o in the output_mat
         // We don't store the is_row_d_d or is_col_d_d like the host version does as they're super cheap to rebuild
         // ~~~~~~~~~~~~~~
         // Copy the is_col_o_d to the host
         PetscInt *is_col_o_host = NULL; 
         PetscCallVoid(PetscMalloc1(is_col_o_d.extent(0), &is_col_o_host));
         PetscIntKokkosViewHost is_col_o_h = PetscIntKokkosViewHost(is_col_o_host, is_col_o_d.extent(0));
         // Copy the is_col_o_d output to the host
         Kokkos::deep_copy(is_col_o_h, is_col_o_d);
         bytes = is_col_o_h.extent(0) * sizeof(PetscInt);
         PetscCallVoid(PetscLogGpuToCpu(bytes));
         // Now create an IS
         IS iscol_o;
         PetscCallVoid(ISCreateGeneral(PETSC_COMM_SELF, is_col_o_h.extent(0), is_col_o_host, PETSC_OWN_POINTER, &iscol_o));      
         // Register it with the output_mat
         PetscCallVoid(PetscObjectCompose((PetscObject)(*output_mat), "iscol_o", (PetscObject)iscol_o));
         // The ref counter is incremented by the compose
         PetscCallVoid(ISDestroy(&iscol_o));
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
// If you pass in our_level != -1 then it uses the fine/coarse indices stored in
// the per-PCAIR handle (built by set_VecISCopyLocal_kokkos_our_level) - they
// should match the passed in is_row and is_col though! Pass NULL for handle if
// our_level == -1.
PETSC_INTERN void MatCreateSubMatrix_kokkos(Mat *input_mat, IS *is_row, IS *is_col, \
                     const int reuse_int, Mat *output_mat, \
                     void *kokkos_is_views_handle, \
                     const int our_level, const int is_row_fine_int, const int is_col_fine_int)
{
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt global_col_start, global_col_end_plus_one;   
   PetscCallVoid(MatGetOwnershipRange(*input_mat, &global_row_start, &global_row_end_plus_one));  
   PetscCallVoid(MatGetOwnershipRangeColumn(*input_mat, &global_col_start, &global_col_end_plus_one)); 
   PetscInt global_rows_row, global_cols_col;
   PetscCallVoid(ISGetSize(*is_row, &global_rows_row));
   PetscCallVoid(ISGetSize(*is_col, &global_cols_col));    
   
   PetscIntKokkosView is_row_d_d, is_col_d_d;
   auto exec = PetscGetKokkosExecutionSpace();

   // If we want the input is_row and is_col to be used
   if (our_level == -1)
   {
      // Get pointers to the indices on the host
      const PetscInt *is_row_indices_ptr, *is_col_indices_ptr;
      PetscCallVoid(ISGetIndices(*is_row, &is_row_indices_ptr));   
      PetscCallVoid(ISGetIndices(*is_col, &is_col_indices_ptr)); 

      PetscInt local_rows_row, local_cols_col;
      PetscCallVoid(ISGetLocalSize(*is_row, &local_rows_row));   
      PetscCallVoid(ISGetLocalSize(*is_col, &local_cols_col));

      // Create a host view of the existing indices
      auto is_row_view_h = PetscIntConstKokkosViewHost(is_row_indices_ptr, local_rows_row);    
      is_row_d_d = PetscIntKokkosView("is_row_d_d", local_rows_row);   
      auto is_col_view_h = PetscIntConstKokkosViewHost(is_col_indices_ptr, local_cols_col);    
      is_col_d_d = PetscIntKokkosView("is_col_d_d", local_cols_col);      
      // Copy indices to the device
      Kokkos::deep_copy(exec, is_row_d_d, is_row_view_h);
      Kokkos::deep_copy(exec, is_col_d_d, is_col_view_h);
      // Log copy with petsc
      size_t bytes = is_row_view_h.extent(0) * sizeof(PetscInt);
      PetscCallVoid(PetscLogCpuToGpu(bytes));        
      bytes = is_col_view_h.extent(0) * sizeof(PetscInt);
      PetscCallVoid(PetscLogCpuToGpu(bytes));

      // Fence after the async copies of the is_row/is_col indices to the device - ISRestoreIndices may free the host arrays
      Kokkos::fence();
      PetscCallVoid(ISRestoreIndices(*is_row, &is_row_indices_ptr));
      PetscCallVoid(ISRestoreIndices(*is_col, &is_col_indices_ptr));

      // ~~~~~~~~~~~~
      // Rewrite to local indices
      // ~~~~~~~~~~~~     
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, is_row_d_d.extent(0)), KOKKOS_LAMBDA(PetscInt i) {      

            is_row_d_d(i) -= global_row_start; // Make local
      });

      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, is_col_d_d.extent(0)), KOKKOS_LAMBDA(PetscInt i) {

            is_col_d_d(i) -= global_col_start; // Make local
      });
      Kokkos::fence(); 
   }
   // Instead if we tell the routine that the is_row and is_col are fine/coarse local indices
   // that already are on the device
   else
   {
      is_row_d_d = VecISCopyLocal_kokkos_get_view(kokkos_is_views_handle, our_level, is_row_fine_int);
      is_col_d_d = VecISCopyLocal_kokkos_get_view(kokkos_is_views_handle, our_level, is_col_fine_int);
   }

   MatCreateSubMatrix_kokkos_view(input_mat, is_row_d_d, global_rows_row, is_col_d_d, global_cols_col, reuse_int, output_mat);

   return;
}
//------------------------------------------------------------------------------------------------------------------------

// Checks whether the fine-fine block Aff of a square matrix is only a diagonal, without extracting it
// diag_only is 1 if on every rank every fine row has its diagonal stored and no other entry
// in a fine column (local or off-processor), which is what MatGetDiagonalOnly_c checks on the
// extracted Aff (stored entries, whatever their value)
// The fine points of our_level are read from the per-PCAIR IS views, as MatCreateSubMatrix_kokkos does
PETSC_INTERN void mat_fine_block_diagonal_only_kokkos(Mat *input_mat, void *kokkos_is_views_handle, \
                  const int our_level, int *diag_only)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, rows_ao, cols_ao = 0;
   MatType mat_type;
   auto exec = PetscGetKokkosExecutionSpace();

   PetscCallVoid(MatGetType(*input_mat, &mat_type));
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;
   PetscCallVoid(PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX));
   PetscCallVoid(MatGetLocalSize(*input_mat, &local_rows, &local_cols));

   Mat mat_local = NULL, mat_nonlocal = NULL;
   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*input_mat, &mat_local, &mat_nonlocal, NULL));
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao));
   }
   else
   {
      mat_local = *input_mat;
   }

   // The local indices of the fine points, which are both rows and columns as the matrix is square
   PetscIntKokkosView is_fine_d = VecISCopyLocal_kokkos_get_view(kokkos_is_views_handle, our_level, 1);
   const PetscInt local_fine = is_fine_d.extent(0);

   // Mark the local fine columns
   PetscIntKokkosView fine_col_d(Kokkos::view_alloc(exec, "fine_col_d"), local_cols);
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, local_fine), KOKKOS_LAMBDA(PetscInt i) {
         fine_col_d(is_fine_d(i)) = 1;
   });

   // Get the marks for the off-processor columns through the matrix mult SF
   PetscIntKokkosView fine_col_o_d;
   if (mpi)
   {
      PetscSF sf = NULL;
      PetscCallVoid(MatGetMultPetscSF(*input_mat, &sf));
      fine_col_o_d = PetscIntKokkosView("fine_col_o_d", cols_ao);
      // While Begin/End is in flight, do not touch the send/recv buffers
      PetscCallVoid(sf_sync_before_begin_kokkos());
      PetscCallVoid(PetscSFBcastWithMemTypeBegin(sf, MPIU_INT,
                  PETSC_MEMTYPE_KOKKOS, fine_col_d.data(),
                  PETSC_MEMTYPE_KOKKOS, fine_col_o_d.data(),
                  MPI_REPLACE));
      PetscCallVoid(PetscSFBcastEnd(sf, MPIU_INT, fine_col_d.data(), fine_col_o_d.data(), MPI_REPLACE));
      PetscCallVoid(sf_sync_after_end_kokkos());
   }

   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, NULL, &mtype));

   // Count the fine rows that break the diagonal pattern: a fine row without its diagonal, or
   // with any other entry in a fine column
   // Like the extraction of Aff, this assumes the row and column layouts of the matrix are the same
   PetscInt bad_rows = 0;
   Kokkos::parallel_reduce("FineBlockDiagonalOnly", Kokkos::RangePolicy<>(exec, 0, local_fine),
      KOKKOS_LAMBDA(const PetscInt i, PetscInt &thread_bad) {
         const PetscInt row = is_fine_d(i);
         bool found_diag = false, other = false;
         for (PetscInt k = device_local_i[row]; k < device_local_i[row + 1]; k++)
         {
            const PetscInt col = device_local_j[k];
            if (!fine_col_d(col)) continue;
            if (col == row) found_diag = true;
            else other = true;
         }
         if (mpi)
         {
            for (PetscInt k = device_nonlocal_i[row]; k < device_nonlocal_i[row + 1]; k++)
            {
               if (fine_col_o_d(device_nonlocal_j[k])) other = true;
            }
         }
         if (!found_diag || other) thread_bad++;
      }, Kokkos::Sum<PetscInt>(bad_rows));

   // Every rank has to agree
   int diag_only_local = bad_rows == 0;
   PetscCallMPIAbort(MPI_COMM_MATRIX, MPI_Allreduce(&diag_only_local, diag_only, 1, MPI_INT, MPI_LAND, MPI_COMM_MATRIX));
}
