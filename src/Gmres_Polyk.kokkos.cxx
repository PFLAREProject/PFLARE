// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>
#include <vector>
#include <KokkosSparse_SortCrs.hpp>

//------------------------------------------------------------------------------------------------------------------------

// Row kernel for mat_mult_powers_share_sparsity_kokkos
// Each team handles team_size rows, one thread per row, with vector lanes over the
// entries in the row of the sparsity mat
// Each thread keeps the current power of its row in per-thread scratch and computes the next
// power into a second scratch row, swapping the two after each power
// Each lane computes only its own entry l of the next power, summing over the entries j of
// the row in order, so there are no atomics and the summation order is fixed
struct PowersShareSparsityFunctor {

   PetscInt nrows = 0;
   bool mpi = false;
   PetscInt cols_ad = 0, max_nnz = 0;
   int poly_sparsity_order = 0, poly_order = 0, scratch_level = 0;

   // The sparsity mat
   const PetscInt *sp_l_i = nullptr, *sp_l_j = nullptr, *sp_nl_i = nullptr, *sp_nl_j = nullptr;
   Kokkos::View<const PetscScalar *> sp_l_a, sp_nl_a;
   // The input mat
   const PetscInt *in_l_i = nullptr, *in_l_j = nullptr, *in_nl_i = nullptr, *in_nl_j = nullptr;
   Kokkos::View<const PetscScalar *> in_l_a, in_nl_a;
   // The nonlocal rows of the input mat we need, with columns in the submat column space
   // [0..cols_ad-1 for local cols; cols_ad+k for sparsity colmap[k]]
   const PetscInt *sub_i = nullptr, *sub_j = nullptr;
   Kokkos::View<const PetscScalar *> sub_a;
   // The output mat, which we add into
   const PetscInt *out_l_i = nullptr, *out_l_j = nullptr, *out_nl_i = nullptr;
   Kokkos::View<PetscScalar *> out_l_a, out_nl_a;

   // For each entry k of the sparsity colmap, the nonlocal column of the input mat with the
   // same global column, or -1 if the input mat doesn't have it
   PetscIntKokkosView found_diag_row_d, submat_col_to_input_nonlocal_col_d;
   PetscScalarKokkosView coefficients_d;

   // Finds key in the sorted j[start..end), returns false if it isn't there
   // Checks the first and last entries before searching, as most lookups miss
   KOKKOS_INLINE_FUNCTION static bool find_col(const PetscInt *j, PetscInt start, PetscInt end, const PetscInt key,
                                               PetscInt &pos)
   {
      if (start == end || key < j[start] || key > j[end - 1]) return false;
      while (start < end)
      {
         const PetscInt mid = start + (end - start) / 2;
         if (j[mid] < key) start = mid + 1;
         else end = mid;
      }
      pos = start;
      return j[start] == key;
   }

   // Finds the entry of the input mat in row row_of_col (a local row if row_local, otherwise a
   // row of submat) and column col (in the submat column space)
   // col_input is col as a nonlocal column of the input mat (or -1 if the input mat doesn't
   // have it), only used when col is nonlocal
   // Returns false if there is no such entry
   KOKKOS_INLINE_FUNCTION bool input_entry(const bool row_local, const PetscInt row_of_col, const PetscInt col,
                                           const PetscInt col_input, PetscScalar &val) const
   {
      PetscInt pos = 0;
      if (row_local)
      {
         if (col < cols_ad)
         {
            if (!find_col(in_l_j, in_l_i[row_of_col], in_l_i[row_of_col + 1], col, pos)) return false;
            val = in_l_a(pos);
         }
         else
         {
            if (col_input < 0) return false;
            if (!find_col(in_nl_j, in_nl_i[row_of_col], in_nl_i[row_of_col + 1], col_input, pos)) return false;
            val = in_nl_a(pos);
         }
      }
      else
      {
         if (!find_col(sub_j, sub_i[row_of_col], sub_i[row_of_col + 1], col, pos)) return false;
         val = sub_a(pos);
      }
      return true;
   }

   KOKKOS_INLINE_FUNCTION void operator()(const KokkosTeamMemberType &t) const
   {
      // Each thread gets its own scratch, which we reuse for every row it handles
      ScratchScalarView vals_a(t.thread_scratch(scratch_level), max_nnz);
      ScratchScalarView vals_b(t.thread_scratch(scratch_level), max_nnz);

      const PetscInt row_start = static_cast<PetscInt>(t.league_rank()) * t.team_size();
      const PetscInt row_end = row_start + t.team_size() < nrows ? row_start + t.team_size() : nrows;

      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, row_start, row_end), [&](const PetscInt i) {

         // How many local columns do we have in row i
         const PetscInt local_cols_row_i = sp_l_i[i + 1] - sp_l_i[i];
         // ncols_row_i is the total number of columns in this row of the sparsity mat
         PetscInt ncols_row_i = local_cols_row_i;
         if (mpi) ncols_row_i += sp_nl_i[i + 1] - sp_nl_i[i];

         // Fill the first power with this row of the sparsity mat
         Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, ncols_row_i), [&](const PetscInt j) {
            if (j < local_cols_row_i)
            {
               vals_a[j] = sp_l_a(sp_l_i[i] + j);
            }
            // Nonlocal part
            else
            {
               vals_a[j] = sp_nl_a(sp_nl_i[i] + (j - local_cols_row_i));
            }
         });

         // Loop over any matrix powers
         // vals_prev holds A^(term-1) for this row and we compute A^term into vals_next
         // restricted to the sparsity of row i
         ScratchScalarView vals_prev = vals_a, vals_next = vals_b;
         for (int term = poly_sparsity_order + 1; term <= poly_order; term++)
         {
            const PetscScalar coeff_term = coefficients_d(term);

            Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, ncols_row_i), [&](const PetscInt l) {

               // The column of entry l, in the submat column space
               const PetscInt col_l = l < local_cols_row_i ? sp_l_j[sp_l_i[i] + l] : \
                        sp_nl_j[sp_nl_i[i] + (l - local_cols_row_i)] + cols_ad;
               // The same column as a nonlocal column of the input mat, if col_l is nonlocal
               const PetscInt col_input_l = col_l < cols_ad ? -1 : submat_col_to_input_nonlocal_col_d(col_l - cols_ad);

               // Sum over the entries j of row i in order
               // A^term(i, col_l) = sum_j A^(term-1)(i, col_j) * A(col_j, col_l)
               PetscScalar sum = 0;
               for (PetscInt j = 0; j < ncols_row_i; j++)
               {
                  // The row of column j, which we either have locally or have retrieved
                  // from other ranks in submat
                  const bool row_of_col_j_local = j < local_cols_row_i;
                  const PetscInt row_of_col_j = row_of_col_j_local ? sp_l_j[sp_l_i[i] + j] : \
                           sp_nl_j[sp_nl_i[i] + (j - local_cols_row_i)];
                  PetscScalar val_target;
                  if (input_entry(row_of_col_j_local, row_of_col_j, col_l, col_input_l, val_target)) sum += vals_prev[j] * val_target;
               }
               vals_next[l] = sum;

               // Skip any zero coefficients
               // Still have to compute the A^term above though for the next time through
               if (coeff_term != 0.0)
               {
                  // If we're in the local part of the matrix
                  if (l < local_cols_row_i)
                  {
                     // We need to increment the index we access by one
                     // if we don't have a diagonal in the sparsity matrix
                     // as we have one in the output_mat
                     PetscInt diag_increm = 0;
                     if (found_diag_row_d(i) == 0 && out_l_j[out_l_i[i] + l] >= i)
                     {
                        diag_increm = 1;
                     }
                     out_l_a(out_l_i[i] + l + diag_increm) += coeff_term * sum;
                  }
                  // Nonlocal part
                  else
                  {
                     out_nl_a(out_nl_i[i] + (l - local_cols_row_i)) += coeff_term * sum;
                  }
               }
            });

            // Swap the two rows for the next power
            const ScratchScalarView swap = vals_prev;
            vals_prev = vals_next;
            vals_next = swap;
         }
      });
   }
};

//------------------------------------------------------------------------------------------------------------------------

// The rows of a square matrix (a raw shell, with matching row and column layouts) for each of its
// nonlocal columns, in garray order, with the columns in the submat column space: [0, cols_ad) for
// the local columns, cols_ad + k for garray[k], entries in any other column dropped and each row
// sorted. That is what MatCreateSubMatrices gives with the rows garray and the columns [the local
// columns, garray], which the powers kernel below looks entries up in
// The values are copied, so they are exactly those of the matrix
static void fetch_ghost_rows_kokkos(const AIJDeviceAccess &acc, Kokkos::View<PetscInt *> &sub_i, \
                  Kokkos::View<PetscInt *> &sub_j, Kokkos::View<PetscScalar *> &sub_a)
{
   auto exec = PetscGetKokkosExecutionSpace();
   const PetscInt nrows = acc.local_rows, cols_ao = acc.cols_ao, cols_ad = acc.local_cols;
   const PetscInt col_start = acc.col_start, col_end = acc.col_end;
   const PetscInt nnz_l = acc.nnz_local, nnz_nl = acc.nnz_nonlocal;
   const PetscInt *l_i = acc.local_i, *l_j = acc.local_j, *nl_i = acc.nonlocal_i, *nl_j = acc.nonlocal_j;
   auto l_a = acc.local_vals, nl_a = acc.nonlocal_vals;
   PetscSF sf = acc.sf;

   // The garray on the device (sorted)
   PetscIntKokkosView garray_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "garray_d"), cols_ao);
   if (cols_ao > 0)
   {
      Kokkos::deep_copy(exec, garray_d, PetscIntConstKokkosViewHost(acc.colmap, cols_ao));
      PetscCallAbort(PETSC_COMM_SELF, PetscLogCpuToGpu(cols_ao * sizeof(PetscInt)));
   }

   // ~~~~~~~~~~~~
   // The length of each local row and where it starts in the packed rows (each row's local
   // entries then its nonlocal entries), sent to the ranks that have that row as a nonlocal column
   // The halo SF has a root for each local column (= local row) and a leaf for each nonlocal column
   // ~~~~~~~~~~~~
   PetscIntKokkosView root_len(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "root_len"), nrows);
   PetscIntKokkosView root_off(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "root_off"), nrows);
   Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, nrows), KOKKOS_LAMBDA(const PetscInt r) {
      root_len(r) = (l_i[r + 1] - l_i[r]) + (nl_i[r + 1] - nl_i[r]);
      root_off(r) = l_i[r] + nl_i[r];
   });
   PetscIntKokkosView leaf_len(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "leaf_len"), cols_ao);
   PetscIntKokkosView leaf_off(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "leaf_off"), cols_ao);
   PetscCallAbort(PETSC_COMM_SELF, sf_sync_before_begin_kokkos());
   PetscCallAbort(PETSC_COMM_SELF, PetscSFBcastWithMemTypeBegin(sf, MPIU_INT, PETSC_MEMTYPE_KOKKOS, root_len.data(), \
                  PETSC_MEMTYPE_KOKKOS, leaf_len.data(), MPI_REPLACE));
   PetscCallAbort(PETSC_COMM_SELF, PetscSFBcastWithMemTypeBegin(sf, MPIU_INT, PETSC_MEMTYPE_KOKKOS, root_off.data(), \
                  PETSC_MEMTYPE_KOKKOS, leaf_off.data(), MPI_REPLACE));

   // Pack the local rows: global columns and values
   const PetscInt nnz_packed = nnz_l + nnz_nl;
   PetscIntKokkosView packed_col(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "packed_col"), nnz_packed);
   PetscScalarKokkosView packed_val(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "packed_val"), nnz_packed);
   Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, nrows), KOKKOS_LAMBDA(const PetscInt r) {
      PetscInt pos = l_i[r] + nl_i[r];
      for (PetscInt k = l_i[r]; k < l_i[r + 1]; k++)
      {
         packed_col(pos) = l_j[k] + col_start;
         packed_val(pos) = l_a(k);
         pos++;
      }
      for (PetscInt k = nl_i[r]; k < nl_i[r + 1]; k++)
      {
         packed_col(pos) = garray_d(nl_j[k]);
         packed_val(pos) = nl_a(k);
         pos++;
      }
   });

   PetscCallAbort(PETSC_COMM_SELF, PetscSFBcastEnd(sf, MPIU_INT, root_len.data(), leaf_len.data(), MPI_REPLACE));
   PetscCallAbort(PETSC_COMM_SELF, PetscSFBcastEnd(sf, MPIU_INT, root_off.data(), leaf_off.data(), MPI_REPLACE));
   PetscCallAbort(PETSC_COMM_SELF, sf_sync_after_end_kokkos());

   // ~~~~~~~~~~~~
   // An SF over the entries: a leaf for each entry of each fetched row, whose root is that entry in
   // the owner's packed rows (the graph is built on the host from the halo SF's)
   // ~~~~~~~~~~~~
   auto leaf_len_h = Kokkos::create_mirror_view(leaf_len);
   auto leaf_off_h = Kokkos::create_mirror_view(leaf_off);
   Kokkos::deep_copy(exec, leaf_len_h, leaf_len);
   Kokkos::deep_copy(exec, leaf_off_h, leaf_off);
   exec.fence();
   PetscCallAbort(PETSC_COMM_SELF, PetscLogGpuToCpu(2 * cols_ao * sizeof(PetscInt)));

   PetscInt nroots_h, nleaves_h;
   const PetscInt *ilocal_h;
   const PetscSFNode *iremote_h;
   PetscCallAbort(PETSC_COMM_SELF, PetscSFGetGraph(sf, &nroots_h, &nleaves_h, &ilocal_h, &iremote_h));
   PetscCheckAbort(nleaves_h == cols_ao && nroots_h == nrows && acc.row_start == col_start, PETSC_COMM_SELF, \
                  PETSC_ERR_ARG_WRONG, "fetch_ghost_rows_kokkos needs a square matrix with matching row and column layouts");
   // The fetched rows' starts, in garray (leaf) order
   auto full_i_h = Kokkos::View<PetscInt *, Kokkos::HostSpace>("full_i_h", cols_ao + 1);
   full_i_h(0) = 0;
   for (PetscInt p = 0; p < cols_ao; p++) full_i_h(p + 1) = full_i_h(p) + leaf_len_h(p);
   const PetscInt n_entries = full_i_h(cols_ao);
   // The rank owning each leaf (fetched row)
   std::vector<PetscMPIInt> leaf_rank(cols_ao);
   for (PetscInt k = 0; k < nleaves_h; k++) leaf_rank[ilocal_h ? ilocal_h[k] : k] = static_cast<PetscMPIInt>(iremote_h[k].rank);
   PetscSFNode *iremote_e;
   PetscCallAbort(PETSC_COMM_SELF, PetscMalloc1(n_entries, &iremote_e));
   for (PetscInt p = 0; p < cols_ao; p++)
   {
      for (PetscInt t = 0; t < leaf_len_h(p); t++)
      {
         iremote_e[full_i_h(p) + t].rank = leaf_rank[p];
         iremote_e[full_i_h(p) + t].index = leaf_off_h(p) + t;
      }
   }
   PetscSF sf_e;
   PetscCallAbort(PETSC_COMM_SELF, PetscSFCreate(acc.comm, &sf_e));
   PetscCallAbort(PETSC_COMM_SELF, PetscSFSetGraph(sf_e, nnz_packed, n_entries, NULL, PETSC_OWN_POINTER, iremote_e, PETSC_OWN_POINTER));
   PetscCallAbort(PETSC_COMM_SELF, PetscSFSetUp(sf_e));

   // Fetch the entries
   PetscIntKokkosView ent_col(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "ent_col"), n_entries);
   PetscScalarKokkosView ent_val(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "ent_val"), n_entries);
   PetscCallAbort(PETSC_COMM_SELF, sf_sync_before_begin_kokkos());
   PetscCallAbort(PETSC_COMM_SELF, PetscSFBcastWithMemTypeBegin(sf_e, MPIU_INT, PETSC_MEMTYPE_KOKKOS, packed_col.data(), \
                  PETSC_MEMTYPE_KOKKOS, ent_col.data(), MPI_REPLACE));
   PetscCallAbort(PETSC_COMM_SELF, PetscSFBcastWithMemTypeBegin(sf_e, MPIU_SCALAR, PETSC_MEMTYPE_KOKKOS, packed_val.data(), \
                  PETSC_MEMTYPE_KOKKOS, ent_val.data(), MPI_REPLACE));
   PetscCallAbort(PETSC_COMM_SELF, PetscSFBcastEnd(sf_e, MPIU_INT, packed_col.data(), ent_col.data(), MPI_REPLACE));
   PetscCallAbort(PETSC_COMM_SELF, PetscSFBcastEnd(sf_e, MPIU_SCALAR, packed_val.data(), ent_val.data(), MPI_REPLACE));
   PetscCallAbort(PETSC_COMM_SELF, sf_sync_after_end_kokkos());
   PetscCallAbort(PETSC_COMM_SELF, PetscSFDestroy(&sf_e));

   // ~~~~~~~~~~~~
   // Columns into the submat column space (-1 to drop), count the kept entries of each row,
   // then compact and sort each row
   // ~~~~~~~~~~~~
   PetscIntKokkosView full_i_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "full_i_d"), cols_ao + 1);
   Kokkos::deep_copy(exec, full_i_d, full_i_h);
   PetscCallAbort(PETSC_COMM_SELF, PetscLogCpuToGpu((cols_ao + 1) * sizeof(PetscInt)));
   Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, n_entries), KOKKOS_LAMBDA(const PetscInt e) {
      const PetscInt g = ent_col(e);
      PetscInt c = -1;
      if (g >= col_start && g < col_end) c = g - col_start;
      else
      {
         // Binary search in the sorted garray
         PetscInt lo = 0, hi = cols_ao;
         while (lo < hi)
         {
            const PetscInt mid = lo + (hi - lo) / 2;
            if (garray_d(mid) < g) lo = mid + 1;
            else hi = mid;
         }
         if (lo < cols_ao && garray_d(lo) == g) c = cols_ad + lo;
      }
      ent_col(e) = c;
   });
   // Zero initialised on exec
   sub_i = Kokkos::View<PetscInt *>(Kokkos::view_alloc(exec, "sub_i"), cols_ao + 1);
   Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, cols_ao), KOKKOS_LAMBDA(const PetscInt p) {
      PetscInt kept = 0;
      for (PetscInt e = full_i_d(p); e < full_i_d(p + 1); e++) kept += ent_col(e) >= 0 ? 1 : 0;
      sub_i(p + 1) = kept;
   });
   PetscInt nnz_sub = 0;
   Kokkos::parallel_scan(Kokkos::RangePolicy<>(exec, 0, cols_ao),
      KOKKOS_LAMBDA(const PetscInt p, PetscInt &update, const bool final) {
         update += sub_i(p + 1);
         if (final) sub_i(p + 1) = update;
      }, nnz_sub);
   sub_j = Kokkos::View<PetscInt *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "sub_j"), nnz_sub);
   sub_a = Kokkos::View<PetscScalar *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "sub_a"), nnz_sub);
   auto sub_i_l = sub_i, sub_j_l = sub_j;
   auto sub_a_l = sub_a;
   Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, cols_ao), KOKKOS_LAMBDA(const PetscInt p) {
      PetscInt pos = sub_i_l(p);
      for (PetscInt e = full_i_d(p); e < full_i_d(p + 1); e++)
      {
         if (ent_col(e) < 0) continue;
         sub_j_l(pos) = ent_col(e);
         sub_a_l(pos) = ent_val(e);
         pos++;
      }
   });
   KokkosCsrMatrix sub_csr("sub_csr", cols_ao, cols_ad + cols_ao, nnz_sub, sub_a, sub_i, sub_j);
   KokkosSparse::sort_crs_matrix(exec, sub_csr);
   exec.fence();
}

//------------------------------------------------------------------------------------------------------------------------

// Compute matrix-matrix product with fixed order sparsity but with kokkos - keeping everything on the device
// If raw_out is set (only with poly_sparsity_order 1 and no reuse of output_mat), the output is
// returned there as device views and output_mat isn't created; MatScale and MatShift are then done
// on the views with the same arithmetic
static void mat_mult_powers_share_sparsity_impl_kokkos(Mat *input_mat, const int poly_order, const int poly_sparsity_order, PetscReal *coefficients, \
               const int reuse_int_reuse_mat, Mat *reuse_mat, const int reuse_int_cmat, Mat *output_mat, RawAIJKokkos *raw_out)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols;
   PetscInt global_row_start_temp, global_row_end_plus_one_temp;
   PetscInt global_col_start_temp, global_col_end_plus_one_temp;
   PetscInt rows_ao, cols_ao, rows_ad, cols_ad, size_cols;
   MatType mat_type;
   Mat *matrix_powers, *mat_sparsity_match;
   PetscInt one = 1;
   bool deallocate_submatrices = false;

   // The input may be a raw shell (device data), read through its blocks, with its nonlocal rows
   // fetched by fetch_ghost_rows_kokkos rather than MatCreateSubMatrices
   const bool raw_in = is_raw_shell(*input_mat);
   AIJDeviceAccess in_acc;
   if (raw_in)
   {
      PetscCheckAbort(poly_sparsity_order == 1 && !reuse_int_reuse_mat, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, \
                  "A raw shell input needs sparsity order 1 and no reuse");
      aij_device_access_get(*input_mat, in_acc);
   }
   else PetscCallVoid(MatGetType(*input_mat, &mat_type));
   // Are we in parallel?
   const bool mpi = raw_in ? in_acc.mpi : strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat mat_local_sparsity = NULL, mat_nonlocal_sparsity = NULL;
   Mat mat_local_input = NULL, mat_nonlocal_input = NULL;   

   // Get the comm
   PetscCallVoid(PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX));
   PetscCallVoid(MatGetLocalSize(*input_mat, &local_rows, &local_cols));
   // This returns the global index of the local portion of the matrix
   PetscCallVoid(MatGetOwnershipRange(*input_mat, &global_row_start_temp, &global_row_end_plus_one_temp));
   PetscCallVoid(MatGetOwnershipRangeColumn(*input_mat, &global_col_start_temp, &global_col_end_plus_one_temp));
   const PetscInt global_row_start = global_row_start_temp;
   //const PetscInt global_row_end_plus_one = global_row_end_plus_one_temp;
   const PetscInt global_col_start = global_col_start_temp;
   //const PetscInt global_col_end_plus_one = global_col_end_plus_one_temp;

   auto exec = PetscGetKokkosExecutionSpace();

   // We also copy the coefficients over to the device as we need it
   PetscInt coeff_size = poly_order + 1;
   auto coefficients_h = PetscScalarKokkosViewHost(coefficients, coeff_size);
   auto coefficients_d = PetscScalarKokkosView("coefficients_d", coeff_size);
   Kokkos::deep_copy(exec, coefficients_d, coefficients_h);       
   // Log copy with petsc
   size_t bytes = coefficients_h.extent(0) * sizeof(PetscReal);
   PetscCallVoid(PetscLogCpuToGpu(bytes));
   
   // Let's build up our matrix powers
   matrix_powers = new Mat[coeff_size - 1];
   matrix_powers[0] = *input_mat;
   // Compute the matrix powers
   for (int i = 1; i < poly_sparsity_order; i++)
   {
      PetscCallVoid(MatMatMult(*input_mat, matrix_powers[i-1], \
            MAT_INITIAL_MATRIX, 1.5, &(matrix_powers[i])));
   }
   // This is the matrix whose sparsity we want to match
   mat_sparsity_match = &(matrix_powers[poly_sparsity_order - 1]);

   // Copy in the highest unconstrained power
   // Duplicate & copy the matrix, but ensure there is a diagonal present
   if (raw_out)
   {
      PetscCheckAbort(poly_sparsity_order == 1 && !reuse_int_cmat, MPI_COMM_MATRIX, PETSC_ERR_ARG_WRONG, \
                  "The raw output needs sparsity order 1 and no reuse");
      mat_duplicate_copy_plus_diag_raw_kokkos(mat_sparsity_match, *raw_out);
   }
   else
   {
      mat_duplicate_copy_plus_diag_kokkos(mat_sparsity_match, reuse_int_cmat, output_mat);
   }

   PetscInt *col_indices_off_proc_array;
   const PetscInt *colmap_mat_sparsity_match;
   IS col_indices, row_indices;
   Mat *submatrices;

   // Pull out the nonlocal parts of the input mat we need
   const PetscInt *colmap_input_mat;
   PetscInt cols_ao_input = 0;
   cols_ao = 0;
   Kokkos::View<PetscInt *> raw_sub_i, raw_sub_j;
   Kokkos::View<PetscScalar *> raw_sub_a;
   if (raw_in)
   {
      // With sparsity order 1 the sparsity mat is the input
      cols_ad = in_acc.local_cols;
      cols_ao = cols_ao_input = in_acc.cols_ao;
      colmap_input_mat = in_acc.colmap;
      colmap_mat_sparsity_match = in_acc.colmap;
      if (mpi) fetch_ghost_rows_kokkos(in_acc, raw_sub_i, raw_sub_j, raw_sub_a);
      // Freed at the end
      PetscCallVoid(PetscMalloc1(1, &col_indices_off_proc_array));
      submatrices = nullptr;
   }
   else if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*input_mat, &mat_local_input, &mat_nonlocal_input, &colmap_input_mat));

      PetscCallVoid(MatMPIAIJGetSeqAIJ(*mat_sparsity_match, &mat_local_sparsity, &mat_nonlocal_sparsity, &colmap_mat_sparsity_match));
      PetscCallVoid(MatGetSize(mat_nonlocal_sparsity, &rows_ao, &cols_ao));
      PetscCallVoid(MatGetSize(mat_local_sparsity, &rows_ad, &cols_ad));
      PetscInt rows_ao_input;
      PetscCallVoid(MatGetSize(mat_nonlocal_input, &rows_ao_input, &cols_ao_input));
      
      // We need to pull out all the columns in the sparsity mat
      // and the nonlocal rows that correspond to the nonlocal columns
      // from the input mat      
      PetscCallVoid(PetscMalloc1(cols_ad + cols_ao, &col_indices_off_proc_array));
      size_cols = cols_ad + cols_ao;
      for (PetscInt i = 0; i < cols_ad; i++)
      {
         col_indices_off_proc_array[i] = global_row_start + i;
      }
      for (PetscInt i = 0; i < cols_ao; i++)
      {
         col_indices_off_proc_array[cols_ad + i] = colmap_mat_sparsity_match[i];
      }           
      
      // Create the sequential IS we want with the cols we want (written as global indices)
      PetscCallVoid(ISCreateGeneral(PETSC_COMM_SELF, size_cols, \
                  col_indices_off_proc_array, PETSC_USE_POINTER, &col_indices));
      PetscCallVoid(ISCreateGeneral(PETSC_COMM_SELF, cols_ao, \
                  colmap_mat_sparsity_match, PETSC_USE_POINTER, &row_indices));

      PetscCallVoid(MatSetOption(*input_mat, MAT_SUBMAT_SINGLEIS, PETSC_TRUE)); 
      // Now this will be doing comms to get the non-local rows we want and returns in a sequential matrix
      if (!reuse_int_reuse_mat)
      {
         PetscCallVoid(MatCreateSubMatrices(*input_mat, one, &row_indices, &col_indices, MAT_INITIAL_MATRIX, &submatrices));
         *reuse_mat = submatrices[0];
      }
      else
      {
         submatrices = new Mat[1];
         deallocate_submatrices = true;
         submatrices[0] = *reuse_mat;
         PetscCallVoid(MatCreateSubMatrices(*input_mat, one, &row_indices, &col_indices, MAT_REUSE_MATRIX, &submatrices));
      }
      PetscCallVoid(ISDestroy(&col_indices));
      PetscCallVoid(ISDestroy(&row_indices));
   }
   // In serial
   else
   {
      submatrices = new Mat[1];
      deallocate_submatrices = true;      
      submatrices[0] = *input_mat;
      mat_local_input = *input_mat;
      mat_local_sparsity = *mat_sparsity_match;
      cols_ad = local_cols;
      PetscCallVoid(PetscMalloc1(local_rows, &col_indices_off_proc_array));
      for (PetscInt i = 0; i < local_rows; i++)
      {
         col_indices_off_proc_array[i] = i;
      }
   }

   // Get the existing output mat
   Mat mat_local_output = NULL, mat_nonlocal_output = NULL;   
   if (raw_out)
   {
      // Scale the highest constrained power, as MatScale does (KokkosBlas::scal on both blocks):
      // nothing for 1, exact zeros for 0, otherwise c * a
      const PetscScalar c_scale = coefficients[poly_sparsity_order];
      auto out_a_l = raw_out->a_l;
      if (c_scale != 1.0)
      {
         const bool scale_zero = c_scale == 0.0;
         Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, out_a_l.extent(0)), KOKKOS_LAMBDA(const PetscInt k) {
            out_a_l(k) = scale_zero ? 0.0 : c_scale * out_a_l(k);
         });
         if (mpi)
         {
            auto out_a_nl = raw_out->a_nl;
            Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, out_a_nl.extent(0)), KOKKOS_LAMBDA(const PetscInt k) {
               out_a_nl(k) = scale_zero ? 0.0 : c_scale * out_a_nl(k);
            });
         }
      }
      // No lower powers to add in with sparsity order 1
      // Add in the 0th order term, as MatShift does (each diagonal plus the shift; every row has
      // its diagonal in the local block, mat_duplicate_copy_plus_diag made sure of it)
      // MatShift does nothing for a zero shift
      const PetscScalar c_shift = coefficients[0];
      auto out_i_l = raw_out->i_l;
      auto out_j_l = raw_out->j_l;
      const PetscInt diag_offset = raw_out->row_start - raw_out->col_start;
      if (c_shift != 0.0) Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(const PetscInt i) {
         const PetscInt diag_col = i + diag_offset;
         for (PetscInt k = out_i_l(i); k < out_i_l(i + 1); k++)
         {
            if (out_j_l(k) == diag_col)
            {
               out_a_l(k) = out_a_l(k) + c_shift;
               break;
            }
         }
      });
   }
   else
   {
      if (mpi)
      {
         PetscCallVoid(MatMPIAIJGetSeqAIJ(*output_mat, &mat_local_output, &mat_nonlocal_output, NULL));
      }
      else
      {
         mat_local_output = *output_mat;
      }

      // Scale the highest constrained power
      PetscCallVoid(MatScale(*output_mat, coefficients[poly_sparsity_order]));

      // Then go backwards and add in each of the coefficients * A^order from the second highest order down
      for (int i = poly_sparsity_order-1; i > 0; i--)
      {
         PetscCallVoid(MatAXPY(*output_mat, coefficients[i], matrix_powers[i-1], SUBSET_NONZERO_PATTERN));
      }

      // Add in the 0th order term
      PetscCallVoid(MatShift(*output_mat, coefficients[0]));
   }

   // ~~~~~~~~~~~~
   // Get pointers to the i,j on the device and Kokkos views to the values
   // This should happen after all the (potentially) host matscale, mataxpy and matshift above
   // ~~~~~~~~~~~~
   const PetscInt *device_submat_i = nullptr, *device_submat_j = nullptr;
   PetscMemType mtype;
   Kokkos::View<const PetscScalar *> device_submat_vals;
   const PetscInt *device_local_i_input = nullptr, *device_local_j_input = nullptr, *device_nonlocal_i_input = nullptr, *device_nonlocal_j_input = nullptr;
   Kokkos::View<const PetscScalar *> device_local_vals_input;
   Kokkos::View<const PetscScalar *> device_nonlocal_vals_input;
   const PetscInt *device_local_i_sparsity = nullptr, *device_local_j_sparsity = nullptr, *device_nonlocal_i_sparsity = nullptr, *device_nonlocal_j_sparsity = nullptr;
   Kokkos::View<const PetscScalar *> device_local_vals_sparsity;
   Kokkos::View<const PetscScalar *> device_nonlocal_vals_sparsity;
   if (raw_in)
   {
      // The input is also the sparsity mat; in serial the submat is its local block
      device_local_i_input = device_local_i_sparsity = in_acc.local_i;
      device_local_j_input = device_local_j_sparsity = in_acc.local_j;
      device_local_vals_input = device_local_vals_sparsity = in_acc.local_vals;
      if (mpi)
      {
         device_nonlocal_i_input = device_nonlocal_i_sparsity = in_acc.nonlocal_i;
         device_nonlocal_j_input = device_nonlocal_j_sparsity = in_acc.nonlocal_j;
         device_nonlocal_vals_input = device_nonlocal_vals_sparsity = in_acc.nonlocal_vals;
         device_submat_i = raw_sub_i.data();
         device_submat_j = raw_sub_j.data();
         device_submat_vals = raw_sub_a;
      }
      else
      {
         device_submat_i = in_acc.local_i;
         device_submat_j = in_acc.local_j;
         device_submat_vals = in_acc.local_vals;
      }
   }
   else
   {
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(submatrices[0], &device_submat_i, &device_submat_j, NULL, &mtype));
   PetscCallVoid(MatSeqAIJGetKokkosView(submatrices[0], &device_submat_vals));

   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_input, &device_local_i_input, &device_local_j_input, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_input, &device_nonlocal_i_input, &device_nonlocal_j_input, NULL, &mtype));
   PetscCallVoid(MatSeqAIJGetKokkosView(mat_local_input, &device_local_vals_input));
   if (mpi) PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal_input, &device_nonlocal_vals_input));

   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_sparsity, &device_local_i_sparsity, &device_local_j_sparsity, NULL, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_sparsity, &device_nonlocal_i_sparsity, &device_nonlocal_j_sparsity, NULL, &mtype));
   PetscCallVoid(MatSeqAIJGetKokkosView(mat_local_sparsity, &device_local_vals_sparsity));
   if (mpi) PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal_sparsity, &device_nonlocal_vals_sparsity));
   }

   const PetscInt *device_local_i_output = nullptr, *device_local_j_output = nullptr, *device_nonlocal_i_output = nullptr, *device_nonlocal_j_output = nullptr;
   // Output values are accumulated into via += so we need read-write access
   Kokkos::View<PetscScalar *> device_local_vals_output;
   Kokkos::View<PetscScalar *> device_nonlocal_vals_output;
   if (raw_out)
   {
      device_local_i_output = raw_out->i_l.data();
      device_local_j_output = raw_out->j_l.data();
      device_local_vals_output = raw_out->a_l;
      if (mpi)
      {
         device_nonlocal_i_output = raw_out->i_nl.data();
         device_nonlocal_j_output = raw_out->j_nl.data();
         device_nonlocal_vals_output = raw_out->a_nl;
      }
   }
   else
   {
      PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local_output, &device_local_i_output, &device_local_j_output, NULL, &mtype));
      if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal_output, &device_nonlocal_i_output, &device_nonlocal_j_output, NULL, &mtype));
      PetscCallVoid(MatSeqAIJGetKokkosView(mat_local_output, &device_local_vals_output));
      if (mpi) PetscCallVoid(MatSeqAIJGetKokkosView(mat_nonlocal_output, &device_nonlocal_vals_output));
   }

   // ~~~~~~~~~~~~~~
   // The kernel below looks up entries of the input mat in the column space of the sparsity
   // mat ("local" submat column space), which is defined as:
   //   [0..cols_ad-1] for local columns, [cols_ad..cols_ad+cols_ao-1] for sparsity colmap columns
   //
   // 1. Local columns of local rows: the local input matrix has the same local indices
   // 2. Any column of nonlocal rows: submat only has the nonlocal rows in it, and its column
   // indices match col_indices_off_proc_array given we create it with MatCreateSubMatrices
   // 3. Nonlocal columns of local rows: the nonlocal block of the input matrix, but
   // (for higher order fixed sparsity) the colmap of the input matrix is not the same
   // as the colmap of the sparsity matrix, so below we create a mapping from the sparsity
   // colmap to the nonlocal columns of the input matrix
   // ~~~~~~~~~~~~~~

   // For each entry of the sparsity colmap, the nonlocal column of the input mat with the
   // same global column, or -1 if the input mat doesn't have that column
   auto submat_col_to_input_nonlocal_col_d = PetscIntKokkosView("submat_col_to_input_nonlocal_col_d", (mpi && cols_ao > 0) ? cols_ao : 1);
   if (mpi && cols_ao > 0)
   {
      // Build the mapping on the host
      // Both colmaps are sorted, so we can do a merge-style scan
      auto submat_col_to_input_nonlocal_col_h = Kokkos::create_mirror_view(submat_col_to_input_nonlocal_col_d);
      PetscInt input_colmap_idx = 0;
      for (PetscInt k = 0; k < cols_ao; k++)
      {
         const PetscInt global_col = colmap_mat_sparsity_match[k];
         // Advance the input colmap index (both are sorted)
         while (input_colmap_idx < cols_ao_input && colmap_input_mat[input_colmap_idx] < global_col)
         {
            input_colmap_idx++;
         }
         if (input_colmap_idx < cols_ao_input && colmap_input_mat[input_colmap_idx] == global_col)
         {
            submat_col_to_input_nonlocal_col_h(k) = input_colmap_idx;
         }
         else
         {
            submat_col_to_input_nonlocal_col_h(k) = -1;
         }
      }
      Kokkos::deep_copy(exec, submat_col_to_input_nonlocal_col_d, submat_col_to_input_nonlocal_col_h);
      // Log copy with petsc
      bytes = submat_col_to_input_nonlocal_col_h.extent(0) * sizeof(PetscInt);
      PetscCallVoid(PetscLogCpuToGpu(bytes));
      // Fence after the async copy of the mapping to the device - the host mirror is freed at scope exit
      Kokkos::fence();
   }

   // ~~~~~~~~~~~~~~
   // Find maximum non-zeros per row for sizing scratch memory
   // ~~~~~~~~~~~~~~
   PetscInt sparsity_max_nnz = 0, sparsity_max_nnz_local = 0, sparsity_max_nnz_nonlocal = 0;
   if (local_rows > 0) {        
      // Also consider sparsity matrix row width if needed
      Kokkos::parallel_reduce("FindMaxNNZSparsity", Kokkos::RangePolicy<>(exec, 0, local_rows),
         KOKKOS_LAMBDA(const PetscInt i, PetscInt& thread_max) {
            PetscInt row_nnz = device_local_i_sparsity[i + 1] - device_local_i_sparsity[i];
            thread_max = (row_nnz > thread_max) ? row_nnz : thread_max;
         },
         Kokkos::Max<PetscInt>(sparsity_max_nnz_local)
      );
      if (mpi)
      {
         Kokkos::parallel_reduce("FindMaxNNZSparsityNonLocal", Kokkos::RangePolicy<>(exec, 0, local_rows),
            KOKKOS_LAMBDA(const PetscInt i, PetscInt& thread_max) {
               PetscInt row_nnz = device_nonlocal_i_sparsity[i + 1] - device_nonlocal_i_sparsity[i];
               thread_max = (row_nnz > thread_max) ? row_nnz : thread_max;
            },
            Kokkos::Max<PetscInt>(sparsity_max_nnz_nonlocal)
         );
      }  
      sparsity_max_nnz = sparsity_max_nnz_local + sparsity_max_nnz_nonlocal; 
   }

   // ~~~~~~~~~~~~~
   // Now we have to be careful 
   // mat_sparsity_match may not have diagonal entries in some rows
   // but we know our gmres polynomial inverse must 
   // and after calling mat_duplicate_copy_plus_diag_kokkos we know our output_mat
   // has diagonals
   // But when we write out to output_mat below, we assume it has the same sparsity as 
   // mat_sparsity_match
   // So we have to track which rows don't have diagonals in mat_sparsity_match
   // so we can increment the writing out by one in output_mat
   // The reason we don't have to do this in the cpu version is because we are calling
   // matsetvalues to write out to output_mat, rather than writing out to the csr directly
   // ~~~~~~~~~~~~~

   auto found_diag_row_d = PetscIntKokkosView("found_diag_row_d", local_rows);    
   Kokkos::deep_copy(exec, found_diag_row_d, 0);

   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

      // Row
      const PetscInt i = t.league_rank();
      // ncols_local
      const PetscInt ncols_local = device_local_i_sparsity[i + 1] - device_local_i_sparsity[i];
      const PetscInt row_index_global = i + global_row_start;
      
      // Loop over all the columns in this row of sparsity mat
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

         // Is this column the diagonal
         const bool is_diagonal = (device_local_j_sparsity[device_local_i_sparsity[i] + j] + global_col_start == row_index_global);         
         // This will only happen on a max of one thread per row
         if (is_diagonal) found_diag_row_d(i) = 1;

      });
   });
   
   // ~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~

   // ~~~~~~~~~~~~~
   // Each team handles team_size rows, one thread per row, with vector lanes over the entries
   // The vector length comes from the average row length of the sparsity mat (local plus nonlocal)
   // The host backends run the vector lanes serially, so there it makes no difference
   // ~~~~~~~~~~~~~
   if (local_rows > 0)
   {
      PowersShareSparsityFunctor functor;
      functor.nrows = local_rows;
      functor.mpi = mpi;
      functor.cols_ad = cols_ad;
      functor.max_nnz = sparsity_max_nnz;
      functor.poly_sparsity_order = poly_sparsity_order;
      functor.poly_order = poly_order;
      functor.sp_l_i = device_local_i_sparsity;
      functor.sp_l_j = device_local_j_sparsity;
      functor.sp_l_a = device_local_vals_sparsity;
      functor.in_l_i = device_local_i_input;
      functor.in_l_j = device_local_j_input;
      functor.in_l_a = device_local_vals_input;
      functor.sub_i = device_submat_i;
      functor.sub_j = device_submat_j;
      functor.sub_a = device_submat_vals;
      functor.out_l_i = device_local_i_output;
      functor.out_l_j = device_local_j_output;
      functor.out_l_a = device_local_vals_output;
      if (mpi)
      {
         functor.sp_nl_i = device_nonlocal_i_sparsity;
         functor.sp_nl_j = device_nonlocal_j_sparsity;
         functor.sp_nl_a = device_nonlocal_vals_sparsity;
         functor.in_nl_i = device_nonlocal_i_input;
         functor.in_nl_j = device_nonlocal_j_input;
         functor.in_nl_a = device_nonlocal_vals_input;
         functor.out_nl_i = device_nonlocal_i_output;
         functor.out_nl_a = device_nonlocal_vals_output;
      }
      functor.found_diag_row_d = found_diag_row_d;
      functor.submat_col_to_input_nonlocal_col_d = submat_col_to_input_nonlocal_col_d;
      functor.coefficients_d = coefficients_d;

      // As MatGetInfo's nz_used (MAT_LOCAL): both blocks
      double nz_used;
      if (raw_in) nz_used = static_cast<double>(in_acc.nnz_local + in_acc.nnz_nonlocal);
      else
      {
         MatInfo info;
         PetscCallVoid(MatGetInfo(*mat_sparsity_match, MAT_LOCAL, &info));
         nz_used = info.nz_used;
      }
      const double avg_row_nnz = nz_used / local_rows;
      int vector_length = 1;
      while (2 * vector_length <= avg_row_nnz && vector_length < 32) vector_length *= 2;
      vector_length = std::min(vector_length, Kokkos::TeamPolicy<>::vector_length_max());

      // Two scratch rows of the longest row per thread
      const size_t per_thread = 2 * ScratchScalarView::shmem_size(sparsity_max_nnz);
      Kokkos::TeamPolicy<> probe(exec, 1, 1, vector_length);
      int team_size = std::max(1, 256 / vector_length);
      team_size = std::min(team_size, probe.team_size_max(functor, Kokkos::ParallelForTag()));
      // Use level 0 scratch if it fits, shrinking the team if needed, otherwise level 1
      // with the full team
      const size_t scratch_max_0 = Kokkos::TeamPolicy<>::scratch_size_max(0);
      int team_size_0 = team_size;
      while (team_size_0 > 1 && team_size_0 * per_thread > scratch_max_0) team_size_0 /= 2;
      functor.scratch_level = team_size_0 * per_thread <= scratch_max_0 ? 0 : 1;
      if (functor.scratch_level == 0) team_size = team_size_0;

      const int league_size = static_cast<int>((local_rows + team_size - 1) / team_size);
      Kokkos::TeamPolicy<> policy(exec, league_size, team_size, vector_length);
      policy.set_scratch_size(functor.scratch_level, Kokkos::PerThread(per_thread));
      Kokkos::parallel_for(policy, functor);
   }

   Kokkos::fence();

   // Restore the read-only input/sparsity/submat views
   if (raw_in) aij_device_access_restore(in_acc);
   else
   {
      PetscCallVoid(MatSeqAIJRestoreKokkosView(submatrices[0], &device_submat_vals));
      PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local_input, &device_local_vals_input));
      if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal_input, &device_nonlocal_vals_input));
      PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local_sparsity, &device_local_vals_sparsity));
      if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal_sparsity, &device_nonlocal_vals_sparsity));
   }

   // The matching restore handles MatSeqAIJKokkosModifyDevice (clears sync state,
   // marks device modified, invalidates transpose/hermitian, bumps object state).
   if (raw_out)
   {
      exec.fence();
   }
   else
   {
      PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_local_output, &device_local_vals_output));
      if (mpi) PetscCallVoid(MatSeqAIJRestoreKokkosView(mat_nonlocal_output, &device_nonlocal_vals_output));
   }

   for (int i = 1; i < poly_sparsity_order; i++)
   {
      PetscCallVoid(MatDestroy(&(matrix_powers[i])));
   }   
   delete[] matrix_powers;
   if (deallocate_submatrices) delete[] submatrices;
   // We don't explicitly call matdestroysubmatrices because lord knows how we pass out submatrices back
   // to fortran - to prevent leaking memory we destroy the submatrices pointer ourself, and if we come back
   // into this routine with reuse we just create a new submatrices array    
   if (mpi && !reuse_int_reuse_mat) PetscCallVoid(PetscFree(submatrices));
   PetscCallVoid(PetscFree(col_indices_off_proc_array));

   return;
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN void mat_mult_powers_share_sparsity_kokkos(Mat *input_mat, const int poly_order, const int poly_sparsity_order, PetscReal *coefficients, \
               const int reuse_int_reuse_mat, Mat *reuse_mat, const int reuse_int_cmat, Mat *output_mat)
{
   mat_mult_powers_share_sparsity_impl_kokkos(input_mat, poly_order, poly_sparsity_order, coefficients, \
               reuse_int_reuse_mat, reuse_mat, reuse_int_cmat, output_mat, nullptr);
}

// mat_mult_powers_share_sparsity_kokkos with poly_sparsity_order 1 and no reuse of the output, which
// is returned in raw_handle as device views (a RawAIJKokkos) rather than a PETSc matrix
// raw_handle has to be freed (raw_free_kokkos)
PETSC_INTERN void mat_mult_powers_share_sparsity_raw_kokkos(Mat *input_mat, const int poly_order, PetscReal *coefficients, \
               const int reuse_int_reuse_mat, Mat *reuse_mat, void **raw_handle)
{
   RawAIJKokkos *raw = new RawAIJKokkos();
   mat_mult_powers_share_sparsity_impl_kokkos(input_mat, poly_order, 1, coefficients, \
               reuse_int_reuse_mat, reuse_mat, 0, NULL, raw);
   *raw_handle = raw;
}
