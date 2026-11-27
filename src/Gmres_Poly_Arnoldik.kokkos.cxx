// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <vector>

// The number of columns of an s-step block that one thread of the projection kernel updates
#define SSTEP_GROUP 4

//------------------------------------------------------------------------------------------------------------------------

// The basis vectors of the s-step Arnoldi as the columns of one dense device matrix
// The vectors made by vec_create_device_vecs_kokkos are slices of one device buffer with a fixed
// stride (the padding between them is zero and stays zero), so vecs[0] to vecs[n_vecs-1] are the
// leading columns of a column-major matrix with that stride as its number of rows
// found is false if the vectors were not made that way
// The first n_read vectors are synced to the device, and the n_write after those are also
// marked as modified on the device
static PetscErrorCode sstep_basis_view(Vec *vecs, const PetscInt n_read, const PetscInt n_write, \
               Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, DefaultMemorySpace> *basis, bool *found)
{
   PetscScalarKokkosView *buffer = NULL;
   PetscInt n;

   PetscFunctionBegin;
   *found = false;
   PetscCall(PetscObjectContainerQuery((PetscObject)vecs[0], "pflare_duplicate_vecs", &buffer));
   if (!buffer) PetscFunctionReturn(PETSC_SUCCESS);

   PetscCall(VecGetLocalSize(vecs[0], &n));
   const PetscInt64 align = PetscMax((PetscInt64)(64 / sizeof(PetscScalar)), (PetscInt64)1);
   const PetscInt64 lda = ((n + align - 1) / align) * align;

   for (PetscInt i = 0; i < n_read; i++)
   {
      ConstPetscScalarKokkosView v;
      PetscCall(VecGetKokkosView(vecs[i], &v));
      PetscCall(VecRestoreKokkosView(vecs[i], &v));
   }
   for (PetscInt i = n_read; i < n_read + n_write; i++)
   {
      PetscScalarKokkosView v;
      PetscCall(VecGetKokkosView(vecs[i], &v));
      PetscCall(VecRestoreKokkosView(vecs[i], &v));
   }

   *basis = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, DefaultMemorySpace>(buffer->data(), (size_t)lda, (size_t)(n_read + n_write));
   *found = true;
   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

// Overwrites the block of n_cols vectors after the n_old existing basis vectors in vecs with the
// block times the upper triangular R_inv (n_cols x n_cols, column-major), in one pass over the rows
// done is zero (and nothing is changed) if the vectors are not slices of one device buffer
PETSC_INTERN void sstep_apply_inverse_kokkos(Vec *vecs, const int n_old, const int n_cols, \
               PetscReal *R_inv, int *done)
{
   typedef Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, DefaultMemorySpace> DenseView;
   typedef Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::HostSpace> DenseHostView;
   DenseView basis;
   bool found;

   *done = 0;
   PetscCallVoid(sstep_basis_view(vecs, n_old, n_cols, &basis, &found));
   if (!found) return;

   auto exec = PetscGetKokkosExecutionSpace();
   DenseView R_d("sstep_R_inv", n_cols, n_cols);
   DenseHostView R_h(R_inv, n_cols, n_cols);
   Kokkos::deep_copy(R_d, R_h);

   // Going backwards through the columns, each new column only needs the old values of
   // the columns before it, so this can be done in place
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, basis.extent(0)), KOKKOS_LAMBDA(const size_t i) {
         for (int c = n_cols - 1; c >= 0; c--)
         {
            PetscScalar sum = 0.0;
            for (int l = 0; l <= c; l++) sum += basis(i, n_old + l) * R_d(l, c);
            basis(i, n_old + c) = sum;
         }
   });

   *done = 1;
   return;
}

//------------------------------------------------------------------------------------------------------------------------

// The sums of one tile of dot products in the s-step Arnoldi, as a reduction type
template <int N>
struct SStepTile {
   PetscScalar v[N];

   KOKKOS_INLINE_FUNCTION
   SStepTile() { for (int a = 0; a < N; a++) v[a] = 0.0; }

   KOKKOS_INLINE_FUNCTION
   void operator+=(const SStepTile &src) { for (int a = 0; a < N; a++) v[a] += src.v[a]; }
};

namespace Kokkos {
    template <int N>
    struct reduction_identity<SStepTile<N>> {
        KOKKOS_INLINE_FUNCTION
        static SStepTile<N> sum() { return SStepTile<N>(); }
    };
}

// D = L^T R for tall L and R, in tiles of TQ columns of L by TC columns of R
// One team does a tile, with a single reduction over the rows that reads each of its TQ + TC
// columns once for its TQ * TC dot products. One team for each entry would read two columns
// for every dot product, and the time of this is all in reading the vectors
template <int TQ, int TC, class LView, class RView, class DView>
static void sstep_tiled_dots(const LView &L, const RView &R, const DView &D_d, const int n_left, const int n_right, \
               const PetscInt n_rows)
{
   const int n_tq = (n_left + TQ - 1) / TQ;
   const int n_tc = (n_right + TC - 1) / TC;
   auto exec = PetscGetKokkosExecutionSpace();

   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(exec, n_tq * n_tc, Kokkos::AUTO()), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &t) {
         const int q0 = (t.league_rank() % n_tq) * TQ;
         const int c0 = (t.league_rank() / n_tq) * TC;
         // A tile that hangs over the edge repeats the last column, and those sums are not kept
         int q_idx[TQ], c_idx[TC];
         for (int a = 0; a < TQ; a++) q_idx[a] = (q0 + a < n_left) ? q0 + a : n_left - 1;
         for (int b = 0; b < TC; b++) c_idx[b] = (c0 + b < n_right) ? c0 + b : n_right - 1;

         SStepTile<TQ * TC> sums;
         Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(t, n_rows), [&](const PetscInt i, SStepTile<TQ * TC> &sums_loc) {
               PetscScalar l_val[TQ], r_val[TC];
               for (int a = 0; a < TQ; a++) l_val[a] = L(i, q_idx[a]);
               for (int b = 0; b < TC; b++) r_val[b] = R(i, c_idx[b]);
               for (int b = 0; b < TC; b++)
               {
                  for (int a = 0; a < TQ; a++) sums_loc.v[b * TQ + a] += l_val[a] * r_val[b];
               }
            }, sums);

         for (int b = 0; b < TC; b++)
         {
            for (int a = 0; a < TQ; a++)
            {
               if (q0 + a < n_left && c0 + b < n_right) D_d(q0 + a, c0 + b) = sums.v[b * TQ + a];
            }
         }
   });
}

//------------------------------------------------------------------------------------------------------------------------

// All the dot products of one set of basis vectors against another in one reduction, for the
// low synchronisation s-step Arnoldi: out (n_left x n_right, column-major) is
// vecs[l_start : l_start + n_left]^T vecs[r_start : r_start + n_right]
// If with_w is non-zero the squared norm of w is computed in the same reduction
// done is zero if the vectors are not slices of one device buffer
PETSC_INTERN void sstep_fused_dots_kokkos(Vec *vecs, const int l_start, const int n_left, const int r_start, \
               const int n_right, Vec *w, const int with_w, PetscReal *out, PetscReal *w_norm_sq, \
               int *done)
{
   typedef Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, DefaultMemorySpace> DenseView;
   typedef Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::HostSpace> DenseHostView;
   DenseView basis;
   bool found;
   MPI_Comm comm;
   PetscMPIInt count;

   *done = 0;
   *w_norm_sq = 0.0;
   if (with_w)
   {
      PetscBool is_kokkos;
      PetscCallVoid(PetscObjectTypeCompareAny((PetscObject)(*w), &is_kokkos, VECSEQKOKKOS, VECMPIKOKKOS, ""));
      if (!is_kokkos) return;
   }
   PetscCallVoid(sstep_basis_view(vecs, PetscMax(l_start + n_left, r_start + n_right), 0, &basis, &found));
   if (!found) return;
   PetscCallVoid(PetscObjectGetComm((PetscObject)vecs[0], &comm));

   auto L = Kokkos::subview(basis, Kokkos::ALL, std::make_pair((size_t)l_start, (size_t)(l_start + n_left)));
   auto R = Kokkos::subview(basis, Kokkos::ALL, std::make_pair((size_t)r_start, (size_t)(r_start + n_right)));

   DenseView D_d("sstep_D", n_left, n_right);
   DenseHostView D_h("sstep_D_h", n_left, n_right);
   const PetscInt n_rows = basis.extent(0);
   // Tiles of 4 by 4 were the fastest on LUMI (MI250X) of those up to 16 by 4. One with more
   // than 32 sums was slower than a dot product at a time
   sstep_tiled_dots<4, 4>(L, R, D_d, n_left, n_right, n_rows);
   Kokkos::deep_copy(D_h, D_d);

   // The local squared norm of w goes on the end of the same buffer
   const size_t n_entries = (size_t)n_left * (size_t)n_right;
   std::vector<PetscScalar> send(n_entries + 1), recv(n_entries + 1);
   for (size_t i = 0; i < n_entries; i++) send[i] = D_h.data()[i];
   send[n_entries] = 0.0;
   if (with_w)
   {
      ConstPetscScalarKokkosView w_d;
      PetscScalar sum = 0.0;
      PetscCallVoid(VecGetKokkosView(*w, &w_d));
      Kokkos::parallel_reduce(
         Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, w_d.extent(0)), KOKKOS_LAMBDA(const size_t i, PetscScalar &sum_loc) {
            sum_loc += w_d(i) * w_d(i);
         }, sum);
      PetscCallVoid(VecRestoreKokkosView(*w, &w_d));
      send[n_entries] = sum;
   }
   PetscCallVoid(PetscMPIIntCast((PetscInt)(n_entries + 1), &count));
   PetscCallMPIAbort(comm, MPI_Allreduce(send.data(), recv.data(), count, MPIU_SCALAR, MPIU_SUM, comm));
   for (size_t i = 0; i < n_entries; i++) out[i] = recv[i];
   *w_norm_sq = recv[n_entries];

   *done = 1;
   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Removes the components X (n_old x n_cols, column-major) along the n_old existing basis vectors
// from the block of n_cols vectors after them in vecs, with no reduction
// done is zero (and nothing is changed) if the vectors are not slices of one device buffer
PETSC_INTERN void sstep_project_kokkos(Vec *vecs, const int n_old, const int n_cols, \
               PetscReal *X, int *done)
{
   typedef Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, DefaultMemorySpace> DenseView;
   typedef Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::HostSpace> DenseHostView;
   DenseView basis;
   bool found;

   *done = 0;
   PetscCallVoid(sstep_basis_view(vecs, n_old, n_cols, &basis, &found));
   if (!found) return;

   auto Q = Kokkos::subview(basis, Kokkos::ALL, std::make_pair((size_t)0, (size_t)n_old));
   auto V = Kokkos::subview(basis, Kokkos::ALL, std::make_pair((size_t)n_old, (size_t)(n_old + n_cols)));

   auto exec = PetscGetKokkosExecutionSpace();
   DenseView X_d("sstep_X", n_old, n_cols);
   DenseHostView X_h(X, n_old, n_cols);
   Kokkos::deep_copy(X_d, X_h);
   const PetscInt n_rows = basis.extent(0);

   // V = V - Q X, with one thread for each row and each group of SSTEP_GROUP columns of the
   // block, which reads that row of the basis once for the group
   // On LUMI (MI250X) groups of four or eight were 1.7 times faster than a column at a time,
   // and the whole block in one thread three times slower
   const PetscInt n_groups = (n_cols + SSTEP_GROUP - 1) / SSTEP_GROUP;
   Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<PetscInt>>(exec, {0, 0}, {n_rows, n_groups}), KOKKOS_LAMBDA(const PetscInt i, const PetscInt g) {
         const int c0 = g * SSTEP_GROUP;
         const int n_c = (n_cols - c0 < SSTEP_GROUP) ? n_cols - c0 : SSTEP_GROUP;
         PetscScalar sum[SSTEP_GROUP];
         for (int b = 0; b < n_c; b++) sum[b] = 0.0;
         for (int q = 0; q < n_old; q++)
         {
            const PetscScalar q_val = Q(i, q);
            for (int b = 0; b < n_c; b++) sum[b] += q_val * X_d(q, c0 + b);
         }
         for (int b = 0; b < n_c; b++) V(i, c0 + b) -= sum[b];
   });

   *done = 1;
   return;
}
