#if !defined (KOKKOS_HELPER_DEF_H)
#define KOKKOS_HELPER_DEF_H

// petscvec_kokkos.hpp has to go first
#include <petscvec_kokkos.hpp>
#include <petscmat_kokkos.hpp>
#include "petsc.h"
#include <../src/mat/impls/aij/seq/kokkos/aijkok.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <../src/vec/vec/impls/seq/kokkos/veckokkosimpl.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <KokkosSparse_spadd.hpp>
#include <Kokkos_NestedSort.hpp>
#include <KokkosBatched_Gesv.hpp>
#include <cstdio>

struct PflareKokkosTrace {
   const char *name;
   PflareKokkosTrace(const char *n) : name(n) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      fprintf(stderr, "[PFLARE kokkos rank=%d] Entering %s\n", rank, name);
      fflush(stderr);
   }
   ~PflareKokkosTrace() {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      fprintf(stderr, "[PFLARE kokkos rank=%d] Leaving %s\n", rank, name);
      fflush(stderr);
   }
};

using DefaultExecutionSpace = Kokkos::DefaultExecutionSpace;
using DefaultMemorySpace    = Kokkos::DefaultExecutionSpace::memory_space;
using HostMirrorMemorySpace = Kokkos::DualView<PetscScalar *>::host_mirror_space::memory_space;
using PetscIntConstKokkosViewHost = Kokkos::View<const PetscInt *, HostMirrorMemorySpace>;
using intKokkosViewHost = Kokkos::View<int *, HostMirrorMemorySpace>;
using intKokkosView = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;
using boolKokkosView = Kokkos::View<bool *, Kokkos::DefaultExecutionSpace>;
using ConstMatRowMapKokkosView = KokkosCsrGraph::row_map_type::const_type;

// Create views using scratch memory space
typedef Kokkos::DefaultExecutionSpace::scratch_memory_space
  ScratchSpace;
using ScratchIntView = Kokkos::View<PetscInt*, ScratchSpace, Kokkos::MemoryUnmanaged>;
using ScratchScalarView = Kokkos::View<PetscScalar*, ScratchSpace, Kokkos::MemoryUnmanaged>;
using Scratch2DIntView = Kokkos::View<PetscInt**, ScratchSpace, Kokkos::MemoryUnmanaged>;
using Scratch2DScalarView = Kokkos::View<PetscScalar**, ScratchSpace, Kokkos::MemoryUnmanaged>;
using ViewPetscIntPtr = std::shared_ptr<PetscIntKokkosView>;

PETSC_INTERN void mat_duplicate_copy_plus_diag_kokkos(Mat *, int, Mat *);
PETSC_INTERN void mat_sync(Mat *);
PETSC_INTERN void rewrite_j_global_to_local(PetscInt, PetscInt&, PetscIntKokkosView, PetscInt**);
PETSC_INTERN void create_cf_is_device_kokkos(Mat *input_mat, const int match_cf, PetscIntKokkosView &is_local_d);
PETSC_INTERN void pmisr_existing_measure_cf_markers_kokkos(Mat *strength_mat, const int max_luby_steps, const int pmis_int, PetscScalarKokkosView &measure_local_d, intKokkosView &cf_markers_d, const int zero_measure_c_point_int);
PETSC_INTERN void pmisr_existing_measure_implicit_transpose_kokkos(Mat *strength_mat, const int max_luby_steps, const int pmis_int, PetscScalarKokkosView &measure_local_d, intKokkosView &cf_markers_d, const int zero_measure_c_point_int);
PETSC_INTERN void copy_diag_dom_ratio_d2h(PetscReal *diag_dom_ratio_local);
PETSC_INTERN void delete_device_diag_dom_ratio();

// Define array of shared pointers representing fine and coarse IS's 
// on each level on the device
extern ViewPetscIntPtr* IS_fine_views_local;
extern ViewPetscIntPtr* IS_coarse_views_local;
extern int max_levels;
extern intKokkosView cf_markers_local_d;
extern PetscScalarKokkosView diag_dom_ratio_local_d;

// ~~~~~~~~~~~~~~~~~~
// Some custom reductions we use 
// ~~~~~~~~~~~~~~~~~~

struct ReduceData {
   PetscInt count;
   bool found_diagonal;
   
   // Set count to zero and found_diagonal to false
   KOKKOS_INLINE_FUNCTION
   ReduceData() : count(0), found_diagonal(false) {}
   
   // We use this in our parallel reduction
   KOKKOS_INLINE_FUNCTION
   void operator+=(const ReduceData& src) {
      // Add all the counts
      count += src.count;
      // If we have found a diagonal entry at any point in this row
      // found_diagonal becomes true      
      found_diagonal |= src.found_diagonal;
   }

   // Required for Kokkos reduction
   KOKKOS_INLINE_FUNCTION
   static void join(volatile ReduceData& dest, const volatile ReduceData& src) {
      dest.count = dest.count + src.count;
      dest.found_diagonal = dest.found_diagonal || src.found_diagonal;
   }   
};

namespace Kokkos {
    template<>
    struct reduction_identity<ReduceData> {
        KOKKOS_INLINE_FUNCTION
        static ReduceData sum() {
            return ReduceData();  // Returns {count=0, found_diagonal=false}
        }
    };
}

struct ReduceDataMaxRow {
   PetscInt col;
   PetscReal val;
   
   // Set col to negative one and val to -1.0
   KOKKOS_INLINE_FUNCTION
   ReduceDataMaxRow() : col(-1), val(-1.0) {}
   
   // We use this in our parallel reduction to find maximum
   KOKKOS_INLINE_FUNCTION
   void operator+=(const ReduceDataMaxRow& src) {
      // If src has a larger value, take it
      if (src.val > val) {
         val = src.val;
         col = src.col;
      }
   }

   // Required for Kokkos reduction
   KOKKOS_INLINE_FUNCTION
   static void join(volatile ReduceDataMaxRow& dest, const volatile ReduceDataMaxRow& src) {
      if (src.val > dest.val) {
         dest.val = src.val;
         dest.col = src.col;
      }
   }   
};

namespace Kokkos {
    template<>
    struct reduction_identity<ReduceDataMaxRow> {
        KOKKOS_INLINE_FUNCTION
        static ReduceDataMaxRow sum() {
            return ReduceDataMaxRow();  // Returns {col=-1, val=-1}
        }
    };
}

// Binary search for target in a sorted array, returns the index or -1 if not found
template <typename ViewType>
KOKKOS_INLINE_FUNCTION
PetscInt binary_search_sorted(const ViewType &sorted_view, const PetscInt size, const PetscInt target)
{
   PetscInt lo = 0, hi = size - 1;
   while (lo <= hi)
   {
      PetscInt mid = (lo + hi) / 2;
      if (sorted_view(mid) == target)
         return mid;
      else if (sorted_view(mid) < target)
         lo = mid + 1;
      else
         hi = mid - 1;
   }
   return -1;
}

// Check that every entry in cf_markers_d is either -1 (F) or 1 (C).
// Calls MPI_Abort if any local point is not marked.
inline void check_cf_markers_all_marked_kokkos(
   const intKokkosView &cf_markers_d,
   const PetscInt local_rows,
   MPI_Comm MPI_COMM_MATRIX)
{
   auto exec = PetscGetKokkosExecutionSpace();
   PetscInt bad_count = 0;
   Kokkos::parallel_reduce(
      "check_cf_markers",
      Kokkos::RangePolicy<>(exec, 0, local_rows),
      KOKKOS_LAMBDA(const PetscInt i, PetscInt &count) {
         if (cf_markers_d(i) != -1 && cf_markers_d(i) != 1) count++;
      }, bad_count);
   Kokkos::fence();
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_MATRIX, &rank);
   if (bad_count > 0) {
      fprintf(stderr,
         "[PFLARE kokkos rank=%d] ERROR check_cf_markers_all_marked_kokkos: "
         "%d / %d local points are NOT marked F or C\n",
         rank, (int)bad_count, (int)local_rows);
      fflush(stderr);
      MPI_Abort(MPI_COMM_MATRIX, 1);
   } else {
      fprintf(stderr,
         "[PFLARE kokkos rank=%d] check_cf_markers_all_marked_kokkos: "
         "all %d local points marked F or C OK\n",
         rank, (int)local_rows);
      fflush(stderr);
   }
}

// Check that is_fine_local_d and is_coarse_local_d together cover every local
// point [0, local_rows-1] exactly once (no missing, no duplicates).
// Call before global-index conversion (entries are local offsets [0, local_rows-1]).
// Calls MPI_Abort if any point is missing or duplicated.
inline void check_cf_is_all_local_kokkos(
   const PetscIntKokkosView &is_fine_local_d,
   const PetscIntKokkosView &is_coarse_local_d,
   const PetscInt local_rows,
   MPI_Comm MPI_COMM_MATRIX)
{
   auto exec = PetscGetKokkosExecutionSpace();
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_MATRIX, &rank);

   // Allocate hit-count array, initialised to 0
   intKokkosView hit_count("hit_count", local_rows);
   Kokkos::deep_copy(exec, hit_count, 0);

   // Mark each fine index (atomic to catch duplicates within the fine set)
   Kokkos::parallel_for(
      "check_cf_is_mark_fine",
      Kokkos::RangePolicy<>(exec, 0, (PetscInt)is_fine_local_d.extent(0)),
      KOKKOS_LAMBDA(const PetscInt i) {
         const PetscInt idx = is_fine_local_d(i);
         if (idx >= 0 && idx < local_rows)
            Kokkos::atomic_add(&hit_count(idx), 1);
      });

   // Mark each coarse index
   Kokkos::parallel_for(
      "check_cf_is_mark_coarse",
      Kokkos::RangePolicy<>(exec, 0, (PetscInt)is_coarse_local_d.extent(0)),
      KOKKOS_LAMBDA(const PetscInt i) {
         const PetscInt idx = is_coarse_local_d(i);
         if (idx >= 0 && idx < local_rows)
            Kokkos::atomic_add(&hit_count(idx), 1);
      });

   // Count any point not hit exactly once
   PetscInt bad_count = 0;
   Kokkos::parallel_reduce(
      "check_cf_is_count_bad",
      Kokkos::RangePolicy<>(exec, 0, local_rows),
      KOKKOS_LAMBDA(const PetscInt i, PetscInt &count) {
         if (hit_count(i) != 1) count++;
      }, bad_count);

   Kokkos::fence();

   if (bad_count > 0) {
      fprintf(stderr,
         "[PFLARE kokkos rank=%d] ERROR check_cf_is_all_local_kokkos: "
         "%d / %d local points are not covered exactly once by fine+coarse IS\n",
         rank, (int)bad_count, (int)local_rows);
      fflush(stderr);
      MPI_Abort(MPI_COMM_MATRIX, 1);
   } else {
      fprintf(stderr,
         "[PFLARE kokkos rank=%d] check_cf_is_all_local_kokkos: "
         "fine=%d coarse=%d, all %d local points covered exactly once OK\n",
         rank, (int)is_fine_local_d.extent(0), (int)is_coarse_local_d.extent(0), (int)local_rows);
      fflush(stderr);
   }
}

#endif