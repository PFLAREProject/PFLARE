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

#endif