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

using DefaultExecutionSpace = Kokkos::DefaultExecutionSpace;
using DefaultMemorySpace    = Kokkos::DefaultExecutionSpace::memory_space;
using HostMirrorMemorySpace = Kokkos::DualView<PetscScalar *>::host_mirror_space::memory_space;
using PetscIntConstKokkosViewHost = Kokkos::View<const PetscInt *, HostMirrorMemorySpace>;
using intKokkosViewHost = Kokkos::View<int *, HostMirrorMemorySpace>;
using intKokkosView = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;
using boolKokkosView = Kokkos::View<bool *, Kokkos::DefaultExecutionSpace>;
using ConstMatRowMapKokkosView = KokkosCsrGraph::row_map_type::const_type;

// Create views using scratch memory space
using ScratchSpace = typename KokkosTeamMemberType::scratch_memory_space;
using ScratchIntView = Kokkos::View<PetscInt*, ScratchSpace, Kokkos::MemoryUnmanaged>;
using ScratchScalarView = Kokkos::View<PetscScalar*, ScratchSpace, Kokkos::MemoryUnmanaged>;
using Scratch2DIntView = Kokkos::View<PetscInt**, ScratchSpace, Kokkos::MemoryUnmanaged>;
using Scratch2DScalarView = Kokkos::View<PetscScalar**, ScratchSpace, Kokkos::MemoryUnmanaged>;

PETSC_INTERN void mat_duplicate_copy_plus_diag_kokkos(Mat *, int, Mat *);
PETSC_INTERN void rewrite_j_global_to_local(PetscInt, PetscInt&, PetscIntKokkosView, PetscInt**);

// Define array of shared pointers representing fine and coarse IS's 
// on each level on the device
using ViewPtr = std::shared_ptr<PetscIntKokkosView>;
extern ViewPtr* IS_fine_views_local;
extern ViewPtr* IS_coarse_views_local;
extern int max_levels;

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
      dest.count += src.count;
      dest.found_diagonal |= src.found_diagonal;
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

#endif