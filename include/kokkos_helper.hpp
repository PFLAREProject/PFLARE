#if !defined (KOKKOS_HELPER_DEF_H)
#define KOKKOS_HELPER_DEF_H

// petscvec_kokkos.hpp has to go first
#include <petscvec_kokkos.hpp>
#include <petscmat_kokkos.hpp>
#include <petsc_kokkos.hpp>
#include "petsc.h"
#include <petscdevice.h>
#include <Kokkos_StdAlgorithms.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <KokkosSparse_spadd.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <Kokkos_NestedSort.hpp>
#include <KokkosBatched_Gesv.hpp>
#include <KokkosBlas2_team_gemv.hpp>
#include <petsc/private/kokkosimpl.hpp>

using DefaultExecutionSpace = Kokkos::DefaultExecutionSpace;
using DefaultMemorySpace    = Kokkos::DefaultExecutionSpace::memory_space;
using HostMirrorMemorySpace = Kokkos::DualView<PetscScalar *>::host_mirror_space::memory_space;
using PetscIntConstKokkosViewHost = Kokkos::View<const PetscInt *, HostMirrorMemorySpace>;
using intKokkosViewHost = Kokkos::View<int *, HostMirrorMemorySpace>;
using intKokkosView = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;
using boolKokkosView = Kokkos::View<bool *, Kokkos::DefaultExecutionSpace>;

// Create views using scratch memory space
typedef Kokkos::DefaultExecutionSpace::scratch_memory_space
  ScratchSpace;
using ScratchIntView = Kokkos::View<PetscInt*, ScratchSpace, Kokkos::MemoryUnmanaged>;
using ScratchScalarView = Kokkos::View<PetscScalar*, ScratchSpace, Kokkos::MemoryUnmanaged>;
using Scratch2DIntView = Kokkos::View<PetscInt**, ScratchSpace, Kokkos::MemoryUnmanaged>;
using Scratch2DScalarView = Kokkos::View<PetscScalar**, ScratchSpace, Kokkos::MemoryUnmanaged>;
using ViewPetscIntPtr = std::shared_ptr<PetscIntKokkosView>;
using KokkosTeamMemberType = Kokkos::TeamPolicy<DefaultExecutionSpace>::member_type;
using KokkosCsrMatrix = KokkosSparse::CrsMatrix<PetscScalar, PetscInt, DefaultMemorySpace, void, PetscInt>;

// The blocks of a parallel (or serial) AIJ matrix as device views, not (yet) a PETSc matrix
struct RawAIJKokkos {
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

PETSC_INTERN void mat_duplicate_copy_plus_diag_kokkos(Mat *, int, Mat *);
PETSC_INTERN void rewrite_j_global_to_local(PetscInt, PetscInt&, PetscIntKokkosView, PetscInt**);
PETSC_INTERN void create_cf_is_device_kokkos(Mat *input_mat, const int match_cf, PetscIntKokkosView &is_local_d);
PETSC_INTERN void pmisr_existing_measure_cf_markers_kokkos(Mat *strength_mat, const int max_luby_steps, const int pmis_int, PetscScalarKokkosView &measure_local_d, intKokkosView &cf_markers_d, const int zero_measure_c_point_int);
PETSC_INTERN void pmisr_existing_measure_implicit_transpose_kokkos(Mat *strength_mat, const int max_luby_steps, const int pmis_int, PetscScalarKokkosView &measure_local_d, intKokkosView &cf_markers_d, const int zero_measure_c_point_int);
PETSC_INTERN void copy_diag_dom_ratio_d2h(PetscReal *diag_dom_ratio_local);
PETSC_INTERN void delete_device_diag_dom_ratio();

// Per-PCAIR IS views (fine/coarse per multigrid level) live behind an opaque
// handle owned by the air_data on the Fortran side; see VecISCopyLocalk for
// the definition of the storage struct and accessors.
// Not PETSC_INTERN: that forces extern "C" linkage, which is incompatible with
// returning a C++ Kokkos View. Callers are all C++ (.kokkos.cxx).
PETSC_VISIBILITY_INTERNAL PetscIntKokkosView VecISCopyLocal_kokkos_get_view(void *handle, int our_level, int fine_int);
// mat_duplicate_copy_plus_diag_kokkos without reuse, returning the output as device views (PETSc_Helperk)
PETSC_VISIBILITY_INTERNAL void mat_duplicate_copy_plus_diag_raw_kokkos(Mat *input_mat, RawAIJKokkos &out);
// The global column and value of each row's entry in a one point shell prolongator
// (Grid_Transferk), false if P isn't one
PETSC_VISIBILITY_INTERNAL bool one_point_shell_targets_kokkos(Mat P, const PetscInt sentinel, PetscIntKokkosView &target_global_d, PetscScalarKokkosView &weight_d);

extern intKokkosView cf_markers_local_d;
extern PetscScalarKokkosView diag_dom_ratio_local_d;

// ~~~~~~~~~~~~~~~~~~
// Ordering PetscSF comms on Kokkos views
// PetscSF packs and unpacks on the PETSc execution space with the Kokkos SF backend, and on
// the stream of the current PETSc device context with the native CUDA/HIP backend (the
// default for CUDA/HIP once PETSc MR !9662 is in). By default both are the same stream as
// PetscGetKokkosExecutionSpace(), and then nothing extra is needed. If the current context's
// stream differs, fence the PETSc execution space before Begin (kernels writing the root or
// reading the old leaf are done) and synchronize the device context after End (the unpack
// into the leaf is done)
// This mirrors PetscSFSynchronizeKokkos_Private from that MR, which is internal to PETSc, but
// doesn't check the SF backend; with the Kokkos backend the extra sync is just redundant
// Without a CUDA/HIP device context (host or SYCL builds) we always sync, which costs a fence
// on SYCL but keeps the ordering safe
// ~~~~~~~~~~~~~~~~~~
static inline PetscErrorCode petsc_streams_differ_kokkos(PetscDeviceContext *dctx, PetscBool *differ)
{
   PetscFunctionBegin;
   *differ = PETSC_TRUE;
   PetscCall(PetscDeviceContextGetCurrentContext(dctx));
#if (defined(KOKKOS_ENABLE_CUDA) && PetscDefined(HAVE_CUDA)) || (defined(KOKKOS_ENABLE_HIP) && PetscDefined(HAVE_HIP))
   PetscDeviceType type;
   PetscCall(PetscDeviceContextGetDeviceType(*dctx, &type));
   #if defined(KOKKOS_ENABLE_CUDA) && PetscDefined(HAVE_CUDA)
   if (type == PETSC_DEVICE_CUDA)
   {
      cudaStream_t *stream;
      PetscCall(PetscDeviceContextGetStreamHandle(*dctx, reinterpret_cast<void **>(&stream)));
      *differ = *stream != PetscGetKokkosExecutionSpace().cuda_stream() ? PETSC_TRUE : PETSC_FALSE;
   }
   #else
   if (type == PETSC_DEVICE_HIP)
   {
      hipStream_t *stream;
      PetscCall(PetscDeviceContextGetStreamHandle(*dctx, reinterpret_cast<void **>(&stream)));
      *differ = *stream != PetscGetKokkosExecutionSpace().hip_stream() ? PETSC_TRUE : PETSC_FALSE;
   }
   #endif
#endif
   PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode sf_sync_before_begin_kokkos()
{
   PetscDeviceContext dctx;
   PetscBool differ;

   PetscFunctionBegin;
   PetscCall(petsc_streams_differ_kokkos(&dctx, &differ));
   if (differ) PetscGetKokkosExecutionSpace().fence();
   PetscFunctionReturn(PETSC_SUCCESS);
}

// Wait for work on the current PETSc device context stream if that is not the stream of
// the PETSc execution space, e.g. before Kokkos kernels use data produced by native
// CUDA/HIP PETSc objects
static inline PetscErrorCode sync_device_context_if_streams_differ_kokkos()
{
   PetscDeviceContext dctx;
   PetscBool differ;

   PetscFunctionBegin;
   PetscCall(petsc_streams_differ_kokkos(&dctx, &differ));
   if (differ) PetscCall(PetscDeviceContextSynchronize(dctx));
   PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode sf_sync_after_end_kokkos()
{
   PetscFunctionBegin;
   PetscCall(sync_device_context_if_streams_differ_kokkos());
   PetscFunctionReturn(PETSC_SUCCESS);
}

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
   
   // Is (v, c) a better max than this - bigger value, or on ties the smaller column
   // This matches the CPU maxloc (first entry in the sorted row) regardless of
   // how the entries were split across threads or the order the threads are joined
   KOKKOS_INLINE_FUNCTION
   bool is_better(const PetscReal v, const PetscInt c) const {
      return v > val || (v == val && c < col);
   }

   // We use this in our parallel reduction to find maximum
   KOKKOS_INLINE_FUNCTION
   void operator+=(const ReduceDataMaxRow& src) {
      // If src has a larger value (or same value with smaller column), take it
      if (is_better(src.val, src.col)) {
         val = src.val;
         col = src.col;
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