// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>

// Per-PCAIR storage for the device-side fine/coarse index views, indexed by
// multigrid level. Previously these arrays were file-scope globals, which
// caused two concurrent PCAIR instances to collide on the same level slot
// (overwriting each other's shared_ptr to the view, leading to use-after-free
// in the older PCAIR's apply path). The handle is now owned by the air_data
// structure on the Fortran side and threaded through every kokkos call that
// needs it.
struct VecISCopyLocalKokkosCtx {
   ViewPetscIntPtr* IS_fine_views_local   = nullptr;
   ViewPetscIntPtr* IS_coarse_views_local = nullptr;
   int max_levels = -1;
};

//------------------------------------------------------------------------------------------------------------------------

// Destroys the data. handle is a pointer to a void* (Fortran c_ptr by ref);
// sets it to NULL on exit so the caller's c_ptr field becomes c_null_ptr.
PETSC_INTERN void destroy_VecISCopyLocal_kokkos(void **handle)
{
   if (!handle || !*handle) return;
   auto *ctx = static_cast<VecISCopyLocalKokkosCtx *>(*handle);
   if (ctx->IS_fine_views_local) {
      delete[] ctx->IS_fine_views_local;
      ctx->IS_fine_views_local = nullptr;
   }
   if (ctx->IS_coarse_views_local) {
      delete[] ctx->IS_coarse_views_local;
      ctx->IS_coarse_views_local = nullptr;
   }
   delete ctx;
   *handle = nullptr;

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Creates the per-PCAIR data we need to do the equivalent of veciscopy on
// local data in kokkos. handle is a pointer to a void* (Fortran c_ptr by
// ref); on entry, if *handle is NULL we allocate fresh; if it is already
// allocated with a matching max_levels we keep it; if max_levels differs we
// destroy and re-create.
PETSC_INTERN void create_VecISCopyLocal_kokkos(int max_levels_input, void **handle)
{
   if (!handle) return;

   if (*handle) {
      auto *existing = static_cast<VecISCopyLocalKokkosCtx *>(*handle);
      if (existing->max_levels == max_levels_input) return;  // already correct size
      destroy_VecISCopyLocal_kokkos(handle);
   }

   auto *ctx = new VecISCopyLocalKokkosCtx;
   ctx->max_levels = max_levels_input;
   ctx->IS_fine_views_local   = new ViewPetscIntPtr[max_levels_input];
   ctx->IS_coarse_views_local = new ViewPetscIntPtr[max_levels_input];
   for (int i = 0; i < max_levels_input; i++) {
      ctx->IS_fine_views_local[i]   = nullptr;
      ctx->IS_coarse_views_local[i] = nullptr;
   }
   *handle = ctx;

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Copy the input IS's to the device for our_level
PETSC_INTERN void set_VecISCopyLocal_kokkos_our_level(void *handle, int our_level, PetscInt global_row_start, IS *index_fine, IS *index_coarse)
{
   auto *ctx = static_cast<VecISCopyLocalKokkosCtx *>(handle);
   auto exec = PetscGetKokkosExecutionSpace();
   const int level_idx = our_level - 1;

   // Get the sizes of the local component of the input IS's
   PetscInt fine_local_size, coarse_local_size;
   PetscCallVoid(ISGetLocalSize(*index_fine, &fine_local_size));
   PetscCallVoid(ISGetLocalSize(*index_coarse, &coarse_local_size));

   // Get pointers to the indices on the host
   const PetscInt *fine_indices_ptr, *coarse_indices_ptr;
   PetscCallVoid(ISGetIndices(*index_fine, &fine_indices_ptr));

   // Create a host view of the existing indices
   auto fine_view_h = PetscIntConstKokkosViewHost(fine_indices_ptr, fine_local_size);
   // Create a device view - make sure to index with 0 based
   ctx->IS_fine_views_local[level_idx] = std::make_shared<PetscIntKokkosView>("IS_fine_view_" + std::to_string(our_level), fine_local_size);
   // Copy the indices over to the device
   Kokkos::deep_copy(exec, *ctx->IS_fine_views_local[level_idx], fine_view_h);
   // Log copy with petsc
   size_t bytes = fine_view_h.extent(0) * sizeof(PetscInt);
   PetscCallVoid(PetscLogCpuToGpu(bytes));
   PetscCallVoid(ISRestoreIndices(*index_fine, &fine_indices_ptr));

   // Rewrite the indices as local - save us a minus during VecISCopyLocal_kokkos
   PetscIntKokkosView is_d;
   is_d = *ctx->IS_fine_views_local[level_idx];
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, is_d.extent(0)), KOKKOS_LAMBDA(PetscInt i) {
         is_d[i] -= global_row_start;
   });

   PetscCallVoid(ISGetIndices(*index_coarse, &coarse_indices_ptr));
   auto coarse_view_h = PetscIntConstKokkosViewHost(coarse_indices_ptr, coarse_local_size);
   // Create a device view - make sure to index with 0 based
   ctx->IS_coarse_views_local[level_idx] = std::make_shared<PetscIntKokkosView>("IS_coarse_view_" + std::to_string(our_level), coarse_local_size);
   // Copy the indices over to the device
   Kokkos::deep_copy(exec, *ctx->IS_coarse_views_local[level_idx], coarse_view_h);
   // Log copy with petsc
   bytes = coarse_view_h.extent(0) * sizeof(PetscInt);
   PetscCallVoid(PetscLogCpuToGpu(bytes));
   PetscCallVoid(ISRestoreIndices(*index_coarse, &coarse_indices_ptr));

   // Rewrite the indices as local - save us a minus during VecISCopyLocal_kokkos
   is_d = *ctx->IS_coarse_views_local[level_idx];
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, is_d.extent(0)), KOKKOS_LAMBDA(PetscInt i) {
         is_d[i] -= global_row_start;
   });
   // Ensure we're finished before we exit
   Kokkos::fence();

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Do the equivalent of veciscopy on local data using the IS data on the device
PETSC_INTERN void VecISCopyLocal_kokkos(void *handle, int our_level, int fine_int, Vec *vfull, int mode_int, Vec *vreduced)
{
   auto *ctx = static_cast<VecISCopyLocalKokkosCtx *>(handle);
   const int level_idx = our_level - 1;

   // Can't use the shared pointer directly within the parallel
   // regions on the device
   PetscIntKokkosView is_d;
   auto exec = PetscGetKokkosExecutionSpace();
   // Make sure to index with 0 based
   if (fine_int)
   {
      is_d = *ctx->IS_fine_views_local[level_idx];
   }
   else
   {
      is_d = *ctx->IS_coarse_views_local[level_idx];
   }

   // SCATTER_REVERSE=1
   // vreduced[i] = vfull[is[i]]
   if (mode_int == 1)
   {
      PetscScalarKokkosView vreduced_d;
      PetscCallVoid(VecGetKokkosViewWrite(*vreduced, &vreduced_d));
      ConstPetscScalarKokkosView vfull_d;
      PetscCallVoid(VecGetKokkosView(*vfull, &vfull_d));

      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, is_d.extent(0)), KOKKOS_LAMBDA(PetscInt i) {
            vreduced_d[i] = vfull_d[is_d(i)];
      });

      PetscCallVoid(VecRestoreKokkosViewWrite(*vreduced, &vreduced_d));
      PetscCallVoid(VecRestoreKokkosView(*vfull, &vfull_d));

   }
   // SCATTER_FORWARD=0
   // vfull[is[i]] = vreduced[i]
   else if (mode_int == 0)
   {
      ConstPetscScalarKokkosView vreduced_d;
      PetscCallVoid(VecGetKokkosView(*vreduced, &vreduced_d));
      PetscScalarKokkosView vfull_d;
      PetscCallVoid(VecGetKokkosViewWrite(*vfull, &vfull_d));

      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, is_d.extent(0)), KOKKOS_LAMBDA(PetscInt i) {
            vfull_d[is_d(i)] = vreduced_d[i];
      });

      PetscCallVoid(VecRestoreKokkosView(*vreduced, &vreduced_d));
      PetscCallVoid(VecRestoreKokkosViewWrite(*vfull, &vfull_d));
   }
   // Ensure we're done before we exit
   Kokkos::fence();

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Accessor used by MatCreateSubMatrix_kokkos to read the per-PCAIR IS view
// for a given level/fine-or-coarse. Returns a copy of the shared view (cheap;
// just bumps the underlying refcount of the kokkos View handle).
PETSC_VISIBILITY_INTERNAL PetscIntKokkosView VecISCopyLocal_kokkos_get_view(void *handle, int our_level, int fine_int)
{
   auto *ctx = static_cast<VecISCopyLocalKokkosCtx *>(handle);
   const int level_idx = our_level - 1;
   if (fine_int) return *ctx->IS_fine_views_local[level_idx];
   return *ctx->IS_coarse_views_local[level_idx];
}
