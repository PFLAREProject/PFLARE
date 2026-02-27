// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>

struct PflareTraceScope {
   const char *func;
   explicit PflareTraceScope(const char *name) : func(name) {
      fprintf(stderr, "[PFLARE][TRACE] ENTER %s\n", func);
      fflush(stderr);
   }
   ~PflareTraceScope() {
      fprintf(stderr, "[PFLARE][TRACE] EXIT %s\n", func);
      fflush(stderr);
   }
};

// These are defined as extern in kokkos_helper.hpp but we allocate
// them in this .cxx
ViewPetscIntPtr* IS_fine_views_local = nullptr;
ViewPetscIntPtr* IS_coarse_views_local = nullptr;
PetscInt* IS_views_row_start = nullptr;
int max_levels = -1;

//------------------------------------------------------------------------------------------------------------------------

// Destroys the data
PETSC_INTERN void destroy_VecISCopyLocal_kokkos()
{
   PflareTraceScope trace_scope("destroy_VecISCopyLocal_kokkos");
   auto exec = PetscGetKokkosExecutionSpace();
   exec.fence();

   if ((IS_fine_views_local && !IS_coarse_views_local) || (!IS_fine_views_local && IS_coarse_views_local))
   {
      fprintf(stderr,
         "[PFLARE] destroy_VecISCopyLocal_kokkos inconsistent cache pointers (fine_ptr=%p, coarse_ptr=%p)\n",
         (void *)IS_fine_views_local, (void *)IS_coarse_views_local);
      fflush(stderr);
   }

   if (IS_fine_views_local) {
      // Will automatically call the destructor on each element
      delete[] IS_fine_views_local;
      IS_fine_views_local = nullptr;
   }
   if (IS_coarse_views_local) {
      delete[] IS_coarse_views_local;
      IS_coarse_views_local = nullptr;
   }
   if (IS_views_row_start) {
      delete[] IS_views_row_start;
      IS_views_row_start = nullptr;
   }

    return;
}

//------------------------------------------------------------------------------------------------------------------------

// Creates the data we need to do the equivalent of veciscopy on local data in kokkos
PETSC_INTERN void create_VecISCopyLocal_kokkos(int max_levels_input)
{
   PflareTraceScope trace_scope("create_VecISCopyLocal_kokkos");
   if (max_levels_input <= 0)
   {
      fprintf(stderr,
         "[PFLARE] create_VecISCopyLocal_kokkos invalid max_levels_input=%d\n",
         max_levels_input);
      fflush(stderr);
      return;
   }

   // If not built
   if (!IS_fine_views_local)
   {
      // Allocate array of pointers
      max_levels = max_levels_input;

      // Initialise fine
      IS_fine_views_local = new ViewPetscIntPtr[max_levels];
      // Initialize each element as null until it's set
      // we don't want to accidently call the constructor on any of the views
      for (int i = 0; i < max_levels; i++) {
         IS_fine_views_local[i] = nullptr;
      }
      // Initialise coarse
      IS_coarse_views_local = new ViewPetscIntPtr[max_levels];
      for (int i = 0; i < max_levels; i++) {
         IS_coarse_views_local[i] = nullptr;
      }
      IS_views_row_start = new PetscInt[max_levels];
      for (int i = 0; i < max_levels; i++) {
         IS_views_row_start[i] = PETSC_MIN_INT;
      }
   }
   // Built but different max size, destroy and rebuild
   else if (max_levels_input != max_levels)
   {
      destroy_VecISCopyLocal_kokkos();
      create_VecISCopyLocal_kokkos(max_levels_input);
   }

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Copy the input IS's to the device for our_level
PETSC_INTERN void set_VecISCopyLocal_kokkos_our_level(int our_level, PetscInt global_row_start, IS *index_fine, IS *index_coarse)
{
   PflareTraceScope trace_scope("set_VecISCopyLocal_kokkos_our_level");
   auto exec = PetscGetKokkosExecutionSpace();
   const int level_idx = our_level - 1;

   if (!index_fine || !index_coarse)
   {
      fprintf(stderr,
         "[PFLARE] set_VecISCopyLocal_kokkos_our_level null IS pointer (level=%d)\n",
         our_level);
      fflush(stderr);
      return;
   }

   if (level_idx < 0 || level_idx >= max_levels || !IS_fine_views_local || !IS_coarse_views_local || !IS_views_row_start)
   {
      MPI_Comm comm = MPI_COMM_NULL;
      PetscCallVoid(PetscObjectGetComm((PetscObject)*index_fine, &comm));
      int rank = -1;
      MPI_Comm_rank(comm, &rank);
      fprintf(stderr,
         "[PFLARE][rank %d] set_VecISCopyLocal_kokkos_our_level invalid cache state (level=%d, level_idx=%d, max_levels=%d, fine_ptr=%p, coarse_ptr=%p, row_start_ptr=%p)\\n",
         rank, our_level, level_idx, max_levels,
         (void *)IS_fine_views_local, (void *)IS_coarse_views_local, (void *)IS_views_row_start);
      fflush(stderr);
      return;
   }

   // Get the sizes of the local component of the input IS's
   PetscInt fine_local_size, coarse_local_size;
   PetscCallVoid(ISGetLocalSize(*index_fine, &fine_local_size));
   PetscCallVoid(ISGetLocalSize(*index_coarse, &coarse_local_size));

   if (fine_local_size < 0 || coarse_local_size < 0)
   {
      MPI_Comm comm = MPI_COMM_NULL;
      PetscCallVoid(PetscObjectGetComm((PetscObject)*index_fine, &comm));
      int rank = -1;
      MPI_Comm_rank(comm, &rank);
      fprintf(stderr,
         "[PFLARE][rank %d] set_VecISCopyLocal_kokkos_our_level invalid IS sizes (fine_n=%" PetscInt_FMT ", coarse_n=%" PetscInt_FMT ", level=%d)\n",
         rank, fine_local_size, coarse_local_size, our_level);
      fflush(stderr);
      return;
   }

   // Get pointers to the indices on the host
   const PetscInt *fine_indices_ptr, *coarse_indices_ptr;
   PetscCallVoid(ISGetIndices(*index_fine, &fine_indices_ptr));

   // Create a host view of the existing indices
   auto fine_view_h = PetscIntConstKokkosViewHost(fine_indices_ptr, fine_local_size);
   // Create a device view - make sure to index with 0 based
   IS_fine_views_local[level_idx] = std::make_shared<PetscIntKokkosView>("IS_fine_view_" + std::to_string(our_level), fine_local_size);
   // Copy the indices over to the device
   Kokkos::deep_copy(*IS_fine_views_local[level_idx], fine_view_h);
   // Log copy with petsc
   size_t bytes = fine_view_h.extent(0) * sizeof(PetscInt);
   PetscCallVoid(PetscLogCpuToGpu(bytes));
   PetscCallVoid(ISRestoreIndices(*index_fine, &fine_indices_ptr));

   // Rewrite the indices as local - save us a minus during VecISCopyLocal_kokkos
   PetscIntKokkosView is_d;
   is_d = *IS_fine_views_local[level_idx];
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, is_d.extent(0)), KOKKOS_LAMBDA(PetscInt i) {
         is_d[i] -= global_row_start;
   });   

   if (is_d.extent(0) > 0)
   {
      PetscInt fine_min = PETSC_MAX_INT, fine_max = PETSC_MIN_INT;
      Kokkos::parallel_reduce(
         Kokkos::RangePolicy<>(exec, 0, is_d.extent(0)),
         KOKKOS_LAMBDA(const PetscInt i, PetscInt &thread_min) {
            const PetscInt val = is_d(i);
            if (val < thread_min) thread_min = val;
         },
         Kokkos::Min<PetscInt>(fine_min));
      Kokkos::parallel_reduce(
         Kokkos::RangePolicy<>(exec, 0, is_d.extent(0)),
         KOKKOS_LAMBDA(const PetscInt i, PetscInt &thread_max) {
            const PetscInt val = is_d(i);
            if (val > thread_max) thread_max = val;
         },
         Kokkos::Max<PetscInt>(fine_max));
      if (fine_min < 0)
      {
         MPI_Comm comm = MPI_COMM_NULL;
         PetscCallVoid(PetscObjectGetComm((PetscObject)*index_fine, &comm));
         int rank = -1;
         MPI_Comm_rank(comm, &rank);
         fprintf(stderr,
            "[PFLARE][rank %d] set_VecISCopyLocal_kokkos_our_level fine local index underflow (level=%d, min=%" PetscInt_FMT ", max=%" PetscInt_FMT ", row_start=%" PetscInt_FMT ")\\n",
            rank, our_level, fine_min, fine_max, global_row_start);
         fflush(stderr);
      }
   }

   PetscCallVoid(ISGetIndices(*index_coarse, &coarse_indices_ptr));
   auto coarse_view_h = PetscIntConstKokkosViewHost(coarse_indices_ptr, coarse_local_size);
   // Create a device view - make sure to index with 0 based
   IS_coarse_views_local[level_idx] = std::make_shared<PetscIntKokkosView>("IS_coarse_view_" + std::to_string(our_level), coarse_local_size);
   // Copy the indices over to the device
   Kokkos::deep_copy(*IS_coarse_views_local[level_idx], coarse_view_h);  
   // Log copy with petsc
   bytes = coarse_view_h.extent(0) * sizeof(PetscInt);
   PetscCallVoid(PetscLogCpuToGpu(bytes));   
   PetscCallVoid(ISRestoreIndices(*index_coarse, &coarse_indices_ptr));
   
   // Rewrite the indices as local - save us a minus during VecISCopyLocal_kokkos
   is_d = *IS_coarse_views_local[level_idx];
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, is_d.extent(0)), KOKKOS_LAMBDA(PetscInt i) {
         is_d[i] -= global_row_start;
   });

   if (is_d.extent(0) > 0)
   {
      PetscInt coarse_min = PETSC_MAX_INT, coarse_max = PETSC_MIN_INT;
      Kokkos::parallel_reduce(
         Kokkos::RangePolicy<>(exec, 0, is_d.extent(0)),
         KOKKOS_LAMBDA(const PetscInt i, PetscInt &thread_min) {
            const PetscInt val = is_d(i);
            if (val < thread_min) thread_min = val;
         },
         Kokkos::Min<PetscInt>(coarse_min));
      Kokkos::parallel_reduce(
         Kokkos::RangePolicy<>(exec, 0, is_d.extent(0)),
         KOKKOS_LAMBDA(const PetscInt i, PetscInt &thread_max) {
            const PetscInt val = is_d(i);
            if (val > thread_max) thread_max = val;
         },
         Kokkos::Max<PetscInt>(coarse_max));
      if (coarse_min < 0)
      {
         MPI_Comm comm = MPI_COMM_NULL;
         PetscCallVoid(PetscObjectGetComm((PetscObject)*index_coarse, &comm));
         int rank = -1;
         MPI_Comm_rank(comm, &rank);
         fprintf(stderr,
            "[PFLARE][rank %d] set_VecISCopyLocal_kokkos_our_level coarse local index underflow (level=%d, min=%" PetscInt_FMT ", max=%" PetscInt_FMT ", row_start=%" PetscInt_FMT ")\\n",
            rank, our_level, coarse_min, coarse_max, global_row_start);
         fflush(stderr);
      }
   }
   IS_views_row_start[level_idx] = global_row_start;
   exec.fence();    

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Do the equivalent of veciscopy on local data using the IS data on the device
PETSC_INTERN void VecISCopyLocal_kokkos(int our_level, int fine_int, Vec *vfull, int mode_int, Vec *vreduced)
{
   PflareTraceScope trace_scope("VecISCopyLocal_kokkos");
   const int level_idx = our_level - 1;

   if (level_idx < 0 || level_idx >= max_levels || !IS_fine_views_local || !IS_coarse_views_local)
   {
      MPI_Comm comm = MPI_COMM_NULL;
      PetscCallVoid(PetscObjectGetComm((PetscObject)*vfull, &comm));
      int rank = -1;
      MPI_Comm_rank(comm, &rank);
      fprintf(stderr,
         "[PFLARE][rank %d] VecISCopyLocal_kokkos invalid cache state (level=%d, level_idx=%d, max_levels=%d, fine_ptr=%p, coarse_ptr=%p)\\n",
         rank, our_level, level_idx, max_levels, (void *)IS_fine_views_local, (void *)IS_coarse_views_local);
      fflush(stderr);
      return;
   }

   // Can't use the shared pointer directly within the parallel 
   // regions on the device
   PetscIntKokkosView is_d;
   // Make sure to index with 0 based
   if (fine_int)
   {
      if (!IS_fine_views_local[level_idx])
      {
         MPI_Comm comm = MPI_COMM_NULL;
         PetscCallVoid(PetscObjectGetComm((PetscObject)*vfull, &comm));
         int rank = -1;
         MPI_Comm_rank(comm, &rank);
         fprintf(stderr,
            "[PFLARE][rank %d] VecISCopyLocal_kokkos missing fine cache view (level=%d, level_idx=%d)\\n",
            rank, our_level, level_idx);
         fflush(stderr);
         return;
      }
      is_d = *IS_fine_views_local[level_idx];
   }
   else
   {
      if (!IS_coarse_views_local[level_idx])
      {
         MPI_Comm comm = MPI_COMM_NULL;
         PetscCallVoid(PetscObjectGetComm((PetscObject)*vfull, &comm));
         int rank = -1;
         MPI_Comm_rank(comm, &rank);
         fprintf(stderr,
            "[PFLARE][rank %d] VecISCopyLocal_kokkos missing coarse cache view (level=%d, level_idx=%d)\\n",
            rank, our_level, level_idx);
         fflush(stderr);
         return;
      }
      is_d = *IS_coarse_views_local[level_idx];
   } 

   PetscInt vfull_local_size = 0, vreduced_local_size = 0;
   PetscCallVoid(VecGetLocalSize(*vfull, &vfull_local_size));
   PetscCallVoid(VecGetLocalSize(*vreduced, &vreduced_local_size));
   auto exec = PetscGetKokkosExecutionSpace();

   if ((PetscInt)is_d.extent(0) != vreduced_local_size)
   {
      MPI_Comm comm = MPI_COMM_NULL;
      PetscCallVoid(PetscObjectGetComm((PetscObject)*vfull, &comm));
      int rank = -1;
      MPI_Comm_rank(comm, &rank);
      fprintf(stderr,
         "[PFLARE][rank %d] VecISCopyLocal_kokkos extent mismatch (level=%d, fine=%d, mode=%d, is_n=%" PetscInt_FMT ", vreduced_n=%" PetscInt_FMT ", vfull_n=%" PetscInt_FMT ")\\n",
         rank, our_level, fine_int, mode_int, (PetscInt)is_d.extent(0), vreduced_local_size, vfull_local_size);
      fflush(stderr);
   }

   // SCATTER_REVERSE=1
   // vreduced[i] = vfull[is[i]]
   if (mode_int == 1)
   {
      PetscScalarKokkosView vreduced_d;
      PetscCallVoid(VecGetKokkosViewWrite(*vreduced, &vreduced_d));
      ConstPetscScalarKokkosView vfull_d;
      PetscCallVoid(VecGetKokkosView(*vfull, &vfull_d));

      PetscIntKokkosView bad_count_d("veciscopy_bad_count_d", 1);
      Kokkos::deep_copy(exec, bad_count_d, (PetscInt)0);

      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, is_d.extent(0)), KOKKOS_LAMBDA(PetscInt i) {
            const PetscInt idx = is_d(i);
            if (i >= vreduced_local_size || idx < 0 || idx >= vfull_local_size)
            {
               Kokkos::atomic_add(&bad_count_d(0), (PetscInt)1);
            }
            else
            {
               vreduced_d[i] = vfull_d[idx];
            }
      });
      exec.fence();
      auto bad_count_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bad_count_d);
      if (bad_count_h(0) > 0)
      {
         MPI_Comm comm = MPI_COMM_NULL;
         PetscCallVoid(PetscObjectGetComm((PetscObject)*vfull, &comm));
         int rank = -1;
         MPI_Comm_rank(comm, &rank);
         fprintf(stderr,
            "[PFLARE][rank %d] VecISCopyLocal_kokkos reverse invalid accesses=%" PetscInt_FMT " (level=%d, fine=%d, is_n=%" PetscInt_FMT ", vreduced_n=%" PetscInt_FMT ", vfull_n=%" PetscInt_FMT ")\\n",
            rank, bad_count_h(0), our_level, fine_int, (PetscInt)is_d.extent(0), vreduced_local_size, vfull_local_size);
         fflush(stderr);
      }

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

      PetscIntKokkosView bad_count_d("veciscopy_bad_count_d", 1);
      Kokkos::deep_copy(exec, bad_count_d, (PetscInt)0);

      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, is_d.extent(0)), KOKKOS_LAMBDA(PetscInt i) {
            const PetscInt idx = is_d(i);
            if (i >= vreduced_local_size || idx < 0 || idx >= vfull_local_size)
            {
               Kokkos::atomic_add(&bad_count_d(0), (PetscInt)1);
            }
            else
            {
               vfull_d[idx] = vreduced_d[i];
            }
      });     
      exec.fence();
      auto bad_count_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bad_count_d);
      if (bad_count_h(0) > 0)
      {
         MPI_Comm comm = MPI_COMM_NULL;
         PetscCallVoid(PetscObjectGetComm((PetscObject)*vfull, &comm));
         int rank = -1;
         MPI_Comm_rank(comm, &rank);
         fprintf(stderr,
            "[PFLARE][rank %d] VecISCopyLocal_kokkos forward invalid accesses=%" PetscInt_FMT " (level=%d, fine=%d, is_n=%" PetscInt_FMT ", vreduced_n=%" PetscInt_FMT ", vfull_n=%" PetscInt_FMT ")\\n",
            rank, bad_count_h(0), our_level, fine_int, (PetscInt)is_d.extent(0), vreduced_local_size, vfull_local_size);
         fflush(stderr);
      }

      PetscCallVoid(VecRestoreKokkosView(*vreduced, &vreduced_d));
      PetscCallVoid(VecRestoreKokkosViewWrite(*vfull, &vfull_d));  
   }

   return;
}

//------------------------------------------------------------------------------------------------------------------------