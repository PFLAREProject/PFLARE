// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>

using ViewPtr = std::shared_ptr<PetscIntKokkosView>;

// Define array of shared pointers representing fine and coarse IS's 
// on each level on the device
ViewPtr* IS_fine_views_local = nullptr;
ViewPtr* IS_coarse_views_local = nullptr;
int max_levels = -1;

//------------------------------------------------------------------------------------------------------------------------

// Destroys the data
PETSC_INTERN void destroy_VecISCopyLocal_kokkos()
{
   if (IS_fine_views_local) {
      // Will automatically call the destructor on each element
      delete[] IS_fine_views_local;
      IS_fine_views_local = nullptr;
   }
   if (IS_coarse_views_local) {
      delete[] IS_coarse_views_local;
      IS_coarse_views_local = nullptr;
   }   

    return;
}

//------------------------------------------------------------------------------------------------------------------------

// Creates the data we need to do the equivalent of veciscopy on local data in kokkos
PETSC_INTERN void create_VecISCopyLocal_kokkos(int max_levels_input)
{
   // If not built
   if (!IS_fine_views_local)
   {
      // Allocate array of pointers
      max_levels = max_levels_input;

      // Initialise fine
      IS_fine_views_local = new ViewPtr[max_levels];
      // Initialize each element as null until it's set
      // we don't want to accidently call the constructor on any of the views
      for (int i = 0; i < max_levels; i++) {
         IS_fine_views_local[i] = nullptr;
      }
      // Initialise coarse
      IS_coarse_views_local = new ViewPtr[max_levels];
      for (int i = 0; i < max_levels; i++) {
         IS_coarse_views_local[i] = nullptr;
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
   // Get the sizes of the local component of the input IS's
   PetscInt fine_local_size, coarse_local_size;
   ISGetLocalSize(*index_fine, &fine_local_size);
   ISGetLocalSize(*index_coarse, &coarse_local_size);

   // Get pointers to the indices on the host
   const PetscInt *fine_indices_ptr, *coarse_indices_ptr;
   ISGetIndices(*index_fine, &fine_indices_ptr);

   // Create a host view of the existing indices
   auto fine_view_h = PetscIntConstKokkosViewHost(fine_indices_ptr, fine_local_size);
   // Create a device view
   IS_fine_views_local[our_level] = std::make_shared<PetscIntKokkosView>("IS_fine_view_" + std::to_string(our_level), fine_local_size);
   // Copy the indices over to the device
   Kokkos::deep_copy(*IS_fine_views_local[our_level], fine_view_h);
   // Log copy with petsc
   size_t bytes = fine_view_h.extent(0) * sizeof(PetscInt);
   PetscLogCpuToGpu(bytes);
   ISRestoreIndices(*index_fine, &fine_indices_ptr);

   // Rewrite the indices as local - save us a minus during VecISCopyLocal_kokkos
   PetscIntKokkosView is_d;
   is_d = *IS_fine_views_local[our_level];
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, is_d.extent(0)), KOKKOS_LAMBDA(int i) {     
         is_d[i] -= global_row_start;
   });   

   ISGetIndices(*index_coarse, &coarse_indices_ptr);
   auto coarse_view_h = PetscIntConstKokkosViewHost(coarse_indices_ptr, coarse_local_size);
   // Create a device view
   IS_coarse_views_local[our_level] = std::make_shared<PetscIntKokkosView>("IS_coarse_view_" + std::to_string(our_level), coarse_local_size);
   // Copy the indices over to the device
   Kokkos::deep_copy(*IS_coarse_views_local[our_level], coarse_view_h);  
   // Log copy with petsc
   bytes = coarse_view_h.extent(0) * sizeof(PetscInt);
   PetscLogCpuToGpu(bytes);   
   ISRestoreIndices(*index_coarse, &coarse_indices_ptr);
   
   // Rewrite the indices as local - save us a minus during VecISCopyLocal_kokkos
   is_d = *IS_coarse_views_local[our_level];
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, is_d.extent(0)), KOKKOS_LAMBDA(int i) {     
         is_d[i] -= global_row_start;
   });    

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Do the equivalent of veciscopy on local data using the IS data on the device
PETSC_INTERN void VecISCopyLocal_kokkos(int our_level, int fine_int, Vec *vfull, int mode_int, Vec *vreduced)
{
   // Can't use the shared pointer directly within the parallel 
   // regions on the device
   PetscIntKokkosView is_d;
   if (fine_int)
   {
      is_d = *IS_fine_views_local[our_level];
   }
   else
   {
      is_d = *IS_coarse_views_local[our_level];
   } 

   // SCATTER_REVERSE=1
   // vreduced[i] = vfull[is[i]]
   if (mode_int == 1)
   {
      PetscScalarKokkosView vreduced_d;
      VecGetKokkosViewWrite(*vreduced, &vreduced_d);
      ConstPetscScalarKokkosView vfull_d;
      VecGetKokkosView(*vfull, &vfull_d);      

      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, is_d.extent(0)), KOKKOS_LAMBDA(int i) {           
            vreduced_d[i] = vfull_d[is_d(i)];
      });

      VecRestoreKokkosViewWrite(*vreduced, &vreduced_d);
      VecRestoreKokkosView(*vfull, &vfull_d);      

   }        
   // SCATTER_FORWARD=0
   // vfull[is[i]] = vreduced[i]
   else if (mode_int == 0)
   {
      ConstPetscScalarKokkosView vreduced_d;
      VecGetKokkosView(*vreduced, &vreduced_d);
      PetscScalarKokkosView vfull_d;
      VecGetKokkosViewWrite(*vfull, &vfull_d);

      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, is_d.extent(0)), KOKKOS_LAMBDA(int i) {           
            vfull_d[is_d(i)] = vreduced_d[i];
      });     

      VecRestoreKokkosView(*vreduced, &vreduced_d);
      VecRestoreKokkosViewWrite(*vfull, &vfull_d);           
   }

   return;
}

//------------------------------------------------------------------------------------------------------------------------