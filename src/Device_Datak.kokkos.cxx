// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

// This is a device copy of the cf markers on a given level
// to save having to copy it to/from the host between pmisr and ddc calls
intKokkosView cf_markers_local_d;
// Device copy of local fine-point diagonal-dominance ratios for DDC
PetscScalarKokkosView diag_dom_ratio_local_d;

//------------------------------------------------------------------------------------------------------------------------

// Copy the global cf_markers_local_d back to the host
PETSC_INTERN void copy_cf_markers_d2h(int *cf_markers_local)
{
   // Host wrapper for cf_markers_local
   intKokkosViewHost cf_markers_local_h(cf_markers_local, cf_markers_local_d.extent(0));

   // Now copy device cf_markers_local_d back to host
   // Don't need to specify exec here as it's a copy back to the host
   Kokkos::deep_copy(cf_markers_local_h, cf_markers_local_d);
   // Log copy with petsc
   size_t bytes = cf_markers_local_d.extent(0) * sizeof(int);
   PetscCallVoid(PetscLogGpuToCpu(bytes));

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Copy the global diag_dom_ratio_local_d back to the host
PETSC_INTERN void copy_diag_dom_ratio_d2h(PetscReal *diag_dom_ratio_local)
{
   // Host wrapper for diag_dom_ratio_local
   PetscScalarKokkosViewHost diag_dom_ratio_h(diag_dom_ratio_local, diag_dom_ratio_local_d.extent(0));

   // Copy device diag_dom_ratio_local_d back to host
   // Don't need to specify exec here as it's a copy back to the host
   Kokkos::deep_copy(diag_dom_ratio_h, diag_dom_ratio_local_d);
   // Log copy with petsc
   size_t bytes = diag_dom_ratio_local_d.extent(0) * sizeof(PetscReal);
   PetscCallVoid(PetscLogGpuToCpu(bytes));

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Delete the global cf_markers_local_d
PETSC_INTERN void delete_device_cf_markers()
{
   // Delete the device view - this assigns an empty view
   // and hence the old view has its ref counter decremented
   cf_markers_local_d = intKokkosView();

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Delete the global diag_dom_ratio_local_d
PETSC_INTERN void delete_device_diag_dom_ratio()
{
   // Delete the device view - this assigns an empty view
   // and hence the old view has its ref counter decremented
   diag_dom_ratio_local_d = PetscScalarKokkosView();

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Creates the device local indices for F or C points based on the global cf_markers_local_d
PETSC_INTERN void create_cf_is_device_kokkos(Mat *input_mat, const int match_cf, PetscIntKokkosView &is_local_d)
{
   PetscInt local_rows, local_cols;
   PetscCallVoid(MatGetLocalSize(*input_mat, &local_rows, &local_cols));
   auto exec = PetscGetKokkosExecutionSpace();

   // Can't use the global directly within the parallel
   // regions on the device
   intKokkosView cf_markers_d = cf_markers_local_d;

   // ~~~~~~~~~~~~
   // Get the F point local indices from cf_markers_local_d
   // ~~~~~~~~~~~~
   PetscIntKokkosView point_offsets_d("point_offsets_d", local_rows+1);

   // Doing an exclusive scan to get the offsets for our local indices
   // Doing one larger so we can get the total number of points
   Kokkos::parallel_scan("point_offsets_d_scan",
      Kokkos::RangePolicy<>(exec, 0, local_rows+1),
      KOKKOS_LAMBDA(const PetscInt i, PetscInt& update, const bool final_pass) {
         bool is_f_point = false;
         if (i < local_rows) { // Predicate is based on original data up to local_rows-1
               is_f_point = (cf_markers_d(i) == match_cf); // is this point match_cf
         }
         if (final_pass) {
               point_offsets_d(i) = update;
         }
         if (is_f_point) {
               update++;
         }
      }
   );

   // The last entry in point_offsets_d is the total number of points that match match_cf
   PetscInt local_rows_row = 0;
   Kokkos::deep_copy(local_rows_row, Kokkos::subview(point_offsets_d, local_rows));

   // This will be equivalent to is_fine - global_row_start, ie the local indices
   is_local_d = PetscIntKokkosView("is_local_d", local_rows_row);

   // ~~~~~~~~~~~~
   // Write the local indices
   // ~~~~~~~~~~~~
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
         // Is this point match_cf
         if (cf_markers_d(i) == match_cf) {
            // point_offsets_d(i) gives the correct local index
            is_local_d(point_offsets_d(i)) = i;
         }
   });
   // Ensure we're done before we exit
   Kokkos::fence();
}

//------------------------------------------------------------------------------------------------------------------------

// Creates the host IS is_fine and is_coarse based on the global cf_markers_local_d
PETSC_INTERN void create_cf_is_kokkos(Mat *input_mat, IS *is_fine, IS *is_coarse)
{
   PetscIntKokkosView is_fine_local_d, is_coarse_local_d;
   MPI_Comm MPI_COMM_MATRIX;
   auto exec = PetscGetKokkosExecutionSpace();
   PetscCallVoid(PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX));

   // Create the local f point indices
   const int match_fine = -1; // F_POINT == -1
   create_cf_is_device_kokkos(input_mat, match_fine, is_fine_local_d);

   // Create the local C point indices
   const int match_coarse = 1; // C_POINT == 1
   create_cf_is_device_kokkos(input_mat, match_coarse, is_coarse_local_d);

   // Now convert them back to global indices
   PetscInt global_row_start, global_row_end_plus_one;
   PetscCallVoid(MatGetOwnershipRange(*input_mat, &global_row_start, &global_row_end_plus_one));

   // Convert F points
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, is_fine_local_d.extent(0)), KOKKOS_LAMBDA(PetscInt i) {

      is_fine_local_d(i) += global_row_start;
   });
   // Convert C points
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, is_coarse_local_d.extent(0)), KOKKOS_LAMBDA(PetscInt i) {

      is_coarse_local_d(i) += global_row_start;
   });

   // Create some host space for the indices
   PetscInt *is_fine_array = nullptr, *is_coarse_array = nullptr;
   PetscInt n_fine = is_fine_local_d.extent(0);
   PetscCallVoid(PetscMalloc1(n_fine, &is_fine_array));
   PetscIntKokkosViewHost is_fine_h = PetscIntKokkosViewHost(is_fine_array, is_fine_local_d.extent(0));
   PetscInt n_coarse = is_coarse_local_d.extent(0);
   PetscCallVoid(PetscMalloc1(n_coarse, &is_coarse_array));
   PetscIntKokkosViewHost is_coarse_h = PetscIntKokkosViewHost(is_coarse_array, n_coarse);

   // Copy over the indices to the host
   // Don't need to specify exec here as it's a copy back to the host
   Kokkos::deep_copy(is_fine_h, is_fine_local_d);
   Kokkos::deep_copy(is_coarse_h, is_coarse_local_d);
   // Log copy with petsc
   size_t bytes_fine = is_fine_local_d.extent(0) * sizeof(PetscInt);
   size_t bytes_coarse = is_coarse_local_d.extent(0) * sizeof(PetscInt);
   PetscCallVoid(PetscLogGpuToCpu(bytes_fine + bytes_coarse));

   // Now we can create the IS objects
   PetscCallVoid(ISCreateGeneral(MPI_COMM_MATRIX, is_fine_local_d.extent(0), is_fine_array, PETSC_OWN_POINTER, is_fine));
   PetscCallVoid(ISCreateGeneral(MPI_COMM_MATRIX, is_coarse_local_d.extent(0), is_coarse_array, PETSC_OWN_POINTER, is_coarse));
}

//------------------------------------------------------------------------------------------------------------------------
