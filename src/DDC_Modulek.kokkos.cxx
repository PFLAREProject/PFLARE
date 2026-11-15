// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>

// The definition of the device copy of the cf markers on a given level 
// is stored in Device_Datak.kokkos.cxx and imported as extern from 
// kokkos_helper.hpp

//------------------------------------------------------------------------------------------------------------------------

// ddc cleanup but on the device - uses the global variable cf_markers_local_d
// This no longer copies back to the host pointer cf_markers_local at the end
// You have to explicitly call copy_cf_markers_d2h(cf_markers_local) to do this
PETSC_INTERN void ddc_kokkos(Mat *input_mat, const PetscReal fraction_swap, const PetscReal max_dd_ratio, const PetscReal max_dd_ratio_achieved, PetscReal *random_numbers)
{
   // Can't use the global directly within the parallel 
   // regions on the device
   intKokkosView cf_markers_d = cf_markers_local_d;  
   PetscScalarKokkosView diag_dom_ratio_d = diag_dom_ratio_local_d;
   PetscIntKokkosView is_fine_local_d;

   const int match_cf = -1; // F_POINT == -1
   create_cf_is_device_kokkos(input_mat, match_cf, is_fine_local_d);
   PetscInt local_rows_aff = is_fine_local_d.extent(0);

   bool trigger_dd_ratio_compute = max_dd_ratio > 0;
   auto exec = PetscGetKokkosExecutionSpace();   

   // Do a fixed alpha_diag
   PetscInt search_size;
   if (fraction_swap < 0) {
      // We have to look through all the local rows
      search_size = local_rows_aff;
   }
   // Or pick alpha_diag based on the worst % of rows
   else {
      // Only need to go through the biggest % of indices
      PetscInt one = 1;
      
      // If we are trying to hit a given max_dd_ratio, then we need to continue coarsening, even
      // if we only change one dof at a time
      if (trigger_dd_ratio_compute)
      {
         search_size = std::max(one, static_cast<PetscInt>(double(local_rows_aff) * fraction_swap));
      }
      // If we're not trying to hit a given max_dd_ratio, then if fraction_swap is small
      // we allow it to just not swap anything if the number of local rows is small
      // This stops many lower levels in parallel where we are only changing one dof at a time
      else
      {
         search_size = static_cast<PetscInt>(double(local_rows_aff) * fraction_swap);
      }
   }
   
   if (trigger_dd_ratio_compute) 
   {
      // ~~~~~~~~~~~~~~~
      // Ratio not met - use PMIS-based independent set to swap F points
      // This mirrors the CPU ddc_cpu logic when trigger_dd_ratio_compute is true
      // We build an independent set in Aff + Aff^T with a measure given by the
      // diagonal dominance ratio, swap those to C points, and let the outer loop
      // recompute
      // The set is built straight from A rather than an extracted Aff: every point other than
      // the F points still above the target ratio starts out assigned, so the only edges are
      // the entries of Aff, and the local and global orders of the F points are those of Aff,
      // so the ties break the same way
      // ~~~~~~~~~~~~~~~
      {
         PetscInt local_rows, local_cols;
         PetscCallVoid(MatGetLocalSize(*input_mat, &local_rows, &local_cols));

         // The measure and cf_markers for the rows of A
         // Assigned (as C) unless they're an F point that may swap
         PetscScalarKokkosView measure_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "measure_d"), local_rows);
         intKokkosView cf_markers_swap_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "cf_markers_swap_d"), local_rows);
         Kokkos::deep_copy(exec, measure_d, PETSC_MAX_REAL);
         Kokkos::deep_copy(exec, cf_markers_swap_d, 1);

         // Copy the random numbers from host to device
         // These are generated in the Fortran wrapper so CPU and Kokkos use the same randoms
         // The random number of an F point is that of its row in Aff
         PetscScalarKokkosViewHost random_h(random_numbers, local_rows_aff);
         PetscScalarKokkosView random_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "random_d"), local_rows_aff);
         Kokkos::deep_copy(exec, random_d, random_h);
         PetscCallVoid(PetscLogCpuToGpu(local_rows_aff * sizeof(PetscReal)));

         const PetscReal max_scale = std::max(10.0, max_dd_ratio_achieved * 2.0);
         const PetscReal target_ratio = max_dd_ratio;

         // Build the measure:
         // The PMISR tags the smallest measure as F points
         // So we feed in measure = max(10, max_achieved*2) - (diag_dom_ratio - random/1e10)
         // which picks the biggest diagonal dominance ratio
         // We have to ensure abs(measure) >= 1 as PMISR sets anything with measure < 1 as F directly
         // Points already below the target ratio keep a measure of PETSC_MAX_REAL and stay C
         // so they won't be swapped
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(exec, 0, local_rows_aff), KOKKOS_LAMBDA(PetscInt i) {
               if (!(diag_dom_ratio_d(i) < target_ratio)) {
                  const PetscInt row = is_fine_local_d(i);
                  measure_d(row) = max_scale - (diag_dom_ratio_d(i) - random_d(i) / 1e10);
                  cf_markers_swap_d(row) = 0;
               }
         });

         // As many steps as necessary
         pmisr_existing_measure_from_matrix_kokkos(input_mat, -1, measure_d, cf_markers_swap_d);

         // Swap the F-tagged points in cf_markers_d
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(exec, 0, local_rows_aff), KOKKOS_LAMBDA(PetscInt i) {
               const PetscInt idx = is_fine_local_d(i);
               if (cf_markers_swap_d(idx) == -1) cf_markers_d(idx) *= -1; // F_POINT
         });
         Kokkos::fence();
      }
      return;
   }

   // Can't put this above because of collective operations in parallel (namely the MatDiagDomRatio_kokkos)
   // If we have local points to swap
   if (search_size > 0)
   {
      // If we reach here then we want to swap some local F points to C points

      // Create device memory for bins
      auto dom_bins_d = PetscIntKokkosView("dom_bins_d", 1000);
      Kokkos::deep_copy(exec, dom_bins_d, 0);

      // Bin the diagonal dominance ratio
      if (fraction_swap > 0)
      {
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(exec, 0, local_rows_aff), KOKKOS_LAMBDA(PetscInt i) {

            // Let's bin the entry
            // The top bin also takes ratios above one, along with negative or NaN ratios
            // We test in floating point first so we never convert NaN or a huge value to an int
            int bin;
            const double scaled = diag_dom_ratio_d(i) * double(dom_bins_d.extent(0));
            if (scaled >= 0.0 && scaled < double(dom_bins_d.extent(0) - 1)) {
               bin = int(floor(scaled)) + 1;
            }
            else {
               bin = dom_bins_d.extent(0);
            }
            // Has to be atomic as many threads from different rows
            // may be writing to the same bin
            Kokkos::atomic_add(&dom_bins_d(bin - 1), 1);
         });
      }

      PetscReal swap_dom_val;
      // Do a fixed alpha_diag
      if (fraction_swap < 0){
         swap_dom_val = -fraction_swap;
      }
      // Otherwise swap everything bigger than a fixed fraction
      else{

         // Parallel scan to inclusive sum the number of entries we have in 
         // the bins
         Kokkos::parallel_scan(Kokkos::RangePolicy<>(exec, 0, dom_bins_d.extent(0)), KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
            // Inclusive scan
            update += dom_bins_d(i);         
            if (final) {
               dom_bins_d(i) = update; // only update array on final pass
            }
         });        

         // Now if we reduce how many are > the search_size, we know the bin boundary we want
         int bin_boundary = 0;  
         Kokkos::parallel_reduce ("ReductionBin", Kokkos::RangePolicy<>(exec, 0, dom_bins_d.extent(0)), KOKKOS_LAMBDA (const int i, int& update) {
            if (dom_bins_d(i) > dom_bins_d(dom_bins_d.extent(0)-1) - search_size) update++;
         }, bin_boundary);   

         bin_boundary = dom_bins_d.extent(0) - bin_boundary;                
         swap_dom_val = double(bin_boundary) / double(dom_bins_d.extent(0));

      }

      // Go and swap F points to C points
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(exec, 0, local_rows_aff), KOKKOS_LAMBDA(PetscInt i) {

            if (diag_dom_ratio_d(i) != 0.0 && diag_dom_ratio_d(i) >= swap_dom_val)
            {
               // This is the actual numbering in A, rather than Aff
               PetscInt idx = is_fine_local_d(i);
               cf_markers_d(idx) *= -1;
            }
      });
      // Ensure we're done before we exit
      Kokkos::fence(); 
   }   

   return;
}

//------------------------------------------------------------------------------------------------------------------------