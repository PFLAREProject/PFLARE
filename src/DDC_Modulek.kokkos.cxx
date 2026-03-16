// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

// The definition of the device copy of the cf markers on a given level 
// is stored in Device_Datak.kokkos.cxx and imported as extern from 
// kokkos_helper.hpp

//------------------------------------------------------------------------------------------------------------------------

// ddc cleanup but on the device - uses the global variable cf_markers_local_d
// This no longer copies back to the host pointer cf_markers_local at the end
// You have to explicitly call copy_cf_markers_d2h(cf_markers_local) to do this
PETSC_INTERN void ddc_kokkos(Mat *input_mat, const PetscReal fraction_swap, const PetscReal max_dd_ratio, const PetscReal max_dd_ratio_achieved, Mat *aff, PetscReal *random_numbers)
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
      // ~~~~~~~~~~~~~~~
      {
         // Create measure and cf_markers for Aff
         PetscScalarKokkosView measure_d("measure_d", local_rows_aff);
         intKokkosView cf_markers_aff_d("cf_markers_aff_d", local_rows_aff);
         Kokkos::deep_copy(cf_markers_aff_d, 0);

         // Copy the random numbers from host to device
         // These are generated in the Fortran wrapper so CPU and Kokkos use the same randoms
         PetscScalarKokkosViewHost random_h(random_numbers, local_rows_aff);
         PetscScalarKokkosView random_d("random_d", local_rows_aff);
         Kokkos::deep_copy(random_d, random_h);
         PetscCallVoid(PetscLogCpuToGpu(local_rows_aff * sizeof(PetscReal)));

         const PetscReal max_scale = std::max(10.0, max_dd_ratio_achieved * 2.0);
         const PetscReal target_ratio = max_dd_ratio;

         // Build the measure:
         // pmisr_existing_measure_cf_markers tags the smallest measure as F points
         // So we feed in measure = max(10, max_achieved*2) - (diag_dom_ratio - random/1e10)
         // which picks the biggest diagonal dominance ratio
         // We have to ensure abs(measure) >= 1 as PMISR sets anything with measure < 1 as F directly
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, local_rows_aff), KOKKOS_LAMBDA(PetscInt i) {

               // Scale: measure = max(10, max_achieved*2) - (diag_dom_ratio - random/1e10)
               measure_d(i) = max_scale - (diag_dom_ratio_d(i) - random_d(i) / 1e10);

               // Points already below threshold: set measure to max and mark as C
               // so they won't be swapped
               if (diag_dom_ratio_d(i) < target_ratio) {
                  measure_d(i) = PETSC_MAX_REAL;
                  cf_markers_aff_d(i) = 1; // C_POINT
               }
         });
         exec.fence();

         // Call PMISR with implicit transpose - takes Aff directly, handles Aff+Aff^T internally
         // pmis_int=0 means PMISR, zero_measure_c_point_int=0
         pmisr_existing_measure_implicit_transpose_kokkos(aff, -1, 0, measure_d, cf_markers_aff_d, 0);

         // Swap F-tagged points back into cf_markers_d
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, local_rows_aff), KOKKOS_LAMBDA(PetscInt i) {
               if (cf_markers_aff_d(i) == -1) { // F_POINT
                  PetscInt idx = is_fine_local_d(i);
                  cf_markers_d(idx) *= -1;
               }
         });
         exec.fence();
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
      Kokkos::deep_copy(dom_bins_d, 0);

      // Bin the diagonal dominance ratio
      if (fraction_swap > 0)
      {
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, local_rows_aff), KOKKOS_LAMBDA(PetscInt i) { 

            // Let's bin the entry
            int bin;
            int test_bin = floor(diag_dom_ratio_d(i) * double(dom_bins_d.extent(0))) + 1;
            if (test_bin < int(dom_bins_d.extent(0)) && test_bin >= 0) {
               bin = test_bin;
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
         Kokkos::parallel_scan(dom_bins_d.extent(0), KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
            // Inclusive scan
            update += dom_bins_d(i);         
            if (final) {
               dom_bins_d(i) = update; // only update array on final pass
            }
         });        

         // Now if we reduce how many are > the search_size, we know the bin boundary we want
         int bin_boundary = 0;  
         Kokkos::parallel_reduce ("ReductionBin", dom_bins_d.extent(0), KOKKOS_LAMBDA (const int i, int& update) {
            if (dom_bins_d(i) > dom_bins_d(dom_bins_d.extent(0)-1) - search_size) update++;
         }, bin_boundary);   

         bin_boundary = dom_bins_d.extent(0) - bin_boundary;                
         swap_dom_val = double(bin_boundary) / double(dom_bins_d.extent(0));

      }

      // Go and swap F points to C points
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows_aff), KOKKOS_LAMBDA(PetscInt i) {

            if (diag_dom_ratio_d(i) != 0.0 && diag_dom_ratio_d(i) >= swap_dom_val)
            {
               // This is the actual numbering in A, rather than Aff
               PetscInt idx = is_fine_local_d(i);
               cf_markers_d(idx) *= -1;
            }
      });
      // Ensure we're done before we exit
      exec.fence(); 
   }   

   return;
}

//------------------------------------------------------------------------------------------------------------------------