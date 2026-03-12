// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

// The definition of the device copy of the cf markers on a given level 
// is stored in Device_CF_Markersk.kokkos.cxx and imported as extern from 
// kokkos_helper.hpp

//------------------------------------------------------------------------------------------------------------------------

// Computes the diagonal dominance ratio of the input matrix over fine points in global variable cf_markers_local_d
// This code is very similar to MatCreateSubMatrix_kokkos
PETSC_INTERN void MatDiagDomRatio_kokkos(Mat *input_mat, PetscIntKokkosView &is_fine_local_d, PetscScalarKokkosView &diag_dom_ratio_d)
{
   PetscInt local_rows, local_cols;

   // Are we in parallel?
   MatType mat_type;
   MPI_Comm MPI_COMM_MATRIX;
   PetscCallVoid(MatGetType(*input_mat, &mat_type));

   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;   
   PetscCallVoid(PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX));
   PetscCallVoid(MatGetLocalSize(*input_mat, &local_rows, &local_cols)); 

   Mat_MPIAIJ *mat_mpi = nullptr;
   Mat mat_local = NULL, mat_nonlocal = NULL;   

   PetscInt rows_ao, cols_ao;
   if (mpi)
   {
      mat_mpi = (Mat_MPIAIJ *)(*input_mat)->data;
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*input_mat, &mat_local, &mat_nonlocal, NULL));      
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao));
   }
   else
   {
      mat_local = *input_mat;
   }   

   // Can't use the global directly within the parallel 
   // regions on the device
   intKokkosView cf_markers_d = cf_markers_local_d;   
   intKokkosView sf_int_dummy_d("sf_int_dummy_d", 1);
   intKokkosView cf_markers_nonlocal_d;
   intKokkosView cf_markers_send_d;
   auto exec = PetscGetKokkosExecutionSpace();

   // ~~~~~~~~~~~~
   // Get the F point local indices from cf_markers_local_d
   // ~~~~~~~~~~~~
   const int match_cf = -1; // F_POINT == -1
   create_cf_is_device_kokkos(input_mat, match_cf, is_fine_local_d);
   PetscInt local_rows_row = is_fine_local_d.extent(0);

   // Create device memory for the diag_dom_ratio
   diag_dom_ratio_d = PetscScalarKokkosView("diag_dom_ratio_d", local_rows_row);    

   // ~~~~~~~~~~~~~~~
   // Can now go and compute the diagonal dominance sums
   // ~~~~~~~~~~~~~~~
   // PetscSF comms cannot be started with a pointer derived from a zero-extent Kokkos view -
   // doing so causes intermittent failures in parallel on GPUs. Use a size-1 dummy view
   // so that every pointer passed to PetscSF is always backed by valid device memory.
   int *cf_markers_nonlocal_d_ptr = NULL;
   int *cf_markers_send_d_ptr = NULL;
   PetscMemType mem_type = PETSC_MEMTYPE_KOKKOS;       
   PetscMemType mtype;

   // The off-diagonal component requires some comms which we can start now
   if (mpi)
   {
      cf_markers_send_d = intKokkosView("cf_markers_send_d", local_rows);
      // Copy cf_markers_d into a temporary buffer
      // If we gave the comms routine cf_markers_d we couldn't even read from 
      // it until comms ended, meaning we couldn't do the work overlapping below      
      Kokkos::deep_copy(cf_markers_send_d, cf_markers_d);
      cf_markers_send_d_ptr = local_rows > 0 ? cf_markers_send_d.data() : sf_int_dummy_d.data();
      exec.fence();
      cf_markers_nonlocal_d = intKokkosView("cf_markers_nonlocal_d", cols_ao); 
      cf_markers_nonlocal_d_ptr = cols_ao > 0 ? cf_markers_nonlocal_d.data() : sf_int_dummy_d.data();

      // Start the scatter of the cf splitting - the kokkos memtype is set as PETSC_MEMTYPE_HOST or 
      // one of the kokkos backends like PETSC_MEMTYPE_HIP
      // Be careful these aren't petscints
      // PetscSF owns cf_markers_send_d_ptr as the active send buffer until End.
      // Do not even read from that send buffer before End is called.
      // If you alias it in overlapped GPU work, the failure shows up intermittently
      // in parallel runs on GPUs.
      PetscCallVoid(PetscSFBcastWithMemTypeBegin(mat_mpi->Mvctx, MPI_INT,
                  mem_type, cf_markers_send_d_ptr,
                  mem_type, cf_markers_nonlocal_d_ptr,
                  MPI_REPLACE));
   }   

   // ~~~~~~~~~~~~~~~
   // Do the local component so work/comms are overlapped
   // ~~~~~~~~~~~~~~~

   // ~~~~~~~~~~~~
   // Get pointers to the local i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr;
   PetscScalar *device_local_vals = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype));

   // Have to store the diagonal entry
   PetscScalarKokkosView diag_entry_d = PetscScalarKokkosView("diag_entry_d", local_rows_row);   
   Kokkos::deep_copy(diag_entry_d, 0);

   // Scoping to reduce peak memory
   {
      // We now go and do a reduce to get the diagonal entry, while also 
      // summing up the local non-diagonals into diag_dom_ratio_d
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(exec, local_rows_row, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            const PetscInt i_idx_is_row = t.league_rank();
            const PetscInt i = is_fine_local_d(i_idx_is_row);
            const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];

            PetscScalar sum_val = 0.0;

            // Reduce over local columns
            Kokkos::parallel_reduce(
               Kokkos::TeamVectorRange(t, ncols_local),
               [&](const PetscInt j, PetscScalar& thread_sum) {

                  // Get this local column in the input_mat
                  const PetscInt target_col = device_local_j[device_local_i[i] + j];
                  // Is this column fine? F_POINT == -1
                  if (cf_markers_d(target_col) == -1)
                  {               
                     // Is this column the diagonal
                     const bool is_diagonal = i == target_col;

                     // Get the abs value of the entry
                     PetscScalar val = Kokkos::abs(device_local_vals[device_local_i[i] + j]);   
                     
                     // We have found a diagonal in this row
                     if (is_diagonal) {
                        // Will only happen for one thread
                        diag_entry_d(i_idx_is_row) = val;                     
                     }                
                     else
                     {
                        thread_sum += val;
                     }
                  }
               },
               Kokkos::Sum<PetscScalar>(sum_val)
            );

            // Only want one thread in the team to write the result
            Kokkos::single(Kokkos::PerTeam(t), [&]() {
               diag_dom_ratio_d(i_idx_is_row) = sum_val;
            });
      });  
   }

   // ~~~~~~~~~~~~~~~
   // Finish the comms and add the non-local entries to diag_dom_ratio_d
   // before we divide by the diagonal entry
   // ~~~~~~~~~~~~~~~

   // The off-diagonal component requires some comms
   // Basically a copy of MatCreateSubMatrix_MPIAIJ_SameRowColDist
   if (mpi)
   {           
      // Finish the scatter of the cf splitting
      // Be careful these aren't petscints
      // End releases the send snapshot for normal access again.
      // The scattered cf_markers_nonlocal_d values are now safe to read.
      PetscCallVoid(PetscSFBcastEnd(mat_mpi->Mvctx, MPI_INT, cf_markers_send_d_ptr, cf_markers_nonlocal_d_ptr, MPI_REPLACE));

      // ~~~~~~~~~~~~
      // Get pointers to the nonlocal i,j,vals on the device
      // ~~~~~~~~~~~~
      const PetscInt *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
      PetscScalar *device_nonlocal_vals = nullptr;        
      PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype));

      // Sum up the nonlocal matching entries into diag_dom_ratio_d
      if (cols_ao > 0) 
      {      
         Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(exec, local_rows_row, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

               const PetscInt i_idx_is_row = t.league_rank();
               const PetscInt i = is_fine_local_d(i_idx_is_row);
               const PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];

               PetscScalar sum_val = 0.0;

               // Reduce over local columns
               Kokkos::parallel_reduce(
                  Kokkos::TeamVectorRange(t, ncols_nonlocal),
                  [&](const PetscInt j, PetscScalar& thread_sum) {

                     // This is the non-local column we have to check is present
                     const PetscInt target_col = device_nonlocal_j[device_nonlocal_i[i] + j];
                     // Is this column in the input IS? F_POINT == -1
                     if (cf_markers_nonlocal_d(target_col) == -1)
                     {               
                        // Get the abs value of the entry
                        thread_sum += Kokkos::abs(device_nonlocal_vals[device_nonlocal_i[i] + j]);
                     }
                  },
                  Kokkos::Sum<PetscScalar>(sum_val)
               );

               // Only want one thread in the team to write the result
               Kokkos::single(Kokkos::PerTeam(t), [&]() {
                  // Add into existing
                  diag_dom_ratio_d(i_idx_is_row) += sum_val;
               });
         });  
      }       
   }

   // ~~~~~~~~~~~~~
   // Compute the diag dominance ratio
   // ~~~~~~~~~~~~~
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows_row), KOKKOS_LAMBDA(PetscInt i) {     
         
      // If diag_val is zero we didn't find a diagonal
      if (diag_entry_d(i) != 0.0){
         // Compute the diagonal dominance ratio
         diag_dom_ratio_d(i) = diag_dom_ratio_d(i) / diag_entry_d(i);
      }
      else{
         diag_dom_ratio_d(i) = 0.0;
      }
   });   
   // Ensure we're done before we exit
   exec.fence();

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// ddc cleanup but on the device - uses the global variable cf_markers_local_d
// This no longer copies back to the host pointer cf_markers_local at the end
// You have to explicitly call copy_cf_markers_d2h(cf_markers_local) to do this
PETSC_INTERN void ddc_kokkos(Mat *input_mat, const PetscReal fraction_swap, PetscReal *max_dd_ratio)
{
   // Can't use the global directly within the parallel 
   // regions on the device
   intKokkosView cf_markers_d = cf_markers_local_d;  
   PetscScalarKokkosView diag_dom_ratio_d;
   PetscIntKokkosView is_fine_local_d;
   
   // Compute the diagonal dominance ratio over the fine points in cf_markers_local_d
   // ie the diag domminance ratio of Aff
   MatDiagDomRatio_kokkos(input_mat, is_fine_local_d, diag_dom_ratio_d);
   PetscInt local_rows_aff = is_fine_local_d.extent(0);
   MPI_Comm MPI_COMM_MATRIX;
   PetscCallVoid(PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX));

   bool trigger_dd_ratio_compute = *max_dd_ratio > 0;
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
   
   PetscReal max_dd_ratio_achieved = 0.0;
   // Compute the maximum diagonal dominance ratio
   if (trigger_dd_ratio_compute) 
   {
      PetscReal max_dd_ratio_local = 0.0;
      // Do a reduction to get the local max
      Kokkos::parallel_reduce("max_dd_ratio", local_rows_aff,
         KOKKOS_LAMBDA(const PetscInt i, PetscReal& thread_max) {
            PetscReal dd_ratio = diag_dom_ratio_d(i);
            thread_max = (dd_ratio > thread_max) ? dd_ratio : thread_max;
         },
         Kokkos::Max<PetscReal>(max_dd_ratio_local)
      );
      // Comms to get the global max
      PetscCallMPIAbort(MPI_COMM_MATRIX, MPI_Allreduce(&max_dd_ratio_local, &max_dd_ratio_achieved, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_MATRIX));

      //printf("computed diag dom ratio %f \n", max_dd_ratio_achieved);
      // If we have hit the required diagonal dominance ratio, then return without swapping any F points
      if (max_dd_ratio_achieved < *max_dd_ratio)
      {
         *max_dd_ratio = max_dd_ratio_achieved;
         return;
      }
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