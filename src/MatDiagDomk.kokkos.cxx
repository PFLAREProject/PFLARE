// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>

// The definition of the device copy of the cf markers on a given level 
// is stored in Device_Datak.kokkos.cxx and imported as extern from 
// kokkos_helper.hpp

//------------------------------------------------------------------------------------------------------------------------

// Computes the diagonal dominance ratio of the input matrix over fine points in global variable cf_markers_local_d
// This code is very similar to MatCreateSubMatrix_kokkos
PETSC_INTERN void MatDiagDomRatio_kokkos(Mat *input_mat, PetscReal *max_dd_ratio_achieved, PetscInt *local_rows_aff)
{
   PetscInt local_rows, local_cols;

   // Are we in parallel?
   MatType mat_type;
   MPI_Comm MPI_COMM_MATRIX;
   PetscCallVoid(MatGetType(*input_mat, &mat_type));

   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;
   PetscCallVoid(PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX));
   PetscCallVoid(MatGetLocalSize(*input_mat, &local_rows, &local_cols));

   Mat mat_local = NULL, mat_nonlocal = NULL;
   // The matrix's own mult SF (the borrowed mvctx) drives the halo broadcast
   PetscSF sf = NULL;

   PetscInt rows_ao, cols_ao;
   if (mpi)
   {
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
   intKokkosView cf_markers_nonlocal_d;
   PetscIntKokkosView is_fine_local_d;
   auto exec = PetscGetKokkosExecutionSpace();

   // ~~~~~~~~~~~~
   // Get the F point local indices from cf_markers_local_d
   // ~~~~~~~~~~~~
   const int match_cf = -1; // F_POINT == -1
   create_cf_is_device_kokkos(input_mat, match_cf, is_fine_local_d);
   PetscInt local_rows_row = is_fine_local_d.extent(0);
   *local_rows_aff = local_rows_row;

   // Create device memory for the diag_dom_ratio
   diag_dom_ratio_local_d = PetscScalarKokkosView("diag_dom_ratio_local_d", local_rows_row);
   PetscScalarKokkosView diag_dom_ratio_d = diag_dom_ratio_local_d;

   // ~~~~~~~~~~~~~~~
   // Can now go and compute the diagonal dominance sums
   // ~~~~~~~~~~~~~~~
   PetscMemType mtype;

   // The off-diagonal component requires some comms which we can start now
   if (mpi)
   {
      cf_markers_nonlocal_d = intKokkosView("cf_markers_nonlocal_d", cols_ao);

      PetscCallVoid(MatGetMultPetscSF(*input_mat, &sf));

      // Start the broadcast of the cf splitting, then overlap with local-only work below.
      // The local work only reads the root cf_markers_d, which is allowed while the
      // broadcast is in flight. Nothing may touch the leaf until End.
      // Be careful these aren't petscints
      PetscCallVoid(sf_sync_before_begin_kokkos());
      PetscCallVoid(PetscSFBcastWithMemTypeBegin(sf, MPI_INT,
                  PETSC_MEMTYPE_KOKKOS, cf_markers_d.data(),
                  PETSC_MEMTYPE_KOKKOS, cf_markers_nonlocal_d.data(),
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

   // Have to store the diagonal entry, every entry is written by the local kernel
   PetscScalarKokkosView diag_entry_d(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "diag_entry_d"), local_rows_row);

   // Get the diagonal entry of each fine row, while also summing up the
   // local non-diagonal fine entries into diag_dom_ratio_d
   // One thread per row, summing in the row's order into a single accumulator
   // (continued by the nonlocal kernel below), which is exactly the order of
   // MatDiagDomRatio_cpu, so the device ratios are bitwise the same as the CPU ones
   // The DDC compares these ratios against bin edges and many sit exactly on one,
   // so a different summation order would change the CF splitting
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, local_rows_row), KOKKOS_LAMBDA(const PetscInt i_idx_is_row) {

         const PetscInt i = is_fine_local_d(i_idx_is_row);
         PetscScalar sum_val = 0.0, diag_val = 0.0;
         for (PetscInt k = device_local_i[i]; k < device_local_i[i + 1]; k++)
         {
            const PetscInt target_col = device_local_j[k];
            // Is this column fine? F_POINT == -1
            if (cf_markers_d(target_col) == -1)
            {
               const PetscScalar val = Kokkos::abs(device_local_vals[k]);
               if (target_col == i) diag_val = val;
               else sum_val += val;
            }
         }
         diag_entry_d(i_idx_is_row) = diag_val;
         diag_dom_ratio_d(i_idx_is_row) = sum_val;
   });

   // Finish the in-flight broadcast and only then read from the leaf
   if (mpi)
   {
      PetscCallVoid(PetscSFBcastEnd(sf, MPI_INT, cf_markers_d.data(), cf_markers_nonlocal_d.data(), MPI_REPLACE));
      PetscCallVoid(sf_sync_after_end_kokkos());
   }

   // ~~~~~~~~~~~~~~~
   // Finish the comms and add the non-local entries to diag_dom_ratio_d
   // before we divide by the diagonal entry
   // ~~~~~~~~~~~~~~~

   // The off-diagonal component requires some comms
   // Basically a copy of MatCreateSubMatrix_MPIAIJ_SameRowColDist
   if (mpi)
   {
      // ~~~~~~~~~~~~
      // Get pointers to the nonlocal i,j,vals on the device
      // ~~~~~~~~~~~~
      const PetscInt *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
      PetscScalar *device_nonlocal_vals = nullptr;        
      PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype));

      // Sum up the nonlocal matching entries into diag_dom_ratio_d
      // Nothing to do if this rank has no off-process columns
      if (cols_ao > 0)
      {
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(exec, 0, local_rows_row), KOKKOS_LAMBDA(const PetscInt i_idx_is_row) {

               const PetscInt i = is_fine_local_d(i_idx_is_row);
               // Most rows have no nonlocal entries
               if (device_nonlocal_i[i + 1] == device_nonlocal_i[i]) return;

               // Carry on from the local sum, as the CPU does
               PetscScalar sum_val = diag_dom_ratio_d(i_idx_is_row);
               for (PetscInt k = device_nonlocal_i[i]; k < device_nonlocal_i[i + 1]; k++)
               {
                  // Is this nonlocal column fine? F_POINT == -1
                  if (cf_markers_nonlocal_d(device_nonlocal_j[k]) == -1) sum_val += Kokkos::abs(device_nonlocal_vals[k]);
               }
               diag_dom_ratio_d(i_idx_is_row) = sum_val;
         });
      }
   }

   // ~~~~~~~~~~~~~
   // Compute the diag dominance ratio
   // ~~~~~~~~~~~~~
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(exec, 0, local_rows_row), KOKKOS_LAMBDA(PetscInt i) {

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
   Kokkos::fence();

   // Kokkos::Max over an empty range returns the reducer identity (lowest representable value),
   // so only reduce if we have local F points (matches the CPU which uses zero if there are none)
   PetscReal max_dd_ratio_local = 0.0;
   if (local_rows_row > 0) {
      Kokkos::parallel_reduce("max_dd_ratio", Kokkos::RangePolicy<>(exec, 0, local_rows_row),
         KOKKOS_LAMBDA(const PetscInt i, PetscReal& thread_max) {
            PetscReal dd_ratio = diag_dom_ratio_d(i);
            thread_max = (dd_ratio > thread_max) ? dd_ratio : thread_max;
         },
         Kokkos::Max<PetscReal>(max_dd_ratio_local)
      );
   }

   PetscCallMPIAbort(MPI_COMM_MATRIX, MPI_Allreduce(&max_dd_ratio_local, max_dd_ratio_achieved, 1, MPIU_REAL, MPI_MAX, MPI_COMM_MATRIX));

   return;
}

//------------------------------------------------------------------------------------------------------------------------