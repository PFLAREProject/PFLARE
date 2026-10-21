/*
  C port of DDC_Module.F90 (module ddc_module)

  The optional arguments of ddc_cpu are NULL when absent. The CF markers are
  int arrays (the Fortran integer) with one entry per local row of input_mat,
  the arrays the Fortran allocated are allocated with PetscMalloc1 and freed
  with PetscFree. The loops over the rows are 0-based; the bin numbers keep
  the Fortran's 1-based values (bin_boundary ends at 0 if the search does not
  stop early, as the Fortran do loop variable does), so dom_bins is indexed
  with bin - 1
*/

#include <petscmat.h>
#include "pflareimpl.h"
#include "PETSc_Helper.h"
#include "Device_Datak.h"
#include "DDC_Modulek.h"
#include "MatDiagDomk.h"
#include "PMISR_Module.h"
#include "MatDiagDom.h"
#include "Pflare_Random.h"
#include "DDC_Module.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode ddc(Mat input_mat, IS is_fine, PetscReal fraction_swap, PetscReal *max_dd_ratio, int *cf_markers_local)
{
   // Second pass diagonal dominance cleanup
   // Flips the F definitions to C based on least diagonally dominant local rows
   // If fraction_swap = 0 this does nothing
   // If fraction_swap < 0 it uses abs(fraction_swap) to be a threshold
   //  for swapping C to F based on row-wise diagonal dominance (ie alpha_diag)
   // If fraction_swap > 0 it uses fraction_swap as the local fraction of worst C points to swap to F
   //  though it won't hit that fraction exactly as we bin the diag dom ratios for speed, it will be close to the fraction

   // ~~~~~~
   Mat Aff_ddc = NULL;
   PetscBool trigger_dd_ratio_compute_local;
   PetscInt local_rows, local_cols;
   PetscReal max_dd_ratio_achieved;
   PetscMPIInt comm_rank_ddc;
   PetscReal *diag_dom_ratio = NULL;
   PetscReal *diag_dom_ratio_random = NULL;
   PetscReal *random_numbers_ptr;
   MPI_Comm MPI_COMM_MATRIX;

#if defined(PETSC_HAVE_KOKKOS)
   Mat A_array, Aff_array;
   IS is_fine_array, is_coarse_array;
   MatType mat_type;
   int *cf_markers_local_ptr;
   PetscInt kfree;
   int *cf_markers_local_two = NULL;
   IS is_fine_temp, is_coarse_temp;
   PetscBool is_kokkos;
#endif
   // ~~~~~~

   PetscFunctionBegin;

   trigger_dd_ratio_compute_local = *max_dd_ratio > 0 ? PETSC_TRUE : PETSC_FALSE;

   // If we don't need to swap anything, return
   if (fraction_swap == 0.0 && !trigger_dd_ratio_compute_local) {
      PetscFunctionReturn(PETSC_SUCCESS);
   }

   PetscCall(PetscObjectGetComm((PetscObject)input_mat, &MPI_COMM_MATRIX));
   PetscCallMPI(MPI_Comm_rank(MPI_COMM_MATRIX, &comm_rank_ddc));

   // Compute the diagonal dominance ratio - either returned in diag_dom_ratio
   // or stored in a device copy for kokkos
   // max_dd_ratio_achieved is always returned and is the max diag dom ratio across
   // all ranks
   PetscCall(MatDiagDomRatio(input_mat, is_fine, cf_markers_local, &diag_dom_ratio, &max_dd_ratio_achieved));

   // If we have hit the required diagonal dominance ratio, return
   if (trigger_dd_ratio_compute_local && max_dd_ratio_achieved < *max_dd_ratio) {
      *max_dd_ratio = max_dd_ratio_achieved;
      PetscCall(PetscFree(diag_dom_ratio));
      PetscFunctionReturn(PETSC_SUCCESS);
   }

   // Generate random numbers for the PMIS tie-breaking in the trigger path
   // These are generated here so both CPU and Kokkos use the same randoms
   random_numbers_ptr = NULL;
   if (trigger_dd_ratio_compute_local) {

      PetscCall(MatGetLocalSize(input_mat, &local_rows, &local_cols));
      // We allocate randoms here to be the size of input, rather than
      // just F points as if we are on the device the is_fine won't be allocated
      // on the host yet
      // At least one entry, so that like the Fortran's (possibly zero-size) allocation
      // the array pflare_random_fill is handed is never NULL, which
      // PetscMalloc1 gives for zero entries
      PetscCall(PetscMalloc1(PetscMax(local_rows, 1), &diag_dom_ratio_random));

      pflare_random_fill(comm_rank_ddc, local_rows, diag_dom_ratio_random);

      random_numbers_ptr = diag_dom_ratio_random;
   }

#if defined(PETSC_HAVE_KOKKOS)

   PetscCall(MatGetType(input_mat, &mat_type));
   PetscCall(PetscStrcmpAny(mat_type, &is_kokkos, MATMPIAIJKOKKOS, MATSEQAIJKOKKOS, MATAIJKOKKOS, ""));
   if (is_kokkos) {

      // Kokkos path: only extract Aff if trigger_dd_ratio_compute
      // as the kokkos ddc computes diag dominance ratio without needing Aff
      Aff_array = NULL;
      A_array = input_mat;
      if (trigger_dd_ratio_compute_local) {

         // Create the host is_fine and is_coarse based on device cf_markers
         create_cf_is_kokkos(&A_array, &is_fine_array, &is_coarse_array);
         is_fine_temp = is_fine_array;
         is_coarse_temp = is_coarse_array;

         PetscCall(MatCreateSubMatrixWrapper(input_mat,
                     is_fine_temp, is_fine_temp, MAT_INITIAL_MATRIX,
                     &Aff_ddc, NULL, NULL, NULL, NULL));

         Aff_array = Aff_ddc;
         PetscCall(ISDestroy(&is_fine_temp));
         PetscCall(ISDestroy(&is_coarse_temp));
      }

      cf_markers_local_ptr = cf_markers_local;

      // If debugging do a comparison between CPU and Kokkos results
      if (kokkos_debug()) {
         // The size of cf_markers_local, which the Fortran array carries
         PetscCall(MatGetLocalSize(input_mat, &local_rows, &local_cols));
         PetscCall(PetscMalloc1(local_rows, &cf_markers_local_two));
         for (kfree = 0; kfree < local_rows; kfree++) cf_markers_local_two[kfree] = cf_markers_local[kfree];
      }

      // Modifies the existing device cf_markers created by the pmisr
      ddc_kokkos(&A_array, fraction_swap, *max_dd_ratio, max_dd_ratio_achieved, &Aff_array,
         random_numbers_ptr);

      // If debugging do a comparison between CPU and Kokkos results
      if (kokkos_debug()) {

         // Kokkos DDC by default now doesn't copy back to the host, as any subsequent ddc calls
         // use the existing device data
         copy_cf_markers_d2h(cf_markers_local_ptr);
         if (trigger_dd_ratio_compute_local) {
            PetscCall(ddc_cpu(input_mat, is_fine, fraction_swap, *max_dd_ratio, max_dd_ratio_achieved,
               diag_dom_ratio, cf_markers_local_two,
               Aff_ddc,
               diag_dom_ratio_random));
         } else {
            PetscCall(ddc_cpu(input_mat, is_fine, fraction_swap, *max_dd_ratio, max_dd_ratio_achieved,
               diag_dom_ratio, cf_markers_local_two, NULL, NULL));
         }

         for (kfree = 0; kfree < local_rows; kfree++) {
            // if (cf_markers_local(kfree) /= cf_markers_local_two(kfree)) then
            //    print *, kfree-1, "no match", cf_markers_local(kfree), cf_markers_local_two(kfree)
            // end if
            PetscCheck(cf_markers_local[kfree] == cf_markers_local_two[kfree], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Kokkos and CPU versions of ddc do not match");
         }
         PetscCall(PetscFree(cf_markers_local_two));
      }

      // Cleanup
      if (trigger_dd_ratio_compute_local) {
         PetscCall(MatDestroy(&Aff_ddc));
      }

   } else {
      // CPU path: only extract Aff if trigger_dd_ratio_compute
      if (trigger_dd_ratio_compute_local) {
         PetscCall(MatCreateSubMatrix(input_mat,
               is_fine, is_fine, MAT_INITIAL_MATRIX,
               &Aff_ddc));
         PetscCall(ddc_cpu(input_mat, is_fine, fraction_swap, *max_dd_ratio, max_dd_ratio_achieved,
            diag_dom_ratio, cf_markers_local,
               Aff_ddc,
               diag_dom_ratio_random));
         PetscCall(MatDestroy(&Aff_ddc));
      } else {
         PetscCall(ddc_cpu(input_mat, is_fine, fraction_swap, *max_dd_ratio, max_dd_ratio_achieved,
            diag_dom_ratio, cf_markers_local, NULL, NULL));
      }
   }
#else
   // Only the Kokkos path uses random_numbers_ptr
   (void)random_numbers_ptr;
   // CPU path: only extract Aff if trigger_dd_ratio_compute
   if (trigger_dd_ratio_compute_local) {
      PetscCall(MatCreateSubMatrix(input_mat,
            is_fine, is_fine, MAT_INITIAL_MATRIX,
            &Aff_ddc));
      PetscCall(ddc_cpu(input_mat, is_fine, fraction_swap, *max_dd_ratio, max_dd_ratio_achieved,
            diag_dom_ratio, cf_markers_local,
            Aff_ddc,
            diag_dom_ratio_random));
      PetscCall(MatDestroy(&Aff_ddc));
   } else {
         PetscCall(ddc_cpu(input_mat, is_fine, fraction_swap, *max_dd_ratio, max_dd_ratio_achieved,
            diag_dom_ratio, cf_markers_local, NULL, NULL));
   }
#endif

   PetscCall(PetscFree(diag_dom_ratio_random));
   PetscCall(PetscFree(diag_dom_ratio));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode ddc_cpu(Mat input_mat, IS is_fine, PetscReal fraction_swap, PetscReal max_dd_ratio, PetscReal max_dd_ratio_achieved,
   const PetscReal *diag_dom_ratio, int *cf_markers_local, Mat Aff, const PetscReal *diag_dom_ratio_random)
{
   // Second pass diagonal dominance cleanup
   // Flips the F definitions to C based on least diagonally dominant local rows
   // If fraction_swap = 0 this does nothing
   // If fraction_swap < 0 it uses abs(fraction_swap) to be a threshold
   //  for swapping C to F based on row-wise diagonal dominance (ie alpha_diag)
   // If fraction_swap > 0 it uses fraction_swap as the local fraction of worst C points to swap to F
   //  though it won't hit that fraction exactly as we bin the diag dom ratios for speed, it will be close to the fraction
   //
   // When Aff is present and the trigger path runs, the halo scatters inside
   // pmisr_existing_measure_implicit_transpose use Aff's own mult PetscSF
   // (obtained via MatGetMultPetscSF), so nothing needs to be forwarded here.

   // ~~~~~~

   // Local
   PetscInt local_rows, one = 1;
   PetscInt ifree;
   PetscInt input_row_start, input_row_end_plus_one;
   PetscInt idx, search_size = 0, fine_size, frac_size;
   int bin_sum, bin_boundary, bin;
   int max_luby_steps;
   PetscReal *diag_dom_ratio_measure;
   int *cf_markers_local_aff;
   const PetscInt *is_pointer;
   PetscReal swap_dom_val;
   int dom_bins[1000];
   // size(dom_bins)
   const int size_dom_bins = (int)PETSC_STATIC_ARRAY_LENGTH(dom_bins);
   // The floor of the scaled ratio, before it is converted to an int
   PetscReal bin_real;
   MPI_Comm MPI_COMM_MATRIX;
   PetscBool trigger_dd_ratio_compute;

   // ~~~~~~

   PetscFunctionBegin;

   // Get the communicator
   PetscCall(PetscObjectGetComm((PetscObject)input_mat, &MPI_COMM_MATRIX));

   // The indices are the numbering in the local fine row set
   PetscCall(ISGetIndices(is_fine, &is_pointer));
   PetscCall(ISGetLocalSize(is_fine, &fine_size));

   trigger_dd_ratio_compute = max_dd_ratio > 0 ? PETSC_TRUE : PETSC_FALSE;

   // Trigger path requires Aff and pre-generated random numbers
   // (ddc allocates diag_dom_ratio_random with at least one entry, so it is only NULL when absent)
   if (trigger_dd_ratio_compute) {
      PetscCheck(Aff && diag_dom_ratio_random, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "ddc_cpu missing Aff/randoms for trigger path");
   }

   // Do a fixed alpha_diag
   if (fraction_swap < 0) {
      // We have to look through all the local rows
      search_size = fine_size;

   // Or pick alpha_diag based on the worst % of rows
   } else {
      // Only need to go through the biggest % of indices
      frac_size = (PetscInt)((double)fine_size * (double)fraction_swap);

      // If we are trying to hit a given max_dd_ratio, then we need to continue coarsening, even
      // if we only change one dof at a time, otherwise we could get stuck
      if (trigger_dd_ratio_compute) {
         search_size = PetscMax(one, frac_size);
      // If we're not trying to hit a given max_dd_ratio, then if fraction_swap is small
      // we allow it to just not swap anything if the number of local rows is small
      // This stops many lower levels in parallel where we are only changing one dof at a time
      } else {
         search_size = frac_size;
      }
   }

   PetscCall(MatGetOwnershipRange(input_mat, &input_row_start, &input_row_end_plus_one));

   local_rows = fine_size;
   for (bin = 0; bin < size_dom_bins; bin++) dom_bins[bin] = 0;

   // ~~~~~~~~
   // Trigger path: use PMIS-based independent set to swap F points
   // ~~~~~~~~
   if (trigger_dd_ratio_compute) {
      // ~~~~~~~~
      // If we haven't hit the required diagonal dominance ratio,
      // then we need to swap some F points to C points, and we will do that with a
      // PMIS style algorithm
      // This lets us swap many points at once, without just picking every point that
      // is above the max ratio and swapping all of them - this would coarsen faster than
      // necessary as the removal of any one F point changes the diag dom of any connected
      // F points
      // We also don't want to do it one F point at a time as that would be slow
      // Hence the independent set is the best of both worlds and very parallel
      //
      // We go over all existing F points and compute an independent set
      // in Aff + Aff^T with a measure given by the diagonal dominance ratio
      // This will build an independent set of the biggest diagonal dominance ratio
      // We then swap all of those to C points and then the outer loop outside
      // this routine can recompute the diagonal dominace ratio and decide if we
      // want to do this again
      // If there are F points > max_ratio but with neighbours all < max_ratio this
      //    point will be swapped
      // If there are F points > max_ratio and with some neighbours > max_ratio then
      //    only one of those points in the neighbourhood will be swapped, namely the one
      //    with the worst diagonal dominance ratio (this is a heuristic).
      //    MacLachlan & Saad (2007) page 2120 for example multiply the diagonal dominance ratio
      //    by the 1 on the number of neighbours in S - this would prioritise swapping bad entries
      //    with many F-F connections (ie keeping Aff sparse)
      //    The ratio of all neighbouring
      //    rows will change and may be below the max_ratio after the swap
      //    That will be picked up in the next outer loop.
      // ~~~~~~~~
      PetscCall(PetscMalloc1(local_rows, &diag_dom_ratio_measure));

      // Use the random numbers passed in from the wrapper
      // so that CPU and Kokkos use the same randoms for PMIS tie-breaking
      for (ifree = 0; ifree < local_rows; ifree++) diag_dom_ratio_measure[ifree] = diag_dom_ratio_random[ifree];

      // ~~~~~~~~
      // pmisr_existing_measure_cf_markers tags the points with the smallest
      // measure as F points
      // So if we feed in a measure that is like 10 - diag_dom_ratio, it will
      // pick the points with the biggest diagonal dominace ratio
      // If a point is already below the requested ratio, we set it to be
      // PETSC_MAX_REAL so it will never be picked
      // ~~~~~~~~

      // Now we take the existing random number and scale it down
      // to break ties but not change the diagonal dominance very much
      // PMISR sets the smallest measure as F points (which is what
      // we're going to use to denote points that need to swap in the loop below)
      // We feed in only F points and a zero cf_markers_local_aff and then
      // after the PMISR we take any points tagged as "F" from that result
      // and swap them.
      // The reason we feed in something like 10 - diag_dominance_ratio is not only
      // as we want it to pick the biggest diag dominance ratio but also
      // we have to ensure abs(measure) .ge. 1
      // as the PMISR has a step where it sets anything with measure < 1 as F directly
      // given PMISR is normally called with the measure being the number of strong neighbours
      // (in double precision, as the Fortran's 10d0, 2d0 and 1d10 are)
      for (ifree = 0; ifree < local_rows; ifree++) {
         diag_dom_ratio_measure[ifree] = (PetscReal)(PetscMax(10.0, (double)max_dd_ratio_achieved * 2.0) -
            ((double)diag_dom_ratio[ifree] - (double)diag_dom_ratio_measure[ifree] / 1e10));
      }

      // Allocated and set to zero
      PetscCall(PetscCalloc1(local_rows, &cf_markers_local_aff));

      // And then any points with diagonal dominance ratio already below
      // the minimum, we set the measure to PETSC_MAX_REAL and assign them as "C" already
      // so they won't be swapped
      for (ifree = 0; ifree < local_rows; ifree++) {
         // Check against the diag_dom_ratio that we haven't modified
         if (diag_dom_ratio[ifree] < max_dd_ratio) {
            diag_dom_ratio_measure[ifree] = PETSC_MAX_REAL;
            cf_markers_local_aff[ifree] = PFLARE_C_POINT;
         }
      }

      // Call PMISR with as many steps as necessary
      // Uses the implicit transpose version which takes Aff directly
      // and handles Aff+Aff^T internally without forming the explicit sum
      max_luby_steps = -1;
      PetscCall(pmisr_existing_measure_implicit_transpose(Aff,
               max_luby_steps, PETSC_FALSE,
               diag_dom_ratio_measure, cf_markers_local_aff, NULL));

      // Let's go and swap the badly diagonally dominant rows to F points
      for (ifree = 0; ifree < local_rows; ifree++) {

         // The pmisr_existing_measure_cf_markers marked the points we want to swap as F
         if (cf_markers_local_aff[ifree] == PFLARE_F_POINT) {
            // This is the actual numbering in A, rather than Aff
            // Careful here to minus away the row_start of A, not Aff
            // as cf_markers_local is as big as A
            idx = is_pointer[ifree] - input_row_start;

            // Swap by multiplying by -1
            cf_markers_local[idx] = cf_markers_local[idx] * (-1);
         }
      }

      PetscCall(PetscFree(cf_markers_local_aff));
      PetscCall(PetscFree(diag_dom_ratio_measure));
      PetscCall(ISRestoreIndices(is_fine, &is_pointer));

      // Return as we're done
      PetscFunctionReturn(PETSC_SUCCESS);
   }

   // ~~~~~~~~~~~~~
   // If we got here then the user doesn't want us to hit a given diagonal dominance ratio
   // So we just swap a fixed fraction of the worst F points to C
   // ~~~~~~~~~~~~~

   // If we have local points to swap
   if (search_size > 0) {

      // If we reach here then we want to swap some local F points to C points

      for (ifree = 0; ifree < local_rows; ifree++) {

         // Bin the entries between 0 and 1
         // The top bin has entries greater than 0.9 (including greater than 1)
         bin_real = PetscFloorReal(diag_dom_ratio[ifree] * size_dom_bins);
         // A floor too big (or too small) for an int, or a NaN, can't be converted
         // in C, the Fortran conversion gives a negative int there, which the
         // check below puts in the top bin
         if (!(bin_real > -2147483648.0 && bin_real < 2147483647.0)) {
            bin = -1;
         } else {
            bin = PetscMin((int)bin_real + 1, size_dom_bins);
         }
         // If the diagonal dominance ratio is really large the expression above will overflow
         // the int to negative, so we just stick that in the top bin
         if (bin < 0) {
            bin = size_dom_bins;
         }
         dom_bins[bin - 1] = dom_bins[bin - 1] + 1;

      }

      // Do a fixed alpha_diag
      if (fraction_swap < 0) {
         swap_dom_val = -fraction_swap;

      // Otherwise swap everything bigger than a fixed fraction
      } else {

         // In order to reduce the size of the sort required, we have binned the entries into 1000 bins
         // Let's count backwards from the biggest entries to find which bin we know the nth_element is in
         // and then we only include those bins and higher into the sort
         bin_sum = 0;
         for (bin_boundary = size_dom_bins; bin_boundary >= 1; bin_boundary--) {
            bin_sum = bin_sum + dom_bins[bin_boundary - 1];
            if (bin_sum >= search_size) break;
         }
         // Now bin_boundary holds the bin whose lower boundary is guaranteed to be <= the n_th element

         // Rather than do any type of sort, just swap everything above that bin boundary
         // This will give a fraction_swap that is very close to that passed in as long as the
         // size of the bins is small
         swap_dom_val = (PetscReal)(bin_boundary - 1) / (PetscReal)size_dom_bins;

      }

      // Let's go and swap F points to C points
      for (ifree = 0; ifree < local_rows; ifree++) {

         // If this row only has a single diagonal entry, or is below the threshold we swap, skip
         if (diag_dom_ratio[ifree] == 0 || diag_dom_ratio[ifree] < swap_dom_val) continue;

         // This is the actual numbering in A, rather than Aff
         // Careful here to minus away the row_start of A, not Aff, as cf_markers_local is as big as A
         idx = is_pointer[ifree] - input_row_start;

         // Swap by multiplying by -1
         cf_markers_local[idx] = cf_markers_local[idx] * (-1);
      }
   }

   PetscCall(ISRestoreIndices(is_fine, &is_pointer));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------
