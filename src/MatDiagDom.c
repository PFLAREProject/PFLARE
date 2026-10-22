/*
  C port of MatDiagDom.F90 (module matdiagdom)

  diag_dom_ratio is an allocatable intent(out) in the Fortran, so it is
  freed on entry here (the caller passes NULL or an array from PetscMalloc1)
  and allocated where the Fortran allocated it. The loops over the fine rows
  and the row entries are 0-based
  The module parameters dd_ratio_abs_tol and dd_ratio_rel_tol are file-static
*/

#include <petscmat.h>
#include <math.h>
#include "pflareimpl.h"
#include "PETSc_Helper.h"
#include "Device_Datak.h"
#include "MatDiagDomk.h"
#include "MatDiagDom.h"

// Only used with Kokkos, so only defined with it (an unused static is a warning)
#if defined(PETSC_HAVE_KOKKOS)
static const PetscReal dd_ratio_abs_tol = PFLARE_DD_RATIO_ABS_TOL;
static const PetscReal dd_ratio_rel_tol = PFLARE_DD_RATIO_REL_TOL;
#endif

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode MatDiagDomRatio(Mat input_mat, IS is_fine, const int *cf_markers_local, PetscReal **diag_dom_ratio, PetscReal *max_dd_ratio_achieved)
{
   // Wrapper for diagonal-dominance ratio computation.
   // Chooses Kokkos or CPU implementation and optionally compares the
   // resulting host ratios in debug mode.

#if defined(PETSC_HAVE_KOKKOS)
   PetscInt ifree;
   MatType mat_type;
   PetscReal tol, diff;
   Mat A_array;
   PetscInt local_rows_aff_kokkos;
   PetscReal *diag_dom_ratio_ptr;
   PetscReal *diag_dom_ratio_cpu = NULL;
   PetscReal max_dd_ratio_cpu;
   PetscBool is_kokkos;
#endif

   PetscFunctionBegin;

   // The Fortran deallocates an intent(out) allocatable on entry
   PetscCall(PetscFree(*diag_dom_ratio));

#if defined(PETSC_HAVE_KOKKOS)
   PetscCall(MatGetType(input_mat, &mat_type));
   PetscCall(PetscStrcmpAny(mat_type, &is_kokkos, MATMPIAIJKOKKOS, MATSEQAIJKOKKOS, MATAIJKOKKOS, ""));
   if (is_kokkos) {

      A_array = input_mat;
      local_rows_aff_kokkos = 0;
      *max_dd_ratio_achieved = 0.0;

      MatDiagDomRatio_kokkos(&A_array, max_dd_ratio_achieved, &local_rows_aff_kokkos);

      if (kokkos_debug()) {
         PetscCall(PetscMalloc1(local_rows_aff_kokkos, diag_dom_ratio));
         if (local_rows_aff_kokkos > 0) {
            diag_dom_ratio_ptr = *diag_dom_ratio;
            copy_diag_dom_ratio_d2h(diag_dom_ratio_ptr);
         }

         PetscCall(MatDiagDomRatio_cpu(input_mat, is_fine, cf_markers_local,
                                  &diag_dom_ratio_cpu, &max_dd_ratio_cpu));

         tol = dd_ratio_abs_tol + dd_ratio_rel_tol * PetscMax(PetscAbsReal(max_dd_ratio_cpu), PetscAbsReal(*max_dd_ratio_achieved));
         PetscCheck(!(PetscAbsReal(max_dd_ratio_cpu - *max_dd_ratio_achieved) > tol), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Kokkos and CPU MatDiagDomRatio global max do not match");

         for (ifree = 0; ifree < local_rows_aff_kokkos; ifree++) {
            diff = PetscAbsReal((*diag_dom_ratio)[ifree] - diag_dom_ratio_cpu[ifree]);
            tol = dd_ratio_abs_tol + dd_ratio_rel_tol * PetscMax(PetscAbsReal((*diag_dom_ratio)[ifree]), PetscAbsReal(diag_dom_ratio_cpu[ifree]));
            PetscCheck(!(diff > tol), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Kokkos and CPU MatDiagDomRatio entries do not match");
         }

         PetscCall(PetscFree(diag_dom_ratio_cpu));
      }
   } else {
      PetscCall(MatDiagDomRatio_cpu(input_mat, is_fine, cf_markers_local,
                               diag_dom_ratio, max_dd_ratio_achieved));
   }
#else
   PetscCall(MatDiagDomRatio_cpu(input_mat, is_fine, cf_markers_local,
                            diag_dom_ratio, max_dd_ratio_achieved));
#endif

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode MatDiagDomRatio_cpu(Mat input_mat, IS is_fine, const int *cf_markers_local, PetscReal **diag_dom_ratio, PetscReal *max_dd_ratio_achieved)
{
   // Compute diagonal dominance ratio over local fine rows of input_mat
   // without extracting Aff. This mirrors the Kokkos MatDiagDomRatio path:
   // sum abs(F-neighbour off-diagonals) / abs(F-diagonal), with nonlocal
   // F markers obtained from the matrix halo scatter.
   //
   // The halo scatter uses input_mat's own mult PetscSF (via MatGetMultPetscSF
   // inside the C scatter routines); we build a matching leaf Vec locally and
   // destroy it at the end.

   // ~~~~~~

   // Local
   PetscInt local_rows, local_cols, global_rows, global_cols, fine_size;
   PetscInt input_row_start, input_row_end_plus_one;
   PetscInt ifree, jfree, local_row, target_col, rows_ao, cols_ao;
   PetscInt n_ad, n_ao;
   PetscInt one = 1;
   PetscMPIInt comm_size;
   MPI_Comm MPI_COMM_MATRIX;
   Mat Ad, Ao = NULL;
   Vec cf_markers_vec = NULL, leaf_vec = NULL;
   PetscSF sf = NULL;
   const PetscInt *is_pointer = NULL, *colmap = NULL;
   const PetscInt *ad_ia = NULL, *ad_ja = NULL, *ao_ia = NULL, *ao_ja = NULL;
   const PetscScalar *ad_vals = NULL, *ao_vals = NULL;
   // The raw Vec array (PetscScalar) returned by VecGetArrayRead
   const PetscScalar *cf_markers_nonlocal = NULL;
   PetscReal *cf_markers_local_real = NULL;
   PetscInt shift = 0;
   PetscBool symmetric = PETSC_FALSE, inodecompressed = PETSC_FALSE, done;
   PetscReal diag_val, off_diag_sum;
   PetscReal max_dd_ratio_local;
   PetscBool mpi;

   // ~~~~~~

   PetscFunctionBegin;

   // The Fortran deallocates an intent(out) allocatable on entry
   PetscCall(PetscFree(*diag_dom_ratio));

   PetscCall(PetscObjectGetComm((PetscObject)input_mat, &MPI_COMM_MATRIX));
   PetscCall(MatGetLocalSize(input_mat, &local_rows, &local_cols));
   PetscCall(MatGetSize(input_mat, &global_rows, &global_cols));
   PetscCall(MatGetOwnershipRange(input_mat, &input_row_start, &input_row_end_plus_one));
   PetscCall(ISGetLocalSize(is_fine, &fine_size));
   PetscCall(ISGetIndices(is_fine, &is_pointer));

   // Allocated and set to zero
   PetscCall(PetscCalloc1(fine_size, diag_dom_ratio));

   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));
   mpi = comm_size != 1 ? PETSC_TRUE : PETSC_FALSE;

   if (mpi) {
      PetscCall(MatMPIAIJGetSeqAIJ(input_mat, &Ad, &Ao, &colmap));
      PetscCall(MatGetSize(Ao, &rows_ao, &cols_ao));
   } else {
      Ad = input_mat;
   }

   // Get pointers to the local/off-diagonal CSR structures.
   // This mirrors the Kokkos path, which accesses local and nonlocal CSR directly.
   PetscCall(MatGetRowIJ(Ad, shift, symmetric, inodecompressed, &n_ad, &ad_ia, &ad_ja, &done));
   PetscCheck(done, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Pointers not set in call to MatGetRowIJ");

   if (mpi) {
      PetscCall(MatGetRowIJ(Ao, shift, symmetric, inodecompressed, &n_ao, &ao_ia, &ao_ja, &done));
      PetscCheck(done, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Pointers not set in call to MatGetRowIJ");
   }

   PetscCall(MatSeqAIJGetArrayRead(Ad, &ad_vals));

   // Off-diagonal rows require a halo exchange of cf markers.
   // Start and finish the scatter once, then reuse the received nonlocal markers
   // while looping over all local fine rows.
   if (mpi) {
      PetscCall(MatSeqAIJGetArrayRead(Ao, &ao_vals));

      PetscCall(PetscMalloc1(local_rows, &cf_markers_local_real));
      for (ifree = 0; ifree < local_rows; ifree++) cf_markers_local_real[ifree] = (PetscReal)cf_markers_local[ifree];

      PetscCall(VecCreateMPIWithArray(MPI_COMM_MATRIX, one, local_rows, global_rows,
            cf_markers_local_real, &cf_markers_vec));

      // Leaf Vec sized/typed to match the off-diagonal block
      PetscCall(MatCreateVecs(Ao, &leaf_vec, NULL));

      // Use input_mat's mult SF + the local leaf Vec for the single halo scatter.
      PetscCall(MatGetMultPetscSF(input_mat, &sf));
      PetscCall(VecScatterBegin(sf, cf_markers_vec, leaf_vec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(sf, cf_markers_vec, leaf_vec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecGetArrayRead(leaf_vec, &cf_markers_nonlocal));
   }

   // Compute diagonal-dominance sums over the local fine-row list.
   // For each row: accumulate abs(off-diagonal) over F neighbors only,
   // store abs(diagonal) for an F diagonal entry, then form ratio.
   for (ifree = 0; ifree < fine_size; ifree++) {
      local_row = is_pointer[ifree] - input_row_start;
      diag_val = 0.0;
      off_diag_sum = 0.0;

      for (jfree = ad_ia[local_row]; jfree < ad_ia[local_row + 1]; jfree++) {
         target_col = ad_ja[jfree];

         if (cf_markers_local[target_col] != PFLARE_F_POINT) continue;

         if (target_col == local_row) {
            diag_val = PetscAbsScalar(ad_vals[jfree]);
         } else {
            off_diag_sum = off_diag_sum + PetscAbsScalar(ad_vals[jfree]);
         }
      }

      if (mpi) {
         for (jfree = ao_ia[local_row]; jfree < ao_ia[local_row + 1]; jfree++) {
            target_col = ao_ja[jfree];

            // nint, which rounds half away from zero as round does
            if ((int)round((double)PetscRealPart(cf_markers_nonlocal[target_col])) != PFLARE_F_POINT) continue;

            off_diag_sum = off_diag_sum + PetscAbsScalar(ao_vals[jfree]);
         }
      }

      // If no diagonal was found, keep ratio at zero.
      // This matches the Kokkos behavior for rows without a diagonal entry.
      if (diag_val != 0.0) {
         (*diag_dom_ratio)[ifree] = off_diag_sum / diag_val;
      }
   }

   // Cleanup for halo scatter resources.
   if (mpi) {
      PetscCall(VecRestoreArrayRead(leaf_vec, &cf_markers_nonlocal));
      PetscCall(VecDestroy(&cf_markers_vec));
      PetscCall(VecDestroy(&leaf_vec));
      PetscCall(PetscFree(cf_markers_local_real));
   }

   // Restore CSR pointers before returning.
   PetscCall(MatSeqAIJRestoreArrayRead(Ad, &ad_vals));
   PetscCall(MatRestoreRowIJ(Ad, shift, symmetric, inodecompressed, &n_ad, &ad_ia, &ad_ja, &done));
   if (mpi) {
      PetscCall(MatSeqAIJRestoreArrayRead(Ao, &ao_vals));
      PetscCall(MatRestoreRowIJ(Ao, shift, symmetric, inodecompressed, &n_ao, &ao_ia, &ao_ja, &done));
   }

   PetscCall(ISRestoreIndices(is_fine, &is_pointer));

   if (fine_size == 0) {
      max_dd_ratio_local = 0.0;
   } else {
      // maxval
      max_dd_ratio_local = (*diag_dom_ratio)[0];
      for (ifree = 1; ifree < fine_size; ifree++) {
         if ((*diag_dom_ratio)[ifree] > max_dd_ratio_local) max_dd_ratio_local = (*diag_dom_ratio)[ifree];
      }
   }
   PetscCallMPI(MPI_Allreduce(&max_dd_ratio_local, max_dd_ratio_achieved, 1, MPIU_REAL,
                         MPI_MAX, MPI_COMM_MATRIX));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------
