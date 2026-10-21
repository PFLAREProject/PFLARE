/*
  C port of CR_Splitting.F90 (module cr_splitting)

  The CF markers are int arrays (the Fortran integer) with one entry per local
  row of input_mat. The loops over the fine rows are 0-based
*/

#include <petscmat.h>
#include "pflare.h"
#include "pflareimpl.h"
#include "PETSc_Helper.h"
#include "PMISR_Module.h"
#include "Pflare_Random.h"
#include "Approx_Inverse_Setup.h"
#include "Weighted_Jacobi.h"
#include "CR_Splitting.h"

// The CR relaxation whose error decides the splitting is the approximate
// inverse of Aff built with the same inverse type, polynomial order, sparsity
// order and diagonal scaling that PCAIR uses for its Aff inverse, so the CR
// rate certifies the one-application contraction of the F-solve actually
// applied in the hierarchy
// It is ALWAYS built assembled, never matrix-free, even when
// -pc_air_matrix_free_polys is on: the ideal restrictor is always built from
// the assembled sparsified inverse (see AIR_Operators_Setup), which is
// weaker than the matrix-free polynomial, so the assembled inverse is the
// binding constraint on the splitting - certifying with the matrix-free
// polynomial under-coarsens and the restrictor quality degrades with
// problem size, while certifying with the assembled inverse guarantees the
// restrictor and the matrix-free smoother simply does better than predicted

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode cr_pass(Mat input_mat, IS is_fine, PetscReal target_cr_rate,
   int cr_inverse_type, int cr_poly_order, int cr_sparsity_order, PetscBool cr_diag_scale,
   PetscReal *cr_rate_achieved, PetscInt *n_swapped_global, int *cf_markers_local)
{
   // One pass of compatible relaxation (CR) coarsening - the CR splitting is
   // built entirely from repeated calls to this, starting from all points F
   // Runs nu applications of the CR relaxation on Aff with a zero rhs and a random
   // initial error (habituated CR - the C points are held at zero so the relaxation
   // never touches the whole matrix), then promotes an independent set of the
   // slowest converging F rows to C
   // The relaxation is the assembled approximate inverse with the settings
   // given by cr_inverse_type/cr_poly_order/cr_sparsity_order/cr_diag_scale
   // (mirroring AIRG's F-point smoothing) - see the module comment above for
   // why it is always assembled. Jacobi types and tiny Aff blocks (where an
   // Arnoldi of the polynomial order doesn't fit) use the inline Jacobi path,
   // which sanitizes zero diagonals
   // Candidates for promotion are the rows where the relaxed error remains
   // large relative to the biggest remaining error (the hypre CR candidate
   // measure), so rows where the relaxation diverges or stalls are promoted first
   // cr_rate_achieved returns the estimated CR rate rho = (||e_nu||/||e_0||)^(1/nu)
   // and the caller iterates until rho <= target_cr_rate or nothing is swapped

   // ~~~~~~

   // Local
   Mat Aff, temp_mat, inv_Aff = NULL;
   Vec diag_vec, scale_vec, e_vec, work_vec, work2_vec;
   PetscInt fine_local, fine_global, global_rows, global_cols;
   PetscInt input_row_start, input_row_end_plus_one, idx, ifree;
   PetscInt n_swapped_local;
   const PetscInt *is_pointer;
   PetscScalar *vec_array;
   PetscScalar *e_array;
   const PetscScalar *e_array_read;
   PetscReal *cr_measure;
   PetscReal *rand_full;
   int *cf_markers_local_aff;
   PetscBool *forced_c;
   PetscBool extracted;
   PetscReal norm_inf, weight, norm_e0, norm_enu, e_norm_inf;
   PetscMPIInt comm_rank;
   int sweep, max_luby_steps, nu;
   PetscBool use_poly, any_forced_c;
   MPI_Comm MPI_COMM_MATRIX;

   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)input_mat, &MPI_COMM_MATRIX));
   PetscCallMPI(MPI_Comm_rank(MPI_COMM_MATRIX, &comm_rank));

   *cr_rate_achieved = 0;
   *n_swapped_global = 0;

   PetscCall(ISGetLocalSize(is_fine, &fine_local));
   PetscCall(ISGetSize(is_fine, &fine_global));

   // Nothing to relax on - every point is C
   // This is on the global size so all ranks agree, everything below is collective
   if (fine_global == 0) PetscFunctionReturn(PETSC_SUCCESS);

   PetscCall(MatGetOwnershipRange(input_mat, &input_row_start, &input_row_end_plus_one));
   PetscCall(MatGetSize(input_mat, &global_rows, &global_cols));
   PetscCall(ISGetIndices(is_fine, &is_pointer));

   // While every point is still F (the initial rounds, as CR starts with an
   // empty C set) Aff is just the input matrix so we can skip the extraction
   extracted = fine_global != global_rows ? PETSC_TRUE : PETSC_FALSE;
   if (extracted) {
      PetscCall(MatCreateSubMatrixWrapper(input_mat, is_fine, is_fine,
               MAT_INITIAL_MATRIX, &Aff, NULL, NULL, NULL, NULL));
   } else {
      Aff = input_mat;
   }

   // Allocated and set to false
   PetscCall(PetscCalloc1(fine_local, &forced_c));

   // Jacobi inverse types use the inline path below (which sanitizes zero
   // diagonals), as does any Aff too tiny to fit an Arnoldi of the
   // polynomial order
   use_poly = (cr_inverse_type != PFLAREINV_WJACOBI &&
              cr_inverse_type != PFLAREINV_JACOBI &&
              cr_inverse_type != PFLAREINV_L1JACOBI &&
              fine_global > cr_poly_order + 1) ? PETSC_TRUE : PETSC_FALSE;
   nu = PFLARE_CR_NU;
   if (use_poly) nu = PFLARE_CR_NU_POLY;

   if (use_poly) {

      // ~~~~~~~~
      // Assembled approximate inverse of Aff with the same settings as the
      // F-point smoothing AIRG applies in the hierarchy, so the CR rate
      // certifies the contraction of the actual F-solve including the
      // sparsity truncation
      // Always assembled, never matrix-free - see the module comment
      // ~~~~~~~~
      // Without the optional coefficients argument, as the Fortran called it
      PetscCall(calculate_and_build_approximate_inverse(Aff, cr_inverse_type,
               cr_poly_order, cr_sparsity_order,
               PETSC_FALSE, cr_diag_scale, PETSC_FALSE,
               &inv_Aff, NULL, NULL, NULL));

   } else {

      // ~~~~~~~~
      // Build the Jacobi inverse (weighted, unweighted or l1) of the diagonal of Aff
      // This uses the same weights as calculate_and_build_weighted_jacobi_inverse
      // but we sanitize exactly zero diagonals so they can't poison the weights
      // with infs/nans - those rows can never be relaxed so they are force
      // promoted to C (and their inverse zeroed) below
      // ~~~~~~~~
      PetscCall(MatCreateVecs(Aff, NULL, &diag_vec));
      PetscCall(MatGetDiagonal(Aff, diag_vec));
      PetscCall(VecGetArray(diag_vec, &vec_array));
      for (ifree = 0; ifree < fine_local; ifree++) {
         if (vec_array[ifree] == 0.0) {
            vec_array[ifree] = 1.0;
            forced_c[ifree] = PETSC_TRUE;
         }
      }
      PetscCall(VecRestoreArray(diag_vec, &vec_array));

      // This is the weight that hypre's BoomerAMG computes with relax_weight = 0
      // (see par_amg_setup.c)
      // 4 / ( 3 * || D^(-1/2) * Aff * D^(-1/2) ||_inf )
      // Note the hypre docs for HYPRE_BoomerAMGSetRelaxWt quote 3 / (4 * ||.||)
      // but the code uses 4 / (3 * ||.||)
      // Unweighted plain Jacobi if that's what the hierarchy smooths with
      // and l1 Jacobi has its own weight on each row
      weight = 1.0;
      if (cr_inverse_type == PFLAREINV_WJACOBI) {
         PetscCall(MatDuplicate(Aff, MAT_COPY_VALUES, &temp_mat));
         PetscCall(VecDuplicate(diag_vec, &scale_vec));
         PetscCall(VecCopy(diag_vec, scale_vec));
         PetscCall(VecSqrtAbs(scale_vec));
         PetscCall(VecReciprocal(scale_vec));
         PetscCall(MatDiagonalScale(temp_mat, scale_vec, scale_vec));
         PetscCall(MatNorm(temp_mat, NORM_INFINITY, &norm_inf));
         PetscCall(MatDestroy(&temp_mat));
         PetscCall(VecDestroy(&scale_vec));
         // In PetscReal, as the Fortran's 4.0 and 3.0 are default reals that are exact in any precision
         if (norm_inf != 0.0) weight = (PetscReal)4.0 / ((PetscReal)3.0 * norm_inf);
      }

      // Invert and apply the weight, zeroing the rows we can't relax
      if (cr_inverse_type == PFLAREINV_L1JACOBI) {
         PetscCall(calculate_l1_jacobi_inverse_diagonal(Aff, diag_vec));
      } else {
         PetscCall(VecReciprocal(diag_vec));
         PetscCall(VecScale(diag_vec, weight));
      }
      // any(forced_c)
      any_forced_c = PETSC_FALSE;
      for (ifree = 0; ifree < fine_local; ifree++) {
         if (forced_c[ifree]) any_forced_c = PETSC_TRUE;
      }
      if (any_forced_c) {
         PetscCall(VecGetArray(diag_vec, &vec_array));
         for (ifree = 0; ifree < fine_local; ifree++) {
            if (forced_c[ifree]) vec_array[ifree] = 0.0;
         }
         PetscCall(VecRestoreArray(diag_vec, &vec_array));
      }

      // The matrix takes ownership of diag_vec
      PetscCall(MatCreateDiagonal(diag_vec, &inv_Aff));
      PetscCall(VecDestroy(&diag_vec));

   }

   // ~~~~~~~~
   // Random initial error, generated on the host over all the local rows of A
   // with the same seeding convention as the pmisr/ddc, so any future device
   // port of this routine is bit-identical
   // ~~~~~~~~
   // At least one entry, so that like the Fortran's (possibly zero-size) allocation
   // the array the Fortran pflare_random_fill is handed is never NULL, which
   // PetscMalloc1 gives for zero entries
   PetscCall(PetscMalloc1(PetscMax(input_row_end_plus_one - input_row_start, 1), &rand_full));
   pflare_random_fill(comm_rank, input_row_end_plus_one - input_row_start, rand_full);

   PetscCall(MatCreateVecs(Aff, &e_vec, &work_vec));
   PetscCall(VecDuplicate(e_vec, &work2_vec));

   // Scatter the random error into the F rows
   PetscCall(VecGetArray(e_vec, &e_array));
   for (ifree = 0; ifree < fine_local; ifree++) {
      e_array[ifree] = rand_full[is_pointer[ifree] - input_row_start];
   }
   PetscCall(VecRestoreArray(e_vec, &e_array));

   PetscCall(VecNorm(e_vec, NORM_2, &norm_e0));

   // ~~~~~~~~
   // The relaxation - nu applications of e <- e - M Aff e, where M is the
   // approximate inverse of Aff built above
   // All default petsc mat/vec ops so this runs on the device for kokkos types
   // ~~~~~~~~
   for (sweep = 0; sweep < nu; sweep++) {
      PetscCall(MatMult(Aff, e_vec, work_vec));
      PetscCall(MatMult(inv_Aff, work_vec, work2_vec));
      PetscCall(VecAXPY(e_vec, PFLARE_MINUS_ONE, work2_vec));
   }

   PetscCall(VecNorm(e_vec, NORM_2, &norm_enu));
   if (norm_e0 != 0.0) {
      *cr_rate_achieved = PetscPowReal(norm_enu / norm_e0, PFLARE_ONE / (PetscReal)nu);
   }

   // Only promote F points if we haven't already hit the target rate
   // The rate is computed from global norms so all ranks take the same branch
   if (*cr_rate_achieved > target_cr_rate) {

      // ~~~~~~~~
      // Candidate F rows are those where the relaxed error is still big
      // relative to the biggest remaining error anywhere - the hypre CR
      // candidate measure. Wherever error persists the relaxation is failing,
      // and promoting the biggest entries cuts the slowly converging modes -
      // a per-row convergence factor can look fine on every row while a
      // global slow mode remains, which stalls the coarsening
      // ~~~~~~~~
      PetscCall(VecNorm(e_vec, NORM_INFINITY, &e_norm_inf));

      PetscCall(PetscMalloc1(fine_local, &cr_measure));
      // Allocated and set to zero
      PetscCall(PetscCalloc1(fine_local, &cf_markers_local_aff));

      PetscCall(VecGetArrayRead(e_vec, &e_array_read));
      for (ifree = 0; ifree < fine_local; ifree++) {
         if (forced_c[ifree]) {
            // Rows we couldn't relax (zero diagonal) always get promoted first
            cr_measure[ifree] = 2.0;
         } else if (e_norm_inf != 0.0) {
            cr_measure[ifree] = PetscAbsScalar(e_array_read[ifree]) / e_norm_inf;
         } else {
            cr_measure[ifree] = 0.0;
         }
      }
      PetscCall(VecRestoreArrayRead(e_vec, &e_array_read));

      // ~~~~~~~~
      // Same trick as the ddc trigger path - the pmisr tags the points with the
      // smallest measure as "F", so feed in 10 - candidate measure and it picks
      // an independent set of the biggest remaining errors
      // The base of 10 keeps abs(measure) .ge. 1, which the pmisr requires
      // (the candidate measure is in [0, 2]), and the scaled down random
      // breaks ties without changing the ordering very much
      // There is always at least one candidate when the rate is above the
      // target (the row holding the biggest error), so every pass promotes
      // something and the outer loop can't stall short of the target
      // (in double precision, as the Fortran's 10d0 and 1d10 are)
      // ~~~~~~~~
      for (ifree = 0; ifree < fine_local; ifree++) {
         if (cr_measure[ifree] < PFLARE_CR_CANDIDATE) {
            // Error already small here - never swap these
            cr_measure[ifree] = PETSC_MAX_REAL;
            cf_markers_local_aff[ifree] = PFLARE_C_POINT;
         } else {
            cr_measure[ifree] = (PetscReal)(10.0 - ((double)cr_measure[ifree] -
                  (double)rand_full[is_pointer[ifree] - input_row_start] / 1e10));
         }
      }

      // Call PMISR with as many steps as necessary
      // Uses the implicit transpose version which takes Aff directly
      // and handles Aff+Aff^T internally without forming the explicit sum
      max_luby_steps = -1;
      PetscCall(pmisr_existing_measure_implicit_transpose(Aff,
               max_luby_steps, PETSC_FALSE,
               cr_measure, cf_markers_local_aff, NULL));

      // Swap the slowest converging rows to C points
      n_swapped_local = 0;
      for (ifree = 0; ifree < fine_local; ifree++) {

         // The pmisr marked the points we want to swap as F
         if (cf_markers_local_aff[ifree] == PFLARE_F_POINT) {
            // This is the actual numbering in A, rather than Aff
            // Careful here to minus away the row_start of A, not Aff
            // as cf_markers_local is as big as A
            idx = is_pointer[ifree] - input_row_start;

            // Swap by multiplying by -1
            cf_markers_local[idx] = cf_markers_local[idx] * (-1);
            n_swapped_local = n_swapped_local + 1;
         }
      }

      // The caller needs to know if anything changed globally to keep all
      // ranks consistent in the outer loop
      PetscCallMPI(MPI_Allreduce(&n_swapped_local, n_swapped_global, 1, MPIU_INT,
               MPI_SUM, MPI_COMM_MATRIX));

      PetscCall(PetscFree(cr_measure));
      PetscCall(PetscFree(cf_markers_local_aff));
   }

   // Cleanup
   PetscCall(VecDestroy(&e_vec));
   PetscCall(VecDestroy(&work_vec));
   PetscCall(VecDestroy(&work2_vec));
   PetscCall(MatDestroy(&inv_Aff));
   if (extracted) PetscCall(MatDestroy(&Aff));
   PetscCall(ISRestoreIndices(is_fine, &is_pointer));
   PetscCall(PetscFree(rand_full));
   PetscCall(PetscFree(forced_c));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------
