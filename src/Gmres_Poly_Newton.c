/*
  C port of Gmres_Poly_Newton.F90 (module gmres_poly_newton)

  The roots are column-major in coefficients, the real parts in the first
  column and the imaginary parts in the second, with coefficients_rows rows.
  The loops over the roots that pair up complex conjugates keep the Fortran
  1-based root index (i, term, i_sparse, ...) as their arithmetic is easier to
  check side by side that way, and read the roots through the REAL_ROOT and
  IMAG_ROOT macros (the Fortran coefficients(i, 1) and coefficients(i, 2)). The
  status_output array is read the same way through STATUS. The other loops
  are 0-based. modified_leja returns 0-based indices; the Fortran wrapper in
  src/fortran/Gmres_Poly_Newton.F90 hands 1-based ones to Fortran callers
*/

#include <petscmat.h>
#include <petscblaslapack.h>
#include "pflare.h"
#include "pflareimpl.h"
#include "Sorting.h"
#include "Gmres_Poly_Newtonk.h"
#include "Matshell_Data_Type.h"
#include "Gmres_Poly_Apply.h"
#include "PETSc_Helper.h"
#include "Gmres_Poly.h"
#include "Gmres_Poly_Newton.h"

// The Fortran coefficients(i, 1), coefficients(i, 2) and status_output(i, j), with i and j 1-based
#define REAL_ROOT(i)   coefficients[(i) - 1]
#define IMAG_ROOT(i)   coefficients[(i) - 1 + coefficients_rows]
#define STATUS(i, j)   status_output[(i) - 1 + ((j) - 1) * status_rows]

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode modified_leja(PetscReal real_roots[], PetscReal imag_roots[], PetscInt roots_size, PetscInt **indices)
{
   // Computes a modified leja ordering of the eigenvalues
   // and re-orders in place
   // The roots must be passed in with complex conjugate pairs next to each other
   // with positive imag e'vals first
   // indices is allocated here (roots_size entries, 0-based) and the caller frees it

   // ~~~~~~
   PetscInt i_loc, k_loc, counter;
   PetscInt max_loc;
   PetscReal *magnitude;
   PetscReal a, b, squares, max_mag;
   PetscBool *sorted;
   PetscBool found;
   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscMalloc1(roots_size, &magnitude));
   PetscCall(PetscMalloc1(roots_size, indices));
   PetscCall(PetscMalloc1(roots_size, &sorted));

   // Compute the magnitudes of the evals
   for (i_loc = 0; i_loc < roots_size; i_loc++) {
      magnitude[i_loc] = PetscSqrtReal(real_roots[i_loc] * real_roots[i_loc] + imag_roots[i_loc] * imag_roots[i_loc]);
   }
   // Find the biggest
   // This is the Fortran maxloc, the first of the biggest (ignoring NaNs, the first
   // entry if they are all NaN)
   max_loc = 0;
   for (i_loc = 0; i_loc < roots_size; i_loc++) {
      if (!PetscIsNanReal(magnitude[i_loc])) {
         max_loc = i_loc;
         break;
      }
   }
   for (i_loc = max_loc + 1; i_loc < roots_size; i_loc++) {
      if (magnitude[i_loc] > magnitude[max_loc]) max_loc = i_loc;
   }
   // counter is the number of entries of indices we've filled
   counter = 0;

   for (i_loc = 0; i_loc < roots_size; i_loc++) sorted[i_loc] = PETSC_FALSE;

   // That is our first entry
   (*indices)[counter] = max_loc;
   sorted[(*indices)[counter]] = PETSC_TRUE;
   counter = counter + 1;
   // If it was imaginary its complex conjugate is next to this one
   if (imag_roots[(*indices)[counter - 1]] != 0.0) {
      // dgeev returns roots with the positive imaginary part first
      if (imag_roots[(*indices)[counter - 1]] > 0) {
         // So if positive we know the conjugate is ahead one
         (*indices)[counter] = (*indices)[counter - 1] + 1;
      } else {
         // If negative we know the conjugate is one behind
         (*indices)[counter] = (*indices)[counter - 1] - 1;
      }
      sorted[(*indices)[counter]] = PETSC_TRUE;
      counter = counter + 1;
   }

   // Do while we still have some sorting to do
   while (counter < roots_size) {

      max_mag = -PETSC_MAX_REAL;
      found = PETSC_FALSE;

      // For each value compute a product of differences
      for (i_loc = 0; i_loc < roots_size; i_loc++) {

         // Skip any we've already sorted
         if (sorted[i_loc]) continue;
         magnitude[i_loc] = 1.0;

         // Loop over all those we've sorted so far
         for (k_loc = 0; k_loc < counter; k_loc++) {

            // Distance
            a = real_roots[i_loc] - real_roots[(*indices)[k_loc]];
            b = imag_roots[i_loc] - imag_roots[(*indices)[k_loc]];

            // If we have repeated roots this will be exactly zero, and magnitude will be -infinity
            squares = a * a + b * b;
            magnitude[i_loc] = magnitude[i_loc] + PetscLog10Real(PetscSqrtReal(squares));
         }

         // Store the biggest as we're going
         if (magnitude[i_loc] > max_mag) {
            max_mag = magnitude[i_loc];
            max_loc = i_loc;
            found = PETSC_TRUE;
         }
      }

      // If we found nothing (ie the only unsorted are repeated roots with zero distance)
      // just have the next entry in the list
      // Note we can't test max_mag < 0 here, as the log of the product of distances
      // is legitimately negative whenever that product is small (e.g., tightly
      // clustered or small magnitude eigenvalues), which would make the ordering scale dependent
      if (!found) {
         for (i_loc = 0; i_loc < roots_size; i_loc++) {
            if (!sorted[i_loc]) {
               max_loc = i_loc;
               break;
            }
         }
      }

      // The new index is the biggest in distance
      (*indices)[counter] = max_loc;
      sorted[(*indices)[counter]] = PETSC_TRUE;
      counter = counter + 1;
      // If it was imaginary its complex conjugate is next to this one
      if (imag_roots[(*indices)[counter - 1]] != 0.0) {
         // dgeev returns roots with the positive imaginary part first
         // We did actually find a situation where due to rounding differnces, the
         // conjugate with negative imaginary root had a bigger product of factors
         // (by 1e-14) than the positive imaginary root, so we can't guarantee we hit
         // the positive one first
         if (imag_roots[(*indices)[counter - 1]] > 0) {
            // So if positive we know the conjugate is ahead one

            (*indices)[counter] = (*indices)[counter - 1] + 1;
         } else {
            // If negative we know the conjugate is one behind
            (*indices)[counter] = (*indices)[counter - 1] - 1;
         }
         sorted[(*indices)[counter]] = PETSC_TRUE;
         counter = counter + 1;
      }
   }

   PetscCall(PetscFree(magnitude));
   PetscCall(PetscFree(sorted));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode cluster_eigenvalues_stable(PetscReal real_roots[], PetscReal imag_roots[], PetscInt roots_size, PetscReal rel_tol, PetscReal abs_tol)
{
   // Clustering of (possibly complex) harmonic Ritz values.
   // Numerically distinct clusters are moved to the front.
   // Remaining entries are set to zero.
   // Skips eigenvalues that are exactly zero (both real and imag parts).
   //
   // Inputs:
   //   real_roots, imag_roots : eigenvalues (length k)
   //   rel_tol               : relative tolerance (suggest sqrt(eps) ~ 1e-8)
   //   abs_tol               : absolute tolerance (suggest eps * ||H|| ~ 1e-15)
   //
   // Outputs:
   //   real_roots, imag_roots : clustered eigenvalues at front, zeros after

   // ~~~~~~
   PetscInt i, j, n, n_unique, cluster_size;
   PetscBool *used;
   PetscReal dist, mag_i, mag_j, scale;
   PetscReal sum_real, sum_imag;
   PetscReal *rtmp, *itmp;
   // ~~~~~~

   PetscFunctionBegin;

   n = roots_size;

   PetscCall(PetscMalloc1(n, &used));
   PetscCall(PetscMalloc1(n, &rtmp));
   PetscCall(PetscMalloc1(n, &itmp));

   for (i = 0; i < n; i++) used[i] = PETSC_FALSE;
   n_unique = 0;

   // ---------------------------------------------------------
   // All-pairs clustering
   // ---------------------------------------------------------
   for (i = 0; i < n; i++) {

      if (used[i]) continue;

      // Skip eigenvalues that are exactly zero
      if (real_roots[i] == 0.0 && imag_roots[i] == 0.0) {
         used[i] = PETSC_TRUE;
         continue;
      }

      // Start new cluster with eigenvalue i
      sum_real = real_roots[i];
      sum_imag = imag_roots[i];
      cluster_size = 1;
      used[i] = PETSC_TRUE;

      mag_i = PetscSqrtReal(real_roots[i] * real_roots[i] + imag_roots[i] * imag_roots[i]);

      // Look for all other eigenvalues close to this one
      for (j = i + 1; j < n; j++) {

         if (used[j]) continue;

         // Skip exactly zero eigenvalues
         if (real_roots[j] == 0.0 && imag_roots[j] == 0.0) {
            used[j] = PETSC_TRUE;
            continue;
         }

         mag_j = PetscSqrtReal(real_roots[j] * real_roots[j] + imag_roots[j] * imag_roots[j]);

         // Distance between eigenvalues
         dist = PetscSqrtReal((real_roots[j] - real_roots[i]) * (real_roots[j] - real_roots[i]) +
                              (imag_roots[j] - imag_roots[i]) * (imag_roots[j] - imag_roots[i]));

         // Use the larger magnitude for relative scaling
         scale = PetscMax(PetscMax(mag_i, mag_j), PFLARE_ONE);

         // Check if within tolerance
         if (dist <= abs_tol + rel_tol * scale) {
            sum_real = sum_real + real_roots[j];
            sum_imag = sum_imag + imag_roots[j];
            cluster_size = cluster_size + 1;
            used[j] = PETSC_TRUE;
         }
      }

      // Compute cluster centroid (mean)
      rtmp[n_unique] = sum_real / (PetscReal)cluster_size;
      itmp[n_unique] = sum_imag / (PetscReal)cluster_size;
      n_unique = n_unique + 1;
   }

   // ---------------------------------------------------------
   // Output compact form
   // ---------------------------------------------------------
   for (i = 0; i < n; i++) {
      real_roots[i] = 0.0;
      imag_roots[i] = 0.0;
   }

   for (i = 0; i < n_unique; i++) {
      real_roots[i] = rtmp[i];
      imag_roots[i] = itmp[i];
   }

   PetscCall(PetscFree(used));
   PetscCall(PetscFree(rtmp));
   PetscCall(PetscFree(itmp));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode compute_extra_roots(PetscReal real_roots[], PetscReal imag_roots[], PetscInt roots_size,
   PetscReal **real_roots_output, PetscReal **imag_roots_output, PetscInt *roots_output_size)
{
   // Add extra roots for stability
   // Computes the product of factors for each eigenvalue and adds extra copies
   // of roots that have large products (to improve polynomial stability)
   // Only non-zero eigenvalues should be passed in
   // real_roots_output, imag_roots_output are allocated and filled with the original
   // roots plus any extra (roots_output_size entries each), the caller frees them

   // ~~~~~~
   PetscInt i_loc, j_loc, k_loc, n_roots, total_extra, counter;
   PetscReal a, b, c, d, div_real, div_imag, div_mag;
   PetscReal *pof;
   int *extra_pair_roots, *overflow;
   // ~~~~~~

   PetscFunctionBegin;

   n_roots = roots_size;

   PetscCall(PetscMalloc1(n_roots, &pof));
   PetscCall(PetscMalloc1(n_roots, &extra_pair_roots));
   PetscCall(PetscMalloc1(n_roots, &overflow));

   // Compute the product of factors
   for (k_loc = 0; k_loc < n_roots; k_loc++) {
      pof[k_loc] = 1;
      extra_pair_roots[k_loc] = 0;
      overflow[k_loc] = 0;
   }
   total_extra = 0;
   for (k_loc = 0; k_loc < n_roots; k_loc++) {

      a = real_roots[k_loc];
      b = imag_roots[k_loc];

      // We have already computed pof for the positive imaginary complex conjugate
      if (b < 0) continue;

      // Skips eigenvalues that are numerically zero
      // This matches the zero test used when applying the polynomial, ie
      // a purely imaginary root (a == 0, b /= 0) is not zero
      if (b == 0.0 && PetscAbsReal(a) < PFLARE_TOL_ZERO) continue;
      if (a * a + b * b < PFLARE_TOL_ZERO) continue;

      // Compute product(k)_{i, j/=i} * | 1 - theta_j/theta_i|
      for (i_loc = 0; i_loc < n_roots; i_loc++) {

         // Skip
         if (k_loc == i_loc) continue;

         c = real_roots[i_loc];
         d = imag_roots[i_loc];

         // Skips eigenvalues that are numerically zero
         if (d == 0.0 && PetscAbsReal(c) < PFLARE_TOL_ZERO) continue;
         if (c * c + d * d < PFLARE_TOL_ZERO) continue;

         // theta_k/theta_i
         div_real = (a * c + b * d) / (c * c + d * d);
         div_imag = (b * c - a * d) / (c * c + d * d);

         // |1 - theta_k/theta_i|
         div_mag = PetscSqrtReal((1 - div_real) * (1 - div_real) + div_imag * div_imag);

         // Pof is about to overflow, store the exponent and
         // reset pof back to one
         // We can hit this for very high order polynomials, where we have to
         // add more roots than 22 (ie pof > 1e308)
         if (PetscLog10Real(pof[k_loc]) + PetscLog10Real(div_mag) > 307) {
            overflow[k_loc] = overflow[k_loc] + (int)PetscLog10Real(pof[k_loc]);
            pof[k_loc] = 1;
         }

         // Product
         pof[k_loc] = pof[k_loc] * div_mag;
      }

      // If pof > 1e4, we add an extra root, plus one extra for every 1e14
      if (PetscLog10Real(pof[k_loc]) > 4 || overflow[k_loc] != 0) {

         // if real extra_pair_roots counts each distinct real root we're adding
         // if imaginary it only counts a pair as one
         extra_pair_roots[k_loc] = (int)PetscCeilReal((PetscLog10Real(pof[k_loc]) + overflow[k_loc] - (PetscReal)4.0) / (PetscReal)14.0);
         total_extra = total_extra + extra_pair_roots[k_loc];

         // If imaginary, the pof is the same for the conjugate, let's just set it to -1
         if (b > 0) {
            // We know the positive imaginary value is first, so the conjugate follows it
            pof[k_loc + 1] = -1;
            // We need the conjugates as well
            total_extra = total_extra + extra_pair_roots[k_loc];
         }
      }
   }

   // Allocate output arrays (original roots + extra roots)
   *roots_output_size = n_roots + total_extra;
   PetscCall(PetscCalloc1(n_roots + total_extra, real_roots_output));
   PetscCall(PetscCalloc1(n_roots + total_extra, imag_roots_output));

   // Copy in original roots
   for (i_loc = 0; i_loc < n_roots; i_loc++) {
      (*real_roots_output)[i_loc] = real_roots[i_loc];
      (*imag_roots_output)[i_loc] = imag_roots[i_loc];
   }

   // Add the extra copies of roots, ensuring conjugate pairs we add
   // are next to each other
   counter = n_roots;
   for (i_loc = 0; i_loc < n_roots; i_loc++) {

      // For each extra root pair to add
      for (j_loc = 0; j_loc < extra_pair_roots[i_loc]; j_loc++) {

         (*real_roots_output)[counter] = real_roots[i_loc];
         (*imag_roots_output)[counter] = imag_roots[i_loc];
         // Add in the conjugate
         if (imag_roots[i_loc] > 0) {
            (*real_roots_output)[counter + 1] = real_roots[i_loc];
            (*imag_roots_output)[counter + 1] = -imag_roots[i_loc];
         }

         counter = counter + 1;
         if (imag_roots[i_loc] > 0) counter = counter + 1;
      }
   }

   PetscCall(PetscFree(pof));
   PetscCall(PetscFree(extra_pair_roots));
   PetscCall(PetscFree(overflow));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode calculate_gmres_polynomial_roots_newton(Mat matrix, int poly_order, PetscBool add_roots,
   PetscReal **coefficients, PetscInt *coefficients_rows, PetscInt *coefficients_cols)
{
   // Computes a fixed order gmres polynomial for the matrix passed in
   // and outputs the Harmonic Ritz values (ie the roots) which we can use to apply
   // a polynomial in the Newton basis. This should be stable at high order
   // The cost of computing the newton basis is many reductions in parallel.
   // We don't provide a way to compute the monomial polynomial coefficients from the roots
   // (although we could by rearranging the newton polynomial) as I don't think this would be
   // stable at high order anyway. The only reason we use the monomial coefficients
   // is to easily build an assembled (approximate) matrix inverse, and
   // f you want that with these roots you could build one directly (although it would not
   // be very sparse at high order, and a fixed sparsity version at higher order than either the
   // power basis or Arnoldi basis could compute is likely not any better)
   // coefficients (column-major, *coefficients_rows x 2) may be reallocated here
   // (with PetscFree/PetscMalloc) to hold the extra roots

   // ~~~~~~
   PetscInt global_rows, global_cols, local_rows, local_cols, vecs_needed;
   PetscInt subspace_size, i_loc, j_loc, iwork_size;
   PetscInt total_extra, counter, k_loc, m, numerical_order, rows, ld_h, n_added, ld_temp;
   PetscMPIInt comm_size, comm_rank;
   // BLAS/LAPACK integer arguments must be PetscBLASInt
   PetscBLASInt lwork, rank, info;
   PetscBLASInt np1_bl, nrhs_bl, lda_hnt_bl, ldb_bl, lda_hn_bl, one_bl;
   MPI_Comm MPI_COMM_MATRIX;
   // H_n is column-major (poly_order+2) x (poly_order+1), H_n_T (poly_order+1) x (poly_order+2)
   PetscReal *H_n, *H_n_T;
   PetscReal *e_d, *solution, *s;
   PetscBLASInt *iwork_allocated;
   PetscInt *indices;
   PetscReal *work, *real_roots_added, *imag_roots_added;
   PetscReal *perturbed_real, *perturbed_imag;
   // Not used, but geev needs somewhere to point
   PetscReal VL[1], VR[1];
   PetscReal beta, h_sub_squared;
   // Column-major numerical_order x 2
   PetscReal *coefficients_temp;
   // The square part of H_n, column by column, for the Fortran norm2(H_n(1:m,1:m))
   PetscReal *H_n_square;
   Vec w_j;
   Vec *V_n;
   PetscBool use_harmonic_ritz = PETSC_TRUE;
   PetscReal rcond = PFLARE_TOL_RCOND, rel_tol, abs_tol, H_norm;
   // ~~~~~~

   PetscFunctionBegin;

   // This is how many columns we have in K_m
   subspace_size = poly_order + 1;

   // We might want to call the gmres poly creation on a sub communicator
   // so let's get the comm attached to the matrix and make sure to use that
   PetscCall(PetscObjectGetComm((PetscObject)matrix, &MPI_COMM_MATRIX));
   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));
   // Get the comm rank
   PetscCallMPI(MPI_Comm_rank(MPI_COMM_MATRIX, &comm_rank));
   // Get the matrix sizes
   PetscCall(MatGetSize(matrix, &global_rows, &global_cols));
   PetscCall(MatGetLocalSize(matrix, &local_rows, &local_cols));

   PetscCheck(subspace_size <= global_rows, MPI_COMM_MATRIX, PETSC_ERR_ARG_SIZ, "The input subspace size is greater than the matrix size");

   // If we're reusing, coefficients may still be sized for the extra roots
   // added in a previous setup, so make sure we start from p+1 roots
   // Otherwise the old roots would be included in the clustering below
   if (*coefficients_rows != poly_order + 1) {
      PetscCall(PetscFree(*coefficients));
      *coefficients_rows = poly_order + 1;
      *coefficients_cols = 2;
      PetscCall(PetscMalloc1((poly_order + 1) * 2, coefficients));
   }

   ld_h = poly_order + 2;
   PetscCall(PetscMalloc1((poly_order + 2) * (poly_order + 1), &H_n));
   PetscCall(PetscMalloc1((poly_order + 1) * (poly_order + 2), &H_n_T));
   PetscCall(PetscMalloc1(poly_order + 1, &e_d));
   PetscCall(PetscMalloc1(poly_order + 1, &solution));
   PetscCall(PetscMalloc1(poly_order + 1, &s));

   // ~~~~~~~~~~
   // Allocate space and create random numbers
   // The first vec has random numbers in it
   // ~~~~~~~~~~
   PetscCall(create_temp_space_box_muller(matrix, subspace_size, &V_n));

   // Create an extra vector for storage
   PetscCall(VecDuplicate(V_n[0], &w_j));

   // Do the Arnoldi and compute H_n
   // Use the same lucky tolerance as petsc
   PetscCall(arnoldi(matrix, poly_order, PFLARE_TOL_LUCKY, V_n, w_j, &beta, H_n, &m, NULL, NULL, NULL));

   // ~~~~~~~~~~~
   // Now the Ritz values are just the eigenvalues of the square part of H_n
   // We're actually going to use the Harmonic Ritz values which are the reciprocals
   // of the (ordinary) Ritz values of A^-1
   // see Embree - Polynomial preconditioned arnoldi with stability control
   // and Goossens - Ritz and Harmonic Ritz Values and the Convergence of FOM and GMRES
   // ~~~~~~~~~~~
   if (use_harmonic_ritz) {

      for (i_loc = 0; i_loc < poly_order + 1; i_loc++) e_d[i_loc] = 0;
      e_d[poly_order] = 1;

      // Now we have to be careful here
      // PETSc and Trilinos both just use an LU factorisation here to solve
      // H_d^-H e_d
      // But we can have the situation where the user has requested a high
      // polynomial order and H_d ends up not full rank
      // In that case the LU factorisation fails
      // Previously we just used to abort and tell the user to use lower order
      // But that is not super helpful
      // Instead we use a rank revealing factorisation that gives the
      // minimum norm solution
      // What we find is that when use this to compute eigenvalues we find e-vals
      // as we might expect up to the rank
      // but then we have some eigenvalues that are numerically zero
      // Given the way the outside code is structured, we can't lower the poly_order
      // in this routine and return
      // Instead we keep the "zero" eigenvalues our application of the newton polynomial
      // just skips them and hence we don't do any
      // extra work in the application phase than we would have done with lower order

      // ~~~~~~~~~~~
      // ~~~~~~~~~~~

      // Rank revealing min norm solution
      for (j_loc = 0; j_loc < poly_order + 2; j_loc++) {
         for (i_loc = 0; i_loc < poly_order + 1; i_loc++) {
            H_n_T[i_loc + j_loc * (poly_order + 1)] = H_n[j_loc + i_loc * ld_h];
         }
      }

      PetscCall(PetscMalloc1(1, &work));
      PetscCall(PetscMalloc1(1, &iwork_allocated));
      lwork = -1;

      // We have rcond = PFLARE_TOL_RCOND which is used to decide what singular values to drop
      // Matlab uses max(size(A))*eps(norm(A)) in their pinv
      // Kind-correct BLAS integer dimensions
      PetscCall(PetscBLASIntCast(poly_order + 1, &np1_bl));
      nrhs_bl = 1;
      PetscCall(PetscBLASIntCast(poly_order + 1, &lda_hnt_bl));
      PetscCall(PetscBLASIntCast(poly_order + 1, &ldb_bl));
      PetscCallBLAS("LAPACKgelsd", LAPACKgelsd_(&np1_bl, &np1_bl, &nrhs_bl, H_n_T, &lda_hnt_bl,
               e_d, &ldb_bl, s, &rcond, &rank,
               work, &lwork, iwork_allocated, &info));
      // Workspace query returns optimal sizes in work(1)/iwork(1); int() is exact
      // for all plausible sizes < 2^24 even when work is single precision
      PetscCall(PetscBLASIntCast((PetscInt)work[0], &lwork));
      iwork_size = iwork_allocated[0];
      PetscCall(PetscFree(work));
      PetscCall(PetscFree(iwork_allocated));
      PetscCall(PetscMalloc1(lwork, &work));
      PetscCall(PetscMalloc1(iwork_size, &iwork_allocated));
      PetscCallBLAS("LAPACKgelsd", LAPACKgelsd_(&np1_bl, &np1_bl, &nrhs_bl, H_n_T, &lda_hnt_bl,
               e_d, &ldb_bl, s, &rcond, &rank,
               work, &lwork, iwork_allocated, &info));
      PetscCall(PetscFree(work));
      PetscCall(PetscFree(iwork_allocated));

      // Copy in the solution
      for (i_loc = 0; i_loc < poly_order + 1; i_loc++) solution[i_loc] = e_d[i_loc];

      PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Harmonic Ritz solve failed");

      // Scale f by H(d+1,d)^2
      h_sub_squared = H_n[(poly_order + 1) + poly_order * ld_h] * H_n[(poly_order + 1) + poly_order * ld_h];
      for (i_loc = 0; i_loc < poly_order + 1; i_loc++) solution[i_loc] = solution[i_loc] * h_sub_squared;
      // Add to the last column of H_n
      for (i_loc = 0; i_loc < poly_order + 1; i_loc++) {
         H_n[i_loc + poly_order * ld_h] = H_n[i_loc + poly_order * ld_h] + solution[i_loc];
      }
   }

   // ~~~~~~~~~~~~~~
   // Now compute the eigenvalues of the square part of H_n
   // ie either compute the Ritz or Harmonic Ritz values
   // ~~~~~~~~~~~~~~

   PetscCall(PetscMalloc1(1, &work));
   lwork = -1;

   // Compute the eigenvalues
   // Kind-correct BLAS integer dimensions
   PetscCall(PetscBLASIntCast(poly_order + 1, &np1_bl));
   PetscCall(PetscBLASIntCast(ld_h, &lda_hn_bl));
   one_bl = 1;
   rows = *coefficients_rows;
   // DEFERRED (complex-prep): coefficients kept PetscReal by design; see docs/plan.
   // geev writes the strictly-real eigenvalue parts wr/wi directly into
   // coefficients(:,1)/(:,2) - these are LAPACK real arrays, so PetscReal is correct.
   PetscCallBLAS("LAPACKgeev", LAPACKgeev_("N", "N", &np1_bl, H_n, &lda_hn_bl,
            &(*coefficients)[0], &(*coefficients)[rows], VL, &one_bl, VR, &one_bl,
            work, &lwork, &info));
   // int() is exact for all plausible sizes < 2^24 even when work is single precision
   PetscCall(PetscBLASIntCast((PetscInt)work[0], &lwork));
   PetscCall(PetscFree(work));
   PetscCall(PetscMalloc1(lwork, &work));
   PetscCallBLAS("LAPACKgeev", LAPACKgeev_("N", "N", &np1_bl, H_n, &lda_hn_bl,
            &(*coefficients)[0], &(*coefficients)[rows], VL, &one_bl, VR, &one_bl,
            work, &lwork, &info));
   PetscCall(PetscFree(work));

   PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Eig decomposition failed");

   // ~~~~~~~~~~~~~~
   // Now we have to check the output eigenvalues
   // ~~~~~~~~~~~~~~

   // These are the tolerances that control the clustering
   // (norm2 of the square section H_n(1:m,1:m), taken in Fortran array element order)
   PetscCall(PetscMalloc1(PetscMax(m * m, 1), &H_n_square));
   for (j_loc = 0; j_loc < m; j_loc++) {
      for (i_loc = 0; i_loc < m; i_loc++) H_n_square[i_loc + j_loc * m] = H_n[i_loc + j_loc * ld_h];
   }
   H_norm = fortran_norm2(H_n_square, m * m);
   PetscCall(PetscFree(H_n_square));
   rel_tol = PetscSqrtReal(PFLARE_EPS);
   abs_tol = PFLARE_EPS * PetscMax(H_norm, beta);

   // In some cases with numerical rank deficiency, we can still
   // end up with non-zero (or negative) eigenvalues that
   // are trivially small - we set them explicitly to zero
   for (i_loc = 0; i_loc < poly_order + 1; i_loc++) {
      if ((*coefficients)[i_loc] * (*coefficients)[i_loc] + (*coefficients)[i_loc + rows] * (*coefficients)[i_loc + rows] <
          (abs_tol + rel_tol * H_norm) * (abs_tol + rel_tol * H_norm)) {
         (*coefficients)[i_loc] = 0.0;
         (*coefficients)[i_loc + rows] = 0.0;
      }
   }

   // ~~~~~~~~~~~~~~
   // Cluster close eigenvalues together to improve stability of the polynomial evaluation
   // For example when computing the e'vals of a constant diagonal matrix
   // the rank revealing factorisation above doesn't always report a rank of 1 given roundoff
   // Instead it returns multiple eigenvalues that are very close to each other,
   // and we want to cluster those together and treat them as one root
   // ~~~~~~~~~~~~~~

   // This places all exactly zero eigenvalues at the end of coefficients
   PetscCall(cluster_eigenvalues_stable(&(*coefficients)[0], &(*coefficients)[rows], rows, rel_tol, abs_tol));

   // ~~~~~~~~~~~~~~
   // Extract the non-zero eigenvalues for root adding and leja ordering
   // Zero eigenvalues will be appended at the end
   // ~~~~~~~~~~~~~~
   // Count the number of non-zero eigenvalues after clustering
   numerical_order = 0;
   for (i_loc = 0; i_loc < poly_order + 1; i_loc++) {
      if ((*coefficients)[i_loc] != 0.0 || (*coefficients)[i_loc + rows] != 0.0) {
         numerical_order = numerical_order + 1;
      }
   }

   // ~~~~~~~~~~~~~~
   // Add roots for stability (only on non-zero eigenvalues)
   // ~~~~~~~~~~~~~~
   if (add_roots && numerical_order > 0) {

      // Extract non-zero eigenvalues into a temporary array
      ld_temp = numerical_order;
      PetscCall(PetscMalloc1(numerical_order * 2, &coefficients_temp));
      counter = 0;
      for (i_loc = 0; i_loc < poly_order + 1; i_loc++) {
         if ((*coefficients)[i_loc] != 0.0 || (*coefficients)[i_loc + rows] != 0.0) {
            coefficients_temp[counter] = (*coefficients)[i_loc];
            coefficients_temp[counter + ld_temp] = (*coefficients)[i_loc + rows];
            counter = counter + 1;
         }
      }

      // Call compute_extra_roots only on the non-zero eigenvalues
      // This allocates real_roots_added/imag_roots_added with the original + extra roots
      PetscCall(compute_extra_roots(&coefficients_temp[0], &coefficients_temp[ld_temp], numerical_order,
               &real_roots_added, &imag_roots_added, &n_added));

      // total number of non-zero roots after adding extras
      total_extra = n_added - numerical_order;
      (void)total_extra;

      // Resize coefficients to hold non-zero roots (with extras) + zero roots at end
      PetscCall(PetscFree(*coefficients));
      *coefficients_rows = n_added + (poly_order + 1 - numerical_order);
      *coefficients_cols = 2;
      rows = *coefficients_rows;
      PetscCall(PetscCalloc1(rows * 2, coefficients));

      // Create perturbed copy for leja ordering
      PetscCall(PetscMalloc1(n_added, &perturbed_real));
      PetscCall(PetscMalloc1(n_added, &perturbed_imag));
      for (i_loc = 0; i_loc < n_added; i_loc++) {
         perturbed_real[i_loc] = real_roots_added[i_loc];
         perturbed_imag[i_loc] = imag_roots_added[i_loc];
      }

      // Perturb the extra roots so they have unique values for the leja sort
      counter = numerical_order;
      for (i_loc = 0; i_loc < numerical_order; i_loc++) {
         k_loc = 0;
         for (j_loc = counter; j_loc < n_added; j_loc++) {
            // Check if this extra root matches the original
            if (real_roots_added[j_loc] == coefficients_temp[i_loc] &&
                PetscAbsReal(imag_roots_added[j_loc]) == PetscAbsReal(coefficients_temp[i_loc + ld_temp])) {
               k_loc = k_loc + 1;
               perturbed_real[j_loc] = real_roots_added[j_loc] + (PetscReal)k_loc * (PetscReal)PFLARE_TOL_LEJA_PERTURB;
            }
         }
      }

      // Leja order only the non-zero eigenvalues (with extras)
      PetscCall(modified_leja(perturbed_real, perturbed_imag, n_added, &indices));

      // Reorder the (non-perturbed) roots using the leja ordering
      for (i_loc = 0; i_loc < n_added; i_loc++) {
         (*coefficients)[i_loc] = real_roots_added[indices[i_loc]];
         (*coefficients)[i_loc + rows] = imag_roots_added[indices[i_loc]];
      }

      // Zero eigenvalues are already zero at the end from the coefficients = 0d0 above

      // Cleanup
      PetscCall(PetscFree(coefficients_temp));
      PetscCall(PetscFree(real_roots_added));
      PetscCall(PetscFree(imag_roots_added));
      PetscCall(PetscFree(perturbed_real));
      PetscCall(PetscFree(perturbed_imag));
      PetscCall(PetscFree(indices));

   } else {

      // No root adding - just leja order the non-zero eigenvalues
      // and put zeros at the end
      if (numerical_order > 0) {

         // Extract non-zero eigenvalues
         ld_temp = numerical_order;
         PetscCall(PetscMalloc1(numerical_order * 2, &coefficients_temp));
         counter = 0;
         for (i_loc = 0; i_loc < poly_order + 1; i_loc++) {
            if ((*coefficients)[i_loc] != 0.0 || (*coefficients)[i_loc + rows] != 0.0) {
               coefficients_temp[counter] = (*coefficients)[i_loc];
               coefficients_temp[counter + ld_temp] = (*coefficients)[i_loc + rows];
               counter = counter + 1;
            }
         }

         // Leja order the non-zero eigenvalues
         PetscCall(modified_leja(&coefficients_temp[0], &coefficients_temp[ld_temp], numerical_order, &indices));

         // Reorder and put zeros at the end
         for (i_loc = 0; i_loc < rows * 2; i_loc++) (*coefficients)[i_loc] = 0.0;
         for (i_loc = 0; i_loc < numerical_order; i_loc++) {
            (*coefficients)[i_loc] = coefficients_temp[indices[i_loc]];
            (*coefficients)[i_loc + rows] = coefficients_temp[indices[i_loc] + ld_temp];
         }

         PetscCall(PetscFree(coefficients_temp));
         PetscCall(PetscFree(indices));
      }
   }

   // Cleanup
   vecs_needed = subspace_size + 1;
   PetscCall(VecDestroyVecs(vecs_needed, &V_n));
   PetscCall(VecDestroy(&w_j));

   PetscCall(PetscFree(H_n));
   PetscCall(PetscFree(H_n_T));
   PetscCall(PetscFree(e_d));
   PetscCall(PetscFree(solution));
   PetscCall(PetscFree(s));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode build_newton_fixed_sparsity_start(Mat matrix, int poly_order, int poly_sparsity_order, PetscReal coefficients[], PetscInt coefficients_rows,
   Mat *cmat, Mat *mat_sparsity_match, Mat *mat_product_save, int status_output[], PetscInt status_rows, PetscBool *output_first_complex)
{
   // Builds cmat with values correct up to the power poly_sparsity_order, along with
   // the products we need to compute the remaining fixed sparsity terms
   // This is shared by the cpu and kokkos versions of mat_mult_powers_share_sparsity_newton
   // and only involves petsc mat operations, so it runs on the device with kokkos mats

   // ~~~~~~~~~~
   PetscBool reuse_triggered;
   // Column-major (poly_sparsity_order+1) x 2
   PetscReal *coeffs_contig;
   PetscInt i_loc;
   // ~~~~~~~~~~

   PetscFunctionBegin;

   // coefficients is (number of roots, 2), so check against the number of roots
   PetscCheck(poly_sparsity_order < coefficients_rows - 1, PetscObjectComm((PetscObject)matrix), PETSC_ERR_ARG_OUTOFRANGE, "Requested sparsity is greater than or equal to the order");

   reuse_triggered = *cmat ? PETSC_TRUE : PETSC_FALSE;

   // ~~~~~~~~~~
   // Compute cmat for all powers up to poly_sparsity_order
   // We have to be more careful here than in the monomial case
   // In the mononomial case we just compute the matrix powers up to poly_sparsity_order
   // and add them times the coefficients to cmat
   // Here though we have to build the Newton basis polynomials
   // As a rule, the value input here into cmat is correct up to the power
   // of poly_sparsity_order, with all other terms being added in the fixed sparsity loops
   // that occur after this routine is called
   //
   // For complex conjugate roots, two terms are computed at a time, but if the sparsity order
   // falls in between those two roots, only part of the output cmat is included
   // Any remaining terms are output in either mat_sparsity_match or mat_product_save depending
   // on the case
   // mat_sparsity_match has either temp or prod in it depending on the case
   // mat_product_save only exists in some cases and stores prod from the previous term
   // mat_sparsity_match and mat_product_save are always output with the sparsity of sparsity order
   //
   // status_output is an array of length the number of roots
   // and has a 1 in each position if that term has been added to the output
   // It just helps us keep track of what has gone into cmat and what hasn't
   // It breaks up complex conjugate pairs into the first root (i) which has the same power as prod
   //   tmp = 2 * a * prod
   //   p = p + 1/(a^2 + b^2) * tmp
   // and the second root (i+1) which has the same power as prod * A:
   //   tmp = -A * prod
   //   p = p + 1/(a^2 + b^2) * tmp
   // ~~~~~~~~~~

   // Have to be careful to pass in a contiguous piece of memory here
   PetscCall(PetscMalloc1((poly_sparsity_order + 1) * 2, &coeffs_contig));
   for (i_loc = 0; i_loc < poly_sparsity_order + 1; i_loc++) {
      coeffs_contig[i_loc] = REAL_ROOT(i_loc + 1);
      coeffs_contig[i_loc + poly_sparsity_order + 1] = IMAG_ROOT(i_loc + 1);
   }

   *output_first_complex = PETSC_FALSE;
   if (poly_sparsity_order == 1) {

      // If we have a real first coefficient and a second complex
      // we can't call build_gmres_polynomial_newton_inverse_1st_1st as it is only correct
      // for valid coefficients up to 1st order (ie both real or both complex)
      if (IMAG_ROOT(1) == 0.0 && IMAG_ROOT(2) != 0.0) {

         PetscCall(build_gmres_polynomial_newton_inverse_full(matrix, poly_order, coefficients, coefficients_rows,
                  cmat, mat_sparsity_match, &poly_sparsity_order, output_first_complex,
                  status_output, status_rows, mat_product_save));

      // Valid 1st order polynomial, so this case is easy
      } else {

         // Duplicate & copy the matrix, but ensure there is a diagonal present
         PetscCall(mat_duplicate_copy_plus_diag(matrix, reuse_triggered, cmat));
         PetscCall(build_gmres_polynomial_newton_inverse_1st_1st(matrix, 1,
                  coeffs_contig, poly_sparsity_order + 1,
                  cmat, mat_sparsity_match,
                  status_output, status_rows));
      }
   } else {

      // If we're any higher, then we build cmat up to that order
      // But we have to be careful because the last root we want to explicitly
      // build up to here (ie the power of the matrix given by sparsity_order)
      // might be the first root of a complex conjugate pair
      PetscCall(build_gmres_polynomial_newton_inverse_full(matrix, poly_order,
               coeffs_contig, poly_sparsity_order + 1,
               cmat, mat_sparsity_match, &poly_sparsity_order, output_first_complex,
               status_output, status_rows, mat_product_save));
   }
   PetscCall(PetscFree(coeffs_contig));

   // We know we will never have non-zero locations outside of the highest constrained sparsity power
   PetscCall(MatSetOption(*cmat, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
   PetscCall(MatSetOption(*cmat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
   // We know we are only going to insert local vals
   // These options should turn off any reductions in the assembly
   PetscCall(MatSetOption(*cmat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode mat_mult_powers_share_sparsity_newton(Mat matrix, int poly_order, int poly_sparsity_order, PetscReal coefficients[], PetscInt coefficients_rows,
   Mat *reuse_mat, Mat **reuse_submatrices, Mat *cmat)
{
   // Wrapper around mat_mult_powers_share_sparsity_cpu and mat_mult_powers_share_sparsity_kokkos

   // ~~~~~~~~~~
#if defined(PETSC_HAVE_KOKKOS)
   Mat A_array, B_array, reuse_array;
   Mat sparsity_array, prod_save_array;
   int reuse_int_reuse_mat, num_terms;
   int prod_save_exists_int, output_first_complex_int;
   MatType mat_type;
   Mat temp_mat = NULL, temp_mat_reuse, temp_mat_compare;
   PetscReal normy;
   PetscBool reuse_triggered_cmat, reuse_triggered_reuse_mat;
   Mat reuse_mat_cpu = NULL;
   Mat *reuse_submatrices_cpu;
   Vec max_vec;
   PetscInt row_loc;
   Mat mat_sparsity_match = NULL, mat_product_save = NULL;
   int *status_output;
   PetscBool output_first_complex, is_kokkos;
#endif
   // ~~~~~~~~~~

   PetscFunctionBegin;

   // ~~~~~~~~~~
   // Special case if we just want to return a gmres polynomial with the sparsity of the diagonal
   // This is like a damped Jacobi
   // ~~~~~~~~~~
   if (poly_sparsity_order == 0) {

      PetscCall(build_gmres_polynomial_inverse_0th_order_sparsity_newton(matrix, poly_order,
               coefficients, coefficients_rows, cmat));

      PetscFunctionReturn(PETSC_SUCCESS);
   }

#if defined(PETSC_HAVE_KOKKOS)

   PetscCall(MatGetType(matrix, &mat_type));
   PetscCall(PetscStrcmpAny(mat_type, &is_kokkos, MATMPIAIJKOKKOS, MATSEQAIJKOKKOS, MATAIJKOKKOS, ""));
   if (is_kokkos) {

      PetscCall(PetscMalloc1(coefficients_rows * 2, &status_output));

      A_array = matrix;
      reuse_triggered_cmat = *cmat ? PETSC_TRUE : PETSC_FALSE;
      reuse_triggered_reuse_mat = *reuse_mat ? PETSC_TRUE : PETSC_FALSE;
      reuse_int_reuse_mat = 0;
      if (reuse_triggered_reuse_mat) {
         reuse_int_reuse_mat = 1;
      }
      reuse_array = *reuse_mat;
      num_terms = (int)coefficients_rows;

      // Build cmat with values correct up to the power poly_sparsity_order, along with
      // mat_sparsity_match, mat_product_save, status_output and output_first_complex
      // This is all petsc mat operations so runs on the device with kokkos mats
      PetscCall(build_newton_fixed_sparsity_start(matrix, poly_order, poly_sparsity_order, coefficients, coefficients_rows,
               cmat, &mat_sparsity_match, &mat_product_save, status_output, coefficients_rows, &output_first_complex));

      // The build has always created cmat by this point, even if we're not reusing
      B_array = *cmat;
      sparsity_array = mat_sparsity_match;
      prod_save_array = mat_product_save;
      prod_save_exists_int = 0;
      if (mat_product_save) prod_save_exists_int = 1;
      output_first_complex_int = 0;
      if (output_first_complex) output_first_complex_int = 1;

      // Adds in all the fixed sparsity terms on the device
      mat_mult_powers_share_sparsity_newton_kokkos(&A_array, &sparsity_array, &prod_save_array,
               prod_save_exists_int, num_terms, poly_sparsity_order,
               coefficients, status_output, output_first_complex_int,
               PFLARE_TOL_ZERO, reuse_int_reuse_mat, &reuse_array, &B_array);

      *reuse_mat = reuse_array;

      // Delete the temporaries from the build
      PetscCall(MatDestroy(&mat_sparsity_match));
      PetscCall(MatDestroy(&mat_product_save));
      PetscCall(PetscFree(status_output));

      // If debugging do a comparison between CPU and Kokkos results
      if (kokkos_debug()) {

         // If we're doing reuse and debug, then we have to always output the result
         // from the cpu version, as it will have coo preallocation structures set
         // They aren't copied over if you do a matcopy (or matconvert)
         // If we didn't do that the next time we come through this routine
         // and try to call the cpu version with reuse, it will segfault
         if (reuse_triggered_cmat) {
            temp_mat = *cmat;
            PetscCall(MatConvert(*cmat, MATSAME, MAT_INITIAL_MATRIX, &temp_mat_compare));
         } else {
            temp_mat_compare = *cmat;
         }

         // Debug check if the CPU and Kokkos versions are the same
         // We send in an empty reuse_mat_cpu here always, as we can't pass through
         // the same one Kokkos uses as it now only gets out the non-local rows we need
         // (ie reuse_mat and reuse_mat_cpu are no longer the same size)
         reuse_submatrices_cpu = NULL;
         PetscCall(mat_mult_powers_share_sparsity_newton_cpu(matrix, poly_order, poly_sparsity_order,
                  coefficients, coefficients_rows, &reuse_mat_cpu, &reuse_submatrices_cpu, &temp_mat));
         PetscCall(destroy_matrix_reuse(&reuse_mat_cpu, &reuse_submatrices_cpu));

         PetscCall(MatConvert(temp_mat, MATSAME, MAT_INITIAL_MATRIX, &temp_mat_reuse));

         PetscCall(MatAXPYWrapper(temp_mat_reuse, PFLARE_MINUS_ONE, temp_mat_compare));
         // Find the biggest entry in the difference
         PetscCall(MatCreateVecs(temp_mat_reuse, NULL, &max_vec));
         PetscCall(MatGetRowMaxAbs(temp_mat_reuse, max_vec, NULL));
         PetscCall(VecMax(max_vec, &row_loc, &normy));
         PetscCall(VecDestroy(&max_vec));

         // There is floating point compute in these inverses, so we have to be a
         // bit more tolerant to rounding differences
         // This recurrence is num_terms long, and per-term rounding differences between
         // optimized and unoptimized builds (compiler reordering, not fp-contraction -
         // this happens on the CPU and Kokkos sides independently) compound roughly
         // quadratically, so scale the tolerance with num_terms rather than using a
         // fixed bound that only holds at low order
         if (normy > PFLARE_TOL_MATFREE_NEWTON * ((double)num_terms * (double)num_terms) || PetscIsNanReal(normy)) {
            //call MatFilter(temp_mat_reuse, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
            //call MatView(temp_mat_reuse, PETSC_VIEWER_STDOUT_WORLD, ierr)
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Diff Kokkos and CPU mat_mult_powers_share_sparsity %g row %" PetscInt_FMT, (double)normy, row_loc);
         }
         PetscCall(MatDestroy(&temp_mat_reuse));
         if (!reuse_triggered_cmat) {
            PetscCall(MatDestroy(cmat));
         } else {
            PetscCall(MatDestroy(&temp_mat_compare));
         }
         *cmat = temp_mat;
      }

   } else {

      PetscCall(mat_mult_powers_share_sparsity_newton_cpu(matrix, poly_order, poly_sparsity_order,
               coefficients, coefficients_rows, reuse_mat, reuse_submatrices, cmat));
   }
#else
   PetscCall(mat_mult_powers_share_sparsity_newton_cpu(matrix, poly_order, poly_sparsity_order,
            coefficients, coefficients_rows, reuse_mat, reuse_submatrices, cmat));
#endif

   // ~~~~~~~~~~

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode mat_mult_powers_share_sparsity_newton_cpu(Mat matrix, int poly_order, int poly_sparsity_order, PetscReal coefficients[], PetscInt coefficients_rows,
   Mat *reuse_mat, Mat **reuse_submatrices, Mat *cmat)
{
   // Compute newton powers with the same sparsity

   // ~~~~~~~~~~
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one, row_index_into_submatrix;
   PetscInt global_col_start, global_col_end_plus_one, n, ncols, ncols_two, ifree, max_nnzs;
   PetscInt i_loc, j_loc, k_loc, row_size, rows_ao, cols_ao, rows_ad, cols_ad, shift = 0, diag_index;
   int match_counter;
   PetscInt term;
   PetscMPIInt comm_size;
   int *cols_index_one, *cols_index_two;
   PetscInt *col_indices_off_proc_array, *ad_indices, *cols;
   // Matrix entries filled by MatGetRow are PetscScalar
   PetscScalar *vals;
   IS col_indices, row_indices;
   Mat Ad, Ao, mat_sparsity_match = NULL, mat_product_save = NULL;
   const PetscInt *colmap = NULL;
   PetscBool deallocate_submatrices;
   // The symbolic of each row: the matching local column indices and the matching values
   // of matrix for each column of the row of mat_sparsity_match (NULL when there are none)
   int **symbolic_ones;
   PetscReal **symbolic_vals;
   PetscInt *symbolic_counts;
   MPI_Comm MPI_COMM_MATRIX;
   PetscReal *vals_power_temp, *vals_previous_power_temp, *temp;
   // The values handed to MatSetValues (the Fortran passed an array expression)
   PetscScalar *set_vals;
   const PetscInt *submatrices_ia, *submatrices_ja, *cols_two_ptr, *cols_ptr;
   PetscInt *cols_two_alloc = NULL;
   // Matrix entries filled by MatSeqAIJGetArray/MatGetRow are PetscScalar
   const PetscScalar *vals_two_ptr, *vals_ptr;
   const PetscScalar *submatrices_vals;
   PetscBool symmetric = PETSC_FALSE, inodecompressed = PETSC_FALSE, done;
   PetscInt one = 1, global_row;
   PetscBool output_first_complex, skip_add;
   PetscReal square_sum;
   // Column-major coefficients_rows x 2
   int *status_output;
   PetscInt status_rows;
   // ~~~~~~~~~~

   PetscFunctionBegin;

   deallocate_submatrices = PETSC_FALSE;

   status_rows = coefficients_rows;
   PetscCall(PetscMalloc1(status_rows * 2, &status_output));

   PetscCall(PetscObjectGetComm((PetscObject)matrix, &MPI_COMM_MATRIX));
   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));

   // Get the local sizes
   PetscCall(MatGetLocalSize(matrix, &local_rows, &local_cols));
   PetscCall(MatGetSize(matrix, &global_rows, &global_cols));
   // This returns the global index of the local portion of the matrix
   PetscCall(MatGetOwnershipRange(matrix, &global_row_start, &global_row_end_plus_one));
   PetscCall(MatGetOwnershipRangeColumn(matrix, &global_col_start, &global_col_end_plus_one));

   // Build cmat with values correct up to the power poly_sparsity_order, along with
   // mat_sparsity_match, mat_product_save, status_output and output_first_complex
   // which we need to compute the remaining fixed sparsity terms below
   PetscCall(build_newton_fixed_sparsity_start(matrix, poly_order, poly_sparsity_order, coefficients, coefficients_rows,
            cmat, &mat_sparsity_match, &mat_product_save, status_output, status_rows, &output_first_complex));

   // ~~~~~~~~~~~~
   // If we're in parallel we need to get the off-process rows of matrix that correspond
   // to the columns of mat_sparsity_match
   // We can therefore do the matmult for every constrained power locally with just that data
   // ~~~~~~~~~~~~
   // Have to double check comm_size /= 1 as we might be on a subcommunicator and we can't call
   // MatMPIAIJGetSeqAIJ specifically if that's the case
   cols_ao = 0;
   if (comm_size != 1) {

      // ~~~~
      // Get the cols
      // ~~~~
      PetscCall(MatMPIAIJGetSeqAIJ(mat_sparsity_match, &Ad, &Ao, &colmap));

      PetscCall(MatGetSize(Ad, &rows_ad, &cols_ad));
      // We know the col size of Ao is the size of colmap, the number of non-zero offprocessor columns
      PetscCall(MatGetSize(Ao, &rows_ao, &cols_ao));

      // These are the global indices of the columns we want
      PetscCall(PetscMalloc1(cols_ad + cols_ao, &col_indices_off_proc_array));
      PetscCall(PetscMalloc1(cols_ad, &ad_indices));
      // Local rows (as global indices)
      for (ifree = 0; ifree < cols_ad; ifree++) {
         ad_indices[ifree] = global_row_start + ifree;
      }

      // col_indices_off_proc_array is now sorted, which are the global indices of the columns we want
      PetscCall(merge_pre_sorted(ad_indices, cols_ad, colmap, cols_ao, col_indices_off_proc_array));
      PetscCall(PetscFree(ad_indices));

      // Create the sequential IS we want with the cols we want (written as global indices)
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF, cols_ad + cols_ao,
               col_indices_off_proc_array, PETSC_USE_POINTER, &col_indices));
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF, cols_ao,
               colmap, PETSC_USE_POINTER, &row_indices));

      // ~~~~~~~
      // Now we can pull out the chunk of matrix that we need
      // ~~~~~~~

      // We need off-processor rows to compute matrix powers
      // Setting this is necessary to avoid an allreduce when calling createsubmatrices
      // This will be reset to false after the call to createsubmatrices
      PetscCall(MatSetOption(matrix, MAT_SUBMAT_SINGLEIS, PETSC_TRUE));

      // Now this will be doing comms to get the non-local rows we want
      // But only including the columns of the local fixed sparsity, as we don't need all the
      // columns of the non-local entries unless we are doing a full matmatmult
      // This returns a sequential matrix
      if (*reuse_mat) {
         (*reuse_submatrices)[0] = *reuse_mat;
         PetscCall(MatCreateSubMatrices(matrix, one, &row_indices, &col_indices, MAT_REUSE_MATRIX, reuse_submatrices));
      } else {
         PetscCall(MatCreateSubMatrices(matrix, one, &row_indices, &col_indices, MAT_INITIAL_MATRIX, reuse_submatrices));
         *reuse_mat = (*reuse_submatrices)[0];
      }
      row_size = cols_ad + cols_ao;
      PetscCall(ISDestroy(&col_indices));
      PetscCall(ISDestroy(&row_indices));

   // Easy in serial as we have everything we neeed
   } else {

      Ad = mat_sparsity_match;
      cols_ad = local_cols;
      PetscCall(PetscMalloc1(1, reuse_submatrices));
      deallocate_submatrices = PETSC_TRUE;
      (*reuse_submatrices)[0] = matrix;
      row_size = local_rows;
      PetscCall(PetscMalloc1(local_rows, &col_indices_off_proc_array));
      for (ifree = 0; ifree < local_rows; ifree++) {
         col_indices_off_proc_array[ifree] = ifree;
      }
   }
   (void)row_size;
   (void)Ad;

   // ~~~~~~~~~
   // Now that we are here, reuse_submatrices(1) contains A^poly_sparsity_order with all of the rows
   // that correspond to the non-zero columns of matrix
   // ~~~~~~~~~

   // Have to get the max nnzs of the local and off-local rows we've just retrieved
   max_nnzs = 0;
   for (ifree = global_row_start; ifree < global_row_end_plus_one; ifree++) {
      PetscCall(MatGetRow(matrix, ifree, &ncols, NULL, NULL));
      if (ncols > max_nnzs) max_nnzs = ncols;
      PetscCall(MatRestoreRow(matrix, ifree, &ncols, NULL, NULL));
   }
   if (comm_size != 1) {
      for (ifree = 0; ifree < cols_ao; ifree++) {
         PetscCall(MatGetRow((*reuse_submatrices)[0], ifree, &ncols, NULL, NULL));
         if (ncols > max_nnzs) max_nnzs = ncols;
         PetscCall(MatRestoreRow((*reuse_submatrices)[0], ifree, &ncols, NULL, NULL));
      }
   }
   // and also the sparsity power
   for (ifree = global_row_start; ifree < global_row_end_plus_one; ifree++) {
      PetscCall(MatGetRow(mat_sparsity_match, ifree, &ncols, NULL, NULL));
      if (ncols > max_nnzs) max_nnzs = ncols;
      PetscCall(MatRestoreRow(mat_sparsity_match, ifree, &ncols, NULL, NULL));
   }

   // ~~~~~~~~
   // Get pointers to the sequential aij structure so we don't have to put critical regions
   // around the matgetrow
   // ~~~~~~~~
   PetscCall(MatGetRowIJ((*reuse_submatrices)[0], shift, symmetric, inodecompressed, &n, &submatrices_ia, &submatrices_ja, &done));
   PetscCheck(done, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Pointers not set in call to MatGetRowIJF");
   // Read-only access to the value array - the read variant doesn't bump
   // the matrix object state, so passing in e.g. a pc->pmat won't trigger
   // a spurious pc re-setup on every iteration
   PetscCall(MatSeqAIJGetArrayRead((*reuse_submatrices)[0], &submatrices_vals));

   // ~~~~~~~~~~

   PetscCall(PetscMalloc1(max_nnzs, &cols));
   PetscCall(PetscMalloc1(max_nnzs, &vals));
   PetscCall(PetscMalloc1(max_nnzs, &vals_power_temp));
   PetscCall(PetscMalloc1(max_nnzs, &vals_previous_power_temp));
   PetscCall(PetscMalloc1(max_nnzs, &temp));
   PetscCall(PetscMalloc1(max_nnzs, &cols_index_one));
   PetscCall(PetscMalloc1(max_nnzs, &cols_index_two));
   PetscCall(PetscMalloc1(max_nnzs, &set_vals));

   // ~~~~~~~~~~~~
   // From here we now have cmat with the correct values up to the power poly_sparsity_order
   // and hence we want to add in the sparsity constrained powers
   // ~~~~~~~~~~~~

   // Now go through and compute the sum of the matrix powers
   // We're doing row-wise matmatmults here assuming the fixed sparsity
   // We exploit the fact that the subsequent matrix powers can be done
   // one row at a time, so we only have to retrieve the needed vals from mat_sparsity_match once
   for (i_loc = 0; i_loc < local_rows; i_loc++) {

      global_row = global_row_start + i_loc;

      // Get the row of mat_sparsity_match
      PetscCall(MatGetRow(mat_sparsity_match, global_row, &ncols_two, &cols_ptr, &vals_ptr));
      // Copying here because mat_sparsity_match and matrix are often the same matrix
      // and hence we can only have one active matgetrow
      ncols = ncols_two;
      for (j_loc = 0; j_loc < ncols; j_loc++) {
         cols[j_loc] = cols_ptr[j_loc];
         vals[j_loc] = vals_ptr[j_loc];
      }
      PetscCall(MatRestoreRow(mat_sparsity_match, global_row, &ncols_two, &cols_ptr, &vals_ptr));
      diag_index = -1;
      // Find the diagonal index in this row
      for (j_loc = 0; j_loc < ncols; j_loc++) {
         if (cols[j_loc] == global_row) {
            diag_index = j_loc;
            break;
         }
      }

      // This is just a symbolic for the set of rows given in cols
      // Let's just do all the column matching and extraction of the values once

      // Allocate some space to store the matching indices
      PetscCall(PetscMalloc1(ncols, &symbolic_ones));
      PetscCall(PetscMalloc1(ncols, &symbolic_vals));
      PetscCall(PetscMalloc1(ncols, &symbolic_counts));
      row_index_into_submatrix = 0;

      // This is a row-wise product
      for (j_loc = 0; j_loc < ncols; j_loc++) {

         // If we're trying to access a local row in matrix
         if (cols[j_loc] >= global_row_start && cols[j_loc] < global_row_end_plus_one) {

            PetscCall(MatGetRow(matrix, cols[j_loc], &ncols_two, &cols_two_ptr, &vals_two_ptr));

         // If we're trying to access a non-local row in matrix
         } else {

            // this is local row index we want into reuse_submatrices(1) (as row_indices used to extract are just colmap)
            // We know cols is sorted, so every non-local index will be greater than the last one
            // (it's just that cols could have some local ones between different non-local)
            // colmap is also sorted and we know every single non-local entry in cols(j_loc) is in colmap
            while (row_index_into_submatrix < cols_ao && colmap[row_index_into_submatrix] < cols[j_loc]) {
               row_index_into_submatrix = row_index_into_submatrix + 1;
            }

            // This is the number of columns
            ncols_two = submatrices_ia[row_index_into_submatrix + 1] - submatrices_ia[row_index_into_submatrix];
            PetscCall(PetscMalloc1(ncols_two, &cols_two_alloc));
            // This is the local column indices in reuse_submatrices(1)
            // Because col_indices_off_proc_array (and hence the column indices in reuse_submatrices(1) is sorted,
            // then cols_two_ptr contains the sorted global column indices
            for (k_loc = 0; k_loc < ncols_two; k_loc++) {
               cols_two_alloc[k_loc] = col_indices_off_proc_array[submatrices_ja[submatrices_ia[row_index_into_submatrix] + k_loc]];
            }
            cols_two_ptr = cols_two_alloc;

            // This is the values
            vals_two_ptr = &submatrices_vals[submatrices_ia[row_index_into_submatrix]];
         }

         // Search for the matching column
         // We're intersecting the global column indices of mat_sparsity_match (cols) and matrix (cols_two_ptr)
         PetscCall(intersect_pre_sorted_indices_only(cols, ncols, cols_two_ptr, ncols_two, cols_index_one, cols_index_two, &match_counter));

         // Don't need to do anything if we have no matches
         if (match_counter == 0) {
            // Store that we can skip this entry
            symbolic_ones[j_loc] = NULL;
            symbolic_vals[j_loc] = NULL;
            symbolic_counts[j_loc] = 0;
         } else {

            // These are the matching local column indices for this row of mat_sparsity_match
            PetscCall(PetscMalloc1(match_counter, &symbolic_ones[j_loc]));
            // These are the matching values of matrix
            PetscCall(PetscMalloc1(match_counter, &symbolic_vals[j_loc]));
            for (k_loc = 0; k_loc < match_counter; k_loc++) {
               symbolic_ones[j_loc][k_loc] = cols_index_one[k_loc];
               symbolic_vals[j_loc][k_loc] = vals_two_ptr[cols_index_two[k_loc]];
            }
            symbolic_counts[j_loc] = match_counter;
         }

         // Restore local row of matrix
         if (cols[j_loc] >= global_row_start && cols[j_loc] < global_row_end_plus_one) {
            PetscCall(MatRestoreRow(matrix, cols[j_loc], &ncols_two, &cols_two_ptr, &vals_two_ptr));
         } else {
            PetscCall(PetscFree(cols_two_alloc));
         }
      }

      // Start with the values of mat_sparsity_match in it
      for (j_loc = 0; j_loc < ncols; j_loc++) vals_previous_power_temp[j_loc] = vals[j_loc];

      // Loop over any matrix powers
      // vals_power_temp stores the prod for this row, and we update this as we go through
      // the term loop
      term = poly_sparsity_order + 1;
      skip_add = PETSC_FALSE;
      // If the fixed sparsity root is the second of a complex pair, we start one term earlier
      // so that we can compute the correct part of the fixed sparsity product, we just make sure not to add
      // anything to cmat as it is already correct up to the fixed sparsity order
      if (IMAG_ROOT(term) != 0.0 && !output_first_complex) {
         term = term - 1;
         skip_add = PETSC_TRUE;
      }

      // This loop skips the last coefficient
      while (term <= coefficients_rows - 1) {

         // If real
         if (IMAG_ROOT(term) == 0.0) {

            // Skips eigenvalues that are numerically zero - see
            // the comment in calculate_gmres_polynomial_roots_newton
            if (PetscAbsReal(REAL_ROOT(term)) < PFLARE_TOL_ZERO) {
               term = term + 1;
               continue;
            }

            // ~~~~~~~~~~~
            // Now can add the value to our matrix
            // ~~~~~~~~~~~
            if (ncols != 0 && STATUS(term, 1) != 1) {
               for (j_loc = 0; j_loc < ncols; j_loc++) set_vals[j_loc] = PFLARE_ONE / REAL_ROOT(term) * vals_previous_power_temp[j_loc];
               PetscCall(MatSetValues(*cmat, one, &global_row, ncols, cols, set_vals, ADD_VALUES));
            }

            // Initialize with previous product before the A*prod subtraction
            for (j_loc = 0; j_loc < ncols; j_loc++) vals_power_temp[j_loc] = vals_previous_power_temp[j_loc];

            // This is the 1/theta_i * A * prod but where A * prod has fixed sparsity
            for (j_loc = 0; j_loc < ncols; j_loc++) {

               // If we have no matching columns cycle this row
               if (!symbolic_ones[j_loc]) continue;

               // symbolic_vals(j_loc)%ptr has the matching values of A in it
               for (k_loc = 0; k_loc < symbolic_counts[j_loc]; k_loc++) {
                  vals_power_temp[symbolic_ones[j_loc][k_loc]] = vals_power_temp[symbolic_ones[j_loc][k_loc]] -
                           PFLARE_ONE / REAL_ROOT(term) *
                           symbolic_vals[j_loc][k_loc] * vals_previous_power_temp[j_loc];
               }
            }

            term = term + 1;

         // If complex
         } else {

            // Skips eigenvalues that are numerically zero - see
            // the comment in calculate_gmres_polynomial_roots_newton
            if (REAL_ROOT(term) * REAL_ROOT(term) + IMAG_ROOT(term) * IMAG_ROOT(term) < PFLARE_TOL_ZERO) {
               term = term + 2;
               continue;
            }

            square_sum = PFLARE_ONE / (REAL_ROOT(term) * REAL_ROOT(term) + IMAG_ROOT(term) * IMAG_ROOT(term));

            // If our fixed sparsity order falls on the first of a complex conjugate pair
            if (!skip_add) {

               // We skip the 2 * a * prod from the first root of a complex pair if that has already
               // been included in the cmat from build_gmres_polynomial_newton_inverse_full
               if (STATUS(term, 2) != 1) {
                  for (j_loc = 0; j_loc < ncols; j_loc++) temp[j_loc] = 2 * REAL_ROOT(term) * vals_previous_power_temp[j_loc];
               } else {
                  for (j_loc = 0; j_loc < ncols; j_loc++) temp[j_loc] = 0.0;
               }

               // This is the -A * prod
               for (j_loc = 0; j_loc < ncols; j_loc++) {

                  // If we have no matching columns cycle this row
                  if (!symbolic_ones[j_loc]) continue;

                  // symbolic_vals(j_loc)%ptr has the matching values of A in it
                  for (k_loc = 0; k_loc < symbolic_counts[j_loc]; k_loc++) {
                     temp[symbolic_ones[j_loc][k_loc]] = temp[symbolic_ones[j_loc][k_loc]] -
                              symbolic_vals[j_loc][k_loc] * vals_previous_power_temp[j_loc];
                  }
               }

               // This is the p = p + 1/(a^2 + b^2) * temp
               if (ncols != 0) {
                  for (j_loc = 0; j_loc < ncols; j_loc++) set_vals[j_loc] = square_sum * temp[j_loc];
                  PetscCall(MatSetValues(*cmat, one, &global_row, ncols, cols, set_vals, ADD_VALUES));
               }

               // Here we need to go back in and ensure 2 *a * prod is in temp if we skipped it
               // above. We know it is already in cmat, but it has to be in temp when we
               // do the next product
               if (STATUS(term, 2) == 1) {
                  if (output_first_complex) {
                     for (j_loc = 0; j_loc < ncols; j_loc++) temp[j_loc] = temp[j_loc] + PFLARE_TWO * REAL_ROOT(term) * vals_previous_power_temp[j_loc];
                  }
               }

            // If our fixed sparsity order falls on the second of a complex conjugate pair
            } else {

               // In this case we have already included both 2*a*prod - A * prod into cmat
               // But we still have to compute the product for the next term
               // The problem here is that mat_sparsity_match has temp in it in this case, not
               // the old prod from whatever the previous loop is
               // In that case build_gmres_polynomial_newton_inverse_full also outputs
               // mat_product_save which is the old value of prod but with the sparsity of
               // mat_sparsity_match (with zeros if needed)

               // This case only occurs once for each row, so once we've hit this
               // we will always have our correct prod
               skip_add = PETSC_FALSE;
               // temp is output into mat_sparsity_match in this case
               for (j_loc = 0; j_loc < ncols; j_loc++) temp[j_loc] = vals_previous_power_temp[j_loc];

               // If sparsity order is 1, the previous product will have been the identity
               // and we don't output it into mat_product_save because that is a trivial case
               // we can do ourselves
               if (term == 1) {
                  for (j_loc = 0; j_loc < ncols; j_loc++) vals_previous_power_temp[j_loc] = 0.0;
                  if (diag_index != -1) {
                     vals_previous_power_temp[diag_index] = 1.0;
                  }

               // In the case the mat_product_save is not the identity, we need to pull it's value out
               } else {

                  PetscCall(MatGetRow(mat_product_save, global_row, &ncols_two, &cols_two_ptr, &vals_two_ptr));

                  // We have guaranteed in the build_gmres_polynomial_newton_inverse_full
                  // version that mat_product_save has fixed sparsity
                  for (j_loc = 0; j_loc < ncols_two; j_loc++) vals_previous_power_temp[j_loc] = vals_two_ptr[j_loc];

                  PetscCall(MatRestoreRow(mat_product_save, global_row, &ncols_two, &cols_two_ptr, &vals_two_ptr));
               }
            }

            // Now we compute the next product
            if (term <= coefficients_rows - 2) {

               for (j_loc = 0; j_loc < ncols; j_loc++) vals_power_temp[j_loc] = vals_previous_power_temp[j_loc];

               // This is prod = prod - 1/(a^2 + b^2) * A * temp
               for (j_loc = 0; j_loc < ncols; j_loc++) {

                  // If we have no matching columns cycle this row
                  if (!symbolic_ones[j_loc]) continue;

                  // symbolic_vals(j_loc)%ptr has the matching values of A in it
                  for (k_loc = 0; k_loc < symbolic_counts[j_loc]; k_loc++) {
                     vals_power_temp[symbolic_ones[j_loc][k_loc]] = vals_power_temp[symbolic_ones[j_loc][k_loc]] -
                              square_sum *
                              symbolic_vals[j_loc][k_loc] * temp[j_loc];
                  }
               }
            }

            term = term + 2;
         }

         // This should now have the value of prod in it
         for (j_loc = 0; j_loc < ncols; j_loc++) vals_previous_power_temp[j_loc] = vals_power_temp[j_loc];
      }

      // Final step if last root is real
      if (IMAG_ROOT(coefficients_rows) == 0.0) {
         if (ncols != 0 && PetscAbsReal(REAL_ROOT(coefficients_rows)) > PFLARE_TOL_ZERO) {
            for (j_loc = 0; j_loc < ncols; j_loc++) set_vals[j_loc] = PFLARE_ONE / REAL_ROOT(coefficients_rows) * vals_power_temp[j_loc];
            PetscCall(MatSetValues(*cmat, one, &global_row, ncols, cols, set_vals, ADD_VALUES));
         }
      }

      // ~~~~~~~~~~~~~~~

      for (j_loc = 0; j_loc < ncols; j_loc++) {
         if (symbolic_ones[j_loc]) {
            PetscCall(PetscFree(symbolic_ones[j_loc]));
            PetscCall(PetscFree(symbolic_vals[j_loc]));
         }
      }
      PetscCall(PetscFree(symbolic_vals));
      PetscCall(PetscFree(symbolic_ones));
      PetscCall(PetscFree(symbolic_counts));
   }

   PetscCall(MatRestoreRowIJ((*reuse_submatrices)[0], shift, symmetric, inodecompressed, &n, &submatrices_ia, &submatrices_ja, &done));
   PetscCall(MatSeqAIJRestoreArrayRead((*reuse_submatrices)[0], &submatrices_vals));

   // ~~~~~~~~~~~

   // Do the assembly, should need zero reductions in this given we've set the
   // flags above
   PetscCall(MatAssemblyBegin(*cmat, MAT_FINAL_ASSEMBLY));

   // Delete temporaries
   PetscCall(MatDestroy(&mat_sparsity_match));
   PetscCall(MatDestroy(&mat_product_save));
   if (deallocate_submatrices) {
      PetscCall(PetscFree(*reuse_submatrices));
   }

   PetscCall(PetscFree(col_indices_off_proc_array));
   PetscCall(PetscFree(cols));
   PetscCall(PetscFree(vals));
   PetscCall(PetscFree(vals_power_temp));
   PetscCall(PetscFree(vals_previous_power_temp));
   PetscCall(PetscFree(temp));
   PetscCall(PetscFree(cols_index_one));
   PetscCall(PetscFree(cols_index_two));
   PetscCall(PetscFree(set_vals));
   PetscCall(PetscFree(status_output));

   // Finish assembly
   PetscCall(MatAssemblyEnd(*cmat, MAT_FINAL_ASSEMBLY));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode build_gmres_polynomial_newton_inverse(Mat matrix, int poly_order, PetscReal coefficients[], PetscInt coefficients_rows,
   int poly_sparsity_order, PetscBool matrix_free, PetscBool diag_scale_polys, Mat *reuse_mat, Mat **reuse_submatrices, Mat *inv_matrix)
{
   // Builds a matrix which is an approximation to the inverse of a matrix using the
   // gmres polynomial in the newton basis
   // Can only be applied matrix-free

   // ~~~~~~
   PetscInt global_rows, global_cols, local_rows, local_cols;
   PetscMPIInt comm_size;
   MPI_Comm MPI_COMM_MATRIX;
   VecType vtype;
   mat_ctxtype *mat_ctx = NULL, *mat_ctx_scaled = NULL;
   PetscBool reuse_triggered;
   Mat diag_scaled_mat;
   Vec diag_inverse_vec = NULL;
   // ~~~~~~

   PetscFunctionBegin;

   // We might want to call the gmres poly creation on a sub communicator
   // so let's get the comm attached to the matrix and make sure to use that
   PetscCall(PetscObjectGetComm((PetscObject)matrix, &MPI_COMM_MATRIX));
   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));

   // Get the local sizes
   PetscCall(MatGetLocalSize(matrix, &local_rows, &local_cols));
   PetscCall(MatGetSize(matrix, &global_rows, &global_cols));

   // ~~~~~~~
   // Just build a matshell that applies our polynomial matrix-free
   // ~~~~~~~
   if (matrix_free) {

      // If not re-using
      if (!*inv_matrix) {

         // Have to dynamically allocate this
         PetscCall(create_mat_ctx(&mat_ctx));

         // We pass in the polynomial coefficients as the context
         PetscCall(MatCreateShell(MPI_COMM_MATRIX, local_rows, local_cols, global_rows, global_cols,
                  mat_ctx, inv_matrix));
         if (diag_scale_polys) {
            // The subroutine petsc_matvec_right_scale_poly_newton_mf applies
            // q(mat_ctx->mat_scaled) D^-1
            PetscCall(MatShellSetOperation(*inv_matrix,
                     MATOP_MULT, (PetscErrorCodeFn *)petsc_matvec_right_scale_poly_newton_mf));
            PetscCall(MatCreateVecs(matrix, &mat_ctx->mf_temp_vec[MF_VEC_RHS], &mat_ctx->mf_temp_vec[MF_VEC_DIAG]));
         } else {
            // The subroutine petsc_matvec_poly_newton_mf applies the polynomial inverse, q(mat_ctx->mat)
            PetscCall(MatShellSetOperation(*inv_matrix,
                     MATOP_MULT, (PetscErrorCodeFn *)petsc_matvec_poly_newton_mf));
         }
         // The subroutine petsc_matvec_poly_transpose_mf applies the transpose of
         // whichever of the two above we just set - it builds a transposed twin of
         // this matshell on demand, see ensure_transpose_mat
         PetscCall(MatShellSetOperation(*inv_matrix,
                  MATOP_MULT_TRANSPOSE, (PetscErrorCodeFn *)petsc_matvec_poly_transpose_mf));

         PetscCall(MatAssemblyBegin(*inv_matrix, MAT_FINAL_ASSEMBLY));
         PetscCall(MatAssemblyEnd(*inv_matrix, MAT_FINAL_ASSEMBLY));
         // Have to make sure to set the type of vectors the shell creates
         PetscCall(MatGetVecType(matrix, &vtype));
         PetscCall(MatShellSetVecType(*inv_matrix, vtype));

         // Create temporary vectors we use during application
         // Make sure to use matrix here to get the right type (as the shell doesn't know about gpus)
         PetscCall(MatCreateVecs(matrix, &mat_ctx->mf_temp_vec[MF_VEC_TEMP], NULL));
         PetscCall(MatCreateVecs(matrix, &mat_ctx->mf_temp_vec[MF_VEC_TEMP_THREE], &mat_ctx->mf_temp_vec[MF_VEC_TEMP_TWO]));

         // Need to build mat_ctx->mat_scaled, ie D^-1 A
         if (diag_scale_polys) {

            // Have to dynamically allocate this
            PetscCall(create_mat_ctx(&mat_ctx_scaled));

            // Create the matshell
            PetscCall(MatCreateShell(MPI_COMM_MATRIX, local_rows, local_cols, global_rows, global_cols,
                     mat_ctx_scaled, &mat_ctx->mat_scaled));
            // The subroutine petsc_matvec_da_poly_mf applies D^-1 A
            PetscCall(MatShellSetOperation(mat_ctx->mat_scaled,
                     MATOP_MULT, (PetscErrorCodeFn *)petsc_matvec_da_poly_mf));

            PetscCall(MatAssemblyBegin(mat_ctx->mat_scaled, MAT_FINAL_ASSEMBLY));
            PetscCall(MatAssemblyEnd(mat_ctx->mat_scaled, MAT_FINAL_ASSEMBLY));
            // Have to make sure to set the type of vectors the shell creates
            PetscCall(MatGetVecType(matrix, &vtype));
            PetscCall(MatShellSetVecType(mat_ctx->mat_scaled, vtype));
         }

      // Reusing
      } else {
         PetscCall(MatShellGetContext(*inv_matrix, &mat_ctx));
         if (diag_scale_polys) {
            PetscCall(MatShellGetContext(mat_ctx->mat_scaled, &mat_ctx_scaled));
         }
      }

      // Free old owned coefficients before reassigning.
      // Avoids a memory leak when the same matshell is reused across PCSetUp calls
      // (SAME_NONZERO_PATTERN) but fresh coefficients are being computed.
      // For PCAIR, own_coefficients is always .FALSE., so this is a no-op there.
      if (mat_ctx->own_coefficients && mat_ctx->coefficients) {
         PetscCall(PetscFree(mat_ctx->coefficients));
      }

      mat_ctx->real_roots = &coefficients[0];
      mat_ctx->imag_roots = &coefficients[coefficients_rows];
      mat_ctx->roots_size = coefficients_rows;
      // Now because the context reset deallocates the coefficient pointer
      // we want to make sure we don't leak memory, so mat_ctx->coefficients points at
      // both columns (the whole allocation) and then the free of mat_ctx->coefficients
      // should still delete all the memory
      mat_ctx->coefficients = &coefficients[0];
      mat_ctx->coefficients_size = 2 * coefficients_rows;
      // This is the matrix whose inverse we are applying (just copying the pointer here)
      mat_ctx->mat = matrix;

      if (diag_scale_polys) {
         mat_ctx_scaled->mat = matrix;
         mat_ctx_scaled->coefficients = &coefficients[0];
         mat_ctx_scaled->coefficients_size = 2 * coefficients_rows;
         // The coefficients may have been reallocated since the last setup
         mat_ctx_scaled->real_roots = &coefficients[0];
         mat_ctx_scaled->imag_roots = &coefficients[coefficients_rows];
         mat_ctx_scaled->roots_size = coefficients_rows;

         // Get the diagonal
         PetscCall(MatGetDiagonal(matrix, mat_ctx->mf_temp_vec[MF_VEC_DIAG]));
         mat_ctx_scaled->mf_temp_vec[MF_VEC_DIAG] = mat_ctx->mf_temp_vec[MF_VEC_DIAG];
      }

      // We're done
      PetscFunctionReturn(PETSC_SUCCESS);
   }

   // ~~~~~~~~~~~~
   // If we're here then we want an assembled approximate inverse
   // ~~~~~~~~~~~~
   reuse_triggered = *inv_matrix ? PETSC_TRUE : PETSC_FALSE;

   // If we want to diagonally scale (ie apply Jacobi preconditioned gmres polynomial)
   if (diag_scale_polys) {
      PetscCall(MatCreateVecs(matrix, NULL, &diag_inverse_vec));
      PetscCall(MatGetDiagonal(matrix, diag_inverse_vec));
      PetscCall(VecReciprocal(diag_inverse_vec));
      PetscCall(MatDuplicate(matrix, MAT_COPY_VALUES, &diag_scaled_mat));
      PetscCall(MatDiagonalScale(diag_scaled_mat, diag_inverse_vec, NULL));

   } else {
      diag_scaled_mat = matrix;
   }

   // If we're zeroth order poly this is trivial as it's just 1/theta_1 I
   if (poly_order == 0) {

      PetscCall(build_gmres_polynomial_newton_inverse_0th_order(diag_scaled_mat, poly_order, coefficients, coefficients_rows,
               inv_matrix));

      if (diag_scale_polys) {
         PetscCall(MatDestroy(&diag_scaled_mat));
         PetscCall(MatDiagonalScale(*inv_matrix, NULL, diag_inverse_vec));
         PetscCall(VecDestroy(&diag_inverse_vec));
      }

      // Then just return
      PetscFunctionReturn(PETSC_SUCCESS);

   // For poly_order 1 and poly_sparsity_order 1 this is easy
   } else if (poly_order == 1 && poly_sparsity_order == 1) {

      // Duplicate & copy the matrix, but ensure there is a diagonal present
      PetscCall(mat_duplicate_copy_plus_diag(diag_scaled_mat, reuse_triggered, inv_matrix));

      PetscCall(build_gmres_polynomial_newton_inverse_1st_1st(diag_scaled_mat,
               poly_order, coefficients, coefficients_rows, inv_matrix, NULL, NULL, 0));

      if (diag_scale_polys) {
         PetscCall(MatDestroy(&diag_scaled_mat));
         PetscCall(MatDiagonalScale(*inv_matrix, NULL, diag_inverse_vec));
         PetscCall(VecDestroy(&diag_inverse_vec));
      }

      // Then just return
      PetscFunctionReturn(PETSC_SUCCESS);
   }

   // If we're constraining sparsity we've built a custom matrix-powers that assumes fixed sparsity
   if (poly_sparsity_order < poly_order) {

      // This routine is a custom one that builds our matrix powers and assumes fixed sparsity
      // so that it doen't have to do much comms
      PetscCall(mat_mult_powers_share_sparsity_newton(diag_scaled_mat, poly_order, poly_sparsity_order, coefficients, coefficients_rows,
               reuse_mat, reuse_submatrices, inv_matrix));

      if (diag_scale_polys) {
         PetscCall(MatDestroy(&diag_scaled_mat));
         PetscCall(MatDiagonalScale(*inv_matrix, NULL, diag_inverse_vec));
         PetscCall(VecDestroy(&diag_inverse_vec));
      }

      // ! Then just return
      PetscFunctionReturn(PETSC_SUCCESS);
   }

   // ~~~~~~~~~~
   // We are only here if we don't constrain_sparsity
   // ~~~~~~~~~~
   PetscCall(build_gmres_polynomial_newton_inverse_full(diag_scaled_mat, poly_order, coefficients, coefficients_rows,
            inv_matrix, NULL, NULL, NULL, NULL, 0, NULL));

   if (diag_scale_polys) {
      PetscCall(MatDestroy(&diag_scaled_mat));
      PetscCall(MatDiagonalScale(*inv_matrix, NULL, diag_inverse_vec));
      PetscCall(VecDestroy(&diag_inverse_vec));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode build_gmres_polynomial_newton_inverse_0th_order(Mat matrix, int poly_order, PetscReal coefficients[], PetscInt coefficients_rows, Mat *inv_matrix)
{
   // Specific 0th order inverse

   // ~~~~~~
   PetscBool reuse_triggered;
   Vec diag_vec;
   // ~~~~~~

   PetscFunctionBegin;

   (void)coefficients_rows;

   PetscCheck(poly_order == 0, PetscObjectComm((PetscObject)matrix), PETSC_ERR_ARG_WRONG, "This is a 0th order inverse, but poly_order is not 0");

   // Let's create a matrix to represent the inverse diagonal
   reuse_triggered = *inv_matrix ? PETSC_TRUE : PETSC_FALSE;

   if (!reuse_triggered) {
      PetscCall(MatCreateVecs(matrix, NULL, &diag_vec));
   } else {
      PetscCall(MatDiagonalGetDiagonal(*inv_matrix, &diag_vec));
   }

   // Must be real as we only have one coefficient
   // Skips eigenvalues that are numerically zero, like the matrix-free apply
   if (PetscAbsReal(REAL_ROOT(1)) > PFLARE_TOL_ZERO) {
      PetscCall(VecSet(diag_vec, PFLARE_ONE / REAL_ROOT(1)));
   } else {
      PetscCall(VecSet(diag_vec, PFLARE_ZERO));
   }

   // We may be reusing with the same sparsity
   if (!reuse_triggered) {
      // The matrix takes ownership of diag_vec and increases ref counter
      PetscCall(MatCreateDiagonal(diag_vec, inv_matrix));
      PetscCall(VecDestroy(&diag_vec));
   } else {
      PetscCall(MatDiagonalRestoreDiagonal(*inv_matrix, &diag_vec));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode build_gmres_polynomial_inverse_0th_order_sparsity_newton(Mat matrix, int poly_order, PetscReal coefficients[], PetscInt coefficients_rows, Mat *inv_matrix)
{
   // Specific inverse with 0th order sparsity

   // ~~~~~~
   PetscInt i;
   PetscBool reuse_triggered;
   Vec inv_vec, diag_vec, product_vec, temp_vec_A, one_vec, temp_vec_two;
   // ~~~~~~

   PetscFunctionBegin;

   (void)poly_order;

   reuse_triggered = *inv_matrix ? PETSC_TRUE : PETSC_FALSE;

   // Our matrix has to be square
   PetscCall(MatCreateVecs(matrix, &product_vec, &diag_vec));
   PetscCall(MatGetDiagonal(matrix, diag_vec));

   if (!reuse_triggered) {
      PetscCall(VecDuplicate(diag_vec, &inv_vec));
   } else {
      PetscCall(MatDiagonalGetDiagonal(*inv_matrix, &inv_vec));
   }
   PetscCall(VecDuplicate(diag_vec, &temp_vec_A));
   PetscCall(VecDuplicate(diag_vec, &one_vec));
   PetscCall(VecDuplicate(diag_vec, &temp_vec_two));

   // Set to zero as we add to it
   PetscCall(VecSet(inv_vec, PFLARE_ZERO));
   // We start with an identity in product_vec
   PetscCall(VecSet(product_vec, PFLARE_ONE));
   PetscCall(VecSet(one_vec, PFLARE_ONE));

   i = 1;
   while (i <= coefficients_rows - 1) {

      // temp_vec_A is going to store things with the sparsity of A
      PetscCall(VecCopy(diag_vec, temp_vec_A));

      // If real this is easy
      if (IMAG_ROOT(i) == 0.0) {

         // Skips eigenvalues that are numerically zero - see
         // the comment in calculate_gmres_polynomial_roots_newton
         if (PetscAbsReal(REAL_ROOT(i)) < PFLARE_TOL_ZERO) {
            i = i + 1;
            continue;
         }

         PetscCall(VecAXPY(inv_vec, PFLARE_ONE / REAL_ROOT(i), product_vec));

         // temp_vec_A = A_ff/theta_k
         PetscCall(VecScale(temp_vec_A, PFLARE_MINUS_ONE / REAL_ROOT(i)));
         // temp_vec_A = I - A_ff/theta_k
         PetscCall(VecAXPY(temp_vec_A, PFLARE_ONE, one_vec));

         // product_vec = product_vec * temp_vec_A
         PetscCall(VecPointwiseMult(product_vec, product_vec, temp_vec_A));

         i = i + 1;

      // Complex
      } else {

         // Skips eigenvalues that are numerically zero
         if (REAL_ROOT(i) * REAL_ROOT(i) + IMAG_ROOT(i) * IMAG_ROOT(i) < PFLARE_TOL_ZERO) {
            i = i + 2;
            continue;
         }

         // Compute 2a I - A
         // temp_vec_A = -A
         PetscCall(VecScale(temp_vec_A, PFLARE_MINUS_ONE));
         // temp_vec_A = 2a I - A_ff
         PetscCall(VecAXPY(temp_vec_A, PFLARE_TWO * REAL_ROOT(i), one_vec));
         // temp_vec_A = (2a I - A_ff)/(a^2 + b^2)
         PetscCall(VecScale(temp_vec_A, PFLARE_ONE / (REAL_ROOT(i) * REAL_ROOT(i) + IMAG_ROOT(i) * IMAG_ROOT(i))));

         // temp_vec_two = temp_vec_A * product_vec
         PetscCall(VecPointwiseMult(temp_vec_two, temp_vec_A, product_vec));
         PetscCall(VecAXPY(inv_vec, PFLARE_ONE, temp_vec_two));

         if (i <= coefficients_rows - 2) {
            // temp_vec_two = A * temp_vec_two
            PetscCall(VecPointwiseMult(temp_vec_two, diag_vec, temp_vec_two));
            PetscCall(VecAXPY(product_vec, PFLARE_MINUS_ONE, temp_vec_two));
         }

         // Skip two evals
         i = i + 2;
      }
   }

   // Final step if last root is real
   if (IMAG_ROOT(coefficients_rows) == 0.0) {
      // Add in the final term multiplied by 1/theta_poly_order

      // Skips eigenvalues that are numerically zero
      if (PetscAbsReal(REAL_ROOT(coefficients_rows)) > PFLARE_TOL_ZERO) {
         PetscCall(VecAXPY(inv_vec, PFLARE_ONE / REAL_ROOT(coefficients_rows), product_vec));
      }
   }

   // We may be reusing with the same sparsity
   if (!reuse_triggered) {
      // The matrix takes ownership of inv_vec and increases ref counter
      PetscCall(MatCreateDiagonal(inv_vec, inv_matrix));
      PetscCall(VecDestroy(&inv_vec));
   } else {
      PetscCall(MatDiagonalRestoreDiagonal(*inv_matrix, &inv_vec));
   }

   PetscCall(VecDestroy(&diag_vec));
   PetscCall(VecDestroy(&product_vec));
   PetscCall(VecDestroy(&temp_vec_A));
   PetscCall(VecDestroy(&one_vec));
   PetscCall(VecDestroy(&temp_vec_two));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode build_gmres_polynomial_newton_inverse_1st_1st(Mat matrix, int poly_order, PetscReal coefficients[], PetscInt coefficients_rows,
   Mat *inv_matrix, Mat *mat_prod_or_temp, int status_output[], PetscInt status_rows)
{
   // Specific 1st order with 1st order sparsity
   // mat_prod_or_temp and status_output are optional (NULL when absent)

   // ~~~~~~
   PetscBool reuse_triggered, output_product;
   PetscReal square_sum, inv_theta_one;
   PetscInt i_loc;
   // ~~~~~~

   PetscFunctionBegin;

   (void)matrix;
   (void)poly_order;

   reuse_triggered = *inv_matrix ? PETSC_TRUE : PETSC_FALSE;
   (void)reuse_triggered;
   output_product = mat_prod_or_temp ? PETSC_TRUE : PETSC_FALSE;

   // Flags to prevent reductions when assembling (there are assembles in the shift)
   PetscCall(MatSetOption(*inv_matrix, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
   PetscCall(MatSetOption(*inv_matrix, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
   PetscCall(MatSetOption(*inv_matrix, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));

   if (output_product) {
      for (i_loc = 0; i_loc < status_rows * 2; i_loc++) status_output[i_loc] = 0;
   }

   // We only have two coefficients, so they are either both real or complex conjugates
   // If real
   if (IMAG_ROOT(1) == 0.0) {

      // Skips eigenvalues that are numerically zero, like the matrix-free apply
      // A skipped first root leaves the product as the identity
      if (PetscAbsReal(REAL_ROOT(1)) < PFLARE_TOL_ZERO) {
         inv_theta_one = 0;
      } else {
         inv_theta_one = PFLARE_ONE / REAL_ROOT(1);
      }

      // Have to be careful here, as we may be first order, but the second eigenvaule
      // might have been set to zero thanks to the rank reducing solve
      // So we just check if the second real part is zero and if it is
      // we just compute a 0th order inverse - annoyingly we can't call
      // build_gmres_polynomial_newton_inverse_0th_order as that builds a MATDIAGONAL
      // and in the tests there is a problem where we reuse the sparsity, in the first
      // solve we don't have a zero coefficient but in the second solve we do
      // So the mat type needs to remain consistent
      // This can't happen in the complex case
      // The matrix-free apply only applies the last root if it is > PFLARE_TOL_ZERO
      if (!(PetscAbsReal(REAL_ROOT(2)) > PFLARE_TOL_ZERO)) {

         // Set to zero
         PetscCall(MatScale(*inv_matrix, PFLARE_ZERO));

         // Tricky case here as we want to pass out the identity with the
         // sparsity of A
         if (output_product) {
            PetscCall(MatConvert(*inv_matrix, MATSAME, MAT_INITIAL_MATRIX, mat_prod_or_temp));
            PetscCall(MatShift(*mat_prod_or_temp, PFLARE_ONE));
            STATUS(1, 1) = 1;
            STATUS(2, 1) = 1;
         }

         // Then add in the 0th order inverse
         PetscCall(MatShift(*inv_matrix, inv_theta_one));

         // Then just return
         PetscFunctionReturn(PETSC_SUCCESS);
      }

      // Could just compute the equivalent mononomial here to save some flops
      // but the whole point to doing the Newton form is to avoid the
      // theta_1 * theta_2 that would result

      // result = -A_ff/theta_1
      PetscCall(MatScale(*inv_matrix, -inv_theta_one));
      // result = I -A_ff/theta_1
      PetscCall(MatShift(*inv_matrix, PFLARE_ONE));
      // If we're doing this as part of fixed sparsity multiply,
      // we need to return mat_prod_or_temp
      if (output_product) {
         PetscCall(MatConvert(*inv_matrix, MATSAME, MAT_INITIAL_MATRIX, mat_prod_or_temp));
      }

      // result = 1/theta_2 * (I -A_ff/theta_1)
      PetscCall(MatScale(*inv_matrix, PFLARE_ONE / (REAL_ROOT(2))));

      // result = 1/theta_1 + 1/theta_2 * (I -A_ff/theta_1)
      // Don't need an assemble as there is one called in this
      PetscCall(MatShift(*inv_matrix, inv_theta_one));

      if (output_product) {
         STATUS(1, 1) = 1;
         STATUS(2, 1) = 1;
      }

   // Complex conjugate roots, a +- ib
   } else {
      // 1/(a^2 + b^2)
      // Skips eigenvalues that are numerically zero, like the matrix-free apply
      square_sum = REAL_ROOT(1) * REAL_ROOT(1) + IMAG_ROOT(1) * IMAG_ROOT(1);
      if (square_sum < PFLARE_TOL_ZERO) {
         square_sum = 0;
      } else {
         square_sum = PFLARE_ONE / square_sum;
      }

      // Complex conjugate roots
      // result = -A_ff
      PetscCall(MatScale(*inv_matrix, PFLARE_MINUS_ONE));
      // result = 2a I - A_ff
      // Don't need an assemble as there is one called in this
      PetscCall(MatShift(*inv_matrix, PFLARE_TWO * REAL_ROOT(1)));
      // If we're doing this as part of fixed sparsity multiply,
      // we need to return mat_prod_or_temp
      if (output_product) {
         PetscCall(MatConvert(*inv_matrix, MATSAME, MAT_INITIAL_MATRIX, mat_prod_or_temp));
      }
      // result = 2a I - A_ff/(a^2 + b^2)
      PetscCall(MatScale(*inv_matrix, square_sum));

      if (output_product) {
         STATUS(1, 2) = 1;
         STATUS(2, 2) = 1;
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode build_gmres_polynomial_newton_inverse_full(Mat matrix, int poly_order, PetscReal coefficients[], PetscInt coefficients_rows,
   Mat *inv_matrix, Mat *mat_prod_or_temp, const int *poly_sparsity_order, PetscBool *output_first_complex,
   int status_output[], PetscInt status_rows, Mat *mat_product_save)
{
   // No constrained sparsity by default
   // If you pass in mat_prod_or_temp, poly_sparsity_order, output_first_complex, status_output and
   // mat_product_save
   // then it will build part of the terms, up to poly_sparsity_order, and return the product
   // in mat_prod_or_temp and mat_product_save that you need to compute the rest of the fixed sparsity terms
   // (the optional arguments are NULL when absent)

   // ~~~~~~
   PetscBool reuse_triggered, output_product, first_complex;
   PetscInt i, i_sparse, sparsity_order, i_loc;
   Mat mat_product = NULL, temp_mat_A = NULL, temp_mat_two = NULL, temp_mat_three = NULL, mat_product_k_plus_1 = NULL;
   PetscReal square_sum, a_coeff;
   // ~~~~~~

   PetscFunctionBegin;

   reuse_triggered = *inv_matrix ? PETSC_TRUE : PETSC_FALSE;
   output_product = mat_prod_or_temp ? PETSC_TRUE : PETSC_FALSE;

   if (!reuse_triggered) {
      // Duplicate & copy the matrix, but ensure there is a diagonal present
      PetscCall(mat_duplicate_copy_plus_diag(matrix, PETSC_FALSE, inv_matrix));
   }

   // Set to zero as we add in each product of terms
   PetscCall(MatScale(*inv_matrix, PFLARE_ZERO));

   // Don't set any off processor entries so no need for a reduction when assembling
   PetscCall(MatSetOption(*inv_matrix, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));

   // We start with an identity in mat_product
   PetscCall(generate_identity(matrix, &mat_product));
   if (output_product) {
      for (i_loc = 0; i_loc < status_rows * 2; i_loc++) status_output[i_loc] = 0;
   }
   sparsity_order = poly_order;
   if (poly_sparsity_order) {
      sparsity_order = *poly_sparsity_order;
   }

   // If we're going to output the product as part of a fixed sparsity multiply,
   // we may be asking to constrain the sparsity to a power in between order and order + 2
   // if there is a complex root at poly_sparsity_order
   // ie if we have roots (theta_1^r, theta_2^c, theta_3^c, theta_4^r)
   // where ^r means a purely real root and ^c means a complex root
   // want poly_sparsity_order = 1, we can't process all the way up to theta_3^c as that would
   // compute up to an A^2 term which is beyond our sparsity constraint

   // i_sparse tells us how many roots we are going to process
   // Normally this would just be size(coefficients, 1) and the loop below goes up
   // to size(coefficients, 1) - 1. The last real root gets its final term added outside the loop
   // and if the last root is complex then we only have to hit the first of the pair in the loop
   //
   // If we have fixed sparsity:
   //
   // if the fixed sparsity root is real then we want to set i_sparse to poly_sparsity_order+1
   // so we hit the roots up to poly_sparsity_order in the loop and then we take care of the
   // poly_sparsity_order + 1 root outside the loop
   //
   // if the fixed sparsity root is complex but poly_sparsity_order + 1 hits the second of the pair
   //    then we only need to set i_sparse to poly_sparsity_order + 1 so we only hit the first
   //    pair in the loop below
   //
   // if the fixed sparsity root is complex but poly_sparsity_order + 1 hits the first of the pair
   //    then we need to set i_sparse to poly_sparsity_order + 2
   //    otherwise we would never hit the first pair

   i_sparse = coefficients_rows;
   first_complex = PETSC_FALSE;

   if (output_product) {

      *output_first_complex = PETSC_FALSE;
      if (output_product) {
         i_sparse = sparsity_order + 1;

         // If the last root is real we don't have to do anything
         if (IMAG_ROOT(i_sparse) != 0.0) {

            // If the one before is real, then we know we're on the first
            if (IMAG_ROOT(i_sparse - 1) == 0.0) {
               *output_first_complex = PETSC_TRUE;
               // See discussion above
               i_sparse = i_sparse + 1;

            // If the one before is complex
            } else {

               // Check if the distance between the fixed sparsity root and the one before
               // If > zero then they are not complex conjugates and hence we are on the first of the pair
               // Don't divide by the root, as the real part may be zero (or negative)
               if (PetscAbsReal(REAL_ROOT(i_sparse) - REAL_ROOT(i_sparse - 1)) >
                     PFLARE_TOL_CONSISTENCY * PetscAbsReal(REAL_ROOT(i_sparse)) ||
                   PetscAbsReal(IMAG_ROOT(i_sparse) + IMAG_ROOT(i_sparse - 1)) >
                     PFLARE_TOL_CONSISTENCY * PetscAbsReal(IMAG_ROOT(i_sparse))) {
                  *output_first_complex = PETSC_TRUE;
                  i_sparse = i_sparse + 1;
               }
            }
         }
      }
      first_complex = *output_first_complex;
   }

   // ~~~~~~~~~~~~
   // Iterate over the i
   // This is basically the same as the MF application but we have to build the powers
   // ~~~~~~~~~~~~
   i = 1;
   // Loop through to one fewer than the number of roots
   // We're always building up the next product
   while (i <= i_sparse - 1) {

      // Duplicate & copy the matrix, but ensure there is a diagonal present
      // temp_mat_A is going to store things with the sparsity of A
      if (!temp_mat_A) {
         PetscCall(mat_duplicate_copy_plus_diag(matrix, PETSC_FALSE, &temp_mat_A));
      } else {
         // Can reuse the sparsity
         PetscCall(mat_duplicate_copy_plus_diag(matrix, PETSC_TRUE, &temp_mat_A));
      }

      // If real this is easy
      if (IMAG_ROOT(i) == 0.0) {

         // Skips eigenvalues that are numerically zero
         // We still compute the entries as as zero because we need the sparsity
         // to be correct for the next iteration
         if (PetscAbsReal(REAL_ROOT(i)) < PFLARE_TOL_ZERO) {
            square_sum = 0;
         } else {
            square_sum = PFLARE_ONE / REAL_ROOT(i);
         }

         // Then add the scaled version of each product
         if (i == 1) {
            // If i == 1 then we know mat_product is identity so we can do it directly
            PetscCall(MatShift(*inv_matrix, square_sum));
         } else {
            if (reuse_triggered) {
               // If doing reuse we know our nonzeros are a subset
               PetscCall(MatAXPY(*inv_matrix, square_sum, mat_product, SUBSET_NONZERO_PATTERN));
            } else {
               // Have to use the DIFFERENT_NONZERO_PATTERN here
               PetscCall(MatAXPYWrapper(*inv_matrix, square_sum, mat_product));
            }
         }
         if (output_product) STATUS(i, 1) = 1;

         // temp_mat_A = A_ff/theta_k
         PetscCall(MatScale(temp_mat_A, -square_sum));
         // temp_mat_A = I - A_ff/theta_k
         PetscCall(MatShift(temp_mat_A, PFLARE_ONE));

         // mat_product_k_plus_1 = mat_product * temp_mat_A
         if (i == 1) {
            // If i == 1 then we know mat_product is identity so we can just copy
            PetscCall(MatDestroy(&mat_product));
            PetscCall(MatConvert(temp_mat_A, MATSAME, MAT_INITIAL_MATRIX, &mat_product));
         } else {
            PetscCall(MatMatMult(temp_mat_A, mat_product,
                     MAT_INITIAL_MATRIX, PFLARE_MATMULT_FILL, &mat_product_k_plus_1));
            PetscCall(MatDestroy(&mat_product));
            mat_product = mat_product_k_plus_1;
         }

         // We copy out the last product if we're doing this as part of a fixed sparsity multiply
         if (output_product && i == i_sparse - 1) {
            PetscCall(MatDestroy(mat_prod_or_temp));
            PetscCall(MatConvert(mat_product, MATSAME, MAT_INITIAL_MATRIX, mat_prod_or_temp));
         }

         i = i + 1;

      // Complex
      } else {

         // Skips eigenvalues that are numerically zero
         if (REAL_ROOT(i) * REAL_ROOT(i) + IMAG_ROOT(i) * IMAG_ROOT(i) < PFLARE_TOL_ZERO) {
            square_sum = 0;
            a_coeff = 0;
         } else {
            square_sum = PFLARE_ONE / (REAL_ROOT(i) * REAL_ROOT(i) + IMAG_ROOT(i) * IMAG_ROOT(i));
            a_coeff = PFLARE_TWO * REAL_ROOT(i);
         }

         // If our fixed sparsity root is the first of a complex conjugate pair
         // We want to pass out mat_product and only add that to inv_matrix
         // This is equivalent to only part of tmp on Line 9 of Loe
         // The fixed sparsity loop will then finish the tmp with the term -A * prod/(a^2+b^2)
         // as this is the part that would increase the sparsity beyond poly_sparsity_order
         if (i == sparsity_order + 1 && first_complex) {

            // Copy mat_product into temp_mat_two
            PetscCall(MatConvert(mat_product, MATSAME, MAT_INITIAL_MATRIX, &temp_mat_two));

            // temp_mat_two = 2a * mat_product
            PetscCall(MatScale(temp_mat_two, a_coeff));
            if (output_product) STATUS(i, 2) = 1;

            // We copy out the last part of the product if we're doing this as part of a fixed sparsity multiply
            if (output_product && i > i_sparse - 2) {
               PetscCall(MatDestroy(mat_prod_or_temp));
               PetscCall(MatConvert(mat_product, MATSAME, MAT_INITIAL_MATRIX, mat_prod_or_temp));
            }

         // Just do the normal loop
         } else {

            // temp_mat_A = -A
            PetscCall(MatScale(temp_mat_A, PFLARE_MINUS_ONE));
            // temp_mat_A = 2a I - A_ff
            PetscCall(MatShift(temp_mat_A, a_coeff));
            if (output_product) {
               STATUS(i, 2) = 1;
               STATUS(i + 1, 2) = 1;
            }

            if (i == 1) {
               // If i == 1 then we know mat_product is identity so we can do it directly
               PetscCall(MatConvert(temp_mat_A, MATSAME, MAT_INITIAL_MATRIX, &temp_mat_two));
            } else {
               // temp_mat_two = temp_mat_A * mat_product
               PetscCall(MatMatMult(temp_mat_A, mat_product,
                        MAT_INITIAL_MATRIX, PFLARE_MATMULT_FILL, &temp_mat_two));
            }

            // We copy out the last part of the old product if we're doing this as part of a fixed sparsity multiply
            if (output_product && i > i_sparse - 2) {

               PetscCall(MatDestroy(mat_prod_or_temp));
               PetscCall(MatConvert(temp_mat_two, MATSAME, MAT_INITIAL_MATRIX, mat_prod_or_temp));
               // If i == 1 then we know mat_product is the identity and we don't bother
               // to write it out, we just have some custom code in the product given its trivial
               if (i != 1) {
                  // This ensures it has the matching sparsity
                  PetscCall(MatDestroy(mat_product_save));
                  PetscCall(MatConvert(*mat_prod_or_temp, MATSAME, MAT_INITIAL_MATRIX, mat_product_save));
                  // This zeros mat_product_save and then puts mat_product into the sparsity pattern
                  // of mat_prod_or_temp
                  PetscCall(MatCopy(mat_product, *mat_product_save, DIFFERENT_NONZERO_PATTERN));
               }
            }
         }

         // Then add the scaled version of each product
         if (reuse_triggered) {
            // If doing reuse we know our nonzeros are a subset
            PetscCall(MatAXPY(*inv_matrix, square_sum,
                     temp_mat_two, SUBSET_NONZERO_PATTERN));
         } else {
            // Have to use the DIFFERENT_NONZERO_PATTERN here
            PetscCall(MatAXPYWrapper(*inv_matrix, square_sum, temp_mat_two));
         }

         if (i <= i_sparse - 2) {

            // temp_mat_three = matrix * temp_mat_two
            PetscCall(MatMatMult(matrix, temp_mat_two,
                     MAT_INITIAL_MATRIX, PFLARE_MATMULT_FILL, &temp_mat_three));
            PetscCall(MatDestroy(&temp_mat_two));
            if (output_product) {
               STATUS(i, 2) = 1;
            }

            // Then add the scaled version of each product
            if (reuse_triggered) {
               // If doing reuse we know our nonzeros are a subset
               PetscCall(MatAXPY(mat_product, -square_sum,
                        temp_mat_three, SUBSET_NONZERO_PATTERN));
            } else {
               // Have to use the DIFFERENT_NONZERO_PATTERN here
               PetscCall(MatAXPYWrapper(mat_product, -square_sum, temp_mat_three));
            }

            // We copy out the last part of the product if we're doing this as part of a fixed sparsity multiply
            if (output_product && !first_complex) {
               PetscCall(MatDestroy(mat_prod_or_temp));
               PetscCall(MatConvert(mat_product, MATSAME, MAT_INITIAL_MATRIX, mat_prod_or_temp));
            }

            PetscCall(MatDestroy(&temp_mat_three));
         } else {
            PetscCall(MatDestroy(&temp_mat_two));
         }

         // Skip two evals
         i = i + 2;
      }
   }

   // Final step if last root is real
   if (!first_complex) {
      if (IMAG_ROOT(i_sparse) == 0.0) {
         // Add in the final term multiplied by 1/theta_poly_order

         // Skips eigenvalues that are numerically zero
         if (PetscAbsReal(REAL_ROOT(i_sparse)) > PFLARE_TOL_ZERO) {

            if (reuse_triggered) {
               // If doing reuse we know our nonzeros are a subset
               PetscCall(MatAXPY(*inv_matrix, PFLARE_ONE / REAL_ROOT(i_sparse), mat_product, SUBSET_NONZERO_PATTERN));
            } else {
               // Have to use the DIFFERENT_NONZERO_PATTERN here
               PetscCall(MatAXPYWrapper(*inv_matrix, PFLARE_ONE / REAL_ROOT(i_sparse), mat_product));
            }
            if (output_product) STATUS(i_sparse, 1) = 1;
         }
      }
   }

   PetscCall(MatDestroy(&temp_mat_A));
   PetscCall(MatDestroy(&mat_product));

   PetscFunctionReturn(PETSC_SUCCESS);
}
