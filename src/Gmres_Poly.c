/*
  C port of Gmres_Poly.F90 (module gmres_poly)

  The dense arrays (the Hessenberg matrix, the power basis, the TSQR R) are
  column-major and are passed with their dimensions. The coefficient arrays
  are passed with their sizes. The loops over rows and columns are 0-based,
  the loops over polynomial orders keep the Fortran (1-based) order and index
  coefficients[order - 1]

  The GMRES start vector draws its uniform random numbers from
  pflare_random_fill (Pflare_Random.h), and the Box-Muller transform is
  evaluated here in the same order and with the same constant as the Fortran,
  so the start vector is bitwise the same. The Fortran norm2 intrinsic is
  fortran_norm2, which evaluates it exactly as gfortran does
*/

#include <petscmat.h>
#include <petscblaslapack.h>
#include "pflare.h"
#include "pflareimpl.h"
#include "Sorting.h"
#include "Gmres_Polyk.h"
#include "Matshell_Data_Type.h"
#include "Gmres_Poly_Apply.h"
#include "TSQR.h"
#include "Gmres_Poly_Data_Type.h"
#include "PETSc_Helper.h"
#include "Pflare_Random.h"
#include "Gmres_Poly.h"

// Just define pi
// The same constant and kind as the Fortran parameter (3.141592653589793_PFLARE_REAL_KIND)
static const PetscReal pi = PetscRealConstant(3.141592653589793);

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscReal fortran_norm2(const PetscReal x[], PetscInt n)
{
   // The Fortran norm2 intrinsic of the n entries of x, evaluated as gfortran
   // evaluates it (a scaled sum of squares that cannot overflow), so the result
   // is bitwise the same as the Fortran's

   // ~~~~~~
   PetscReal scale, result, absx, val;
   PetscInt i;
   // ~~~~~~

   scale = 1.0;
   result = 0.0;
   for (i = 0; i < n; i++) {
      if (x[i] != 0.0) {
         absx = PetscAbsReal(x[i]);
         if (absx > scale) {
            val = scale / absx;
            result = (val * val) * result + 1.0;
            scale = absx;
         } else {
            val = absx / scale;
            result = val * val + result;
         }
      }
   }
   return PetscSqrtReal(result) * scale;
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode setup_gmres_poly_data(PetscInt global_rows, int inverse_type, int poly_order,
   int poly_sparsity_order, PetscBool subcomm, PetscInt proc_stride, gmres_poly_data *poly_data)
{
   // Setups the gmres poly data structure and does some size checking

   // ~~~~~~
   tsqr_buffers *buffers;
   PetscInt cols;
   // ~~~~~~

   PetscFunctionBegin;

   // The Fortran type held its buffers directly, here they are created on first use
   PetscCall(poly_data_buffers(poly_data, &buffers));

   // For gmres polynomial calculation we can move the reductions onto a subcomm
   if (inverse_type == PFLAREINV_POWER ||
       inverse_type == PFLAREINV_ARNOLDI ||
       inverse_type == PFLAREINV_NEWTON ||
       inverse_type == PFLAREINV_NEWTON_NO_EXTRA) {
      buffers->subcomm = subcomm;
   }
   buffers->proc_stride = (int)proc_stride;

   poly_data->inverse_type = inverse_type;
   // For matrices with size smaller than the subspace size (ie polynomial order + 1)
   // we'll have (close to) an exact solver and only need to go up to the matrix size
   poly_data->gmres_poly_order = poly_order;
   if (poly_data->gmres_poly_order + 1 > global_rows) {
      poly_data->gmres_poly_order = (int)(global_rows - 1);
   }

   // If we've changed the polynomial order lets make sure we aren't trying to take
   // sparsity bigger than it
   poly_data->gmres_poly_sparsity_order = poly_sparsity_order;
   if (poly_data->gmres_poly_sparsity_order > poly_data->gmres_poly_order) {
      poly_data->gmres_poly_sparsity_order = poly_data->gmres_poly_order;
   }

   // Now we know the order we can create our coefficient vector
   if (inverse_type == PFLAREINV_NEWTON || inverse_type == PFLAREINV_NEWTON_NO_EXTRA) {
      // Newton basis we have to store real and imag roots
      cols = 2;
   } else {
      cols = 1;
   }
   if (!poly_data->coefficients) {
      PetscCall(PetscMalloc1((PetscInt)(poly_data->gmres_poly_order + 1) * cols, &poly_data->coefficients));
      poly_data->coefficients_rows = poly_data->gmres_poly_order + 1;
      poly_data->coefficients_cols = cols;
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode compute_mf_gmres_poly_num_matvecs(int inverse_type, const PetscReal coefficients[], PetscInt coefficients_rows, PetscInt coefficients_cols, int *matvecs)
{
   // Computes how many matvecs we need to apply our matrix-free gmres polynomial

   // ~~~~~~
   PetscInt i_loc;
   // ~~~~~~

   PetscFunctionBegin;

   (void)coefficients_cols;

   // What order is our polynomial
   *matvecs = (int)coefficients_rows;

   // We may have zero roots with newton that are skipped
   if (inverse_type == PFLAREINV_NEWTON || inverse_type == PFLAREINV_NEWTON_NO_EXTRA) {
      for (i_loc = 0; i_loc < coefficients_rows; i_loc++) {
         // The zero check here has to match that in
         // petsc_matvec_gmres_newton_mf and petsc_matvec_gmres_newton_mf_residual
         // ie only exactly zero real roots are skipped, a complex pair is never zero
         if (coefficients[i_loc] == 0.0 && coefficients[i_loc + coefficients_rows] == 0.0) {
            *matvecs = *matvecs - 1;
         }
      }

   // For power or Arnoldi
   } else {
      // Horner can only skip the leading (highest order) zero coefficients,
      // normally can only happen in an Arnoldi that terminates early
      // This has to match petsc_horner
      *matvecs = (int)highest_nonzero_coeff(coefficients, coefficients_rows);
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode create_temp_space_box_muller(Mat matrix, PetscInt subspace_size, Vec **V_n)
{
   // Creates some temporary space and computes box muller random numbers
   // in the first column

   // ~~~~~~
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one, row_i, vecs_needed;
   PetscCount vec_size;
   PetscInt *indices;
   PetscMPIInt comm_size, comm_rank;
   // The random numbers are PetscReal, as pflare_random_fill fills a real array
   // (as the Fortran random_number did). Its use in VecSetValuesCOO below is an identity conversion in all
   // real builds
   // Column-major local_rows x 2
   PetscReal *random_data;
   Vec temp_vec;
   // ~~~~~~

   PetscFunctionBegin;

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
   PetscCall(MatGetOwnershipRange(matrix, &global_row_start, &global_row_end_plus_one));

   // ~~~~~~~~~~
   // Create random vector - this happens on the cpu
   // We could do most of the box-muller on the gpu, but currently the random numbers in petsc
   // are still generated on the host (and we don't have portable trig functions in petsc across the different
   // gpu vec types), so that would cause more copies to/from the gpu
   // If that changes this should be rewritten so this all happens on the gpu
   // ~~~~~~~~~~

   // If you ever change this to use PetscRandomCreate and VecSetRandom in petsc
   // make sure the seed is different than when random is used to evaluate
   // the auto truncation in AIR_MG_Setup

   // Gives random numbers between 0 <= u < 1
   // The seed is the same so subsequent runs get the same random, but
   // different on each process so we don't have the same random numbers on each processor
   // At least one entry, so pflare_random_fill is never handed a NULL on a rank without rows
   PetscCall(PetscMalloc1(PetscMax(2 * local_rows, 1), &random_data));
   // random_data(:, 1:2) is contiguous, so it is filled column by column as one
   // array of 2 * local_rows, in the same order random_number(random_data(:, 1:2)) does
   pflare_random_fill((int)comm_rank, 2 * local_rows, random_data);

   // We want our random rhs to be a normal distribution with zero mean as that preserves
   // white noise in the eigenspace (ie it is rotation invariant to unitary transforms)
   // Do a box-muller to take two numbers with uniform distribution and produce a number
   // that is normally distributed
   // random_number can return exactly 0 (with probability 2^-24 per draw in single precision,
   // so likely once there are more than 2^24 rows), so map 0 to 1 to keep the log finite
   for (row_i = 0; row_i < local_rows; row_i++) {
      if (random_data[row_i] == 0) random_data[row_i] = 1;
   }
   for (row_i = 0; row_i < local_rows; row_i++) {
      random_data[row_i] = PetscSqrtReal(-2 * PetscLogReal(random_data[row_i])) * PetscCosReal(2 * pi * random_data[row_i + local_rows]);
   }

   // ~~~~~~~~~~
   // Create a single template vec, set the random values into it, then use
   // VecDuplicateVecs to allocate all V_n vectors in one contiguous block.
   // Contiguous allocation lets VecMAXPY use BLAS-2 operations.
   // ~~~~~~~~~~
   PetscCall(MatCreateVecs(matrix, &temp_vec, NULL));

   PetscCall(PetscMalloc1(local_rows, &indices));
   // Set the random values into the template vector
   // Data will be copied to the gpu when needed
   for (row_i = 0; row_i < local_rows; row_i++) {
      indices[row_i] = global_row_start + row_i;
   }
   // PetscCount vs PetscInt??
   vec_size = local_rows;
   PetscCall(VecSetPreallocationCOO(temp_vec, vec_size, indices));
   PetscCall(PetscFree(indices));
   PetscCall(VecSetValuesCOO(temp_vec, random_data, INSERT_VALUES));

   PetscCall(PetscFree(random_data));
   vecs_needed = subspace_size + 1;

   // Allocate all subspace_size+1 vectors contiguously from the template
   PetscCall(VecDuplicateVecs(temp_vec, vecs_needed, V_n));
   // Copy random data from template into the first vector
   PetscCall(VecCopy(temp_vec, (*V_n)[0]));
   PetscCall(VecDestroy(&temp_vec));

   // ~~~~~~~~~~~~

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode ls_solve_arnoldi(PetscReal beta, PetscInt m, const PetscReal H_n[], PetscInt H_n_rows, PetscInt H_n_cols, PetscReal y[], PetscInt y_size)
{
   // Do the least-squares solve used in an arnoldi
   // H_n is column-major H_n_rows x H_n_cols and y has y_size entries

   // ~~~~~~
   // BLAS/LAPACK integer arguments must be PetscBLASInt
   PetscBLASInt errorcode, lwork;
   PetscBLASInt m_bl, n_bl, nrhs_bl, lda_bl, ldb_bl;
   PetscReal *H_n_copy, *g0, *work;
   PetscInt i_loc, j_loc;
   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscMalloc1(H_n_rows * H_n_cols, &H_n_copy));
   PetscCall(PetscMalloc1(m + 1, &g0));

   // This is the vector we will use as rhs in the least square solve
   for (i_loc = 0; i_loc < m + 1; i_loc++) g0[i_loc] = 0.0;
   g0[0] = beta;

   // Let's solve our least squares system
   PetscCall(PetscMalloc1(1, &work));
   lwork = -1;

   // Make a copy as we do a least-squares solve in place
   for (j_loc = 0; j_loc < m; j_loc++) {
      for (i_loc = 0; i_loc < m + 1; i_loc++) H_n_copy[i_loc + j_loc * H_n_rows] = H_n[i_loc + j_loc * H_n_rows];
   }

   // Kind-correct BLAS integer dimensions
   PetscCall(PetscBLASIntCast(m + 1, &m_bl));
   PetscCall(PetscBLASIntCast(m, &n_bl));
   nrhs_bl = 1;
   PetscCall(PetscBLASIntCast(H_n_rows, &lda_bl));
   PetscCall(PetscBLASIntCast(m + 1, &ldb_bl));

   lwork = -1;
   // Overwrite y
   errorcode = 0;
   PetscCallBLAS("LAPACKgels", LAPACKgels_("N", &m_bl, &n_bl, &nrhs_bl, &H_n_copy[0], &lda_bl, g0, &ldb_bl, work, &lwork, &errorcode));
   // Workspace query returns optimal lwork in work(1); int() is exact for all
   // plausible sizes < 2^24 even when work is single precision
   PetscCall(PetscBLASIntCast((PetscInt)work[0], &lwork));
   PetscCall(PetscFree(work));
   PetscCall(PetscMalloc1(lwork, &work));
   PetscCallBLAS("LAPACKgels", LAPACKgels_("N", &m_bl, &n_bl, &nrhs_bl, &H_n_copy[0], &lda_bl, g0, &ldb_bl, work, &lwork, &errorcode));
   PetscCall(PetscFree(work));

   PetscCheck(errorcode == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LS solve failed");

   // Set to zero necessary as sometimes we terminate our arnoldi early
   for (i_loc = 0; i_loc < y_size; i_loc++) y[i_loc] = 0;
   for (i_loc = 0; i_loc < m; i_loc++) y[i_loc] = g0[i_loc];

   PetscCall(PetscFree(H_n_copy));
   PetscCall(PetscFree(g0));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode arnoldi(Mat matrix, int poly_order, PetscReal lucky_tol, Vec V_n[], Vec w_j, PetscReal *beta, PetscReal H_n[], PetscInt *m,
   PetscReal C_n[], PetscReal y[], PetscReal *user_rel_tol)
{
   // Arnoldi to compute H_n and optionally C_n (although computing C_n
   // won't be stable at high order)
   // H_n is column-major (poly_order+2) x (poly_order+1), C_n (poly_order+2) x (poly_order+2)
   // and y has poly_order+1 entries
   // m is the number of Arnoldi steps taken (1-based, as the Fortran loop variable)

   // ~~~~~~
   PetscInt i_loc, j_loc, subspace_size, ld_h, ld_c;
   PetscReal *c_j, *g0, *neg_h;
   PetscBool compute_cn;
   PetscReal rel_tol;
   PetscInt m_petscint;
   // Kind-correct BLAS integer/real arguments for the dgemv call
   PetscBLASInt m_bl, n_bl, lda_bl, one_bl;
   PetscReal blas_one, blas_zero;
   // ~~~~~~

   PetscFunctionBegin;

   compute_cn = PETSC_FALSE;
   if (C_n) compute_cn = PETSC_TRUE;
   // Only compute H_n until we hit a given relative residual tolerance if it is input by the user
   rel_tol = -1;
   if (user_rel_tol) rel_tol = *user_rel_tol;

   // This is how many columns we have in K_m
   subspace_size = poly_order + 1;
   ld_h = poly_order + 2;
   ld_c = poly_order + 2;

   PetscCall(PetscMalloc1(poly_order + 2, &c_j));
   PetscCall(PetscMalloc1(poly_order + 2, &g0));
   PetscCall(PetscMalloc1(poly_order + 1, &neg_h));

   // This is the hessenberg matrix
   for (i_loc = 0; i_loc < (poly_order + 2) * (poly_order + 1); i_loc++) H_n[i_loc] = 0.0;
   // This stores part of the polynomial coefficients
   if (compute_cn) {
      for (i_loc = 0; i_loc < (poly_order + 2) * (poly_order + 2); i_loc++) C_n[i_loc] = 0.0;
   }

   // ~~~~~~~~~
   // Now time for a GMRES
   // ~~~~~~~~~

   // Compute the norm of the initial residual
   PetscCall(VecNorm(V_n[0], NORM_2, beta));
   // Now normalise to create the first orthogonal vector in V_n(1)
   PetscCall(VecNormalize(V_n[0], beta));

   // The first entry in C_n - As V_n = K_n C_n
   // the first orthogonalised vector in V_n is just r0/beta,
   // and we know r0 is the first vector in our krylov subspace
   if (compute_cn) C_n[0] = PFLARE_ONE / *beta;

   // Now loop through and do the iterations up to the max order of the polynomial
   // we want to build
   for (*m = 1; *m <= subspace_size; (*m)++) {

      m_petscint = *m;

      // Compute w_j = A v_j
      PetscCall(MatMult(matrix, V_n[*m - 1], w_j));

      // Now compute the updated relationship between K_n and V_n
      if (compute_cn) {
         for (i_loc = 0; i_loc < poly_order + 2; i_loc++) c_j[i_loc] = 0.0;
         // This is [0 c1..cn] for column m
         for (i_loc = 0; i_loc < *m; i_loc++) c_j[i_loc + 1] = C_n[i_loc + (*m - 1) * ld_c];
      }

      // Compute all hessenberg entries H_n(1:m, m) = <w_j, V_n(i)> in one reduction
      // neg_h is used as a contiguous temporary
      PetscCall(VecMDot(w_j, m_petscint, V_n, neg_h));
      for (i_loc = 0; i_loc < *m; i_loc++) H_n[i_loc + (*m - 1) * ld_h] = neg_h[i_loc];

      // w_j = w_j - sum_i H_n(i,m) * V_n(i)  (all at once)
      for (i_loc = 0; i_loc < *m; i_loc++) neg_h[i_loc] = -neg_h[i_loc];
      PetscCall(VecMAXPY(w_j, m_petscint, neg_h, V_n));

      // This is doing -C_n * h_n
      if (compute_cn) {
         for (i_loc = 1; i_loc <= *m; i_loc++) {
            for (j_loc = 0; j_loc < i_loc; j_loc++) {
               c_j[j_loc] = c_j[j_loc] - C_n[j_loc + (i_loc - 1) * ld_c] * H_n[(i_loc - 1) + (*m - 1) * ld_h];
            }
         }
      }

      // Now compute new hessenberg entry
      PetscCall(VecNorm(w_j, NORM_2, &H_n[*m + (*m - 1) * ld_h]));

      // GMRES lucky tolerance, we're fully converged
      if (H_n[*m + (*m - 1) * ld_h] < lucky_tol) {
         // Don't forget to update the ls solution if you exit early
         if (rel_tol > 0) PetscCall(ls_solve_arnoldi(*beta, *m, H_n, poly_order + 2, poly_order + 1, y, poly_order + 1));
         break;
      }

      // v_j+1 = w_j / h_j+1,j
      PetscCall(VecAXPBY(V_n[*m], PFLARE_ONE / H_n[*m + (*m - 1) * ld_h], PFLARE_ZERO, w_j));

      // Now we've taken out the H_n(m+1, m) factor from above
      if (compute_cn) {
         for (i_loc = 0; i_loc < *m + 1; i_loc++) C_n[i_loc + *m * ld_c] = c_j[i_loc] / H_n[*m + (*m - 1) * ld_h];
      }

      // ~~~~~~
      // Compute the residual if the user requested only solving to a specific
      // relative residual - we don't need it otherwise
      // The residual is just ||H_m y - e_1 beta||_2, so we have to solve that least squares problem to find
      // y and then just compute
      // ~~~~~~
      if (rel_tol > 0) {

         PetscCall(ls_solve_arnoldi(*beta, *m, H_n, poly_order + 2, poly_order + 1, y, poly_order + 1));

         // Compute H_n y
         PetscCall(PetscBLASIntCast(*m + 1, &m_bl));
         PetscCall(PetscBLASIntCast(*m, &n_bl));
         PetscCall(PetscBLASIntCast(ld_h, &lda_bl));
         one_bl = 1;
         blas_one = 1.0;
         blas_zero = 0.0;
         PetscCallBLAS("BLASgemv", BLASgemv_("N", &m_bl, &n_bl,
                  &blas_one, H_n, &lda_bl,
                  y, &one_bl,
                  &blas_zero, &g0[0], &one_bl));

         // Minus away e1 beta
         g0[0] = g0[0] - *beta;
         // This is the relative residual - H_n is (m+1) x m so the residual
         // has m+1 entries (the last is typically the largest)
         *user_rel_tol = fortran_norm2(g0, *m + 1) / *beta;
         //print *, m, "rel residual", user_rel_tol
         if (*user_rel_tol < rel_tol) break;
      }
   }

   // Fortran allocates loop variables one bigger if we get to the end of the loop
   if (*m == subspace_size + 1) *m = *m - 1;

   PetscCall(PetscFree(c_j));
   PetscCall(PetscFree(g0));
   PetscCall(PetscFree(neg_h));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode calculate_gmres_polynomial_coefficients_arnoldi(Mat matrix, int poly_order, PetscReal coefficients[], PetscReal *user_rel_tol)
{
   // Computes a fixed order gmres polynomial for the matrix passed in
   // and stores each of the polynomial coefficients in coefficients
   // This computes the coefficients using the Arnoldi basis
   // This won't be stable at high order, but is stable at higher order than
   // using the power basis. The cost of this is many reductions in parallel
   // There is a comms-avoiding one given in start_gmres_polynomial_coefficients_power
   // which is stable for low order
   // coefficients has poly_order+1 entries

   // ~~~~~~
   PetscInt global_rows, global_cols, vecs_needed;
   PetscInt subspace_size, m, i_loc;
   PetscReal *H_n, *C_n, *y;
   PetscReal beta;
   Vec w_j;
   Vec *V_n;
   PetscReal rel_tol;
   // Kind-correct BLAS integer/real arguments for the dgemv call
   PetscBLASInt m_bl, lda_bl, one_bl;
   PetscReal blas_one, blas_zero;
   // ~~~~~~

   PetscFunctionBegin;

   // This is how many columns we have in K_m
   subspace_size = poly_order + 1;
   // Get the matrix sizes
   PetscCall(MatGetSize(matrix, &global_rows, &global_cols));

   // ~~~~~~~~~~~~~
   // We're basically just writing a simple GMRES here with no restart, the reason we can't use the petsc one
   // is we need access to the unrotated hessenberg matrix, and the orthogonalised vectors
   // to compute the gmres polynomial
   // See Nachtigal et al 1992 for details on how we generate the gmres polynomial
   // coefficients using the Arnoldi basis
   // ~~~~~~~~~~~~~

   PetscCheck(subspace_size <= global_rows, PetscObjectComm((PetscObject)matrix), PETSC_ERR_ARG_SIZ, "The input subspace size is greater than the matrix size");

   PetscCall(PetscMalloc1((poly_order + 2) * (poly_order + 1), &H_n));
   PetscCall(PetscMalloc1((poly_order + 2) * (poly_order + 2), &C_n));
   PetscCall(PetscMalloc1(poly_order + 1, &y));

   // ~~~~~~~~~~
   // Allocate space and create random numbers
   // The first vec has random numbers in it
   // ~~~~~~~~~~
   PetscCall(create_temp_space_box_muller(matrix, subspace_size, &V_n));

   // Create an extra vector for storage
   PetscCall(VecDuplicate(V_n[0], &w_j));

   // Do the Arnoldi and compute H_n and C_n
   // We only compute H_n until we hit a relative residual of 1e-14 against the random rhs
   // or we hit the given poly_order
   rel_tol = PFLARE_TOL_ARNOLDI;
   if (user_rel_tol) rel_tol = *user_rel_tol;
   PetscCall(arnoldi(matrix, poly_order, PFLARE_TOL_LUCKY, V_n, w_j, &beta, H_n, &m, C_n, y, &rel_tol));
   if (user_rel_tol) *user_rel_tol = rel_tol;

   // ~~~~~~~~~~~~~
   // Compute the polynomial coefficients, this is C_n(1:m, 1:m) y
   // ~~~~~~~~~~~~~
   // Set to zero is necessary as m may be less than the subspace_size
   for (i_loc = 0; i_loc < poly_order + 1; i_loc++) coefficients[i_loc] = 0;
   PetscCall(PetscBLASIntCast(m, &m_bl));
   PetscCall(PetscBLASIntCast(poly_order + 2, &lda_bl));
   one_bl = 1;
   blas_one = 1.0;
   blas_zero = 0.0;
   PetscCallBLAS("BLASgemv", BLASgemv_("N", &m_bl, &m_bl,
            &blas_one, C_n, &lda_bl,
            y, &one_bl,
            &blas_zero, &coefficients[0], &one_bl));

   vecs_needed = subspace_size + 1;
   PetscCall(VecDestroyVecs(vecs_needed, &V_n));
   PetscCall(VecDestroy(&w_j));

   PetscCall(PetscFree(H_n));
   PetscCall(PetscFree(C_n));
   PetscCall(PetscFree(y));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode start_gmres_polynomial_coefficients_power(Mat matrix, int poly_order, tsqr_buffers *buffers, PetscReal coefficients[])
{
   // Computes a fixed order gmres polynomial for the matrix passed in
   // and stores each of the polynomial coefficients in coefficients
   // This computes the polynomial coefficients in the power basis using
   // the power basis
   // This allows us to do a 1-step method that is comms reducing
   // only requiring a single reduction in parallel
   // If in parallel this is non-blocking and you have to call
   // finish_gmres_polynomial_coefficients_power outside this routine to finish

   // ~~~~~~
   PetscInt global_rows, global_cols, local_rows, local_cols;
   PetscInt global_row_start, global_row_end_plus_one, row_i, vecs_needed;
   PetscInt *global_indices;
   PetscInt subspace_size, i_loc;
   PetscMPIInt comm_size, comm_rank;
   MPI_Comm MPI_COMM_MATRIX;
   // Column-major local_rows x (subspace_size+1)
   PetscReal *K_m_plus_1_data;
   Vec *V_n;
   // ~~~~~~

   PetscFunctionBegin;

   (void)coefficients;

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

   PetscCheck(subspace_size <= global_rows, MPI_COMM_MATRIX, PETSC_ERR_ARG_SIZ, "The input subspace size must be smaller than the matrix size");

   // ~~~~~~~~~
   // GMRES(m) will minimise the residual over the subspace
   // K_m = [ b, A b, A^2 b, ... A^(m-1) b]
   // A normal GMRES algorithm will use m dot products (ie m reductions) and m matvecs to
   // compute an orthgonal basis of this subspace directly with an Arnoldi
   // without needing to compute the power basis K_m directly.
   // Equivalently, we can compute the gmres polynomial coefficients, g, by
   // solving the least squares system
   // min || b - A K_m g ||
   // Rather than having to compute the QR factorisation of A K_m and
   // than also have to compute the norm of b, we note that
   // A K_m = K_m+1(:,2:end)
   // It takes m matvecs to compute K_m+1 (same number of matvecs as in using the Arnoldi
   // to compute an orthogonalised basis)
   // but by doing a QR on K_m+1, the norm of r_0 will be in the first row/col position of R, and
   // then we can use the other columns of R as they will be the QR of A K_m
   // That way we only have to do a single reduction in parallel
   // This is equivalent to computing the gmres polynomial coefficients for
   // a single step (s=1) communication avoiding GMRES
   // Using the power basis is a bad idea at high order due to stability issues, but
   // we are staying low order so should be fine  (e.g, < 10th order)
   // ~~~~~~~~~

   // ~~~~~~~~~~
   // Allocate space and create random numbers
   // The first vec has random numbers in it
   // ~~~~~~~~~~
   PetscCall(create_temp_space_box_muller(matrix, subspace_size, &V_n));

   // ~~~~~~~~~~~~

   // Now loop through and build the power basis K_m+1
   for (i_loc = 0; i_loc < subspace_size; i_loc++) {

      // Compute V_n(j+1) = A * V_n(j)
      // Could do this with a matrix-power kernel with less comms if desired, except
      // that we only ever need this once for very low order so that would prob result in more comms overall
      PetscCall(MatMult(matrix, V_n[i_loc], V_n[i_loc + 1]));
   }

   // Need to copy the individual vectors into a dense slab of memory
   // Used to just have a dense slab of memory and then
   // place the pointers into the vecs, but that
   // became cumbersome when supporting different vec types (eg cuda on a gpu)
   // At least one entry, as the Fortran had zero-size arrays with an address
   PetscCall(PetscMalloc1(PetscMax(local_rows * (subspace_size + 1), 1), &K_m_plus_1_data));
   PetscCall(PetscMalloc1(PetscMax(local_rows, 1), &global_indices));
   PetscCall(MatGetOwnershipRange(matrix, &global_row_start, &global_row_end_plus_one));
   for (row_i = 0; row_i < local_rows; row_i++) {
      global_indices[row_i] = global_row_start + row_i;
   }

   // Copy into K_m_plus_1_data and then destroy the vecs
   for (i_loc = 0; i_loc < subspace_size + 1; i_loc++) {
      // Have to use vecgetvalues interface for gpus
      // This does a copy of the vec from gpu to cpu if necessary
      PetscCall(VecGetValues(V_n[i_loc], local_rows, global_indices, &K_m_plus_1_data[i_loc * local_rows]));
   }
   vecs_needed = subspace_size + 1;
   PetscCall(VecDestroyVecs(vecs_needed, &V_n));

   // ~~~~~~~~~~~~~
   // Start the tall-skinny QR factorisation of the power basis - this all happens
   // on the cpu
   // ~~~~~~~~~~~~~
   PetscCall(start_tsqr(MPI_COMM_MATRIX, K_m_plus_1_data, local_rows, subspace_size + 1, buffers));

   // ~~~~~
   // ~~~~~
   // Now have to call finish_gmres_polynomial_coefficients_power outside this routine
   // to resolve the async comms in the tsqr
   // ~~~~~
   // ~~~~~

   PetscCall(PetscFree(K_m_plus_1_data));
   PetscCall(PetscFree(global_indices));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode finish_gmres_polynomial_coefficients_power(int poly_order, tsqr_buffers *buffers, PetscReal coefficients[])
{
   // Finishes the non-blocking all reduce from start_gmres_polynomial_coefficients_power in parallel
   // The polynomial coefficients will be output into coefficients (poly_order+1 entries)

   // ~~~~~~
   PetscInt subspace_size, n_size, iwork_size, i_loc;
   // BLAS/LAPACK integer arguments must be PetscBLASInt
   PetscBLASInt lwork, rank, info;
   PetscBLASInt m_bl, n_bl, nrhs_bl, lda_bl, ldb_bl;
   PetscBLASInt *iwork;
   PetscReal *g0, *s;
   PetscReal *work;
   // Column-major n_size x n_size
   PetscReal *R_pointer;
   PetscReal rcond = -1.0;
   // ~~~~~~

   PetscFunctionBegin;

   // We call build_gmres_polynomial_inverse to build our neumann polynomials
   // but they don't need to calculate coefficients, so this skips this routine
   if (!buffers->R_buffer_receive) PetscFunctionReturn(PETSC_SUCCESS);

   // This is how many columns we have in K_m
   subspace_size = poly_order + 1;

   // Finish the asynchronous wait for the tsqr
   // This will fill buffers->R_buffer_receive with [n_size R(:,:)] from our tsqr
   PetscCall(finish_tsqr_parallel(buffers));
   // This should be the same size as subspace_size + 1
   n_size = (PetscInt)buffers->R_buffer_receive[0];
   PetscCheck(n_size == subspace_size + 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ERROR in sizes in TSQR");

   // Point directly at the R block
   // NOTE: this raw pointer ties R_pointer (PetscReal) to the TSQR
   // R_buffer_receive. The whole TSQR -> coefficients chain is deliberately kept
   // PetscReal (see docs/plan); any future PetscScalar retype must treat TSQR.c
   // and this finish path as one atomic unit.
   R_pointer = &buffers->R_buffer_receive[1];
   PetscCall(PetscCalloc1(poly_order + 2, &g0));
   PetscCall(PetscMalloc1(poly_order + 2, &s));
   // We know beta is in the top left of R
   g0[0] = R_pointer[0];

   // We've now finished orthogonalising, let's solve our least squares system
   PetscCall(PetscMalloc1(1, &work));
   PetscCall(PetscMalloc1(1, &iwork));
   lwork = -1;

   // Now we are using dgelsd here as there is no guarantee our power basis will be full rank
   // eg if our matrix is a (scaled) version of the identity, our columns in the power basis
   // will be very close (given the random rhs)
   // to orthogonal and dgels fails without full rank
   // Start from the second column in R_pointer
   // Kind-correct BLAS integer dimensions
   PetscCall(PetscBLASIntCast(subspace_size + 1, &m_bl));
   PetscCall(PetscBLASIntCast(subspace_size, &n_bl));
   nrhs_bl = 1;
   PetscCall(PetscBLASIntCast(subspace_size + 1, &lda_bl));
   PetscCall(PetscBLASIntCast(poly_order + 2, &ldb_bl));
   PetscCallBLAS("LAPACKgelsd", LAPACKgelsd_(&m_bl, &n_bl, &nrhs_bl, &R_pointer[n_size], &lda_bl,
            g0, &ldb_bl, s, &rcond, &rank,
            work, &lwork, iwork, &info));
   // Workspace query returns optimal sizes in work(1)/iwork(1); int() is exact
   // for all plausible sizes < 2^24 even when work is single precision
   PetscCall(PetscBLASIntCast((PetscInt)work[0], &lwork));
   iwork_size = iwork[0];
   PetscCall(PetscFree(work));
   PetscCall(PetscFree(iwork));
   PetscCall(PetscMalloc1(lwork, &work));
   PetscCall(PetscMalloc1(iwork_size, &iwork));
   PetscCallBLAS("LAPACKgelsd", LAPACKgelsd_(&m_bl, &n_bl, &nrhs_bl, &R_pointer[n_size], &lda_bl,
            g0, &ldb_bl, s, &rcond, &rank,
            work, &lwork, iwork, &info));
   PetscCall(PetscFree(work));
   PetscCall(PetscFree(iwork));

   PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LS solve failed");

   // These are the polynomial coefficients
   for (i_loc = 0; i_loc < subspace_size; i_loc++) coefficients[i_loc] = g0[i_loc];

   // Deallocate the receive buffer
   PetscCall(PetscFree(buffers->R_buffer_receive));

   PetscCall(PetscFree(g0));
   PetscCall(PetscFree(s));

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode mat_mult_powers_share_sparsity(Mat matrix, int poly_order, int poly_sparsity_order, tsqr_buffers *buffers,
   PetscReal coefficients[], PetscInt coefficients_size, Mat *reuse_mat, Mat **reuse_submatrices, Mat *cmat)
{
   // Wrapper around mat_mult_powers_share_sparsity_cpu and mat_mult_powers_share_sparsity_kokkos

   // ~~~~~~~~~~
#if defined(PETSC_HAVE_KOKKOS)
   Mat A_array, B_array = NULL, reuse_array;
   int reuse_int_cmat, reuse_int_reuse_mat;
   MatType mat_type;
   Mat temp_mat = NULL, temp_mat_reuse, temp_mat_compare;
   PetscReal normy;
   PetscBool reuse_triggered_cmat, reuse_triggered_reuse_mat;
   Mat reuse_mat_cpu = NULL;
   Mat *reuse_submatrices_cpu;
   Vec max_vec;
   PetscInt row_loc;
   PetscBool is_kokkos;
#endif
   // ~~~~~~~~~~

   PetscFunctionBegin;

   // ~~~~~~~~~~
   // Special case if we just want to return a gmres polynomial with the sparsity of the diagonal
   // This is like a damped Jacobi
   // ~~~~~~~~~~
   if (poly_sparsity_order == 0) {

      PetscCall(build_gmres_polynomial_inverse_0th_order_sparsity(matrix, poly_order, buffers,
               coefficients, cmat));

      PetscFunctionReturn(PETSC_SUCCESS);
   }

#if defined(PETSC_HAVE_KOKKOS)

   PetscCall(MatGetType(matrix, &mat_type));
   PetscCall(PetscStrcmpAny(mat_type, &is_kokkos, MATMPIAIJKOKKOS, MATSEQAIJKOKKOS, MATAIJKOKKOS, ""));
   if (is_kokkos) {

      // Finish off the non-blocking all reduce to compute our coefficients
      // This was a fortran routine, so it is called here rather than inside the kokkos
      PetscCall(finish_gmres_polynomial_coefficients_power(poly_order, buffers, coefficients));

      A_array = matrix;
      reuse_triggered_cmat = *cmat ? PETSC_TRUE : PETSC_FALSE;
      reuse_triggered_reuse_mat = *reuse_mat ? PETSC_TRUE : PETSC_FALSE;
      reuse_int_cmat = 0;
      if (reuse_triggered_cmat) {
         reuse_int_cmat = 1;
         B_array = *cmat;
      }
      reuse_int_reuse_mat = 0;
      if (reuse_triggered_reuse_mat) {
         reuse_int_reuse_mat = 1;
      }
      reuse_array = *reuse_mat;
      // The kokkos version doesn't take the buffers, we finish all the async stuff before
      // the kokkos routine is called
      mat_mult_powers_share_sparsity_kokkos(&A_array, poly_order, poly_sparsity_order,
               coefficients, reuse_int_reuse_mat, &reuse_array, reuse_int_cmat, &B_array);

      *reuse_mat = reuse_array;
      *cmat = B_array;

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
         PetscCall(mat_mult_powers_share_sparsity_cpu(matrix, poly_order, poly_sparsity_order,
                  buffers, coefficients, coefficients_size, &reuse_mat_cpu, &reuse_submatrices_cpu, &temp_mat));
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
         if (normy > PFLARE_TOL_MATFREE_4EM11 || PetscIsNanReal(normy)) {
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

      PetscCall(mat_mult_powers_share_sparsity_cpu(matrix, poly_order, poly_sparsity_order,
               buffers, coefficients, coefficients_size, reuse_mat, reuse_submatrices, cmat));
   }
#else
   PetscCall(mat_mult_powers_share_sparsity_cpu(matrix, poly_order, poly_sparsity_order,
            buffers, coefficients, coefficients_size, reuse_mat, reuse_submatrices, cmat));
#endif

   // ~~~~~~~~~~

   PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode mat_mult_powers_share_sparsity_cpu(Mat matrix, int poly_order, int poly_sparsity_order, tsqr_buffers *buffers,
   PetscReal coefficients[], PetscInt coefficients_size, Mat *reuse_mat, Mat **reuse_submatrices, Mat *cmat)
{
   // Compute matrix powers c = coeff(1) * I + coeff(2) * A + coeff(3) * A^2 + coeff(4) * A^3 + ...
   // where a c and the powers all share the same sparsity as the power input in poly_sparsity_order
   // This also finishes the async comms required to compute the gmres poly coefficients if buffers->request is allocated

   // ~~~~~~~~~~
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one, row_index_into_submatrix;
   PetscInt global_col_start, global_col_end_plus_one, n, ncols, ncols_two, ifree, max_nnzs;
   PetscInt i_loc, j_loc, k_loc, row_size, rows_ao, cols_ao, rows_ad, cols_ad, shift = 0;
   int match_counter;
   PetscInt term, order;
   PetscMPIInt comm_size;
   int *cols_index_one, *cols_index_two;
   PetscInt *col_indices_off_proc_array, *ad_indices, *cols;
   // Matrix entries filled by MatGetRow are PetscScalar
   PetscScalar *vals;
   IS col_indices, row_indices;
   Mat Ad, Ao;
   const PetscInt *colmap = NULL;
   PetscBool deallocate_submatrices;
   Mat *matrix_powers;
   Mat mat_sparsity_match;
   // The symbolic of each row: the matching local column indices and the matching values
   // of matrix for each column of the row of mat_sparsity_match (NULL when there are none)
   int **symbolic_ones;
   PetscReal **symbolic_vals;
   PetscInt *symbolic_counts;
   MPI_Comm MPI_COMM_MATRIX;
   PetscReal *vals_power_temp, *vals_previous_power_temp;
   // The values handed to MatSetValues (the Fortran passed an array expression)
   PetscScalar *set_vals;
   const PetscInt *submatrices_ia, *submatrices_ja, *cols_two_ptr, *cols_ptr;
   PetscInt *cols_two_alloc = NULL;
   // Matrix entries filled by MatSeqAIJGetArray/MatGetRow are PetscScalar
   const PetscScalar *vals_two_ptr, *vals_ptr;
   const PetscScalar *submatrices_vals;
   PetscBool reuse_triggered;
   PetscBool symmetric = PETSC_FALSE, inodecompressed = PETSC_FALSE, done;
   PetscInt one = 1, global_row;
   // ~~~~~~~~~~

   PetscFunctionBegin;

   deallocate_submatrices = PETSC_FALSE;

   PetscCheck(poly_sparsity_order < coefficients_size - 1, PetscObjectComm((PetscObject)matrix), PETSC_ERR_ARG_OUTOFRANGE, "Requested sparsity is greater than or equal to the order");

   PetscCall(PetscObjectGetComm((PetscObject)matrix, &MPI_COMM_MATRIX));
   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));

   // Get the local sizes
   PetscCall(MatGetLocalSize(matrix, &local_rows, &local_cols));
   PetscCall(MatGetSize(matrix, &global_rows, &global_cols));
   // This returns the global index of the local portion of the matrix
   PetscCall(MatGetOwnershipRange(matrix, &global_row_start, &global_row_end_plus_one));
   PetscCall(MatGetOwnershipRangeColumn(matrix, &global_col_start, &global_col_end_plus_one));

   reuse_triggered = *cmat ? PETSC_TRUE : PETSC_FALSE;

   // ~~~~~~~~~~
   // Compute any matrix powers we might need to constrain sparsity and start assembling the
   // components of the output matrix up to the order of poly_sparsity_order
   // The powers higher than poly_sparsity_order can be done with only
   // a single bit of comms and is done below this
   // ~~~~~~~~~~

   // matrix_powers stores all the powers of the input matrix
   // (matrix_powers[order - 1] is the power order)
   PetscCall(PetscMalloc1(coefficients_size - 1, &matrix_powers));
   matrix_powers[0] = matrix;

   // What power of A do we want to match the sparsity of
   // Compute the power we need if we're two or above
   for (order = 2; order <= poly_sparsity_order; order++) {

      // Let's just store each power, that way we can set the sparsity
      // as the highest (unconstrained) power and do the mataxpy with a subset of entries
      // Takes more memory to do this but is faster
      PetscCall(MatMatMult(matrix, matrix_powers[order - 2],
               MAT_INITIAL_MATRIX, PFLARE_MATMULT_FILL, &matrix_powers[order - 1]));
   }

   // mat_sparsity_match now contains the sparsity of the power of A we want to match
   mat_sparsity_match = matrix_powers[poly_sparsity_order - 1];

   // Copy in the highest unconstrained power
   // Duplicate & copy the matrix, but ensure there is a diagonal present
   PetscCall(mat_duplicate_copy_plus_diag(matrix_powers[poly_sparsity_order - 1], reuse_triggered, cmat));

   // We know we will never have non-zero locations outside of the highest constrained sparsity power
   PetscCall(MatSetOption(*cmat, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
   PetscCall(MatSetOption(*cmat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
   // We know we are only going to insert local vals
   // These options should turn off any reductions in the assembly
   PetscCall(MatSetOption(*cmat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));

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
   PetscCall(PetscMalloc1(max_nnzs, &cols_index_one));
   PetscCall(PetscMalloc1(max_nnzs, &cols_index_two));
   PetscCall(PetscMalloc1(max_nnzs, &set_vals));

   // ~~~~~~~~~~~
   // Finish the non-blocking all-reduce and compute our polynomial coefficients
   // ~~~~~~~~~~~
   PetscCall(finish_gmres_polynomial_coefficients_power(poly_order, buffers, coefficients));

   // Scale the highest constrained power
   PetscCall(MatScale(*cmat, coefficients[poly_sparsity_order]));

   // Then go backwards and add in each of the coefficients * A^order from the second highest order down
   for (order = poly_sparsity_order - 1; order >= 1; order--) {

      // Do result = alpha_1 * A_ff + alpha_2 * A_ff^2 + ....
      // Can use SUBSET_NONZERO_PATTERN as we have put the highest order power in first
      PetscCall(MatAXPY(*cmat, coefficients[order], matrix_powers[order - 1], SUBSET_NONZERO_PATTERN));
   }

   // Add in the 0th order term
   for (i_loc = 0; i_loc < local_rows; i_loc++) {

      // Add in the I term - 0th order term
      PetscCall(MatSetValue(*cmat, global_row_start + i_loc, global_row_start + i_loc,
               coefficients[0], ADD_VALUES));
   }

   // ~~~~~~~~~~~~
   // From here we now have cmat with the correct values up to the power poly_sparsity_order
   // and hence we want to add in the sparsity constrained powers
   // ~~~~~~~~~~~~

   // Now go through and compute the sum of the matrix powers
   // We're doing row-wise matmatmults here assuming the fixed sparsity
   // We exploit the fact that the subsequent matrix powers can be done
   // one row at a time, so we only have to retrieve the needed vals from mat_sparsity_match once
   for (i_loc = 0; i_loc < local_rows; i_loc++) {

      // Get the row of mat_sparsity_match
      PetscCall(MatGetRow(mat_sparsity_match, i_loc + global_row_start, &ncols_two, &cols_ptr, &vals_ptr));
      // Copying here because mat_sparsity_match and matrix are often the same matrix
      // and hence we can only have one active matgetrow
      ncols = ncols_two;
      for (j_loc = 0; j_loc < ncols; j_loc++) {
         cols[j_loc] = cols_ptr[j_loc];
         vals[j_loc] = vals_ptr[j_loc];
      }
      PetscCall(MatRestoreRow(mat_sparsity_match, i_loc + global_row_start, &ncols_two, &cols_ptr, &vals_ptr));

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
      // vals_power_temp stores the value of A^(term-1) for this row, and we update this as we go through
      // the term loop
      for (term = poly_sparsity_order + 2; term <= coefficients_size; term++) {

         // We need to sum up the product of vals_previous_power_temp(j_loc) * matching columns
         for (j_loc = 0; j_loc < ncols; j_loc++) vals_power_temp[j_loc] = 0;

         // Have to finish all the columns before we move onto the next coefficient
         for (j_loc = 0; j_loc < ncols; j_loc++) {

            // If we have no matching columns cycle this row
            if (!symbolic_ones[j_loc]) continue;

            // symbolic_vals(j_loc)%ptr has the matching values of A in it
            for (k_loc = 0; k_loc < symbolic_counts[j_loc]; k_loc++) {
               vals_power_temp[symbolic_ones[j_loc][k_loc]] = vals_power_temp[symbolic_ones[j_loc][k_loc]] +
                        vals_previous_power_temp[j_loc] * symbolic_vals[j_loc][k_loc];
            }
         }

         // ~~~~~~~~~~~
         // Now can add the value of coeff * A^(term-1) to our matrix
         // Can skip this if coeff is zero, but still need to compute A^(term-1)
         // for the next time through
         // ~~~~~~~~~~~
         if (ncols != 0 && coefficients[term - 1] != 0.0) {
            for (j_loc = 0; j_loc < ncols; j_loc++) set_vals[j_loc] = coefficients[term - 1] * vals_power_temp[j_loc];
            global_row = global_row_start + i_loc;
            PetscCall(MatSetValues(*cmat, one, &global_row, ncols, cols, set_vals, ADD_VALUES));
         }

         // This should now have the value of A^(term-1) in it
         for (j_loc = 0; j_loc < ncols; j_loc++) vals_previous_power_temp[j_loc] = vals_power_temp[j_loc];
      }

      // Delete our symbolic
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
   for (order = 2; order <= poly_sparsity_order; order++) {
      PetscCall(MatDestroy(&matrix_powers[order - 1]));
   }
   PetscCall(PetscFree(matrix_powers));
   if (deallocate_submatrices) {
      PetscCall(PetscFree(*reuse_submatrices));
   }

   PetscCall(PetscFree(col_indices_off_proc_array));
   PetscCall(PetscFree(cols));
   PetscCall(PetscFree(vals));
   PetscCall(PetscFree(vals_power_temp));
   PetscCall(PetscFree(vals_previous_power_temp));
   PetscCall(PetscFree(cols_index_one));
   PetscCall(PetscFree(cols_index_two));
   PetscCall(PetscFree(set_vals));

   // Finish assembly
   PetscCall(MatAssemblyEnd(*cmat, MAT_FINAL_ASSEMBLY));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode build_gmres_polynomial_inverse(Mat matrix, int poly_order, tsqr_buffers *buffers, PetscReal coefficients[], PetscInt coefficients_size,
   int poly_sparsity_order, PetscBool matrix_free, PetscBool diag_scale_polys, Mat *reuse_mat, Mat **reuse_submatrices, Mat *inv_matrix)
{
   // Assembles a matrix which is an approximation to the inverse of a matrix using the
   // gmres polynomial coefficients
   // This finishes the async comms required to compute the coefficients as late as possible
   // Can constrain the sparsity to that an arbritrary matrix power of the input by setting
   // poly_sparsity_order < poly_order

   // ~~~~~~
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt global_rows, global_cols, local_rows, local_cols;
   PetscInt order, top_order;
   PetscMPIInt comm_size;
   MPI_Comm MPI_COMM_MATRIX;
   VecType vtype;
   Mat mat_power = NULL, temp_mat, diag_scaled_mat;
   Vec diag_inverse_vec = NULL;
   mat_ctxtype *mat_ctx = NULL, *mat_ctx_scaled = NULL;
   PetscBool reuse_triggered;
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
   // This returns the global index of the local portion of the matrix
   PetscCall(MatGetOwnershipRange(matrix, &global_row_start, &global_row_end_plus_one));

   // Just build a matshell that applies our polynomial matrix-free
   if (matrix_free) {

      // Finish off the non-blocking all reduce to compute our coefficients
      PetscCall(finish_gmres_polynomial_coefficients_power(poly_order, buffers, coefficients));

      // If not re-using
      if (!*inv_matrix) {

         // Have to dynamically allocate this
         PetscCall(create_mat_ctx(&mat_ctx));

         // We pass in the polynomial coefficients as the context
         PetscCall(MatCreateShell(MPI_COMM_MATRIX, local_rows, local_cols, global_rows, global_cols,
                  mat_ctx, inv_matrix));
         if (diag_scale_polys) {
            // The subroutine petsc_matvec_right_scale_poly_mf applies
            // q(mat_ctx->mat_scaled) D^-1
            PetscCall(MatShellSetOperation(*inv_matrix,
                     MATOP_MULT, (PetscErrorCodeFn *)petsc_matvec_right_scale_poly_mf));
         } else {
            // The subroutine petsc_matvec_poly_mf applies the polynomial inverse, q(mat_ctx->mat)
            PetscCall(MatShellSetOperation(*inv_matrix,
                     MATOP_MULT, (PetscErrorCodeFn *)petsc_matvec_poly_mf));
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

         // Create temporary vector we use during horner
         // Make sure to use matrix here to get the right type (as the shell doesn't know about gpus)
         PetscCall(MatCreateVecs(matrix, &mat_ctx->mf_temp_vec[MF_VEC_TEMP], NULL));

         // Need to build mat_ctx->mat_scaled, ie D^-1 A
         if (diag_scale_polys) {

            // Have to dynamically allocate this
            PetscCall(create_mat_ctx(&mat_ctx_scaled));
            mat_ctx_scaled->coefficients = coefficients;
            mat_ctx_scaled->coefficients_size = coefficients_size;

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

            // Create temporary vector we use during horner
            // Make sure to use matrix here to get the right type (as the shell doesn't know about gpus)
            PetscCall(MatCreateVecs(matrix, &mat_ctx->mf_temp_vec[MF_VEC_RHS], &mat_ctx->mf_temp_vec[MF_VEC_DIAG]));
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

      // This is the matrix whose inverse we are applying (just copying the pointer here)
      mat_ctx->mat = matrix;
      mat_ctx->coefficients = coefficients;
      mat_ctx->coefficients_size = coefficients_size;
      if (diag_scale_polys) {
         mat_ctx_scaled->mat = matrix;

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

   // If we're zeroth order poly this is trivial as it's just the coefficient(1) on the diagonal
   if (poly_order == 0) {

      PetscCall(build_gmres_polynomial_inverse_0th_order(diag_scaled_mat, poly_order, buffers, coefficients,
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

      // Flags to prevent reductions when assembling (there are assembles in the shift)
      PetscCall(MatSetOption(*inv_matrix, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
      PetscCall(MatSetOption(*inv_matrix, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
      PetscCall(MatSetOption(*inv_matrix, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));

      // Finish off the non-blocking all reduce to compute our coefficients
      PetscCall(finish_gmres_polynomial_coefficients_power(poly_order, buffers, coefficients));

      // result = alpha_1 * A_ff
      PetscCall(MatScale(*inv_matrix, coefficients[1]));
      // Now have to go through and add alpha_0 to the other terms
      // Don't need an assemble as there is one called in this
      PetscCall(MatShift(*inv_matrix, coefficients[0]));

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
      // This also finishes off the asyn comms and computes the coefficients
      PetscCall(mat_mult_powers_share_sparsity(diag_scaled_mat, poly_order, poly_sparsity_order, buffers, coefficients, coefficients_size,
               reuse_mat, reuse_submatrices, inv_matrix));

      if (diag_scale_polys) {
         PetscCall(MatDestroy(&diag_scaled_mat));
         PetscCall(MatDiagonalScale(*inv_matrix, NULL, diag_inverse_vec));
         PetscCall(VecDestroy(&diag_inverse_vec));
      }

      // Then just return
      PetscFunctionReturn(PETSC_SUCCESS);
   }

   // ~~~~~~~~~~
   // We are only here if we don't constrain_sparsity
   // ~~~~~~~~~~

   // If not re-using
   // Copy in the initial matrix
   if (!reuse_triggered) {
      // Duplicate & copy the matrix, but ensure there is a diagonal present
      PetscCall(mat_duplicate_copy_plus_diag(diag_scaled_mat, PETSC_FALSE, inv_matrix));
   } else {
      // For the powers > 1 the pattern of the original matrix will be different
      // to the resulting inverse
      PetscCall(MatCopy(diag_scaled_mat, *inv_matrix, DIFFERENT_NONZERO_PATTERN));
   }

   // Don't set any off processor entries so no need for a reduction when assembling
   PetscCall(MatSetOption(*inv_matrix, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));

   // Finish off the non-blocking all reduce to compute our coefficients
   PetscCall(finish_gmres_polynomial_coefficients_power(poly_order, buffers, coefficients));

   // result = alpha_1 * A_ff
   PetscCall(MatScale(*inv_matrix, coefficients[1]));

   // ~~~~~~~~~~~~~
   // For terms 2nd order and higher
   // ~~~~~~~~~~~~~

   // Only the leading (highest order) zero coefficients can be skipped entirely,
   // any lower order zero coefficient still needs its matrix power computed for the next order
   top_order = highest_nonzero_coeff(coefficients, coefficients_size);
   for (order = 2; order <= top_order - 1; order++) {

      // TODO - these can be reused
      if (order == 2) {
         PetscCall(MatMatMult(diag_scaled_mat, diag_scaled_mat,
                  MAT_INITIAL_MATRIX, PFLARE_MATMULT_FILL, &temp_mat));
      } else {
         PetscCall(MatMatMult(diag_scaled_mat, mat_power,
                  MAT_INITIAL_MATRIX, PFLARE_MATMULT_FILL, &temp_mat));
         PetscCall(MatDestroy(&mat_power));
      }

      // Copy the temporary into mat_power
      PetscCall(MatDuplicate(temp_mat, MAT_COPY_VALUES, &mat_power));
      PetscCall(MatDestroy(&temp_mat));

      // Do result = alpha_1 * A_ff + alpha_2 * A_ff^2 + ....
      // Skip the add if the coefficient is zero
      if (coefficients[order] == 0.0) {
         continue;
      } else if (reuse_triggered) {
         // If doing reuse we know our nonzeros are a subset
         PetscCall(MatAXPY(*inv_matrix, coefficients[order], mat_power, SUBSET_NONZERO_PATTERN));
      } else {
         // Have to use the DIFFERENT_NONZERO_PATTERN here
         PetscCall(MatAXPYWrapper(*inv_matrix, coefficients[order], mat_power));
      }
   }
   // Don't forget to destroy the last power
   PetscCall(MatDestroy(&mat_power));

   // The alpha_0 just gets added to the diagonal
   // There is an assemble in the shift so don't need a separate one
   PetscCall(MatShift(*inv_matrix, coefficients[0]));

   if (diag_scale_polys) {
      PetscCall(MatDestroy(&diag_scaled_mat));
      PetscCall(MatDiagonalScale(*inv_matrix, NULL, diag_inverse_vec));
      PetscCall(VecDestroy(&diag_inverse_vec));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode build_gmres_polynomial_inverse_0th_order(Mat matrix, int poly_order, tsqr_buffers *buffers, PetscReal coefficients[], Mat *inv_matrix)
{
   // Specific 0th order inverse

   // ~~~~~~
   PetscBool reuse_triggered;
   Vec diag_vec;
   // ~~~~~~

   PetscFunctionBegin;

   PetscCheck(poly_order == 0, PetscObjectComm((PetscObject)matrix), PETSC_ERR_ARG_WRONG, "This is a 0th order inverse, but poly_order is not 0");

   // Let's create a matrix to represent the inverse diagonal
   reuse_triggered = *inv_matrix ? PETSC_TRUE : PETSC_FALSE;

   if (!reuse_triggered) {
      PetscCall(MatCreateVecs(matrix, NULL, &diag_vec));
   } else {
      PetscCall(MatDiagonalGetDiagonal(*inv_matrix, &diag_vec));
   }

   // Finish off the non-blocking all reduce to compute our coefficients
   PetscCall(finish_gmres_polynomial_coefficients_power(poly_order, buffers, coefficients));
   PetscCall(VecSet(diag_vec, coefficients[0]));

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

PETSC_INTERN PetscErrorCode build_gmres_polynomial_inverse_0th_order_sparsity(Mat matrix, int poly_order, tsqr_buffers *buffers, PetscReal coefficients[], Mat *inv_matrix)
{
   // Specific inverse with 0th order sparsity

   // ~~~~~~
   PetscInt order;
   PetscBool reuse_triggered;
   Vec inv_vec, diag_vec, power_vec;
   // ~~~~~~

   PetscFunctionBegin;

   reuse_triggered = *inv_matrix ? PETSC_TRUE : PETSC_FALSE;

   // Our matrix has to be square
   PetscCall(MatCreateVecs(matrix, &power_vec, &diag_vec));
   PetscCall(MatGetDiagonal(matrix, diag_vec));

   // This stores D^order
   if (!reuse_triggered) {
      PetscCall(VecDuplicate(diag_vec, &inv_vec));
   } else {
      PetscCall(MatDiagonalGetDiagonal(*inv_matrix, &inv_vec));
   }
   PetscCall(VecCopy(diag_vec, power_vec));

   // ~~~~~~~~~~~
   // Finish the non-blocking all-reduce and compute our polynomial coefficients
   // ~~~~~~~~~~~
   PetscCall(finish_gmres_polynomial_coefficients_power(poly_order, buffers, coefficients));

   // Set: alpha_0 * I
   PetscCall(VecSet(inv_vec, coefficients[0]));

   // Calculate: alpha_0 * I + alpha_1 * D + alpha_2 * D^2
   for (order = 1; order <= poly_order; order++) {
      PetscCall(VecAXPY(inv_vec, coefficients[order], power_vec));
      // Compute power_vec = power_vec * D
      if (order != poly_order) PetscCall(VecPointwiseMult(power_vec, power_vec, diag_vec));
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
   PetscCall(VecDestroy(&power_vec));

   PetscFunctionReturn(PETSC_SUCCESS);
}
