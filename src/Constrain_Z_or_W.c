/*
  C port of Constrain_Z_or_W.F90 (module constrain_z_or_w)

  The arrays of near nullspace vectors are a Vec array plus its size (size() of
  the Fortran array). get_near_nullspace allocates the two arrays it returns
  (PetscCalloc1, so the vectors it does not create are NULL, as a Fortran tVec
  is null until created); the caller frees them with PetscFree after destroying
  the vectors. The dense work arrays of constrain_grid_transfer are column-major,
  with the leading dimensions of the Fortran arrays, and the BLAS calls take the
  same arguments as the Fortran PFLAREgemv/PFLAREgemm calls
*/

#include <petscksp.h>
#include <petscblaslapack.h>
#include "pflareimpl.h"
#include "PETSc_Helper.h"
#include "Constrain_Z_or_W.h"

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode get_near_nullspace(Mat input_mat, PetscBool left, PetscBool right,
   Vec **left_null_vecs, Vec **right_null_vecs, PetscInt *null_vecs_size)
{
   // Gets left and/or right near nullspace vectors set by the user with
   // matsetnearnullspace

   // ~~~~~~
   PetscMPIInt comm_size;
   MatNullSpace nullspace = NULL;
   MPI_Comm MPI_COMM_MATRIX;
   PetscBool has_constant = PETSC_FALSE;
   PetscInt no_nullspace, no_nullspace_vecs, i_loc;
   const Vec *null_vecs = NULL;
   PetscBool cst_nullspace, called_get_vecs;
   // ~~~~~

   PetscFunctionBegin;

   // Get the communicator the input matrix is on, we build everything on that
   PetscCall(PetscObjectGetComm((PetscObject)input_mat, &MPI_COMM_MATRIX));

   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));
   called_get_vecs = PETSC_FALSE;

   // Get the near nullspace if the user has set one
   if (left || right) PetscCall(MatGetNearNullSpace(input_mat, &nullspace));
   // If the user hasn't set a near nullspace and still wants constraints
   // we use the constant as a default
   cst_nullspace = PETSC_FALSE;
   no_nullspace_vecs = 0;
   no_nullspace = 0;
   if (!nullspace &&
         (left || right)) {

      cst_nullspace = PETSC_TRUE;
      no_nullspace_vecs = 1;

   // If the user has provided a near null-space
   } else if (nullspace &&
               (left || right)) {

      // Get the nullspace vectors
      PetscCall(MatNullSpaceGetVecs(nullspace, &has_constant, &no_nullspace, &null_vecs));
      called_get_vecs = PETSC_TRUE;
      no_nullspace_vecs = no_nullspace;

      // If has_constant is true, then null_vecs won't contain the constant vector
      // but it is still part of the nullspace - we explicitly create it below
      if (has_constant) {
         cst_nullspace = PETSC_TRUE;
         no_nullspace_vecs = no_nullspace_vecs + 1;
      }
   }

   // Create some storage for all the near nullspace vectors
   // constant included
   PetscCall(PetscCalloc1(no_nullspace_vecs, left_null_vecs));
   PetscCall(PetscCalloc1(no_nullspace_vecs, right_null_vecs));
   *null_vecs_size = no_nullspace_vecs;
   // Have to make copies of the nullspace vecs passed in, as MatNullSpaceCreate locks the vectors
   // to be read-only in newer petsc
   for (i_loc = 0; i_loc < no_nullspace; i_loc++) {
      if (left) {
         PetscCall(VecDuplicate(null_vecs[i_loc], &(*left_null_vecs)[i_loc]));
         PetscCall(VecCopy(null_vecs[i_loc], (*left_null_vecs)[i_loc]));
      }
      if (right) {
         PetscCall(VecDuplicate(null_vecs[i_loc], &(*right_null_vecs)[i_loc]));
         PetscCall(VecCopy(null_vecs[i_loc], (*right_null_vecs)[i_loc]));
      }
   }

   // If we want to create a constant nullspace vector on the end of left_null_vecs or right_null_vecs
   if (cst_nullspace) {

      if (left) {
         PetscCall(MatCreateVecs(input_mat, &(*left_null_vecs)[no_nullspace_vecs - 1], NULL));
         // Set to the constant
         PetscCall(VecSet((*left_null_vecs)[no_nullspace_vecs - 1], PFLARE_ONE));
      }

      if (right) {
         PetscCall(MatCreateVecs(input_mat, &(*right_null_vecs)[no_nullspace_vecs - 1], NULL));
         // Set to the constant
         PetscCall(VecSet((*right_null_vecs)[no_nullspace_vecs - 1], PFLARE_ONE));
      }
   }

   // The Fortran calls MatNullSpaceRestoreVecs here when it called MatNullSpaceGetVecs;
   // that routine is Fortran only, in C the array is owned by the MatNullSpace
   (void)called_get_vecs;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode smooth_near_nullspace(Mat input_mat, PetscBool left, PetscBool right,
   Vec left_null_vecs[], PetscInt left_null_vecs_size, Vec right_null_vecs[], PetscInt right_null_vecs_size)
{
   // Smooths left and/or right near nullspace vectors

   // ~~~~~~
   PetscMPIInt comm_size;
   PetscInt i_loc;
   PetscInt local_rows, global_rows, local_cols, global_cols;
   MPI_Comm MPI_COMM_MATRIX;
   Vec vec_rhs;
   KSP ksp;
   PC pc;
   PetscInt maxits = 15;
   // ~~~~~~

   PetscFunctionBegin;

   // Just return if we don't want either left or right (or both)
   if (!(left || right)) {
      PetscFunctionReturn(PETSC_SUCCESS);
   }

   PetscCall(PetscObjectGetComm((PetscObject)input_mat, &MPI_COMM_MATRIX));
   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));
   PetscCall(MatGetSize(input_mat, &global_rows, &global_cols));
   PetscCall(MatGetLocalSize(input_mat, &local_rows, &local_cols));

   // Create the vecs
   PetscCall(MatCreateVecs(input_mat, &vec_rhs, NULL));

   // ~~~~~~~~~~~~
   // Setup a petsc ksp to solve Ax = 0
   // ~~~~~~~~~~~~

   // We want the nullspace so solve homog problem
   PetscCall(VecSet(vec_rhs, PFLARE_ZERO));

   PetscCall(KSPCreate(MPI_COMM_MATRIX, &ksp));
   //call KSPSetType(ksp, KSPGMRES, ierr)
   PetscCall(KSPSetType(ksp, KSPRICHARDSON));
   // Scale the richardson with a dampening parameter
   PetscCall(KSPRichardsonSetSelfScale(ksp, PETSC_TRUE));

   PetscCall(KSPGetPC(ksp, &pc));
   PetscCall(PCSetType(pc, PCJACOBI));
   // Use the L1 Jacobi
   //call PCJacobiSetType(pc, 1, ierr)

   // Do 15 iterations
   // (PETSC_CURRENT is the value of the Fortran PETSC_DEFAULT_REAL)
   PetscCall(KSPSetTolerances(ksp, PFLARE_KSP_RTOL_CONSTRAIN,
            PFLARE_KSP_ATOL_OFF,
            PETSC_CURRENT,
            maxits));

   PetscCall(KSPSetOperators(ksp, input_mat, input_mat));
   //call KSPSetFromOptions(ksp_coarse_solver, ierr)
   PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
   // Keep the residual norm on (in both precisions) so the convergence test
   // halts the solve before the self-scale Richardson divides by a zero
   // (Ap,Ap). In single precision that dot product underflows once the
   // near-nullspace residual gets small (Ap ~ 1e-20 squared underflows float
   // tiny ~1e-38); in double it becomes exactly 0/0 when the vector collapses
   // entirely to zero, which happens on coarse levels of operators with no
   // left near-nullspace (pure advection with outflow BCs). The convergence
   // test runs before the scale computation each iteration, so a collapsed
   // residual exits cleanly instead of raising FE_INVALID - fatal under
   // -fp_trap. Healthy smoothing never contracts by the relative tolerance
   // within maxits sweeps, so results are unchanged where the vectors do not
   // collapse.
   PetscCall(KSPSetUp(ksp));

   // ~~~~~~~
   // Compute the left near nullspace vec
   // ~~~~~~~
   if (left) {

      // Smooth each vector
      for (i_loc = 0; i_loc < left_null_vecs_size; i_loc++) {

         // To get at the left near nullspace vectors we solve the transposed problem
         PetscCall(KSPSolveTranspose(ksp, vec_rhs, left_null_vecs[i_loc]));

      }
   }

   // ~~~~~~~
   // Compute the right near nullspace vec
   // ~~~~~~~
   if (right) {

      // Smooth each vector
      for (i_loc = 0; i_loc < right_null_vecs_size; i_loc++) {

         // To get at the right near nullspace vectors we solve the untransposed problem
         PetscCall(KSPSolve(ksp, vec_rhs, right_null_vecs[i_loc]));
      }
   }

   PetscCall(VecDestroy(&vec_rhs));
   PetscCall(KSPDestroy(&ksp));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode constrain_grid_transfer(Mat *z_or_w, PetscBool is_z, Vec null_vecs_f[], Vec null_vecs_c[],
   PetscInt null_vecs_size)
{
   // Applies constraints based on near nullspace vectors to Z or W passed in
   // For Z, we need to apply constraints to the columns, so is_z should be passed in true
   // so that we can take a transpose to make things easy
   // For W, we need to apply constraints to the rows

   // ~~~~~~
   // Local variables
   PetscInt local_rows, local_cols, global_row_start;
   PetscInt global_row_end_plus_one, global_col_end_plus_one;
   PetscInt global_rows, global_cols, i_loc, j_loc;
   PetscInt rows_ao, cols_ao = 0, global_col_start, ncols;
   PetscInt cols_ad, rows_ad, ncols_ad, ncols_ao, ncols_temp;
   PetscInt null_vec;
   PetscMPIInt comm_size;
   MPI_Comm MPI_COMM_MATRIX;
   Mat row_mat, temp_mat_aij = NULL;
   PetscInt *col_indices_off_proc_array;
   const PetscInt *cols = NULL;
   // Matrix entries filled by MatGetRow are PetscScalar
   const PetscScalar *vals = NULL;
   // Dense near-nullspace work arrays (fed to dgemv/dgemm then MatSetValues)
   PetscScalar *row_vals, *diff;
   Mat new_z_or_w;
   Mat Ad, Ao = NULL, scatter_mat = NULL;
   Vec leaf_vec;
   PetscSF sf;
   const PetscInt *colmap = NULL;
   // VecGetArrayRead returns the raw Vec array (PetscScalar)
   const PetscScalar *b_c_nonlocal = NULL;
   const PetscScalar *b_c_local, *b_f_vals;
   // Column-major: b_c_nonlocal_alloc is cols_ao x size(null_vecs_c) (NULL until allocated),
   // b_c_vals ncols x size(null_vecs_c), bctbc and pseudo size(null_vecs_c) x size(null_vecs_c)
   // and temp_mat size(null_vecs_c) x ncols
   PetscScalar *b_c_nonlocal_alloc = NULL, *b_c_vals, *bctbc, *pseudo, *temp_mat;
   // Column-major: b_c_local_alloc is local_size_c x size(null_vecs_c),
   // b_f_local_alloc local_size_f x size(null_vecs_c)
   PetscScalar *b_c_local_alloc, *b_f_local_alloc;
   PetscInt local_size_c, local_size_f;
   PetscInt one = 1;
   // Global scale of the coarse near-nullspace vectors and the per-row test
   // against it, see PFLARE_TOL_CONSTRAIN_REL below
   PetscReal b_c_scale, b_c_row_norm, vec_norm_loc;
   // Kind-correct BLAS integer/real arguments for the dgemv/dgemm calls
   PetscBLASInt m_bl, n_bl, k_bl, lda_bl, ldb_bl, ldc_bl, one_bl;
   PetscScalar blas_one, blas_zero, blas_minus_one;
   MatType mat_type;
   // Whether row_mat is "mpiaij". The Fortran compares the type name again at the end,
   // but by then row_mat may have been destroyed, and with it the name a MatType points to
   PetscBool is_mpiaij, has_nan;
   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)*z_or_w, &MPI_COMM_MATRIX));
   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));

   // Have to do an explicit transpose if applying constraints to columns of Z
   if (is_z) {
      PetscCall(MatTranspose(*z_or_w, MAT_INITIAL_MATRIX, &row_mat));
   } else {
      row_mat = *z_or_w;
   }

   // Duplicate the sparsity pattern for our new Z or W
   PetscCall(MatDuplicate(row_mat, MAT_COPY_VALUES, &new_z_or_w));

   // We know we will never have non-zero locations outside of the sparsity power
   PetscCall(MatSetOption(new_z_or_w, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
   PetscCall(MatSetOption(new_z_or_w, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
   // We know we are only going to insert local vals
   // These options should turn off any reductions in the assembly
   PetscCall(MatSetOption(new_z_or_w, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));

   // ~~~~~~~~~

   // Get the local sizes
   PetscCall(MatGetLocalSize(row_mat, &local_rows, &local_cols));
   PetscCall(MatGetSize(row_mat, &global_rows, &global_cols));

   // Get the nnzs in row_mat
   PetscCall(MatGetOwnershipRange(row_mat, &global_row_start, &global_row_end_plus_one));
   PetscCall(MatGetOwnershipRangeColumn(row_mat, &global_col_start, &global_col_end_plus_one));

   // ~~~~~
   // What we are doing, with W for example:
   // The constraints we need to enforce are P B_c^R = B^R
   // ie preservation of the right near nullspace vectors
   // which given our ideal prolongator is equivalent to
   // W B_c^R = B_f^R     - (1)
   // This is enforced by computing an orthgonal projector and just doing
   // a one step method
   // W = W - (W * B_c^R - B_f^R ) * inv((B_c^R)^T * B_c^R) * (B_c^R)^T
   // which we can do one row at a time
   // See Eq 3.2 in Olson 2011 or
   // filter_operator in utils.py in PyAMG
   // If we wanted we could modify this to be done iteratively like clAIR, but
   // our approx Aff inverses are so good we don't really need to
   // If B is a vector of 1 for example, all the constraint does is scales by a row sum
   // We take a transpose of Z at the start so it is easy to access the columns
   // via the rows in petsc, we then just transpose at the end
   // ~~~~~

   // ~~~~
   // Now for each row, we need to get the values of B_c for both the local and nonlocal col indices
   // The nonlocal col indices are colmap in petsc
   // We can do the comms for this once by using the VecScatter built into the mat
   // ~~~~

   PetscCall(MatGetType(row_mat, &mat_type));
   PetscCall(PetscStrcmp(mat_type, "mpiaij", &is_mpiaij));

   // Nonlocal if needed
   if (comm_size != 1) {

      // Much more annoying in older petsc
      if (is_mpiaij) {
         PetscCall(MatMPIAIJGetSeqAIJ(row_mat, &Ad, &Ao, &colmap));
         scatter_mat = row_mat;

      // If on the gpu, just do a convert to mpiaij format first
      // This will be expensive but the best we can do for now without writing our
      // own version of this subroutine in cuda/kokkos
      } else {
         PetscCall(MatConvert(row_mat, MATMPIAIJ, MAT_INITIAL_MATRIX, &temp_mat_aij));
         PetscCall(MatMPIAIJGetSeqAIJ(temp_mat_aij, &Ad, &Ao, &colmap));
         scatter_mat = temp_mat_aij;
      }

      PetscCall(MatGetSize(Ad, &rows_ad, &cols_ad));
      // We know the col size of Ao is the size of colmap, the number of non-zero offprocessor columns
      PetscCall(MatGetSize(Ao, &rows_ao, &cols_ao));

      PetscCall(PetscMalloc1(cols_ao * null_vecs_size, &b_c_nonlocal_alloc));

      // Build a leaf Vec once (sized/typed to match Ao) and reuse it across
      // every null-vec iteration. The halo scatter uses the matrix's own mult
      // PetscSF (MatGetMultPetscSF), and every null_vecs_c(i) shares the column
      // layout of the matrix. Check the SF out once and reuse it for every vec.
      PetscCall(MatCreateVecs(Ao, &leaf_vec, NULL));
      PetscCall(MatGetMultPetscSF(scatter_mat, &sf));

      // Loop over all the near nullspace vectors and get the nonlocal components
      for (null_vec = 0; null_vec < null_vecs_size; null_vec++) {

         // We want the nonlocal values in B_c
         PetscCall(VecScatterBegin(sf, null_vecs_c[null_vec], leaf_vec, INSERT_VALUES, SCATTER_FORWARD));
         PetscCall(VecScatterEnd(sf, null_vecs_c[null_vec], leaf_vec, INSERT_VALUES, SCATTER_FORWARD));
         // b_c_nonlocal now contains all the nonlocal values of B_c we need for all the nonlocal columns
         // in every local row
         PetscCall(VecGetArrayRead(leaf_vec, &b_c_nonlocal));

         // Copy the values
         for (j_loc = 0; j_loc < cols_ao; j_loc++) b_c_nonlocal_alloc[j_loc + null_vec * cols_ao] = b_c_nonlocal[j_loc];
         // Make sure to restore as soon as we're done with it
         PetscCall(VecRestoreArrayRead(leaf_vec, &b_c_nonlocal));

      }

      PetscCall(VecDestroy(&leaf_vec));

   // In serial this is simple
   } else {

      Ad = row_mat;

   }

   // ~~~~~
   // Global scale of the coarse near-nullspace vectors.
   // The row-wise projection below divides by ||B_c|S_i||^2, where B_c|S_i is
   // B_c restricted to the sparsity pattern of row i. For some operators (e.g.
   // pure advection, where the smoothed left near-nullspace vector is driven to
   // zero in a layer near the outflow boundary) that restriction is numerically
   // zero for a subset of rows. Dividing by it amplifies the grid-transfer
   // entries by its reciprocal - observed as ||Z_constrained||/||Z|| ~ 1e2 and
   // a divergent solve. Such a row carries no usable constraint information, so
   // we leave it unconstrained instead. The test is relative to the vectors'
   // own global scale so it is invariant to a rescaling of B.
   // ~~~~~
   b_c_scale = 0.0;
   for (null_vec = 0; null_vec < null_vecs_size; null_vec++) {
      PetscCall(VecNorm(null_vecs_c[null_vec], NORM_INFINITY, &vec_norm_loc));
      if (vec_norm_loc > b_c_scale) b_c_scale = vec_norm_loc;
   }

   // If the near-nullspace vectors have collapsed to (numerically) zero
   // everywhere there is nothing to constrain against, and the per-row test
   // below cannot detect it because it is relative to this same scale. This
   // happens on coarse levels of operators with no left near-nullspace at all
   // - pure advection with outflow BCs is the model case, where relaxing on
   // A^T x = 0 converges to the zero vector - and dividing by ||B_c|S_i||^2
   // then produces Inf/NaN in Z. The vectors start as the constant (all ones)
   // before smoothing, so an absolute floor is meaningful here.
   if (!(b_c_scale > (PetscReal)PFLARE_TOL_CONSTRAIN_ABS)) {
      PetscCall(MatAssemblyBegin(new_z_or_w, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(new_z_or_w, MAT_FINAL_ASSEMBLY));
      PetscCall(PetscFree(b_c_nonlocal_alloc));
      PetscCall(MatDestroy(z_or_w));
      if (is_z) {
         PetscCall(MatDestroy(&row_mat));
         PetscCall(MatTranspose(new_z_or_w, MAT_INITIAL_MATRIX, &row_mat));
         *z_or_w = row_mat;
         PetscCall(MatDestroy(&new_z_or_w));
      } else {
         *z_or_w = new_z_or_w;
      }
      if (comm_size != 1 && !is_mpiaij) {
         PetscCall(MatDestroy(&temp_mat_aij));
      }
      PetscFunctionReturn(PETSC_SUCCESS);
   }

   // ~~~~~
   // Copy out the local values of B_c and B_f once, rather than getting the
   // arrays in every row (which on the gpu would copy to the host every row)
   // ~~~~~
   PetscCall(VecGetLocalSize(null_vecs_c[0], &local_size_c));
   PetscCall(VecGetLocalSize(null_vecs_f[0], &local_size_f));
   PetscCall(PetscMalloc1(local_size_c * null_vecs_size, &b_c_local_alloc));
   PetscCall(PetscMalloc1(local_size_f * null_vecs_size, &b_f_local_alloc));
   for (null_vec = 0; null_vec < null_vecs_size; null_vec++) {
      PetscCall(VecGetArrayRead(null_vecs_c[null_vec], &b_c_local));
      for (j_loc = 0; j_loc < local_size_c; j_loc++) b_c_local_alloc[j_loc + null_vec * local_size_c] = b_c_local[j_loc];
      PetscCall(VecRestoreArrayRead(null_vecs_c[null_vec], &b_c_local));
      PetscCall(VecGetArrayRead(null_vecs_f[null_vec], &b_f_vals));
      for (j_loc = 0; j_loc < local_size_f; j_loc++) b_f_local_alloc[j_loc + null_vec * local_size_f] = b_f_vals[j_loc];
      PetscCall(VecRestoreArrayRead(null_vecs_f[null_vec], &b_f_vals));
   }

   // Now go through each of the rows
   // GetRow has to happen over the global indices
   for (i_loc = global_row_start; i_loc < global_row_end_plus_one; i_loc++) {

      // Get how many non-zero columsn we have for this row
      // Convenience given we are going to interact with Ad and Ao separately
      PetscCall(MatGetRow(row_mat, i_loc, &ncols_temp,
               &cols, NULL));
      ncols = ncols_temp;
      // We're done with this row
      PetscCall(MatRestoreRow(row_mat, i_loc, &ncols_temp,
         &cols, NULL));

      // Skip this row if there are no non-zeros
      if (ncols == 0) continue;

      // This is the rhs
      PetscCall(PetscMalloc1(ncols, &row_vals));
      PetscCall(PetscMalloc1(ncols * null_vecs_size, &b_c_vals));
      PetscCall(PetscMalloc1(null_vecs_size * null_vecs_size, &bctbc));
      PetscCall(PetscMalloc1(null_vecs_size * null_vecs_size, &pseudo));
      PetscCall(PetscMalloc1(null_vecs_size * ncols, &temp_mat));

      PetscCall(PetscMalloc1(null_vecs_size, &diff));
      // This stores the global indices, with local indices first then global
      PetscCall(PetscMalloc1(ncols, &col_indices_off_proc_array));

      // ~~~~~~~~~~~
      // Let's pull out existing entries in W and the coarse nullspace components
      // We have the local values first then the nonlocal
      // This is so the ordering is easy with colmap and b_c_nonlocal_alloc above
      // ~~~~~~~~~~~
      // We do the local values first - remember Ad and Ao need sequential indices
      PetscCall(MatGetRow(Ad, i_loc - global_row_start, &ncols_temp,
               &cols, &vals));
      ncols_ad = ncols_temp;

      // this is the local values of W
      for (j_loc = 0; j_loc < ncols_ad; j_loc++) row_vals[j_loc] = vals[j_loc];
      // Global indices of our local cols - be careful here to use col_start not row_start
      for (j_loc = 0; j_loc < ncols_ad; j_loc++) col_indices_off_proc_array[j_loc] = cols[j_loc] + global_col_start;

      // Get the local b_c values
      // Loop over all the near nullspace vectors and get the local components
      for (null_vec = 0; null_vec < null_vecs_size; null_vec++) {
         // cols are local indices
         for (j_loc = 0; j_loc < ncols_ad; j_loc++) b_c_vals[j_loc + null_vec * ncols] = b_c_local_alloc[cols[j_loc] + null_vec * local_size_c];
      }
      PetscCall(MatRestoreRow(Ad, i_loc - global_row_start, &ncols_temp,
               &cols, &vals));

      // ~~~~~~~~~
      // Nonlocal if needed
      if (comm_size != 1) {

         // Now the nonlocal
         PetscCall(MatGetRow(Ao, i_loc - global_row_start, &ncols_temp,
                  &cols, &vals));

         ncols_ao = ncols_temp;
         // this is the nonlocal values of W
         // after the local values in row_vals
         for (j_loc = 0; j_loc < ncols_ao; j_loc++) row_vals[ncols_ad + j_loc] = vals[j_loc];
         // Global indices of our nonlocal cols
         for (j_loc = 0; j_loc < ncols_ao; j_loc++) col_indices_off_proc_array[ncols_ad + j_loc] = colmap[cols[j_loc]];

         // Get the nonlocal b_c values
         // Remembering cols are the local indices into colmap (or equivalently the first dim of b_c_nonlocal_alloc)
         for (null_vec = 0; null_vec < null_vecs_size; null_vec++) {
            for (j_loc = 0; j_loc < ncols_ao; j_loc++) {
               b_c_vals[ncols_ad + j_loc + null_vec * ncols] = b_c_nonlocal_alloc[cols[j_loc] + null_vec * cols_ao];
            }
         }

         PetscCall(MatRestoreRow(Ao, i_loc - global_row_start, &ncols_temp,
                  &cols, &vals));

      }

      // ~~~~~~~~~
      // Skip rows whose near-nullspace block is numerically zero relative to
      // the global scale of B_c - see the comment above the row loop. The row
      // keeps the values already copied into new_z_or_w by MatDuplicate.
      // ~~~~~~~~~
      // Largest near-nullspace entry seen anywhere in this row's pattern
      // (maxval(abs(b_c_vals)), which skips NaNs as gfortran's maxval does and
      // is NaN only if every entry is)
      b_c_row_norm = PetscAbsScalar(b_c_vals[0]);
      for (j_loc = 1; j_loc < ncols * null_vecs_size; j_loc++) {
         if (PetscIsNanReal(b_c_row_norm) || PetscAbsScalar(b_c_vals[j_loc]) > b_c_row_norm) b_c_row_norm = PetscAbsScalar(b_c_vals[j_loc]);
      }
      if (b_c_row_norm <= (PetscReal)PFLARE_TOL_CONSTRAIN_REL * b_c_scale) {
         PetscCall(PetscFree(row_vals));
         PetscCall(PetscFree(col_indices_off_proc_array));
         PetscCall(PetscFree(b_c_vals));
         PetscCall(PetscFree(diff));
         PetscCall(PetscFree(bctbc));
         PetscCall(PetscFree(pseudo));
         PetscCall(PetscFree(temp_mat));
         continue;
      }

      // ~~~~~~~~~
      // Now we pull out the fine near nullspace component
      // We are then going to compute the residual for this row
      // ie W B_c^R - B_f^R
      // ~~~~~~~~~
      // W B_c^R can be done as (B_c^R)^T W
      one_bl = 1;
      blas_one = 1.0;
      blas_zero = 0.0;
      blas_minus_one = -1.0;
      PetscCall(PetscBLASIntCast(ncols, &m_bl));
      PetscCall(PetscBLASIntCast(null_vecs_size, &n_bl));
      PetscCall(PetscBLASIntCast(ncols, &lda_bl));
      PetscCallBLAS("BLASgemv", BLASgemv_("T", &m_bl, &n_bl,
               &blas_one, b_c_vals, &lda_bl,
               row_vals, &one_bl,
               &blas_zero, diff, &one_bl));

      // Compute W * B_c^R - B_f^R
      // Loop over near-nullspace vecs
      for (null_vec = 0; null_vec < null_vecs_size; null_vec++) {
         // b_f is local (we want b_f for this row) but so is the value we want
         diff[null_vec] = diff[null_vec] - b_f_local_alloc[i_loc - global_row_start + null_vec * local_size_f];
      }

      // Compute (B_c^R)^T * B_c^R
      PetscCall(PetscBLASIntCast(null_vecs_size, &n_bl));
      PetscCall(PetscBLASIntCast(ncols, &k_bl));
      PetscCall(PetscBLASIntCast(ncols, &lda_bl));
      PetscCall(PetscBLASIntCast(null_vecs_size, &ldc_bl));
      PetscCallBLAS("BLASgemm", BLASgemm_("T", "N", &n_bl, &n_bl, &k_bl,
               &blas_one, b_c_vals, &lda_bl,
               b_c_vals, &lda_bl,
               &blas_zero, bctbc, &ldc_bl));

      // Compute the pseudo inverse of (B_c^R)^T * B_c^R
      PetscCall(pseudo_inv(bctbc, null_vecs_size, null_vecs_size, pseudo));

      // Compute inv((B_c^R)^T * B_c^R) * (B_c^R)^T
      PetscCall(PetscBLASIntCast(null_vecs_size, &m_bl));
      PetscCall(PetscBLASIntCast(ncols, &n_bl));
      PetscCall(PetscBLASIntCast(null_vecs_size, &k_bl));
      PetscCall(PetscBLASIntCast(null_vecs_size, &lda_bl));
      PetscCall(PetscBLASIntCast(ncols, &ldb_bl));
      PetscCall(PetscBLASIntCast(null_vecs_size, &ldc_bl));
      PetscCallBLAS("BLASgemm", BLASgemm_("N", "T", &m_bl, &n_bl, &k_bl,
               &blas_one, pseudo, &lda_bl,
               b_c_vals, &ldb_bl,
               &blas_zero, temp_mat, &ldc_bl));

      // Now compute -(W * B_c^R - B_f^R ) * inv((B_c^R)^T * B_c^R) * (B_c^R)^T
      // Again we're doing the left vec mat mult with a transpose
      PetscCall(PetscBLASIntCast(null_vecs_size, &m_bl));
      PetscCall(PetscBLASIntCast(ncols, &n_bl));
      PetscCall(PetscBLASIntCast(null_vecs_size, &lda_bl));
      PetscCallBLAS("BLASgemv", BLASgemv_("T", &m_bl, &n_bl,
               &blas_minus_one, temp_mat, &lda_bl,
               diff, &one_bl,
               &blas_zero, b_c_vals, &one_bl));

      // ~~~~~~~~~~~~~
      // Belt and braces: never write a non-finite update into Z or W. The
      // tests above bound ||B_c|S_i|| from below relative to the vectors'
      // own scale, but a pathological B could still make the projection
      // overflow; leaving the row unconstrained is always safe.
      // ~~~~~~~~~~~~~
      has_nan = PETSC_FALSE;
      for (j_loc = 0; j_loc < ncols; j_loc++) {
         if (PetscIsNanScalar(b_c_vals[j_loc])) has_nan = PETSC_TRUE;
      }
      if (has_nan) {
         PetscCall(PetscFree(row_vals));
         PetscCall(PetscFree(col_indices_off_proc_array));
         PetscCall(PetscFree(b_c_vals));
         PetscCall(PetscFree(diff));
         PetscCall(PetscFree(bctbc));
         PetscCall(PetscFree(pseudo));
         PetscCall(PetscFree(temp_mat));
         continue;
      }

      // ~~~~~~~~~~~~~
      // Set all the row values, same sparsity pattern
      // ~~~~~~~~~~~~~
      PetscCall(MatSetValues(new_z_or_w, one, &i_loc, ncols, col_indices_off_proc_array, b_c_vals, ADD_VALUES));
      PetscCall(PetscFree(row_vals));
      PetscCall(PetscFree(col_indices_off_proc_array));
      PetscCall(PetscFree(b_c_vals));
      PetscCall(PetscFree(diff));
      PetscCall(PetscFree(bctbc));
      PetscCall(PetscFree(pseudo));
      PetscCall(PetscFree(temp_mat));

   }

   PetscCall(MatAssemblyBegin(new_z_or_w, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(new_z_or_w, MAT_FINAL_ASSEMBLY));

   PetscCall(PetscFree(b_c_nonlocal_alloc));
   PetscCall(PetscFree(b_c_local_alloc));
   PetscCall(PetscFree(b_f_local_alloc));

   // Destroy the old input matrix
   PetscCall(MatDestroy(z_or_w));

   if (is_z) {
      // Destroy the transposed original matrix
      PetscCall(MatDestroy(&row_mat));
      // Now go and transpose the new one back to the original sizes
      PetscCall(MatTranspose(new_z_or_w, MAT_INITIAL_MATRIX, &row_mat));
      // And set the output to the newly constrained mat
      *z_or_w = row_mat;
      PetscCall(MatDestroy(&new_z_or_w));

   } else {
      // And set the output to the newly constrained mat
      *z_or_w = new_z_or_w;
   }

   if (comm_size != 1 && !is_mpiaij) {
      PetscCall(MatDestroy(&temp_mat_aij));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------
