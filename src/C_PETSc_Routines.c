/*
  C routines that are used to get around a lack of some PETSc fortran interfaces
*/

// Include the petsc header files
#include <petsc.h>
#include <petsc/private/pcimpl.h>

// MatPartitioningSetNParts doesn't have a fortran interface (and can't pass around
// a matpartioning object as it doesnt have a %v), so have to do 
// all the partitioning in C
// We have modified this from PCGAMGCreateLevel_GAMG, and we enforce that 
// new_size <= comm_world/proc_stride
// This always does an interleaved partitioning
// PCGAMGCreateLevel_GAMG can modify new_size, but we don't allow that, we always partition
// onto new_size active ranks and we enforce the interleaving happens nested to reduce comms
PETSC_INTERN void MatPartitioning_c(Mat *adj, PetscInt new_size, PetscInt *proc_stride, IS *index)
{
   // This is all taken from PCGAMGCreateLevel_GAMG in gamg.c
   MPI_Comm comm;
   MatPartitioning mpart;
   IS proc_is, proc_is_eq_num;
   PetscInt comm_size_petsc;
   int size, rank, errorcode;
   // If you want to use the improve, then new_size needs to equal the current size
   // ie you can't reduce the number of active ranks
   int improve = 0;
   double ratio;

   // Get the comm
   PetscCallVoid(PetscObjectGetComm((PetscObject)*adj, &comm));
   PetscCallMPIAbort(comm, MPI_Comm_size(comm, &size));
   comm_size_petsc = (PetscInt)size;
   PetscInt* counts;
   PetscCallVoid(PetscMalloc1(comm_size_petsc, &counts));
   PetscCallMPIAbort(comm, MPI_Comm_rank(comm, &rank));

   // Check that new_size <= comm_world/proc_stride
   ratio = ((double)size)/((PetscReal)*proc_stride);
   if (new_size > ratio) 
   {
      errorcode = 1;
      PetscCallMPIAbort(comm, MPI_Abort(comm, errorcode));
   }

   PetscInt ncrs_eq_local, ncrs_eq_global;
   PetscCallVoid(MatGetLocalSize(*adj, &ncrs_eq_local, NULL));
   PetscCallVoid(MatGetSize(*adj, &ncrs_eq_global, NULL));
   const PetscInt *is_idx;
   PetscInt* newproc_idx;
   PetscCallVoid(PetscMalloc1(ncrs_eq_local, &newproc_idx));

   // Construct an IS with the current partition in it
   if (improve)
   {
      for (PetscInt i = 0; i < ncrs_eq_local; i++)
      {
         newproc_idx[i] = rank;
      }
      PetscCallVoid(ISCreateGeneral(comm, ncrs_eq_local, newproc_idx, PETSC_COPY_VALUES, &proc_is));
   }

   // Now we want to interleave the processors as it reduces the amount of out of node comms
   PetscInt expand_factor=1, kk;   

   // If you're doing the improve you can't reduce the number of active ranks
   if (!improve)
   {
      // GAMG
      // PetscInt rfactor=1, jj, fact;
      // /* find factor */
      // // This is the closest match 
      // // I've modified this to have rfactor 1
      // if (new_size == 1) rfactor = 1;
      // else {
      //    PetscReal best_fact = 0.;
      //    jj                  = -1;
      //    for (kk = 1; kk <= size; kk++) {
      //       if (!(size % kk)) { /* a candidate */
      //          PetscReal nactpe = (PetscReal)size / (PetscReal)kk, fact = nactpe / (PetscReal)new_size;
      //          if (fact > 1.0) fact = 1. / fact; /* keep fact < 1 */
      //          if (fact > best_fact) {
      //          best_fact = fact;
      //          jj        = kk;
      //          }
      //       }
      //    }
      //    if (jj != -1) rfactor = jj;
      //    else rfactor = 1; /* a prime */

      //    fprintf(stderr,"rfactor %d", rfactor);

      //    rfactor = floor((PetscReal)size/(PetscReal)new_size);
         
      //    // If you want continuous instead of interleaved, just make expand_factor 1
      //    //expand_factor = 1;
      //    expand_factor = rfactor;
      // }

      /* Now we have modified this from GAMG
         proc_stride is passed in from outside and keeps track of the stride changes
         it is just *= processor_agglom_factor after each repartitioning
         as we coarsen on lower levels, e.g., with size = 11, new_size = 5, proc_stride = 2
         active ranks: 0 1 2 3 4 5 6 7 8 9 10
         active ranks: 0   2   4   6   8             
         then new_size = 2, proc_stride becomes 4
         active ranks: 0       4
      */

      // If you want continuous instead of interleaved, just make expand_factor 1
      //expand_factor = 1;     
      expand_factor = *proc_stride;

   }   

   // Create the partitioning - this should use parmetis by default
   // but you can change via the command line options
   PetscCallVoid(MatPartitioningCreate(comm, &mpart));
   PetscCallVoid(MatPartitioningSetAdjacency(mpart, *adj));
   // Uses parmetis by default but you can change via the command line options
#if defined(PETSC_HAVE_PARMETIS)
   PetscCallVoid(MatPartitioningSetType(mpart, MATPARTITIONINGPARMETIS));
#else
   // ParMETIS is not available (e.g. PETSc built without MPI); fall back to a
   // built-in partitioner that needs no external package. This path is only
   // reached in parallel, so it never runs in a serial (MPIUNI) build anyway
   PetscCallVoid(MatPartitioningSetType(mpart, MATPARTITIONINGAVERAGE));
#endif
   PetscCallVoid(MatPartitioningSetFromOptions(mpart));
   PetscCallVoid(MatPartitioningSetNParts(mpart, new_size));

   // This returns an IS with the new rank of each equation 
   if (improve)
   {  // Either improve the existing partition
      PetscCallVoid(MatPartitioningImprove(mpart, &proc_is));
   }
   else
   {
      // Or compute a brand new one
      PetscCallVoid(MatPartitioningApply(mpart, &proc_is));
   }
   PetscCallVoid(MatPartitioningDestroy(&mpart));

   // If you're doing the improve you can't reduce the number of active ranks
   if (!improve)
   {
      PetscCallVoid(ISGetIndices(proc_is, &is_idx));
      // Modified as we never have block equations
      for (kk = 0; kk < ncrs_eq_local; kk++) {
         newproc_idx[kk] = is_idx[kk] * expand_factor; /* distribution */
      }
      PetscCallVoid(ISRestoreIndices(proc_is, &is_idx));
      PetscCallVoid(ISDestroy(&proc_is));
      PetscCallVoid(ISCreateGeneral(comm, ncrs_eq_local, newproc_idx, PETSC_COPY_VALUES, &proc_is));
   }

   // Get how many eqs have been partitioned onto each rank
   PetscCallVoid(ISPartitioningCount(proc_is, comm_size_petsc, counts));

   // This turns the rank into a new global numbering
   // ie the number for each local variable in this is the new global numbering
   PetscCallVoid(ISPartitioningToNumbering(proc_is, &proc_is_eq_num));
   PetscCallVoid(ISDestroy(&proc_is));

   // This comms the new global numbering to the rank it belongs onto
   // so we can us new_eq_indices (an IS) in a call to matgetsubmatrix
   // The petsc doc says this is expensive for greater than 10M indices....
   PetscCallVoid(ISInvertPermutation(proc_is_eq_num, counts[rank], index));
   PetscCallVoid(ISDestroy(&proc_is_eq_num));

   PetscCallVoid(PetscFree(counts));
   PetscCallVoid(PetscFree(newproc_idx));
   return;
}

// Create data for the process agglomeration
PETSC_INTERN void GenerateIS_ProcAgglomeration_c(PetscInt proc_stride, PetscInt global_size, PetscInt *local_size_reduced, PetscInt *start)
{

   PetscSubcomm psubcomm = NULL;
   MPI_Comm subcomm = MPI_COMM_NULL;

   // Taken from telescope.c Line 515, we're creating a subcomm
   // letting petsc decide on which ranks are interlaced
   PetscCallVoid(PetscSubcommCreate(MPI_COMM_WORLD,&psubcomm));
   PetscCallVoid(PetscSubcommSetNumber(psubcomm,proc_stride));
   // The petsc subcomm splits into proc_stride groups
   PetscCallVoid(PetscSubcommSetType(psubcomm,PETSC_SUBCOMM_INTERLACED));
   // e.g., if noprocs = 5, proc_stride = 2
   // then we have two subcomms, one with 3 procs and one with 2
   // the one with 3 corresponds to the first color, etc
   subcomm = PetscSubcommChild(psubcomm);

   Vec xred;
   PetscInt ed;

   // Taken from PCTelescopeSetUp_default
   // Even split across the cores in the first grouping 
   // of the subcomm
   if (psubcomm->color == 0)
   {
      // This does the split for us on the reduced number of cores
      PetscCallVoid(VecCreateMPI(subcomm, PETSC_DECIDE, global_size, &xred));
      PetscCallVoid(VecGetOwnershipRange(xred, start, &ed));
      *local_size_reduced = ed - *start;
   }
   // zero entries
   else
   {
      *local_size_reduced = 0;
      *start = 0;
   }

   if (psubcomm->color == 0)
   {
      PetscCallVoid(VecDestroy(&xred));
   }
   PetscCallVoid(PetscSubcommDestroy(&psubcomm));
   return;
}

// Takes a Mat and returns a Mat on a subcomm without ranks with no rows
// If there are no empty ranks, or the matrix is not MPIAIJ, then it returns the same matrix
// If not there is a new copy that must be destroyed
// Basically taken from MatMPIAdjCreateNonemptySubcommMat_MPIAdj
PETSC_INTERN PetscErrorCode MatMPICreateNonemptySubcomm_c(Mat *A, int *on_subcomm, Mat *B)
{
  const PetscInt *ranges;
  MPI_Comm        acomm, bcomm, pcomm;
  MPI_Group       agroup, bgroup;
  PetscMPIInt     i, size, nranks, *ranks;
  Mat mat_local, mat_nonlocal;
  const PetscInt *colmap_input;
  PetscBool       is_mpiaij;

  PetscFunctionBegin;

  *on_subcomm = 1;

  // Ensure we return petsc_null so we can test if we are on this subcomm
  // outside the function
  *B = PETSC_NULLPTR;
  PetscCall(PetscObjectGetComm((PetscObject)(*A), &acomm));
  (void)MPI_Comm_size(acomm, &size);
  PetscCall(MatGetOwnershipRanges((*A), &ranges));
  for (i = 0, nranks = 0; i < size; i++) {
    if (ranges[i + 1] - ranges[i] > 0) nranks++;
  }
  // We can only build the subcomm copy from MPIAIJ matrices (this includes mpiaijkokkos)
  // For any other type (e.g., a MATDIAGONAL Aff) just do the reductions on the original comm
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)(*A), MATMPIAIJ, &is_mpiaij));
  if (nranks == size || !is_mpiaij) { /* All ranks have a positive number of rows, so we do not need to create a subcomm; */
    PetscCall(PetscObjectReference((PetscObject)(*A)));
    *B = (*A);
    // We haven't actually ended up on a subcomm
    *on_subcomm = 0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(MatMPIAIJGetSeqAIJ(*A, &mat_local, &mat_nonlocal, &colmap_input));
  PetscCall(PetscMalloc1(nranks, &ranks));
  for (i = 0, nranks = 0; i < size; i++) {
    if (ranges[i + 1] - ranges[i] > 0) ranks[nranks++] = i;
  }
  (void)MPI_Comm_group(acomm, &agroup);
  (void)MPI_Group_incl(agroup, nranks, ranks, &bgroup);
  PetscCall(PetscFree(ranks));
  (void)MPI_Comm_create(acomm, bgroup, &bcomm);
  (void)MPI_Group_free(&agroup);
  (void)MPI_Group_free(&bgroup);
  if (bcomm != MPI_COMM_NULL) {
    PetscInt    m, n, M, N;
    Mat Ad_copy, Ao_copy; 
    PetscCall(MatGetLocalSize((*A), &m, &n));
    PetscCall(MatGetSize((*A), &M, &N));

    // If we've gone down to a serial matrix
    if (nranks == 1) 
    {
      // Just send out a copy of the local part of the input matrix
      PetscCall(MatDuplicate(mat_local, MAT_COPY_VALUES, B));
      (void)MPI_Comm_free(&bcomm);
    }
    else{
      // Copy the local and off-diagonal sequential matrices
      PetscCall(MatDuplicate(mat_local, MAT_COPY_VALUES, &Ad_copy));
      PetscCall(MatDuplicate(mat_nonlocal, MAT_COPY_VALUES, &Ao_copy));

      PetscInt *garray_host = NULL; 
      PetscInt rows_ao, cols_ao;
      PetscCall(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao));
      PetscCall(PetscMalloc1(cols_ao, &garray_host));
      for (PetscInt i = 0; i < cols_ao; i++)
      {
         garray_host[i] = colmap_input[i];
      }      

      // MAT_NO_OFF_PROC_ENTRIES is set to true in this routine so 
      // don't need to set it externally
      // Have to be careful here as need to feed in copies of A and B
      // We can't just create B on bcomm and then free bcomm, as the layouts
      // of B store the comm they are given without duplicating it
      // Instead create B on the PETSc inner comm of bcomm, which B then holds
      // a reference to (and hence is freed when B is destroyed)
      PetscCall(PetscCommDuplicate(bcomm, &pcomm, NULL));
      PetscCall(MatCreateMPIAIJWithSeqAIJ(pcomm, M, N, Ad_copy, Ao_copy, garray_host, B));
      // Drop our reference to the inner comm and free bcomm, otherwise bcomm leaks
      PetscCall(PetscCommDestroy(&pcomm));
      (void)MPI_Comm_free(&bcomm);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Returns pc->flag so we can have access to the structure flag
// for re-use
PETSC_INTERN void c_PCGetStructureFlag(PC *pc, int *flag)
{
   *flag = (*pc)->flag;
}

// Returns pc->setupcalled
PETSC_INTERN void PCGetSetupCalled_c(PC *pc, PetscInt *setupcalled)
{
   *setupcalled = (*pc)->setupcalled;
}

// Mark that we need to do setup again (e.g. after a reset)
PETSC_INTERN void PCMarkNotSetUp_c(PC *pc)
{
   (*pc)->setupcalled = PETSC_FALSE;
}

PETSC_EXTERN PetscErrorCode MatGetDiagonalMarkers_SeqAIJ(Mat, const PetscInt **, PetscBool *);
// Returns true if a matrix is only the diagonal
PETSC_INTERN PetscErrorCode MatGetDiagonalOnly_c(Mat *A, int *diag_only)
{
  MPI_Comm        acomm;
  PetscMPIInt     size;
  PetscInt local_rows, local_cols;
  PetscInt rank_diag = 0, rank_diag_serial = 0;

  PetscFunctionBegin;

  PetscCall(PetscObjectGetComm((PetscObject)(*A), &acomm));
  PetscCallMPI(MPI_Comm_size(acomm, &size));

  Mat mat_local = NULL, mat_nonlocal = NULL; 

  // Get the existing output mats
  if (size != 1)
  {
     PetscCall(MatMPIAIJGetSeqAIJ(*A, &mat_local, &mat_nonlocal, NULL));
  }
  else
  {
     mat_local = *A;
  } 

  PetscCall(MatGetLocalSize(*A, &local_rows, &local_cols));
  *diag_only = 0;

  // Use MatGetRowIJ to read the SeqAIJ nnz count without touching the private struct.
  PetscInt local_nnz = 0, nonlocal_nnz = 0;
  {
     PetscInt n;
     PetscBool symmetric = PETSC_FALSE, inodecompressed = PETSC_FALSE, done;
     const PetscInt *ad_ia;
     PetscCall(MatGetRowIJ(mat_local, 0, symmetric, inodecompressed, &n, &ad_ia, NULL, &done));
     local_nnz = ad_ia[n];
     PetscCall(MatRestoreRowIJ(mat_local, 0, symmetric, inodecompressed, &n, &ad_ia, NULL, &done));
  }

  if (size != 1)
  {
      PetscBool diagDense;

      {
         PetscInt n;
         PetscBool symmetric = PETSC_FALSE, inodecompressed = PETSC_FALSE, done;
         const PetscInt *ao_ia;
         PetscCall(MatGetRowIJ(mat_nonlocal, 0, symmetric, inodecompressed, &n, &ao_ia, NULL, &done));
         nonlocal_nnz = ao_ia[n];
         PetscCall(MatRestoreRowIJ(mat_nonlocal, 0, symmetric, inodecompressed, &n, &ao_ia, NULL, &done));
      }

      // In parallel also have to check the nonlocal has nothing in it
      PetscCall(MatGetDiagonalMarkers_SeqAIJ(mat_local, NULL, &diagDense));
      if (diagDense && local_rows == local_nnz && nonlocal_nnz == 0)
      {
         rank_diag_serial++;
      }

      // Reduction to check every rank
      PetscCallMPI(MPI_Allreduce(&rank_diag_serial, &rank_diag, 1, MPIU_INT, MPI_SUM, acomm));
  }
  else
  {
      PetscBool diagDense;
      PetscCall(MatGetDiagonalMarkers_SeqAIJ(*A, NULL, &diagDense));
      // In serial easy
      if (diagDense && local_rows == local_nnz)
      {
         rank_diag++;
      }
  }

  // If every rank is diagonal only, the entire matrix is diagonal
  if (rank_diag == size)
  {
     *diag_only = 1;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Local triangular solve coarse-grid solver
// A fixed number of Richardson iterations on A x = b preconditioned by an ILU(0) factorisation
// of the local (on-rank) diagonal block of A, ie block Jacobi with ILU(0) on each block
// If the local block is lower triangular (eg upwind advection in a downstream ordering, which
// AIR coarse grids inherit from the top grid) then ILU(0) is exact and the preconditioner
// is a local forward Gauss-Seidel sweep
// This is a fixed linear operator (zero initial guess, no norms), so it can be used as the
// coarse grid solver in a V-cycle and tested by the auto truncation like the polynomials
// The triangular solves either use the default factorisation for the matrix type (Kokkos-Kernels
// SpTRSV for aijkokkos) or hipSPARSE on a hipSPARSE copy of the local block
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

typedef struct {
   Mat       A;          // the operator (we hold a reference)
   Mat       A_local;    // hipSPARSE copy of the local block if used
   Mat       factor;     // ILU(0) factor of the local block
   Vec       r, w;       // parallel work vectors
   Vec       b_local, x_local; // sequential wrappers of the local parts
   PetscInt  its;
   PetscBool hipsparse;
} LocalTriCtx;

static PetscErrorCode LocalTriCtxDestroy(void *ctx_ptr)
{
   LocalTriCtx *ctx = *(LocalTriCtx **)ctx_ptr;

   PetscFunctionBegin;
   PetscCall(MatDestroy(&ctx->A));
   PetscCall(MatDestroy(&ctx->A_local));
   PetscCall(MatDestroy(&ctx->factor));
   PetscCall(VecDestroy(&ctx->r));
   PetscCall(VecDestroy(&ctx->w));
   PetscCall(VecDestroy(&ctx->b_local));
   PetscCall(VecDestroy(&ctx->x_local));
   PetscCall(PetscFree(ctx));
   *(LocalTriCtx **)ctx_ptr = NULL;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// y = M^-1 x with M the block diagonal ILU(0)
static PetscErrorCode LocalTriApplyPC(LocalTriCtx *ctx, Vec x, Vec y)
{
   PetscFunctionBegin;
   if (ctx->hipsparse)
   {
      // Wrap the device pointers of the (kokkos) vectors in hip vectors so the
      // hipSPARSE solve runs on the device with no copies
      // Kokkos in PETSc runs on PETSc's default hip stream so the solve is stream ordered
      const PetscScalar *x_array;
      PetscScalar *y_array;
      PetscMemType mtype_x, mtype_y;
      PetscCall(VecGetArrayReadAndMemType(x, &x_array, &mtype_x));
      PetscCall(VecGetArrayWriteAndMemType(y, &y_array, &mtype_y));
      PetscCheck(PetscMemTypeDevice(mtype_x) && PetscMemTypeDevice(mtype_y), PETSC_COMM_SELF, PETSC_ERR_SUP, "hipSPARSE local triangular solve needs device vectors");
      PetscCall(VecHIPPlaceArray(ctx->b_local, x_array));
      PetscCall(VecHIPPlaceArray(ctx->x_local, y_array));
      PetscCall(MatSolve(ctx->factor, ctx->b_local, ctx->x_local));
      PetscCall(VecHIPResetArray(ctx->b_local));
      PetscCall(VecHIPResetArray(ctx->x_local));
      PetscCall(VecRestoreArrayWriteAndMemType(y, &y_array));
      PetscCall(VecRestoreArrayReadAndMemType(x, &x_array));
   }
   else
   {
      PetscCall(VecGetLocalVectorRead(x, ctx->b_local));
      PetscCall(VecGetLocalVector(y, ctx->x_local));
      PetscCall(MatSolve(ctx->factor, ctx->b_local, ctx->x_local));
      PetscCall(VecRestoreLocalVectorRead(x, ctx->b_local));
      PetscCall(VecRestoreLocalVector(y, ctx->x_local));
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LocalTri(Mat mat, Vec b, Vec x)
{
   LocalTriCtx *ctx;

   PetscFunctionBegin;
   PetscCall(PetscObjectContainerQuery((PetscObject)mat, "pflare_local_tri", (PetscCtxRt)&ctx));
   // Zero initial guess, so the first iteration is just x = M^-1 b
   PetscCall(LocalTriApplyPC(ctx, b, x));
   for (PetscInt i = 1; i < ctx->its; i++)
   {
      // r = b - A x
      PetscCall(MatMult(ctx->A, x, ctx->r));
      PetscCall(VecAYPX(ctx->r, -1.0, b));
      // x = x + M^-1 r
      PetscCall(LocalTriApplyPC(ctx, ctx->r, ctx->w));
      PetscCall(VecAXPY(x, 1.0, ctx->w));
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

// Creates a matshell whose MatMult applies its iterations of the local triangular solver
// The shell context is left null (the Fortran matshell routines check for that), the solver
// data is composed on the matshell and destroyed with it
PETSC_INTERN PetscErrorCode create_local_tri_inverse_c(Mat *A, PetscInt its, int hipsparse, Mat *inv)
{
   MPI_Comm      comm;
   LocalTriCtx  *ctx;
   Mat           A_diag, A_fact;
   IS            row_perm, col_perm;
   MatFactorInfo info;
   PetscInt      local_rows, local_cols, global_rows, global_cols;

   PetscFunctionBegin;
   PetscCheck(its > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Need at least one iteration");
   PetscCall(PetscObjectGetComm((PetscObject)(*A), &comm));
   PetscCall(PetscNew(&ctx));
   ctx->its       = its;
   ctx->hipsparse = hipsparse ? PETSC_TRUE : PETSC_FALSE;
   PetscCall(PetscObjectReference((PetscObject)(*A)));
   ctx->A = *A;

   // The local block (the matrix itself in serial)
   PetscCall(MatGetDiagonalBlock(*A, &A_diag));
   A_fact = A_diag;
   {
      // ILU(0) needs every diagonal entry present
      PetscBool diag_dense;
      PetscCall(MatGetDiagonalMarkers_SeqAIJ(A_diag, NULL, &diag_dense));
      PetscCheck(diag_dense, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Local block is missing diagonal entries");
   }
   if (ctx->hipsparse)
   {
      // Build a plain seqaij copy of the local block and convert that in place, rather
      // than converting the (eg kokkos) block directly, which may not have a converter
      const PetscInt    *ia, *ja;
      const PetscScalar *aa;
      PetscInt           n;
      PetscBool          done;
      PetscCall(MatGetRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, &ja, &done));
      PetscCheck(done, PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot get the local block CSR");
      PetscCall(MatSeqAIJGetArrayRead(A_diag, &aa));
      PetscCall(MatCreate(PETSC_COMM_SELF, &ctx->A_local));
      PetscCall(MatSetSizes(ctx->A_local, n, n, n, n));
      PetscCall(MatSetType(ctx->A_local, MATSEQAIJ));
      PetscCall(MatSeqAIJSetPreallocationCSR(ctx->A_local, ia, ja, aa));
      PetscCall(MatSeqAIJRestoreArrayRead(A_diag, &aa));
      PetscCall(MatRestoreRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, &ja, &done));
      PetscCall(MatConvert(ctx->A_local, MATSEQAIJHIPSPARSE, MAT_INPLACE_MATRIX, &ctx->A_local));
      A_fact = ctx->A_local;
   }

   // ILU(0) in the natural ordering
   PetscCall(MatGetFactor(A_fact, ctx->hipsparse ? MATSOLVERHIPSPARSE : NULL, MAT_FACTOR_ILU, &ctx->factor));
   PetscCall(MatGetOrdering(A_fact, MATORDERINGNATURAL, &row_perm, &col_perm));
   PetscCall(MatFactorInfoInitialize(&info));
   info.levels = 0;
   info.fill   = 1.0;
   PetscCall(MatILUFactorSymbolic(ctx->factor, A_fact, row_perm, col_perm, &info));
   PetscCall(MatLUFactorNumeric(ctx->factor, A_fact, &info));
   PetscCall(ISDestroy(&row_perm));
   PetscCall(ISDestroy(&col_perm));

   // Work vectors
   PetscCall(MatCreateVecs(*A, &ctx->r, &ctx->w));
   PetscCall(MatGetLocalSize(*A, &local_rows, &local_cols));
   if (ctx->hipsparse)
   {
      PetscCall(VecCreateSeqHIPWithArrays(PETSC_COMM_SELF, 1, local_rows, NULL, NULL, &ctx->b_local));
      PetscCall(VecCreateSeqHIPWithArrays(PETSC_COMM_SELF, 1, local_rows, NULL, NULL, &ctx->x_local));
   }
   else
   {
      PetscCall(VecCreateLocalVector(ctx->r, &ctx->b_local));
      PetscCall(VecCreateLocalVector(ctx->r, &ctx->x_local));
   }

   PetscCall(MatGetSize(*A, &global_rows, &global_cols));
   PetscCall(MatCreateShell(comm, local_rows, local_cols, global_rows, global_cols, NULL, inv));
   PetscCall(MatShellSetOperation(*inv, MATOP_MULT, (PetscErrorCodeFn *)MatMult_LocalTri));
   PetscCall(MatShellSetVecType(*inv, ((PetscObject)ctx->r)->type_name));
   PetscCall(PetscObjectContainerCompose((PetscObject)(*inv), "pflare_local_tri", ctx, LocalTriCtxDestroy));
   PetscFunctionReturn(PETSC_SUCCESS);
}
