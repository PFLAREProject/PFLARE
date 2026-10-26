/*
  C port of Repartition.F90 (module repartition)

  The C helpers the Fortran called (MatPartitioning_c, GenerateIS_ProcAgglomeration_c
  and MatMPICreateNonemptySubcomm_c) were in C_PETSc_Routines.c and are at the end
  of this file, now returning a PetscErrorCode
*/

#include <petscmat.h>
#include "pflareimpl.h"
#include "PETSc_Helper.h"
#include "Repartition.h"

// -------------------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------------------------------------
// Some helper functions that handle repartitioning of petsc matrices
// -------------------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode compute_mat_ratio_local_nonlocal_nnzs(Mat input_mat, PetscInt no_active_cores, PetscReal *ratio)
{
   // Computes the local to non local ratios of nnzs in input_mat

   // ~~~~~~
   // Local
   PetscInt local_nnzs, off_proc_nnzs;
   MPI_Comm MPI_COMM_MATRIX;
   PetscReal ratio_parallel;

   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)input_mat, &MPI_COMM_MATRIX));
   PetscCall(MatGetNNZs_both(input_mat, &local_nnzs, &off_proc_nnzs));

   // ~~~~~~~~~~~
   // Get the ratio of local to non-local nnzs
   // ~~~~~~~~~~~
   // If a processor is entirely local, don't include it in the ratio
   // It's a bit hard to decide here what to do, as we don't want to comm how many
   // processors actually have non local work
   // So the proxy for that is no_active_cores but
   // there is no guarantee that that many procs all have nonlocal entries
   if (off_proc_nnzs == 0) {
      *ratio = 0;
   } else {
      *ratio = (PetscReal)local_nnzs / (PetscReal)off_proc_nnzs;
   }

   PetscCallMPI(MPI_Allreduce(ratio, &ratio_parallel, 1, MPIU_REAL, MPI_SUM, MPI_COMM_MATRIX));

   *ratio = ratio_parallel;
   // Only divide by the number of processors that are active
   *ratio = *ratio / (PetscReal)no_active_cores;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode calculate_repartition(Mat input_mat, PetscInt proc_stride, PetscInt *no_active_cores, PetscBool simple, IS *index)
{
   // Return an IS that represents a repartitioning of the matrix
   // Simple being true is just processor aggregation to load balance
   // partitions

   // ~~~~~~
   // Local
   PetscInt global_rows, global_cols;
   PetscMPIInt comm_size;
   MPI_Comm MPI_COMM_MATRIX;
   Mat adj, input_transpose;
   PetscInt local_size_is, start_is;
   Mat A_array;
   PetscInt one = 1;

   // ~~~~~~

   PetscFunctionBegin;

   PetscCall(PetscObjectGetComm((PetscObject)input_mat, &MPI_COMM_MATRIX));
   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));

   // If doing simple processor aggregation
   if (simple) {

      PetscCall(MatGetSize(input_mat, &global_rows, &global_cols));
      PetscCall(GenerateIS_ProcAgglomeration_c(proc_stride, global_rows, &local_size_is, &start_is));

      // Specifically on comm_world as we aren't going onto a subcomm
      // If we're on a processor we wish to accumulate onto the local_size_is will have
      // other enties, it will be zero on processors that will have all entries removed from
      PetscCall(ISCreateStride(MPI_COMM_WORLD, local_size_is, start_is, one, index));

   // Else call one of the partitioners through petsc
   } else {

      // Number of cores we want dofs on
      *no_active_cores = (PetscInt)floor((double)comm_size / (double)proc_stride);
      // Be careful of rounding!
      if (*no_active_cores == 0) *no_active_cores = 1;

      // Have to symmetrize the input matrix or it won't work in parmetis
      // as it expects a symmetric graph
      PetscCall(MatTranspose(input_mat, MAT_INITIAL_MATRIX, &input_transpose));
      PetscCall(MatAXPYWrapper(input_transpose, PFLARE_ONE, input_mat));

      // Compute the adjancency graph of the symmetrized input matrix
      PetscCall(MatConvert(input_transpose, MATMPIADJ, MAT_INITIAL_MATRIX, &adj));

      A_array = adj;
      PetscCall(MatPartitioning_c(&A_array, *no_active_cores, &proc_stride, index));

      // Destroy the adjacency matrices
      PetscCall(MatDestroy(&adj));
      PetscCall(MatDestroy(&input_transpose));

   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode MatMPICreateNonemptySubcomm(Mat input_mat, PetscBool *on_subcomm, Mat *output_mat)
{
   // Just calls MatMPICreateNonemptySubcomm_c

   // ~~~~~~
   Mat A_array, B_array;
   int on_subcomm_int;

   // ~~~~~~

   PetscFunctionBegin;

   A_array = input_mat;
   // Don't carry on with a garbage B_array if petsc failed in the c routine
   PetscCall(MatMPICreateNonemptySubcomm_c(&A_array, &on_subcomm_int, &B_array));
   if (on_subcomm_int == 1) {
      *on_subcomm = PETSC_TRUE;
   } else {
      *on_subcomm = PETSC_FALSE;
   }
   // Assign the index to the mat we get back from c
   *output_mat = B_array;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------
// The C helpers that were in C_PETSc_Routines.c
// -------------------------------------------------------------------------------------------------------------------------------

// MatPartitioningSetNParts doesn't have a fortran interface (and can't pass around
// a matpartioning object as it doesnt have a %v), so have to do
// all the partitioning in C
// We have modified this from PCGAMGCreateLevel_GAMG, and we enforce that
// new_size <= comm_world/proc_stride
// This always does an interleaved partitioning
// PCGAMGCreateLevel_GAMG can modify new_size, but we don't allow that, we always partition
// onto new_size active ranks and we enforce the interleaving happens nested to reduce comms
PETSC_INTERN PetscErrorCode MatPartitioning_c(Mat *adj, PetscInt new_size, PetscInt *proc_stride, IS *index)
{
   // This is all taken from PCGAMGCreateLevel_GAMG in gamg.c
   MPI_Comm comm;
   MatPartitioning mpart;
   IS proc_is, proc_is_eq_num;
   PetscInt comm_size_petsc;
   int size, rank;
   // If you want to use the improve, then new_size needs to equal the current size
   // ie you can't reduce the number of active ranks
   int improve = 0;
   double ratio;

   PetscFunctionBegin;

   // Get the comm
   PetscCall(PetscObjectGetComm((PetscObject)*adj, &comm));
   PetscCallMPI(MPI_Comm_size(comm, &size));
   comm_size_petsc = (PetscInt)size;
   PetscInt* counts;
   PetscCall(PetscMalloc1(comm_size_petsc, &counts));
   PetscCallMPI(MPI_Comm_rank(comm, &rank));

   // Check that new_size <= comm_world/proc_stride
   ratio = ((double)size)/((PetscReal)*proc_stride);
   PetscCheck(!(new_size > ratio), comm, PETSC_ERR_ARG_OUTOFRANGE, "The number of partitions %" PetscInt_FMT " is more than the comm size divided by proc_stride", new_size);

   PetscInt ncrs_eq_local, ncrs_eq_global;
   PetscCall(MatGetLocalSize(*adj, &ncrs_eq_local, NULL));
   PetscCall(MatGetSize(*adj, &ncrs_eq_global, NULL));
   const PetscInt *is_idx;
   PetscInt* newproc_idx;
   PetscCall(PetscMalloc1(ncrs_eq_local, &newproc_idx));

   // Construct an IS with the current partition in it
   if (improve)
   {
      for (PetscInt i = 0; i < ncrs_eq_local; i++)
      {
         newproc_idx[i] = rank;
      }
      PetscCall(ISCreateGeneral(comm, ncrs_eq_local, newproc_idx, PETSC_COPY_VALUES, &proc_is));
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
   PetscCall(MatPartitioningCreate(comm, &mpart));
   PetscCall(MatPartitioningSetAdjacency(mpart, *adj));
   // Uses parmetis by default but you can change via the command line options
#if defined(PETSC_HAVE_PARMETIS)
   PetscCall(MatPartitioningSetType(mpart, MATPARTITIONINGPARMETIS));
#else
   // ParMETIS is not available (e.g. PETSc built without MPI); fall back to a
   // built-in partitioner that needs no external package. This path is only
   // reached in parallel, so it never runs in a serial (MPIUNI) build anyway
   PetscCall(MatPartitioningSetType(mpart, MATPARTITIONINGAVERAGE));
#endif
   PetscCall(MatPartitioningSetFromOptions(mpart));
   PetscCall(MatPartitioningSetNParts(mpart, new_size));

   // This returns an IS with the new rank of each equation
   if (improve)
   {  // Either improve the existing partition
      PetscCall(MatPartitioningImprove(mpart, &proc_is));
   }
   else
   {
      // Or compute a brand new one
      PetscCall(MatPartitioningApply(mpart, &proc_is));
   }
   PetscCall(MatPartitioningDestroy(&mpart));

   // If you're doing the improve you can't reduce the number of active ranks
   if (!improve)
   {
      PetscCall(ISGetIndices(proc_is, &is_idx));
      // Modified as we never have block equations
      for (kk = 0; kk < ncrs_eq_local; kk++) {
         newproc_idx[kk] = is_idx[kk] * expand_factor; /* distribution */
      }
      PetscCall(ISRestoreIndices(proc_is, &is_idx));
      PetscCall(ISDestroy(&proc_is));
      PetscCall(ISCreateGeneral(comm, ncrs_eq_local, newproc_idx, PETSC_COPY_VALUES, &proc_is));
   }

   // Get how many eqs have been partitioned onto each rank
   PetscCall(ISPartitioningCount(proc_is, comm_size_petsc, counts));

   // This turns the rank into a new global numbering
   // ie the number for each local variable in this is the new global numbering
   PetscCall(ISPartitioningToNumbering(proc_is, &proc_is_eq_num));
   PetscCall(ISDestroy(&proc_is));

   // This comms the new global numbering to the rank it belongs onto
   // so we can us new_eq_indices (an IS) in a call to matgetsubmatrix
   // The petsc doc says this is expensive for greater than 10M indices....
   PetscCall(ISInvertPermutation(proc_is_eq_num, counts[rank], index));
   PetscCall(ISDestroy(&proc_is_eq_num));

   PetscCall(PetscFree(counts));
   PetscCall(PetscFree(newproc_idx));
   PetscFunctionReturn(PETSC_SUCCESS);
}

// Create data for the process agglomeration
PETSC_INTERN PetscErrorCode GenerateIS_ProcAgglomeration_c(PetscInt proc_stride, PetscInt global_size, PetscInt *local_size_reduced, PetscInt *start)
{

   PetscSubcomm psubcomm = NULL;
   MPI_Comm subcomm = MPI_COMM_NULL;

   PetscFunctionBegin;

   // Taken from telescope.c Line 515, we're creating a subcomm
   // letting petsc decide on which ranks are interlaced
   PetscCall(PetscSubcommCreate(MPI_COMM_WORLD,&psubcomm));
   PetscCall(PetscSubcommSetNumber(psubcomm,proc_stride));
   // The petsc subcomm splits into proc_stride groups
   PetscCall(PetscSubcommSetType(psubcomm,PETSC_SUBCOMM_INTERLACED));
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
      PetscCall(VecCreateMPI(subcomm, PETSC_DECIDE, global_size, &xred));
      PetscCall(VecGetOwnershipRange(xred, start, &ed));
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
      PetscCall(VecDestroy(&xred));
   }
   PetscCall(PetscSubcommDestroy(&psubcomm));
   PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCallMPI(MPI_Comm_size(acomm, &size));
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
  PetscCallMPI(MPI_Comm_group(acomm, &agroup));
  PetscCallMPI(MPI_Group_incl(agroup, nranks, ranks, &bgroup));
  PetscCall(PetscFree(ranks));
  PetscCallMPI(MPI_Comm_create(acomm, bgroup, &bcomm));
  PetscCallMPI(MPI_Group_free(&agroup));
  PetscCallMPI(MPI_Group_free(&bgroup));
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
      PetscCallMPI(MPI_Comm_free(&bcomm));
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
      PetscCallMPI(MPI_Comm_free(&bcomm));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
