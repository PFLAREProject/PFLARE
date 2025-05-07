/*
  C routines that are used to get around a lack of some PETSc fortran interfaces
*/

// Include the petsc header files
#include <petsc.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petsc/private/pcimpl.h>

// Set shell vec type - need this for the r, b and rhs it creates during 
// the mg setup
PETSC_INTERN void ShellSetVecType_c(Mat *matrix, Mat *shellmatrix)
{
   VecType vtype;
   MatGetVecType(*matrix, &vtype);
   MatShellSetVecType(*shellmatrix, vtype);
   return;
}

// Does a vecscatter according to the pattern in the given Mat
// Have to do this in C as there is no fortran interface to MatGetCommunicationStructs
PETSC_INTERN void vecscatter_mat_begin_c(Mat *matrix, Vec *vec_long, double **nonlocal_vals)
{

   Mat_MPIAIJ *a = (Mat_MPIAIJ *)((*matrix)->data);
   // Could call this but can just access them directly
   //MatGetCommunicationStructs(*matrix, &lvec, PetscInt *colmap[], &multScatter)

   // Do the scatter
   VecScatterBegin(a->Mvctx, *vec_long, a->lvec, INSERT_VALUES, SCATTER_FORWARD);
}

// End the scatter started in vecscatter_mat_begin_c and return a pointer
// Have to call vecscatter_mat_restore_c on the pointer when done
PETSC_INTERN void vecscatter_mat_end_c(Mat *matrix, Vec *vec_long, double **nonlocal_vals)
{

   Mat_MPIAIJ *a = (Mat_MPIAIJ *)((*matrix)->data);
   VecScatterEnd(a->Mvctx, *vec_long, a->lvec, INSERT_VALUES, SCATTER_FORWARD);

   // lvec will now have the updated nonlocal values in it
   // and so we will just return a pointer to the values in that vec
   // these correspond to the numbering in Ao, with the local to global map for the columns
   // in colmap
   VecGetArray(a->lvec, nonlocal_vals);
   // Have to call a restore once you're done
}

PETSC_INTERN void vecscatter_mat_restore_c(Mat *matrix, double **nonlocal_vals)
{

   Mat_MPIAIJ *a = (Mat_MPIAIJ *)((*matrix)->data);
   VecRestoreArray(a->lvec, nonlocal_vals);
}

// Does a reverse scatter
// Must have updated the values in lvec before calling this
PETSC_INTERN void vecscatter_mat_reverse_begin_c(Mat *matrix, Vec *vec_long)
{

   Mat_MPIAIJ *a = (Mat_MPIAIJ *)((*matrix)->data);
   // Could call this but can just access them directly
   //MatGetCommunicationStructs(*matrix, &lvec, PetscInt *colmap[], &multScatter)

   // Do the reverse scatter - has to be an add
   VecScatterBegin(a->Mvctx, a->lvec, *vec_long, ADD_VALUES, SCATTER_REVERSE);
}

PETSC_INTERN void vecscatter_mat_reverse_end_c(Mat *matrix, Vec *vec_long)
{

   Mat_MPIAIJ *a = (Mat_MPIAIJ *)((*matrix)->data);
   VecScatterEnd(a->Mvctx, a->lvec, *vec_long, ADD_VALUES, SCATTER_REVERSE);
}

// Annoying as the fortran pointer returned by MatSeqAIJGetArrayF90 seems to be 
// the wrong size and that can break things sometimes
PETSC_INTERN void MatSeqAIJGetArrayF90_mine(Mat *matrix, double **array)
{
   Mat_SeqAIJ *a = (Mat_SeqAIJ*)((*matrix)->data);
   *array = a->a;
   return;
}

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
   PetscObjectGetComm((PetscObject)*adj, &comm);
   MPI_Comm_size(comm, &size);
   comm_size_petsc = (PetscInt)size;
   PetscInt* counts;
   PetscMalloc1(comm_size_petsc, &counts);
   MPI_Comm_rank(comm, &rank);

   // Check that new_size <= comm_world/proc_stride
   ratio = ((double)size)/((PetscReal)*proc_stride);
   if (new_size > ratio) 
   {
      errorcode = 1;
      MPI_Abort(comm, errorcode);
   }

   PetscInt ncrs_eq_local, ncrs_eq_global;
   MatGetLocalSize(*adj, &ncrs_eq_local, NULL);
   MatGetSize(*adj, &ncrs_eq_global, NULL);
   const PetscInt *is_idx;
   PetscInt* newproc_idx;
   PetscMalloc1(ncrs_eq_local, &newproc_idx);
   
   // Construct an IS with the current partition in it
   if (improve)
   {
      for (PetscInt i = 0; i < ncrs_eq_local; i++)
      {
         newproc_idx[i] = rank;
      }
      ISCreateGeneral(comm, ncrs_eq_local, newproc_idx, PETSC_COPY_VALUES, &proc_is);
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
   MatPartitioningCreate(comm, &mpart);
   MatPartitioningSetAdjacency(mpart, *adj);
   // Uses parmetis by default but you can change via the command line options
   MatPartitioningSetType(mpart, MATPARTITIONINGPARMETIS);
   MatPartitioningSetFromOptions(mpart);
   MatPartitioningSetNParts(mpart, new_size);
   
   // This returns an IS with the new rank of each equation 
   if (improve)
   {  // Either improve the existing partition
      MatPartitioningImprove(mpart, &proc_is);
   }
   else
   {
      // Or compute a brand new one
      MatPartitioningApply(mpart, &proc_is);
   }
   MatPartitioningDestroy(&mpart);   

   // If you're doing the improve you can't reduce the number of active ranks
   if (!improve)
   {
      ISGetIndices(proc_is, &is_idx);
      // Modified as we never have block equations
      for (kk = 0; kk < ncrs_eq_local; kk++) {
         newproc_idx[kk] = is_idx[kk] * expand_factor; /* distribution */
      }
      ISRestoreIndices(proc_is, &is_idx);   
      ISDestroy(&proc_is);
      ISCreateGeneral(comm, ncrs_eq_local, newproc_idx, PETSC_COPY_VALUES, &proc_is);
   }

   // Get how many eqs have been partitioned onto each rank
   ISPartitioningCount(proc_is, comm_size_petsc, counts);

   // This turns the rank into a new global numbering
   // ie the number for each local variable in this is the new global numbering
   ISPartitioningToNumbering(proc_is, &proc_is_eq_num);
   ISDestroy(&proc_is);

   // This comms the new global numbering to the rank it belongs onto
   // so we can us new_eq_indices (an IS) in a call to matgetsubmatrix
   // The petsc doc says this is expensive for greater than 10M indices....
   ISInvertPermutation(proc_is_eq_num, counts[rank], index);
   ISDestroy(&proc_is_eq_num);

   (void)PetscFree(counts);
   (void)PetscFree(newproc_idx);
   return;
}

// Create data for the process agglomeration
PETSC_INTERN void GenerateIS_ProcAgglomeration_c(PetscInt proc_stride, PetscInt global_size, PetscInt *local_size_reduced, PetscInt *start)
{

   PetscSubcomm psubcomm = NULL;
   MPI_Comm subcomm = MPI_COMM_NULL;

   // Taken from telescope.c Line 515, we're creating a subcomm
   // letting petsc decide on which ranks are interlaced
   PetscSubcommCreate(MPI_COMM_WORLD,&psubcomm);
   PetscSubcommSetNumber(psubcomm,proc_stride);
   // The petsc subcomm splits into proc_stride groups
   PetscSubcommSetType(psubcomm,PETSC_SUBCOMM_INTERLACED);
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
      VecCreateMPI(subcomm, PETSC_DECIDE, global_size, &xred);
      VecGetOwnershipRange(xred, start, &ed);
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
      VecDestroy(&xred);
   }
   PetscSubcommDestroy(&psubcomm);
   return;
}

// Computes a symbolic mat mat mult - the fortran interface doesn't have
// MATPRODUCT_AB or MatProductSetFromOptions set
PETSC_INTERN void mat_mat_symbolic_c(Mat *A, Mat *B, Mat *result)
{
   MPI_Comm comm;
   int comm_size;
   MatType mat_type_a, mat_type_b;

   // Get the comm
   PetscObjectGetComm((PetscObject)*A, &comm);
   MPI_Comm_size(comm, &comm_size);

   MatGetType(*A, &mat_type_a);
   MatGetType(*B, &mat_type_b);

   if (strcmp(mat_type_a, MATDIAGONAL) == 0)
   {
      MatDuplicate(*B, MAT_DO_NOT_COPY_VALUES, result);
   }
   else if (strcmp(mat_type_b, MATDIAGONAL) == 0)
   {  
      MatDuplicate(*A, MAT_DO_NOT_COPY_VALUES, result);
   }
   else
   {
      // For some reason in serial matduplicate is not defined on unassembled matrices
      // ie we call a matduplicate on the symbolic sparsity_mat returned from this
      // So we just do an ordinary matmatmult in serial
      if (comm_size == 1) 
      {
         MatMatMult(*A, *B, MAT_INITIAL_MATRIX, 1.0, result);
      }
      else
      {
         MatProductCreate(*A, *B, NULL, result);
         MatProductSetType(*result, MATPRODUCT_AB);
         MatProductSetAlgorithm(*result, "default");
         MatProductSetFill(*result, PETSC_DEFAULT);
         MatProductSetFromOptions(*result);
         MatProductSymbolic(*result);
         MatProductClear(*result);
      }
   }

   return;
}

// Takes a Mat and returns a Mat on a subcomm without ranks with no rows
// If there are no empty ranks, then it returns the same matrix
// If not there is a new copy that must be destroyed
// Basically taken from MatMPIAdjCreateNonemptySubcommMat_MPIAdj
PETSC_INTERN PetscErrorCode MatMPICreateNonemptySubcomm_c(Mat *A, int *on_subcomm, Mat *B)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ *)(*A)->data;
  const PetscInt *ranges;
  MPI_Comm        acomm, bcomm;
  MPI_Group       agroup, bgroup;
  PetscMPIInt     i, rank, size, nranks, *ranks;

  PetscFunctionBegin;

  *on_subcomm = 1;

  // Ensure we return petsc_null so we can test if we are on this subcomm
  // outside the function
  *B = PETSC_NULLPTR;
  PetscObjectGetComm((PetscObject)(*A), &acomm);
  MPI_Comm_size(acomm, &size);
  MPI_Comm_size(acomm, &rank);
  MatGetOwnershipRanges((*A), &ranges);
  for (i = 0, nranks = 0; i < size; i++) {
    if (ranges[i + 1] - ranges[i] > 0) nranks++;
  }
  if (nranks == size) { /* All ranks have a positive number of rows, so we do not need to create a subcomm; */
    PetscObjectReference((PetscObject)(*A));
    *B = (*A);
    // We haven't actually ended up on a subcomm
    *on_subcomm = 0;
    PetscFunctionReturn(0);
  }

  PetscMalloc1(nranks, &ranks);
  for (i = 0, nranks = 0; i < size; i++) {
    if (ranges[i + 1] - ranges[i] > 0) ranks[nranks++] = i;
  }
  MPI_Comm_group(acomm, &agroup);
  MPI_Group_incl(agroup, nranks, ranks, &bgroup);
  (void)PetscFree(ranks);
  MPI_Comm_create(acomm, bgroup, &bcomm);
  MPI_Group_free(&agroup);
  MPI_Group_free(&bgroup);
  if (bcomm != MPI_COMM_NULL) {
    PetscInt    m, n, M, N;
    Mat Ad_copy, Ao_copy; 
    MatGetLocalSize((*A), &m, &n);
    MatGetSize((*A), &M, &N);

    // If we've gone down to a serial matrix
    if (nranks == 1) 
    {
      // Just send out a copy of the local part of the input matrix
      MatDuplicate(a->A, MAT_COPY_VALUES, B);
    }
    else{
      // Copy the local and off-diagonal sequential matrices
      MatDuplicate(a->A, MAT_COPY_VALUES, &Ad_copy);
      MatDuplicate(a->B, MAT_COPY_VALUES, &Ao_copy);

      // MAT_NO_OFF_PROC_ENTRIES is set to true in this routine so 
      // don't need to set it externally
      // Have to be careful here as need to feed in copies of A and B
      MatCreateMPIAIJWithSeqAIJ(bcomm, M, N, Ad_copy, Ao_copy, a->garray, B);       
    }

    MPI_Comm_free(&bcomm);
  }
  PetscFunctionReturn(0);
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

// Gets the number of nonzeros in the local 
PETSC_INTERN PetscErrorCode MatGetNNZs_local_c(Mat *A, PetscInt *nnzs)
{
  MPI_Comm        acomm;
  PetscMPIInt     size;
  MatType mat_type;

  PetscFunctionBegin;

  PetscObjectGetComm((PetscObject)(*A), &acomm);
  MPI_Comm_size(acomm, &size);

  MatGetType(*A, &mat_type);
  if (strcmp(mat_type, MATDIAGONAL) == 0)
  {
      // This is a diagonal matrix, so just return the number of rows
      MatGetLocalSize(*A, nnzs, NULL);
      PetscFunctionReturn(0);   
  }

  Mat_MPIAIJ *mat_mpi = NULL;
  Mat mat_local; 

  // Get the existing output mats
  if (size != 1)
  {
     mat_mpi = (Mat_MPIAIJ *)(*A)->data;
     mat_local = mat_mpi->A;
  }
  else
  {
     mat_local = *A;
  } 

  Mat_SeqAIJ *a = (Mat_SeqAIJ *)mat_local->data;

  // Set the number of non-zeros
  *nnzs = a->nz;

  PetscFunctionReturn(0);
}

// Gets the number of nonzeros in the both local and nonlocal 
PETSC_INTERN PetscErrorCode MatGetNNZs_both_c(Mat *A, PetscInt *nnzs_local, PetscInt *nnzs_nonlocal)
{
  MPI_Comm        acomm;
  PetscMPIInt     size;
  MatType mat_type;

  PetscFunctionBegin;

  PetscObjectGetComm((PetscObject)(*A), &acomm);
  MPI_Comm_size(acomm, &size);
  *nnzs_nonlocal = 0;

  MatGetType(*A, &mat_type);
  if (strcmp(mat_type, MATDIAGONAL) == 0)
  {
      // This is a diagonal matrix, so just return the number of rows
      MatGetLocalSize(*A, nnzs_local, NULL);
      PetscFunctionReturn(0);   
  }  

  Mat_MPIAIJ *mat_mpi = NULL;
  Mat mat_local = NULL, mat_nonlocal = NULL; 

  // Get the existing output mats
  if (size != 1)
  {
     mat_mpi = (Mat_MPIAIJ *)(*A)->data;
     mat_local = mat_mpi->A;
     mat_nonlocal = mat_mpi->B;
  }
  else
  {
     mat_local = *A;
  } 

  Mat_SeqAIJ *a = (Mat_SeqAIJ *)mat_local->data;

  // Set the number of non-zeros
  *nnzs_local = a->nz;

  if (size != 1)
  {
     Mat_SeqAIJ *b = (Mat_SeqAIJ *)mat_nonlocal->data;
     *nnzs_nonlocal = b->nz;
  }

  PetscFunctionReturn(0);
}

// Returns true if a matrix is only the diagonal
PETSC_INTERN PetscErrorCode MatGetDiagonalOnly_c(Mat *A, int *diag_only)
{
  MPI_Comm        acomm;
  PetscMPIInt     size;
  PetscInt local_rows, local_cols;
  int rank_diag = 0, rank_diag_serial = 0;

  PetscFunctionBegin;

  PetscObjectGetComm((PetscObject)(*A), &acomm);
  MPI_Comm_size(acomm, &size);

  Mat_MPIAIJ *mat_mpi = NULL;
  Mat mat_local = NULL, mat_nonlocal = NULL; 

  // Get the existing output mats
  if (size != 1)
  {
     mat_mpi = (Mat_MPIAIJ *)(*A)->data;
     mat_local = mat_mpi->A;
     mat_nonlocal = mat_mpi->B;
  }
  else
  {
     mat_local = *A;
  } 

  Mat_SeqAIJ *a = (Mat_SeqAIJ *)mat_local->data;
  MatGetLocalSize(*A, &local_rows, &local_cols);
  *diag_only = 0;

  if (size != 1)
  {
      Mat_SeqAIJ *b = (Mat_SeqAIJ *)mat_nonlocal->data;

      // In parallel also have to check the nonlocal has nothing in it
      if (a->diagonaldense && local_rows == a->nz && b->nz == 0)
      {
         rank_diag_serial++;
      }

      // Reduction to check every rank
      (void)MPI_Allreduce(&rank_diag_serial, &rank_diag, 1, MPI_INTEGER, MPI_SUM, acomm);
  }
  else
  {
      // In serial easy 
      if (a->diagonaldense && local_rows == a->nz)
      {
         rank_diag++;
      }
  }   

  // If every rank is diagonal only, the entire matrix is diagonal
  if (rank_diag == size)
  {
     *diag_only = 1;
  }

  PetscFunctionReturn(0);
}