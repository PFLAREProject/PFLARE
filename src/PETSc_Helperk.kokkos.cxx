#include <petscvec_kokkos.hpp>
#include <petsc.h>
#include <iostream>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/aij/seq/kokkos/aijkok.hpp>
#include "Kokkos_UnorderedMap.hpp"
#include <Kokkos_StdAlgorithms.hpp>
#include <../src/vec/vec/impls/seq/kokkos/veckokkosimpl.hpp>
// Our kokkos definitions
#include "kokkos_helper.h"

struct ReduceData {
   PetscInt count;
   bool found_diagonal;
   
   // Set count to zero and found_diagonal to false
   KOKKOS_INLINE_FUNCTION
   ReduceData() : count(0), found_diagonal(false) {}
   
   // We use this in our parallel reduction
   KOKKOS_INLINE_FUNCTION
   void operator+=(const ReduceData& src) {
      // Add all the counts
      count += src.count;
      // If we have found a diagonal entry at any point in this row
      // found_diagonal becomes true      
      found_diagonal |= src.found_diagonal;
   }

   // Required for Kokkos reduction
   KOKKOS_INLINE_FUNCTION
   static void join(volatile ReduceData& dest, const volatile ReduceData& src) {
      dest.count += src.count;
      dest.found_diagonal |= src.found_diagonal;
   }   
};

namespace Kokkos {
    template<>
    struct reduction_identity<ReduceData> {
        KOKKOS_INLINE_FUNCTION
        static ReduceData sum() {
            return ReduceData();  // Returns {count=0, found_diagonal=false}
        }
    };
}

struct ReduceDataMaxRow {
   PetscInt col;
   PetscReal val;
   
   // Set col to negative one and val to -1.0
   KOKKOS_INLINE_FUNCTION
   ReduceDataMaxRow() : col(-1), val(-1.0) {}
   
   // We use this in our parallel reduction to find maximum
   KOKKOS_INLINE_FUNCTION
   void operator+=(const ReduceDataMaxRow& src) {
      // If src has a larger value, take it
      if (src.val > val) {
         val = src.val;
         col = src.col;
      }
   }

   // Required for Kokkos reduction
   KOKKOS_INLINE_FUNCTION
   static void join(volatile ReduceDataMaxRow& dest, const volatile ReduceDataMaxRow& src) {
      if (src.val > dest.val) {
         dest.val = src.val;
         dest.col = src.col;
      }
   }   
};

namespace Kokkos {
    template<>
    struct reduction_identity<ReduceDataMaxRow> {
        KOKKOS_INLINE_FUNCTION
        static ReduceDataMaxRow sum() {
            return ReduceDataMaxRow();  // Returns {col=-1, val=-1}
        }
    };
}

// Another horrid copy given it's only declared in the .cxx
PetscErrorCode MatSetMPIAIJKokkosWithSplitSeqAIJKokkosMatrices_mine(Mat mat, Mat A, Mat B, PetscInt *garray)
{
  Mat_MPIAIJ *mpiaij = static_cast<Mat_MPIAIJ *>(mat->data);
  PetscInt    m, n, M, N, Am, An, Bm, Bn;

  PetscFunctionBegin;
  PetscCall(MatGetSize(mat, &M, &N));
  PetscCall(MatGetLocalSize(mat, &m, &n));
  PetscCall(MatGetLocalSize(A, &Am, &An));
  PetscCall(MatGetLocalSize(B, &Bm, &Bn));

  PetscCheck(m == Am && m == Bm, PETSC_COMM_SELF, PETSC_ERR_PLIB, "local number of rows do not match");
  PetscCheck(n == An, PETSC_COMM_SELF, PETSC_ERR_PLIB, "local number of columns do not match");
  // PetscCheck(N == Bn, PETSC_COMM_SELF, PETSC_ERR_PLIB, "global number of columns do not match");
  PetscCheck(!mpiaij->A && !mpiaij->B, PETSC_COMM_SELF, PETSC_ERR_PLIB, "A, B of the MPIAIJ matrix are not empty");
  mpiaij->A      = A;
  mpiaij->B      = B;
  mpiaij->garray = garray;

  mat->preallocated     = PETSC_TRUE;
  mat->nooffprocentries = PETSC_TRUE; /* See MatAssemblyBegin_MPIAIJ. In effect, making MatAssemblyBegin a nop */

  PetscCall(MatSetOption(mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  /* MatAssemblyEnd is critical here. It sets mat->offloadmask according to A and B's, and
    also gets mpiaij->B compacted, with its col ids and size reduced
  */
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_FALSE));
  PetscCall(MatSetOption(mat, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Horrid copy of this code and MatSetSeqAIJKokkosWithCSRMatrix_mine as they're declared PETSC_INTERN
// so I can't get at them
PetscErrorCode MatSeqAIJSetPreallocation_SeqAIJ_mine(Mat B, PetscInt nz, const PetscInt *nnz)
{
  Mat_SeqAIJ *b              = (Mat_SeqAIJ *)B->data;
  PetscBool   skipallocation = PETSC_FALSE, realalloc = PETSC_FALSE;
  PetscInt    i;

  PetscFunctionBegin;
  if (B->hash_active) {
    B->ops[0] = b->cops;
    PetscCall(PetscHMapIJVDestroy(&b->ht));
    PetscCall(PetscFree(b->dnz));
    B->hash_active = PETSC_FALSE;
  }
  if (nz >= 0 || nnz) realalloc = PETSC_TRUE;
  if (nz == MAT_SKIP_ALLOCATION) {
    skipallocation = PETSC_TRUE;
    nz             = 0;
  }
  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutSetUp(B->cmap));

  if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
  PetscCheck(nz >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "nz cannot be less than 0: value %" PetscInt_FMT, nz);
  if (nnz) {
    for (i = 0; i < B->rmap->n; i++) {
      PetscCheck(nnz[i] >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "nnz cannot be less than 0: local row %" PetscInt_FMT " value %" PetscInt_FMT, i, nnz[i]);
      PetscCheck(nnz[i] <= B->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "nnz cannot be greater than row length: local row %" PetscInt_FMT " value %" PetscInt_FMT " rowlength %" PetscInt_FMT, i, nnz[i], B->cmap->n);
    }
  }

  B->preallocated = PETSC_TRUE;
  if (!skipallocation) {
    if (!b->imax) { PetscCall(PetscMalloc1(B->rmap->n, &b->imax)); }
    if (!b->ilen) {
      /* b->ilen will count nonzeros in each row so far. */
      PetscCall(PetscCalloc1(B->rmap->n, &b->ilen));
    } else {
      PetscCall(PetscMemzero(b->ilen, B->rmap->n * sizeof(PetscInt)));
    }
    if (!b->ipre) PetscCall(PetscMalloc1(B->rmap->n, &b->ipre));
    if (!nnz) {
      if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 10;
      else if (nz < 0) nz = 1;
      nz = PetscMin(nz, B->cmap->n);
      for (i = 0; i < B->rmap->n; i++) b->imax[i] = nz;
      PetscCall(PetscIntMultError(nz, B->rmap->n, &nz));
    } else {
      PetscInt64 nz64 = 0;
      for (i = 0; i < B->rmap->n; i++) {
        b->imax[i] = nnz[i];
        nz64 += nnz[i];
      }
      PetscCall(PetscIntCast(nz64, &nz));
    }

    /* allocate the matrix space */
    PetscCall(MatSeqXAIJFreeAIJ(B, &b->a, &b->j, &b->i));
    PetscCall(PetscShmgetAllocateArray(nz, sizeof(PetscInt), (void **)&b->j));
    PetscCall(PetscShmgetAllocateArray(B->rmap->n + 1, sizeof(PetscInt), (void **)&b->i));
    b->free_ij = PETSC_TRUE;
    if (B->structure_only) {
      b->free_a = PETSC_FALSE;
    } else {
      PetscCall(PetscShmgetAllocateArray(nz, sizeof(PetscScalar), (void **)&b->a));
      b->free_a = PETSC_TRUE;
    }
    b->i[0] = 0;
    for (i = 1; i < B->rmap->n + 1; i++) b->i[i] = b->i[i - 1] + b->imax[i - 1];
  } else {
    b->free_a  = PETSC_FALSE;
    b->free_ij = PETSC_FALSE;
  }

  if (b->ipre && nnz != b->ipre && b->imax) {
    /* reserve user-requested sparsity */
    PetscCall(PetscArraycpy(b->ipre, b->imax, B->rmap->n));
  }

  b->nz               = 0;
  b->maxnz            = nz;
  B->info.nz_unneeded = (double)b->maxnz;
  if (realalloc) PetscCall(MatSetOption(B, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  B->was_assembled = PETSC_FALSE;
  B->assembled     = PETSC_FALSE;
  /* We simply deem preallocation has changed nonzero state. Updating the state
     will give clients (like AIJKokkos) a chance to know something has happened.
  */
  B->nonzerostate++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSetSeqAIJKokkosWithCSRMatrix_mine(Mat A, Mat_SeqAIJKokkos *akok)
{
  Mat_SeqAIJ *aseq;
  PetscInt    i, m, n;
  auto       &exec = PetscGetKokkosExecutionSpace();

  PetscFunctionBegin;
  PetscCheck(!A->spptr, PETSC_COMM_SELF, PETSC_ERR_PLIB, "A->spptr is supposed to be empty");

  m = akok->nrows();
  n = akok->ncols();
  PetscCall(MatSetSizes(A, m, n, m, n));
  PetscCall(MatSetType(A, MATSEQAIJKOKKOS));

  /* Set up data structures of A as a MATSEQAIJ */
  PetscCall(MatSeqAIJSetPreallocation_SeqAIJ_mine(A, MAT_SKIP_ALLOCATION, NULL));
  aseq = (Mat_SeqAIJ *)A->data;

  PetscCallCXX(akok->i_dual.sync_host(exec)); /* We always need sync'ed i, j on host */
  PetscCallCXX(akok->j_dual.sync_host(exec));
  PetscCallCXX(exec.fence());

  aseq->i       = akok->i_host_data();
  aseq->j       = akok->j_host_data();
  aseq->a       = akok->a_host_data();
  aseq->nonew   = -1; /*this indicates that inserting a new value in the matrix that generates a new nonzero is an error*/
  aseq->free_a  = PETSC_FALSE;
  aseq->free_ij = PETSC_FALSE;
  aseq->nz      = akok->nnz();
  aseq->maxnz   = aseq->nz;

  PetscCall(PetscMalloc1(m, &aseq->imax));
  PetscCall(PetscMalloc1(m, &aseq->ilen));
  for (i = 0; i < m; i++) aseq->ilen[i] = aseq->imax[i] = aseq->i[i + 1] - aseq->i[i];

  /* It is critical to set the nonzerostate, as we use it to check if sparsity pattern (hence data) has changed on host in MatAssemblyEnd */
  akok->nonzerostate = A->nonzerostate;
  A->spptr           = akok; /* Set A->spptr before MatAssembly so that A->spptr won't be allocated again there */
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------------------------------------------------------------------------------------------------------

// Drop according to a tolerance but with kokkos - keeping everything on the device
PETSC_INTERN void remove_small_from_sparse_kokkos(Mat *input_mat, PetscReal tol, Mat *output_mat, \
                  int relative_max_row_tolerance_int, int lump_int, int allow_drop_diagonal_int)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt global_col_start, global_col_end_plus_one;
   PetscInt rows_ao, cols_ao;
   MatType mat_type;
   PetscInt nnzs_match_local, nnzs_match_nonlocal;
   Mat output_mat_local, output_mat_nonlocal;

   MatGetType(*input_mat, &mat_type);
   // Are we in parallel?
   bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat_MPIAIJ *mat_mpi = nullptr;
   Mat mat_local, mat_nonlocal;

   PetscIntKokkosViewHost colmap_input_h;
   PetscIntKokkosView colmap_input_d;   
   if (mpi)
   {
      mat_mpi = (Mat_MPIAIJ *)(*input_mat)->data;
      mat_local = mat_mpi->A;
      mat_nonlocal = mat_mpi->B;
      MatGetSize(mat_nonlocal, &rows_ao, &cols_ao); 

      // We also copy the input mat colmap over to the device as we need it
      colmap_input_h = PetscIntKokkosViewHost(mat_mpi->garray, cols_ao);
      colmap_input_d = PetscIntKokkosView("colmap_input_d", cols_ao);
      Kokkos::deep_copy(colmap_input_d, colmap_input_h);        
   }
   else
   {
      mat_local = *input_mat;
   }

   // Get the comm
   PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX);
   MatGetLocalSize(*input_mat, &local_rows, &local_cols);
   MatGetSize(*input_mat, &global_rows, &global_cols);
   // This returns the global index of the local portion of the matrix
   MatGetOwnershipRange(*input_mat, &global_row_start, &global_row_end_plus_one);
   MatGetOwnershipRangeColumn(*input_mat, &global_col_start, &global_col_end_plus_one);   

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr, *device_nonlocal_vals = nullptr;  
   MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype);  
   if (mpi) MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype);          

   // ~~~~~~~~~~~~
   // Get the number of nnzs
   // ~~~~~~~~~~~~
   nnzs_match_local = 0;
   nnzs_match_nonlocal = 0;

   // ~~~~~~~~~~~~~~~~~~~~~~~
   // Let's build our i, j, and a on the device
   // ~~~~~~~~~~~~~~~~~~~~~~~
   // We need the relative row tolerance, let's create some device memory to store it
   PetscScalarKokkosView rel_row_tol_d("rel_row_tol_d", local_rows);    
   // Copy in the tolerance
   Kokkos::deep_copy(rel_row_tol_d, tol);     
   // We need to know how many entries are in each row after our dropping  
   PetscIntKokkosView nnz_match_local_row_d("nnz_match_local_row_d", local_rows);             
   PetscIntKokkosView nnz_match_nonlocal_row_d;
   if (mpi) nnz_match_nonlocal_row_d = PetscIntKokkosView("nnz_match_nonlocal_row_d", local_rows);                  
   
   // Compute the relative row tolerances if needed
   if (relative_max_row_tolerance_int) 
   {       
      // Reduction over all rows
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            PetscInt i = t.league_rank();
            PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];
            PetscScalar max_val = -1.0;

            // Reduce over local columns
            Kokkos::parallel_reduce(
               Kokkos::TeamThreadRange(t, ncols_local),
               [&](const PetscInt j, PetscScalar& thread_max) {

                  // Is this column the diagonal
                  bool is_diagonal = (device_local_j[device_local_i[i] + j] + global_col_start == i + global_row_start);

                  // If our current tolerance is bigger than the max value we've seen so far
                  PetscScalar val = abs(device_local_vals[device_local_i[i] + j]);
                  // If we're not comparing against the diagonal when computing relative residual
                  if (relative_max_row_tolerance_int == -1 && is_diagonal) val = -1.0;
                  if (val > thread_max) thread_max = val;

               },
               Kokkos::Max<PetscScalar>(max_val)
            );

            if (mpi) {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
               PetscScalar max_val_nonlocal = -1.0;
               
               // Reduce over nonlocal columns
               Kokkos::parallel_reduce(
                  Kokkos::TeamThreadRange(t, ncols_nonlocal),
                  [&](const PetscInt j, PetscScalar& thread_max) {

                     // Is this column the diagonal
                     bool is_diagonal = (colmap_input_d(device_nonlocal_j[device_nonlocal_i[i] + j]) == i + global_row_start);

                     // If our current tolerance is bigger than the max value we've seen so far
                     PetscScalar val = abs(device_nonlocal_vals[device_nonlocal_i[i] + j]);
                     // If we're not comparing against the diagonal when computing relative residual
                     if (relative_max_row_tolerance_int == -1 && is_diagonal) val = -1.0;                  
                     if (val > thread_max) thread_max = val;

                  },
                  Kokkos::Max<PetscScalar>(max_val_nonlocal)
               );
               // Take max of local and nonlocal
               if (max_val_nonlocal > max_val) max_val = max_val_nonlocal;               
            }

            // Only want one thread in the team to write the result
            Kokkos::single(Kokkos::PerTeam(t), [&]() {
               rel_row_tol_d(i) *= max_val;
            });
      });
   }

   // ~~~~~~~~~~~~
   // Need to count the number of nnzs we end up with, on each row and in total
   // ~~~~~~~~~~~~
   // Reduce over all the rows
   Kokkos::parallel_reduce(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t, PetscInt& thread_total) {

      PetscInt i   = t.league_rank(); // row i
      PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];

      // For this row, would we expect the diagonal to be in the local block or in the nonlocal?
      // Trivially true in the local block for square matrices
      bool expect_local_diagonal = i + global_row_start >= global_col_start && \
                           i + global_row_start < global_col_end_plus_one;

      // We have a custom reduction type defined - ReduceData
      // Which has both a nnz count for this row, but also tracks whether we 
      // found the diagonal
      ReduceData row_result;

      // Reduce over all the columns
      Kokkos::parallel_reduce(
         Kokkos::TeamThreadRange(t, ncols_local),
         [&](const PetscInt j, ReduceData& thread_data) {

            // Is this column the diagonal
            bool is_diagonal = (device_local_j[device_local_i[i] + j] + global_col_start == i + global_row_start);
            
            // We have found a diagonal in this row
            if (is_diagonal) {
               thread_data.found_diagonal = true;
            }
            
            // If the value is bigger than the tolerance, we keep it
            if (abs(device_local_vals[device_local_i[i] + j]) >= rel_row_tol_d(i)) {
               // If this is the diagonal and we're dropping all diagonals don't add it
               if (!(allow_drop_diagonal_int == -1 && is_diagonal)) thread_data.count++;
            }
            // Or if it's small but its the diagonal and we're keeping diagonals
            else if (allow_drop_diagonal_int == 0 && is_diagonal) {
               thread_data.count++;
            }
         }, row_result
      );

      // We're finished our parallel reduction for this row
      // If we're lumping but there was no diagonal in this row and there
      // should be we'll have to add in a diagonal to the local block
      // This will add one for every thread in this team, but all 
      // the threads in this team share the same result after the reduction
      if (lump_int && expect_local_diagonal && !row_result.found_diagonal) row_result.count++;
      // Only want one thread in the team to write the result
      Kokkos::single(Kokkos::PerTeam(t), [&]() {      
         nnz_match_local_row_d(i) = row_result.count;
         thread_total += row_result.count;
      });
      },
      nnzs_match_local
   );

   // ~~~~~~~~~~~~

   // Need to do a scan on nnz_match_local_row_d to get where each row starts
   Kokkos::parallel_scan (local_rows, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
      // Inclusive scan
      update += nnz_match_local_row_d(i);         
      if (final) {
         nnz_match_local_row_d(i) = update; // only update array on final pass
      }
   });            

   if (mpi) 
   {

      // ~~~~~~~~~~~~
      // Need to count the number of nnzs we end up with, on each row and in total
      // ~~~~~~~~~~~~
      // Reduce over all the rows
      Kokkos::parallel_reduce(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t, PetscInt& thread_total) {
            
            PetscInt i = t.league_rank();
            PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];

            // For this row, would we expect the diagonal to be in the local block or in the nonlocal?
            bool expect_local_diagonal = i + global_row_start >= global_col_start && \
                                 i + global_row_start < global_col_end_plus_one;

            ReduceData row_result;          

            // Reduce over all the columns
            Kokkos::parallel_reduce(
               Kokkos::TeamThreadRange(t, ncols_nonlocal),
               [&](const PetscInt j, ReduceData& thread_data) {

                  // Is this column the diagonal
                  bool is_diagonal = (colmap_input_d(device_nonlocal_j[device_nonlocal_i[i] + j]) == i + global_row_start);

                  // We have found a diagonal in this row
                  if (is_diagonal) {
                     thread_data.found_diagonal = true;
                  }                  

                  // If the value is bigger than the tolerance, we keep it
                  if (abs(device_nonlocal_vals[device_nonlocal_i[i] + j]) >= rel_row_tol_d(i)) {
                     // If this is the diagonal and we're dropping all diagonals don't add it
                     if (!(allow_drop_diagonal_int == -1 && is_diagonal)) thread_data.count++;
                  }
                  // Or if it's small but its the diagonal and we're keeping diagonals
                  else if (allow_drop_diagonal_int == 0 && is_diagonal) {
                     thread_data.count++;
                  }                  
               },
               row_result
            );
            
            // We're finished our parallel reduction for this row
            // If we're lumping but there was no diagonal in this row and 
            // there should be (ie !expect_local_diagonal) we'll have to add in a diagonal
            // to the nonlocal block
            // This will add one for every thread in this team, but all 
            // the threads in this team share the same result after the reduction         
            if (lump_int && !expect_local_diagonal && !row_result.found_diagonal) row_result.count++;
            // Only want one thread in the team to write the result
            Kokkos::single(Kokkos::PerTeam(t), [&]() {      
               nnz_match_nonlocal_row_d(i) = row_result.count;
               thread_total += row_result.count;
            });
         },
         nnzs_match_nonlocal
      );

      // ~~~~~~~~~~~~

      // Need to do a scan on nnz_match_nonlocal_row_d to get where each row starts
      Kokkos::parallel_scan (local_rows, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
         // Inclusive scan
         update += nnz_match_nonlocal_row_d(i);         
         if (final) {
            nnz_match_nonlocal_row_d(i) = update; // only update array on final pass
         }
      });               
   }       

   // ~~~~~~~~~~~~~~~~~  
   // We need to assemble our i,j, vals so we can build our matrix
   // ~~~~~~~~~~~~~~~~~
   // Create dual memory on the device and host
   MatScalarKokkosDualView a_local_dual = MatScalarKokkosDualView("a_local_dual", nnzs_match_local);
   MatRowMapKokkosDualView i_local_dual = MatRowMapKokkosDualView("i_local_dual", local_rows+1);
   MatColIdxKokkosDualView j_local_dual = MatColIdxKokkosDualView("j_local_dual", nnzs_match_local);

   // Get device views
   MatScalarKokkosView a_local_d = a_local_dual.view_device();
   MatRowMapKokkosView i_local_d = i_local_dual.view_device();
   // Initialize first entry to zero - the rest get set below
   Kokkos::deep_copy(Kokkos::subview(i_local_d, 0), 0);       
   MatColIdxKokkosView j_local_d = j_local_dual.view_device();  

   // Nonlocal stuff 
   MatScalarKokkosDualView a_nonlocal_dual;
   MatRowMapKokkosDualView i_nonlocal_dual;
   MatColIdxKokkosDualView j_nonlocal_dual;
   MatScalarKokkosView a_nonlocal_d;
   MatRowMapKokkosView i_nonlocal_d;          
   MatColIdxKokkosView j_nonlocal_d;    

   // we also have to go and build the a, i, j for the non-local off-diagonal block
   if (mpi) 
   {
      // Non-local 
      a_nonlocal_dual = MatScalarKokkosDualView("a_nonlocal_dual", nnzs_match_nonlocal);
      i_nonlocal_dual = MatRowMapKokkosDualView("i_nonlocal_dual", local_rows+1);
      j_nonlocal_dual = MatColIdxKokkosDualView("j_nonlocal_dual", nnzs_match_nonlocal);  

      a_nonlocal_d = a_nonlocal_dual.view_device();
      i_nonlocal_d = i_nonlocal_dual.view_device();
      // Initialize first entry to zero - the rest get set below
      Kokkos::deep_copy(Kokkos::subview(i_nonlocal_d, 0), 0);                
      j_nonlocal_d = j_nonlocal_dual.view_device();   
   }               

   // Have to use rangepolicy here rather than teampolicy, as we don't want
   // threads in each team to get their own copies of things like counter
   // Desperately need to rewrite this whole loop to exploit more parallelism!
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(int i) {
         
      PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];
      
      // For this row, would we expect the diagonal to be in the local block or in the nonlocal?
      // Trivially true in the local block for square matrices
      bool expect_local_diagonal = i + global_row_start >= global_col_start && \
                           i + global_row_start < global_col_end_plus_one;

      // The start of our row index comes from the scan
      i_local_d(i + 1) = nnz_match_local_row_d(i);
      PetscInt pos = (i == 0) ? 0 : nnz_match_local_row_d(i-1);

      // Keep track of the lumping
      PetscScalar lump_val = 0;
      
      // Single sequential pass through columns
      for (int j = 0; j < ncols_local; j++) {
         bool keep_col = false;
         bool is_diagonal = (device_local_j[device_local_i[i] + j] + global_col_start == i + global_row_start);

         // If we hit a diagonal put it in the lump'd value
         if (is_diagonal && lump_int) lump_val+= device_local_vals[device_local_i[i] + j];            
         
         // Check if we keep this column because of size
         if (abs(device_local_vals[device_local_i[i] + j]) >= rel_row_tol_d(i)) {
            // If this is the diagonal and we're dropping all diagonals don't add it
            if (!(allow_drop_diagonal_int == -1 && is_diagonal)) keep_col = true;
         }
         // Or if we keep it because we're not dropping diagonals
         else if (allow_drop_diagonal_int == 0 && is_diagonal) {
            keep_col = true;
         }
         
         // Add in this column if we have to keep it
         if (keep_col) {
            j_local_d(pos) = device_local_j[device_local_i[i] + j];
            a_local_d(pos) = device_local_vals[device_local_i[i] + j];
            pos++;
         }
         // If we're not on the diagonal and we're small enough to lump
         // add in
         else if (lump_int && !is_diagonal) {
            lump_val+= device_local_vals[device_local_i[i] + j];
         }
      }  

      // Copy the off-diagonal entries
      if (mpi)
      {
         PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
         // The start of our row index comes from the scan
         i_nonlocal_d(i + 1) = nnz_match_nonlocal_row_d(i);         
         PetscInt pos_nonlocal = (i == 0) ? 0 : nnz_match_nonlocal_row_d(i-1);

         // Single sequential pass through columns
         for (int j = 0; j < ncols_nonlocal; j++) {
            bool keep_col = false;
            // Remember we can have diagonals in the off-diagonal block if we're rectangular
            bool is_diagonal = (colmap_input_d(device_nonlocal_j[device_nonlocal_i[i] + j]) == i + global_row_start);

            // If we hit a diagonal put it in the lump'd value
            if (is_diagonal && lump_int) lump_val+= device_nonlocal_vals[device_nonlocal_i[i] + j];            
            
            // Check if we keep this column because of size
            if (abs(device_nonlocal_vals[device_nonlocal_i[i] + j]) >= rel_row_tol_d(i)) {
               // If this is the diagonal and we're dropping all diagonals don't add it
               if (!(allow_drop_diagonal_int == -1 && is_diagonal)) keep_col = true;
            }
            // Or if we keep it because we're not dropping diagonals
            else if (allow_drop_diagonal_int == 0 && is_diagonal) {
               keep_col = true;
            }
            
            // Let's not add in the column yet, let's do it below
            if (keep_col) {
               j_nonlocal_d(pos_nonlocal) = colmap_input_d(device_nonlocal_j[device_nonlocal_i[i] + j]);
               a_nonlocal_d(pos_nonlocal) = device_nonlocal_vals[device_nonlocal_i[i] + j];
               pos_nonlocal++;               
            }
            // If we're not on the diagonal and we're small enough to lump
            // add in
            else if (lump_int && !is_diagonal) {
               lump_val+= device_nonlocal_vals[device_nonlocal_i[i] + j];
            }
         }   
      }     

      // We have two cases here with lumping
      // One where we have an existing diagonal and one where we don't
      if (lump_int)
      {
         // Find where the diagonal is - this has to happen after we've put in the existing entries
         // because the number of columns isn't necessary ncols_local, its i_local_d(i+1) - i_local_d(i)
         PetscInt diag_index = -1;
         PetscInt before_diag_index = -1;

         // Should we have a diagonal in the local block
         if (expect_local_diagonal) 
         {
            // This is the new ncols_local
            if (i_local_d(i+1) - i_local_d(i) != 0)
            {
               for (int j = 0; j < i_local_d(i+1) - i_local_d(i); j++)
               {
                  // Get the index one before where the diagonal would be
                  if (j_local_d(i_local_d(i) + j) + global_col_start < i + global_row_start) 
                  {
                     before_diag_index = j;
                  }
                  // Get the diagonal index (if it exists)
                  else if (j_local_d(i_local_d(i) + j) + global_col_start == i + global_row_start)
                  {
                     diag_index = j;
                  } 
               }
               // If we have an existing diagonal, replace the value with the lumped value
               if (diag_index != -1)
               {
                  a_local_d(i_local_d(i) + diag_index) = lump_val;
               }
               // If we don't have an existing diagonal, add it in where it would be
               else
               {  
                  for (int j = ncols_local-1; j > before_diag_index; j--)
                  {
                     j_local_d(i_local_d(i) + j + 1) = j_local_d(i_local_d(i) + j);
                     a_local_d(i_local_d(i) + j + 1) = a_local_d(i_local_d(i) + j);                 
                  }            

                  // Has to be the local column index
                  j_local_d(i_local_d(i) + before_diag_index + 1) = i;
                  a_local_d(i_local_d(i) + before_diag_index + 1) = lump_val;
               }
            }
         }
         // If we know the diagonal is in the nonlocal block
         // Shouldn't need the else if (mpi) here
         else if (mpi)
         {
            // This is the new ncols_nonlocal
            if (i_nonlocal_d(i+1) - i_nonlocal_d(i) != 0)
            {
               for (int j = 0; j < i_nonlocal_d(i+1) - i_nonlocal_d(i); j++)
               {
                  // Get the index one before where the diagonal would be
                  // j_nonlocal_d has the global indices in it already
                  if (j_nonlocal_d(i_nonlocal_d(i) + j) < i + global_row_start) 
                  {
                     before_diag_index = j;
                  }
                  // Get the diagonal index (if it exists)
                  else if (j_nonlocal_d(i_nonlocal_d(i) + j) == i + global_row_start)
                  {
                     diag_index = j;
                  } 
               }
               // If we have an existing diagonal, replace the value with the lumped value
               if (diag_index != -1)
               {
                  a_nonlocal_d(i_nonlocal_d(i) + diag_index) = lump_val;
               }
               // If we don't have an existing diagonal, add it in where it would be
               else
               {  
                  for (int j = ncols_local-1; j > before_diag_index; j--)
                  {
                     j_nonlocal_d(i_nonlocal_d(i) + j + 1) = j_nonlocal_d(i_nonlocal_d(i) + j);
                     a_nonlocal_d(i_nonlocal_d(i) + j + 1) = a_nonlocal_d(i_nonlocal_d(i) + j);                 
                  }            

                  // Has to be the global column index
                  j_nonlocal_d(i_nonlocal_d(i) + before_diag_index + 1) = i + global_row_start;
                  a_nonlocal_d(i_nonlocal_d(i) + before_diag_index + 1) = lump_val;
               }
            }
         }            
      }      
   });

   // Have to specify that we've modified the device data
   a_local_dual.modify_device();
   i_local_dual.modify_device();
   j_local_dual.modify_device();

   // We can create our local diagonal block matrix directly on the device
   // See MatSeqAIJKokkosMergeMats for example
   auto akok_local = new Mat_SeqAIJKokkos(local_rows, local_cols, nnzs_match_local, i_local_dual, j_local_dual, a_local_dual);    
   // The equivalent of calling the internal MatCreateSeqAIJKokkosWithCSRMatrix
   MatCreate(PETSC_COMM_SELF, &output_mat_local);
   // Why isn't this publically available??
   MatSetSeqAIJKokkosWithCSRMatrix_mine(output_mat_local, akok_local);   

   // we also have to go and build the a, i, j for the non-local off-diagonal block
   if (mpi) 
   {
      // Have to specify that we've modified the device data
      a_nonlocal_dual.modify_device();
      i_nonlocal_dual.modify_device();
      j_nonlocal_dual.modify_device();

      // Now we need to build garray on the host and rewrite the j_nonlocal_d indices so they are local
      // The default values here are for the case where we 
      // let do it, it resets this internally in MatSetUpMultiply_MPIAIJ
      PetscInt *garray_host = NULL;
      PetscInt col_ao_output = global_cols;
      if (cols_ao == 0)
      {
         // Silly but depending on the compiler this may return a non-null pointer
         col_ao_output = 0;
         PetscMalloc1(col_ao_output, &garray_host);
      }

      // We can use the Kokkos::UnorderedMap to do this if our 
      // off diagonal block has fewer than 4 billion non-zero columns (max capacity of uint32_t)
      // Otherwise we can just tell petsc to do do it on the host (in MatSetUpMultiply_MPIAIJ)
      // and rely on the hash tables in petsc on the host which can handle more than 4 billion entries
      // We trigger petsc doing it by passing in null as garray_host to MatSetMPIAIJKokkosWithSplitSeqAIJKokkosMatrices
      // If we have no off-diagonal entries (either we started with zero or we've dropped them all)
      // just skip all this and leave garray_host as null

      // If we have 4 bit ints, we know cols_ao can never be bigger than the capacity of uint32_t
      bool size_small_enough = sizeof(PetscInt) == 4 || \
                  (sizeof(PetscInt) > 4 && cols_ao < 4294967295);
      if (size_small_enough && cols_ao > 0 && nnzs_match_nonlocal > 0)
      {
         // Have to tell it the max capacity, we know we will have no more 
         // than the input off-diag columns
         Kokkos::UnorderedMap<PetscInt, PetscInt> hashmap((uint32_t)(cols_ao+1));

         // Let's insert all the existing global col indices as keys (with no value to start)
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, nnzs_match_nonlocal), KOKKOS_LAMBDA(int i) {      
            
            // Insert the key (global col indices) without a value
            // Duplicates will be ignored
            hashmap.insert(j_nonlocal_d(i));
         });

         // We now know how many unique global columns we have
         col_ao_output = hashmap.size();

         // Tag which of the original garray stick around  
         PetscIntKokkosView colmap_output_d_big("colmap_output_d_big", cols_ao);
         Kokkos::deep_copy(colmap_output_d_big, colmap_input_d);                

         // Mark which of the keys don't exist
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, cols_ao), KOKKOS_LAMBDA(int i) { 

            // If the key doesn't exist set the global index to -1
            if (!hashmap.exists(colmap_output_d_big(i))) colmap_output_d_big(i) = -1; 
         });         

         // Now sort the global columns indices
         // All the -1 should be at the start
         Kokkos::sort(colmap_output_d_big);

         // Count the number of -1 - this will be the index of the first entry
         // that isn't -1
         // It should never be equal to start index, because otherwise we
         // have dropped all nonlocal entries
         auto &exec = PetscGetKokkosExecutionSpace();
         PetscInt start_index = Kokkos::Experimental::count(exec, colmap_output_d_big, -1);

         // Our final colmap_output_d is colmap_output_d_big(start_index:end)
         PetscIntKokkosView colmap_output_d = Kokkos::subview(colmap_output_d_big, \
                  Kokkos::make_pair(start_index, cols_ao));

         // Now we can clear the hash and instead stick in the global indices
         // but now with the local indices as values
         hashmap.clear();
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, colmap_output_d.extent(0)), KOKKOS_LAMBDA(int i) { 

            hashmap.insert(colmap_output_d(i), i);
         });          

         // And now we can overwrite j_nonlocal_d with the local indices
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, nnzs_match_nonlocal), KOKKOS_LAMBDA(int i) {     

            // Find where our global col index is at
            uint32_t loc = hashmap.find(j_nonlocal_d(i));
            // And get the value (the new local index)
            j_nonlocal_d(i) = hashmap.value_at(loc);
         });      
         hashmap.clear();

         // Create some host space for the output garray (that stays in scope) and copy it
         PetscMalloc1(colmap_output_d.extent(0), &garray_host);
         PetscIntKokkosViewHost colmap_output_h = PetscIntKokkosViewHost(garray_host, colmap_output_d.extent(0));
         Kokkos::deep_copy(colmap_output_h, colmap_output_d);   
      }

      // We can create our nonlocal diagonal block matrix directly on the device
      auto akok_nonlocal = new Mat_SeqAIJKokkos(local_rows, col_ao_output, \
               nnzs_match_nonlocal, i_nonlocal_dual, j_nonlocal_dual, a_nonlocal_dual);  
      // The equivalent of calling the internal MatCreateSeqAIJKokkosWithCSRMatrix
      MatCreate(PETSC_COMM_SELF, &output_mat_nonlocal);
      // Why isn't this publically available??
      MatSetSeqAIJKokkosWithCSRMatrix_mine(output_mat_nonlocal, akok_nonlocal);    

      // Build our mpi kokkos matrix by passing in the local and 
      // nonlocal kokkos matrices and the colmap
      // If you read the description of MatSetMPIAIJKokkosWithSplitSeqAIJKokkosMatrices it says 
      // B     - the offdiag matrix using global col ids
      // but reading the code, if you pass in garray as not null
      // then the column id's of B should be the local indices,
      // as when it calls MatAssemblyEnd of the mpikokkos, it calls the MatAssemblyEnd of the aijmpi matrix
      // which then calls MatSetUpMultiply_MPIAIJ
      // and looking at that it only goes and builds garray and compactifies (and turns indices to local)
      // in B if garray is null 
      MatCreate(MPI_COMM_MATRIX, output_mat);
      // Only have to set the size * type in the mpi case, the serial case it gets set in 
      // MatSetSeqAIJKokkosWithCSRMatrix
      MatSetSizes(*output_mat, local_rows, local_cols, global_rows, global_cols);
      PetscLayoutSetUp((*output_mat)->rmap);
      PetscLayoutSetUp((*output_mat)->cmap);
      MatSetType(*output_mat, mat_type);
      // Why isn't this publically available??
      MatSetMPIAIJKokkosWithSplitSeqAIJKokkosMatrices_mine(*output_mat, output_mat_local, output_mat_nonlocal, garray_host);

   }     
   // If in serial 
   else
   {
      *output_mat = output_mat_local;
   }

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Stick W in a full sized P but with kokkos - keeping everything on the device
PETSC_INTERN void compute_P_from_W_kokkos(Mat *input_mat, PetscInt global_row_start, IS *is_fine, \
                  IS *is_coarse, int identity_int, int reuse_int, Mat *output_mat)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt global_row_start_W, global_row_end_plus_one_W;
   PetscInt global_col_start_W, global_col_end_plus_one_W;
   PetscInt local_rows_coarse, local_rows, local_cols, local_cols_coarse;
   PetscInt cols_z, rows_z, local_rows_fine, global_cols_coarse;
   PetscInt rows_ao, cols_ao, global_cols, global_rows;
   MatType mat_type;
   PetscInt nnzs_match_local, nnzs_match_nonlocal;
   Mat output_mat_local, output_mat_nonlocal;

   MatGetType(*input_mat, &mat_type);
   // Are we in parallel?
   bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat_MPIAIJ *mat_mpi = nullptr;
   Mat mat_local, mat_nonlocal;

   PetscIntKokkosViewHost colmap_input_h;
   PetscIntKokkosView colmap_input_d;   
   if (mpi)
   {
      mat_mpi = (Mat_MPIAIJ *)(*input_mat)->data;
      mat_local = mat_mpi->A;
      mat_nonlocal = mat_mpi->B;
      MatGetSize(mat_nonlocal, &rows_ao, &cols_ao); 

      // We also copy the input mat colmap over to the device as we need it
      // Don't actually need on the device for this routine
      // colmap_input_h = PetscIntKokkosViewHost(mat_mpi->garray, cols_ao);
      //colmap_input_d = PetscIntKokkosView("colmap_input_d", cols_ao);
      //Kokkos::deep_copy(colmap_input_d, colmap_input_h);        
   }
   else
   {
      mat_local = *input_mat;
   }

   MatGetOwnershipRange(*input_mat, &global_row_start_W, &global_row_end_plus_one_W);  
   MatGetOwnershipRangeColumn(*input_mat, &global_col_start_W, &global_col_end_plus_one_W);                  

   MatGetSize(*input_mat, &cols_z, &rows_z);

   // Get pointers to the indices on the host
   const PetscInt *fine_indices_ptr, *coarse_indices_ptr;
   ISGetIndices(*is_fine, &fine_indices_ptr);   
   ISGetIndices(*is_coarse, &coarse_indices_ptr); 

   ISGetLocalSize(*is_coarse, &local_rows_coarse);
   ISGetLocalSize(*is_fine, &local_rows_fine);   

   // Create a host view of the existing indices
   auto fine_view_h = PetscIntConstKokkosViewHost(fine_indices_ptr, local_rows_fine);    
   auto fine_view_d = PetscIntKokkosView("fine_view_d", local_rows_fine);   
   auto coarse_view_h = PetscIntConstKokkosViewHost(coarse_indices_ptr, local_rows_coarse);    
   auto coarse_view_d = PetscIntKokkosView("coarse_view_d", local_rows_coarse);      
   // Copy indices to the device
   Kokkos::deep_copy(fine_view_d, fine_view_h);     
   Kokkos::deep_copy(coarse_view_d, coarse_view_h);      

   local_cols_coarse = local_rows_coarse;
   local_cols = local_rows_coarse + local_rows_fine;
   local_rows = local_cols; 
   
   global_cols = rows_z + cols_z;
   global_rows = global_cols;
   //global_rows_coarse = rows_z;
   global_cols_coarse = rows_z;    

   // Get the comm
   PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX);

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr, *device_nonlocal_vals = nullptr;  
   MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype);  
   if (mpi) MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype);          

   PetscIntKokkosView nnz_match_local_row_d;
   PetscIntKokkosView nnz_match_nonlocal_row_d;
   MatScalarKokkosDualView a_local_dual;
   MatRowMapKokkosDualView i_local_dual;
   MatColIdxKokkosDualView j_local_dual;

   // Get device views
   MatScalarKokkosView a_local_d;
   MatRowMapKokkosView i_local_d;  
   MatColIdxKokkosView j_local_d;    

   // Nonlocal stuff 
   MatScalarKokkosDualView a_nonlocal_dual;
   MatRowMapKokkosDualView i_nonlocal_dual;
   MatColIdxKokkosDualView j_nonlocal_dual;
   MatScalarKokkosView a_nonlocal_d;
   MatRowMapKokkosView i_nonlocal_d;          
   MatColIdxKokkosView j_nonlocal_d;  

   Mat_MPIAIJ *mat_mpi_output = nullptr;
   Mat mat_local_output, mat_nonlocal_output;   

   // Only need things to do with the sparsity pattern if we're not reusing
   if (!reuse_int)
   {
      // ~~~~~~~~~~~~
      // Get the number of nnzs
      // ~~~~~~~~~~~~
      nnzs_match_local = 0;
      nnzs_match_nonlocal = 0;

      // ~~~~~~~~~~~~~~~~~~~~~~~
      // Let's build our i, j, and a on the device
      // ~~~~~~~~~~~~~~~~~~~~~~~ 
      // We need to know how many entries are in each row 
      nnz_match_local_row_d = PetscIntKokkosView("nnz_match_local_row_d", local_rows);    
      // We may have identity
      Kokkos::deep_copy(nnz_match_local_row_d, 0);         
      if (mpi) nnz_match_nonlocal_row_d = PetscIntKokkosView("nnz_match_nonlocal_row_d", local_rows);                  

      // ~~~~~~~~~~~~
      // Need to count the number of nnzs we end up with, on each row and in total
      // ~~~~~~~~~~~~
      // Loop over the rows of W
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows_fine), KOKKOS_LAMBDA(int i) {

            // Convert to global fine index into a local index in the full matrix
            PetscInt row_index = fine_view_d(i) - global_row_start;
            // Still using i here (the local index into W)
            PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];
            nnz_match_local_row_d(row_index) = ncols_local;

            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
               nnz_match_nonlocal_row_d(row_index) = ncols_nonlocal;
            }
      });

      // Loop over all the C points - we know they're in the local block
      if (identity_int) 
      {
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, local_rows_coarse), KOKKOS_LAMBDA(int i) {

            // Convert to global coarse index into a local index into the full matrix
            PetscInt row_index = coarse_view_d(i) - global_row_start;
            nnz_match_local_row_d(row_index)++;
         }); 
      }  

      // Get number of nnzs
      Kokkos::parallel_reduce ("ReductionLocal", local_rows, KOKKOS_LAMBDA (const int i, PetscInt& update) {
         update += nnz_match_local_row_d(i); 
      }, nnzs_match_local);   
      if (mpi)
      {
         Kokkos::parallel_reduce ("ReductionNonLocal", local_rows, KOKKOS_LAMBDA (const int i, PetscInt& update) {
            update += nnz_match_nonlocal_row_d(i); 
         }, nnzs_match_nonlocal);       
      }

      // ~~~~~~~~~~~~

      // Need to do a scan on nnz_match_local_row_d to get where each row starts
      Kokkos::parallel_scan (local_rows, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
         // Inclusive scan
         update += nnz_match_local_row_d(i);         
         if (final) {
            nnz_match_local_row_d(i) = update; // only update array on final pass
         }
      });      
      if (mpi)
      { 
         // Need to do a scan on nnz_match_nonlocal_row_d to get where each row starts
         Kokkos::parallel_scan (local_rows, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
            // Inclusive scan
            update += nnz_match_nonlocal_row_d(i);         
            if (final) {
               nnz_match_nonlocal_row_d(i) = update; // only update array on final pass
            }
         });               
      }           

      // ~~~~~~~~~~~~~~~~~  
      // We need to assemble our i,j, vals so we can build our matrix
      // ~~~~~~~~~~~~~~~~~
      // Create dual memory on the device and host
      a_local_dual = MatScalarKokkosDualView("a_local_dual", nnzs_match_local);
      i_local_dual = MatRowMapKokkosDualView("i_local_dual", local_rows+1);
      j_local_dual = MatColIdxKokkosDualView("j_local_dual", nnzs_match_local);

      // Get device views
      a_local_d = a_local_dual.view_device();
      i_local_d = i_local_dual.view_device();
      // Initialize first entry to zero - the rest get set below
      Kokkos::deep_copy(Kokkos::subview(i_local_d, 0), 0);       
      j_local_d = j_local_dual.view_device();    

      // we also have to go and build the a, i, j for the non-local off-diagonal block
      if (mpi) 
      {
         // Non-local 
         a_nonlocal_dual = MatScalarKokkosDualView("a_nonlocal_dual", nnzs_match_nonlocal);
         i_nonlocal_dual = MatRowMapKokkosDualView("i_nonlocal_dual", local_rows+1);
         j_nonlocal_dual = MatColIdxKokkosDualView("j_nonlocal_dual", nnzs_match_nonlocal);  

         a_nonlocal_d = a_nonlocal_dual.view_device();
         i_nonlocal_d = i_nonlocal_dual.view_device();
         // Initialize first entry to zero - the rest get set below
         Kokkos::deep_copy(Kokkos::subview(i_nonlocal_d, 0), 0);                
         j_nonlocal_d = j_nonlocal_dual.view_device();   
      }  

      // ~~~~~~~~~~~~~~~
      // Have to build i_local_d and the nonlocal for every row (F and C points)
      // regardless of if we are sticking 1 in (ie identity)
      // This has to happen before the main loop as the f and c 
      // points are placed in different orders (ie not in order as the index 
      // is row_index 
      // ~~~~~~~~~~~~~~~
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows_fine), KOKKOS_LAMBDA(int i) {

            // Convert to global fine index into a local index in the full matrix
            PetscInt row_index = fine_view_d(i) - global_row_start;       

            // The start of our row index comes from the scan
            i_local_d(row_index + 1) = nnz_match_local_row_d(row_index);   
            if (mpi) i_nonlocal_d(row_index + 1) = nnz_match_nonlocal_row_d(row_index);         
      });            

      // Always have to set the i_local_d for C points, regardless of if we are setting
      // 1 in the identity part for them
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows_coarse), KOKKOS_LAMBDA(int i) {

         // Convert to global coarse index into a local index into the full matrix
         PetscInt row_index = coarse_view_d(i) - global_row_start;

         // The start of our row index comes from the scan
         i_local_d(row_index + 1) = nnz_match_local_row_d(row_index);
         if (mpi) i_nonlocal_d(row_index + 1) = nnz_match_nonlocal_row_d(row_index);        

      });  

      // Loop over the rows of W
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows_fine, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            PetscInt i = t.league_rank();

            // Convert to global fine index into a local index in the full matrix
            PetscInt row_index = fine_view_d(i) - global_row_start;
            // Still using i here (the local index into W)
            PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];         

            // For over local columns - copy in W
            Kokkos::parallel_for(
               Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

               j_local_d(i_local_d(row_index) + j) = device_local_j[device_local_i[i] + j];
               a_local_d(i_local_d(row_index) + j) = device_local_vals[device_local_i[i] + j];
                     
            });     

            // For over nonlocal columns - copy in W
            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];         

               Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {

                  // We keep the existing local indices in the off-diagonal block here
                  // we have all the same columns as W and hence the same garray
                  j_nonlocal_d(i_nonlocal_d(row_index) + j) = device_nonlocal_j[device_nonlocal_i[i] + j];
                  a_nonlocal_d(i_nonlocal_d(row_index) + j) = device_nonlocal_vals[device_nonlocal_i[i] + j];
                        
               });          
            }
      }); 
   }
   // If we're reusing, we can just write directly to the existing views
   else
   {
      // Get the existing output mats
      if (mpi)
      {
         mat_mpi_output = (Mat_MPIAIJ *)(*output_mat)->data;
         mat_local_output = mat_mpi_output->A;
         mat_nonlocal_output = mat_mpi_output->B;     
      }
      else
      {
         mat_local_output = *output_mat;
      }     
      Mat_SeqAIJKokkos *aijkok_local_output = static_cast<Mat_SeqAIJKokkos *>(mat_local_output->spptr);
      Mat_SeqAIJKokkos *aijkok_nonlocal_output;
      if (mpi) aijkok_nonlocal_output = static_cast<Mat_SeqAIJKokkos *>(mat_nonlocal_output->spptr);

      // Annoying we can't just call MatSeqAIJGetKokkosView
      a_local_d = aijkok_local_output->a_dual.view_device();
      if (mpi) a_nonlocal_d = aijkok_nonlocal_output->a_dual.view_device();

      // Annoyingly there isn't currently the ability to get views for i (or j)
      const PetscInt *device_local_i_output = nullptr, *device_nonlocal_i_ouput = nullptr;
      PetscMemType mtype;
      MatSeqAIJGetCSRAndMemType(mat_local_output, &device_local_i_output, NULL, NULL, &mtype);  
      if (mpi) MatSeqAIJGetCSRAndMemType(mat_nonlocal_output, &device_nonlocal_i_ouput, NULL, NULL, &mtype);  

      // Have these point at the existing i pointers - we don't need j if we're reusing
      ConstMatRowMapKokkosView i_local_const_d = ConstMatRowMapKokkosView(device_local_i_output, local_rows+1);
      ConstMatRowMapKokkosView i_nonlocal_const_d;
      if (mpi) i_nonlocal_const_d = ConstMatRowMapKokkosView(device_nonlocal_i_ouput, local_rows+1);        

      // Only have to write W as the identity block cannot change
      // Loop over the rows of W - annoying we have const views as this is just the same loop as above
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows_fine, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            PetscInt i = t.league_rank();

            // Convert to global fine index into a local index in the full matrix
            PetscInt row_index = fine_view_d(i) - global_row_start;
            // Still using i here (the local index into W)
            PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];         

            // For over local columns - copy in W
            Kokkos::parallel_for(
               Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

               a_local_d(i_local_const_d(row_index) + j) = device_local_vals[device_local_i[i] + j];
                     
            });     

            // For over nonlocal columns - copy in W
            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];         

               Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {

                  // We keep the existing local indices in the off-diagonal blocl here
                  // we have all the same columns as W and hence the same garray
                  a_nonlocal_d(i_nonlocal_const_d(row_index) + j) = device_nonlocal_vals[device_nonlocal_i[i] + j];
                        
               });          
            }
      });   

      // Have to specify we've modifed data on the device
      // Want to call MatSeqAIJKokkosModifyDevice but its PETSC_INTERN
      aijkok_local_output->a_dual.clear_sync_state();
      aijkok_local_output->a_dual.modify_device();
      aijkok_local_output->transpose_updated = PETSC_FALSE;
      aijkok_local_output->hermitian_updated = PETSC_FALSE;
      // Invalidate diagonals
      Mat_SeqAIJ *a = (Mat_SeqAIJ *)mat_local_output->data;
      a->idiagvalid  = PETSC_FALSE;
      a->ibdiagvalid = PETSC_FALSE;      
      a->inode.ibdiagvalid = PETSC_FALSE;        
      if (mpi)
      {
         aijkok_nonlocal_output->a_dual.clear_sync_state();
         aijkok_nonlocal_output->a_dual.modify_device();
         aijkok_nonlocal_output->transpose_updated = PETSC_FALSE;
         aijkok_nonlocal_output->hermitian_updated = PETSC_FALSE;
         a = (Mat_SeqAIJ *)mat_nonlocal_output->data;
         a->idiagvalid  = PETSC_FALSE;
         a->ibdiagvalid = PETSC_FALSE;   
         a->inode.ibdiagvalid = PETSC_FALSE; 
      }      
      PetscObjectStateIncrease((PetscObject)(*output_mat));    

   }

   // ~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~   

   if (!reuse_int)
   {
      // Loop over all the C points - we know they're in the local block
      if (identity_int) 
      {
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, local_rows_coarse), KOKKOS_LAMBDA(int i) {

            // Convert to global coarse index into a local index into the full matrix
            PetscInt row_index = coarse_view_d(i) - global_row_start;

            // Only a single column
            j_local_d(i_local_d(row_index)) = i;
            a_local_d(i_local_d(row_index)) = 1.0;         

         }); 
      }   

      // Have to specify that we've modified the device data
      a_local_dual.modify_device();
      i_local_dual.modify_device();
      j_local_dual.modify_device();      
    
      // We can create our local diagonal block matrix directly on the device
      // See MatSeqAIJKokkosMergeMats for example
      auto akok_local = new Mat_SeqAIJKokkos(local_rows, local_cols_coarse, nnzs_match_local, i_local_dual, j_local_dual, a_local_dual);    
      // The equivalent of calling the internal MatCreateSeqAIJKokkosWithCSRMatrix
      MatCreate(PETSC_COMM_SELF, &output_mat_local);
      // Why isn't this publically available??
      MatSetSeqAIJKokkosWithCSRMatrix_mine(output_mat_local, akok_local);   

      // we also have to go and build the a, i, j for the non-local off-diagonal block
      if (mpi) 
      {
         // Have to specify that we've modified the device data
         a_nonlocal_dual.modify_device();
         i_nonlocal_dual.modify_device();
         j_nonlocal_dual.modify_device();

         // We can create our nonlocal diagonal block matrix directly on the device
         // Same number of col_ao as W
         auto akok_nonlocal = new Mat_SeqAIJKokkos(local_rows, cols_ao, \
                  nnzs_match_nonlocal, i_nonlocal_dual, j_nonlocal_dual, a_nonlocal_dual);  
         // The equivalent of calling the internal MatCreateSeqAIJKokkosWithCSRMatrix
         MatCreate(PETSC_COMM_SELF, &output_mat_nonlocal);
         // Why isn't this publically available??
         MatSetSeqAIJKokkosWithCSRMatrix_mine(output_mat_nonlocal, akok_nonlocal);   

         // We just take a copy of the original garray
         PetscInt *garray_host = NULL; 
         PetscMalloc1(cols_ao, &garray_host);
         for (int i = 0; i < cols_ao; i++)
         {
            garray_host[i] = mat_mpi->garray[i];
         }

         // Build our mpi kokkos matrix by passing in the local and 
         // nonlocal kokkos matrices and the colmap
         // If you read the description of MatSetMPIAIJKokkosWithSplitSeqAIJKokkosMatrices it says 
         // B     - the offdiag matrix using global col ids
         // but reading the code, if you pass in garray as not null
         // then the column id's of B should be the local indices,
         // as when it calls MatAssemblyEnd of the mpikokkos, it calls the MatAssemblyEnd of the aijmpi matrix
         // which then calls MatSetUpMultiply_MPIAIJ
         // and looking at that it only goes and builds garray and compactifies (and turns indices to local)
         // in B if garray is null 
         MatCreate(MPI_COMM_MATRIX, output_mat);
         // Only have to set the size * type in the mpi case, the serial case it gets set in 
         // MatSetSeqAIJKokkosWithCSRMatrix
         MatSetSizes(*output_mat, local_rows, local_cols_coarse, global_rows, global_cols_coarse);
         PetscLayoutSetUp((*output_mat)->rmap);
         PetscLayoutSetUp((*output_mat)->cmap);
         MatSetType(*output_mat, mat_type);
         // The garray is the same as the W
         // Why isn't this publically available??
         MatSetMPIAIJKokkosWithSplitSeqAIJKokkosMatrices_mine(*output_mat, output_mat_local, output_mat_nonlocal, garray_host);

      }     
      // If in serial 
      else
      {
         *output_mat = output_mat_local;
      }
   }

   ISRestoreIndices(*is_fine, &fine_indices_ptr);
   ISRestoreIndices(*is_coarse, &coarse_indices_ptr); 

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Set all the values of the matrix to val
PETSC_INTERN void MatSetAllValues_kokkos(Mat *input_mat, PetscReal val)
{
   MatType mat_type;

   MatGetType(*input_mat, &mat_type);
   // Are we in parallel?
   bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat_MPIAIJ *mat_mpi = nullptr;
   Mat mat_local, mat_nonlocal;
  
   if (mpi)
   {
      mat_mpi = (Mat_MPIAIJ *)(*input_mat)->data;
      mat_local = mat_mpi->A;
      mat_nonlocal = mat_mpi->B;
   }
   else
   {
      mat_local = *input_mat;
   }
   PetscInt local_rows, local_cols;
   MatGetLocalSize(*input_mat, &local_rows, &local_cols);

   Mat_SeqAIJKokkos *aijkok_nonlocal;
   Mat_SeqAIJKokkos *aijkok_local = static_cast<Mat_SeqAIJKokkos *>(mat_local->spptr);
   if(mpi) aijkok_nonlocal = static_cast<Mat_SeqAIJKokkos *>(mat_nonlocal->spptr);
   
   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr, *device_nonlocal_vals = nullptr;  
   MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype);  
   if (mpi) MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype);          

   MatScalarKokkosView a_local_d, a_nonlocal_d;
   a_local_d = PetscScalarKokkosView(device_local_vals, aijkok_local->csrmat.nnz());   
   if (mpi) a_nonlocal_d = PetscScalarKokkosView(device_nonlocal_vals, aijkok_nonlocal->csrmat.nnz()); 
   // Copy in the val
   Kokkos::deep_copy(a_local_d, val); 
   if (mpi) Kokkos::deep_copy(a_nonlocal_d, val); 

   // Have to specify we've modifed data on the device
   // Want to call MatSeqAIJKokkosModifyDevice but its PETSC_INTERN

   aijkok_local->a_dual.clear_sync_state();
   aijkok_local->a_dual.modify_device();
   aijkok_local->transpose_updated = PETSC_FALSE;
   aijkok_local->hermitian_updated = PETSC_FALSE;
   // Invalidate diagonals
   Mat_SeqAIJ *a = (Mat_SeqAIJ *)mat_local->data;
   a->idiagvalid  = PETSC_FALSE;
   a->ibdiagvalid = PETSC_FALSE;      
   a->inode.ibdiagvalid = PETSC_FALSE;  

   if (mpi)
   {
      aijkok_nonlocal->a_dual.clear_sync_state();
      aijkok_nonlocal->a_dual.modify_device();
      aijkok_nonlocal->transpose_updated = PETSC_FALSE;
      aijkok_nonlocal->hermitian_updated = PETSC_FALSE;
      a = (Mat_SeqAIJ *)mat_nonlocal->data;
      a->idiagvalid  = PETSC_FALSE;
      a->ibdiagvalid = PETSC_FALSE;   
      a->inode.ibdiagvalid = PETSC_FALSE;    
   }
   PetscObjectStateIncrease((PetscObject)(*input_mat));

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Generate one point classical prolongator but with kokkos - keeping everything on the device
PETSC_INTERN void generate_one_point_with_one_entry_from_sparse_kokkos(Mat *input_mat, Mat *output_mat)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt global_col_start, global_col_end_plus_one;
   PetscInt rows_ao, cols_ao;
   MatType mat_type;
   PetscInt nnzs_match_local, nnzs_match_nonlocal;
   Mat output_mat_local, output_mat_nonlocal;

   MatGetType(*input_mat, &mat_type);
   // Are we in parallel?
   bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat_MPIAIJ *mat_mpi = nullptr;
   Mat mat_local, mat_nonlocal;

   PetscIntKokkosViewHost colmap_input_h;
   PetscIntKokkosView colmap_input_d;   
   if (mpi)
   {
      mat_mpi = (Mat_MPIAIJ *)(*input_mat)->data;
      mat_local = mat_mpi->A;
      mat_nonlocal = mat_mpi->B;
      MatGetSize(mat_nonlocal, &rows_ao, &cols_ao); 

      // We also copy the input mat colmap over to the device as we need it
      colmap_input_h = PetscIntKokkosViewHost(mat_mpi->garray, cols_ao);
      colmap_input_d = PetscIntKokkosView("colmap_input_d", cols_ao);
      Kokkos::deep_copy(colmap_input_d, colmap_input_h);        
   }
   else
   {
      mat_local = *input_mat;
   }

   // Get the comm
   PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX);
   MatGetLocalSize(*input_mat, &local_rows, &local_cols);
   MatGetSize(*input_mat, &global_rows, &global_cols);
   // This returns the global index of the local portion of the matrix
   MatGetOwnershipRange(*input_mat, &global_row_start, &global_row_end_plus_one);
   MatGetOwnershipRangeColumn(*input_mat, &global_col_start, &global_col_end_plus_one);   

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr, *device_nonlocal_vals = nullptr;  
   MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype);  
   if (mpi) MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype);          

   // ~~~~~~~~~~~~
   // Get the number of nnzs
   // ~~~~~~~~~~~~
   nnzs_match_local = 0;
   nnzs_match_nonlocal = 0;

   // ~~~~~~~~~~~~~~~~~~~~~~~
   // Let's build our i, j, and a on the device
   // ~~~~~~~~~~~~~~~~~~~~~~~   
   // We need to know where our max values are
   PetscIntKokkosView max_col_row_d("max_col_row_d", local_rows);    
   // We need to know how many entries are in each row  
   PetscIntKokkosView nnz_match_local_row_d("nnz_match_local_row_d", local_rows);             
   Kokkos::deep_copy(nnz_match_local_row_d, 0);
   PetscIntKokkosView nnz_match_nonlocal_row_d;
   if (mpi) 
   {
      nnz_match_nonlocal_row_d = PetscIntKokkosView("nnz_match_nonlocal_row_d", local_rows); 
      Kokkos::deep_copy(nnz_match_nonlocal_row_d, 0);
   }

   // Loop over the rows and find the biggest entry in each row
   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

      PetscInt i   = t.league_rank(); // row i
      PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];

      // We have a custom reduction type defined - ReduceDataMaxRow
      ReduceDataMaxRow local_row_result, nonlocal_row_result;

      // Reduce over all the columns
      Kokkos::parallel_reduce(
         Kokkos::TeamThreadRange(t, ncols_local),
         [&](const PetscInt j, ReduceDataMaxRow& thread_data) {

            // If it's the biggest value keep it
            if (abs(device_local_vals[device_local_i[i] + j]) > thread_data.val) {
               thread_data.val = abs(device_local_vals[device_local_i[i] + j]);
               thread_data.col = device_local_j[device_local_i[i] + j];
            }
         }, local_row_result
      );

      if (mpi)
      {
         PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
         Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(t, ncols_nonlocal),
            [&](const PetscInt j, ReduceDataMaxRow& thread_data) {

               // If it's the biggest value keep it
               if (abs(device_nonlocal_vals[device_nonlocal_i[i] + j]) > thread_data.val) {
                  thread_data.val = abs(device_nonlocal_vals[device_nonlocal_i[i] + j]);
                  // Set the global index
                  thread_data.col = colmap_input_d(device_nonlocal_j[device_nonlocal_i[i] + j]);
               }
            }, nonlocal_row_result
         );         
      }

      // Only want one thread in the team to write the result
      Kokkos::single(Kokkos::PerTeam(t), [&]() {     

         // We know the entry is local
         if (!mpi)
         {
            // Check we found an entry
            if (local_row_result.col != -1) {
               max_col_row_d(i) = local_row_result.col;
               nnz_match_local_row_d(i)++;
            }
         }
         // If we have mpi we have to check both the local
         // and nonlocal block maxs
         else
         {
            // If our biggest entry is nonlocal
            if (nonlocal_row_result.val > local_row_result.val) {
               // Check we found an entry
               if (nonlocal_row_result.col != -1) {
                  max_col_row_d(i) = nonlocal_row_result.col;
                  nnz_match_nonlocal_row_d(i)++;
               }
            }
            // The local entry is the biggest
            else if (nonlocal_row_result.val < local_row_result.val) {
                  // Check we found an entry
                  if (local_row_result.col != -1) {
                     max_col_row_d(i) = local_row_result.col;
                     nnz_match_local_row_d(i)++;
                  }
            }        
            // If they are equal - let's check they're valid to start
            else if (local_row_result.col != -1 && nonlocal_row_result.col != -1)
            {
               // We want to match the same as the cpu results, which 
               // uses matgetrow and then finds the first max global index
               // This means they are sorted
               // Would be a nicer thing to always pick the local entry
               if (local_row_result.col + global_col_start < nonlocal_row_result.col) {
                  max_col_row_d(i) = local_row_result.col;
                  nnz_match_local_row_d(i)++;
               }
               else {
                  max_col_row_d(i) = nonlocal_row_result.col;
                  nnz_match_nonlocal_row_d(i)++;
               }
            }    
         }
      });      
   });      

   // Get number of nnzs
   Kokkos::parallel_reduce ("ReductionLocal", local_rows, KOKKOS_LAMBDA (const int i, PetscInt& update) {
      update += nnz_match_local_row_d(i); 
   }, nnzs_match_local);   
   if (mpi)
   {
      Kokkos::parallel_reduce ("ReductionNonLocal", local_rows, KOKKOS_LAMBDA (const int i, PetscInt& update) {
         update += nnz_match_nonlocal_row_d(i); 
      }, nnzs_match_nonlocal);       
   }   

   // ~~~~~~~~~~~~

   // Store original counts before scan
   PetscIntKokkosView has_entry_local_d("has_entry_local_d", local_rows);
   Kokkos::deep_copy(has_entry_local_d, nnz_match_local_row_d); 
   PetscIntKokkosView has_entry_nonlocal_d;
   if (mpi)
   {
      has_entry_nonlocal_d = PetscIntKokkosView ("has_entry_nonlocal_d", local_rows);
      Kokkos::deep_copy(has_entry_nonlocal_d, nnz_match_nonlocal_row_d);
   }  

   // Need to do a scan on nnz_match_local_row_d to get where each row starts
   Kokkos::parallel_scan (local_rows, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
      // Inclusive scan
      update += nnz_match_local_row_d(i);         
      if (final) {
         nnz_match_local_row_d(i) = update; // only update array on final pass
      }
   });            

   if (mpi) 
   {
      // Need to do a scan on nnz_match_nonlocal_row_d to get where each row starts
      Kokkos::parallel_scan (local_rows, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
         // Inclusive scan
         update += nnz_match_nonlocal_row_d(i);         
         if (final) {
            nnz_match_nonlocal_row_d(i) = update; // only update array on final pass
         }
      });               
   }       

   // ~~~~~~~~~~~~~~~~~  
   // We need to assemble our i,j, vals so we can build our matrix
   // ~~~~~~~~~~~~~~~~~
   // Create dual memory on the device and host
   MatScalarKokkosDualView a_local_dual = MatScalarKokkosDualView("a_local_dual", nnzs_match_local);
   MatRowMapKokkosDualView i_local_dual = MatRowMapKokkosDualView("i_local_dual", local_rows+1);
   MatColIdxKokkosDualView j_local_dual = MatColIdxKokkosDualView("j_local_dual", nnzs_match_local);

   // Get device views
   MatScalarKokkosView a_local_d = a_local_dual.view_device();
   MatRowMapKokkosView i_local_d = i_local_dual.view_device();
   // Initialize first entry to zero - the rest get set below
   Kokkos::deep_copy(Kokkos::subview(i_local_d, 0), 0);       
   MatColIdxKokkosView j_local_d = j_local_dual.view_device();  

   // Nonlocal stuff 
   MatScalarKokkosDualView a_nonlocal_dual;
   MatRowMapKokkosDualView i_nonlocal_dual;
   MatColIdxKokkosDualView j_nonlocal_dual;
   MatScalarKokkosView a_nonlocal_d;
   MatRowMapKokkosView i_nonlocal_d;          
   MatColIdxKokkosView j_nonlocal_d;    

   // we also have to go and build the a, i, j for the non-local off-diagonal block
   if (mpi) 
   {
      // Non-local 
      a_nonlocal_dual = MatScalarKokkosDualView("a_nonlocal_dual", nnzs_match_nonlocal);
      i_nonlocal_dual = MatRowMapKokkosDualView("i_nonlocal_dual", local_rows+1);
      j_nonlocal_dual = MatColIdxKokkosDualView("j_nonlocal_dual", nnzs_match_nonlocal);  

      a_nonlocal_d = a_nonlocal_dual.view_device();
      i_nonlocal_d = i_nonlocal_dual.view_device();
      // Initialize first entry to zero - the rest get set below
      Kokkos::deep_copy(Kokkos::subview(i_nonlocal_d, 0), 0);                
      j_nonlocal_d = j_nonlocal_dual.view_device();   
   }        

   // Initialize i_local_d row pointers (1 to local_rows) with cumulative sums from the scan
   PetscInt one = 1;
   auto i_local_range = Kokkos::subview(i_local_d, Kokkos::make_pair(one, local_rows+1));
   Kokkos::deep_copy(i_local_range, nnz_match_local_row_d);
   
   // Similarly for MPI nonlocal case if needed
   if (mpi) {
      auto i_nonlocal_range = Kokkos::subview(i_nonlocal_d, Kokkos::make_pair(one, local_rows+1));
      Kokkos::deep_copy(i_nonlocal_range, nnz_match_nonlocal_row_d);
   }          
   
   // Filling the matrix is easy as we know we only have one non-zero per row
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(int i) {

      // If our max val is in the local block
      if (has_entry_local_d(i) > 0) {
         j_local_d(i_local_d(i)) = max_col_row_d(i);
         a_local_d(i_local_d(i)) = 1.0;
      }
      else if (mpi && has_entry_nonlocal_d(i) > 0)
      {
         j_nonlocal_d(i_nonlocal_d(i)) = max_col_row_d(i);
         a_nonlocal_d(i_nonlocal_d(i)) = 1.0;         
      }   
   });      

   // Have to specify that we've modified the device data
   a_local_dual.modify_device();
   i_local_dual.modify_device();
   j_local_dual.modify_device();

   // We can create our local diagonal block matrix directly on the device
   // See MatSeqAIJKokkosMergeMats for example
   auto akok_local = new Mat_SeqAIJKokkos(local_rows, local_cols, nnzs_match_local, i_local_dual, j_local_dual, a_local_dual);    
   // The equivalent of calling the internal MatCreateSeqAIJKokkosWithCSRMatrix
   MatCreate(PETSC_COMM_SELF, &output_mat_local);
   // Why isn't this publically available??
   MatSetSeqAIJKokkosWithCSRMatrix_mine(output_mat_local, akok_local);   

   // we also have to go and build the a, i, j for the non-local off-diagonal block
   if (mpi) 
   {
      // Have to specify that we've modified the device data
      a_nonlocal_dual.modify_device();
      i_nonlocal_dual.modify_device();
      j_nonlocal_dual.modify_device();

      // Now we need to build garray on the host and rewrite the j_nonlocal_d indices so they are local
      // The default values here are for the case where we 
      // let do it, it resets this internally in MatSetUpMultiply_MPIAIJ
      PetscInt *garray_host = NULL;
      PetscInt col_ao_output = global_cols;
      if (cols_ao == 0)
      {
         // Silly but depending on the compiler this may return a non-null pointer
         col_ao_output = 0;
         PetscMalloc1(col_ao_output, &garray_host);
      }

      // We can use the Kokkos::UnorderedMap to do this if our 
      // off diagonal block has fewer than 4 billion non-zero columns (max capacity of uint32_t)
      // Otherwise we can just tell petsc to do do it on the host (in MatSetUpMultiply_MPIAIJ)
      // and rely on the hash tables in petsc on the host which can handle more than 4 billion entries
      // We trigger petsc doing it by passing in null as garray_host to MatSetMPIAIJKokkosWithSplitSeqAIJKokkosMatrices
      // If we have no off-diagonal entries (either we started with zero or we've dropped them all)
      // just skip all this and leave garray_host as null

      // If we have 4 bit ints, we know cols_ao can never be bigger than the capacity of uint32_t
      bool size_small_enough = sizeof(PetscInt) == 4 || \
                  (sizeof(PetscInt) > 4 && cols_ao < 4294967295);
      if (size_small_enough && cols_ao > 0 && nnzs_match_nonlocal > 0)
      {
         // Have to tell it the max capacity, we know we will have no more 
         // than the input off-diag columns
         Kokkos::UnorderedMap<PetscInt, PetscInt> hashmap((uint32_t)(cols_ao+1));

         // Let's insert all the existing global col indices as keys (with no value to start)
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, nnzs_match_nonlocal), KOKKOS_LAMBDA(int i) {      
            
            // Insert the key (global col indices) without a value
            // Duplicates will be ignored
            hashmap.insert(j_nonlocal_d(i));
         });

         // We now know how many unique global columns we have
         col_ao_output = hashmap.size();

         // Tag which of the original garray stick around  
         PetscIntKokkosView colmap_output_d_big("colmap_output_d_big", cols_ao);
         Kokkos::deep_copy(colmap_output_d_big, colmap_input_d);                

         // Mark which of the keys don't exist
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, cols_ao), KOKKOS_LAMBDA(int i) { 

            // If the key doesn't exist set the global index to -1
            if (!hashmap.exists(colmap_output_d_big(i))) colmap_output_d_big(i) = -1; 
         });         

         // Now sort the global columns indices
         // All the -1 should be at the start
         Kokkos::sort(colmap_output_d_big);

         // Count the number of -1 - this will be the index of the first entry
         // that isn't -1
         // It should never be equal to start index, because otherwise we
         // have dropped all nonlocal entries
         auto &exec = PetscGetKokkosExecutionSpace();
         PetscInt start_index = Kokkos::Experimental::count(exec, colmap_output_d_big, -1);

         // Our final colmap_output_d is colmap_output_d_big(start_index:end)
         PetscIntKokkosView colmap_output_d = Kokkos::subview(colmap_output_d_big, \
                  Kokkos::make_pair(start_index, cols_ao));

         // Now we can clear the hash and instead stick in the global indices
         // but now with the local indices as values
         hashmap.clear();
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, colmap_output_d.extent(0)), KOKKOS_LAMBDA(int i) { 

            hashmap.insert(colmap_output_d(i), i);
         });          

         // And now we can overwrite j_nonlocal_d with the local indices
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, nnzs_match_nonlocal), KOKKOS_LAMBDA(int i) {     

            // Find where our global col index is at
            uint32_t loc = hashmap.find(j_nonlocal_d(i));
            // And get the value (the new local index)
            j_nonlocal_d(i) = hashmap.value_at(loc);
         });      
         hashmap.clear();

         // Create some host space for the output garray (that stays in scope) and copy it
         PetscMalloc1(colmap_output_d.extent(0), &garray_host);
         PetscIntKokkosViewHost colmap_output_h = PetscIntKokkosViewHost(garray_host, colmap_output_d.extent(0));
         Kokkos::deep_copy(colmap_output_h, colmap_output_d);   
      }

      // We can create our nonlocal diagonal block matrix directly on the device
      auto akok_nonlocal = new Mat_SeqAIJKokkos(local_rows, col_ao_output, \
               nnzs_match_nonlocal, i_nonlocal_dual, j_nonlocal_dual, a_nonlocal_dual);  
      // The equivalent of calling the internal MatCreateSeqAIJKokkosWithCSRMatrix
      MatCreate(PETSC_COMM_SELF, &output_mat_nonlocal);
      // Why isn't this publically available??
      MatSetSeqAIJKokkosWithCSRMatrix_mine(output_mat_nonlocal, akok_nonlocal);    

      // Build our mpi kokkos matrix by passing in the local and 
      // nonlocal kokkos matrices and the colmap
      // If you read the description of MatSetMPIAIJKokkosWithSplitSeqAIJKokkosMatrices it says 
      // B     - the offdiag matrix using global col ids
      // but reading the code, if you pass in garray as not null
      // then the column id's of B should be the local indices,
      // as when it calls MatAssemblyEnd of the mpikokkos, it calls the MatAssemblyEnd of the aijmpi matrix
      // which then calls MatSetUpMultiply_MPIAIJ
      // and looking at that it only goes and builds garray and compactifies (and turns indices to local)
      // in B if garray is null 
      MatCreate(MPI_COMM_MATRIX, output_mat);
      // Only have to set the size * type in the mpi case, the serial case it gets set in 
      // MatSetSeqAIJKokkosWithCSRMatrix
      MatSetSizes(*output_mat, local_rows, local_cols, global_rows, global_cols);
      PetscLayoutSetUp((*output_mat)->rmap);
      PetscLayoutSetUp((*output_mat)->cmap);
      MatSetType(*output_mat, mat_type);
      // Why isn't this publically available??
      MatSetMPIAIJKokkosWithSplitSeqAIJKokkosMatrices_mine(*output_mat, output_mat_local, output_mat_nonlocal, garray_host);

   }     
   // If in serial 
   else
   {
      *output_mat = output_mat_local;
   }

   return;
}


//------------------------------------------------------------------------------------------------------------------------

// Stick Z in a full sized R but with kokkos - keeping everything on the device
PETSC_INTERN void compute_R_from_Z_kokkos(Mat *input_mat, PetscInt global_row_start, IS *is_fine, \
                  IS *is_coarse, IS *orig_fine_col_indices, int identity_int, int reuse_int, int reuse_indices_int, \
                  Mat *output_mat)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt global_row_start_Z, global_row_end_plus_one_Z;
   PetscInt global_col_start_Z, global_col_end_plus_one_Z;
   PetscInt local_coarse_size, local_fine_size, local_full_cols;
   PetscInt global_coarse_size, global_fine_size, global_full_cols;
   PetscInt rows_ao, cols_ao, rows_ad, cols_ad, size_cols;
   PetscInt global_rows_z, global_cols_z;
   PetscInt local_rows_z, local_cols_z;
   MatType mat_type;
   PetscInt nnzs_match_local, nnzs_match_nonlocal;
   Mat output_mat_local, output_mat_nonlocal;

   MatGetType(*input_mat, &mat_type);
   // Are we in parallel?
   bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat_MPIAIJ *mat_mpi = nullptr;
   Mat mat_local, mat_nonlocal;

   PetscIntKokkosViewHost colmap_input_h;
   PetscIntKokkosView colmap_input_d;   
   if (mpi)
   {
      mat_mpi = (Mat_MPIAIJ *)(*input_mat)->data;
      mat_local = mat_mpi->A;
      mat_nonlocal = mat_mpi->B;
      MatGetSize(mat_nonlocal, &rows_ao, &cols_ao); 

      // We also copy the input mat colmap over to the device as we need it
      colmap_input_h = PetscIntKokkosViewHost(mat_mpi->garray, cols_ao);
      colmap_input_d = PetscIntKokkosView("colmap_input_d", cols_ao);
      Kokkos::deep_copy(colmap_input_d, colmap_input_h);        
   }
   else
   {
      mat_local = *input_mat;
   }

   // Get the comm
   PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX);
   ISGetLocalSize(*is_coarse, &local_coarse_size);
   ISGetLocalSize(*is_fine, &local_fine_size);
   ISGetSize(*is_coarse, &global_coarse_size);
   ISGetSize(*is_fine, &global_fine_size);      

   local_full_cols = local_coarse_size + local_fine_size;
   global_full_cols = global_coarse_size + global_fine_size;

   MatGetLocalSize(*input_mat, &local_rows_z, &local_cols_z); 
   MatGetSize(*input_mat, &global_rows_z, &global_cols_z);
   
   MatGetOwnershipRange(*input_mat, &global_row_start_Z, &global_row_end_plus_one_Z);
   MatGetOwnershipRangeColumn(*input_mat, &global_col_start_Z, &global_col_end_plus_one_Z);

   MatGetType(*input_mat, &mat_type);
   MatGetSize(mat_local, &rows_ad, &cols_ad);

   // We can reuse the orig_fine_col_indices as they can be expensive to generate in parallel
   if (!reuse_indices_int)
   {
      PetscInt *col_indices_off_proc_array;
      IS col_indices;

      // Build these on the host as we need to call host routines 
      // on them anyway, we can transfer the result to the device
      if (mpi)
      {
         PetscMalloc1(cols_ad + cols_ao, &col_indices_off_proc_array);
         size_cols = cols_ad + cols_ao;
         for (int i = 0; i < cols_ad; i++)
         {
            col_indices_off_proc_array[i] = global_col_start_Z + i;
         }
         for (int i = 0; i < cols_ao; i++)
         {
            col_indices_off_proc_array[cols_ad + i] = mat_mpi->garray[i];
         }                   
      }
      else
      {
         PetscMalloc1(cols_ad, &col_indices_off_proc_array);
         size_cols = cols_ad;
         for (int i = 0; i < cols_ad; i++)
         {
            col_indices_off_proc_array[i] = global_col_start_Z + i;
         }
      }

      // Create the IS we want with the cols we want (written as global indices)
      ISCreateGeneral(MPI_COMM_MATRIX, size_cols, col_indices_off_proc_array, PETSC_USE_POINTER, &col_indices);

      // Now let's do the comms to get what the original column indices in the full matrix are, given these indices for all 
      // the columns of Z - ie we need to check in the original fine indices at the positions given by col_indices_off_proc_array
      // This could be expensive as the number of off-processor columns in Z grows!
      ISCreateSubIS(*is_fine, col_indices, orig_fine_col_indices);

      // We've now built the original fine indices
      ISDestroy(&col_indices);
      PetscFree(col_indices_off_proc_array);
   }
   else
   {
      ISGetLocalSize(*orig_fine_col_indices, &size_cols);
   }

   // Get pointers to the indices on the host
   const PetscInt *fine_indices_ptr, *coarse_indices_ptr, *is_pointer_orig_fine_col;
   ISGetIndices(*is_fine, &fine_indices_ptr);   
   ISGetIndices(*is_coarse, &coarse_indices_ptr); 
   ISGetIndices(*orig_fine_col_indices, &is_pointer_orig_fine_col);     

   // Create a host view of the existing indices
   auto fine_view_h = PetscIntConstKokkosViewHost(fine_indices_ptr, local_fine_size);    
   auto fine_view_d = PetscIntKokkosView("fine_view_d", local_fine_size);   
   auto coarse_view_h = PetscIntConstKokkosViewHost(coarse_indices_ptr, local_coarse_size);    
   auto coarse_view_d = PetscIntKokkosView("coarse_view_d", local_coarse_size);    
   auto orig_view_h = PetscIntConstKokkosViewHost(is_pointer_orig_fine_col, size_cols);    
   auto orig_view_d = PetscIntKokkosView("orig_view_d", size_cols);       
   // Copy indices to the device
   Kokkos::deep_copy(fine_view_d, fine_view_h);     
   Kokkos::deep_copy(coarse_view_d, coarse_view_h);    
   Kokkos::deep_copy(orig_view_d, orig_view_h);      

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr, *device_nonlocal_vals = nullptr;  
   MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype);  
   if (mpi) MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype);          

   PetscIntKokkosView nnz_match_local_row_d;
   PetscIntKokkosView nnz_match_nonlocal_row_d;
   MatScalarKokkosDualView a_local_dual;
   MatRowMapKokkosDualView i_local_dual;
   MatColIdxKokkosDualView j_local_dual;

   // Get device views
   MatScalarKokkosView a_local_d;
   MatRowMapKokkosView i_local_d;  
   MatColIdxKokkosView j_local_d;    

   // Nonlocal stuff 
   MatScalarKokkosDualView a_nonlocal_dual;
   MatRowMapKokkosDualView i_nonlocal_dual;
   MatColIdxKokkosDualView j_nonlocal_dual;
   MatScalarKokkosView a_nonlocal_d;
   MatRowMapKokkosView i_nonlocal_d;          
   MatColIdxKokkosView j_nonlocal_d;  

   Mat_MPIAIJ *mat_mpi_output = nullptr;
   Mat mat_local_output, mat_nonlocal_output;   

   // Only need things to do with the sparsity pattern if we're not reusing
   if (!reuse_int)
   {
      // ~~~~~~~~~~~~
      // Get the number of nnzs
      // ~~~~~~~~~~~~
      nnzs_match_local = 0;
      nnzs_match_nonlocal = 0;

      // ~~~~~~~~~~~~~~~~~~~~~~~
      // Let's build our i, j, and a on the device
      // ~~~~~~~~~~~~~~~~~~~~~~~ 
      // We need to know how many entries are in each row 
      nnz_match_local_row_d = PetscIntKokkosView("nnz_match_local_row_d", local_rows_z);    
      // We may have identity
      Kokkos::deep_copy(nnz_match_local_row_d, 0);         
      if (mpi) nnz_match_nonlocal_row_d = PetscIntKokkosView("nnz_match_nonlocal_row_d", local_rows_z);                  

      // ~~~~~~~~~~~~
      // Need to count the number of nnzs we end up with, on each row and in total
      // ~~~~~~~~~~~~
      // Loop over the rows of Z
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows_z), KOKKOS_LAMBDA(int i) {

            // Row index is simple
            PetscInt row_index = i;
            // Still using i here (the local index into Z)
            PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];
            nnz_match_local_row_d(row_index) = ncols_local;
            // Add one extra in this local block for the identity
            if (identity_int) nnz_match_local_row_d(row_index)++;

            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
               nnz_match_nonlocal_row_d(row_index) = ncols_nonlocal;
            }
      });

      // Get number of nnzs
      Kokkos::parallel_reduce ("ReductionLocal", local_rows_z, KOKKOS_LAMBDA (const int i, PetscInt& update) {
         update += nnz_match_local_row_d(i); 
      }, nnzs_match_local);   
      if (mpi)
      {
         Kokkos::parallel_reduce ("ReductionNonLocal", local_rows_z, KOKKOS_LAMBDA (const int i, PetscInt& update) {
            update += nnz_match_nonlocal_row_d(i); 
         }, nnzs_match_nonlocal);       
      }

      // ~~~~~~~~~~~~

      // Need to do a scan on nnz_match_local_row_d to get where each row starts
      Kokkos::parallel_scan (local_rows_z, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
         // Inclusive scan
         update += nnz_match_local_row_d(i);         
         if (final) {
            nnz_match_local_row_d(i) = update; // only update array on final pass
         }
      });      
      if (mpi)
      { 
         // Need to do a scan on nnz_match_nonlocal_row_d to get where each row starts
         Kokkos::parallel_scan (local_rows_z, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
            // Inclusive scan
            update += nnz_match_nonlocal_row_d(i);         
            if (final) {
               nnz_match_nonlocal_row_d(i) = update; // only update array on final pass
            }
         });               
      }           

      // ~~~~~~~~~~~~~~~~~  
      // We need to assemble our i,j, vals so we can build our matrix
      // ~~~~~~~~~~~~~~~~~
      // Create dual memory on the device and host
      a_local_dual = MatScalarKokkosDualView("a_local_dual", nnzs_match_local);
      i_local_dual = MatRowMapKokkosDualView("i_local_dual", local_rows_z+1);
      j_local_dual = MatColIdxKokkosDualView("j_local_dual", nnzs_match_local);

      // Get device views
      a_local_d = a_local_dual.view_device();
      i_local_d = i_local_dual.view_device();
      // Initialize first entry to zero - the rest get set below
      Kokkos::deep_copy(Kokkos::subview(i_local_d, 0), 0);       
      j_local_d = j_local_dual.view_device();    

      // we also have to go and build the a, i, j for the non-local off-diagonal block
      if (mpi) 
      {
         // Non-local 
         a_nonlocal_dual = MatScalarKokkosDualView("a_nonlocal_dual", nnzs_match_nonlocal);
         i_nonlocal_dual = MatRowMapKokkosDualView("i_nonlocal_dual", local_rows_z+1);
         j_nonlocal_dual = MatColIdxKokkosDualView("j_nonlocal_dual", nnzs_match_nonlocal);  

         a_nonlocal_d = a_nonlocal_dual.view_device();
         i_nonlocal_d = i_nonlocal_dual.view_device();
         // Initialize first entry to zero - the rest get set below
         Kokkos::deep_copy(Kokkos::subview(i_nonlocal_d, 0), 0);                
         j_nonlocal_d = j_nonlocal_dual.view_device();   
      }  

      // ~~~~~~~~~~~~~~~
      // Create i indices
      // ~~~~~~~~~~~~~~~
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows_z), KOKKOS_LAMBDA(int i) {

            // Row index is simple
            PetscInt row_index = i;       

            // The start of our row index comes from the scan
            i_local_d(row_index + 1) = nnz_match_local_row_d(row_index);   
            if (mpi) i_nonlocal_d(row_index + 1) = nnz_match_nonlocal_row_d(row_index);         
      });            


      // Loop over the rows of Z
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows_z, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            PetscInt i = t.league_rank();

            // Row index is simple
            PetscInt row_index = i;
            // Still using i here (the local index into Z)
            PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];         

            // For over local columns - copy in Z
            Kokkos::parallel_for(
               Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

               // Want the local col indices for the local block
               // The orig_view_d contains the global indices for the original full matrix
               j_local_d(i_local_d(row_index) + j) = orig_view_d(device_local_j[device_local_i[i] + j]) - global_row_start;
               a_local_d(i_local_d(row_index) + j) = device_local_vals[device_local_i[i] + j];
                     
            });     

            // For over nonlocal columns - copy in Z
            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];         

               Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {

                  // We keep the existing local indices in the off-diagonal block here
                  // we have all the same non-local local column indices as Z (as the identity added is always local)
                  // The garray is the same size, its just the global indices that have changed
                  j_nonlocal_d(i_nonlocal_d(row_index) + j) = device_nonlocal_j[device_nonlocal_i[i] + j];
                  a_nonlocal_d(i_nonlocal_d(row_index) + j) = device_nonlocal_vals[device_nonlocal_i[i] + j];
                        
               });          
            }

            // Only want one thread to deal with the single identity value
            if (identity_int)
            {
               Kokkos::single(Kokkos::PerTeam(t), [&]() {
                  // Let's just stick it at the end and we will sort after
                  // The coarse_view_d contains the global indices for the original full matrix
                  j_local_d(i_local_d(row_index) + ncols_local) = coarse_view_d(i) - global_row_start;
                  a_local_d(i_local_d(row_index) + ncols_local) = 1.0;
               });     
            }         
      }); 
   }
   // If we're reusing, we can just write directly to the existing views
   else
   {
      // Get the existing output mats
      if (mpi)
      {
         mat_mpi_output = (Mat_MPIAIJ *)(*output_mat)->data;
         mat_local_output = mat_mpi_output->A;
         mat_nonlocal_output = mat_mpi_output->B;     
      }
      else
      {
         mat_local_output = *output_mat;
      }     
      Mat_SeqAIJKokkos *aijkok_local_output = static_cast<Mat_SeqAIJKokkos *>(mat_local_output->spptr);
      Mat_SeqAIJKokkos *aijkok_nonlocal_output;
      if (mpi) aijkok_nonlocal_output = static_cast<Mat_SeqAIJKokkos *>(mat_nonlocal_output->spptr);

      // Annoying we can't just call MatSeqAIJGetKokkosView
      a_local_d = aijkok_local_output->a_dual.view_device();
      if (mpi) a_nonlocal_d = aijkok_nonlocal_output->a_dual.view_device();

      // Annoyingly there isn't currently the ability to get views for i (or j)
      const PetscInt *device_local_i_output = nullptr, *device_local_j_output = nullptr, *device_nonlocal_i_ouput = nullptr;
      PetscMemType mtype;
      MatSeqAIJGetCSRAndMemType(mat_local_output, &device_local_i_output, &device_local_j_output, NULL, &mtype);  
      if (mpi) MatSeqAIJGetCSRAndMemType(mat_nonlocal_output, &device_nonlocal_i_ouput, NULL, NULL, &mtype);  

      // Have these point at the existing i pointers - we only need the local j
      ConstMatRowMapKokkosView i_local_const_d = ConstMatRowMapKokkosView(device_local_i_output, local_rows_z+1);
      ConstMatColIdxKokkosView j_local_const_d = ConstMatColIdxKokkosView(device_local_j_output, aijkok_local_output->csrmat.nnz());
      ConstMatRowMapKokkosView i_nonlocal_const_d;
      if (mpi) i_nonlocal_const_d = ConstMatRowMapKokkosView(device_nonlocal_i_ouput, local_rows_z+1);        

      // Only have to write Z but have to be careful as we have the identity mixed 
      // in there
      // Loop over the rows of Z - annoying we have const views as this is just the same loop as above
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows_z, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            PetscInt i = t.league_rank();

            // Simple row index
            PetscInt row_index = i;
            // Still using i here (the local index into Z)
            PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];         

            // For over local columns - copy in Z
            // We have to skip over the identity entries, which we know are always C points
            Kokkos::parallel_for(
               Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

               PetscInt offset = 0;

               // If we're at or after the C point identity, our index into R gets a +1
               // so we skip over writing to that index in R
               if (j_local_const_d(i_local_const_d(row_index) + j) >= coarse_view_d(i) - global_row_start) offset = 1;
               a_local_d(i_local_const_d(row_index) + j + offset) = device_local_vals[device_local_i[i] + j];
            });     

            // For over nonlocal columns - copy in Z - identical structure in the off-diag block
            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];         

               Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {

                  // We keep the existing local indices in the off-diagonal block here
                  // we have all the same columns as Z and hence the same garray
                  a_nonlocal_d(i_nonlocal_const_d(row_index) + j) = device_nonlocal_vals[device_nonlocal_i[i] + j];
                        
               });          
            }
      });   

      // Have to specify we've modifed data on the device
      // Want to call MatSeqAIJKokkosModifyDevice but its PETSC_INTERN
      aijkok_local_output->a_dual.clear_sync_state();
      aijkok_local_output->a_dual.modify_device();
      aijkok_local_output->transpose_updated = PETSC_FALSE;
      aijkok_local_output->hermitian_updated = PETSC_FALSE;
      // Invalidate diagonals
      Mat_SeqAIJ *a = (Mat_SeqAIJ *)mat_local_output->data;
      a->idiagvalid  = PETSC_FALSE;
      a->ibdiagvalid = PETSC_FALSE;      
      a->inode.ibdiagvalid = PETSC_FALSE;      
      if (mpi)
      {
         aijkok_nonlocal_output->a_dual.clear_sync_state();
         aijkok_nonlocal_output->a_dual.modify_device();
         aijkok_nonlocal_output->transpose_updated = PETSC_FALSE;
         aijkok_nonlocal_output->hermitian_updated = PETSC_FALSE;
         a = (Mat_SeqAIJ *)mat_nonlocal_output->data;
         a->idiagvalid  = PETSC_FALSE;
         a->ibdiagvalid = PETSC_FALSE;   
         a->inode.ibdiagvalid = PETSC_FALSE;       
      }        
      PetscObjectStateIncrease((PetscObject)(*output_mat));    

    }

   // ~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~   

   if (!reuse_int)
   {
      // Have to specify that we've modified the device data
      a_local_dual.modify_device();
      i_local_dual.modify_device();
      j_local_dual.modify_device();      
    
      // We can create our local diagonal block matrix directly on the device
      // See MatSeqAIJKokkosMergeMats for example
      auto akok_local = new Mat_SeqAIJKokkos(local_rows_z, local_full_cols, nnzs_match_local, i_local_dual, j_local_dual, a_local_dual);    

      // Now we have to sort the local column indices, as we add in the identity at the 
      // end of our local j indices
      KokkosSparse::sort_crs_matrix(akok_local->csrmat);

      a_local_dual.modify_device();
      i_local_dual.modify_device();
      j_local_dual.modify_device();        

      // The equivalent of calling the internal MatCreateSeqAIJKokkosWithCSRMatrix
      MatCreate(PETSC_COMM_SELF, &output_mat_local);
      // Why isn't this publically available??
      MatSetSeqAIJKokkosWithCSRMatrix_mine(output_mat_local, akok_local);   

      // we also have to go and build our off block matrix and then the output
      if (mpi) 
      {
         // Have to specify that we've modified the device data
         a_nonlocal_dual.modify_device();
         i_nonlocal_dual.modify_device();
         j_nonlocal_dual.modify_device();

         // We know the garray is just the original but rewritten to be 
         // the full indices, which we have in in is_pointer_orig_fine_col(cols_ad:end)
         PetscInt *garray_host = NULL; 
         PetscMalloc1(cols_ao, &garray_host);
         for (int i = 0; i < cols_ao; i++)
         {
            garray_host[i] = is_pointer_orig_fine_col[i + cols_ad];
         }    

         // We can create our nonlocal diagonal block matrix directly on the device
         auto akok_nonlocal = new Mat_SeqAIJKokkos(local_rows_z, cols_ao, \
                  nnzs_match_nonlocal, i_nonlocal_dual, j_nonlocal_dual, a_nonlocal_dual);  
         // The equivalent of calling the internal MatCreateSeqAIJKokkosWithCSRMatrix
         MatCreate(PETSC_COMM_SELF, &output_mat_nonlocal);
         // Why isn't this publically available??
         MatSetSeqAIJKokkosWithCSRMatrix_mine(output_mat_nonlocal, akok_nonlocal);    

         // Build our mpi kokkos matrix by passing in the local and 
         // nonlocal kokkos matrices and the colmap
         // If you read the description of MatSetMPIAIJKokkosWithSplitSeqAIJKokkosMatrices it says 
         // B     - the offdiag matrix using global col ids
         // but reading the code, if you pass in garray as not null
         // then the column id's of B should be the local indices,
         // as when it calls MatAssemblyEnd of the mpikokkos, it calls the MatAssemblyEnd of the aijmpi matrix
         // which then calls MatSetUpMultiply_MPIAIJ
         // and looking at that it only goes and builds garray and compactifies (and turns indices to local)
         // in B if garray is null 
         MatCreate(MPI_COMM_MATRIX, output_mat);
         // Only have to set the size * type in the mpi case, the serial case it gets set in 
         // MatSetSeqAIJKokkosWithCSRMatrix
         MatSetSizes(*output_mat, local_rows_z, local_full_cols, global_rows_z, global_full_cols);
         PetscLayoutSetUp((*output_mat)->rmap);
         PetscLayoutSetUp((*output_mat)->cmap);
         MatSetType(*output_mat, mat_type);
         // Why isn't this publically available??
         MatSetMPIAIJKokkosWithSplitSeqAIJKokkosMatrices_mine(*output_mat, output_mat_local, output_mat_nonlocal, garray_host);

      }    
      // If in serial 
      else
      {
         *output_mat = output_mat_local;
      }
   }

   ISRestoreIndices(*is_fine, &fine_indices_ptr);
   ISRestoreIndices(*is_coarse, &coarse_indices_ptr); 
   ISRestoreIndices(*orig_fine_col_indices, &is_pointer_orig_fine_col);    

   return;
}



//------------------------------------------------------------------------------------------------------------------------

// Duplicate and copy a matrix ensuring it always has a diagonal but with kokkos - keeping everything on the device
PETSC_INTERN void mat_duplicate_copy_plus_diag_kokkos(Mat *input_mat, int reuse_int, Mat *output_mat)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt global_col_start, global_col_end_plus_one;
   PetscInt rows_ao, cols_ao;
   PetscInt global_rows, global_cols;
   PetscInt local_rows, local_cols;
   MatType mat_type;
   PetscInt nnzs_match_local, nnzs_match_nonlocal;
   Mat output_mat_local, output_mat_nonlocal;

   MatGetType(*input_mat, &mat_type);
   // Are we in parallel?
   bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat_MPIAIJ *mat_mpi = nullptr;
   Mat mat_local, mat_nonlocal;

   PetscIntKokkosViewHost colmap_input_h;
   PetscIntKokkosView colmap_input_d;   
   if (mpi)
   {
      mat_mpi = (Mat_MPIAIJ *)(*input_mat)->data;
      mat_local = mat_mpi->A;
      mat_nonlocal = mat_mpi->B;
      MatGetSize(mat_nonlocal, &rows_ao, &cols_ao); 

      // // We also copy the input mat colmap over to the device as we need it
      // colmap_input_h = PetscIntKokkosViewHost(mat_mpi->garray, cols_ao);
      // colmap_input_d = PetscIntKokkosView("colmap_input_d", cols_ao);
      // Kokkos::deep_copy(colmap_input_d, colmap_input_h);        
   }
   else
   {
      mat_local = *input_mat;
   }

   // Get the comm
   PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX);
   MatGetLocalSize(*input_mat, &local_rows, &local_cols); 
   MatGetSize(*input_mat, &global_rows, &global_cols);
   MatGetOwnershipRange(*input_mat, &global_row_start, &global_row_end_plus_one);
   MatGetOwnershipRangeColumn(*input_mat, &global_col_start, &global_col_end_plus_one);
   MatGetType(*input_mat, &mat_type);

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr, *device_nonlocal_vals = nullptr;  
   MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype);  
   if (mpi) MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype);          

   PetscIntKokkosView nnz_match_local_row_d;
   PetscIntKokkosView nnz_match_nonlocal_row_d;
   MatScalarKokkosDualView a_local_dual;
   MatRowMapKokkosDualView i_local_dual;
   MatColIdxKokkosDualView j_local_dual;

   // Get device views
   MatScalarKokkosView a_local_d;
   MatRowMapKokkosView i_local_d;  
   MatColIdxKokkosView j_local_d;    

   // Nonlocal stuff 
   MatScalarKokkosDualView a_nonlocal_dual;
   MatRowMapKokkosDualView i_nonlocal_dual;
   MatColIdxKokkosDualView j_nonlocal_dual;
   MatScalarKokkosView a_nonlocal_d;
   MatRowMapKokkosView i_nonlocal_d;          
   MatColIdxKokkosView j_nonlocal_d;  

   Mat_MPIAIJ *mat_mpi_output = nullptr;
   Mat mat_local_output, mat_nonlocal_output;   

   // We always need to know if we found a diagonal in each row of the input_matrix
   auto found_diag_row_d = PetscIntKokkosView("found_diag_row_d", local_rows);    
   Kokkos::deep_copy(found_diag_row_d, 0); 

   // ~~~~~~~~~~~~
   // Get the number of nnzs
   // ~~~~~~~~~~~~
   nnzs_match_local = 0;
   nnzs_match_nonlocal = 0;

   // We need to know how many entries are in each row 
   nnz_match_local_row_d = PetscIntKokkosView("nnz_match_local_row_d", local_rows);    
   // We may have identity
   Kokkos::deep_copy(nnz_match_local_row_d, 0);              
   if (mpi) nnz_match_nonlocal_row_d = PetscIntKokkosView("nnz_match_nonlocal_row_d", local_rows);      

   // Calculate if each row has a diagonal, we need to know this for both 
   // reuse and not reuse
   // For the local block we need to count the nnzs
   // but if there is no diagonal we need to add one in
   Kokkos::parallel_reduce(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t, PetscInt& thread_total) {

      PetscInt i   = t.league_rank(); // row i
      PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];

      // We have a custom reduction type defined - ReduceData
      // Which has both a nnz count for this row, but also tracks whether we 
      // found the diagonal
      ReduceData row_result;

      // Reduce over all the columns
      Kokkos::parallel_reduce(
         Kokkos::TeamThreadRange(t, ncols_local),
         [&](const PetscInt j, ReduceData& thread_data) {

            // Is this column the diagonal
            bool is_diagonal = (device_local_j[device_local_i[i] + j] + global_col_start == i + global_row_start);
            
            // We have found a diagonal in this row
            if (is_diagonal) {
               thread_data.found_diagonal = true;
            }
            thread_data.count++;
         }, row_result
      );
      
      // Only want one thread in the team to write the result
      Kokkos::single(Kokkos::PerTeam(t), [&]() {      
         if (!row_result.found_diagonal) 
         {
            row_result.count++;
         }
         else
         {
            found_diag_row_d(i) = 1;
         }
         nnz_match_local_row_d(i) = row_result.count;
         thread_total += row_result.count;            
      });
      },
      nnzs_match_local
   );         

   // Only need things to do with the sparsity pattern if we're not reusing
   if (!reuse_int)
   {
      // ~~~~~~~~~~~~
      // Need to count the number of nnzs we end up with, on each row and in total
      // ~~~~~~~~~~~~
      // Loop over the rows
      if (mpi)
      {      
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(int i) {

               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
               nnz_match_nonlocal_row_d(i) = ncols_nonlocal;
         });
      }
      if (mpi)
      {
         Kokkos::parallel_reduce ("ReductionNonLocal", local_rows, KOKKOS_LAMBDA (const int i, PetscInt& update) {
            update += nnz_match_nonlocal_row_d(i); 
         }, nnzs_match_nonlocal);       
      }

      // ~~~~~~~~~~~~

      // Need to do a scan on nnz_match_local_row_d to get where each row starts
      Kokkos::parallel_scan (local_rows, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
         // Inclusive scan
         update += nnz_match_local_row_d(i);         
         if (final) {
            nnz_match_local_row_d(i) = update; // only update array on final pass
         }
      });      
      if (mpi)
      { 
         // Need to do a scan on nnz_match_nonlocal_row_d to get where each row starts
         Kokkos::parallel_scan (local_rows, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
            // Inclusive scan
            update += nnz_match_nonlocal_row_d(i);         
            if (final) {
               nnz_match_nonlocal_row_d(i) = update; // only update array on final pass
            }
         });               
      }           

      // ~~~~~~~~~~~~~~~~~  
      // We need to assemble our i,j, vals so we can build our matrix
      // ~~~~~~~~~~~~~~~~~
      // Create dual memory on the device and host
      a_local_dual = MatScalarKokkosDualView("a_local_dual", nnzs_match_local);
      i_local_dual = MatRowMapKokkosDualView("i_local_dual", local_rows+1);
      j_local_dual = MatColIdxKokkosDualView("j_local_dual", nnzs_match_local);

      // Get device views
      a_local_d = a_local_dual.view_device();
      i_local_d = i_local_dual.view_device();
      // Initialize first entry to zero - the rest get set below
      Kokkos::deep_copy(Kokkos::subview(i_local_d, 0), 0);       
      j_local_d = j_local_dual.view_device();    

      // we also have to go and build the a, i, j for the non-local off-diagonal block
      if (mpi) 
      {
         // Non-local 
         a_nonlocal_dual = MatScalarKokkosDualView("a_nonlocal_dual", nnzs_match_nonlocal);
         i_nonlocal_dual = MatRowMapKokkosDualView("i_nonlocal_dual", local_rows+1);
         j_nonlocal_dual = MatColIdxKokkosDualView("j_nonlocal_dual", nnzs_match_nonlocal);  

         a_nonlocal_d = a_nonlocal_dual.view_device();
         i_nonlocal_d = i_nonlocal_dual.view_device();
         // Initialize first entry to zero - the rest get set below
         Kokkos::deep_copy(Kokkos::subview(i_nonlocal_d, 0), 0);                
         j_nonlocal_d = j_nonlocal_dual.view_device();   
      }  

      // ~~~~~~~~~~~~~~~
      // Create i indices
      // ~~~~~~~~~~~~~~~
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(int i) {

            // Row index is simple
            PetscInt row_index = i;       

            // The start of our row index comes from the scan
            i_local_d(row_index + 1) = nnz_match_local_row_d(row_index);   
            if (mpi) i_nonlocal_d(row_index + 1) = nnz_match_nonlocal_row_d(row_index);         
      });            


      // Loop over the rows
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            PetscInt i = t.league_rank();

            // Row index is simple
            PetscInt row_index = i;
            // Still using i here (the local index into input)
            PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];         

            // For over local columns - copy in input
            Kokkos::parallel_for(
               Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

               // Want the local col indices for the local block
               j_local_d(i_local_d(row_index) + j) = device_local_j[device_local_i[i] + j];
               a_local_d(i_local_d(row_index) + j) = device_local_vals[device_local_i[i] + j];
                     
            });     

            // For over nonlocal columns - copy in input
            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];         

               Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {

                  // We keep the existing local indices in the off-diagonal block here
                  j_nonlocal_d(i_nonlocal_d(row_index) + j) = device_nonlocal_j[device_nonlocal_i[i] + j];
                  a_nonlocal_d(i_nonlocal_d(row_index) + j) = device_nonlocal_vals[device_nonlocal_i[i] + j];
                        
               });          
            }

            // Only want one thread to deal with the diagonal
            Kokkos::single(Kokkos::PerTeam(t), [&]() {            
               // If we didn't find a diagonal
               if (!found_diag_row_d(i))
               {
                  // Let's just stick it at the end and we will sort after
                  j_local_d(i_local_d(row_index) + ncols_local) = i;
                  a_local_d(i_local_d(row_index) + ncols_local) = 0.0;
               }
            });              
      }); 
   }
   // If we're reusing, we can just write directly to the existing views
   else
   {
      // Get the existing output mats
      if (mpi)
      {
         mat_mpi_output = (Mat_MPIAIJ *)(*output_mat)->data;
         mat_local_output = mat_mpi_output->A;
         mat_nonlocal_output = mat_mpi_output->B;     
      }
      else
      {
         mat_local_output = *output_mat;
      }     
      Mat_SeqAIJKokkos *aijkok_local_output = static_cast<Mat_SeqAIJKokkos *>(mat_local_output->spptr);
      Mat_SeqAIJKokkos *aijkok_nonlocal_output;
      if (mpi) aijkok_nonlocal_output = static_cast<Mat_SeqAIJKokkos *>(mat_nonlocal_output->spptr);

      // Annoying we can't just call MatSeqAIJGetKokkosView
      a_local_d = aijkok_local_output->a_dual.view_device();
      if (mpi) a_nonlocal_d = aijkok_nonlocal_output->a_dual.view_device();
      // Because we might be missing diagonals and we're going to skip some of them 
      // in the writing loop below
      Kokkos::deep_copy(a_local_d, 0.0);

      // Annoyingly there isn't currently the ability to get views for i (or j)
      const PetscInt *device_local_i_output = nullptr, *device_local_j_output = nullptr, *device_nonlocal_i_ouput = nullptr;
      PetscMemType mtype;
      MatSeqAIJGetCSRAndMemType(mat_local_output, &device_local_i_output, &device_local_j_output, NULL, &mtype);  
      if (mpi) MatSeqAIJGetCSRAndMemType(mat_nonlocal_output, &device_nonlocal_i_ouput, NULL, NULL, &mtype);  

      // Have these point at the existing i pointers - we only need the local j
      ConstMatRowMapKokkosView i_local_const_d = ConstMatRowMapKokkosView(device_local_i_output, local_rows+1);
      ConstMatColIdxKokkosView j_local_const_d = ConstMatColIdxKokkosView(device_local_j_output, aijkok_local_output->csrmat.nnz());
      ConstMatRowMapKokkosView i_nonlocal_const_d;
      if (mpi) i_nonlocal_const_d = ConstMatRowMapKokkosView(device_nonlocal_i_ouput, local_rows+1);         

      // Only have to write a but have to be careful as we may not have diagonals in some rows
      // in the input, but they are in the output
      // Loop over the rows - annoying we have const views as this is just the same loop as above
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            PetscInt i = t.league_rank();

            // Simple row index
            PetscInt row_index = i;
            // Still using i here (the local index into input)
            PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];         

            // For over local columns - copy in input
            // We have to skip over the identity entries, which we know are always C points
            Kokkos::parallel_for(
               Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

               PetscInt offset = 0;

               // If we're at or after the diagonal and there isn't actually a diagonal in the input
               // we know the output has a diagonal, so we skip ahead one in the output and 
               // leave it unassigned in the output (it gets set to zero above)
               if (j_local_const_d(i_local_const_d(row_index) + j) >= i && \
                     !found_diag_row_d(i)) offset = 1;

               a_local_d(i_local_const_d(row_index) + j + offset) = device_local_vals[device_local_i[i] + j];
            });  
           
            // For over nonlocal columns - copy in input - identical structure in the off-diag block
            if (mpi)
            {
               PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];         

               Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {

                  // We keep the existing local indices in the off-diagonal block here
                  // we have all the same columns as input and hence the same garray
                  a_nonlocal_d(i_nonlocal_const_d(row_index) + j) = device_nonlocal_vals[device_nonlocal_i[i] + j];
                        
               });          
            }
      });   

      // Have to specify we've modifed data on the device
      // Want to call MatSeqAIJKokkosModifyDevice but its PETSC_INTERN
      aijkok_local_output->a_dual.clear_sync_state();
      aijkok_local_output->a_dual.modify_device();
      aijkok_local_output->transpose_updated = PETSC_FALSE;
      aijkok_local_output->hermitian_updated = PETSC_FALSE;
      // Invalidate diagonals
      Mat_SeqAIJ *a = (Mat_SeqAIJ *)mat_local_output->data;
      a->idiagvalid  = PETSC_FALSE;
      a->ibdiagvalid = PETSC_FALSE;      
      a->inode.ibdiagvalid = PETSC_FALSE;      
      if (mpi)
      {
         aijkok_nonlocal_output->a_dual.clear_sync_state();
         aijkok_nonlocal_output->a_dual.modify_device();
         aijkok_nonlocal_output->transpose_updated = PETSC_FALSE;
         aijkok_nonlocal_output->hermitian_updated = PETSC_FALSE;
         a = (Mat_SeqAIJ *)mat_nonlocal_output->data;
         a->idiagvalid  = PETSC_FALSE;
         a->ibdiagvalid = PETSC_FALSE;   
         a->inode.ibdiagvalid = PETSC_FALSE;       
      }        
      PetscObjectStateIncrease((PetscObject)(*output_mat));    

    }

   // ~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~   

   if (!reuse_int)
   {
      // Have to specify that we've modified the device data
      a_local_dual.modify_device();
      i_local_dual.modify_device();
      j_local_dual.modify_device();      
    
      // We can create our local diagonal block matrix directly on the device
      // See MatSeqAIJKokkosMergeMats for example
      auto akok_local = new Mat_SeqAIJKokkos(local_rows, local_cols, nnzs_match_local, i_local_dual, j_local_dual, a_local_dual);    

      // Now we have to sort the local column indices, as we add in the diagonal at the 
      // end of our local j indices in some rows
      KokkosSparse::sort_crs_matrix(akok_local->csrmat);

      a_local_dual.modify_device();
      i_local_dual.modify_device();
      j_local_dual.modify_device();        

      // The equivalent of calling the internal MatCreateSeqAIJKokkosWithCSRMatrix
      MatCreate(PETSC_COMM_SELF, &output_mat_local);
      // Why isn't this publically available??
      MatSetSeqAIJKokkosWithCSRMatrix_mine(output_mat_local, akok_local);   

      // we also have to go and build our off block matrix and then the output
      if (mpi) 
      {
         // Have to specify that we've modified the device data
         a_nonlocal_dual.modify_device();
         i_nonlocal_dual.modify_device();
         j_nonlocal_dual.modify_device();

         // We know the garray is just the original
         PetscInt *garray_host = NULL; 
         PetscMalloc1(cols_ao, &garray_host);
         for (int i = 0; i < cols_ao; i++)
         {
            garray_host[i] = mat_mpi->garray[i];
         }    

         // We can create our nonlocal diagonal block matrix directly on the device
         auto akok_nonlocal = new Mat_SeqAIJKokkos(local_rows, cols_ao, \
                  nnzs_match_nonlocal, i_nonlocal_dual, j_nonlocal_dual, a_nonlocal_dual);  
         // The equivalent of calling the internal MatCreateSeqAIJKokkosWithCSRMatrix
         MatCreate(PETSC_COMM_SELF, &output_mat_nonlocal);
         // Why isn't this publically available??
         MatSetSeqAIJKokkosWithCSRMatrix_mine(output_mat_nonlocal, akok_nonlocal);    

         // Build our mpi kokkos matrix by passing in the local and 
         // nonlocal kokkos matrices and the colmap
         // If you read the description of MatSetMPIAIJKokkosWithSplitSeqAIJKokkosMatrices it says 
         // B     - the offdiag matrix using global col ids
         // but reading the code, if you pass in garray as not null
         // then the column id's of B should be the local indices,
         // as when it calls MatAssemblyEnd of the mpikokkos, it calls the MatAssemblyEnd of the aijmpi matrix
         // which then calls MatSetUpMultiply_MPIAIJ
         // and looking at that it only goes and builds garray and compactifies (and turns indices to local)
         // in B if garray is null 
         MatCreate(MPI_COMM_MATRIX, output_mat);
         // Only have to set the size * type in the mpi case, the serial case it gets set in 
         // MatSetSeqAIJKokkosWithCSRMatrix
         MatSetSizes(*output_mat, local_rows, local_cols, global_rows, global_cols);
         PetscLayoutSetUp((*output_mat)->rmap);
         PetscLayoutSetUp((*output_mat)->cmap);
         MatSetType(*output_mat, mat_type);
         // Why isn't this publically available??
         MatSetMPIAIJKokkosWithSplitSeqAIJKokkosMatrices_mine(*output_mat, output_mat_local, output_mat_nonlocal, garray_host);

      }    
      // If in serial 
      else
      {
         *output_mat = output_mat_local;
      }
   }  

   return;
}
