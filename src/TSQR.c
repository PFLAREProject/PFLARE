/*
  C port of TSQR.F90 (module tsqr)

  The dense arrays are column-major with explicit leading dimensions. The
  module variables built_custom_op_tsqr and reduction_op_tsqr are file-static
  with the same lifetime: the custom reduction is created the first time it is
  needed and never freed. The tsqr_buffers struct is in pflareimpl.h, with
  create_tsqr_buffers giving the defaults of the Fortran type and
  destroy_tsqr_buffers freeing what the Fortran deallocated automatically when
  a tsqr_buffers went out of scope
*/

#include <petscmat.h>
#include <petscblaslapack.h>
#include "pflareimpl.h"
#include "TSQR.h"

// ~~~~~~~~~
// These globals are gross, but we use them to make sure we only create our custom reduction once
// in case its expensive to create it
// ~~~~~~~~~
static PetscBool built_custom_op_tsqr = PETSC_FALSE;
static MPI_Op    reduction_op_tsqr;

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode create_tsqr_buffers(tsqr_buffers **buffers)
{
   // Creates a tsqr_buffers with the defaults of the Fortran type

   PetscFunctionBegin;

   PetscCall(PetscNew(buffers));
   (*buffers)->request = MPI_REQUEST_NULL;
   (*buffers)->R_buffer_send = NULL;
   (*buffers)->R_buffer_receive = NULL;
   (*buffers)->matrix = NULL;
   // Has the user asked us to be on a subcomm
   (*buffers)->subcomm = PETSC_FALSE;
   // Did we actually end up on a subcomm
   (*buffers)->on_subcomm = PETSC_FALSE;
   (*buffers)->proc_stride = 1;

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode destroy_tsqr_buffers(tsqr_buffers **buffers)
{
   // Destroys a tsqr_buffers and any send/receive buffers it still holds, which
   // the Fortran deallocated automatically when the type went out of scope

   PetscFunctionBegin;

   if (!*buffers) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PetscFree((*buffers)->R_buffer_send));
   PetscCall(PetscFree((*buffers)->R_buffer_receive));
   PetscCall(PetscFree(*buffers));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode setup_custom_tsqr_reduction(void)
{
   // Creates our custom reduction

   PetscFunctionBegin;

   // Point at the custom reduction function
   if (!built_custom_op_tsqr) {
      // We should be able to set commute as true, as the order doesn't matter for QRs
      // But I have bumped into cases where due to rounding (typically in low-rank systems)
      // the small entries in the resulting R can differ on different cores
      // When we use this as part of our gmres polynomials, this can give 10% difference in some of
      // the coefficients on different cores which is not acceptable
      PetscCallMPI(MPI_Op_create(custom_reduction_tsqr, 0, &reduction_op_tsqr));
      built_custom_op_tsqr = PETSC_TRUE;
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode start_tsqr(MPI_Comm MPI_COMM_MATRIX, PetscReal A[], PetscInt m_size, PetscInt n_size, tsqr_buffers *buffers)
{
   // Starts the compute for an asynchronous tsqr, have to call finish_tsqr_parallel outside of this routine
   // A has the local mxn component of data we want to do a QR on (column-major, leading dimension m_size)
   // This routine only computes the R, we don't need Q
   // After finish_tsqr_parallel has been called, buffers->R_buffer_receive holds R

   // ~~~~~~
   PetscReal *work, *tau, *R_pointer;
   PetscBool *scale_row;
   // BLAS/LAPACK integer arguments must be PetscBLASInt
   PetscBLASInt lwork, info, m_bl, n_bl, lda_bl;
   PetscMPIInt comm_size, count;
   PetscInt row_length, i_loc, j_loc;
   // ~~~~~~

   PetscFunctionBegin;

   // Setup the custom mpi reduction if we haven't done it already
   PetscCall(setup_custom_tsqr_reduction());

   // Get the comm size
   PetscCallMPI(MPI_Comm_size(MPI_COMM_MATRIX, &comm_size));

   PetscCall(PetscMalloc1(n_size, &tau));
   PetscCall(PetscMalloc1(n_size, &scale_row));
   PetscCall(PetscMalloc1(n_size, &work));
   lwork = -1;

   // ~~~~~~~~~~
   // Do the local component of the QR
   // ~~~~~~~~~~

   // On coarse grids we can get to the point where we don't have a tall-skinny anymore
   // and no_of_row_blocks goes negative, in that case just use an ordinary QR
   // Or if dlatsqr isnt' available on the platform can just comment out this if statement and
   // the else below and use the ordinary QR here
   // (the Fortran keeps a commented out tall-skinny dlatsqr version here)

   // Skip the qr if we have zero local rows - this case can only occur here where m_size .le. n_size
   if (m_size != 0) {

      // ~~~~~~~~
      // This is the BLAS level 3 QR, but there is also a specific tall-skinny in
      // lapack 3.4 we use below
      // ~~~~~~~~
      // Start with a workspace query on the size of lwork
      // Kind-correct BLAS integer dimensions
      PetscCall(PetscBLASIntCast(m_size, &m_bl));
      PetscCall(PetscBLASIntCast(n_size, &n_bl));
      PetscCall(PetscBLASIntCast(m_size, &lda_bl));
      PetscCallBLAS("LAPACKgeqrf", LAPACKgeqrf_(&m_bl, &n_bl, &A[0], &lda_bl, tau, work, &lwork, &info));
      // int() is exact for all plausible sizes < 2^24 even when work is single precision
      PetscCall(PetscBLASIntCast((PetscInt)work[0], &lwork));
      PetscCall(PetscFree(work));
      PetscCall(PetscMalloc1(lwork, &work));

      // Now actually do the qr
      PetscCallBLAS("LAPACKgeqrf", LAPACKgeqrf_(&m_bl, &n_bl, &A[0], &lda_bl, tau, work, &lwork, &info));
   }
   PetscCall(PetscFree(work));

   // We use this variable to note which rows need to be scaled by -1
   // This is so we can enforce a unique solution by enforcing positive diagonal entries
   // in R, different versions of lapack either do or don't enforce this
   // If you want to enforce positive diagonal enties the rows
   // of R are scaled by +- 1, and then the columns of Q by +- 1
   // (but we don't actually need Q so we don't bother scaling them)
   for (i_loc = 0; i_loc < n_size; i_loc++) scale_row[i_loc] = PETSC_FALSE;

   // Now our buffers are going to be of size n_size * n_size to hold the R block, but
   // we are also sticking an integer at the start which is just the integer n_size
   // as there is no way to tell both how big n_size is and how many chunks the mpi
   // library is making us reduce at the same time
   // ie in our buffer we send [n_size R(:,:)]
   // e.g., if the mpi library feeds our custom reduction routine a buffer of size
   // 64, how do we know if it is 16 chunks of size 2x2, or 4 chunks of size 4x4
   // We could just build an mpi derived data type and define a custom reduction on that
   // but this is reasonably simple and only involves sending a single extra double
   // which doesn't matter
   PetscCall(PetscCalloc1(n_size * n_size + 1, &buffers->R_buffer_receive));
   // n_size is smuggled as a real in buffer element 1 and read back with int()
   // (see Gmres_Poly finish path / custom_reduction_tsqr here). Exact in float up to 2^24,
   // so this survives single precision. Kept PetscReal with the TSQR chain.
   buffers->R_buffer_receive[0] = (PetscReal)n_size;

   // If we are in parallel we need to copy the local QR into R_buffer_send so it can be
   // part of the mpi reduction, where R_buffer_receive is filled by the reduction
   // In serial we can copy it directly to R_buffer_receive
   if (comm_size != 1) {
      PetscCall(PetscCalloc1(n_size * n_size + 1, &buffers->R_buffer_send));
      // n_size smuggled as a real in element 1 - exact in float up to 2^24
      buffers->R_buffer_send[0] = (PetscReal)n_size;
      // Just have a pointer pointing to the R block specifically for ease
      // (column-major n_size x n_size)
      R_pointer = &buffers->R_buffer_send[1];
   } else {
      R_pointer = &buffers->R_buffer_receive[1];
   }

   // Have to be careful here to only go up to the right size here
   // as in parallel we could have the case where a local process has m_size < n_size
   // but when we receive the reduction it will still be of size n_size, unless the
   // global_row size < n_size - this should never happen and should be detected outside this
   // routine
   row_length = PetscMin(m_size, n_size);
   for (i_loc = 0; i_loc < row_length; i_loc++) {
      // Record if we have to scale this row
      if (A[i_loc + i_loc * m_size] < 0.0) scale_row[i_loc] = PETSC_TRUE;

      // Do the copy
      for (j_loc = 0; j_loc <= i_loc; j_loc++) R_pointer[j_loc + i_loc * n_size] = A[j_loc + i_loc * m_size];
   }
   // Do all the extra columns if there are any
   // This is only relevant for the case where the m_size < n_size
   if (m_size != 0) {
      for (i_loc = m_size - 1; i_loc < n_size; i_loc++) {
         // Do the copy
         for (j_loc = 0; j_loc < row_length; j_loc++) R_pointer[j_loc + i_loc * n_size] = A[j_loc + i_loc * m_size];
      }
   }

   // Scale the rows, enforce uniqueness
   for (i_loc = 0; i_loc < n_size; i_loc++) {
      if (scale_row[i_loc]) {
         for (j_loc = 0; j_loc < n_size; j_loc++) R_pointer[i_loc + j_loc * n_size] = -R_pointer[i_loc + j_loc * n_size];
      }
   }

   // ~~~~~~~~~~~
   // This is where the parallel comms need to go on R, can ignore Q
   // ~~~~~~~~~~~
   // comm_size is always 1 in serial, so this non-blocking reduction only
   // happens in parallel
   if (comm_size != 1) {

      // Only need a single all reduce, this is the only comms outside of the
      // matvecs above
      buffers->request = MPI_REQUEST_NULL;
      // This is now a non-blocking allreduce, you have to finish this where needed
      PetscCall(PetscMPIIntCast(n_size * n_size + 1, &count));
      PetscCallMPI(MPI_Iallreduce(buffers->R_buffer_send, buffers->R_buffer_receive,
               count, MPIU_REAL,
               reduction_op_tsqr, MPI_COMM_MATRIX, &buffers->request));

      // ~~~~~
      // ~~~~~
      // This non-blocking is resolved in finish_tsqr_parallel, which you must call
      // outside this routine
      // ~~~~~
      // ~~~~~
   }

   PetscCall(PetscFree(tau));
   PetscCall(PetscFree(scale_row));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN void custom_reduction_tsqr(void *invec, void *inoutvec, int *len, MPI_Datatype *type)
{
   // This is a custom reduction for mpi that implements a tall-skinny qr in parallel
   // Takes stacked local R from different ranks and then calls a QR on those
   // This is the MPI_User_function of the custom MPI_Op, so it cannot return an
   // error code - any PETSc error inside it aborts, and the LAPACK calls are made
   // directly rather than through PetscCallBLAS (which returns an error code in
   // debug builds); the Fortran did not check their info either

   // ~~~~~~~~~
   PetscReal *invec_r, *inoutvec_r;
   PetscInt number_chunks, i_loc, chunk_size, row, col;
   PetscInt start_chunk, j_loc, nb, n_size;
   // BLAS/LAPACK integer arguments must be PetscBLASInt
   PetscBLASInt lwork, info, m_bl, n_bl, lda_bl;
   PetscReal *R_stacked, *work, *tau;
   // ~~~~~~~~~

   (void)type;

   // ~~~~~~~~~~
   // The invec and inoutvec will be structured with number_chunks of [n_size R(:,:)] like:
   // [n_size R(:,:), n_size R(:,:), n_size R(:,:) ...]
   // We have to compute
   // do i = 1, number_chunks
   //  inoutvec(i) = invec(i) o inoutvec(i)
   // where o is our reduction
   // ~~~~~~~~~~

   // Let's get pointers we can use
   invec_r = (PetscReal *)invec;
   inoutvec_r = (PetscReal *)inoutvec;

   // We know the n_size is the first entry in each block
   n_size = (PetscInt)invec_r[0];

   // len is the number of actual doubles that have been requested to be
   // reduced here, but we know they come in chunks of [n_size R(:,:)]
   // which are of size (n_size**2 + 1)
   chunk_size = n_size * n_size + 1;
   number_chunks = *len / chunk_size;

   // Let's stack the two different R's on top of each other, the first one
   // from invec and the first one from inoutvec
   // (column-major, 2 * n_size x n_size)
   PetscCallAbort(PETSC_COMM_SELF, PetscMalloc1(2 * n_size * n_size, &R_stacked));

   // Use nb = 2 with dtpqrt
   //nb = 2
   nb = 1;
   PetscCallAbort(PETSC_COMM_SELF, PetscMalloc1(nb * n_size, &tau));
   PetscCallAbort(PETSC_COMM_SELF, PetscMalloc1(nb * n_size, &work));

   // we have to do a single qr on the stacked 2x1 R blocks
   // [ R_invec(i)    ]
   // [ R_inoutvec(i) ]
   // for each chunk
   for (i_loc = 0; i_loc < number_chunks; i_loc++) {

      // Compute the bounds, starting from the R block (ie ignoring n_size at invec(1) and inoutvec(1))
      start_chunk = 1 + i_loc * chunk_size;

      for (col = 0; col < n_size; col++) {
         for (row = 0; row < n_size; row++) {
            // Copy in the R from invec for this chunk
            R_stacked[row + col * 2 * n_size] = invec_r[start_chunk + row + col * n_size];
            // Then copy in the R from inoutvec for this chunk
            R_stacked[n_size + row + col * 2 * n_size] = inoutvec_r[start_chunk + row + col * n_size];
         }
      }

      // ~~~~~~~~~~
      // Now do the QR on R_stacked
      // This will do it in place and overwrite the top R block in R_stacked
      // ~~~~~~~~~~
      // The Fortran keeps a commented out version that uses ?tpqrt, which specifically
      // assumes this sort of stacked tridiagonal form (with nb = 2)

      // Here is some code that just uses the ordinary QR fatorisation in case dtpqrt
      // isn't available on the platform - if you use this set nb = 1 above
      // There won't be a big flop difference for such small matrices

      //Start with a workspace query on the size of lwork
      lwork = -1;
      // Kind-correct BLAS integer dimensions
      PetscCallAbort(PETSC_COMM_SELF, PetscBLASIntCast(2 * n_size, &m_bl));
      PetscCallAbort(PETSC_COMM_SELF, PetscBLASIntCast(n_size, &n_bl));
      PetscCallAbort(PETSC_COMM_SELF, PetscBLASIntCast(2 * n_size, &lda_bl));
      LAPACKgeqrf_(&m_bl, &n_bl, &R_stacked[0], &lda_bl, tau, work, &lwork, &info);
      // int() is exact for all plausible sizes < 2^24 even when work is single precision
      PetscCallAbort(PETSC_COMM_SELF, PetscBLASIntCast((PetscInt)work[0], &lwork));
      PetscCallAbort(PETSC_COMM_SELF, PetscFree(work));
      PetscCallAbort(PETSC_COMM_SELF, PetscMalloc1(lwork, &work));

      // Now actually do the qr
      LAPACKgeqrf_(&m_bl, &n_bl, &R_stacked[0], &lda_bl, tau, work, &lwork, &info);

      // We can enforce a unique solution here by enforcing positive diagonal entries
      // in R, different versions of lapack either do or don't enforce this
      // If you want to enforce positive diagonal enties then just scale the rows
      // of R by +- 1, and then the columns of Q by +- 1
      // (but we don't actually need Q so we don't bother scaling them)
      for (j_loc = 0; j_loc < n_size; j_loc++) {
         if (R_stacked[j_loc + j_loc * 2 * n_size] < 0) {
            for (col = 0; col < n_size; col++) R_stacked[j_loc + col * 2 * n_size] = -R_stacked[j_loc + col * 2 * n_size];
         }
      }

      // Now copy back the result which has been done in place in R_stacked into inoutvec
      // We don't have to change n_size so we're not touching inoutvec(1)
      for (col = 0; col < n_size; col++) {
         for (row = 0; row < n_size; row++) {
            inoutvec_r[start_chunk + row + col * n_size] = R_stacked[row + col * 2 * n_size];
         }
      }
   }

   PetscCallAbort(PETSC_COMM_SELF, PetscFree(R_stacked));
   PetscCallAbort(PETSC_COMM_SELF, PetscFree(tau));
   PetscCallAbort(PETSC_COMM_SELF, PetscFree(work));
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN PetscErrorCode finish_tsqr_parallel(tsqr_buffers *buffers)
{
   // Finishes the non-blocking all reduce in parallel
   // Once this has been called the buffers->R_buffer_receive used will have R in it

   PetscFunctionBegin;

   // We might want to call this on a sub communicator
   // if that's the case then if we're on a processor not on the subcomm
   // and we have nothing to do
   if (buffers->request == MPI_REQUEST_NULL) PetscFunctionReturn(PETSC_SUCCESS);

   // Finish the non-blocking comms
   PetscCallMPI(MPI_Wait(&buffers->request, MPI_STATUS_IGNORE));

   buffers->request = MPI_REQUEST_NULL;
   PetscCall(PetscFree(buffers->R_buffer_send));

   PetscFunctionReturn(PETSC_SUCCESS);
}
