module tsqr

   ! Get MPI types from PETSc as it can control whether using mpi or mpif08
   use petscmat

#include "petsc/finclude/petscmat.h"
   
   implicit none

   ! ~~~~~~~~~
   ! These globals are gross, but we use them to make sure we only create our custom reduction once
   ! in case its expensive to create it
   ! ~~~~~~~~~
   logical, protected :: built_custom_op_tsqr = .false.
   MPIU_Op, protected :: reduction_op_tsqr

   ! ~~~~~~~~
   ! Stores data for the asynchronous comms
   ! ~~~~~~~~
   type tsqr_buffers
      MPIU_Request                          :: request = MPI_REQUEST_NULL
      PetscReal, dimension(:), allocatable  :: R_buffer_send, R_buffer_receive
      ! In case this comms request is done on a matrix on a subcomm, we 
      ! need to keep a pointer to it
      type(tMat)                       :: matrix
      ! Has the user asked us to be on a subcomm
      logical                          :: subcomm = .FALSE.
      ! Did we actually end up on a subcomm
      logical                          :: on_subcomm = .FALSE.
      integer                          :: proc_stride = 1
   end type tsqr_buffers   

   public 
   
   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine setup_custom_tsqr_reduction()

      ! Creates our custom reduction

      ! ~~~~~~
      integer :: errorcode
      ! ~~~~~~    

      ! Point at the custom reduction function
      if (.NOT. built_custom_op_tsqr) then
         ! We should be able to set commute as true, as the order doesn't matter for QRs
         ! But I have bumped into cases where due to rounding (typically in low-rank systems)
         ! the small entries in the resulting R can differ on different cores
         ! When we use this as part of our gmres polynomials, this can give 10% difference in some of 
         ! the coefficients on different cores which is not acceptable
         call MPI_Op_create(custom_reduction_tsqr, .FALSE., reduction_op_tsqr, errorcode)
         built_custom_op_tsqr = .true.
      end if
      
   end subroutine setup_custom_tsqr_reduction   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine start_tsqr(MPI_COMM_MATRIX, A, buffers)

      ! Starts the compute for an asynchronous tsqr, have to call finish_tsqr_parallel outside of this routine
      ! A has the local mxn component of data we want to do a QR on
      ! This routine only computes the R, we don't need Q
      ! After finish_tsqr_parallel has been called, buffers%R_buffer_receive holds R

      ! ~~~~~~
      MPIU_Comm, intent(in)                                     :: MPI_COMM_MATRIX
      PetscReal, dimension(:, :), intent(inout)                     :: A
      type(tsqr_buffers), target, intent(inout)                :: buffers
      
      PetscReal, dimension(:), allocatable :: work
      ! PetscReal, dimension(:), allocatable :: T
      PetscReal, dimension(size(A, 2)) :: tau
      logical, dimension(size(A, 2)) :: scale_row
      PetscReal, dimension(:,:), pointer :: R_pointer

      ! integer :: column_block_size, row_block_size, no_of_row_blocks
      integer :: lwork
      integer :: m_size, n_size, errorcode, comm_size, row_length, i_loc
      ! ~~~~~~    

      ! Get the local sizes
      m_size = size(A, 1)
      n_size = size(A, 2)      

      ! Setup the custom mpi reduction if we haven't done it already
      call setup_custom_tsqr_reduction()      

      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)       

      allocate(work(n_size))
      lwork = -1

      ! ~~~~~~~~~~
      ! Do the local component of the QR
      ! ~~~~~~~~~~

      ! On coarse grids we can get to the point where we don't have a tall-skinny anymore
      ! and no_of_row_blocks goes negative, in that case just use an ordinary QR
      ! Or if dlatsqr isnt' available on the platform can just comment out this if statement and 
      ! the else below and use the ordinary QR here
      !if (m_size .le. n_size) then

         ! Skip the qr if we have zero local rows - this case can only occur here where m_size .le. n_size
         if (m_size /= 0) then

            ! ~~~~~~~~
            ! This is the BLAS level 3 QR, but there is also a specific tall-skinny in 
            ! lapack 3.4 we use below
            ! ~~~~~~~~
            ! Start with a workspace query on the size of lwork
            call dgeqrf(m_size, n_size, &
                        A(1, 1), m_size, &
                        tau, work, lwork, errorcode)
            lwork = int(work(1))
            deallocate(work)
            allocate(work(lwork))

            ! Now actually do the qr
            call dgeqrf(m_size, n_size, &
                        A(1, 1), m_size, &
                        tau, work, lwork, errorcode)  
         end if
         deallocate(work)            

      ! ~~~~~~~~
      ! Tall-skinny specific QR - hopefully faster
      ! ~~~~~~~~                         
      ! else
      !    ! Uncomment the column_block_size variables and T above if using this
      !    ! Start with a workspace query on the size of lwork
      !    ! Just set the column block size to be the number of columns
      !    column_block_size = n_size
      !    ! Have no idea about the block sizes, but I've seen stuff saying a row block size
      !    ! of 64 is a good start         
      !    row_block_size = 64
      !    if (row_block_size > m_size) row_block_size = m_size

      !    no_of_row_blocks = ceiling(dble(m_size-(n_size))/dble(row_block_size-(n_size)))

      !    allocate(T(column_block_size * (n_size) * no_of_row_blocks))
      !    call dlatsqr(m_size, n_size, &
      !                row_block_size, column_block_size, &
      !                A(1, 1), m_size, &
      !                T, column_block_size, &
      !                work, lwork, errorcode)  
      !    lwork = int(work(1))
      !    deallocate(work)
      !    allocate(work(lwork))    
         
      !    ! Now do the actual qr
      !    call dlatsqr(m_size, n_size, &
      !                row_block_size, column_block_size, &
      !                A(1, 1), m_size, &
      !                T, column_block_size, &
      !                work, lwork, errorcode)       
      !    ! Don't need the bits of Q
      !    deallocate(T, work)

      ! end if

  
      ! We use this variable to note which rows need to be scaled by -1
      ! This is so we can enforce a unique solution by enforcing positive diagonal entries
      ! in R, different versions of lapack either do or don't enforce this
      ! If you want to enforce positive diagonal enties the rows 
      ! of R are scaled by +- 1, and then the columns of Q by +- 1
      ! (but we don't actually need Q so we don't bother scaling them)       
      scale_row = .FALSE.  

      ! Now our buffers are going to be of size n_size * n_size to hold the R block, but 
      ! we are also sticking an integer at the start which is just the integer n_size
      ! as there is no way to tell both how big n_size is and how many chunks the mpi
      ! library is making us reduce at the same time
      ! ie in our buffer we send [n_size R(:,:)]
      ! e.g., if the mpi library feeds our custom reduction routine a buffer of size
      ! 64, how do we know if it is 16 chunks of size 2x2, or 4 chunks of size 4x4
      ! We could just build an mpi derived data type and define a custom reduction on that
      ! but this is reasonably simple and only involves sending a single extra double
      ! which doesn't matter
      allocate(buffers%R_buffer_receive(n_size * n_size + 1))
      buffers%R_buffer_receive = 0
      buffers%R_buffer_receive(1) = dble(n_size)

      ! If we are in parallel we need to copy the local QR into R_buffer_send so it can be 
      ! part of the mpi reduction, where R_buffer_receive is filled by the reduction
      ! In serial we can copy it directly to R_buffer_receive
      if (comm_size/=1) then
         allocate(buffers%R_buffer_send(n_size * n_size + 1))
         buffers%R_buffer_send = 0      
         buffers%R_buffer_send(1) = dble(n_size)
         ! Just have a pointer pointing to the R block specifically for ease
         R_pointer(1:n_size, 1:n_size) => buffers%R_buffer_send(2:n_size * n_size + 1)
      else
         R_pointer(1:n_size, 1:n_size) => buffers%R_buffer_receive(2:n_size * n_size + 1)
      end if

      ! Have to be careful here to only go up to the right size here
      ! as in parallel we could have the case where a local process has m_size < n_size 
      ! but when we receive the reduction it will still be of size n_size, unless the 
      ! global_row size < n_size - this should never happen and should be detected outside this 
      ! routine
      row_length = min(m_size, n_size)     
      do i_loc = 1, row_length
         ! Record if we have to scale this row
         if (A(i_loc, i_loc) < 0d0) scale_row(i_loc) = .TRUE.

         ! Do the copy
         R_pointer(1:i_loc, i_loc) = A(1:i_loc, i_loc)
      end do  
      ! Do all the extra columns if there are any
      ! This is only relevant for the case where the m_size < n_size
      if (m_size /= 0) then
         do i_loc = m_size, n_size
            ! Do the copy
            R_pointer(1:row_length, i_loc) = A(1:row_length, i_loc)            
         end do
      end if

      ! Scale the rows, enforce uniqueness
      do i_loc = 1, n_size
         if (scale_row(i_loc)) R_pointer(i_loc, :) = R_pointer(i_loc, :) * (-1d0)
      end do       

      ! ~~~~~~~~~~~
      ! This is where the parallel comms need to go on R, can ignore Q
      ! ~~~~~~~~~~~  
      if (comm_size/=1) then      

         ! Only need a single all reduce, this is the only comms outside of the 
         ! matvecs above
         buffers%request = MPI_REQUEST_NULL
         ! This is now a non-blocking allreduce, you have to finish this where needed
         call MPI_IAllreduce(buffers%R_buffer_send, buffers%R_buffer_receive, &
                  n_size * n_size + 1, MPI_DOUBLE_PRECISION, &
                  reduction_op_tsqr, MPI_COMM_MATRIX, buffers%request, errorcode)
         if (errorcode /= MPI_SUCCESS) then
            print *, "MPI_IAllreduce failed"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
         end if 

         ! ~~~~~
         ! ~~~~~
         ! This non-blocking is resolved in finish_tsqr_parallel, which you must call 
         ! outside this routine
         ! ~~~~~
         ! ~~~~~

      end if
      
   end subroutine start_tsqr   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine custom_reduction_tsqr(invec, inoutvec, len, type) 

   ! This is a custom reduction for mpi that implements a tall-skinny qr in parallel
   ! Takes stacked local R from different ranks and then calls a QR on those
      
      ! ~~~~~~~~~
      use, intrinsic :: iso_c_binding, only : c_ptr, c_f_pointer 
      type(c_ptr), value :: invec, inoutvec 
      integer            :: len 
      MPIU_Datatype      :: type 

      PetscReal, pointer :: invec_r(:), inoutvec_r(:) 
      integer :: number_chunks, i_loc, lwork, chunk_size
      integer :: start_chunk, end_chunk, errorcode, j_loc, nb, n_size
      PetscReal, dimension(:, :), allocatable :: R_stacked
      PetscReal, dimension(:), allocatable :: work, tau
      ! ~~~~~~~~~

      ! ~~~~~~~~~~
      ! The invec and inoutvec will be structured with number_chunks of [n_size R(:,:)] like:
      ! [n_size R(:,:), n_size R(:,:), n_size R(:,:) ...]
      ! We have to compute
      ! do i = 1, number_chunks
      !  inoutvec(i) = invec(i) o inoutvec(i)
      ! where o is our reduction
      ! ~~~~~~~~~~

      ! Let's get pointers we can use
      call c_f_pointer(invec, invec_r, (/ len /) ) 
      call c_f_pointer(inoutvec, inoutvec_r, (/ len /) ) 

      ! We know the n_size is the first entry in each block
      n_size = int(invec_r(1))

      ! len is the number of actual doubles that have been requested to be 
      ! reduced here, but we know they come in chunks of [n_size R(:,:)]
      ! which are of size (n_size**2 + 1)
      chunk_size = n_size**2 + 1
      number_chunks = len / chunk_size

      ! Let's stack the two different R's on top of each other, the first one
      ! from invec and the first one from inoutvec
      allocate(R_stacked(2 * n_size, n_size))  

      ! Use nb = 2 with dtpqrt
      !nb = 2               
      nb = 1
      allocate(tau(nb * n_size))
      allocate(work(nb * n_size))

      ! we have to do a single qr on the stacked 2x1 R blocks
      ! [ R_invec(i)    ] 
      ! [ R_inoutvec(i) ]
      ! for each chunk
      do i_loc = 1, number_chunks   
         
         ! Compute the bounds, starting from the R block (ie ignoring n_size at invec(1) and inoutvec(1))
         start_chunk = 2 + (i_loc-1) * chunk_size
         end_chunk = start_chunk + n_size**2 - 1

         ! Copy in the R from invec for this chunk
         R_stacked(1:n_size, 1:n_size) = &
            reshape(invec_r(start_chunk:end_chunk), (/n_size,n_size/))  

         ! Then copy in the R from inoutvec for this chunk
         R_stacked(n_size+1:2*n_size, 1:n_size) = &
            reshape(inoutvec_r(start_chunk:end_chunk), (/n_size,n_size/))  
                  
         ! ~~~~~~~~~~
         ! Now do the QR on R_stacked
         ! This will do it in place and overwrite the top R block in R_stacked
         ! ~~~~~~~~~~
         ! This uses ?tpqrt which specifically assumes this sort of stacked tridiagonal form
         ! Setting M=L=N as we don't have the rectagonal component of B
         ! we just have two stacked upper triangles
         ! call dtpqrt(n_size, n_size, n_size, &
         !          nb, R_stacked(1, 1), size(R_stacked, 1), &
         !          R_stacked(n_size+1, 1), size(R_stacked, 1), &
         !          tau, nb, &
         !          work, errorcode)

         ! Here is some code that just uses the ordinary QR fatorisation in case dtpqrt
         ! isn't available on the platform - if you use this set nb = 1 above
         ! There won't be a big flop difference for such small matrices

         !Start with a workspace query on the size of lwork
         lwork = -1
         call dgeqrf(size(R_stacked, 1), size(R_stacked, 2), &
                     R_stacked(1, 1), size(R_stacked, 1), &
                     tau, work, lwork, errorcode)
         lwork = int(work(1))
         deallocate(work)
         allocate(work(lwork))

         ! Now actually do the qr
         call dgeqrf(size(R_stacked, 1), size(R_stacked, 2), &
                     R_stacked(1, 1), size(R_stacked, 1), &
                     tau, work, lwork, errorcode)  

         ! We can enforce a unique solution here by enforcing positive diagonal entries
         ! in R, different versions of lapack either do or don't enforce this
         ! If you want to enforce positive diagonal enties then just scale the rows 
         ! of R by +- 1, and then the columns of Q by +- 1
         ! (but we don't actually need Q so we don't bother scaling them)      
         do j_loc = 1, n_size
            if (R_stacked(j_loc, j_loc) < 0d0) R_stacked(j_loc, :) = R_stacked(j_loc, :) * (-1d0)
         end do   
         
         ! Now copy back the result which has been done in place in R_stacked into inoutvec
         ! We don't have to change n_size so we're not touching inoutvec(1)
         inoutvec_r(start_chunk:end_chunk) = &
            reshape(R_stacked(1:n_size, 1:n_size), (/ n_size**2 /))
                                       
      end do

      deallocate(R_stacked, tau, work)

    end subroutine custom_reduction_tsqr

! -------------------------------------------------------------------------------------------------------------------------------

    subroutine finish_tsqr_parallel(buffers)
      
      ! Finishes the non-blocking all reduce in parallel
      ! Once this has been called the buffers%R_buffer_receive used will have R in it

      ! ~~~~~~
      type(tsqr_buffers), intent(inout)   :: buffers

      integer :: errorcode
#if defined(PETSC_USE_MPI_F08)
      MPIU_Status :: status
#else
      MPIU_Status, dimension(MPI_STATUS_SIZE) :: status
#endif
      
      ! ~~~~~

      ! We might want to call this on a sub communicator
      ! if that's the case then if we're on a processor not on the subcomm
      ! and we have nothing to do
      if (buffers%request == MPI_REQUEST_NULL) return

      ! Finish the non-blocking comms
      call mpi_wait(buffers%request, &
                     status, errorcode)
         
      if (errorcode /= MPI_SUCCESS) then
         print *, "mpi_wait failed"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)         
      end if    
      buffers%request = MPI_REQUEST_NULL  
      if (allocated(buffers%R_buffer_send)) deallocate(buffers%R_buffer_send)
     
   end subroutine finish_tsqr_parallel 

! -------------------------------------------------------------------------------------------------------------------------------

end module tsqr

