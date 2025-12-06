module repartition

   use petscmat
   use c_petsc_interfaces
   use petsc_helper

#include "petsc/finclude/petscmat.h"
                
   implicit none
   public

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! Some helper functions that handle repartitioning of petsc matrices
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------      

   contains 

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine compute_mat_ratio_local_nonlocal_nnzs(input_mat, no_active_cores, ratio)
      
      ! Computes the local to non local ratios of nnzs in input_mat

      ! ~~~~~~
      type(tMat), target, intent(in)      :: input_mat
      PetscInt, intent(in)                :: no_active_cores
      PetscReal, intent(out)                   :: ratio

      ! Local
      PetscInt :: local_nnzs, off_proc_nnzs
      integer :: errorcode
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX      
      PetscReal :: ratio_parallel

      ! ~~~~~~  

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)      
      call MatGetNNZs_both_c(input_mat%v, local_nnzs, off_proc_nnzs)
      
      ! ~~~~~~~~~~~
      ! Get the ratio of local to non-local nnzs
      ! ~~~~~~~~~~~
      ! If a processor is entirely local, don't include it in the ratio
      ! It's a bit hard to decide here what to do, as we don't want to comm how many 
      ! processors actually have non local work
      ! So the proxy for that is no_active_cores but
      ! there is no guarantee that that many procs all have nonlocal entries
      if (off_proc_nnzs == 0) then
         ratio = 0
      else
         ratio = dble(local_nnzs)/dble(off_proc_nnzs)
      end if

      call MPI_Allreduce(ratio, ratio_parallel, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_MATRIX, errorcode)

      ratio = ratio_parallel  
      ! Only divide by the number of processors that are active
      ratio = ratio / dble(no_active_cores)

   end subroutine      
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine calculate_repartition(input_mat, proc_stride, no_active_cores, simple, index)
      
      ! Return an IS that represents a repartitioning of the matrix
      ! Simple being true is just processor aggregation to load balance
      ! partitions

      ! ~~~~~~
      type(tMat), target, intent(in)      :: input_mat
      PetscInt, intent(in)                :: proc_stride
      PetscInt, intent(inout)             :: no_active_cores
      logical, intent(in)                 :: simple
      type(tIS), intent(out)              :: index

      ! Local
      PetscInt :: global_rows, global_cols
      integer :: errorcode, comm_size
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX      
      type(tMat) :: adj, input_transpose
      PetscInt :: local_size_is, start_is
      integer(c_long_long) :: A_array, index_array
      PetscInt, parameter :: one=1, zero=0

      ! ~~~~~~  

      call PetscObjectGetComm(input_mat, MPI_COMM_MATRIX, ierr)  
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)      

      ! If doing simple processor aggregation
      if (simple) then

         call MatGetSize(input_mat, global_rows, global_cols, ierr)
         call GenerateIS_ProcAgglomeration_c(proc_stride, global_rows, local_size_is, start_is)

         ! Specifically on comm_world as we aren't going onto a subcomm
         ! If we're on a processor we wish to accumulate onto the local_size_is will have
         ! other enties, it will be zero on processors that will have all entries removed from
         call ISCreateStride(MPI_COMM_WORLD, local_size_is, start_is, one, index, ierr)      

      ! Else call one of the partitioners through petsc
      else

         ! Number of cores we want dofs on
         no_active_cores = floor(dble(comm_size)/dble(proc_stride))
         ! Be careful of rounding!
         if (no_active_cores == 0) no_active_cores = 1

         ! Have to symmetrize the input matrix or it won't work in parmetis
         ! as it expects a symmetric graph
         call MatTranspose(input_mat, MAT_INITIAL_MATRIX, input_transpose, ierr)
         call MatAXPYWrapper(input_transpose, 1d0, input_mat)

         ! Compute the adjancency graph of the symmetrized input matrix
         call MatConvert(input_transpose, MATMPIADJ, MAT_INITIAL_MATRIX, adj, ierr)

         A_array = adj%v
         call MatPartitioning_c(A_array, no_active_cores, proc_stride, index_array)

         ! Assign the index to the IS pointer we get back from c
         index%v = index_array

         ! Destroy the adjacency matrices
         call MatDestroy(adj, ierr)
         call MatDestroy(input_transpose, ierr)

      end if

   end subroutine

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine MatMPICreateNonemptySubcomm(input_mat, on_subcomm, output_mat)
      
      ! Just calls MatMPICreateNonemptySubcomm_c

      ! ~~~~~~
      type(tMat), target, intent(in)      :: input_mat
      logical, intent(out)                :: on_subcomm
      type(tMat), target, intent(inout)   :: output_mat

      integer(c_long_long) :: A_array, B_array
      integer(c_int)       :: on_subcomm_int

      ! ~~~~~~  

      A_array = input_mat%v
      call MatMPICreateNonemptySubcomm_c(A_array, on_subcomm_int, B_array)
      if (on_subcomm_int == 1) then
         on_subcomm = .TRUE.
      else
         on_subcomm = .FALSE.
      end if
      ! Assign the index to the mat we get back from c
      output_mat%v = B_array

   end subroutine     

   !-------------------------------------------------------------------------------------------------------------------------------

end module repartition

