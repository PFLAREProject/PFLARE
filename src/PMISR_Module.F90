module pmisr_module

   use iso_c_binding
   use petscmat
   use petsc_helper, only: kokkos_debug
   use c_petsc_interfaces, only: pmisr_kokkos, copy_cf_markers_d2h, &
         vecscatter_mat_begin_c, vecscatter_mat_end_c, vecscatter_mat_restore_c, &
         allreducesum_petscint_mine, boolscatter_mat_begin_c, boolscatter_mat_end_c, &
         boolscatter_mat_reverse_begin_c, boolscatter_mat_reverse_end_c
   use pflare_parameters, only: C_POINT, F_POINT

#include "petsc/finclude/petscmat.h"
#include "finclude/PETSc_ISO_Types.h"

   implicit none

   public   
   
   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine pmisr(strength_mat, max_luby_steps, pmis, cf_markers_local, zero_measure_c_point)

      ! Wrapper

      ! ~~~~~~

      type(tMat), target, intent(in)      :: strength_mat
      integer, intent(in)                 :: max_luby_steps
      logical, intent(in)                 :: pmis
      integer, dimension(:), allocatable, target, intent(inout) :: cf_markers_local
      logical, optional, intent(in)       :: zero_measure_c_point

#if defined(PETSC_HAVE_KOKKOS)                     
      integer(c_long_long) :: A_array
      PetscErrorCode :: ierr
      MatType :: mat_type
      integer :: pmis_int, zero_measure_c_point_int, seed_size, kfree, comm_rank, errorcode
      integer, dimension(:), allocatable :: seed
      PetscReal, dimension(:), allocatable, target :: measure_local
      PetscInt :: local_rows, local_cols
      MPIU_Comm :: MPI_COMM_MATRIX    
      type(c_ptr)  :: measure_local_ptr, cf_markers_local_ptr
      integer, dimension(:), allocatable :: cf_markers_local_two
#endif        
      ! ~~~~~~~~~~

#if defined(PETSC_HAVE_KOKKOS)    

      call MatGetType(strength_mat, mat_type, ierr)
      if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) then  

         call PetscObjectGetComm(strength_mat, MPI_COMM_MATRIX, ierr)    
         call MPI_Comm_rank(MPI_COMM_MATRIX, comm_rank, errorcode)                  

         A_array = strength_mat%v  
         pmis_int = 0
         if (pmis) pmis_int = 1
         zero_measure_c_point_int = 0
         if (present(zero_measure_c_point)) then
            if (zero_measure_c_point) zero_measure_c_point_int = 1
         end if

         ! Let's generate the random values on the host for now so they match
         ! for comparisons with pmisr_cpu
         call MatGetLocalSize(strength_mat, local_rows, local_cols, ierr)
         allocate(measure_local(local_rows))   
         call random_seed(size=seed_size)
         allocate(seed(seed_size))
         do kfree = 1, seed_size
            seed(kfree) = comm_rank + 1 + kfree
         end do   
         call random_seed(put=seed) 
         ! Fill the measure with random numbers
         call random_number(measure_local)
         deallocate(seed)   
         
         measure_local_ptr = c_loc(measure_local)

         allocate(cf_markers_local(local_rows))  
         cf_markers_local_ptr = c_loc(cf_markers_local)

         ! Creates a cf_markers on the device
         call pmisr_kokkos(A_array, max_luby_steps, pmis_int, measure_local_ptr, zero_measure_c_point_int)

         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then         

            ! Kokkos PMISR by default now doesn't copy back to the host, as any following ddc calls 
            ! use the device data
            call copy_cf_markers_d2h(cf_markers_local_ptr)
            call pmisr_cpu(strength_mat, max_luby_steps, pmis, cf_markers_local_two, zero_measure_c_point)  
            
            if (any(cf_markers_local /= cf_markers_local_two)) then

               ! do kfree = 1, local_rows
               !    if (cf_markers_local(kfree) /= cf_markers_local_two(kfree)) then
               !       print *, kfree, "no match", cf_markers_local(kfree), cf_markers_local_two(kfree)
               !    end if
               ! end do
               print *, "Kokkos and CPU versions of pmisr do not match"
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode) 
            end if
            deallocate(cf_markers_local_two)
         end if

      else
         call pmisr_cpu(strength_mat, max_luby_steps, pmis, cf_markers_local, zero_measure_c_point)       
      end if
#else
      call pmisr_cpu(strength_mat, max_luby_steps, pmis, cf_markers_local, zero_measure_c_point)
#endif        

      ! ~~~~~~ 

   end subroutine pmisr
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine pmisr_cpu(strength_mat, max_luby_steps, pmis, cf_markers_local, zero_measure_c_point)

      ! Let's do our own independent set with a Luby algorithm
      ! If PMIS is true, this is a traditional PMIS algorithm
      ! If PMIS is false, this is a PMISR
      ! PMISR swaps the C-F definition compared to a PMIS and 
      ! also checks the measure from smallest, rather than the largest
      ! PMISR should give an Aff with no off-diagonal strong connections 
      ! If you set positive max_luby_steps, it will avoid all parallel reductions
      ! by taking a fixed number of times in the Luby top loop

      ! ~~~~~~

      type(tMat), target, intent(in)      :: strength_mat
      integer, intent(in)                 :: max_luby_steps
      logical, intent(in)                 :: pmis
      integer, dimension(:), allocatable, intent(inout) :: cf_markers_local
      logical, optional, intent(in)       :: zero_measure_c_point

      ! Local
      PetscInt :: local_rows, local_cols, global_rows, global_cols
      PetscInt :: global_row_start, global_row_end_plus_one, ifree, ncols
      PetscInt :: rows_ao, cols_ao, n_ad, n_ao
      integer :: comm_size, seed_size
      integer :: comm_rank, errorcode       
      integer :: kfree
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX      
      integer, dimension(:), allocatable :: seed
      PetscReal, dimension(:), allocatable :: measure_local
      type(tMat) :: Ad, Ao
      PetscInt, dimension(:), pointer :: colmap
      PetscInt, dimension(:), pointer :: ad_ia, ad_ja, ao_ia, ao_ja
      PetscInt :: shift = 0
      PetscBool :: symmetric = PETSC_FALSE, inodecompressed = PETSC_FALSE, done
      logical :: zero_measure_c = .FALSE.  

      ! ~~~~~~           

      if (present(zero_measure_c_point)) zero_measure_c = zero_measure_c_point

      ! Get the comm size 
      call PetscObjectGetComm(strength_mat, MPI_COMM_MATRIX, ierr)    
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)
      ! Get the comm rank 
      call MPI_Comm_rank(MPI_COMM_MATRIX, comm_rank, errorcode)      

      ! Get the local sizes
      call MatGetLocalSize(strength_mat, local_rows, local_cols, ierr)
      call MatGetSize(strength_mat, global_rows, global_cols, ierr)      
      call MatGetOwnershipRange(strength_mat, global_row_start, global_row_end_plus_one, ierr)   

      if (comm_size /= 1) then
         call MatMPIAIJGetSeqAIJ(strength_mat, Ad, Ao, colmap, ierr) 
         ! We know the col size of Ao is the size of colmap, the number of non-zero offprocessor columns
         call MatGetSize(Ao, rows_ao, cols_ao, ierr)    
      else
         Ad = strength_mat    
      end if

      ! ~~~~~~~~
      ! Get pointers to the sequential diagonal and off diagonal aij structures 
      ! ~~~~~~~~
      call MatGetRowIJ(Ad,shift,symmetric,inodecompressed,n_ad,ad_ia,ad_ja,done,ierr) 
      if (.NOT. done) then
         print *, "Pointers not set in call to MatGetRowIJ"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      if (comm_size /= 1) then
         call MatGetRowIJ(Ao,shift,symmetric,inodecompressed,n_ao,ao_ia,ao_ja,done,ierr) 
         if (.NOT. done) then
            print *, "Pointers not set in call to MatGetRowIJ"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
         end if
      end if      
      ! ~~~~~~~~~~      

      ! Get the number of connections in S
      allocate(measure_local(local_rows))
      allocate(cf_markers_local(local_rows))  
      cf_markers_local = 0

      ! ~~~~~~~~~~~~
      ! Seed the measure_local between 0 and 1
      ! ~~~~~~~~~~~~
      call random_seed(size=seed_size)
      allocate(seed(seed_size))
      do kfree = 1, seed_size
         seed(kfree) = comm_rank + 1 + kfree
      end do   
      call random_seed(put=seed) 

      ! To get the same results regardless of number of processors, you can 
      ! force the random number on each node to match across all processors
      ! This is tricky to do, given the numbering of rows is different in parallel 
      ! I did code up a version that used the unique spatial node positions to seed the random 
      ! number generator and test that and it works the same regardless of num of procs
      ! so I'm fairly confident things are correct

      ! Fill the measure with random numbers
      call random_number(measure_local)
      deallocate(seed)
  
      ! ~~~~~~~~~~      

      ! ~~~~~~~~~~~~
      ! Add the number of connections in S to the randomly seeded measure_local
      ! The number of connections is just equal to a matvec with a vec of all ones and the strength_mat
      ! We don't have to bother with a matvec though as we know the strenth_mat has entries of one
      ! ~~~~~~~~~~~~
      do ifree = 1, local_rows     
      
         ! Do local component
         ncols = ad_ia(ifree+1) - ad_ia(ifree)      
         measure_local(ifree) = measure_local(ifree) + ncols

         ! Do non local component
         if (comm_size /= 1) then
            ncols = ao_ia(ifree+1) - ao_ia(ifree)      
            measure_local(ifree) = measure_local(ifree) + ncols
         end if
      end do    
      
      ! Restore the sequantial pointers once we're done
      call MatRestoreRowIJ(Ad,shift,symmetric,inodecompressed,n_ad,ad_ia,ad_ja,done,ierr) 
      if (comm_size /= 1) then
         call MatRestoreRowIJ(Ao,shift,symmetric,inodecompressed,n_ao,ao_ia,ao_ja,done,ierr) 
      end if         

      ! If PMIS then we want to search the measure based on the largest entry
      ! PMISR searches the measure based on the smallest entry
      ! We just let the measure be negative rather than change the .ge. comparison 
      ! in our Luby below
      if (pmis) measure_local = measure_local * (-1)
      
      call pmisr_existing_measure_cf_markers(strength_mat, max_luby_steps, pmis, &
               measure_local, cf_markers_local, zero_measure_c_point)

      deallocate(measure_local)
      ! If PMIS then we swap the CF markers from PMISR
      if (pmis) then
         cf_markers_local = cf_markers_local * (-1)
      end if               

   end subroutine pmisr_cpu  

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine pmisr_existing_measure_cf_markers(strength_mat, max_luby_steps, pmis, &
                  measure_local, cf_markers_local, zero_measure_c_point)

      ! PMISR implementation that takes an existing measure_local and cf_markers_local 
      ! and then does the Luby algorithm to assign the rest of the CF markers

      ! ~~~~~~

      type(tMat), target, intent(in)       :: strength_mat
      integer, intent(in)                  :: max_luby_steps
      logical, intent(in)                  :: pmis
      PetscReal, dimension(:), allocatable :: measure_local
      integer, dimension(:), intent(inout) :: cf_markers_local
      logical, optional, intent(in)        :: zero_measure_c_point

      ! Local
      PetscInt :: local_rows, local_cols, global_rows, global_cols
      PetscInt :: global_row_start, global_row_end_plus_one, ifree
      PetscInt :: jfree
      PetscInt :: rows_ao, cols_ao, n_ad, n_ao
      PetscInt :: counter_undecided, counter_in_set_start, counter_parallel
      integer :: comm_size, loops_through
      integer :: comm_rank, errorcode       
      PetscErrorCode :: ierr
      MPIU_Comm :: MPI_COMM_MATRIX      
      PFLARE_PETSCBOOL_C_TYPE, dimension(:), allocatable :: in_set_this_loop
      PFLARE_PETSCBOOL_C_TYPE, dimension(:), allocatable, target :: assigned_local, assigned_nonlocal
      type(c_ptr) :: measure_nonlocal_ptr=c_null_ptr, assigned_local_ptr=c_null_ptr, assigned_nonlocal_ptr=c_null_ptr
      real(c_double), pointer :: measure_nonlocal(:) => null()
      type(tMat) :: Ad, Ao
      type(tVec) :: measure_vec
      PetscInt, dimension(:), pointer :: colmap
      integer(c_long_long) :: A_array, vec_long
      PetscInt, dimension(:), pointer :: ad_ia, ad_ja, ao_ia, ao_ja
      PetscInt :: shift = 0
      PetscBool :: symmetric = PETSC_FALSE, inodecompressed = PETSC_FALSE, done
      logical :: zero_measure_c = .FALSE.  
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0

      ! ~~~~~~           

      if (present(zero_measure_c_point)) zero_measure_c = zero_measure_c_point

      ! Get the comm size 
      call PetscObjectGetComm(strength_mat, MPI_COMM_MATRIX, ierr)    
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)
      ! Get the comm rank 
      call MPI_Comm_rank(MPI_COMM_MATRIX, comm_rank, errorcode)      

      ! Get the local sizes
      call MatGetLocalSize(strength_mat, local_rows, local_cols, ierr)
      call MatGetSize(strength_mat, global_rows, global_cols, ierr)      
      call MatGetOwnershipRange(strength_mat, global_row_start, global_row_end_plus_one, ierr)   

      if (comm_size /= 1) then
         call MatMPIAIJGetSeqAIJ(strength_mat, Ad, Ao, colmap, ierr) 
         ! We know the col size of Ao is the size of colmap, the number of non-zero offprocessor columns
         call MatGetSize(Ao, rows_ao, cols_ao, ierr)    
      else
         Ad = strength_mat    
      end if

      ! ~~~~~~~~
      ! Get pointers to the sequential diagonal and off diagonal aij structures 
      ! ~~~~~~~~
      call MatGetRowIJ(Ad,shift,symmetric,inodecompressed,n_ad,ad_ia,ad_ja,done,ierr) 
      if (.NOT. done) then
         print *, "Pointers not set in call to MatGetRowIJ"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      if (comm_size /= 1) then
         call MatGetRowIJ(Ao,shift,symmetric,inodecompressed,n_ao,ao_ia,ao_ja,done,ierr) 
         if (.NOT. done) then
            print *, "Pointers not set in call to MatGetRowIJ"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
         end if
      end if      
      ! ~~~~~~~~~~      

      ! Get the number of connections in S
      allocate(in_set_this_loop(local_rows))
      allocate(assigned_local(local_rows))
        
      ! ~~~~~~~~~~~~
      ! Create parallel vec and scatter the measure
      ! ~~~~~~~~~~~~
      if (comm_size/=1) then

         ! This is fine being mpi type specifically as strength_mat is always a mataij
         call VecCreateMPIWithArray(MPI_COMM_MATRIX, one, &
            local_rows, global_rows, measure_local, measure_vec, ierr)

         A_array = strength_mat%v
         vec_long = measure_vec%v
         ! We're just going to use the existing lvec to scatter the measure
         ! Have to call restore after we're done with lvec (ie measure_nonlocal_ptr)
         call vecscatter_mat_begin_c(A_array, vec_long, measure_nonlocal_ptr)
         call vecscatter_mat_end_c(A_array, vec_long, measure_nonlocal_ptr)
         ! This is the lvec so we have to make sure we don't do a matvec anywhere 
         ! before calling restore
         call c_f_pointer(measure_nonlocal_ptr, measure_nonlocal, shape=[cols_ao])

         allocate(assigned_nonlocal(cols_ao))
         assigned_local_ptr = c_loc(assigned_local)
         assigned_nonlocal_ptr = c_loc(assigned_nonlocal)
      else
         ! Need to avoid uninitialised warning
         allocate(assigned_nonlocal(0))
      end if

      ! ~~~~~~~~~~~~
      ! Initialise the set
      ! ~~~~~~~~~~~~
      counter_in_set_start = 0
      assigned_local = .FALSE.
      assigned_nonlocal = .FALSE.

      ! If already assigned by the input
      do ifree = 1, local_rows
         if (cf_markers_local(ifree) /= 0) assigned_local(ifree) = .TRUE.         
      end do

      do ifree = 1, local_rows

         ! Skip if already assigned
         if (assigned_local(ifree)) then
            counter_in_set_start = counter_in_set_start + 1
            cycle
         end if

         ! If there are no strong neighbours (not measure_local == 0 as we have added a random number to it)
         ! then we treat it special
         ! Absolute value here given measure_local could be negative (pmis) or positive (pmisr)
         if (abs(measure_local(ifree)) < 1) then

            ! Assign this node
            assigned_local(ifree) = .TRUE.

            ! This is typically enabled in a second pass of PMIS just on C points 
            ! (ie aggressive coarsening based on MIS(MIS(1))), we want to keep  
            ! C-points with no other strong C connections as C points
            if (zero_measure_c) then
               if (pmis) then
                  ! Set as F here but reversed below to become C
                  cf_markers_local(ifree) = F_POINT
               else
                  ! Becomes C
                  cf_markers_local(ifree) = C_POINT
               end if  
            else
               if (pmis) then
                  ! Set as C here but reversed below to become F
                  ! Otherwise dirichlet conditions persist down onto the coarsest grid
                  cf_markers_local(ifree) = C_POINT
               else
                  ! Becomes F
                  cf_markers_local(ifree) = F_POINT
               end if
            end if
            counter_in_set_start = counter_in_set_start + 1
         end if
      end do       

      ! Check the total number of undecided in parallel
      if (max_luby_steps < 0) then
         counter_undecided = local_rows - counter_in_set_start
         ! Parallel reduction!
         ! This is just an allreduce sum, but we can't use MPIU_INTEGER, as if we call the pmisr
         ! cf splitting from C it is not defined - also have to pass the matrix so we can get the comm
         ! given they're different in C and fortran
         A_array = strength_mat%v
         call allreducesum_petscint_mine(A_array, counter_undecided, counter_parallel)
         counter_undecided = counter_parallel

      ! If we're doing a fixed number of steps, then we don't care
      ! how many undecided nodes we have - have to take care here not to use
      ! local_rows for counter_undecided, as we may have zero DOFs on some procs
      ! but we have to enter the loop below for the collective scatters 
      else
         counter_undecided = 1
      end if

      ! ~~~~~~~~~~~~
      ! Now go through the outer Luby loop
      ! ~~~~~~~~~~~~      

      ! Let's keep track of how many times we go through the loops
      loops_through = -1

      do while (counter_undecided /= 0)   

         ! If max_luby_steps is positive, then we only take that many times through this top loop
         ! We typically find 2-3 iterations decides >99% of the nodes 
         ! and a fixed number of outer loops means we don't have to do any parallel reductions
         ! We will do redundant nearest neighbour comms in the case we have already 
         ! finished deciding all the nodes, but who cares
         ! Any undecided nodes just get turned into C points
         ! We can do this as we know we won't ruin Aff by doing so, unlike in a normal multigrid
         if (max_luby_steps > 0 .AND. max_luby_steps+1 == -loops_through) exit

         ! ~~~~~~~~~
         ! Start the async broadcast of assigned_local to assigned_nonlocal
         ! ~~~~~~~~~
         if (comm_size /= 1) then
            call boolscatter_mat_begin_c(A_array, assigned_local_ptr, assigned_nonlocal_ptr)
         end if

         ! Reset in_set_this_loop, which keeps track of which nodes are added to the set this loop
         do ifree = 1, local_rows
            ! If they're already assigned they can't be added
            if (assigned_local(ifree)) then
               in_set_this_loop(ifree) = .FALSE.
            ! We assume any unassigned are added to the set this loop and then rule them out below
            else
               in_set_this_loop(ifree) = .TRUE.
            end if
         end do

         ! ~~~~~~~~
         ! The Luby algorithm has measure_local(v) > measure_local(u) for all u in active neighbours
         ! and then you have to loop from the nodes with biggest measure_local down
         ! That is the definition of PMIS
         ! PMISR swaps the CF definitions from a traditional PMIS
         ! PMISR starts from the smallest measure_local and ensure 
         ! measure_local(v) < measure_local(u) for all u in active neighbours
         ! measure_local is negative for PMIS and positive for PMISR
         ! that way we dont have to change the .ge. in the comparison code below
         ! ~~~~~~~~

         ! ~~~~~~~~
         ! Go and do the local component
         ! ~~~~~~~~
         node_loop_local: do ifree = 1, local_rows

            ! Check if this node is already in A
            if (assigned_local(ifree)) cycle node_loop_local
            ! Loop over all the active strong neighbours on the local processors
            do jfree = ad_ia(ifree)+1, ad_ia(ifree+1)  

               ! Have to only check unassigned strong neighbours
               if (assigned_local(ad_ja(jfree) + 1)) cycle

               ! Check the measure_local
               if (measure_local(ifree) .ge. measure_local(ad_ja(jfree) + 1)) then
                  in_set_this_loop(ifree) = .FALSE.
                  cycle node_loop_local
               end if
            end do
         end do node_loop_local

         ! ~~~~~~~~
         ! Finish the async broadcast, assigned_nonlocal is now correct
         ! ~~~~~~~~
         if (comm_size /= 1) then
            call boolscatter_mat_end_c(A_array, assigned_local_ptr, assigned_nonlocal_ptr)
         end if     
                 
         ! ~~~~~~~~
         ! Now go through and do the non-local part of the matrix
         ! ~~~~~~~~            
         if (comm_size /= 1) then

            node_loop: do ifree = 1, local_rows   

               ! Check if already ruled out by local loop or already assigned
               if (assigned_local(ifree) .OR. .NOT. in_set_this_loop(ifree)) cycle node_loop       

               ! Loop over all the active strong neighbours on the non-local processors
               do jfree = ao_ia(ifree)+1, ao_ia(ifree+1)
                  
                  ! Have to only check unassigned strong neighbours
                  if (assigned_nonlocal(ao_ja(jfree) + 1)) cycle
   
                  ! Check the measure_local
                  if (measure_local(ifree) .ge. measure_nonlocal(ao_ja(jfree) + 1)) then
                     in_set_this_loop(ifree) = .FALSE.
                     cycle node_loop
                  end if
               end do

            end do node_loop
         end if

         ! We now know all nodes which were added to the set this loop, so let's record them
         do ifree = 1, local_rows
            if (in_set_this_loop(ifree)) then
               assigned_local(ifree) = .TRUE.
               cf_markers_local(ifree) = F_POINT
            end if
         end do

         ! ~~~~~~~~~~~~~~
         ! All the work below here is now to ensure assigned_local is correct for the next iteration
         ! Update the nonlocal values first then comm them
         ! ~~~~~~~~~~~~~~
         if (comm_size /= 1) then

            ! We're going to do an LOR reduce so start all as false
            assigned_nonlocal = .FALSE.
      
            do ifree = 1, local_rows   

               ! Only need to update neighbours of nodes assigned this top loop
               if (.NOT. in_set_this_loop(ifree)) cycle        

               ! We know all neighbours of points assigned this loop are C points
               ! We don't actually need to record that they're C points, just that they're assigned
               do jfree = ao_ia(ifree)+1, ao_ia(ifree+1)
                  assigned_nonlocal(ao_ja(jfree) + 1) = .TRUE.
               end do            
            end do 

            ! ~~~~~~~~~~~
            ! We need to start the async reduce LOR of the assigned_nonlocal into assigned_local     
            ! After this comms finishes any local node in another processors halo 
            ! that has been assigned on another process will be correctly marked in assigned_local
            ! ~~~~~~~~~~~    
            call boolscatter_mat_reverse_begin_c(A_array, assigned_local_ptr, assigned_nonlocal_ptr)       

         end if

         ! ~~~~~~~~~~~~~~
         ! Now go and update the local values
         ! ~~~~~~~~~~~~~~
        
         do ifree = 1, local_rows   

            ! Only need to update neighbours of nodes assigned this top loop
            if (.NOT. in_set_this_loop(ifree)) cycle

            ! Don't need a guard here to check if they're already assigned, as we 
            ! can guarantee they won't be 
            do jfree = ad_ia(ifree)+1, ad_ia(ifree+1)
               assigned_local(ad_ja(jfree) + 1) = .TRUE.
            end do            
         end do
      
         ! ~~~~~~~~~
         ! In parallel we have to finish our asyn comms
         ! ~~~~~~~~~
         if (comm_size /= 1) then
            ! Finishes the reduce LOR, assigned_local will now be correct
            call boolscatter_mat_reverse_end_c(A_array, assigned_local_ptr, assigned_nonlocal_ptr)       
         end if

         ! ~~~~~~~~~~~~
         ! We've now done another top level loop
         ! ~~~~~~~~~~~~
         loops_through = loops_through - 1

         ! ~~~~~~~~~~~~
         ! Check the total number of undecided in parallel before we loop again
         ! ~~~~~~~~~~~~
         if (max_luby_steps < 0) then
            ! Count how many are undecided
            counter_undecided =  local_rows - count(assigned_local)
            ! Parallel reduction!
            A_array = strength_mat%v
            call allreducesum_petscint_mine(A_array, counter_undecided, counter_parallel)
            counter_undecided = counter_parallel            
         end if
      end do

      ! Any unassigned become C points
      do ifree = 1, local_rows
         if (cf_markers_local(ifree) == 0) cf_markers_local(ifree) = C_POINT
      end do

      ! ~~~~~~~~~~~~
      ! We're finished our IS now
      ! ~~~~~~~~~~~~

      ! Restore the sequantial pointers once we're done
      call MatRestoreRowIJ(Ad,shift,symmetric,inodecompressed,n_ad,ad_ia,ad_ja,done,ierr) 
      if (comm_size /= 1) then
         call MatRestoreRowIJ(Ao,shift,symmetric,inodecompressed,n_ao,ao_ia,ao_ja,done,ierr) 
      end if    

      ! ~~~~~~~~~
      ! Cleanup
      ! ~~~~~~~~~      
      deallocate(in_set_this_loop, assigned_local)
      if (comm_size/=1) then
         call VecDestroy(measure_vec, ierr)    
         ! Don't forget to restore on lvec from our matrix
         call vecscatter_mat_restore_c(A_array, measure_nonlocal_ptr)             
      end if
      deallocate(assigned_nonlocal)    

   end subroutine pmisr_existing_measure_cf_markers  

   ! -------------------------------------------------------------------------------------------------------------------------------   

end module pmisr_module

