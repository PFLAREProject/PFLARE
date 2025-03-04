module pmisr_ddc

   use iso_c_binding
   use petsc
   use petsc_helper
   use c_petsc_interfaces

#include "petsc/finclude/petsc.h"

   implicit none

#include "petsc_legacy.h"

   ! Define C and F points in the CF marker 
   integer, parameter :: C_POINT = 1
   integer, parameter :: F_POINT = -1

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
      MPI_Comm :: MPI_COMM_MATRIX    
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

         call pmisr_kokkos(A_array, max_luby_steps, pmis_int, measure_local_ptr, cf_markers_local_ptr, zero_measure_c_point_int)

         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then         
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
      PetscInt :: jfree
      PetscInt :: rows_ao, cols_ao, n_ad, n_ao
      integer :: comm_size, comm_size_world, loops_through, seed_size
      integer :: comm_rank, counter_parallel, errorcode
#ifdef _OPENMP
      integer :: omp_threads
#endif        
      integer :: counter_undecided, counter_in_set_start, kfree
      PetscErrorCode :: ierr
      MPI_Comm :: MPI_COMM_MATRIX      
      integer, dimension(:), allocatable :: seed
      PetscReal, dimension(:), allocatable :: measure_local, measure_nonlocal
      PetscReal, dimension(:), allocatable :: cf_markers_local_real
      logical, dimension(:), allocatable :: mark
      type(c_ptr) :: cf_markers_nonlocal_ptr
      real(c_double), pointer :: cf_markers_nonlocal(:) => null()
      type(tMat) :: Ad, Ao
      type(tVec) :: cf_markers_vec
      PetscOffset :: iicol
      PetscInt :: icol(1)
      integer(c_long_long) :: A_array, vec_long
      PetscInt, dimension(:), pointer :: ad_ia, ad_ja, cols_ptr, ao_ia, ao_ja
      PetscInt :: shift = 0
      logical :: symmetric = .false., inodecompressed=.false., done    
      logical :: zero_measure_c = .FALSE.  
      PetscInt, parameter :: nz_ignore = -1, one=1, zero=0

      ! ~~~~~~           

      if (present(zero_measure_c_point)) zero_measure_c = zero_measure_c_point

      ! Get the comm size 
      call PetscObjectGetComm(strength_mat, MPI_COMM_MATRIX, ierr)    
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)
      call MPI_Comm_size(MPI_COMM_WORLD, comm_size_world, errorcode)      
      ! Get the comm rank 
      call MPI_Comm_rank(MPI_COMM_MATRIX, comm_rank, errorcode)      

      ! Get the local sizes
      call MatGetLocalSize(strength_mat, local_rows, local_cols, ierr)
      call MatGetSize(strength_mat, global_rows, global_cols, ierr)      
      call MatGetOwnershipRange(strength_mat, global_row_start, global_row_end_plus_one, ierr)   

      if (comm_size /= 1) then
         ! Let's get the diagonal and off-diagonal parts of the strength matrix
         call MatMPIAIJGetSeqAIJ(strength_mat, Ad, Ao, icol, iicol, ierr)
         ! We know the col size of Ao is the size of colmap, the number of non-zero offprocessor columns
         call MatGetSize(Ao, rows_ao, cols_ao, ierr)    
      else
         Ad = strength_mat    
      end if

      ! ~~~~~~~~
      ! Get pointers to the sequential diagonal and off diagonal aij structures 
      ! so we don't have to put critical regions around the matgetrow
      ! ~~~~~~~~
      call MatGetRowIJF90(Ad,shift,symmetric,inodecompressed,n_ad,ad_ia,ad_ja,done,ierr) 
      if (.NOT. done) then
         print *, "Pointers not set in call to MatGetRowIJF90"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      if (comm_size /= 1) then
         call MatGetRowIJF90(Ao,shift,symmetric,inodecompressed,n_ao,ao_ia,ao_ja,done,ierr) 
         if (.NOT. done) then
            print *, "Pointers not set in call to MatGetRowIJF90"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
         end if
      end if      
      ! ~~~~~~~~~~      

      ! Get the number of connections in S
      allocate(measure_local(local_rows))
      allocate(cf_markers_local(local_rows))  
      allocate(cf_markers_local_real(local_rows))
      allocate(mark(local_rows))

      ! ~~~~~~~~~
      ! We're using reals here to represent the cf marker just because I want to use the existing petsc
      ! scatter for strength_mat - shouldn't be a big difference in comms 
      ! ~~~~~~~~~

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
      ! so I'm fairly confident things are correct - we don't care if the CF splitting
      ! is identical for different numbers of procs so we haven't kept that (as its expensive)

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

      ! If PMIS then we want to search the measure based on the largest entry
      ! PMISR searches the measure based on the smallest entry
      ! We just let the measure be negative rather than change the .ge. comparison 
      ! in our Luby below
      if (pmis) measure_local = measure_local * (-1)
      
      ! ~~~~~~~~~~~~
      ! Create parallel cf_marker vec and scatter the measure
      ! ~~~~~~~~~~~~
      ! If in parallel we're going to have to do scatters
      if (comm_size/=1) then

         ! This is fine being mpi type specifically as strength_mat is always a mataij
         call VecCreateMPIWithArray(MPI_COMM_MATRIX, one, &
            local_rows, global_rows, cf_markers_local_real, cf_markers_vec, ierr)            

         ! Let's scatter the measure now
         cf_markers_local_real = measure_local

         A_array = strength_mat%v
         vec_long = cf_markers_vec%v
         ! Have to call restore after we're done with lvec (ie cf_markers_nonlocal_ptr)
         call vecscatter_mat_begin_c(A_array, vec_long, cf_markers_nonlocal_ptr)
         call vecscatter_mat_end_c(A_array, vec_long, cf_markers_nonlocal_ptr)
         ! Nonlocal vals only pointer
         call c_f_pointer(cf_markers_nonlocal_ptr, cf_markers_nonlocal, shape=[cols_ao])

         allocate(measure_nonlocal(cols_ao))
         measure_nonlocal = cf_markers_nonlocal
         ! Don't forget to restore
         call vecscatter_mat_restore_c(A_array, cf_markers_nonlocal_ptr)

      end if

      ! ~~~~~~~~~~~~
      ! Initialise the set
      ! ~~~~~~~~~~~~
      counter_in_set_start = 0

      do ifree = 1, local_rows

         ! If there are no strong neighbours (not measure_local == 0 as we have added a random number to it)
         ! then we treat it special
         ! Absolute value here given measure_local could be negative (pmis) or positive (pmisr)
         if (abs(measure_local(ifree)) < 1) then

            ! This is typically enabled in a second pass of PMIS just on C points 
            ! (ie aggressive coarsening based on MIS(MIS(1))), we want to keep  
            ! C-points with no other strong C connections as C points
            if (zero_measure_c) then
               if (pmis) then
                  ! Set as F here but reversed below to become C
                  cf_markers_local_real(ifree) = F_POINT
               else
                  ! Becomes C
                  cf_markers_local_real(ifree) = C_POINT
               end if  
            else
               if (pmis) then
                  ! Set as C here but reversed below to become F
                  ! Otherwise dirichlet conditions persist down onto the coarsest grid
                  cf_markers_local_real(ifree) = C_POINT
               else
                  ! Becomes F
                  cf_markers_local_real(ifree) = F_POINT
               end if
            end if
            counter_in_set_start = counter_in_set_start + 1
         else
            cf_markers_local_real(ifree) = 0
         end if
      end do       

      ! Check the total number of undecided in parallel
      if (max_luby_steps < 0) then
         ! Assuming here we don't have more than 2B local rows
         counter_undecided = int(local_rows) - counter_in_set_start
         ! Parallel reduction!
         call MPI_Allreduce(counter_undecided, counter_parallel, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_MATRIX, errorcode)
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

         !if (getprocno() == 1) print *, "TOP LOOP", local_rows, counter_undecided

         ! ~~~~~~~~~
         ! Start the async scatter of the nonlocal cf_markers
         ! ~~~~~~~~~
         if (comm_size /= 1) then
            ! Have to call restore after we're done with lvec (ie cf_markers_nonlocal_ptr)
            call vecscatter_mat_begin_c(A_array, vec_long, cf_markers_nonlocal_ptr)
         end if

         ! ~~~~~~~~
         ! This keeps track of which of the candidate nodes can become in the set
         ! Only need this because we want to do async comms so we need a way to trigger
         ! a node not being in the set due to either strong local neighbours *or* strong offproc neighbours
         mark = .TRUE.

         ! Any that aren't zero cf marker are already assigned so set to to false
         do ifree = 1, local_rows
            if (cf_markers_local_real(ifree) /= 0) mark(ifree) = .FALSE.
         end do  

         ! ~~~~~~~~~~~~
         ! Loop over all the undecided rows
         ! Now this occurs based on the strong neighbours in the active set
         ! ie cf_markers == 0 is A
         ! cf_markers == loops_through is M' 
         ! ~~~~~~~~~~~~

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

            ! Check if this node is in A
            ! This can't be mark as we don't want it to be updated during the loop
            if (cf_markers_local_real(ifree) /= 0) cycle node_loop_local        

            ! Get S_i
            ! This is the number of columns
            ncols = ad_ia(ifree+1) - ad_ia(ifree)      
            ! This is the column indices
            cols_ptr => ad_ja(ad_ia(ifree)+1:ad_ia(ifree+1))

            ! Loop over all the active strong neighbours on the local processors
            do jfree = 1, ncols
               
               ! Have to only check active strong neighbours
               ! This can't be mark as we don't want it to be updated during the loop
               if (cf_markers_local_real(cols_ptr(jfree) + 1) /= 0) cycle

               ! Check the measure_local
               if (measure_local(ifree) .ge. measure_local(cols_ptr(jfree) + 1)) then
                  mark(ifree) = .FALSE.
                  cycle node_loop_local
               end if
            end do
         end do node_loop_local

         ! ~~~~~~~~
         ! Finish the async scatter
         ! ~~~~~~~~
         if (comm_size /= 1) then
            call vecscatter_mat_end_c(A_array, vec_long, cf_markers_nonlocal_ptr)
            ! Nonlocal vals only pointer
            call c_f_pointer(cf_markers_nonlocal_ptr, cf_markers_nonlocal, shape=[cols_ao])
         end if     
                 
         ! ~~~~~~~~
         ! Now go through and do the non-local part of the matrix
         ! ~~~~~~~~            
         if (comm_size /= 1) then

            node_loop: do ifree = 1, local_rows   

               ! Check if this node is in A
               ! This can't be mark as we don't want it to be updated during the loop
               if (cf_markers_local_real(ifree) /= 0) cycle node_loop        

               ! Get S_i
               ! This is the number of columns
               ncols = ao_ia(ifree+1) - ao_ia(ifree)      
               ! This is the column indices
               cols_ptr => ao_ja(ao_ia(ifree)+1:ao_ia(ifree+1))

               ! Loop over all the active strong neighbours on the local processors
               do jfree = 1, ncols
                  
                  ! Have to only check active strong neighbours
                  ! This can't be mark as we don't want it to be updated during the loop
                  if (cf_markers_nonlocal(cols_ptr(jfree) + 1) /= 0) cycle
   
                  ! Check the measure_local
                  if (measure_local(ifree) .ge. measure_nonlocal(cols_ptr(jfree) + 1)) then
                     mark(ifree) = .FALSE.
                     cycle node_loop
                  end if
               end do

            end do node_loop
         end if

         ! The nodes that have mark equal to true have no strong active neighbours in the IS
         ! hence they can be in the IS
         do ifree = 1, local_rows
            if (mark(ifree)) then
               cf_markers_local_real(ifree) = dble(loops_through)
            end if
         end do

         ! ~~~~~~~~~~~~
         ! Once we're here
         ! cf_markers_local <0 is M
         ! And now we have to go and update cf_markers to exclude any neighbours from A
         ! ready for cf_markers == 0 to be A in the next loop
         ! ~~~~~~~~~~~~       

         ! ~~~~~~~~~~~~~~
         ! Update the nonlocal values first then start the async scatter
         ! The only change to nonlocal nodes is that they could have been assigned to 
         ! not be in the IS - we only assign local nodes to be in the IS
         ! The other processor is guaranteed to not have picked that node in the IS
         ! We know the other processor will eventually set that node to be not in the IS
         ! but it might be doing that sometime much later. It has to be deleted in this step
         ! otherwise it could be considered an active vertex on the other processor
         ! ~~~~~~~~~~~~~~
         if (comm_size /= 1) then

            ! We're going to do an add reverse scatter, so set them to zero
            cf_markers_nonlocal = 0
      
            do ifree = 1, local_rows   

               ! Check if this node has been assigned during this top loop
               if (cf_markers_local_real(ifree) /= loops_through) cycle        

               ! Get S_i
               ! This is the number of columns
               ncols = ao_ia(ifree+1) - ao_ia(ifree)      
               ! This is the column indices
               cols_ptr => ao_ja(ao_ia(ifree)+1:ao_ia(ifree+1))

               do jfree = 1, ncols
                  cf_markers_nonlocal(cols_ptr(jfree) + 1) = C_POINT
               end do            
            end do 

            ! ~~~~~~~~~~~
            ! Now we have to reverse add scatter cf_markers_nonlocal back into cf_markers
            ! If a node on the other processor assigned any of our halo nodes to not in the IS and we didn't
            ! the add reverse will make it 1, and vice versa. If both (or more) of them set it to not in the IS
            ! then the add reverse will make it >1, which is fine because the only checks we do are nonzero checks            
            ! ~~~~~~~~~~~
            
            ! We've updated the values in cf_markers_nonlocal, which is a pointer to lvec
            ! Calling a reverse scatter add will then update the values of vec_long (ie cf_markers_vec)
            ! Begin the scatter asynchronously
            call vecscatter_mat_reverse_begin_c(A_array, vec_long)            

         end if

         ! ~~~~~~~~~~~~~~
         ! Now go and update the local values
         ! ~~~~~~~~~~~~~~
        
         do ifree = 1, local_rows   

            ! Check if this node has been assigned during this top loop
            if (cf_markers_local_real(ifree) /= loops_through) cycle

            ! Get S_i
            ! This is the number of columns
            ncols = ad_ia(ifree+1) - ad_ia(ifree)      
            ! This is the column indices
            cols_ptr => ad_ja(ad_ia(ifree)+1:ad_ia(ifree+1))

            ! Set unassigned strong dependencies as not in the IS
            ! Don't need a guard here to check if they're already assigned, as we 
            ! can guarantee they won't be 
            do jfree = 1, ncols
               cf_markers_local_real(cols_ptr(jfree) + 1) = C_POINT
            end do            
         end do
      
         ! ~~~~~~~~~
         ! In parallel we have to finish our asyn scatter
         ! ~~~~~~~~~
         if (comm_size /= 1) then

            ! Don't forget to finish the scatter
            call vecscatter_mat_reverse_end_c(A_array, vec_long)

            ! We're now done with our markers
            ! and have to call restore on the lvec we've pulled from strength_mat
            call vecscatter_mat_restore_c(A_array, cf_markers_nonlocal_ptr)
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
            counter_undecided = count(cf_markers_local_real == 0)
            ! Parallel reduction!
            call MPI_Allreduce(counter_undecided, counter_parallel, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_MATRIX, errorcode)
            counter_undecided = counter_parallel            
         end if
      end do

      ! ~~~~~~~~~~~~
      ! We're finished our IS now
      ! ~~~~~~~~~~~~

      ! Restore the sequantial pointers once we're done
      call MatRestoreRowIJF90(Ad,shift,symmetric,inodecompressed,n_ad,ad_ia,ad_ja,done,ierr) 
      if (comm_size /= 1) then
         call MatRestoreRowIJF90(Ao,shift,symmetric,inodecompressed,n_ao,ao_ia,ao_ja,done,ierr) 
      end if    
      
      ! ~~~~~~~~~
      ! Now assign our final cf markers
      ! ~~~~~~~~~

      ! We set cf_markers_local to be -1, -2, -3... if its in the set, based on which 
      ! outer iteration it was set in, just reset them all to -1
      ! And for not in the set, they are 1 or will have a value greater than one 
      ! if an add reverse scatter from other processors touched them (ie told them they were 
      ! not in the set from another processor)
      ! If we're 0, then we broke out of our CF loop early as we had only a small fraction of 
      ! undecided left and we can easily just call them as not in the set 
      
      do ifree = 1, local_rows
         ! Because we have reals here, make sure to stick the zero comparison first
         ! as not sure which sign the fp zero matches
         ! If unassigned its C
         if (cf_markers_local_real(ifree) == 0) then
            cf_markers_local(ifree) = C_POINT
         ! Less than zero its F
         else if (cf_markers_local_real(ifree) < 0) then
            cf_markers_local(ifree) = F_POINT
         ! Greater than zero its C
         else 
            cf_markers_local(ifree) = C_POINT
         end if
      end do
      
      ! If PMIS then we swap the CF markers from PMISR
      if (pmis) then
         cf_markers_local = cf_markers_local * (-1)
      end if

      ! ~~~~~~~~~
      ! Cleanup
      ! ~~~~~~~~~      
      deallocate(measure_local, cf_markers_local_real, mark)
      if (comm_size/=1) then
         call VecDestroy(cf_markers_vec, ierr)    
         deallocate(measure_nonlocal)        
      end if

   end subroutine pmisr_cpu  

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine ddc(input_mat, is_fine, fraction_swap, cf_markers_local)

      ! Second pass diagonal dominance cleanup 
      ! Flips the F definitions to C based on least diagonally dominant local rows
      ! If fraction_swap = 0 this does nothing
      ! If fraction_swap < 0 it uses abs(fraction_swap) to be a threshold 
      !  for swapping C to F based on row-wise diagonal dominance (ie alpha_diag)
      ! If fraction_swap > 0 it uses fraction_swap as the local fraction of worst C points to swap to F
      !  though it won't hit that fraction exactly as we bin the diag dom ratios for speed, it will be close to the fraction

      ! ~~~~~~
      type(tMat), target, intent(in)      :: input_mat
      type(tIS), intent(in)               :: is_fine
      PetscReal, intent(in)               :: fraction_swap
      integer, dimension(:), allocatable, target, intent(inout) :: cf_markers_local

#if defined(PETSC_HAVE_KOKKOS)                     
      integer(c_long_long) :: A_array, indices
      PetscErrorCode :: ierr
      MatType :: mat_type
      type(c_ptr)  :: cf_markers_local_ptr
      integer :: errorcode
      !integer :: kfree
      integer, dimension(:), allocatable :: cf_markers_local_two
#endif 
      ! ~~~~~~  

      ! If we don't need to swap anything, return
      if (fraction_swap == 0d0) then
         return
      end if      

#if defined(PETSC_HAVE_KOKKOS)    

      call MatGetType(input_mat, mat_type, ierr)
      if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) then  

         A_array = input_mat%v  
         indices = is_fine%v
         cf_markers_local_ptr = c_loc(cf_markers_local)

         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then         
            allocate(cf_markers_local_two(size(cf_markers_local)))
            cf_markers_local_two = cf_markers_local
         end if

         call ddc_kokkos(A_array, indices, fraction_swap, cf_markers_local_ptr)

         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then  
            call ddc_cpu(input_mat, is_fine, fraction_swap, cf_markers_local_two)  

            if (any(cf_markers_local /= cf_markers_local_two)) then

               ! do kfree = 1, size(cf_markers_local)
               !    if (cf_markers_local(kfree) /= cf_markers_local_two(kfree)) then
               !       print *, kfree-1, "no match", cf_markers_local(kfree), cf_markers_local_two(kfree)
               !    end if
               ! end do
               print *, "Kokkos and CPU versions of ddc do not match"
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode) 
            end if
            deallocate(cf_markers_local_two)
         end if

      else
         call ddc_cpu(input_mat, is_fine, fraction_swap, cf_markers_local)     
      end if
#else
      call ddc_cpu(input_mat, is_fine, fraction_swap, cf_markers_local)
#endif        
      
   end subroutine ddc
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine ddc_cpu(input_mat, is_fine, fraction_swap, cf_markers_local)

      ! Second pass diagonal dominance cleanup 
      ! Flips the F definitions to C based on least diagonally dominant local rows
      ! If fraction_swap = 0 this does nothing
      ! If fraction_swap < 0 it uses abs(fraction_swap) to be a threshold 
      !  for swapping C to F based on row-wise diagonal dominance (ie alpha_diag)
      ! If fraction_swap > 0 it uses fraction_swap as the local fraction of worst C points to swap to F
      !  though it won't hit that fraction exactly as we bin the diag dom ratios for speed, it will be close to the fraction

      ! ~~~~~~
      type(tMat), target, intent(in)      :: input_mat
      type(tIS), intent(in)               :: is_fine
      PetscReal, intent(in)               :: fraction_swap
      integer, dimension(:), allocatable, intent(inout) :: cf_markers_local

      ! Local
      PetscInt :: local_rows, local_cols
      PetscInt :: a_global_row_start, a_global_row_end_plus_one, ifree, ncols
      PetscInt :: input_row_start, input_row_end_plus_one
      PetscInt :: max_nnzs, jfree, idx, search_size, diag_index
      integer :: bin_sum, bin_boundary, bin
      PetscErrorCode :: ierr
      PetscInt, dimension(:), allocatable :: cols
      PetscReal, dimension(:), allocatable :: vals, diag_dom_ratio, diag_dom_ratio_small
      PetscInt, dimension(:), pointer :: is_pointer
      type(tMat) :: Aff
      PetscReal :: diag_val
      real(c_double) :: swap_dom_val
      integer, dimension(1000) :: dom_bins

      ! ~~~~~~  

      ! The indices are the numbering in Aff matrix
      call ISGetIndicesF90(is_fine, is_pointer, ierr)   
      
      ! Do a fixed alpha_diag
      if (fraction_swap < 0) then
         ! We have to look through all the local rows
         search_size = size(is_pointer)

      ! Or pick alpha_diag based on the worst % of rows
      else
         ! Only need to go through the biggest % of indices
         search_size = int(dble(size(is_pointer)) * fraction_swap)          
      end if
      
      ! ~~~~~~~~~~~~~
 
      ! Pull out Aff for ease of use
      call MatCreateSubMatrix(input_mat, &
            is_fine, is_fine, MAT_INITIAL_MATRIX, &
            Aff, ierr)

      ! Can't put this above because of collective operations in parallel (namely the getsubmatrix)
      ! If we have local points to swap
      if (search_size > 0) then            

         ! Get the local sizes
         call MatGetLocalSize(Aff, local_rows, local_cols, ierr)
         call MatGetOwnershipRange(Aff, a_global_row_start, a_global_row_end_plus_one, ierr)
         call MatGetOwnershipRange(input_mat, input_row_start, input_row_end_plus_one, ierr)                              

         max_nnzs = 0
         do ifree = a_global_row_start, a_global_row_end_plus_one-1                  
            call MatGetRow(Aff, ifree, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
            if (ncols > max_nnzs) max_nnzs = ncols
            call MatRestoreRow(Aff, ifree, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
         end do         
         
         allocate(cols(max_nnzs))
         allocate(vals(max_nnzs))     
         allocate(diag_dom_ratio(local_rows))   
         diag_dom_ratio = 0
         dom_bins = 0
         
         ! Sum the rows and find the diagonal entry in each local row
         do ifree = a_global_row_start, a_global_row_end_plus_one-1                  
            call MatGetRow(Aff, ifree, ncols, cols, vals, ierr)

            ! Index of the diagonal
            diag_index = -1
            diag_val = 1.0d0

            do jfree = 1, ncols
               ! Store the diagonal
               if (cols(jfree) == ifree) then
                  diag_val = abs(vals(jfree))
                  diag_index = jfree
               else
                  ! Row sum of off-diagonals
                  diag_dom_ratio(ifree - a_global_row_start + 1) = diag_dom_ratio(ifree - a_global_row_start + 1) + abs(vals(jfree))
               end if
            end do

            ! If we don't have a diagonal entry in this row there is no point trying to 
            ! compute a diagonal dominance ratio
            ! We set diag_dom_ratio to zero and that means this row will stay as an F point
            if (diag_index == -1) then
               diag_dom_ratio(ifree - a_global_row_start + 1) = 0.0
               call MatRestoreRow(Aff, ifree, ncols, cols, vals, ierr)    
               cycle
            end if 

            ! If we have non-diagonal entries
            if (diag_dom_ratio(ifree - a_global_row_start + 1) /= 0d0) then
               ! Compute the diagonal dominance ratio
               diag_dom_ratio(ifree - a_global_row_start + 1) = diag_dom_ratio(ifree - a_global_row_start + 1) / diag_val
            end if

            ! Bin the entries between 0 and 1
            ! The top bin has entries greater than 0.9 (including greater than 1)
            bin = min(floor(diag_dom_ratio(ifree - a_global_row_start + 1) * size(dom_bins)) + 1, size(dom_bins))
            ! If the diagonal dominance ratio is really large the expression above will overflow
            ! the int to negative, so we just stick that in the top bin            
            if (bin < 0) then
               bin = size(dom_bins)
            end if
            dom_bins(bin) = dom_bins(bin) + 1

            call MatRestoreRow(Aff, ifree, ncols, cols, vals, ierr)                                 
         end do   
      
         ! Do a fixed alpha_diag
         if (fraction_swap< 0) then    
            swap_dom_val = -fraction_swap

         ! Otherwise swap everything bigger than a fixed fraction
         else

            ! In order to reduce the size of the sort required, we have binned the entries into 1000 bins
            ! Let's count backwards from the biggest entries to find which bin we know the nth_element is in
            ! and then we only include those bins and higher into the sort
            bin_sum = 0
            do bin_boundary = size(dom_bins), 1, -1
               bin_sum = bin_sum + dom_bins(bin_boundary)
               if (bin_sum .ge. search_size) exit
            end do
            ! Now bin_boundary holds the bin whose lower boundary is guaranteed to be <= the n_th element

            ! Rather than do any type of sort, just swap everything above that bin boundary
            ! This will give a fraction_swap that is very close to that passed in as long as the 
            ! size of the bins is small
            swap_dom_val = dble(bin_boundary-1)/dble(size(dom_bins))

         end if

         ! Let's go and swap F points to C points
         do ifree = 1, local_rows

            ! If this row only has a single diagonal entry, or is below the threshold we swap, skip
            if (diag_dom_ratio(ifree) == 0 .OR. diag_dom_ratio(ifree) < swap_dom_val) cycle

            ! This is the actual numbering in A, rather than Aff
            ! Careful here to minus away the row_start of A, not Aff, as cf_markers_local is as big as A
            idx = is_pointer(ifree) - input_row_start + 1

            ! Swap by multiplying by -1
            cf_markers_local(idx) = cf_markers_local(idx) * (-1)
         end do
         deallocate(cols, vals, diag_dom_ratio)
         if (allocated(diag_dom_ratio_small)) deallocate(diag_dom_ratio_small)
      end if

      call ISRestoreIndicesF90(is_fine, is_pointer, ierr)
      call MatDestroy(Aff, ierr)     

   end subroutine ddc_cpu      
   
! -------------------------------------------------------------------------------------------------------------------------------

end module pmisr_ddc

