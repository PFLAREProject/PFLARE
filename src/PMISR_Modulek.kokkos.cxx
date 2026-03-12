// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

// The definition of the device copy of the cf markers on a given level
// is stored in Device_CF_Markersk.kokkos.cxx and imported as extern from
// kokkos_helper.hpp

//------------------------------------------------------------------------------------------------------------------------

// PMISR implementation that takes an existing measure and cf_markers on the device
// and then does the Luby algorithm to assign the rest of the CF markers
// This mirrors the CPU version pmisr_existing_measure_cf_markers in PMISR_Module.F90
PETSC_INTERN void pmisr_existing_measure_cf_markers_kokkos(Mat *strength_mat, const int max_luby_steps, const int pmis_int, PetscScalarKokkosView &measure_local_d, intKokkosView &cf_markers_d, const int zero_measure_c_point_int)
{

   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt rows_ao, cols_ao;
   MatType mat_type;

   PetscCallVoid(MatGetType(*strength_mat, &mat_type));
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat_MPIAIJ *mat_mpi = nullptr;
   Mat mat_local = NULL, mat_nonlocal = NULL;

   if (mpi)
   {
      mat_mpi = (Mat_MPIAIJ *)(*strength_mat)->data;
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*strength_mat, &mat_local, &mat_nonlocal, NULL));
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao)); 
   }
   else
   {
      mat_local = *strength_mat;
   }

   // Get the comm
   PetscCallVoid(PetscObjectGetComm((PetscObject)*strength_mat, &MPI_COMM_MATRIX));
   PetscCallVoid(MatGetLocalSize(*strength_mat, &local_rows, &local_cols));
   PetscCallVoid(MatGetSize(*strength_mat, &global_rows, &global_cols));
   // This returns the global index of the local portion of the matrix
   PetscCallVoid(MatGetOwnershipRange(*strength_mat, &global_row_start, &global_row_end_plus_one));

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr, *device_nonlocal_vals = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype));

   // PetscSF comms cannot be started with a pointer derived from a zero-extent Kokkos view -
   // doing so causes intermittent failures in parallel on GPUs. Use a size-1 dummy view
   // so that every pointer passed to PetscSF is always backed by valid device memory.
   intKokkosView sf_int_dummy_d("sf_int_dummy_d", 1);
   PetscScalarKokkosView sf_scalar_dummy_d("sf_scalar_dummy_d", 1);

   intKokkosView cf_markers_nonlocal_d;
   int *cf_markers_d_ptr = NULL, *cf_markers_nonlocal_d_ptr = NULL;
   cf_markers_d_ptr = local_rows > 0 ? cf_markers_d.data() : sf_int_dummy_d.data();

   intKokkosView cf_markers_send_d;
   int *cf_markers_send_d_ptr = NULL;

   PetscScalar *measure_local_d_ptr = NULL, *measure_nonlocal_d_ptr = NULL;
   measure_local_d_ptr = local_rows > 0 ? measure_local_d.data() : sf_scalar_dummy_d.data();
   PetscScalarKokkosView measure_nonlocal_d;

   if (mpi) {
      measure_nonlocal_d = PetscScalarKokkosView("measure_nonlocal_d", cols_ao);   
      measure_nonlocal_d_ptr = cols_ao > 0 ? measure_nonlocal_d.data() : sf_scalar_dummy_d.data();
      cf_markers_nonlocal_d = intKokkosView("cf_markers_nonlocal_d", cols_ao); 
      cf_markers_nonlocal_d_ptr = cols_ao > 0 ? cf_markers_nonlocal_d.data() : sf_int_dummy_d.data();
      cf_markers_send_d = intKokkosView("cf_markers_send_d", local_rows);
      cf_markers_send_d_ptr = local_rows > 0 ? cf_markers_send_d.data() : sf_int_dummy_d.data();       
   }

   // Device memory for the mark
   boolKokkosView mark_d("mark_d", local_rows);
   auto exec = PetscGetKokkosExecutionSpace();

   // Start the scatter of the measure - the kokkos memtype is set as PETSC_MEMTYPE_HOST or
   // one of the kokkos backends like PETSC_MEMTYPE_HIP
   PetscMemType mem_type = PETSC_MEMTYPE_KOKKOS;
   if (mpi)
   {
      // PetscSF owns measure_local_d_ptr as the active send buffer until End.
      // Do not even read from that send buffer before End is called.
      // If you alias it in overlapped GPU work, the failure shows up intermittently
      // in parallel runs on GPUs.
      PetscCallVoid(PetscSFBcastWithMemTypeBegin(mat_mpi->Mvctx, MPIU_SCALAR,
                                 mem_type, measure_local_d_ptr,
                                 mem_type, measure_nonlocal_d_ptr,
                                 MPI_REPLACE));      
   }

   // Initialise the set
   PetscInt counter_in_set_start = 0;
   // Count how many in the set to begin with and set their CF markers
   Kokkos::parallel_reduce ("Reduction", local_rows, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update) {
      // If already assigned by the input
      if (cf_markers_d(i) != 0)
      {
         update++;
      }
      else if (Kokkos::abs(measure_local_d[i]) < 1)
      {
         if (zero_measure_c_point_int == 1) {
            if (pmis_int == 1) {
               // Set as F here but reversed below to become C
               cf_markers_d(i) = -1;
            }
            else {
               // Becomes C
               cf_markers_d(i) = 1;
            }  
         }
         else {
            if (pmis_int == 1) {
               // Set as C here but reversed below to become F
               // Otherwise dirichlet conditions persist down onto the coarsest grid
               cf_markers_d(i) = 1;
            }
            else {
               // Becomes F
               cf_markers_d(i) = -1;
            }
         }         
         // Count
         update++;
      }
   }, counter_in_set_start);

   // Check the total number of undecided in parallel
   PetscInt counter_undecided, counter_parallel;
   if (max_luby_steps < 0) {
      counter_undecided = local_rows - counter_in_set_start;
      // Parallel reduction!
      PetscCallMPIAbort(MPI_COMM_MATRIX, MPI_Allreduce(&counter_undecided, &counter_parallel, 1, MPIU_INT, MPI_SUM, MPI_COMM_MATRIX));
      counter_undecided = counter_parallel;
      
   // If we're doing a fixed number of steps, then we don't care
   // how many undecided nodes we have - have to take care here not to use
   // local_rows for counter_undecided, as we may have zero DOFs on some procs
   // but we have to enter the loop below for the collective scatters 
   }
   else {
      counter_undecided = 1;
   }   

   // Finish the broadcast for the nonlocal measure
   if (mpi)
   {
      // End releases the active send buffer for normal access again.
      // The scattered values in measure_nonlocal_d are now safe to consume.
      PetscCallVoid(PetscSFBcastEnd(mat_mpi->Mvctx, MPIU_SCALAR, measure_local_d_ptr, measure_nonlocal_d_ptr, MPI_REPLACE));
   }   

   // ~~~~~~~~~~~~
   // Now go through the outer Luby loop
   // ~~~~~~~~~~~~      

   // Let's keep track of how many times we go through the loops
   int loops_through = -1;
   do 
   {
      // Match the fortran version and include a pre-test on the do-while
      if (counter_undecided == 0) break;

      // If max_luby_steps is positive, then we only take that many times through this top loop
      // We typically find 2-3 iterations decides >99% of the nodes 
      // and a fixed number of outer loops means we don't have to do any parallel reductions
      // We will do redundant nearest neighbour comms in the case we have already
      // finished deciding all the nodes, but who cares
      // Any undecided nodes just get turned into C points
      // We can do this as we know we won't ruin Aff by doing so, unlike in a normal multigrid
      if (max_luby_steps > 0 && max_luby_steps+1 == -loops_through) break;

      // ~~~~~~~~~
      // Start the async scatter of the nonlocal cf_markers
      // ~~~~~~~~~
      if (mpi) {
         // Copy cf_markers_d into a temporary buffer
         // If we gave the comms routine cf_markers_d we couldn't even read from
         // it until comms ended, meaning we couldn't do the work overlapping below
         Kokkos::deep_copy(cf_markers_send_d, cf_markers_d);
         // Be careful these aren't petscints
         // PetscSF owns cf_markers_send_d_ptr as the active send buffer until End.
         // Do not even read from that send buffer before End is called.
         // If you alias it in overlapped GPU work, the failure shows up intermittently
         // in parallel runs on GPUs.
         PetscCallVoid(PetscSFBcastWithMemTypeBegin(mat_mpi->Mvctx, MPI_INT,
                     mem_type, cf_markers_send_d_ptr,
                     mem_type, cf_markers_nonlocal_d_ptr,
                     MPI_REPLACE));
      }


      // mark_d keeps track of which of the candidate nodes can become in the set
      // Only need this because we want to do async comms so we need a way to trigger
      // a node not being in the set due to either strong local neighbours *or* strong offproc neighbours

      // ~~~~~~~~
      // Go and do the local component
      // ~~~~~~~~      
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            const PetscInt i = t.league_rank();
            PetscInt strong_neighbours = 0;

            // Check this row isn't already marked
            if (cf_markers_d(i) == 0)
            {
               const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];

               // Reduce over local columns to get the number of strong neighbours
               Kokkos::parallel_reduce(
                  Kokkos::TeamThreadRange(t, ncols_local),
                  [&](const PetscInt j, PetscInt& strong_count) {     

                  // Have to only check active strong neighbours
                  if (measure_local_d(i) >= measure_local_d(device_local_j[device_local_i[i] + j]) && \
                           cf_markers_d(device_local_j[device_local_i[i] + j]) == 0)
                  {
                     strong_count++;
                  }
               
               }, strong_neighbours
               );     

               // Only want one thread in the team to write the result
               Kokkos::single(Kokkos::PerTeam(t), [&]() {                  
                  // If we have any strong neighbours
                  if (strong_neighbours > 0) 
                  {
                     mark_d(i) = false;     
                  }
                  else
                  {
                     mark_d(i) = true;  
                  }
               });
            }
            // Any that aren't zero cf marker are already assigned so set to to false
            else
            {
               // Only want one thread in the team to write the result
               Kokkos::single(Kokkos::PerTeam(t), [&]() {                  
                  mark_d(i) = false;
               });
            }
      });

      // ~~~~~~~~
      // Now go through and do the non-local part of the matrix
      // ~~~~~~~~           
      if (mpi) {

         // Finish the async scatter
         // Be careful these aren't petscints
         // End releases the send snapshot for normal access again.
         // The scattered cf_markers_nonlocal_d values are now safe to read.
         PetscCallVoid(PetscSFBcastEnd(mat_mpi->Mvctx, MPI_INT, cf_markers_send_d_ptr, cf_markers_nonlocal_d_ptr, MPI_REPLACE));

         Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

               // Row
               const PetscInt i = t.league_rank();
               PetscInt strong_neighbours = 0;

               // Check this row isn't already marked
               if (mark_d(i))
               {
                  PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];

                  // Reduce over nonlocal columns to get the number of strong neighbours
                  Kokkos::parallel_reduce(
                     Kokkos::TeamThreadRange(t, ncols_nonlocal),
                     [&](const PetscInt j, PetscInt& strong_count) {     

                     if (measure_local_d(i) >= measure_nonlocal_d(device_nonlocal_j[device_nonlocal_i[i] + j])  && \
                              cf_markers_nonlocal_d(device_nonlocal_j[device_nonlocal_i[i] + j]) == 0)
                     {
                        strong_count++;
                     }
                  
                  }, strong_neighbours
                  );     

                  // Only want one thread in the team to write the result
                  Kokkos::single(Kokkos::PerTeam(t), [&]() {                  
                     // If we don't have any strong neighbours
                     if (strong_neighbours == 0) cf_markers_d(i) = loops_through;
                  });
               }
         });
      }
      // This cf_markers_d(i) = loops_through happens above in the case of mpi, saves a kernel launch
      else
      {
         // The nodes that have mark equal to true have no strong active neighbours in the IS
         // hence they can be in the IS
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt i) {

               if (mark_d(i)) cf_markers_d(i) = loops_through;
         });      
      }

      if (mpi) 
      {
         // We're going to do an add reverse scatter, so set them to zero
         Kokkos::deep_copy(cf_markers_nonlocal_d, 0);

         Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

               // Row
               const PetscInt i = t.league_rank();

               // Check if this node has been assigned during this top loop
               if (cf_markers_d(i) == loops_through)
               {
                  PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];

                  // For over nonlocal columns
                  Kokkos::parallel_for(
                     Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {

                        // Needs to be atomic as may being set by many threads
                        Kokkos::atomic_store(&cf_markers_nonlocal_d(device_nonlocal_j[device_nonlocal_i[i] + j]), 1);
                  });
               }
         });

         // Ensure everything is done before we comm
         exec.fence();

         // We've updated the values in cf_markers_nonlocal
         // Calling a reverse scatter add will then update the values of cf_markers_local
         // Reduce with a sum, equivalent to VecScatterBegin with ADD_VALUES, SCATTER_REVERSE
         // Be careful these aren't petscints
         // PetscSF now owns cf_markers_nonlocal_d_ptr as the active send buffer.
         // The local kernel below only touches cf_markers_d, and that is fine here
         // because we only care about zero versus nonzero after ReduceEnd.
         PetscCallVoid(PetscSFReduceWithMemTypeBegin(mat_mpi->Mvctx, MPI_INT,
            mem_type, cf_markers_nonlocal_d_ptr,
            mem_type, cf_markers_d_ptr,
            MPIU_SUM));
      }

      // Go and do local
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(exec, local_rows, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            const PetscInt i = t.league_rank();

            // Check if this node has been assigned during this top loop
            if (cf_markers_d(i) == loops_through)
            {
               const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];

               // For over nonlocal columns
               Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

                     // Needs to be atomic as may being set by many threads
                     // Tried a version where instead of a "push" approach I tried a pull approach
                     // that doesn't need an atomic, but it was slower
                     Kokkos::atomic_store(&cf_markers_d(device_local_j[device_local_i[i] + j]), 1);     
               });     
            }
      });   

      if (mpi)
      {
         // Finish the scatter
         // Be careful these aren't petscints
         // After End the accumulated cf_markers_d values are complete.
         // This is the first point where later logic should consume the reduced
         // result rather than the in-flight root buffer.
         PetscCallVoid(PetscSFReduceEnd(mat_mpi->Mvctx, MPI_INT, cf_markers_nonlocal_d_ptr, cf_markers_d_ptr, MPIU_SUM));
      }

      // We've done another top level loop
      loops_through = loops_through - 1;

      // ~~~~~~~~~~~~
      // Check the total number of undecided in parallel before we loop again
      // ~~~~~~~~~~~~
      if (max_luby_steps < 0) {

         counter_undecided = 0;  
         Kokkos::parallel_reduce ("ReductionCounter_undecided", local_rows, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update) {
            if (cf_markers_d(i) == 0) update++;
         }, counter_undecided); 

         // Parallel reduction!
         PetscCallMPIAbort(MPI_COMM_MATRIX, MPI_Allreduce(&counter_undecided, &counter_parallel, 1, MPIU_INT, MPI_SUM, MPI_COMM_MATRIX));
         counter_undecided = counter_parallel;
      } else {
         // If we're doing a fixed number of steps, then we need an extra fence
         // as we don't hit the parallel reduce above (which implicitly fences)
         exec.fence();
      }

   }
   while (counter_undecided != 0);

   // ~~~~~~~~~
   // Now assign our final cf markers
   // ~~~~~~~~~

   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
         
      if (cf_markers_d(i) == 0)
      {
         cf_markers_d(i) = 1;
      }
      else if (cf_markers_d(i) < 0)
      {
         cf_markers_d(i) = -1;
      }
      else
      {
         cf_markers_d(i) = 1;
      }
   });
   // Ensure we're done before we exit
   exec.fence();

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// PMISR cf splitting but on the device
// This no longer copies back to the host pointer cf_markers_local at the end
// You have to explicitly call copy_cf_markers_d2h(cf_markers_local) to do this
PETSC_INTERN void pmisr_kokkos(Mat *strength_mat, const int max_luby_steps, const int pmis_int, PetscReal *measure_local, const int zero_measure_c_point_int)
{

   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt rows_ao, cols_ao;
   MatType mat_type;

   PetscCallVoid(MatGetType(*strength_mat, &mat_type));
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat mat_local = NULL, mat_nonlocal = NULL;

   if (mpi)
   {
      PetscCallVoid(MatMPIAIJGetSeqAIJ(*strength_mat, &mat_local, &mat_nonlocal, NULL));
      PetscCallVoid(MatGetSize(mat_nonlocal, &rows_ao, &cols_ao));
   }
   else
   {
      mat_local = *strength_mat;
   }

   // Get the comm
   PetscCallVoid(PetscObjectGetComm((PetscObject)*strength_mat, &MPI_COMM_MATRIX));
   PetscCallVoid(MatGetLocalSize(*strength_mat, &local_rows, &local_cols));
   PetscCallVoid(MatGetSize(*strength_mat, &global_rows, &global_cols));

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr, *device_nonlocal_vals = nullptr;
   PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype));
   if (mpi) PetscCallVoid(MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype));

   // Device memory for the global variable cf_markers_local_d - be careful these aren't petsc ints
   cf_markers_local_d = intKokkosView("cf_markers_local_d", local_rows);
   // Can't use the global directly within the parallel
   // regions on the device so just take a shallow copy
   intKokkosView cf_markers_d = cf_markers_local_d;

   // Host and device memory for the measure
   PetscScalarKokkosViewHost measure_local_h(measure_local, local_rows);
   PetscScalarKokkosView measure_local_d("measure_local_d", local_rows);

   auto exec = PetscGetKokkosExecutionSpace();

   // If you want to generate the randoms on the device
   //Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
   // Copy the input measure from host to device
   Kokkos::deep_copy(measure_local_d, measure_local_h);
   // Log copy with petsc
   size_t bytes = measure_local_h.extent(0) * sizeof(PetscReal);
   PetscCallVoid(PetscLogCpuToGpu(bytes));

   // Compute the measure
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt i) {

      // Randoms on the device
      // auto generator = random_pool.get_state();
      // measure_local_d(i) = generator.drand(0., 1.);
      // random_pool.free_state(generator);

      const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];
      measure_local_d(i) += ncols_local;

      if (mpi)
      {
         PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];
         measure_local_d(i) += ncols_nonlocal;
      }
      // Flip the sign if pmis
      if (pmis_int == 1) measure_local_d(i) *= -1;
   });
   // Have to ensure the parallel for above finishes before comms
   exec.fence();

   // Call the existing measure cf markers function
   pmisr_existing_measure_cf_markers_kokkos(strength_mat, max_luby_steps, pmis_int, measure_local_d, cf_markers_d, zero_measure_c_point_int);

   // If PMIS then we swap the CF markers from PMISR
   if (pmis_int) {
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
            cf_markers_d(i) *= -1;
      });
      // Ensure we're done before we exit
      exec.fence();
   }

   return;
}

//------------------------------------------------------------------------------------------------------------------------
