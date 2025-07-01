// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

// This is a device copy of the cf markers on a given level
// to save having to copy it to/from the host between pmisr and ddc calls
intKokkosView cf_markers_local_d;

//------------------------------------------------------------------------------------------------------------------------

// Copy the global cf_markers_local_d back to the host
PETSC_INTERN void copy_cf_markers_d2h(int *cf_markers_local)
{
   // Host wrapper for cf_markers_local
   intKokkosViewHost cf_markers_local_h(cf_markers_local, cf_markers_local_d.extent(0));

   // Now copy device cf_markers_local_d back to host
   Kokkos::deep_copy(cf_markers_local_h, cf_markers_local_d);
   // Log copy with petsc
   size_t bytes = cf_markers_local_d.extent(0) * sizeof(int);
   PetscLogGpuToCpu(bytes);

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Delete the global cf_markers_local_d
PETSC_INTERN void delete_device_cf_markers()
{
   // Delete the device view - this assigns an empty view
   // and hence the old view has its ref counter decremented
   cf_markers_local_d = intKokkosView(); 

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// PMISR cf splitting but on the device
// This no longer copies back to the host pointer cf_markers_local at the end
// You have to explicitly call copy_cf_markers_d2h(cf_markers_local) to do this
// The strength matrix you pass in should just be S, not S + S^T, this routine now does the transpose
// implicitly
PETSC_INTERN void pmisr_kokkos(Mat *strength_mat, const int max_luby_steps, const int pmis_int, PetscReal *measure_local, const int zero_measure_c_point_int)
{

   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one;
   PetscInt rows_ao, cols_ao;
   MatType mat_type;

   MatGetType(*strength_mat, &mat_type);
   // Are we in parallel?
   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   Mat_MPIAIJ *mat_mpi = nullptr;
   Mat mat_local = NULL, mat_nonlocal = NULL, mat_local_transpose = NULL, mat_nonlocal_transpose = NULL;   

   // ~~~~~~~~~~~~~~~~~~~~~
   // PMISR needs to work with S+S^T to keep out large entries from Aff
   // but we never want to form S+S^T explicitly as it is expensive
   // So instead we do several comms steps in our Luby loop to get/send the data we need
   // We do compute local copies of the transpose of S (which happen on the device)
   // but we never have the full parallel S+S^T 
   // On this rank we have the number of:
   // local strong dependencies (from the local S) 
   // local strong influences (from the local S^T)
   // non-local strong dependencies (from the non-local part of S)
   // But we don't have the number of non-local strong influences (from the non-local part of S^T)   
   // ~~~~~~~~~~~~~~~~~~~~~

   if (mpi)
   {
      mat_mpi = (Mat_MPIAIJ *)(*strength_mat)->data;
      mat_local = mat_mpi->A;
      mat_nonlocal = mat_mpi->B;
      MatGetSize(mat_nonlocal, &rows_ao, &cols_ao); 
      MatTranspose(mat_nonlocal, MAT_INITIAL_MATRIX, &mat_nonlocal_transpose);     
   }
   else
   {
      mat_local = *strength_mat;
   }   

   // Get the comm
   PetscObjectGetComm((PetscObject)*strength_mat, &MPI_COMM_MATRIX);
   MatGetLocalSize(*strength_mat, &local_rows, &local_cols);
   MatGetSize(*strength_mat, &global_rows, &global_cols);
   // This returns the global index of the local portion of the matrix
   MatGetOwnershipRange(*strength_mat, &global_row_start, &global_row_end_plus_one);

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr, *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr, *device_nonlocal_vals = nullptr;  
   MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype);  
   if (mpi) MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype);   
   const PetscInt *device_local_i_transpose = nullptr, *device_local_j_transpose = nullptr, *device_nonlocal_i_transpose = nullptr, *device_nonlocal_j_transpose = nullptr;
   if (mpi) MatSeqAIJGetCSRAndMemType(mat_nonlocal_transpose, &device_nonlocal_i_transpose, &device_nonlocal_j_transpose, NULL, &mtype);   

   // Device memory for the global variable cf_markers_local_d - be careful these aren't petsc ints
   cf_markers_local_d = intKokkosView("cf_markers_local_d", local_rows);
   // Can't use the global directly within the parallel 
   // regions on the device so just take a shallow copy
   intKokkosView cf_markers_d = cf_markers_local_d;    

   intKokkosView cf_markers_nonlocal_d;
   int *cf_markers_d_ptr = NULL, *cf_markers_nonlocal_d_ptr = NULL;
   cf_markers_d_ptr = cf_markers_d.data();

   // Host and device memory for the measure
   PetscScalarKokkosViewHost measure_local_h(measure_local, local_rows);
   PetscScalarKokkosView measure_local_d("measure_local_d", local_rows);   
   // Initialise to zero as we start with a comm'd sum
   Kokkos::deep_copy(measure_local_d, 0.0);     
   PetscScalar *measure_local_d_ptr = NULL, *measure_nonlocal_d_ptr = NULL;
   measure_local_d_ptr = measure_local_d.data();
   PetscScalarKokkosView measure_nonlocal_d;

   // ~~~~~~~~~~~~~~~
   // veto stores whether a node has been veto'd as a candidate
   // .NOT. veto(i) means the node can be in the set
   // veto(i) means the node cannot be in the set
   // ~~~~~~~~~~~~~~~
   // Device memory for the veto
   boolKokkosView veto_local_d("veto_local_d", local_rows);       
   boolKokkosView veto_nonlocal_d;   
   bool *veto_local_d_ptr = nullptr, *veto_nonlocal_d_ptr = nullptr;
   veto_local_d_ptr = veto_local_d.data();

   if (mpi) {
      measure_nonlocal_d = PetscScalarKokkosView("measure_nonlocal_d", cols_ao);   
      measure_nonlocal_d_ptr = measure_nonlocal_d.data();
      cf_markers_nonlocal_d = intKokkosView("cf_markers_nonlocal_d", cols_ao); 
      cf_markers_nonlocal_d_ptr = cf_markers_nonlocal_d.data();
      veto_nonlocal_d = boolKokkosView("veto_nonlocal_d", cols_ao); 
      veto_nonlocal_d_ptr = veto_nonlocal_d.data();
   }

   // The PETSC_MEMTYPE_KOKKOS is either as PETSC_MEMTYPE_HOST or 
   // one of the backends like PETSC_MEMTYPE_HIP
   PetscMemType mem_type = PETSC_MEMTYPE_KOKKOS;   

   // ~~~~~~~~~~~~
   // Compute the measure for S + S^T
   // ~~~~~~~~~~~~

   // We start by filling in the non-local strong influences
   if (mpi)
   {
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, cols_ao), KOKKOS_LAMBDA(PetscInt i) {

         // This is all the local strong influences
         measure_nonlocal_d(i) = device_nonlocal_i_transpose[i + 1] - device_nonlocal_i_transpose[i];
      });

      // We need a sum here as there are many ranks that might contribute
      PetscSFReduceWithMemTypeBegin(mat_mpi->Mvctx, MPIU_SCALAR,
                                 mem_type, measure_nonlocal_d_ptr,
                                 mem_type, measure_local_d_ptr,
                                 MPI_SUM);    


      // Compute the local transpose while we wait
      MatTranspose(mat_local, MAT_INITIAL_MATRIX, &mat_local_transpose);  

      // Finish the comms                            
      PetscSFReduceEnd(mat_mpi->Mvctx, MPIU_SCALAR, measure_nonlocal_d_ptr, measure_local_d_ptr, MPI_SUM);                                      
   }
   else
   {
      MatTranspose(mat_local, MAT_INITIAL_MATRIX, &mat_local_transpose); 
   }

   // Get pointers to the local transpose
   MatSeqAIJGetCSRAndMemType(mat_local_transpose, &device_local_i_transpose, &device_local_j_transpose, NULL, &mtype);

   // Scope this as we don't need the device copy of the randoms for very long
   {
      PetscScalarKokkosView measure_rand_d("measure_rand_d", local_rows);   
      // If you want to generate the randoms on the device
      //Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
      // Copy the input measure from host to device
      Kokkos::deep_copy(measure_rand_d, measure_local_h);  
      // Log copy with petsc
      size_t bytes = measure_local_h.extent(0) * sizeof(PetscReal);
      PetscLogCpuToGpu(bytes);      

      // Now measure_local_d should have the non-local strong influences in it
      // So now we can add in the local information we have plus a random number
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt i) {

         // Randoms on the device
         // auto generator = random_pool.get_state();
         // measure_rand_d(i) = generator.drand(0., 1.);
         // random_pool.free_state(generator);

         // Add in the random number
         measure_local_d(i) += measure_rand_d(i);

         // This is all the local strong dependencies
         measure_local_d(i) += device_local_i[i + 1] - device_local_i[i];

         // This is all the local strong influences
         measure_local_d(i) += device_local_i_transpose[i + 1] - device_local_i_transpose[i];

         // This is the non-local strong dependencies
         if (mpi)
         {
            measure_local_d(i) += device_nonlocal_i[i + 1] - device_nonlocal_i[i];
         }
         // Flip the sign if pmis
         if (pmis_int == 1) measure_local_d(i) *= -1;
      });
   }

   // Now our local measure_local_d is correct 
   // We need to comm it to other ranks
   if (mpi)
   {
      // Have to make sure we don't modify measure_local_d while the comms is in progress
      PetscSFBcastWithMemTypeBegin(mat_mpi->Mvctx, MPIU_SCALAR,
                                 mem_type, measure_local_d_ptr,
                                 mem_type, measure_nonlocal_d_ptr,
                                 MPI_REPLACE);      
   }

   // ~~~~~~~~~~~~
   // Initialise the set
   // ~~~~~~~~~~~~
   PetscInt counter_in_set_start = 0;
   // Count how many in the set to begin with and set their CF markers
   Kokkos::parallel_reduce ("Reduction", local_rows, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update) {
      if (Kokkos::abs(measure_local_d[i]) < 1) 
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
      else
      {
         cf_markers_d(i) = 0;
      }      
   }, counter_in_set_start);

   // Check the total number of undecided in parallel
   PetscInt counter_undecided, counter_parallel;
   if (max_luby_steps < 0) {
      counter_undecided = local_rows - counter_in_set_start;
      // Parallel reduction!
      MPI_Allreduce(&counter_undecided, &counter_parallel, 1, MPIU_INT, MPI_SUM, MPI_COMM_MATRIX);
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
      PetscSFBcastEnd(mat_mpi->Mvctx, MPIU_SCALAR, measure_local_d_ptr, measure_nonlocal_d_ptr, MPI_REPLACE);     
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

         // We can't overwrite any of the values in cf_markers_d while the forward scatter is still going
         // Be careful these aren't petscints
         PetscSFBcastWithMemTypeBegin(mat_mpi->Mvctx, MPI_INT,
                     mem_type, cf_markers_d_ptr,
                     mem_type, cf_markers_nonlocal_d_ptr,
                     MPI_REPLACE);
      }

      // ~~~~~~~~
      // Now we use veto to keep track of which candidates can be in the set
      // Locally we know which ones cannot be in the set due to local strong dependencies (mat_local),
      // strong influences (mat_local_tranpose), and non-local dependencies (mat_nonlocal)
      // but not the non-local influences as they are stored on many other ranks (ie in S^T)
      // ~~~~~~~~      

      // Let's start by veto'ing any candidates that have strong local dependencies or influences
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            const PetscInt i = t.league_rank();
            PetscInt strong_influences = 0, strong_dependencies = 0;

            // Check this row is unassigned
            if (cf_markers_d(i) == 0)
            {
               PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];

               // Reduce over local columns to get the number of strong unassigned dependencies
               Kokkos::parallel_reduce(
                  Kokkos::TeamThreadRange(t, ncols_local),
                  [&](const PetscInt j, PetscInt& strong_count) {     

                  // Have to only check active strong dependencies
                  if (measure_local_d(i) >= measure_local_d(device_local_j[device_local_i[i] + j]) && \
                           cf_markers_d(device_local_j[device_local_i[i] + j]) == 0)
                  {
                     strong_count++;
                  }
               
               }, strong_dependencies
               );     

               // Only bother doing the influences if needed
               if (strong_dependencies == 0) 
               {
                  ncols_local = device_local_i_transpose[i + 1] - device_local_i_transpose[i];

                  // Reduce over local columns to get the number of strong unassigned influences
                  Kokkos::parallel_reduce(
                     Kokkos::TeamThreadRange(t, ncols_local),
                     [&](const PetscInt j, PetscInt& strong_count) {     

                     // Have to only check active strong influences
                     if (measure_local_d(i) >= measure_local_d(device_local_j_transpose[device_local_i_transpose[i] + j]) && \
                              cf_markers_d(device_local_j_transpose[device_local_i_transpose[i] + j]) == 0)
                     {
                        strong_count++;
                     }

                  }, strong_influences
                  );      
               }           

               // Only want one thread in the team to write the result
               Kokkos::single(Kokkos::PerTeam(t), [&]() {                  
                  // If we have any strong neighbours
                  if (strong_dependencies > 0 || strong_influences > 0) 
                  {
                     veto_local_d(i) = true;     
                  }
                  else
                  {
                     veto_local_d(i) = false;  
                  }
               });
            }
            // Any that aren't zero cf marker are already assigned so set to 1
            else
            {
               // Only want one thread in the team to write the result
               Kokkos::single(Kokkos::PerTeam(t), [&]() {                  
                  veto_local_d(i) = true;
               });
            }
      });

      // ~~~~~~~~
      // Now let's go through and veto candidates which have strong influences on this rank
      // ~~~~~~~~           
      if (mpi) {

         // Initialise to true
         Kokkos::deep_copy(veto_nonlocal_d, true);     
         
         // Finish the async scatter
         // Be careful these aren't petscints
         PetscSFBcastEnd(mat_mpi->Mvctx, MPI_INT, cf_markers_d_ptr, cf_markers_nonlocal_d_ptr, MPI_REPLACE);         

         // Let's go and mark any non-local entries that don't have strong influences and comm to them other ranks
         Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), cols_ao, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

               // Row
               const PetscInt i = t.league_rank();
               PetscInt strong_influences = 0;

               // Check this row is unassigned
               if (cf_markers_nonlocal_d(i) == 0) 
               {
                  PetscInt ncols_nonlocal = device_nonlocal_i_transpose[i + 1] - device_nonlocal_i_transpose[i];

                  // Reduce over nonlocal columns in the transpose to get the number of strong unassigned influences
                  Kokkos::parallel_reduce(
                     Kokkos::TeamThreadRange(t, ncols_nonlocal),
                     [&](const PetscInt j, PetscInt& strong_count) {     

                     // Have to only check active strong influences
                     if (measure_nonlocal_d(i) >= measure_local_d(device_nonlocal_j_transpose[device_nonlocal_i_transpose[i] + j]) && \
                              cf_markers_local_d(device_nonlocal_j_transpose[device_nonlocal_i_transpose[i] + j]) == 0)
                     {
                        strong_count++;
                     }
                  
                  }, strong_influences
                  );    
               } 

               // Only want one thread in the team to write the result
               Kokkos::single(Kokkos::PerTeam(t), [&]() {                  
                  // If this non-local node doesn't have strong influences on this rank
                  // it may be a candidate to be in the set
                  if (strong_influences == 0) veto_nonlocal_d(i) = false;
               });
         });

         // Now we reduce the vetos with a lor
         PetscSFReduceWithMemTypeBegin(mat_mpi->Mvctx, MPI_C_BOOL,
            mem_type, veto_nonlocal_d_ptr,
            mem_type, veto_local_d_ptr,
            MPI_LOR);
         // Not sure we have any chance to overlap this with anything else
         PetscSFReduceEnd(mat_mpi->Mvctx, MPI_C_BOOL, veto_nonlocal_d_ptr, veto_local_d_ptr, MPI_LOR);

         // Now the comms have finished, we know exactly which local nodes on this rank have no 
         // local strong dependencies, influences, non-local influences but not yet non-local dependencies
         // Let's do the non-local dependencies and then now that the comms are done on veto_local_d
         // the combination of both of those gives us all our vetos, so we can assign anything without
         // a veto into the set 
         Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

               // Row
               const PetscInt i = t.league_rank();
               PetscInt strong_neighbours = 0;

               // Check this row isn't already marked
               if (!veto_local_d(i))
               {
                  PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];

                  // Reduce over nonlocal columns to get the number of non-local strong unassigned dependencies
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
                     // If we don't have any non-local strong dependencies and the rest of our vetos are false
                     // we know we are in the set
                     if (strong_neighbours == 0 && !veto_local_d(i)) cf_markers_d(i) = loops_through;
                  });
               }
         });
      }
      // This cf_markers_d(i) = loops_through happens above in the case of mpi, saves a kernel launch
      else
      {
         // The nodes that have .NOT. veto(i) have no strong active neighbours in the IS
         // hence they can be in the IS
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt i) {

               if (!veto_local_d(i)) cf_markers_d(i) = loops_through;
         });      
      }

      // ~~~~~~~~~~~~~
      // At this point all the local cf_markers that have been included in the set in this loop are correct
      // We need to set all the strong neighbours of these as not in the set
      // We can do all the local strong dependencies and influences without comms, but we need to do 
      // comms to set the non-local strong dependencies and influences
      // ~~~~~~~~~~~~~

      // Go and do local strong dependencies and influences
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            const PetscInt i = t.league_rank();

            // Check if this node has been assigned during this top loop
            if (cf_markers_d(i) == loops_through)
            {
               // Do the strong dependencies
               PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];

               Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

                     // Needs to be atomic as may being set by many threads
                     // Tried a version where instead of a "push" approach I tried a pull approach
                     // that doesn't need an atomic, but it was slower
                     Kokkos::atomic_store(&cf_markers_d(device_local_j[device_local_i[i] + j]), 1);     
               });
               
               // Do the strong influences
               ncols_local = device_local_i_transpose[i + 1] - device_local_i_transpose[i];

               Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

                     // Needs to be atomic as may being set by many threads
                     Kokkos::atomic_store(&cf_markers_d(device_local_j_transpose[device_local_i_transpose[i] + j]), 1);     
               });                 
            }
      });  

      // Now we need to set any non-local dependencies or influences of local nodes added to the set in this loop
      // to be not in the set
      if (mpi) 
      {
         // Now for the influences, we need to broadcast the cf_markers so that 
         // on other ranks we know which nodes have cf_markers_nonlocal_d(i) == loops_through
         // Be careful these aren't petscints
         PetscSFBcastWithMemTypeBegin(mat_mpi->Mvctx, MPI_INT,
                     mem_type, cf_markers_d_ptr,
                     mem_type, cf_markers_nonlocal_d_ptr,
                     MPI_REPLACE); 
                     
         // We can overlap this with setting the non-local dependencies

         // We use the veto arrays here to do this comms
         Kokkos::deep_copy(veto_nonlocal_d, false);
         Kokkos::deep_copy(veto_local_d, false);

         // Set non-local strong dependencies 
         Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
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
                        // If false set it with true, if not do nothing
                        Kokkos::atomic_compare_exchange(&veto_nonlocal_d(device_nonlocal_j[device_nonlocal_i[i] + j]), false, true);
                  });
               }
         });

         // Now we reduce the veto_nonlocal_d with a lor
         // Any local node with veto set to true is not in the set
         PetscSFReduceWithMemTypeBegin(mat_mpi->Mvctx, MPI_C_BOOL,
            mem_type, veto_nonlocal_d_ptr,
            mem_type, veto_local_d_ptr,
            MPI_LOR);
         // Not sure if there is anywhere to overlap these comms
         PetscSFReduceEnd(mat_mpi->Mvctx, MPI_C_BOOL, veto_nonlocal_d_ptr, veto_local_d_ptr, MPI_LOR);
         
         // Finish this before we write to cf_markers_d
         PetscSFBcastEnd(mat_mpi->Mvctx, MPI_INT, cf_markers_d_ptr, cf_markers_nonlocal_d_ptr, MPI_REPLACE);

         // Let's finish the non-local dependencies
         // If this node has been veto'd, then set it to not in the set
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt i) {
               if (veto_local_d(i)) {
                  cf_markers_d(i) = 1;
               }
         });           

         // And now we have the information we need to set any of the non-local influences
         Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), cols_ao, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

               // Row
               const PetscInt i = t.league_rank();

               // Check if this node has been assigned during this top loop
               if (cf_markers_nonlocal_d(i) == loops_through)
               {
                  PetscInt ncols_nonlocal = device_nonlocal_i_transpose[i + 1] - device_nonlocal_i_transpose[i];

                  // For over nonlocal columns
                  Kokkos::parallel_for(
                     Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {

                        // Needs to be atomic as may being set by many threads
                        Kokkos::atomic_store(&cf_markers_d(device_nonlocal_j_transpose[device_nonlocal_i_transpose[i] + j]), 1);     
                  });     
               }
         });
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
         MPI_Allreduce(&counter_undecided, &counter_parallel, 1, MPIU_INT, MPI_SUM, MPI_COMM_MATRIX);
         counter_undecided = counter_parallel;            
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
      if (pmis_int) cf_markers_d(i) *= -1;
   });

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Creates the device local indices for F or C points based on the global cf_markers_local_d
PETSC_INTERN void create_cf_is_device_kokkos(Mat *input_mat, const int match_cf, PetscIntKokkosView &is_local_d)
{
   PetscInt local_rows, local_cols;
   MatGetLocalSize(*input_mat, &local_rows, &local_cols); 

   // Can't use the global directly within the parallel 
   // regions on the device
   intKokkosView cf_markers_d = cf_markers_local_d;   

   // ~~~~~~~~~~~~
   // Get the F point local indices from cf_markers_local_d
   // ~~~~~~~~~~~~   
   PetscIntKokkosView point_offsets_d("point_offsets_d", local_rows+1);

   // Doing an exclusive scan to get the offsets for our local indices
   // Doing one larger so we can get the total number of points
   Kokkos::parallel_scan("point_offsets_d_scan",
      Kokkos::RangePolicy<>(0, local_rows+1),
      KOKKOS_LAMBDA(const PetscInt i, PetscInt& update, const bool final_pass) {
         bool is_f_point = false;
         if (i < local_rows) { // Predicate is based on original data up to local_rows-1
               is_f_point = (cf_markers_d(i) == match_cf); // is this point match_cf
         }         
         if (final_pass) {
               point_offsets_d(i) = update;
         }
         if (is_f_point) {
               update++;
         }
      }
   ); 

   // The last entry in point_offsets_d is the total number of points that match match_cf
   PetscInt local_rows_row = 0;
   Kokkos::deep_copy(local_rows_row, Kokkos::subview(point_offsets_d, local_rows));

   // This will be equivalent to is_fine - global_row_start, ie the local indices
   is_local_d = PetscIntKokkosView("is_local_d", local_rows_row);    

   // ~~~~~~~~~~~~
   // Write the local indices
   // ~~~~~~~~~~~~     
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt i) { 
         // Is this point match_cf
         if (cf_markers_d(i) == match_cf) {
            // point_offsets_d(i) gives the correct local index
            is_local_d(point_offsets_d(i)) = i;
         }              
   });
}

//------------------------------------------------------------------------------------------------------------------------

// Creates the host IS is_fine and is_coarse based on the global cf_markers_local_d
PETSC_INTERN void create_cf_is_kokkos(Mat *input_mat, IS *is_fine, IS *is_coarse)
{
   PetscIntKokkosView is_fine_local_d, is_coarse_local_d;
   MPI_Comm MPI_COMM_MATRIX;
   PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX);

   // Create the local f point indices
   const int match_fine = -1; // F_POINT == -1
   create_cf_is_device_kokkos(input_mat, match_fine, is_fine_local_d);

   // Create the local C point indices
   const int match_coarse = 1; // C_POINT == 1
   create_cf_is_device_kokkos(input_mat, match_coarse, is_coarse_local_d);

   // Now convert them back to global indices
   PetscInt global_row_start, global_row_end_plus_one;
   MatGetOwnershipRange(*input_mat, &global_row_start, &global_row_end_plus_one);   

   // Convert F points
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, is_fine_local_d.extent(0)), KOKKOS_LAMBDA(PetscInt i) { 
   
      is_fine_local_d(i) += global_row_start;
   });
   // Convert C points
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, is_coarse_local_d.extent(0)), KOKKOS_LAMBDA(PetscInt i) { 
   
      is_coarse_local_d(i) += global_row_start;
   });       

   // Create some host space for the indices
   PetscInt *is_fine_array = nullptr, *is_coarse_array = nullptr;
   PetscInt n_fine = is_fine_local_d.extent(0);
   PetscMalloc1(n_fine, &is_fine_array);
   PetscIntKokkosViewHost is_fine_h = PetscIntKokkosViewHost(is_fine_array, is_fine_local_d.extent(0));   
   PetscInt n_coarse = is_coarse_local_d.extent(0);
   PetscMalloc1(n_coarse, &is_coarse_array);
   PetscIntKokkosViewHost is_coarse_h = PetscIntKokkosViewHost(is_coarse_array, n_coarse);   

   // Copy over the indices to the host
   Kokkos::deep_copy(is_fine_h, is_fine_local_d);
   Kokkos::deep_copy(is_coarse_h, is_coarse_local_d);
   // Log copy with petsc
   size_t bytes_fine = is_fine_local_d.extent(0) * sizeof(PetscInt);
   size_t bytes_coarse = is_coarse_local_d.extent(0) * sizeof(PetscInt);
   PetscLogGpuToCpu(bytes_fine + bytes_coarse);

   // Now we can create the IS objects
   ISCreateGeneral(MPI_COMM_MATRIX, is_fine_local_d.extent(0), is_fine_array, PETSC_OWN_POINTER, is_fine);
   ISCreateGeneral(MPI_COMM_MATRIX, is_coarse_local_d.extent(0), is_coarse_array, PETSC_OWN_POINTER, is_coarse);
}

//------------------------------------------------------------------------------------------------------------------------

// Computes the diagonal dominance ratio of the input matrix over fine points in global variable cf_markers_local_d
// This code is very similar to MatCreateSubMatrix_kokkos
PETSC_INTERN void MatDiagDomRatio_kokkos(Mat *input_mat, PetscIntKokkosView &is_fine_local_d, PetscScalarKokkosView &diag_dom_ratio_d)
{
   PetscInt local_rows, local_cols;

   // Are we in parallel?
   MatType mat_type;
   MPI_Comm MPI_COMM_MATRIX;
   MatGetType(*input_mat, &mat_type);

   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;   
   PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX);
   MatGetLocalSize(*input_mat, &local_rows, &local_cols); 
   
   Mat_MPIAIJ *mat_mpi = nullptr;
   Mat mat_local = NULL, mat_nonlocal = NULL;   

   PetscInt rows_ao, cols_ao;
   if (mpi)
   {
      mat_mpi = (Mat_MPIAIJ *)(*input_mat)->data;
      mat_local = mat_mpi->A;
      mat_nonlocal = mat_mpi->B;
      MatGetSize(mat_nonlocal, &rows_ao, &cols_ao);
   }
   else
   {
      mat_local = *input_mat;
   }   

   // Can't use the global directly within the parallel 
   // regions on the device
   intKokkosView cf_markers_d = cf_markers_local_d;   
   intKokkosView cf_markers_nonlocal_d;

   // ~~~~~~~~~~~~
   // Get the F point local indices from cf_markers_local_d
   // ~~~~~~~~~~~~
   const int match_cf = -1; // F_POINT == -1
   create_cf_is_device_kokkos(input_mat, match_cf, is_fine_local_d);
   PetscInt local_rows_row = is_fine_local_d.extent(0);

   // Create device memory for the diag_dom_ratio
   diag_dom_ratio_d = PetscScalarKokkosView("diag_dom_ratio_d", local_rows_row);    

   // ~~~~~~~~~~~~~~~
   // Can now go and compute the diagonal dominance sums
   // ~~~~~~~~~~~~~~~
   int *cf_markers_d_ptr = cf_markers_d.data();
   int *cf_markers_nonlocal_d_ptr = NULL;
   PetscMemType mem_type = PETSC_MEMTYPE_KOKKOS;       
   PetscMemType mtype;

   // The off-diagonal component requires some comms which we can start now
   if (mpi)
   {
      cf_markers_nonlocal_d = intKokkosView("cf_markers_nonlocal_d", cols_ao); 
      cf_markers_nonlocal_d_ptr = cf_markers_nonlocal_d.data();   

      // Start the scatter of the cf splitting - the kokkos memtype is set as PETSC_MEMTYPE_HOST or 
      // one of the kokkos backends like PETSC_MEMTYPE_HIP
      // Be careful these aren't petscints
      PetscSFBcastWithMemTypeBegin(mat_mpi->Mvctx, MPI_INT,
                  mem_type, cf_markers_d_ptr,
                  mem_type, cf_markers_nonlocal_d_ptr,
                  MPI_REPLACE);
   }   

   // ~~~~~~~~~~~~~~~
   // Do the local component so work/comms are overlapped
   // ~~~~~~~~~~~~~~~

   // ~~~~~~~~~~~~
   // Get pointers to the local i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr;
   PetscScalar *device_local_vals = nullptr;
   MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype);  

   // Have to store the diagonal entry
   PetscScalarKokkosView diag_entry_d = PetscScalarKokkosView("diag_entry_d", local_rows_row);   
   Kokkos::deep_copy(diag_entry_d, 0);

   // Scoping to reduce peak memory
   {
      // We now go and do a reduce to get the diagonal entry, while also 
      // summing up the local non-diagonals into diag_dom_ratio_d
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows_row, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            const PetscInt i_idx_is_row = t.league_rank();
            const PetscInt i = is_fine_local_d(i_idx_is_row);
            const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];

            PetscScalar sum_val = 0.0;

            // Reduce over local columns
            Kokkos::parallel_reduce(
               Kokkos::TeamVectorRange(t, ncols_local),
               [&](const PetscInt j, PetscScalar& thread_sum) {

                  // Get this local column in the input_mat
                  const PetscInt target_col = device_local_j[device_local_i[i] + j];
                  // Is this column fine? F_POINT == -1
                  if (cf_markers_d(target_col) == -1)
                  {               
                     // Is this column the diagonal
                     const bool is_diagonal = i == target_col;

                     // Get the abs value of the entry
                     PetscScalar val = Kokkos::abs(device_local_vals[device_local_i[i] + j]);   
                     
                     // We have found a diagonal in this row
                     if (is_diagonal) {
                        // Will only happen for one thread
                        diag_entry_d(i_idx_is_row) = val;                     
                     }                
                     else
                     {
                        thread_sum += val;
                     }
                  }
               },
               Kokkos::Sum<PetscScalar>(sum_val)
            );

            // Only want one thread in the team to write the result
            Kokkos::single(Kokkos::PerTeam(t), [&]() {
               diag_dom_ratio_d(i_idx_is_row) = sum_val;
            });
      });  
   }

   // ~~~~~~~~~~~~~~~
   // Finish the comms and add the non-local entries to diag_dom_ratio_d
   // before we divide by the diagonal entry
   // ~~~~~~~~~~~~~~~

   // The off-diagonal component requires some comms
   // Basically a copy of MatCreateSubMatrix_MPIAIJ_SameRowColDist
   if (mpi)
   {           
      // Finish the scatter of the cf splitting
      // Be careful these aren't petscints
      PetscSFBcastEnd(mat_mpi->Mvctx, MPI_INT, cf_markers_d_ptr, cf_markers_nonlocal_d_ptr, MPI_REPLACE);

      // ~~~~~~~~~~~~
      // Get pointers to the nonlocal i,j,vals on the device
      // ~~~~~~~~~~~~
      const PetscInt *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
      PetscScalar *device_nonlocal_vals = nullptr;        
      MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype);        

      // Sum up the nonlocal matching entries into diag_dom_ratio_d
      if (cols_ao > 0) 
      {      
         Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows_row, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

               const PetscInt i_idx_is_row = t.league_rank();
               const PetscInt i = is_fine_local_d(i_idx_is_row);
               const PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];

               PetscScalar sum_val = 0.0;

               // Reduce over local columns
               Kokkos::parallel_reduce(
                  Kokkos::TeamVectorRange(t, ncols_nonlocal),
                  [&](const PetscInt j, PetscScalar& thread_sum) {

                     // This is the non-local column we have to check is present
                     const PetscInt target_col = device_nonlocal_j[device_nonlocal_i[i] + j];
                     // Is this column in the input IS? F_POINT == -1
                     if (cf_markers_nonlocal_d(target_col) == -1)
                     {               
                        // Get the abs value of the entry
                        thread_sum += Kokkos::abs(device_nonlocal_vals[device_nonlocal_i[i] + j]);
                     }
                  },
                  Kokkos::Sum<PetscScalar>(sum_val)
               );

               // Only want one thread in the team to write the result
               Kokkos::single(Kokkos::PerTeam(t), [&]() {
                  // Add into existing
                  diag_dom_ratio_d(i_idx_is_row) += sum_val;
               });
         });  
      }       
   }

   // ~~~~~~~~~~~~~
   // Compute the diag dominance ratio
   // ~~~~~~~~~~~~~
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows_row), KOKKOS_LAMBDA(PetscInt i) {     
         
      // If diag_val is zero we didn't find a diagonal
      if (diag_entry_d(i) != 0.0){
         // Compute the diagonal dominance ratio
         diag_dom_ratio_d(i) = diag_dom_ratio_d(i) / diag_entry_d(i);
      }
      else{
         diag_dom_ratio_d(i) = 0.0;
      }
   });   

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// ddc cleanup but on the device - uses the global variable cf_markers_local_d
// This no longer copies back to the host pointer cf_markers_local at the end
// You have to explicitly call copy_cf_markers_d2h(cf_markers_local) to do this
PETSC_INTERN void ddc_kokkos(Mat *input_mat, const PetscReal fraction_swap)
{
   // Can't use the global directly within the parallel 
   // regions on the device
   intKokkosView cf_markers_d = cf_markers_local_d;  
   PetscScalarKokkosView diag_dom_ratio_d;
   PetscIntKokkosView is_fine_local_d;
   
   // Compute the diagonal dominance ratio over the fine points in cf_markers_local_d
   // ie the diag domminance ratio of Aff
   MatDiagDomRatio_kokkos(input_mat, is_fine_local_d, diag_dom_ratio_d);
   PetscInt local_rows_aff = is_fine_local_d.extent(0);

   // Do a fixed alpha_diag
   PetscInt search_size;
   if (fraction_swap < 0) {
      // We have to look through all the local rows
      search_size = local_rows_aff;
   }
   // Or pick alpha_diag based on the worst % of rows
   else {
      // Only need to go through the biggest % of indices
      search_size = static_cast<PetscInt>(double(local_rows_aff) * fraction_swap);
   }   

   // Can't put this above because of collective operations in parallel (namely the MatDiagDomRatio_kokkos)
   // If we have local points to swap
   if (search_size > 0)
   {
      // Create device memory for bins
      auto dom_bins_d = PetscIntKokkosView("dom_bins_d", 1000);
      Kokkos::deep_copy(dom_bins_d, 0);

      // Bin the diagonal dominance ratio
      if (fraction_swap > 0)
      {
         Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, local_rows_aff), KOKKOS_LAMBDA(PetscInt i) { 

            // Let's bin the entry
            int bin;
            int test_bin = floor(diag_dom_ratio_d(i) * double(dom_bins_d.extent(0))) + 1;
            if (test_bin < int(dom_bins_d.extent(0)) && test_bin >= 0) {
               bin = test_bin;
            }
            else {
               bin = dom_bins_d.extent(0);
            }
            // Has to be atomic as many threads from different rows
            // may be writing to the same bin
            Kokkos::atomic_add(&dom_bins_d(bin - 1), 1);
         });
      }

      PetscReal swap_dom_val;
      // Do a fixed alpha_diag
      if (fraction_swap < 0){
         swap_dom_val = -fraction_swap;
      }
      // Otherwise swap everything bigger than a fixed fraction
      else{

         // Parallel scan to inclusive sum the number of entries we have in 
         // the bins
         Kokkos::parallel_scan (dom_bins_d.extent(0), KOKKOS_LAMBDA (const PetscInt i, PetscInt& update, const bool final) {
            // Inclusive scan
            update += dom_bins_d(i);         
            if (final) {
               dom_bins_d(i) = update; // only update array on final pass
            }
         });        

         // Now if we reduce how many are > the search_size, we know the bin boundary we want
         int bin_boundary = 0;  
         Kokkos::parallel_reduce ("ReductionBin", dom_bins_d.extent(0), KOKKOS_LAMBDA (const int i, int& update) {
            if (dom_bins_d(i) > dom_bins_d(dom_bins_d.extent(0)-1) - search_size) update++;
         }, bin_boundary);   

         bin_boundary = dom_bins_d.extent(0) - bin_boundary;                
         swap_dom_val = double(bin_boundary) / double(dom_bins_d.extent(0));

      }

      // Go and swap F points to C points
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows_aff), KOKKOS_LAMBDA(PetscInt i) {

            if (diag_dom_ratio_d(i) != 0.0 && diag_dom_ratio_d(i) >= swap_dom_val)
            {
               // This is the actual numbering in A, rather than Aff
               PetscInt idx = is_fine_local_d(i);
               cf_markers_d(idx) *= -1;
            }
      }); 
   }   

   return;
}

//------------------------------------------------------------------------------------------------------------------------