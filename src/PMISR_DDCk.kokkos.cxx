// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

//------------------------------------------------------------------------------------------------------------------------

// PMISR cf splitting but on the device
PETSC_INTERN void pmisr_kokkos(Mat *strength_mat, const int max_luby_steps, const int pmis_int, PetscReal *measure_local, int *cf_markers_local, const int zero_measure_c_point_int)
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
   Mat mat_local = NULL, mat_nonlocal = NULL;

   if (mpi)
   {
      mat_mpi = (Mat_MPIAIJ *)(*strength_mat)->data;
      mat_local = mat_mpi->A;
      mat_nonlocal = mat_mpi->B;
      MatGetSize(mat_nonlocal, &rows_ao, &cols_ao); 
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

   // Host and device memory for the cf_markers - be careful these aren't petsc ints
   intKokkosViewHost cf_markers_local_h(cf_markers_local, local_rows);
   intKokkosView cf_markers_local_d("cf_markers_local_d", local_rows);
   PetscScalar *cf_markers_local_real_d_ptr = NULL, *cf_markers_nonlocal_real_d_ptr = NULL;
   // The real equivalents so we can use the existing petscsf from the input matrix
   PetscScalarKokkosView cf_markers_local_real_d("cf_markers_local_real_d", local_rows);
   PetscScalarKokkosView cf_markers_nonlocal_real_d;
   cf_markers_local_real_d_ptr = cf_markers_local_real_d.data();

   // Host and device memory for the measure
   PetscScalarKokkosViewHost measure_local_h(measure_local, local_rows);
   PetscScalarKokkosView measure_local_d("measure_local_d", local_rows);   
   PetscScalarKokkosView measure_nonlocal_d;
   if (mpi) {
      measure_nonlocal_d = PetscScalarKokkosView("measure_nonlocal_d", cols_ao);   
      cf_markers_nonlocal_real_d = PetscScalarKokkosView("cf_markers_nonlocal_real_d", cols_ao); 
      cf_markers_nonlocal_real_d_ptr = cf_markers_nonlocal_real_d.data();
   }

   // Device memory for the mark
   boolKokkosView mark_d("mark_d", local_rows);   

   // If you want to generate the randoms on the device
   //Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
   // Copy the input measure from host to device
   Kokkos::deep_copy(measure_local_d, measure_local_h);  
   // Log copy with petsc
   size_t bytes = measure_local_h.extent(0) * sizeof(PetscReal);
   PetscLogCpuToGpu(bytes);   

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

   // Copy the measure over
   Kokkos::deep_copy(cf_markers_local_real_d, measure_local_d);  

   // Start the scatter of the measure - the kokkos memtype is set as PETSC_MEMTYPE_HOST or 
   // one of the kokkos backends like PETSC_MEMTYPE_HIP
   PetscMemType mem_type = PETSC_MEMTYPE_KOKKOS;
   if (mpi)
   {
      PetscSFBcastWithMemTypeBegin(mat_mpi->Mvctx, MPIU_SCALAR,
                                 mem_type, cf_markers_local_real_d_ptr,
                                 mem_type, cf_markers_nonlocal_real_d_ptr,
                                 MPI_REPLACE);      
   }

   // Initialise the set
   PetscInt counter_in_set_start = 0;
   // Count how many in the set to begin with
   Kokkos::parallel_reduce ("Reduction", local_rows, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update) {
      if (Kokkos::abs(measure_local_d[i]) < 1) update++;
   }, counter_in_set_start);

   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt i) {

      if (Kokkos::abs(measure_local_d(i)) < 1)
      {
         if (zero_measure_c_point_int == 1) {
            if (pmis_int == 1) {
               // Set as F here but reversed below to become C
               cf_markers_local_real_d(i) = -1;
            }
            else {
               // Becomes C
               cf_markers_local_real_d(i) = 1;
            }  
         }
         else {
            if (pmis_int == 1) {
               // Set as C here but reversed below to become F
               // Otherwise dirichlet conditions persist down onto the coarsest grid
               cf_markers_local_real_d(i) = 1;
            }
            else {
               // Becomes F
               cf_markers_local_real_d(i) = -1;
            }
         }
      }
      else
      {
         cf_markers_local_real_d(i) = 0;
      }
   });  

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
      PetscSFBcastEnd(mat_mpi->Mvctx, MPIU_SCALAR, cf_markers_local_real_d_ptr, cf_markers_nonlocal_real_d_ptr, MPI_REPLACE);
      // Copy the non local measure
      Kokkos::deep_copy(measure_nonlocal_d, cf_markers_nonlocal_real_d);       
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
         // We can't overwrite any of the values in cf_markers_local_real_d while the forward scatter is still going
         PetscSFBcastWithMemTypeBegin(mat_mpi->Mvctx, MPIU_SCALAR,
                     mem_type, cf_markers_local_real_d_ptr,
                     mem_type, cf_markers_nonlocal_real_d_ptr,
                     MPI_REPLACE);
      }

      // This keeps track of which of the candidate nodes can become in the set
      // Only need this because we want to do async comms so we need a way to trigger
      // a node not being in the set due to either strong local neighbours *or* strong offproc neighbours
      Kokkos::deep_copy(mark_d, true);   

      // Any that aren't zero cf marker are already assigned so set to to false
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt i) {

            if (cf_markers_local_real_d(i) != 0) mark_d(i) = false;
      });

      // ~~~~~~~~
      // Go and do the local component
      // ~~~~~~~~      
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            const PetscInt i = t.league_rank();
            PetscInt strong_neighbours = 0;

            // Check this row isn't already marked
            if (cf_markers_local_real_d(i) == 0)
            {
               const PetscInt i = t.league_rank();
               const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];

               // Reduce over local columns to get the number of strong neighbours
               Kokkos::parallel_reduce(
                  Kokkos::TeamThreadRange(t, ncols_local),
                  [&](const PetscInt j, PetscInt& strong_count) {     

                  // Have to only check active strong neighbours
                  if (measure_local_d(i) >= measure_local_d(device_local_j[device_local_i[i] + j]) && \
                           cf_markers_local_real_d(device_local_j[device_local_i[i] + j]) == 0)
                  {
                     strong_count++;
                  }
               
               }, strong_neighbours
               );     

               // Only want one thread in the team to write the result
               Kokkos::single(Kokkos::PerTeam(t), [&]() {                  
                  // If we have any strong neighbours
                  if (strong_neighbours > 0) mark_d(i) = false;     
               });
            }
      });

      // ~~~~~~~~
      // Now go through and do the non-local part of the matrix
      // ~~~~~~~~           
      if (mpi) {

         // Finish the async scatter
         PetscSFBcastEnd(mat_mpi->Mvctx, MPIU_SCALAR, cf_markers_local_real_d_ptr, cf_markers_nonlocal_real_d_ptr, MPI_REPLACE);

         Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

               // Row
               const PetscInt i = t.league_rank();
               PetscInt strong_neighbours = 0;

               // Check this row isn't already marked
               if (cf_markers_local_real_d(i) == 0)
               {
                  const PetscInt i = t.league_rank();
                  PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];

                  // Reduce over nonlocal columns to get the number of strong neighbours
                  Kokkos::parallel_reduce(
                     Kokkos::TeamThreadRange(t, ncols_nonlocal),
                     [&](const PetscInt j, PetscInt& strong_count) {     

                     if (measure_local_d(i) >= measure_nonlocal_d(device_nonlocal_j[device_nonlocal_i[i] + j])  && \
                              cf_markers_nonlocal_real_d(device_nonlocal_j[device_nonlocal_i[i] + j]) == 0)
                     {
                        strong_count++;
                     }
                  
                  }, strong_neighbours
                  );     

                  // Only want one thread in the team to write the result
                  Kokkos::single(Kokkos::PerTeam(t), [&]() {                  
                     // If we have any strong neighbours
                     if (strong_neighbours > 0) mark_d(i) = false;     
                  });
               }
         });
      }

      // The nodes that have mark equal to true have no strong active neighbours in the IS
      // hence they can be in the IS
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(PetscInt i) {

            if (mark_d(i)) cf_markers_local_real_d(i) = double(loops_through);
      });      

      if (mpi) 
      {
         // We're going to do an add reverse scatter, so set them to zero
         Kokkos::deep_copy(cf_markers_nonlocal_real_d, 0.0);  

         Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

               // Row
               const PetscInt i = t.league_rank();

               // Check if this node has been assigned during this top loop
               if (cf_markers_local_real_d(i) == loops_through)
               {
                  const PetscInt i = t.league_rank();
                  PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];

                  // For over nonlocal columns
                  Kokkos::parallel_for(
                     Kokkos::TeamThreadRange(t, ncols_nonlocal), [&](const PetscInt j) {

                        // Needs to be atomic as may being set by many threads
                        Kokkos::atomic_store(&cf_markers_nonlocal_real_d(device_nonlocal_j[device_nonlocal_i[i] + j]), 1.0);     
                  });     
               }
         });

         // We've updated the values in cf_markers_nonlocal, which is a pointer to lvec
         // Calling a reverse scatter add will then update the values of cf_markers_local
         // Reduce with a sum, equivalent to VecScatterBegin with ADD_VALUES, SCATTER_REVERSE
         PetscSFReduceWithMemTypeBegin(mat_mpi->Mvctx, MPIU_SCALAR,
            mem_type, cf_markers_nonlocal_real_d_ptr,
            mem_type, cf_markers_local_real_d_ptr,
            MPIU_SUM);
      }

      // Go and do local
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            // Row
            const PetscInt i = t.league_rank();

            // Check if this node has been assigned during this top loop
            if (cf_markers_local_real_d(i) == loops_through)
            {
               const PetscInt i = t.league_rank();
               const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];

               // For over nonlocal columns
               Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(t, ncols_local), [&](const PetscInt j) {

                     // Needs to be atomic as may being set by many threads
                     Kokkos::atomic_store(&cf_markers_local_real_d(device_local_j[device_local_i[i] + j]), 1.0);     
               });     
            }
      });   

      if (mpi) 
      {
         // Finish the scatter
         PetscSFReduceEnd(mat_mpi->Mvctx, MPIU_SCALAR, cf_markers_nonlocal_real_d_ptr, cf_markers_local_real_d_ptr, MPIU_SUM);         
      }

      // We've done another top level loop
      loops_through = loops_through - 1;

      // ~~~~~~~~~~~~
      // Check the total number of undecided in parallel before we loop again
      // ~~~~~~~~~~~~
      if (max_luby_steps < 0) {

         counter_undecided = 0;  
         Kokkos::parallel_reduce ("ReductionCounter_undecided", local_rows, KOKKOS_LAMBDA (const PetscInt i, PetscInt& update) {
            if (cf_markers_local_real_d(i) == 0) update++;
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
         
      if (cf_markers_local_real_d(i) == 0)
      {
         cf_markers_local_d(i) = 1;
      }
      else if (cf_markers_local_real_d(i) < 0)
      {
         cf_markers_local_d(i) = -1;
      }
      else
      {
         cf_markers_local_d(i) = 1;
      }
      if (pmis_int) cf_markers_local_d(i) *= -1;
   });

   // Now copy device cf_markers_local_d back to host
   Kokkos::deep_copy(cf_markers_local_h, cf_markers_local_d);
   // Log copy with petsc
   bytes = cf_markers_local_d.extent(0) * sizeof(int);
   PetscLogGpuToCpu(bytes);

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Computes the diagonal dominance ratio of the input matrix over the input IS row/cols
// This version only works if the input IS have the same parallel row/column distribution 
// as the matrices
// is_col must be sorted
// This code is very similar to MatCreateSubMatrix_kokkos
PETSC_INTERN void MatDiagDomRatio_kokkos(Mat *input_mat, IS *is_row, PetscScalarKokkosView &diag_dom_ratio_d)
{
   PetscInt local_rows, local_cols;
   PetscInt global_rows, global_cols;
   PetscInt global_row_start, global_row_end_plus_one;
   MatGetOwnershipRange(*input_mat, &global_row_start, &global_row_end_plus_one);

   // Are we in parallel?
   MatType mat_type;
   MPI_Comm MPI_COMM_MATRIX;
   MatGetType(*input_mat, &mat_type);

   const bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;   
   PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX);
   MatGetSize(*input_mat, &global_rows, &global_cols); 
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

   // ~~~~~~~~~~~~
   // Transfer the IS's to the device
   // ~~~~~~~~~~~~   

   // Get pointers to the indices on the host
   const PetscInt *is_row_indices_ptr;
   ISGetIndices(*is_row, &is_row_indices_ptr);   

   PetscInt local_rows_row, global_rows_row;
   ISGetLocalSize(*is_row, &local_rows_row);   
   ISGetSize(*is_row, &global_rows_row);

   // Create a host view of the existing indices
   auto is_row_view_h = PetscIntConstKokkosViewHost(is_row_indices_ptr, local_rows_row);    
   auto is_row_d_d = PetscIntKokkosView("is_row_d_d", local_rows_row);   
     
   // Copy indices to the device
   Kokkos::deep_copy(is_row_d_d, is_row_view_h);     
   // Log copy with petsc
   size_t bytes = is_row_view_h.extent(0) * sizeof(PetscInt);
   PetscLogCpuToGpu(bytes);        

   ISRestoreIndices(*is_row, &is_row_indices_ptr);   

   // ~~~~~~~~~~~~
   // Rewrite to local indices
   // ~~~~~~~~~~~~     
   Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, is_row_d_d.extent(0)), KOKKOS_LAMBDA(PetscInt i) {      

         is_row_d_d(i) -= global_row_start; // Make local
   });

   // ~~~~~~~~~~~~~~~
   // Can now go and compute the sum over the diagonal component
   // ~~~~~~~~~~~~~~~

   // ~~~~~~~~~~~~
   // Get pointers to the i,j,vals on the device
   // ~~~~~~~~~~~~
   const PetscInt *device_local_i = nullptr, *device_local_j = nullptr;
   PetscMemType mtype;
   PetscScalar *device_local_vals = nullptr;
   MatSeqAIJGetCSRAndMemType(mat_local, &device_local_i, &device_local_j, &device_local_vals, &mtype);  

   // Have to store the diagonal entry
   PetscScalarKokkosView diag_entry_d = PetscScalarKokkosView("diag_entry_d", local_rows_row);   
   Kokkos::deep_copy(diag_entry_d, 0);

   // Scoping to reduce peak memory
   {
      // Map which columns in the original mat are in is_row
      PetscIntKokkosView smap_d = PetscIntKokkosView("smap_d", local_cols);  
      Kokkos::deep_copy(smap_d, 0); 
      // Loop over all the cols in is_row
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows_row), KOKKOS_LAMBDA(PetscInt i) {      

            smap_d(is_row_d_d(i)) = i + 1; 
      });    

      // We now go and do a reduce to get the diagonal entry, while also 
      // summing up the local non-diagonals into diag_dom_ratio_d
      Kokkos::parallel_for(
         Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows_row, Kokkos::AUTO()),
         KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

            const PetscInt i_idx_is_row = t.league_rank();
            const PetscInt i = is_row_d_d(i_idx_is_row);
            const PetscInt ncols_local = device_local_i[i + 1] - device_local_i[i];

            PetscScalar sum_val = 0.0;

            // Reduce over local columns
            Kokkos::parallel_reduce(
               Kokkos::TeamVectorRange(t, ncols_local),
               [&](const PetscInt j, PetscScalar& thread_sum) {

                  // Get this local column in the input_mat
                  PetscInt target_col = device_local_j[device_local_i[i] + j];
                  // Is this column in the input IS?
                  if (smap_d(target_col))
                  {               
                     // Is this column the diagonal
                     const bool is_diagonal = smap_d(target_col) - 1 == i_idx_is_row;

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
   // Now need to go and add the non-local entries to diag_dom_ratio_d
   // before we divide by the diagonal entry
   // ~~~~~~~~~~~~~~~

   // The off-diagonal component requires some comms
   // Basically a copy of MatCreateSubMatrix_MPIAIJ_SameRowColDist
   if (mpi)
   {
      // Basically a copy of ISGetSeqIS_SameColDist_Private
      /* (1) iscol is a sub-column vector of mat, pad it with '-1.' to form a full vector x */
      Vec x;
      Vec lvec = mat_mpi->lvec;
      MatCreateVecs(*input_mat, &x, NULL);
      VecSet(x, -1.0);

      // Use the vecs in the scatter provided by the input mat
      PetscScalarKokkosView x_d;
      VecGetKokkosView(x, &x_d);
      PetscScalarKokkosView lvec_d;
      VecGetKokkosView(lvec, &lvec_d);

      // Loop over all the cols in is_col
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows_row), KOKKOS_LAMBDA(PetscInt i) {      

            x_d(is_row_d_d(i)) = (PetscScalar)is_row_d_d(i); 
      });

      PetscScalar *x_d_ptr = NULL;
      x_d_ptr = x_d.data();      
      PetscScalar *lvec_d_ptr = NULL;
      lvec_d_ptr = lvec_d.data();       

      // Start the scatter of the x - the kokkos memtype is set as PETSC_MEMTYPE_HOST or 
      // one of the kokkos backends like PETSC_MEMTYPE_HIP
      PetscMemType mem_type = PETSC_MEMTYPE_KOKKOS;      
      PetscSFBcastWithMemTypeBegin(mat_mpi->Mvctx, MPIU_SCALAR,
                  mem_type, x_d_ptr,
                  mem_type, lvec_d_ptr,
                  MPI_REPLACE);        
                  
      // ~~~~~~~~~~~~
      // Get pointers to the i,j,vals on the device
      // ~~~~~~~~~~~~
      const PetscInt *device_nonlocal_i = nullptr, *device_nonlocal_j = nullptr;
      PetscScalar *device_nonlocal_vals = nullptr;
      MatSeqAIJGetCSRAndMemType(mat_nonlocal, &device_nonlocal_i, &device_nonlocal_j, &device_nonlocal_vals, &mtype);                  

      // Finish the x scatter
      PetscSFBcastEnd(mat_mpi->Mvctx, MPIU_SCALAR, x_d_ptr, lvec_d_ptr, MPI_REPLACE);      
      // We're done with x now
      VecRestoreKokkosView(x, &x_d);
      VecDestroy(&x);

      // Sum up the nonlocal matching entries into diag_dom_ratio_d
      if (cols_ao > 0) 
      {      
         Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows_row, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

               const PetscInt i_idx_is_row = t.league_rank();
               const PetscInt i = is_row_d_d(i_idx_is_row);
               const PetscInt ncols_nonlocal = device_nonlocal_i[i + 1] - device_nonlocal_i[i];

               PetscScalar sum_val = 0.0;

               // Reduce over local columns
               Kokkos::parallel_reduce(
                  Kokkos::TeamVectorRange(t, ncols_nonlocal),
                  [&](const PetscInt j, PetscScalar& thread_sum) {

                     // This is the non-local column we have to check is present
                     PetscInt target_col = device_nonlocal_j[device_nonlocal_i[i] + j];
                     // Is this column in the input IS?
                     if (lvec_d_ptr[target_col] > -1.0)
                     {               
                        // Get the abs value of the entry
                        sum_val += Kokkos::abs(device_nonlocal_vals[device_nonlocal_i[i] + j]);
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

      VecRestoreKokkosView(lvec, &lvec_d);
      
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

// ddc cleanup but on the device
PETSC_INTERN void ddc_kokkos(Mat *input_mat, IS *is_fine, const PetscReal fraction_swap, int *cf_markers_local)
{
   PetscInt local_rows, local_cols;

   // Get the comm
   MatGetLocalSize(*input_mat, &local_rows, &local_cols);

   // Host and device memory for the cf_markers - be careful these aren't petsc ints
   intKokkosViewHost cf_markers_local_h(cf_markers_local, local_rows);
   intKokkosView cf_markers_local_d("cf_markers_local_d", local_rows);   
   // Now copy cf markers to device
   Kokkos::deep_copy(cf_markers_local_d, cf_markers_local_h);  
   // Log copy with petsc
   size_t bytes = cf_markers_local_h.extent(0) * sizeof(int);
   PetscLogCpuToGpu(bytes);     

   // Get pointers to the indices on the host
   const PetscInt *fine_indices_ptr;
   ISGetIndices(*is_fine, &fine_indices_ptr);

   PetscInt fine_local_size;
   ISGetLocalSize(*is_fine, &fine_local_size);    
   
   PetscInt local_rows_aff;
   local_rows_aff = fine_local_size;
   PetscInt input_row_start, input_row_end_plus_one;
   MatGetOwnershipRange(*input_mat, &input_row_start, &input_row_end_plus_one);      

   // Create a host view of the existing indices
   auto fine_view_h = PetscIntConstKokkosViewHost(fine_indices_ptr, fine_local_size);    
   auto fine_view_d = PetscIntKokkosView("fine_view_d", fine_local_size);   
   // Copy fine indices to the device
   Kokkos::deep_copy(fine_view_d, fine_view_h);
   // Log copy with petsc
   bytes = fine_view_h.extent(0) * sizeof(PetscInt);
   PetscLogCpuToGpu(bytes);   

   // Do a fixed alpha_diag
   PetscInt search_size;
   if (fraction_swap < 0) {
      // We have to look through all the local rows
      search_size = fine_local_size;
   }
   // Or pick alpha_diag based on the worst % of rows
   else {
      // Only need to go through the biggest % of indices
      search_size = static_cast<PetscInt>(double(fine_local_size) * fraction_swap);
   }

   // Create device memory for the diag_dom_ratio
   auto diag_dom_ratio_d = PetscScalarKokkosView("diag_dom_ratio_d", local_rows_aff); 
   // Compute the diagonal dominance ratio over the is_fine rows/cols
   // ie the diag domminance ratio of Aff
   MatDiagDomRatio_kokkos(input_mat, is_fine, diag_dom_ratio_d);

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
               PetscInt idx = fine_view_d(i) - input_row_start;
               cf_markers_local_d(idx) *= -1;
            }
      }); 
   }

   // Now copy device cf_markers_local_d back to host
   Kokkos::deep_copy(cf_markers_local_h, cf_markers_local_d);
   bytes = cf_markers_local_d.extent(0) * sizeof(int);
   PetscLogGpuToCpu(bytes);       

   ISRestoreIndices(*is_fine, &fine_indices_ptr);

   return;
}

//------------------------------------------------------------------------------------------------------------------------