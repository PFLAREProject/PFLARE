// Our petsc kokkos definitions - has to go first
#include "kokkos_helper.hpp"
#include <iostream>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>


//------------------------------------------------------------------------------------------------------------------------

// Build a 0th order gmres polynomial but with kokkos - keeping everything on the device
PETSC_INTERN void build_gmres_polynomial_inverse_0th_order_kokkos(Mat *input_mat, int poly_order, PetscReal *coefficients, \
                     int reuse_int, Mat *output_mat)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   MatType mat_type;
   PetscInt nnzs_match_local, nnzs_match_nonlocal;
   Mat output_mat_local, output_mat_nonlocal;

   MatGetType(*input_mat, &mat_type);
   // Are we in parallel?
   bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   // Get the comm
   PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX);
   MatGetLocalSize(*input_mat, &local_rows, &local_cols);
   MatGetSize(*input_mat, &global_rows, &global_cols);       

   // ~~~~~~~~~~~~
   // Get the number of nnzs
   // ~~~~~~~~~~~~
   nnzs_match_local = local_rows;
   nnzs_match_nonlocal = 0;

   // Get device views
   Kokkos::View<PetscScalar *> a_local_d;
   Kokkos::View<PetscInt *> i_local_d;    
   Kokkos::View<PetscInt *> j_local_d;

   // Nonlocal stuff 
   Kokkos::View<PetscScalar *> a_nonlocal_d;
   Kokkos::View<PetscInt *> i_nonlocal_d;          
   Kokkos::View<PetscInt *> j_nonlocal_d;  

   // ~~~~~~~~~~~~~~~~~  
   // We need to assemble our i,j, vals so we can build our matrix
   // ~~~~~~~~~~~~~~~~~   
   if (!reuse_int)
   {  
      // Create device & host memory
      a_local_d = Kokkos::View<PetscScalar *>("a_local_d", nnzs_match_local);
      i_local_d = Kokkos::View<PetscInt *>("i_local_d", local_rows+1);
      j_local_d = Kokkos::View<PetscInt *>("j_local_d", nnzs_match_local);           

      // we also have to go and build the a, i, j for the non-local off-diagonal block
      if (mpi) 
      {
         // Create non-local host and device memory
         a_nonlocal_d = Kokkos::View<PetscScalar *>("a_nonlocal_d", nnzs_match_nonlocal);
         i_nonlocal_d = Kokkos::View<PetscInt *>("i_nonlocal_d", local_rows+1);
         j_nonlocal_d = Kokkos::View<PetscInt *>("j_nonlocal_d", nnzs_match_nonlocal);  

         // All zero, no non-local entries
         Kokkos::deep_copy(i_nonlocal_d, 0);                
      }               

      // ~~~~~~~~~~~~~~~
      // Create i indices
      // ~~~~~~~~~~~~~~~
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows+1), KOKKOS_LAMBDA(int i) {

            i_local_d(i) = i;
      });  
      // ~~~~~~~~~~~~~~~
      // Create j indices
      // ~~~~~~~~~~~~~~~
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(int i) {

            j_local_d(i) = i;
      });    
      // 0th order polynomial is just the first coefficient on the diagonal
      // Copy it straight from the host
      Kokkos::deep_copy(a_local_d, coefficients[0]);  
      // Log copy with petsc
      size_t bytes = sizeof(PetscReal);
      PetscLogCpuToGpu(bytes);  
      
      // Let's make sure everything on the device is finished
      auto exec = PetscGetKokkosExecutionSpace();
      exec.fence();
      
      // We can create our local diagonal block matrix directly on the device
      MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, local_cols, i_local_d, j_local_d, a_local_d, &output_mat_local);        

      // we also have to go and build the a, i, j for the non-local off-diagonal block
      if (mpi) 
      {
         // Zero off-diagonal entries
         PetscInt *garray_host = NULL;
         PetscInt col_ao_output = 0;
         // Silly but depending on the compiler this may return a non-null pointer
         PetscMalloc1(col_ao_output, &garray_host);      
         
         // We can create our nonlocal diagonal block matrix directly on the device
         MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, col_ao_output, i_nonlocal_d, j_nonlocal_d, a_nonlocal_d, &output_mat_nonlocal);         

         // We can now create our MPI matrix
         MatCreateMPIAIJWithSeqAIJ(MPI_COMM_MATRIX, output_mat_local, output_mat_nonlocal, garray_host, output_mat);
      }     
      // If in serial 
      else
      {
         *output_mat = output_mat_local;
      }
   }
   // With re-use
   else
   {
      Mat_MPIAIJ *mat_mpi_output = nullptr;
      Mat mat_local_output; 

      // Get the existing output mats
      if (mpi)
      {
         mat_mpi_output = (Mat_MPIAIJ *)(*output_mat)->data;
         mat_local_output = mat_mpi_output->A;
      }
      else
      {
         mat_local_output = *output_mat;
      }     
      Mat_SeqAIJKokkos *aijkok_local_output = static_cast<Mat_SeqAIJKokkos *>(mat_local_output->spptr);
      // Annoying we can't just call MatSeqAIJGetKokkosView
      a_local_d = aijkok_local_output->a_dual.view_device();
      // Copy in the host value directly
      Kokkos::deep_copy(a_local_d, coefficients[0]);  
      // Log copy with petsc
      size_t bytes = sizeof(PetscReal);
      PetscLogCpuToGpu(bytes);         

      // Have to specify we've modifed local data on the device
      // Want to call MatSeqAIJKokkosModifyDevice but its PETSC_INTERN
      aijkok_local_output->a_dual.clear_sync_state();
      aijkok_local_output->a_dual.modify_device();
      // Transpose is the same
      //aijkok_local_output->transpose_updated = PETSC_FALSE;
      //aijkok_local_output->hermitian_updated = PETSC_FALSE;
      // Invalidate diagonals
      Mat_SeqAIJ *a = (Mat_SeqAIJ *)mat_local_output->data;
      a->idiagvalid  = PETSC_FALSE;
      a->ibdiagvalid = PETSC_FALSE;      
      a->inode.ibdiagvalid = PETSC_FALSE;           
      PetscObjectStateIncrease((PetscObject)(*output_mat));
   }

   return;
}

//------------------------------------------------------------------------------------------------------------------------

// Build a gmres polynomial with 0th order sparsity but with kokkos - keeping everything on the device
PETSC_INTERN void build_gmres_polynomial_inverse_0th_order_sparsity_kokkos(Mat *input_mat, int poly_order, PetscReal *coefficients, \
                     int reuse_int, Mat *output_mat)
{
   MPI_Comm MPI_COMM_MATRIX;
   PetscInt local_rows, local_cols, global_rows, global_cols;
   MatType mat_type;
   PetscInt nnzs_match_local, nnzs_match_nonlocal;
   Mat output_mat_local, output_mat_nonlocal;

   MatGetType(*input_mat, &mat_type);
   // Are we in parallel?
   bool mpi = strcmp(mat_type, MATMPIAIJKOKKOS) == 0;

   // Get the comm
   PetscObjectGetComm((PetscObject)*input_mat, &MPI_COMM_MATRIX);
   MatGetLocalSize(*input_mat, &local_rows, &local_cols);
   MatGetSize(*input_mat, &global_rows, &global_cols);  

   // We also copy the coefficients over to the device as we need it
   PetscInt coeff_size = poly_order + 1;
   auto coefficients_h = PetscScalarKokkosViewHost(coefficients, coeff_size);
   auto coefficients_d = PetscScalarKokkosView("coefficients_d", coeff_size);
   Kokkos::deep_copy(coefficients_d, coefficients_h);       
   // Log copy with petsc
   size_t bytes = coefficients_h.extent(0) * sizeof(PetscReal);
   PetscLogCpuToGpu(bytes);      

   // ~~~~~~~~~~~~
   // Get the number of nnzs
   // ~~~~~~~~~~~~
   nnzs_match_local = local_rows;
   nnzs_match_nonlocal = 0;

   // Get device views
   Kokkos::View<PetscScalar *> a_local_d;
   Kokkos::View<PetscInt *> i_local_d;    
   Kokkos::View<PetscInt *> j_local_d;

   // Nonlocal stuff 
   Kokkos::View<PetscScalar *> a_nonlocal_d;
   Kokkos::View<PetscInt *> i_nonlocal_d;          
   Kokkos::View<PetscInt *> j_nonlocal_d;  

   Mat_MPIAIJ *mat_mpi_output = nullptr;
   Mat mat_local_output; 
   Mat_SeqAIJKokkos *aijkok_local_output;

   // ~~~~~~~~~~~~~~~~~  
   // We need to assemble our i,j, vals so we can build our matrix
   // ~~~~~~~~~~~~~~~~~
   if (!reuse_int)
   {  
      // Create device & host memory
      a_local_d = Kokkos::View<PetscScalar *>("a_local_d", nnzs_match_local);
      i_local_d = Kokkos::View<PetscInt *>("i_local_d", local_rows+1);
      j_local_d = Kokkos::View<PetscInt *>("j_local_d", nnzs_match_local);      

      // we also have to go and build the a, i, j for the non-local off-diagonal block
      if (mpi) 
      {
         // Create non-local host and device memory
         a_nonlocal_d = Kokkos::View<PetscScalar *>("a_nonlocal_d", nnzs_match_nonlocal);
         i_nonlocal_d = Kokkos::View<PetscInt *>("i_nonlocal_d", local_rows+1);
         j_nonlocal_d = Kokkos::View<PetscInt *>("j_nonlocal_d", nnzs_match_nonlocal);  

         // All zero, no non-local entries
         Kokkos::deep_copy(i_nonlocal_d, 0);                
      }               

      // ~~~~~~~~~~~~~~~
      // Create i indices
      // ~~~~~~~~~~~~~~~
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows+1), KOKKOS_LAMBDA(int i) {

            i_local_d(i) = i;
      });  
      // ~~~~~~~~~~~~~~~
      // Create j indices
      // ~~~~~~~~~~~~~~~
      Kokkos::parallel_for(
         Kokkos::RangePolicy<>(0, local_rows), KOKKOS_LAMBDA(int i) {

            j_local_d(i) = i;
      });    
   }
   // Reuse - get view to a 
   else
   {
      mat_mpi_output = nullptr;

      // Get the existing output mats
      if (mpi)
      {
         mat_mpi_output = (Mat_MPIAIJ *)(*output_mat)->data;
         mat_local_output = mat_mpi_output->A;
      }
      else
      {
         mat_local_output = *output_mat;
      }     
      aijkok_local_output = static_cast<Mat_SeqAIJKokkos *>(mat_local_output->spptr);
      // Annoying we can't just call MatSeqAIJGetKokkosView
      a_local_d = aijkok_local_output->a_dual.view_device();      
   }

   // ~~~~~~~~~~~~~~~~~~~~~~~
   // Compute the diagonal entries
   // ~~~~~~~~~~~~~~~~~~~~~~~

   // Get the matrix diagonal
   Vec diag_vec;
   MatCreateVecs(*input_mat, NULL, &diag_vec);
   MatGetDiagonal(*input_mat, diag_vec);
   ConstPetscScalarKokkosView diag_vec_d;
   VecGetKokkosView(diag_vec, &diag_vec_d);    

   // Loop over the rows
   Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), local_rows, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {

      PetscInt i   = t.league_rank(); // row i
      PetscReal row_val = 0.0;

      // For this row, can do all the powers independently and then sum them
      // Add in the 1st order term to the last
      Kokkos::parallel_reduce(
         Kokkos::TeamThreadRange(t, 1, poly_order+1),
         [&](const PetscInt j, PetscReal& thread_data) {

            thread_data += coefficients_d[j] * pow(diag_vec_d(i), j);
         }, row_val
      );

      // Only want one thread in the team to write the result
      Kokkos::single(Kokkos::PerTeam(t), [&]() {     
         // Add the powers and the 0th order coefficient
         a_local_d(i) = row_val + coefficients_d[0];
      });      
   });    

   VecRestoreKokkosView(diag_vec, &diag_vec_d);    
   VecDestroy(&diag_vec);

   // ~~~~~~~~~~~~~~~~~~~~~~~

   // If we're not reusing we need to build our matrices
   if (!reuse_int)
   {
      // Let's make sure everything on the device is finished
      auto exec = PetscGetKokkosExecutionSpace();
      exec.fence();

      // We can create our local diagonal block matrix directly on the device
      MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, local_cols, i_local_d, j_local_d, a_local_d, &output_mat_local);  

      // we also have to go and build the a, i, j for the non-local off-diagonal block
      if (mpi) 
      {
         // Zero off-diagonal entries
         PetscInt *garray_host = NULL;
         PetscInt col_ao_output = 0;
         // Silly but depending on the compiler this may return a non-null pointer
         PetscMalloc1(col_ao_output, &garray_host);      
         
         // We can create our nonlocal diagonal block matrix directly on the device
         MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, local_rows, col_ao_output, i_nonlocal_d, j_nonlocal_d, a_nonlocal_d, &output_mat_nonlocal);

         // We can now create our MPI matrix
         MatCreateMPIAIJWithSeqAIJ(MPI_COMM_MATRIX, output_mat_local, output_mat_nonlocal, garray_host, output_mat);
      }     
      // If in serial 
      else
      {
         *output_mat = output_mat_local;
      }
   }
   // With re-use
   else
   {
      // Have to specify we've modifed local data on the device
      // Want to call MatSeqAIJKokkosModifyDevice but its PETSC_INTERN
      aijkok_local_output->a_dual.clear_sync_state();
      aijkok_local_output->a_dual.modify_device();
      // Transpose is the same
      //aijkok_local_output->transpose_updated = PETSC_FALSE;
      //aijkok_local_output->hermitian_updated = PETSC_FALSE;
      // Invalidate diagonals
      Mat_SeqAIJ *a = (Mat_SeqAIJ *)mat_local_output->data;
      a->idiagvalid  = PETSC_FALSE;
      a->ibdiagvalid = PETSC_FALSE;      
      a->inode.ibdiagvalid = PETSC_FALSE;           
      PetscObjectStateIncrease((PetscObject)(*output_mat));
   }

   return;
}
