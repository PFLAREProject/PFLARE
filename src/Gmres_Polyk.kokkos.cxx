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

   // We also copy the coefficients over to the device as we need it
   PetscInt one = 1;
   auto coefficients_h = PetscScalarKokkosViewHost(coefficients, one);
   auto coefficients_d = PetscScalarKokkosView("colmap_input_d", one);
   Kokkos::deep_copy(coefficients_d, coefficients_h);   

   // ~~~~~~~~~~~~
   // Get the number of nnzs
   // ~~~~~~~~~~~~
   nnzs_match_local = local_rows;
   nnzs_match_nonlocal = 0;

   // Dual views
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

   // ~~~~~~~~~~~~~~~~~  
   // We need to assemble our i,j, vals so we can build our matrix
   // ~~~~~~~~~~~~~~~~~   
   if (!reuse_int)
   {  
      // Create device & host memory
      a_local_dual = MatScalarKokkosDualView("a_local_dual", nnzs_match_local);
      i_local_dual = MatRowMapKokkosDualView("i_local_dual", local_rows+1);
      j_local_dual = MatColIdxKokkosDualView("j_local_dual", nnzs_match_local);      

      // Get device views
      a_local_d = a_local_dual.view_device();
      i_local_d = i_local_dual.view_device();     
      j_local_d = j_local_dual.view_device();        

      // we also have to go and build the a, i, j for the non-local off-diagonal block
      if (mpi) 
      {
         // Create non-local host and device memory
         a_nonlocal_dual = MatScalarKokkosDualView("a_nonlocal_dual", nnzs_match_nonlocal);
         i_nonlocal_dual = MatRowMapKokkosDualView("i_nonlocal_dual", local_rows+1);
         j_nonlocal_dual = MatColIdxKokkosDualView("j_nonlocal_dual", nnzs_match_nonlocal);  

         // Get device views
         a_nonlocal_d = a_nonlocal_dual.view_device();
         i_nonlocal_d = i_nonlocal_dual.view_device();
         // All zero, no non-local entries
         Kokkos::deep_copy(i_nonlocal_d, 0);                
         j_nonlocal_d = j_nonlocal_dual.view_device();   
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
      Kokkos::deep_copy(a_local_d, coefficients_d(0));   

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

         // Zero off-diagonal entries
         PetscInt *garray_host = NULL;
         PetscInt col_ao_output = 0;
         // Silly but depending on the compiler this may return a non-null pointer
         PetscMalloc1(col_ao_output, &garray_host);      

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
      Kokkos::deep_copy(a_local_d, coefficients_d(0));  

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
   auto coefficients_d = PetscScalarKokkosView("colmap_input_d", coeff_size);
   Kokkos::deep_copy(coefficients_d, coefficients_h);         

   // ~~~~~~~~~~~~
   // Get the number of nnzs
   // ~~~~~~~~~~~~
   nnzs_match_local = local_rows;
   nnzs_match_nonlocal = 0;

   // Dual views
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
   Mat mat_local_output; 
   Mat_SeqAIJKokkos *aijkok_local_output;

   // ~~~~~~~~~~~~~~~~~  
   // We need to assemble our i,j, vals so we can build our matrix
   // ~~~~~~~~~~~~~~~~~
   if (!reuse_int)
   {  
      // Create device & host memory
      a_local_dual = MatScalarKokkosDualView("a_local_dual", nnzs_match_local);
      i_local_dual = MatRowMapKokkosDualView("i_local_dual", local_rows+1);
      j_local_dual = MatColIdxKokkosDualView("j_local_dual", nnzs_match_local);      

      // Get device views
      a_local_d = a_local_dual.view_device();
      i_local_d = i_local_dual.view_device();     
      j_local_d = j_local_dual.view_device();  

      // we also have to go and build the a, i, j for the non-local off-diagonal block
      if (mpi) 
      {
         // Create non-local host and device memory
         a_nonlocal_dual = MatScalarKokkosDualView("a_nonlocal_dual", nnzs_match_nonlocal);
         i_nonlocal_dual = MatRowMapKokkosDualView("i_nonlocal_dual", local_rows+1);
         j_nonlocal_dual = MatColIdxKokkosDualView("j_nonlocal_dual", nnzs_match_nonlocal);  

         // Get device views
         a_nonlocal_d = a_nonlocal_dual.view_device();
         i_nonlocal_d = i_nonlocal_dual.view_device();
         // All zero, no non-local entries
         Kokkos::deep_copy(i_nonlocal_d, 0);                
         j_nonlocal_d = j_nonlocal_dual.view_device();   
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

         // Zero off-diagonal entries
         PetscInt *garray_host = NULL;
         PetscInt col_ao_output = 0;
         // Silly but depending on the compiler this may return a non-null pointer
         PetscMalloc1(col_ao_output, &garray_host);      

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
