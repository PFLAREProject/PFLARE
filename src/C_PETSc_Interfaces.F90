module c_petsc_interfaces

   use iso_c_binding
   use petscsys

#include "petsc/finclude/petscsys.h"

   implicit none

   public

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! Contains interfaces to C functions defined in C_routines.cpp which 
   ! are used to get around a lack of some PETSc fortran interfaces
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------      

   interface   
      
      subroutine ShellSetVecType_c(A_array, B_array) &
         bind(c, name="ShellSetVecType_c")
         use iso_c_binding
         integer(c_long_long) :: A_array, B_array
      end subroutine ShellSetVecType_c         
 
   end interface     

   interface   
       
      subroutine mat_mat_symbolic_c(A_array, B_array, C_array) &
         bind(c, name="mat_mat_symbolic_c")
         use iso_c_binding
         integer(c_long_long) :: A_array
         integer(c_long_long) :: B_array
         integer(c_long_long) :: C_array
      end subroutine mat_mat_symbolic_c         

   end interface

   interface   
      
      subroutine vecscatter_mat_begin_c(A_array, vec_long, cf_markers_nonlocal) &
         bind(c, name="vecscatter_mat_begin_c")
         use iso_c_binding
         integer(c_long_long) :: A_array, vec_long
         type(c_ptr) :: cf_markers_nonlocal

      end subroutine vecscatter_mat_begin_c         
 
   end interface   
   
   interface   
      
      subroutine vecscatter_mat_end_c(A_array, vec_long, cf_markers_nonlocal) &
         bind(c, name="vecscatter_mat_end_c")
         use iso_c_binding
         integer(c_long_long) :: A_array, vec_long
         type(c_ptr) :: cf_markers_nonlocal

      end subroutine vecscatter_mat_end_c         
 
   end interface     

   interface   
      
      subroutine vecscatter_mat_restore_c(A_array, cf_markers_nonlocal) &
         bind(c, name="vecscatter_mat_restore_c")
         use iso_c_binding
         integer(c_long_long) :: A_array
         type(c_ptr) :: cf_markers_nonlocal

      end subroutine vecscatter_mat_restore_c         
 
   end interface  
   
   interface   
      
      subroutine vecscatter_mat_reverse_begin_c(A_array, vec_long) &
         bind(c, name="vecscatter_mat_reverse_begin_c")
         use iso_c_binding
         integer(c_long_long) :: A_array, vec_long

      end subroutine vecscatter_mat_reverse_begin_c         
 
   end interface   
   
   interface   
      
      subroutine vecscatter_mat_reverse_end_c(A_array, vec_long) &
         bind(c, name="vecscatter_mat_reverse_end_c")
         use iso_c_binding
         integer(c_long_long) :: A_array, vec_long

      end subroutine vecscatter_mat_reverse_end_c         
 
   end interface    
   
   interface   
      
      subroutine MatSeqAIJGetArrayF90_mine(A_array, array) &
            bind(c, name="MatSeqAIJGetArrayF90_mine")
            use iso_c_binding
            integer(c_long_long) :: A_array
            type(c_ptr) :: array

      end subroutine MatSeqAIJGetArrayF90_mine         
 
   end interface   
   
   interface   
      
      subroutine allreducesum_petscint_mine(A_array, first_int, return_int) &
            bind(c, name="allreducesum_petscint_mine")
            use iso_c_binding
         integer(c_long_long) :: A_array
         PetscInt, value :: first_int
         PetscInt :: return_int

      end subroutine allreducesum_petscint_mine         
 
   end interface      

   interface   
      
      subroutine GenerateIS_ProcAgglomeration_c(proc_stride, global_size, local_size_reduced, start) &
         bind(c, name="GenerateIS_ProcAgglomeration_c")
         use iso_c_binding
         PetscInt, value :: proc_stride
         PetscInt, value :: global_size
         PetscInt :: local_size_reduced, start

      end subroutine GenerateIS_ProcAgglomeration_c         
 
   end interface 
   
   interface   
      
      subroutine MatPartitioning_c(A_array, n_parts, proc_stride, index) &
         bind(c, name="MatPartitioning_c")
         use iso_c_binding
         integer(c_long_long) :: A_array
         PetscInt, value :: n_parts
         PetscInt :: proc_stride
         integer(c_long_long) :: index
      end subroutine MatPartitioning_c         
 
   end interface  
   
   interface   
      
      subroutine MatMPICreateNonemptySubcomm_c(A_array, on_subcomm, B_array) &
         bind(c, name="MatMPICreateNonemptySubcomm_c")
         use iso_c_binding
         integer(c_long_long) :: A_array
         integer(c_int)       :: on_subcomm
         integer(c_long_long) :: B_array
      end subroutine MatMPICreateNonemptySubcomm_c         
 
   end interface

   interface   
      
      subroutine MatGetNNZs_local_c(A_array, nnzs) &
         bind(c, name="MatGetNNZs_local_c")
         use iso_c_binding
         integer(c_long_long) :: A_array
         PetscInt :: nnzs
      end subroutine MatGetNNZs_local_c
 
   end interface
   
   interface   
      
      subroutine MatGetNNZs_both_c(A_array, nnzs_local, nnzs_nonlocal) &
         bind(c, name="MatGetNNZs_both_c")
         use iso_c_binding
         integer(c_long_long) :: A_array
         PetscInt :: nnzs_local, nnzs_nonlocal
      end subroutine MatGetNNZs_both_c
 
   end interface      
   
   interface   
      
      subroutine c_PCGetStructureFlag(A_array, flag) &
         bind(c, name="c_PCGetStructureFlag")
         use iso_c_binding
         integer(c_long_long) :: A_array
         integer(c_int) :: flag
      end subroutine c_PCGetStructureFlag
 
   end interface
   
   interface   
      
      subroutine PCGetSetupCalled_c(A_array, setupcalled) &
         bind(c, name="PCGetSetupCalled_c")
         use iso_c_binding
         integer(c_long_long) :: A_array
         PetscInt :: setupcalled
      end subroutine PCGetSetupCalled_c         
 
   end interface

   interface   
      
      subroutine MatGetDiagonalOnly_c(A_array, diag_only) &
         bind(c, name="MatGetDiagonalOnly_c")
         use iso_c_binding
         integer(c_long_long) :: A_array
         integer(c_int) :: diag_only
      end subroutine MatGetDiagonalOnly_c         
 
   end interface   

   interface   
      
      subroutine generate_identity_is_kokkos(A_array, index, B_array) &
         bind(c, name="generate_identity_is_kokkos")
         use iso_c_binding
         integer(c_long_long) :: A_array
         integer(c_long_long) :: index
         integer(c_long_long) :: B_array
      end subroutine generate_identity_is_kokkos         
 
   end interface 
   
   interface   
      
      subroutine remove_small_from_sparse_kokkos(A_array, tol, B_array, &
                     relative_max_row_tolerance_int, lump_int, allow_drop_diagonal_int) &
         bind(c, name="remove_small_from_sparse_kokkos")
         use iso_c_binding
         integer(c_long_long) :: A_array
         PetscReal, value :: tol
         integer(c_long_long) :: B_array
         integer(c_int), value :: relative_max_row_tolerance_int
         integer(c_int), value :: lump_int
         integer(c_int), value :: allow_drop_diagonal_int
      end subroutine remove_small_from_sparse_kokkos         
 
   end interface

   interface   
      
      subroutine remove_from_sparse_match_kokkos(A_array, B_array, lump_int, alpha_int, alpha) &
         bind(c, name="remove_from_sparse_match_kokkos")
         use iso_c_binding
         integer(c_long_long) :: A_array
         integer(c_long_long) :: B_array
         integer(c_int), value :: lump_int, alpha_int
         PetscReal, value :: alpha
      end subroutine remove_from_sparse_match_kokkos         
 
   end interface   

   interface   
      
      subroutine MatSetAllValues_kokkos(A_array, val) &
         bind(c, name="MatSetAllValues_kokkos")
         use iso_c_binding
         integer(c_long_long) :: A_array
         PetscReal, value :: val
      end subroutine MatSetAllValues_kokkos         
 
   end interface   
   
   interface   
      
      subroutine create_VecISCopyLocal_kokkos(max_levels_input) &
         bind(c, name="create_VecISCopyLocal_kokkos")
         use iso_c_binding
         integer(c_int), value :: max_levels_input
      end subroutine create_VecISCopyLocal_kokkos         
 
   end interface  
   
   interface   
      
      subroutine destroy_VecISCopyLocal_kokkos() &
         bind(c, name="destroy_VecISCopyLocal_kokkos")
         use iso_c_binding
      end subroutine destroy_VecISCopyLocal_kokkos         
 
   end interface    
   
   interface   
      
      subroutine set_VecISCopyLocal_kokkos_our_level(our_level, global_row_start, index_fine, index_coarse) &
         bind(c, name="set_VecISCopyLocal_kokkos_our_level")
         use iso_c_binding
         integer(c_int), value :: our_level
         PetscInt, value :: global_row_start
         integer(c_long_long) :: index_fine
         integer(c_long_long) :: index_coarse
      end subroutine set_VecISCopyLocal_kokkos_our_level         
 
   end interface
   
   interface   
      
      subroutine VecISCopyLocal_kokkos(our_level, fine_int, vfull, mode_int, vreduced) &
         bind(c, name="VecISCopyLocal_kokkos")
         use iso_c_binding
         integer(c_int), value :: our_level, fine_int, mode_int
         integer(c_long_long) :: vfull
         integer(c_long_long) :: vreduced
      end subroutine VecISCopyLocal_kokkos         
 
   end interface    

   interface   
      
      subroutine create_cf_is_kokkos(A_array, index_fine, index_coarse) &
         bind(c, name="create_cf_is_kokkos")
         use iso_c_binding
         integer(c_long_long) :: A_array
         integer(c_long_long) :: index_fine
         integer(c_long_long) :: index_coarse
      end subroutine create_cf_is_kokkos         
 
   end interface   

   interface   
      
      subroutine pmisr_kokkos(A_array, max_luby_steps, pmis_int, measure_local, zero_meaure_c_point_int) &
         bind(c, name="pmisr_kokkos")
         use iso_c_binding
         integer(c_long_long) :: A_array
         type(c_ptr), value :: measure_local
         integer(c_int), value :: max_luby_steps, pmis_int, zero_meaure_c_point_int
      end subroutine pmisr_kokkos         
 
   end interface     

   interface   
      
      subroutine ddc_kokkos(A_array, fraction_swap) &
         bind(c, name="ddc_kokkos")
         use iso_c_binding
         integer(c_long_long) :: A_array
         PetscReal, value :: fraction_swap
      end subroutine ddc_kokkos         
 
   end interface 
   
   interface   
      
      subroutine copy_cf_markers_d2h(cf_markers_local) &
         bind(c, name="copy_cf_markers_d2h")
         use iso_c_binding
         type(c_ptr), value :: cf_markers_local
      end subroutine copy_cf_markers_d2h         
 
   end interface 
   
   interface   
      
      subroutine delete_device_cf_markers() &
         bind(c, name="delete_device_cf_markers")
         use iso_c_binding
      end subroutine delete_device_cf_markers         
 
   end interface       

   interface   
      
      subroutine compute_P_from_W_kokkos(A_array, global_row_start, indices_fine, &
                     indices_coarse, identity_int, reuse_int, B_array) &
         bind(c, name="compute_P_from_W_kokkos")
         use iso_c_binding
         integer(c_long_long) :: A_array, indices_fine, indices_coarse
         integer(c_long_long) :: B_array
         PetscInt, value :: global_row_start
         integer(c_int), value :: identity_int, reuse_int
      end subroutine compute_P_from_W_kokkos         
 
   end interface   

   interface   
      
      subroutine generate_one_point_with_one_entry_from_sparse_kokkos(A_array, B_array) &
         bind(c, name="generate_one_point_with_one_entry_from_sparse_kokkos")
         use iso_c_binding
         integer(c_long_long) :: A_array, B_array
      end subroutine generate_one_point_with_one_entry_from_sparse_kokkos         
 
   end interface    
   
   interface   
      
      subroutine compute_R_from_Z_kokkos(A_array, global_row_start, indices_fine, &
                     indices_coarse, indices_orig, identity_int, reuse_int, reuse_indices_int, B_array) &
         bind(c, name="compute_R_from_Z_kokkos")
         use iso_c_binding
         integer(c_long_long) :: A_array, indices_fine, indices_coarse, indices_orig
         integer(c_long_long) :: B_array
         PetscInt, value :: global_row_start
         integer(c_int), value :: identity_int, reuse_int, reuse_indices_int
      end subroutine compute_R_from_Z_kokkos         
 
   end interface
   
   interface   
      
      subroutine build_gmres_polynomial_inverse_0th_order_kokkos(A_array, poly_order, &
                  coefficients, reuse_int, B_array) &
         bind(c, name="build_gmres_polynomial_inverse_0th_order_kokkos")
         use iso_c_binding
         integer(c_long_long) :: A_array
         integer(c_long_long) :: B_array
         integer(c_int), value :: poly_order
         type(c_ptr), value :: coefficients
         integer(c_int), value :: reuse_int
      end subroutine build_gmres_polynomial_inverse_0th_order_kokkos         
 
   end interface
   
   interface   
      
      subroutine build_gmres_polynomial_inverse_0th_order_sparsity_kokkos(A_array, poly_order, &
                  coefficients, reuse_int, B_array) &
         bind(c, name="build_gmres_polynomial_inverse_0th_order_sparsity_kokkos")
         use iso_c_binding
         integer(c_long_long) :: A_array
         integer(c_long_long) :: B_array
         integer(c_int), value :: poly_order
         type(c_ptr), value :: coefficients
         integer(c_int), value :: reuse_int
      end subroutine build_gmres_polynomial_inverse_0th_order_sparsity_kokkos         
 
   end interface    

   interface   
      
      subroutine mat_mult_powers_share_sparsity_kokkos(A_array, poly_order, poly_sparsity_order, &
                  coefficients, reuse_int_reuse_mat, reuse_array, reuse_int_cmat, B_array) &
         bind(c, name="mat_mult_powers_share_sparsity_kokkos")
         use iso_c_binding
         integer(c_long_long) :: A_array
         integer(c_long_long) :: B_array, reuse_array
         integer(c_int), value :: poly_order, poly_sparsity_order
         type(c_ptr), value :: coefficients
         integer(c_int), value :: reuse_int_cmat, reuse_int_reuse_mat
      end subroutine mat_mult_powers_share_sparsity_kokkos         
 
   end interface     

   interface   
      
      subroutine mat_duplicate_copy_plus_diag_kokkos(A_array, reuse_int, B_array) &
         bind(c, name="mat_duplicate_copy_plus_diag_kokkos")
         use iso_c_binding
         integer(c_long_long) :: A_array
         integer(c_long_long) :: B_array
         integer(c_int), value :: reuse_int
      end subroutine mat_duplicate_copy_plus_diag_kokkos         
 
   end interface   

   interface   
      
      subroutine MatAXPY_kokkos(A_array, alpha, B_array) &
         bind(c, name="MatAXPY_kokkos")
         use iso_c_binding
         integer(c_long_long) :: A_array
         integer(c_long_long) :: B_array
         PetscScalar, value :: alpha
      end subroutine MatAXPY_kokkos         
 
   end interface
   
   interface   
      
      subroutine MatCreateSubMatrix_kokkos(A_array, is_row, is_col, &
                     reuse_int, B_array, &
                     our_level, is_row_fine_int, is_col_fine_int) &
         bind(c, name="MatCreateSubMatrix_kokkos")
         use iso_c_binding
         integer(c_long_long) :: A_array
         integer(c_long_long) :: B_array
         integer(c_long_long) :: is_row, is_col
         integer(c_int), value :: our_level, is_row_fine_int, is_col_fine_int, reuse_int

      end subroutine MatCreateSubMatrix_kokkos

   end interface

! -------------------------------------------------------------------------------------------------------------------------------

end module c_petsc_interfaces

