module c_fortran_bindings

   use petscksp
   use iso_c_binding
   use pcair_data_type, only: pc_air_multigrid_data
   use pcair_shell, only: PCReset_AIR_Shell, create_pc_air_shell
   use approx_inverse_setup, only: calculate_and_build_approximate_inverse, reset_inverse_mat
   use cf_splitting, only: compute_cf_splitting
   use matdiagdomsubmatrix, only: compute_diag_dom_submatrix
   use air_data_type_routines, only: create_air_data

#include "petsc/finclude/petscksp.h"

   implicit none

   public

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! Iso C bindings for PFLARE routines  
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------      

   contains 

   !------------------------------------------------------------------------------------------------------------------------

   subroutine create_pc_air_data_c(pc_air_data_c_ptr) bind(C,name='create_pc_air_data_c')

      ! Creates an air_data object, calls setup and returns a C pointer

      ! ~~~~~~~~
      type(c_ptr), intent(inout)             :: pc_air_data_c_ptr

      type(pc_air_multigrid_data), pointer   :: pc_air_data
      ! ~~~~~~~~     

      allocate(pc_air_data)
      call create_air_data(pc_air_data%air_data)
      ! Pass the setup pc_air_data object back into C 
      pc_air_data_c_ptr = c_loc(pc_air_data)

   end subroutine create_pc_air_data_c 

   !------------------------------------------------------------------------------------------------------------------------

   subroutine PCReset_AIR_Shell_c(pc_ptr) bind(C,name='PCReset_AIR_Shell_c')

      ! Calls the Fortran routine

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: pc_ptr

      type(tPC)  :: pc
      PetscErrorCode :: ierr
      ! ~~~~~~~~

      pc%v = pc_ptr

      ! Call the destroy routine
      call PCReset_AIR_Shell(pc, ierr)

   end subroutine PCReset_AIR_Shell_c  
   
   !------------------------------------------------------------------------------------------------------------------------

   subroutine create_pc_air_shell_c(pc_air_data_c_ptr, pc_ptr) bind(C,name='create_pc_air_shell_c')

      ! Calls the setup routine for air and returns a PC Shell as a long long
      ! The longlong pointer is defined in PC%v to pass in

      ! ~~~~~~~~
      type(c_ptr), intent(inout)          :: pc_air_data_c_ptr
      integer(c_long_long), intent(inout) :: pc_ptr

      type(pc_air_multigrid_data), pointer   :: pc_air_data
      type(tPC)  :: pc
      ! ~~~~~~~~

      ! Should have already been allocated in setup_pc_air_data_c
      call c_f_pointer(pc_air_data_c_ptr, pc_air_data)
      ! Now the input mat long long just gets copied into pc%v
      ! This works as the PETSc types are essentially just wrapped around
      ! pointers stored in %v
      pc%v = pc_ptr

      ! Call the setup routine
      call create_pc_air_shell(pc_air_data, pc)

      ! Now the PC has been modified so make sure to copy the pointer back
      pc_ptr = pc%v

   end subroutine create_pc_air_shell_c    
   
   !------------------------------------------------------------------------------------------------------------------------

   subroutine calculate_and_build_approximate_inverse_c(input_mat_ptr, inverse_type, &
         poly_order, poly_sparsity_order, &
         matrix_free_int, diag_scale_polys_int, subcomm_int, &
         coeffs_ptr, row_size, col_size, &
         inv_matrix_ptr) &
         bind(C, name='calculate_and_build_approximate_inverse_c')

      ! Builds an approximate inverse, with optional coefficient passing.
      !
      ! coeffs_ptr/row_size/col_size are in/out:
      !   On entry, if coeffs_ptr is c_null_ptr: compute fresh polynomial coefficients.
      !     On return, coeffs_ptr points to a C-malloc'd copy of the coefficients;
      !     the caller owns this memory and must free it with C free().
      !   On entry, if coeffs_ptr is non-null: reuse those coefficients; the polynomial
      !     computation is skipped (see calculate_and_build_approximate_inverse).
      !     coeffs_ptr/row_size/col_size are unchanged on return.

#include "finclude/PETSc_ISO_Types.h"

      ! Interface to C stdlib malloc
      interface
         function c_malloc(sz) bind(C, name='malloc')
            use iso_c_binding
            integer(c_size_t), value :: sz
            type(c_ptr) :: c_malloc
         end function c_malloc
      end interface

      ! ~~~~~~~~
      integer(c_long_long), intent(in)                   :: input_mat_ptr
      integer(c_int), value, intent(in)                  :: inverse_type, poly_order, poly_sparsity_order
      integer(c_int), value, intent(in)                  :: matrix_free_int, diag_scale_polys_int, subcomm_int
      type(c_ptr), intent(inout)                         :: coeffs_ptr
      integer(PFLARE_PETSCINT_C_KIND), intent(inout)      :: row_size, col_size
      integer(c_long_long), intent(inout)                :: inv_matrix_ptr

      type(tMat)  :: input_mat, inv_matrix
      logical     :: matrix_free, subcomm, diag_scale_polys
      PetscReal, dimension(:, :), contiguous, pointer :: coefficients
      type(c_ptr) :: c_buf
      real(PFLARE_PETSCREAL_C_KIND), pointer :: c_view(:,:)
      integer(PFLARE_PETSCINT_C_KIND) :: nr, nc
      ! ~~~~~~~~

      input_mat%v = input_mat_ptr
      ! inv_matrix_ptr could be passed in as null or as an existing matrix
      ! whose sparsity we want to reuse, so we have to pass that in too
      inv_matrix%v = inv_matrix_ptr

      matrix_free    = .FALSE.
      diag_scale_polys = .FALSE.
      subcomm        = .FALSE.
      if (matrix_free_int    == 1) matrix_free    = .TRUE.
      if (diag_scale_polys_int == 1) diag_scale_polys = .TRUE.
      if (subcomm_int        == 1) subcomm        = .TRUE.

      if (c_associated(coeffs_ptr)) then

         ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         ! Reuse path: wrap the caller's buffer in a Fortran pointer and pass it in.
         ! calculate_and_build_approximate_inverse uses the null-mat trick to skip
         ! polynomial computation when coefficients is already associated on entry.
         ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         call c_f_pointer(coeffs_ptr, coefficients, [int(row_size), int(col_size)])

      else

         ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         ! Fresh path: nullify so calculate_and_build_approximate_inverse allocates
         ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         nullify(coefficients)

      end if

      call calculate_and_build_approximate_inverse(input_mat, inverse_type, &
               poly_order, poly_sparsity_order, &
               matrix_free, diag_scale_polys, subcomm, &
               inv_matrix, coefficients)

      if (.NOT. c_associated(coeffs_ptr)) then
         ! Fresh path: Fortran allocate may use a compiler-specific allocator
         ! (e.g. _mm_malloc on Intel) that is incompatible with C free().
         ! Copy the data into a C-malloc'd buffer so the C side can safely free() it.
         nr = int(size(coefficients, 1), PFLARE_PETSCINT_C_KIND)
         nc = int(size(coefficients, 2), PFLARE_PETSCINT_C_KIND)
         c_buf = c_malloc(int(nr, c_size_t) * int(nc, c_size_t) * int(PFLARE_PETSCREAL_C_KIND, c_size_t))
         call c_f_pointer(c_buf, c_view, [int(nr), int(nc)])
         c_view = coefficients
         ! For non-matrix-free: the matshell does not exist, so the Fortran allocation
         ! is no longer needed once we have the C copy.
         ! For matrix-free: the matshell owns its Fortran allocation (own_coefficients=.TRUE.)
         ! and will deallocate it independently via reset_inverse_mat. The C copy is
         ! stored separately in poly_coeffs and freed via free() in PCReset_PFLAREINV_c.
         if (.NOT. matrix_free) deallocate(coefficients)
         coeffs_ptr = c_buf
         row_size   = nr
         col_size   = nc
      end if

      ! Pass out the new inverse matrix
      inv_matrix_ptr = inv_matrix%v

   end subroutine calculate_and_build_approximate_inverse_c

   !------------------------------------------------------------------------------------------------------------------------

   subroutine reset_inverse_mat_c(mat_ptr) bind(C,name='reset_inverse_mat_c')

      ! Calls the Fortran routine

      ! ~~~~~~~~
      integer(c_long_long), intent(inout) :: mat_ptr

      type(tMat)  :: mat
      ! ~~~~~~~~

      mat%v = mat_ptr
      call reset_inverse_mat(mat)
      mat_ptr = mat%v

   end subroutine reset_inverse_mat_c    
   
   !------------------------------------------------------------------------------------------------------------------------

   subroutine compute_cf_splitting_c(input_mat_ptr, symmetric_int, &
         strong_threshold, max_luby_steps, &
         cf_splitting_type, ddc_its, fraction_swap, max_dd_ratio, &
         is_fine_ptr, is_coarse_ptr) &
         bind(C,name='compute_cf_splitting_c')

      ! Computes a CF splitting

      ! ~~~~~~~~
      integer(c_long_long), intent(in)       :: input_mat_ptr
      integer(c_int), value, intent(in)      :: symmetric_int, max_luby_steps, cf_splitting_type, ddc_its
      real(c_double), value, intent(in)      :: strong_threshold, fraction_swap, max_dd_ratio
      integer(c_long_long), intent(inout)    :: is_fine_ptr, is_coarse_ptr

      type(tMat)  :: input_mat
      type(tIS)   :: is_fine, is_coarse
      logical     :: symmetric = .FALSE.
      ! ~~~~~~~~   
      
      ! Now the input mat long long just gets copied into input_mat%v
      ! This works as the PETSc types are essentially just wrapped around
      ! pointers stored in %v
      input_mat%v = input_mat_ptr  
      
      if (symmetric_int == 1) symmetric = .TRUE.
      call compute_cf_splitting(input_mat, symmetric, &
                        strong_threshold, max_luby_steps, &
                        cf_splitting_type, ddc_its, fraction_swap, max_dd_ratio, &
                        is_fine, is_coarse)

      ! Pass out the IS's
      is_fine_ptr = is_fine%v
      is_coarse_ptr = is_coarse%v

   end subroutine compute_cf_splitting_c

   !------------------------------------------------------------------------------------------------------------------------

   subroutine compute_diag_dom_submatrix_c(input_mat_ptr, max_dd_ratio, output_mat_ptr) &
         bind(C,name='compute_diag_dom_submatrix_c')

      ! Computes a diagonally dominant submatrix

      ! ~~~~~~~~
      integer(c_long_long), intent(in)       :: input_mat_ptr
      real(c_double), value, intent(in)      :: max_dd_ratio
      integer(c_long_long), intent(inout)    :: output_mat_ptr

      type(tMat)  :: input_mat, output_mat
      ! ~~~~~~~~

      ! Copy the input matrix pointer into the Fortran PETSc handle wrapper
      input_mat%v = input_mat_ptr

      call compute_diag_dom_submatrix(input_mat, max_dd_ratio, output_mat)

      ! Pass out the resulting submatrix handle
      output_mat_ptr = output_mat%v

   end subroutine compute_diag_dom_submatrix_c

   !------------------------------------------------------------------------------------------------------------------------

end module c_fortran_bindings

