module cf_splitting

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! The Fortran interface to the CF splittings, which are implemented in C in
   ! src/CF_Splitting.c (see docs/dev/fortran_removal_plan.md)
   ! compute_cf_splitting keeps its Fortran signature, including the optional cr_*
   ! arguments, and calls compute_cf_splitting_c, which takes the full argument list
   ! (the optional arguments are pointers, null when absent)
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------

   use iso_c_binding
   use petscmat

#include "petsc/finclude/petscmat.h"

   implicit none
   public

   ! The CFSplittingType values in include/pflare.h
   PetscEnum, parameter :: CF_PMISR_DDC=0
   PetscEnum, parameter :: CF_DIAG_DOM=1
   PetscEnum, parameter :: CF_PMIS=2
   PetscEnum, parameter :: CF_PMIS_DIST2=3
   PetscEnum, parameter :: CF_AGG=4
   PetscEnum, parameter :: CF_PMIS_AGG=5
   PetscEnum, parameter :: CF_CR=6

   ! The C is called compute_cf_splitting_c, as compute_cf_splitting is the public
   ! C API in include/pflare.h
   interface
      function compute_cf_splitting_c(input_mat, skip_symmetrize, &
                  strong_threshold, max_luby_steps, &
                  cf_splitting_type, ddc_its, fraction_swap, &
                  is_fine, is_coarse, &
                  cr_inverse_type, cr_poly_order, cr_inverse_sparsity_order, &
                  cr_diag_scale_polys) &
         bind(c, name="compute_cf_splitting_c")
         use iso_c_binding
         integer(c_long_long), value :: input_mat
         PetscBool, value :: skip_symmetrize
         PetscReal, value :: strong_threshold, fraction_swap
         integer(c_int), value :: max_luby_steps, cf_splitting_type, ddc_its
         integer(c_long_long) :: is_fine, is_coarse
         type(c_ptr), value :: cr_inverse_type, cr_poly_order, cr_inverse_sparsity_order, cr_diag_scale_polys
         PetscErrorCode :: compute_cf_splitting_c
      end function compute_cf_splitting_c
   end interface

   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine compute_cf_splitting(input_mat, skip_symmetrize, &
                     strong_threshold, max_luby_steps, &
                     cf_splitting_type, ddc_its, fraction_swap, &
                     is_fine, is_coarse, &
                     cr_inverse_type, cr_poly_order, cr_inverse_sparsity_order, &
                     cr_diag_scale_polys)

      ! Computes a CF splitting and returns the F and C point ISs
      ! skip_symmetrize skips symmetrizing the strength matrix (it does not
      ! apply to the PMIS-based splittings, which are defined on the
      ! symmetrized graph and always symmetrize)
      ! The optional cr_* arguments only apply to CF_CR and set the
      ! approximate inverse used as the CR relaxation - PCAIR passes its
      ! Aff inverse settings so the CR rate certifies the F-solve actually
      ! applied in the hierarchy; when absent an assembled arnoldi-basis
      ! GMRES polynomial of order PFLARE_CR_POLY_ORDER with sparsity order 1
      ! and no diagonal scaling is used, matching the PCAIR defaults

      ! ~~~~~~
      type(tMat), target, intent(in)      :: input_mat
      logical, intent(in)                 :: skip_symmetrize
      PetscReal, intent(in)               :: strong_threshold
      integer, intent(in)                 :: max_luby_steps, cf_splitting_type, ddc_its
      PetscReal, intent(in)               :: fraction_swap
      type(tIS), intent(inout)            :: is_fine, is_coarse
      integer, intent(in), optional       :: cr_inverse_type, cr_poly_order
      integer, intent(in), optional       :: cr_inverse_sparsity_order
      logical, intent(in), optional       :: cr_diag_scale_polys

      PetscErrorCode :: ierr
      integer :: errorcode
      PetscBool :: skip_symmetrize_c
      integer(c_long_long) :: is_fine_handle, is_coarse_handle
      integer(c_int), target :: cr_inverse_type_c, cr_poly_order_c, cr_inverse_sparsity_order_c
      PetscBool, target :: cr_diag_scale_polys_c
      type(c_ptr) :: cr_inverse_type_ptr, cr_poly_order_ptr, cr_inverse_sparsity_order_ptr, &
            cr_diag_scale_polys_ptr
      ! ~~~~~~

      cr_inverse_type_ptr = c_null_ptr
      cr_poly_order_ptr = c_null_ptr
      cr_inverse_sparsity_order_ptr = c_null_ptr
      cr_diag_scale_polys_ptr = c_null_ptr
      if (present(cr_inverse_type)) then
         cr_inverse_type_c = cr_inverse_type
         cr_inverse_type_ptr = c_loc(cr_inverse_type_c)
      end if
      if (present(cr_poly_order)) then
         cr_poly_order_c = cr_poly_order
         cr_poly_order_ptr = c_loc(cr_poly_order_c)
      end if
      if (present(cr_inverse_sparsity_order)) then
         cr_inverse_sparsity_order_c = cr_inverse_sparsity_order
         cr_inverse_sparsity_order_ptr = c_loc(cr_inverse_sparsity_order_c)
      end if
      if (present(cr_diag_scale_polys)) then
         cr_diag_scale_polys_c = cr_diag_scale_polys
         cr_diag_scale_polys_ptr = c_loc(cr_diag_scale_polys_c)
      end if

      skip_symmetrize_c = skip_symmetrize
      ierr = compute_cf_splitting_c(input_mat%v, skip_symmetrize_c, &
                  strong_threshold, max_luby_steps, &
                  cf_splitting_type, ddc_its, fraction_swap, &
                  is_fine_handle, is_coarse_handle, &
                  cr_inverse_type_ptr, cr_poly_order_ptr, cr_inverse_sparsity_order_ptr, &
                  cr_diag_scale_polys_ptr)
      ! The C returns a PETSc error code, and PETSc has already printed the error
      ! and traceback if it is not zero, but there is no error code to pass it back through
      if (ierr /= 0) then
         print *, "compute_cf_splitting failed"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      is_fine%v = is_fine_handle
      is_coarse%v = is_coarse_handle

   end subroutine compute_cf_splitting

! -------------------------------------------------------------------------------------------------------------------------------

end module cf_splitting
