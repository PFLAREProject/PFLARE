module gmres_poly_data_type

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! The gmres polynomial data is the C struct gmres_poly_data in src/pflareimpl.h
   ! (see docs/dev/fortran_removal_plan.md). The Fortran reference of the AIR hierarchy
   ! in src/fref (PFLARE_PORT_DEBUG only) holds these, so the Fortran type here is a
   ! bind(C) mirror of the C struct: the same memory is read and written by both languages
   ! The two definitions must stay in step - the same fields, in the same order,
   ! with the same types
   ! The coefficients are column-major (coefficients_rows x coefficients_cols),
   ! allocated by the C with PetscMalloc; poly_coeffs gives a Fortran pointer to them
   ! and poly_coeffs_allocated replaces associated(%coefficients)
   ! The C frees the coefficients (reset_poly_data) and the coefficients plus the
   ! TSQR buffers (destroy_poly_data), where the Fortran used to deallocate them
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------

   use iso_c_binding
   use petscsys

#include "petsc/finclude/petscsys.h"

   implicit none

   public

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   ! Stores the data needed for gmres polynomials
   ! Must stay in step with gmres_poly_data in src/pflareimpl.h
   type, bind(C) :: gmres_poly_data

      ! Coefficients for the gmres polynomial (a PetscReal array, column-major)
      ! If using the newton basis this has two columns with the real
      ! and imaginary roots
      type(c_ptr) :: coefficients = c_null_ptr
      PetscInt :: coefficients_rows = 0, coefficients_cols = 0

      ! Inverse type
      integer(c_int) :: inverse_type = 0

      ! Order of the gmres polynomial
      integer(c_int) :: gmres_poly_order = 6
      ! The sparsity order of the polynomial if assembled
      integer(c_int) :: gmres_poly_sparsity_order = 1

      ! Asynchronous buffers for the TSQR (a C tsqr_buffers, created on first use)
      type(c_ptr) :: buffers = c_null_ptr

   end type gmres_poly_data

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   interface
      function reset_poly_data_c(poly_data) &
         bind(c, name="reset_poly_data")
         use iso_c_binding
         import :: gmres_poly_data
         type(gmres_poly_data) :: poly_data
         PetscErrorCode :: reset_poly_data_c
      end function reset_poly_data_c
   end interface

   interface
      function destroy_poly_data_c(poly_data) &
         bind(c, name="destroy_poly_data")
         use iso_c_binding
         import :: gmres_poly_data
         type(gmres_poly_data) :: poly_data
         PetscErrorCode :: destroy_poly_data_c
      end function destroy_poly_data_c
   end interface

   contains

! -------------------------------------------------------------------------------------------------------------------------------

   function poly_coeffs(poly_data) result(coeffs)

      ! Returns a Fortran pointer to the coefficients (null if they aren't allocated)

      ! ~~~~~~
      type(gmres_poly_data), intent(in)              :: poly_data
      PetscReal, dimension(:, :), pointer, contiguous :: coeffs
      ! ~~~~~~

      coeffs => null()
      if (c_associated(poly_data%coefficients)) then
         call c_f_pointer(poly_data%coefficients, coeffs, &
                  [int(poly_data%coefficients_rows), int(poly_data%coefficients_cols)])
      end if

   end function poly_coeffs

! -------------------------------------------------------------------------------------------------------------------------------

   logical function poly_coeffs_allocated(poly_data)

      ! Replaces associated(poly_data%coefficients)

      ! ~~~~~~
      type(gmres_poly_data), intent(in) :: poly_data
      ! ~~~~~~

      poly_coeffs_allocated = c_associated(poly_data%coefficients)

   end function poly_coeffs_allocated

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine reset_poly_data(poly_data)

      ! Frees the coefficients, where the Fortran used to deallocate them

      ! ~~~~~~
      type(gmres_poly_data), intent(inout) :: poly_data

      PetscErrorCode :: ierr
      integer :: errorcode
      ! ~~~~~~

      ierr = reset_poly_data_c(poly_data)
      if (ierr /= 0) then
         print *, "reset_poly_data failed"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

   end subroutine reset_poly_data

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine destroy_poly_data(poly_data)

      ! Frees the coefficients and the TSQR buffers, which the Fortran type
      ! deallocated automatically

      ! ~~~~~~
      type(gmres_poly_data), intent(inout) :: poly_data

      PetscErrorCode :: ierr
      integer :: errorcode
      ! ~~~~~~

      ierr = destroy_poly_data_c(poly_data)
      if (ierr /= 0) then
         print *, "destroy_poly_data failed"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

   end subroutine destroy_poly_data

! -------------------------------------------------------------------------------------------------------------------------------

end module gmres_poly_data_type
