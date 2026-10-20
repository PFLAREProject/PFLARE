module matdiagdomsubmatrix

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! The Fortran interface to the diagonally dominant submatrix extraction, which is
   ! implemented in C in src/MatDiagDomSubmatrix.c (see docs/dev/fortran_removal_plan.md)
   ! compute_diag_dom_submatrix keeps its Fortran signature and calls
   ! compute_diag_dom_submatrix_c
   ! With PFLARE_PORT_DEBUG it also runs the Fortran reference implementation in
   ! src/fref/MatDiagDomSubmatrix.F90 on the same input and aborts if the matrices differ
   ! -------------------------------------------------------------------------------------------------------------------------------
   ! -------------------------------------------------------------------------------------------------------------------------------

   use iso_c_binding
   use petscmat
   ! The module this replaces also made these two available
   use cf_splitting, only: compute_cf_splitting, CF_DIAG_DOM
   ! The Fortran reference implementation, under #ifdef rather than #if defined
   ! so that make depend (makedepf90 -D PFLARE_PORT_DEBUG) sees these uses only
   ! in the port debug build
#ifdef PFLARE_PORT_DEBUG
   use matdiagdomsubmatrix_fref, only: compute_diag_dom_submatrix_fref => compute_diag_dom_submatrix
   use port_interfaces, only: port_compare_mats
#endif

#include "petsc/finclude/petscmat.h"

   implicit none
   public

   ! The C is called compute_diag_dom_submatrix_c, as compute_diag_dom_submatrix is
   ! the public C API in include/pflare.h
   interface
      function compute_diag_dom_submatrix_c(input_mat, max_dd_ratio, output_mat) &
         bind(c, name="compute_diag_dom_submatrix_c")
         use iso_c_binding
         integer(c_long_long), value :: input_mat
         PetscReal, value :: max_dd_ratio
         integer(c_long_long) :: output_mat
         PetscErrorCode :: compute_diag_dom_submatrix_c
      end function compute_diag_dom_submatrix_c
   end interface

   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine compute_diag_dom_submatrix(input_mat, max_dd_ratio, output_mat)

      ! Returns a diagonally dominant submatrix taken from input_mat where every row's
      ! diagonal dominance ratio is < max_dd_ratio
      ! Not guaranteed to be the optimal (ie largest) submatrix but should be close
      ! This works for symmetric and asymmetric input_mat
      ! Works in serial, parallel and kokkos (and hence gpus)
      ! This is just a convenience wrapper around compute_cf_splitting and matcreatesubmatrixwrapper

      ! ~~~~~~
      type(tMat), target, intent(in)      :: input_mat
      PetscReal, intent(in)               :: max_dd_ratio
      type(tMat), intent(inout)           :: output_mat

      PetscErrorCode :: ierr
      integer :: errorcode
      integer(c_long_long) :: output_mat_handle
#if defined(PFLARE_PORT_DEBUG)
      type(tMat) :: output_mat_fref
#endif
      ! ~~~~~~

      ierr = compute_diag_dom_submatrix_c(input_mat%v, max_dd_ratio, output_mat_handle)
      ! The C returns a PETSc error code, and PETSc has already printed the error
      ! and traceback if it is not zero, but there is no error code to pass it back through
      if (ierr /= 0) then
         print *, "compute_diag_dom_submatrix failed"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      output_mat%v = output_mat_handle

#if defined(PFLARE_PORT_DEBUG)
      call compute_diag_dom_submatrix_fref(input_mat, max_dd_ratio, output_mat_fref)
      call port_compare_mats(output_mat, output_mat_fref, "compute_diag_dom_submatrix")
      call MatDestroy(output_mat_fref, ierr)
#endif

   end subroutine compute_diag_dom_submatrix

! -------------------------------------------------------------------------------------------------------------------------------

end module matdiagdomsubmatrix
