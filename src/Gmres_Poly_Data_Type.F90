module gmres_poly_data_type

   use tsqr
   use petscsys

#include "petsc/finclude/petscsys.h"   
   
   implicit none
   
   public

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
   
   ! Stores the data needed for gmres polynomials
   type gmres_poly_data

      ! Order of the gmres polynomial
      integer :: gmres_poly_order = 6
      ! The sparsity order of the polynomial if assembled
      integer :: gmres_poly_sparsity_order = 1
      
      ! Coefficients for the gmres polynomial
      ! If using the newton basis this has two columns with the real 
      ! and imaginary roots
      PetscReal, pointer, dimension(:, :), contiguous :: coefficients => null()

      ! Asynchronous buffers for the TSQR
      type(tsqr_buffers) :: buffers

   end type gmres_poly_data  
   
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~       
   
   
end module gmres_poly_data_type
