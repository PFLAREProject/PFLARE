module matshell_data_type

   use petscmat
   use air_data_type
   use pflare_parameters, only: MF_VEC_TEMP, MF_VEC_TEMP_TWO, MF_VEC_TEMP_THREE, &
         MF_VEC_DIAG, MF_VEC_RHS

#include "petsc/finclude/petscmat.h"   

   implicit none

   public

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ! This is our context type for matshells
   ! This has to be in a separate file to matshell

   type :: mat_ctxtype
      integer :: our_level = -1
      PetscReal, dimension(:), pointer :: coefficients => null()
      logical                     :: own_coefficients = .FALSE.
      PetscReal, dimension(:), pointer :: real_roots => null()
      PetscReal, dimension(:), pointer :: imag_roots => null()
      type(tMat) :: mat, mat_scaled
      ! Temporary vectors we use
      type(tVec), dimension(5) :: mf_temp_vec
      type(air_multigrid_data), pointer :: air_data => null()

   end type mat_ctxtype
   
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
   
   contains

end module matshell_data_type

