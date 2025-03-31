module matshell_data_type

   use petscmat
   use air_data_type

#include "petsc/finclude/petscmat.h"   

   implicit none

   public

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~       
   ! This is our context type for matshells
   ! This has to be in a separate file to matshell

   ! Indices into mf_temp_vec
   integer, parameter :: MF_VEC_TEMP = 1
   integer, parameter :: MF_VEC_DIAG = 2
   integer, parameter :: MF_VEC_RHS = 3
   
   type :: mat_ctxtype

      integer :: our_level = -1
      PetscReal, dimension(:), pointer :: coefficients => null()
      logical                     :: own_coefficients = .FALSE.
      PetscReal, dimension(:), pointer :: real_roots => null()
      PetscReal, dimension(:), pointer :: imag_roots => null()
      type(tMat) :: mat, mat_ida
      ! Temporary vectors we use
      type(tVec), dimension(3) :: mf_temp_vec
      type(air_multigrid_data), pointer :: air_data => null()

   end type mat_ctxtype
   
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
   
   contains

end module matshell_data_type

