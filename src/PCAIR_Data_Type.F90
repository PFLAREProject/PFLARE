module pcair_data_type

   use petscksp
   use air_data_type

#include "petsc/finclude/petscksp.h"

   implicit none

   public

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~       
   ! This is our context type for pcshells
   ! This has to be in a separate file to pcshell
   
   type :: pc_air_multigrid_data

      type(air_multigrid_data) :: air_data
      ! AIR under the hood is just a PCMG object 
      type(tPC) :: pcmg

   end type pc_air_multigrid_data
   
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     

end module pcair_data_type

