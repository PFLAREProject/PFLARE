! ~~~~~~~~~~~~~~~~~
! PFLARE - Steven Dargaville
! Please see README.md for build/use details 
! ~~~~~~~~~~~~~~~~~

module pflare

   use pcpflareinv_interfaces
   use pcair_interfaces
   use pflare_parameters

   implicit none

   public

   ! ~~~~~~~~~~~~~~~~
   
   interface   
      
      subroutine PCRegister_AIR() &
         bind(c, name="PCRegister_AIR")
         use iso_c_binding
      end subroutine PCRegister_AIR         
 
   end interface
   
   interface   
      
      subroutine PCRegister_PFLAREINV() &
         bind(c, name="PCRegister_PFLAREINV")
         use iso_c_binding
      end subroutine PCRegister_PFLAREINV         
 
   end interface   

   interface   
      
      ! This routine should be called in external program
      ! This registers the PC types to PETSc
      ! Users can then just call PCSetType as normal 
      ! with the different types offered by this library
      ! (PCRegister_PFLARE in PCAIR.c)
      subroutine PCRegister_PFLARE() &
         bind(c, name="PCRegister_PFLARE")
         use iso_c_binding
      end subroutine PCRegister_PFLARE         
 
   end interface   

! -------------------------------------------------------------------------------------------------------------------------------

end module pflare