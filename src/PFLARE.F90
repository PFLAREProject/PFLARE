! ~~~~~~~~~~~~~~~~~
! PFLARE - Steven Dargaville
! Please see README.md for build/use details 
! ~~~~~~~~~~~~~~~~~

module pflare

   use air_mg_setup
   use pcair_shell
   use pcpflareinv_interfaces
   use pcair_interfaces

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
   
   contains  

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine PCRegister_PFLARE() bind(C,name='PCRegister_PFLARE')
      
      ! This routine should be called in external program
      ! This registers the PC types to PETSc
      ! Users can then just call PCSetType as normal 
      ! with the different types offered by this library            
 
      ! Register the PC AIR as a new type
      call PCRegister_AIR()
      ! Register the PC approximate inverses as a new type
      call PCRegister_PFLAREINV()      

   end subroutine PCRegister_PFLARE 

! -------------------------------------------------------------------------------------------------------------------------------

end module pflare