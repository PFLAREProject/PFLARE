module matshell_pflare

   use petscmat
   use matshell_data_type

#include "petsc/finclude/petscmat.h"   

   implicit none

   public

   ! You have to provide this to get the context type correct for PETSc
   interface MatShellGetContext
      subroutine MatShellGetContext(mat,ctx,ierr)
         use petsc
         use matshell_data_type
         type(tMat) :: mat
         type(mat_ctxtype), pointer :: ctx
         PetscErrorCode :: ierr
      end subroutine MatShellGetContext  
   end interface MatShellGetContext

   interface MatCreateShell
      subroutine MatCreateShell(comm,mloc,nloc,m,n,ctx,mat,ierr)
         use petsc
         use matshell_data_type
         MPIU_Comm :: comm
         PetscInt :: mloc,nloc,m,n
         type(mat_ctxtype) :: ctx
         type(tMat) :: mat
         PetscErrorCode :: ierr
      end subroutine MatCreateShell  
    end interface MatCreateShell     
   
   contains

end module matshell_pflare
