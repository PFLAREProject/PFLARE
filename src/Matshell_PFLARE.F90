module matshell_pflare

   use petscmat
   use matshell_data_type

#include "petsc/finclude/petscmat.h"

   implicit none

   public

   ! You have to provide this to get the context type correct for PETSc
   Interface_MatShellGetContext(mat_ctxtype)

   contains

end module matshell_pflare
