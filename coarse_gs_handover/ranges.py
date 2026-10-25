import sys
from petsc4py import PETSc
A = PETSc.Mat().load(PETSc.Viewer().createBinary(sys.argv[1], 'r'))
r0, r1 = A.getOwnershipRange()
PETSc.Sys.syncPrint(f"{r0} {r1}"); PETSc.Sys.syncFlush()
