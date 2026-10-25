import sys, numpy as np
from petsc4py import PETSc
import scipy.sparse as sp
comm = PETSc.COMM_WORLD
v = PETSc.Viewer().createBinary(sys.argv[1], 'r')
A = PETSc.Mat().load(v)
r0, r1 = A.getOwnershipRange()
Ad = A.getDiagonalBlock()
ia, ja, a = Ad.getValuesCSR()
L = sp.csr_matrix((a, ja, ia), shape=(r1-r0, r1-r0))
up = sp.triu(L, 1); lo = sp.tril(L, -1)
ia2, ja2, a2 = A.getValuesCSR()
G = sp.csr_matrix((a2, ja2, ia2), shape=(r1-r0, A.getSize()[1]))
offnorm = np.sqrt(max((G.multiply(G)).sum() - (L.multiply(L)).sum(), 0))
msg = f"rank {comm.rank}: n_local={r1-r0} nnz_diagblk={L.nnz} upper_nnz={up.nnz} |U|_F/|A_d|_F={sp.linalg.norm(up)/sp.linalg.norm(L):.2e} max|U|/max|A|={abs(up).max() if up.nnz else 0:.2e} |A_o|_F/|A|_F={offnorm/np.sqrt((G.multiply(G)).sum()):.2e}"
PETSc.Sys.syncPrint(msg); PETSc.Sys.syncFlush()
