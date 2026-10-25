# Compare coarse-grid polynomial solvers on a dumped AIR coarse matrix.
# Usage: mpirun -n P python3 coarse_poly.py c.dat k1,k2,...
import sys, numpy as np
from petsc4py import PETSc
import scipy.sparse as sp, scipy.sparse.linalg as spla, scipy.linalg as la
A = PETSc.Mat().load(PETSc.Viewer(PETSc.COMM_SELF).createBinary(sys.argv[1], 'r', comm=PETSc.COMM_SELF), )
ks = [int(s) for s in sys.argv[2].split(',')]
ranges = [tuple(int(t) for t in l.split()) for l in open(sys.argv[3]) if l.strip()]
ia, ja, a = A.getValuesCSR(); N = A.getSize()[0]
A = sp.csr_matrix((a, ja, ia), shape=(N, N))
P = len(ranges)
D = A.diagonal()
# Block-diagonal local forward GS: M = blockdiag(tril(A_dd))
blocks = [sp.tril(A[s:e, s:e]).tocsr() for (s, e) in ranges]
def Minv(v):
    out = np.empty_like(v)
    for (s, e), B in zip(ranges, blocks):
        out[s:e] = spla.spsolve_triangular(B, v[s:e], lower=True) if e > s else v[s:e]
    return out
from scipy.sparse.csgraph import connected_components
def topo_perm(B, drop):
    # Directed upwind graph: edge j->i if row i depends on j. Edges weaker than
    # drop*max|row| are ignored when ordering. Cycles are collapsed into strongly
    # connected components, which are then topologically sorted (Kahn) with
    # members kept contiguous.
    n = B.shape[0]
    C = B.tocoo(); m = (C.row != C.col)
    rmax = abs(B).max(axis=1).toarray().ravel()
    m &= abs(C.data) > drop*rmax[C.row]
    G = sp.csr_matrix((np.ones(m.sum()), (C.col[m], C.row[m])), shape=(n, n))  # j->i
    ncomp, lab = connected_components(G, directed=True, connection='strong')
    Gc = sp.csr_matrix((np.ones(G.nnz), (lab[G.tocoo().row], lab[G.tocoo().col])), shape=(ncomp, ncomp))
    Gc.setdiag(0); Gc.eliminate_zeros()
    indeg = np.diff(Gc.tocsc().indptr); order = []; stack = list(np.where(indeg == 0)[0])
    while stack:
        c = stack.pop(); order.append(c)
        for d in Gc.indices[Gc.indptr[c]:Gc.indptr[c+1]]:
            indeg[d] -= 1
            if indeg[d] == 0: stack.append(d)
    rank = np.empty(ncomp, int); rank[np.array(order)] = np.arange(ncomp)
    perm = np.lexsort((np.arange(n), rank[lab]))
    sizes = np.bincount(lab)
    return perm, ncomp, sizes.max()
def make_Minv(perms):
    bl = []
    for (s, e), pm in zip(ranges, perms):
        B = A[s:e, s:e][pm][:, pm]
        bl.append((sp.tril(B).tocsr(), pm))
    def f(v):
        out = np.empty_like(v)
        for (s, e), (B, pm) in zip(ranges, bl):
            if e > s:
                y = spla.spsolve_triangular(B, v[s:e][pm], lower=True)
                out[s:e][pm] = y
        return out
    return f, bl
def levels(B):
    # longest path (level-scheduling depth) of the strictly lower DAG
    n = B.shape[0]; lev = np.zeros(n, int); L = sp.tril(B, -1).tocsr()
    for i in range(n):
        c = L.indices[L.indptr[i]:L.indptr[i+1]]
        if len(c): lev[i] = lev[c].max() + 1
    return lev.max() + 1 if n else 0
def harmonic_ritz(op, v, k):
    n = len(v); V = np.zeros((n, k+1)); H = np.zeros((k+1, k))
    V[:, 0] = v/np.linalg.norm(v)
    for j in range(k):
        w = op(V[:, j])
        for i in range(j+1):
            H[i, j] = V[:, i] @ w; w -= H[i, j]*V[:, i]
        H[j+1, j] = np.linalg.norm(w)
        if H[j+1, j] < 1e-10*np.linalg.norm(H[:j+2, j]): k = j+1; break
        V[:, j+1] = w/H[j+1, j]
    Hk = H[:k, :k]; ek = np.zeros(k); ek[-1] = 1
    th = la.eigvals(Hk + H[k, k-1]**2*np.outer(np.linalg.solve(Hk.conj().T, ek), ek))
    # Leja order
    order = [np.argmax(abs(th))]; rest = list(set(range(len(th))) - set(order))
    while rest:
        j = max(rest, key=lambda r: np.prod([abs(th[r]-th[o]) for o in order])); order.append(j); rest.remove(j)
    th = th[order]
    # keep complex conjugate pairs adjacent
    out = []; used = set()
    for i, t in enumerate(th):
        if i in used: continue
        out.append(t); used.add(i)
        if abs(t.imag) > 1e-12*abs(t):
            j = min((jj for jj in range(len(th)) if jj not in used), key=lambda jj: abs(th[jj]-t.conj()))
            out.append(th[j]); used.add(j)
    return np.array(out)
def apply_poly(op, th, b):
    x = np.zeros(len(b), complex); r = b.astype(complex)
    for t in th:
        x = x + r/t; r = r - op(r)/t
    return x.real
rng = np.random.default_rng(0)
vbuild = rng.standard_normal(N); btest = rng.standard_normal(N)
Amv = lambda v: A @ v
tp = [topo_perm(A[s:e, s:e], 1e-8) for (s, e) in ranges]
Minv_topo, bl_topo = make_Minv([t[0] for t in tp])
print(f"topo: SCCs per rank {[t[1] for t in tp][:4]}..., max SCC size {max(t[2] for t in tp)}; upper-part frac after perm "
      f"{max(sp.linalg.norm(sp.triu(A[s:e,s:e][t[0]][:,t[0]],1))/sp.linalg.norm(A[s:e,s:e]) for (s,e),t in zip(ranges,tp)):.2e}"
      f" (natural {max(sp.linalg.norm(sp.triu(A[s:e,s:e],1))/sp.linalg.norm(A[s:e,s:e]) for (s,e) in ranges):.2e})")
cases = {
  'poly(A)':            (Amv,                         lambda x: x),
  'poly(AD^-1)D^-1':    (lambda v: A @ (v/D),         lambda x: x/D),
  'poly(AM^-1)M^-1':    (lambda v: A @ Minv(v),       Minv),
  'poly(AMt^-1)Mt^-1':  (lambda v: A @ Minv_topo(v),  Minv_topo),
}
def opcplx(f):
    return lambda v: f(v.real) + 1j*f(v.imag) if np.iscomplexobj(v) else f(v)
nb = np.linalg.norm(btest)
print(f"N={N} P={P} nnz={A.nnz} nnz/row={A.nnz/N:.1f} GS-levels natural={max(levels(B) for B in blocks)} topo={max(levels(B) for B,_ in bl_topo)}")
x = Minv(btest); print(f"  GS alone: rel res {np.linalg.norm(btest - A@x)/nb:.2e}")
for k in ks:
    line = f"  k={k:3d}"
    for name, (op, post) in cases.items():
        th = harmonic_ritz(op, vbuild, k)
        y = apply_poly(opcplx(op), th, btest); xx = post(y)
        line += f" | {name} {np.linalg.norm(btest - A@xx)/nb:.2e}"
    # Richardson with the (reordered) local GS: residual poly (1-z)^k in A M^-1
    xx = np.zeros(N)
    for _ in range(k): xx = xx + Minv_topo(btest - A@xx)
    line += f" | Rich(Mt) {np.linalg.norm(btest - A@xx)/nb:.2e}"
    # multiplicative: GS sweep then poly(A) on the residual
    th = harmonic_ritz(Amv, vbuild, k); x0 = Minv(btest); r = btest - A@x0
    xx = x0 + apply_poly(opcplx(Amv), th, r)
    line += f" | GS*poly(A) {np.linalg.norm(btest - A@xx)/nb:.2e}"
    print(line)
