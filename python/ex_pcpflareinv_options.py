'''
Tests the direct Python API for PCPFLAREINV get/set option functions introduced in
pflare.py (backed by PCPFLAREINV_Interfaces.F90).

Checks performed:
  1. Round-trip: set a value via the direct API, get it back, verify it matches.
  2. Functional: configure a Newton matrix-free PCPFLAREINV via the direct API,
     run a solve, verify convergence.
  3. PCPFLAREINVGetInverseMat: the accessor returns None before PCSetUp; after
     setup it returns the underlying approximate-inverse matrix (assembled aij
     or a matrix-free shell) which the user can operate on directly, and which
     is a refcounted borrowed reference (deleting it must not free the PC's
     matrix).
'''

import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

import pflare

comm = PETSc.COMM_WORLD
rank = comm.getRank()

# -----------------------------------------------------------------------
# Build a small 2-D five-point Laplacian
# -----------------------------------------------------------------------
m, n = 8, 8

A = PETSc.Mat().create(comm=comm)
A.setSizes((m * n, m * n))
A.setFromOptions()
A.setUp()

Istart, Iend = A.getOwnershipRange()
for II in range(Istart, Iend):
    i = II // n
    j = II - i * n
    if i > 0:
        A.setValues(II, II - n, -1.0, addv=True)
    if i < m - 1:
        A.setValues(II, II + n, -1.0, addv=True)
    if j > 0:
        A.setValues(II, II - 1, -1.0, addv=True)
    if j < n - 1:
        A.setValues(II, II + 1, -1.0, addv=True)
    A.setValues(II, II, 4.0, addv=True)

A.assemblyBegin(A.AssemblyType.FINAL)
A.assemblyEnd(A.AssemblyType.FINAL)

# Build the vectors from the matrix so they match its type (e.g. kokkos when
# -vec_type kokkos is set); creating them independently can mismatch the matrix.
u = A.createVecs('right')
b = u.duplicate()
x = u.duplicate()

# -----------------------------------------------------------------------
# Set up a KSP with PCPFLAREINV configured entirely via the direct Python API
# -----------------------------------------------------------------------
ksp = PETSc.KSP().create(comm=comm)
ksp.setOperators(A, A)

pc = ksp.getPC()
pc.setType('pflareinv')

# Configure a Newton matrix-free polynomial via the direct Python API
pflare.pcpflareinv_set_type(pc, pflare.PFLAREINV_NEWTON)
pflare.pcpflareinv_set_poly_order(pc, 10)
pflare.pcpflareinv_set_matrix_free(pc, True)

ksp.setFromOptions()
ksp.setTolerances(rtol=1e-8, max_it=200)

# -----------------------------------------------------------------------
# Solve and check convergence
# -----------------------------------------------------------------------
u.set(1.0)
A.mult(u, b)
ksp.solve(b, x)

reason = ksp.getConvergedReason()
if reason <= 0:
    if rank == 0:
        print(f'FAIL: KSP did not converge (reason {reason})')
    sys.exit(1)

# -----------------------------------------------------------------------
# Round-trip checks: set a value, get it back, verify they match
# -----------------------------------------------------------------------
errors = []

def check(name, got, expected):
    if got != expected:
        errors.append(f'{name}: expected {expected!r}, got {got!r}')

# Verify settings from the solve above were actually applied
check('type',        pflare.pcpflareinv_get_type(pc),        pflare.PFLAREINV_NEWTON)
check('poly_order',  pflare.pcpflareinv_get_poly_order(pc),  10)
check('matrix_free', pflare.pcpflareinv_get_matrix_free(pc), True)

# Round-trip a selection of other options
pflare.pcpflareinv_set_type(pc, pflare.PFLAREINV_ARNOLDI)
check('type_arnoldi', pflare.pcpflareinv_get_type(pc), pflare.PFLAREINV_ARNOLDI)

pflare.pcpflareinv_set_type(pc, pflare.PFLAREINV_POWER)
check('type_power', pflare.pcpflareinv_get_type(pc), pflare.PFLAREINV_POWER)

pflare.pcpflareinv_set_poly_order(pc, 4)
check('poly_order_4', pflare.pcpflareinv_get_poly_order(pc), 4)

pflare.pcpflareinv_set_sparsity_order(pc, 2)
check('sparsity_order', pflare.pcpflareinv_get_sparsity_order(pc), 2)

pflare.pcpflareinv_set_matrix_free(pc, False)
check('matrix_free_false', pflare.pcpflareinv_get_matrix_free(pc), False)

pflare.pcpflareinv_set_matrix_free(pc, True)
check('matrix_free_true', pflare.pcpflareinv_get_matrix_free(pc), True)

pflare.pcpflareinv_set_reuse_poly_coeffs(pc, True)
check('reuse_poly_coeffs', pflare.pcpflareinv_get_reuse_poly_coeffs(pc), True)

pflare.pcpflareinv_set_reuse_poly_coeffs(pc, False)
check('reuse_poly_coeffs_false', pflare.pcpflareinv_get_reuse_poly_coeffs(pc), False)

# -----------------------------------------------------------------------
# PCPFLAREINVGetInverseMat: access the underlying approximate-inverse matrix
# -----------------------------------------------------------------------

# Assembled inverse on a fresh PC
pc_inv = PETSc.PC().create(comm=comm)
pc_inv.setType('pflareinv')
pc_inv.setOperators(A, A)
pflare.pcpflareinv_set_type(pc_inv, pflare.PFLAREINV_POWER)
pflare.pcpflareinv_set_matrix_free(pc_inv, False)

# Before PCSetUp the inverse matrix does not exist yet
check('inverse_mat_pre_setup_none', pflare.pcpflareinv_get_inverse_mat(pc_inv), None)

pc_inv.setUp()
M = pflare.pcpflareinv_get_inverse_mat(pc_inv)
check('inverse_mat_not_none', M is not None, True)
if M is not None:
    check('inverse_mat_is_aij', 'aij' in M.getType().lower(), True)
    check('inverse_mat_size', M.getSize(), A.getSize())

    # Use the returned matrix directly (Schur-complement-style matrix product)
    M2 = M.matMult(M)
    check('inverse_mat_matmult_size', M2.getSize(), A.getSize())
    M2.destroy()
pc_inv.destroy()

# Matrix-free inverse returns a usable shell
pc_inv_mf = PETSc.PC().create(comm=comm)
pc_inv_mf.setType('pflareinv')
pc_inv_mf.setOperators(A, A)
pflare.pcpflareinv_set_type(pc_inv_mf, pflare.PFLAREINV_POWER)
pflare.pcpflareinv_set_matrix_free(pc_inv_mf, True)
pc_inv_mf.setUp()
Mmf = pflare.pcpflareinv_get_inverse_mat(pc_inv_mf)
check('inverse_mat_mf_not_none', Mmf is not None, True)
if Mmf is not None:
    check('inverse_mat_mf_is_shell', Mmf.getType(), 'shell')
    xf, yf = A.createVecs()
    xf.set(1.0)
    Mmf.mult(xf, yf)           # shell MatMult must work
    check('inverse_mat_mf_matmult', yf.norm() > 0.0, True)
    xf.destroy()
    yf.destroy()
pc_inv_mf.destroy()

if errors:
    if rank == 0:
        for e in errors:
            print(f'FAIL: {e}')
    sys.exit(1)

# Tidy up
u.destroy()
b.destroy()
x.destroy()
A.destroy()
ksp.destroy()
