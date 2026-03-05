'''
Tests the direct Python API for PCPFLAREINV get/set option functions introduced in
pflare.py (backed by PCPFLAREINV_Interfaces.F90).

Two checks are performed:
  1. Round-trip: set a value via the direct API, get it back, verify it matches.
  2. Functional: configure a Newton matrix-free PCPFLAREINV via the direct API,
     run a solve, verify convergence.
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

u = PETSc.Vec().createSeq(m * n, comm=comm) if comm.getSize() == 1 else PETSc.Vec().create(comm=comm)
if comm.getSize() > 1:
    u.setSizes(m * n)
    u.setFromOptions()
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
