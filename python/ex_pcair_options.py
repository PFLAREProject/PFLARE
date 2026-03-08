'''
Tests the direct Python API for PCAIR get/set option functions introduced in
pflare.py (backed by PCAIR_C_Fortran_Bindings.F90).

Two checks are performed:
  1. Round-trip: set a value via the direct API, get it back, verify it matches.
  2. Functional: configure lAIR with WJacobi smoothing via the direct API,
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
# Set up a KSP with PCAIR configured entirely via the direct Python API
# -----------------------------------------------------------------------
ksp = PETSc.KSP().create(comm=comm)
ksp.setOperators(A, A)

pc = ksp.getPC()
pc.setType('air')

# Configure lAIR with WJacobi smoother via the direct Python API
pflare.pcair_set_z_type(pc, pflare.AIR_Z_LAIR)
pflare.pcair_set_inverse_type(pc, pflare.PFLAREINV_WJACOBI)
pflare.pcair_set_smooth_type(pc, 'fcf')
pflare.pcair_set_poly_order(pc, 4)
pflare.pcair_set_strong_threshold(pc, 0.25)

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
check('z_type',          pflare.pcair_get_z_type(pc),          pflare.AIR_Z_LAIR)
check('inverse_type',    pflare.pcair_get_inverse_type(pc),    pflare.PFLAREINV_WJACOBI)
check('smooth_type',     pflare.pcair_get_smooth_type(pc),     'fcf')
check('poly_order',      pflare.pcair_get_poly_order(pc),      4)
check('strong_threshold',pflare.pcair_get_strong_threshold(pc),0.25)

# Round-trip a selection of other options
pflare.pcair_set_max_levels(pc, 10)
check('max_levels',      pflare.pcair_get_max_levels(pc),      10)

pflare.pcair_set_coarse_eq_limit(pc, 12)
check('coarse_eq_limit', pflare.pcair_get_coarse_eq_limit(pc), 12)

pflare.pcair_set_ddc_its(pc, 3)
check('ddc_its',         pflare.pcair_get_ddc_its(pc),         3)

pflare.pcair_set_ddc_fraction(pc, 0.2)
check('ddc_fraction',    pflare.pcair_get_ddc_fraction(pc),    0.2)

pflare.pcair_set_r_drop(pc, 0.05)
check('r_drop',          pflare.pcair_get_r_drop(pc),          0.05)

pflare.pcair_set_a_drop(pc, 0.001)
check('a_drop',          pflare.pcair_get_a_drop(pc),          0.001)

pflare.pcair_set_lair_distance(pc, 1)
check('lair_distance',   pflare.pcair_get_lair_distance(pc),   1)

pflare.pcair_set_coarsest_inverse_type(pc, pflare.PFLAREINV_ARNOLDI)
check('coarsest_inverse_type', pflare.pcair_get_coarsest_inverse_type(pc), pflare.PFLAREINV_ARNOLDI)

pflare.pcair_set_coarsest_poly_order(pc, 8)
check('coarsest_poly_order', pflare.pcair_get_coarsest_poly_order(pc), 8)

pflare.pcair_set_matrix_free_polys(pc, True)
check('matrix_free_polys', pflare.pcair_get_matrix_free_polys(pc), True)

pflare.pcair_set_matrix_free_polys(pc, False)
check('matrix_free_polys_false', pflare.pcair_get_matrix_free_polys(pc), False)

pflare.pcair_set_reuse_sparsity(pc, True)
check('reuse_sparsity',  pflare.pcair_get_reuse_sparsity(pc),  True)

pflare.pcair_set_reuse_sparsity(pc, False)
check('reuse_sparsity_false', pflare.pcair_get_reuse_sparsity(pc), False)

pflare.pcair_set_reuse_amount(pc, 1)
check('reuse_amount_1',  pflare.pcair_get_reuse_amount(pc),  1)

pflare.pcair_set_reuse_amount(pc, 2)
check('reuse_amount_2',  pflare.pcair_get_reuse_amount(pc),  2)

pflare.pcair_set_reuse_amount(pc, 3)
check('reuse_amount_3',  pflare.pcair_get_reuse_amount(pc),  3)

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
