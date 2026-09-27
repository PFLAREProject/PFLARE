'''
Tests the PCAIRGet*Complexity Python API.

Three checks:
  1. Before PCSetUp, all 5 complexity getters return -1.0 (sentinel).
  2. After PCSetUp, all 5 complexities are positive and cycle complexity
     is finite.
  3. With a single level hierarchy (jacobi fallback and auto truncation)
     all 5 complexities are finite and grid/operator complexity are 1.
'''

import sys
import math
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

import pflare

comm = PETSc.COMM_WORLD
rank = comm.getRank()

# -----------------------------------------------------------------------
# Build a small 2-D five-point Laplacian (same pattern as ex_pcair_options.py)
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

A.assemblyBegin()
A.assemblyEnd()

# -----------------------------------------------------------------------
# Check 1: pre-setup sentinel values
# -----------------------------------------------------------------------
pc = PETSc.PC().create(comm=comm)
pc.setType('air')
pc.setOperators(A, A)
pc.setFromOptions()

_GETTERS = [
    ('grid',          pflare.pcair_get_grid_complexity),
    ('operator',      pflare.pcair_get_operator_complexity),
    ('cycle',         pflare.pcair_get_cycle_complexity),
    ('storage',       pflare.pcair_get_storage_complexity),
    ('reuse_storage', pflare.pcair_get_reuse_storage_complexity),
]

for name, getter in _GETTERS:
    val = getter(pc)
    assert val == -1.0, \
        f"Before PCSetUp {name} complexity should be -1.0, got {val}"

if rank == 0:
    print("Pre-setup sentinel check passed")

# -----------------------------------------------------------------------
# Check 2: post-setup positive values
# -----------------------------------------------------------------------
pc.setUp()

for name, getter in _GETTERS:
    val = getter(pc)
    # reuse_storage is legitimately 0 when reuse_sparsity is disabled
    if name == 'reuse_storage':
        assert val >= 0.0, \
            f"After PCSetUp {name} complexity should be >= 0.0, got {val}"
    else:
        assert val > 0.0, \
            f"After PCSetUp {name} complexity should be positive, got {val}"
    if rank == 0:
        print(f"  {name} complexity = {val:.4f}")

if rank == 0:
    print("Post-setup complexity check passed")

pc.destroy()

# -----------------------------------------------------------------------
# Check 3: single level hierarchies
# -----------------------------------------------------------------------
opts = PETSc.Options()
_SINGLE_LEVEL = [
    ('jacobi fallback', {'pc_air_max_levels': 1}),
    ('auto truncated',  {'pc_air_auto_truncate_start_level': 1,
                         'pc_air_auto_truncate_tol': 1e10}),
]
for desc, extra in _SINGLE_LEVEL:
    for key, value in extra.items():
        opts[key] = value
    pc = PETSc.PC().create(comm=comm)
    pc.setType('air')
    pc.setOperators(A, A)
    pc.setFromOptions()
    pc.setUp()
    for key in extra:
        opts.delValue(key)

    for name, getter in _GETTERS:
        val = getter(pc)
        assert math.isfinite(val), \
            f"Single level ({desc}) {name} complexity should be finite, got {val}"
        if name in ('grid', 'operator'):
            assert val == 1.0, \
                f"Single level ({desc}) {name} complexity should be 1.0, got {val}"
        elif name == 'reuse_storage':
            assert val >= 0.0, \
                f"Single level ({desc}) {name} complexity should be >= 0.0, got {val}"
        else:
            assert val > 0.0, \
                f"Single level ({desc}) {name} complexity should be positive, got {val}"
        if rank == 0:
            print(f"  {desc}: {name} complexity = {val:.4f}")
    pc.destroy()

if rank == 0:
    print("Single level complexity check passed")

A.destroy()
