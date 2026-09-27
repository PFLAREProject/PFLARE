'''
Tests the direct Python API for PCAIR get/set option functions introduced in
pflare.py (backed by PCAIR_C_Fortran_Bindings.F90).

Three checks are performed:
  1. Round-trip: set a value via the direct API, get it back, verify it matches.
  2. Functional: configure lAIR with WJacobi smoothing via the direct API,
     run a solve, verify convergence.
  3. Wrong type: the get/set functions raise PETSc.Error on a PC that is not
     of type PCAIR, rather than corrupting memory.
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
    # PetscReal options round-trip through the build's real precision (float32
    # under single), so floats are compared with a relative tolerance rather than
    # for exact equality (double round-trips exactly, so this is a no-op there).
    # Non-float values (ints, enums, strings, bools) are compared exactly.
    if isinstance(expected, float):
        ok = math.isclose(got, expected, rel_tol=1e-6, abs_tol=1e-12)
    else:
        ok = got == expected
    if not ok:
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

# Neumann always diagonally scales, but setting the flag while Neumann is the
# inverse type must still be stored, as it is used by the other (e.g., C point)
# inverses and must survive a later change of inverse type
pflare.pcair_set_inverse_type(pc, pflare.PFLAREINV_NEUMANN)
pflare.pcair_set_diag_scale_polys(pc, True)
pflare.pcair_set_inverse_type(pc, pflare.PFLAREINV_ARNOLDI)
check('diag_scale_polys_after_neumann', pflare.pcair_get_diag_scale_polys(pc), True)

pflare.pcair_set_diag_scale_polys(pc, False)
check('diag_scale_polys_false', pflare.pcair_get_diag_scale_polys(pc), False)

# Reuse amounts outside 1, 2 or 3 must be rejected and leave the value alone
for bad_amount in (0, 4):
    try:
        pflare.pcair_set_reuse_amount(pc, bad_amount)
        errors.append(f'reuse_amount_{bad_amount}: expected ValueError')
    except ValueError:
        pass
    check(f'reuse_amount_{bad_amount}_unchanged', pflare.pcair_get_reuse_amount(pc), 3)

# -----------------------------------------------------------------------
# Wrong PC type: the pcair_* (and pcpflareinv_* getter) wrappers must raise
# PETSc.Error rather than let the Fortran interpret another PC type's data
# as PCAIR data. Push the python error handler so -on_error_abort doesn't
# abort on the (expected) PETSc errors
# -----------------------------------------------------------------------
def expect_petsc_error(name, fn, *args):
    try:
        fn(*args)
    except PETSc.Error:
        return
    errors.append(f'{name}: expected PETSc.Error on a PC of the wrong type')

PETSc.Sys.pushErrorHandler('python')
for pc_type in ['jacobi', 'pflareinv', None]:
    pc_wrong = PETSc.PC().create(comm=comm)
    if pc_type is not None:
        pc_wrong.setType(pc_type)
    label = f'wrong_type_{pc_type}'
    expect_petsc_error(label + '_get_num_levels', pflare.pcair_get_num_levels, pc_wrong)
    expect_petsc_error(label + '_get_max_levels', pflare.pcair_get_max_levels, pc_wrong)
    expect_petsc_error(label + '_get_strong_threshold', pflare.pcair_get_strong_threshold, pc_wrong)
    expect_petsc_error(label + '_get_symmetric', pflare.pcair_get_symmetric, pc_wrong)
    expect_petsc_error(label + '_get_smooth_type', pflare.pcair_get_smooth_type, pc_wrong)
    expect_petsc_error(label + '_get_grid_complexity', pflare.pcair_get_grid_complexity, pc_wrong)
    expect_petsc_error(label + '_set_max_levels', pflare.pcair_set_max_levels, pc_wrong, 5)
    expect_petsc_error(label + '_set_strong_threshold', pflare.pcair_set_strong_threshold, pc_wrong, 0.5)
    expect_petsc_error(label + '_set_symmetric', pflare.pcair_set_symmetric, pc_wrong, True)
    expect_petsc_error(label + '_set_smooth_type', pflare.pcair_set_smooth_type, pc_wrong, 'fc')
    expect_petsc_error(label + '_set_inverse_type', pflare.pcair_set_inverse_type, pc_wrong, pflare.PFLAREINV_POWER)
    if pc_type != 'pflareinv':
        expect_petsc_error(label + '_pflareinv_get_poly_order', pflare.pcpflareinv_get_poly_order, pc_wrong)
        expect_petsc_error(label + '_pflareinv_get_matrix_free', pflare.pcpflareinv_get_matrix_free, pc_wrong)
    pc_wrong.destroy()
PETSc.Sys.popErrorHandler()

# A 10 character smooth type is the longest supported and must round-trip
pflare.pcair_set_smooth_type(pc, 'fcfcfcfcfc')
check('smooth_type_10',  pflare.pcair_get_smooth_type(pc),     'fcfcfcfcfc')

# Longer smooth types must error rather than be silently truncated
try:
    pflare.pcair_set_smooth_type(pc, 'ffffffffffc')
    errors.append('smooth_type_11: expected ValueError from pcair_set_smooth_type')
except ValueError:
    pass
check('smooth_type_11_unchanged', pflare.pcair_get_smooth_type(pc), 'fcfcfcfcfc')

# Same through the options database, which goes through the C PCAIRSetSmoothType
# The error handler is swapped so the expected error is returned rather than aborting
pc_opts = PETSc.PC().create(comm=comm)
pc_opts.setType('air')
pc_opts.setOptionsPrefix('long_smooth_')
opts = PETSc.Options()
opts['long_smooth_pc_air_smooth_type'] = 'ffffffffffc'
PETSc.Sys.pushErrorHandler('return')
try:
    pc_opts.setFromOptions()
    errors.append('smooth_type_11_options: expected an error from -pc_air_smooth_type')
except PETSc.Error:
    pass
finally:
    PETSc.Sys.popErrorHandler()
del opts['long_smooth_pc_air_smooth_type']
pc_opts.destroy()

# -----------------------------------------------------------------------
# C point smoother options default to the F point values if unset
# -----------------------------------------------------------------------
# Without calling setFromOptions, unset C values follow the F values
pc_c = PETSc.PC().create(comm=comm)
pc_c.setType('air')
pflare.pcair_set_inverse_type(pc_c, pflare.PFLAREINV_POWER)
pflare.pcair_set_poly_order(pc_c, 5)
pflare.pcair_set_inverse_sparsity_order(pc_c, 2)
check('c_inverse_type_default', pflare.pcair_get_c_inverse_type(pc_c), pflare.PFLAREINV_POWER)
check('c_poly_order_default', pflare.pcair_get_c_poly_order(pc_c), 5)
check('c_inverse_sparsity_order_default', pflare.pcair_get_c_inverse_sparsity_order(pc_c), 2)
# setFromOptions must not pin the unset C values, they keep following F
pc_c.setFromOptions()
pflare.pcair_set_poly_order(pc_c, 7)
check('c_poly_order_follows_f', pflare.pcair_get_c_poly_order(pc_c), 7)
pc_c.destroy()

# C values set explicitly must survive setFromOptions and later F changes
pc_c = PETSc.PC().create(comm=comm)
pc_c.setType('air')
pflare.pcair_set_c_inverse_type(pc_c, pflare.PFLAREINV_NEUMANN)
pflare.pcair_set_c_poly_order(pc_c, 3)
pflare.pcair_set_c_inverse_sparsity_order(pc_c, 0)
pc_c.setFromOptions()
pflare.pcair_set_inverse_type(pc_c, pflare.PFLAREINV_POWER)
pflare.pcair_set_poly_order(pc_c, 9)
pflare.pcair_set_inverse_sparsity_order(pc_c, 2)
check('c_inverse_type_explicit', pflare.pcair_get_c_inverse_type(pc_c), pflare.PFLAREINV_NEUMANN)
check('c_poly_order_explicit', pflare.pcair_get_c_poly_order(pc_c), 3)
check('c_inverse_sparsity_order_explicit', pflare.pcair_get_c_inverse_sparsity_order(pc_c), 0)
pc_c.destroy()

# Setting a C value equal to the current F value still pins it
pc_c = PETSc.PC().create(comm=comm)
pc_c.setType('air')
pflare.pcair_set_c_poly_order(pc_c, pflare.pcair_get_poly_order(pc_c))
pflare.pcair_set_poly_order(pc_c, 2)
check('c_poly_order_pinned', pflare.pcair_get_c_poly_order(pc_c), 6)
pc_c.destroy()

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
