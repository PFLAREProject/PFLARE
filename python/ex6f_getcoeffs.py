'''
Demonstrates saving and restoring GMRES polynomial coefficients so that a
preconditioner built for a previous linear system can be exactly reproduced
without re-running the parallel polynomial iteration.

Mirrors tests/ex6f_getcoeffs.F90.  Works with both PCAIR (multi-level) and
PCPFLAREINV (single-level); the PC type is chosen via -pc_type on the command
line.

Three solves are performed:
  Solve 1  – original system  (poly coefficients are saved afterwards)
  Solve 2  – perturbed system (forces a fresh setup, discards any saved state)
  Solve 3  – original system again, but with the poly coefficients from solve 1
             restored and the reuse flag set.

The test passes when the residual norm of solve 3 equals that of solve 1 to
within a relative tolerance of 1e-8.
'''

import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

import pflare

comm = PETSc.COMM_WORLD
rank = comm.getRank()

OptDB = PETSc.Options()
m = OptDB.getInt('m', 5)
n = OptDB.getInt('n', 5)

# -----------------------------------------------------------------------
# Build the 2-D five-point Laplacian (same matrix as ex6f_getcoeffs.F90)
# -----------------------------------------------------------------------
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

# -----------------------------------------------------------------------
# Vectors
# -----------------------------------------------------------------------
u = PETSc.Vec().create(comm=comm)
u.setSizes(m * n)
u.setFromOptions()
b = u.duplicate()
x = u.duplicate()

# -----------------------------------------------------------------------
# KSP / PC  (PC type and options come from the command line)
# -----------------------------------------------------------------------
ksp = PETSc.KSP().create(comm=comm)
ksp.setFromOptions()

pc = ksp.getPC()
pc_type = pc.getType()
is_air       = (pc_type == 'air')
is_pflareinv = (pc_type == 'pflareinv')

if not (is_air or is_pflareinv):
    if rank == 0:
        print(f"Unexpected PC type '{pc_type}'; expected 'air' or 'pflareinv'")
    sys.exit(1)

# For PCAIR we also need to reuse sparsity so that the same hierarchy is
# available when we restore poly coefficients.
if is_air:
    OptDB['pc_air_reuse_sparsity'] = ''

# -----------------------------------------------------------------------
# Storage for saved coefficients
# -----------------------------------------------------------------------
coeffs_air       = {}   # petsc_level -> numpy array  (PCAIR)
coeffs_pflareinv = None  # numpy array                (PCPFLAREINV)

norm_first = None
nsteps = 3

for count in range(1, nsteps + 1):

    # -- Modify A so that solve 3 reproduces solve 1 --------------------
    Istart, Iend = A.getOwnershipRange()
    if count == 1:
        delta = 2.0
    elif count == 2:
        delta = 0.1
    else:  # count == 3
        delta = -0.1

    for II in range(Istart, Iend):
        A.setValues(II, II, delta, addv=True)
    A.assemblyBegin(A.AssemblyType.FINAL)
    A.assemblyEnd(A.AssemblyType.FINAL)

    ksp.setOperators(A, A)
    u.set(1.0)
    A.mult(u, b)

    # -- On solve 3 restore saved polynomial coefficients ---------------
    if count == 3:
        if is_air:
            num_levels = pflare.pcair_get_num_levels(pc)
            for petsc_level in range(num_levels - 1, 0, -1):
                pflare.pcair_set_poly_coeffs(pc, petsc_level,
                                              pflare.COEFFS_INV_AFF,
                                              coeffs_air[petsc_level])
            pflare.pcair_set_poly_coeffs(pc, 0,
                                          pflare.COEFFS_INV_COARSE,
                                          coeffs_air[0])
            OptDB['pc_air_reuse_poly_coeffs'] = ''

        elif is_pflareinv:
            pflare.pcpflareinv_set_poly_coeffs(pc, coeffs_pflareinv)
            OptDB['pc_pflareinv_reuse_poly_coeffs'] = ''

    ksp.solve(b, x)

    reason = ksp.getConvergedReason()
    if reason <= 0:
        if rank == 0:
            print(f"KSP did not converge on solve {count} (reason {reason})")
        sys.exit(1)

    # -- Compute residual norm: r = b - A*x -----------------------------
    r = b - A * x
    norm = r.norm(PETSc.NormType.NORM_2)
    if count == 1:
        norm_first = norm
    if count == 3:
        norm_third = norm

    # -- After solve 1 save polynomial coefficients ---------------------
    if count == 1:
        if is_air:
            num_levels = pflare.pcair_get_num_levels(pc)
            for petsc_level in range(num_levels - 1, 0, -1):
                coeffs_air[petsc_level] = pflare.pcair_get_poly_coeffs(
                    pc, petsc_level, pflare.COEFFS_INV_AFF)
            coeffs_air[0] = pflare.pcair_get_poly_coeffs(
                pc, 0, pflare.COEFFS_INV_COARSE)

        elif is_pflareinv:
            coeffs_pflareinv = pflare.pcpflareinv_get_poly_coeffs(pc)

# -----------------------------------------------------------------------
# Check that solve 3 reproduced solve 1
# -----------------------------------------------------------------------
if rank == 0:
    rel_diff = abs(norm_first - norm_third) / norm_first
    if rel_diff > 1e-8:
        print(f"FAIL: residual norms differ by {rel_diff:.3e} "
              f"(solve1={norm_first:.6e}, solve3={norm_third:.6e})")
        sys.exit(1)

# Tidy up
u.destroy()
b.destroy()
x.destroy()
A.destroy()
ksp.destroy()
