import sys
import petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ========================================================
# 1. PARAMETERS
# ========================================================
N  = 257
nu = 1.0 / (100.0 * np.pi)
dt = 0.002
T  = 1.0

# ========================================================
# 2. DMDA GRID (1D)
# ========================================================
da = PETSc.DMDA().create(
    dim=1,
    sizes=[N],
    dof=1,
    stencil_width=1,
    boundary_type=('ghosted',),
    comm=PETSc.COMM_WORLD
)
da.setUniformCoordinates(-1, 1)
da.setUp()

hx = 2.0 / (N - 1)

# ========================================================
# 3. EXPLICIT PART: CONVECTION
# ========================================================
def RHSFunction(ts, t, X, F):
    dm = ts.getDM()
    Xloc = dm.createLocalVec()
    dm.globalToLocal(X, Xloc)

    u = dm.getVecArray(Xloc)
    f = dm.getVecArray(F)

    (xs, xe) = dm.getRanges()[0]

    for i in range(xs, xe):
        if i == 0 or i == N - 1:
            f[i] = 0.0
            continue

        ui = u[i]

        # Upwind scheme
        if ui >= 0:
            du_dx = (ui - u[i - 1]) / hx
        else:
            du_dx = (u[i + 1] - ui) / hx

        f[i] = -ui * du_dx

# ========================================================
# 4. IMPLICIT PART: DIFFUSION
# ========================================================
def IFunction(ts, t, X, Xdot, F):
    dm = ts.getDM()
    Xloc = dm.createLocalVec()
    dm.globalToLocal(X, Xloc)

    u    = dm.getVecArray(Xloc)
    xdot = dm.getVecArray(Xdot)
    f    = dm.getVecArray(F)

    (xs, xe) = dm.getRanges()[0]

    for i in range(xs, xe):
        if i == 0 or i == N - 1:
            f[i] = xdot[i]
            continue

        uxx = (u[i + 1] - 2.0 * u[i] + u[i - 1]) / hx**2
        f[i] = xdot[i] - nu * uxx

# ========================================================
# 5. TS: IMEX SOLVER
# ========================================================
ts = PETSc.TS().create()
ts.setDM(da)
ts.setType(PETSc.TS.Type.ARKIMEX)
ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)

ts.setRHSFunction(RHSFunction)
ts.setIFunction(IFunction)

ts.setTimeStep(dt)
ts.setMaxTime(T)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)

# ========================================================
# 6. LINEAR SOLVER (DIFFUSION)
# ========================================================
opts = PETSc.Options()
opts["ksp_type"] = "cg"
opts["pc_type"] = "mg"
opts["mg_levels_pc_type"] = "sor"
opts["mg_coarse_pc_type"] = "lu"

# ========================================================
# 7. ITERATION MONITORS (SNES + KSP)
# ========================================================
snes = ts.getSNES()
ksp  = snes.getKSP()

def snes_monitor(snes, its, norm):
    print(f"[SNES] Iter {its:2d}  Residual = {norm:.3e}")

def ksp_monitor(ksp, its, rnorm):
    print(f"    [KSP]  Iter {its:2d}  Residual = {rnorm:.3e}")

snes.setMonitor(snes_monitor)
ksp.setMonitor(ksp_monitor)

ts.setFromOptions()

# ========================================================
# 8. INITIAL CONDITION (GAUSSIAN PULSE)
# ========================================================
X = da.createGlobalVec()
arr = da.getVecArray(X)
(xs, xe) = da.getRanges()[0]

x0    = 0.0
sigma = 0.15

for i in range(xs, xe):
    x = -1.0 + i * hx
    arr[i] = np.exp(-((x - x0)**2) / (2.0 * sigma**2))

# ========================================================
# 9. SOLVE
# ========================================================
if PETSc.COMM_WORLD.getRank() == 0:
    print("Running 1D viscous Burgers (Gaussian IC, IMEX)...\n")

ts.solve(X)

# ========================================================
# 10. FINAL ITERATION COUNTS
# ========================================================
if PETSc.COMM_WORLD.getRank() == 0:
    print("\n===== Solver Statistics =====")
    print("Total SNES iterations :", snes.getIterationNumber())
    print("Total KSP iterations  :", ksp.getIterationNumber())

# ========================================================
# 11. PLOT RESULT
# ========================================================
if PETSc.COMM_WORLD.getRank() == 0:
    u = X.getArray()
    x = np.linspace(-1, 1, N)

    plt.figure(figsize=(7, 5))
    plt.plot(x, u, linewidth=2)
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("1D Viscous Burgers (Gaussian Pulse IC)")
    plt.tight_layout()
    plt.savefig("burgers_1d_viscous_gaussian.png", dpi=150)
    plt.close()

    print("Saved burgers_1d_viscous_gaussian.png")
