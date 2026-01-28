import sys
import petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ========================================================
# 1. PARAMETERS (INVISCID)
# ========================================================
N  = 257
dt = 0.002          # CFL-limited
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
# 3. RHS FUNCTION (PURE ADVECTION, UPWIND)
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

        # Upwind discretization
        if ui >= 0:
            du_dx = (ui - u[i - 1]) / hx
        else:
            du_dx = (u[i + 1] - ui) / hx

        f[i] = -ui * du_dx

# ========================================================
# 4. TS SETUP (EXPLICIT SSP)
# ========================================================
ts = PETSc.TS().create()
ts.setDM(da)
ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)
ts.setType(PETSc.TS.Type.SSP)

ts.setRHSFunction(RHSFunction)
ts.setTimeStep(dt)
ts.setMaxTime(T)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)

# ========================================================
# 5. TIME-STEP ITERATION MONITOR  ⭐ ADDED ⭐
# ========================================================
def ts_monitor(ts, step, time, X):
    print(f"[TS] Step {step:4d}  Time = {time:.4f}")

ts.setMonitor(ts_monitor)

# ========================================================
# 6. INITIAL CONDITION (SINUSOIDAL)
# ========================================================
X = da.createGlobalVec()
arr = da.getVecArray(X)
(xs, xe) = da.getRanges()[0]

for i in range(xs, xe):
    x = -1.0 + i * hx
    arr[i] = -np.sin(np.pi * x)

# ========================================================
# 7. SOLVE
# ========================================================
if PETSc.COMM_WORLD.getRank() == 0:
    print("Running 1D inviscid Burgers (SSP)...\n")

ts.solve(X)

# ========================================================
# 8. FINAL STEP COUNT
# ========================================================
if PETSc.COMM_WORLD.getRank() == 0:
    print("\n===== Time Stepping Statistics =====")
    print("Total time steps :", ts.getStepNumber())

# ========================================================
# 9. PLOT RESULT
# ========================================================
if PETSc.COMM_WORLD.getRank() == 0:
    u = X.getArray()
    x = np.linspace(-1, 1, N)

    plt.figure(figsize=(7, 5))
    plt.plot(x, u, linewidth=2)
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("1D Inviscid Burgers (Sinusoidal IC)")
    plt.tight_layout()
    plt.savefig("burgers_1d_inviscid.png", dpi=150)
    plt.close()

    print("Saved burgers_1d_inviscid.png")
