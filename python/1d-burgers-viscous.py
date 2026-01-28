import sys
import petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameters (shock-safe)
# ----------------------------
nu = 1e-3
N  = 257
dt = 2e-4
T  = 0.3

# ----------------------------
# DMDA
# ----------------------------
da = PETSc.DMDA().create(
    dim=1,
    sizes=[N],
    dof=1,
    stencil_width=1,
    boundary_type=('ghosted',),
    comm=PETSc.COMM_WORLD
)
da.setUp()

# ----------------------------
# RHSFunction (explicit form)
# ----------------------------
def RHSFunction(ts, t, U, F):
    dm = ts.getDM()
    Uloc = dm.createLocalVec()
    dm.globalToLocal(U, Uloc)

    u = dm.getVecArray(Uloc)
    f = dm.getVecArray(F)

    (xs, xe), = dm.getRanges()
    h = 1.0 / (N - 1)

    for i in range(xs, xe):
        if i == 0 or i == N - 1:
            f[i] = 0.0
        else:
            # Upwind convection
            if u[i] >= 0:
                ux = (u[i] - u[i-1]) / h
            else:
                ux = (u[i+1] - u[i]) / h

            uxx = (u[i+1] - 2*u[i] + u[i-1]) / (h*h)
            f[i] = -u[i]*ux + nu*uxx

# ----------------------------
# TS setup (EXPLICIT SSP)
# ----------------------------
ts = PETSc.TS().create()
ts.setDM(da)
ts.setType(PETSc.TS.Type.SSP)   # <<< KEY FIX
ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)

ts.setRHSFunction(RHSFunction)
ts.setTimeStep(dt)
ts.setMaxTime(T)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)

# ----------------------------
# Initial condition (shock-forming)
# ----------------------------
u = da.createGlobalVec()
x = np.linspace(0, 1, N)
u_arr = u.getArray()
u_arr[:] = np.where(x < 0.5, 1.0, -1.0)

# ----------------------------
# Store solution for x–t plot
# ----------------------------
solutions = []
times = []

def monitor(ts, step, time, u):
    if PETSc.COMM_WORLD.getRank() == 0:
        u_copy = u.copy()      # SAFE
        solutions.append(u_copy.getArray().copy())
        times.append(time)

ts.setMonitor(monitor)

# ----------------------------
# Solve
# ----------------------------
ts.solve(u)


if PETSc.COMM_WORLD.getRank() == 0:
    steps = ts.getStepNumber()
    print("-" * 30)
    print(f"Simulation Complete")
    print(f"Total Time Steps : {steps}")
    print("-" * 30)
    

# ----------------------------
# Plot x–t diagram
# ----------------------------
if PETSc.COMM_WORLD.getRank() == 0:
    Uplot = np.array(solutions)
    X, Tm = np.meshgrid(x, times)

    plt.figure(figsize=(7,5))
    plt.contourf(X, Tm, Uplot, 100)
    plt.colorbar(label="u(x,t)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Shock propagation in 1D Burgers equation")
    plt.tight_layout()
    plt.savefig("1d-burgers-viscous-xt.png", dpi=300)
    plt.close()

# ----------------------------
# Plot 3D surface u(x,t)
# ----------------------------
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

if PETSc.COMM_WORLD.getRank() == 0:
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    surf = ax.plot_surface(
        X, Tm, Uplot,
        cmap='viridis',
        linewidth=0,
        antialiased=True
    )

    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u(x,t)")
    ax.set_title("Surface plot of shock propagation (Burgers equation)")

    fig.colorbar(surf, shrink=0.6, aspect=10, label="u(x,t)")
    plt.tight_layout()
    plt.savefig("1d-burgers-viscous-surface.png", dpi=300)
    plt.close()