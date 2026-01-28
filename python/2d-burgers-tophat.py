import sys
import petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# 1. PARAMETERS (INVISCID)
# ======================================================
N  = 129
nu = 0.0                 
T  = 1.0
dt = 0.002               

# ======================================================
# 2. DMDA GRID
# ======================================================
da = PETSc.DMDA().create(
    dim=2,
    sizes=[N, N],
    dof=2,
    stencil_width=1,
    boundary_type=('ghosted', 'ghosted'),
    comm=PETSc.COMM_WORLD
)
da.setUniformCoordinates(-1.0, 1.0, -1.0, 1.0)
da.setUp()

hx = 2.0 / (N - 1)
hy = 2.0 / (N - 1)

# ======================================================
# 3. RHS FUNCTION (PURE INVISCID UPWIND)
# ======================================================
def RHSFunction(ts, t, X, F):
    dm = ts.getDM()
    Xloc = dm.createLocalVec()
    dm.globalToLocal(X, Xloc)

    u = dm.getVecArray(Xloc)
    f = dm.getVecArray(F)

    (xs, xe), (ys, ye) = dm.getRanges()

    for j in range(ys, ye):
        for i in range(xs, xe):

            # Dirichlet boundaries
            if i == 0 or j == 0 or i == N-1 or j == N-1:
                f[i, j] = (0.0, 0.0)
                continue

            ui, vi = u[i, j]

            # Upwind x
            if ui >= 0:
                du_dx = (ui - u[i-1, j][0]) / hx
                dv_dx = (vi - u[i-1, j][1]) / hx
            else:
                du_dx = (u[i+1, j][0] - ui) / hx
                dv_dx = (u[i+1, j][1] - vi) / hx

            # Upwind y
            if vi >= 0:
                du_dy = (ui - u[i, j-1][0]) / hy
                dv_dy = (vi - u[i, j-1][1]) / hy
            else:
                du_dy = (u[i, j+1][0] - ui) / hy
                dv_dy = (u[i, j+1][1] - vi) / hy

            f[i, j][0] = -(ui * du_dx + vi * du_dy)
            f[i, j][1] = -(ui * dv_dx + vi * dv_dy)

# ======================================================
# 4. TIME STEPPER (EXPLICIT SSP)
# ======================================================
ts = PETSc.TS().create()
ts.setDM(da)
ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)
ts.setType(PETSc.TS.Type.SSP)     # <<< CORRECT METHOD

ts.setRHSFunction(RHSFunction)
ts.setTimeStep(dt)
ts.setMaxTime(T)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)

if PETSc.COMM_WORLD.getRank() == 0:
    print("--- Configuration ---")
    print("Equation : 2D INVISCID Burgers")
    print("Method   : Explicit SSP Runge–Kutta")
    print("Solver   : No SNES / No MG")
    print("----------------------")

# ======================================================
# 5. INITIAL CONDITION (TOP-HAT / BOX)
# ======================================================
X = da.createGlobalVec()
arr = da.getVecArray(X)
(xs, xe), (ys, ye) = da.getRanges()

probe_i, probe_j = N//4, N//2

for j in range(ys, ye):
    for i in range(xs, xe):
        x = -1.0 + i * hx
        y = -1.0 + j * hy

        if abs(x) < 0.5 and abs(y) < 0.5:
            arr[i, j] = (1.0, 0.0)
        else:
            arr[i, j] = (0.0, 0.0)

# ======================================================
# 6. MONITOR
# ======================================================
history_time = []
history_probe_u = []
history_slices = []

def monitor(ts, step, time, X):
    if PETSc.COMM_WORLD.getRank() == 0:
        print(f"Step {step:4d} | t = {time:.3f}")
        A = X.getArray(readonly=True).reshape((N, N, 2))
        history_time.append(time)
        history_probe_u.append(A[probe_j, probe_i, 0])
        history_slices.append(A[N//2, :, 0].copy())

ts.setMonitor(monitor)

# ======================================================
# 7. SOLVE
# ======================================================
if PETSc.COMM_WORLD.getRank() == 0:
    print("Starting explicit solve...")

ts.solve(X)


# ======================================================
# PRINT TOTAL NUMBER OF ITERATIONS
# ======================================================
if PETSc.COMM_WORLD.getRank() == 0:
    print("\n--- Time Stepping Information ---")
    print("Total TS steps executed :", ts.getStepNumber())

# ======================================================
# 8. PLOTTING
# ======================================================
if PETSc.COMM_WORLD.getRank() == 0:
    print("Generating plots...")

    U = X.getArray().reshape((N, N, 2))[:, :, 0]
    x = np.linspace(-1, 1, N)
    Xg, Yg = np.meshgrid(x, x)

    # 1D slice
    plt.figure(figsize=(7,5))
    idxs = [0, len(history_slices)//2, -1]
    styles = [':', '--', '-']
    for k, idx in enumerate(idxs):
        plt.plot(x, history_slices[idx], styles[k], linewidth=2,
                 label=f"t={history_time[idx]:.2f}")
    plt.legend()
    plt.grid(True)
    plt.title("1D Slice – Inviscid Top-Hat")
    plt.xlabel("x")
    plt.savefig("2d-burgers-tophat-slice.png", dpi=150)
    plt.close()

    # 2D contour
    plt.figure(figsize=(7,6))
    plt.contourf(Xg, Yg, U, 50, cmap="RdBu_r")
    plt.colorbar(label="u(x,y)")
    plt.title("2D Inviscid Burgers – Top-Hat")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("2d-burgers-tophat-contour.png", dpi=150)
    plt.close()

    # 3D surface
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xg, Yg, U, cmap="RdBu_r",
                    rstride=2, cstride=2, linewidth=0)
    ax.set_title("3D Surface – Inviscid Top-Hat")
    ax.set_zlim(-0.1, 1.1)
    plt.savefig("2d-burgers-tophat-surface.png", dpi=150)
    plt.close()

    print("Inviscid plots saved.")
