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
N  = 129
nu = 1.0 / (100.0 * np.pi)
dt = 0.002
T  = 1.0

# ========================================================
# 2. DMDA GRID
# ========================================================
da = PETSc.DMDA().create(
    dim=2,
    sizes=[N, N],
    dof=2,
    stencil_width=1,
    boundary_type=('ghosted', 'ghosted'),
    comm=PETSc.COMM_WORLD
)
da.setUniformCoordinates(-1, 1, -1, 1)
da.setUp()

hx = 2.0 / (N - 1)
hy = 2.0 / (N - 1)

# ========================================================
# 3. EXPLICIT CONVECTION
# ========================================================
def RHSFunction(ts, t, X, F):
    dm = ts.getDM()
    Xloc = dm.createLocalVec()
    dm.globalToLocal(X, Xloc)

    u = dm.getVecArray(Xloc)
    f = dm.getVecArray(F)

    (xs, xe), (ys, ye) = dm.getRanges()

    for j in range(ys, ye):
        for i in range(xs, xe):
            if i == 0 or j == 0 or i == N-1 or j == N-1:
                f[i, j] = (0.0, 0.0)
                continue

            ui, vi = u[i, j]

            du_dx = (u[i+1, j][0] - u[i-1, j][0]) / (2*hx)
            du_dy = (u[i, j+1][0] - u[i, j-1][0]) / (2*hy)
            dv_dx = (u[i+1, j][1] - u[i-1, j][1]) / (2*hx)
            dv_dy = (u[i, j+1][1] - u[i, j-1][1]) / (2*hy)

            f[i, j][0] = -(ui*du_dx + vi*du_dy)
            f[i, j][1] = -(ui*dv_dx + vi*dv_dy)

# ========================================================
# 4. IMPLICIT FUNCTION (LINEAR)
# ========================================================
def IFunction(ts, t, X, Xdot, F):
    # Residual: F = Xdot - nu*ΔX (Δ applied via Jacobian)
    F.copy(Xdot)

# ========================================================
# 5. IMPLICIT JACOBIAN (DIFFUSION)
# ========================================================
def IJacobian(ts, t, X, Xdot, shift, A, B):
    hx2 = hx * hx
    hy2 = hy * hy

    A.zeroEntries()

    row = PETSc.Mat.Stencil()
    col = PETSc.Mat.Stencil()

    for j in range(N):
        for i in range(N):
            for c in range(2):
                row.index = (i, j)
                row.field = c

                if i == 0 or j == 0 or i == N-1 or j == N-1:
                    A.setValueStencil(row, row, shift)
                    continue

                diag = shift + 2*nu*(1/hx2 + 1/hy2)
                A.setValueStencil(row, row, diag)

                col.field = c

                col.index = (i-1, j)
                A.setValueStencil(row, col, -nu/hx2)

                col.index = (i+1, j)
                A.setValueStencil(row, col, -nu/hx2)

                col.index = (i, j-1)
                A.setValueStencil(row, col, -nu/hy2)

                col.index = (i, j+1)
                A.setValueStencil(row, col, -nu/hy2)

    A.assemble()
    if A != B:
        B.assemble()

# ========================================================
# 6. TS IMEX SOLVER
# ========================================================
ts = PETSc.TS().create()
ts.setDM(da)
ts.setType(PETSc.TS.Type.ARKIMEX)
ts.setProblemType(PETSc.TS.ProblemType.LINEAR)

ts.setRHSFunction(RHSFunction)
ts.setIFunction(IFunction)
ts.setIJacobian(IJacobian)

ts.setTimeStep(dt)
ts.setMaxTime(T)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)

# ========================================================
# 7. SOLVER OPTIONS (VERSION SAFE)
# ========================================================
opts = PETSc.Options()

# 🔒 Disable adaptive time stepping
opts["ts_adapt_type"] = "none"

# IMEX solver internals
opts["arkimex_snes_type"] = "ksponly"
opts["arkimex_ksp_type"]  = "gmres"
opts["arkimex_pc_type"]   = "mg"
opts["arkimex_mg_levels_pc_type"] = "jacobi"
opts["arkimex_mg_coarse_pc_type"] = "lu"

ts.setFromOptions()

# ========================================================
# 8. INITIAL CONDITION
# ========================================================
X = da.createGlobalVec()
arr = da.getVecArray(X)
(xs, xe), (ys, ye) = da.getRanges()

for j in range(ys, ye):
    for i in range(xs, xe):
        x = -1 + i * hx
        arr[i, j] = (-np.sin(np.pi * x), 0.0)

# ========================================================
# 9. SOLVE
# ========================================================
if PETSc.COMM_WORLD.getRank() == 0:
    print("Running FIXED-DT IMEX Burgers solver...")

ts.solve(X)

# ========================================================
# 10. STATISTICS
# ========================================================
if PETSc.COMM_WORLD.getRank() == 0:
    print("-" * 40)
    print(f"Time steps      : {ts.getStepNumber()} ")
    print(f"SNES iterations : {ts.getSNESIterations()}")
    print(f"KSP iterations  : {ts.getKSPIterations()} ")
    print("-" * 40)

# ========================================================
# 11. PLOT
# ========================================================
if PETSc.COMM_WORLD.getRank() == 0:
    U = X.getArray().reshape((N, N, 2))[:, :, 0]
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    Xg, Yg = np.meshgrid(x, y)

    plt.figure(figsize=(7, 6))
    plt.contourf(Xg, Yg, U, 50, cmap="RdBu_r")
    plt.colorbar(label="u(x,y)")
    plt.title("2D Viscous Burgers (IMEX, Fixed dt)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig("burgers_imex_fixed_dt.png", dpi=150)
    plt.close()

    print("Saved burgers_imex_fixed_dt.png")
