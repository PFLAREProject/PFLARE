# Handover: local Gauss-Seidel inside the AIRG coarse-grid solver (advection)

**Branch:** `claude/coarse-gs-handover` on `PFLAREProject/PFLARE`, branched from `main` @ 4326ad1.
It only adds this folder (`coarse_gs_handover/`); no library code has changed. Get it with
`git fetch origin claude/coarse-gs-handover && git checkout claude/coarse-gs-handover`.

**Requirement from the user: auto-truncation stays on** (`-pc_air_auto_truncate_start_level`,
`-pc_air_auto_truncate_tol`). The fixed `-pc_air_max_levels` runs below are only a stand-in, used
because auto-truncation can't currently evaluate an overridden coarse solver (see "PFLARE / PETSc
behaviour"). The goal is a coarse solver that auto-truncation itself tests.

## Question
Does putting a *local* (per-rank, block-Jacobi) forward Gauss-Seidel sweep into the coarse-grid
solver make it strong enough to truncate the AIRG hierarchy much higher for advection, and how do
we get a downstream ordering so one forward sweep is good? The target is GPUs (LUMI-G), as in the
GPU paper (Dargaville et al., "Solving advection equations with reduction multigrids on GPUs",
JCP 565 (2026) 115177, arXiv 2508.17517).

**Not done: the paper was not read.** The cloud session's network policy blocked arXiv,
ScienceDirect and DOI hosts. Read it first and match the test problems, truncation settings
(auto-truncate tol, coarse poly order, matrix-free Newton) and weak-scaling setup used there. All
results below are on CPU (4 cores, CI Docker image `stevendargaville/pflare:latest`, PFLARE `main`
@ 4326ad1) and use `tests/adv_diff_fd` / `tests/adv_dg_upwind`.

## Theory (verified numerically below)
1. **AIR coarse grids stay acyclic for pure upwind advection.** If A is lower triangular in a
   downstream (topological) order, then A_ff is lower triangular, so any polynomial in A_ff is too;
   Z = -A_cf p(A_ff) and W only connect a point to points upstream of it; and RAP = A_cc + A_cf W +
   Z A_fc + Z A_ff W is lower triangular in the C-point order inherited from the fine grid.
   Dropping only removes entries, and lumping only changes the diagonal.
2. Split A = A_d + A_o (on-rank diagonal block + off-rank part). In downstream order one forward
   GS sweep **is** A_d^{-1}, so A M^{-1} = I + A_o A_d^{-1} = I + N with N **nilpotent**. Its index is
   the number of rank-subdomain crossings along the longest characteristic: 2*sqrt(P)-1 for a 2D
   sqrt(P) x sqrt(P) decomposition with diagonal flow.
3. So spec(A M^{-1}) = {1}, and the optimal residual polynomial is (1-z)^k, i.e. **plain Richardson
   with the block-GS preconditioner** (a Neumann series in N). It is exact at k = index, and the
   degree needed scales with P^{1/d}, not with the coarse grid's own characteristic length.
4. The critical path is conserved: the plain polynomial is already a Jacobi-iterated triangular
   solve. The GS version replaces "k x (SpMV + halo)" with "a few x (SpTRSV, ~n_local^{1/d} levels,
   no MPI)" and does far fewer flops. Whether that wins on GPU depends on the cost of SpTRSV levels
   at the truncation level. **This is the open question for LUMI.**

## Results so far (CPU)

### A. Coarse matrices are exactly lower triangular in inherited order
`tri.py` on coarse matrices dumped with `-mg_coarse_ksp_view_mat binary:c.dat`.
`adv_diff_fd` 128^2 (np=1,4; truncated at 3/6/9 levels), `-curved_velocity`, and `-dim 3`:
**upper-triangle nnz = 0 in every local diagonal block.** DMDA natural ordering is already
downstream because velocities are restricted to be >= 0.

### B. Coarse-solve quality on real AIR coarse matrices (`coarse_poly.py`, rel. residual, random RHS)
The polynomial is built from a different random vector, as PFLARE does (harmonic Ritz / Leja
Newton form).

adv_diff_fd 256^2, 16 ranks (4x4), truncated at L=6 (N=12545):

| k | poly(A) | poly(A M^-1) M^-1 | Richardson(M) k sweeps | GS then poly(A) (multiplicative) |
|---|---|---|---|---|
| 2 | 2.7e-1 | 1.3e-1 | 1.5e-1 | 1.3e-1 |
| 4 | 1.6e-1 | 4.7e-2 | 5.1e-2 | 9.1e-2 |
| 8 | 9.2e-2 | 7.3e-3 | 7.9e-3 | 6.0e-2 |
| 10 | 7.7e-2 | 1.9e-12 | 3.5e-13 | 5.1e-2 |
| 16 | 5.0e-2 | 5e-16 | 2e-16 | 3.6e-2 |

- Diagonal scaling makes no difference here (constant diagonal).
- The multiplicative composition (GS sweep, then poly(A) on the residual) is weak, as theory
  predicts. GS must be the preconditioner *inside* the polynomial.
- **Robustness issue:** the GMRES polynomial built on A M^{-1} blows up (1e6 to 1e24) at high k
  once Arnoldi passes the nilpotency index, because the happy breakdown is hidden by roundoff.
  Richardson(M) is just as strong and stable at any k, so it should be the default form.

### C. Single-level block-GS Richardson (no MG), adv_diff_fd 256^2
The residual is exactly 0 after 1 sweep (np=1), 3 sweeps (np=4) and 7 sweeps (np=16), with
almost no reduction before that (wavefront behaviour). This is a direct solver on CPU but not
scalable in P, so MG is still needed above the coarse grid.

### D. Full V-cycle, outer Richardson, rtol 1e-10, adv_diff_fd, `mg2.sh`
Coarse solver options compared:
- A: `-pc_air_coarsest_inverse_type newton -pc_air_coarsest_matrix_free_polys -pc_air_coarsest_poly_order k`
- B: `-mg_coarse_ksp_type richardson -mg_coarse_ksp_max_it k -mg_coarse_pc_type sor -mg_coarse_pc_sor_local_forward`

1024^2, np=4, k=4 (full hierarchy is 31 levels):

| levels L | A: its / solve | B: its / solve |
|---|---|---|
| 16 | 10 / 0.44s | 10 / 0.44s |
| 12 | 20 / 0.80s | 10 / 0.40s |
| 10 | 34 / 1.36s | 9 / 0.32s |
| 8 | 60 / 2.20s | 8 / 0.30s |
| 6 | 124 / 3.86s | 6 / 0.18s |
| 4 | 200 (max) / 4.37s | 1 / 0.027s |

512^2, np=4, k=4, advection-diffusion (`-alpha`; diffusion adds cycles, so GS is no longer exact):

| L | alpha=1e-5 A | alpha=1e-5 B | alpha=1e-3 A | alpha=1e-3 B |
|---|---|---|---|---|
| 16 | 6 | 6 | 9 | 9 |
| 10 | 13 | 6 | 20 | 9 |
| 8 | 21 | 6 | 113 | 10 |
| 6 | 40 | 5 | 200 | 18 |
| 4 | 86 | 5 | 127 | 43 |

256^2, np=16 (oversubscribed, so ignore the times), k=2 < index 7, stopped early: A takes
33 / 63 its and B takes 9 / 8 its at L = 8 / 6.
**Still to do: k = 2 and 4 at lower L with 16+ ranks.** This is the regime that matters at
scale, where k is below 2*sqrt(P)-1. The run was stopped because 16 ranks on 4 cores was too slow.

### E. Reordering (unstructured DG, `adv_dg_upwind -dm_plex_filename tests/data/square_unstruc.msh -dm_refine 2`)
DMPlex ordering is **not** downstream: about 44% of the local block's Frobenius norm is above the
diagonal, and GS in natural order barely helps. `coarse_poly.py` computes a local topological
order:
1. Build the directed graph with edge j->i for |a_ij| > 1e-8 max|row i|.
2. Find strongly connected components (SCCs) with `scipy.sparse.csgraph`, then run Kahn's algorithm
   on the condensation, keeping SCC members contiguous.
3. Apply local forward GS in that order.

Results:
- Coarse levels at L>=4 have all SCCs of size 1, so the reordered block is exactly triangular. At
  L=2 (curved velocity) the max SCC size is 3 and the residual upper part is 5e-4.
- Curved velocity, np=2, L=4, k=8: natural-order GS-poly gives 1.6e-1, topo-order GS-poly 6.6e-12,
  topo Richardson 4.2e-12.
- The ordering only needs computing on the fine grid. By (A.1) the restriction to C-points is
  valid on every coarser level, though level-set counts must be recomputed. Not implemented in
  PFLARE yet; the Python version is the reference.

## PFLARE / PETSc behaviour checked or found
- **`-mg_coarse_*` works with PCAIR.** PFLARE sets the coarse KSP to PREONLY + PCSHELL and calls
  KSPSetUp (`src/AIR_MG_Setup.F90` ~1205-1275). PCSetUp_MG then calls KSPSetFromOptions on all level
  KSPs on the first setup, which overrides the shell. On re-setup PFLARE calls PCReset on the inner
  PCMG, so setupcalled=0 and the options are applied again. Verified with `-ksp_view -second_solve`.
  The prefix is propagated in `PCSetUp_AIR_c` (`src/PCAIR.c` ~252).
- Caveats with the override:
  - (a) PFLARE still builds the coarse polynomial, which is wasted setup.
  - (b) Auto-truncation (`-pc_air_auto_truncate_*`, `src/AIR_MG_Setup.F90` ~156-260) builds and
    tests only PFLARE's own polynomial on each candidate level, never the `-mg_coarse_*` solver.
    With auto-truncation on and a `-mg_coarse_*` override, the hierarchy is cut where the
    *polynomial* passes, and the GS solver is then applied there. That is safe but truncates no
    higher than today. The `-pc_air_max_levels` runs here only emulate what a GS-aware
    auto-truncation would choose. **Using auto-truncation (required) means implementing the GS
    coarse solver inside PFLARE**, so the auto-truncation test applies it (step 4 below).
  - (c) The PCAIR view still prints "Coarse grid solver: GMRES polynomial ...", and the printed
    complexities assume it.
  - (d) Don't use a Krylov coarse KSP (e.g. `-mg_coarse_ksp_type gmres`). It builds a different
    polynomial for every RHS, which is not the PFLARE GMRES polynomial being studied, and it makes
    the V-cycle nonlinear. Coarse solvers here must stay fixed linear operators: PFLARE's
    polynomial, or Richardson(k) + local GS.
- **KSPRichardson + PCSOR uses `PCApplyRichardson`.** It runs all `max_it` sweeps inside MatSOR
  without residual checks: exactly k block-GS sweeps on the coarse grid, which is what we want.
  For an outer solve it ignores rtol and reports CONVERGED_ITS. MatSOR_MPIAIJ with
  SOR_LOCAL_FORWARD_SWEEP does a ghost update plus a local sweep each iteration, which is block-GS
  Richardson.
- `local_backward` on a lower-triangular matrix has no strictly-upper part, so it degenerates to
  Jacobi.
- Unrelated test-driver failures at np=4 in the CI image (not investigated):
  - `adv_dg_upwind -dm_plex_simplex 0 -dm_refine 4` gives "No quadrature for celltype unknown".
  - `adv_dg_upwind -dm_plex_filename .../square_unstruc.msh -dm_refine 2 -curved_velocity` is
    SIGKILLed.
  - Both run at np=1 and np=2.

## Suggested next steps on LUMI
1. **Read the paper** and reproduce its baseline (auto-truncation level, iteration counts, solve
   times) on its test problems.
2. **Check whether a local GS sweep actually runs on the GPU.** I believe PETSc has no MatSOR for
   MATSEQAIJKOKKOS, so `-mg_coarse_pc_type sor` with aijkokkos probably falls back to the host;
   check `-log_view` for GpuToCpu copies. A device alternative for the *exact local triangular
   solve*: `-mg_coarse_pc_type bjacobi -mg_coarse_sub_pc_type ilu` (ILU(0)) with aijkokkos. For a
   local block that is lower triangular in its ordering, ILU(0) is exact (L = strictly lower
   scaled, U = diagonal), so this equals block-GS. PETSc's Kokkos ILU path uses Kokkos-Kernels
   SpTRSV on device. Combine it with `-mg_coarse_ksp_type richardson -mg_coarse_ksp_max_it k`
   and confirm the iteration counts match the SOR runs on CPU first.
3. **Weak-scale (P from 8 to 1000s of GCDs)** with k fixed and k ~ sqrt(P). Until step 4
   exists, `-pc_air_max_levels` sweeps are only an exploratory proxy for where a GS-aware
   auto-truncation would cut; the final comparison must use auto-truncation for both solvers. Measure iterations, solve time, and time per coarse application split
   into SpTRSV levels vs SpMV + halo. Compare against the paper's best truncated Newton-mf
   configuration.
   - Expected trade-off: GS iteration counts stay low when truncating, but each sweep costs about
     n_local^{1/2} dependent SpTRSV levels (2D).
   - Re-enabling processor agglomeration on coarse levels reduces P_active, and with it the
     nilpotency index.
   - Kokkos-Kernels SpTRSV options (level-set vs "chain" mode that merges small levels into one
     kernel) matter here.
4. **Implement it in PFLARE; this is needed to keep auto-truncation on:**
   - A coarsest option (e.g. `-pc_air_coarsest_local_gs`) that does Richardson(k) with the local
     block solve, with the topological reordering from section E on the fine level carried to
     the C-points.
   - Make the auto-truncation loop build and apply exactly that solver when it tests a level.
     Today it calls start/finish_approximate_inverse plus a MatMult residual check on the
     polynomial only.
   - Make the Newton/Arnoldi construction detect early breakdown, or fall back to (1-z)^k, when
     the preconditioned operator is near-nilpotent.

## Files here
- `env.sh`: container env (sets PYTHONPATH for petsc4py and pflare; T = tests dir).
- `run.sh np L "args" k-list`: dumps the coarse matrix at truncation L and runs `coarse_poly.py`.
  Set `EXE=adv_dg_upwind` for DG.
- `coarse_poly.py c.dat k-list ranges.txt`: coarse-solve comparison, topo reordering, SpTRSV level
  counts. Needs scipy.
- `ranges.py`: prints the per-rank ownership of the dumped matrix (run under `mpirun -n np`).
- `tri.py`: checks triangularity of the local blocks.
- `mg2.sh np grid k "Ls"`: Richardson V-cycle study, Newton-mf vs Richardson+local SOR, with times.

The scripts assume the CI image with the repo mounted at `/work/PFLARE` and this folder at
`/work/gs`. In the cloud session: `docker run -d --network host -v $PWD:/work/PFLARE
-v <this dir>:/work/gs stevendargaville/pflare:latest sleep infinity`, then `make build_tests`
inside the container. scipy was installed with pip.
