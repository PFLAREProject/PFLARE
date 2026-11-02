# Multi-shift block V-cycle for PCAIR: feasibility prototype

Prototype only, not for `main`. Driver: `tests/adv_diff_fd_multishift.c`. Code: `src/AIR_MultiShift.F90` plus the hooks listed below.
The cycle is PCAIR's normal PCMG Kaskade cycle. There is no separate hand-written cycle and no change to PETSc.
All numbers come from the `stevendargaville/pflare:latest` CI image on one CPU core (2D upwind advection, `D_j = c_j (1 + 0.5 sin(row))`, hierarchy built at `c = 0`).

## What it does

`PCAIRSetMultiShift(pc, Dshift)` takes a dense n x k block whose column j is `diag(D_j) - diag(Pmat_base)`.
After that, `PCMatApply` on k columns preconditions column j with exactly the hierarchy that a single-RHS diagonal update to `A + D_j` with `-pc_air_reuse_poly_coeffs` would build.
Every sparse matrix product is shared between the columns. The per-column terms are:

| Component | Single-RHS update (in place) | Multi-shift (per column, shared SpMM) |
|---|---|---|
| `A_ff`, `A_cc` | set the diagonal | `A_ff X + S .* X`, with S = shifted diagonal minus current diagonal |
| Restrictor | scale R's F columns by `min(1, a/(a+d))` | `R (C .* X)`, with C = that scaling relative to the current R |
| Coarse shifts | `R (d .* P1)`, SpMV | `R^(j) (D .* P1)`, one SpMM per level at setup |
| Polynomial smoothers and coarse solve | same coefficients, new matrix | the same coefficients applied to `A + diag(S_j)`. Diagonal scaling uses a dense reciprocal block |

`setup_air_multishift` is the batched analogue of `update_air_diag`. Because the shifts are stored relative to the *current* hierarchy, a later diagonal update recomputes them automatically.

## Correctness

The test checks the multi-shift on k = 8 columns against `PCApply` after a single-RHS update for each column. It errors above 1e-8.

| Configuration | max relative difference |
|---|---|
| default (power basis, F smoothing), serial | 2.0e-16 |
| newton / arnoldi / neumann | 2.1e-16 / 2.0e-16 / 1.6e-16 |
| `ffc`, `fcf`, `fc` C-point smoothing | 2.4e-16 |
| `-pc_air_symmetric` (R = P^T, no R scaling) | 3.9e-16 |
| 2, 3 and 4 ranks, including `-pc_air_process_eq_limit 400` agglomeration | 2.4e-16 to 3.1e-16 |
| zero shifts vs plain PCMG `PCMatApply` (restrictor shells installed) | 0 (bitwise) |
| after a diagonal update of the multi-shift hierarchy itself | 1e-13 |
| cleared multi-shift vs plain PCMG (original restrictors restored) | 2e-16 |

The "after an update" row is 1e-13 rather than 1e-16 because it compares against a multi-shift recomputed from a different hierarchy state.

## Iterations (n = 200², k = 16, right-preconditioned GMRES, rtol 1e-10)

| c | sequential update | multi-shift | rebuild (fresh) | update with recomputed coefficients |
|---|---|---|---|---|
| 0 | 6 | 6 | 6 | 6 |
| 1e-3 | 9 | 9 | 7 | 9 |
| 1e-2 | 15 | 15 | 6 | 15 |
| 1e-1 | 20 | 20 | 6 | 20 |
| 1 | 22 | 22 | 4 | 23 |
| 10 | 15 | 15 | 3 | 15 |
| 1e2 | 8 | 8 | 3 | 7 |
| 1e3 | 5 | 5 | 2 | 5 |

The multi-shift iteration counts equal the sequential ones in **every column for every k in {1, 4, 8, 16, 32, 64}**. The two rightmost columns come from `adv_diff_fd_diag_update` with a ones rhs, so they are indicative only.

**Phase 1 vs Phase 2.** Recomputing the coefficients per shift gives the same counts as keeping them. On this problem the gap to a rebuild is therefore caused by the frozen hierarchy (CF splitting, Z, R and P built at c = 0), not by sharing the coefficients. Per-column or per-band coefficients (Phase 2) would not recover it. Several anchor hierarchies split by shift range would.

## Wall time (n = 200², one core)

| k | k x PCApply (ms) | multi-shift PCMatApply (ms) | shared-shift PCMatApply (ms) | speedup vs k PCApply | overhead vs shared | k updates (ms) | PCAIRSetMultiShift, first call (ms) |
|---|---|---|---|---|---|---|---|
| 1 | 4.1 | 4.9 | 4.7 | 0.84x | +3% | 2.6 | 10.4 |
| 4 | 16.5 | 9.3 | 7.7 | 1.77x | +21% | 9.6 | 27.0 |
| 8 | 39.8 | 19.9 | 14.6 | 2.00x | +36% | 21.2 | 16.6 |
| 16 | 70.2 | 51.3 | 33.0 | 1.37x | +55% | 46.4 | 113.4 |
| 32 | 139.0 | 110.6 | 72.1 | 1.26x | +53% | 83.8 | 316.6 |
| 64 | 273.7 | 245.8 | 178.0 | 1.11x | +38% | 157.0 | 560.2 |

The `PCAIRSetMultiShift` column is the first call after a clear, so it includes allocating the dense per-level blocks and is noisy. Repeated calls at k = 16 take a steady 20 ms, against 40 ms for the 16 diagonal updates.

The 20–55% overhead over the shared-shift block apply comes from the extra dense passes. These are host loops: `+ S .* X` after each product, `.* recip`, and `.* C` before each restriction. With 3–5 nonzeros per row these passes move about as much memory as the SpMM itself. A fused "SpMM + diagonal shift (+ scale)" kernel would remove most of this.

Full batched solves are currently about 2x slower than sequential solves. The cause is the test driver's batched GMRES, not the preconditioner: it freezes converged columns instead of deflating them (k = 64 ran 1472 column-applies where 858 were needed), and its Gram–Schmidt is a host loop. Raw log: `sweep_n200_cpu1.log`.

## Outer Richardson: entirely within PETSc

PETSc's block Richardson (`KSPMatSolve_Richardson`) only ever applies Amat through `MatProduct`. So a `MATSHELL` Amat whose dense product is `L X + D .* X` (`MatShellSetMatProductOperation`), combined with the multi-shift PCAIR, gives a batched multi-shift solve with **no driver loop and no PETSc change**. This is `-richardson` in the driver. The sequential comparison is an update plus a `KSPSolve` with `KSPRICHARDSON` for each shift. n = 200², one core, rtol 1e-10:

| k | block its | max seq its | column-applies, block / needed | seq Richardson (s) | multi-shift Richardson (s) | seq GMRES (s) | max column rel. residual |
|---|---|---|---|---|---|---|---|
| 1 | 69 | 69 | 69/69 = 1.00 | 0.32 | 0.41 | 0.12 | 9.6e-11 |
| 4 | 66 | 68 | 264/86 = 3.07 | 0.39 | 0.77 | 0.22 | 1.7e-10 |
| 8 | 64 | 67 | 512/199 = 2.57 | 0.90 | 1.45 | 0.50 | 2.3e-10 |
| 16 | 65 | 70 | 1040/454 = 2.29 | 2.04 | 3.37 | 1.07 | 3.3e-10 |
| 32 | 67 | 73 | 2144/965 = 2.22 | 4.22 | 9.41 | 2.34 | 3.9e-10 |
| 64 | 67 | 74 | 4288/1989 = 2.16 | 9.08 | 17.89 | 4.55 | 5.3e-10 |

- **Iterations.** Richardson needs up to 74 iterations where GMRES needs 22–23, so GMRES remains the better outer method for these shifts.
- **Stopping test.** PETSc's block Richardson stops on `||R||_F <= rtol ||B||_F`. The block therefore stops a few iterations before the slowest column would on its own (65 vs 70 at k = 16), and that column ends up to about 5x above rtol (5.3e-10). The fast columns over-converge to about 1e-14. Only `||r_j|| <= rtol ||B||_F` is guaranteed, which the test checks.
- **Wasted work.** Every column runs every block iteration. The shifts converge in anything from 5 to 74 iterations, so the block does 2.2–3x the column-applies needed. On the CPU, the 1.1–2x gain per apply doesn't cover that, and the block is 1.6–2.2x slower than sequential. With shifts grouped into bands of similar convergence, the waste would largely disappear.
- **Small PETSc change that would help.** A per-column stopping test in `KSPMatSolve_Richardson`, plus compacting the unconverged columns into a smaller block (`MatDenseGetSubMatrix` after a column permutation). The same deflation is what the batched GMRES needs.

Raw log: `sweep_richardson_n200_cpu1.log`.

## What has to change

### In PETSc: nothing for the cycle

PCMG's block Kaskade only needs three things from PFLARE, and all three can be supplied through existing APIs:

- **Restriction.** `MatMatRestrict` → `MatMatInterpolateAdd` → `MatMatMult(R, B)` goes through `MatProduct`. Setting a `MATSHELL` restrictor with `MatShellSetMatProductOperation(MATPRODUCT_AB, ..., MATDENSE, MATDENSE)` lets the block product compute `R (C .* B)`. `MatMult` and friends forward to R, so the single-RHS and transposed cycles are unchanged. This is `create_multishift_restrictor_c` in `src/C_PETSc_Routines.c`. It is swapped in with `PCMGSetRestriction` when a multi-shift is set, and swapped out when it is cleared.
- **Smoothers.** These are already PFLARE `PCShell` objects (`mg_FC_block_richardson`), which read the shifts from `air_data`. Kaskade computes no level residuals, so PCMG never needs the shifted level operators.
- **Coarse solve.** This is already a PFLARE shell (`mg_coarse_shell_block_apply`).

This is the only route in the prototype. `PCAIRSetMultiShift(pc, NULL)` puts the original restrictors back.

### In PETSc: worth changing, though not required

1. **Outer Krylov (the main gap).** An outer Richardson already works unchanged; see the section above. `KSPMatSolve` with a non-block KSP falls back to one `KSPSolve` per column. Each of those calls `MatMult` on Amat, which cannot know which shift the column belongs to, and a single-RHS `PCApply`. Block methods (HPDDM BGMRES, block CG) mix the columns, which is wrong for different operators. What is needed is a *batched, non-block* matsolve: k independent Arnoldi processes in lockstep, using `MatProduct` on Amat (a shell can do `L X + D .* X`) and `PCMatApply`, with per-column convergence and deflation. Either add that to `KSPGMRES`/`KSPFGMRES`, or check whether HPDDM's non-block GMRES with several right-hand sides does exactly this. HPDDM is not in the CI image, so I couldn't test that.
2. **Dense column kernels.** A dense Hadamard product (`MatPointwiseMult` for `MATDENSE*` on all backends) and column-wise dot products (currently only `MatGetColumnNorms`). PFLARE can supply its own Kokkos kernels instead.
3. **Optional tidy-up.** A PCMG per-level pre-restriction hook (or a dense-Mat `rscale` for `matapp`) would remove the shell wrapper and its extra copy of the coarse block.

### In PFLARE (prototype, this branch)

- `Gmres_Poly_Apply.F90`: optional `shift_mat` / `recip_mat` arguments on `petsc_horner_block`, `petsc_newton_block` and `shell_poly_block_apply`; dense Hadamard helpers.
- `FC_Smooth_Block.F90`: optional shifts on `f_smooths_block` / `c_smooths_block`; `apply_inverse_block_shifted`; `fc_smooth_multishift`; multi-shift branches in `mg_FC_block_richardson` and `mg_coarse_shell_block_apply`.
- `AIR_MultiShift.F90` (new): `setup_air_multishift`, the batched analogue of `update_air_diag`.
- `AIR_Data_Type(_Routines).F90`: the per-level dense blocks and their teardown on reset.
- `PCAIR_Shell.F90`: set, clear and recompute after an update; installs and removes the PCMG restrictor shells.
- `C_Fortran_Bindings.F90`, `PCAIR.c`, `include/pflare.h`: `PCAIRSetMultiShift(PC, Mat)`.
- `C_PETSc_Routines.c`: the restrictor shell.

### For production

- Kokkos kernels for `Y += S .* X`, `Y .*= R`, and the fused shifted-scaled product. Ideally fuse the shift into the SpMM. Run the `PFLARE_KOKKOS_DEBUG` comparisons; the Kokkos and GPU paths are untested here.
- Requirements and limits: matrix-free polynomial smoothers and coarse solve, because an assembled inverse bakes one matrix into its entries. Jacobi inverses would be easy to add, since they are pure Hadamard terms. A direct coarse solve can't be shared across shifts. Full smoothing up and down is not supported, as with the update.
- Kept coefficients need `-pc_air_diag_scale_polys`. Without diagonal scaling the existing single-RHS update already overflows at c ≥ 1e2 (it trips `-fp_trap` in `adv_diff_fd_diag_update` too). The driver therefore defaults to scaled polynomials.
- Memory: per level, `S_f` and `recip_f` (n_f x k), `C` (n x k; only the F rows are needed), and `S_c`/`recip_c` when C-smoothing, on top of the existing block temporaries.
- Fortran and Python bindings, and docs.
- The batched outer solver, with deflation of converged columns.

## Bug found during development

While building this, the new code had `min(1, real(x))`. In Fortran, `real()` with no kind returns the default (single precision) real, which caused 1e-9 discrepancies. It is fixed. The bug was in the new prototype code only; a grep of `src/` found no other instances.
