# Changelog

Notable changes to PFLARE are documented in this file. Entries reference the
GitHub pull request where the change was made. This file starts at v1.25.0;
for earlier changes please see the git history.

## Unreleased

- Fixed `-pc_air_full_smoothing_up_and_down` with an assembled inverse applying
  the wrong smoother, which stopped it converging. The level smoothers there are
  a `PCMAT` whose Pmat is our approximate inverse, so `PCApply` has to be a
  matmult with it. Since PETSc 3.21 `PCSetUp_Mat` instead defaults to `MatSolve`
  whenever the Pmat has one, which matdiagonal does, so a diagonal approximate
  inverse was being applied as a solve, i.e. multiplied by its own inverse. PCAIR
  now asks for `MATOP_MULT` explicitly with `PCMatSetApplyOperation`, both for
  the level smoothers and for the single level auto-truncated case. This only
  affected the assembled inverses; `-pc_air_matrix_free_polys` goes through a
  PCSHELL and was always correct
- Fixed `-pc_air_full_smoothing_up_and_down` ignoring
  `-pc_air_inverse_sparsity_order` and assembling the inverse of each level
  matrix as a diagonal. When the F-F block `A_ff` is diagonal, which it is on
  many problems with a good CF splitting, PCAIR takes a shortcut and drops the
  sparsity order of the `A_ff` inverse to zero. In full smoothing mode that same
  polynomial data does not invert `A_ff` at all, it inverts the whole level
  matrix, so the shortcut was forcing the inverse of the entire level operator
  to sparsity order zero, i.e. to a diagonal, no matter what
  `-pc_air_inverse_sparsity_order` was set to. The shortcut is now confined to
  the `A_ff` inverse used for the grid-transfer operators, and full smoothing
  builds its inverse with the requested sparsity order and the full level matrix
  size. The assembled full smoothing V-cycle is much stronger as a result: on
  `adv_1d -n 1000` it goes from 74 iterations to 14, and
  `-pc_air_inverse_sparsity_order 2`, which used to make no difference at all,
  now gives 8
- PCPFLAREINV now implements `PCApplyTranspose`, so it can be used as the
  preconditioner in a `KSPSolveTranspose`. This works for every inverse type,
  both assembled and matrix-free. It applies the exact transpose of what
  `PCApply` applies rather than a separately computed approximate inverse of
  `A^T`, so the two really are adjoints. The matrix-free polynomials do this by
  applying the same polynomial to a virtual transpose of the operator, which is
  valid as the coefficients are real, so `q(A)^T = q(A^T)`; that means the
  operator itself has to support `MatMultTranspose`, which is checked with a
  clear error for the case where the user has supplied their own MatShell as the
  operator. `PCMatApplyTranspose` is not implemented directly, so it falls back
  to PETSc applying `PCApplyTranspose` column by column
- Fixed PCAIR not passing its options prefix down to its inner PCMG. The
  `-mg_coarse_*` / `-mg_levels_*` options were read unprefixed by every PCAIR
  in a program, and `-foo_mg_coarse_*` for a PCAIR with prefix `-foo_` was
  silently ignored. The level KSPs now carry the PCAIR prefix, so a prefixed
  PCAIR reads `-foo_mg_coarse_*` and no longer reads the unprefixed options
- Behaviour change: PCAIR now always symmetrizes the strength matrix used to
  compute the CF splitting. Previously `-pc_air_symmetric` skipped the
  symmetrization, so the CF splittings and hence the results of existing runs
  with that flag may change; `-pc_air_symmetric` now only controls whether the
  prolongator is defined as R^T. The second argument of the standalone
  `compute_cf_splitting` still controls the symmetrization with unchanged
  polarity, but has been renamed from `symmetric` to `skip_symmetrize` to say
  what it actually does - existing callers are unaffected other than the
  Python keyword argument name
- Fixed the Kokkos PMISR CF splitting producing different results to the CPU
  implementation on structurally nonsymmetric strength matrices. This was
  previously reachable through `-pc_air_symmetric`, which fed PMISR an
  unsymmetrized strength matrix, and was caught by the `PFLARE_KOKKOS_DEBUG`
  CPU/Kokkos cross-check. Since PCAIR itself now always symmetrizes, a new
  `tests/pmisr_nonsymmetric` driver keeps this path covered by calling
  `compute_cf_splitting` with `skip_symmetrize` true on a nonsymmetric
  operator; both the serial and parallel halo bugs abort the cross-check
  if reintroduced
- New `PCMatApply` callback for PCPFLAREINV, so a block of right-hand sides is
  applied with a single sparse matrix by dense matrix product rather than
  PETSc's default loop over columns; on the GPU backends this keeps the whole
  multiple-RHS apply on the device. Note that `KSPMatSolve` only reaches
  `PCMatApply` for `-ksp_type preonly`, HPDDM or `-ksp_type richardson` (which
  iterates blockwise through `PCMatApplyRichardson`) - any other KSP type falls
  back to solving column by column. New `tests/adv_1d_multi_rhs.c` driver
  builds the block of right-hand sides with `MatCreateDenseFromVecType` and
  solves with `KSPMatSolve`
- The matrix-free polynomial inverses (`power`, `arnoldi`, `newton`,
  `newton_no_extra`, `neumann`) now also apply blockwise in `PCMatApply`: the
  Horner and Newton-basis kernels run sparse matrix by dense matrix products
  on the underlying operator instead of looping matvecs, with dense scratch
  blocks cached in the matshell context. The kernel products stay attached to
  that scratch between applies (the Horner iteration ping-pongs between two
  cached temporaries rather than copying into the output each order), so
  repeat block applies only run the numeric phase of each product. The right diagonally scaled variant
  `q(D^-1 A) D^-1` used by PCAIR's smoothers is supported, as is the Neumann
  polynomial's intrinsically scaled `q(I - D^-1 A) D^-1`; anything else
  unsupported falls back to the previous column-by-column apply. New
  `tests/shell_block_apply.c` driver exercises the block apply directly,
  including the diagonally scaled mode that PCPFLAREINV itself never builds
- New `PCMatApply` callback for PCAIR, so `KSPMatSolve` with
  `-ksp_type preonly` applies the whole AIR hierarchy to a block of right-hand
  sides at once; every product goes through the `MatProduct` API so dense
  blocks on the GPU backends use real SpMM kernels and stay on the device.
  PCMatApply forwards to the underlying PCMG, with PFLARE supplying the block
  FC smoother, the block coarse solve and the dense-block F/C row extraction,
  and dense scratch blocks cached per level. The smoother products are kept
  attached to the per-level scratch, so each smoother iteration only runs the
  numeric phase of each product rather than recreating it. This needs a PETSc main new
  enough to have `PCMatApplyRichardson` and the `PCShellSetMatApply`/
  `PCShellSetMatApplyRichardson` Fortran bindings (merged August 2026). All the
  PCAIR options are covered blockwise in `tests/adv_1d_multi_rhs.c`, in serial
  and parallel, along with `-ksp_type richardson` block solves through
  `KSPMatSolve`
- Fixed PCAIR silently ignoring `KSPSetReusePreconditioner` /
  `-ksp_reuse_preconditioner`: a values-only change to the pmat between
  solves rebuilt the full AIR hierarchy despite the flag; a frozen PCAIR
  now stays frozen (even across sparsity pattern changes, matching PETSc
  semantics) until the flag is unset (#264)
- New compatible relaxation CF splitting (`-pc_air_cf_splitting_type cr`),
  which coarsens from scratch with no strength matrix until one application
  of AIR's F-point smoothing contracts a random error on Aff by the target
  rate given by the strong threshold; the CR relaxation mirrors the PCAIR
  Aff inverse settings and works in serial, parallel and with Kokkos (#261)
- Fixed a latent segfault in `calculate_and_build_approximate_inverse` when
  called without the optional coefficients argument (#261)
- Minimum PETSc version is now 3.25.0; the C/Fortran interface was rewritten to
  use PETSc's native Fortran types instead of a custom ISO C binding shim
- Allow a user-supplied coarse-grid solver in PCAIR via the standard PETSc
  `-mg_coarse_*` options, e.g., `-mg_coarse_pc_type lu` (#256)
- Support for solving sparse triangular systems from ILU factorisations with
  AIRG, with new `tests/ilu_factors.c` examples (#236, #237, #238)
- PFLARE manual pages are now generated as part of the PETSc documentation and
  hosted on petsc.org (#244, #249)
- Support for PETSc built without MPI (MPIUNI) (#243)
- Support for PETSc built in single precision (#247)
- Expose the assembled approximate inverse matrix from PCPFLAREINV (#239)
- Kokkos: reduced memory use in the SAI/ISAI/lAIR SAI iterative kernels, with a
  sparse Aff form used for large row-wise systems (#258)
- Kokkos: assembled Newton polynomial inverses (#248), iterative lAIR (#234),
  removal of global state (#235), and several other GPU kernel improvements
- New DG upwind and CG SUPG advection(-diffusion) test drivers built on
  DMPlex, including 3D and curved velocity cases (#227, #228, #229, #231, #232)

## [v1.26.0] - 2026-03-23

- Fixed a segfault in parallel Kokkos runs (#225)
- SAI/ISAI smoothing and lAIR SAI grid transfers on GPUs with Kokkos (#223)
- PMISR-DDC CF splitting improvements (#220, #222) and reduced communication
  in the lAIR submatrix extraction (#221)
- PCPFLAREINV: get/set the polynomial coefficients from C/Fortran/Python
  (#198, #199, #200, #201)
- New Jupyter notebook tutorials in `notebooks/`, runnable via Binder
  (#192, #195), and the documentation split into separate pages under `docs/`
  (#193)
- Parser for `-pc_air_print_stats_timings` output (#218)
- Several memory leak and valgrind fixes, with valgrind added to CI
  (#204, #205, #206, #207, #208, #212)
- Faster Arnoldi GMRES polynomial setup using VecMDot (#196) and reduced
  compile times (#197)

## [v1.25.1] - 2025-11-14

- Reduced use of PETSc private headers (#156)
- C++20 compatibility for the Kokkos kernels (#155)
- Fixed the Kokkos level 1 scratch memory size calculation (#154)

## [v1.25.0] - 2025-10-29

- macOS builds and CI (#140, #142)
- Fixed CUDA/Kokkos build and link flags, allowing user flags to be appended
  to the PETSc ones (#138, #139)
- Several memory leak fixes (#149, #150) and a malloc dump CI check (#151)
- Python/Cython build fixes, including `PYTHONPATH` handling (#147, #148)
- `make check` now errors out on failure (#136)

[v1.26.0]: https://github.com/PFLAREProject/PFLARE/releases/tag/v1.26.0
[v1.25.1]: https://github.com/PFLAREProject/PFLARE/releases/tag/v1.25.1
[v1.25.0]: https://github.com/PFLAREProject/PFLARE/releases/tag/v1.25.0
