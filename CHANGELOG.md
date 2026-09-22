# Changelog

Notable changes to PFLARE are documented in this file. Entries reference the
GitHub pull request where the change was made. This file starts at v1.25.0;
for earlier changes please see the git history.

## Unreleased

## [v1.27.0]

- Behaviour change: PCAIR now always symmetrizes the strength matrix used to
  compute the CF splitting, so results with `-pc_air_symmetric` may change;
  that option now only controls whether the prolongator is R^T (#266)
- The second argument of the standalone `compute_cf_splitting` has been
  renamed from `symmetric` to `skip_symmetrize`; its meaning is unchanged, so
  only the Python keyword argument name is affected (#266)
- Fixed PCAIR not passing its options prefix down to its inner PCMG: a PCAIR
  with prefix `-foo_` now reads `-foo_mg_coarse_*` / `-foo_mg_levels_*` and no
  longer reads the unprefixed options (#273)
- New `PCMatApply` for PCPFLAREINV, so `KSPMatSolve` applies a block of
  right-hand sides with a single sparse matrix by dense matrix product; the
  matrix-free polynomial inverses also apply blockwise (#265)
- New `PCMatApply` for PCAIR, so the whole AIR hierarchy is applied to a block
  of right-hand sides at once and stays on the device on GPUs. This needs a
  PETSc with `PCMatApplyRichardson` and the `PCShellSetMatApply` Fortran
  bindings (merged August 2026) (#269)
- The symbolic phase of the block apply products is cached, so repeat block
  applies only run the numeric phase (#265, #269, #277)
- PCPFLAREINV now implements `PCApplyTranspose` for every inverse type, both
  assembled and matrix-free, so it can be used in a `KSPSolveTranspose`. The
  matrix-free polynomials need the operator to support `MatMultTranspose`
  (#278)
- New compatible relaxation CF splitting (`-pc_air_cf_splitting_type cr`),
  which works in serial, parallel and with Kokkos (#261)
- Allow a user-supplied coarse-grid solver in PCAIR via the standard PETSc
  `-mg_coarse_*` options, e.g., `-mg_coarse_pc_type lu` (#256)
- Support for solving sparse triangular systems from ILU factorisations with
  AIRG, with new `tests/ilu_factors.c` examples (#236, #238)
- New `PCAIRGet*Complexity` routines return the grid, operator, cycle, storage
  and reuse storage complexities, also available from Python (#237)
- Expose the assembled approximate inverse matrix from PCPFLAREINV with
  `PCPFLAREINVGetInverseMat` (#239)
- `-pc_air_constrain_z` / `-pc_air_constrain_w` are now safeguarded against
  degenerate near-nullspace vectors, which previously gave NaNs or LAPACK
  crashes (#267, #268)
- Minimum PETSc version is now 3.25.0; the C/Fortran interface was rewritten to
  use PETSc's native Fortran types instead of a custom ISO C binding shim
  (#240, #246, #252)
- PFLARE no longer needs the vendored PETSc source and builds against a
  standard PETSc install (#245)
- Support for PETSc built without MPI (MPIUNI) (#243, #251)
- Support for PETSc built in single precision (#247)
- PFLARE is now available through Spack
- New `VERSION.txt` and matching `PFLARE_VERSION_*` macros in `pflare.h`,
  along with this changelog and GitHub issue forms (#260)
- Kokkos: reduced memory use in the SAI/ISAI/lAIR SAI iterative kernels, with a
  sparse Aff form used for large row-wise systems (#258)
- Kokkos: assembled Newton polynomial inverses (#248), iterative lAIR (#234),
  removal of global state (#235), and several other GPU kernel improvements
- The matrix-free Horner apply no longer does a vector copy per order (#276)
- Fixed PCAIR silently ignoring `KSPSetReusePreconditioner` /
  `-ksp_reuse_preconditioner` and rebuilding the hierarchy when only the
  matrix values changed (#264)
- Fixed `-pc_air_full_smoothing_up_and_down` with an assembled inverse not
  converging, and ignoring `-pc_air_inverse_sparsity_order` (#280)
- Fixed the Kokkos PMISR CF splitting producing different results to the CPU
  on structurally nonsymmetric strength matrices (#266)
- Fixed operator reference counting in the one level PCAIR paths when Amat
  and Pmat are different matrices (#275)
- Kokkos: added missing fences after asynchronous host to device copies and
  fixed several view/index bugs (#263); the matrix sort is now only applied
  with local column indices (#262)
- PMISR now rejects `max_luby_steps=0`, which could loop forever (#257)
- Fixed a latent segfault in `calculate_and_build_approximate_inverse` when
  called without the optional coefficients argument (#261)
- Fixed the documented pip install of the Python module failing to link
  against `libpflare` (#279)
- PFLARE manual pages are now generated as part of the PETSc documentation and
  hosted on petsc.org, with option defaults shown (#244, #249, #250)
- Manually triggered GPU CI tests (#271, #272)
- New DG upwind and CG SUPG advection(-diffusion) test drivers built on
  DMPlex, including 3D and curved velocity cases (#227, #228, #229, #231, #232)

## [v1.26.0]

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

## [v1.25.1]

- Reduced use of PETSc private headers (#156)
- C++20 compatibility for the Kokkos kernels (#155)
- Fixed the Kokkos level 1 scratch memory size calculation (#154)

## [v1.25.0]

- macOS builds and CI (#140, #142)
- Fixed CUDA/Kokkos build and link flags, allowing user flags to be appended
  to the PETSc ones (#138, #139)
- Several memory leak fixes (#149, #150) and a malloc dump CI check (#151)
- Python/Cython build fixes, including `PYTHONPATH` handling (#147, #148)
- `make check` now errors out on failure (#136)

[v1.27.0]: https://github.com/PFLAREProject/PFLARE/releases/tag/v1.27.0
[v1.26.0]: https://github.com/PFLAREProject/PFLARE/releases/tag/v1.26.0
[v1.25.1]: https://github.com/PFLAREProject/PFLARE/releases/tag/v1.25.1
[v1.25.0]: https://github.com/PFLAREProject/PFLARE/releases/tag/v1.25.0
