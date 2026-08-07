# Changelog

Notable changes to PFLARE are documented in this file. Entries reference the
GitHub pull request where the change was made. This file starts at v1.25.0;
for earlier changes please see the git history.

## Unreleased

- New `PCMatApply` callback for PCPFLAREINV, so a block of right-hand sides is
  applied with a single sparse matrix by dense matrix product rather than
  PETSc's default loop over columns; on the GPU backends this keeps the whole
  multiple-RHS apply on the device. Note that `KSPMatSolve` only reaches
  `PCMatApply` for `-ksp_type preonly` (or HPDDM) - any other KSP type falls
  back to solving column by column. PCAIR does not support multiple right-hand
  sides yet. New `tests/adv_1d_multi_rhs.c` driver builds the block of
  right-hand sides with `MatCreateDenseFromVecType` and solves with
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
