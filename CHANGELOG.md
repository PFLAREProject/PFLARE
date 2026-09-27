# Changelog

Notable changes to PFLARE are documented in this file. Entries reference the
GitHub pull request where the change was made. This file starts at v1.25.0;
for earlier changes please see the git history.

## Unreleased

- Fixed the shared library registration routine (called by PETSc when it
  loads PFLARE, e.g., with `--download-pflare`) only registering PCAIR, so
  `-pc_type pflareinv` now works without calling `PCRegister_PFLARE()`
- `PCAIRGetPolyCoeffs` / `PCAIRSetPolyCoeffs` (C, Fortran and Python) now
  error on an out of range level, a call before setup, an inverse with no
  stored polynomial, or (set) mismatched sizes, rather than reading or writing
  through invalid memory
- `PCAIRSetReuseAmount` / `-pc_air_reuse_amount` (and the Python
  `pcair_set_reuse_amount`) now reject values other than 1, 2 or 3 with
  `PETSC_ERR_ARG_OUTOFRANGE` (`ValueError` in Python); previously they were
  used unchecked to index the reuse tables
- Fixed `PCAIRSetDiagScalePolys` / `-pc_air_diag_scale_polys` being silently
  ignored while the inverse type was Neumann, which lost the setting for the C
  point inverse and for any later change of inverse type. `PCAIRGetDiagScalePolys`
  now returns the stored value (Neumann still always diagonally scales)
- Fixed the PCAIR C point smoother options (`-pc_air_c_inverse_type`,
  `-pc_air_c_poly_order`, `-pc_air_c_inverse_sparsity_order`): values set via
  the API are no longer overwritten by the F point values in
  `PCSetFromOptions`, and if unset they now follow the F point smoother values
  as documented, even without calling `PCSetFromOptions`
- `PCAIRSetSmoothType` / `-pc_air_smooth_type` (and the Python
  `pcair_set_smooth_type`) now error on smooth types longer than 10 characters
  (`ValueError` in Python) rather than silently truncating them
- The Python `pcair_*` / `pcpflareinv_*` wrappers now call the public C API and
  raise `PETSc.Error` when given a PC of the wrong type, rather than crashing
  (PCAIR) or silently returning a default value (PCPFLAREINV getters)
- Fixed the Fortran `PCPFLAREINVGetMatrixFree` not returning the stored value,
  and made the C/Fortran/Cython prototypes of the PCPFLAREINV and PCAIR bool
  routines match their definitions exactly (`PetscBool`, `int` by value)
- Fixed `-pc_pflareinv_matrix_free` bypassing `PCPFLAREINVSetMatrixFree`, so
  changing it after a setup (e.g. via `KSPSetFromOptions`) now resets the PC
  instead of reusing the old inverse in the wrong form and crashing
- Fixed PCPFLAREINV aborting with the GMRES polynomial types (power, arnoldi,
  newton, newton_no_extra) on operators with fewer rows than the polynomial
  order + 1 (e.g. small block Jacobi sub-blocks); the order is now clamped to
  the matrix size as in PCAIR, so `PCPFLAREINVGetPolyCoeffs` returns the
  clamped size
- Fixed PCPFLAREINV crashing with `-pc_pflareinv_type jacobi` or `wjacobi`
  and `-pc_pflareinv_matrix_free`; matrix-free is now ignored for the Jacobi
  types, as it already was in PCAIR
- Behaviour change: `PCPFLAREINVGetPolyCoeffs` now returns `NULL` and 0x0 for
  the non-polynomial PCPFLAREINV types (sai, isai, wjacobi, jacobi) instead of
  uninitialised memory, and coefficients set with `PCPFLAREINVSetPolyCoeffs`
  are discarded during setup for these types
- Fixed `PCPFLAREINVSetPolyCoeffs` reading freed memory when passed the
  pointer returned by `PCPFLAREINVGetPolyCoeffs`
- Fixed the DDC cleanup of the `pmisr_ddc` and `diag_dom` CF splittings (and
  `compute_diag_dom_submatrix`) leaving F points with a zero or missing
  diagonal but nonzero off-diagonals, which gave Aff a zero diagonal. These
  rows are now always made C points
- Fixed PCAIR with `-pc_air_strong_threshold 0` assuming Aff is diagonal for
  every CF splitting type; this only holds for `pmisr_ddc` and `diag_dom`, so
  other splittings (e.g., `pmis`, `agg`) silently dropped the off-diagonal
  entries of Aff, degrading or breaking convergence
- Fixed the `pmis_agg` CF splitting: in parallel, boundary C points whose
  strong neighbours were all off-process were turned into F points (which
  could remove every C point), and with Kokkos matrices the aggregation read
  uninitialised PMIS markers from the host
- Fixed the `pmis_dist2` CF splitting cancelling signed connections when
  squaring the strength matrix (e.g., skew-symmetric advection gave almost no
  connections and coarsening failed), and leaking an IS on every setup. The
  coarse grids from `pmis_dist2` may change
- Fixed PCAIR auto truncation keeping the coarse grid polynomial coefficients
  from a level that failed the truncation test, which gave the wrong coarse
  grid polynomial (or an out-of-bounds write with the power basis) when the
  coarsest grid had fewer rows than the coarse polynomial order
- Fixed the PCAIR complexities (`-pc_air_print_stats_timings` and
  `PCAIRGet*Complexity`) being zero or NaN when the hierarchy has only a single
  level (the Jacobi fallback or auto truncation on the top level)
- Fixed a leak of the near-nullspace vectors with `-pc_air_symmetric
  -pc_air_constrain_w`; `-pc_air_constrain_w` is ignored with
  `-pc_air_symmetric` as the prolongator is R^T
- Fixed an MPI communicator leak on every setup with `-pc_air_subcomm` when
  some ranks have no rows on a level
- Fixed a leak of the near-nullspace vectors with `-pc_air_constrain_z` or
  `-pc_air_constrain_w` when the hierarchy is capped by `-pc_air_max_levels`
- Behaviour change: fixed PCAIR processor agglomeration going one
  agglomeration factor further than needed, so more cores may now stay active
  on coarse levels
- PETSc errors inside the C routines that build the `-pc_air_subcomm` matrices
  and check whether Aff is diagonal now abort, rather than being ignored
- Fixed PCAIR on non-Kokkos GPU matrix types (e.g. `aijcusparse`,
  `aijhipsparse`) building its F/C point injectors from the not yet created
  Afc/Acf submatrices, which could crash or corrupt the first setup
- Fixed the CPU `remove_from_sparse_match` with lumping discarding the
  existing values of the output matrix: with alpha it now computes
  output += alpha * input (as the Kokkos version does) and entries of the
  output that are not in the input's sparsity are kept. The C
  `remove_from_sparse_match` now also initialises the PETSc Fortran
  interface, as the other standalone C routines do
- Fixed PCAIR crashing on the first F smooth with Kokkos vectors but a
  non-Kokkos matrix type (e.g. `-vec_type kokkos -mat_type aij`)
- Fixed PCAIR leaking one full-size injector matrix per level on every
  reset/destroy with non-Kokkos GPU matrix types and F-point only smoothing
- Fixed the PCAIR block apply (`KSPMatSolve`) passing host pointers to a
  device kernel when given host `MATDENSE` blocks with Kokkos matrices on a
  CUDA/HIP build; it now falls back to a host copy
- The Kokkos one-point prolongator now breaks ties between equal maximum
  entries on the smallest column, as the CPU version does, so on GPUs the
  prolongator (and iteration counts) may change and now match the CPU
- Fixed a crash when an assembled GMRES polynomial or SAI/ISAI inverse (in
  PCPFLAREINV or PCAIR) was set up in parallel after one had been set up on a
  single rank in the same program, eg as a block Jacobi sub-PC
- Behaviour change: the Arnoldi basis GMRES polynomial (the default PCAIR
  inverse type) now includes every entry of the least-squares residual in its
  early termination test, so it no longer stops before reaching its tolerance
  and iteration counts may change slightly
- Fixed monomial GMRES polynomials with exactly zero interior coefficients
  (eg set with `PCPFLAREINVSetPolyCoeffs`) applying the wrong polynomial, both
  assembled and matrix-free
- Newton basis GMRES polynomials: the modified Leja ordering of the roots no
  longer depends on the scaling of the spectrum, and purely imaginary roots now
  receive extra roots for stability, so results may change for tightly
  clustered, small magnitude or skew-symmetric spectra
- Fixed the CPU ISAI exact dense solve wrongly permuting the solution returned
  by LAPACK `gesv`, which gave an incorrect ISAI (and lAIR Z) whenever a local
  submatrix needed row pivoting; a singular local submatrix now aborts rather
  than silently returning the right-hand side
- Fixed the Kokkos dense direct solve used to build lAIR/SAI Z and ISAI
  rows not pivoting, which gave Inf/NaN or inaccurate rows when a local block
  had a zero or small leading pivot (e.g. matrices with zero diagonals); it now
  uses LU with partial pivoting like the CPU LAPACK solve
- Fixed the Kokkos Jacobi approximate local solves used to build lAIR/SAI Z
  and ISAI rows dividing by zero (giving Inf/NaN) when a local block has a
  zero diagonal; zero diagonals are now replaced by 1, matching the CPU PCJACOBI

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
