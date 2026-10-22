Codebase map
- `src/`: the algorithms are C `X.c`/`X.h` (C++ for Kokkos); a Fortran compiler is not needed. Many comments name the Fortran module each file was ported from (history in `docs/dev/fortran_removal_plan.md`). `src/Pflare_Random.c` is the random number generator (reproduces gfortran's `random_number` bitwise). PETSc PC plumbing: `PCAIR.c` (its `pc->data` is the `pc_air_multigrid_data` of `pflareimpl.h`; the PC operations call the former `PCShell` callbacks in `PCAIR_Shell.c`), `PCPFLAREINV.c`, `C_PETSc_Routines.c`. Kokkos kernels: `X.c` has a sibling `Xk.kokkos.cxx` exporting `<snake_case>_kokkos()` functions through a prototype header `Xk.h`, called from C directly.
- `src/fortran/`: the Fortran interface layer (public Fortran modules over the C API), compiled only when PETSc has its Fortran bindings (`PETSC_USE_FORTRAN_BINDINGS` in `petscconf.h`, read by the top `Makefile`).
- `tests/`: C test drivers, Fortran test drivers (`*.F90`, built and run only with PETSc's Fortran bindings; their run lines start with `$(SKIP_FORTRAN)`) + a Makefile of literal run commands. `python/`: Cython bindings. `include/pflare.h`: public C API. `docs/`: user docs.
- Magic tolerances/constants: `src/pflare_parameters.h`, exposed to Fortran as module `pflare_parameters` in `src/fortran/Pflare_Parameters.F90`. Object lists `COBJS`/`KOBJS`/`FOBJS` in the top `Makefile`; build order comes from `Makefile.deps` (`make depend`, which needs `makedepf90` for its Fortran lines).
- Root dir contains gitignored `*.mod` build artifacts (Fortran-enabled builds) — ignore them in listings.
- PETSc source is at `$PETSC_DIR/$PETSC_ARCH`; Kokkos source at `$PETSC_DIR/$PETSC_ARCH/externalpackages/git.kokkos{,-kernels}`. Both env variables must be set.

Read only when the task needs it
- `docs/dev/testing.md` — before adding or modifying tests
- `docs/dev/kokkos.md` — before touching `*.kokkos.cxx` or CPU/Kokkos debug-compare paths
- `docs/dev/ci.md` — when checking or reproducing a CI pipeline

Build
1. In top repo directory: `make -j3 build_tests`
2. If Python code changed, in the top repo directory: `make python`
3. Rule: fix all compile warnings (CI builds with `-Werror`).

Tests
1. Ensure $LD_LIBRARY_PATH matches the $PETSC_DIR and $PETSC_ARCH
2. Run the test targets below once. Trust `make`'s exit code: 0 means all tests passed; any failure breaks the run with a non-zero code and prints the error to the terminal. Don't re-run to grep the output.
3. In top repo directory: `make check`
4. In top repo directory: `make tests_short`
5. Run a specific class of tests only if needed: `make tests_search TEST_MATCH="<substring>"`, where substring is a command-line argument in the `tests/Makefile` (e.g. `-curved_velocity`).

If Kokkos code changed:
6. `export PETSC_OPTIONS="-mat_type aijkokkos -vec_type kokkos -dm_mat_type aijkokkos -dm_vec_type kokkos -on_error_abort"` → run Tests
7. `export PFLARE_KOKKOS_DEBUG=1` → run Tests
8. `unset PETSC_OPTIONS PFLARE_KOKKOS_DEBUG`

CI
1. If asked to check a CI pipeline, use the prebuilt Docker image the CI runs off; always pull the latest image before running. Details in `docs/dev/ci.md`.
