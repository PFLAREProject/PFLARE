Codebase map
- `src/`: the algorithms are being ported from Fortran `.F90` (one module per file; module name = lowercase filename) to C `X.c`/`X.h` with the same stem, one module at a time (`docs/dev/fortran_removal_plan.md`). Fortran modules not yet ported call the C ports through the `bind(C)` interfaces in `src/fortran/Port_Interfaces.F90`; `src/fref/` keeps the retired Fortran for the `make PFLARE_PORT_DEBUG=1` differential checks. PETSc PC plumbing: `PCAIR.c`, `PCPFLAREINV.c`, `C_PETSc_Routines.c`. Kokkos kernels: `X.F90`/`X.c` has a sibling `Xk.kokkos.cxx` exporting `<snake_case>_kokkos()` functions, called from Fortran via ISO-C bindings in `C_PETSc_Interfaces.F90`/`C_Fortran_Bindings.F90` and from C directly.
- `tests/`: test drivers + a Makefile of literal run commands. `python/`: Cython bindings. `include/pflare.h`: public C API. `docs/`: user docs.
- Magic tolerances/constants: `src/pflare_parameters.h` (shared by C and Fortran), exposed to Fortran as module `pflare_parameters` in `src/fortran/Pflare_Parameters.F90`. `src/fortran/` holds the Fortran interface layer (public modules). Object lists `COBJS`/`KOBJS`/`FCORE_OBJS`/`FOBJS` in the top `Makefile`; build order comes from `Makefile.deps` (`make depend`).
- Root dir contains gitignored `*.mod` build artifacts — ignore them in listings.
- PETSc source is at `$PETSC_DIR/$PETSC_ARCH`; Kokkos source at `$PETSC_DIR/$PETSC_ARCH/externalpackages/git.kokkos{,-kernels}`. Both env variables must be set.

Read only when the task needs it
- `docs/dev/testing.md` — before adding or modifying tests
- `docs/dev/kokkos.md` — before touching `*.kokkos.cxx` or CPU/Kokkos debug-compare paths
- `docs/dev/ci.md` — when checking or reproducing a CI pipeline
- `docs/dev/fortran_removal_plan.md` — when working on the staged port of the Fortran core to C

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
