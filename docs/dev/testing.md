# Test system (agent reference)

How tests are wired
- Test executables are listed by hand in `TEST_TARGETS` (top `Makefile`, ~line 160); `adv_1dk` is appended only when PETSc has Kokkos. `CHECK_TARGETS` (`adv_diff_fd matrandom`) is what `make check` runs.
- `tests/Makefile` has no per-test declarations: it is a set of `run_tests_*` recipe targets whose bodies are literal `./exe -options` lines with `@echo` labels. Groups: `run_tests_load_{serial,parallel}`, `run_tests_no_load_short_{serial,parallel}`, `run_tests_no_load_{serial,parallel}`, `run_tests_medium_{serial,parallel}`, `run_tests_copies`, `run_check`.
- `run_tests_copies` (`make tests_copies`, serial and parallel in one group) runs the `-check_copies` drivers that fail on any host/device copy in a second solve. Not part of `make tests` or the default `tests_search` groups: it only means anything on a GPU and its parallel lines need GPU-aware MPI, so only the GPU CI workflow calls it.
- `run_tests_collectives` (`make tests_collectives`, parallel only, its own CI job `gnu_opt_collectives`) runs the drivers with `-check_collectives` (`adv_1d`, `adv_1d_multi_rhs`, `ex6f`) under `env LD_PRELOAD=tests/libcount_collectives.so`. Not part of `make tests`, as it needs `LD_PRELOAD` (linux). The library (`tests/count_collectives.c`) wraps every MPI collective under its `MPI_` and `PMPI_` names, which sees calls from C and Fortran, and counts them on communicators with more than one rank inside windows the drivers open and close with `MPI_Pcontrol` (`tests/check_collectives.h`; level >= 1 opens a window expecting level - 1 collectives, 0 closes it). A window with the wrong count aborts with a backtrace of each collective on each rank. Each driver first checks a calibration window around a `VecNorm` (and a Fortran `MPI_Allreduce` in `ex6f`) sees exactly its reductions, so a run where the counting isn't working fails rather than passing with zero; `-check_collectives` also errors if the library isn't preloaded or there is one rank. To check a new code path, open a window around it in a driver with `CheckCollectivesBegin()`/`CheckCollectivesEnd()` after a warm-up call, and add a line to `run_tests_collectives`.
- Executables are built by PETSc's implicit rules; `tests/Makefile` itself exports `LD_LIBRARY_PATH`/`DYLD_FALLBACK_LIBRARY_PATH` pointing at `lib/`.

Checklist for adding a test
1. Add the driver source in `tests/`.
2. Add the executable name to `TEST_TARGETS` in the top `Makefile`.
3. Add the executable name to `.gitignore` (there is a marked section for compiled test executables).
4. Add invocation line(s) to the appropriate `run_tests_*` group(s) in `tests/Makefile` — usually both a serial and a parallel (`mpiexec -n 2`+) variant, with an `@echo` label describing the test.
5. Add to the short groups sparingly: the valgrind CI jobs run `make tests_short`, which takes a long time. Prefer `run_tests_no_load_*` or `run_tests_medium_*`.

TEST_MATCH mechanics (`make tests_search TEST_MATCH="<substring>"`)
- Dry-runs (`make -n`) the groups in `FILTER_RUN_TARGETS` (default = all groups; load groups dropped when 64-bit indices or single precision), joins continuation lines, greps command lines for the substring with `grep -F`, and evals each matching `./exe` line.
- Empty `TEST_MATCH` is an error (exit 2); zero matches is a success ("No matching test commands found").
- Override the searched groups with `FILTER_RUN_TARGETS="run_tests_no_load_serial ..."`.

Precision / index-size gates
- Load tests (which read binary matrices from `tests/data/`) only run on 32-bit indices + double precision (`RUN_LOAD_TESTS` in the top `Makefile`) — the data files are written in that format.
- `KSP_RTOL` is precision-aware: 1e-5 single, 1e-10 double (`tests/Makefile`).
- Individual commands that fail under single precision are prefixed with `$(SKIP_SINGLE)`, which turns them into echoes in single builds. See the comment at `tests/Makefile:36-47` for the recurring reasons (power/Arnoldi polynomial-basis overflow; `-pc_air_reuse_sparsity` pattern staleness). Gate only the specific failing command only if absolutely necessary when the failure is understood, never a whole class.

Python tests
- `make tests_python` (also included in `make tests`). Requires petsc4py and `PYTHONPATH` containing `python/`.
