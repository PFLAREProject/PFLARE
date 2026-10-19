# Removing Fortran from the PFLARE core: feasibility and staged plan

Status: proposal (not started). Written against commit `4e1f347` (PFLARE 1.27.0, PETSc >= 3.26).

Goal: `libpflare` is built from C/C++ only, so PFLARE can be built and used
without a Fortran compiler. Every existing Fortran interface survives unchanged
(same module names, routine names, argument lists and semantics), but the
Fortran layer becomes a thin set of `bind(C)` wrappers that is only compiled
when PETSc's own Fortran bindings are on (`PETSC_USE_FORTRAN_BINDINGS` in
`petscconf.h`, which PETSc sets by default whenever it has a Fortran compiler
and `--with-fortran-bindings` is not 0). The tests stay as they are, including
the Fortran test drivers, which are built only in Fortran-enabled builds.
Nothing about how a user interacts with PFLARE changes: the public C API in
`include/pflare.h`, the Fortran modules, the Python bindings, the command line
options, the `-pc_air_print_stats_timings` output that `tools/parse_pflare_output.py`
parses, Spack, and PETSc's `--download-pflare` all keep working.

## 1. Verdict

Feasible, but it is a full rewrite of the algorithmic core, not a refactor.

- `src/` holds 28.3k lines of Fortran. About 6.1k of that is interface and
  binding glue that either disappears or is inverted (section 2). About 22.2k
  lines of algorithms have to be ported to C.
- The 14k lines of Kokkos C++ are untouched apart from call-site cleanups:
  their `PETSC_INTERN void <name>_kokkos(...)` entry points are already C
  functions, so the C port calls them directly and `C_PETSc_Interfaces.F90`
  goes away.
- Nothing in the Fortran uses a feature without a direct C equivalent. There is
  no OpenMP, no coarrays, no OOP (no `class`, `procedure` pointers or
  finalizers), one `recursive` routine (`Binary_Tree.F90`), and BLAS/LAPACK is
  already reached through a precision dispatch header that PETSc's
  `petscblaslapack.h` replaces one for one. MPI is used directly in only seven
  distinct calls (`MPI_Allreduce`, `MPI_Iallreduce`, `MPI_Ibcast`,
  `MPI_Op_create`, `MPI_Comm_rank/size`, `MPI_Abort`).
- `PCPFLAREINV` already has the target shape: its C file owns the context, and
  `PCPFLAREINV_Interfaces.F90` is a pure `bind(C)` wrapper over the C API. The
  work is to bring `PCAIR` (whose Fortran interface reaches directly into
  Fortran data) and the algorithms to the same shape.
- PETSc's package file for `--download-pflare` (`config/BuildSystem/config/packages/PFLARE.py`)
  declares no Fortran requirement and does nothing with `.mod` files, so it
  needs no change.

Four things cannot be literally bit-identical after a port, and need to be
accepted up front (section 9 lists the decisions):

1. **Random numbers.** PMISR, DDC, CR and the GMRES polynomial start vector use
   the Fortran intrinsic `random_number` seeded by `random_seed(put=)`. That
   sequence is defined by the Fortran runtime, not by PFLARE. Stage 0 settles
   this: gfortran's generator is xoshiro256** with a documented seeding
   scramble, and a 40-line C reimplementation can be checked bitwise against
   gfortran. If it matches, C builds reproduce today's gfortran builds exactly
   and become compiler independent. If it does not, the migration uses a
   Fortran shim to stay bitwise during verification and switches to a C
   generator in one clearly labelled stage.
2. **Printed output.** The stats and timings are printed with list-directed
   `print *`, whose number formatting is compiler defined. The C port prints
   with explicit formats that keep every label and field that
   `tools/parse_pflare_output.py` reads. Byte-identical stdout is not a goal;
   parser compatibility is.
3. **Error handling.** The Fortran mostly ignores `ierr` and uses
   `print` + `MPI_Abort` for its own checks. The C port uses `PetscCall` and
   `PetscCheck`/`SETERRQ`, so PETSc errors now propagate with a traceback
   instead of being ignored or aborting. This is the direction recent changes
   already took (see `CHANGELOG.md`, "PETSc errors inside the C routines ...
   now abort, rather than being ignored").
4. **Floating-point summation order.** Replacing `matmul`, `sum` and array
   expressions with loops can change the order of a few reductions. Expect
   last-bit differences, verified by tolerance, never by eye.

## 2. Inventory

Fortran in `src/`, grouped by what happens to it:

| Group | Files | Lines | Fate |
|---|---|---|---|
| C-to-Fortran glue (C calls these) | `C_Fortran_Bindings.F90`, `PCAIR_C_Fortran_Bindings.F90` | 2323 | deleted |
| Fortran-to-C glue (Fortran calls C/Kokkos) | `C_PETSc_Interfaces.F90` | 427 | deleted |
| Fortran user API, reaches into Fortran data | `PCAIR_Interfaces.F90` | 2810 | rewritten as thin `bind(C)` wrappers (~900 lines) |
| Fortran user API, already thin over C | `PCPFLAREINV_Interfaces.F90`, `PFLARE.F90` | 517 | kept, moved to `src/fortran/` |
| Constants | `Pflare_Parameters.F90` | 269 | becomes a shared cpp header plus a Fortran module that includes it |
| Algorithms | the other 35 files | 22.2k | ported to C |

C and C++ in `src/`: `PCAIR.c` (3780), `PCPFLAREINV.c` (1167),
`C_PETSc_Routines.c` (411), ten `*k.kokkos.cxx` (10.7k).

Things outside `src/` that mention Fortran and are affected:

- `Makefile`: `OBJS` list, `LINK.F` used to link the shared library,
  `makedepf90` dependency generation, `*.mod` install, `TEST_TARGETS` with ten
  Fortran drivers, `CHECK_TARGETS` includes the Fortran `matrandom`.
- `tests/Makefile`: `run_check` runs `./matrandom`; the `run_tests_*` groups
  run the ten Fortran drivers.
- `include/finclude/pflare.h`, `include/finclude/pflare_types.h` (kept, user
  facing), `include/finclude/pflare_blaslapack.h` (deleted at the end, it is
  only used by the Fortran algorithms).
- `python/pflare_defs.pyx`: comments explain calling the C wrappers so
  `PetscInitializeFortran` runs first; those calls and comments go.
- `.github/workflows/ci_build.yml`, `dockerfiles/*`: pass `FFLAGS=-Werror ...`
  and `EXTRA_FFLAGS=-fcheck=all` (the Fortran runtime-check job).
- `docs/installation.md` (Fortran module files, include path), `docs/dev/kokkos.md`
  (the `X.F90` / `Xk.kokkos.cxx` sibling convention), `AGENTS.md`.

The Fortran tests depend on these Fortran entry points, which therefore must
survive by name:

- `use pflare` and everything it re-exports: `PCRegister_PFLARE`, the `PCAIR*`
  and `PCPFLAREINV*` routines, and the parameters `PFLAREINV_*`, `AIR_Z_*`,
  `CF_*`, `COEFFS_INV_*`.
- `use pcpflareinv_interfaces` (ex6f_getcoeffs, ex12f_gmres_poly).
- `use cf_splitting, only: compute_cf_splitting, CF_PMISR_DDC` (pmisr_nonsymmetric)
  with the full optional-argument Fortran signature of `compute_cf_splitting`.
- `use gmres_poly_newton, only: modified_leja` (leja_ordering).

## 3. Invariants

These hold at the end of every stage and are checked by the verification
framework of section 6.

- `include/pflare.h` is unchanged except for additions.
- All Fortran public names listed in section 2 keep their module, name,
  arguments, kinds and intents. `PCAIRGetOptions`, `PCAIRGetOptionsAndShell`
  and `PCAIRGetPCShell` are public in `pcair_interfaces` today; see decision
  D3 for what happens to them.
- `make`, `make check`, `make tests`, `make python`, `make install`,
  `make tests_search TEST_MATCH=...` keep their names and exit-code contract.
- The whole CI matrix in `docs/dev/ci.md` stays green: GNU debug/opt, 64-bit
  indices, single precision, Kokkos (serial, OpenMP, no-MPI), valgrind,
  `-malloc_dump`, Intel, macOS, notebook, PETSc `--download-pflare` plus its
  doc build (the `/*E ... E*/` and `/*@ ... @*/` doctext in the C files must be
  kept, PETSc's Sphinx build scans them).
- Test iteration counts are identical to the baseline, residual histories agree
  to tolerance, `tools/parse_pflare_output.py` still parses the stats block.
- Compile with `-Wall -Werror` (and `-Wunused-result -std=c++20` on Kokkos).

## 4. Target layout

```
src/
  pflareimpl.h            private structs shared across modules
                          (PCAIR_Options, PCAIR_Data, PFLAREPolyData, PFLAREMatShellCtx, ...)
  pflare_parameters.h     magic constants and enum values as #defines; included by
                          C and by the Fortran parameter module (cpp runs on .F90)
  X.c / X.h               one C file per former Fortran module, same stem, PETSC_INTERN
                          prototypes in X.h (Xk.kokkos.cxx keeps its name, so the
                          "X / Xk" sibling convention of docs/dev/kokkos.md survives)
  PCAIR.c, PCPFLAREINV.c  own their contexts directly, no Fortran externs
  fortran/                compiled only when PETSC_USE_FORTRAN_BINDINGS=1
    Pflare_Parameters.F90   module pflare_parameters (parameters taken from pflare_parameters.h)
    PCPFLAREINV_Interfaces.F90, PCAIR_Interfaces.F90, PFLARE.F90
    CF_Splitting.F90        module cf_splitting: compute_cf_splitting wrapper for the C routine
    Gmres_Poly_Newton.F90   module gmres_poly_newton: modified_leja wrapper
```

The remaining Fortran lives in `src/fortran/` so that its file stems cannot
collide with the C objects of the same name (`src/CF_Splitting.o` is the C
object; `src/fortran/CF_Splitting.o` the wrapper module). During the migration
a second directory `src/fref/` holds the retired Fortran implementations that
the differential tests still call (section 6); it is deleted in stage 8.

Build gating, all in the top `Makefile` with the existing `_have_conf` helper:

```make
export PETSC_USE_FORTRAN_BINDINGS := $(if $(call _have_conf,PETSC_USE_FORTRAN_BINDINGS),1,0)
```

- `OBJS` is split into `COBJS` (always), `KOBJS` (Kokkos) and `FOBJS`
  (`src/fortran/*.o`, only when the flag is 1).
- The shared library is linked with `$(CLINKER)` (or `$(CXXLINKER)` when
  Kokkos objects are present, so the C++ runtime is pulled in) instead of
  `$(LINK.F)`. The Darwin and static branches keep their current logic.
- C header dependencies are generated with `$(CC) -MM` into `Makefile.deps`
  next to the `makedepf90` output; `make depend` produces both.
- The Fortran test drivers and the `matrandom` entry of `CHECK_TARGETS` are
  listed only when the flag is 1; the `run_tests_*` lines that invoke them are
  prefixed with a `$(SKIP_FORTRAN)` guard that works like `SKIP_SINGLE`.
- `install` already tolerates a missing `*.mod`; it copies `src/fortran/*.mod`
  when present.

## 5. Porting conventions (every stage follows these)

1. One C file per Fortran module, same stem, module-level `use` lists become
   `#include "X.h"`. Everything not in `include/pflare.h` is `PETSC_INTERN`.
   Function names keep the Fortran snake_case names so `git log -S` and the
   Kokkos siblings still line up (`compute_cf_splitting`, `setup_air_pcmg`, ...).
2. Every routine returns `PetscErrorCode`, starts with `PetscFunctionBegin`
   and ends with `PetscFunctionReturn(PETSC_SUCCESS)`. Every PETSc call is
   wrapped in `PetscCall`. Fortran `print "..."; MPI_Abort` becomes
   `PetscCheck(cond, comm, PETSC_ERR_..., "same message")` or `SETERRQ`.
   Keep the message text.
3. Derived types become structs in `pflareimpl.h` with explicit
   `..._init`, `..._reset`, `..._destroy` functions mirroring the Fortran
   `create_air_data`/`reset_air_data`/`destroy_air_data` trio. Allocatable
   arrays become pointer plus size, `allocated(x)` becomes `x != NULL`. Use
   `PetscMalloc1`/`PetscCalloc1`/`PetscFree`; memory handed to users through
   the public API (polynomial coefficients) keeps today's contract (documented
   in `PCPFLAREINV.c`: freed with C `free`, so use `malloc` there).
4. Indices: internal loops are 0-based. Level numbering keeps PFLARE's
   convention (`our_level` 1-based in the Fortran) only where it is visible to
   the user, i.e. the stats print and the `petsc_level` arithmetic in
   `PCAIRGetPolyCoeffs`; document each place the two conventions meet.
   Integers: `PetscInt` for everything that can index a matrix, `int` only
   where the public API already says `int`. Options fields take the type of the
   corresponding public getter in `pflare.h`.
5. Two-dimensional coefficient arrays stay column-major
   (`coeffs[i + j*rows]`); `PCPFLAREINVGetPolyCoeffs` documents column-major.
6. BLAS/LAPACK through `petscblaslapack.h`: `LAPACKgels_`, `LAPACKgelsd_`,
   `BLASgemv_`, `BLASgemm_`, `LAPACKgeev_`, `LAPACKgesvd_`, `LAPACKgeqrf_`,
   `LAPACKgesv_` with `PetscBLASInt` and `PetscCallBLAS`/`PetscCheck` on
   `info`. `include/finclude/pflare_blaslapack.h` is deleted once no Fortran
   algorithm remains. Complex builds stay unsupported (`#error` moves to
   `pflareimpl.h`).
7. `matmul`, `where`, `sum`, array sections and `reshape` become explicit
   loops; when a `matmul` is on dense blocks larger than a few tens of entries,
   use `BLASgemm_`. Keep the arithmetic order where it is cheap to do so.
8. Printing: `PetscPrintf(comm, ...)` on rank 0 using the exact labels in
   `tools/parse_pflare_output.py` (`~~~~~~~~~~~~ Level %d`, `Global rows %d
   Global F-points %d Global C-points %d`, `<timer label> : %.15g`, the five
   complexity lines, `Total cumulative setup time : %.15g`).
9. Kokkos: C calls `*_kokkos` directly; the signatures in `src/*.kokkos.cxx`
   are already C and already take `Mat *`, `IS *`, `Vec *`. The
   `PFLARE_KOKKOS_DEBUG` compare paths are ported verbatim (`kokkos_debug()`
   becomes a C function in `PETSc_Helper.c`). The Kokkos test sequence of
   `AGENTS.md` runs for every stage that touches a module with a Kokkos sibling.
10. Fortran module variables (`Timers.F90` arrays, `TSQR.F90` custom MPI op,
    `PETSc_Helper.F90` debug flag) become file-static variables with the same
    lifetime semantics.
11. No behaviour "improvements" during a port stage. A bug found while porting
    is recorded in the stage's PR description and fixed in a separate PR
    against both implementations (or after stage 8 against the C only), so
    the differential tests stay meaningful.
12. Doctext: the `/*@ ... @*/` manual-page comments stay in `PCAIR.c` and
    `PCPFLAREINV.c`; new C files do not need them (they are `PETSC_INTERN`).

## 6. Verification framework

Each stage ships with evidence, not a claim. Three layers, cheapest first.

**Layer A: the existing suite, on the CI matrix.** `make check`,
`make tests_short`, `make tests` (which includes the Python tests) locally and
the full workflow on the PR. For stages that touch a module with a Kokkos
sibling, additionally the two Kokkos environment runs in `AGENTS.md`
(`PETSC_OPTIONS="-mat_type aijkokkos ..."` and `PFLARE_KOKKOS_DEBUG=1`).
Valgrind and `-malloc_dump` jobs catch leaks in the new manual memory
management; they already exist in CI.

**Layer B: golden logs (added in stage 0).** `tools/capture_test_logs.sh` runs
the `run_tests_*` groups with `-ksp_converged_reason -ksp_monitor` appended
through `PETSC_OPTIONS`, and `tools/compare_test_logs.py` extracts, per test
command, the iteration count, the residual history, and the parsed stats block
(via `parse_pflare_output.py`). Baselines are captured once per CI
configuration from the pre-port commit and stored outside the repo (a CI
artifact keyed by Docker image tag and commit; the GNU debug baseline is also
committed under `tests/golden/` as it is small text). The comparison rule:
iteration counts and converged reasons must be identical; residual norms agree
to a relative tolerance (default 1e-8 double, 1e-4 single); stats fields
`global_rows`, `global_f_points`, `global_c_points`, `n_levels` identical;
complexities to 1e-10. Any miss is a stage failure until explained and signed
off. Stage 8 (RNG swap, if needed) is the only stage allowed to re-baseline.

**Layer C: in-process differential tests.** This is the repo's own
`PFLARE_KOKKOS_DEBUG` idea applied to the port. While a stage is in flight,
the retired Fortran implementation is kept compiled under `src/fref/`
(renamed `bind(C)` exports get an `_fref` suffix; module procedure names are
mangled by the compiler and do not clash). With `make PFLARE_PORT_DEBUG=1`
the C code runs both itself and the Fortran reference and compares:

- leaf stages (2, 3): the call site that was switched to C also calls the
  `_fref` routine and compares outputs (`ISEqual`, `MatEqual`, `VecEqual`,
  or elementwise with the `PFLARE_TOL_*` tolerances from `pflare_parameters.h`),
  exactly as the Kokkos debug paths do;
- stage 4 (CF splitting): `tests/port_compare_cf.c` calls
  `compute_cf_splitting` and `compute_cf_splitting_fref` on the load-test and
  generated matrices for every `CFSplittingType` and checks `ISEqual`;
- stage 5 (inverses): a reference PC type `pflareinv_fref` is registered
  alongside `pflareinv`; `tests/port_compare_pc.c` sets both up on the same
  operator for every inverse type and compares coefficients bitwise (or to
  tolerance where a reduction order changed), the assembled inverse with
  `MatEqual`/`MatAXPY`+`MatNorm`, and the matrix-free apply on random vectors;
- stage 7 (AIR hierarchy): a reference PC type `air_fref` built from the
  Fortran `PCAIR_Shell` path; `tests/port_compare_pc.c -pc_type air` compares
  level by level: `IS_fine`/`IS_coarse` (`ISEqual`), restrictors,
  prolongators, `A_ff`, `A_cc`, coarse matrices (`MatEqual`, then norm of the
  difference), polynomial coefficients, and the result of `PCApply`,
  `PCApplyTranspose` and `PCMatApply`.

The differential drivers are part of `TEST_TARGETS` only while
`PFLARE_PORT_DEBUG=1`, and are deleted with `src/fref/` in stage 8.

**Random numbers during verification.** If the stage 0 experiment shows the C
xoshiro256** matches gfortran bitwise, layers B and C compare exactly on GNU
builds. If not, stage 0 provides `pflare_random_fortran_c(rank, n, out)` in
`src/fref/` which the C port calls until stage 8, so layer C still compares
bitwise, and stage 8 swaps in the C generator and re-baselines layer B.

## 7. Stages

Dependency graph (an arrow means "must be merged first"):

```
0 harness ──> 1 build ──> 2 leaves ──> 3 helpers ──┬──> 4 CF splitting ──┐
                                                   ├──> 5 inverses ──────┼──> 7 AIR core ──> 8 cleanup/RNG ──> 9 gating/CI/docs
                                                   └──> 6 grid transfer ─┘
```

Stages 4, 5 and 6 are independent of each other and can run concurrently on
separate branches; their only shared edits are the `OBJS` list and the
`C_Fortran_Bindings.F90` extern block, so merge them smallest first
(6, 4, 5). Everything else is sequential. Each stage is one PR from one
subagent, reviewed and merged before the next starts, and the subagent's
brief is the stage section below plus sections 3, 5 and 6 of this file.
Sizes are lines of Fortran going in; the C coming out is usually 1.1 to 1.3
times that.

Every stage brief has the same closing requirement: build with
`make -j3 build_tests` warning-free, run layer A (plus the Kokkos runs where
applicable), run layers B and C where listed, and report the results verbatim
in the PR description. A stage is not done while any CI job is red.

### Stage 0: verification harness and RNG experiment (no library change)

Deliverables

- `tools/capture_test_logs.sh`, `tools/compare_test_logs.py` and
  `tests/golden/` as described in section 6, wired into a `make tests_golden`
  target and a CI step that runs it on every job (comparing against the
  baseline artifact for that job's image) and uploads the fresh logs.
- `PFLARE_PORT_DEBUG` make variable plumbed through to `CPPFLAGS`/`FPPFLAGS`
  as `-DPFLARE_PORT_DEBUG`, the `src/fref/` directory added to the build when
  set (empty for now), and `tests/port_compare_pc.c`, `tests/port_compare_cf.c`
  skeletons that build and exit 0.
- `src/Pflare_Random.c` + `src/Pflare_Random.h`: xoshiro256** with gfortran's
  `random_seed(put=)` scramble, a function
  `pflare_random_fill(PetscMPIInt rank, PetscInt n, PetscReal *out)` that
  reproduces the seeding loop used in `PMISR_Module.F90`, `DDC_Module.F90`,
  `CR_Splitting.F90` and `Gmres_Poly.F90` (`seed(i) = rank + 1 + i`), and a
  test `tests/random_compare.F90` that fills the same arrays with
  `random_number` and compares bitwise. Record the outcome (match or not) in
  this file, section 9, D1. If it does not match, add
  `src/fref/Pflare_Random_Fortran.F90` exporting `pflare_random_fortran_c`.
- Baseline golden logs captured from the current `main` for every CI image.

Verification: the harness compares `main` against itself with zero
differences; the RNG test result is recorded.

### Stage 1: build system restructure (no behaviour change)

- Split `OBJS` into `COBJS`, `KOBJS`, `FOBJS`; add the
  `PETSC_USE_FORTRAN_BINDINGS` detection; add `src/fortran/` and `src/fref/`
  to the search path and `clean`; add `$(CC) -MM` dependency generation to
  `make depend`; keep linking with `LINK.F` for now (switching the linker is
  stage 9, when no Fortran object remains).
- Move `PFLARE.F90`, `PCPFLAREINV_Interfaces.F90`, `PCAIR_Interfaces.F90`,
  `PCAIR_C_Fortran_Bindings.F90` and `Pflare_Parameters.F90` to
  `src/fortran/` (git renames). Update `Makefile.deps` and the `INCLUDE`
  paths so `.mod` files still land where `tests/` and `python/` find them.
- Add `src/pflare_parameters.h` with every constant and enum value from
  `Pflare_Parameters.F90` as `#define`s, and rewrite the Fortran module to
  take its values from that header (the Fortran `parameter` statements stay,
  so `use pflare_parameters` is unchanged). Values must be copied exactly,
  including the precision-dependent `PFLARE_TOL_*` branches.
- Add `src/pflareimpl.h` (empty struct section, the complex `#error`, the
  include of `pflare_parameters.h`).

Verification: full CI matrix, layer B identical, `make install` output
directory listing identical to before (compare with `diff -r`).

### Stage 2: leaf modules (1.8k lines)

Port `Timers.F90`, `Binary_Tree.F90`, `Sorting.F90`, `Aggregation.F90`,
`Weighted_Jacobi.F90`, `TSQR.F90` to `src/<Stem>.c` + `.h`. These depend only
on `pflare_parameters` and PETSc. For each, the Fortran callers are switched to
the C version through a `bind(C)` interface block placed in a new
`src/fortran/Port_Interfaces.F90` (this file grows through stages 2 to 6 and
is deleted in stage 7). The retired Fortran file moves to `src/fref/` and
keeps compiling.

Notes for the subagent

- `TSQR.F90` holds a custom `MPI_Op` created once (`built_custom_op_tsqr`) and
  uses `MPI_Iallreduce`/`MPI_Ibcast` with request handles stored in
  `tsqr_buffers`; the C struct `PFLARETSQRBuffers` must keep the asynchronous
  protocol (start in one routine, wait in another) exactly.
- `Timers.F90` module arrays become `static PetscReal` arrays; the ids in
  `pflare_parameters.h`.
- `Binary_Tree.F90` is the only recursive code; port as is.
- `gmres_poly_data_type` holds a `tsqr_buffers` component. Until stage 5 that
  component must stay usable from Fortran: make the C `PFLARETSQRBuffers`
  opaque behind a `type(c_ptr)` in `gmres_poly_data` and give
  `Gmres_Poly.F90` the `bind(C)` create/destroy/start/finish calls it needs.

Verification: layers A and C (call-site compare under `PFLARE_PORT_DEBUG`),
layer B identical.

### Stage 3: PETSc helpers and repartitioning (2.3k lines)

Port `PETSc_Helper.F90` and `Repartition.F90` to C and fold
`C_PETSc_Routines.c` into `PETSc_Helper.c`; `kokkos_debug()`,
`remove_small_from_sparse`, `remove_from_sparse_match`,
`MatCreateSubMatrixWrapper`, `destroy_matrix_reuse` and the rest become
`PETSC_INTERN` C functions returning `PetscErrorCode`. The public
`remove_from_sparse_match` in `PCAIR.c` calls the C version directly and
drops `PetscInitializeFortran`. All 21 helper routines get `bind(C)`
interfaces in `Port_Interfaces.F90` for the Fortran callers that remain
(nearly every module uses `petsc_helper`), so this stage is mostly plumbing
and must be done before 4, 5 and 6.

Notes: `PETSc_Helperk.kokkos.cxx` is the sibling; the `PFLARE_KOKKOS_DEBUG`
compare paths in the helper move with it. `MatMPICreateNonemptySubcomm_c`
already returns `PetscErrorCode`.

Verification: A (with both Kokkos runs), B identical, C for every ported
helper that has a Kokkos sibling (CPU vs `_fref` vs Kokkos).

### Stage 4: CF splitting family (3.1k lines)

Port `PMISR_Module.F90`, `MatDiagDom.F90`, `DDC_Module.F90`, `SAbs.F90`,
`CR_Splitting.F90`, `CF_Splitting.F90`, `MatDiagDomSubmatrix.F90`. The public
C entry points `compute_cf_splitting` and `compute_diag_dom_submatrix` in
`PCAIR.c` call the C implementations and drop the `_c` externs and
`PetscInitializeFortran`. `CR_Splitting` needs an approximate inverse; until
stage 5 it calls the existing C-callable
`calculate_and_build_approximate_inverse_c` (Fortran) and `reset_inverse_mat_c`.

The Fortran module `cf_splitting` is kept as `src/fortran/CF_Splitting.F90`
with the same `compute_cf_splitting` signature, including the optional `cr_*`
arguments, implemented as a wrapper over a new `PETSC_INTERN` C function that
takes the full argument list (`pflare.h`'s `compute_cf_splitting` has the
shorter list and stays as is).

Random numbers: `pmisr`, `ddc` and `cr` draw their measures with the stage 0
generator (or the Fortran shim, per D1).

Verification: A with both Kokkos runs (PMISR, DDC and MatDiagDom have Kokkos
siblings and `PFLARE_KOKKOS_DEBUG` paths), B identical,
`tests/port_compare_cf.c` under `PFLARE_PORT_DEBUG` for every
`CFSplittingType` on `tests/data` matrices and on the generated advection
problems, serial and `mpiexec -n 2`/`-n 4`.

### Stage 5: polynomials and approximate inverses (7.5k lines)

Port `Gmres_Poly_Data_Type.F90`, `Matshell_Data_Type.F90`,
`Gmres_Poly_Apply.F90`, `Gmres_Poly.F90`, `Gmres_Poly_Newton.F90`,
`Neumann_Poly.F90`, `SAI_Z.F90`, `Approx_Inverse_Setup.F90`. After this stage
`PCPFLAREINV.c` is pure C: `calculate_and_build_approximate_inverse`,
`reset_inverse_mat`, `inverse_type_is_polynomial` and
`pflareinv_shell_block_matapply` are C functions, and the `PetscInitializeFortran`
call goes. The matrix-free polynomial `MatShell` operations (`MATOP_MULT`,
`MATOP_MULT_TRANSPOSE`, the block applies, the lazily built transpose twin)
are C callbacks on a `PFLAREMatShellCtx` whose layout mirrors `mat_ctxtype`
(including `void *air_data` and `our_level`, which the Fortran `FC_Smooth`
still sets until stage 7).

Data crossing the boundary until stage 7: `air_multigrid_data` holds arrays of
`gmres_poly_data` and the AIR Fortran modules reference `%coefficients` and
`%buffers` in about 110 places. Make the Fortran `gmres_poly_data` a
`type, bind(C)` mirror of the C `PFLAREPolyData` (`type(c_ptr) :: coefficients;
PetscInt :: rows, cols; integer(c_int) :: inverse_type, poly_order,
sparsity_order; type(c_ptr) :: tsqr`) and give the AIR modules a small helper
`poly_coeffs(pd)` returning a contiguous 2D Fortran pointer via `c_f_pointer`,
so the existing `%coefficients` expressions become one-line edits. The 18
`mat_ctx%` references in `FC_Smooth*.F90` and `AIR_MG_Setup.F90` go through
`bind(C)` accessors.

`modified_leja` keeps its Fortran name in `src/fortran/Gmres_Poly_Newton.F90`
as a wrapper that passes the three assumed-shape arrays by base address and
size to the C `modified_leja`.

Random numbers: the GMRES polynomial start vector in `Gmres_Poly.F90`
(`random_data(:,1:2)` then Box-Muller) uses the stage 0 generator.

Verification: A with both Kokkos runs (`Gmres_Polyk`, `Gmres_Poly_Newtonk`,
`SAI_Zk` siblings), B identical, `tests/port_compare_pc.c -pc_type pflareinv`
under `PFLARE_PORT_DEBUG` for every `PCPFLAREINVType`, assembled and
matrix-free, with and without diagonal scaling and `-pc_pflareinv_subcomm`,
serial and parallel, plus `PCApplyTranspose` and `PCMatApply`. The Python
PFLAREINV tests (`ex_pcpflareinv_options.py`, `ex6f_getcoeffs.py`) exercise
the coefficient memory contract.

### Stage 6: grid transfer and constraints (1.8k lines)

Port `Grid_Transfer.F90`, `Grid_Transfer_Improve.F90`, `Constrain_Z_or_W.F90`.
They take matrices, index sets and scalars only (no `air_data`), so the
boundary is clean. `Grid_Transferk.kokkos.cxx` is the sibling. The dense BLAS
calls in `Constrain_Z_or_W.F90` use `BLASgemv_`/`BLASgemm_`.

Verification: A with both Kokkos runs, B identical, C call-site compare
(`MatEqual` on the produced R/P/W/Z, elementwise tolerance after `improve`).

### Stage 7: AIR hierarchy and the PCAIR Fortran interface (6.1k lines ported, 2.8k rewritten, 2.3k deleted)

Port `AIR_Data_Type.F90`, `AIR_Data_Type_Routines.F90`, `FC_Smooth.F90`,
`FC_Smooth_Block.F90`, `AIR_MG_Stats.F90`, `AIR_Operators_Setup.F90`,
`AIR_MG_Setup.F90`, `PCAIR_Shell.F90`, `PCAIR_Data_Type.F90`. Then:

- `PCAIR.c` owns `PCAIR_Data` (`air_multigrid_data` + the inner `PCMG`)
  directly. Keep the inner `PCShell` layer so `-ksp_view` output and
  `PCAIRGetPCShell` are unchanged (decision D3); the shell's apply, apply
  transpose, setup, view and destroy callbacks are the C ports of
  `PCAIR_Shell.F90`, and reset is done as today by calling
  `PCReset` on the shell plus the data reset. All 53 getters and 47 setters in
  `PCAIR.c` read and write the C options struct directly, with the same
  "reset the hierarchy if the value changed" logic that
  `PCAIR_Interfaces.F90` has today (`PCAIRSetMaxLevels` is the template:
  compare, `PCReset` the shell, reset the data, mark the outer PC not set up).
  Delete the `_c` externs.
- `src/fortran/PCAIR_Interfaces.F90` is rewritten in the style of
  `PCPFLAREINV_Interfaces.F90`: every public routine keeps its signature and
  calls the C function through a `bind(C)` interface, converting `PetscBool`
  and `PetscEnum` as that file does. `PCAIRGetPolyCoeffs` and
  `PCAIRSetPolyCoeffs` keep their allocatable-array semantics by copying
  through the C getter/setter. `PCAIRGetOptions`/`PCAIRGetOptionsAndShell`
  follow D3.
- Delete `C_Fortran_Bindings.F90`, `PCAIR_C_Fortran_Bindings.F90`,
  `C_PETSc_Interfaces.F90`, `Port_Interfaces.F90`; drop every remaining
  `PetscInitializeFortran`.
- Stats printing follows convention 8.

Notes for the subagent

- `air_multigrid_data` has 22 reuse matrix slots and 2 IS slots with the
  `REUSE_MAT_ACTIVE`/`REUSE_IS_ACTIVE` tables; keep them as `static const`
  tables in `pflareimpl.h` with the same indices.
- `fast_veciscopy_exists` and `kokkos_is_views_handle` wire into
  `VecISCopyLocalk.kokkos.cxx`; the handle is already a `void *`.
- The timers (`Timers.c`) and stats (`AIR_MG_Stats.c`) are the only places the
  1-based `our_level` is user visible.
- `tests/reuse_preconditioner.c`, `pcair_reuse_amount_range.c`,
  `operator_refcount.c`, `comm_self_then_world.c` and the `-check_copies`
  drivers are the sharpest tests of lifetime and reuse handling; run
  `make tests` (not just `tests_short`) locally before pushing.

Verification: A with both Kokkos runs, B identical (iteration counts and stats
fields) on every image, `tests/port_compare_pc.c -pc_type air` against
`air_fref` under `PFLARE_PORT_DEBUG` over a matrix of options (every
`CFSplittingType`, every inverse type, `-pc_air_z_type` product/lair/lair_sai,
`-pc_air_smooth_type ff/fc/fcf`, constrain W/Z, improve W/Z its, processor
agglomeration, subcomm, reuse sparsity with `-pc_air_reuse_amount 0..3`,
symmetric, auto truncation), serial and parallel. The Python PCAIR tests,
the Fortran `ex6f*`/`ex12f*` drivers and the notebook job all pass unchanged.
The valgrind and `-malloc_dump` jobs are the leak check for the new struct
lifetimes.

### Stage 8: remove the reference Fortran, finalise the random generator

- Delete `src/fref/`, the `PFLARE_PORT_DEBUG` plumbing, and
  `tests/port_compare_*.c`. The compare drivers are not kept: `PFLARE_KOKKOS_DEBUG`
  remains the only in-process compare mode.
- Per D1: if the C generator already matches gfortran, nothing to do. If the
  Fortran shim was used, switch `pmisr`, `ddc`, `cr` and the polynomial start
  vector to `Pflare_Random.c`, re-capture the layer B baselines, and adjust
  any `-ksp_max_it` in `tests/Makefile` that no longer bounds the iteration
  count (expected to be rare and by at most one or two iterations; list each
  change in the PR).
- Delete `include/finclude/pflare_blaslapack.h`.
- `CHANGELOG.md`: one entry describing the C core, the error-handling change
  (convention 2) and, if applicable, the RNG change.

Verification: A on the full matrix, B identical to the (possibly new)
baseline.

### Stage 9: gating, CI, docs

- `Makefile`: the `PETSC_USE_FORTRAN_BINDINGS` gate on `FOBJS`, the Fortran
  `TEST_TARGETS`, `matrandom` in `CHECK_TARGETS`, `makedepf90` in `depend`,
  and the `.mod` install; switch the shared-library link to
  `$(CLINKER)`/`$(CXXLINKER)`. `tests/Makefile`: `$(SKIP_FORTRAN)` prefix on
  the Fortran driver lines and `run_check` prints "Fortran check skipped
  (PETSc configured without Fortran bindings)" like the MPIUNI case.
- CI: a new `gnu_opt_no_fortran` job (PETSc configured `--with-fc=0`, no
  gfortran in the image) and a `gnu_debug_no_fortran_bindings` job
  (`--with-fortran-bindings=0` with a Fortran compiler present, which is the
  case where PETSc's `.mod` files are absent but `FC` is set). The
  `EXTRA_FFLAGS=-fcheck=all` job becomes `-fsanitize=address,undefined` on
  the C core (same purpose: runtime bounds and lifetime checks).
  `dockerfiles/Dockerfile_no_fortran`. The `FFLAGS=-Werror ...` arguments stay
  for the Fortran-enabled jobs.
- Docs: `docs/installation.md` gains "Building without a Fortran compiler";
  `docs/dev/kokkos.md` conventions say `X.c` / `Xk.kokkos.cxx`;
  `docs/dev/testing.md` and `docs/dev/ci.md` describe the gate and the new
  jobs; `AGENTS.md` codebase map updated; `python/pflare_defs.pyx` comments
  updated. `README.md` still lists the Fortran interface.
- Spack: `PFLARE_spack/package.py` (separate repository) can later drop its
  Fortran requirement; note it in the CHANGELOG, no change here.

Verification: full CI matrix including the two new jobs; in the no-Fortran
image `make`, `make check`, `make tests`, `make python`, `make install` all
succeed and `make install` writes no `.mod`; `nm libpflare.so | grep -i
_gfortran` is empty.

## 8. Risks

| Risk | Where | Mitigation |
|---|---|---|
| Lifetime bugs (double destroy, leak, use after reset) in the hand-managed C structs | stages 5, 7 | explicit init/reset/destroy trio, valgrind + `-malloc_dump` CI, `operator_refcount` and reuse tests, `MatDestroy`-nulls-the-handle semantics reproduced (several Fortran comments rely on copying handles back) |
| `PetscInt`/`int` width mix on 64-bit index builds | everywhere | convention 4, the `gnu_opt_64_bit` and `gnu_opt_64_bit_kokkos` jobs |
| Single precision | everywhere | PETSc BLAS macros, `PetscReal` only, the `gnu_debug_kokkos_single_prefix` job and the `SKIP_SINGLE` gates stay |
| Summation-order drift masquerading as a bug, or hiding one | stages 5, 6, 7 | layer C compares with named tolerances; any difference above `PFLARE_TOL_*` is investigated, not widened |
| The in-process reference (`src/fref/`) drifting from `main` while a stage is open | stages 4 to 7 | the reference is never edited; bug fixes land after the stage or in both (convention 11) |
| Output parser contract | stage 7 | `make tests_python` runs `run_parse_tests.py`; convention 8 |
| Stage 7 is large | stage 7 | the brief orders it: data type + routines, smoothers, stats, operators setup, MG setup, shell, PCAIR.c getters/setters, Fortran interface; `air_fref` gives level-by-level diagnosis |
| PETSc doc build in the `gnu_petsc_config` job breaking on moved doctext | stages 4, 5, 7 | keep `/*@ @*/` blocks in `PCAIR.c`/`PCPFLAREINV.c`; the job already checks the rendered pages exist |
| macOS link of a C++-containing shared library without the Fortran driver | stage 9 | use `$(CXXLINKER)` when `KOBJS` is non-empty; the `macos_64_bit_kokkos` job covers it |

## 9. Decisions needed from the maintainer

- **D1 Random numbers.** Preferred: the C xoshiro256** reimplementation with
  gfortran seeding, if the stage 0 bitwise test passes (then nothing changes
  for GNU builds and Intel/Cray builds become identical to GNU, which is a
  behaviour change only for them). Fallback: `PetscRandom` (`PETSCRANDER48`)
  seeded `rank + 1`, accepted as a one-off re-baseline in stage 8. Record the
  stage 0 result here.
- **D2 Error handling.** Confirm that PETSc errors propagating (and PFLARE's
  own checks becoming `PetscCheck`) is acceptable in place of ignored `ierr`
  and `MPI_Abort`. Tests already run with `-on_error_abort` where it matters.
- **D3 Fortran-only accessors.** `PCAIRGetOptions`, `PCAIRGetOptionsAndShell`
  (return a pointer to the internal `air_options` derived type) and
  `PCAIRGetPCShell` are public in `pcair_interfaces` but undocumented and
  unused by tests and Python. Preferred: keep `PCAIRGetPCShell` (the shell
  layer is kept anyway), and keep `PCAIRGetOptions`/`PCAIRGetOptionsAndShell`
  by declaring `air_options` as a `type, bind(C)` mirror of the C
  `PCAIR_Options` struct with the same field names (`logical` fields become
  `logical(c_bool)`, which plain Fortran assignments still accept). Alternative:
  drop the two options accessors with a CHANGELOG note. Either way the
  `PCAIRGet*/Set*` routines are the supported API.
- **D4 Printed number format.** `%.15g` for reals in the stats block (section
  5, convention 8). The parser accepts any non-space token.
- **D5 Stage parallelism.** Whether to run stages 4, 5 and 6 concurrently
  (three subagents, three branches, merge order 6, 4, 5) or strictly
  sequentially. Sequential is simpler to review; concurrent saves wall-clock
  time at the cost of two small rebases.
