# Removing Fortran from the PFLARE core: feasibility and staged plan

Status: proposal (not started). Written against branch
`claude/beautiful-davinci-gxkf35` at `85b4f7f` (PFLARE 1.27.0, PETSc >= 3.26),
which is about to be merged into `main`. Every stage branches from that
branch (or from `main` once it carries it) plus the previously merged stages.

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

- `src/` holds 28.5k lines of Fortran. About 6.3k of that is interface and
  binding glue that either disappears or is inverted (section 2). About 22.1k
  lines of algorithms have to be ported to C.
- The 14k lines of Kokkos C++ are untouched apart from call-site cleanups:
  their `PETSC_INTERN void <name>_kokkos(...)` entry points are already C
  functions, so the C port calls them directly and `C_PETSc_Interfaces.F90`
  goes away.
- Nothing in the Fortran uses a feature without a direct C equivalent. There is
  no OpenMP, no coarrays, no OOP (no `class`, `procedure` pointers or
  finalizers), one `recursive` routine (`Binary_Tree.F90`, which is replaced
  by PETSc's sort utilities anyway), and BLAS/LAPACK is already reached through
  a precision dispatch header that PETSc's `petscblaslapack.h` replaces one
  for one. MPI is used directly in only seven distinct calls (`MPI_Allreduce`,
  `MPI_Iallreduce`, `MPI_Ibcast`, `MPI_Op_create`, `MPI_Comm_rank/size`,
  `MPI_Abort`).
- `PCPFLAREINV` already has the target shape: its C file owns the context, and
  `PCPFLAREINV_Interfaces.F90` is a pure `bind(C)` wrapper over the C API. The
  work is to bring `PCAIR` (whose Fortran interface reaches directly into
  Fortran data) and the algorithms to the same shape. The comment in
  `PCCreate_AIR` says why the `PCShell` layer exists: there was no reliable
  way to hand a Fortran context to `pc->data`. In C there is, so the shell
  layer and the Fortran-only accessors that reached through it go.
- PETSc's package file for `--download-pflare` (`config/BuildSystem/config/packages/PFLARE.py`)
  declares no Fortran requirement and does nothing with `.mod` files, so it
  needs no change.

Decisions already taken by the maintainer, which this plan follows:

1. **Random numbers stay in Fortran until the very end.** PMISR, DDC, CR and
   the GMRES polynomial start vector draw from the Fortran intrinsic
   `random_number`, seeded by `random_seed(put=)`. That sequence is defined by
   the Fortran runtime, so a C generator would move every random-dependent
   result. Stage 0 wraps the four draws in one small `bind(C)` Fortran routine
   that the C ports call; it is the last piece of core Fortran and is removed
   in stage 9, after everything else has been verified against the Fortran
   with results that are bit-identical up to summation order.
2. **Error handling.** Today `ierr` is mostly ignored and PFLARE's own checks
   are `print` + `MPI_Abort`. The C port wraps every PETSc call in `PetscCall`
   and turns the checks into `PetscCheck`/`SETERRQ`, so errors propagate with
   a PETSc traceback.
3. **Fortran-only accessors.** `PCAIRGetOptions`, `PCAIRGetOptionsAndShell`,
   `PCAIRGetPCShell` and `PCAIRGetPolyCoeffsPointer` (all public in
   `pcair_interfaces`, none documented, none used by tests, Python or docs)
   existed only to make the Fortran implementation reachable from the
   interface layer. They are removed, together with the `PCShell` layer and
   the C symbol `c_PCAIRGetPCShell`. The `PCAIRGet*/Set*` routines are the
   API.
4. **Printed output.** The stats and timings are printed with list-directed
   `print *`, whose number formatting is compiler defined. The C port prints
   with explicit formats that keep every label and field that
   `tools/parse_pflare_output.py` reads. Byte-identical stdout is not a goal;
   parser compatibility is. The `-ksp_view` output loses the inner
   `PC Object: ... type: shell` block when the shell layer goes; the
   `The underlying PCMG:` line and the PCMG view stay.
5. **Floating-point summation order.** Replacing `matmul`, `sum` and array
   expressions with loops can change the order of a few reductions. Last-bit
   differences are expected and verified by tolerance.
6. **No external libraries.** Only PETSc (with the BLAS/LAPACK and MPI it was
   configured with) and the C standard library. Modules that only existed
   because Fortran has no standard containers (`Binary_Tree.F90`, parts of
   `Sorting.F90`) are replaced by PETSc's `PetscSort*` utilities, not by a
   third-party library.
7. **The C mirrors the Fortran.** Same files, same routine names and order,
   same argument order, same local variable names, same comments (section 5).

## 2. Inventory

Fortran in `src/`, grouped by what happens to it:

| Group | Files | Lines | Fate |
|---|---|---|---|
| C-to-Fortran glue (C calls these) | `C_Fortran_Bindings.F90`, `PCAIR_C_Fortran_Bindings.F90` | 2323 | deleted |
| Fortran-to-C glue (Fortran calls C/Kokkos) | `C_PETSc_Interfaces.F90` | 427 | deleted |
| Fortran user API, reaches into Fortran data | `PCAIR_Interfaces.F90` | 2810 | rewritten as thin `bind(C)` wrappers (~900 lines) |
| Fortran user API, already thin over C | `PCPFLAREINV_Interfaces.F90`, `PFLARE.F90` | 517 | kept, moved to `src/fortran/` |
| Constants | `Pflare_Parameters.F90` | 269 | becomes a shared cpp header plus a Fortran module that includes it |
| Fortran-only containers | `Binary_Tree.F90` | 552 | replaced by `PetscSortInt`/`PetscSortedRemoveDupsInt` |
| Algorithms | the other 34 files | 21.6k | ported to C |

C and C++ in `src/`: `PCAIR.c` (3780), `PCPFLAREINV.c` (1167),
`C_PETSc_Routines.c` (411), ten `*k.kokkos.cxx` (10.7k).

Fortran dependency order (from `Makefile.deps`), which fixes the stage order:

```
Pflare_Parameters, Binary_Tree, TSQR, Aggregation, Sorting, C_PETSc_Interfaces
  -> Gmres_Poly_Data_Type, AIR_Data_Type, Timers, PETSc_Helper, Weighted_Jacobi
  -> Matshell_Data_Type, Gmres_Poly_Apply, SAbs, Constrain_Z_or_W, Grid_Transfer(_Improve),
     MatDiagDom, PMISR_Module, Repartition, SAI_Z
  -> Gmres_Poly, DDC_Module, FC_Smooth, FC_Smooth_Block, AIR_MG_Stats
  -> Gmres_Poly_Newton, Neumann_Poly, Approx_Inverse_Setup
  -> CR_Splitting -> CF_Splitting -> MatDiagDomSubmatrix
  -> AIR_Data_Type_Routines, AIR_Operators_Setup, AIR_MG_Setup, PCAIR_Shell
  -> PCAIR_Interfaces, PFLARE, C_Fortran_Bindings, PCAIR_C_Fortran_Bindings
```

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
  arguments, kinds and intents.
- `make`, `make check`, `make tests`, `make python`, `make install`,
  `make tests_search TEST_MATCH=...` keep their names and exit-code contract.
- The whole CI matrix in `docs/dev/ci.md` stays green: GNU debug/opt, 64-bit
  indices, single precision, Kokkos (serial, OpenMP, no-MPI), valgrind,
  `-malloc_dump`, Intel, macOS, notebook, PETSc `--download-pflare` plus its
  doc build (the `/*E ... E*/`, `/*MC ... M*/` and `/*@ ... @*/` doctext in
  the C files must be kept, PETSc's Sphinx build scans them).
- Test iteration counts are identical to the baseline, residual histories agree
  to tolerance, `tools/parse_pflare_output.py` still parses the stats block.
- Compile with `-Wall -Werror` (and `-Wunused-result -std=c++20` on Kokkos).

## 4. Target layout

```
src/
  pflareimpl.h            private structs shared across modules
                          (air_options, air_multigrid_data, gmres_poly_data, mat_ctxtype, ...)
  pflare_parameters.h     magic constants and enum values as #defines; included by
                          C and by the Fortran parameter module (cpp runs on .F90)
  X.c / X.h               one C file per former Fortran module, same stem, PETSC_INTERN
                          prototypes in X.h (Xk.kokkos.cxx keeps its name, so the
                          "X / Xk" sibling convention of docs/dev/kokkos.md survives)
  PCAIR.c, PCPFLAREINV.c  own their contexts directly, no Fortran externs, no PCShell
  Pflare_Random.F90       (stages 0 to 8 only) the four random_number draws behind one
                          bind(C) routine; the last core Fortran, removed in stage 9
  fortran/                compiled only when PETSC_USE_FORTRAN_BINDINGS=1
    Pflare_Parameters.F90   module pflare_parameters (parameters taken from pflare_parameters.h)
    PCPFLAREINV_Interfaces.F90, PCAIR_Interfaces.F90, PFLARE.F90
    CF_Splitting.F90        module cf_splitting: compute_cf_splitting wrapper for the C routine
    MatDiagDomSubmatrix.F90 module matdiagdomsubmatrix: compute_diag_dom_submatrix wrapper
    Gmres_Poly_Newton.F90   module gmres_poly_newton: modified_leja wrapper
    Gmres_Poly_Data_Type.F90 (stages 5 to 7) bind(C) mirror of gmres_poly_data
    Port_Interfaces_Poly.F90 (stages 5 to 7) wrappers for the AIR Fortran to call
                            the stage 5 C ports (port_interfaces cannot use the fref
                            modules, which use port_interfaces)
    Port_Interfaces_Grid.F90 (stages 6 to 7) the same for the stage 6 C ports
  fref/                   (stages 2 to 7 only) the retired Fortran implementations,
                          compiled under PFLARE_PORT_DEBUG=1 for the differential tests
```

The remaining Fortran lives in `src/fortran/` so that its file stems cannot
collide with the C objects of the same name (`src/CF_Splitting.o` is the C
object; `src/fortran/CF_Splitting.o` the wrapper module). `src/fref/` is
deleted in stage 8 and `src/Pflare_Random.F90` in stage 9.

Build gating, all in the top `Makefile` with the existing `_have_conf` helper:

```make
export PETSC_USE_FORTRAN_BINDINGS := $(if $(call _have_conf,PETSC_USE_FORTRAN_BINDINGS),1,0)
```

- `OBJS` is split into `COBJS` (always), `KOBJS` (Kokkos), `FCORE_OBJS`
  (core Fortran still being ported, empty after stage 9) and `FOBJS`
  (`src/fortran/*.o`, only when the flag is 1).
- The shared library is linked with `$(CLINKER)` (or `$(CXXLINKER)` when
  Kokkos objects are present, so the C++ runtime is pulled in) instead of
  `$(LINK.F)`; both variables come from `lib/petsc/conf/variables`. The
  Darwin and static branches keep their current logic.
- C header dependencies are generated with `$(CC) -MM` into `Makefile.deps`
  next to the `makedepf90` output; `make depend` produces both.
- The Fortran test drivers and the `matrandom` entry of `CHECK_TARGETS` are
  listed only when the flag is 1; the `run_tests_*` lines that invoke them are
  prefixed with a `$(SKIP_FORTRAN)` guard that works like `SKIP_SINGLE`.
- `install` already tolerates a missing `*.mod`; it copies the `*.mod` files
  when present (the compiler writes them to the top directory, those of
  `src/fortran/` included).

## 5. Porting conventions (every stage follows these)

1. **Mirror the Fortran.** One C file per Fortran module with the same stem;
   routines appear in the same order with the same names (`compute_cf_splitting`,
   `setup_air_pcmg`, `calculate_and_build_approximate_inverse`, ...), the
   same argument order, and the same local variable names. The Fortran
   comments are carried over verbatim (the `! ~~~~~~~~` section separators
   become `// ~~~~~~~~`, the routine header comments become the C routine's
   header comment). Where a routine's control flow is restructured (for
   instance a `where` becoming a loop), keep the comment that explains why the
   code is there. A reviewer should be able to read `X.F90` and `X.c` side by
   side. Module-level `use` lists become `#include "X.h"` lines in the same
   order. Everything not in `include/pflare.h` is `PETSC_INTERN`.
2. **Errors.** Every routine returns `PetscErrorCode`, starts with
   `PetscFunctionBegin` and ends with `PetscFunctionReturn(PETSC_SUCCESS)`.
   Every PETSc call is wrapped in `PetscCall` (`PetscCallMPI` for MPI,
   `PetscCallBLAS` for BLAS/LAPACK). Fortran `print "..."; MPI_Abort` becomes
   `PetscCheck(cond, comm, PETSC_ERR_..., "same message")` or `SETERRQ`,
   keeping the message text. Kokkos entry points that return `void` are
   called as today.
3. **Data.** Derived types become structs in `pflareimpl.h` with the same
   type and field names, and explicit `create`/`reset`/`destroy` functions
   mirroring `create_air_data`/`reset_air_data`/`destroy_air_data`.
   Allocatable arrays become pointer plus size, `allocated(x)` becomes
   `x != NULL`. Use `PetscMalloc1`/`PetscCalloc1`/`PetscFree`; memory handed
   to users through the public API (polynomial coefficients) keeps today's
   contract (documented in `PCPFLAREINV.c`: freed with C `free`, so use
   `malloc` there). The Fortran idiom "`MatDestroy` nulls the local copy of
   the handle, copy it back into the slot" is reproduced where the comments
   point it out, because the reuse logic depends on which slots are null.
4. **Indices and integer kinds.** Internal loops are 0-based. Level numbering
   keeps PFLARE's 1-based `our_level` only where it is visible to the user
   (the stats print and the `petsc_level` arithmetic in `PCAIRGetPolyCoeffs`);
   document each place the two conventions meet. `PetscInt` for everything
   that can index a matrix, `int` only where the public API already says
   `int`. Options fields take the type of the corresponding public getter in
   `pflare.h`.
5. **Coefficient arrays** stay column-major (`coeffs[i + j*rows]`);
   `PCPFLAREINVGetPolyCoeffs` documents column-major.
6. **BLAS/LAPACK** through `petscblaslapack.h`: `LAPACKgels_`, `LAPACKgelsd_`,
   `BLASgemv_`, `BLASgemm_`, `LAPACKgeev_`, `LAPACKgesvd_`, `LAPACKgeqrf_`,
   `LAPACKgesv_` with `PetscBLASInt` and `PetscCallBLAS`, checking `info` as
   the Fortran does. Complex builds stay unsupported (the `#error` moves to
   `pflareimpl.h`).
7. **Array intrinsics.** `matmul`, `where`, `sum`, array sections and
   `reshape` become explicit loops in the same place in the routine; a dense
   `matmul` already surrounded by BLAS calls may become `BLASgemm_`. Keep the
   arithmetic order where it is cheap to do so.
8. **Printing.** `pflare_print(...)` (PETSc_Helper.c: `vfprintf` to the C
   stdout plus `fflush`, so the text reaches file descriptor 1 exactly as the
   Fortran `print *` did; not `PetscPrintf`, because petsc4py replaces
   `PetscVFPrintf` with a routine that writes through Python's `sys.stdout`,
   which in Jupyter is not descriptor 1, and the notebooks capture the stats
   by redirecting that descriptor) on rank 0 using the exact labels in
   `tools/parse_pflare_output.py` (`~~~~~~~~~~~~ Level %d`, `Global rows %d
   Global F-points %d Global C-points %d`, `<timer label> : %.15g`, the five
   complexity lines, `Total cumulative setup time : %.15g`).
9. **Kokkos.** C calls `*_kokkos` directly; the signatures in `src/*.kokkos.cxx`
   are already C and already take `Mat *`, `IS *`, `Vec *`. The
   `PFLARE_KOKKOS_DEBUG` compare paths are ported verbatim (`kokkos_debug()`
   becomes a C function in `PETSc_Helper.c`). The Kokkos test sequence of
   `AGENTS.md` runs for every stage that touches a module with a Kokkos sibling.
10. **Module state.** Fortran module variables (`Timers.F90` arrays,
    `TSQR.F90` custom MPI op, `PETSc_Helper.F90` debug flag) become
    file-static variables with the same lifetime semantics.
11. **Standard containers.** Where the Fortran hand-rolled something the C
    standard library or PETSc provides (binary tree used as a sorted set,
    Knuth shuffle to balance it), use `PetscSortInt`, `PetscSortedRemoveDupsInt`,
    `PetscSortIntWithArray`, `qsort` and friends. The result must be the same
    (a sorted unique set is unique). Nothing outside PETSc and libc.
12. **No behaviour improvements during a port stage.** A bug found while
    porting is recorded in the stage's PR description and fixed in a separate
    PR against both implementations (or after stage 8 against the C only), so
    the differential tests stay meaningful.
13. **Doctext.** The `/*MC M*/` and `/*@ @*/` manual-page comments stay in
    `PCAIR.c` and `PCPFLAREINV.c`; new C files do not need them.

## 6. Verification framework

Each stage ships with evidence, not a claim. Everything is built and run
inside the CI Docker images, which is also how a subagent gets PETSc,
gfortran, MPI and Kokkos without installing anything.

**Build environment.** Pull before every use (`docker pull`, the images track
PETSc `main`). The working tree is mounted over the image's copy of PFLARE:

```sh
# GNU, non-Kokkos; PETSC_ARCH=arch-linux-c-debug or arch-linux-c-opt
docker pull stevendargaville/petsc:latest
docker run --rm -v "$PWD":/build/PFLARE -w /build/PFLARE -e PETSC_ARCH=arch-linux-c-debug \
  -e PETSC_OPTIONS="-on_error_abort -fp_trap on" stevendargaville/petsc:latest bash -c '
    FL="-Werror -Wall -ffree-line-length-132 -ffixed-line-length-132 -fcheck=all -g"
    make -j3 build_tests FFLAGS="$FL" CFLAGS="-Werror -Wall" CXXFLAGS="-Werror -Wall" CPPFLAGS="-Werror -Wall" &&
    make python && make check && make tests'
# Kokkos (the base image sets neither PETSC_OPTIONS nor PFLARE_KOKKOS_DEBUG; pass what
# dockerfiles/Dockerfile_kokkos sets)
docker pull stevendargaville/petsc_kokkos:latest
docker run --rm -v "$PWD":/build/PFLARE -w /build/PFLARE -e PFLARE_KOKKOS_DEBUG=1 -e OMP_NUM_THREADS=1 \
  -e PETSC_OPTIONS="-mat_type aijkokkos -vec_type kokkos -dm_mat_type aijkokkos -dm_vec_type kokkos -on_error_abort" \
  stevendargaville/petsc_kokkos:latest bash -c '
    make -j3 build_tests FFLAGS="-Werror -Wall -ffree-line-length-132 -ffixed-line-length-132" \
      CFLAGS="-Wall -Werror -Wunused-result" CXXFLAGS="-Wall -Werror -Wunused-result -std=c++20" CPPFLAGS="-Wall -Werror" &&
    make check && make tests'
```

`docs/dev/ci.md` lists the other variants (64-bit, single precision, no-MPI,
valgrind, `-malloc_dump`) and the `dockerfiles/` build arguments that select
them; a stage runs the variants its section names, and the PR runs all of
them. The `-fcheck=all` flags matter while Fortran reference code is still
compiled; `-fsanitize=address,undefined` on `CFLAGS` is the C equivalent and
is used from stage 2 on in the debug image.

Three layers of checks, cheapest first.

**Layer A: the existing suite, on the CI matrix.** `make check`,
`make tests_short`, `make tests` (which includes the Python tests) in the
image, and the full workflow on the PR. For stages that touch a module with a
Kokkos sibling, additionally the Kokkos image with and without
`PFLARE_KOKKOS_DEBUG` (`AGENTS.md`, Tests 6 to 8). Valgrind and `-malloc_dump`
catch leaks in the new manual memory management; they already exist in CI.

**Layer B: golden logs (added in stage 0).** `tools/capture_test_logs.sh` runs
the `run_tests_*` groups with `-ksp_converged_reason -ksp_monitor` appended
through `PETSC_OPTIONS`, and `tools/compare_test_logs.py` extracts, per test
command, the iteration count, the residual history, and the parsed stats block
(via `parse_pflare_output.py`). No baselines are committed: a stage captures
the previous stage's branch and its own branch in the same image and compares
the two directly, and the `gnu_opt_golden_logs` CI job does the same for the
merge-base of the umbrella PR (`main`) and its head. The comparison rule: iteration counts and converged
reasons must be identical; residual norms agree to a relative tolerance
(default 1e-8 double, 1e-4 single); stats fields `global_rows`,
`global_f_points`, `global_c_points`, `n_levels` identical; complexities to
1e-10. Any miss is a stage failure until explained and signed off. Stage 9
(the random generator) is the only stage allowed to re-baseline.

**Layer C: in-process differential tests.** This is the repo's own
`PFLARE_KOKKOS_DEBUG` idea applied to the port. While a stage is in flight,
the retired Fortran implementation is kept compiled under `src/fref/`
(renamed `bind(C)` exports get an `_fref` suffix; module procedure names are
mangled by the compiler and do not clash). Fortran that uses a `src/fref`
module does so under `#ifdef PFLARE_PORT_DEBUG` (makedepf90 evaluates `#ifdef`
but not `#if defined`), and `make depend` writes the extra dependency lines of
that build inside `ifeq ($(PFLARE_PORT_DEBUG),1)`. With `make PFLARE_PORT_DEBUG=1`
the C code runs both itself and the Fortran reference and compares:

- leaf stages (2, 3, 6): the call site that was switched to C also calls the
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
- stage 7 (AIR hierarchy): a reference PC type `air_fref` built from today's
  `PCAIR.c` shell plumbing plus the Fortran `PCAIR_Shell` path;
  `tests/port_compare_pc.c -pc_type air` compares level by level:
  `IS_fine`/`IS_coarse` (`ISEqual`), restrictors, prolongators, `A_ff`,
  `A_cc`, coarse matrices (`MatEqual`, then norm of the difference),
  polynomial coefficients, and the result of `PCApply`, `PCApplyTranspose`
  and `PCMatApply`.

Because the random draws stay in Fortran until stage 9 (section 1, decision
1), layers B and C compare bit-for-bit except where a reduction order changed,
which keeps the differential tests sharp. The differential drivers are part
of `TEST_TARGETS` only while `PFLARE_PORT_DEBUG=1`, and are deleted with
`src/fref/` in stage 8.

## 7. Stages

Dependency graph (an arrow means "must be merged first"):

```
0 harness ─> 1 build ─> 2 leaves ─> 3 helpers ─┬─> 4 CF splitting ─┐
                                               ├─> 5 inverses ─────┼─> 7 AIR core ─> 8 drop fref ─> 9 RNG ─> 10 gating/CI/docs
                                               └─> 6 grid transfer ┘
```

Branch workflow: nothing reaches `main` until the port is complete. Each
stage is developed by one subagent on its own branch,
`claude/fortran-port-stage-N`, created from the previous stage's branch (with
`main` merged in first). When a stage is reviewed, its branch is force-pushed
over `claude/loving-volta-ap1rin`, the head of the single umbrella PR
(PFLAREProject/PFLARE#300), so CI runs on the cumulative port and the PR's
commit list shows the stage; the PR is only merged after the last stage. The
stage branches stay as the record of each step. Stages 4, 5 and 6 are
independent of each other and could run concurrently off the stage 3 branch;
their only shared edits are the `OBJS` lists and the `C_Fortran_Bindings.F90`
extern block. Everything else is sequential. The subagent's brief is the
stage section below plus sections 3, 5 and 6 of this file. Sizes are lines
of Fortran going in; the C coming out is usually 1.1 to 1.3 times that.

Merges from main: when `main` moves while the port is open, the current port
branch takes it with a merge commit (not a rebase) on a new branch,
`claude/fortran-port-merge-main-N`, so the stage branches keep their history. Git
follows the renames, so `main`'s hunks to the retired Fortran land in the
`src/fref` copies and in `src/fortran`; a file the port replaced (for instance
`src/Pflare_Parameters.F90`, now `src/pflare_parameters.h` plus
`src/fortran/Pflare_Parameters.F90`) is resolved by moving `main`'s change into
its replacement. The merge is followed by commits that re-port every Fortran
hunk into the live C line for line (convention 1), and that apply `main`'s hunks to
the C files the reference keeps as copies (`src/fref/PCPFLAREINV_fref.c`,
`src/fref/PCAIR_fref.c`), so the reference stack is exactly `main`. New option
values join the type lists of the `port_compare_*` drivers (new option sets are
appended to `port_compare_air`, so the `-configs` indices of the run lines keep
their meaning). The checks are layer C (`PFLARE_PORT_DEBUG=1` in the debug and
Kokkos images, the C against the reference that now carries `main`'s Fortran)
and layer B against `main` itself: a worktree of `origin/main` and the merge
branch built and captured in the same image for `gnu_debug`, `gnu_opt` and
Kokkos, with no differences, which is what the `gnu_opt_golden_logs` CI job
checks on the umbrella PR. The first such merge brought in PR 301 (the
`l1jacobi` inverse type and hypre's `4 / (3 ||D^-1/2 A D^-1/2||_inf)` weighted
Jacobi weight): `calculate_and_build_weighted_jacobi_inverse` takes the inverse
type instead of a flag, `calculate_l1_jacobi_inverse_diagonal` is new in
`Weighted_Jacobi.c` (with a `Port_Interfaces.F90` wrapper that the reference
`CR_Splitting.F90` calls), and `port_compare_air` options sets 59 to 63 cover
`l1jacobi` as the Aff, C point and coarsest inverse and the CR splitting with
`l1jacobi` and `wjacobi`.

Every stage brief has the same closing requirement: build warning-free in the
debug and opt GNU images (and the Kokkos image where the stage says so), run
layer A, run layers B and C where listed, and report the results verbatim in
its report (which becomes the stage's entry in the umbrella PR description).
A stage is not done while any CI job on the umbrella PR is red.

### Stage 0: verification harness and the random-number wrapper (no algorithm change)

Deliverables

- `tools/capture_test_logs.sh` and `tools/compare_test_logs.py` as described
  in section 6, wired into a `make tests_golden_capture` target and a CI job
  that captures the merge-base and the head in the same container, compares
  them and uploads the head capture.
- `PFLARE_PORT_DEBUG` make variable plumbed through to `CPPFLAGS`/`FPPFLAGS`
  as `-DPFLARE_PORT_DEBUG`, the `src/fref/` directory added to the build when
  set (empty for now), and `tests/port_compare_pc.c`, `tests/port_compare_cf.c`
  skeletons that build and exit 0.
- `src/Pflare_Random.F90`: module `pflare_random` with one routine
  `pflare_random_fill(comm_rank, n, out) bind(C)` that does exactly what the
  four call sites do today (`random_seed(size=)`, `seed(i) = comm_rank + 1 + i`,
  `random_seed(put=)`, `random_number(out)`), and the four call sites in
  `PMISR_Module.F90` (two), `DDC_Module.F90`, `CR_Splitting.F90` and
  `Gmres_Poly.F90` switched to call it. Bitwise identical by construction
  (same calls in the same order; the Box-Muller transform in `Gmres_Poly.F90`
  stays at the call site). A header `src/Pflare_Random.h` declares the C
  prototype for the later C ports.

Verification: layer A on the full matrix, layer B identical (the random
wrapper must not move a single iteration count).

### Stage 1: build system restructure (no behaviour change)

- Split `OBJS` into `COBJS`, `KOBJS`, `FCORE_OBJS`, `FOBJS`; add the
  `PETSC_USE_FORTRAN_BINDINGS` detection (unused until stage 10); add
  `src/fortran/` and `src/fref/` to the search path and `clean`; add
  `$(CC) -MM` dependency generation to `make depend`; keep linking with
  `LINK.F` for now (switching the linker is stage 10).
- Move `PFLARE.F90`, `PCPFLAREINV_Interfaces.F90`, `PCAIR_Interfaces.F90`,
  `PCAIR_C_Fortran_Bindings.F90` and `Pflare_Parameters.F90` to
  `src/fortran/` (git renames). Update `Makefile.deps`; the `.mod` files
  still land in the top directory, where `tests/` and `python/` find them.
- Add `src/pflare_parameters.h` with every constant and enum value from
  `Pflare_Parameters.F90` as `#define`s, and rewrite the Fortran module to
  take its values from that header (the Fortran `parameter` statements stay,
  so `use pflare_parameters` is unchanged). Values must be copied exactly,
  including the precision-dependent `PFLARE_TOL_*` branches.
- Add `src/pflareimpl.h` (empty struct section, the complex `#error`, the
  include of `pflare_parameters.h`).

Verification: full CI matrix, layer B identical, `make install` output
directory listing identical to before (compare with `diff -r`).

### Stage 2: leaf modules (1.5k lines ported, 0.5k replaced)

Port `Timers.F90`, `Sorting.F90`, `Aggregation.F90`, `Weighted_Jacobi.F90`,
`TSQR.F90` to `src/<Stem>.c` + `.h`. These depend only on `pflare_parameters`
and PETSc. For each, the Fortran callers are switched to the C version through
a `bind(C)` interface block placed in a new `src/fortran/Port_Interfaces.F90`
(this file grows through stages 2 to 6 and is deleted in stage 7). The retired
Fortran file moves to `src/fref/` and keeps compiling.
`TSQR.F90` was moved out of this stage into stage 5, because its `tsqr_buffers`
live inside `gmres_poly_data`, which stage 5 owns; the TSQR notes below apply there.

Notes for the subagent

- `Binary_Tree.F90` is not ported. Its only use is
  `create_knuth_shuffle_tree_array` + `itree2vector` in `SAI_Z.F90`, which
  builds a sorted set of unique column indices (the shuffle only balances the
  tree). `Sorting.c` provides `sorted_unique(array, n)` (in place, `n`
  in/out) on `PetscSortInt` + `PetscSortedRemoveDupsInt`; `SAI_Z.F90` calls it through
  `Port_Interfaces.F90` from this stage on, and `Binary_Tree.F90` goes to
  `src/fref/` with the rest. `merge_pre_sorted` and
  `intersect_pre_sorted_indices_only` are ported as they are (PETSc has no
  merge/intersect helpers); `create_knuth_shuffle` has no remaining caller
  and is dropped.
- `TSQR.F90` holds a custom `MPI_Op` created once (`built_custom_op_tsqr`) and
  uses `MPI_Iallreduce`/`MPI_Ibcast` with request handles stored in
  `tsqr_buffers`; the C struct must keep the asynchronous protocol (start in
  one routine, wait in another) exactly.
- `Timers.F90` module arrays become `static PetscReal` arrays; the ids in
  `pflare_parameters.h`.
- `gmres_poly_data_type` holds a `tsqr_buffers` component. Until stage 5 that
  component must stay usable from Fortran: make the C `tsqr_buffers` opaque
  behind a `type(c_ptr)` in `gmres_poly_data` and give `Gmres_Poly.F90` the
  `bind(C)` create/destroy/start/finish calls it needs.

Verification: layers A and C (call-site compare under `PFLARE_PORT_DEBUG`),
layer B identical.

### Stage 3: PETSc helpers and repartitioning (2.3k lines)

Port `PETSc_Helper.F90` and `Repartition.F90` to C and fold
`C_PETSc_Routines.c` into `PETSc_Helper.c`; `kokkos_debug()`,
`remove_small_from_sparse`, `remove_from_sparse_match`,
`MatCreateSubMatrixWrapper`, `destroy_matrix_reuse` and the rest become
`PETSC_INTERN` C functions returning `PetscErrorCode`. The public
`remove_from_sparse_match` in `PCAIR.c` calls the C version directly and
drops `PetscInitializeFortran`. All helper routines get `bind(C)` interfaces in
`Port_Interfaces.F90` for the Fortran callers that remain (nearly every
module uses `petsc_helper`), so this stage is mostly plumbing and must be done
before 4, 5 and 6.

Notes: `PETSc_Helperk.kokkos.cxx` is the sibling; the `PFLARE_KOKKOS_DEBUG`
compare paths in the helper move with it. `MatMPICreateNonemptySubcomm_c`
already returns `PetscErrorCode`. The base branch rewrote `PETSc_Helper.F90`
to build CPU matrices with `MatSetValues` rather than COO; port that version.

As done: `C_PETSc_Routines.c` is not folded into `PETSc_Helper.c`. The three
helpers only the repartitioning used (`MatPartitioning_c`,
`GenerateIS_ProcAgglomeration_c`, `MatMPICreateNonemptySubcomm_c`) moved to
the end of `Repartition.c` and return `PetscErrorCode`; the ones the Fortran
still calls (`c_PCGetStructureFlag`, `PCGetSetupCalled_c`, `PCMarkNotSetUp_c`,
`MatGetDiagonalOnly_c`, `PetscFree_c`) stay in `C_PETSc_Routines.c` until
stage 7. The C port of `remove_from_sparse_match` is called
`remove_from_sparse_match_c`, as `include/pflare.h` already has a public
`remove_from_sparse_match` (stage 4 meets the same clash with
`compute_cf_splitting` and `compute_diag_dom_submatrix`). Optional Fortran
arguments are pointers in C, NULL when absent. The Kokkos kernel prototypes
the C calls are in `src/PETSc_Helperk.h`, which the kernel file includes too.
The Fortran interfaces that only `src/fref` still uses (the six kernels and the
three repartitioning helpers) are compiled in `C_PETSc_Interfaces.F90` under
`PFLARE_PORT_DEBUG` only. For the call-site compares, a copy of a parallel AIJ
matrix that the reference sets values in has to be rebuilt with `MatSetValues`:
`MatDuplicate` does not mark the duplicate as assembled before
(`was_assembled`), so `MatSetValues` reads the global columns of the
off-diagonal block as the local ones of its compressed storage.

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

Random numbers: `pmisr`, `ddc` and `cr` call `pflare_random_fill` from
`Pflare_Random.h` at the same point and with the same sizes as the Fortran.

Verification: A with both Kokkos runs (PMISR, DDC and MatDiagDom have Kokkos
siblings and `PFLARE_KOKKOS_DEBUG` paths), B identical,
`tests/port_compare_cf.c` under `PFLARE_PORT_DEBUG` for every
`CFSplittingType` on `tests/data` matrices and on the generated advection
problems, serial and `mpiexec -n 2`/`-n 4`.

As done: the C files keep the module stems, and the C names of the two public
routines are `compute_cf_splitting_c` and `compute_diag_dom_submatrix_c` (the
stage 3 clash rule). Besides `cf_splitting`, `matdiagdomsubmatrix` stays a
Fortran module, as `docs/new_methods.md` documents `compute_diag_dom_submatrix`
for Fortran users: both are wrappers in `src/fortran/` (the `CF_*` parameters stay
in `cf_splitting`), and their `src/fref/` copies are renamed `cf_splitting_fref`
and `matdiagdomsubmatrix_fref`, the only edits to `src/fref`, since one build
cannot hold two modules of the same name. No other remaining Fortran called the
ported routines, so the call-site compare is in the two wrappers (`ISEqual`, and
`port_compare_mats` row by row), and `tests/port_compare_cf.c` compares the public
C API against the old bindings, kept as `compute_cf_splitting_c_fref` and
`compute_diag_dom_submatrix_c_fref`. The kernel prototypes are in
`PMISR_Modulek.h`, `DDC_Modulek.h`, `MatDiagDomk.h` and `Device_Datak.h`, and
their Fortran interfaces are compiled for the reference only. `cr_pass` passes
NULL coefficients to `calculate_and_build_approximate_inverse_c` and frees the
copy it returns, and calls `PetscInitializeFortran` first, because the Fortran it
reaches needs the Fortran PETSc constants and the public `compute_cf_splitting`
no longer sets them; stage 5 removes both. The arrays handed to
`pflare_random_fill` are allocated with at least one entry: `PetscMalloc1`
gives NULL for zero entries where a Fortran zero-size array has an address, and
UBSan reports the NULL in the Fortran on a rank without rows (stage 5's GMRES
start vector needs the same). `make depend` cannot produce the
Fortran lines any more: `makedepf90 -b src` gives `src/fortran/CF_Splitting.F90`
and `src/fref/CF_Splitting.F90` the same object, which `FDEP_RELOCATE` cannot
tell apart (stage 5 meets the same with `Gmres_Poly_Newton.F90`), so they are
edited by hand, as before.

### Stage 5: polynomials and approximate inverses (7.5k lines)

Port `Gmres_Poly_Data_Type.F90`, `Matshell_Data_Type.F90`,
`Gmres_Poly_Apply.F90`, `Gmres_Poly.F90`, `Gmres_Poly_Newton.F90`,
`Neumann_Poly.F90`, `SAI_Z.F90`, `Approx_Inverse_Setup.F90`, plus `TSQR.F90`
(moved here from stage 2, see the TSQR notes there). After this stage
`PCPFLAREINV.c` is pure C: `calculate_and_build_approximate_inverse`,
`reset_inverse_mat`, `inverse_type_is_polynomial` and
`pflareinv_shell_block_matapply` are C functions, and the `PetscInitializeFortran`
call goes. The matrix-free polynomial `MatShell` operations (`MATOP_MULT`,
`MATOP_MULT_TRANSPOSE`, the block applies, the lazily built transpose twin)
are C callbacks on a `mat_ctxtype` struct with the same fields as today
(including `air_data` and `our_level`, which the Fortran `FC_Smooth` still
sets until stage 7; `air_data` is a `void *` until then).

Data crossing the boundary until stage 7: `air_multigrid_data` holds arrays of
`gmres_poly_data` and the AIR Fortran modules reference `%coefficients` and
`%buffers` in about 110 places. Make the Fortran `gmres_poly_data` a
`type, bind(C)` mirror of the C struct (`type(c_ptr) :: coefficients;
PetscInt :: rows, cols; integer(c_int) :: inverse_type, gmres_poly_order,
gmres_poly_sparsity_order; type(c_ptr) :: buffers`) and give the AIR modules
a small helper `poly_coeffs(pd)` returning a contiguous 2D Fortran pointer via
`c_f_pointer`, so the existing `%coefficients` expressions become one-line
edits. The 18 `mat_ctx%` references in `FC_Smooth*.F90` and `AIR_MG_Setup.F90`
go through `bind(C)` accessors.

`modified_leja` keeps its Fortran name in `src/fortran/Gmres_Poly_Newton.F90`
as a wrapper that passes the three assumed-shape arrays by base address and
size to the C `modified_leja`.

Random numbers: the GMRES polynomial start vector calls `pflare_random_fill`
for the two uniform columns and does the Box-Muller transform in C, in the
same order as `Gmres_Poly.F90`.

Verification: A with both Kokkos runs (`Gmres_Polyk`, `Gmres_Poly_Newtonk`,
`SAI_Zk` siblings), B identical, `tests/port_compare_pc.c -pc_type pflareinv`
under `PFLARE_PORT_DEBUG` for every `PCPFLAREINVType`, assembled and
matrix-free, with and without diagonal scaling and `-pc_pflareinv_subcomm`,
serial and parallel, plus `PCApplyTranspose` and `PCMatApply`. The Python
PFLAREINV tests (`ex_pcpflareinv_options.py`, `ex6f_getcoeffs.py`) exercise
the coefficient memory contract.

As done: the AIR Fortran calls the C through `src/fortran/Port_Interfaces_Poly.F90`
(same names and arguments, except that `start/finish_approximate_inverse` take the
`gmres_poly_data` instead of its `%buffers` and `%coefficients`); the coefficients
are `PetscMalloc`'d by the C, the TSQR buffers are created on first use, and
`reset_poly_data`/`destroy_poly_data` replace the Fortran deallocations. The
placeholder matshells of the FC smoothers get a C `mat_ctxtype` from
`mat_ctx_create_shell` (on the comm of a matrix, as a Fortran comm handle would
need converting) and are read back through `mat_ctx_get_our_level`/`_air_data`.
`PCPFLAREINV.c` keeps the old binding's coefficient contract in a static helper
(a `malloc`'d copy freed with `free`), and the C entry points
`tests/shell_block_apply.c` used (`calculate_and_build_approximate_inverse_c`,
`reset_inverse_mat_c`, `pflareinv_shell_block_matapply_c`) stay exported from it in
C. The fref modules with a live namesake are `gmres_poly_data_type_fref` and
`gmres_poly_newton_fref`; the old bindings are kept with an `_fref` suffix for
`pflareinv_fref` (`src/fref/PCPFLAREINV_fref.c`, the pre-port PC, registered only
under `PFLARE_PORT_DEBUG`). Layer C is in the wrappers (coefficients bitwise,
inverses row by row or, matrix-free, by applying both and their transposes) and
in `tests/port_compare_pc.c`, which covers diagonal scaling and the subcomm
through the `_c`/`_c_fref` entry points, as PCPFLAREINV has neither option (there
is no `-pc_pflareinv_subcomm`). Everything compares bitwise. The kernel
prototypes are in `Gmres_Polyk.h`, `Gmres_Poly_Newtonk.h` and `SAI_Zk.h`. `make depend`
still cannot tell `src/fortran/X.F90` and `src/fref/X.F90` apart, so the Fortran
lines of `Makefile.deps` were taken from `makedepf90` without `-b` (which has the
right objects), in the order `make depend` gives them.

### Stage 6: grid transfer and constraints (1.8k lines)

Port `Grid_Transfer.F90`, `Grid_Transfer_Improve.F90`, `Constrain_Z_or_W.F90`.
They take matrices, index sets and scalars only (no `air_data`), so the
boundary is clean. `Grid_Transferk.kokkos.cxx` is the sibling. The dense BLAS
calls in `Constrain_Z_or_W.F90` use `BLASgemv_`/`BLASgemm_`. The base branch
rewrote `Grid_Transfer.F90` to build with `MatSetValues`; port that version.

Verification: A with both Kokkos runs, B identical, C call-site compare
(`MatEqual` on the produced R/P/W/Z, elementwise tolerance after `improve`).

As done: the C files keep the module stems and the routine names (none clashes with
`include/pflare.h`). `AIR_MG_Setup.F90` and `AIR_Operators_Setup.F90` call them through
`src/fortran/Port_Interfaces_Grid.F90` (module `port_interfaces_grid`, separate from
`port_interfaces` for the reason `port_interfaces_poly` is) with the Fortran signatures
unchanged; arrays of Vecs cross as arrays of handles with their size, and the two arrays
`get_near_nullspace` returns are allocated by the C (`PetscCalloc1`, NULL for a Vec it does
not create) and copied into the Fortran allocatables. No live Fortran module keeps the names,
so the fref modules are not renamed. The kernels are declared in `Grid_Transferk.h` and their
Fortran interfaces compiled for the reference only. Layer C is in the wrappers, bitwise
everywhere (no arithmetic order changed: `improve_w`/`improve_z` are PETSc calls, the BLAS
calls take the Fortran arguments): every matrix produced (the one-point W, P, R, the improved
W/Z, the reuse matrices `improve_*` created, the constrained W/Z) row by row, the
`orig_fine_col_indices` the C created with `ISEqualUnsorted`, the near nullspace vectors with
`VecEqual`. The reference runs on copies of what the C refills (P and R when reusing, W/Z in
`improve_*`, the matrix and vectors `constrain_grid_transfer` takes, the smoothed vectors),
taken before the C runs (its `VecGetArray` drops the cached norms of the vectors only on the
ranks with rows, so a later `VecNorm` aborts on norms cached on some ranks); the reference of
`improve_*` starts without the reuse matrices, as a copy cannot carry the `MatProduct` the C
reuses and reusing it gives the same values. `constrain_grid_transfer` compares the type name
of `row_mat` once, before it may be destroyed: the Fortran compares it again at the end, but a
C `MatType` points into the object. `MatNullSpaceRestoreVecs` is Fortran only. The stage 6
lines of `Makefile.deps` are those `make depend` gives; the lines of the same-stem files stay
as stage 5 left them.

### Stage 7: AIR hierarchy and the PCAIR Fortran interface (6.2k lines ported, 2.8k rewritten, 2.3k deleted)

Port `AIR_Data_Type.F90`, `AIR_Data_Type_Routines.F90`, `FC_Smooth.F90`,
`FC_Smooth_Block.F90`, `AIR_MG_Stats.F90`, `AIR_Operators_Setup.F90`,
`AIR_MG_Setup.F90`, `PCAIR_Shell.F90`, `PCAIR_Data_Type.F90`. Then:

- `PCAIR.c` owns `pc_air_multigrid_data` (`air_multigrid_data` + the inner
  `PCMG`) directly in `pc->data`; the `PCShell` layer is removed. The
  setup logic of `PCSetUp_AIR_Shell` (first setup, `DIFFERENT_NONZERO_PATTERN`
  reset, `SAME_NONZERO_PATTERN` with or without `reuse_sparsity`) moves into
  `PCSetUp_AIR_c` reading `pc->flag` and `pc->setupcalled` directly, so
  `c_PCGetStructureFlag`, `PCGetSetupCalled_c`, `PCMarkNotSetUp_c` and the
  shell's `reusepreconditioner` sync disappear. `PCReset_AIR_c` becomes
  `reset_pc_air_data`; apply, apply transpose, mat apply, view and destroy
  call the ported shell routines directly. The options prefix is passed to
  the PCMG as today.
- All 53 getters and 47 setters in `PCAIR.c` read and write the C options
  struct directly, with the same "reset the hierarchy if the value changed"
  logic that `PCAIR_Interfaces.F90` has today (`PCAIRSetMaxLevels` is the
  template: compare, reset the data, mark the PC not set up). Delete the
  `_c` externs and `c_PCAIRGetPCShell`.
- `src/fortran/PCAIR_Interfaces.F90` is rewritten in the style of
  `PCPFLAREINV_Interfaces.F90`: every remaining public routine keeps its
  signature and calls the C function through a `bind(C)` interface,
  converting `PetscBool` and `PetscEnum` as that file does.
  `PCAIRGetPolyCoeffs` and `PCAIRSetPolyCoeffs` keep their allocatable-array
  semantics by copying through the C getter/setter. `PCAIRGetOptions`,
  `PCAIRGetOptionsAndShell`, `PCAIRGetPCShell` and `PCAIRGetPolyCoeffsPointer`
  are removed (decision 3).
- Delete `C_Fortran_Bindings.F90`, `PCAIR_C_Fortran_Bindings.F90`,
  `C_PETSc_Interfaces.F90`, `Port_Interfaces.F90`; drop every remaining
  `PetscInitializeFortran`.
- Stats printing follows convention 8.

Notes for the subagent

- `air_multigrid_data` has 22 reuse matrix slots and 2 IS slots with the
  `REUSE_MAT_ACTIVE`/`REUSE_IS_ACTIVE` tables; keep them as `static const`
  tables in `pflareimpl.h` with the same indices and the same row comments.
- `fast_veciscopy_exists` and `kokkos_is_views_handle` wire into
  `VecISCopyLocalk.kokkos.cxx`; the handle is already a `void *`.
- The base branch pools the smoothing temporaries across levels and stops
  creating a distinct up smoother when FC smoothing; port those versions.
- The timers (`Timers.c`) and stats (`AIR_MG_Stats.c`) are the only places the
  1-based `our_level` is user visible.
- `tests/reuse_preconditioner.c`, `pcair_reuse_amount_range.c`,
  `operator_refcount.c`, `comm_self_then_world.c` and the `-check_copies`
  drivers are the sharpest tests of lifetime and reuse handling; run
  `make tests` (not just `tests_short`) in the image before pushing.
- The reference type `air_fref` for layer C is today's `PCAIR.c` shell
  plumbing copied to `src/fref/PCAIR_fref.c` and registered under that name,
  so both hierarchies can be built in one process.

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

As done (part a; part b moves the data into `pc->data` and removes the `PCShell`): the nine
modules are C files with the module stems, the Fortran moved unchanged to `src/fref` and built
only with `PFLARE_PORT_DEBUG=1`. The structs are in `pflareimpl.h` (`air_options` with the
public getter types, `air_multigrid_data` with 0-based levels and per-level arrays of
`levels_size` entries, `air_reuse_data`, `pc_air_multigrid_data`, the `REUSE_*_ACTIVE` tables
indexed `[MAT_*][reuse_amount - 1]`), and `mat_ctxtype.air_data` is a typed pointer. For part a
`pc->data` is still the `PCShell`, whose context is the C `pc_air_multigrid_data` and whose
callbacks are the C ports of `PCAIR_Shell.F90` (`PCSetUp_AIR_Shell` reads `pc->flag` and
`pc->setupcalled` of the shell). The five entry points `PCAIR.c` reached through
`C_Fortran_Bindings.F90` keep their names in `PCAIR_Shell.c` (`create_pc_air_data_c`,
`PCReset_AIR_Shell_c`, `create_pc_air_shell_c`, `pcair_shell_get_pcmg_c`,
`pcair_shell_block_matapply_c`) and return a `PetscErrorCode` (the last loses its
`error_code` argument). The getters and setters mirror the Fortran ones, except that the C
errors with `PetscCheck` where the Fortran printed and called `MPI_Abort` (`PCAIRSetSmoothType`,
`PCAIRGet/SetPolyCoeffs`); the Fortran `PCAIRSetPolyCoeffs` still aborts on an unassociated
`coeffs`. `C_Fortran_Bindings.F90`, `C_PETSc_Interfaces.F90`, `PCAIR_C_Fortran_Bindings.F90`
and a copy of the old `PCAIR_Interfaces.F90` (module `pcair_interfaces_fref`) are in `src/fref`
with `_fref` bind names, under the reference PC type `air_fref` (`src/fref/PCAIR_fref.c`, the
old `PCAIR.c`); `c_PCGetStructureFlag`, `PCGetSetupCalled_c`, `PCMarkNotSetUp_c` and the stage 5
matshell context accessors are `PFLARE_PORT_DEBUG` only. `src/fortran/Port_Interfaces*.F90` and
`Gmres_Poly_Data_Type.F90` stay (no live Fortran uses them outside `PFLARE_PORT_DEBUG`) and go
with the reference in stage 8; `FCORE_OBJS` is `Pflare_Random.o`. Layer C is a second driver,
`tests/port_compare_air.c`, comparing `air` against `air_fref` bitwise over a list of option
sets (every getter, the coefficients of every level and `COEFFS_INV_*`, `PCApply`,
`PCApplyTranspose`, `PCMatApply`) after the first setup, a `SAME_NONZERO_PATTERN` setup with the
reuse of the sparsity off and with each reuse amount, `PCAIRSetPolyCoeffs` with
`PCAIRSetReusePolyCoeffs`, `PCReset`, and options changed by the setters (every option set in serial on the generated
operators, subsets of them in parallel and on `mat_stream_2364`, through `-configs`). Two option sets fail
with a reuse amount of 3 in the Fortran too (symmetric, and auto truncation once the
coefficients are reused), so they stop at 2; the coefficients of the non-GMRES inverses and of
the unused dropped `Aff` inverse are left uninitialised by both, so only their sizes are compared.
It runs on the generated operators and `mat_stream_2364`, not `e05r0100`: on that matrix the
stage 3 and 5 checks inside the reference stack (`remove_from_sparse_match`,
`finish_approximate_inverse`) stop some option sets, and ISAI and lAIR hit singular local
systems in both. `-pc_air_subcomm` is compared with the PMIS splitting and the Arnoldi basis,
as it errors in parallel on the `MATDIAGONAL` `Aff` of the PMISR DDC splitting (in the Fortran
before the port too), and not on Kokkos matrices, where it errors in both
(`MatMPICreateNonemptySubcomm_c` calls `MatMPIAIJGetSeqAIJ` on a `MATMPIAIJKOKKOS`).
`-pc_air_symmetric` is compared in serial only: its `MatPtAP` is not deterministic in parallel
(the Fortran before the port gives different roundoff from run to run too), and its
`MAT_REUSE_MATRIX` `MatPtAP` errors in the C where the Fortran ignored the error.

### Stage 8: remove the reference Fortran

- Delete `src/fref/`, the `PFLARE_PORT_DEBUG` plumbing, and
  `tests/port_compare_*.c`. `PFLARE_KOKKOS_DEBUG` remains the only
  in-process compare mode.
- Delete `include/finclude/pflare_blaslapack.h`.
- At this point `src/Pflare_Random.F90` is the only core Fortran file.

Verification: A on the full matrix, B identical.

### Stage 9: the random generator, the last core Fortran

- Replace `src/Pflare_Random.F90` with `src/Pflare_Random.c`. Preferred
  generator: xoshiro256** with gfortran's `random_seed(put=)` scramble,
  which can be checked bitwise against the Fortran module before it is
  deleted (a throwaway `tests/random_compare.F90` fills the same arrays with
  both and compares). If it matches, the GNU baselines do not move and Intel
  and Cray builds become identical to GNU. If it does not match, use
  `PetscRandom` (`PETSCRANDER48`) seeded from `comm_rank + 1` as the Fortran
  seeds are, and re-capture the layer B baselines; adjust any `-ksp_max_it`
  in `tests/Makefile` that no longer bounds the iteration count (expected to
  be rare and by one or two iterations; list each change in the PR).
- Empty `FCORE_OBJS`.
- `CHANGELOG.md`: one entry describing the C core, the error-handling change
  (decision 2), the removed Fortran-only accessors and shell layer
  (decision 3), the `-ksp_view` change (decision 4) and, if applicable, the
  random generator change.

Verification: A on the full matrix, B identical to the (possibly new)
baseline, and `grep -rl random_number src/` is empty.

### Stage 10: gating, CI, docs

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
  the C core. `dockerfiles/Dockerfile_no_fortran` plus a base image built
  without Fortran. The `FFLAGS=-Werror ...` arguments stay for the
  Fortran-enabled jobs.
- Docs: `docs/installation.md` gains "Building without a Fortran compiler";
  `docs/dev/kokkos.md` conventions say `X.c` / `Xk.kokkos.cxx`;
  `docs/dev/testing.md` and `docs/dev/ci.md` describe the gate and the new
  jobs; `AGENTS.md` codebase map updated (C is the bulk of the algorithms,
  `src/fortran/` is the interface layer, magic constants in
  `src/pflare_parameters.h`); `python/pflare_defs.pyx` comments updated.
  `README.md` still lists the Fortran interface.
- Spack: `PFLARE_spack/package.py` (separate repository) can later drop its
  Fortran requirement; note it in the CHANGELOG, no change here.

Verification: full CI matrix including the two new jobs; in the no-Fortran
image `make`, `make check`, `make tests`, `make python`, `make install` all
succeed and `make install` writes no `.mod`; `nm libpflare.so | grep -i
_gfortran` is empty.

## 8. Risks

| Risk | Where | Mitigation |
|---|---|---|
| Lifetime bugs (double destroy, leak, use after reset) in the hand-managed C structs | stages 5, 7 | explicit create/reset/destroy trio, valgrind + `-malloc_dump` CI, `-fsanitize` in the debug image, `operator_refcount` and reuse tests, the handle-nulling idiom reproduced (convention 3) |
| `PetscInt`/`int` width mix on 64-bit index builds | everywhere | convention 4, the `gnu_opt_64_bit` and `gnu_opt_64_bit_kokkos` jobs |
| Single precision | everywhere | PETSc BLAS macros, `PetscReal` only, the `gnu_debug_kokkos_single_prefix` job and the `SKIP_SINGLE` gates stay |
| Summation-order drift masquerading as a bug, or hiding one | stages 5, 6, 7 | layer C compares with named tolerances; any difference above `PFLARE_TOL_*` is investigated, not widened; random draws are shared so they cannot be the cause |
| The in-process reference (`src/fref/`) drifting from the base while a stage is open | stages 2 to 7 | the reference is never edited; bug fixes land after the stage or in both (convention 12) |
| Output parser contract | stage 7 | `make tests_python` runs `run_parse_tests.py`; convention 8 |
| Stage 7 is large | stage 7 | the brief orders it: data type + routines, smoothers, stats, operators setup, MG setup, shell routines into `PCAIR.c`, getters/setters, Fortran interface; `air_fref` gives level-by-level diagnosis |
| PETSc doc build in the `gnu_petsc_config` job breaking on moved doctext | stages 4, 5, 7 | keep the `/*MC M*/` and `/*@ @*/` blocks in `PCAIR.c`/`PCPFLAREINV.c`; the job already checks the rendered pages exist |
| macOS link of a C++-containing shared library without the Fortran driver | stage 10 | use `$(CXXLINKER)` when `KOBJS` is non-empty; the `macos_64_bit_kokkos` job covers it |
| Base branch moves under the plan (PETSc `main` changes, PFLARE fixes) | all | each stage branches from the current `main` plus merged stages; the Docker images are re-pulled before every build |

## 9. Open decisions

- **Stage parallelism.** Whether to run stages 4, 5 and 6 concurrently (three
  subagents, three branches, merge order 6, 4, 5) or strictly sequentially.
  Sequential is simpler to review; concurrent saves wall-clock time at the
  cost of two small rebases.
- **Stage 9 generator** if the xoshiro256** reimplementation does not match
  gfortran bitwise: `PetscRandom` as written above, or keep the xoshiro code
  anyway for compiler independence. Either way the re-baseline happens once,
  in stage 9, and nowhere else.
