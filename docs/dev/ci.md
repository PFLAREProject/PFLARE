# CI (agent reference)

Workflow: `.github/workflows/ci_build.yml` (push/PR to main, monthly cron, manual dispatch). Every job builds a Docker image and runs the full test suite inside it.

Jobs, grouped
- GNU debug/opt: `gnu_debug`, `gnu_opt` (pushes `stevendargaville/pflare:latest` on non-PR events), `gnu_opt_64_bit`.
- Leak/memory checks: `*_malloc_dump` variants run with `-malloc_dump` and fail if `PetscTrMalloc` appears in the test log; `gnu_debug_valgrind`, `gnu_debug_kokkos_valgrind`.
- Fortran runtime checks: `gnu_debug` builds with `EXTRA_FFLAGS="-fcheck=all -g -fbacktrace"` (a `dockerfiles/Dockerfile` build arg) and runs the full suite, so out-of-bounds indexing, unallocated/unassociated arrays and non-conforming array assignments abort with a `Fortran runtime error`. To reproduce locally, add those flags to `FFLAGS`.
- Kokkos: `gnu_opt_kokkos`, `gnu_opt_64_bit_kokkos`, `gnu_opt_omp_kokkos`, `gnu_debug_no_mpi_kokkos`, `gnu_debug_kokkos_single_prefix` (single precision).
- Other: `intel_opt`, `macos_debug`, `macos_64_bit_kokkos`, `notebook_tutorial`, `gnu_petsc_config` (builds PETSc `main` with `--download-pflare` and gates the Sphinx doc build).

Images
- Base images: `stevendargaville/petsc` (`dockerfiles/Dockerfile`) and `stevendargaville/petsc_kokkos` (`dockerfiles/Dockerfile_kokkos`); published result: `stevendargaville/pflare:latest`.
- Inside the image: PFLARE at `/build/PFLARE`, tests at `/build/PFLARE/tests`, PETSc at `/build/petsc`.
- To reproduce a CI failure locally: `docker pull` the latest image first, then run the failing make target inside it.

Enforced flags (why "fix all compile warnings" is a hard rule)
- Non-Kokkos builds: `FFLAGS="-Werror -Wall -ffree-line-length-132 -ffixed-line-length-132"`, `CFLAGS/CXXFLAGS/CPPFLAGS="-Werror -Wall"`.
- Kokkos builds: `-Wall -Werror -Wunused-result`, `CXXFLAGS` additionally `-std=c++20`.
- Default runtime options in the image: `PETSC_OPTIONS="-on_error_abort -fp_trap on"` (Kokkos image sets the aijkokkos/kokkos mat/vec types too).

Output-parser checks (part of `gnu_opt`)
- `python/run_parse_tests.py` with `tools/parse_pflare_output.py` runs inside the image (`PFLARE_TESTS_DIR`, `PFLARE_TOOLS_DIR` env vars) and asserts: grid complexity < 3.0, reuse storage == 0.0, KSP iteration counts below their maxima. Changes that alter PFLARE's printed stats/timing output can break these.

GPU tests (`.github/workflows/gpu_tests.yml`, manual trigger only)
- `runs-on: cirun-gpu-runner--<run id>`: Cirun (cirun.io GitHub App, free for public repos) reads `.cirun.yml`, rents a Vast.ai RTX 3060 container from `stevendargaville/petsc_kokkos_cuda:latest` (CI_builds `Dockerfile_petsc_kokkos_cuda`: PETSc main debug, Kokkos CUDA sm_86, CUDA-aware MPICH, no PFLARE), and destroys it when the job ends. Gated by the `protected-manual-builds` environment; no secrets are exposed to the job.
- Steps: record hardware (fails if the GPU's compute capability cannot run the image's arch), PETSc `make check` (deliberately not run in the image build), then PFLARE `make`, `make python`, `make check`, `make tests` with the Kokkos CI flags (`PFLARE_KOKKOS_DEBUG` not set for now). The `smoke_only` input runs only the hardware step.
- The image arch (`CUDA_ARCH` in CI_builds) and `instance_type` in `.cirun.yml` must be changed together. The image's ENV is not visible to the runner session on Vast.ai, so the workflow sets `PETSC_DIR`/`PETSC_ARCH`/`PATH` itself.
