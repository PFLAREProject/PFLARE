# Kokkos development (agent reference)

Conventions
- A Fortran file `src/X.F90` has its GPU/threaded sibling `src/Xk.kokkos.cxx` (trailing `k` on the stem; the `.kokkos.cxx` suffix triggers PETSc's Kokkos build rules). Existing pairs include `PMISR_Modulek`, `Grid_Transferk`, `Gmres_Polyk`, `Gmres_Poly_Newtonk`, `SAI_Zk`, `DDC_Modulek`, `MatDiagDomk`, `Device_Datak`, `PETSc_Helperk`, `VecISCopyLocalk`.
- Kernels are exported as `PETSC_INTERN void <snake_case>_kokkos(...)` and called from Fortran through ISO-C interfaces in `src/C_PETSc_Interfaces.F90` / `src/C_Fortran_Bindings.F90`. Shared typedefs/helpers live in `include/kokkos_helper.hpp`.
- Build constraints: C++20; CI compiles with `-Wall -Werror -Wunused-result`.
- Kokkos has a hard 20MB limit on level-1 team scratch memory. Row-wise systems that can exceed it must fall back to a sparse/global-memory formulation (see `SAI_Zk.kokkos.cxx` and commit e49d5ec for the pattern).

Debug-compare mode (`PFLARE_KOKKOS_DEBUG=1`)
- Read once and cached by `kokkos_debug()` in `src/PETSc_Helper.F90` — restart the process to change it.
- When set, code paths in `CF_Splitting.F90`, `FC_Smooth.F90`, `DDC_Module.F90`, `Grid_Transfer.F90`, `Gmres_Poly_Newton.F90`, `SAI_Z.F90` run both the CPU and Kokkos implementations and compare results against the precision-dependent `PFLARE_TOL_MATFREE_*` tolerances in `src/Pflare_Parameters.F90`. A mismatch aborts with an error.
- When adding a new Kokkos kernel, wire its CPU/Kokkos comparison into the calling Fortran routine under `if (kokkos_debug())`, following the existing patterns in those files.

Test sequence after any Kokkos change (each step must pass `make check` + `make tests_short`):
1. `export PETSC_OPTIONS="-mat_type aijkokkos -vec_type kokkos -dm_mat_type aijkokkos -dm_vec_type kokkos -on_error_abort"` → run tests
2. `export PFLARE_KOKKOS_DEBUG=1` (keeping the same PETSC_OPTIONS) → run tests
3. `unset PETSC_OPTIONS PFLARE_KOKKOS_DEBUG`
