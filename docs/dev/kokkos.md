# Kokkos development (agent reference)

Conventions
- A Fortran file `src/X.F90` (or its C port `src/X.c`) has its GPU/threaded sibling `src/Xk.kokkos.cxx` (trailing `k` on the stem; the `.kokkos.cxx` suffix triggers PETSc's Kokkos build rules). Existing pairs include `PMISR_Modulek`, `Grid_Transferk`, `Gmres_Polyk`, `Gmres_Poly_Newtonk`, `SAI_Zk`, `DDC_Modulek`, `MatDiagDomk`, `Device_Datak`, `PETSc_Helperk`, `VecISCopyLocalk`.
- Kernels are exported as `PETSC_INTERN void <snake_case>_kokkos(...)`. The C ports call them through a prototype header `src/Xk.h` that the kernel file includes too (`PETSc_Helperk.h`, `PMISR_Modulek.h`, `DDC_Modulek.h`, `MatDiagDomk.h`, `Device_Datak.h`, `Gmres_Polyk.h`, `Gmres_Poly_Newtonk.h`, `SAI_Zk.h`, `Grid_Transferk.h`, `VecISCopyLocalk.h`); the Fortran reference in `src/fref` (built only with `make PFLARE_PORT_DEBUG=1`) calls them through ISO-C interfaces in `src/fref/C_PETSc_Interfaces.F90` / `src/fref/C_Fortran_Bindings.F90`. Shared typedefs/helpers live in `include/kokkos_helper.hpp`.
- Build constraints: C++20; CI compiles with `-Wall -Werror -Wunused-result`.
- Kokkos has a hard 20MB limit on level-1 team scratch memory. Row-wise systems that can exceed it must fall back to a sparse/global-memory formulation (see `SAI_Zk.kokkos.cxx` and commit e49d5ec for the pattern).

Debug-compare mode (`PFLARE_KOKKOS_DEBUG=1`)
- Read once and cached by `kokkos_debug()` in `src/PETSc_Helper.c` (the Fortran reference reaches it through `port_interfaces`) — restart the process to change it.
- When set, code paths in `PETSc_Helper.c`, `CF_Splitting.c`, `PMISR_Module.c`, `DDC_Module.c`, `MatDiagDom.c`, `Gmres_Poly.c`, `Gmres_Poly_Newton.c`, `SAI_Z.c`, `Grid_Transfer.c`, `FC_Smooth.c`, `FC_Smooth_Block.c` run both the CPU and Kokkos implementations and compare results against the precision-dependent `PFLARE_TOL_MATFREE_*` tolerances in `src/pflare_parameters.h`. A mismatch aborts with an error. The tolerances are absolute, so the compare of an assembled high-order power basis polynomial can trip on the rounding in the large entries of a high power of the operator (`tests/port_compare_pc.c` scales its operator down for the reuse path for this reason).
- When adding a new Kokkos kernel, wire its CPU/Kokkos comparison into the calling routine under `if (kokkos_debug())`, following the existing patterns in those files.

Test sequence after any Kokkos change (each step must pass `make check` + `make tests_short`):
1. `export PETSC_OPTIONS="-mat_type aijkokkos -vec_type kokkos -dm_mat_type aijkokkos -dm_vec_type kokkos -on_error_abort"` → run tests
2. `export PFLARE_KOKKOS_DEBUG=1` (keeping the same PETSC_OPTIONS) → run tests
3. `unset PETSC_OPTIONS PFLARE_KOKKOS_DEBUG`
