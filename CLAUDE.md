Explore
1. If you need the petsc source: `$PETSC_DIR/$PETSC_ARCH`

Build
1. In top repo directory: `make -j3 build_tests`

Tests
1. In top repo directory: `make check`
2. In top repo directory: `make tests_short`

If Kokkos code changed:
3. `export PETSC_OPTIONS="-mat_type aijkokkos -vec_type kokkos -dm_mat_type aijkokkos -dm_vec_type kokkos -on_error_abort"` → run Tests
4. `export PFLARE_KOKKOS_DEBUG=1` → run Tests
5. `unset PETSC_OPTIONS PFLARE_KOKKOS_DEBUG`

Rule: fix all compile warnings.
