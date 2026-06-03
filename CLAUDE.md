Explore
1. Check the values of env variables: $PETSC_DIR and $PETSC_ARCH
2. The PETSc source/headers/implementations are located at: `$PETSC_DIR/$PETSC_ARCH`
3. The Kokkos source/headers/implementations are located at: `$PETSC_DIR/$PETSC_ARCH/externalpackages/git.kokkos` and `$PETSC_DIR/$PETSC_ARCH/externalpackages/git.kokkos-kernels`

Build
1. In top repo directory: `make -j3 build_tests`
2. If Python code changed, in the top repo directory: `make python`
3. Rule: fix all compile warnings.

Tests
1. Run the test targets below once. Trust `make`'s exit code: 0 means all tests passed; any failure breaks the run with a non-zero code and prints the error to the terminal. Don't re-run to grep the output.
2. In top repo directory: `make check`
3. In top repo directory: `make tests_short`

If Kokkos code changed:
4. `export PETSC_OPTIONS="-mat_type aijkokkos -vec_type kokkos -dm_mat_type aijkokkos -dm_vec_type kokkos -on_error_abort"` → run Tests
5. `export PFLARE_KOKKOS_DEBUG=1` → run Tests
6. `unset PETSC_OPTIONS PFLARE_KOKKOS_DEBUG`