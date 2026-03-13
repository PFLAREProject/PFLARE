# CLAUDE.md

## Build

Build the library and all tests from the top repo directory:

```
make -j3 build_tests
```

## Testing

### Quick verification after a change

Run an appropriate short test from the `tests/Makefile` in the `tests/` directory, e.g.:

```
cd tests
make run_tests_no_load_short_serial
```

Then verify with the short test suite from the top repo directory:

```
make tests_short
```

### Kokkos verification

After passing the standard tests, verify Kokkos code with two additional passes:

1. Run the tests with Kokkos mat/vec types:

```
export PETSC_OPTIONS="-mat_type aijkokkos -vec_type kokkos -dm_mat_type aijkokkos -dm_vec_type kokkos -on_error_abort"
make tests_short
```

2. Run the tests with the Kokkos debug flag:

```
export PFLARE_KOKKOS_DEBUG=1
make tests_short
```

Remember to unset these environment variables after Kokkos testing.

## Code style

- Always fix all compile warnings.
- Never change existing whitespace.
