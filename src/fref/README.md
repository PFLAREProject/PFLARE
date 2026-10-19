# Fortran reference implementations

While a module of the Fortran core is being ported to C, the retired Fortran
version is kept here so the C port can be compared against it in the same
process. Exported `bind(C)` names get an `_fref` suffix.

The files here are only compiled with `make PFLARE_PORT_DEBUG=1`, which also
defines `PFLARE_PORT_DEBUG` for the C, C++ and Fortran preprocessors and adds
the `tests/port_compare_pc` and `tests/port_compare_cf` drivers to the test
targets. See section 6 of `docs/dev/fortran_removal_plan.md`.

The directory is deleted once the port is complete.
