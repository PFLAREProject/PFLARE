## Installing PFLARE

PFLARE depends on MPI, BLAS, LAPACK and PETSc configured with a graph partitioner (e.g., ParMETIS). PFLARE uses the same compilers and flags defined in the PETSc configure.

If you wish to run PFLARE on GPUs you should configure PETSc with Kokkos and the relevant GPU backend. PFLARE has been tested with GNU, Intel, LLVM, NVIDIA and Cray compilers. 

Please choose one of the build methods below. If you wish to contribute to PFLARE we would recommend building from the source.

### PETSc configure build

PFLARE has now been added to the PETSc configure as an external package (currently on the PETSc `main` branch, which will be included in the PETSc 3.25 release). PFLARE can be built by adding ``--download-pflare`` to the PETSc configure. 

The PC types added by PFLARE can then be used as native PETSc PC types through command line arguments without any changes to existing code. The [Linking to PFLARE](#linking-to-pflare) and [Modifying existing code](#modifying-existing-code) sections below can therefore be skipped. 

If however you also wish to be able to set the PETSc types/options separate to the command line arguments, the PFLARE includes in [Modifying existing code](#modifying-existing-code) should be added to existing code, but the ``PCRegister_PFLARE()`` call in C/Fortran should not be added. The PETSc ``include`` and ``lib`` directories will contain the required files. 

Note: static builds with PFLARE are not available when building through the PETSc configure, please use one of the other build methods below.

### Spack package manager build

If you use Spack, an up to date PFLARE package is available at:
https://github.com/PFLAREProject/PFLARE_spack

### Source build

1) Clone this repository
2) Set `PETSC_DIR` and `PETSC_ARCH` environmental variables.
3) ``make`` in the top level directory.

If PETSc was configured with petsc4py, the PFLARE Python interfaces can be built with:

4) ``make python`` in the top level directory.

Then if desired, check that PFLARE was built successfully by running some simple tests with:

5) ``make check`` in the top level directory.

The full set of tests can be run with:

6) ``make tests`` in the top level directory.

Please use the `main` branch of PETSc and compile PETSc directly from the source code, as PFLARE requires access to some of the PETSc types only available in the source. If PETSc was installed out of place, you should add the `/include` directory from the PETSc source location to `CFLAGS, CXXFLAGS, CPPFLAGS` before calling `make` for PFLARE. 

If you wish to use an older version of PETSc, please see the [PFLARE Spack](https://github.com/PFLAREProject/PFLARE_spack) `package.py` where a list of compatible PFLARE release versions is maintained.

### Docker

An up to date Docker image is available on Dockerhub. To download the image and check the build:

     docker run -it stevendargaville/pflare
     make check

## Linking to PFLARE

For Fortran/C:

1) Link the library `libpflare` to your application; it is output to `lib/`
2) Add `include/` to your include path
3) For Fortran also add the location of the .mod files to your include path (compiler dependent, often `lib/`, `include/` or the top level directory)

For Python:

1) Ensure the full path to `python/` is in your `PYTHONPATH` environmental variable (along with the library path for PETSc). 
2) `LD_LIBRARY_PATH` must include the `lib/` directory (along with the library path for PETSc).

## Modifying existing code

After linking your application to the PFLARE library, using the components of PFLARE through PETSc is very simple. 

For Fortran/C, the user must call a single function which registers the new PC types with PETSc, while in Python this is handled by the import statement. For example, the only modifications required in an existing code are:

in Fortran:

     #include "finclude/pflare.h"
     ! ...
     call PCRegister_PFLARE()
     ! ...

or in C:

     #include "pflare.h"
     // ...
     PCRegister_PFLARE();
     // ...

or in Python with petsc4py:

     import pflare
     # ...
