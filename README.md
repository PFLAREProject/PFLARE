[![CI](https://img.shields.io/github/actions/workflow/status/PFLAREProject/PFLARE/ci_build.yml?branch=main&label=CI)](https://github.com/PFLAREProject/PFLARE/actions/workflows/ci_build.yml)
[![spack](https://img.shields.io/github/actions/workflow/status/PFLAREProject/PFLARE_spack/ci_build.yml?branch=main&label=spack)](https://github.com/PFLAREProject/PFLARE_spack/actions/workflows/ci_build.yml)

<img align="right" img src="PFLARE_logo.png" width="300" height="300" />

# PFLARE library
### Author: Steven Dargaville

This library contains methods which can be used to solve linear systems in parallel with PETSc, with interfaces in C/Fortran/Python. 

It aims to provide fast & scalable iterative methods for asymmetric linear systems, in parallel and on both CPUs and GPUs.
   
Some examples of asymmetric linear systems that PFLARE can scalably solve include:   
   - Advection equations
   - Streaming operators from Boltzmann applications
   - Space-time discretisations
   - Heavily anisotropic Poisson/diffusion equations
   
without requiring Gauss-Seidel methods. This includes time dependent or independent equations, with structured or unstructured grids, with lower triangular structure or without.

## Methods available in PFLARE

PFLARE adds new methods to PETSc, including:
1) Polynomial approximate inverses, e.g., GMRES and Neumann polynomials
2) Reduction multigrids, e.g., AIRG, nAIR and lAIR
3) CF splittings, e.g., PMISR DDC

Details for each are given below, please also see references [1-8]. 

Note: for methods with GPU setup labelled "No" in the tables below, this indicates that some/all of the setup occurs on the CPU before being transferred to the GPU. The setup for these methods may therefore be slow. The solves however all occur on the GPU.   

### PCPFLAREINV - A new PETSc PC type

PCPFLAREINV contains methods for computing approximate inverses, most of which can be applied as assembled matrices or matrix-free. PCPFLAREINV can be used with the command line argument ``-pc_type pflareinv``, with several different PFLAREINV types available with ``-pc_pflareinv_type``:

   | Command line type  | Flag | Description | GPU setup |
   | ------------- | -- | ------------- | -- |
   | power  |  PFLAREINV_POWER  | GMRES polynomial with the power basis  | Yes |
   | arnoldi  |  PFLAREINV_ARNOLDI  | GMRES polynomial with the Arnoldi basis  | Yes |
   | newton  |  PFLAREINV_NEWTON  | GMRES polynomial with the Newton basis with extra roots for stability  | Yes |
   | newton_no_extra  |  PFLAREINV_NEWTON_NO_EXTRA  | GMRES polynomial with the Newton basis with no extra roots   | Yes |
   | neumann  |  PFLAREINV_NEUMANN  | Neumann polynomial  | Yes |
   | sai  |  PFLAREINV_SAI  | Sparse approximate inverse  | No |
   | isai  |  PFLAREINV_ISAI  | Incomplete sparse approximate inverse (equivalent to a one-level RAS)  | No |
   | wjacobi  |  PFLAREINV_WJACOBI  | Weighted Jacobi  | Partial |
   | jacobi  |  PFLAREINV_JACOBI  | Jacobi  | Yes |

### PCAIR - A new PETSc PC type

PCAIR contains different types of reduction multigrids. PCAIR can be used with the command line argument ``-pc_type air``. The combination of ``-pc_air_z_type`` and ``-pc_air_inverse_type`` (given by the PCPFLAREINV types above) defines several different reduction multigrids:

   | ``-pc_air_z_type``  | ``-pc_air_inverse_type`` | Description | GPU setup |
   | ------------- | -- | ------------- | --- |
   | product  |  power, arnoldi or newton  | AIRG  | Yes |
   | product  |  neumann  | nAIR with Neumann smoothing  | Yes |
   | product  |  sai  | SAI reduction multigrid  | No |
   | product  |  isai  | ISAI reduction multigrid  | No |
   | product  |  wjacobi or jacobi  | Distance 0 reduction multigrid  | Yes |
   | lair  |  wjacobi or jacobi  | lAIR  | No |
   | lair_sai  |  wjacobi or jacobi  | SAI version of lAIR  | No |

Different combinations of these types can also be used, e.g., ``-pc_air_z_type lair -pc_air_inverse_type power`` uses a lAIR grid transfer operator and GMRES polynomial smoothing with the power basis.

There are several features used to improve the parallel performance of PCAIR [2-3]:

   - The number of active MPI ranks on lower levels can be reduced where necessary. If this is used then:
     - Repartitioning with graph partitioners can be applied.
     - Calculation of polynomial coefficients can be done on subcommunicators.
   - The PCPFLAREINV methods above can be used as parallel coarse grid solvers, allowing heavy truncation of the multigrid hierarchy.
   - The multigrid hierarchy can be automatically truncated depending on the quality of the coarse grid solver
   - The sparsity of the multigrid hierarchy (and hence the CF splitting, repartitioning and symbolic matrix-matrix products) can be reused during setup.    

### CF splittings

The CF splittings in PFLARE are used within PCAIR to form the multigrid hierarchy. They can also be called independently from PCAIR. The CF splitting type within PCAIR can be specified with ``-pc_air_cf_splitting_type``: 

   | Command line type  | Flag | Description | GPU setup |
   | ------------- | -- | ------------- | -- |
   | pmisr_ddc  |  CF_PMISR_DDC  | Two-pass splitting giving diagonally dominant $\mathbf{A}_\textrm{ff}$ | Yes |
   | pmis  |  CF_PMIS  | PMIS method with symmetrized strength matrix | Yes |
   | pmis_dist2  |  CF_PMIS_DIST2  | Distance 2 PMIS method with strength matrix formed by S'S + S and then symmetrized | Partial |
   | agg  |  CF_AGG  | Aggregation method with root-nodes as C points. In parallel this is processor local aggregation  | No |
   | pmis_agg  |  CF_PMIS_AGG  | PMIS method with symmetrized strength matrix on boundary nodes, then processor local aggregation.  | Partial |

## Building PFLARE

PFLARE depends on MPI, BLAS, LAPACK and PETSc (>=3.24.0) configured with a graph partitioner (e.g., ParMETIS). PFLARE uses the same compilers and flags defined in the PETSc configure.

If you wish to run PFLARE on GPUs you should configure PETSc with Kokkos and the relevant GPU backend. PFLARE has been tested with GNU, Intel, LLVM, NVIDIA and Cray compilers. 

Please choose one of the build methods below. If you wish to contribute to PFLARE we would recommend building from the source.

### PETSc configure build

PFLARE has now been added to the PETSc configure as an external package (currently on the `main` branch, which will be included in the PETSc 3.25 release). PFLARE can be built by adding ``--download-pflare`` to the PETSc configure. 

The PC types added by PFLARE can then be used as native PETSc PC types through command line arguments without any changes to existing code. The [Linking to PFLARE](#linking-to-pflare) and [Modifying existing code](#modifying-existing-code) sections below can therefore be skipped. 

If however you also wish to be able to set the PETSc types/options separate to the command line arguments, the PFLARE includes in [Modifying existing code](#modifying-existing-code) should be added to existing code, but the ``PCRegister_PFLARE()`` call in C/Fortran should not be added. The PETSc ``include`` and ``lib`` directories will contain the required files. 

Note: static builds with PFLARE are not available when building through the PETSc configure, please use one of the other build methods below.

### Spack package manager build

If you use Spack, an up to date PFLARE package is available at:
https://github.com/PFLAREProject/PFLARE_spack

### Source build

1) Set `PETSC_DIR` and `PETSC_ARCH` environmental variables.
2) ``make`` in the top level directory.

If PETSc was configured with petsc4py, the PFLARE Python interfaces can be built with:

3) ``make python`` in the top level directory.

Then if desired, check that PFLARE was built successfully by running some simple tests with:

4) ``make check`` in the top level directory.

The full set of tests can be run with:

5) ``make tests`` in the top level directory.

Please use the `release` branch of PETSc and compile PETSc directly from the source code, as PFLARE requires access to some of the PETSc types only available in the source. If PETSc was installed out of place, you should add the `/include` directory from the PETSc source location to `CFLAGS, CXXFLAGS, CPPFLAGS` before calling `make` for PFLARE. 

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

## Using PFLARE

The PFLARE PC types can then be used like native PETSc PC types, either by writing code to set the PETSc type/options, or through command line arguments. A few examples include:

#### 1) Using PCAIR to apply AIRG with default options:

in Fortran:

     call KSPGetPC(ksp, pc, ierr)
     call PCSetType(pc, PCAIR, ierr)

     ! ...[e.g., KSPSolve somewhere here]

or in C:

     ierr = KSPGetPC(ksp, &pc);
     ierr = PCSetType(pc, PCAIR);

     // ...[e.g., KSPSolve somewhere here]

or in Python with petsc4py:

     pc = ksp.getPC()
     pc.setType("air")

     # ...[e.g., KSPSolve somewhere here]
     
or via the command line: ``-pc_type air``. 

#### 2) Using PCAIR to apply distance 2 lAIR with FCF Weighted Jacobi smoothing:

in Fortran:

     call KSPGetPC(ksp, pc, ierr)
     call PCSetType(pc, PCAIR, ierr)

     call PCAIRSetZType(pc, AIR_Z_LAIR, ierr)
     call PCAIRSetInverseType(pc, PFLAREINV_WJACOBI, ierr)
     call PCAIRSetSmoothType(pc, "fcf", ierr)

     ! ...[e.g., KSPSolve somewhere here]

or in C:

     ierr = KSPGetPC(ksp, &pc);
     ierr = PCSetType(pc, PCAIR);

     ierr = PCAIRSetZType(pc, AIR_Z_LAIR);
     ierr = PCAIRSetInverseType(pc, PFLAREINV_WJACOBI);
     ierr = PCAIRSetSmoothType(pc, "fcf");

     // ...[e.g., KSPSolve somewhere here]

or in Python with petsc4py:

     pc = ksp.getPC()
     pc.setType("air")

     petsc_options = PETSc.Options()
     petsc_options['pc_air_z_type'] = 'lair'
     petsc_options['pc_air_inverse_type'] = 'wjacobi'
     petsc_options['pc_air_smooth_type'] = 'fcf'     

     # ...[e.g., KSPSolve somewhere here]
     
or via the command line: ``-pc_type air -pc_air_z_type lair -pc_air_inverse_type wjacobi -pc_air_smooth_type fcf``.

#### 3) Using PCPFLAREINV to apply a 20th order GMRES polynomial in the Newton basis matrix free:

in Fortran:

     call KSPGetPC(ksp, pc, ierr)
     call PCSetType(pc, PCPFLAREINV, ierr)

     call PCPFLAREINVSetOrder(pc, 20, ierr)
     call PCPFLAREINVSetType(pc, PFLAREINV_NEWTON, ierr)
     call PCPFLAREINVSetMatrixFree(pc, PETSC_TRUE, ierr)

     ! ...[e.g., KSPSolve somewhere here]

or in C:

     ierr = KSPGetPC(ksp, &pc);
     ierr = PCSetType(pc, PCPFLAREINV);

     ierr = PCPFLAREINVSetOrder(pc, 20);
     ierr = PCPFLAREINVSetType(pc, PFLAREINV_NEWTON);
     ierr = PCPFLAREINVSetMatrixFree(pc, PETSC_TRUE);

     // ...[e.g., KSPSolve somewhere here]

or in Python with petsc4py:

     pc = ksp.getPC()
     pc.setType("pflareinv")

     petsc_options = PETSc.Options()
     petsc_options['pc_pflareinv_type'] = 'newton'
     petsc_options['pc_pflareinv_order'] = '20'
     petsc_options['pc_pflareinv_matrix_free'] = ''

     # ...[e.g., KSPSolve somewhere here]
     
or via the command line: ``-pc_type pflareinv -pc_pflareinv_type newton -pc_pflareinv_order 20 -pc_pflareinv_matrix_free``.

## GPU support           

If PETSc has been configured with GPU support then PCPFLAREINV and PCAIR support GPUs [3]. We recommend configuring PETSc with Kokkos and always specifying the matrix/vector types as Kokkos as this works across different GPU hardware (Nvidia, AMD, Intel). PFLARE also contains Kokkos routines to speed-up the setup/solve on GPUs. 

By default the tests run on the CPU unless the matrix/vector types are specified as those compatible with GPUs. For example, the following arguments specify that the 1D advection problem ``tests/adv_1d`` will use a 30th order GMRES polynomial applied matrix-free to solve on the CPU:

``./adv_1d -n 1000 -ksp_type richardson -pc_type pflareinv -pc_pflareinv_type arnoldi -pc_pflareinv_matrix_free -pc_pflareinv_order 30``

To run on GPUs, we set the matrix/vector types as Kokkos, which can be easily done through command line arguments. Our tests use either ``-mat_type`` and ``-vec_type``, or if set by a DM directly use ``-dm_mat_type`` and ``-dm_vec_type``.

For example, running the same problem on a single GPU with KOKKOS:

``./adv_1d -n 1000 -ksp_type richardson -pc_type pflareinv -pc_pflareinv_type arnoldi -pc_pflareinv_matrix_free -pc_pflareinv_order 30 -mat_type aijkokkos -vec_type kokkos``

Note: many of the tests allow the option ``-second_solve`` which turns on two solves, the first to trigger any copies to the GPU (e.g., the top grid matrix if created on the host) and the second to allow accurate timing. 

Development of the setup on GPUs is ongoing, please get in touch if you would like to contribute. The main areas requiring development are:

1) Processor agglomeration - GPU libraries exist which could replace the CPU-based calls to the PETSc graph partitioners
2) GPU optimisation - There are several Kokkos routines in PFLARE which would benefit from further optimisation

### Performance notes

1 - Typically we find good performance using as many DOFs per GPU as possible. 

2 - The processor agglomeration happens through the graph partitioners in PETSc and currently there is no GPU partitioner, hence this could be slow. The default parameters used in the processor agglomeration in PCAIR (e.g., ``-pc_air_process_eq_limit``) have also not been optimised for GPUs. You may wish to disable the processor agglomeration in parallel on GPUs (``-pc_air_processor_agglom 0``). Using heavy truncation may also help mitigate the the impact of turning off processor agglomeration on GPUs, see below.

3 - Multigrid methods on GPUs will often pin the coarse grids to the CPU, as GPUs are not very fast at the small solves that occur on coarse grids. We do not do this in PCAIR; instead we use the same approach we used in [2] to improve parallel scaling on CPUs. 

This is based around using the high-order polynomials applied matrix free as a coarse solver. For many problems GMRES polynomials in the Newton basis are stable at high order and can therefore be combined with heavy truncation of the multigrid hierarchy. We also have an automated way to determine at what level of the multigrid hierarchy to truncate. 

For example, on a single GPU with a 2D structured grid advection problem we apply a high order (10th order) Newton polynomial matrix-free as a coarse grid solver:

``./adv_diff_2d -da_grid_x 1000 -da_grid_y 1000 -ksp_type richardson -pc_type air -pc_air_coarsest_inverse_type newton -pc_air_coarsest_matrix_free_polys -pc_air_coarsest_poly_order 10 -dm_mat_type aijkokkos -dm_vec_type kokkos``

The hierarchy in this case has 29 levels. If we turn on the auto truncation and set a very large truncation tolerance  

``./adv_diff_2d -da_grid_x 1000 -da_grid_y 1000 -ksp_type richardson -pc_type air -pc_air_coarsest_inverse_type newton -pc_air_coarsest_matrix_free_polys -pc_air_coarsest_poly_order 10 -dm_mat_type aijkokkos -dm_vec_type kokkos -pc_air_auto_truncate_start_level 1 -pc_air_auto_truncate_tol 1e-1``

we find that the 10th order polynomials are good enough coarse solvers to enable truncation of the hierarchy at level 11. This gives the same iteration count as without truncation and we see an overall speedup of ~1.47x in the solve in this example. The speedup is typically greater in parallel. Please see [3] for more details.

## OpenMP support

If PETSc has been configured with Kokkos using OpenMP as the backend then PCPFLAREINV and PCAIR support OpenMP. To enable OpenMP throughout the setup/solve the matrix/vector types must be specified as Kokkos (see above) and the ``OMP_NUM_THREADS`` environmental variable must be set. Good performance is dependent on appropriate pinning of MPI ranks and OpenMP threads to CPU cores/NUMA regions.

## Reuse with PCAIR:

If solving multiple linear systems, there are several ways to reuse components of the PCAIR preconditioner to reduce setup times:

#### 1) Setup once 

In PETSc the ``KSPSetReusePreconditioner`` flag can be set to ensure the preconditioner is only setup once, even if the entries or sparsity of the underlying matrix in the KSP is changed. This can be useful in time stepping problems where the linear system (or Jacobian for non-linear problems) does not change, or if the changes are small. 

#### 2) Reuse sparsity during setup

When solving a linear system where the matrix has the same sparsity pattern as in a previous solve, PCAIR can can reuse the CF splitting, repartitioning, symbolic matrix-matrix products and resulting sparsity throughout the hierarchy during the setup. This takes more memory (typically 2-5x the storage complexity) but significantly reduces the time required in subsequent setups (10-20x). If we have a PETSc matrix $\mathbf{A}$:

in Fortran:

     call KSPGetPC(ksp, pc, ierr)
     call PCSetType(pc, PCAIR, ierr)
     
     ! Before the first setup, we tell it to 
     ! store any data we need for reuse
     call PCAIRSetReuseSparsity(pc, PETSC_TRUE, ierr)

     ! First solve - the PCAIR will be setup
     call KSPSolve(ksp, b, x, ierr)
     
     ! ...[Modify entries in A but keep the same sparsity]

     ! Second solve - the PCAIR will be setup
     ! but reusing the same sparsity
     call KSPSolve(ksp, b, x, ierr)     

or in C:

     ierr = KSPGetPC(ksp, &pc);
     ierr = PCSetType(pc, PCAIR);

     // Before the first setup, we tell it to 
     // store any data we need for reuse
     ierr = PCAIRSetReuseSparsity(pc, PETSC_TRUE);

     // First solve - the PCAIR will be setup
     ierr = KSPSolve(ksp, b, x)
     
     // ...[Modify entries in A but keep the same sparsity]

     // Second solve - the PCAIR will be setup 
     // but reusing the same sparsity
     ierr = KSPSolve(ksp, b, x)

or in Python with petsc4py:

     pc = ksp.getPC()
     pc.setType("air")

     petsc_options = PETSc.Options()
     petsc_options['pc_air_reuse_sparsity'] = ''

     # First solve - the PCAIR will be setup
     ksp.solve(b,x)
     
     # ...[Modify entries in A but keep the same sparsity]

     # Second solve - the PCAIR will be setup 
     # but reusing the same sparsity
     ksp.solve(b,x)
     
or via the command line: ``-pc_type air -pc_air_reuse_sparsity``.

#### 3) Reuse sparsity and GMRES polynomial coefficients during setup with AIRG

When using AIRG in PCAIR (which is the default), in addition to reusing the sparsity in the setup, the GMRES polynomial coefficients can also be reused. This is useful if the matrix is very similar to that used in a previous solve and saves parallel reductions in the setup. Note that the GMRES polynomial coefficients are very sensitive to changes in the matrix, so this may not always give good results. To enable this we simply set another reuse flag before (or after) the first solve:

in Fortran:

     ! Before the first setup, we tell it to 
     ! store any data we need for reuse
     call PCAIRSetReuseSparsity(pc, PETSC_TRUE, ierr)

     ! Tell it to store the gmres polynomial coefficients
     call PCAIRSetReusePolyCoeffs(pc, PETSC_TRUE, ierr)        
    
     ! First solve - the PCAIR will be setup
     call KSPSolve(ksp, b, x, ierr)  
     
     ! ...[Modify entries in A but keep the same sparsity]

     ! Second solve - the PCAIR will be setup
     ! but reusing sparsity & polynomial coefficients
     call KSPSolve(ksp, b, x, ierr)     

or in C:

     // Before the first setup, we tell it to 
     // store any data we need for reuse
     ierr = PCAIRSetReuseSparsity(pc, PETSC_TRUE);

     ! Tell it to store the gmres polynomial coefficients
     ierr = PCAIRSetReusePolyCoeffs(pc, PETSC_TRUE);

     // First solve - the PCAIR will be setup
     ierr = KSPSolve(ksp, b, x)
     
     // ...[Modify entries in A but keep the same sparsity]

     // Second solve - the PCAIR will be setup
     // but reusing sparsity & polynomial coefficients
     ierr = KSPSolve(ksp, b, x)

or in Python with petsc4py:

     pc = ksp.getPC()
     pc.setType("air")

     petsc_options = PETSc.Options()
     petsc_options['pc_air_reuse_sparsity'] = ''
     petsc_options['pc_air_reuse_poly_coeffs'] = ''

     # First solve - the PCAIR will be setup
     ksp.solve(b,x)
     
     # ...[Modify entries in A but keep the same sparsity]

     # Second solve - the PCAIR will be setup 
     # but reusing sparsity & polynomial coefficients
     ksp.solve(b,x)

or via the command line: ``-pc_type air -pc_air_reuse_sparsity -pc_air_reuse_poly_coeffs``.

#### 4) Reuse when solving the same linear system with AIRG

Often an outer loop (e.g., eigenvalue) will require solving a series of different linear systems one after another, and then returning to the first linear system to start the loop again. If the linear systems are close enough to each other to get good performance from reusing the sparsity, but different enough from each other that reusing the GMRES polynomials coefficients from a different linear system gives poor performance, PCAIR allows the GMRES polynomial coefficients to be saved externally and then restored before a solve. 

This means we can cheaply generate the exact same preconditioner when periodically solving the same linear system. An example of this is given in ``tests/ex6f_getcoeffs.F90``, where we solve a linear system, store the resulting GMRES polynomial coefficients, reuse the sparsity to solve a different linear system, then reuse the sparsity and restore the GMRES polynomial coefficients to solve the first linear system again. We should note this type of reuse is not yet available in Python. 

## CF splittings

The CF splittings can be called separately to PCAIR and are returned in two PETSc IS's representing the coarse and fine points. For example, to compute a PMISR DDC CF splitting of a PETSc matrix $\mathbf{A}$:

in Fortran:

     IS :: is_fine, is_coarse
     ! Threshold for a strong connection
     PetscReal :: strong_threshold = 0.5
     ! Second pass cleanup - one iteration
     int :: ddc_its = 1
     ! Fraction of F points to convert to C per ddc it
     PetscReal :: ddc_fraction = 0.1
     ! If not 0, keep doing ddc its until this diagonal dominance
     ! ratio is hit
     PetscReal :: max_dd_ratio = 0.0
     ! As many steps as needed
     int :: max_luby_steps = -1
     ! PMISR DDC
     integer :: algorithm = CF_PMISR_DDC
     ! Is the matrix symmetric?
     logical :: symmetric = .FALSE.
     
     call compute_cf_splitting(A, &
           symmetric, &
           strong_threshold, max_luby_steps, &
           algorithm, &
           ddc_its, &
           ddc_fraction, &
           max_dd_ratio, &
           is_fine, is_coarse) 

or in C (please note the slightly modified name in C):

     IS is_fine, is_coarse;
     // Threshold for a strong connection
     PetscReal strong_threshold = 0.5;
     // Second pass cleanup - one iteration
     int ddc_its = 1;
     // Fraction of F points to convert to C per ddc it
     PetscReal ddc_fraction = 0.1;
     // If not 0, keep doing ddc its until this diagonal dominance
     // ratio is hit
     PetscReal max_dd_ratio = 0.0;
     // As many steps as needed
     int max_luby_steps = -1;
     // PMISR DDC
     int algorithm = CF_PMISR_DDC;
     // Is the matrix symmetric?
     int symmetric = 0;

     compute_cf_splitting_c(&A, \
         symmetric, \
         strong_threshold, max_luby_steps, \
         algorithm, \
         ddc_its, \
         ddc_fraction, \
         max_dd_ratio, \
         &is_fine, &is_coarse);

or in Python with petsc4py:

     # Threshold for a strong connection
     strong_threshold = 0.5
     # Second pass cleanup - one iteration
     ddc_its = 1
     # Fraction of F points to convert to C per ddc it
     ddc_fraction = 0.1
     # If not 0, keep doing ddc its until this diagonal dominance
     # ratio is hit
     max_dd_ratio = 0.0     
     # As many steps as needed
     max_luby_steps = -1
     # PMISR DDC
     algorithm = pflare.CF_PMISR_DDC
     # Is the matrix symmetric?
     symmetric = False

     [is_fine, is_coarse] = pflare.pflare_defs.compute_cf_splitting(A, \
           symmetric, \
           strong_threshold, max_luby_steps, \
           algorithm, \
           ddc_its, \
           ddc_fraction, \
           max_dd_ratio)

## Options         

A brief description of the available options in PFLARE are given below and their default values.

### PCPFLAREINV

   | Command line  | Routine | Description | Default |
   | ------------- | -- | ------------- | --- |
   | ``-pc_pflareinv_type``  |  PCPFLAREINVGetType  PCPFLAREINVSetType  | The inverse type, given above | arnoldi |   
   | ``-pc_pflareinv_order``  |  PCPFLAREINVGetOrder  PCPFLAREINVSetOrder  | If using a polynomial inverse type, this determines the order of the polynomial | 6 |
   | ``-pc_pflareinv_sparsity_order``  |  PCPFLAREINVGetSparsityOrder  PCPFLAREINVSetSparsityOrder  | This power of A is used as the sparsity in assembled inverses | 1 |   
   | ``-pc_pflareinv_matrix_free``  |  PCPFLAREINVGetMatrixFree  PCPFLAREINVSetMatrixFree  | Is the inverse applied matrix free, or is an assembled matrix built and used | false |                                        

### PCAIR

#### Hierarchy options

   | Command line  | Routine | Description | Default |
   | ------------- | -- | ------------- | --- |
   | ``-pc_air_print_stats_timings``  |  PCAIRGetPrintStatsTimings  PCAIRSetPrintStatsTimings  | Print out statistics about the multigrid hierarchy and timings | false |       
   | ``-pc_air_max_levels``  |  PCAIRGetMaxLevels  PCAIRSetMaxLevels  | Maximum number of levels in the hierarchy | 300 |
   | ``-pc_air_coarse_eq_limit``  |  PCAIRGetCoarseEqLimit  PCAIRSetCoarseEqLimit  | Minimum number of global unknowns on the coarse grid | 6 |   
   | ``-pc_air_auto_truncate_start_level``  |  PCAIRGetAutoTruncateStartLevel  PCAIRSetAutoTruncateStartLevel  | Build a coarse solver on each level from this one and use it to determine if we can truncate the hierarchy | -1 |
   | ``-pc_air_auto_truncate_tol``  |  PCAIRGetAutoTruncateTol  PCAIRSetAutoTruncateTol  | Tolerance used to determine if the coarse solver is good enough to truncate at a given level | 1e-14 | 
   | ``-pc_air_r_drop``  |  PCAIRGetRDrop  PCAIRSetRDrop  | Drop tolerance applied to R on each level after it is built | 0.01 |
   | ``-pc_air_a_drop``  |  PCAIRGetADrop  PCAIRSetADrop  | Drop tolerance applied to the coarse matrix on each level after it is built | 0.0001 |
   | ``-pc_air_a_lump``  |  PCAIRSetALump  PCAIRSetALump  | Lump to the diagonal rather than drop for the coarse matrix | false |         

#### Parallel options

   | Command line  | Routine | Description | Default |
   | ------------- | -- | ------------- | --- |
   | ``-pc_air_processor_agglom``  |  PCAIRGetProcessorAgglom  PCAIRSetProcessorAgglom  | Whether to use a graph partitioner to repartition the coarse grids and reduce the number of active MPI ranks  | true |
   | ``-pc_air_processor_agglom_ratio``  |  PCAIRGetProcessorAgglomRatio  PCAIRSetProcessorAgglomRatio  | The local to non-local nnzs ratio that is used to trigger processor agglomeration on all levels  | 2.0 | 
   | ``-pc_air_processor_agglom_factor``  |  PCAIRGetProcessorAgglomFactor  PCAIRSetProcessorAgglomFactor  | What factor to reduce the number of active MPI ranks by each time when doing processor agglomeration  | 2 | 
   | ``-pc_air_process_eq_limit``  |  PCAIRGetProcessEqLimit  PCAIRSetProcessEqLimit  | If on average there are fewer than this number of equations per rank processor agglomeration will be triggered  | 50 |    
   | ``-pc_air_subcomm``  |  PCAIRGetSubcomm  PCAIRSetSubcomm  | If computing a polynomial inverse with type arnoldi or newton and we have performed processor agglomeration, we can exclude the MPI ranks with no non-zeros from reductions in parallel by moving onto a subcommunicator | false |  

#### CF splitting options

   | Command line  | Routine | Description | Default |
   | ------------- | -- | ------------- | --- |
   | ``-pc_air_cf_splitting_type``  |  PCAIRGetCFSplittingType  PCAIRSetCFSplittingType  | The type of CF splitting to use, given above | pmisr_ddc |    
   | ``-pc_air_strong_threshold``  |  PCAIRGetStrongThreshold  PCAIRSetStrongThreshold  | The strong threshold to use in the CF splitting | 0.5 |
   | ``-pc_air_max_luby_steps``  |  PCAIRGetMaxLubySteps  PCAIRSetMaxLubySteps  | If using CF splitting type pmisr_ddc, pmis, or pmis_dist2, this is the maximum number of Luby steps to use. If negative, use as many steps as necessary | -1 |   
   | ``-pc_air_ddc_its``  |  PCAIRGetDDCIts  PCAIRSetDDCIts  | If using CF splitting type pmisr_ddc, this is the number of iterations of DDC performed | 1 |   
   | ``-pc_air_ddc_fraction``  |  PCAIRGetDDCFraction  PCAIRSetDDCFraction  | If using CF splitting type pmisr_ddc, this is the local fraction of F points to convert to C points based on diagonal dominance. If negative, any row which has a diagonal dominance ratio less than the absolute value will be converted from F to C | 0.1 |
   | ``-pc_air_max_dd_ratio``  |  PCAIRGetMaxDDRatio  PCAIRSetMaxDDRatio  | If using CF splitting type pmisr_ddc, do as many DDC iterations as necessary to hit this diagonal dominance ratio. If 0.0 do the number in -pc_air_ddc_its | 0.0 |   

#### Approximate inverse options

   | Command line  | Routine | Description | Default |
   | ------------- | -- | ------------- | --- |
   | ``-pc_air_inverse_type``  |  PCAIRGetInverseType  PCAIRSetInverseType  | The inverse type, given above | arnoldi |
   | ``-pc_air_poly_order``  |  PCAIRGetPolyOrder  PCAIRSetPolyOrder  | If using a polynomial inverse type, this determines the order of the polynomial | 6 |
   | ``-pc_air_inverse_sparsity_order``  |  PCAIRGetInverseSparsityOrder  PCAIRSetInverseSparsityOrder  | This power of A is used as the sparsity in assembled inverses | 1 |        
   | ``-pc_air_matrix_free_polys``  |  PCAIRGetMatrixFreePolys  PCAIRSetMatrixFreePolys  | Do smoothing matrix-free if possible | false |   
   | ``-pc_air_smooth_type``  |  PCAIRGetSmoothType  PCAIRSetSmoothType  | Type and number of smooths | "ff" |
   | ``-pc_air_full_smoothing_up_and_down``  |  PCAIRGetFullSmoothingUpAndDown  PCAIRSetFullSmoothingUpAndDown  | Up and down smoothing on all points at once, rather than only down F and C smoothing which is the default  | false |     
   | ``-pc_air_c_inverse_type``  |  PCAIRGetCInverseType  PCAIRSetCInverseType  | The inverse type for the C smooth, given above. If unset this defaults to the same as the F point smoother | -pc_air_inverse_type |
   | ``-pc_air_c_poly_order``  |  PCAIRGetCPolyOrder  PCAIRSetCPolyOrder  | If using a polynomial inverse type, this determines the order of the polynomial for the C smooth. If unset this defaults to the same as the F point smoother | -pc_air_poly_order |
   | ``-pc_air_c_inverse_sparsity_order``  |  PCAIRGetCInverseSparsityOrder  PCAIRSetCInverseSparsityOrder  | This power of A is used as the sparsity in assembled inverses for the C smooth. If unset this defaults to the same as the F point smoother | -pc_air_inverse_sparsity_order |    
   

#### Grid transfer options

   | Command line  | Routine | Description | Default |
   | ------------- | -- | ------------- | --- |
   | ``-pc_air_one_point_classical_prolong``  |  PCAIRGetOnePointClassicalProlong  PCAIRSetOnePointClassicalProlong  | Use a one-point classical prolongator, instead of an approximate ideal prolongator | true |   
   | ``-pc_air_symmetric``  |  PCAIRGetSymmetric  PCAIRSetSymmetric  | Do we define our prolongator as R^T?  | false |     
   | ``-pc_air_strong_r_threshold``  |  PCAIRGetStrongRThreshold  PCAIRSetStrongRThreshold  | Threshold to drop when forming the grid-transfer operators  | 0.0 |
| ``-pc_air_z_type``  |  PCAIRGetZType  PCAIRSetZType  | Type of grid-transfer operator, see above  | product |
| ``-pc_air_lair_distance``  |  PCAIRGetLairDistance  PCAIRSetLairDistance  | If Z type is lair or lair_sai, this defines the distance of the grid-transfer operators  | 2 |          
   | ``-pc_air_constrain_w``  |  PCAIRGetConstrainW  PCAIRSetConstrainW  | Apply constraints to the prolongator. If enabled, by default it will smooth the constant vector and force the prolongator to interpolate it exactly. Can use MatSetNearNullSpace to give other vectors   | false |
   | ``-pc_air_constrain_z``  |  PCAIRGetConstrainZ  PCAIRSetConstrainZ  | Apply constraints to the restrictor. If enabled, by default it will smooth the constant vector and force the restrictor to restrict it exactly. Can use MatSetNearNullSpace to give other vectors   | false |
   | ``-pc_air_improve_w_its``  |  PCAIRGetImproveWIts  PCAIRSetImproveWIts  | Apply a number of Richardson iterations to improve the approximate prolongator. Uses the existing W as an initial guess.   | 0 |    
   | ``-pc_air_improve_z_its``  |  PCAIRGetImproveZIts  PCAIRSetImproveZIts  | Apply a number of Richardson iterations to improve the approximate restrictor. Uses the existing Z as an initial guess.    | 0 |  

#### Coarse grid solver options

   | Command line  | Routine | Description | Default |
   | ------------- | -- | ------------- | --- |
   | ``-pc_air_coarsest_inverse_type``  |  PCAIRGetCoarsestInverseType  PCAIRSetCoarsestInverseType  | Coarse grid inverse type, given above | arnoldi |
   | ``-pc_air_coarsest_poly_order``  |  PCAIRGetCoarsestPolyOrder  PCAIRSetCoarsestPolyOrder  | Coarse grid polynomial order | 6 |
   | ``-pc_air_coarsest_inverse_sparsity_order``  |  PCAIRGetCoarsestInverseSparsityOrder  PCAIRSetCoarsestInverseSparsityOrder  | Coarse grid sparsity order | 1 |
   | ``-pc_air_coarsest_matrix_free_polys``  |  PCAIRGetCoarsestMatrixFreePolys  PCAIRSetCoarsestMatrixFreePolys  | Do smoothing matrix-free if possible on the coarse grid | false |              
   | ``-pc_air_coarsest_subcomm``  |  PCAIRGetCoarsestSubcomm  PCAIRSetCoarsestSubcomm  | Use a subcommunicator on the coarse grid | false |

#### Reuse options

   | Command line  | Routine | Description | Default |
   | ------------- | -- | ------------- | --- |
   | ``-pc_air_reuse_sparsity``  |  PCAIRGetReuseSparsity  PCAIRSetReuseSparsity  | Store temporary data to allow fast setup with reuse | false |
   | ``-pc_air_reuse_poly_coeffs``  |  PCAIRGetReusePolyCoeffs  PCAIRSetReusePolyCoeffs  | Don't recompute the polynomial inverse coefficients during setup with reuse | false |         
     
## More examples

For more ways to use the library please see the Fortran/C examples and the Makefile in `tests/`, along with the Python examples in `python/`.

## References \& citing

Please see the references below for more details. If you use PFLARE in your work, please consider citing [1-3].

1. S. Dargaville, R. P. Smedley-Stevenson, P. N. Smith, C. C. Pain, AIR multigrid with GMRES polynomials (AIRG) and additive preconditioners for Boltzmann transport, Journal of Computational Physics 518 (2024) 113342  
2. S. Dargaville, R. P. Smedley-Stevenson, P. N. Smith, C. C. Pain, Coarsening and parallelism with reduction multigrids for hyperbolic Boltzmann transport, The International Journal of High Performance Computing Applications 39(3) (2025) 364-384  
3. S. Dargaville, R. P. Smedley-Stevenson, P. N. Smith, C. C. Pain, Solving advection equations with reduction multigrids on GPUs (2025) http://arxiv.org/abs/2508.17517  
4. T. A. Manteuffel, S. Münzenmaier, J. Ruge, B. Southworth, Nonsymmetric Reduction-Based Algebraic Multigrid, SIAM Journal on Scientific Computing 41 (2019) S242–S268  
5. T. A. Manteuffel, J. Ruge, B. S. Southworth, Nonsymmetric algebraic multigrid based on local approximate ideal restriction (lAIR), SIAM Journal on Scientific Computing 40 (2018) A4105–A4130   
6. A. Ali, J. J. Brannick, K. Kahl, O. A. Krzysik, J. B. Schroder, B. S. Southworth, Constrained local approximate ideal restriction for advection-diffusion problems, SIAM Journal on Scientific Computing (2024) S96–S122  
7. T. Zaman, N. Nytko, A. Taghibakhshi, S. MacLachlan, L. Olson, M. West, Generalizing reduction-based algebraic multigrid, Numerical Linear
Algebra with Applications 31(3) (2024) e2543   
8. Loe, J.A., Morgan, R.B, Toward efficient polynomial preconditioning for GMRES. Numerical Linear Algebra with Applications 29 (2022) e2427 
