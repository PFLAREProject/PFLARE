## New methods in PFLARE

PFLARE adds new methods to PETSc, described below. Note: for methods with GPU setup labelled "No" in the tables below, this indicates that some/all of the setup occurs on the CPU before being transferred to the GPU. The setup for these methods may therefore be slow. The solves however all occur on the GPU.    

### PCPFLAREINV - A new PETSc PC type

PCPFLAREINV contains methods for computing approximate inverses, most of which can be applied as assembled matrices or matrix-free. PCPFLAREINV can be used with the command line argument ``-pc_type pflareinv``, with several different PFLAREINV types available with ``-pc_pflareinv_type``:

   | Command line type  | Flag | Description | GPU setup |
   | ------------- | -- | ------------- | -- |
   | power  |  PFLAREINV_POWER  | GMRES polynomial, applied as a mononomial, with coefficients computed with a power basis  | Yes |
   | arnoldi  |  PFLAREINV_ARNOLDI  | GMRES polynomial, applied as a mononomial, with coefficients computed with an Arnoldi method  | Yes |
   | newton  |  PFLAREINV_NEWTON  | GMRES polynomial, applied as a Newton polynomial, with roots computed with an Arnoldi method and with extra roots added for stability  | Matrix-free: Yes Assembled: No |
   | newton_no_extra  |  PFLAREINV_NEWTON_NO_EXTRA  | GMRES polynomial, applied as a Newton polynomial, with roots computed with an Arnoldi method and with no extra roots added   | Matrix-free: Yes Assembled: No |
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
   | pmis  |  CF_PMIS  | PMIS method with symmetrised strength matrix | Yes |
   | pmis_dist2  |  CF_PMIS_DIST2  | Distance 2 PMIS method with strength matrix formed by S'S + S and then symmetrised | Partial |
   | agg  |  CF_AGG  | Aggregation method with root-nodes as C points. In parallel this is processor local aggregation  | No |
   | pmis_agg  |  CF_PMIS_AGG  | PMIS method with symmetrised strength matrix on boundary nodes, then processor local aggregation.  | Partial |

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