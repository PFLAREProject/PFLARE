module air_data_type

   use tsqr
   use gmres_poly_data_type
   
   ! PETSc
   use petscmat
   use petscvec
   use petscis

#include "petsc/finclude/petscmat.h"
#include "petsc/finclude/petscvec.h"
#include "petsc/finclude/petscis.h"

   implicit none

   public

   type petsc_mat_pointer_array
      type(tMat), dimension(:), pointer :: array => null()
   end type petsc_mat_pointer_array
   type petsc_vec_array
      type(tVec), allocatable, dimension(:) :: array
   end type petsc_vec_array

   type int_vec_array
      integer, allocatable, dimension(:) :: array
   end type int_vec_array   

   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
   
   ! Stores the options needed for air multigrid
   ! These defaults are reasonable settings for time independent 
   ! advection equations (ie purely hyperbolic) on a 2D unstructured
   ! triangular mesh
   ! Can be changed via command line arguments
   ! These defaults need to match those in destroy_air_data
   type air_options

      ! Print out stats and timings
      ! These require some parallel reductions to compute
      ! so they are off by default
      ! -pc_air_print_stats_timings
      logical :: print_stats_timings = .FALSE.

      ! Maximum number of levels in the multigrid hierarchy
      ! -pc_air_max_levels
      integer :: max_levels = 300
      ! Minimum number of global unknowns on the coarse grid
      ! -pc_air_coarse_eq_limit
      PetscInt :: coarse_eq_limit = 6
      ! From this level onwards, build and then evaluate if the coarse grid solver
      ! is good enough and use that to determine if we should truncate on that level
      ! -pc_air_auto_truncate_start_level
      integer :: auto_truncate_start_level = -1
      ! What relative tolerance to use to determine if a coarse grid solver is good enough
      ! -pc_air_auto_truncate_tol
      PetscReal :: auto_truncate_tol = 1e-14

      ! Perform processor agglomeration throughout the hierarchy
      ! This reduces the number of active MPI ranks as we coarsen
      ! by a factor of processor_agglom_factor, whenever either there 
      ! are fewer unkowns per core (on average) than process_eq_limit or
      ! the local to non-local ratio of nnzs is < processor_agglom_ratio
      ! The entire hierarchy stays on comm_world however
      ! Only happens where necessary, not on every level
      ! -pc_air_processor_agglom
      logical :: processor_agglom = .TRUE.
      ! The local to nonlocal ratio of nnzs that is used to 
      ! trigger processor agglomeration on all level
      ! -pc_air_processor_agglom_ratio
      PetscReal :: processor_agglom_ratio = 2
      ! What factor to reduce the number of active MPI ranks by
      ! each time when doing processor agglomeration
      ! -pc_air_processor_agglom_ratio
      integer :: processor_agglom_factor = 2   
      ! If on average there are fewer than this number of equations per rank
      ! processor agglomeration will be triggered
      ! -pc_air_process_eq_limit
      PetscInt :: process_eq_limit = 50  
      ! If we are doing processor agglomeration, then we have 
      ! some ranks with no rows
      ! If computing a gmres polynomial inverse 
      ! with inverse_type arnoldi or newton, then we can have 
      ! the reductions occur on a subcomm if we want to reduce the cost
      ! -pc_air_subcomm
      logical :: subcomm = .FALSE.

      ! This is used in the CF splitting to define strong dependencies/influences
      ! -pc_air_strong_threshold
      PetscReal :: strong_threshold = 0.5
      ! How many passes of DDC to do
      ! -pc_air_ddc_its
      integer :: ddc_its = 1
      ! The max diagonal dominace ratio we want in ddc iterations
      ! -pc_air_max_dd_ratio
      PetscReal :: max_dd_ratio = 0.0
      ! Second pass in the PMISR DDC CF splitting converts 
      ! this fraction of local F points to C based on diagonal dominance
      ! -pc_air_ddc_fraction
      PetscReal :: ddc_fraction = 0.1
      ! What CF splitting algorithm to use
      ! 0 - PMISR DDC
      ! 1 - PMIS distance 1
      ! 2 - PMIS distance 2 - uses S'S + S 
      ! 3 - Root node aggregation
      ! -pc_air_cf_splitting_type
      integer :: cf_splitting_type = 0
      ! Maximum number of Luby steps to do in CF splitting
      ! Negative means do as many as needed (at the cost of a parallel
      ! reduction everytime we finish a Luby step)
      ! -pc_air_max_luby_steps
      integer :: max_luby_steps = -1

      ! Stores what smoothing happens
      ! Positive means F smooth, negative means C smooth
      ! The size of the integer is how many consecutive smooths
      ! e.g., (/ 2 -2 2 /) means 2 F, then 2 C, then 2 F
      ! 2 F smooths by default
      integer, dimension(10) :: smooth_order = (/ 2, 0, 0, 0, 0, 0, 0, 0, 0, 0 /)
      ! Do we do any C point smooths?
      logical :: any_c_smooths = .FALSE.
      ! Do we apply our polynomials matrix free when smoothing?
      ! -pc_air_matrix_free_polys
      logical :: matrix_free_polys = .FALSE.
      ! Do we use a one point injection classical prolongator or an AIR-style prolongator
      ! -pc_air_one_point_classical_prolong
      logical :: one_point_classical_prolong = .TRUE.
      ! Do we do full smoothing up and down rather than FF or FFC
      ! -pc_air_full_smoothing_up_and_down
      logical :: full_smoothing_up_and_down = .FALSE.
      ! Do we define our prolongator as R^T?
      ! -pc_air_symmetric
      logical :: symmetric = .FALSE.
      ! Do we compute a near-nullspace vector and apply it as a constaint 
      ! to the prolongator?
      ! -pc_air_constrain_w
      logical :: constrain_w = .FALSE.
      ! Do we compute a near-nullspace vector and apply it as a constaint 
      ! to the restrictor?
      ! -pc_air_constrain_z
      logical :: constrain_z = .FALSE.    
      ! How many iterations of improvement do we want to do to Z
      ! -pc_air_improve_z_its
      integer :: improve_z_its = 0
      ! How many iterations of improvement do we want to do to W  
      ! -pc_air_improve_w_its
      integer :: improve_w_its = 0   

      ! Strong R threshold to apply dropping prior to computing Z
      ! This only applies when computing Z, ie if you build a GMRES polynomial approximation
      ! to Aff^-1, it applies this dropping, then computes Z, then rebuilds 
      ! an Aff^-1 approximation without the dropping for smoothing
      ! -pc_air_strong_r_threshold
      PetscReal :: strong_r_threshold = 0d0

      ! What type of approximation do we use for Z?
      ! 0 - Aff^-1 approximation determined by inverse type (below) and then Z computed with matmatmult
      ! 1 - lAIR computes Z directly
      ! 2 - SAI version of lAIR computes Z directly
      ! -pc_air_z_type
      integer :: z_type = 0

      ! If z_type == 1 or 2, this is the distance the grid-transfer operators go out to
      ! This is so we can have lair out to some distance, and then a different sparsity 
      ! for our smoothers
      ! If z_type == 0 this is ignored, and the distance is determined by inverse_sparsity_order + 1
      ! -pc_air_lair_distance 
      integer :: lair_distance = 2      

      ! What type of approximation do we use for Aff^-1 
      ! This is used both for Z (if z_type == 0, see below) and for F smoothing
      ! These are defined by PCPFLAREINVType 
      ! 0 - GMRES polynomial with the power basis 
      ! 1 - GMRES polynomial with the arnoldi basis 
      ! 2 - GMRES polynomial with the newton basis - can only be used matrix-free atm      
      ! 3 - Neumann polynomial
      ! 4 - SAI
      ! 5 - Incomplete SAI (ie a restricted additive schwartz)
      ! 6 - Weighted Jacobi with weight 3 / ( 4 * || Dff^(-1/2) * Aff * Dff^(-1/2) ||_inf )
      ! 7 - Unweighted Jacobi
      ! -pc_air_inverse_type
      integer :: inverse_type = 1      

      ! This is the order of polynomial we use in air if inverse_type is 
      ! power, arnoldi, newton or neumann
      ! -pc_air_poly_order
      integer :: poly_order = 6
      ! This is the order of sparsity we use if we assemble our approximate inverses
      ! If z_type == 0 this also determines what distance our grid-transfer operators are
      ! distance = inverse_sparsity_order + 1
      ! -pc_air_inverse_sparsity_order
      integer :: inverse_sparsity_order = 1

      ! Inverse type for c smoothing
      ! This defaults to whatever the F point smoother is atm
      ! -pc_air_c_inverse_type
      integer :: c_inverse_type = 1      
      ! Poly order for c smoothing
      ! This defaults to whatever the F point smoother is atm
      ! -pc_air_c_poly_order
      integer :: c_poly_order = 6
      ! Inverse sparsity order for c smoothing
      ! This defaults to whatever the F point smoother is atm
      ! -pc_air_c_inverse_sparsity_order
      integer :: c_inverse_sparsity_order = 1      
      
      ! These are for the coarse grid solver
      ! -pc_air_coarsest_inverse_type
      integer :: coarsest_inverse_type = 1
      ! -pc_air_coarsest_poly_order
      integer :: coarsest_poly_order = 6
      ! -pc_air_coarsest_inverse_sparsity_order
      integer :: coarsest_inverse_sparsity_order = 1
      ! -pc_air_coarsest_matrix_free_polys
      logical :: coarsest_matrix_free_polys = .FALSE.
      ! -pc_air_coarsest_subcomm
      logical :: coarsest_subcomm = .FALSE.

      ! These are the relative drop tolerances (inf norm) on R and A after they are built
      ! -pc_air_r_drop
      PetscReal :: r_drop = 0.01
      ! -pc_air_a_drop
      PetscReal :: a_drop = 1e-4
      ! Whether to lump in A or drop
      ! -pc_air_a_lump
      logical :: a_lump = .FALSE.

      ! Whether or not to re-use the existing sparsity when PCSetup is called
      ! with SAME_NONZERO_PATTERN
      ! This involves re-using the CF splitting, the symbolic mat-mat mults, 
      ! the repartitioning, the structure of the matrices with drop tolerances applied, etc
      ! This will take more memory but 
      ! will make the setup much cheaper on subsequent calls. If the matrix has 
      ! changed entries convergence may suffer if the matrix is sufficiently different
      ! -pc_air_reuse_sparsity
      logical :: reuse_sparsity = .FALSE.

      ! Whether or not to also re-use the gmres polynomial coefficients when 
      ! reuse_sparsity is set to true
      ! If the matrix has been changed the reused coefficients won't be correct, 
      ! and the coefficients are very sensitive to changes in the matrix
      ! This is really only a useful option if you are regenerating 
      ! the hierarchy for the exact same matrix where you have stored 
      ! the gmres polynomial coefficients externally and restore them
      ! -pc_air_reuse_poly_coeffs
      logical :: reuse_poly_coeffs = .FALSE.

   end type air_options 
   
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
   
   ! Indices into reuse_mat
   integer, parameter :: MAT_AP = 1
   integer, parameter :: MAT_RAP = 2
   integer, parameter :: MAT_RAP_DROP = 3
   integer, parameter :: MAT_Z_DROP = 4
   integer, parameter :: MAT_W_DROP = 5
   integer, parameter :: MAT_COARSE_REPARTITIONED = 6
   integer, parameter :: MAT_P_REPARTITIONED = 7
   integer, parameter :: MAT_R_REPARTITIONED = 8
   integer, parameter :: MAT_AFF_DROP = 9
   integer, parameter :: MAT_ACF_DROP = 10
   integer, parameter :: MAT_AFC_DROP = 11
   integer, parameter :: MAT_A_DROP = 12
   integer, parameter :: MAT_W = 13
   integer, parameter :: MAT_Z = 14
   integer, parameter :: MAT_INV_AFF = 15
   integer, parameter :: MAT_INV_AFF_DROPPED = 16
   integer, parameter :: MAT_INV_ACC = 17
   integer, parameter :: MAT_SAI_SUB = 18
   integer, parameter :: MAT_Z_AFF = 19
   integer, parameter :: MAT_Z_NO_SPARSITY = 20
   integer, parameter :: MAT_W_AFF = 21
   integer, parameter :: MAT_W_NO_SPARSITY = 22

   ! Indices into reuse_is
   integer, parameter :: IS_REPARTITION = 1
   integer, parameter :: IS_R_Z_FINE_COLS = 2
   
   ! Stores temporary data we use for re-use
   ! The things stored in these structures 
   ! would normally be destroyed during the setup
   type air_reuse_data

      type(tMat), dimension(22) :: reuse_mat
      type(petsc_mat_pointer_array), dimension(22) :: reuse_submatrices
      type(tIS), dimension(2) :: reuse_is

   end type air_reuse_data
   
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
   
   ! Stores the data needed for air multigrid
   type air_multigrid_data

      ! Options for AIR multigrid - these are applied to all levels
      type(air_options) :: options

      ! Actual number of levels in the hierarchy after coarsening
      integer :: no_levels = -1   
      ! Do we have a gpu matrix or one on the cpu 
      logical :: fast_veciscopy_exists = .FALSE.
      ! Can we reuse a one point classical prolongator
      ! We only need this explicitly due to kokkos which is fussy
      ! about the exact same pointers being passed into spgemm_numeric
      ! (rather than just having the same sparsity) once we've 
      ! repartitioned
      logical :: reuse_one_point_classical_prolong = .FALSE.

      ! Indices of F and C points on each level
      type(tIS), allocatable, dimension(:)               :: IS_fine_index, IS_coarse_index

      ! Storage for ideal restrictors/prolongators that includes I block
      type(tMat), allocatable, dimension(:)              :: restrictors
      type(tMat), allocatable, dimension(:)              :: prolongators

      ! Restrictor style injectors we use as gpus don't currently support veciscopy
      type(tMat), allocatable, dimension(:)              :: i_fine_full, i_coarse_full
      type(tMat), allocatable, dimension(:)              :: i_fine_full_full, i_coarse_full_full

      ! Extracted matrices on each level
      type(tMat), allocatable, dimension(:)              :: A_ff, A_fc, A_cf, A_cc
      ! Approximate inverse of Aff
      type(tMat), allocatable, dimension(:)              :: inv_A_ff
      ! GMRES polynomial data for inv_A_ff
      type(gmres_poly_data), allocatable, dimension(:)   :: inv_A_ff_poly_data
      ! GMRES polynomial data for inv_A_ff that has had entries dropped
      type(gmres_poly_data), allocatable, dimension(:)   :: inv_A_ff_poly_data_dropped      
      ! Approximate inverse of Acc
      type(tMat), allocatable, dimension(:)              :: inv_A_cc
      ! GMRES polynomial data for inv_A_cc
      type(gmres_poly_data), allocatable, dimension(:)   :: inv_A_cc_poly_data      
      ! Boolean to tell whether we have allocated petsc matrices
      logical, allocatable, dimension(:)                 :: allocated_matrices_A_ff, allocated_matrices_A_cc 
      logical, allocatable, dimension(:)                 :: allocated_is

      ! Coarse matrices on each level
      type(tMat), allocatable, dimension(:)              :: coarse_matrix
      ! Boolean to tell whether we have allocated petsc matrices
      logical, allocatable, dimension(:)                 :: allocated_coarse_matrix

      ! The number of smooths we do on each level
      type(int_vec_array), allocatable, dimension(:)     :: smooth_order_levels

      ! The coarsest grid is stored in coarse_matrix(no_levels) and this is the 
      ! GMRES polynomial data for the coarsest grid solver
      ! The approx inverse for the coarse grid solve is stored in inv_A_ff(no_levels)
      type(gmres_poly_data) :: inv_coarsest_poly_data       

      ! Temporary storage
      type(petsc_vec_array), dimension(4) :: temp_vecs_fine, temp_vecs_coarse 
      type(petsc_vec_array), dimension(1) :: temp_vecs

      ! Temporary reuse
      type(air_reuse_data), allocatable, dimension(:) :: reuse

      ! ~~~~~~~~~~~~~~~~~~~~~~

      ! Storage for nnzs of each matrix on each level
      integer(kind=8), allocatable, dimension(:) :: coarse_matrix_nnzs, A_ff_nnzs, inv_A_ff_nnzs, inv_A_cc_nnzs
      integer(kind=8), allocatable, dimension(:) :: restrictor_nnzs, prolongator_nnzs, A_fc_nnzs, A_cf_nnzs, A_cc_nnzs

   end type air_multigrid_data  
   
   ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~       

   contains    

! -------------------------------------------------------------------------------------------------------------------------------
      
end module air_data_type
