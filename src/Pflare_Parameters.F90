module pflare_parameters

   ! Centralised integer/enum parameters for PFLARE.
   ! Collecting them here avoids downstream files having to load
   ! heavy computational modules just to get constants.

   use petscsys

#include "petsc/finclude/petscsys.h"

   implicit none
   public

   ! --------------------------------------------------------
   ! Approximate inverse types (PCPFLAREINVType)
   ! --------------------------------------------------------
   PetscEnum, parameter :: PFLAREINV_POWER             = 0
   PetscEnum, parameter :: PFLAREINV_ARNOLDI           = 1
   PetscEnum, parameter :: PFLAREINV_NEWTON            = 2
   PetscEnum, parameter :: PFLAREINV_NEWTON_NO_EXTRA   = 3
   PetscEnum, parameter :: PFLAREINV_NEUMANN           = 4
   PetscEnum, parameter :: PFLAREINV_SAI               = 5
   PetscEnum, parameter :: PFLAREINV_ISAI              = 6
   PetscEnum, parameter :: PFLAREINV_WJACOBI           = 7
   PetscEnum, parameter :: PFLAREINV_JACOBI            = 8

   ! --------------------------------------------------------
   ! Z / restrictor construction types (PCAIRZType)
   ! --------------------------------------------------------
   PetscEnum, parameter :: AIR_Z_PRODUCT   = 0
   PetscEnum, parameter :: AIR_Z_LAIR      = 1
   PetscEnum, parameter :: AIR_Z_LAIR_SAI  = 2

   ! --------------------------------------------------------
   ! CF-splitting point markers
   ! --------------------------------------------------------
   integer, parameter :: C_POINT =  1
   integer, parameter :: F_POINT = -1

   ! --------------------------------------------------------
   ! Indices into air_reuse_data%reuse_mat(:)   (from air_data_type)
   ! --------------------------------------------------------
   integer, parameter :: MAT_AP                   = 1
   integer, parameter :: MAT_RAP                  = 2
   integer, parameter :: MAT_RAP_DROP             = 3
   integer, parameter :: MAT_Z_DROP               = 4
   integer, parameter :: MAT_W_DROP               = 5
   integer, parameter :: MAT_COARSE_REPARTITIONED = 6
   integer, parameter :: MAT_P_REPARTITIONED      = 7
   integer, parameter :: MAT_R_REPARTITIONED      = 8
   integer, parameter :: MAT_AFF_DROP             = 9
   integer, parameter :: MAT_ACF_DROP             = 10
   integer, parameter :: MAT_AFC_DROP             = 11
   integer, parameter :: MAT_A_DROP               = 12
   integer, parameter :: MAT_W                    = 13
   integer, parameter :: MAT_Z                    = 14
   integer, parameter :: MAT_INV_AFF              = 15
   integer, parameter :: MAT_INV_AFF_DROPPED      = 16
   integer, parameter :: MAT_INV_ACC              = 17
   integer, parameter :: MAT_SAI_SUB              = 18
   integer, parameter :: MAT_Z_AFF                = 19
   integer, parameter :: MAT_Z_NO_SPARSITY        = 20
   integer, parameter :: MAT_W_AFF                = 21
   integer, parameter :: MAT_W_NO_SPARSITY        = 22

   ! --------------------------------------------------------
   ! Indices into air_reuse_data%reuse_is(:)   (from air_data_type)
   ! --------------------------------------------------------
   integer, parameter :: IS_REPARTITION   = 1
   integer, parameter :: IS_R_Z_FINE_COLS = 2

   ! --------------------------------------------------------
   ! MatShell temporary-vector slot indices   (from matshell_data_type)
   ! --------------------------------------------------------
   integer, parameter :: MF_VEC_TEMP       = 1
   integer, parameter :: MF_VEC_TEMP_TWO   = 2
   integer, parameter :: MF_VEC_TEMP_THREE = 3
   integer, parameter :: MF_VEC_DIAG       = 4
   integer, parameter :: MF_VEC_RHS        = 5

   ! --------------------------------------------------------
   ! Timer IDs   (from timers)
   ! --------------------------------------------------------
   integer, parameter :: TIMER_ID_AIR_SETUP        = 1
   integer, parameter :: TIMER_ID_AIR_INVERSE      = 2
   integer, parameter :: TIMER_ID_AIR_DROP         = 3
   integer, parameter :: TIMER_ID_AIR_RAP          = 4
   integer, parameter :: TIMER_ID_AIR_EXTRACT      = 5
   integer, parameter :: TIMER_ID_AIR_PROLONG      = 6
   integer, parameter :: TIMER_ID_AIR_RESTRICT     = 7
   integer, parameter :: TIMER_ID_AIR_PROC_AGGLOM  = 8
   integer, parameter :: TIMER_ID_AIR_COARSEN      = 9
   integer, parameter :: TIMER_ID_AIR_CONSTRAIN    = 10
   integer, parameter :: TIMER_ID_AIR_IDENTITY     = 11
   integer, parameter :: TIMER_ID_AIR_TRUNCATE     = 12

   ! --------------------------------------------------------
   ! PCAIRGetPolyCoeffs / PCAIRSetPolyCoeffs selector indices
   ! --------------------------------------------------------
   integer, parameter :: COEFFS_INV_AFF         = 0
   integer, parameter :: COEFFS_INV_AFF_DROPPED = 1
   integer, parameter :: COEFFS_INV_ACC         = 2
   integer, parameter :: COEFFS_INV_COARSE      = 3

   ! --------------------------------------------------------
   ! Precision-aware tolerances
   ! --------------------------------------------------------
   ! The double branch below reproduces the previous hardcoded literals EXACTLY
   ! (including whether a literal was single-default `1e-N` or double `1d-N`, which
   ! are bit-distinct once widened), so double builds stay bit-identical. The
   ! single branch scales each to a reachable magnitude (eps_single ~ 1.19e-7 vs
   ! eps_double ~ 2.2e-16); the single values are validated/tuned in the
   ! single-precision bring-up. PetscReal expands to real(C_FLOAT/C_DOUBLE), so
   ! a d0/e0 literal assigned to a PetscReal parameter converts exactly at compile
   ! time, and epsilon() of a PetscReal entity gives working-precision eps.

   ! Machine epsilon of the build's PetscReal (auto per precision - no branch needed)
   PetscReal, parameter, private :: PFLARE_ONE_REAL = 1.0
   PetscReal, parameter :: PFLARE_EPS = epsilon(PFLARE_ONE_REAL)

#if defined(PETSC_USE_REAL_SINGLE)
   ! Coefficient/root "is effectively zero" tests (Gmres_Poly/Gmres_Poly_Newton)
   PetscReal, parameter :: PFLARE_TOL_ZERO           = 1e-6
   ! dgelsd rcond (harmonic Ritz min-norm solve)
   PetscReal, parameter :: PFLARE_TOL_RCOND          = 1e-6
   ! Matrix-free vs assembled reconstruction NaN/blow-up guards
   PetscReal, parameter :: PFLARE_TOL_MATFREE_12     = 1e-4
   PetscReal, parameter :: PFLARE_TOL_MATFREE_13     = 1e-4
   PetscReal, parameter :: PFLARE_TOL_MATFREE_4EM11  = 1e-3
   ! pseudo-inverse singular-value drop
   PetscReal, parameter :: PFLARE_TOL_SIGMA_DROP     = 1e-6
   ! Arnoldi least-squares relative-residual target (default)
   PetscReal, parameter :: PFLARE_TOL_ARNOLDI        = 1e-6
   ! Complex-conjugate root-pair consistency check
   PetscReal, parameter :: PFLARE_TOL_CONSISTENCY    = 1e-5
   ! Auto-truncate tolerance (user-tunable default)
   PetscReal, parameter :: PFLARE_TOL_AUTO_TRUNCATE  = 1e-6
   ! Constrain-grid-transfer inner KSP relative tolerance
   PetscReal, parameter :: PFLARE_KSP_RTOL_CONSTRAIN = 1e-6
   ! Diagonal-dominance ratio tolerances
   PetscReal, parameter :: PFLARE_DD_RATIO_ABS_TOL   = 1e-6
   PetscReal, parameter :: PFLARE_DD_RATIO_REL_TOL   = 1e-6
   ! Smoother / coarse-solver KSP tolerances
   PetscReal, parameter :: PFLARE_KSP_ATOL_SMOOTH    = 1e-6
   PetscReal, parameter :: PFLARE_KSP_ATOL_COARSE    = 1e-6
   ! Arnoldi "lucky breakdown" tolerance (was below double norms; sub-tiny in single)
   PetscReal, parameter :: PFLARE_TOL_LUCKY          = 1e-20
   ! remove_small "drop essentially nothing" sentinel (1d-100 underflows single)
   PetscReal, parameter :: PFLARE_SENTINEL_DROP      = 1e-30
#else
   ! Double branch - EXACT previous literals (bit-identical double builds)
   PetscReal, parameter :: PFLARE_TOL_ZERO           = 1e-12
   PetscReal, parameter :: PFLARE_TOL_RCOND          = 1e-12
   PetscReal, parameter :: PFLARE_TOL_MATFREE_12     = 1d-12
   PetscReal, parameter :: PFLARE_TOL_MATFREE_13     = 1d-13
   PetscReal, parameter :: PFLARE_TOL_MATFREE_4EM11  = 4d-11
   PetscReal, parameter :: PFLARE_TOL_SIGMA_DROP     = 1e-13
   PetscReal, parameter :: PFLARE_TOL_ARNOLDI        = 1e-14
   PetscReal, parameter :: PFLARE_TOL_CONSISTENCY    = 1e-14
   PetscReal, parameter :: PFLARE_TOL_AUTO_TRUNCATE  = 1e-14
   PetscReal, parameter :: PFLARE_KSP_RTOL_CONSTRAIN = 1d-14
   PetscReal, parameter :: PFLARE_DD_RATIO_ABS_TOL   = 1d-12
   PetscReal, parameter :: PFLARE_DD_RATIO_REL_TOL   = 1d-10
   PetscReal, parameter :: PFLARE_KSP_ATOL_SMOOTH    = 1d-10
   PetscReal, parameter :: PFLARE_KSP_ATOL_COARSE    = 1d-13
   PetscReal, parameter :: PFLARE_TOL_LUCKY          = 1d-30
   PetscReal, parameter :: PFLARE_SENTINEL_DROP      = 1d-100
#endif

end module pflare_parameters
