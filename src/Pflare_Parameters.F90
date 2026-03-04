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

end module pflare_parameters
