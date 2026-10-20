module pflare_parameters

   ! Centralised integer/enum parameters for PFLARE.
   ! Collecting them here avoids downstream files having to load
   ! heavy computational modules just to get constants.
   ! The values are defined in src/pflare_parameters.h, which the C also includes.

   use petscsys

#include "petsc/finclude/petscsys.h"
! Gives the Fortran spelling of the real literals in pflare_parameters.h and
! leaves out its C names, which are the names of some of the parameters below
#define PFLARE_PARAMETERS_FORTRAN
! The path is relative to the top directory, which is on the include path,
! so makedepf90 finds the header
#include "src/pflare_parameters.h"

   implicit none
   public

   ! --------------------------------------------------------
   ! Approximate inverse types (PCPFLAREINVType)
   ! --------------------------------------------------------
   PetscEnum, parameter :: PFLAREINV_POWER             = PFLARE_PFLAREINV_POWER
   PetscEnum, parameter :: PFLAREINV_ARNOLDI           = PFLARE_PFLAREINV_ARNOLDI
   PetscEnum, parameter :: PFLAREINV_NEWTON            = PFLARE_PFLAREINV_NEWTON
   PetscEnum, parameter :: PFLAREINV_NEWTON_NO_EXTRA   = PFLARE_PFLAREINV_NEWTON_NO_EXTRA
   PetscEnum, parameter :: PFLAREINV_NEUMANN           = PFLARE_PFLAREINV_NEUMANN
   PetscEnum, parameter :: PFLAREINV_SAI               = PFLARE_PFLAREINV_SAI
   PetscEnum, parameter :: PFLAREINV_ISAI              = PFLARE_PFLAREINV_ISAI
   PetscEnum, parameter :: PFLAREINV_WJACOBI           = PFLARE_PFLAREINV_WJACOBI
   PetscEnum, parameter :: PFLAREINV_JACOBI            = PFLARE_PFLAREINV_JACOBI

   ! --------------------------------------------------------
   ! Z / restrictor construction types (PCAIRZType)
   ! --------------------------------------------------------
   PetscEnum, parameter :: AIR_Z_PRODUCT   = PFLARE_AIR_Z_PRODUCT
   PetscEnum, parameter :: AIR_Z_LAIR      = PFLARE_AIR_Z_LAIR
   PetscEnum, parameter :: AIR_Z_LAIR_SAI  = PFLARE_AIR_Z_LAIR_SAI

   ! --------------------------------------------------------
   ! CF-splitting point markers
   ! --------------------------------------------------------
   integer, parameter :: C_POINT = PFLARE_C_POINT
   integer, parameter :: F_POINT = PFLARE_F_POINT

   ! --------------------------------------------------------
   ! Compatible relaxation (CF_CR) parameters
   ! --------------------------------------------------------
   ! Number of weighted Jacobi sweeps on Aff per CR pass
   integer, parameter :: PFLARE_CR_NU = PFLARE_CR_NU_VALUE
   ! Number of GMRES polynomial applications per CR pass
   ! This is deliberately 1 - AIR applies its F-solve once per smoothing step,
   ! so the quantity that gates AIR convergence is the ONE application
   ! contraction ||I - M Aff||, not the asymptotic rate over many applications
   ! (for advection Aff is near triangular so I - M Aff is close to nilpotent
   ! and the asymptotic rate looks great for any splitting)
   integer, parameter :: PFLARE_CR_NU_POLY = PFLARE_CR_NU_POLY_VALUE
   ! Order of the assembled power-basis GMRES polynomial used as the CR
   ! relaxation, matching the default -pc_air_poly_order so the CR rate
   ! measures the same F-point smoothing that AIRG actually applies
   integer, parameter :: PFLARE_CR_POLY_ORDER = PFLARE_CR_POLY_ORDER_VALUE
   ! Safety cap on the number of CR passes
   integer, parameter :: PFLARE_CR_MAX_ITS = PFLARE_CR_MAX_ITS_VALUE
   ! Candidate threshold - F rows whose relaxed error is bigger than this
   ! fraction of the biggest remaining error are eligible for promotion to C
   ! (the hypre CR candidate measure; permissive, the independent set then
   ! decides which candidates actually get promoted each pass)
   PetscReal, parameter :: PFLARE_CR_CANDIDATE = PFLARE_CR_CANDIDATE_VALUE

   ! --------------------------------------------------------
   ! Indices into air_reuse_data%reuse_mat(:)   (from air_data_type)
   ! --------------------------------------------------------
   integer, parameter :: MAT_AP                   = PFLARE_MAT_AP
   integer, parameter :: MAT_RAP                  = PFLARE_MAT_RAP
   integer, parameter :: MAT_RAP_DROP             = PFLARE_MAT_RAP_DROP
   integer, parameter :: MAT_Z_DROP               = PFLARE_MAT_Z_DROP
   integer, parameter :: MAT_W_DROP               = PFLARE_MAT_W_DROP
   integer, parameter :: MAT_COARSE_REPARTITIONED = PFLARE_MAT_COARSE_REPARTITIONED
   integer, parameter :: MAT_P_REPARTITIONED      = PFLARE_MAT_P_REPARTITIONED
   integer, parameter :: MAT_R_REPARTITIONED      = PFLARE_MAT_R_REPARTITIONED
   integer, parameter :: MAT_AFF_DROP             = PFLARE_MAT_AFF_DROP
   integer, parameter :: MAT_ACF_DROP             = PFLARE_MAT_ACF_DROP
   integer, parameter :: MAT_AFC_DROP             = PFLARE_MAT_AFC_DROP
   integer, parameter :: MAT_A_DROP               = PFLARE_MAT_A_DROP
   integer, parameter :: MAT_W                    = PFLARE_MAT_W
   integer, parameter :: MAT_Z                    = PFLARE_MAT_Z
   integer, parameter :: MAT_INV_AFF              = PFLARE_MAT_INV_AFF
   integer, parameter :: MAT_INV_AFF_DROPPED      = PFLARE_MAT_INV_AFF_DROPPED
   integer, parameter :: MAT_INV_ACC              = PFLARE_MAT_INV_ACC
   integer, parameter :: MAT_SAI_SUB              = PFLARE_MAT_SAI_SUB
   integer, parameter :: MAT_Z_AFF                = PFLARE_MAT_Z_AFF
   integer, parameter :: MAT_Z_NO_SPARSITY        = PFLARE_MAT_Z_NO_SPARSITY
   integer, parameter :: MAT_W_AFF                = PFLARE_MAT_W_AFF
   integer, parameter :: MAT_W_NO_SPARSITY        = PFLARE_MAT_W_NO_SPARSITY

   ! --------------------------------------------------------
   ! Indices into air_reuse_data%reuse_is(:)   (from air_data_type)
   ! --------------------------------------------------------
   integer, parameter :: IS_REPARTITION   = PFLARE_IS_REPARTITION
   integer, parameter :: IS_R_Z_FINE_COLS = PFLARE_IS_R_Z_FINE_COLS

   ! --------------------------------------------------------
   ! MatShell temporary-vector slot indices   (from matshell_data_type)
   ! --------------------------------------------------------
   integer, parameter :: MF_VEC_TEMP       = PFLARE_MF_VEC_TEMP
   integer, parameter :: MF_VEC_TEMP_TWO   = PFLARE_MF_VEC_TEMP_TWO
   integer, parameter :: MF_VEC_TEMP_THREE = PFLARE_MF_VEC_TEMP_THREE
   integer, parameter :: MF_VEC_DIAG       = PFLARE_MF_VEC_DIAG
   integer, parameter :: MF_VEC_RHS        = PFLARE_MF_VEC_RHS

   ! --------------------------------------------------------
   ! MatShell temporary dense matrix slot indices   (from matshell_data_type)
   ! These are the scratch dense blocks used by the multiple rhs (block)
   ! matrix-free polynomial applies
   ! --------------------------------------------------------
   integer, parameter :: MF_MAT_TEMP       = PFLARE_MF_MAT_TEMP
   integer, parameter :: MF_MAT_TEMP_TWO   = PFLARE_MF_MAT_TEMP_TWO
   integer, parameter :: MF_MAT_TEMP_THREE = PFLARE_MF_MAT_TEMP_THREE
   integer, parameter :: MF_MAT_RHS        = PFLARE_MF_MAT_RHS

   ! --------------------------------------------------------
   ! AIR temporary dense matrix slot indices   (from air_data_type)
   ! These are the scratch dense blocks used by the multiple rhs (block) FC
   ! smooths and they mirror the roles the temp_vecs_fine/temp_vecs_coarse
   ! vectors play in the single rhs smooths
   ! --------------------------------------------------------
   integer, parameter :: AIR_MAT_SOL      = PFLARE_AIR_MAT_SOL
   integer, parameter :: AIR_MAT_TEMP     = PFLARE_AIR_MAT_TEMP
   integer, parameter :: AIR_MAT_RESIDUAL = PFLARE_AIR_MAT_RESIDUAL
   integer, parameter :: AIR_MAT_RHS      = PFLARE_AIR_MAT_RHS
   ! Target of the cached off-diagonal product (A_fc * x_c or A_cf * x_f) - this
   ! has no single rhs twin, it exists so every cached product has its own
   ! dedicated target block (a mat can only hold one attached product)
   integer, parameter :: AIR_MAT_OFF_DIAG = PFLARE_AIR_MAT_OFF_DIAG

   ! --------------------------------------------------------
   ! Timer IDs   (from timers)
   ! --------------------------------------------------------
   integer, parameter :: TIMER_ID_AIR_SETUP        = PFLARE_TIMER_ID_AIR_SETUP
   integer, parameter :: TIMER_ID_AIR_INVERSE      = PFLARE_TIMER_ID_AIR_INVERSE
   integer, parameter :: TIMER_ID_AIR_DROP         = PFLARE_TIMER_ID_AIR_DROP
   integer, parameter :: TIMER_ID_AIR_RAP          = PFLARE_TIMER_ID_AIR_RAP
   integer, parameter :: TIMER_ID_AIR_EXTRACT      = PFLARE_TIMER_ID_AIR_EXTRACT
   integer, parameter :: TIMER_ID_AIR_PROLONG      = PFLARE_TIMER_ID_AIR_PROLONG
   integer, parameter :: TIMER_ID_AIR_RESTRICT     = PFLARE_TIMER_ID_AIR_RESTRICT
   integer, parameter :: TIMER_ID_AIR_PROC_AGGLOM  = PFLARE_TIMER_ID_AIR_PROC_AGGLOM
   integer, parameter :: TIMER_ID_AIR_COARSEN      = PFLARE_TIMER_ID_AIR_COARSEN
   integer, parameter :: TIMER_ID_AIR_CONSTRAIN    = PFLARE_TIMER_ID_AIR_CONSTRAIN
   integer, parameter :: TIMER_ID_AIR_IDENTITY     = PFLARE_TIMER_ID_AIR_IDENTITY
   integer, parameter :: TIMER_ID_AIR_TRUNCATE     = PFLARE_TIMER_ID_AIR_TRUNCATE

   ! --------------------------------------------------------
   ! PCAIRGetPolyCoeffs / PCAIRSetPolyCoeffs selector indices
   ! --------------------------------------------------------
   integer, parameter :: COEFFS_INV_AFF         = PFLARE_COEFFS_INV_AFF
   integer, parameter :: COEFFS_INV_AFF_DROPPED = PFLARE_COEFFS_INV_AFF_DROPPED
   integer, parameter :: COEFFS_INV_ACC         = PFLARE_COEFFS_INV_ACC
   integer, parameter :: COEFFS_INV_COARSE      = PFLARE_COEFFS_INV_COARSE

   ! --------------------------------------------------------
   ! Precision-aware tolerances
   ! --------------------------------------------------------
   ! The double branch in pflare_parameters.h reproduces the previous hardcoded literals EXACTLY
   ! (including whether a literal was single-default `1e-N` or double `1d-N`, which
   ! are bit-distinct once widened), so double builds stay bit-identical. The
   ! single branch scales each to a reachable magnitude (eps_single ~ 1.19e-7 vs
   ! eps_double ~ 2.2e-16); the single values are validated/tuned in the
   ! single-precision bring-up. PetscReal expands to real(C_FLOAT/C_DOUBLE), so
   ! a d0/e0 literal assigned to a PetscReal parameter converts exactly at compile
   ! time, and epsilon() of a PetscReal entity gives working-precision eps.

   ! Machine epsilon of the build's PetscReal (auto per precision - no branch needed)
   PetscReal, parameter, private :: PFLARE_ONE_REAL = PFLARE_ONE_VALUE
   PetscReal, parameter :: PFLARE_EPS = epsilon(PFLARE_ONE_REAL)
   ! Kind of the build's PetscReal - use for kind-suffixed literals / real() casts
   ! so double builds stay bit-identical while single builds avoid -Wconversion.
   integer, parameter :: PFLARE_REAL_KIND = kind(PFLARE_ONE_REAL)

   ! Kind-correct 0/1/-1 for PETSc value arguments (MatAXPY/MatScale/VecSet/
   ! VecAXPY alpha, MatSetValue value, MatMatMult fill, ...). A bare d0 literal
   ! actual is REAL(8) and mismatches a PetscScalar/PetscReal REAL(4) dummy under
   ! single precision. Typed as PetscScalar, which shares PetscReal's kind in all
   ! real builds, so these are valid actuals for both PetscScalar and PetscReal
   ! dummies. Bit-identical to 0d0/1d0/-1d0 in double builds.
   PetscScalar, parameter :: PFLARE_ZERO      = PFLARE_ZERO_VALUE
   PetscScalar, parameter :: PFLARE_ONE       = PFLARE_ONE_VALUE
   PetscScalar, parameter :: PFLARE_MINUS_ONE = PFLARE_MINUS_ONE_VALUE
   PetscScalar, parameter :: PFLARE_TWO       = PFLARE_TWO_VALUE
   ! MatMatMult / MatPtAP fill-ratio estimates (PetscReal args; separate values)
   PetscReal, parameter :: PFLARE_MATMULT_FILL = PFLARE_MATMULT_FILL_VALUE
   PetscReal, parameter :: PFLARE_PTAP_FILL    = PFLARE_PTAP_FILL_VALUE

   ! Coefficient/root "is effectively zero" tests (Gmres_Poly/Gmres_Poly_Newton)
   PetscReal, parameter :: PFLARE_TOL_ZERO           = PFLARE_TOL_ZERO_VALUE
   ! dgelsd rcond (harmonic Ritz min-norm solve)
   PetscReal, parameter :: PFLARE_TOL_RCOND          = PFLARE_TOL_RCOND_VALUE
   ! Kokkos-vs-CPU consistency (PFLARE_KOKKOS_DEBUG) and matrix-free reconstruction
   ! guards. In single these diffs are ordinary float32 noise (kokkos reduction
   ! order + accumulated error); 4EM11 covers the noisier dense-solve/matmul-powers
   ! comparisons (~2e-3 on large problems), well below the O(1) a real bug produces.
   PetscReal, parameter :: PFLARE_TOL_MATFREE_12     = PFLARE_TOL_MATFREE_12_VALUE
   PetscReal, parameter :: PFLARE_TOL_MATFREE_13     = PFLARE_TOL_MATFREE_13_VALUE
   PetscReal, parameter :: PFLARE_TOL_MATFREE_4EM11  = PFLARE_TOL_MATFREE_4EM11_VALUE
   ! Newton fixed-sparsity Kokkos-vs-CPU base tolerance, scaled by num_terms**2 at
   ! the call site to cover compounded per-term rounding drift at high poly order
   PetscReal, parameter :: PFLARE_TOL_MATFREE_NEWTON = PFLARE_TOL_MATFREE_NEWTON_VALUE
   ! pseudo-inverse singular-value drop
   PetscReal, parameter :: PFLARE_TOL_SIGMA_DROP     = PFLARE_TOL_SIGMA_DROP_VALUE
   ! Relative companion to PFLARE_TOL_SIGMA_DROP (numpy pinv rcond convention);
   ! the effective cutoff is max(absolute, relative * sigma_max)
   PetscReal, parameter :: PFLARE_TOL_SIGMA_DROP_REL = PFLARE_TOL_SIGMA_DROP_REL_VALUE
   ! Constrain-grid-transfer: a row is left unconstrained when the near-nullspace
   ! vectors restricted to that row's sparsity pattern are negligible relative to
   ! their own global scale, ie max|B_c|S_i| <= tol * ||B_c||_inf. Without this
   ! the row-wise projection divides by ||B_c|S_i||^2 ~ 0 and amplifies the
   ! grid-transfer entries without bound.
   PetscReal, parameter :: PFLARE_TOL_CONSTRAIN_REL  = PFLARE_TOL_CONSTRAIN_REL_VALUE
   ! Absolute floor below which the near-nullspace vectors are treated as having
   ! collapsed to zero and no constraints are applied on that level. They start
   ! as the constant (all ones) before smoothing, so this is a meaningful scale.
   PetscReal, parameter :: PFLARE_TOL_CONSTRAIN_ABS  = PFLARE_TOL_CONSTRAIN_ABS_VALUE
   ! Arnoldi least-squares relative-residual target (default)
   PetscReal, parameter :: PFLARE_TOL_ARNOLDI        = PFLARE_TOL_ARNOLDI_VALUE
   ! Complex-conjugate root-pair consistency check
   PetscReal, parameter :: PFLARE_TOL_CONSISTENCY    = PFLARE_TOL_CONSISTENCY_VALUE
   ! Auto-truncate tolerance (user-tunable default)
   PetscReal, parameter :: PFLARE_TOL_AUTO_TRUNCATE  = PFLARE_TOL_AUTO_TRUNCATE_VALUE
   ! Constrain-grid-transfer inner KSP relative tolerance
   PetscReal, parameter :: PFLARE_KSP_RTOL_CONSTRAIN = PFLARE_KSP_RTOL_CONSTRAIN_VALUE
   ! Diagonal-dominance ratio tolerances
   PetscReal, parameter :: PFLARE_DD_RATIO_ABS_TOL   = PFLARE_DD_RATIO_ABS_TOL_VALUE
   PetscReal, parameter :: PFLARE_DD_RATIO_REL_TOL   = PFLARE_DD_RATIO_REL_TOL_VALUE
   ! Smoother / coarse-solver KSP tolerances
   PetscReal, parameter :: PFLARE_KSP_ATOL_SMOOTH    = PFLARE_KSP_ATOL_SMOOTH_VALUE
   PetscReal, parameter :: PFLARE_KSP_ATOL_COARSE    = PFLARE_KSP_ATOL_COARSE_VALUE
   ! Coarse KSPPREONLY rtol (value irrelevant for preonly; kept for kind-correctness)
   PetscReal, parameter :: PFLARE_KSP_RTOL_COARSE    = PFLARE_KSP_RTOL_COARSE_VALUE
   ! "atol effectively off" sentinel (1d-50 underflows single)
   PetscReal, parameter :: PFLARE_KSP_ATOL_OFF       = PFLARE_KSP_ATOL_OFF_VALUE
   ! Arnoldi "lucky breakdown" tolerance (was below double norms; sub-tiny in single)
   PetscReal, parameter :: PFLARE_TOL_LUCKY          = PFLARE_TOL_LUCKY_VALUE
   ! remove_small "drop essentially nothing" sentinel (1d-100 underflows single)
   PetscReal, parameter :: PFLARE_SENTINEL_DROP      = PFLARE_SENTINEL_DROP_VALUE
   ! Leja-ordering perturbation for duplicate added roots (Gmres_Poly_Newton) -
   ! 5e-8 underflows single eps (~1.19e-7), so duplicates would stay bit-identical
   PetscReal, parameter :: PFLARE_TOL_LEJA_PERTURB   = PFLARE_TOL_LEJA_PERTURB_VALUE

end module pflare_parameters
