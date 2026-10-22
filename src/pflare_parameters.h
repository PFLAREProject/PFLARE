/*
  Constants and enum values used by PFLARE

  This is the one place the values are defined. The C includes this header
  (through pflareimpl.h) and so does the Fortran module pflare_parameters in
  src/fortran/Pflare_Parameters.F90, which defines its parameters from these
  values, so this header can only contain #define, #if and #include <petscconf.h>
  lines and comments like this one (the Fortran is preprocessed in traditional
  mode, which knows nothing of C). Please see the Fortran module for the
  reasons behind each value.

  Names: each value is the name of the Fortran parameter with a PFLARE_ prefix,
  or the Fortran name itself when that already starts with PFLARE_. The Fortran
  module cannot see a macro with the same name as one of its parameters (the
  preprocessor would replace the parameter name), so those values are defined
  as <name>_VALUE and the C name is defined from it at the end of this header,
  for the C only. The Fortran module defines PFLARE_PARAMETERS_FORTRAN before
  it includes this header.
*/

#if !defined(PFLARE_PARAMETERS_H)
#define PFLARE_PARAMETERS_H

#include <petscconf.h>

/* ~~~~~~~~~~~~~~~~~~~~~~~~
   Real literals
   A Fortran 1e-12 is a single precision literal and a C 1e-12 a double one,
   and 1d-12 is not C, so the real values are written with these macros, which
   give a literal of the same kind and value in both languages:
     PFLARE_SINGLE_LITERAL(x)    the single precision literal x
                                 (x in Fortran, xf in C)
     PFLARE_DOUBLE_LITERAL(m,n)  the double precision literal m times 10^-n
                                 (m.d-n in Fortran, me-n in C)
   Write the arguments without spaces: the Fortran preprocessor keeps them and
   a literal cannot contain a space
   ~~~~~~~~~~~~~~~~~~~~~~~~ */
#if defined(PFLARE_PARAMETERS_FORTRAN)
#define PFLARE_SINGLE_LITERAL(x) x
#define PFLARE_DOUBLE_LITERAL(m,n) m.d-n
#else
#define PFLARE_SINGLE_LITERAL(x) x##f
#define PFLARE_DOUBLE_LITERAL(m,n) m##e##-##n
#endif

/* --------------------------------------------------------
   Approximate inverse types (PCPFLAREINVType)
   -------------------------------------------------------- */
#define PFLARE_PFLAREINV_POWER           0
#define PFLARE_PFLAREINV_ARNOLDI         1
#define PFLARE_PFLAREINV_NEWTON          2
#define PFLARE_PFLAREINV_NEWTON_NO_EXTRA 3
#define PFLARE_PFLAREINV_NEUMANN         4
#define PFLARE_PFLAREINV_SAI             5
#define PFLARE_PFLAREINV_ISAI            6
#define PFLARE_PFLAREINV_WJACOBI         7
#define PFLARE_PFLAREINV_JACOBI          8
#define PFLARE_PFLAREINV_L1JACOBI        9

/* --------------------------------------------------------
   Z / restrictor construction types (PCAIRZType)
   -------------------------------------------------------- */
#define PFLARE_AIR_Z_PRODUCT  0
#define PFLARE_AIR_Z_LAIR     1
#define PFLARE_AIR_Z_LAIR_SAI 2

/* --------------------------------------------------------
   CF-splitting point markers
   -------------------------------------------------------- */
#define PFLARE_C_POINT  1
#define PFLARE_F_POINT -1

/* --------------------------------------------------------
   Compatible relaxation (CF_CR) parameters
   -------------------------------------------------------- */
/* Number of weighted Jacobi sweeps on Aff per CR pass */
#define PFLARE_CR_NU_VALUE 10
/* Number of GMRES polynomial applications per CR pass */
#define PFLARE_CR_NU_POLY_VALUE 1
/* Order of the assembled power-basis GMRES polynomial used as the CR relaxation */
#define PFLARE_CR_POLY_ORDER_VALUE 6
/* Safety cap on the number of CR passes */
#define PFLARE_CR_MAX_ITS_VALUE 100
/* Candidate threshold - F rows whose relaxed error is bigger than this
   fraction of the biggest remaining error are eligible for promotion to C */
#define PFLARE_CR_CANDIDATE_VALUE PFLARE_SINGLE_LITERAL(0.1)

/* --------------------------------------------------------
   Indices into air_reuse_data%reuse_mat(:)   (from air_data_type)
   -------------------------------------------------------- */
#define PFLARE_MAT_AP                   1
#define PFLARE_MAT_RAP                  2
#define PFLARE_MAT_RAP_DROP             3
#define PFLARE_MAT_Z_DROP               4
#define PFLARE_MAT_W_DROP               5
#define PFLARE_MAT_COARSE_REPARTITIONED 6
#define PFLARE_MAT_P_REPARTITIONED      7
#define PFLARE_MAT_R_REPARTITIONED      8
#define PFLARE_MAT_AFF_DROP             9
#define PFLARE_MAT_ACF_DROP             10
#define PFLARE_MAT_AFC_DROP             11
#define PFLARE_MAT_A_DROP               12
#define PFLARE_MAT_W                    13
#define PFLARE_MAT_Z                    14
#define PFLARE_MAT_INV_AFF              15
#define PFLARE_MAT_INV_AFF_DROPPED      16
#define PFLARE_MAT_INV_ACC              17
#define PFLARE_MAT_SAI_SUB              18
#define PFLARE_MAT_Z_AFF                19
#define PFLARE_MAT_Z_NO_SPARSITY        20
#define PFLARE_MAT_W_AFF                21
#define PFLARE_MAT_W_NO_SPARSITY        22

/* --------------------------------------------------------
   Indices into air_reuse_data%reuse_is(:)   (from air_data_type)
   -------------------------------------------------------- */
#define PFLARE_IS_REPARTITION   1
#define PFLARE_IS_R_Z_FINE_COLS 2

/* --------------------------------------------------------
   MatShell temporary-vector slot indices   (from matshell_data_type)
   -------------------------------------------------------- */
#define PFLARE_MF_VEC_TEMP       1
#define PFLARE_MF_VEC_TEMP_TWO   2
#define PFLARE_MF_VEC_TEMP_THREE 3
#define PFLARE_MF_VEC_DIAG       4
#define PFLARE_MF_VEC_RHS        5

/* --------------------------------------------------------
   MatShell temporary dense matrix slot indices   (from matshell_data_type)
   These are the scratch dense blocks used by the multiple rhs (block)
   matrix-free polynomial applies
   -------------------------------------------------------- */
#define PFLARE_MF_MAT_TEMP       1
#define PFLARE_MF_MAT_TEMP_TWO   2
#define PFLARE_MF_MAT_TEMP_THREE 3
#define PFLARE_MF_MAT_RHS        4

/* --------------------------------------------------------
   AIR temporary dense matrix slot indices   (from air_data_type)
   These are the scratch dense blocks used by the multiple rhs (block) FC
   smooths and they mirror the roles the temp_vecs_fine/temp_vecs_coarse
   vectors play in the single rhs smooths
   -------------------------------------------------------- */
#define PFLARE_AIR_MAT_SOL      1
#define PFLARE_AIR_MAT_TEMP     2
#define PFLARE_AIR_MAT_RESIDUAL 3
#define PFLARE_AIR_MAT_RHS      4
/* Target of the cached off-diagonal product (A_fc * x_c or A_cf * x_f) */
#define PFLARE_AIR_MAT_OFF_DIAG 5

/* --------------------------------------------------------
   Timer IDs   (from timers)
   -------------------------------------------------------- */
#define PFLARE_TIMER_ID_AIR_SETUP       1
#define PFLARE_TIMER_ID_AIR_INVERSE     2
#define PFLARE_TIMER_ID_AIR_DROP        3
#define PFLARE_TIMER_ID_AIR_RAP         4
#define PFLARE_TIMER_ID_AIR_EXTRACT     5
#define PFLARE_TIMER_ID_AIR_PROLONG     6
#define PFLARE_TIMER_ID_AIR_RESTRICT    7
#define PFLARE_TIMER_ID_AIR_PROC_AGGLOM 8
#define PFLARE_TIMER_ID_AIR_COARSEN     9
#define PFLARE_TIMER_ID_AIR_CONSTRAIN   10
#define PFLARE_TIMER_ID_AIR_IDENTITY    11
#define PFLARE_TIMER_ID_AIR_TRUNCATE    12

/* --------------------------------------------------------
   PCAIRGetPolyCoeffs / PCAIRSetPolyCoeffs selector indices
   -------------------------------------------------------- */
#define PFLARE_COEFFS_INV_AFF         0
#define PFLARE_COEFFS_INV_AFF_DROPPED 1
#define PFLARE_COEFFS_INV_ACC         2
#define PFLARE_COEFFS_INV_COARSE      3

/* --------------------------------------------------------
   Precision-aware tolerances
   -------------------------------------------------------- */
/* Kind-correct 0/1/-1/2 for PETSc value arguments (PetscScalar in the Fortran) */
#define PFLARE_ZERO_VALUE      PFLARE_SINGLE_LITERAL(0.0)
#define PFLARE_ONE_VALUE       PFLARE_SINGLE_LITERAL(1.0)
#define PFLARE_MINUS_ONE_VALUE PFLARE_SINGLE_LITERAL(-1.0)
#define PFLARE_TWO_VALUE       PFLARE_SINGLE_LITERAL(2.0)
/* MatMatMult / MatPtAP fill-ratio estimates */
#define PFLARE_MATMULT_FILL_VALUE PFLARE_SINGLE_LITERAL(1.5)
#define PFLARE_PTAP_FILL_VALUE    PFLARE_SINGLE_LITERAL(1.58)

#if defined(PETSC_USE_REAL_SINGLE)
/* Coefficient/root "is effectively zero" tests (Gmres_Poly/Gmres_Poly_Newton) */
#define PFLARE_TOL_ZERO_VALUE           PFLARE_SINGLE_LITERAL(1e-6)
/* dgelsd rcond (harmonic Ritz min-norm solve) */
#define PFLARE_TOL_RCOND_VALUE          PFLARE_SINGLE_LITERAL(1e-6)
/* Kokkos-vs-CPU consistency (PFLARE_KOKKOS_DEBUG) and matrix-free reconstruction
   guards. In single these diffs are ordinary float32 noise (kokkos reduction
   order + accumulated error); 4EM11 covers the noisier dense-solve/matmul-powers
   comparisons (~2e-3 on large problems), well below the O(1) a real bug produces. */
#define PFLARE_TOL_MATFREE_12_VALUE     PFLARE_SINGLE_LITERAL(1e-4)
#define PFLARE_TOL_MATFREE_13_VALUE     PFLARE_SINGLE_LITERAL(1e-4)
#define PFLARE_TOL_MATFREE_4EM11_VALUE  PFLARE_SINGLE_LITERAL(1e-2)
/* Newton fixed-sparsity Kokkos-vs-CPU base tolerance, scaled by num_terms**2 at
   the call site to cover compounded per-term rounding drift at high poly order */
#define PFLARE_TOL_MATFREE_NEWTON_VALUE PFLARE_SINGLE_LITERAL(1e-3)
/* pseudo-inverse singular-value drop */
#define PFLARE_TOL_SIGMA_DROP_VALUE     PFLARE_SINGLE_LITERAL(1e-6)
/* Relative companion to PFLARE_TOL_SIGMA_DROP (numpy pinv rcond convention);
   the effective cutoff is max(absolute, relative * sigma_max) */
#define PFLARE_TOL_SIGMA_DROP_REL_VALUE PFLARE_SINGLE_LITERAL(1e-6)
/* Constrain-grid-transfer: a row is left unconstrained when the near-nullspace
   vectors restricted to that row's sparsity pattern are negligible relative to
   their own global scale, ie max|B_c|S_i| <= tol * ||B_c||_inf */
#define PFLARE_TOL_CONSTRAIN_REL_VALUE  PFLARE_SINGLE_LITERAL(1e-1)
/* Absolute floor below which the near-nullspace vectors are treated as having
   collapsed to zero and no constraints are applied on that level */
#define PFLARE_TOL_CONSTRAIN_ABS_VALUE  PFLARE_SINGLE_LITERAL(1e-6)
/* Arnoldi least-squares relative-residual target (default) */
#define PFLARE_TOL_ARNOLDI_VALUE        PFLARE_SINGLE_LITERAL(1e-6)
/* Complex-conjugate root-pair consistency check */
#define PFLARE_TOL_CONSISTENCY_VALUE    PFLARE_SINGLE_LITERAL(1e-5)
/* Auto-truncate tolerance (user-tunable default) */
#define PFLARE_TOL_AUTO_TRUNCATE_VALUE  PFLARE_SINGLE_LITERAL(1e-6)
/* Constrain-grid-transfer inner KSP relative tolerance */
#define PFLARE_KSP_RTOL_CONSTRAIN_VALUE PFLARE_SINGLE_LITERAL(1e-6)
/* Diagonal-dominance ratio tolerances */
#define PFLARE_DD_RATIO_ABS_TOL_VALUE   PFLARE_SINGLE_LITERAL(1e-6)
#define PFLARE_DD_RATIO_REL_TOL_VALUE   PFLARE_SINGLE_LITERAL(1e-6)
/* Smoother / coarse-solver KSP tolerances */
#define PFLARE_KSP_ATOL_SMOOTH_VALUE    PFLARE_SINGLE_LITERAL(1e-6)
#define PFLARE_KSP_ATOL_COARSE_VALUE    PFLARE_SINGLE_LITERAL(1e-6)
/* Coarse KSPPREONLY rtol (value irrelevant for preonly; kept for kind-correctness) */
#define PFLARE_KSP_RTOL_COARSE_VALUE    PFLARE_SINGLE_LITERAL(1e-3)
/* "atol effectively off" sentinel (1d-50 underflows single) */
#define PFLARE_KSP_ATOL_OFF_VALUE       PFLARE_SINGLE_LITERAL(1e-30)
/* Arnoldi "lucky breakdown" tolerance (was below double norms; sub-tiny in single) */
#define PFLARE_TOL_LUCKY_VALUE          PFLARE_SINGLE_LITERAL(1e-20)
/* remove_small "drop essentially nothing" sentinel (1d-100 underflows single) */
#define PFLARE_SENTINEL_DROP_VALUE      PFLARE_SINGLE_LITERAL(1e-30)
/* Leja-ordering perturbation for duplicate added roots (Gmres_Poly_Newton) -
   5e-8 underflows single eps (~1.19e-7), so duplicates would stay bit-identical */
#define PFLARE_TOL_LEJA_PERTURB_VALUE   PFLARE_SINGLE_LITERAL(1e-4)
#else
/* Double branch - EXACT previous literals (bit-identical double builds), including
   whether a literal was single-default `1e-N` or double `1d-N`, which are
   bit-distinct once widened */
#define PFLARE_TOL_ZERO_VALUE           PFLARE_SINGLE_LITERAL(1e-12)
#define PFLARE_TOL_RCOND_VALUE          PFLARE_SINGLE_LITERAL(1e-12)
#define PFLARE_TOL_MATFREE_12_VALUE     PFLARE_DOUBLE_LITERAL(1,12)
#define PFLARE_TOL_MATFREE_13_VALUE     PFLARE_DOUBLE_LITERAL(1,13)
#define PFLARE_TOL_MATFREE_4EM11_VALUE  PFLARE_DOUBLE_LITERAL(4,11)
#define PFLARE_TOL_MATFREE_NEWTON_VALUE PFLARE_DOUBLE_LITERAL(1,11)
#define PFLARE_TOL_SIGMA_DROP_VALUE     PFLARE_SINGLE_LITERAL(1e-13)
#define PFLARE_TOL_SIGMA_DROP_REL_VALUE PFLARE_SINGLE_LITERAL(1e-13)
#define PFLARE_TOL_CONSTRAIN_REL_VALUE  PFLARE_SINGLE_LITERAL(1e-1)
#define PFLARE_TOL_CONSTRAIN_ABS_VALUE  PFLARE_SINGLE_LITERAL(1e-10)
#define PFLARE_TOL_ARNOLDI_VALUE        PFLARE_SINGLE_LITERAL(1e-14)
#define PFLARE_TOL_CONSISTENCY_VALUE    PFLARE_SINGLE_LITERAL(1e-14)
#define PFLARE_TOL_AUTO_TRUNCATE_VALUE  PFLARE_SINGLE_LITERAL(1e-14)
#define PFLARE_KSP_RTOL_CONSTRAIN_VALUE PFLARE_DOUBLE_LITERAL(1,14)
#define PFLARE_DD_RATIO_ABS_TOL_VALUE   PFLARE_DOUBLE_LITERAL(1,12)
#define PFLARE_DD_RATIO_REL_TOL_VALUE   PFLARE_DOUBLE_LITERAL(1,10)
#define PFLARE_KSP_ATOL_SMOOTH_VALUE    PFLARE_DOUBLE_LITERAL(1,10)
#define PFLARE_KSP_ATOL_COARSE_VALUE    PFLARE_DOUBLE_LITERAL(1,13)
#define PFLARE_KSP_RTOL_COARSE_VALUE    PFLARE_DOUBLE_LITERAL(1,3)
#define PFLARE_KSP_ATOL_OFF_VALUE       PFLARE_DOUBLE_LITERAL(1,50)
#define PFLARE_TOL_LUCKY_VALUE          PFLARE_DOUBLE_LITERAL(1,30)
#define PFLARE_SENTINEL_DROP_VALUE      PFLARE_DOUBLE_LITERAL(1,100)
#define PFLARE_TOL_LEJA_PERTURB_VALUE   PFLARE_SINGLE_LITERAL(5e-8)
#endif

/* ~~~~~~~~~~~~~~~~~~~~~~~~
   C names of the values whose Fortran parameter already starts with PFLARE_
   (the Fortran module defines its parameters of the same names from the
   _VALUE macros above)
   PFLARE_EPS is epsilon() of PetscReal in the Fortran, PETSC_MACHINE_EPSILON in C.
   The Fortran only parameters PFLARE_REAL_KIND (the kind of PetscReal) and the
   private PFLARE_ONE_REAL (PetscReal 1, from PFLARE_ONE_VALUE) have no C name
   ~~~~~~~~~~~~~~~~~~~~~~~~ */
#if !defined(PFLARE_PARAMETERS_FORTRAN)
#define PFLARE_CR_NU              PFLARE_CR_NU_VALUE
#define PFLARE_CR_NU_POLY         PFLARE_CR_NU_POLY_VALUE
#define PFLARE_CR_POLY_ORDER      PFLARE_CR_POLY_ORDER_VALUE
#define PFLARE_CR_MAX_ITS         PFLARE_CR_MAX_ITS_VALUE
#define PFLARE_CR_CANDIDATE       PFLARE_CR_CANDIDATE_VALUE
#define PFLARE_EPS                PETSC_MACHINE_EPSILON
#define PFLARE_ZERO               PFLARE_ZERO_VALUE
#define PFLARE_ONE                PFLARE_ONE_VALUE
#define PFLARE_MINUS_ONE          PFLARE_MINUS_ONE_VALUE
#define PFLARE_TWO                PFLARE_TWO_VALUE
#define PFLARE_MATMULT_FILL       PFLARE_MATMULT_FILL_VALUE
#define PFLARE_PTAP_FILL          PFLARE_PTAP_FILL_VALUE
#define PFLARE_TOL_ZERO           PFLARE_TOL_ZERO_VALUE
#define PFLARE_TOL_RCOND          PFLARE_TOL_RCOND_VALUE
#define PFLARE_TOL_MATFREE_12     PFLARE_TOL_MATFREE_12_VALUE
#define PFLARE_TOL_MATFREE_13     PFLARE_TOL_MATFREE_13_VALUE
#define PFLARE_TOL_MATFREE_4EM11  PFLARE_TOL_MATFREE_4EM11_VALUE
#define PFLARE_TOL_MATFREE_NEWTON PFLARE_TOL_MATFREE_NEWTON_VALUE
#define PFLARE_TOL_SIGMA_DROP     PFLARE_TOL_SIGMA_DROP_VALUE
#define PFLARE_TOL_SIGMA_DROP_REL PFLARE_TOL_SIGMA_DROP_REL_VALUE
#define PFLARE_TOL_CONSTRAIN_REL  PFLARE_TOL_CONSTRAIN_REL_VALUE
#define PFLARE_TOL_CONSTRAIN_ABS  PFLARE_TOL_CONSTRAIN_ABS_VALUE
#define PFLARE_TOL_ARNOLDI        PFLARE_TOL_ARNOLDI_VALUE
#define PFLARE_TOL_CONSISTENCY    PFLARE_TOL_CONSISTENCY_VALUE
#define PFLARE_TOL_AUTO_TRUNCATE  PFLARE_TOL_AUTO_TRUNCATE_VALUE
#define PFLARE_KSP_RTOL_CONSTRAIN PFLARE_KSP_RTOL_CONSTRAIN_VALUE
#define PFLARE_DD_RATIO_ABS_TOL   PFLARE_DD_RATIO_ABS_TOL_VALUE
#define PFLARE_DD_RATIO_REL_TOL   PFLARE_DD_RATIO_REL_TOL_VALUE
#define PFLARE_KSP_ATOL_SMOOTH    PFLARE_KSP_ATOL_SMOOTH_VALUE
#define PFLARE_KSP_ATOL_COARSE    PFLARE_KSP_ATOL_COARSE_VALUE
#define PFLARE_KSP_RTOL_COARSE    PFLARE_KSP_RTOL_COARSE_VALUE
#define PFLARE_KSP_ATOL_OFF       PFLARE_KSP_ATOL_OFF_VALUE
#define PFLARE_TOL_LUCKY          PFLARE_TOL_LUCKY_VALUE
#define PFLARE_SENTINEL_DROP      PFLARE_SENTINEL_DROP_VALUE
#define PFLARE_TOL_LEJA_PERTURB   PFLARE_TOL_LEJA_PERTURB_VALUE
#endif

#endif
