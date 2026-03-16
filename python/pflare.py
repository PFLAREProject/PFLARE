# Just import the pflare definitions
import pflare_defs
# And register all the types with PETSc
pflare_defs.py_PCRegister_PFLARE()

# CF splitting type constants
CF_PMISR_DDC  = 0
CF_DIAG_DOM   = 1
CF_PMIS       = 2
CF_PMIS_DIST2 = 3
CF_AGG        = 4
CF_PMIS_AGG   = 5

# Approximate inverse type constants (PCPFLAREINVType / PCAIRInverseType)
PFLAREINV_POWER           = 0
PFLAREINV_ARNOLDI         = 1
PFLAREINV_NEWTON          = 2
PFLAREINV_NEWTON_NO_EXTRA = 3
PFLAREINV_NEUMANN         = 4
PFLAREINV_SAI             = 5
PFLAREINV_ISAI            = 6
PFLAREINV_WJACOBI         = 7
PFLAREINV_JACOBI          = 8

# Z / restrictor type constants (PCAIRZType)
AIR_Z_PRODUCT  = 0
AIR_Z_LAIR     = 1
AIR_Z_LAIR_SAI = 2

# Selector constants for PCAIRGetPolyCoeffs / PCAIRSetPolyCoeffs
# These match the Fortran COEFFS_INV_* parameters in pflare_parameters
COEFFS_INV_AFF         = 0  # Inverse of the fine-fine block A_ff
COEFFS_INV_AFF_DROPPED = 1  # Inverse of the dropped fine-fine block
COEFFS_INV_ACC         = 2  # Inverse of the coarse-coarse block A_cc
COEFFS_INV_COARSE      = 3  # Inverse on the coarsest grid

# Standalone matrix utility wrappers
compute_cf_splitting          = pflare_defs.compute_cf_splitting
compute_diag_dom_submatrix    = pflare_defs.compute_diag_dom_submatrix

# -----------------------------------------------------------------------
# PCAIR Get functions
# The exact Python names for all pflare functions are listed here in
# pflare.py; they follow the pattern pcair_<option_name> matching the
# Fortran/C routine names without the PCAIR prefix and using snake_case.
# -----------------------------------------------------------------------
pcair_get_num_levels                   = pflare_defs.pcair_get_num_levels
pcair_get_print_stats_timings          = pflare_defs.pcair_get_print_stats_timings
pcair_get_max_levels                   = pflare_defs.pcair_get_max_levels
pcair_get_coarse_eq_limit              = pflare_defs.pcair_get_coarse_eq_limit
pcair_get_auto_truncate_start_level    = pflare_defs.pcair_get_auto_truncate_start_level
pcair_get_auto_truncate_tol            = pflare_defs.pcair_get_auto_truncate_tol
pcair_get_processor_agglom             = pflare_defs.pcair_get_processor_agglom
pcair_get_processor_agglom_ratio       = pflare_defs.pcair_get_processor_agglom_ratio
pcair_get_processor_agglom_factor      = pflare_defs.pcair_get_processor_agglom_factor
pcair_get_process_eq_limit             = pflare_defs.pcair_get_process_eq_limit
pcair_get_subcomm                      = pflare_defs.pcair_get_subcomm
pcair_get_strong_threshold             = pflare_defs.pcair_get_strong_threshold
pcair_get_ddc_its                      = pflare_defs.pcair_get_ddc_its
pcair_get_ddc_fraction                 = pflare_defs.pcair_get_ddc_fraction
pcair_get_cf_splitting_type            = pflare_defs.pcair_get_cf_splitting_type
pcair_get_max_luby_steps               = pflare_defs.pcair_get_max_luby_steps
pcair_get_diag_scale_polys             = pflare_defs.pcair_get_diag_scale_polys
pcair_get_matrix_free_polys            = pflare_defs.pcair_get_matrix_free_polys
pcair_get_one_point_classical_prolong  = pflare_defs.pcair_get_one_point_classical_prolong
pcair_get_full_smoothing_up_and_down   = pflare_defs.pcair_get_full_smoothing_up_and_down
pcair_get_symmetric                    = pflare_defs.pcair_get_symmetric
pcair_get_constrain_w                  = pflare_defs.pcair_get_constrain_w
pcair_get_constrain_z                  = pflare_defs.pcair_get_constrain_z
pcair_get_improve_w_its                = pflare_defs.pcair_get_improve_w_its
pcair_get_improve_z_its                = pflare_defs.pcair_get_improve_z_its
pcair_get_strong_r_threshold           = pflare_defs.pcair_get_strong_r_threshold
pcair_get_inverse_type                 = pflare_defs.pcair_get_inverse_type
pcair_get_c_inverse_type               = pflare_defs.pcair_get_c_inverse_type
pcair_get_z_type                       = pflare_defs.pcair_get_z_type
pcair_get_lair_distance                = pflare_defs.pcair_get_lair_distance
pcair_get_poly_order                   = pflare_defs.pcair_get_poly_order
pcair_get_inverse_sparsity_order       = pflare_defs.pcair_get_inverse_sparsity_order
pcair_get_c_poly_order                 = pflare_defs.pcair_get_c_poly_order
pcair_get_c_inverse_sparsity_order     = pflare_defs.pcair_get_c_inverse_sparsity_order
pcair_get_coarsest_inverse_type        = pflare_defs.pcair_get_coarsest_inverse_type
pcair_get_coarsest_poly_order          = pflare_defs.pcair_get_coarsest_poly_order
pcair_get_coarsest_inverse_sparsity_order = pflare_defs.pcair_get_coarsest_inverse_sparsity_order
pcair_get_coarsest_matrix_free_polys   = pflare_defs.pcair_get_coarsest_matrix_free_polys
pcair_get_coarsest_diag_scale_polys    = pflare_defs.pcair_get_coarsest_diag_scale_polys
pcair_get_coarsest_subcomm             = pflare_defs.pcair_get_coarsest_subcomm
pcair_get_r_drop                       = pflare_defs.pcair_get_r_drop
pcair_get_a_drop                       = pflare_defs.pcair_get_a_drop
pcair_get_a_lump                       = pflare_defs.pcair_get_a_lump
pcair_get_reuse_sparsity               = pflare_defs.pcair_get_reuse_sparsity
pcair_get_reuse_poly_coeffs            = pflare_defs.pcair_get_reuse_poly_coeffs
pcair_get_reuse_amount                 = pflare_defs.pcair_get_reuse_amount
pcair_get_smooth_type                  = pflare_defs.pcair_get_smooth_type
pcair_get_poly_coeffs                  = pflare_defs.pcair_get_poly_coeffs

# -----------------------------------------------------------------------
# PCAIR Set functions
# -----------------------------------------------------------------------
pcair_set_print_stats_timings          = pflare_defs.pcair_set_print_stats_timings
pcair_set_max_levels                   = pflare_defs.pcair_set_max_levels
pcair_set_coarse_eq_limit              = pflare_defs.pcair_set_coarse_eq_limit
pcair_set_auto_truncate_start_level    = pflare_defs.pcair_set_auto_truncate_start_level
pcair_set_auto_truncate_tol            = pflare_defs.pcair_set_auto_truncate_tol
pcair_set_processor_agglom             = pflare_defs.pcair_set_processor_agglom
pcair_set_processor_agglom_ratio       = pflare_defs.pcair_set_processor_agglom_ratio
pcair_set_processor_agglom_factor      = pflare_defs.pcair_set_processor_agglom_factor
pcair_set_process_eq_limit             = pflare_defs.pcair_set_process_eq_limit
pcair_set_subcomm                      = pflare_defs.pcair_set_subcomm
pcair_set_strong_threshold             = pflare_defs.pcair_set_strong_threshold
pcair_set_ddc_its                      = pflare_defs.pcair_set_ddc_its
pcair_set_ddc_fraction                 = pflare_defs.pcair_set_ddc_fraction
pcair_set_cf_splitting_type            = pflare_defs.pcair_set_cf_splitting_type
pcair_set_max_luby_steps               = pflare_defs.pcair_set_max_luby_steps
pcair_set_smooth_type                  = pflare_defs.pcair_set_smooth_type
pcair_set_diag_scale_polys             = pflare_defs.pcair_set_diag_scale_polys
pcair_set_matrix_free_polys            = pflare_defs.pcair_set_matrix_free_polys
pcair_set_one_point_classical_prolong  = pflare_defs.pcair_set_one_point_classical_prolong
pcair_set_full_smoothing_up_and_down   = pflare_defs.pcair_set_full_smoothing_up_and_down
pcair_set_symmetric                    = pflare_defs.pcair_set_symmetric
pcair_set_constrain_w                  = pflare_defs.pcair_set_constrain_w
pcair_set_constrain_z                  = pflare_defs.pcair_set_constrain_z
pcair_set_improve_w_its                = pflare_defs.pcair_set_improve_w_its
pcair_set_improve_z_its                = pflare_defs.pcair_set_improve_z_its
pcair_set_strong_r_threshold           = pflare_defs.pcair_set_strong_r_threshold
pcair_set_inverse_type                 = pflare_defs.pcair_set_inverse_type
pcair_set_c_inverse_type               = pflare_defs.pcair_set_c_inverse_type
pcair_set_z_type                       = pflare_defs.pcair_set_z_type
pcair_set_lair_distance                = pflare_defs.pcair_set_lair_distance
pcair_set_poly_order                   = pflare_defs.pcair_set_poly_order
pcair_set_inverse_sparsity_order       = pflare_defs.pcair_set_inverse_sparsity_order
pcair_set_c_poly_order                 = pflare_defs.pcair_set_c_poly_order
pcair_set_c_inverse_sparsity_order     = pflare_defs.pcair_set_c_inverse_sparsity_order
pcair_set_coarsest_inverse_type        = pflare_defs.pcair_set_coarsest_inverse_type
pcair_set_coarsest_poly_order          = pflare_defs.pcair_set_coarsest_poly_order
pcair_set_coarsest_inverse_sparsity_order = pflare_defs.pcair_set_coarsest_inverse_sparsity_order
pcair_set_coarsest_matrix_free_polys   = pflare_defs.pcair_set_coarsest_matrix_free_polys
pcair_set_coarsest_diag_scale_polys    = pflare_defs.pcair_set_coarsest_diag_scale_polys
pcair_set_coarsest_subcomm             = pflare_defs.pcair_set_coarsest_subcomm
pcair_set_r_drop                       = pflare_defs.pcair_set_r_drop
pcair_set_a_drop                       = pflare_defs.pcair_set_a_drop
pcair_set_a_lump                       = pflare_defs.pcair_set_a_lump
pcair_set_reuse_sparsity               = pflare_defs.pcair_set_reuse_sparsity
pcair_set_reuse_poly_coeffs            = pflare_defs.pcair_set_reuse_poly_coeffs
pcair_set_reuse_amount                 = pflare_defs.pcair_set_reuse_amount
pcair_set_poly_coeffs                  = pflare_defs.pcair_set_poly_coeffs

# -----------------------------------------------------------------------
# PCPFLAREINV Get functions
# -----------------------------------------------------------------------
pcpflareinv_get_poly_order         = pflare_defs.pcpflareinv_get_poly_order
pcpflareinv_get_sparsity_order     = pflare_defs.pcpflareinv_get_sparsity_order
pcpflareinv_get_type               = pflare_defs.pcpflareinv_get_type
pcpflareinv_get_matrix_free        = pflare_defs.pcpflareinv_get_matrix_free
pcpflareinv_get_reuse_poly_coeffs  = pflare_defs.pcpflareinv_get_reuse_poly_coeffs
pcpflareinv_get_poly_coeffs        = pflare_defs.pcpflareinv_get_poly_coeffs

# -----------------------------------------------------------------------
# PCPFLAREINV Set functions
# -----------------------------------------------------------------------
pcpflareinv_set_poly_order         = pflare_defs.pcpflareinv_set_poly_order
pcpflareinv_set_sparsity_order     = pflare_defs.pcpflareinv_set_sparsity_order
pcpflareinv_set_type               = pflare_defs.pcpflareinv_set_type
pcpflareinv_set_matrix_free        = pflare_defs.pcpflareinv_set_matrix_free
pcpflareinv_set_poly_coeffs        = pflare_defs.pcpflareinv_set_poly_coeffs
pcpflareinv_set_reuse_poly_coeffs  = pflare_defs.pcpflareinv_set_reuse_poly_coeffs
