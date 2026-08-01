# Just import the pflare definitions
import pflare_defs
# And register all the types with PETSc
pflare_defs.py_PCRegister_PFLARE()

from enum import IntEnum

# The enum values match the C enums in include/pflare.h; the members are also
# aliased at module level (e.g. pflare.CF_PMISR_DDC) for convenience

class CFSplittingType(IntEnum):
    """CF splitting algorithms for PCAIR (CFSplittingType in C)"""
    CF_PMISR_DDC  = 0
    CF_DIAG_DOM   = 1
    CF_PMIS       = 2
    CF_PMIS_DIST2 = 3
    CF_AGG        = 4
    CF_PMIS_AGG   = 5
    CF_CR         = 6

class PCPFLAREINVType(IntEnum):
    """Approximate inverse types for PCPFLAREINV and the PCAIR smoothers (PCPFLAREINVType in C)"""
    PFLAREINV_POWER           = 0
    PFLAREINV_ARNOLDI         = 1
    PFLAREINV_NEWTON          = 2
    PFLAREINV_NEWTON_NO_EXTRA = 3
    PFLAREINV_NEUMANN         = 4
    PFLAREINV_SAI             = 5
    PFLAREINV_ISAI            = 6
    PFLAREINV_WJACOBI         = 7
    PFLAREINV_JACOBI          = 8

class PCAIRZType(IntEnum):
    """Z / restrictor types for PCAIR (PCAIRZType in C)"""
    AIR_Z_PRODUCT  = 0
    AIR_Z_LAIR     = 1
    AIR_Z_LAIR_SAI = 2

class WhichInverseType(IntEnum):
    """Selectors for pcair_get_poly_coeffs / pcair_set_poly_coeffs (WhichInverseType in C)"""
    COEFFS_INV_AFF         = 0  # Inverse of the fine-fine block A_ff
    COEFFS_INV_AFF_DROPPED = 1  # Inverse of the dropped fine-fine block
    COEFFS_INV_ACC         = 2  # Inverse of the coarse-coarse block A_cc
    COEFFS_INV_COARSE      = 3  # Inverse on the coarsest grid

# Alias the enum members at module level for backwards compatibility
for _enum_class in (CFSplittingType, PCPFLAREINVType, PCAIRZType, WhichInverseType):
    for _member in _enum_class:
        globals()[_member.name] = _member
del _enum_class, _member

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
pcair_get_grid_complexity              = pflare_defs.pcair_get_grid_complexity
pcair_get_operator_complexity          = pflare_defs.pcair_get_operator_complexity
pcair_get_cycle_complexity             = pflare_defs.pcair_get_cycle_complexity
pcair_get_storage_complexity           = pflare_defs.pcair_get_storage_complexity
pcair_get_reuse_storage_complexity     = pflare_defs.pcair_get_reuse_storage_complexity
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
pcpflareinv_get_inverse_mat        = pflare_defs.pcpflareinv_get_inverse_mat

# -----------------------------------------------------------------------
# PCPFLAREINV Set functions
# -----------------------------------------------------------------------
pcpflareinv_set_poly_order         = pflare_defs.pcpflareinv_set_poly_order
pcpflareinv_set_sparsity_order     = pflare_defs.pcpflareinv_set_sparsity_order
pcpflareinv_set_type               = pflare_defs.pcpflareinv_set_type
pcpflareinv_set_matrix_free        = pflare_defs.pcpflareinv_set_matrix_free
pcpflareinv_set_poly_coeffs        = pflare_defs.pcpflareinv_set_poly_coeffs
pcpflareinv_set_reuse_poly_coeffs  = pflare_defs.pcpflareinv_set_reuse_poly_coeffs

# -----------------------------------------------------------------------
# Docstrings
# The descriptions below match docs/options.md and are attached to the
# functions above so that help() works on them
# -----------------------------------------------------------------------

# Get/set pairs: option suffix -> (description, command line option, default)
_PCAIR_OPTION_DOCS = {
    'print_stats_timings':          ("whether to print out statistics about the multigrid hierarchy and timings", "-pc_air_print_stats_timings", "False"),
    'max_levels':                   ("the maximum number of levels in the hierarchy", "-pc_air_max_levels", "300"),
    'coarse_eq_limit':              ("the minimum number of global unknowns on the coarse grid", "-pc_air_coarse_eq_limit", "6"),
    'auto_truncate_start_level':    ("the level from which a coarse solver is built and tested to determine if the hierarchy can be truncated", "-pc_air_auto_truncate_start_level", "-1"),
    'auto_truncate_tol':            ("the tolerance used to determine if the coarse solver is good enough to truncate at a given level", "-pc_air_auto_truncate_tol", "1e-14"),
    'r_drop':                       ("the drop tolerance applied to R on each level after it is built", "-pc_air_r_drop", "0.01"),
    'a_drop':                       ("the drop tolerance applied to the coarse matrix on each level after it is built", "-pc_air_a_drop", "0.0001"),
    'a_lump':                       ("whether to lump to the diagonal rather than drop for the coarse matrix", "-pc_air_a_lump", "False"),
    'processor_agglom':             ("whether to use a graph partitioner to repartition the coarse grids and reduce the number of active MPI ranks", "-pc_air_processor_agglom", "True"),
    'processor_agglom_ratio':       ("the local to non-local nnzs ratio that triggers processor agglomeration on all levels", "-pc_air_processor_agglom_ratio", "2.0"),
    'processor_agglom_factor':      ("the factor by which the number of active MPI ranks is reduced each time processor agglomeration is triggered", "-pc_air_processor_agglom_factor", "2"),
    'process_eq_limit':             ("the average number of equations per rank below which processor agglomeration is triggered", "-pc_air_process_eq_limit", "50"),
    'subcomm':                      ("whether to exclude MPI ranks with no non-zeros from reductions by moving onto a subcommunicator after processor agglomeration", "-pc_air_subcomm", "False"),
    'cf_splitting_type':            ("the type of CF splitting to use (a CFSplittingType)", "-pc_air_cf_splitting_type", "CF_PMISR_DDC"),
    'strong_threshold':             ("the strong threshold used in the CF splitting", "-pc_air_strong_threshold", "0.5"),
    'max_luby_steps':               ("the maximum number of Luby steps in the CF splitting; if negative, as many steps as necessary", "-pc_air_max_luby_steps", "-1"),
    'ddc_its':                      ("the number of DDC iterations in the pmisr_ddc CF splitting", "-pc_air_ddc_its", "1"),
    'ddc_fraction':                 ("the local fraction of F points converted to C points by diagonal dominance in the pmisr_ddc CF splitting", "-pc_air_ddc_fraction", "0.1"),
    'inverse_type':                 ("the F-point approximate inverse type (a PCPFLAREINVType)", "-pc_air_inverse_type", "PFLAREINV_ARNOLDI"),
    'poly_order':                   ("the order of the polynomial if using a polynomial inverse type", "-pc_air_poly_order", "6"),
    'inverse_sparsity_order':       ("the power of A used as the sparsity in assembled inverses", "-pc_air_inverse_sparsity_order", "1"),
    'diag_scale_polys':             ("whether to diagonally scale before computing a polynomial inverse", "-pc_air_diag_scale_polys", "False"),
    'matrix_free_polys':            ("whether to do smoothing matrix-free if possible", "-pc_air_matrix_free_polys", "False"),
    'smooth_type':                  ("the type and number of smooths as a string of f and c characters (e.g. 'ff', 'fc', 'fcf')", "-pc_air_smooth_type", "'ff'"),
    'full_smoothing_up_and_down':   ("whether to smooth up and down on all points at once, rather than only down F and C smoothing", "-pc_air_full_smoothing_up_and_down", "False"),
    'c_inverse_type':               ("the approximate inverse type for the C smooth (a PCPFLAREINVType); defaults to the F point smoother type", "-pc_air_c_inverse_type", "inverse_type"),
    'c_poly_order':                 ("the polynomial order for the C smooth; defaults to the F point smoother order", "-pc_air_c_poly_order", "poly_order"),
    'c_inverse_sparsity_order':     ("the power of A used as the sparsity in assembled inverses for the C smooth; defaults to the F point smoother order", "-pc_air_c_inverse_sparsity_order", "inverse_sparsity_order"),
    'one_point_classical_prolong':  ("whether to use a one-point classical prolongator, instead of an approximate ideal prolongator", "-pc_air_one_point_classical_prolong", "True"),
    'symmetric':                    ("whether the prolongator is defined as R^T", "-pc_air_symmetric", "False"),
    'strong_r_threshold':           ("the threshold to drop when forming the grid-transfer operators", "-pc_air_strong_r_threshold", "0.0"),
    'z_type':                       ("the type of grid-transfer operator (a PCAIRZType)", "-pc_air_z_type", "AIR_Z_PRODUCT"),
    'lair_distance':                ("the distance of the grid-transfer operators if the Z type is lair or lair_sai", "-pc_air_lair_distance", "2"),
    'constrain_w':                  ("whether to apply constraints to the prolongator", "-pc_air_constrain_w", "False"),
    'constrain_z':                  ("whether to apply constraints to the restrictor", "-pc_air_constrain_z", "False"),
    'improve_w_its':                ("the number of Richardson iterations applied to improve the approximate prolongator", "-pc_air_improve_w_its", "0"),
    'improve_z_its':                ("the number of Richardson iterations applied to improve the approximate restrictor", "-pc_air_improve_z_its", "0"),
    'coarsest_inverse_type':        ("the coarse grid approximate inverse type (a PCPFLAREINVType)", "-pc_air_coarsest_inverse_type", "PFLAREINV_ARNOLDI"),
    'coarsest_poly_order':          ("the coarse grid polynomial order", "-pc_air_coarsest_poly_order", "6"),
    'coarsest_inverse_sparsity_order': ("the coarse grid sparsity order", "-pc_air_coarsest_inverse_sparsity_order", "1"),
    'coarsest_matrix_free_polys':   ("whether to do smoothing matrix-free if possible on the coarse grid", "-pc_air_coarsest_matrix_free_polys", "False"),
    'coarsest_diag_scale_polys':    ("whether to diagonally scale on the coarse grid before computing a polynomial inverse", "-pc_air_coarsest_diag_scale_polys", "False"),
    'coarsest_subcomm':             ("whether to use a subcommunicator on the coarse grid", "-pc_air_coarsest_subcomm", "False"),
    'reuse_sparsity':               ("whether to store temporary data to allow fast setup with reuse", "-pc_air_reuse_sparsity", "False"),
    'reuse_amount':                 ("how much data is stored when reuse sparsity is enabled: 1=CF splitting only, 2=CF splitting + SpGEMM sparsity, 3=everything", "-pc_air_reuse_amount", "3"),
    'reuse_poly_coeffs':            ("whether to skip recomputing the polynomial inverse coefficients during setup with reuse", "-pc_air_reuse_poly_coeffs", "False"),
    'poly_coeffs':                  ("the polynomial inverse coefficients for a given inverse (a WhichInverseType selector)", "N/A", "N/A"),
}

_PCPFLAREINV_OPTION_DOCS = {
    'type':              ("the approximate inverse type (a PCPFLAREINVType)", "-pc_pflareinv_type", "PFLAREINV_ARNOLDI"),
    'poly_order':        ("the order of the polynomial if using a polynomial inverse type", "-pc_pflareinv_poly_order", "6"),
    'sparsity_order':    ("the power of A used as the sparsity in assembled inverses", "-pc_pflareinv_sparsity_order", "1"),
    'matrix_free':       ("whether the inverse is applied matrix-free, or an assembled matrix is built and used", "-pc_pflareinv_matrix_free", "False"),
    'reuse_poly_coeffs': ("whether to skip recomputing the polynomial inverse coefficients during setup with reuse", "-pc_pflareinv_reuse_poly_coeffs", "False"),
    'poly_coeffs':       ("the polynomial inverse coefficients", "N/A", "N/A"),
}

# Get-only routines: function name -> docstring
_GET_ONLY_DOCS = {
    'pcair_get_num_levels':               "Return the number of levels in the built PCAIR hierarchy.",
    'pcair_get_grid_complexity':          "Return the grid complexity of the PCAIR hierarchy after setup; returns -1 if not yet set up.",
    'pcair_get_operator_complexity':      "Return the operator complexity of the PCAIR hierarchy after setup; returns -1 if not yet set up.",
    'pcair_get_cycle_complexity':         "Return the cycle complexity of the PCAIR hierarchy after setup; returns -1 if not yet set up.",
    'pcair_get_storage_complexity':       "Return the storage complexity of the PCAIR hierarchy after setup; returns -1 if not yet set up.",
    'pcair_get_reuse_storage_complexity': "Return the reuse storage complexity of the PCAIR hierarchy after setup (0 when reuse is disabled); returns -1 if not yet set up.",
    'pcpflareinv_get_inverse_mat':        "Return the assembled approximate inverse matrix from PCPFLAREINV.",
    'compute_cf_splitting':               "Compute a CF splitting of a matrix, returning the fine and coarse index sets.",
    'compute_diag_dom_submatrix':         "Extract a diagonally dominant submatrix from a matrix.",
}

def _attach_docstrings():
    for _prefix, _docs in (('pcair', _PCAIR_OPTION_DOCS), ('pcpflareinv', _PCPFLAREINV_OPTION_DOCS)):
        for _name, (_desc, _opt, _default) in _docs.items():
            _extra = "" if _opt == "N/A" else " Command line option: %s (default %s)." % (_opt, _default)
            _getter = globals().get('%s_get_%s' % (_prefix, _name))
            if _getter is not None:
                _getter.__doc__ = "Return %s.%s" % (_desc, _extra)
            _setter = globals().get('%s_set_%s' % (_prefix, _name))
            if _setter is not None:
                _setter.__doc__ = "Set %s.%s" % (_desc, _extra)
    for _name, _doc in _GET_ONLY_DOCS.items():
        globals()[_name].__doc__ = _doc

_attach_docstrings()
