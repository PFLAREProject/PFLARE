# Just import the pflare definitions
import pflare_defs
# And register all the types with PETSc
pflare_defs.py_PCRegister_PFLARE()
# Define CF splitting variables
CF_PMISR_DDC = 0
CF_PMIS=1
CF_PMIS_DIST2=2
CF_AGG=3
CF_PMIS_AGG=4

# Selector constants for PCAIRGetPolyCoeffs / PCAIRSetPolyCoeffs
# These match the Fortran COEFFS_INV_* parameters in pflare_parameters
COEFFS_INV_AFF         = 0  # Inverse of the fine-fine block A_ff
COEFFS_INV_AFF_DROPPED = 1  # Inverse of the dropped fine-fine block
COEFFS_INV_ACC         = 2  # Inverse of the coarse-coarse block A_cc
COEFFS_INV_COARSE      = 3  # Inverse on the coarsest grid

# PCAIR polynomial coefficient functions
pcair_get_num_levels    = pflare_defs.pcair_get_num_levels
pcair_get_poly_coeffs   = pflare_defs.pcair_get_poly_coeffs
pcair_set_poly_coeffs   = pflare_defs.pcair_set_poly_coeffs

# PCPFLAREINV polynomial coefficient functions
pcpflareinv_get_poly_coeffs = pflare_defs.pcpflareinv_get_poly_coeffs
pcpflareinv_set_poly_coeffs = pflare_defs.pcpflareinv_set_poly_coeffs
