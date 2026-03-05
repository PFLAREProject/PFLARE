import numpy as np
cimport numpy as cnp
from libc.string cimport memcpy

from petsc4py.PETSc cimport Mat, PetscMat
from petsc4py.PETSc cimport PC, PetscPC
from petsc4py.PETSc cimport IS, PetscIS

# Required for NumPy C-API initialisation; no-op in Cython 3+
cnp.import_array()

# Bring PetscInt and PetscReal from petsc.h so the C compiler resolves the
# correct widths regardless of whether PETSc was built with 32- or 64-bit
# indices, or single/double precision.
cdef extern from "petsc.h":
	ctypedef int    PetscInt  "PetscInt"
	ctypedef double PetscReal "PetscReal"

cdef extern:
	void PCRegister_PFLARE()
	void compute_cf_splitting_c(PetscMat *A, int symmetric_int, double strong_threshold, int max_luby_steps, int cf_splitting_type, int ddc_its, double fraction_swap, double max_dd_ratio, PetscIS* is_fine, PetscIS* is_coarse)

	# PCAIR - number of multigrid levels
	void PCAIRGetNumLevels_c(PetscPC *pc, PetscInt *input_int)

	# PCAIR - polynomial coefficients
	# Returns a pointer into internal PCAIR memory (valid until the next PCSetUp or PCReset).
	# The Python wrapper copies the data before returning.
	void PCAIRGetPolyCoeffs_c(PetscPC *pc, PetscInt petsc_level, int which_inverse,
	                           PetscReal **coeffs_ptr, PetscInt *row_size, PetscInt *col_size)

	# PCAIR - set polynomial coefficients (copies from the provided pointer)
	void PCAIRSetPolyCoeffs_c(PetscPC *pc, PetscInt petsc_level, int which_inverse,
	                           PetscReal *coeffs_ptr, PetscInt row_size, PetscInt col_size)

	# PCPFLAREINV - polynomial coefficients (PC passed by value, not pointer)
	# Returns a pointer into internal PCPFLAREINV memory (valid until the next PCSetUp or PCReset).
	# The Python wrapper copies the data before returning.
	int PCPFLAREINVGetPolyCoeffs(PetscPC pc, PetscReal **coeffs, PetscInt *rows, PetscInt *cols)

	# PCPFLAREINV - set polynomial coefficients (copies from the provided pointer)
	int PCPFLAREINVSetPolyCoeffs(PetscPC pc, PetscReal *coeffs, PetscInt rows, PetscInt cols)

	# PCAIR - reuse flags (must be called directly; the options-DB equivalents are
	# only processed during KSPSetFromOptions / PCSetFromOptions)
	void PCAIRSetReuseSparsity_c(PetscPC *pc, int input_bool)
	void PCAIRSetReusePolyCoeffs_c(PetscPC *pc, int input_bool)

	# PCPFLAREINV - reuse flag
	int PCPFLAREINVSetReusePolyCoeffs(PetscPC pc, int flg)


cpdef py_PCRegister_PFLARE():
	PCRegister_PFLARE()

cpdef compute_cf_splitting(Mat A, bint symmetric, double strong_threshold, int max_luby_steps, int cf_splitting_type, int ddc_its, double fraction_swap, double max_dd_ratio):
	cdef IS is_fine
	cdef IS is_coarse
	is_fine = IS()
	is_coarse = IS()
	compute_cf_splitting_c(&(A.mat), symmetric, strong_threshold, max_luby_steps, cf_splitting_type, ddc_its, fraction_swap, max_dd_ratio, &(is_fine.iset), &(is_coarse.iset))
	return is_fine, is_coarse

cpdef int pcair_get_num_levels(PC pc):
	"""Return the number of multigrid levels in a PCAIR preconditioner."""
	cdef PetscInt num_levels = 0
	PCAIRGetNumLevels_c(&(pc.pc), &num_levels)
	return <int>num_levels

cpdef cnp.ndarray pcair_get_poly_coeffs(PC pc, int petsc_level, int which_inverse):
	"""Return a copy of the GMRES polynomial coefficients at the given PCAIR level.

	Parameters
	----------
	pc : PC
	    A PCAIR preconditioner that has been set up (PCSetUp already called).
	petsc_level : int
	    PETSc level index. Use values from 1 to num_levels-1 with COEFFS_INV_AFF,
	    and 0 with COEFFS_INV_COARSE (as returned by pcair_get_num_levels).
	which_inverse : int
	    Selector constant: COEFFS_INV_AFF, COEFFS_INV_AFF_DROPPED,
	    COEFFS_INV_ACC, or COEFFS_INV_COARSE.

	Returns
	-------
	numpy.ndarray, shape (poly_order+1, 1_or_2), Fortran-contiguous
	    Column 0: real coefficients (power/Arnoldi/Neumann) or real roots (Newton).
	    Column 1: imaginary roots (Newton basis only).
	"""
	cdef PetscReal *coeffs_ptr = NULL
	cdef PetscInt row_size = 0, col_size = 0
	PCAIRGetPolyCoeffs_c(&(pc.pc), petsc_level, which_inverse,
	                      &coeffs_ptr, &row_size, &col_size)
	cdef cnp.ndarray result = np.empty((row_size, col_size), dtype=np.float64, order='F')
	memcpy(<void*>result.data, <void*>coeffs_ptr, row_size * col_size * sizeof(PetscReal))
	return result

cpdef pcair_set_poly_coeffs(PC pc, int petsc_level, int which_inverse, cnp.ndarray coeffs):
	"""Copy polynomial coefficients into the PCAIR preconditioner at the given level.

	Parameters
	----------
	pc : PC
	    A PCAIR preconditioner.
	petsc_level : int
	    PETSc level index (as used in pcair_get_poly_coeffs).
	which_inverse : int
	    Selector constant: COEFFS_INV_AFF, COEFFS_INV_AFF_DROPPED,
	    COEFFS_INV_ACC, or COEFFS_INV_COARSE.
	coeffs : numpy.ndarray
	    Coefficient array as returned by pcair_get_poly_coeffs.
	    Must have shape (poly_order+1, 1_or_2).
	"""
	cdef cnp.ndarray coeffs_f = np.asfortranarray(coeffs, dtype=np.float64)
	cdef PetscInt row_size = <PetscInt>coeffs_f.shape[0]
	cdef PetscInt col_size = <PetscInt>coeffs_f.shape[1]
	PCAIRSetPolyCoeffs_c(&(pc.pc), petsc_level, which_inverse,
	                      <PetscReal*>coeffs_f.data, row_size, col_size)

cpdef cnp.ndarray pcpflareinv_get_poly_coeffs(PC pc):
	"""Return a copy of the GMRES polynomial coefficients from a PCPFLAREINV preconditioner.

	Returns
	-------
	numpy.ndarray, shape (poly_order+1, 1_or_2), Fortran-contiguous
	    Column 0: real coefficients (power/Arnoldi/Neumann) or real roots (Newton).
	    Column 1: imaginary roots (Newton basis only).
	"""
	cdef PetscReal *coeffs_ptr = NULL
	cdef PetscInt rows = 0, cols = 0
	PCPFLAREINVGetPolyCoeffs(pc.pc, &coeffs_ptr, &rows, &cols)
	cdef cnp.ndarray result = np.empty((rows, cols), dtype=np.float64, order='F')
	memcpy(<void*>result.data, <void*>coeffs_ptr, rows * cols * sizeof(PetscReal))
	return result

cpdef pcpflareinv_set_poly_coeffs(PC pc, cnp.ndarray coeffs):
	"""Copy polynomial coefficients into the PCPFLAREINV preconditioner.

	Parameters
	----------
	pc : PC
	    A PCPFLAREINV preconditioner.
	coeffs : numpy.ndarray
	    Coefficient array as returned by pcpflareinv_get_poly_coeffs.
	    Must have shape (poly_order+1, 1_or_2).
	"""
	cdef cnp.ndarray coeffs_f = np.asfortranarray(coeffs, dtype=np.float64)
	cdef PetscInt rows = <PetscInt>coeffs_f.shape[0]
	cdef PetscInt cols = <PetscInt>coeffs_f.shape[1]
	PCPFLAREINVSetPolyCoeffs(pc.pc, <PetscReal*>coeffs_f.data, rows, cols)

cpdef pcair_set_reuse_sparsity(PC pc, bint flag):
	"""Tell PCAIR to reuse sparsity (CF splitting and matrix structure) on the next setup.

	Must be called before KSPSolve to take effect.
	"""
	PCAIRSetReuseSparsity_c(&(pc.pc), <int>flag)

cpdef pcair_set_reuse_poly_coeffs(PC pc, bint flag):
	"""Tell PCAIR to reuse the current polynomial coefficients on the next setup.

	Must be called before KSPSolve, after pcair_set_poly_coeffs, to take effect.
	"""
	PCAIRSetReusePolyCoeffs_c(&(pc.pc), <int>flag)

cpdef pcpflareinv_set_reuse_poly_coeffs(PC pc, bint flag):
	"""Tell PCPFLAREINV to reuse the current polynomial coefficients on the next setup.

	Must be called before KSPSolve, after pcpflareinv_set_poly_coeffs, to take effect.
	"""
	PCPFLAREINVSetReusePolyCoeffs(pc.pc, <int>flag)
