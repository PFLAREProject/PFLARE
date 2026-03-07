import numpy as np
cimport numpy as cnp
from libc.string cimport memcpy, strlen

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

	# -----------------------------------------------------------------------
	# PCAIR Get routines
	# -----------------------------------------------------------------------

	# PCAIR - number of multigrid levels
	void PCAIRGetNumLevels_c(PetscPC *pc, PetscInt *input_int)

	void PCAIRGetPrintStatsTimings_c(PetscPC *pc, unsigned char *print_stats)
	void PCAIRGetMaxLevels_c(PetscPC *pc, PetscInt *max_levels)
	void PCAIRGetCoarseEqLimit_c(PetscPC *pc, PetscInt *coarse_eq_limit)
	void PCAIRGetAutoTruncateStartLevel_c(PetscPC *pc, PetscInt *start_level)
	void PCAIRGetAutoTruncateTol_c(PetscPC *pc, PetscReal *tol)
	void PCAIRGetProcessorAgglom_c(PetscPC *pc, unsigned char *processor_agglom)
	void PCAIRGetProcessorAgglomRatio_c(PetscPC *pc, PetscReal *ratio)
	void PCAIRGetProcessorAgglomFactor_c(PetscPC *pc, PetscInt *factor)
	void PCAIRGetProcessEqLimit_c(PetscPC *pc, PetscInt *limit)
	void PCAIRGetSubcomm_c(PetscPC *pc, unsigned char *subcomm)
	void PCAIRGetStrongThreshold_c(PetscPC *pc, PetscReal *thresh)
	void PCAIRGetDDCIts_c(PetscPC *pc, PetscInt *its)
	void PCAIRGetMaxDDRatio_c(PetscPC *pc, PetscReal *ratio)
	void PCAIRGetDDCFraction_c(PetscPC *pc, PetscReal *frac)
	void PCAIRGetCFSplittingType_c(PetscPC *pc, int *algo)
	void PCAIRGetMaxLubySteps_c(PetscPC *pc, PetscInt *steps)
	void PCAIRGetDiagScalePolys_c(PetscPC *pc, unsigned char *scale)
	void PCAIRGetMatrixFreePolys_c(PetscPC *pc, unsigned char *mf)
	void PCAIRGetOnePointClassicalProlong_c(PetscPC *pc, unsigned char *onep)
	void PCAIRGetFullSmoothingUpAndDown_c(PetscPC *pc, unsigned char *full)
	void PCAIRGetSymmetric_c(PetscPC *pc, unsigned char *sym)
	void PCAIRGetConstrainW_c(PetscPC *pc, unsigned char *constrain)
	void PCAIRGetConstrainZ_c(PetscPC *pc, unsigned char *constrain)
	void PCAIRGetImproveWIts_c(PetscPC *pc, PetscInt *its)
	void PCAIRGetImproveZIts_c(PetscPC *pc, PetscInt *its)
	void PCAIRGetStrongRThreshold_c(PetscPC *pc, PetscReal *thresh)
	void PCAIRGetInverseType_c(PetscPC *pc, int *inv_type)
	void PCAIRGetCInverseType_c(PetscPC *pc, int *inv_type)
	void PCAIRGetZType_c(PetscPC *pc, int *z_type)
	void PCAIRGetLairDistance_c(PetscPC *pc, PetscInt *distance)
	void PCAIRGetPolyOrder_c(PetscPC *pc, PetscInt *order)
	void PCAIRGetInverseSparsityOrder_c(PetscPC *pc, PetscInt *order)
	void PCAIRGetCPolyOrder_c(PetscPC *pc, PetscInt *order)
	void PCAIRGetCInverseSparsityOrder_c(PetscPC *pc, PetscInt *order)
	void PCAIRGetCoarsestInverseType_c(PetscPC *pc, int *inv_type)
	void PCAIRGetCoarsestPolyOrder_c(PetscPC *pc, PetscInt *order)
	void PCAIRGetCoarsestInverseSparsityOrder_c(PetscPC *pc, PetscInt *order)
	void PCAIRGetCoarsestMatrixFreePolys_c(PetscPC *pc, unsigned char *mf)
	void PCAIRGetCoarsestDiagScalePolys_c(PetscPC *pc, unsigned char *scale)
	void PCAIRGetCoarsestSubcomm_c(PetscPC *pc, unsigned char *subcomm)
	void PCAIRGetRDrop_c(PetscPC *pc, PetscReal *rdrop)
	void PCAIRGetADrop_c(PetscPC *pc, PetscReal *adrop)
	void PCAIRGetALump_c(PetscPC *pc, unsigned char *lump)
	void PCAIRGetReuseSparsity_c(PetscPC *pc, unsigned char *reuse)
	void PCAIRGetReusePolyCoeffs_c(PetscPC *pc, unsigned char *reuse)
	void PCAIRGetReuseAmount_c(PetscPC *pc, PetscInt *amount)
	void PCAIRGetSmoothType_c(PetscPC *pc, char *output_string)

	# PCAIR - polynomial coefficients
	# Returns a pointer into internal PCAIR memory (valid until the next PCSetUp or PCReset).
	# The Python wrapper copies the data before returning.
	void PCAIRGetPolyCoeffs_c(PetscPC *pc, PetscInt petsc_level, int which_inverse,
	                           PetscReal **coeffs_ptr, PetscInt *row_size, PetscInt *col_size)

	# -----------------------------------------------------------------------
	# PCAIR Set routines
	# -----------------------------------------------------------------------

	void PCAIRSetPrintStatsTimings_c(PetscPC *pc, int print_stats)
	void PCAIRSetMaxLevels_c(PetscPC *pc, PetscInt max_levels)
	void PCAIRSetCoarseEqLimit_c(PetscPC *pc, PetscInt coarse_eq_limit)
	void PCAIRSetAutoTruncateStartLevel_c(PetscPC *pc, PetscInt start_level)
	void PCAIRSetAutoTruncateTol_c(PetscPC *pc, PetscReal tol)
	void PCAIRSetProcessorAgglom_c(PetscPC *pc, int processor_agglom)
	void PCAIRSetProcessorAgglomRatio_c(PetscPC *pc, PetscReal ratio)
	void PCAIRSetProcessorAgglomFactor_c(PetscPC *pc, PetscInt factor)
	void PCAIRSetProcessEqLimit_c(PetscPC *pc, PetscInt limit)
	void PCAIRSetSubcomm_c(PetscPC *pc, int subcomm)
	void PCAIRSetStrongThreshold_c(PetscPC *pc, PetscReal thresh)
	void PCAIRSetDDCIts_c(PetscPC *pc, PetscInt its)
	void PCAIRSetMaxDDRatio_c(PetscPC *pc, PetscReal ratio)
	void PCAIRSetDDCFraction_c(PetscPC *pc, PetscReal frac)
	void PCAIRSetCFSplittingType_c(PetscPC *pc, int algo)
	void PCAIRSetMaxLubySteps_c(PetscPC *pc, PetscInt steps)
	void PCAIRSetSmoothType_c(PetscPC *pc, char *input_string)
	void PCAIRSetDiagScalePolys_c(PetscPC *pc, int scale)
	void PCAIRSetMatrixFreePolys_c(PetscPC *pc, int mf)
	void PCAIRSetOnePointClassicalProlong_c(PetscPC *pc, int onep)
	void PCAIRSetFullSmoothingUpAndDown_c(PetscPC *pc, int full)
	void PCAIRSetSymmetric_c(PetscPC *pc, int sym)
	void PCAIRSetConstrainW_c(PetscPC *pc, int constrain)
	void PCAIRSetConstrainZ_c(PetscPC *pc, int constrain)
	void PCAIRSetImproveWIts_c(PetscPC *pc, PetscInt its)
	void PCAIRSetImproveZIts_c(PetscPC *pc, PetscInt its)
	void PCAIRSetStrongRThreshold_c(PetscPC *pc, PetscReal thresh)
	void PCAIRSetInverseType_c(PetscPC *pc, int inv_type)
	void PCAIRSetCInverseType_c(PetscPC *pc, int inv_type)
	void PCAIRSetZType_c(PetscPC *pc, int z_type)
	void PCAIRSetLairDistance_c(PetscPC *pc, PetscInt distance)
	void PCAIRSetPolyOrder_c(PetscPC *pc, PetscInt order)
	void PCAIRSetInverseSparsityOrder_c(PetscPC *pc, PetscInt order)
	void PCAIRSetCPolyOrder_c(PetscPC *pc, PetscInt order)
	void PCAIRSetCInverseSparsityOrder_c(PetscPC *pc, PetscInt order)
	void PCAIRSetCoarsestInverseType_c(PetscPC *pc, int inv_type)
	void PCAIRSetCoarsestPolyOrder_c(PetscPC *pc, PetscInt order)
	void PCAIRSetCoarsestInverseSparsityOrder_c(PetscPC *pc, PetscInt order)
	void PCAIRSetCoarsestMatrixFreePolys_c(PetscPC *pc, int mf)
	void PCAIRSetCoarsestDiagScalePolys_c(PetscPC *pc, int scale)
	void PCAIRSetCoarsestSubcomm_c(PetscPC *pc, int subcomm)
	void PCAIRSetRDrop_c(PetscPC *pc, PetscReal rdrop)
	void PCAIRSetADrop_c(PetscPC *pc, PetscReal adrop)
	void PCAIRSetALump_c(PetscPC *pc, int lump)

	# PCAIR - reuse flags
	void PCAIRSetReuseSparsity_c(PetscPC *pc, int input_bool)
	void PCAIRSetReusePolyCoeffs_c(PetscPC *pc, int input_bool)
	void PCAIRSetReuseAmount_c(PetscPC *pc, PetscInt amount)

	# PCAIR - set polynomial coefficients (copies from the provided pointer)
	void PCAIRSetPolyCoeffs_c(PetscPC *pc, PetscInt petsc_level, int which_inverse,
	                           PetscReal *coeffs_ptr, PetscInt row_size, PetscInt col_size)

	# PCPFLAREINV - Get routines (PC passed by value, not pointer)
	int PCPFLAREINVGetPolyOrder(PetscPC pc, PetscInt *order)
	int PCPFLAREINVGetSparsityOrder(PetscPC pc, PetscInt *order)
	int PCPFLAREINVGetType(PetscPC pc, int *pflare_type)
	int PCPFLAREINVGetMatrixFree(PetscPC pc, int *flag)
	int PCPFLAREINVGetReusePolyCoeffs(PetscPC pc, int *flag)

	# PCPFLAREINV - polynomial coefficients (PC passed by value, not pointer)
	# Returns a pointer into internal PCPFLAREINV memory (valid until the next PCSetUp or PCReset).
	# The Python wrapper copies the data before returning.
	int PCPFLAREINVGetPolyCoeffs(PetscPC pc, PetscReal **coeffs, PetscInt *rows, PetscInt *cols)

	# PCPFLAREINV - Set routines (PC passed by value, not pointer)
	int PCPFLAREINVSetPolyOrder(PetscPC pc, PetscInt order)
	int PCPFLAREINVSetSparsityOrder(PetscPC pc, PetscInt order)
	int PCPFLAREINVSetType(PetscPC pc, int pflare_type)
	int PCPFLAREINVSetMatrixFree(PetscPC pc, int flag)

	# PCPFLAREINV - set polynomial coefficients (copies from the provided pointer)
	int PCPFLAREINVSetPolyCoeffs(PetscPC pc, PetscReal *coeffs, PetscInt rows, PetscInt cols)

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

# -----------------------------------------------------------------------
# PCAIR Get wrappers
# -----------------------------------------------------------------------

cpdef int pcair_get_num_levels(PC pc):
	"""Return the number of multigrid levels in a PCAIR preconditioner."""
	cdef PetscInt num_levels = 0
	PCAIRGetNumLevels_c(&(pc.pc), &num_levels)
	return <int>num_levels

cpdef bint pcair_get_print_stats_timings(PC pc):
	cdef unsigned char result = 0
	PCAIRGetPrintStatsTimings_c(&(pc.pc), &result)
	return bool(result)

cpdef int pcair_get_max_levels(PC pc):
	cdef PetscInt result = 0
	PCAIRGetMaxLevels_c(&(pc.pc), &result)
	return <int>result

cpdef int pcair_get_coarse_eq_limit(PC pc):
	cdef PetscInt result = 0
	PCAIRGetCoarseEqLimit_c(&(pc.pc), &result)
	return <int>result

cpdef int pcair_get_auto_truncate_start_level(PC pc):
	cdef PetscInt result = 0
	PCAIRGetAutoTruncateStartLevel_c(&(pc.pc), &result)
	return <int>result

cpdef double pcair_get_auto_truncate_tol(PC pc):
	cdef PetscReal result = 0.0
	PCAIRGetAutoTruncateTol_c(&(pc.pc), &result)
	return <double>result

cpdef bint pcair_get_processor_agglom(PC pc):
	cdef unsigned char result = 0
	PCAIRGetProcessorAgglom_c(&(pc.pc), &result)
	return bool(result)

cpdef double pcair_get_processor_agglom_ratio(PC pc):
	cdef PetscReal result = 0.0
	PCAIRGetProcessorAgglomRatio_c(&(pc.pc), &result)
	return <double>result

cpdef int pcair_get_processor_agglom_factor(PC pc):
	cdef PetscInt result = 0
	PCAIRGetProcessorAgglomFactor_c(&(pc.pc), &result)
	return <int>result

cpdef int pcair_get_process_eq_limit(PC pc):
	cdef PetscInt result = 0
	PCAIRGetProcessEqLimit_c(&(pc.pc), &result)
	return <int>result

cpdef bint pcair_get_subcomm(PC pc):
	cdef unsigned char result = 0
	PCAIRGetSubcomm_c(&(pc.pc), &result)
	return bool(result)

cpdef double pcair_get_strong_threshold(PC pc):
	cdef PetscReal result = 0.0
	PCAIRGetStrongThreshold_c(&(pc.pc), &result)
	return <double>result

cpdef int pcair_get_ddc_its(PC pc):
	cdef PetscInt result = 0
	PCAIRGetDDCIts_c(&(pc.pc), &result)
	return <int>result

cpdef double pcair_get_max_dd_ratio(PC pc):
	cdef PetscReal result = 0.0
	PCAIRGetMaxDDRatio_c(&(pc.pc), &result)
	return <double>result

cpdef double pcair_get_ddc_fraction(PC pc):
	cdef PetscReal result = 0.0
	PCAIRGetDDCFraction_c(&(pc.pc), &result)
	return <double>result

cpdef int pcair_get_cf_splitting_type(PC pc):
	cdef int result = 0
	PCAIRGetCFSplittingType_c(&(pc.pc), &result)
	return result

cpdef int pcair_get_max_luby_steps(PC pc):
	cdef PetscInt result = 0
	PCAIRGetMaxLubySteps_c(&(pc.pc), &result)
	return <int>result

cpdef bint pcair_get_diag_scale_polys(PC pc):
	cdef unsigned char result = 0
	PCAIRGetDiagScalePolys_c(&(pc.pc), &result)
	return bool(result)

cpdef bint pcair_get_matrix_free_polys(PC pc):
	cdef unsigned char result = 0
	PCAIRGetMatrixFreePolys_c(&(pc.pc), &result)
	return bool(result)

cpdef bint pcair_get_one_point_classical_prolong(PC pc):
	cdef unsigned char result = 0
	PCAIRGetOnePointClassicalProlong_c(&(pc.pc), &result)
	return bool(result)

cpdef bint pcair_get_full_smoothing_up_and_down(PC pc):
	cdef unsigned char result = 0
	PCAIRGetFullSmoothingUpAndDown_c(&(pc.pc), &result)
	return bool(result)

cpdef bint pcair_get_symmetric(PC pc):
	cdef unsigned char result = 0
	PCAIRGetSymmetric_c(&(pc.pc), &result)
	return bool(result)

cpdef bint pcair_get_constrain_w(PC pc):
	cdef unsigned char result = 0
	PCAIRGetConstrainW_c(&(pc.pc), &result)
	return bool(result)

cpdef bint pcair_get_constrain_z(PC pc):
	cdef unsigned char result = 0
	PCAIRGetConstrainZ_c(&(pc.pc), &result)
	return bool(result)

cpdef int pcair_get_improve_w_its(PC pc):
	cdef PetscInt result = 0
	PCAIRGetImproveWIts_c(&(pc.pc), &result)
	return <int>result

cpdef int pcair_get_improve_z_its(PC pc):
	cdef PetscInt result = 0
	PCAIRGetImproveZIts_c(&(pc.pc), &result)
	return <int>result

cpdef double pcair_get_strong_r_threshold(PC pc):
	cdef PetscReal result = 0.0
	PCAIRGetStrongRThreshold_c(&(pc.pc), &result)
	return <double>result

cpdef int pcair_get_inverse_type(PC pc):
	cdef int result = 0
	PCAIRGetInverseType_c(&(pc.pc), &result)
	return result

cpdef int pcair_get_c_inverse_type(PC pc):
	cdef int result = 0
	PCAIRGetCInverseType_c(&(pc.pc), &result)
	return result

cpdef int pcair_get_z_type(PC pc):
	cdef int result = 0
	PCAIRGetZType_c(&(pc.pc), &result)
	return result

cpdef int pcair_get_lair_distance(PC pc):
	cdef PetscInt result = 0
	PCAIRGetLairDistance_c(&(pc.pc), &result)
	return <int>result

cpdef int pcair_get_poly_order(PC pc):
	cdef PetscInt result = 0
	PCAIRGetPolyOrder_c(&(pc.pc), &result)
	return <int>result

cpdef int pcair_get_inverse_sparsity_order(PC pc):
	cdef PetscInt result = 0
	PCAIRGetInverseSparsityOrder_c(&(pc.pc), &result)
	return <int>result

cpdef int pcair_get_c_poly_order(PC pc):
	cdef PetscInt result = 0
	PCAIRGetCPolyOrder_c(&(pc.pc), &result)
	return <int>result

cpdef int pcair_get_c_inverse_sparsity_order(PC pc):
	cdef PetscInt result = 0
	PCAIRGetCInverseSparsityOrder_c(&(pc.pc), &result)
	return <int>result

cpdef int pcair_get_coarsest_inverse_type(PC pc):
	cdef int result = 0
	PCAIRGetCoarsestInverseType_c(&(pc.pc), &result)
	return result

cpdef int pcair_get_coarsest_poly_order(PC pc):
	cdef PetscInt result = 0
	PCAIRGetCoarsestPolyOrder_c(&(pc.pc), &result)
	return <int>result

cpdef int pcair_get_coarsest_inverse_sparsity_order(PC pc):
	cdef PetscInt result = 0
	PCAIRGetCoarsestInverseSparsityOrder_c(&(pc.pc), &result)
	return <int>result

cpdef bint pcair_get_coarsest_matrix_free_polys(PC pc):
	cdef unsigned char result = 0
	PCAIRGetCoarsestMatrixFreePolys_c(&(pc.pc), &result)
	return bool(result)

cpdef bint pcair_get_coarsest_diag_scale_polys(PC pc):
	cdef unsigned char result = 0
	PCAIRGetCoarsestDiagScalePolys_c(&(pc.pc), &result)
	return bool(result)

cpdef bint pcair_get_coarsest_subcomm(PC pc):
	cdef unsigned char result = 0
	PCAIRGetCoarsestSubcomm_c(&(pc.pc), &result)
	return bool(result)

cpdef double pcair_get_r_drop(PC pc):
	cdef PetscReal result = 0.0
	PCAIRGetRDrop_c(&(pc.pc), &result)
	return <double>result

cpdef double pcair_get_a_drop(PC pc):
	cdef PetscReal result = 0.0
	PCAIRGetADrop_c(&(pc.pc), &result)
	return <double>result

cpdef bint pcair_get_a_lump(PC pc):
	cdef unsigned char result = 0
	PCAIRGetALump_c(&(pc.pc), &result)
	return bool(result)

cpdef bint pcair_get_reuse_sparsity(PC pc):
	cdef unsigned char result = 0
	PCAIRGetReuseSparsity_c(&(pc.pc), &result)
	return bool(result)

cpdef bint pcair_get_reuse_poly_coeffs(PC pc):
	cdef unsigned char result = 0
	PCAIRGetReusePolyCoeffs_c(&(pc.pc), &result)
	return bool(result)

cpdef int pcair_get_reuse_amount(PC pc):
	cdef PetscInt amount = 3
	PCAIRGetReuseAmount_c(&(pc.pc), &amount)
	return int(amount)

cpdef str pcair_get_smooth_type(PC pc):
	cdef char buf[256]
	cdef int i
	for i in range(256):
		buf[i] = 0
	PCAIRGetSmoothType_c(&(pc.pc), buf)
	return buf[:strlen(buf)].decode('utf-8')

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

# -----------------------------------------------------------------------
# PCAIR Set wrappers
# -----------------------------------------------------------------------

cpdef pcair_set_print_stats_timings(PC pc, bint flag):
	PCAIRSetPrintStatsTimings_c(&(pc.pc), <int>flag)

cpdef pcair_set_max_levels(PC pc, int max_levels):
	PCAIRSetMaxLevels_c(&(pc.pc), <PetscInt>max_levels)

cpdef pcair_set_coarse_eq_limit(PC pc, int coarse_eq_limit):
	PCAIRSetCoarseEqLimit_c(&(pc.pc), <PetscInt>coarse_eq_limit)

cpdef pcair_set_auto_truncate_start_level(PC pc, int start_level):
	PCAIRSetAutoTruncateStartLevel_c(&(pc.pc), <PetscInt>start_level)

cpdef pcair_set_auto_truncate_tol(PC pc, double tol):
	PCAIRSetAutoTruncateTol_c(&(pc.pc), <PetscReal>tol)

cpdef pcair_set_processor_agglom(PC pc, bint flag):
	PCAIRSetProcessorAgglom_c(&(pc.pc), <int>flag)

cpdef pcair_set_processor_agglom_ratio(PC pc, double ratio):
	PCAIRSetProcessorAgglomRatio_c(&(pc.pc), <PetscReal>ratio)

cpdef pcair_set_processor_agglom_factor(PC pc, int factor):
	PCAIRSetProcessorAgglomFactor_c(&(pc.pc), <PetscInt>factor)

cpdef pcair_set_process_eq_limit(PC pc, int limit):
	PCAIRSetProcessEqLimit_c(&(pc.pc), <PetscInt>limit)

cpdef pcair_set_subcomm(PC pc, bint flag):
	PCAIRSetSubcomm_c(&(pc.pc), <int>flag)

cpdef pcair_set_strong_threshold(PC pc, double thresh):
	PCAIRSetStrongThreshold_c(&(pc.pc), <PetscReal>thresh)

cpdef pcair_set_ddc_its(PC pc, int its):
	PCAIRSetDDCIts_c(&(pc.pc), <PetscInt>its)

cpdef pcair_set_max_dd_ratio(PC pc, double ratio):
	PCAIRSetMaxDDRatio_c(&(pc.pc), <PetscReal>ratio)

cpdef pcair_set_ddc_fraction(PC pc, double frac):
	PCAIRSetDDCFraction_c(&(pc.pc), <PetscReal>frac)

cpdef pcair_set_cf_splitting_type(PC pc, int algo):
	PCAIRSetCFSplittingType_c(&(pc.pc), algo)

cpdef pcair_set_max_luby_steps(PC pc, int steps):
	PCAIRSetMaxLubySteps_c(&(pc.pc), <PetscInt>steps)

cpdef pcair_set_smooth_type(PC pc, str smooth_type):
	"""Set the smooth type string (e.g. 'ff', 'fcf', 'f')."""
	cdef bytes encoded = smooth_type.encode('utf-8')
	cdef char buf[11]
	cdef int i, n
	for i in range(11):
		buf[i] = 0
	n = min(len(encoded), 10)
	for i in range(n):
		buf[i] = encoded[i]
	PCAIRSetSmoothType_c(&(pc.pc), buf)

cpdef pcair_set_diag_scale_polys(PC pc, bint flag):
	PCAIRSetDiagScalePolys_c(&(pc.pc), <int>flag)

cpdef pcair_set_matrix_free_polys(PC pc, bint flag):
	PCAIRSetMatrixFreePolys_c(&(pc.pc), <int>flag)

cpdef pcair_set_one_point_classical_prolong(PC pc, bint flag):
	PCAIRSetOnePointClassicalProlong_c(&(pc.pc), <int>flag)

cpdef pcair_set_full_smoothing_up_and_down(PC pc, bint flag):
	PCAIRSetFullSmoothingUpAndDown_c(&(pc.pc), <int>flag)

cpdef pcair_set_symmetric(PC pc, bint flag):
	PCAIRSetSymmetric_c(&(pc.pc), <int>flag)

cpdef pcair_set_constrain_w(PC pc, bint flag):
	PCAIRSetConstrainW_c(&(pc.pc), <int>flag)

cpdef pcair_set_constrain_z(PC pc, bint flag):
	PCAIRSetConstrainZ_c(&(pc.pc), <int>flag)

cpdef pcair_set_improve_w_its(PC pc, int its):
	PCAIRSetImproveWIts_c(&(pc.pc), <PetscInt>its)

cpdef pcair_set_improve_z_its(PC pc, int its):
	PCAIRSetImproveZIts_c(&(pc.pc), <PetscInt>its)

cpdef pcair_set_strong_r_threshold(PC pc, double thresh):
	PCAIRSetStrongRThreshold_c(&(pc.pc), <PetscReal>thresh)

cpdef pcair_set_inverse_type(PC pc, int inv_type):
	PCAIRSetInverseType_c(&(pc.pc), inv_type)

cpdef pcair_set_c_inverse_type(PC pc, int inv_type):
	PCAIRSetCInverseType_c(&(pc.pc), inv_type)

cpdef pcair_set_z_type(PC pc, int z_type):
	PCAIRSetZType_c(&(pc.pc), z_type)

cpdef pcair_set_lair_distance(PC pc, int distance):
	PCAIRSetLairDistance_c(&(pc.pc), <PetscInt>distance)

cpdef pcair_set_poly_order(PC pc, int order):
	PCAIRSetPolyOrder_c(&(pc.pc), <PetscInt>order)

cpdef pcair_set_inverse_sparsity_order(PC pc, int order):
	PCAIRSetInverseSparsityOrder_c(&(pc.pc), <PetscInt>order)

cpdef pcair_set_c_poly_order(PC pc, int order):
	PCAIRSetCPolyOrder_c(&(pc.pc), <PetscInt>order)

cpdef pcair_set_c_inverse_sparsity_order(PC pc, int order):
	PCAIRSetCInverseSparsityOrder_c(&(pc.pc), <PetscInt>order)

cpdef pcair_set_coarsest_inverse_type(PC pc, int inv_type):
	PCAIRSetCoarsestInverseType_c(&(pc.pc), inv_type)

cpdef pcair_set_coarsest_poly_order(PC pc, int order):
	PCAIRSetCoarsestPolyOrder_c(&(pc.pc), <PetscInt>order)

cpdef pcair_set_coarsest_inverse_sparsity_order(PC pc, int order):
	PCAIRSetCoarsestInverseSparsityOrder_c(&(pc.pc), <PetscInt>order)

cpdef pcair_set_coarsest_matrix_free_polys(PC pc, bint flag):
	PCAIRSetCoarsestMatrixFreePolys_c(&(pc.pc), <int>flag)

cpdef pcair_set_coarsest_diag_scale_polys(PC pc, bint flag):
	PCAIRSetCoarsestDiagScalePolys_c(&(pc.pc), <int>flag)

cpdef pcair_set_coarsest_subcomm(PC pc, bint flag):
	PCAIRSetCoarsestSubcomm_c(&(pc.pc), <int>flag)

cpdef pcair_set_r_drop(PC pc, double rdrop):
	PCAIRSetRDrop_c(&(pc.pc), <PetscReal>rdrop)

cpdef pcair_set_a_drop(PC pc, double adrop):
	PCAIRSetADrop_c(&(pc.pc), <PetscReal>adrop)

cpdef pcair_set_a_lump(PC pc, bint flag):
	PCAIRSetALump_c(&(pc.pc), <int>flag)

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

cpdef pcair_set_reuse_amount(PC pc, int amount):
	"""Set how much data PCAIR stores for reuse when reuse_sparsity is enabled.

	1 - store only graph-partitioner IS and symbolic SpGEMM matrices (MAT_AP, MAT_RAP)
	2 - additionally store repartitioned matrices and CF-splitting related matrices/IS
	3 - store everything (default, preserves previous behaviour)
	"""
	PCAIRSetReuseAmount_c(&(pc.pc), <PetscInt>amount)

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

# -----------------------------------------------------------------------
# PCPFLAREINV wrappers
# -----------------------------------------------------------------------

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

cpdef pcpflareinv_set_reuse_poly_coeffs(PC pc, bint flag):
	"""Tell PCPFLAREINV to reuse the current polynomial coefficients on the next setup.

	Must be called before KSPSolve, after pcpflareinv_set_poly_coeffs, to take effect.
	"""
	PCPFLAREINVSetReusePolyCoeffs(pc.pc, <int>flag)

cpdef int pcpflareinv_get_poly_order(PC pc):
	cdef PetscInt result = 0
	PCPFLAREINVGetPolyOrder(pc.pc, &result)
	return <int>result

cpdef int pcpflareinv_get_sparsity_order(PC pc):
	cdef PetscInt result = 0
	PCPFLAREINVGetSparsityOrder(pc.pc, &result)
	return <int>result

cpdef int pcpflareinv_get_type(PC pc):
	cdef int result = 0
	PCPFLAREINVGetType(pc.pc, &result)
	return result

cpdef bint pcpflareinv_get_matrix_free(PC pc):
	cdef int result = 0
	PCPFLAREINVGetMatrixFree(pc.pc, &result)
	return bool(result)

cpdef bint pcpflareinv_get_reuse_poly_coeffs(PC pc):
	cdef int result = 0
	PCPFLAREINVGetReusePolyCoeffs(pc.pc, &result)
	return bool(result)

cpdef pcpflareinv_set_poly_order(PC pc, int order):
	PCPFLAREINVSetPolyOrder(pc.pc, <PetscInt>order)

cpdef pcpflareinv_set_sparsity_order(PC pc, int order):
	PCPFLAREINVSetSparsityOrder(pc.pc, <PetscInt>order)

cpdef pcpflareinv_set_type(PC pc, int pflare_type):
	PCPFLAREINVSetType(pc.pc, pflare_type)

cpdef pcpflareinv_set_matrix_free(PC pc, bint flag):
	PCPFLAREINVSetMatrixFree(pc.pc, <int>flag)
