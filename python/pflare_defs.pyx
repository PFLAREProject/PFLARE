import numpy as np
from libc.string cimport strlen

from petsc4py import PETSc

from petsc4py.PETSc cimport Mat, PetscMat
from petsc4py.PETSc cimport PC, PetscPC
from petsc4py.PETSc cimport IS, PetscIS
from petsc4py.PETSc cimport CHKERR, PetscErrorCode


# Bring PetscInt and PetscReal from petsc.h so the C compiler resolves the
# correct widths regardless of whether PETSc was built with 32- or 64-bit
# indices, or single/double precision.
#
# NOTE: `ctypedef double PetscReal` is a Cython-side alias only. For *scalar*
# parameters and return values Cython emits the true C type `PetscReal` (so the
# C compiler picks the correct width) and coerces through Python float, which is
# always correct. The alias is a problem ONLY for bulk memory paths - a
# `double[::1, :]` memoryview or a raw memcpy sized with sizeof(PetscReal) would
# assume 8-byte elements and corrupt data under single precision. Those paths
# (the poly-coeffs get/set below) size their numpy arrays from PETSc.RealType and
# copy element-wise through the PetscReal* pointer instead. Do NOT "fix" this
# ctypedef to float - scalar coercion depends on it staying double.
cdef extern from "petsc.h":
	ctypedef int    PetscInt  "PetscInt"
	ctypedef double PetscReal "PetscReal"
	ctypedef bint   PetscBool "PetscBool"
	# Increment a PETSc object's reference count (used to balance petsc4py's
	# automatic destroy when wrapping a borrowed reference). Declared here (from
	# petsc.h) so Cython does not emit a prototype that conflicts with PETSc's.
	int PetscObjectReference(void *obj)

cdef extern:
	void PCRegister_PFLARE()
	# Call the C wrappers (not the _c Fortran routines directly): they call
	# PetscInitializeFortran() first, which petsc4py does not, so the Fortran
	# module block (MPIU_REAL etc.) is populated before the Fortran code runs.
	# Aliased to distinct Cython names so they don't clash with the cpdef wrappers.
	void compute_cf_splitting_cwrap "compute_cf_splitting" (PetscMat A, int skip_symmetrize_int, PetscReal strong_threshold, int max_luby_steps, int cf_splitting_type, int ddc_its, PetscReal fraction_swap, PetscIS* is_fine, PetscIS* is_coarse)
	void compute_diag_dom_submatrix_cwrap "compute_diag_dom_submatrix" (PetscMat A, PetscReal max_dd_ratio, PetscMat *output_mat)

	# -----------------------------------------------------------------------
	# PCAIR Get routines
	# Call the public C API (not the _c Fortran routines directly): it checks
	# the PC is of type PCAIR first, as the Fortran casts pc->data blindly.
	# Wrapping the calls in CHKERR raises PETSc.Error on a wrong type.
	# -----------------------------------------------------------------------

	# PCAIR - number of multigrid levels
	PetscErrorCode PCAIRGetNumLevels(PetscPC pc, PetscInt *input_int)

	PetscErrorCode PCAIRGetPrintStatsTimings(PetscPC pc, PetscBool *print_stats)
	PetscErrorCode PCAIRGetMaxLevels(PetscPC pc, PetscInt *max_levels)
	PetscErrorCode PCAIRGetCoarseEqLimit(PetscPC pc, PetscInt *coarse_eq_limit)
	PetscErrorCode PCAIRGetAutoTruncateStartLevel(PetscPC pc, PetscInt *start_level)
	PetscErrorCode PCAIRGetAutoTruncateTol(PetscPC pc, PetscReal *tol)
	PetscErrorCode PCAIRGetProcessorAgglom(PetscPC pc, PetscBool *processor_agglom)
	PetscErrorCode PCAIRGetProcessorAgglomRatio(PetscPC pc, PetscReal *ratio)
	PetscErrorCode PCAIRGetProcessorAgglomFactor(PetscPC pc, PetscInt *factor)
	PetscErrorCode PCAIRGetProcessEqLimit(PetscPC pc, PetscInt *limit)
	PetscErrorCode PCAIRGetSubcomm(PetscPC pc, PetscBool *subcomm)
	PetscErrorCode PCAIRGetStrongThreshold(PetscPC pc, PetscReal *thresh)
	PetscErrorCode PCAIRGetDDCIts(PetscPC pc, PetscInt *its)
	PetscErrorCode PCAIRGetDDCFraction(PetscPC pc, PetscReal *frac)
	PetscErrorCode PCAIRGetCFSplittingType(PetscPC pc, int *algo)
	PetscErrorCode PCAIRGetMaxLubySteps(PetscPC pc, PetscInt *steps)
	PetscErrorCode PCAIRGetDiagScalePolys(PetscPC pc, PetscBool *scale)
	PetscErrorCode PCAIRGetMatrixFreePolys(PetscPC pc, PetscBool *mf)
	PetscErrorCode PCAIRGetOnePointClassicalProlong(PetscPC pc, PetscBool *onep)
	PetscErrorCode PCAIRGetFullSmoothingUpAndDown(PetscPC pc, PetscBool *full)
	PetscErrorCode PCAIRGetSymmetric(PetscPC pc, PetscBool *sym)
	PetscErrorCode PCAIRGetConstrainW(PetscPC pc, PetscBool *constrain)
	PetscErrorCode PCAIRGetConstrainZ(PetscPC pc, PetscBool *constrain)
	PetscErrorCode PCAIRGetImproveWIts(PetscPC pc, PetscInt *its)
	PetscErrorCode PCAIRGetImproveZIts(PetscPC pc, PetscInt *its)
	PetscErrorCode PCAIRGetStrongRThreshold(PetscPC pc, PetscReal *thresh)
	PetscErrorCode PCAIRGetInverseType(PetscPC pc, int *inv_type)
	PetscErrorCode PCAIRGetCInverseType(PetscPC pc, int *inv_type)
	PetscErrorCode PCAIRGetZType(PetscPC pc, int *z_type)
	PetscErrorCode PCAIRGetLairDistance(PetscPC pc, PetscInt *distance)
	PetscErrorCode PCAIRGetPolyOrder(PetscPC pc, PetscInt *order)
	PetscErrorCode PCAIRGetInverseSparsityOrder(PetscPC pc, PetscInt *order)
	PetscErrorCode PCAIRGetCPolyOrder(PetscPC pc, PetscInt *order)
	PetscErrorCode PCAIRGetCInverseSparsityOrder(PetscPC pc, PetscInt *order)
	PetscErrorCode PCAIRGetCoarsestInverseType(PetscPC pc, int *inv_type)
	PetscErrorCode PCAIRGetCoarsestPolyOrder(PetscPC pc, PetscInt *order)
	PetscErrorCode PCAIRGetCoarsestInverseSparsityOrder(PetscPC pc, PetscInt *order)
	PetscErrorCode PCAIRGetCoarsestMatrixFreePolys(PetscPC pc, PetscBool *mf)
	PetscErrorCode PCAIRGetCoarsestDiagScalePolys(PetscPC pc, PetscBool *scale)
	PetscErrorCode PCAIRGetCoarsestSubcomm(PetscPC pc, PetscBool *subcomm)
	PetscErrorCode PCAIRGetRDrop(PetscPC pc, PetscReal *rdrop)
	PetscErrorCode PCAIRGetADrop(PetscPC pc, PetscReal *adrop)
	PetscErrorCode PCAIRGetALump(PetscPC pc, PetscBool *lump)
	PetscErrorCode PCAIRGetReuseSparsity(PetscPC pc, PetscBool *reuse)
	PetscErrorCode PCAIRGetReusePolyCoeffs(PetscPC pc, PetscBool *reuse)
	PetscErrorCode PCAIRGetReuseAmount(PetscPC pc, PetscInt *amount)
	PetscErrorCode PCAIRGetSmoothType(PetscPC pc, char *output_string)

	# PCAIR - polynomial coefficients
	# Returns a pointer into internal PCAIR memory (valid until the next PCSetUp or PCReset).
	# The Python wrapper copies the data before returning.
	PetscErrorCode PCAIRGetPolyCoeffs(PetscPC pc, PetscInt petsc_level, int which_inverse,
	                           PetscReal **coeffs_ptr, PetscInt *row_size, PetscInt *col_size)
	PetscErrorCode PCAIRGetGridComplexity(PetscPC pc, PetscReal *complexity)
	PetscErrorCode PCAIRGetOperatorComplexity(PetscPC pc, PetscReal *complexity)
	PetscErrorCode PCAIRGetCycleComplexity(PetscPC pc, PetscReal *complexity)
	PetscErrorCode PCAIRGetStorageComplexity(PetscPC pc, PetscReal *complexity)
	PetscErrorCode PCAIRGetReuseStorageComplexity(PetscPC pc, PetscReal *complexity)

	# -----------------------------------------------------------------------
	# PCAIR Set routines
	# -----------------------------------------------------------------------

	PetscErrorCode PCAIRSetPrintStatsTimings(PetscPC pc, PetscBool print_stats)
	PetscErrorCode PCAIRSetMaxLevels(PetscPC pc, PetscInt max_levels)
	PetscErrorCode PCAIRSetCoarseEqLimit(PetscPC pc, PetscInt coarse_eq_limit)
	PetscErrorCode PCAIRSetAutoTruncateStartLevel(PetscPC pc, PetscInt start_level)
	PetscErrorCode PCAIRSetAutoTruncateTol(PetscPC pc, PetscReal tol)
	PetscErrorCode PCAIRSetProcessorAgglom(PetscPC pc, PetscBool processor_agglom)
	PetscErrorCode PCAIRSetProcessorAgglomRatio(PetscPC pc, PetscReal ratio)
	PetscErrorCode PCAIRSetProcessorAgglomFactor(PetscPC pc, PetscInt factor)
	PetscErrorCode PCAIRSetProcessEqLimit(PetscPC pc, PetscInt limit)
	PetscErrorCode PCAIRSetSubcomm(PetscPC pc, PetscBool subcomm)
	PetscErrorCode PCAIRSetStrongThreshold(PetscPC pc, PetscReal thresh)
	PetscErrorCode PCAIRSetDDCIts(PetscPC pc, PetscInt its)
	PetscErrorCode PCAIRSetDDCFraction(PetscPC pc, PetscReal frac)
	PetscErrorCode PCAIRSetCFSplittingType(PetscPC pc, int algo)
	PetscErrorCode PCAIRSetMaxLubySteps(PetscPC pc, PetscInt steps)
	PetscErrorCode PCAIRSetSmoothType(PetscPC pc, const char *input_string)
	PetscErrorCode PCAIRSetDiagScalePolys(PetscPC pc, PetscBool scale)
	PetscErrorCode PCAIRSetMatrixFreePolys(PetscPC pc, PetscBool mf)
	PetscErrorCode PCAIRSetOnePointClassicalProlong(PetscPC pc, PetscBool onep)
	PetscErrorCode PCAIRSetFullSmoothingUpAndDown(PetscPC pc, PetscBool full)
	PetscErrorCode PCAIRSetSymmetric(PetscPC pc, PetscBool sym)
	PetscErrorCode PCAIRSetConstrainW(PetscPC pc, PetscBool constrain)
	PetscErrorCode PCAIRSetConstrainZ(PetscPC pc, PetscBool constrain)
	PetscErrorCode PCAIRSetImproveWIts(PetscPC pc, PetscInt its)
	PetscErrorCode PCAIRSetImproveZIts(PetscPC pc, PetscInt its)
	PetscErrorCode PCAIRSetStrongRThreshold(PetscPC pc, PetscReal thresh)
	PetscErrorCode PCAIRSetInverseType(PetscPC pc, int inv_type)
	PetscErrorCode PCAIRSetCInverseType(PetscPC pc, int inv_type)
	PetscErrorCode PCAIRSetZType(PetscPC pc, int z_type)
	PetscErrorCode PCAIRSetLairDistance(PetscPC pc, PetscInt distance)
	PetscErrorCode PCAIRSetPolyOrder(PetscPC pc, PetscInt order)
	PetscErrorCode PCAIRSetInverseSparsityOrder(PetscPC pc, PetscInt order)
	PetscErrorCode PCAIRSetCPolyOrder(PetscPC pc, PetscInt order)
	PetscErrorCode PCAIRSetCInverseSparsityOrder(PetscPC pc, PetscInt order)
	PetscErrorCode PCAIRSetCoarsestInverseType(PetscPC pc, int inv_type)
	PetscErrorCode PCAIRSetCoarsestPolyOrder(PetscPC pc, PetscInt order)
	PetscErrorCode PCAIRSetCoarsestInverseSparsityOrder(PetscPC pc, PetscInt order)
	PetscErrorCode PCAIRSetCoarsestMatrixFreePolys(PetscPC pc, PetscBool mf)
	PetscErrorCode PCAIRSetCoarsestDiagScalePolys(PetscPC pc, PetscBool scale)
	PetscErrorCode PCAIRSetCoarsestSubcomm(PetscPC pc, PetscBool subcomm)
	PetscErrorCode PCAIRSetRDrop(PetscPC pc, PetscReal rdrop)
	PetscErrorCode PCAIRSetADrop(PetscPC pc, PetscReal adrop)
	PetscErrorCode PCAIRSetALump(PetscPC pc, PetscBool lump)

	# PCAIR - reuse flags
	PetscErrorCode PCAIRSetReuseSparsity(PetscPC pc, PetscBool input_bool)
	PetscErrorCode PCAIRSetReusePolyCoeffs(PetscPC pc, PetscBool input_bool)
	PetscErrorCode PCAIRSetReuseAmount(PetscPC pc, PetscInt amount)

	# PCAIR - set polynomial coefficients (copies from the provided pointer)
	PetscErrorCode PCAIRSetPolyCoeffs(PetscPC pc, PetscInt petsc_level, int which_inverse,
	                           PetscReal *coeffs_ptr, PetscInt row_size, PetscInt col_size)

	# PCPFLAREINV - Get routines (PC passed by value, not pointer)
	PetscErrorCode PCPFLAREINVGetPolyOrder(PetscPC pc, PetscInt *order)
	PetscErrorCode PCPFLAREINVGetSparsityOrder(PetscPC pc, PetscInt *order)
	PetscErrorCode PCPFLAREINVGetType(PetscPC pc, int *pflare_type)
	PetscErrorCode PCPFLAREINVGetMatrixFree(PetscPC pc, PetscBool *flag)
	PetscErrorCode PCPFLAREINVGetReusePolyCoeffs(PetscPC pc, PetscBool *flag)

	# PCPFLAREINV - underlying approximate-inverse matrix (borrowed reference)
	PetscErrorCode PCPFLAREINVGetInverseMat(PetscPC pc, PetscMat *mat)

	# PCPFLAREINV - polynomial coefficients (PC passed by value, not pointer)
	# Returns a pointer into internal PCPFLAREINV memory (valid until the next PCSetUp or PCReset).
	# The Python wrapper copies the data before returning.
	PetscErrorCode PCPFLAREINVGetPolyCoeffs(PetscPC pc, PetscReal **coeffs, PetscInt *rows, PetscInt *cols)

	# PCPFLAREINV - Set routines (PC passed by value, not pointer)
	PetscErrorCode PCPFLAREINVSetPolyOrder(PetscPC pc, PetscInt order)
	PetscErrorCode PCPFLAREINVSetSparsityOrder(PetscPC pc, PetscInt order)
	PetscErrorCode PCPFLAREINVSetType(PetscPC pc, int pflare_type)
	PetscErrorCode PCPFLAREINVSetMatrixFree(PetscPC pc, PetscBool flag)

	# PCPFLAREINV - set polynomial coefficients (copies from the provided pointer)
	PetscErrorCode PCPFLAREINVSetPolyCoeffs(PetscPC pc, PetscReal *coeffs, PetscInt rows, PetscInt cols)

	# PCPFLAREINV - reuse flag
	PetscErrorCode PCPFLAREINVSetReusePolyCoeffs(PetscPC pc, PetscBool flg)


cpdef py_PCRegister_PFLARE():
	PCRegister_PFLARE()

cpdef compute_cf_splitting(Mat A, bint skip_symmetrize, PetscReal strong_threshold, int max_luby_steps, int cf_splitting_type, int ddc_its, PetscReal fraction_swap):
	cdef IS is_fine
	cdef IS is_coarse
	is_fine = IS()
	is_coarse = IS()
	compute_cf_splitting_cwrap(A.mat, skip_symmetrize, strong_threshold, max_luby_steps, cf_splitting_type, ddc_its, fraction_swap, &(is_fine.iset), &(is_coarse.iset))
	return is_fine, is_coarse

cpdef compute_diag_dom_submatrix(Mat A, PetscReal max_dd_ratio):
	cdef Mat output_mat
	output_mat = Mat()
	compute_diag_dom_submatrix_cwrap(A.mat, max_dd_ratio, &(output_mat.mat))
	return output_mat

# -----------------------------------------------------------------------
# PCAIR Get wrappers
# -----------------------------------------------------------------------

cpdef int pcair_get_num_levels(PC pc):
	"""Return the number of multigrid levels in a PCAIR preconditioner."""
	cdef PetscInt num_levels = 0
	CHKERR(PCAIRGetNumLevels(pc.pc, &num_levels))
	return <int>num_levels

cpdef bint pcair_get_print_stats_timings(PC pc):
	cdef PetscBool result = 0
	CHKERR(PCAIRGetPrintStatsTimings(pc.pc, &result))
	return bool(result)

cpdef int pcair_get_max_levels(PC pc):
	cdef PetscInt result = 0
	CHKERR(PCAIRGetMaxLevels(pc.pc, &result))
	return <int>result

cpdef int pcair_get_coarse_eq_limit(PC pc):
	cdef PetscInt result = 0
	CHKERR(PCAIRGetCoarseEqLimit(pc.pc, &result))
	return <int>result

cpdef int pcair_get_auto_truncate_start_level(PC pc):
	cdef PetscInt result = 0
	CHKERR(PCAIRGetAutoTruncateStartLevel(pc.pc, &result))
	return <int>result

cpdef double pcair_get_auto_truncate_tol(PC pc):
	cdef PetscReal result = 0.0
	CHKERR(PCAIRGetAutoTruncateTol(pc.pc, &result))
	return <double>result

cpdef bint pcair_get_processor_agglom(PC pc):
	cdef PetscBool result = 0
	CHKERR(PCAIRGetProcessorAgglom(pc.pc, &result))
	return bool(result)

cpdef double pcair_get_processor_agglom_ratio(PC pc):
	cdef PetscReal result = 0.0
	CHKERR(PCAIRGetProcessorAgglomRatio(pc.pc, &result))
	return <double>result

cpdef int pcair_get_processor_agglom_factor(PC pc):
	cdef PetscInt result = 0
	CHKERR(PCAIRGetProcessorAgglomFactor(pc.pc, &result))
	return <int>result

cpdef int pcair_get_process_eq_limit(PC pc):
	cdef PetscInt result = 0
	CHKERR(PCAIRGetProcessEqLimit(pc.pc, &result))
	return <int>result

cpdef bint pcair_get_subcomm(PC pc):
	cdef PetscBool result = 0
	CHKERR(PCAIRGetSubcomm(pc.pc, &result))
	return bool(result)

cpdef double pcair_get_strong_threshold(PC pc):
	cdef PetscReal result = 0.0
	CHKERR(PCAIRGetStrongThreshold(pc.pc, &result))
	return <double>result

cpdef int pcair_get_ddc_its(PC pc):
	cdef PetscInt result = 0
	CHKERR(PCAIRGetDDCIts(pc.pc, &result))
	return <int>result

cpdef double pcair_get_ddc_fraction(PC pc):
	cdef PetscReal result = 0.0
	CHKERR(PCAIRGetDDCFraction(pc.pc, &result))
	return <double>result

cpdef int pcair_get_cf_splitting_type(PC pc):
	cdef int result = 0
	CHKERR(PCAIRGetCFSplittingType(pc.pc, &result))
	return result

cpdef int pcair_get_max_luby_steps(PC pc):
	cdef PetscInt result = 0
	CHKERR(PCAIRGetMaxLubySteps(pc.pc, &result))
	return <int>result

cpdef bint pcair_get_diag_scale_polys(PC pc):
	cdef PetscBool result = 0
	CHKERR(PCAIRGetDiagScalePolys(pc.pc, &result))
	return bool(result)

cpdef bint pcair_get_matrix_free_polys(PC pc):
	cdef PetscBool result = 0
	CHKERR(PCAIRGetMatrixFreePolys(pc.pc, &result))
	return bool(result)

cpdef bint pcair_get_one_point_classical_prolong(PC pc):
	cdef PetscBool result = 0
	CHKERR(PCAIRGetOnePointClassicalProlong(pc.pc, &result))
	return bool(result)

cpdef bint pcair_get_full_smoothing_up_and_down(PC pc):
	cdef PetscBool result = 0
	CHKERR(PCAIRGetFullSmoothingUpAndDown(pc.pc, &result))
	return bool(result)

cpdef bint pcair_get_symmetric(PC pc):
	cdef PetscBool result = 0
	CHKERR(PCAIRGetSymmetric(pc.pc, &result))
	return bool(result)

cpdef bint pcair_get_constrain_w(PC pc):
	cdef PetscBool result = 0
	CHKERR(PCAIRGetConstrainW(pc.pc, &result))
	return bool(result)

cpdef bint pcair_get_constrain_z(PC pc):
	cdef PetscBool result = 0
	CHKERR(PCAIRGetConstrainZ(pc.pc, &result))
	return bool(result)

cpdef int pcair_get_improve_w_its(PC pc):
	cdef PetscInt result = 0
	CHKERR(PCAIRGetImproveWIts(pc.pc, &result))
	return <int>result

cpdef int pcair_get_improve_z_its(PC pc):
	cdef PetscInt result = 0
	CHKERR(PCAIRGetImproveZIts(pc.pc, &result))
	return <int>result

cpdef double pcair_get_strong_r_threshold(PC pc):
	cdef PetscReal result = 0.0
	CHKERR(PCAIRGetStrongRThreshold(pc.pc, &result))
	return <double>result

cpdef int pcair_get_inverse_type(PC pc):
	cdef int result = 0
	CHKERR(PCAIRGetInverseType(pc.pc, &result))
	return result

cpdef int pcair_get_c_inverse_type(PC pc):
	cdef int result = 0
	CHKERR(PCAIRGetCInverseType(pc.pc, &result))
	return result

cpdef int pcair_get_z_type(PC pc):
	cdef int result = 0
	CHKERR(PCAIRGetZType(pc.pc, &result))
	return result

cpdef int pcair_get_lair_distance(PC pc):
	cdef PetscInt result = 0
	CHKERR(PCAIRGetLairDistance(pc.pc, &result))
	return <int>result

cpdef int pcair_get_poly_order(PC pc):
	cdef PetscInt result = 0
	CHKERR(PCAIRGetPolyOrder(pc.pc, &result))
	return <int>result

cpdef int pcair_get_inverse_sparsity_order(PC pc):
	cdef PetscInt result = 0
	CHKERR(PCAIRGetInverseSparsityOrder(pc.pc, &result))
	return <int>result

cpdef int pcair_get_c_poly_order(PC pc):
	cdef PetscInt result = 0
	CHKERR(PCAIRGetCPolyOrder(pc.pc, &result))
	return <int>result

cpdef int pcair_get_c_inverse_sparsity_order(PC pc):
	cdef PetscInt result = 0
	CHKERR(PCAIRGetCInverseSparsityOrder(pc.pc, &result))
	return <int>result

cpdef int pcair_get_coarsest_inverse_type(PC pc):
	cdef int result = 0
	CHKERR(PCAIRGetCoarsestInverseType(pc.pc, &result))
	return result

cpdef int pcair_get_coarsest_poly_order(PC pc):
	cdef PetscInt result = 0
	CHKERR(PCAIRGetCoarsestPolyOrder(pc.pc, &result))
	return <int>result

cpdef int pcair_get_coarsest_inverse_sparsity_order(PC pc):
	cdef PetscInt result = 0
	CHKERR(PCAIRGetCoarsestInverseSparsityOrder(pc.pc, &result))
	return <int>result

cpdef bint pcair_get_coarsest_matrix_free_polys(PC pc):
	cdef PetscBool result = 0
	CHKERR(PCAIRGetCoarsestMatrixFreePolys(pc.pc, &result))
	return bool(result)

cpdef bint pcair_get_coarsest_diag_scale_polys(PC pc):
	cdef PetscBool result = 0
	CHKERR(PCAIRGetCoarsestDiagScalePolys(pc.pc, &result))
	return bool(result)

cpdef bint pcair_get_coarsest_subcomm(PC pc):
	cdef PetscBool result = 0
	CHKERR(PCAIRGetCoarsestSubcomm(pc.pc, &result))
	return bool(result)

cpdef double pcair_get_r_drop(PC pc):
	cdef PetscReal result = 0.0
	CHKERR(PCAIRGetRDrop(pc.pc, &result))
	return <double>result

cpdef double pcair_get_a_drop(PC pc):
	cdef PetscReal result = 0.0
	CHKERR(PCAIRGetADrop(pc.pc, &result))
	return <double>result

cpdef double pcair_get_grid_complexity(PC pc):
	cdef PetscReal result = 0.0
	CHKERR(PCAIRGetGridComplexity(pc.pc, &result))
	return <double>result

cpdef double pcair_get_operator_complexity(PC pc):
	cdef PetscReal result = 0.0
	CHKERR(PCAIRGetOperatorComplexity(pc.pc, &result))
	return <double>result

cpdef double pcair_get_cycle_complexity(PC pc):
	cdef PetscReal result = 0.0
	CHKERR(PCAIRGetCycleComplexity(pc.pc, &result))
	return <double>result

cpdef double pcair_get_storage_complexity(PC pc):
	cdef PetscReal result = 0.0
	CHKERR(PCAIRGetStorageComplexity(pc.pc, &result))
	return <double>result

cpdef double pcair_get_reuse_storage_complexity(PC pc):
	cdef PetscReal result = 0.0
	CHKERR(PCAIRGetReuseStorageComplexity(pc.pc, &result))
	return <double>result

cpdef bint pcair_get_a_lump(PC pc):
	cdef PetscBool result = 0
	CHKERR(PCAIRGetALump(pc.pc, &result))
	return bool(result)

cpdef bint pcair_get_reuse_sparsity(PC pc):
	cdef PetscBool result = 0
	CHKERR(PCAIRGetReuseSparsity(pc.pc, &result))
	return bool(result)

cpdef bint pcair_get_reuse_poly_coeffs(PC pc):
	cdef PetscBool result = 0
	CHKERR(PCAIRGetReusePolyCoeffs(pc.pc, &result))
	return bool(result)

cpdef int pcair_get_reuse_amount(PC pc):
	cdef PetscInt amount = 3
	CHKERR(PCAIRGetReuseAmount(pc.pc, &amount))
	return int(amount)

cpdef str pcair_get_smooth_type(PC pc):
	cdef char buf[256]
	cdef int i
	for i in range(256):
		buf[i] = 0
	CHKERR(PCAIRGetSmoothType(pc.pc, buf))
	return buf[:strlen(buf)].decode('utf-8')

cpdef pcair_get_poly_coeffs(PC pc, int petsc_level, int which_inverse):
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
	cdef PetscInt i, j
	CHKERR(PCAIRGetPolyCoeffs(pc.pc, petsc_level, which_inverse,
	                             &coeffs_ptr, &row_size, &col_size))
	# Match the numpy dtype to the build's PetscReal width (float32 single /
	# float64 double) and copy element-wise through the PetscReal* pointer.
	# A raw memcpy sized with sizeof(PetscReal) into a double[::1,:] view would
	# misinterpret the buffer under single precision. The coefficients array is
	# Fortran-ordered (column-major): element (i,j) sits at i + j*row_size.
	result = np.empty((row_size, col_size), dtype=np.dtype(PETSc.RealType), order='F')
	for j in range(col_size):
		for i in range(row_size):
			result[i, j] = coeffs_ptr[i + j * row_size]
	return result

# -----------------------------------------------------------------------
# PCAIR Set wrappers
# -----------------------------------------------------------------------

cpdef pcair_set_print_stats_timings(PC pc, bint flag):
	CHKERR(PCAIRSetPrintStatsTimings(pc.pc, <PetscBool>flag))

cpdef pcair_set_max_levels(PC pc, int max_levels):
	CHKERR(PCAIRSetMaxLevels(pc.pc, <PetscInt>max_levels))

cpdef pcair_set_coarse_eq_limit(PC pc, int coarse_eq_limit):
	CHKERR(PCAIRSetCoarseEqLimit(pc.pc, <PetscInt>coarse_eq_limit))

cpdef pcair_set_auto_truncate_start_level(PC pc, int start_level):
	CHKERR(PCAIRSetAutoTruncateStartLevel(pc.pc, <PetscInt>start_level))

cpdef pcair_set_auto_truncate_tol(PC pc, double tol):
	CHKERR(PCAIRSetAutoTruncateTol(pc.pc, <PetscReal>tol))

cpdef pcair_set_processor_agglom(PC pc, bint flag):
	CHKERR(PCAIRSetProcessorAgglom(pc.pc, <PetscBool>flag))

cpdef pcair_set_processor_agglom_ratio(PC pc, double ratio):
	CHKERR(PCAIRSetProcessorAgglomRatio(pc.pc, <PetscReal>ratio))

cpdef pcair_set_processor_agglom_factor(PC pc, int factor):
	CHKERR(PCAIRSetProcessorAgglomFactor(pc.pc, <PetscInt>factor))

cpdef pcair_set_process_eq_limit(PC pc, int limit):
	CHKERR(PCAIRSetProcessEqLimit(pc.pc, <PetscInt>limit))

cpdef pcair_set_subcomm(PC pc, bint flag):
	CHKERR(PCAIRSetSubcomm(pc.pc, <PetscBool>flag))

cpdef pcair_set_strong_threshold(PC pc, double thresh):
	CHKERR(PCAIRSetStrongThreshold(pc.pc, <PetscReal>thresh))

cpdef pcair_set_ddc_its(PC pc, int its):
	CHKERR(PCAIRSetDDCIts(pc.pc, <PetscInt>its))

cpdef pcair_set_ddc_fraction(PC pc, double frac):
	CHKERR(PCAIRSetDDCFraction(pc.pc, <PetscReal>frac))

cpdef pcair_set_cf_splitting_type(PC pc, int algo):
	CHKERR(PCAIRSetCFSplittingType(pc.pc, algo))

cpdef pcair_set_max_luby_steps(PC pc, int steps):
	CHKERR(PCAIRSetMaxLubySteps(pc.pc, <PetscInt>steps))

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
	CHKERR(PCAIRSetSmoothType(pc.pc, buf))

cpdef pcair_set_diag_scale_polys(PC pc, bint flag):
	CHKERR(PCAIRSetDiagScalePolys(pc.pc, <PetscBool>flag))

cpdef pcair_set_matrix_free_polys(PC pc, bint flag):
	CHKERR(PCAIRSetMatrixFreePolys(pc.pc, <PetscBool>flag))

cpdef pcair_set_one_point_classical_prolong(PC pc, bint flag):
	CHKERR(PCAIRSetOnePointClassicalProlong(pc.pc, <PetscBool>flag))

cpdef pcair_set_full_smoothing_up_and_down(PC pc, bint flag):
	CHKERR(PCAIRSetFullSmoothingUpAndDown(pc.pc, <PetscBool>flag))

cpdef pcair_set_symmetric(PC pc, bint flag):
	CHKERR(PCAIRSetSymmetric(pc.pc, <PetscBool>flag))

cpdef pcair_set_constrain_w(PC pc, bint flag):
	CHKERR(PCAIRSetConstrainW(pc.pc, <PetscBool>flag))

cpdef pcair_set_constrain_z(PC pc, bint flag):
	CHKERR(PCAIRSetConstrainZ(pc.pc, <PetscBool>flag))

cpdef pcair_set_improve_w_its(PC pc, int its):
	CHKERR(PCAIRSetImproveWIts(pc.pc, <PetscInt>its))

cpdef pcair_set_improve_z_its(PC pc, int its):
	CHKERR(PCAIRSetImproveZIts(pc.pc, <PetscInt>its))

cpdef pcair_set_strong_r_threshold(PC pc, double thresh):
	CHKERR(PCAIRSetStrongRThreshold(pc.pc, <PetscReal>thresh))

cpdef pcair_set_inverse_type(PC pc, int inv_type):
	CHKERR(PCAIRSetInverseType(pc.pc, inv_type))

cpdef pcair_set_c_inverse_type(PC pc, int inv_type):
	CHKERR(PCAIRSetCInverseType(pc.pc, inv_type))

cpdef pcair_set_z_type(PC pc, int z_type):
	CHKERR(PCAIRSetZType(pc.pc, z_type))

cpdef pcair_set_lair_distance(PC pc, int distance):
	CHKERR(PCAIRSetLairDistance(pc.pc, <PetscInt>distance))

cpdef pcair_set_poly_order(PC pc, int order):
	CHKERR(PCAIRSetPolyOrder(pc.pc, <PetscInt>order))

cpdef pcair_set_inverse_sparsity_order(PC pc, int order):
	CHKERR(PCAIRSetInverseSparsityOrder(pc.pc, <PetscInt>order))

cpdef pcair_set_c_poly_order(PC pc, int order):
	CHKERR(PCAIRSetCPolyOrder(pc.pc, <PetscInt>order))

cpdef pcair_set_c_inverse_sparsity_order(PC pc, int order):
	CHKERR(PCAIRSetCInverseSparsityOrder(pc.pc, <PetscInt>order))

cpdef pcair_set_coarsest_inverse_type(PC pc, int inv_type):
	CHKERR(PCAIRSetCoarsestInverseType(pc.pc, inv_type))

cpdef pcair_set_coarsest_poly_order(PC pc, int order):
	CHKERR(PCAIRSetCoarsestPolyOrder(pc.pc, <PetscInt>order))

cpdef pcair_set_coarsest_inverse_sparsity_order(PC pc, int order):
	CHKERR(PCAIRSetCoarsestInverseSparsityOrder(pc.pc, <PetscInt>order))

cpdef pcair_set_coarsest_matrix_free_polys(PC pc, bint flag):
	CHKERR(PCAIRSetCoarsestMatrixFreePolys(pc.pc, <PetscBool>flag))

cpdef pcair_set_coarsest_diag_scale_polys(PC pc, bint flag):
	CHKERR(PCAIRSetCoarsestDiagScalePolys(pc.pc, <PetscBool>flag))

cpdef pcair_set_coarsest_subcomm(PC pc, bint flag):
	CHKERR(PCAIRSetCoarsestSubcomm(pc.pc, <PetscBool>flag))

cpdef pcair_set_r_drop(PC pc, double rdrop):
	CHKERR(PCAIRSetRDrop(pc.pc, <PetscReal>rdrop))

cpdef pcair_set_a_drop(PC pc, double adrop):
	CHKERR(PCAIRSetADrop(pc.pc, <PetscReal>adrop))

cpdef pcair_set_a_lump(PC pc, bint flag):
	CHKERR(PCAIRSetALump(pc.pc, <PetscBool>flag))

cpdef pcair_set_reuse_sparsity(PC pc, bint flag):
	"""Tell PCAIR to reuse sparsity (CF splitting and matrix structure) on the next setup.

	Must be called before KSPSolve to take effect.
	"""
	CHKERR(PCAIRSetReuseSparsity(pc.pc, <PetscBool>flag))

cpdef pcair_set_reuse_poly_coeffs(PC pc, bint flag):
	"""Tell PCAIR to reuse the current polynomial coefficients on the next setup.

	Must be called before KSPSolve, after pcair_set_poly_coeffs, to take effect.
	"""
	CHKERR(PCAIRSetReusePolyCoeffs(pc.pc, <PetscBool>flag))

cpdef pcair_set_reuse_amount(PC pc, int amount):
	"""Set how much data PCAIR stores for reuse when reuse_sparsity is enabled.

	1 - store only graph-partitioner IS and symbolic SpGEMM matrices (MAT_AP, MAT_RAP)
	2 - additionally store repartitioned matrices and CF-splitting related matrices/IS
	3 - store everything (default, preserves previous behaviour)
	"""
	CHKERR(PCAIRSetReuseAmount(pc.pc, <PetscInt>amount))

cpdef pcair_set_poly_coeffs(PC pc, int petsc_level, int which_inverse, coeffs):
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
	# Stage in the build's PetscReal dtype (float32 single / float64 double) so the
	# PetscReal* the C setter reads has the right element width. Casting a float64
	# buffer to PetscReal* would stride wrongly under single precision.
	staging = np.asfortranarray(coeffs, dtype=np.dtype(PETSc.RealType))
	cdef PetscInt row_size = <PetscInt>staging.shape[0]
	cdef PetscInt col_size = <PetscInt>staging.shape[1]
	cdef PetscReal *sptr = <PetscReal*><size_t>staging.ctypes.data
	CHKERR(PCAIRSetPolyCoeffs(pc.pc, petsc_level, which_inverse,
	                             sptr, row_size, col_size))

# -----------------------------------------------------------------------
# PCPFLAREINV wrappers
# -----------------------------------------------------------------------

cpdef pcpflareinv_get_poly_coeffs(PC pc):
	"""Return a copy of the GMRES polynomial coefficients from a PCPFLAREINV preconditioner.

	Returns
	-------
	numpy.ndarray, shape (poly_order+1, 1_or_2), Fortran-contiguous
	    Column 0: real coefficients (power/Arnoldi/Neumann) or real roots (Newton).
	    Column 1: imaginary roots (Newton basis only).
	"""
	cdef PetscReal *coeffs_ptr = NULL
	cdef PetscInt rows = 0, cols = 0
	cdef PetscInt i, j
	CHKERR(PCPFLAREINVGetPolyCoeffs(pc.pc, &coeffs_ptr, &rows, &cols))
	# See pcair_get_poly_coeffs: dtype must track the build's PetscReal width and
	# the copy must go element-wise through the PetscReal* (Fortran/column-major).
	result = np.empty((rows, cols), dtype=np.dtype(PETSc.RealType), order='F')
	for j in range(cols):
		for i in range(rows):
			result[i, j] = coeffs_ptr[i + j * rows]
	return result

cpdef pcpflareinv_set_poly_coeffs(PC pc, coeffs):
	"""Copy polynomial coefficients into the PCPFLAREINV preconditioner.

	Parameters
	----------
	pc : PC
	    A PCPFLAREINV preconditioner.
	coeffs : numpy.ndarray
	    Coefficient array as returned by pcpflareinv_get_poly_coeffs.
	    Must have shape (poly_order+1, 1_or_2).
	"""
	# See pcair_set_poly_coeffs: stage in the build's PetscReal dtype so the
	# PetscReal* has the right element width under single precision.
	staging = np.asfortranarray(coeffs, dtype=np.dtype(PETSc.RealType))
	cdef PetscInt rows = <PetscInt>staging.shape[0]
	cdef PetscInt cols = <PetscInt>staging.shape[1]
	cdef PetscReal *sptr = <PetscReal*><size_t>staging.ctypes.data
	CHKERR(PCPFLAREINVSetPolyCoeffs(pc.pc, sptr, rows, cols))

cpdef pcpflareinv_set_reuse_poly_coeffs(PC pc, bint flag):
	"""Tell PCPFLAREINV to reuse the current polynomial coefficients on the next setup.

	Must be called before KSPSolve, after pcpflareinv_set_poly_coeffs, to take effect.
	"""
	CHKERR(PCPFLAREINVSetReusePolyCoeffs(pc.pc, <PetscBool>flag))

cpdef int pcpflareinv_get_poly_order(PC pc):
	cdef PetscInt result = 0
	CHKERR(PCPFLAREINVGetPolyOrder(pc.pc, &result))
	return <int>result

cpdef int pcpflareinv_get_sparsity_order(PC pc):
	cdef PetscInt result = 0
	CHKERR(PCPFLAREINVGetSparsityOrder(pc.pc, &result))
	return <int>result

cpdef int pcpflareinv_get_type(PC pc):
	cdef int result = 0
	CHKERR(PCPFLAREINVGetType(pc.pc, &result))
	return result

cpdef bint pcpflareinv_get_matrix_free(PC pc):
	cdef PetscBool result = 0
	CHKERR(PCPFLAREINVGetMatrixFree(pc.pc, &result))
	return bool(result)

cpdef bint pcpflareinv_get_reuse_poly_coeffs(PC pc):
	cdef PetscBool result = 0
	CHKERR(PCPFLAREINVGetReusePolyCoeffs(pc.pc, &result))
	return bool(result)

cpdef pcpflareinv_get_inverse_mat(PC pc):
	"""Return the underlying approximate-inverse matrix of a PCPFLAREINV.

	Borrowed reference into the PC; valid only after PCSetUp and until the next
	setup/reset. Returns None if PCSetUp has not been called yet.
	"""
	cdef Mat mat = Mat()
	CHKERR(PCPFLAREINVGetInverseMat(pc.pc, &(mat.mat)))
	if mat.mat == NULL:
		return None
	# Borrowed reference: petsc4py will MatDestroy on garbage collection, so
	# increment the count to keep the PC's matrix valid
	PetscObjectReference(<void*>mat.mat)
	return mat

cpdef pcpflareinv_set_poly_order(PC pc, int order):
	CHKERR(PCPFLAREINVSetPolyOrder(pc.pc, <PetscInt>order))

cpdef pcpflareinv_set_sparsity_order(PC pc, int order):
	CHKERR(PCPFLAREINVSetSparsityOrder(pc.pc, <PetscInt>order))

cpdef pcpflareinv_set_type(PC pc, int pflare_type):
	CHKERR(PCPFLAREINVSetType(pc.pc, pflare_type))

cpdef pcpflareinv_set_matrix_free(PC pc, bint flag):
	CHKERR(PCPFLAREINVSetMatrixFree(pc.pc, <PetscBool>flag))
