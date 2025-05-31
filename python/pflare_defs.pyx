from petsc4py.PETSc cimport Mat, PetscMat
from petsc4py.PETSc cimport PC, PetscPC
from petsc4py.PETSc cimport IS, PetscIS

cdef extern:
	void PCRegister_PFLARE()
	void compute_cf_splitting_c(PetscMat *A, int symmetric_int, double strong_threshold, int max_luby_steps, int cf_splitting_type, int ddc_its, double fraction_swap, PetscIS* is_fine, PetscIS* is_coarse)

cpdef py_PCRegister_PFLARE():
	PCRegister_PFLARE()

cpdef compute_cf_splitting(Mat A, bint symmetric, double strong_threshold, int max_luby_steps, int cf_splitting_type, int ddc_its, double fraction_swap):
	cdef IS is_fine
	cdef IS is_coarse
	is_fine = IS()
	is_coarse = IS()
	compute_cf_splitting_c(&(A.mat), symmetric, strong_threshold, max_luby_steps, cf_splitting_type, ddc_its, fraction_swap, &(is_fine.iset), &(is_coarse.iset))
	return is_fine, is_coarse
	