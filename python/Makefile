# ~~~~~~~~~~~~~~~~~
# PFLARE - Steven Dargaville
# Makefile for python interface
# Copied from $PETSC_DIR/share/petsc/Makefile.basic.user
# This uses the compilers and flags defined in the PETSc configuration
# ~~~~~~~~~~~~~~~~~

# Can be overridden from the command line if needed
PYTHON ?= python3
CFLAGS   =
FFLAGS   =
CPPFLAGS =
FPPFLAGS =

# Read in the petsc compile/linking variables and makefile rules
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# Make sure the setup.py can see the petsc compilers
export CC
export CXX
export FC

# Required during compilation to tell Cython where to look for libpflare
export LIBRARY_PATH := $(if $(LIBRARY_PATH),$(LIBRARY_PATH):)$(LIBDIR)
# Required to run the python tests, telling it the location of libpetsc, blas, lapack and libpflare
export LD_LIBRARY_PATH := $(if $(LD_LIBRARY_PATH),$(LD_LIBRARY_PATH):)$(PETSC_DIR)/$(PETSC_ARCH)/lib:$(LIBDIR)
# Required to locate petsc4py and our pflare python module
export PYTHONPATH := $(if $(PYTHONPATH),$(PYTHONPATH):)$(PETSC_DIR)/$(PETSC_ARCH)/lib:$(CURDIR)

# Build the Cython interface
python:
	$(PYTHON) setup.py build_ext --inplace

run_tests:
# Only run if the python module has been built (i.e. a .so file exists)
ifneq ($(wildcard *.so),)
#
	@echo ""
	@echo "Test AIRG with GMRES polynomials for 2D finite difference stencil with Python"
	$(PYTHON) ex2.py
	@echo "Test AIRG with GMRES polynomials for 2D finite difference stencil with Python in parallel"
	$(MPIEXEC) -n 2 $(PYTHON) ex2.py
#
	@echo ""
	@echo "Test lAIR with GMRES polynomials for 2D finite difference stencil with Python"
	$(PYTHON) ex2.py -pc_air_z_type lair
	@echo "Test lAIR with GMRES polynomials for 2D finite difference stencil with Python in parallel"
	$(MPIEXEC) -n 2 $(PYTHON) ex2.py -pc_air_z_type lair
#
	@echo ""
	@echo "Test single level GMRES polynomial preconditioning with Python"
	$(PYTHON) ex2.py -pc_type pflareinv -pc_pflareinv_type power
	@echo "Test single level GMRES polynomial preconditioning with Python in parallel"
	$(MPIEXEC) -n 2 $(PYTHON) ex2.py -pc_type pflareinv -pc_pflareinv_type power
#
	@echo ""
	@echo "Test PMISR DDC CF splitting with Python"
	$(PYTHON) ex2_cf_splitting.py
	@echo "Test PMISR DDC CF splitting with Python in parallel"
	$(MPIEXEC) -n 2 $(PYTHON) ex2_cf_splitting.py	
else
	@echo "Python tests skipped (PFLARE Python module has not been built)"
endif		

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_check:
# Only run if the python module has been built (i.e. a .so file exists)
ifneq ($(wildcard *.so),)
	@echo "Python check"
	@$(PYTHON) ex2.py -ksp_max_it 5&& echo "OK" || (echo "FAIL" && exit 1)
	@echo "Parallel Python check"
	@$(MPIEXEC) -n 2 $(PYTHON) ex2.py -ksp_max_it 5 && echo "OK" || (echo "FAIL" && exit 1)	
else
	@echo "Python check skipped (PFLARE Python module has not been built)"
endif	

# Cleanup
clean::
	$(RM) *.so; $(RM) pflare.c; $(RM) pflare_defs.c; $(RM) -r build/; $(RM) -r __pycache__/
