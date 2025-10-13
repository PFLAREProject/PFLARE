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
ifeq ($(shell uname -s 2>/dev/null),Darwin)
# macOS
export DYLD_FALLBACK_LIBRARY_PATH := $(if $(DYLD_FALLBACK_LIBRARY_PATH),$(DYLD_FALLBACK_LIBRARY_PATH):)$(PETSC_DIR)/$(PETSC_ARCH)/lib:$(LIBDIR)
else
# Required to run the python tests, telling it the location of libpetsc, blas, lapack and libpflare
export LD_LIBRARY_PATH := $(if $(LD_LIBRARY_PATH),$(LD_LIBRARY_PATH):)$(PETSC_DIR)/$(PETSC_ARCH)/lib:$(LIBDIR)
endif
# Required to locate petsc4py and our pflare python module
export PYTHONPATH := $(if $(PYTHONPATH),$(PYTHONPATH):)$(PETSC_DIR)/$(PETSC_ARCH)/lib:$(CURDIR)

# Build the Cython interface
python:
	$(PYTHON) setup.py build_ext --inplace

.ONESHELL:
run_tests:
   # Only run the rest of the recipe if the python module has been built and is importable
	@$(PYTHON) -c "import pflare" 2>/dev/null || { echo 'Python check skipped (PFLARE Python module not built or not found)'; exit 0; }
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

# ~~~~~~~~~~~
# ~~~~~~~~~~~
.ONESHELL:
run_check:

   # Only run the rest of the recipe if the python module has been built and is importable
	@$(PYTHON) -c "import pflare" 2>/dev/null || { echo 'Python check skipped (PFLARE Python module not built or not found)'; exit 0; }

	@echo "Python check"
	@$(PYTHON) ex2.py -ksp_max_it 5 -on_error_abort && echo "OK" || (echo "FAIL" && exit 1)
	@echo "Parallel Python check"
	@$(MPIEXEC) -n 2 $(PYTHON) ex2.py -ksp_max_it 5 -on_error_abort && echo "OK" || (echo "FAIL" && exit 1)	

# Cleanup
clean::
	$(RM) *.so; $(RM) pflare.c; $(RM) pflare_defs.c; $(RM) -r build/; $(RM) -r __pycache__/
