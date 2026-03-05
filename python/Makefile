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
export PYTHONPATH := $(PETSCPYTHONPATH):$(CURDIR):$(if $(PYTHONPATH),$(PYTHONPATH):)

# Build the Cython interface
python:
	$(PYTHON) setup.py build_ext --inplace

run_tests:
	 @if $(PYTHON) -c "import pflare" >/dev/null 2>&1; then \
		echo ""; \
		echo "Test AIRG with GMRES polynomials for 2D finite difference stencil with Python"; \
		$(PYTHON) ex2.py; \
		echo "Test AIRG with GMRES polynomials for 2D finite difference stencil with Python in parallel"; \
		$(MPIEXEC) -n 2 $(PYTHON) ex2.py; \
		echo ""; \
		echo "Test lAIR with GMRES polynomials for 2D finite difference stencil with Python"; \
		$(PYTHON) ex2.py -pc_air_z_type lair; \
		echo "Test lAIR with GMRES polynomials for 2D finite difference stencil with Python in parallel"; \
		$(MPIEXEC) -n 2 $(PYTHON) ex2.py -pc_air_z_type lair; \
		echo ""; \
		echo "Test single level GMRES polynomial preconditioning with Python"; \
		$(PYTHON) ex2.py -pc_type pflareinv -pc_pflareinv_type power; \
		echo "Test single level GMRES polynomial preconditioning with Python in parallel"; \
		$(MPIEXEC) -n 2 $(PYTHON) ex2.py -pc_type pflareinv -pc_pflareinv_type power; \
		echo ""; \
		echo "Test PMISR DDC CF splitting with Python"; \
		$(PYTHON) ex2_cf_splitting.py; \
		echo "Test PMISR DDC CF splitting with Python in parallel"; \
		$(MPIEXEC) -n 2 $(PYTHON) ex2_cf_splitting.py; \
		echo ""; \
		echo "Test PCAIR direct option set/get API with Python"; \
		$(PYTHON) ex_pcair_options.py && echo "OK"; \
		echo "Test PCAIR direct option set/get API with Python in parallel"; \
		$(MPIEXEC) -n 2 $(PYTHON) ex_pcair_options.py && echo "OK"; \
		echo ""; \
		echo "Test save/restore poly coefficients with PCAIR with Python"; \
		$(PYTHON) ex6f_getcoeffs.py -pc_type air; \
		echo "Test save/restore poly coefficients with PCAIR in parallel with Python"; \
		$(MPIEXEC) -n 2 $(PYTHON) ex6f_getcoeffs.py -pc_type air; \
		echo ""; \
		echo "Test save/restore poly coefficients with PCPFLAREINV with Python"; \
		$(PYTHON) ex6f_getcoeffs.py -pc_type pflareinv; \
		echo "Test save/restore poly coefficients with PCPFLAREINV in parallel with Python"; \
		$(MPIEXEC) -n 2 $(PYTHON) ex6f_getcoeffs.py -pc_type pflareinv; \
		echo "Test save/restore poly coefficients with PCPFLAREINV with Python with Newton"; \
		$(PYTHON) ex6f_getcoeffs.py -pc_type pflareinv -pc_pflareinv_type newton; \
		echo "Test save/restore poly coefficients with PCPFLAREINV in parallel with Python with Newton"; \
		$(MPIEXEC) -n 2 $(PYTHON) ex6f_getcoeffs.py -pc_type pflareinv -pc_pflareinv_type newton; \
	 else \
		echo 'Python tests skipped (PFLARE Python module not built or not found)'; \
	 fi

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_check:
	 @if $(PYTHON) -c "import pflare" >/dev/null 2>&1; then \
		echo "Python check"; \
		$(PYTHON) ex2.py -ksp_max_it 5 -on_error_abort && echo "OK" || (echo "FAIL" && exit 1); \
		echo "Parallel Python check"; \
		$(MPIEXEC) -n 2 $(PYTHON) ex2.py -ksp_max_it 5 -on_error_abort && echo "OK" || (echo "FAIL" && exit 1); \
	 else \
		echo 'Python check skipped (PFLARE Python module not built or not found)'; \
	 fi

# Cleanup
clean::
	$(RM) *.so; $(RM) pflare.c; $(RM) pflare_defs.c; $(RM) -r build/; $(RM) -r __pycache__/
