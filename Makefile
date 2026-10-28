# ~~~~~~~~~~~~~~~~~
# PFLARE - Steven Dargaville
# Makefile for PFLARE
#
# Must have defined PETSC_DIR and PETSC_ARCH before calling
# Copied from $PETSC_DIR/share/petsc/Makefile.basic.user
# This uses the compilers and flags defined in the PETSc configuration
# ~~~~~~~~~~~~~~~~~

# Check PETSc version is at least 3.26.0
PETSC_VERSION_MIN := $(shell ${PETSC_DIR}/lib/petsc/bin/petscversion ge 3.26.0)
ifeq ($(PETSC_VERSION_MIN),0)
$(error PETSc version is too old. PFLARE requires at least version 3.26.0)
endif

# PFLARE version - the VERSION.txt file is the single source of truth
# (it has a .txt extension as -I$(CURDIR) is on the include path and a file
# named VERSION shadows the C++ <version> header on case-insensitive macOS)
PFLARE_VERSION := $(strip $(shell cat $(CURDIR)/VERSION.txt))
# The PFLARE_VERSION_* macros in include/pflare.h must be kept in sync with VERSION.txt
PFLARE_VERSION_HEADER := $(strip $(shell awk '/define PFLARE_VERSION_MAJOR/{ma=$$3} /define PFLARE_VERSION_MINOR/{mi=$$3} /define PFLARE_VERSION_SUBMINOR/{su=$$3} END{print ma"."mi"."su}' $(CURDIR)/include/pflare.h))
ifneq ($(PFLARE_VERSION),$(PFLARE_VERSION_HEADER))
$(error VERSION.txt ($(PFLARE_VERSION)) does not match the PFLARE_VERSION_* macros in include/pflare.h ($(PFLARE_VERSION_HEADER)))
endif

# Get the flags we have on input
# These are appended to the flags set by PETSc
# so that users can add their own flags
# but not override the PETSc ones which we use for our builds
CFLAGS_INPUT := $(CFLAGS)
FFLAGS_INPUT := $(FFLAGS)
FPPFLAGS_INPUT := $(FPPFLAGS)
CPPFLAGS_INPUT := $(CPPFLAGS)
CXXPPFLAGS_INPUT := $(CXXPPFLAGS)
CXXFLAGS_INPUT := $(CXXFLAGS)
CUDAC_FLAGS_INPUT := $(CUDAC_FLAGS)
MPICXX_INCLUDES_INPUT := $(MPICXX_INCLUDES)
HIPC_FLAGS_INPUT := $(HIPC_FLAGS)
SYCLC_FLAGS_INPUT := $(SYCLC_FLAGS)

# Directories we want
INCLUDEDIR  := include
SRCDIR      := src
# The Fortran interface layer (the public Fortran modules)
FORTRANDIR  := $(SRCDIR)/fortran
# This needs to be exported into the sub-makefile for Cython
export LIBDIR := $(CURDIR)/lib

# Include directories - include top level directory in case compilers output modules there
INCLUDE := -I$(CURDIR) -I$(INCLUDEDIR)

# Read in the petsc compile/linking variables and makefile rules
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# We then have to add the flags back in after the petsc rules/variables
# have overwritten
override CFLAGS += $(CFLAGS_INPUT) $(INCLUDE)
override FFLAGS += $(FFLAGS_INPUT) $(INCLUDE)
override FPPFLAGS += $(FPPFLAGS_INPUT) $(INCLUDE)
override CPPFLAGS += $(CPPFLAGS_INPUT) $(INCLUDE)
override CXXPPFLAGS += $(CXXPPFLAGS_INPUT) $(INCLUDE)
override CXXFLAGS += $(CXXFLAGS_INPUT) $(INCLUDE)
override CUDAC_FLAGS += $(CUDAC_FLAGS_INPUT) $(INCLUDE)
override MPICXX_INCLUDES += $(MPICXX_INCLUDES_INPUT) $(INCLUDE)
override HIPC_FLAGS += $(HIPC_FLAGS_INPUT) $(INCLUDE)
override SYCLC_FLAGS += $(SYCLC_FLAGS_INPUT) $(INCLUDE)

# ~~~~~~~~~~~~~~~~~~~~~~~~
# Check if petsc has been configured with various options
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Read petscconf.h via awk (portable on macOS)
define _have_conf
$(shell awk '/^[[:space:]]*#define[[:space:]]+$(1)[[:space:]]+1/{print 1; exit}' $(PETSCCONF_H))
endef

export PETSC_USE_64BIT_INDICES := $(if $(call _have_conf,PETSC_USE_64BIT_INDICES),1,0)
export PETSC_USE_SHARED_LIBRARIES := $(if $(call _have_conf,PETSC_USE_SHARED_LIBRARIES),1,0)
export PETSC_HAVE_KOKKOS := $(if $(call _have_conf,PETSC_HAVE_KOKKOS),1,0)
# Detect if PETSc was configured without MPI
export PETSC_HAVE_MPIUNI := $(if $(call _have_conf,PETSC_HAVE_MPIUNI),1,0)
# Detect single-precision PETSc: the tests/data binary matrices are double-width
# and cannot be loaded under PETSC_USE_REAL_SINGLE, so the load tests are gated off
export PETSC_USE_REAL_SINGLE := $(if $(call _have_conf,PETSC_USE_REAL_SINGLE),1,0)
# Detect if PETSc was configured with its Fortran bindings - the Fortran interface
# (src/fortran) and the Fortran tests are only built when it has them. PETSc sets
# this whenever it has a Fortran compiler, unless configured with --with-fortran-bindings=0
export PETSC_USE_FORTRAN_BINDINGS := $(if $(call _have_conf,PETSC_USE_FORTRAN_BINDINGS),1,0)

# The load tests read the double-width binaries in tests/data, so they only run
# when PETSc uses 32-bit indices AND double precision (both flags 0).
RUN_LOAD_TESTS := $(if $(filter 00,$(PETSC_USE_64BIT_INDICES)$(PETSC_USE_REAL_SINGLE)),1,)

# To prevent overlinking with conda builds, only explicitly link 
# to the libraries we use in pflare
ifeq ($(CONDA_BUILD),1)
	PETSC_LINK_LIBS = -L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc ${BLASLAPACK_LIB}
ifeq ($(PETSC_HAVE_KOKKOS),1)
	PETSC_LINK_LIBS += ${KOKKOS_LIB} ${KOKKOS_KERNELS_LIB}
endif	 
# Otherwise just use everything petsc uses to be safe
else
	PETSC_LINK_LIBS = $(LDLIBS)
endif

# On macOS, strip any -Wl,-rpath,* when linking the shared library to avoid duplicate LC_RPATH
ifeq ($(shell uname -s 2>/dev/null),Darwin)
PETSC_LINK_LIBS_NORPATH := $(strip $(foreach w,$(PETSC_LINK_LIBS),$(if $(findstring -Wl,-rpath,$(w)),,$(w))))
else
PETSC_LINK_LIBS_NORPATH := $(PETSC_LINK_LIBS)
endif

# ~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~

# The C objects
COBJS := $(SRCDIR)/C_PETSc_Routines.o \
		  $(SRCDIR)/Pflare_Random.o \
		  $(SRCDIR)/TSQR.o \
		  $(SRCDIR)/Gmres_Poly_Data_Type.o \
		  $(SRCDIR)/Matshell_Data_Type.o \
		  $(SRCDIR)/Gmres_Poly_Apply.o \
		  $(SRCDIR)/Gmres_Poly.o \
		  $(SRCDIR)/Gmres_Poly_Newton.o \
		  $(SRCDIR)/Neumann_Poly.o \
		  $(SRCDIR)/SAI_Z.o \
		  $(SRCDIR)/Approx_Inverse_Setup.o \
		  $(SRCDIR)/Grid_Transfer.o \
		  $(SRCDIR)/Grid_Transfer_Improve.o \
		  $(SRCDIR)/Constrain_Z_or_W.o \
		  $(SRCDIR)/Sorting.o \
		  $(SRCDIR)/Aggregation.o \
		  $(SRCDIR)/Weighted_Jacobi.o \
		  $(SRCDIR)/PETSc_Helper.o \
		  $(SRCDIR)/Repartition.o \
		  $(SRCDIR)/SAbs.o \
		  $(SRCDIR)/MatDiagDom.o \
		  $(SRCDIR)/PMISR_Module.o \
		  $(SRCDIR)/DDC_Module.o \
		  $(SRCDIR)/CR_Splitting.o \
		  $(SRCDIR)/CF_Splitting.o \
		  $(SRCDIR)/MatDiagDomSubmatrix.o \
		  $(SRCDIR)/Timers.o \
		  $(SRCDIR)/AIR_Data_Type.o \
		  $(SRCDIR)/FC_Smooth.o \
		  $(SRCDIR)/FC_Smooth_Block.o \
		  $(SRCDIR)/AIR_Data_Type_Routines.o \
		  $(SRCDIR)/AIR_MG_Stats.o \
		  $(SRCDIR)/AIR_Operators_Setup.o \
		  $(SRCDIR)/AIR_MG_Setup.o \
		  $(SRCDIR)/PCAIR_Data_Type.o \
		  $(SRCDIR)/PCAIR_Shell.o \
		  $(SRCDIR)/PCAIR.o \
		  $(SRCDIR)/PCPFLAREINV.o

# The Kokkos objects (only when PETSc has Kokkos)
KOBJS :=
ifeq ($(PETSC_HAVE_KOKKOS),1)
KOBJS := $(SRCDIR)/PETSc_Helperk.o \
		  $(SRCDIR)/Grid_Transferk.o \
		  $(SRCDIR)/VecISCopyLocalk.o \
		  $(SRCDIR)/Device_Datak.o \
		  $(SRCDIR)/MatDiagDomk.o \
		  $(SRCDIR)/PMISR_Modulek.o \
		  $(SRCDIR)/DDC_Modulek.o \
		  $(SRCDIR)/Gmres_Polyk.o \
		  $(SRCDIR)/Gmres_Poly_Newtonk.o \
		  $(SRCDIR)/SAI_Zk.o
endif

# The Fortran interface layer (only when PETSc has its Fortran bindings)
FOBJS :=
ifeq ($(PETSC_USE_FORTRAN_BINDINGS),1)
FOBJS := $(FORTRANDIR)/Pflare_Parameters.o \
		  $(FORTRANDIR)/Gmres_Poly_Newton.o \
		  $(FORTRANDIR)/CF_Splitting.o \
		  $(FORTRANDIR)/MatDiagDomSubmatrix.o \
		  $(FORTRANDIR)/PCPFLAREINV_Interfaces.o \
		  $(FORTRANDIR)/PCAIR_Interfaces.o \
		  $(FORTRANDIR)/PFLARE.o
endif

# All the files required by PFLARE
# The Fortran list above is in module dependency order, but the build order
# comes from the dependencies in Makefile.deps
OBJS := $(KOBJS) $(FOBJS) $(COBJS)

# Define a variable containing all the tests
export TEST_TARGETS = ex12f \
		  ex6f \
		  ex6f_getcoeffs \
		  ex6_getcoeffs \
		  ex6f_reuse_amount \
		  ex6 \
		  adv_1d \
		  adv_1d_multi_rhs \
		  shell_block_apply \
		  pflareinv_apply_transpose \
		  pcair_apply_transpose \
		  poly_zero_coeffs \
		  newton_imag_roots \
		  newton_zero_roots \
		  newton_scale \
		  pflareinv_exact_inverse \
		  pflareinv_sai_rank_deficient \
		  adv_diff_fd \
		  ex6_cf_splitting \
		  adv_diff_cg_supg \
		  matrandom \
		  matrandom_check_reset \
		  ex12f_gmres_poly \
		  mat_diag \
		  adv_dg_upwind \
		  ex6_two_airg \
		  ilu_factors \
		  reuse_preconditioner \
		  operator_refcount \
		  pcair_reuse_amount_range \
		  comm_self_then_world \
		  pmisr_nonsymmetric \
		  leja_ordering \
		  newton_root_clustering
# Include kokkos examples
ifeq ($(PETSC_HAVE_KOKKOS),1)
export TEST_TARGETS := $(TEST_TARGETS) adv_1dk
endif
# Define a variable containing all the tests that the make check runs
export CHECK_TARGETS = adv_diff_fd matrandom
# The tests written in Fortran are only built when PETSc has its Fortran bindings
# (tests/Makefile skips their commands with SKIP_FORTRAN otherwise)
FORTRAN_TEST_TARGETS := ex12f \
		  ex12f_gmres_poly \
		  ex6f \
		  ex6f_getcoeffs \
		  ex6f_reuse_amount \
		  leja_ordering \
		  mat_diag \
		  matrandom \
		  matrandom_check_reset \
		  newton_root_clustering \
		  pmisr_nonsymmetric
ifeq ($(PETSC_USE_FORTRAN_BINDINGS),0)
export TEST_TARGETS := $(filter-out $(FORTRAN_TEST_TARGETS),$(TEST_TARGETS))
export CHECK_TARGETS := $(filter-out $(FORTRAN_TEST_TARGETS),$(CHECK_TARGETS))
endif

# Output the library - either static or dynamic
ifeq ($(PETSC_USE_SHARED_LIBRARIES),0)
OUT = $(LIBDIR)/libpflare.a
else
# mac osx name is different
ifeq ($(shell uname -s 2>/dev/null),Darwin)
OUT = $(LIBDIR)/libpflare.dylib
else
OUT = $(LIBDIR)/libpflare.so
endif
endif

# ~~~~~~~~~~~~~~~~~~~~~~~~
# Dependency generation for the parallel build - make depend writes Makefile.deps
# The Fortran module dependencies come from makedepf90 and the C/C++ header
# dependencies from the compilers (-MM). Run it in a build with Kokkos and the
# Fortran bindings so the Kokkos and Fortran objects are included (without the
# Fortran bindings the Fortran lines already in Makefile.deps are kept)
# ~~~~~~~~~~~~~~~~~~~~~~~~
DEPFILE = Makefile.deps

# Find Fortran source files in the src directories for makedepf90
FSRC_SUBDIRS := $(sort $(wildcard $(FORTRANDIR)/*.F90))
FSRC := $(sort $(wildcard $(SRCDIR)/*.f90) $(wildcard $(SRCDIR)/*.F90)) $(FSRC_SUBDIRS)
# makedepf90 has to see all the Fortran at once to find modules used across
# directories, but it puts every object in the -b directory, so the objects
# of the subdirectories are moved back into their own directory afterwards
FDEP_RELOCATE := $(foreach f,$(basename $(FSRC_SUBDIRS)),-e 's|$(SRCDIR)/$(notdir $(f))\.o |$(f).o |g')

# Find C and Kokkos source files for the compilers
CSRC := $(sort $(wildcard $(SRCDIR)/*.c))
KSRC := $(if $(filter 1,$(PETSC_HAVE_KOKKOS)),$(sort $(wildcard $(SRCDIR)/*.kokkos.cxx)))
# PFLARE's headers are found through relative paths and PETSc's (with those of
# MPI and Kokkos) are system headers, so -MM lists only PFLARE's own headers
DEP_CPPFLAGS := -I. -I$(INCLUDEDIR) $(patsubst -I%,-isystem %,$(PETSC_CC_INCLUDES)) $(PETSCFLAGS)
# Writes each compiler rule on one line in the makedepf90 format and drops any
# prerequisite with an absolute path (e.g. headers from a compiler wrapper)
DEP_TIDY := awk '{ line = line $$0 } /\\$$/ { sub(/\\$$/, "", line); next } \
	{ n = split(line, w, " "); printf "%s :", substr(w[1], 1, length(w[1]) - 1); \
	for (i = 2; i <= n; i++) if (substr(w[i], 1, 1) != "/") printf " %s", w[i]; \
	printf " \n"; line = "" }'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Rules
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.DEFAULT_GOAL := all		  	
all: $(OUT)

# The shared library is linked with the C linker, or the C++ linker when the
# Kokkos (C++) objects are in it, so no Fortran linker is needed. When the
# Fortran interface is built, the Fortran runtime it needs is on PETSc's link
# line (LDLIBS), as PETSc with Fortran bindings links it for its own Fortran
ifeq ($(PETSC_HAVE_KOKKOS),1)
LINK_SHARED = $(CXXLINKER) $(LDFLAGS)
else
LINK_SHARED = $(CLINKER) $(LDFLAGS)
endif

# # Create our directory structure and build the library
# # (either static or dynamic depending on what petsc was configured with)
$(OUT): $(OBJS)
	@mkdir -p $(LIBDIR)
	@mkdir -p $(INCLUDEDIR)
ifeq ($(PETSC_USE_SHARED_LIBRARIES),0)	
	$(AR) $(AR_FLAGS) $(OUT) $(OBJS)
	$(RANLIB) $(OUT)
else
ifeq ($(shell uname -s 2>/dev/null),Darwin)
# macOS: Use -dynamiclib and set a relocatable @rpath install_name. Do not embed rpaths.
	$(LINK_SHARED) -dynamiclib -o $(OUT) $(OBJS) $(PETSC_LINK_LIBS_NORPATH) -install_name @rpath/$(notdir $(OUT))
else	
# Linux: Use -shared and set the soname.
	$(LINK_SHARED) -shared -o $(OUT) $(OBJS) $(PETSC_LINK_LIBS) -Wl,-soname,$(notdir $(OUT))
endif
endif

# Generate dependencies for parallel build
# The .c files are compiled with PETSc's PCC (the C++ compiler if PETSc was
# configured with --with-clanguage=cxx) and the Kokkos files with CXX
.PHONY: depend
depend:
	@echo "Generating dependencies..."
ifeq ($(PETSC_USE_FORTRAN_BINDINGS),1)
	@makedepf90 -b $(SRCDIR) $(FSRC) > $(DEPFILE).tmp
	@$(if $(FDEP_RELOCATE),sed $(FDEP_RELOCATE),cat) $(DEPFILE).tmp > $(DEPFILE)
else
	@echo "PETSc has no Fortran bindings: keeping the Fortran lines of $(DEPFILE)"
	@grep '^$(FORTRANDIR)/' $(DEPFILE) > $(DEPFILE).tmp 2>/dev/null || true
	@mv $(DEPFILE).tmp $(DEPFILE)
endif
	@for f in $(CSRC); do \
		$(PCC) -MM -MT $${f%.c}.o $(DEP_CPPFLAGS) $$f > $(DEPFILE).tmp || exit 1; \
		$(DEP_TIDY) $(DEPFILE).tmp >> $(DEPFILE); \
	done
	@for f in $(KSRC); do \
		$(CXX) -MM -MT $${f%.kokkos.cxx}.o $(CXX_FLAGS) $(DEP_CPPFLAGS) $$f > $(DEPFILE).tmp || exit 1; \
		$(DEP_TIDY) $(DEPFILE).tmp >> $(DEPFILE); \
	done
	@$(RM) $(DEPFILE).tmp

# Include dependencies if the file exists
-include $(DEPFILE)

# Build the tests (in parallel)
.PHONY: build_tests
build_tests: $(OUT)
	+$(MAKE) -C tests $(TEST_TARGETS)

# Build the tests used in the check
.PHONY: build_tests_check
build_tests_check: $(OUT)
	+$(MAKE) -C tests $(CHECK_TARGETS)	

# Separate out the different test cases
.PHONY: tests_short_serial
tests_short_serial: build_tests
	$(MAKE) -C tests run_tests_no_load_short_serial

.PHONY: tests_short_parallel
tests_short_parallel: build_tests
	$(MAKE) -C tests run_tests_no_load_short_parallel

# Only run the tests that load the 32 bit test matrix in /tests/data
# if PETSC has been configured without 64 bit integers
.PHONY: tests_load_serial
tests_load_serial: build_tests
ifeq ($(RUN_LOAD_TESTS),1)
	$(MAKE) -C tests run_tests_load_serial
endif

.PHONY: tests_load_parallel
tests_load_parallel: build_tests
ifeq ($(RUN_LOAD_TESTS),1)
	$(MAKE) -C tests run_tests_load_parallel
endif

.PHONY: tests_medium_serial
tests_medium_serial: build_tests
	$(MAKE) -C tests run_tests_medium_serial

.PHONY: tests_medium_parallel
tests_medium_parallel: build_tests
	$(MAKE) -C tests run_tests_medium_parallel		

# Default target groups scanned by tests_search.
# Keep load tests only when PETSc uses 32-bit indices and double precision.
ifeq ($(RUN_LOAD_TESTS),1)
FILTER_RUN_TARGETS_DEFAULT := run_tests_load_serial run_tests_load_parallel run_tests_no_load_short_serial run_tests_no_load_short_parallel run_tests_no_load_serial run_tests_no_load_parallel run_tests_medium_serial run_tests_medium_parallel
else
FILTER_RUN_TARGETS_DEFAULT := run_tests_no_load_short_serial run_tests_no_load_short_parallel run_tests_no_load_serial run_tests_no_load_parallel run_tests_medium_serial run_tests_medium_parallel
endif

# Run only tests whose command lines contain TEST_MATCH.
# Example: make tests_search TEST_MATCH="-curved_velocity"
.PHONY: tests_search
tests_search: build_tests
	$(MAKE) -C tests run_tests_search TEST_MATCH="$(TEST_MATCH)" FILTER_RUN_TARGETS="$(if $(strip $(FILTER_RUN_TARGETS)),$(FILTER_RUN_TARGETS),$(FILTER_RUN_TARGETS_DEFAULT))"

# Very quick tests
.PHONY: tests_short
tests_short: build_tests
	$(MAKE) tests_short_serial
	$(MAKE) tests_short_parallel

# Longer tests
.PHONY: tests_medium
tests_medium: build_tests
	$(MAKE) tests_medium_serial
	$(MAKE) tests_medium_parallel

# Host/device copy checks - only meaningful on a GPU, so run from the GPU CI
# workflow and deliberately not part of make tests
.PHONY: tests_copies
tests_copies: build_tests
	$(MAKE) -C tests run_tests_copies

# Build and run all the tests
# The python tests only run if the python module has been built
.PHONY: tests
tests: build_tests
	($(MAKE) tests_short || (echo "Short tests failed" && exit 1)) && \
	($(MAKE) tests_load_serial || (echo "Load serial tests failed" && exit 1)) && \
	($(MAKE) tests_load_parallel || (echo "Load parallel tests failed" && exit 1)) && \
	($(MAKE) -C tests run_tests_no_load_serial || (echo "No-load serial tests failed" && exit 1)) && \
	($(MAKE) -C tests run_tests_no_load_parallel || (echo "No-load parallel tests failed" && exit 1)) && \
	($(MAKE) tests_medium || (echo "Medium tests failed" && exit 1)) && \
	($(MAKE) tests_python || (echo "Python tests failed" && exit 1)) && \
	echo "All tests passed: OK"

# ~~~~~~~~~~~~~~~~~~~~~~~~
# Golden logs - see docs/dev/testing.md
# tests_golden_capture runs every test command with -ksp_converged_reason -ksp_monitor
# and keeps a log per command in GOLDEN_DIR
# tests_golden also compares that capture against a locally produced baseline
# GOLDEN_BASELINE_DIR/<config>.json (see docs/dev/testing.md; baselines are not committed)
# GOLDEN_JOBS runs that many test commands at once
# ~~~~~~~~~~~~~~~~~~~~~~~~
GOLDEN_DIR ?= $(CURDIR)/tests/golden_logs
GOLDEN_JOBS ?= 1
GOLDEN_BASELINE_DIR ?= $(CURDIR)/tests/golden

.PHONY: tests_golden_capture
tests_golden_capture: build_tests
	MAKE="$(MAKE)" $(CURDIR)/tools/capture_test_logs.sh -j $(GOLDEN_JOBS) $(GOLDEN_DIR)

.PHONY: tests_golden
tests_golden: build_tests
	-MAKE="$(MAKE)" $(CURDIR)/tools/capture_test_logs.sh -j $(GOLDEN_JOBS) $(GOLDEN_DIR)
	@config="$$(sed -n 's/^config=//p' $(GOLDEN_DIR)/meta.txt)"; \
	baseline="$(GOLDEN_BASELINE_DIR)/$$config.json"; \
	if [ ! -f "$$baseline" ]; then \
		echo "ERROR: no golden baseline $$baseline for configuration $$config"; \
		exit 1; \
	fi; \
	echo "Comparing against $$baseline"; \
	$(PYTHON) $(CURDIR)/tools/compare_test_logs.py compare "$$baseline" $(GOLDEN_DIR)

# A quick sanity check with simple tests
.PHONY: check
check: build_tests_check
	@$(MAKE) --no-print-directory -C tests run_check
	@$(MAKE) --no-print-directory -C python run_check

# Build the Python module with Cython
.PHONY: python
python: $(OUT)
	$(MAKE) -C python python

# Run the python tests
.PHONY: tests_python
tests_python: $(OUT)
	$(MAKE) -C python run_tests

# Cleanup
clean::
	$(RM) -r $(LIBDIR)
	$(RM) $(INCLUDEDIR)/*.mod
	$(RM) $(SRCDIR)/*.mod
	$(RM) $(SRCDIR)/*.o
	$(RM) $(FORTRANDIR)/*.mod
	$(RM) $(FORTRANDIR)/*.o
	$(RM) $(CURDIR)/*.mod
	$(MAKE) -C tests clean
	$(MAKE) -C python clean	

# Make install
# Default install location which can be overridden by user
PREFIX        ?= /usr/local
INCLUDEDIR_INSTALL := $(DESTDIR)$(PREFIX)/include
LIBDIR_INSTALL     := $(DESTDIR)$(PREFIX)/lib
# Writes out a pkg-config file
PKGCONFIGDIR_INSTALL := $(LIBDIR_INSTALL)/pkgconfig
INSTALL ?= install
INSTALL_DATA = $(INSTALL) -m 644
INSTALL_PROG = $(INSTALL) -m 755

.PHONY: install install_python
install: all
	@echo "==> Installing PFLARE into $(DESTDIR)$(PREFIX)"
	@$(INSTALL) -d $(INCLUDEDIR_INSTALL) $(LIBDIR_INSTALL) $(PKGCONFIGDIR_INSTALL)
	@if [ -f $(OUT) ]; then \
		$(INSTALL_PROG) $(OUT) $(LIBDIR_INSTALL)/; \
	else \
		echo "ERROR: Library $(OUT) not found"; exit 1; \
	fi
	@if [ -d include ]; then \
		cd include && \
		find . -type d ! -name '.' -exec $(INSTALL) -d "$(INCLUDEDIR_INSTALL)/{}" \; && \
		find . -type f -name "*.h" -exec $(INSTALL_DATA) "{}" "$(INCLUDEDIR_INSTALL)/{}" \; ; \
	fi
ifeq ($(PETSC_USE_FORTRAN_BINDINGS),1)
	@for m in *.mod lib/*.mod; do \
		if [ -f "$$m" ]; then $(INSTALL_DATA) "$$m" $(INCLUDEDIR_INSTALL)/; fi; \
	done
endif
	@$(MAKE) --no-print-directory install_python PREFIX="$(PREFIX)" DESTDIR="$(DESTDIR)"
	@printf 'prefix=%s\nexec_prefix=$${prefix}\nlibdir=$${prefix}/lib\nincludedir=$${prefix}/include\n\nName: pflare\nDescription: Library with parallel iterative methods for asymmetric linear systems built on PETSc.\nVersion: %s\nCflags: -I$${includedir}\nLibs: -L$${libdir} -lpflare\nRequires: petsc\n' "$(PREFIX)" "$(PFLARE_VERSION)" > $(PKGCONFIGDIR_INSTALL)/pflare.pc
	@echo '==> Install complete'

install_python:
	@if ls python/pflare_defs.cpython-*.so >/dev/null 2>&1; then \
		PYVER=$${PYVER:-$$(python -c "import sys;print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>/dev/null || echo "3.11")}; \
		SITELIB_INSTALL="$(DESTDIR)$(PREFIX)/lib/python$${PYVER}/site-packages"; \
		$(INSTALL) -d "$$SITELIB_INSTALL"; \
		$(INSTALL_PROG) python/pflare_defs.cpython-*.so "$$SITELIB_INSTALL/"; \
		if [ -f python/pflare.py ]; then $(INSTALL_DATA) python/pflare.py "$$SITELIB_INSTALL/"; fi; \
		echo "==> Installed Python bindings to $$SITELIB_INSTALL"; \
	else \
		echo "==> Python extension not built, skipping Python install"; \
	fi
