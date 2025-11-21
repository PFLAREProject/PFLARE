# ~~~~~~~~~~~~~~~~~
# PFLARE - Steven Dargaville
# Makefile for PFLARE
#
# Must have defined PETSC_DIR and PETSC_ARCH before calling
# Copied from $PETSC_DIR/share/petsc/Makefile.basic.user
# This uses the compilers and flags defined in the PETSc configuration
# ~~~~~~~~~~~~~~~~~

# Check PETSc version is at least 3.24.1
PETSC_VERSION_MIN := $(shell ${PETSC_DIR}/lib/petsc/bin/petscversion ge 3.24.1)
ifeq ($(PETSC_VERSION_MIN),0)
$(error PETSc version is too old. PFLARE requires at least version 3.24.1)
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

# All the files required by PFLARE
OBJS := $(SRCDIR)/Binary_Tree.o \
		  $(SRCDIR)/TSQR.o \
		  $(SRCDIR)/Gmres_Poly_Data_Type.o \
		  $(SRCDIR)/AIR_Data_Type.o \
		  $(SRCDIR)/Matshell_Data_Type.o \
		  $(SRCDIR)/Matshell_PFLARE.o \
		  $(SRCDIR)/Sorting.o \
		  $(SRCDIR)/C_PETSc_Interfaces.o \
		  $(SRCDIR)/PCPFLAREINV_Interfaces.o \
		  $(SRCDIR)/PCAIR_Data_Type.o

# Include kokkos src files
ifeq ($(PETSC_HAVE_KOKKOS),1)
export OBJS := $(OBJS) $(SRCDIR)/PETSc_Helperk.o \
							  $(SRCDIR)/Grid_Transferk.o \
							  $(SRCDIR)/VecISCopyLocalk.o \
							  $(SRCDIR)/PMISR_DDCk.o \
							  $(SRCDIR)/Gmres_Polyk.o
endif	

OBJS := $(OBJS) $(SRCDIR)/PETSc_Helper.o \
		  $(SRCDIR)/FC_Smooth.o \
		  $(SRCDIR)/Gmres_Poly.o \
		  $(SRCDIR)/Gmres_Poly_Newton.o \
		  $(SRCDIR)/AIR_MG_Stats.o \
		  $(SRCDIR)/SAI_Z.o \
		  $(SRCDIR)/Constrain_Z_or_W.o \
		  $(SRCDIR)/PMISR_DDC.o \
		  $(SRCDIR)/Aggregation.o \
		  $(SRCDIR)/CF_Splitting.o \
		  $(SRCDIR)/Repartition.o \
		  $(SRCDIR)/Timers.o \
		  $(SRCDIR)/Weighted_Jacobi.o \
		  $(SRCDIR)/Neumann_Poly.o \
		  $(SRCDIR)/Approx_Inverse_Setup.o \
		  $(SRCDIR)/AIR_Data_Type_Routines.o \
		  $(SRCDIR)/Grid_Transfer.o \
		  $(SRCDIR)/Grid_Transfer_Improve.o \
		  $(SRCDIR)/AIR_Operators_Setup.o \
		  $(SRCDIR)/AIR_MG_Setup.o \
		  $(SRCDIR)/PCAIR_Shell.o \
		  $(SRCDIR)/PCAIR_Interfaces.o \
		  $(SRCDIR)/PFLARE.o \
		  $(SRCDIR)/C_PETSc_Routines.o \
		  $(SRCDIR)/C_Fortran_Bindings.o \
		  $(SRCDIR)/PCAIR_C_Fortran_Bindings.o \
		  $(SRCDIR)/PCAIR.o \
		  $(SRCDIR)/PCPFLAREINV.o	

# Define a variable containing all the tests
export TEST_TARGETS = ex12f ex6f ex6f_getcoeffs ex6 adv_1d adv_diff_2d ex6_cf_splitting adv_diff_cg_supg matrandom
# Include kokkos examples
ifeq ($(PETSC_HAVE_KOKKOS),1)
export TEST_TARGETS := $(TEST_TARGETS) adv_1dk
endif
# Define a variable containing all the tests that the make check runs
export CHECK_TARGETS = adv_diff_2d matrandom

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

# Dependency generation with makedepf90
DEPFILE = Makefile.deps

# Find Fortran source files in the src directory for makedepf90
FSRC := $(wildcard $(SRCDIR)/*.f90) $(wildcard $(SRCDIR)/*.F90)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Rules
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.DEFAULT_GOAL := all		  	
all: $(OUT)

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
	$(LINK.F) -dynamiclib -o $(OUT) $(OBJS) $(PETSC_LINK_LIBS_NORPATH) -install_name @rpath/$(notdir $(OUT))
else	
# Linux: Use -shared and set the soname.
	$(LINK.F) -shared -o $(OUT) $(OBJS) $(PETSC_LINK_LIBS) -Wl,-soname,$(notdir $(OUT))
endif
endif

# Generate dependencies for parallel build with makedepf90
.PHONY: depend
depend:
	@echo "Generating dependencies..."
	@makedepf90 -b $(SRCDIR) $(FSRC) > $(DEPFILE)

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
# Only run the tests that load the 32 bit test matrix in /tests/data
# if PETSC has been configured without 64 bit integers
.PHONY: tests_short_serial
tests_short_serial: build_tests
ifeq ($(PETSC_USE_64BIT_INDICES),0)
	$(MAKE) -C tests run_tests_load_serial
endif	
	$(MAKE) -C tests run_tests_no_load_serial

.PHONY: tests_short_parallel
tests_short_parallel: build_tests
ifeq ($(PETSC_USE_64BIT_INDICES),0)
	$(MAKE) -C tests run_tests_load_parallel
endif	
	$(MAKE) -C tests run_tests_no_load_parallel	

.PHONY: tests_medium_serial
tests_medium_serial: build_tests
	$(MAKE) -C tests run_tests_medium_serial

.PHONY: tests_medium_parallel
tests_medium_parallel: build_tests
	$(MAKE) -C tests run_tests_medium_parallel		

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

# Build and run all the tests
# The python tests only run if the python module has been built
.PHONY: tests
tests: build_tests
	($(MAKE) tests_short || (echo "Short tests failed" && exit 1)) && \
	($(MAKE) tests_medium || (echo "Medium tests failed" && exit 1)) && \
	($(MAKE) tests_python || (echo "Python tests failed" && exit 1)) && \
	echo "All tests passed: OK"

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
	@for m in *.mod lib/*.mod; do \
		if [ -f "$$m" ]; then $(INSTALL_DATA) "$$m" $(INCLUDEDIR_INSTALL)/; fi; \
	done
	@$(MAKE) --no-print-directory install_python PREFIX="$(PREFIX)" DESTDIR="$(DESTDIR)"
	@printf 'prefix=%s\nexec_prefix=$${prefix}\nlibdir=$${prefix}/lib\nincludedir=$${prefix}/include\n\nName: pflare\nDescription: Library with parallel iterative methods for asymmetric linear systems built on PETSc.\nVersion: 1.24.9\nCflags: -I$${includedir}\nLibs: -L$${libdir} -lpflare\nRequires: petsc\n' "$(PREFIX)" > $(PKGCONFIGDIR_INSTALL)/pflare.pc
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
