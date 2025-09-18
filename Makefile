# ~~~~~~~~~~~~~~~~~
# PFLARE - Steven Dargaville
# Makefile for PFLARE
#
# Must have defined PETSC_DIR and PETSC_ARCH before calling
# Copied from $PETSC_DIR/share/petsc/Makefile.basic.user
# This uses the compilers and flags defined in the PETSc configuration
# ~~~~~~~~~~~~~~~~~

# Check PETSc version is at least 3.23.1
PETSC_VERSION_MIN := $(shell ${PETSC_DIR}/lib/petsc/bin/petscversion ge 3.23.1)
ifeq ($(PETSC_VERSION_MIN),0)
$(error PETSc version is too old. PFLARE requires at least version 3.23.1)
endif

# Get the flags we have on input
CFLAGS_INPUT := $(CFLAGS)
FFLAGS_INPUT := $(FFLAGS)
CPPFLAGS_INPUT := $(CPPFLAGS)
CXXFLAGS_INPUT := $(CXXFLAGS)
FPPFLAGS_INPUT := $(FPPFLAGS)
CUDAC_FLAGS_INPUT := $(CUDAC_FLAGS)
HIPC_FLAGS_INPUT := $(HIPC_FLAGS)
SYCLC_FLAGS_INPUT := $(SYCLC_FLAGS)

# Directories we want
INCLUDEDIR  := include
SRCDIR      := src
# This needs to be exported into the sub-makefile for Cython
export LIBDIR := $(CURDIR)/lib

# Include directories - include top level directory in case compilers output modules there
INCLUDE := -I$(CURDIR) -I$(INCLUDEDIR)

CPPFLAGS = $(INCLUDE)
FPPFLAGS = $(INCLUDE)
CPPFLAGS = $(INCLUDE)
CXXPPFLAGS = $(INCLUDE)

# Read in the petsc compile/linking variables and makefile rules
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# ~~~~~~~~~~~~~~~~~~~~~~~~
# Check if petsc has been configured with various options
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Read in the petscconf.h
CONTENTS := $(file < $(PETSCCONF_H))
export PETSC_USE_64BIT_INDICES := 0
ifneq (,$(findstring PETSC_USE_64BIT_INDICES 1,$(CONTENTS)))
PETSC_USE_64BIT_INDICES := 1
endif  		  
PETSC_USE_SHARED_LIBRARIES := 0
ifneq (,$(findstring PETSC_USE_SHARED_LIBRARIES 1,$(CONTENTS)))
PETSC_USE_SHARED_LIBRARIES := 1
endif
export PETSC_HAVE_KOKKOS := 0
ifneq (,$(findstring PETSC_HAVE_KOKKOS 1,$(CONTENTS)))
export PETSC_HAVE_KOKKOS := 1
endif

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
export TEST_TARGETS = ex12f ex6f ex6f_getcoeffs ex6 adv_1d adv_diff_2d ex6_cf_splitting adv_diff_cg_supg
# Include kokkos examples
ifeq ($(PETSC_HAVE_KOKKOS),1)
export TEST_TARGETS := $(TEST_TARGETS) adv_1dk
endif
# Define a variable containing all the tests that the make check runs
export CHECK_TARGETS = adv_diff_2d

# Include any additional flags we input
CFLAGS += $(CFLAGS_INPUT)
FFLAGS += $(FFLAGS_INPUT)
CPPFLAGS += $(CPPFLAGS_INPUT)
CXXFLAGS += $(CXXFLAGS_INPUT)
FPPFLAGS += $(FPPFLAGS_INPUT)
CUDAC_FLAGS += $(CUDAC_FLAGS_INPUT)
HIPC_FLAGS += $(HIPC_FLAGS_INPUT)
SYCLC_FLAGS += $(SYCLC_FLAGS_INPUT)

# Output the library - either static or dynamic
ifeq ($(PETSC_USE_SHARED_LIBRARIES),0)
OUT = $(LIBDIR)/libpflare.a
else
OUT = $(LIBDIR)/libpflare.so
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
ifeq ($(findstring darwin,$(PETSC_ARCH)),darwin)
   # macOS: Use -dynamiclib and set a relocatable @rpath install_name.
	$(LINK.F) -dynamiclib -o $(OUT) $(OBJS) $(PETSC_LINK_LIBS) -install_name @rpath/$(notdir $(OUT))
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
