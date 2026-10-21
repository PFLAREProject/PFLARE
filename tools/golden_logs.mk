# ~~~~~~~~~~~~~~~~~
# PFLARE - helper makefile for tools/capture_test_logs.sh
#
# Read after the top level Makefile of the tree being captured:
#   make -C <tree> -f Makefile -f <this file> golden_info
#   make -C <tree> -f Makefile -f <this file> golden_dry_run [GOLDEN_GROUPS="..."]
# so the gates the top Makefile computes and exports (RUN_LOAD_TESTS,
# PETSC_HAVE_KOKKOS, PETSC_HAVE_MPIUNI, PETSC_USE_REAL_SINGLE, ...) are the
# ones the tests/Makefile groups see, exactly as in make tests and make tests_search.
# It is kept separate from the top Makefile so it can also capture a tree
# (e.g. the base commit of a PR) that does not have the golden log targets.
# ~~~~~~~~~~~~~~~~~

# Read a define out of petscconf.h
define _golden_have_conf
$(shell awk '/^[[:space:]]*#define[[:space:]]+$(1)[[:space:]]+1/{print 1; exit}' $(PETSCCONF_H))
endef

# The groups captured by default: everything tests_search scans (which already
# drops the load tests when they can't run) plus the make check commands
GOLDEN_GROUPS ?= $(FILTER_RUN_TARGETS_DEFAULT) run_check

# ~~~~~~~~~~~~~~~~~
# Name of the configuration, used to pick the baseline tests/golden/<config>.json.
# Built to match the CI job names, e.g. gnu_debug, gnu_opt, gnu_opt_64_bit,
# gnu_opt_kokkos, gnu_opt_omp_kokkos, gnu_debug_no_mpi_kokkos, gnu_debug_kokkos_single,
# gnu_opt_no_fortran (PETSc without its Fortran bindings)
# Override with GOLDEN_CONFIG=<name>
# ~~~~~~~~~~~~~~~~~
_GOLDEN_COMPILER_VERSION := $(if $(strip $(FC_VERSION)),$(FC_VERSION),$(C_VERSION))
ifeq ($(shell uname -s 2>/dev/null),Darwin)
_GOLDEN_COMPILER := macos
else ifneq ($(findstring GNU,$(_GOLDEN_COMPILER_VERSION))$(findstring gcc,$(_GOLDEN_COMPILER_VERSION)),)
_GOLDEN_COMPILER := gnu
else ifneq ($(findstring Intel,$(_GOLDEN_COMPILER_VERSION))$(findstring ifx,$(_GOLDEN_COMPILER_VERSION)),)
_GOLDEN_COMPILER := intel
else ifneq ($(findstring clang,$(_GOLDEN_COMPILER_VERSION)),)
_GOLDEN_COMPILER := clang
else
_GOLDEN_COMPILER := other
endif
_GOLDEN_DEBUG := $(if $(call _golden_have_conf,PETSC_USE_DEBUG),debug,opt)
GOLDEN_CONFIG ?= $(_GOLDEN_COMPILER)_$(_GOLDEN_DEBUG)$(if $(filter 1,$(PETSC_USE_64BIT_INDICES)),_64_bit)$(if $(call _golden_have_conf,PETSC_HAVE_OPENMP),_omp)$(if $(filter 1,$(PETSC_HAVE_MPIUNI)),_no_mpi)$(if $(filter 1,$(PETSC_HAVE_KOKKOS)),_kokkos)$(if $(filter 1,$(PETSC_USE_REAL_SINGLE)),_single)$(if $(filter 0,$(PETSC_USE_FORTRAN_BINDINGS)),_no_fortran)

.PHONY: golden_info golden_dry_run

# Describe the configuration as key=value lines
golden_info:
	@echo "config=$(GOLDEN_CONFIG)"
	@echo "petsc_dir=$(PETSC_DIR)"
	@echo "petsc_arch=$(PETSC_ARCH)"
	@echo "compiler=$(strip $(_GOLDEN_COMPILER_VERSION))"
	@echo "debug=$(if $(filter debug,$(_GOLDEN_DEBUG)),1,0)"
	@echo "precision=$(if $(filter 1,$(PETSC_USE_REAL_SINGLE)),single,double)"
	@echo "64bit_indices=$(PETSC_USE_64BIT_INDICES)"
	@echo "kokkos=$(PETSC_HAVE_KOKKOS)"
	@echo "mpiuni=$(PETSC_HAVE_MPIUNI)"
	@echo "fortran_bindings=$(PETSC_USE_FORTRAN_BINDINGS)"
	@echo "groups=$(strip $(GOLDEN_GROUPS))"

# Dry run each group in turn, with a marker line before each so the
# capture script knows which group a command came from
golden_dry_run:
	@for group in $(GOLDEN_GROUPS); do \
		echo "#GOLDEN_GROUP $$group"; \
		$(MAKE) --no-print-directory -C tests -n $$group || exit 1; \
	done
