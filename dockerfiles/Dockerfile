ARG BASE_IMAGE=stevendargaville/petsc

FROM ${BASE_IMAGE}
ARG PETSC_DIR=/build/petsc
# Use the optimised petsc build by default
ARG ARCH=arch-linux-c-opt
# Don't use malloc dump by default
ARG MALLOC_DUMP=false
# Extra Fortran flags appended to the defaults, e.g. "-fcheck=all -g" for runtime checking
ARG EXTRA_FFLAGS=""
# Extra C and C++ flags appended to the defaults, e.g. "-fsanitize=address,undefined"
# (then also add the sanitizer to EXTRA_FFLAGS, so the Fortran tests link its runtime)
ARG EXTRA_CFLAGS=""
ARG EXTRA_CXXFLAGS=""
# Sanitizer runtime options for a sanitizer build, e.g. ASAN_OPTIONS=detect_leaks=0
# (build args are in the environment of the RUN below, but not kept in the image)
ARG ASAN_OPTIONS=""
ARG UBSAN_OPTIONS=""

LABEL maintainer="Steven Dargaville"
LABEL description="PFLARE"

ENV PETSC_ARCH=$ARCH
ENV PETSC_DIR=$PETSC_DIR
# -on_error_abort ensures any test failures are caught and the build fails
# fp trap turns on floating point exception trapping in petsc
ENV PETSC_OPTIONS="-on_error_abort -fp_trap on"

WORKDIR /build

COPY . /build/PFLARE

# Ensure pipelines fail if any stage fails
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Build and test PFLARE
# Set the line length to 132 to check for truncation problems - the cray fortran
# compiler was defaulting to 132 and we don't have a CI for that, so this will trigger it
# If MALLOC_DUMP is true:
# 1) Remove -on_error_abort and strip it from the test Makefiles as it interferes with the output
# 2) Output the terminal to a log file
# 3) Check the log for any PetscTrMalloc lines which indicate memory leaks
# The Python interpreter is not built with the sanitizers, so it can't load a libpflare
# built with them (without an LD_PRELOAD of the sanitizer runtime), so a sanitizer build
# skips the Python bindings and their tests (the gnu_opt job tests them)
RUN set -euo pipefail; \
    if [ "${MALLOC_DUMP}" = "true" ]; then export PETSC_OPTIONS="-malloc_dump"; fi; \
    echo "Using PETSC_OPTIONS=${PETSC_OPTIONS} ASAN_OPTIONS=${ASAN_OPTIONS} UBSAN_OPTIONS=${UBSAN_OPTIONS}"; \
    cd PFLARE && \
    if [ "${MALLOC_DUMP}" = "true" ]; then \
        echo "Stripping -on_error_abort from test Makefiles for malloc dump run"; \
        sed -i 's/-on_error_abort//g' tests/Makefile; \
        sed -i 's/-on_error_abort//g' python/Makefile; \
    fi && \
    if [ "${MALLOC_DUMP}" = "true" ]; then LOG=/tmp/test.log; : > "$LOG"; else LOG=/dev/null; fi; \
    make -j2 FFLAGS="-Werror -Wall -ffree-line-length-132 -ffixed-line-length-132 ${EXTRA_FFLAGS}" CFLAGS="-Werror -Wall ${EXTRA_CFLAGS}" CXXFLAGS="-Werror -Wall ${EXTRA_CXXFLAGS}" CPPFLAGS="-Werror -Wall" 2>&1 | tee -a "$LOG"; \
    if [[ "${EXTRA_CFLAGS}" == *-fsanitize* ]]; then \
        echo "Sanitizer build: not building the Python bindings"; \
    else \
        make python 2>&1 | tee -a "$LOG"; \
    fi; \
    make -j2 check FFLAGS="-Werror -Wall -ffree-line-length-132 -ffixed-line-length-132 ${EXTRA_FFLAGS}" CFLAGS="-Werror -Wall ${EXTRA_CFLAGS}" CXXFLAGS="-Werror -Wall ${EXTRA_CXXFLAGS}" CPPFLAGS="-Werror -Wall" 2>&1 | tee -a "$LOG"; \
    make -j2 tests FFLAGS="-Werror -Wall -ffree-line-length-132 -ffixed-line-length-132 ${EXTRA_FFLAGS}" CFLAGS="-Werror -Wall ${EXTRA_CFLAGS}" CXXFLAGS="-Werror -Wall ${EXTRA_CXXFLAGS}" CPPFLAGS="-Werror -Wall" 2>&1 | tee -a "$LOG"; \
    if [ "${MALLOC_DUMP}" = "true" ]; then \
      if grep -q "PetscTrMalloc" "$LOG"; then \
        echo "Error: found PetscTrMalloc in test output" >&2; exit 1; \
      fi; \
    fi

WORKDIR /build/PFLARE
