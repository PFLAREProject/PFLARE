ARG BASE_IMAGE=stevendargaville/petsc

FROM ${BASE_IMAGE}
# Checkout the main branch by default
ARG BRANCH=main
# Use the optimised petsc build by default
ARG ARCH=arch-linux-c-opt
# Don't use malloc dump by default
ARG MALLOC_DUMP=false

LABEL maintainer="Steven Dargaville"
LABEL description="PFLARE"

ENV CHECKOUT_BRANCH=$BRANCH
ENV PETSC_ARCH=$ARCH
# -on_error_abort ensures any test failures are caught and the build fails
ENV PETSC_OPTIONS="-on_error_abort"

WORKDIR /build

# Ensure pipelines fail if any stage fails
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Clone PFLARE and run all the tests
# Set the line length to 132 to check for truncation problems - the cray fortran
# compiler was defaulting to 132 and we don't have a CI for that, so this will trigger it
# If MALLOC_DUMP is true:
# 1) Remove -on_error_abort and strip it from the test Makefiles as it interferes with the output
# 2) Output the terminal to a log file
# 3) Check the log for any PetscTrMalloc lines which indicate memory leaks
RUN set -euo pipefail; \
    if [ "${MALLOC_DUMP}" = "true" ]; then export PETSC_OPTIONS="-malloc_dump"; fi; \
    echo "Using PETSC_OPTIONS=${PETSC_OPTIONS}"; \
    echo "Cloning branch: ${CHECKOUT_BRANCH}" && \
    git clone --branch ${CHECKOUT_BRANCH} https://github.com/PFLAREProject/PFLARE.git && \
    cd PFLARE && \
    if [ "${MALLOC_DUMP}" = "true" ]; then \
        echo "Stripping -on_error_abort from test Makefiles for malloc dump run"; \
        sed -i 's/-on_error_abort//g' tests/Makefile; \
        sed -i 's/-on_error_abort//g' python/Makefile; \
    fi && \
    if [ "${MALLOC_DUMP}" = "true" ]; then LOG=/tmp/test.log; : > "$LOG"; else LOG=/dev/null; fi; \
    make -j2 "FFLAGS=-ffree-line-length-132 -ffixed-line-length-132" 2>&1 | tee -a "$LOG"; \
    make python 2>&1 | tee -a "$LOG"; \
    make -j2 check "FFLAGS=-ffree-line-length-132 -ffixed-line-length-132" 2>&1 | tee -a "$LOG"; \
    make -j2 tests "FFLAGS=-ffree-line-length-132 -ffixed-line-length-132" 2>&1 | tee -a "$LOG"; \
    if [ "${MALLOC_DUMP}" = "true" ]; then \
      if grep -q "PetscTrMalloc" "$LOG"; then \
        echo "Error: found PetscTrMalloc in test output" >&2; exit 1; \
      fi; \
    fi

WORKDIR /build/PFLARE
