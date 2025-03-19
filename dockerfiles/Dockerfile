ARG BASE_IMAGE=stevendargaville/petsc

FROM ${BASE_IMAGE}
# Checkout the main branch by default
ARG BRANCH=main
# Use the optimised petsc build by default
ARG ARCH=arch-linux-c-opt

LABEL maintainer="Steven Dargaville"
LABEL description="PFLARE"

ENV CHECKOUT_BRANCH=$BRANCH
ENV PETSC_ARCH=$ARCH
# Ensure any test failures are caught and the build fails
ENV PETSC_OPTIONS="-on_error_abort"

WORKDIR /build

# Clone PFLARE and run all the tests
# Set the line length to 132 to check for truncation problems - the cray fortran
# compiler was defaulting to 132 and we don't have a CI for that, so this will trigger it
RUN set -e; \
    echo "Cloning branch: ${CHECKOUT_BRANCH}" && \
    git clone --branch ${CHECKOUT_BRANCH} https://github.com/stevendargaville/PFLARE.git && \
    cd PFLARE && \
    make -j2 "FFLAGS=-ffree-line-length-132 -ffixed-line-length-132" && \
    make tests -j2 "FFLAGS=-ffree-line-length-132 -ffixed-line-length-132" && \
    make python && make tests_python

WORKDIR /build/PFLARE
