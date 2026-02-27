# Start from the pre-built PFLARE image — PETSc, PFLARE library, and
# the Python interface are all already compiled and installed.
FROM stevendargaville/pflare

LABEL maintainer="Steven Dargaville"
LABEL description="PFLARE – Binder-compatible image for interactive notebooks"

# ---------------------------------------------------------------
# Install Jupyter and notebook dependencies
# ---------------------------------------------------------------
RUN pip install --no-cache-dir \
    notebook \
    jupyterlab \
    matplotlib \
    scipy \
    numpy

# ---------------------------------------------------------------
# Binder requires a non-root user with UID=1000
# ---------------------------------------------------------------
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV NB_USER=${NB_USER}
ENV NB_UID=${NB_UID}
ENV HOME=/home/${NB_USER}

# Only create the user if UID 1000 is not already taken in the base image
RUN id -u ${NB_UID} >/dev/null 2>&1 || \
    useradd --create-home --shell /bin/bash --uid ${NB_UID} ${NB_USER}

# ---------------------------------------------------------------
# Set env vars matching python/Makefile:
#   PYTHONPATH = ${PETSCPYTHONPATH}:/build/PFLARE/python
#             = ${PETSC_DIR}/${PETSC_ARCH}/lib:/build/PFLARE/python
#   LD_LIBRARY_PATH = ${PETSC_DIR}/${PETSC_ARCH}/lib:/build/PFLARE/lib
# PETSC_DIR=/build/petsc and PETSC_ARCH=arch-linux-c-opt are ENV vars in
# the base image; Docker expands them here.
# ---------------------------------------------------------------
ENV PYTHONPATH=${PETSC_DIR}/${PETSC_ARCH}/lib:/build/PFLARE/python
ENV LD_LIBRARY_PATH=${PETSC_DIR}/${PETSC_ARCH}/lib:/build/PFLARE/lib
ENV PETSC_OPTIONS=-on_error_abort

# ---------------------------------------------------------------
# Copy notebooks and hand ownership to the Binder user
# ---------------------------------------------------------------
COPY notebooks/ ${HOME}/notebooks/
RUN chown -R ${NB_UID}:${NB_UID} ${HOME}

USER ${NB_USER}
WORKDIR ${HOME}
