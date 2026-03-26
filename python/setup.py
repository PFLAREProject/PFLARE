#!/usr/bin/env python3

#$ python setup.py build_ext --inplace

from setuptools import setup, Extension
from Cython.Build import cythonize

import numpy
import petsc4py

def configure():
    INCLUDE_DIRS = []
    LIBRARY_DIRS = []
    LIBRARIES    = []

    # PETSc
    import os
    PETSC_DIR  = os.environ['PETSC_DIR']
    PETSC_ARCH = os.environ.get('PETSC_ARCH', '')
    from os.path import join, isdir
    if PETSC_ARCH and isdir(join(PETSC_DIR, PETSC_ARCH)):
        INCLUDE_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'include'),
                         join(PETSC_DIR, 'include')]
        LIBRARY_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'lib')]
    else:
        if PETSC_ARCH: pass # XXX should warn ...
        INCLUDE_DIRS += [join(PETSC_DIR, 'include')]
        LIBRARY_DIRS += [join(PETSC_DIR, 'lib')]
    LIBRARIES += ['petsc', 'pflare']

    # PETSc for Python
    INCLUDE_DIRS += [petsc4py.get_include()]

    # NumPy
    INCLUDE_DIRS += [numpy.get_include()]

    return dict(
        include_dirs=INCLUDE_DIRS + [os.curdir],
        libraries=LIBRARIES,
        library_dirs=LIBRARY_DIRS,
        runtime_library_dirs=LIBRARY_DIRS,
    )

# -Werror is added to ensure any warnings building the Python 
# interface (e.g., deprecation warnings) are treated as errors
# to match the PETSc CI
extensions = [
    Extension('pflare_defs',
              sources = ['pflare_defs.pyx'],
              extra_link_args=['-lpetsc'],
              extra_compile_args=['-Werror=cpp'],
              **configure()),
]

setup(name = "pflare_defs",
      ext_modules = cythonize(
          extensions, 
          include_path=[petsc4py.get_include()],
          compiler_directives={'language_level': 3}  # Use python3
      ),
)
