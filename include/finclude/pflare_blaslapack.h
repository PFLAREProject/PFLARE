! Include file for PFLARE's Fortran BLAS/LAPACK dispatch
#if !defined(PFLARE_BLASLAPACK_H)
#define PFLARE_BLASLAPACK_H

! petscconf.h is included DIRECTLY here (proven pattern: src/PETSc_Helper.F90)
! so this header is self-sufficient regardless of include order - do NOT rely on
! a PETSc finclude appearing earlier in the including file, or the precision
! branch below would silently pick double. This maps PFLARE's generic
! BLAS/LAPACK names to the correct real-precision routine.
!
! Complex is intentionally unsupported (would need c*/z* routines with different
! signatures). NOTE if dot/nrm2-family calls are ever added: some BLAS return
! double from snrm2/sdot (see PETSC_BLASLAPACK_SNRM2_RETURNS_DOUBLE); PFLARE
! currently calls no dot/nrm2-family routines so this quirk does not apply.
!
! This header swaps one Fortran-source routine name for another (e.g. dgels ->
! sgels); the Fortran compiler applies its own (unchanged) linkage convention,
! exactly as the previous hardcoded d-prefix calls did. It deliberately does NOT
! depend on PETSC_BLASLAPACK_UNDERSCORE/PETSC_BLASLAPACK_CAPS or any C-side
! name-mangling macro.
#include <petscconf.h>

#if defined(PETSC_USE_COMPLEX)
#error "PFLARE BLAS/LAPACK dispatch does not support complex PETSc builds"
#endif

#if defined(PETSC_USE_REAL_SINGLE)
#define PFLAREgels sgels
#define PFLAREgelsd sgelsd
#define PFLAREgemv sgemv
#define PFLAREgemm sgemm
#define PFLAREgeev sgeev
#define PFLAREgesvd sgesvd
#define PFLAREgeqrf sgeqrf
#define PFLAREgesv sgesv
#else
#define PFLAREgels dgels
#define PFLAREgelsd dgelsd
#define PFLAREgemv dgemv
#define PFLAREgemm dgemm
#define PFLAREgeev dgeev
#define PFLAREgesvd dgesvd
#define PFLAREgeqrf dgeqrf
#define PFLAREgesv dgesv
#endif

#endif
