#include "petscconf.h"

! Define an interoperable kind matching PetscInt for BIND(C) use
#ifdef PETSC_USE_64BIT_INDICES
#define PFLARE_PETSCINT_C_KIND c_long_long
#else
#define PFLARE_PETSCINT_C_KIND c_int
#endif

! Define an interoperable kind matching PetscReal for BIND(C) use
#ifdef PETSC_USE_REAL_SINGLE
#define PFLARE_PETSCREAL_C_KIND c_float
#elif defined(PETSC_USE_REAL_DOUBLE)
#define PFLARE_PETSCREAL_C_KIND c_double
#elif defined(PETSC_USE_REAL___FLOAT128)
#define PFLARE_PETSCREAL_C_KIND c_long_double
#warning "Using long double as stand-in for __float128; verify sizes."
#else
#error "Unknown PETSc real precision"
#endif

! Define an interoperable type matching PetscScalar for BIND(C) use
#ifdef PETSC_USE_COMPLEX
#define PFLARE_PETSCSCALAR_C_TYPE complex(PFLARE_PETSCREAL_C_KIND)
#else
#define PFLARE_PETSCSCALAR_C_TYPE real(PFLARE_PETSCREAL_C_KIND)
#endif

! Define an interoperable kind matching PetscErrorCode for BIND(C) use
#define PFLARE_PETSCERRORCODE_C_KIND c_int

! Define an interoperable kind matching PetscBool for BIND(C) use
#define PFLARE_PETSCBOOL_C_TYPE logical(c_bool)