/*
  Prototypes of the Kokkos kernels in VecISCopyLocalk.kokkos.cxx that the FC
  smoothers (FC_Smooth.c, FC_Smooth_Block.c) call

  VecISCopyLocalk.kokkos.cxx includes this header too, so the compiler checks
  that these match the definitions
  The levels are numbered from 1 here, as in the Fortran (our_level + 1 of the
  0-based our_level of the C)
*/

#if !defined(VECISCOPYLOCALK_H)
#define VECISCOPYLOCALK_H

#include <petscmat.h>

#if defined(PETSC_HAVE_KOKKOS)
PETSC_INTERN void destroy_VecISCopyLocal_kokkos(void **handle);
PETSC_INTERN void create_VecISCopyLocal_kokkos(int max_levels_input, void **handle);
PETSC_INTERN void set_VecISCopyLocal_kokkos_our_level(void *handle, int our_level, PetscInt global_row_start, IS *index_fine, IS *index_coarse);
PETSC_INTERN void VecISCopyLocal_kokkos(void *handle, int our_level, int fine_int, Vec *vfull, int mode_int, Vec *vreduced);
PETSC_INTERN void mat_iscopy_local_kokkos(void *handle, int our_level, int fine_int, Mat *xfull, int mode_int, Mat *xreduced, int *done);
#endif

#endif
