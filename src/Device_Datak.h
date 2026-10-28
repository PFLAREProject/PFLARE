/*
  Prototypes of the routines in Device_Datak.kokkos.cxx that the C calls: the
  lifecycle of the device copies of the CF markers and the diagonal dominance
  ratios that the Kokkos PMISR, DDC and MatDiagDomRatio share between calls

  Device_Datak.kokkos.cxx includes this header too, so the compiler checks
  that these match the definitions (create_cf_is_device_kokkos takes a Kokkos
  view, so it is declared in include/kokkos_helper.hpp instead)
*/

#if !defined(DEVICE_DATAK_H)
#define DEVICE_DATAK_H

#include <petscmat.h>

#if defined(PETSC_HAVE_KOKKOS)
PETSC_INTERN void copy_cf_markers_d2h(int *cf_markers_local);
PETSC_INTERN void copy_diag_dom_ratio_d2h(PetscReal *diag_dom_ratio_local);
PETSC_INTERN void delete_device_cf_markers(void);
PETSC_INTERN void delete_device_diag_dom_ratio(void);
PETSC_INTERN void create_cf_is_kokkos(Mat *input_mat, IS *is_fine, IS *is_coarse);
#endif

#endif
