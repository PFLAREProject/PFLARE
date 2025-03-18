#if !defined (KOKKOS_HELPER_DEF_H)
#define KOKKOS_HELPER_DEF_H

#include "petsc.h"

using DefaultExecutionSpace = Kokkos::DefaultExecutionSpace;
using DefaultMemorySpace    = Kokkos::DefaultExecutionSpace::memory_space;
using HostMirrorMemorySpace = Kokkos::DualView<PetscScalar *>::host_mirror_space::memory_space;
using PetscIntConstKokkosViewHost = Kokkos::View<const PetscInt *, HostMirrorMemorySpace>;
using intKokkosViewHost = Kokkos::View<int *, HostMirrorMemorySpace>;
using intKokkosView = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;
using boolKokkosView = Kokkos::View<bool *, Kokkos::DefaultExecutionSpace>;

PetscErrorCode MatSetMPIAIJKokkosWithSplitSeqAIJKokkosMatrices_mine(Mat, Mat, Mat, PetscInt *);
PetscErrorCode MatSetMPIAIJKokkosWithSplitSeqAIJKokkosMatrices_mine(Mat, Mat, Mat, PetscInt *);
PetscErrorCode MatSeqAIJSetPreallocation_SeqAIJ_mine(Mat, PetscInt, const PetscInt *);
PetscErrorCode MatSetSeqAIJKokkosWithCSRMatrix_mine(Mat, Mat_SeqAIJKokkos *);

#endif