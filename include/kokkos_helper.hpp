#if !defined (KOKKOS_HELPER_DEF_H)
#define KOKKOS_HELPER_DEF_H

// petscvec_kokkos.hpp has to go first
#include <petscvec_kokkos.hpp>
#include "petsc.h"
#include <../src/mat/impls/aij/seq/kokkos/aijkok.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <../src/vec/vec/impls/seq/kokkos/veckokkosimpl.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

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