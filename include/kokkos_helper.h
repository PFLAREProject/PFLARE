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
using ConstMatRowMapKokkosView = KokkosCsrGraph::row_map_type::const_type;

#endif