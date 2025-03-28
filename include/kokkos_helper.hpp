#if !defined (KOKKOS_HELPER_DEF_H)
#define KOKKOS_HELPER_DEF_H

// petscvec_kokkos.hpp has to go first
#include <petscvec_kokkos.hpp>
#include <petscmat_kokkos.hpp>
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
using ConstMatRowMapKokkosView = KokkosCsrGraph::row_map_type::const_type;
using ScratchMemSpace = typename KokkosTeamMemberType::scratch_memory_space;

PETSC_INTERN void mat_duplicate_copy_plus_diag_kokkos(Mat *, int, Mat *);

#endif