/*
  An LD_PRELOAD library that counts the MPI collectives a program calls inside
  windows it opens and closes with MPI_Pcontrol, and aborts if a window sees a
  different number than it expects. It is how make tests_collectives checks that
  PCAIR v-cycles (and diagonal updates) do no global reductions, see the
  -check_collectives option of the test drivers and check_collectives.h

  It is not linked into anything, run a test with it preloaded instead:
    mpiexec -n 2 env LD_PRELOAD=$PWD/libcount_collectives.so ./adv_1d -check_collectives

  MPI_Pcontrol(level), which MPI defines for profiling layers and is a no-op
  without one:
    level >= 1 opens a window that expects exactly level - 1 collectives
    level == 0 closes the window and checks the count on this rank
  On a mismatch it prints a backtrace of (up to MAX_TRACES of) the collectives
  it counted and calls MPI_Abort

  We can't use -log_view for this: with an MPI that has large count support,
  PETSc's MPIU_Allreduce calls MPI_Allreduce_c, which the PETSc logging doesn't
  count, so -log_view shows no reductions at all

  Every collective is wrapped under both its MPI_ and its PMPI_ name and the
  real PMPI_ version is found with dlsym(RTLD_NEXT). This catches calls from C
  (PETSc, PFLARE's C) and from Fortran (PFLARE's Fortran, which goes through the
  MPI Fortran bindings into the C library under one of those two names). As
  that last part depends on the MPI, the drivers first open a calibration
  window with a known number of collectives from both C and Fortran, so a build
  where the counting doesn't see a call fails rather than passing with zero
  Only communicators with more than one rank are counted, and neighbourhood
  collectives and point-to-point messages are not, as they are not global
  synchronisations
*/
#define _GNU_SOURCE
#include <mpi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_TRACES 8
#define MAX_FRAMES 32

// Depth stops a collective being counted twice if the MPI library calls one
// of our wrapped names from inside another
static __thread int depth = 0;
static int          window_open = 0, window_id = 0, windows_checked = 0;
static long         expected = 0, counted = 0;
static void        *traces[MAX_TRACES][MAX_FRAMES];
static int          trace_frames[MAX_TRACES];
static const char  *trace_name[MAX_TRACES];

// Lets the drivers check this library has been preloaded, as otherwise every
// MPI_Pcontrol is a no-op and every check would pass
__attribute__((constructor)) static void mark_loaded(void)
{
  setenv("PFLARE_COUNT_COLLECTIVES_LOADED", "1", 1);
}

static void *real_function(const char *name)
{
  void *f = dlsym(RTLD_NEXT, name);
  if (!f) {
    fprintf(stderr, "count_collectives: could not find %s in the MPI library\n", name);
    abort();
  }
  return f;
}

static void record(const char *name, MPI_Comm comm)
{
  int size;

  if (!window_open || depth || comm == MPI_COMM_NULL) return;
  PMPI_Comm_size(comm, &size);
  if (size < 2) return;
  if (counted < MAX_TRACES) {
    trace_frames[counted] = backtrace(traces[counted], MAX_FRAMES);
    trace_name[counted]   = name;
  }
  counted++;
}

static int control(int level)
{
  int rank;

  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (level >= 1) {
    if (window_open) {
      fprintf(stderr, "count_collectives: [%d] window %d opened while window %d is still open\n", rank, window_id + 1, window_id);
      PMPI_Abort(MPI_COMM_WORLD, 1);
    }
    window_open = 1;
    window_id++;
    expected = level - 1;
    counted  = 0;
  } else if (level == 0) {
    if (!window_open) {
      fprintf(stderr, "count_collectives: [%d] window closed when none is open\n", rank);
      PMPI_Abort(MPI_COMM_WORLD, 1);
    }
    window_open = 0;
    windows_checked++;
    if (counted != expected) {
      fprintf(stderr, "count_collectives: [%d] window %d expected %ld collectives but saw %ld\n", rank, window_id, expected, counted);
      // Every line has the rank on it, as the ranks print at the same time
      for (long i = 0; i < counted && i < MAX_TRACES; i++) {
        char **symbols = backtrace_symbols(traces[i], trace_frames[i]);

        fprintf(stderr, "count_collectives: [%d] collective %ld is %s, called from:\n", rank, i + 1, trace_name[i]);
        // The first two frames are record() and the wrapper
        for (int j = 2; symbols && j < trace_frames[i]; j++) fprintf(stderr, "count_collectives: [%d]   %s\n", rank, symbols[j]);
        free(symbols);
      }
      fflush(stderr);
      PMPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  return MPI_SUCCESS;
}

int MPI_Pcontrol(const int level, ...)
{
  return control(level);
}
int PMPI_Pcontrol(const int level, ...)
{
  return control(level);
}

static int finalize(void)
{
  int rank;
  int (*real_finalize)(void) = (int (*)(void))real_function("PMPI_Finalize");

  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (window_open) {
    fprintf(stderr, "count_collectives: [%d] window %d still open at MPI_Finalize\n", rank, window_id);
    PMPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (rank == 0) printf("count_collectives: %d windows checked\n", windows_checked);
  return real_finalize();
}
int MPI_Finalize(void)
{
  return finalize();
}
int PMPI_Finalize(void)
{
  return finalize();
}

// Defines MPI_fn and PMPI_fn, which count the call and then call the real PMPI_fn
// Every wrapped function has its communicator argument called comm
#define WRAP(fn, PARAMS, ARGS) \
  typedef int(*real_##fn##_t) PARAMS; \
  static real_##fn##_t real_##fn = NULL; \
  static int           count_##fn PARAMS \
  { \
    int err; \
    if (!real_##fn) real_##fn = (real_##fn##_t)real_function("PMPI_" #fn); \
    record("MPI_" #fn, comm); \
    depth++; \
    err = real_##fn ARGS; \
    depth--; \
    return err; \
  } \
  int MPI_##fn PARAMS \
  { \
    return count_##fn ARGS; \
  } \
  int PMPI_##fn PARAMS \
  { \
    return count_##fn ARGS; \
  }

// clang-format off
WRAP(Barrier, (MPI_Comm comm), (comm))
WRAP(Ibarrier, (MPI_Comm comm, MPI_Request *request), (comm, request))
WRAP(Comm_dup, (MPI_Comm comm, MPI_Comm *newcomm), (comm, newcomm))
WRAP(Comm_split, (MPI_Comm comm, int color, int key, MPI_Comm *newcomm), (comm, color, key, newcomm))

#define REDUCE_LIKE(fn, COUNT) \
  WRAP(fn, (const void *sendbuf, void *recvbuf, COUNT count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm), \
           (sendbuf, recvbuf, count, datatype, op, comm))
#define IREDUCE_LIKE(fn, COUNT) \
  WRAP(fn, (const void *sendbuf, void *recvbuf, COUNT count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request *request), \
           (sendbuf, recvbuf, count, datatype, op, comm, request))
#define ROOTED_REDUCE(fn, COUNT) \
  WRAP(fn, (const void *sendbuf, void *recvbuf, COUNT count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm), \
           (sendbuf, recvbuf, count, datatype, op, root, comm))
#define IROOTED_REDUCE(fn, COUNT) \
  WRAP(fn, (const void *sendbuf, void *recvbuf, COUNT count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm, MPI_Request *request), \
           (sendbuf, recvbuf, count, datatype, op, root, comm, request))
#define BCAST_LIKE(fn, COUNT) \
  WRAP(fn, (void *buffer, COUNT count, MPI_Datatype datatype, int root, MPI_Comm comm), \
           (buffer, count, datatype, root, comm))
#define IBCAST_LIKE(fn, COUNT) \
  WRAP(fn, (void *buffer, COUNT count, MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Request *request), \
           (buffer, count, datatype, root, comm, request))
#define ALLGATHER_LIKE(fn, COUNT) \
  WRAP(fn, (const void *sendbuf, COUNT sendcount, MPI_Datatype sendtype, void *recvbuf, COUNT recvcount, MPI_Datatype recvtype, MPI_Comm comm), \
           (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm))
#define IALLGATHER_LIKE(fn, COUNT) \
  WRAP(fn, (const void *sendbuf, COUNT sendcount, MPI_Datatype sendtype, void *recvbuf, COUNT recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request), \
           (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request))
#define ROOTED_GATHER_LIKE(fn, COUNT) \
  WRAP(fn, (const void *sendbuf, COUNT sendcount, MPI_Datatype sendtype, void *recvbuf, COUNT recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm), \
           (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm))
#define ALLGATHERV_LIKE(fn, COUNT, DISP) \
  WRAP(fn, (const void *sendbuf, COUNT sendcount, MPI_Datatype sendtype, void *recvbuf, const COUNT recvcounts[], const DISP displs[], MPI_Datatype recvtype, MPI_Comm comm), \
           (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm))
#define IALLGATHERV_LIKE(fn, COUNT, DISP) \
  WRAP(fn, (const void *sendbuf, COUNT sendcount, MPI_Datatype sendtype, void *recvbuf, const COUNT recvcounts[], const DISP displs[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request), \
           (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request))
#define GATHERV_LIKE(fn, COUNT, DISP) \
  WRAP(fn, (const void *sendbuf, COUNT sendcount, MPI_Datatype sendtype, void *recvbuf, const COUNT recvcounts[], const DISP displs[], MPI_Datatype recvtype, int root, MPI_Comm comm), \
           (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm))
#define SCATTERV_LIKE(fn, COUNT, DISP) \
  WRAP(fn, (const void *sendbuf, const COUNT sendcounts[], const DISP displs[], MPI_Datatype sendtype, void *recvbuf, COUNT recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm), \
           (sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm))
#define ALLTOALLV_LIKE(fn, COUNT, DISP) \
  WRAP(fn, (const void *sendbuf, const COUNT sendcounts[], const DISP sdispls[], MPI_Datatype sendtype, void *recvbuf, const COUNT recvcounts[], const DISP rdispls[], MPI_Datatype recvtype, MPI_Comm comm), \
           (sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm))
#define ALLTOALLW_LIKE(fn, COUNT, DISP) \
  WRAP(fn, (const void *sendbuf, const COUNT sendcounts[], const DISP sdispls[], const MPI_Datatype sendtypes[], void *recvbuf, const COUNT recvcounts[], const DISP rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm), \
           (sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm))
#define REDUCE_SCATTER_LIKE(fn, COUNT) \
  WRAP(fn, (const void *sendbuf, void *recvbuf, const COUNT recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm), \
           (sendbuf, recvbuf, recvcounts, datatype, op, comm))
#define REDUCE_SCATTER_BLOCK_LIKE(fn, COUNT) \
  WRAP(fn, (const void *sendbuf, void *recvbuf, COUNT recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm), \
           (sendbuf, recvbuf, recvcount, datatype, op, comm))

REDUCE_LIKE(Allreduce, int)
REDUCE_LIKE(Scan, int)
REDUCE_LIKE(Exscan, int)
IREDUCE_LIKE(Iallreduce, int)
ROOTED_REDUCE(Reduce, int)
IROOTED_REDUCE(Ireduce, int)
BCAST_LIKE(Bcast, int)
IBCAST_LIKE(Ibcast, int)
ALLGATHER_LIKE(Allgather, int)
ALLGATHER_LIKE(Alltoall, int)
IALLGATHER_LIKE(Iallgather, int)
ROOTED_GATHER_LIKE(Gather, int)
ROOTED_GATHER_LIKE(Scatter, int)
ALLGATHERV_LIKE(Allgatherv, int, int)
IALLGATHERV_LIKE(Iallgatherv, int, int)
GATHERV_LIKE(Gatherv, int, int)
SCATTERV_LIKE(Scatterv, int, int)
ALLTOALLV_LIKE(Alltoallv, int, int)
ALLTOALLW_LIKE(Alltoallw, int, int)
REDUCE_SCATTER_LIKE(Reduce_scatter, int)
REDUCE_SCATTER_BLOCK_LIKE(Reduce_scatter_block, int)

// The large count versions from MPI 4, which PETSc uses when the MPI has them
#if MPI_VERSION >= 4
REDUCE_LIKE(Allreduce_c, MPI_Count)
REDUCE_LIKE(Scan_c, MPI_Count)
REDUCE_LIKE(Exscan_c, MPI_Count)
IREDUCE_LIKE(Iallreduce_c, MPI_Count)
ROOTED_REDUCE(Reduce_c, MPI_Count)
IROOTED_REDUCE(Ireduce_c, MPI_Count)
BCAST_LIKE(Bcast_c, MPI_Count)
IBCAST_LIKE(Ibcast_c, MPI_Count)
ALLGATHER_LIKE(Allgather_c, MPI_Count)
ALLGATHER_LIKE(Alltoall_c, MPI_Count)
IALLGATHER_LIKE(Iallgather_c, MPI_Count)
ROOTED_GATHER_LIKE(Gather_c, MPI_Count)
ROOTED_GATHER_LIKE(Scatter_c, MPI_Count)
ALLGATHERV_LIKE(Allgatherv_c, MPI_Count, MPI_Aint)
IALLGATHERV_LIKE(Iallgatherv_c, MPI_Count, MPI_Aint)
GATHERV_LIKE(Gatherv_c, MPI_Count, MPI_Aint)
SCATTERV_LIKE(Scatterv_c, MPI_Count, MPI_Aint)
ALLTOALLV_LIKE(Alltoallv_c, MPI_Count, MPI_Aint)
ALLTOALLW_LIKE(Alltoallw_c, MPI_Count, MPI_Aint)
REDUCE_SCATTER_LIKE(Reduce_scatter_c, MPI_Count)
REDUCE_SCATTER_BLOCK_LIKE(Reduce_scatter_block_c, MPI_Count)
#endif
// clang-format on
