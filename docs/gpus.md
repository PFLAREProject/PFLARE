## GPU support           

If PETSc has been configured with GPU support then PCPFLAREINV and PCAIR support GPUs [3]. We recommend configuring PETSc with Kokkos and always specifying the matrix/vector types as Kokkos as this works across different GPU hardware (Nvidia, AMD, Intel). PFLARE also contains Kokkos routines to speed-up the setup/solve on GPUs. 

By default the tests run on the CPU unless the matrix/vector types are specified as those compatible with GPUs. For example, the following arguments specify that the 1D advection problem ``tests/adv_1d`` will use a 30th order GMRES polynomial applied matrix-free to solve on the CPU:

``./adv_1d -n 1000 -ksp_type richardson -pc_type pflareinv -pc_pflareinv_type arnoldi -pc_pflareinv_matrix_free -pc_pflareinv_poly_order 30``

To run on GPUs, we set the matrix/vector types as Kokkos, which can be easily done through command line arguments. Our tests use either ``-mat_type`` and ``-vec_type``, or if set by a DM directly use ``-dm_mat_type`` and ``-dm_vec_type``.

For example, running the same problem on a single GPU with KOKKOS:

``./adv_1d -n 1000 -ksp_type richardson -pc_type pflareinv -pc_pflareinv_type arnoldi -pc_pflareinv_matrix_free -pc_pflareinv_poly_order 30 -mat_type aijkokkos -vec_type kokkos``

Note: many of the tests allow the option ``-second_solve`` which turns on two solves, the first to trigger any copies to the GPU (e.g., the top grid matrix if created on the host) and the second to allow accurate timing. 

Development of the setup on GPUs is ongoing, please get in touch if you would like to contribute. The main areas requiring development are:

1) Processor agglomeration - GPU libraries exist which could replace the CPU-based calls to the PETSc graph partitioners
2) GPU optimisation - There are several Kokkos routines in PFLARE which would benefit from further optimisation

### Performance notes

1 - Typically we find good performance using as many DOFs per GPU as possible. 

2 - The processor agglomeration happens through the graph partitioners in PETSc and currently there is no GPU partitioner, hence this could be slow. The default parameters used in the processor agglomeration in PCAIR (e.g., ``-pc_air_process_eq_limit``) have also not been optimised for GPUs. You may wish to disable the processor agglomeration in parallel on GPUs (``-pc_air_processor_agglom 0``). Using heavy truncation may also help mitigate the the impact of turning off processor agglomeration on GPUs, see below.

3 - Multigrid methods on GPUs will often pin the coarse grids to the CPU, as GPUs are not very fast at the small solves that occur on coarse grids. We do not do this in PCAIR; instead we use the same approach we used in [2] to improve parallel scaling on CPUs. 

This is based around using the high-order polynomials applied matrix free as a coarse solver. For many problems GMRES polynomials in the Newton basis are stable at high order and can therefore be combined with heavy truncation of the multigrid hierarchy. We also have an automated way to determine at what level of the multigrid hierarchy to truncate. 

For example, on a single GPU with a 2D structured grid advection problem we apply a high order (10th order) GMRES polynomial as a Newton polynomial matrix-free as a coarse grid solver:

``./adv_diff_fd -da_grid_x 1000 -da_grid_y 1000 -ksp_type richardson -pc_type air -pc_air_coarsest_inverse_type newton -pc_air_coarsest_matrix_free_polys -pc_air_coarsest_poly_order 10 -dm_mat_type aijkokkos -dm_vec_type kokkos``

The hierarchy in this case has 29 levels. If we turn on the auto truncation and set a very large truncation tolerance  

``./adv_diff_fd -da_grid_x 1000 -da_grid_y 1000 -ksp_type richardson -pc_type air -pc_air_coarsest_inverse_type newton -pc_air_coarsest_matrix_free_polys -pc_air_coarsest_poly_order 10 -dm_mat_type aijkokkos -dm_vec_type kokkos -pc_air_auto_truncate_start_level 1 -pc_air_auto_truncate_tol 1e-1``

we find that the 10th order polynomials are good enough coarse solvers to enable truncation of the hierarchy at level 11. This gives the same iteration count as without truncation and we see an overall speedup of ~1.47x in the solve in this example. The speedup is typically greater in parallel. Please see [3] for more details.

## OpenMP support

If PETSc has been configured with Kokkos using OpenMP as the backend then PCPFLAREINV and PCAIR support OpenMP. To enable OpenMP throughout the setup/solve the matrix/vector types must be specified as Kokkos (see above) and the ``OMP_NUM_THREADS`` environmental variable must be set. Good performance is dependent on appropriate pinning of MPI ranks and OpenMP threads to CPU cores/NUMA regions.