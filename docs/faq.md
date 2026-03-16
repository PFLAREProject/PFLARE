# Frequently asked questions

Listed below are some answers to frequently asked questions when using PFLARE. The Jupyter notebooks in the `notebooks/` directory are also a good place to start for understanding the methods in PFLARE.

## How to improve convergence with `PCAIR`

Below are some of the factors to consider when trying to improve convergence with `PCAIR`. This is not an exhaustive list, but covers the most common improvements.

Reduction multigrid methods like AIR can become direct solvers as the approximation to $A_{ff}^{-1}$ improves; this means there is almost always a route to better convergence. In practice, of course, this limit is not the most performant or memory efficient, especially in parallel. The goal is typically to strike a balance between convergence, memory usage and parallel performance.

### Quick fix

If the default options do not give good convergence, try:

`-pc_air_diag_scale_polys -pc_air_coarsest_diag_scale_polys -pc_air_a_lump -pc_air_cf_splitting_type diag_dom -pc_air_strong_threshold 0.3 -pc_air_a_drop 1e-6 -pc_air_r_drop 1e-3`

If that did not work or for further understanding of these parameters, please see below.

### 1) Diagonal scaling

AIR methods work best when the rows of the matrix are all scaled roughly the same. Applying a diagonal scaling before solving with `PCAIR` can help improve convergence.

If it is difficult to manually apply a diagonal scaling (e.g., inside a non-linear or time-stepping loop), the options `-pc_air_diag_scale_polys` and `-pc_air_coarsest_diag_scale_polys` can be used to scale the diagonals of $A_{ff}$ on each level during the multigrid setup. This can often restore good convergence when poor scaling is the issue.

For DG FEM discretisations, a block diagonal scaling is typically required, where each block is the inverse of the element matrix. This is particularly important when using high-order basis functions and must be applied prior to solving with `PCAIR`.

If you suspect scaling is still responsible for poor convergence, try using an algebraic approximate inverse method which is agnostic to scalings. For example, the ISAI (Incomplete Sparse Approximate Inverse) can be used with `-pc_air_inverse_type isai` instead of GMRES polynomials.

### 2) Lumping

A relative row-wise drop tolerance is applied to the coarse matrix on each level of the multigrid hierarchy. By default, dropped entries are simply discarded.

Most discretisations benefit from lumping dropped entries to the diagonal instead, which can be enabled with `-pc_air_a_lump`.

### 3) Coarsening rate

Slowing the coarsening rate improves the diagonal dominance of $A_{ff}$ and hence improves the quality of the approximate inverses used to build the hierarchy. The quality of the CF splitting is central to how well reduction multigrids work: the goal is to find a large $A_{ff}$ submatrix whose inverse can be well approximated by a sparse (or low-order polynomial) approximation.

Several parameters control how fast the default CF splitting (PMISR DDC) coarsens.

**Strong threshold:** The most impactful parameter is the strong threshold, set with `-pc_air_strong_threshold` (default 0.5). This determines which connections between unknowns are considered "strong". Decreasing it slows the coarsening and should improve convergence.

Internally, PMISR computes a maximal independent set in the symmetrized strong connections ($S + S^T$). This ensures that no two F-points are strongly connected, keeping large off-diagonal entries out of $A_{ff}$.

**DDC parameters:** The second pass of PMISR DDC performs a diagonal dominance cleanup (DDC), which converts the least diagonally dominant F-points to C-points. The number of DDC iterations can be increased with `-pc_air_ddc_its` (default 1) and the fraction of local F-points changed per iteration can be modified with `-pc_air_ddc_fraction` (default 0.1, i.e., 10% of F-points per iteration). It is often helpful to do more iterations but change fewer F-points per iteration (e.g., `-pc_air_ddc_its 3 -pc_air_ddc_fraction 0.01`).

**Direct diagonal dominance control:** Modifying and balancing the strong threshold, DDC iterations and DDC fraction to get optimal performance can be difficult. The easiest way to automate this is to control the diagonal dominance ratio of $A_{ff}$ directly.

Setting `-pc_air_cf_splitting_type diag_dom` switches to a CF splitting that directly enforces a maximum row-wise diagonal dominance ratio. With this splitting, `-pc_air_strong_threshold` controls the maximum allowed dominance ratio in every row. Decreasing this value from the default 0.5 will slow the coarsening and improve convergence, e.g., test 0.4, 0.3, 0.2, 0.1.

The `-pc_air_ddc_its` and `-pc_air_ddc_fraction` options are ignored with this CF splitting. This splitting is more expensive to compute than the default PMISR DDC, but can give more reliable and direct control over convergence.

### 4) Drop tolerances

Relative row-wise drop tolerances are applied to both the coarse matrix and the approximate ideal grid-transfer operators at every level in the multigrid hierarchy.

The tolerance on the coarse matrix is controlled with `-pc_air_a_drop` (default 1e-4).

The tolerance on the grid-transfer operators is controlled with `-pc_air_r_drop` (default 1e-2).

Decreasing both of these values improves convergence but requires more memory. Decreasing `-pc_air_a_drop` is often more impactful. For example, when solving advection problems, values of `-pc_air_a_drop 1e-6` and `-pc_air_r_drop 1e-3` have been found to give scalable results up to hundreds of billions of unknowns.

### 5) Further options

In rough order of importance, additional options that can improve convergence include:

- Add C-point smoothing with `-pc_air_smooth_type fc`, or multiple iterations of smoothing, e.g., FCF smoothing with `-pc_air_smooth_type fcf`. Note that C-point smoothing requires storing more matrices on each level and hence uses more memory.
- Increase the non-zeros retained in the assembled approximate inverse with `-pc_air_inverse_sparsity_order 2` (default 1). The sparsity order controls how much fill-in is allowed. Higher orders give a better approximation to $A_{ff}^{-1}$ but use more memory and have a more expensive setup.
- Use an approximate ideal prolongator instead of the default classical one-point prolongator with `-pc_air_one_point_classical_prolong false`.
- Improve the approximate ideal operators with Richardson iterations, e.g., `-pc_air_improve_z_its 1 -pc_air_improve_w_its 1`.

## How to improve scalability with `PCAIR`

If the iteration count is growing during weak scaling studies, it can usually be improved by following the convergence steps above. Experience with `PCAIR` has shown that starting with options that give a well-converging solve (e.g., solving to a relative tolerance of 1e-10 in 5 or 6 iterations) will typically lead to low growth in iteration count with refinement.

Starting from a solve that requires 10-20 (or more) iterations almost never leads to scalable iteration counts. Think of reduction multigrid smoothers more like solvers than smoothers: you want all error modes to go to zero, not just a subset. If the solve is not converging well on a coarse problem, it will only get worse at scale.

## How to improve parallel performance with `PCAIR`

Once good convergence (and a scalable iteration count) has been achieved, the focus can turn to ensuring the solve time itself is scalable. Growth in the run time in parallel (separate from the iteration count) is typically caused by too many levels in the hierarchy. On coarse levels, the amount of work decreases but the relative amount of communication increases, creating bottlenecks.

`PCAIR` provides several tools to address this. The strategies below can also be combined.

### 1) Processor agglomeration

Processor agglomeration decreases the number of active MPI ranks on lower levels and enables repartitioning with ParMETIS. This is enabled with `-pc_air_processor_agglom` (on by default).

Repartitioning the coarse grids is important for performance. It improves the ratio of local to non-local work on lower levels, which directly affects the cost of both the matrix-matrix products in the setup and the matrix-vector products in the solve. Without repartitioning, the setup time can grow very quickly, as the matrix-matrix products used to compute the restrictor and coarse grid matrix become communication bound in the middle of the hierarchy.

This is typically visible in the cost of SpGEMMs during the setup increasing considerably in the middle of the hierarchy. Using the option `-pc_air_print_stats_timings` outputs the cumulative timers across the setup and the Python script `tools/parse_pflare_output.py` can be used to extract timings from the saved terminal output.

The repartitioning is triggered when the average ratio of local to non-local non-zeros drops below `-pc_air_processor_agglom_ratio` (default 2). When triggered, the number of active MPI ranks is halved (matching the expected coarsening rate of around 1/2 in advection-type problems).

### 2) Truncating the hierarchy

The multigrid hierarchy can be truncated early and an iterative coarse-grid solver used instead, often without any change in iteration count. This is one of the most effective ways to improve parallel performance, both on CPUs and GPUs. The key insight is that the lower levels of the hierarchy have very little local work but still incur communication costs (and kernel launch overheads on GPUs). Replacing these levels with an iterative solver that only requires matrix-vector products avoids these bottlenecks entirely.

**Automatic truncation:** The option `-pc_air_auto_truncate_start_level 1` enables automatic truncation of the multigrid hierarchy. It starts from the specified level and tests if the coarse grid solver can solve the current coarse problem to a given tolerance. If so, the hierarchy is truncated there.

The tolerance is controlled with `-pc_air_auto_truncate_tol` (default 1e-14). A surprisingly loose tolerance often works well; for advection problems, values between 1e-1 and 1e-3 have been found to allow heavy truncation while giving the same iteration count as no truncation.

If the iteration count increases with truncation, reduce this tolerance. If the current tolerance is performing well, try increasing it to allow more aggressive truncation and fewer levels.

**Estimating the start level:** Testing the coarse solver at every level starting from the top of the hierarchy can be expensive and requires more memory. For production runs, the start level should be set based on a rough estimate from smaller problems. Given that the coarsening rate in advection problems is around 1/2, the number of levels can be estimated from the number of unknowns.

**Choosing a coarse grid solver:** A high-order GMRES polynomial in Newton form applied matrix-free is well suited as a coarse grid solver, particularly on GPUs. For example:

`-pc_air_coarsest_poly_order 100 -pc_air_coarsest_inverse_type newton -pc_air_coarsest_matrix_free_polys`

This only requires matrix-vector products (no dot products) and uses asynchronous communication, making it well suited to coarse levels.

**Impact on cycle complexity:** Truncation increases the cycle complexity (amount of work per V-cycle) because the coarse grid solver does more work. However, this extra work is all matrix-vector products, which often costs less wall time than the communication-bound operations on the truncated levels. In GPU runs, the cycle complexity can increase by 2x or more while the solve time stays the same or even decreases. As you weak scale and add more levels, the relative cost of the coarse grid solve decreases and the cycle complexity with truncation approaches that without.

### 3) OpenMP with Kokkos on CPUs

The parallel performance of the SpGEMMs `PCAIR` uses to form much of the hierarchy can depend on the MPI decomposition. Having more unknowns per MPI rank can improve performance and scalability.

On CPUs, rather than using many MPI ranks per CPU (e.g., one per core), hybrid MPI/OpenMP can be beneficial. This is supported in `PCAIR` through Kokkos with an OpenMP backend. This can also reduce memory use, as the halo sizes throughout the hierarchy decrease.

## How to get faster solves with GPUs in parallel with `PCAIR`

Several techniques can improve the solve time on GPUs. Many of these take advantage of the fact that GPUs are well suited to trading extra FLOPS for less communication and less memory.

### 1) Many unknowns per GPU

GPU throughput improves significantly with many unknowns per MPI rank (e.g., tens of millions per GPU). This keeps the GPU saturated and ensures good performance on the top levels of the hierarchy. However, strong scaling studies still show reductions in wall time down to around 1M DOFs/rank.

### 2) Matrix-free smoothing

GPUs are well suited to repeatedly applying the same matrix-vector product. Turning on the option `-pc_air_matrix_free_polys` applies the GMRES polynomial smoother matrix-free. This uses no extra memory (only the polynomial coefficients are stored) and gives a better approximation to $A_{ff}^{-1}$ than the assembled fixed-sparsity version, at the cost of more SpMVs per smooth.

In practice, the extra SpMVs are memory-bandwidth bound rather than FLOP bound, so GPUs handle them well. For example, with 6th-order polynomials applied matrix-free, the per V-cycle time is only about 20% more than using the assembled approximation (which requires only one SpMV), despite requiring six SpMVs. The iteration count also typically improves.

### 3) Truncating the hierarchy

Truncating the hierarchy and applying a matrix-free coarse grid solver (see above) is often very effective on GPUs. GPUs are poorly suited to the small solves on the bottom levels of a multigrid hierarchy, where kernel launch overheads become significant and there is not enough work to hide communication. Truncation can give large speed-ups in the solve (e.g., 4x) and also reduces memory use.

## Resolving out-of-memory errors with `PCAIR`

AIR multigrid methods for asymmetric linear systems require considerable memory, particularly compared with classical multigrid methods for elliptic problems. They often require more memory than a direct LU factorisation, but the benefit is that scalable solves are possible in parallel.

Typical storage complexity (the sum of non-zeros across all matrices needed in the solve, relative to the top grid matrix) is around 5-6 on unstructured meshes and 10-13 on structured meshes. The higher values on structured meshes are due to the slower coarsening required. For comparison, elliptic multigrid methods typically require much less.

Besides decreasing the number of unknowns, there are several ways to decrease the memory required.

### 1) Matrix-free smoothing

Using `-pc_air_matrix_free_polys` avoids assembling and storing the approximate inverse on each level. This saves significant memory, as only the polynomial coefficients (a few scalars per level) need to be stored instead of a full sparse matrix.

### 2) F-point only smoothing

The default F-point only smoothing (`-pc_air_smooth_type f`) avoids storing $A_{cc}$ and $A_{cf}$ on each level. Switching to FC or FCF smoothing can nearly double the storage complexity, so only use it if the convergence improvement justifies the memory cost.

### 3) Truncating the hierarchy

Truncating the hierarchy (see above) reduces the total number of levels and hence the total amount of storage required.

### 4) Processor agglomeration

In parallel, the processor agglomeration (through ParMETIS) can require considerable memory when repartitioning. Disabling it with `-pc_air_processor_agglom 0` reduces peak memory, but can increase the run time of the setup and solve.

Instead of disabling it entirely, `-pc_air_processor_agglom_ratio` can control at what level it is first triggered. By lowering the value (e.g., `-pc_air_processor_agglom_ratio 0.1`), processor agglomeration will only be triggered further down the hierarchy, where the coarse grid matrices are smaller and less memory is needed.

### 5) Drop tolerances

The drop tolerances `-pc_air_a_drop` and `-pc_air_r_drop` can be increased to make the hierarchy more sparse, but this can negatively affect convergence.

### 6) Coarsening rate

Faster coarsenings (higher strong threshold) give fewer levels and lower memory use. The tradeoff is that faster coarsenings can hurt convergence.
