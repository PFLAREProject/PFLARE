## Options

A brief description of the available options in PFLARE is given below and their default values.

All options can be set either through command line arguments or programmatically via the routine names listed in the table below. In Python (via the `pflare` module), each routine is available under a snake\_case name: for example, `PCAIRSetZType` → `pflare.pcair_set_z_type`, `PCAIRGetPolyOrder` → `pflare.pcair_get_poly_order`. The complete list of Python names and the associated enum constants (e.g. `PFLAREINV_ARNOLDI`, `AIR_Z_LAIR`, `CF_PMISR_DDC`) are defined in `python/pflare.py`.

### PCPFLAREINV

   | Command line  | Routine | Description | Default |
   | ------------- | -- | ------------- | --- |
   | ``-pc_pflareinv_type``  |  PCPFLAREINVGetType  PCPFLAREINVSetType  | The inverse type, given above | arnoldi |   
   | ``-pc_pflareinv_poly_order``  |  PCPFLAREINVGetPolyOrder  PCPFLAREINVSetPolyOrder  | If using a polynomial inverse type, this determines the order of the polynomial | 6 |
   | ``-pc_pflareinv_sparsity_order``  |  PCPFLAREINVGetSparsityOrder  PCPFLAREINVSetSparsityOrder  | This power of A is used as the sparsity in assembled inverses | 1 |   
   | ``-pc_pflareinv_matrix_free``  |  PCPFLAREINVGetMatrixFree  PCPFLAREINVSetMatrixFree  | Is the inverse applied matrix free, or is an assembled matrix built and used | false |
   | ``-pc_pflareinv_reuse_poly_coeffs``  |  PCPFLAREINVGetReusePolyCoeffs  PCPFLAREINVSetReusePolyCoeffs  | Don't recompute the polynomial inverse coefficients during setup with reuse | false |                                        

### PCAIR

#### Hierarchy options

   | Command line  | Routine | Description | Default |
   | ------------- | -- | ------------- | --- |
   | ``-pc_air_print_stats_timings``  |  PCAIRGetPrintStatsTimings  PCAIRSetPrintStatsTimings  | Print out statistics about the multigrid hierarchy and timings | false |       
   | ``-pc_air_max_levels``  |  PCAIRGetMaxLevels  PCAIRSetMaxLevels  | Maximum number of levels in the hierarchy | 300 |
   | ``-pc_air_coarse_eq_limit``  |  PCAIRGetCoarseEqLimit  PCAIRSetCoarseEqLimit  | Minimum number of global unknowns on the coarse grid | 6 |   
   | ``-pc_air_auto_truncate_start_level``  |  PCAIRGetAutoTruncateStartLevel  PCAIRSetAutoTruncateStartLevel  | Build a coarse solver on each level from this one and use it to determine if we can truncate the hierarchy | -1 |
   | ``-pc_air_auto_truncate_tol``  |  PCAIRGetAutoTruncateTol  PCAIRSetAutoTruncateTol  | Tolerance used to determine if the coarse solver is good enough to truncate at a given level | 1e-14 | 
   | ``-pc_air_r_drop``  |  PCAIRGetRDrop  PCAIRSetRDrop  | Drop tolerance applied to R on each level after it is built | 0.01 |
   | ``-pc_air_a_drop``  |  PCAIRGetADrop  PCAIRSetADrop  | Drop tolerance applied to the coarse matrix on each level after it is built | 0.0001 |
   | ``-pc_air_a_lump``  |  PCAIRSetALump  PCAIRSetALump  | Lump to the diagonal rather than drop for the coarse matrix | false |         

#### Parallel options

   | Command line  | Routine | Description | Default |
   | ------------- | -- | ------------- | --- |
   | ``-pc_air_processor_agglom``  |  PCAIRGetProcessorAgglom  PCAIRSetProcessorAgglom  | Whether to use a graph partitioner to repartition the coarse grids and reduce the number of active MPI ranks  | true |
   | ``-pc_air_processor_agglom_ratio``  |  PCAIRGetProcessorAgglomRatio  PCAIRSetProcessorAgglomRatio  | The local to non-local nnzs ratio that is used to trigger processor agglomeration on all levels  | 2.0 | 
   | ``-pc_air_processor_agglom_factor``  |  PCAIRGetProcessorAgglomFactor  PCAIRSetProcessorAgglomFactor  | What factor to reduce the number of active MPI ranks by each time when doing processor agglomeration  | 2 | 
   | ``-pc_air_process_eq_limit``  |  PCAIRGetProcessEqLimit  PCAIRSetProcessEqLimit  | If on average there are fewer than this number of equations per rank processor agglomeration will be triggered  | 50 |    
   | ``-pc_air_subcomm``  |  PCAIRGetSubcomm  PCAIRSetSubcomm  | If computing a polynomial inverse with type arnoldi or newton and we have performed processor agglomeration, we can exclude the MPI ranks with no non-zeros from reductions in parallel by moving onto a subcommunicator | false |  

#### CF splitting options

   | Command line  | Routine | Description | Default |
   | ------------- | -- | ------------- | --- |
   | ``-pc_air_cf_splitting_type``  |  PCAIRGetCFSplittingType  PCAIRSetCFSplittingType  | The type of CF splitting to use, given above | pmisr_ddc |    
   | ``-pc_air_strong_threshold``  |  PCAIRGetStrongThreshold  PCAIRSetStrongThreshold  | The strong threshold to use in the CF splitting | 0.5 |
   | ``-pc_air_max_luby_steps``  |  PCAIRGetMaxLubySteps  PCAIRSetMaxLubySteps  | If using CF splitting type pmisr_ddc, diag_dom, pmis, or pmis_dist2, this is the maximum number of Luby steps to use. If negative, use as many steps as necessary | -1 |   
   | ``-pc_air_ddc_its``  |  PCAIRGetDDCIts  PCAIRSetDDCIts  | If using CF splitting type pmisr_ddc, this is the number of iterations of DDC performed | 1 |   
   | ``-pc_air_ddc_fraction``  |  PCAIRGetDDCFraction  PCAIRSetDDCFraction  | If using CF splitting type pmisr_ddc, this is the local fraction of F points to convert to C points based on diagonal dominance. If negative, any row which has a diagonal dominance ratio less than the absolute value will be converted from F to C | 0.1 |

#### Approximate inverse options

   | Command line  | Routine | Description | Default |
   | ------------- | -- | ------------- | --- |
   | ``-pc_air_inverse_type``  |  PCAIRGetInverseType  PCAIRSetInverseType  | The inverse type, given above | arnoldi |
   | ``-pc_air_poly_order``  |  PCAIRGetPolyOrder  PCAIRSetPolyOrder  | If using a polynomial inverse type, this determines the order of the polynomial | 6 |
   | ``-pc_air_inverse_sparsity_order``  |  PCAIRGetInverseSparsityOrder  PCAIRSetInverseSparsityOrder  | This power of A is used as the sparsity in assembled inverses | 1 |        
   | ``-pc_air_diag_scale_polys``  |  PCAIRGetDiagScalePolys  PCAIRSetDiagScalePolys  | If using a polynomial inverse type, diagonally scale before computing | false (if inverse type neumann this is always true and cannot be overridden) |    
   | ``-pc_air_matrix_free_polys``  |  PCAIRGetMatrixFreePolys  PCAIRSetMatrixFreePolys  | Do smoothing matrix-free if possible | false |   
   | ``-pc_air_smooth_type``  |  PCAIRGetSmoothType  PCAIRSetSmoothType  | Type and number of smooths | ff |
   | ``-pc_air_full_smoothing_up_and_down``  |  PCAIRGetFullSmoothingUpAndDown  PCAIRSetFullSmoothingUpAndDown  | Up and down smoothing on all points at once, rather than only down F and C smoothing which is the default  | false |     
   | ``-pc_air_c_inverse_type``  |  PCAIRGetCInverseType  PCAIRSetCInverseType  | The inverse type for the C smooth, given above. If unset this defaults to the same as the F point smoother | -pc_air_inverse_type |
   | ``-pc_air_c_poly_order``  |  PCAIRGetCPolyOrder  PCAIRSetCPolyOrder  | If using a polynomial inverse type, this determines the order of the polynomial for the C smooth. If unset this defaults to the same as the F point smoother | -pc_air_poly_order |
   | ``-pc_air_c_inverse_sparsity_order``  |  PCAIRGetCInverseSparsityOrder  PCAIRSetCInverseSparsityOrder  | This power of A is used as the sparsity in assembled inverses for the C smooth. If unset this defaults to the same as the F point smoother | -pc_air_inverse_sparsity_order |    
   

#### Grid transfer options

   | Command line  | Routine | Description | Default |
   | ------------- | -- | ------------- | --- |
   | ``-pc_air_one_point_classical_prolong``  |  PCAIRGetOnePointClassicalProlong  PCAIRSetOnePointClassicalProlong  | Use a one-point classical prolongator, instead of an approximate ideal prolongator | true |   
   | ``-pc_air_symmetric``  |  PCAIRGetSymmetric  PCAIRSetSymmetric  | Do we define our prolongator as R^T?  | false |     
   | ``-pc_air_strong_r_threshold``  |  PCAIRGetStrongRThreshold  PCAIRSetStrongRThreshold  | Threshold to drop when forming the grid-transfer operators  | 0.0 |
| ``-pc_air_z_type``  |  PCAIRGetZType  PCAIRSetZType  | Type of grid-transfer operator, see above  | product |
| ``-pc_air_lair_distance``  |  PCAIRGetLairDistance  PCAIRSetLairDistance  | If Z type is lair or lair_sai, this defines the distance of the grid-transfer operators  | 2 |          
   | ``-pc_air_constrain_w``  |  PCAIRGetConstrainW  PCAIRSetConstrainW  | Apply constraints to the prolongator. If enabled, by default it will smooth the constant vector and force the prolongator to interpolate it exactly. Can use MatSetNearNullSpace to give other vectors   | false |
   | ``-pc_air_constrain_z``  |  PCAIRGetConstrainZ  PCAIRSetConstrainZ  | Apply constraints to the restrictor. If enabled, by default it will smooth the constant vector and force the restrictor to restrict it exactly. Can use MatSetNearNullSpace to give other vectors   | false |
   | ``-pc_air_improve_w_its``  |  PCAIRGetImproveWIts  PCAIRSetImproveWIts  | Apply a number of Richardson iterations to improve the approximate prolongator. Uses the existing W as an initial guess.   | 0 |    
   | ``-pc_air_improve_z_its``  |  PCAIRGetImproveZIts  PCAIRSetImproveZIts  | Apply a number of Richardson iterations to improve the approximate restrictor. Uses the existing Z as an initial guess.    | 0 |  

#### Coarse grid solver options

   | Command line  | Routine | Description | Default |
   | ------------- | -- | ------------- | --- |
   | ``-pc_air_coarsest_inverse_type``  |  PCAIRGetCoarsestInverseType  PCAIRSetCoarsestInverseType  | Coarse grid inverse type, given above | arnoldi |
   | ``-pc_air_coarsest_poly_order``  |  PCAIRGetCoarsestPolyOrder  PCAIRSetCoarsestPolyOrder  | Coarse grid polynomial order | 6 |
   | ``-pc_air_coarsest_inverse_sparsity_order``  |  PCAIRGetCoarsestInverseSparsityOrder  PCAIRSetCoarsestInverseSparsityOrder  | Coarse grid sparsity order | 1 |
   | ``-pc_air_coarsest_matrix_free_polys``  |  PCAIRGetCoarsestMatrixFreePolys  PCAIRSetCoarsestMatrixFreePolys  | Do smoothing matrix-free if possible on the coarse grid | false |
   | ``-pc_air_coarsest_diag_scale_polys``  |  PCAIRGetCoarsestDiagScalePolys  PCAIRSetCoarsestDiagScalePolys  | If using a polynomial inverse type, diagonally scale on the coarse grid before computing | false (if coarsest inverse type neumann this is always true and cannot be overridden) |                 
   | ``-pc_air_coarsest_subcomm``  |  PCAIRGetCoarsestSubcomm  PCAIRSetCoarsestSubcomm  | Use a subcommunicator on the coarse grid | false |

#### Reuse options

   | Command line  | Routine | Description | Default |
   | ------------- | -- | ------------- | --- |
   | ``-pc_air_reuse_sparsity``  |  PCAIRGetReuseSparsity  PCAIRSetReuseSparsity  | Store temporary data to allow fast setup with reuse | false |
   | ``-pc_air_reuse_amount``  |  PCAIRGetReuseAmount  PCAIRSetReuseAmount  | Control how much data is stored when reuse sparsity is enabled: 1=CF splitting only, 2=CF splitting + SpGEMM sparsity, 3=everything | 3 |
   | ``-pc_air_reuse_poly_coeffs``  |  PCAIRGetReusePolyCoeffs  PCAIRSetReusePolyCoeffs  | Don't recompute the polynomial inverse coefficients during setup with reuse | false |         
