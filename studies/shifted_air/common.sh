# Recommended diagonal update options from the report (Section 6)
OPTS="-ksp_type gmres -ksp_rtol 1e-8 -ksp_max_it 300 -ksp_gmres_restart 300 -pc_air_smooth_type fc -pc_air_inverse_type newton_no_extra -pc_air_matrix_free_polys -pc_air_coarsest_inverse_type newton_no_extra -pc_air_coarsest_matrix_free_polys -pc_air_diag_scale_polys -pc_air_coarsest_diag_scale_polys"
