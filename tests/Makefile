# ~~~~~~~~~~~~~~~~~
# PFLARE - Steven Dargaville
# Makefile for tests
# Copied from $PETSC_DIR/share/petsc/Makefile.basic.user
# This uses the compilers and flags defined in the PETSc configuration
# ~~~~~~~~~~~~~~~~~

# Get the flags we have on input
CFLAGS_INPUT := $(CFLAGS)
FFLAGS_INPUT := $(FFLAGS)
CPPFLAGS_INPUT := $(CPPFLAGS)
FPPFLAGS_INPUT := $(FPPFLAGS)

# ~~~~~~~~~~~
# PFLARE specific changes
# ~~~~~~~~~~~
# Include directories - include top level directory in case compilers output modules there
INCLUDE := -I$(CURDIR)/../ -I../include

CPPFLAGS = $(INCLUDE)
FPPFLAGS = $(INCLUDE)
CPPFLAGS = $(INCLUDE)
CXXPPFLAGS = $(INCLUDE)

# Read in the petsc compile/linking variables and makefile rules
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# Include any additional flags we input
CFLAGS += $(CFLAGS_INPUT)
FFLAGS += $(FFLAGS_INPUT)
CPPFLAGS += $(CPPFLAGS_INPUT)
FPPFLAGS += $(FPPFLAGS_INPUT)

# Link to pflare - order is important
LDLIBS := -L$(LIBDIR) -lpflare -Wl,-rpath,$(LIBDIR):$(LDLIBS)

clean::
	$(RM) -f $(TEST_TARGETS)

# ~~~~~~~~~~~
# Run PFLARE tests
# ~~~~~~~~~~~
run_tests_load:
#
	@echo ""
	@echo "Test AIRG with GMRES polynomials for hyperbolic streaming problem"
	./ex12f -f data/mat_stream_2364 -ksp_max_it 5
	@echo "Test AIRG with GMRES polynomials for hyperbolic streaming problem, matrix-free smoothing"
	./ex12f -f data/mat_stream_2364 -pc_air_matrix_free_polys -ksp_max_it 5
#
	@echo ""
	@echo "Test AIRG with GMRES polynomials for hyperbolic streaming problem in C"
	./ex6 -f data/mat_stream_2364 -ksp_max_it 5
	@echo "Test AIRG with GMRES polynomials for hyperbolic streaming problem, matrix-free smoothing in C in parallel"
	$(MPIEXEC) -n 2 ./ex6 -f data/mat_stream_2364 -pc_air_matrix_free_polys -ksp_max_it 5
#
	@echo ""
	@echo "Test lAIR with GMRES polynomial smoothing for hyperbolic streaming problem"
	./ex12f -f data/mat_stream_2364 -pc_air_z_type lair -ksp_max_it 5
	@echo "Test lAIR with strong R tolerance with GMRES polynomial smoothing for hyperbolic streaming problem"
	./ex12f -f data/mat_stream_2364 -pc_air_z_type lair	-pc_air_strong_r_threshold 0.01 -ksp_max_it 5
# 
	@echo ""
	@echo "Test single level GMRES polynomial preconditioning for hyperbolic streaming problem in C"
	./ex6 -f data/mat_stream_2364 -pc_type pflareinv -pc_pflareinv_type power -ksp_max_it 21	
	@echo "Test single level GMRES polynomial preconditioning for hyperbolic streaming problem in C in parallel"
	$(MPIEXEC) -n 2 ./ex6 -f data/mat_stream_2364	-pc_type pflareinv -pc_pflareinv_type power -ksp_max_it 21
# 
	@echo ""
	@echo "Test single level GMRES polynomial preconditioning with the Newton basis matrix-free for hyperbolic streaming problem in C"
	./ex6 -f data/mat_stream_2364 -pc_type pflareinv -pc_pflareinv_type newton -pc_pflareinv_matrix_free -ksp_max_it 13
	@echo "Test single level GMRES polynomial preconditioning with the Newton basis matrix-free for hyperbolic streaming problem in C in parallel"
	$(MPIEXEC) -n 2 ./ex6 -f data/mat_stream_2364	-pc_type pflareinv -pc_pflareinv_type newton -pc_pflareinv_matrix_free -ksp_max_it 13
#
	@echo ""	 
	@echo "Test AIRG as an exact solver, truncating hierarchy and using high order mf GMRES poly in the\
	 Arnoldi basis as a coarse solver for hyperbolic streaming"
	./ex12f -f data/mat_stream_2364 -pc_air_strong_threshold 0.0 -pc_air_a_drop 0.0 -pc_air_r_drop 0.0 \
	 -pc_air_inverse_type jacobi -mg_coarse_ksp_type richardson -mg_coarse_ksp_max_it 5 -ksp_type richardson -ksp_norm_type unpreconditioned \
	 -pc_air_max_levels 30 -pc_air_coarsest_poly_order 18 \
	 -pc_air_coarsest_matrix_free_polys -pc_air_coarsest_inverse_type arnoldi -ksp_max_it 1
# 
	@echo ""		 
	@echo "Test AIRG as an exact solver, heavily truncating hierarchy and using high order mf GMRES poly in the\
	 Newton basis as a coarse solver for hyperbolic streaming"
	./ex12f -f data/mat_stream_2364 -pc_air_strong_threshold 0.0 -pc_air_a_drop 0.0 -pc_air_r_drop 0.0 \
	 -pc_air_inverse_type jacobi -ksp_type richardson -ksp_norm_type unpreconditioned \
	 -pc_air_max_levels 10 -pc_air_coarsest_poly_order 60 \
	 -pc_air_coarsest_matrix_free_polys -pc_air_coarsest_inverse_type newton -pc_air_max_luby_steps 3 -ksp_max_it 1
	@echo "Test AIRG as an exact solver, heavily truncating hierarchy and using high order mf GMRES poly in the\
	 Newton basis as a coarse solver for hyperbolic streaming in parallel"
	$(MPIEXEC) -n 2 ./ex6 -f data/mat_stream_2364 -pc_air_strong_threshold 0.0 -pc_air_a_drop 0.0 -pc_air_r_drop 0.0 \
	 -pc_air_inverse_type jacobi -ksp_type richardson -ksp_norm_type unpreconditioned \
	 -pc_air_max_levels 10 -pc_air_coarsest_poly_order 60 \
	 -pc_air_coarsest_matrix_free_polys -pc_air_coarsest_inverse_type newton -pc_air_max_luby_steps 3 -ksp_max_it 1	
#
	@echo ""
	@echo "Test AIRG with GMRES polynomials for hyperbolic streaming problem with coefficients calculated on subcomms"
	$(MPIEXEC) -n 2 ./ex12f -f data/mat_stream_2364 -pc_air_subcomm -pc_air_inverse_type arnoldi -pc_air_coarsest_subcomm \
	 -pc_air_coarsest_inverse_type arnoldi -ksp_max_it 5
# 
	@echo ""
	@echo "Test PMISR DDC CF splitting in C"
	./ex6_cf_splitting  -f data/mat_stream_2364
	@echo "Test PMISR DDC CF splitting in C in parallel"
	$(MPIEXEC) -n 2 ./ex6_cf_splitting  -f data/mat_stream_2364
#
	@echo ""
	@echo "Test AIRG with GMRES polynomials in indefinite problem with zero diagonals"
	./ex6 -f data/e05r0100_petsc -b_in_f 0 -ksp_max_it 15	

run_tests_no_load:
#
	@echo ""
	@echo "Test AIRG with GMRES polynomials for 2D finite difference stencil"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -ksp_max_it 5
	@echo "Test AIRG with GMRES polynomials Arnoldi basis for 2D finite difference stencil"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -pc_air_inverse_type arnoldi -pc_air_coarsest_inverse_type arnoldi -ksp_max_it 5
	@echo "Test AIRG with GMRES polynomials for 2D finite difference stencil in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -ksp_max_it 5
	@echo "Test AIRG with GMRES polynomials Arnoldi basis for 2D finite difference stencil in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -pc_air_inverse_type arnoldi -pc_air_coarsest_inverse_type arnoldi -ksp_max_it 5
	@echo ""
	@echo "Test lAIR with GMRES polynomial smoothing for 2D finite difference stencil in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -pc_air_z_type lair -ksp_max_it 4
# 
	@echo ""
	@echo "Test single level GMRES polynomial preconditioning for 2D finite difference stencil"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type pflareinv -pc_pflareinv_type power -ksp_max_it 8
	@echo "Test single level GMRES polynomial preconditioning for 2D finite difference stencil in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type pflareinv -pc_pflareinv_type power -ksp_max_it 8
#
	@echo ""
	@echo "Test AIRG with Neumann polynomials for 2D finite difference stencil"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -pc_air_inverse_type neumann -ksp_max_it 5
	@echo "Test AIRG with Neumann polynomials for 2D finite difference stencil in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air  -pc_air_inverse_type neumann -ksp_max_it 5
	@echo "Test AIRG with Neumann polynomials for 2D finite difference stencil, matrix-free smoothing"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -pc_air_inverse_type neumann -pc_air_matrix_free_polys -ksp_max_it 5
	@echo "Test AIRG with Neumann polynomials for 2D finite difference stencil, matrix-free smoothing in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air  -pc_air_inverse_type neumann -pc_air_matrix_free_polys -ksp_max_it 5
#
	@echo ""
	@echo "Test AIRG with SAIs for 2D finite difference stencil"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -pc_air_inverse_type sai -ksp_max_it 5
	@echo "Test AIRG with SAI for 2D finite difference stencil in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air  -pc_air_inverse_type sai -ksp_max_it 5
# 
	@echo ""
	@echo "Test AIRG with ISAIs for 2D finite difference stencil"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -pc_air_inverse_type isai -ksp_max_it 5
	@echo "Test AIRG with ISAIs for 2D finite difference stencil in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air  -pc_air_inverse_type isai -ksp_max_it 5
# 
	@echo ""
	@echo "Test AIRG with Weighted Jacobi for 2D finite difference stencil"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -pc_air_inverse_type wjacobi -ksp_max_it 8
	@echo "Test AIRG with Weighted Jacobi for 2D finite difference stencil in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air  -pc_air_inverse_type wjacobi -ksp_max_it 8
# 
	@echo ""
	@echo "Test AIRG with Unweighted Jacobi for 2D finite difference stencil"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -pc_air_inverse_type jacobi -ksp_max_it 5
	@echo "Test AIRG with Unweighted Jacobi for 2D finite difference stencil in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air  -pc_air_inverse_type jacobi -ksp_max_it 5
# 
	@echo ""
	@echo "Test AIRG as an exact solver for 2D finite difference stencil"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air \
	 -pc_air_strong_threshold 0.0 -pc_air_a_drop 0.0 -pc_air_r_drop 0.0 -pc_air_inverse_type jacobi \
	 -mg_coarse_ksp_type richardson -mg_coarse_ksp_max_it 10 -ksp_type richardson -ksp_norm_type unpreconditioned -ksp_max_it 1
	@echo "Test AIRG as an exact solver for 2D finite difference stencil in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air \
	 -pc_air_strong_threshold 0.0 -pc_air_a_drop 0.0 -pc_air_r_drop 0.0 -pc_air_inverse_type jacobi \
	 -mg_coarse_ksp_type richardson -mg_coarse_ksp_max_it 10 -ksp_type richardson -ksp_norm_type unpreconditioned -ksp_max_it 1
# 
	@echo ""
	@echo "Test AIRG with GMRES polynomials with PC reused with no sparsity change"
	./ex6f -m 10 -n 10 -ksp_max_it 10
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change"
	./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3	
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change and not classical prolong"
	./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3	-pc_air_one_point_classical_prolong 0
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change and not classical prolong in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_one_point_classical_prolong 0	
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change and strong R threshold"
	./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -pc_air_strong_r_threshold 0.01 -ksp_max_it 3
	@echo "Test lAIR with GMRES polynomials with PC regenerated with no sparsity change"
	./ex6f -m 10 -n 10 -regen -pc_air_z_type lair -pc_air_reuse_sparsity -ksp_max_it 3
	@echo "Test lAIR SAI with GMRES polynomials with PC regenerated with no sparsity change"
	./ex6f -m 10 -n 10 -regen -pc_air_z_type lair_sai -pc_air_reuse_sparsity -ksp_max_it 3
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change with constrain w"
	./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_constrain_w
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change with constrain w"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_constrain_w	
# 
	@echo ""
	@echo "Test AIRG with 0th order GMRES polynomials with PC reused with no sparsity change"
	./ex6f -m 10 -n 10 -ksp_max_it 10 -pc_air_poly_order 0
	@echo "Test AIRG with 0th order GMRES polynomials with PC reused with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -ksp_max_it 10 -pc_air_poly_order 0	
	@echo "Test AIRG with 0th order GMRES polynomials with PC regenerated with no sparsity change"
	./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_poly_order 0
	@echo "Test AIRG with 0th order GMRES polynomials with PC regenerated with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_poly_order 0
# 
	@echo ""
	@echo "Test AIRG with 1st order GMRES polynomials with PC reused with no sparsity change"
	./ex6f -m 10 -n 10 -ksp_max_it 10 -pc_air_poly_order 1
	@echo "Test AIRG with 1st order GMRES polynomials with PC reused with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -ksp_max_it 10 -pc_air_poly_order 1
	@echo "Test AIRG with 1st order GMRES polynomials with PC regenerated with no sparsity change"
	./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_poly_order 1
	@echo "Test AIRG with 1st order GMRES polynomials with PC regenerated with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_poly_order 1		
# 
	@echo ""
	@echo "Test AIRG with 2nd order GMRES polynomials with PC reused with no sparsity change"
	./ex6f -m 10 -n 10 -ksp_max_it 10 -pc_air_poly_order 2 -pc_air_inverse_sparsity_order 2
	@echo "Test AIRG with 2nd order GMRES polynomials with PC reused with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -ksp_max_it 10 -pc_air_poly_order 2 -pc_air_inverse_sparsity_order 2
	@echo "Test AIRG with 2nd order GMRES polynomials with PC regenerated with no sparsity change"
	./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_poly_order 2 -pc_air_inverse_sparsity_order 2
	@echo "Test AIRG with 2nd order GMRES polynomials with PC regenerated with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_poly_order 2 -pc_air_inverse_sparsity_order 2
# 
	@echo ""
	@echo "Test AIRG with GMRES polynomials with PC reused with no sparsity change with 0th order fixed sparsity"
	./ex6f -m 10 -n 10 -ksp_max_it 10 -pc_air_inverse_sparsity_order 0
	@echo "Test AIRG with GMRES polynomials with PC reused with no sparsity change in parallel with 0th order fixed sparsity"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -ksp_max_it 10 -pc_air_inverse_sparsity_order 0	
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change with 0th order fixed sparsity"
	./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_inverse_sparsity_order 0
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change in parallel with 0th order fixed sparsity"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_inverse_sparsity_order 0	
# 
	@echo ""
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change and polynomial coeffs stored"
	./ex6f_getcoeffs -m 10 -n 10 -ksp_max_it 3
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change and polynomial coeffs stored and lumping"
	./ex6f_getcoeffs -m 10 -n 10 -pc_air_a_lump -ksp_max_it 3
# 
	@echo ""
	@echo "Test single level GMRES polynomials with PC reused with no sparsity change"
	./ex6f -m 10 -n 10 -pc_type pflareinv -ksp_max_it 8
	@echo "Test single level GMRES polynomials with PC regenerated with no sparsity change"
	./ex6f -m 10 -n 10 -pc_type pflareinv -regen -ksp_max_it 8
# 
	@echo ""
	@echo "Test solving isotropic diffusion with fast coarsening and near-nullspace"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 50 -da_grid_y 50 -pc_type air -pc_air_z_type lair \
	 -pc_air_one_c_smooth -pc_air_symmetric -pc_air_constrain_z \
	 -pc_air_cf_splitting_type agg -pc_air_a_drop 1e-5 -pc_air_a_lump\
	 -pc_air_r_drop 0 -ksp_rtol 1e-10 -ksp_pc_side right -ksp_max_it 14
	@echo "Test solving isotropic diffusion with fast coarsening and near-nullspace in parallel"
	$(MPIEXEC) -n 4 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 50 -da_grid_y 50 -pc_type air -pc_air_z_type lair \
	 -pc_air_one_c_smooth -pc_air_symmetric -pc_air_constrain_z \
	 -pc_air_cf_splitting_type agg -pc_air_a_drop 1e-5 -pc_air_a_lump\
	 -pc_air_r_drop 0 -ksp_rtol 1e-10 -ksp_pc_side right -ksp_max_it 14
# 
	@echo ""
	@echo "Test AIRG on steady 1D advection"
	./adv_1d -n 1000 -ksp_rtol 1e-10 -ksp_atol 1e-50 -ksp_pc_side right \
	 -pc_air_coarsest_inverse_type newton -pc_air_coarsest_poly_order 10 -pc_air_coarsest_matrix_free_polys -ksp_max_it 1
	@echo "Test AIRG on steady 1D advection in parallel"
	$(MPIEXEC) -n 2 ./adv_1d -n 1000 -ksp_rtol 1e-10 -ksp_atol 1e-50 -ksp_pc_side right \
	 -pc_air_coarsest_inverse_type newton -pc_air_coarsest_poly_order 10 -pc_air_coarsest_matrix_free_polys -ksp_max_it 10
# 
	@echo ""
	@echo "Test lAIR on steady 2D structured advection"
	./adv_diff_2d -da_grid_x 50 -da_grid_y 50 -pc_type air -ksp_pc_side right \
	 -ksp_rtol 1e-10 -ksp_atol 1e-50 -pc_air_a_lump -pc_air_a_drop 1e-4 -pc_air_one_c_smooth \
	 -pc_air_z_type lair -pc_air_inverse_type wjacobi -ksp_max_it 10
# 
	@echo ""
	@echo "Test AIRG on steady 2D structured advection with 0th order sparsity GMRES poly C smooth and faster coarsening"
	./adv_diff_2d -da_grid_x 50 -da_grid_y 50 -pc_type air -ksp_pc_side right \
	 -ksp_rtol 1e-10 -ksp_atol 1e-50 -pc_air_a_lump -pc_air_a_drop 1e-4 -pc_air_one_c_smooth \
	 -pc_air_c_inverse_sparsity_order 0 -pc_air_strong_threshold 0.99 -pc_air_ddc_fraction 0.0 -ksp_max_it 7
# 
	@echo ""
	@echo "Test high order GMRES polynomials in small linear system"
	./adv_diff_2d -da_grid_x 5 -da_grid_y 5 -pc_type pflareinv -pc_pflareinv_type newton \
	 -pc_pflareinv_matrix_free -pc_pflareinv_order 16 -ksp_max_it 1
	@echo "Test high order GMRES polynomials in small linear system - slightly bigger"
	./adv_diff_2d -da_grid_x 10 -da_grid_y 10 -pc_type pflareinv -pc_pflareinv_type newton \
	 -pc_pflareinv_matrix_free -pc_pflareinv_order 50 -ksp_max_it 1
# 
	@echo ""
	@echo "Test auto truncation with matrix-free Newton polynomials as coarse grid solver" 
	./adv_diff_2d -da_grid_x 10 -da_grid_y 10 -ksp_type richardson -pc_type air \
	 -pc_air_coarsest_inverse_type newton -pc_air_coarsest_poly_order 10 -pc_air_coarsest_matrix_free_polys \
	 -pc_air_auto_truncate_start_level 1 -pc_air_auto_truncate_tol 1e-2 -ksp_max_it 3
#	 
# ~~~~~~~~~~~~~~~~~~~~~~~	 
# Include kokkos examples	 
# ~~~~~~~~~~~~~~~~~~~~~~~
ifeq ($(PETSC_HAVE_KOKKOS),1)
# 
	@echo ""
	@echo "Test AIRG on steady 1D advection KOKKOS"
	./adv_1dk -n 1000 -ksp_rtol 1e-10 -ksp_atol 1e-50 -ksp_pc_side right \
	 -pc_air_coarsest_inverse_type newton -pc_air_coarsest_poly_order 10 -pc_air_coarsest_matrix_free_polys -ksp_max_it 1 \
	 -vec_type kokkos -mat_type aijkokkos
	@echo "Test AIRG on steady 1D advection in parallel KOKKOS"
	$(MPIEXEC) -n 2 ./adv_1dk -n 1000 -ksp_rtol 1e-10 -ksp_atol 1e-50 -ksp_pc_side right \
	 -pc_air_coarsest_inverse_type newton -pc_air_coarsest_poly_order 10 -pc_air_coarsest_matrix_free_polys -ksp_max_it 10 \
	 -vec_type kokkos -mat_type aijkokkos	

endif
# ~~~~~~~~~~~~~~~~~~~~~~~	 
# Include CUDA examples	 
# ~~~~~~~~~~~~~~~~~~~~~~~
ifeq ($(PETSC_HAVE_CUDA),1)
# 
	@echo ""
	@echo "Test AIRG on steady 2D structured advection with 0th order sparsity GMRES poly C smooth and faster coarsening CUDA"
	./adv_diff_2d -da_grid_x 50 -da_grid_y 50 -pc_type air -ksp_pc_side right \
	 -ksp_rtol 1e-10 -ksp_atol 1e-50 -pc_air_a_lump -pc_air_a_drop 1e-4 -pc_air_one_c_smooth \
	 -pc_air_c_inverse_sparsity_order 0 -pc_air_strong_threshold 0.99 -pc_air_ddc_fraction 0.0 -ksp_max_it 7 \
	 -dm_mat_type aijcusparse -dm_vec_type cuda

endif
# ~~~~~~~~~~~~~~~~~~~~~~~	 
# Include HIP examples	 
# ~~~~~~~~~~~~~~~~~~~~~~~
ifeq ($(PETSC_HAVE_HIP),1)
# 
	@echo ""
	@echo "Test AIRG on steady 2D structured advection with 0th order sparsity GMRES poly C smooth and faster coarsening HIP"
	./adv_diff_2d -da_grid_x 50 -da_grid_y 50 -pc_type air -ksp_pc_side right \
	 -ksp_rtol 1e-10 -ksp_atol 1e-50 -pc_air_a_lump -pc_air_a_drop 1e-4 -pc_air_one_c_smooth \
	 -pc_air_c_inverse_sparsity_order 0 -pc_air_strong_threshold 0.99 -pc_air_ddc_fraction 0.0 -ksp_max_it 7 \
	 -dm_mat_type aijhipsparse -dm_vec_type hip

endif