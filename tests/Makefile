# ~~~~~~~~~~~~~~~~~
# PFLARE - Steven Dargaville
# Makefile for tests
# Copied from $PETSC_DIR/share/petsc/Makefile.basic.user
# This uses the compilers and flags defined in the PETSc configuration
# ~~~~~~~~~~~~~~~~~

# Get the flags we have on input
# These are appended to the flags set by PETSc
# so that users can add their own flags
# but not override the PETSc ones which we use for our builds
CFLAGS_INPUT := $(CFLAGS)
FFLAGS_INPUT := $(FFLAGS)
FPPFLAGS_INPUT := $(FPPFLAGS)
CPPFLAGS_INPUT := $(CPPFLAGS)
CXXPPFLAGS_INPUT := $(CXXPPFLAGS)
CXXFLAGS_INPUT := $(CXXFLAGS)
CUDAC_FLAGS_INPUT := $(CUDAC_FLAGS)
MPICXX_INCLUDES_INPUT := $(MPICXX_INCLUDES)
HIPC_FLAGS_INPUT := $(HIPC_FLAGS)
SYCLC_FLAGS_INPUT := $(SYCLC_FLAGS)

# ~~~~~~~~~~~
# PFLARE specific changes
# ~~~~~~~~~~~
# Include directories - include top level directory in case compilers output modules there
INCLUDE := -I$(CURDIR)/../ -I../include

# Read in the petsc compile/linking variables and makefile rules
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# We then have to add the flags back in after the petsc rules/variables
# have overwritten
override CFLAGS += $(CFLAGS_INPUT) $(INCLUDE)
override FFLAGS += $(FFLAGS_INPUT) $(INCLUDE)
override FPPFLAGS += $(FPPFLAGS_INPUT) $(INCLUDE)
override CPPFLAGS += $(CPPFLAGS_INPUT) $(INCLUDE)
override CXXPPFLAGS += $(CXXPPFLAGS_INPUT) $(INCLUDE)
override CXXFLAGS += $(CXXFLAGS_INPUT) $(INCLUDE)
override CUDAC_FLAGS += $(CUDAC_FLAGS_INPUT) $(INCLUDE)
override MPICXX_INCLUDES += $(MPICXX_INCLUDES_INPUT) $(INCLUDE)
override HIPC_FLAGS += $(HIPC_FLAGS_INPUT) $(INCLUDE)
override SYCLC_FLAGS += $(SYCLC_FLAGS_INPUT) $(INCLUDE)

# Link to pflare - order is important 
# pflare must be first for static linking
LDFLAGS += -Wl,-rpath,$(LIBDIR)
LDLIBS  := -L$(LIBDIR) -lpflare $(LDLIBS)

EXISTING_LD := $(LD_LIBRARY_PATH)
EXISTING_DYLD := $(DYLD_FALLBACK_LIBRARY_PATH)

ifeq ($(shell uname -s 2>/dev/null),Darwin)
  export DYLD_FALLBACK_LIBRARY_PATH=$(LIBDIR)$(if $(EXISTING_DYLD),:$(EXISTING_DYLD))
else
  export LD_LIBRARY_PATH=$(LIBDIR)$(if $(EXISTING_LD),:$(EXISTING_LD))
endif

clean::
	$(RM) -f $(TEST_TARGETS)

# ~~~~~~~~~~~
# Run PFLARE tests
# ~~~~~~~~~~~

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_tests_load_serial:
#
	@echo ""
	@echo "Test AIRG with GMRES polynomials for hyperbolic streaming problem with default options"
	./ex12f -f data/mat_stream_2364 -ksp_max_it 5	
	@echo "Test AIRG with GMRES polynomials for hyperbolic streaming problem"
	./ex12f -f data/mat_stream_2364 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -ksp_max_it 5
	@echo "Test AIRG with GMRES polynomials for hyperbolic streaming problem with fcf smoothing"
	./ex12f -f data/mat_stream_2364 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -ksp_max_it 5 -pc_air_smooth_type fcf
	@echo "Test AIRG with GMRES polynomials for hyperbolic streaming problem, matrix-free smoothing"
	./ex12f -f data/mat_stream_2364 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -pc_air_matrix_free_polys -ksp_max_it 5
#
	@echo ""
	@echo "Test AIRG with GMRES polynomials for hyperbolic streaming problem in C"
	./ex6 -f data/mat_stream_2364 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -ksp_max_it 5
#
	@echo ""
	@echo "Test lAIR with GMRES polynomial smoothing for hyperbolic streaming problem"
	./ex12f -f data/mat_stream_2364 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -pc_air_z_type lair -ksp_max_it 5
	@echo "Test lAIR with strong R tolerance with GMRES polynomial smoothing for hyperbolic streaming problem"
	./ex12f -f data/mat_stream_2364 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -pc_air_z_type lair -pc_air_strong_r_threshold 0.01 -ksp_max_it 5
# 
	@echo ""
	@echo "Test single level GMRES polynomial preconditioning for hyperbolic streaming problem in C with default options"
	./ex6 -f data/mat_stream_2364 -pc_type pflareinv -ksp_max_it 21		
	@echo "Test single level GMRES polynomial preconditioning for hyperbolic streaming problem in C"
	./ex6 -f data/mat_stream_2364 -pc_type pflareinv -pc_pflareinv_type power -ksp_max_it 21	
# 
	@echo ""
	@echo "Test single level GMRES polynomial preconditioning with the Newton basis matrix-free for hyperbolic streaming problem in C"
	./ex6 -f data/mat_stream_2364 -pc_type pflareinv -pc_pflareinv_type newton -pc_pflareinv_matrix_free -ksp_max_it 13
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
# 
	@echo ""
	@echo "Test PMISR DDC CF splitting in C"
	./ex6_cf_splitting  -f data/mat_stream_2364
#
	@echo ""
	@echo "Test AIRG with GMRES polynomials in indefinite problem with zero diagonals"
	./ex6 -f data/e05r0100_petsc -b_in_f 0 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -ksp_max_it 26

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_tests_load_parallel:
#
	@echo "Test AIRG with GMRES polynomials for hyperbolic streaming problem, matrix-free smoothing in C in parallel"
	$(MPIEXEC) -n 2 ./ex6 -f data/mat_stream_2364 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -pc_air_matrix_free_polys -ksp_max_it 5
#
	@echo "Test single level GMRES polynomial preconditioning for hyperbolic streaming problem in C in parallel"
	$(MPIEXEC) -n 2 ./ex6 -f data/mat_stream_2364	-pc_type pflareinv -pc_pflareinv_type power -ksp_max_it 21
# 
	@echo "Test single level GMRES polynomial preconditioning with the Newton basis matrix-free for hyperbolic streaming problem in C in parallel"
	$(MPIEXEC) -n 2 ./ex6 -f data/mat_stream_2364	-pc_type pflareinv -pc_pflareinv_type newton -pc_pflareinv_matrix_free -ksp_max_it 13
#
	@echo "Test AIRG as an exact solver, heavily truncating hierarchy and using high order mf GMRES poly in the\
	 Newton basis as a coarse solver for hyperbolic streaming in parallel"
	$(MPIEXEC) -n 2 ./ex6 -f data/mat_stream_2364 -pc_air_strong_threshold 0.0 -pc_air_a_drop 0.0 -pc_air_r_drop 0.0 \
	 -pc_air_inverse_type jacobi -ksp_type richardson -ksp_norm_type unpreconditioned \
	 -pc_air_max_levels 10 -pc_air_coarsest_poly_order 60 \
	 -pc_air_coarsest_matrix_free_polys -pc_air_coarsest_inverse_type newton -pc_air_max_luby_steps 3 -ksp_max_it 1
#
	@echo ""
	@echo "Test AIRG with GMRES polynomials for hyperbolic streaming problem with coefficients calculated on subcomms serial"
	$(MPIEXEC) -n 2 ./ex12f -f data/mat_stream_2364 -pc_air_subcomm -pc_air_inverse_type arnoldi -pc_air_coarsest_subcomm \
	 -pc_air_coarsest_inverse_type arnoldi -ksp_max_it 5 -pc_air_a_drop 1e-3
	@echo "Test AIRG with GMRES polynomials for hyperbolic streaming problem with coefficients calculated on subcomms parallel"
	$(MPIEXEC) -n 4 ./ex12f -f data/mat_stream_2364 -pc_air_subcomm -pc_air_inverse_type arnoldi -pc_air_coarsest_subcomm \
	 -pc_air_coarsest_inverse_type arnoldi -ksp_max_it 5 -pc_air_a_drop 1e-3	 
# 
	@echo "Test PMISR DDC CF splitting in C in parallel"
	$(MPIEXEC) -n 2 ./ex6_cf_splitting  -f data/mat_stream_2364
#
	@echo ""
	@echo "Test AIRG with GMRES polynomials in indefinite problem with zero diagonals in parallel"
	$(MPIEXEC) -n 2 ./ex6 -f data/e05r0100_petsc -b_in_f 0 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -ksp_max_it 26		

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_tests_no_load_serial:
#
	@echo ""
	@echo "Test AIRG with random diagonally dominant matrix"
	./matrandom -ksp_max_it 5
	@echo ""
	@echo "Test AIRG with parameter resets on random diagonally dominant matrix"
	./matrandom_check_reset -ksp_max_it 7
	@echo "Test PCPFLAREINV with parameter resets on random diagonally dominant matrix"
	./matrandom_check_reset -ksp_max_it 12 -check_airg 0
	@echo ""
	@echo "Test AIRG with SUPG CG FEM in 2D"
	./adv_diff_cg_supg -adv_diff_petscspace_degree 1 -dm_plex_simplex 0 -dm_refine 1 -pc_type air -pc_air_a_drop 1e-3 -pc_air_inverse_type power \
	 -ksp_type richardson -ksp_norm_type unpreconditioned -ksp_max_it 4
	@echo "Test AIRG with SUPG CG FEM in 3D"
	./adv_diff_cg_supg -adv_diff_petscspace_degree 1 -dm_plex_simplex 0 -dm_plex_dim 3 -dm_plex_box_faces 5,5,5 -pc_type air -pc_air_a_drop 1e-3 -pc_air_inverse_type power \
	 -ksp_type richardson -ksp_norm_type unpreconditioned -ksp_max_it 4
#
	@echo ""
	@echo "Test AIRG with GMRES polynomials for 2D finite difference stencil"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -ksp_max_it 5
	@echo "Test AIRG with GMRES polynomials Arnoldi basis for 2D finite difference stencil"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -pc_air_inverse_type arnoldi \
	 -pc_air_coarsest_inverse_type arnoldi -ksp_max_it 5 -pc_air_a_drop 1e-3
# 
	@echo ""
	@echo "Test single level GMRES polynomial preconditioning for 2D finite difference stencil"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type pflareinv -pc_pflareinv_type power -ksp_max_it 8
#
	@echo ""
	@echo "Test AIRG with Neumann polynomials for 2D finite difference stencil"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -pc_air_inverse_type neumann -ksp_max_it 5 -pc_air_a_drop 1e-3
	@echo "Test AIRG with Neumann polynomials for 2D finite difference stencil, matrix-free smoothing"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -pc_air_inverse_type neumann \
	 -pc_air_matrix_free_polys -ksp_max_it 5 -pc_air_a_drop 1e-3
#
	@echo ""
	@echo "Test AIRG with SAIs for 2D finite difference stencil"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -pc_air_inverse_type sai -ksp_max_it 5 -pc_air_a_drop 1e-3
# 
	@echo ""
	@echo "Test AIRG with ISAIs for 2D finite difference stencil"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -pc_air_inverse_type isai -ksp_max_it 5 -pc_air_a_drop 1e-3
# 
	@echo ""
	@echo "Test AIRG with Weighted Jacobi for 2D finite difference stencil"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -pc_air_inverse_type wjacobi -ksp_max_it 8 -pc_air_a_drop 1e-3
# 
	@echo ""
	@echo "Test AIRG with Unweighted Jacobi for 2D finite difference stencil"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -pc_air_inverse_type jacobi -ksp_max_it 5 -pc_air_a_drop 1e-3
# 
	@echo ""
	@echo "Test AIRG as an exact solver for 2D finite difference stencil"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air \
	 -pc_air_strong_threshold 0.0 -pc_air_a_drop 0.0 -pc_air_r_drop 0.0 -pc_air_inverse_type jacobi \
	 -mg_coarse_ksp_type richardson -mg_coarse_ksp_max_it 10 -ksp_type richardson -ksp_norm_type unpreconditioned -ksp_max_it 1
# 
	@echo ""
	@echo "Test AIRG with GMRES polynomials with PC reused with no sparsity change"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -ksp_max_it 10
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -regen -pc_air_reuse_sparsity -ksp_max_it 3	
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change and not classical prolong"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -regen -pc_air_reuse_sparsity -ksp_max_it 3	-pc_air_one_point_classical_prolong 0
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change and strong R threshold"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -regen -pc_air_reuse_sparsity -pc_air_strong_r_threshold 0.01 -ksp_max_it 3
	@echo "Test lAIR with GMRES polynomials with PC regenerated with no sparsity change"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -regen -pc_air_z_type lair -pc_air_reuse_sparsity -ksp_max_it 3
	@echo "Test lAIR SAI with GMRES polynomials with PC regenerated with no sparsity change"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -regen -pc_air_z_type lair_sai -pc_air_reuse_sparsity -ksp_max_it 3
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change with constrain w"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_constrain_w
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change and 0 strong threshold"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_strong_threshold 0.0
# 
	@echo ""
	@echo "Test AIRG with 0th order GMRES polynomials with PC reused with no sparsity change"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -ksp_max_it 10 -pc_air_poly_order 0
	@echo "Test AIRG with 0th order GMRES polynomials with PC regenerated with no sparsity change"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_poly_order 0
# 
	@echo ""
	@echo "Test AIRG with 1st order GMRES polynomials with PC reused with no sparsity change"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -ksp_max_it 10 -pc_air_poly_order 1
	@echo "Test AIRG with 1st order GMRES polynomials with PC regenerated with no sparsity change"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_poly_order 1	
# 
	@echo ""
	@echo "Test AIRG with 2nd order GMRES polynomials with PC reused with no sparsity change"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -ksp_max_it 10 -pc_air_poly_order 2 -pc_air_inverse_sparsity_order 2
	@echo "Test AIRG with 2nd order GMRES polynomials with PC regenerated with no sparsity change"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_poly_order 2 -pc_air_inverse_sparsity_order 2
	@echo "Test AIRG with 2nd order fixed sparsity GMRES polynomials with PC regenerated with no sparsity change"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_inverse_sparsity_order 2	
# 
	@echo ""
	@echo "Test AIRG with GMRES polynomials with PC reused with no sparsity change with 0th order fixed sparsity"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -ksp_max_it 10 -pc_air_inverse_sparsity_order 0
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change with 0th order fixed sparsity"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_inverse_sparsity_order 0
# 
	@echo ""
	@echo "Test AIRG Newton with 0th order GMRES polynomials with PC reused with no sparsity change"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type newton -ksp_max_it 10 -pc_air_poly_order 0
	@echo "Test AIRG Newton with 0th order GMRES polynomials with PC regenerated with no sparsity change"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type newton -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_poly_order 0	
# 
	@echo ""
	@echo "Test AIRG Newton with 1st order GMRES polynomials with PC reused with no sparsity change"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type newton -ksp_max_it 10 -pc_air_poly_order 1
	@echo "Test AIRG Newton with 1st order GMRES polynomials with PC regenerated with no sparsity change"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type newton -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_poly_order 1		
# 
	@echo ""
	@echo "Test AIRG Newton with 2nd order GMRES polynomials with PC reused with no sparsity change"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type newton -ksp_max_it 10 -pc_air_poly_order 2 -pc_air_inverse_sparsity_order 2
	@echo "Test AIRG Newton with 2nd order GMRES polynomials with PC regenerated with no sparsity change"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type newton -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_poly_order 2 -pc_air_inverse_sparsity_order 2
	@echo "Test AIRG Newton with 2nd order fixed sparsity GMRES polynomials with PC regenerated with no sparsity change"
	./ex6f -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type newton -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_inverse_sparsity_order 2		
# 
	@echo ""
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change and polynomial coeffs stored"
	./ex6f_getcoeffs -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -ksp_max_it 3
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change and polynomial coeffs stored and lumping"
	./ex6f_getcoeffs -m 10 -n 10 -pc_air_a_drop 1e-3 -pc_air_inverse_type power -pc_air_a_lump -ksp_max_it 3
# 
	@echo ""
	@echo "Test single level GMRES polynomials with PC reused with no sparsity change"
	./ex6f -m 10 -n 10 -pc_type pflareinv -pc_pflareinv_type power -ksp_max_it 8
	@echo "Test single level GMRES polynomials with PC regenerated with no sparsity change"
	./ex6f -m 10 -n 10 -pc_type pflareinv -pc_pflareinv_type power -regen -ksp_max_it 8
# 
	@echo ""
	@echo "Test solving isotropic diffusion with fast coarsening and near-nullspace"
	./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 50 -da_grid_y 50 -pc_type air -pc_air_z_type lair \
	 -pc_air_smooth_type ffc -pc_air_symmetric -pc_air_constrain_z \
	 -pc_air_cf_splitting_type agg -pc_air_a_drop 1e-5 -pc_air_a_lump \
	 -pc_air_r_drop 0 -ksp_rtol 1e-10 -ksp_pc_side right -ksp_max_it 14 \
	 -pc_air_inverse_type power
# 
	@echo ""
	@echo "Test AIRG on steady 1D advection"
	./adv_1d -n 1000 -ksp_rtol 1e-10 -ksp_atol 1e-50 -ksp_pc_side right \
	 -pc_air_coarsest_inverse_type newton -pc_air_coarsest_poly_order 10 -pc_air_coarsest_matrix_free_polys -ksp_max_it 2 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
# 
	@echo ""
	@echo "Test lAIR on steady 2D structured advection"
	./adv_diff_2d -da_grid_x 50 -da_grid_y 50 -pc_type air -ksp_pc_side right \
	 -ksp_rtol 1e-10 -ksp_atol 1e-50 -pc_air_a_lump -pc_air_a_drop 1e-4 -pc_air_smooth_type ffc \
	 -pc_air_z_type lair -pc_air_inverse_type wjacobi -ksp_max_it 10
# 
	@echo ""
	@echo "Test AIRG on steady 2D structured advection with 0th order sparsity GMRES poly C smooth and faster coarsening"
	./adv_diff_2d -da_grid_x 50 -da_grid_y 50 -pc_type air -ksp_pc_side right \
	 -ksp_rtol 1e-10 -ksp_atol 1e-50 -pc_air_a_lump -pc_air_a_drop 1e-4 -pc_air_smooth_type ffc \
	 -pc_air_c_inverse_sparsity_order 0 -pc_air_strong_threshold 0.99 -pc_air_ddc_its 0 -ksp_max_it 7 \
	 -pc_air_inverse_type power
# 
	@echo ""
	@echo "Test AIRG on steady 2D structured advection with 0th order sparsity GMRES poly C smooth and multiple DDC its"
	./adv_diff_2d -da_grid_x 50 -da_grid_y 50 -pc_type air -ksp_pc_side right \
	 -ksp_rtol 1e-10 -ksp_atol 1e-50 -pc_air_a_lump -pc_air_a_drop 1e-4 -pc_air_smooth_type ffc \
	 -pc_air_c_inverse_sparsity_order 0 -pc_air_strong_threshold 0.99 -pc_air_ddc_its 2 -pc_air_ddc_fraction 0.02 -ksp_max_it 7 \
	 -pc_air_inverse_type power	 
# 
	@echo ""
	@echo "Test AIRG on steady 2D structured advection with 0th order sparsity GMRES poly C smooth and max DD ratio"
	./adv_diff_2d -da_grid_x 100 -da_grid_y 100 -pc_type air -ksp_pc_side right \
	 -ksp_rtol 1e-10 -ksp_atol 1e-50 -pc_air_a_lump -pc_air_a_drop 1e-4 \
    -pc_air_strong_threshold 0.99 -pc_air_max_dd_ratio 0.50001 -pc_air_ddc_fraction 0.01 -ksp_max_it 7 \
	 -pc_air_inverse_type power		 
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
	 -pc_air_auto_truncate_start_level 1 -pc_air_auto_truncate_tol 1e-2 -ksp_max_it 3 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
#
	@echo ""
	@echo "Test improving Z" 
	./adv_diff_2d -da_grid_x 50 -da_grid_y 50 -pc_type air -pc_air_improve_z_its 3 -ksp_max_it 3 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
	@echo "Test improving Z with 1 iteration" 
	./adv_diff_2d -da_grid_x 50 -da_grid_y 50 -pc_type air -pc_air_improve_z_its 1 -ksp_max_it 3	-pc_air_a_drop 1e-3 -pc_air_inverse_type power
	@echo "Test improving Z with PC regenerated with no sparsity change"
	./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -pc_air_improve_z_its 3 -ksp_max_it 3 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
	@echo "Test improving Z with PC regenerated with no sparsity change with 1 iteration"
	./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -pc_air_improve_z_its 1 -ksp_max_it 3 -pc_air_a_drop 1e-3 -pc_air_inverse_type power			
#
	@echo ""
	@echo "Test improving W" 
	./adv_diff_2d -da_grid_x 50 -da_grid_y 50 -pc_type air -pc_air_one_point_classical_prolong 0 \
	 -pc_air_improve_w_its 3 -ksp_max_it 4 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
	@echo "Test improving W with 1 iteration" 
	./adv_diff_2d -da_grid_x 50 -da_grid_y 50 -pc_type air -pc_air_one_point_classical_prolong 0 \
	 -pc_air_improve_w_its 1 -ksp_max_it 4	-pc_air_a_drop 1e-3 -pc_air_inverse_type power
	@echo "Test improving Z with PC regenerated with no sparsity change"
	./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -pc_air_one_point_classical_prolong 0 \
	 -pc_air_improve_w_its 3 -ksp_max_it 3 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
	@echo "Test improving Z with PC regenerated with no sparsity change with 1 iteration"
	./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -pc_air_one_point_classical_prolong 0 \
	 -pc_air_improve_w_its 1 -ksp_max_it 3 -pc_air_a_drop 1e-3 -pc_air_inverse_type power	 	
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
	 -vec_type kokkos -mat_type aijkokkos -pc_air_a_drop 1e-3 -pc_air_inverse_type power

endif

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_tests_no_load_parallel:
#
	@echo ""
	@echo "Test AIRG with SUPG CG FEM in 2D in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_cg_supg -adv_diff_petscspace_degree 1 -dm_plex_simplex 0 -dm_refine 1 -pc_type air \
	 -ksp_type richardson -ksp_norm_type unpreconditioned -ksp_max_it 4 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
	@echo "Test AIRG with SUPG CG FEM in 3D in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_cg_supg -adv_diff_petscspace_degree 1 -dm_plex_simplex 0 -dm_plex_dim 3 -dm_plex_box_faces 5,5,5 \
	 -pc_type air -ksp_type richardson -ksp_norm_type unpreconditioned -ksp_max_it 4 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
#
	@echo ""
	@echo "Test AIRG with GMRES polynomials for 2D finite difference stencil in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -ksp_max_it 5 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
	@echo "Test AIRG with GMRES polynomials Arnoldi basis for 2D finite difference stencil in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -pc_air_inverse_type arnoldi -pc_air_coarsest_inverse_type arnoldi -ksp_max_it 5
	@echo ""
	@echo "Test lAIR with GMRES polynomial smoothing for 2D finite difference stencil in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air -pc_air_z_type lair -ksp_max_it 4 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
# 
	@echo ""
	@echo "Test single level GMRES polynomial preconditioning for 2D finite difference stencil in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type pflareinv -pc_pflareinv_type power -ksp_max_it 8
#
	@echo ""
	@echo "Test AIRG with Neumann polynomials for 2D finite difference stencil in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air  -pc_air_inverse_type neumann -ksp_max_it 5 -pc_air_a_drop 1e-3
	@echo "Test AIRG with Neumann polynomials for 2D finite difference stencil, matrix-free smoothing in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air  -pc_air_inverse_type neumann \
	 -pc_air_matrix_free_polys -ksp_max_it 5 -pc_air_a_drop 1e-3
#
	@echo ""
	@echo "Test AIRG with SAI for 2D finite difference stencil in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air  -pc_air_inverse_type sai -ksp_max_it 5 -pc_air_a_drop 1e-3
# 
	@echo ""
	@echo "Test AIRG with ISAIs for 2D finite difference stencil in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air  -pc_air_inverse_type isai -ksp_max_it 5 -pc_air_a_drop 1e-3
# 
	@echo ""
	@echo "Test AIRG with Weighted Jacobi for 2D finite difference stencil in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air  -pc_air_inverse_type wjacobi -ksp_max_it 8 -pc_air_a_drop 1e-3
# 
	@echo ""
	@echo "Test AIRG with Unweighted Jacobi for 2D finite difference stencil in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air  -pc_air_inverse_type jacobi -ksp_max_it 5 -pc_air_a_drop 1e-3
# 
	@echo ""
	@echo "Test AIRG as an exact solver for 2D finite difference stencil in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 8 -da_grid_y 8 -pc_type air \
	 -pc_air_strong_threshold 0.0 -pc_air_a_drop 0.0 -pc_air_r_drop 0.0 -pc_air_inverse_type jacobi \
	 -mg_coarse_ksp_type richardson -mg_coarse_ksp_max_it 10 -ksp_type richardson -ksp_norm_type unpreconditioned -ksp_max_it 1
# 
	@echo ""
	@echo "Test AIRG on steady 2D structured advection with 0th order sparsity GMRES poly C smooth and max DD ratio in parallel"
	$(MPIEXEC) -n 2 ./adv_diff_2d -da_grid_x 100 -da_grid_y 100 -pc_type air -ksp_pc_side right \
	 -ksp_rtol 1e-10 -ksp_atol 1e-50 -pc_air_a_lump -pc_air_a_drop 1e-4 \
	 -pc_air_strong_threshold 0.99 -pc_air_max_dd_ratio 0.50001 -pc_air_ddc_fraction 0.01 -ksp_max_it 7 \
	 -pc_air_inverse_type power		 
# 
	@echo ""
	@echo "Test auto truncation with matrix-free Newton polynomials as coarse grid solver in parallel" 
	$(MPIEXEC) -n 2 ./adv_diff_2d -da_grid_x 10 -da_grid_y 10 -ksp_type richardson -pc_type air \
	 -pc_air_coarsest_inverse_type newton -pc_air_coarsest_poly_order 10 -pc_air_coarsest_matrix_free_polys \
	 -pc_air_auto_truncate_start_level 1 -pc_air_auto_truncate_tol 1e-2 -ksp_max_it 3 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type power	 
# 
	@echo ""
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change and not classical prolong in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_one_point_classical_prolong 0	\
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change with constrain w in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_constrain_w	\
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change and 0 strong threshold in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_strong_threshold 0.0	\
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
# 
	@echo ""
	@echo "Test AIRG with 0th order GMRES polynomials with PC reused with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -ksp_max_it 10 -pc_air_poly_order 0 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
	@echo "Test AIRG with 0th order GMRES polynomials with PC regenerated with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_poly_order 0 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
# 
	@echo ""
	@echo "Test AIRG with 1st order GMRES polynomials with PC reused with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -ksp_max_it 10 -pc_air_poly_order 1 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
	@echo "Test AIRG with 1st order GMRES polynomials with PC regenerated with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_poly_order 1 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
# 
	@echo ""
	@echo "Test AIRG with 2nd order GMRES polynomials with PC reused with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -ksp_max_it 10 -pc_air_poly_order 2 -pc_air_inverse_sparsity_order 2 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
	@echo "Test AIRG with 2nd order GMRES polynomials with PC regenerated with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_poly_order 2 -pc_air_inverse_sparsity_order 2 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
	@echo "Test AIRG with 2nd order fixed sparsity GMRES polynomials with PC regenerated with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_inverse_sparsity_order 2 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
# 
	@echo ""
	@echo "Test AIRG with GMRES polynomials with PC reused with no sparsity change in parallel with 0th order fixed sparsity"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -ksp_max_it 10 -pc_air_inverse_sparsity_order 0 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type power	
	@echo "Test AIRG with GMRES polynomials with PC regenerated with no sparsity change in parallel with 0th order fixed sparsity"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_inverse_sparsity_order 0 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
# 
	@echo ""
	@echo "Test AIRG Newton with 0th order GMRES polynomials with PC reused with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -ksp_max_it 10 -pc_air_poly_order 0 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type newton
	@echo "Test AIRG Newton with 0th order GMRES polynomials with PC regenerated with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_poly_order 0 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type newton
# 
	@echo ""
	@echo "Test AIRG Newton with 1st order GMRES polynomials with PC reused with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -ksp_max_it 10 -pc_air_poly_order 1 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type newton
	@echo "Test AIRG Newton with 1st order GMRES polynomials with PC regenerated with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_poly_order 1 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type newton	 
# 
	@echo ""
	@echo "Test AIRG Newton with 2nd order GMRES polynomials with PC reused with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -ksp_max_it 10 -pc_air_poly_order 2 -pc_air_inverse_sparsity_order 2 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type newton
	@echo "Test AIRG Newton with 2nd order GMRES polynomials with PC regenerated with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_poly_order 2 -pc_air_inverse_sparsity_order 2 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type newton
	@echo "Test AIRG Newton with 2nd order fixed sparsity GMRES polynomials with PC regenerated with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -ksp_max_it 3 -pc_air_inverse_sparsity_order 2 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type newton	 	 
# 
	@echo ""
	@echo "Test solving isotropic diffusion with fast coarsening and near-nullspace in parallel"
	$(MPIEXEC) -n 4 ./adv_diff_2d -u 0 -v 0 -alpha 1.0 -da_grid_x 50 -da_grid_y 50 -pc_type air -pc_air_z_type lair \
	 -pc_air_smooth_type ffc -pc_air_symmetric -pc_air_constrain_z \
	 -pc_air_cf_splitting_type agg -pc_air_a_drop 1e-5 -pc_air_a_lump \
	 -pc_air_r_drop 0 -ksp_rtol 1e-10 -ksp_pc_side right -ksp_max_it 14 \
	 -pc_air_inverse_type power
# 
	@echo ""
	@echo "Test AIRG on steady 1D advection in parallel"
	$(MPIEXEC) -n 2 ./adv_1d -n 1000 -ksp_rtol 1e-10 -ksp_atol 1e-50 -ksp_pc_side right \
	 -pc_air_coarsest_inverse_type newton -pc_air_coarsest_poly_order 10 -pc_air_coarsest_matrix_free_polys -ksp_max_it 10 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
#
	@echo ""
	@echo "Test improving Z in parallel" 
	$(MPIEXEC) -n 2 ./adv_diff_2d -da_grid_x 50 -da_grid_y 50 -pc_type air -pc_air_improve_z_its 3 -ksp_max_it 3 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
	@echo "Test improving Z with PC regenerated with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -pc_air_improve_z_its 3 -ksp_max_it 3 \
	 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
#
	@echo ""
	@echo "Test improving W in parallel" 
	$(MPIEXEC) -n 2 ./adv_diff_2d -da_grid_x 50 -da_grid_y 50 -pc_type air -pc_air_one_point_classical_prolong 0 \
	 -pc_air_improve_w_its 3 -ksp_max_it 4 -pc_air_a_drop 1e-3 -pc_air_inverse_type power
	@echo "Test improving W with PC regenerated with no sparsity change in parallel"
	$(MPIEXEC) -n 2 ./ex6f -m 10 -n 10 -regen -pc_air_reuse_sparsity -pc_air_one_point_classical_prolong 0 \
	 -pc_air_improve_w_its 3 -ksp_max_it 3	-pc_air_a_drop 1e-3 -pc_air_inverse_type power

#	 
# ~~~~~~~~~~~~~~~~~~~~~~~	 
# Include kokkos examples	 
# ~~~~~~~~~~~~~~~~~~~~~~~
ifeq ($(PETSC_HAVE_KOKKOS),1)
# 
	@echo ""
	@echo "Test AIRG on steady 1D advection in parallel KOKKOS"
	$(MPIEXEC) -n 2 ./adv_1dk -n 1000 -ksp_rtol 1e-10 -ksp_atol 1e-50 -ksp_pc_side right \
	 -pc_air_coarsest_inverse_type newton -pc_air_coarsest_poly_order 10 -pc_air_coarsest_matrix_free_polys -ksp_max_it 10 \
	 -vec_type kokkos -mat_type aijkokkos -pc_air_a_drop 1e-3 -pc_air_inverse_type power	

endif

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_tests_medium_serial:

#
	@echo ""
	@for size in $$(seq 100 20 200); do \
		echo "--- Testing size = $$size x $$size ---"; \
		./adv_diff_2d -da_grid_x $$size -da_grid_y $$size -pc_type air -ksp_pc_side right \
	    -ksp_rtol 1e-10 -ksp_atol 1e-50 -pc_air_a_lump -pc_air_a_drop 1e-5 -pc_air_strong_threshold 0.99 -ksp_max_it 6; \
	done
#
	@echo ""
	@echo "--- Running scaling study on adv_diff_2d ---"
	@for size in 100 200 400 800; do \
		echo "--- Testing size = $$size x $$size ---"; \
		./adv_diff_2d -da_grid_x $$size -da_grid_y $$size -pc_type air -ksp_pc_side right \
	    -ksp_rtol 1e-10 -ksp_atol 1e-50 -pc_air_a_lump -pc_air_a_drop 1e-5 -pc_air_strong_threshold 0.99 -ksp_max_it 6; \
	done

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_tests_medium_parallel:

#
	@echo ""
	@for size in $$(seq 100 20 200); do \
		echo "--- Testing size = $$size x $$size ---"; \
		$(MPIEXEC) -n 2 ./adv_diff_2d -da_grid_x $$size -da_grid_y $$size -pc_type air -ksp_pc_side right \
	    -ksp_rtol 1e-10 -ksp_atol 1e-50 -pc_air_a_lump -pc_air_a_drop 1e-5 -pc_air_strong_threshold 0.99 -ksp_max_it 6; \
	done
#
	@echo ""
	@echo "--- Running scaling study on adv_diff_2d in parallel ---"
	@for size in 100 200 400 800; do \
		echo "--- Testing size = $$size x $$size ---"; \
		$(MPIEXEC) -n 2 ./adv_diff_2d -da_grid_x $$size -da_grid_y $$size -pc_type air -ksp_pc_side right \
	    -ksp_rtol 1e-10 -ksp_atol 1e-50 -pc_air_a_lump -pc_air_a_drop 1e-5 -pc_air_strong_threshold 0.99 -ksp_max_it 6; \
	done	

# ~~~~~~~~~~~
# ~~~~~~~~~~~
run_check:

#
	@echo ""
	@echo "Serial Fortran check"
	@./matrandom -ksp_max_it 5 -on_error_abort  && echo "OK" || (echo "FAIL" && exit 1)	
	@echo "Serial check"
	@./adv_diff_2d -da_grid_x 25 -da_grid_y 25 -pc_type air -ksp_max_it 5 -on_error_abort  && echo "OK" || (echo "FAIL" && exit 1)
	@echo "Parallel check"
	@$(MPIEXEC) -n 2 ./adv_diff_2d -da_grid_x 25 -da_grid_y 25 -pc_type air -ksp_max_it 5 -on_error_abort  && echo "OK" || (echo "FAIL" && exit 1)
ifeq ($(PETSC_HAVE_KOKKOS),1)
# 
	@echo "Serial KOKKOS check"
	@./adv_diff_2d -da_grid_x 25 -da_grid_y 25 -pc_type air -ksp_max_it 5 -dm_vec_type kokkos -dm_mat_type aijkokkos -on_error_abort  && echo "OK" || (echo "FAIL" && exit 1)
	@echo "Parallel KOKKOS check"
	@$(MPIEXEC) -n 2 ./adv_diff_2d -da_grid_x 25 -da_grid_y 25 -pc_type air -ksp_max_it 5 -dm_vec_type kokkos -dm_mat_type aijkokkos -on_error_abort  && echo "OK" || (echo "FAIL" && exit 1)
else	
	@echo "KOKKOS check skipped (PETSc has not been configured with KOKKOS)"	

endif	