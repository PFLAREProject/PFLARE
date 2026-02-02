/*   DMDA/KSP solving a system of linear equations.
     Steady advection-diffusion equation in 2D or 3D with finite difference, advection is upwinded
     Default is 2D. Use -dim 3 for 3D.

     ./adv_diff_fd
             : pure advection with theta = pi/4, dimensionless
               BCs left and bottom dirichlet (u=0), top and right outflow
               Same equation as advection_2d in PyAMG, except we don't eliminate the dirichlet dofs
     ./adv_diff_fd -adv_nondim 0
             : pure advection with theta = pi/4, scaled by Hx * Hy
               BCs left and bottom dirichlet (u=0), top and right outflow
     ./adv_diff_fd -u 0 -v 0 -alpha 1.0
             : pure diffusion scaled by Hx * Hy
               BCs dirichlet on all sides
     ./adv_diff_fd -alpha 1.0
             : advection-diffusion scaled by Hx * Hy with theta=pi/4
               BCs dirichlet on all sides
     ./adv_diff_fd -bottom_only_inflow_one
             : pure advection with theta = pi/4, dimensionless
               BC bottom face dirichlet inflow u=1, left face dirichlet u=0, top and right outflow
     ./adv_diff_fd -dim 3
             : pure advection in 3D with default velocity (1,1,1) normalised
               BCs west (x=0), south (y=0), bottom (z=0) dirichlet u=0; other faces outflow
     ./adv_diff_fd -dim 3 -alpha 1.0
             : advection-diffusion in 3D scaled by Hx*Hy*Hz
               BCs dirichlet on all sides

     Can control dimension with -dim (2 default)
     Can control grid size with -da_grid_x, -da_grid_y, -da_grid_z (default 11)
     Can control the direction of advection with -theta (pi/4 default, 2D only),
       or by giving the -u and -v and -w directly
     In 3D the default velocity is (1,1,1) normalised; -theta only affects the 2D x-y plane
     Can optionally left scale the matrix by the inverse diagonal before solving (-diag_scale)
     Can specify inflow of 1 on the bottom face with -bottom_only_inflow_one (default false),
       in 2D: bottom (j=0) face; in 3D: bottom (k=0, z=0) face
       all other inflow faces are set to u=0
     Can change default velocity from straight line to curved with -curved_velocity (default false)
       curved velocity field: u(x,y) = y, v(x,y) = 1-x, w=0 (rotating, always >= 0 on [0,1]^2)
     Can normalise velocity with -unit_velocity (default true) so that we have a unit velocity.
       If any of u,v,w are set explicitly then unit velocity will be disabled
     Can control domain size in each direction with -L_x, -L_y, -L_z (default 1.0)
     Can write the solution to a VTK structured grid file with -vec_view vtk:solution.vts
       (note: DMDA is a structured mesh so PETSc produces .vts, not .vtu)

*/

static char help[] = "Solves steady advection-diffusion with finite-difference on a structured grid in 2D or 3D.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscvec.h>

#include "pflare.h"

// Helper function to compute velocity at a point.
// dim: spatial dimension (2 or 3)
// u_const,v_const,w_const: constant velocity components (used when curved_velocity is false)
// x[3]: node coordinates (x, y, z)
// curved_velocity: if true, use spatially-varying field u(x,y)=y, v(x,y)=1-x, w=0
// unit_velocity: if true, normalise the velocity to unit magnitude
static inline void GetVelocity(PetscInt dim, PetscReal u_const, PetscReal v_const, PetscReal w_const,
                               const PetscReal x[], PetscBool curved_velocity, PetscBool unit_velocity,
                               PetscReal vel[])
{
  if (curved_velocity) {
    // Spatially-varying velocity field: top-left quadrant of a rotating circle (center at 1,0)
    // u(x,y) = y, v(x,y) = 1-x, w = 0
    vel[0] = x[1];        // u(x,y) = y
    vel[1] = 1.0 - x[0]; // v(x,y) = 1-x
    vel[2] = 0.0;         // w = 0 (curved field defined in 2D plane only)
  } else {
    // Constant velocity field
    vel[0] = u_const;
    vel[1] = v_const;
    vel[2] = w_const;
  }

  // Normalise velocity magnitude if unit_velocity flag is set
  if (unit_velocity) {
    PetscReal mag = 0.0;
    for (int d = 0; d < dim; ++d) mag += vel[d]*vel[d];
    mag = PetscSqrtReal(mag);
    if (mag > 1e-12) { for (int d = 0; d < dim; ++d) vel[d] /= mag; }
  }
}

extern PetscErrorCode ComputeMat(DM,Mat,PetscInt,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscBool,PetscBool,PetscBool);

int main(int argc,char **argv)
{
  KSP            ksp;
  PC             pc;
  DM             da;
  PetscInt       dim;
  PetscScalar    theta, alpha, u, v, w, u_test, v_test, w_test;
  PetscScalar    L_x, L_y, L_z, L_x_test, L_y_test, L_z_test;
  PetscBool      option_found_u, option_found_v, option_found_w;
  PetscBool      adv_nondim, check_nondim, diag_scale, bottom_only_inflow_one, curved_velocity, unit_velocity;
  Vec x, b, diag_vec;
  Mat A, A_temp;
  KSPConvergedReason reason;
  PetscLogStage setup, gpu_copy;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //       Let's use PFLARE
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  

  // Register the pflare types
  PCRegister_PFLARE();

  PetscCall(PetscLogStageRegister("Setup", &setup));
  PetscCall(PetscLogStageRegister("GPU copy stage - triggered by a prelim KSPSolve", &gpu_copy));

  // Dimension - default to 2D
  dim = 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));
  PetscCheck(dim == 2 || dim == 3, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "dim must be 2 or 3");

  // Dimensions of box - default to [0, 1]^dim
  L_x = 1.0; L_y = 1.0; L_z = 1.0;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-L_x", &L_x_test, &option_found_u));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-L_y", &L_y_test, &option_found_v));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-L_z", &L_z_test, &option_found_w));

  if (option_found_u) PetscCheck(L_x_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "L_x must be positive");
  if (option_found_v) PetscCheck(L_y_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "L_y must be positive");
  if (option_found_w) PetscCheck(L_z_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "L_z must be positive");

  if (option_found_u) L_x = L_x_test;
  if (option_found_v) L_y = L_y_test;
  if (option_found_w) L_z = L_z_test;

  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  if (dim == 2) {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                           11, 11, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  } else {
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                           11, 11, 11, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, NULL, &da));
  }
  PetscCall(DMSetFromOptions(da));
  // This stops the zero entries in the stencil from being added to the matrix
  // It still allocates the memory for a 5/7 point stencil but the sparsity doesn't include the zero entries
  // We do this instead of calling MatFilter as there is no Kokkos implementation so its very slow
  PetscCall(DMSetMatrixPreallocateOnly(da,PETSC_TRUE));
  PetscCall(DMSetUp(da));
  if (dim == 2) {
    PetscCall(DMDASetUniformCoordinates(da, 0.0, L_x, 0.0, L_y, 0.0, 0.0));
  } else {
    PetscCall(DMDASetUniformCoordinates(da, 0.0, L_x, 0.0, L_y, 0.0, L_z));
  }
  PetscCall(KSPSetDM(ksp,(DM)da));
  // We generate the matrix ourselves
  PetscCall(KSPSetDMActive(ksp, KSP_DMACTIVE_ALL, PETSC_FALSE));

  // Create empty matrix and vectors
  PetscCall(DMCreateMatrix(da, &A));
  PetscCall(MatSetOption(A,MAT_IGNORE_ZERO_ENTRIES ,PETSC_TRUE));
  PetscCall(DMCreateGlobalVector(da, &x));
  PetscCall(PetscObjectSetName((PetscObject)x, "adv_diff"));
  PetscCall(DMCreateGlobalVector(da, &b));

  // Zero rhs
  PetscCall(VecSet(b, 0.0));

  // ~~~~~~~~~~~~~~
  // Get command line options
  // ~~~~~~~~~~~~~~

  PetscBool second_solve= PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-second_solve", &second_solve, NULL));

  // Advection velocities
  // Default in 2D/3D: (1,1,1), normalised when unit_velocity is true
  // If -theta is provided, set (u,v,w) = (cos(theta), sin(theta), 0)
  PetscReal pi = 4*atan(1.0);
  theta = pi/4.0;
  PetscBool theta_found;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-theta", &theta, &theta_found));
  u = 1.0;
  v = 1.0;
  w = 1.0;
  if (theta_found) {
    PetscCheck(theta <= pi/2.0 && theta >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "Theta must be between 0 and pi/2");
    u = cos(theta);
    v = sin(theta);
    w = 0.0;
  }

  // Or the user can pass in the individual advection velocities
  // This will override theta and unit velocity options if they are set
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-u", &u_test, &option_found_u));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-v", &v_test, &option_found_v));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-w", &w_test, &option_found_w));

  if (option_found_u) PetscCheck(u_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "u must be positive");
  if (option_found_v) PetscCheck(v_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "v must be positive");
  if (option_found_w) PetscCheck(w_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "w must be positive");

  if (option_found_u) u = u_test;
  if (option_found_v) v = v_test;
  if (option_found_w) w = w_test;

  // Curved velocity field option (u(x,y) = y, v(x,y) = 1-x)
  // Default is straight line (false)
  curved_velocity = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-curved_velocity", &curved_velocity, NULL));

  // Unit velocity normalization option - default to unit velocity
  unit_velocity = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-unit_velocity", &unit_velocity, NULL));

  // Don't normalise if user has explicitly set a velocity
  if (option_found_u || option_found_v || option_found_w) {
    unit_velocity = PETSC_FALSE;
  }

  // Diffusion coefficient
  // Default alpha is 0 - pure advection
  alpha = 0.0;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-alpha", &alpha, NULL));

  // If we just have advection, rather than scaling by Hx * Hy, we can just have 
  // a dimensionless advection problem - this is enabled by default
  // If we have any diffusion this is turned off by default
  adv_nondim = PETSC_TRUE;
  if (alpha != 0.0)
  {
   adv_nondim = PETSC_FALSE;
  }  
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-adv_nondim", &adv_nondim, NULL));
  // We can only nondimensionalise the advection if we don't have any diffusion
  check_nondim = PETSC_TRUE;
  if (alpha != 0.0)
  {
   if (adv_nondim)
   {
      check_nondim = PETSC_FALSE;
   }
  }

  PetscCheck(check_nondim, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "Non-dimensional advection only applies without diffusion");

  // Do we diagonally scale our matrix before solving
  // Defaults to false
  diag_scale = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-diag_scale", &diag_scale, NULL));

  // Set only the bottom face inflow to u=1, all other inflow faces to u=0
  // In 2D: bottom face is j==0; in 3D: bottom face is k==0 (z=0)
  // Defaults to false (all inflow faces are u=0)
  bottom_only_inflow_one = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-bottom_only_inflow_one", &bottom_only_inflow_one, NULL));

  // If bottom_only_inflow_one is set, set the RHS to 1 on the bottom boundary
  if (bottom_only_inflow_one) {
    PetscInt xs_b, ys_b, zs_b, xm_b, ym_b, zm_b;
    if (dim == 2) {
      PetscScalar **b_arr;
      PetscCall(DMDAGetCorners(da, &xs_b, &ys_b, NULL, &xm_b, &ym_b, NULL));
      PetscCall(DMDAVecGetArray(da, b, &b_arr));
      if (ys_b == 0) {
        for (PetscInt i_b = xs_b; i_b < xs_b + xm_b; i_b++) b_arr[0][i_b] = 1.0;
      }
      PetscCall(DMDAVecRestoreArray(da, b, &b_arr));
    } else {
      PetscScalar ***b_arr;
      PetscCall(DMDAGetCorners(da, &xs_b, &ys_b, &zs_b, &xm_b, &ym_b, &zm_b));
      PetscCall(DMDAVecGetArray(da, b, &b_arr));
      if (zs_b == 0) {
        for (PetscInt j_b = ys_b; j_b < ys_b + ym_b; j_b++)
          for (PetscInt i_b = xs_b; i_b < xs_b + xm_b; i_b++)
            b_arr[0][j_b][i_b] = 1.0;
      }
      PetscCall(DMDAVecRestoreArray(da, b, &b_arr));
    }
  }

  // ~~~~~~~~~~~~~~
  // ~~~~~~~~~~~~~~

  // Compute our matrix
  PetscCall(ComputeMat(da, A, dim, u, v, w, L_x, L_y, L_z, alpha, adv_nondim, curved_velocity, unit_velocity));
  // This will compress out the extra memory
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &A_temp));
  PetscCall(MatDestroy(&A));
  A = A_temp;

  // Set the operator and options
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));

  // Diagonally scale our matrix 
  if (diag_scale) {
   PetscCall(VecDuplicate(x, &diag_vec));
   PetscCall(MatGetDiagonal(A, diag_vec));
   PetscCall(VecReciprocal(diag_vec));
   PetscCall(MatDiagonalScale(A, diag_vec, PETSC_NULLPTR));
   PetscCall(VecPointwiseMult(b, diag_vec, b));
   PetscCall(VecDestroy(&diag_vec));
  }

  // Setup the ksp
  PetscCall(PetscLogStagePush(setup));
  PetscCall(KSPSetUp(ksp));
  PetscCall(PetscLogStagePop());

  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(VecSet(x, 1.0));

  // Do a preliminary KSPSolve so all the vecs and mats get copied to the gpu
  // before the solve we're trying to time
  PetscCall(PetscLogStagePush(gpu_copy));
  PetscCall(KSPSolve(ksp,b,x));
  PetscCall(PetscLogStagePop());

  // Solve
  // We set x to 1 rather than random as the vecrandom doesn't yet have a
  // gpu implementation and we don't want a copy occuring back to the cpu
  if (second_solve)
  {
   PetscCall(VecSet(x, 1.0));
   PetscCall(KSPSolve(ksp,b,x));
  }

  // Write out the iteration count
  PetscCall(KSPGetConvergedReason(ksp,&reason));

  // Optionally write the solution to a VTK structured grid file.
  // Use -vec_view vtk:solution.vts on the command line to enable.
  PetscCall(VecViewFromOptions(x, NULL, "-vec_view"));
   
  // ~~~~~~~~~~~~~~
  // ~~~~~~~~~~~~~~  

  // Cleanup
  PetscCall(DMDestroy(&da));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  if (reason < 0)
  {
   return 1;
  }
  return 0;
}

PetscErrorCode ComputeMat(DM da, Mat A, PetscInt dim,
                          PetscScalar u, PetscScalar v, PetscScalar w,
                          PetscScalar L_x, PetscScalar L_y, PetscScalar L_z,
                          PetscScalar alpha, PetscBool adv_nondim,
                          PetscBool curved_velocity, PetscBool unit_velocity)
{
  PetscInt    i, j, k, M, N, P, xs, ys, zs, xm, ym, zm;
  PetscScalar val[7], Hx, Hy, Hz;
  PetscScalar HxdHy, HydHx, adv_x_scale, adv_y_scale;
  PetscScalar HyHz_Hx, HxHz_Hy, HxHy_Hz, adv_yz_scale, adv_xz_scale, adv_xy_scale;
  MatStencil  row, col[7];

  PetscCall(DMDAGetInfo(da, NULL, &M, &N, &P, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  PetscCheck(M > 1 && N > 1 && (dim == 2 || P > 1), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
             "Grid dimensions must be at least 2 points in each active direction");

  // Node-centred spacing: domain length divided by number of intervals
  Hx = L_x / (PetscReal)(M - 1);
  Hy = L_y / (PetscReal)(N - 1);
  Hz = (dim == 3) ? L_z / (PetscReal)(P - 1) : 1.0;

  HxdHy = Hx / Hy;
  HydHx = Hy / Hx;

  // 2D advection scale factors: multiply upwinded equation by Hx*Hy
  //   adv_y_scale goes with u (west neighbour, x-direction upwinding)
  //   adv_x_scale goes with v (south neighbour, y-direction upwinding)
  // For dimensionless pure advection (adv_nondim), divide through by Hx
  adv_x_scale = Hx;
  adv_y_scale = Hy;
  if (adv_nondim) {
    adv_x_scale = 1.0;
    adv_y_scale = HydHx;
  }

  // 3D advection scale factors: multiply upwinded equation by Hx*Hy*Hz
  //   adv_yz_scale goes with u (west neighbour)
  //   adv_xz_scale goes with v (south neighbour)
  //   adv_xy_scale goes with w (bottom neighbour)
  // For dimensionless pure advection (adv_nondim), divide through by Hx
  HyHz_Hx    = Hy * Hz / Hx;
  HxHz_Hy    = Hx * Hz / Hy;
  HxHy_Hz    = Hx * Hy / Hz;
  adv_yz_scale = Hy * Hz;
  adv_xz_scale = Hx * Hz;
  adv_xy_scale = Hx * Hy;
  if (adv_nondim) {
    // For 3D to be dimensionless like 2D, we divide by Hx*Hx
    adv_yz_scale = HyHz_Hx / Hx;
    adv_xz_scale = Hz / Hx;
    adv_xy_scale = Hy / Hx;
  }

  if (dim == 2) {

    PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

    for (j = ys; j < ys+ym; j++) {
      for (i = xs; i < xs+xm; i++) {
        row.i = i; row.j = j;

        // Compute velocity at this node
        PetscReal x_node[3] = {i * Hx, j * Hy, 0.0};
        PetscReal vel[3];
        GetVelocity(dim, (PetscReal)u, (PetscReal)v, (PetscReal)w, x_node, curved_velocity, unit_velocity, vel);
        PetscScalar u_loc = vel[0], v_loc = vel[1];

        // Boundary values
        if (i==0 || j==0 || i==M-1 || j==N-1) {

          // Dirichlets left and bottom
          if (i==0 || j==0) {
            val[0] = 1.0; col[0].i = i; col[0].j = j;
            PetscCall(MatSetValuesStencil(A, 1, &row, 1, col, val, ADD_VALUES));

          // Top or right - depends on if we have any diffusion
          } else {

            // If we have no diffusion, the top and right nodes just get the normal
            // upwinded stencil, representing an outflow bc
            if (alpha == 0.0) {
              // Upwind advection with theta between 0 and pi/2
              // south (lower y)
              val[0] = -v_loc * adv_x_scale;                  col[0].i = i;   col[0].j = j-1;
              // west (lower x)
              val[1] = -u_loc * adv_y_scale;                  col[1].i = i-1; col[1].j = j;
              // centre
              val[2] = u_loc*adv_y_scale + v_loc*adv_x_scale; col[2].i = i;   col[2].j = j;
              PetscCall(MatSetValuesStencil(A, 1, &row, 3, col, val, ADD_VALUES));

            // If we have diffusion we have dirichlet bcs on the top and right
            } else {
              val[0] = 1.0; col[0].i = i; col[0].j = j;
              PetscCall(MatSetValuesStencil(A, 1, &row, 1, col, val, ADD_VALUES));
            }
          }

        // interior stencil
        } else {

          // If we have diffusion
          if (alpha != 0.0) {
            // south
            val[0] = -alpha * HxdHy;              col[0].i = i;   col[0].j = j-1;
            // west
            val[1] = -alpha * HydHx;              col[1].i = i-1; col[1].j = j;
            // centre
            val[2] =  alpha * 2.0*(HxdHy+HydHx);  col[2].i = i;   col[2].j = j;
            // east
            val[3] = -alpha * HydHx;              col[3].i = i+1; col[3].j = j;
            // north
            val[4] = -alpha * HxdHy;              col[4].i = i;   col[4].j = j+1;
            PetscCall(MatSetValuesStencil(A, 1, &row, 5, col, val, ADD_VALUES));
          }

          // Upwind advection with theta between 0 and pi/2
          if (u_loc != 0.0 || v_loc != 0.0) {
            // south (lower y)
            val[0] = -v_loc * adv_x_scale;                  col[0].i = i;   col[0].j = j-1;
            // west (lower x)
            val[1] = -u_loc * adv_y_scale;                  col[1].i = i-1; col[1].j = j;
            // centre
            val[2] = u_loc*adv_y_scale + v_loc*adv_x_scale; col[2].i = i;   col[2].j = j;
            PetscCall(MatSetValuesStencil(A, 1, &row, 3, col, val, ADD_VALUES));
          }
        }
      }
    }

  } else { // dim == 3

    PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));

    for (k = zs; k < zs+zm; k++) {
      for (j = ys; j < ys+ym; j++) {
        for (i = xs; i < xs+xm; i++) {
          row.i = i; row.j = j; row.k = k;

          // Compute velocity at this node
          PetscReal x_node[3] = {i * Hx, j * Hy, k * Hz};
          PetscReal vel[3];
          GetVelocity(dim, (PetscReal)u, (PetscReal)v, (PetscReal)w, x_node, curved_velocity, unit_velocity, vel);
          PetscScalar u_loc = vel[0], v_loc = vel[1], w_loc = vel[2];

          // Boundary values
          if (i==0 || j==0 || k==0 || i==M-1 || j==N-1 || k==P-1) {

            // Dirichlets west, south and bottom (inflow faces)
            if (i==0 || j==0 || k==0) {
              val[0] = 1.0; col[0].i = i; col[0].j = j; col[0].k = k;
              PetscCall(MatSetValuesStencil(A, 1, &row, 1, col, val, ADD_VALUES));

            // East, north or top - depends on if we have any diffusion
            } else {

              // If we have no diffusion, the outflow nodes just get the normal
              // upwinded stencil, representing an outflow bc
              if (alpha == 0.0) {
                // Upwind advection with velocity in [0,1]^3 quadrant
                // bottom (lower z)
                val[0] = -w_loc * adv_xy_scale;                                        col[0].i = i;   col[0].j = j;   col[0].k = k-1;
                // south (lower y)
                val[1] = -v_loc * adv_xz_scale;                                        col[1].i = i;   col[1].j = j-1; col[1].k = k;
                // west (lower x)
                val[2] = -u_loc * adv_yz_scale;                                        col[2].i = i-1; col[2].j = j;   col[2].k = k;
                // centre
                val[3] = u_loc*adv_yz_scale + v_loc*adv_xz_scale + w_loc*adv_xy_scale; col[3].i = i;   col[3].j = j;   col[3].k = k;
                PetscCall(MatSetValuesStencil(A, 1, &row, 4, col, val, ADD_VALUES));

              // If we have diffusion we have dirichlet bcs on the east, north and top
              } else {
                val[0] = 1.0; col[0].i = i; col[0].j = j; col[0].k = k;
                PetscCall(MatSetValuesStencil(A, 1, &row, 1, col, val, ADD_VALUES));
              }
            }

          // interior stencil
          } else {

            // If we have diffusion
            if (alpha != 0.0) {
              // bottom
              val[0] = -alpha * HxHy_Hz; col[0].i = i;   col[0].j = j;   col[0].k = k-1;
              // south
              val[1] = -alpha * HxHz_Hy; col[1].i = i;   col[1].j = j-1; col[1].k = k;
              // west
              val[2] = -alpha * HyHz_Hx; col[2].i = i-1; col[2].j = j;   col[2].k = k;
              // centre
              val[3] =  alpha * 2.0*(HyHz_Hx + HxHz_Hy + HxHy_Hz); col[3].i = i; col[3].j = j; col[3].k = k;
              // east
              val[4] = -alpha * HyHz_Hx; col[4].i = i+1; col[4].j = j;   col[4].k = k;
              // north
              val[5] = -alpha * HxHz_Hy; col[5].i = i;   col[5].j = j+1; col[5].k = k;
              // top
              val[6] = -alpha * HxHy_Hz; col[6].i = i;   col[6].j = j;   col[6].k = k+1;
              PetscCall(MatSetValuesStencil(A, 1, &row, 7, col, val, ADD_VALUES));
            }

            // Upwind advection with velocity in [0,1]^3 quadrant
            if (u_loc != 0.0 || v_loc != 0.0 || w_loc != 0.0) {
              // bottom (lower z)
              val[0] = -w_loc * adv_xy_scale;                                        col[0].i = i;   col[0].j = j;   col[0].k = k-1;
              // south (lower y)
              val[1] = -v_loc * adv_xz_scale;                                        col[1].i = i;   col[1].j = j-1; col[1].k = k;
              // west (lower x)
              val[2] = -u_loc * adv_yz_scale;                                        col[2].i = i-1; col[2].j = j;   col[2].k = k;
              // centre
              val[3] = u_loc*adv_yz_scale + v_loc*adv_xz_scale + w_loc*adv_xy_scale; col[3].i = i;   col[3].j = j;   col[3].k = k;
              PetscCall(MatSetValuesStencil(A, 1, &row, 4, col, val, ADD_VALUES));
            }
          }
        }
      }
    }

  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  return PETSC_SUCCESS;
}
