/*   DMDA/KSP solving a system of linear equations.
     Steady advection-diffusion equation in 2D with finite difference, advection is upwinded

     ./adv_diff_2d
             : pure advection with theta = pi/4, dimensionless
               BCs left and bottom dirichlet (u=0), top and right outflow
               Same equation as advection_2d in PyAMG, except we don't eliminate the dirichlet dofs
     ./adv_diff_2d -adv_nondim 0
             : pure advection with theta = pi/4, scaled by Hx * Hy
               BCs left and bottom dirichlet (u=0), top and right outflow
     ./adv_diff_2d -u 0 -v 0 -alpha 1.0 
             : pure diffusion scaled by Hx * Hy
               BCs dirichlet on all sides
     ./adv_diff_2d -alpha 1.0 
             : advection-diffusion scaled by Hx * Hy with theta=pi/4             
               BCs dirichlet on all sides
     ./adv_diff_2d -bottom_only_inflow_one
             : pure advection with theta = pi/4, dimensionless
               BC bottom face dirichlet inflow u=1, left face dirichlet u=0, top and right outflow

     Can control the direction of advection with -theta (pi/4 default), or by giving the -u and -v directly
     Can optionally left scale the matrix by the inverse diagonal before solving (-diag_scale)
     Can specify inflow of 1 on the bottom face with -bottom_only_inflow_one (default false),
       left face and other inflow faces are set to u=0
     Can change default velocity from straight line to curved with -curved_velocity (default false)
       curved velocity field: u(x,y) = y, v(x,y) = 1-x (rotating, always >= 0 on [0,1]^2)
     Can normalise velocity with -unit_velocity (default true) so that we have a unit velocity.
       If any of u,v are set explicitly then unit velocity will be disabled
     Can write the solution to a VTK structured grid file with -vec_view vtk:solution.vts
       (note: DMDA is a structured mesh so PETSc produces .vts, not .vtu)

*/

static char help[] = "Solves 2D steady advection-diffusion on a structured grid.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscvec.h>

#include "pflare.h"

// Helper function to compute velocity at a point.
// u_const, v_const: constant velocity components (used when curved_velocity is false)
// x[2]: node coordinates (x, y)
// curved_velocity: if true, use spatially-varying field u(x,y) = y, v(x,y) = 1-x
// unit_velocity: if true, normalise the velocity to unit magnitude
static inline void GetVelocity(PetscReal u_const, PetscReal v_const, const PetscReal x[],
                               PetscBool curved_velocity, PetscBool unit_velocity,
                               PetscReal vel[])
{
  if (curved_velocity) {
    // Spatially-varying velocity field: top-left quadrant of a rotating circle (center at 1,0)
    // u(x,y) = y, v(x,y) = 1-x
    vel[0] = x[1];        // u(x,y) = y
    vel[1] = 1.0 - x[0]; // v(x,y) = 1-x
  } else {
    // Constant velocity field
    vel[0] = u_const;
    vel[1] = v_const;
  }

  // Normalise velocity magnitude if unit_velocity flag is set
  if (unit_velocity) {
    PetscReal mag = PetscSqrtReal(vel[0]*vel[0] + vel[1]*vel[1]);
    if (mag > 1e-12) { vel[0] /= mag; vel[1] /= mag; }
  }
}

extern PetscErrorCode ComputeMat(DM,Mat,PetscScalar,PetscScalar,PetscScalar,PetscScalar, PetscScalar, PetscBool, PetscBool, PetscBool);

int main(int argc,char **argv)
{
  KSP            ksp;
  PC             pc;
  DM             da;
  PetscInt M, N;
  PetscScalar theta, alpha, u, v, u_test, v_test, L_x, L_y, L_x_test, L_y_test;
  PetscBool option_found_u, option_found_v, adv_nondim, check_nondim, diag_scale, bottom_only_inflow_one, curved_velocity, unit_velocity;
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

  // Dimensions of box, L_y x L_x - default to [0, 1]^2
  L_x = 1.0;
  L_y = 1.0;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-L_x", &L_x_test, &option_found_u));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-L_y", &L_y_test, &option_found_v));

  if (option_found_u) PetscCheck(L_x_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "L_x must be positive");
  if (option_found_v) PetscCheck(L_y_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "L_y must be positive");

  if (option_found_u) {
   L_x = L_x_test;
  }
  if (option_found_v) {
   L_y = L_y_test;
  }    

  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,11,11,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  // This stops the zero entries in the stencil from being added to the matrix
  // It still allocates the memory for a 5 point stencil but the sparsity doesn't include the zero entries
  // We do this instead of calling MatFilter as there is no Kokkos implementation so its very slow
  PetscCall(DMSetMatrixPreallocateOnly(da,PETSC_TRUE));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0.0, L_x, 0.0, L_y, 0.0, 0.0));
  PetscCall(KSPSetDM(ksp,(DM)da));
  // We generate the matrix ourselves
  PetscCall(KSPSetDMActive(ksp, PETSC_FALSE));

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

  // Advection velocities - direction is [cos(theta), sin(theta)]
  // Default theta is pi/4
  PetscReal pi = 4*atan(1.0);
  theta = pi/4.0;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-theta", &theta, NULL));
  PetscCheck(theta <= pi/2.0 && theta >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "Theta must be between 0 and pi/2");

  // Coefficients for 2d advection
  u = cos(theta);
  v = sin(theta);

  // Or the user can pass in the individual advection velocities
  // This will override theta
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-u", &u_test, &option_found_u));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-v", &v_test, &option_found_v));

  if (option_found_u) PetscCheck(u_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "u must be positive");
  if (option_found_v) PetscCheck(v_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "v must be positive");

  if (option_found_u && option_found_v) {
   u = u_test;
   v = v_test;
  }

  // Curved velocity field option (u(x,y) = y, v(x,y) = 1-x)
  // Default is straight line (false)
  curved_velocity = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-curved_velocity", &curved_velocity, NULL));

  // Unit velocity normalization option - default to unit velocity
  unit_velocity = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-unit_velocity", &unit_velocity, NULL));

  // Don't normalise if user has explicitly set a velocity
  if (option_found_u || option_found_v) {
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

  // Set only the bottom face (j==0) inflow to u=1, all other inflow faces (left, i==0) to u=0
  // Defaults to false (all inflow faces are u=0)
  bottom_only_inflow_one = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-bottom_only_inflow_one", &bottom_only_inflow_one, NULL));

  // If bottom_only_inflow_one is set, set the RHS to 1 on the bottom boundary (j==0)
  if (bottom_only_inflow_one) {
    PetscScalar **b_arr;
    PetscInt     xs_b, ys_b, xm_b, ym_b, i_b;
    PetscCall(DMDAGetCorners(da, &xs_b, &ys_b, 0, &xm_b, &ym_b, 0));
    PetscCall(DMDAVecGetArray(da, b, &b_arr));
    if (ys_b == 0) {
      for (i_b = xs_b; i_b < xs_b + xm_b; i_b++) {
        b_arr[0][i_b] = 1.0;
      }
    }
    PetscCall(DMDAVecRestoreArray(da, b, &b_arr));
  }

  // ~~~~~~~~~~~~~~
  // ~~~~~~~~~~~~~~

  // Compute our matrix
  PetscCall(ComputeMat(da, A, u, v, L_x, L_y, alpha, adv_nondim, curved_velocity, unit_velocity));
  // This will compress out the extra memory
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &A_temp));
  PetscCall(MatDestroy(&A));
  A = A_temp;

  // Set the operator and options
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));

  PetscCall(DMDAGetInfo(da,0,&M,&N,0,0,0,0,0,0,0,0,0,0));

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

PetscErrorCode ComputeMat(DM da, Mat A, PetscScalar u, PetscScalar v, PetscScalar L_x, PetscScalar L_y, PetscScalar alpha, PetscBool adv_nondim, PetscBool curved_velocity, PetscBool unit_velocity)
{
  PetscInt       i, j, M, N, xm, ym, xs, ys;
  PetscScalar    val[5], Hx, Hy, HydHx, HxdHy, adv_x_scale, adv_y_scale;
  MatStencil     row, col[5];

  PetscCall(DMDAGetInfo(da,0,&M,&N,0,0,0,0,0,0,0,0,0,0));
  Hx    = L_x / (PetscReal)(M);
  Hy    = L_y / (PetscReal)(N);
  HxdHy = Hx/Hy;
  HydHx = Hy/Hx;
  adv_x_scale = Hx;
  adv_y_scale = Hy;
  // If dimensionless
  if (adv_nondim) {
   adv_x_scale = 1;
   adv_y_scale = HydHx;   
  }

  // Node spacing for coordinate computation
  // M nodes span [0, L_x], so node i is at x = i * L_x / (M-1)
  PetscReal Hx_node = (M > 1) ? L_x / (PetscReal)(M - 1) : 0.0;
  PetscReal Hy_node = (N > 1) ? L_y / (PetscReal)(N - 1) : 0.0;

  PetscCall(DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0));

  // Loop over the nodes
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;

      // Compute velocity at this node via GetVelocity
      PetscReal x_node[2] = {i * Hx_node, j * Hy_node};
      PetscReal vel[2];
      GetVelocity((PetscReal)u, (PetscReal)v, x_node, curved_velocity, unit_velocity, vel);
      PetscScalar u_loc = vel[0], v_loc = vel[1];

      // Boundary values
      if (i==0 || j==0 || i==M-1 || j==N-1) {
         
         // Dirichlets left and bottom
         if (i==0 || j==0) {

            val[0] = 1.0; col[0].i = i;   col[0].j = j;        
            PetscCall(MatSetValuesStencil(A,1,&row,1,col,val,ADD_VALUES));

         // Top or right - depends on if we have any diffusion
         } else {

            // If we have no diffusion, the top and right nodes just get the normal
            // upwinded stencil, representing an outflow bc
            if (alpha == 0.0){
               // Upwind advection with theta between 0 and pi/2
               // south (lower y)
               val[0] = -v_loc * adv_x_scale;                     col[0].i = i;   col[0].j = j-1;
               // west (lower x)
               val[1] = -u_loc * adv_y_scale;                     col[1].i = i-1; col[1].j = j;
               // centre
               val[2] = u_loc*adv_y_scale + v_loc*adv_x_scale;    col[2].i = i;   col[2].j = j;
               PetscCall(MatSetValuesStencil(A,1,&row,3,col,val,ADD_VALUES));

            // If we have diffusion we have dirichlet bcs on the top and right
            } else{

               val[0] = 1.0; col[0].i = i;   col[0].j = j;
               PetscCall(MatSetValuesStencil(A,1,&row,1,col,val,ADD_VALUES));
            }
         }

      // interior stencil
      } else {

         // If we have diffusion
         if (alpha != 0.0) {

            // bottom
            val[0] = -alpha * HxdHy;              col[0].i = i;   col[0].j = j-1;
            // left
            val[1] = -alpha * HydHx;              col[1].i = i-1; col[1].j = j;
            // centre
            val[2] = alpha * 2.0*(HxdHy + HydHx); col[2].i = i;   col[2].j = j;
            // right
            val[3] = -alpha * HydHx;              col[3].i = i+1; col[3].j = j;
            // top
            val[4] = -alpha * HxdHy;              col[4].i = i;   col[4].j = j+1;
            PetscCall(MatSetValuesStencil(A,1,&row,5,col,val,ADD_VALUES));
         }

        // Upwind advection with theta between 0 and pi/2
        if (u_loc != 0.0 || v_loc != 0.0) {
            // south (lower y)
            val[0] = -v_loc * adv_x_scale;                     col[0].i = i;   col[0].j = j-1;
            // west (lower x)
            val[1] = -u_loc * adv_y_scale;                     col[1].i = i-1; col[1].j = j;
            // centre
            val[2] = u_loc*adv_y_scale + v_loc*adv_x_scale;    col[2].i = i;   col[2].j = j;
            PetscCall(MatSetValuesStencil(A,1,&row,3,col,val,ADD_VALUES));
        }
      }
    }
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  return PETSC_SUCCESS;
}