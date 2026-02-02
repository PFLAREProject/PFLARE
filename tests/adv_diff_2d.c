/*   DMDA/KSP solving a system of linear equations.
     Steady advection-diffusion equation in 2D with finite difference, advection is upwinded

     ./adv_diff_2d
             : pure advection with theta = pi/4, dimensionless
               BCs left and bottom dirichlet, top and right outflow
               Same equation as advection_2d in PyAMG, except we don't eliminate the dirichlet dofs
     ./adv_diff_2d -adv_nondim 0
             : pure advection with theta = pi/4, scaled by Hx * Hy
               BCs left and bottom dirichlet, top and right outflow
     ./adv_diff_2d -u 0 -v 0 -alpha 1.0 
             : pure diffusion scaled by Hx * Hy
               BCs dirichlet on all sides
     ./adv_diff_2d -alpha 1.0 
             : advection-diffusion scaled by Hx * Hy with theta=pi/4             
               BCs dirichlet on all sides

     Can control the direction of advection with -theta (pi/4 default), or by giving the -u and -v directly
     Can optionally left scale the matrix by the inverse diagonal before solving (-diag_scale)
     Modified from ex50.c by Michael Boghosian <boghmic@iit.edu>, 2008,

*/

static char help[] = "Solves 2D steady advection-diffusion on a structured grid.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscvec.h>

#include "pflare.h"

extern PetscErrorCode ComputeMat(DM,Mat,PetscScalar,PetscScalar,PetscScalar,PetscScalar, PetscScalar, PetscBool);

int main(int argc,char **argv)
{
  KSP            ksp;
  PC             pc;
  DM             da;
  PetscInt M, N;
  PetscScalar theta, alpha, u, v, u_test, v_test, L_x, L_y, L_x_test, L_y_test;
  PetscBool option_found_u, option_found_v, adv_nondim, check_nondim, diag_scale;
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
  PetscCall(KSPSetDMActive(ksp, KSP_DMACTIVE_ALL, PETSC_FALSE));

  // Create empty matrix and vectors
  PetscCall(DMCreateMatrix(da, &A));
  PetscCall(MatSetOption(A,MAT_IGNORE_ZERO_ENTRIES ,PETSC_TRUE));
  PetscCall(DMCreateGlobalVector(da, &x));
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

  // ~~~~~~~~~~~~~~
  // ~~~~~~~~~~~~~~

  // Compute our matrix
  PetscCall(ComputeMat(da, A, u, v, L_x, L_y, alpha, adv_nondim));
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

PetscErrorCode ComputeMat(DM da, Mat A, PetscScalar u, PetscScalar v, PetscScalar L_x, PetscScalar L_y, PetscScalar alpha, PetscBool adv_nondim)
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

  PetscCall(DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0));

  // Loop over the nodes
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;

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
               // left
               val[0] = -u * adv_y_scale;                 col[0].i = i;   col[0].j = j-1;
               // bottom
               val[1] = -v * adv_x_scale;                 col[1].i = i-1; col[1].j = j;
               // centre
               val[2] = u*adv_y_scale + v*adv_x_scale;    col[2].i = i;   col[2].j = j;
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
        if (u != 0.0 || v != 0.0) {
            // left
            val[0] = -u * adv_y_scale;                 col[0].i = i;   col[0].j = j-1;
            // bottom
            val[1] = -v * adv_x_scale;                 col[1].i = i-1; col[1].j = j;
            // centre
            val[2] = u*adv_y_scale + v*adv_x_scale;    col[2].i = i;   col[2].j = j;
            PetscCall(MatSetValuesStencil(A,1,&row,3,col,val,ADD_VALUES));
        }
      }
    }
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  return PETSC_SUCCESS;
}
