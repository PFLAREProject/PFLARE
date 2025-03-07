/*   DMDA/KSP solving a system of linear equations.
     Steady advection-diffusion equation in 2D with finite difference, advection is upwinded

     ./adv_diff_2d.o 
             : pure advection with theta = pi/4, dimensionless
               BCs left and bottom dirichlet, top and right outflow
               Same equation as advection_2d in PyAMG, except we don't eliminate the dirichlet dofs
     ./adv_diff_2d.o -adv_nondim 0
             : pure advection with theta = pi/4, scaled by Hx * Hy
               BCs left and bottom dirichlet, top and right outflow
     ./adv_diff_2d.o -u 0 -v 0 -alpha 1.0 
             : pure diffusion scaled by Hx * Hy
               BCs dirichlet on all sides
     ./adv_diff_2d.o -alpha 1.0 
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

extern PetscErrorCode ComputeMat(DM,Mat,PetscScalar,PetscScalar, PetscScalar, PetscBool);

int main(int argc,char **argv)
{
  KSP            ksp;
  PC             pc;
  DM             da;
  PetscErrorCode ierr;
  PetscInt its, M, N;
  PetscScalar theta, alpha, u, v, u_test, v_test;
  PetscBool option_found_u, option_found_v, adv_nondim, check_nondim, diag_scale;
  Vec x, b, diag_vec;
  Mat A;
  KSPConvergedReason reason;
  PetscLogStage setup, gpu_copy;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //       Let's use PFLARE
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  

  // Register the pflare types
  PCRegister_PFLARE();

  PetscLogStageRegister("Setup", &setup);
  PetscLogStageRegister("GPU copy stage - triggered by a prelim KSPSolve", &gpu_copy);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,11,11,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = KSPSetDM(ksp,(DM)da);CHKERRQ(ierr);
  // We generate the matrix ourselves
  ierr = KSPSetDMActive(ksp, PETSC_FALSE);CHKERRQ(ierr);

  // Create empty matrix and vectors
  ierr = DMCreateMatrix(da, &A);
  ierr = DMCreateGlobalVector(da, &x);
  ierr = DMCreateGlobalVector(da, &b);

  // Zero rhs
  VecSet(b, 0.0);

  // ~~~~~~~~~~~~~~
  // Get command line options
  // ~~~~~~~~~~~~~~

  PetscBool second_solve= PETSC_FALSE;
  PetscOptionsGetBool(NULL, NULL, "-second_solve", &second_solve, NULL);  

  // Advection velocities - direction is [cos(theta), sin(theta)]
  // Default theta is pi/4
  PetscReal pi = 4*atan(1.0);
  theta = pi/4.0;
  PetscOptionsGetReal(NULL, NULL, "-theta", &theta, NULL);
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR >= 17)
      PetscCheck(theta <= pi/2.0 && theta >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "Theta must be between 0 and pi/2");
#else
      if (theta > pi/2.0 || theta < 0.0) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "Theta must be between 0 and pi/2");
#endif
  // Coefficients for 2d advection
  u = cos(theta);
  v = sin(theta);

  // Or the user can pass in the individual advection velocities
  // This will override theta
  PetscOptionsGetReal(NULL, NULL, "-u", &u_test, &option_found_u);
  PetscOptionsGetReal(NULL, NULL, "-v", &v_test, &option_found_v);

#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR >= 17)
      if (option_found_u) PetscCheck(u_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "u must be positive");
      if (option_found_v) PetscCheck(v_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "v must be positive");
#else
      if (option_found_u) if (u_test < 0) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "u must be positive");
      if (option_found_v) if (v_test < 0) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "v must be positive");
#endif  
  if (option_found_u && option_found_v) {
   u = u_test;
   v = v_test;
  }

  // Diffusion coefficient
  // Default alpha is 0 - pure advection
  alpha = 0.0;
  PetscOptionsGetReal(NULL, NULL, "-alpha", &alpha, NULL);

  // If we just have advection, rather than scaling by Hx * Hy, we can just have 
  // a dimensionless advection problem - this is enabled by default
  // If we have any diffusion this is turned off by default
  adv_nondim = PETSC_TRUE;
  if (alpha != 0.0)
  {
   adv_nondim = PETSC_FALSE;
  }  
  PetscOptionsGetBool(NULL, NULL, "-adv_nondim", &adv_nondim, NULL);
  // We can only nondimensionalise the advection if we don't have any diffusion
  check_nondim = PETSC_TRUE;
  if (alpha != 0.0)
  {
   if (adv_nondim)
   {
      check_nondim = PETSC_FALSE;
   }
  }

#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR >= 17)
      PetscCheck(check_nondim, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "Non-dimensional advection only applies without diffusion");
#else
      if (!check_nondim) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "Non-dimensional advection only applies without diffusion");
#endif  

  // Do we diagonally scale our matrix before solving
  // Defaults to false
  diag_scale = PETSC_FALSE;
  PetscOptionsGetBool(NULL, NULL, "-diag_scale", &diag_scale, NULL);

  // ~~~~~~~~~~~~~~
  // ~~~~~~~~~~~~~~

  // Compute our matrix
  ComputeMat(da, A, u, v, alpha, adv_nondim);

  // Set the operator and options
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);

  ierr  = DMDAGetInfo(da,0,&M,&N,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr); 

  // Diagonally scale our matrix 
  if (diag_scale) {
   ierr = VecDuplicate(x, &diag_vec);CHKERRQ(ierr);
   ierr = MatGetDiagonal(A, diag_vec);CHKERRQ(ierr);
   ierr = VecReciprocal(diag_vec);CHKERRQ(ierr);
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR >= 19)
   ierr = MatDiagonalScale(A, diag_vec, PETSC_NULLPTR);CHKERRQ(ierr);    
#else
   ierr = MatDiagonalScale(A, diag_vec, PETSC_NULL);CHKERRQ(ierr);    
#endif
   ierr = VecPointwiseMult(b, diag_vec, b); CHKERRQ(ierr);
   ierr = VecDestroy(&diag_vec); CHKERRQ(ierr);
  }

  // Setup the ksp
  ierr = PetscLogStagePush(setup);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
  ierr = VecSet(x, 1.0);CHKERRQ(ierr);

  // Do a preliminary KSPSolve so all the vecs and mats get copied to the gpu
  // before the solve we're trying to time
  ierr = PetscLogStagePush(gpu_copy);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  // Solve
  // We set x to 1 rather than random as the vecrandom doesn't yet have a
  // gpu implementation and we don't want a copy occuring back to the cpu
  if (second_solve)
  {
   ierr = VecSet(x, 1.0);CHKERRQ(ierr);
   ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  }

  // Write out the iteration count
  KSPGetIterationNumber(ksp,&its);
  KSPGetConvergedReason(ksp,&reason);
   
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Number of iterations = %3" PetscInt_FMT "\n", its);

  // ~~~~~~~~~~~~~~
  // ~~~~~~~~~~~~~~  

  // Cleanup
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  if (reason < 0)
  {
   return 1;
  }
  return 0;
}

PetscErrorCode ComputeMat(DM da, Mat A, PetscScalar u, PetscScalar v, PetscScalar alpha, PetscBool adv_nondim)
{
  PetscErrorCode ierr;
  PetscInt       i, j, M, N, xm, ym, xs, ys;
  PetscScalar    val[5], Hx, Hy, HydHx, HxdHy, adv_x_scale, adv_y_scale;
  MatStencil     row, col[5];

  ierr  = DMDAGetInfo(da,0,&M,&N,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx    = 1.0 / (PetscReal)(M);
  Hy    = 1.0 / (PetscReal)(N);
  HxdHy = Hx/Hy;
  HydHx = Hy/Hx;
  adv_x_scale = Hx;
  adv_y_scale = Hy;
  // Don't need to scale the advection terms if dimensionless
  if (adv_nondim) {
   adv_x_scale = 1;
   adv_y_scale = 1;   
  }

  ierr  = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);

  // Loop over the nodes
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;

      // Boundary values
      if (i==0 || j==0 || i==M-1 || j==N-1) {
         
         // Dirichlets left and bottom
         if (i==0 || j==0) {

            val[0] = 1.0; col[0].i = i;   col[0].j = j;        
            ierr = MatSetValuesStencil(A,1,&row,1,col,val,ADD_VALUES);CHKERRQ(ierr);      

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
               ierr = MatSetValuesStencil(A,1,&row,3,col,val,ADD_VALUES);CHKERRQ(ierr);                

            // If we have diffusion we have dirichlet bcs on the top and right
            } else{

               val[0] = 1.0; col[0].i = i;   col[0].j = j;        
               ierr = MatSetValuesStencil(A,1,&row,1,col,val,ADD_VALUES);CHKERRQ(ierr);                   
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
            ierr = MatSetValuesStencil(A,1,&row,5,col,val,ADD_VALUES);CHKERRQ(ierr);            
         }

        // Upwind advection with theta between 0 and pi/2
        if (u != 0.0 || v != 0.0) {
            // left
            val[0] = -u * adv_y_scale;                 col[0].i = i;   col[0].j = j-1;
            // bottom
            val[1] = -v * adv_x_scale;                 col[1].i = i-1; col[1].j = j;
            // centre
            val[2] = u*adv_y_scale + v*adv_x_scale;    col[2].i = i;   col[2].j = j;   
            ierr = MatSetValuesStencil(A,1,&row,3,col,val,ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  // Eliminate the zeros that the 5 point stencil may have in it with only advection
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR>=20)
  MatFilter(A, 0.0, PETSC_TRUE, PETSC_TRUE);
#endif

  return 0;
}