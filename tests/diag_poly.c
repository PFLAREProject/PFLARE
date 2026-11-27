static char help[] = "Solves a diagonal system whose matrix has many more distinct eigenvalues than a polynomial\n\
preconditioner has roots, with a few small isolated eigenvalues.\n\
Input arguments are:\n\
  -n <n> : size of the matrix (default 5000). The diagonal is 0.01, 0.02, ..., 0.1, then 30, 60,\n\
           then 100, 101, 102, ...\n\
  -constant <c> : use a constant diagonal c instead, ie a multiple of the identity, for which the\n\
                  Arnoldi breaks down after one step\n\
  -true_residual_factor <f> : fail if ||b - Ax|| of the returned solution is more than f times\n\
                              the residual norm the KSP reports (default 10)\n\n";

#include <petscksp.h>
#include "pflare.h"

// For a diagonal matrix the rounding in a product form polynomial apply acts on every
// eigencomponent separately, so the order of the roots has to keep the partial products
// small at every eigenvalue, including the many that lie between the roots
// This is the diagonal matrix of Loe & Morgan (2022) with small isolated eigenvalues
// (called loe_diag7 in our tests); a reordering of the roots chosen only from the
// Arnoldi Hessenberg matrix can be much worse than a leja ordering here

int main(int argc,char **args)
{
  Vec            x, b, u;
  Mat            A;
  KSP            ksp;
  PC             pc;
  KSPConvergedReason reason;
  PetscInt       n = 5000, i, start, end;
  PetscReal      true_residual_factor = 10.0, rnorm_reported, rnorm_true, constant = 0.0;
  PetscBool      use_constant = PETSC_FALSE;
  PetscScalar    val;
  PetscBool      true_residual_bad = PETSC_FALSE;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-true_residual_factor", &true_residual_factor, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-constant", &constant, &use_constant));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, 1, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 1, NULL, 0, NULL));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A, &start, &end));
  for (i = start; i < end; i++)
  {
   if (use_constant) val = constant;
   else if (i < 10)  val = 0.01 * (PetscReal)(i + 1);
   else if (i == 10) val = 30.0;
   else if (i == 11) val = 60.0;
   else              val = 100.0 + (PetscReal)(i - 12);
   PetscCall(MatSetValues(A, 1, &i, 1, &i, &val, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  // Zero rhs and a random initial guess
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecDuplicate(b, &u));
  PetscCall(VecSetRandom(x, NULL));
  PetscCall(VecSet(b, 0.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(KSPGetPC(ksp, &pc));
  // Register the pflare types
  PCRegister_PFLARE();
  PetscCall(PCSetType(pc, PCPFLAREINV));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(KSPGetConvergedReason(ksp, &reason));

  // Compare the true residual of the returned solution with the one the KSP reports
  if (true_residual_factor > 0 && reason > 0)
  {
   PetscReal norm_ax, norm_b;
   PetscCall(KSPGetResidualNorm(ksp, &rnorm_reported));
   // u = b - A x
   PetscCall(MatMult(A, x, u));
   PetscCall(VecNorm(u, NORM_2, &norm_ax));
   PetscCall(VecNorm(b, NORM_2, &norm_b));
   PetscCall(VecAYPX(u, -1.0, b));
   PetscCall(VecNorm(u, NORM_2, &rnorm_true));
   // A true residual at rounding level always passes, as the reported one can be exactly zero
   // The comparison is false for a NaN
   if (!(rnorm_true <= true_residual_factor * rnorm_reported || rnorm_true <= 1000 * PETSC_MACHINE_EPSILON * (norm_ax + norm_b)))
   {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "True residual norm %g is more than %g times the reported residual norm %g\n", \
               (double)rnorm_true, (double)true_residual_factor, (double)rnorm_reported));
      true_residual_bad = PETSC_TRUE;
   }
  }

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  if (reason < 0 || true_residual_bad)
  {
   return 1;
  }
  return 0;
}
