static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
Input arguments are:\n\
  -f <input_file> : file to load. For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\n";

#include <petscksp.h>
#include <petsclog.h>
#include "pflare.h"

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       its;
#if defined(PETSC_USE_LOG)
  PetscLogStage  stage1,stage2, gpu_copy;
#endif
  PetscReal      norm;
  Vec            x,b,u, diag_vec, b_diff_type;
  Mat            A, A_diff_type;
  char           file[PETSC_MAX_PATH_LEN];
  PetscViewer    fd;
  PetscBool      flg,b_in_f = PETSC_TRUE, diag_scale = PETSC_FALSE;
  KSP            ksp;
  PC             pc;
  KSPConvergedReason reason;
  VecType vtype;
  PetscInt one=1, m, n, M, N;
  MatType mtype, mtype_input;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetBool(NULL,NULL,"-b_in_f",&b_in_f,NULL);CHKERRQ(ierr);

  PetscBool second_solve= PETSC_FALSE;
  PetscOptionsGetBool(NULL, NULL, "-second_solve", &second_solve, NULL);   

  /* Read matrix and RHS */
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate binary file with the -f option");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  if (b_in_f) {
    ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
    ierr = VecLoad(b,fd);CHKERRQ(ierr);
  } else {
    ierr = MatCreateVecs(A,NULL,&b);CHKERRQ(ierr);
    ierr = VecSetRandom(b,NULL);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  PetscOptionsGetBool(NULL, NULL, "-diag_scale", &diag_scale, NULL);

  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);

  // Test and see if the user wants us to use a different matrix type
  // with -mat_type on the command line
  // This lets us easily test our cpu and kokkos versions through our CI
  ierr = MatCreateFromOptions(PETSC_COMM_WORLD,NULL,\
               one,m,n,M,N,&A_diff_type);
  ierr = MatAssemblyBegin(A_diff_type, MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(A_diff_type, MAT_FINAL_ASSEMBLY);
      
  ierr = MatGetType(A, &mtype);
  ierr = MatGetType(A_diff_type, &mtype_input); 

  if (mtype != mtype_input) 
  {
      // Doesn't seem like there is a converter to kokkos
      // So instead we just copy into the empty A_diff_type
      // This will be slow as its not preallocated, but this is just for testing
      ierr = MatCopy(A, A_diff_type, DIFFERENT_NONZERO_PATTERN);
      ierr = MatDestroy(&A);
      A = A_diff_type;

      // Mat and vec types have to match
      ierr = VecCreateFromOptions(PETSC_COMM_WORLD,NULL, \
               one,n,N,&b_diff_type);
      ierr = VecCopy(b,b_diff_type);
      ierr = VecDestroy(&b);
      b = b_diff_type;
  }
  else
  {
      MatDestroy(&A_diff_type);
  }   

  /*
   If the load matrix is larger then the vector, due to being padded
   to match the blocksize then create a new padded vector
  */
  {
    PetscInt    j,mvec,start,end,indx;
    Vec         tmp;
    PetscScalar *bold;

    ierr = VecCreate(PETSC_COMM_WORLD,&tmp);CHKERRQ(ierr);
    ierr = VecSetSizes(tmp,m,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecGetType(b, &vtype);CHKERRQ(ierr);
    ierr = VecSetType(tmp, vtype);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(b,&start,&end);CHKERRQ(ierr);
    ierr = VecGetLocalSize(b,&mvec);CHKERRQ(ierr);
    ierr = VecGetArray(b,&bold);CHKERRQ(ierr);
    for (j=0; j<mvec; j++) {
      indx = start+j;
      ierr = VecSetValues(tmp,1,&indx,bold+j,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(b,&bold);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(tmp);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(tmp);CHKERRQ(ierr);
    b    = tmp;
  }
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&u);CHKERRQ(ierr);

  // If we don't load b, just set x to random and the rhs to zero
  if (!b_in_f) {
    ierr = VecSetRandom(x,NULL);CHKERRQ(ierr);
    ierr = VecSet(b,0.0);CHKERRQ(ierr);
  }
  ierr = PetscBarrier((PetscObject)A);CHKERRQ(ierr);

  // Diagonally scale our matrix 
  if (diag_scale) {
   ierr = VecDuplicate(x, &diag_vec);
   ierr = MatGetDiagonal(A, diag_vec);
   ierr = VecReciprocal(diag_vec);
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR >= 19)
   ierr = MatDiagonalScale(A, diag_vec, PETSC_NULLPTR);    
#else
   ierr = MatDiagonalScale(A, diag_vec, PETSC_NULL);    
#endif
   ierr = VecPointwiseMult(b, diag_vec, b); CHKERRQ(ierr);
   ierr = VecDestroy(&diag_vec); CHKERRQ(ierr);
  }  

  ierr = PetscLogStageRegister("mystage 1",&stage1);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage1);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //       Let's use AIRG as our PC
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  

  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
  // Register the pflare types
  PCRegister_PFLARE();
  ierr = PCSetType(pc, PCAIR);CHKERRQ(ierr);
  ierr = KSPSetPC(ksp, pc);CHKERRQ(ierr);

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSetUpOnBlocks(ksp);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  ierr = PetscBarrier((PetscObject)A);CHKERRQ(ierr);

  ierr = PetscLogStageRegister("GPU copy stage - triggered by a prelim KSPSolve",&gpu_copy);CHKERRQ(ierr);
  ierr = PetscLogStagePush(gpu_copy);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);  

  if (second_solve)
  {
   ierr = VecSet(x, 1.0);CHKERRQ(ierr);
   ierr = PetscLogStageRegister("mystage 2",&stage2);CHKERRQ(ierr);
   ierr = PetscLogStagePush(stage2);CHKERRQ(ierr);
   ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
   ierr = PetscLogStagePop();CHKERRQ(ierr);  
  }

  /* Show result */
  ierr = MatMult(A,x,u);CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
 
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Number of iterations = %3" PetscInt_FMT "\n", its);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Residual norm = %g\n", (double)norm);

  /* Cleanup */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  if (reason < 0)
  {
   return 1;
  }
  return 0;
}