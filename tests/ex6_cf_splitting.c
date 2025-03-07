static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
Input arguments are:\n\
  -f <input_file> : file to load. For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\n";

#include <petscksp.h>
#include <petsclog.h>
#include "pflare.h"

int main(int argc,char **args)
{
  PetscErrorCode ierr;
#if defined(PETSC_USE_LOG)
  PetscLogStage  stage1;
#endif
  Vec            x,b,u;
  Mat            A;
  char           file[PETSC_MAX_PATH_LEN];
  PetscViewer    fd;
  PetscBool      flg,b_in_f = PETSC_TRUE;
  IS is_fine, is_coarse;
  VecType vtype;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetBool(NULL,NULL,"-b_in_f",&b_in_f,NULL);CHKERRQ(ierr);

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

  /*
   If the load matrix is larger then the vector, due to being padded
   to match the blocksize then create a new padded vector
  */
  {
    PetscInt    m,n,j,mvec,start,end,indx;
    Vec         tmp;
    PetscScalar *bold;

    ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
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

  ierr = VecSet(x,0.0);CHKERRQ(ierr);
  ierr = PetscBarrier((PetscObject)A);CHKERRQ(ierr);

  ierr = PetscLogStageRegister("mystage 1",&stage1);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage1);CHKERRQ(ierr);

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //       Compute a cf splitting
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  

 // Threshold for a strong connection
 PetscReal strong_threshold = 0.5;
 // Second pass cleanup
 PetscReal ddc_fraction = 0.1;
 // As many steps as needed
 PetscInt max_luby_steps = -1;
 // PMISR DDC
 int algorithm = CF_PMISR_DDC;
 // Is the matrix symmetric?
 int symmetric = 0;

 compute_cf_splitting_c(&A, \
     symmetric, \
     strong_threshold, max_luby_steps, \
     algorithm, \
     ddc_fraction, \
     &is_fine, &is_coarse);

  PetscInt n_fine, n_coarse;
  PetscInt local_rows, local_cols;
  ierr = MatGetLocalSize(A, &local_rows, &local_cols);
  ierr = ISGetLocalSize(is_fine, &n_fine);
  ierr = ISGetLocalSize(is_coarse, &n_coarse);

  if (n_fine + n_coarse == local_rows)
  {
   ierr = PetscPrintf(PETSC_COMM_WORLD, "OK \n");
  }
  else{
   ierr = PetscPrintf(PETSC_COMM_WORLD, "NOT OK \n");
   return 1;
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /* Cleanup */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = ISDestroy(&is_fine); CHKERRQ(ierr);
  ierr = ISDestroy(&is_coarse); CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}