static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
Input arguments are:\n\
  -f <input_file> : file to load. For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\n";

#include <petscksp.h>
#include <petsclog.h>
#include "pflare.h"

int main(int argc,char **args)
{
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

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-b_in_f",&b_in_f,NULL));

  /* Read matrix and RHS */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate binary file with the -f option");
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatLoad(A,fd));
  if (b_in_f) {
    PetscCall(VecCreate(PETSC_COMM_WORLD,&b));
    PetscCall(VecLoad(b,fd));
  } else {
    PetscCall(MatCreateVecs(A,NULL,&b));
    PetscCall(VecSetRandom(b,NULL));
  }
  PetscCall(PetscViewerDestroy(&fd));

  /*
   If the load matrix is larger then the vector, due to being padded
   to match the blocksize then create a new padded vector
  */
  {
    PetscInt    m,n,j,mvec,start,end,indx;
    Vec         tmp;
    PetscScalar *bold;

    PetscCall(MatGetLocalSize(A,&m,&n));
    PetscCall(VecCreate(PETSC_COMM_WORLD,&tmp));
    PetscCall(VecSetSizes(tmp,m,PETSC_DECIDE));
    PetscCall(VecGetType(b, &vtype));
    PetscCall(VecSetType(tmp, vtype));
    PetscCall(VecGetOwnershipRange(b,&start,&end));
    PetscCall(VecGetLocalSize(b,&mvec));
    PetscCall(VecGetArray(b,&bold));
    for (j=0; j<mvec; j++) {
      indx = start+j;
      PetscCall(VecSetValues(tmp,1,&indx,bold+j,INSERT_VALUES));
    }
    PetscCall(VecRestoreArray(b,&bold));
    PetscCall(VecDestroy(&b));
    PetscCall(VecAssemblyBegin(tmp));
    PetscCall(VecAssemblyEnd(tmp));
    b    = tmp;
  }
  PetscCall(VecDuplicate(b,&x));
  PetscCall(VecDuplicate(b,&u));

  PetscCall(VecSet(x,0.0));
  PetscCall(PetscBarrier((PetscObject)A));

  PetscCall(PetscLogStageRegister("mystage 1",&stage1));
  PetscCall(PetscLogStagePush(stage1));

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //       Compute a cf splitting
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  

 // Threshold for a strong connection
 PetscReal strong_threshold = 0.5;
 // Second pass cleanup - one iteration
 int ddc_its = 1;
 // Fraction of F points to convert to C per ddc it
 PetscReal ddc_fraction = 0.1;
 // If not 0, keep doing ddc its until this diagonal dominance
 // ratio is hit
 PetscReal max_dd_ratio = 0.0;
 // As many steps as needed
 int max_luby_steps = -1;
 // PMISR DDC
 int algorithm = CF_PMISR_DDC;
 // Is the matrix symmetric?
 int symmetric = 0;

 compute_cf_splitting_c(&A, \
     symmetric, \
     strong_threshold, max_luby_steps, \
     algorithm, \
     ddc_its, \
     ddc_fraction, \
     max_dd_ratio, \
     &is_fine, &is_coarse);

  PetscInt n_fine, n_coarse;
  PetscInt local_rows, local_cols;
  PetscCall(MatGetLocalSize(A, &local_rows, &local_cols));
  PetscCall(ISGetLocalSize(is_fine, &n_fine));
  PetscCall(ISGetLocalSize(is_coarse, &n_coarse));

  if (n_fine + n_coarse == local_rows)
  {
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, "OK \n"));
  }
  else{
   PetscCall(PetscPrintf(PETSC_COMM_WORLD, "NOT OK \n"));
   return 1;
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /* Cleanup */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(MatDestroy(&A));
  PetscCall(ISDestroy(&is_fine));
  PetscCall(ISDestroy(&is_coarse));
  PetscCall(PetscFinalize());
  return 0;
}