static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
Input arguments are:\n\
  -f <input_file> : file to load. For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\n";

#include <petscksp.h>
#include <petsclog.h>
#include "pflare.h"

int main(int argc,char **args)
{
#if defined(PETSC_USE_LOG)
  PetscLogStage  stage1,stage2, gpu_copy;
#endif
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

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-b_in_f",&b_in_f,NULL));

  PetscBool second_solve= PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-second_solve", &second_solve, NULL));

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
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-diag_scale", &diag_scale, NULL));

  // ~~~~~~~~~~~~~~~~~~
  // If we're in parallel do a partitioning so our loaded matrix is sensibly distributed
  MatPartitioning part;
  IS is, isrows;
  Mat A_partitioned;
  Vec b_partitioned;
  VecScatter vec_scatter;
  int npe;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &npe));
  PetscInt num_proc = npe;
  if (num_proc != 1)
  {
      // Partition the matrix
      PetscCall(MatPartitioningCreate(PETSC_COMM_WORLD, &part));
      PetscCall(MatPartitioningSetAdjacency(part, A));
      PetscCall(MatPartitioningSetNParts(part, num_proc));
      PetscCall(MatPartitioningSetFromOptions(part));
      PetscCall(MatPartitioningApply(part, &is));
      PetscCall(ISBuildTwoSided(is, NULL, &isrows));
      PetscCall(MatCreateSubMatrix(A, isrows, isrows, MAT_INITIAL_MATRIX, &A_partitioned));
      PetscCall(MatCreateVecs(A_partitioned,NULL,&b_partitioned));

      // Scatter the b to match the partitioning
      PetscCall(VecScatterCreate(b, isrows, b_partitioned, NULL, &vec_scatter));
      PetscCall(VecScatterBegin(vec_scatter, b, b_partitioned, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(vec_scatter, b, b_partitioned, INSERT_VALUES, SCATTER_FORWARD));

      PetscCall(MatDestroy(&A));
      PetscCall(VecDestroy(&b));
      PetscCall(MatPartitioningDestroy(&part));
      PetscCall(VecScatterDestroy(&vec_scatter));
      PetscCall(ISDestroy(&is));
      PetscCall(ISDestroy(&isrows));
      A = A_partitioned;
      b = b_partitioned;
  }

  // ~~~~~~~~~~~~~~~~~~

  PetscCall(MatGetLocalSize(A,&m,&n));
  PetscCall(MatGetSize(A,&M,&N));

  // Test and see if the user wants us to use a different matrix type
  // with -mat_type on the command line
  // This lets us easily test our cpu and kokkos versions through our CI
  PetscCall(MatCreateFromOptions(PETSC_COMM_WORLD,NULL,\
               one,m,n,M,N,&A_diff_type));
  PetscCall(MatAssemblyBegin(A_diff_type, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A_diff_type, MAT_FINAL_ASSEMBLY));

  PetscCall(MatGetType(A, &mtype));
  PetscCall(MatGetType(A_diff_type, &mtype_input));

  if (mtype != mtype_input) 
  {
      // Doesn't seem like there is a converter to kokkos
      // So instead we just copy into the empty A_diff_type
      // This will be slow as its not preallocated, but this is just for testing
      PetscCall(MatCopy(A, A_diff_type, DIFFERENT_NONZERO_PATTERN));
      PetscCall(MatDestroy(&A));
      A = A_diff_type;

      // Mat and vec types have to match
      PetscCall(VecCreateFromOptions(PETSC_COMM_WORLD,NULL, \
               one,n,N,&b_diff_type));
      PetscCall(VecCopy(b,b_diff_type));
      PetscCall(VecDestroy(&b));
      b = b_diff_type;
  }
  else
  {
      PetscCall(MatDestroy(&A_diff_type));
  }   

  /*
   If the load matrix is larger then the vector, due to being padded
   to match the blocksize then create a new padded vector
  */
  {
    PetscInt    j,mvec,start,end,indx;
    Vec         tmp;
    PetscScalar *bold;

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

  // If we don't load b, just set x to random and the rhs to zero
  if (!b_in_f) {
    PetscCall(VecSetRandom(x,NULL));
    PetscCall(VecSet(b,0.0));
  }
  PetscCall(PetscBarrier((PetscObject)A));

  // Diagonally scale our matrix 
  if (diag_scale) {
   PetscCall(VecDuplicate(x, &diag_vec));
   PetscCall(MatGetDiagonal(A, diag_vec));
   PetscCall(VecReciprocal(diag_vec));
   PetscCall(MatDiagonalScale(A, diag_vec, PETSC_NULLPTR));
   PetscCall(VecPointwiseMult(b, diag_vec, b));
   PetscCall(VecDestroy(&diag_vec));
  }

  PetscCall(PetscLogStageRegister("mystage 1",&stage1));
  PetscCall(PetscLogStagePush(stage1));
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //       Let's use AIRG as our PC
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  

  PetscCall(KSPGetPC(ksp, &pc));
  // Register the pflare types
  PCRegister_PFLARE();
  PetscCall(PCSetType(pc, PCAIR));
  PetscCall(KSPSetPC(ksp, pc));

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPSetUpOnBlocks(ksp));
  PetscCall(PetscLogStagePop());
  PetscCall(PetscBarrier((PetscObject)A));

  PetscCall(PetscLogStageRegister("GPU copy stage - triggered by a prelim KSPSolve",&gpu_copy));
  PetscCall(PetscLogStagePush(gpu_copy));
  PetscCall(KSPSolve(ksp,b,x));
  PetscCall(PetscLogStagePop());

  if (second_solve)
  {
   PetscCall(VecSet(x, 1.0));
   PetscCall(PetscLogStageRegister("mystage 2",&stage2));
   PetscCall(PetscLogStagePush(stage2));
   PetscCall(KSPSolve(ksp,b,x));
   PetscCall(PetscLogStagePop());
  }

  PetscCall(KSPGetConvergedReason(ksp,&reason));

  /* Cleanup */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  if (reason < 0)
  {
   return 1;
  }
  return 0;
}