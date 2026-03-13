static char help[] = "Checks we can read in a linear system and compute a CF splitting.\n\
Input arguments are:\n\
  -f <input_file> : file to load. For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\n";

#include <petscksp.h>
#include <petsclog.h>
#include "pflare.h"

// Check that a given CF splitting is valid
static PetscErrorCode CheckSplitting(Mat A, IS is_fine, IS is_coarse, const char *label)
{
  PetscInt        n_fine_local, n_coarse_local;
  PetscInt        n_fine_global, n_coarse_global;
  PetscInt        local_rows, local_cols;
  PetscInt        global_rows, global_cols;
  PetscInt        rstart, rend, i;
  PetscInt        nlocal;
  const PetscInt  *idx_fine, *idx_coarse;
  PetscInt        *seen;
  PetscMPIInt     rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCall(MatGetLocalSize(A, &local_rows, &local_cols));
  PetscCall(MatGetSize(A, &global_rows, &global_cols));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));

  PetscCall(ISGetLocalSize(is_fine, &n_fine_local));
  PetscCall(ISGetLocalSize(is_coarse, &n_coarse_local));
  PetscCall(ISGetSize(is_fine, &n_fine_global));
  PetscCall(ISGetSize(is_coarse, &n_coarse_global));

  PetscCheck(n_fine_local + n_coarse_local == local_rows, PETSC_COMM_SELF, PETSC_ERR_PLIB,
             "%s: local size mismatch on rank %d: n_fine_local (%" PetscInt_FMT ") + n_coarse_local (%" PetscInt_FMT ") != local_rows (%" PetscInt_FMT ")",
             label, rank, n_fine_local, n_coarse_local, local_rows);
  PetscCheck(n_fine_global + n_coarse_global == global_rows, PETSC_COMM_SELF, PETSC_ERR_PLIB,
             "%s: global size mismatch: n_fine_global (%" PetscInt_FMT ") + n_coarse_global (%" PetscInt_FMT ") != global_rows (%" PetscInt_FMT ")",
             label, n_fine_global, n_coarse_global, global_rows);

  PetscCall(PetscCalloc1(local_rows, &seen));

  PetscCall(ISGetLocalSize(is_fine, &nlocal));
  PetscCall(ISGetIndices(is_fine, &idx_fine));
  for (i = 0; i < nlocal; i++) {
    PetscCheck(idx_fine[i] >= rstart && idx_fine[i] < rend, PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "%s: fine index %" PetscInt_FMT " is not in local row ownership range [%" PetscInt_FMT ", %" PetscInt_FMT ") on rank %d",
               label, idx_fine[i], rstart, rend, rank);
    seen[idx_fine[i] - rstart]++;
  }
  PetscCall(ISRestoreIndices(is_fine, &idx_fine));

  PetscCall(ISGetLocalSize(is_coarse, &nlocal));
  PetscCall(ISGetIndices(is_coarse, &idx_coarse));
  for (i = 0; i < nlocal; i++) {
    PetscCheck(idx_coarse[i] >= rstart && idx_coarse[i] < rend, PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "%s: coarse index %" PetscInt_FMT " is not in local row ownership range [%" PetscInt_FMT ", %" PetscInt_FMT ") on rank %d",
               label, idx_coarse[i], rstart, rend, rank);
    seen[idx_coarse[i] - rstart]++;
  }
  PetscCall(ISRestoreIndices(is_coarse, &idx_coarse));

  for (i = 0; i < local_rows; i++) {
    PetscCheck(seen[i] == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "%s: owned row %" PetscInt_FMT " appears %" PetscInt_FMT " times in local fine/coarse IS on rank %d",
               label, rstart + i, seen[i], rank);
  }

  PetscCall(PetscFree(seen));

  if (!rank) PetscCall(PetscPrintf(PETSC_COMM_SELF, "%s: OK\n", label));

  PetscFunctionReturn(PETSC_SUCCESS);
}

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

 compute_cf_splitting(A, \
     symmetric, \
     strong_threshold, max_luby_steps, \
     algorithm, \
     ddc_its, \
     ddc_fraction, \
     max_dd_ratio, \
     &is_fine, &is_coarse);

  PetscCall(CheckSplitting(A, is_fine, is_coarse, "default max_dd_ratio"));

  PetscCall(ISDestroy(&is_fine));
  PetscCall(ISDestroy(&is_coarse));  

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //       Compute a cf splitting with max_dd_ratio set
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  

 // This will keep doing ddc until this diagonal dominance
 // ratio is hit
 max_dd_ratio = 0.5;

 compute_cf_splitting(A, \
     symmetric, \
     strong_threshold, max_luby_steps, \
     algorithm, \
     ddc_its, \
     ddc_fraction, \
     max_dd_ratio, \
     &is_fine, &is_coarse);

  PetscCall(CheckSplitting(A, is_fine, is_coarse, "max_dd_ratio=0.5"));

  PetscCall(ISDestroy(&is_fine));
  PetscCall(ISDestroy(&is_coarse));  

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  

  /* Cleanup */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(MatDestroy(&A));

  PetscCall(PetscFinalize());
  return 0;
}