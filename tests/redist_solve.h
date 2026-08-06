#ifndef REDIST_SOLVE_H
#define REDIST_SOLVE_H

/* -----------------------------------------------------------------------
   Redistribute a linear system onto a subcommunicator and solve it there.

   Given a matrix assembled on `world` and an agglomeration factor k, the rows
   owned by each group of k consecutive world ranks are merged onto a single
   solve rank (world rank % k == 0), and the entire solve - outer KSP, MatMult
   and PC - runs on the resulting subcommunicator of size world_size/k. The
   right hand side is scattered in and the solution scattered back out, so the
   caller only ever sees vectors in the original fine layout.

   The typical use is a mesh/assembly phase that scales to many CPU ranks
   followed by a solve on fewer ranks (e.g. one per GPU).

   The matrix and vector movement follows PCTelescope's default mode, see
   PCTelescopeSetUp_default / PCTelescopeMatCreate_default in
   $PETSC_DIR/src/ksp/pc/impls/telescope/telescope.c. PCTelescope itself cannot
   be used here because it telescopes only the preconditioner while the outer
   KSP and MatMult stay on the fine communicator.

   Why the group's rows are one contiguous global range: a PETSc Mat row layout
   is always contiguous and monotone in rank order, so MatGetOwnershipRanges
   gives the exact prefix sum and rows [ranges[j*k], ranges[(j+1)*k]) are the
   union of group j's rows - whatever the individual per-rank row counts are.
   Nothing here assumes ranks own equal numbers of rows. What the mesh
   generator has to provide is not contiguity but that this contiguous block of
   rows is a compact sub-block of the domain, otherwise the merged partition is
   correct but poor. BoxMeshDM's GenerateBoxMeshDMAgglom guarantees exactly
   that for the k consecutive ranks of each group.

   Control flow invariant: every world-collective call (MatCreateSubMatrices,
   both scatters, the MPI_Bcast of the converged reason) is executed by all
   ranks. MatCreateMPIMatConcatenateSeqMat, MatCreateVecs on the redistributed
   operator, KSPCreate and KSPSolve are subcommunicator-collective only.
   ----------------------------------------------------------------------- */

#include <petscksp.h>

typedef struct {
  PetscInt      k;              /* agglomeration factor, > 1 */
  MPI_Comm      subcomm;        /* MPI_COMM_NULL on non-solve ranks */
  PetscBool     is_solve_rank;  /* world_rank % k == 0 */
  Mat           Bred;           /* redistributed operator (solve ranks only) */
  Vec           bred, xred;     /* subcomm vecs matching Bred */
  Vec           xtmp;           /* world staging vec, local size group_nrows or 0 */
  VecScatter    scatter;        /* fine layout <-> xtmp */
  KSP           ksp;            /* subcomm KSP (solve ranks only) */
  PetscInt      group_nrows;    /* rows merged onto this rank, 0 if not a solve rank */
  PetscLogStage redist_stage;   /* all redistribution work, no solve time */
} RedistCtx;

/* -----------------------------------------------------------------------
   Strip a leading "seq"/"mpi" from a matrix type name to get the root type,
   e.g. "mpiaijkokkos" -> "aijkokkos". Converting to the root type lets PETSc
   pick the seq or mpi implementation appropriate to the target communicator,
   which matters because MatCreateMPIMatConcatenateSeqMat returns a SeqAIJ on
   PETSC_COMM_SELF when the subcommunicator has size 1.
   ----------------------------------------------------------------------- */
static PetscErrorCode RedistRootMatType(Mat A, char root[], size_t len)
{
  MatType   type;
  PetscBool pre;

  PetscFunctionBeginUser;
  PetscCall(MatGetType(A, &type));
  PetscCall(PetscStrbeginswith(type, "seq", &pre));
  if (!pre) PetscCall(PetscStrbeginswith(type, "mpi", &pre));
  PetscCall(PetscStrncpy(root, pre ? type + 3 : type, len));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   Collective on world. Takes ownership of *A: the fine operator is destroyed
   as soon as the group's rows have been extracted from it, and *A is set to
   NULL on return. Destroying it before the merge is what keeps the peak
   memory at max(fine A + local rows, local rows + redistributed operator)
   rather than the sum of all three.
   ----------------------------------------------------------------------- */
static PetscErrorCode RedistSetup(MPI_Comm world, Mat *A, PetscInt k, RedistCtx *ctx)
{
  PetscMPIInt     size, rank;
  const PetscInt *ranges;
  PetscInt        j, group_rstart, N;
  IS              isrow, iscol;
  Mat             Blocal, *submats;
  Vec             xfine;
  VecType         vectype;
  char            root_type[256];

  PetscFunctionBeginUser;
  /* Zero first so a context from a failed setup is still safe to destroy */
  PetscCall(PetscMemzero(ctx, sizeof(*ctx)));
  ctx->subcomm = MPI_COMM_NULL;

  PetscCallMPI(MPI_Comm_size(world, &size));
  PetscCallMPI(MPI_Comm_rank(world, &rank));
  PetscCheck(k > 1, world, PETSC_ERR_ARG_OUTOFRANGE,
             "RedistSetup requires an agglomeration factor > 1, got %" PetscInt_FMT, k);
  PetscCheck(size % k == 0, world, PETSC_ERR_ARG_INCOMP,
             "Number of MPI ranks %d is not divisible by the agglomeration factor %" PetscInt_FMT, size, k);

  ctx->k             = k;
  ctx->is_solve_rank = (rank % k == 0) ? PETSC_TRUE : PETSC_FALSE;

  PetscCall(PetscLogStageRegister("Redistribution", &ctx->redist_stage));
  PetscCall(PetscLogStagePush(ctx->redist_stage));

  /* Key rank/k makes subcomm rank j own coarse group j, so the merged matrix
     keeps the fine global row numbering exactly */
  PetscCallMPI(MPI_Comm_split(world, ctx->is_solve_rank ? 0 : MPI_UNDEFINED, rank / k, &ctx->subcomm));

  /* The group's row range, no communication - the ownership ranges array is a
     prefix sum already replicated on every rank */
  PetscCall(MatGetOwnershipRanges(*A, &ranges));
  j            = rank / k;
  group_rstart = ranges[j * k];
  if (ctx->is_solve_rank) {
    ctx->group_nrows = ranges[j * k + k] - group_rstart;
  } else {
    /* Zero length, but starting at our own rows so the scatter below is a
       no-op for us rather than a duplicate of someone else's rows */
    PetscCall(MatGetOwnershipRange(*A, &group_rstart, NULL));
    ctx->group_nrows = 0;
  }
  PetscCall(ISCreateStride(world, ctx->group_nrows, group_rstart, 1, &isrow));

  /* The root type is needed after the fine operator has been destroyed, and
     MatGetType hands back a pointer the matrix owns, so copy it out now */
  PetscCall(MatGetSize(*A, NULL, &N));
  PetscCall(RedistRootMatType(*A, root_type, sizeof(root_type)));

  /* Staging vector on world, holding the group's rows on the solve rank, and
     the scatter between the fine layout and it. VecScatterCreate builds its
     own PetscSF from the indices and does not hold on to the IS. */
  PetscCall(MatGetVecType(*A, &vectype));
  PetscCall(VecCreate(world, &ctx->xtmp));
  PetscCall(VecSetSizes(ctx->xtmp, ctx->group_nrows, PETSC_DECIDE));
  PetscCall(VecSetType(ctx->xtmp, vectype));
  PetscCall(MatCreateVecs(*A, &xfine, NULL));
  PetscCall(VecScatterCreate(xfine, isrow, ctx->xtmp, NULL, &ctx->scatter));
  PetscCall(VecDestroy(&xfine));

  /* Pull out the group's rows. ISSetIdentity and MAT_SUBMAT_SINGLEIS are
     load bearing: they trigger the allcolumns fast path. Without them every
     rank builds an O(N) column map, which is multiple GB at 1e8 rows. */
  PetscCall(ISCreateStride(PETSC_COMM_SELF, N, 0, 1, &iscol));
  PetscCall(ISSetIdentity(iscol));
  PetscCall(MatSetOption(*A, MAT_SUBMAT_SINGLEIS, PETSC_TRUE));
  PetscCall(MatCreateSubMatrices(*A, 1, &isrow, &iscol, MAT_INITIAL_MATRIX, &submats));
  Blocal = submats[0];
  PetscCall(PetscFree(submats));
  PetscCall(ISDestroy(&iscol));
  PetscCall(ISDestroy(&isrow));

  /* The fine operator is dead from here on */
  PetscCall(MatDestroy(A));

  /* Merge the groups' rows onto the solve ranks. n is the local column count;
     the operator is square so it equals group_nrows, and these sum to N over
     the subcommunicator. A MAT_REUSE_MATRIX re-move telescope-style is
     possible but composes with neither the in-place conversion below nor
     destroying the fine operator - and it is not needed, since a repeat solve
     reuses the operator untouched. */
  if (ctx->is_solve_rank) {
    PetscCall(MatCreateMPIMatConcatenateSeqMat(ctx->subcomm, Blocal, ctx->group_nrows, MAT_INITIAL_MATRIX, &ctx->Bred));
  }
  PetscCall(MatDestroy(&Blocal));

  if (ctx->is_solve_rank) {
    /* Give the redistributed operator the same type the fine one had. Going
       through the root type is a no-op when the merge already produced the
       right type, including the size 1 subcommunicator case where it comes
       back as a SeqAIJ on PETSC_COMM_SELF. */
    PetscCall(MatConvert(ctx->Bred, root_type, MAT_INPLACE_MATRIX, &ctx->Bred));
    PetscCall(MatCreateVecs(ctx->Bred, &ctx->xred, &ctx->bred));

    /* No options prefix, so -ksp_type, -pc_type and all the PFLARE options
       apply to this KSP unchanged */
    PetscCall(KSPCreate(ctx->subcomm, &ctx->ksp));
    PetscCall(KSPSetOperators(ctx->ksp, ctx->Bred, ctx->Bred));
    PetscCall(KSPSetFromOptions(ctx->ksp));
  }

  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   Copy the local part of one vector into another of the same local size but
   living on a different communicator. Not VecPlaceArray: the vectors may be
   on a device, and VecGetArrayWrite handles the coherence.
   ----------------------------------------------------------------------- */
static PetscErrorCode RedistCopyLocal(Vec from, Vec to, PetscInt n)
{
  const PetscScalar *src;
  PetscScalar       *dst;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(from, &src));
  PetscCall(VecGetArrayWrite(to, &dst));
  PetscCall(PetscArraycpy(dst, src, n));
  PetscCall(VecRestoreArrayWrite(to, &dst));
  PetscCall(VecRestoreArrayRead(from, &src));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   Collective on world. b_fine and x_fine are in the original fine layout;
   on return x_fine holds the solution on every rank, including the ranks
   that took no part in the solve.

   The log stage covers only the data movement - the KSPSolve is left in the
   caller's ambient stage so solve time is not attributed to redistribution.
   ----------------------------------------------------------------------- */
static PetscErrorCode RedistSolve(RedistCtx *ctx, Vec b_fine, Vec x_fine, PetscBool nonzero_guess, KSPConvergedReason *reason)
{
  PetscMPIInt ireason = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscLogStagePush(ctx->redist_stage));

  PetscCall(VecScatterBegin(ctx->scatter, b_fine, ctx->xtmp, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx->scatter, b_fine, ctx->xtmp, INSERT_VALUES, SCATTER_FORWARD));
  if (ctx->is_solve_rank) PetscCall(RedistCopyLocal(ctx->xtmp, ctx->bred, ctx->group_nrows));

  if (nonzero_guess) {
    PetscCall(VecScatterBegin(ctx->scatter, x_fine, ctx->xtmp, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ctx->scatter, x_fine, ctx->xtmp, INSERT_VALUES, SCATTER_FORWARD));
    if (ctx->is_solve_rank) PetscCall(RedistCopyLocal(ctx->xtmp, ctx->xred, ctx->group_nrows));
  } else if (ctx->is_solve_rank) {
    PetscCall(VecSet(ctx->xred, 0.0));
  }
  if (ctx->is_solve_rank) PetscCall(KSPSetInitialGuessNonzero(ctx->ksp, nonzero_guess));

  PetscCall(PetscLogStagePop());

  if (ctx->is_solve_rank) {
    PetscCall(KSPSolve(ctx->ksp, ctx->bred, ctx->xred));
    PetscCall(KSPGetConvergedReason(ctx->ksp, reason));
    PetscCall(PetscMPIIntCast((PetscInt)*reason, &ireason));
  }

  PetscCall(PetscLogStagePush(ctx->redist_stage));

  /* World rank 0 is always a solve rank. Broadcasting the reason lets every
     rank react to a failed solve identically, and is the rendezvous point for
     the ranks that did not take part in it. */
  PetscCallMPI(MPI_Bcast(&ireason, 1, MPI_INT, 0, PetscObjectComm((PetscObject)ctx->xtmp)));
  *reason = (KSPConvergedReason)ireason;

  if (ctx->is_solve_rank) PetscCall(RedistCopyLocal(ctx->xred, ctx->xtmp, ctx->group_nrows));
  PetscCall(VecScatterBegin(ctx->scatter, ctx->xtmp, x_fine, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(ctx->scatter, ctx->xtmp, x_fine, INSERT_VALUES, SCATTER_REVERSE));

  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* NULL safe, and safe on a zeroed context that was never set up */
static PetscErrorCode RedistDestroy(RedistCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(KSPDestroy(&ctx->ksp));
  PetscCall(MatDestroy(&ctx->Bred));
  PetscCall(VecDestroy(&ctx->bred));
  PetscCall(VecDestroy(&ctx->xred));
  PetscCall(VecDestroy(&ctx->xtmp));
  PetscCall(VecScatterDestroy(&ctx->scatter));
  if (ctx->subcomm != MPI_COMM_NULL) PetscCallMPI(MPI_Comm_free(&ctx->subcomm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif /* REDIST_SOLVE_H */
