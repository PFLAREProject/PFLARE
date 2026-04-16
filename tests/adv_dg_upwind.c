/*  DMPlex/KSP solving a scalar advection equation with upwinded DG FEM.
    Pure advection only (no diffusion) on a 2D or 3D mesh.
    Default is 2D with default velocity (1,1) normalised.
    In 3D default velocity is (1,1,1) normalised.    
    The function space is a broken polynomial space: each cell owns its own
    DOFs with no inter-element continuity enforced.  The upwind numerical
    flux on interior facets and inflow boundary facets couples the cells.

    Usage mirrors the SUPG CG code:
      Can control dimension with         -dm_plex_dim
      Can control simplex/quad with      -dm_plex_simplex
      Can control face count with        -dm_plex_box_faces
      Can refine with                    -dm_refine
      Can read a gmsh file with          -dm_plex_filename
      Can specify basis function order   -adv_diff_petscspace_degree
      Specify inflow of 1 on bottom face -bottom_only_inflow_one
      Can write out vtk solution with    -write_vtk
      Time dependent solve               -time_depend

      The time dependent solve does backward-Euler TS integration
        instead of the steady KSP solve
        Integrates   M du/dt + A_stiff u = b_stiff
        rewritten as du/dt + (M^-1 A_stiff) u = M^-1 b_stiff
        using a PETSc TS with backward Euler.  Swaps the
        diag_scale preconditioner from the stiffness block-
        diagonal to the DG mass-matrix block-diagonal and skips the
        steady KSPSolve entirely.  Requires -diag_scale true.
        Should run with -bottom_only_inflow_one otherwise 
        the solution is zero at all time.
        Timestep / step count / final time / inner solver are
        all controlled by the standard PETSc TS options:
          -ts_dt <dt>            -ts_max_steps <n>
          -ts_max_time <T>       -ts_type <type>
          -ts_monitor            -ts_view
          -ts_ksp_... / -ts_snes_...
        When -write_vtk is also set, one VTU is written per
        TS step (basename dg_solution_ts_XXXX).

    ./adv_dg_upwind -dm_refine 2
            : pure advection in 2D with linear FEM with u = v = 1, normalised (theta=pi/4)
              BCs left and bottom 0 inflow dirichlet, the others outflow

    Boundary label conventions (matching the SUPG code):
      2D: Dirichlet inflow faces = {1, 4}  (bottom/y=0, left/x=0)
          Neumann outflow faces  = {2, 3}  (top/y=1, right/x=1)
      3D: Dirichlet inflow faces = {1, 3, 6}  (bottom/z=0, front/y=0, left/x=0)
          Neumann outflow faces  = {2, 4, 5}  (top/z=1, back/y=1, right/x=1)

    Note: unlike the SUPG code, actual inflow/outflow at each quadrature point
    is always determined physically by the sign of b.n (the dot product of
    the advection velocity with the outward face normal).  The Face Sets IDs
    above only control which boundary faces are candidates for applying
    Dirichlet inflow data via InflowValue().  On faces labelled as outflow
    no Dirichlet value is imposed even if b.n < 0 locally.

    Design notes
    ============
    We deliberately avoid the DS/SNES callback machinery.  Instead:

      1.  A broken PetscSection is built manually: each cell owns Nb DOFs,
          no DOFs live on facets/edges/vertices.

      2.  The global matrix sparsity is preallocated by walking interior
          facets and recording which cell pairs are coupled.

      3.  Volume integrals and facet flux integrals are assembled explicitly
          with MatSetValues / VecSetValues, using PetscFECreateTabulation to
          obtain basis values and derivatives at quadrature points.

      4.  A standard KSP solve is performed on the assembled system.

    Weak form (per cell K, outward normal n_K):
    -------------------------------------------
    Find u_h in V_h (broken polynomial space) such that for all v_h in V_h:

      -  integral_K  u_h (b . grad v_h) dx          [volume, integrated by parts]
      +  sum_{F in dK} integral_F  Fhat(u+,u-,n) v_h^+  ds  =  integral_K f v_h dx

    where the upwind flux is:
      Fhat = (b.n)^+ u^+  +  (b.n)^- u^-
           = max(b.n, 0) u^+  +  min(b.n, 0) u^-

    u^+ is the trace from the current cell (the "inside" cell for this facet),
    u^- is the trace from the neighbour cell (or the boundary value).

    Note: the volume term uses IBP to move the derivative onto the test
    function, which is the standard DG approach that avoids requiring
    derivatives of u on the facets for the upwind flux.
*/

static char help[] = "Solves steady advection with upwinded DG FEM.\n\n";

#include <petscdmplex.h>
#include <petscksp.h>
#include <petscts.h>
#include <petscds.h>
#include <petscfe.h>
#include <petscdt.h>
#include <petscblaslapack.h>
#include <math.h>

#include "pflare.h"

/* -----------------------------------------------------------------------
   Application context
   ----------------------------------------------------------------------- */
typedef struct {
  PetscReal advection_velocity[3];
  PetscBool curved_velocity;
  PetscBool unit_velocity;
  PetscBool bottom_only_inflow_one;
  PetscBool diag_scale;
  PetscBool second_solve;
  PetscBool time_depend;
  PetscBool write_vtk;
  PetscBool verify_solution;
} AppCtx;

/* -----------------------------------------------------------------------
   Velocity evaluation — identical logic to the SUPG code
   ----------------------------------------------------------------------- */
static inline void GetVelocity(PetscInt dim, const AppCtx *ctx, const PetscReal x[], PetscReal v[])
{
  if (ctx->curved_velocity) {
    v[0] = x[1];
    v[1] = 1.0 - x[0];
    v[2] = 0.0;
  } else {
    v[0] = ctx->advection_velocity[0];
    v[1] = ctx->advection_velocity[1];
    v[2] = ctx->advection_velocity[2];
  }
  if (ctx->unit_velocity) {
    PetscReal mag = 0.0;
    for (PetscInt d = 0; d < dim; d++) mag += v[d] * v[d];
    mag = PetscSqrtReal(mag);
    if (mag > 1e-12) { for (PetscInt d = 0; d < dim; d++) v[d] /= mag; }
  }
}

/* Inflow Dirichlet value by boundary face id.
   Default: homogeneous (0) inflow.
   With -bottom_only_inflow_one: set g=1 on face set 1, g=0 elsewhere. */
static inline PetscScalar InflowValue(const AppCtx *ctx, PetscInt faceId,
                                      const PetscReal x[])
{
  (void)x;
  if (ctx->bottom_only_inflow_one) return (faceId == 1) ? 1.0 : 0.0;
  return 0.0;
}

/* -----------------------------------------------------------------------
   Option processing
   ----------------------------------------------------------------------- */
static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *opt)
{
  PetscFunctionBeginUser;
  PetscOptionsBegin(comm, "", "DG Advection Options", "DMPLEX");

  opt->curved_velocity = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-curved_velocity", &opt->curved_velocity, NULL));

  opt->unit_velocity = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-unit_velocity", &opt->unit_velocity, NULL));

  opt->bottom_only_inflow_one = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-bottom_only_inflow_one", &opt->bottom_only_inflow_one, NULL));

  opt->diag_scale = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-diag_scale", &opt->diag_scale, NULL));

  opt->second_solve = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-second_solve", &opt->second_solve, NULL));

  opt->time_depend = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-time_depend", &opt->time_depend, NULL));

  opt->write_vtk = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-write_vtk", &opt->write_vtk, NULL));

  // Initialize advection to the diagonal direction by default
  opt->advection_velocity[0] = 1.0; // u
  opt->advection_velocity[1] = 1.0; // v
  opt->advection_velocity[2] = 1.0; // w (for 3D)  

  // User can specify direction with theta, or by giving the velocity components directly
  PetscReal pi = 4.0 * atan(1.0), theta = pi / 4.0;
  PetscBool theta_found;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-theta", &theta, &theta_found));
  if (theta_found)
  {
      PetscCheck(theta >= 0.0 && theta <= pi / 2.0, comm, PETSC_ERR_ARG_WRONGSTATE,
                  "theta must be in [0, pi/2]");
      opt->advection_velocity[0] = PetscCosReal(theta);
      opt->advection_velocity[1] = PetscSinReal(theta);
      opt->advection_velocity[2] = 0.0; // w component is 0 with theta
  }

  PetscReal uv = 0.0, vv = 0.0, wv = 0.0;
  PetscBool fu, fv, fw;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-u", &uv, &fu));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-v", &vv, &fv));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-w", &wv, &fw));
  if (fu) opt->advection_velocity[0] = uv;
  if (fv) opt->advection_velocity[1] = vv;
  if (fw) opt->advection_velocity[2] = wv;
  if (fu || fv || fw) opt->unit_velocity = PETSC_FALSE;

  opt->verify_solution = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-verify_solution", &opt->verify_solution, NULL));
  if (opt->verify_solution) {
    opt->bottom_only_inflow_one = PETSC_TRUE;
    opt->unit_velocity          = PETSC_FALSE;
  }

  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   Mesh creation — identical to the SUPG code
   ----------------------------------------------------------------------- */
static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *opt, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  /* DG requires a 1-cell overlap so that each rank sees the neighbour cells
     across partition boundaries.  Set this as a default; the user can still
     override with -dm_distribute_overlap <n> on the command line.         */
  {
    PetscBool overlap_set;
    PetscCall(PetscOptionsHasName(NULL, NULL, "-dm_distribute_overlap", &overlap_set));
    if (!overlap_set)
      PetscCall(PetscOptionsSetValue(NULL, "-dm_distribute_overlap", "1"));
  }
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMGetCoordinatesLocalSetUp(*dm));
  PetscCall(DMSetApplicationContext(*dm, opt));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   Build a broken PetscSection: each cell owns Nb DOFs, nothing else does.

   A "broken" space means DOFs are not shared between cells.  This is the
   correct layout for a DG space.  We visit every point in the DMPlex DAG
   and assign Nb DOFs to cells (height == 0 in the cell-to-face ordering,
   i.e. depth == dim) and 0 DOFs to everything else.
   ----------------------------------------------------------------------- */
static PetscErrorCode BuildBrokenSection(DM dm, PetscInt Nb, PetscSection *sec)
{
  PetscInt pStart, pEnd, cStart, cEnd;

  PetscFunctionBeginUser;
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd)); /* cells */

  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), sec));
  PetscCall(PetscSectionSetNumFields(*sec, 1));
  PetscCall(PetscSectionSetFieldComponents(*sec, 0, 1));
  PetscCall(PetscSectionSetChart(*sec, pStart, pEnd));

  for (PetscInt c = cStart; c < cEnd; c++) {
    PetscCall(PetscSectionSetDof(*sec, c, Nb));
    PetscCall(PetscSectionSetFieldDof(*sec, c, 0, Nb));
  }
  PetscCall(PetscSectionSetUp(*sec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   Preallocate the global matrix.

   Each cell has a diagonal block of size Nb x Nb (from the volume term).
   Each interior facet contributes an off-diagonal block in exactly one
   direction (the downwind cell's rows need the upwind cell's columns).
   Within that off-diagonal block, only DOFs whose basis functions are
   non-zero on the shared face participate: for the downwind cell these
   are the face-active test DOFs (rows), and for the upwind cell these
   are the face-active trial DOFs (columns).  The face tabulation from
   PetscFE is used to identify them exactly.

   For the parallel case we must distinguish on-process (d_nnz) from
   off-process (o_nnz) column entries.  For DG this is straightforward:
   a neighbour cell is off-process if and only if its global DOF offset
   lies outside the local ownership range [rstart, rend).  We obtain the
   global offsets via the global PetscSection (obtained through
   DMGetGlobalSection after DMSetLocalSection has been called).

   Ghost cells (halo cells that are owned by another rank but appear in
   our local numbering) have a negative offset in the global section —
   PETSc convention is that ghost points have offset -(global_offset+1).
   We decode this to get the true global column index and compare it
   against [rstart, rend) to decide d_nnz vs o_nnz.
   ----------------------------------------------------------------------- */
/* Forward declaration — FacetOutwardNormal is defined after this function */
static PetscErrorCode FacetOutwardNormal(DM dm, PetscInt f, PetscInt c,
                                          PetscInt dim, PetscReal n_out[]);

static PetscErrorCode PreallocateMatrix(DM dm, PetscFE fe, PetscInt Nb, const AppCtx *ctx, Mat A)
{
  PetscInt        fStart, fEnd, cStart, cEnd;
  PetscInt       *d_nnz, *o_nnz;
  PetscInt        nlocal, rstart, rend;
  PetscSection    gsec; /* global section — gives global DOF offsets */
  PetscInt        dim;
  PetscTabulation Tf;
  PetscQuadrature fquad;
  PetscInt        Nfq;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));

  PetscCall(MatGetLocalSize(A, &nlocal, NULL));
  PetscCall(PetscCalloc1(nlocal, &d_nnz));
  PetscCall(PetscCalloc1(nlocal, &o_nnz));

  /* The global section maps each local DAG point to its global DOF offset.
     Ghost points carry a negative encoded offset: goff = -(global+1).    */
  PetscCall(DMGetGlobalSection(dm, &gsec));

  /* Local ownership range in global DOF indices */
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));

  /* Face tabulation: Tf->T[0][(fLocal * Nfq + q) * Nb + i] is the value
     of basis function i at the q-th quadrature point on local face fLocal.
     A basis function is zero on a face iff all its values there are 0.0.  */
  PetscCall(PetscFEGetFaceTabulation(fe, 0, &Tf));
  PetscCall(PetscFEGetFaceQuadrature(fe, &fquad));
  PetscCall(PetscQuadratureGetData(fquad, NULL, NULL, &Nfq, NULL, NULL));

  /* Helper lambda (C99 inline logic): given a DAG point p, return its
     global DOF base index (always non-negative).                          */
#define GlobalOffset(p, goff_out)                                    \
  do {                                                               \
    PetscInt _g;                                                     \
    PetscCall(PetscSectionGetOffset(gsec, (p), &_g));                \
    (goff_out) = (_g >= 0) ? _g : -(_g + 1);                        \
  } while (0)

  /* Every locally-owned cell contributes a diagonal block.
     Ghost cells (negative global offset) are skipped — their diagonal
     blocks are owned and assembled by the rank that owns them.            */
  for (PetscInt c = cStart; c < cEnd; c++) {
    PetscInt goff;
    GlobalOffset(c, goff);
    if (goff < rstart || goff >= rend) continue; /* ghost cell */
    for (PetscInt i = 0; i < Nb; i++) d_nnz[goff - rstart + i] += Nb;
  }

  /* Interior facets: both K_LR (supp[0] rows, supp[1] cols) and K_RL
     (supp[1] rows, supp[0] cols) can be non-zero for the same face when
     the velocity field is curved or when a non-planar face has a varying
     normal — both b·n > 0 and b·n < 0 can occur at different quadrature
     points.  We therefore always preallocate both off-diagonal blocks.
     Only the face-active DOFs (those whose basis functions are non-zero
     on the shared face) participate; this is read from the face tabulation
     and gives a much tighter allocation than a full Nb x Nb block.         */
  for (PetscInt f = fStart; f < fEnd; f++) {
    const PetscInt *supp;
    PetscInt        suppSize;
    PetscCall(DMPlexGetSupportSize(dm, f, &suppSize));
    PetscCall(DMPlexGetSupport(dm, f, &supp));
    if (suppSize != 2) continue; /* boundary facet — no inter-cell coupling */

    PetscInt goff0, goff1;
    GlobalOffset(supp[0], goff0);
    GlobalOffset(supp[1], goff1);

    PetscBool own0 = (goff0 >= rstart && goff0 < rend) ? PETSC_TRUE : PETSC_FALSE;
    PetscBool own1 = (goff1 >= rstart && goff1 < rend) ? PETSC_TRUE : PETSC_FALSE;
    if (!own0 && !own1) continue;

    /* Find local face indices in each cell's cone */
    PetscInt        fL = -1, fR = -1;
    const PetscInt *coneL, *coneR;
    PetscInt        coneSizeL, coneSizeR;
    PetscCall(DMPlexGetConeSize(dm, supp[0], &coneSizeL));
    PetscCall(DMPlexGetCone(dm, supp[0], &coneL));
    for (PetscInt lf = 0; lf < coneSizeL; lf++) {
      if (coneL[lf] == f) { fL = lf; break; }
    }
    PetscCall(DMPlexGetConeSize(dm, supp[1], &coneSizeR));
    PetscCall(DMPlexGetCone(dm, supp[1], &coneR));
    for (PetscInt lf = 0; lf < coneSizeR; lf++) {
      if (coneR[lf] == f) { fR = lf; break; }
    }

    /* Count face-active DOFs for each cell */
    PetscInt nactiveL = 0, nactiveR = 0;
    for (PetscInt j = 0; j < Nb; j++) {
      for (PetscInt q = 0; q < Nfq; q++) {
        if (Tf->T[0][(fL * Nfq + q) * Nb + j] != 0.0) { nactiveL++; break; }
      }
      for (PetscInt q = 0; q < Nfq; q++) {
        if (Tf->T[0][(fR * Nfq + q) * Nb + j] != 0.0) { nactiveR++; break; }
      }
    }

    /* K_LR: supp[0] (active rows) needs supp[1] (active cols) */
    if (own0) {
      for (PetscInt i = 0; i < Nb; i++) {
        PetscBool active_i = PETSC_FALSE;
        for (PetscInt q = 0; q < Nfq; q++) {
          if (Tf->T[0][(fL * Nfq + q) * Nb + i] != 0.0) { active_i = PETSC_TRUE; break; }
        }
        if (!active_i) continue;
        if (own1) d_nnz[goff0 - rstart + i] += nactiveR;
        else      o_nnz[goff0 - rstart + i] += nactiveR;
      }
    }

    /* K_RL: supp[1] (active rows) needs supp[0] (active cols) */
    if (own1) {
      for (PetscInt i = 0; i < Nb; i++) {
        PetscBool active_i = PETSC_FALSE;
        for (PetscInt q = 0; q < Nfq; q++) {
          if (Tf->T[0][(fR * Nfq + q) * Nb + i] != 0.0) { active_i = PETSC_TRUE; break; }
        }
        if (!active_i) continue;
        if (own0) d_nnz[goff1 - rstart + i] += nactiveL;
        else      o_nnz[goff1 - rstart + i] += nactiveL;
      }
    }
  }
#undef GlobalOffset

  PetscCall(MatSeqAIJSetPreallocation(A, 0, d_nnz));
  PetscCall(MatMPIAIJSetPreallocation(A, 0, d_nnz, 0, o_nnz));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(MatSetOption(A,MAT_IGNORE_ZERO_ENTRIES ,PETSC_TRUE));

  PetscCall(PetscFree(d_nnz));
  PetscCall(PetscFree(o_nnz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   Utility: get the outward normal of facet f as seen from cell c.

   DMPlexComputeCellGeometryFVM on the facet gives the area-weighted normal
   pointing in an arbitrary orientation.  We check which side of the facet
   the cell centre lies on and flip if necessary so the normal points away
   from cell c.
   ----------------------------------------------------------------------- */
static PetscErrorCode FacetOutwardNormal(DM dm, PetscInt f, PetscInt c,
                                          PetscInt dim, PetscReal n_out[])
{
  PetscReal area;

  PetscFunctionBeginUser;
  if (dim == 2) {
    DM              cdm;
    PetscSection    csec;
    Vec             coords;
    const PetscInt *cone;
    PetscInt        coneSize;
    PetscReal       x0[2], x1[2], fc[2], cc[2], tx, ty, nx, ny, nlen, dot;

    PetscCall(DMPlexGetConeSize(dm, f, &coneSize));
    PetscCheck(coneSize == 2, PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "Expected edge facet with 2 vertices, got %" PetscInt_FMT, coneSize);
    PetscCall(DMPlexGetCone(dm, f, &cone));

    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMGetLocalSection(cdm, &csec));
    PetscCall(DMGetCoordinatesLocal(dm, &coords));
    for (PetscInt iv = 0; iv < 2; iv++) {
      PetscInt    cs;
      PetscScalar *cv = NULL;
      PetscCall(DMPlexVecGetClosure(cdm, csec, coords, cone[iv], &cs, &cv));
      PetscCheck(cs >= 2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Vertex coord closure too small");
      if (iv == 0) { x0[0] = PetscRealPart(cv[0]); x0[1] = PetscRealPart(cv[1]); }
      else         { x1[0] = PetscRealPart(cv[0]); x1[1] = PetscRealPart(cv[1]); }
      PetscCall(DMPlexVecRestoreClosure(cdm, csec, coords, cone[iv], &cs, &cv));
    }

    tx = x1[0] - x0[0];
    ty = x1[1] - x0[1];
    nx =  ty;
    ny = -tx;
    nlen = PetscSqrtReal(nx * nx + ny * ny);
    PetscCheck(nlen > 0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Degenerate edge normal");
    nx /= nlen;
    ny /= nlen;

    fc[0] = 0.5 * (x0[0] + x1[0]);
    fc[1] = 0.5 * (x0[1] + x1[1]);
    PetscCall(DMPlexComputeCellGeometryFVM(dm, c, NULL, cc, NULL));
    dot = nx * (cc[0] - fc[0]) + ny * (cc[1] - fc[1]);
    if (dot > 0.0) { nx = -nx; ny = -ny; }

    n_out[0] = nx;
    n_out[1] = ny;
    n_out[2] = 0.0;
  } else {
    /* 3D: compute face normal via FVM geometry and orient outward from cell c */
    PetscReal rawNormal[3], fc[3], cc[3];
    PetscCall(DMPlexComputeCellGeometryFVM(dm, f, &area, fc, rawNormal));
    /* rawNormal from DMPlexComputeCellGeometryFVM is already a unit normal */
    for (PetscInt d = 0; d < dim; d++) n_out[d] = rawNormal[d];
    /* Orient so that n_out points away from cell c */
    PetscCall(DMPlexComputeCellGeometryFVM(dm, c, NULL, cc, NULL));
    PetscReal dot = 0.0;
    for (PetscInt d = 0; d < dim; d++) dot += n_out[d] * (cc[d] - fc[d]);
    if (dot > 0.0) {
      for (PetscInt d = 0; d < dim; d++) n_out[d] = -n_out[d];
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   Volume integral assembly
   -------------------------
   For cell c, compute the local Nb x Nb matrix A_vol and Nb RHS vector
   from the volume term of the DG weak form:

     A_vol[i][j] = -integral_K  phi_j  (b . grad psi_i)  dx
                 +  integral_K  f  psi_i  dx  (goes to RHS, here f=0)

   where phi_j are trial basis functions and psi_i are test basis functions
   (same space in Galerkin).  The minus sign comes from IBP:

     integral_K  (b . grad u_h) v_h  dx
       = -integral_K  u_h  (b . grad v_h)  dx
         + integral_{dK}  (b.n) u_h v_h  ds

   The boundary integral is handled by the facet assembly, so here we only
   keep the volume (interior) part with the sign flipped.

   Inputs from PetscFECreateTabulation:
     T->T[0]  : basis values,      shape [1][Nq][Nb][Nc]  (Nc=1 for scalar)
     T->T[1]  : basis gradients,   shape [1][Nq][Nb][dim]

   The physical-space gradient is obtained by applying the inverse Jacobian
   of the reference-to-physical map, which DMPlexComputeCellGeometryFEM
   provides as invJ.
   ----------------------------------------------------------------------- */
static PetscErrorCode AssembleVolumeCell(DM dm, PetscFE fe, PetscInt c,
                                          const AppCtx *ctx,
                                          PetscInt Nb, PetscInt Nq,
                                          PetscScalar *A_loc,   /* Nb*Nb */
                                          PetscScalar *b_loc)   /* Nb    */
{
  PetscTabulation T;
  PetscQuadrature quad;
  const PetscReal *qpoints, *qweights;
  PetscInt         Nqp, dim;
  PetscReal        v0[3], J[9], invJ[9], detJ;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));

  /* Cell geometry: Jacobian, inverse Jacobian, and determinant.
     DMPlexComputeCellGeometryFEM with quad=NULL returns v0 at the first
     vertex and J as the half-edge vectors of the affine map.  The true
     mapping for PETSc's biunit reference triangle (-1,-1),(1,-1),(-1,1)
     is  x = v0 + J*(xi - xi0) with xi0 = (-1,...,-1).  We shift v0 so
     that the simpler formula  x = v0_shifted + J*xi  can be used with
     the quadrature/tabulation points that live in the biunit cell.

     detJ may be negative when the cell vertex ordering is opposite to
     PETSc's reference convention (common with gmsh imports).  We take
     |detJ| for integration weights; invJ is consistent with the signed
     detJ, so  invJ^T * grad_ref  gives correct physical gradients
     regardless of the sign of detJ.                                      */
  PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ));
  for (PetscInt d = 0; d < dim; d++)
    for (PetscInt e = 0; e < dim; e++) v0[d] += J[d * dim + e];
  detJ = PetscAbsReal(detJ);

  /* Tabulate basis and gradients at reference quadrature points.
     PetscFECreateTabulation(fe, nrepl, npoints, points, K, T) computes
     T->T[0] (values) and T->T[1] (gradients) when K >= 1.               */
  PetscCall(PetscFEGetQuadrature(fe, &quad));
  PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &Nqp, &qpoints, &qweights));
  PetscCall(PetscFECreateTabulation(fe, 1, Nqp, qpoints, 1, &T));

  /* Zero local arrays */
  PetscCall(PetscArrayzero(A_loc, Nb * Nb));
  PetscCall(PetscArrayzero(b_loc, Nb));

  for (PetscInt q = 0; q < Nqp; q++) {
    /* Physical coordinates of this quadrature point */
    PetscReal xq[3] = {0.0, 0.0, 0.0};
    for (PetscInt d = 0; d < dim; d++) {
      xq[d] = v0[d];
      for (PetscInt e = 0; e < dim; e++)
        xq[d] += J[d * dim + e] * qpoints[q * dim + e];
    }

    /* Velocity at this physical point */
    PetscReal bv[3];
    GetVelocity(dim, ctx, xq, bv);

    PetscReal wdetJ = qweights[q] * detJ;

    for (PetscInt i = 0; i < Nb; i++) {
      /* Physical gradient of test function psi_i:  grad_x psi_i = invJ^T * grad_ref psi_i */
      PetscReal grad_psi_i[3] = {0.0, 0.0, 0.0};
      for (PetscInt d = 0; d < dim; d++)
        for (PetscInt e = 0; e < dim; e++)
          grad_psi_i[d] += invJ[e * dim + d] * T->T[1][(q * Nb + i) * dim + e];

      /* b . grad psi_i */
      PetscReal b_dot_grad_psi_i = 0.0;
      for (PetscInt d = 0; d < dim; d++) b_dot_grad_psi_i += bv[d] * grad_psi_i[d];

      for (PetscInt j = 0; j < Nb; j++) {
        /* phi_j evaluated at this quadrature point (scalar: component 0) */
        PetscReal phi_j = T->T[0][(q * Nb + j) * 1 + 0];
        /* Volume term: -phi_j * (b . grad psi_i), integrated */
        A_loc[i * Nb + j] -= phi_j * b_dot_grad_psi_i * wdetJ;
      }
    }
    /* RHS: f=0 so b_loc stays zero */
  }

  PetscCall(PetscTabulationDestroy(&T));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   Per-cell DG mass-matrix block
     M_ij = integral_K  phi_i(x) phi_j(x) dx
          = sum_q  w_q * |detJ| * phi_i(xi_q) * phi_j(xi_q)
   M is block-diagonal cell-by-cell in DG because the basis is broken.
   Inverted per cell this is the correct left-preconditioner for the
   time-dependent system   M du/dt + A_stiff u = b_stiff.
   ----------------------------------------------------------------------- */
static PetscErrorCode AssembleCellMassBlock(DM dm, PetscFE fe, PetscInt c,
                                            PetscInt Nb,
                                            PetscScalar *M_loc /* Nb*Nb */)
{
  PetscTabulation   T;
  PetscQuadrature   quad;
  const PetscReal  *qpoints, *qweights;
  PetscInt          Nqp;
  PetscReal         v0[3], J[9], invJ[9], detJ;

  PetscFunctionBeginUser;
  PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ));
  detJ = PetscAbsReal(detJ);

  PetscCall(PetscFEGetQuadrature(fe, &quad));
  PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &Nqp, &qpoints, &qweights));
  PetscCall(PetscFECreateTabulation(fe, 1, Nqp, qpoints, 0, &T));

  PetscCall(PetscArrayzero(M_loc, Nb * Nb));
  for (PetscInt q = 0; q < Nqp; q++) {
    const PetscReal wdetJ = qweights[q] * detJ;
    for (PetscInt i = 0; i < Nb; i++) {
      const PetscReal phi_i = T->T[0][(q * Nb + i) * 1 + 0];
      for (PetscInt j = 0; j < Nb; j++) {
        const PetscReal phi_j = T->T[0][(q * Nb + j) * 1 + 0];
        M_loc[i * Nb + j] += phi_i * phi_j * wdetJ;
      }
    }
  }
  PetscCall(PetscTabulationDestroy(&T));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   Facet flux assembly
   --------------------
   For a facet f shared by cells cL (left/plus side) and cR (right/minus
   side, or cR < 0 for a boundary facet), assemble the upwind flux
   contribution into the global matrix and RHS.

   The upwind flux across the facet (with outward normal n from cL) is:

     Fhat(u+, u-, n) = max(b.n, 0) * u+  +  min(b.n, 0) * u-

   where u+ is the trace from cL and u- is from cR (or the boundary value).

   This contributes four blocks to the matrix:
     K_LL[i][j]:  test from cL, trial from cL  -> adds to A[rowL, colL]
     K_LR[i][j]:  test from cL, trial from cR  -> adds to A[rowL, colR]
     K_RL[i][j]:  test from cR, trial from cL  -> adds to A[rowR, colL]
     K_RR[i][j]:  test from cR, trial from cR  -> adds to A[rowR, colR]
   (K_RL and K_RR are only assembled for interior facets)

   For an inflow boundary facet (b.n < 0), the u- contribution goes to
   the RHS vector rather than the matrix.

   Facet quadrature:
   PetscFEGetFaceTabulation returns basis values at expanded face
   quadrature points (all faces x face quad points, mapped into the
   cell's reference space).  PetscFEExpandFaceQuadrature provides the
   corresponding reference-space coordinates so we can map them to
   physical space for velocity evaluation.

   Quadrature-point matching on interior facets:
   The expanded face quadrature for cL's local face fL and cR's local
   face fR may traverse the shared edge in opposite directions, depending
   on cone orientations.  We map both orderings (identity and reversed) to
   physical space and pick whichever produces coincident points.
   ----------------------------------------------------------------------- */

static PetscErrorCode AssembleFacet(DM dm, PetscFE fe,
                                     PetscInt f,
                                     PetscInt cL, PetscInt cR, /* cR < 0 => boundary */
                                     PetscBool isInflowBoundary,
                                     PetscInt boundaryFaceId,
                                     const AppCtx *ctx,
                                     PetscInt Nb,
                                     Mat A, Vec b_rhs)
{
  PetscTabulation Tf;
  PetscQuadrature fquad;
  PetscQuadrature efq;
  const PetscReal *fqpoints, *fqweights;
  const PetscReal *efqpoints;
  PetscInt         Nfq, dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));

  /* Face quadrature: weights and points in the (dim-1)-dimensional reference
     space of the facet itself.                                              */
  PetscCall(PetscFEGetFaceQuadrature(fe, &fquad));
  PetscCall(PetscQuadratureGetData(fquad, NULL, NULL, &Nfq, &fqpoints, &fqweights));
  PetscCall(PetscFEGetFaceTabulation(fe, 0, &Tf));
  PetscCall(PetscFEExpandFaceQuadrature(fe, fquad, &efq));
  PetscInt Nefq;
  PetscCall(PetscQuadratureGetData(efq, NULL, NULL, &Nefq, &efqpoints, NULL));

  const PetscInt *coneL, *coneR = NULL;
  PetscInt        coneSizeL, coneSizeR = 0, fL = -1, fR = -1;
  PetscCall(DMPlexGetConeSize(dm, cL, &coneSizeL));
  PetscCall(DMPlexGetCone(dm, cL, &coneL));
  for (PetscInt lf = 0; lf < coneSizeL; lf++) {
    if (coneL[lf] == f) { fL = lf; break; }
  }
  PetscCheck(fL >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB,
             "Facet not found in cone of cell cL");
  PetscCheck(Nefq == coneSizeL * Nfq, PETSC_COMM_SELF, PETSC_ERR_PLIB,
             "Expanded face quadrature size mismatch: %" PetscInt_FMT " != %" PetscInt_FMT,
             Nefq, coneSizeL * Nfq);

  if (cR >= 0) {
    PetscCall(DMPlexGetConeSize(dm, cR, &coneSizeR));
    PetscCall(DMPlexGetCone(dm, cR, &coneR));
    for (PetscInt lf = 0; lf < coneSizeR; lf++) {
      if (coneR[lf] == f) { fR = lf; break; }
    }
    PetscCheck(fR >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "Facet not found in cone of cell cR");
  }

  /* Cell geometry for cL and cR: v0 (shifted to biunit-origin image) and J.
     Used to map expanded face quadrature points to physical space.
     We also retrieve invJ and detJ so that the facet normal and area can be
     derived from the SAME affine mapping as the volume integral.  This is
     essential for consistency on non-affine cells (e.g. twisted hexes):
     using the FVM-computed facet area/normal would break the discrete
     divergence theorem and prevent the method from reproducing constants. */
  PetscReal v0L[3], JL[9], invJL[9], detJL;
  PetscCall(DMPlexComputeCellGeometryFEM(dm, cL, NULL, v0L, JL, invJL, &detJL));
  for (PetscInt d = 0; d < dim; d++)
    for (PetscInt e = 0; e < dim; e++) v0L[d] += JL[d * dim + e];

  PetscReal v0R[3], JR[9], invJR[9], detJR;
  PetscInt  *qRfromL = NULL;
  if (cR >= 0) {
    PetscCall(DMPlexComputeCellGeometryFEM(dm, cR, NULL, v0R, JR, invJR, &detJR));
    for (PetscInt d = 0; d < dim; d++)
      for (PetscInt e = 0; e < dim; e++) v0R[d] += JR[d * dim + e];
  /* Quadrature-point matching for interior facets.
     The expanded face quadrature gives reference-space points for each
     local face of a cell.  The q-th point on face fL of cL must
     correspond to the same physical location as some point on face fR of
     cR.  In 2D (edge facets) only identity or reversed orderings are
     possible, but in 3D (face facets) there can be rotations and flips.
     We use a general nearest-neighbour match in physical space.          */
    PetscCall(PetscMalloc1(Nfq, &qRfromL));

    for (PetscInt qL = 0; qL < Nfq; qL++) {
      PetscReal xL[3] = {0.0, 0.0, 0.0};
      const PetscReal *xiLq = &efqpoints[(fL * Nfq + qL) * dim];
      for (PetscInt d = 0; d < dim; d++) {
        xL[d] = v0L[d];
        for (PetscInt e = 0; e < dim; e++) xL[d] += JL[d * dim + e] * xiLq[e];
      }
      PetscReal bestDist = PETSC_MAX_REAL;
      PetscInt  bestQ    = 0;
      for (PetscInt qR = 0; qR < Nfq; qR++) {
        PetscReal xR[3] = {0.0, 0.0, 0.0};
        const PetscReal *xiRq = &efqpoints[(fR * Nfq + qR) * dim];
        for (PetscInt d = 0; d < dim; d++) {
          xR[d] = v0R[d];
          for (PetscInt e = 0; e < dim; e++) xR[d] += JR[d * dim + e] * xiRq[e];
        }
        PetscReal dist = 0.0;
        for (PetscInt d = 0; d < dim; d++) dist += (xR[d] - xL[d]) * (xR[d] - xL[d]);
        if (dist < bestDist) { bestDist = dist; bestQ = qR; }
      }
      qRfromL[qL] = bestQ;
    }
  }

  /* Determine the Jacobian-consistent outward normal for face fL of cL.

     For tensor-product cells (hex/quad), the faces align with reference
     coordinate planes (xi_k = ±1).  We identify the constant coordinate
     from the expanded face quadrature points and compute:
       n_w = |det(J)| * invJ^T * n_ref
     so that the facet integral is consistent with the volume integral
     (both use the same affine Jacobian).  This is essential for non-affine
     tensor-product cells (e.g. twisted hexes) where the FVM geometry would
     break the discrete divergence theorem.

     For simplices (tri/tet), the affine Jacobian is exact, so we use the
     FVM-computed normal and area (which are also exact).  We always use
     the FVM path for simplices because simplex faces are not generally
     aligned with reference coordinate planes, and with Nfq=1 the constant-
     direction detection cannot distinguish between faces.                  */

  /* Detect cell type to choose simplex vs tensor-product path */
  DMPolytopeType ctL;
  PetscCall(DMPlexGetCellType(dm, cL, &ctL));
  PetscBool isTensorProduct = (ctL == DM_POLYTOPE_QUADRILATERAL ||
                                ctL == DM_POLYTOPE_HEXAHEDRON);

  PetscReal n_refL[3] = {0.0, 0.0, 0.0};
  PetscBool isSimplexFace;
  PetscReal area_fvm = 0.0;

  if (isTensorProduct) {
    /* Tensor-product cell: find constant reference coordinate on face fL */
    isSimplexFace = PETSC_FALSE;
    const PetscReal *xi0 = &efqpoints[(fL * Nfq) * dim];
    for (PetscInt d = 0; d < dim; d++) {
      PetscBool allSame = PETSC_TRUE;
      for (PetscInt qq = 1; qq < Nfq; qq++) {
        const PetscReal *xiq = &efqpoints[(fL * Nfq + qq) * dim];
        if (PetscAbsReal(xiq[d] - xi0[d]) > 1e-12) { allSame = PETSC_FALSE; break; }
      }
      if (allSame && Nfq > 1) {
        n_refL[d] = (xi0[d] > 0) ? 1.0 : -1.0;
        break;
      } else if (Nfq == 1) {
        /* Single quad point: determine from the point value directly.
           For a [-1,1]^dim hex/quad, face quad points are at ±1 in the
           normal direction and in (-1,1) in the tangential directions. */
        if (PetscAbsReal(PetscAbsReal(xi0[d]) - 1.0) < 1e-12) {
          n_refL[d] = (xi0[d] > 0) ? 1.0 : -1.0;
          break;
        }
      }
    }
  } else {
    /* Simplex cell: use FVM geometry (exact for affine simplices) */
    isSimplexFace = PETSC_TRUE;
    PetscCall(FacetOutwardNormal(dm, f, cL, dim, n_refL));
    PetscCall(DMPlexComputeCellGeometryFVM(dm, f, &area_fvm, NULL, NULL));
  }

  /* Compute n_ref for face fR of cell cR (only for interior tensor-product faces) */
  PetscReal n_refR[3] = {0.0, 0.0, 0.0};
  if (cR >= 0 && !isSimplexFace) {
    const PetscReal *xi0R = &efqpoints[(fR * Nfq) * dim];
    for (PetscInt d = 0; d < dim; d++) {
      PetscBool allSame = PETSC_TRUE;
      for (PetscInt qq = 1; qq < Nfq; qq++) {
        const PetscReal *xiq = &efqpoints[(fR * Nfq + qq) * dim];
        if (PetscAbsReal(xiq[d] - xi0R[d]) > 1e-12) { allSame = PETSC_FALSE; break; }
      }
      if (allSame && Nfq > 1) {
        n_refR[d] = (xi0R[d] > 0) ? 1.0 : -1.0;
        break;
      } else if (Nfq == 1) {
        if (PetscAbsReal(PetscAbsReal(xi0R[d]) - 1.0) < 1e-12) {
          n_refR[d] = (xi0R[d] > 0) ? 1.0 : -1.0;
          break;
        }
      }
    }
  }

  /* Compute Jacobian-weighted normals n_w_L (and n_w_R for interior faces).
     For hex/quad: n_w = det(J) * invJ^T * n_ref
     For simplex:  n_w = n_out * facetArea / refFacetMeasure  (FVM fallback) */
  PetscReal n_wL[3] = {0.0, 0.0, 0.0};
  PetscReal n_wR[3] = {0.0, 0.0, 0.0};
  if (!isSimplexFace) {
    PetscReal absDetJL = PetscAbsReal(detJL);
    for (PetscInt d = 0; d < dim; d++)
      for (PetscInt e = 0; e < dim; e++)
        n_wL[d] += absDetJL * invJL[e * dim + d] * n_refL[e];
    if (cR >= 0) {
      PetscReal absDetJR = PetscAbsReal(detJR);
      for (PetscInt d = 0; d < dim; d++)
        for (PetscInt e = 0; e < dim; e++)
          n_wR[d] += absDetJR * invJR[e * dim + d] * n_refR[e];
    }
  } else {
    PetscReal refMeas = 0.0;
    for (PetscInt qq = 0; qq < Nfq; qq++) refMeas += fqweights[qq];
    PetscReal sc = area_fvm / refMeas;
    for (PetscInt d = 0; d < dim; d++) n_wL[d] = n_refL[d] * sc;
    /* For simplex, the normal is exact and -n_L works for cR */
    if (cR >= 0) {
      for (PetscInt d = 0; d < dim; d++) n_wR[d] = -n_wL[d];
    }
  }

  /* Local matrices / vectors */
  PetscScalar *K_LL, *K_LR = NULL, *K_RL = NULL, *K_RR = NULL;
  PetscScalar *g_L, *g_R = NULL;
  PetscCall(PetscCalloc1(Nb * Nb, &K_LL));
  PetscCall(PetscCalloc1(Nb, &g_L));
  if (cR >= 0) {
    PetscCall(PetscCalloc1(Nb * Nb, &K_LR));
    PetscCall(PetscCalloc1(Nb * Nb, &K_RL));
    PetscCall(PetscCalloc1(Nb * Nb, &K_RR));
    PetscCall(PetscCalloc1(Nb, &g_R));
  }

  /* Quadrature loop over face quadrature points — cL's face parameterization.
     K_LL and K_LR are assembled here, contributing to cL's rows.
     Using cL's canonical quad points and weights ensures cL's
     per-cell divergence theorem is satisfied.                             */
  for (PetscInt q = 0; q < Nfq; q++) {

    PetscReal xq[3] = {0.0, 0.0, 0.0};
    const PetscReal *xiLq = &efqpoints[(fL * Nfq + q) * dim];
    for (PetscInt d = 0; d < dim; d++) {
      xq[d] = v0L[d];
      for (PetscInt e = 0; e < dim; e++) xq[d] += JL[d * dim + e] * xiLq[e];
    }

    /* Velocity at physical point */
    PetscReal bv[3];
    GetVelocity(dim, ctx, xq, bv);

    /* Normal flux from cL's perspective using cL's Jacobian-weighted normal */
    PetscReal bnL = 0.0;
    for (PetscInt d = 0; d < dim; d++) bnL += bv[d] * n_wL[d];

    PetscReal bnL_plus  = PetscMax(bnL, 0.0);
    PetscReal bnL_minus = PetscMin(bnL, 0.0);

    /* Facet integration weight: face quadrature weight only.
       The surface-element scaling is already in n_w.                     */
    PetscReal wFacet = fqweights[q];

    /* Basis values at this facet quadrature point for each side */
    for (PetscInt i = 0; i < Nb; i++) {
      PetscReal psi_iL = Tf->T[0][(fL * Nfq + q) * Nb + i];
      PetscInt  qR = (cR >= 0) ? qRfromL[q] : -1;

      for (PetscInt j = 0; j < Nb; j++) {
        PetscReal phi_jL = Tf->T[0][(fL * Nfq + q) * Nb + j];

        /* K_LL: test from cL, trial from cL (upwind from cL contribution) */
        K_LL[i * Nb + j] += bnL_plus * phi_jL * psi_iL * wFacet;

        if (cR >= 0) {
          PetscReal phi_jR = Tf->T[0][(fR * Nfq + qR) * Nb + j];

          /* K_LR: test from cL, trial from cR (upwind from cR into cL) */
          K_LR[i * Nb + j] += bnL_minus * phi_jR * psi_iL * wFacet;
        }
      }

      if (cR < 0 && isInflowBoundary && bnL < 0.0) {
        PetscReal g_val = (PetscReal)InflowValue(ctx, boundaryFaceId, xq);
        g_L[i] -= bnL_minus * g_val * psi_iL * wFacet;
      }
    }
  }

  /* Second quadrature loop — cR's face parameterization (interior faces only).
     K_RL and K_RR are assembled here, contributing to cR's rows.
     Using cR's canonical quad points and weights ensures cR's
     per-cell divergence theorem is satisfied.  This is critical for
     non-affine cells where the qRfromL matching is a non-isometric
     permutation, causing weight mismatches if we use cL's weights.      */
  if (cR >= 0) {
    /* Build reverse matching: qLfromR maps cR's q-th point to cL's nearest */
    PetscInt *qLfromR;
    PetscCall(PetscMalloc1(Nfq, &qLfromR));
    for (PetscInt qR = 0; qR < Nfq; qR++) {
      PetscReal xR[3] = {0.0, 0.0, 0.0};
      const PetscReal *xiRq = &efqpoints[(fR * Nfq + qR) * dim];
      for (PetscInt d = 0; d < dim; d++) {
        xR[d] = v0R[d];
        for (PetscInt e = 0; e < dim; e++) xR[d] += JR[d * dim + e] * xiRq[e];
      }
      PetscReal bestDist = PETSC_MAX_REAL;
      PetscInt  bestQ    = 0;
      for (PetscInt qL = 0; qL < Nfq; qL++) {
        PetscReal xL[3] = {0.0, 0.0, 0.0};
        const PetscReal *xiLq = &efqpoints[(fL * Nfq + qL) * dim];
        for (PetscInt d = 0; d < dim; d++) {
          xL[d] = v0L[d];
          for (PetscInt e = 0; e < dim; e++) xL[d] += JL[d * dim + e] * xiLq[e];
        }
        PetscReal dist = 0.0;
        for (PetscInt d = 0; d < dim; d++) dist += (xR[d] - xL[d]) * (xR[d] - xL[d]);
        if (dist < bestDist) { bestDist = dist; bestQ = qL; }
      }
      qLfromR[qR] = bestQ;
    }

    for (PetscInt qR = 0; qR < Nfq; qR++) {
      PetscReal xqR[3] = {0.0, 0.0, 0.0};
      const PetscReal *xiRq = &efqpoints[(fR * Nfq + qR) * dim];
      for (PetscInt d = 0; d < dim; d++) {
        xqR[d] = v0R[d];
        for (PetscInt e = 0; e < dim; e++) xqR[d] += JR[d * dim + e] * xiRq[e];
      }

      PetscReal bv[3];
      GetVelocity(dim, ctx, xqR, bv);

      /* Normal flux from cR's perspective using cR's Jacobian-weighted normal */
      PetscReal bnR = 0.0;
      for (PetscInt d = 0; d < dim; d++) bnR += bv[d] * n_wR[d];
      PetscReal bnR_plus  = PetscMax(bnR, 0.0);
      PetscReal bnR_minus = PetscMin(bnR, 0.0);

      PetscReal wFacetR = fqweights[qR];
      PetscInt  qL      = qLfromR[qR];

      for (PetscInt i = 0; i < Nb; i++) {
        PetscReal psi_iR = Tf->T[0][(fR * Nfq + qR) * Nb + i];
        for (PetscInt j = 0; j < Nb; j++) {
          PetscReal phi_jL = Tf->T[0][(fL * Nfq + qL) * Nb + j];
          PetscReal phi_jR = Tf->T[0][(fR * Nfq + qR) * Nb + j];

          /* K_RL: test from cR, trial from cL.
                   cR sees inflow (bnR < 0) from cL. */
          K_RL[i * Nb + j] += bnR_minus * phi_jL * psi_iR * wFacetR;

          /* K_RR: test from cR, trial from cR.
                   cR sees outflow (bnR > 0) from itself. */
          K_RR[i * Nb + j] += bnR_plus * phi_jR * psi_iR * wFacetR;
        }
      }
    }
    PetscCall(PetscFree(qLfromR));
  }

  /* Scatter into global matrix and RHS — use global section offsets.
     Only assemble rows for locally-owned cells (ghost rows are assembled
     by the rank that owns them).                                          */
  PetscSection gsec;
  PetscCall(DMGetGlobalSection(dm, &gsec));

  PetscInt goffL;
  PetscCall(PetscSectionGetOffset(gsec, cL, &goffL));
  PetscBool ownL = (goffL >= 0) ? PETSC_TRUE : PETSC_FALSE;
  if (!ownL) goffL = -(goffL + 1); /* decode ghost → true global offset */

  PetscInt *colsL;
  PetscCall(PetscMalloc1(Nb, &colsL));
  for (PetscInt i = 0; i < Nb; i++) colsL[i] = goffL + i;

  if (ownL) {
    PetscCall(MatSetValues(A, Nb, colsL, Nb, colsL, K_LL, ADD_VALUES));
    PetscCall(VecSetValues(b_rhs, Nb, colsL, g_L, ADD_VALUES));
  }

  if (cR >= 0) {
    PetscInt goffR;
    PetscCall(PetscSectionGetOffset(gsec, cR, &goffR));
    PetscBool ownR = (goffR >= 0) ? PETSC_TRUE : PETSC_FALSE;
    if (!ownR) goffR = -(goffR + 1);

    PetscInt *colsR;
    PetscCall(PetscMalloc1(Nb, &colsR));
    for (PetscInt i = 0; i < Nb; i++) colsR[i] = goffR + i;

    if (ownL) {
      PetscCall(MatSetValues(A, Nb, colsL, Nb, colsR, K_LR, ADD_VALUES));
    }
    if (ownR) {
      PetscCall(MatSetValues(A, Nb, colsR, Nb, colsL, K_RL, ADD_VALUES));
      PetscCall(MatSetValues(A, Nb, colsR, Nb, colsR, K_RR, ADD_VALUES));
      PetscCall(VecSetValues(b_rhs, Nb, colsR, g_R, ADD_VALUES));
    }

    PetscCall(PetscFree(colsR));
    PetscCall(PetscFree(K_LR)); PetscCall(PetscFree(K_RL));
    PetscCall(PetscFree(K_RR)); PetscCall(PetscFree(g_R));
  }

  PetscCall(PetscFree(colsL));
  PetscCall(PetscFree(K_LL)); PetscCall(PetscFree(g_L));
  PetscCall(PetscQuadratureDestroy(&efq));
  if (cR >= 0) {
    PetscCall(PetscFree(qRfromL));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   Classify boundary facets using Face Sets IDs.

   *isBoundary is PETSC_TRUE if f has any Face Sets label value.
   *isInflow   is PETSC_TRUE if f belongs to one of the designated
               Dirichlet-inflow face sets (where InflowValue() may be
               applied).  These face sets correspond to the geometrically
               inflow sides for the default diagonal velocity:
                 2D: {1,4}   (bottom/y=0, left/x=0)
                 3D: {1,3,6} (bottom/z=0, front/y=0, left/x=0)

   IMPORTANT: *isInflow does NOT determine the upwind direction.  The
   physical inflow condition (b.n < 0) is checked separately in the
   assembly loop.  A Dirichlet value is only actually imposed when
   *isInflow is PETSC_TRUE AND b.n < 0 at the quadrature point.
   ----------------------------------------------------------------------- */
static PetscErrorCode ClassifyBoundaryFacet(DM dm, DMLabel label,
                                             PetscInt f, PetscInt dim,
                                             const AppCtx *ctx,
                                             PetscBool *isBoundary,
                                             PetscBool *isInflow)
{
  PetscInt val;

  PetscFunctionBeginUser;
  *isBoundary = PETSC_FALSE;
  *isInflow   = PETSC_FALSE;
  if (!label) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(DMLabelGetValue(label, f, &val));
  if (val < 0) PetscFunctionReturn(PETSC_SUCCESS);
  *isBoundary = PETSC_TRUE;

  if (dim == 2) {
    if (val == 1 || val == 4) *isInflow = PETSC_TRUE;
  } else {
    if (val == 1 || val == 3 || val == 6) *isInflow = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   Main assembly routine
   -----------------------------------------------------------------------
   Loops over:
     (a) cells  -> volume integrals  -> diagonal blocks
     (b) facets -> flux integrals    -> diagonal + off-diagonal blocks
   ----------------------------------------------------------------------- */
static PetscErrorCode AssembleSystem(DM dm, PetscFE fe,
                                      const AppCtx *ctx,
                                      PetscInt Nb, PetscInt Nq,
                                      Mat A, Vec b_rhs)
{
  PetscInt     cStart, cEnd, fStart, fEnd, dim;
  DMLabel      label;
  PetscScalar *A_loc, *b_loc;
  PetscSection gsec;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  PetscCall(DMGetLabel(dm, "Face Sets", &label));
  PetscCall(DMGetGlobalSection(dm, &gsec));

  PetscCall(PetscMalloc1(Nb * Nb, &A_loc));
  PetscCall(PetscMalloc1(Nb,      &b_loc));

  /* (a) Volume integrals — owned cells only */
  for (PetscInt c = cStart; c < cEnd; c++) {
    PetscInt goff;
    PetscCall(PetscSectionGetOffset(gsec, c, &goff));
    if (goff < 0) continue; /* ghost cell — assembled by owning rank */

    PetscCall(AssembleVolumeCell(dm, fe, c, ctx, Nb, Nq, A_loc, b_loc));

    PetscInt *rows;
    PetscCall(PetscMalloc1(Nb, &rows));
    for (PetscInt i = 0; i < Nb; i++) rows[i] = goff + i;
    PetscCall(MatSetValues(A, Nb, rows, Nb, rows, A_loc, ADD_VALUES));
    PetscCall(VecSetValues(b_rhs, Nb, rows, b_loc, ADD_VALUES));
    PetscCall(PetscFree(rows));
  }

  /* (b) Facet flux integrals */
  for (PetscInt f = fStart; f < fEnd; f++) {
    const PetscInt *supp;
    PetscInt        suppSize;
    PetscCall(DMPlexGetSupportSize(dm, f, &suppSize));
    PetscCall(DMPlexGetSupport(dm, f, &supp));

    if (suppSize == 2) {
      /* Interior facet — skip if both cells are ghosts */
      PetscInt g0, g1;
      PetscCall(PetscSectionGetOffset(gsec, supp[0], &g0));
      PetscCall(PetscSectionGetOffset(gsec, supp[1], &g1));
      if (g0 < 0 && g1 < 0) continue;

      PetscCall(AssembleFacet(dm, fe, f, supp[0], supp[1],
                               PETSC_FALSE, -1, ctx, Nb, A, b_rhs));
    } else if (suppSize == 1) {
      /* Boundary facet — skip if the cell is a ghost */
      PetscInt g0;
      PetscCall(PetscSectionGetOffset(gsec, supp[0], &g0));
      if (g0 < 0) continue;

      PetscInt  faceId = -1;
      PetscBool isBoundary, isInflow;
      PetscCall(ClassifyBoundaryFacet(dm, label, f, dim, ctx, &isBoundary, &isInflow));
      if (!isBoundary) isInflow = PETSC_FALSE;
      if (isBoundary) PetscCall(DMLabelGetValue(label, f, &faceId));
      PetscCall(AssembleFacet(dm, fe, f, supp[0], -1,
                               isInflow, faceId, ctx, Nb, A, b_rhs));
    }
  }

  PetscCall(PetscFree(A_loc));
  PetscCall(PetscFree(b_loc));

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(VecAssemblyBegin(b_rhs));
  PetscCall(VecAssemblyEnd(b_rhs));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   Write DG field to VTU/PVTU as a discontinuous point field on a broken
   mesh.  Fully parallel: each rank writes its own .vtu piece file, and
   rank 0 writes a .pvtu master file that references all piece files.

   Each original cell gets its own private copy of vertices, so jumps
   across facets are represented exactly.  For each copied vertex, we
   evaluate the DG polynomial in that owning cell and write it as point
   data.  Only locally-owned cells (non-ghost) are written.

   Supports triangles (3, VTK_TRIANGLE=5), quads (4, VTK_QUAD=9),
   tetrahedra (4, VTK_TETRA=10), and hexahedra (8, VTK_HEXAHEDRON=12).
   ----------------------------------------------------------------------- */
static PetscErrorCode WriteDGFieldVTU(DM dm, PetscFE fe, PetscSection sec,
                                      Vec x, const char basename[])
{
  PetscInt          dim, cStart, cEnd, vStart, vEnd, Nb;
  DM                cdm;
  PetscSection      csec, gsec;
  Vec               coordsLocal;
  const PetscScalar *xArr;
  PetscInt          numOwnedCells, numPoints, totalConn;
  PetscReal         *pts;
  PetscScalar       *vals;
  PetscInt          *conn, *offsets;
  unsigned char     *types;
  PetscInt          pCursor = 0, cellCursor = 0, connCursor = 0;
  PetscMPIInt       rank, size;
  char              piece_fname[PETSC_MAX_PATH_LEN];
  PetscInt          rstart;
  PetscInt          vertsPerCell; /* 3 tri, 4 quad/tet, 8 hex */
  unsigned char     vtkCellType;  /* 5 tri, 9 quad, 10 tet, 12 hex */

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));

  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(PetscFEGetDimension(fe, &Nb));
  PetscCall(DMGetGlobalSection(dm, &gsec));

  /* Detect cell type from the first cell */
  {
    DMPolytopeType ct;
    PetscCall(DMPlexGetCellType(dm, cStart, &ct));
    switch (ct) {
      case DM_POLYTOPE_TRIANGLE:      vertsPerCell = 3; vtkCellType =  5; break;
      case DM_POLYTOPE_QUADRILATERAL: vertsPerCell = 4; vtkCellType =  9; break;
      case DM_POLYTOPE_TETRAHEDRON:   vertsPerCell = 4; vtkCellType = 10; break;
      case DM_POLYTOPE_HEXAHEDRON:    vertsPerCell = 8; vtkCellType = 12; break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP,
                "Unsupported cell type %s for VTK output",
                DMPolytopeTypes[ct]);
    }
  }

  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetLocalSection(cdm, &csec));
  PetscCall(DMGetCoordinatesLocal(dm, &coordsLocal));

  /* Count owned (non-ghost) cells */
  numOwnedCells = 0;
  for (PetscInt c = cStart; c < cEnd; c++) {
    PetscInt goff;
    PetscCall(PetscSectionGetOffset(gsec, c, &goff));
    if (goff >= 0) numOwnedCells++;
  }
  numPoints = vertsPerCell * numOwnedCells;
  totalConn = vertsPerCell * numOwnedCells;

  PetscCall(PetscMalloc1(3 * numPoints, &pts));
  PetscCall(PetscMalloc1(numPoints, &vals));
  PetscCall(PetscMalloc1(totalConn, &conn));
  PetscCall(PetscMalloc1(numOwnedCells, &offsets));
  PetscCall(PetscMalloc1(numOwnedCells, &types));

  /* Get the start of the local ownership range for indexing into xArr */
  {
    PetscInt rend;
    PetscCall(VecGetOwnershipRange(x, &rstart, &rend));
  }

  PetscCall(VecGetArrayRead(x, &xArr));
  for (PetscInt c = cStart; c < cEnd; c++) {
    PetscInt  goff;
    PetscCall(PetscSectionGetOffset(gsec, c, &goff));
    if (goff < 0) continue; /* skip ghost cells */

    PetscInt            ncl, *closure = NULL;
    PetscInt            verts[8], nv = 0;  /* max 8 for hexes */
    PetscReal           v0[3], J[9], invJ[9], detJ;

    PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ));
    /* Shift v0 to biunit-origin image (see AssembleVolumeCell comment) */
    for (PetscInt dd = 0; dd < dim; dd++)
      for (PetscInt ee = 0; ee < dim; ee++) v0[dd] += J[dd * dim + ee];

    PetscCall(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &ncl, &closure));
    for (PetscInt i = 0; i < ncl && nv < vertsPerCell; i++) {
      PetscInt p = closure[2 * i];
      if (p >= vStart && p < vEnd) {
        PetscBool seen = PETSC_FALSE;
        for (PetscInt j = 0; j < nv; j++) {
          if (verts[j] == p) { seen = PETSC_TRUE; break; }
        }
        if (!seen) verts[nv++] = p;
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &ncl, &closure));
    PetscCheck(nv == vertsPerCell, PETSC_COMM_SELF, PETSC_ERR_SUP,
               "Expected cell with %" PetscInt_FMT " vertices, got %" PetscInt_FMT,
               vertsPerCell, nv);

    /* Reorder closure vertices into VTK order.  Map each vertex to
       reference coordinates via invJ, then classify by the sign of
       each component.  In the reference element every hex vertex sits
       at a corner of [-1,1]^dim, so the octant is always unique —
       even for skewed / rotated / non-axis-aligned hexes.

       VTK_HEXAHEDRON (type 12) vertex ordering:
         0:(−,−,−)  1:(+,−,−)  2:(+,+,−)  3:(−,+,−)
         4:(−,−,+)  5:(+,−,+)  6:(+,+,+)  7:(−,+,+)

       VTK_QUAD (type 9) vertex ordering:
         0:(−,−)  1:(+,−)  2:(+,+)  3:(−,+)                    */
    if (vtkCellType == 12 || vtkCellType == 9) {
      /* Octant-to-VTK index mapping.  Octant bits: bit0=x+, bit1=y+, bit2=z+ */
      static const PetscInt oct2vtk_hex[8]  = {0, 1, 3, 2, 4, 5, 7, 6};
      static const PetscInt oct2vtk_quad[4] = {0, 1, 3, 2};

      PetscInt  sorted[8];
      PetscBool octant_ok = PETSC_TRUE;
      PetscInt  used[8]   = {0, 0, 0, 0, 0, 0, 0, 0};
      for (PetscInt lv = 0; lv < vertsPerCell; lv++) {
        /* Get vertex physical coordinates */
        PetscInt csz; PetscScalar *cv = NULL;
        PetscReal xvert[3] = {0.0, 0.0, 0.0};
        PetscCall(DMPlexVecGetClosure(cdm, csec, coordsLocal, verts[lv], &csz, &cv));
        for (PetscInt d = 0; d < dim; d++) xvert[d] = PetscRealPart(cv[d]);
        PetscCall(DMPlexVecRestoreClosure(cdm, csec, coordsLocal, verts[lv], &csz, &cv));

        /* Map to reference coordinates: xi = invJ * (x - v0)
           v0 is the shifted origin (center of biunit cell), so xi ∈ [-1,1]^dim */
        PetscReal xi[3] = {0.0, 0.0, 0.0};
        for (PetscInt e = 0; e < dim; e++)
          for (PetscInt d = 0; d < dim; d++)
            xi[e] += invJ[e * dim + d] * (xvert[d] - v0[d]);

        PetscInt octant = 0;
        for (PetscInt d = 0; d < dim; d++) {
          if (xi[d] > 0.0) octant |= (1 << d);
        }
        PetscInt vtkIdx = (vtkCellType == 12) ? oct2vtk_hex[octant]
                                               : oct2vtk_quad[octant];
        if (used[vtkIdx]) { octant_ok = PETSC_FALSE; break; }
        used[vtkIdx] = 1;
        sorted[vtkIdx] = verts[lv];
      }
      if (octant_ok) {
        for (PetscInt lv = 0; lv < vertsPerCell; lv++) verts[lv] = sorted[lv];
      } else if (vtkCellType == 12) {
        /* ---- Fallback: face-based topological reordering for hexes ----
           Uses DMPlex cone (face) structure to determine correct VTK
           connectivity without relying on coordinate classification.   */
        const PetscInt *cellCone;
        PetscInt        cellConeSize;
        PetscCall(DMPlexGetConeSize(dm, c, &cellConeSize));
        PetscCall(DMPlexGetCone(dm, c, &cellCone));

        /* Gather vertices for each face */
        PetscInt fv[6][4], fnv[6];
        for (PetscInt fi = 0; fi < cellConeSize; fi++) {
          fnv[fi] = 0;
          PetscInt fncl; PetscInt *fcl = NULL;
          PetscCall(DMPlexGetTransitiveClosure(dm, cellCone[fi], PETSC_TRUE, &fncl, &fcl));
          for (PetscInt k = 0; k < fncl; k++) {
            PetscInt p = fcl[2 * k];
            if (p >= vStart && p < vEnd && fnv[fi] < 4) {
              PetscBool dup = PETSC_FALSE;
              for (PetscInt m = 0; m < fnv[fi]; m++)
                if (fv[fi][m] == p) { dup = PETSC_TRUE; break; }
              if (!dup) fv[fi][fnv[fi]++] = p;
            }
          }
          PetscCall(DMPlexRestoreTransitiveClosure(dm, cellCone[fi], PETSC_TRUE, &fncl, &fcl));
        }

        /* Pick face 0 as bottom; find opposite face (shares no vertices) */
        PetscInt bf = 0, tf = -1;
        for (PetscInt fi = 1; fi < cellConeSize; fi++) {
          PetscBool shared = PETSC_FALSE;
          for (PetscInt a = 0; a < fnv[bf] && !shared; a++)
            for (PetscInt b = 0; b < fnv[fi] && !shared; b++)
              if (fv[bf][a] == fv[fi][b]) shared = PETSC_TRUE;
          if (!shared) { tf = fi; break; }
        }

        /* Match each bottom vertex to its top counterpart.
           The correct pair shares TWO side faces (connected by a
           vertical edge); diagonal neighbours share only ONE.       */
        PetscInt bot[4], top[4];
        for (PetscInt i = 0; i < 4; i++) bot[i] = fv[bf][i];
        for (PetscInt i = 0; i < 4; i++) {
          top[i] = -1;
          for (PetscInt j = 0; j < 4; j++) {
            PetscInt cnt = 0;
            for (PetscInt fi = 0; fi < cellConeSize; fi++) {
              if (fi == bf || fi == tf) continue;
              PetscBool has_b = PETSC_FALSE, has_t = PETSC_FALSE;
              for (PetscInt k = 0; k < fnv[fi]; k++) {
                if (fv[fi][k] == bot[i])    has_b = PETSC_TRUE;
                if (fv[fi][k] == fv[tf][j]) has_t = PETSC_TRUE;
              }
              if (has_b && has_t) cnt++;
            }
            if (cnt == 2) { top[i] = fv[tf][j]; break; }
          }
        }

        for (PetscInt i = 0; i < 4; i++) {
          verts[i]     = bot[i];
          verts[i + 4] = top[i];
        }
      }
      /* else (quad fallback): keep original closure order */
    }

    /* Index into the local portion of the global solution vector */
    PetscInt localDGoff = goff - rstart;

    for (PetscInt lv = 0; lv < vertsPerCell; lv++) {
      PetscInt            coordSize;
      PetscScalar         *coord = NULL;
      PetscReal           xvert[3] = {0.0, 0.0, 0.0}, xi[3] = {0.0, 0.0, 0.0};
      PetscTabulation     T;
      PetscScalar         uval = 0.0;

      PetscCall(DMPlexVecGetClosure(cdm, csec, coordsLocal, verts[lv], &coordSize, &coord));
      PetscCheck(coordSize >= dim, PETSC_COMM_SELF, PETSC_ERR_PLIB,
                 "Vertex coordinate closure too small");
      for (PetscInt d = 0; d < dim; d++) xvert[d] = PetscRealPart(coord[d]);
      PetscCall(DMPlexVecRestoreClosure(cdm, csec, coordsLocal, verts[lv], &coordSize, &coord));

      for (PetscInt e = 0; e < dim; e++) {
        for (PetscInt d = 0; d < dim; d++)
          xi[e] += invJ[e * dim + d] * (xvert[d] - v0[d]);
      }

      PetscCall(PetscFECreateTabulation(fe, 1, 1, xi, 0, &T));
      for (PetscInt k = 0; k < Nb; k++) uval += xArr[localDGoff + k] * T->T[0][k];
      PetscCall(PetscTabulationDestroy(&T));

      pts[3 * pCursor + 0] = xvert[0];
      pts[3 * pCursor + 1] = xvert[1];
      pts[3 * pCursor + 2] = (dim > 2) ? xvert[2] : 0.0;
      vals[pCursor] = uval;
      conn[connCursor++] = pCursor;
      pCursor++;
    }

    offsets[cellCursor] = vertsPerCell * (cellCursor + 1);
    types[cellCursor]   = vtkCellType;
    cellCursor++;
  }
  PetscCall(VecRestoreArrayRead(x, &xArr));

  /* ---- Each rank writes its own .vtu piece file ---- */
  PetscCall(PetscSNPrintf(piece_fname, sizeof(piece_fname), "%s_%d.vtu", basename, (int)rank));
  {
    FILE *fp;
    PetscCall(PetscFOpen(PETSC_COMM_SELF, piece_fname, "w", &fp));
    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    fprintf(fp, "  <UnstructuredGrid>\n");
    fprintf(fp, "    <Piece NumberOfPoints=\"%" PetscInt_FMT "\" NumberOfCells=\"%" PetscInt_FMT "\">\n",
            numPoints, numOwnedCells);

    /* Point data */
    fprintf(fp, "      <PointData Scalars=\"adv_dg_solution\">\n");
    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"adv_dg_solution\" format=\"ascii\">\n");
    for (PetscInt p = 0; p < numPoints; p++) fprintf(fp, "%.16e ", (double)PetscRealPart(vals[p]));
    fprintf(fp, "\n        </DataArray>\n");
    fprintf(fp, "      </PointData>\n");
    fprintf(fp, "      <CellData/>\n");

    /* Points */
    fprintf(fp, "      <Points>\n");
    fprintf(fp, "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (PetscInt p = 0; p < numPoints; p++)
      fprintf(fp, "%.16e %.16e %.16e ", (double)pts[3*p+0], (double)pts[3*p+1], (double)pts[3*p+2]);
    fprintf(fp, "\n        </DataArray>\n");
    fprintf(fp, "      </Points>\n");

    /* Cells */
    fprintf(fp, "      <Cells>\n");
    fprintf(fp, "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
    for (PetscInt i = 0; i < totalConn; i++) fprintf(fp, "%" PetscInt_FMT " ", conn[i]);
    fprintf(fp, "\n        </DataArray>\n");

    fprintf(fp, "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
    for (PetscInt i = 0; i < numOwnedCells; i++) fprintf(fp, "%" PetscInt_FMT " ", offsets[i]);
    fprintf(fp, "\n        </DataArray>\n");

    fprintf(fp, "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
    for (PetscInt i = 0; i < numOwnedCells; i++) fprintf(fp, "%u ", (unsigned int)types[i]);
    fprintf(fp, "\n        </DataArray>\n");
    fprintf(fp, "      </Cells>\n");

    fprintf(fp, "    </Piece>\n");
    fprintf(fp, "  </UnstructuredGrid>\n");
    fprintf(fp, "</VTKFile>\n");
    PetscCall(PetscFClose(PETSC_COMM_SELF, fp));
  }

  /* ---- Rank 0 writes the .pvtu master file ---- */
  if (rank == 0) {
    char pvtu_fname[PETSC_MAX_PATH_LEN];
    FILE *fp;
    PetscCall(PetscSNPrintf(pvtu_fname, sizeof(pvtu_fname), "%s.pvtu", basename));
    PetscCall(PetscFOpen(PETSC_COMM_SELF, pvtu_fname, "w", &fp));
    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    fprintf(fp, "  <PUnstructuredGrid GhostLevel=\"0\">\n");
    fprintf(fp, "    <PPointData Scalars=\"adv_dg_solution\">\n");
    fprintf(fp, "      <PDataArray type=\"Float64\" Name=\"adv_dg_solution\"/>\n");
    fprintf(fp, "    </PPointData>\n");
    fprintf(fp, "    <PPoints>\n");
    fprintf(fp, "      <PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>\n");
    fprintf(fp, "    </PPoints>\n");
    for (PetscMPIInt r = 0; r < size; r++) {
      fprintf(fp, "    <Piece Source=\"%s_%d.vtu\"/>\n", basename, (int)r);
    }
    fprintf(fp, "  </PUnstructuredGrid>\n");
    fprintf(fp, "</VTKFile>\n");
    PetscCall(PetscFClose(PETSC_COMM_SELF, fp));
  }

  PetscCall(PetscFree(pts));
  PetscCall(PetscFree(vals));
  PetscCall(PetscFree(conn));
  PetscCall(PetscFree(offsets));
  PetscCall(PetscFree(types));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   PETSc TS callbacks for the -time_depend path

   Equation integrated:   du/dt + A u = b_rhs
   where A = M^{-1} A_stiff and b_rhs = M^{-1} b_stiff after the
   diag_scale step has applied the block-diagonal DG mass-matrix inverse
   (see AssembleCellMassBlock).  The effective mass matrix is therefore
   the identity, so the implicit form reduces to
     F(t, u, udot) = udot + A u - b_rhs
     J            = shift * I + A         (shift = 1/dt)

   Memory note: A is pre-shifted in place (A := A + shift_pre*I) before
   TSSolve so it is *itself* the implicit Jacobian.  This avoids a separate
   copy of J. IFunctionAdv
   subtracts the pre-shift contribution back out so the residual is
   unchanged, and IJacobianAdv just verifies the shift PETSc passes in
   matches our pre-applied value (it must, since dt is constant).
   ----------------------------------------------------------------------- */
typedef struct {
  Mat          A;           /* pre-shifted: holds shift_pre*I + A_orig    */
  Vec          b_rhs;
  DM           dm;
  PetscFE      fe;
  PetscSection sec;
  PetscBool    write_vtk;
  PetscReal    shift_pre;   /* the shift baked into A (= 1/dt for BEULER) */
} TSAdvCtx;

static PetscErrorCode IFunctionAdv(TS ts, PetscReal t, Vec u, Vec udot, Vec F, void *ctx_void)
{
  TSAdvCtx *c = (TSAdvCtx *)ctx_void;

  PetscFunctionBeginUser;
  (void)ts; (void)t;
  /* MatMult gives (A_orig + shift_pre*I)*u; subtract shift_pre*u to recover A_orig*u. */
  PetscCall(MatMult(c->A, u, F));
  PetscCall(VecAXPY(F, -c->shift_pre, u));
  PetscCall(VecAXPY(F, 1.0, udot));
  PetscCall(VecAXPY(F, -1.0, c->b_rhs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* A is already shift_pre*I + A_orig, so the implicit Jacobian is A itself.
   We just check that the shift PETSc requests is the one we baked in. */
static PetscErrorCode IJacobianAdv(TS ts, PetscReal t, Vec u, Vec udot, PetscReal shift,
                                   Mat J, Mat Jpre, void *ctx_void)
{
  TSAdvCtx *c = (TSAdvCtx *)ctx_void;

  PetscFunctionBeginUser;
  (void)ts; (void)t; (void)u; (void)udot; (void)J; (void)Jpre;
  PetscCheck(PetscAbsReal(shift - c->shift_pre) <= 1e-12 * PetscAbsReal(c->shift_pre),
             PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONGSTATE,
             "TS shift %g != pre-applied shift %g; -time_depend assumes constant dt",
             (double)shift, (double)c->shift_pre);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* TSMonitor that writes one VTU per step (including the IC at step 0)
   when -write_vtk is also set. */
static PetscErrorCode TSMonitorWriteVTK(TS ts, PetscInt step, PetscReal time, Vec u, void *ctx_void)
{
  TSAdvCtx *c = (TSAdvCtx *)ctx_void;
  char      basename[64];

  PetscFunctionBeginUser;
  (void)ts; (void)time;
  if (!c->write_vtk) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscSNPrintf(basename, sizeof(basename),
                          "dg_solution_ts_%04" PetscInt_FMT, step));
  PetscCall(WriteDGFieldVTU(c->dm, c->fe, c->sec, u, basename));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -----------------------------------------------------------------------
   main
   ----------------------------------------------------------------------- */
int main(int argc, char **argv)
{
  DM          dm;
  PetscFE     fe;
  PetscSection sec;
  Mat         A;
  Vec         x, b_rhs;
  KSP         ksp = NULL;
  AppCtx      ctx;
  PetscInt    dim, cStart, cEnd, Nb, Nq, nrows;
  DMPolytopeType ct;
  PetscBool   simplex;
  PetscLogStage setup_stage, gpu_copy_stage;
  MatType mat_type;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PCRegister_PFLARE();

  PetscCall(PetscLogStageRegister("Setup", &setup_stage));
  PetscCall(PetscLogStageRegister("GPU copy stage - triggered by a prelim KSPSolve", &gpu_copy_stage));

  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &ctx));

  /* ---- Mesh ---- */
  PetscCall(PetscLogStagePush(setup_stage));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &ctx, &dm));

  PetscCall(DMGetDimension(dm, &dim));

  /* To verify the solution set the bottom inlet condition as 1
     and then have the velocity either as [0, 1] in 2D or [0, 0, 1] in 3D
     so the solution should be 1 everywhere */
  if (ctx.verify_solution) {
    ctx.advection_velocity[0] = 0.0;
    ctx.advection_velocity[1] = (dim == 2) ? 1.0 : 0.0;
    ctx.advection_velocity[2] = (dim == 3) ? 1.0 : 0.0;
  }

  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  simplex = (DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct) + 1)
            ? PETSC_TRUE : PETSC_FALSE;

  /* ---- Build PetscFE (broken space: discontinuous Lagrange) ---- */
  {
    char prefix[] = "adv_dg_";
    /* PetscFECreateDefault reads the polynomial degree from the options
       database under -adv_dg_petscspace_degree.  PETSc's own internal
       default is 0 (piecewise constant), but we want 1 (linear) as the
       code default.  Only insert the default if the user has not already
       set it on the command line.
       PetscOptionsHasName(opts, prefix, name) requires name to start with
       '-'; passing NULL prefix and the full key is the simplest form.    */
    PetscBool degree_set;
    PetscCall(PetscOptionsHasName(NULL, NULL, "-adv_dg_petscspace_degree", &degree_set));
    if (!degree_set)
      PetscCall(PetscOptionsSetValue(NULL, "-adv_dg_petscspace_degree", "1"));

    /* PetscFECreateDefault creates a standard Lagrange element.
       We use it purely for its basis/quadrature machinery; the broken DOF
       layout is enforced by our manual PetscSection below, NOT by the FE.
       For facet integrals we call PetscFECreateTabulation directly at
       mapped reference points with K=0, bypassing the internal face
       tabulation cache entirely.                                          */
    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex,
                                   prefix, -1, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, "adv_dg"));

    /* Use sufficiently accurate quadrature for DG volume and face terms.
       Under-integration on facets can create rank-deficient local blocks
       after refinement.  For degree p, products in the bilinear form are
       integrated with order at least 2p+1. */
    {
      PetscSpace      sp;
      PetscInt        pmin, pmax, p, qorder;
      PetscQuadrature q = NULL, fq = NULL;

      PetscCall(PetscFEGetBasisSpace(fe, &sp));
      PetscCall(PetscSpaceGetDegree(sp, &pmin, &pmax));
      p      = (pmax >= 0) ? pmax : pmin;
      qorder = PetscMax(2 * p + 1, 10);
      PetscCall(PetscDTCreateDefaultQuadrature(ct, qorder, &q, &fq));
      PetscCall(PetscFESetQuadrature(fe, q));
      PetscCall(PetscFESetFaceQuadrature(fe, fq));
      PetscCall(PetscQuadratureDestroy(&q));
      PetscCall(PetscQuadratureDestroy(&fq));
    }
  }

  /* Number of basis functions and quadrature points */
  PetscCall(PetscFEGetDimension(fe, &Nb));
  {
    PetscQuadrature quad;
    PetscInt Nqp;
    PetscCall(PetscFEGetQuadrature(fe, &quad));
    PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &Nqp, NULL, NULL));
    Nq = Nqp;
  }

  /* ---- Build broken PetscSection ---- */
  PetscCall(BuildBrokenSection(dm, Nb, &sec));
  PetscCall(DMSetLocalSection(dm, sec));

  /* Compute local row count from owned cells only (exclude ghosts) */
  {
    PetscSection gsec;
    PetscCall(DMGetGlobalSection(dm, &gsec));
    nrows = 0;
    for (PetscInt c = cStart; c < cEnd; c++) {
      PetscInt goff;
      PetscCall(PetscSectionGetOffset(gsec, c, &goff));
      if (goff >= 0) nrows += Nb;
    }
  }

  /* ---- Allocate matrix and vectors ---- */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, nrows, nrows, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(PreallocateMatrix(dm, fe, Nb, &ctx, A));

  PetscCall(MatCreateVecs(A, &b_rhs, NULL));
  PetscCall(VecSet(b_rhs, 0.0));

  PetscCall(VecDuplicate(b_rhs, &x));
  PetscCall(VecSet(x, 1.0));
  PetscCall(PetscObjectSetName((PetscObject)x, "adv_dg_solution"));

  /* ---- Assemble ---- */
  PetscCall(AssembleSystem(dm, fe, &ctx, Nb, Nq, A, b_rhs));
  PetscCall(MatGetType(A, &mat_type));

  /* ---- Block diagonal scaling: D^{-1} A x = D^{-1} b ---- */
  if (ctx.diag_scale) {
    Mat         Dinv, DinvA, DinvA_f;
    Vec         Dinvb;
    PetscScalar *block, *work;
    PetscBLASInt *ipiv, nb_blas, lwork, info_blas;
    PetscInt    *dof_idx;

    nb_blas = (PetscBLASInt)Nb;
    lwork   = nb_blas;
    PetscCall(PetscMalloc1(Nb * Nb, &block));
    PetscCall(PetscMalloc1(Nb, &ipiv));
    PetscCall(PetscMalloc1(Nb, &work));
    PetscCall(PetscMalloc1(Nb, &dof_idx));

    /* Build the block-diagonal inverse matrix Dinv */
    PetscCall(MatCreate(PETSC_COMM_WORLD, &Dinv));
    PetscCall(MatSetSizes(Dinv, nrows, nrows, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatSetType(Dinv, mat_type));
    PetscCall(MatSeqAIJSetPreallocation(Dinv, Nb, NULL));
    PetscCall(MatMPIAIJSetPreallocation(Dinv, Nb, NULL, 0, NULL));
    PetscCall(MatSetUp(Dinv));

    {
    PetscSection gsec_ds;
    PetscCall(DMGetGlobalSection(dm, &gsec_ds));
    for (PetscInt c = cStart; c < cEnd; c++) {
      PetscInt goff;
      PetscCall(PetscSectionGetOffset(gsec_ds, c, &goff));
      if (goff < 0) continue; /* skip ghost cells */
      for (PetscInt i = 0; i < Nb; i++) dof_idx[i] = goff + i;

      if (ctx.time_depend) {
        /* Physically-correct preconditioner for M du/dt + A_stiff u = b:
           use the DG mass-matrix block M_K (block-diagonal in DG). */
        PetscCall(AssembleCellMassBlock(dm, fe, c, Nb, block));
      } else {
        /* Block-Jacobi on the steady stiffness. */
        PetscCall(MatGetValues(A, Nb, dof_idx, Nb, dof_idx, block));
      }

      /* Invert in-place via LAPACK: getrf (LU factor) then getri (invert) */
      LAPACKgetrf_(&nb_blas, &nb_blas, block, &nb_blas, ipiv, &info_blas);
      PetscCheck(info_blas == 0, PETSC_COMM_SELF, PETSC_ERR_LIB,
                 "LAPACK getrf failed (info=%d) on cell %" PetscInt_FMT, (int)info_blas, c);
      LAPACKgetri_(&nb_blas, block, &nb_blas, ipiv, work, &lwork, &info_blas);
      PetscCheck(info_blas == 0, PETSC_COMM_SELF, PETSC_ERR_LIB,
                 "LAPACK getri failed (info=%d) on cell %" PetscInt_FMT, (int)info_blas, c);

      PetscCall(MatSetValues(Dinv, Nb, dof_idx, Nb, dof_idx, block, INSERT_VALUES));
    }
    }
    PetscCall(MatAssemblyBegin(Dinv, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Dinv, MAT_FINAL_ASSEMBLY));

    /* DinvA = Dinv * A */
    PetscCall(MatMatMult(Dinv, A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &DinvA));

    /* Dinvb = Dinv * b */
    PetscCall(VecDuplicate(b_rhs, &Dinvb));
    PetscCall(MatMult(Dinv, b_rhs, Dinvb));

    /* Filter near-zero entries from DinvA */
    {
      PetscReal     filter_tol = 1e-15;
      PetscInt      rstart, rend;
      MPI_Comm      comm;
      PetscMPIInt   comm_size;
      PetscCount    total_nnz, count;
      PetscInt      *coo_i, *coo_j;
      PetscScalar   *coo_v;

      PetscCall(MatGetOwnershipRange(DinvA, &rstart, &rend));

      /* Count total (unfiltered) NNZ without copying — NULL cols/vals skips copy+sort */
      total_nnz = 0;
      for (PetscInt i = rstart; i < rend; i++) {
        PetscInt ncols;
        PetscCall(MatGetRow(DinvA, i, &ncols, NULL, NULL));
        total_nnz += ncols;
        PetscCall(MatRestoreRow(DinvA, i, &ncols, NULL, NULL));
      }

      /* Preallocate COO arrays to the total unfiltered NNZ */
      PetscCall(PetscMalloc1(total_nnz, &coo_i));
      PetscCall(PetscMalloc1(total_nnz, &coo_j));
      PetscCall(PetscMalloc1(total_nnz, &coo_v));

      /* Access diagonal (+off-diagonal) CSR blocks directly and fill COO */
      PetscCall(PetscObjectGetComm((PetscObject)DinvA, &comm));
      PetscCallMPI(MPI_Comm_size(comm, &comm_size));

      /* For MPI decompose into diagonal + off-diagonal sub-matrices */
      Mat             mat_local, mat_nonlocal = NULL;
      const PetscInt *garray = NULL;
      if (comm_size != 1) {
        PetscCall(MatMPIAIJGetSeqAIJ(DinvA, &mat_local, &mat_nonlocal, &garray));
      } else {
        mat_local = DinvA;
      }

      /* Scan local (diagonal) block — col global index = rstart + aj[k] */
      {
        const PetscInt   *ai, *aj;
        const PetscScalar *av;
        PetscInt          n;
        PetscBool         done;

        PetscCall(MatGetRowIJ(mat_local, 0, PETSC_FALSE, PETSC_FALSE, &n, &ai, &aj, &done));
        PetscCall(MatSeqAIJGetArrayRead(mat_local, &av));

        count = 0;
        for (PetscInt i = 0; i < nrows; i++) {
          for (PetscInt k = ai[i]; k < ai[i + 1]; k++) {
            if (PetscAbsScalar(av[k]) > filter_tol) {
              coo_i[count] = rstart + i;
              coo_j[count] = rstart + aj[k];
              coo_v[count] = av[k];
              count++;
            }
          }
        }

        PetscCall(MatSeqAIJRestoreArrayRead(mat_local, &av));
        PetscCall(MatRestoreRowIJ(mat_local, 0, PETSC_FALSE, PETSC_FALSE, &n, &ai, &aj, &done));
      }

      /* Scan off-diagonal block (parallel only) — col global index = garray[aj[k]] */
      if (comm_size != 1) {
        const PetscInt   *ai, *aj;
        const PetscScalar *av;
        PetscInt          n;
        PetscBool         done;

        PetscCall(MatGetRowIJ(mat_nonlocal, 0, PETSC_FALSE, PETSC_FALSE, &n, &ai, &aj, &done));
        PetscCall(MatSeqAIJGetArrayRead(mat_nonlocal, &av));

        for (PetscInt i = 0; i < nrows; i++) {
          for (PetscInt k = ai[i]; k < ai[i + 1]; k++) {
            if (PetscAbsScalar(av[k]) > filter_tol) {
              coo_i[count] = rstart + i;
              coo_j[count] = garray[aj[k]];
              coo_v[count] = av[k];
              count++;
            }
          }
        }

        PetscCall(MatSeqAIJRestoreArrayRead(mat_nonlocal, &av));
        PetscCall(MatRestoreRowIJ(mat_nonlocal, 0, PETSC_FALSE, PETSC_FALSE, &n, &ai, &aj, &done));
      }

      /* Build DinvA_f via COO interface */
      PetscCall(MatCreate(PETSC_COMM_WORLD, &DinvA_f));
      PetscCall(MatSetSizes(DinvA_f, nrows, nrows, PETSC_DETERMINE, PETSC_DETERMINE));
      PetscCall(MatSetType(DinvA_f, mat_type));
      PetscCall(MatSetPreallocationCOO(DinvA_f, count, coo_i, coo_j));
      PetscCall(MatSetValuesCOO(DinvA_f, coo_v, INSERT_VALUES));

      PetscCall(PetscFree(coo_i));
      PetscCall(PetscFree(coo_j));
      PetscCall(PetscFree(coo_v));
    }

    /* Replace A and b_rhs with the scaled versions */
    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&DinvA));
    PetscCall(MatDestroy(&Dinv));
    A = DinvA_f;
    PetscCall(VecDestroy(&b_rhs));
    b_rhs = Dinvb;

    PetscCall(PetscFree(block));
    PetscCall(PetscFree(ipiv));
    PetscCall(PetscFree(work));
    PetscCall(PetscFree(dof_idx));
  }

  PetscCall(PetscLogStagePop());

  if (ctx.time_depend) {

    /* -time_depend: no KSPSolve is performed here. The TS below is the
       only linear-solve driver. The first TS step's inner KSP builds its
       PC on J = shift*I + A (one host->device transfer on GPU builds);
       subsequent steps reuse the PC and run entirely on-device. */
    PetscCheck(ctx.diag_scale, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
               "-time_depend requires -diag_scale true");

    TS        ts;
    TSAdvCtx  tsctx;
    PetscReal dt;

    tsctx.A         = A;
    tsctx.b_rhs     = b_rhs;
    tsctx.dm        = dm;
    tsctx.fe        = fe;
    tsctx.sec       = sec;
    tsctx.write_vtk = ctx.write_vtk;

    PetscCall(PetscLogStagePush(gpu_copy_stage));
    // Set solution to zero at t=0
    PetscCall(VecSet(x, 0.0));

    PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
    PetscCall(TSSetType(ts, TSBEULER));
    PetscCall(TSSetEquationType(ts, TS_EQ_IMPLICIT));
    /* Linear problem: SNES reduces to KSPONLY (one linear solve per step,
       no Newton iteration).  Combined with the lag settings below this
       gives exactly one PC setup across the whole TS loop. */
    PetscCall(TSSetProblemType(ts, TS_LINEAR));
    PetscCall(TSSetIFunction(ts, NULL, IFunctionAdv, &tsctx));

    /* Defaults; user overrides via -ts_dt / -ts_max_steps / -ts_max_time /
       -ts_type / -ts_monitor / -ts_view / -ts_ksp_... / -ts_snes_... */
    PetscCall(TSSetTimeStep(ts, 0.01));
    PetscCall(TSSetMaxTime(ts, 0.1));
    PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));

    PetscCall(TSMonitorSet(ts, TSMonitorWriteVTK, &tsctx, NULL));
    PetscCall(TSSetSolution(ts, x));
    PetscCall(TSSetFromOptions(ts));

    /* Pre-shift A in place so it serves as the implicit Jacobian without
       a separate matrix. IFunctionAdv subtracts the
       pre-shift contribution back out so the residual is unchanged. */
    PetscCall(TSGetTimeStep(ts, &dt));
    tsctx.shift_pre = 1.0 / dt;
    PetscCall(MatShift(A, tsctx.shift_pre));
    PetscCall(TSSetIJacobian(ts, A, A, IJacobianAdv, &tsctx));

    /* Constant dt => shift = 1/dt is constant => the pre-shifted A is
       the correct Jacobian for every step.  Build the PC once and reuse. */
    {
      SNES snes;
      PetscCall(TSGetSNES(ts, &snes));
      PetscCall(SNESSetLagJacobian(snes, -2));
      PetscCall(SNESSetLagJacobianPersists(snes, PETSC_TRUE));
      PetscCall(SNESSetLagPreconditioner(snes, -2));
      PetscCall(SNESSetLagPreconditionerPersists(snes, PETSC_TRUE));
    }

    // Do the setup
    PetscCall(TSSetUp(ts));

    // Force the PC setup before TSSolve so we can time them separately.
    // TSSetUp does not propagate down to KSP/PC setup, so we do it manually.
    SNES snes;
    KSP  ksp_ts;
    PetscCall(TSGetSNES(ts, &snes));
    PetscCall(SNESSetUp(snes));
    PetscCall(SNESGetKSP(snes, &ksp_ts));
    PetscCall(KSPSetOperators(ksp_ts, A, A));
    PetscCall(KSPSetUp(ksp_ts));
    if (ctx.second_solve) {
      // We do an initial KSPSolve here to trigger the GPU copy 
      PetscCall(KSPSolve(ksp_ts, b_rhs, x));
      // Reset the initial condition
      PetscCall(VecSet(x, 0.0));
    }
    PetscCall(PetscLogStagePop());

    // Start the time stepping
    PetscCall(TSSolve(ts, x));

    TSConvergedReason ts_reason;
    PetscCall(TSGetConvergedReason(ts, &ts_reason));
    PetscCall(TSDestroy(&ts));
    if (ts_reason < 0) return 1;

  } else {

    /* ---- KSP solve ---- */
    PetscCall(PetscLogStagePush(gpu_copy_stage));
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
    PetscCall(KSPSolve(ksp, b_rhs, x));
    PetscCall(PetscLogStagePop());

    KSPConvergedReason reason;
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    if (reason < 0) {
      return 1;
    }

    /* ---- Optional second solve (e.g. to time the solve without setup) ---- */
    if (ctx.second_solve) {
      PetscCall(VecSet(x, 1.0));
      PetscCall(KSPSolve(ksp, b_rhs, x));
      PetscCall(KSPGetConvergedReason(ksp, &reason));
      if (reason < 0) {
        return 1;
      }
    }
  }

  /* ---- View solution ---- */
  if (ctx.write_vtk) {
    PetscCall(WriteDGFieldVTU(dm, fe, sec, x, "dg_solution"));
  }

  /* ---- Verify solution is 1 everywhere ---- */
  if (ctx.verify_solution) {
    Vec       x_ones;
    PetscReal err_norm;
    PetscCall(VecDuplicate(x, &x_ones));
    PetscCall(VecSet(x_ones, 1.0));
    PetscCall(VecAXPY(x_ones, -1.0, x));
    PetscCall(VecNorm(x_ones, NORM_INFINITY, &err_norm));
    PetscCall(VecDestroy(&x_ones));
    PetscCheck(err_norm < 1e-10, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
               "Solution verification FAILED: max|u - 1| = %g > 1e-10", (double)err_norm);
  }

  /* ---- Cleanup ---- */
  if (!ctx.time_depend) PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b_rhs));
  PetscCall(PetscSectionDestroy(&sec));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}