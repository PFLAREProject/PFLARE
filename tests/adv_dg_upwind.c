/*   DMPlex/KSP solving a scalar advection equation with upwinded DG FEM.
     Pure advection only (no diffusion).
     Default is 2D triangles
     Can control dimension with -dm_plex_dim
     Can control quad/hex tri/tet with -dm_plex_simplex (if tri/tet need to configure petsc with triangle/ctetgen)
     Can control number of faces with -dm_plex_box_faces (if in parallel make sure you start with enough faces
       to sensibly distribute the initial mesh before refining)
     Can refine with -dm_refine
     Can read in an unstructured gmsh file with -dm_plex_filename
         - have to make sure boundary ids match (1 through 4 in 2D, 1 through 6 in 3D)
     Can view the solution with -write_vtk

     ./adv_dg_upwind -adv_dg_petscspace_degree 1 -dm_refine 1
             : pure advection with linear DG FEM with theta = pi/4
               BCs left and bottom and back dirichlet (zero), the others outflow
     ./adv_dg_upwind -adv_dg_petscspace_degree 1 -dm_refine 1 -bottom_only_inflow_one
             : pure advection with linear DG FEM with theta = pi/4
               BCs bottom inflow set to 1, left/back set to 0, the others outflow

     Can change default velocity from straight line to curved with -curved_velocity (default false)
     Can normalise velocity with -unit_velocity (default true) so that we have a unit velocity.
     Can control the direction of advection with -theta (pi/4 default), or by giving the -u and -v and -w directly
     If any of u,v,w are set then they will override the theta and unit velocity will be disabled
     Can enable diagonal block element inverse scaling with -diag_scale (default false) which is essential for scalable 
       convergence with higher order basis functions

     Boundary label conventions (matching the SUPG code):
       2D: inflow  = {1, 4}  (left, bottom)
           outflow = {2, 3}  (right, top)
       3D: inflow  = {1, 3, 6}
           outflow = {2, 4, 5}

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
  PetscBool write_vtk;
} AppCtx;

/* -----------------------------------------------------------------------
   Velocity evaluation — identical logic to the SUPG code
   ----------------------------------------------------------------------- */
static inline void GetVelocity(const AppCtx *ctx, const PetscReal x[], PetscReal v[])
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
    for (PetscInt d = 0; d < 3; d++) mag += v[d] * v[d];
    mag = PetscSqrtReal(mag);
    if (mag > 1e-12) { v[0] /= mag; v[1] /= mag; v[2] /= mag; }
  }
}

/* Inflow Dirichlet value by boundary face id.
   Default: homogeneous (0) inflow.
   With -bottom_only_inflow_one in 2D: set g=1 on bottom (id 4),
   while keeping g=0 on other inflow sides (e.g. left id 1). */
static inline PetscScalar InflowValue(const AppCtx *ctx, PetscInt dim, PetscInt faceId,
                                      const PetscReal x[])
{
  (void)x;
  if (ctx->bottom_only_inflow_one && dim == 2) return (faceId == 1) ? 1.0 : 0.0;
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

  opt->diag_scale = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-diag_scale", &opt->diag_scale, NULL));

  opt->second_solve = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-second_solve", &opt->second_solve, NULL));

  opt->write_vtk = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-write_vtk", &opt->write_vtk, NULL));

  PetscReal pi = 4.0 * atan(1.0), theta = pi / 4.0;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-theta", &theta, NULL));
  PetscCheck(theta >= 0.0 && theta <= pi / 2.0, comm, PETSC_ERR_ARG_WRONGSTATE,
             "theta must be in [0, pi/2]");
  opt->advection_velocity[0] = PetscCosReal(theta);
  opt->advection_velocity[1] = PetscSinReal(theta);
  opt->advection_velocity[2] = 0.0;

  PetscReal uv, vv, wv;
  PetscBool fu, fv, fw;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-u", &uv, &fu));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-v", &vv, &fv));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-w", &wv, &fw));
  if (fu) opt->advection_velocity[0] = uv;
  if (fv) opt->advection_velocity[1] = vv;
  if (fw) opt->advection_velocity[2] = wv;
  if (fu || fv || fw) opt->unit_velocity = PETSC_FALSE;

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
  /* DMSetUp and DMLocalizeCoordinates ensure the local coordinate DM is
     fully initialised for DMPlexComputeCellGeometryFEM.  The box-mesh
     generator does this internally, but when reading a file via
     -dm_plex_filename DMSetFromOptions handles the read and we must call
     these explicitly to finish coordinate setup.  They are safe to call
     on the generated mesh too (they become no-ops if already done).      */
  //PetscCall(DMSetUp(*dm));
  /* GMsh (and other file formats) may use vertex orderings that give a
     negative Jacobian determinant under PETSc's reference element convention.
     DMPlexOrient corrects all cell orientations to be consistent with
     PETSc's outward-normal convention before any geometry is computed.    */
  PetscCall(DMPlexOrient(*dm));  
  //PetscCall(DMLocalizeCoordinates(*dm));
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

   Each cell has a diagonal block of size Nb x Nb (from volume + own-facet
   terms).  Each interior facet couples two cells, adding two off-diagonal
   blocks of size Nb x Nb.  We walk all facets of height 1 (codimension 1)
   and use DMPlexGetSupport to find the owning cells.

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
static PetscErrorCode PreallocateMatrix(DM dm, PetscInt Nb, Mat A)
{
  PetscInt     fStart, fEnd, cStart, cEnd;
  PetscInt    *d_nnz, *o_nnz;
  PetscInt     nlocal, rstart, rend;
  PetscSection gsec; /* global section — gives global DOF offsets */

  PetscFunctionBeginUser;
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

  /* Interior facets couple two cells.  For each locally-owned cell on the
     facet we record Nb column entries in d_nnz (neighbour is on-process)
     or o_nnz (neighbour is a ghost owned by another rank).               */
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

    if (own0) {
      for (PetscInt i = 0; i < Nb; i++) {
        if (own1) d_nnz[goff0 - rstart + i] += Nb;  /* both cells on this rank */
        else      o_nnz[goff0 - rstart + i] += Nb;  /* supp[1] lives on another rank */
      }
    }
    if (own1) {
      for (PetscInt i = 0; i < Nb; i++) {
        if (own0) d_nnz[goff1 - rstart + i] += Nb;  /* both cells on this rank */
        else      o_nnz[goff1 - rstart + i] += Nb;  /* supp[0] lives on another rank */
      }
    }
  }
#undef GlobalOffset

  PetscCall(MatSeqAIJSetPreallocation(A, 0, d_nnz));
  PetscCall(MatMPIAIJSetPreallocation(A, 0, d_nnz, 0, o_nnz));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));

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
    PetscReal rawNormal[3];
    PetscCall(DMPlexComputeCellGeometryFVM(dm, f, &area, NULL, rawNormal));
    for (PetscInt d = 0; d < dim; d++) n_out[d] = rawNormal[d] / area;
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
    GetVelocity(ctx, xq, bv);

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
  PetscReal        n_out[3]; /* outward normal from cL */

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));

  /* Outward normal from cL for this facet */
  PetscCall(FacetOutwardNormal(dm, f, cL, dim, n_out));

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
     Used to map expanded face quadrature points to physical space.        */
  PetscReal v0L[3], JL[9], invJL_dummy[9], detJL_dummy;
  PetscCall(DMPlexComputeCellGeometryFEM(dm, cL, NULL, v0L, JL, invJL_dummy, &detJL_dummy));
  for (PetscInt d = 0; d < dim; d++)
    for (PetscInt e = 0; e < dim; e++) v0L[d] += JL[d * dim + e];

  PetscReal v0R[3], JR[9], invJR_dummy[9], detJR_dummy;
  PetscInt  *qRfromL = NULL;
  if (cR >= 0) {
    PetscCall(DMPlexComputeCellGeometryFEM(dm, cR, NULL, v0R, JR, invJR_dummy, &detJR_dummy));
    for (PetscInt d = 0; d < dim; d++)
      for (PetscInt e = 0; e < dim; e++) v0R[d] += JR[d * dim + e];
  /* Quadrature-point matching for interior facets.
     The expanded face quadrature gives reference-space points for each
     local face of a cell.  The q-th point on face fL of cL must
     correspond to the same physical location as some point on face fR of
     cR.  Depending on the relative cone orientation, the ordering may be
     identity or reversed.  We test both and pick the one that minimises
     the summed squared physical distance.                                */
    PetscCall(PetscMalloc1(Nfq, &qRfromL));

    PetscReal idCost = 0.0, revCost = 0.0;
    for (PetscInt q = 0; q < Nfq; q++) {
      PetscReal xL[3] = {0.0, 0.0, 0.0}, xRid[3] = {0.0, 0.0, 0.0}, xRrev[3] = {0.0, 0.0, 0.0};
      const PetscReal *xiLq   = &efqpoints[(fL * Nfq + q) * dim];
      const PetscReal *xiRid  = &efqpoints[(fR * Nfq + q) * dim];
      const PetscReal *xiRrev = &efqpoints[(fR * Nfq + (Nfq - 1 - q)) * dim];

      for (PetscInt d = 0; d < dim; d++) {
        for (PetscInt e = 0; e < dim; e++) {
          xL[d]    += JL[d * dim + e] * xiLq[e];
          xRid[d]  += JR[d * dim + e] * xiRid[e];
          xRrev[d] += JR[d * dim + e] * xiRrev[e];
        }
        xL[d]    += v0L[d];
        xRid[d]  += v0R[d];
        xRrev[d] += v0R[d];
        idCost  += (xRid[d] - xL[d]) * (xRid[d] - xL[d]);
        revCost += (xRrev[d] - xL[d]) * (xRrev[d] - xL[d]);
      }
    }

    if (idCost <= revCost) {
      for (PetscInt q = 0; q < Nfq; q++) qRfromL[q] = q;
    } else {
      for (PetscInt q = 0; q < Nfq; q++) qRfromL[q] = Nfq - 1 - q;
    }
  }

  /* Facet area for integration weight scaling */
  PetscReal facetArea;
  PetscCall(DMPlexComputeCellGeometryFVM(dm, f, &facetArea, NULL, NULL));

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

  PetscReal refFacetMeasure = 0.0;
  for (PetscInt qq = 0; qq < Nfq; qq++) refFacetMeasure += fqweights[qq];
  PetscReal wScale = facetArea / refFacetMeasure;

  /* Quadrature loop over face quadrature points */
  for (PetscInt q = 0; q < Nfq; q++) {

    PetscReal xq[3] = {0.0, 0.0, 0.0};
    const PetscReal *xiLq = &efqpoints[(fL * Nfq + q) * dim];
    for (PetscInt d = 0; d < dim; d++) {
      xq[d] = v0L[d];
      for (PetscInt e = 0; e < dim; e++) xq[d] += JL[d * dim + e] * xiLq[e];
    }

    /* Velocity at physical point */
    PetscReal bv[3];
    GetVelocity(ctx, xq, bv);

    /* Normal flux: b . n_out (outward from cL) */
    PetscReal bn = 0.0;
    for (PetscInt d = 0; d < dim; d++) bn += bv[d] * n_out[d];

    PetscReal bn_plus  = PetscMax(bn, 0.0);  /* upwind from cL */
    PetscReal bn_minus = PetscMin(bn, 0.0);  /* upwind from cR */

    /* Facet integration weight: reference-face weight times physical
       facet measure scaling from DMPlex geometry. */
    PetscReal wFacet = fqweights[q] * wScale;

    /* Basis values at this facet quadrature point for each side */
    for (PetscInt i = 0; i < Nb; i++) {
      PetscReal psi_iL = Tf->T[0][(fL * Nfq + q) * Nb + i];
      PetscInt  qR = (cR >= 0) ? qRfromL[q] : -1;

      for (PetscInt j = 0; j < Nb; j++) {
        PetscReal phi_jL = Tf->T[0][(fL * Nfq + q) * Nb + j];

        /* K_LL: test from cL, trial from cL (upwind from cL contribution) */
        K_LL[i * Nb + j] += bn_plus * phi_jL * psi_iL * wFacet;

        if (cR >= 0) {
          PetscReal phi_jR = Tf->T[0][(fR * Nfq + qR) * Nb + j];
          PetscReal psi_iR = Tf->T[0][(fR * Nfq + qR) * Nb + i];

          /* K_LR: test from cL, trial from cR (upwind from cR contribution,
                   b.n < 0 so information flows from cR into cL)            */
          K_LR[i * Nb + j] += bn_minus * phi_jR * psi_iL * wFacet;

          /* K_RL: test from cR, trial from cL.
                   The outward normal from cR is -n_out, so b.n_R = -bn.
                   The cR test function sees the flux from cL when b.n_R < 0,
                   i.e. when bn > 0.  Contribution: -bn_plus * phi_jL * psi_iR.
                   The minus sign comes from the sign flip of the normal.    */
          K_RL[i * Nb + j] -= bn_plus * phi_jL * psi_iR * wFacet;

          /* K_RR: test from cR, trial from cR.
                   b.n_R = -bn, upwind from cR means b.n_R > 0, i.e. bn < 0.
                   Contribution: -bn_minus * phi_jR * psi_iR.               */
          K_RR[i * Nb + j] -= bn_minus * phi_jR * psi_iR * wFacet;
        }
        /* For outflow boundary (bn >= 0): only K_LL is nonzero via bn_plus,
           and g_L gets no boundary contribution. */
      }

      if (cR < 0 && isInflowBoundary && bn < 0.0) {
        /* Boundary: bn < 0 (inflow), u- = g (boundary value).
           The flux term bn_minus * g * psi_iL goes to RHS, and must be
           added once per test function (not once per trial function). */
        PetscReal g_val = (PetscReal)InflowValue(ctx, dim, boundaryFaceId, xq);
        g_L[i] -= bn_minus * g_val * psi_iL * wFacet;
      }
    }
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
   Classify boundary facets into inflow / outflow using Face Sets IDs.

   Matching the SUPG code conventions:
     2D inflow  = {1,4}, outflow = {2,3}
     3D inflow  = {1,3,6}, outflow = {2,4,5}

   *isBoundary is PETSC_TRUE if f has any Face Sets label value.
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

  if (ctx->bottom_only_inflow_one && dim == 2) {
    if (val == 1 || val == 4) *isInflow = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

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

   Supports triangles (3 vertices, VTK_TRIANGLE=5) and axis-aligned
   rectangular quads (4 vertices, VTK_QUAD=9).
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
  PetscInt          vertsPerCell; /* 3 for triangles, 4 for quads */
  unsigned char     vtkCellType;  /* 5 = VTK_TRIANGLE, 9 = VTK_QUAD */
  PetscBool         isSimplex;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));

  PetscCall(DMGetDimension(dm, &dim));
  PetscCheck(dim == 2, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP,
             "WriteDGFieldVTU currently supports only 2D meshes");

  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(PetscFEGetDimension(fe, &Nb));
  PetscCall(DMGetGlobalSection(dm, &gsec));

  /* Detect cell type from the first cell */
  {
    DMPolytopeType ct;
    PetscCall(DMPlexGetCellType(dm, cStart, &ct));
    isSimplex = (DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct) + 1)
                ? PETSC_TRUE : PETSC_FALSE;
  }
  if (isSimplex) {
    vertsPerCell = 3;
    vtkCellType  = 5; /* VTK_TRIANGLE */
  } else {
    vertsPerCell = 4;
    vtkCellType  = 9; /* VTK_QUAD */
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
    PetscInt            verts[4], nv = 0;  /* max 4 for quads */
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
      pts[3 * pCursor + 2] = 0.0;
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
   main
   ----------------------------------------------------------------------- */
int main(int argc, char **argv)
{
  DM          dm;
  PetscFE     fe;
  PetscSection sec;
  Mat         A;
  Vec         x, b_rhs;
  KSP         ksp;
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
      qorder = PetscMax(2 * p + 1, 2);
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
  PetscCall(PreallocateMatrix(dm, Nb, A));

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

      /* Extract the local element block from A */
      PetscCall(MatGetValues(A, Nb, dof_idx, Nb, dof_idx, block));

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
      PetscReal filter_tol = 1e-15;
      PetscInt  rstart, rend;

      PetscCall(MatCreate(PETSC_COMM_WORLD, &DinvA_f));
      PetscCall(MatSetSizes(DinvA_f, nrows, nrows, PETSC_DETERMINE, PETSC_DETERMINE));
      PetscCall(MatSetType(DinvA_f, mat_type));
      PetscCall(MatSetOption(DinvA_f, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
      PetscCall(MatSetUp(DinvA_f));

      PetscCall(MatGetOwnershipRange(DinvA, &rstart, &rend));
      for (PetscInt i = rstart; i < rend; i++) {
        PetscInt          ncols;
        const PetscInt    *cols;
        const PetscScalar *vals;
        PetscCall(MatGetRow(DinvA, i, &ncols, &cols, &vals));
        for (PetscInt j = 0; j < ncols; j++) {
          if (PetscAbsScalar(vals[j]) > filter_tol) {
            PetscCall(MatSetValue(DinvA_f, i, cols[j], vals[j], INSERT_VALUES));
          }
        }
        PetscCall(MatRestoreRow(DinvA, i, &ncols, &cols, &vals));
      }
      PetscCall(MatAssemblyBegin(DinvA_f, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(DinvA_f, MAT_FINAL_ASSEMBLY));
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

  /* ---- View solution ---- */
  if (ctx.write_vtk) {
    PetscCall(WriteDGFieldVTU(dm, fe, sec, x, "dg_solution"));
  }

  /* ---- Cleanup ---- */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b_rhs));
  PetscCall(PetscSectionDestroy(&sec));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}