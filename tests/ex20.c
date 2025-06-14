static char help[] = "Poisson Problem with finite elements.\n\
This example supports automatic convergence estimation for multilevel solvers\n\
and solver adaptivity.\n\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscconvest.h>
#include "pflare.h"

typedef struct {
  PetscReal alpha; /* Diffusion coefficient */
  PetscReal advection_velocity[3]; // Advection velocity, in 2D or 3D
} AppCtx;

// Dirichlet BC: u = 1.0 (for inflow boundaries)
static PetscErrorCode dirichlet_bc_inflow(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 1.0;
  return PETSC_SUCCESS;
}

// Neumann BC: du/dn = 0.0 (for outflow/zero flux boundaries)
// This function must provide the flux value.
static PetscErrorCode neumann_bc_zero_flux(PetscInt dim, PetscReal time, const PetscReal x[], const PetscReal n[], PetscInt Nc, PetscScalar *flux, void *ctx)
{
  flux[0] = 0.0; // Zero flux
  return PETSC_SUCCESS;
}

// RHS source term
static void rhs_source_term(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 0.0;
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d * dim + d] = 1.0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  PetscOptionsBegin(comm, "", "Advection Problem Options", "DMPLEX");
  // Diffusion coefficient
  // Default alpha is 0 - pure advection
  options->alpha = 0.0;
  PetscOptionsGetReal(NULL, NULL, "-alpha", &options->alpha, NULL);

  // Initialize advection to zero
  options->advection_velocity[0] = 0.0; // u
  options->advection_velocity[1] = 0.0; // v
  options->advection_velocity[2] = 0.0; // w (for 3D)

  // Advection velocities - direction is [cos(theta), sin(theta)]
  // Default theta is pi/4
  PetscReal pi = 4*atan(1.0);
  PetscReal theta = pi/4.0;
  PetscOptionsGetReal(NULL, NULL, "-theta", &theta, NULL);
  PetscCheck(theta <= pi/2.0 && theta >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "Theta must be between 0 and pi/2");

  // Coefficients for 2d advection
  options->advection_velocity[0] = cos(theta);
  options->advection_velocity[1] = sin(theta);

  // Or the user can pass in the individual advection velocities
  // This will override theta
  PetscReal u_test = 0.0, v_test = 0.0, w_test = 0.0;
  PetscBool option_found_u, option_found_v, option_found_w;
  PetscOptionsGetReal(NULL, NULL, "-u", &u_test, &option_found_u);
  PetscOptionsGetReal(NULL, NULL, "-v", &v_test, &option_found_v);
  PetscOptionsGetReal(NULL, NULL, "-w", &w_test, &option_found_w);

  if (option_found_u) PetscCheck(u_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "u must be positive");
  if (option_found_v) PetscCheck(v_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "v must be positive");
  if (option_found_w) PetscCheck(w_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "w must be positive");

  if (option_found_u && option_found_v && option_found_w) {
   options->advection_velocity[0] = u_test;
   options->advection_velocity[1] = v_test;
   options->advection_velocity[2] = w_test;
  }

  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetApplicationContext(*dm, user));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  DMLabel        label;
  PetscInt       dim;
  PetscCall(DMGetDimension(dm, &dim));

  PetscInt numInflow = 1, numOutflow = 0;
  // This is the boundary id we apply the inflow condition to, which is the bottom face
  PetscInt inflowids[] = {1};
  // The remaining boundary ids for outflow conditions
  PetscInt outflowids[] = {2, 3, 4, 5, 6}; // Default for 3D
  if (dim == 2)
  {
    numOutflow = 3;
  }
  else
  {
    numOutflow = 5;
  }

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(DMGetLabel(dm, "Face Sets", &label));
  PetscCall(PetscDSSetResidual(ds, 0, rhs_source_term, f1_u));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
  // Dirichlet condition on bottom surface as inflow
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "inflow", label, numInflow, inflowids, 0, 0, NULL, (void (*)(void))dirichlet_bc_inflow, NULL, user, NULL));
  // Neumann condition (outflow - zero flux) on other surfaces
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "outflow", label, numOutflow, outflowids, 0, 0, NULL, (void (*)(void))neumann_bc_zero_flux, NULL, user, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], PetscErrorCode (*setup)(DM, AppCtx *), AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscBool      simplex;
  PetscInt       dim, cStart;
  char           prefix[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct) + 1 ? PETSC_TRUE : PETSC_FALSE;

  PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name));
  PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, 1, simplex, name ? prefix : NULL, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, name));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall((*setup)(dm, user));
  while (cdm) {
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM     dm;   /* Problem specification */
  SNES   snes; /* Nonlinear solver */
  Vec    u;    /* Solutions */
  AppCtx user; /* User-defined work context */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));

  // Register the pflare types
  PCRegister_PFLARE();

  /* Primal system */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(SetupDiscretization(dm, "potential", SetupPrimalProblem, &user));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(VecSet(u, 0.0));
  PetscCall(PetscObjectSetName((PetscObject)u, "potential"));
  PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_FALSE, &user));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSolve(snes, NULL, u));
  PetscCall(SNESGetSolution(snes, &u));
  /* Cleanup */
  PetscCall(VecDestroy(&u));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}