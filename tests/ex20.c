static char help[] = "Advection diffusion FEM problem.\n\n\n";

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

// The f0 term in the weak form integral
static void advection_diffusion_f0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  // constants[1] is u, constants[2] is v, constants[3] is w
  // Pointer to the first component of the velocity in the constants array
  const PetscReal *advection_velocity = &constants[1]; 

  PetscReal adv_dot_grad_u = 0.0;
  PetscReal volumetric_source = 0.0; // Assuming f = 0 for now

  // Compute the advection term: v . grad(u)
  for (PetscInt d = 0; d < dim; ++d) {
    adv_dot_grad_u += advection_velocity[d] * u_x[d];
  }

  // integral ( (v . grad u) - f ) phi dx
  f0[0] = adv_dot_grad_u - volumetric_source;
}

// The f1 term in the weak form integral
static void advection_diffusion_f1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  // Get the diffusion coefficient
  const PetscReal alpha = constants[0];
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = alpha * u_x[d];
}

// The Jacobian for advection term
static void g1_jacobian_advection_term(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  // constants[1] is u, constants[2] is v, constants[3] is w
  // Pointer to the first component of the velocity
  const PetscReal *advection_velocity = &constants[1]; 

  PetscInt d;
  for (d = 0; d < dim; ++d) {
    g1[d] = advection_velocity[d];
  }
}

// The Jacobian for the diffusion term
static void g3_jacobian_diffusion_term(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  // Get the diffusion coefficient
  const PetscReal alpha = constants[0];   
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d * dim + d] = alpha;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  PetscOptionsBegin(comm, "", "Advection Problem Options", "DMPLEX");
  // Diffusion coefficient
  // Default alpha is 1 - pure diffusion
  options->alpha = 1.0;
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

  if (option_found_u) options->advection_velocity[0] = u_test;
  if (option_found_v) options->advection_velocity[1] = v_test;
  if (option_found_w) options->advection_velocity[2] = w_test;

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
  PetscCall(PetscDSSetResidual(ds, 0, advection_diffusion_f0, advection_diffusion_f1));
  // Set the Jacobian terms
  // g0_uu: d(f0)/d(u) - NULL (advection term is v . grad u, no direct u dependence)
  // g1_uu: d(f0)/d(grad u) - g1_jacobian_advection_term (contribution from advection)
  // g2_uu: d(f1)/d(u) - NULL (diffusion term is alpha grad u, no direct u dependence)
  // g3_uu: d(f1)/d(grad u) - g3_jacobian_diffusion_term (contribution from diffusion)  
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, g1_jacobian_advection_term, NULL, g3_jacobian_diffusion_term));
  // Dirichlet condition on bottom surface as inflow
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "inflow", label, numInflow, inflowids, 0, 0, NULL, (void (*)(void))dirichlet_bc_inflow, NULL, user, NULL));
  // Neumann condition (outflow - zero flux) on other surfaces
  PetscCall(DMAddBoundary(dm, DM_BC_NATURAL, "outflow", label, numOutflow, outflowids, 0, 0, NULL, (void (*)(void))neumann_bc_zero_flux, NULL, user, NULL));

  /* Setup constants that get passed into the FEM functions*/
  {
    PetscScalar constants[4];

    constants[0] = user->alpha;
    constants[1] = user->advection_velocity[0];
    constants[2] = user->advection_velocity[1];
    constants[3] = user->advection_velocity[2];
    PetscCall(PetscDSSetConstants(ds, 4, constants));
  }

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