/*   DMPlex/SNES/KSP solving a system of linear equations.
     Steady advection-diffusion equation with SUPG stabilised CG FEM
     Default is 2D triangles
     Can control dimension with -dm_plex_dim
     Can control quad/hex tri/tet with -dm_plex_simplex (if tri/tet need to configure petsc with triangle/ctetgen)
     Can control number of faces with -dm_plex_box_faces (if in parallel make sure you start with enough faces
       to sensibly distribute the initial mesh before refining)
     Can refine with -dm_refine
     Can read in an unstructured gmsh file with -dm_plex_filename
         - have to make sure boundary ids match (1 through 6)
     Can view the solution with -snes_view_solution vtk:solution.vtu

     ./adv_diff_cg_supg -adv_diff_petscspace_degree 1 -dm_refine 1
             : pure advection with linear FEM with theta = pi/4
               BCs left and bottom and back dirichlet, the others outflow
     ./adv_diff_cg_supg -adv_diff_petscspace_degree 1 -dm_refine 1 -u 0 -v 0 -alpha 1.0
             : pure diffusion with linear FEM
               BCs dirichlet on all sides
     ./adv_diff_cg_supg -adv_diff_petscspace_degree 1 -dm_refine 1 -alpha 1.0
             : advection-diffusion with linear FEM with theta=pi/4
               BCs dirichlet on all sides

     Can change default velocity from straight line to curved with -curved_velocity (default false)
     Can normalise velocity with -unit_velocity (default true) so that we have a unit velocity.
     Can control the direction of advection with -theta (pi/4 default), or by giving the -u and -v and -w directly
     If any of u,v,w are set then they will override the theta and unit velocity will be disabled

*/

static char help[] = "Solves steady advection-diffusion FEM problem with SUPG stabilization.\n\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscconvest.h>
#include <math.h>

#include "pflare.h"
#include "box_2D_gen_unstruc_mesh.h"

typedef struct {
  PetscReal alpha;                   // Diffusion coefficient
  PetscReal advection_velocity[3];   // Advection velocity, in 2D or 3D (for straight line case)
  PetscBool curved_velocity;         // Use curved velocity field if true
  PetscBool unit_velocity;           // Normalize velocity magnitude if true
} AppCtx;

// Helper function to compute velocity at a point
// constants[0] is alpha, constants[1-3] are constant velocity, constants[4] is curved flag (1.0 if curved)
// and constant[5] is velocity normalisation flag (1.0 if normalise)
static inline void GetVelocity(const PetscScalar constants[], const PetscReal x[], PetscReal v[])
{
  if (constants[4] == 1.0) {
    // Spatially-varying velocity field: top-left quadrant of a rotating circle (center at 1,0)
    // u(x,y) = y, v(x,y) = 1-x
    v[0] = x[1];         // u(x,y)
    v[1] = 1.0 - x[0];   // v(x,y)
    v[2] = 0.0;          // w (unused in 2D)
  } else {
    // Constant velocity field
    v[0] = constants[1];
    v[1] = constants[2];
    v[2] = constants[3];
  }

  // Normalize velocity if unit_velocity flag is set
  if (constants[5] == 1.0) {
    PetscReal mag = 0.0;
    for (int d = 0; d < 2; ++d) mag += v[d]*v[d];
    mag = sqrt(mag);
    if (mag > 1e-12) {
      v[0] /= mag;
      v[1] /= mag;
      v[2] /= mag;
    }
  }
}

// Helper function to compute the SUPG stabilization parameter tau
static inline PetscErrorCode ComputeSUPGStabilization(PetscInt dim, PetscReal h, PetscReal alpha, const PetscReal v[], PetscReal *tau)
{
  PetscReal v_mag = 0.0;
  PetscReal Pe, xi;
  PetscInt  d;

  PetscFunctionBeginUser;
  for (d = 0; d < dim; ++d) v_mag += v[d] * v[d];
  v_mag = PetscSqrtReal(v_mag);

  if (v_mag < 1.e-12) {
    *tau = 0.0;
  } else {
    // Peclet number: Pe = |v|*h / (2*alpha)
    if (alpha < 1.e-12) { // Handle pure advection
      Pe = 1.0e12;
    } else {
      Pe = (v_mag * h) / (2.0 * alpha);
    }

    // Upwinding function: xi(Pe) = coth(Pe) - 1/Pe
    if (Pe < 1.e-6) { // Taylor expansion for small Pe
      xi = Pe / 3.0 - Pe * Pe * Pe / 45.0;
    } else if (Pe > 1.0e8) { // Asymptotic limit for large Pe
      xi = 1.0;
    } else {
      xi = (1.0 / tanh(Pe)) - (1.0 / Pe);
    }
    // Stabilization parameter: tau = (h / 2|v|) * xi(Pe)
    *tau = (h / (2.0 * v_mag)) * xi;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Dirichlet BC: u = 0.0 (for inflow boundaries)
static PetscErrorCode dirichlet_bc(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return PETSC_SUCCESS;
}

// Dirichlet BC: u = 0.0 (for inflow boundaries)
static PetscErrorCode dirichlet_bc_one(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
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
  PetscReal advection_velocity[3];
  GetVelocity(constants, x, advection_velocity);

  PetscReal adv_dot_grad_u = 0.0;
  PetscReal volumetric_source = 0.0; // Assuming f = 0 for now

  // Compute the advection term: v . grad(u)
  for (PetscInt d = 0; d < dim; ++d) {
    adv_dot_grad_u += advection_velocity[d] * u_x[d];
  }

  // integral ( (v . grad u) - f ) phi dx
  f0[0] = adv_dot_grad_u - volumetric_source;
}

// The f1 term in the weak form integral, now with SUPG stabilization
static void advection_diffusion_f1_supg(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal  alpha = constants[0];
  const PetscReal  h     = a[0]; // Cell size from auxiliary field
  
  // Get velocity field
  PetscReal v[3];
  GetVelocity(constants, x, v);
  
  PetscReal        tau, v_dot_grad_u = 0.0;
  PetscInt         d;

  //printf("h = %g\n", (double)h);
  PetscCallAbort(PETSC_COMM_SELF, ComputeSUPGStabilization(dim, h, alpha, v, &tau));
  //printf("tau = %g\n", (double)tau);
  for (d = 0; d < dim; ++d) v_dot_grad_u += v[d] * u_x[d];

  // f1[d] = alpha * u_x[d] (diffusion) + tau * (v . grad_u) * v[d] (SUPG)
  for (d = 0; d < dim; ++d) f1[d] = alpha * u_x[d] + tau * v_dot_grad_u * v[d];
}

// The Jacobian for advection term
static void g1_jacobian_advection_term(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscReal advection_velocity[3];
  GetVelocity(constants, x, advection_velocity);

  PetscInt d;
  for (d = 0; d < dim; ++d) {
    g1[d] = advection_velocity[d];
  }
}

// The Jacobian for the diffusion term, now with SUPG stabilization
static void g3_jacobian_diffusion_term_supg(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal  alpha = constants[0];
  const PetscReal  h     = a[0]; // Cell size from auxiliary field
  
  // Get velocity field
  PetscReal v[3];
  GetVelocity(constants, x, v);
  
  PetscReal        tau;
  PetscInt         d, c;

  PetscCallAbort(PETSC_COMM_SELF, ComputeSUPGStabilization(dim, h, alpha, v, &tau));

  // g3[d,c] = d(f1[d])/d(u_x[c]) = alpha * delta_dc + tau * v[d] * v[c]
  for (d = 0; d < dim; ++d) {
    for (c = 0; c < dim; ++c) g3[d * dim + c] = tau * v[d] * v[c];
    g3[d * dim + d] += alpha;
  }
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  PetscOptionsBegin(comm, "", "Advection Problem Options", "DMPLEX");
  // Diffusion coefficient
  // Default alpha is 0 - pure advection
  options->alpha = 0.0;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-alpha", &options->alpha, NULL));

  // Curved velocity field option
  options->curved_velocity = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-curved_velocity", &options->curved_velocity, NULL));

  // Unit velocity normalization option - default to unit velocity
  options->unit_velocity = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-unit_velocity", &options->unit_velocity, NULL));  

  // Initialize advection to zero
  options->advection_velocity[0] = 0.0; // u
  options->advection_velocity[1] = 0.0; // v
  options->advection_velocity[2] = 0.0; // w (for 3D)

  // Advection velocities - direction is [cos(theta), sin(theta)]
  // Default theta is pi/4
  PetscReal pi = 4*atan(1.0);
  PetscReal theta = pi/4.0;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-theta", &theta, NULL));
  PetscCheck(theta <= pi/2.0 && theta >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "Theta must be between 0 and pi/2");

  // Coefficients for 2d advection
  options->advection_velocity[0] = cos(theta);
  options->advection_velocity[1] = sin(theta);

  // Or the user can pass in the individual advection velocities
  // This will override theta and unit velocity options if they are set
  PetscReal u_test = 0.0, v_test = 0.0, w_test = 0.0;
  PetscBool option_found_u, option_found_v, option_found_w;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-u", &u_test, &option_found_u));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-v", &v_test, &option_found_v));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-w", &w_test, &option_found_w));

  if (option_found_u) PetscCheck(u_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "u must be positive");
  if (option_found_v) PetscCheck(v_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "v must be positive");
  if (option_found_w) PetscCheck(w_test >= 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "w must be positive");

  if (option_found_u) options->advection_velocity[0] = u_test;
  if (option_found_v) options->advection_velocity[1] = v_test;
  if (option_found_w) options->advection_velocity[2] = w_test;

  // Don't normalise if user has explicitly set a velocity
  if (option_found_u || option_found_v || option_found_w) {
    options->unit_velocity = PETSC_FALSE;
  }

  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, double target_edge_length,int final_smooths, AppCtx *options, DM *dm)
{
  PetscFunctionBeginUser;
  //PetscCall(DMCreate(comm, dm));
  //PetscCall(DMSetType(*dm, DMPLEX));
  //PetscCall(DMSetFromOptions(*dm));
  *dm = GenerateBoxMeshDM(comm, target_edge_length, final_smooths, PETSC_TRUE);
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetApplicationContext(*dm, options));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *options)
{
  PetscDS        ds;
  DMLabel        label;
  PetscInt       dim;
  PetscCall(DMGetDimension(dm, &dim));

  PetscInt numInflow = 0, numOutflow = 0;
  PetscInt *inflow = NULL, *inflow_one = NULL, *outflow = NULL;

  // ~~~~~~~~~~~~~~~~~
  // For advection we just apply dirichlet inflow conditions on incoming faces
  // and Neumann outflow conditions (zero flux) on the other faces.
  // Solution should just be equal to the inflow condition across the domain
  // ~~~~~~~~~~~~~~~~~

  // In 2D its bottom and left
  PetscInt inflowids_2d[] = {1, 4};
  PetscInt inflowids_2d_one[] = {1};
  PetscInt outflowids_2d[] =  {2,3};
  // In 3D its bottom, left and back face
  PetscInt inflowids_3d[] = {1,3,6};
  PetscInt outflowids_3d[] =  {2,4,5};
  if (dim == 2)
  {
    numInflow = 2;
    inflow = &inflowids_2d[0];
    inflow_one = &inflowids_2d_one[0];
    numOutflow = 2;
    outflow = &outflowids_2d[0];
  }
  else
  {
    numInflow = 3;
    inflow = &inflowids_3d[0];
    numOutflow = 3;
    outflow = &outflowids_3d[0];
  }

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(DMGetLabel(dm, "Face Sets", &label));
  PetscCall(PetscDSSetResidual(ds, 0, advection_diffusion_f0, advection_diffusion_f1_supg));
  // Set the Jacobian terms
  // g0_uu: d(f0)/d(u) - NULL (advection term is v . grad u, no direct u dependence)
  // g1_uu: d(f0)/d(grad u) - g1_jacobian_advection_term (contribution from advection)
  // g2_uu: d(f1)/d(u) - NULL (diffusion term is alpha grad u, no direct u dependence)
  // g3_uu: d(f1)/d(grad u) - g3_jacobian_diffusion_term (contribution from diffusion)  
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, g1_jacobian_advection_term, NULL, g3_jacobian_diffusion_term_supg));
  
  // Dirichlet condition on bottom surface as inflow
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "inflow", label, numInflow, inflow, 0, 0, NULL, (void (*)(void))dirichlet_bc, NULL, options, NULL));
  //PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "inflow_one", label, numInflow, inflow_one, 0, 0, NULL, (void (*)(void))dirichlet_bc_one, NULL, options, NULL));
  // If no diffusion, Neumann condition (outflow - zero flux) on other surfaces
  if (options->alpha == 0.0)
  {
   PetscCall(DMAddBoundary(dm, DM_BC_NATURAL, "outflow", label, numOutflow, outflow, 0, 0, NULL, (void (*)(void))neumann_bc_zero_flux, NULL, options, NULL));
  }
  // If we have diffusion we have dirichlet bcs on the other surfaces
  else
  {
   PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "outflow", label, numOutflow, outflow, 0, 0, NULL, (void (*)(void))dirichlet_bc, NULL, options, NULL));
  }

  /* Setup constants that get passed into the FEM functions*/
  {
    PetscScalar constants[6];

    constants[0] = options->alpha;
    constants[1] = options->advection_velocity[0];
    constants[2] = options->advection_velocity[1];
    constants[3] = options->advection_velocity[2];
    constants[4] = options->curved_velocity ? 1.0 : 0.0;  // 1.0 indicates curved velocity
    constants[5] = options->unit_velocity ? 1.0 : 0.0;    // 1.0 indicates unit velocity normalization
    PetscCall(PetscDSSetConstants(ds, 6, constants));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Sets up the auxiliary DM and vector for characteristic length (h)
// We stabilize with a scaled version of the element volume - very simple
static PetscErrorCode SetupSUPG(DM dm)
{
  DM           aux_dm;
  PetscDS      ds;
  PetscSection  aux_section; // The section for the auxiliary data
  PetscFE      fe, fe_aux;
  Vec          h_vec;
  PetscScalar *h_arr;
  PetscInt     dim, c, cStart = 0, cEnd, zero = 0;
  DMPolytopeType ct;

  PetscFunctionBeginUser;
  // Get the quadrature rule from the main field's FE
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));

  // Clone the DM to create a DM for the auxiliary data
  PetscCall(DMGetDimension(dm, &dim));
  // Clone the DM to create a DM for the auxiliary data
  PetscCall(DMClone(dm, &aux_dm));

  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));

  // Create a P0 FE space and FORCE it to use the main field's quadrature
  PetscCall(PetscFECreateLagrangeByCell(PETSC_COMM_SELF, dim, 1, ct, zero, PETSC_DETERMINE, &fe_aux));
  PetscCall(PetscObjectSetName((PetscObject)fe_aux, "h"));
  PetscCall(PetscFECopyQuadrature(fe, fe_aux));

   // Set this FE in the auxiliary DM
  PetscCall(DMSetField(aux_dm, 0, NULL, (PetscObject)fe_aux));
  PetscCall(PetscFEDestroy(&fe_aux));
  // Create the DS for the auxiliary DM
  PetscCall(DMCreateDS(aux_dm));
  // Create a local vector to hold the cell-size data
  PetscCall(DMCreateLocalVector(aux_dm, &h_vec));
  PetscCall(PetscObjectSetName((PetscObject)h_vec, "h"));
  PetscCall(DMGetLocalSection(aux_dm, &aux_section));
  // Compute cell geometry and fill the vector
  PetscCall(VecGetArray(h_vec, &h_arr));
  for (c = cStart; c < cEnd; ++c) {
    PetscReal vol;
    PetscInt off;
    PetscCall(DMPlexComputeCellGeometryFVM(dm, c, &vol, NULL, NULL));
    PetscCall(PetscSectionGetOffset(aux_section, c, &off));
    //printf("Cell %d: vol = %g, off = %d\n", c, (double)vol, off);
    // Characteristic length h = V^(1/d)
    h_arr[off] = pow(vol, 1.0 / ((PetscReal)dim));
  }
  PetscCall(VecRestoreArray(h_vec, &h_arr));

  // Set this vector as the source for auxiliary fields in the main DM's DS
  PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 0, h_vec));
  // Clean up
  PetscCall(VecDestroy(&h_vec));
  PetscCall(DMDestroy(&aux_dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], PetscErrorCode (*setup)(DM, AppCtx *), AppCtx *options)
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
  PetscCall((*setup)(dm, options));
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
  AppCtx options; /* options-defined work context */
  PetscLogStage setup, gpu_copy;
  KSPConvergedReason reason;
  KSP ksp;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  // Register the pflare types
  PCRegister_PFLARE();

  PetscCall(PetscLogStageRegister("Setup", &setup));
  PetscCall(PetscLogStageRegister("GPU copy stage - triggered by a prelim KSPSolve", &gpu_copy));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &options));

  PetscBool second_solve= PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-second_solve", &second_solve, NULL));

    double target_len = 0.0025;
    PetscBool set;
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-target_edge_length", &target_len, &set));

    PetscInt final_smooth_its = 4;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-final_smooth_its", &final_smooth_its, &set));  
    int final_smooths = final_smooth_its;

  /* Primal system */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(CreateMesh(PETSC_COMM_WORLD,target_len, final_smooths, &options, &dm));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(SetupDiscretization(dm, "adv_diff", SetupPrimalProblem, &options));
  // *** Set up the auxiliary vector for SUPG stabilization ***
  PetscCall(SetupSUPG(dm));

  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(VecSet(u, 1.0));
  PetscCall(PetscObjectSetName((PetscObject)u, "adv_diff"));
  PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_FALSE, &options));
  // Only compute the jacobian once in case we're doing a second solve
  PetscInt lag = -2;
  PetscCall(SNESSetLagPreconditioner(snes, lag));
  PetscCall(SNESSetLagJacobian(snes, lag));

  // Only solving a linear problem for now
  PetscCall(SNESSetType(snes, SNESKSPONLY));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(PetscLogStagePush(setup));
  PetscCall(SNESSetUp(snes));
  PetscCall(PetscLogStagePop());  

  PetscCall(PetscLogStagePush(gpu_copy));
  PetscCall(SNESSolve(snes, NULL, u));
  PetscCall(PetscLogStagePop());

  // Get the iteration count
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPGetConvergedReason(ksp,&reason));  
  if (reason < 0)
  {
   return 1;
  }  

  // Solve
  // We set x to 1 rather than random as the vecrandom doesn't yet have a
  // gpu implementation and we don't want a copy occuring back to the cpu
  if (second_solve)
  {
   PetscCall(VecSet(u, 1.0));
   PetscCall(SNESSolve(snes, NULL, u));
  }
  
  PetscCall(KSPGetConvergedReason(ksp,&reason));  
  if (reason < 0)
  {
   return 1;
  }    

  PetscCall(SNESGetSolution(snes, &u));
  /* Cleanup */
  PetscCall(VecDestroy(&u));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}
