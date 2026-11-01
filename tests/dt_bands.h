/*
  Stores the PCAIR polynomial coefficients for bands of the time step, used with
  -pc_air_diag_update by the time stepping tests with -dt_bands factor

  The time step is split into bands [factor^k, factor^(k+1)). The first time the time step
  is in a band, the diagonal update recomputes the polynomial coefficients and they are
  stored for that band after the PC setup. When the time step returns to a band with
  stored coefficients, they are restored with PCAIRSetPolyCoeffs and the diagonal update
  keeps them (PCAIRSetReusePolyCoeffs), so it doesn't compute any coefficients
  The hierarchy is never rebuilt by a diagonal update, so the stored coefficients stay
  valid for every band
  The number of coefficients restored has to match those stored by PCAIR, so this doesn't
  work with the newton inverse type, which can add a different number of extra roots

  With -dt_bands_check this errors if the bands did nothing: if coefficients aren't restored,
  if a new band doesn't compute its own, or if the update doesn't keep those restored
*/

#include <petscksp.h>
#include "pflare.h"

#define DT_BANDS_NUM    400
#define DT_BANDS_OFFSET 200
#define DT_BANDS_NONE   -1

typedef struct {
  PetscInt   petsc_level;
  int        which_inverse;
  PetscInt   rows, cols;
  PetscReal *coeffs;
} DtBandEntry;

typedef struct {
  // The ratio between band edges, the bands are off if <= 1
  PetscReal    factor;
  PC           pc;
  // The band to store after the next PC setup
  PetscInt     pending;
  PetscInt     n_entries[DT_BANDS_NUM];
  DtBandEntry *entries[DT_BANDS_NUM];
  // How many times coefficients were computed and stored, and restored
  PetscInt     n_stored, n_restored;
  // Do we check the bands, the last band stored and the band that should be in the PC
  PetscBool    check;
  PetscInt     last_stored, restored;
} DtBands;

static PetscErrorCode DtBandsCreate(DtBands *bands, PetscReal factor)
{
  PetscInt i;

  PetscFunctionBeginUser;
  bands->factor  = factor;
  bands->pc      = NULL;
  bands->pending = DT_BANDS_NONE;
  for (i = 0; i < DT_BANDS_NUM; i++) {
    bands->n_entries[i] = 0;
    bands->entries[i]   = NULL;
  }
  bands->n_stored   = 0;
  bands->n_restored = 0;
  bands->check       = PETSC_FALSE;
  bands->last_stored = DT_BANDS_NONE;
  bands->restored    = DT_BANDS_NONE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-dt_bands_check", &bands->check, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscInt DtBandIndex(DtBands *bands, PetscReal dt)
{
  PetscInt b = (PetscInt)PetscFloorReal(PetscLogReal(dt) / PetscLogReal(bands->factor)) + DT_BANDS_OFFSET;
  return PetscMin(PetscMax(b, 0), DT_BANDS_NUM - 1);
}

// Called when the Jacobian changes to the time step dt, before the PC setup
// Restores the coefficients for the band of dt if we have them, otherwise
// the diagonal update computes them and we store them after the setup
static PetscErrorCode DtBandsPrepare(DtBands *bands, PetscReal dt)
{
  PetscInt b, i;

  PetscFunctionBeginUser;
  if (bands->factor <= 1.0 || !bands->pc) PetscFunctionReturn(PETSC_SUCCESS);
  b = DtBandIndex(bands, dt);
  if (bands->n_entries[b] > 0) {
    for (i = 0; i < bands->n_entries[b]; i++) {
      DtBandEntry *e = &bands->entries[b][i];
      PetscCall(PCAIRSetPolyCoeffs(bands->pc, e->petsc_level, e->which_inverse, e->coeffs, e->rows, e->cols));
    }
    PetscCall(PCAIRSetReusePolyCoeffs(bands->pc, PETSC_TRUE));
    bands->pending  = DT_BANDS_NONE;
    bands->restored = b;
    bands->n_restored++;
  } else {
    PetscCall(PCAIRSetReusePolyCoeffs(bands->pc, PETSC_FALSE));
    bands->pending  = b;
    bands->restored = DT_BANDS_NONE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Copies one set of coefficients into the pending band, if PCAIR stores them
static PetscErrorCode DtBandsStoreEntry(DtBands *bands, PetscInt petsc_level, int which_inverse)
{
  PetscReal     *coeffs;
  PetscInt       rows, cols, b = bands->pending;
  PetscErrorCode ierr;
  DtBandEntry   *e;

  PetscFunctionBeginUser;
  // Not every level stores every inverse as a polynomial, so just skip those
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = PCAIRGetPolyCoeffs(bands->pc, petsc_level, which_inverse, &coeffs, &rows, &cols);
  PetscCall(PetscPopErrorHandler());
  if (ierr) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscRealloc(sizeof(DtBandEntry) * (bands->n_entries[b] + 1), &bands->entries[b]));
  e = &bands->entries[b][bands->n_entries[b]++];
  e->petsc_level   = petsc_level;
  e->which_inverse = which_inverse;
  e->rows          = rows;
  e->cols          = cols;
  PetscCall(PetscMalloc1(rows * cols, &e->coeffs));
  PetscCall(PetscArraycpy(e->coeffs, coeffs, rows * cols));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Errors if a new band has stored the same coefficients as the last band stored
static PetscErrorCode DtBandsCheckStored(DtBands *bands, PetscInt b)
{
  PetscInt  i;
  PetscBool same = PETSC_TRUE;

  PetscFunctionBeginUser;
  if (!bands->check || bands->last_stored == DT_BANDS_NONE) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(bands->n_entries[b] == bands->n_entries[bands->last_stored], PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Time step bands store different inverses");
  for (i = 0; i < bands->n_entries[b] && same; i++) {
    DtBandEntry *e = &bands->entries[b][i], *e_last = &bands->entries[bands->last_stored][i];
    if (e->rows != e_last->rows || e->cols != e_last->cols) same = PETSC_FALSE;
    else PetscCall(PetscArraycmp(e->coeffs, e_last->coeffs, e->rows * e->cols, &same));
  }
  PetscCheck(!same, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "A new time step band didn't compute new polynomial coefficients");
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Errors if the PC isn't using the coefficients of the band just restored
static PetscErrorCode DtBandsCheckRestored(DtBands *bands)
{
  PetscInt i;

  PetscFunctionBeginUser;
  if (!bands->check || bands->restored == DT_BANDS_NONE) PetscFunctionReturn(PETSC_SUCCESS);
  for (i = 0; i < bands->n_entries[bands->restored]; i++) {
    DtBandEntry *e = &bands->entries[bands->restored][i];
    PetscReal   *coeffs;
    PetscInt     rows, cols;
    PetscBool    same = PETSC_FALSE;

    PetscCall(PCAIRGetPolyCoeffs(bands->pc, e->petsc_level, e->which_inverse, &coeffs, &rows, &cols));
    if (rows == e->rows && cols == e->cols) PetscCall(PetscArraycmp(coeffs, e->coeffs, rows * cols, &same));
    PetscCheck(same, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "The polynomial coefficients restored for a time step band were not kept on petsc_level %" PetscInt_FMT, e->petsc_level);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Called after the PC setup, stores the coefficients for the pending band
static PetscErrorCode DtBandsStore(DtBands *bands)
{
  PetscInt num_levels, level;

  PetscFunctionBeginUser;
  if (bands->factor <= 1.0 || bands->pending == DT_BANDS_NONE) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PCAIRGetNumLevels(bands->pc, &num_levels));
  for (level = 1; level < num_levels; level++) {
    PetscCall(DtBandsStoreEntry(bands, level, COEFFS_INV_AFF));
    PetscCall(DtBandsStoreEntry(bands, level, COEFFS_INV_ACC));
  }
  PetscCall(DtBandsStoreEntry(bands, 0, COEFFS_INV_COARSE));
  PetscCall(DtBandsCheckStored(bands, bands->pending));
  bands->last_stored = bands->pending;
  bands->pending     = DT_BANDS_NONE;
  bands->n_stored++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// KSP presolve callback, which runs after the PC setup in KSPSolve
static PetscErrorCode DtBandsPreSolve(KSP ksp, Vec b, Vec x, void *ctx)
{
  PetscFunctionBeginUser;
  (void)ksp; (void)b; (void)x;
  PetscCall(DtBandsStore((DtBands *)ctx));
  PetscCall(DtBandsCheckRestored((DtBands *)ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Attaches the bands to the PC of a KSP
static PetscErrorCode DtBandsSetKSP(DtBands *bands, KSP ksp)
{
  PetscFunctionBeginUser;
  if (bands->factor <= 1.0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(KSPGetPC(ksp, &bands->pc));
  PetscCall(KSPSetPreSolve(ksp, DtBandsPreSolve, bands));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DtBandsDestroy(DtBands *bands)
{
  PetscInt b, i;

  PetscFunctionBeginUser;
  if (bands->factor > 1.0) {
    PetscInt n_bands = 0;
    for (b = 0; b < DT_BANDS_NUM; b++) if (bands->n_entries[b] > 0) n_bands++;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "dt bands: %" PetscInt_FMT " bands stored, coefficients computed %" PetscInt_FMT " times and restored %" PetscInt_FMT " times\n", n_bands, bands->n_stored, bands->n_restored));
    if (bands->check) PetscCheck(n_bands > 1 && bands->n_restored > 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "The time step bands were not used");
  }
  for (b = 0; b < DT_BANDS_NUM; b++) {
    for (i = 0; i < bands->n_entries[b]; i++) PetscCall(PetscFree(bands->entries[b][i].coeffs));
    PetscCall(PetscFree(bands->entries[b]));
    bands->n_entries[b] = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
