/*   Tests -pc_pflareinv_poly_shift on a series of shifted systems (L + c_g I) x = b, where L is a
     steady 2D upwind finite difference advection-diffusion operator
     For each c_g we solve with:
       - fresh: PCPFLAREINV built from L + c_g I
       - shift: PCPFLAREINV built once from L with PCPFLAREINVSetPolyShift set, then for each g the
                Pmat is changed in place to L + c_g I, so PCPFLAREINV recomputes the Newton roots
                from the stored Hessenberg matrix rather than doing an Arnoldi
     As the Arnoldi of L + c_g I gives the same basis and the Hessenberg matrix of L shifted by c_g,
     the shifted roots should match the fresh roots, and give the same iteration count
     For large c_g the fresh roots are not accurate, as forming (L + c_g I) v in the Arnoldi and then
     removing its c_g v component loses the information about L to cancellation, whereas the
     shifted roots are computed from the Hessenberg matrix of L itself. So the roots are only
     compared for c_g up to -root_c_max (default 1), while the iteration counts are always compared
     ./pflareinv_poly_shift -pc_pflareinv_type newton
             : sweep with the default c_g, on a 2D grid of size 30 x 30
     Can control the grid size with -n (default 30)
     Can control the diffusion with -alpha (default 0.1)
     Can give the list of c_g with -c_values c1,c2,...
     Can instead change the diagonal by c_g (1 + 0.5 sin(row)) with -varying, where the shift is only
       an approximation, so only checks the solves converge
     Can control the relative tolerance on the difference between the shifted and fresh roots 
       with -root_tol (default sqrt(machine epsilon), 1e-2 in single precision)
     Can control the largest c_g whose roots are compared with -root_c_max (default 1)
     Can print the roots with -print_roots
     Can check with -check_reuse that, when -pc_pflareinv_reuse_poly_coeffs is also set, the coefficients
       never change, ie reuse takes priority over the shift
     The checks always done are:
       - after the sweep, going back to the Pmat L gives exactly the roots of the hierarchy built from L
       - changing to a different Pmat rebuilds the polynomial, giving the fresh roots
*/

static char help[] = "Tests updating a PCPFLAREINV Newton polynomial by shifting the stored Hessenberg matrix.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

#include "pflare.h"

static PetscErrorCode ComputeMat(DM da, Mat A, PetscReal alpha)
{
  PetscInt    i, j, mx, my, xs, ys, xm, ym, ncols;
  PetscScalar vals[5];
  MatStencil  row, cols[5];

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da, 0, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0));

  // Unit velocity at theta = pi/4, dimensionless upwinding
  const PetscReal u = PetscCosReal(PETSC_PI/4.0), v = PetscSinReal(PETSC_PI/4.0);

  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      row.i = i; row.j = j;
      ncols = 0;
      // Dirichlet on the inflow faces
      if (i == 0 || j == 0) {
        cols[0] = row; vals[0] = 1.0; ncols = 1;
      } else {
        cols[ncols].i = i; cols[ncols].j = j; vals[ncols++] = u + v + 4.0 * alpha;
        cols[ncols].i = i - 1; cols[ncols].j = j; vals[ncols++] = -u - alpha;
        cols[ncols].i = i; cols[ncols].j = j - 1; vals[ncols++] = -v - alpha;
        if (alpha != 0.0) {
          if (i < mx - 1) { cols[ncols].i = i + 1; cols[ncols].j = j; vals[ncols++] = -alpha; }
          if (j < my - 1) { cols[ncols].i = i; cols[ncols].j = j + 1; vals[ncols++] = -alpha; }
        }
      }
      PetscCall(MatSetValuesStencil(A, 1, &row, ncols, cols, vals, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Sets A = L + c I, or L + c diag(1 + 0.5 sin(row)) if varying
static PetscErrorCode SetShifted(Mat L, Mat A, PetscReal c, PetscBool varying)
{
  Vec          d;
  PetscInt     start, end, i;
  PetscScalar *d_array;

  PetscFunctionBeginUser;
  PetscCall(MatCopy(L, A, SAME_NONZERO_PATTERN));
  if (varying) {
    PetscCall(MatCreateVecs(A, NULL, &d));
    PetscCall(VecGetOwnershipRange(d, &start, &end));
    PetscCall(VecGetArray(d, &d_array));
    for (i = start; i < end; i++) d_array[i - start] = c * (1.0 + 0.5 * PetscSinReal((PetscReal)i));
    PetscCall(VecRestoreArray(d, &d_array));
    PetscCall(MatDiagonalSet(A, d, ADD_VALUES));
    PetscCall(VecDestroy(&d));
  } else {
    PetscCall(MatShift(A, c));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Solves and returns the iteration count, erroring if we didn't converge
static PetscErrorCode Solve(KSP ksp, Vec b, Vec x, PetscInt *its)
{
  KSPConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(VecSet(x, 0.0));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(KSPGetIterationNumber(ksp, its));
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  PetscCheck(reason > 0, PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, "Solve did not converge, reason %d", (int)reason);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Sorts roots stored as [real parts, imaginary parts] (column-major, rows x 2) by real
// then imaginary part, so roots can be compared independent of their Leja ordering
static PetscErrorCode SortRoots(const PetscReal *roots, PetscInt rows, PetscReal *sorted)
{
  PetscInt  i, j;
  PetscReal re, im;

  PetscFunctionBeginUser;
  for (i = 0; i < rows; i++) {
    sorted[2 * i] = roots[i];
    sorted[2 * i + 1] = roots[i + rows];
  }
  // Insertion sort, there are only a handful of roots
  for (i = 1; i < rows; i++) {
    re = sorted[2 * i]; im = sorted[2 * i + 1];
    j = i - 1;
    while (j >= 0 && (sorted[2 * j] > re || (sorted[2 * j] == re && sorted[2 * j + 1] > im))) {
      sorted[2 * (j + 1)] = sorted[2 * j];
      sorted[2 * (j + 1) + 1] = sorted[2 * j + 1];
      j--;
    }
    sorted[2 * (j + 1)] = re; sorted[2 * (j + 1) + 1] = im;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Returns the max difference between the roots of two PCs, relative to the largest root,
// or a negative value if they have a different number of roots
static PetscErrorCode CompareRoots(PC pc_a, PC pc_b, PetscReal *rel_diff)
{
  PetscReal *roots_a, *roots_b, *sorted_a, *sorted_b, max_diff = 0.0, max_root = 0.0;
  PetscInt   rows_a, cols_a, rows_b, cols_b, i;

  PetscFunctionBeginUser;
  PetscCall(PCPFLAREINVGetPolyCoeffs(pc_a, &roots_a, &rows_a, &cols_a));
  PetscCall(PCPFLAREINVGetPolyCoeffs(pc_b, &roots_b, &rows_b, &cols_b));
  PetscCheck(cols_a == 2 && cols_b == 2, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Expected Newton roots with two columns");
  if (rows_a != rows_b) {
    *rel_diff = -1.0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscMalloc2(2 * rows_a, &sorted_a, 2 * rows_b, &sorted_b));
  PetscCall(SortRoots(roots_a, rows_a, sorted_a));
  PetscCall(SortRoots(roots_b, rows_b, sorted_b));
  for (i = 0; i < 2 * rows_a; i++) {
    max_diff = PetscMax(max_diff, PetscAbsReal(sorted_a[i] - sorted_b[i]));
    max_root = PetscMax(max_root, PetscAbsReal(sorted_b[i]));
  }
  PetscCall(PetscFree2(sorted_a, sorted_b));
  *rel_diff = max_root > 0 ? max_diff / max_root : max_diff;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Prints the roots of a PC
static PetscErrorCode PrintRoots(PC pc, const char *name)
{
  PetscReal *roots;
  PetscInt   rows, cols, i;

  PetscFunctionBeginUser;
  PetscCall(PCPFLAREINVGetPolyCoeffs(pc, &roots, &rows, &cols));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  %s roots:", name));
  for (i = 0; i < rows; i++) PetscCall(PetscPrintf(PETSC_COMM_WORLD, " (%.10g, %.4g)", (double)roots[i], (double)roots[i + rows]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Copies the roots of a PC
static PetscErrorCode CopyRoots(PC pc, PetscReal **roots, PetscInt *rows, PetscInt *cols)
{
  PetscReal *pc_roots;

  PetscFunctionBeginUser;
  PetscCall(PCPFLAREINVGetPolyCoeffs(pc, &pc_roots, rows, cols));
  PetscCall(PetscMalloc1((*rows) * (*cols), roots));
  PetscCall(PetscArraycpy(*roots, pc_roots, (*rows) * (*cols)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Checks the roots of a PC are bitwise identical to the given ones
static PetscErrorCode CheckSameRoots(PC pc, const PetscReal *roots, PetscInt rows, PetscInt cols, const char *what)
{
  PetscReal *pc_roots;
  PetscInt   pc_rows, pc_cols, i;

  PetscFunctionBeginUser;
  PetscCall(PCPFLAREINVGetPolyCoeffs(pc, &pc_roots, &pc_rows, &pc_cols));
  PetscCheck(pc_rows == rows && pc_cols == cols, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "%s: the number of roots changed", what);
  for (i = 0; i < rows * cols; i++) {
    PetscCheck(pc_roots[i] == roots[i], PETSC_COMM_WORLD, PETSC_ERR_PLIB, "%s: the roots changed", what);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  KSP       ksp_fresh, ksp_shift;
  PC        pc_fresh, pc_shift;
  DM        da;
  Mat       L, A_fresh, A_shift, A_new;
  Vec       x, b;
  PetscInt  n = 30, n_c = 16, i_loc, its_fresh, its_shift, base_rows, base_cols;
  PetscReal alpha = 0.1, rel_diff, tol, root_c_max = 1.0;
  PetscReal c_values[16] = {1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e4, 1e6};
  PetscReal *base_roots;
  PetscBool varying = PETSC_FALSE, check_reuse = PETSC_FALSE, print_roots = PETSC_FALSE, poly_shift, flg;
  PCPFLAREINVType type;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  // Register the pflare types
  PCRegister_PFLARE();

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-alpha", &alpha, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-varying", &varying, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-check_reuse", &check_reuse, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-print_roots", &print_roots, NULL));
  PetscCall(PetscOptionsGetRealArray(NULL, NULL, "-c_values", c_values, &n_c, &flg));
  if (!flg) n_c = 8;

  // The roots of the shifted and fresh polynomials only agree to rounding, amplified by the 
  // conditioning of the (harmonic Ritz) eigenvalue problem
#if defined(PETSC_USE_REAL_SINGLE)
  tol = 1e-2;
#else
  tol = PetscSqrtReal(PETSC_MACHINE_EPSILON);
#endif
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-root_tol", &tol, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-root_c_max", &root_c_max, NULL));

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, n, n, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  PetscCall(DMCreateMatrix(da, &L));
  PetscCall(ComputeMat(da, L, alpha));
  PetscCall(MatDuplicate(L, MAT_COPY_VALUES, &A_fresh));
  PetscCall(MatDuplicate(L, MAT_COPY_VALUES, &A_shift));
  PetscCall(MatCreateVecs(L, &x, &b));
  PetscCall(VecSet(b, 1.0));

  // The shift PC, built once from L
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_shift));
  PetscCall(KSPSetOperators(ksp_shift, A_shift, A_shift));
  PetscCall(KSPGetPC(ksp_shift, &pc_shift));
  PetscCall(PCSetType(pc_shift, PCPFLAREINV));
  PetscCall(PCPFLAREINVSetType(pc_shift, PFLAREINV_NEWTON));
  PetscCall(PCPFLAREINVSetPolyShift(pc_shift, PETSC_TRUE));
  PetscCall(KSPSetFromOptions(ksp_shift));
  PetscCall(PCPFLAREINVGetPolyShift(pc_shift, &poly_shift));
  PetscCheck(poly_shift, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PCPFLAREINVGetPolyShift should return true");
  PetscCall(PCPFLAREINVGetType(pc_shift, &type));
  PetscCheck(type == PFLAREINV_NEWTON || type == PFLAREINV_NEWTON_NO_EXTRA, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "This test needs a Newton inverse type");

  // The fresh PC, rebuilt for every shift
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_fresh));
  PetscCall(KSPSetOperators(ksp_fresh, A_fresh, A_fresh));
  PetscCall(KSPGetPC(ksp_fresh, &pc_fresh));
  PetscCall(PCSetType(pc_fresh, PCPFLAREINV));
  PetscCall(PCPFLAREINVSetType(pc_fresh, type));
  PetscCall(KSPSetFromOptions(ksp_fresh));
  // The fresh PC must not shift or reuse anything
  PetscCall(PCPFLAREINVSetPolyShift(pc_fresh, PETSC_FALSE));
  PetscCall(PCPFLAREINVSetReusePolyCoeffs(pc_fresh, PETSC_FALSE));

  // Build from L
  PetscCall(Solve(ksp_shift, b, x, &its_shift));
  PetscCall(CopyRoots(pc_shift, &base_roots, &base_rows, &base_cols));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Iterations with the polynomial built from L: %" PetscInt_FMT "\n", its_shift));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%12s %8s %8s %16s\n", "c", "fresh", "shift", "rel root diff"));

  for (i_loc = 0; i_loc < n_c; i_loc++) {
    PetscCall(SetShifted(L, A_fresh, c_values[i_loc], varying));
    PetscCall(SetShifted(L, A_shift, c_values[i_loc], varying));

    // Fresh polynomial built from scratch
    // PCReset drops the operators, so set them again
    PetscCall(PCReset(pc_fresh));
    PetscCall(KSPSetOperators(ksp_fresh, A_fresh, A_fresh));
    PetscCall(Solve(ksp_fresh, b, x, &its_fresh));
    // Polynomial updated by shifting the stored Hessenberg matrix
    PetscCall(Solve(ksp_shift, b, x, &its_shift));
    PetscCall(CompareRoots(pc_shift, pc_fresh, &rel_diff));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%12.3e %8" PetscInt_FMT " %8" PetscInt_FMT " %16.3e\n", (double)c_values[i_loc], its_fresh, its_shift, (double)rel_diff));
    if (print_roots) {
      PetscCall(PrintRoots(pc_fresh, "fresh"));
      PetscCall(PrintRoots(pc_shift, "shift"));
    }

    if (check_reuse) {
      // Reuse takes priority, so the roots never change
      PetscCall(CheckSameRoots(pc_shift, base_roots, base_rows, base_cols, "With reuse"));
    } else if (!varying) {
      // A constant shift gives the fresh polynomial
      if (c_values[i_loc] <= root_c_max) {
        PetscCheck(rel_diff >= 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "The shifted polynomial has a different number of roots to the fresh one");
        PetscCheck(rel_diff <= tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "The shifted roots differ from the fresh roots by %g relative", (double)rel_diff);
      }
      PetscCheck(its_shift == its_fresh, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "The shifted polynomial took %" PetscInt_FMT " iterations rather than %" PetscInt_FMT, its_shift, its_fresh);
    }
  }

  // Going back to L, the shift is exactly zero, so we should get exactly the roots built from L
  PetscCall(MatCopy(L, A_shift, SAME_NONZERO_PATTERN));
  PetscCall(Solve(ksp_shift, b, x, &its_shift));
  PetscCall(CheckSameRoots(pc_shift, base_roots, base_rows, base_cols, "Going back to L"));

  // A different Pmat rebuilds the polynomial, giving the fresh roots
  if (!check_reuse) {
    PetscCall(MatDuplicate(L, MAT_COPY_VALUES, &A_new));
    PetscCall(MatShift(A_new, 1.0));
    PetscCall(MatCopy(A_new, A_fresh, SAME_NONZERO_PATTERN));
    PetscCall(KSPSetOperators(ksp_shift, A_new, A_new));
    PetscCall(Solve(ksp_shift, b, x, &its_shift));
    PetscCall(PCReset(pc_fresh));
    PetscCall(KSPSetOperators(ksp_fresh, A_fresh, A_fresh));
    PetscCall(Solve(ksp_fresh, b, x, &its_fresh));
    PetscCall(CompareRoots(pc_shift, pc_fresh, &rel_diff));
    PetscCheck(rel_diff >= 0 && rel_diff <= tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Rebuilding with a new Pmat should give the fresh roots, they differ by %g", (double)rel_diff);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Iterations with a new Pmat: %" PetscInt_FMT ", fresh %" PetscInt_FMT "\n", its_shift, its_fresh));
    PetscCall(MatDestroy(&A_new));
  }

  PetscCall(PetscFree(base_roots));
  PetscCall(KSPDestroy(&ksp_fresh));
  PetscCall(KSPDestroy(&ksp_shift));
  PetscCall(MatDestroy(&L));
  PetscCall(MatDestroy(&A_fresh));
  PetscCall(MatDestroy(&A_shift));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}
