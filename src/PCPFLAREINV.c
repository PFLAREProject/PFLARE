/*
  Definition and registration of the PC PFLARE INV type
  Largely just a wrapper around all the fortran 
  using a Mat
*/

// Include the petsc header files
#include <petsc/private/pcimpl.h>
#include "pflare.h"

// Defined in C_Fortran_Bindings.F90
PETSC_EXTERN void reset_inverse_mat_c(Mat *mat);
// coeffs_ptr/row_size/col_size are in/out:
//   *coeffs_ptr == NULL on entry  -> fresh: Fortran allocates, writes c_loc to *coeffs_ptr on return
//   *coeffs_ptr != NULL on entry  -> reuse: existing coefficients used, polynomial step skipped
PETSC_EXTERN void calculate_and_build_approximate_inverse_c(Mat *input_mat, PetscInt inverse_type, PetscInt order, \
                     PetscInt sparsity_order, PetscInt matrix_free_int, PetscInt diag_scale_polys_int, \
                     PetscInt subcomm_int, \
                     PetscReal **coeffs_ptr, PetscInt *row_size, PetscInt *col_size, \
                     Mat *inv_matrix);

// The types available as approximate inverses are (see include/pflare.h):
//
// PFLAREINV_POWER      - GMRES polynomial with the power basis 
// PFLAREINV_ARNOLDI    - GMRES polynomial with the arnoldi basis 
// PFLAREINV_NEWTON     - GMRES polynomial with the newton basis with extra roots for stability 
// PFLAREINV_NEWTON_NO_EXTRA     - GMRES polynomial with the newton basis without extra roots    
// PFLAREINV_NEUMANN    - Neumann polynomial
// PFLAREINV_SAI        - SAI - cannot be used matrix-free atm
// PFLAREINV_ISAI       - Incomplete SAI - cannot be used matrix-free atm
//                           ie a one-level restricted additive schwartz where each unknown is its own 
//                           subdomain with overlap given by the connectivity of that unknown in matrix powers
// PFLAREINV_WJACOBI    - Weighted Jacobi - The two Jacobi types really only exist for use in PCAIR, if you want to use 
//                           a Jacobi PC just use the existing one in petsc
// PFLAREINV_JACOBI     - Unweighted Jacobi - 

 /*
    Private context (data structure) for the PFLAREINV preconditioner.
 */
typedef struct {
   // Stores the mat which is our inverse
   Mat mat_inverse;

   // What type of inverse to apply
   int inverse_type;
   // The polynomial order
   int poly_order;
   // The power of the sparsity we use (if assembled)
   int inverse_sparsity_order;
   // Whether or not the mat_inverse is just a matshell and applied
   // matrix free
   PetscBool matrix_free;
   int subcomm_int;

   // Stored polynomial coefficients (column-major).
   // Populated after every PCSetUp call. Memory is either Fortran-allocated (from
   // calculate_and_build_approximate_inverse_c) or C-malloc'd (from SetPolyCoeffs);
   // in both cases a plain C free() is correct because Fortran allocate uses the C heap.
   // Populated/updated after every PCSetUp call
   PetscReal *poly_coeffs;
   PetscInt   poly_coeffs_rows;
   PetscInt   poly_coeffs_cols;
   // If PETSC_TRUE and SAME_NONZERO_PATTERN, skip coefficient recomputation
   PetscBool  reuse_poly_coeffs;

} PC_PFLAREINV;

// ~~~~~~~~~~

static PetscErrorCode PCReset_PFLAREINV_c(PC pc)
{
   PC_PFLAREINV *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;    
   reset_inverse_mat_c(&(inv_data->mat_inverse));
   // Free stored polynomial coefficients (Fortran-allocated or C-malloc'd, both freed with free())
   free(inv_data->poly_coeffs);
   inv_data->poly_coeffs      = NULL;
   inv_data->poly_coeffs_rows = 0;
   inv_data->poly_coeffs_cols = 0;
   // Note: reuse_poly_coeffs is intentionally NOT reset here (persists across resets)
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~
// Now all the get/set routines for options
// For explanation see the comments above the set routines
// ~~~~~~~~~~~~~~~~~~~~~

// Get routines

PetscErrorCode PCPFLAREINVGetPolyOrder(PC pc, PetscInt *poly_order)
{
   PetscFunctionBegin;
   PetscUseMethod(pc, "PCPFLAREINVGetPolyOrder_C", (PC, PetscInt *), (pc, poly_order));
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPFLAREINVGetPolyOrder_PFLAREINV(PC pc, PetscInt *poly_order)
{
   PC_PFLAREINV *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;    
   *poly_order = inv_data->poly_order;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

PetscErrorCode PCPFLAREINVGetSparsityOrder(PC pc, PetscInt *inverse_sparsity_order)
{
   PetscFunctionBegin;
   PetscUseMethod(pc, "PCPFLAREINVGetSparsityOrder_C", (PC, PetscInt *), (pc, inverse_sparsity_order));
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPFLAREINVGetSparsityOrder_PFLAREINV(PC pc, PetscInt *inverse_sparsity_order)
{
   PC_PFLAREINV *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;    
   *inverse_sparsity_order = inv_data->inverse_sparsity_order;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

PetscErrorCode PCPFLAREINVGetType(PC pc, PCPFLAREINVType *type)
{
   PetscFunctionBegin;
   PetscUseMethod(pc, "PCPFLAREINVGetType_C", (PC, PCPFLAREINVType *), (pc, type));
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPFLAREINVGetType_PFLAREINV(PC pc, PCPFLAREINVType *type)
{
   PC_PFLAREINV *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;    
   *type = (PCPFLAREINVType)inv_data->inverse_type;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

PetscErrorCode PCPFLAREINVGetMatrixFree(PC pc, PetscBool *flg)
{
   PetscFunctionBegin;
   PetscUseMethod(pc, "PCPFLAREINVGetMatrixFree_C", (PC, PetscBool *), (pc, flg));
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPFLAREINVGetMatrixFree_PFLAREINV(PC pc, PetscBool *flg)
{
   PC_PFLAREINV *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;   
   *flg = inv_data->matrix_free;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// Set routines

// This is the order of polynomial we use
// Default: 6
// -pc_pflareinv_poly_order 
PetscErrorCode PCPFLAREINVSetPolyOrder(PC pc, PetscInt poly_order)
{
   PetscFunctionBegin;
   PetscTryMethod(pc, "PCPFLAREINVSetPolyOrder_C", (PC, PetscInt), (pc, poly_order));
   PetscFunctionReturn(PETSC_SUCCESS);
} 

 static PetscErrorCode PCPFLAREINVSetPolyOrder_PFLAREINV(PC pc, PetscInt poly_order)
{
   PC_PFLAREINV *inv_data;
   PetscInt old_order;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;
   PetscCall(PCPFLAREINVGetPolyOrder(pc, &old_order));
   if (old_order == poly_order) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_PFLAREINV_c(pc));
   inv_data->poly_order = (int)poly_order;
   pc->setupcalled = PETSC_FALSE;  
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

// This is the order of sparsity we use if we assemble our approximate inverses
// Default: 1
// -pc_pflareinv_sparsity_order 
PetscErrorCode PCPFLAREINVSetSparsityOrder(PC pc, PetscInt inverse_sparsity_order)
{
   PetscFunctionBegin;
   PetscTryMethod(pc, "PCPFLAREINVSetSparsityOrder_C", (PC, PetscInt), (pc, inverse_sparsity_order));
   PetscFunctionReturn(PETSC_SUCCESS);
} 

 static PetscErrorCode PCPFLAREINVSetSparsityOrder_PFLAREINV(PC pc, PetscInt inverse_sparsity_order)
{
   PC_PFLAREINV *inv_data;
   PetscInt old_order;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;
   PetscCall(PCPFLAREINVGetSparsityOrder(pc, &old_order));
   if (old_order == inverse_sparsity_order) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_PFLAREINV_c(pc));
   inv_data->inverse_sparsity_order = (int)inverse_sparsity_order;
   pc->setupcalled = PETSC_FALSE;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

// What type of approximation do we use for our inverse - see PCPFLAREINVType
// Default: power
// -pc_pflareinv_type
PetscErrorCode PCPFLAREINVSetType(PC pc, PCPFLAREINVType type)
{
   PetscFunctionBegin;
   PetscTryMethod(pc, "PCPFLAREINVSetType_C", (PC, PCPFLAREINVType), (pc, type));
   PetscFunctionReturn(PETSC_SUCCESS);
} 

 static PetscErrorCode PCPFLAREINVSetType_PFLAREINV(PC pc, PCPFLAREINVType type)
{
   PC_PFLAREINV *inv_data;
   PCPFLAREINVType old_type;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;
   PetscCall(PCPFLAREINVGetType(pc, &old_type));
   if (old_type == type) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_PFLAREINV_c(pc));
   inv_data->inverse_type = type;  
   pc->setupcalled = PETSC_FALSE;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

// Do we apply our approximate inverse matrix-free?
// Default: false
// -pc_pflareinv_matrix_free
PetscErrorCode PCPFLAREINVSetMatrixFree(PC pc, PetscBool flg)
{
   PetscFunctionBegin;
   PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
   PetscTryMethod(pc, "PCPFLAREINVSetMatrixFree_C", (PC, PetscBool), (pc, flg));
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPFLAREINVSetMatrixFree_PFLAREINV(PC pc, PetscBool flg)
{
   PC_PFLAREINV *inv_data;
   PetscBool old_flag;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;
   PetscCall(PCPFLAREINVGetMatrixFree(pc, &old_flag));
   if (old_flag == flg) PetscFunctionReturn(PETSC_SUCCESS);
   PetscCall(PCReset_PFLAREINV_c(pc));   
   inv_data->matrix_free = flg;
   pc->setupcalled = PETSC_FALSE;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

// Get the stored polynomial coefficients and their dimensions
// This routine returns a pointer to the coefficients in the PCPFLAREINV object
// If you want to save/restore them later you will need to copy them yourself
// The pointer is valid only until the next PCSetUp or PCReset call
// This is different to the Fortran interface to this routine, which returns a copy
// in an allocatable array (which knows its own size)
PetscErrorCode PCPFLAREINVGetPolyCoeffs(PC pc, PetscReal **coeffs, PetscInt *rows, PetscInt *cols)
{
   PetscFunctionBegin;
   PetscUseMethod(pc, "PCPFLAREINVGetPolyCoeffs_C", (PC, PetscReal **, PetscInt *, PetscInt *), (pc, coeffs, rows, cols));
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPFLAREINVGetPolyCoeffs_PFLAREINV(PC pc, PetscReal **coeffs, PetscInt *rows, PetscInt *cols)
{
   PC_PFLAREINV *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;
   *coeffs = inv_data->poly_coeffs;
   *rows   = inv_data->poly_coeffs_rows;
   *cols   = inv_data->poly_coeffs_cols;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

// Set (copy in) polynomial coefficients; does not trigger rebuild
// This routine copies the data from coeffs_ptr into the PCPFLAREINV object
// row_size == poly_order+1, col_size == 1 (power/arnoldi/neumann) or 2 (newton)
// The caller's array is not referenced after this call and can be freed or modified
PetscErrorCode PCPFLAREINVSetPolyCoeffs(PC pc, PetscReal *coeffs, PetscInt rows, PetscInt cols)
{
   PetscFunctionBegin;
   PetscTryMethod(pc, "PCPFLAREINVSetPolyCoeffs_C", (PC, PetscReal *, PetscInt, PetscInt), (pc, coeffs, rows, cols));
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPFLAREINVSetPolyCoeffs_PFLAREINV(PC pc, PetscReal *coeffs, PetscInt rows, PetscInt cols)
{
   PC_PFLAREINV *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;
   free(inv_data->poly_coeffs);
   inv_data->poly_coeffs = (PetscReal *)malloc((size_t)rows * (size_t)cols * sizeof(PetscReal));
   PetscCheck(inv_data->poly_coeffs, PETSC_COMM_SELF, PETSC_ERR_MEM, "malloc failed in PCPFLAREINVSetPolyCoeffs");
   memcpy(inv_data->poly_coeffs, coeffs, (size_t)rows * (size_t)cols * sizeof(PetscReal));
   inv_data->poly_coeffs_rows = rows;
   inv_data->poly_coeffs_cols = cols;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

// Get the flag that controls whether polynomial coefficients are reused on next setup
PetscErrorCode PCPFLAREINVGetReusePolyCoeffs(PC pc, PetscBool *flg)
{
   PetscFunctionBegin;
   PetscUseMethod(pc, "PCPFLAREINVGetReusePolyCoeffs_C", (PC, PetscBool *), (pc, flg));
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPFLAREINVGetReusePolyCoeffs_PFLAREINV(PC pc, PetscBool *flg)
{
   PC_PFLAREINV *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;
   *flg = inv_data->reuse_poly_coeffs;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

// Set the flag that controls whether polynomial coefficients are reused on next setup
PetscErrorCode PCPFLAREINVSetReusePolyCoeffs(PC pc, PetscBool flg)
{
   PetscFunctionBegin;
   PetscTryMethod(pc, "PCPFLAREINVSetReusePolyCoeffs_C", (PC, PetscBool), (pc, flg));
   PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPFLAREINVSetReusePolyCoeffs_PFLAREINV(PC pc, PetscBool flg)
{
   PC_PFLAREINV *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;
   inv_data->reuse_poly_coeffs = flg;
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static PetscErrorCode PCApply_PFLAREINV_c(PC pc, Vec x, Vec y)
{
   PC_PFLAREINV *inv_data;
   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;
   // Just call a matmult
   PetscCall(MatMult(inv_data->mat_inverse, x, y));
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

static PetscErrorCode PCDestroy_PFLAREINV_c(PC pc)
{
   PC_PFLAREINV *inv_data;

   PetscFunctionBegin;

   inv_data = (PC_PFLAREINV *)pc->data;

   // Reset the mat
   PetscCall(PCReset_PFLAREINV_c(pc));

   // Then destroy the heap pointer
   PetscCall(PetscFree(inv_data));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetType_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetType_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetMatrixFree_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetMatrixFree_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetPolyOrder_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetPolyOrder_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetSparsityOrder_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetSparsityOrder_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetPolyCoeffs_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetPolyCoeffs_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetReusePolyCoeffs_C", NULL));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetReusePolyCoeffs_C", NULL));
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

static PetscErrorCode PCSetFromOptions_PFLAREINV_c(PC pc, PetscOptionItems PetscOptionsObject)
{
   PetscBool    flg;
   PCPFLAREINVType deflt, type;
   PetscInt poly_order, inverse_sparsity_order;
   PC_PFLAREINV *inv_data;

   PetscFunctionBegin;

   inv_data = (PC_PFLAREINV *)pc->data;

   PetscCall(PCPFLAREINVGetType(pc, &deflt));
   PetscOptionsHeadBegin(PetscOptionsObject, "PCPFLAREINV options");
   const char *const PCPFLAREINVTypes[] = {"POWER", "ARNOLDI", "NEWTON", "NEWTON_NO_EXTRA", "NEUMANN", "SAI", "ISAI", "WJACOBI", "JACOBI", "PCPFLAREINVType", "PFLAREINV_", NULL};
   PetscCall(PetscOptionsEnum("-pc_pflareinv_type", "Inverse type", "PCPFLAREINVSetType", PCPFLAREINVTypes, (PetscEnum)deflt, (PetscEnum *)&type, &flg));
   if (flg) PetscCall(PCPFLAREINVSetType(pc, type));
   PetscCall(PetscOptionsBool("-pc_pflareinv_matrix_free", "Apply matrix free", "PCPFLAREINVSetMatrixFree", inv_data->matrix_free, &inv_data->matrix_free, NULL));
   PetscCall(PetscOptionsBool("-pc_pflareinv_reuse_poly_coeffs", "Reuses gmres polynomial coefficients during setup", "PCPFLAREINVSetReusePolyCoeffs", inv_data->reuse_poly_coeffs, &inv_data->reuse_poly_coeffs, NULL));
   PetscCall(PetscOptionsInt("-pc_pflareinv_poly_order", "Order of polynomial", "PCPFLAREINVSetPolyOrder", inv_data->poly_order, &poly_order, &flg));
   if (flg) PetscCall(PCPFLAREINVSetPolyOrder(pc, poly_order));
   PetscCall(PetscOptionsInt("-pc_pflareinv_sparsity_order", "Sparsity order of assembled inverse", "PCPFLAREINVSetSparsityOrder", inv_data->inverse_sparsity_order, &inverse_sparsity_order, &flg));
   if (flg) PetscCall(PCPFLAREINVSetSparsityOrder(pc, inverse_sparsity_order));
   PetscOptionsHeadEnd();
   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

static PetscErrorCode PCSetUp_PFLAREINV_c(PC pc)
{
   PCPFLAREINVType type;
   MPI_Comm   comm; 
   PC_PFLAREINV *inv_data;
   PetscReal *new_coeffs;
   PetscInt   new_rows, new_cols;

   PetscFunctionBegin;

   inv_data = (PC_PFLAREINV *)pc->data;
   comm = PetscObjectComm((PetscObject)pc);

   // ~~~~~~~
   // Check options
   // ~~~~~~~
   PetscCall(PCPFLAREINVGetType(pc, &type));

   // SAI/ISAI can't be matrix free
   if (type == PFLAREINV_SAI || type == PFLAREINV_ISAI)
   {
      PetscCheck(!inv_data->matrix_free, comm, PETSC_ERR_ARG_WRONGSTATE, "PFLARE SAI/ISAI inverses cannot be applied matrix-free - Use PCASM");
   }     

   // We have to pass in an int
   int matrix_free_int = inv_data->matrix_free == PETSC_TRUE;   
   // We don't allow polynomail diagonal scaling through PCPFLAREINV
   // as the user could pass in a matshell as the matrix
   // It is the user's responsibility to have done the scaling beforehand
   // if they want
   int diag_scale_polys_int = 0;
   // Similarly for the subcomm
   int subcomm_int = 0;

   // If we haven't setup yet
   if (pc->setupcalled == 0)
   {
      // Fresh setup: pass NULL to let Fortran allocate and return the coefficient pointer
      new_coeffs = NULL; new_rows = 0; new_cols = 0;
      calculate_and_build_approximate_inverse_c(&(pc->pmat), \
            type, \
            inv_data->poly_order, inv_data->inverse_sparsity_order, \
            matrix_free_int, diag_scale_polys_int, subcomm_int, \
            &new_coeffs, &new_rows, &new_cols, \
            &(inv_data->mat_inverse));
      free(inv_data->poly_coeffs);
      inv_data->poly_coeffs      = new_coeffs;
      inv_data->poly_coeffs_rows = new_rows;
      inv_data->poly_coeffs_cols = new_cols;
   }
   else
   {
      // If we've got a different non-zero pattern we've got to 
      // start again       
      if (pc->flag == DIFFERENT_NONZERO_PATTERN)
      {
         // PCReset also frees poly_coeffs
         PetscCall(PCReset_PFLAREINV_c(pc));
         // Fresh: pass NULL to let Fortran allocate and return the coefficient pointer
         new_coeffs = NULL; new_rows = 0; new_cols = 0;
         calculate_and_build_approximate_inverse_c(&(pc->pmat), \
               type, \
               inv_data->poly_order, inv_data->inverse_sparsity_order, \
               matrix_free_int, diag_scale_polys_int, subcomm_int, \
               &new_coeffs, &new_rows, &new_cols, \
               &(inv_data->mat_inverse));
         inv_data->poly_coeffs      = new_coeffs;
         inv_data->poly_coeffs_rows = new_rows;
         inv_data->poly_coeffs_cols = new_cols;
      }
      else if (pc->flag == SAME_NONZERO_PATTERN)
      {
         // We don't call reset on the pc here so it reuses the sparsity of mat_inverse
         // Optionally reuse stored polynomial coefficients:
         //   poly_coeffs != NULL and reuse flag set  ->  pass the existing pointer (reuse path)
         //   otherwise                               ->  pass NULL so Fortran computes fresh ones
         if (!(inv_data->reuse_poly_coeffs == PETSC_TRUE && inv_data->poly_coeffs != NULL)) {
            // Fresh: free old coefficients so poly_coeffs is NULL going into the call
            free(inv_data->poly_coeffs);
            inv_data->poly_coeffs      = NULL;
            inv_data->poly_coeffs_rows = 0;
            inv_data->poly_coeffs_cols = 0;
         }
         // Pass poly_coeffs by address: NULL -> fresh allocation on return; non-NULL -> reuse
         calculate_and_build_approximate_inverse_c(&(pc->pmat), \
               type, \
               inv_data->poly_order, inv_data->inverse_sparsity_order, \
               matrix_free_int, diag_scale_polys_int, subcomm_int, \
               &inv_data->poly_coeffs, &inv_data->poly_coeffs_rows, &inv_data->poly_coeffs_cols, \
               &(inv_data->mat_inverse));
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

static PetscErrorCode PCView_PFLAREINV_c(PC pc, PetscViewer viewer)
{

   PC_PFLAREINV *inv_data;
   PetscBool  iascii;
   PCPFLAREINVType type;

   PetscFunctionBegin;

   inv_data = (PC_PFLAREINV *)pc->data;

   // Print out details about our PFLAREINV
   PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));

   if (iascii) {
      PetscCall(PCPFLAREINVGetType(pc, &type));

      // What type of inverse
      if (type == PFLAREINV_POWER)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  GMRES polynomial, power basis, order %i \n", inv_data->poly_order));
      }
      else if (type == PFLAREINV_ARNOLDI)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  GMRES polynomial, arnoldi basis, order %i \n", inv_data->poly_order));
      }
      else if (type == PFLAREINV_NEWTON)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  GMRES polynomial, newton basis with extra roots, order %i \n", inv_data->poly_order));
      }
      else if (type == PFLAREINV_NEWTON_NO_EXTRA)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  GMRES polynomial, newton basis without extra roots, order %i \n", inv_data->poly_order));
      }
      else if (type == PFLAREINV_SAI)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  SAI \n"));
      }
      else if (type == PFLAREINV_ISAI)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  ISAI \n"));
      }
      else if (type == PFLAREINV_NEUMANN)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  Neumann polynomial, order %i \n", inv_data->poly_order));
      }
      else if (type == PFLAREINV_WJACOBI)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  Weighted Jacobi \n"));
      }
      else if (type == PFLAREINV_JACOBI)
      {
         PetscCall(PetscViewerASCIIPrintf(viewer, "  Unweighted Jacobi \n"));
      }
      if (type != PFLAREINV_WJACOBI && type != PFLAREINV_JACOBI)
      {
         // If matrix-free or not
         if (inv_data->matrix_free)
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "  matrix-free inverse \n"));
         }
         else
         {
            PetscCall(PetscViewerASCIIPrintf(viewer, "  assembled inverse, sparsity order %i\n", inv_data->inverse_sparsity_order));
         }
         if (inv_data->reuse_poly_coeffs) PetscCall(PetscViewerASCIIPrintf(viewer, "  Reusing gmres polynomial coefficients during setup \n"));
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

// Creates the structure we need for this PC
PETSC_EXTERN PetscErrorCode PCCreate_PFLAREINV(PC pc)
{
   // Create our data structure on the heap
   PC_PFLAREINV *inv_data;

   PetscFunctionBegin;

   // Now we call petsc fortran routines from this PC
   // so we have to have made sure this is called
   PetscCall(PetscInitializeFortran());

   PetscCall(PetscNew(&inv_data));
   pc->data = (void *)inv_data;   

   // ~~~~~~~~~~~~
   // Default options for the PCPFLAREINV
   // ~~~~~~~~~~~~

   // Have to be very careful in the Fortran routines 
   // with any matrix that has been set to null in C
   // in petsc > 3.22
   // See calculate_and_build_approximate_inverse in 
   // Approx_Inverse_Setup.F90
   inv_data->mat_inverse = NULL;   
   // What type of inverse to apply
   // Default to gmres polynomial with the Arnoldi basis
   inv_data->inverse_type = PFLAREINV_ARNOLDI;
   // The polynomial order
   inv_data->poly_order = 6;
   // The power of the sparsity we use (if assembled)
   inv_data->inverse_sparsity_order = 1;
   // Whether or not the mat_inverse is just a matshell and applied
   // matrix free
   inv_data->matrix_free = PETSC_FALSE;
   // Polynomial coefficient storage (initially empty)
   inv_data->poly_coeffs       = NULL;
   inv_data->poly_coeffs_rows  = 0;
   inv_data->poly_coeffs_cols  = 0;
   inv_data->reuse_poly_coeffs = PETSC_FALSE;

   // Set the method functions
   pc->ops->apply               = PCApply_PFLAREINV_c;
   pc->ops->setup               = PCSetUp_PFLAREINV_c;
   pc->ops->destroy             = PCDestroy_PFLAREINV_c;
   pc->ops->view                = PCView_PFLAREINV_c;  
   pc->ops->reset               = PCReset_PFLAREINV_c;
   pc->ops->setfromoptions      = PCSetFromOptions_PFLAREINV_c;

   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetType_C", PCPFLAREINVSetType_PFLAREINV));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetType_C", PCPFLAREINVGetType_PFLAREINV));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetMatrixFree_C", PCPFLAREINVSetMatrixFree_PFLAREINV));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetMatrixFree_C", PCPFLAREINVGetMatrixFree_PFLAREINV));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetPolyOrder_C", PCPFLAREINVSetPolyOrder_PFLAREINV));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetPolyOrder_C", PCPFLAREINVGetPolyOrder_PFLAREINV));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetSparsityOrder_C", PCPFLAREINVSetSparsityOrder_PFLAREINV));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetSparsityOrder_C", PCPFLAREINVGetSparsityOrder_PFLAREINV));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetPolyCoeffs_C", PCPFLAREINVGetPolyCoeffs_PFLAREINV));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetPolyCoeffs_C", PCPFLAREINVSetPolyCoeffs_PFLAREINV));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetReusePolyCoeffs_C", PCPFLAREINVGetReusePolyCoeffs_PFLAREINV));
   PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetReusePolyCoeffs_C", PCPFLAREINVSetReusePolyCoeffs_PFLAREINV));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~

// Registers the PC type
PETSC_EXTERN void PCRegister_PFLAREINV()
{
   PetscCallVoid(PCRegister("pflareinv", PCCreate_PFLAREINV));
}