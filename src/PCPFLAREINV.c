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
PETSC_EXTERN void calculate_and_build_approximate_inverse_c(Mat *input_mat, PetscInt inverse_type, PetscInt order, \
                     PetscInt sparsity_order, PetscInt matrix_free_int, \
                     PetscInt subcomm_int, Mat *inv_matrix);

// The types available as approximate inverses are (see include/pflare.h):
//
// PFLAREINV_POWER      - GMRES polynomial with the power basis 
// PFLAREINV_ARNOLDI    - GMRES polynomial with the arnoldi basis 
// PFLAREINV_NEWTON     - GMRES polynomial with the newton basis with extra roots for stability - can only be used matrix-free atm      
// PFLAREINV_NEWTON_NO_EXTRA     - GMRES polynomial with the newton basis without extra roots - can only be used matrix-free atm      
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

} PC_PFLAREINV;

// ~~~~~~~~~~

static PetscErrorCode PCReset_PFLAREINV_c(PC pc)
{
   PC_PFLAREINV *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;    
   reset_inverse_mat_c(&(inv_data->mat_inverse));
   PetscFunctionReturn(0);
}

// ~~~~~~~~~~~~~~~~~~~~~
// Now all the get/set routines for options
// For explanation see the comments above the set routines
// ~~~~~~~~~~~~~~~~~~~~~

// Get routines

PetscErrorCode PCPFLAREINVGetOrder(PC pc, PetscInt *poly_order)
{
   PetscFunctionBegin;
   PetscUseMethod(pc, "PCPFLAREINVGetOrder_C", (PC, PetscInt *), (pc, poly_order));
   PetscFunctionReturn(0);
}

static PetscErrorCode PCPFLAREINVGetOrder_PFLAREINV(PC pc, PetscInt *poly_order)
{
   PC_PFLAREINV *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;    
   *poly_order = inv_data->poly_order;
   PetscFunctionReturn(0);
}

// ~~~~~~~~~~

PetscErrorCode PCPFLAREINVGetSparsityOrder(PC pc, PetscInt *inverse_sparsity_order)
{
   PetscFunctionBegin;
   PetscUseMethod(pc, "PCPFLAREINVGetSparsityOrder_C", (PC, PetscInt *), (pc, inverse_sparsity_order));
   PetscFunctionReturn(0);
}

static PetscErrorCode PCPFLAREINVGetSparsityOrder_PFLAREINV(PC pc, PetscInt *inverse_sparsity_order)
{
   PC_PFLAREINV *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;    
   *inverse_sparsity_order = inv_data->inverse_sparsity_order;
   PetscFunctionReturn(0);
}

// ~~~~~~~~~~

PetscErrorCode PCPFLAREINVGetType(PC pc, PCPFLAREINVType *type)
{
   PetscFunctionBegin;
   PetscUseMethod(pc, "PCPFLAREINVGetType_C", (PC, PCPFLAREINVType *), (pc, type));
   PetscFunctionReturn(0);
}

static PetscErrorCode PCPFLAREINVGetType_PFLAREINV(PC pc, PCPFLAREINVType *type)
{
   PC_PFLAREINV *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;    
   *type = (PCPFLAREINVType)inv_data->inverse_type;
   PetscFunctionReturn(0);
}

// ~~~~~~~~~~

PetscErrorCode PCPFLAREINVGetMatrixFree(PC pc, PetscBool *flg)
{
   PetscFunctionBegin;
   PetscUseMethod(pc, "PCPFLAREINVGetMatrixFree_C", (PC, PetscBool *), (pc, flg));
   PetscFunctionReturn(0);
}

static PetscErrorCode PCPFLAREINVGetMatrixFree_PFLAREINV(PC pc, PetscBool *flg)
{
   PC_PFLAREINV *inv_data;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;   
   *flg = inv_data->matrix_free;
   PetscFunctionReturn(0);
}

// Set routines

// This is the order of polynomial we use
// Default: 6
// -pc_pflareinv_order 
PetscErrorCode PCPFLAREINVSetOrder(PC pc, PetscInt poly_order)
{
   PetscFunctionBegin;
   PetscTryMethod(pc, "PCPFLAREINVSetOrder_C", (PC, PetscInt), (pc, poly_order));
   PetscFunctionReturn(0);
} 

 static PetscErrorCode PCPFLAREINVSetOrder_PFLAREINV(PC pc, PetscInt poly_order)
{
   PC_PFLAREINV *inv_data;
   PetscInt old_order;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;
   PCPFLAREINVGetOrder(pc, &old_order);
   if (old_order == poly_order) PetscFunctionReturn(0);
   PCReset_PFLAREINV_c(pc);
   inv_data->poly_order = (int)poly_order;  
   PetscFunctionReturn(0);
}

// ~~~~~~~~~~

// This is the order of sparsity we use if we assemble our approximate inverses
// Default: 1
// -pc_pflareinv_sparsity_order 
PetscErrorCode PCPFLAREINVSetSparsityOrder(PC pc, PetscInt inverse_sparsity_order)
{
   PetscFunctionBegin;
   PetscTryMethod(pc, "PCPFLAREINVSetSparsityOrder_C", (PC, PetscInt), (pc, inverse_sparsity_order));
   PetscFunctionReturn(0);
} 

 static PetscErrorCode PCPFLAREINVSetSparsityOrder_PFLAREINV(PC pc, PetscInt inverse_sparsity_order)
{
   PC_PFLAREINV *inv_data;
   PetscInt old_order;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;
   PCPFLAREINVGetSparsityOrder(pc, &old_order);
   if (old_order == inverse_sparsity_order) PetscFunctionReturn(0);
   PCReset_PFLAREINV_c(pc);
   inv_data->inverse_sparsity_order = (int)inverse_sparsity_order;  
   PetscFunctionReturn(0);
}

// ~~~~~~~~~~

// What type of approximation do we use for our inverse - see PCPFLAREINVType
// Default: power
// -pc_pflareinv_type
PetscErrorCode PCPFLAREINVSetType(PC pc, PCPFLAREINVType type)
{
   PetscFunctionBegin;
   PetscTryMethod(pc, "PCPFLAREINVSetType_C", (PC, PCPFLAREINVType), (pc, type));
   PetscFunctionReturn(0);
} 

 static PetscErrorCode PCPFLAREINVSetType_PFLAREINV(PC pc, PCPFLAREINVType type)
{
   PC_PFLAREINV *inv_data;
   PCPFLAREINVType old_type;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;
   PCPFLAREINVGetType(pc, &old_type);
   if (old_type == type) PetscFunctionReturn(0);
   PCReset_PFLAREINV_c(pc);
   inv_data->inverse_type = type;  
   PetscFunctionReturn(0);
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
   PetscFunctionReturn(0);
}

static PetscErrorCode PCPFLAREINVSetMatrixFree_PFLAREINV(PC pc, PetscBool flg)
{
   PC_PFLAREINV *inv_data;
   PetscBool old_flag;

   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;
   PCPFLAREINVGetMatrixFree(pc, &old_flag);
   if (old_flag == flg) PetscFunctionReturn(0);
   PCReset_PFLAREINV_c(pc);   
   inv_data->matrix_free = flg;
   PetscFunctionReturn(0);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static PetscErrorCode PCApply_PFLAREINV_c(PC pc, Vec x, Vec y)
{
   PC_PFLAREINV *inv_data;
   PetscFunctionBegin;
   inv_data = (PC_PFLAREINV *)pc->data;
   // Just call a matmult
   MatMult(inv_data->mat_inverse, x, y);
   PetscFunctionReturn(0);
}

// ~~~~~~~~~~

static PetscErrorCode PCDestroy_PFLAREINV_c(PC pc)
{
   PC_PFLAREINV *inv_data;

   PetscFunctionBegin;

   inv_data = (PC_PFLAREINV *)pc->data;

   // Reset the mat
   PCReset_PFLAREINV_c(pc);

   // Then destroy the heap pointer
   PetscCall(PetscFree(inv_data));
   PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetType_C", NULL);
   PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetType_C", NULL);
   PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetMatrixFree_C", NULL);
   PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetMatrixFree_C", NULL);
   PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetOrder_C", NULL);
   PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetOrder_C", NULL);
   PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetSparsityOrder_C", NULL);
   PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetSparsityOrder_C", NULL);       
   PetscFunctionReturn(0);
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

   PCPFLAREINVGetType(pc, &deflt);
   PetscOptionsHeadBegin(PetscOptionsObject, "PCPFLAREINV options"); 
   const char *const PCPFLAREINVTypes[] = {"POWER", "ARNOLDI", "NEWTON", "NEWTON_NO_EXTRA", "NEUMANN", "SAI", "ISAI", "WJACOBI", "JACOBI", "PCPFLAREINVType", "PFLAREINV_", NULL};
   PetscOptionsEnum("-pc_pflareinv_type", "Inverse type", "PCPFLAREINVSetType", PCPFLAREINVTypes, (PetscEnum)deflt, (PetscEnum *)&type, &flg);
   if (flg) PCPFLAREINVSetType(pc, type);
   PetscOptionsBool("-pc_pflareinv_matrix_free", "Apply matrix free", "PCPFLAREINVSetMatrixFree", inv_data->matrix_free, &inv_data->matrix_free, NULL);
   PetscOptionsInt("-pc_pflareinv_order", "Order of polynomial", "PCPFLAREINVSetOrder", inv_data->poly_order, &poly_order, &flg);
   if (flg) PCPFLAREINVSetOrder(pc, poly_order);
   PetscOptionsInt("-pc_pflareinv_sparsity_order", "Sparsity order of assembled inverse", "PCPFLAREINVSetSparsityOrder", inv_data->inverse_sparsity_order, &inverse_sparsity_order, &flg);
   if (flg) PCPFLAREINVSetSparsityOrder(pc, inverse_sparsity_order);     
   PetscOptionsHeadEnd();
   PetscFunctionReturn(0);
}

// ~~~~~~~~~~

static PetscErrorCode PCSetUp_PFLAREINV_c(PC pc)
{
   PCPFLAREINVType type;
   MPI_Comm   comm; 
   PC_PFLAREINV *inv_data;

   PetscFunctionBegin;

   inv_data = (PC_PFLAREINV *)pc->data;
   comm = PetscObjectComm((PetscObject)pc);

   // ~~~~~~~
   // Check options
   // ~~~~~~~
   PCPFLAREINVGetType(pc, &type);

   // Newton has to be matrix free
   if (type == PFLAREINV_NEWTON || type == PFLAREINV_NEWTON_NO_EXTRA)
   {
      PetscCheck(inv_data->matrix_free, comm, PETSC_ERR_ARG_WRONGSTATE, "GMRES polynomial with Newton basis must be applied matrix-free");
   }
   // SAI/ISAI can't be matrix free
   if (type == PFLAREINV_SAI || type == PFLAREINV_ISAI)
   {
      PetscCheck(!inv_data->matrix_free, comm, PETSC_ERR_ARG_WRONGSTATE, "PFLARE SAI/ISAI inverses cannot be applied matrix-free - Use PCASM");
   }     

   // We have to pass in an int
   int matrix_free_int = inv_data->matrix_free == PETSC_TRUE;   

   // If we haven't setup yet
   if (pc->setupcalled == 0)
   {   
      // Build the polynomial inverse as a Mat
      calculate_and_build_approximate_inverse_c(&(pc->pmat), \
            type, \
            inv_data->poly_order, inv_data->inverse_sparsity_order, \
            matrix_free_int, 0, \
            &(inv_data->mat_inverse));      
   }
   else
   {
      // If we've got a different non-zero pattern we've got to 
      // start again       
      if (pc->flag == DIFFERENT_NONZERO_PATTERN)
      {
         PCReset_PFLAREINV_c(pc);
         // Build the polynomial inverse as a Mat
         calculate_and_build_approximate_inverse_c(&(pc->pmat), \
               type, \
               inv_data->poly_order, inv_data->inverse_sparsity_order, \
               matrix_free_int, 0, \
               &(inv_data->mat_inverse));          
      }
      else if (pc->flag == SAME_NONZERO_PATTERN)
      {
         // We don't call reset on the pc here so it reuses the sparsity
         // Build the polynomial inverse as a Mat
         calculate_and_build_approximate_inverse_c(&(pc->pmat), \
               type, \
               inv_data->poly_order, inv_data->inverse_sparsity_order, \
               matrix_free_int, 0, \
               &(inv_data->mat_inverse));            
      }
   }

   PetscFunctionReturn(0);
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
   PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii);

   if (iascii) {
      PCPFLAREINVGetType(pc, &type);

      // What type of inverse
      if (type == PFLAREINV_POWER)
      {
         PetscViewerASCIIPrintf(viewer, "  GMRES polynomial, power basis, order %i \n", inv_data->poly_order);
      }
      else if (type == PFLAREINV_ARNOLDI)
      {
         PetscViewerASCIIPrintf(viewer, "  GMRES polynomial, arnoldi basis, order %i \n", inv_data->poly_order);
      }
      else if (type == PFLAREINV_NEWTON)
      {
         PetscViewerASCIIPrintf(viewer, "  GMRES polynomial, newton basis with extra roots, order %i \n", inv_data->poly_order); 
      }
      else if (type == PFLAREINV_NEWTON_NO_EXTRA)
      {
         PetscViewerASCIIPrintf(viewer, "  GMRES polynomial, newton basis without extra roots, order %i \n", inv_data->poly_order);               
      }
      else if (type == PFLAREINV_SAI)
      {
         PetscViewerASCIIPrintf(viewer, "  SAI \n");      
      }
      else if (type == PFLAREINV_ISAI)
      {
         PetscViewerASCIIPrintf(viewer, "  ISAI \n");      
      }      
      else if (type == PFLAREINV_NEUMANN)
      {
         PetscViewerASCIIPrintf(viewer, "  Neumann polynomial, order %i \n", inv_data->poly_order);      
      }     
      else if (type == PFLAREINV_WJACOBI)
      {
         PetscViewerASCIIPrintf(viewer, "  Weighted Jacobi \n");      
      }
      else if (type == PFLAREINV_JACOBI)
      {
         PetscViewerASCIIPrintf(viewer, "  Unweighted Jacobi \n");      
      }                  
      if (type != PFLAREINV_WJACOBI && type != PFLAREINV_JACOBI)
      {
         // If matrix-free or not
         if (inv_data->matrix_free)
         {
            PetscViewerASCIIPrintf(viewer, "  matrix-free inverse \n");
         }
         else
         {
            PetscViewerASCIIPrintf(viewer, "  assembled inverse, sparsity order %i\n", inv_data->inverse_sparsity_order);
         }
      }
   }

   PetscFunctionReturn(0);
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
   PetscInitializeFortran();

   PetscNew(&inv_data);
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
   // Default to gmres polynomial with the power basis
   inv_data->inverse_type = PFLAREINV_POWER;
   // The polynomial order
   inv_data->poly_order = 6;
   // The power of the sparsity we use (if assembled)
   inv_data->inverse_sparsity_order = 1;
   // Whether or not the mat_inverse is just a matshell and applied
   // matrix free
   inv_data->matrix_free = PETSC_FALSE;   

   // Set the method functions
   pc->ops->apply               = PCApply_PFLAREINV_c;
   pc->ops->setup               = PCSetUp_PFLAREINV_c;
   pc->ops->destroy             = PCDestroy_PFLAREINV_c;
   pc->ops->view                = PCView_PFLAREINV_c;  
   pc->ops->reset               = PCReset_PFLAREINV_c;
   pc->ops->setfromoptions      = PCSetFromOptions_PFLAREINV_c;

   PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetType_C", PCPFLAREINVSetType_PFLAREINV);
   PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetType_C", PCPFLAREINVGetType_PFLAREINV);
   PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetMatrixFree_C", PCPFLAREINVSetMatrixFree_PFLAREINV);
   PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetMatrixFree_C", PCPFLAREINVGetMatrixFree_PFLAREINV);
   PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetOrder_C", PCPFLAREINVSetOrder_PFLAREINV);
   PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetOrder_C", PCPFLAREINVGetOrder_PFLAREINV);
   PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVSetSparsityOrder_C", PCPFLAREINVSetSparsityOrder_PFLAREINV);
   PetscObjectComposeFunction((PetscObject)pc, "PCPFLAREINVGetSparsityOrder_C", PCPFLAREINVGetSparsityOrder_PFLAREINV);     

   PetscFunctionReturn(0);
}

// ~~~~~~~~~~

// Registers the PC type
PETSC_EXTERN void PCRegister_PFLAREINV()
{
   PCRegister("pflareinv", PCCreate_PFLAREINV);
}