## Reuse with PCAIR:

If solving multiple linear systems, there are several ways to reuse components of the PCAIR preconditioner to reduce setup times:

#### 1) Setup once 

In PETSc the ``KSPSetReusePreconditioner`` flag can be set to ensure the preconditioner is only setup once, even if the entries or sparsity of the underlying matrix in the KSP is changed. This can be useful in time stepping problems where the linear system (or Jacobian for non-linear problems) does not change, or if the changes are small. 

#### 2) Reuse sparsity during setup

When solving a linear system where the matrix has the same sparsity pattern as in a previous solve, PCAIR can can reuse the CF splitting, repartitioning, symbolic matrix-matrix products and resulting sparsity throughout the hierarchy during the setup. This takes more memory (typically 2-5x the storage complexity) but significantly reduces the time required in subsequent setups (10-20x). If we have a PETSc matrix $\mathbf{A}$:

in Fortran:

     call KSPGetPC(ksp, pc, ierr)
     call PCSetType(pc, PCAIR, ierr)
     
     ! Before the first setup, we tell it to 
     ! store any data we need for reuse
     call PCAIRSetReuseSparsity(pc, PETSC_TRUE, ierr)

     ! First solve - the PCAIR will be setup
     call KSPSolve(ksp, b, x, ierr)
     
     ! ...[Modify entries in A but keep the same sparsity]

     ! Second solve - the PCAIR will be setup
     ! but reusing the same sparsity
     call KSPSolve(ksp, b, x, ierr)     

or in C:

     ierr = KSPGetPC(ksp, &pc);
     ierr = PCSetType(pc, PCAIR);

     // Before the first setup, we tell it to 
     // store any data we need for reuse
     ierr = PCAIRSetReuseSparsity(pc, PETSC_TRUE);

     // First solve - the PCAIR will be setup
     ierr = KSPSolve(ksp, b, x)
     
     // ...[Modify entries in A but keep the same sparsity]

     // Second solve - the PCAIR will be setup 
     // but reusing the same sparsity
     ierr = KSPSolve(ksp, b, x)

or in Python with petsc4py:

     pc = ksp.getPC()
     pc.setType("air")

     petsc_options = PETSc.Options()
     petsc_options['pc_air_reuse_sparsity'] = ''

     # First solve - the PCAIR will be setup
     ksp.solve(b,x)
     
     # ...[Modify entries in A but keep the same sparsity]

     # Second solve - the PCAIR will be setup 
     # but reusing the same sparsity
     ksp.solve(b,x)
     
or via the command line: ``-pc_type air -pc_air_reuse_sparsity``.

#### 3) Reuse sparsity and GMRES polynomial coefficients during setup with AIRG

When using AIRG in PCAIR (which is the default), in addition to reusing the sparsity in the setup, the GMRES polynomial coefficients can also be reused. This is useful if the matrix is very similar to that used in a previous solve and saves parallel reductions in the setup. Note that the GMRES polynomial coefficients are very sensitive to changes in the matrix, so this may not always give good results. To enable this we simply set another reuse flag before (or after) the first solve:

in Fortran:

     ! Before the first setup, we tell it to 
     ! store any data we need for reuse
     call PCAIRSetReuseSparsity(pc, PETSC_TRUE, ierr)

     ! Tell it to store the gmres polynomial coefficients
     call PCAIRSetReusePolyCoeffs(pc, PETSC_TRUE, ierr)        
    
     ! First solve - the PCAIR will be setup
     call KSPSolve(ksp, b, x, ierr)  
     
     ! ...[Modify entries in A but keep the same sparsity]

     ! Second solve - the PCAIR will be setup
     ! but reusing sparsity & polynomial coefficients
     call KSPSolve(ksp, b, x, ierr)     

or in C:

     // Before the first setup, we tell it to 
     // store any data we need for reuse
     ierr = PCAIRSetReuseSparsity(pc, PETSC_TRUE);

     ! Tell it to store the gmres polynomial coefficients
     ierr = PCAIRSetReusePolyCoeffs(pc, PETSC_TRUE);

     // First solve - the PCAIR will be setup
     ierr = KSPSolve(ksp, b, x)
     
     // ...[Modify entries in A but keep the same sparsity]

     // Second solve - the PCAIR will be setup
     // but reusing sparsity & polynomial coefficients
     ierr = KSPSolve(ksp, b, x)

or in Python with petsc4py:

     pc = ksp.getPC()
     pc.setType("air")

     petsc_options = PETSc.Options()
     petsc_options['pc_air_reuse_sparsity'] = ''
     petsc_options['pc_air_reuse_poly_coeffs'] = ''

     # First solve - the PCAIR will be setup
     ksp.solve(b,x)
     
     # ...[Modify entries in A but keep the same sparsity]

     # Second solve - the PCAIR will be setup 
     # but reusing sparsity & polynomial coefficients
     ksp.solve(b,x)

or via the command line: ``-pc_type air -pc_air_reuse_sparsity -pc_air_reuse_poly_coeffs``.

#### 4) Reuse when solving the same linear system with AIRG

Often an outer loop (e.g., eigenvalue) will require solving a series of different linear systems one after another, and then returning to the first linear system to start the loop again. If the linear systems are close enough to each other to get good performance from reusing the sparsity, but different enough from each other that reusing the GMRES polynomials coefficients from a different linear system gives poor performance, PCAIR allows the GMRES polynomial coefficients to be saved externally and then restored before a solve. 

This means we can cheaply generate the exact same preconditioner when periodically solving the same linear system. An example of this is given in ``tests/ex6f_getcoeffs.F90``, where we solve a linear system, store the resulting GMRES polynomial coefficients, reuse the sparsity to solve a different linear system, then reuse the sparsity and restore the GMRES polynomial coefficients to solve the first linear system again. We should note this type of reuse is not yet available in Python. 