## Reuse in PFLARE

If solving multiple linear systems, there are several ways to reuse components of PFLARE preconditioners (PCAIR and PCPFLAREINV) to reduce setup times:

#### 1) Setup once

In PETSc the ``KSPSetReusePreconditioner`` flag can be set to ensure the preconditioner is only setup once, even if the entries or sparsity of the underlying matrix in the KSP is changed. This can be useful in time stepping problems where the linear system (or Jacobian for non-linear problems) does not change, or if the changes are small.

#### 2) Reuse sparsity during setup

When solving a linear system where the matrix has the same sparsity pattern as in a previous solve, PCAIR can reuse the CF splitting, repartitioning, symbolic matrix-matrix products and resulting sparsity throughout the hierarchy during the setup. This takes more memory (typically 2-5x the storage complexity) but significantly reduces the time required in subsequent setups (10-20x). If we have a PETSc matrix $\mathbf{A}$:

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

#### 3) Reuse GMRES polynomial coefficients during setup

When the matrix is very similar to that used in a previous solve, the GMRES polynomial coefficients can be reused to save the parallel polynomial iteration in the setup. Note that the GMRES polynomial coefficients are very sensitive to changes in the matrix, so this may not always give good results.

**With PCAIR (AIRG):** In addition to reusing the polynomial coefficients, the sparsity must also be set to reuse (see section 2 above). To enable both:

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

**With PCPFLAREINV:** The sparsity of the assembled approximate inverse is determined by powers of A, so polynomial coefficient reuse requires the sparsity pattern of A to be unchanged between solves. There is no separate sparsity-reuse flag; only the polynomial coefficients need to be flagged for reuse:

in Fortran:

     call KSPGetPC(ksp, pc, ierr)
     call PCSetType(pc, PCPFLAREINV, ierr)

     ! Tell it to reuse the gmres polynomial coefficients
     call PCPFLAREINVSetReusePolyCoeffs(pc, PETSC_TRUE, ierr)

     ! First solve - the PCPFLAREINV will be setup
     call KSPSolve(ksp, b, x, ierr)

     ! ...[Modify entries in A]

     ! Second solve - the PCPFLAREINV will be setup
     ! but reusing the polynomial coefficients
     call KSPSolve(ksp, b, x, ierr)

or in C:

     ierr = KSPGetPC(ksp, &pc);
     ierr = PCSetType(pc, PCPFLAREINV);

     // Tell it to reuse the gmres polynomial coefficients
     ierr = PCPFLAREINVSetReusePolyCoeffs(pc, PETSC_TRUE);

     // First solve - the PCPFLAREINV will be setup
     ierr = KSPSolve(ksp, b, x);

     // ...[Modify entries in A]

     // Second solve - the PCPFLAREINV will be setup
     // but reusing the polynomial coefficients
     ierr = KSPSolve(ksp, b, x);

or via the command line: ``-pc_type pflareinv -pc_pflareinv_reuse_poly_coeffs``.

#### 4) Reuse when solving the same linear system

Often an outer loop (e.g., eigenvalue) will require solving a series of different linear systems one after another, and then returning to the first linear system to start the loop again. If the linear systems are close enough to each other to get good performance from reusing sparsity, but different enough from each other that reusing the GMRES polynomial coefficients from a different linear system gives poor performance, both PCAIR and PCPFLAREINV allow the GMRES polynomial coefficients to be saved externally and then restored before a solve.

This means we can cheaply generate the exact same preconditioner when periodically solving the same linear system. Examples of this are given in ``tests/ex6f_getcoeffs.F90`` and ``python/ex6f_getcoeffs.py``, which demonstrate this for both PCAIR and PCPFLAREINV.

**With PCAIR (AIRG):** The sparsity should also be reused alongside the polynomial coefficients:

in Fortran:

     call KSPGetPC(ksp, pc, ierr)
     call PCSetType(pc, PCAIR, ierr)
     call PCAIRSetReuseSparsity(pc, PETSC_TRUE, ierr)

     ! First solve - the PCAIR will be setup
     call KSPSolve(ksp, b, x, ierr)

     ! Save the polynomial coefficients from the first solve
     call PCAIRGetNumLevels(pc, num_levels, ierr)
     do petsc_level = 1, num_levels
        call PCAIRGetPolyCoeffs(pc, petsc_level, COEFFS_INV_AFF, coeffs_levels(petsc_level)%coeffs, ierr)
     end do

     ! ...[Modify entries in A to get a different linear system]

     ! Second solve - the PCAIR will be setup with reuse for the new system
     call KSPSolve(ksp, b2, x2, ierr)

     ! ...[Restore the original A]

     ! Restore the saved polynomial coefficients and tell PCAIR to reuse them
     do petsc_level = 1, num_levels
        call PCAIRSetPolyCoeffs(pc, petsc_level, COEFFS_INV_AFF, coeffs_levels(petsc_level)%coeffs, ierr)
     end do
     call PCAIRSetReusePolyCoeffs(pc, PETSC_TRUE, ierr)

     ! Third solve - reproduces the preconditioner from the first solve
     call KSPSolve(ksp, b, x, ierr)

or in C:

     ierr = KSPGetPC(ksp, &pc);
     ierr = PCSetType(pc, PCAIR);
     ierr = PCAIRSetReuseSparsity(pc, PETSC_TRUE);

     // First solve - the PCAIR will be setup
     ierr = KSPSolve(ksp, b, x);

     // ...[Modify entries in A to get a different linear system]

     // Second solve - the PCAIR will be setup with reuse for the new system
     ierr = KSPSolve(ksp, b2, x2);

     // ...[Restore the original A]

     // Restore the saved polynomial coefficients and tell PCAIR to reuse them
     ierr = PCAIRSetReusePolyCoeffs(pc, PETSC_TRUE);

     // Third solve - reproduces the preconditioner from the first solve
     ierr = KSPSolve(ksp, b, x);

or in Python with petsc4py:

     import pflare

     pc = ksp.getPC()
     pc.setType("air")

     # Reuse sparsity must be set via the direct API after ksp.setFromOptions()
     pflare.pcair_set_reuse_sparsity(pc, True)

     # First solve - the PCAIR will be setup
     ksp.solve(b, x)

     # Save the polynomial coefficients from the first solve
     num_levels = pflare.pcair_get_num_levels(pc)
     coeffs_levels = {}
     for petsc_level in range(num_levels - 1, 0, -1):
         coeffs_levels[petsc_level] = pflare.pcair_get_poly_coeffs(pc, petsc_level, pflare.COEFFS_INV_AFF)
     coeffs_levels[0] = pflare.pcair_get_poly_coeffs(pc, 0, pflare.COEFFS_INV_COARSE)

     # ...[Modify entries in A to get a different linear system]

     # Second solve - the PCAIR will be setup with reuse for the new system
     ksp.solve(b2, x2)

     # ...[Restore the original A]

     # Restore the saved polynomial coefficients and tell PCAIR to reuse them
     for petsc_level in range(num_levels - 1, 0, -1):
         pflare.pcair_set_poly_coeffs(pc, petsc_level, pflare.COEFFS_INV_AFF, coeffs_levels[petsc_level])
     pflare.pcair_set_poly_coeffs(pc, 0, pflare.COEFFS_INV_COARSE, coeffs_levels[0])
     pflare.pcair_set_reuse_poly_coeffs(pc, True)

     # Third solve - reproduces the preconditioner from the first solve
     ksp.solve(b, x)

**With PCPFLAREINV:** As with PCAIR, this requires that the sparsity pattern of A is the same for the first and third solves, since the sparsity of the assembled approximate inverse depends on A. The polynomial coefficients can be saved and restored without a separate sparsity-reuse flag:

in Fortran:

     call KSPGetPC(ksp, pc, ierr)
     call PCSetType(pc, PCPFLAREINV, ierr)

     ! First solve - the PCPFLAREINV will be setup
     call KSPSolve(ksp, b, x, ierr)

     ! Save the polynomial coefficients from the first solve
     call PCPFLAREINVGetPolyCoeffs(pc, coeffs, ierr)

     ! ...[Modify entries in A to get a different linear system]

     ! Second solve - the PCPFLAREINV will be setup for the new system
     call KSPSolve(ksp, b2, x2, ierr)

     ! ...[Restore the original A]

     ! Restore the saved polynomial coefficients and tell PCPFLAREINV to reuse them
     call PCPFLAREINVSetPolyCoeffs(pc, coeffs, ierr)
     call PCPFLAREINVSetReusePolyCoeffs(pc, PETSC_TRUE, ierr)

     ! Third solve - reproduces the preconditioner from the first solve
     call KSPSolve(ksp, b, x, ierr)

or in C:

     ierr = KSPGetPC(ksp, &pc);
     ierr = PCSetType(pc, PCPFLAREINV);

     // First solve - the PCPFLAREINV will be setup
     ierr = KSPSolve(ksp, b, x);

     // Save the polynomial coefficients from the first solve
     ierr = PCPFLAREINVGetPolyCoeffs(pc, &coeffs, &rows, &cols);

     // ...[Modify entries in A to get a different linear system]

     // Second solve - the PCPFLAREINV will be setup for the new system
     ierr = KSPSolve(ksp, b2, x2);

     // ...[Restore the original A]

     // Restore the saved polynomial coefficients and tell PCPFLAREINV to reuse them
     ierr = PCPFLAREINVSetPolyCoeffs(pc, coeffs, rows, cols);
     ierr = PCPFLAREINVSetReusePolyCoeffs(pc, PETSC_TRUE);

     // Third solve - reproduces the preconditioner from the first solve
     ierr = KSPSolve(ksp, b, x);

or in Python with petsc4py:

     import pflare

     pc = ksp.getPC()
     pc.setType("pflareinv")

     # First solve - the PCPFLAREINV will be setup
     ksp.solve(b, x)

     # Save the polynomial coefficients from the first solve
     coeffs = pflare.pcpflareinv_get_poly_coeffs(pc)

     # ...[Modify entries in A to get a different linear system]

     # Second solve - the PCPFLAREINV will be setup for the new system
     ksp.solve(b2, x2)

     # ...[Restore the original A]

     # Restore the saved polynomial coefficients and tell PCPFLAREINV to reuse them
     pflare.pcpflareinv_set_poly_coeffs(pc, coeffs)
     pflare.pcpflareinv_set_reuse_poly_coeffs(pc, True)

     # Third solve - reproduces the preconditioner from the first solve
     ksp.solve(b, x)