## Reuse in PFLARE

If solving multiple linear systems, there are several ways to reuse components of PFLARE preconditioners (PCAIR and PCPFLAREINV) to reduce setup times.

#### 1) Setup once

In PETSc the ``KSPSetReusePreconditioner`` flag can be set to ensure the preconditioner is only setup once, even if the entries or sparsity of the underlying matrix in the KSP is changed. This can be useful in time stepping problems where the linear system (or Jacobian for non-linear problems) does not change, or if the changes are small.

#### 2) Reuse sparsity in the setup when the diagonal changes

**With PCAIR:** By default, if the values of the Pmat change between solves, PCAIR rebuilds the entire hierarchy on the next setup, even if only the diagonal has changed. 

Instead, if PCAIR has been built from a matrix $\mathbf{A}$ and we want to solve with $\mathbf{A} + \mathbf{D}$, where $\mathbf{D}$ is diagonal, then the flag ``-pc_air_diag_update`` can be set before the PCAIR hierarchy is first set up. In subsequent setups, PCAIR doesn't do a full rebuild, instead it modifies the existing hierarchy so that it approximates the hierarchy of $\mathbf{A} + \mathbf{D}$. 

When this flag is set, PCAIR does not check that only the diagonal has changed. This means the update can also be used when other entries of the matrix change (e.g., if the biggest change is on the diagonal); only the new diagonal is extracted and the hierarchy modified.

in Fortran:

     call KSPGetPC(ksp, pc, ierr)
     call PCSetType(pc, PCAIR, ierr)

     ! Before the first setup, we tell it to update the
     ! hierarchy when the diagonal of the Pmat changes
     call PCAIRSetDiagUpdate(pc, PETSC_TRUE, ierr)
     ! Adding C point smoothing is recommended with diagonal updates, e.g., fc or ffc
     call PCAIRSetSmoothType(pc, "fc", ierr)

     ! First solve - the PCAIR will be setup from A
     call KSPSolve(ksp, b, x, ierr)

     ! ...[Change the diagonal of A in place, e.g., with MatDiagonalSet or MatShift]

     ! Second solve - the PCAIR hierarchy will be updated
     call KSPSolve(ksp, b, x, ierr)

or in C:

     ierr = KSPGetPC(ksp, &pc);
     ierr = PCSetType(pc, PCAIR);

     // Before the first setup, we tell it to update the
     // hierarchy when the diagonal of the Pmat changes
     ierr = PCAIRSetDiagUpdate(pc, PETSC_TRUE);
     // Adding C point smoothing is recommended with diagonal updates, e.g., fc or ffc
     ierr = PCAIRSetSmoothType(pc, "fc");

     // First solve - the PCAIR will be setup from A
     ierr = KSPSolve(ksp, b, x);

     // ...[Change the diagonal of A in place, e.g., with MatDiagonalSet or MatShift]

     // Second solve - the PCAIR hierarchy will be updated
     ierr = KSPSolve(ksp, b, x);

or in Python with petsc4py:

     pc = ksp.getPC()
     pc.setType("air")

     # Before the first setup, we tell it to update the
     # hierarchy when the diagonal of the Pmat changes
     pflare.pcair_set_diag_update(pc, True)
     # Adding C point smoothing is recommended with diagonal updates, e.g., fc or ffc
     pflare.pcair_set_smooth_type(pc, "fc")

     # First solve - the PCAIR will be setup from A
     ksp.solve(b, x)

     # ...[Change the diagonal of A in place, e.g., with A.setDiagonal or A.shift]

     # Second solve - the PCAIR hierarchy will be updated
     ksp.solve(b, x)

or via the command line: ``-pc_type air -pc_air_diag_update -pc_air_smooth_type fc``.

A change to the nonzero pattern of the Pmat, or passing a different Pmat to ``KSPSetOperators``, rebuilds the hierarchy as normal. As with any change to the Pmat, nothing happens if ``KSPSetReusePreconditioner`` has been set.

##### How the update is computed

Consider a single level, with the level matrix split into fine (F) and coarse (C) points

$$
\mathbf{A} = 
\begin{bmatrix}
\mathbf{A}_\textrm{ff} & \mathbf{A}_\textrm{fc} \\
\mathbf{A}_\textrm{cf} & \mathbf{A}_\textrm{cc}
\end{bmatrix}, \quad
\mathbf{D} = 
\begin{bmatrix}
\mathbf{D}_\textrm{f} & \\
 & \mathbf{D}_\textrm{c}
\end{bmatrix},
$$

with restrictor $\mathbf{R} = [\mathbf{Z} \;\; \mathbf{I}]$, where $\mathbf{Z} \approx -\mathbf{A}_\textrm{cf} \mathbf{A}_\textrm{ff}^{-1}$, and prolongator $\mathbf{P} = [\mathbf{W} ;\; \mathbf{I}]$. If we rebuilt the hierarchy from $\mathbf{A} + \mathbf{D}$ everything on the level would change. Instead, the update keeps the CF splitting, $\mathbf{P}$, $\mathbf{A}_\textrm{fc}$, $\mathbf{A}_\textrm{cf}$ and the sparsity of every matrix, and changes only:

**The diagonal blocks:** $\mathbf{A}_\textrm{ff} \leftarrow \mathbf{A}_\textrm{ff} + \mathbf{D}_\textrm{f}$ and, if C point smoothing is used, $\mathbf{A}_\textrm{cc} \leftarrow \mathbf{A}_\textrm{cc} + \mathbf{D}_\textrm{c}$. 

**The restrictor:** The F point columns of $\mathbf{R}$ are therefore rescaled

$$
\mathbf{Z} \leftarrow \mathbf{Z} \, \textrm{diag}\left( \frac{a_\textrm{ff}}{a_\textrm{ff} + d_\textrm{f}} \right),
$$

where $a_\textrm{ff}$ and $d_\textrm{f}$ are the diagonals of $\mathbf{A}_\textrm{ff}$ and $\mathbf{D}_\textrm{f}$. This is exact if $\mathbf{A}_\textrm{ff}$ is diagonal and a good approximation when $\mathbf{A}_\textrm{ff}$ is strongly diagonally dominant, as the CF splittings in PCAIR aim for. With ``-pc_air_symmetric`` the restrictor is $\mathbf{P}^T$ and is not rescaled.

**The coarse matrix:** the Galerkin coarse matrix of $\mathbf{A} + \mathbf{D}$ would be $\mathbf{R}\mathbf{A}\mathbf{P} + \mathbf{R}\mathbf{D}\mathbf{P}$. Rather than computing $\mathbf{R}\mathbf{D}\mathbf{P}$ with matrix-matrix products, it is lumped onto the diagonal with a row sum, which preserves its action on the constant vector

$$
\mathbf{d}_\textrm{coarse} = \mathbf{R}(\mathbf{d} \circ \mathbf{P}\mathbf{1}),
$$

where $\mathbf{d}$ is the diagonal of $\mathbf{D}$, $\mathbf{1}$ is the vector of ones and $\circ$ is the entrywise product. With the default one-point prolongator $\mathbf{P}\mathbf{1} = \mathbf{1}$, so this is just one matrix-vector product with the (rescaled) restrictor, $\mathbf{d}_\textrm{coarse} = \mathbf{R}\mathbf{d}$. The coarse matrix becomes $\mathbf{R}\mathbf{A}\mathbf{P} + \textrm{diag}(\mathbf{d}_\textrm{coarse})$ and the diagonal update can proceed on the next level. 

**The smoothers:** the approximate inverses of $\mathbf{A}_\textrm{ff} + \mathbf{D}_\textrm{f}$ (and of $\mathbf{A}_\textrm{cc} + \mathbf{D}_\textrm{c}$) used in the smoothers are rebuilt. The updated $\mathbf{R}$ and coarse matrix do not depend on an updated approximation to $\mathbf{A}_\textrm{ff}^{-1}$, so with matrix-free smoothing (``-pc_air_matrix_free_polys`` and ``-pc_air_coarsest_matrix_free_polys``) an update does not assemble any approximate inverses and only computes the new polynomial coefficients.

Using this diagonal update in PCAIR is very fast and uses almost no extra memory. The frozen grid-transfer operators and the lumping, however, mean the updated hierarchy is not identical to one rebuilt from $\mathbf{A} + \mathbf{D}$. With F point smoothing only, errors introduced at the C points by the frozen grid-transfer operators are never corrected, so adding C point smoothing (e.g., ``-pc_air_smooth_type fc`` or ``ffc``) is recommended with this type of diagonal update.

On a 2D advection problem with diagonal perturbations varying over six orders of magnitude, adding C point smoothing kept the iteration count within a few iterations of rebuilding the hierarchy, independent of the grid size. The gap is largest when $\mathbf{D}$ is of similar size to the diagonal of $\mathbf{A}$; when $\mathbf{D}$ is either small or dominant the updated hierarchy performs like a rebuilt one.

If the off-diagonal entries also change significantly, or the iteration count with the updated hierarchy grows too large, the hierarchy should instead be rebuilt. The reuse described in the sections below makes that rebuild cheaper.

#### 3) Reuse sparsity during setup

**With PCAIR:** When solving a linear system where the matrix has the same sparsity pattern as in a previous solve, PCAIR can reuse the CF splitting, repartitioning, symbolic matrix-matrix products and resulting sparsity throughout the hierarchy during the setup. This takes more memory (typically 2-5x the storage complexity) but significantly reduces the time required in subsequent setups (10-20x). If we have a PETSc matrix $\mathbf{A}$:

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

     # Before the first setup, we tell it to
     # store any data we need for reuse
     pflare.pcair_set_reuse_sparsity(pc, True)

     # First solve - the PCAIR will be setup
     ksp.solve(b, x)

     # ...[Modify entries in A but keep the same sparsity]

     # Second solve - the PCAIR will be setup
     # but reusing the same sparsity
     ksp.solve(b, x)

or via the command line: ``-pc_type air -pc_air_reuse_sparsity``.

##### Controlling the amount of data stored with reuse_amount

By default (``-pc_air_reuse_amount = 3``), PCAIR stores everything when reuse is enabled via ``-pc_air_reuse_sparsity`` : the CF splitting, all intermediate SpGEMM results, and the approximate inverse matrices. This gives the fastest subsequent setups but uses the most memory (typically 2-5x the storage complexity of the matrix).

The ``-pc_air_reuse_amount`` option trades memory for a slightly higher setup cost on subsequent solves:

- **1** — Store only the CF splitting (IS_fine_index, IS_coarse_index) and parallel repartitioning. All matrices are rebuilt from scratch each setup. Lowest memory overhead.
- **2** — Store the CF splitting plus all matrices needed to guarantee stable SpGEMM sparsity (W, Z, W_DROP, Z_DROP, AP, RAP, RAP_DROP, and when ``strong_r_threshold != 0`` also the dropped submatrices). A_ff/A_fc/A_cf, prolongators and restrictors are rebuilt each setup. Moderate memory overhead.
- **3** — Store everything, including the approximate inverse matrices and the grid-transfer operators. Fastest setup but highest memory (default).

in Fortran:

     call PCAIRSetReuseSparsity(pc, PETSC_TRUE, ierr)
     call PCAIRSetReuseAmount(pc, 2, ierr)   ! store CF splitting + SpGEMM sparsity only

or in C:

     ierr = PCAIRSetReuseSparsity(pc, PETSC_TRUE);
     ierr = PCAIRSetReuseAmount(pc, 2);

or in Python with petsc4py:

     pflare.pcair_set_reuse_sparsity(pc, True)
     pflare.pcair_set_reuse_amount(pc, 2)

or via the command line: ``-pc_type air -pc_air_reuse_sparsity -pc_air_reuse_amount 2``.

#### 4) Reuse GMRES polynomial coefficients during setup

When the matrix is very similar to that used in a previous solve, the GMRES polynomial coefficients can be reused to save the parallel polynomial iteration in the setup. Note that the GMRES polynomial coefficients are very sensitive to changes in the matrix, so this may not always give good results.

**With PCAIR (AIRG):** In addition to reusing the polynomial coefficients, the sparsity must also be set to reuse (see section 3 above). To enable both:

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

     # Before the first setup, we tell it to
     # store any data we need for reuse
     pflare.pcair_set_reuse_sparsity(pc, True)

     # Tell it to store the gmres polynomial coefficients
     pflare.pcair_set_reuse_poly_coeffs(pc, True)

     # First solve - the PCAIR will be setup
     ksp.solve(b, x)

     # ...[Modify entries in A but keep the same sparsity]

     # Second solve - the PCAIR will be setup
     # but reusing sparsity & polynomial coefficients
     ksp.solve(b, x)

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

or in Python with petsc4py:

     pc = ksp.getPC()
     pc.setType("pflareinv")

     # Tell it to reuse the gmres polynomial coefficients
     pflare.pcpflareinv_set_reuse_poly_coeffs(pc, True)

     # First solve - the PCPFLAREINV will be setup
     ksp.solve(b, x)

     # ...[Modify entries in A]

     # Second solve - the PCPFLAREINV will be setup
     # but reusing the polynomial coefficients
     ksp.solve(b, x)

or via the command line: ``-pc_type pflareinv -pc_pflareinv_reuse_poly_coeffs``.

#### 5) Reuse when solving the same linear system

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
     // The returned pointer is owned by the PC and is invalidated by the next setup,
     // so copy the coefficients into our own storage
     ierr = PCPFLAREINVGetPolyCoeffs(pc, &pc_coeffs, &rows, &cols);
     ierr = PetscMalloc1(rows * cols, &coeffs);
     ierr = PetscArraycpy(coeffs, pc_coeffs, rows * cols);

     // ...[Modify entries in A to get a different linear system]

     // Second solve - the PCPFLAREINV will be setup for the new system
     ierr = KSPSolve(ksp, b2, x2);

     // ...[Restore the original A]

     // Restore the saved polynomial coefficients and tell PCPFLAREINV to reuse them
     ierr = PCPFLAREINVSetPolyCoeffs(pc, coeffs, rows, cols);
     ierr = PCPFLAREINVSetReusePolyCoeffs(pc, PETSC_TRUE);

     // Third solve - reproduces the preconditioner from the first solve
     ierr = KSPSolve(ksp, b, x);

     // PCPFLAREINVSetPolyCoeffs copies the coefficients, so we can free our copy
     ierr = PetscFree(coeffs);

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