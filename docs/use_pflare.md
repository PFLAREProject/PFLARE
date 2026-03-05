## Using PFLARE

After installation, the PFLARE PC types can then be used like native PETSc PC types, either by writing code to set the PETSc type/options, or through command line arguments. A few examples include:

#### 1) Using PCAIR to apply AIRG with default options:

in Fortran:

     call KSPGetPC(ksp, pc, ierr)
     call PCSetType(pc, PCAIR, ierr)

     ! ...[e.g., KSPSolve somewhere here]

or in C:

     ierr = KSPGetPC(ksp, &pc);
     ierr = PCSetType(pc, PCAIR);

     // ...[e.g., KSPSolve somewhere here]

or in Python with petsc4py:

     pc = ksp.getPC()
     pc.setType("air")

     # ...[e.g., KSPSolve somewhere here]
     
or via the command line: ``-pc_type air``. 

#### 2) Using PCAIR to apply distance 2 lAIR with FCF Weighted Jacobi smoothing:

in Fortran:

     call KSPGetPC(ksp, pc, ierr)
     call PCSetType(pc, PCAIR, ierr)

     call PCAIRSetZType(pc, AIR_Z_LAIR, ierr)
     call PCAIRSetInverseType(pc, PFLAREINV_WJACOBI, ierr)
     call PCAIRSetSmoothType(pc, "fcf", ierr)

     ! ...[e.g., KSPSolve somewhere here]

or in C:

     ierr = KSPGetPC(ksp, &pc);
     ierr = PCSetType(pc, PCAIR);

     ierr = PCAIRSetZType(pc, AIR_Z_LAIR);
     ierr = PCAIRSetInverseType(pc, PFLAREINV_WJACOBI);
     ierr = PCAIRSetSmoothType(pc, "fcf");

     // ...[e.g., KSPSolve somewhere here]

or in Python with petsc4py:

     import pflare

     pc = ksp.getPC()
     pc.setType("air")

     pflare.pcair_set_z_type(pc, pflare.AIR_Z_LAIR)
     pflare.pcair_set_inverse_type(pc, pflare.PFLAREINV_WJACOBI)
     pflare.pcair_set_smooth_type(pc, "fcf")

     # ...[e.g., KSPSolve somewhere here]

or via the command line: ``-pc_type air -pc_air_z_type lair -pc_air_inverse_type wjacobi -pc_air_smooth_type fcf``.

#### 3) Using PCPFLAREINV to apply a 20th order GMRES polynomial as a Newton polynomial matrix free:

in Fortran:

     call KSPGetPC(ksp, pc, ierr)
     call PCSetType(pc, PCPFLAREINV, ierr)

     call PCPFLAREINVSetPolyOrder(pc, 20, ierr)
     call PCPFLAREINVSetType(pc, PFLAREINV_NEWTON, ierr)
     call PCPFLAREINVSetMatrixFree(pc, PETSC_TRUE, ierr)

     ! ...[e.g., KSPSolve somewhere here]

or in C:

     ierr = KSPGetPC(ksp, &pc);
     ierr = PCSetType(pc, PCPFLAREINV);

     ierr = PCPFLAREINVSetPolyOrder(pc, 20);
     ierr = PCPFLAREINVSetType(pc, PFLAREINV_NEWTON);
     ierr = PCPFLAREINVSetMatrixFree(pc, PETSC_TRUE);

     // ...[e.g., KSPSolve somewhere here]

or in Python with petsc4py (via the options database, or equivalently via the command line):

     pc = ksp.getPC()
     pc.setType("pflareinv")

     petsc_options = PETSc.Options()
     petsc_options['pc_pflareinv_type'] = 'newton'
     petsc_options['pc_pflareinv_poly_order'] = '20'
     petsc_options['pc_pflareinv_matrix_free'] = ''

     # ...[e.g., KSPSolve somewhere here]

or via the command line: ``-pc_type pflareinv -pc_pflareinv_type newton -pc_pflareinv_poly_order 20 -pc_pflareinv_matrix_free``.

#### 4) Jupyter notebooks:

There are a number of Jupyter notebooks in `notebooks/`, which include an introduction to the key components of PFLARE and some simple Python examples. These can be run locally, or you can click the Binder badge at the top of the README to run these interactively.