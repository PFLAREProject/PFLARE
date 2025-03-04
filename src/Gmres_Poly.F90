module gmres_poly

   use petsc
   use sorting
   use c_petsc_interfaces
   use matshell
   use tsqr
   use gmres_poly_data_type
   use nonbusywait
   use petsc_helper
                
   ! OpenMP module
#ifdef _OPENMP
   use omp_lib
#endif

#include "petsc/finclude/petsc.h"   

   implicit none

#include "petsc_legacy.h"

   ! Just define pi
   PetscReal, parameter, private :: pi = 3.141592653589793
   type int_vec
      integer, dimension(:), pointer :: ptr
   end type int_vec
   type real_vec
      PetscReal, dimension(:), pointer :: ptr
   end type real_vec   

   public

   PetscEnum, parameter :: PFLAREINV_POWER=0
   PetscEnum, parameter :: PFLAREINV_ARNOLDI=1
   PetscEnum, parameter :: PFLAREINV_NEWTON=2
   PetscEnum, parameter :: PFLAREINV_NEWTON_NO_EXTRA=3
   
   contains

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine setup_gmres_poly_data(global_rows, inverse_type, poly_order, &
                  poly_sparsity_order, subcomm, proc_stride, poly_data)   

      ! Setups the gmres poly data structure and does some size checking

      ! ~~~~~~
      PetscInt, intent(in)                              :: global_rows, proc_stride
      integer, intent(in)                               :: inverse_type, poly_order
      integer, intent(in)                               :: poly_sparsity_order
      logical, intent(in)                               :: subcomm
      type(gmres_poly_data), intent(inout)              :: poly_data
      ! ~~~~~~            

      ! For gmres polynomial calculation we can move the reductions onto a subcomm
      if (inverse_type == PFLAREINV_POWER .OR. &
          inverse_type == PFLAREINV_ARNOLDI .OR. &
          inverse_type == PFLAREINV_NEWTON .OR. &
          inverse_type == PFLAREINV_NEWTON_NO_EXTRA) then
         poly_data%buffers%subcomm = subcomm
      end if
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
      poly_data%buffers%matrix = PETSC_NULL_MAT
#endif      
      poly_data%buffers%proc_stride = int(proc_stride)

      ! For matrices with size smaller than the subspace size (ie polynomial order + 1)
      ! we'll have (close to) an exact solver and only need to go up to the matrix size
      poly_data%gmres_poly_order = poly_order
      if (poly_data%gmres_poly_order + 1 > global_rows) then
         poly_data%gmres_poly_order = int(global_rows - 1)
      end if        

      ! If we've changed the polynomial order lets make sure we aren't trying to take 
      ! sparsity bigger than it
      poly_data%gmres_poly_sparsity_order = poly_sparsity_order
      if (poly_data%gmres_poly_sparsity_order > poly_data%gmres_poly_order) then
         poly_data%gmres_poly_sparsity_order = poly_data%gmres_poly_order
      end if      

      ! Now we know the order we can create our coefficient vector
      if (inverse_type == PFLAREINV_NEWTON .OR. inverse_type == PFLAREINV_NEWTON_NO_EXTRA) then
         ! Newton basis we have to store real and imag roots
         if (.NOT. associated(poly_data%coefficients)) allocate(poly_data%coefficients(poly_data%gmres_poly_order + 1, 2))
      else
         if (.NOT. associated(poly_data%coefficients)) allocate(poly_data%coefficients(poly_data%gmres_poly_order + 1, 1))
      end if

   end subroutine setup_gmres_poly_data

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine compute_mf_gmres_poly_num_matvecs(inverse_type, coefficients, matvecs)   

      ! Computes how many matvecs we need to apply our matrix-free gmres polynomial

      ! ~~~~~~
      integer, intent(in)                               :: inverse_type
      PetscReal, dimension(:, :), intent(in)                 :: coefficients
      integer, intent(out)                              :: matvecs

      integer :: i_loc
      logical :: zero_root
      ! ~~~~~~       
      
      ! What order is our polynomial
      matvecs = size(coefficients,1)

      ! We may have zero roots with newton that are skipped
      if (inverse_type == PFLAREINV_NEWTON) then
         do i_loc = 1, size(coefficients,1)
            zero_root = .FALSE.
            if (coefficients(i_loc,2) == 0d0) then
               ! The size of the zero check here has to match that in 
               ! petsc_matvec_gmres_newton_mf 
               if (abs(coefficients(i_loc,1)) < 1e-12) zero_root = .TRUE.
            else
               if (coefficients(i_loc,1)**2 + &
                        coefficients(i_loc,2)**2 < 1e-12) zero_root = .TRUE.
            end if

            if (zero_root) then
               matvecs = matvecs - 1
            end if
         end do

      ! For power or Arnoldi   
      else
         ! The size of the zero check is exactly 0d0, normally can only happen
         ! in an Arnoldi that terminates early
         do i_loc = 1, size(coefficients,1)
            if (abs(coefficients(i_loc,1)) == 0d0) then
               matvecs = matvecs - 1
            end if
         end do         
      end if      

   end subroutine compute_mf_gmres_poly_num_matvecs   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine create_temp_space_box_mueller(matrix, subspace_size, V_n)
      
      ! Creates some temporary space and computes box mueller random numbers
      ! in the first column

      ! ~~~~~~
      type(tMat), intent(in)                                      :: matrix
      integer, intent(in)                                         :: subspace_size
      type(tVec), dimension(:)                                    :: V_n

      ! Local variables
      MPI_Comm :: MPI_COMM_MATRIX
      PetscInt :: local_rows, local_cols, global_rows, global_cols
      PetscInt :: global_row_start, global_row_end_plus_one, row_i
      PetscCount :: vec_size
      PetscInt, allocatable, dimension(:) :: indices
      integer :: i_loc, seed_size, comm_size, comm_rank, errorcode
      PetscErrorCode :: ierr      
      integer, dimension(:), allocatable :: seed
      PetscReal, dimension(:, :), allocatable, target   :: random_data
      PetscInt, parameter :: one=1, zero=0
      ! ~~~~~~    

      ! We might want to call the gmres poly creation on a sub communicator
      ! so let's get the comm attached to the matrix and make sure to use that 
      call PetscObjectGetComm(matrix, MPI_COMM_MATRIX, ierr)  
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)     
      ! Get the comm rank
      call MPI_Comm_rank(MPI_COMM_MATRIX, comm_rank, errorcode)     
      ! Get the matrix sizes
      call MatGetSize(matrix, global_rows, global_cols, ierr)
      call MatGetLocalSize(matrix, local_rows, local_cols, ierr)     
      call MatGetOwnershipRange(matrix, global_row_start, global_row_end_plus_one, ierr)  

      ! ~~~~~~~~~~
      ! Create random vector - this happens on the cpu
      ! We could do most of the box-muller on the gpu, but currently the random numbers in petsc 
      ! are still generated on the host (and we don't have portable trig functions in petsc across the different
      ! gpu vec types), so that would cause more copies to/from the gpu     
      ! If that changes this should be rewritten so this all happens on the gpu  
      ! ~~~~~~~~~~      

      ! If you ever change this to use PetscRandomCreate and VecSetRandom in petsc
      ! make sure the seed is different than when random is used to evaluate 
      ! the auto truncation in AIR_MG_Setup

      call random_seed(size=seed_size)
      allocate(seed(seed_size))
      ! Ensure we seed the same so subsequent runs get the same random
      ! Seed different on each process so we don't have the same random numbers on each processor      
      do i_loc = 1, seed_size
         seed(i_loc) = comm_rank + 1 + i_loc
      end do   
      call random_seed(put=seed)    

      ! Gives random numbers between 0 <= u < 1  
      allocate(random_data(local_rows, 2))
      call random_number(random_data(:, 1:2))

      ! Remove the u = 0 (ie epsilon) case
      ! to change from [0,1) to (0,1], which we need for the log */
      random_data(:, 1) = 1d0 - random_data(:, 1)

      ! We want our random rhs to be a normal distribution with zero mean as that preserves
      ! white noise in the eigenspace (ie it is rotation invariant to unitary transforms)
      ! Do a box-muller to take two numbers with uniform distribution and produce a number
      ! that is normally distributed       
      random_data(:, 1) = sqrt(-2 * log(random_data(:, 1))) * cos(2 * pi * random_data(:, 2))
      deallocate(seed)

      ! ~~~~~~~~~~
      ! Create some petsc vecs and assign the data in them to point at K_m+1
      ! ~~~~~~~~~~ 
      ! Create vectors pointing at the columns in K_m+1
      do i_loc = 1, subspace_size+1
         call MatCreateVecs(matrix, V_n(i_loc), PETSC_NULL_VEC, ierr)         
      end do     
      
      allocate(indices(local_rows))
      ! Set the random values into the first vector
      ! V_n(1) data will be copied to the gpu when needed
      do row_i = 1, local_rows
         indices(row_i) = global_row_start + row_i-1
      end do
      ! PetscCount vs PetscInt??
      vec_size = local_rows
      call VecSetPreallocationCOO(V_n(1), vec_size, indices, ierr)
      deallocate(indices)
      call VecSetValuesCOO(V_n(1), random_data, INSERT_VALUES, ierr)
      
      deallocate(random_data)

      ! ~~~~~~~~~~~~

   end subroutine create_temp_space_box_mueller  
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine ls_solve_arnoldi(beta, m, H_n, y)

      ! Do the least-squares solve used in an arnoldi

      ! ~~~~~~
      PetscReal, intent(in)                     :: beta
      integer, intent(in)                  :: m
      PetscReal, dimension(:,:), intent(in)     :: H_n
      PetscReal, dimension(:), intent(inout)    :: y

      ! Local variables
      integer :: errorcode, lwork
      PetscReal, dimension(size(H_n,1), size(H_n,2)) :: H_n_copy
      PetscReal, dimension(m+1) :: g0
      PetscReal, dimension(:), allocatable :: work

      ! ~~~~~~   
            
      ! This is the vector we will use as rhs in the least square solve
      g0 = 0d0
      g0(1) = beta 

      ! Let's solve our least squares system
      allocate(work(1))
      lwork = -1      

      ! Make a copy as we do a least-squares solve in place
      H_n_copy(1:m+1, 1:m) = H_n(1:m+1, 1:m)

      lwork = -1
      ! Overwrite y
      errorcode = 0
      call dgels('N', m+1, m, 1, H_n_copy(1,1), size(H_n_copy, 1), g0, size(g0), work, lwork, errorcode)
      lwork = int(work(1))
      deallocate(work)
      allocate(work(lwork))  
      call dgels('N', m+1, m, 1, H_n_copy(1,1), size(H_n_copy, 1), g0, size(g0), work, lwork, errorcode)
      deallocate(work)            

      if (errorcode /= 0) then
         print *, "LS solve failed"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if  

      ! Set to zero necessary as sometimes we terminate our arnoldi early
      y = 0
      y(1:m) = g0(1:m)

   end subroutine ls_solve_arnoldi   
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine arnoldi(matrix, poly_order, lucky_tol, V_n, w_j, beta, H_n, m, C_n, y, user_rel_tol)

      ! Arnoldi to compute H_n and optionally C_n (although computing C_n 
      ! won't be stable at high order)

      ! ~~~~~~
      type(tMat), intent(in)                            :: matrix
      integer, intent(in)                               :: poly_order
      PetscReal, intent(in)                                  :: lucky_tol
      type(tVec), dimension(:), intent(in)              :: V_n
      type(tVec), intent(in)                            :: w_j
      PetscReal, intent(out)                                 :: beta
      PetscReal, dimension(:,:), intent(inout)               :: H_n
      PetscReal, dimension(:,:), optional, intent(inout)     :: C_n
      PetscReal, optional, intent(inout)                     :: user_rel_tol
      PetscReal, dimension(:), optional, intent(inout)       :: y
      integer, intent(out)                              :: m

      ! Local variables
      integer :: i_loc, subspace_size
      PetscErrorCode :: ierr      
      PetscReal, dimension(poly_order+2) :: c_j, g0
      logical :: compute_cn
      PetscReal :: rel_tol

      ! ~~~~~~   
            
      compute_cn = .FALSE.
      if (present(C_n)) compute_cn = .TRUE.
      ! Only compute H_n until we hit a given relative residual tolerance if it is input by the user
      rel_tol = -1
      if (present(user_rel_tol)) rel_tol = user_rel_tol

      ! This is how many columns we have in K_m
      subspace_size = poly_order + 1       

      ! This is the hessenberg matrix
      H_n = 0d0
      ! This stores part of the polynomial coefficients             
      if (compute_cn) C_n = 0d0

      ! ~~~~~~~~~
      ! Now time for a GMRES
      ! ~~~~~~~~~

      ! Compute the norm of the initial residual
      call VecNorm(V_n(1), NORM_2, beta, ierr)
      ! Now normalise to create the first orthogonal vector in V_n(1)
      call VecNormalize(V_n(1), beta, ierr)

      ! The first entry in C_n - As V_n = K_n C_n
      ! the first orthogonalised vector in V_n is just r0/beta, 
      ! and we know r0 is the first vector in our krylov subspace
      if (compute_cn) C_n(1,1) = 1d0/beta

      ! Now loop through and do the iterations up to the max order of the polynomial
      ! we want to build
      do m = 1, subspace_size

         ! Compute w_j = A v_j
         call MatMult(matrix, &
                  V_n(m), &
                  w_j, ierr)  

         ! Now compute the updated relationship between K_n and V_n
         if (compute_cn) then
            c_j = 0d0
            ! This is [0 c1..cn] for column m
            c_j(2:m + 1) = C_n(1:m, m)  
         end if
                  
         ! Now loop 
         do i_loc = 1, m

            ! Computes the hessenberg entry by the dot product of w_j and v_i
            call VecDot(w_j, V_n(i_loc), H_n(i_loc, m), ierr)               

            ! w_j = w_j - h_ij v_i
            call VecAXPY(w_j, -H_n(i_loc, m), V_n(i_loc), ierr)

            ! This is doing -C_n * h_n
            if (compute_cn) c_j(1:i_loc) = c_j(1:i_loc) - C_n(1:i_loc, i_loc) * H_n(i_loc, m)

         end do

         ! Now compute new hessenberg entry
         call VecNorm(w_j, NORM_2, H_n(m+1, m), ierr)

         ! GMRES lucky tolerance, we're fully converged
         if (H_n(m+1, m) < lucky_tol) then
            ! Don't forget to update the ls solution if you exit early
            if (rel_tol > 0) call ls_solve_arnoldi(beta, m, H_n, y)
            exit
         end if

         ! v_j+1 = w_j / h_j+1,j
         call VecAXPBY(V_n(m + 1), &
                  1d0/H_n(m+1, m), &
                  0d0, &
                  w_j, ierr)           

         ! Now we've taken out the H_n(m+1, m) factor from above
         if (compute_cn) C_n(1:m+1, m + 1) = c_j(1:m+1)/H_n(m+1, m)

         ! ~~~~~~
         ! Compute the residual if the user requested only solving to a specific 
         ! relative residual - we don't need it otherwise
         ! The residual is just ||H_m y - e_1 beta||_2, so we have to solve that least squares problem to find 
         ! y and then just compute 
         ! ~~~~~~
         if (rel_tol > 0) then

            call ls_solve_arnoldi(beta, m, H_n, y)
            
            ! Compute H_n y
            call dgemv("N", m+1, m, &
                  1d0, H_n, size(H_n,1), &
                  y, 1, &
                  0d0, g0(1), 1) 

            ! Minus away e1 beta
            g0(1) = g0(1) - beta
            ! This is the relative residual
            user_rel_tol = norm2(g0(1:m))/beta
            !print *, m, "rel residual", user_rel_tol
            if (user_rel_tol < rel_tol) exit
         end if

      end do

      ! Fortran allocates loop variables one bigger if we get to the end of the loop
      if (m == subspace_size + 1) m = m - 1

   end subroutine arnoldi      

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine calculate_gmres_polynomial_coefficients_arnoldi(matrix, poly_order, coefficients, user_rel_tol)

      ! Computes a fixed order gmres polynomial for the matrix passed in
      ! and stores each of the polynomial coefficients in coefficients
      ! This computes the coefficients using the Arnoldi basis
      ! This won't be stable at high order, but is stable at higher order than
      ! using the power basis. The cost of this is many reductions in parallel
      ! There is a comms-avoiding one given in start_gmres_polynomial_coefficients_power 
      ! which is stable for low order

      ! ~~~~~~
      type(tMat), intent(in)                            :: matrix
      integer, intent(in)                               :: poly_order
      PetscReal, dimension(:), intent(out)                   :: coefficients
      PetscReal, optional, intent(inout)                     :: user_rel_tol

      ! Local variables
      PetscInt :: global_rows, global_cols
      integer :: i_loc, subspace_size, m
      integer :: errorcode
      PetscErrorCode :: ierr      
      PetscReal, dimension(poly_order+2,poly_order+1) :: H_n
      PetscReal, dimension(poly_order+2,poly_order+2) :: C_n
      PetscReal, dimension(poly_order+1) :: y
      PetscReal :: beta
      type(tVec) :: w_j
      type(tVec), dimension(poly_order+2) :: V_n
      PetscReal :: rel_tol

      ! ~~~~~~    

      ! This is how many columns we have in K_m
      subspace_size = poly_order + 1   
      ! Get the matrix sizes
      call MatGetSize(matrix, global_rows, global_cols, ierr)

      ! ~~~~~~~~~~~~~
      ! We're basically just writing a simple GMRES here with no restart, the reason we can't use the petsc one
      ! is we need access to the unrotated hessenberg matrix, and the orthogonalised vectors
      ! to compute the gmres polynomial
      ! See Nachtigal et al 1992 for details on how we generate the gmres polynomial 
      ! coefficients using the Arnoldi basis
      ! ~~~~~~~~~~~~~

      if (subspace_size > global_rows) then
         print *, "The input subspace size is greater than the matrix size"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! ~~~~~~~~~~
      ! Allocate space and create random numbers 
      ! The first vec has random numbers in it
      ! ~~~~~~~~~~ 
      call create_temp_space_box_mueller(matrix, subspace_size, V_n)
      
      ! Create an extra vector for storage
      call VecDuplicate(V_n(1), w_j, ierr)         

      ! Do the Arnoldi and compute H_n and C_n
      ! We only compute H_n until we hit a relative residual of 1e-14 against the random rhs
      ! or we hit the given poly_order
      rel_tol = 1e-14
      if (present(user_rel_tol)) rel_tol = user_rel_tol
      call arnoldi(matrix, poly_order, 1d-30, V_n, w_j, beta, H_n, m, C_n, y, rel_tol)
      if (present(user_rel_tol)) user_rel_tol = rel_tol

      ! ~~~~~~~~~~~~~
      ! Compute the polynomial coefficients, this is C_n(1:m, 1:m) y
      ! ~~~~~~~~~~~~~
      ! Set to zero is necessary as m may be less than the subspace_size
      coefficients = 0
      call dgemv("N", m, m, &
               1d0, C_n, size(C_n,1), &
               y, 1, &
               0d0, coefficients(1), 1) 

      do i_loc = 1, subspace_size+1
         call VecDestroy(V_n(i_loc), ierr)
      end do
      call VecDestroy(w_j, ierr)

   end subroutine calculate_gmres_polynomial_coefficients_arnoldi   

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine start_gmres_polynomial_coefficients_power(matrix, poly_order, buffers, coefficients)
      
      ! Computes a fixed order gmres polynomial for the matrix passed in
      ! and stores each of the polynomial coefficients in coefficients
      ! This computes the polynomial coefficients in the power basis using 
      ! the power basis
      ! This allows us to do a 1-step method that is comms reducing
      ! only requiring a single reduction in parallel
      ! If in parallel this is non-blocking and you have to call
      ! finish_gmres_polynomial_coefficients_power outside this routine to finish

      ! ~~~~~~
      type(tMat), intent(in)                            :: matrix
      integer, intent(in)                               :: poly_order
      type(tsqr_buffers), intent(inout)                 :: buffers
      PetscReal, dimension(:), intent(inout)                 :: coefficients

      ! Local variables
      PetscInt :: global_rows, global_cols, local_rows, local_cols
      PetscInt :: global_row_start, global_row_end_plus_one, row_i
      PetscInt, dimension(:), allocatable :: global_indices
      integer :: comm_size, subspace_size, comm_rank, i_loc
      integer :: errorcode
      PetscErrorCode :: ierr      
      MPI_Comm :: MPI_COMM_MATRIX
      PetscReal, dimension(:, :), allocatable, target :: K_m_plus_1_data
      type(tVec), dimension(poly_order+2) :: V_n
      ! ~~~~~~    

      ! This is how many columns we have in K_m
      subspace_size = poly_order + 1

      ! We might want to call the gmres poly creation on a sub communicator
      ! so let's get the comm attached to the matrix and make sure to use that 
      call PetscObjectGetComm(matrix, MPI_COMM_MATRIX, ierr) 
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)     
      ! Get the comm rank
      call MPI_Comm_rank(MPI_COMM_MATRIX, comm_rank, errorcode)     
      ! Get the matrix sizes
      call MatGetSize(matrix, global_rows, global_cols, ierr)
      call MatGetLocalSize(matrix, local_rows, local_cols, ierr)

      if (subspace_size > global_rows) then
         print *, "The input subspace size must be smaller than the matrix size"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! ~~~~~~~~~
      ! GMRES(m) will minimise the residual over the subspace
      ! K_m = [ b, A b, A^2 b, ... A^(m-1) b]
      ! A normal GMRES algorithm will use m dot products (ie m reductions) and m matvecs to 
      ! compute an orthgonal basis of this subspace directly with an Arnoldi
      ! without needing to compute the power basis K_m directly. 
      ! Equivalently, we can compute the gmres polynomial coefficients, g, by 
      ! solving the least squares system
      ! min || b - A K_m g ||
      ! Rather than having to compute the QR factorisation of A K_m and 
      ! than also have to compute the norm of b, we note that
      ! A K_m = K_m+1(:,2:end)
      ! It takes m matvecs to compute K_m+1 (same number of matvecs as in using the Arnoldi
      ! to compute an orthogonalised basis)
      ! but by doing a QR on K_m+1, the norm of r_0 will be in the first row/col position of R, and 
      ! then we can use the other columns of R as they will be the QR of A K_m
      ! That way we only have to do a single reduction in parallel
      ! This is equivalent to computing the gmres polynomial coefficients for 
      ! a single step (s=1) communication avoiding GMRES
      ! Using the power basis is a bad idea at high order due to stability issues, but 
      ! we are staying low order so should be fine  (e.g, < 10th order)
      ! ~~~~~~~~~

      ! ~~~~~~~~~~
      ! Allocate space and create random numbers 
      ! The first vec has random numbers in it
      ! ~~~~~~~~~~      
      call create_temp_space_box_mueller(matrix, subspace_size, V_n)

      ! ~~~~~~~~~~~~

      ! Now loop through and build the power basis K_m+1
      do i_loc = 1, subspace_size

         ! Compute V_n(j+1) = A * V_n(j)
         ! Could do this with a matrix-power kernel with less comms if desired, except
         ! that we only ever need this once for very low order so that would prob result in more comms overall
         call MatMult(matrix, &
                  V_n(i_loc), &
                  V_n(i_loc+1), ierr)  
      end do      

      ! Need to copy the individual vectors into a dense slab of memory
      ! Used to just have a dense slab of memory and then
      ! place the pointers into the vecs, but that 
      ! became cumbersome when supporting different vec types (eg cuda on a gpu)
      allocate(K_m_plus_1_data(local_rows, subspace_size+1))
      allocate(global_indices(local_rows))
      call MatGetOwnershipRange(matrix, global_row_start, global_row_end_plus_one, ierr)  
      do row_i = 1, local_rows
         global_indices(row_i) = global_row_start + row_i - 1
      end do
        
      ! Copy into K_m_plus_1_data and then destroy the vecs
      do i_loc = 1, subspace_size+1
         ! Have to use vecgetvalues interface for gpus
         ! This does a copy of the vec from gpu to cpu if necessary
         call VecGetValues(V_n(i_loc), local_rows, global_indices, K_m_plus_1_data(:, i_loc), ierr)
         call VecDestroy(V_n(i_loc), ierr)
      end do         

      ! ~~~~~~~~~~~~~
      ! Start the tall-skinny QR factorisation of the power basis - this all happens
      ! on the cpu
      ! ~~~~~~~~~~~~~
      call start_tsqr(MPI_COMM_MATRIX, K_m_plus_1_data, buffers)

      ! ~~~~~
      ! ~~~~~
      ! Now have to call finish_gmres_polynomial_coefficients_power outside this routine
      ! to resolve the async comms in the tsqr
      ! ~~~~~
      ! ~~~~~

      deallocate(K_m_plus_1_data, global_indices)    

   end subroutine start_gmres_polynomial_coefficients_power 

! -------------------------------------------------------------------------------------------------------------------------------

subroutine  finish_gmres_polynomial_coefficients_power(poly_order, buffers, coefficients)
      
      ! Finishes the non-blocking all reduce from start_gmres_polynomial_coefficients_power in parallel
      ! The polynomial coefficients will be output into coefficients

      ! ~~~~~~
      integer, intent(in)                                      :: poly_order
      type(tsqr_buffers), target, intent(inout)                :: buffers
      PetscReal, dimension(:), intent(inout)                        :: coefficients
 
      integer :: lwork, subspace_size, n_size, rank, iwork_size
      integer :: errorcode
      integer, dimension(:), allocatable :: iwork
      PetscReal, dimension(poly_order+2) :: g0, s
      PetscReal, dimension(:), allocatable :: work
      PetscReal, dimension(:,:), pointer :: R_pointer
      PetscReal :: rcond = -1d0

      ! ~~~~~

      ! We call build_gmres_polynomial_inverse to build our neumann polynomials
      ! but they don't need to calculate coefficients, so this skips this routine
      if (.NOT. allocated(buffers%R_buffer_receive)) return

      ! This is how many columns we have in K_m
      subspace_size = poly_order + 1      

      ! Finish the asynchronous wait for the tsqr
      ! This will fill buffers%R_buffer_receive with [n_size R(:,:)] from our tsqr
      call finish_tsqr_parallel(buffers)
      ! This should be the same size as subspace_size + 1
      n_size = int(buffers%R_buffer_receive(1))
      if (n_size /= subspace_size + 1) then
         print *, "ERROR in sizes in TSQR"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! Point directly at the R block
      R_pointer(1:n_size, 1:n_size) => buffers%R_buffer_receive(2:n_size * n_size + 1)
      g0 = 0      
      ! We know beta is in the top left of R
      g0(1) = R_pointer(1, 1)   
      
      ! We've now finished orthogonalising, let's solve our least squares system
      allocate(work(1))
      allocate(iwork(1))
      lwork = -1

      ! Now we are using dgelsd here as there is no guarantee our power basis will be full rank
      ! eg if our matrix is a (scaled) version of the identity, our columns in the power basis
      ! will be very close (given the random rhs)
      ! to orthogonal and dgels fails without full rank
      ! Start from the second column in R_pointer
      call dgelsd(subspace_size+1, subspace_size, 1, R_pointer(1, 2), subspace_size+1, &
                     g0, size(g0), s, rcond, rank, &
                     work, lwork, iwork, errorcode)
      lwork = int(work(1))
      iwork_size = iwork(1)
      deallocate(work, iwork)
      allocate(work(lwork)) 
      allocate(iwork(iwork_size))
      call dgelsd(subspace_size+1, subspace_size, 1, R_pointer(1, 2), subspace_size+1, &
                     g0, size(g0), s, rcond, rank, &
                     work, lwork, iwork, errorcode)
      deallocate(work, iwork)

      if (errorcode /= 0) then
         print *, "LS solve failed"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if  
      
      ! These are the polynomial coefficients
      coefficients = g0(1:subspace_size)      

      ! Deallocate the receive buffer
      if (allocated(buffers%R_buffer_receive)) deallocate(buffers%R_buffer_receive)    

   end subroutine  finish_gmres_polynomial_coefficients_power     

!------------------------------------------------------------------------------------------------------------------------
   
   subroutine mat_mult_powers_share_sparsity(matrix, poly_order, poly_sparsity_order, buffers, coefficients, reuse_mat, cmat)

      ! Compute matrix powers c = coeff(1) * I + coeff(2) * A + coeff(3) * A^2 + coeff(4) * A^3 + ... 
      ! where a c and the powers all share the same sparsity as the power input in poly_sparsity_order
      ! Assuming cmat has not been built/allocated
      ! This also finishes the async comms required to compute the gmres poly coefficients if buffers%request is allocated
   
      ! ~~~~~~~~~~
      ! Input 
      type(tMat), target, intent(in)                  :: matrix
      integer, intent(in)                             :: poly_order, poly_sparsity_order
      type(tsqr_buffers), intent(inout)               :: buffers
      PetscReal, dimension(:), intent(inout)               :: coefficients
      type(tMat), intent(inout)                       :: reuse_mat, cmat
      
      PetscInt :: local_rows, local_cols, global_rows, global_cols
      PetscInt :: global_row_start, global_row_end_plus_one
      PetscInt :: global_col_start, global_col_end_plus_one, n, ncols, ncols_two, ifree, max_nnzs
      PetscInt :: i_loc, j_loc, row_size, rows_ao, cols_ao, rows_ad, cols_ad, shift = 0
#ifdef _OPENMP
      integer :: omp_threads, omp_threads_env, length, status
      CHARACTER(len=255) :: omp_threads_env_char
#endif      
      integer :: errorcode, match_counter, term, order, location
      integer :: comm_size
      PetscErrorCode :: ierr      
      PetscReal, dimension(:), allocatable :: vals_two, vals_power_temp, vals_previous_power_temp
      PetscReal, dimension(:), allocatable, target :: vals
      integer, dimension(:), allocatable :: cols_index_one, cols_index_two
      PetscInt, dimension(:), allocatable :: cols, cols_two, cols_local
      PetscInt, dimension(:), allocatable :: col_indices_off_proc_array
      PetscInt, pointer :: colmap_c(:)
      type(tIS) :: col_indices
      type(tMat) :: Ad, Ao
      type(tMat), target :: temp_mat
      PetscOffset :: iicol
      PetscInt :: icol(1)
      type(c_ptr) :: colmap_c_ptr, vals_c_ptr
      ! In fortran this needs to be of size n+1 where n is the number of submatrices we want
      type(tMat), dimension(2) :: submatrices
      type(tMat), dimension(size(coefficients)-1), target :: matrix_powers
      type(tMat), pointer :: mat_sparsity_match
      type(int_vec), dimension(:), allocatable :: symbolic_ones
      type(real_vec), dimension(:), allocatable :: symbolic_vals
      integer(c_long_long) A_array
      MPI_Comm :: MPI_COMM_MATRIX
      PetscReal, dimension(:), allocatable :: vals_temp, vals_prev_temp
      PetscInt, dimension(:), pointer :: submatrices_ia, submatrices_ja, cols_two_ptr, cols_ptr
      PetscReal, dimension(:), pointer :: vals_two_ptr, vals_ptr
      real(c_double), pointer :: submatrices_vals(:)
      logical :: symmetric = .false., inodecompressed=.false., done, reuse_triggered
      PetscInt, parameter :: one = 1, zero = 0
      MatType:: mat_type    
      
      ! ~~~~~~~~~~  

      if (poly_sparsity_order .ge. size(coefficients)-1) then      
         print *, "Requested sparsity is greater than or equal to the order"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if      

      call MatGetType(matrix, mat_type, ierr)

      call PetscObjectGetComm(matrix, MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)

      ! Get the local sizes
      call MatGetLocalSize(matrix, local_rows, local_cols, ierr)
      call MatGetSize(matrix, global_rows, global_cols, ierr)
      ! This returns the global index of the local portion of the matrix
      call MatGetOwnershipRange(matrix, global_row_start, global_row_end_plus_one, ierr)  
      call MatGetOwnershipRangeColumn(matrix, global_col_start, global_col_end_plus_one, ierr)  

      reuse_triggered = .NOT. PetscMatIsNull(cmat) 

      ! ~~~~~~~~~~
      ! Special case if we just want to return a gmres polynomial with the sparsity of the diagonal
      ! This is like a damped Jacobi
      ! ~~~~~~~~~~
      if (poly_sparsity_order == 0) then

         call build_gmres_polynomial_inverse_0th_order_sparsity(matrix, poly_order, buffers, &
                  coefficients, cmat)     

         return
      end if

      ! ~~~~~~~~~~
      ! Compute any matrix powers we might need to constrain sparsity and start assembling the 
      ! components of the output matrix up to the order of poly_sparsity_order
      ! The powers higher than poly_sparsity_order can be done with only
      ! a single bit of comms and is done below this
      ! ~~~~~~~~~~

      ! matrix_powers stores all the powers of the input matrix
      matrix_powers(1) = matrix

      ! What power of A do we want to match the sparsity of
      ! Compute the power we need if we're two or above
      do order = 2, poly_sparsity_order

         ! Let's just store each power, that way we can set the sparsity 
         ! as the highest (unconstrained) power and do the mataxpy with a subset of entries
         ! Takes more memory to do this but is faster
         call MatMatMult(matrix, matrix_powers(order-1), &
               MAT_INITIAL_MATRIX, 1.5d0, matrix_powers(order), ierr)        
      end do  
      
      ! mat_sparsity_match now contains the sparsity of the power of A we want to match
      mat_sparsity_match => matrix_powers(poly_sparsity_order)

      ! Copy in the highest unconstrained power
      ! Duplicate & copy the matrix, but ensure there is a diagonal present
      call mat_duplicate_copy_plus_diag(matrix_powers(poly_sparsity_order), reuse_triggered, cmat)

      ! We know we will never have non-zero locations outside of the highest constrained sparsity power 
      call MatSetOption(cmat, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE,  ierr)     
      call MatSetOption(cmat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr) 
      ! We know we are only going to insert local vals
      ! These options should turn off any reductions in the assembly
      call MatSetOption(cmat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)     
      
      ! ~~~~~~~~~~~~
      ! If we're in parallel we need to get the off-process rows of matrix that correspond
      ! to the columns of mat_sparsity_match
      ! We can therefore do the matmult for every constrained power locally with just that data
      ! ~~~~~~~~~~~~
      ! Have to double check comm_size /= 1 as we might be on a subcommunicator and we can't call
      ! MatMPIAIJGetSeqAIJ specifically if that's the case
      if (comm_size /= 1) then

         ! Petsc stores the diagonal and off diagonal components of our local rows - ie Ad and Ao
         ! colmap gives the map between local columns of Ao and the global numbering      
         ! In petsc <3.14, we have to call some custom c code to get at colmap directly (as I can't 
         ! seem to get the icol and iicol variables to work in the fortran interface of MatMPIAIJGetSeqAIJ)
         ! In petsc 3.15 - 3.18 we could use MatMPIAIJGetLocalMatMerge, as that returns colmap
         ! but it also constructs [Ad Ao] which we don't really need
         ! In petsc 3.19 we could call the new fortran interface MatMPIAIJGetSeqAIJF90
         ! (in fact we would have to given the fortran interface to MatMPIAIJGetSeqAIJ is deprecated)
         ! So once we're on petsc 3.19 we should just use MatMPIAIJGetSeqAIJF90 and then we can do 
         ! away with get_colmap_c

         ! ~~~~
         ! Get the cols
         ! ~~~~
         if (mat_type == "mpiaij") then
            ! Much more annoying in older petsc
            call MatMPIAIJGetSeqAIJ(mat_sparsity_match, Ad, Ao, icol, iicol, ierr)            
         
         ! If on the gpu, just do a convert to mpiaij format first
         ! This will be expensive but the best we can do for now without writing our 
         ! own version of this subroutine in cuda/kokkos
         else
            call MatConvert(mat_sparsity_match, MATMPIAIJ, MAT_INITIAL_MATRIX, temp_mat, ierr)
            call MatMPIAIJGetSeqAIJ(temp_mat, Ad, Ao, icol, iicol, ierr) 
            mat_sparsity_match => temp_mat
         end if

         call MatGetSize(Ad, rows_ad, cols_ad, ierr)             
         ! We know the col size of Ao is the size of colmap, the number of non-zero offprocessor columns
         call MatGetSize(Ao, rows_ao, cols_ao, ierr)         

         ! For the column indices we need to take all the columns of mat_sparsity_match
         A_array = mat_sparsity_match%v
         call get_colmap_c(A_array, colmap_c_ptr)
         call c_f_pointer(colmap_c_ptr, colmap_c, shape=[cols_ao])

         ! These are the global indices of the columns we want
         allocate(col_indices_off_proc_array(cols_ad + cols_ao))
         ! Local rows (as global indices)
         do ifree = 1, cols_ad
            col_indices_off_proc_array(ifree) = global_row_start + ifree - 1
         end do
         ! Off diagonal rows we want (as global indices)
         do ifree = 1, cols_ao
            col_indices_off_proc_array(cols_ad + ifree) = colmap_c(ifree)
         end do

         ! Create the sequential IS we want with the cols we want (written as global indices)
         call ISCreateGeneral(PETSC_COMM_SELF, cols_ad + cols_ao, &
                     col_indices_off_proc_array, PETSC_USE_POINTER, col_indices, ierr) 

         if (poly_sparsity_order /= 1) then
            call ISSort(col_indices, ierr)
         end if

         ! ~~~~~~~
         ! Now we can pull out the chunk of matrix that we need
         ! ~~~~~~~

         ! We need off-processor rows to compute matrix powers   
         ! Setting this is necessary to avoid an allreduce when calling createsubmatrices
         ! This will be reset to false after the call to createsubmatrices
         call MatSetOption(matrix, MAT_SUBMAT_SINGLEIS, PETSC_TRUE, ierr)       
         
         ! Now this will be doing comms to get the non-local rows we want
         ! But only including the columns of the local fixed sparsity, as we don't need all the 
         ! columns of the non-local entries unless we are doing a full matmatmult         
         ! This matrix has the local rows and the non-local rows in it
         ! We could just request the non-local rows, but it's easier to just get the whole slab
         ! as then the row indices match colmap
         ! This returns a sequential matrix
         if (.NOT. PetscMatIsNull(reuse_mat)) then
            submatrices(1) = reuse_mat
            call MatCreateSubMatrices(matrix, one, col_indices, col_indices, MAT_REUSE_MATRIX, submatrices, ierr)
         else
            call MatCreateSubMatrices(matrix, one, col_indices, col_indices, MAT_INITIAL_MATRIX, submatrices, ierr)
            reuse_mat = submatrices(1)
         end if
         row_size = size(col_indices_off_proc_array)
         call ISDestroy(col_indices, ierr)

      ! Easy in serial as we have everything we neeed
      else
         
         submatrices(1) = matrix
         row_size = local_rows
         allocate(col_indices_off_proc_array(local_rows))
         do ifree = 1, local_rows
            col_indices_off_proc_array(ifree) = ifree-1
         end do
      end if   
      
      ! ~~~~~~~~~
      ! Now that we are here, submatrices(1) contains A^poly_sparsity_order with all of the rows
      ! that correspond to the non-zero columns of matrix
      ! ~~~~~~~~~      

      ! Have to get the max nnzs of the local and off-local rows we've just retrieved
      max_nnzs = 0
      ! Check the nnzs of the serial copy of matrix
      do ifree = 1, row_size            
         call MatGetRow(submatrices(1), ifree-1, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
         if (ncols > max_nnzs) max_nnzs = ncols
         call MatRestoreRow(submatrices(1), ifree-1, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
      end do
      ! and also the sparsity power
      do ifree = global_row_start, global_row_end_plus_one-1     
         call MatGetRow(mat_sparsity_match, ifree, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
         if (ncols > max_nnzs) max_nnzs = ncols
         call MatRestoreRow(mat_sparsity_match, ifree, ncols, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, ierr)
      end do 
      
      ! ~~~~~~~~
      ! Get pointers to the sequential aij structure so we don't have to put critical regions
      ! around the matgetrow
      ! ~~~~~~~~
      call MatGetRowIJF90(submatrices(1),shift,symmetric,inodecompressed,n,submatrices_ia,submatrices_ja,done,ierr) 
      if (.NOT. done) then
         print *, "Pointers not set in call to MatGetRowIJF90"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if
      ! Returns the wrong size pointer and can break if that size goes negative??
      !call MatSeqAIJGetArrayF90(submatrices(1),submatrices_vals,ierr);
      A_array = submatrices(1)%v
      ! Now we must never overwrite the values in this pointer, and we must 
      ! never call restore on it, see comment on top of the commented out
      ! MatSeqAIJRestoreArrayF90 below
      call MatSeqAIJGetArrayF90_mine(A_array, vals_c_ptr)
      call c_f_pointer(vals_c_ptr, submatrices_vals, shape=[size(submatrices_ja)])
      
      ! ~~~~~~~~~~
      
      allocate(cols(max_nnzs))
      allocate(cols_two(max_nnzs))
      allocate(vals(max_nnzs)) 
      allocate(vals_two(max_nnzs)) 
      allocate(vals_power_temp(max_nnzs))
      allocate(vals_previous_power_temp(max_nnzs))
      allocate(cols_index_one(max_nnzs))
      allocate(cols_index_two(max_nnzs))      

      ! ~~~~~~~~~~~      
      ! Finish the non-blocking all-reduce and compute our polynomial coefficients
      ! ~~~~~~~~~~~
      call finish_gmres_polynomial_coefficients_power(poly_order, buffers, coefficients)

      ! Scale the highest constrained power
      call MatScale(cmat, coefficients(poly_sparsity_order+1), ierr)

      ! Then go backwards and add in each of the coefficients * A^order from the second highest order down
      do order = poly_sparsity_order - 1, 1, -1

         ! Do result = alpha_1 * A_ff + alpha_2 * A_ff^2 + ....
         ! Can use SUBSET_NONZERO_PATTERN as we have put the highest order power in first
         call MatAXPY(cmat, coefficients(order+1), matrix_powers(order), SUBSET_NONZERO_PATTERN , ierr)       
      end do 

      ! Add in the 0th order term
      do i_loc = 1, local_rows   
         
         ! Add in the I term - 0th order term
         call MatSetValue(cmat, global_row_start + i_loc-1, global_row_start + i_loc-1, &
                     coefficients(1), ADD_VALUES, ierr)           
      end do

      ! ~~~~~~~~~~~~
      ! From here we now have cmat with the correct values up to the power poly_sparsity_order
      ! and hence we want to add in the sparsity constrained powers
      ! ~~~~~~~~~~~~


#ifdef _OPENMP
      ! Any rank with 0 local_rows hits the non-busy wait
      ! before the collective matassembly (below) and goes idle and so we can use it to thread
      ! This typically requires the omp threads to be pinned reasonably 
      ! for good performance
      ! buffers%proc_stride tells us how many idle ranks we have total
      ! after processor agglomeration
      ! Get the number of omp threads set in the environmental variable
      call get_environment_variable("OMP_NUM_THREADS", omp_threads_env_char, &
               length=length, status=status)
      if (status /= 1 .AND. length /= 0) then
         read(omp_threads_env_char(1:5),'(i5)') omp_threads_env
         ! If we have idle ranks
         if (buffers%proc_stride /= 1) then
            ! Don't use more than omp_threads_env omp threads
            ! It's up to the user to tell us the max number of omp threads
            ! to use via OMP_NUM_THREADS - at most the user should
            ! set it as the number of threads per numa region
            omp_threads = min(omp_threads_env, buffers%proc_stride)
            call omp_set_num_threads(omp_threads)
         ! If we have no idle ranks then don't thread 
         else
            call omp_set_num_threads(1)
         end if
      ! If the OMP_NUM_THREADS is not set
      else
         call omp_set_num_threads(1)
      end if
#endif
      
      ! Now go through and compute the sum of the matrix powers
      ! We're doing row-wise matmatmults here assuming the fixed sparsity
      ! We exploit the fact that the subsequent matrix powers can be done
      ! one row at a time, so we only have to retrieve the needed vals from mat_sparsity_match once

      ! Need dynamic scheduling in the omp here as each row will have different amounts of work
      !$omp parallel   proc_bind(close) &
      !$omp            shared(local_rows, poly_sparsity_order, submatrices, mat_sparsity_match) &
      !$omp            shared(global_row_start, comm_size, col_indices_off_proc_array, row_size) &
      !$omp            shared(coefficients, cmat, submatrices_ia, submatrices_ja, submatrices_vals) &
      !$omp            private(i_loc, ncols_two, cols_two, vals_two, ierr) & 
      !$omp            private(ncols, cols, cols_local, vals, location, j_loc, symbolic_ones) & 
      !$omp            private(symbolic_vals, cols_index_one, cols_index_two, match_counter) & 
      !$omp            private(vals_temp, vals_prev_temp, term, cols_two_ptr, vals_two_ptr) &
      !$omp            private(cols_ptr, vals_ptr)
      !$omp do schedule(dynamic,10) 
      do i_loc = 1, local_rows 
                          
         ! Get the row of mat_sparsity_match
         ! We need both the local indices into submatrices(1) and the global indices
         ! We have local but want global
         if (poly_sparsity_order == 1) then

            ! This is the number of columns
            ncols = submatrices_ia(i_loc+1) - submatrices_ia(i_loc)      
            ! This is the column indices
            cols_ptr => submatrices_ja(submatrices_ia(i_loc)+1:submatrices_ia(i_loc+1))
            ! This is the values
            vals_ptr => submatrices_vals(submatrices_ia(i_loc)+1:submatrices_ia(i_loc+1))                  

            ! Need to modify the column positions to be global indices
            allocate(cols_local(ncols))
            ! Already has the local indices
            cols_local = cols_ptr
            if (comm_size /= 1) then
               ! Gives us the global indices
               cols(1:ncols) = col_indices_off_proc_array(cols_local+1)
            else
               cols(1:ncols) = cols_ptr               
            end if  

         ! We have global but want local
         else

            ! Should optimise this to use the pointers rather than a critical protected access
            ! but the matmatmults to 
            ! create the powers are very expensive so should deal with that first            
            !$omp critical
            call MatGetRow(mat_sparsity_match, global_row_start + i_loc-1, ncols_two, &
                  cols_two, vals_two, ierr)  
                  
            ! Can't have two getrows active at the same time, so store this one
            ncols = ncols_two
            cols(1:ncols) = cols_two(1:ncols)
            allocate(cols_local(ncols))
            vals(1:ncols) = vals_two(1:ncols) 
            vals_ptr => vals(1:ncols)
            
            call MatRestoreRow(mat_sparsity_match, global_row_start + i_loc-1, ncols_two, &
                     cols_two, vals_two, ierr)     
            !$omp end critical           
                     
            ! We already have the global indices, we need the local ones
            do j_loc = 1, ncols
               ! We have a local column
               !if (cols(j_loc) .ge. global_col_start .AND. cols(j_loc) < global_col_end_plus_one) then
               !   cols_local(j_loc) = cols(j_loc) - global_col_start

               ! Otherwise we have a parallel column we have to find the local index in col_indices_off_proc_array
               ! We have sorted col_indices_off_proc_array, as they are not necessarily sorted if we just 
               ! consider the col indices augmented with those from colmap
               !else
                  call sorted_binary_search(col_indices_off_proc_array, cols(j_loc), location)
                  if (location == -1) then
                     print *, "Couldn't find location"
                     call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
                  end if
                  cols_local(j_loc) = location-1
               !end if
            end do  
         end if
         
         if (any(cols_local > row_size)) then
            print *, "Local size bigger than it should be"
            call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
         end if

         ! If we have to do matrix powers, then let's just do all the column matching
         ! and extraction of the values once
         ! This is just basically a symbolic for the set of rows given in cols
            
         ! Allocate some space to store the matching indices
         allocate(symbolic_ones(ncols))
         allocate(symbolic_vals(ncols))

         ! This is a row-wise product
         do j_loc = 1, ncols

            ! This is the number of columns
            ncols_two = submatrices_ia(cols_local(j_loc)+2) - submatrices_ia(cols_local(j_loc)+1)
            ! This is the column indices
            cols_two_ptr => submatrices_ja(submatrices_ia(cols_local(j_loc)+1)+1:submatrices_ia(cols_local(j_loc)+2))
            ! This is the values
            vals_two_ptr => submatrices_vals(submatrices_ia(cols_local(j_loc)+1)+1:submatrices_ia(cols_local(j_loc)+2))
            
            ! Search for the matching column - assuming sorted
            ! Need to make sure we're matching the global indices
            call intersect_pre_sorted_indices_only(cols_local(1:ncols), cols_two_ptr, cols_index_one, cols_index_two, match_counter)                 
            
            ! Don't need to do anything if we have no matches
            if (match_counter == 0) then 
               ! Store that we can skip this entry
               symbolic_ones(j_loc)%ptr => null()
               symbolic_vals(j_loc)%ptr => null()                        
            else

               ! These are the matching local column indices for this row of mat_sparsity_match
               allocate(symbolic_ones(j_loc)%ptr(match_counter))
               symbolic_ones(j_loc)%ptr = cols_index_one(1:match_counter)

               ! These are the matching values of matrix
               allocate(symbolic_vals(j_loc)%ptr(match_counter))
               symbolic_vals(j_loc)%ptr = vals_two_ptr(cols_index_two(1:match_counter)) 
            end if          
         end do    

         ! Store an extra exact size copy to ensure the omp reduction
         ! doesn't have to add extra stuff outside of ncols
         ! The newer omp standards lets you put a size bound on the reduction
         allocate(vals_temp(ncols))
         allocate(vals_prev_temp(ncols))
         
         ! Start with the values of mat_sparsity_match in it
         vals_prev_temp = vals_ptr
                     
         ! Loop over any matrix powers
         ! vals_power_temp stores the value of A^(term-1) for this row, and we update this as we go through 
         ! the term loop
         do term = poly_sparsity_order+2, size(coefficients)

            ! Skip this term if the coefficient is zero
            if (coefficients(term) == 0d0) cycle

            ! We need to sum up the product of vals_previous_power_temp(j_loc) * matching columns
            vals_temp = 0

            ! Have to finish all the columns before we move onto the next coefficient
            do j_loc = 1, ncols

               ! If we have no matching columns cycle this row
               if (.NOT. associated(symbolic_ones(j_loc)%ptr)) cycle

               ! symbolic_vals(j_loc)%ptr has the matching values of A in it
               vals_temp(symbolic_ones(j_loc)%ptr) = vals_temp(symbolic_ones(j_loc)%ptr) + &
                        vals_prev_temp(j_loc) * symbolic_vals(j_loc)%ptr

            end do
               
            ! ~~~~~~~~~~~
            ! Now can add the value of coeff * A^(term-1) to our matrix
            ! ~~~~~~~~~~~
            ! Has to be critical as can't add to the matrix from multiple threads at once
            !$omp critical
            call MatSetValues(cmat, one, [global_row_start + i_loc-1], ncols, cols(1:ncols), &
                     coefficients(term) * vals_temp, ADD_VALUES, ierr)   
            !$omp end critical

            ! This should now have the value of A^(term-1) in it
            vals_prev_temp = vals_temp
         end do   

         deallocate(vals_temp, vals_prev_temp)

         ! Delete our symbolic
         do j_loc = 1, ncols
            if (associated(symbolic_ones(j_loc)%ptr)) then
               deallocate(symbolic_ones(j_loc)%ptr)
               deallocate(symbolic_vals(j_loc)%ptr)
            end if      
         end do  
         deallocate(symbolic_vals, symbolic_ones, cols_local)  
      end do 
      !$omp end do
      !$omp end parallel      

      ! ~~~~~~~~~~~
      ! Now if we are a rank with zero local rows, we will skip all 
      ! the compute above and we want to sleep so we can be available as a thread
      ! to the omp above
      ! This has to be above the collective matassemblybegin/end, otherwise we would
      ! rely on the waits in our mpi library being non-busy and there are very 
      ! few which do this, let alone by default 
      ! ~~~~~~~~~~~      
#ifdef _OPENMP
      ! Set the number of threads back to omp_threads_env when we're done
      call omp_set_num_threads(omp_threads_env)

      ! Non-busy wait
      if (buffers%proc_stride /= 1) then
         call non_busy_wait(MPI_COMM_MATRIX, local_rows)
      end if      
#endif  

      call MatRestoreRowIJF90(submatrices(1),shift,symmetric,inodecompressed,n,submatrices_ia,submatrices_ja,done,ierr) 
      ! We very deliberately don't call restorearray here!
      ! There is no matseqaijgetarrayread or matseqaijrestorearrayread in Fortran
      ! Those routines don't increment the PetscObjectStateGet which tells petsc
      ! the mat has changed. Hence above we directly access the data pointer with 
      ! a call to MatSeqAIJGetArrayF90_mine and then never write into it
      ! If we call the restorearrayf90, that does increment the object state
      ! even though we only read from the array
      ! That would mean if we pass in a pc->pmat for example, just setting up a pc
      ! would trigger petsc setting up the pc on every iteration of the pc
      ! call MatSeqAIJRestoreArrayF90(submatrices(1),submatrices_vals,ierr);

      ! ~~~~~~~~~~~

      ! Do the assembly, should need zero reductions in this given we've set the 
      ! flags above
      call MatAssemblyBegin(cmat, MAT_FINAL_ASSEMBLY, ierr)

      ! Delete temporaries
      do order = 2, poly_sparsity_order
         call MatDestroy(matrix_powers(order), ierr)
      end do
      if (comm_size /= 1 .AND. mat_type /= "mpiaij") then
         call MatDestroy(temp_mat, ierr)
      end if

      deallocate(col_indices_off_proc_array)
      deallocate(cols, vals, cols_two, vals_two, vals_power_temp, vals_previous_power_temp, cols_index_one, cols_index_two)

      ! Finish assembly
      call MatAssemblyEnd(cmat, MAT_FINAL_ASSEMBLY, ierr) 

         
   end subroutine mat_mult_powers_share_sparsity
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine petsc_horner(mat, coefficients, temp_vec, x, y)

      ! Uses a horner iteration to apply
      ! y = (coeff(1) + coeff(2) * A + coeff(3) * A^2 + ...) x

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! Input
      type(tMat), intent(in)    :: mat
      PetscReal, dimension(:)        :: coefficients
      type(tVec)                :: x, temp_vec
      type(tVec)                :: y

      ! Local
      integer :: order
      PetscErrorCode :: ierr      

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! ~~~~~~~
      ! This applies the polynomial of order n as:
      ! xn - x0 = q_n-1(A) r_0 
      ! where 
      ! q_n-1(z) = alpha_0 + alpha_1 * z + alpha_2 * z^2 + ...
      ! So the q_n-1(A) r_0 term is just the repeated application of 
      ! y = alpha_n-1 r_0
      ! do i = 1, n-1
      !   y = A * y + alpha_n-i-1 r_0
      ! where the output y is xn - x0.
      ! Practically, if we choose ksprichardson and then use this as a preconditioner B, 
      ! we are doing (where these n are the iteration count, not the order of the polynomial we're 
      ! applying above)
      ! x^n+1 = x^n + B * r^n
      ! so the x passed in should be the residual r^n, and we don't need to add x^n to
      ! the solution, as the richardson is doing that for us. We have to ensure the richardson scale is one though.
      ! ~~~~~~~

      ! Let's do the first y = alpha_n-1 r_0 (ie the highest order term first)
      call VecAXPBY(y, &
               coefficients(size(coefficients)), &
               0d0, &
               x, ierr)

      ! If we are doing a first order polynomial or above, we have to do an extra matvec per order
      if (size(coefficients, 1) > 1) then     

         ! Loop down from the second highest order term down to the constant
         do order = size(coefficients, 1)-1, 1, -1

            ! Skip this coefficient if zero
            if (coefficients(order) == 0d0) cycle

            ! Copy y into temp_vec
            call VecCopy(y, temp_vec, ierr)             

            ! Now do y = A * temp_vec
            call MatMult(mat, temp_vec, y, ierr)

            ! Compute y = A * temp_vec + alpha_n-i-1 r_0
            call VecAXPBY(y, &
                     coefficients(order), &
                     1d0, &
                     x, ierr)
         end do
      end if

   end subroutine petsc_horner     

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine petsc_matvec_poly_mf(mat, x, y)

      ! Applies a matrix polynomial matrix-free as an inverse
      ! Just uses a Horner iteration to apply the mat_ctx%coefficients
      ! to mat_ctx%mat in the input matshell
      ! y = A x

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      ! Input
      type(tMat), intent(in)    :: mat
      type(tVec) :: x
      type(tVec) :: y

      ! Local
      PetscErrorCode :: ierr
      integer :: errorcode
      type(mat_ctxtype), pointer :: mat_ctx => null()

      ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~

      call MatShellGetContext(mat, mat_ctx, ierr)
      if (.NOT. associated(mat_ctx%coefficients)) then
         print *, "Polynomial coefficients in context aren't found"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! Call the Horner iteration
      call petsc_horner(mat_ctx%mat, mat_ctx%coefficients, mat_ctx%mf_temp_vec(MF_VEC_TEMP), x, y)

   end subroutine petsc_matvec_poly_mf      

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine build_gmres_polynomial_inverse(matrix, poly_order, buffers, coefficients, &
                  poly_sparsity_order, matrix_free, reuse_mat, inv_matrix)

      ! Assembles a matrix which is an approximation to the inverse of a matrix using the 
      ! gmres polynomial coefficients 
      ! This finishes the async comms required to compute the coefficients as late as possible
      ! Can constrain the sparsity to that an arbritrary matrix power of the input by setting
      ! poly_sparsity_order < poly_order

      ! ~~~~~~
      type(tMat), intent(in)                            :: matrix
      integer, intent(in)                               :: poly_order
      type(tsqr_buffers), intent(inout)                 :: buffers
      PetscReal, dimension(:), target, intent(inout)         :: coefficients
      integer, intent(in)                               :: poly_sparsity_order
      logical, intent(in)                               :: matrix_free
      type(tMat), intent(inout)                         :: reuse_mat, inv_matrix

      ! Local variables
      PetscInt :: global_row_start, global_row_end_plus_one
      PetscInt :: global_rows, global_cols, local_rows, local_cols
      integer :: comm_size, errorcode, order
      PetscErrorCode :: ierr      
      MPI_Comm :: MPI_COMM_MATRIX
      type(tMat) :: mat_power, temp_mat
      type(mat_ctxtype), pointer :: mat_ctx
      logical :: reuse_triggered      

      ! ~~~~~~       

      ! We might want to call the gmres poly creation on a sub communicator
      ! so let's get the comm attached to the matrix and make sure to use that 
      call PetscObjectGetComm(matrix, MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)        

      ! Get the local sizes
      call MatGetLocalSize(matrix, local_rows, local_cols, ierr)
      call MatGetSize(matrix, global_rows, global_cols, ierr)
      ! This returns the global index of the local portion of the matrix
      call MatGetOwnershipRange(matrix, global_row_start, global_row_end_plus_one, ierr)     
      
      ! Just build a matshell that applies our polynomial matrix-free
      if (matrix_free) then

         ! Finish off the non-blocking all reduce to compute our coefficients
         call finish_gmres_polynomial_coefficients_power(poly_order, buffers, coefficients)

         ! If not re-using
         if (PetscMatIsNull(inv_matrix)) then

            ! Have to dynamically allocate this
            allocate(mat_ctx)
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR<22)      
            mat_ctx%mat_ida = PETSC_NULL_MAT
#endif             

            ! We pass in the polynomial coefficients as the context
            call MatCreateShell(MPI_COMM_MATRIX, local_rows, local_cols, global_rows, global_cols, &
                        mat_ctx, inv_matrix, ierr)
            ! The subroutine petsc_matvec_poly_mf applies the polynomial inverse
            call MatShellSetOperation(inv_matrix, &
                        MATOP_MULT, petsc_matvec_poly_mf, ierr)

            call MatAssemblyBegin(inv_matrix, MAT_FINAL_ASSEMBLY, ierr)
            call MatAssemblyEnd(inv_matrix, MAT_FINAL_ASSEMBLY, ierr) 
            ! Have to make sure to set the type of vectors the shell creates
            call ShellSetVecType(matrix, inv_matrix)              

            ! Create temporary vector we use during horner
            ! Make sure to use matrix here to get the right type (as the shell doesn't know about gpus)            
            call MatCreateVecs(matrix, mat_ctx%mf_temp_vec(MF_VEC_TEMP), PETSC_NULL_VEC, ierr)         

         ! Reusing 
         else
            call MatShellGetContext(inv_matrix, mat_ctx, ierr)
         end if
         
         ! This is the matrix whose inverse we are applying (just copying the pointer here)
         mat_ctx%mat = matrix 
         mat_ctx%coefficients => coefficients

         ! We're done
         return
      end if

      ! ~~~~~~~~~~~~
      ! If we're here then we want an assembled approximate inverse
      ! ~~~~~~~~~~~~
      reuse_triggered = .NOT. PetscMatIsNull(inv_matrix) 

      ! If we're zeroth order poly this is trivial as it's just the coefficient(1) on the diagonal
      if (poly_order == 0) then

         call build_gmres_polynomial_inverse_0th_order(matrix, poly_order, buffers, coefficients, &
               inv_matrix)         

         ! Then just return
         return

      ! For poly_order 1 and poly_sparsity_order 1 this is easy
      else if (poly_order == 1 .AND. poly_sparsity_order == 1) then

         ! Duplicate & copy the matrix, but ensure there is a diagonal present
         call mat_duplicate_copy_plus_diag(matrix, reuse_triggered, inv_matrix)

         ! Flags to prevent reductions when assembling (there are assembles in the shift)
         call MatSetOption(inv_matrix, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr) 
         call MatSetOption(inv_matrix, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE,  ierr)     
         call MatSetOption(inv_matrix, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)                   

         ! Finish off the non-blocking all reduce to compute our coefficients
         call finish_gmres_polynomial_coefficients_power(poly_order, buffers, coefficients)

         ! result = alpha_1 * A_ff
         call MatScale(inv_matrix, coefficients(2), ierr)
         ! Now have to go through and add alpha_0 to the other terms
         ! Don't need an assemble as there is one called in this
         call MatShift(inv_matrix, coefficients(1), ierr)       

         ! Then just return
         return

      end if    

      ! If we're constraining sparsity we've built a custom matrix-powers that assumes fixed sparsity
      if (poly_sparsity_order < poly_order) then       
               
         ! This routine is a custom one that builds our matrix powers and assumes fixed sparsity
         ! so that it doen't have to do much comms
         ! This also finishes off the asyn comms and computes the coefficients
         call mat_mult_powers_share_sparsity(matrix, poly_order, poly_sparsity_order, buffers, coefficients, &
                  reuse_mat, inv_matrix)     

         ! Then just return
         return

      end if

      ! ~~~~~~~~~~
      ! We are only here if we don't constrain_sparsity
      ! ~~~~~~~~~~

      ! If not re-using
      ! Copy in the initial matrix
      if (.NOT. reuse_triggered) then
         ! Duplicate & copy the matrix, but ensure there is a diagonal present
         call mat_duplicate_copy_plus_diag(matrix, .FALSE., inv_matrix)
      else
         ! For the powers > 1 the pattern of the original matrix will be different
         ! to the resulting inverse
         call MatCopy(matrix, inv_matrix, DIFFERENT_NONZERO_PATTERN, ierr)
      end if

      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(inv_matrix, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)  
      
      ! Finish off the non-blocking all reduce to compute our coefficients
      call finish_gmres_polynomial_coefficients_power(poly_order, buffers, coefficients)

      ! result = alpha_1 * A_ff
      call MatScale(inv_matrix, coefficients(2), ierr)

      ! ~~~~~~~~~~~~~
      ! For terms 2nd order and higher
      ! ~~~~~~~~~~~~~   

      do order = 2, poly_order

         ! Skip this term if the coefficient is zero
         if (coefficients(order+1) == 0d0) cycle

         ! TODO - these can be reused
         if (order == 2) then
            call MatMatMult(matrix, matrix, &
                  MAT_INITIAL_MATRIX, 1.5d0, temp_mat, ierr)     
         else
            call MatMatMult(matrix, mat_power, &
                  MAT_INITIAL_MATRIX, 1.5d0, temp_mat, ierr)      
            call MatDestroy(mat_power, ierr)
         end if       

         ! Copy the temporary into mat_power
         call MatDuplicate(temp_mat, MAT_COPY_VALUES, mat_power, ierr)
         call MatDestroy(temp_mat, ierr)

         ! Do result = alpha_1 * A_ff + alpha_2 * A_ff^2 + ....
         if (reuse_triggered) then
            ! If doing reuse we know our nonzeros are a subset
            call MatAXPY(inv_matrix, coefficients(order+1), mat_power, SUBSET_NONZERO_PATTERN, ierr)              
         else
            ! Have to use the DIFFERENT_NONZERO_PATTERN here
            call MatAXPY(inv_matrix, coefficients(order+1), mat_power, DIFFERENT_NONZERO_PATTERN, ierr)              
         end if
         
      end do

      ! The alpha_0 just gets added to the diagonal
      ! There is an assemble in the shift so don't need a separate one       
      call MatShift(inv_matrix, coefficients(1), ierr)       

   end subroutine build_gmres_polynomial_inverse  
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine build_gmres_polynomial_inverse_0th_order(matrix, poly_order, buffers, coefficients, &
                  inv_matrix)

      ! Wrapper around build_gmres_polynomial_inverse_0th_order_kokkos and 
      ! build_gmres_polynomial_inverse_0th_order_cpu

      ! ~~~~~~
      type(tMat), intent(in)                            :: matrix
      integer, intent(in)                               :: poly_order
      type(tsqr_buffers), intent(inout)                 :: buffers
      PetscReal, dimension(:), target, intent(inout)    :: coefficients
      type(tMat), intent(inout)                         :: inv_matrix
      
#if defined(PETSC_HAVE_KOKKOS)                     
      integer(c_long_long) :: A_array, B_array
      integer :: errorcode, reuse_int
      PetscErrorCode :: ierr
      MatType :: mat_type
      Mat :: temp_mat, temp_mat_reuse, temp_mat_compare
      PetscScalar normy;
      logical :: reuse_triggered
      type(c_ptr)  :: coefficients_ptr
#endif      
      ! ~~~~~~~~~~

#if defined(PETSC_HAVE_KOKKOS)    

      call MatGetType(matrix, mat_type, ierr)
      if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) then   
               
         ! Finish off the non-blocking all reduce to compute our coefficients
         ! This is a fortran routine and we don't want to call this inside the kokkos
         call finish_gmres_polynomial_coefficients_power(poly_order, buffers, coefficients)               

         A_array = matrix%v             
         reuse_triggered = .NOT. PetscMatIsNull(inv_matrix) 
         reuse_int = 0
         if (reuse_triggered) then
            reuse_int = 1
            B_array = inv_matrix%v
         end if
         coefficients_ptr = c_loc(coefficients)
         call build_gmres_polynomial_inverse_0th_order_kokkos(A_array, poly_order, coefficients_ptr, &
                  reuse_int, B_array)
         inv_matrix%v = B_array

         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then

            ! If we're doing reuse and debug, then we have to always output the result 
            ! from the cpu version, as it will have coo preallocation structures set
            ! They aren't copied over if you do a matcopy (or matconvert)
            ! If we didn't do that the next time we come through this routine 
            ! and try to call the cpu version with reuse, it will segfault
            if (reuse_triggered) then
               temp_mat = inv_matrix
               call MatConvert(inv_matrix, MATSAME, MAT_INITIAL_MATRIX, temp_mat_compare, ierr)  
            else
               temp_mat_compare = inv_matrix                         
            end if            

            ! Debug check if the CPU and Kokkos versions are the same
            call build_gmres_polynomial_inverse_0th_order_cpu(matrix, poly_order, buffers, coefficients, &
                     temp_mat)  
                     
            call MatConvert(temp_mat, MATSAME, MAT_INITIAL_MATRIX, &
                        temp_mat_reuse, ierr)                        

            call MatAXPY(temp_mat_reuse, -1d0, temp_mat_compare, DIFFERENT_NONZERO_PATTERN, ierr)
            call MatNorm(temp_mat_reuse, NORM_FROBENIUS, normy, ierr)
            if (normy .gt. 1d-13) then
               !call MatFilter(temp_mat, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
               !call MatView(temp_mat, PETSC_VIEWER_STDOUT_WORLD, ierr)
               print *, "Kokkos and CPU versions of build_gmres_polynomial_inverse_0th_order do not match"
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)  
            end if
            call MatDestroy(temp_mat_reuse, ierr)
            if (.NOT. reuse_triggered) then
               call MatDestroy(inv_matrix, ierr)
            else
               call MatDestroy(temp_mat_compare, ierr)
            end if
            inv_matrix = temp_mat
         end if

      else

         call build_gmres_polynomial_inverse_0th_order_cpu(matrix, poly_order, buffers, coefficients, &
                  inv_matrix)        

      end if
#else
      call build_gmres_polynomial_inverse_0th_order_cpu(matrix, poly_order, buffers, coefficients, &
            inv_matrix) 
#endif                

   end subroutine build_gmres_polynomial_inverse_0th_order
   
! -------------------------------------------------------------------------------------------------------------------------------

   subroutine build_gmres_polynomial_inverse_0th_order_cpu(matrix, poly_order, buffers, coefficients, &
                  inv_matrix)

      ! Specific 0th order inverse

      ! ~~~~~~
      type(tMat), intent(in)                            :: matrix
      integer, intent(in)                               :: poly_order
      type(tsqr_buffers), intent(inout)                 :: buffers
      PetscReal, dimension(:), target, intent(inout)    :: coefficients
      type(tMat), intent(inout)                         :: inv_matrix

      ! Local variables
      PetscInt :: global_row_start, global_row_end_plus_one, counter
      PetscInt :: global_rows, global_cols, local_rows, local_cols, j_loc
      integer :: comm_size, errorcode
      PetscErrorCode :: ierr      
      MPI_Comm :: MPI_COMM_MATRIX
      logical :: reuse_triggered
      MatType:: mat_type
      PetscInt, allocatable, dimension(:) :: indices
      PetscReal, allocatable, dimension(:) :: v       

      ! ~~~~~~      
      
      if (poly_order /= 0) then
         print *, "This is a 0th order inverse, but poly_order is not 0"
         call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)
      end if

      ! We might want to call the gmres poly creation on a sub communicator
      ! so let's get the comm attached to the matrix and make sure to use that 
      call PetscObjectGetComm(matrix, MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)        

      ! Get the local sizes
      call MatGetLocalSize(matrix, local_rows, local_cols, ierr)
      call MatGetSize(matrix, global_rows, global_cols, ierr)
      ! This returns the global index of the local portion of the matrix
      call MatGetOwnershipRange(matrix, global_row_start, global_row_end_plus_one, ierr)     

      reuse_triggered = .NOT. PetscMatIsNull(inv_matrix) 
      ! If not re-using
      if (.NOT. reuse_triggered) then

         call MatCreate(MPI_COMM_MATRIX, inv_matrix, ierr)
         call MatSetSizes(inv_matrix, local_rows, local_cols, &
                           global_rows, global_cols, ierr)
         ! Match the output type
         call MatGetType(matrix, mat_type, ierr)
         call MatSetType(inv_matrix, mat_type, ierr)
         call MatSetUp(inv_matrix, ierr)                

      end if        

      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(inv_matrix, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)
      call MatSetOption(inv_matrix, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)                     

      ! Finish off the non-blocking all reduce to compute our coefficients
      call finish_gmres_polynomial_coefficients_power(poly_order, buffers, coefficients)

      allocate(v(local_rows))
      ! Set the diagonal
      v = coefficients(1)      
      
      ! Set the diagonal
      if (.NOT. reuse_triggered) then
         allocate(indices(local_rows))

         ! Set the diagonal
         counter = 1
         do j_loc = global_row_start, global_row_end_plus_one-1
            indices(counter) = j_loc
            counter = counter + 1
         end do            
         call MatSetPreallocationCOO(inv_matrix, local_rows, indices, indices, ierr)
         deallocate(indices)

      end if

      call MatSetValuesCOO(inv_matrix, v, INSERT_VALUES, ierr)    
      deallocate(v)                     

   end subroutine build_gmres_polynomial_inverse_0th_order_cpu   

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine build_gmres_polynomial_inverse_0th_order_sparsity(matrix, poly_order, buffers, coefficients, &
                  inv_matrix)

      ! Wrapper around build_gmres_polynomial_inverse_0th_order_sparsity_kokkos and 
      ! build_gmres_polynomial_inverse_0th_order_sparsity_cpu

      ! ~~~~~~
      type(tMat), intent(in)                            :: matrix
      integer, intent(in)                               :: poly_order
      type(tsqr_buffers), intent(inout)                 :: buffers
      PetscReal, dimension(:), target, intent(inout)    :: coefficients
      type(tMat), intent(inout)                         :: inv_matrix
      
#if defined(PETSC_HAVE_KOKKOS)                     
      integer(c_long_long) :: A_array, B_array
      integer :: errorcode, reuse_int
      PetscErrorCode :: ierr
      MatType :: mat_type
      Mat :: temp_mat, temp_mat_reuse, temp_mat_compare
      PetscScalar normy;
      logical :: reuse_triggered
      type(c_ptr)  :: coefficients_ptr
#endif      
      ! ~~~~~~~~~~

#if defined(PETSC_HAVE_KOKKOS)    

      call MatGetType(matrix, mat_type, ierr)
      if (mat_type == MATMPIAIJKOKKOS .OR. mat_type == MATSEQAIJKOKKOS .OR. &
            mat_type == MATAIJKOKKOS) then   
               
         ! Finish off the non-blocking all reduce to compute our coefficients
         ! This is a fortran routine and we don't want to call this inside the kokkos
         call finish_gmres_polynomial_coefficients_power(poly_order, buffers, coefficients)               

         A_array = matrix%v             
         reuse_triggered = .NOT. PetscMatIsNull(inv_matrix) 
         reuse_int = 0
         if (reuse_triggered) then
            reuse_int = 1
            B_array = inv_matrix%v
         end if
         coefficients_ptr = c_loc(coefficients)
         call build_gmres_polynomial_inverse_0th_order_sparsity_kokkos(A_array, poly_order, coefficients_ptr, &
                  reuse_int, B_array)
         inv_matrix%v = B_array

         ! If debugging do a comparison between CPU and Kokkos results
         if (kokkos_debug()) then

            ! If we're doing reuse and debug, then we have to always output the result 
            ! from the cpu version, as it will have coo preallocation structures set
            ! They aren't copied over if you do a matcopy (or matconvert)
            ! If we didn't do that the next time we come through this routine 
            ! and try to call the cpu version with reuse, it will segfault
            if (reuse_triggered) then
               temp_mat = inv_matrix
               call MatConvert(inv_matrix, MATSAME, MAT_INITIAL_MATRIX, temp_mat_compare, ierr)  
            else
               temp_mat_compare = inv_matrix                         
            end if            

            ! Debug check if the CPU and Kokkos versions are the same
            call build_gmres_polynomial_inverse_0th_order_sparsity_cpu(matrix, poly_order, buffers, coefficients, &
                     temp_mat)  
                     
            call MatConvert(temp_mat, MATSAME, MAT_INITIAL_MATRIX, &
                        temp_mat_reuse, ierr)                        

            call MatAXPY(temp_mat_reuse, -1d0, temp_mat_compare, DIFFERENT_NONZERO_PATTERN, ierr)
            call MatNorm(temp_mat_reuse, NORM_FROBENIUS, normy, ierr)
            ! There is floating point compute in these inverses, so we have to be a 
            ! bit more tolerant to rounding differences
            if (normy .gt. 1d-11) then
               !call MatFilter(temp_mat, 1d-14, PETSC_TRUE, PETSC_FALSE, ierr)
               !call MatView(temp_mat, PETSC_VIEWER_STDOUT_WORLD, ierr)
               print *, "Kokkos and CPU versions of build_gmres_polynomial_inverse_0th_order_sparsity do not match"
               call MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER, errorcode)  
            end if
            call MatDestroy(temp_mat_reuse, ierr)
            if (.NOT. reuse_triggered) then
               call MatDestroy(inv_matrix, ierr)
            else
               call MatDestroy(temp_mat_compare, ierr)
            end if
            inv_matrix = temp_mat
         end if

      else

         call build_gmres_polynomial_inverse_0th_order_sparsity_cpu(matrix, poly_order, buffers, coefficients, &
                  inv_matrix)        

      end if
#else
      call build_gmres_polynomial_inverse_0th_order_sparsity_cpu(matrix, poly_order, buffers, coefficients, &
            inv_matrix) 
#endif                      

   end subroutine build_gmres_polynomial_inverse_0th_order_sparsity  

! -------------------------------------------------------------------------------------------------------------------------------

   subroutine build_gmres_polynomial_inverse_0th_order_sparsity_cpu(matrix, poly_order, buffers, coefficients, &
                  inv_matrix)

      ! Specific inverse with 0th order sparsity

      ! ~~~~~~
      type(tMat), intent(in)                            :: matrix
      integer, intent(in)                               :: poly_order
      type(tsqr_buffers), intent(inout)                 :: buffers
      PetscReal, dimension(:), target, intent(inout)    :: coefficients
      type(tMat), intent(inout)                         :: inv_matrix

      ! Local variables
      PetscInt :: global_row_start, global_row_end_plus_one, counter
      PetscInt :: global_rows, global_cols, local_rows, local_cols, i_loc
      integer :: comm_size, errorcode, order
      PetscErrorCode :: ierr      
      MPI_Comm :: MPI_COMM_MATRIX
      logical :: reuse_triggered
      MatType:: mat_type
      PetscInt, allocatable, dimension(:) :: indices
      type(tVec) :: rhs_copy, diag_vec, power_vec
      ! ~~~~~~      

      call PetscObjectGetComm(matrix, MPI_COMM_MATRIX, ierr)    
      ! Get the comm size 
      call MPI_Comm_size(MPI_COMM_MATRIX, comm_size, errorcode)
      call MatGetType(matrix, mat_type, ierr)

      ! Get the local sizes
      call MatGetLocalSize(matrix, local_rows, local_cols, ierr)
      call MatGetSize(matrix, global_rows, global_cols, ierr)
      ! This returns the global index of the local portion of the matrix
      call MatGetOwnershipRange(matrix, global_row_start, global_row_end_plus_one, ierr)  

      reuse_triggered = .NOT. PetscMatIsNull(inv_matrix)

       ! Our matrix has to be square
      call MatCreateVecs(matrix, rhs_copy, diag_vec, ierr)
      call MatGetDiagonal(matrix, diag_vec, ierr)

      ! This stores D^order
      call VecDuplicate(diag_vec, power_vec, ierr)
      call VecCopy(diag_vec, power_vec, ierr)

      ! ~~~~~~~~~~~      
      ! Finish the non-blocking all-reduce and compute our polynomial coefficients
      ! ~~~~~~~~~~~
      call finish_gmres_polynomial_coefficients_power(poly_order, buffers, coefficients)         

      ! Set: alpha_0 * I
      call VecSet(rhs_copy, coefficients(1), ierr)

      ! Calculate: alpha_0 * I + alpha_1 * D + alpha_2 * D^2
      do order = 1, poly_order
         call VecAXPY(rhs_copy, coefficients(order+1), power_vec, ierr)
         ! Compute power_vec = power_vec * D
         call VecPointwiseMult(power_vec, power_vec, diag_vec, ierr)
      end do

      ! If not re-using
      if (.NOT. reuse_triggered) then

         call MatCreate(MPI_COMM_MATRIX, inv_matrix, ierr)
         call MatSetSizes(inv_matrix, local_rows, local_cols, &
                          global_rows, global_cols, ierr)
         ! Match the output type
         call MatSetType(inv_matrix, mat_type, ierr)
         call MatSetUp(inv_matrix, ierr)

      end if

      ! Don't set any off processor entries so no need for a reduction when assembling
      call MatSetOption(inv_matrix, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr)
      call MatSetOption(inv_matrix, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE,  ierr)                

      if (.NOT. reuse_triggered) then
         allocate(indices(local_rows))

         ! Set the diagonal
         counter = 1
         do i_loc = global_row_start, global_row_end_plus_one-1
            indices(counter) = i_loc
            counter = counter + 1
         end do
         ! Set the diagonal
         call MatSetPreallocationCOO(inv_matrix, local_rows, indices, indices, ierr)
         deallocate(indices)

      end if                   

      ! Set the diagonal to our polynomial
      call MatDiagonalSet(inv_matrix, rhs_copy, INSERT_VALUES, ierr)
   
      call VecDestroy(diag_vec, ierr)
      call VecDestroy(rhs_copy, ierr) 
      call VecDestroy(power_vec, ierr)                      

   end subroutine build_gmres_polynomial_inverse_0th_order_sparsity_cpu      
   

! -------------------------------------------------------------------------------------------------------------------------------

end module gmres_poly

