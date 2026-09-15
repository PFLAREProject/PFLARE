module air_mg_apply_transpose

   use iso_c_binding
   use petscksp
   use air_data_type, only: air_multigrid_data
   use fc_smooth, only: mg_FC_point_richardson_transpose
   use c_petsc_interfaces, only: PCMGGetRhs_c, PCMGGetX_c, PCMGGetR_c

#include "petsc/finclude/petscksp.h"

   implicit none
   public

   ! -------------------------------------------------------------------------------------------------------------------------------
   ! Applies the transpose of the air multigrid
   ! -------------------------------------------------------------------------------------------------------------------------------

   contains

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine apply_air_transpose(air_data, pcmg, b, y, ierr)

      ! Applies the exact transpose of what apply does, y = M^T b
      !
      ! We can't just hand this to petsc for the default F-C smoothing, for two
      ! reasons. PCMGKCycle_Private ignores the transpose flag entirely, so a
      ! PCApplyTranspose on the pcmg would silently run the forward cycle. And even
      ! if it didn't, the transposed cycle drives the level smoothers with
      ! KSPSolveTranspose, which skips the PCApplyRichardson fast path our F-C
      ! smoother lives in and would then want a MatMultTranspose on the level
      ! operator, which is an empty placeholder matshell when we're F-C smoothing.
      ! So we walk the levels here instead.
      !
      ! The forward kaskade cycle is, with E_l and N_l the iteration and smoothing
      ! matrices of the level smoother (x_new = E_l x + N_l b, E_l = I - N_l A_l),
      ! P_l the prolongator, R_l the restrictor and C the coarse solve
      !
      !     M_l = E_l P_l M_{l+1} R_l + N_l,      M_N = C
      !
      ! so transposing
      !
      !     M_l^T = R_l^T M_{l+1}^T P_l^T E_l^T + N_l^T
      !
      ! which is the down loop / coarse solve / up loop below. There is no post
      ! smooth as the forward kaskade has no pre smooth

      ! ~~~~~~
      type(air_multigrid_data), target, intent(inout) :: air_data
      type(tPC), intent(inout)                        :: pcmg
      type(tVec), intent(inout)                       :: b, y
      PetscErrorCode, intent(inout)                   :: ierr

      integer                 :: our_level, no_levels
      PetscInt                :: petsc_level
      type(tKSP)              :: ksp_coarse_solver, ksp_smoother_down
      type(tVec)              :: b_level, x_level, r_level, b_coarse, x_coarse
      PetscBool, allocatable, dimension(:) :: guess_nonzero

      ! ~~~~~~

      ierr = 0
      no_levels = air_data%no_levels

      ! With a single level the pc underneath us is not a PCMG at all, it's a PCMAT
      ! with our truncated coarse inverse or a PCJACOBI, both of which know their
      ! own transpose
      if (no_levels == 1) then
         call PCApplyTranspose(pcmg, b, y, ierr)
         return
      end if

      ! ~~~~~~~~~~~~~~~
      ! Full smoothing up and down is a plain petsc multiplicative v-cycle, and
      ! PCMGMCycle_Private with the transpose flag set is algebraically exactly its
      ! transpose, so we let petsc drive it. That also means any -mg_levels_*
      ! overrides the user has set keep working
      ! ~~~~~~~~~~~~~~~
      if (air_data%options%full_smoothing_up_and_down) then

         ! The one thing petsc gets wrong for us is the initial guess. The
         ! transposed cycle finishes each level with KSPSolveTranspose(smoothd, b, x)
         ! on a nonzero x holding the coarse grid correction, but PCSetUp_MG only
         ! turns the nonzero initial guess on for smoothd when smoothd and smoothu
         ! are the same ksp (they are not for us, we ask for a distinct up smoother),
         ! and KSPSolve_Private zeros vec_sol when guess_zero is set, which would
         ! throw the coarse grid correction away.
         ! We put the flags back as we found them afterwards - leaving them on would
         ! cost the forward pre-smooth a wasted matvec against zero on every level
         allocate(guess_nonzero(no_levels-1))
         do petsc_level = 1, no_levels-1
            call PCMGGetSmootherDown(pcmg, petsc_level, ksp_smoother_down, ierr)
            call KSPGetInitialGuessNonzero(ksp_smoother_down, guess_nonzero(petsc_level), ierr)
            call KSPSetInitialGuessNonzero(ksp_smoother_down, PETSC_TRUE, ierr)
         end do

         call PCApplyTranspose(pcmg, b, y, ierr)

         do petsc_level = 1, no_levels-1
            call PCMGGetSmootherDown(pcmg, petsc_level, ksp_smoother_down, ierr)
            call KSPSetInitialGuessNonzero(ksp_smoother_down, guess_nonzero(petsc_level), ierr)
         end do
         deallocate(guess_nonzero)
         return
      end if

      ! ~~~~~~~~~~~~~~~
      ! Down: on each level the transposed smoother from a zero initial guess gives
      ! x_l = N_l^T b_l and r_l = E_l^T b_l, and then P_l^T restricts the residual.
      ! This is the transpose of the forward kaskade restricting b with R_l on the
      ! way down and smoothing with the interpolated guess on the way up
      ! ~~~~~~~~~~~~~~~
      do our_level = 1, no_levels-1

         petsc_level = no_levels - our_level

         ! We don't allocate any work vectors of our own - PCSetUp_MG has already
         ! built a rhs and a solution on petsc levels 0 to N-2 and a residual on
         ! petsc levels 1 to N-1, which is exactly what we need. They belong to
         ! petsc, so we refetch them every apply and never destroy them.
         ! On the finest level the rhs and solution are what we were handed
         if (our_level == 1) then
            b_level = b
            x_level = y
         else
            call get_pcmg_rhs(pcmg, petsc_level, b_level)
            call get_pcmg_x(pcmg, petsc_level, x_level)
         end if
         call get_pcmg_r(pcmg, petsc_level, r_level)

         call mg_FC_point_richardson_transpose(air_data, our_level, b_level, x_level, r_level)

         ! b_coarse = P_l^T r_l
         call get_pcmg_rhs(pcmg, petsc_level-1, b_coarse)
         call MatMultTranspose(air_data%prolongators(our_level), r_level, b_coarse, ierr)

      end do

      ! ~~~~~~~~~~~~~~~
      ! The transposed coarse grid solve. Going through the ksp rather than
      ! straight to our shell is what keeps any -mg_coarse_* the user has set
      ! working, e.g. -mg_coarse_pc_type lu solves with the transposed factors
      ! ~~~~~~~~~~~~~~~
      call PCMGGetCoarseSolve(pcmg, ksp_coarse_solver, ierr)
      petsc_level = 0
      call get_pcmg_rhs(pcmg, petsc_level, b_coarse)
      call get_pcmg_x(pcmg, petsc_level, x_coarse)
      call KSPSolveTranspose(ksp_coarse_solver, b_coarse, x_coarse, ierr)

      ! ~~~~~~~~~~~~~~~
      ! Up: add R_l^T of the coarse solution into each level. This is the transpose
      ! of the forward kaskade interpolating with P_l on the way up
      ! ~~~~~~~~~~~~~~~
      do our_level = no_levels-1, 1, -1

         petsc_level = no_levels - our_level

         if (our_level == 1) then
            x_level = y
         else
            call get_pcmg_x(pcmg, petsc_level, x_level)
         end if
         call get_pcmg_x(pcmg, petsc_level-1, x_coarse)

         if (air_data%options%symmetric) then
            ! We never store the restrictor when symmetric, R_l is P_l^T so R_l^T is P_l
            call MatMultAdd(air_data%prolongators(our_level), x_coarse, x_level, x_level, ierr)
         else
            call MatMultTransposeAdd(air_data%restrictors(our_level), x_coarse, x_level, x_level, ierr)
         end if

      end do

   end subroutine apply_air_transpose

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine get_pcmg_rhs(pcmg, petsc_level, vec)

      ! Thin wrappers around the C accessors for the per-level work vectors of a
      ! PCMG - see the comments above PCMGGetRhs_c in C_PETSc_Routines.c for why
      ! these have to go through C at all

      ! ~~~~~~
      type(tPC), intent(in)        :: pcmg
      PetscInt, intent(in)         :: petsc_level
      type(tVec), intent(inout)    :: vec

      integer(c_long_long) :: pc_array, vec_array
      ! ~~~~~~

      pc_array = pcmg%v
      call PCMGGetRhs_c(pc_array, petsc_level, vec_array)
      vec%v = vec_array

   end subroutine get_pcmg_rhs

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine get_pcmg_x(pcmg, petsc_level, vec)

      ! ~~~~~~
      type(tPC), intent(in)        :: pcmg
      PetscInt, intent(in)         :: petsc_level
      type(tVec), intent(inout)    :: vec

      integer(c_long_long) :: pc_array, vec_array
      ! ~~~~~~

      pc_array = pcmg%v
      call PCMGGetX_c(pc_array, petsc_level, vec_array)
      vec%v = vec_array

   end subroutine get_pcmg_x

   ! -------------------------------------------------------------------------------------------------------------------------------

   subroutine get_pcmg_r(pcmg, petsc_level, vec)

      ! ~~~~~~
      type(tPC), intent(in)        :: pcmg
      PetscInt, intent(in)         :: petsc_level
      type(tVec), intent(inout)    :: vec

      integer(c_long_long) :: pc_array, vec_array
      ! ~~~~~~

      pc_array = pcmg%v
      call PCMGGetR_c(pc_array, petsc_level, vec_array)
      vec%v = vec_array

   end subroutine get_pcmg_r

   ! -------------------------------------------------------------------------------------------------------------------------------

end module air_mg_apply_transpose
