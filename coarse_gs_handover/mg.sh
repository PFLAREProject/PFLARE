source /work/gs/env.sh
np=$1; grid=$2; k=$3; Ls=$4
base="-da_grid_x $grid -da_grid_y $grid -pc_type air -ksp_type fgmres -ksp_rtol 1e-10 -ksp_max_it 300 -ksp_norm_type unpreconditioned -ksp_converged_reason"
its() { mpirun -n $np $T/adv_diff_fd $base "$@" 2>&1 | grep -oE "iterations [0-9]+|DIVERGED.*" | head -1; }
nlev() { mpirun -n $np $T/adv_diff_fd $base -pc_air_print_stats_timings "$@" 2>&1 | grep -c "~~~~~~~~~~~~ Level"; }
echo "np=$np grid=$grid k=$k"
printf "%5s %10s | %-14s | %-14s | %-14s | %-14s\n" L coarseN "newton-mf(k)" "gmres(k)" "gmres(k)+GSfwd" "gmres(k)+GSbwd"
for L in $Ls; do
  N=""
  a=$(its -pc_air_max_levels $L -pc_air_coarsest_inverse_type newton -pc_air_coarsest_matrix_free_polys -pc_air_coarsest_poly_order $k)
  b=""
  c=$(its -pc_air_max_levels $L -mg_coarse_ksp_type gmres -mg_coarse_ksp_max_it $k -mg_coarse_ksp_pc_side right -mg_coarse_pc_type sor -mg_coarse_pc_sor_local_forward)
  d=""
  printf "%5s %10s | %-14s | %-14s | %-14s | %-14s\n" $L "$N" "$a" "$b" "$c" "$d"
done
