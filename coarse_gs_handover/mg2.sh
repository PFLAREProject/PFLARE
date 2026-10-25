source /work/gs/env.sh
np=$1; grid=$2; k=$3; Ls=$4
base="-da_grid_x $grid -da_grid_y $grid -pc_type air -ksp_type richardson -ksp_rtol 1e-10 -ksp_max_it 200 -ksp_norm_type unpreconditioned -ksp_converged_reason -log_view"
run() { mpirun -n $np $T/adv_diff_fd $base "$@" > r.txt 2>&1; it=$(grep -oE "iterations [0-9]+" r.txt | awk '{print $2}'); t=$(grep -E "^KSPSolve " r.txt | awk '{print $4}'); su=$(grep -E "^PCSetUp " r.txt | awk '{print $4}'); echo "$it its, solve ${t}s"; }
echo "np=$np grid=$grid k=$k"
printf "%3s | %-24s | %-24s\n" L "newton-mf(k) coarse" "richardson(k)+localGS"
for L in $Ls; do
  a=$(run -pc_air_max_levels $L -pc_air_coarsest_inverse_type newton -pc_air_coarsest_matrix_free_polys -pc_air_coarsest_poly_order $k)
  c=$(run -pc_air_max_levels $L -mg_coarse_ksp_type richardson -mg_coarse_ksp_max_it $k -mg_coarse_pc_type sor -mg_coarse_pc_sor_local_forward)
  printf "%3s | %-24s | %-24s\n" $L "$a" "$c"
done
