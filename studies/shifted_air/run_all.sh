#!/bin/bash
# Runs every capability study in the CI image: docker exec pf bash run_all.sh
cd "$(dirname "$0")"; . ./common.sh; mkdir -p results
R() { name=$1; shift; echo "== $name: $*" > results/$name.log; "$@" >> results/$name.log 2>&1; tail -2 results/$name.log | head -1; }
G=$(cat groups16.txt); SP=$(cat species_dt.txt); RA=$(cat radau_dt.txt)
# E1 multigroup 2D, velocity fields x material contrast
for v in const curved recirc; do for c in 1 100 10000; do
  a=0; [ $v = recirc ] && a=1e-3
  R e1_${v}_c${c} ./shift_study -n 400 -vel $v -alpha $a -contrast $c -shifts $G $OPTS
done; done
# E2 multigroup 3D
for v in const curved; do for c in 1 100; do
  R e2_3d_${v}_c${c} ./shift_study -dim 3 -n 50 -vel $v -contrast $c -shifts $G $OPTS
done; done
# E3 where to build: least (s=0), middle (s=1) and most (s=1000) diagonally dominant, with frozen
for sb in 0 1 1000; do
  R e3_build_${sb} ./shift_study -n 200 -vel curved -contrast 100 -shifts $G -s_build $sb -frozen $OPTS
done
R e3_build_1000_unscaled ./shift_study -n 200 -vel curved -contrast 100 -shifts $G -s_build 1000 -rebuild 0 ${OPTS/-pc_air_diag_scale_polys -pc_air_coarsest_diag_scale_polys/}
# E4 refinement
for n in 100 200 400 800; do
  R e4_n${n} ./shift_study -n $n -vel curved -contrast 100 -shifts $G $OPTS
done
# E5 multi-species x time steps (150 systems), with and without bands of 1/shift
R e5_species ./shift_study -n 300 -vel curved -contrast 1 -shifts $SP $OPTS
R e5_species_bands2 ./shift_study -n 300 -vel curved -contrast 1 -shifts $SP -rebuild 0 -bands 2 $OPTS
R e5_species_keep ./shift_study -n 300 -vel curved -contrast 1 -shifts $SP -rebuild 0 -pc_air_reuse_poly_coeffs $OPTS
# E6 Radau IIA stage shifts
R e6_radau ./shift_study -n 300 -vel recirc -alpha 1e-3 -shifts $RA $OPTS
R e6_radau_bands2 ./shift_study -n 300 -vel recirc -alpha 1e-3 -shifts $RA -rebuild 0 -bands 2 $OPTS
# E7 diffusion strength
for a in 0.01 0.1 1; do
  R e7_alpha${a} ./shift_study -n 300 -vel curved -alpha $a -contrast 100 -shifts $G $OPTS
done
# E8 parallel
R e8_np4 mpiexec -n 4 ./shift_study -n 800 -vel curved -contrast 100 -shifts $G $OPTS
R e8_np1 ./shift_study -n 800 -vel curved -contrast 100 -shifts $G $OPTS -rebuild 0
# E9 multiple right-hand sides on an updated hierarchy
for k in 1 4 8 16 32 64; do
  R e9_mrhs${k} ./shift_study -n 400 -vel curved -contrast 100 -shifts 0,1 -rebuild 0 -nrhs $k $OPTS
done
echo ALLDONE
