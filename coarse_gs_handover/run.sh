# run.sh np levels "extra args" k-list
source /work/gs/env.sh
np=$1; L=$2; extra=$3; ks=$4
rm -f c.dat c.dat.info; mpirun -n $np $T/${EXE:-adv_diff_fd} -pc_type air -ksp_type richardson -pc_air_max_levels $L -ksp_max_it 1 -mg_coarse_ksp_view_mat binary:c.dat $extra > log.txt 2>&1; test -s c.dat || { tail -5 log.txt; exit 1; }
mpirun -n $np python3 ranges.py c.dat > ranges.txt
python3 coarse_poly.py c.dat $ks ranges.txt
