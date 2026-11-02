cd /work/PFLARE/tests
for k in 1 4 8 16 32 64; do
  c=$(python3 -c "import numpy as np;k=$k;v=[0.0]+list(np.logspace(-4,3,k-1)) if k>1 else [1.0];print(','.join('%g'%x for x in v))")
  echo "### k=$k"
  ./adv_diff_fd_multishift -n 200 -solve -max_extra_its 0 -c_values $c 2>&1 | grep -E "seq its|^ +[0-9]|Solve|Setup|Apply|Max" 
done
