cd /work/PFLARE/tests
for lo in 0.01 0.1 1 10; do
  hi=$(python3 -c "print('%g'%(4*$lo))")
  c=$(python3 -c "import numpy as np;print(','.join('%g'%x for x in np.logspace(np.log10($lo),np.log10(4*$lo),16)))")
  echo "### decrease band [$lo, $hi] c_base=$hi"
  ./adv_diff_fd_multishift -n 200 -solve -richardson -frozen -max_extra_its -1 -c_base $hi -c_frozen $hi -c_values $c 2>&1 | grep -E "Max relative|Richardson: (multi|frozen)|^ +[0-9]"
done
