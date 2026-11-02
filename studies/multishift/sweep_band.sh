cd /work/PFLARE/tests
for lo in 0.001 0.01 0.1 1 10 100; do
  c=$(python3 -c "import numpy as np;print(','.join('%g'%x for x in np.logspace(np.log10($lo),np.log10(4*$lo),16)))")
  for fr in 1 2 4; do
    cf=$(python3 -c "print('%g'%($fr*$lo))")
    echo "### band [$lo, 4*$lo] c_base=$lo c_frozen=$cf"
    ./adv_diff_fd_multishift -n 200 -solve -richardson -frozen -max_extra_its -1 -c_base $lo -c_frozen $cf -c_values $c 2>&1 | grep -E "Richardson: (multi|frozen)|Richardson solve|^Solve|^Setup|^Apply|^Frozen|^ +[0-9]"
  done
done
