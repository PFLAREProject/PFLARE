"""Generates the shift sequences used by run_all.sh, one comma separated list per file."""
import numpy as np
w = lambda name, s: open(name, 'w').write(','.join('%.6g' % v for v in s))
# Multigroup: streaming group plus 15 groups with removal over 7 decades
w('groups16.txt', np.r_[0, np.logspace(-4, 3, 15)])
# Multi-species transport with a growing time step: shift = R_i/dt_n + lambda_i (lumped mass, grid units)
dts = 0.05 * (20 / 0.05) ** (np.arange(25) / 24)
species = [(1, 0), (2, 1e-3), (5, 1e-2), (10, 0.1), (20, 0.5), (50, 1.0)]
w('species_dt.txt', [R / dt + lam for dt in dts for (R, lam) in species])
# Radau IIA(3) stage shifts (real eigenvalue and real part of the complex pair of A^{-1}), dt growing
dts = 0.05 * (20 / 0.05) ** (np.arange(30) / 29)
w('radau_dt.txt', [g / dt for dt in dts for g in (3.6378, 2.6811)])
