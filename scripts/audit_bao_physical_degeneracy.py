#!/usr/bin/env python3
"""Demonstrate that very different H0/r_s values can have the same BAO fit."""

from pathlib import Path

from late_time_cosmology.bao_reduced import BAOReducedFlatLCDM
from late_time_cosmology.bao_physical_fit import derived

ROOT = Path(__file__).resolve().parents[1]
like = BAOReducedFlatLCDM(ROOT / "data" / "bao")

points = [
    (0.69189404, 0.02417695, 0.11718890),
    (0.79869342, 0.04199697, 0.14637896),
]

for i, p in enumerate(points, 1):
    d = derived(p)
    chi2 = like.chi2(d["Omega_m"], d["r_s_h_Mpc"])

    print(f"Point {i}")
    print(f"  H0       = {d['H0']:.6f}")
    print(f"  Omega_bh2= {d['omega_b_h2']:.8f}")
    print(f"  Omega_ch2= {d['omega_c_h2']:.8f}")
    print(f"  Omega_m  = {d['Omega_m']:.9f}")
    print(f"  r_s      = {d['r_s_Mpc']:.6f} Mpc")
    print(f"  r_s h    = {d['r_s_h_Mpc']:.6f} Mpc")
    print(f"  chi2     = {chi2:.9f}")
    print()
