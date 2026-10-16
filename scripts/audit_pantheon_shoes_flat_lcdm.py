#!/usr/bin/env python3
"""Audit Pantheon+ + SH0ES calibrated flat-LambdaCDM likelihood."""

from pathlib import Path

from late_time_cosmology.pantheon_shoes import (
    PantheonSH0ESFlatLCDM,
)

ROOT = Path(__file__).resolve().parents[1]

DATA = (
    ROOT
    / "data"
    / "pantheon_plus"
    / "Pantheon+SH0ES.dat"
)

COV = (
    ROOT
    / "data"
    / "pantheon_plus"
    / "Pantheon+SH0ES_STAT+SYS.cov"
)

like = PantheonSH0ESFlatLCDM(
    DATA,
    COV,
)

print("Pantheon+ + SH0ES calibrated likelihood")
print("=" * 54)
print("Selected rows      :", like.n)
print("Cepheid calibrators:", like.n_calibrators)
print("Hubble-flow rows   :", like.n_hubble_flow)
print(
    "zHD range         :",
    float(like.z_hd.min()),
    "to",
    float(like.z_hd.max()),
)
print(
    "max raw |C-C^T|  :",
    like.max_raw_asymmetry,
)
print(
    "conditional sigma(M_B):",
    like.sigma_MB_conditional,
)

opt = like.optimize()

H0_best = float(opt.x[0])
Om_best = float(opt.x[1])
MB_best = like.best_MB(
    H0_best,
    Om_best,
)

print()
print("Best fit")
print("--------")
print(f"H0       = {H0_best:.8f} km/s/Mpc")
print(f"Omega_m  = {Om_best:.8f}")
print(f"M_B      = {MB_best:.8f}")
print(f"chi2_min = {float(opt.fun):.8f}")

print()
print("Published Pantheon+ & SH0ES flat-LambdaCDM benchmark")
print("-----------------------------------------------------")
print("H0      = 73.6 +/- 1.1 km/s/Mpc")
print("Omega_m = 0.334 +/- 0.018")
