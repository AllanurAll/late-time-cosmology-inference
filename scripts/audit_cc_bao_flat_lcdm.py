#!/usr/bin/env python3
"""Deterministic CC+BAO flat-LambdaCDM audit before MCMC."""

from pathlib import Path
import json

from late_time_cosmology.cc import CCLikelihood
from late_time_cosmology.bao_reduced import BAOReducedFlatLCDM
from late_time_cosmology.cc_bao_fit import (
    optimize_cc_bao,
    derived,
    chi2_components,
)

ROOT = Path(__file__).resolve().parents[1]

cc = CCLikelihood(
    ROOT / "data" / "cc"
)

bao = BAOReducedFlatLCDM(
    ROOT / "data" / "bao"
)

opt = optimize_cc_bao(
    cc,
    bao,
)

cosmo, d = derived(
    opt.x
)

pieces = chi2_components(
    opt.x,
    cc,
    bao,
)

print("success:", opt.success)
print("message:", opt.message)
print()
print("Best-fitting sampled parameters:")
print(f"  h            = {d['h']:.9f}")
print(f"  Omega_b h^2  = {d['omega_b_h2']:.9f}")
print(f"  Omega_c h^2  = {d['omega_c_h2']:.9f}")
print()
print("Derived:")
print(f"  H0           = {d['H0_km_s_Mpc']:.9f} km/s/Mpc")
print(f"  Omega_m      = {d['Omega_m']:.9f}")
print(f"  r_s          = {d['r_s_Mpc']:.9f} Mpc")
print(f"  r_s h        = {d['r_s_h_Mpc']:.9f} Mpc")
print()
print(f"chi2_min       = {opt.fun:.9f}")
print()
print("chi2 components:")
for key, value in pieces.items():
    print(f"  {key:16s}: {value:.9f}")
print(f"  {'total':16s}: {sum(pieces.values()):.9f}")

print()
print("Reference benchmark from Cao, Ryan & Ratra (2021)")
print("  H0          = 69.11 km/s/Mpc")
print("  Omega_b h^2 = 0.0240")
print("  Omega_c h^2 = 0.1179")
print("  Omega_m     = 0.299")
print("  chi2        = 23.64")
print()
print(
    "Exact agreement is not expected because this reconstruction uses "
    "the updated 32-point CC vector and correlated CC covariance."
)
