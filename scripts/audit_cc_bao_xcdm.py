#!/usr/bin/env python3
from pathlib import Path

from late_time_cosmology.cc import CCLikelihood
from late_time_cosmology.bao_xcdm import BAOFlatXCDM
from late_time_cosmology.cc_bao_xcdm_fit import (
    CCFlatXCDM,
    optimize_cc_bao_xcdm,
    derived,
    chi2_components,
)

ROOT = Path(__file__).resolve().parents[1]

cc_base = CCLikelihood(
    ROOT
    / "data"
    / "cc"
)

cc = CCFlatXCDM(
    cc_base
)

bao = BAOFlatXCDM(
    ROOT
    / "data"
    / "bao"
)

print("CC + BAO flat XCDM deterministic audit")
print("=" * 48)

opt = optimize_cc_bao_xcdm(
    cc,
    bao,
)

d = derived(
    opt.x
)

parts = chi2_components(
    opt.x,
    cc,
    bao,
)

print()
print("Best fit")
print("--------")
print(f"h            = {d['h']:.9f}")
print(f"H0           = {d['H0_km_s_Mpc']:.9f} km/s/Mpc")
print(f"Omega_b h^2  = {d['omega_b_h2']:.9f}")
print(f"Omega_c h^2  = {d['omega_c_h2']:.9f}")
print(f"Omega_m      = {d['Omega_m']:.9f}")
print(f"w_X          = {d['w_X']:.9f}")
print(f"r_s          = {d['r_s_Mpc']:.9f} Mpc")
print(f"r_s h        = {d['r_s_h_Mpc']:.9f} Mpc")
print(f"chi2_min     = {sum(parts.values()):.9f}")

print()
print("chi2 components")
print("---------------")
for key, value in parts.items():
    print(f"{key:16s}: {value:.9f}")
print(
    f"{'total':16s}: "
    f"{sum(parts.values()):.9f}"
)

print()
print("Cao & Ratra (2023) updated H(z)+BAO flat-XCDM benchmark")
print("---------------------------------------------------------")
print("H0      = 67.18 +/- 3.18 km/s/Mpc")
print("Omega_m = 0.285 +/- 0.019")
print("w_X     = -0.776 +0.130 -0.103")
print()
print(
    "Exact agreement is not expected because this project uses the "
    "32-point CC vector with the correlated Moresco covariance."
)
