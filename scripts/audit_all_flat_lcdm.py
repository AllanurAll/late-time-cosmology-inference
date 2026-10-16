#!/usr/bin/env python3
"""Audit the full Pantheon+SH0ES + CC + BAO flat-LambdaCDM likelihood."""

from pathlib import Path

from late_time_cosmology.cc import CCLikelihood
from late_time_cosmology.bao_reduced import BAOReducedFlatLCDM
from late_time_cosmology.pantheon_shoes import PantheonSH0ESFlatLCDM
from late_time_cosmology.all_flat_fit import (
    PantheonSH0ESFastProfile,
    optimize_all,
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

sn = PantheonSH0ESFlatLCDM(
    ROOT / "data" / "pantheon_plus" / "Pantheon+SH0ES.dat",
    ROOT / "data" / "pantheon_plus" / "Pantheon+SH0ES_STAT+SYS.cov",
)

print("Building fast Pantheon+SH0ES profile cache...")
sn_fast = PantheonSH0ESFastProfile(
    sn,
    omega_min=0.20,
    omega_max=0.45,
    n_omega=801,
)

points = [
    (73.5, 0.332),
    (72.5, 0.310),
    (71.0, 0.295),
    (74.5, 0.350),
]

err = sn_fast.max_abs_error(
    points
)

print(
    "Max |chi2_fast - chi2_exact| on audit points =",
    err,
)

if err > 1e-4:
    raise RuntimeError(
        "Pantheon+SH0ES spline cache is not accurate enough."
    )

print()
print("Optimizing full joint likelihood...")

opt = optimize_all(
    cc,
    bao,
    sn_fast,
)

_, d = derived(
    opt.x
)

parts = chi2_components(
    opt.x,
    cc,
    bao,
    sn_fast,
)

MB = sn.best_MB(
    d["H0_km_s_Mpc"],
    d["Omega_m"],
)

print()
print("Best fit")
print("--------")
print(f"h            = {d['h']:.9f}")
print(f"H0           = {d['H0_km_s_Mpc']:.9f} km/s/Mpc")
print(f"Omega_b h^2  = {d['omega_b_h2']:.9f}")
print(f"Omega_c h^2  = {d['omega_c_h2']:.9f}")
print(f"Omega_m      = {d['Omega_m']:.9f}")
print(f"r_s          = {d['r_s_Mpc']:.9f} Mpc")
print(f"r_s h        = {d['r_s_h_Mpc']:.9f} Mpc")
print(f"M_B          = {MB:.9f}")
print(f"chi2_min     = {sum(parts.values()):.9f}")

print()
print("chi2 components")
print("---------------")
for key, value in parts.items():
    print(f"{key:22s}: {value:.9f}")

print(
    f"{'total':22s}: "
    f"{sum(parts.values()):.9f}"
)

print()
print("Internal reference results")
print("--------------------------")
print("Pantheon+SH0ES only: H0 = 73.53, Omega_m = 0.332")
print("CC+BAO only        : H0 = 71.11, Omega_m = 0.295")
print()
print(
    "The all-data result should lie where the full joint likelihood "
    "places it; it is not forced to either reference value."
)
