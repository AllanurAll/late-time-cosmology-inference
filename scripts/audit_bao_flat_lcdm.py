#!/usr/bin/env python3
"""Deterministic BAO likelihood audit before MCMC."""

from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from late_time_cosmology.bao_reduced import BAOReducedFlatLCDM

ROOT = Path(__file__).resolve().parents[1]
like = BAOReducedFlatLCDM(ROOT / "data" / "bao")

res = minimize(
    lambda x: like.chi2(x[0], x[1]),
    x0=np.array([0.30, 101.0]),
    method="L-BFGS-B",
    bounds=[(0.05, 0.60), (50.0, 150.0)],
)

print("success:", res.success)
print("message:", res.message)
print(f"Omega_m best fit = {res.x[0]:.9f}")
print(f"r_s h best fit   = {res.x[1]:.9f} Mpc")
print(f"H0 r_s best fit  = {100*res.x[1]:.9f} km/s")
print(f"chi2_min         = {res.fun:.9f}")

parts = like.chi2_components(res.x[0], res.x[1])
print()
print("chi2 components:")
for k, v in parts.items():
    print(f"  {k:12s}: {v:.9f}")
print("  total       :", sum(parts.values()))
