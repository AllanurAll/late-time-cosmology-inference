#!/usr/bin/env python3
"""Compare the correlated CC fit with a diagonal-only sensitivity check."""

from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor
from scipy.optimize import minimize

from late_time_cosmology.background import FlatLCDMBackground
from late_time_cosmology.cc import CCLikelihood

ROOT = Path(__file__).resolve().parents[1]
like = CCLikelihood(ROOT / "data" / "cc")

def fit(cov):
    inv = np.linalg.inv(cov)

    def objective(theta):
        H0, om = theta
        if not (40 <= H0 <= 100 and 0.05 <= om <= 0.60):
            return np.inf
        model = FlatLCDMBackground(H0=H0, omega_m=om)
        d = like.Hobs - model.H(like.z)
        return float(d @ inv @ d)

    res = minimize(
        objective,
        [70.0, 0.30],
        method="L-BFGS-B",
        bounds=[(40,100),(0.05,0.60)],
    )
    return res

full = fit(like.cov)

# Published tabulated uncertainties treated as independent.
diag = np.diag(like.df["sigma_diag"].to_numpy(float)**2)
diagonal = fit(diag)

print("Correlated-covariance best fit:")
print(f"  H0      = {full.x[0]:.6f}")
print(f"  Omega_m = {full.x[1]:.6f}")
print(f"  chi2    = {full.fun:.6f}")

print()
print("Diagonal-only sensitivity check:")
print(f"  H0      = {diagonal.x[0]:.6f}")
print(f"  Omega_m = {diagonal.x[1]:.6f}")
print(f"  chi2    = {diagonal.fun:.6f}")

print()
print("Shift caused by covariance treatment:")
print(f"  Delta H0      = {full.x[0]-diagonal.x[0]:+.6f}")
print(f"  Delta Omega_m = {full.x[1]-diagonal.x[1]:+.6f}")
