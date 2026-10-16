#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from late_time_cosmology.cc import load_cc32, build_cc_covariance

base = Path("data/cc")
df, sys = load_cc32(base)
cov = build_cc_covariance(df, sys)

print("CC points:", len(df))
print("Correlated Moresco points:", int(df["moresco_correlated"].sum()))
print("Covariance shape:", cov.shape)
print("Symmetry max |C-C^T|:", np.max(np.abs(cov-cov.T)))
print("Minimum eigenvalue:", np.linalg.eigvalsh(cov).min())
print("Condition number:", np.linalg.cond(cov))

print("\nPublished-value spot checks:")
for z in [0.1791, 0.3519, 0.3802, 0.7812, 1.037, 1.965]:
    row = df[np.isclose(df["z"], z)].iloc[0]
    print(f"z={row.z:6.4f}  H={row.H:6.1f}  sigma={row.sigma_diag:5.1f}")
