#!/usr/bin/env python3
from pathlib import Path
import json

import numpy as np
from getdist import MCSamples, plots
import matplotlib.pyplot as plt

from late_time_cosmology.pantheon_shoes import (
    PantheonSH0ESFlatLCDM,
    summarize_grid,
    draw_grid_samples,
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

OUT = (
    ROOT
    / "results"
    / "pantheon_shoes_flat_lcdm"
)
OUT.mkdir(
    parents=True,
    exist_ok=True,
)

like = PantheonSH0ESFlatLCDM(
    DATA,
    COV,
)

print("Pantheon+ + SH0ES flat LambdaCDM")
print("================================")
print("Selected rows      :", like.n)
print("Cepheid calibrators:", like.n_calibrators)
print("Hubble-flow rows   :", like.n_hubble_flow)
print()

opt = like.optimize()

H0_best = float(opt.x[0])
Om_best = float(opt.x[1])
MB_best = like.best_MB(
    H0_best,
    Om_best,
)

print("Best fit")
print("--------")
print(f"H0       = {H0_best:.8f} km/s/Mpc")
print(f"Omega_m  = {Om_best:.8f}")
print(f"M_B      = {MB_best:.8f}")
print(f"chi2_min = {float(opt.fun):.8f}")
print()

# The published posterior lies comfortably inside these ranges.
grid = like.posterior_grid(
    n_omega=321,
    n_h0=321,
    omega_range=(0.25, 0.42),
    h0_range=(68.0, 79.0),
)

summary = summarize_grid(
    grid
)

H = summary["H0_km_s_Mpc"]
Om = summary["Omega_m"]

result = {
    "dataset": "Pantheon+_plus_SH0ES",
    "model": "flat_LCDM",
    "likelihood_definition": {
        "selection": "(zHD > 0.01) OR IS_CALIBRATOR",
        "observed_vector": "m_b_corr",
        "calibrator_theory": "CEPH_DIST + M_B",
        "hubble_flow_theory": "mu_LCDM(H0,Omega_m) + M_B",
        "covariance": "Pantheon+SH0ES_STAT+SYS.cov",
        "M_B_treatment": "analytic marginalization",
    },
    "best_fit": {
        "H0_km_s_Mpc": H0_best,
        "Omega_m": Om_best,
        "M_B": MB_best,
        "chi2_min": float(opt.fun),
    },
    "posterior": summary,
    "published_benchmark": {
        "H0_km_s_Mpc": {
            "central": 73.6,
            "sigma_68": 1.1,
        },
        "Omega_m": {
            "central": 0.334,
            "sigma_68": 0.018,
        },
        "source": "Brout et al. 2022, Table 3, Pantheon+ & SH0ES FlatLCDM",
    },
}

(
    OUT
    / "summary.json"
).write_text(
    json.dumps(
        result,
        indent=2,
    )
)

print("Posterior")
print("---------")
print(
    "H0 = "
    f"{H['median']:.3f} "
    f"+{H['plus_68']:.3f} "
    f"-{H['minus_68']:.3f} km/s/Mpc"
)
print(
    "Omega_m = "
    f"{Om['median']:.5f} "
    f"+{Om['plus_68']:.5f} "
    f"-{Om['minus_68']:.5f}"
)

# ------------------------------------------------------------
# GetDist samples from deterministic posterior grid.
# ------------------------------------------------------------
samples = draw_grid_samples(
    grid,
    n_samples=300_000,
)

gd = MCSamples(
    samples=samples,
    names=[
        "H0",
        "Omega_m",
    ],
    labels=[
        r"H_0",
        r"\Omega_m",
    ],
)

g = plots.get_subplot_plotter(
    width_inch=6.4
)

g.triangle_plot(
    [gd],
    filled=True,
)

g.subplots[0, 0].set_title(
    rf"$H_0 = "
    rf"{H['median']:.2f}"
    rf"^{{+{H['plus_68']:.2f}}}"
    rf"_{{-{H['minus_68']:.2f}}}$",
    fontsize=13,
    pad=10,
)

g.subplots[1, 1].set_title(
    rf"$\Omega_m = "
    rf"{Om['median']:.3f}"
    rf"^{{+{Om['plus_68']:.3f}}}"
    rf"_{{-{Om['minus_68']:.3f}}}$",
    fontsize=13,
    pad=10,
)

g.export(
    str(
        OUT
        / "pantheon_shoes_H0_Omega_m_triangle.pdf"
    )
)

g.export(
    str(
        OUT
        / "pantheon_shoes_H0_Omega_m_triangle.png"
    )
)

# ------------------------------------------------------------
# 2D posterior image: useful as a direct grid diagnostic.
# ------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=(6.6, 5.2)
)

mesh = ax.contourf(
    grid["h0_grid"],
    grid["omega_grid"],
    grid["posterior"],
    levels=40,
)

ax.plot(
    H0_best,
    Om_best,
    marker="x",
    ms=8,
    label="best fit",
)

ax.set_xlabel(
    r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]"
)
ax.set_ylabel(
    r"$\Omega_m$"
)
ax.set_title(
    "Pantheon+ + SH0ES: flat $\\Lambda$CDM"
)
ax.legend()

fig.tight_layout()
fig.savefig(
    OUT / "pantheon_shoes_posterior_grid.png",
    dpi=250,
)
plt.close(fig)

print()
print("Published benchmark")
print("-------------------")
print("H0      = 73.6 +/- 1.1 km/s/Mpc")
print("Omega_m = 0.334 +/- 0.018")
print()
print("Saved to:", OUT)
