#!/usr/bin/env python3
"""Convert the validated BAO-only (Omega_m, r_s h) chain to a fixed-r_s H0 fit.

This reproduces the calibration logic of the older analysis:

    r_s = 147.4 Mpc

With r_s fixed, H0 is no longer degenerate with the ruler scale:

    H0 = 100 (r_s h) / r_s.

No new MCMC is required because this is an exact one-to-one transformation
of the Stage-03 posterior.
"""

from pathlib import Path
import json

import emcee
import numpy as np
from getdist import MCSamples, plots

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "bao_flat_lcdm"
OUT = ROOT / "results" / "bao_flat_lcdm_fixed_rs"
OUT.mkdir(parents=True, exist_ok=True)

RS_FIXED_MPC = 147.4

summary0 = json.loads((SRC / "summary.json").read_text())

backend = emcee.backends.HDFBackend(
    str(SRC / "chain.h5"),
    read_only=True,
)

burn = int(summary0["mcmc"]["burn_in"])
raw = backend.get_chain(discard=burn, flat=True)

omega_m = raw[:, 0]
rs_h = raw[:, 1]
H0 = 100.0 * rs_h / RS_FIXED_MPC

samples = np.column_stack([H0, omega_m])


def summarize(x):
    q = np.percentile(x, [2.5, 16, 50, 84, 97.5])
    return {
        "q2.5": float(q[0]),
        "q16": float(q[1]),
        "median": float(q[2]),
        "q84": float(q[3]),
        "q97.5": float(q[4]),
        "minus_68": float(q[2] - q[1]),
        "plus_68": float(q[3] - q[2]),
        "minus_95": float(q[2] - q[0]),
        "plus_95": float(q[4] - q[2]),
    }


posterior = {
    "H0_km_s_Mpc": summarize(H0),
    "Omega_m": summarize(omega_m),
}

best0 = summary0["best_fit"]
best_H0 = 100.0 * float(best0["r_s_h_Mpc"]) / RS_FIXED_MPC

summary = {
    "dataset": "BAO11_Ratra2021",
    "model": "flat_LCDM_fixed_rs",
    "calibration": {
        "r_s_fixed_Mpc": RS_FIXED_MPC,
        "interpretation": (
            "H0 is conditional on the fixed sound-horizon calibration. "
            "This reproduces the logic of the older BAO analysis."
        ),
    },
    "best_fit": {
        "H0_km_s_Mpc": best_H0,
        "Omega_m": float(best0["Omega_m"]),
        "chi2_min": float(best0["chi2_min"]),
    },
    "posterior": posterior,
    "mcmc_source": {
        "source_chain": "results/bao_flat_lcdm/chain.h5",
        "transformation_only": True,
        "post_burn_samples": int(len(samples)),
    },
    "fit_quality": summary0["fit_quality"],
}

(OUT / "summary.json").write_text(json.dumps(summary, indent=2))

# Triangle plot
gd = MCSamples(
    samples=samples,
    names=["H0", "Omega_m"],
    labels=[r"H_0", r"\Omega_m"],
)

g = plots.get_subplot_plotter(width_inch=6.2)
g.triangle_plot([gd], filled=True)

H = posterior["H0_km_s_Mpc"]
Om = posterior["Omega_m"]

g.subplots[0, 0].set_title(
    rf"$H_0 = {H['median']:.2f}"
    rf"^{{+{H['plus_68']:.2f}}}"
    rf"_{{-{H['minus_68']:.2f}}}$",
    fontsize=13,
    pad=10,
)

g.subplots[1, 1].set_title(
    rf"$\Omega_m = {Om['median']:.3f}"
    rf"^{{+{Om['plus_68']:.3f}}}"
    rf"_{{-{Om['minus_68']:.3f}}}$",
    fontsize=13,
    pad=10,
)

g.export(str(OUT / "bao_fixed_rs_triangle.pdf"))
g.export(str(OUT / "bao_fixed_rs_triangle.png"))

print("Fixed-r_s BAO result")
print("====================")
print(f"r_s fixed = {RS_FIXED_MPC:.1f} Mpc")
print(
    "H0 = "
    f"{H['median']:.3f} +{H['plus_68']:.3f} "
    f"-{H['minus_68']:.3f} km/s/Mpc"
)
print(
    "Omega_m = "
    f"{Om['median']:.5f} +{Om['plus_68']:.5f} "
    f"-{Om['minus_68']:.5f}"
)
print()
print("Best-fit H0 =", best_H0)
print("Saved to:", OUT)
