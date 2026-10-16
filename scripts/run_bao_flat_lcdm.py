#!/usr/bin/env python3
from pathlib import Path
import json

from late_time_cosmology.bao_reduced import BAOReducedFlatLCDM
from late_time_cosmology.bao_fit import (
    run_chain,
    write_summary,
    plot_triangle,
    plot_prediction_table,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bao"
RESULTS = ROOT / "results" / "bao_flat_lcdm"
RESULTS.mkdir(parents=True, exist_ok=True)

like = BAOReducedFlatLCDM(DATA)

print("Running BAO-only flat LambdaCDM")
print("Data values:", like.n_data)
print()
print("Important:")
print(
    "BAO alone constrains Omega_m and r_s*h. "
    "It does not separately determine H0 and r_s."
)

sampler, opt = run_chain(
    like,
    RESULTS / "chain.h5",
    nwalkers=48,
    nsteps=12000,
    reset=True,
    progress=True,
)

summary, samples = write_summary(
    RESULTS / "summary.json",
    like,
    sampler,
    opt,
)

plot_triangle(
    RESULTS / "bao_flat_lcdm_triangle.pdf",
    RESULTS / "bao_flat_lcdm_triangle.png",
    samples,
    summary,
)

plot_prediction_table(
    RESULTS / "bao_best_fit_residuals.png",
    like,
    summary,
)

print()
print("Best fit")
print(json.dumps(summary["best_fit"], indent=2))

print()
print("Posterior")
print(json.dumps(summary["posterior"], indent=2))

print()
print("MCMC diagnostics")
print(json.dumps(summary["mcmc"], indent=2))

print()
print("Fit quality")
print(json.dumps(summary["fit_quality"], indent=2))

print()
print("Saved to:", RESULTS)
