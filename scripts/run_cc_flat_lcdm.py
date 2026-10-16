#!/usr/bin/env python3
from pathlib import Path
import json

from late_time_cosmology.cc import CCLikelihood
from late_time_cosmology.cc_fit import (
    run_chain,
    write_summary,
    plot_cc_fit,
    plot_triangle,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cc"
RESULTS = ROOT / "results" / "cc_flat_lcdm"
RESULTS.mkdir(parents=True, exist_ok=True)

like = CCLikelihood(DATA)

print("Running CC-only flat LambdaCDM")
print("Data points:", len(like.z))
print("Covariance shape:", like.cov.shape)

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

plot_cc_fit(
    RESULTS / "cc_best_fit.png",
    like,
    summary,
)

plot_triangle(
    RESULTS / "cc_flat_lcdm_triangle.pdf",
    samples,
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
