#!/usr/bin/env python3
from pathlib import Path
import json

from late_time_cosmology.bao_reduced import BAOReducedFlatLCDM
from late_time_cosmology.bao_physical_fit import (
    run_chain,
    write_summary,
    plot_derived_triangle,
    plot_sampled_triangle,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bao"
RESULTS = ROOT / "results" / "bao_flat_lcdm_physical"
RESULTS.mkdir(parents=True, exist_ok=True)

like = BAOReducedFlatLCDM(DATA)

print("BAO-only flat LambdaCDM: physical-density parameterization")
print("Sampled parameters: h, Omega_b h^2, Omega_c h^2")
print("Priors follow Cao, Ryan & Ratra (2021).")
print()
print(
    "Important: the BAO-only H0 posterior is prior-conditioned. "
    "The robust combinations are Omega_m and r_s h."
)

sampler, opt = run_chain(
    like,
    RESULTS / "chain.h5",
    nwalkers=64,
    nsteps=20000,
    reset=True,
    progress=True,
)

summary, samples, arr = write_summary(
    RESULTS / "summary.json",
    like,
    sampler,
    opt,
)

plot_derived_triangle(
    RESULTS / "bao_physical_derived_triangle.pdf",
    RESULTS / "bao_physical_derived_triangle.png",
    arr,
    summary,
)

plot_sampled_triangle(
    RESULTS / "bao_physical_sampled_triangle.pdf",
    RESULTS / "bao_physical_sampled_triangle.png",
    samples,
    summary,
)

print()
print("Representative maximum-likelihood point")
print(
    json.dumps(
        summary["representative_best_fit"],
        indent=2,
    )
)

print()
print("Posterior")
print(
    json.dumps(
        summary["posterior"],
        indent=2,
    )
)

print()
print("MCMC diagnostics")
print(
    json.dumps(
        summary["mcmc"],
        indent=2,
    )
)

print()
print("Fit quality")
print(
    json.dumps(
        summary["fit_quality"],
        indent=2,
    )
)

print()
print("Saved to:", RESULTS)
