#!/usr/bin/env python3
from pathlib import Path
import json

from late_time_cosmology.cc import CCLikelihood
from late_time_cosmology.bao_reduced import BAOReducedFlatLCDM
from late_time_cosmology.pantheon_shoes import PantheonSH0ESFlatLCDM
from late_time_cosmology.all_flat_fit import (
    PantheonSH0ESFastProfile,
    run_chain,
    write_summary,
    plot_main_triangle,
    plot_sampled_triangle,
    plot_h0_comparison,
)

ROOT = Path(__file__).resolve().parents[1]

RESULTS = (
    ROOT
    / "results"
    / "all_flat_lcdm"
)

RESULTS.mkdir(
    parents=True,
    exist_ok=True,
)

cc = CCLikelihood(
    ROOT / "data" / "cc"
)

bao = BAOReducedFlatLCDM(
    ROOT / "data" / "bao"
)

sn = PantheonSH0ESFlatLCDM(
    ROOT
    / "data"
    / "pantheon_plus"
    / "Pantheon+SH0ES.dat",
    ROOT
    / "data"
    / "pantheon_plus"
    / "Pantheon+SH0ES_STAT+SYS.cov",
)

print(
    "Pantheon+SH0ES + CC + BAO: flat LambdaCDM"
)
print("=" * 52)
print(
    "Pantheon+SH0ES rows:",
    sn.n,
)
print(
    "CC values           :",
    len(cc.z),
)
print(
    "BAO values          :",
    bao.n_data,
)
print(
    "Total observations  :",
    sn.n + len(cc.z) + bao.n_data,
)
print()

print(
    "Building fast Pantheon+SH0ES profile cache..."
)

sn_fast = PantheonSH0ESFastProfile(
    sn,
    omega_min=0.20,
    omega_max=0.45,
    n_omega=801,
)

err = sn_fast.max_abs_error([
    (73.5, 0.332),
    (72.0, 0.310),
    (74.5, 0.350),
])

print(
    "Pantheon cache max audit error:",
    err,
)

if err > 1e-4:
    raise RuntimeError(
        "Pantheon+SH0ES cache accuracy check failed."
    )

sampler, opt = run_chain(
    cc,
    bao,
    sn_fast,
    RESULTS / "chain.h5",
    nwalkers=64,
    nsteps=15000,
    reset=True,
    progress=True,
)

summary, samples, arr = write_summary(
    RESULTS / "summary.json",
    cc,
    bao,
    sn,
    sn_fast,
    sampler,
    opt,
)

plot_main_triangle(
    RESULTS / "all_flat_main_triangle.pdf",
    RESULTS / "all_flat_main_triangle.png",
    arr,
    summary,
)

plot_sampled_triangle(
    RESULTS / "all_flat_sampled_triangle.pdf",
    RESULTS / "all_flat_sampled_triangle.png",
    samples,
    summary,
)

plot_h0_comparison(
    RESULTS / "all_flat_H0_comparison.png",
    ROOT,
    summary,
)

print()
print("Best fit")
print(
    json.dumps(
        summary["best_fit"],
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
print(
    "Saved to:",
    RESULTS,
)
