#!/usr/bin/env python3
from pathlib import Path
import json

from late_time_cosmology.cc import CCLikelihood
from late_time_cosmology.bao_reduced import BAOReducedFlatLCDM
from late_time_cosmology.cc_bao_fit import (
    run_chain,
    write_summary,
    plot_main_triangle,
    plot_sampled_triangle,
    plot_cc_best_fit,
)

ROOT = Path(__file__).resolve().parents[1]

CC_DATA = ROOT / "data" / "cc"
BAO_DATA = ROOT / "data" / "bao"

RESULTS = ROOT / "results" / "cc_bao_flat_lcdm"
RESULTS.mkdir(
    parents=True,
    exist_ok=True,
)

cc = CCLikelihood(CC_DATA)
bao = BAOReducedFlatLCDM(BAO_DATA)

print("Joint CC + BAO flat LambdaCDM")
print("================================")
print("CC data values :", len(cc.z))
print("BAO data values:", bao.n_data)
print("Total           :", len(cc.z) + bao.n_data)
print()
print(
    "Sampled parameters: "
    "h, Omega_b h^2, Omega_c h^2"
)

sampler, opt = run_chain(
    cc,
    bao,
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
    sampler,
    opt,
)

plot_main_triangle(
    RESULTS / "cc_bao_main_triangle.pdf",
    RESULTS / "cc_bao_main_triangle.png",
    arr,
    summary,
)

plot_sampled_triangle(
    RESULTS / "cc_bao_sampled_triangle.pdf",
    RESULTS / "cc_bao_sampled_triangle.png",
    samples,
    summary,
)

plot_cc_best_fit(
    RESULTS / "cc_bao_best_fit_Hz.png",
    cc,
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
print("Literature benchmark")
print(
    json.dumps(
        summary["literature_benchmark"],
        indent=2,
    )
)

print()
print("Saved to:", RESULTS)
