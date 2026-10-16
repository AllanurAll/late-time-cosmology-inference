#!/usr/bin/env python3
from pathlib import Path
import json

from late_time_cosmology.cc import CCLikelihood
from late_time_cosmology.bao_xcdm import BAOFlatXCDM
from late_time_cosmology.cc_bao_xcdm_fit import (
    CCFlatXCDM,
    run_chain,
    write_summary,
    plot_main_triangle,
    plot_ruler_triangle,
)

ROOT = Path(__file__).resolve().parents[1]

RESULTS = (
    ROOT
    / "results"
    / "cc_bao_flat_xcdm"
)

RESULTS.mkdir(
    parents=True,
    exist_ok=True,
)

cc_base = CCLikelihood(
    ROOT
    / "data"
    / "cc"
)

cc = CCFlatXCDM(
    cc_base
)

bao = BAOFlatXCDM(
    ROOT
    / "data"
    / "bao"
)

print("CC + BAO: flat XCDM")
print("=====================")
print("CC values :", cc.n_data)
print("BAO values:", bao.n_data)
print("Total     :", cc.n_data + bao.n_data)
print()
print(
    "Sampled parameters: "
    "h, Omega_b h^2, Omega_c h^2, w_X"
)
print(
    "LambdaCDM is recovered exactly at w_X = -1."
)

sampler, opt = run_chain(
    cc,
    bao,
    RESULTS / "chain.h5",
    nwalkers=64,
    nsteps=20000,
    reset=True,
    progress=True,
)

summary, samples, arr = write_summary(
    RESULTS / "summary.json",
    cc,
    bao,
    sampler,
    opt,
    project_root=ROOT,
)

plot_main_triangle(
    RESULTS
    / "cc_bao_xcdm_main_triangle.pdf",
    RESULTS
    / "cc_bao_xcdm_main_triangle.png",
    arr,
    summary,
)

plot_ruler_triangle(
    RESULTS
    / "cc_bao_xcdm_ruler_triangle.pdf",
    RESULTS
    / "cc_bao_xcdm_ruler_triangle.png",
    arr,
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
print("LambdaCDM comparison")
print(
    json.dumps(
        summary["model_comparison"],
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
