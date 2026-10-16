#!/usr/bin/env python3
from pathlib import Path
import json

from late_time_cosmology.cc import (
    CCLikelihood,
)
from late_time_cosmology.bao_xcdm import (
    BAOFlatXCDM,
)
from late_time_cosmology.cc_bao_xcdm_fit import (
    CCFlatXCDM,
)
from late_time_cosmology.pantheon_shoes_xcdm import (
    PantheonSH0ESFlatXCDM,
    PantheonSH0ESXCDMProfileCache,
)
from late_time_cosmology.all_xcdm_fit import (
    run_chain,
    write_summary,
    plot_main_triangle,
    plot_ruler_triangle,
    plot_model_comparison,
)

ROOT = Path(__file__).resolve().parents[1]

RESULTS = (
    ROOT
    / "results"
    / "all_flat_xcdm"
)

RESULTS.mkdir(
    parents=True,
    exist_ok=True,
)

cc = CCFlatXCDM(
    CCLikelihood(
        ROOT
        / "data"
        / "cc"
    )
)

bao = BAOFlatXCDM(
    ROOT
    / "data"
    / "bao"
)

sn_exact = PantheonSH0ESFlatXCDM(
    ROOT
    / "data"
    / "pantheon_plus"
    / "Pantheon+SH0ES.dat",
    ROOT
    / "data"
    / "pantheon_plus"
    / "Pantheon+SH0ES_STAT+SYS.cov",
)

cache_path = (
    ROOT
    / "cache"
    / "pantheon_shoes_xcdm_profile.npz"
)

print(
    "Pantheon+SH0ES + CC + BAO: flat XCDM"
)
print("=" * 49)
print(
    "Pantheon+SH0ES rows:",
    sn_exact.n,
)
print(
    "CC values           :",
    cc.n_data,
)
print(
    "BAO values          :",
    bao.n_data,
)
print(
    "Total observations  :",
    (
        sn_exact.n
        + cc.n_data
        + bao.n_data
    ),
)
print()

print(
    "Loading/building Pantheon+SH0ES XCDM profile cache..."
)

sn_fast = (
    PantheonSH0ESXCDMProfileCache
    .load_or_build(
        sn_exact,
        cache_path,
        omega_min=0.16,
        omega_max=0.48,
        n_omega=101,
        w_min=-1.80,
        w_max=-0.25,
        n_w=121,
        progress=True,
    )
)

err = sn_fast.max_abs_error([
    (73.5, 0.332, -0.90),
    (72.5, 0.310, -0.80),
    (73.0, 0.300, -1.00),
    (71.5, 0.290, -0.70),
])

print(
    "Pantheon XCDM cache max audit error:",
    err,
)

if err > 5e-4:
    raise RuntimeError(
        "Pantheon XCDM cache accuracy check failed."
    )

sampler, opt = run_chain(
    cc,
    bao,
    sn_fast,
    RESULTS
    / "chain.h5",
    nwalkers=72,
    nsteps=20000,
    reset=True,
    progress=True,
)

summary, samples, arr = write_summary(
    RESULTS
    / "summary.json",
    cc,
    bao,
    sn_exact,
    sn_fast,
    sampler,
    opt,
    project_root=ROOT,
)

plot_main_triangle(
    RESULTS
    / "all_xcdm_main_triangle.pdf",
    RESULTS
    / "all_xcdm_main_triangle.png",
    arr,
    summary,
)

plot_ruler_triangle(
    RESULTS
    / "all_xcdm_ruler_triangle.pdf",
    RESULTS
    / "all_xcdm_ruler_triangle.png",
    arr,
    summary,
)

plot_model_comparison(
    RESULTS
    / "all_xcdm_H0_model_comparison.png",
    summary,
    ROOT,
)

print()
print("Best fit")
print(
    json.dumps(
        summary[
            "best_fit"
        ],
        indent=2,
    )
)

print()
print("Posterior")
print(
    json.dumps(
        summary[
            "posterior"
        ],
        indent=2,
    )
)

print()
print("MCMC diagnostics")
print(
    json.dumps(
        summary[
            "mcmc"
        ],
        indent=2,
    )
)

print()
print("Fit quality")
print(
    json.dumps(
        summary[
            "fit_quality"
        ],
        indent=2,
    )
)

print()
print("LambdaCDM comparison")
print(
    json.dumps(
        summary[
            "model_comparison"
        ],
        indent=2,
    )
)

print()
print("Pantheon XCDM cache")
print(
    json.dumps(
        summary[
            "pantheon_xcdm_cache"
        ],
        indent=2,
    )
)

print()
print(
    "Saved to:",
    RESULTS,
)
