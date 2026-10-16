#!/usr/bin/env python3
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from late_time_cosmology.pantheon_flat import (
    PantheonFlatLCDM,
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

RESULTS = ROOT / "results" / "pantheon_flat_lcdm"
RESULTS.mkdir(
    parents=True,
    exist_ok=True,
)


def title_from_stats(s):
    return (
        rf"$\Omega_m = "
        rf"{s['median']:.3f}"
        rf"^{{+{s['plus_68']:.3f}}}"
        rf"_{{-{s['minus_68']:.3f}}}$"
    )


def fit_selection(selection):
    like = PantheonFlatLCDM(
        DATA,
        COV,
        selection=selection,
    )

    print()
    print("Selection:", selection)
    print("N =", like.n)
    print(
        "z range =",
        float(like.z_hd.min()),
        "to",
        float(like.z_hd.max()),
    )
    print(
        "max raw covariance asymmetry =",
        like.max_raw_asymmetry,
    )

    fit = like.posterior_grid(
        n_grid=501,
        half_width=0.15,
    )

    q = fit["quantiles"]

    result = {
        "dataset": "Pantheon+",
        "selection": selection,
        "model": "flat_LCDM",
        "important_note": (
            "Pantheon+ is uncalibrated here. The additive SN intercept "
            "(absolute magnitude/H0 combination) is marginalized "
            "analytically, so no Pantheon+-only H0 value is quoted."
        ),
        "n_data": like.n,
        "zHD_min": float(like.z_hd.min()),
        "zHD_max": float(like.z_hd.max()),
        "best_fit": {
            "Omega_m": fit["best_fit_omega_m"],
            "chi2_min": fit["chi2_min"],
            "intercept": fit["best_intercept"],
        },
        "posterior": {
            "Omega_m": q,
        },
        "fit_quality": {
            # Omega_m plus the profiled additive intercept.
            "n_fitted_parameters": 2,
            "dof": int(like.n - 2),
            "reduced_chi2": float(
                fit["chi2_min"] / (like.n - 2)
            ),
        },
    }

    out = RESULTS / selection
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        out
        / "summary.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print(
        "Omega_m = "
        f"{q['median']:.5f} "
        f"+{q['plus_68']:.5f} "
        f"-{q['minus_68']:.5f}"
    )
    print(
        "best-fit Omega_m =",
        fit["best_fit_omega_m"],
    )
    print(
        "chi2_min =",
        fit["chi2_min"],
    )
    print(
        "reduced chi2 =",
        result["fit_quality"]["reduced_chi2"],
    )

    return like, fit, result


official_like, official_fit, official = fit_selection(
    "official"
)

strict_like, strict_fit, strict = fit_selection(
    "no_calibrators"
)

# ------------------------------------------------------------
# Overlay the two one-dimensional posteriors.
# ------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=(7.0, 5.0)
)

ax.plot(
    official_fit["omega_m_grid"],
    official_fit["density"],
    label=(
        "official Pantheon+ selection "
        f"(N={official_like.n})"
    ),
)

ax.plot(
    strict_fit["omega_m_grid"],
    strict_fit["density"],
    label=(
        "calibrators excluded "
        f"(N={strict_like.n})"
    ),
)

ax.set_xlabel(r"$\Omega_m$")
ax.set_ylabel("posterior density")
ax.set_title(
    "Pantheon+ flat $\\Lambda$CDM"
)
ax.legend()

fig.tight_layout()
fig.savefig(
    RESULTS / "pantheon_flat_lcdm_posterior.png",
    dpi=250,
)
fig.savefig(
    RESULTS / "pantheon_flat_lcdm_posterior.pdf",
)
plt.close(fig)

# ------------------------------------------------------------
# Publication-facing official-selection posterior with title.
# ------------------------------------------------------------
q = official["posterior"]["Omega_m"]

fig, ax = plt.subplots(
    figsize=(6.2, 4.7)
)

ax.plot(
    official_fit["omega_m_grid"],
    official_fit["density"],
)

ax.axvline(
    q["median"],
    lw=1.0,
)

ax.set_xlabel(r"$\Omega_m$")
ax.set_ylabel("posterior density")
ax.set_title(
    title_from_stats(q)
)

fig.tight_layout()
fig.savefig(
    RESULTS / "pantheon_official_Omega_m.png",
    dpi=250,
)
fig.savefig(
    RESULTS / "pantheon_official_Omega_m.pdf",
)
plt.close(fig)

print()
print("Literature checks")
print("-----------------")
print(
    "Brout et al. (2022) Pantheon+ SNe-only flat LCDM: "
    "Omega_m = 0.334 +/- 0.018."
)
print(
    "Independent 1580-object no-calibrator cross-check: "
    "chi2_min ~ 1387.10."
)
print()
print("Saved to:", RESULTS)
