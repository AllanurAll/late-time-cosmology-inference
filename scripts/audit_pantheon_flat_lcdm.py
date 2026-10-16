#!/usr/bin/env python3
"""Audit the official Pantheon+ files and both cosmology-only selections."""

from pathlib import Path

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

if not DATA.exists() or not COV.exists():
    raise FileNotFoundError(
        "Official Pantheon+ files not found. Run "
        "`python scripts/fetch_pantheon_plus.py` first."
    )

for selection in [
    "official",
    "no_calibrators",
]:
    print()
    print("=" * 70)
    print(selection)
    print("=" * 70)

    like = PantheonFlatLCDM(
        DATA,
        COV,
        selection=selection,
    )

    print("N =", like.n)
    print(
        "zHD range =",
        like.z_hd.min(),
        "to",
        like.z_hd.max(),
    )
    print(
        "covariance shape =",
        like.cov.shape,
    )
    print(
        "max raw |C-C^T| =",
        like.max_raw_asymmetry,
    )

    fit = like.posterior_grid(
        n_grid=301,
        half_width=0.12,
    )

    print(
        "best Omega_m =",
        fit["best_fit_omega_m"],
    )
    print(
        "chi2_min =",
        fit["chi2_min"],
    )

    q = fit["quantiles"]

    print(
        "posterior Omega_m = "
        f"{q['median']:.5f} "
        f"+{q['plus_68']:.5f} "
        f"-{q['minus_68']:.5f}"
    )

    if selection == "official":
        print(
            "Benchmark: Brout et al. (2022) report "
            "Omega_m = 0.334 +/- 0.018 for Pantheon+ SNe alone."
        )
    else:
        print(
            "Benchmark: independent analyses of this 1580-object "
            "selection report chi2_min = 1387.10."
        )
