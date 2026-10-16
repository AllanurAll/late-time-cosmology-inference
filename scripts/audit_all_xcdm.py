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
    derived,
)
from late_time_cosmology.pantheon_shoes_xcdm import (
    PantheonSH0ESFlatXCDM,
    PantheonSH0ESXCDMProfileCache,
)
from late_time_cosmology.all_xcdm_fit import (
    optimize_all_xcdm,
    chi2,
    chi2_components,
)

ROOT = Path(__file__).resolve().parents[1]

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

audit_points = [
    (73.5, 0.332, -0.90),
    (72.5, 0.310, -0.80),
    (73.0, 0.300, -1.00),
    (71.5, 0.290, -0.70),
    (74.0, 0.350, -1.15),
]

err = sn_fast.max_abs_error(
    audit_points
)

print()
print(
    "Max |chi2_fast - chi2_exact| on audit points =",
    err,
)

if err > 5e-4:
    raise RuntimeError(
        "Pantheon+SH0ES XCDM spline cache is not accurate enough."
    )

# ------------------------------------------------------------
# Nested LambdaCDM identity check against the already validated
# full all-data LambdaCDM minimum.
# ------------------------------------------------------------
lcdm_path = (
    ROOT
    / "results"
    / "all_flat_lcdm"
    / "summary.json"
)

if lcdm_path.exists():
    lcdm = json.loads(
        lcdm_path.read_text()
    )

    b = lcdm[
        "best_fit"
    ]

    theta_lcdm = [
        b["h"],
        b["omega_b_h2"],
        b["omega_c_h2"],
        -1.0,
    ]

    x_chi2 = chi2(
        theta_lcdm,
        cc,
        bao,
        sn_fast,
    )

    l_chi2 = float(
        b[
            "chi2_min"
        ]
    )

    print()
    print(
        "Nested w_X=-1 consistency check"
    )
    print(
        "XCDM implementation chi2 =",
        x_chi2,
    )
    print(
        "validated LambdaCDM chi2 =",
        l_chi2,
    )
    print(
        "difference              =",
        x_chi2 - l_chi2,
    )

    if abs(
        x_chi2
        - l_chi2
    ) > 2e-3:
        raise RuntimeError(
            "Full XCDM likelihood does not recover the validated "
            "LambdaCDM likelihood closely enough at w_X=-1."
        )

print()
print(
    "Optimizing full Pantheon+SH0ES + CC + BAO XCDM likelihood..."
)

opt = optimize_all_xcdm(
    cc,
    bao,
    sn_fast,
)

d = derived(
    opt.x
)

parts = chi2_components(
    opt.x,
    cc,
    bao,
    sn_fast,
)

MB = sn_exact.best_MB_xcdm(
    d[
        "H0_km_s_Mpc"
    ],
    d[
        "Omega_m"
    ],
    d[
        "w_X"
    ],
)

print()
print("Best fit")
print("--------")
print(f"h            = {d['h']:.9f}")
print(f"H0           = {d['H0_km_s_Mpc']:.9f} km/s/Mpc")
print(f"Omega_b h^2  = {d['omega_b_h2']:.9f}")
print(f"Omega_c h^2  = {d['omega_c_h2']:.9f}")
print(f"Omega_m      = {d['Omega_m']:.9f}")
print(f"w_X          = {d['w_X']:.9f}")
print(f"r_s          = {d['r_s_Mpc']:.9f} Mpc")
print(f"r_s h        = {d['r_s_h_Mpc']:.9f} Mpc")
print(f"M_B          = {MB:.9f}")
print(f"chi2_min     = {sum(parts.values()):.9f}")

print()
print("chi2 components")
print("---------------")
for key, value in parts.items():
    print(f"{key:22s}: {value:.9f}")
print(
    f"{'total':22s}: "
    f"{sum(parts.values()):.9f}"
)
