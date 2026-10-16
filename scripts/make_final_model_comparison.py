#!/usr/bin/env python3
from pathlib import Path
import json

import emcee
import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, plots
from scipy.stats import gaussian_kde

from late_time_cosmology.model_comparison import (
    build_comparison,
    load_summary,
    write_csv,
    write_json,
    write_markdown,
)

ROOT = Path(__file__).resolve().parents[1]

LCDM_DIR = (
    ROOT
    / "results"
    / "all_flat_lcdm"
)

XCDM_DIR = (
    ROOT
    / "results"
    / "all_flat_xcdm"
)

OUT = (
    ROOT
    / "results"
    / "model_comparison"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

lcdm = load_summary(
    LCDM_DIR / "summary.json"
)
xcdm = load_summary(
    XCDM_DIR / "summary.json"
)

comparison = build_comparison(
    lcdm,
    xcdm,
)

write_json(
    OUT / "flat_lcdm_xcdm_comparison.json",
    comparison,
)

write_csv(
    OUT / "flat_lcdm_xcdm_comparison.csv",
    comparison,
)

write_markdown(
    OUT / "flat_lcdm_xcdm_comparison.md",
    comparison,
)

# ------------------------------------------------------------
# Load thinned converged samples.
# Only H0, Omega_m and w_X are needed for the comparison figures.
# ------------------------------------------------------------
def load_chain(
    result_dir,
    summary,
    thin=10,
):
    backend = emcee.backends.HDFBackend(
        str(
            result_dir
            / "chain.h5"
        ),
        read_only=True,
    )

    burn = int(
        summary[
            "mcmc"
        ][
            "burn_in"
        ]
    )

    return backend.get_chain(
        discard=burn,
        thin=thin,
        flat=True,
    )


lcdm_chain = load_chain(
    LCDM_DIR,
    lcdm,
    thin=10,
)

xcdm_chain = load_chain(
    XCDM_DIR,
    xcdm,
    thin=10,
)

# The project samples h, Omega_b h^2 and Omega_c h^2.
# Omega_m includes the same fixed radiation/neutrino convention used by
# Cosmology, but at z=0 the tiny radiation correction is negligible for
# visualization. For exact summary numbers the JSON products remain primary.
#
# To preserve the package's exact convention in the overlay, compute a small
# thinned derived set through the existing helper functions.
from late_time_cosmology.all_flat_fit import (
    derived_arrays as derived_lcdm,
)
from late_time_cosmology.all_xcdm_fit import (
    derived_arrays as derived_xcdm,
)

lcdm_arr = derived_lcdm(
    lcdm_chain
)
xcdm_arr = derived_xcdm(
    xcdm_chain
)

# ------------------------------------------------------------
# 1. H0-Omega_m posterior overlay
# ------------------------------------------------------------
gd_l = MCSamples(
    samples=np.column_stack([
        lcdm_arr[
            "H0_km_s_Mpc"
        ],
        lcdm_arr[
            "Omega_m"
        ],
    ]),
    names=[
        "H0",
        "Omega_m",
    ],
    labels=[
        r"H_0",
        r"\Omega_m",
    ],
    label=r"flat \Lambda CDM",
)

gd_x = MCSamples(
    samples=np.column_stack([
        xcdm_arr[
            "H0_km_s_Mpc"
        ],
        xcdm_arr[
            "Omega_m"
        ],
    ]),
    names=[
        "H0",
        "Omega_m",
    ],
    labels=[
        r"H_0",
        r"\Omega_m",
    ],
    label="flat XCDM",
)

g = plots.get_subplot_plotter(
    width_inch=6.4
)

g.plot_2d(
    [
        gd_l,
        gd_x,
    ],
    "H0",
    "Omega_m",
    filled=True,
)

g.add_legend([
    r"flat $\Lambda$CDM",
    "flat XCDM",
])

g.export(
    str(
        OUT
        / "all_data_H0_Omega_m_overlay.pdf"
    )
)
g.export(
    str(
        OUT
        / "all_data_H0_Omega_m_overlay.png"
    )
)

# ------------------------------------------------------------
# 2. H0 forest comparison
# ------------------------------------------------------------
labels = [
    r"flat $\Lambda$CDM",
    "flat XCDM",
]

centers = np.array([
    lcdm[
        "posterior"
    ][
        "H0_km_s_Mpc"
    ][
        "median"
    ],
    xcdm[
        "posterior"
    ][
        "H0_km_s_Mpc"
    ][
        "median"
    ],
])

lower = np.array([
    lcdm[
        "posterior"
    ][
        "H0_km_s_Mpc"
    ][
        "minus_68"
    ],
    xcdm[
        "posterior"
    ][
        "H0_km_s_Mpc"
    ][
        "minus_68"
    ],
])

upper = np.array([
    lcdm[
        "posterior"
    ][
        "H0_km_s_Mpc"
    ][
        "plus_68"
    ],
    xcdm[
        "posterior"
    ][
        "H0_km_s_Mpc"
    ][
        "plus_68"
    ],
])

y = np.arange(
    len(labels)
)

fig, ax = plt.subplots(
    figsize=(7.0, 3.6)
)

ax.errorbar(
    centers,
    y,
    xerr=np.vstack([
        lower,
        upper,
    ]),
    fmt="o",
    capsize=4,
)

ax.set_yticks(
    y,
    labels,
)
ax.invert_yaxis()
ax.set_xlabel(
    r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]"
)
ax.set_title(
    "Pantheon+SH0ES + CC + BAO: 68% intervals"
)

fig.tight_layout()
fig.savefig(
    OUT
    / "all_data_H0_model_comparison.png",
    dpi=250,
)
plt.close(fig)

# ------------------------------------------------------------
# 3. Omega_m forest comparison
# ------------------------------------------------------------
centers = np.array([
    lcdm[
        "posterior"
    ][
        "Omega_m"
    ][
        "median"
    ],
    xcdm[
        "posterior"
    ][
        "Omega_m"
    ][
        "median"
    ],
])

lower = np.array([
    lcdm[
        "posterior"
    ][
        "Omega_m"
    ][
        "minus_68"
    ],
    xcdm[
        "posterior"
    ][
        "Omega_m"
    ][
        "minus_68"
    ],
])

upper = np.array([
    lcdm[
        "posterior"
    ][
        "Omega_m"
    ][
        "plus_68"
    ],
    xcdm[
        "posterior"
    ][
        "Omega_m"
    ][
        "plus_68"
    ],
])

fig, ax = plt.subplots(
    figsize=(7.0, 3.6)
)

ax.errorbar(
    centers,
    y,
    xerr=np.vstack([
        lower,
        upper,
    ]),
    fmt="o",
    capsize=4,
)

ax.set_yticks(
    y,
    labels,
)
ax.invert_yaxis()
ax.set_xlabel(
    r"$\Omega_m$"
)
ax.set_title(
    "Pantheon+SH0ES + CC + BAO: 68% intervals"
)

fig.tight_layout()
fig.savefig(
    OUT
    / "all_data_Omega_m_model_comparison.png",
    dpi=250,
)
plt.close(fig)

# ------------------------------------------------------------
# 4. XCDM w_X posterior with LambdaCDM reference
# ------------------------------------------------------------
wx = np.asarray(
    xcdm_arr[
        "w_X"
    ],
    dtype=float,
)

kde = gaussian_kde(
    wx
)

xgrid = np.linspace(
    np.percentile(
        wx,
        0.1,
    ),
    np.percentile(
        wx,
        99.9,
    ),
    600,
)

fig, ax = plt.subplots(
    figsize=(7.0, 4.3)
)

ax.plot(
    xgrid,
    kde(
        xgrid
    ),
)

ax.axvline(
    -1.0,
    linestyle="--",
    linewidth=1.2,
    label=r"$\Lambda$CDM: $w_X=-1$",
)

q = xcdm[
    "posterior"
][
    "w_X"
]

ax.axvspan(
    q["q16"],
    q["q84"],
    alpha=0.20,
    label="68% credible interval",
)

ax.axvspan(
    q["q2.5"],
    q["q97.5"],
    alpha=0.10,
    label="95% credible interval",
)

ax.set_xlabel(
    r"$w_X$"
)
ax.set_ylabel(
    "posterior density"
)
ax.set_title(
    "Flat XCDM: marginalized dark-energy equation of state"
)
ax.legend()

fig.tight_layout()
fig.savefig(
    OUT
    / "all_data_wX_posterior.png",
    dpi=250,
)
plt.close(fig)

# ------------------------------------------------------------
# 5. Information-criterion differences
# ------------------------------------------------------------
d = comparison[
    "delta_XCDM_minus_LambdaCDM"
]

fig, ax = plt.subplots(
    figsize=(6.4, 4.2)
)

names = [
    r"$\Delta$AIC",
    r"$\Delta$BIC",
]

values = [
    d[
        "delta_AIC"
    ],
    d[
        "delta_BIC"
    ],
]

ax.bar(
    names,
    values,
)

ax.axhline(
    0.0,
    linewidth=1.0,
)

ax.set_ylabel(
    "XCDM - LambdaCDM"
)
ax.set_title(
    "Information-criterion comparison"
)

fig.tight_layout()
fig.savefig(
    OUT
    / "all_data_information_criteria.png",
    dpi=250,
)
plt.close(fig)

print("Final model-comparison products created.")
print()
print(
    json.dumps(
        comparison,
        indent=2,
    )
)
print()
print("Saved to:", OUT)
