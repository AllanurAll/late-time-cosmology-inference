#!/usr/bin/env python3

from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid
from getdist import MCSamples, plots

from late_time_cosmology.pantheon_flat import PantheonFlatLCDM

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

OUT = ROOT / "results" / "pantheon_flat_lcdm"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Official Pantheon+ cosmology-only selection
# ------------------------------------------------------------
like = PantheonFlatLCDM(
    DATA,
    COV,
    selection="official",
)

fit = like.posterior_grid(
    n_grid=501,
    half_width=0.15,
)

omega_grid = fit["omega_m_grid"]
density = fit["density"]

# ------------------------------------------------------------
# Construct normalized CDF for Omega_m
# ------------------------------------------------------------
cdf = cumulative_trapezoid(
    density,
    omega_grid,
    initial=0.0,
)
cdf /= cdf[-1]

# ------------------------------------------------------------
# The conditional intercept distribution is Gaussian:
#
#   M | Omega_m ~ N(M_best, 1/C)
#
# where C = 1^T Cov^-1 1.
# ------------------------------------------------------------
intercept_grid = np.array([
    like.best_intercept(om)
    for om in omega_grid
])

sigma_intercept = 1.0 / np.sqrt(like._C)

# ------------------------------------------------------------
# Draw samples from the exact 2D posterior.
# This is posterior sampling from the analytically known
# conditional distribution, not a new MCMC.
# ------------------------------------------------------------
rng = np.random.default_rng(20261016)

N = 200_000

u = rng.random(N)

omega_samples = np.interp(
    u,
    cdf,
    omega_grid,
)

mean_intercept = np.interp(
    omega_samples,
    omega_grid,
    intercept_grid,
)

intercept_samples = (
    mean_intercept
    + rng.normal(
        scale=sigma_intercept,
        size=N,
    )
)

samples = np.column_stack([
    omega_samples,
    intercept_samples,
])

# ------------------------------------------------------------
# 68% marginalized credible intervals
# ------------------------------------------------------------
def summarize(x):
    q16, q50, q84 = np.percentile(
        x,
        [16, 50, 84],
    )

    return (
        q50,
        q50 - q16,
        q84 - q50,
    )


om50, om_minus, om_plus = summarize(
    omega_samples
)

m50, m_minus, m_plus = summarize(
    intercept_samples
)

# ------------------------------------------------------------
# GetDist triangle plot
# ------------------------------------------------------------
gd = MCSamples(
    samples=samples,
    names=[
        "Omega_m",
        "Mcal",
    ],
    labels=[
        r"\Omega_m",
        r"\mathcal{M}",
    ],
)

g = plots.get_subplot_plotter(
    width_inch=6.2
)

g.triangle_plot(
    [gd],
    filled=True,
)

g.subplots[0, 0].set_title(
    rf"$\Omega_m = "
    rf"{om50:.3f}"
    rf"^{{+{om_plus:.3f}}}"
    rf"_{{-{om_minus:.3f}}}$",
    fontsize=13,
    pad=10,
)

g.subplots[1, 1].set_title(
    rf"$\mathcal{{M}} = "
    rf"{m50:.3f}"
    rf"^{{+{m_plus:.3f}}}"
    rf"_{{-{m_minus:.3f}}}$",
    fontsize=13,
    pad=10,
)

g.export(
    str(
        OUT
        / "pantheon_flat_lcdm_triangle.pdf"
    )
)

g.export(
    str(
        OUT
        / "pantheon_flat_lcdm_triangle.png"
    )
)

print("Pantheon+ flat LambdaCDM triangle created.")
print()
print(
    f"Omega_m = "
    f"{om50:.5f} "
    f"+{om_plus:.5f} "
    f"-{om_minus:.5f}"
)
print(
    f"Mcal = "
    f"{m50:.5f} "
    f"+{m_plus:.5f} "
    f"-{m_minus:.5f}"
)
print()
print(
    "Important: Mcal is the SN nuisance intercept, not H0."
)
print(
    "Saved to:",
    OUT / "pantheon_flat_lcdm_triangle.png",
)
