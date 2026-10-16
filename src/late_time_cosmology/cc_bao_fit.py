"""Joint CC+BAO flat-LambdaCDM inference.

The sampled parameters follow Cao, Ryan & Ratra (2021):

    h, Omega_b h^2, Omega_c h^2

with broad flat priors

    0.2   <= h           <= 1.0
    0.005 <= Omega_b h^2 <= 0.1
    0.001 <= Omega_c h^2 <= 0.99.

Unlike BAO alone, the CC likelihood supplies an absolute H(z) scale and
therefore breaks the H0-r_s scaling degeneracy.

The BAO likelihood uses the validated Stage-03 reduced representation,
which depends only on the physically relevant flat-LambdaCDM combinations
Omega_m and r_s h. This is algebraically equivalent to evaluating the full
11-point BAO vector from the same physical cosmology, while being much faster.
"""

import json
from pathlib import Path

import emcee
import numpy as np
from scipy.optimize import minimize

from .models import Cosmology
from .statistics import information_criteria, reduced_chi2


H_BOUNDS = (0.2, 1.0)
OBH2_BOUNDS = (0.005, 0.1)
OCH2_BOUNDS = (0.001, 0.99)


def derived(theta):
    h, obh2, och2 = [float(v) for v in theta]

    cosmo = Cosmology(
        h=h,
        omega_b_h2=obh2,
        omega_c_h2=och2,
    )

    rs = float(cosmo.sound_horizon_aubourg())

    return cosmo, {
        "h": h,
        "H0_km_s_Mpc": 100.0 * h,
        "omega_b_h2": obh2,
        "omega_c_h2": och2,
        "Omega_m": float(cosmo.omega_m),
        "r_s_Mpc": rs,
        "r_s_h_Mpc": rs * h,
    }


def log_prior(theta):
    h, obh2, och2 = theta

    if not (H_BOUNDS[0] <= h <= H_BOUNDS[1]):
        return -np.inf
    if not (OBH2_BOUNDS[0] <= obh2 <= OBH2_BOUNDS[1]):
        return -np.inf
    if not (OCH2_BOUNDS[0] <= och2 <= OCH2_BOUNDS[1]):
        return -np.inf

    cosmo, d = derived(theta)

    if not (0.0 < d["Omega_m"] < 1.0):
        return -np.inf
    if not (0.0 < cosmo.omega_de < 1.0):
        return -np.inf
    if not np.isfinite(d["r_s_Mpc"]) or d["r_s_Mpc"] <= 0.0:
        return -np.inf

    return 0.0


def chi2_components(theta, cc_like, bao_like):
    if not np.isfinite(log_prior(theta)):
        return None

    cosmo, d = derived(theta)

    cc = float(cc_like.chi2(cosmo))
    bao_parts = bao_like.chi2_components(
        d["Omega_m"],
        d["r_s_h_Mpc"],
    )

    return {
        "CC": cc,
        "BOSS_DR12": float(bao_parts["BOSS_DR12"]),
        "BAO_independent": float(bao_parts["independent"]),
        "Ly_alpha": float(bao_parts["Ly_alpha"]),
    }


def chi2(theta, cc_like, bao_like):
    parts = chi2_components(theta, cc_like, bao_like)
    if parts is None:
        return np.inf
    return float(sum(parts.values()))


def log_probability(theta, cc_like, bao_like):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    value = chi2(theta, cc_like, bao_like)
    if not np.isfinite(value):
        return -np.inf

    return lp - 0.5 * value


def optimize_cc_bao(cc_like, bao_like):
    """Robust multi-start maximum-likelihood optimization."""

    starts = [
        np.array([0.69, 0.024, 0.118]),
        np.array([0.71, 0.026, 0.120]),
        np.array([0.68, 0.020, 0.120]),
        np.array([0.73, 0.030, 0.125]),
        np.array([0.67, 0.035, 0.100]),
    ]

    results = []

    for x0 in starts:
        result = minimize(
            lambda x: chi2(x, cc_like, bao_like),
            x0=x0,
            method="L-BFGS-B",
            bounds=[
                H_BOUNDS,
                OBH2_BOUNDS,
                OCH2_BOUNDS,
            ],
            options={
                "ftol": 1e-13,
                "gtol": 1e-8,
                "maxiter": 5000,
            },
        )

        if result.success and np.isfinite(result.fun):
            results.append(result)

    if not results:
        raise RuntimeError("All CC+BAO optimizations failed.")

    return min(results, key=lambda r: r.fun)


def initialize_walkers(best, nwalkers=64, seed=20261016):
    rng = np.random.default_rng(seed)

    best = np.asarray(best, dtype=float)

    # The joint likelihood is much less degenerate than BAO alone.
    # These widths place the ensemble across the local posterior while
    # remaining broad enough to avoid an artificially tiny initial cloud.
    scales = np.array([0.025, 0.006, 0.012])

    walkers = []

    while len(walkers) < nwalkers:
        trial = best + rng.normal(scale=scales)

        if np.isfinite(log_prior(trial)):
            walkers.append(trial)

    return np.asarray(walkers, dtype=float)


def run_chain(
    cc_like,
    bao_like,
    backend_path,
    nwalkers=64,
    nsteps=15000,
    seed=20261016,
    reset=True,
    progress=True,
):
    backend_path = Path(backend_path)
    backend_path.parent.mkdir(parents=True, exist_ok=True)

    opt = optimize_cc_bao(cc_like, bao_like)

    p0 = initialize_walkers(
        opt.x,
        nwalkers=nwalkers,
        seed=seed,
    )

    backend = emcee.backends.HDFBackend(str(backend_path))

    if reset:
        backend.reset(nwalkers, 3)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        3,
        log_probability,
        args=(cc_like, bao_like),
        backend=backend,
    )

    sampler.run_mcmc(
        p0,
        nsteps,
        progress=progress,
    )

    return sampler, opt


def autocorrelation_report(sampler):
    tau = np.asarray(
        sampler.get_autocorr_time(tol=0),
        dtype=float,
    )

    ratio = sampler.iteration / tau

    return {
        "tau_h": float(tau[0]),
        "tau_omega_b_h2": float(tau[1]),
        "tau_omega_c_h2": float(tau[2]),
        "steps_over_tau_h": float(ratio[0]),
        "steps_over_tau_omega_b_h2": float(ratio[1]),
        "steps_over_tau_omega_c_h2": float(ratio[2]),
        "passes_50_tau_rule": bool(np.all(ratio >= 50.0)),
    }


def post_burn_samples(sampler, tau):
    burn = int(np.ceil(5.0 * np.max(tau)))

    if burn >= sampler.iteration:
        raise RuntimeError(
            f"Burn-in {burn} exceeds chain length {sampler.iteration}."
        )

    samples = sampler.get_chain(
        discard=burn,
        flat=True,
    )

    return samples, burn


def summarize(x):
    q = np.percentile(
        np.asarray(x, dtype=float),
        [2.5, 16, 50, 84, 97.5],
    )

    return {
        "q2.5": float(q[0]),
        "q16": float(q[1]),
        "median": float(q[2]),
        "q84": float(q[3]),
        "q97.5": float(q[4]),
        "minus_68": float(q[2] - q[1]),
        "plus_68": float(q[3] - q[2]),
        "minus_95": float(q[2] - q[0]),
        "plus_95": float(q[4] - q[2]),
    }


def derived_arrays(samples):
    n = len(samples)

    h = samples[:, 0]
    obh2 = samples[:, 1]
    och2 = samples[:, 2]

    H0 = 100.0 * h

    omega_m = np.empty(n, dtype=float)
    rs = np.empty(n, dtype=float)

    for i in range(n):
        cosmo = Cosmology(
            h=float(h[i]),
            omega_b_h2=float(obh2[i]),
            omega_c_h2=float(och2[i]),
        )

        omega_m[i] = cosmo.omega_m
        rs[i] = cosmo.sound_horizon_aubourg()

    rsh = rs * h

    return {
        "h": h,
        "H0_km_s_Mpc": H0,
        "omega_b_h2": obh2,
        "omega_c_h2": och2,
        "Omega_m": omega_m,
        "r_s_Mpc": rs,
        "r_s_h_Mpc": rsh,
    }


def write_summary(path, cc_like, bao_like, sampler, opt):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tau = np.asarray(
        sampler.get_autocorr_time(tol=0),
        dtype=float,
    )

    samples, burn = post_burn_samples(sampler, tau)

    arr = derived_arrays(samples)

    posterior = {
        key: summarize(value)
        for key, value in arr.items()
    }

    cosmo_best, best = derived(opt.x)
    chi2_min = float(opt.fun)
    pieces = chi2_components(
        opt.x,
        cc_like,
        bao_like,
    )

    n_data = 32 + bao_like.n_data
    n_parameters = 3
    loglike_max = -0.5 * chi2_min

    result = {
        "dataset": "CC32_plus_BAO11_Ratra2021",
        "model": "flat_LCDM",
        "sampled_parameters": [
            "h",
            "omega_b_h2",
            "omega_c_h2",
        ],
        "priors": {
            "h": list(H_BOUNDS),
            "omega_b_h2": list(OBH2_BOUNDS),
            "omega_c_h2": list(OCH2_BOUNDS),
        },
        "best_fit": {
            **best,
            "chi2_min": chi2_min,
            "chi2_components": pieces,
        },
        "posterior": posterior,
        "mcmc": {
            "nwalkers": int(sampler.nwalkers),
            "nsteps": int(sampler.iteration),
            "burn_in": int(burn),
            **autocorrelation_report(sampler),
        },
        "fit_quality": {
            "n_data": n_data,
            "n_parameters": n_parameters,
            "reduced_chi2": reduced_chi2(
                chi2_min,
                n_data,
                n_parameters,
            )[0],
            "dof": reduced_chi2(
                chi2_min,
                n_data,
                n_parameters,
            )[1],
            **information_criteria(
                loglike_max,
                n_parameters,
                n_data,
            ),
        },
        "literature_benchmark": {
            "source": "Cao, Ryan & Ratra (2021), flat LCDM H(z)+BAO",
            "best_fit_H0_km_s_Mpc": 69.11,
            "best_fit_omega_b_h2": 0.0240,
            "best_fit_omega_c_h2": 0.1179,
            "best_fit_Omega_m": 0.299,
            "best_fit_chi2": 23.64,
            "note": (
                "This is a benchmark, not a target. The present reconstruction "
                "uses the updated 32-point CC vector and correlated CC covariance, "
                "so exact agreement is not expected."
            ),
        },
    }

    path.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    return result, samples, arr


def _title(label, stats, digits=3, unit=None):
    unit_text = "" if unit is None else rf"\ {unit}"

    return (
        rf"${label} = "
        rf"{stats['median']:.{digits}f}"
        rf"^{{+{stats['plus_68']:.{digits}f}}}"
        rf"_{{-{stats['minus_68']:.{digits}f}}}"
        rf"{unit_text}$"
    )


def plot_main_triangle(pdf_path, png_path, arr, summary):
    """Publication-facing triangle in derived physical quantities."""

    from getdist import MCSamples, plots

    matrix = np.column_stack([
        arr["H0_km_s_Mpc"],
        arr["Omega_m"],
        arr["r_s_Mpc"],
    ])

    gd = MCSamples(
        samples=matrix,
        names=[
            "H0",
            "Omega_m",
            "rs",
        ],
        labels=[
            r"H_0",
            r"\Omega_m",
            r"r_s\ [\mathrm{Mpc}]",
        ],
    )

    g = plots.get_subplot_plotter(
        width_inch=7.4
    )

    g.triangle_plot(
        [gd],
        filled=True,
    )

    post = summary["posterior"]

    g.subplots[0, 0].set_title(
        _title(
            "H_0",
            post["H0_km_s_Mpc"],
            digits=2,
        ),
        fontsize=11,
        pad=8,
    )

    g.subplots[1, 1].set_title(
        _title(
            r"\Omega_m",
            post["Omega_m"],
            digits=3,
        ),
        fontsize=11,
        pad=8,
    )

    g.subplots[2, 2].set_title(
        _title(
            "r_s",
            post["r_s_Mpc"],
            digits=2,
            unit=r"\mathrm{Mpc}",
        ),
        fontsize=11,
        pad=8,
    )

    g.export(str(pdf_path))
    g.export(str(png_path))


def plot_sampled_triangle(pdf_path, png_path, samples, summary):
    """Diagnostic triangle for the actual sampled parameters."""

    from getdist import MCSamples, plots

    gd = MCSamples(
        samples=samples,
        names=[
            "h",
            "obh2",
            "och2",
        ],
        labels=[
            r"h",
            r"\Omega_b h^2",
            r"\Omega_c h^2",
        ],
    )

    g = plots.get_subplot_plotter(
        width_inch=7.4
    )

    g.triangle_plot(
        [gd],
        filled=True,
    )

    post = summary["posterior"]

    g.subplots[0, 0].set_title(
        _title(
            "h",
            post["h"],
            digits=3,
        ),
        fontsize=10,
        pad=8,
    )

    g.subplots[1, 1].set_title(
        _title(
            r"\Omega_b h^2",
            post["omega_b_h2"],
            digits=4,
        ),
        fontsize=10,
        pad=8,
    )

    g.subplots[2, 2].set_title(
        _title(
            r"\Omega_c h^2",
            post["omega_c_h2"],
            digits=4,
        ),
        fontsize=10,
        pad=8,
    )

    g.export(str(pdf_path))
    g.export(str(png_path))


def plot_cc_best_fit(path, cc_like, summary):
    """CC data and best-fitting joint CC+BAO expansion history."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    b = summary["best_fit"]

    cosmo = Cosmology(
        h=b["h"],
        omega_b_h2=b["omega_b_h2"],
        omega_c_h2=b["omega_c_h2"],
    )

    zgrid = np.linspace(
        0.0,
        2.05,
        400,
    )

    yerr = np.sqrt(
        np.diag(cc_like.cov)
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 5.2)
    )

    ax.errorbar(
        cc_like.z,
        cc_like.Hobs,
        yerr=yerr,
        fmt="o",
        ms=4,
        capsize=2,
        label="CC data",
    )

    ax.plot(
        zgrid,
        cosmo.H(zgrid),
        lw=1.8,
        label="joint CC+BAO best fit",
    )

    ax.set_xlabel("$z$")
    ax.set_ylabel(
        "$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]"
    )
    ax.set_title(
        "CC + BAO: flat $\\Lambda$CDM"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=250,
    )
    plt.close(fig)
