"""MCMC and plotting utilities for the BAO-only flat-LambdaCDM validation."""

import json
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from .bao_reduced import BAOReducedFlatLCDM
from .statistics import information_criteria, reduced_chi2


OMEGA_M_BOUNDS = (0.05, 0.60)
RS_H_BOUNDS = (50.0, 150.0)  # Mpc


def log_prior(theta):
    omega_m, rs_h = theta
    if not (OMEGA_M_BOUNDS[0] <= omega_m <= OMEGA_M_BOUNDS[1]):
        return -np.inf
    if not (RS_H_BOUNDS[0] <= rs_h <= RS_H_BOUNDS[1]):
        return -np.inf
    return 0.0


def chi2(theta, likelihood):
    if not np.isfinite(log_prior(theta)):
        return np.inf
    return likelihood.chi2(theta[0], theta[1])


def log_probability(theta, likelihood):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    value = likelihood.chi2(theta[0], theta[1])
    if not np.isfinite(value):
        return -np.inf
    return lp - 0.5 * value


def optimize_bao_flat_lcdm(likelihood):
    result = minimize(
        lambda x: chi2(x, likelihood),
        x0=np.array([0.30, 101.0]),
        method="L-BFGS-B",
        bounds=[OMEGA_M_BOUNDS, RS_H_BOUNDS],
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result


def run_chain(
    likelihood,
    backend_path,
    nwalkers=48,
    nsteps=12000,
    seed=20261015,
    reset=True,
    progress=True,
):
    backend_path = Path(backend_path)
    backend_path.parent.mkdir(parents=True, exist_ok=True)

    opt = optimize_bao_flat_lcdm(likelihood)
    center = np.asarray(opt.x, dtype=float)

    rng = np.random.default_rng(seed)
    scale = np.array([0.01, 1.0])
    p0 = center + rng.normal(size=(nwalkers, 2)) * scale

    for i in range(nwalkers):
        while not np.isfinite(log_prior(p0[i])):
            p0[i] = center + rng.normal(size=2) * scale

    backend = emcee.backends.HDFBackend(str(backend_path))
    if reset:
        backend.reset(nwalkers, 2)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        2,
        log_probability,
        args=(likelihood,),
        backend=backend,
    )
    sampler.run_mcmc(p0, nsteps, progress=progress)
    return sampler, opt


def autocorrelation_report(sampler):
    tau = np.asarray(sampler.get_autocorr_time(tol=0), dtype=float)
    ratios = sampler.iteration / tau
    return {
        "tau_Omega_m": float(tau[0]),
        "tau_rs_h": float(tau[1]),
        "steps_over_tau_Omega_m": float(ratios[0]),
        "steps_over_tau_rs_h": float(ratios[1]),
        "passes_50_tau_rule": bool(np.all(ratios >= 50.0)),
    }


def posterior_samples(sampler, tau):
    burn = int(np.ceil(5.0 * np.max(tau)))
    if burn >= sampler.iteration:
        raise RuntimeError(
            f"Burn-in {burn} exceeds chain length {sampler.iteration}."
        )
    return sampler.get_chain(discard=burn, flat=True), burn


def summarize_column(x):
    q = np.percentile(x, [2.5, 16, 50, 84, 97.5])
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


def write_summary(path, likelihood, sampler, opt):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tau = np.asarray(sampler.get_autocorr_time(tol=0), dtype=float)
    samples, burn = posterior_samples(sampler, tau)

    omega_m = samples[:, 0]
    rs_h = samples[:, 1]

    post = {
        "Omega_m": summarize_column(omega_m),
        "r_s_h_Mpc": summarize_column(rs_h),
        "H0_r_s_km_s": summarize_column(100.0 * rs_h),
    }

    chi2_min = float(opt.fun)
    om_best, rsh_best = [float(x) for x in opt.x]
    loglike_max = -0.5 * chi2_min

    components = likelihood.chi2_components(om_best, rsh_best)

    result = {
        "dataset": "BAO11_Ratra2021",
        "model": "flat_LCDM_BAO_reduced",
        "important_note": (
            "BAO alone constrains Omega_m and r_s*h, not H0 and r_s "
            "separately. No BAO-only H0 value is quoted."
        ),
        "parameters": ["Omega_m", "r_s_h_Mpc"],
        "priors": {
            "Omega_m": list(OMEGA_M_BOUNDS),
            "r_s_h_Mpc": list(RS_H_BOUNDS),
        },
        "best_fit": {
            "Omega_m": om_best,
            "r_s_h_Mpc": rsh_best,
            "H0_r_s_km_s": 100.0 * rsh_best,
            "chi2_min": chi2_min,
            "chi2_components": components,
        },
        "posterior": post,
        "mcmc": {
            "nwalkers": int(sampler.nwalkers),
            "nsteps": int(sampler.iteration),
            "burn_in": burn,
            **autocorrelation_report(sampler),
        },
        "fit_quality": {
            "reduced_chi2": reduced_chi2(
                chi2_min, likelihood.n_data, 2
            )[0],
            "dof": reduced_chi2(
                chi2_min, likelihood.n_data, 2
            )[1],
            **information_criteria(
                loglike_max, 2, likelihood.n_data
            ),
        },
    }

    path.write_text(json.dumps(result, indent=2))
    return result, samples


def _titles(summary):
    Om = summary["posterior"]["Omega_m"]
    R = summary["posterior"]["r_s_h_Mpc"]

    om_title = (
        rf"$\Omega_m = "
        rf"{Om['median']:.3f}"
        rf"^{{+{Om['plus_68']:.3f}}}"
        rf"_{{-{Om['minus_68']:.3f}}}$"
    )

    r_title = (
        rf"$r_s h = "
        rf"{R['median']:.2f}"
        rf"^{{+{R['plus_68']:.2f}}}"
        rf"_{{-{R['minus_68']:.2f}}}\ \mathrm{{Mpc}}$"
    )

    return om_title, r_title


def plot_triangle(pdf_path, png_path, samples, summary):
    from getdist import MCSamples, plots

    gd = MCSamples(
        samples=samples,
        names=["Omega_m", "rs_h"],
        labels=[r"\Omega_m", r"r_s h\ [\mathrm{Mpc}]"],
    )

    g = plots.get_subplot_plotter(width_inch=6.2)
    g.triangle_plot([gd], filled=True)

    om_title, r_title = _titles(summary)
    g.subplots[0, 0].set_title(om_title, fontsize=13, pad=10)
    g.subplots[1, 1].set_title(r_title, fontsize=13, pad=10)

    g.export(str(pdf_path))
    g.export(str(png_path))


def plot_prediction_table(path, likelihood, summary):
    """Create a compact observed-vs-best-fit BAO diagnostic."""
    om = summary["best_fit"]["Omega_m"]
    rsh = summary["best_fit"]["r_s_h_Mpc"]
    th = likelihood.theory_vector(om, rsh)

    # Build labels in the exact theory-vector order.
    labels = [
        "BOSS z=0.38 DM",
        "BOSS z=0.38 H",
        "BOSS z=0.51 DM",
        "BOSS z=0.51 H",
        "BOSS z=0.61 DM",
        "BOSS z=0.61 H",
        "z=0.122 DV",
        "z=0.81 DA/rs",
        "z=1.52 DV",
        "Lyα z=2.334 DM/rs",
        "Lyα z=2.334 DH/rs",
    ]

    obs = (
        list(likelihood.meta["boss_dr12"]["value"])
        + [x["value"] for x in likelihood.meta["independent"]]
        + list(likelihood.meta["lya_dr16"]["value"])
    )
    obs = np.asarray(obs, dtype=float)

    frac = (th - obs) / obs

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    y = np.arange(len(labels))
    ax.axvline(0.0, lw=1.0)
    ax.scatter(frac * 100.0, y, s=28)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Best-fit minus observed [% of observed value]")
    ax.set_title("BAO best-fit residual overview")
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)
