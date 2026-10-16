"""Inference helpers for CC-only flat LambdaCDM.

The CC-only fit is performed in (H0, Omega_m), because H(z) does not
separately constrain Omega_b h^2 and Omega_c h^2. BAO-inclusive fits use the
physical-density parameterization in the main sampling module.
"""

from dataclasses import asdict
import json
from pathlib import Path

import emcee
import h5py  # noqa: F401  (required by emcee.HDFBackend)
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from .background import FlatLCDMBackground
from .cc import CCLikelihood
from .statistics import information_criteria, reduced_chi2


H0_BOUNDS = (40.0, 100.0)
OMEGA_M_BOUNDS = (0.05, 0.60)


def log_prior(theta):
    H0, omega_m = theta
    if not (H0_BOUNDS[0] <= H0 <= H0_BOUNDS[1]):
        return -np.inf
    if not (OMEGA_M_BOUNDS[0] <= omega_m <= OMEGA_M_BOUNDS[1]):
        return -np.inf
    return 0.0


def cosmology_from_theta(theta):
    H0, omega_m = theta
    return FlatLCDMBackground(H0=float(H0), omega_m=float(omega_m))


def chi2(theta, likelihood):
    if not np.isfinite(log_prior(theta)):
        return np.inf
    return likelihood.chi2(cosmology_from_theta(theta))


def log_probability(theta, likelihood):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    value = likelihood.chi2(cosmology_from_theta(theta))
    if not np.isfinite(value):
        return -np.inf
    return lp - 0.5 * value


def optimize_cc_flat_lcdm(likelihood):
    result = minimize(
        lambda x: chi2(x, likelihood),
        x0=np.array([70.0, 0.30]),
        method="L-BFGS-B",
        bounds=[H0_BOUNDS, OMEGA_M_BOUNDS],
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

    opt = optimize_cc_flat_lcdm(likelihood)
    center = np.asarray(opt.x, dtype=float)

    rng = np.random.default_rng(seed)
    scale = np.array([0.8, 0.008])
    p0 = center + rng.normal(size=(nwalkers, 2)) * scale

    # Guarantee all walkers begin inside the prior.
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
    nsteps = sampler.iteration
    ratios = nsteps / tau
    return {
        "tau_H0": float(tau[0]),
        "tau_Omega_m": float(tau[1]),
        "steps_over_tau_H0": float(ratios[0]),
        "steps_over_tau_Omega_m": float(ratios[1]),
        "passes_50_tau_rule": bool(np.all(ratios >= 50.0)),
    }


def posterior_samples(sampler, tau):
    burn = int(np.ceil(5.0 * np.max(tau)))
    if burn >= sampler.iteration:
        raise RuntimeError(
            f"Burn-in {burn} exceeds chain length {sampler.iteration}. "
            "Run a longer chain."
        )
    # No arbitrary thinning: retain every post-burn sample.
    return sampler.get_chain(discard=burn, flat=True), burn


def summarize_samples(samples):
    names = ("H0", "Omega_m")
    summary = {}
    for i, name in enumerate(names):
        q = np.percentile(samples[:, i], [2.5, 16, 50, 84, 97.5])
        summary[name] = {
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
    return summary


def write_summary(path, likelihood, sampler, opt):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tau = np.asarray(sampler.get_autocorr_time(tol=0), dtype=float)
    samples, burn = posterior_samples(sampler, tau)
    post = summarize_samples(samples)

    chi2_min = float(opt.fun)
    H0_best, om_best = [float(x) for x in opt.x]
    loglike_max = -0.5 * chi2_min

    result = {
        "dataset": "CC32",
        "model": "flat_LCDM",
        "parameters": ["H0", "Omega_m"],
        "priors": {
            "H0": list(H0_BOUNDS),
            "Omega_m": list(OMEGA_M_BOUNDS),
        },
        "best_fit": {
            "H0": H0_best,
            "Omega_m": om_best,
            "chi2_min": chi2_min,
        },
        "posterior": post,
        "mcmc": {
            "nwalkers": int(sampler.nwalkers),
            "nsteps": int(sampler.iteration),
            "burn_in": burn,
            **autocorrelation_report(sampler),
        },
        "fit_quality": {
            "reduced_chi2": reduced_chi2(chi2_min, 32, 2)[0],
            "dof": reduced_chi2(chi2_min, 32, 2)[1],
            **information_criteria(loglike_max, 2, 32),
        },
    }

    path.write_text(json.dumps(result, indent=2))
    return result, samples


def plot_cc_fit(path, likelihood, summary):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    H0 = summary["best_fit"]["H0"]
    omega_m = summary["best_fit"]["Omega_m"]
    cosmo = FlatLCDMBackground(H0=H0, omega_m=omega_m)

    zgrid = np.linspace(0.0, 2.05, 400)
    yerr = np.sqrt(np.diag(likelihood.cov))

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.errorbar(
        likelihood.z,
        likelihood.Hobs,
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
        label="best-fit flat $\\Lambda$CDM",
    )
    ax.set_xlabel("$z$")
    ax.set_ylabel("$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_title("Cosmic chronometers: flat $\\Lambda$CDM")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def plot_triangle(path, samples, summary):
    from getdist import MCSamples, plots

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gd = MCSamples(
        samples=samples,
        names=["H0", "Omega_m"],
        labels=[r"H_0", r"\Omega_m"],
    )

    g = plots.get_subplot_plotter(width_inch=6.2)

    # GetDist displays the standard 68% and 95%
    # marginalized credible regions in the 2D panel.
    g.triangle_plot(
        [gd],
        filled=True,
    )

    # Posterior medians with marginalized 68% credible intervals.
    H0 = summary["posterior"]["H0"]
    Om = summary["posterior"]["Omega_m"]

    h0_title = (
        rf"$H_0 = "
        rf"{H0['median']:.2f}"
        rf"^{{+{H0['plus_68']:.2f}}}"
        rf"_{{-{H0['minus_68']:.2f}}}$"
    )

    om_title = (
        rf"$\Omega_m = "
        rf"{Om['median']:.3f}"
        rf"^{{+{Om['plus_68']:.3f}}}"
        rf"_{{-{Om['minus_68']:.3f}}}$"
    )

    g.subplots[0, 0].set_title(
        h0_title,
        fontsize=13,
        pad=10,
    )

    g.subplots[1, 1].set_title(
        om_title,
        fontsize=13,
        pad=10,
    )

    g.export(str(path))
