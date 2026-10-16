"""BAO-only flat-LambdaCDM fit with a Ratra-style physical-density parameterization.

The sampled parameters are

    h, Omega_b h^2, Omega_c h^2

with the broad flat priors used by Cao, Ryan & Ratra (2021):

    0.2   <= h             <= 1.0
    0.005 <= Omega_b h^2   <= 0.1
    0.001 <= Omega_c h^2   <= 0.99

Omega_m is derived using Omega_nu0 = 0.0014, and r_s is computed with the
Aubourg et al. approximation used in that analysis.

Important:
BAO supplies essentially two late-time combinations (Omega_m and r_s h) for
this flat-LambdaCDM problem. With three sampled physical-density parameters,
a strong degeneracy remains. The resulting BAO-only H0 posterior is therefore
prior-conditioned and must not be described as an assumption-free H0
measurement.
"""

import json
from pathlib import Path

import emcee
import numpy as np
from scipy.optimize import brentq, minimize

from .bao_reduced import BAOReducedFlatLCDM
from .models import Cosmology, OMEGA_NU0
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
    return {
        "h": h,
        "H0": 100.0 * h,
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

    d = derived(theta)
    if not np.isfinite(d["Omega_m"]) or d["Omega_m"] <= 0.0:
        return -np.inf
    if not np.isfinite(d["r_s_Mpc"]) or d["r_s_Mpc"] <= 0.0:
        return -np.inf

    return 0.0


def chi2(theta, likelihood):
    if not np.isfinite(log_prior(theta)):
        return np.inf

    d = derived(theta)
    return likelihood.chi2(
        d["Omega_m"],
        d["r_s_h_Mpc"],
    )


def log_probability(theta, likelihood):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    value = chi2(theta, likelihood)
    if not np.isfinite(value):
        return -np.inf

    return lp - 0.5 * value


def optimize_physical(likelihood):
    # The maximum is non-unique along the BAO degeneracy ridge.
    # This optimization returns one valid representative point.
    result = minimize(
        lambda x: chi2(x, likelihood),
        x0=np.array([0.69, 0.024, 0.118]),
        method="L-BFGS-B",
        bounds=[H_BOUNDS, OBH2_BOUNDS, OCH2_BOUNDS],
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result


def _ridge_point(h, omega_m=0.2967011, target_rsh=100.92527):
    """Construct an approximate high-likelihood point at a chosen h."""

    def objective(obh2):
        och2 = h*h*(omega_m - OMEGA_NU0) - obh2

        if not (OCH2_BOUNDS[0] < och2 < OCH2_BOUNDS[1]):
            return np.nan

        c = Cosmology(
            h=h,
            omega_b_h2=obh2,
            omega_c_h2=och2,
        )
        return c.sound_horizon_aubourg()*h - target_rsh

    grid = np.linspace(
        OBH2_BOUNDS[0] + 1e-6,
        OBH2_BOUNDS[1] - 1e-6,
        300,
    )

    for left, right in zip(grid[:-1], grid[1:]):
        fl = objective(left)
        fr = objective(right)

        if (
            np.isfinite(fl)
            and np.isfinite(fr)
            and fl * fr <= 0.0
        ):
            obh2 = brentq(objective, left, right)
            och2 = h*h*(omega_m - OMEGA_NU0) - obh2
            return np.array([h, obh2, och2], dtype=float)

    raise RuntimeError(f"Could not construct ridge point for h={h}")


def initialize_walkers(nwalkers=64, seed=20261015):
    """Spread walkers along the BAO degeneracy ridge before perturbing them."""

    rng = np.random.default_rng(seed)

    # The broad Ratra priors permit a long BAO degeneracy ridge. Populate
    # that ridge from the start so the ensemble does not have to discover
    # it from one narrow local cloud.
    h_values = np.linspace(0.51, 0.98, nwalkers)
    rng.shuffle(h_values)

    walkers = []

    for h in h_values:
        center = _ridge_point(h)

        # Small perturbations normal to/along the ridge.
        for _ in range(10000):
            trial = center + rng.normal(
                scale=np.array([0.003, 0.0010, 0.0015])
            )
            if np.isfinite(log_prior(trial)):
                walkers.append(trial)
                break
        else:
            raise RuntimeError("Failed to initialize a valid walker.")

    return np.asarray(walkers, dtype=float)


def run_chain(
    likelihood,
    backend_path,
    nwalkers=64,
    nsteps=20000,
    seed=20261015,
    reset=True,
    progress=True,
):
    backend_path = Path(backend_path)
    backend_path.parent.mkdir(parents=True, exist_ok=True)

    p0 = initialize_walkers(
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
        args=(likelihood,),
        backend=backend,
    )

    sampler.run_mcmc(
        p0,
        nsteps,
        progress=progress,
    )

    return sampler, optimize_physical(likelihood)


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

    return (
        sampler.get_chain(discard=burn, flat=True),
        burn,
    )


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
    h = samples[:, 0]
    obh2 = samples[:, 1]
    och2 = samples[:, 2]

    H0 = 100.0*h
    omega_m = (obh2 + och2)/h**2 + OMEGA_NU0

    rs = np.empty(len(samples), dtype=float)
    for i in range(len(samples)):
        c = Cosmology(
            h=float(h[i]),
            omega_b_h2=float(obh2[i]),
            omega_c_h2=float(och2[i]),
        )
        rs[i] = c.sound_horizon_aubourg()

    rsh = rs*h

    return {
        "h": h,
        "H0_km_s_Mpc": H0,
        "omega_b_h2": obh2,
        "omega_c_h2": och2,
        "Omega_m": omega_m,
        "r_s_Mpc": rs,
        "r_s_h_Mpc": rsh,
    }


def write_summary(path, likelihood, sampler, opt):
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

    representative_best = derived(opt.x)
    chi2_min = float(opt.fun)
    loglike_max = -0.5*chi2_min

    result = {
        "dataset": "BAO11_Ratra2021",
        "model": "flat_LCDM_physical_density_BAO_only",
        "important_note": (
            "The BAO-only H0 posterior in this parameterization is "
            "prior-conditioned because BAO leaves a long degeneracy in "
            "(h, Omega_b h^2, Omega_c h^2). The robust BAO combinations "
            "remain Omega_m and r_s h."
        ),
        "priors": {
            "h": list(H_BOUNDS),
            "omega_b_h2": list(OBH2_BOUNDS),
            "omega_c_h2": list(OCH2_BOUNDS),
        },
        "representative_best_fit": {
            **representative_best,
            "chi2_min": chi2_min,
            "note": (
                "The maximum is non-unique along the BAO degeneracy ridge."
            ),
        },
        "posterior": posterior,
        "mcmc": {
            "nwalkers": int(sampler.nwalkers),
            "nsteps": int(sampler.iteration),
            "burn_in": int(burn),
            **autocorrelation_report(sampler),
        },
        "fit_quality": {
            "reduced_chi2": reduced_chi2(
                chi2_min, likelihood.n_data, 3
            )[0],
            "dof": reduced_chi2(
                chi2_min, likelihood.n_data, 3
            )[1],
            **information_criteria(
                loglike_max, 3, likelihood.n_data
            ),
        },
    }

    path.write_text(json.dumps(result, indent=2))
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


def plot_derived_triangle(pdf_path, png_path, arr, summary):
    from getdist import MCSamples, plots

    matrix = np.column_stack([
        arr["H0_km_s_Mpc"],
        arr["r_s_Mpc"],
        arr["Omega_m"],
    ])

    gd = MCSamples(
        samples=matrix,
        names=["H0", "rs", "Omega_m"],
        labels=[
            r"H_0",
            r"r_s\ [\mathrm{Mpc}]",
            r"\Omega_m",
        ],
    )

    g = plots.get_subplot_plotter(width_inch=7.4)
    g.triangle_plot([gd], filled=True)

    post = summary["posterior"]

    g.subplots[0, 0].set_title(
        _title("H_0", post["H0_km_s_Mpc"], digits=1),
        fontsize=11,
        pad=8,
    )
    g.subplots[1, 1].set_title(
        _title(
            "r_s",
            post["r_s_Mpc"],
            digits=1,
            unit=r"\mathrm{Mpc}",
        ),
        fontsize=11,
        pad=8,
    )
    g.subplots[2, 2].set_title(
        _title(
            r"\Omega_m",
            post["Omega_m"],
            digits=3,
        ),
        fontsize=11,
        pad=8,
    )

    g.export(str(pdf_path))
    g.export(str(png_path))


def plot_sampled_triangle(pdf_path, png_path, samples, summary):
    from getdist import MCSamples, plots

    gd = MCSamples(
        samples=samples,
        names=["h", "obh2", "och2"],
        labels=[
            r"h",
            r"\Omega_b h^2",
            r"\Omega_c h^2",
        ],
    )

    g = plots.get_subplot_plotter(width_inch=7.4)
    g.triangle_plot([gd], filled=True)

    post = summary["posterior"]

    g.subplots[0, 0].set_title(
        _title("h", post["h"], digits=3),
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
