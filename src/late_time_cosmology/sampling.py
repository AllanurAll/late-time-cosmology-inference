import numpy as np
import emcee
from .models import Cosmology


MODEL_PARAMETERS = {
    "flat_lcdm": ("h", "omega_b_h2", "omega_c_h2"),
    "nonflat_lcdm": ("h", "omega_b_h2", "omega_c_h2", "omega_k"),
    "flat_xcdm": ("h", "omega_b_h2", "omega_c_h2", "w_x"),
    "nonflat_xcdm": ("h", "omega_b_h2", "omega_c_h2", "omega_k", "w_x"),
}

PRIOR_BOUNDS = {
    "h": (0.2, 1.0),
    "omega_b_h2": (0.005, 0.1),
    "omega_c_h2": (0.001, 0.99),
    "omega_k": (-0.7, 0.7),
    "w_x": (-3.0, 0.2),
}


def vector_to_cosmology(theta, model):
    names = MODEL_PARAMETERS[model]
    d = dict(zip(names, theta))
    return Cosmology(
        h=d["h"],
        omega_b_h2=d["omega_b_h2"],
        omega_c_h2=d["omega_c_h2"],
        omega_k=d.get("omega_k", 0.0),
        w_x=d.get("w_x", -1.0),
    )


def log_prior(theta, model):
    names = MODEL_PARAMETERS[model]
    for name, value in zip(names, theta):
        lo, hi = PRIOR_BOUNDS[name]
        if not (lo <= value <= hi):
            return -np.inf
    cosmo = vector_to_cosmology(theta, model)
    if cosmo.omega_m <= 0.0 or cosmo.omega_de <= 0.0:
        return -np.inf
    # Guard against pathological E(z) over the relevant late-time range.
    ztest = np.linspace(0.0, 2.5, 64)
    if np.any(~np.isfinite(cosmo.E(ztest))):
        return -np.inf
    return 0.0


def log_probability(theta, model, likelihood):
    lp = log_prior(theta, model)
    if not np.isfinite(lp):
        return -np.inf
    cosmo = vector_to_cosmology(theta, model)
    return lp + likelihood.loglike(cosmo)


def default_center(model):
    base = {
        "h": 0.69,
        "omega_b_h2": 0.024,
        "omega_c_h2": 0.118,
        "omega_k": 0.0,
        "w_x": -1.0,
    }
    return np.array([base[n] for n in MODEL_PARAMETERS[model]], dtype=float)


def initialize_walkers(model, nwalkers=48, scale=1e-3, rng=None):
    rng = np.random.default_rng(rng)
    center = default_center(model)
    p0 = center + scale * rng.normal(size=(nwalkers, len(center)))
    return p0


def run_emcee(model, likelihood, nwalkers=48, nsteps=20000, backend=None,
              progress=True, rng=None):
    ndim = len(MODEL_PARAMETERS[model])
    if nwalkers < 2 * ndim:
        raise ValueError("nwalkers must be at least 2*ndim")
    p0 = initialize_walkers(model, nwalkers=nwalkers, rng=rng)
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(model, likelihood), backend=backend
    )
    sampler.run_mcmc(p0, nsteps, progress=progress)
    return sampler


def posterior_summary(samples, names):
    out = {}
    for i, name in enumerate(names):
        q2p5, q16, q50, q84, q97p5 = np.percentile(
            samples[:, i], [2.5, 16, 50, 84, 97.5]
        )
        out[name] = {
            "median": q50,
            "minus_1sigma": q50-q16,
            "plus_1sigma": q84-q50,
            "minus_2sigma": q50-q2p5,
            "plus_2sigma": q97p5-q50,
        }
    return out
