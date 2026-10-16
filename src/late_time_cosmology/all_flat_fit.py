"""Pantheon+SH0ES + CC + BAO flat-LambdaCDM inference.

Sampled cosmological parameters
-------------------------------
    h, Omega_b h^2, Omega_c h^2

Derived
-------
    H0, Omega_m, r_s, r_s h

Pantheon+SH0ES treatment
------------------------
The calibrated SN likelihood is the public Pantheon+SH0ES construction:
calibrator-host SNe use CEPH_DIST, Hubble-flow SNe use the cosmological
luminosity distance, and the global SN absolute magnitude M_B is analytically
profiled/marginalized.

For MCMC speed, the Pantheon+SH0ES profiled chi2 is represented by a high-
accuracy one-dimensional spline cache in Omega_m. At fixed Omega_m its H0
dependence is exactly quadratic in x = 5 log10(H0), so no approximation is
made in the H0 direction.
"""

from pathlib import Path
import json

import emcee
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import cho_solve
from scipy.optimize import minimize, differential_evolution

from .models import Cosmology
from .statistics import information_criteria, reduced_chi2


H_BOUNDS = (0.2, 1.0)
OBH2_BOUNDS = (0.005, 0.1)
OCH2_BOUNDS = (0.001, 0.99)


class PantheonSH0ESFastProfile:
    """Fast profiled Pantheon+SH0ES chi2 for joint MCMC.

    The exact profiled likelihood at fixed Omega_m can be written

        chi2(H0,Omega_m) = A(Omega_m)
                         + B(Omega_m) x
                         + C x^2,

    where x = 5 log10(H0) and C is constant.

    A and B are precomputed from the full covariance on a dense Omega_m grid
    and cubic-spline interpolated. This avoids an O(N_SN^2) triangular solve
    at every MCMC point.
    """

    def __init__(
        self,
        exact_like,
        omega_min=0.20,
        omega_max=0.45,
        n_omega=801,
    ):
        self.exact = exact_like

        self.omega_grid = np.linspace(
            float(omega_min),
            float(omega_max),
            int(n_omega),
        )

        # Build all residual-base vectors and solve C^{-1} A in one batch.
        amat = np.column_stack([
            exact_like._a_vector(om)
            for om in self.omega_grid
        ])

        cinv_a = cho_solve(
            exact_like._cho,
            amat,
            check_finite=False,
        )

        aa = np.sum(
            amat * cinv_a,
            axis=0,
        )
        one_a = exact_like._one @ cinv_a
        hf_a = exact_like._hf @ cinv_a

        one_one = exact_like._one_one
        one_hf = exact_like._one_hf
        hf_hf = exact_like._hf_hf

        self.A = (
            aa
            - one_a**2 / one_one
        )

        self.B = 2.0 * (
            hf_a
            - one_a * one_hf / one_one
        )

        self.C = float(
            hf_hf
            - one_hf**2 / one_one
        )

        self._A_spline = CubicSpline(
            self.omega_grid,
            self.A,
            bc_type="natural",
        )
        self._B_spline = CubicSpline(
            self.omega_grid,
            self.B,
            bc_type="natural",
        )

        self.omega_min = float(self.omega_grid[0])
        self.omega_max = float(self.omega_grid[-1])

    def chi2(self, H0, omega_m):
        if not (
            self.omega_min
            <= omega_m
            <= self.omega_max
        ):
            return np.inf

        x = 5.0 * np.log10(float(H0))

        return float(
            self._A_spline(omega_m)
            + self._B_spline(omega_m) * x
            + self.C * x*x
        )

    def max_abs_error(self, test_points):
        errs = []

        for H0, omega_m in test_points:
            fast = self.chi2(
                H0,
                omega_m,
            )
            exact = self.exact.chi2_profiled(
                H0,
                omega_m,
            )
            errs.append(
                abs(fast - exact)
            )

        return float(max(errs))


def derived(theta):
    h, obh2, och2 = [
        float(v)
        for v in theta
    ]

    cosmo = Cosmology(
        h=h,
        omega_b_h2=obh2,
        omega_c_h2=och2,
    )

    rs = float(
        cosmo.sound_horizon_aubourg()
    )

    return cosmo, {
        "h": h,
        "H0_km_s_Mpc": 100.0*h,
        "omega_b_h2": obh2,
        "omega_c_h2": och2,
        "Omega_m": float(cosmo.omega_m),
        "r_s_Mpc": rs,
        "r_s_h_Mpc": rs*h,
    }


def log_prior(theta):
    h, obh2, och2 = theta

    if not (
        H_BOUNDS[0]
        <= h
        <= H_BOUNDS[1]
    ):
        return -np.inf

    if not (
        OBH2_BOUNDS[0]
        <= obh2
        <= OBH2_BOUNDS[1]
    ):
        return -np.inf

    if not (
        OCH2_BOUNDS[0]
        <= och2
        <= OCH2_BOUNDS[1]
    ):
        return -np.inf

    cosmo, d = derived(theta)

    if not (
        0.0
        < d["Omega_m"]
        < 1.0
    ):
        return -np.inf

    if not (
        0.0
        < cosmo.omega_de
        < 1.0
    ):
        return -np.inf

    if not (
        np.isfinite(d["r_s_Mpc"])
        and d["r_s_Mpc"] > 0.0
    ):
        return -np.inf

    return 0.0


def chi2_components(
    theta,
    cc_like,
    bao_like,
    sn_fast,
):
    if not np.isfinite(
        log_prior(theta)
    ):
        return None

    cosmo, d = derived(theta)

    cc = float(
        cc_like.chi2(cosmo)
    )

    bao = bao_like.chi2_components(
        d["Omega_m"],
        d["r_s_h_Mpc"],
    )

    sn = float(
        sn_fast.chi2(
            d["H0_km_s_Mpc"],
            d["Omega_m"],
        )
    )

    return {
        "Pantheon_plus_SH0ES": sn,
        "CC": cc,
        "BOSS_DR12": float(
            bao["BOSS_DR12"]
        ),
        "BAO_independent": float(
            bao["independent"]
        ),
        "Ly_alpha": float(
            bao["Ly_alpha"]
        ),
    }


def chi2(
    theta,
    cc_like,
    bao_like,
    sn_fast,
):
    parts = chi2_components(
        theta,
        cc_like,
        bao_like,
        sn_fast,
    )

    if parts is None:
        return np.inf

    return float(
        sum(parts.values())
    )


def log_probability(
    theta,
    cc_like,
    bao_like,
    sn_fast,
):
    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    value = chi2(
        theta,
        cc_like,
        bao_like,
        sn_fast,
    )

    if not np.isfinite(value):
        return -np.inf

    return lp - 0.5*value


def _finite_optimizer_objective(
    x,
    cc_like,
    bao_like,
    sn_fast,
):
    """Finite-valued wrapper used only by deterministic optimizers.

    The scientific likelihood correctly returns +inf outside the allowed
    physical/prior region.  Some scipy line-search algorithms, however,
    evaluate algebra such as inf-inf while building trial steps, which can
    generate NaNs and make an otherwise healthy optimization report failure.

    MCMC still uses the exact log-prior with -inf outside support.  This
    wrapper is only an optimizer numerical safeguard.
    """
    value = chi2(
        x,
        cc_like,
        bao_like,
        sn_fast,
    )

    if np.isfinite(value):
        return float(value)

    # Large finite penalty: far above any plausible cosmological chi2,
    # but safe for optimizer arithmetic.
    return 1.0e30


def optimize_all(
    cc_like,
    bao_like,
    sn_fast,
):
    """Robust global + local optimization of the full joint likelihood."""

    bounds = [
        H_BOUNDS,
        OBH2_BOUNDS,
        OCH2_BOUNDS,
    ]

    objective = lambda x: _finite_optimizer_objective(
        x,
        cc_like,
        bao_like,
        sn_fast,
    )

    starts = [
        np.array([
            0.73,
            0.027,
            0.121,
        ]),
        np.array([
            0.71,
            0.026,
            0.120,
        ]),
        np.array([
            0.735,
            0.030,
            0.120,
        ]),
        np.array([
            0.70,
            0.024,
            0.118,
        ]),
    ]

    candidates = []

    # First keep the explicitly chosen physically sensible starts.
    for x0 in starts:
        result = minimize(
            objective,
            x0=x0,
            method="Powell",
            bounds=bounds,
            options={
                "xtol": 1e-9,
                "ftol": 1e-10,
                "maxiter": 5000,
            },
        )

        if (
            np.isfinite(result.fun)
            and result.fun < 1.0e20
        ):
            candidates.append(result)

    # Independent global search.  This makes the audit insensitive to a
    # particular local starting point and is inexpensive for only 3 parameters.
    de = differential_evolution(
        objective,
        bounds=bounds,
        seed=20261016,
        tol=1e-8,
        atol=1e-8,
        popsize=18,
        maxiter=600,
        polish=False,
        updating="immediate",
        workers=1,
    )

    if (
        np.isfinite(de.fun)
        and de.fun < 1.0e20
    ):
        candidates.append(de)

        # Local polish from the global solution.
        polished = minimize(
            objective,
            x0=np.asarray(de.x, dtype=float),
            method="Powell",
            bounds=bounds,
            options={
                "xtol": 1e-10,
                "ftol": 1e-11,
                "maxiter": 8000,
            },
        )

        if (
            np.isfinite(polished.fun)
            and polished.fun < 1.0e20
        ):
            candidates.append(polished)

    if not candidates:
        raise RuntimeError(
            "Combined-data optimization could not find a finite "
            "point inside the physical likelihood support."
        )

    best = min(
        candidates,
        key=lambda r: float(r.fun),
    )

    # Re-evaluate with the actual scientific chi2, not the finite penalty.
    exact_fun = chi2(
        best.x,
        cc_like,
        bao_like,
        sn_fast,
    )

    if not np.isfinite(exact_fun):
        raise RuntimeError(
            "Optimizer returned a point outside the physical likelihood support."
        )

    best.fun = float(exact_fun)
    return best


def initialize_walkers(
    best,
    nwalkers=64,
    seed=20261016,
):
    rng = np.random.default_rng(seed)

    best = np.asarray(
        best,
        dtype=float,
    )

    scales = np.array([
        0.012,
        0.0035,
        0.007,
    ])

    walkers = []

    while len(walkers) < nwalkers:
        trial = (
            best
            + rng.normal(scale=scales)
        )

        if np.isfinite(
            log_prior(trial)
        ):
            walkers.append(trial)

    return np.asarray(
        walkers,
        dtype=float,
    )


def run_chain(
    cc_like,
    bao_like,
    sn_fast,
    backend_path,
    nwalkers=64,
    nsteps=15000,
    seed=20261016,
    reset=True,
    progress=True,
):
    backend_path = Path(
        backend_path
    )
    backend_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    opt = optimize_all(
        cc_like,
        bao_like,
        sn_fast,
    )

    p0 = initialize_walkers(
        opt.x,
        nwalkers=nwalkers,
        seed=seed,
    )

    backend = emcee.backends.HDFBackend(
        str(backend_path)
    )

    if reset:
        backend.reset(
            nwalkers,
            3,
        )

    sampler = emcee.EnsembleSampler(
        nwalkers,
        3,
        log_probability,
        args=(
            cc_like,
            bao_like,
            sn_fast,
        ),
        backend=backend,
    )

    sampler.run_mcmc(
        p0,
        nsteps,
        progress=progress,
    )

    return sampler, opt


def autocorrelation_report(
    sampler,
):
    tau = np.asarray(
        sampler.get_autocorr_time(
            tol=0
        ),
        dtype=float,
    )

    ratio = (
        sampler.iteration
        / tau
    )

    return {
        "tau_h": float(tau[0]),
        "tau_omega_b_h2": float(tau[1]),
        "tau_omega_c_h2": float(tau[2]),
        "steps_over_tau_h": float(
            ratio[0]
        ),
        "steps_over_tau_omega_b_h2": float(
            ratio[1]
        ),
        "steps_over_tau_omega_c_h2": float(
            ratio[2]
        ),
        "passes_50_tau_rule": bool(
            np.all(
                ratio >= 50.0
            )
        ),
    }


def post_burn_samples(
    sampler,
    tau,
):
    burn = int(
        np.ceil(
            5.0*np.max(tau)
        )
    )

    if burn >= sampler.iteration:
        raise RuntimeError(
            f"Burn-in {burn} exceeds "
            f"chain length {sampler.iteration}."
        )

    samples = sampler.get_chain(
        discard=burn,
        flat=True,
    )

    return samples, burn


def summarize(x):
    q = np.percentile(
        np.asarray(
            x,
            dtype=float,
        ),
        [
            2.5,
            16,
            50,
            84,
            97.5,
        ],
    )

    return {
        "q2.5": float(q[0]),
        "q16": float(q[1]),
        "median": float(q[2]),
        "q84": float(q[3]),
        "q97.5": float(q[4]),
        "minus_68": float(
            q[2] - q[1]
        ),
        "plus_68": float(
            q[3] - q[2]
        ),
        "minus_95": float(
            q[2] - q[0]
        ),
        "plus_95": float(
            q[4] - q[2]
        ),
    }


def derived_arrays(samples):
    n = len(samples)

    h = samples[:, 0]
    obh2 = samples[:, 1]
    och2 = samples[:, 2]

    H0 = 100.0*h

    omega_m = np.empty(
        n,
        dtype=float,
    )

    rs = np.empty(
        n,
        dtype=float,
    )

    for i in range(n):
        c = Cosmology(
            h=float(h[i]),
            omega_b_h2=float(
                obh2[i]
            ),
            omega_c_h2=float(
                och2[i]
            ),
        )

        omega_m[i] = c.omega_m
        rs[i] = (
            c.sound_horizon_aubourg()
        )

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


def write_summary(
    path,
    cc_like,
    bao_like,
    sn_exact,
    sn_fast,
    sampler,
    opt,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tau = np.asarray(
        sampler.get_autocorr_time(
            tol=0
        ),
        dtype=float,
    )

    samples, burn = post_burn_samples(
        sampler,
        tau,
    )

    arr = derived_arrays(
        samples
    )

    posterior = {
        key: summarize(value)
        for key, value in arr.items()
    }

    _, best = derived(
        opt.x
    )

    components = chi2_components(
        opt.x,
        cc_like,
        bao_like,
        sn_fast,
    )

    chi2_min = float(
        sum(components.values())
    )

    H0_best = best[
        "H0_km_s_Mpc"
    ]
    Om_best = best[
        "Omega_m"
    ]

    MB_best = float(
        sn_exact.best_MB(
            H0_best,
            Om_best,
        )
    )

    n_data = (
        sn_exact.n
        + len(cc_like.z)
        + bao_like.n_data
    )

    # 3 sampled cosmological parameters + the profiled SN M_B nuisance.
    n_parameters_effective = 4

    loglike_max = -0.5*chi2_min

    result = {
        "dataset": (
            "Pantheon+SH0ES_plus_CC32_plus_BAO11"
        ),
        "model": "flat_LCDM",
        "sampled_parameters": [
            "h",
            "omega_b_h2",
            "omega_c_h2",
        ],
        "profiled_nuisance": [
            "M_B",
        ],
        "best_fit": {
            **best,
            "M_B": MB_best,
            "chi2_min": chi2_min,
            "chi2_components": components,
        },
        "posterior": posterior,
        "mcmc": {
            "nwalkers": int(
                sampler.nwalkers
            ),
            "nsteps": int(
                sampler.iteration
            ),
            "burn_in": int(burn),
            **autocorrelation_report(
                sampler
            ),
        },
        "fit_quality": {
            "n_data": int(n_data),
            "n_cosmological_parameters": 3,
            "n_profiled_nuisance_parameters": 1,
            "n_parameters_effective": 4,
            "dof": int(
                n_data
                - n_parameters_effective
            ),
            "reduced_chi2": float(
                chi2_min
                / (
                    n_data
                    - n_parameters_effective
                )
            ),
            **information_criteria(
                loglike_max,
                n_parameters_effective,
                n_data,
            ),
        },
        "pantheon_fast_profile": {
            "omega_range": [
                sn_fast.omega_min,
                sn_fast.omega_max,
            ],
            "n_omega": int(
                len(
                    sn_fast.omega_grid
                )
            ),
        },
    }

    path.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    return (
        result,
        samples,
        arr,
    )


def _title(
    label,
    stats,
    digits=3,
    unit=None,
):
    unit_text = (
        ""
        if unit is None
        else rf"\ {unit}"
    )

    return (
        rf"${label} = "
        rf"{stats['median']:.{digits}f}"
        rf"^{{+{stats['plus_68']:.{digits}f}}}"
        rf"_{{-{stats['minus_68']:.{digits}f}}}"
        rf"{unit_text}$"
    )


def plot_main_triangle(
    pdf_path,
    png_path,
    arr,
    summary,
):
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
            post[
                "H0_km_s_Mpc"
            ],
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

    g.export(
        str(pdf_path)
    )
    g.export(
        str(png_path)
    )


def plot_sampled_triangle(
    pdf_path,
    png_path,
    samples,
    summary,
):
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
            post[
                "omega_b_h2"
            ],
            digits=4,
        ),
        fontsize=10,
        pad=8,
    )

    g.subplots[2, 2].set_title(
        _title(
            r"\Omega_c h^2",
            post[
                "omega_c_h2"
            ],
            digits=4,
        ),
        fontsize=10,
        pad=8,
    )

    g.export(
        str(pdf_path)
    )
    g.export(
        str(png_path)
    )


def plot_h0_comparison(
    path,
    project_root,
    combined_summary,
):
    """Compare the calibrated H0 posteriors already produced by the project."""

    import matplotlib.pyplot as plt

    root = Path(
        project_root
    )

    entries = []

    def load_optional(
        label,
        relpath,
        key,
    ):
        p = root / relpath
        if p.exists():
            s = json.loads(
                p.read_text()
            )
            q = s["posterior"][key]
            entries.append(
                (
                    label,
                    q["median"],
                    q["minus_68"],
                    q["plus_68"],
                )
            )

    load_optional(
        "CC",
        "results/cc_flat_lcdm/summary.json",
        "H0",
    )

    load_optional(
        r"BAO ($r_s=147.4$ Mpc)",
        "results/bao_flat_lcdm_fixed_rs/summary.json",
        "H0_km_s_Mpc",
    )

    load_optional(
        "CC + BAO",
        "results/cc_bao_flat_lcdm/summary.json",
        "H0_km_s_Mpc",
    )

    load_optional(
        "Pantheon+ + SH0ES",
        "results/pantheon_shoes_flat_lcdm/summary.json",
        "H0_km_s_Mpc",
    )

    q = combined_summary[
        "posterior"
    ][
        "H0_km_s_Mpc"
    ]

    entries.append(
        (
            "Pantheon+SH0ES + CC + BAO",
            q["median"],
            q["minus_68"],
            q["plus_68"],
        )
    )

    labels = [
        x[0]
        for x in entries
    ]

    centers = np.array([
        x[1]
        for x in entries
    ])

    lower = np.array([
        x[2]
        for x in entries
    ])

    upper = np.array([
        x[3]
        for x in entries
    ])

    y = np.arange(
        len(entries)
    )

    fig, ax = plt.subplots(
        figsize=(8.0, 5.2)
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
        "Flat $\\Lambda$CDM: 68% credible intervals"
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=250,
    )

    plt.close(fig)
