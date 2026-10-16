"""Joint CC+BAO inference for spatially-flat XCDM."""

import json
from pathlib import Path

import emcee
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import differential_evolution, minimize

from .models import Cosmology
from .statistics import (
    information_criteria,
    reduced_chi2,
)
from .xcdm_background import (
    H_flat_xcdm,
)


H_BOUNDS = (0.2, 1.0)
OBH2_BOUNDS = (0.005, 0.1)
OCH2_BOUNDS = (0.001, 0.99)

# Cao & Ratra (2023) Table 3 uses the broad flat prior [-5, 0.33].
W_X_BOUNDS = (-5.0, 0.33)


class CCFlatXCDM:
    """CC covariance likelihood evaluated with a flat-XCDM H(z)."""

    def __init__(self, cc_like):
        self.z = np.asarray(
            cc_like.z,
            dtype=float,
        )
        self.Hobs = np.asarray(
            cc_like.Hobs,
            dtype=float,
        )
        self.cov = np.asarray(
            cc_like.cov,
            dtype=float,
        )

        self._cho = cho_factor(
            self.cov,
            lower=True,
            check_finite=False,
        )

        self.n_data = len(
            self.z
        )

    def chi2(
        self,
        H0,
        omega_m,
        w_x,
    ):
        model = H_flat_xcdm(
            self.z,
            H0,
            omega_m,
            w_x,
        )

        d = (
            self.Hobs
            - model
        )

        return float(
            d
            @ cho_solve(
                self._cho,
                d,
                check_finite=False,
            )
        )


def derived(theta):
    h, obh2, och2, w_x = [
        float(v)
        for v in theta
    ]

    # The standard early-time sound-horizon approximation depends on
    # physical matter/baryon densities.  w_X enters the late-time expansion.
    c = Cosmology(
        h=h,
        omega_b_h2=obh2,
        omega_c_h2=och2,
    )

    rs = float(
        c.sound_horizon_aubourg()
    )

    return {
        "h": h,
        "H0_km_s_Mpc": 100.0 * h,
        "omega_b_h2": obh2,
        "omega_c_h2": och2,
        "Omega_m": float(
            c.omega_m
        ),
        "w_X": w_x,
        "r_s_Mpc": rs,
        "r_s_h_Mpc": rs * h,
    }


def log_prior(theta):
    h, obh2, och2, w_x = theta

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

    if not (
        W_X_BOUNDS[0]
        <= w_x
        <= W_X_BOUNDS[1]
    ):
        return -np.inf

    d = derived(theta)

    if not (
        0.0
        < d["Omega_m"]
        < 1.0
    ):
        return -np.inf

    if not (
        np.isfinite(
            d["r_s_Mpc"]
        )
        and d[
            "r_s_Mpc"
        ] > 0.0
    ):
        return -np.inf

    return 0.0


def chi2_components(
    theta,
    cc_x,
    bao_x,
):
    if not np.isfinite(
        log_prior(theta)
    ):
        return None

    d = derived(theta)

    cc = cc_x.chi2(
        d["H0_km_s_Mpc"],
        d["Omega_m"],
        d["w_X"],
    )

    bao = bao_x.chi2_components(
        d["Omega_m"],
        d["w_X"],
        d["r_s_h_Mpc"],
    )

    return {
        "CC": float(cc),
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
    cc_x,
    bao_x,
):
    pieces = chi2_components(
        theta,
        cc_x,
        bao_x,
    )

    if pieces is None:
        return np.inf

    return float(
        sum(
            pieces.values()
        )
    )


def log_probability(
    theta,
    cc_x,
    bao_x,
):
    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    value = chi2(
        theta,
        cc_x,
        bao_x,
    )

    if not np.isfinite(value):
        return -np.inf

    return (
        lp
        - 0.5 * value
    )


def _finite_objective(
    theta,
    cc_x,
    bao_x,
):
    value = chi2(
        theta,
        cc_x,
        bao_x,
    )

    if np.isfinite(value):
        return float(value)

    return 1.0e30


def optimize_cc_bao_xcdm(
    cc_x,
    bao_x,
):
    bounds = [
        H_BOUNDS,
        OBH2_BOUNDS,
        OCH2_BOUNDS,
        W_X_BOUNDS,
    ]

    objective = lambda x: _finite_objective(
        x,
        cc_x,
        bao_x,
    )

    de = differential_evolution(
        objective,
        bounds=bounds,
        seed=20261017,
        popsize=18,
        maxiter=700,
        tol=1e-8,
        atol=1e-8,
        polish=False,
        workers=1,
        updating="immediate",
    )

    local = minimize(
        objective,
        x0=np.asarray(
            de.x,
            dtype=float,
        ),
        method="Powell",
        bounds=bounds,
        options={
            "xtol": 1e-10,
            "ftol": 1e-11,
            "maxiter": 8000,
        },
    )

    candidates = [
        de,
        local,
    ]

    best = min(
        candidates,
        key=lambda r: float(
            r.fun
        ),
    )

    exact = chi2(
        best.x,
        cc_x,
        bao_x,
    )

    if not np.isfinite(
        exact
    ):
        raise RuntimeError(
            "XCDM optimizer returned an invalid point."
        )

    best.fun = float(
        exact
    )

    return best


def initialize_walkers(
    best,
    nwalkers=64,
    seed=20261017,
):
    rng = np.random.default_rng(
        seed
    )

    best = np.asarray(
        best,
        dtype=float,
    )

    scales = np.array([
        0.020,
        0.005,
        0.012,
        0.08,
    ])

    walkers = []

    while (
        len(walkers)
        < nwalkers
    ):
        trial = (
            best
            + rng.normal(
                scale=scales
            )
        )

        if np.isfinite(
            log_prior(trial)
        ):
            walkers.append(
                trial
            )

    return np.asarray(
        walkers,
        dtype=float,
    )


def run_chain(
    cc_x,
    bao_x,
    backend_path,
    nwalkers=64,
    nsteps=20000,
    seed=20261017,
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

    opt = optimize_cc_bao_xcdm(
        cc_x,
        bao_x,
    )

    p0 = initialize_walkers(
        opt.x,
        nwalkers=nwalkers,
        seed=seed,
    )

    backend = emcee.backends.HDFBackend(
        str(
            backend_path
        )
    )

    if reset:
        backend.reset(
            nwalkers,
            4,
        )

    sampler = emcee.EnsembleSampler(
        nwalkers,
        4,
        log_probability,
        args=(
            cc_x,
            bao_x,
        ),
        backend=backend,
    )

    sampler.run_mcmc(
        p0,
        nsteps,
        progress=progress,
    )

    return (
        sampler,
        opt,
    )


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
        "tau_h": float(
            tau[0]
        ),
        "tau_omega_b_h2": float(
            tau[1]
        ),
        "tau_omega_c_h2": float(
            tau[2]
        ),
        "tau_w_X": float(
            tau[3]
        ),
        "steps_over_tau_h": float(
            ratio[0]
        ),
        "steps_over_tau_omega_b_h2": float(
            ratio[1]
        ),
        "steps_over_tau_omega_c_h2": float(
            ratio[2]
        ),
        "steps_over_tau_w_X": float(
            ratio[3]
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
            5.0
            * np.max(tau)
        )
    )

    if (
        burn
        >= sampler.iteration
    ):
        raise RuntimeError(
            "Autocorrelation burn-in exceeds chain length."
        )

    return (
        sampler.get_chain(
            discard=burn,
            flat=True,
        ),
        burn,
    )


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
            q[2]
            - q[1]
        ),
        "plus_68": float(
            q[3]
            - q[2]
        ),
        "minus_95": float(
            q[2]
            - q[0]
        ),
        "plus_95": float(
            q[4]
            - q[2]
        ),
    }


def derived_arrays(samples):
    h = samples[:, 0]
    obh2 = samples[:, 1]
    och2 = samples[:, 2]
    w_x = samples[:, 3]

    n = len(samples)

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
            h=float(
                h[i]
            ),
            omega_b_h2=float(
                obh2[i]
            ),
            omega_c_h2=float(
                och2[i]
            ),
        )

        omega_m[i] = (
            c.omega_m
        )
        rs[i] = (
            c.sound_horizon_aubourg()
        )

    H0 = 100.0 * h
    rsh = rs * h

    return {
        "h": h,
        "H0_km_s_Mpc": H0,
        "omega_b_h2": obh2,
        "omega_c_h2": och2,
        "Omega_m": omega_m,
        "w_X": w_x,
        "r_s_Mpc": rs,
        "r_s_h_Mpc": rsh,
    }


def write_summary(
    path,
    cc_x,
    bao_x,
    sampler,
    opt,
    project_root=None,
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

    best = derived(
        opt.x
    )

    parts = chi2_components(
        opt.x,
        cc_x,
        bao_x,
    )

    chi2_min = float(
        sum(
            parts.values()
        )
    )

    n_data = (
        cc_x.n_data
        + bao_x.n_data
    )
    k = 4
    loglike_max = (
        -0.5 * chi2_min
    )

    comparison = None

    if project_root is not None:
        flat_path = (
            Path(project_root)
            / "results"
            / "cc_bao_flat_lcdm"
            / "summary.json"
        )

        if flat_path.exists():
            flat = json.loads(
                flat_path.read_text()
            )

            comparison = {
                "LambdaCDM": {
                    "chi2_min": float(
                        flat[
                            "best_fit"
                        ][
                            "chi2_min"
                        ]
                    ),
                    "AIC": float(
                        flat[
                            "fit_quality"
                        ][
                            "AIC"
                        ]
                    ),
                    "BIC": float(
                        flat[
                            "fit_quality"
                        ][
                            "BIC"
                        ]
                    ),
                }
            }

    x_aic = information_criteria(
        loglike_max,
        k,
        n_data,
    )

    fit_quality = {
        "n_data": int(
            n_data
        ),
        "n_parameters": int(
            k
        ),
        "dof": int(
            n_data - k
        ),
        "reduced_chi2": reduced_chi2(
            chi2_min,
            n_data,
            k,
        )[0],
        **x_aic,
    }

    if comparison is not None:
        comparison[
            "XCDM"
        ] = {
            "chi2_min": chi2_min,
            "AIC": float(
                fit_quality[
                    "AIC"
                ]
            ),
            "BIC": float(
                fit_quality[
                    "BIC"
                ]
            ),
        }

        comparison[
            "delta_XCDM_minus_LambdaCDM"
        ] = {
            "delta_chi2": float(
                chi2_min
                - comparison[
                    "LambdaCDM"
                ][
                    "chi2_min"
                ]
            ),
            "delta_AIC": float(
                fit_quality[
                    "AIC"
                ]
                - comparison[
                    "LambdaCDM"
                ][
                    "AIC"
                ]
            ),
            "delta_BIC": float(
                fit_quality[
                    "BIC"
                ]
                - comparison[
                    "LambdaCDM"
                ][
                    "BIC"
                ]
            ),
        }

    result = {
        "dataset": "CC32_plus_BAO11",
        "model": "flat_XCDM",
        "sampled_parameters": [
            "h",
            "omega_b_h2",
            "omega_c_h2",
            "w_X",
        ],
        "priors": {
            "h": list(
                H_BOUNDS
            ),
            "omega_b_h2": list(
                OBH2_BOUNDS
            ),
            "omega_c_h2": list(
                OCH2_BOUNDS
            ),
            "w_X": list(
                W_X_BOUNDS
            ),
        },
        "best_fit": {
            **best,
            "chi2_min": chi2_min,
            "chi2_components": parts,
        },
        "posterior": posterior,
        "mcmc": {
            "nwalkers": int(
                sampler.nwalkers
            ),
            "nsteps": int(
                sampler.iteration
            ),
            "burn_in": int(
                burn
            ),
            **autocorrelation_report(
                sampler
            ),
        },
        "fit_quality": fit_quality,
        "model_comparison": comparison,
        "literature_benchmark": {
            "source": (
                "Cao & Ratra (2023), updated H(z)+BAO, flat XCDM"
            ),
            "H0_km_s_Mpc": {
                "central": 67.18,
                "sigma": 3.18,
            },
            "Omega_m": {
                "central": 0.285,
                "sigma": 0.019,
            },
            "w_X": {
                "central": -0.776,
                "minus_68": 0.103,
                "plus_68": 0.130,
            },
            "note": (
                "Benchmark uses a related but not identical H(z) treatment; "
                "the present CC likelihood includes the correlated Moresco covariance."
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
):
    return (
        rf"${label} = "
        rf"{stats['median']:.{digits}f}"
        rf"^{{+{stats['plus_68']:.{digits}f}}}"
        rf"_{{-{stats['minus_68']:.{digits}f}}}$"
    )


def plot_main_triangle(
    pdf_path,
    png_path,
    arr,
    summary,
):
    from getdist import (
        MCSamples,
        plots,
    )

    matrix = np.column_stack([
        arr[
            "H0_km_s_Mpc"
        ],
        arr[
            "Omega_m"
        ],
        arr[
            "w_X"
        ],
    ])

    gd = MCSamples(
        samples=matrix,
        names=[
            "H0",
            "Omega_m",
            "wX",
        ],
        labels=[
            r"H_0",
            r"\Omega_m",
            r"w_X",
        ],
    )

    g = plots.get_subplot_plotter(
        width_inch=7.4
    )

    g.triangle_plot(
        [gd],
        filled=True,
    )

    post = summary[
        "posterior"
    ]

    g.subplots[
        0,
        0,
    ].set_title(
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

    g.subplots[
        1,
        1,
    ].set_title(
        _title(
            r"\Omega_m",
            post[
                "Omega_m"
            ],
            digits=3,
        ),
        fontsize=11,
        pad=8,
    )

    g.subplots[
        2,
        2,
    ].set_title(
        _title(
            r"w_X",
            post[
                "w_X"
            ],
            digits=3,
        ),
        fontsize=11,
        pad=8,
    )

    # LambdaCDM reference w_X=-1.
    g.subplots[
        2,
        2,
    ].axvline(
        -1.0,
        linestyle="--",
        linewidth=1.0,
    )

    g.subplots[
        2,
        0,
    ].axhline(
        -1.0,
        linestyle="--",
        linewidth=1.0,
    )

    g.subplots[
        2,
        1,
    ].axhline(
        -1.0,
        linestyle="--",
        linewidth=1.0,
    )

    g.export(
        str(
            pdf_path
        )
    )
    g.export(
        str(
            png_path
        )
    )


def plot_ruler_triangle(
    pdf_path,
    png_path,
    arr,
    summary,
):
    from getdist import (
        MCSamples,
        plots,
    )

    matrix = np.column_stack([
        arr[
            "H0_km_s_Mpc"
        ],
        arr[
            "r_s_Mpc"
        ],
        arr[
            "w_X"
        ],
    ])

    gd = MCSamples(
        samples=matrix,
        names=[
            "H0",
            "rs",
            "wX",
        ],
        labels=[
            r"H_0",
            r"r_s\ [\mathrm{Mpc}]",
            r"w_X",
        ],
    )

    g = plots.get_subplot_plotter(
        width_inch=7.4
    )

    g.triangle_plot(
        [gd],
        filled=True,
    )

    g.subplots[
        2,
        2,
    ].axvline(
        -1.0,
        linestyle="--",
        linewidth=1.0,
    )

    g.export(
        str(
            pdf_path
        )
    )
    g.export(
        str(
            png_path
        )
    )
