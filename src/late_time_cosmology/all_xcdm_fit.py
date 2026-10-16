"""Pantheon+SH0ES + CC + BAO inference for spatially-flat XCDM."""

import json
from pathlib import Path

import emcee
import numpy as np
from scipy.optimize import (
    differential_evolution,
    minimize,
)

from .cc_bao_xcdm_fit import (
    CCFlatXCDM,
    H_BOUNDS,
    OBH2_BOUNDS,
    OCH2_BOUNDS,
    W_X_BOUNDS,
    derived,
)
from .statistics import (
    information_criteria,
    reduced_chi2,
)


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

    d = derived(
        theta
    )

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
    sn_x,
):
    if not np.isfinite(
        log_prior(theta)
    ):
        return None

    d = derived(
        theta
    )

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

    sn = sn_x.chi2(
        d["H0_km_s_Mpc"],
        d["Omega_m"],
        d["w_X"],
    )

    return {
        "Pantheon_plus_SH0ES": float(
            sn
        ),
        "CC": float(
            cc
        ),
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
    sn_x,
):
    pieces = chi2_components(
        theta,
        cc_x,
        bao_x,
        sn_x,
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
    sn_x,
):
    lp = log_prior(
        theta
    )

    if not np.isfinite(
        lp
    ):
        return -np.inf

    value = chi2(
        theta,
        cc_x,
        bao_x,
        sn_x,
    )

    if not np.isfinite(
        value
    ):
        return -np.inf

    return (
        lp
        - 0.5 * value
    )


def _finite_objective(
    theta,
    cc_x,
    bao_x,
    sn_x,
):
    value = chi2(
        theta,
        cc_x,
        bao_x,
        sn_x,
    )

    if np.isfinite(
        value
    ):
        return float(
            value
        )

    return 1.0e30


def optimize_all_xcdm(
    cc_x,
    bao_x,
    sn_x,
):
    # The scientific prior remains the broad Stage-09 prior.
    # The global optimizer is restricted to a deliberately generous
    # high-likelihood box so it does not spend thousands of expensive
    # evaluations in the extreme prior tails.  The MCMC itself retains
    # the complete prior and exact fallback outside the spline cache.
    bounds = [
        (0.55, 0.85),
        (0.010, 0.060),
        (0.060, 0.220),
        (-1.60, -0.30),
    ]

    objective = lambda x: _finite_objective(
        x,
        cc_x,
        bao_x,
        sn_x,
    )

    de = differential_evolution(
        objective,
        bounds=bounds,
        seed=20261018,
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

    best = min(
        [de, local],
        key=lambda r: float(
            r.fun
        ),
    )

    exact = chi2(
        best.x,
        cc_x,
        bao_x,
        sn_x,
    )

    if not np.isfinite(
        exact
    ):
        raise RuntimeError(
            "Full XCDM optimizer returned an invalid point."
        )

    best.fun = float(
        exact
    )

    return best


def initialize_walkers(
    best,
    nwalkers=72,
    seed=20261018,
):
    rng = np.random.default_rng(
        seed
    )

    best = np.asarray(
        best,
        dtype=float,
    )

    scales = np.array([
        0.010,
        0.0030,
        0.0060,
        0.055,
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
            log_prior(
                trial
            )
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
    sn_x,
    backend_path,
    nwalkers=72,
    nsteps=20000,
    seed=20261018,
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

    opt = optimize_all_xcdm(
        cc_x,
        bao_x,
        sn_x,
    )

    p0 = initialize_walkers(
        opt.x,
        nwalkers=nwalkers,
        seed=seed,
    )

    backend = (
        emcee.backends.HDFBackend(
            str(
                backend_path
            )
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
            sn_x,
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
            * np.max(
                tau
            )
        )
    )

    if burn >= sampler.iteration:
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
        "q2.5": float(
            q[0]
        ),
        "q16": float(
            q[1]
        ),
        "median": float(
            q[2]
        ),
        "q84": float(
            q[3]
        ),
        "q97.5": float(
            q[4]
        ),
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


def derived_arrays(
    samples
):
    # Reuse the exact Stage-09 derived mapping one sample at a time
    # for the sound horizon and Omega_m.
    values = [
        derived(row)
        for row in samples
    ]

    keys = [
        "h",
        "H0_km_s_Mpc",
        "omega_b_h2",
        "omega_c_h2",
        "Omega_m",
        "w_X",
        "r_s_Mpc",
        "r_s_h_Mpc",
    ]

    return {
        key: np.asarray(
            [
                d[key]
                for d in values
            ],
            dtype=float,
        )
        for key in keys
    }


def write_summary(
    path,
    cc_x,
    bao_x,
    sn_exact,
    sn_fast,
    sampler,
    opt,
    project_root,
):
    path = Path(
        path
    )
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
        key: summarize(
            value
        )
        for (
            key,
            value,
        ) in arr.items()
    }

    best = derived(
        opt.x
    )

    parts = chi2_components(
        opt.x,
        cc_x,
        bao_x,
        sn_fast,
    )

    chi2_min = float(
        sum(
            parts.values()
        )
    )

    MB_best = (
        sn_exact
        .best_MB_xcdm(
            best[
                "H0_km_s_Mpc"
            ],
            best[
                "Omega_m"
            ],
            best[
                "w_X"
            ],
        )
    )

    n_data = (
        sn_exact.n
        + cc_x.n_data
        + bao_x.n_data
    )

    # h, obh2, och2, w_X + profiled M_B
    k = 5
    loglike_max = (
        -0.5
        * chi2_min
    )

    fit_quality = {
        "n_data": int(
            n_data
        ),
        "n_cosmological_parameters": 4,
        "n_profiled_nuisance_parameters": 1,
        "n_parameters_effective": 5,
        "dof": int(
            n_data
            - k
        ),
        "reduced_chi2": reduced_chi2(
            chi2_min,
            n_data,
            k,
        )[0],
        **information_criteria(
            loglike_max,
            k,
            n_data,
        ),
    }

    root = Path(
        project_root
    )

    lcdm_path = (
        root
        / "results"
        / "all_flat_lcdm"
        / "summary.json"
    )

    comparison = None

    if lcdm_path.exists():
        lcdm = json.loads(
            lcdm_path.read_text()
        )

        comparison = {
            "LambdaCDM": {
                "chi2_min": float(
                    lcdm[
                        "best_fit"
                    ][
                        "chi2_min"
                    ]
                ),
                "AIC": float(
                    lcdm[
                        "fit_quality"
                    ][
                        "AIC"
                    ]
                ),
                "BIC": float(
                    lcdm[
                        "fit_quality"
                    ][
                        "BIC"
                    ]
                ),
            },
            "XCDM": {
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
            },
        }

        comparison[
            "delta_XCDM_minus_LambdaCDM"
        ] = {
            "delta_chi2": float(
                comparison[
                    "XCDM"
                ][
                    "chi2_min"
                ]
                - comparison[
                    "LambdaCDM"
                ][
                    "chi2_min"
                ]
            ),
            "delta_AIC": float(
                comparison[
                    "XCDM"
                ][
                    "AIC"
                ]
                - comparison[
                    "LambdaCDM"
                ][
                    "AIC"
                ]
            ),
            "delta_BIC": float(
                comparison[
                    "XCDM"
                ][
                    "BIC"
                ]
                - comparison[
                    "LambdaCDM"
                ][
                    "BIC"
                ]
            ),
        }

    cache_fraction = float(
        np.mean(
            (
                arr[
                    "Omega_m"
                ]
                >= sn_fast.omega_min
            )
            & (
                arr[
                    "Omega_m"
                ]
                <= sn_fast.omega_max
            )
            & (
                arr[
                    "w_X"
                ]
                >= sn_fast.w_min
            )
            & (
                arr[
                    "w_X"
                ]
                <= sn_fast.w_max
            )
        )
    )

    result = {
        "dataset": (
            "Pantheon+SH0ES_plus_CC32_plus_BAO11"
        ),
        "model": "flat_XCDM",
        "sampled_parameters": [
            "h",
            "omega_b_h2",
            "omega_c_h2",
            "w_X",
        ],
        "profiled_nuisance": [
            "M_B",
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
            "M_B": float(
                MB_best
            ),
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
        "pantheon_xcdm_cache": {
            "omega_range": [
                sn_fast.omega_min,
                sn_fast.omega_max,
            ],
            "w_X_range": [
                sn_fast.w_min,
                sn_fast.w_max,
            ],
            "n_omega": int(
                len(
                    sn_fast.omega_grid
                )
            ),
            "n_w_X": int(
                len(
                    sn_fast.w_grid
                )
            ),
            "posterior_sample_fraction_inside_cache": cache_fraction,
            "note": (
                "Outside the cache domain the exact Pantheon+SH0ES XCDM "
                "likelihood is evaluated, so the cache does not truncate "
                "the scientific prior."
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

    p = summary[
        "posterior"
    ]

    g.subplots[
        0,
        0,
    ].set_title(
        _title(
            "H_0",
            p[
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
            p[
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
            p[
                "w_X"
            ],
            digits=3,
        ),
        fontsize=11,
        pad=8,
    )

    # LambdaCDM reference.
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

    p = summary[
        "posterior"
    ]

    g.subplots[
        0,
        0,
    ].set_title(
        _title(
            "H_0",
            p[
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
            "r_s",
            p[
                "r_s_Mpc"
            ],
            digits=2,
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
            p[
                "w_X"
            ],
            digits=3,
        ),
        fontsize=11,
        pad=8,
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


def plot_model_comparison(
    path,
    xcdm_summary,
    project_root,
):
    """Compact LambdaCDM/XCDM comparison for the same all-data combination."""

    import matplotlib.pyplot as plt

    root = Path(
        project_root
    )

    lcdm_path = (
        root
        / "results"
        / "all_flat_lcdm"
        / "summary.json"
    )

    if not lcdm_path.exists():
        return

    lcdm = json.loads(
        lcdm_path.read_text()
    )

    xq = xcdm_summary[
        "posterior"
    ]

    lq = lcdm[
        "posterior"
    ]

    labels = [
        r"flat $\Lambda$CDM",
        "flat XCDM",
    ]

    centers = np.array([
        lq[
            "H0_km_s_Mpc"
        ][
            "median"
        ],
        xq[
            "H0_km_s_Mpc"
        ][
            "median"
        ],
    ])

    lower = np.array([
        lq[
            "H0_km_s_Mpc"
        ][
            "minus_68"
        ],
        xq[
            "H0_km_s_Mpc"
        ][
            "minus_68"
        ],
    ])

    upper = np.array([
        lq[
            "H0_km_s_Mpc"
        ][
            "plus_68"
        ],
        xq[
            "H0_km_s_Mpc"
        ][
            "plus_68"
        ],
    ])

    y = np.arange(
        2
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 3.8)
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
        "All-data 68% intervals"
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=250,
    )
    plt.close(
        fig
    )
