"""Final flat-LambdaCDM versus flat-XCDM comparison utilities."""

from pathlib import Path
import csv
import json

import numpy as np
from scipy.stats import chi2 as chi2_dist


def interval_text(stats, digits=2):
    return (
        f"{stats['median']:.{digits}f}"
        f" +{stats['plus_68']:.{digits}f}"
        f" -{stats['minus_68']:.{digits}f}"
    )


def load_summary(path):
    return json.loads(
        Path(path).read_text()
    )


def build_comparison(lcdm, xcdm):
    lfit = lcdm["fit_quality"]
    xfit = xcdm["fit_quality"]

    delta_chi2 = (
        xcdm["best_fit"]["chi2_min"]
        - lcdm["best_fit"]["chi2_min"]
    )
    delta_aic = (
        xfit["AIC"]
        - lfit["AIC"]
    )
    delta_bic = (
        xfit["BIC"]
        - lfit["BIC"]
    )

    improvement = max(
        0.0,
        -float(delta_chi2),
    )

    # XCDM is nested in LambdaCDM at w_X=-1 and adds one parameter.
    # This is an approximate Wilks-theorem diagnostic, not a Bayes factor.
    p_lrt = float(
        chi2_dist.sf(
            improvement,
            df=1,
        )
    )

    sigma_equiv = float(
        np.sqrt(
            improvement
        )
    )

    return {
        "LambdaCDM": {
            "H0": lcdm["posterior"]["H0_km_s_Mpc"],
            "Omega_m": lcdm["posterior"]["Omega_m"],
            "r_s": lcdm["posterior"]["r_s_Mpc"],
            "chi2_min": float(
                lcdm["best_fit"]["chi2_min"]
            ),
            "AIC": float(
                lfit["AIC"]
            ),
            "BIC": float(
                lfit["BIC"]
            ),
            "k_effective": int(
                lfit["n_parameters_effective"]
            ),
        },
        "XCDM": {
            "H0": xcdm["posterior"]["H0_km_s_Mpc"],
            "Omega_m": xcdm["posterior"]["Omega_m"],
            "r_s": xcdm["posterior"]["r_s_Mpc"],
            "w_X": xcdm["posterior"]["w_X"],
            "chi2_min": float(
                xcdm["best_fit"]["chi2_min"]
            ),
            "AIC": float(
                xfit["AIC"]
            ),
            "BIC": float(
                xfit["BIC"]
            ),
            "k_effective": int(
                xfit["n_parameters_effective"]
            ),
        },
        "delta_XCDM_minus_LambdaCDM": {
            "delta_chi2": float(
                delta_chi2
            ),
            "delta_AIC": float(
                delta_aic
            ),
            "delta_BIC": float(
                delta_bic
            ),
        },
        "nested_likelihood_ratio": {
            "delta_chi2_improvement": float(
                improvement
            ),
            "df": 1,
            "p_value_approx": p_lrt,
            "sqrt_delta_chi2_sigma_approx": sigma_equiv,
            "note": (
                "Approximate nested-model likelihood-ratio diagnostic. "
                "It is not a Bayes factor and does not replace AIC/BIC "
                "or a dedicated evidence calculation."
            ),
        },
    }


def write_csv(path, comparison):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [
        [
            "model",
            "H0_median",
            "H0_minus68",
            "H0_plus68",
            "Omega_m_median",
            "Omega_m_minus68",
            "Omega_m_plus68",
            "r_s_median_Mpc",
            "r_s_minus68_Mpc",
            "r_s_plus68_Mpc",
            "w_X_median",
            "w_X_minus68",
            "w_X_plus68",
            "chi2_min",
            "AIC",
            "BIC",
        ]
    ]

    for model in (
        "LambdaCDM",
        "XCDM",
    ):
        d = comparison[model]
        wx = d.get("w_X")

        rows.append([
            model,
            d["H0"]["median"],
            d["H0"]["minus_68"],
            d["H0"]["plus_68"],
            d["Omega_m"]["median"],
            d["Omega_m"]["minus_68"],
            d["Omega_m"]["plus_68"],
            d["r_s"]["median"],
            d["r_s"]["minus_68"],
            d["r_s"]["plus_68"],
            "" if wx is None else wx["median"],
            "" if wx is None else wx["minus_68"],
            "" if wx is None else wx["plus_68"],
            d["chi2_min"],
            d["AIC"],
            d["BIC"],
        ])

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def write_markdown(path, comparison):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    l = comparison["LambdaCDM"]
    x = comparison["XCDM"]
    d = comparison[
        "delta_XCDM_minus_LambdaCDM"
    ]
    lr = comparison[
        "nested_likelihood_ratio"
    ]

    wx = x["w_X"]

    text = f"""# Flat LambdaCDM and XCDM comparison

The same calibrated late-time data combination is used in both models:

- Pantheon+SH0ES with the full released STAT+SYS covariance
- 32 cosmic-chronometer measurements with the correlated Moresco covariance
- 11 BAO measurements, including the BOSS DR12 and Ly-alpha covariance blocks

The quoted parameter uncertainties are marginalized 68% credible intervals.

| Model | H0 [km s^-1 Mpc^-1] | Omega_m | r_s [Mpc] | w_X | chi2_min | AIC | BIC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat LambdaCDM | {interval_text(l["H0"], 2)} | {interval_text(l["Omega_m"], 3)} | {interval_text(l["r_s"], 2)} | -1 (fixed) | {l["chi2_min"]:.3f} | {l["AIC"]:.3f} | {l["BIC"]:.3f} |
| flat XCDM | {interval_text(x["H0"], 2)} | {interval_text(x["Omega_m"], 3)} | {interval_text(x["r_s"], 2)} | {interval_text(wx, 3)} | {x["chi2_min"]:.3f} | {x["AIC"]:.3f} | {x["BIC"]:.3f} |

For XCDM relative to LambdaCDM,

- Delta chi2 = {d["delta_chi2"]:+.3f}
- Delta AIC = {d["delta_AIC"]:+.3f}
- Delta BIC = {d["delta_BIC"]:+.3f}

The extra XCDM parameter lowers the minimum chi-square by {-d["delta_chi2"]:.3f}. The corresponding one-parameter nested likelihood-ratio diagnostic is approximately sqrt(Delta chi2) = {lr["sqrt_delta_chi2_sigma_approx"]:.2f}, with p approximately {lr["p_value_approx"]:.3f}. This is only an approximate frequentist diagnostic.

The information criteria give a deliberately mixed conclusion. AIC mildly favors XCDM because the improvement in fit is large enough to offset its one extra parameter, while BIC mildly favors the simpler LambdaCDM model. The result is therefore best described as a mild preference for w_X greater than -1, not decisive evidence for dynamical dark energy.

The XCDM marginalized result is

w_X = {wx["median"]:.3f} +{wx["plus_68"]:.3f} -{wx["minus_68"]:.3f}  (68%)

with a central 95% interval

{wx["q2.5"]:.3f} < w_X < {wx["q97.5"]:.3f}.

The LambdaCDM value w_X=-1 lies just outside this central 95% interval in the present combined-data analysis. This statement is specific to the likelihoods, calibrations, priors, and data compilation used in this repository.
"""

    path.write_text(text)


def write_json(path, comparison):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            comparison,
            indent=2,
        )
    )
