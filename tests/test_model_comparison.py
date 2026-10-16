from late_time_cosmology.model_comparison import (
    build_comparison,
)


def _summary(
    chi2,
    aic,
    bic,
    k,
    wx=None,
):
    posterior = {
        "H0_km_s_Mpc": {
            "median": 73.0,
            "minus_68": 1.0,
            "plus_68": 1.0,
        },
        "Omega_m": {
            "median": 0.30,
            "minus_68": 0.01,
            "plus_68": 0.01,
        },
        "r_s_Mpc": {
            "median": 137.0,
            "minus_68": 2.0,
            "plus_68": 2.0,
        },
    }

    if wx is not None:
        posterior["w_X"] = {
            "median": wx,
            "minus_68": 0.05,
            "plus_68": 0.05,
            "q2.5": wx - 0.10,
            "q97.5": wx + 0.10,
        }

    return {
        "best_fit": {
            "chi2_min": chi2,
        },
        "fit_quality": {
            "AIC": aic,
            "BIC": bic,
            "n_parameters_effective": k,
        },
        "posterior": posterior,
    }


def test_model_comparison_deltas():
    lcdm = _summary(
        100.0,
        108.0,
        120.0,
        4,
    )

    xcdm = _summary(
        95.0,
        105.0,
        122.0,
        5,
        wx=-0.9,
    )

    result = build_comparison(
        lcdm,
        xcdm,
    )

    d = result[
        "delta_XCDM_minus_LambdaCDM"
    ]

    assert d[
        "delta_chi2"
    ] == -5.0
    assert d[
        "delta_AIC"
    ] == -3.0
    assert d[
        "delta_BIC"
    ] == 2.0


def test_nested_one_parameter_sigma_is_sqrt_delta_chi2():
    lcdm = _summary(
        100.0,
        108.0,
        120.0,
        4,
    )

    xcdm = _summary(
        96.0,
        106.0,
        123.0,
        5,
        wx=-0.9,
    )

    result = build_comparison(
        lcdm,
        xcdm,
    )

    assert abs(
        result[
            "nested_likelihood_ratio"
        ][
            "sqrt_delta_chi2_sigma_approx"
        ]
        - 2.0
    ) < 1e-14
