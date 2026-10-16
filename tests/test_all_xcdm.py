import numpy as np

from late_time_cosmology.cc_bao_xcdm_fit import (
    derived,
)


def test_all_xcdm_derived_parameters_plausible():
    d = derived([
        0.73,
        0.029,
        0.130,
        -0.9,
    ])

    assert np.isclose(
        d[
            "H0_km_s_Mpc"
        ],
        73.0,
    )

    assert (
        0.20
        < d[
            "Omega_m"
        ]
        < 0.40
    )

    assert (
        120.0
        < d[
            "r_s_Mpc"
        ]
        < 160.0
    )

    assert np.isclose(
        d[
            "w_X"
        ],
        -0.9,
    )


def test_all_xcdm_rsh_identity():
    d = derived([
        0.72,
        0.027,
        0.125,
        -1.0,
    ])

    assert np.isclose(
        d[
            "r_s_h_Mpc"
        ],
        d[
            "r_s_Mpc"
        ]
        * d["h"],
        rtol=0,
        atol=1e-13,
    )
