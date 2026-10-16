import numpy as np

from late_time_cosmology.all_flat_fit import (
    derived,
)


def test_all_flat_derived_parameters_plausible():
    _, d = derived([
        0.73,
        0.027,
        0.121,
    ])

    assert np.isclose(
        d["H0_km_s_Mpc"],
        73.0,
    )

    assert (
        0.20
        < d["Omega_m"]
        < 0.40
    )

    assert (
        120.0
        < d["r_s_Mpc"]
        < 160.0
    )


def test_all_flat_rsh_identity():
    _, d = derived([
        0.72,
        0.026,
        0.120,
    ])

    assert np.isclose(
        d["r_s_h_Mpc"],
        d["r_s_Mpc"] * d["h"],
        rtol=0,
        atol=1e-13,
    )
