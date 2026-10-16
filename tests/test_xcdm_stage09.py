from pathlib import Path

import numpy as np

from late_time_cosmology.bao_reduced import (
    BAOReducedFlatLCDM,
    E_flat_lcdm,
    chi_dimensionless,
)
from late_time_cosmology.bao_xcdm import (
    BAOFlatXCDM,
)
from late_time_cosmology.xcdm_background import (
    E_flat_xcdm,
    chi_dimensionless_xcdm,
)

DATA = (
    Path(__file__).parents[1]
    / "data"
    / "bao"
)


def test_xcdm_background_reduces_to_lcdm_at_wminus1():
    z = np.array([
        0.0,
        0.1,
        0.5,
        1.0,
        2.3,
    ])

    om = 0.3

    assert np.allclose(
        E_flat_xcdm(
            z,
            om,
            -1.0,
        ),
        E_flat_lcdm(
            z,
            om,
        ),
        rtol=0,
        atol=1e-14,
    )


def test_xcdm_distance_reduces_to_lcdm_at_wminus1():
    z = np.array([
        0.122,
        0.38,
        0.81,
        1.52,
        2.334,
    ])

    om = 0.3

    assert np.allclose(
        chi_dimensionless_xcdm(
            z,
            om,
            -1.0,
        ),
        chi_dimensionless(
            z,
            om,
        ),
        rtol=0,
        atol=1e-13,
    )


def test_xcdm_bao_reduces_to_lcdm_at_wminus1():
    lcdm = BAOReducedFlatLCDM(
        DATA
    )

    xcdm = BAOFlatXCDM(
        DATA
    )

    om = 0.297
    rsh = 100.9

    tv_l = lcdm.theory_vector(
        om,
        rsh,
    )

    tv_x = xcdm.theory_vector(
        om,
        -1.0,
        rsh,
    )

    assert np.allclose(
        tv_x,
        tv_l,
        rtol=0,
        atol=2e-12,
    )

    assert np.isclose(
        xcdm.chi2(
            om,
            -1.0,
            rsh,
        ),
        lcdm.chi2(
            om,
            rsh,
        ),
        rtol=0,
        atol=1e-10,
    )
