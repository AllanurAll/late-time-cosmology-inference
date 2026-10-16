import numpy as np
from scipy.integrate import quad

from late_time_cosmology.pantheon_flat import (
    E_flat_lcdm,
    chi_dimensionless,
)


def test_pantheon_quadrature_against_quad():
    omega_m = 0.334

    z = np.array([
        0.0102,
        0.1,
        0.5,
        1.0,
        2.26,
    ])

    fast = chi_dimensionless(
        z,
        omega_m,
    )

    slow = np.array([
        quad(
            lambda zp: 1.0 / E_flat_lcdm(
                zp,
                omega_m,
            ),
            0.0,
            float(zz),
            epsabs=1e-12,
            epsrel=1e-12,
        )[0]
        for zz in z
    ])

    assert np.allclose(
        fast,
        slow,
        rtol=1e-11,
        atol=1e-12,
    )


def test_analytic_offset_marginalization_identity():
    # Small synthetic covariance/data example: compare the analytic
    # A-B^2/C expression against explicit minimization over the offset.
    C = np.array([
        [0.04, 0.01, 0.00],
        [0.01, 0.05, 0.01],
        [0.00, 0.01, 0.06],
    ])

    inv = np.linalg.inv(C)
    one = np.ones(3)

    d = np.array([
        0.30,
        0.10,
        -0.20,
    ])

    A = d @ inv @ d
    B = one @ inv @ d
    D = one @ inv @ one

    chi_analytic = A - B*B/D

    offset = B/D
    r = d - offset*one
    chi_explicit = r @ inv @ r

    assert np.isclose(
        chi_analytic,
        chi_explicit,
        atol=1e-13,
    )
