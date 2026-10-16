from pathlib import Path
import numpy as np
from scipy.integrate import quad

from late_time_cosmology.bao_reduced import (
    BAOReducedFlatLCDM,
    E_flat_lcdm,
    chi_dimensionless,
)

DATA = Path(__file__).parents[1] / "data" / "bao"


def test_gauss_legendre_distance_against_quad():
    omega_m = 0.30
    z = np.array([0.122, 0.38, 0.81, 1.52, 2.334])
    fast = chi_dimensionless(z, omega_m)
    slow = np.array([
        quad(
            lambda zp: 1.0 / E_flat_lcdm(zp, omega_m),
            0.0,
            float(zz),
            epsabs=1e-12,
            epsrel=1e-12,
        )[0]
        for zz in z
    ])
    assert np.allclose(fast, slow, rtol=1e-11, atol=1e-12)


def test_bao_covariances_positive_definite():
    like = BAOReducedFlatLCDM(DATA)
    assert like.boss_cov.shape == (6, 6)
    assert like.lya_cov.shape == (2, 2)
    assert np.linalg.eigvalsh(like.boss_cov).min() > 0.0
    assert np.linalg.eigvalsh(like.lya_cov).min() > 0.0


def test_bao_best_fit_reference_point():
    # Deterministic regression check for the Stage-03 BAO compilation.
    like = BAOReducedFlatLCDM(DATA)
    chi2 = like.chi2(0.29670113, 100.92526745)
    assert np.isclose(chi2, 8.9958801731, rtol=0, atol=2e-7)
