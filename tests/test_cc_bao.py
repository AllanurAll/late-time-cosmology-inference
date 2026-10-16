from pathlib import Path
import numpy as np

from late_time_cosmology.cc import CCLikelihood
from late_time_cosmology.bao_reduced import BAOReducedFlatLCDM
from late_time_cosmology.cc_bao_fit import (
    derived,
    chi2,
    chi2_components,
)

ROOT = Path(__file__).parents[1]

CC = CCLikelihood(
    ROOT / "data" / "cc"
)

BAO = BAOReducedFlatLCDM(
    ROOT / "data" / "bao"
)


def test_cc_bao_reference_point():
    theta = np.array([
        0.7060755,
        0.0261885,
        0.1204187,
    ])

    value = chi2(
        theta,
        CC,
        BAO,
    )

    assert np.isclose(
        value,
        20.67170861,
        atol=3e-6,
    )


def test_cc_bao_components_sum_to_total():
    theta = np.array([
        0.7060755,
        0.0261885,
        0.1204187,
    ])

    parts = chi2_components(
        theta,
        CC,
        BAO,
    )

    total = chi2(
        theta,
        CC,
        BAO,
    )

    assert np.isclose(
        sum(parts.values()),
        total,
        rtol=0,
        atol=1e-12,
    )


def test_joint_reference_is_physically_plausible():
    _, d = derived([
        0.7060755,
        0.0261885,
        0.1204187,
    ])

    assert 65.0 < d["H0_km_s_Mpc"] < 75.0
    assert 0.25 < d["Omega_m"] < 0.35
    assert 130.0 < d["r_s_Mpc"] < 155.0
