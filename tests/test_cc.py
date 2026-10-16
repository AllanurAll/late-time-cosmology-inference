from pathlib import Path
import numpy as np
from late_time_cosmology.cc import load_cc32, build_cc_covariance

DATA = Path(__file__).parents[1] / "data" / "cc"


def test_cc_data_vector_matches_published_32_point_compilation():
    df, _ = load_cc32(DATA)
    assert len(df) == 32

    checks = {
        0.1791: (78.0, 6.2),
        0.3519: (85.5, 15.7),
        0.3802: (86.2, 14.6),
        0.7812: (96.5, 12.5),
        1.0370: (133.5, 17.6),
        1.9650: (186.5, 50.6),
    }

    for z, (H_expected, s_expected) in checks.items():
        row = df[np.isclose(df["z"], z)].iloc[0]
        assert np.isclose(row["H"], H_expected)
        assert np.isclose(row["sigma_diag"], s_expected)


def test_cc_covariance():
    df, sys = load_cc32(DATA)
    cov = build_cc_covariance(df, sys)
    assert cov.shape == (32, 32)
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0.0
    assert int(df["moresco_correlated"].sum()) == 15


def test_cc_has_off_diagonal_systematics():
    df, sys = load_cc32(DATA)
    cov = build_cc_covariance(df, sys)
    idx = np.where(df["moresco_correlated"].to_numpy(int) == 1)[0]
    block = cov[np.ix_(idx, idx)]
    offdiag = block - np.diag(np.diag(block))
    assert np.max(np.abs(offdiag)) > 0.0
