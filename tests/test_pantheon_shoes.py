import numpy as np


def test_profiled_MB_identity():
    # Synthetic covariance check for
    # r = a + x*q - M*1.
    C = np.array([
        [0.04, 0.01, 0.00, 0.00],
        [0.01, 0.05, 0.01, 0.00],
        [0.00, 0.01, 0.05, 0.01],
        [0.00, 0.00, 0.01, 0.04],
    ])

    inv = np.linalg.inv(C)

    a = np.array([
        0.10,
        -0.05,
        0.20,
        0.05,
    ])

    q = np.array([
        0.0,
        0.0,
        1.0,
        1.0,
    ])

    one = np.ones(4)
    x = 0.17

    r0 = a + x*q

    MB = (
        one @ inv @ r0
        / (one @ inv @ one)
    )

    r = r0 - MB*one

    explicit = r @ inv @ r

    profiled = (
        r0 @ inv @ r0
        - (
            one @ inv @ r0
        )**2
        / (one @ inv @ one)
    )

    assert np.isclose(
        explicit,
        profiled,
        atol=1e-13,
    )


def test_H0_shift_is_five_log10():
    # In the Hubble-flow magnitude model,
    # changing H0 from H1 to H2 changes mu by
    # -5 log10(H2/H1).
    H1 = 70.0
    H2 = 74.0

    shift = (
        -5.0
        * np.log10(H2/H1)
    )

    mu1 = (
        5.0
        * np.log10(1000.0/H1)
    )

    mu2 = (
        5.0
        * np.log10(1000.0/H2)
    )

    assert np.isclose(
        mu2 - mu1,
        shift,
        atol=1e-14,
    )
