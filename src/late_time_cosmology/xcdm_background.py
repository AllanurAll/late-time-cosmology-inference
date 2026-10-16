"""Spatially-flat XCDM background functions.

XCDM treats dark energy as a homogeneous component with constant equation
of state w_X.  In a flat late-time background,

    E(z)^2 = Omega_m (1+z)^3
           + (1-Omega_m) (1+z)^[3(1+w_X)].

The LambdaCDM limit is exactly w_X = -1.
"""

import numpy as np

_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def E_flat_xcdm(z, omega_m, w_x):
    z = np.asarray(z, dtype=float)
    return np.sqrt(
        omega_m * (1.0 + z) ** 3
        + (1.0 - omega_m)
        * (1.0 + z) ** (3.0 * (1.0 + w_x))
    )


def chi_dimensionless_xcdm(z, omega_m, w_x):
    """Return integral_0^z dz'/E(z') using 64-point Gauss-Legendre."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    x = 0.5 * z[:, None] * (_GL_X[None, :] + 1.0)
    integrand = 1.0 / E_flat_xcdm(
        x,
        omega_m,
        w_x,
    )
    return (
        0.5
        * z
        * np.sum(
            _GL_W[None, :] * integrand,
            axis=1,
        )
    )


def H_flat_xcdm(z, H0, omega_m, w_x):
    return (
        float(H0)
        * E_flat_xcdm(
            z,
            omega_m,
            w_x,
        )
    )
