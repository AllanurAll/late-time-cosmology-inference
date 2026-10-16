"""Fast Pantheon+ flat-LambdaCDM likelihood with analytic offset marginalization.

Two data selections are supported:

1. ``official``:
       zHD > 0.01
   This reproduces the selection in the public Pantheon+ CosmoSIS
   likelihood and retains 1590 released light-curve entries.

2. ``no_calibrators``:
       zHD > 0.01 and IS_CALIBRATOR == 0
   This removes the SH0ES calibrator-host rows and retains 1580 entries.
   It provides a useful cross-check against independent published
   re-analyses that quote chi2_min ~ 1387.10 for flat LambdaCDM.

The supernova absolute magnitude / distance-scale intercept is eliminated
analytically. Pantheon+ alone therefore constrains the *shape* parameter
Omega_m in flat LambdaCDM; it does not provide an independent H0
measurement without an external absolute calibration.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar

# Fixed Gauss-Legendre quadrature for the smooth flat-LCDM distance integral.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)

OMEGA_M_PRIOR = (0.05, 0.60)


def E_flat_lcdm(z, omega_m):
    z = np.asarray(z, dtype=float)
    return np.sqrt(
        omega_m * (1.0 + z) ** 3
        + (1.0 - omega_m)
    )


def chi_dimensionless(z, omega_m):
    """Integral_0^z dz'/E(z') using vectorized 64-point Gauss-Legendre."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    x = 0.5 * z[:, None] * (_GL_X[None, :] + 1.0)
    integrand = 1.0 / E_flat_lcdm(x, omega_m)
    return 0.5 * z * np.sum(
        _GL_W[None, :] * integrand,
        axis=1,
    )


def magnitude_shape(z_hd, z_hel, omega_m):
    """Pantheon+ magnitude shape with the additive intercept removed.

    The public Pantheon+ likelihood evaluates
        5 log10[(1+zHD)(1+zHEL) D_A(zHD)] + constant.
    Since D_A = D_M/(1+zHD), this is equivalent, up to the additive
    H0/M_B intercept, to
        5 log10[(1+zHEL) * integral dz/E(z)].
    """
    d = (1.0 + np.asarray(z_hel, float)) * chi_dimensionless(
        z_hd,
        omega_m,
    )
    if np.any(d <= 0.0):
        return np.full_like(d, np.nan)
    return 5.0 * np.log10(d)


def read_full_covariance(path, n_full):
    vals = np.loadtxt(path)

    if vals.size == n_full * n_full + 1 and int(vals[0]) == n_full:
        vals = vals[1:]
    elif vals.size == n_full * n_full:
        pass
    else:
        raise ValueError(
            f"Unexpected Pantheon+ covariance size {vals.size}; "
            f"expected {n_full*n_full} or {n_full*n_full+1}."
        )

    cov = vals.reshape(n_full, n_full)
    return cov


class PantheonFlatLCDM:
    def __init__(
        self,
        data_file,
        cov_file,
        selection="official",
    ):
        self.selection = selection

        data_file = Path(data_file)
        cov_file = Path(cov_file)

        data = pd.read_csv(
            data_file,
            sep=r"\s+",
        )

        n_full = len(data)
        cov_full = read_full_covariance(
            cov_file,
            n_full,
        )

        zmask = data["zHD"].to_numpy(float) > 0.01

        if selection == "official":
            mask = zmask
        elif selection == "no_calibrators":
            mask = (
                zmask
                & (
                    data["IS_CALIBRATOR"]
                    .to_numpy(int)
                    == 0
                )
            )
        else:
            raise ValueError(
                "selection must be 'official' or 'no_calibrators'"
            )

        self.mask = mask
        self.data = data.loc[mask].reset_index(drop=True)

        self.z_hd = self.data["zHD"].to_numpy(float)
        self.z_hel = self.data["zHEL"].to_numpy(float)
        self.m_obs = self.data["m_b_corr"].to_numpy(float)

        cov = cov_full[np.ix_(mask, mask)]

        # The text release can contain tiny decimal-rounding asymmetries.
        self.max_raw_asymmetry = float(
            np.max(np.abs(cov - cov.T))
        )

        self.cov = 0.5 * (cov + cov.T)

        self._cho = cho_factor(
            self.cov,
            lower=True,
            check_finite=False,
        )

        self._one = np.ones(self.n)
        self._cinv_one = cho_solve(
            self._cho,
            self._one,
            check_finite=False,
        )
        self._C = float(
            self._one @ self._cinv_one
        )

    @property
    def n(self):
        return len(self.m_obs)

    def _raw_residual(self, omega_m):
        theory = magnitude_shape(
            self.z_hd,
            self.z_hel,
            omega_m,
        )
        return self.m_obs - theory

    def best_intercept(self, omega_m):
        d = self._raw_residual(omega_m)
        cinv_d = cho_solve(
            self._cho,
            d,
            check_finite=False,
        )
        return float(
            self._one @ cinv_d / self._C
        )

    def chi2(self, omega_m):
        if not (
            OMEGA_M_PRIOR[0]
            <= omega_m
            <= OMEGA_M_PRIOR[1]
        ):
            return np.inf

        d = self._raw_residual(omega_m)

        if np.any(~np.isfinite(d)):
            return np.inf

        cinv_d = cho_solve(
            self._cho,
            d,
            check_finite=False,
        )

        A = float(d @ cinv_d)
        B = float(self._one @ cinv_d)

        return A - B * B / self._C

    def optimize(self):
        result = minimize_scalar(
            self.chi2,
            bounds=OMEGA_M_PRIOR,
            method="bounded",
            options={
                "xatol": 1e-10,
                "maxiter": 500,
            },
        )

        if not result.success:
            raise RuntimeError(
                f"Pantheon+ optimization failed: {result.message}"
            )

        return result

    def posterior_grid(
        self,
        n_grid=501,
        half_width=0.15,
    ):
        """Deterministic one-dimensional posterior for uniform Omega_m prior."""

        opt = self.optimize()
        center = float(opt.x)

        lo = max(
            OMEGA_M_PRIOR[0],
            center - half_width,
        )
        hi = min(
            OMEGA_M_PRIOR[1],
            center + half_width,
        )

        grid = np.linspace(
            lo,
            hi,
            int(n_grid),
        )

        chi2 = np.array(
            [self.chi2(x) for x in grid],
            dtype=float,
        )

        logp = -0.5 * (
            chi2 - np.min(chi2)
        )
        density = np.exp(logp)

        norm = np.trapezoid(
            density,
            grid,
        )
        density /= norm

        cdf = cumulative_trapezoid(
            density,
            grid,
            initial=0.0,
        )
        cdf /= cdf[-1]

        def q(prob):
            return float(
                np.interp(
                    prob,
                    cdf,
                    grid,
                )
            )

        quantiles = {
            "q2.5": q(0.025),
            "q16": q(0.16),
            "median": q(0.50),
            "q84": q(0.84),
            "q97.5": q(0.975),
        }

        quantiles.update({
            "minus_68": (
                quantiles["median"]
                - quantiles["q16"]
            ),
            "plus_68": (
                quantiles["q84"]
                - quantiles["median"]
            ),
            "minus_95": (
                quantiles["median"]
                - quantiles["q2.5"]
            ),
            "plus_95": (
                quantiles["q97.5"]
                - quantiles["median"]
            ),
        })

        return {
            "omega_m_grid": grid,
            "chi2_grid": chi2,
            "density": density,
            "quantiles": quantiles,
            "best_fit_omega_m": float(opt.x),
            "chi2_min": float(opt.fun),
            "best_intercept": self.best_intercept(
                float(opt.x)
            ),
        }
