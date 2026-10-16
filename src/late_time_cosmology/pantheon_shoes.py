"""Pantheon+ + SH0ES flat-LambdaCDM likelihood.

This implements the public Pantheon+SH0ES CosmoSIS likelihood logic:

    selection = (zHD > 0.01) OR IS_CALIBRATOR

For Cepheid-host calibrators, the "theory distance" is CEPH_DIST.
For all other selected SNe, the theory is the cosmological luminosity
distance evaluated using zHD and zHEL.  The corrected apparent magnitude
m_b_corr is compared to

    m_model = mu_theory + M_B,

where M_B is the SN absolute-magnitude nuisance parameter.

The full released STAT+SYS covariance is used.

M_B is linear and is marginalized analytically.  The remaining two-dimensional
posterior is evaluated deterministically in (H0, Omega_m), so there is no
need to run an MCMC for this 2-parameter calibrated problem.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import differential_evolution, minimize

from .pantheon_flat import read_full_covariance, chi_dimensionless

C_KM_S = 299792.458

H0_PRIOR = (50.0, 90.0)
OMEGA_M_PRIOR = (0.10, 0.60)


class PantheonSH0ESFlatLCDM:
    def __init__(self, data_file, cov_file):
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

        # Exact public Pantheon+SH0ES likelihood selection.
        mask = (
            (data["zHD"].to_numpy(float) > 0.01)
            | (data["IS_CALIBRATOR"].to_numpy(int).astype(bool))
        )

        self.mask = mask
        self.data = data.loc[mask].reset_index(drop=True)

        self.z_hd = self.data["zHD"].to_numpy(float)
        self.z_hel = self.data["zHEL"].to_numpy(float)
        self.m_obs = self.data["m_b_corr"].to_numpy(float)
        self.is_calibrator = (
            self.data["IS_CALIBRATOR"].to_numpy(int).astype(bool)
        )
        self.ceph_dist = self.data["CEPH_DIST"].to_numpy(float)

        cov = cov_full[np.ix_(mask, mask)]
        self.max_raw_asymmetry = float(
            np.max(np.abs(cov - cov.T))
        )
        self.cov = 0.5 * (cov + cov.T)

        self._cho = cho_factor(
            self.cov,
            lower=True,
            check_finite=False,
        )

        self._one = np.ones(self.n, dtype=float)
        self._hf = (~self.is_calibrator).astype(float)

        self._cinv_one = cho_solve(
            self._cho,
            self._one,
            check_finite=False,
        )
        self._cinv_hf = cho_solve(
            self._cho,
            self._hf,
            check_finite=False,
        )

        # Constant contractions used by the fast profiled likelihood.
        self._one_one = float(self._one @ self._cinv_one)
        self._hf_hf = float(self._hf @ self._cinv_hf)
        self._one_hf = float(self._one @ self._cinv_hf)

        # Conditional Gaussian width of M_B at fixed (H0, Omega_m).
        self.sigma_MB_conditional = float(
            1.0 / np.sqrt(self._one_one)
        )

    @property
    def n(self):
        return len(self.m_obs)

    @property
    def n_calibrators(self):
        return int(np.sum(self.is_calibrator))

    @property
    def n_hubble_flow(self):
        return int(np.sum(~self.is_calibrator))

    def _a_vector(self, omega_m):
        """Residual vector before adding the H0 shift and subtracting M_B.

        For Hubble-flow SNe:
            a = m_obs - [5log10(c (1+zHEL) chi(zHD)) + 25]

        Since
            mu = 5log10[c (1+zHEL) chi/H0] + 25,
        the complete residual is
            r = a + 5log10(H0) * hf - M_B * 1.

        For calibrators:
            a = m_obs - CEPH_DIST.
        """
        a = np.empty(self.n, dtype=float)

        cal = self.is_calibrator
        hf = ~cal

        a[cal] = (
            self.m_obs[cal]
            - self.ceph_dist[cal]
        )

        chi = chi_dimensionless(
            self.z_hd[hf],
            omega_m,
        )

        dist_without_h0 = (
            C_KM_S
            * (1.0 + self.z_hel[hf])
            * chi
        )

        if np.any(dist_without_h0 <= 0.0):
            return np.full(self.n, np.nan)

        mu_without_h0 = (
            5.0 * np.log10(dist_without_h0)
            + 25.0
        )

        a[hf] = (
            self.m_obs[hf]
            - mu_without_h0
        )

        return a

    def contractions(self, omega_m):
        """Return covariance contractions for one Omega_m."""
        a = self._a_vector(omega_m)

        if np.any(~np.isfinite(a)):
            return None

        cinv_a = cho_solve(
            self._cho,
            a,
            check_finite=False,
        )

        return {
            "aa": float(a @ cinv_a),
            "one_a": float(self._one @ cinv_a),
            "hf_a": float(self._hf @ cinv_a),
        }

    def chi2_profiled(self, H0, omega_m):
        """Chi^2 with M_B analytically profiled/marginalized."""
        if not (
            H0_PRIOR[0] <= H0 <= H0_PRIOR[1]
            and OMEGA_M_PRIOR[0] <= omega_m <= OMEGA_M_PRIOR[1]
        ):
            return np.inf

        c = self.contractions(omega_m)
        if c is None:
            return np.inf

        x = 5.0 * np.log10(float(H0))

        # r0 = a + x * hf  (before subtracting M_B)
        r0r0 = (
            c["aa"]
            + 2.0 * x * c["hf_a"]
            + x * x * self._hf_hf
        )

        one_r0 = (
            c["one_a"]
            + x * self._one_hf
        )

        # Profile over global additive M_B.
        return float(
            r0r0
            - one_r0 * one_r0 / self._one_one
        )

    def best_MB(self, H0, omega_m):
        """Best-fit SN absolute magnitude at fixed H0, Omega_m."""
        c = self.contractions(omega_m)
        if c is None:
            return np.nan

        x = 5.0 * np.log10(float(H0))

        return float(
            (
                c["one_a"]
                + x * self._one_hf
            )
            / self._one_one
        )

    def optimize(self):
        """Robust 2D profile-likelihood maximum."""

        def objective(theta):
            return self.chi2_profiled(
                theta[0],
                theta[1],
            )

        # Global pass followed by local polish.
        de = differential_evolution(
            objective,
            bounds=[
                H0_PRIOR,
                OMEGA_M_PRIOR,
            ],
            seed=20261016,
            polish=False,
            tol=1e-8,
        )

        result = minimize(
            objective,
            x0=de.x,
            method="Nelder-Mead",
            options={
                "xatol": 1e-9,
                "fatol": 1e-8,
                "maxiter": 5000,
            },
        )

        if not result.success:
            raise RuntimeError(
                f"Pantheon+SH0ES optimization failed: {result.message}"
            )

        return result

    def posterior_grid(
        self,
        n_omega=321,
        n_h0=321,
        omega_range=(0.25, 0.42),
        h0_range=(68.0, 79.0),
    ):
        """Evaluate the M_B-marginalized posterior on a regular 2D grid.

        Because the M_B curvature is independent of H0 and Omega_m, analytic
        marginalization differs from profiling only by a constant factor.
        With uniform priors in H0 and Omega_m, exp(-chi2_profile/2) is therefore
        the correctly normalized posterior shape.
        """
        om_grid = np.linspace(
            omega_range[0],
            omega_range[1],
            int(n_omega),
        )
        h0_grid = np.linspace(
            h0_range[0],
            h0_range[1],
            int(n_h0),
        )

        # Build all a(Omega_m) vectors, then solve C^{-1} for all columns
        # in one BLAS-backed operation.
        A = np.column_stack([
            self._a_vector(om)
            for om in om_grid
        ])

        cinv_A = cho_solve(
            self._cho,
            A,
            check_finite=False,
        )

        aa = np.sum(
            A * cinv_A,
            axis=0,
        )
        one_a = self._one @ cinv_A
        hf_a = self._hf @ cinv_A

        x = 5.0 * np.log10(h0_grid)

        chi2 = (
            aa[:, None]
            + 2.0 * hf_a[:, None] * x[None, :]
            + self._hf_hf * x[None, :] ** 2
            - (
                one_a[:, None]
                + self._one_hf * x[None, :]
            ) ** 2 / self._one_one
        )

        chi2_min = float(np.min(chi2))
        logp = -0.5 * (chi2 - chi2_min)

        posterior = np.exp(logp)
        posterior /= np.sum(posterior)

        return {
            "omega_grid": om_grid,
            "h0_grid": h0_grid,
            "chi2": chi2,
            "posterior": posterior,
            "chi2_min_grid": chi2_min,
        }


def weighted_quantile_grid(values, weights, probs):
    order = np.argsort(values)
    x = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return np.interp(
        probs,
        cdf,
        x,
    )


def summarize_weighted(values, weights):
    q = weighted_quantile_grid(
        values,
        weights,
        [0.025, 0.16, 0.50, 0.84, 0.975],
    )

    return {
        "q2.5": float(q[0]),
        "q16": float(q[1]),
        "median": float(q[2]),
        "q84": float(q[3]),
        "q97.5": float(q[4]),
        "minus_68": float(q[2] - q[1]),
        "plus_68": float(q[3] - q[2]),
        "minus_95": float(q[2] - q[0]),
        "plus_95": float(q[4] - q[2]),
    }


def summarize_grid(grid):
    p = grid["posterior"]

    p_om = np.sum(p, axis=1)
    p_h0 = np.sum(p, axis=0)

    return {
        "H0_km_s_Mpc": summarize_weighted(
            grid["h0_grid"],
            p_h0,
        ),
        "Omega_m": summarize_weighted(
            grid["omega_grid"],
            p_om,
        ),
    }


def draw_grid_samples(
    grid,
    n_samples=300_000,
    seed=20261016,
):
    """Draw smooth samples from a regular posterior grid for GetDist."""
    rng = np.random.default_rng(seed)

    p = grid["posterior"]
    flat_p = p.ravel()

    indices = rng.choice(
        flat_p.size,
        size=int(n_samples),
        replace=True,
        p=flat_p,
    )

    iom, ih0 = np.unravel_index(
        indices,
        p.shape,
    )

    dom = (
        grid["omega_grid"][1]
        - grid["omega_grid"][0]
    )
    dh0 = (
        grid["h0_grid"][1]
        - grid["h0_grid"][0]
    )

    om = (
        grid["omega_grid"][iom]
        + rng.uniform(
            -0.5 * dom,
            0.5 * dom,
            len(indices),
        )
    )

    h0 = (
        grid["h0_grid"][ih0]
        + rng.uniform(
            -0.5 * dh0,
            0.5 * dh0,
            len(indices),
        )
    )

    return np.column_stack([
        h0,
        om,
    ])
