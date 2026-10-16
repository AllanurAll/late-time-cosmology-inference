"""Pantheon+SH0ES calibrated likelihood for spatially-flat XCDM.

The public Pantheon+SH0ES selection/covariance treatment is retained.
Only the Hubble-flow luminosity distance is generalized from LambdaCDM to

    E(z)^2 = Omega_m (1+z)^3
           + (1-Omega_m) (1+z)^[3(1+w_X)].

At fixed (Omega_m, w_X), the profiled supernova likelihood remains exactly
quadratic in x = 5 log10(H0).  A reusable two-dimensional cache therefore
stores only the Omega_m/w_X-dependent coefficients of that quadratic.
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import cho_solve

from .pantheon_shoes import (
    PantheonSH0ESFlatLCDM,
)
from .xcdm_background import (
    chi_dimensionless_xcdm,
)

C_KM_S = 299792.458


class PantheonSH0ESFlatXCDM(
    PantheonSH0ESFlatLCDM
):
    """Exact full-covariance Pantheon+SH0ES XCDM likelihood."""

    def _a_vector_xcdm(
        self,
        omega_m,
        w_x,
    ):
        a = np.empty(
            self.n,
            dtype=float,
        )

        cal = self.is_calibrator
        hf = ~cal

        a[cal] = (
            self.m_obs[cal]
            - self.ceph_dist[cal]
        )

        chi = chi_dimensionless_xcdm(
            self.z_hd[hf],
            omega_m,
            w_x,
        )

        dist_without_h0 = (
            C_KM_S
            * (
                1.0
                + self.z_hel[hf]
            )
            * chi
        )

        if np.any(
            dist_without_h0 <= 0.0
        ):
            return np.full(
                self.n,
                np.nan,
            )

        mu_without_h0 = (
            5.0
            * np.log10(
                dist_without_h0
            )
            + 25.0
        )

        a[hf] = (
            self.m_obs[hf]
            - mu_without_h0
        )

        return a

    def contractions_xcdm(
        self,
        omega_m,
        w_x,
    ):
        a = self._a_vector_xcdm(
            omega_m,
            w_x,
        )

        if np.any(
            ~np.isfinite(a)
        ):
            return None

        cinv_a = cho_solve(
            self._cho,
            a,
            check_finite=False,
        )

        return {
            "aa": float(
                a @ cinv_a
            ),
            "one_a": float(
                self._one
                @ cinv_a
            ),
            "hf_a": float(
                self._hf
                @ cinv_a
            ),
        }

    def chi2_profiled_xcdm(
        self,
        H0,
        omega_m,
        w_x,
    ):
        c = self.contractions_xcdm(
            omega_m,
            w_x,
        )

        if c is None:
            return np.inf

        x = (
            5.0
            * np.log10(
                float(H0)
            )
        )

        r0r0 = (
            c["aa"]
            + 2.0
            * x
            * c["hf_a"]
            + x*x
            * self._hf_hf
        )

        one_r0 = (
            c["one_a"]
            + x
            * self._one_hf
        )

        return float(
            r0r0
            - one_r0**2
            / self._one_one
        )

    def best_MB_xcdm(
        self,
        H0,
        omega_m,
        w_x,
    ):
        c = self.contractions_xcdm(
            omega_m,
            w_x,
        )

        if c is None:
            return np.nan

        x = (
            5.0
            * np.log10(
                float(H0)
            )
        )

        return float(
            (
                c["one_a"]
                + x
                * self._one_hf
            )
            / self._one_one
        )


class PantheonSH0ESXCDMProfileCache:
    """Spline cache for the exact profiled Pantheon+SH0ES XCDM likelihood.

    At fixed Omega_m and w_X,

        chi2 = A(Omega_m,w_X)
             + B(Omega_m,w_X) x
             + C x^2,

    with x = 5 log10(H0) and constant C.

    A and B are evaluated from the full covariance on a 2-D grid and stored.
    Inside the cache domain they are bicubically interpolated.  Outside the
    cache domain the exact likelihood is evaluated, so the scientific prior
    is never truncated by the acceleration scheme.
    """

    CACHE_VERSION = 1

    def __init__(
        self,
        exact_like,
        omega_grid,
        w_grid,
        A,
        B,
        C,
    ):
        self.exact = exact_like

        self.omega_grid = np.asarray(
            omega_grid,
            dtype=float,
        )
        self.w_grid = np.asarray(
            w_grid,
            dtype=float,
        )
        self.A = np.asarray(
            A,
            dtype=float,
        )
        self.B = np.asarray(
            B,
            dtype=float,
        )
        self.C = float(C)

        self.omega_min = float(
            self.omega_grid[0]
        )
        self.omega_max = float(
            self.omega_grid[-1]
        )
        self.w_min = float(
            self.w_grid[0]
        )
        self.w_max = float(
            self.w_grid[-1]
        )

        self._A_spline = (
            RectBivariateSpline(
                self.omega_grid,
                self.w_grid,
                self.A,
                kx=3,
                ky=3,
                s=0.0,
            )
        )

        self._B_spline = (
            RectBivariateSpline(
                self.omega_grid,
                self.w_grid,
                self.B,
                kx=3,
                ky=3,
                s=0.0,
            )
        )

    @classmethod
    def build(
        cls,
        exact_like,
        omega_min=0.16,
        omega_max=0.48,
        n_omega=101,
        w_min=-1.80,
        w_max=-0.25,
        n_w=121,
        progress=True,
    ):
        omega_grid = np.linspace(
            omega_min,
            omega_max,
            int(n_omega),
        )
        w_grid = np.linspace(
            w_min,
            w_max,
            int(n_w),
        )

        A = np.empty(
            (
                len(omega_grid),
                len(w_grid),
            ),
            dtype=float,
        )

        B = np.empty_like(
            A
        )

        one_one = (
            exact_like._one_one
        )
        one_hf = (
            exact_like._one_hf
        )
        hf_hf = (
            exact_like._hf_hf
        )

        for j, w_x in enumerate(
            w_grid
        ):
            if (
                progress
                and (
                    j == 0
                    or (j + 1) % 10 == 0
                    or j + 1 == len(w_grid)
                )
            ):
                print(
                    "Pantheon XCDM cache row "
                    f"{j+1}/{len(w_grid)}"
                )

            amat = np.column_stack([
                exact_like._a_vector_xcdm(
                    om,
                    w_x,
                )
                for om in omega_grid
            ])

            cinv_a = cho_solve(
                exact_like._cho,
                amat,
                check_finite=False,
            )

            aa = np.sum(
                amat * cinv_a,
                axis=0,
            )
            one_a = (
                exact_like._one
                @ cinv_a
            )
            hf_a = (
                exact_like._hf
                @ cinv_a
            )

            A[:, j] = (
                aa
                - one_a**2
                / one_one
            )

            B[:, j] = (
                2.0
                * (
                    hf_a
                    - one_a
                    * one_hf
                    / one_one
                )
            )

        C = (
            hf_hf
            - one_hf**2
            / one_one
        )

        return cls(
            exact_like,
            omega_grid,
            w_grid,
            A,
            B,
            C,
        )

    @classmethod
    def load_or_build(
        cls,
        exact_like,
        cache_path,
        **build_kwargs,
    ):
        cache_path = Path(
            cache_path
        )
        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if cache_path.exists():
            data = np.load(
                cache_path
            )

            version = int(
                data[
                    "cache_version"
                ]
            )

            if (
                version
                != cls.CACHE_VERSION
            ):
                raise RuntimeError(
                    "Pantheon XCDM cache version mismatch. "
                    "Delete the cache file and rebuild it."
                )

            print(
                "Loaded Pantheon+SH0ES XCDM cache:",
                cache_path,
            )

            return cls(
                exact_like,
                data[
                    "omega_grid"
                ],
                data[
                    "w_grid"
                ],
                data["A"],
                data["B"],
                float(
                    data["C"]
                ),
            )

        obj = cls.build(
            exact_like,
            **build_kwargs,
        )

        np.savez_compressed(
            cache_path,
            cache_version=np.asarray(
                cls.CACHE_VERSION
            ),
            omega_grid=obj.omega_grid,
            w_grid=obj.w_grid,
            A=obj.A,
            B=obj.B,
            C=np.asarray(
                obj.C
            ),
        )

        print(
            "Saved Pantheon+SH0ES XCDM cache:",
            cache_path,
        )

        return obj

    def inside_cache(
        self,
        omega_m,
        w_x,
    ):
        return (
            self.omega_min
            <= omega_m
            <= self.omega_max
            and self.w_min
            <= w_x
            <= self.w_max
        )

    def chi2(
        self,
        H0,
        omega_m,
        w_x,
    ):
        if self.inside_cache(
            omega_m,
            w_x,
        ):
            A = float(
                self._A_spline.ev(
                    omega_m,
                    w_x,
                )
            )
            B = float(
                self._B_spline.ev(
                    omega_m,
                    w_x,
                )
            )

            x = (
                5.0
                * np.log10(
                    float(H0)
                )
            )

            return float(
                A
                + B*x
                + self.C*x*x
            )

        # Exact fallback means the cache never changes the scientific support.
        return self.exact.chi2_profiled_xcdm(
            H0,
            omega_m,
            w_x,
        )

    def max_abs_error(
        self,
        test_points,
    ):
        errs = []

        for (
            H0,
            omega_m,
            w_x,
        ) in test_points:
            fast = self.chi2(
                H0,
                omega_m,
                w_x,
            )

            exact = (
                self.exact
                .chi2_profiled_xcdm(
                    H0,
                    omega_m,
                    w_x,
                )
            )

            errs.append(
                abs(
                    fast
                    - exact
                )
            )

        return float(
            max(errs)
        )
