"""BAO-only flat-LambdaCDM likelihood in the actually constrained variables.

BAO-only data do not separately determine H0 and r_s. In a flat-LambdaCDM
background they constrain the matter-density shape and the product H0*r_s,
equivalently r_s*h.

This module therefore parameterizes the BAO-only validation fit with

    theta = (Omega_m, r_s*h)

where r_s*h is in Mpc. This avoids presenting a prior-dominated H0 constraint
from BAO alone.

The full BAO+CC and BAO+SN combined analyses should return to the physical
density parameterization (h, Omega_b h^2, Omega_c h^2, ...), because the
additional datasets break the BAO scaling degeneracy.
"""

from pathlib import Path
import json
import numpy as np
from scipy.linalg import cho_factor, cho_solve

C_KM_S = 299792.458

# Fixed Gauss-Legendre quadrature nodes. 64 points are ample for the smooth
# late-time E(z) integrals over z <= 2.334 and are substantially faster than
# calling scipy.integrate.quad millions of times inside MCMC.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def E_flat_lcdm(z, omega_m):
    z = np.asarray(z, dtype=float)
    return np.sqrt(omega_m * (1.0 + z) ** 3 + 1.0 - omega_m)


def chi_dimensionless(z, omega_m):
    """Return integral_0^z dz'/E(z') using fixed Gauss-Legendre quadrature."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    x = 0.5 * z[:, None] * (_GL_X[None, :] + 1.0)
    integrand = 1.0 / E_flat_lcdm(x, omega_m)
    result = 0.5 * z * np.sum(_GL_W[None, :] * integrand, axis=1)
    return result


class BAOReducedFlatLCDM:
    """BAO-only likelihood in (Omega_m, r_s*h)."""

    def __init__(self, data_dir):
        data_dir = Path(data_dir)
        self.meta = json.loads((data_dir / "bao_ratra2021.json").read_text())

        self.boss_cov = np.loadtxt(data_dir / "boss_dr12_cov.txt")
        self.lya_cov = np.loadtxt(data_dir / "lya_dr16_cov.txt")

        self._boss_cho = cho_factor(
            self.boss_cov, lower=True, check_finite=False
        )
        self._lya_cho = cho_factor(
            self.lya_cov, lower=True, check_finite=False
        )

        self.n_data = 11

    @staticmethod
    def _q(rs_h):
        # H0 * r_s = (100 h) * r_s = 100 * (r_s h), in km/s.
        return 100.0 * float(rs_h)

    def theory_vector(self, omega_m, rs_h):
        """Return the 11 theoretical observables in the data-vector order."""
        q = self._q(rs_h)

        # All unique redshifts needed by the likelihood.
        z = np.array([0.38, 0.51, 0.61, 0.122, 0.81, 1.52, 2.334])
        I = chi_dimensionless(z, omega_m)
        Ez = E_flat_lcdm(z, omega_m)

        ints = dict(zip(z.tolist(), I.tolist()))
        evals = dict(zip(z.tolist(), Ez.tolist()))

        # BOSS DR12 correlated six-vector:
        # [DM*rsfid/rs, H*rs/rsfid] at z=0.38,0.51,0.61.
        boss = []
        rfid_boss = float(self.meta["boss_dr12"]["r_s_fid_Mpc"])
        for zz in (0.38, 0.51, 0.61):
            dm_scaled = C_KM_S * rfid_boss / q * ints[zz]
            h_scaled = q * evals[zz] / rfid_boss
            boss += [dm_scaled, h_scaled]

        # Independent points.
        independent = []
        for item in self.meta["independent"]:
            zz = float(item["z"])
            obs = item["observable"]

            if obs == "DV_scaled":
                f = (
                    zz
                    * ints[zz] ** 2
                    / evals[zz]
                ) ** (1.0 / 3.0)
                th = (
                    C_KM_S
                    * float(item["r_s_fid_Mpc"])
                    / q
                    * f
                )
            elif obs == "DA_over_rs":
                th = (
                    C_KM_S
                    / q
                    * ints[zz]
                    / (1.0 + zz)
                )
            else:
                raise ValueError(f"Unknown BAO observable: {obs}")

            independent.append(th)

        # Correlated Ly-alpha pair [DM/rs, DH/rs].
        zz = float(self.meta["lya_dr16"]["z"])
        lya = [
            C_KM_S / q * ints[zz],
            C_KM_S / (q * evals[zz]),
        ]

        return np.asarray(boss + independent + lya, dtype=float)

    def chi2_components(self, omega_m, rs_h):
        theory = self.theory_vector(omega_m, rs_h)

        # BOSS correlated block.
        boss_obs = np.asarray(
            self.meta["boss_dr12"]["value"], dtype=float
        )
        db = boss_obs - theory[:6]
        chi_boss = float(
            db @ cho_solve(self._boss_cho, db, check_finite=False)
        )

        # Three independent points.
        chi_ind = 0.0
        for i, item in enumerate(self.meta["independent"]):
            d = float(item["value"]) - theory[6 + i]
            chi_ind += (d / float(item["sigma"])) ** 2

        # Ly-alpha correlated pair.
        lya_obs = np.asarray(
            self.meta["lya_dr16"]["value"], dtype=float
        )
        dl = lya_obs - theory[9:11]
        chi_lya = float(
            dl @ cho_solve(self._lya_cho, dl, check_finite=False)
        )

        return {
            "BOSS_DR12": chi_boss,
            "independent": float(chi_ind),
            "Ly_alpha": chi_lya,
        }

    def chi2(self, omega_m, rs_h):
        c = self.chi2_components(omega_m, rs_h)
        return float(sum(c.values()))
