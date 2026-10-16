"""11-value BAO likelihood for spatially-flat XCDM."""

from pathlib import Path
import json

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .xcdm_background import (
    E_flat_xcdm,
    chi_dimensionless_xcdm,
)

C_KM_S = 299792.458


class BAOFlatXCDM:
    """The Stage-03 BAO compilation generalized from LambdaCDM to XCDM.

    The late-time observables depend on

        Omega_m, w_X, H0*r_s,

    so the theory is evaluated with r_s*h as the ruler-scale parameter.
    """

    def __init__(self, data_dir):
        data_dir = Path(data_dir)

        self.meta = json.loads(
            (
                data_dir
                / "bao_ratra2021.json"
            ).read_text()
        )

        self.boss_cov = np.loadtxt(
            data_dir / "boss_dr12_cov.txt"
        )
        self.lya_cov = np.loadtxt(
            data_dir / "lya_dr16_cov.txt"
        )

        self._boss_cho = cho_factor(
            self.boss_cov,
            lower=True,
            check_finite=False,
        )
        self._lya_cho = cho_factor(
            self.lya_cov,
            lower=True,
            check_finite=False,
        )

        self.n_data = 11

    @staticmethod
    def _q(rs_h):
        # H0*r_s = 100*(r_s h), in km/s.
        return 100.0 * float(rs_h)

    def theory_vector(
        self,
        omega_m,
        w_x,
        rs_h,
    ):
        q = self._q(rs_h)

        z = np.array([
            0.38,
            0.51,
            0.61,
            0.122,
            0.81,
            1.52,
            2.334,
        ])

        I = chi_dimensionless_xcdm(
            z,
            omega_m,
            w_x,
        )

        Ez = E_flat_xcdm(
            z,
            omega_m,
            w_x,
        )

        ints = dict(
            zip(
                z.tolist(),
                I.tolist(),
            )
        )
        evals = dict(
            zip(
                z.tolist(),
                Ez.tolist(),
            )
        )

        # BOSS DR12 correlated six-vector:
        # [D_M r_s,fid/r_s, H r_s/r_s,fid]
        boss = []
        rfid = float(
            self.meta[
                "boss_dr12"
            ][
                "r_s_fid_Mpc"
            ]
        )

        for zz in (
            0.38,
            0.51,
            0.61,
        ):
            dm_scaled = (
                C_KM_S
                * rfid
                / q
                * ints[zz]
            )

            h_scaled = (
                q
                * evals[zz]
                / rfid
            )

            boss += [
                dm_scaled,
                h_scaled,
            ]

        independent = []

        for item in self.meta[
            "independent"
        ]:
            zz = float(
                item["z"]
            )
            obs = item[
                "observable"
            ]

            if obs == "DV_scaled":
                f = (
                    zz
                    * ints[zz] ** 2
                    / evals[zz]
                ) ** (1.0 / 3.0)

                th = (
                    C_KM_S
                    * float(
                        item[
                            "r_s_fid_Mpc"
                        ]
                    )
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
                raise ValueError(
                    f"Unknown BAO observable: {obs}"
                )

            independent.append(th)

        zz = float(
            self.meta[
                "lya_dr16"
            ][
                "z"
            ]
        )

        lya = [
            C_KM_S
            / q
            * ints[zz],
            C_KM_S
            / (
                q
                * evals[zz]
            ),
        ]

        return np.asarray(
            boss
            + independent
            + lya,
            dtype=float,
        )

    def chi2_components(
        self,
        omega_m,
        w_x,
        rs_h,
    ):
        theory = self.theory_vector(
            omega_m,
            w_x,
            rs_h,
        )

        boss_obs = np.asarray(
            self.meta[
                "boss_dr12"
            ][
                "value"
            ],
            dtype=float,
        )

        db = (
            boss_obs
            - theory[:6]
        )

        chi_boss = float(
            db
            @ cho_solve(
                self._boss_cho,
                db,
                check_finite=False,
            )
        )

        chi_ind = 0.0

        for i, item in enumerate(
            self.meta[
                "independent"
            ]
        ):
            d = (
                float(
                    item["value"]
                )
                - theory[6 + i]
            )

            chi_ind += (
                d
                / float(
                    item["sigma"]
                )
            ) ** 2

        lya_obs = np.asarray(
            self.meta[
                "lya_dr16"
            ][
                "value"
            ],
            dtype=float,
        )

        dl = (
            lya_obs
            - theory[9:11]
        )

        chi_lya = float(
            dl
            @ cho_solve(
                self._lya_cho,
                dl,
                check_finite=False,
            )
        )

        return {
            "BOSS_DR12": chi_boss,
            "independent": float(
                chi_ind
            ),
            "Ly_alpha": chi_lya,
        }

    def chi2(
        self,
        omega_m,
        w_x,
        rs_h,
    ):
        parts = self.chi2_components(
            omega_m,
            w_x,
            rs_h,
        )
        return float(
            sum(
                parts.values()
            )
        )
