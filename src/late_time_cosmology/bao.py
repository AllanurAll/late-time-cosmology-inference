from pathlib import Path
import json
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .distances import D_M, D_H, D_A, D_V


class BAOLikelihood:
    def __init__(self, data_dir):
        data_dir = Path(data_dir)
        self.meta = json.loads((data_dir / "bao_ratra2021.json").read_text())
        self.boss_cov = np.loadtxt(data_dir / "boss_dr12_cov.txt")
        self.lya_cov = np.loadtxt(data_dir / "lya_dr16_cov.txt")
        self._boss_cho = cho_factor(self.boss_cov, lower=True, check_finite=False)
        self._lya_cho = cho_factor(self.lya_cov, lower=True, check_finite=False)

    def chi2(self, cosmo):
        rs = cosmo.sound_horizon_aubourg()
        if not np.isfinite(rs) or rs <= 0.0:
            return np.inf

        # BOSS DR12 correlated 6-vector
        zvals = (0.38, 0.51, 0.61)
        rfid = self.meta["boss_dr12"]["r_s_fid_Mpc"]
        pred = []
        for z in zvals:
            pred += [D_M(z, cosmo) * rfid / rs, float(cosmo.H(z)) * rs / rfid]
        pred = np.asarray(pred)
        obs = np.asarray(self.meta["boss_dr12"]["value"], dtype=float)
        d = obs - pred
        chi2 = float(d @ cho_solve(self._boss_cho, d, check_finite=False))

        # Independent measurements
        for item in self.meta["independent"]:
            z = float(item["z"])
            if item["observable"] == "DV_scaled":
                th = D_V(z, cosmo) * float(item["r_s_fid_Mpc"]) / rs
            elif item["observable"] == "DA_over_rs":
                th = D_A(z, cosmo) / rs
            else:
                raise ValueError(item["observable"])
            chi2 += ((float(item["value"]) - th) / float(item["sigma"]))**2

        # Correlated Ly-alpha pair, order [DM/rs, DH/rs]
        z = float(self.meta["lya_dr16"]["z"])
        pred_lya = np.array([D_M(z, cosmo) / rs, D_H(z, cosmo) / rs])
        obs_lya = np.asarray(self.meta["lya_dr16"]["value"], dtype=float)
        dlya = obs_lya - pred_lya
        chi2 += float(dlya @ cho_solve(self._lya_cho, dlya, check_finite=False))
        return chi2
