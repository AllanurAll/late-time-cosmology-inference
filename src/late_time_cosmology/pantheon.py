from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from .distances import sn_magnitude_without_intercept


def read_flat_covariance(path, n_full):
    vals = np.loadtxt(path)
    if vals.size == n_full*n_full + 1 and int(vals[0]) == n_full:
        vals = vals[1:]
    elif vals.size == n_full*n_full:
        pass
    else:
        raise ValueError(f"Unexpected covariance size: {vals.size}")
    return vals.reshape(n_full, n_full)


class PantheonPlusLikelihood:
    """Pantheon+ relative-distance likelihood with analytic intercept marginalization.

    Default selection:
      zHD > 0.01
      IS_CALIBRATOR == 0

    This is intentionally Pantheon+ only, not Pantheon+SH0ES calibration.
    """
    def __init__(self, data_file, cov_file, exclude_calibrators=True):
        data = pd.read_csv(data_file, sep=r"\s+")
        mask = data["zHD"].to_numpy(float) > 0.01
        if exclude_calibrators:
            mask &= data["IS_CALIBRATOR"].to_numpy(int) == 0

        full_cov = read_flat_covariance(cov_file, len(data))
        self.data = data.loc[mask].reset_index(drop=True)
        self.cov = full_cov[np.ix_(mask, mask)]
        self.cov = 0.5 * (self.cov + self.cov.T)

        self.z_hd = self.data["zHD"].to_numpy(float)
        self.z_hel = self.data["zHEL"].to_numpy(float)
        self.m_obs = self.data["m_b_corr"].to_numpy(float)

        self._cho = cho_factor(self.cov, lower=True, check_finite=False)
        self._one = np.ones(len(self.m_obs))
        self._cinv_one = cho_solve(self._cho, self._one, check_finite=False)
        self._C = float(self._one @ self._cinv_one)

    @property
    def n(self):
        return len(self.m_obs)

    def chi2_marginalized(self, cosmo):
        base = sn_magnitude_without_intercept(self.z_hd, self.z_hel, cosmo)
        if np.any(~np.isfinite(base)):
            return np.inf
        d = self.m_obs - base
        cinv_d = cho_solve(self._cho, d, check_finite=False)
        A = float(d @ cinv_d)
        B = float(self._one @ cinv_d)
        return A - B*B/self._C

    def best_intercept(self, cosmo):
        base = sn_magnitude_without_intercept(self.z_hd, self.z_hel, cosmo)
        d = self.m_obs - base
        cinv_d = cho_solve(self._cho, d, check_finite=False)
        B = float(self._one @ cinv_d)
        return B/self._C
