from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve


def load_cc32(data_dir):
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "cc32.csv")
    sys = np.loadtxt(data_dir / "moresco2020_systematics.dat", comments="#")
    return df, sys


def build_cc_covariance(df, sys_table):
    """Construct the 32x32 cosmic-chronometer covariance.

    The 32-point data vector follows the compilation used by Favale,
    Gomez-Valent & Migliaccio (2023). For the 15 Moresco measurements,
    the central values are the arithmetic means of the BC03 and MaStro
    determinations quoted in that compilation.

    The correlated systematic prescription follows the public Moresco
    CCcovariance example. Its full covariance example uses

        C = C_diag + C_IMF + C_SPS,OOO

    where IMF and SPS out-of-ordinary-model contributions are treated as
    fully correlated across the Moresco redshift bins. The other 17 CC
    measurements retain diagonal published uncertainties.
    """
    H = df["H"].to_numpy(float)
    sig = df["sigma_diag"].to_numpy(float)

    cov = np.diag(sig**2)

    mask = df["moresco_correlated"].to_numpy(int).astype(bool)
    idx = np.where(mask)[0]
    zc = df.loc[mask, "z"].to_numpy(float)
    hc = H[mask]

    zmod = sys_table[:, 0]
    imf_frac = np.interp(zc, zmod, sys_table[:, 1]) / 100.0
    spsooo_frac = np.interp(zc, zmod, sys_table[:, 4]) / 100.0

    block = np.diag(sig[mask]**2)
    block += np.outer(hc * imf_frac, hc * imf_frac)
    block += np.outer(hc * spsooo_frac, hc * spsooo_frac)

    cov[np.ix_(idx, idx)] = block
    return cov


class CCLikelihood:
    def __init__(self, data_dir):
        self.df, sys = load_cc32(data_dir)
        self.z = self.df["z"].to_numpy(float)
        self.Hobs = self.df["H"].to_numpy(float)
        self.cov = build_cc_covariance(self.df, sys)
        self._cho = cho_factor(self.cov, lower=True, check_finite=False)

    def chi2(self, cosmo):
        theory = cosmo.H(self.z)
        if np.any(~np.isfinite(theory)):
            return np.inf
        d = self.Hobs - theory
        return float(d @ cho_solve(self._cho, d, check_finite=False))
