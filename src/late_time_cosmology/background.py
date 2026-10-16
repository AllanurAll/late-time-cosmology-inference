"""Late-time background models used when the sound horizon is irrelevant.

This module is deliberately parameterized directly by H0 and Omega_m.
That avoids introducing unconstrained baryon/CDM splits into CC-only and
SN-only fits.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FlatLCDMBackground:
    H0: float
    omega_m: float

    def E2(self, z):
        z = np.asarray(z, dtype=float)
        return self.omega_m * (1.0 + z)**3 + (1.0 - self.omega_m)

    def E(self, z):
        e2 = self.E2(z)
        if np.any(e2 <= 0.0):
            return np.full_like(np.asarray(z, dtype=float), np.nan)
        return np.sqrt(e2)

    def H(self, z):
        return self.H0 * self.E(z)


@dataclass(frozen=True)
class FlatXCDMBackground:
    H0: float
    omega_m: float
    w_x: float

    def E2(self, z):
        z = np.asarray(z, dtype=float)
        zp1 = 1.0 + z
        return (
            self.omega_m * zp1**3
            + (1.0 - self.omega_m) * zp1**(3.0 * (1.0 + self.w_x))
        )

    def E(self, z):
        e2 = self.E2(z)
        if np.any(e2 <= 0.0):
            return np.full_like(np.asarray(z, dtype=float), np.nan)
        return np.sqrt(e2)

    def H(self, z):
        return self.H0 * self.E(z)
