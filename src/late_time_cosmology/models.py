from dataclasses import dataclass
import numpy as np

C_KM_S = 299792.458
OMEGA_NU0 = 0.0014  # Cao, Ryan & Ratra (2021), following Carter et al. (2018)


@dataclass(frozen=True)
class Cosmology:
    h: float
    omega_b_h2: float
    omega_c_h2: float
    omega_k: float = 0.0
    w_x: float = -1.0

    @property
    def H0(self) -> float:
        return 100.0 * self.h

    @property
    def omega_m(self) -> float:
        return (self.omega_b_h2 + self.omega_c_h2) / self.h**2 + OMEGA_NU0

    @property
    def omega_de(self) -> float:
        return 1.0 - self.omega_m - self.omega_k

    def E2(self, z):
        z = np.asarray(z, dtype=float)
        zp1 = 1.0 + z
        return (
            self.omega_m * zp1**3
            + self.omega_k * zp1**2
            + self.omega_de * zp1**(3.0 * (1.0 + self.w_x))
        )

    def E(self, z):
        e2 = self.E2(z)
        if np.any(e2 <= 0.0):
            return np.full_like(np.asarray(z, dtype=float), np.nan)
        return np.sqrt(e2)

    def H(self, z):
        return self.H0 * self.E(z)

    def sound_horizon_aubourg(self) -> float:
        """Aubourg et al. (2015) approximation used in Cao, Ryan & Ratra (2021)."""
        omega_nu_h2 = OMEGA_NU0 * self.h**2
        numerator = 55.154 * np.exp(-72.3 * (omega_nu_h2 + 0.0006)**2)
        denominator = (
            self.omega_b_h2**0.12807
            * (self.omega_c_h2 + self.omega_b_h2)**0.25351
        )
        return numerator / denominator
