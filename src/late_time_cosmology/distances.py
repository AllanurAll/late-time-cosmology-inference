import numpy as np
from scipy.integrate import quad
from .models import C_KM_S, Cosmology


def chi_dimensionless(z: float, cosmo: Cosmology) -> float:
    val, _ = quad(lambda zp: 1.0 / float(cosmo.E(zp)), 0.0, float(z),
                  epsabs=1e-9, epsrel=1e-9, limit=200)
    return val


def transverse_dimensionless(z: float, cosmo: Cosmology) -> float:
    chi = chi_dimensionless(z, cosmo)
    ok = cosmo.omega_k
    if abs(ok) < 1e-10:
        return chi
    if ok > 0.0:
        s = np.sqrt(ok)
        return np.sinh(s * chi) / s
    s = np.sqrt(-ok)
    return np.sin(s * chi) / s


def D_M(z: float, cosmo: Cosmology) -> float:
    return C_KM_S / cosmo.H0 * transverse_dimensionless(z, cosmo)


def D_H(z: float, cosmo: Cosmology) -> float:
    return C_KM_S / float(cosmo.H(z))


def D_A(z: float, cosmo: Cosmology) -> float:
    return D_M(z, cosmo) / (1.0 + z)


def D_V(z: float, cosmo: Cosmology) -> float:
    return (z * D_H(z, cosmo) * D_M(z, cosmo)**2) ** (1.0 / 3.0)


def sn_dimensionless_luminosity_distance(z_hd, z_hel, cosmo: Cosmology):
    z_hd = np.asarray(z_hd, dtype=float)
    z_hel = np.asarray(z_hel, dtype=float)
    return np.array([
        (1.0 + zh) * transverse_dimensionless(zc, cosmo)
        for zc, zh in zip(z_hd, z_hel)
    ])


def sn_magnitude_without_intercept(z_hd, z_hel, cosmo: Cosmology):
    dl = sn_dimensionless_luminosity_distance(z_hd, z_hel, cosmo)
    if np.any(dl <= 0.0):
        return np.full_like(dl, np.nan)
    return 5.0 * np.log10(dl)
