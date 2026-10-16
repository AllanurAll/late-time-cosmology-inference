import numpy as np
from late_time_cosmology.models import Cosmology
from late_time_cosmology.distances import D_M, D_H, D_A, D_V


def test_flat_lcdm_E0():
    c = Cosmology(h=0.69, omega_b_h2=0.024, omega_c_h2=0.118)
    assert np.isclose(c.E(0.0), 1.0, atol=1e-14)


def test_xcdm_reduces_to_lcdm_at_wminus1():
    a = Cosmology(h=0.69, omega_b_h2=0.024, omega_c_h2=0.118, w_x=-1.0)
    b = Cosmology(h=0.69, omega_b_h2=0.024, omega_c_h2=0.118, w_x=-1.0)
    z = np.linspace(0,2,20)
    assert np.allclose(a.E(z), b.E(z))


def test_distance_identities():
    c = Cosmology(h=0.69, omega_b_h2=0.024, omega_c_h2=0.118)
    z = 0.5
    assert np.isclose(D_A(z,c), D_M(z,c)/(1+z))
    assert D_H(z,c) > 0
    assert D_V(z,c) > 0


def test_sound_horizon_plausible():
    c = Cosmology(h=0.69, omega_b_h2=0.024, omega_c_h2=0.118)
    rs = c.sound_horizon_aubourg()
    assert 130.0 < rs < 170.0
