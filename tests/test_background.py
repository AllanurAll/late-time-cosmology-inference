import numpy as np
from late_time_cosmology.background import FlatLCDMBackground, FlatXCDMBackground


def test_flat_lcdm_background_E0():
    c = FlatLCDMBackground(H0=70.0, omega_m=0.3)
    assert np.isclose(c.E(0.0), 1.0)
    assert np.isclose(c.H(0.0), 70.0)


def test_flat_xcdm_reduces_to_flat_lcdm():
    z = np.linspace(0.0, 2.0, 30)
    lcdm = FlatLCDMBackground(H0=70.0, omega_m=0.3)
    xcdm = FlatXCDMBackground(H0=70.0, omega_m=0.3, w_x=-1.0)
    assert np.allclose(lcdm.H(z), xcdm.H(z))
