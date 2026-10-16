from pathlib import Path
import numpy as np
from late_time_cosmology import Cosmology, BAOLikelihood

DATA = Path(__file__).parents[1] / "data" / "bao"


def test_bao_likelihood_finite():
    like = BAOLikelihood(DATA)
    c = Cosmology(h=0.69, omega_b_h2=0.024, omega_c_h2=0.118)
    chi2 = like.chi2(c)
    assert np.isfinite(chi2)
    assert chi2 > 0.0
