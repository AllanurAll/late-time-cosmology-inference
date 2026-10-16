from pathlib import Path
import numpy as np

from late_time_cosmology.bao_reduced import BAOReducedFlatLCDM
from late_time_cosmology.bao_physical_fit import derived

DATA = Path(__file__).parents[1] / "data" / "bao"


def test_fixed_rs_best_fit_transformation():
    rsh_best = 100.92526856737831
    rs_fixed = 147.4
    H0 = 100.0 * rsh_best / rs_fixed
    assert np.isclose(H0, 68.4703314568, atol=1e-9)


def test_physical_density_points_share_bao_ridge():
    like = BAOReducedFlatLCDM(DATA)

    a = derived(
        [0.69189404, 0.02417695, 0.11718890]
    )
    b = derived(
        [0.79869342, 0.04199697, 0.14637896]
    )

    assert abs(a["Omega_m"] - b["Omega_m"]) < 2e-6
    assert abs(a["r_s_h_Mpc"] - b["r_s_h_Mpc"]) < 2e-5

    ca = like.chi2(a["Omega_m"], a["r_s_h_Mpc"])
    cb = like.chi2(b["Omega_m"], b["r_s_h_Mpc"])

    assert abs(ca - cb) < 1e-6
    assert np.isclose(ca, 8.995880173, atol=2e-6)
