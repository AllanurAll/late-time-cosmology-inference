import numpy as np


def information_criteria(loglike_max, n_parameters, n_data):
    """AIC/BIC from maximum likelihood, not the posterior-mean likelihood."""
    aic = -2.0 * loglike_max + 2.0 * n_parameters
    bic = -2.0 * loglike_max + n_parameters * np.log(n_data)
    return {"AIC": float(aic), "BIC": float(bic)}


def reduced_chi2(chi2_min, n_data, n_parameters):
    dof = n_data - n_parameters
    return float(chi2_min / dof), int(dof)
