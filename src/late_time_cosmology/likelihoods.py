import numpy as np


class CombinedLikelihood:
    def __init__(self, *components):
        self.components = tuple(c for c in components if c is not None)

    def chi2(self, cosmo):
        total = 0.0
        for component in self.components:
            if hasattr(component, "chi2_marginalized"):
                val = component.chi2_marginalized(cosmo)
            else:
                val = component.chi2(cosmo)
            if not np.isfinite(val):
                return np.inf
            total += val
        return total

    def loglike(self, cosmo):
        chi2 = self.chi2(cosmo)
        return -0.5 * chi2 if np.isfinite(chi2) else -np.inf
