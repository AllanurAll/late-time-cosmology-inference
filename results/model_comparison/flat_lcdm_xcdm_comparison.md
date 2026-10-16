# Flat LambdaCDM and XCDM comparison

The same calibrated late-time data combination is used in both models:

- Pantheon+SH0ES with the full released STAT+SYS covariance
- 32 cosmic-chronometer measurements with the correlated Moresco covariance
- 11 BAO measurements, including the BOSS DR12 and Ly-alpha covariance blocks

The quoted parameter uncertainties are marginalized 68% credible intervals.

| Model | H0 [km s^-1 Mpc^-1] | Omega_m | r_s [Mpc] | w_X | chi2_min | AIC | BIC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat LambdaCDM | 73.20 +0.94 -0.93 | 0.308 +0.012 -0.011 | 136.83 +2.06 -2.03 | -1 (fixed) | 1477.325 | 1485.325 | 1507.079 |
| flat XCDM | 72.89 +0.95 -0.94 | 0.285 +0.016 -0.015 | 136.49 +2.08 -2.03 | -0.884 +0.050 -0.051 | 1472.580 | 1482.580 | 1509.772 |

For XCDM relative to LambdaCDM,

- Delta chi2 = -4.746
- Delta AIC = -2.746
- Delta BIC = +2.693

The extra XCDM parameter lowers the minimum chi-square by 4.746. The corresponding one-parameter nested likelihood-ratio diagnostic is approximately sqrt(Delta chi2) = 2.18, with p approximately 0.029. This is only an approximate frequentist diagnostic.

The information criteria give a deliberately mixed conclusion. AIC mildly favors XCDM because the improvement in fit is large enough to offset its one extra parameter, while BIC mildly favors the simpler LambdaCDM model. The result is therefore best described as a mild preference for w_X greater than -1, not decisive evidence for dynamical dark energy.

The XCDM marginalized result is

w_X = -0.884 +0.050 -0.051  (68%)

with a central 95% interval

-0.986 < w_X < -0.787.

The LambdaCDM value w_X=-1 lies just outside this central 95% interval in the present combined-data analysis. This statement is specific to the likelihoods, calibrations, priors, and data compilation used in this repository.
