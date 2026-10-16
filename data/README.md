# Data provenance

## Cosmic chronometers

`cc/cc32.csv` follows the 32-point CCH compilation tabulated in Gómez-Valent
et al. (2023). For the Moresco et al. (2012, 2016) entries, the published
compilation uses the arithmetic mean of the BC03 and MaStro SPS-model results.
The systematic redshift-dependent percentages in
`cc/moresco2020_systematics.dat` are from Michele Moresco's public
`CCcovariance` repository.

The code constructs the covariance explicitly rather than treating every
chronometer measurement as statistically independent.

## BAO

The starter BAO likelihood follows the compilation and covariance information
used by Cao, Ryan & Ratra (2021), including the correlated BOSS DR12 block and
the high-redshift Lyman-alpha pair.

## Pantheon+

The official data vector and covariance are obtained from the
PantheonPlusSH0ES/DataRelease repository, under
`Pantheon+_Data/4_DISTANCES_AND_COVAR`.

The analysis uses the Pantheon+ relative-distance information and does not use
the SH0ES distance-ladder calibration as a prior.
