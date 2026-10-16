#!/usr/bin/env python3
"""Fetch the official Pantheon+ distance vector and full STAT+SYS covariance."""

from pathlib import Path
from urllib.request import urlretrieve

BASE = (
    "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/"
    "Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
)
FILES = [
    "Pantheon%2BSH0ES.dat",
    "Pantheon%2BSH0ES_STAT%2BSYS.cov",
]
OUT_NAMES = [
    "Pantheon+SH0ES.dat",
    "Pantheon+SH0ES_STAT+SYS.cov",
]

outdir = Path("data/pantheon_plus")
outdir.mkdir(parents=True, exist_ok=True)

for remote, local in zip(FILES, OUT_NAMES):
    url = BASE + remote
    target = outdir / local
    print(f"Fetching {url}")
    urlretrieve(url, target)
    print(f"Saved {target}")
