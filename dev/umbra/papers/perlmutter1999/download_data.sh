#!/usr/bin/env bash
# Download Pantheon+ Type Ia supernova data (Scolnic et al. 2022, Brout et al. 2022).
#
# The Pantheon+ compilation contains 1701 light curves of 1550 unique SNe Ia
# spanning 0.001 < z < 2.26, extending the original 42 high-z supernovae from
# Perlmutter et al. (1999) that first demonstrated cosmic acceleration.
#
# Source: https://github.com/PantheonPlusSH0ES/DataRelease
# Papers: arXiv:2112.03863 (data), arXiv:2202.04077 (cosmology)

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${DIR}/data"
mkdir -p "${DATA_DIR}"

BASE_URL="https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR"

echo "Downloading Pantheon+ SN Ia distance data..."
curl -fSL "${BASE_URL}/Pantheon%2BSH0ES.dat" -o "${DATA_DIR}/Pantheon+SH0ES.dat"
echo "  -> ${DATA_DIR}/Pantheon+SH0ES.dat ($(wc -l < "${DATA_DIR}/Pantheon+SH0ES.dat") lines)"

echo "Downloading paper PDF (Perlmutter et al. 1999, arXiv:astro-ph/9812133)..."
curl -fSL "https://arxiv.org/pdf/astro-ph/9812133" -o "${DATA_DIR}/perlmutter1999.pdf"
echo "  -> ${DATA_DIR}/perlmutter1999.pdf"

echo "Done."
