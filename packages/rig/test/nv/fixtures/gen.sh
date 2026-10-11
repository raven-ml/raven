#!/bin/sh
#---------------------------------------------------------------------------
#  Copyright (c) 2026 The Raven authors. All rights reserved.
#  SPDX-License-Identifier: ISC
#---------------------------------------------------------------------------

# Makes each FILE of this directory from its source, or every file without
# one. Dune never runs it. After a change to a source, run it and commit the
# bytes. Runs on Linux x86_64. grid_params_sm89.cubin needs nvcc of CUDA 13.4
# (V13.4.92) on PATH, which PATH=/usr/local/cuda/bin:$PATH gives, and the
# others ../../gen/nvrtc.sh.
#
# usage: gen.sh [FILE...]

set -eu
cd "$(dirname "$0")"

# [need TOOL VERSION] fails unless TOOL --version prints VERSION.
need() {
  "$1" --version 2>/dev/null | grep -qF "$2" ||
    { echo "gen.sh: needs $1 $2 on PATH" >&2; exit 1; }
}

fixture() {
  case $1 in
  grid_params_sm89.cubin) need nvcc V13.4.92
    nvcc -cubin -arch=sm_89 -O3 grid_params.cu -o "$1" ;;
  *_sm89.cubin) ../../gen/nvrtc.sh "${1%_sm89.cubin}.cu" sm_89 "$1" ;;
  *) echo "gen.sh: no recipe for $1" >&2; exit 2 ;;
  esac
}

[ $# -gt 0 ] || set -- \
  grid_params_sm89.cubin kernels_sm89.cubin launch_sm89.cubin \
  thrice_sm89.cubin twice_sm89.cubin
for f in "$@"; do fixture "$f"; done
