#!/bin/sh
#---------------------------------------------------------------------------
#  Copyright (c) 2026 The Raven authors. All rights reserved.
#  SPDX-License-Identifier: ISC
#---------------------------------------------------------------------------

# Makes each FILE of this directory from its source, or every file without
# one. Dune never runs it. After a change to a source, run it and commit the
# bytes. Runs on Linux x86_64. many_sm89.cubin needs nvcc of CUDA 13.4
# (V13.4.92) on PATH, which PATH=/usr/local/cuda/bin:$PATH gives, and
# globals_sm89.cubin ../../../gen/nvrtc.sh.
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
  many_sm89.cubin) need nvcc V13.4.92
    nvcc -cubin -arch=sm_89 -O2 many.cu -o "$1" ;;
  globals_sm89.cubin) ../../../gen/nvrtc.sh globals.cu sm_89 "$1" ;;
  *) echo "gen.sh: no recipe for $1" >&2; exit 2 ;;
  esac
}

[ $# -gt 0 ] || set -- globals_sm89.cubin many_sm89.cubin
for f in "$@"; do fixture "$f"; done
