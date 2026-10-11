#!/bin/sh
#---------------------------------------------------------------------------
#  Copyright (c) 2026 The Raven authors. All rights reserved.
#  SPDX-License-Identifier: ISC
#---------------------------------------------------------------------------

# Makes each FILE of this directory from its source, or every file without
# one. Dune never runs it. After a change to a source, run it and commit the
# bytes. The PTX files are written by hand. Needs ptxas of CUDA 13.4
# (V13.4.92) on PATH, which PATH=/usr/local/cuda/bin:$PATH gives.
#
# usage: gen.sh [FILE...]

set -eu
cd "$(dirname "$0")"

# [need TOOL VERSION] fails unless TOOL --version prints VERSION.
need() {
  "$1" --version 2>/dev/null | grep -qF "$2" ||
    { echo "gen.sh: needs $1 $2 on PATH" >&2; exit 1; }
}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# [many] is the PTX of 140 kernels, k0 to k139. Kernel i stores i into the
# 32-bit word at its one parameter.
many() {
  printf '.version 7.0\n.target sm_50\n.address_size 64\n'
  for i in $(seq 0 139); do
    printf '.visible .entry k%d(.param .u64 out)\n{\n  .reg .b64 %%rd<2>;\n  ld.param.u64 %%rd1, [out];\n  cvta.to.global.u64 %%rd1, %%rd1;\n  st.global.u32 [%%rd1], %d;\n  ret;\n}\n' "$i" "$i"
  done
}

fixture() {
  need ptxas V13.4.92
  case $1 in
  kernels.cubin) ptxas -arch=sm_89 -o "$1" kernels.ptx ;;
  many.cubin)
    many > "$tmp/many.ptx"
    ptxas -arch=sm_89 -o "$1" "$tmp/many.ptx" ;;
  *) echo "gen.sh: no recipe for $1" >&2; exit 2 ;;
  esac
}

[ $# -gt 0 ] || set -- kernels.cubin many.cubin
for f in "$@"; do fixture "$f"; done
