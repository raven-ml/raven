#!/bin/sh
#---------------------------------------------------------------------------
#  Copyright (c) 2026 The Raven authors. All rights reserved.
#  SPDX-License-Identifier: ISC
#---------------------------------------------------------------------------

# Makes each FILE of this directory from its source, or every file without
# one. Dune never runs it. After a change to a source, run it and commit the
# bytes. The code objects need clang 22.1.7 and ld.lld 21.1.8 on PATH, and
# PATH=/opt/homebrew/opt/llvm/bin:/opt/homebrew/opt/lld@21/bin:$PATH puts
# Homebrew's first on macOS. The discovery tables need uv on PATH.
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

# [code_object SOURCE GPU OUT] compiles SOURCE, OpenCL C or assembly, for GPU
# and links it into the code object OUT.
code_object() {
  need clang 22.1.7
  need ld.lld 21.1.8
  case $1 in
  *.cl) lang="-x cl -cl-std=CL2.0 -nogpulib -O2" ;;
  *.s) lang="-x assembler" ;;
  esac
  clang -c $lang -target amdgcn-amd-amdhsa -mcpu="$2" -mcode-object-version=5 \
    "$1" -o "$tmp/object.o"
  ld.lld -shared "$tmp/object.o" -o "$3"
  chmod a-x "$3"
}

fixture() {
  case $1 in
  hidden_gfx1201.hsaco) code_object hidden.s gfx1201 "$1" ;;
  launch_gfx942.hsaco) code_object launch.cl gfx942 "$1" ;;
  *_gfx1201.hsaco) code_object "${1%_gfx1201.hsaco}.cl" gfx1201 "$1" ;;
  r9700.bin | r9700_wide.bin | r9700_fused.bin) uv run -q discovery.py ;;
  *) echo "gen.sh: no recipe for $1" >&2; exit 2 ;;
  esac
}

[ $# -gt 0 ] || set -- \
  hidden_gfx1201.hsaco kernels_gfx1201.hsaco launch_gfx1201.hsaco \
  launch_gfx942.hsaco other_gfx1201.hsaco work_gfx1201.hsaco \
  r9700.bin r9700_wide.bin r9700_fused.bin
for f in "$@"; do fixture "$f"; done
