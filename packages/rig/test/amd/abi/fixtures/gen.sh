#!/bin/sh
#---------------------------------------------------------------------------
#  Copyright (c) 2026 The Raven authors. All rights reserved.
#  SPDX-License-Identifier: ISC
#---------------------------------------------------------------------------

# Makes each FILE of this directory from its source, or every file without
# one. Dune never runs it. After a change to a source, run it and commit the
# bytes. The thread trace is a capture, which README.md describes. Needs clang
# 22.1.7 and ld.lld 21.1.8 on PATH.
# PATH=/opt/homebrew/opt/llvm/bin:/opt/homebrew/opt/lld@21/bin:$PATH puts
# Homebrew's first on macOS.
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
T="-target amdgcn-amd-amdhsa -mcode-object-version=5"

fixture() {
  need clang 22.1.7
  need ld.lld 21.1.8
  case $1 in
  kernels_gfx1030.o)
    clang -c -x assembler $T -mcpu=gfx1030 kernels.s -o "$1" ;;
  kernels_gfx1030.hsaco)
    ld.lld -shared kernels_gfx1030.o -o "$1"
    chmod a-x "$1" ;;
  many_gfx1201.hsaco)
    clang -c -x cl -cl-std=CL2.0 -nogpulib -O2 $T -mcpu=gfx1201 many.cl \
      -o "$tmp/many.o"
    ld.lld -shared --strip-all "$tmp/many.o" -o "$1"
    chmod a-x "$1" ;;
  *) echo "gen.sh: no recipe for $1" >&2; exit 2 ;;
  esac
}

[ $# -gt 0 ] || set -- \
  kernels_gfx1030.o kernels_gfx1030.hsaco many_gfx1201.hsaco
for f in "$@"; do fixture "$f"; done
