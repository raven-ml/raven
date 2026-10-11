#!/bin/sh
#---------------------------------------------------------------------------
#  Copyright (c) 2026 The Raven authors. All rights reserved.
#  SPDX-License-Identifier: ISC
#---------------------------------------------------------------------------

# Makes each FILE of this directory from its source, or every file without
# one. Dune never runs it. After a change to a source, run it and commit the
# bytes. Needs macOS with the Metal toolchain 32023.864 of Xcode 26.3.
#
# usage: gen.sh [FILE...]

set -eu
cd "$(dirname "$0")"

# [need TOOL VERSION] fails unless xcrun's TOOL --version prints VERSION.
need() {
  xcrun -sdk macosx "$1" --version 2>/dev/null | grep -qF "$2" ||
    { echo "gen.sh: needs $1 $2" >&2; exit 1; }
}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fixture() {
  need metal 32023.864
  need metallib 32023.864
  case $1 in
  *.metallib) name=${1%.metallib}
    xcrun -sdk macosx metal -c "$name.metal" -o "$tmp/$name.air"
    xcrun -sdk macosx metallib "$tmp/$name.air" -o "$1" ;;
  *) echo "gen.sh: no recipe for $1" >&2; exit 2 ;;
  esac
}

[ $# -gt 0 ] || set -- \
  fill.metallib launch.metallib threadgroup.metallib vertex.metallib
for f in "$@"; do fixture "$f"; done
