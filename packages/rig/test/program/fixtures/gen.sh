#!/bin/sh
#---------------------------------------------------------------------------
#  Copyright (c) 2026 The Raven authors. All rights reserved.
#  SPDX-License-Identifier: ISC
#---------------------------------------------------------------------------

# Makes each FILE of this directory from its source, or every file without
# one. Dune never runs it. After a change to a source, run it and commit the
# bytes. Needs clang 22.1.7 on PATH. PATH=/opt/homebrew/opt/llvm/bin:$PATH
# puts Homebrew's first on macOS.
#
# usage: gen.sh [FILE...]

set -eu
cd "$(dirname "$0")"

# [need TOOL VERSION] fails unless TOOL --version prints VERSION.
need() {
  "$1" --version 2>/dev/null | grep -qF "$2" ||
    { echo "gen.sh: needs $1 $2 on PATH" >&2; exit 1; }
}

H="-c -x c -O2 -fPIC -ffreestanding -fno-math-errno -nostdlib -fno-ident"
X="--target=x86_64-none-unknown-elf"
A="-ffixed-x18 --target=aarch64-none-unknown-elf"

fixture() {
  need clang 22.1.7
  case $1 in
  *_x86_64.o) clang $H $X "${1%_x86_64.o}.c" -o "$1" ;;
  *_aarch64.o) clang $H $A "${1%_aarch64.o}.c" -o "$1" ;;
  *) echo "gen.sh: no recipe for $1" >&2; exit 2 ;;
  esac
}

[ $# -gt 0 ] || set -- fill_x86_64.o fill_aarch64.o spin_x86_64.o spin_aarch64.o
for f in "$@"; do fixture "$f"; done
