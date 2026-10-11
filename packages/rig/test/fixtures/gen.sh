#!/bin/sh
#---------------------------------------------------------------------------
#  Copyright (c) 2026 The Raven authors. All rights reserved.
#  SPDX-License-Identifier: ISC
#---------------------------------------------------------------------------

# Makes each FILE of this directory from its source, or every file without
# one. Dune never runs it. After a change to a source, run it and commit the
# bytes. The host objects need clang 22.1.7 on PATH, and
# PATH=/opt/homebrew/opt/llvm/bin:$PATH puts Homebrew's first on macOS. The
# cubin needs ../gen/nvrtc.sh, which runs on Linux.
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
  case $1 in
  *_x86_64.o) need clang 22.1.7
    clang $H $X "${1%_x86_64.o}.c" -o "$1" ;;
  *_x86_64_windows.o) need clang 22.1.7
    clang $H $X -DRIG_WINDOWS "${1%_x86_64_windows.o}.c" -o "$1" ;;
  *_aarch64.o) need clang 22.1.7
    clang $H $A "${1%_aarch64.o}.c" -o "$1" ;;
  *_sm89.cubin) ../gen/nvrtc.sh "${1%_sm89.cubin}.cu" sm_89 "$1" ;;
  *) echo "gen.sh: no recipe for $1" >&2; exit 2 ;;
  esac
}

[ $# -gt 0 ] || set -- \
  affine_x86_64.o affine_aarch64.o \
  calling_x86_64.o calling_aarch64.o \
  waiting_x86_64.o waiting_aarch64.o ready_x86_64.o ready_aarch64.o \
  ready_x86_64_windows.o \
  arrival_x86_64.o arrival_aarch64.o \
  simple_add_sm89.cubin
for f in "$@"; do fixture "$f"; done
