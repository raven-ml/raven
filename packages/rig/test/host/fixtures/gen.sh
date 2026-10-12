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

# got's call of an external function goes through a word on x86_64 too, and
# linked's Windows object follows that system's convention for process
# functions.
fixture() {
  need clang 22.1.7
  case $1 in
  got_x86_64.o) clang $H -fno-plt $X got.c -o "$1" ;;
  linked_x86_64_windows.o) clang $H -DWINDOWS $X linked.c -o "$1" ;;
  *_x86_64.o) clang $H $X "${1%_x86_64.o}.c" -o "$1" ;;
  *_aarch64.o) clang $H $A "${1%_aarch64.o}.c" -o "$1" ;;
  *) echo "gen.sh: no recipe for $1" >&2; exit 2 ;;
  esac
}

[ $# -gt 0 ] || set -- \
  blocks_x86_64.o blocks_aarch64.o deep_x86_64.o deep_aarch64.o \
  data_entry_x86_64.o data_entry_aarch64.o \
  empty_x86_64.o empty_aarch64.o got_x86_64.o got_aarch64.o \
  linked_x86_64.o linked_aarch64.o linked_x86_64_windows.o \
  many_x86_64.o many_aarch64.o \
  scale_x86_64.o scale_aarch64.o undefined_x86_64.o undefined_aarch64.o \
  writable_x86_64.o writable_aarch64.o
for f in "$@"; do fixture "$f"; done
