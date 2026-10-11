#!/bin/sh
#---------------------------------------------------------------------------
#  Copyright (c) 2026 The Raven authors. All rights reserved.
#  SPDX-License-Identifier: ISC
#---------------------------------------------------------------------------

# Makes each FILE of this directory from its source, or every file without
# one. Dune never runs it. After a change to a source, run it and commit the
# bytes. The host and AMD objects need clang 22.1.7 and ld.lld 21.1.8 on PATH,
# and PATH=/opt/homebrew/opt/llvm/bin:/opt/homebrew/opt/lld@21/bin:$PATH puts
# Homebrew's first on macOS. The cubin needs ../../gen/nvrtc.sh, which runs on
# Linux.
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
H="-c -x c -O2 -fPIC -ffreestanding -fno-math-errno -nostdlib -fno-ident"
A="-c -x cl -cl-std=CL2.0 -target amdgcn-amd-amdhsa -mcpu=gfx1100
  -mcode-object-version=5 -nogpulib -O2"

# The AMD code objects of KERNELS kernels have no symbol table.
fixture() {
  case $1 in
  *_sm89.cubin) ../../gen/nvrtc.sh "${1%_sm89.cubin}.cu" sm_89 "$1"; return ;;
  esac
  need clang 22.1.7
  need ld.lld 21.1.8
  case $1 in
  host_x86_64.o) clang $H --target=x86_64-none-unknown-elf host.c -o "$1" ;;
  host_aarch64.o)
    clang $H -ffixed-x18 --target=aarch64-none-unknown-elf host.c -o "$1" ;;
  host_unwind_x86_64.o)
    clang $H -funwind-tables --target=x86_64-none-unknown-elf host.c -o "$1" ;;
  amd_gfx1100.o) clang $A amd.cl -o "$1" ;;
  amd_gfx1100.hsaco)
    ld.lld -shared amd_gfx1100.o -o "$1"
    chmod a-x "$1" ;;
  amd_*_gfx1100.hsaco) k=${1#amd_} k=${k%_gfx1100.hsaco}
    clang $A -DKERNELS="$k" amd.cl -o "$tmp/amd.o"
    ld.lld -shared --strip-all "$tmp/amd.o" -o "$1"
    chmod a-x "$1" ;;
  *) echo "gen.sh: no recipe for $1" >&2; exit 2 ;;
  esac
}

[ $# -gt 0 ] || set -- \
  host_x86_64.o host_aarch64.o host_unwind_x86_64.o amd_gfx1100.o \
  amd_gfx1100.hsaco amd_16_gfx1100.hsaco amd_128_gfx1100.hsaco \
  global_sm89.cubin
for f in "$@"; do fixture "$f"; done
