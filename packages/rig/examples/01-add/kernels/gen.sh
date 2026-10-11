#!/bin/sh
#---------------------------------------------------------------------------
#  Copyright (c) 2026 The Raven authors. All rights reserved.
#  SPDX-License-Identifier: ISC
#---------------------------------------------------------------------------

# Makes each FILE of this directory from its source, or every file without
# one. Dune never runs it. After a change to a source, run it and commit the
# bytes. The host objects and the AMD code object need clang 22.1.7 and
# ld.lld 21.1.8 on PATH, and
# PATH=/opt/homebrew/opt/llvm/bin:/opt/homebrew/opt/lld@21/bin:$PATH puts
# Homebrew's first on macOS. The metallib needs macOS with Xcode's Metal
# toolchain 32023.864. The cubin needs Linux x86_64 with uv on PATH, which
# fetches ptxas 12.8.93 (the PyPI package nvidia-cuda-nvcc-cu12).
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

binary() {
  case $1 in
  add_x86_64.o) need clang 22.1.7
    clang $H --target=x86_64-none-unknown-elf add.c -o "$1" ;;
  add_arm64.o) need clang 22.1.7
    clang $H -ffixed-x18 --target=aarch64-none-unknown-elf add.c -o "$1" ;;
  add_gfx1201.hsaco) need clang 22.1.7; need ld.lld 21.1.8
    clang -c -x cl -cl-std=CL2.0 -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
      -mcode-object-version=5 -nogpulib -O2 add.cl -o add_gfx1201.o
    ld.lld -shared add_gfx1201.o -o "$1"
    chmod 644 "$1"
    rm add_gfx1201.o ;;
  add.metallib) xcrun -sdk macosx metal --version | grep -qF 32023.864 ||
      { echo "gen.sh: needs Xcode's metal 32023.864" >&2; exit 1; }
    xcrun -sdk macosx metal -c add.metal -o add.air
    xcrun -sdk macosx metallib add.air -o "$1"
    rm add.air ;;
  add_sm_89.cubin)
    nvcc=$(uv run -q --no-project --with nvidia-cuda-nvcc-cu12==12.8.93 \
      python -c 'import nvidia.cuda_nvcc as m; print(m.__path__[0])')
    "$nvcc/bin/ptxas" -arch=sm_89 add.ptx -o "$1" ;;
  *) echo "gen.sh: no recipe for $1" >&2; exit 2 ;;
  esac
}

[ $# -gt 0 ] || set -- \
  add_x86_64.o add_arm64.o add_gfx1201.hsaco add.metallib add_sm_89.cubin
for f in "$@"; do binary "$f"; done
