#!/bin/sh
#---------------------------------------------------------------------------
#  Copyright (c) 2026 The Raven authors. All rights reserved.
#  SPDX-License-Identifier: ISC
#---------------------------------------------------------------------------

# Compiles the CUDA source SOURCE, in the current directory, to the cubin OUT
# for ARCH with nvrtc.c over NVRTC 12.8.93 (CUDA 12.8), the PyPI package
# nvidia-cuda-nvrtc-cu12, which uv fetches. Another release of NVRTC writes
# its own version into the cubin. Runs on Linux x86_64 with uv on PATH.
#
# usage: nvrtc.sh SOURCE ARCH OUT

set -eu
[ $# -eq 3 ] || { echo "usage: nvrtc.sh SOURCE ARCH OUT" >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd)
lib=$(uv run -q --no-project --with nvidia-cuda-nvrtc-cu12==12.8.93 \
  python -c 'import nvidia.cuda_nvrtc as m; print(m.__path__[0])')
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cc "$here/nvrtc.c" -I"$lib/include" -L"$lib/lib" -l:libnvrtc.so.12 \
  -Wl,-rpath,"$lib/lib" -o "$tmp/nvrtc"
"$tmp/nvrtc" "$1" "$2" "$3"
