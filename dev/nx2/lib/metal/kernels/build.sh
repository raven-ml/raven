#!/bin/sh
# Compiles nx.metal's kernels and the Metal suite's harness kernels to the
# metallibs they embed, on a Mac with the Metal toolchain. Run by hand, from
# anywhere.
#
# A metallib holds AIR, which is compiled for the GPU when a pipeline is
# made, so one serves every Apple GPU. No reassociation, no contraction of a
# product into a sum, and the precise float32 functions; the AIR targets the
# oldest macOS rig.metal opens a device on.

set -eu
nx2=$(cd "$(dirname "$0")/../../.." && pwd)
air=$(mktemp -d)
trap 'rm -rf "$air"' EXIT

# metallib OUT SRC... compiles each SRC to AIR and links them into OUT. The
# units are appended to the arguments, then the sources shifted off, so a
# path with spaces stays one argument.
metallib() {
  out=$1
  shift
  n=$#
  for src in "$@"; do
    unit="$air/$(basename "$src").air"
    xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=15.0 \
      -fmetal-math-mode=safe -fmetal-math-fp32-functions=precise \
      -ffp-contract=off -Wall -Werror -I "$nx2/lib/array" -I "$nx2/lib/metal" \
      -c "$src" -o "$unit"
    set -- "$@" "$unit"
  done
  shift "$n"
  xcrun -sdk macosx metallib "$@" -o "$out"
}

# stamp OUT SOURCE... writes to OUT a unit of one empty function named after
# the digest of the SOURCEs, which dune checks on a Mac.
stamp() {
  out=$1
  shift
  digest=$(for f in "$@"; do md5 -q "$f"; done | md5 -q)
  printf 'kernel void nx_metal_sources_%s() {}\n' "$digest" >"$out"
}

# Each metallib's stamp names the files it is compiled from, this script
# included for its flags, in the order its dune rule lists them.
stamp "$air/kernels_stamp.metal" "$nx2/lib/metal/kernels/build.sh" \
  "$nx2/lib/metal/kernels.h" "$nx2/lib/array/nx_dtype.h" \
  "$nx2/lib/metal/kernels/src/elements.h" \
  "$nx2/lib/metal/kernels/src/contract.metal"
stamp "$air/harness_stamp.metal" "$nx2/lib/metal/kernels/build.sh" \
  "$nx2/test/metal/support/harness.h" "$nx2/lib/array/nx_dtype.h" \
  "$nx2/test/metal/support/harness.metal"

metallib "$nx2/lib/metal/kernels/kernels.metallib" \
  "$nx2/lib/metal/kernels/src/contract.metal" "$air/kernels_stamp.metal"
metallib "$nx2/test/metal/support/harness.metallib" \
  "$nx2/test/metal/support/harness.metal" "$air/harness_stamp.metal"
