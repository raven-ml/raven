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

# metallib OUT SRC... compiles each SRC to AIR and links them into OUT.
metallib() {
  out=$1
  shift
  units=
  for src in "$@"; do
    unit="$air/$(basename "$src").air"
    xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=15.0 \
      -fmetal-math-mode=safe -fmetal-math-fp32-functions=precise \
      -ffp-contract=off -Wall -Werror -I "$nx2/lib/array" -I "$nx2/lib/metal" \
      -c "$src" -o "$unit"
    units="$units $unit"
  done
  # shellcheck disable=SC2086 # one word per unit
  xcrun -sdk macosx metallib $units -o "$out"
}

# The files nx.metal's kernels are compiled from, this script included for
# its flags, in the order kernels/dune lists them. The metallib gets an empty
# function named after their digest, which dune checks on a Mac.
sources="$nx2/lib/metal/kernels/build.sh $nx2/lib/metal/kernels.h
  $nx2/lib/array/nx_dtype.h $nx2/lib/metal/kernels/src/elements.h
  $nx2/lib/metal/kernels/src/contract.metal"
digest=$(for f in $sources; do md5 -q "$f"; done | md5 -q)
printf 'kernel void nx_metal_sources_%s() {}\n' "$digest" >"$air/stamp.metal"

metallib "$nx2/lib/metal/kernels/kernels.metallib" \
  "$nx2/lib/metal/kernels/src/contract.metal" "$air/stamp.metal"
metallib "$nx2/test/metal/support/harness.metallib" \
  "$nx2/test/metal/support/harness.metal"
