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

# metallib SRC OUT compiles SRC to AIR and links it alone into OUT.
metallib() {
  xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=15.0 \
    -fmetal-math-mode=safe -fmetal-math-fp32-functions=precise \
    -ffp-contract=off -Wall -Werror -I "$nx2/lib/array" -I "$nx2/lib/metal" \
    -c "$1" -o "$air/unit.air"
  xcrun -sdk macosx metallib "$air/unit.air" -o "$2"
}

metallib "$nx2/lib/metal/kernels/src/contract.metal" "$nx2/lib/metal/kernels/kernels.metallib"
metallib "$nx2/test/metal/support/harness.metal" "$nx2/test/metal/support/harness.metallib"
