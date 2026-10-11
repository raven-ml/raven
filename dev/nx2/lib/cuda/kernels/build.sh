#!/bin/sh
# Compiles nx.cuda's kernels and the CUDA suite's harness kernels to the
# cubins they embed, with the CUDA toolkit at $CUDA_HOME, else
# /usr/local/cuda. Run by hand, from anywhere.
#
# Floats as the kernel contract states them: no contraction of a product
# into a sum, division and square roots correctly rounded, subnormals kept.
# A cubin holds no PTX, which the driver would compile at load for a GPU no
# machine here measured.

set -eu
nx2=$(cd "$(dirname "$0")/../../.." && pwd)
stamp_dir=$(mktemp -d)
trap 'rm -rf "$stamp_dir"' EXIT

nvcc() {
  "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" -cubin -arch=sm_89 -std=c++17 -O3 \
    --fmad=false -prec-div=true -prec-sqrt=true -ftz=false \
    -Werror all-warnings -I "$nx2/lib/array" -I "$nx2/lib/kernel" -I "$nx2/lib/cuda" "$@"
}

# Each cubin carries an empty kernel named after its sources' digest. The
# build checks this name before embedding it. Include this script for flags;
# keep the files in the same order as the corresponding dune rule.
stamp() {
  out=$1
  shift
  digest=$(for src in "$@"; do md5sum "$src" | cut -c1-32; done | md5sum | cut -c1-32)
  printf 'extern "C" __global__ void nx_cuda_sources_%s() {}\n' "$digest" >"$out"
}

stamp "$stamp_dir/kernels.h" "$nx2/lib/cuda/kernels/build.sh" \
  "$nx2/lib/cuda/kernels.h" "$nx2/lib/array/nx_dtype.h" \
  "$nx2/lib/kernel/nx_kinds.h" "$nx2/lib/kernel/nx_kinds_real.h" \
  "$nx2/lib/cuda/kernels/src/kernels.cu" \
  "$nx2/lib/cuda/kernels/src/contract.cu" \
  "$nx2/lib/cuda/kernels/src/combine.cuh" \
  "$nx2/lib/cuda/kernels/src/elements.cuh" \
  "$nx2/lib/cuda/kernels/src/mma.cuh" \
  "$nx2/lib/cuda/kernels/src/kinds.cuh" \
  "$nx2/lib/cuda/kernels/src/fold.cu"
stamp "$stamp_dir/harness.h" "$nx2/lib/cuda/kernels/build.sh" \
  "$nx2/test/cuda/support/harness.h" "$nx2/lib/array/nx_dtype.h" \
  "$nx2/test/cuda/support/harness.cu"

nvcc -include "$stamp_dir/kernels.h" -o "$nx2/lib/cuda/kernels/sm_89.cubin" \
  "$nx2/lib/cuda/kernels/src/kernels.cu"
nvcc -include "$stamp_dir/harness.h" -o "$nx2/test/cuda/support/sm_89.cubin" \
  "$nx2/test/cuda/support/harness.cu"
