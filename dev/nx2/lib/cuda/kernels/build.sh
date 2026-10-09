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

nvcc() {
  "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" -cubin -arch=sm_89 -std=c++17 -O3 \
    --fmad=false -prec-div=true -prec-sqrt=true -ftz=false \
    -Werror all-warnings -I "$nx2/lib/array" -I "$nx2/lib/kernel" -I "$nx2/lib/cuda" "$@"
}

nvcc -o "$nx2/lib/cuda/kernels/sm_89.cubin" "$nx2/lib/cuda/kernels/src/kernels.cu"
nvcc -o "$nx2/test/cuda/support/sm_89.cubin" "$nx2/test/cuda/support/harness.cu"
