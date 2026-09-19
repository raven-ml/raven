#!/usr/bin/env python3
"""Parity case: clamped SwiGLU over interleaved gate and linear features.

The expert activation: features `[...; 2 * half]` alternate gate and
linear, read as pairs `[-1; half; 2]` and split by a shrink of the last
axis. The gate is clamped above, the linear feature on both sides, and the
result is `gate * sigmoid (1.702 * gate) * (linear + 1)`, the sigmoid
written as `1 / (1 + 2 ** (x * -1 / ln 2))`. One kernel reading the
interleaved buffer at strides of two.

Backends are limited to cpu and metal: kernel-name counters are shared
across backends, so the reference must be generated with exactly the
backends the OCaml side renders.

Paired with main.ml. Run to regenerate *.expected files.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import ALL_BACKENDS, dump_tensor, mk_param, wrap_sink  # noqa: E402

from tinygrad import Tensor  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402

BACKENDS = {k: v for k, v in ALL_BACKENDS.items() if k in ("cpu", "metal")}


# A scalar as the nx frontend hands it to a binary op: a constant broadcast to
# the shape of its peer.
def scalar(value, dtype, shape):
    const = Tensor.full((), value, dtype=dtype, buffer=False)
    return const.reshape((1,) * len(shape)).expand(shape)

TOKENS, K, HALF = 2, 2, 8
LIMIT = 7.0


def build():
    f, out = dtypes.float32, (TOKENS, K, 1, HALF)
    h = Tensor(mk_param(0, TOKENS, K, 1, 2 * HALF))
    pairs = h.reshape(TOKENS * K, HALF, 2)

    def feature(i):
        column = pairs.shrink(((0, TOKENS * K), (0, HALF), (i, i + 1)))
        return column.reshape(TOKENS * K, HALF).reshape(out)

    gate = feature(0).minimum(scalar(LIMIT, f, out))
    linear = feature(1).maximum(scalar(-LIMIT, f, out))
    linear = linear.minimum(scalar(LIMIT, f, out))
    x = gate * scalar(1.702, f, out)
    sigmoid = ((x * scalar(-1.0 / math.log(2.0), f, out)).exp2()
               + scalar(1.0, f, out)).reciprocal()
    return wrap_sink((gate * sigmoid * (linear + scalar(1.0, f, out))).uop)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
