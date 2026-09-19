#!/usr/bin/env python3
"""Parity case: masked softmax whose normaliser includes a per-head sink.

Scores `[heads; queries; keys]` at float32, masked to -inf, and one sink
logit per head: a key of value zero that has no column. The shift is the
maximum over the row and the sink, and the sink's exponential joins the
sum. A row with every key masked has a finite shift, the sink, so its
weights are 0 / 1 and not 0 / 0. Three kernels: shift, total, quotient,
each applying the mask itself; the sink's term fuses into the total.

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

HEADS, QUERIES, KEYS = 2, 3, 4


def build():
    shape, row = (HEADS, QUERIES, KEYS), (HEADS, QUERIES, 1)
    scores = Tensor(mk_param(0, *shape))
    sink = Tensor(mk_param(1, HEADS, 1)).reshape(HEADS, 1, 1)
    mask = Tensor(mk_param(2, QUERIES, KEYS, dtype=dtypes.bool))
    scores = mask._broadcast_to(shape).where(
        scores, scalar(-math.inf, dtypes.float32, shape))
    sink = sink._broadcast_to(row)
    top = scores.max(axis=(2,)).reshape(row).maximum(sink)
    e = (scores - top._broadcast_to(shape)).exp()
    total = e.sum(axis=(2,), dtype=dtypes.float32).reshape(row)
    total = total + (sink - top).exp()
    return wrap_sink((e / total._broadcast_to(shape)).uop)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
