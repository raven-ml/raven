#!/usr/bin/env python3
"""Parity case: sliding-window causal mask built from positions.

Each query carries its position p, and a key at column j is seen when
`j <= p` and `j > p - window`: a band of `window` keys ending at the query.
The mask selects scores against -inf. The columns are a buffer, as a
traced function captures its `arange`. One kernel: both comparisons fold
into the select and no mask is written out.

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

BATCH, QUERIES, KEYS, WINDOW = 1, 3, 8, 4


def build():
    shape = (BATCH, QUERIES, KEYS)
    pos = Tensor(mk_param(0, BATCH, QUERIES, dtype=dtypes.int32))
    column = Tensor(mk_param(1, KEYS, dtype=dtypes.int32))
    scores = Tensor(mk_param(2, *shape))
    keys = column.reshape(1, 1, KEYS)._broadcast_to(shape)
    query = pos.reshape(BATCH, QUERIES, 1)._broadcast_to(shape)
    causal = keys <= query
    recent = query - scalar(WINDOW, dtypes.int32, shape) < keys
    out = (causal & recent).where(
        scores, scalar(-math.inf, dtypes.float32, shape))
    return wrap_sink(out.uop)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
