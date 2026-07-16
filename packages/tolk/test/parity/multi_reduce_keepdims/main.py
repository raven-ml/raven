#!/usr/bin/env python3
"""Parity case: keepdims reduce on a sharded tensor, broadcast back.

`a` is sharded on axis 0 across CPU:0/CPU:1, `b` is replicated. The max
of `a + b` over axis 1 (the non-shard axis) is reduced per shard,
reshaped to keep the reduced axis as size 1, expanded back to the full
shape, and subtracted — the softmax-style keepdims pattern. The reduce
output is realized into a per-shard buffer whose shape must broadcast
against the sharded operand.

Backends are limited to cpu and cuda: kernel-name counters are shared
across backends, so the reference must be generated with exactly the
backends the OCaml side renders.

Paired with main.ml. Run to regenerate *.expected files.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import ALL_BACKENDS, dump_tensor, wrap_sink  # noqa: E402

from tinygrad.uop.ops import UOp, Ops  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402

BACKENDS = {k: v for k, v in ALL_BACKENDS.items() if k in ("cpu", "cuda")}
DEVICES = ("CPU:0", "CPU:1")


def build():
    a = UOp.param(0, dtypes.float32, shape=(4, 8), device=DEVICES, axis=0)
    b = UOp.param(1, dtypes.float32, shape=(8, 8), device=DEVICES)
    h = a.alu(Ops.ADD, b)
    red = h._rop(Ops.MAX, (1,))
    keep = red.reshape((8, 1))
    exp = keep.expand((8, 8))
    return wrap_sink(h.alu(Ops.SUB, exp))


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
