#!/usr/bin/env python3
"""Parity case: out = (w * x).sum(1) + b, w=[2304, 768], x=[768], b=[2304].

The gpt2 qkv-projection shape (kernel r_64_12_3_192_4 on CPU). The upcast
axis is unit-stride in the reduce epilogue: the bias load and the output
store address lanes 1..N-1 as `alu + c` while lane 0 uses the shared `alu`
itself. Pins that lane 0's address is the same uop as the base of the
other lanes' adds, so the renderer reuses the named subexpression instead
of re-deriving it.

Backends are limited to cpu and cuda: kernel-name counters are shared
across backends, so the reference must be generated with exactly the
backends the OCaml side renders.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import ALL_BACKENDS, dump_tensor, mk_param, wrap_sink  # noqa: E402
from tinygrad.uop.ops import UOp, Ops  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402

BACKENDS = {k: v for k, v in ALL_BACKENDS.items() if k in ("cpu", "cuda")}


def build():
    x = mk_param(0, 768)
    w = mk_param(1, 2304, 768)
    b = mk_param(2, 2304)
    xe = x.reshape((1, 768)).expand((2304, 768))
    red = UOp(Ops.REDUCE, dtypes.float32, (w * xe,), (Ops.ADD, (1,)))
    return wrap_sink(red + b)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
