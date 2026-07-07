#!/usr/bin/env python3
"""Parity case: b = a[:start_pos+1].sum(), a=[128], start_pos in [1,127].

The gpt2 attention shape: a reduce whose bound is a symbolic variable. Pins
the variable-as-kernel-argument signature and the symbolic loop bound.

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
    a = mk_param(0, 128)
    v = UOp.variable("start_pos", 1, 127)
    shrunk = a.shrink(((0, v + 1),))
    red = UOp(Ops.REDUCE, dtypes.float32, (shrunk,), (Ops.ADD, (0,)))
    return wrap_sink(red)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
