#!/usr/bin/env python3
"""Parity case: c = (a + b * -1.0) * reciprocal(768.0), a=b=[128].

The layernorm mean/variance shape: a float difference spelled as
add-of-negated (the frontend's sub) times a folded reciprocal constant.
Pins the float `x + y*-1 -> x - y` late rewrite and the full-precision
constant folding of `1/768` (the folded constant keeps host precision;
only the emitted literal narrows).

Backends are limited to cpu and cuda: kernel-name counters are shared
across backends, so the reference must be generated with exactly the
backends the OCaml side renders.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import ALL_BACKENDS, dump_tensor, mk_param, wrap_sink  # noqa: E402
from tinygrad.uop.ops import UOp  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402

BACKENDS = {k: v for k, v in ALL_BACKENDS.items() if k in ("cpu", "cuda")}


def build():
    a = mk_param(0, 128)
    b = mk_param(1, 128)
    diff = a + b * UOp.const(dtypes.float32, -1.0)
    return wrap_sink(diff * UOp.const(dtypes.float32, 768.0).reciprocal())


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
