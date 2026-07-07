#!/usr/bin/env python3
"""Parity case: c = a[tok:tok+1] + b[pos:pos+1], two bound variables.

A kernel taking two scalar symbolic parameters (the gpt2 decode step's
`tokens` and `start_pos`). Pins the scalar-variable argument handling when
more than one variable reaches a single kernel.

Backends are limited to cpu and cuda: kernel-name counters are shared
across backends, so the reference must be generated with exactly the
backends the OCaml side renders.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import ALL_BACKENDS, dump_tensor, mk_param, wrap_sink  # noqa: E402
from tinygrad.uop.ops import UOp  # noqa: E402

BACKENDS = {k: v for k, v in ALL_BACKENDS.items() if k in ("cpu", "cuda")}


def build():
    a = mk_param(0, 128)
    b = mk_param(1, 128)
    tok = UOp.variable("tokens", 0, 127)
    pos = UOp.variable("start_pos", 1, 127)
    return wrap_sink(a.shrink(((tok, tok + 1),)) + b.shrink(((pos, pos + 1),)))


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
