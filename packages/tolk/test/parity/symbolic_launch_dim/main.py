#!/usr/bin/env python3
"""Parity case: b = -a[:start_pos+1], a=[128], start_pos in [1,127].

An elementwise kernel over a variable-sized axis: on GPU backends the
global launch dimension is the symbolic expression itself. Pins the
symbolic launch-dim rendering next to the variable kernel argument.

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
    v = UOp.variable("start_pos", 1, 127)
    return wrap_sink(-a.shrink(((0, v + 1),)))


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
