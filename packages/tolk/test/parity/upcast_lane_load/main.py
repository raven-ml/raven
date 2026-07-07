#!/usr/bin/env python3
"""Parity case: b = a.permute(1, 0).contiguous(), a=[768, 2304].

The gpt2 kv-cache transpose shape (kernel E_192_192_12_4 on CPU). The
upcast axis strides the *input*, so lanes 1..N-1 load at `alu0 + c` while
lane 0 loads at the shared `alu0` itself. Pins that lane 0's address is
the same uop as the base of the other lanes' adds, so the renderer reuses
the named subexpression instead of re-deriving it.

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
    a = mk_param(0, 768, 2304)
    permed = UOp(Ops.PERMUTE, dtypes.float32, (a,), (1, 0))
    return wrap_sink(permed)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
