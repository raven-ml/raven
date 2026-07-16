#!/usr/bin/env python3
"""Parity case: late (default) allreduce as a precompiled function.

Same graph as multi_allreduce_naive but with the default LATE_ALLREDUCE=1:
the ALLREDUCE survives the multi rewrite and scheduling wraps it into a
precompiled allreduce function. Only the per-shard reduce and the copy
kernel around the opaque allreduce call appear in the extracted schedule.

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
    a = UOp.param(0, dtypes.float32, shape=(8, 16), device=DEVICES, axis=0)
    red = a._rop(Ops.ADD, (0,))
    return wrap_sink(red)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
