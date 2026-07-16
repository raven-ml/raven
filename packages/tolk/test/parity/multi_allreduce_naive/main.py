#!/usr/bin/env python3
"""Parity case: naive allreduce from a reduce over the sharded axis.

2 devices, `a` sharded on axis 0, summed over axis 0. The per-shard
reduce yields one partial per device; with 2 devices the allreduce takes
the naive path (copy every shard to each device and add). LATE_ALLREDUCE=0
expands the allreduce inline during the multi rewrite so its kernels are
visible in the schedule.

Backends are limited to cpu and cuda: kernel-name counters are shared
across backends, so the reference must be generated with exactly the
backends the OCaml side renders.

Paired with main.ml. Run to regenerate *.expected files.
"""

import os
import sys

os.environ["LATE_ALLREDUCE"] = "0"

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
