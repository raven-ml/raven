#!/usr/bin/env python3
"""Parity case: sharded + replicated elementwise add on 2 devices.

`a` is sharded on axis 0 across CPU:0/CPU:1, `b` is replicated (multi
device, no axis). The replicated input is sharded symbolically, so the
kernel indexes it with the `_device_num` variable.

Backends are limited to cpu and cuda: kernel-name counters are shared
across backends, so the reference must be generated with exactly the
backends the OCaml side renders.

Paired with main.ml. Run to regenerate *.expected files.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import ALL_BACKENDS, dump_tensor, wrap_sink  # noqa: E402

from tinygrad.uop.ops import UOp  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402

BACKENDS = {k: v for k, v in ALL_BACKENDS.items() if k in ("cpu", "cuda")}
DEVICES = ("CPU:0", "CPU:1")


def build():
    a = UOp.param(0, dtypes.float32, shape=(8, 8), device=DEVICES, axis=0)
    b = UOp.param(1, dtypes.float32, shape=(16, 8), device=DEVICES)
    return wrap_sink(a + b)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
