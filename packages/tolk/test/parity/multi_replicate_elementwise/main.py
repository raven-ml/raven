#!/usr/bin/env python3
"""Parity case: replicate (copy-to-tuple) + elementwise on replicated input.

`a` is a single-device param sharded onto 2 devices via shard() (a copy
to the device tuple followed by a symbolic shrink), `b` is replicated by
a copy to the device tuple. The broadcast copy becomes per-device copies
in an MSTACK, and the shard's shrink is moved before the MSTACK with the
`_device_num` variable substituted per device.

Backends are limited to cpu and cuda: kernel-name counters are shared
across backends, so the reference must be generated with exactly the
backends the OCaml side renders.

Paired with main.ml. Run to regenerate *.expected files.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import ALL_BACKENDS, dump_tensor, mk_param, wrap_sink  # noqa: E402

BACKENDS = {k: v for k, v in ALL_BACKENDS.items() if k in ("cpu", "cuda")}
DEVICES = ("CPU:0", "CPU:1")


def build():
    a = mk_param(0, 16, 8).shard(DEVICES, axis=0)
    b = mk_param(1, 16, 8).copy_to_device(DEVICES)
    return wrap_sink(a + b)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
