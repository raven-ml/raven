#!/usr/bin/env python3
"""Parity case: constant-index views of a computed tensor fuse without a copy.

The gpt2 qkv shape: one computed tensor viewed at constant indices along an
axis (the q/k/v selectors), with the views feeding a single reduce. The
partially-realized view axis must be re-read through the producer directly --
no bufferized copy of the whole tensor.

Backends are limited to cpu and cuda: kernel-name counters are shared
across backends, so the reference must be generated with exactly the
backends the OCaml side renders.

Paired with main.ml. Run to regenerate *.expected files.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import ALL_BACKENDS, dump_tensor, mk_param, wrap_sink  # noqa: E402

from tinygrad.uop.ops import UOp, Ops  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402

BACKENDS = {k: v for k, v in ALL_BACKENDS.items() if k in ("cpu", "cuda")}


def build():
    base = mk_param(0, 3, 8)
    comp = base * base

    def sel(i):
        return comp.shrink(((i, i + 1), (0, 8))).reshape((8,))

    mul = sel(1) * sel(2)
    red = mul._rop(Ops.ADD, (0,))
    return wrap_sink(red)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
