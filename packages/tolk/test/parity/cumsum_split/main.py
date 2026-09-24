#!/usr/bin/env python3
"""Parity case: a running sum along the leading axis of a (1000, 3) tensor.

An axis longer than 512 scans in two stages. The axis is padded to 1024 and
split into four chunks of 256, each scanned on its own; the four chunk totals
are scanned in turn, shifted by one so each chunk sees the total before it,
and added back to every element of their chunk. The padding is dropped from
the front. Three kernels: the chunk scans, the scan of their totals, and the
combine.

Backends are limited to cpu and metal: kernel-name counters are shared
across backends, so the reference must be generated with exactly the
backends the OCaml side renders.

Paired with main.ml. Run to regenerate *.expected files.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import ALL_BACKENDS, dump_tensor, mk_param, wrap_sink  # noqa: E402

from tinygrad import Tensor  # noqa: E402

BACKENDS = {k: v for k, v in ALL_BACKENDS.items() if k in ("cpu", "metal")}


def build():
    x = Tensor(mk_param(0, 1000, 3))
    return wrap_sink(x.cumsum(0).uop)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
