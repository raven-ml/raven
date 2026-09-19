#!/usr/bin/env python3
"""Parity case: row gather from a table above the reduce-split threshold.

A uint8 table of 65536 rows of 16 bytes, gathered by three int32 row ids.
The gather's one-hot sum reduces 65536 rows to one, twice the ratio at which
`split_reduceop` splits a reduce in the tensor graph. tolk leaves a one-hot
sum whole, so the gather schedules to one kernel with one gated load per
element, the same kernel as below the threshold. The reference is generated
with `SPLIT_REDUCEOP=0`: at its default it splits the sum into 256 chunks,
each collapsing to a load gated on its chunk, and sums the 256 slots in a
second kernel.

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
from tinygrad.dtype import dtypes  # noqa: E402
from tinygrad.helpers import Context  # noqa: E402

BACKENDS = {k: v for k, v in ALL_BACKENDS.items() if k in ("cpu", "metal")}

ROWS, BYTES, TOKENS = 65536, 16, 3


def build():
    table = Tensor(mk_param(0, ROWS, BYTES, dtype=dtypes.uint8))
    ids = Tensor(mk_param(1, TOKENS, dtype=dtypes.int32))
    index = ids.reshape(TOKENS, 1).expand(TOKENS, BYTES)
    return wrap_sink(table.gather(0, index).uop)


if __name__ == "__main__":
    with Context(SPLIT_REDUCEOP=0):
        dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                    stages=("stage5", "stage7"), backends=BACKENDS)
