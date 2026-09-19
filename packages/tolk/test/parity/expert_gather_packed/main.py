#!/usr/bin/env python3
"""Parity case: gather of k experts' packed rows from a uint8 table.

The decode form of a mixture of experts over block-quantised weights: a
table `[experts; rows; groups; 16]` of packed bytes, and per token the ids
of its k = 2 experts out of 4. The ids index axis 0 through `gather`, whose
one-hot `where` + `sum` must collapse to one gated load per selected row:
the kernel reads `tokens * k` rows and never forms a table-sized
intermediate.

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

BACKENDS = {k: v for k, v in ALL_BACKENDS.items() if k in ("cpu", "metal")}

EXPERTS, ROWS, GROUPS, BYTES = 4, 3, 2, 16
TOKENS, K = 3, 2


def build():
    table = Tensor(mk_param(0, EXPERTS, ROWS, GROUPS, BYTES, dtype=dtypes.uint8))
    ids = Tensor(mk_param(1, TOKENS, K, dtype=dtypes.int32))
    n = TOKENS * K
    index = ids.reshape(n, 1, 1, 1).expand(n, ROWS, GROUPS, BYTES)
    rows = table.gather(0, index).reshape(TOKENS, K, ROWS, GROUPS, BYTES)
    return wrap_sink(rows.uop)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
