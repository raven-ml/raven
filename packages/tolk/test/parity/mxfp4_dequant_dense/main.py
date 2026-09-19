#!/usr/bin/env python3
"""Parity case: MXFP4 dequantisation of a whole table by two lookups.

Blocks `[rows; groups; 16]` of packed bytes and one uint8 scale per group.
A byte splits into two 4-bit codes, a code reads a 16-entry value table, a
scale reads a 256-entry table of powers of two, and the product is laid out
as `[rows; groups * 32]`. The code lookup fuses into the multiply: its
index is elementwise over the blocks, and the one-hot sum collapses to an
ungated load because a nibble cannot leave the table. The scale lookup is a
kernel of its own, one float32 per group: a reduce is realised before the
broadcast over the group's 32 values.

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


# A scalar as the nx frontend hands it to a binary op: a constant broadcast to
# the shape of its peer.
def scalar(value, dtype, shape):
    const = Tensor.full((), value, dtype=dtype, buffer=False)
    return const.reshape((1,) * len(shape)).expand(shape)


# table[codes], as `take` of a flat table.
def lookup(table, codes):
    index = codes.cast(dtypes.int32).reshape(-1)
    return table.gather(0, index).reshape(codes.shape)


# Each byte holds two codes, low nibble first.
def nibbles(blocks):
    low = blocks & scalar(15, dtypes.uint8, blocks.shape)
    high = blocks.div(scalar(16, dtypes.uint8, blocks.shape),
                      rounding_mode="trunc")
    return low.unsqueeze(-1).cat(high.unsqueeze(-1), dim=-1)

ROWS, GROUPS = 3, 2


def build():
    blocks = Tensor(mk_param(0, ROWS, GROUPS, 16, dtype=dtypes.uint8))
    scales = Tensor(mk_param(1, ROWS, GROUPS, dtype=dtypes.uint8))
    code_table = Tensor(mk_param(2, 16))
    scale_table = Tensor(mk_param(3, 256))
    values = lookup(code_table, nibbles(blocks))
    scale = lookup(scale_table, scales).reshape(ROWS, GROUPS, 1, 1)
    return wrap_sink((values * scale).reshape(ROWS, GROUPS * 32).uop)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
