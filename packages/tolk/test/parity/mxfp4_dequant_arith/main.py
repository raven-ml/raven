#!/usr/bin/env python3
"""Parity case: MXFP4 dequantisation of gathered experts, codes by arithmetic.

The decode form: the ids of k = 2 experts gather their blocks
`[rows; groups; 16]` and scales `[rows; groups]` from the tables of 4
experts. A byte splits into two 4-bit codes and a code's value is computed
from its bits, so the gathered bytes feed the multiply directly: the block
gather, the decode and the multiply are one kernel and no code is written
out. The scales still read a 256-entry table of powers of two, which costs
two kernels: the gathered scale bytes as int32, then their table values.
Three kernels.

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


# t[ids] along axis 0, as `take ~axis:0`.
def take_rows(ids, t):
    n, rest = ids.numel(), t.shape[1:]
    index = ids.reshape((n,) + (1,) * len(rest)).expand((n,) + rest)
    return t.gather(0, index).reshape(ids.shape + rest)


# Each byte holds two codes, low nibble first.
def nibbles(blocks):
    low = blocks & scalar(15, dtypes.uint8, blocks.shape)
    high = blocks.div(scalar(16, dtypes.uint8, blocks.shape),
                      rounding_mode="trunc")
    return low.unsqueeze(-1).cat(high.unsqueeze(-1), dim=-1)


# The magnitudes 0, 0.5, 1, 1.5, 2, 3, 4 and 6 are m / 2 up to 4, m - 2 for 5
# and 6, then 6; bit 3 is the sign.
def code_values(codes):
    f = dtypes.float32
    m = (codes & scalar(7, dtypes.uint8, codes.shape)).cast(f)
    magnitude = (m < scalar(5.0, f, m.shape)).where(
        m * scalar(0.5, f, m.shape),
        (m < scalar(7.0, f, m.shape)).where(m - scalar(2.0, f, m.shape),
                                            scalar(6.0, f, m.shape)))
    sign = codes.div(scalar(8, dtypes.uint8, codes.shape),
                     rounding_mode="trunc").cast(f)
    return magnitude * (scalar(1.0, f, m.shape)
                        - sign * scalar(2.0, f, m.shape))

EXPERTS, ROWS, GROUPS = 4, 3, 2
TOKENS, K = 1, 2


def build():
    blocks = Tensor(mk_param(0, EXPERTS, ROWS, GROUPS, 16, dtype=dtypes.uint8))
    scales = Tensor(mk_param(1, EXPERTS, ROWS, GROUPS, dtype=dtypes.uint8))
    ids = Tensor(mk_param(2, TOKENS, K, dtype=dtypes.int32))
    scale_table = Tensor(mk_param(3, 256))
    values = code_values(nibbles(take_rows(ids, blocks)))
    scale = lookup(scale_table, take_rows(ids, scales))
    scale = scale.reshape(TOKENS, K, ROWS, GROUPS, 1, 1)
    rows = (values * scale).reshape(TOKENS, K, ROWS, GROUPS * 32)
    return wrap_sink(rows.uop)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
