#!/usr/bin/env python3
"""Parity case: one gpt-oss MoE block in its decode form, end to end.

Router, top-2 of 4 experts by selection rounds, softmax over the selected
logits, MXFP4 rows of the selected experts decoded by arithmetic and made
contiguous, the gate-up product with its gathered bias, the clamped SwiGLU,
the down product, and the sum weighted by the routing weights. Each piece
has a case of its own; this one holds their composition, where the kernels
share operands across one another: 22 kernels for two tokens.

Backends are limited to cpu and metal: kernel-name counters are shared
across backends, so the reference must be generated with exactly the
backends the OCaml side renders.

Paired with main.ml. Run to regenerate *.expected files.
"""

import math
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


EXPERTS, K, TOKENS = 4, 2, 2
WIDTH, HIDDEN = 64, 32
LIMIT = 7.0

def dequant_rows(ids, blocks, scales, scale_table, outputs, groups):
    values = code_values(nibbles(take_rows(ids, blocks)))
    scale = lookup(scale_table, take_rows(ids, scales)).reshape(TOKENS, K, outputs, groups, 1, 1)
    return (values * scale).reshape(TOKENS, K, outputs, groups * 32).contiguous().transpose(-1, -2)

def top_k(x):
    shape, row = (TOKENS, EXPERTS), (TOKENS, 1)
    position = Tensor(mk_param(10, EXPERTS, dtype=dtypes.int32)).reshape(1, EXPERTS)._broadcast_to(shape)
    low = scalar(-math.inf, dtypes.float32, shape)
    real = x.ne(x) ^ scalar(True, dtypes.bool, shape)
    free = scalar(True, dtypes.bool, shape)
    picks = []
    for _ in range(K):
        live = free & real
        greatest = live.where(x, low).max(axis=(1,)).reshape(row)
        best = live & x.eq(greatest._broadcast_to(shape))
        some = live.ne(scalar(False, dtypes.bool, shape)).max(axis=(1,))
        chosen = some.reshape(row)._broadcast_to(shape).where(best, free)
        index = chosen.cast(dtypes.int32).argmax(axis=1, keepdim=True).cast(dtypes.int32)
        picks.append(index)
        free = free & position.ne(index._broadcast_to(shape))
    indices = picks[0].cat(*picks[1:], dim=1)
    logits = x.gather(1, indices)
    top = logits.max(axis=(1,)).reshape(row)._broadcast_to((TOKENS, K))
    e = (logits - top).exp()
    total = e.sum(axis=(1,), dtype=dtypes.float32).reshape(row)
    return indices, e / total._broadcast_to((TOKENS, K))

def activation(h):
    f, out = dtypes.float32, (TOKENS, K, 1, HIDDEN)
    pairs = h.reshape(TOKENS * K, HIDDEN, 2)
    def feature(i):
        return pairs.shrink(((0, TOKENS * K), (0, HIDDEN), (i, i + 1))).reshape(TOKENS * K, HIDDEN).reshape(out)
    gate = feature(0).minimum(scalar(LIMIT, f, out))
    linear = feature(1).maximum(scalar(-LIMIT, f, out)).minimum(scalar(LIMIT, f, out))
    x = gate * scalar(1.702, f, out)
    sigmoid = ((x * scalar(-1.0 / math.log(2.0), f, out)).exp2() + scalar(1.0, f, out)).reciprocal()
    return gate * sigmoid * (linear + scalar(1.0, f, out))

def build():
    x = Tensor(mk_param(0, TOKENS, WIDTH))
    wr = Tensor(mk_param(1, WIDTH, EXPERTS))
    br = Tensor(mk_param(2, EXPERTS))
    gu_blocks = Tensor(mk_param(3, EXPERTS, 2 * HIDDEN, WIDTH // 32, 16, dtype=dtypes.uint8))
    gu_scales = Tensor(mk_param(4, EXPERTS, 2 * HIDDEN, WIDTH // 32, dtype=dtypes.uint8))
    gu_bias = Tensor(mk_param(5, EXPERTS, 2 * HIDDEN))
    dn_blocks = Tensor(mk_param(6, EXPERTS, WIDTH, HIDDEN // 32, 16, dtype=dtypes.uint8))
    dn_scales = Tensor(mk_param(7, EXPERTS, WIDTH, HIDDEN // 32, dtype=dtypes.uint8))
    dn_bias = Tensor(mk_param(8, EXPERTS, WIDTH))
    scale_table = Tensor(mk_param(9, 256))
    logits = x.matmul(wr) + br.reshape(1, EXPERTS)._broadcast_to((TOKENS, EXPERTS))
    ids, weights = top_k(logits)
    gate_up = dequant_rows(ids, gu_blocks, gu_scales, scale_table, 2 * HIDDEN, WIDTH // 32)
    down = dequant_rows(ids, dn_blocks, dn_scales, scale_table, WIDTH, HIDDEN // 32)
    h = x.reshape(TOKENS, 1, 1, WIDTH).matmul(gate_up) + take_rows(ids, gu_bias).reshape(TOKENS, K, 1, 2 * HIDDEN)
    y = activation(h).contiguous().matmul(down) + take_rows(ids, dn_bias).reshape(TOKENS, K, 1, WIDTH)
    w = weights.reshape(TOKENS, K, 1, 1)._broadcast_to((TOKENS, K, 1, WIDTH))
    return wrap_sink((y * w).sum(axis=(1, 2), dtype=dtypes.float32).uop)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
