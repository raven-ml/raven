#!/usr/bin/env python3
"""Parity case: top-2 of 4 router logits by selection rounds, then a softmax.

The graph `Nx.top_k` emits for a small k, in reference ops. A round takes
the maximum over the entries still free (NaNs aside), marks the entries
equal to it, and `argmax` picks one of them; the pick leaves the free set
by comparing positions. The two picks are concatenated, gather their
logits, and a softmax over the k logits gives the routing weights. The
reference's own `topk` is a full sort followed by a shrink; this case pins
the rounds. A round costs four kernels: whether any entry is live, the
maximum, and the two reduces of `argmax` (a maximum, then the last position
holding it); the second round first writes its candidate mask out as int32.
Then one kernel for the concatenation, one for the gather and three for the
softmax. Fourteen kernels.

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

TOKENS, EXPERTS, K = 3, 4, 2


def build():
    shape, row = (TOKENS, EXPERTS), (TOKENS, 1)
    x = Tensor(mk_param(0, *shape))
    position = Tensor(mk_param(1, EXPERTS, dtype=dtypes.int32))
    position = position.reshape(1, EXPERTS)._broadcast_to(shape)
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
        index = chosen.cast(dtypes.int32).argmax(axis=1, keepdim=True)
        index = index.cast(dtypes.int32)
        picks.append(index)
        free = free & position.ne(index._broadcast_to(shape))
    indices = picks[0].cat(*picks[1:], dim=1)
    logits = x.gather(1, indices)
    top = logits.max(axis=(1,)).reshape(row)._broadcast_to((TOKENS, K))
    e = (logits - top).exp()
    total = e.sum(axis=(1,), dtype=dtypes.float32).reshape(row)
    weights = e / total._broadcast_to((TOKENS, K))
    return wrap_sink(indices.uop, weights.uop)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
