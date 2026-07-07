#!/usr/bin/env python3
"""Parity case: embedding-style gather of an arange folds at schedule time.

The gpt2 positional-embedding shape: `weight[arange(32)[:, :13]]` written
as the one-hot `eq` + `where` + `sum` gather. The arange must never
materialize as an int buffer — the whole gather folds into a single kernel
that reads the weight rows directly.

Backends are limited to cpu and cuda: kernel-name counters are shared
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

BACKENDS = {k: v for k, v in ALL_BACKENDS.items() if k in ("cpu", "cuda")}


def build():
    w = Tensor(mk_param(0, 32, 4))
    allpos = Tensor.arange(0, 32).reshape(1, -1)
    pos = allpos.shrink((None, (0, 13)))
    arange = Tensor.arange(32)
    one_hot = (arange == pos.unsqueeze(-1)).unsqueeze(-1)
    # the +1 keeps the store from being a raw vector-load passthrough,
    # which renders differently and is unrelated to the gather fold
    out = one_hot.where(w, 0).sum(-2, dtype=dtypes.float32) + 1.0
    return wrap_sink(out.uop)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
