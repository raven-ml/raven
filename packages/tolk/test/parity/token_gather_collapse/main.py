#!/usr/bin/env python3
"""Parity case: gated-load reduce collapse of a token gather.

Backends are limited to cpu and cuda: kernel-name counters are shared
across backends, so the reference must be generated with exactly the
backends the OCaml side renders.

The gpt2 wte-embedding stage-1 kernel: a sum over a 1733-wide vocab chunk
gated by `chunk_base + r != tokens[i]`. The reduce must collapse to a
single gated load of `wte[tokens[i]*768 + col]` with the validity bounds
rewritten onto the raw token value and the index math kept in int32.

Paired with main.ml. Run to regenerate *.expected files.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import ALL_BACKENDS, dump  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType, ParamArg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402

BACKENDS = {k: v for k, v in ALL_BACKENDS.items() if k in ("cpu", "cuda")}


def kernel():
    out = UOp(Ops.PARAM, dtypes.float.ptr(289536), (), ParamArg(0))
    col = UOp.range(768, 2, AxisType.LOOP)
    chunk = UOp.range(29, 3, AxisType.LOOP)
    tok_i = UOp.range(13, 1, AxisType.LOOP)
    r = UOp.range(1733, 0, AxisType.REDUCE)
    vocab = chunk * UOp.const(dtypes.weakint, 1733) + r
    toks = UOp(Ops.PARAM, dtypes.int.ptr(13), (), ParamArg(1))
    wte = UOp(Ops.PARAM, dtypes.float.ptr(38597376), (), ParamArg(2))
    gate = vocab.cast(dtypes.int) != toks.index(tok_i)
    body = gate.where(UOp.const(dtypes.float, 0.0),
                      wte.index(vocab * UOp.const(dtypes.weakint, 768) + col))
    red = UOp(Ops.REDUCE, dtypes.float, (body, r), (Ops.ADD, ()))
    out_idx = (col * UOp.const(dtypes.weakint, 29) + chunk
               + tok_i * UOp.const(dtypes.weakint, 22272))
    st = out.index(out_idx, ptr=True).store(red).end(tok_i, col, chunk)
    return UOp.sink(
        st,
        arg=KernelInfo(name="token_gather_collapse", opts_to_apply=()),
    )


if __name__ == "__main__":
    dump(kernel(), os.path.dirname(os.path.abspath(__file__)),
         stages=("stage5", "stage7"), backends=BACKENDS)
