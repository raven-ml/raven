#!/usr/bin/env python3
# Reference: tinygrad d0c9745274335e44b5dd7c15b8422c2b684a3ed9.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import ALL_BACKENDS, dump_tensor, mk_param, wrap_sink
from tinygrad.uop.ops import UOp
from tinygrad.dtype import dtypes

BACKENDS = {k: ALL_BACKENDS[k] for k in ("cpu", "cuda")}

def build():
    a = mk_param(0, 2, 4, device="CPU:0")
    b = mk_param(1, 2, 4, device="CPU:1")
    return wrap_sink(a.mstack(b).unshard(0) + UOp.const(1.0, dtypes.float32))

if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
