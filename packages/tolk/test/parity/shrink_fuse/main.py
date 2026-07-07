#!/usr/bin/env python3
"""Parity case: e = (a*b)[0] * d, shape [8192,16], d=[1,16]."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_tensor, mk_param, wrap_sink  # noqa: E402


def build():
    a = mk_param(0, 8192, 16)
    b = mk_param(1, 8192, 16)
    d = mk_param(2, 1, 16)
    mul = a * b
    shrunk = mul.shrink(((0, 1), (0, 16)))
    return wrap_sink(shrunk * d)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"))
