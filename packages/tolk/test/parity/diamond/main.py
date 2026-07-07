#!/usr/bin/env python3
"""Parity case: e = (a+b+c) + (a+b+d), shared subexpression."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_tensor, mk_param, wrap_sink  # noqa: E402


def build():
    a = mk_param(0, 10)
    b = mk_param(1, 10)
    c = mk_param(2, 10)
    d = mk_param(3, 10)
    ab = a + b
    return wrap_sink(ab + c + ab + d)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"))
