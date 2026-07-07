#!/usr/bin/env python3
"""Parity case: d = a + b + c, shape [256]."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_tensor, mk_param, wrap_sink  # noqa: E402


def build():
    a = mk_param(0, 256)
    b = mk_param(1, 256)
    c = mk_param(2, 256)
    return wrap_sink(a + b + c)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"))
