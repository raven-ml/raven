#!/usr/bin/env python3
"""Generate tinygrad reference .expected files for debug golden tests.

Captures the final graph after full_rewrite_to_sink.  Tolk intentionally keeps
the debug hook minimal: DEBUG=6 prints a single stable graph dump that can be
diffed verbatim, without porting tinygrad's full tracing infrastructure.

Usage:
    uv run packages/tolk/test/golden/debug/generate_expected.py
"""

import io
import os
import re
import sys
import contextlib

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "..", "_tinygrad"
    ),
)

# Disable ANSI color in the reference — auto-generated kernel names embed ANSI
# escape codes per axis type that would otherwise leak into print_uops output.
os.environ["NO_COLOR"] = "1"

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.uop.render import print_uops
from tinygrad.dtype import dtypes
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import ClangRenderer

OUT_DIR = os.path.dirname(__file__)
RENDERER = ClangRenderer(Target("CPU", arch="x86_64,znver2"))
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

def strip_ansi(s):
    return ANSI_RE.sub("", s)


def capture_print_uops(uops):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_uops(uops)
    return strip_ansi(buf.getvalue().rstrip("\n"))


def build_elementwise_add(name, opts_to_apply, ptr_size=-1):
    p0 = UOp.param(0, dtypes.float32, shape=(ptr_size,))
    p1 = UOp.param(1, dtypes.float32, shape=(ptr_size,))
    p2 = UOp.param(2, dtypes.float32, shape=(ptr_size,))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld_a = p0.index(r0).load()
    ld_b = p1.index(r0).load()
    add = ld_a + ld_b
    st = p2.index(r0).store(add)
    end = st.end(r0)
    return UOp.sink(end, arg=KernelInfo(
        name=name, axis_types=(AxisType.GLOBAL,), opts_to_apply=opts_to_apply))


def generate_test(name, sink):
    rewritten = full_rewrite_to_sink(sink, RENDERER, optimize=True)
    content = "=== lower ===\n" + capture_print_uops(list(rewritten.toposort()))
    path = os.path.join(OUT_DIR, f"{name}.expected")
    with open(path, "w") as f:
        f.write(content + "\n")
    print(f"wrote {path}")


def main():
    generate_test("elementwise_add",
        build_elementwise_add("elementwise_add", opts_to_apply=()))
    generate_test("elementwise_add_opt",
        build_elementwise_add("elementwise_add_opt", opts_to_apply=None, ptr_size=256))


if __name__ == "__main__":
    main()
