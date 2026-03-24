#!/usr/bin/env python3
"""Generate tinygrad reference .expected files for debug golden tests.

Captures the exact print_uops output after each named graph_rewrite stage
in full_rewrite_to_sink, concatenated into a single file with === headers.

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

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType, print_uops, graph_rewrite
from tinygrad.dtype import dtypes
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.renderer.cstyle import ClangRenderer

OUT_DIR = os.path.dirname(__file__)
RENDERER = ClangRenderer()
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Names of codegen stages in full_rewrite_to_sink.
CODEGEN_STAGES = {
    "early movement ops",
    "load collapse",
    "split ranges",
    "initial symbolic",
    "simplify ranges",
    "postopt symbolic",
    "expander",
    "add local buffers",
    "remove_reduce",
    "add gpudims",
    "** add loads (code)",
    "devectorize",
    "lower all index dtypes",
    "post index symbolic",
    "decompositions",
    "decomp dtypes",
    "transcendental",
    "final rewrite",
    "add control flow",
}


def strip_ansi(s):
    return ANSI_RE.sub("", s)


def capture_print_uops(uops):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_uops(uops)
    return strip_ansi(buf.getvalue().rstrip("\n"))


def build_elementwise_add(name, opts_to_apply):
    p0 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), 0)
    p1 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), 1)
    p2 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), 2)
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld_a = p0.index(r0, ptr=True).load()
    ld_b = p1.index(r0, ptr=True).load()
    add = ld_a + ld_b
    st = p2.index(r0, ptr=True).store(add)
    end = st.end(r0)
    return UOp.sink(end, arg=KernelInfo(
        name=name, axis_types=(AxisType.GLOBAL,), opts_to_apply=opts_to_apply))


def generate_test(name, sink):
    import tinygrad.codegen as codegen_mod
    orig_gr = codegen_mod.graph_rewrite
    sections = []

    def capturing_graph_rewrite(*args, **kwargs):
        result = orig_gr(*args, **kwargs)
        stage_name = kwargs.get("name", "")
        if stage_name in CODEGEN_STAGES:
            uops = list(result.toposort())
            sections.append(f"=== {stage_name} ===\n" + capture_print_uops(uops))
        return result

    codegen_mod.graph_rewrite = capturing_graph_rewrite
    try:
        full_rewrite_to_sink(sink, RENDERER, optimize=True)
    finally:
        codegen_mod.graph_rewrite = orig_gr

    content = "\n".join(sections)
    path = os.path.join(OUT_DIR, f"{name}.expected")
    with open(path, "w") as f:
        f.write(content + "\n")
    print(f"wrote {path} ({len(sections)} stages)")


def main():
    # Test 1: no optimization (scalar)
    generate_test("elementwise_add",
        build_elementwise_add("elementwise_add", opts_to_apply=()))
    # Test 2: auto-optimized (float4 upcast)
    generate_test("elementwise_add_opt",
        build_elementwise_add("elementwise_add_opt", opts_to_apply=None))


if __name__ == "__main__":
    main()
