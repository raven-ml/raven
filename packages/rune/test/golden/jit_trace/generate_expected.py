#!/usr/bin/env python3
"""Generate tinygrad reference .expected files for JIT trace golden tests.

Constructs tensor-level UOp DAGs (matching what Rune's JIT capture handler
would produce) and runs them through tinygrad's
get_kernel_graph + full_rewrite_to_sink + linearize + render pipeline.

Usage:
    uv run packages/rune/test/golden/jit_trace/generate_expected.py

After running, commit the generated .expected files.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "..", "_tinygrad"
    ),
)

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.dtype import dtypes
from tinygrad.schedule.rangeify import get_kernel_graph
from tinygrad.codegen import full_rewrite_to_sink, line_rewrite, pm_linearize_cleanups
from tinygrad.codegen.late.linearizer import linearize
from tinygrad.renderer.cstyle import ClangRenderer

OUT_DIR = os.path.dirname(__file__)

renderer = ClangRenderer()


def write_expected(name, content):
    path = os.path.join(OUT_DIR, f"{name}.expected")
    with open(path, "w") as f:
        f.write(content + "\n")
    print(f"  wrote {path}")


def render_kernel(ast, optimize=True):
    """Run full codegen pipeline on a kernel AST and return rendered source."""
    rewritten = full_rewrite_to_sink(ast, renderer, optimize=optimize)
    lst = linearize(rewritten)
    lst = line_rewrite(lst, pm_linearize_cleanups)
    return renderer.render(lst).strip()


def get_source(sink, optimize=True):
    """Build tensor graph, rangeify, codegen, render all kernels."""
    kg = get_kernel_graph(sink)
    sources = []
    for u in kg.toposort():
        if u.op is Ops.CALL and isinstance(u.src[0].arg, KernelInfo):
            sources.append(render_kernel(u.src[0], optimize))
    return "\n---\n".join(sources)


# ── Helpers ──


def mk_shape(*dims):
    if len(dims) == 1:
        return UOp.const(dtypes.index, dims[0])
    return UOp(
        Ops.VECTORIZE,
        dtypes.index.vec(len(dims)),
        tuple(UOp.const(dtypes.index, d) for d in dims),
    )


def mk_param(slot, *shape, dtype=dtypes.float32):
    dev = UOp(Ops.DEVICE, arg="CPU")
    return UOp(Ops.PARAM, dtype, (mk_shape(*shape), dev), slot)


def wrap_sink(*srcs):
    contigs = [UOp(Ops.CONTIGUOUS, s.dtype, (s,)) for s in srcs]
    return UOp.sink(*contigs)


# ── Test cases ──
# Each matches a test case in generate_actual.ml


def broadcast_scalar(c, *shape):
    """Broadcast a scalar constant to a target shape via RESHAPE + EXPAND."""
    ones = tuple(1 for _ in shape)
    reshaped = UOp(Ops.RESHAPE, c.dtype, (c, mk_shape(*ones)))
    return UOp(Ops.EXPAND, c.dtype, (reshaped, mk_shape(*shape)))


def build_add_const():
    """c = a + scalar(1.0), shape [256].

    The JIT handler captures Nx.scalar as a Const (shape []) and the Add
    operates on shapes [256] + []. Tinygrad requires explicit broadcast,
    so we reshape+expand the constant to [256] to match.
    """
    a = mk_param(0, 256)
    one = broadcast_scalar(UOp.const(dtypes.float32, 1.0), 256)
    return wrap_sink(a + one)


def build_mul_self():
    """c = a * a, shape [256]."""
    a = mk_param(0, 256)
    return wrap_sink(a * a)


def build_sum():
    """c = sum(a), shape [256] -> scalar."""
    a = mk_param(0, 256)
    red = UOp(Ops.REDUCE_AXIS, dtypes.float32, (a,), (Ops.ADD, (0,)))
    return wrap_sink(red)


def build_chain():
    """c = (a + 1) * 2, shape [256].

    Both constants are scalar and need broadcast to [256].
    """
    a = mk_param(0, 256)
    one = broadcast_scalar(UOp.const(dtypes.float32, 1.0), 256)
    two = broadcast_scalar(UOp.const(dtypes.float32, 2.0), 256)
    return wrap_sink((a + one) * two)


TEST_CASES = [
    ("add_const", build_add_const),
    ("mul_self", build_mul_self),
    ("sum", build_sum),
    ("chain", build_chain),
]


def main():
    total = 0
    for case_name, builder in TEST_CASES:
        print(f"\n{case_name}:")
        sink = builder()
        try:
            src = get_source(sink)
            write_expected(case_name, src)
            total += 1
        except Exception as e:
            print(f"  FAIL {case_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Generated {total} .expected files in {OUT_DIR}")


if __name__ == "__main__":
    main()
