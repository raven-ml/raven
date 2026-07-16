#!/usr/bin/env python3
"""Generate tinygrad reference .expected files for expect tests.

Constructs linearized UOp programs and calls the renderer directly (bypassing
get_program's rewrite pipeline). This produces rendered source code from
tinygrad's renderer that matches the flat IR programs constructed in tolk's
generate_actual.ml.

Usage:
    uv run tolk/test/golden/cstyle/generate_expected.py

After running, commit the generated .expected files. Dune's expect tests diff
tolk's .actual output against these tinygrad-generated .expected files.
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
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.helpers import Target
from tinygrad.codegen.opt import tc
from tinygrad.renderer.cstyle import ClangRenderer, CUDARenderer, MetalRenderer, OpenCLRenderer

OUT_DIR = os.path.dirname(__file__)

RENDERERS = {}

class _RenderOnlyCUDARenderer(CUDARenderer):
    """CUDARenderer without NVRTC init — render-only, no execution."""

    def __init__(self, target):
        self.target, self.compiler = target, None
        ver = int(target.arch[3:])
        self.tensor_cores = (
            tc.cuda_sm89 if ver >= 89
            else tc.cuda_sm80 if ver >= 80
            else tc.cuda_sm75 if ver >= 75
            else []
        )

for _name, _ctor in [
    ("cuda", lambda: _RenderOnlyCUDARenderer(Target("CUDA", arch="sm_80"))),
    ("metal", lambda: MetalRenderer(Target("METAL"))),
    ("opencl", lambda: OpenCLRenderer(Target("CL"))),
    ("clang", lambda: ClangRenderer(Target("CPU", arch="x86_64,znver2"))),
]:
    try:
        RENDERERS[_name] = _ctor()
    except Exception as e:
        print(f"WARNING: skipping {_name} renderer: {e}")


def write_expected(name, content):
    """Write a .expected file."""
    path = os.path.join(OUT_DIR, f"{name}.expected")
    with open(path, "w") as f:
        f.write(content + "\n")
    print(f"  wrote {path}")


# ── Linearized program builders ──
# Each returns a list[UOp] in linearized (topologically sorted) form.
# These correspond to the OCaml make_* functions in test_renderer.ml.
#
# Key differences from kernel-level UOps:
# - Buffers are PARAMs carrying their element dtype; pointer-ness comes from
#   AddrSpace, not a PtrDType. INDEX takes the buffer plus index sources.
# - RANGE has a single source (upper bound), not (start, end)
# - RANGE arg is (axis_index, AxisType) tuple
# - LOCAL buffers are placeholders in AddrSpace.LOCAL.


def build_simple_add_f32():
    """Two loads, one add, one store (float32)."""
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    b = UOp.param(1, dtypes.float32, shape=(-1,))
    c = UOp.param(2, dtypes.float32, shape=(-1,))
    idx = UOp.const(dtypes.int, 0)
    idx_a = a.index(idx)
    ld_a = UOp(Ops.LOAD, dtypes.float32, (idx_a,))
    idx_b = b.index(idx)
    ld_b = UOp(Ops.LOAD, dtypes.float32, (idx_b,))
    add = ld_a + ld_b
    idx_c = c.index(idx)
    store = UOp(Ops.STORE, dtypes.void, (idx_c, add))
    return [sink, a, b, c, idx, idx_a, ld_a, idx_b, ld_b, add, idx_c, store]


def build_simple_mul_i32():
    """Integer multiply."""
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.int32, shape=(-1,))
    b = UOp.param(1, dtypes.int32, shape=(-1,))
    c = UOp.param(2, dtypes.int32, shape=(-1,))
    idx = UOp.const(dtypes.int, 0)
    idx_a = a.index(idx)
    ld_a = UOp(Ops.LOAD, dtypes.int32, (idx_a,))
    idx_b = b.index(idx)
    ld_b = UOp(Ops.LOAD, dtypes.int32, (idx_b,))
    mul = ld_a * ld_b
    idx_c = c.index(idx)
    store = UOp(Ops.STORE, dtypes.void, (idx_c, mul))
    return [sink, a, b, c, idx, idx_a, ld_a, idx_b, ld_b, mul, idx_c, store]


def build_loop():
    """For loop with load/store."""
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    ten = UOp.const(dtypes.int, 10)
    ridx = UOp(Ops.RANGE, dtypes.int, (ten,), (0, AxisType.LOOP))
    idx_ld = a.index(ridx)
    ld = UOp(Ops.LOAD, dtypes.float32, (idx_ld,))
    idx_st = a.index(ridx)
    store = UOp(Ops.STORE, dtypes.void, (idx_st, ld))
    end = UOp(Ops.END, dtypes.void, (ridx,))
    return [sink, a, ten, ridx, idx_ld, ld, idx_st, store, end]


def build_gated_load():
    """Gated load with alt value."""
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    b = UOp.param(1, dtypes.float32, shape=(-1,))
    idx = UOp.const(dtypes.int, 0)
    gate = UOp.const(dtypes.bool, True)
    alt = UOp.const(dtypes.float32, 0.0)
    idx_a = a.index(idx)
    ld = UOp(Ops.LOAD, dtypes.float32, (idx_a, alt, gate))
    idx_b = b.index(idx)
    store = UOp(Ops.STORE, dtypes.void, (idx_b, ld))
    return [sink, a, b, idx, gate, alt, idx_a, ld, idx_b, store]


def build_shared_memory():
    """Shared memory + barrier."""
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    temp = UOp.placeholder((256,), dtypes.float32, 0, AddrSpace.LOCAL)
    idx = UOp.const(dtypes.int, 0)
    zero = UOp.const(dtypes.float32, 0.0)
    idx_local = temp.index(idx)
    store_local = UOp(Ops.STORE, dtypes.void, (idx_local, zero))
    barrier = UOp(Ops.BARRIER, dtypes.void, (store_local,))
    after = temp.after(barrier)
    idx_local2 = after.index(idx)
    ld = UOp(Ops.LOAD, dtypes.float32, (idx_local2,))
    idx_global = a.index(idx)
    store_global = UOp(Ops.STORE, dtypes.void, (idx_global, ld))
    return [sink, a, temp, idx, zero, idx_local, store_local, barrier, after, idx_local2, ld, idx_global, store_global]


def build_where_select():
    """Ternary where."""
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    b = UOp.param(1, dtypes.float32, shape=(-1,))
    c = UOp.param(2, dtypes.float32, shape=(-1,))
    idx = UOp.const(dtypes.int, 0)
    idx_a = a.index(idx)
    ld_a = UOp(Ops.LOAD, dtypes.float32, (idx_a,))
    idx_b = b.index(idx)
    ld_b = UOp(Ops.LOAD, dtypes.float32, (idx_b,))
    cond = UOp.const(dtypes.bool, True)
    where = cond.where(ld_a, ld_b)
    idx_c = c.index(idx)
    store = UOp(Ops.STORE, dtypes.void, (idx_c, where))
    return [sink, a, b, c, idx, idx_a, ld_a, idx_b, ld_b, cond, where, idx_c, store]


def build_cast_f16_to_f32():
    """Float16 to Float32 cast."""
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.half, shape=(-1,))
    b = UOp.param(1, dtypes.float32, shape=(-1,))
    idx = UOp.const(dtypes.int, 0)
    idx_a = a.index(idx)
    ld = UOp(Ops.LOAD, dtypes.half, (idx_a,))
    cast = ld.cast(dtypes.float32)
    idx_b = b.index(idx)
    store = UOp(Ops.STORE, dtypes.void, (idx_b, cast))
    return [sink, a, b, idx, idx_a, ld, cast, idx_b, store]


def build_nested_loops():
    """Two nested loops."""
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    ten = UOp.const(dtypes.int, 10)
    five = UOp.const(dtypes.int, 5)
    ridx0 = UOp(Ops.RANGE, dtypes.int, (ten,), (0, AxisType.LOOP))
    ridx1 = UOp(Ops.RANGE, dtypes.int, (five,), (1, AxisType.LOOP))
    combined = ridx0 + ridx1
    idx_ld = a.index(combined)
    ld = UOp(Ops.LOAD, dtypes.float32, (idx_ld,))
    idx_st = a.index(combined)
    store = UOp(Ops.STORE, dtypes.void, (idx_st, ld))
    end1 = UOp(Ops.END, dtypes.void, (ridx1,))
    end0 = UOp(Ops.END, dtypes.void, (ridx0,))
    return [sink, a, ten, five, ridx0, ridx1, combined, idx_ld, ld, idx_st, store, end1, end0]


def build_multi_param():
    """4 params, add two and store."""
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    b = UOp.param(1, dtypes.float32, shape=(-1,))
    c = UOp.param(2, dtypes.float32, shape=(-1,))
    d = UOp.param(3, dtypes.float32, shape=(-1,))
    idx = UOp.const(dtypes.int, 0)
    idx_a = a.index(idx)
    ld_a = UOp(Ops.LOAD, dtypes.float32, (idx_a,))
    idx_b = b.index(idx)
    ld_b = UOp(Ops.LOAD, dtypes.float32, (idx_b,))
    add = ld_a + ld_b
    idx_d = d.index(idx)
    store = UOp(Ops.STORE, dtypes.void, (idx_d, add))
    return [sink, a, b, c, d, idx, idx_a, ld_a, idx_b, ld_b, add, idx_d, store]


def build_unary_sqrt_f32():
    """Sqrt on float32."""
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    b = UOp.param(1, dtypes.float32, shape=(-1,))
    idx = UOp.const(dtypes.int, 0)
    idx_a = a.index(idx)
    ld = UOp(Ops.LOAD, dtypes.float32, (idx_a,))
    sq = UOp(Ops.SQRT, dtypes.float32, (ld,))
    idx_b = b.index(idx)
    store = UOp(Ops.STORE, dtypes.void, (idx_b, sq))
    return [sink, a, b, idx, idx_a, ld, sq, idx_b, store]


def build_unary_sqrt_f16():
    """Sqrt on float16 — exercises half-precision intrinsic paths."""
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.half, shape=(-1,))
    b = UOp.param(1, dtypes.half, shape=(-1,))
    idx = UOp.const(dtypes.int, 0)
    idx_a = a.index(idx)
    ld = UOp(Ops.LOAD, dtypes.half, (idx_a,))
    sq = UOp(Ops.SQRT, dtypes.half, (ld,))
    idx_b = b.index(idx)
    store = UOp(Ops.STORE, dtypes.void, (idx_b, sq))
    return [sink, a, b, idx, idx_a, ld, sq, idx_b, store]


def build_special_dims():
    """GPU special dimensions (group_id, local_id)."""
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    bound = UOp.const(dtypes.int, 32)
    gid = UOp(Ops.SPECIAL, dtypes.int, (bound,), "gidx0")
    lid = UOp(Ops.SPECIAL, dtypes.int, (bound,), "lidx0")
    combined = gid + lid
    idx_a = a.index(combined)
    ld = UOp(Ops.LOAD, dtypes.float32, (idx_a,))
    idx_st = a.index(combined)
    store = UOp(Ops.STORE, dtypes.void, (idx_st, ld))
    return [sink, a, bound, gid, lid, combined, idx_a, ld, idx_st, store]


def build_bitcast_f32_to_i32():
    """Bitcast float32 to int32."""
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    b = UOp.param(1, dtypes.int32, shape=(-1,))
    idx = UOp.const(dtypes.int, 0)
    idx_a = a.index(idx)
    ld = UOp(Ops.LOAD, dtypes.float32, (idx_a,))
    bc = ld.bitcast(dtypes.int32)
    idx_b = b.index(idx)
    store = UOp(Ops.STORE, dtypes.void, (idx_b, bc))
    return [sink, a, b, idx, idx_a, ld, bc, idx_b, store]


def build_conditional():
    """If/Endif control flow."""
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    idx = UOp.const(dtypes.int, 0)
    cond = UOp.const(dtypes.bool, True)
    if_op = UOp(Ops.IF, dtypes.void, (cond,))
    idx_a = a.index(idx)
    one = UOp.const(dtypes.float32, 1.0)
    store = UOp(Ops.STORE, dtypes.void, (idx_a, one))
    endif = UOp(Ops.ENDIF, dtypes.void, (if_op,))
    return [sink, a, idx, cond, if_op, idx_a, one, store, endif]


def build_const_inf_nan():
    """Special float constants: infinity and NaN."""
    import math
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    idx0 = UOp.const(dtypes.int, 0)
    idx1 = UOp.const(dtypes.int, 1)
    inf_val = UOp.const(dtypes.float32, math.inf)
    nan_val = UOp.const(dtypes.float32, math.nan)
    idx_a0 = a.index(idx0)
    store0 = UOp(Ops.STORE, dtypes.void, (idx_a0, inf_val))
    idx_a1 = a.index(idx1)
    store1 = UOp(Ops.STORE, dtypes.void, (idx_a1, nan_val))
    return [sink, a, idx0, idx1, inf_val, nan_val, idx_a0, store0, idx_a1, store1]


def build_vectorize_index():
    """Vectorize 4 floats, then index element 2."""
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    b = UOp.param(1, dtypes.float32, shape=(-1,))
    idx0 = UOp.const(dtypes.int, 0)
    idx1 = UOp.const(dtypes.int, 1)
    idx2 = UOp.const(dtypes.int, 2)
    idx3 = UOp.const(dtypes.int, 3)
    ia0 = a.index(idx0)
    ia1 = a.index(idx1)
    ia2 = a.index(idx2)
    ia3 = a.index(idx3)
    v0 = UOp(Ops.LOAD, dtypes.float32, (ia0,))
    v1 = UOp(Ops.LOAD, dtypes.float32, (ia1,))
    v2 = UOp(Ops.LOAD, dtypes.float32, (ia2,))
    v3 = UOp(Ops.LOAD, dtypes.float32, (ia3,))
    vec = UOp.stack(v0, v1, v2, v3)
    lane = vec.index(idx2).simplify()
    idx_b = b.index(idx0)
    store = UOp(Ops.STORE, dtypes.void, (idx_b, lane))
    return [sink, a, b, idx0, idx1, idx2, idx3,
            ia0, ia1, ia2, ia3, v0, v1, v2, v3,
            vec, lane, idx_b, store]


def build_vectorize_index_scalarized():
    """Scalarized equivalent for backends that cannot render STACK lane extract."""
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    b = UOp.param(1, dtypes.float32, shape=(-1,))
    idx0 = UOp.const(dtypes.int, 0)
    idx1 = UOp.const(dtypes.int, 1)
    idx2 = UOp.const(dtypes.int, 2)
    idx3 = UOp.const(dtypes.int, 3)
    ia0 = a.index(idx0)
    ia1 = a.index(idx1)
    ia2 = a.index(idx2)
    ia3 = a.index(idx3)
    v0 = UOp(Ops.LOAD, dtypes.float32, (ia0,))
    v1 = UOp(Ops.LOAD, dtypes.float32, (ia1,))
    v2 = UOp(Ops.LOAD, dtypes.float32, (ia2,))
    v3 = UOp(Ops.LOAD, dtypes.float32, (ia3,))
    idx_b = b.index(idx0)
    store = UOp(Ops.STORE, dtypes.void, (idx_b, v2))
    return [sink, a, b, idx0, idx1, idx2, idx3,
            ia0, ia1, ia2, ia3, v0, v1, v2, v3, idx_b, store]


# ── Main ──

TEST_CASES = [
    ("simple_add_f32", build_simple_add_f32, None),
    ("simple_mul_i32", build_simple_mul_i32, None),
    ("loop", build_loop, None),
    ("gated_load", build_gated_load, None),
    ("shared_memory", build_shared_memory, ["cuda", "metal", "opencl"]),
    ("where_select", build_where_select, None),
    ("cast_f16_to_f32", build_cast_f16_to_f32, None),
    ("nested_loops", build_nested_loops, None),
    ("multi_param", build_multi_param, None),
    ("unary_sqrt_f32", build_unary_sqrt_f32, None),
    ("unary_sqrt_f16", build_unary_sqrt_f16, None),
    ("special_dims", build_special_dims, ["cuda", "metal", "opencl"]),
    ("bitcast_f32_to_i32", build_bitcast_f32_to_i32, None),
    ("conditional", build_conditional, None),
    ("const_inf_nan", build_const_inf_nan, None),
    ("vectorize_index", build_vectorize_index, None),
]


def main():
    total = 0
    for case_name, builder, backends in TEST_CASES:
        print(f"\n{case_name}:")
        targets = backends if backends else list(RENDERERS.keys())
        for backend_name in targets:
            if backend_name not in RENDERERS:
                print(f"  SKIP {backend_name}_{case_name}: renderer not available")
                continue
            renderer = RENDERERS[backend_name]
            snap_name = f"{backend_name}_{case_name}"
            try:
                if case_name == "vectorize_index" and backend_name in {"clang", "cuda"}:
                    uops = build_vectorize_index_scalarized()
                else:
                    uops = builder()
                src = renderer.render(uops).strip()
                write_expected(snap_name, src)
                total += 1
            except Exception as e:
                print(f"  SKIP {snap_name}: {e}")

    print(f"\nDone. Generated {total} .expected files in {OUT_DIR}")


if __name__ == "__main__":
    main()
