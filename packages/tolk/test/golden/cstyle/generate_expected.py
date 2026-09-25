#!/usr/bin/env python3
"""Generate tinygrad reference .expected files for expect tests.

Constructs linearized UOp programs and calls the renderer directly (bypassing
get_program's rewrite pipeline). This produces rendered source code from
tinygrad's renderer that matches the flat IR programs constructed in tolk's
generate_actual.ml.

Usage:
    python3 packages/tolk/test/golden/cstyle/generate_expected.py \
      --tinygrad _plans/tinygrad-a83c6f801 --output _plans/goldens-a83/cstyle

Output is explicit so changes can be attributed before updating expectations.
Dune compares tolk output with the reviewed reference corpus.
"""

import argparse
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tinygrad", type=Path, default=HERE.parents[4] / "_tinygrad_target")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.tinygrad.resolve()))
for key in ("DEBUG", "VIZ", "PROFILE", "DEV"):
    os.environ.pop(key, None)

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import (
    ClangRenderer, CUDARenderer, HIPRenderer, MetalRenderer, OpenCLRenderer,
)

OUT_DIR = args.output
OUT_DIR.mkdir(parents=True, exist_ok=True)
RENDERERS = {}


def render_only(ctor, target, compiler):
    # Run the target renderer's real initializer/matchers. Only the external
    # compiler constructor is disabled: these fixtures never compile or execute.
    with patch(compiler, return_value=None):
        return ctor(target)

for _name, _ctor in [
    ("cuda", lambda: render_only(CUDARenderer, Target("CUDA", arch="sm_80"),
                                         "tinygrad.runtime.support.compiler_cuda.NVRTCCompiler")),
    ("metal", lambda: MetalRenderer(Target("METAL"))),
    ("opencl", lambda: OpenCLRenderer(Target("CL"))),
    ("clang", lambda: ClangRenderer(Target("CPU", arch="x86_64,znver2"))),
    ("amd", lambda: render_only(HIPRenderer, Target("AMD", arch="gfx1100"),
                                        "tinygrad.runtime.support.compiler_amd.HIPCompiler")),
]:
    try:
        RENDERERS[_name] = _ctor()
    except Exception as e:
        raise RuntimeError(f"required {_name} renderer failed to initialize") from e


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
    sink = UOp(Ops.SINK, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    b = UOp.param(1, dtypes.float32, shape=(-1,))
    c = UOp.param(2, dtypes.float32, shape=(-1,))
    idx = UOp.cconst(0, dtypes.int)
    idx_a = a.index(idx)
    ld_a = UOp(Ops.LOAD, (idx_a,))
    idx_b = b.index(idx)
    ld_b = UOp(Ops.LOAD, (idx_b,))
    add = ld_a + ld_b
    idx_c = c.index(idx)
    store = UOp(Ops.STORE, (idx_c, add))
    return [sink, a, b, c, idx, idx_a, ld_a, idx_b, ld_b, add, idx_c, store]


def build_simple_mul_i32():
    """Integer multiply."""
    sink = UOp(Ops.SINK, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.int32, shape=(-1,))
    b = UOp.param(1, dtypes.int32, shape=(-1,))
    c = UOp.param(2, dtypes.int32, shape=(-1,))
    idx = UOp.cconst(0, dtypes.int)
    idx_a = a.index(idx)
    ld_a = UOp(Ops.LOAD, (idx_a,))
    idx_b = b.index(idx)
    ld_b = UOp(Ops.LOAD, (idx_b,))
    mul = ld_a * ld_b
    idx_c = c.index(idx)
    store = UOp(Ops.STORE, (idx_c, mul))
    return [sink, a, b, c, idx, idx_a, ld_a, idx_b, ld_b, mul, idx_c, store]


def build_loop():
    """For loop with load/store."""
    sink = UOp(Ops.SINK, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    ten = UOp.cconst(10, dtypes.int)
    ridx = UOp(Ops.RANGE, (ten,), (0, AxisType.WEAK))
    idx_ld = a.index(ridx)
    ld = UOp(Ops.LOAD, (idx_ld,))
    idx_st = a.index(ridx)
    store = UOp(Ops.STORE, (idx_st, ld))
    end = UOp(Ops.END, (store, ridx))
    return [sink, a, ten, ridx, idx_ld, ld, idx_st, store, end]


def build_gated_load():
    """Gated load with alt value."""
    sink = UOp(Ops.SINK, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    b = UOp.param(1, dtypes.float32, shape=(-1,))
    idx = UOp.cconst(0, dtypes.int)
    gate = UOp.cconst(True, dtypes.bool)
    alt = UOp.cconst(0.0, dtypes.float32)
    idx_a = a.index(idx)
    ld = UOp(Ops.LOAD, (idx_a, alt, gate))
    idx_b = b.index(idx)
    store = UOp(Ops.STORE, (idx_b, ld))
    return [sink, a, b, idx, gate, alt, idx_a, ld, idx_b, store]


def build_shared_memory():
    """Shared memory + barrier."""
    sink = UOp(Ops.SINK, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    temp = UOp.placeholder((256,), dtypes.float32, 0, AddrSpace.LOCAL)
    idx = UOp.cconst(0, dtypes.int)
    zero = UOp.cconst(0.0, dtypes.float32)
    idx_local = temp.index(idx)
    store_local = UOp(Ops.STORE, (idx_local, zero))
    barrier = UOp(Ops.BARRIER, (store_local,))
    after = temp.after(barrier)
    idx_local2 = after.index(idx)
    ld = UOp(Ops.LOAD, (idx_local2,))
    idx_global = a.index(idx)
    store_global = UOp(Ops.STORE, (idx_global, ld))
    return [sink, a, temp, idx, zero, idx_local, store_local, barrier, after, idx_local2, ld, idx_global, store_global]


def build_where_select():
    """Ternary where."""
    sink = UOp(Ops.SINK, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    b = UOp.param(1, dtypes.float32, shape=(-1,))
    c = UOp.param(2, dtypes.float32, shape=(-1,))
    idx = UOp.cconst(0, dtypes.int)
    idx_a = a.index(idx)
    ld_a = UOp(Ops.LOAD, (idx_a,))
    idx_b = b.index(idx)
    ld_b = UOp(Ops.LOAD, (idx_b,))
    cond = UOp.cconst(True, dtypes.bool)
    where = cond.where(ld_a, ld_b)
    idx_c = c.index(idx)
    store = UOp(Ops.STORE, (idx_c, where))
    return [sink, a, b, c, idx, idx_a, ld_a, idx_b, ld_b, cond, where, idx_c, store]


def build_cast_f16_to_f32():
    """Float16 to Float32 cast."""
    sink = UOp(Ops.SINK, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.half, shape=(-1,))
    b = UOp.param(1, dtypes.float32, shape=(-1,))
    idx = UOp.cconst(0, dtypes.int)
    idx_a = a.index(idx)
    ld = UOp(Ops.LOAD, (idx_a,))
    cast = ld.cast(dtypes.float32)
    idx_b = b.index(idx)
    store = UOp(Ops.STORE, (idx_b, cast))
    return [sink, a, b, idx, idx_a, ld, cast, idx_b, store]


def build_nested_loops():
    """Two nested loops."""
    sink = UOp(Ops.SINK, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    ten = UOp.cconst(10, dtypes.int)
    five = UOp.cconst(5, dtypes.int)
    ridx0 = UOp(Ops.RANGE, (ten,), (0, AxisType.WEAK))
    ridx1 = UOp(Ops.RANGE, (five,), (1, AxisType.WEAK))
    combined = ridx0 + ridx1
    idx_ld = a.index(combined)
    ld = UOp(Ops.LOAD, (idx_ld,))
    idx_st = a.index(combined)
    store = UOp(Ops.STORE, (idx_st, ld))
    end1 = UOp(Ops.END, (store, ridx1))
    end0 = UOp(Ops.END, (end1, ridx0))
    return [sink, a, ten, five, ridx0, ridx1, combined, idx_ld, ld, idx_st, store, end1, end0]


def build_multi_param():
    """4 params, add two and store."""
    sink = UOp(Ops.SINK, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    b = UOp.param(1, dtypes.float32, shape=(-1,))
    c = UOp.param(2, dtypes.float32, shape=(-1,))
    d = UOp.param(3, dtypes.float32, shape=(-1,))
    idx = UOp.cconst(0, dtypes.int)
    idx_a = a.index(idx)
    ld_a = UOp(Ops.LOAD, (idx_a,))
    idx_b = b.index(idx)
    ld_b = UOp(Ops.LOAD, (idx_b,))
    add = ld_a + ld_b
    idx_d = d.index(idx)
    store = UOp(Ops.STORE, (idx_d, add))
    return [sink, a, b, c, d, idx, idx_a, ld_a, idx_b, ld_b, add, idx_d, store]


def build_unary_sqrt_f32():
    """Sqrt on float32."""
    sink = UOp(Ops.SINK, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    b = UOp.param(1, dtypes.float32, shape=(-1,))
    idx = UOp.cconst(0, dtypes.int)
    idx_a = a.index(idx)
    ld = UOp(Ops.LOAD, (idx_a,))
    sq = UOp(Ops.SQRT, (ld,))
    idx_b = b.index(idx)
    store = UOp(Ops.STORE, (idx_b, sq))
    return [sink, a, b, idx, idx_a, ld, sq, idx_b, store]


def build_unary_sqrt_f16():
    """Sqrt on float16 — exercises half-precision intrinsic paths."""
    sink = UOp(Ops.SINK, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.half, shape=(-1,))
    b = UOp.param(1, dtypes.half, shape=(-1,))
    idx = UOp.cconst(0, dtypes.int)
    idx_a = a.index(idx)
    ld = UOp(Ops.LOAD, (idx_a,))
    sq = UOp(Ops.SQRT, (ld,))
    idx_b = b.index(idx)
    store = UOp(Ops.STORE, (idx_b, sq))
    return [sink, a, b, idx, idx_a, ld, sq, idx_b, store]


def build_special_dims():
    """GPU special dimensions (group_id, local_id)."""
    sink = UOp(Ops.SINK, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    bound = UOp.cconst(32, dtypes.int)
    gid = UOp(Ops.SPECIAL, (bound,), "gidx0")
    lid = UOp(Ops.SPECIAL, (bound,), "lidx0")
    combined = gid + lid
    idx_a = a.index(combined)
    ld = UOp(Ops.LOAD, (idx_a,))
    idx_st = a.index(combined)
    store = UOp(Ops.STORE, (idx_st, ld))
    return [sink, a, bound, gid, lid, combined, idx_a, ld, idx_st, store]


def build_bitcast_f32_to_i32():
    """Bitcast float32 to int32."""
    sink = UOp(Ops.SINK, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    b = UOp.param(1, dtypes.int32, shape=(-1,))
    idx = UOp.cconst(0, dtypes.int)
    idx_a = a.index(idx)
    ld = UOp(Ops.LOAD, (idx_a,))
    bc = ld.bitcast(dtypes.int32)
    idx_b = b.index(idx)
    store = UOp(Ops.STORE, (idx_b, bc))
    return [sink, a, b, idx, idx_a, ld, bc, idx_b, store]


def build_conditional():
    """If/Endif control flow."""
    sink = UOp(Ops.SINK, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    idx = UOp.cconst(0, dtypes.int)
    cond = UOp.cconst(True, dtypes.bool)
    if_op = UOp(Ops.IF, (cond,))
    idx_a = a.index(idx)
    one = UOp.cconst(1.0, dtypes.float32)
    store = UOp(Ops.STORE, (idx_a, one))
    endif = UOp(Ops.ENDIF, (if_op,))
    return [sink, a, idx, cond, if_op, idx_a, one, store, endif]


def build_const_inf_nan():
    """Special float constants: infinity and NaN."""
    import math
    sink = UOp(Ops.SINK, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    idx0 = UOp.cconst(0, dtypes.int)
    idx1 = UOp.cconst(1, dtypes.int)
    inf_val = UOp.cconst(math.inf, dtypes.float32)
    nan_val = UOp.cconst(math.nan, dtypes.float32)
    idx_a0 = a.index(idx0)
    store0 = UOp(Ops.STORE, (idx_a0, inf_val))
    idx_a1 = a.index(idx1)
    store1 = UOp(Ops.STORE, (idx_a1, nan_val))
    return [sink, a, idx0, idx1, inf_val, nan_val, idx_a0, store0, idx_a1, store1]


def build_vectorize_index():
    """Vectorize 4 floats, then index element 2."""
    sink = UOp(Ops.SINK, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    b = UOp.param(1, dtypes.float32, shape=(-1,))
    idx0 = UOp.cconst(0, dtypes.int)
    idx1 = UOp.cconst(1, dtypes.int)
    idx2 = UOp.cconst(2, dtypes.int)
    idx3 = UOp.cconst(3, dtypes.int)
    ia0 = a.index(idx0)
    ia1 = a.index(idx1)
    ia2 = a.index(idx2)
    ia3 = a.index(idx3)
    v0 = UOp(Ops.LOAD, (ia0,))
    v1 = UOp(Ops.LOAD, (ia1,))
    v2 = UOp(Ops.LOAD, (ia2,))
    v3 = UOp(Ops.LOAD, (ia3,))
    vec = UOp.stack(v0, v1, v2, v3)
    lane = vec.index(idx2).simplify()
    idx_b = b.index(idx0)
    store = UOp(Ops.STORE, (idx_b, lane))
    return [sink, a, b, idx0, idx1, idx2, idx3,
            ia0, ia1, ia2, ia3, v0, v1, v2, v3,
            vec, lane, idx_b, store]


def build_vectorize_index_scalarized():
    """Scalarized equivalent for backends that cannot render STACK lane extract."""
    sink = UOp(Ops.SINK, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    b = UOp.param(1, dtypes.float32, shape=(-1,))
    idx0 = UOp.cconst(0, dtypes.int)
    idx1 = UOp.cconst(1, dtypes.int)
    idx2 = UOp.cconst(2, dtypes.int)
    idx3 = UOp.cconst(3, dtypes.int)
    ia0 = a.index(idx0)
    ia1 = a.index(idx1)
    ia2 = a.index(idx2)
    ia3 = a.index(idx3)
    v0 = UOp(Ops.LOAD, (ia0,))
    v1 = UOp(Ops.LOAD, (ia1,))
    v2 = UOp(Ops.LOAD, (ia2,))
    v3 = UOp(Ops.LOAD, (ia3,))
    idx_b = b.index(idx0)
    store = UOp(Ops.STORE, (idx_b, v2))
    return [sink, a, b, idx0, idx1, idx2, idx3,
            ia0, ia1, ia2, ia3, v0, v1, v2, v3, idx_b, store]


# ── Main ──

TEST_CASES = [
    ("simple_add_f32", build_simple_add_f32, None),
    ("simple_mul_i32", build_simple_mul_i32, None),
    ("loop", build_loop, None),
    ("gated_load", build_gated_load, None),
    ("shared_memory", build_shared_memory, ["cuda", "metal", "opencl", "amd"]),
    ("where_select", build_where_select, None),
    ("cast_f16_to_f32", build_cast_f16_to_f32, None),
    ("nested_loops", build_nested_loops, None),
    ("multi_param", build_multi_param, None),
    ("unary_sqrt_f32", build_unary_sqrt_f32, None),
    ("unary_sqrt_f16", build_unary_sqrt_f16, None),
    ("special_dims", build_special_dims, ["cuda", "metal", "opencl", "amd"]),
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
                raise RuntimeError(f"required renderer {backend_name} is unavailable")
            renderer = RENDERERS[backend_name]
            snap_name = f"{backend_name}_{case_name}"
            try:
                if case_name == "vectorize_index" and backend_name in {"clang", "cuda"}:
                    uops = build_vectorize_index_scalarized()
                else:
                    uops = builder()
                # These are deliberately ordered renderer fixtures, not a
                # lowering pass. Typed constants now have weak CONST sources;
                # include those sources before their committed casts.
                complete = []
                seen = set()
                for u in uops:
                    if u.op is Ops.CAST and u.src[0].op is Ops.CONST and u.src[0] not in seen:
                        complete.append(u.src[0])
                        seen.add(u.src[0])
                    complete.append(u)
                    seen.add(u)
                src = renderer.render(complete).strip()
                write_expected(snap_name, src)
                total += 1
            except Exception as e:
                raise RuntimeError(f"failed to generate {snap_name}") from e

    if total == 0:
        raise RuntimeError("no reference cases were generated")
    print(f"\nDone. Generated {total} .expected files in {OUT_DIR}")


if __name__ == "__main__":
    main()
