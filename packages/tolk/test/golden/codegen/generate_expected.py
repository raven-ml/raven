#!/usr/bin/env python3
"""Generate tinygrad reference .expected files for codegen pipeline golden tests.

Constructs kernel-level UOp DAGs (SINK-rooted) and runs them through tinygrad's
full_rewrite_to_sink + linearize + render pipeline.  This produces the reference
source code that Tolk's Pipeline.full_rewrite_to_sink must match.

Usage:
    uv run packages/tolk/test/golden/codegen/generate_expected.py

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

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType, ParamArg
from tinygrad.dtype import dtypes, Invalid
from tinygrad.helpers import Target
from tinygrad.codegen import full_rewrite_to_sink, line_rewrite, pm_linearize_cleanups
from tinygrad.codegen.late.linearizer import linearize
from tinygrad.renderer.cstyle import (
    ClangRenderer,
    CUDARenderer,
    MetalRenderer,
    OpenCLRenderer,
)
import tinygrad.renderer.cstyle as _cstyle_mod
from tinygrad import Tensor, nn
from extra.models.llama import Transformer

OUT_DIR = os.path.dirname(__file__)


class _RenderOnlyCUDARenderer(CUDARenderer):
    """CUDARenderer that skips compiler init (nvrtc not needed for rendering)."""

    def __init__(self, target):
        self.target, self.compiler = target, None
        arch = target.arch
        ver = int(arch[3:])
        tc = _cstyle_mod.tc
        self.tensor_cores = (
            tc.cuda_sm89 if ver >= 89
            else tc.cuda_sm80 if ver >= 80
            else tc.cuda_sm75 if ver >= 75
            else []
        )


RENDERERS = {}
for _name, _ctor in [
    ("clang", lambda: ClangRenderer(Target("CPU", arch="x86_64,znver2"))),
    ("cuda", lambda: _RenderOnlyCUDARenderer(Target("CUDA", arch="sm_80"))),
    ("metal", lambda: MetalRenderer(Target("METAL"))),
    ("opencl", lambda: OpenCLRenderer(Target("CL"))),
]:
    try:
        RENDERERS[_name] = _ctor()
    except Exception as e:
        print(f"WARNING: skipping {_name} renderer: {e}")


def write_expected(name, content):
    path = os.path.join(OUT_DIR, f"{name}.expected")
    with open(path, "w") as f:
        f.write(content + "\n")
    print(f"  wrote {path}")


def get_source(sink, renderer, optimize=True):
    """Run the full tinygrad codegen pipeline and return rendered source."""
    rewritten = full_rewrite_to_sink(sink, renderer, optimize=optimize)
    lst = linearize(rewritten)
    lst = line_rewrite(lst, pm_linearize_cleanups)
    return renderer.render(lst).strip()


def ki(name="test", **kwargs):
    """Build a KernelInfo with deterministic defaults.

    Using name != "test" forces apply_opts to preserve the name rather than
    auto-generating one with a global counter, which avoids order-dependent
    naming mismatches between the Python and OCaml generators.
    """
    defaults = dict(name=name, axis_types=(), opts_to_apply=())
    defaults.update(kwargs)
    return KernelInfo(**defaults)


# ── Kernel AST builders ──
# Each builds a SINK-rooted kernel DAG matching the equivalent Tolk Kernel.t
# construction in generate_actual.ml.


def build_elementwise_add():
    """c[i] = a[i] + b[i], 1 Global range."""
    p0 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    p1 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    p2 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(2))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld_a = p0.index(r0, ptr=True).load()
    ld_b = p1.index(r0, ptr=True).load()
    add = ld_a + ld_b
    st = p2.index(r0, ptr=True).store(add)
    end = st.end(r0)
    return UOp.sink(end, arg=ki("elementwise_add", axis_types=(AxisType.GLOBAL,)))


def build_sum_reduce():
    """b[0] = sum(a[i]), 1 Reduce range."""
    p0 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    p1 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    r0 = UOp.range(256, 0, AxisType.REDUCE)
    ld = p0.index(r0, ptr=True).load()
    red = UOp(Ops.REDUCE, dtypes.float32, (ld, r0), (Ops.ADD, ()))
    c0 = UOp.const(dtypes.int, 0)
    st = p1.index(c0, ptr=True).store(red)
    return UOp.sink(st, arg=ki("sum_reduce", axis_types=(AxisType.REDUCE,)))


def build_max_reduce():
    """b[0] = max(a[i]), 1 Reduce range."""
    p0 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    p1 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    r0 = UOp.range(64, 0, AxisType.REDUCE)
    ld = p0.index(r0, ptr=True).load()
    red = UOp(Ops.REDUCE, dtypes.float32, (ld, r0), (Ops.MAX, ()))
    c0 = UOp.const(dtypes.int, 0)
    st = p1.index(c0, ptr=True).store(red)
    return UOp.sink(st, arg=ki("max_reduce", axis_types=(AxisType.REDUCE,)))


def build_dot_product():
    """c[0] = sum_k(a[k] * b[k]), 1 Reduce range."""
    p0 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    p1 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    p2 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(2))
    r0 = UOp.range(128, 0, AxisType.REDUCE)
    ld_a = p0.index(r0, ptr=True).load()
    ld_b = p1.index(r0, ptr=True).load()
    mul = ld_a * ld_b
    red = UOp(Ops.REDUCE, dtypes.float32, (mul, r0), (Ops.ADD, ()))
    c0 = UOp.const(dtypes.int, 0)
    st = p2.index(c0, ptr=True).store(red)
    return UOp.sink(st, arg=ki("dot_product", axis_types=(AxisType.REDUCE,)))


def build_matmul_small():
    """C[i*4+j] = sum_k(A[i*4+k] * B[k*4+j]), M=N=K=4."""
    M, N, K = 4, 4, 4
    pA = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    pB = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    pC = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(2))
    ri = UOp.range(M, 0, AxisType.GLOBAL)
    rj = UOp.range(N, 1, AxisType.GLOBAL)
    rk = UOp.range(K, 2, AxisType.REDUCE)
    a_idx = ri * K + rk
    b_idx = rk * N + rj
    c_idx = ri * N + rj
    ld_a = pA.index(a_idx, ptr=True).load()
    ld_b = pB.index(b_idx, ptr=True).load()
    mul = ld_a * ld_b
    red = UOp(Ops.REDUCE, dtypes.float32, (mul, rk), (Ops.ADD, ()))
    st = pC.index(c_idx, ptr=True).store(red)
    end = st.end(ri, rj)
    return UOp.sink(
        end,
        arg=ki(
            "matmul_small",
            axis_types=(AxisType.GLOBAL, AxisType.GLOBAL, AxisType.REDUCE),
        ),
    )


def build_elementwise_2d():
    """c[i*16+j] = a[i*16+j] + b[i*16+j], 2 Global ranges."""
    ROWS, COLS = 8, 16
    p0 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    p1 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    p2 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(2))
    ri = UOp.range(ROWS, 0, AxisType.GLOBAL)
    rj = UOp.range(COLS, 1, AxisType.GLOBAL)
    flat = ri * COLS + rj
    ld_a = p0.index(flat, ptr=True).load()
    ld_b = p1.index(flat, ptr=True).load()
    add = ld_a + ld_b
    st = p2.index(flat, ptr=True).store(add)
    end = st.end(ri, rj)
    return UOp.sink(
        end, arg=ki("elementwise_2d", axis_types=(AxisType.GLOBAL, AxisType.GLOBAL))
    )


def build_reduce_rows():
    """b[i] = sum_j(a[i*32+j]), 1 Global + 1 Reduce range."""
    ROWS, COLS = 8, 32
    p0 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    p1 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    ri = UOp.range(ROWS, 0, AxisType.GLOBAL)
    rj = UOp.range(COLS, 1, AxisType.REDUCE)
    flat = ri * COLS + rj
    ld = p0.index(flat, ptr=True).load()
    red = UOp(Ops.REDUCE, dtypes.float32, (ld, rj), (Ops.ADD, ()))
    st = p1.index(ri, ptr=True).store(red)
    end = st.end(ri)
    return UOp.sink(
        end, arg=ki("reduce_rows", axis_types=(AxisType.GLOBAL, AxisType.REDUCE))
    )


def build_multi_output():
    """b[i] = a[i] + 1.0; c[i] = a[i] * 2.0, 1 Global range, 2 stores."""
    p0 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    p1 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    p2 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(2))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld_a = p0.index(r0, ptr=True).load()
    st1 = p1.index(r0, ptr=True).store(ld_a + UOp.const(dtypes.float32, 1.0))
    e1 = st1.end(r0)
    st2 = p2.index(r0, ptr=True).store(ld_a * UOp.const(dtypes.float32, 2.0))
    e2 = st2.end(r0)
    return UOp.sink(e1, e2, arg=ki("multi_output", axis_types=(AxisType.GLOBAL,)))


def build_gated_store():
    """c[i] = a[i] + b[i] with store gated by i < 200, range size=256."""
    p0 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    p1 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    p2 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(2))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld_a = p0.index(r0, ptr=True).load()
    ld_b = p1.index(r0, ptr=True).load()
    add = ld_a + ld_b
    gate = r0 < UOp.const(dtypes.int, 200)
    st = p2.index(r0, ptr=True).store(
        gate.where(add, UOp(Ops.CONST, dtypes.float32, (), Invalid))
    )
    end = st.end(r0)
    return UOp.sink(end, arg=ki("gated_store", axis_types=(AxisType.GLOBAL,)))


# ── Test cases ──
# (name, builder, backends_or_None, optimize)

GPU_RENDERERS = ["cuda", "metal", "opencl"]


def build_no_optimize():
    """Same as elementwise_add but with optimize=false and unique name."""
    p0 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    p1 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    p2 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(2))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld_a = p0.index(r0, ptr=True).load()
    ld_b = p1.index(r0, ptr=True).load()
    add = ld_a + ld_b
    st = p2.index(r0, ptr=True).store(add)
    end = st.end(r0)
    return UOp.sink(end, arg=ki("no_optimize", axis_types=(AxisType.GLOBAL,)))


def build_elementwise_where():
    """c[i] = (a[i] > 0) ? a[i] : 0.0 (ReLU pattern), 1 Global range."""
    p0 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    p1 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld = p0.index(r0, ptr=True).load()
    zero = UOp.const(dtypes.float32, 0.0)
    cond = zero.alu(Ops.CMPLT, ld)  # 0.0 < a[i] => a[i] > 0
    val = cond.where(ld, zero)
    st = p1.index(r0, ptr=True).store(val)
    end = st.end(r0)
    return UOp.sink(end, arg=ki("elementwise_where", axis_types=(AxisType.GLOBAL,)))


def build_elementwise_cast_f16():
    """c[i] = (float32)a_f16[i] + b[i], 1 Global range, mixed dtypes."""
    p0 = UOp(Ops.PARAM, dtypes.half.ptr(), (), ParamArg(0))
    p1 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    p2 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(2))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld_a = p0.index(r0, ptr=True).load()
    cast_a = UOp(Ops.CAST, dtypes.float32, (ld_a,))
    ld_b = p1.index(r0, ptr=True).load()
    add = cast_a + ld_b
    st = p2.index(r0, ptr=True).store(add)
    end = st.end(r0)
    return UOp.sink(end, arg=ki("elementwise_cast_f16", axis_types=(AxisType.GLOBAL,)))


def build_elementwise_sqrt():
    """c[i] = sqrt(a[i]), 1 Global range, exercises unary SQRT through pipeline."""
    p0 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    p1 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld = p0.index(r0, ptr=True).load()
    sq = UOp(Ops.SQRT, dtypes.float32, (ld,))
    st = p1.index(r0, ptr=True).store(sq)
    end = st.end(r0)
    return UOp.sink(end, arg=ki("elementwise_sqrt", axis_types=(AxisType.GLOBAL,)))


def build_parallel_reduce():
    """b[0] = sum(a[i]); c[0] = sum(a[i]*a[i]), 1 Reduce range, 2 stores."""
    p0 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    p1 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    p2 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(2))
    r0 = UOp.range(128, 0, AxisType.REDUCE)
    ld = p0.index(r0, ptr=True).load()
    red1 = UOp(Ops.REDUCE, dtypes.float32, (ld, r0), (Ops.ADD, ()))
    red2 = UOp(Ops.REDUCE, dtypes.float32, (ld * ld, r0), (Ops.ADD, ()))
    c0 = UOp.const(dtypes.int, 0)
    st1 = p1.index(c0, ptr=True).store(red1)
    st2 = p2.index(c0, ptr=True).store(red2)
    return UOp.sink(st1, st2, arg=ki("parallel_reduce", axis_types=(AxisType.REDUCE,)))


def build_elementwise_int32():
    """c[i] = a[i] + b[i] (all int32), 1 Global range."""
    p0 = UOp(Ops.PARAM, dtypes.int32.ptr(), (), ParamArg(0))
    p1 = UOp(Ops.PARAM, dtypes.int32.ptr(), (), ParamArg(1))
    p2 = UOp(Ops.PARAM, dtypes.int32.ptr(), (), ParamArg(2))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld_a = p0.index(r0, ptr=True).load()
    ld_b = p1.index(r0, ptr=True).load()
    add = ld_a + ld_b
    st = p2.index(r0, ptr=True).store(add)
    end = st.end(r0)
    return UOp.sink(end, arg=ki("elementwise_int32", axis_types=(AxisType.GLOBAL,)))


_LLAMA_MODEL_SINKS = None


def llama_model_sinks():
    """Kernels scheduled from tinygrad's own LLaMA/Qwen-family model code."""
    global _LLAMA_MODEL_SINKS
    if _LLAMA_MODEL_SINKS is not None:
        return _LLAMA_MODEL_SINKS

    model = Transformer(
        dim=8,
        hidden_dim=16,
        n_heads=2,
        n_kv_heads=1,
        n_layers=1,
        norm_eps=1e-5,
        vocab_size=32,
        max_context=8,
        jit=False,
        disable_kv_cache=True,
    )
    for param in nn.state.get_parameters(model):
        param.replace(Tensor.empty(param.shape, dtype=param.dtype))

    tokens = Tensor.empty(1, 2, dtype=dtypes.int)
    logits = model.forward(tokens, 0, float("nan"), 0, 1.0, 0.0, 0.0)
    linear = logits.schedule_linear(*nn.state.get_parameters(model))
    sinks = []
    seen = set()
    for call in linear.src:
        if call.op is not Ops.CALL or call.src[0].op is not Ops.SINK:
            continue
        sink = call.src[0]
        if sink.key not in seen:
            sinks.append(sink)
            seen.add(sink.key)

    targets = {
        "E_8_2": "llama_embedding",
        "r_2_8": "llama_rmsnorm",
        "r_2_8_8": "llama_ffn_gate",
        "E_2_2_4": "llama_vector_scale",
        "r_2_32_8": "llama_output_projection",
    }
    _LLAMA_MODEL_SINKS = {}
    for sink in sinks:
        rewritten = full_rewrite_to_sink(sink, RENDERERS["clang"], optimize=True)
        case_name = targets.get(rewritten.arg.name)
        if case_name is not None:
            _LLAMA_MODEL_SINKS[case_name] = sink

    missing = sorted(set(targets.values()) - set(_LLAMA_MODEL_SINKS))
    if missing:
        raise RuntimeError(f"tinygrad LLaMA kernels not found: {missing}")
    return _LLAMA_MODEL_SINKS


def build_llama_embedding():
    return llama_model_sinks()["llama_embedding"]


def build_llama_rmsnorm():
    return llama_model_sinks()["llama_rmsnorm"]


def build_llama_ffn_gate():
    return llama_model_sinks()["llama_ffn_gate"]


def build_llama_vector_scale():
    return llama_model_sinks()["llama_vector_scale"]


def build_llama_output_projection():
    return llama_model_sinks()["llama_output_projection"]


TEST_CASES = [
    ("elementwise_add", build_elementwise_add, None, True),
    ("sum_reduce", build_sum_reduce, None, True),
    ("max_reduce", build_max_reduce, None, True),
    ("dot_product", build_dot_product, None, True),
    ("matmul_small", build_matmul_small, GPU_RENDERERS, True),
    ("elementwise_2d", build_elementwise_2d, GPU_RENDERERS, True),
    ("reduce_rows", build_reduce_rows, None, True),
    ("no_optimize", build_no_optimize, None, False),
    ("multi_output", build_multi_output, None, True),
    ("gated_store", build_gated_store, None, True),
    ("elementwise_where", build_elementwise_where, None, True),
    ("elementwise_cast_f16", build_elementwise_cast_f16, None, True),
    ("elementwise_sqrt", build_elementwise_sqrt, None, True),
    ("parallel_reduce", build_parallel_reduce, None, True),
    ("elementwise_int32", build_elementwise_int32, None, True),
    ("llama_embedding", build_llama_embedding, None, True),
    ("llama_rmsnorm", build_llama_rmsnorm, None, True),
    ("llama_ffn_gate", build_llama_ffn_gate, None, True),
    ("llama_vector_scale", build_llama_vector_scale, None, True),
    ("llama_output_projection", build_llama_output_projection, None, True),
]


def main():
    total = 0
    for case_name, builder, backends, optimize in TEST_CASES:
        print(f"\n{case_name} (optimize={optimize}):")
        sink = builder()
        targets = backends if backends else list(RENDERERS.keys())
        for backend_name in targets:
            if backend_name not in RENDERERS:
                print(f"  SKIP {backend_name}_{case_name}: renderer not available")
                continue
            renderer = RENDERERS[backend_name]
            snap_name = f"{backend_name}_{case_name}"
            try:
                src = get_source(sink, renderer, optimize=optimize)
                write_expected(snap_name, src)
                total += 1
            except Exception as e:
                print(f"  FAIL {snap_name}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nDone. Generated {total} .expected files in {OUT_DIR}")


if __name__ == "__main__":
    main()
