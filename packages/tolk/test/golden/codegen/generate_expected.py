#!/usr/bin/env python3
"""Generate tinygrad reference .expected files for codegen pipeline golden tests.

Constructs kernel-level UOp DAGs (SINK-rooted) and runs them through tinygrad's
full_rewrite_to_sink + linearize + render pipeline.  This produces the reference
source code that Tolk's Pipeline.full_rewrite_to_sink must match.

Usage:
    python3 packages/tolk/test/golden/codegen/generate_expected.py \
      --tinygrad _plans/tinygrad-a83c6f801 --output _plans/goldens-a83/codegen

Generate separately and review differences before updating expectations.
"""

import argparse
import os
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tinygrad", type=Path, default=HERE.parents[4] / "_tinygrad")
parser.add_argument("--output", type=Path, default=HERE)
args = parser.parse_args()
sys.path.insert(0, str(args.tinygrad.resolve()))
for key in ("DEBUG", "VIZ", "PROFILE"):
    os.environ.pop(key, None)

# Disable ANSI color in the reference — auto-generated kernel names embed ANSI
# escape codes per axis type, which would leak into rendered source and the
# KernelInfo.name used to look up model kernels.
os.environ["NO_COLOR"] = "1"

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.codegen import full_rewrite_to_sink, do_linearize
from tinygrad.renderer.cstyle import (
    ClangRenderer,
    CUDARenderer,
    HIPRenderer,
    MetalRenderer,
    OpenCLRenderer,
)
from tinygrad import Tensor, nn
from extra.models.llama import Transformer

OUT_DIR = args.output
OUT_DIR.mkdir(parents=True, exist_ok=True)


def render_only(ctor, target, compiler):
    # Keep the target's initializer and matchers; only compiler construction
    # is disabled because these fixtures render without compiling or executing.
    with patch(compiler, return_value=None):
        return ctor(target)


RENDERERS = {}
for _name, _ctor in [
    ("clang", lambda: ClangRenderer(Target("CPU", arch="x86_64,znver2"))),
    ("cuda", lambda: render_only(CUDARenderer, Target("CUDA", arch="sm_80"),
                                  "tinygrad.runtime.support.compiler_cuda.NVRTCCompiler")),
    ("metal", lambda: MetalRenderer(Target("METAL", arch="Apple7"))),
    ("opencl", lambda: OpenCLRenderer(Target("CL"))),
    # Keep backend order aligned with the existing reference corpus.
    ("amd", lambda: render_only(HIPRenderer, Target("AMD", arch="gfx1100"),
                                 "tinygrad.runtime.support.compiler_amd.HIPCompiler")),
]:
    try:
        RENDERERS[_name] = _ctor()
    except Exception as e:
        raise RuntimeError(f"required {_name} renderer failed to initialize") from e


def write_expected(name, content):
    path = os.path.join(OUT_DIR, f"{name}.expected")
    with open(path, "w") as f:
        f.write(content + "\n")
    print(f"  wrote {path}")


def get_source(sink, renderer, optimize=True):
    """Run the full tinygrad codegen pipeline and return rendered source."""
    # Match generate_actual.ml: hand-built GPU ranges are software loops on CPU.
    if not renderer.has_local:
        sink = sink.substitute({u: u.replace(arg=(*u.axis_id, AxisType.WEAK))
                                for u in sink.toposort()
                                if u.op is Ops.RANGE and u.axis_type is AxisType.GLOBAL})
    rewritten = full_rewrite_to_sink(sink, renderer, optimize=optimize)
    program = do_linearize(renderer, UOp(Ops.PROGRAM), rewritten)
    return renderer.render(list(program.src[1].src)).strip()


def ki(name="test", **kwargs):
    """Build a KernelInfo with deterministic defaults.

    Explicit names keep hand-built fixtures identifiable after optimization;
    model fixtures retain the target's automatically derived kernel names.
    """
    defaults = dict(name=name, opts_to_apply=())
    defaults.update(kwargs)
    return KernelInfo(**defaults)


# ── Kernel AST builders ──
# Each builds a SINK-rooted kernel DAG matching the equivalent Tolk Kernel.t
# construction in generate_actual.ml.


def build_elementwise_add():
    """c[i] = a[i] + b[i], 1 Global range."""
    p0 = UOp.param(0, dtypes.float32, shape=(-1,))
    p1 = UOp.param(1, dtypes.float32, shape=(-1,))
    p2 = UOp.param(2, dtypes.float32, shape=(-1,))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld_a = p0.index(r0).load()
    ld_b = p1.index(r0).load()
    add = ld_a + ld_b
    st = p2.index(r0).store(add)
    end = st.end(r0)
    return UOp.sink(end, arg=ki("elementwise_add"))


def build_sum_reduce():
    """b[0] = sum(a[i]), 1 Reduce range."""
    p0 = UOp.param(0, dtypes.float32, shape=(-1,))
    p1 = UOp.param(1, dtypes.float32, shape=(-1,))
    r0 = UOp.range(256, 0, AxisType.REDUCE)
    ld = p0.index(r0).load()
    red = UOp(Ops.REDUCE, src=(ld, r0), arg=(Ops.ADD, 0))
    c0 = UOp.const(0, dtypes.int)
    st = p1.index(c0).store(red)
    return UOp.sink(st, arg=ki("sum_reduce"))


def build_max_reduce():
    """b[0] = max(a[i]), 1 Reduce range."""
    p0 = UOp.param(0, dtypes.float32, shape=(-1,))
    p1 = UOp.param(1, dtypes.float32, shape=(-1,))
    r0 = UOp.range(64, 0, AxisType.REDUCE)
    ld = p0.index(r0).load()
    red = UOp(Ops.REDUCE, src=(ld, r0), arg=(Ops.MAX, 0))
    c0 = UOp.const(0, dtypes.int)
    st = p1.index(c0).store(red)
    return UOp.sink(st, arg=ki("max_reduce"))


def build_dot_product():
    """c[0] = sum_k(a[k] * b[k]), 1 Reduce range."""
    p0 = UOp.param(0, dtypes.float32, shape=(-1,))
    p1 = UOp.param(1, dtypes.float32, shape=(-1,))
    p2 = UOp.param(2, dtypes.float32, shape=(-1,))
    r0 = UOp.range(128, 0, AxisType.REDUCE)
    ld_a = p0.index(r0).load()
    ld_b = p1.index(r0).load()
    mul = ld_a * ld_b
    red = UOp(Ops.REDUCE, src=(mul, r0), arg=(Ops.ADD, 0))
    c0 = UOp.const(0, dtypes.int)
    st = p2.index(c0).store(red)
    return UOp.sink(st, arg=ki("dot_product"))


def build_matmul_small():
    """C[i*4+j] = sum_k(A[i*4+k] * B[k*4+j]), M=N=K=4."""
    M, N, K = 4, 4, 4
    pA = UOp.param(0, dtypes.float32, shape=(-1,))
    pB = UOp.param(1, dtypes.float32, shape=(-1,))
    pC = UOp.param(2, dtypes.float32, shape=(-1,))
    ri = UOp.range(M, 0, AxisType.GLOBAL)
    rj = UOp.range(N, 1, AxisType.GLOBAL)
    rk = UOp.range(K, 2, AxisType.REDUCE)
    a_idx = ri * K + rk
    b_idx = rk * N + rj
    c_idx = ri * N + rj
    ld_a = pA.index(a_idx).load()
    ld_b = pB.index(b_idx).load()
    mul = ld_a * ld_b
    red = UOp(Ops.REDUCE, src=(mul, rk), arg=(Ops.ADD, 0))
    st = pC.index(c_idx).store(red)
    end = st.end(ri, rj)
    return UOp.sink(
        end,
        arg=ki(
            "matmul_small",
        ),
    )


def build_elementwise_2d():
    """c[i*16+j] = a[i*16+j] + b[i*16+j], 2 Global ranges."""
    ROWS, COLS = 8, 16
    p0 = UOp.param(0, dtypes.float32, shape=(-1,))
    p1 = UOp.param(1, dtypes.float32, shape=(-1,))
    p2 = UOp.param(2, dtypes.float32, shape=(-1,))
    ri = UOp.range(ROWS, 0, AxisType.GLOBAL)
    rj = UOp.range(COLS, 1, AxisType.GLOBAL)
    flat = ri * COLS + rj
    ld_a = p0.index(flat).load()
    ld_b = p1.index(flat).load()
    add = ld_a + ld_b
    st = p2.index(flat).store(add)
    end = st.end(ri, rj)
    return UOp.sink(
        end, arg=ki("elementwise_2d")
    )


def build_reduce_rows():
    """b[i] = sum_j(a[i*32+j]), 1 Global + 1 Reduce range."""
    ROWS, COLS = 8, 32
    p0 = UOp.param(0, dtypes.float32, shape=(-1,))
    p1 = UOp.param(1, dtypes.float32, shape=(-1,))
    ri = UOp.range(ROWS, 0, AxisType.GLOBAL)
    rj = UOp.range(COLS, 1, AxisType.REDUCE)
    flat = ri * COLS + rj
    ld = p0.index(flat).load()
    red = UOp(Ops.REDUCE, src=(ld, rj), arg=(Ops.ADD, 0))
    st = p1.index(ri).store(red)
    end = st.end(ri)
    return UOp.sink(
        end, arg=ki("reduce_rows")
    )


def build_multi_output():
    """b[i] = a[i] + 1.0; c[i] = a[i] * 2.0, 1 Global range, 2 stores."""
    p0 = UOp.param(0, dtypes.float32, shape=(-1,))
    p1 = UOp.param(1, dtypes.float32, shape=(-1,))
    p2 = UOp.param(2, dtypes.float32, shape=(-1,))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld_a = p0.index(r0).load()
    st1 = p1.index(r0).store(ld_a + UOp.const(1.0, dtypes.float32))
    st2 = p2.index(r0).store(ld_a * UOp.const(2.0, dtypes.float32))
    end = UOp.group(st1, st2).end(r0)
    return UOp.sink(end, arg=ki("multi_output"))


def build_gated_store():
    """c[i] = a[i] + b[i] with store gated by i < 200, range size=256."""
    p0 = UOp.param(0, dtypes.float32, shape=(-1,))
    p1 = UOp.param(1, dtypes.float32, shape=(-1,))
    p2 = UOp.param(2, dtypes.float32, shape=(-1,))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld_a = p0.index(r0).load()
    ld_b = p1.index(r0).load()
    add = ld_a + ld_b
    gate = r0 < UOp.const(200, dtypes.weakint)
    st = p2.index(r0).store(
        gate.where(add, UOp.invalid())
    )
    end = st.end(r0)
    return UOp.sink(end, arg=ki("gated_store"))


# ── Test cases ──
# (name, builder, backends_or_None, optimize)

GPU_RENDERERS = ["cuda", "metal", "opencl", "amd"]


def build_no_optimize():
    """Same as elementwise_add but with optimize=false and unique name."""
    p0 = UOp.param(0, dtypes.float32, shape=(-1,))
    p1 = UOp.param(1, dtypes.float32, shape=(-1,))
    p2 = UOp.param(2, dtypes.float32, shape=(-1,))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld_a = p0.index(r0).load()
    ld_b = p1.index(r0).load()
    add = ld_a + ld_b
    st = p2.index(r0).store(add)
    end = st.end(r0)
    return UOp.sink(end, arg=ki("no_optimize"))


def build_elementwise_where():
    """c[i] = (a[i] > 0) ? a[i] : 0.0 (ReLU pattern), 1 Global range."""
    p0 = UOp.param(0, dtypes.float32, shape=(-1,))
    p1 = UOp.param(1, dtypes.float32, shape=(-1,))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld = p0.index(r0).load()
    zero = UOp.const(0.0, dtypes.float32)
    cond = zero.alu(Ops.CMPLT, ld)  # 0.0 < a[i] => a[i] > 0
    val = cond.where(ld, zero)
    st = p1.index(r0).store(val)
    end = st.end(r0)
    return UOp.sink(end, arg=ki("elementwise_where"))


def build_elementwise_cast_f16():
    """c[i] = (float32)a_f16[i] + b[i], 1 Global range, mixed dtypes."""
    p0 = UOp.param(0, dtypes.half, shape=(-1,))
    p1 = UOp.param(1, dtypes.float32, shape=(-1,))
    p2 = UOp.param(2, dtypes.float32, shape=(-1,))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld_a = p0.index(r0).load()
    cast_a = ld_a.cast(dtypes.float32)
    ld_b = p1.index(r0).load()
    add = cast_a + ld_b
    st = p2.index(r0).store(add)
    end = st.end(r0)
    return UOp.sink(end, arg=ki("elementwise_cast_f16"))


def build_elementwise_sqrt():
    """c[i] = sqrt(a[i]), 1 Global range, exercises unary SQRT through pipeline."""
    p0 = UOp.param(0, dtypes.float32, shape=(-1,))
    p1 = UOp.param(1, dtypes.float32, shape=(-1,))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld = p0.index(r0).load()
    sq = UOp(Ops.SQRT, src=(ld,))
    st = p1.index(r0).store(sq)
    end = st.end(r0)
    return UOp.sink(end, arg=ki("elementwise_sqrt"))


def build_parallel_reduce():
    """b[0] = sum(a[i]); c[0] = sum(a[i]*a[i]), 1 Reduce range, 2 stores."""
    p0 = UOp.param(0, dtypes.float32, shape=(-1,))
    p1 = UOp.param(1, dtypes.float32, shape=(-1,))
    p2 = UOp.param(2, dtypes.float32, shape=(-1,))
    r0 = UOp.range(128, 0, AxisType.REDUCE)
    ld = p0.index(r0).load()
    red1 = UOp(Ops.REDUCE, src=(ld, r0), arg=(Ops.ADD, 0))
    red2 = UOp(Ops.REDUCE, src=(ld * ld, r0), arg=(Ops.ADD, 0))
    c0 = UOp.const(0, dtypes.int)
    st1 = p1.index(c0).store(red1)
    st2 = p2.index(c0).store(red2)
    return UOp.sink(st1, st2, arg=ki("parallel_reduce"))


def build_elementwise_int32():
    """c[i] = a[i] + b[i] (all int32), 1 Global range."""
    p0 = UOp.param(0, dtypes.int32, shape=(-1,))
    p1 = UOp.param(1, dtypes.int32, shape=(-1,))
    p2 = UOp.param(2, dtypes.int32, shape=(-1,))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld_a = p0.index(r0).load()
    ld_b = p1.index(r0).load()
    add = ld_a + ld_b
    st = p2.index(r0).store(add)
    end = st.end(r0)
    return UOp.sink(end, arg=ki("elementwise_int32"))


def build_lorenz_fold():
    """Short Euler Lorenz fold. Reuses a constant-scaled difference that the
    next step negates, exercising the negation/const-fold canonicalization."""
    px = UOp.param(0, dtypes.float32, shape=(-1,))
    py = UOp.param(1, dtypes.float32, shape=(-1,))
    pz = UOp.param(2, dtypes.float32, shape=(-1,))
    po = UOp.param(3, dtypes.float32, shape=(-1,))
    r0 = UOp.range(16, 0, AxisType.GLOBAL)
    x = px.index(r0).load()
    y = py.index(r0).load()
    z = pz.index(r0).load()
    sigma = UOp.const(10.0, dtypes.float32)
    rho = UOp.const(28.0, dtypes.float32)
    beta = UOp.const(2.5, dtypes.float32)
    dt = UOp.const(0.0625, dtypes.float32)
    for _ in range(3):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x, y, z = x + dt * dx, y + dt * dy, z + dt * dz
    st = po.index(r0).store((x + y) + z)
    end = st.end(r0)
    return UOp.sink(end, arg=ki("lorenz_fold"))


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
    ("lorenz_fold", build_lorenz_fold, None, True),
    ("llama_embedding", build_llama_embedding, None, True),
    ("llama_rmsnorm", build_llama_rmsnorm, None, True),
    ("llama_ffn_gate", build_llama_ffn_gate, None, True),
    ("llama_vector_scale", build_llama_vector_scale, None, True),
    ("llama_output_projection", build_llama_output_projection, None, True),
]


def main():
    only = os.environ.get("ONLY")
    total = 0
    for case_name, builder, backends, optimize in TEST_CASES:
        if only is not None and case_name != only:
            continue
        print(f"\n{case_name} (optimize={optimize}):")
        sink = builder()
        targets = backends if backends else list(RENDERERS.keys())
        for backend_name in targets:
            if backend_name not in RENDERERS:
                raise RuntimeError(f"required renderer {backend_name} is unavailable")
            renderer = RENDERERS[backend_name]
            snap_name = f"{backend_name}_{case_name}"
            try:
                src = get_source(sink, renderer, optimize=optimize)
                write_expected(snap_name, src)
                total += 1
            except Exception as e:
                raise RuntimeError(f"failed to generate {snap_name}") from e

    if total == 0:
        raise RuntimeError("no reference cases were generated")
    print(f"\nDone. Generated {total} .expected files in {OUT_DIR}")


if __name__ == "__main__":
    main()
