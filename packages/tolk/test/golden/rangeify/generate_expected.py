#!/usr/bin/env python3
"""Generate tinygrad reference .expected files for rangeify pipeline golden tests.

Constructs tensor-level UOp DAGs and runs them through tinygrad's
get_kernel_graph + full_rewrite_to_sink + linearize + render pipeline.
This produces the reference source code that Tolk's
Rangeify.get_kernel_graph -> Pipeline -> Linearizer -> Renderer must match.

Usage:
    uv run packages/tolk/test/golden/rangeify/generate_expected.py

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
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.schedule.rangeify import get_kernel_graph
from tinygrad.codegen import full_rewrite_to_sink, line_rewrite, pm_linearize_cleanups
from tinygrad.codegen.opt.postrange import Scheduler
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


def render_kernel(ast, renderer, optimize=True):
    """Run full codegen pipeline on a kernel AST and return rendered source."""
    rewritten = full_rewrite_to_sink(ast, renderer, optimize=optimize)
    lst = linearize(rewritten)
    lst = line_rewrite(lst, pm_linearize_cleanups)
    return renderer.render(lst).strip()


def get_source(sink, renderer, optimize=True):
    """Build tensor graph, rangeify, codegen, render all kernels."""
    kg = get_kernel_graph(sink)
    sources = []
    for u in kg.toposort():
        if u.op is Ops.CALL and isinstance(u.src[0].arg, KernelInfo):
            sources.append(render_kernel(u.src[0], renderer, optimize))
    return "\n---\n".join(sources)


# ── Helpers ──


def mk_shape(*dims):
    """Encode a shape as a VECTORIZE of index consts (or single const for 1-D)."""
    if len(dims) == 1:
        return UOp.const(dtypes.int, dims[0])
    return UOp.vectorize(*(UOp.const(dtypes.int, d) for d in dims))


def mk_param(slot, *shape, dtype=dtypes.float32):
    """Build a PARAM with a known shape and CPU device."""
    return UOp.param(slot, dtype, shape=shape, device="CPU")


def wrap_sink(*srcs):
    """Wrap source(s) in CONTIGUOUS -> SINK."""
    contigs = [UOp(Ops.CONTIGUOUS, s.dtype, (s,)) for s in srcs]
    return UOp.sink(*contigs)


# ── Tensor graph builders ──
# Each builds a tensor-level SINK-rooted graph that get_kernel_graph will
# transform into kernel(s). These match the Tolk generate_actual.ml builders.


def build_elementwise_add():
    """c = a + b, shape [256]."""
    a = mk_param(0, 256)
    b = mk_param(1, 256)
    return wrap_sink(a + b)


def build_elementwise_3way():
    """d = a + b + c, shape [256]."""
    a = mk_param(0, 256)
    b = mk_param(1, 256)
    c = mk_param(2, 256)
    return wrap_sink(a + b + c)


def build_mulacc():
    """c = sum(a * b), shape [256] -> scalar."""
    a = mk_param(0, 256)
    b = mk_param(1, 256)
    mul = a * b
    red = UOp(Ops.REDUCE, dtypes.float32, (mul,), (Ops.ADD, (0,)))
    return wrap_sink(red)


def build_binop_reshape():
    """d = (a + b).reshape(5, 2) + c."""
    a = mk_param(0, 10)
    b = mk_param(1, 10)
    c = mk_param(2, 5, 2)
    add = a + b
    reshaped = UOp(Ops.RESHAPE, dtypes.float32, (add, mk_shape(5, 2)))
    return wrap_sink(reshaped + c)


def build_binop_permute():
    """d = (a + b).permute(1, 0) + c."""
    a = mk_param(0, 2, 5)
    b = mk_param(1, 2, 5)
    c = mk_param(2, 5, 2)
    add = a + b
    permed = UOp(Ops.PERMUTE, dtypes.float32, (add,), (1, 0))
    return wrap_sink(permed + c)


def build_diamond():
    """e = (a+b+c) + (a+b+d), shared subexpression a+b."""
    a = mk_param(0, 10)
    b = mk_param(1, 10)
    c = mk_param(2, 10)
    d = mk_param(3, 10)
    ab = a + b
    return wrap_sink(ab + c + ab + d)


def build_reduce_unary():
    """c = neg(sqrt(sum(a))), shape [16] -> scalar."""
    a = mk_param(0, 16)
    red = UOp(Ops.REDUCE, dtypes.float32, (a,), (Ops.ADD, (0,)))
    sq = UOp(Ops.SQRT, dtypes.float32, (red,))
    neg = UOp(Ops.NEG, dtypes.float32, (sq,))
    return wrap_sink(neg)


def build_reduce_reshape_binop():
    """c = a.sum(0).reshape(10) + b, shape [10, 10] -> [10]."""
    a = mk_param(0, 10, 10)
    b = mk_param(1, 10)
    red = UOp(Ops.REDUCE, dtypes.float32, (a,), (Ops.ADD, (0,)))
    reshaped = UOp(Ops.RESHAPE, dtypes.float32, (red, mk_shape(10)))
    return wrap_sink(reshaped + b)


def build_reduce_permute_binop():
    """c = a.sum(0).permute(1,0) + b, shape [10,10,10]."""
    a = mk_param(0, 10, 10, 10)
    b = mk_param(1, 10, 10)
    red = UOp(Ops.REDUCE, dtypes.float32, (a,), (Ops.ADD, (0,)))
    permed = UOp(Ops.PERMUTE, dtypes.float32, (red,), (1, 0))
    return wrap_sink(permed + b)


def build_permute_through_reshape():
    """c = (a+b).reshape(4,4,4,4).permute(2,3,0,1)."""
    a = mk_param(0, 16, 16)
    b = mk_param(1, 16, 16)
    add = a + b
    reshaped = UOp(Ops.RESHAPE, dtypes.float32, (add, mk_shape(4, 4, 4, 4)))
    permed = UOp(Ops.PERMUTE, dtypes.float32, (reshaped,), (2, 3, 0, 1))
    return wrap_sink(permed)


def build_expand_permute():
    """d = (a+b).expand(10,10,10) + (a+b).permute(2,1,0).expand(10,10,10)."""
    a = mk_param(0, 10, 10, 1)
    b = mk_param(1, 10, 10, 1)
    ab = a + b
    expanded = ab.expand((10, 10, 10))
    permed = ab.permute((2, 1, 0))
    permed_expanded = permed.expand((10, 10, 10))
    return wrap_sink(expanded + permed_expanded)


def build_shrink_fuse():
    """e = (a*b)[0] * d, shape [8192,16], d=[1,16]."""
    a = mk_param(0, 8192, 16)
    b = mk_param(1, 8192, 16)
    d = mk_param(2, 1, 16)
    mul = a * b
    shrunk = mul.shrink(((0, 1), (0, 16)))
    return wrap_sink(shrunk * d)


def build_multistage_reduce():
    """c = a.sum(2).relu().sum(1), shape [32,32,32]."""
    a = mk_param(0, 32, 32, 32)
    red1 = a._rop(Ops.ADD, (2,))
    relu = red1.alu(Ops.MAX, red1.const_like(0.0))
    reshaped = relu.reshape((32, 32))
    red2 = reshaped._rop(Ops.ADD, (1,))
    return wrap_sink(red2)


def build_two_sum():
    """c = a.sum(0) + a.sum(1), shape [64,64]."""
    a = mk_param(0, 64, 64)
    red0 = a._rop(Ops.ADD, (0,))
    red1 = a._rop(Ops.ADD, (1,))
    reshaped0 = red0.reshape((64,))
    reshaped1 = red1.reshape((64,))
    return wrap_sink(reshaped0 + reshaped1)


def build_reduce_shrink():
    """c = a.sum(1)[:16] + b, shape [32,32], b=[16]."""
    a = mk_param(0, 32, 32)
    b = mk_param(1, 16)
    red = UOp(Ops.REDUCE, dtypes.float32, (a,), (Ops.ADD, (1,)))
    reshaped = UOp(Ops.RESHAPE, dtypes.float32, (red, mk_shape(32)))
    shrunk = reshaped.shrink(((0, 16),))
    return wrap_sink(shrunk + b)


def build_contiguous_add():
    """d = (x+y).contiguous() + z, produces 2 kernels."""
    x = mk_param(0, 32)
    y = mk_param(1, 32)
    z = mk_param(2, 32)
    add = x + y
    contig = UOp(Ops.CONTIGUOUS, dtypes.float32, (add,))
    return wrap_sink(contig + z)


def build_reshape_chain():
    """c = a.reshape(16).reshape(2,8) + b, shape [4,4], b=[2,8]."""
    a = mk_param(0, 4, 4)
    b = mk_param(1, 2, 8)
    r1 = UOp(Ops.RESHAPE, dtypes.float32, (a, mk_shape(16)))
    r2 = UOp(Ops.RESHAPE, dtypes.float32, (r1, mk_shape(2, 8)))
    return wrap_sink(r2 + b)


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
        "r_2_8": "llama_rmsnorm",
        "r_2_8_8": "llama_ffn_gate",
        "E_2_2_4": "llama_vector_scale",
        "r_2_32_8": "llama_output_projection",
    }
    _LLAMA_MODEL_SINKS = {}
    for sink in sinks:
        name_counts = Scheduler.kernel_cnt.copy()
        rewritten = full_rewrite_to_sink(sink, RENDERERS["clang"], optimize=True)
        Scheduler.kernel_cnt.clear()
        Scheduler.kernel_cnt.update(name_counts)
        case_name = targets.get(rewritten.arg.name)
        if case_name is not None:
            _LLAMA_MODEL_SINKS[case_name] = sink

    missing = sorted(set(targets.values()) - set(_LLAMA_MODEL_SINKS))
    if missing:
        raise RuntimeError(f"tinygrad LLaMA kernels not found: {missing}")
    return _LLAMA_MODEL_SINKS


def get_llama_source(case_name, renderer, optimize=True):
    return render_kernel(llama_model_sinks()[case_name], renderer, optimize)


def get_llama_forward_from_embedding_source(renderer, optimize=True):
    """Tinygrad LLaMA forward from post-token-embedding activations to logits."""
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

    h = Tensor.empty(1, 2, 8, dtype=dtypes.float)
    freqs_cis = model.freqs_cis.cast(h.dtype)[:, 0:2, :, :, :]
    for layer in model.layers:
        h = layer(h, 0, freqs_cis, None)
    logits = model.output(model.norm(h).contiguous().contiguous_backward())
    linear = logits.contiguous_backward().schedule_linear(
        h, *nn.state.get_parameters(model)
    )

    sources = []
    seen = set()
    for call in linear.src:
        if call.op is not Ops.CALL or call.src[0].op is not Ops.SINK:
            continue
        sink = call.src[0]
        if sink.key in seen:
            continue
        seen.add(sink.key)
        sources.append(render_kernel(sink, renderer, optimize))
    return "\n---\n".join(sources)


# ── Test cases ──
# (name, builder, backends_or_None, optimize)

GPU_RENDERERS = ["cuda", "metal", "opencl"]

TEST_CASES = [
    # Tier 1: Core fusion (1 kernel each)
    ("elementwise_add", build_elementwise_add, None, True),
    ("elementwise_3way", build_elementwise_3way, None, True),
    ("mulacc", build_mulacc, None, True),
    ("binop_reshape", build_binop_reshape, None, True),
    ("binop_permute", build_binop_permute, None, True),
    ("diamond", build_diamond, None, True),
    ("reduce_unary", build_reduce_unary, None, True),
    ("reduce_reshape_binop", build_reduce_reshape_binop, None, True),
    # Tier 2: Movement ops (1 kernel each)
    ("reduce_permute_binop", build_reduce_permute_binop, None, True),
    ("permute_through_reshape", build_permute_through_reshape, None, True),
    ("expand_permute", build_expand_permute, None, True),
    ("shrink_fuse", build_shrink_fuse, None, True),
    # Tier 3: Multi-reduce / multi-kernel
    ("multistage_reduce", build_multistage_reduce, None, True),
    ("two_sum", build_two_sum, None, True),
    ("reduce_shrink", build_reduce_shrink, None, True),
    # Tier 4: Edge cases
    ("contiguous_add", build_contiguous_add, None, True),
    ("reshape_chain", build_reshape_chain, None, True),
    ("llama_rmsnorm", lambda: "llama_rmsnorm", None, True),
    ("llama_ffn_gate", lambda: "llama_ffn_gate", None, True),
    ("llama_vector_scale", lambda: "llama_vector_scale", None, True),
    ("llama_output_projection", lambda: "llama_output_projection", None, True),
    ("llama_forward_from_embedding", lambda: "llama_forward_from_embedding", None, True),
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
                if isinstance(sink, str) and sink.startswith("llama_"):
                    if sink == "llama_forward_from_embedding":
                        src = get_llama_forward_from_embedding_source(
                            renderer, optimize=optimize
                        )
                    else:
                        src = get_llama_source(sink, renderer, optimize=optimize)
                else:
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
