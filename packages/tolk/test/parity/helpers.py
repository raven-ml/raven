"""Shared plumbing for parity case main.py scripts.

Centralises the pieces every case needs: the sys.path hook that makes the
reference clone at `_tinygrad/` importable, render-only compiler construction mocks
(the target renderer initializers run unchanged), the table of backends we
diff against, and the canonical call sequence through the reference codegen
pipeline.
"""

import contextlib
import io
import os
import sys
from unittest.mock import patch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "..", "_tinygrad"))

# Disable ANSI color in the reference — without this, auto-generated kernel
# names embed ANSI escape codes inside the repr() of KernelInfo.name, which
# leaks through [print_uops] into the .expected files and breaks parity.
os.environ["NO_COLOR"] = "1"

from tinygrad.codegen import (  # noqa: E402
    full_rewrite_to_sink, line_rewrite, pm_linearize_cleanups, pm_alloc_to_buf,
)
from tinygrad.codegen.late.linearizer import linearize  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402
from tinygrad.helpers import Target  # noqa: E402
from tinygrad.renderer.cstyle import (  # noqa: E402
    ClangRenderer, CUDARenderer, HIPRenderer, MetalRenderer, OpenCLRenderer,
)
from tinygrad.schedule.rangeify import get_kernel_graph  # noqa: E402
from tinygrad.schedule.prepare import prepare_rangeify  # noqa: E402
from tinygrad.uop.ops import AxisType, KernelInfo, Ops  # noqa: E402
from tinygrad.uop.render import print_uops  # noqa: E402


def render_only(ctor, target, compiler):
    """Run the real renderer initializer, replacing only compiler construction."""
    with patch(compiler, return_value=None):
        return ctor(target)


def cuda_renderer(target):
    return render_only(CUDARenderer, target,
                       "tinygrad.runtime.support.compiler_cuda.NVRTCCompiler")


def hip_renderer(target):
    return render_only(HIPRenderer, target,
                       "tinygrad.runtime.support.compiler_amd.HIPCompiler")


ALL_BACKENDS = {}
for _name, _ctor in [
    ("cpu", lambda: ClangRenderer(Target("CPU", arch="x86_64,znver2"))),
    ("cuda", lambda: cuda_renderer(Target("CUDA", arch="sm_80"))),
    ("metal", lambda: MetalRenderer(Target("METAL", arch="Apple7"))),
    ("opencl", lambda: OpenCLRenderer(Target("CL"))),
    # amd stays last: auto-generated kernel names carry a per-process counter,
    # so inserting a backend mid-list would renumber every later backend's
    # kernels and rewrite the pre-existing goldens.
    ("amd", lambda: hip_renderer(Target("AMD", arch="gfx1100"))),
]:
    try:
        ALL_BACKENDS[_name] = _ctor()
    except Exception as e:
        raise RuntimeError(f"required {_name} renderer failed to initialize") from e

GPU_BACKENDS = {k: v for k, v in ALL_BACKENDS.items() if k != "cpu"}

def kernel_for_renderer(ren, sink):
    """Match the paired fixture: hardware ranges become CPU software loops."""
    if ren.has_local:
        return sink
    return sink.substitute({u: u.replace(arg=(*u.arg[:-1], AxisType.WEAK))
                            for u in sink.toposort()
                            if u.op is Ops.RANGE and u.arg[-1] is AxisType.GLOBAL})


def stage5(ren, sink, optimize=True):
    """Columnar print_uops of the kernel after full_rewrite_to_sink."""
    rewritten = full_rewrite_to_sink(kernel_for_renderer(ren, sink), ren, optimize=optimize)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_uops(list(rewritten.toposort()))
    return buf.getvalue().rstrip("\n")


def stage7(ren, sink, optimize=True):
    """Final rendered backend source (trimmed)."""
    program = linearize(full_rewrite_to_sink(kernel_for_renderer(ren, sink), ren, optimize=optimize))
    program = line_rewrite(program, pm_linearize_cleanups + pm_alloc_to_buf)
    return ren.render(program).strip()


def mk_param(slot, *shape, dtype=None, device="CPU"):
    """Tensor-level PARAM with concrete shape."""
    from tinygrad.dtype import dtypes as _dt
    from tinygrad.uop.ops import UOp as _UOp
    return _UOp.param(slot, dtype if dtype is not None else _dt.float32,
                      shape=shape, device=device)


def wrap_sink(*srcs):
    """Materialize each source with the target contiguous operation."""
    from tinygrad.uop.ops import UOp as _UOp
    contigs = [s.contiguous() for s in srcs]
    return _UOp.sink(*contigs)


def _extract_kernels(sink):
    """Run rangeify, return inline kernel AST roots (CALL srcs) in toposort."""
    kg = get_kernel_graph(prepare_rangeify(sink))
    out = []
    for u in kg.toposort():
        if u.op is Ops.CALL and isinstance(u.src[0].arg, KernelInfo):
            out.append(u.src[0])
    return out


def stage5_tensor(ren, tensor_sink, optimize=True):
    """Columnar print_uops for each kernel produced by rangeify."""
    kernels = _extract_kernels(tensor_sink)
    parts = []
    for i, k in enumerate(kernels):
        rewritten = full_rewrite_to_sink(k, ren, optimize=optimize)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_uops(list(rewritten.toposort()))
        body = buf.getvalue().rstrip("\n")
        if len(kernels) == 1:
            parts.append(body)
        else:
            parts.append(f"=== kernel {i} ===\n{body}")
    return "\n".join(parts)


def stage7_tensor(ren, tensor_sink, optimize=True):
    """Rendered backend source for each kernel, joined by '\\n---\\n'."""
    kernels = _extract_kernels(tensor_sink)
    sources = [
        ren.render(line_rewrite(
            linearize(full_rewrite_to_sink(k, ren, optimize=optimize)),
            pm_linearize_cleanups + pm_alloc_to_buf,
        )).strip()
        for k in kernels
    ]
    return "\n---\n".join(sources)


_STAGES = {"stage5": stage5, "stage7": stage7}
_TENSOR_STAGES = {"stage5": stage5_tensor, "stage7": stage7_tensor}


def dump(sink, out_dir, stages=("stage7",), backends=None, optimize=True):
    """Write <out_dir>/<stage>_<backend>.expected for each (stage, backend)."""
    for name, ren in (backends or ALL_BACKENDS).items():
        for stage in stages:
            src = _STAGES[stage](ren, sink, optimize=optimize)
            path = os.path.join(out_dir, f"{stage}_{name}.expected")
            with open(path, "w") as f:
                f.write(src + "\n")


def dump_stage7(sink, out_dir, backends=None, optimize=True):
    """Stage-7-only convenience wrapper."""
    dump(sink, out_dir, stages=("stage7",), backends=backends, optimize=optimize)


def dump_tensor(sink, out_dir, stages=("stage7",), backends=None, optimize=True):
    """Tensor-graph counterpart of [dump]: rangeify first, then per kernel."""
    for name, ren in (backends or ALL_BACKENDS).items():
        for stage in stages:
            src = _TENSOR_STAGES[stage](ren, sink, optimize=optimize)
            path = os.path.join(out_dir, f"{stage}_{name}.expected")
            with open(path, "w") as f:
                f.write(src + "\n")


def stage7_program(ren, program):
    """Render a pre-linearized flat UOp program directly.

    No full_rewrite_to_sink or linearize pass is run. [program] is a
    ``list[UOp]`` already in linearized (toposorted) form.
    """
    return ren.render(program).strip()


def dump_stage7_program(program, out_dir, backends=None, name=None):
    """Write stage7_<backend>.expected files from a pre-linearized program.

    [name] is unused on the reference side — the kernel name is carried by
    the SINK's KernelInfo — but accepted for symmetry with the OCaml side.
    """
    del name
    for backend, ren in (backends or ALL_BACKENDS).items():
        src = stage7_program(ren, program)
        path = os.path.join(out_dir, f"stage7_{backend}.expected")
        with open(path, "w") as f:
            f.write(src + "\n")
