"""Shared plumbing for parity case main.py scripts.

Centralises the pieces every case needs: the sys.path hook that makes the
reference clone at `_tinygrad/` importable, a CUDA renderer subclass that
skips NVRTC init (we only render, never execute), the table of backends we
diff against, and the canonical call sequence through the reference codegen
pipeline.
"""

import contextlib
import io
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "..", "_tinygrad"))

# Disable ANSI color in the reference — without this, auto-generated kernel
# names embed ANSI escape codes inside the repr() of KernelInfo.name, which
# leaks through [print_uops] into the .expected files and breaks parity.
os.environ["NO_COLOR"] = "1"

from tinygrad.codegen import (  # noqa: E402
    full_rewrite_to_sink, line_rewrite, pm_linearize_cleanups,
)
from tinygrad.codegen.late.linearizer import linearize  # noqa: E402
from tinygrad.codegen.opt import tc  # noqa: E402
from tinygrad.helpers import Target  # noqa: E402
from tinygrad.renderer.cstyle import (  # noqa: E402
    ClangRenderer, CUDARenderer, MetalRenderer, OpenCLRenderer,
)
from tinygrad.schedule.rangeify import get_kernel_graph  # noqa: E402
from tinygrad.uop.ops import KernelInfo, Ops  # noqa: E402
from tinygrad.uop.render import print_uops  # noqa: E402


class _CudaNoNvrtc(CUDARenderer):
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


ALL_BACKENDS = {
    "cpu": ClangRenderer(Target("CPU", arch="x86_64,znver2")),
    "cuda": _CudaNoNvrtc(Target("CUDA", arch="sm_80")),
    "metal": MetalRenderer(Target("METAL")),
    "opencl": OpenCLRenderer(Target("CL")),
}

GPU_BACKENDS = {k: v for k, v in ALL_BACKENDS.items() if k != "cpu"}

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s):
    return _ANSI.sub("", s)


def stage5(ren, sink, optimize=True):
    """Columnar print_uops of the kernel after full_rewrite_to_sink."""
    rewritten = full_rewrite_to_sink(sink, ren, optimize=optimize)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_uops(list(rewritten.toposort()))
    return _strip_ansi(buf.getvalue().rstrip("\n"))


def stage7(ren, sink, optimize=True):
    """Final rendered backend source (trimmed)."""
    program = linearize(full_rewrite_to_sink(sink, ren, optimize=optimize))
    program = line_rewrite(program, pm_linearize_cleanups)
    return ren.render(program).strip()


def mk_param(slot, *shape, dtype=None, device="CPU"):
    """Tensor-level PARAM with concrete shape."""
    from tinygrad.dtype import dtypes as _dt
    from tinygrad.uop.ops import UOp as _UOp
    return _UOp.param(slot, dtype if dtype is not None else _dt.float32,
                      shape=shape, device=device)


def wrap_sink(*srcs):
    """Wrap each source in CONTIGUOUS and return a SINK."""
    from tinygrad.uop.ops import UOp as _UOp, Ops as _Ops
    contigs = [_UOp(_Ops.CONTIGUOUS, s.dtype, (s,)) for s in srcs]
    return _UOp.sink(*contigs)


def _extract_kernels(sink):
    """Run rangeify, return inline kernel AST roots (CALL srcs) in toposort."""
    kg = get_kernel_graph(sink)
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
        body = _strip_ansi(buf.getvalue().rstrip("\n"))
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
            pm_linearize_cleanups,
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
