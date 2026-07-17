#!/usr/bin/env python3
"""Reference-side compile-pipeline timing for the comparative runner.

Builds the same three workload graphs as the OCaml side (see graphs.ml —
the UOp construction is mirrored op for op) and times the counterpart of
each tolk pipeline stage the same way: warm once, then median and min of N
samples on a monotonic clock in one warm process. Import cost is excluded.

Stage mapping to the tolk side:
  schedule_linear   tolk rangeify + schedule combined
                    (create_schedule(get_kernel_graph(sink)) + memory plan)
  codegen           per-kernel full_rewrite_to_sink
  linearize         per-kernel linearize + line_rewrite cleanups
  render            per-kernel renderer.render
  compile           per-kernel device compile (host clang, cold)

Writes <out>/tinygrad.json (timing rows) and <out>/tinygrad.verify.json
(per-workload kernel count and first-kernel source) for report.py to join
and cross-check against the tolk side.

Run from the repo root:  uv run packages/tolk/bench/compare/bench_compare.py
"""

import os
import platform
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "..", "_tinygrad"))

# Read once at import in tinygrad: disable every cache so repeated passes and
# the stage-7 compile measure cold work, and strip ANSI from kernel names so
# the rendered source matches the tolk side byte for byte.
os.environ.setdefault("SCACHE", "0")
os.environ.setdefault("CACHELEVEL", "0")
os.environ.setdefault("CCACHE", "0")
os.environ.setdefault("BEAM", "0")
os.environ.setdefault("IGNORE_BEAM_CACHE", "1")
os.environ.setdefault("NO_COLOR", "1")

import json  # noqa: E402

from tinygrad.codegen import (  # noqa: E402
    full_rewrite_to_sink, line_rewrite, pm_linearize_cleanups,
)
from tinygrad.codegen.late.linearizer import linearize  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402
from tinygrad.helpers import Target  # noqa: E402
from tinygrad.renderer.cstyle import ClangRenderer  # noqa: E402
from tinygrad.schedule import create_schedule  # noqa: E402
from tinygrad.schedule.memory import memory_plan_rewrite  # noqa: E402
from tinygrad.schedule.rangeify import get_kernel_graph  # noqa: E402
from tinygrad.uop.ops import (  # noqa: E402
    KernelInfo, Ops, UOp, shape_to_shape_arg,
)

OPTIMIZE = True
N_STAGE = 20
N_COMPILE = 5

# Render-only renderer for stages 2-6: matches the tolk clang_no_abi/X86_64
# goldens byte for byte.
REN = ClangRenderer(Target("CPU", arch="x86_64,znver2"))

# Device renderer + compiler for stage 7: the host clang path, which yields a
# compilable translation unit.
_HOST = {"amd64": "x86_64", "aarch64": "arm64"}.get(
    platform.machine().lower(), platform.machine().lower())
DEVICE_REN = ClangRenderer(Target("CPU", arch=_HOST + ",native"))
DEVICE_COMP = DEVICE_REN.compiler


# Graph builders — mirror graphs.ml op for op.

def mk_param(slot, *shape):
    return UOp.param(slot, dtypes.float32, shape=shape, device="CPU")


def wrap_sink(*srcs):
    contigs = [UOp(Ops.CONTIGUOUS, s.dtype, (s,)) for s in srcs]
    return UOp.sink(*contigs)


def build_elementwise():
    a = mk_param(0, 256, 256)
    b = mk_param(1, 256, 256)
    c = mk_param(2, 256, 256)
    return wrap_sink(a + b * c)


def build_reduce():
    x = mk_param(0, 512, 512)
    return wrap_sink(x._rop(Ops.ADD, (1,)))


def build_matmul_small():
    m = n = k = 128
    a = mk_param(0, m, k)
    b = mk_param(1, k, n)
    ar = UOp(Ops.RESHAPE, dtypes.float32, (a, shape_to_shape_arg((m, 1, k))))
    ae = ar.expand((m, n, k))
    bt = UOp(Ops.PERMUTE, dtypes.float32, (b,), (1, 0))
    br = UOp(Ops.RESHAPE, dtypes.float32, (bt, shape_to_shape_arg((1, n, k))))
    be = br.expand((m, n, k))
    red = (ae * be)._rop(Ops.ADD, (2,))
    return wrap_sink(red)


WORKLOADS = [
    ("elementwise", "256x256", build_elementwise),
    ("reduce", "512x512", build_reduce),
    ("matmul_small", "128x128x128", build_matmul_small),
]


# Pipeline seams.

def extract_kernels(sink):
    kg = get_kernel_graph(sink)
    return [u.src[0] for u in kg.toposort()
            if u.op is Ops.CALL and isinstance(u.src[0].arg, KernelInfo)]


def schedule_linear(sink):
    return memory_plan_rewrite(create_schedule(get_kernel_graph(sink)))


def linearize_kernel(processed):
    return line_rewrite(linearize(processed), pm_linearize_cleanups)


def render_kernel(ren, kernel):
    processed = full_rewrite_to_sink(kernel, ren, optimize=OPTIMIZE)
    return ren.render(linearize_kernel(processed))


# Timing.

def median_min(samples):
    s = sorted(samples)
    return s[len(s) // 2], s[0]


def time_stage(fn, n):
    fn()  # warm
    out = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        r = fn()
        t1 = time.perf_counter_ns()
        del r
        out.append((t1 - t0) / 1e6)
    return median_min(out)


def measure(name, size, build):
    sink = build()
    kernels = extract_kernels(sink)
    n_kernels = len(kernels)
    codegen = [full_rewrite_to_sink(k, REN, optimize=OPTIMIZE) for k in kernels]
    programs = [linearize_kernel(p) for p in codegen]
    render_srcs = [REN.render(p) for p in programs]
    src_bytes = sum(len(s) for s in render_srcs)
    device_srcs = [render_kernel(DEVICE_REN, k) for k in kernels]
    compile_bytes = sum(len(s) for s in device_srcs)
    first_kernel_src = render_srcs[0].strip() if render_srcs else ""

    def row(stage, timing, bytes_):
        median, minimum = timing
        return {"workload": name, "size": size, "stage": stage,
                "ms_median": median, "ms_min": minimum,
                "n_kernels": n_kernels, "src_bytes": bytes_}

    rows = [
        row("schedule_linear",
            time_stage(lambda: schedule_linear(sink), N_STAGE), src_bytes),
        row("codegen",
            time_stage(lambda: [full_rewrite_to_sink(k, REN, optimize=OPTIMIZE)
                                for k in kernels], N_STAGE), src_bytes),
        row("linearize",
            time_stage(lambda: [linearize_kernel(p) for p in codegen],
                       N_STAGE), src_bytes),
        row("render",
            time_stage(lambda: [REN.render(p) for p in programs], N_STAGE),
            src_bytes),
        row("compile",
            time_stage(lambda: [DEVICE_COMP.compile_cached(s)
                                for s in device_srcs], N_COMPILE),
            compile_bytes),
    ]
    verify = {"n_kernels": n_kernels, "first_kernel_src": first_kernel_src}
    return rows, (name, verify)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    rows = []
    verify = {}
    for name, size, build in WORKLOADS:
        r, (vname, v) = measure(name, size, build)
        rows.extend(r)
        verify[vname] = v
    with open(os.path.join(out_dir, "tinygrad.json"), "w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
    with open(os.path.join(out_dir, "tinygrad.verify.json"), "w") as f:
        json.dump(verify, f, indent=2)
        f.write("\n")
    print(f"wrote {len(rows)} rows for {len(WORKLOADS)} workloads "
          f"to {out_dir}/tinygrad.json")


if __name__ == "__main__":
    main()
