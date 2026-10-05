"""Goldens of tinygrad/codegen/opt/search.py: the actions a beam search tries,
the candidates it makes of real kernels, and the kernels it chooses under a
measurement that computes each program's time.

`actions` lists the table in order, and `actions_padto` the table under
ENVIRONMENT, whose variables the module reads when it is imported. A
kernel golden, named after the kernel, is the sink that compiling a real kernel
hands to `apply_opts`. `targets` lists the renderers.

`candidates` lists, for a kernel on a target, the positions `get_kernel_actions`
returns for the kernel as `apply_opts` hands it to the search, with the global
axes converted, and `max_up` if set.

`searches` lists what `beam_search` chooses, of width `amt`, with the
measurement `time` below in place of running programs: its optimisations, and
how many times it measured. The search is tinygrad's as tolk departs from it
(D117): it times the kernel itself before its first round and starts the beam
from it, and a round progresses only when every sample of its fastest
candidate is faster than every sample of the beam's first kernel by
`BEAM_MIN_PROGRESS`, else the search answers the beam's first kernel. No program is compiled: the binary of a program is
its source's bytes. The measurement is a function of what the program's kernel
and launch record, so that tolk's suite can compute it too.
`searches_environment` lists the same under ENVIRONMENT: there a transposing
copy has no arithmetic, and each pad adds a selection per element, so the
search leaves out the pads, which cost more than a thousand times the fewest
operations.
"""

import contextlib
import importlib
import math
import os

import tinygrad.runtime.support.compiler_amd as compiler_amd

# Making a HIP renderer makes its compiler, which asserts that comgr is loaded;
# comgr is absent where the goldens are generated, and no case compiles.
compiler_amd.c.DLL._loaded_.add(compiler_amd.comgr.dll.nm)

import tinygrad.runtime.ops_metal as ops_metal

# Making a Metal renderer starts Metal's code generation service, whose threads
# make the processes forked for each golden crash at random.
ops_metal.MetalCompiler.support.MTLCodeGenServiceCreate = lambda name: None

from golden import graph, table
from graph import kernels, stage
from tinygrad import Tensor, UOp, dtypes
from tinygrad.codegen.opt import search
from tinygrad.codegen.opt.postrange import Scheduler, args_from_ast
from tinygrad.helpers import Context, Target, getenv, prod
from tinygrad.renderer.cstyle import ClangRenderer, CUDARenderer, HIPRenderer, MetalRenderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, sym_infer

TARGETS = {
    "clang": (ClangRenderer, Target("CPU", "CLANG", "x86_64,x86-64")),
    "metal": (MetalRenderer, Target("METAL", "METAL", "Apple9")),
    "cuda": (CUDARenderer, Target("CUDA", "CUDA", "sm_89")),
    "hip": (HIPRenderer, Target("AMD", "HIP", "gfx1100")),
}


def renderer(target):
    cls, t = TARGETS[target]
    ren = cls(t)
    # No case compiles: the binary is the source's bytes.
    ren.compiler.compile_cached = lambda src: src.encode()
    return ren


RENDERERS = {name: renderer(name) for name in TARGETS}


@table
def targets():
    return ["target", "device", "renderer", "arch"], [
        (name, t.device, t.renderer, t.arch) for name, (_, t) in TARGETS.items()]


# Actions

@table
def actions():
    return ["action", "opt"], list(enumerate(search.actions))


# The variables of the process in which tolk's suite checks what they
# change, less those that only print.
ENVIRONMENT = {"BEAM_PADTO": "1", "TC": "2", "BEAM_STRICT_MODE": "1", "BEAM_UOPS_MAX": "43"}


@contextlib.contextmanager
def environment():
    os.environ.update(ENVIRONMENT)
    getenv.cache_clear()
    importlib.reload(search)
    try:
        yield
    finally:
        for k in ENVIRONMENT: del os.environ[k]
        getenv.cache_clear()
        importlib.reload(search)


@table
def actions_padto():
    with environment():
        return ["action", "opt"], list(enumerate(search.actions))


# Kernels

def last(*tensors): return kernels(*tensors)[-1]


def empty(*shape, dtype=dtypes.float): return Tensor.empty(*shape, dtype=dtype)


def with_vars(*tensors):
    return [c.src[0] for c in Tensor.linear_with_vars(*tensors)[0].src if c.src[0].op is Ops.SINK][-1]


KERNELS = {
    "add_small": lambda: last(empty(16) + 1),
    "add": lambda: last(empty(64, 64) + empty(64, 64)),
    "add_large": lambda: last(empty(1024, 1024) + 1),
    "transpose_33": lambda: last(empty(33, 33).T.contiguous()),
    "add_3d": lambda: last(empty(16, 16, 4096) + empty(16, 1, 4096)),
    "sum_rows": lambda: last(empty(32, 32).sum(1)),
    "sum_3x3": lambda: last(empty(4096, 3, 3).sum((1, 2))),
    "pad_7x7": lambda: last(empty(64, 5, 5).pad(((0, 0), (1, 1), (1, 1))) + 1),
    "matmul_small": lambda: last(empty(8, 8) @ empty(8, 8)),
    "matmul_half": lambda: last(empty(64, 64, dtype=dtypes.half).matmul(empty(64, 64, dtype=dtypes.half), dtype=dtypes.float)),
    "conv": lambda: last(empty(1, 4, 8, 8).conv2d(empty(8, 4, 3, 3), padding=1)),
    "variable": lambda: with_vars(empty(1024)[:UOp.variable("n", 1, 1024).bind(512)].contiguous() + 1),
    # runtime/test_search.py's test_beam_symbolic_kernel
    "symbolic": lambda: with_vars(empty(8, 8)[:UOp.variable("size", 1, 8).bind(4)] + 1),
    "variable_rows": lambda: with_vars(empty(16, 64)[:UOp.variable("n", 1, 16).bind(4)].contiguous() + 1),
}

# Built by hand: axes whose lanes or threads multiply past 64-bit integers.
def huge_axes(size, *types):
    out = UOp.param(0, dtypes.float, 1)
    rngs = [UOp.range(size, i, t) for i, t in enumerate(types)]
    return UOp.sink(out.index(UOp.const(0, dtypes.weakint)).store(UOp.const(1.0, dtypes.float)).end(*rngs), arg=KernelInfo())


BUILT = {
    "huge_upcasts": lambda: huge_axes(2**32, AxisType.GLOBAL, AxisType.UPCAST, AxisType.UPCAST),
    "huge_locals": lambda: huge_axes(2**32, AxisType.GLOBAL, AxisType.LOCAL, AxisType.LOCAL),
}

INPUTS = {}


def kernel_input(kernel):
    if kernel in BUILT: return BUILT[kernel]()
    if kernel not in INPUTS:
        with Context(DEV="CPU"): INPUTS[kernel] = stage("apply_opts", KERNELS[kernel](), RENDERERS["clang"])
    return INPUTS[kernel]


def scheduled(kernel, target):
    k = Scheduler(kernel_input(kernel), RENDERERS[target])
    k.convert_loop_to_global()
    return k


def declare(name, fn):
    fn.__name__ = name
    graph(fn)


for kernel in [*KERNELS, *BUILT]:
    declare(kernel, lambda kernel=kernel: kernel_input(kernel))


# Candidates

CANDIDATES = [(kernel, target, None) for kernel in KERNELS for target in TARGETS if kernel not in ("add_large", "add_3d")]
CANDIDATES += [("add", "clang", 4), ("matmul_half", "metal", 16), ("huge_upcasts", "metal", None), ("huge_locals", "metal", None)]


@table
def candidates():
    rows = []
    for kernel, target, max_up in CANDIDATES:
        acted = search.get_kernel_actions(scheduled(kernel, target), max_up=max_up)
        rows.append((kernel, target, "" if max_up is None else str(max_up), " ".join(map(str, acted))))
    return ["kernel", "target", "max_up", "actions"], rows


# Searches

def time(prg, failing):
    """The time of the program `prg`: an optimum at two optimisations, each
    weighed by its action's position, scaled by the launch's workgroups. With
    `failing`, the programs whose actions' positions sum to a multiple of 3
    fail."""
    positions = [search.actions.index(o) for o in prg.src[0].arg.applied_opts]
    if failing and positions and sum(positions) % 3 == 0: raise RuntimeError("failing measurement")
    tm = 1e-3 * (1 + 3 * abs(len(positions) - 2))
    for p in positions: tm *= 0.8 + (37 * p % 41) / 100
    return tm * (1 + prod(prg.arg.global_size) % 5 / 10)


class Measured:
    """The measurement in place of `time_call`, counting its runs, and a device
    that has no caches to clear."""

    def __init__(self, failing):
        self.failing, self.count = failing, 0

    def time_call(self, call, var_vals=None, timeout=None, clear_l2=False):
        tm = time(call.src[0], self.failing)
        while True:
            self.count += 1
            yield tm


def beam_search(s, rawbufs, var_vals, amt, allow_test_size=True):
    """tinygrad's `beam_search` with tolk's D117, uncached and on one
    process."""
    min_progress = getenv("BEAM_MIN_PROGRESS", 0.01) / 1e6
    dev = search.Device[s.ren.target.device]
    seen_libs = set()

    def timed(prg, early_stop):
        try:
            return search._time_program(prg, var_vals, rawbufs, early_stop=early_stop, allow_test_size=allow_test_size,
                                        clear_l2=hasattr(dev, "invalidate_caches"), dev_timeout=getenv("BEAM_DEV_TIMEOUT", 1))
        except RuntimeError:
            return None

    # The kernel itself, timed with no early stop, starts the beam.
    _, proc = search._try_compile((0, s))
    start = []
    if proc is not None:
        seen_libs.add(proc[0].src[3].arg)
        start = timed(proc[0], None) or []
    beam = [(s, start)]
    while True:
        candidates = [c for si, _ in beam for c in search.get_kernel_actions(si, include_0=False).values()]
        incumbent = min(beam[0][1], default=math.inf)
        opts, least_compute_ops = [], math.inf
        for i, proc in map(search._try_compile, enumerate(candidates)):
            if proc is None: continue
            prg, _ = proc
            if (lib := prg.src[3].arg) in seen_libs: continue
            estimates = prg.src[0].arg.estimates
            least_compute_ops = min(this_compute_ops := sym_infer(estimates.ops if estimates is not None else 0, var_vals), least_compute_ops)
            if least_compute_ops * 1000 < this_compute_ops: continue
            seen_libs.add(lib)
            if (tms := timed(prg, incumbent * 3)) is not None: opts.append((candidates[i], tms))
        opts.sort(key=lambda x: min(x[1]))
        if not opts or not max(opts[0][1]) + min_progress < incumbent: return beam[0][0]
        beam = opts[:amt]


def searched(kernel, target, amt, failing=False, allow_test_size=True):
    ast = kernel_input(kernel)
    measured = Measured(failing)
    search.time_call, search._ensure_buffer_alloc = measured.time_call, lambda bufs: bufs
    search.Device = {TARGETS[target][1].device: object()}
    rawbufs, var_vals = args_from_ast(ast, TARGETS[target][1].device)
    with Context(CACHELEVEL=0, IGNORE_BEAM_CACHE=1, PARALLEL=0):
        k = beam_search(scheduled(kernel, target), rawbufs, var_vals, amt, allow_test_size)
    return tuple(k.applied_opts), measured.count


SEARCHES = [
    ("add_small", "clang", 1, False, True),
    ("add_small", "clang", 2, False, True),
    ("add_small", "clang", 2, True, True),
    ("sum_rows", "clang", 2, False, True),
    ("variable", "clang", 1, False, True),
    ("variable_rows", "clang", 1, False, True),
    ("variable_rows", "metal", 1, False, True),
    ("add", "metal", 1, False, True),
    ("add_large", "metal", 1, False, True),
    ("add_large", "metal", 1, False, False),
    ("sum_rows", "metal", 2, False, True),
    ("matmul_half", "metal", 1, False, True),
    ("matmul_half", "cuda", 1, True, True),
    ("pad_7x7", "hip", 1, False, True),
]


@table
def searches():
    rows = []
    for kernel, target, amt, failing, allow_test_size in SEARCHES:
        opts, count = searched(kernel, target, amt, failing, allow_test_size)
        rows.append((kernel, target, amt, failing, allow_test_size, repr(opts), count))
    return ["kernel", "target", "amt", "failing", "allow_test_size", "opts", "measurements"], rows


@table
def searches_environment():
    with environment():
        opts, count = searched("transpose_33", "metal", 1)
    return ["kernel", "target", "amt", "opts", "measurements"], [("transpose_33", "metal", 1, repr(opts), count)]
