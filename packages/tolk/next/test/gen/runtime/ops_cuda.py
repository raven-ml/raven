"""Goldens of tinygrad/runtime/ops_cuda.py: batches on a CUDA device, encoded
by CUDAQueue, and their host programs.

The device CUDA is described without opening a GPU, on an sm_89 target, and
reaches the host's memory and no other GPU's. The host is the CPU, which
compiles for Clang on x86_64. No case compiles: a program's binary is its
source's bytes. For each case:
- `<case>_prepared.golden` is the schedule `sched_batches` receives, its
  kernels compiled and its calls prepared;
- `<case>_compiled.golden` is what `compile_linear` returns, and
  `<case>_host.golden` the source of each batch's host program, in order.

tinygrad is changed as tolk.next differs from it:
- DIVERGENCES D1, as hcq2_d1.py applies it;
- DIVERGENCES D36: a kernel's function and the stamping host function are
  read from words of the device, where tinygrad takes the address of a buffer
  placed at them;
- an address is taken on one device, as tolk.next names one device.
"""

from golden import graph, text
import hcq2_d1  # noqa: F401
from tinygrad import Tensor, Variable
from tinygrad.device import Buffer, Compiled, Compiler
from tinygrad.dtype import dtypes
from tinygrad.helpers import DEV
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.runtime.support.hcq2 import HCQInfo
from tinygrad.uop.ops import UOp
import tinygrad.engine.realize as realize
import tinygrad.runtime.ops_cuda as ops_cuda
import tinygrad.runtime.support.compiler_cuda as compiler_cuda
import tinygrad.runtime.support.hcq2 as hcq2

DEV.value = "CPU::x86_64,x86-64"
Compiler.compile_cached = lambda self, src: src.encode()
# Workers compile in processes of their own, with tinygrad's compilers.
realize.get_worker_pool = lambda: None
# Making the renderer makes NVRTC's compiler, which loads NVRTC.
compiler_cuda.NVRTCCompiler.__init__ = lambda self, arch, ptx=True, cache_key="cuda": None


# The device, without a GPU. Its memory is reached by itself alone, the host's
# by every device.

def cuda_init(self, device=""):
    Compiled.__init__(self, device, None, [CUDARenderer], None, arch="sm_89")


def get_buf(self, device):
    if not (self.device == device or self.device.startswith("CPU")): raise RuntimeError(f"{device} cannot reach {self.device}")


ops_cuda.CUDADevice.__init__ = cuda_init
Buffer.get_buf = get_buf


# One device names an address

getaddr = UOp.getaddr
UOp.getaddr = lambda self, device=None: getaddr(self, device[0] if isinstance(device, tuple) else device)


# DIVERGENCES D36

ops_cuda.CUDAQueue.extern = lambda self, tag: UOp.placeholder((1,), dtypes.uint64, 0, device=self.devs, tag=tag).index(0).load()


# Cases

def empty(n=4, device="CUDA"): return Tensor.empty(n, device=device)


def chain(x, n):
    for _ in range(n): x = (x + 1).contiguous()
    return x


CASES = {
    "chain": (lambda: chain(empty(), 3), {}),
    "chain_profile": (lambda: chain(empty(), 2), {"profile": True}),
    "variable": (lambda: (empty(10)[:Variable("v", 1, 10).bind(3)] + 1).contiguous(), {}),
    "copy_in": (lambda: (empty(device="CPU").to("CUDA") + 1).contiguous(), {}),
    "copy_in_profile": (lambda: (empty(device="CPU").to("CUDA") + 1).contiguous(), {"profile": True}),
    "host_split": (lambda: ((empty() + 1).contiguous().to("CPU") + 2).contiguous().to("CUDA") + 3, {}),
}


def compiled(case):
    """(prepared, compiled) of the case."""
    program, options = CASES[case]
    captured, sched = [], hcq2.sched_batches

    def capture(l, profile):
        captured.append(l)
        return sched(l, profile)

    hcq2.sched_batches = capture
    try:
        linear, _ = program().linear_with_vars()
        out = realize.compile_linear(linear, profile=options.get("profile", False))
    finally:
        hcq2.sched_batches = sched
    (prepared,) = captured
    return prepared, out


def host_sources(linear):
    calls = [c.without_after for c in linear.src]
    return "".join(c.body.src[2].arg for c in calls if isinstance(c.arg.aux, HCQInfo))


def declare(case):
    def prepared(): return compiled(case)[0]
    def compiled_(): return compiled(case)[1]
    def host(): return host_sources(compiled(case)[1])
    prepared.__name__, compiled_.__name__, host.__name__ = f"{case}_prepared", f"{case}_compiled", f"{case}_host"
    graph(prepared)
    graph(compiled_)
    text(host)


for case in CASES: declare(case)
