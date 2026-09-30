"""Goldens of tinygrad/runtime/support/hcq2.py: batches of calls on devices
with command queues, and their host programs.

The devices with queues are CPU:1, CPU:2 and CPU:3, whose queues are encoded
with the NULL device's commands (runtime/ops_null.py's NullQueue) and whose
host is the CPU; each compiles for Clang on x86_64. No case compiles: a
program's binary is its source's bytes. For each case:
- `<case>.golden` is the schedule of a `Tensor` program on those devices, the
  input of `compile_linear`;
- `<case>_prepared.golden` is the schedule `sched_batches` receives, its
  kernels compiled and its calls prepared, and `<case>_batched.golden` what it
  returns;
- `<case>_compiled.golden` is what `compile_linear` returns, and
  `<case>_host.golden` the source of each batch's host program, in order.

tinygrad is changed as tolk.next differs from it:
- DIVERGENCES D1, as hcq2_d1.py applies it;
- NullQueue writes a variable by value, as a variable has no address, and
  takes an address on its first device, as tolk.next names one device.
"""

from golden import graph, text
import hcq2_d1  # noqa: F401
from tinygrad import Tensor, Variable, dtypes
from tinygrad.device import Compiler
from tinygrad.helpers import DEV, to_tuple
from tinygrad.runtime.ops_cpu import CPUDevice
from tinygrad.runtime.ops_null import NullQueue, EXEC, TIMESTAMP
from tinygrad.runtime.support.hcq2 import HCQInfo
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat
import tinygrad.engine.realize as realize
import tinygrad.runtime.support.hcq2 as hcq2

DEV.value = "CPU::x86_64,x86-64"
Compiler.compile_cached = lambda self, src: src.encode()
# Workers compile in processes of their own, with tinygrad's compilers.
realize.get_worker_pool = lambda: None

QUEUED = ("CPU:1", "CPU:2", "CPU:3")
hcq2.all_devices_in = lambda d, c: all(x in QUEUED for x in to_tuple(d))


# The NULL device's queues, for the CPU devices

def cmd(self, op, *args):
    words = [(a if a.is_variable else a.getaddr(self.devs[0])) if isinstance(a, UOp) else UOp.const(a, dtypes.uint64)
             for a in (op, *args, 0, 0, 0)]
    self.q(*words[:4])


def exec_(self, call, prg):
    args = [a.getaddr(self.devs[0]) for a in realize.get_call_arg_uops(call)] + \
           [v.cast(dtypes.uint64) for v in realize.get_call_var_uops(call, prg)]
    kernargs = UOp(Ops.LINEAR, src=tuple(hcq2.pack_args(hcq2.layout_args(args), 8 * max(len(args), 1))), arg="kernargs")
    self.cmd(EXEC, kernargs, len(args), self.event(self.devs[0], prg.src[0].arg.function_name, prg.key))


NullQueue.cmd, NullQueue.exec = cmd, exec_
NullQueue.timestamp = lambda self, signal: self.cmd(TIMESTAMP, signal.getaddr(self.devs[0]) + UOp.const(8, dtypes.uint64))
CPUDevice.pm_encode = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg=f"submit_cpu_{q}", name="submit"),
                                       lambda submit: hcq2.encode_submit(NullQueue(submit))) for q in ("compute", "copy")])


# Cases

def empty(n=4, device="CPU:1"): return Tensor.empty(n, device=device)


def chain(x, n):
    for _ in range(n): x = (x + 1).contiguous()
    return x


def sharded(n, devices):
    return Tensor.empty(n, device=devices[0]).shard(devices, axis=0).contiguous()


CASES = {
    "chain": (lambda: chain(empty(), 3), {}),
    "chain_profile": (lambda: chain(empty(), 2), {"profile": True}),
    "peer_copy": (lambda: (empty().to("CPU:2") + 1).contiguous(), {}),
    "peer_copy_kernel": (lambda: (empty().to("CPU:2") + 1).contiguous(), {"copy_queue": False}),
    "peer_copy_profile": (lambda: (empty().to("CPU:2") + 1).contiguous(), {"profile": True}),
    "sharded": (lambda: (sharded(8, ("CPU:1", "CPU:2")) + 1).contiguous(), {}),
    "host_split": (lambda: ((empty() + 1).contiguous().to("CPU") + 2).contiguous().to("CPU:1") + 3, {}),
    "sharded_sum": (lambda: (sharded(6, QUEUED) + 1).sum(0).contiguous(), {}),
    "copies": (lambda: (empty().to("CPU:2") + 1).to("CPU:3").contiguous().to("CPU:1") + 1, {}),
    "variable": (lambda: (empty(10)[:Variable("v", 1, 10).bind(3)] + 1).contiguous(), {}),
}


def schedule(case):
    program, _ = CASES[case]
    linear, _ = program().linear_with_vars()
    return linear


def compiled(case):
    """(prepared, batched, compiled) of the case."""
    _, options = CASES[case]
    captured, sched = [], hcq2.sched_batches

    def capture(l, profile):
        captured.append((l, out := sched(l, profile)))
        return out

    has_copy_queue = CPUDevice.has_copy_queue
    CPUDevice.has_copy_queue = property(lambda _: options.get("copy_queue", True))
    hcq2.sched_batches = capture
    try:
        out = realize.compile_linear(schedule(case), profile=options.get("profile", False))
    finally:
        hcq2.sched_batches, CPUDevice.has_copy_queue = sched, has_copy_queue
    (prepared, batched), = captured
    return prepared, batched, out


def host_sources(linear):
    calls = [c.without_after for c in linear.src]
    return "".join(c.body.src[2].arg for c in calls if isinstance(c.arg.aux, HCQInfo))


def declare(case):
    def input_(): return schedule(case)
    def prepared(): return compiled(case)[0]
    def batched(): return compiled(case)[1]
    def compiled_(): return compiled(case)[2]
    def host(): return host_sources(compiled(case)[2])
    input_.__name__, prepared.__name__, batched.__name__ = case, f"{case}_prepared", f"{case}_batched"
    compiled_.__name__, host.__name__ = f"{case}_compiled", f"{case}_host"
    for fn in (input_, prepared, batched, compiled_): graph(fn)
    text(host)


for case in CASES: declare(case)
