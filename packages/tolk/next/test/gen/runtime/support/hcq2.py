"""Goldens of tinygrad/runtime/support/hcq2.py: batches of calls on devices
with command queues, and their host programs.

The devices are those of hcq2_null.py: CPU:1, CPU:2 and CPU:3 have the NULL
device's queues, and the CPU is their host. For each case:
- `<case>.golden` is the schedule of a `Tensor` program on those devices, the
  input of `compile_linear`;
- `<case>_prepared.golden` is the schedule `sched_batches` receives, its
  kernels compiled and its calls prepared, and `<case>_batched.golden` what it
  returns;
- `<case>_compiled.golden` is what `compile_linear` returns, and
  `<case>_host.golden` the source of each batch's host program, in order.
"""

from golden import graph, text
from hcq2_null import QUEUED
from tinygrad import Tensor, Variable
from tinygrad.runtime.ops_cpu import CPUDevice
from tinygrad.runtime.support.hcq2 import HCQInfo
import tinygrad.engine.realize as realize
import tinygrad.runtime.support.hcq2 as hcq2


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
