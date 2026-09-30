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
- DIVERGENCES D1: a device's signal word is one word, the value the work
  before a batch signalled is the host program's variable `submitted_<d>`, the
  value the batch signals `value_<d>`, a queue waits for the first and the
  batch stores the second, and the fence only re-arms the queue signals, after
  the first, so the slots hold no timeline slot;
- NullQueue writes a variable by value, as a variable has no address, and
  takes an address on its first device, as tolk.next names one device.
"""

from golden import graph, text
from tinygrad import Device, Tensor, Variable, dtypes
from tinygrad.device import Compiler
from tinygrad.helpers import DEV, dedup, to_tuple
from tinygrad.renderer import Estimates
from tinygrad.runtime.ops_cpu import CPUDevice
from tinygrad.runtime.ops_null import NullQueue, EXEC, TIMESTAMP
from tinygrad.runtime.support.hcq2 import HCQInfo
from tinygrad.uop.ops import KernelInfo, Ops, PatternMatcher, UOp, UPat
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


# DIVERGENCES D1

def signal_word(devs): return UOp.placeholder((1,), dtypes.uint64, 0, device=devs, volatile=True, tag="timeline")
def submitted(d): return UOp.variable(hcq2.to_name("submitted", d), 0, 2**62 - 1, dtypes.uint64)
def value(d): return UOp.variable(hcq2.to_name("value", d), 0, 2**62 - 1, dtypes.uint64)


def ins(name, *src): return UOp(Ops.INS, arg=(name, dtypes.void), src=src)


def post_init(self):
    self.queues, self.last, self.prev, self.peers = {}, {}, [], {}
    for tag, (c, devs, q) in enumerate(self.batch):
        if q not in self.queues.setdefault(devs[0], []): self.queues[devs[0]].append(q)
        self.prev.append(self.last.get((devs[0], q)))
        self.last[(devs[0], q)] = tag
        for d in {Device.canonicalize(x) for b in realize.get_call_arg_uops(c) for x in to_tuple(b.device)
                  if hcq2.all_devices_in(x, None)} - {devs[0]}:
            self.peers.setdefault((devs[0], q), set()).add(d)
            self.queues.setdefault(d, [])
    self.signal_tags = {tag for (dev, q), tag in self.last.items() if q != self.epilogue_queue(dev) or (dev, q) in self.peers}
    self.slots = {dev: UOp.placeholder((2 * (len(qs) + (2 * len(self.batch) if self.profile else 0)),), dtypes.uint64, device=(dev,),
                                       volatile=True, tag="slots") for dev, qs in self.queues.items()}


def start_ins(ctx, dev, queue):
    return [ins("barrier")] + [ins("wait", signal_word((d,)), submitted(d)) for d in [dev, *sorted(ctx.peers.get((dev, queue), ()))]]


def build_queues(ctx):
    call_waits = [hcq2._wait_ins(ctx, c, d[0], q, tag) for tag, (c, d, q) in enumerate(ctx.batch)]
    queues = {}
    for tag, ((call, devices, queue), waits) in enumerate(zip(ctx.batch, call_waits)):
        if not (q := queues.setdefault((devices, queue), [])): q += start_ins(ctx, devices[0], queue)
        ts_ins = [ins("timestamp", ctx.slot(devices, i)) for i in ctx.stamps(devices, tag)]
        q += waits + ts_ins[:1] + [call] + ts_ins[1:]
        if tag in ctx.signal_tags: q += [ins("store", ctx.queue_signal(devices, queue), UOp.const(tag + 1, dtypes.uint64))]
    for dev in ctx.queues:
        queue = ctx.epilogue_queue(dev)
        waits = [ins("wait", ctx.queue_signal((d,), q), UOp.const(ctx.last[(d, q)] + 1, dtypes.uint64))
                 for d, q in [(dev, q) for q in ctx.queues[dev] if q != queue] + sorted(k for k, ds in ctx.peers.items() if dev in ds)]
        if not (q := queues.setdefault(((dev,), queue), [])) and dev not in {d for d, _ in ctx.last}: q += start_ins(ctx, dev, queue)
        q.extend([*waits, ins("store", signal_word((dev,)), value(dev))])
    return queues


def finalize_batch(ctx):
    queues = build_queues(ctx)
    submits = []
    fence = UOp.custom_function("hcq_fence", *[ctx.queue_signal((dev,), q) for dev, qs in ctx.queues.items() for q in qs])
    for (devs, queue), cmds in queues.items(): submits.append(hcq2.make_submit(*cmds, devs=devs, queue=queue).after(fence, *submits[-1:]))
    sink = UOp.sink(*submits, arg=KernelInfo("hcq_submit", estimates=Estimates()), tag=1)
    names = [realize.get_call_name(c, realize.get_call_arg_uops(c)) for c, _, _ in ctx.batch]
    estimates = [realize.estimate_uop(c) for c, _, _ in ctx.batch]
    stamps = [tuple(2 * s + 1 for s in ctx.stamps(d, tag)) for tag, (_, d, _) in enumerate(ctx.batch)]
    profile_keys = [c.body.key if c.body.op is Ops.PROGRAM else None for c, _, _ in ctx.batch]
    args = [[hcq2.unwrap_lane(bs[g])[:2] for g in getattr(c.body.arg, "globals", range(len(bs)))]
            for c, _, _ in ctx.batch for bs in [realize.get_call_arg_uops(c)]]
    bufs = [tuple(b.arg.slot for b, _ in a) if all(b.op is Ops.PARAM and lane is None for b, lane in a) else () for a in args]
    kerns = tuple(zip([d for _, d, _ in ctx.batch], names, estimates, stamps, profile_keys, bufs,
                      [realize.get_call_outs_ins(c) for c, _, _ in ctx.batch]))
    written_bufs = tuple(dedup(b for c, _, _ in ctx.batch for b in realize.get_call_written_bufs(c)))
    info = HCQInfo(tuple(ctx.queues), kernels=kerns, written_bufs=written_bufs, estimates=sum(estimates, start=Estimates()).simplify())
    return sink.call(*(ctx.slots.values() if ctx.profile else ()), aux=info)


def hcq_fence(f):
    # the re-arms are ordered after the values the work before the batch
    # signalled, so that none is a link patch and each runs on every run
    last = tuple(submitted(d) for d in sorted({to_tuple(s.device)[0] for s in f.src}))
    for sig in f.src:
        base, off = hcq2.unwrap_view(sig)
        last = (base.after(*last).index(off // sig.dtype.itemsize).store(0),)
    return last[0].barrier(*last[1:])


hcq2.timeline = signal_word
hcq2.BatchCtx.__post_init__ = post_init
hcq2.BatchCtx.stamps = lambda self, devs, tag: (st := len(self.queues[devs[0]]) + 2 * tag, st + 1) if self.profile else ()
hcq2._start_ins, hcq2._build_queues, hcq2._finalize_batch = start_ins, build_queues, finalize_batch
hcq2.pm_hcq_encode = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg="hcq_fence", name="f"), hcq_fence),
                                     *hcq2.pm_hcq_encode.patterns[1:]])


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
