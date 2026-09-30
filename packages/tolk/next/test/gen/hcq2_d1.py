"""tinygrad's hcq2.py as tolk.next differs from it, for the generators of
batches and their host programs.

- DIVERGENCES D1: a device's signal word is one word, the value the work
  before a batch signalled is the host program's variable `submitted_<d>`, the
  value the batch signals `value_<d>`, a queue waits for the first and the
  batch stores the second, and the fence only re-arms the queue signals, after
  the first, so the slots hold no timeline slot (D7).

Importing the module patches tinygrad.
"""

from tinygrad import Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import dedup, to_tuple
from tinygrad.renderer import Estimates
from tinygrad.runtime.support.hcq2 import HCQInfo
from tinygrad.uop.ops import KernelInfo, Ops, PatternMatcher, UOp, UPat
import tinygrad.engine.realize as realize
import tinygrad.runtime.support.hcq2 as hcq2


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
                  if hcq2.all_devices_in(x, hcq2.HCQ_DEVS)} - {devs[0]}:
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
