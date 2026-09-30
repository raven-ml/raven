"""Goldens of tinygrad/runtime/ops_metal.py: batches on a Metal device, encoded
by MetalQueue, and their host programs.

The device METAL is described without opening a GPU: its GPU family and
whether it has a residency set are each case's, and its programs compile for
Metal. The host is the CPU, which compiles for Clang on x86_64. No case
compiles: a program's binary is its source's bytes. For each case:
- `<case>_prepared.golden` is the schedule `sched_batches` receives, its
  kernels compiled and its calls prepared;
- `<case>_compiled.golden` is what `compile_linear` returns, and
  `<case>_host.golden` the source of each batch's host program, in order.

tinygrad is changed as tolk.next differs from it:
- DIVERGENCES D1, as hcq2_d1.py applies it, and D7: a batch's slots hold no
  timeline slot, so a command's stamps are words 3 and 5 of its slots, not 5
  and 7;
- DIVERGENCES D34: a command buffer waits in its stamps retained, after the
  release of one an earlier run left there unread;
- MetalQueue takes an address on its first device, as tolk.next names one
  device.
"""

from types import SimpleNamespace

from golden import graph, text
import hcq2_d1  # noqa: F401
from tinygrad import Device, Tensor, Variable
from tinygrad.device import Compiled, Compiler
from tinygrad.dtype import dtypes
from tinygrad.helpers import DEV, dedup, round_up
from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.runtime.support.hcq2 import HCQInfo, ccall, layout_args, patch
from tinygrad.uop.ops import UOp
import tinygrad.engine.realize as realize
import tinygrad.runtime.ops_metal as ops_metal
import tinygrad.runtime.support.hcq2 as hcq2

DEV.value = "CPU::x86_64,x86-64"
Compiler.compile_cached = lambda self, src: src.encode()
# Workers compile in processes of their own, with tinygrad's compilers.
realize.get_worker_pool = lambda: None
# Metal's code generation service makes the forked processes crash at random.
ops_metal.MetalCompiler.support.MTLCodeGenServiceCreate = lambda name: None


# The device, without a GPU: its family and residency set are the case's.

def metal_init(self, device=""):
    Compiled.__init__(self, device, None, [MetalRenderer], None, arch="Apple9")
    self.residency = SimpleNamespace(value=1)


ops_metal.MetalDevice.__init__ = metal_init


# One device names an address

def exec_(self, call, prg):
    bufs, vals, obj = realize.get_call_arg_uops(call), realize.get_call_var_uops(call, prg), prg.to_elf()
    args = [bufs[i].getaddr(self.devs[0]) for i in prg.arg.globals] + [v.ccast(var.dtype) for v, var in zip(vals, prg.arg.vars)]
    self.rows += (rows := layout_args(args, off := round_up(self.nbytes, 256)))
    self.nbytes = max([o + w.dtype.itemsize for o, w in rows], default=off + 8)
    dims = (*prg.arg.global_size, *prg.arg.local_size)
    if any(isinstance(d, UOp) for d in dims):
        self.sizes.append((len(self.cmds), at := round_up(self.nbytes, 8)))
        self.rows += layout_args([d.cast(dtypes.uint64) if isinstance(d, UOp) else UOp.const(d, dtypes.uint64) for d in dims], at)
        self.nbytes = at + 48
    self.cmds.append((obj.lib, obj.name, tuple(1 if isinstance(d, UOp) else int(d) for d in dims), off))


ops_metal.MetalQueue.exec = exec_


# DIVERGENCES D7 and D34

ops_metal.SELECTORS += ("retain", "release")
msg_send, mtl_sel, mtl_cb, mtl_enc, mtl_msg = (ops_metal.MSGSEND, ops_metal.mtl_sel, ops_metal.mtl_cb, ops_metal.mtl_enc,
                                                ops_metal.mtl_msg)


def submit(self, cmdbuf):
    n, zero, pipes = len(self.cmds), round_up(self.nbytes, 8), dedup(c[:2] for c in self.cmds)
    buf = UOp.placeholder((zero + 24 + 8 * (1 + n + len(pipes)),), dtypes.uint8, device=self.devs, volatile=True,
                          tag=("mtl_icb", tuple(self.cmds), zero + 24))
    args = patch(buf, self.rows + [(zero + 8 * i, UOp.const(0, dtypes.uint64)) for i in range(3)])
    header, cb, enc, h = args.bitcast(dtypes.uint64)[zero // 8 + 3:], mtl_cb(self.devs), mtl_enc(self.devs), args

    for ci, off in self.sizes:
        h = mtl_msg(h, header.index(1 + ci), "concurrentDispatchThreadgroups:threadsPerThreadgroup:", args.index(off), args.index(off + 24))

    def run(h, first, count, last):
        h = cb.store(cbuf := mtl_msg(h, mtl_sel(self.devs, "queue"), "commandBuffer", ret=ops_metal.ctypes.c_void_p))
        h = enc.store(mtl_msg(h, cb, "computeCommandEncoder", ret=ops_metal.ctypes.c_void_p))
        h = mtl_msg(h, enc, "waitForFence:", mtl_sel(self.devs, "fence").load())
        if self.dev.residency.value is None:
            h = mtl_msg(h, enc, "useResources:count:usage:", mtl_sel(self.devs, "resources").load(), mtl_sel(self.devs, "count").load(), 3)
        if not self.dev.arch.startswith("Apple") or int(self.dev.arch[5:]) < 9:
            r = UOp.range(len(pipes), next(UOp.unique_num), dtype=dtypes.uint64)
            h = mtl_msg(h, enc, "setComputePipelineState:", header.index(1 + n + r).load())
            h = mtl_msg(h, enc, "dispatchThreadgroups:threadsPerThreadgroup:", args.index(zero), args.index(zero)).end(r)
        h = mtl_msg(h, enc, "executeCommandsInBuffer:withRange:", header.after(h).index(0).load(), first, count)
        h = mtl_msg(h, enc, "updateFence:", mtl_sel(self.devs, "fence").load())
        h = mtl_msg(h, enc, "endEncoding")
        if self.stamps:
            slots = self.stamps[0].src[0]
            start, end = slots.after(h).index(3 + 4 * first).load(), slots.after(h).index(5 + 4 * first).load()
            unread = end.eq(UOp.const(0, dtypes.uint64)).where(start, UOp.const(0, dtypes.uint64))
            h = ccall(msg_send[None], unread, mtl_sel(self.devs, "release").load())
            h = slots.after(h).index(5 + 4 * first).store(0)
            h = slots.after(h).index(3 + 4 * first).store(ccall(msg_send[ops_metal.ctypes.c_void_p], cbuf, mtl_sel(self.devs, "retain").load()))
        if last: h = mtl_msg(h, cb, "encodeSignalEvent:value:", mtl_sel(self.devs, "event").load(), self.value)
        return mtl_msg(h, cb, "commit")

    if not self.stamps: return run(h, 0, n, True)
    if n > 1: h = run(h.after(r := UOp.range(n - 1, next(UOp.unique_num), dtype=dtypes.uint64)), r, 1, False).end(r)
    return run(h, n - 1, 1, True)


ops_metal.MetalQueue.submit = submit


# Cases

def empty(n=4): return Tensor.empty(n, device="METAL")


def chain(x, n):
    for _ in range(n): x = (x + 1).contiguous()
    return x


CASES = {
    "chain": (lambda: chain(empty(), 3), {}),
    "chain_apple7": (lambda: chain(empty(), 3), {"arch": "Apple7"}),
    "chain_mac2": (lambda: chain(empty(), 2), {"arch": "Mac2"}),
    "chain_no_residency_set": (lambda: chain(empty(), 2), {"residency_set": False}),
    "chain_profile": (lambda: chain(empty(), 3), {"profile": True}),
    "one_profile": (lambda: chain(empty(), 1), {"profile": True}),
    "variable": (lambda: (empty(10)[:Variable("v", 1, 10).bind(3)] + 1).contiguous(), {}),
    "variable_second": (lambda: ((empty(10) + 1).contiguous()[:Variable("v", 1, 10).bind(3)] * 2).contiguous(), {}),
    "host_split": (lambda: ((empty() + 1).contiguous().to("CPU") + 2).contiguous().to("METAL") + 3, {}),
}


def compiled(case):
    """(prepared, compiled) of the case."""
    program, options = CASES[case]
    metal = Device["METAL"]
    metal.arch = options.get("arch", "Apple9")
    metal.residency = SimpleNamespace(value=1 if options.get("residency_set", True) else None)
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
