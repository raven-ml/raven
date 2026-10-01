"""Goldens of tinygrad/runtime/ops_nv.py: batches on an NV device, encoded by
NVComputeQueue and NVCopyQueue, and their host programs.

The device NV is described without opening a GPU, as an Ada GPU (sm_89) or,
for the cases whose name ends in `_blackwell`, a Blackwell one, whose launch
descriptors are of version 5. It reaches the host's memory and no other GPU's.
The host is the CPU, which compiles for Clang on x86_64. Every program's binary
is the cubin runtime/ops_nv/simple_add_sm89.cubin, NVRTC 12.8's sm_89 code of
simple_add.cu, followed by the program's source: the encoder reads a
program's layout from its cubin. The `crafted` cases run a cubin built here
(CRAFTED) with what simple_add's lacks. For each
case:
- `<case>_prepared.golden` is the schedule `sched_batches` receives, its
  kernels compiled and its calls prepared;
- `<case>_compiled.golden` is what `compile_linear` returns, and
  `<case>_host.golden` the source of each batch's host program, in order.

The cases whose name starts with `queue_` encode one compute queue of the
simple_add kernel directly: `<case>_batch.golden` is the batch, and
`<case>_lowered.golden` and `<case>_host.golden` what `lower_call` makes of
it.

tinygrad is changed as tolk.next differs from it:
- DIVERGENCES D1, as hcq2_d1.py applies it;
- DIVERGENCES D38: a program's placeholder names its cubin and kernel, which
  the engine loads, instead of holding the image a link patch writes;
- DIVERGENCES D40: the copy engine signals the high word of a value too, when
  its low word is 0, and into a word of its own otherwise;
- DIVERGENCES D43: a launch descriptor and its constant buffer 0 are nested
  LINEARs of 256-byte alignment, the queue ends a chain of launches when a
  command stops it or the queue is submitted, and it takes its command buffer
  once it has; every region the words address, directly or through the words
  of a region, is laid out once, in the buffer of its name;
- DIVERGENCES D51: the local memory a descriptor states is a word of the
  device, which the engine fills, where tinygrad writes the device's value
  into the descriptor;
- an address is taken on one device, as tolk.next names one device.
"""

import itertools
import struct
from pathlib import Path
from types import SimpleNamespace

from golden import graph, text
import hcq2_d1  # noqa: F401
from tinygrad import Tensor, Variable
from tinygrad.device import Buffer, Compiled, Compiler
from tinygrad.dtype import dtypes
from tinygrad.helpers import DEV, Target, prod, round_up
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.runtime.support.hcq2 import HCQInfo, layout_args, patch
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp
import tinygrad.engine.realize as realize
import tinygrad.runtime.ops_nv as ops_nv
import tinygrad.runtime.support.compiler_cuda as compiler_cuda
import tinygrad.runtime.support.hcq2 as hcq2

nv_gpu = ops_nv.nv_gpu
CUBIN = (Path(__file__).resolve().parents[2] / "runtime" / "ops_nv" / "simple_add_sm89.cubin").read_bytes()

DEV.value = "CPU::x86_64,x86-64"
Compiler.compile_cached = lambda self, src: src.encode()
# A program's binary is the cubin, then its source's bytes, which the cubin's
# ELF ignores: each program has a binary of its own, as a compiler's are.
compiler_cuda.NVRTCCompiler.compile_cached = lambda self, src: CUBIN + src.encode()
# Workers compile in processes of their own, with tinygrad's compilers.
realize.get_worker_pool = lambda: None
# Making the renderer makes NVRTC's compiler, which loads NVRTC.
compiler_cuda.NVRTCCompiler.__init__ = lambda self, arch, ptx=False, cache_key="cuda": None

BLACKWELL = False


# The device, without a GPU. Its memory is reached by itself alone, the host's
# by every device.

def nv_init(self, device=""):
    self.iface = SimpleNamespace(compute_class=nv_gpu.BLACKWELL_COMPUTE_A if BLACKWELL else nv_gpu.ADA_COMPUTE_A)
    self.sass_version = 0x89
    self.shared_mem_window, self.local_mem_window = 0x729400000000, 0x729300000000
    self.slm_per_thread, self.pma_enabled = 0, False
    self.fifos = {"COMPUTE:0": SimpleNamespace(entries=0x10000, token=0x11), "COPY:0": SimpleNamespace(entries=0x10000, token=0x22)}
    Compiled.__init__(self, device, None, [CUDARenderer], None, arch="sm_89")


def get_buf(self, device):
    if not (self.device == device or self.device.startswith("CPU")): raise RuntimeError(f"{device} cannot reach {self.device}")


ops_nv.NVDevice.__init__ = nv_init
Buffer.get_buf = get_buf


# One device names an address

getaddr = UOp.getaddr
UOp.getaddr = lambda self, device=None: getaddr(self, device[0] if isinstance(device, tuple) else device)


# DIVERGENCES D43: regions of their own alignment, taken by the queue

def region(name, blob, patches):
    words, pos = [], 0
    for off, w in sorted(patches.items()):
        if off > pos: words.append(UOp(Ops.BINARY, arg=bytes(blob[pos:off])))
        words.append(w)
        pos = off + w.dtype.itemsize
    if pos < len(blob): words.append(UOp(Ops.BINARY, arg=bytes(blob[pos:])))
    return UOp(Ops.LINEAR, src=tuple(words), arg=(name, 256))


def region_name(arg): return arg[0] if isinstance(arg, tuple) else arg
def region_align(arg): return arg[1] if isinstance(arg, tuple) else 128


def contents(hq):
    stream, patches = bytes(hq.blob), hq.patches
    rt = [(o, w) for o, w in patches if isinstance(o, int) and o % 4 == 0 and w.dtype.itemsize in (4, 8) and not hcq2._is_link_patch(w)]
    uses = {w: [o for o, _ in grp] for w, grp in itertools.groupby(sorted(rt, key=lambda p: p[1].key), key=lambda p: p[1])}
    looped = {w: (at, [*w.ranges][0] if w.ranges else UOp.range(len(at), next(UOp.unique_num))) for w, at in uses.items() if len(at) > 1 or w.ranges}
    dwords = [(UOp(Ops.BINARY, arg=struct.pack(f"<{len(at)}I", *at)).bitcast(dtypes.uint32).index(r).load() + 4 * k, (w >> 32 * k).cast(dtypes.uint32))
              for w, (at, r) in looped.items() for k in range(w.dtype.itemsize // 4)]
    return stream, [(o, w) for o, w in patches if w not in looped] + dwords


# every region the words address, directly or through a region's words, is laid out once, in the buffer of its name
def bufferize_cmdbuf(hq, name, device):
    stream, patches = contents(hq)
    nested = hcq2.dedup([g.src[0] for _, w in patches for g in w.toposort() if g.op is Ops.GETADDR and g.src[0].op is Ops.LINEAR])
    bufs = []
    for lname, ls in itertools.groupby(sorted(nested, key=lambda l: region_name(l.arg)), key=lambda l: region_name(l.arg)):
        hq.blob, hq.patches = bytearray(), []
        offs = {l: (hq.q(UOp(Ops.BINARY, arg=bytes(-len(hq.blob) % region_align(l.arg)))), hq.q(*l.src)) for l in ls}
        s, p = contents(hq)
        bufs.append((offs, UOp.placeholder((len(s),), dtypes.uint8, device=hq.devs, tag=hcq2.to_name(lname, hq.queue)), s, p))
    views = {l: buf[o:e] for offs, buf, _, _ in bufs for l, (o, e) in offs.items()}

    def write(buf, s, p):
        words = UOp.sink(*[w for _, w in p]).substitute(views).src
        return patch(buf, list(zip([o for o, _ in p], words)), s)
    top = UOp.placeholder((len(stream),), dtypes.uint8, device=device, tag=hcq2.to_name(name, hq.queue))
    return write(top, stream, patches).after(*[write(b, s, p) for _, b, s, p in bufs])


def encode_submit(hq):
    for u in hq.lin.src: hq.q_rewrite.rewrite(u, ctx=hq)
    return hq.submit()


hcq2.bufferize_cmdbuf, hcq2.encode_submit, ops_nv.encode_submit = bufferize_cmdbuf, encode_submit, encode_submit


def submit(self):
    self.end_chain()
    cmdbuf = bufferize_cmdbuf(self, "cmdbuf", self.devs)
    fifo, (ib, off) = self.dev.fifos[self.queue], hcq2.unwrap_view(cmdbuf)
    ring, gpput, doorbell, put, gpentry = [UOp.placeholder((sz,), dt, device=self.devs, volatile=True, tag=hcq2.to_name(nm, self.queue))
      for nm, dt, sz in (("ring", dtypes.uint64, fifo.entries), ("gpput", dtypes.uint32, 1), ("doorbell", dtypes.uint32, 1),
                         ("put_value", dtypes.uint64, 1), ("gpentry", dtypes.uint64, 1))]
    gpentry = patch(gpentry, [(0, ib.getaddr(self.devs) + UOp.const(off | (cmdbuf.max_numel() // 4 << 42) | (1 << 41), dtypes.uint64))])
    p = put.index(0).load()
    written = UOp.barrier(ring.after(cmdbuf).index((p % fifo.entries).cast(dtypes.int)).store(gpentry.index(0).load()), put.index(0).store(p + 1))
    queued = UOp.barrier(gpput.after(written).index(0).store(((p + 1) % fifo.entries).cast(dtypes.uint32)))
    return doorbell.after(queued).index(0).store(UOp.const(fifo.token, dtypes.uint32))


ops_nv.NVQueue.submit = submit
ops_nv.NVQueue.end_chain = lambda self: None


# The local memory a descriptor states: a word of the device

build_program = ops_nv.nv_build_program

# one placeholder for every launch that needs the same local memory on the same devices: placeholders of one tag in a
# batch would become views of one buffer, which the device's word does not hold
local_words = {}


def nv_build_program(dev, prg, devs):
    if (cached := ops_nv._nv_program_cache.get((prg.src[3].arg, devs))) is not None: return cached
    required = []
    dev._ensure_has_local_memory = required.append
    data, _ = build_program(dev, prg, devs)
    if (devs, required[0]) not in local_words:
        local_words[(devs, required[0])] = UOp.placeholder((1,), dtypes.uint32, device=devs, tag=("nv_local", required[0]))
    local = local_words[(devs, required[0])].index(0).load()
    if data.qmd.ver >= 4: data.qmd.write(shader_local_memory_high_size_shifted4=local >> 4)
    else: data.qmd.write(shader_local_memory_high_size=local)
    # DIVERGENCES D38: the placeholder names the cubin and its kernel, which the engine loads
    program = UOp.placeholder((len(data.image),), dtypes.uint8, 0, device=devs, tag=("program", prg.src[3].arg, prg.to_elf().name))
    ops_nv._nv_program_cache[(prg.src[3].arg, devs)] = (data, program)
    return data, program


ops_nv.nv_build_program = nv_build_program


# The compute queue: launches in regions, chains ended when stopped

def compute_init(self, submit):
    ops_nv.NVQueue.__init__(self, submit)
    self.chain = []


def end_chain(self):
    head = None
    for launch in reversed(self.chain):
        if head is not None: launch.write(dependent_qmd0_pointer=head.getaddr(self.devs) >> 8)
        head = region("qmd", launch.mv, launch.patches)
    if head is not None:
        self.nvm(1, nv_gpu.NVC6C0_SEND_PCAS_A, (head.getaddr(self.devs) >> 8).cast(dtypes.uint32))
        self.nvm(1, nv_gpu.NVC6C0_SEND_SIGNALING_PCAS2_B, nv_gpu.NVC6C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_PREFETCH_SCHEDULE)
    self.chain = []


def compute_wait(self, signal, value):
    self.end_chain()
    ops_nv.NVQueue.wait(self, signal, value)


def compute_release(self, signal, value, timestamp=False):
    if not self.chain or not self.chain[-1].set_release(signal.getaddr(self.devs), value, timestamp):
        self.end_chain()
        ops_nv.NVQueue.release(self, signal, value, timestamp)


def memory_barrier(self):
    self.end_chain()
    self.nvm(1, nv_gpu.NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI,
             ops_nv.nv_flags("NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI", instruction="true", global_data="true", constant="true"))


def exec_(self, call, prg):
    data, lib = ops_nv.nv_build_program(self.dev, prg, self.devs)
    global_size, local_size = prg.arg.global_size, prg.arg.local_size
    if prod(local_size) > 1024 or data.max_threads < prod(local_size):
        raise RuntimeError(f"Too many resources requested for launch, {prod(local_size)=}, {data.max_threads=}")
    if any(g > mx for g, mx in zip(global_size, [2147483647, 65535, 65535]) if isinstance(g, int)) or \
       any(l > mx for l, mx in zip(local_size, [1024, 1024, 64])):
        raise RuntimeError(f"Invalid global/local dims {global_size=}, {local_size=}")
    qmd = ops_nv.QMD(self.dev, bytearray(data.qmd.mv))
    qmd.patches = dict(data.qmd.patches)
    qmd.write(**dict(zip(qmd.grid, global_size)), **{f"cta_thread_dimension{j}": l for j, l in enumerate(local_size)})
    qmd.set_program_addr(lib.getaddr(self.devs) + data.prog_off)
    bufs, vals = [ops_nv.get_call_arg_uops(call)[j] for j in prg.arg.globals], ops_nv.get_call_var_uops(call, prg)
    at = len(data.cbuf_0) * 4
    driver = bytearray(data.kernargs_size)
    driver[:at] = bytes(ops_nv.array.array('I', data.cbuf_0).tobytes())
    rows = dict(layout_args([b.getaddr(self.devs) for b in bufs] + [v.ccast(dt) for v, dt in zip(vals, data.vars)], at))
    cbuf = region("cbuf", driver, rows)
    for j, (off, _) in data.constbufs.items():
        qmd.set_constant_buf_addr(j, cbuf.getaddr(self.devs) if j == 0 else lib.getaddr(self.devs) + off)
    if self.chain: self.chain[-1].write(dependent_qmd0_action=1, dependent_qmd0_prefetch=1, dependent_qmd0_enable=1)
    self.chain.append(qmd)


for name, f in dict(__init__=compute_init, end_chain=end_chain, wait=compute_wait, release=compute_release,
                    memory_barrier=memory_barrier, exec=exec_, submit=submit).items():
    setattr(ops_nv.NVComputeQueue, name, f)


# DIVERGENCES D40

def copy_signal(self, signal, value):
    addr = signal.getaddr(self.devs)
    sink = UOp.placeholder((1,), dtypes.uint32, device=self.devs, volatile=True, tag="nv_sink")
    self.semaphore(addr, value, "one")
    self.semaphore(value.cast(dtypes.uint32).eq(UOp.const(0, dtypes.uint32)).where(addr + UOp.const(4, dtypes.uint64), sink.getaddr(self.devs)),
                   value >> 32, "one")


ops_nv.NVCopyQueue.signal = copy_signal


# Cases

def empty(n=4, device="NV"): return Tensor.empty(n, device=device)


def chain(x, n):
    for _ in range(n): x = (x + 1).contiguous()
    return x


# simple_add.cu's kernel, compiled: its launches read the cubin's own layout
# (its code's offset, its constant banks, its registers and stack).

def simple_add(n=32, global_size=(1, 1, 1), local_size=(32, 1, 1), binary=CUBIN, name="simple_add"):
    params = [UOp.param(i, dtypes.int32, shape=(n,), device="NV") for i in range(3)]
    var = UOp.variable("n", 1, n, dtype=dtypes.int32)
    formal = var.replace(op=Ops.PARAM)
    prg = UOp(Ops.PROGRAM, src=(UOp(Ops.SINK, arg=KernelInfo(name=name)), UOp(Ops.LINEAR, src=(*params, formal)),
                                UOp(Ops.SOURCE, arg=""), UOp(Ops.BINARY, arg=binary)),
              arg=ProgramInfo(global_size=global_size, local_size=local_size, globals=(0, 1, 2), vars=(formal,), outs=(0,), ins=(1, 2),
                              target=Target("NV", "CUDA", "sm_89")))
    return prg, var


# A cubin built here for the kernel k, to reach what simple_add's does not: 128
# registers, shared memory, a stack, a parameter bank of its own size, constant
# bank 3, attributes of each format, and the three relocations of the program's
# own address.

PROGBITS, SYMTAB, STRTAB, RELA, NOBITS = 1, 2, 3, 4, 8


def elf64(sections, symbols):
    """An ELF object whose sections are (name, type, align, content, link,
    info) after the null section, then .symtab of `symbols` ((section index,
    value)), then .shstrtab."""
    # the symbols are nameless: their names are .shstrtab's empty string
    secs = [*sections, (".symtab", SYMTAB, 8, bytes(24) + b"".join(struct.pack("<IBBHQQ", 0, 0, 0, i, v, 0) for i, v in symbols),
                        len(sections) + 2, 0)]
    names, offsets = b"\0", []
    for sec in [*secs, (".shstrtab",)]:
        offsets.append(len(names))
        names += sec[0].encode() + b"\0"
    secs.append((".shstrtab", STRTAB, 1, names, 0, 0))
    data, placed = b"", []
    for sec in secs:
        placed.append(64 + len(data))
        data += sec[3]
    header = b"\x7fELF" + bytes([2, 1, 1]) + bytes(9) + struct.pack("<HHIQQQIHHHHHH", 1, 0, 1, 0, 0, 64 + len(data), 0, 64, 0, 0, 64,
                                                                   len(secs) + 1, len(secs))
    shdrs = bytes(64)
    for (nm, typ, align, content, link, info), off, n in zip(secs, placed, offsets):
        entsize = {SYMTAB: 24, RELA: 24}.get(typ, 0)
        shdrs += struct.pack("<IIQQQQIIQQ", n, typ, 0, 0, off, len(content), link, info, align, entsize)
    return header + data + shdrs


def attr(typ, param, value):
    """An .nv.info attribute: its data when of format 4, else a 16-bit value."""
    return struct.pack("<BBH", typ, param, len(value)) + value if typ == 4 else struct.pack("<BBH", typ, param, value)


CRAFTED = elf64([
    (".text.k", PROGBITS, 128, bytes(range(256)) * 3, 0, 0),
    (".nv.constant0.k", PROGBITS, 4, bytes(0x180), 0, 0),
    (".nv.constant3", PROGBITS, 4, bytes(range(16)), 0, 0),
    (".nv.shared.k", NOBITS, 16, bytes(0x800), 0, 0),
    (".nv.info", PROGBITS, 4, attr(4, 0x2f, struct.pack("<II", 0, 128)) + attr(3, 0x1b, 0x0102) + attr(4, 0x12, struct.pack("<II", 0, 0x100))
     + attr(1, 0x05, 0), 0, 0),
    (".nv.info.k", PROGBITS, 4, attr(4, 0x0a, struct.pack("<IHH", 0, 0x1f0, 0x20)), 0, 0),
    (".rela.text.k", RELA, 8, b"".join(struct.pack("<QQq", off, (sym << 32) | typ, addend)
                                         for off, sym, typ, addend in [(0x10, 1, 2, 0), (0x20, 2, 0x38, 8), (0x30, 2, 0x39, 8)]), 8, 1),
], [(3, 0), (1, 0x40)])


class Linear:
    """A schedule of calls, as linear_with_vars returns one."""
    def __init__(self, calls): self.calls = calls
    def linear_with_vars(self): return UOp(Ops.LINEAR, src=tuple(self.calls)), {}


def simple_adds(k, **kw):
    prg, var = simple_add(**kw)
    bufs = [UOp.new_buffer("NV", 32, dtypes.int32) for _ in range(3)]
    return Linear([prg.call(*bufs, var.bind(32)) for _ in range(k)])


CASES = {
    "simple_add": (lambda: simple_adds(1), {}),
    "simple_add_blackwell": (lambda: simple_adds(1), {"blackwell": True}),
    "simple_add_chain": (lambda: simple_adds(3), {}),
    "simple_add_chain_blackwell": (lambda: simple_adds(3), {"blackwell": True}),
    "simple_add_profile": (lambda: simple_adds(2), {"profile": True}),
    "simple_add_grid": (lambda: simple_adds(1, global_size=(4, 3, 2), local_size=(8, 4, 1)), {}),
    "crafted": (lambda: simple_adds(2, binary=CRAFTED, name="k"), {}),
    "crafted_blackwell": (lambda: simple_adds(2, binary=CRAFTED, name="k"), {"blackwell": True}),
    "chain": (lambda: chain(empty(), 3), {}),
    "chain_blackwell": (lambda: chain(empty(), 3), {"blackwell": True}),
    "chain_profile": (lambda: chain(empty(), 2), {"profile": True}),
    "variable": (lambda: (empty(10)[:Variable("v", 1, 10).bind(3)] + 1).contiguous(), {}),
    "copy_in": (lambda: (empty(device="CPU").to("NV") + 1).contiguous(), {}),
    "copy_out_profile": (lambda: (empty() + 1).contiguous().to("CPU"), {"profile": True}),
    # a copy of more than 2 GiB goes in lines of at most 2 GiB
    "copy_large": (lambda: empty((1 << 29) + 4, device="CPU").to("NV").contiguous(), {}),
    "host_split": (lambda: ((empty() + 1).contiguous().to("CPU") + 2).contiguous().to("NV") + 3, {}),
}


def with_device(options):
    global BLACKWELL
    BLACKWELL = options.get("blackwell", False)


def compiled(case):
    """(prepared, compiled) of the case."""
    program, options = CASES[case]
    with_device(options)
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
