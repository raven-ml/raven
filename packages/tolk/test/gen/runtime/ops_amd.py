"""Goldens of tinygrad/runtime/ops_amd.py: batches on AMD GPUs, encoded by
AMDComputeQueue, AMDComputeAQLQueue and AMDSDMAQueue, and their host
programs.

The device AMD is described without opening a GPU: its graphics target, the
versions of its blocks, its dies, compute units and queues are each case's
(GPUS). Its kernels are rendered for HIP, and each compiles to a real AMD code
object of `ops_amd_fixtures/`, whose descriptor shapes the packets:
`simple_add_<arch>`, or, in the cases that ask for it, `scratch_<arch>`, a
kernel that reads its dispatch packet and scratch memory.
`simple_add_gfx1100.hsaco` is the old tolk's fixture, compiled from HIP by
comgr; the others are the modules of `kernels.ll`, compiled once by tinygrad's
AMDLLVMCompiler (LLVM 21). The host is the CPU, which compiles for Clang on
x86_64; no host program compiles: its binary is its source's bytes. For each case:
- `<case>_prepared.golden` is the schedule `sched_batches` receives, its
  kernels compiled and its calls prepared;
- `<case>_compiled.golden` is what `compile_linear` returns, and
  `<case>_host.golden` the source of each batch's host program, in order.
`signal_words.golden` is the words of waits and signals on a device's signal
word at values that carry into its high half, and on a queue's signal.

The cases that trace (`*traces*`) take tinygrad's default traces: 256 MiB of
each shader engine's traces over the log's runs.
The cases that count (`counters*`) take tinygrad's default counters of their
GPU, with a profile log of PROF_SLOTS runs; the work-group processor 2 of the
shader engine 1 is inactive.

tinygrad is changed as tolk differs from it:
- a device's signal word is one word, as hcq2_d1.py applies it;
- a wait on a device's signal word is for equality of its
  low 32 bits, and a signal of one writes all 64 bits: in one write on the
  compute queue, and on a copy queue its high half after its low half when the
  low half is 0, four NOPs otherwise;
- a program's placeholder names its code object and kernel,
  which the engine loads, instead of holding the image a link patch writes;
- the grid of a dispatch packet in a kernel's arguments is
  words, 32-bit constants when it is known;
- a counted run's entry in the profile log is three words: its kernel
  descriptor's address, which the device maps to the kernel's name, and the
  GPU's clock before and after the run, so that its counters are timed by the
  run itself;
- a dispatch's thread-trace marker numbers the dispatches of its queue from 0,
  so that a batch's packets depend on the batch alone;
- an address is taken on the queue's first device, as tolk names one
  device;
- a memory barrier invalidates the GPU's caches only: the host flushes the
  host data path before each submission.
"""

from types import SimpleNamespace

from golden import graph, table, text
import hcq2_d1  # noqa: F401
from tinygrad import Tensor, Variable
from tinygrad.device import Buffer, Compiled, Compiler
from tinygrad.dtype import dtypes, truncate
from tinygrad.helpers import DEV
from pathlib import Path
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.runtime.support.compiler_amd import HIPCompiler
from tinygrad.runtime.support.hcq2 import HCQInfo
from tinygrad.uop.ops import Ops, UOp, exec_alu
from tinygrad import Device
import tinygrad.engine.realize as realize
import tinygrad.runtime.ops_amd as ops_amd
import tinygrad.runtime.support.hcq2 as hcq2

DEV.value = "CPU::x86_64,x86-64"
# The host's programs are not compiled: a binary is its source's bytes. An AMD
# kernel's binary is a fixture of its target.
Compiler.compile_cached = lambda self, src: src.encode()
FIXTURES, KERNEL = Path(__file__).parent / "ops_amd_fixtures", SimpleNamespace(value="simple_add")


def hip_init(self, arch):
    self.arch = arch
    Compiler.__init__(self, None)


def fixture(self, src):
    name = f"{KERNEL.value}_{self.arch}"
    return (FIXTURES / (f"{name}.hsaco" if name == "simple_add_gfx1100" else f"{name}.o")).read_bytes()


HIPCompiler.__init__, HIPCompiler.compile_cached = hip_init, fixture
# Workers compile in processes of their own, with tinygrad's compilers.
realize.get_worker_pool = lambda: None


# The GPUs, without hardware: (target, sdma version, dies, shader engines,
# compute units, aql, copy queues).

GPUS = {
    "gfx1100": ((11, 0, 0), (6, 0, 0), 1, 6, 48, False, 2),
    "gfx1201": ((12, 0, 1), (7, 0, 0), 1, 4, 32, False, 1),
    "gfx942": ((9, 4, 2), (4, 4, 2), 8, 4, 38, True, 1),
    "gfx942_cpx": ((9, 4, 2), (4, 4, 2), 1, 4, 38, False, 1),
    "gfx1100_sdma5": ((11, 0, 0), (5, 0, 0), 1, 6, 48, False, 1),
    "gfx1100_sdma52": ((11, 0, 0), (5, 2, 0), 1, 6, 48, False, 1),
    "gfx1100_no_sdma": ((11, 0, 0), (6, 0, 0), 1, 6, 48, False, 0),
}
RING, SCRATCH_SLOTS = 16 << 20, 32
GPU, COUNTERS, TRACES = SimpleNamespace(value="gfx1100"), SimpleNamespace(value=False), SimpleNamespace(value=False)
GC = {9: (9, 4, 3), 11: (11, 0, 0), 12: (12, 0, 0)}
CU_PER_SIMD_ARRAY, PROF_SLOTS = 4, 32


def amd_init(self, device=""):
    target, sdma, xccs, ses, cus, aql, copies = GPUS[GPU.value]
    self.target, self.arch = target, "gfx%d%x%x" % target
    self.iface = SimpleNamespace(props={"lds_size_in_kb": 64, "max_slots_scratch_cu": SCRATCH_SLOTS,
                                        "cu_per_simd_array": CU_PER_SIMD_ARRAY},
                                 is_wgp_active=lambda xcc, se, sa, wgp: (se, wgp) != (1, 2))
    self.xccs, self.se_cnt, self.cu_cnt, self.is_aql = xccs, ses, cus, int(aql)
    self.soc, self.pm4 = ops_amd.import_soc(target), ops_amd.importlib.import_module(
        f"tinygrad.runtime.autogen.am.pm4_{'soc15' if target[0] == 9 else 'nv'}")
    self.sdma = ops_amd.import_module("sdma", min(sdma, (6, 0, 0)))
    offsets = ops_amd.importlib.import_module(f"tinygrad.runtime.autogen.am.{'vega' if target[0] == 9 else 'navi'}_offsets")
    base = lambda ip, n: {i: tuple(getattr(offsets, f"{ip}_BASE__INST{i}_SEG{s}", 0) for s in range(n)) for i in range(6)}
    self.gc = ops_amd.AMDIP("gc", GC[target[0]], bases=base("GC", 6))
    self.nbio = ops_amd.AMDIP("nbio" if target[0] < 12 else "nbif", {9: (7, 9, 0), 11: (4, 3, 0), 12: (6, 3, 1)}[target[0]],
                              bases=base("NBIO", 9))
    self.max_copy_size = 0x40000000 if (4, 4, 2) <= sdma < (5, 0, 0) or sdma >= (5, 2, 0) else 0x400000
    self.pmc_enabled, self.sqtt_enabled = COUNTERS.value, TRACES.value
    self.prof_slots, self.sqtt_next_cmd_id = PROF_SLOTS, ops_amd.itertools.count(0)
    self.sqtt_ses, self.sqtt_win = ses * xccs, (256 << 20) // PROF_SLOTS
    if self.pmc_enabled:
        self.pmc_counters = ops_amd.import_pmc(target)
        l2, lds = ("TCC", "SQ") if target[0] == 9 else ("GL2C", "SQC")
        self.pmc_names = ["SQ_BUSY_CYCLES", "SQ_INSTS_VALU", "SQ_INSTS_SALU", f"{lds}_LDS_IDX_ACTIVE",
                          f"{lds}_LDS_BANK_CONFLICT", "GRBM_GUI_ACTIVE", f"{l2}_HIT", f"{l2}_MISS"]
    self.copies = copies
    Compiled.__init__(self, device, None, [HIPRenderer], None, arch=self.arch)
    self.compute_queue = SimpleNamespace(ring=SimpleNamespace(size=RING // 4, dtype=dtypes.uint32))


ops_amd.AMDDevice.__init__ = amd_init
ops_amd.AMDDevice.finalize = lambda self: None
ops_amd.AMDDevice.has_copy_queue = property(lambda self: self.copies > 0)
ops_amd.AMDDevice.sdma_queue = lambda self, idx: SimpleNamespace(ring=SimpleNamespace(size=RING // 4, dtype=dtypes.uint32)) \
    if idx < self.copies else None
# The engine grows the scratch memory when it loads a program.
ops_amd.AMDDevice.scratch_buffer = lambda self, n: None


# An AMD GPU's queues address the host's memory, the only other memory of the
# cases: no copy is staged.

Buffer.get_buf = lambda self, device: None


# One device names an address

getaddr = UOp.getaddr
UOp.getaddr = lambda self, device: getaddr(self, device[0] if isinstance(device, tuple) else device)


# Signal words

def is_signal_word(signal): return hcq2.unwrap_view(signal)[0].tag == "timeline"


def compute_wait(self, signal, value):
    op = ops_amd.WAIT_REG_MEM_FUNCTION_EQ if is_signal_word(signal) else ops_amd.WAIT_REG_MEM_FUNCTION_GEQ
    self.wait_reg_mem(value.cast(dtypes.uint32), mem=signal.getaddr(self.devs), op=op)


def compute_signal(self, signal, value):
    data_sel = self.pm4.data_sel__mec_release_mem__send_64_bit_data if is_signal_word(signal) else \
        self.pm4.data_sel__mec_release_mem__send_32_bit_low
    with self.pred_exec(xcc_mask=0b1):
        self.release_mem(signal.getaddr(self.devs), value, data_sel,
                         self.pm4.int_sel__mec_release_mem__send_interrupt_after_write_confirm, cache_flush=True)


def sdma_wait(self, signal, value, eq=False):
    AMDSDMAWait(self, signal, value, eq=eq or is_signal_word(signal))


def sdma_signal(self, signal, value):
    op = self.sdma.SDMA_OP_FENCE | (self.sdma.SDMA_PKT_FENCE_HEADER_MTYPE(3) if self.target[0] != 9 else 0)
    high = []
    if is_signal_word(signal):
        carry = value.cast(dtypes.uint32).eq(UOp.const(0, dtypes.uint32))
        high = [carry.where(w, w.const_like(0)) for w in
                (UOp.const(op, dtypes.uint32), signal.getaddr(self.devs) + UOp.const(4, dtypes.uint64), (value >> 32).cast(dtypes.uint32))]
    self.q(op, signal.getaddr(self.devs), value.cast(dtypes.uint32), *high, self.sdma.SDMA_OP_TRAP, 0)


AMDSDMAWait = ops_amd.AMDSDMAQueue.wait
ops_amd.AMDComputeQueue.wait, ops_amd.AMDComputeQueue.signal = compute_wait, compute_signal
ops_amd.AMDSDMAQueue.wait, ops_amd.AMDSDMAQueue.signal = sdma_wait, sdma_signal


# Programs the engine loads

def amd_build_program(dev, prg, devs):
    data, image = ops_amd._amd_program_image(dev, lib := prg.src[3].arg)
    return data, UOp.placeholder((len(image),), dtypes.uint8, 0, device=devs, tag=("program", lib, prg.to_elf().name))


ops_amd.amd_build_program = amd_build_program


# The profile buffers, which only their placeholders' shapes need.

ops_amd.AMDDevice.prof_log = property(lambda self: SimpleNamespace(size=1 + 3 * self.prof_slots, dtype=dtypes.uint64))
ops_amd.AMDDevice.pmc_buf = property(lambda self: SimpleNamespace(size=self.pmc_size * self.prof_slots, dtype=dtypes.uint8))
ops_amd.AMDDevice.sqtt_buf = property(lambda self: SimpleNamespace(size=self.sqtt_win * self.prof_slots * self.sqtt_ses,
                                                                     dtype=dtypes.uint8))
ops_amd.AMDDevice.sqtt_wptrs = property(lambda self: SimpleNamespace(size=self.prof_slots * self.sqtt_ses, dtype=dtypes.uint32))


# Counted runs in the profile log

def clock_into(q, slot, i):
    address = q.prof_buf("prof_log").getaddr(q.devs) + (1 + 3 * slot + i) * 8
    with q.pred_exec(xcc_mask=0b1):
        q.release_mem(address, 0, q.pm4.data_sel__mec_release_mem__send_gpu_clock_counter,
                      q.pm4.int_sel__mec_release_mem__none)


def prof_start(self, data, info, lib):
    if not (self.dev.pmc_enabled or self.dev.sqtt_enabled): return None
    slot = (self.prof_buf("prof_log").index(0).load() + len(self.profiled)) % self.dev.prof_slots
    tag = lib.getaddr(self.devs) + data.desc_offset
    self.profiled.append(self.prof_buf("prof_log").index(1 + 3 * slot.cast(dtypes.int)).store(tag))
    clock_into(self, slot, 1)
    if self.dev.sqtt_enabled:
        self.sqtt_start(slot)
        self.sqtt_setup_exec(data, info)
    return slot


prof_stop = ops_amd.AMDComputeQueue.prof_stop


def timed_prof_stop(self, slot):
    if slot is not None: clock_into(self, slot, 2)
    prof_stop(self, slot)


ops_amd.AMDComputeQueue.prof_start, ops_amd.AMDComputeQueue.prof_stop = prof_start, timed_prof_stop

compute_init = ops_amd.AMDComputeQueue.__init__


def numbered_init(self, submit):
    compute_init(self, submit)
    self.dev.sqtt_next_cmd_id = ops_amd.itertools.count(0)


ops_amd.AMDComputeQueue.__init__ = numbered_init


# Memory barriers that leave the host data path to the host

ops_amd.AMDComputeQueue.memory_barrier = lambda self: self.acquire_mem()


# Dispatch grids as words

dispatch_packet = ops_amd.dispatch_packet
ops_amd.dispatch_packet = lambda *args, **kwargs: [UOp.const(w, dtypes.uint32) if isinstance(w, int) else w
                                                   for w in dispatch_packet(*args, **kwargs)]


# Cases

def empty(n=4, device="AMD"): return Tensor.empty(n, device=device)


def chain(x, n):
    for _ in range(n): x = (x + 1).contiguous()
    return x


def copies(): return ((empty(device="CPU").to("AMD") + 1).contiguous().to("CPU") + 2).contiguous()


CASES = {
    "chain": (lambda: chain(empty(), 3), {}),
    "chain_gfx1201": (lambda: chain(empty(), 2), {"gpu": "gfx1201"}),
    "chain_gfx942": (lambda: chain(empty(), 3), {"gpu": "gfx942"}),
    "chain_gfx942_cpx": (lambda: chain(empty(), 2), {"gpu": "gfx942_cpx"}),
    "profile": (lambda: chain(empty(), 2), {"profile": True}),
    "profile_gfx942": (lambda: chain(empty(), 2), {"gpu": "gfx942", "profile": True}),
    "copies": (copies, {}),
    "copies_gfx942": (copies, {"gpu": "gfx942"}),
    "copies_no_sdma": (copies, {"gpu": "gfx1100_no_sdma"}),
    "copies_profile": (copies, {"profile": True}),
    "large_copy": (lambda: (empty(5 << 18, device="CPU").to("AMD") + 1).contiguous(), {"gpu": "gfx1100_sdma5"}),
    "large_copy_sdma52": (lambda: (empty(5 << 18, device="CPU").to("AMD") + 1).contiguous(), {"gpu": "gfx1100_sdma52"}),
    "large_copy_gfx942": (lambda: (empty(5 << 18, device="CPU").to("AMD") + 1).contiguous(), {"gpu": "gfx942"}),
    "lds": (lambda: (empty() + 1).contiguous(), {"kernel": "lds"}),
    "variable": (lambda: (empty(10)[:Variable("v", 1, 10).bind(3)] + 1).contiguous(), {}),
    "variable_gfx942": (lambda: (empty(10)[:Variable("v", 1, 10).bind(3)] + 1).contiguous(), {"gpu": "gfx942"}),
    "scratch": (lambda: (empty() + 1).contiguous(), {"kernel": "scratch"}),
    "scratch_gfx942": (lambda: (empty() + 1).contiguous(), {"gpu": "gfx942", "kernel": "scratch"}),
    "counters": (lambda: chain(empty(), 2), {"counters": True}),
    "counters_gfx1201": (lambda: chain(empty(), 2), {"gpu": "gfx1201", "counters": True}),
    "counters_gfx942": (lambda: chain(empty(), 2), {"gpu": "gfx942", "counters": True}),
    "traces": (lambda: chain(empty(), 2), {"traces": True}),
    "traces_gfx1201": (lambda: chain(empty(), 2), {"gpu": "gfx1201", "traces": True}),
    "traces_gfx942": (lambda: chain(empty(), 2), {"gpu": "gfx942", "traces": True}),
    "counters_traces": (lambda: chain(empty(), 2), {"counters": True, "traces": True}),
}


def compiled(case):
    """(prepared, compiled) of the case."""
    program, options = CASES[case]
    GPU.value = options.get("gpu", "gfx1100")
    COUNTERS.value, TRACES.value = options.get("counters", False), options.get("traces", False)
    KERNEL.value = options.get("kernel", "simple_add")
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


# The words of a wait on a device's signal word for a value, and of a signal of
# one, and of a queue's signal in the slots, at values that carry into the high
# half: the signal word is at 0x10000, the slot at 0x20000.

SIGNAL_WORD, SLOT = 0x10000, 0x20000
CARRIES = (2**32 - 1, 2**32, 2**32 + 1, 2**33 - 1, 2**33)


def evaluate(u):
    if u.op is Ops.CONST: return u.arg
    if u.op is Ops.CAST: return truncate[u.dtype](evaluate(u.src[0]))
    return exec_alu(u.op, u.dtype, [evaluate(x) for x in u.src])


def command_words(device, queue, command, word, v):
    submit = hcq2.make_submit(devs=(device,), queue=queue)
    q = (ops_amd.amd_compute_queue if queue.startswith("COMPUTE") else ops_amd.AMDSDMAQueue)(submit)
    value = hcq2_d1.value(device) if word == "signal word" else UOp.const(3, dtypes.uint64)
    target = hcq2_d1.signal_word((device,)) if word == "signal word" else \
        UOp.placeholder((2,), dtypes.uint64, 0, device=(device,), volatile=True, tag="slots")
    getattr(q, command)(target, value)
    blob, addr = bytearray(q.blob), {target: SIGNAL_WORD if word == "signal word" else SLOT}
    for off, w in q.patches:
        w = w.substitute({g: UOp.const(addr[g.src[0]], dtypes.uint64) for g in w.toposort() if g.op is Ops.GETADDR})
        w = w.substitute({hcq2_d1.value(device): UOp.const(v, dtypes.uint64)})
        blob[off:off + w.dtype.itemsize] = int(evaluate(w)).to_bytes(w.dtype.itemsize, "little")
    return " ".join(f"{int.from_bytes(blob[i:i + 4], 'little'):08x}" for i in range(0, len(blob), 4))


@table
def signal_words():
    rows = []
    for device, gpu in (("AMD:1", "gfx1100"), ("AMD:2", "gfx942_cpx"), ("AMD:3", "gfx942")):
        GPU.value = gpu
        Device[device]
        for queue in ("COMPUTE:0", "COPY:0"):
            for command in ("wait", "signal"):
                for word, values in (("signal word", CARRIES), ("slot", (3,))):
                    for v in values:
                        rows.append((gpu, queue, command, word, str(v), command_words(device, queue, command, word, v)))
    return ("gpu", "queue", "command", "word", "value", "words"), rows
